"""VPS launcher for depth extraction. Only `launch` submits a GPU call."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import io
import json
import math
import os
import time

import modal
from modal.exception import FunctionTimeoutError

from depth_inputs import INPUTS_PATH, LOCAL_INPUTS, LOCAL_PROBE, PROBE_PATH, check_probe_sha, probe_documents, validate_frozen
from extract_depth import run_unit
from modal_app import APP_NAME as STAGE_A_VOLUME
from modal_app import CPU_RATE, GPU_RATES, HERE, MEM_RATE, ROOT, STARTUP_SECONDS, extract_image
from probe import MODELS, write_json_atomic

GPU, CPU, GIB, TIMEOUT = 'L4', 4, 16, 600
MAX_ATTEMPTS = 2
CAP_USD = 1.0
LOCAL_LEDGER = LOCAL_INPUTS.with_name('ledger_depth.json')
REMOTE_LEDGER = 'ledger/depth.json'

app = modal.App('tm-exp1-depth', include_source=False)
volume = modal.Volume.from_name(STAGE_A_VOLUME, create_if_missing=False)
image = extract_image
for name in ('depth_inputs.py', 'extract_depth.py'):
    image = image.add_local_file(str(HERE / name), f'/opt/tm_exp1/{name}')
worker = app.function(image=image, gpu=GPU, cpu=(CPU, CPU), memory=(GIB * 1024, GIB * 1024), timeout=TIMEOUT,
                      startup_timeout=STARTUP_SECONDS, volumes={str(ROOT): volume}, retries=0,
                      max_containers=1, single_use_containers=True)(run_unit)


def reservation(*, rate_override=None):
    """Hold both models' initial attempts and both manual retries, even if launching just one."""
    rate = (Decimal(str(GPU_RATES[GPU])) + CPU * Decimal(str(CPU_RATE)) + GIB * Decimal(str(MEM_RATE))
            if rate_override is None else Decimal(str(rate_override)))
    if not rate.is_finite() or rate <= 0:
        raise ValueError('expected a finite positive rate')
    seconds_per_attempt = TIMEOUT + STARTUP_SECONDS
    attempts = len(MODELS) * MAX_ATTEMPTS
    reserved = attempts * seconds_per_attempt * rate
    if reserved > Decimal(str(CAP_USD)):
        raise ValueError(f'refusing ${reserved:.8f} reservation above ${CAP_USD:.2f}')
    return dict(cap_usd=CAP_USD, gpu=GPU, cpu=CPU, memory_gib=GIB, timeout_seconds=TIMEOUT,
                startup_seconds=STARTUP_SECONDS, max_attempts_per_model=MAX_ATTEMPTS, attempts_reserved=attempts,
                rate_usd_per_second=float(rate), seconds_per_attempt=seconds_per_attempt,
                reserved_usd=float(reserved))


@contextmanager
def ledger_lock():
    # All launches/collections must use this one VPS and directory. No lock file is needed.
    fd = os.open(LOCAL_LEDGER.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def read_volume(path):
    try:
        return b''.join(volume.read_file(path))
    except FileNotFoundError:
        return None


def put_volume(path, data, *, replace=False):
    with volume.batch_upload(force=replace) as batch:
        batch.put_file(io.BytesIO(data), path)


def save_ledger(ledger, *, replace=True):
    # Volume is authoritative; a failed local mirror must never authorize another launch.
    put_volume(REMOTE_LEDGER, json.dumps(ledger, allow_nan=False).encode('utf-8'), replace=replace)
    write_json_atomic(LOCAL_LEDGER, ledger)


def load_ledger(quote, inputs_sha256, *, create):
    raw = read_volume(REMOTE_LEDGER)
    if raw is None:
        if not create or LOCAL_LEDGER.exists():
            raise ValueError('Volume ledger is missing; refusing to reset the attempt budget')
        return dict(version=1, reservation=quote, inputs_sha256=inputs_sha256, attempts=[]), True
    ledger = json.loads(raw)
    if ledger['version'] != 1 or ledger['reservation'] != quote or ledger['inputs_sha256'] != inputs_sha256:
        raise ValueError('Volume ledger does not match this reservation and frozen input file')
    if LOCAL_LEDGER.exists():
        local = json.loads(LOCAL_LEDGER.read_bytes())
        if (local['reservation'] != quote or local['inputs_sha256'] != inputs_sha256
                or len(local['attempts']) > len(ledger['attempts'])
                or any(a['model'] != b['model'] or a['started_at'] != b['started_at']
                       or (a['status'] != 'pending' and a != b)
                       or ('call_id' in a and a['call_id'] != b.get('call_id'))
                       for a, b in zip(local['attempts'], ledger['attempts']))):
            raise ValueError('local and Volume ledgers conflict; refusing to reset the attempt budget')
    write_json_atomic(LOCAL_LEDGER, ledger)
    return ledger, False


def next_attempt(ledger, model, retry):
    attempts = ledger['attempts']
    if any(row['status'] == 'pending' for row in attempts):
        raise ValueError('an attempt is unresolved; collect its receipt before launching any new call')
    previous = [row for row in attempts if row['model'] == model]
    if len(attempts) >= ledger['reservation']['attempts_reserved'] or len(previous) >= MAX_ATTEMPTS:
        raise ValueError('attempt budget exhausted')
    if previous:
        if not retry or previous[-1]['status'] not in ('failed', 'timed_out'):
            raise ValueError('only a confirmed failure can be retried, and only with --retry')
    elif retry:
        raise ValueError('--retry requires a previous failed attempt')
    return len(previous) + 1


def settle(ledger, row, call):
    quote = ledger['reservation']
    try:
        result = call.get()  # A lost connection/result is not proof of failure; leave it pending.
    except FunctionTimeoutError as error:
        # Modal did not return a worker timer. Keep the full hold, not client wall time.
        result = dict(status='timed_out', seconds=None, error=str(error))
    seconds = result['seconds']
    if result['status'] not in ('done', 'failed', 'timed_out') or (
            seconds is None and result['status'] != 'timed_out') or (
            seconds is not None and (not math.isfinite(seconds) or seconds < 0)):
        raise ValueError('invalid worker receipt; attempt remains pending')
    billed_seconds = quote['seconds_per_attempt'] if seconds is None else seconds + quote['startup_seconds']
    row.update(status=result['status'], seconds=seconds, billed_seconds_estimate=billed_seconds,
               estimated_usd=billed_seconds * quote['rate_usd_per_second'], result=result, finished_at=time.time())
    save_ledger(ledger)
    print(json.dumps(row, indent=2), flush=True)
    if row['status'] != 'done':
        raise SystemExit(1)


def launch(model, retry, quote, raw_inputs, input_sha):
    check_probe_sha(read_volume(PROBE_PATH))  # Verify the existing Volume before writing or starting an app.
    ledger, new = load_ledger(quote, input_sha, create=not retry)
    attempt = next_attempt(ledger, model, retry)
    remote_inputs = read_volume(INPUTS_PATH)
    if remote_inputs is None:
        if not new:
            raise ValueError('frozen inputs are missing on the Volume; refusing to change the existing run')
        put_volume(INPUTS_PATH, raw_inputs)
    elif remote_inputs != raw_inputs:
        raise ValueError('frozen inputs already exist on the Volume with different bytes')
    if new:
        save_ledger(ledger, replace=False)
    print(f"Reserved ${quote['reserved_usd']:.7f} across both models and both possible retries", flush=True)
    with modal.enable_output(), app.run(detach=True):
        row = dict(model=model, attempt=attempt, status='pending', started_at=time.time(),
                   reserved_usd=quote['seconds_per_attempt'] * quote['rate_usd_per_second'])
        ledger['attempts'].append(row)
        save_ledger(ledger)  # Consume the attempt before making any request that could launch it.
        call = worker.spawn(model, str(ROOT), input_sha, volume)
        row['call_id'] = call.object_id
        print(f'{model} attempt {attempt}: {call.object_id}', flush=True)
        save_ledger(ledger)
        settle(ledger, row, call)


def collect(model, quote, input_sha):
    ledger, _ = load_ledger(quote, input_sha, create=False)
    pending = [row for row in ledger['attempts'] if row['model'] == model and row['status'] == 'pending']
    if len(pending) != 1 or not pending[0].get('call_id'):
        raise ValueError('no pending attempt with a known Modal call ID; inspect the ledger and Modal Dashboard')
    row = pending[0]
    settle(ledger, row, modal.FunctionCall.from_id(row['call_id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('quote', help='print the reservation without contacting Modal')
    for action in ('launch', 'collect'):
        cmd = sub.add_parser(action)
        cmd.add_argument('model', choices=MODELS)
        if action == 'launch':
            cmd.add_argument('--retry', action='store_true', help='explicitly retry a confirmed failed first attempt')
    args = parser.parse_args()
    quote = reservation()  # Fail before any Volume or App action.
    if args.action == 'quote':
        print(json.dumps(quote, indent=2))
        return
    with ledger_lock():
        raw = LOCAL_INPUTS.read_bytes()
        probe_bytes = LOCAL_PROBE.read_bytes()
        check_probe_sha(probe_bytes)
        documents, allowed = probe_documents(json.loads(probe_bytes))
        validate_frozen(json.loads(raw), documents, allowed)
        input_sha = hashlib.sha256(raw).hexdigest()
        if args.action == 'launch':
            launch(args.model, args.retry, quote, raw, input_sha)
        else:
            collect(args.model, quote, input_sha)


if __name__ == '__main__':
    main()
