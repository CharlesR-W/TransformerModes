"""Detached CPU launcher for the existing depth pipeline; `quote` is offline.

Use one VPS to submit and collect. Only launch, launch-rest and aggregate
submit work. Do not use `modal run` or deploy this script. The reservation
bounds modeled compute for admitted submissions, not Modal's invoice.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / 'SPEC_DEPTH.md'
SOURCE = ('depth.py', 'depth_dm.py', 'spacetime.py')
VOLUME = 'tm-exp1'
ROOT = Path('/vol')
OUT = ROOT / 'depth'
REMOTE_LEDGER = 'ledger/depth_run.json'
LOCAL_LEDGER = Path('/data/tm_depth_run_ledger.json')
MODELS = ('pythia', 'olmo')
SETS = ('doc00', 'doc01', 'doc02', 'doc04', 'pairA', 'pairB', 'd04', 'd08', 'd16', 'd32', 'rand')
UNITS = tuple(f'{model}/{name}' for model in MODELS for name in SETS)
PILOT = 'pythia/d32'
REST = tuple(unit for unit in UNITS if unit != PILOT)
AGGREGATES = ('cross', 'controls')
PANEL_KEYS = {'natural', 'random', 'shuffled'}
PATH_KEYS = {'c00', 'c01', 'c02', 'c04', 's0', 's1', 's2', 's3'}
NUMPY, SCIPY = '2.1.3', '1.15.3'  # modal_app.py's Stage A pins, not the VPS's versions.
CPU, GIB, STARTUP = 4, 16, 120
UNIT_TIMEOUT, CROSS_TIMEOUT, CONTROLS_TIMEOUT = 1800, 900, 3600
PILOT_CEILING = UNIT_TIMEOUT * 0.75  # Keep at least 25% of the deadline for variation/parallel I/O.
GATE_BYTES = 4_500_000_000
# 647 s partial d32 pilot -> 3.71x. Controls: two 455 s refits -> 4.95x their sum, including references.
# Base rates copied from modal_app.py. Preemptible containers (multiplier 1): units are short and resumable.
CPU_RATE, MEM_RATE, NONPREEMPTIBLE = Decimal('0.0000131'), Decimal('0.00000222'), 1
RATE = NONPREEMPTIBLE * (CPU * CPU_RATE + GIB * MEM_RATE)
PROBE = 'probe/probe.json'
PROBE_SHA256 = 'e6beef6cf07ad0a0492072dd2fc46d785696341fe666bd58b6de9953a2d74d4d'  # depth_inputs.py
INPUTS = 'acts_depth/inputs.json'
SYNTHETIC = tuple(f'{p}{i}.npy' for p, n in (('r', 5), ('s', 4)) for i in range(n))
PROVENANCE = tuple(f'{kind}/{model}/provenance.json' for kind in ('acts', 'acts_depth') for model in MODELS)
STAGE_A_MARKERS = tuple(f'state/completed/U1-{model}.json' for model in MODELS)


def timeout(task):
    if task in UNITS:
        return UNIT_TIMEOUT
    return {'cross': CROSS_TIMEOUT, 'controls': CONTROLS_TIMEOUT}[task]


def reservation(pilot=False):
    # One pilot retry, plus one shared retry anywhere after the pilot (hold the most expensive slot).
    # The full quote includes the pilot and its retry. Manifest is inside the pilot's timed call.
    slots = {PILOT: 2} if pilot else {**dict.fromkeys(UNITS, 1), PILOT: 2, 'cross': 1, 'controls': 2}
    seconds = sum(n * (timeout(task) + STARTUP) for task, n in slots.items())
    return dict(scope='pilot' if pilot else 'full', cpu=CPU, gib=GIB, startup_seconds=STARTUP,
                nonpreemptible_multiplier=NONPREEMPTIBLE, cpu_rate=str(CPU_RATE), memory_rate=str(MEM_RATE),
                rate_per_second=str(RATE), reserved_seconds=seconds, reserved_usd=str(seconds * RATE),
                timeout_seconds=dict(unit=UNIT_TIMEOUT, cross=CROSS_TIMEOUT, controls=CONTROLS_TIMEOUT))


def approve(pilot, maximum):
    quote = reservation(pilot)
    print(json.dumps(quote, indent=2), flush=True)
    try:
        cap = Decimal(maximum)
    except (InvalidOperation, TypeError):
        raise ValueError('--max-usd must be a finite positive decimal') from None
    if not cap.is_finite() or cap <= 0 or cap < Decimal(quote['reserved_usd']):
        raise ValueError(f"refusing ${quote['reserved_usd']} reservation at --max-usd {maximum!r}")
    return dict(quote, max_usd=str(cap))


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    tmp.write_bytes(json_bytes(data))
    tmp.replace(path)


def code_identity():
    return dict(code_sha256=sha(json_bytes({name: file_hash(HERE / name) for name in SOURCE})),
                spec_sha256=file_hash(SPEC), launcher_sha256=file_hash(Path(__file__)))


def input_identity(read, names=None):
    raw = {path: read(path) for path in (PROBE, INPUTS, *PROVENANCE, *STAGE_A_MARKERS)}
    if any(v is None for v in raw.values()):
        raise ValueError('missing probe, synthetic ids or activation provenance on Volume')
    digests = {path: sha(v) for path, v in raw.items()}
    if digests[PROBE] != PROBE_SHA256:
        raise ValueError('Volume probe differs from the frozen Stage A probe')
    for model, marker in zip(MODELS, STAGE_A_MARKERS):
        record = json.loads(raw[marker])
        if (record['version'] != 2 or record['probe_sha256'] != digests[PROBE]
                or record['provenance_path'] != f'/acts/{model}/provenance.json'):
            raise ValueError(f'{marker}: incorrect Stage A extraction receipt')
        path = f'acts_depth/{model}/provenance.json'
        synthetic = json.loads(raw[path])
        if synthetic['probe_sha256'] != digests[PROBE] or synthetic['inputs_sha256'] != digests[INPUTS]:
            raise ValueError(f'{path}: frozen synthetic inputs or probe mismatch')
    if names:
        for model in MODELS:
            for directory, required in ((f'acts/{model}', {f'c{i:02}.npy' for i in range(32)}),
                                        (f'acts_depth/{model}', set(SYNTHETIC))):
                if missing := required - names(directory):
                    raise ValueError(f'{directory}: missing {sorted(missing)}')
    return digests


def read_remote(volume, path):
    try:
        return b''.join(volume.read_file(path))
    except FileNotFoundError:
        return None


def save_ledger(volume, ledger, local=LOCAL_LEDGER, *, create=False):
    # The client alone edits the central ledger. Workers commit disjoint receipt files.
    with volume.batch_upload(force=not create) as batch:
        batch.put_file(io.BytesIO(json_bytes(ledger)), REMOTE_LEDGER)
    atomic_json(local, ledger)


def load_ledger(volume, local=LOCAL_LEDGER):
    raw = read_remote(volume, REMOTE_LEDGER)
    if raw is None:
        if local.exists():
            raise ValueError('Volume ledger missing; refusing to reset the attempt budget')
        return None
    ledger = json.loads(raw)
    if ledger['version'] != 1:
        raise ValueError('unknown Volume ledger version')
    if local.exists():
        previous = read_json(local)
        old, new = previous['attempts'], ledger['attempts']
        if (previous['run'] != ledger['run'] or len(old) > len(new)
                or any(any(b.get(k) != v for k, v in a.items())
                       for a, b in zip(old, new) if a['status'] != 'pending')
                or any(any(b.get(k) != v for k, v in a.items() if k != 'status')
                       for a, b in zip(old, new) if a['status'] == 'pending')):
            raise ValueError('local mirror and Volume ledger conflict; refusing to reset the attempt budget')
    atomic_json(local, ledger)
    return ledger


@contextmanager
def ledger_lock():
    # Same-VPS only. Lock the directory, not the atomically replaced JSON inode.
    fd = os.open(LOCAL_LEDGER.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def latest(ledger):
    return {row['task']: row for row in ledger['attempts']}


def require_pilot(ledger):
    row = latest(ledger).get(PILOT, {})
    r = row.get('receipt', {})
    if row.get('status') != 'done' or not r.get('fresh'):
        raise ValueError('collect a successful fresh Pythia d32 pilot receipt first')
    gate_seconds = r['seconds']
    if gate_seconds is None:
        # A recovered stamp measures computation, not its output commit. If observed
        # early enough, submission-to-collection bounds the time through publication.
        gate_seconds = row['collected_at'] - row['submitted_at']
        if gate_seconds < r['precommit_seconds']:
            raise ValueError('VPS timestamps cannot bound the recovered pilot runtime')
    if r['peak_rss_bytes'] >= GATE_BYTES or gate_seconds >= PILOT_CEILING:
        raise ValueError('pilot did not establish the memory and 30-minute runtime gate; reassess before launching the rest')
    return r


def choose_tasks(ledger, phase, retry=None):
    seen = latest(ledger)
    if phase != 'pilot':
        require_pilot(ledger)
    if phase == 'aggregate' and any(seen.get(u, {}).get('status') != 'done' for u in UNITS):
        raise ValueError('collect all 22 successful construction units before aggregating')
    choices = {'pilot': (PILOT,), 'rest': REST, 'aggregate': AGGREGATES}[phase]
    if retry is not None:
        if retry not in choices or seen.get(retry, {}).get('status') not in ('failed', 'timed_out'):
            raise ValueError('only a confirmed failure in this phase can be retried')
        pilot = phase == 'pilot'
        if any(a['retry'] and (a['task'] == PILOT) == pilot for a in ledger['attempts']):
            raise ValueError('manual retry allowance exhausted for this scope')
        return (retry,)
    return tuple(t for t in choices if t not in seen)


def new_attempt(ledger, task, approval, retry):
    row = dict(id=uuid.uuid4().hex, task=task, status='pending', retry=retry,
               reserved_seconds=timeout(task) + STARTUP, approval=approval, submitted_at=time.time())
    ledger['attempts'].append(row)
    return row


def receipt_path(run, attempt):
    return f'ledger/depth_run_receipts/{run}/{attempt}.json'


def recover_unit(read, identity):
    """Recover the same attempt if its unit committed before its final receipt."""
    task = identity['task']
    if task not in UNITS:
        return None
    dest = f'depth/units/{task}'
    telemetry, done = (read(f'{dest}/{name}') for name in ('run.json', 'done.json'))
    if telemetry is None or done is None:
        return None
    saved = json.loads(telemetry).get('modal_attempt', {})
    if any(saved.get(k) != v for k, v in identity.items()) or saved.get('marker_sha256') != sha(done):
        return None
    return dict(saved, recovered=True, status='done', seconds=None, telemetry_sha256=sha(telemetry))


def settle(ledger, row, receipt):
    if (row['status'] != 'pending' or receipt.get('run') != ledger['run']
            or receipt.get('id') != row['id'] or receipt.get('task') != row['task']):
        raise ValueError('attempt and receipt do not match')
    seconds, status = receipt['seconds'], receipt['status']
    if status not in ('done', 'failed', 'timed_out') or (
            seconds is None and status != 'timed_out' and not receipt.get('recovered')):
        raise ValueError('invalid receipt status/runtime')
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0):
        raise ValueError('invalid measured seconds')
    estimate = Decimal(row['reserved_seconds']) if seconds is None else Decimal(str(seconds)) + STARTUP
    row.update(status=status, seconds=seconds, estimated_billed_seconds=float(estimate),
               estimated_usd=str(estimate * Decimal(row['approval']['rate_per_second'])), receipt=receipt,
               collected_at=time.time())


def elapsed(label, fn, *args):
    start = time.monotonic()
    print(f'{label}: start', flush=True)
    try:
        return fn(*args)
    finally:
        print(f'{label}: {time.monotonic() - start:.2f}s', flush=True)


def pipeline(out, command, *args):
    start = time.monotonic()
    argv = [sys.executable, '-uB', str(HERE / 'depth.py'), '--out', str(out), command, *args]
    with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            print(f'{command} +{time.monotonic() - start:.2f}s: {line.rstrip()}', flush=True)
        if proc.wait():
            raise subprocess.CalledProcessError(proc.returncode, argv)


def unit_info(path, model, name, identity, refit=False):
    marker = path / 'done.json'
    done, meta, run = (read_json(path / f) for f in ('done.json', 'unit.json', 'run.json'))
    for key in ('code_sha256', 'spec_sha256'):
        if done[key] != identity[key]:
            raise ValueError(f'{path}: incompatible {key}')
    inventory = {str(p.relative_to(path)): file_hash(p) for p in path.rglob('*')
                 if p.is_file() and p.name not in ('done.json', 'run.json')}
    if inventory != done['files'] or sha(json_bytes(inventory)) != done['outputs_sha256']:
        raise ValueError(f'{path}: output inventory mismatch')
    actual = set(meta['datasets'])
    expected = set() if refit else {f'panels/{k}' for k in PANEL_KEYS}
    if not refit and name in ('d32', 'rand'):
        expected |= {f'paths/{k}' for k in PATH_KEYS}
    strengths = {'tstar'} if refit else {'tstar', 'tstar_div16'}
    checks = read_json(path / 'checks.json')
    if (meta['model'] != model or meta['set'] != name or meta['refit'] != refit or actual != expected
            or any(meta['missing'].values()) or any(run['missing'].values())
            or not all(f'{dataset}.npz' in inventory for dataset in expected)
            or not all(f'{s}/modes.npy' in inventory and f'{s}/facts.json' in inventory for s in strengths)
            or set(checks) != strengths or not all(set(c) == {'algebra', 'replay_256', 'replay_7'}
                 and all(x['passed'] is True for x in c.values()) for c in checks.values())):
        raise ValueError(f'{path}: missing panels, paths, modes or passing replay/algebra checks')
    if (run['numpy'], run['scipy']) != (NUMPY, SCIPY):
        raise ValueError(f'{path}: numerical environment differs from the CPU image pins')
    return dict(peak_rss_bytes=run['peak_rss_bytes'], output_bytes=run['output_bytes'],
                pipeline_seconds=run['wall_seconds'], marker_sha256=file_hash(marker),
                telemetry_sha256=file_hash(path / 'run.json'))


def publish(source, dest, volume):
    # Compute on container scratch. Stage each delivery in its own Volume path before publishing.
    staging = OUT / '.publishing' / uuid.uuid4().hex
    staging.parent.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, staging)
    else:
        shutil.copyfile(source, staging)
    try:
        if dest.exists():
            raise ValueError(f'refusing to overwrite committed outputs: {dest}')
        staging.rename(dest)
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
        else:
            staging.unlink(missing_ok=True)
    elapsed(f'commit {dest.name}', volume.commit)


def manifest(out, volume):
    if not (OUT / 'manifest.json').exists():
        elapsed('manifest', pipeline, out, 'manifest')
        created = read_json(out / 'manifest.json')
        if set(created['sets']) != set(SETS) or not created['synthetic_inputs']:
            raise ValueError('manifest is incomplete')
        publish(out / 'manifest.json', OUT / 'manifest.json', volume)


def validate_summary(path, expected):
    data = read_json(path)
    records = data['records']
    matches = (len(records) == expected if isinstance(expected, int)
               else isinstance(records, dict) and set(records) == set(expected))
    if data['skipped'] or not matches:
        raise ValueError(f'incomplete aggregate: {path}')


def aggregate_worker(task, out, job, volume):
    if set(job['units']) != set(UNITS):
        raise ValueError('missing construction unit receipts')
    seen = {}
    for u in UNITS:
        path = OUT / 'units' / u
        r = unit_info(path, *u.split('/'), job['identity'])
        if any(r[k] != job['units'][u][k] for k in ('marker_sha256', 'telemetry_sha256')):
            raise ValueError(f'{u}: committed unit differs from its receipt')
        for ctx, digest in read_json(path / 'input_files.json')['activations'].items():
            key = (u.split('/')[0], ctx)
            if seen.setdefault(key, digest) != digest:
                raise ValueError(f'{u}: activation changed between units: {key}')
    (out / 'units').symlink_to(OUT / 'units', target_is_directory=True)
    m = read_json(out / 'manifest.json')
    # Two strengths, construction plus each nonempty panel cohort, plus d32/rand's extra comparison.
    paired_records = 2 * (len(SETS) + sum(len(set(labels)) for s in SETS for labels in m['cohorts'][s].values()) + 2)

    def report(stage, name, expected):
        dest = OUT / name
        if not dest.exists():
            elapsed(stage, pipeline, out, stage)
            validate_summary(out / name, expected)
            publish(out / name, dest, volume)
        validate_summary(dest, expected)
        return file_hash(dest)

    if task == 'cross':
        # Cross also has six within-model ladder comparisons per model and strength.
        return {'cross.json': report('_cross', 'cross.json', paired_records + 6 * 2 * len(MODELS))}
    for model in MODELS:
        rel = Path('refits') / model / 'd08'
        if not (OUT / rel).exists():
            elapsed(f'refit {model}', pipeline, out, '_refit', '--model', model, '--set', 'd08')
            unit_info(out / rel, model, 'd08', job['identity'], refit=True)
            publish(out / rel, OUT / rel, volume)
        unit_info(OUT / rel, model, 'd08', job['identity'], refit=True)
        for ctx, digest in read_json(OUT / rel / 'input_files.json')['activations'].items():
            if seen.get((model, ctx)) != digest:
                raise ValueError(f'{model} refit: activation changed since construction: {ctx}')
    if (out / 'refits').exists():
        shutil.rmtree(out / 'refits')
    (out / 'refits').symlink_to(OUT / 'refits', target_is_directory=True)
    return {name: report(stage, name, expected) for stage, name, expected in (
        ('_refit_scores', 'refit_agreement.json', MODELS), ('_permutations', 'permutations.json', paired_records),
        ('_gaussian', 'gaussian.json', len(SETS)))}


def run_worker(job, volume):
    started = time.monotonic()
    task = job['task']
    receipt = {k: job[k] for k in ('run', 'id', 'task')}
    path = ROOT / receipt_path(job['run'], job['id'])
    try:
        elapsed('volume reload', volume.reload)
        if path.exists():
            return read_json(path)  # Provider redelivery after a receipt committed.
        if code_identity() != job['identity'] or input_identity(lambda p: (ROOT / p).read_bytes()) != job['inputs']:
            raise ValueError('worker code or Volume inputs differ from the admitted run')
        if recovered := recover_unit(lambda p: (ROOT / p).read_bytes() if (ROOT / p).is_file() else None, receipt):
            unit_info(OUT / 'units' / task, *task.split('/'), job['identity'])
            atomic_json(path, recovered)
            elapsed('recovered receipt commit', volume.commit)
            return recovered
        with tempfile.TemporaryDirectory(prefix='tm-depth-') as directory:
            out = Path(directory)
            if task == PILOT:
                manifest(out, volume)
            shutil.copyfile(OUT / 'manifest.json', out / 'manifest.json')
            manifest_sha = file_hash(out / 'manifest.json')
            if task != PILOT and manifest_sha != job['manifest_sha256']:
                raise ValueError('manifest differs from the collected pilot')
            if task in UNITS:
                model, name = task.split('/')
                dest = OUT / 'units' / task
                # Never confuse a cached pilot with a timing measurement.
                elapsed(task, pipeline, out, '_unit', '--model', model, '--set', name)
                unit_path = out / 'units' / task
                receipt.update(unit_info(unit_path, model, name, job['identity']), fresh=True,
                               manifest_sha256=manifest_sha, precommit_seconds=time.monotonic() - started,
                               blas_threads=CPU)
                # The unit and its recovery stamp commit together. A final receipt also times the output commit.
                telemetry = unit_path / 'run.json'
                atomic_json(telemetry, dict(read_json(telemetry), modal_attempt=receipt))
                receipt['telemetry_sha256'] = file_hash(telemetry)
                publish(unit_path, dest, volume)
            else:
                receipt['outputs'] = elapsed('verify units and aggregate', aggregate_worker, task, out, job, volume)
            receipt.update(status='done', manifest_sha256=manifest_sha)
    except Exception as exc:
        traceback.print_exc()
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    receipt['seconds'] = time.monotonic() - started
    receipt['blas_threads'] = CPU
    atomic_json(path, receipt)
    elapsed('receipt commit', volume.commit)
    print(json.dumps(receipt), flush=True)
    return receipt


def _remote(job, volume):
    return run_worker(job, volume)


def make_app(modal, volume):
    app = modal.App('tm-exp1-depth-cpu', include_source=False)
    code = '/opt/tm_depth/exp1'
    env = dict(PYTHONPATH=code, TM_DEPTH_ACTS=str(ROOT / 'acts'), TM_DEPTH_SYNTH=str(ROOT / 'acts_depth'),
               TM_DEPTH_PROBE=str(ROOT / PROBE), TM_DEPTH_OUT=str(OUT), TM_DEPTH_BLAS_THREADS=str(CPU))
    env.update({f'{lib}_NUM_THREADS': str(CPU) for lib in ('OPENBLAS', 'OMP', 'MKL', 'BLIS')})
    image = (modal.Image.debian_slim(python_version='3.12')
             .pip_install(f'numpy=={NUMPY}', f'scipy=={SCIPY}').env(env))
    for name in (*SOURCE, 'modal_depth_run.py'):
        image = image.add_local_file(str(HERE / name), f'{code}/{name}')
    image = image.add_local_file(str(SPEC), '/opt/tm_depth/SPEC_DEPTH.md')
    # With `python modal_depth_run.py`, __main__ is not importable in the worker.
    # Register by reference to the module explicitly bundled above (no cross-version bytecode pickle).
    from modal_depth_run import _remote as worker_entry

    workers = {kind: app.function(name=kind, image=image, cpu=(CPU, CPU),
                                  memory=(GIB * 1024, GIB * 1024), volumes={str(ROOT): volume},
                                  timeout=limit, startup_timeout=STARTUP, retries=0, nonpreemptible=False,
                                  single_use_containers=True, max_containers=n)(worker_entry)
               for kind, limit, n in (('unit', UNIT_TIMEOUT, len(UNITS)), ('cross', CROSS_TIMEOUT, 1),
                                      ('controls', CONTROLS_TIMEOUT, 1))}
    return app, workers


def collect(modal, volume, ledger):
    from modal.exception import FunctionTimeoutError, OutputExpiredError

    for row in ledger['attempts']:
        if row['status'] != 'pending':
            continue
        identity = {**{k: row[k] for k in ('id', 'task')}, 'run': ledger['run']}
        read = lambda path: read_remote(volume, path)
        raw = read(receipt_path(ledger['run'], row['id']))
        result = json.loads(raw) if raw is not None else None
        reason = 'no call ID or receipt'
        if result is None and row.get('call_id'):
            try:
                result = modal.FunctionCall.from_id(row['call_id']).get(timeout=0)
            except FunctionTimeoutError as error:
                result = dict(identity, status='timed_out', seconds=None, error=str(error))
            except OutputExpiredError:
                reason = 'provider result expired without a durable receipt'
            except TimeoutError:
                print(f"{row['task']}: queued or running", flush=True)
                continue
            except Exception as error:
                # A connection error or unclassified provider failure is not failure proof.
                reason = f'{type(error).__name__}: {error}'
        if result is None or result['status'] != 'done':
            result = recover_unit(read, identity) or result
        if result is None:
            print(f"{row['task']}: unresolved ({reason}); retains full hold ({row['id']})", flush=True)
            continue
        settle(ledger, row, result)
        save_ledger(volume, ledger)
    for row in latest(ledger).values():
        print(json.dumps(row), flush=True)
    statuses = {row['status'] for row in latest(ledger).values()}
    return 1 if statuses & {'failed', 'timed_out'} else 2 if 'pending' in statuses else 0


def launch(modal, volume, ledger, phase, retry, approval):
    create = ledger is None
    if create and (phase != 'pilot' or retry is not None):
        raise ValueError('launch and collect the pilot first')
    if not create:
        tasks = choose_tasks(ledger, phase, retry)
        if not tasks:
            raise ValueError('no unsubmitted tasks in this phase; collect or explicitly retry a confirmed failure')
    identity = code_identity()
    if not create and ledger['identity'] != identity:
        raise ValueError('code or spec changed since the pilot; refusing to mix runs')
    inputs = input_identity(lambda p: read_remote(volume, p),
                            lambda d: {Path(e.path).name for e in volume.listdir(d)})
    if create:
        try:
            existing = volume.listdir('depth')
        except (FileNotFoundError, modal.exception.NotFoundError):
            existing = []
        if existing:
            raise ValueError('Volume depth/ is not empty; refusing to mix with previous work')
        ledger = dict(version=1, run=uuid.uuid4().hex, identity=identity, inputs=inputs, attempts=[], approvals=[])
        tasks = choose_tasks(ledger, phase)
    elif ledger['inputs'] != inputs:
        raise ValueError('Volume input metadata changed since the pilot')
    job = {k: ledger[k] for k in ('run', 'identity', 'inputs')}
    if phase != 'pilot':
        job['manifest_sha256'] = require_pilot(ledger)['manifest_sha256']
        raw = read_remote(volume, 'depth/manifest.json')
        if raw is None or sha(raw) != job['manifest_sha256']:
            raise ValueError('missing manifest or manifest differs from the pilot')
    if phase == 'aggregate':
        job['units'] = {t: latest(ledger)[t]['receipt'] for t in UNITS}
    ledger['approvals'].append(dict(phase=phase, approved_at=time.time(), tasks=tasks, **approval))
    save_ledger(volume, ledger, create=create)  # Persist the scope before even building the app image.
    app, workers = make_app(modal, volume)
    with modal.enable_output(), app.run(detach=True):
        for task in tasks:
            row = new_attempt(ledger, task, approval, retry is not None)
            save_ledger(volume, ledger)  # Write ahead: uncertainty must never authorize a duplicate launch.
            call = workers['unit' if task in UNITS else task].spawn(dict(job, task=task, id=row['id']), volume)
            row['call_id'] = call.object_id
            print(f"{task}: Modal call {call.object_id}, attempt {row['id']}", flush=True)
            save_ledger(volume, ledger)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('quote').add_argument('--pilot', action='store_true')
    sub.add_parser('collect', help='reconcile completed attempts without waiting or launching')
    pilot = sub.add_parser('launch', help='submit the single Pythia d32 pilot')
    pilot.add_argument('--pilot', required=True, action='store_true')
    pilot.add_argument('--model', required=True, choices=('pythia',))
    pilot.add_argument('--set', dest='name', required=True, choices=('d32',))
    rest = sub.add_parser('launch-rest', help='submit the other 21 units or finish interrupted fan-out')
    agg = sub.add_parser('aggregate', help='submit cross and controls after all units are collected')
    for cmd in (pilot, rest, agg):
        cmd.add_argument('--max-usd', required=True, help='explicit approved cumulative USD ceiling for this scope')
    pilot.add_argument('--retry', action='store_true')
    rest.add_argument('--retry', choices=REST, metavar='MODEL/SET')
    agg.add_argument('--retry', choices=AGGREGATES)
    args = p.parse_args(argv)
    if args.action == 'quote':
        print(json.dumps(reservation(args.pilot), indent=2))
        return 0
    try:
        phase = {'launch': 'pilot', 'launch-rest': 'rest', 'aggregate': 'aggregate'}.get(args.action)
        approval = approve(phase == 'pilot', args.max_usd) if phase else None
        import modal

        with ledger_lock():
            volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
            ledger = load_ledger(volume)
            if phase:
                retry = PILOT if phase == 'pilot' and args.retry else args.retry or None
                launch(modal, volume, ledger, phase, retry, approval)
                return 0
            if ledger is None:
                raise ValueError('no Volume ledger to collect')
            return collect(modal, volume, ledger)
    except (ValueError, OSError) as error:
        p.exit(1, f'{error}\n')


if __name__ == '__main__':
    sys.exit(main())
