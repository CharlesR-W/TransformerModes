"""Detached Modal entry points for experiment 1. Deploy and launch cost money.

After explicit approval (not needed for --help):
    uv run --with modal==1.6.2 python modal_app.py deploy
    uv run --with modal==1.6.2 python modal_app.py launch probe
    uv run --with modal==1.6.2 python modal_app.py launch extract
    uv run --with modal==1.6.2 python modal_app.py status [run_id]
    uv run --with modal==1.6.2 python modal_app.py fetch [out]

U1 requires a completed U0. /state holds provenance, completion markers,
and one ledger record per attempt. Only /analysis, /audit, /site are fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import modal

from probe import CONTEXT_TOKENS, CORPUS, MODELS, POOL_SIZE, write_json_atomic

HERE = Path(__file__).resolve().parent
APP_NAME = "tm-exp1"
ROOT = Path("/vol")
CPU_RATE = 0.0000131
MEM_RATE = 0.00000222
L4_RATE = 0.000164
STARTUP_SECONDS = 120
STAGE_A_CAP = 8.0
EXTRACTION_VERSION = 1  # Bump when the U1 data or diagnostics change.
# (GPU, reserved cores, GiB, timeout seconds, expected wall seconds incl. startup)
HW = {"U0": (None, 2, 8, 40 * 60, 20 * 60),
      "U1": ("L4", 4, 32, 50 * 60, 25 * 60),
      "coordinator": (None, 0.25, 1, 65 * 60, 0)}
PROBE_PATH = ROOT / "probe" / "probe.json"
CONTEXT_IDS = range(POOL_SIZE + 4)
# Extension point: map a supported stage to its real workers and dependencies
# here when census/Stage B are implemented. Do not register stages early.
STAGES = {"probe": (("U0", None),), "extract": tuple(("U1", model) for model in MODELS)}
FETCH_PREFIXES = ("analysis", "audit", "site")


def rate(kind):
    gpu, cpu, gib, _, _ = HW[kind]
    return (L4_RATE if gpu == "L4" else 0.0) + cpu * CPU_RATE + gib * MEM_RATE


base = (modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy==2.1.3")
        .env({"PYTHONPATH": "/opt/tm_exp1", "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
              "HF_HOME": "/vol/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}))
model_deps = base.pip_install("transformers==4.57.6", "huggingface_hub==0.36.0", "tokenizers==0.22.2",
                              "safetensors==0.6.2")
coord_image = base.add_local_file(str(HERE / "probe.py"), "/opt/tm_exp1/probe.py")
probe_image = (model_deps.pip_install("zstandard==0.25.0")
               .add_local_file(str(HERE / "probe.py"), "/opt/tm_exp1/probe.py"))
extract_image = (model_deps.pip_install("torch==2.5.1")
                 .add_local_file(str(HERE / "probe.py"), "/opt/tm_exp1/probe.py")
                 .add_local_file(str(HERE / "extract.py"), "/opt/tm_exp1/extract.py"))
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(APP_NAME, create_if_missing=True)


def resources(kind):
    gpu, cpu, gib, timeout, _ = HW[kind]
    return dict(gpu=gpu, cpu=cpu, memory=gib * 1024, timeout=timeout, volumes={str(ROOT): volume})


def _json_if_exists(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _probe_identity():
    if not PROBE_PATH.is_file():
        return None
    raw = PROBE_PATH.read_bytes()
    probe = json.loads(raw)
    if (probe["schema_version"] != 1 or probe["corpus"] != CORPUS or probe["models"] != MODELS
            or probe["context_tokens"] != CONTEXT_TOKENS
            or [c["id"] for c in probe["contexts"]] != list(CONTEXT_IDS)):
        raise ValueError("existing probe output does not match the U0 contract; inspect it before replacing")
    return hashlib.sha256(raw).hexdigest()


def _completion_path(model):
    return ROOT / "state" / "completed" / f"U1-{model}.json"


def _complete(kind, model, probe_sha):
    if kind == "U0":
        return {"probe_sha256": probe_sha} if probe_sha else None
    if kind != "U1":
        raise ValueError(f"unregistered worker kind: {kind}")
    record = _json_if_exists(_completion_path(model))
    if not record or record["probe_sha256"] != probe_sha or record["version"] != EXTRACTION_VERSION:
        return None
    if not (ROOT / "attn" / f"{model}.npz").is_file():
        return None
    for c in CONTEXT_IDS:
        if not ((ROOT / "acts" / model / f"c{c:02d}.npy").is_file()
                and (ROOT / "pred" / model / f"c{c:02d}.npz").is_file()):
            return None
    return record


@app.function(image=probe_image, max_containers=1, **resources("U0"))
def u0_probe():
    import probe

    start = time.monotonic()
    volume.reload()
    if (cached := _complete("U0", None, _probe_identity())) is not None:
        return dict(cached, cached=True, seconds=time.monotonic() - start)
    summary = probe.main(PROBE_PATH, ROOT / "hf")
    volume.commit()
    return dict(summary, cached=False, probe_sha256=_probe_identity(), seconds=time.monotonic() - start)


@app.function(image=extract_image, max_containers=2, **resources("U1"))
def u1_extract(model: str):
    import torch
    import extract

    start = time.monotonic()
    volume.reload()
    probe_sha = _probe_identity()
    if probe_sha is None:
        raise RuntimeError("U1 requires a completed probe stage")
    if (cached := _complete("U1", model, probe_sha)) is not None:
        return dict(cached, cached=True, seconds=time.monotonic() - start)
    if not torch.cuda.is_available():
        raise RuntimeError("U1 worker requires CUDA")
    marker = _completion_path(model)
    if marker.exists():
        marker.unlink()
        volume.commit()
    summary = extract.main(model, ROOT, ROOT / "hf")
    # Publish the marker only after all output files have been synced.
    volume.commit()
    record = dict(summary, cached=False, version=EXTRACTION_VERSION, probe_sha256=probe_sha)
    write_json_atomic(marker, record)
    volume.commit()
    return dict(record, seconds=time.monotonic() - start)


def _ledger_path(attempt):
    return ROOT / "state" / "attempts" / attempt["run_id"] / f"{attempt['unit']}.json"


def _reservation(run_id, kind, model=None):
    return dict(run_id=run_id, unit=kind if model is None else f"{kind}-{model}", model=model, kind=kind,
                started_at=time.time(), status="reserved", rate_usd_per_second=rate(kind),
                reserved_usd=(HW[kind][3] + STARTUP_SECONDS) * rate(kind))


def _reconcile(attempt, status, seconds, **extra):
    kind = attempt["kind"]
    seconds = min(max(seconds, 0.0), HW[kind][3])
    return dict(attempt, status=status, seconds=seconds, estimated_usd=(seconds + STARTUP_SECONDS) * rate(kind), **extra)


def _ledger_total():
    total = 0.0
    for path in (ROOT / "state" / "attempts").glob("*/*.json"):
        row = _json_if_exists(path)
        total += row.get("estimated_usd", row["reserved_usd"])
    return total


@app.function(image=coord_image, max_containers=1, **resources("coordinator"))
def coordinator(run_id: str, stage: str):
    """Run even after the caller exits; reserve attempts durably before dispatch."""
    volume.reload()
    owner = _reservation(run_id, "coordinator")
    write_json_atomic(_ledger_path(owner), owner)
    volume.commit()
    status = "failed"
    try:
        if stage not in STAGES:
            raise ValueError(f"unregistered stage: {stage}")
        probe_sha = _probe_identity()
        if stage == "extract" and probe_sha is None:
            raise RuntimeError("launch probe and wait for success before extract")
        pending, skipped = [], []
        for kind, model in STAGES[stage]:
            attempt = _reservation(run_id, kind, model)
            if _complete(kind, model, probe_sha) is None:
                pending.append(attempt)
            else:
                skipped.append(attempt["unit"])
        if _ledger_total() + sum(row["reserved_usd"] for row in pending) > STAGE_A_CAP:
            raise RuntimeError("Stage A ledger plus pending reservation exceeds the $8 cap")
        for attempt in pending:
            write_json_atomic(_ledger_path(attempt), attempt)
        volume.commit()

        # Extension point: add stage-specific fan-out here, not generic calls
        # that could run unimplemented stages. No automatic retry/spending.
        if stage == "extract" and pending:
            results = u1_extract.map([a["model"] for a in pending], return_exceptions=True)
        elif stage == "probe" and pending:
            results = u0_probe.starmap([()], return_exceptions=True)
        else:
            results = ()
        rows = []
        for attempt, result in zip(pending, results, strict=True):
            if isinstance(result, Exception):
                row = _reconcile(attempt, "failed", time.time() - attempt["started_at"],
                                 error_type=type(result).__name__)
            else:
                row = _reconcile(attempt, "cached" if result["cached"] else "done", result["seconds"], result=result)
            write_json_atomic(_ledger_path(row), row)
            volume.commit()
            rows.append(row)
        status = "failed" if any(row["status"] == "failed" for row in rows) else "done"
        manifest = dict(run_id=run_id, stage=stage, status=status, skipped=skipped, attempts=rows)
        write_json_atomic(ROOT / "state" / "runs" / f"{run_id}.json", manifest)
        return manifest
    finally:
        write_json_atomic(_ledger_path(owner), _reconcile(owner, status, time.time() - owner["started_at"]))
        volume.commit()


def _runs_dir():
    return HERE / "runs"


def cmd_deploy():
    subprocess.run([sys.executable, "-m", "modal", "deploy", str(Path(__file__).resolve())], check=True)


def cmd_launch(stage):
    units = STAGES[stage]
    wall = max(HW[k][4] for k, _ in units)
    estimate = sum(rate(k) * HW[k][4] for k, _ in units) + rate("coordinator") * (wall + STARTUP_SECONDS)
    print(f"Uncached estimate: ${estimate:.2f}, about {wall / 60:.0f} minutes (before queueing)", flush=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    fn = modal.Function.from_name(APP_NAME, "coordinator")
    call = fn.spawn(run_id, stage)
    record = dict(run_id=run_id, stage=stage, call_id=call.object_id, projected_usd=estimate)
    write_json_atomic(_runs_dir() / f"{run_id}.json", record)
    print(json.dumps(record, indent=2))


def cmd_status(run_id):
    records = sorted(_runs_dir().glob("*.json"))
    if not run_id and not records:
        raise SystemExit("No local launch records")
    path = _runs_dir() / f"{run_id}.json" if run_id else records[-1]
    record = _json_if_exists(path)
    if not record:
        raise SystemExit("Unknown run id")
    call = modal.FunctionCall.from_id(record["call_id"])
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        print(f"{record['run_id']}: queued or running; attempt records are under /state/attempts on the Volume")
    except Exception as error:
        raise SystemExit(f"Coordinator failed ({type(error).__name__}); inspect the tm-exp1 app logs") from None
    else:
        print(json.dumps(result, indent=2))
        if result["status"] != "done":
            raise SystemExit(1)


def cmd_fetch(out):
    dest = Path(out)
    count = 0
    for prefix in FETCH_PREFIXES:
        try:
            files = volume.listdir(f"/{prefix}", recursive=True)
        except FileNotFoundError:
            continue
        for item in files:
            if item.type != modal.volume.FileEntryType.FILE:
                continue
            path = Path(item.path.lstrip("/"))
            if not path.parts or path.parts[0] != prefix or ".." in path.parts:
                raise ValueError(f"unexpected Volume path: {path}")
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                for chunk in volume.read_file(item.path):
                    output.write(chunk)
            count += 1
    print(f"Fetched {count} files to {dest.resolve()}")


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deploy", help="deploy the app (requires spending approval)")
    commands.add_parser("launch", help="spawn a worker coordinator (requires spending approval)").add_argument(
        "stage", choices=STAGES)
    commands.add_parser("status").add_argument("run_id", nargs="?")
    commands.add_parser("fetch").add_argument("out", nargs="?", default=HERE / "out")
    args = parser.parse_args()
    if args.command == "deploy":
        cmd_deploy()
    elif args.command == "launch":
        cmd_launch(args.stage)
    elif args.command == "status":
        cmd_status(args.run_id)
    else:
        cmd_fetch(args.out)


if __name__ == "__main__":
    cli()
