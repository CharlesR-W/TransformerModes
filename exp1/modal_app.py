"""Detached Modal entry points for experiment 1. Deploy and launch cost money.

After explicit approval (not needed for --help):
    uv run --with modal==1.6.2 python modal_app.py deploy
    uv run --with modal==1.6.2 python modal_app.py launch probe
    uv run --with modal==1.6.2 python modal_app.py launch extract
    uv run --with modal==1.6.2 python modal_app.py status [run_id]
    uv run --with modal==1.6.2 python modal_app.py fetch [out]

U1 requires a completed U0. /state holds completion markers, /ledger holds
attempt and run records. Only /analysis, /audit_a, /ledger, /site are fetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import modal

from probe import CONTEXT_TOKENS, CORPUS, MODELS, POOL_SIZE, PROBE_SCHEMA_VERSION, write_json_atomic

HERE = Path(__file__).resolve().parent
APP_NAME = "tm-exp1"
ROOT = Path("/vol")
CPU_RATE = 0.0000131
MEM_RATE = 0.00000222
GPU_RATES = {"L4": 0.000164, "A100-80GB": 0.000694}
STARTUP_SECONDS = 120
# $8 at first approval; raised to $11 by the owner on 2026-11-23 after the pilot measured 63 s per census pass,
# so that the dot reading is not dropped (expected total about $7.30; the rest is retry holds and margin).
STAGE_A_CAP = 11.0
EXTRACTION_VERSION = 2  # Bump when the U1 data or diagnostics change.
# (GPU, reserved cores, GiB, timeout seconds, expected work seconds before startup)
HW = {"U0": (None, 2, 8, 40 * 60, 20 * 60),
      "U1": ("L4", 4, 32, 50 * 60, 25 * 60),
      "pilot-gpu": ("A100-80GB", 8, 64, 15 * 60, 15 * 60),
      "pilot-cpu": (None, 16, 64, 10 * 60, 10 * 60),
      # Ceiling 55 min: 40 passes at the pilot's 63 s are 42 min before margin.
      "census": ("A100-80GB", 8, 64, 55 * 60, 42 * 60),
      # Two attempts failed on 2026-11-23: a 40-minute limit expired inside the synthetic census (100 runs, about
      # 34 minutes, untimed by the pilot), then a restarted worker repeated it and outlasted the coordinator. The
      # predictions are done; what remains is single-threaded, so the unit is smaller and every step is resumable.
      "readout": (None, 8, 32, 60 * 60, 30 * 60),
      # 50 min: the first attempt (2026-11-23) finished every check in about 25 minutes and then hit a 30-minute
      # limit while hashing the published files for the inventory, so no report was written.
      "audit-a": (None, 16, 64, 50 * 60, 40 * 60),
      # Must outlast the longest unit it waits for.
      "coordinator": (None, 0.25, 1, 100 * 60, 0)}
PROBE_PATH = ROOT / "probe" / "probe.json"
CONTEXT_IDS = range(POOL_SIZE + 4)
PRIORITY = (("cosine", "C"), ("cosine", "O"), ("dot", "C"), ("dot", "O"))
SITES, READOUT_SITES, AUDIT_SITES = tuple(range(2, 17, 2)), (4, 8, 12, 16), (4, 12)
MEMORIES, MASKS = ("none", "2", "8", "64", "whole"), ("none", "window", "other_texts")
PROJECTION_MARGIN, SPEC_MARGIN = .20, .10
PROJECTION_PASSES, SPEC_PASSES = 40, 39
VERIFICATION = ("readout", "audit-a")  # Up to two attempts per unit, not automatic retries.
A_VERSION = 1
STAGE_A_WORKERS = {"probe": (("U0", None),), "extract": tuple(("U1", model) for model in MODELS),
                   "A-pilot": (("pilot-gpu", None), ("pilot-cpu", None)),
                   "census": tuple(("census", model) for model in MODELS),
                   "readout": (("readout", None),), "audit-a": (("audit-a", None),)}
STAGE_B_WORKERS = {}  # Future Stage B workers require the separate approval flag.
FETCH_PREFIXES = ("analysis", "audit_a", "ledger", "site")


def _stage_units(stage, approve_stage_b):
    if stage in STAGE_B_WORKERS:
        if not approve_stage_b:
            raise ValueError("Stage B requires a separate --approve-stage-b flag")
        return STAGE_B_WORKERS[stage]
    if stage not in STAGE_A_WORKERS:
        raise ValueError(f"stage {stage!r} has no implemented worker")
    return STAGE_A_WORKERS[stage]


def rate(kind):
    gpu, cpu, gib, _, _ = HW[kind]
    return (GPU_RATES[gpu] if gpu else 0.0) + cpu * CPU_RATE + gib * MEM_RATE


base = (modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy==2.1.3")
        .env({"PYTHONPATH": "/opt/tm_exp1", "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false",
              "HF_HOME": "/vol/hf", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
              # hf_xet keeps a log file open under HF_HOME; on the Volume that blocks volume.reload() in a reused container.
              "HF_HUB_DISABLE_XET": "1"}))
model_deps = base.pip_install("transformers==4.57.6", "huggingface_hub==0.36.0", "tokenizers==0.22.2",
                              "safetensors==0.6.2")
coord_image = base.add_local_file(str(HERE / "probe.py"), "/opt/tm_exp1/probe.py")
probe_image = (model_deps.pip_install("zstandard==0.25.0")
               .add_local_file(str(HERE / "probe.py"), "/opt/tm_exp1/probe.py"))
extract_image = (model_deps.pip_install("torch==2.5.1")
                 .add_local_file(str(HERE / "probe.py"), "/opt/tm_exp1/probe.py")
                 .add_local_file(str(HERE / "extract.py"), "/opt/tm_exp1/extract.py"))
def _stage_image(image, *names):
    for name in ("probe.py", *names):
        image = image.add_local_file(str(HERE / name), f"/opt/tm_exp1/{name}")
    return image


cpu_env = {"OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "MKL_NUM_THREADS": "16"}
a_gpu_base = base.pip_install("torch==2.5.1")
a_cpu_base = (base.pip_install("torch==2.5.1+cpu", extra_index_url="https://download.pytorch.org/whl/cpu")
              .env(cpu_env))
census_image = _stage_image(a_gpu_base, "census.py")
pilot_gpu_image = _stage_image(a_gpu_base.pip_install("pytest==8.4.2", "scipy==1.15.3"), "census.py", "readout.py",
                               "synthetic.py", "select_b.py", "kernels.py", "test_census.py")
# transformers is here only so pack_site can decode prediction ids with the cached Pythia tokenizer (no download).
readout_image = _stage_image(
    a_cpu_base.pip_install("scipy==1.15.3", "transformers==4.57.6", "huggingface_hub==0.36.0",
                           "tokenizers==0.22.2", "safetensors==0.6.2"),
    "census.py", "readout.py", "synthetic.py", "select_b.py", "pack_site.py")
pilot_cpu_image = _stage_image(a_cpu_base, "census.py", "readout.py", "audit_a.py")
audit_image = _stage_image(base.env(cpu_env), "audit_a.py")
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
    if (probe["schema_version"] != PROBE_SCHEMA_VERSION or probe["corpus"] != CORPUS or probe["models"] != MODELS
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
    if not ((ROOT / "attn" / f"{model}.npz").is_file() and (ROOT / "acts" / model / "provenance.json").is_file()):
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
    record = dict(contexts=summary["contexts"], cached=False, version=EXTRACTION_VERSION, probe_sha256=probe_sha,
                  provenance_path=f"/acts/{model}/provenance.json")
    write_json_atomic(marker, record)
    volume.commit()
    return dict(record, seconds=time.monotonic() - start)


def _stage_marker(name):
    return ROOT / "state" / "stage_a" / f"{name}.json"


def _a_complete(name, probe_sha):
    record = _json_if_exists(_stage_marker(name))
    if record and record.get("version") == A_VERSION and record.get("probe_sha256") == probe_sha:
        return record
    return None


def _finish(name, probe_sha, started, **record):
    payload = dict(version=A_VERSION, probe_sha256=probe_sha, **record)
    write_json_atomic(_stage_marker(name), payload)
    volume.commit()
    return dict(payload, cached=False, seconds=time.monotonic() - started)


def _require_extracts():
    sha = _probe_identity()
    if not sha or not all(_complete("U1", model, sha) for model in MODELS):
        raise RuntimeError("A-pilot requires both completed extractions")
    return sha


def _group_done(root, model, group, probe_sha, sites=SITES):
    reading, centring = group
    key = f"{reading}_{centring}"
    base = root / "census" / model
    folder = base / key
    config = _json_if_exists(folder / "group.json")
    inputs = ("random", "embedding", *(f"site{s:02d}" for s in sites))
    if not (config and config.get("complete") is True and config.get("lists_saved") is True
            and config.get("probe_sha256") == probe_sha and config.get("inputs") == list(inputs)
            and config.get("memories") == list(MEMORIES) and config.get("masks") == list(MASKS)
            and config.get("suffix_screen") is True and config.get("pair_count") == 1000):
        return False
    audit = root / "audit_a" / "expected" / model / key
    return ((base / "pool.npz").is_file() and (root / "analysis" / "geometry" / model / f"{key}.json").is_file()
            and all((audit / f"{name}_pairs.npz").is_file() for name in inputs)
            and all(((folder / name / f"{memory}_{mask}.npz").is_file()
                     and (audit / name / f"{memory}_{mask}.npz").is_file())
                    for name in inputs for memory in MEMORIES for mask in MASKS))


def _nvidia_memory_mib(stop, values):
    while not stop.is_set():
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    check=True, capture_output=True, text=True, timeout=5)
            values.extend(int(line.strip()) for line in result.stdout.splitlines())
        except (OSError, subprocess.SubprocessError, ValueError):
            return
        stop.wait(1)


@app.function(image=pilot_gpu_image, max_containers=1, **resources("pilot-gpu"))
def a_pilot_gpu():
    import census
    import torch

    started = time.monotonic()
    volume.reload()
    sha = _require_extracts()
    if not torch.cuda.is_available():
        raise RuntimeError("A-pilot requires CUDA")
    model, group, site = "pythia", PRIORITY[0], 8
    stop, memory = threading.Event(), []
    sampler = threading.Thread(target=_nvidia_memory_mib, args=(stop, memory), daemon=True)
    sampler.start()
    try:
        env = dict(os.environ, CENSUS_DEVICE="cuda")
        # Do not inherit optional integration flags or download additional checkpoints.
        for key in ("CENSUS_TINY_REAL", "TM_EXP1_HF_FIXTURES"):
            env.pop(key, None)
        t = time.monotonic()
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "/opt/tm_exp1/test_census.py"], env=env, check=True)
        tests_seconds = time.monotonic() - t
        torch.cuda.reset_peak_memory_stats()
        t = time.monotonic()
        results = census.run_model(ROOT, model, groups=(group,), sites=(site,), device="cuda",
                                   output_root=ROOT / "a_pilot")
        torch.cuda.synchronize()
        shard_seconds = time.monotonic() - t
        t = time.monotonic()
        volume.commit()
        io_seconds = time.monotonic() - t
    finally:
        stop.set()
        sampler.join(timeout=6)
    if len(results) != 1 or results[0]["inputs"] != ["random", "embedding", "site08"]:
        raise RuntimeError("A-pilot did not run the model site and both adversary inputs")
    if not _group_done(ROOT / "a_pilot", model, group, sha, sites=(site,)):
        raise RuntimeError("A-pilot census shard did not publish its full outputs")
    seconds = time.monotonic() - started
    return _finish("pilot-gpu", sha, started, model=model, site=site, group=list(group),
                   stage_seconds=dict(cuda_tests=tests_seconds, shard=shard_seconds, volume_commit=io_seconds),
                   measured_passes=3, torch_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   nvidia_smi_peak_mib=max(memory, default=None), nvidia_smi_sample_period_seconds=1,
                   estimated_usd=(seconds + STARTUP_SECONDS) * rate("pilot-gpu"))


@app.function(image=pilot_cpu_image, max_containers=1, **resources("pilot-cpu"))
def a_pilot_cpu():
    import audit_a
    import readout

    started = time.monotonic()
    volume.reload()
    sha = _require_extracts()
    gpu = _a_complete("pilot-gpu", sha)
    if not gpu or not _group_done(ROOT / "a_pilot", "pythia", PRIORITY[0], sha, sites=(8,)):
        raise RuntimeError("CPU pilot requires its completed GPU census shard")
    t = time.monotonic()
    paths = readout.run_readout(ROOT, "pythia", groups=(PRIORITY[0],), sites=(8,), output_root=ROOT / "a_pilot")
    prediction_seconds = time.monotonic() - t
    if len(paths) != len(MEMORIES) * len(MASKS):
        raise RuntimeError("CPU pilot readout did not consume every memory and mask of its shard")
    t = time.monotonic()
    audit_row_seconds = float(audit_a.time_one_row(ROOT, "pythia"))
    audit_timing_seconds = time.monotonic() - t
    if not math.isfinite(audit_row_seconds) or audit_row_seconds <= 0:
        raise ValueError("audit pilot returned an invalid row duration")
    record = dict(
        gpu=gpu, stage_seconds=dict(prediction_readout=prediction_seconds, audit_timing=audit_timing_seconds),
        audit_row_seconds=audit_row_seconds,
        estimated_usd=(time.monotonic() - started + STARTUP_SECONDS) * rate("pilot-cpu"),
        projection=dict(margin=PROJECTION_MARGIN, passes_per_model=PROJECTION_PASSES,
                        spec_margin=SPEC_MARGIN, spec_passes_per_model=SPEC_PASSES, reuse_pilot_passes=False),
        readout_limitations="The pilot measures prediction readout at one site only. Timescales, stitches, synthetic "
                            "census and selection are unmeasured; the §11 duration is only a floor for projection.",
    )
    write_json_atomic(ROOT / "analysis" / "a_pilot.json", dict(record, probe_sha256=sha))
    return _finish("pilot-cpu", sha, started, **record)


@app.function(image=census_image, max_containers=2, **resources("census"))
def a_census(model: str, groups: tuple, probe_sha: str):
    import census
    import torch

    started = time.monotonic()
    volume.reload()
    if not torch.cuda.is_available():
        raise RuntimeError("UA census requires CUDA")
    if _require_extracts() != probe_sha:
        raise RuntimeError("census plan and source probe differ")
    if tuple(groups) != PRIORITY[:len(groups)] or len(groups) < 2:
        raise ValueError("census group list must be an approved priority prefix")
    done, reused = [], []
    for group in groups:
        if _group_done(ROOT, model, group, probe_sha):
            reused.append(group)
            continue
        # The census APIs recompute an entire group, even though its NPZ
        # files publish atomically. Commit the whole-group checkpoint.
        census.run_model(ROOT, model, groups=(group,), device="cuda")
        if not _group_done(ROOT, model, group, probe_sha):
            raise RuntimeError(f"incomplete census group for {model}/{group}")
        volume.commit()
        done.append(group)
    return dict(cached=not done, seconds=time.monotonic() - started, completed_groups=done, reused_groups=reused)


def _approved_groups(plan):
    return tuple(tuple(g) for g in plan["approved_groups"])


def _require_census(plan):
    groups = _approved_groups(plan)
    if groups not in (PRIORITY, PRIORITY[:3], PRIORITY[:2]):
        raise RuntimeError("no approved census group prefix")
    for model in MODELS:
        for group in groups:
            if not _group_done(ROOT, model, group, plan["probe_sha256"]):
                raise RuntimeError(f"readout requires completed census group {model}/{group}")
    return groups


@app.function(image=readout_image, max_containers=1, **resources("readout"))
def a_readout(plan: dict):
    import census
    import pack_site
    import readout
    import select_b
    import synthetic

    started = time.monotonic()
    volume.reload()
    groups = _require_census(plan)
    timings = {}

    def timed(key, fn, *args, **kwargs):
        t = time.monotonic()
        print(f"[readout] {key}: start", flush=True)
        value = fn(*args, **kwargs)
        timings[key] = time.monotonic() - t
        # Publish each step, so a later timeout does not lose it and the log shows where the time goes.
        volume.commit()
        print(f"[readout] {key}: {timings[key]:.0f} s", flush=True)
        return value

    readings = tuple(r for r, c in groups if c == "C")
    expected = len(groups) * len(READOUT_SITES) * len(MEMORIES) * len(MASKS)
    for model in MODELS:
        # Resume: the first attempt (run 20261123-231754) finished predictions and timescales for both models and
        # timed out later. A step is reused only if its summary exists and every per-setting file is present.
        have = sum(1 for _ in (ROOT / "readout" / model).rglob("*.npz")) if (ROOT / "readout" / model).is_dir() else 0
        if (ROOT / "analysis" / "prediction" / f"{model}.json").is_file() and have == expected:
            timings[f"{model}_predictions"] = 0.0
            print(f"[readout] {model}_predictions: reused {have} files", flush=True)
        else:
            paths = timed(f"{model}_predictions", readout.run_readout, ROOT, model, groups=groups)
            if len(paths) != expected:
                raise RuntimeError(f"incomplete prediction readout for {model}")
        if (ROOT / "analysis" / f"{model}_timescales.json").is_file():
            timings[f"{model}_timescales"] = 0.0
            print(f"[readout] {model}_timescales: reused", flush=True)
        else:
            timed(f"{model}_timescales", readout.run_timescales, ROOT, model)
        timed(f"{model}_stitches", census.run_stitches, ROOT, model, device="cpu", readings=readings)
    selection = ROOT / "analysis" / "stage_b_selection.json"
    if selection.is_file():
        timings["selection"] = 0.0
        print("[readout] selection: reused", flush=True)
    else:
        timed("selection", select_b.select, ROOT)
    # The synthetic census is a reported control, not a gate. All approved groups would be 100 runs of about 20 s;
    # it is run for the first group only (cosine reading, corpus mean: 25 runs), and the page says so.
    synthetic_done = ROOT / "analysis" / "synthetic" / "_done.json"
    synthetic_groups = groups[:1]
    if synthetic_done.is_file():
        timings["synthetic"] = 0.0
        print("[readout] synthetic: reused", flush=True)
    else:
        timed("synthetic", synthetic.run_synthetic, ROOT, device="cpu", groups=synthetic_groups)
        write_json_atomic(synthetic_done, dict(groups=[list(g) for g in synthetic_groups],
                                               note="first approved group only; see modal_app.a_readout"))
        volume.commit()
    timed("pack_site", pack_site.run, ROOT)
    _write_probe_summary()
    return _finish("readout", plan["probe_sha256"], started, stage_seconds=timings, approved_groups=plan["approved_groups"])


def _write_probe_summary():
    """Small, fetchable facts about the probe for the site's Method page (the probe file itself stays on the Volume)."""
    path = ROOT / "probe" / "probe.json"
    probe = json.loads(path.read_text(encoding="utf-8"))
    light = ("id", "kind", "subset", "file_index", "text_sha256", "eligible_shuffle_rank", "schedule", "segments")
    summary = dict(
        probe_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), corpus=probe["corpus"],
        models=probe["models"], context_tokens=probe["context_tokens"], pool_size=probe["pool_size"],
        counts=probe["counts"], seeds=probe["seeds"], token_identity=probe["token_identity"],
        tokenizer_provenance=probe["tokenizer_provenance"],
        contexts=[{k: c[k] for k in light if k in c} for c in probe["contexts"]],
        stitch_sources=[{k: c[k] for k in light if k in c} for c in probe["stitch_sources"]])
    write_json_atomic(ROOT / "analysis" / "probe_summary.json", summary)


def _audit_passed(report):
    return (isinstance(report, dict) and report.get("passed") is True and isinstance(report.get("checks"), list)
            and bool(report["checks"]) and isinstance(report.get("not_reproduced"), list))


def _readout_complete(plan):
    sha = plan["probe_sha256"]
    done = _a_complete("readout", sha)
    return bool(done and done.get("approved_groups") == plan["approved_groups"]
                and (ROOT / "analysis" / "stage_b_selection.json").is_file()
                and (ROOT / "site" / "manifest.json").is_file()
                and all((ROOT / "analysis" / "prediction" / f"{model}.json").is_file()
                        and (ROOT / "analysis" / f"{model}_timescales.json").is_file() for model in MODELS))


def _audit_complete(plan):
    done = _a_complete("audit-a", plan["probe_sha256"])
    return bool(done and done.get("approved_groups") == plan["approved_groups"]
                and done.get("rows") == plan["audit_rows"]
                and _audit_passed(_json_if_exists(ROOT / "audit_a" / "report.json")))


@app.function(image=audit_image, max_containers=1, **resources("audit-a"))
def a_audit(plan: dict):
    import audit_a

    started = time.monotonic()
    volume.reload()
    if not _readout_complete(plan):
        raise RuntimeError("audit requires completed Stage A readout")
    out = ROOT / "audit_a" / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    _stage_marker("audit-a").unlink(missing_ok=True)
    volume.commit()
    try:
        report = audit_a.run(ROOT, out, rows=plan["audit_rows"])
    finally:
        volume.commit()  # Preserve failed reports too; never write an expected result here.
    if not _audit_passed(report) or not _audit_passed(_json_if_exists(out)):
        return dict(passed=False, cached=False, seconds=time.monotonic() - started, report_path="/audit_a/report.json")
    return _finish("audit-a", plan["probe_sha256"], started, rows=plan["audit_rows"],
                   approved_groups=plan["approved_groups"], passed=True, report_path="/audit_a/report.json")


def _ledger_path(attempt):
    return ROOT / "ledger" / "attempts" / attempt["run_id"] / f"{attempt['unit']}.json"


def _reservation(run_id, kind, model=None, timeout=None):
    timeout = HW[kind][3] if timeout is None else timeout
    if not 0 < timeout <= HW[kind][3]:
        raise ValueError(f"invalid timeout for {kind}: {timeout}")
    return dict(run_id=run_id, unit=kind if model is None else f"{kind}-{model}", model=model, kind=kind,
                started_at=time.time(), status="reserved", rate_usd_per_second=rate(kind), timeout_seconds=timeout,
                reserved_usd=timeout * rate(kind), startup_allowance_usd=STARTUP_SECONDS * rate(kind))


def _reconcile(attempt, status, seconds, **extra):
    runtime = min(max(seconds, 0.0), attempt["timeout_seconds"])
    billed = runtime + STARTUP_SECONDS
    return dict(attempt, status=status, seconds=seconds, billed_seconds_estimate=billed,
                estimated_usd=billed * attempt["rate_usd_per_second"], **extra)


def _charge(row):
    # A reconciled provider bill (if later available) supersedes the estimate.
    if row.get("actual_usd") is not None:
        return row["actual_usd"]
    return row.get("estimated_usd", row["reserved_usd"] + row["startup_allowance_usd"])


def _ledger_rows():
    return [_json_if_exists(path) for path in sorted((ROOT / "ledger" / "attempts").glob("*/*.json"))]


def _ledger_total():
    return sum(map(_charge, _ledger_rows()))


def _require_budget(committed, pending, cap=None):
    cap = STAGE_A_CAP if cap is None else cap  # read at call time, so the cap is one setting
    if committed + sum(_charge(row) for row in pending) > cap + 1e-12:
        raise RuntimeError(f"Stage A ledger plus pending reservations exceeds the ${cap:g} cap")


def _plan_path():
    return ROOT / "analysis" / "stage_a_plan.json"


# Attempts allowed per verification unit. The readout gets a third: its first two attempts (2026-11-23) failed on
# limits that were set too short (a 40-minute unit limit, then the coordinator's 65-minute limit after a worker
# restart), not on its results; the third still fits under the owner's cap with the audit's two attempts held.
MAX_ATTEMPTS = {"readout": 3, "audit-a": 2}


def _held_verification(rows, plan):
    if not plan or plan.get("status") != "approved":
        return {}
    held = {}
    for kind in VERIFICATION:
        attempts = [r for r in rows if r["kind"] == kind]
        slots = (0 if any(r["status"] in ("done", "cached") for r in attempts)
                 else max(0, MAX_ATTEMPTS[kind] - len(attempts)))
        if slots:
            held[kind] = slots * _charge(_reservation("hold", kind))
    return held


# The audit's work that does not scale with the number of full rows (loading and centring, saved pairs, the 1% token
# sample, readout): its author's estimate at full size, not a measurement.
AUDIT_FIXED_SECONDS = 25 * 60  # measured on the first attempt: about 13 minutes of checks plus the inventory


def _audit_rows(pilot, groups):
    # audit_a.time_one_row times one query at one site of one model for ALL groups (both readings, both centrings,
    # all memories), so the number of approved groups does not multiply it. Each audit query visits both models and
    # both audit sites.
    del groups
    seconds_per_query = pilot["audit_row_seconds"] * len(MODELS) * len(AUDIT_SITES)
    budget = HW["audit-a"][3] - AUDIT_FIXED_SECONDS
    return min(POOL_SIZE, math.floor(budget / ((1 + PROJECTION_MARGIN) * seconds_per_query)))


def _remaining_groups(plan, model):
    return tuple(g for g in _approved_groups(plan) if not _group_done(ROOT, model, g, plan["probe_sha256"]))


def _census_reservations(run_id, plan):
    attempts = []
    for model in MODELS:
        groups = _remaining_groups(plan, model)
        if groups:
            seconds = max(1, math.ceil(plan["gpu_seconds_per_group"] * len(groups) * (1 + PROJECTION_MARGIN)))
            if seconds > HW["census"][3]:
                raise RuntimeError(f"projected census of {model} exceeds the configured "
                                   f"{HW['census'][3] // 60}-minute invocation ceiling")
            attempts.append(dict(_reservation(run_id, "census", model, seconds), groups=groups))
    return attempts


def _plan_census(sha, pilot, rows):
    gpu = pilot["gpu"]
    seconds_per_pass = ((gpu["stage_seconds"]["shard"] + gpu["stage_seconds"]["volume_commit"])
                        / gpu["measured_passes"])
    group_seconds = seconds_per_pass * PROJECTION_PASSES / len(PRIORITY)
    spent_and_active = sum(map(_charge, rows))
    base = dict(version=A_VERSION, probe_sha256=sha, projection_margin=PROJECTION_MARGIN,
                spec_projection_margin=SPEC_MARGIN, full_passes_per_model=PROJECTION_PASSES,
                spec_full_passes_per_model=SPEC_PASSES, gpu_seconds_per_group=group_seconds,
                spent_and_active_at_plan_usd=spent_and_active)
    trials = []
    for n in (4, 3, 2):
        groups = PRIORITY[:n]
        plan = dict(base, status="approved", approved_groups=[list(g) for g in groups],
                    dropped_groups=[list(g) for g in PRIORITY[n:]], audit_rows=_audit_rows(pilot, groups))
        holds = _held_verification(rows, plan)
        prediction_seconds = pilot["stage_seconds"]["prediction_readout"] * len(MODELS) * len(READOUT_SITES) * n
        cpu_seconds = max(HW["readout"][4], prediction_seconds)
        missing = [len(_remaining_groups(plan, model)) for model in MODELS]
        projected_census = sum((group_seconds * m + STARTUP_SECONDS) * rate("census") for m in missing if m)
        projected_verification = ((cpu_seconds + STARTUP_SECONDS) * rate("readout")
                                  + (HW["audit-a"][4] + STARTUP_SECONDS) * rate("audit-a"))
        projected = spent_and_active + (projected_census + projected_verification) * (1 + PROJECTION_MARGIN)
        reasons = []
        if projected > STAGE_A_CAP:
            reasons.append("projected work plus margin exceeds cap")
        if prediction_seconds > HW["readout"][3]:
            reasons.append("prediction readout alone exceeds its timeout")
        if plan["audit_rows"] < 1:
            reasons.append("one audit query across all models, sites and settings does not fit")
        hard_reserved = None
        try:
            attempts = _census_reservations("projection", plan)
            hard_reserved = spent_and_active + sum(holds.values()) + sum(map(_charge, attempts))
            _require_budget(spent_and_active + sum(holds.values()), attempts)
        except RuntimeError as error:
            reasons.append(str(error))
        trials.append(dict(groups=plan["approved_groups"], projected_usd=projected, admit_usd=hard_reserved,
                           projected_prediction_seconds=prediction_seconds, audit_rows=plan["audit_rows"], reasons=reasons))
        if not reasons:
            return dict(plan, planned_verification_holds_usd=holds, trials=trials)
    return dict(base, status="stopped", approved_groups=[], dropped_groups=[list(g) for g in PRIORITY], trials=trials)


def _run_batch(pending, dispatch, *, hold_usd=0.0, validate=None):
    if not pending:
        return []
    _require_budget(_ledger_total() + hold_usd, pending)
    for attempt in pending:
        write_json_atomic(_ledger_path(attempt), attempt)
    volume.commit()
    rows = []
    for attempt, result in zip(pending, dispatch(), strict=True):
        if isinstance(result, Exception):
            row = _reconcile(attempt, "failed", time.time() - attempt["started_at"],
                             error_type=type(result).__name__)
        else:
            valid = result.get("passed", True) is True
            if validate is not None and valid:
                volume.reload()
                valid = validate()
            outcome = ("cached" if result["cached"] else "done") if valid else "failed"
            row = _reconcile(attempt, outcome, result["seconds"], result=result)
        write_json_atomic(_ledger_path(row), row)
        volume.commit()
        rows.append(row)
    return rows


def _run_stage(run_id, stage, units):
    sha = _probe_identity()
    plan = _json_if_exists(_plan_path())
    if plan and plan.get("probe_sha256") != sha:
        raise RuntimeError("existing Stage A plan uses a different probe")
    rows, skipped = [], []
    hold = sum(_held_verification(_ledger_rows(), plan).values())
    if stage in ("probe", "extract"):
        if stage == "extract" and sha is None:
            raise RuntimeError("launch probe and wait for success before extract")
        pending = []
        for kind, model in units:
            attempt = _reservation(run_id, kind, model)
            if _complete(kind, model, sha) is None:
                pending.append(attempt)
            else:
                skipped.append(attempt["unit"])
        dispatch = (lambda: u0_probe.starmap([()], return_exceptions=True)) if stage == "probe" else (
            lambda: u1_extract.map([a["model"] for a in pending], return_exceptions=True))
        return _run_batch(pending, dispatch, hold_usd=hold), skipped, plan
    if stage == "A-pilot":
        sha = _require_extracts()
        # Each unit is admitted separately, before its own dispatch, even on retry.
        for kind, worker in (("pilot-gpu", a_pilot_gpu), ("pilot-cpu", a_pilot_cpu)):
            if _a_complete(kind, sha):
                skipped.append(kind)
                continue
            outcome = _run_batch([_reservation(run_id, kind)], lambda: worker.starmap([()], return_exceptions=True),
                                 hold_usd=hold, validate=lambda: _a_complete(kind, sha) is not None)
            rows.extend(outcome)
            if outcome[0]["status"] == "failed":
                break
        return rows, skipped, plan
    if stage == "census":
        sha = _require_extracts()
        pilot = _a_complete("pilot-cpu", sha)
        if not _a_complete("pilot-gpu", sha) or not pilot or not (ROOT / "analysis" / "a_pilot.json").is_file():
            raise RuntimeError("census requires both completed A-pilot units")
        if not plan or plan.get("status") != "approved":
            plan = _plan_census(sha, pilot, _ledger_rows())
            # Persist the priority decision and both retry holds before any census work starts.
            write_json_atomic(_plan_path(), plan)
            volume.commit()
        if plan["status"] != "approved":
            raise RuntimeError("even the two cosine groups do not fit; see /analysis/stage_a_plan.json")
        attempts = _census_reservations(run_id, plan)
        hold = sum(_held_verification(_ledger_rows(), plan).values())
        skipped = [f"census-{m}" for m in MODELS if all(a["model"] != m for a in attempts)]
        # One worker per model, with each platform timeout exactly matching its reservation.
        # Spawn the full admitted wave, then collect results; no remote retries.
        def dispatch():
            calls = []
            for attempt in attempts:
                try:
                    worker = a_census.with_options(timeout=attempt["timeout_seconds"], retries=0)
                    calls.append(worker.spawn(attempt["model"], _approved_groups(plan), sha))
                except Exception as error:
                    calls.append(error)
            for call in calls:
                try:
                    yield call if isinstance(call, Exception) else call.get()
                except Exception as error:
                    yield error
        rows = _run_batch(attempts, dispatch, hold_usd=hold)
        if all(r["status"] != "failed" for r in rows):
            volume.reload()
            _require_census(plan)
            write_json_atomic(_stage_marker("census"), dict(version=A_VERSION, probe_sha256=sha,
                                                            approved_groups=plan["approved_groups"]))
            volume.commit()
        return rows, skipped, plan
    if stage not in VERIFICATION:
        raise ValueError(f"no dispatch implemented for {stage!r}")
    if not plan or plan.get("status") != "approved":
        raise RuntimeError(f"{stage} requires an approved census plan")
    _require_census(plan)
    if stage == "audit-a" and not _readout_complete(plan):
        raise RuntimeError("audit requires completed Stage A readout")
    complete = _readout_complete if stage == "readout" else _audit_complete
    if complete(plan):
        return [], [stage], plan
    held = _held_verification(_ledger_rows(), plan)
    if stage not in held:
        raise RuntimeError(f"{stage} has no retry reservation left")
    attempt = _reservation(run_id, stage)
    worker = a_readout if stage == "readout" else a_audit
    return (_run_batch([attempt], lambda: worker.starmap([(plan,)], return_exceptions=True),
                       hold_usd=sum(held.values()) - _charge(attempt), validate=lambda: complete(plan)), [], plan)


# Non-preemptible so that a preempted coordinator cannot take its GPU workers down with it (the workers resume per
# group, the coordinator does not). Added on 2026-11-23 while diagnosing a stalled census; the actual cause of that
# stall was the Modal workspace reaching its billing-cycle spend limit, which stops every container.
@app.function(image=coord_image, max_containers=1, nonpreemptible=True, **resources("coordinator"))
def coordinator(run_id: str, stage: str, approve_stage_b: bool = False):
    """Run even after the caller exits; reserve attempts durably before dispatch."""
    units = _stage_units(stage, approve_stage_b)
    volume.reload()
    owner = _reservation(run_id, "coordinator")
    write_json_atomic(_ledger_path(owner), owner)
    volume.commit()
    status = "failed"
    try:
        rows, skipped, plan = _run_stage(run_id, stage, units)
        status = "failed" if any(row["status"] == "failed" for row in rows) else "done"
        manifest = dict(run_id=run_id, stage=stage, status=status, skipped=skipped, attempts=rows)
        if plan:
            manifest.update(approved_groups=plan["approved_groups"], dropped_groups=plan["dropped_groups"])
        write_json_atomic(ROOT / "ledger" / "runs" / f"{run_id}.json", manifest)
        return manifest
    except (RuntimeError, ValueError) as error:
        manifest = dict(run_id=run_id, stage=stage, status="failed", reason=str(error))
        write_json_atomic(ROOT / "ledger" / "runs" / f"{run_id}.json", manifest)
        return manifest
    finally:
        write_json_atomic(_ledger_path(owner), _reconcile(owner, status, time.time() - owner["started_at"]))
        volume.commit()


def _runs_dir():
    return HERE / "runs"


def cmd_deploy():
    subprocess.run([sys.executable, "-m", "modal", "deploy", str(Path(__file__).resolve())], check=True)


def _volume_files(prefix):
    try:
        return volume.listdir(prefix, recursive=True)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []


def _volume_json(path):
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except FileNotFoundError:
        return None


def _remote_ledger():
    return [_volume_json(e.path) for e in _volume_files("/ledger/attempts")
            if e.type == modal.volume.FileEntryType.FILE and e.path.endswith(".json")]


def _spend_summary(rows, plan):
    by_unit = {}
    for row in rows:
        entry = by_unit.setdefault(row["unit"], dict(spent_usd=0., active_usd=0., held_usd=0.))
        active = row["status"] == "reserved" and row.get("actual_usd") is None
        entry["active_usd" if active else "spent_usd"] += _charge(row)
    for kind, held in _held_verification(rows, plan).items():
        by_unit.setdefault(kind, dict(spent_usd=0., active_usd=0., held_usd=0.))["held_usd"] = held
    spent = sum(r["spent_usd"] for r in by_unit.values())
    reserved = sum(r["active_usd"] + r["held_usd"] for r in by_unit.values())
    return dict(spent_usd=spent, reserved_usd=reserved, remaining_usd=STAGE_A_CAP - spent - reserved, units=by_unit)


def _check_launch_prereqs(stage):
    required = {
        "A-pilot": tuple(f"/state/completed/U1-{model}.json" for model in MODELS),
        "census": ("/state/stage_a/pilot-gpu.json", "/state/stage_a/pilot-cpu.json", "/analysis/a_pilot.json"),
        "readout": ("/state/stage_a/census.json", "/analysis/stage_a_plan.json"),
        "audit-a": ("/state/stage_a/readout.json", "/analysis/stage_a_plan.json", "/analysis/stage_b_selection.json"),
    }
    for path in required.get(stage, ()):
        if not _volume_json(path):
            raise SystemExit(f"Cannot launch {stage}: missing predecessor output {path}")


def cmd_launch(stage, approve_stage_b=False):
    units = _stage_units(stage, approve_stage_b)
    _check_launch_prereqs(stage)
    durations = [HW[k][4] + STARTUP_SECONDS for k, _ in units]
    wall = sum(durations) if stage == "A-pilot" else max(durations)
    estimate = (sum(rate(k) * (HW[k][4] + STARTUP_SECONDS) for k, _ in units)
                + rate("coordinator") * (wall + STARTUP_SECONDS))
    print(f"Uncached ledger prior: ${estimate:.2f}, about {wall / 60:.0f} minutes (before queueing)", flush=True)
    if stage == "census":
        print("The coordinator will use the measured pilot and retry holds before admitting any setting.", flush=True)
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    # Prevent a new paid coordinator when even its reservation cannot fit.
    # The serialized coordinator remains authoritative for worker admission.
    if stage not in ("probe", "extract"):
        summary = _spend_summary(_remote_ledger(), _volume_json("/analysis/stage_a_plan.json"))
        _require_budget(summary["spent_usd"] + summary["reserved_usd"], [_reservation(run_id, "coordinator")])
    fn = modal.Function.from_name(APP_NAME, "coordinator")
    call = fn.spawn(run_id, stage, approve_stage_b)
    record = dict(run_id=run_id, stage=stage, call_id=call.object_id, projected_usd=estimate)
    write_json_atomic(_runs_dir() / f"{run_id}.json", record)
    print(json.dumps(record, indent=2))


def cmd_status(run_id):
    summary = _spend_summary(_remote_ledger(), _volume_json("/analysis/stage_a_plan.json"))
    print(f"Stage A ledger: spent ${summary['spent_usd']:.3f}, reserved ${summary['reserved_usd']:.3f}, "
          f"remaining ${summary['remaining_usd']:.3f} of ${STAGE_A_CAP:.2f} (estimates)")
    for unit, costs in sorted(summary["units"].items()):
        print(f"  {unit}: spent ${costs['spent_usd']:.3f}, reserved "
              f"${costs['active_usd'] + costs['held_usd']:.3f} (retry/verification holds ${costs['held_usd']:.3f})")
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
        print(f"{record['run_id']}: queued or running; attempt records are under /ledger/attempts on the Volume")
    except Exception as error:
        raise SystemExit(f"Coordinator failed ({type(error).__name__}); inspect the tm-exp1 app logs") from None
    else:
        print(json.dumps(result, indent=2))
        if result["status"] != "done":
            raise SystemExit(1)


def cmd_fetch(out):
    dest = Path(out)
    count = 0
    audit_received = False
    for prefix in FETCH_PREFIXES:
        for item in _volume_files(f"/{prefix}"):
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
            if path == Path("audit_a/report.json"):
                audit_received = True
    print(f"Fetched {count} files to {dest.resolve()}")
    if not audit_received or not _audit_passed(_json_if_exists(dest / "audit_a" / "report.json")):
        raise SystemExit("Stage A audit is missing or failed; these files are not verified Stage A results")


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("deploy", help="deploy the app (requires spending approval)")
    launch = commands.add_parser("launch", help="spawn a worker coordinator (requires spending approval)")
    launch.add_argument("stage", choices=(*STAGE_A_WORKERS, *STAGE_B_WORKERS))
    launch.add_argument("--approve-stage-b", action="store_true", help="separate approval for future Stage B workers")
    commands.add_parser("status").add_argument("run_id", nargs="?")
    commands.add_parser("fetch").add_argument("out", nargs="?", default=HERE / "out")
    args = parser.parse_args()
    if args.command == "deploy":
        cmd_deploy()
    elif args.command == "launch":
        cmd_launch(args.stage, args.approve_stage_b)
    elif args.command == "status":
        cmd_status(args.run_id)
    else:
        cmd_fetch(args.out)


if __name__ == "__main__":
    cli()
