"""Development fixture: run the real Stage A pipeline end to end on a tiny case and lay out its outputs exactly as
`modal_app.py fetch` will (analysis/, audit_a/, ledger/, site/), so the site can be built and checked before the
real census exists.

    TMPDIR=/var/tmp uv run --with numpy --with pytest --with torch==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu --with transformers==4.57.6 --with scipy \
        --with zstandard python site/dev_fixture.py [--out data/fixture]

What it runs (all CPU, about five minutes, under 2 GB): the small case of exp1/test_audit_a.py (six standard-library
Python modules tokenised by Pythia: four texts of 299 tokens with nodes from position 40, one stitched text), extended
with display strings so the packer can show text; extraction with two cached models, Pythia-70m (as "pythia", hidden
states 2, 4, 6) and Pythia-160m (as "pythia160m", hidden states 4, 8, 12), so the viewer's Model and Layer controls
have something to switch; census, stitched census, prediction readout, timescales, a one-seed synthetic census,
the Stage B selection, the viewer pack and the independent audit. The ledger is a placeholder (zero dollars, marked
"fixture"), because no money was spent.

None of these numbers are results. The site build marks every page of a build from this directory as a development
fixture (`build_site.py --dev`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXP1 = REPO / "exp1"
HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
MODELS = {  # key -> (hub id, revision, census sites)
    "pythia": ("EleutherAI/pythia-70m", "a39f36b100fe8a5377810d56c3f4789b9c53ac42", (2, 4, 6)),
    "pythia160m": ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46", (4, 8, 12)),
}
SUBSETS = ["Python stdlib: textwrap", "Python stdlib: heapq", "Python stdlib: fractions", "Python stdlib: calendar"]


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def build(work: Path):
    sys.path.insert(0, str(EXP1))
    import numpy as np
    import torch
    import transformers

    import audit_a
    import census
    import extract
    import pack_site
    import probe as probe_mod
    import readout
    import select_b
    import synthetic
    import test_audit_a as t

    torch.set_num_threads(2)
    pid, prev, _ = MODELS["pythia"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(pid, revision=prev, local_files_only=True)
    data = t._probe(tokenizer)
    # The audit fixture omits display strings and document metadata; add them the way probe.py does.
    for c in data["contexts"]:
        if c["kind"] == "pool":
            c["token_strings"] = [probe_mod.token_display(tokenizer, i) for i in c["token_ids"]]
            c["text"] = tokenizer.decode(c["token_ids"], clean_up_tokenization_spaces=False)
            c["subset"] = SUBSETS[c["id"]]
            c["file_index"] = c["id"]
            c["text_sha256"] = hashlib.sha256(c["text"].encode()).hexdigest()
        else:
            c["token_strings"] = [probe_mod.token_display(tokenizer, i) for i in c["token_ids"]]
            c["text"] = tokenizer.decode(c["token_ids"], clean_up_tokenization_spaces=False)
    data["models"] = {k: dict(id=v[0], revision=v[1], eot=0) for k, v in MODELS.items()}
    data["description"] = "development fixture: four Python standard-library modules and one stitched text"
    data["corpus"] = dict(id="fixture (Python standard library source, not the Pile)")
    census.write_json(work / "probe" / "probe.json", data)
    (work / "hf").symlink_to(HUB)          # lets pack_site decode prediction ids with the cached tokenizer
    plan = dict(approved_groups=[list(g) for g in census.PRIORITY], dropped_groups=[], models=list(MODELS),
                probe_sha256=hashlib.sha256((work / "probe" / "probe.json").read_bytes()).hexdigest(),
                note="development fixture plan: every group approved, nothing spent")
    census.write_json(work / "analysis" / "stage_a_plan.json", plan)

    for key, (hub_id, rev, sites) in MODELS.items():
        log(f"extract {key} ({hub_id})")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            hub_id, revision=rev, local_files_only=True, attn_implementation="eager", use_safetensors=True,
            dtype=torch.float32).eval()
        extract.extract_contexts(model, data["contexts"], tokenizer.eos_token_id, work, key, query_start=t.START,
                                 query_split=t.SUFFIX, causal_split=150, model_revision=rev)
        del model
        log(f"census {key} sites {sites}")
        census.run_model(work, key, sites=sites, device="cpu", row_block=64)
        census.run_stitches(work, key, sites=sites, row_block=64)
        log(f"readout and timescales {key}")
        readout.run_readout(work, key, sites=tuple(s for s in sites if s in (4, 8, 12, 16)))
        readout.run_timescales(work, key, sites=sites)
    log("synthetic census (one seed, small)")
    synthetic.run_synthetic(work, seeds=range(1), device="cpu", row_block=64, start=40,
                            generator_options=dict(texts=4, positions=300, width=32, sites=3, vocab=64))
    log("selection")
    select_b.select(work, models=tuple(MODELS), sites=(4,), readout_sites=(4,), required_sites=1)
    log("pack")
    pack_site.run(work)
    log("audit")
    report = audit_a.run(work, work / "audit_a" / "report.json", spec_parameters=False)
    log(f"audit passed: {report.get('passed')}")
    if not report.get("passed"):
        failed = sorted({c["name"] for c in report.get("checks", []) if not c["passed"]})
        log("audit failures:", failed, report.get("error"))
    # Placeholder ledger in the fetched layout: the fixture cost nothing.
    run_id = "fixture"
    for unit in ("probe", "extract-pythia", "census-pythia", "readout", "audit-a"):
        row = dict(run_id=run_id, unit=unit, kind=unit.split("-")[0], status="finished", fixture=True,
                   rate_usd_per_second=0.0, timeout_seconds=1, reserved_usd=0.0, startup_allowance_usd=0.0,
                   seconds=0.0, billed_seconds_estimate=0.0, estimated_usd=0.0)
        census.write_json(work / "ledger" / "attempts" / run_id / f"{unit}.json", row)
    census.write_json(work / "ledger" / "runs" / f"{run_id}.json", dict(run_id=run_id, stage="fixture", projected_usd=0.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "data" / "fixture", help="fetched-layout output (replaced)")
    ap.add_argument("--keep-work", action="store_true", help="keep the full pipeline tree (in $TMPDIR)")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="tm-fixture-"))
    log(f"work tree {work}")
    try:
        build(work)
        out = a.out.resolve()
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for part in ("analysis", "audit_a", "ledger", "site"):   # what `modal_app.py fetch` copies
            shutil.copytree(work / part, out / part)
        (out / "FIXTURE.json").write_text(json.dumps(dict(
            kind="development fixture", made=time.strftime("%Y-%m-%d %H:%M"), models={k: v[0] for k, v in MODELS.items()},
            note="not results; see site/dev_fixture.py"), indent=1))
        log(f"wrote {out}")
    finally:
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
