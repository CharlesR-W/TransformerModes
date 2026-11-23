"""The independent Stage A audit against a small case produced by running the pipeline.

The pipeline (probe structures, extract.py on Pythia-70m, census.py, readout.py, select_b.py) writes a
complete small Stage A tree; audit_a.py must pass on it unchanged and fail when any one stored value it
is meant to guard is corrupted. Pythia-70m (revision a39f36b) must be in the local Hugging Face cache.
"""
from __future__ import annotations

import inspect
import json
import shutil

import numpy as np
import pytest

import audit_a

L, START, SUFFIX = 299, 40, 200          # positions 1..L; pool START..L; suffix SUFFIX..L
POOL_TEXTS, STITCH = (0, 1, 2, 3), 4
SITES = (2, 4, 6)                        # Pythia-70m has hidden states 0..6
MODULES = ("textwrap", "heapq", "fractions", "calendar", "shlex", "queue")
REVISION = "a39f36b100fe8a5377810d56c3f4789b9c53ac42"


def _texts(tokenizer):
    import importlib
    out = []
    for name in MODULES:
        ids = tokenizer.encode(inspect.getsource(importlib.import_module(name)), add_special_tokens=False)
        ids = [t for t in ids if t not in (0, 1)]
        assert len(ids) >= 2 * L
        out.append(ids)
    return out


def _probe(tokenizer):
    import probe
    ids = _texts(tokenizer)
    contexts = [dict(id=c, kind="pool", token_ids=ids[c][:L], model_input_ids=probe.model_input_ids(ids[c][:L]))
                for c in POOL_TEXTS]
    sources = [dict(source=label, subset=f"subset {label}", file_index=100 + n, token_ids=ids[4 + n],
                    token_strings=[str(t) for t in ids[4 + n]]) for n, label in enumerate("AB")]
    contexts.append(probe.stitch(sources, STITCH, "short", np.random.default_rng(0), tokenizer, size=L))
    q_rng, a_rng = np.random.default_rng(21), np.random.default_rng(22)
    query, audit = [], []
    for c in POOL_TEXTS:
        late = q_rng.choice(L + 1 - SUFFIX, size=8, replace=False) + SUFFIX
        early = q_rng.choice(SUFFIX - START, size=4, replace=False) + START
        query += [[c, int(p)] for p in (*late, *early)]
        audit.append([c, int(a_rng.choice(late))])
    nodes = dict(census_pool=[[c, START, L + 1] for c in POOL_TEXTS], suffix=[[c, SUFFIX, L + 1] for c in POOL_TEXTS],
                 query_sample=query, audit_queries=audit,
                 one_text={str(c): [[c, START, L + 1]] for c in (*POOL_TEXTS, STITCH)})
    return dict(contexts=contexts, node_sets=nodes, models={"pythia": dict(id="EleutherAI/pythia-70m", eot=0)})


def build_case(root):
    """Run the pipeline end to end on a small probe; returns the root."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import census
    import extract
    import readout
    import select_b
    torch.set_num_threads(4)
    try:
        options = dict(revision=REVISION, local_files_only=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", **options)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m", attn_implementation="eager", use_safetensors=True, dtype=torch.float32,
            **options).eval()
    except OSError as error:
        pytest.skip(f"Pythia-70m is not in the local cache: {error}")
    data = _probe(tokenizer)
    census.write_json(root / "probe" / "probe.json", data)
    extract.extract_contexts(model, data["contexts"], tokenizer.eos_token_id, root, "pythia", query_start=START,
                             query_split=SUFFIX, causal_split=150, model_revision=REVISION)
    census.run_model(root, "pythia", sites=SITES, device="cpu", row_block=64)
    census.run_stitches(root, "pythia", sites=SITES, row_block=64)
    readout.run_readout(root, "pythia", sites=(4,))
    select_b.select(root, models=("pythia",), sites=SITES, readout_sites=(4,), required_sites=2)
    return root


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return build_case(tmp_path_factory.mktemp("stage_a"))


def _audit(root):
    return audit_a.run(root, root / "audit_report.json", spec_parameters=False)


def _failed(report):
    return {c["name"] for c in report["checks"] if not c["passed"]}


def test_audit_passes_on_unmodified_pipeline_output(case):
    report = _audit(case)
    assert "error" not in report, report.get("traceback")
    assert report["passed"], json.dumps([c for c in report["checks"] if not c["passed"]], indent=1)[:4000]
    assert json.loads((case / "audit_report.json").read_text())["passed"]
    compared = {}
    for c in report["checks"]:
        compared[c["name"]] = compared.get(c["name"], 0) + c["compared"]
    for name in ("row.membership", "row.cutoff_d64", "row.tie_ids", "row.overlap_counts", "row.overlap_maximum",
                 "screen.membership", "screen.mass", "screen.removed_short_range_mass", "screen.overlap_maximum",
                 "pairs.kernel", "pairs.distance", "pairs.diagonal", "pairs.sample_replay",
                 "sample.origins", "sample.overlap_counts", "sample.stitch_baseline", "aggregate.geometry_summary",
                 "readout.control_replay", "readout.control_constraints", "readout.document_contributions",
                 "readout.score", "selection.M", "selection.site_values", "layout.pool_text"):
        assert compared.get(name, 0) > 0, name
    # 4 audit queries x 4 groups x 5 memories x 3 masks at the one audit site present (site 4)
    assert compared["row.membership"] == 4 * 4 * 5 * 3
    assert any(a["what"] == "pythia/site12" for a in report["absent"])
    assert report["not_reproduced"] and report["conventions_trusted"]


def test_time_one_row(case):
    assert audit_a.time_one_row(case, "pythia") > 0


def test_plan_file_governs_groups(case, tmp_path):
    """A capped run: the dot groups were dropped by the plan, so they were never computed."""
    import select_b
    root = tmp_path / "planned"
    shutil.copytree(case, root)
    for group in ("dot_C", "dot_O"):
        for part in ("census", "readout", "audit_a/expected"):
            shutil.rmtree(root / part / "pythia" / group)
    select_b.select(root, models=("pythia",), sites=SITES, readout_sites=(4,), required_sites=2)
    plan = dict(approved_groups=[["cosine", "C"], ["cosine", "O"]], dropped_groups=[["dot", "C"], ["dot", "O"]],
                models=["pythia"])
    (root / "analysis" / "stage_a_plan.json").write_text(json.dumps(plan))
    report = audit_a.run(root, root / "r.json", rows=1, spec_parameters=False)
    assert report["passed"], _failed(report)
    assert report["groups"] == [["cosine", "C"], ["cosine", "O"]]
    assert sum("absent by plan" in a["reason"] for a in report["absent"]) == 2
    assert not [c["scope"] for c in report["checks"] if "dot_" in c["scope"]]   # stale stitched dot output ignored
    shutil.rmtree(root / "census" / f"pythia_stitch{STITCH:02d}" / "cosine_C")  # approved stitched work missing
    report = audit_a.run(root, root / "r.json", rows=1, spec_parameters=False)
    assert "presence.stitched_group_complete" in _failed(report)
    shutil.rmtree(root / "census" / "pythia" / "cosine_O" / "site04")
    report = audit_a.run(root, root / "r.json", rows=1, spec_parameters=False)
    assert not report["passed"] and "presence.setting_file" in _failed(report)


def test_tie_rule_overlaps_and_conditionals_agree_with_pipeline_on_planted_ties():
    """Model-site rows rarely tie, so compare the audit's section 4/5B arithmetic with census.py on planted
    exact ties (both implementations are independent; the test is where they meet)."""
    torch = pytest.importorskip("torch")
    import census
    rng = np.random.default_rng(3)
    k = 64
    for trial in range(300):
        n = int(rng.integers(40, 400))
        levels = np.sort(rng.uniform(0, 1, size=int(rng.integers(1, 6))))
        rows = []
        for _ in range(2):
            d = rng.choice(levels, size=n) if trial % 3 else rng.uniform(0, 1, n)
            d = d * (1 + rng.choice([0, 0, 4e-13, -4e-13, 4e-12], size=n))   # inside and outside the tolerance
            d[rng.random(n) < .2] = np.inf                                     # masked
            rows.append(d)
        tops, mine = [], []
        for d in rows:
            t = census.top_with_ties(torch.as_tensor(d[None]), k)
            cand = np.flatnonzero(np.isfinite(d))
            m = audit_a.top_with_ties(cand, d[cand], k, 1e-12)
            got = t.indices[0].numpy()
            got = got[got >= 0]
            assert set(got.tolist()) == set(np.r_[m["strict"], m["taken"]].tolist())
            assert set(np.flatnonzero(t.classes[0].numpy() == 2).tolist()) == set(m["ties"].tolist())
            assert (int(t.t_all[0]), int(t.t_in[0])) == (m["t_all"], m["t_in"])
            tops.append(t)
            mine.append(m)
        counts = census.overlap_counts(*tops).numpy()
        e, mx = census.overlap_values(counts, *(v.numpy() for v in (tops[0].t_in, tops[0].t_all,
                                                                     tops[1].t_in, tops[1].t_all)))
        c, e2, m2 = audit_a.overlap(*mine)
        assert counts[0].tolist() == c and float(mx[0]) == m2 and abs(float(e[0]) - e2) < 1e-12
    # section 9.1: 64 picked from 1,024 tied against a set tying all 1,024
    a = dict(strict=np.arange(0), ties=np.arange(64), taken=np.arange(64), t_in=64, t_all=64)
    b = dict(strict=np.arange(0), ties=np.arange(1024), taken=np.arange(64), t_in=64, t_all=1024)
    c, e, m = audit_a.overlap(a, b)
    assert e / 64 == .0625 and m / 64 == 1
    # suffix-screen conditionals, including unattainable targets from tied minima
    d = rng.uniform(.05, 1, (40, 300))
    d[rng.random(d.shape) < .3] = np.inf
    d[0, :70] = .01
    d[1, :64] = .01
    d[2, 70:] = np.inf
    probs, status, beta, dmin, _ = census.cond_rows(torch.as_tensor(d))
    for row in range(len(d)):
        ok = np.isfinite(d[row])
        mine = audit_a.conditional_row(d[row][ok], 64)
        assert mine["status"] == int(status[row])
        if mine["status"] == 0:
            np.testing.assert_allclose(probs[row].numpy()[ok], mine["q"], atol=1e-12, rtol=0)
            assert abs(mine["beta"] - float(beta[row])) <= 1e-9 * max(1., abs(mine["beta"]))


def test_selection_with_an_example_x(tmp_path):
    """Hand-built screen and readout statistics (as in test_census) on which select_b.py chooses M and X."""
    import census as c
    import readout as r
    import select_b as s
    probe = dict(contexts=[dict(id=i, kind="pool", token_ids=list(range(10, 22))) for i in range(2)],
                 node_sets=dict(census_pool=[[i, 2, 13] for i in range(2)], suffix=[[i, 8, 13] for i in range(2)],
                                query_sample=[[i, 9] for i in range(2)], audit_queries=[[i, 9] for i in range(2)]))
    c.write_json(tmp_path / "probe" / "probe.json", probe)

    def fixture(model, reading, centring, memory, mask, site, worst=.35, tied=False):
        p = c.setting_path(tmp_path, model, reading, centring, c.input_name(site), memory, mask)
        c.atomic(p, np.savez, screen_mass=[[.66, .25, .09 if mask == "none" else 0.]],
                 screen_status=[0], screen_origins=[[45, 14, 5]], screen_removed=[.3],
                 screen_overlap_expected=[[20, 4 if tied else 64 * worst, 10]],
                 screen_overlap_maximum=[[20, 64 if tied else 64 * worst, 10]])

    for model, (reading, centring) in [(m, g) for m in ("pythia", "olmo") for g in c.PRIORITY]:
        c.write_json(tmp_path / "census" / model / f"{reading}_{centring}" / "group.json",
                     dict(inputs=["site04", "site08"], memories=list(c.MEMORIES), masks=list(c.MASKS), k=64,
                          references=["local", "random", "embedding"], complete=True, window=64, target=64))
        for site in (4, 8):
            for memory, mask, worst, tie in (("2", "none", .35, True), ("8", "none", .25, False),
                                             ("64", "none", .35, False), ("whole", "none", .4, False),
                                             ("2", "window", .05, False)):
                fixture(model, reading, centring, memory, mask, site, worst, tie)
                by_doc = np.zeros((2, 6, len(r.FIELDS)))
                by_doc[:, 0, r.F["matched"]] = 201
                by_doc[:, 0, r.F["advantage"]] = 20 if (site == 4 and reading == "cosine" and centring == "C"
                                                        and memory == "64") else 2
                c.atomic(r.readout_path(tmp_path, model, reading, centring, site, memory, mask), np.savez,
                         by_document=by_doc, documents=np.arange(2), min_matched=200, fields=r.FIELDS,
                         reasons=r.REASONS, strata=r.STRATA, bootstrap_seed=r.BOOTSTRAP_SEED, bootstrap_draws=200)
    stored = s.select(tmp_path, sites=(4, 8), readout_sites=(4,), required_sites=2)
    assert stored["X"] is not None

    def audit_selection():
        a = audit_a.Audit(tmp_path, spec_parameters=False)
        a.plan()
        a.selection()
        return [ch for ch in a.rep.result() if ch["name"].startswith("selection")]

    checks = audit_selection()
    assert checks and all(ch["passed"] for ch in checks), [ch for ch in checks if not ch["passed"]]
    path = tmp_path / "analysis" / "stage_b_selection.json"
    data = json.loads(path.read_text())
    original = json.loads(path.read_text())
    data["X"] = dict(data["X"], memory="whole")
    path.write_text(json.dumps(data))
    assert "selection.X" in {ch["name"] for ch in audit_selection() if not ch["passed"]}
    # false evidence behind the right winner
    data = json.loads(json.dumps(original))
    row = next(r for r in data["table"] if r["construction"] == data["X"])
    row["prediction"]["common_bootstrap_p5"] = -1.0
    path.write_text(json.dumps(data))
    assert {ch["name"] for ch in audit_selection() if not ch["passed"]} == {"selection.X_evidence"}


def _rewrite(path, change):
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    change(data)
    np.savez(path, **data)


def _audit_row(root):
    probe = json.loads((root / "probe" / "probe.json").read_text())
    c, p = probe["node_sets"]["audit_queries"][0]
    return POOL_TEXTS.index(c) * (L + 1 - START) + p - START


def corrupt_neighbour(root):
    i = _audit_row(root)

    def change(d):
        row = d["indices"][i]
        d["indices"][i, 5] = next(j for j in range(len(d["t_in"])) if j not in row and j != i)
    _rewrite(root / "census/pythia/cosine_C/site04/8_none.npz", change)
    return "row.membership"


def corrupt_tie_count(root):
    i = _audit_row(root)
    _rewrite(root / "census/pythia/dot_C/site04/whole_window.npz", lambda d: d["t_all"].__setitem__(i, d["t_all"][i] + 1))
    return "row.t_all"


def corrupt_saved_distance(root):
    def change(d):
        d["distance"][3, 10] += 1e-7
    _rewrite(root / "audit_a/expected/pythia/dot_O/site02_pairs.npz", change)
    return "pairs.distance"


def corrupt_census_statistic(root):
    def change(d):
        row = next(r for r in range(len(d["origins"])) if d["origins"][r, 0] > 0)
        d["origins"][row] += [-1, 1, 0]
    _rewrite(root / "census/pythia/cosine_O/site02/64_window.npz", change)
    return "aggregate.geometry_summary"


def corrupt_readout_contribution(root):
    path = root / "readout/pythia/cosine_C/site04/2_none.npz"

    def change(d):
        fields = [str(f) for f in d["fields"]]
        matched = d["by_document"][..., fields.index("matched")]
        doc, stratum = np.unravel_index(int(np.argmax(matched)), matched.shape)
        d["by_document"][doc, stratum, fields.index("advantage")] += 0.25
    _rewrite(path, change)
    return "readout.stratum_values"


def corrupt_selected_construction(root):
    path = root / "analysis" / "stage_b_selection.json"
    data = json.loads(path.read_text())
    m = data["M"] or dict(reading="cosine", centring="C", memory="8", mask="none")
    data["M"] = dict(m, memory="whole" if m["memory"] != "whole" else "2")
    path.write_text(json.dumps(data))
    return "selection.M"


def corrupt_control(root):
    path = root / "readout/pythia/dot_O/site04/64_none.npz"

    def change(d):
        k = int(np.flatnonzero(d["reason"] == 0)[0])
        d["control"][k] = d["neighbour"][k]
    _rewrite(path, change)
    return "readout.control_replay"


def corrupt_removed_mass(root):
    i = _audit_row(root)
    path = root / "census/pythia/cosine_C/site04/8_window.npz"

    def change(d):
        r = int(np.flatnonzero(d["screen_query"] == i)[0])
        d["screen_removed"][r] = 0.5 if np.isnan(d["screen_removed"][r]) else d["screen_removed"][r] + 1e-6
    _rewrite(path, change)
    return "screen.removed_short_range_mass"


def corrupt_missing_member(root):
    path = root / "census/pythia/cosine_C/site02/8_window.npz"
    _rewrite(path, lambda d: d.pop("indices"))
    return "integrity.setting_members"


def corrupt_screen_maximum_overlap(root):
    def change(d):
        r = next(r for r, q in enumerate(d["screen_query"]) if q != _audit_row(root))
        d["screen_overlap_maximum"][r, 1] += 1
    _rewrite(root / "census/pythia/dot_O/site06/64_none.npz", change)
    return "formula.screen_overlap_maximum"


@pytest.mark.parametrize("corrupt", [corrupt_neighbour, corrupt_tie_count, corrupt_saved_distance,
                                     corrupt_census_statistic, corrupt_readout_contribution,
                                     corrupt_selected_construction, corrupt_control, corrupt_removed_mass,
                                     corrupt_missing_member, corrupt_screen_maximum_overlap],
                         ids=lambda f: f.__name__)
def test_audit_fails_on_each_corruption(case, tmp_path, corrupt):
    root = tmp_path / "corrupted"
    shutil.copytree(case, root)
    expected_check = corrupt(root)
    report = _audit(root)
    assert "error" not in report, report.get("traceback")
    assert not report["passed"]
    assert expected_check in _failed(report), sorted(_failed(report))
