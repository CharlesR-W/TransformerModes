"""Offline Stage A state-machine checks. All remote workers and the Volume are fakes."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

modal = pytest.importorskip("modal")
import modal_app as a


@pytest.fixture
def env(tmp_path, monkeypatch):
    class FakeVolume:
        last_commit = set()

        def reload(self):
            pass

        def commit(self):
            self.last_commit = {str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.json")}

    vol, done, extracted = FakeVolume(), set(), set(a.MODELS)
    monkeypatch.setattr(a, "ROOT", tmp_path)
    monkeypatch.setattr(a, "volume", vol)
    monkeypatch.setattr(a, "_probe_identity", lambda: "probe-sha")
    monkeypatch.setattr(a, "_complete", lambda kind, model, sha: model in extracted or None)
    monkeypatch.setattr(a, "_group_done", lambda root, model, group, sha, sites=a.SITES: (model, group, sites) in done)
    return SimpleNamespace(root=tmp_path, vol=vol, done=done, extracted=extracted)


def _run(run_id, stage):
    with pytest.warns(UserWarning, match="executing locally"):
        return a.coordinator.local(run_id, stage)


def _stub(fn):
    def mapper(args, *, return_exceptions):
        assert return_exceptions
        return iter(fn(*row) for row in args)
    return SimpleNamespace(starmap=mapper)


def _marker(name, **kwargs):
    a.write_json_atomic(a._stage_marker(name), dict(version=a.A_VERSION, probe_sha256="probe-sha", **kwargs))


def _pilot(shard=120, audit=25):
    return dict(gpu=dict(stage_seconds=dict(shard=shard, volume_commit=0), measured_passes=3),
                stage_seconds=dict(prediction_readout=2), audit_row_seconds=audit)


def _historic_spend(amount):
    row = dict(a._reservation("history", "U0"), status="done", actual_usd=amount)
    a.write_json_atomic(a._ledger_path(row), row)
    return row


def _complete_census_setup(env):
    plan = a._plan_census("probe-sha", _pilot(), [_historic_spend(2)])
    assert plan["status"] == "approved"
    a.write_json_atomic(a._plan_path(), plan)
    for model in a.MODELS:
        for group in a._approved_groups(plan):
            env.done.add((model, group, a.SITES))
    return plan


def _readout_outputs(plan):
    _marker("readout", approved_groups=plan["approved_groups"])
    for path in ("analysis/stage_b_selection.json", "site/manifest.json",
                 *(f"analysis/prediction/{m}.json" for m in a.MODELS),
                 *(f"analysis/{m}_timescales.json" for m in a.MODELS)):
        a.write_json_atomic(a.ROOT / path, {"test": True})


def test_prerequisite_refusals_before_dispatch(env, monkeypatch):
    env.extracted.remove("olmo")
    assert "both completed extractions" in _run("bad1", "A-pilot")["reason"]
    env.extracted.add("olmo")
    assert "pilot" in _run("bad2", "census")["reason"]
    assert "approved census plan" in _run("bad3", "readout")["reason"]
    _complete_census_setup(env)
    assert "completed Stage A readout" in _run("bad4", "audit-a")["reason"]
    assert {row["kind"] for row in a._ledger_rows() if row["run_id"].startswith("bad")} == {"coordinator"}
    monkeypatch.setattr(a, "_volume_json", lambda _: None)
    with pytest.raises(SystemExit, match="missing predecessor output"):
        a.cmd_launch("readout")  # returns before remote coordinator lookup


def test_pilot_dispatches_gpu_then_cpu_with_its_own_reservation(env, monkeypatch):
    seen = []

    def worker(kind):
        path = f"ledger/attempts/pilot/{kind}.json"
        assert path in env.vol.last_commit
        assert json.loads((a.ROOT / path).read_text())["status"] == "reserved"
        if kind == "pilot-gpu":
            assert "ledger/attempts/pilot/pilot-cpu.json" not in env.vol.last_commit
        else:
            assert (a.ROOT / "state/stage_a/pilot-gpu.json").is_file()
        seen.append(kind)
        _marker(kind)
        return dict(seconds=12, cached=False)

    monkeypatch.setattr(a, "a_pilot_gpu", _stub(lambda: worker("pilot-gpu")))
    monkeypatch.setattr(a, "a_pilot_cpu", _stub(lambda: worker("pilot-cpu")))
    assert _run("pilot", "A-pilot")["status"] == "done"
    assert seen == ["pilot-gpu", "pilot-cpu"]
    assert _run("again", "A-pilot")["skipped"] == seen


@pytest.mark.parametrize("shard,spent,n", [(60, 1.6, 4), (110, 1.6, 3), (120, 2.0, 2), (310, 2.0, 0)])
def test_stop_rule_drops_tails_without_breaking_cosine_pair(env, shard, spent, n):
    plan = a._plan_census("probe-sha", _pilot(shard), [_historic_spend(spent)])
    assert a._approved_groups(plan) == a.PRIORITY[:n]
    assert plan["dropped_groups"] == [list(g) for g in a.PRIORITY[n:]]
    assert plan["projection_margin"] == .20 and plan["spec_projection_margin"] == .10
    assert plan["full_passes_per_model"] == 40 and plan["spec_full_passes_per_model"] == 39
    if n:
        assert 0 < plan["audit_rows"] < 32
        assert plan["trials"][-1]["admit_usd"] <= a.STAGE_A_CAP
        assert plan["trials"][-1]["projected_usd"] <= a.STAGE_A_CAP
        assert set(plan["planned_verification_holds_usd"]) == set(a.VERIFICATION)
    else:
        assert plan["status"] == "stopped" and all(trial["reasons"] for trial in plan["trials"])


def test_census_holds_both_verifiers_before_dispatch_and_skips_completed_models(env, monkeypatch):
    _historic_spend(2)
    _marker("pilot-gpu")
    _marker("pilot-cpu", **_pilot())
    a.write_json_atomic(a.ROOT / "analysis/a_pilot.json", {"test": True})
    seen = []

    def options(*, timeout, retries):
        assert timeout <= 2400 and retries == 0

        def spawn(model, groups, sha):
            assert sha == "probe-sha"
            plan = json.loads(a._plan_path().read_text())
            assert "analysis/stage_a_plan.json" in env.vol.last_commit
            assert f"ledger/attempts/census/census-{model}.json" in env.vol.last_commit
            assert groups == a._approved_groups(plan)
            holds = plan["planned_verification_holds_usd"]
            assert holds == {k: 2 * a._charge(a._reservation("fixture", k)) for k in a.VERIFICATION}
            assert a._ledger_total() + sum(holds.values()) <= 8
            for group in groups:
                env.done.add((model, group, a.SITES))
            seen.append(model)
            return SimpleNamespace(get=lambda: dict(cached=False, seconds=60))
        return SimpleNamespace(spawn=spawn)

    monkeypatch.setattr(a, "a_census", SimpleNamespace(with_options=options))
    assert _run("census", "census")["status"] == "done"
    assert seen == list(a.MODELS) and a._stage_marker("census").is_file()
    assert _run("again", "census")["skipped"] == ["census-pythia", "census-olmo"]
    assert seen == list(a.MODELS)


def test_worker_skips_only_completed_groups(env, monkeypatch):
    import torch

    env.done.add(("pythia", a.PRIORITY[0], a.SITES))
    ran = []

    def run_model(root, model, *, groups, device):
        assert root == env.root and device == "cuda"
        ran.extend(groups)
        for group in groups:
            env.done.add((model, group, a.SITES))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, "census", SimpleNamespace(run_model=run_model))
    with pytest.warns(UserWarning, match="executing locally"):
        result = a.a_census.local("pythia", a.PRIORITY[:2], "probe-sha")
    assert ran == [a.PRIORITY[1]] and result["reused_groups"] == [a.PRIORITY[0]]


def test_partial_group_is_not_a_checkpoint(tmp_path):
    model, group, sha = "pythia", a.PRIORITY[0], "fixture"
    base = tmp_path / "census" / model / "cosine_C"
    audit = tmp_path / "audit_a/expected" / model / "cosine_C"
    names = ["random", "embedding", "site08"]
    config = dict(complete=False, lists_saved=True, probe_sha256=sha, suffix_screen=True, pair_count=1000,
                  inputs=names, memories=list(a.MEMORIES), masks=list(a.MASKS))
    a.write_json_atomic(base / "group.json", config)
    assert not a._group_done(tmp_path, model, group, sha, sites=(8,))
    a.write_json_atomic(base / "group.json", dict(config, complete=True))
    assert not a._group_done(tmp_path, model, group, sha, sites=(8,))  # marker alone is insufficient
    geometry = tmp_path / "analysis/geometry" / model / "cosine_C.json"
    a.write_json_atomic(geometry, [])
    files = [base.parent / "pool.npz", *(audit / f"{name}_pairs.npz" for name in names)]
    files += [tree / name / f"{mem}_{mask}.npz" for tree in (base, audit) for name in names
              for mem in a.MEMORIES for mask in a.MASKS]
    for file in files:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    assert a._group_done(tmp_path, model, group, sha, sites=(8,))
    files[-1].unlink()
    assert not a._group_done(tmp_path, model, group, sha, sites=(8,))


def test_readout_and_audit_convert_holds_and_charge_failed_audit(env, monkeypatch):
    plan = _complete_census_setup(env)

    def readout(received):
        assert received == plan
        held = a._held_verification(a._ledger_rows(), plan)
        assert held["readout"] == a._charge(a._reservation("check", "readout"))
        assert held["audit-a"] == 2 * a._charge(a._reservation("check", "audit-a"))
        _readout_outputs(plan)
        return dict(cached=False, seconds=40)

    monkeypatch.setattr(a, "a_readout", _stub(readout))
    assert _run("readout", "readout")["status"] == "done"
    assert "readout" not in a._held_verification(a._ledger_rows(), plan)

    def lying_audit(received):
        assert received == plan
        a.write_json_atomic(a.ROOT / "audit_a/expected/fake.json", {"passed": True})
        return dict(passed=True, cached=False, seconds=20)  # no independent report or marker

    monkeypatch.setattr(a, "a_audit", _stub(lying_audit))
    failed = _run("audit1", "audit-a")
    assert failed["status"] == "failed" and a._charge(failed["attempts"][0]) > 0
    assert a._held_verification(a._ledger_rows(), plan) == {"audit-a": a._charge(a._reservation("check", "audit-a"))}
    assert _run("audit2", "audit-a")["status"] == "failed"
    assert "no retry reservation" in _run("audit3", "audit-a")["reason"]
    summary = a._spend_summary(a._ledger_rows(), plan)
    assert summary["reserved_usd"] == 0 and summary["units"]["audit-a"]["spent_usd"] > 0
    assert summary["remaining_usd"] == pytest.approx(8 - summary["spent_usd"])


def test_cpu_pilot_uses_shard_but_times_audit_at_real_root(env, monkeypatch):
    _marker("pilot-gpu", **_pilot()["gpu"])
    env.done.add(("pythia", a.PRIORITY[0], (8,)))

    def run_readout(root, model, *, groups, sites, output_root):
        assert (root, model, groups, sites, output_root) == (env.root, "pythia", (a.PRIORITY[0],), (8,), env.root / "a_pilot")
        return range(len(a.MEMORIES) * len(a.MASKS))

    def time_one_row(root, model):
        assert (root, model) == (env.root, "pythia")
        return 12.5

    monkeypatch.setitem(sys.modules, "readout", SimpleNamespace(run_readout=run_readout))
    monkeypatch.setitem(sys.modules, "audit_a", SimpleNamespace(time_one_row=time_one_row))
    with pytest.warns(UserWarning, match="executing locally"):
        pilot = a.a_pilot_cpu.local()
    assert pilot["audit_row_seconds"] == 12.5 and "unmeasured" in pilot["readout_limitations"]
    durable = json.loads((env.root / "analysis/a_pilot.json").read_text())
    assert durable["projection"]["passes_per_model"] == 40


def test_readout_worker_routes_groups_sites_and_stitch_readings(env, monkeypatch):
    plan = _complete_census_setup(env)
    groups = a._approved_groups(plan)
    seen = []

    def prediction(root, model, *, groups):
        assert root == env.root and groups == a._approved_groups(plan)
        seen.append((model, "prediction"))
        return range(len(groups) * len(a.READOUT_SITES) * len(a.MEMORIES) * len(a.MASKS))

    def timescales(root, model):
        assert root == env.root
        seen.append((model, "timescales"))

    def stitches(root, model, *, device, readings):
        assert root == env.root and device == "cpu" and readings == tuple(r for r, c in groups if c == "C")
        seen.append((model, "stitches"))

    def synthetic(root, *, device, groups):
        assert root == env.root and device == "cpu" and groups == a._approved_groups(plan)
        seen.append("synthetic")

    def select(root):
        assert root == env.root
        seen.append("select")

    def pack(root):
        assert root == env.root
        seen.append("pack_site")

    monkeypatch.setitem(sys.modules, "readout", SimpleNamespace(run_readout=prediction, run_timescales=timescales))
    monkeypatch.setitem(sys.modules, "census", SimpleNamespace(run_stitches=stitches))
    monkeypatch.setitem(sys.modules, "synthetic", SimpleNamespace(run_synthetic=synthetic))
    monkeypatch.setitem(sys.modules, "select_b", SimpleNamespace(select=select))
    monkeypatch.setitem(sys.modules, "pack_site", SimpleNamespace(run=pack))
    with pytest.warns(UserWarning, match="executing locally"):
        result = a.a_readout.local(plan)
    assert result["approved_groups"] == plan["approved_groups"]
    assert seen == [(m, part) for m in a.MODELS for part in ("prediction", "timescales", "stitches")] + ["synthetic", "select", "pack_site"]


def test_cpu_audit_uses_only_explicit_independent_report(env, monkeypatch):
    plan = _complete_census_setup(env)
    _readout_outputs(plan)

    def run(root, out, rows=None):
        assert root == env.root and out == root / "audit_a/report.json"
        assert rows == plan["audit_rows"]
        result = {"passed": True, "checks": ["fixture"], "not_reproduced": []}
        a.write_json_atomic(out, result)
        return result

    monkeypatch.setitem(sys.modules, "audit_a", SimpleNamespace(run=run))
    with pytest.warns(UserWarning, match="executing locally"):
        result = a.a_audit.local(plan)
    assert result["passed"] and a._audit_complete(plan)
    (env.root / "audit_a/report.json").unlink()
    assert not a._audit_complete(plan)


def test_fetch_requires_current_passing_report_even_if_local_report_is_stale(tmp_path, monkeypatch):
    report_path = "audit_a/report.json"
    passing = dict(passed=True, checks=["fixture"], not_reproduced=[])
    paths = {"audit_a/expected/example.json": passing}
    a.write_json_atomic(tmp_path / report_path, passing)

    def listdir(prefix, *, recursive):
        assert recursive and prefix.lstrip("/") in a.FETCH_PREFIXES
        return [SimpleNamespace(path=path, type=modal.volume.FileEntryType.FILE)
                for path in paths if path.startswith(prefix.lstrip("/"))]

    monkeypatch.setattr(a, "volume", SimpleNamespace(listdir=listdir,
                                                     read_file=lambda path: [json.dumps(paths[path]).encode()]))
    with pytest.raises(SystemExit, match="missing or failed"):
        a.cmd_fetch(tmp_path)
    paths[report_path] = dict(passing, passed=False)
    with pytest.raises(SystemExit, match="missing or failed"):
        a.cmd_fetch(tmp_path)
    paths[report_path] = passing
    a.cmd_fetch(tmp_path)
