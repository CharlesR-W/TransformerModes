"""Offline launcher tests; no Modal app runs, API requests or numerical solves."""
from contextlib import contextmanager
from decimal import Decimal
import json
import sys
from types import SimpleNamespace

import pytest

import modal_depth_run as launcher


def ledger():
    return dict(version=1, run='test-run', attempts=[])


def receipt(run, row, *, status='done', seconds=700, **extra):
    return dict(run=run['run'], id=row['id'], task=row['task'], status=status, seconds=seconds,
                fresh=True, peak_rss_bytes=1_900_000_000, **extra)


def add(run, task, *, retry=False):
    return launcher.new_attempt(run, task, launcher.reservation(task == launcher.PILOT), retry)


def pilot_done(run):
    p = add(run, launcher.PILOT)
    launcher.settle(run, p, receipt(run, p))
    return p


def expected_rate():
    return launcher.NONPREEMPTIBLE * (launcher.CPU * launcher.CPU_RATE + launcher.GIB * launcher.MEM_RATE)


def expected_full_seconds():
    # One attempt per unit, one pilot retry, both aggregates, and one shared retry priced as controls.
    return ((len(launcher.UNITS) + 1) * (launcher.UNIT_TIMEOUT + launcher.STARTUP)
            + (launcher.CROSS_TIMEOUT + launcher.STARTUP)
            + 2 * (launcher.CONTROLS_TIMEOUT + launcher.STARTUP))


def test_reservations_price_timeout_startup_and_retries():
    rate = expected_rate()
    assert launcher.RATE == rate
    pilot_seconds = 2 * (launcher.UNIT_TIMEOUT + launcher.STARTUP)
    full_seconds = expected_full_seconds()
    pilot, full = launcher.reservation(True), launcher.reservation()
    assert pilot['reserved_seconds'] == pilot_seconds
    assert full['reserved_seconds'] == full_seconds
    assert Decimal(pilot['reserved_usd']) == pilot_seconds * rate
    assert Decimal(full['reserved_usd']) == full_seconds * rate
    assert launcher.approve(False, full['reserved_usd'])['max_usd'] == full['reserved_usd']


@pytest.mark.parametrize('cap', ['below-quote', 'nan', 'inf', '-1', 'junk'])
def test_refuses_full_launch_before_contact(cap, monkeypatch, capsys):
    expected_usd = expected_full_seconds() * expected_rate()
    if cap == 'below-quote':
        cap = str(expected_usd.next_minus())
    monkeypatch.setitem(sys.modules, 'modal', None)  # The refusal must precede even importing the SDK.
    with pytest.raises(SystemExit) as error:
        launcher.main(['launch-rest', '--max-usd', cap])
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)['reserved_usd'] == str(expected_usd)


def test_launch_requires_explicit_max_usd(monkeypatch):
    monkeypatch.setitem(sys.modules, 'modal', None)
    with pytest.raises(SystemExit) as error:
        launcher.main(['launch', '--pilot', '--model', 'pythia', '--set', 'd32'])
    assert error.value.code == 2


def test_unit_list_matches_pipeline(monkeypatch):
    # Importing the actual list does not run any numerical work.
    monkeypatch.setenv('TM_DEPTH_BLAS_THREADS', '1')
    import depth

    expected = tuple(f'{m}/{s}' for m in depth.MODELS for s in depth.SETS)
    assert launcher.UNITS == expected
    assert len(launcher.UNITS) == len(set(launcher.UNITS)) == 22
    assert launcher.choose_tasks(ledger(), 'pilot') == ('pythia/d32',)
    assert len(launcher.REST) == 21 and 'pythia/d32' not in launcher.REST


def test_rest_requires_collected_fresh_healthy_pilot():
    run = ledger()
    with pytest.raises(ValueError, match='pilot receipt'):
        launcher.choose_tasks(run, 'rest')
    row = add(run, launcher.PILOT)
    with pytest.raises(ValueError, match='pilot receipt'):
        launcher.choose_tasks(run, 'rest')
    # The pilot must leave 25% of the unit deadline unused.
    ceiling = launcher.UNIT_TIMEOUT * 0.75
    launcher.settle(run, row, receipt(run, row, seconds=ceiling))
    with pytest.raises(ValueError, match='runtime gate'):
        launcher.choose_tasks(run, 'rest')
    row['receipt']['seconds'] = ceiling - 1
    row['receipt']['fresh'] = False
    with pytest.raises(ValueError, match='pilot receipt'):
        launcher.choose_tasks(run, 'rest')
    row['receipt'].update(fresh=True, peak_rss_bytes=4_500_000_000)
    with pytest.raises(ValueError, match='memory'):
        launcher.choose_tasks(run, 'rest')
    row['receipt']['peak_rss_bytes'] -= 1
    assert launcher.choose_tasks(run, 'rest') == launcher.REST


def test_interrupted_fanout_and_retry_budget():
    run = ledger()
    bad_pilot = add(run, launcher.PILOT)
    launcher.settle(run, bad_pilot, receipt(run, bad_pilot, status='timed_out', seconds=None))
    assert launcher.choose_tasks(run, 'pilot', launcher.PILOT) == (launcher.PILOT,)
    again = add(run, launcher.PILOT, retry=True)
    launcher.settle(run, again, receipt(run, again))
    a, b, c, *rest = launcher.REST
    ambiguous = add(run, a)
    failed = add(run, b)
    pending = add(run, c)
    pending['call_id'] = 'fc-test'
    launcher.settle(run, failed, receipt(run, failed, status='failed', seconds=12))
    assert launcher.choose_tasks(run, 'rest') == tuple(rest)
    with pytest.raises(ValueError, match='confirmed failure'):
        launcher.choose_tasks(run, 'rest', ambiguous['task'])
    assert launcher.choose_tasks(run, 'rest', b) == (b,)
    retried = add(run, b, retry=True)
    launcher.settle(run, retried, receipt(run, retried, status='failed'))
    with pytest.raises(ValueError, match='exhausted'):
        launcher.choose_tasks(run, 'rest', b)
    with pytest.raises(ValueError, match='22 successful'):
        launcher.choose_tasks(run, 'aggregate')


def test_aggregates_need_distinct_successful_units_and_share_retry():
    run = ledger()
    pilot_done(run)
    for task in launcher.REST:
        row = add(run, task)
        launcher.settle(run, row, receipt(run, row))
    assert launcher.choose_tasks(run, 'aggregate') == ('cross', 'controls')
    for task in launcher.AGGREGATES:
        row = add(run, task)
        launcher.settle(run, row, receipt(run, row, status='failed'))
    assert launcher.choose_tasks(run, 'aggregate', 'controls') == ('controls',)
    add(run, 'controls', retry=True)
    with pytest.raises(ValueError, match='exhausted'):
        launcher.choose_tasks(run, 'aggregate', 'cross')


class FakeVolume:
    def __init__(self):
        self.files = {}

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        yield self.files[path]

    @contextmanager
    def batch_upload(self, *, force):
        def put_file(source, path):
            if not force and path in self.files:
                raise FileExistsError(path)
            self.files[path] = source.read()
        yield SimpleNamespace(put_file=put_file)


def test_remote_ledger_is_authoritative_and_cannot_reset(tmp_path):
    volume, local = FakeVolume(), tmp_path / 'ledger.json'
    run = ledger()
    row = add(run, 'cross')
    launcher.save_ledger(volume, run, local, create=True)
    assert json.loads(volume.files[launcher.REMOTE_LEDGER]) == json.loads(local.read_bytes()) == run
    with pytest.raises(FileExistsError):
        launcher.save_ledger(volume, run, local, create=True)
    stale = local.read_bytes()
    row['call_id'] = 'fc-cross'
    seconds = Decimal('71.25')
    launcher.settle(run, row, receipt(run, row, seconds=float(seconds)))
    launcher.save_ledger(volume, run, local)
    billed_seconds = seconds + launcher.STARTUP
    assert row['estimated_billed_seconds'] == billed_seconds
    assert Decimal(row['estimated_usd']) == billed_seconds * expected_rate()
    local.write_bytes(stale)
    assert launcher.load_ledger(volume, local) == json.loads(local.read_bytes()) == run
    del volume.files[launcher.REMOTE_LEDGER]
    with pytest.raises(ValueError, match='refusing to reset'):
        launcher.load_ledger(volume, local)


def test_receipt_identity_and_charge_for_timeout():
    run = ledger()
    row = add(run, 'controls')
    wrong = dict(receipt(run, row), task='cross')
    with pytest.raises(ValueError, match='do not match'):
        launcher.settle(run, row, wrong)
    assert row['status'] == 'pending'
    launcher.settle(run, row, receipt(run, row, status='timed_out', seconds=None))
    billed_seconds = launcher.CONTROLS_TIMEOUT + launcher.STARTUP
    assert row['seconds'] is None and row['estimated_billed_seconds'] == billed_seconds
    assert Decimal(row['estimated_usd']) == billed_seconds * expected_rate()


def test_collect_recovers_receipt_after_lost_call_id(tmp_path, monkeypatch):
    import modal

    volume, local = FakeVolume(), tmp_path / 'ledger.json'
    run = ledger()
    row = add(run, launcher.PILOT)
    save = launcher.save_ledger
    save(volume, run, local, create=True)
    k = launcher.receipt_path(run['run'], row['id'])
    volume.files[k] = launcher.json_bytes(receipt(run, row, manifest_sha256='abc'))
    monkeypatch.setattr(launcher, 'save_ledger', lambda v, data: save(v, data, local))
    assert launcher.collect(modal, volume, run) == 0
    assert launcher.load_ledger(volume, local)['attempts'][0]['status'] == 'done'
    assert launcher.choose_tasks(run, 'rest') == launcher.REST


def test_recovers_committed_unit_before_final_receipt(tmp_path, monkeypatch):
    import modal

    volume, local = FakeVolume(), tmp_path / 'ledger.json'
    run = ledger()
    row = add(run, launcher.PILOT)
    row['submitted_at'] -= 900  # The original call has had time to produce the measured artifact.
    marker = launcher.json_bytes({'outputs_sha256': 'test'})
    stamp = receipt(run, row, precommit_seconds=701, marker_sha256=launcher.sha(marker), manifest_sha256='test')
    stamp.pop('seconds')
    stamp.pop('status')
    dest = 'depth/units/pythia/d32'
    volume.files[f'{dest}/run.json'] = launcher.json_bytes({'modal_attempt': stamp})
    # Neither an unstamped marker nor a stamp for another attempt can settle this attempt.
    assert launcher.recover_unit(volume.files.get, {**stamp, 'id': 'another-attempt'}) is None
    assert launcher.recover_unit(volume.files.get, {k: stamp[k] for k in ('id', 'run', 'task')}) is None
    volume.files[f'{dest}/done.json'] = marker
    save = launcher.save_ledger
    monkeypatch.setattr(launcher, 'save_ledger', lambda v, data: save(v, data, local))
    assert launcher.collect(modal, volume, run) == 0
    assert row['receipt']['recovered'] is True
    assert row['seconds'] is None and row['estimated_billed_seconds'] == launcher.UNIT_TIMEOUT + launcher.STARTUP
    assert launcher.choose_tasks(run, 'rest') == launcher.REST
    row['collected_at'] = row['submitted_at'] + 2000
    with pytest.raises(ValueError, match='runtime gate'):
        launcher.choose_tasks(run, 'rest')
    row['collected_at'] = row['submitted_at'] + 600
    with pytest.raises(ValueError, match='timestamps'):
        launcher.choose_tasks(run, 'rest')


def test_stage_a_probe_binding_is_in_completion_markers(monkeypatch):
    files = {launcher.PROBE: b'probe', launcher.INPUTS: b'frozen'}
    probe_sha, inputs_sha = (launcher.sha(files[key]) for key in (launcher.PROBE, launcher.INPUTS))
    monkeypatch.setattr(launcher, 'PROBE_SHA256', probe_sha)
    for model, marker in zip(launcher.MODELS, launcher.STAGE_A_MARKERS):
        path = f'acts/{model}/provenance.json'
        files[path] = launcher.json_bytes({'libraries': {'numpy': '2.1.3'}})  # No probe hash here.
        files[marker] = launcher.json_bytes(dict(version=2, probe_sha256=probe_sha, provenance_path=f'/{path}'))
        files[f'acts_depth/{model}/provenance.json'] = launcher.json_bytes(
            dict(probe_sha256=probe_sha, inputs_sha256=inputs_sha))
    assert launcher.input_identity(files.get) == {k: launcher.sha(v) for k, v in files.items()}
    files[launcher.INPUTS] = b'different frozen ids'
    with pytest.raises(ValueError, match='frozen synthetic inputs or probe mismatch'):
        launcher.input_identity(files.get)


def test_offline_sdk_registration():
    import modal

    volume = modal.Volume.from_name(launcher.VOLUME, create_if_missing=False)
    _, workers = launcher.make_app(modal, volume)  # Lazy SDK construction; not app.run().
    assert set(workers) == {'unit', 'cross', 'controls'}
