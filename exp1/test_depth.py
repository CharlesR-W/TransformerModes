"""Known-answer and reproducibility checks for the token depth pipeline."""
from collections import Counter

import numpy as np
import pytest

import depth
import depth_dm as dm
import spacetime as st


def test_radius_replay_with_duplicates_and_reindexing():
    rng = np.random.default_rng(991)
    raw = rng.standard_normal((65, 29))
    raw[:20] = raw[0]
    raw[21:24] = raw[20]
    x = dm.unit_vectors(raw)
    baseline = dm.radius_kernel(x)
    assert baseline['tie_share'] >= 20 / len(x)
    assert np.count_nonzero(baseline['P'][0, 1:20] > 1e-3) == 19
    assert np.array_equal(baseline['P'], baseline['P'].T)
    np.testing.assert_array_equal(baseline['P'].diagonal(), 0)
    query_order = rng.permutation(len(x))
    for order in (np.arange(len(x)), rng.permutation(len(x))):
        z = dm.unit_vectors(raw[order])
        target = baseline['P'][np.ix_(order, order)]
        ker = dm.radius_kernel(z)
        np.testing.assert_array_equal(ker['rk'], baseline['rk'][order])
        np.testing.assert_allclose(ker['P'], target, atol=1e-11, rtol=0)
        for size in (1, 7, len(x)):
            for start in range(0, len(x), size):
                ii = query_order[start:start + size]
                pr, _ = dm.radius_rows(dm.unit_vectors(raw[order[ii]]), z, ker, mask_self=ii)
                np.testing.assert_array_equal(pr[np.arange(len(ii)), ii], 0)
                np.testing.assert_array_equal(pr > 1e-5, target[ii] > 1e-5)
                assert (.5 * np.abs(pr - target[ii]).sum(axis=1)).max() < 1e-9


def planted(layers):
    rng = np.random.default_rng(72)
    ps = np.empty((layers, 48, 48))
    axes = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    label = np.repeat([-1.0, 1.0], 24)
    for l in range(layers):
        x = label[:, None] * axes + .03 * rng.standard_normal((48, len(axes)))
        ps[l] = dm.radius_kernel(dm.unit_vectors(x))['P']
    ms = (ps + np.eye(48)) / 2
    eigs = st.site_eigs(ms)
    c = (1 - eigs[0][:, 1:9]).mean(axis=1)
    return ps, ms, c, st.balance_direct(ms, c, eigs=eigs), label


def test_planted_clusters_at_every_layer():
    _, ms, c, t, label = planted(17)
    vals, f = st.modes(ms, c, t, k=1, tol=1e-10)
    assert np.all(np.sign(f[:, :, 0]) == np.sign(f[0, 0, 0]) * -label[None, :])
    assert st.residual(ms, c, t, vals, f).max() < 1e-7
    assert abs(vals[0] - st.temporal_eigs(17, t)[1]) < 1e-10


def test_joint_algebra_and_placement_residual():
    rng = np.random.default_rng(61)
    p = np.stack([dm.radius_kernel(dm.unit_vectors(rng.standard_normal((32, 8))))['P'] for _ in range(4)])
    ms = (p + np.eye(32)) / 2
    eigs = st.site_eigs(ms)
    c = (1 - eigs[0][:, 1:9]).mean(axis=1)
    t = st.balance_direct(ms, c, eigs=eigs)
    for coupling in (t, t / 16):
        vals, v = st.modes(ms, c, coupling, k=7, tol=1e-10)
        jm = dict(vecs=v, vals=vals, c=c, t=coupling)
        upstream, condition = dm.joint_forward(p, jm)
        cfg = depth.extension_config(jm)
        actual = depth.extend(p, jm, cfg)
        assert not cfg['withheld'].any()
        np.testing.assert_allclose(cfg['condition'], condition, rtol=1e-12)
        np.testing.assert_allclose(actual, upstream, atol=1e-12, rtol=1e-12)
        assert np.linalg.norm(actual - v) / np.linalg.norm(v) < 1e-8
        assert cfg['max_relative_residual'].max() < 1e-12
        assert np.all(depth.extend(np.zeros_like(p), jm, cfg) == 0)


def test_overlap_rank_and_per_layer_centering():
    # Neither layer alone supports rank two; the separately centred stack does.
    a = np.array([[[-1, 0], [0, 0], [1, 0]], [[0, -1], [0, 0], [0, 1]]], dtype=float)
    rot = np.array([[0.6, -0.8], [0.8, 0.6]])
    b = a @ rot + np.array([[[22, -9]], [[10, 42]]])
    ov = depth.overlap(a, b, 2)
    assert [(x['rank_a'], x['rank_b'], x['overlap']) for x in ov['layers']] == [(1, 1, None)] * 2
    assert ov['stacked']['rank_a'] == ov['stacked']['rank_b'] == 2
    assert ov['stacked']['overlap'] == pytest.approx(1)
    b[:, :, 1] = b[:, :, 0]
    ov = depth.overlap(a, b, 2)['stacked']
    assert (ov['rank_a'], ov['rank_b'], ov['overlap']) == (2, 1, None)


def test_band_boundaries_and_guard_state():
    vals = np.arange(1., 73.)
    vals[8] = vals[7] + .05
    vals[-1] = vals[-2] + .5
    bands, cuts = depth.bands(vals)
    assert cuts == {'8': True, '16': False, '64': False}
    assert bands == [dict(first=8, last=9, unresolved=False, complete_in_release=True),
                     dict(first=71, last=72, unresolved=True, complete_in_release=False)]


def probe_fixture():
    return dict(contexts=[dict(token_ids=[(i + c) % 199 for i in range(2047)], subset='fixture') for c in range(32)])


def test_manifest_determinism_and_cohorts():
    probe = probe_fixture()
    bare = depth.make_manifest(probe)
    assert depth.json_bytes(bare) == depth.json_bytes(depth.make_manifest(probe))
    assert [len(p) for p in bare['panels'].values()] == [1024, 768, 1024]
    expectations = {'doc00': (32, 0, 992), 'doc01': (32, 0, 992), 'doc02': (32, 0, 992), 'doc04': (32, 0, 992),
                    'pairA': (0, 64, 960), 'pairB': (0, 64, 960), 'd04': (0, 128, 896), 'd08': (0, 256, 768),
                    'd16': (0, 512, 512), 'd32': (0, 1024, 0), 'rand': (0, 0, 1024)}
    for name, expected in expectations.items():
        counts = Counter(bare['cohorts'][name]['natural'])
        assert tuple(counts[k] for k in depth.COHORTS) == expected
        assert set(bare['cohorts'][name]['shuffled']) == {'unseen_sequence'}
        build = bare['sets'][name]
        assert len(set(map(depth.row_key, build))) == len(build)
        assert all(1 <= r[1] <= 2047 for r in build)
    counts = Counter(bare['cohorts']['rand']['random'])
    assert tuple(counts[k] for k in depth.COHORTS) == (0, 512, 256)
    assert all(r[2] is None for r in bare['panels']['random'])
    assert [r[1] for r in bare['paths']['c00']] == list(range(1, 2048))

    sources = {f'r{i}': {'ids': [4] * 2047} for i in range(5)}
    for i, c in enumerate(depth.SOURCES):
        perm = np.random.default_rng(20261127 + i).permutation(2047).tolist()
        sources[f's{i}'] = dict(source_context=c, ids=[probe['contexts'][c]['token_ids'][p] for p in perm],
                                perm=[p + 1 for p in perm])
    synth = dict(contexts=sources, seed_random=20261126, seed_shuffle=20261127, allowed_ids_sha256='fixture', sha256={})
    full = depth.make_manifest(probe, synth)
    assert full['cohorts'] == bare['cohorts']
    for kind in full['panels']:
        assert list(map(depth.row_key, full['panels'][kind])) == list(map(depth.row_key, bare['panels'][kind]))
    assert full['contexts']['s3']['perm'] == sources['s3']['perm']
    assert full['contexts']['s3']['source_context'] == 4
    first = full['panels']['random'][0]
    assert first[0] == 'r0' and first[2] == 4


def test_document_work_plans_stay_live_without_synthetic_data(monkeypatch, tmp_path):
    m = depth.make_manifest(probe_fixture())
    present = tmp_path / 'present.npy'
    present.touch()
    monkeypatch.setattr(depth, 'act_path', lambda model, ctx: present if isinstance(ctx, int) else tmp_path / ctx)
    for name in depth.SETS:
        p = depth.unit_plan(m, 'pythia', name)
        if name == 'rand':
            assert p is None
            continue
        own, panels, paths, missing = p
        assert own == m['sets'][name]
        assert list(panels) == ['natural']
        assert missing['panels'] == ['random', 'shuffled']
        if name == 'd32':
            assert set(paths) == {'c00', 'c01', 'c02', 'c04'}
            assert missing['paths'] == ['s0', 's1', 's2', 's3']
        else:
            assert paths == {} and not missing['paths']
    assert len(depth.unit_plan(m, 'olmo', 'd08', refit=True)[0]) == 2304


def test_permutations_keep_whole_occurrences_in_their_group():
    ids = [1, 1, 2, 1, 1, 2, 1]
    cohorts = ['copy', 'copy', 'copy', 'place', 'place', 'place', 'place']
    groups = depth.permutation_groups(ids, cohorts)
    draws = depth.permutations(np.random.default_rng(20261129), len(ids), groups)
    assert len(draws['within_id_and_cohort']) == len(draws['free']) == 20
    for p in draws['within_id_and_cohort']:
        assert [ids[i] for i in p] == ids
        assert [cohorts[i] for i in p] == cohorts
        assert set(p.tolist()) == set(range(len(ids)))
    rng = np.random.default_rng(62)
    a, b = rng.normal(size=(2, len(ids), 3)), rng.normal(size=(2, len(ids), 3))
    af, bf = depth.overlap_frame(a, 3), depth.overlap_frame(b, 3)
    for p in draws['free'][:3]:
        cached = depth.frame_scores(af, bf, 3, perm=p)
        direct = depth.overlap(a, b[:, p], 3)
        assert cached['stacked']['overlap'] == pytest.approx(direct['stacked']['overlap'], abs=1e-13)


def test_placement_numerical_denominator_and_offset():
    own = np.ones((3, 17, 64))
    own[2] *= 1e-7
    placed = np.broadcast_to(np.array([1., -1., 0.1])[:, None, None], (3, 5, 64)).copy()
    near = np.full((3, 5), .78)
    skip = np.zeros(64, dtype=bool)
    skip[60] = True
    out = depth.placement_measures(placed, own, near, np.arange(5), skip, {'8': False, '64': True})
    np.testing.assert_allclose(out['8']['centre_offset'][:2], 1)
    assert np.isnan(out['8']['centre_offset'][2])
    assert out['8']['pooled_centre_offset'] is None
    assert out['64'] == dict(cuts_band=True, unavailable='withheld_mode')
    np.testing.assert_allclose(out['nearest_cosine_median'], .78)
