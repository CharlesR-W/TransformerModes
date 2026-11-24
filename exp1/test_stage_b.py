"""Stage B fixtures, also callable by gate_b without pytest installed."""
from functools import wraps
from itertools import permutations, product

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_raises
from scipy import linalg

import gate_b as gate
import kernels
import spacetime
import stage_b as b


def fixture(measure):
    """Expose numerical evidence to the CLI without returning it to pytest."""
    @wraps(measure)
    def check():
        measure()
    return check


def complete(n):
    return (np.ones((n, n)) - np.eye(n)) / (n-1)


def random_walk(rng, n):
    w = rng.lognormal(0, .65, (n, n))
    w = (w+w.T)*.5
    np.fill_diagonal(w, 0.)
    v, err = kernels.sinkhorn_sym(w, tol=1e-14)
    assert err < 1e-14
    return v[:, None] * w * v


@fixture
def test_chunk_reversal_and_explicit_maps():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(15, 2, 11, 24))
    out = b.chunk_distances(x, block=3)
    maximum = 0.
    for l, site in enumerate(out):
        y = x[:, l, ::-1] - x[:, l].mean((0, 1))
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        for name, w in b.lag_weights(y.shape[1]).items():
            g = np.einsum('st,isd,jtd->ij', w, y, y)
            want = 1 - g / np.sqrt(np.outer(g.diagonal(), g.diagonal()))
            assert_allclose(site[name], want, atol=1e-12, rtol=0)
            maximum = max(maximum, np.max(abs(site[name]-want)))
    eye = np.eye(64)
    # Negative pairs force exactly zero global mean, with no centring shortcuts.
    reverse = np.stack([eye, eye[::-1], -eye, -eye[::-1]])
    d = b.chunk_distances(reverse, check_duplicates=False)
    assert_allclose([d['cut'][0, 1], d['recent'][0, 1], d['average'][0, 1]], [1, 1, 0], atol=1e-12)
    with assert_raises(ValueError):
        b.chunk_distances(reverse)
    pooled_zero = np.stack([eye[:2] - eye[1::-1], eye[2:4] - eye[3:1:-1]])
    with assert_raises(ValueError):
        b.chunk_distances(pooled_zero, views=('average',))
    with assert_raises(ValueError):
        b.chunk_distances(np.array([[[0., 0.]], [[1., 0.]], [[-1., 0.]]]), views=('cut',))
    return dict(max_explicit_error=maximum, reversal={k: v[0, 1] for k, v in d.items()})


@fixture
def test_layer0_and_wide_indices():
    rng = np.random.default_rng(10)
    emb = rng.normal(size=(51000, 4)).astype(np.float32)
    ids = np.array([[12, 50_303], [22, 50_303], [50_302, 23], [43_000, 24]], dtype=np.int32)
    # A flattened (row, token) int32 key would overflow for the first row.
    assert int(ids[0, 1]) * len(emb) > np.iinfo(np.int32).max
    out = b.layer0_distances(ids, emb, views=('cut',))['cut']
    ref = b.chunk_distances(emb[ids.astype(np.int64)], views=('cut',), check_duplicates=False)['cut']
    assert_allclose(out, ref, atol=1e-12, rtol=0)
    assert out[0, 1] < 1e-12
    with assert_raises(ValueError):
        b.chunk_distances(emb[ids], views=('cut',))
    labels = np.array([2, 2, 2**33+2, 2**33+2, 2**34, 2**35], dtype=np.int64)
    ps, meta = b.relabellings(len(labels), 199, rng, groups=labels, subsets=[0, 1, 1, 1, 0, 1])
    assert ps.dtype == np.int64
    assert_array_equal(labels[ps], np.broadcast_to(labels, ps.shape))
    assert meta['group_size_distribution'] == {1: 2, 2: 2}
    assert meta['expected_moved_fraction'] == 2/6 and meta['coverage'] == 4/6
    assert .25 < meta['moved_fraction'] < .4 and len(meta['sha256']) == 199
    return meta


@fixture
def test_blocked_hellinger():
    rng = np.random.default_rng(13)
    p = rng.uniform(.01, 1., (200, 5000))
    p /= p.sum(1, keepdims=True)
    p[1] = p[0]
    p[2] = p[0] + 1e-8 * (p[2] - p[0])
    d = b.hellinger_distances(p, row_block=17, pair_block=2)
    r = np.sqrt(p)
    direct = np.zeros_like(d)
    for i in range(len(p)):
        for j in range(i):
            q = r[i] - r[j]
            direct[i, j] = direct[j, i] = q @ q / 2
    assert_allclose(d, direct, rtol=0, atol=1e-13)
    assert d[0, 1] == 0 and d[0, 2] > 0
    assert_allclose(d[0, 2], direct[0, 2], rtol=1e-13, atol=0)
    return dict(max_error=np.max(abs(d-direct)), near_pair=d[0, 2], exact_pair=d[0, 1])


@fixture
def test_participation_tie_and_oracle():
    rng = np.random.default_rng(24)
    d = rng.uniform(.1, 1., (120, 120))
    for i, k in enumerate((63, 64, 100, 64, 1)):
        other = np.flatnonzero(np.arange(len(d)) != i)
        d[i, other] = 1.
        d[i, other[:k]] = 0.
        if i == 3:
            d[i, other[1:k]] = 1e-13
        if i == 4:
            d[i, other[1:65]] = 1e-5
            d[i, other[65:]] = 1e3
    rows = b.pr_rows(d, block=11)
    oracle = np.exp(kernels._cond_rows(d, 64, True, renyi=2)[-1])
    ordinary = ~rows.tie_rows
    assert_allclose(rows.q[ordinary], oracle[ordinary], atol=1e-12, rtol=0)
    assert_array_equal(rows.tie_rows[:5], [False, True, True, True, False])
    assert_array_equal(rows.unattainable[:5], [False, False, True, False, False])
    assert_array_equal(rows.support[:5], [119, 64, 100, 64, 65])
    assert_allclose(rows.participation[:5], [64, 64, 100, 64, 64], atol=1e-11, rtol=0)
    assert rows.underflow[4] == 54
    p = rng.permutation(len(d))
    reordered = b.pr_rows(d[np.ix_(p, p)])
    assert_allclose(reordered.q, rows.q[np.ix_(p, p)], rtol=0, atol=1e-14)
    too_small = b.build_walk(np.ones((10, 10)) - np.eye(10))
    assert not too_small.report['primary_admissible'] and len(too_small.report['unattainable_rows']) == 10
    return dict(support=rows.support[:5], participation=rows.participation[:5], underflow=rows.underflow[:5],
                maximum_oracle_difference=np.max(np.abs(rows.q[ordinary]-oracle[ordinary])))


@fixture
def test_graph_disconnected_and_nearcut():
    n = 132
    labels = np.arange(n)//66
    w = b.build_walk((labels[:, None] != labels).astype(float))
    r = w.report
    assert r['tie_rows'] == n and r['components'] == 2
    assert_array_equal(r['component_sizes'], [66, 66])
    assert_array_equal(r['conditional_support'], 65*np.ones(n))
    assert r['row_residual'] < 1e-11 and abs(r['first_gap']) < 1e-12
    assert not r['primary_admissible'] and not r['through_depth_admissible']
    assert_allclose(r['conditional_tail'], 1/65, atol=1e-12)
    assert_allclose(r['final_tail'], r['conditional_tail'], atol=1e-12)
    with assert_raises(b.Withheld):
        b.prepare_partner([w]*2)
    near = []
    for groups in (2, 9):
        cl = linalg.block_diag(*([complete(10)]*groups))
        s = (1-1e-10)*cl + 1e-10*complete(len(cl))
        gaps = 1 - b.mz_eigh(s, 8, vectors=False)
        assert np.sum(gaps < b.GAP) == min(8, groups-1)
        with assert_raises(b.Withheld):
            b.prepare_partner([s]*2)
        near.append(gaps)
    return dict(disconnected=dict(components=r['components'], gap=r['first_gap'], residual=r['row_residual']), nearcut=near)


@fixture
def test_product_p1_p6_and_p9():
    rng = np.random.default_rng(71_003)  # not a Mess3 seed
    n, nl, trials = 23, 4, 3
    maximum = np.zeros(5)
    for _ in range(trials):
        layers = [random_walk(rng, n) for _ in range(nl)]
        s = random_walk(rng, n)
        past, product = [], []
        for sl in layers:
            a = b.lazy_walk(sl)
            m, m0 = b.product_walk(a, s), b.product_walk(a)
            past.append(m0)
            product.append(m)
            vals = linalg.eigvalsh(m)
            assert_allclose(m, m.T, atol=1e-14, rtol=0)
            assert_allclose(m.sum(1), 1, atol=1e-13, rtol=0)
            assert m.min() >= 0 and vals[0] >= -1-1e-13 and vals[-1] <= 1+1e-13
            err = np.max(np.abs(m0-m-a @ (np.eye(n)-s) @ a))
            assert err < 1e-13
            maximum[0] = max(err, maximum[0])
            assert np.max(vals-linalg.eigvalsh(m0)) < 1e-13
            wl = linalg.eigvalsh(s)
            assert np.max((vals-wl)[wl >= 0]) < 1e-13
            assert b.mz_eigh(m, 1, vectors=False)[0] < 1-1e-8
            assert_allclose(b.product_walk(a, np.eye(n)), m0, atol=1e-13, rtol=0)
            comp = b.product_walk(a, complete(n))
            h = linalg.helmert(n).T
            assert_allclose(h.T @ comp @ h, -h.T @ m0 @ h/(n-1), atol=1e-13, rtol=0)
            mu, v = linalg.eigh(sl)
            assert_allclose(b.product_walk(a, sl) @ v, v * ((1+mu)/2)**2 * mu, atol=1e-13, rtol=0)
        c, t = b.site_units(np.array(past))
        w0, _ = b._depth_dense(np.stack(past), c, t)
        w, _ = b._depth_dense(np.stack(product), c, t)
        assert np.min(w-w0) > -1e-12 and (w[:16]-w0[:16]).sum() >= -1e-12
        maximum[1] = max(maximum[1], np.max(w0-w))
        maximum[2] = max(maximum[2], (w[:16]-w0[:16]).sum())
        u = np.linalg.qr(spacetime.centre(rng.normal(size=(1, n, 4)))[0])[0]
        v = np.linalg.qr(spacetime.centre(rng.normal(size=(1, n, 6)))[0])[0]
        uv = linalg.svdvals(u.T @ v)**2
        z = linalg.eigvalsh(u.T @ v @ v.T @ u)[::-1]
        assert_allclose(z, uv, atol=1e-13, rtol=0)
        maximum[3] = max(maximum[3], np.max(abs(z-uv)))
    # P4: aligned partitions share a nonconstant eigenvalue 1; crossing
    # partitions connect their union and do not share such a direction.
    a = b.lazy_walk(linalg.block_diag(complete(6), complete(6)))
    s = linalg.block_diag(complete(6), complete(6))
    p = np.r_[np.arange(0, 12, 2), np.arange(1, 12, 2)]
    same = b.mz_eigh(b.product_walk(a, s), 1, vectors=False)[0]
    crossed = b.mz_eigh(b.product_walk(a, s, p), 1, vectors=False)[0]
    assert_allclose(same, 1, atol=1e-13, rtol=0)
    assert crossed < .99
    return dict(p2_error=maximum[0], largest_p9_order_violation=maximum[1], largest_energy=maximum[2],
                p6_error=maximum[3], p4_same=same, p4_crossed=crossed)


@fixture
def test_p10_token_function_and_equivariance():
    rng = np.random.default_rng(76)
    n = 32
    labels = np.arange(n)//4
    d = np.abs(labels[:, None]-labels).astype(float) + .3
    np.fill_diagonal(d, 0)
    w = b.build_walk(d, target=8)
    ps, _ = b.relabellings(n, 4, rng, groups=labels)
    s = random_walk(rng, n)
    pt = b.prepare_partner([w]*2)
    obs = b.solve_product(pt, s)
    es, pe = [], []
    for p in ps:
        wp = b.build_walk(d[np.ix_(p, p)], target=8)
        assert_allclose(wp.s, w.s[np.ix_(p, p)], atol=1e-12, rtol=0)
        m = b.product_walk(pt.lazy[0], s)
        mp = b.product_walk(pt.lazy[0], s, p)
        assert_allclose(mp, m[np.ix_(p, p)], atol=1e-12, rtol=0)
        out = b.solve_product(pt, s, p)
        diff = abs(b.energy(pt.past, obs)-b.energy(pt.past, out))
        assert diff < 1e-10
        es.append(diff)
        # The theorem only needs an isolated boundary, not the stricter
        # 1% graphical reporting convention. Compare the top two projector.
        idx = 2
        assert b.cutoff_gap(obs.values, idx) > 1e-6
        g = obs.vectors[:, p, :idx].reshape(2*n, idx)
        h = out.vectors[:, :, :idx].reshape(2*n, idx)
        err = np.max(abs(g @ g.T-h @ h.T))
        assert err < 1e-8
        pe.append(err)
    return dict(energy_error=max(es), projector_error=max(pe))


@fixture
def test_flat_k6_unresolved():
    r = gate.k6_flat(diagnose_reused=True)
    assert r['passed'], r
    assert r['past_lowest_multiplicity'] == r['n']-1
    assert r['past']['picture_k'] is None and r['past']['picture_status'] == 'unresolved'
    assert r['delta']['display'] == r['delta']['reason'] == 'unresolved'
    assert r['energy'] > 0
    return r


@fixture
def test_complete_prediction_observed_and_null():
    rng = np.random.default_rng(7)
    n = 70
    expected = complete(n)
    w = b.build_walk(b.hellinger_distances(np.broadcast_to([.2, .1, .7], (n, 3))))
    assert_allclose(w.s, expected, atol=1e-13, rtol=0)
    assert w.report['tie_rows'] == n and w.report['primary_admissible']
    ps, _ = b.relabellings(n, 6, rng)
    partners = [b.prepare_partner([random_walk(rng, n) for _ in range(2)]) for _ in range(2)]
    counts, es, control_pairs = [], [], []
    obs = tuple(b.solve_product(pt, w, own_units=True) for pt in partners)
    for p in ps:
        control_pairs.append(tuple(b.solve_product(pt, w, p) for pt in partners))
    for i, pt in enumerate(partners):
        for a in pt.lazy:
            count = b.site_count(a, w.s, ps)
            assert count['count'] == 0 and count['null_maximum'] < 0
            counts.append(count['count'])
        e = b.energy(pt.past, obs[i])
        null = [b.energy(pt.past, pair[i]) for pair in control_pairs]
        p = b.energy_reference(e, null)
        assert e > 0 and p['p'] == 1
        assert_allclose(null, e, atol=b.energy_margin(e), rtol=0)
        es.append(p)
    delta = b.increment_reference(tuple(pt.past for pt in partners), obs, control_pairs)
    assert delta['status'] == 'withheld' or delta['p'] == 1
    return dict(counts=counts, energies=es, delta=delta)


@fixture
def test_negative_spectrum_and_matrix_free():
    rng = np.random.default_rng(6)
    n = 37
    a = b.lazy_walk(random_walk(rng, n))
    m = b.product_walk(a, complete(n))
    assert np.linalg.eigvalsh(m)[-2] < 0.
    want = 1 - np.linalg.eigvalsh(m)[-9:-1].mean()
    got = b.site_scales_mz([m])[0]
    assert_allclose(got, want, rtol=0, atol=1e-13)
    assert abs(got-spacetime.site_scales([m], 'own')[0]) > 1e-5
    p = rng.permutation(n)
    res = []
    for sf in [complete(n), random_walk(rng, n)]:
        w, err = b.null_maximum(a, sf, p)
        dense = np.linalg.eigvalsh(b.product_walk(a, sf, p) - 3/n)[-1]
        assert_allclose(w, dense, rtol=0, atol=1e-10)
        res.append(dict(null=w, dense=dense, residual=err))
    assert res[0]['null'] < 0
    return dict(scale=got, prior_scale=spacetime.site_scales([m], 'own')[0], nulls=res)


@fixture
def test_balance_handoff_nonflat_unique():
    rng = np.random.default_rng(15)
    n, nl = 23, 3
    layers = [random_walk(rng, n) for _ in range(nl)]
    partner = b.prepare_partner(layers, modes=20)
    ms = np.stack([b.product_walk(a) for a in partner.lazy])
    assert partner.past.report['accepted']
    for m in ms:
        w = b.mz_eigh(m, 2, vectors=False)
        assert w[0]-w[1] > 1e-4
    eigenvalues, _ = b._depth_dense(ms, partner.c, partner.t)
    target = spacetime.temporal_eigs(nl, partner.t)[1]
    assert_allclose(eigenvalues[0], target, atol=1e-10, rtol=0)
    assert_allclose(partner.past.values, eigenvalues[:20], atol=1e-10, rtol=0)
    bad = [linalg.eigh(m-3/n) for m in ms]
    wrong = spacetime.balance_direct(ms, partner.c,
            (np.stack([w[::-1] for w, _ in bad]), np.stack([u[:, ::-1] for _, u in bad])))
    wrong_leading, _ = b._depth_dense(ms, partner.c, wrong)
    mismatch = abs(wrong_leading[0] / spacetime.temporal_eigs(nl, wrong)[1] - 1)
    assert mismatch > 1e-3, 'fixture must detect the sorted-shifted hand-off bug'
    return dict(c=partner.c, correct_t=partner.t, wrong_t=wrong, wrong_relative_mismatch=mismatch)


@fixture
def test_exhaustive_calibration():
    rng = np.random.default_rng(130)
    n, nl, k, r = 6, 2, 2, 2
    pairing = linalg.block_diag(*[complete(2)]*3)
    # A fixed block pattern within each model isolates the boundary. Each
    # layer and prediction walk has its own independent random perturbation.
    patterns = (np.array([0, 3, 1, 4, 2, 5]), np.array([0, 4, 1, 2, 3, 5]))
    layers = [[.80*pairing[np.ix_(p, p)] + .20*random_walk(rng, n) for _ in range(nl)] for p in patterns]
    pt = [b.prepare_partner(model, r=r, k=k, modes=10, dense=True) for model in layers]
    # Independent random stochastic walks with a strong holding probability
    # (toy only), to guarantee an isolated 2/3 boundary under all pairings.
    sf = [.65*np.eye(n) + .35*random_walk(rng, n) for _ in range(2)]
    ps = list(map(np.asarray, permutations(range(n))))
    leading, energy, delta, gaps = [], [], [], []
    past = tuple(p.past for p in pt)
    for p in ps:
        runs = tuple(b.solve_product(q, s, p) for q, s in zip(pt, sf, strict=True))
        gaps.extend(run.report['cutoff_gap'] for run in runs)
        leading.append(b.mz_eigh(b.product_walk(pt[0].lazy[0], sf[0], p), 1, vectors=False)[0])
        energy.append(b.energy(pt[0].past, runs[0], k))
        # Only extracting Δ here; the exact orbit becomes the reference below.
        inc = b.increment_reference(past, runs, [runs], k=k)
        assert inc['status'] == 'computed', (p, gaps[-2:])
        delta.append(inc['value'])
    # One unresolved null boundary withholds the entire reference.
    tied_values = runs[1].values.copy()
    tied_values[k] = tied_values[k-1]
    tied = b.Depth(tied_values, runs[1].vectors, runs[1].report)
    withheld = b.increment_reference(past, runs, [runs, (runs[0], tied)], k=k)
    assert withheld['status'] == 'withheld' and withheld['reason'] == 'unresolved'
    pairs = [(np.array(leading), b.leading_rank), (np.array(energy), b.energy_rank), (np.array(delta), b.increment_rank)]
    summary = []
    for values, rank_fn in pairs:
        rks = []
        for i, x in enumerate(values):
            others = np.delete(values, i)
            ranking = rank_fn(x, others)
            rks.append(ranking['rank'])
            assert ranking['p'] == ranking['rank']/720
            if rank_fn is b.leading_rank:
                count = b.count_from_spectra([x], others, np.zeros(len(others)))
                assert bool(count['count']) == (ranking['rank'] == 1)
        rks = np.array(rks)
        levels = np.arange(1, 721)
        cdf = np.searchsorted(np.sort(rks), levels, side='right')
        assert np.all(cdf <= levels)
        summary.append(dict(ranks=rks, maximum_cdf_excess=np.max(cdf-levels)))
    assert b.energy_rank(2., [2.+1e-8])['p'] == 1
    assert b.increment_rank(.1, [.1-5e-9])['p'] == 1
    assert b.leading_rank(-.1, [-.1-5e-9])['rank'] == 2
    return dict(n=n, sites=nl, rank=r, k=k, permutations=len(ps), minimum_boundary_gap=min(gaps),
                leading=summary[0], energy=summary[1], increment=summary[2])


@fixture
def test_pre_b_analytic_identity_prediction():
    n = 48
    labels = np.arange(2*n)//n
    x = np.c_[np.eye(2*n), np.eye(2)[labels]]
    d = b.chunk_distances(x[:, None], views=('cut',))['cut']
    graph = b.build_walk(d)
    # Solve (n-1)a + nb = 1, (n-1)a² + nb² = 1/64.
    a = 1/(2*n-1) + np.sqrt((1/64 - 1/(2*n-1))*n/((n-1)*(2*n-1)))
    far = (1-(n-1)*a)/n
    mu = (n-1)*a - n*far
    assert_allclose(graph.s[0, [1, n]], [a, far], rtol=0, atol=1e-12)
    walk = b.product_walk(b.lazy_walk(graph.s), np.eye(2*n))
    v = np.where(labels == 0, 1., -1.)/np.sqrt(2*n)
    answers = []
    for nl in (4, 8):
        ms = np.stack([walk]*nl)
        c, t = b.site_units(ms)
        modes = b.solve_depth(ms, c, t, past=True)
        assert modes.report['accepted']
        h = (1-((1+mu)/2)**2)/c[0]
        f = np.broadcast_to(v/np.sqrt(nl), (nl, 2*n))
        assert_allclose(abs(np.sum(modes.vectors[:, :, 0] * f)), 1, atol=1e-10, rtol=0)
        g = np.sqrt(2/nl)*np.cos(np.pi*(np.arange(nl)+.5)/nl)[:, None]*v
        gh = h+spacetime.temporal_eigs(nl, t)[1]
        assert_allclose(modes.values[0], h, atol=1e-10, rtol=0)
        assert_allclose(spacetime.apply_L(ms, c, t, g[:, :, None])[:, :, 0], gh*g, atol=1e-10, rtol=0)
        answers.append(dict(sites=nl, ground=modes.values[0], expected=h, harmonic=gh))
    return dict(within=a, across=far, answers=answers)


@fixture
def test_forward_filter_and_primary_decision():
    symbols = np.array([[0, 1, 0], [2, 1, 1]], dtype=np.int64)
    f = gate.forward_filter(symbols)
    assert_allclose(f[0, 0], [.85, .075, .075], atol=1e-13, rtol=0)
    for i, row in enumerate(symbols):
        # Enumerate histories ending in each hidden state independently of filter.
        for j in range(len(row)):
            truth = np.zeros(3)
            for history in product(range(3), repeat=j+1):
                pr = gate.EMISSION[history[0], row[0]] / 3
                for s in range(1, len(history)):
                    pr *= gate.T[history[s-1], history[s]]*gate.EMISSION[history[s], row[s]]
                truth[history[-1]] += pr
            assert_allclose(f[i, j], truth/truth.sum(), atol=1e-13, rtol=0)
    assert_allclose(gate.next_symbol(f).sum(1), 1, atol=1e-13, rtol=0)
    sig = b.energy_reference(1., [2.]*199)
    tied = b.energy_reference(1., [1.]*199)
    withheld = b.energy_reference(1., [None]+[2.]*198)
    assert 'cheaper' in b.primary_decision([sig, sig], .3)
    assert b.primary_decision([sig, tied], .3) == 'not shown'
    assert b.primary_decision([sig, withheld], .3) == 'withheld'
    assert b.primary_decision([sig, sig], .299) == 'uninformative token reference'
    assert_allclose(b.alignment(np.eye(3)[:, :2], np.eye(3)[:, 1:]), .5, atol=1e-13, rtol=0)
    return dict(initial=f[:, 0], next_symbol=gate.next_symbol(f), best_p=sig['p'], tie_p=tied['p'])


@fixture
def test_readout_hand_cases():
    x = linalg.helmert(4).T[:, :2]
    share = b.attribute_share(x, np.c_[np.ones(4), x[:, 0]*7, x[:, 0]*2])
    assert_allclose([share['share'], share['chance']], [.5, 1/3], atol=1e-13, rtol=0)
    s = np.array([[0., 1.], [1., 0.]])
    p = np.array([[.9, .05, .05], [.05, .05, .9]])
    out = b.hellinger_token_mass(s, p, vocab_block=1)
    direct = (np.sqrt(p[0])-np.sqrt(p[1]))**2
    assert_allclose(out['mass'], direct, atol=1e-13, rtol=0)
    assert_allclose(out['top1_share'], 1, atol=1e-13, rtol=0)
    assert b.hellinger_token_mass(s, p[[0, 0]])['shares'] is None
    return dict(attribute=share, token=out)


@fixture
def test_scrambled_k3_requires_both_references():
    def cells():
        return [dict(references={kind: dict(gain=.05, count=dict(count=0))
                                 for kind in ('random', 'symbol')}) for _ in gate.ROLES]

    true, unrelated = cells(), cells()
    assert gate._conditions(true, unrelated)['K3']
    refs = true[gate.SCRAMBLED]['references']
    for kind in refs:
        refs[kind]['gain'] = np.nextafter(.05, 1.)
        assert not gate._conditions(true, unrelated)['K3']
        refs[kind]['gain'] = .05
    refs['random']['count']['count'] = 2
    assert not gate._conditions(true, unrelated)['K3']


@fixture
def test_frozen_fixture_streams_and_distractor_scaling():
    p = gate.DEFAULT_PARAMETERS
    assert gate.DEVELOPMENT_SEEDS == (1170, 1171, 1172, 1173, 1174)
    assert p.sites == (('absent', 0, 0), ('absent', 0, 0), ('symbol-only', 0, 1),
                       ('present', 1, 1), ('present', 1, 1), ('present', 1, 1),
                       ('scrambled', 1, 1), ('symbol-only', 0, 1))
    dev = gate.generator_seeds(gate.DEVELOPMENT_SEEDS, p)
    confirm = gate.generator_seeds(gate.FROZEN_SEEDS, p)
    all_dev, all_confirm = set().union(*dev.values()), set().union(*confirm.values())
    assert all_dev & all_confirm == {31, 33}
    assert (dev['sequences'] | dev['distractor_noise']).isdisjoint(all_confirm | {51, 52} |
                                                                    {s+o for s in gate.FROZEN_SEEDS for o in (80, 90, 95)})
    assert dev['loadings'] == confirm['loadings'] == {31}
    assert dev['scramble'] == confirm['scramble'] == {33}

    # No protected seed is sampled. Independently reconstruct all eight
    # sites on a few records and verify that only the recipient g changes.
    n, seed = 7, 444
    symbols, beliefs = gate.source(n, seed, p)
    p9 = gate.replace(p, v_g=9.)
    assert_array_equal(gate.source(n, seed, p9)[0], symbols)
    draws = np.random.default_rng(544)
    rotations = np.random.default_rng(31)
    donor = np.random.default_rng(33).permutation(n)

    def scaled(x, variance):
        x = x - x.reshape(-1, x.shape[-1]).mean(0)
        return x / np.linalg.norm(x) * np.sqrt(variance * np.prod(x.shape[:-1]))

    g = scaled(draws.standard_normal((n, 4)), 6)
    c = scaled(beliefs[:, -64:], 2)
    e = scaled(np.eye(3)[symbols[:, -64:]], 1)
    for (role, beta, eta), x6, x9 in zip(p.sites, gate.synthetic_layers(symbols, beliefs, seed, p),
                                        gate.synthetic_layers(symbols, beliefs, seed, p9), strict=True):
        q, r = np.linalg.qr(rotations.standard_normal((64, 10)))
        q *= np.sign(r.diagonal())
        noise = .07 * draws.standard_normal((n, 64, 64))
        distractor = (g @ q[:, 3:7].T)[:, None, :]
        cb, es = (c[donor], e[donor]) if role == 'scrambled' else (c, e)
        want = noise + distractor + beta*(cb @ q[:, :3].T) + eta*(es @ q[:, 7:].T)
        assert_allclose(x6, want, atol=1e-13, rtol=0)
        assert_allclose(x9-x6, np.broadcast_to((np.sqrt(9/6)-1)*distractor, x6.shape), atol=1e-13, rtol=0)
    return dict(development={k: sorted(v) for k, v in dev.items()}, confirmatory={k: sorted(v) for k, v in confirm.items()})


@fixture
def test_calibration_uses_only_prediction_and_present_graphs():
    from contextlib import ExitStack
    from unittest.mock import patch

    p = gate.replace(gate.DEFAULT_PARAMETERS, n=68)
    with ExitStack() as stack:
        for module, names in ((gate, ('run_fixtures', 'run_k0', 'remaining', '_conditions', 'site_control', 'k6_flat')),
                              (b, ('product_walk', 'relabellings', 'site_count', 'null_maximum', 'count_from_spectra',
                                   'solve_depth', 'prepare_partner', 'solve_product', 'energy', 'energy_reference'))):
            for name in names:
                stack.enter_context(patch.object(module, name, side_effect=AssertionError(f'calibration called {name}')))
        graph = stack.enter_context(patch.object(b, 'build_walk', wraps=b.build_walk))
        pred = gate._calibration_prediction(446, p)
        assert graph.call_count == 1
        past = gate._calibration_past(446, p)
        assert graph.call_count == 4
    assert pred['graph']['target'] == 64
    assert [s['site'] for s in past['sites']] == [3, 4, 5]


def test_calibration_order_stop_and_checksum(tmp_path):
    import json
    from unittest.mock import patch

    seeds = gate.DEVELOPMENT_SEEDS
    calls = []

    def prediction(seed, parameters):
        calls.append(('prediction', seed))
        return dict(prediction_only_capture=.54 if seed == seeds[-1] else .6)

    def past(seed, parameters):
        v = parameters.v_g
        calls.append((v, seed))
        peak = .271 if v == 6. else .27
        return dict(seed=seed, past_only_max=peak, sites=[dict(site=i, capture=peak) for i in gate.PRESENT])

    stopped = tmp_path / 'stopped'
    with (patch.object(gate, '_calibration_prediction', side_effect=prediction),
          patch.object(gate, '_calibration_past', side_effect=AssertionError('grid ran after prediction failed'))):
        assert gate.calibrate(stopped) == 2
    result = json.loads((stopped/'report.json').read_text())
    assert result['status'] == 'fixture uninformative at design' and result['grid'] == []
    assert not (stopped/'parameters.json').exists()
    assert calls == [('prediction', s) for s in seeds]

    chosen = tmp_path / 'chosen'
    calls.clear()
    with (patch.object(gate, '_calibration_prediction', return_value=dict(prediction_only_capture=.55)) as pred,
          patch.object(gate, '_calibration_past', side_effect=past)):
        assert gate.calibrate(chosen) == 0
        assert pred.call_count == 5
    assert calls == [(v, s) for v in (6., 9.) for s in seeds]
    result = json.loads((chosen/'report.json').read_text())
    assert result['status'] == 'parameters frozen' and result['chosen_v_g'] == 9.
    path = chosen/'parameters.json'
    params, digest = gate.read_frozen_parameters(path)
    assert params.v_g == 9. and digest == result['parameter_sha256']
    argv = ['gate_b.py', '--out', str(tmp_path/'confirm'), '--seeds', '70', '71', '72', '--parameters', str(path)]
    with patch('sys.argv', argv), patch.object(gate, 'run', return_value=0) as run:
        assert gate.main() == 0
        config = run.call_args.args[0]
        assert config.parameters.v_g == 9. and config.parameter_sha256 == digest
        path.write_text(path.read_text() + '\n')  # valid JSON, different bytes
        with assert_raises(SystemExit):
            gate.main()
        assert run.call_count == 1  # mismatch refused before any gate path

    exhausted = tmp_path/'exhausted'
    with (patch.object(gate, '_calibration_prediction', return_value=dict(prediction_only_capture=.56)),
          patch.object(gate, '_calibration_past', side_effect=lambda s, p: dict(seed=s, past_only_max=.271, sites=[])) as failed_past):
        assert gate.calibrate(exhausted) == 2
    result = json.loads((exhausted/'report.json').read_text())
    assert result['status'] == 'fixture uninformative at design' and not (exhausted/'parameters.json').exists()
    assert failed_past.call_count == 20
    assert [g['v_g'] for g in result['grid']] == list(gate.VG_GRID)


def test_checkpoint_order_stop_and_continuation(tmp_path):
    """Software-only control-flow test with fabricated K0 checkpoints.

    It does not sample or retry a Mess3 K0 record and tests no gate threshold.
    The positive case runs the real remaining pipeline on arbitrary walks.
    """
    import json
    from unittest.mock import patch

    n, seed = 68, 9
    cfg = gate.Config(n, (seed,), True, 2, 2)
    with (patch.object(gate, 'run_fixtures', return_value=True) as fixtures,
          patch.object(gate, 'run_k0', return_value=(dict(passed=False), {})) as k0,
          patch.object(gate, 'remaining', side_effect=AssertionError('ran after a failed K0'))):
        path = tmp_path / 'terminal'
        assert gate.run(cfg, path, 'all') == gate.run(cfg, path, 'all') == 2
        assert fixtures.call_count == 2 and k0.call_count == 1
        assert json.loads((path/'report.json').read_text())['status'] == 'fixture uninformative'

    rng = np.random.default_rng(784)
    report = dict(through_depth_admissible=True, reasons=[])
    fake_k0 = dict(passed=True, sites=[dict(graph=report)]*8, prediction_graph=report)
    walks = [random_walk(rng, n) for _ in range(8)]
    fake_arrays = dict(symbols=(np.arange(n) % 3)[:, None], beliefs=rng.dirichlet([3, 4, 6], n)[:, None, :],
                       sf=random_walk(rng, n), layer_walks=np.stack(walks),
                       past_u=np.stack([b.mz_eigh(b.product_walk(b.lazy_walk(w)), 16)[1] for w in walks]))
    with (patch.object(gate, 'run_fixtures', return_value=True),
          patch.object(gate, 'run_k0', return_value=(fake_k0, fake_arrays)) as k0):
        path = tmp_path / 'completed'
        assert gate.run(cfg, path, 'k0') == 0
        assert gate.run(cfg, path, 'all') == 1  # one seed cannot meet the confirmatory K5 requirement
        steps = json.loads((path/'report.json').read_text())['steps']
        assert steps[f'{seed}-K6']['passed']
        assert steps[f'{seed}-K1-K6']['K5']['unrelated']['status'] == 'computed'
        assert f'{seed}-true-depth-null-01' in steps and f'{seed}-unrelated-site-7' in steps
        assert k0.call_count == 1
