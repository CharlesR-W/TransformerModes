"""Numerical cross-checks. Run with float64 and bounded BLAS/OpenMP threads."""
import numpy as np
import pytest
import torch

from bench_numerics import ALPHA, dense_calibrate, dense_modes, mode_drift, synthetic_site
from graph import build_walk, conditional_weights
from kernels import _cond_rows, ea_kernel, knn_kernel, sinkhorn_sym
from similarity import memory_kernel_dense, memory_neighbours
from solve import accept, calibrate, solve_modes


@pytest.fixture(scope='module', autouse=True)
def _torch_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def explicit_history(x, a, burn):
    """Independent feature map, including a lag slot for each possible history."""
    t = x.shape[1]
    rows = []
    for context in x:
        for i in range(burn - 1, t):
            rows.append(np.concatenate([a**(s/2) * context[i-s] if s <= i else np.zeros_like(context[0])
                                        for s in range(t)]))
    return np.array(rows)


@pytest.mark.parametrize('reading', ['cosine', 'dot'])
@pytest.mark.parametrize('contexts,burn', [(1, 1), (3, 5)])
def test_memory_dense_explicit_and_streamed(reading, contexts, burn):
    rng = np.random.default_rng(42)
    t = 13
    x = rng.normal(size=(contexts, t, 9))
    x -= x.mean(axis=(0, 1))
    if reading == 'cosine':
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
    else:
        x *= np.linspace(0.1, 3, t)[None, :, None]
    alphas = (0., .2, ALPHA, 1.)
    kd, dd = memory_kernel_dense(x, alphas, burn, reading)
    fmap_errors = []
    for ai, a in enumerate(alphas):
        rows = explicit_history(x, a, burn)
        gram = rows @ rows.T
        if reading == 'cosine':
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            distance = 1 - rows @ rows.T
        else:
            delta = rows[:, None, :] - rows[None, :, :]
            distance = .5 * np.sum(delta**2, axis=2)
        np.testing.assert_allclose(kd[ai], gram, atol=4e-13, rtol=1e-14)
        np.testing.assert_allclose(dd[ai], distance, atol=4e-13, rtol=1e-14)
        fmap_errors.append((np.max(np.abs(kd[ai] - gram)), np.max(np.abs(dd[ai] - distance))))
    k_nb = 6
    nb = memory_neighbours(x, alphas, k_nb, burn, reading, row_block=4)
    assert nb.distances.dtype == nb.similarities.dtype == torch.float64
    j = nb.indices.numpy()
    masked = dd.copy()
    ii = np.arange(masked.shape[-1])
    masked[:, ii, ii] = np.inf
    exact = np.argsort(masked, axis=-1)[..., :k_nb]
    np.testing.assert_array_equal(np.sort(j, axis=-1), np.sort(exact, axis=-1))
    dk = np.take_along_axis(kd, j, axis=-1)
    ds = np.take_along_axis(dd, j, axis=-1)
    np.testing.assert_allclose(nb.similarities.numpy(), dk, atol=4e-13, rtol=1e-14)
    np.testing.assert_allclose(nb.distances.numpy(), ds, atol=4e-13, rtol=1e-14)
    print(f'memory {reading}/{contexts} contexts: fmap {np.max(fmap_errors, axis=0)}, '
          f'kept K {np.max(np.abs(nb.similarities.numpy() - dk)):.3e}, '
          f'd {np.max(np.abs(nb.distances.numpy() - ds)):.3e}')


def test_long_history_topk_matches_dense():
    x = synthetic_site(2, 1023, 2)
    burn, k_nb = 128, 256
    kd, d = memory_kernel_dense(x, ALPHA, burn)
    nb = memory_neighbours(x, ALPHA, k_nb, burn)
    ids = nb.indices.numpy()
    k_error = np.max(np.abs(nb.similarities.numpy() - np.take_along_axis(kd, ids, axis=2)))
    d_error = np.max(np.abs(nb.distances.numpy() - np.take_along_axis(d, ids, axis=2)))
    assert k_error < 1e-11 and d_error < 1e-13
    np.fill_diagonal(d[0], np.inf)
    exact = np.argpartition(d[0], k_nb - 1, axis=1)[:, :k_nb]
    np.testing.assert_array_equal(np.sort(ids[0], axis=1), np.sort(exact, axis=1))
    print(f'memory N={d.shape[-1]}: kept K {k_error:.3e}, d {d_error:.3e}, exact top-{k_nb} sets')


def test_zero_history_and_tied_neighbours():
    x = np.ones((2, 4, 2))
    # All dot distances are zero at a=0. Membership of the selected subset
    # is intentionally unspecified, but every row must have k unique nonself IDs.
    nb = memory_neighbours(x, 0., 3, reading='dot')
    assert (nb.distances == 0).all() and (nb.similarities == 2).all()
    ids = nb.indices[0].numpy()
    assert not (ids == np.arange(len(ids))[:, None]).any()
    assert all(len(set(row)) == 3 for row in ids)
    x[0, 0] = 0
    with pytest.raises(ValueError, match='nonzero'):
        memory_neighbours(x, 0., 3)
    # That zero vector no longer invalidates cosine if excluded as a node.
    assert np.isfinite(memory_neighbours(x, .9, 3, burn_in=2).distances.numpy()).all()


def _dense_walk(p):
    return .5 * (np.eye(len(p)) + p)


@pytest.mark.parametrize('rule,renyi', [('pr', 2), ('shannon', 1)])
def test_untruncated_rule_matches_dense(rule, renyi):
    x = synthetic_site(2, 76, 0, seed=5)
    _, dist = memory_kernel_dense(x, .8)
    C = 1 - dist[0]
    nb = memory_neighbours(x, .8, len(C) - 1)
    ids = nb.indices[0].numpy()
    ds = nb.distances[0].numpy()
    before = ids.copy()
    beta, _, _, logq = _cond_rows(1 - C, 32, True, renyi)
    q, b = conditional_weights(ds, 32, rule)
    q_ref = np.take_along_axis(np.exp(logq), ids, axis=1)
    np.testing.assert_allclose(q, q_ref, atol=2e-14, rtol=0)
    np.testing.assert_allclose(b, beta, atol=2e-11, rtol=2e-13)
    walk = build_walk(ids, ds, rule=rule, k=32, eta=0)
    oracle = ea_kernel(C, 32, renyi=renyi)
    assert oracle['err'] < 1e-10
    expected = _dense_walk(oracle['P'])
    actual = walk.dense()
    np.testing.assert_array_equal(ids, before)  # must not reorder neighbours in place
    np.testing.assert_allclose(actual.sum(axis=1), 1, atol=2e-13, rtol=0)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    print(f'{rule} full: max q error {np.max(np.abs(q-q_ref)):.3e}, '
          f'max M error {np.max(np.abs(actual-expected)):.3e}, dense Sinkhorn {oracle["err"]:.3e}')


def test_rule_floor_and_truncation_drift():
    x = synthetic_site(2, 224, 1, seed=13)
    _, ds = memory_kernel_dense(x, ALPHA)
    C = 1 - ds[0]
    nb = memory_neighbours(x, ALPHA, 256)
    ids, d = nb.indices[0].numpy(), nb.distances[0].numpy()
    eta = 1e-6
    n = len(C)
    walk = build_walk(ids, d, eta=eta)
    # Independent dense reconstruction of the *truncated* rule. Replace
    # unkept similarities by -inf so dense conditional calibration masks them.
    Ccut = np.full_like(C, -np.inf)
    np.put_along_axis(Ccut, ids, np.take_along_axis(C, ids, axis=1), axis=1)
    _, _, _, logq = _cond_rows(1 - Ccut, 64, True, 2)
    q = np.exp(logq)
    np.testing.assert_allclose(1 / np.sum(q*q, axis=1), 64, atol=1e-10, rtol=0)
    a = (q + q.T) / 2
    w = (1 - eta) * a + eta * (1 - np.eye(n))
    v, err = sinkhorn_sym(w, tol=2e-13)
    assert err < 2e-13
    expected = _dense_walk(v[:, None] * w * v[None, :])
    m = walk.dense()
    np.testing.assert_allclose(m, expected, atol=3e-14, rtol=0)
    np.testing.assert_allclose(np.diag(m), .5, atol=1e-16, rtol=0)
    np.testing.assert_allclose(m, m.T, atol=1e-16, rtol=0)
    u = np.random.default_rng(4).normal(size=(n, 3))
    np.testing.assert_allclose(walk @ u, m @ u, atol=2e-14, rtol=0)
    np.testing.assert_allclose(walk @ u[:, 0], m @ u[:, 0], atol=2e-14, rtol=0)
    ref = _dense_walk(ea_kernel(C, 64, renyi=2)['P'])
    print(f'PR256 eta=1e-6: padded dense max M error {np.max(np.abs(m-expected)):.3e}; '
          f'vs untruncated max entry drift {np.max(np.abs(m-ref)):.3e}, '
          f'max row L1 drift {np.linalg.norm(m-ref, ord=np.inf):.3e}')


def test_union_knn_matches_dense():
    x = synthetic_site(1, 100, 0)
    _, distances = memory_kernel_dense(x, .5)
    nb = memory_neighbours(x, .5, 32)
    walk = build_walk(nb.indices[0], nb.distances[0], rule='knn')
    oracle = knn_kernel(1 - distances[0])
    np.testing.assert_allclose(walk.dense(), _dense_walk(oracle['P']), atol=7e-13, rtol=0)
    print(f'kNN15: max M error {np.max(np.abs(walk.dense()-_dense_walk(oracle["P"]))):.3e}')


def test_conditional_endpoints_and_unattainable_ties():
    d = np.array([[0., 0., 1., 2.]])
    for rule in ('pr', 'shannon'):
        q, b = conditional_weights(d, 2, rule)
        np.testing.assert_array_equal(q, [[.5, .5, 0., 0.]])
        assert np.isposinf(b[0])
        q, b = conditional_weights(d, 4, rule)
        np.testing.assert_array_equal(q, [[.25, .25, .25, .25]])
        assert b[0] == 0
        with pytest.raises(ValueError, match='tied'):
            conditional_weights(d, 1, rule)


def test_calibration_rejects_disconnected_walk():
    # Four disjoint triangles; the nonconstant lazy spectrum contains 1.
    n = 12
    ids = np.array([[(i // 3) * 3 + (i + j) % 3 for j in (1, 2)] for i in range(n)])
    distances = np.ones_like(ids, dtype=float)
    walk = build_walk(ids, distances, rule='knn', k=2, eta=0.)
    with pytest.raises(ValueError, match='disconnected'):
        calibrate([walk, walk])


@pytest.mark.parametrize('sites,t,burn', [(4, 600, 89), (8, 1023, 128)])
def test_sparse_solver_matches_dense_oracle(sites, t, burn):
    walks = []
    for site in range(sites):
        nb = memory_neighbours(synthetic_site(2, t, site), ALPHA, 256, burn)
        walks.append(build_walk(nb.indices[0], nb.distances[0]))
    cal = calibrate(walks)
    result = solve_modes(walks, cal)
    assert result.modes.shape == (sites, 2*(t-burn+1), 24) and result.modes.dtype == np.float64
    assert not accept(result)
    # Same mathematical graph, including the positive off-edge floor; this
    # isolates the eigensolver from approximation error due to truncation.
    ms = np.stack([w.dense() for w in walks])
    cs, ts = dense_calibrate(ms)
    vals, vecs, resid = dense_modes(ms, cs, ts)
    np.testing.assert_allclose(cal.scales, cs, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(result.values, vals, atol=2e-11, rtol=2e-10)
    assert vals[16] - vals[15] > 1e-6  # make the compared subspace well-defined
    drift = mode_drift(ts, vals, vecs, cal.t_star, result.values, result.modes)
    assert abs(drift['t_relative']) < 2e-10
    assert drift['subspace_max_sin'] < 5e-8
    print(f"solver N={ms.shape[1]}, L={sites}: t* {ts:.9g}, rel t* {drift['t_relative']:+.3e}, "
          f"max rel eig24 {np.max(np.abs(result.values/vals - 1)):.3e}, "
          f"max sin16 {drift['subspace_max_sin']:.3e}, mean sin²16 {drift['subspace_mean_sin2']:.3e}, "
          f"sparse/dense residual {result.checks['resid'].max():.3e}/{resid:.3e}")
