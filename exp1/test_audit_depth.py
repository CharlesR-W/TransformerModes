"""Small fixtures for the audit's own numerical routines (dense references; no pipeline code).

    TMPDIR=/var/tmp uv run --with numpy --with scipy --with pytest python -m pytest exp1/test_audit_depth.py -q --basetemp=/var/tmp/pt_audit_depth
"""
import numpy as np
import pytest

import audit_depth as ad


def unit_rows(x):
    x = np.asarray(x, dtype=np.float64)
    return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True))


def random_sites(n=40, w=16, seed=0, repeats=0):
    g = np.random.default_rng(seed)
    x = g.standard_normal((n, w))
    if repeats:
        x[:repeats] = x[0]                       # exact copies: ties at layer-0-like radii
    return unit_rows(x)


def dense_L(P, c, t):
    nl, n, _ = P.shape
    L = np.zeros((nl * n, nl * n))
    D = np.diag(ad.path_degree(nl)) - np.eye(nl, k=1) - np.eye(nl, k=-1)
    for l in range(nl):
        L[l * n:(l + 1) * n, l * n:(l + 1) * n] = (np.eye(n) - P[l]) / (2 * c[l])
    L += t * np.kron(D, np.eye(n))
    return L


@pytest.fixture(scope="module")
def stack():
    nl, n = ad.LAYERS, 30
    Ps, c = [], []
    for l in range(nl):
        P, r, v, E, st = ad.build_site(random_sites(n, 12, seed=l, repeats=3 if l == 0 else 0))
        Ps.append(P)
        w = np.sort(np.linalg.eigvalsh(P))[::-1]
        c.append((1 - w[1:9]).mean() / 2)
    return np.array(Ps), np.array(c)


def test_sinkhorn_doubly_stochastic_and_ties():
    U = random_sites(50, 8, seed=3, repeats=20)
    P, r, v, E, st = ad.build_site(U)
    assert st["err"] < 1e-13
    assert np.allclose(P, P.T, atol=0, rtol=0) or np.abs(P - P.T).max() < 1e-17
    # the 20 identical vectors: every one keeps all 19 others (ties at the radius all kept)
    assert E[:20, :20].sum() == 20 * 19
    assert st["tie_share"] >= 20 / 50


def test_graph_independent_of_storage_order():
    U = random_sites(45, 8, seed=4, repeats=6)
    perm = np.random.default_rng(1).permutation(45)
    P1, r1, *_ = ad.build_site(U)
    P2, r2, *_ = ad.build_site(np.ascontiguousarray(U[perm]))
    assert np.array_equal(r1[perm], r2)
    assert np.abs(P1[np.ix_(perm, perm)] - P2).max() < 1e-12


def test_replay_reproduces_construction_row():
    U = random_sites(60, 10, seed=5, repeats=4)
    P, r, v, E, st = ad.build_site(U)
    idx = np.array([0, 2, 17, 59])
    for block in (1, 3, 4):
        for a in range(0, len(idx), block):
            ii = idx[a:a + block]
            W, _ = ad.place_rows(np.ascontiguousarray(U[ii]), U, r, v, mask=ii)
            assert np.abs(W - P[ii]).sum(axis=1).max() < 1e-12


def test_block_solver_matches_dense(stack):
    P, c = stack
    t, sigma = 0.7, -1e-3
    fac = ad.BlockCholesky(P, c, t, sigma)
    b = np.random.default_rng(2).standard_normal((ad.LAYERS, P.shape[1], 3))
    x = fac.solve(b)
    L = dense_L(P, c, t) - sigma * np.eye(P.shape[0] * P.shape[1])
    xd = np.linalg.solve(L, b.reshape(-1, 3))
    assert np.abs(x.reshape(-1, 3) - xd).max() < 1e-9 * np.abs(xd).max()


def dense_mean_zero_eigs(P, c, t):
    nl, n, _ = P.shape
    L = dense_L(P, c, t)
    Q = np.linalg.qr(np.c_[np.ones(n), np.eye(n)[:, :-1]])[0][:, 1:]     # complement of the constant
    Z = np.kron(np.eye(nl), Q)
    return np.linalg.eigvalsh(Z.T @ L @ Z)


def test_lowest_modes_match_dense(stack):
    P, c = stack
    t = 0.5
    want = dense_mean_zero_eigs(P, c, t)[:12]
    vals, F, res, _ = ad.lowest_modes(P, c, t, 12)
    assert np.abs(vals - want).max() < 1e-10
    assert res.max() < 1e-9
    assert np.abs(F.sum(axis=1)).max() < 1e-12


def test_inertia_count_matches_dense(stack):
    P, c = stack
    t = 0.5
    ev = dense_mean_zero_eigs(P, c, t)
    for k in (3, 8, 20):
        mid = 0.5 * (ev[k - 1] + ev[k])
        cnt, info = ad.inertia_count(P, c, t, mid)
        assert cnt == k
        cnt_r, _ = ad.inertia_count(P, c, t, mid, reverse=True)
        assert cnt_r == k


def test_nystrom_extension_of_construction_point(stack):
    P, c = stack
    t = 0.5
    vals, F, res, _ = ad.lowest_modes(P, c, t, 6)
    i = 7
    for k in range(6):
        b = np.array([P[l, i] @ F[l, :, k] for l in range(ad.LAYERS)]) / (2 * c)
        y = np.linalg.solve(ad.tri_matrix(c, t, vals[k]), b)
        assert np.abs(y - F[:, i, k]).max() < 1e-8


def test_bands_and_partition():
    vals = np.array([1.0, 1.005, 1.2, 1.3, 1.301, 1.302, 2.0])
    bands, link = ad.bands_of(vals)
    assert bands == [(1, 2), (4, 6)]
    groups, crossing = ad.partition_modes(bands, 5)
    assert groups == [(1, 2), (3, 3)] and crossing == (4, 6)


def test_encode_rule():
    x = np.array([[[0.0, 1.0, -0.25, np.nan]]], dtype=np.float32).transpose(0, 2, 1)   # (1, 4, 1)
    q = ad.encode(x, np.array([[1.0]]))
    assert q.ravel().tolist() == [0, 127, -64, -128]          # 127 * sqrt(0.25) = 63.5 -> 64 (half to even)


def test_overlap_rank_rule():
    g = np.random.default_rng(0)
    A = g.standard_normal((1, 50, 8))
    B = A.copy()
    B[:, :, 7] = B[:, :, 0]                                   # rank 7 after centring
    fa, fb = ad.frames(A, 8), ad.frames(B, 8)
    r = ad.ov_from(fa, fa, 8)
    assert abs(r[0]["overlap"] - 1) < 1e-12
    r = ad.ov_from(fa, fb, 8)
    assert r[0]["overlap"] is None and r[0]["rank_b"] == 7
