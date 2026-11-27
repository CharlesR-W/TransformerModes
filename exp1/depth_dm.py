# Upstream: DepthEmbedding/invert/dm.py at 6c11546b4793a1ae9193561979f40476aad42fc6
# Upstream SHA-256: 50b4533e5501b12d50ad3e8bd94f40dd15004bb9ddba578c2d621fb576d76f51
"""Smooth doubly stochastic kernel, joint through-depth modes, out-of-sample forward map, decoders.

Everything is numpy and works from cosines alone (Gram C of centred, row-normalised activations and
cross-cosines of new points to the probe), so the same code runs on stored Grams and on raw activations.

Kernel at a site.  d_ij = 1 - cos_ij.  P = diag(v) exp(-d / eps) diag(v) with zero diagonal and v > 0 chosen
so that P is symmetric doubly stochastic (symmetric Sinkhorn).  One parameter, eps, set through the
perplexity exp(mean row entropy of P): the effective number of neighbours of a row.  The scaling v absorbs
any per-point offset d_ij -> d_ij - a_i - a_j, so the kernel adapts to how far each point's neighbours are
(what a per-point bandwidth does) while staying a single smooth symmetric kernel.  Every row and every
column sums to one, so no point can collect more than unit weight.

New point x with cosines c(x, j) to the probe:  p(x, j) = v_j exp(-d(x, j) / eps) / sum_j (same).  A probe
image, treated as a new point with itself left out, gets back its own row of P exactly.

Joint operator (spacetime.py):  L_t = blockdiag((I - M_l) / c_l) + t (D_path (x) I),  M_l = (I + P_l) / 2.
For an eigenpair (mu, f) and a new point with rows p_l(x), the eigen-equation written at the new point is
    (y_l - p_l(x) . f_l) / (2 c_l) + t (deg_l y_l - y_{l-1} - y_{l+1}) = mu y_l ,   l = 0..L-1,
a tridiagonal system in the L unknowns y_l = f_l(x): the Nystrom extension of the joint operator.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import spacetime as st  # noqa: E402


# ------------------------------------------------------------------ kernel

def sinkhorn_sym(K, v0=None, tol=1e-11, max_iter=400):
    """v > 0 with v_i (K v)_i = 1 for a symmetric nonnegative K (damped symmetric Sinkhorn)."""
    v = np.full(len(K), 1 / np.sqrt(K.sum(1).mean())) if v0 is None else v0.copy()
    err = np.inf
    for it in range(max_iter):
        r = v * (K @ v)
        err = np.abs(r - 1).max()
        if err < tol:
            break
        if it < 30 or err > 0.5:                 # damped fixed point far from the solution
            v = v / np.sqrt(r)
            continue
        # Newton on g(u) = log(v (K v)), u = log v:  (diag(r) + P) du = -r g,  P = diag(v) K diag(v)
        P = v[:, None] * K * v[None, :]
        A = P + np.diag(r)
        du = np.linalg.solve((A + A.T) / 2, -r * np.log(r))
        step, ok = 1.0, False
        while step > 1e-4:
            with np.errstate(all='ignore'):
                v2 = v * np.exp(np.clip(step * du, -50, 50))
                e2 = np.abs(v2 * (K @ v2) - 1).max()
            if np.isfinite(e2) and e2 < err:
                ok = True
                break
            step /= 2
        v = v2 if ok else v / np.sqrt(r)
    return v, err


def perplexity(P):
    H = -(P * np.log(np.where(P > 0, P, 1.0))).sum(1)
    return float(np.exp(H.mean()))


def ds_kernel(C, perp=None, eps=None, tol=1e-11, bis=10):
    """Doubly stochastic Gaussian-on-cosine kernel on the probe.  Give eps, or perp to calibrate eps.

    Returns dict(P, eps, logv, perp, err).  logv are the log scalings of the unshifted kernel exp(-d/eps).
    """
    n = len(C)
    D = 1 - np.asarray(C, float)
    D[np.arange(n), np.arange(n)] = np.inf
    a = D.min(1) / 2                          # d_ij - a_i - a_j >= 0: no underflow of a whole row
    Dp = D - a[:, None] - a[None, :]

    def build(e, v0=None):
        K = np.exp(-Dp / e)
        v, err = sinkhorn_sym(K, v0, tol)
        return v[:, None] * K * v[None, :], v, err

    if eps is None:
        lo, hi = np.log(1e-3), np.log(10.0)
        for _ in range(bis):
            mid = (lo + hi) / 2
            P, v, err = build(np.exp(mid))
            if err > 1e-6 or perplexity(P) < perp:   # unconverged only happens at very small eps
                lo = mid
            else:
                hi = mid
        eps = float(np.exp((lo + hi) / 2))
    P, v, err = build(eps)
    P = (P + P.T) / 2
    return dict(P=P, eps=eps, logv=np.log(v) + a / eps, perp=perplexity(P), err=float(err))


def oos_rows(c_new, ker):
    """Rows p(x, .) for new points: c_new (m, n) cosines of the new points to the probe."""
    lw = -(1 - c_new) / float(ker['eps']) + ker['logv'][None, :]
    lw -= lw.max(1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(1, keepdims=True)


def _cond_rows(d, perp, mask_self, renyi=1):
    """Per-row Gaussian conditionals on distances d (m, n) with effective neighbour count `perp`.

    renyi=1: Shannon perplexity exp(entropy).  renyi=2: participation ratio 1 / sum p^2, which bounds every
    single weight by 1/sqrt(perp), so a near-duplicate neighbour cannot hold most of a row.
    Returns beta, dmin, logZ, logp."""
    d = np.array(d, float)
    if mask_self:
        d[np.arange(len(d)), np.arange(len(d))] = np.inf
    dmin = d.min(1)
    ds = d - dmin[:, None]
    fin = np.where(np.isfinite(ds), ds, 0.0)
    target = np.log(perp)

    def cond(beta):
        lg = -beta[:, None] * ds
        mx = lg.max(1, keepdims=True)
        logz = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
        logp = lg - logz[:, None]
        if renyi == 2:
            ent = -np.log((np.exp(2 * logp)).sum(1))
        else:
            ent = logz + beta * (np.exp(logp) * fin).sum(1)
        return ent, logp, logz

    lo, hi = np.zeros(len(d)), np.ones(len(d))
    for _ in range(60):
        bad = cond(hi)[0] > target
        if not bad.any():
            break
        hi[bad] *= 2
    for _ in range(50):
        beta = (lo + hi) / 2
        ent = cond(beta)[0]
        lo = np.where(ent > target, beta, lo); hi = np.where(ent > target, hi, beta)
    beta = (lo + hi) / 2
    ent, logp, logz = cond(beta)
    return beta, dmin, logz, logp


def ea_kernel(C, perp, renyi=1):
    """The existing 'perplexity' rule (CoherentModes sweep_math.perplexity_logkernel), smooth and adaptive.

    Each row gets its own Gaussian width so that its conditional distribution over the other probe images has
    perplexity `perp`; the kernel is the average of the conditional and its transpose, then symmetric Sinkhorn.
    renyi=2 ('pr<k>'): the effective neighbour count is the participation ratio instead of exp(entropy).
    """
    beta, dmin, logz, logp = _cond_rows(1 - np.asarray(C, float), perp, True, renyi)
    K = (np.exp(logp) + np.exp(logp.T)) / 2
    np.fill_diagonal(K, 0)
    v, err = sinkhorn_sym(K)
    P = v[:, None] * K * v[None, :]
    P = (P + P.T) / 2
    return dict(P=P, v=v, beta=beta, dmin=dmin, logz=logz, target=float(perp), renyi=renyi, perp=perplexity(P), err=float(err))


def ea_oos_rows(c_new, ker):
    """Row of a new point for the perplexity rule: its own calibrated conditional towards the probe, averaged
    with each probe image's (frozen) conditional towards it, times the probe's Sinkhorn scalings.
    A probe image is replayed exactly by setting its cosine to itself to -inf (masked)."""
    d = 1 - np.asarray(c_new, float)
    _, _, _, logp_x = _cond_rows(d, float(ker['target']), False, int(ker.get('renyi', 1)))
    with np.errstate(invalid='ignore'):
        logp_j = -ker['beta'][None, :] * (d - ker['dmin'][None, :]) - ker['logz'][None, :]
    logp_j = np.where(np.isfinite(d), logp_j, -np.inf)
    lw = np.logaddexp(logp_x, logp_j) + np.log(ker['v'])[None, :]        # log domain: a new point may sit much
    lw -= lw.max(1, keepdims=True)                                       # nearer a probe image than its neighbours
    w = np.exp(lw)
    return w / w.sum(1, keepdims=True)


def knn_kernel(C, k=15, floor=1e-6):
    """The existing rule: union k-NN graph (edge 1, non-edge `floor`), symmetric Sinkhorn."""
    n = len(C)
    D = 1 - C.copy(); np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, k, 1)[:, :k]
    mask = np.zeros((n, n), bool); mask[np.arange(n)[:, None], nn] = True; mask |= mask.T
    W = np.where(mask, 1.0, floor); np.fill_diagonal(W, 0)
    v, err = sinkhorn_sym(W)
    P = v[:, None] * W * v[None, :]
    rk = np.partition(D, k - 1, 1)[:, k - 1]         # each probe image's k-th neighbour distance
    return dict(P=(P + P.T) / 2, v=v, k=k, floor=floor, rk=rk, err=float(err), perp=perplexity(P))


def knn_oos_rows(c_new, ker):
    """New point for the k-NN rule: an edge to probe image j if j is among the point's k nearest probe images
    or the point is at least as near to j as j's k-th neighbour (the union rule), times the probe's scalings."""
    k = int(ker['k'])
    nn = np.argpartition(-c_new, k, 1)[:, :k]
    W = np.full(c_new.shape, float(ker['floor']))
    W[np.arange(len(c_new))[:, None], nn] = 1.0
    W[(1 - c_new) <= ker['rk'][None, :]] = 1.0
    W[~np.isfinite(c_new)] = 0.0
    W = W * ker['v'][None, :]
    return W / W.sum(1, keepdims=True)


def build_kernel(C, name):
    """'knn<k>', 'ea<perp>' (per-point Shannon perplexity), 'pr<k>' (per-point participation ratio),
    'ds<perp>' (one global width)."""
    if name.startswith('knn'):
        return knn_kernel(C, int(name[3:]))
    if name.startswith('ea'):
        return ea_kernel(C, float(name[2:]))
    if name.startswith('pr'):
        return ea_kernel(C, float(name[2:]), renyi=2)
    return ds_kernel(C, perp=float(name[2:]))


def rows(c_new, ker):
    """Out-of-sample rows for any of the kernels above (dispatch on the stored fields)."""
    if 'beta' in ker:
        return ea_oos_rows(c_new, ker)
    if 'floor' in ker:
        return knn_oos_rows(c_new, ker)
    return oos_rows(c_new, ker)


def k_occurrence(S, k=15):
    """How often each column is among a row's k largest entries of S (diagonal excluded)."""
    S = S.copy(); np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, 1)[:, :k]
    cnt = np.bincount(nn.ravel(), minlength=len(S)).astype(float)
    z = (cnt - cnt.mean()) / cnt.std()
    return dict(max=float(cnt.max()), skew=float((z ** 3).mean()), zero=float((cnt == 0).mean()))


# ------------------------------------------------------------------ joint operator

def joint_modes(Ps, k=16, coupling=1.0, scale='own', tol=1e-10):
    """Lowest k modes of the joint operator at t = coupling * t*.  Ps (L, N, N) non-lazy walks.

    Returns dict(vals (k,), vecs (L, N, k), c (L,), t, tstar).
    """
    L, n, _ = Ps.shape
    ms = (Ps + np.eye(n)[None]) / 2
    lam, vec = st.site_eigs(ms)                      # one full eigendecomposition per site, reused below
    c = (1 - lam[:, 1:1 + st.SCALE_R]).mean(1) if scale == 'own' else np.ones(L)
    tstar = st.balance_direct(ms, c, eigs=(lam, vec))
    del vec
    t = coupling * tstar
    vals, vecs = st.modes(ms, c, t, k=k, tol=tol)
    res = st.residual(ms, c, t, vals, vecs)
    return dict(vals=vals, vecs=vecs, c=c, t=t, tstar=tstar, resid=res)


def joint_forward(rows, jm):
    """Nystrom extension of the joint modes.  rows (L, m, N): p_l(x, .) for m new points.

    Returns y (L, m, k) and cond (k,), the condition number of each mode's L x L system.
    """
    vecs, vals, c, t = jm['vecs'], jm['vals'], jm['c'], jm['t']
    L = len(c)
    deg = np.r_[1.0, np.full(L - 2, 2.0), 1.0] if L > 1 else np.zeros(1)
    b = np.einsum('lmn,lnk->lmk', rows, vecs) / (2 * c[:, None, None])       # (L, m, k)
    y = np.empty_like(b)
    cond = np.empty(len(vals))
    adj = np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
    for j, mu in enumerate(vals):
        T = np.diag(1 / (2 * c) + t * deg - mu) - t * adj
        cond[j] = np.linalg.cond(T)
        y[:, :, j] = np.linalg.solve(T, b[:, :, j])
    return y, cond


def site_forward(rows, lam, vec):
    """Nystrom extension of one site's own walk modes: y = p(x) . phi / lambda.  rows (m, N)."""
    return rows @ vec / lam[None, :]


# ------------------------------------------------------------------ decoders (as weights on probe activations)
# Each returns W (m, n): decoded activation of new point s is sum_j W[s, j] x_j.

def dec_linear(zt, zs):
    """Least-squares projection onto span{1, coordinates}.  No parameters."""
    ft = np.c_[np.ones(len(zt)), zt]; fs = np.c_[np.ones(len(zs)), zs]
    return fs @ np.linalg.pinv(ft)


def sqdist(a, b):
    return np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0)


def dec_nn(zt, zs, k=15):
    d = sqdist(zs, zt)
    nn = np.argpartition(d, k, 1)[:, :k]
    W = np.zeros_like(d); W[np.arange(len(zs))[:, None], nn] = 1 / k
    return W


def knn_scale(zt, k=15):
    d = sqdist(zt, zt); np.fill_diagonal(d, np.inf)
    return float(np.median(np.sqrt(np.partition(d, k - 1, 1)[:, k - 1])))


def gram_scores(W, Ctt, Cst):
    """cos and captured share of each decoded point, from Grams alone (targets have unit norm)."""
    dot = (W * Cst).sum(1)
    n2 = ((W @ Ctt) * W).sum(1)
    return dot / np.sqrt(np.maximum(n2, 1e-300)), 2 * dot - n2


def loo_sse(S_res, Ctt, s_diag):
    """Sum of squared leave-one-out residuals of a linear smoother, from (I - S) and diag(S)."""
    r2 = ((S_res @ Ctt) * S_res).sum(1)
    return float((r2 / (1 - s_diag) ** 2).mean())


def dec_krr(zt, zs, Ctt, hs=(1, 2, 4, 8, 16, 32), ridges=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), linear=True):
    """Kernel ridge (kriging) on the coordinates: Gaussian kernel, optionally after removing the linear fit.

    Bandwidth h (in units of the median 15th-neighbour distance) and ridge chosen by closed-form
    leave-one-out error on the probe.  A fit.  Returns W and the chosen (h, ridge, loo).
    """
    n = len(zt)
    s0 = knn_scale(zt)
    if linear:
        ft = np.c_[np.ones(n), zt]; fs = np.c_[np.ones(len(zs)), zs]
        pin = np.linalg.pinv(ft)
        Hl = ft @ pin                                # projector onto the linear part
        R = np.eye(n) - Hl
    dtt, dst = sqdist(zt, zt), sqdist(zs, zt)
    best = None
    for h in hs:
        K = np.exp(-dtt / (2 * (h * s0) ** 2))
        if linear:
            Kr = R @ K @ R                           # kernel on the residual of the linear fit
        else:
            Kr = K
        w, U = np.linalg.eigh((Kr + Kr.T) / 2)
        w = np.maximum(w, 0)
        for rg in ridges:
            lam = rg * w.max()
            g = w / (w + lam)
            S = (U * g) @ U.T
            if linear:
                S = S + Hl
            loo = loo_sse(np.eye(n) - S, Ctt, np.diag(S).clip(max=1 - 1e-9))
            if best is None or loo < best[0]:
                best = (loo, h, rg)
    loo, h, rg = best
    K = np.exp(-dtt / (2 * (h * s0) ** 2)); Ks = np.exp(-dst / (2 * (h * s0) ** 2))
    if linear:
        Kr = R @ K @ R
        w, U = np.linalg.eigh((Kr + Kr.T) / 2); w = np.maximum(w, 0)
        A = (U / (w + rg * w.max())) @ U.T           # (Kr + lam)^-1
        # prediction: linear part + kernel part on residuals; kernel row for a new point, residualised
        Ksr = (Ks - fs @ pin @ K) @ R
        W = fs @ pin + Ksr @ A @ R
    else:
        w, U = np.linalg.eigh((K + K.T) / 2); w = np.maximum(w, 0)
        W = Ks @ ((U / (w + rg * w.max())) @ U.T)
    return W, dict(h=h, ridge=rg, loo=loo)


def dec_lp(zt, zs, Ctt, levels=14, h0=None):
    """Auto-adaptive Laplacian pyramid (Rabin & Coifman 2012; stopping rule of Fernandez et al.).

    Level i smooths the current residual with a row-normalised Gaussian of width h0 / 2^i whose diagonal is
    zero on the probe, so the probe residual is a leave-one-out residual; stop at its minimum.  No free
    parameter beyond the starting width (the diameter of the cloud).  Returns W and the stopping level.
    """
    n = len(zt)
    dtt, dst = sqdist(zt, zt), sqdist(zs, zt)
    h = h0 or float(np.sqrt(dtt.max()))
    Rt = np.eye(n)                    # residual operator on the probe: residual_i = Rt x
    Ws = np.zeros((len(zs), n))
    best, hist, keep = None, [], None
    for i in range(levels):
        K = np.exp(-dtt / (2 * h * h)); np.fill_diagonal(K, 0)
        rs = K.sum(1, keepdims=True)
        if (rs < 1e-280).any():
            break
        K /= rs
        Ks = np.exp(-dst / (2 * h * h)); rs2 = Ks.sum(1, keepdims=True)
        Ks = Ks / np.maximum(rs2, 1e-300)
        Ws = Ws + Ks @ Rt
        Rt = Rt - K @ Rt
        err = float(((Rt @ Ctt) * Rt).sum(1).mean())
        hist.append(err)
        if best is None or err < best:
            best, keep = err, (Ws.copy(), i)
        elif err > 1.05 * best:
            break
        h /= 2
    return keep[0], dict(level=keep[1], loo=best, hist=hist)


def dec_gh(zt, zs, Ctt, hs=(1, 2, 4, 8, 16, 32), deltas=(1e-1, 1e-2, 1e-3, 1e-4, 1e-6)):
    """Geometric harmonics on the coordinates (Coifman & Lafon 2006; 'latent harmonics').

    Gaussian kernel on the coordinates, expand in its eigenvectors with eigenvalue > delta * largest, extend
    by Nystrom.  (h, delta) chosen by the leave-one-out error of the projection on the probe.  A fit.
    """
    n = len(zt)
    s0 = knn_scale(zt)
    dtt, dst = sqdist(zt, zt), sqdist(zs, zt)
    best = None
    cache = {}
    for h in hs:
        K = np.exp(-dtt / (2 * (h * s0) ** 2))
        w, U = np.linalg.eigh(K)
        w, U = w[::-1], U[:, ::-1]
        cache[h] = (w, U)
        for dl in deltas:
            r = int((w > dl * w[0]).sum())
            Ur = U[:, :r]
            S = Ur @ Ur.T
            loo = loo_sse(np.eye(n) - S, Ctt, np.diag(S).clip(max=1 - 1e-9))
            if best is None or loo < best[0]:
                best = (loo, h, dl, r)
    loo, h, dl, r = best
    w, U = cache[h]
    Ks = np.exp(-dst / (2 * (h * s0) ** 2))
    W = (Ks @ (U[:, :r] / w[:r])) @ U[:, :r].T
    return W, dict(h=h, delta=dl, rank=r, loo=loo)


def dec_pca(Ctt, Cst, k):
    """Best linear k-dimensional decode from the activation itself (not from embedding coordinates)."""
    w, U = np.linalg.eigh(Ctt)
    w, U = w[::-1][:k], U[:, ::-1][:, :k]
    return (Cst @ (U / w)) @ U.T


# ------------------------------------------------------------------ token radius rule (SPEC_DEPTH v3.1)


def unit_vectors(x):
    """C-contiguous float64 unit vectors, without any translation."""
    x = np.array(x, dtype=np.float64, order='C', copy=True)
    norms = np.sqrt(np.einsum('ij,ij->i', x, x))
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('non-finite or zero activation norm')
    x /= norms[:, None]
    return x


def canonical_distances(query, sites=None, *, k=15, frozen_radii=None, mask_self=None):
    """Two passes: GEMM provisional radii; canonical dots near either cutoff; final radii.

    With sites omitted this is construction (both endpoint radii provisional).
    Otherwise compare queries against frozen final site radii. mask_self gives a
    site index per query or -1 for no exclusion. Input rows must be contiguous f64.
    """
    construction = sites is None
    if construction:
        sites = query
        mask_self = np.arange(len(query))
    elif frozen_radii is None or len(frozen_radii) != len(sites):
        raise ValueError('a frozen radius is required for each site')
    if any(x.dtype != np.float64 or not x.flags.c_contiguous or x.ndim != 2 for x in (query, sites)):
        raise ValueError('expected contiguous float64 vectors')
    d = query @ sites.T
    np.subtract(1, d, out=d)
    if mask_self is not None:
        mask_self = np.asarray(mask_self)
        ii = np.flatnonzero(mask_self >= 0)
        d[ii, mask_self[ii]] = np.inf
    provisional = np.partition(d, k - 1, axis=1)[:, k - 1].copy()
    if not np.all(np.isfinite(provisional)):
        raise ValueError('too few unmasked neighbours')
    reverse = provisional if construction else frozen_radii
    # Stream the candidate test by row; do not allocate another full float64 square.
    for i, di in enumerate(d):
        candidates = np.flatnonzero((np.abs(di - provisional[i]) <= 1e-9) |
                                    (np.abs(di - reverse) <= 1e-9))
        qi = query[i]
        for j in candidates:
            # In construction both directed entries are visited, independently.
            di[j] = 1.0 - np.dot(qi, sites[j])
    radius = np.partition(d, k - 1, axis=1)[:, k - 1].copy()
    return d, radius


def radius_kernel(sites, k=15, floor=1e-6):
    """Radius-union walk with inclusive radius ties and symmetric Sinkhorn."""
    d, rk = canonical_distances(sites, k=k)
    forward = d <= rk[:, None]
    ties = (d == rk[:, None]).sum(axis=1) > 1
    forward |= forward.T.copy()
    degree = forward.sum(axis=1)
    del d
    w = np.where(forward, 1.0, floor)
    np.fill_diagonal(w, 0.0)
    v, _ = sinkhorn_sym(w)
    w *= v[:, None] * v[None, :]
    err = float(np.max(np.abs(w.sum(axis=1) - 1)))
    if not err < 1e-10:
        raise RuntimeError(f'Sinkhorn failed: maximum row-sum error {err:.3g}')
    return dict(P=w, v=v, rk=rk, k=k, floor=floor, err=err,
                degree_median=float(np.median(degree)), degree_max=int(degree.max()),
                tie_share=float(ties.mean()))


def radius_rows(query, sites, ker, mask_self=None):
    """Frozen radius-union extension; return rows and nearest unmasked cosine."""
    d, rk = canonical_distances(query, sites, k=ker['k'], frozen_radii=ker['rk'], mask_self=mask_self)
    nearest = 1.0 - d.min(axis=1)
    edge = (d <= rk[:, None]) | (d <= ker['rk'][None, :])
    w = np.where(edge, 1.0, ker['floor'])
    if mask_self is not None:
        mask_self = np.asarray(mask_self)
        ii = np.flatnonzero(mask_self >= 0)
        w[ii, mask_self[ii]] = 0.0
    w *= ker['v'][None, :]
    return w / w.sum(axis=1, keepdims=True), nearest
