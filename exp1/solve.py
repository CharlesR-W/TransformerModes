"""Additive path coupling in slice-mean-zero space, with relative site units.

Adapted from DepthEmbedding opt/sparse_core.py (normal_modes and balance_root).
The operator streams existing per-site CSR walks, rather than copying them
into a block-diagonal CSR. No sparse factorisation or inverse is constructed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigsh

from spacetime import SCALE_R, centre, local_strengths, path_lap, scalar_control, temporal_eigs


@dataclass
class Calibration:
    t_star: float
    scales: np.ndarray
    gaps: np.ndarray
    site_eigenvalues: np.ndarray
    site_matvecs: tuple[int, ...]
    root_matvecs: tuple[int, ...]


@dataclass
class Solution:
    calibration: Calibration
    values: np.ndarray
    modes: np.ndarray                        # (L, N, K), float64, globally orthonormal columns
    matvecs: int
    checks: dict


class Joint:
    def __init__(self, walks, scales):
        self.walks, self.scales = tuple(walks), scales
        self.shape = (len(walks), len(walks[0].u))
        self.upper = 1. / np.min(scales)      # laziness gives lambda_max(I-M) <= 1

    def spatial(self, x):
        y = np.empty_like(x)
        for l, w in enumerate(self.walks):
            y[l] = (x[l] - w @ x[l]) / self.scales[l]
        return y

    def apply(self, x, t):
        y = self.spatial(x)
        if t:
            y += t * path_lap(x)
        return y


def _normal_modes(joint, t, k, ncv, tol, v0):
    """Direct Lanczos; move the entire temporal sector above the spatial spectrum."""
    l, n = joint.shape
    barrier = 1.01 * (joint.upper + 4*t)
    calls = 0

    def mv(x):
        nonlocal calls
        calls += 1
        x = x.reshape(l, n)
        means = x.mean(axis=1, keepdims=True)
        y = centre(joint.apply(x - means, t))
        y += barrier * means
        return y.ravel()

    op = LinearOperator((l*n, l*n), matvec=mv, dtype=np.float64)
    w, v = eigsh(op, k=k, which='SA', ncv=ncv, tol=tol, v0=v0)
    return w, v.reshape(l, n, k), calls


def site_spectra(walks, r=SCALE_R, tol=1e-12):
    """Return the top r *nonconstant* walk eigenvalues and each leading vector."""
    values, first, counts = [], [], []
    for walk in walks:
        n = walk.u.size
        if not 1 <= r < n:
            raise ValueError('need 1 <= r < number of nodes')
        rng = np.random.default_rng(1)
        v0 = rng.standard_normal(n)
        v0 -= v0.mean()
        calls = 0

        def mv(x):
            nonlocal calls
            calls += 1
            avg = x.mean()
            z = walk @ (x - avg)
            return z - (z.mean() + avg)       # move constant to eigenvalue -1

        op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
        w, v = eigsh(op, k=r, which='LA', tol=tol, v0=v0, ncv=min(n, max(2*r + 1, 24)))
        values.append(w[::-1])
        first.append(v[:, -1])
        counts.append(calls)
    return np.array(values), np.array(first), tuple(counts)


def calibrate(walks, r=SCALE_R):
    """Partial site spectra and a scalar bracketed solve for t*. Requires connected sites."""
    walks = tuple(walks)
    if len(walks) < 2 or any(w.u.shape != walks[0].u.shape for w in walks):
        raise ValueError('need >=2 walks on the same number of nodes')
    vals, leading, calls = site_spectra(walks, r)
    raw = 1 - vals
    if np.any(raw[:, 0] <= 1e-13):
        raise ValueError('t* needs positive site gaps; a site is disconnected or numerically so')
    scales = raw.mean(axis=1)
    gaps = raw[:, 0] / scales
    joint = Joint(walks, scales)
    l, n = joint.shape
    tau = temporal_eigs(l, 1.)[1]
    # At t=0, lambda_min=min(gaps). For an upper bound at any t, try each
    # leading spatial mode repeated across depth, making its path energy zero.
    trial = leading.T - leading.mean(axis=1)
    energy = np.zeros(l)
    for w, c in zip(walks, scales):
        energy += np.einsum('ij,ij->j', trial, trial - w @ trial) / c
    energy /= l * np.einsum('ij,ij->j', trial, trial)
    lo, hi = 0.5 * min(gaps) / tau, 1.01 * min(energy) / tau
    noise = centre(np.random.default_rng(0).standard_normal(joint.shape)).ravel()
    noise /= np.linalg.norm(noise)
    seed = (leading[np.argmin(gaps)][None, :] + np.zeros((l, 1))).ravel() / np.sqrt(l)
    counts = []

    def f(t):
        nonlocal seed
        # Adjacent root evaluations typically have the same ground direction.
        # Keep a reproducible broadband component so a switch of branches is
        # not artificially hidden by starting in an invariant subspace.
        w, v, count = _normal_modes(joint, t, 1, ncv=min(l*n, 32), tol=2e-12,
                                    v0=seed + 1e-3 * noise)
        seed = v.ravel().copy()
        counts.append(count)
        return t * tau - w[0]

    t_star = brentq(f, lo, hi, xtol=np.nextafter(0., 1.), rtol=3e-13)
    return Calibration(t_star, scales, gaps, vals, calls, tuple(counts))


def _checks(joint, cal, vals, vecs):
    """Checks corresponding to spacetime_stage.accept that apply at one coupling."""
    l, n, k = vecs.shape
    t = cal.t_star
    spatial = joint.spatial(vecs)
    rough = np.einsum('lnk,lnk->lk', vecs, spatial)
    delta = np.diff(vecs, axis=0)
    change = np.einsum('lnk,lnk->lk', delta, delta).sum(axis=0)
    # This residual is deliberately *not* projected.
    spatial += t * path_lap(vecs)
    spatial -= vecs * vals
    residual = np.sqrt(np.einsum('lnk,lnk->k', spatial, spatial))
    del spatial, delta
    norms = np.einsum('lnk,lnk->lk', vecs, vecs)
    occ = norms.sum(axis=1)
    recurrent_frame = vecs.transpose(1, 0, 2).reshape(n, l*k)
    # Spectrum of R = sum_l G_l G_l^T from the small right Gram; no N-by-N array.
    recurrent = np.maximum(np.linalg.eigvalsh(recurrent_frame.T @ recurrent_frame)[::-1], 0)
    recurrent = recurrent[:min(n, l*k)]
    strength = local_strengths(vecs)
    weight = (norms / norms.sum(axis=0)).T
    retained = np.divide(norms - rough * cal.scales[:, None], norms,
                         out=np.zeros_like(norms), where=norms > 0).T
    cross = np.einsum('lnk,lnk->lk', vecs[:-1], vecs[1:])
    norm = np.sqrt(norms)
    denom = norm[:-1] * norm[1:]
    adj_cos = np.divide(cross, denom, out=np.zeros_like(cross), where=denom > 0).T
    flat = vecs.reshape(l*n, k)
    lower, _ = scalar_control(cal.gaps, t)
    cuts = [i for i in (4, 8, 16) if i <= k]
    trade = np.array([(rough[:, :i].sum(), change[:i].sum()) for i in cuts]).reshape(-1, 2)
    return dict(resid=residual, balance_check=abs(vals[0] / temporal_eigs(l, t)[1] - 1.),
                weight=weight, adj_cos=adj_cos, retention=retained, strengths=strength,
                occupancy=occ, recurrent=recurrent, scalar_energy=lower, ktrade=cuts, tradeoff=trade,
                gaps={i: float(vals[i] - vals[i-1]) for i in cuts if i < k},
                slice_mean_max=np.max(np.abs(vecs.mean(axis=1))),
                orthogonality_error=np.max(np.abs(flat.T @ flat - np.eye(k))))


def accept(result):
    """Return numerical gate violations. Paper modes and multi-t monotonicity do not apply."""
    c = result.checks
    v, t = result.values, result.calibration.t_star
    bad = []

    def chk(ok, message):
        if not ok:
            bad.append(message)

    chk(c['resid'].max() < 1e-8, f"residual {c['resid'].max():.2e}")
    chk(c['balance_check'] < 1e-6, f"balance {c['balance_check']:.2e}")
    chk(np.isclose(c['occupancy'].sum(), len(v), atol=1e-8), 'occupancy does not sum to K')
    chk(np.isclose(c['recurrent'].sum(), len(v), atol=1e-3), 'recurrent strengths do not sum to K')
    chk(c['scalar_energy'] <= v[0] + 1e-10, 'scalar bound violated')
    a, b = c['tradeoff'].T
    chk(np.allclose(a + t*b, [v[:i].sum() for i in c['ktrade']], rtol=1e-8, atol=1e-8),
        'A + tB != sum of eigenvalues')
    chk(c['orthogonality_error'] < 1e-8 and c['slice_mean_max'] < 1e-10, 'orthogonality or slice means')
    for key in ('weight', 'adj_cos', 'retention', 'strengths'):
        chk(np.isfinite(c[key]).all(), f'non-finite {key}')
    chk(np.isfinite(v).all(), 'non-finite eigenvalue')
    return bad


def solve_modes(walks, calibration=None, k=24, *, tol=2e-12, ncv=None):
    """Return lowest k modes at the calibrated t*; raise if acceptance gates fail."""
    walks = tuple(walks)
    cal = calibrate(walks) if calibration is None else calibration
    j = Joint(walks, cal.scales)
    l, n = j.shape
    if not 1 <= k <= l*(n-1):
        raise ValueError('k is outside the slice-mean-zero sector')
    v0 = centre(np.random.default_rng(0).standard_normal(j.shape)).ravel()
    vals, vecs, count = _normal_modes(j, cal.t_star, k, min(l*n, ncv or max(4*k, 40)), tol, v0)
    # ARPACK already returns orthonormal vectors. Clear roundoff leakage and
    # use a reproducible sign, not a prescribed orientation within a cluster.
    vecs -= vecs.mean(axis=1, keepdims=True)
    flat = vecs.reshape(l*n, k)
    # Per-vector argmax avoids allocating another L*N*K matrix for |vecs|.
    for i in range(k):
        col = flat[:, i]
        if col.max() < -col.min():
            col *= -1
    result = Solution(cal, vals, vecs, count, _checks(j, cal, vals, vecs))
    if bad := accept(result):
        raise RuntimeError('; '.join(bad))
    return result
