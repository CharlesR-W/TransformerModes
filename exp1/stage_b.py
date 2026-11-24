"""Stage B numerical primitives. A permutation p maps recipient rows to donor rows.

Chunk arrays are (chunk, site, time, width), oldest first; a single site can
also be passed as (chunk, time, width). Walks and all reductions use float64.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy import linalg
from scipy.sparse import csgraph
from scipy.sparse.linalg import LinearOperator, eigsh

import kernels
import spacetime

PR, TIE, GAP = 64, 1e-12, 1e-8
VIEWS = ('cut', 'recent', 'average')


def lag_weights(length):
    """W in lag order: index zero is the cut."""
    s = np.arange(length, dtype=np.int64)
    return dict(cut=np.outer(s == 0, s == 0).astype(float),
                recent=np.diag((7 / 8) ** s),
                average=np.full((length, length), 1. / length**2))


def _chunk_site(x, views, cut_ids, check_duplicates, block):
    n, h, d = x.shape
    mean = sum(x[i:i+block].sum(axis=(0, 1), dtype=np.float64)
               for i in range(0, n, block)) / (n * h)
    norms = np.empty((n, h))
    grams = {name: np.zeros((n, n)) for name in views}
    pooled = np.zeros((n, d)) if 'average' in grams else None
    w = lag_weights(h)
    for lag in range(h):
        y = x[:, h-1-lag, :] - mean
        z = np.linalg.norm(y, axis=1)
        if not np.isfinite(z).all() or (z == 0).any():
            raise ValueError('nonfinite or zero centred state norm')
        norms[:, lag] = z
        y /= z[:, None]
        if pooled is not None:
            pooled += y
        for name in ('cut', 'recent'):
            if name in grams and w[name][lag, lag]:
                grams[name] += w[name][lag, lag] * (y @ y.T)
    if norms.min() < 1e-6 * np.median(norms):
        raise ValueError('centred state norm below 1e-6 times median')
    if pooled is not None:
        pooled /= h
        z = np.linalg.norm(pooled, axis=1)
        if (z == 0).any() or z.min() < 1e-3 * np.median(z):
            raise ValueError('pooled norm below 1e-3 times median or zero')
        grams['average'] = pooled @ pooled.T
    for name, k in grams.items():
        z = np.sqrt(k.diagonal().copy())
        k /= z[:, None]
        k /= z[None, :]
        k *= -1
        k += 1
        np.maximum(k, 0., out=k)
        np.fill_diagonal(k, 0.)
        if check_duplicates:
            for i, row in enumerate(k):
                j = np.flatnonzero(row[i+1:] < 1e-9) + i + 1
                if len(j) and not (name == 'cut' and cut_ids is not None
                                   and np.all(cut_ids[i] == cut_ids[j])):
                    raise ValueError(f'{name}: off-diagonal distance below 1e-9 in row {i}')
    return grams


def chunk_distances(states, *, views=VIEWS, check_duplicates=True, block=32):
    """One global mean per site; no full float64 copy of the chunk tensor."""
    x = np.asarray(states)
    views = tuple(views)
    if (x.ndim not in (3, 4) or min(x.shape) < 1 or block < 1
            or not views or not set(views) <= set(VIEWS)):
        raise ValueError('invalid chunk shape, views or block size')
    if x.ndim == 3:
        return _chunk_site(x, views, None, check_duplicates, block)
    return [_chunk_site(x[:, l], views, None, check_duplicates, block) for l in range(x.shape[1])]


def layer0_distances(ids, embedding, *, views=VIEWS):
    ids, embedding, views = np.asarray(ids), np.asarray(embedding), tuple(views)
    if (ids.ndim != 2 or min(ids.shape) < 1 or ids.dtype.kind not in 'iu'
            or embedding.ndim != 2 or min(embedding.shape) < 1
            or not views or not set(views) <= set(VIEWS)
            or (ids >= len(embedding)).any() or (ids < 0).any()):
        raise ValueError('invalid embedding, token ids or views')
    ids = ids.astype(np.int64, copy=False)
    return _chunk_site(embedding[ids], views, ids[:, -1], True, 32)


def hellinger_distances(probabilities, *, row_block=64, pair_block=32):
    """Blocked sqrt-probability Gram, with bounded direct near-pair differences.

    Row/column indices remain separate (no row*vocab integer arithmetic).
    Temporary near-pair arrays have at most pair_block vocabulary rows.
    """
    p = np.asarray(probabilities)
    if p.ndim != 2 or min(p.shape) < 1 or min(row_block, pair_block) < 1:
        raise ValueError('invalid prediction shape or block size')
    n = len(p)
    r = np.empty(p.shape, dtype=np.float64)
    for i in range(0, n, row_block):
        b = p[i:i+row_block]
        if (not np.isfinite(b).all() or (b < 0).any()
                or not np.allclose(b.sum(1, dtype=np.float64), 1., atol=1e-12, rtol=0.)):
            raise ValueError('invalid probability rows')
        np.sqrt(b, out=r[i:i+row_block], dtype=np.float64)
    d = np.empty((n, n))
    for i in range(0, n, row_block):
        d[i:i+row_block] = 1 - r[i:i+row_block] @ r.T
    for i in range(0, n, row_block):
        b = d[i:i+row_block]
        a, j = np.nonzero((b < 1e-6) & (np.arange(i, i+len(b))[:, None] < np.arange(n)))
        a += i
        for s in range(0, len(j), pair_block):
            ai, ji = a[s:s+pair_block], j[s:s+pair_block]
            diff = r[ai]
            diff -= r[ji]
            d[ai, ji] = np.einsum('iv,iv->i', diff, diff) / 2
    np.fill_diagonal(d, 0.)
    # Copy only the upper triangle to the lower one, without a full extra matrix.
    for i in range(n):
        d[i+1:, i] = d[i, i+1:]
    return d


@dataclass
class Rows:
    q: np.ndarray
    beta: np.ndarray
    tie_size: np.ndarray
    tie_rows: np.ndarray
    unattainable: np.ndarray
    participation: np.ndarray
    support: np.ndarray
    underflow: np.ndarray


def pr_rows(distances, target=PR, *, block=64):
    """Renyi-2 arithmetic of kernels._cond_rows, with explicit endpoints.

    Tolerance ties are collapsed only in the declared >= target branch. In
    other rows even near minima retain their original distances. If target
    exceeds N-1, report unattainable and return the widest feasible row.
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 2 or not np.isfinite(target) or target < 2 or block < 1:
        raise ValueError('invalid distance shape, participation target or block')
    n = len(d)
    q = np.zeros_like(d)
    beta = np.zeros(n)
    tie_size = np.empty(n, dtype=np.int64)
    for start in range(0, n, block):
        b = d[start:start+block].copy()
        i = np.arange(len(b), dtype=np.int64)
        b[i, start+i] = np.inf
        if np.any(np.isneginf(b)) or np.isnan(b).any() or (b < 0).any() or np.isposinf(b).sum() != len(b):
            raise ValueError('off-diagonal distances must be finite and nonnegative')
        b -= b.min(axis=1, keepdims=True)
        tied = b <= TIE
        k = tie_size[start:start+len(b)] = tied.sum(axis=1)
        out = q[start:start+len(b)]
        special = k >= target
        b_out = beta[start:start+len(b)]
        out[special] = tied[special] / k[special, None]
        b_out[special] = np.inf
        ordinary = ~special
        if target >= n - 1:
            out[ordinary] = np.isfinite(b[ordinary]) / (n - 1)
        elif ordinary.any():
            ds = b[ordinary]
            # Scaling by a target-order distance avoids enormous bracket
            # lengths for small distances. There are < target minima here.
            j = int(np.ceil(target)) - 1
            scale = np.partition(ds, j, axis=1)[:, j]
            with np.errstate(over='ignore', under='ignore'):
                ds /= scale[:, None]

                def cond(betas):
                    # A zero exponent exists in every row, so logsumexp's
                    # row-max subtraction is identically zero here.
                    a = np.exp(-betas[:, None] * ds)
                    z = a.sum(1)
                    return z*z / np.einsum('ij,ij->i', a, a), a, z

                lo, hi = np.zeros(len(ds)), np.ones(len(ds))
                # Bracket from both sides. Specially scaled inputs may need
                # a root much smaller than the absolute bisection resolution.
                below = cond(hi)[0] < target
                while below.any():
                    half = hi / 2
                    lower = cond(half)[0] <= target
                    shrink = below & lower
                    lo[below & ~shrink] = half[below & ~shrink]
                    hi[shrink] = half[shrink]
                    below = shrink
                for _ in range(1024):
                    above = cond(hi)[0] > target
                    if not above.any():
                        break
                    lo[above] = hi[above]
                    hi[above] *= 2
                else:
                    raise ArithmeticError('could not bracket participation target')
                for _ in range(60):
                    mid = (lo + hi) / 2
                    above = cond(mid)[0] > target
                    lo = np.where(above, mid, lo)
                    hi = np.where(above, hi, mid)
                mid = (lo + hi) / 2
                _, a, z = cond(mid)
                out[ordinary] = a / z[:, None]
                b_out[ordinary] = mid / scale
    support = np.count_nonzero(q, axis=1)
    special = tie_size >= target
    participation = 1 / np.einsum('ij,ij->i', q, q)
    if np.max(np.abs(participation[~special] - min(target, n - 1)), initial=0.) > 1e-9:
        raise ArithmeticError('participation bisection did not meet attainable target')
    return Rows(q, beta, tie_size, special, (tie_size > target) | (target > n - 1),
                participation, support, np.where(special, 0, n-1-support))


def _square(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) < 2:
        raise ValueError('expected square matrix of order >= 2')
    return a


def mz_eigh(m, k=None, *, vectors=True):
    """Largest algebraic eigenpairs in the true mean-zero sector, descending."""
    m = _square(m)
    n = len(m)
    k = n - 1 if k is None else k
    if not 1 <= k < n:
        raise ValueError('mean-zero rank out of range')
    result = linalg.eigh(m - 3. / n, subset_by_index=(n-k, n-1),
                         eigvals_only=not vectors, check_finite=True)
    if vectors:
        w, v = result
        return w[::-1], v[:, ::-1]
    return result[::-1]


def _tail(m, k):
    if k >= len(m):
        return np.zeros(len(m))
    return np.maximum(0., m.sum(1) - np.partition(m, -k, axis=1)[:, -k:].sum(1))


def _quantiles(x):
    return dict(zip(('p05', 'median', 'p95'), np.percentile(x, [5, 50, 95]).tolist()))


@dataclass
class Walk:
    s: np.ndarray | None
    report: dict


def build_walk(distances, target=PR):
    """Build S and report reasons not to use it in a primary solve."""
    d = _square(distances)
    n = len(d)
    rows = pr_rows(d, target)
    q = rows.q
    w = (q + q.T) / 2
    components, labels = csgraph.connected_components(w > 0., directed=False)
    off = d[~np.eye(n, dtype=bool)].reshape(n, n-1)
    scale = dict(all=_quantiles(off), nearest=_quantiles(off.min(1)), rank64=None)
    if n > PR:
        scale['rank64'] = _quantiles(np.partition(off, PR-1, axis=1)[:, PR-1])
    del off
    report = dict(target=target, tie_rows=int(rows.tie_rows.sum()), tie_size=rows.tie_size,
                  unattainable_rows=np.flatnonzero(rows.unattainable),
                  participation=rows.participation, conditional_support=rows.support,
                  underflow=rows.underflow, conditional_tail=_tail(q, PR),
                  components=int(components), component_sizes=np.bincount(labels), raw_distance=scale)
    try:
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            v, _ = kernels.sinkhorn_sym(w)
            s = (v[:, None] * w) * v[None, :]
            s = (s + s.T) / 2
        residual = float(np.max(np.abs(s.sum(1) - 1)))
    except (np.linalg.LinAlgError, FloatingPointError):
        s, residual = None, None
    good = residual is not None and np.isfinite(residual) and residual < 1e-11
    spectrum = mz_eigh(s, min(17, n-1), vectors=False) if good else np.empty(0)
    gap = float(1 - spectrum[0]) if len(spectrum) else None
    reasons = []
    if not good:
        reasons.append('Sinkhorn residual')
    if components > 1:
        reasons.append(f'disconnected, {components} components')
    if gap is not None and gap < GAP:
        reasons.append('first nontrivial gap below 1e-8')
    if target > n - 1:
        reasons.append('target unattainable: target exceeds N-1')
    report.update(row_residual=residual, final_support=None if s is None else np.count_nonzero(s, axis=1),
                  final_tail=None if s is None else _tail(s, PR), first_gap=gap,
                  nontrivial_eigenvalues=spectrum[:12],
                  gap_after_16=(float(spectrum[15] - spectrum[16]) if len(spectrum) >= 17 else None),
                  primary_admissible=not reasons, through_depth_admissible=not reasons, reasons=reasons)
    return Walk(s if good else None, report)


def lazy_walk(s):
    a = _square(s) * .5
    a.flat[::len(a)+1] += .5
    return a


def _permutation(p, n):
    p = np.asarray(p)
    if p.shape != (n,) or p.dtype.kind not in 'iu' or not np.array_equal(np.sort(p), np.arange(n)):
        raise ValueError('invalid donor permutation')
    return p.astype(np.int64, copy=False)


def product_walk(a, prediction=None, p=None):
    a = _square(a)
    if prediction is None:
        if p is not None:
            raise ValueError('permutation requires a prediction walk')
        m = a @ a
    else:
        s = _square(prediction)
        if s.shape != a.shape:
            raise ValueError('different layer and prediction sizes')
        if p is not None:
            p = _permutation(p, len(a))
            s = s[np.ix_(p, p)]
        m = (a @ s) @ a
    return (m + m.T) / 2


def null_maximum(a, prediction, p):
    """Matrix-free in the product: x -> A Π S Π^T A x - 3 mean(x)."""
    a, s = _square(a), _square(prediction)
    n = len(a)
    if a.shape != s.shape:
        raise ValueError('different layer and prediction sizes')
    p = _permutation(p, n)
    inv = np.argsort(p)

    def mv(x):
        return a @ (s @ (a @ x)[inv])[p] - 3 * x.mean()

    op = LinearOperator((n, n), matvec=mv, dtype=np.float64)
    v0 = np.random.default_rng(0).standard_normal(n)
    w, v = eigsh(op, k=1, which='LA', tol=1e-10, v0=v0)
    residual = float(np.linalg.norm(mv(v[:, 0]) - w[0]*v[:, 0]))
    if not (-1-1e-9 <= w[0] <= 1+1e-9 and residual < 1e-10
            and abs(v[:, 0].mean()) < 1e-9):
        raise ArithmeticError(f'null solve failed: value={w[0]}, residual={residual}')
    return float(w[0]), residual


def count_margin(maximum):
    return 1e-8 + 1e-8 * np.abs(maximum)


def count_from_spectra(observed, maxima, residuals, *, limit=64):
    w, null, err = map(np.asarray, (observed, maxima, residuals))
    if (w.ndim != 1 or not len(w) or len(w) > limit or null.ndim != 1 or not len(null)
            or null.shape != err.shape or not all(np.isfinite(a).all() for a in (w, null, err))
            or (err < 0).any()):
        raise ValueError('invalid count evidence')
    best = float(null.max())
    eps = float(count_margin(best))
    if eps < 100 * err.max():
        raise ArithmeticError('count tie allowance is too small for stored null residuals')
    count = int(np.count_nonzero(w > best + eps))
    return dict(count=count, censored=count == limit, observed=w, null_maxima=null,
                null_residuals=err, null_maximum=best, tie_margin=eps)


def site_count(a, prediction, permutations, *, limit=64):
    w = mz_eigh(product_walk(a, prediction), min(limit, len(a)-1), vectors=False)
    maxima, err = zip(*(null_maximum(a, prediction, p) for p in permutations), strict=True)
    return count_from_spectra(w, maxima, err, limit=limit)


def site_scales_mz(ms, r=8):
    c = np.array([np.mean(1 - mz_eigh(m, r, vectors=False)) for m in ms])
    if not np.isfinite(c).all() or (c <= 0).any():
        raise ArithmeticError('invalid site scales')
    return c


def site_units(ms, r=8):
    c = site_scales_mz(ms, r)
    # No shifted/sorted eigensystem: balance_direct discards column zero.
    t = spacetime.balance_direct(ms, c)
    if not np.isfinite(t) or t <= 0:
        raise ArithmeticError('invalid balance coupling')
    return c, t


def _depth_dense(ms, c, t):
    """Toy calibration only: direct dense L in a Helmert mean-zero basis."""
    nl, n, _ = ms.shape
    h = linalg.helmert(n, full=False).T
    eye = np.eye(n-1)
    mat = linalg.block_diag(*[(eye - h.T @ m @ h)/scale for m, scale in zip(ms, c, strict=True)])
    mat += t * np.kron(spacetime.path_lap(np.eye(nl)), eye)
    w, v = linalg.eigh(mat)
    return w, h @ v.reshape(nl, n-1, -1)


def _depth_identical(walk, c, t, nl, k):
    """Exact separation for identical blocks, preserving all multiplicities.

    A single-vector Lanczos run can skip members of a repeated site eigenspace
    even with tiny residuals. Identical sites (not merely close sites) have a
    Cartesian-product eigenbasis, so no through-depth Lanczos is needed.
    """
    w, u = mz_eigh(walk, min(k, len(walk)-1))
    temporal, envelope = linalg.eigh(t * spacetime.path_lap(np.eye(nl)))
    vals = (1 - w[None, :])/c + temporal[:, None]
    flat = np.argsort(vals, axis=None, kind='stable')[:k]
    l, i = np.unravel_index(flat, vals.shape)
    return vals[l, i], spacetime.sign_fix(envelope[:, None, l] * u[None, :, i])


@dataclass
class Depth:
    values: np.ndarray
    vectors: np.ndarray
    report: dict


def cutoff_gap(values, k):
    if len(values) <= k or k < 1 or values[k-1] <= 0:
        return None
    return float((values[k] - values[k-1]) / values[k-1])


def _isolated(values, k):
    gap = cutoff_gap(values, k)
    return gap is not None and gap >= .01


def solve_depth(ms, c, t, *, k=16, modes=24, dense=False, past=False):
    """Solve at supplied units; dense=True is for small calibration only."""
    ms, c = np.asarray(ms, dtype=np.float64), np.asarray(c, dtype=np.float64)
    if (ms.ndim != 3 or ms.shape[1] != ms.shape[2] or len(ms) < 2
            or c.shape != (len(ms),) or not np.isfinite(c).all() or (c <= 0).any()
            or not np.isfinite(t) or t <= 0 or not 1 <= k < modes <= len(ms)*(ms.shape[1]-1)):
        raise ValueError('invalid depth problem or mode count')
    if dense:
        w, g = _depth_dense(ms, c, t)
        w, g = w[:modes], g[:, :, :modes]
        solver = 'dense mean-zero'
    elif np.all(c == c[0]) and all(np.array_equal(m, ms[0]) for m in ms[1:]):
        w, g = _depth_identical(ms[0], c[0], t, len(ms), modes)
        solver = 'exact identical sites'
    else:
        w, g = spacetime.modes(ms, c, t, k=modes)
        solver = 'spacetime.modes'
    err = spacetime.residual(ms, c, t, w[:20], g[:, :, :20])
    flat = g.reshape(-1, g.shape[-1])
    ortho = float(np.max(np.abs(flat.T @ flat - np.eye(len(w)))))
    means = float(np.max(np.abs(g.mean(1))))
    occupancy = spacetime.occupancy(g[:, :, :k])
    mismatch = abs(float(w[0]) / spacetime.temporal_eigs(len(ms), t)[1] - 1) if past else None
    accepted = bool(np.isfinite(w).all() and np.isfinite(g).all() and np.max(err) < 1e-8
                    and ortho <= 1e-9 and means <= 1e-9 and abs(occupancy.sum() - k) <= 1e-9
                    and (mismatch is None or mismatch < 1e-6))
    picture_k = next((j for j in range(k, min(k, 8)-1, -1) if _isolated(w, j)), None) if accepted else None
    return Depth(w, g, dict(accepted=accepted, solver=solver, residuals=err, orthonormal_error=ortho,
                           mean_error=means, balance_relative_error=mismatch, occupancy=occupancy,
                           occupancy_fraction=occupancy/k, cutoff_gap=cutoff_gap(w, k), picture_k=picture_k,
                           picture_status=('withheld' if not accepted else 'unresolved' if picture_k is None else 'resolved')))


class Withheld(ValueError):
    """The requested primary/through-depth run is inadmissible."""


def _admitted(walk):
    if isinstance(walk, Walk):
        if not walk.report['through_depth_admissible']:
            raise Withheld('; '.join(walk.report['reasons']))
        return walk.s
    # Directly constructed walks used by fixtures; ordinary data should pass
    # Walk objects so no graph diagnostic is lost.
    s = _square(walk)
    if (not np.isfinite(s).all() or s.min() < 0 or np.max(abs(s-s.T)) > 1e-11
            or np.max(abs(s.sum(1)-1)) >= 1e-11 or 1-mz_eigh(s, 1, vectors=False)[0] < GAP):
        raise Withheld('invalid or inadmissible direct walk')
    return s


@dataclass
class Partner:
    lazy: np.ndarray
    c: np.ndarray
    t: float
    past: Depth
    r: int
    k: int
    modes: int
    dense: bool


def prepare_partner(layers, *, r=8, k=16, modes=24, dense=False):
    a = np.stack([lazy_walk(_admitted(s)) for s in layers])
    ms = np.stack([product_walk(b) for b in a])
    c, t = site_units(ms, r)
    past = solve_depth(ms, c, t, k=k, modes=modes, dense=dense, past=True)
    return Partner(a, c, t, past, r, k, modes, dense)


def solve_product(partner, prediction, p=None, *, own_units=False):
    """Always solve using the partner; optional own units are diagnostic only."""
    s = _admitted(prediction)
    if p is not None:
        p = _permutation(p, len(s))
        s = s[np.ix_(p, p)]
    ms = np.stack([product_walk(a, s) for a in partner.lazy])
    out = solve_depth(ms, partner.c, partner.t, k=partner.k, modes=partner.modes, dense=partner.dense)
    if own_units:
        c, t = site_units(ms, partner.r)
        out.report.update(own_c=c, own_t=t)
    return out


def energy(past, product, k=16):
    if past is None or product is None or not past.report['accepted'] or not product.report['accepted']:
        return None
    if len(past.values) < k or len(product.values) < k or k < 1:
        raise ValueError('insufficient depth modes for energy')
    return float(product.values[:k].sum() - past.values[:k].sum())


def energy_margin(e):
    return 1e-8 * (1 + np.abs(e))


def _rank(observed, null, *, higher, allowance):
    null = np.asarray(null, dtype=float)
    if not np.isfinite(observed) or null.ndim != 1 or not len(null) or not np.isfinite(null).all():
        raise ValueError('permutation reference must be complete and finite')
    against = null >= observed - allowance if higher else null <= observed + allowance
    rank = 1 + int(against.sum())
    return dict(rank=rank, p=rank/(len(null)+1), tie_margin=float(allowance))


def energy_rank(observed, null):
    return _rank(observed, null, higher=False, allowance=energy_margin(observed))


def increment_rank(observed, null):
    return _rank(observed, null, higher=True, allowance=1e-8)


def leading_rank(observed, null):
    """Toy calibration only. Each competitor supplies its own count margin."""
    null = np.asarray(null)
    return _rank(observed, null + count_margin(null), higher=True, allowance=0.)


def energy_reference(observed, null):
    null = list(null)
    if observed is None or any(x is None for x in null):
        return dict(status='withheld', observed=observed, null=null, p=None)
    return dict(status='computed', observed=observed, null=null, **energy_rank(observed, null))


def primary_decision(model_references, moved_fraction):
    refs = tuple(model_references)
    if len(refs) != 2 or not np.isfinite(moved_fraction) or not 0 <= moved_fraction <= 1:
        raise ValueError('primary test needs two models and a valid moved fraction')
    if moved_fraction < .3:
        return 'uninformative token reference'
    if any(a['status'] == 'withheld' for a in refs):
        return 'withheld'
    if any(len(a['null']) != 199 for a in refs):
        raise ValueError('primary reference needs all 199 relabellings')
    return ('the prediction constraint is cheaper for the correct pairing, beyond exact cut-token identity on this sample'
            if all(a['p'] <= .05 for a in refs) else 'not shown')


def alignment(x, y):
    denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError('undefined alignment')
    return float(np.linalg.norm(x.T @ y)**2 / denom)


def attribute_share(x, design):
    """Centred design span share and its rank/(N-1) chance reference."""
    x, z = np.asarray(x, dtype=float), np.asarray(design, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or len(x) != len(z) or len(x) < 2:
        raise ValueError('attribute readout needs two aligned matrices')
    if not np.isfinite(x).all() or not np.isfinite(z).all() or np.max(abs(x.mean(0))) > 1e-9:
        raise ValueError('the mode frame must be finite and centred')
    q = linalg.orth(z - z.mean(0))
    denom = np.linalg.norm(x)**2
    if denom == 0:
        raise ValueError('zero mode frame')
    return dict(share=float(np.linalg.norm(q.T @ x)**2 / denom), rank=q.shape[1], chance=q.shape[1]/(len(x)-1))


def hellinger_token_mass(s, probabilities, *, vocab_block=256):
    """Token contributions to total Hellinger roughness (section 8.4).

    Form S @ sqrt(p) only in vocabulary blocks, then normalise the masses.
    A zero total has no compositional share.
    """
    s, p = _square(s), np.asarray(probabilities)
    if p.ndim != 2 or len(p) != len(s) or p.shape[1] < 1 or vocab_block < 1:
        raise ValueError('invalid token mass problem')
    mass = np.empty(p.shape[1])
    for i in range(0, p.shape[1], vocab_block):
        block = p[:, i:i+vocab_block]
        if not np.isfinite(block).all() or (block < 0).any():
            raise ValueError('invalid probabilities')
        r = np.sqrt(block, dtype=np.float64)
        contributions = block.sum(0, dtype=np.float64) - np.einsum('nv,nv->v', r, s @ r)
        if (contributions < -1e-9).any():
            raise ArithmeticError('negative Hellinger token mass')
        mass[i:i+len(contributions)] = np.maximum(0., contributions)
    total = float(mass.sum())
    shares = mass / total if total > 1e-12 else None
    top = np.argsort(-mass, kind='stable')[:20].astype(np.int64)
    return dict(mass=mass, total=total, shares=shares, top_ids=top,
                top1_share=None if shares is None else float(shares[np.unique(p.argmax(1))].sum()))


def _frame(run, k):
    g = run.vectors[:, :, :k]
    return g.transpose(1, 0, 2).reshape(g.shape[1], -1)


def increment_reference(past, observed, null, *, k=16):
    """Pairs of model runs; all null pairs must use common permutations."""
    null = tuple(null)
    pairs = (past, observed, *null)
    if not null or any(len(pair) != 2 for pair in pairs):
        raise ValueError('increment requires paired models and a complete reference')
    runs = tuple(run for pair in pairs for run in pair)
    if any(run is None or not run.report['accepted'] for run in runs):
        return dict(status='withheld', reason='failed solve acceptance', value=None, p=None)
    if any(not _isolated(run.values, k) for run in runs):
        return dict(status='withheld', reason='unresolved', value=None, p=None)
    base = alignment(*[_frame(run, k) for run in past])
    delta, *deltas = [alignment(*[_frame(run, k) for run in pair]) - base for pair in pairs[1:]]
    return dict(status='computed', value=delta, sign=int(np.sign(delta)), null=deltas,
                **increment_rank(delta, deltas))


def relabellings(n, count, rng, *, groups=None, subsets=None):
    """Uniform permutations, not derangements. Group ordering is sorted by id."""
    if n < 2 or count < 1:
        raise ValueError('invalid relabelling size')
    idx = np.arange(n, dtype=np.int64)
    if groups is None:
        bins, sizes = [idx], np.array([n])
    else:
        labels = np.asarray(groups)
        if labels.shape != (n,):
            raise ValueError('invalid groups')
        _, inverse, sizes = np.unique(labels, return_inverse=True, return_counts=True)
        bins = np.split(np.argsort(inverse, kind='stable').astype(np.int64), np.cumsum(sizes)[:-1])
    if subsets is not None:
        subsets = np.asarray(subsets)
        if subsets.shape != (n,):
            raise ValueError('invalid subsets')
    ps = np.empty((count, n), dtype=np.int64)
    for p in ps:
        p[:] = idx
        for b in bins:
            if len(b) > 1:
                p[b] = rng.permutation(b)
    moved = ps != idx
    size, freq = np.unique(sizes, return_counts=True)
    return ps, dict(moved_fractions=moved.mean(1), moved_fraction=float(moved.mean()),
                    expected_moved_fraction=(n-len(sizes))/n, coverage=float(sizes[sizes > 1].sum()/n),
                    group_size_distribution=dict(zip(size.tolist(), freq.tolist())),
                    sha256=[sha256(p.astype('<i8', copy=False).tobytes()).hexdigest() for p in ps],
                    moved_same_subset=(float((subsets[ps] == subsets)[moved].mean())
                                       if subsets is not None and moved.any() else None))
