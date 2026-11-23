"""Row-adaptive sparse graphs. Compact walk/Sinkhorn adapted from DepthEmbedding opt/sparse_core.py."""
from dataclasses import dataclass

import numpy as np
from scipy import sparse

FLOOR = 1e-6


@dataclass
class Walk:
    """Lazy walk M = base + u u^T. Inputs to products may be vectors or blocks."""

    base: sparse.csr_matrix
    u: np.ndarray
    sinkhorn_iterations: int
    sinkhorn_error: float

    def __matmul__(self, x):
        if x.ndim == 1:
            return self.base @ x + self.u * np.einsum('i,i->', self.u, x)
        return self.base @ x + self.u[:, None] * (self.u @ x)

    def dense(self):
        """Diagnostic only."""
        return self.base.toarray() + np.outer(self.u, self.u)

    @property
    def nbytes(self):
        return self.base.data.nbytes + self.base.indices.nbytes + self.base.indptr.nbytes + self.u.nbytes


def conditional_weights(distances, k=64, rule='pr'):
    """Calibrate a *batch* of retained rows, returning (q, beta) in float64.

    An unattainable target (below the number of exactly tied minima or above
    support size) raises. At those two attainable endpoints use the exact
    uniform limits (beta=inf on minima, beta=0 on all neighbours).
    """
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or not np.isfinite(d).all() or rule not in ('pr', 'shannon'):
        raise ValueError('need finite row distances and rule pr or shannon')
    if not 1 <= k <= d.shape[1]:
        raise ValueError('target k must be in [1, K_nb]')
    if k == d.shape[1]:
        return np.full(d.shape, 1. / k), np.zeros(len(d))

    ds = d - d.min(axis=1, keepdims=True)
    minima = ds == 0
    nties = minima.sum(axis=1)
    if (nties > k).any():
        raise ValueError('target smaller than number of tied nearest neighbours')
    at_limit = nties == k
    # Search in row-relative units to avoid losing scale on small distances.
    span = ds.max(axis=1)
    ds /= span[:, None]  # all-tied rows would have raised or returned above
    target = np.log(k) if rule == 'shannon' else k
    work = np.empty_like(ds)

    def entropy(beta):
        np.multiply(-beta[:, None], ds, out=work)
        np.exp(work, out=work)
        z = work.sum(axis=1)
        if rule == 'pr':
            return z*z / np.einsum('ij,ij->i', work, work)
        return np.log(z) + beta * np.einsum('ij,ij->i', work, ds) / z

    lo, hi = np.zeros(len(d)), np.ones(len(d))
    while np.any(bad := ((entropy(hi) > target) & ~at_limit)):
        lo[bad] = hi[bad]
        hi[bad] *= 2
        if not np.isfinite(hi).all():
            raise ValueError('could not bracket a finite beta')
    for _ in range(48):
        mid = (lo + hi) * 0.5
        above = (entropy(mid) > target) & ~at_limit
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    beta = (lo + hi) * 0.5
    entropy(beta)  # work = exp(-beta * ds)
    work[at_limit] = minima[at_limit]
    work /= work.sum(axis=1, keepdims=True)
    beta /= span
    beta[at_limit] = np.inf
    return work, beta


def balance_weights(a, floor=FLOOR, tol=2e-13, maxiter=5000):
    """Balance (1-floor)*a + floor*(11^T - I) without forming the floor."""
    n = a.shape[0]
    d = np.full(n, 1 / np.sqrt((1 - floor) * a.sum() / n + floor * (n - 1)))
    for i in range(maxiter):
        wd = (1 - floor) * (a @ d) + floor * (d.sum() - d)
        r = d * wd
        err = np.max(np.abs(r - 1))
        if err < tol:
            return d, i + 1, float(err)
        if np.any(r <= 0) or not np.isfinite(err):
            raise RuntimeError('Sinkhorn encountered zero or non-finite row mass')
        d /= np.sqrt(r)
    raise RuntimeError(f'Sinkhorn residual {err:g} after {maxiter} steps')


def _make_walk(a, d, floor, it, err, row_batch):
    # a is private to this builder. Scale in place to avoid multiple full CSR
    # copies, with bounded-size temporaries for the per-edge factors.
    for lo in range(0, len(d), row_batch):
        hi = min(len(d), lo + row_batch)
        start, end = a.indptr[lo], a.indptr[hi]
        factor = np.repeat(d[lo:hi], np.diff(a.indptr[lo:hi + 1])) * d[a.indices[start:end]]
        a.data[start:end] *= 0.5 * (1 - floor) * factor
    a.setdiag(0.5 - 0.5 * floor * d**2)
    a.sort_indices()
    return Walk(a, np.sqrt(0.5 * floor) * d, it, err)


def build_walk(indices, distances, *, rule='pr', k=None, eta=FLOOR, row_batch=256,
               tol=2e-13, maxiter=5000):
    """Build one site's walk from (N, K_nb) sorted, nonself, unique neighbours.

    rule='pr' (k=64), 'shannon' (k=64) or 'knn' (k=15). The latter takes the
    first k neighbours, then the binary union. Adaptive rules fit the target
    on *all retained neighbours* and average q with its transpose. Convert
    GPU results to CPU before calling this function.

    Floor convention: W=(1-eta)A+eta(11^T-I), where A is the symmetric
    weighted graph or binary union, with zero diagonal. Thus kNN edges stay
    at 1, nonedges are eta, and adaptive edges get the same off-diagonal
    floor. This is relative to a unit edge, not renormalised by N or support.
    The diagonal correction in the returned CSR cancels the floor's rank-one
    diagonal, leaving M_ii=1/2 exactly up to roundoff.
    """
    ids = np.asarray(indices)
    d = np.asarray(distances)
    if d.ndim != 2 or ids.shape != d.shape or ids.dtype.kind not in ('i', 'u'):
        raise ValueError('indices and distances must have matching (N, K_nb) shapes')
    n, width = ids.shape
    if not 1 <= width < n or not 0 <= eta < 1 or row_batch < 1 or tol <= 0 or maxiter < 1:
        raise ValueError('invalid support size, floor, batch size or Sinkhorn parameters')
    if rule not in ('pr', 'shannon', 'knn'):
        raise ValueError('unknown rule')
    k = (15 if rule == 'knn' else 64) if k is None else k
    if not 1 <= k <= width or (rule == 'knn' and int(k) != k):
        raise ValueError('target k must be achievable on retained support (integer for kNN)')
    cols = int(k) if rule == 'knn' else width
    # Use int32 CSR when possible; budget to allow a worst-case undirected union.
    ix_dtype = np.int32 if 2*n*cols+n <= np.iinfo(np.int32).max else np.int64
    data = np.empty((n, cols), dtype=np.float64)
    for lo in range(0, n, row_batch):
        hi = min(n, lo + row_batch)
        sel = ids[lo:hi, :cols]
        ds = d[lo:hi, :cols]
        if sel.min() < 0 or sel.max() >= n or (sel == np.arange(lo, hi)[:, None]).any():
            raise ValueError('indices must be in range and exclude the diagonal')
        if rule == 'knn':
            # The boundary matters too: otherwise a malformed unsorted input
            # could silently select the wrong k edges.
            full = d[lo:hi]
            if not np.isfinite(full).all() or (full[:, 1:] < full[:, :-1]).any():
                raise ValueError('kNN requires finite, sorted distances')
            data[lo:hi] = 1.
        else:
            data[lo:hi], _ = conditional_weights(ds, k, rule)
    indptr = np.arange(n + 1, dtype=ix_dtype) * cols
    # CSR.sum_duplicates sorts in place: never share the caller's IDs.
    ix = np.array(ids[:, :cols], dtype=ix_dtype, order='C', copy=True).ravel()
    q = sparse.csr_matrix((data.ravel(), ix, indptr), shape=(n, n))
    q.sum_duplicates()
    if q.nnz != n * cols:
        raise ValueError('neighbour indices must be unique within each retained row')
    q.eliminate_zeros()
    if rule == 'knn':
        a = q.maximum(q.T)
    else:
        a = q + q.T
        a.data *= 0.5
    del q, data
    scale, it, err = balance_weights(a, eta, tol, maxiter)
    return _make_walk(a, scale, eta, it, err, row_batch)
