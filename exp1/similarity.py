"""Synchronous, zero-padded history kernels on (context, position, feature) arrays."""
from dataclasses import dataclass
import operator

import numpy as np
import torch


@dataclass(frozen=True)
class Neighbours:
    """Arrays (len(alphas), N, K_nb), sorted by distance within each row.

    Indices are int32, context-major on the retained suffixes. Similarities are
    the raw K_a, not the cosine-normalised K_a (the latter is 1 - distances).
    Exact distance ties have no guaranteed ordering, including at the cutoff.
    """

    indices: torch.Tensor
    distances: torch.Tensor
    similarities: torch.Tensor
    alphas: tuple[float, ...]


def _parameters(shape, alphas, burn_in, reading):
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('expected nonempty (contexts, positions, features)')
    a = (float(alphas),) if np.isscalar(alphas) else tuple(map(float, alphas))
    if not a or not all(0 <= v <= 1 for v in a):
        raise ValueError('alphas must be a nonempty sequence in [0, 1]')
    b = operator.index(burn_in) - 1
    if not 0 <= b < shape[1] or reading not in ('cosine', 'dot'):
        raise ValueError('burn_in must be in [1, T] and reading cosine or dot')
    return a, b


def memory_kernel_dense(y, alphas, burn_in=1, reading='cosine'):
    """Small reference; return (raw K, distance), each (len(alphas), N, N).

    Positions are numbered from 1; burn_in is inclusive and does not erase
    history. Inputs are already centred and (for cosine) row-normalised.
    """
    y = np.asarray(y, dtype=np.float64)
    alphas, b = _parameters(y.shape, alphas, burn_in, reading)
    m, t, d = y.shape
    x = y.reshape(m * t, d)
    gram = (x @ x.T).reshape(m, t, m, t)
    n = m * (t - b)
    kernels, distances = [], []
    for a in alphas:
        k = gram.copy()
        for i in range(1, t):
            k[:, i, :, 1:] += a * k[:, i - 1, :, :-1]
        k = k[:, b:, :, b:].reshape(n, n)
        h = k.diagonal()
        if reading == 'cosine':
            if np.any(h <= 0):
                raise ValueError('cosine distance requires nonzero retained histories')
            r = 1 / np.sqrt(h)
            dist = 1 - (k * r[:, None]) * r[None, :]
        else:
            dist = 0.5 * (h[:, None] + h[None, :]) - k
        kernels.append(k)
        distances.append(dist)
    return np.stack(kernels), np.stack(distances)


@torch.no_grad()
def memory_neighbours(y, alphas, k_nb, burn_in=1, reading='cosine', *,
                      row_block=64, dtype=torch.float64, device=None):
    """Exact streamed top-k with one Gram row tile and all alphas per context.

    Scratch is O(row_block * M * T * len(alphas)); retained output is
    O(N * k_nb * len(alphas)). No full pairwise array even for M=1 (the output
    itself is quadratic if explicitly requesting all neighbours). Not ANN.
    No re-centring, re-normalising, clipping, or cross-context history.
    """
    y = torch.as_tensor(y, dtype=dtype, device=device)
    alphas, b = _parameters(y.shape, alphas, burn_in, reading)
    m, t, d = y.shape
    kept = t - b
    n = m * kept
    k_nb, row_block = operator.index(k_nb), operator.index(row_block)
    if not 1 <= k_nb < n or row_block < 1 or n > np.iinfo(np.int32).max:
        raise ValueError('need 1 <= k_nb < N, positive row_block and int32 node indices')
    if not bool(torch.isfinite(y).all()):
        raise ValueError('features must be finite')
    # Even with a single context do not turn a full retained suffix into a tile.
    row_block = min(row_block, n - 1)
    a = y.new_tensor(alphas)[:, None, None]
    na = len(alphas)
    flat = y.reshape(m * t, d)
    normsq = (y * y).sum(dim=-1)
    diag = y.new_empty((na, m, t))
    diag[..., 0] = normsq[:, 0]
    for i in range(1, t):
        torch.addcmul(normsq[:, i], a[..., 0], diag[..., i - 1], out=diag[..., i])
    diag = diag[..., b:].reshape(na, n)
    if reading == 'cosine':
        if bool((diag <= 0).any()):
            raise ValueError('cosine distance requires nonzero retained histories')
        diag = diag.rsqrt()
    else:
        diag = diag * 0.5

    indices = torch.empty((na, n, k_nb), dtype=torch.int32, device=y.device)
    distances = y.new_empty((na, n, k_nb))
    similarities = torch.empty_like(distances)
    state = y.new_empty((na, m, t))
    for c in range(m):
        for lo in range(0, t, row_block):
            hi = min(t, lo + row_block)
            gram = (y[c, lo:hi] @ flat.T).reshape(hi - lo, m, t)
            # Keep the predecessor across row tiles, but restart at t=1 of
            # each left context. The column shift is only along each block.
            k = y.new_empty((hi - lo, na, m, t))
            for j in range(hi - lo):
                if lo + j == 0:
                    k[j].copy_(gram[j])
                else:
                    prev = k[j - 1] if j else state
                    k[j, ..., 0] = gram[j, :, 0]
                    torch.addcmul(gram[j, ..., 1:], a, prev[..., :-1], out=k[j, ..., 1:])
            state.copy_(k[-1])
            if hi <= b:
                continue
            start = max(lo, b)
            i0, i1 = c * kept + start - b, c * kept + hi - b
            raw = k[start - lo:, :, :, b:].reshape(hi - start, na, n).transpose(0, 1)
            if reading == 'cosine':
                dist = (raw * diag[:, i0:i1, None]).mul_(diag[:, None, :]).neg_().add_(1)
            else:
                dist = (diag[:, i0:i1, None] + diag[:, None, :]).sub_(raw)
            rows = torch.arange(i1 - i0, device=y.device)
            dist[:, rows, rows + i0] = torch.inf
            v, j = torch.topk(dist, k_nb, dim=-1, largest=False, sorted=True)
            distances[:, i0:i1] = v
            indices[:, i0:i1] = j
            similarities[:, i0:i1] = raw.gather(-1, j)
    return Neighbours(indices, distances, similarities, alphas)
