# Dense oracle excerpts copied unchanged from DepthEmbedding/invert/dm.py, source commit 8dd7a66.
import numpy as np


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
