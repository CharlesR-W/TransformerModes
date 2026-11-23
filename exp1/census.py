"""Stage A geometry. All public node indices are zero-based, text-major.

On disk the index is interpreted via pool.npz (immutable text, 1-based
position). Origin order is other/far/near; overlap order is local/random/
embedding and count order is strict-strict/strict-tied/tied-strict/tied-tied.
Only transient tiles of distances and cutoff membership are ever dense.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time
from typing import Optional

import numpy as np
import torch

MEMORIES = {"none": 0., "2": .5, "8": .875, "64": .984375, "whole": 1.}
PRIORITY = (("cosine", "C"), ("cosine", "O"), ("dot", "C"), ("dot", "O"))
MASKS = ("none", "window", "other_texts")
REFERENCES = ("local", "random", "embedding")
SITES = tuple(range(2, 17, 2))
TIE_RTOL = 1e-12
DISPLAY_K = 8
ORIGIN_BOOTSTRAP_SEED = 23  # Same document-resampling convention as readout.py.
ORIGIN_BOOTSTRAP_DRAWS = 200
DISPLAY_DISTANCE = {
    "cosine": dict(quantity="cosine distance", units="dimensionless", similarity="1 - distance"),
    "dot": dict(quantity="half squared history-feature distance divided by the pool mean K(t,t)",
                units="dimensionless, scaled by mean K(t,t)"),
}


def atomic(path, writer, *args, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            writer(stream, *args, **kwargs)
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda f: f.write(json.dumps(value, indent=2, allow_nan=False).encode()))


@dataclass(frozen=True)
class Pool:
    """Equal length contexts, without the position-0 separator."""
    tokens: np.ndarray
    texts: tuple[int, ...]
    start: int
    suffix_start: int
    query: tuple[int, ...] = ()
    audit: tuple[int, ...] = ()
    source: np.ndarray | None = None  # first position included, separator excluded

    def __post_init__(self):
        if (self.tokens.ndim != 2 or len(self.tokens) != len(self.texts) or not self.texts
                or tuple(sorted(set(self.texts))) != self.texts
                or not 1 <= self.start <= self.suffix_start <= self.tokens.shape[1]):
            raise ValueError("need ordered unique contexts and valid pool/suffix ranges")
        if (len(set(self.query)) != len(self.query) or len(set(self.audit)) != len(self.audit)
                or any(i < 0 or i >= self.n for i in self.query)
                or not set(self.audit).issubset(set(self.query).intersection(self.suffix))):
            raise ValueError("audit queries must be unique suffix members of the fixed Q")
        if self.source is not None and (len(self.texts) != 1 or self.source.shape != (self.length,)):
            raise ValueError("one complete source schedule required for a stitch")

    @property
    def length(self):
        return self.tokens.shape[1]

    @property
    def kept(self):
        return self.length - self.start + 1

    @property
    def n(self):
        return len(self.texts) * self.kept

    @property
    def suffix(self):
        return np.flatnonzero(self.position >= self.suffix_start)

    @property
    def text(self):
        return np.repeat(self.texts, self.kept)

    @property
    def position(self):
        return np.tile(np.arange(self.start, self.length + 1), len(self.texts))

    @property
    def ids(self):
        return self.tokens[:, self.start - 1:].reshape(-1)

    def save(self, path):
        atomic(path, np.savez, text=self.text, position=self.position, token=self.ids, query=self.query,
               audit=self.audit, suffix=self.suffix)


def load_probe(root):
    return json.loads((Path(root) / "probe" / "probe.json").read_bytes())


def pool_from_probe(probe, stitch=None):
    contexts = {c["id"]: c for c in probe["contexts"]}
    sets = probe["node_sets"]
    ranges = sets["census_pool"] if stitch is None else sets["one_text"][str(stitch)]
    texts, starts, stops = zip(*ranges)
    start, stop = starts[0], stops[0]
    if set(starts) != {start} or set(stops) != {stop}:
        raise ValueError("expected equal ranges")
    tokens = np.asarray([contexts[c]["token_ids"] for c in texts], dtype=np.int64)
    if tokens.shape[1] != stop - 1:
        raise ValueError("probe range does not end at the context boundary")
    if stitch is not None:
        return Pool(tokens, texts, start, start, source=np.asarray(contexts[stitch]["source_positions"][1:])[:, 0])
    suffix = sets["suffix"][0][1]
    if sets["suffix"] != [[c, suffix, stop] for c in texts]:
        raise ValueError("unexpected suffix ranges")
    index = {c: j for j, c in enumerate(texts)}

    def convert(pairs):
        if any(not start <= p < stop for _, p in pairs):
            raise ValueError("sample position outside the pool")
        return tuple(index[c] * (stop - start) + p - start for c, p in pairs)

    return Pool(tokens, texts, start, suffix, convert(sets["query_sample"]), convert(sets["audit_queries"]))


def load_site(root, model, pool, site):
    return np.stack([np.load(Path(root) / "acts" / model / f"c{c:02d}.npy", mmap_mode="r")[site, 1:]
                     for c in pool.texts])


def input_name(site):
    return f"site{site:02d}"


def setting_path(root, model, reading, centring, name, memory, mask):
    return Path(root) / "census" / model / f"{reading}_{centring}" / name / f"{memory}_{mask}.npz"


def random_vectors(tokens, width=2048, seed=7):
    """Draw rows by token ID from default_rng(seed); omissions never shift IDs."""
    if np.any(tokens < 0):
        raise ValueError("negative token ID")
    ids, inverse = np.unique(tokens, return_inverse=True)
    rng = np.random.default_rng(seed)
    table = np.empty((len(ids), width), dtype=np.float64)
    stop = int(ids[-1]) + 1
    for start in range(0, stop, 512):
        end = min(start + 512, stop)
        draw = rng.standard_normal((end - start, width))
        lo, hi = np.searchsorted(ids, [start, end])
        table[lo:hi] = draw[ids[lo:hi] - start]
    return table[inverse.reshape(tokens.shape)]


def centre(x, centring, reading, device=None, *, corpus_mean=None):
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    if centring not in ("C", "O") or reading not in ("cosine", "dot") or x.ndim != 3:
        raise ValueError("invalid features, centring or reading")
    if corpus_mean is not None:
        if centring != "C":
            raise ValueError("a fixed corpus mean is only defined for C")
        mu = torch.as_tensor(corpus_mean, device=x.device, dtype=x.dtype).reshape(1, 1, x.shape[-1])
    else:
        mu = x.mean((0, 1), keepdim=True) if centring == "C" else x.mean(1, keepdim=True)
    y = x - mu
    norms = torch.linalg.vector_norm(y, dim=-1)
    median = torch.quantile(norms.flatten(), .5)
    if not bool(torch.isfinite(norms).all()) or bool((norms < 1e-6 * median).any()) or bool((norms == 0).any()):
        raise ValueError("nonfinite or near-zero centred vector")
    return (y / norms[..., None] if reading == "cosine" else y), mu


def _duplicate_labels(tokens, mu):
    """Exact zero distances for identical token vectors / identical histories.

    Both adversaries map each ID to one vector. Equality of centring means
    across texts is allowed. Identical histories need equal lengths since
    position 0 and earlier are zero and retained feature norms are nonzero.
    This removes cancellation noise at mathematical zero, not near-zero ties.
    """
    if len(mu) == 1:
        codes = tokens.copy()
    else:
        # Equal ID multisets have one mathematical mean, even if reduction
        # order changed its last bits; also merge exactly equal computed means.
        _, representatives, same_counts = np.unique(np.sort(tokens, axis=1), axis=0, return_index=True, return_inverse=True)
        _, group = np.unique(mu[representatives, 0], axis=0, return_inverse=True)
        codes = tokens + group[same_counts.reshape(-1)].reshape(-1, 1) * (1 + int(tokens.max()))
    order = np.lexsort(codes[:, ::-1].T)
    sorted_rows = codes[order]
    boundary = np.ones_like(sorted_rows, dtype=bool)
    boundary[1:] = np.logical_or.accumulate(sorted_rows[1:] != sorted_rows[:-1], axis=1)
    histories = np.empty_like(codes)
    histories[order] = np.cumsum(boundary, axis=0) + len(tokens) * np.arange(tokens.shape[1])
    return codes, histories


@torch.jit.script
def _recur_tile(gram: torch.Tensor, alpha: torch.Tensor, state: Optional[torch.Tensor]) -> torch.Tensor:
    # Row recurrence from similarity.py, scripted to keep per-token Python
    # out of the inner loop. Candidate texts and positive alphas are batched.
    k = gram.new_empty((gram.shape[0], alpha.shape[0], gram.shape[1], gram.shape[2]))
    k[..., 0].copy_(gram[:, None, :, 0])
    if state is None:
        k[0].copy_(gram[0])
    else:
        torch.addcmul(gram[0, ..., 1:], alpha, state[..., :-1], out=k[0, ..., 1:])
    for j in range(1, gram.shape[0]):
        torch.addcmul(gram[j, ..., 1:], alpha, k[j - 1, ..., :-1], out=k[j, ..., 1:])
    return k


class Kernel:
    """Per-input float64 distance tiles, computed in ascending time per text."""

    def __init__(self, y, pool, reading, alphas, *, duplicate_labels=None):
        if tuple(y.shape[:2]) != pool.tokens.shape or y.dtype != torch.float64:
            raise ValueError("features and pool disagree")
        self.y = y
        self.pool = pool
        self.reading = reading
        self.alphas = tuple(alphas)
        self.zero, self.nonzero = [], []
        for i, a in enumerate(alphas):
            (self.nonzero if a else self.zero).append(i)
        self.a = y.new_tensor([alphas[i] for i in self.nonzero])[:, None, None]
        self.flat = y.flatten(0, 1)
        self.labels = duplicate_labels
        sq = (y * y).sum(-1)
        h = sq.expand(len(alphas), *sq.shape).clone()
        # Diagonal recurrence from similarity.py: no unstable inverse powers.
        # The vectorised doubling scan here is equivalent to its sequential
        # recurrence up to floating point rounding; the Gram below is sequential.
        aa = y.new_tensor(alphas)[:, None, None]
        step = 1
        while step < pool.length:
            h[..., step:] = torch.addcmul(h[..., step:], aa.pow(step), h[..., :-step])
            step *= 2
        self.diag = h[..., pool.start - 1:].reshape(len(alphas), pool.n)
        self.scale = self.diag.mean(-1)
        self.norm = self.diag.rsqrt() if reading == "cosine" else self.diag * .5
        self.state = None

    def tile(self, c, lo, hi, *, raw_output=False):
        """lo/hi are zero-based history offsets. Returns IDs, raw K if requested, d."""
        p = self.pool
        m, t = p.tokens.shape
        if lo == 0:
            self.state = None
        gram = (self.y[c, lo:hi] @ self.flat.T).reshape(hi - lo, m, t)
        first = max(lo, p.start - 1)
        if self.nonzero:
            k = _recur_tile(gram, self.a, self.state)
            self.state = k[-1].clone()
        if first >= hi:
            return None
        i0 = c * p.kept + first - (p.start - 1)
        i1 = i0 + hi - first
        q = torch.arange(i0, i1, device=self.y.device)
        offset = first - lo
        ds, ks = [], []
        for ai, a in enumerate(self.alphas):
            g = gram[offset:] if not a else k[offset:, self.nonzero.index(ai)]
            raw = g[..., p.start - 1:].reshape(len(q), p.n)
            if raw_output:
                ks.append(raw)
            h = self.norm[ai]
            if self.reading == "cosine":
                d = (raw * h[q, None]).mul_(h).neg_().add_(1.)
            else:
                d = (h[q, None] + h).sub_(raw).div_(self.scale[ai])
            if self.labels is not None:
                code = self.labels[bool(a)]
                d.masked_fill_(code[q, None] == code, 0.)
            ds.append(d)
        return q, ks, ds


def origin_codes(text, position, query, candidate, window=64):
    gap = (position[query, None] - position[candidate]).abs()
    same = text[query, None] == text[candidate]
    return (same.to(torch.int8) + (same & (gap <= window)).to(torch.int8)), gap


def eligible_mask(origin, query, candidate, mask, *, ntexts):
    if mask == "other_texts":
        if ntexts < 2:
            raise ValueError("other-texts mask requires more than one text")
        return origin == 0
    if mask == "window":
        return origin != 2
    if mask == "none":
        return query[:, None] != candidate
    raise ValueError(f"unknown mask {mask}")


@dataclass
class Top:
    indices: torch.Tensor  # index within the candidate array, -1 for padding
    cutoff: torch.Tensor
    t_all: torch.Tensor
    t_in: torch.Tensor
    classes: torch.Tensor  # 0 outside, 1 strictly inside, 2 tied


def tied(a, b):
    return torch.isfinite(b) & ((a - b).abs() <= TIE_RTOL * torch.maximum(a.abs(), b.abs()))


def top_with_ties(d, k):
    """Use topk for the order statistic, never for arbitrary tie membership.

    d is already masked (+inf) and columns are in lexicographic identity order.
    Strict entries are ordered by (distance, identity); the tied cutoff is
    appended in identity order even when the distances differ by rounding.
    """
    n = d.shape[1]
    kk = min(k, n)
    v, j = torch.topk(d, kk, dim=-1, largest=False, sorted=True)
    cutoff = v[:, -1]
    finite = torch.isfinite(d)
    at = finite & tied(d, cutoff[:, None])
    strict = finite & (d < cutoff[:, None]) & ~at
    if kk < k:
        # If the entire candidate set is smaller than k, there is no cutoff.
        strict, at = finite, torch.zeros_like(at)
        cutoff = torch.full_like(cutoff, torch.inf)
    classes = strict.to(torch.int8) + at.to(torch.int8) * 2
    t_all = at.sum(1)
    t_in = torch.clamp(kk - strict.sum(1), min=0).minimum(t_all)

    # Fast path uses only the top-k values/indices to order strict members.
    # Recover full ties on the rows where topk could have picked a wrong ID.
    jj = j.sort(dim=1).values
    vv = d.gather(1, jj)
    cls = classes.gather(1, jj)
    keys = torch.where(cls == 1, vv, torch.inf)
    ordered = keys.argsort(dim=1, stable=True)
    ii = jj.gather(1, ordered)
    picked = cls.gather(1, ordered)
    tied_j = torch.where(cls == 2, jj, n).sort(dim=1).values
    # First kk true columns (by integer identity) for overflowing tie rows.
    over = torch.where(t_all > t_in)[0]
    if over.numel():
        candidates = torch.arange(n, device=d.device)
        tied_j[over] = torch.topk(torch.where(at[over], candidates, n), kk, dim=1, largest=False, sorted=True).values
    slots = torch.arange(kk, device=d.device) - (picked == 1).sum(1, keepdim=True)
    use_tie = (slots >= 0) & (slots < t_in[:, None])
    ii = torch.where(picked == 1, ii, torch.where(use_tie, tied_j.gather(1, slots.clamp_min(0)), -1))
    if kk < k:
        ii = torch.nn.functional.pad(ii, (0, k - kk), value=-1)
    return Top(ii, cutoff, t_all, t_in, classes)


def overlap_counts(a, b):
    """Exploit that strict sets and non-overflowing tie sets fit in top-k."""
    if a is b:
        strict = (a.indices >= 0).sum(1) - a.t_in
        zeros = torch.zeros_like(strict)
        return torch.stack((strict, zeros, zeros, a.t_all), dim=1)
    ai, bi = a.indices.clamp_min(0), b.indices.clamp_min(0)
    aa, ab = a.classes.gather(1, ai), b.classes.gather(1, ai)
    ba, bb = a.classes.gather(1, bi), b.classes.gather(1, bi)
    ast = (aa == 1) & (a.indices >= 0)
    bst = (bb == 1) & (b.indices >= 0)
    aa_t = (aa == 2) & (a.indices >= 0)
    bb_t = (bb == 2) & (b.indices >= 0)
    both_t = torch.where(a.t_all <= a.t_in, (aa_t & (ab == 2)).sum(1), (bb_t & (ba == 2)).sum(1))
    over = torch.where((a.t_all > a.t_in) & (b.t_all > b.t_in))[0]
    if over.numel():
        both_t[over] = ((a.classes[over] == 2) & (b.classes[over] == 2)).sum(1)
    return torch.stack(((ast & (ab == 1)).sum(1), (ast & (ab == 2)).sum(1),
                        (bst & (ba == 2)).sum(1), both_t), dim=1)


def overlap_values(counts, a_in, a_all, b_in, b_all):
    """Expected and maximum *counts* (divide by k to obtain shares).

    f(z) from §4 increases until both min terms begin decreasing, then
    decreases. The last maximiser is the larger breakpoint, clamped to the
    admissible interval. Supports both NumPy and Torch arrays.
    """
    s, c1, c2, c3 = (counts[..., i] for i in range(4))
    xp = torch if isinstance(counts, torch.Tensor) else np
    # Only apply where a cutoff group exists (0/0 would be undefined).
    p_num, q_num = (a_in.double(), b_in.double()) if xp is torch else (a_in, b_in)
    p = p_num / xp.maximum(a_all, a_all * 0 + 1)
    q = q_num / xp.maximum(b_all, b_all * 0 + 1)
    z = xp.minimum(xp.minimum(c3, xp.minimum(a_in, b_in)), xp.maximum(xp.maximum(b_in - c1, a_in - c2), c3 * 0))
    return s + c1 * q + c2 * p + c3 * p * q, s + z + xp.minimum(c1, b_in - z) + xp.minimum(c2, a_in - z)


def cond_rows(d, target=64):
    """Float64 PR rows with the doubling/bisection of kernels._cond_rows.

    d already contains +inf for masked entries. Status: 0 calibrated,
    1 fewer eligible than target, 2 tied minima prevent reaching target at
    finite beta, 3 doubling failed to bracket the target numerically.
    """
    dmin = d.min(dim=1).values
    n = torch.isfinite(d).sum(1)
    minima = ((d == dmin[:, None]) & torch.isfinite(d)).sum(1)
    status = torch.zeros_like(n, dtype=torch.int8)
    status[n < target] = 1
    status[(n > target) & (minima >= target)] = 2
    ids = torch.where((status == 0) & (n > target))[0]
    rows = torch.zeros_like(d)
    beta_out = torch.full_like(dmin, torch.nan)
    logz_out = torch.full_like(dmin, torch.nan)
    uniform = (status == 0) & (n == target)
    rows[uniform] = torch.isfinite(d[uniform]).to(d.dtype) / target
    beta_out[uniform], logz_out[uniform] = 0., np.log(target)
    if len(ids):
        ds = d[ids] - dmin[ids, None]
        target_log = np.log(target)

        def evaluate(beta):
            lg = -beta[:, None] * ds  # row max is exactly zero
            logz = lg.exp().sum(1).log()
            logp = lg - logz[:, None]
            ent = -(2 * logp).exp().sum(1).log()
            return ent, logp, logz

        lo, hi = torch.zeros_like(dmin[ids]), torch.ones_like(dmin[ids])
        for _ in range(60):
            bad = evaluate(hi)[0] > target_log
            if not bool(bad.any()):
                break
            hi *= torch.where(bad, 2., 1.)
        bad = evaluate(hi)[0] > target_log
        status[ids[bad]] = 3
        for _ in range(50):
            beta = (lo + hi) * .5
            bigger = evaluate(beta)[0] > target_log
            lo, hi = torch.where(bigger, beta, lo), torch.where(bigger, hi, beta)
        beta = (lo + hi) * .5
        _, logp, logz = evaluate(beta)
        good = ~bad
        rows[ids[good]] = logp[good].exp()
        beta_out[ids[good]], logz_out[ids[good]] = beta[good], logz[good]
    return rows, status, beta_out, dmin, logz_out


def position_baseline(source):
    """b(signed lag), offset by len(source)-1. Counts only node positions."""
    source = np.asarray(source)
    if source.ndim != 1 or not len(source):
        raise ValueError("expected nonempty source schedule")
    from scipy.signal import correlate
    ans = np.zeros(2 * len(source) - 1)
    for s in np.unique(source):
        x = (source == s).astype(np.int64)
        ans += correlate(x, x, mode="full", method="auto")
    return ans / (len(source) - np.abs(np.arange(1 - len(source), len(source))))


def stacked_counts(mask, origin):
    return torch.stack([(mask & (origin == g)).sum(1) for g in range(3)], dim=1)


def chance_statistics(eligibility, origin, exact_token):
    return dict(eligible=stacked_counts(eligibility, origin), same_token_eligible=(eligibility & exact_token).sum(1))


def statistics(top, origin, gap, eligibility, query, candidates, tokens, *, chance=None):
    valid = top.indices >= 0
    j = top.indices.clamp_min(0)
    o = origin.gather(1, j)
    same_gaps = torch.where(valid & (o != 0), gap.gather(1, j), torch.iinfo(gap.dtype).max).sort(1).values
    nsame = (valid & (o != 0)).sum(1, keepdim=True)
    low = same_gaps.gather(1, ((nsame - 1) // 2).clamp_min(0))[:, 0].double()
    high = same_gaps.gather(1, nsame // 2)[:, 0].double()
    if chance is None:
        chance = chance_statistics(eligibility, origin, tokens[query, None] == tokens[candidates])
    return dict(origins=stacked_counts(valid, o), **chance,
                gap_median=torch.where(nsame[:, 0] > 0, (low + high) * .5, torch.nan),
                same_token=(valid & (tokens[query, None] == tokens[candidates[j]])).sum(1))


def _selected(top, candidates):
    return torch.where(top.indices >= 0, candidates[top.indices.clamp_min(0)], -1)


def _display_distances(top, d):
    """Gather the first eight final IDs, not the provisional torch.topk ranks."""
    idx = top.indices[:, :DISPLAY_K]
    vals = d.gather(1, idx.clamp_min(0)).float().masked_fill_(idx < 0, torch.nan)
    if idx.shape[1] < DISPLAY_K:
        vals = torch.nn.functional.pad(vals, (0, DISPLAY_K - idx.shape[1]), value=torch.nan)
    return vals


def _arrays(top, other, candidates):
    out = dict(indices=_selected(top, candidates), cutoff=top.cutoff, t_all=top.t_all, t_in=top.t_in)
    if other:
        ct = torch.stack([overlap_counts(top, ref) for ref in other], dim=1)
        expected, maximum = overlap_values(ct, top.t_in[:, None], top.t_all[:, None],
                                           torch.stack([r.t_in for r in other], 1),
                                           torch.stack([r.t_all for r in other], 1))
        out.update(overlap_counts=ct, overlap_expected=expected, overlap_maximum=maximum)
    return out


class _Output:
    """Allocate per-setting arrays on first write; store only sampled tie IDs."""
    def __init__(self):
        self.arrays = {}
        self.ragged = {}

    def put(self, arrays, rows, n, prefix=""):
        for key, value in arrays.items():
            a = value.cpu().numpy()
            if a.dtype == np.int64:
                a = a.astype(np.int32)
            field = prefix + key
            if field not in self.arrays:
                self.arrays[field] = np.empty((n, *a.shape[1:]), a.dtype)
            self.arrays[field][rows] = a

    def ties(self, top, candidates, positions, ids, prefix=""):
        if len(positions):
            r, j = torch.where(top.classes[positions] == 2)
            self.ragged.setdefault(prefix, []).append((ids[r.cpu().numpy()], candidates[j].cpu().numpy()))

    def finalize_ties(self, n):
        for prefix, chunks in self.ragged.items():
            rows, ids = np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])
            order = np.argsort(rows, kind="stable")
            self.arrays[prefix + "tie_indices"] = ids[order].astype(np.int32)
            self.arrays[prefix + "tie_indptr"] = np.r_[0, np.bincount(rows, minlength=n).cumsum()]

    def audit_arrays(self, ids):
        out = {}
        for prefix in ("", "screen_"):
            if prefix == "screen_":
                if "screen_query" not in self.arrays:
                    continue
                keep = np.flatnonzero(np.isin(self.arrays["screen_query"], ids))
            else:
                keep = np.asarray(ids)
            out[prefix + "query"] = self.arrays["screen_query"][keep] if prefix else keep
            for field in ("indices", "cutoff", "t_all", "t_in", "eligible", "origins", "gap_median", "same_token",
                          "overlap_counts", "overlap_expected", "overlap_maximum", "mass", "removed", "status", "beta", "dmin", "logz",
                          "display_distances"):
                if prefix + field in self.arrays:
                    out[prefix + field] = self.arrays[prefix + field][keep]
            ptr = self.arrays[prefix + "tie_indptr"]
            ties = self.arrays[prefix + "tie_indices"]
            out[prefix + "tie_indptr"] = np.r_[0, (ptr[keep + 1] - ptr[keep]).cumsum()]
            out[prefix + "tie_indices"] = np.concatenate([ties[ptr[i]:ptr[i + 1]] for i in keep])
        return out


def _origin_document_data(counts, shares, selected, documents):
    """Sufficient statistics for the same query-weighted estimator as `origin`."""
    n = len(documents)
    valid = (selected > 0).reshape(n, -1)
    sums = np.nansum(shares.reshape(n, -1, 3), axis=1)
    support = valid.sum(1)
    contributions = dict(document_ids=[int(i) for i in documents], query_counts=[valid.shape[1]] * n,
                         queries_with_neighbours=support.tolist(),
                         counts=counts.reshape(n, -1, 3).sum(1, dtype=np.int64).tolist(), share_sums=sums.tolist())
    # Same default_rng(seed).integers schedule as readout.bootstrap_indices.
    samples = np.random.default_rng(ORIGIN_BOOTSTRAP_SEED).integers(0, n, size=(ORIGIN_BOOTSTRAP_DRAWS, n))
    denom = support[samples].sum(1)
    have = denom > 0
    boot = sums[samples[have]].sum(1) / denom[have, None]
    limits = np.percentile(boot, [5, 95], axis=0).tolist() if len(boot) else [None, None]
    interval = dict(seed=ORIGIN_BOOTSTRAP_SEED, draws=ORIGIN_BOOTSTRAP_DRAWS, valid_draws=int(have.sum()),
                    p5=limits[0], p95=limits[1], estimator="mean origin share over queries with neighbours")
    return contributions, interval


def geometry_summary(fields, k, *, documents=None):
    """Means over queries, not a pooled pair-weighted average."""
    def mean(a):
        a = np.asarray(a)
        if a.ndim == 1:
            use = np.isfinite(a)
            return float(a[use].mean()) if use.any() else None
        return [mean(x) for x in a.T]

    def ratio(num, den):
        return np.divide(num, den, where=den != 0, out=np.full(num.shape, np.nan))

    eligible = fields["eligible"].sum(1)
    selected = fields["origins"].sum(1)
    shares = ratio(fields["origins"], selected[:, None])
    ans = dict(queries=len(selected), full_topk_queries=int((selected == k).sum()),
               origin=mean(shares),
               chance_origin=mean(ratio(fields["eligible"], eligible[:, None])),
               position_gap=mean(fields["gap_median"]), position_gap_defined=int(np.isfinite(fields["gap_median"]).sum()),
               same_token=mean(ratio(fields["same_token"], selected)),
               chance_same_token=mean(ratio(fields["same_token_eligible"], eligible)),
               overlap_expected=mean(fields["overlap_expected"] / k), overlap_maximum=mean(fields["overlap_maximum"] / k))
    if "stitch_observed" in fields:
        ans.update({name: mean(fields[name]) for name in ("stitch_observed", "stitch_baseline", "stitch_beyond")})
    if "screen_mass" in fields:
        ans.update(screen_mass=mean(fields["screen_mass"]),
                   screen_origin=mean(fields["screen_origins"] / k),
                   screen_status=np.bincount(fields["screen_status"], minlength=4).tolist(),
                   screen_overlap_expected=mean(fields["screen_overlap_expected"] / k),
                   screen_overlap_maximum=mean(fields["screen_overlap_maximum"] / k))
        if "screen_removed" in fields:
            ans["screen_removed"] = mean(fields["screen_removed"])
    if documents is not None:
        contributions, interval = _origin_document_data(fields["origins"], shares, selected, documents)
        ans.update(origin_by_document=contributions, origin_document_bootstrap=interval)
    return ans


def _membership(d, ok, k):
    return top_with_ties(d.masked_fill(~ok, torch.inf), k)


def _pair_sample(pool, model, reading, centring, name, count):
    key = json.dumps([model, reading, centring, name], separators=(",", ":")).encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
    size = pool.n * (pool.n - 1)
    rank = np.random.default_rng(seed).choice(size, min(count, size), replace=False)
    q, j = np.divmod(rank, pool.n - 1)
    j += j >= q
    order = q.argsort(kind="stable")
    return q[order], j[order]


@torch.no_grad()
def run_group(root, model, pool, inputs, reading, centring, *, device="cpu", memories=tuple(MEMORIES),
              masks=MASKS, k=64, window=64, target=64, row_block=128, screen=True, pair_count=1000,
              validate_models=True, on_setting=None, corpus_means=None, persist_lists=True, probe_sha256=None):
    """Complete one reading/centring group for one model or control collection.

    inputs is an ordered mapping name -> raw (context, position, feature)
    float array or a lazy zero-argument loader. If both adversaries are present,
    all inputs are overlapped with them; otherwise just local overlap is saved.
    on_setting(path) is called only once the NPZ is atomically complete. No
    calls to Modal or services are made here. Each input gets one Gram pass.
    """
    start_time = time.perf_counter()
    root = Path(root)
    if not inputs or k < 1 or row_block < 1 or pool.n < 2 or window < 0 or not 1 <= target:
        raise ValueError("invalid group parameters")
    if not memories or memories[0] != "none" or len(set(memories)) != len(memories):
        raise ValueError("memory 'none' must be first, for overlap references")
    if "other_texts" in masks and len(pool.texts) == 1:
        raise ValueError("other-texts mask requires more than one text")
    if screen and "window" in masks and ("none" not in masks or masks.index("none") > masks.index("window")):
        raise ValueError("screen needs the unmasked conditional before the windowed conditional")
    refs = tuple(x for x in REFERENCES[1:] if x in inputs)
    if refs and len(refs) != 2:
        raise ValueError("provide both adversaries or neither")
    names = (*refs, *(name for name in inputs if name not in refs))
    base = root / "census" / model
    marker = base / f"{reading}_{centring}" / "group.json"
    result = dict(schema=1, model=model, reading=reading, centring=centring, inputs=list(names), masks=list(masks),
                  memories=list(memories), k=k, target=target, window=window, tie_rtol=TIE_RTOL,
                  pair_count=pair_count, suffix_screen=screen, lists_saved=persist_lists,
                  origins=["other", "far", "near"], references=list(REFERENCES if refs else REFERENCES[:1]),
                  probe_sha256=probe_sha256)
    if marker.is_file():
        previous = json.loads(marker.read_bytes())
        if any(previous.get(key) != value for key, value in result.items()):
            raise ValueError(f"cannot mix configurations in {marker.parent}; use a separate output root")
    kernels, samples, mins, bad = {}, {}, {}, {}
    for name in names:
        value = inputs[name]
        y, mu = centre(value() if callable(value) else value, centring, reading, device,
                       corpus_mean=(corpus_means or {}).get(name))
        dup = None if name not in refs else tuple(torch.as_tensor(v[:, pool.start - 1:].reshape(-1), device=device)
                                                 for v in _duplicate_labels(pool.tokens, mu.cpu().numpy()))
        kernels[name] = Kernel(y, pool, reading, [MEMORIES[m] for m in memories], duplicate_labels=dup)
        if validate_models and name.startswith("site"):
            mins[name] = y.new_tensor(torch.inf)
        bad[name] = torch.zeros((), dtype=torch.bool, device=device)
        qp, jp = _pair_sample(pool, model, reading, centring, name, pair_count)
        samples[name] = qp, jp, np.empty((len(memories), len(qp))), np.empty((len(memories), len(qp)))
        del y, mu

    row_block = min(row_block, pool.n - 1)
    combos = tuple(itertools.product(memories, masks))
    outputs = {(name, *setting): _Output() for name in names for setting in combos}
    text, pos, tokens, suffix = (torch.as_tensor(v, device=device) for v in (pool.text, pool.position, pool.ids, pool.suffix))
    candidates = torch.arange(pool.n, device=device)
    positions = pool.position
    suffix_q = np.asarray([q for q in pool.query if positions[q] >= pool.suffix_start], dtype=np.int64) if screen else np.empty(0, dtype=np.int64)
    suffix_row = np.full(pool.n, -1, dtype=np.int64)
    suffix_row[suffix_q] = np.arange(len(suffix_q))
    is_q = np.isin(np.arange(pool.n), pool.query)
    is_audit = np.isin(np.arange(pool.n), pool.audit)
    if pool.source is not None:
        s = pool.source[pool.start - 1:]
        source, baseline = [torch.as_tensor(v, device=device) for v in (s, position_baseline(s))]

    for c in range(len(pool.texts)):
        for lo in range(0, pool.length, row_block):
            hi = min(lo + row_block, pool.length)
            i0 = c * pool.kept + max(lo - pool.start + 1, 0)
            i1 = c * pool.kept + max(hi - pool.start + 1, 0)
            if i0 == i1:
                for kernel in kernels.values():
                    kernel.tile(c, lo, hi)
                continue
            row_ids = np.arange(i0, i1)
            q = candidates[i0:i1]
            origin, gap = origin_codes(text, pos, q, candidates, window)
            elig = {mask: eligible_mask(origin, q, candidates, mask, ntexts=len(pool.texts)) for mask in masks}
            same = tokens[q, None] == tokens[candidates]
            chance = {m: chance_statistics(e, origin, same) for m, e in elig.items()}
            del same
            q_rows = np.flatnonzero(is_q[row_ids] | is_audit[row_ids])
            aq_rows = np.flatnonzero(is_audit[row_ids])
            si = np.flatnonzero(suffix_row[row_ids] >= 0)
            sq = q[si]
            srows = suffix_row[row_ids[si]]
            sa_rows = np.flatnonzero(is_audit[row_ids[si]])
            if len(si):
                so, sg = origin_codes(text, pos, sq, suffix, window)
                selig = {mask: eligible_mask(so, sq, suffix, mask, ntexts=len(pool.texts)) for mask in masks}
                same = tokens[sq, None] == tokens[suffix]
                schance = {m: chance_statistics(e, so, same) for m, e in selig.items()}
                del same
            references = {}
            for name in names:
                current = {}
                _, raw, distances = kernels[name].tile(c, lo, hi, raw_output=bool(pair_count))
                qp, jp, pair_raw, pair_dist = samples[name]
                left, right = np.searchsorted(qp, [i0, i1])
                if len(si):
                    # All masks and memories share a batched PR solve; each
                    # row still brackets and calibrates its own bandwidth.
                    suffix_d = torch.stack([d[si][:, suffix] for d in distances])
                    suffix_d[:, torch.arange(len(si), device=device), torch.searchsorted(suffix, sq)] = torch.inf
                    ok = torch.stack([selig[m] for m in masks])
                    sds = suffix_d[:, None].masked_fill(~ok[None], torch.inf)
                    sr, status, beta, dmin, logz = cond_rows(sds.flatten(0, 2), target)
                    sr = sr.reshape_as(sds)
                    screen_mass = torch.stack([(sr * (so == g)).sum(-1) for g in range(3)], dim=-1)
                    ss = status.reshape(sds.shape[:-1])
                    screen_mass[ss != 0] = torch.nan
                    screen_data = {"mass": screen_mass, "status": ss,
                                   **{k: v.reshape_as(ss) for k, v in (("beta", beta), ("dmin", dmin), ("logz", logz))}}
                for ai, mem in enumerate(memories):
                    d = distances[ai]
                    if left != right:
                        rows, js = qp[left:right] - i0, jp[left:right]
                        pair_raw[ai, left:right] = raw[ai][rows, js].cpu().numpy()
                        pair_dist[ai, left:right] = d[rows, js].cpu().numpy()
                    bad[name] |= ~torch.isfinite(d).all()
                    d[torch.arange(i1 - i0, device=device), q] = torch.inf
                    if name in mins:
                        mins[name] = torch.minimum(mins[name], d.min())
                    for mi, mask in enumerate(masks):
                        top = _membership(d, elig[mask], k)
                        displayed = _display_distances(top, d) if persist_lists else None
                        suffix_top = None
                        screen_row = None
                        if len(si):
                            suffix_top = top_with_ties(sds[ai, mi], k)
                            screen_row = {key: value[ai, mi] for key, value in screen_data.items()}
                            if mask == "window":
                                screen_row["removed"] = screen_mass[ai, masks.index("none"), :, 2]
                        current[mem, mask] = (top, suffix_top, screen_row, displayed)
                    # The distances are now reduced to audit pairs, suffix
                    # statistics, and the eight ordered values for the viewer.
                if name in refs:
                    references[name] = current
                    if name != refs[-1]:
                        continue
                    ready = references.items()
                else:
                    ready = ((name, current),)
                for ready_name, rows_by_setting in ready:
                    for mem, mask in combos:
                        top, suffix_top, screen_row, displayed = rows_by_setting[mem, mask]
                        other = (rows_by_setting["none", mask], *(references[r][mem, mask] for r in refs))
                        out = outputs[ready_name, mem, mask]
                        fields = _arrays(top, [r[0] for r in other], candidates)
                        if displayed is not None:
                            fields["display_distances"] = displayed
                        fields.update(statistics(top, origin, gap, elig[mask], q, candidates, tokens, chance=chance[mask]))
                        if pool.source is not None:
                            selected = fields["indices"]
                            j = selected.clamp_min(0)
                            valid = selected >= 0
                            denom = valid.sum(1).double().clamp_min(1.)
                            obs = ((source[q, None] == source[j]) & valid).sum(1) / denom
                            base_at_lag = (baseline[j - q[:, None] + pool.n - 1] * valid).sum(1) / denom
                            obs[~valid.any(1)] = torch.nan
                            base_at_lag[~valid.any(1)] = torch.nan
                            fields.update(stitch_observed=obs, stitch_baseline=base_at_lag, stitch_beyond=obs - base_at_lag)
                        out.put(fields, row_ids, pool.n)
                        keep = aq_rows if ready_name in refs else q_rows
                        out.ties(top, candidates, keep, row_ids[keep])
                        if suffix_top is not None:
                            fields = _arrays(suffix_top, [r[1] for r in other], suffix)
                            fields.update(statistics(suffix_top, so, sg, selig[mask], sq, suffix, tokens, chance=schance[mask]))
                            fields.update(screen_row, query=sq)
                            out.put(fields, srows, len(suffix_q), "screen_")
                            out.ties(suffix_top, suffix, sa_rows, srows[sa_rows], "screen_")
                del current, ready, rows_by_setting, distances, raw

    if any(bool(invalid) for invalid in bad.values()):
        raise ValueError("nonfinite unmasked distances")
    checked_min = {name: float(value) for name, value in mins.items()}
    if any(v <= 1e-9 for v in checked_min.values()):
        raise ValueError(f"model off-diagonal distance <= 1e-9: {checked_min}")
    result["checked_min"] = checked_min
    if persist_lists:
        result["display_distances"] = dict(DISPLAY_DISTANCE[reading], dtype="float32", columns=DISPLAY_K,
                                           missing="NaN", order="first eight indices in the saved top-list order")
    pool.save(base / "pool.npz")
    write_json(marker, dict(result, complete=False))
    audit_base = root / "audit_a" / "expected" / model / f"{reading}_{centring}"
    for name in names:
        qp, jp, raw, dist = samples[name]
        if pair_count:
            atomic(audit_base / f"{name}_pairs.npz", np.savez, query=qp, candidate=jp, kernel=raw,
                   distance=dist, memories=np.asarray(memories), diagonal=kernels[name].diag.cpu().numpy(),
                   scale=kernels[name].scale.cpu().numpy())
    summaries = []
    for (name, mem, mask), out in outputs.items():
        path = setting_path(root, model, reading, centring, name, mem, mask)
        if persist_lists:
            out.finalize_ties(pool.n)
            if pool.audit:
                atomic(audit_base / name / f"{mem}_{mask}.npz", np.savez, **out.audit_arrays(pool.audit))
            atomic(path, np.savez, **out.arrays)
        summaries.append(dict(input=name, memory=mem, mask=mask, **geometry_summary(out.arrays, k, documents=pool.texts)))
        if persist_lists and on_setting is not None:
            on_setting(path)
    write_json(root / "analysis" / "geometry" / model / f"{reading}_{centring}.json", summaries)
    result.update(complete=True, seconds=time.perf_counter() - start_time)
    write_json(marker, result)
    return result


def run_model(root, model, *, groups=PRIORITY, sites=SITES, device="cuda", row_block=128, on_setting=None,
              output_root=None):
    """UA census / A-pilot: run selected whole groups in prespecified priority."""
    groups = tuple(groups)
    if tuple(g for g in PRIORITY if g in groups) != groups:
        raise ValueError("groups must follow cosine C/O, dot C/O priority")
    probe_path = Path(root) / "probe" / "probe.json"
    data = probe_path.read_bytes()
    pool = pool_from_probe(json.loads(data))
    probe_sha256 = hashlib.sha256(data).hexdigest()
    inputs = {"random": lambda: random_vectors(pool.tokens),
              "embedding": lambda: load_site(root, model, pool, 0)}
    inputs.update({input_name(s): (lambda s=s: load_site(root, model, pool, s)) for s in sites})
    return [run_group(output_root or root, model, pool, inputs, reading, centring, device=device, row_block=row_block,
                      on_setting=on_setting, probe_sha256=probe_sha256) for reading, centring in groups]


def run_stitches(root, model, *, device="cpu", sites=SITES, readings=("cosine", "dot"), row_block=128):
    """§5D, every stitched text on its own; never use an other-texts mask."""
    probe = load_probe(root)
    real = pool_from_probe(probe)
    # §4 defines C against the 32 probe texts, even when the graph is one
    # stitched control. Reduce per file to avoid stacking every context in RAM.
    means = {input_name(site): np.mean([np.load(Path(root) / "acts" / model / f"c{c:02d}.npy", mmap_mode="r")[site, 1:]
                                       .mean(0, dtype=np.float64) for c in real.texts], axis=0)
             for site in sites}
    out = []
    for reading in readings:
        for context in probe["contexts"]:
            if context["kind"] != "stitched":
                continue
            c = context["id"]
            pool = pool_from_probe(probe, stitch=c)
            inputs = {input_name(s): (lambda s=s: load_site(root, model, pool, s)) for s in sites}
            out.append(run_group(root, f"{model}_stitch{c:02d}", pool, inputs, reading, "C", masks=MASKS[:2],
                                 device=device, row_block=row_block, screen=False, pair_count=0, corpus_means=means))
    return out
