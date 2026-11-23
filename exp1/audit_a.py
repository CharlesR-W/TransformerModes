#!/usr/bin/env python3
"""Independent Stage A audit (SPEC v3.2, section 12.4).

numpy and the standard library only; nothing from this repository is imported.
Every quantity the spec defines is recomputed here from its definition in the
plainest available form, sharing no code with the pipeline it checks:

* memory kernel: the explicit lag sum K_a(t,t') = sum_s a^s <y_{t-s}, y_{t'-s}>
  (section 4), evaluated lag by lag from a Gram block of the activations; the
  diagonals by the same explicit sum over squared norms. The pipeline's row
  recursion K[t] = G[t] + a K[t-1] and its doubling scan are not used;
* masks, the top-64 tie rule, t_all/t_in, expected and maximum overlaps from
  the sentences of section 4 (the maximum by enumerating z, not a closed form);
* the suffix-screen conditional by an independent bisection on the
  participation ratio (section 5B);
* matched controls by direct filtering of the census pool (section 5C);
* census statistics, the readout summaries and the Stage B selection from their
  definitions in section 5.

Where a stored file forces a convention the spec does not state (pool index
order, key names, seed derivations, the order of stored fields), the audit
follows the pipeline's documented convention and lists it in the report under
"conventions_trusted"; several of them are verified independently as well.

Interface (used by the Modal app):
    run(root, out, rows=None) -> dict      the audit; also writes `out` as JSON
    time_one_row(root, model) -> float      seconds for one explicit-lag row
    python audit_a.py --root ROOT --out OUT [--rows N]   (exit 1 on failure)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- spec values
MEMORY_A = {"none": 0.0, "2": 0.5, "8": 0.875, "64": 0.984375, "whole": 1.0}  # a = 1 - 1/m, section 4
GROUPS = (("cosine", "C"), ("cosine", "O"), ("dot", "C"), ("dot", "O"))      # section 11 priority
MASKS = ("none", "window", "other_texts")
AUDIT_SITES = (4, 12)                                                        # section 12.4
ORIGINS = ("other", "far", "near")
TOL_DIST = 1e-9          # section 12.4: distances
TOL_AGG = 1e-12          # recomputing a mean or sum of stored numbers
TOL_BETA_REL = 1e-7      # bisection bandwidth (not a distance; reported, gated loosely)
SPEC = dict(k=64, window=64, target=64, tie_rtol=1e-12, pair_count=1000,
            memories=["none", "2", "8", "64", "whole"], masks=list(MASKS),
            inputs=sorted(["random", "embedding"] + [f"site{s:02d}" for s in range(2, 17, 2)]),
            sites=list(range(2, 17, 2)), readout_sites=[4, 8, 12, 16], required_sites=5,
            min_matched=200, bootstrap_draws=200, texts=32, pool_start=256, pool_stop=2048, suffix_start=1536,
            q_suffix=64, q_early=32,
            thresholds=dict(other_mass_min=.25, removed_mass_max=.5, unmasked_near_mass_max=.25, overlap_max=.5))
CANDIDATE_MEMORIES = ("2", "8", "64", "whole")
INTERVENTION = (("C", "none"), ("C", "window"), ("O", "none"), ("O", "window"))

CONVENTIONS = [
    "pool index of a census token = (rank of its text among the probe's census_pool texts) * kept + (position - "
    "first pool position); the audit derives this layout from probe.json and checks it against census/<model>/pool.npz",
    "screen rows are the suffix queries of Q in probe order (screen_query); screen_tie_indptr is indexed by screen row",
    "stored list layout: strictly-inside members first, then the t_in taken tied members; -1 pads; tie ids "
    "(tie_indices/tie_indptr) hold every tied candidate for Q and audit rows of model sites, audit rows of adversaries",
    "overlap arrays are ordered by group.json 'references' (local, random, embedding) and counts as "
    "(|A_s&B_s|, |A_s&B_t|, |A_t&B_s|, |A_t&B_t|)",
    "screen status codes 0 calibrated, 1 fewer eligible than the target, 2 tied minima reach the target "
    "(exact equality), 3 no bracket; masses NaN when status != 0",
    "random-vector adversary: row v of default_rng(7).standard_normal((V, 2048)) drawn in token-id order",
    "control seed: SHA-256 of compact JSON [model, [site, reading, centring, memory, mask]], first 8 bytes "
    "little-endian, then x = splitmix64(x ^ field) for (query text, query position, neighbour text, neighbour "
    "position); rank = splitmix64 iterate rejected below 2^64 mod n, then mod n; the control is the rank-th "
    "admissible token in ascending position",
    "'i's other neighbours' excluded from controls = every strictly-inside member and every candidate tied at "
    "the cutoff (the expanded set of section 4)",
    "readout fields: sums weighted by P(j in top-64); 'mass_query'/'mass_neighbour' over all pairs, "
    "'mass_control', 'advantage' and top-1 fields over matched pairs; strata named '<origin>/<rare|frequent>'; "
    "coverage thresholds applied to these weighted sums",
    "bootstrap: default_rng(bootstrap_seed).integers(0, D, size=(draws, D)) over documents, percentiles by "
    "numpy's default (linear) rule",
    "pair sample of each pass: default_rng(int.from_bytes(sha256(JSON [model, reading, centring, input])[:16], "
    "'little')).choice(n(n-1), count, replace=False), decoded as (query, candidate != query), sorted by query",
    "X: a candidate's concordance is eligible only if every (model, readout site) setting is eligible; its score "
    "and bootstrap are means over (model, readout site) on the strata covered in all eligible candidates",
    "selection json, geometry json and readout json key names as written by select_b.py, census.py, readout.py",
]

NOT_REPRODUCED = [
    "The forward passes: activations, predictions and attention sums are taken as stored by extract.py "
    "(the hook-equality and causal checks are extraction's own).",
    "Probe selection (document choice, token assertion); the query sample and audit queries are checked for "
    "structure and, with spec parameters, replayed from their seeds.",
    "Complete distance rows are recomputed only for the audit queries at sites 4 and 12; other sites and the "
    "two adversaries are checked through the 1,000 saved pairs per pass and their stored diagonals.",
    "Adversary (random, embedding) neighbour lists are used as stored when recomputing overlaps with them.",
    "Membership of non-audit tokens: their statistics are recomputed from the stored lists (1% sample), "
    "not from distances.",
    "Suffix-screen conditional masses of non-audit suffix queries (the selection uses them as stored).",
    "Readout contributions of documents other than the sampled one per setting; the remaining documents enter "
    "only through the summaries recomputed from stored contributions.",
    "Stitched controls: lists are not recomputed from distances; their same-source share and position baseline "
    "are recomputed from the stored lists and the probe's segment schedule (1% sample).",
    "Synthetic census (5E), timescales (section 10), viewer data and the page.",
    "The order of strictly-inside members beyond a 1e-9 monotonicity check of recomputed distances.",
    "screen_logz (not a reported quantity).",
]


# --------------------------------------------------------------------- report
class Report:
    def __init__(self):
        self.entries = {}
        self.absent = []
        self.notes = []

    def add(self, name, scope, ok, deviation=None, tolerance=None, example=None, count=1):
        e = self.entries.setdefault((name, scope), dict(name=name, scope=scope, compared=0, failed=0,
                                                        worst_deviation=None, tolerance=tolerance, examples=[]))
        e["compared"] += int(count)
        if deviation is not None and not (isinstance(deviation, float) and math.isnan(deviation)):
            dev = float(deviation)
            e["worst_deviation"] = dev if e["worst_deviation"] is None else max(e["worst_deviation"], dev)
        if not ok:
            e["failed"] += 1
            if example is not None and len(e["examples"]) < 5:
                e["examples"].append(example)

    def close(self, name, scope, got, want, tol, example=None, scale=None):
        got = np.asarray(got, dtype=float)
        want = np.asarray(want, dtype=float)
        if got.shape != want.shape:
            self.add(name, scope, False, None, tol, dict(example or {}, problem=f"shape {got.shape} vs {want.shape}"))
            return False
        diff = np.abs(got - want)
        if scale is not None:
            diff = diff / np.maximum(np.abs(np.asarray(scale, dtype=float)), 1e-300)
        gn, wn = np.isnan(got), np.isnan(want)
        diff = np.where(gn & wn, 0.0, diff)
        diff = np.where(gn ^ wn, np.inf, diff)
        both_inf = np.isinf(got) & np.isinf(want) & (np.sign(got) == np.sign(want))
        diff = np.where(both_inf, 0.0, diff)
        worst = float(diff.max()) if diff.size else 0.0
        ok = worst <= tol
        ex = None
        if not ok:
            at = np.unravel_index(int(np.argmax(diff)), diff.shape) if diff.ndim else ()
            ex = dict(example or {}, index=[int(x) for x in at], got=_num(got[at]), want=_num(want[at]))
        self.add(name, scope, ok, worst if np.isfinite(worst) else None, tol, ex, count=max(diff.size, 1))
        return ok

    def rows(self, name, scope, got, want, tol, example, ids):
        """Row-wise comparison of (R, ...) arrays; NaN equals NaN; one entry per row compared."""
        got = np.asarray(got, dtype=float).reshape(len(ids), -1)
        want = np.asarray(want, dtype=float).reshape(len(ids), -1)
        diff = np.abs(got - want)
        gn, wn = np.isnan(got), np.isnan(want)
        diff = np.where(gn & wn, 0.0, np.where(gn ^ wn, np.inf, diff))
        diff = np.where(np.isinf(got) & np.isinf(want) & (np.sign(got) == np.sign(want)), 0.0, diff)
        per_row = diff.max(axis=1) if diff.shape[1] else np.zeros(len(ids))
        bad = np.flatnonzero(per_row > tol)
        e = self.entries.setdefault((name, scope), dict(name=name, scope=scope, compared=0, failed=0,
                                                        worst_deviation=None, tolerance=tol, examples=[]))
        e["compared"] += len(ids)
        finite = per_row[np.isfinite(per_row)]
        if len(finite):
            e["worst_deviation"] = max(e["worst_deviation"] or 0.0, float(finite.max()))
        e["failed"] += len(bad)
        for b in bad[:max(0, 5 - len(e["examples"]))]:
            e["examples"].append(dict(example, query=int(ids[b]), got=[_num(x) for x in got[b]],
                                      want=[_num(x) for x in want[b]]))

    def same(self, name, scope, got, want, example=None):
        ok = _canon(got) == _canon(want)
        self.add(name, scope, ok, None, "exact", None if ok else dict(example or {}, got=_canon(got), want=_canon(want)))
        return ok

    def require(self, name, scope, ok, example=None):
        self.add(name, scope, bool(ok), None, "required", None if ok else example)
        return bool(ok)

    def result(self):
        checks = sorted(self.entries.values(), key=lambda e: (e["name"], e["scope"]))
        for e in checks:
            e["passed"] = e["failed"] == 0
        return checks


def _num(x):
    x = float(x)
    return None if math.isnan(x) else ("inf" if x == math.inf else "-inf" if x == -math.inf else x)


def _canon(x):
    if isinstance(x, (set, frozenset)):
        return sorted(_canon(v) for v in x)
    if isinstance(x, np.ndarray):
        return [_canon(v) for v in x.tolist()] if x.ndim else _canon(x.item())
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) else ("inf" if x == math.inf else "-inf" if x == -math.inf else x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def log(msg):
    print(f"[audit_a {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ file I/O
class Npz:
    """Members of an .npz; members stored uncompressed (np.savez) are memory-mapped."""

    def __init__(self, path):
        self.path = Path(path)
        self._cache = {}
        with zipfile.ZipFile(self.path) as zf:
            self.infos = {i.filename[:-4]: i for i in zf.infolist() if i.filename.endswith(".npy")}

    def __contains__(self, key):
        return key in self.infos

    def keys(self):
        return list(self.infos)

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._read(key)
        return self._cache[key]

    def _read(self, key):
        info = self.infos[key]
        if info.compress_type == zipfile.ZIP_STORED:
            with open(self.path, "rb") as f:
                f.seek(info.header_offset)
                head = f.read(30)
                if head[:4] == b"PK\x03\x04":
                    name_len, extra_len = struct.unpack("<HH", head[26:30])
                    f.seek(info.header_offset + 30 + name_len + extra_len)
                    version = np.lib.format.read_magic(f)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0,
                              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is not None:
                        shape, fortran, dtype = reader(f)
                        if len(shape) and int(np.prod(shape)) > 0 and not dtype.hasobject:
                            return np.memmap(self.path, dtype=dtype, mode="r", offset=f.tell(), shape=shape,
                                             order="F" if fortran else "C")
        with np.load(self.path, allow_pickle=False) as z:
            return z[key]


def read_json(path):
    return json.loads(Path(path).read_bytes())


# -------------------------------------------------------------- token layout
class Pool:
    """Tokens (text, position) of a node set, in the pipeline's index order (see CONVENTIONS)."""

    def __init__(self, texts, start, stop, tokens_by_text, suffix_start=None, source=None):
        self.texts = [int(c) for c in texts]
        self.start, self.stop = int(start), int(stop)
        self.L = self.stop - 1                      # last position; positions 1..L exist
        self.kept = self.stop - self.start
        self.T = len(self.texts)
        self.n = self.T * self.kept
        self.rank = {c: r for r, c in enumerate(self.texts)}
        self.text = np.repeat(np.asarray(self.texts), self.kept)
        self.pos = np.tile(np.arange(self.start, self.stop), self.T)
        self.tokens = {c: np.asarray(tokens_by_text[c], dtype=np.int64) for c in self.texts}
        for c in self.texts:
            if len(self.tokens[c]) != self.L:
                raise ValueError(f"text {c}: {len(self.tokens[c])} tokens, expected {self.L}")
        self.tok = np.concatenate([self.tokens[c][self.start - 1:self.L] for c in self.texts])
        self.suffix_start = suffix_start
        self.suffix = np.flatnonzero(self.pos >= suffix_start) if suffix_start is not None else None
        self.source = source

    def index(self, c, p):
        if c not in self.rank or not self.start <= p < self.stop:
            raise ValueError(f"({c}, {p}) is not a pool token")
        return self.rank[c] * self.kept + p - self.start

    def origin(self, i, cand, window):
        """0 other text, 1 same text more than `window` away, 2 same text within `window` (section 5A)."""
        same = self.text[cand] == self.text[i]
        near = same & (np.abs(self.pos[cand] - self.pos[i]) <= window)
        return np.where(~same, 0, np.where(near, 2, 1))

    def origin_pairs(self, q, j, window):
        same = self.text[j] == self.text[q]
        return np.where(~same, 0, np.where(np.abs(self.pos[j] - self.pos[q]) <= window, 2, 1))

    def eligible(self, i, cand, mask, window):
        """Section 4 exclusion masks; a token is never its own neighbour."""
        same = self.text[cand] == self.text[i]
        if mask == "none":
            return cand != i
        if mask == "window":
            return ~(same & (np.abs(self.pos[cand] - self.pos[i]) <= window))
        if mask == "other_texts":
            return ~same
        raise ValueError(mask)


class Layout:
    def __init__(self, root):
        self.root = Path(root)
        self.probe = read_json(self.root / "probe" / "probe.json")
        ns = self.probe["node_sets"]
        self.contexts = {int(c["id"]): c for c in self.probe["contexts"]}
        ranges = ns["census_pool"]
        starts, stops = {r[1] for r in ranges}, {r[2] for r in ranges}
        if len(starts) != 1 or len(stops) != 1:
            raise ValueError("census pool ranges differ between texts")
        suffix_starts = {r[1] for r in ns["suffix"]}
        if len(suffix_starts) != 1 or [r[0] for r in ns["suffix"]] != [r[0] for r in ranges]:
            raise ValueError("unexpected suffix ranges")
        texts = [int(r[0]) for r in ranges]
        self.pool = Pool(texts, starts.pop(), stops.pop(),
                         {c: self.contexts[c]["token_ids"] for c in texts}, suffix_start=suffix_starts.pop())
        self.query_pairs = [tuple(map(int, q)) for q in ns["query_sample"]]
        self.audit_pairs = [tuple(map(int, q)) for q in ns["audit_queries"]]
        self.query = [self.pool.index(c, p) for c, p in self.query_pairs]
        self.audit = [self.pool.index(c, p) for c, p in self.audit_pairs]

    def stitch_pool(self, c):
        (cc, start, stop), = self.probe["node_sets"]["one_text"][str(c)]
        ctx = self.contexts[int(cc)]
        source = np.asarray([sp[0] for sp in ctx["source_positions"][1:]])   # positions 1..L
        return Pool([int(cc)], start, stop, {int(cc): ctx["token_ids"]}, source=source)


# ------------------------------------------------------------------ features
def load_input(root, model, pool, name, width_random=2048):
    """Raw float64 features (T, L, W) for positions 1..L of the pool's texts."""
    if name == "random":
        return random_token_vectors(np.stack([pool.tokens[c] for c in pool.texts]), width_random)
    site = 0 if name == "embedding" else int(name[4:])
    out = None
    for r, c in enumerate(pool.texts):
        a = np.load(Path(root) / "acts" / model / f"c{c:02d}.npy", mmap_mode="r")
        if a.ndim != 3 or a.shape[1] != pool.L + 1:
            raise ValueError(f"acts/{model}/c{c:02d}.npy has shape {a.shape}")
        if out is None:
            out = np.empty((pool.T, pool.L, a.shape[2]), dtype=np.float64)
        out[r] = a[site, 1:pool.L + 1]
        del a
    return out


def random_token_vectors(tokens, width):
    """Section 4 adversary (i): one fixed Gaussian vector per token id from default_rng(7), drawn in id order."""
    need = np.unique(tokens)
    table = np.empty((len(need), width))
    rng = np.random.default_rng(7)
    block = 4096
    for lo in range(0, int(need[-1]) + 1, block):
        hi = min(lo + block, int(need[-1]) + 1)
        draw = rng.standard_normal((hi - lo, width))
        sel = (need >= lo) & (need < hi)
        table[sel] = draw[need[sel] - lo]
    return table[np.searchsorted(need, tokens)]


def centred(X, centring):
    """Section 4: corpus mean over all texts and positions 1..L (C) or each text's own mean (O)."""
    mu = X.mean(axis=(0, 1), keepdims=True) if centring == "C" else X.mean(axis=1, keepdims=True)
    Y = X - mu
    norms = np.sqrt(np.einsum("tpw,tpw->tp", Y, Y))
    if not np.all(np.isfinite(norms)) or np.any(norms < 1e-6 * np.median(norms)):
        raise ValueError("near-zero or non-finite centred vector")
    return Y, norms


def explicit_diagonals(sq, alphas):
    """K_a(t,t) = sum_{s=0}^{t-1} a^s |y_{t-s}|^2 for every position t, by the explicit lag sum.

    sq: (T, L) squared norms <y_u, y_u> of positions u = 1..L. Written as sum_u W[t, u] sq[u] with the lag
    weights W[t, u] = a^(t-u) for u <= t and 0 otherwise. Returns (n_mem, T, L)."""
    T, L = sq.shape
    out = np.empty((len(alphas), T, L))
    for m, a in enumerate(alphas):
        out[m] = sq @ _lag_weights(L, float(a)).T
    return out


_LAG_WEIGHTS = {}


def _lag_weights(L, a):
    if (L, a) not in _LAG_WEIGHTS:
        lag = np.arange(L)[:, None] - np.arange(L)[None, :]      # s = t - u
        _LAG_WEIGHTS[L, a] = np.where(lag >= 0, a ** np.maximum(lag, 0).astype(float), 0.0)
    return _LAG_WEIGHTS[L, a]


def gram_block(Y, r, t):
    """G[u-1, c', u'-1] = <y_{c,u}, y_{c',u'}> for the query text's positions u = 1..t, every position u'."""
    T, L, W = Y.shape
    return (Y[r, :t] @ Y.reshape(T * L, W).T).reshape(t, T, L)


def explicit_lag_row(G, t, alphas, start, chunk=256):
    """K_a(t, t') = sum_{s>=0} a^s G(t-s, t'-s) for every pool candidate t' = start..L of every text.

    D[s, c', j] = G(t - s, t'_j - s), the inner product at lag s (zero once t'_j - s < 1, since y_u = 0 for
    u <= 0); then K_a = sum_s a^s D[s]. Lags are taken in chunks only to bound memory."""
    _, T, L = G.shape
    kept = L - start + 1
    K = np.zeros((len(alphas), T, kept))
    D = np.empty((min(chunk, t), T, kept))
    for s0 in range(0, t, chunk):
        S = min(chunk, t - s0)
        D[:S] = 0.0
        for k in range(S):
            s = s0 + k
            first = max(0, s + 1 - start)          # candidates with t' - s >= 1
            if first >= kept:
                continue
            D[k, :, first:] = G[t - 1 - s, :, start - 1 - s + first:L - s]
        weights = np.asarray(alphas)[:, None] ** np.arange(s0, s0 + S)[None, :]
        K += np.tensordot(weights, D[:S], axes=(1, 0))
    return K


def distances_from_kernel(K, Kq, Kc, reading, scale):
    """Section 4 readings: cosine 1 - K/sqrt(K(t,t)K(t',t')); dot [(K(t,t)+K(t',t'))/2 - K]/s."""
    if reading == "cosine":
        return 1.0 - K / np.sqrt(Kq[:, None] * Kc)
    return ((Kq[:, None] + Kc) / 2.0 - K) / scale[:, None]


# ------------------------------------------------------------ section 4 ties
def top_with_ties(cand, d, k, rtol):
    """cand: eligible pool indices in ascending (text, position) order; d: their distances.

    Section 4: d_64 is the k-th smallest distance; tied = within rtol (relative) of d_64; the set is every
    candidate strictly closer plus the first tied candidates in (text, position) order needed to reach k."""
    if len(cand) < k:
        return dict(strict=np.sort(cand), ties=cand[:0], taken=cand[:0], d64=math.inf, t_all=0, t_in=0)
    d64 = float(np.partition(d, k - 1)[k - 1])
    tied = np.abs(d - d64) <= rtol * np.maximum(np.abs(d), abs(d64))
    strict = (d < d64) & ~tied
    t_in = k - int(strict.sum())
    ties = cand[tied]
    return dict(strict=cand[strict], ties=ties, taken=ties[:t_in], d64=d64, t_all=int(tied.sum()), t_in=t_in)


def max_overlap(ss, c1, c2, c3, a, b):
    """Section 4: |A_s&B_s| + max over z = 0..min(c3, a, b) of [z + min(c1, b - z) + min(c2, a - z)]."""
    return ss + max(z + min(c1, b - z) + min(c2, a - z) for z in range(0, min(c3, a, b) + 1))


def overlap(A, B):
    """A, B: dicts with strict (S), ties (all tied at the cutoff), t_in, t_all."""
    As, At, Bs, Bt = set(A["strict"].tolist()), set(A["ties"].tolist()), set(B["strict"].tolist()), set(B["ties"].tolist())
    ss, c1, c2, c3 = len(As & Bs), len(As & Bt), len(At & Bs), len(At & Bt)
    p = A["t_in"] / A["t_all"] if A["t_all"] else 0.0
    q = B["t_in"] / B["t_all"] if B["t_all"] else 0.0
    expected = ss + c1 * q + c2 * p + c3 * p * q           # sum_j P(j in A) P(j in B)
    return [ss, c1, c2, c3], expected, max_overlap(ss, c1, c2, c3, A["t_in"], B["t_in"])


def overlap_values(counts, a, a_all, b, b_all, k):
    """Rows of (|A_s&B_s|, c1, c2, c3) with cutoff sizes: expected overlap sum_j P(j in A) P(j in B), and the
    section 4 maximum by enumerating every admissible z = 0..min(c3, a, b)."""
    counts = np.asarray(counts, dtype=np.int64).reshape(-1, 4)
    ss, c1, c2, c3 = counts.T
    p = np.where(a_all > 0, a / np.maximum(a_all, 1), 0.0)
    q = np.where(b_all > 0, b / np.maximum(b_all, 1), 0.0)
    expected = ss + c1 * q + c2 * p + c3 * p * q
    z = np.arange(k + 1)[None, :]
    f = z + np.minimum(c1[:, None], b[:, None] - z) + np.minimum(c2[:, None], a[:, None] - z)
    f = np.where(z <= np.minimum(c3, np.minimum(a, b))[:, None], f, -10 ** 9)
    return expected, ss + f.max(axis=1)


def conditional_row(d, target):
    """Section 5B: q(j) proportional to exp(-beta (d_j - min d)), beta with 1 / sum q^2 = target, by bisection."""
    n = len(d)
    if n < target:
        return dict(status=1, q=None, beta=math.nan, dmin=math.nan)
    dmin = float(d.min())
    delta = d - dmin
    if n == target:
        return dict(status=0, q=np.full(n, 1.0 / n), beta=0.0, dmin=dmin)
    if int((delta == 0).sum()) >= target:       # participation ratio tends to the number of minima
        return dict(status=2, q=None, beta=math.nan, dmin=dmin)

    def participation(beta):
        w = np.exp(-beta * delta)
        return w.sum() ** 2 / (w * w).sum()

    lo, hi = 0.0, 1.0
    for _ in range(2000):
        if participation(hi) <= target:
            break
        lo, hi = hi, hi * 2.0
    else:
        return dict(status=3, q=None, beta=math.nan, dmin=dmin)
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        if participation(mid) > target:
            lo = mid
        else:
            hi = mid
    beta = 0.5 * (lo + hi)
    w = np.exp(-beta * delta)
    return dict(status=0, q=w / w.sum(), beta=beta, dmin=dmin)


def list_statistics(pool, i, members, window):
    """Section 5A per-query census quantities of a stored or recomputed top-64 list."""
    members = np.asarray(members, dtype=np.int64)
    o = pool.origin(i, members, window)
    gaps = np.abs(pool.pos[members] - pool.pos[i])[o != 0]
    return dict(origins=np.bincount(o, minlength=3), same_token=int((pool.tok[members] == pool.tok[i]).sum()),
                gap_median=float(np.median(gaps)) if len(gaps) else math.nan)


def eligible_statistics(pool, i, cand, ok, window):
    c = cand[ok]
    return dict(eligible=np.bincount(pool.origin(i, c, window), minlength=3),
                same_token_eligible=int((pool.tok[c] == pool.tok[i]).sum()))


# ------------------------------------------------------- stored list access
def stored_row(F, i, prefix="", need_ties=False):
    ind = np.asarray(F[prefix + "indices"][i]).astype(np.int64)
    valid = ind[ind >= 0]
    t_in, t_all = int(F[prefix + "t_in"][i]), int(F[prefix + "t_all"][i])
    ns = len(valid) - t_in
    row = dict(list=valid, strict=valid[:ns], taken=valid[ns:], t_in=t_in, t_all=t_all,
               d64=float(F[prefix + "cutoff"][i]) if prefix + "cutoff" in F else math.nan, ties=None)
    if prefix + "tie_indptr" in F:
        ptr = F[prefix + "tie_indptr"]
        if i + 1 < len(ptr):
            ids = np.asarray(F[prefix + "tie_indices"][int(ptr[i]):int(ptr[i + 1])]).astype(np.int64)
            if len(ids) == t_all:
                row["ties"] = ids
    if row["ties"] is None and t_all == t_in:
        row["ties"] = row["taken"]          # every tied candidate is in the list
    return row


# ------------------------------------------------------ section 5C controls
M64 = (1 << 64) - 1


def splitmix64(x):
    z = (x + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def setting_key(model, setting):
    data = json.dumps([model, list(setting)], separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def keyed_rank(key, fields, n):
    x = key
    for f in fields:
        x = splitmix64(x ^ (int(f) & M64))
    threshold = (1 << 64) % n
    bits = splitmix64(x)
    while bits < threshold:
        bits = splitmix64(bits)
    return bits % n


def lag_bin(gap):
    """1-8, 9-64, 65-512, over 512 (section 5C); lag 0 is in no bin."""
    gap = np.asarray(gap)
    return np.where(gap <= 0, -1, np.where(gap <= 8, 0, np.where(gap <= 64, 1, np.where(gap <= 512, 2, 3))))


def _bin(gap):
    return -1 if gap <= 0 else 0 if gap <= 8 else 1 if gap <= 64 else 2 if gap <= 512 else 3


class Controls:
    def __init__(self, pool):
        self.pool = pool
        self.text, self.pos, self.tok = (a.tolist() for a in (pool.text, pool.pos, pool.tok))
        self.buckets = {}
        for idx in np.lexsort((pool.pos, pool.tok, pool.text)).tolist():   # ascending position in each bucket
            self.buckets.setdefault((self.text[idx], self.tok[idx]), []).append(idx)
        ids, counts = np.unique(pool.tok, return_counts=True)
        self.frequency = counts[np.searchsorted(ids, pool.tok)]

    def draw(self, key, i, j, excluded):
        """Section 5C: same text and current token as j; same |lag| bin if j is in i's text; not i, j or i's
        neighbours (`excluded`, a set). Returns (control, reason): reasons 1 no same-token token in that text,
        2 empty lag bin, 3 all candidates excluded."""
        text, pos = self.text, self.pos
        cand = [h for h in self.buckets[text[j], self.tok[j]] if h != j]
        if not cand:
            return -1, 1
        if text[i] == text[j]:
            want = _bin(abs(pos[j] - pos[i]))
            cand = [h for h in cand if _bin(abs(pos[h] - pos[i])) == want]
            if not cand:
                return -1, 2
        cand = [h for h in cand if h not in excluded]
        if not cand:
            return -1, 3
        return cand[keyed_rank(key, (text[i], pos[i], text[j], pos[j]), len(cand))], 0


def expand(row):
    """Section 4/5C: strictly-inside members with weight 1, every tied candidate with t_in / t_all."""
    pairs = [(int(j), 1.0) for j in row["strict"]]
    if row["t_all"]:
        pairs += [(int(j), row["t_in"] / row["t_all"]) for j in row["ties"]]
    return pairs


def prediction_overlap(ids_a, p_a, ids_b, p_b):
    """o(i,j) = sum over v in both top-20 sets of min(p_i(v), p_j(v))."""
    eq = ids_a[:, :, None] == ids_b[:, None, :]
    return np.where(eq, np.minimum(p_a[:, :, None], p_b[:, None, :]), 0.0).sum(axis=(1, 2))


def readout_summary(by_doc, fields, reasons, min_matched, seed, draws):
    """Section 5C summary from per-document contributions."""
    col = {str(f): n for n, f in enumerate(fields)}
    total = by_doc.sum(0)
    matched = total[:, col["matched"]]
    pairs = matched + sum(total[:, col[str(r)]] for r in reasons[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(pairs > 0, matched / np.where(pairs > 0, pairs, 1), math.nan)
        mean_adv = np.where(matched > 0, total[:, col["advantage"]] / np.where(matched > 0, matched, 1), math.nan)
        top_n = np.where(matched > 0, total[:, col["top1_neighbour"]] / np.where(matched > 0, matched, 1), math.nan)
        top_c = np.where(matched > 0, total[:, col["top1_control"]] / np.where(matched > 0, matched, 1), math.nan)
    absent = pairs == 0
    covered = (pairs > 0) & (matched >= min_matched) & (ratio >= 0.5)
    coverage = float(matched.sum() / pairs.sum()) if pairs.sum() > 0 else 0.0
    eligible = bool(covered.any() and np.all(covered | absent) and coverage >= 0.5)
    D = by_doc.shape[0]
    resample = np.random.default_rng(seed).integers(0, D, size=(draws, D))
    boot = np.full(draws, math.nan)
    if covered.any():
        for b in range(draws):
            t = by_doc[resample[b]].sum(0)[covered]
            with np.errstate(invalid="ignore", divide="ignore"):
                boot[b] = np.mean(t[:, col["advantage"]] / t[:, col["matched"]])
    good = boot[np.isfinite(boot)]
    return dict(eligible=eligible, coverage=coverage, covered=covered, absent=absent, pairs=pairs, matched=matched,
                mean_advantage=mean_adv,
                score=float(mean_adv[covered].mean()) if covered.any() else None,
                top1_neighbour=float(top_n[covered].mean()) if covered.any() else None,
                top1_control=float(top_c[covered].mean()) if covered.any() else None,
                percentiles=np.percentile(good, [5, 50, 95]).tolist() if len(good) else None)


# ---------------------------------------------------------------- the audit
def _setting_path(root, model, reading, centring, name, memory, mask):
    return Path(root) / "census" / model / f"{reading}_{centring}" / name / f"{memory}_{mask}.npz"


def _readout_path(root, model, reading, centring, site, memory, mask):
    return Path(root) / "readout" / model / f"{reading}_{centring}" / f"site{site:02d}" / f"{memory}_{mask}.npz"


class Audit:
    def __init__(self, root, rows=None, spec_parameters=True):
        self.root = Path(root)
        self.rep = Report()
        self.spec_parameters = spec_parameters
        self.layout = Layout(root)
        self.pool = self.layout.pool
        self.rows = len(self.layout.audit) if rows is None else max(0, min(int(rows), len(self.layout.audit)))
        self.audit_rows = self.layout.audit[:self.rows]
        self.configs = {}
        self.files = {}

    # ------------------------------------------------------------ helpers
    def npz(self, path):
        path = Path(path)
        if path not in self.files:
            self.files[path] = Npz(path) if path.is_file() else None
        return self.files[path]

    def spec_check(self, name, scope, got, want):
        """Parameters the spec fixes. With spec_parameters=False (small fixtures) a deviation is a note."""
        if _canon(got) == _canon(want):
            self.rep.add("spec.parameters", scope, True, None, "exact")
        elif self.spec_parameters:
            self.rep.add("spec.parameters", scope, False, None, "exact", dict(parameter=name, got=_canon(got),
                                                                             spec=_canon(want)))
        else:
            self.rep.notes.append(f"{scope}: {name} = {_canon(got)} (spec {_canon(want)}); small-fixture run")

    # --------------------------------------------------------------- plan
    def plan(self):
        rep = self.rep
        plan_path = self.root / "analysis" / "stage_a_plan.json"
        census = self.root / "census"
        discovered = sorted(p.name for p in census.iterdir() if p.is_dir() and "_stitch" not in p.name
                            and not p.name.startswith("synthetic")) if census.is_dir() else []
        if plan_path.is_file():
            plan = read_json(plan_path)
            groups = [tuple(g) for g in plan["approved_groups"]]
            models = list(plan.get("models") or self.layout.probe.get("models") or discovered)
            dropped = [tuple(g) for g in plan.get("dropped_groups", []) if tuple(g) not in groups]
            dropped += [g for g in GROUPS if g not in groups and g not in dropped]
            self.planned = True
            rep.same("plan.priority_order", "plan", groups, [g for g in GROUPS if g in groups])
            for g in dropped:
                rep.absent.append(dict(what=f"group {g[0]}_{g[1]}", reason="absent by plan (dropped by the spending cap)"))
        else:
            self.planned = False
            models = discovered
            groups = [g for g in GROUPS if any((census / m / f"{g[0]}_{g[1]}" / "group.json").is_file() for m in models)]
            rep.notes.append("no analysis/stage_a_plan.json: audited every group present on disk " + str(groups))
            for g in GROUPS:
                if g not in groups:
                    rep.absent.append(dict(what=f"group {g[0]}_{g[1]}", reason="not present (no plan file)"))
        if self.spec_parameters:
            self.spec_check("models", "plan", sorted(models), ["olmo", "pythia"])
        self.models, self.groups = models, groups
        for model in models:
            for reading, centring in groups:
                scope = f"{model}/{reading}_{centring}"
                path = census / model / f"{reading}_{centring}" / "group.json"
                cfg = read_json(path) if path.is_file() else None
                ok = cfg is not None and cfg.get("complete") is True
                if self.planned:
                    rep.require("presence.group_complete", scope, ok, dict(path=str(path)))
                elif not ok:
                    rep.absent.append(dict(what=scope, reason="group.json missing or incomplete"))
                if cfg is not None:
                    self.configs[model, reading, centring] = cfg
        return models, groups

    # ----------------------------------------------------- configuration
    def configuration(self):
        rep, L = self.rep, self.layout
        pool = self.pool
        # Probe structure (section 3)
        per_text = {}
        for c, p in L.query_pairs:
            per_text.setdefault(c, []).append(p)
        q_ok = len(set(L.query_pairs)) == len(L.query_pairs) and set(per_text) == set(pool.texts)
        rep.require("probe.query_sample_valid", "probe", q_ok)
        audit_texts = [c for c, _ in L.audit_pairs]
        rep.same("probe.one_audit_query_per_text", "probe", sorted(audit_texts), sorted(pool.texts))
        for c, p in L.audit_pairs:
            rep.require("probe.audit_query_in_suffix_of_Q", "probe",
                        p in per_text.get(c, []) and p >= pool.suffix_start, dict(text=c, position=p))
        if self.spec_parameters:
            self.spec_check("texts", "probe", pool.T, SPEC["texts"])
            self.spec_check("pool range", "probe", [pool.start, pool.stop], [SPEC["pool_start"], SPEC["pool_stop"]])
            self.spec_check("suffix start", "probe", pool.suffix_start, SPEC["suffix_start"])
            for c in pool.texts:
                late = [p for p in per_text[c] if p >= pool.suffix_start]
                early = [p for p in per_text[c] if p < pool.suffix_start]
                self.spec_check("query sample sizes", f"probe/text{c}", [len(late), len(early)],
                                [SPEC["q_suffix"], SPEC["q_early"]])
            # Replay of the documented draws (convention: probe.node_sets)
            q_rng, a_rng = np.random.default_rng(21), np.random.default_rng(22)
            replay_q, replay_a = [], []
            for c in range(SPEC["texts"]):
                late = q_rng.choice(pool.stop - pool.suffix_start, size=SPEC["q_suffix"], replace=False) + pool.suffix_start
                early = q_rng.choice(pool.suffix_start - pool.start, size=SPEC["q_early"], replace=False) + pool.start
                replay_q += [(c, int(p)) for p in (*late, *early)]
                replay_a.append((c, int(a_rng.choice(late))))
            rep.same("probe.query_draws_replay", "probe", L.query_pairs, replay_q)
            rep.same("probe.audit_draws_replay", "probe", L.audit_pairs, replay_a)
        rep.add("audit.rows", "probe", True, None, None, dict(rows=self.rows, of=len(L.audit)), count=self.rows)
        if self.rows < len(L.audit):
            rep.notes.append(f"{self.rows} of {len(L.audit)} audit queries recomputed in full (rows={self.rows})")
        # Census configuration and pool layout
        for (model, reading, centring), cfg in self.configs.items():
            scope = f"{model}/{reading}_{centring}"
            if self.spec_parameters:
                for key in ("k", "window", "target", "tie_rtol", "pair_count", "memories", "masks"):
                    self.spec_check(key, scope, cfg.get(key), SPEC[key])
                self.spec_check("inputs", scope, sorted(cfg.get("inputs", [])), SPEC["inputs"])
            unknown = [m for m in cfg["memories"] if m not in MEMORY_A]
            rep.require("config.memories_known", scope, not unknown, dict(unknown=unknown))
            if self.planned:
                for name in cfg["inputs"]:
                    for mem in cfg["memories"]:
                        for mask in cfg["masks"]:
                            path = _setting_path(self.root, model, reading, centring, name, mem, mask)
                            rep.require("presence.setting_file", scope, path.is_file(), dict(path=str(path)))
        for model in self.models:
            path = self.root / "census" / model / "pool.npz"
            if path.is_file():
                with np.load(path) as f:
                    scope = f"{model}/pool"
                    rep.same("layout.pool_text", scope, f["text"], pool.text)
                    rep.same("layout.pool_position", scope, f["position"], pool.pos)
                    rep.same("layout.pool_token", scope, f["token"], pool.tok)
                    rep.same("layout.query", scope, f["query"], L.query)
                    rep.same("layout.audit", scope, f["audit"], L.audit)
                    rep.same("layout.suffix", scope, f["suffix"], pool.suffix)
            elif any(m == model for m, _, _ in self.configs):
                rep.require("presence.pool_npz", model, False, dict(path=str(path)))

    # ------------------------------------------------- per-input recompute
    def inputs_of(self, model):
        names = set()
        for (m, r, c), cfg in self.configs.items():
            if m == model:
                names.update(cfg["inputs"])
        order = ["random", "embedding"] + [f"site{s:02d}" for s in range(0, 64)]
        return [n for n in order if n in names]

    def recompute(self):
        rep = self.rep
        for model in self.models:
            names = self.inputs_of(model)
            present_sites = {int(n[4:]) for n in names if n.startswith("site")}
            for s in AUDIT_SITES:
                if s not in present_sites:
                    rep.absent.append(dict(what=f"{model}/site{s:02d}", reason="audit site not in census inputs"))
            for name in names:
                groups = [(r, c) for r, c in self.groups if (model, r, c) in self.configs
                          and name in self.configs[model, r, c]["inputs"]]
                if not groups:
                    continue
                t0 = time.perf_counter()
                X = load_input(self.root, model, self.pool, name)
                is_audit_site = name.startswith("site") and int(name[4:]) in AUDIT_SITES
                for centring in ("C", "O"):
                    readings = [r for r, c in groups if c == centring]
                    if not readings:
                        continue
                    Y, norms = centred(X, centring)
                    state = {}
                    for reading in readings:
                        cfg = self.configs[model, reading, centring]
                        alphas = np.asarray([MEMORY_A[m] for m in cfg["memories"]])
                        sq = norms ** 2 if reading == "dot" else (norms / norms) ** 2
                        diag = explicit_diagonals(sq, alphas)
                        pool_diag = diag[:, :, self.pool.start - 1:].reshape(len(alphas), -1)
                        state[reading] = dict(cfg=cfg, alphas=alphas, diag=diag, pool_diag=pool_diag,
                                              scale=pool_diag.mean(axis=1))
                        self.check_pairs(model, reading, centring, name, Y, norms, state[reading])
                    if is_audit_site and self.audit_rows:
                        self.audit_site_rows(model, centring, name, Y, norms, state)
                    del Y
                del X
                log(f"{model}/{name}: pairs{' and rows' if is_audit_site else ''} in {time.perf_counter() - t0:.1f}s")

    def check_pairs(self, model, reading, centring, name, Y, norms, st):
        rep, pool = self.rep, self.pool
        scope = f"{model}/{reading}_{centring}"
        path = self.root / "audit_a" / "expected" / model / f"{reading}_{centring}" / f"{name}_pairs.npz"
        cfg = st["cfg"]
        if not path.is_file():
            if cfg.get("pair_count", 0):
                rep.require("presence.pairs_file", scope, False, dict(path=str(path)))
            return
        with np.load(path) as f:
            q, j = f["query"].astype(np.int64), f["candidate"].astype(np.int64)
            kernel, dist = f["kernel"], f["distance"]
            memories = [str(m) for m in f["memories"]]
            rep.same("pairs.memories", scope, memories, cfg["memories"], dict(input=name))
            ex = dict(input=name)
            rep.close("pairs.diagonal", scope, f["diagonal"], st["pool_diag"], TOL_DIST, ex, scale=st["pool_diag"])
            rep.close("pairs.scale", scope, f["scale"], st["scale"], TOL_DIST, ex, scale=st["scale"])
        # Replay of the documented pair sample (convention), and validity of the pairs
        n = pool.n
        key = json.dumps([model, reading, centring, name], separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "little")
        count = int(cfg.get("pair_count", len(q)))
        rank = np.random.default_rng(seed).choice(n * (n - 1), min(count, n * (n - 1)), replace=False)
        rq, rj = np.divmod(rank, n - 1)
        rj = rj + (rj >= rq)
        order = np.argsort(rq, kind="stable")
        rep.same("pairs.sample_replay", scope, [q.tolist(), j.tolist()], [rq[order].tolist(), rj[order].tolist()],
                 dict(input=name))
        alphas = st["alphas"]
        K = np.empty((len(alphas), len(q)))
        for p, (a, b) in enumerate(zip(q, j)):
            ra, rb = pool.rank[int(pool.text[a])], pool.rank[int(pool.text[b])]
            ta, tb = int(pool.pos[a]), int(pool.pos[b])
            lags = min(ta, tb)
            A = Y[ra, ta - lags:ta][::-1]               # y_{t-s}, s = 0..lags-1
            B = Y[rb, tb - lags:tb][::-1]               # y_{t'-s}
            g = np.einsum("sw,sw->s", A, B)
            if reading == "cosine":
                g = g / (norms[ra, ta - lags:ta][::-1] * norms[rb, tb - lags:tb][::-1])
            K[:, p] = (alphas[:, None] ** np.arange(lags)[None, :]) @ g
        Kq, Kc = st["pool_diag"][:, q], st["pool_diag"][:, j]
        d = (1.0 - K / np.sqrt(Kq * Kc)) if reading == "cosine" else ((Kq + Kc) / 2.0 - K) / st["scale"][:, None]
        rep.close("pairs.kernel", scope, kernel, K, TOL_DIST, dict(input=name), scale=np.sqrt(Kq * Kc))
        rep.close("pairs.distance", scope, dist, d, TOL_DIST, dict(input=name))

    def audit_site_rows(self, model, centring, name, Y, norms, state):
        pool = self.pool
        for i in self.audit_rows:
            t0 = time.perf_counter()
            r, t = pool.rank[int(pool.text[i])], int(pool.pos[i])
            G = gram_block(Y, r, t)
            for reading in ("dot", "cosine"):
                if reading not in state:
                    continue
                if reading == "cosine":
                    G /= norms[r, :t][:, None, None]       # G of unit vectors, in place
                    G /= norms[None, :, :]
                st = state[reading]
                K = explicit_lag_row(G, t, st["alphas"], pool.start).reshape(len(st["alphas"]), -1)
                Kq = st["diag"][:, r, t - 1]
                self.rep.close("row.self_kernel_vs_diagonal", f"{model}/{reading}_{centring}", K[:, i], Kq,
                               TOL_DIST, dict(input=name, query=int(i)), scale=Kq)
                d = distances_from_kernel(K, Kq, st["pool_diag"], reading, st["scale"])
                self.compare_query(model, reading, centring, name, i, d, st["cfg"])
            del G
            log(f"{model}/{name}/{centring} row {i}: {time.perf_counter() - t0:.1f}s")

    def row_results(self, i, d, cfg):
        """Everything section 12.4 recomputes for one query at one (input, reading, centring)."""
        pool = self.pool
        k, w, target, rtol = cfg["k"], cfg["window"], cfg["target"], cfg["tie_rtol"]
        cand_all = np.arange(pool.n)
        out = {}
        for m, mem in enumerate(cfg["memories"]):
            for mask in cfg["masks"]:
                ok = pool.eligible(i, cand_all, mask, w)
                cand = cand_all[ok]
                top = top_with_ties(cand, d[m, cand], k, rtol)
                top["d"] = d[m]
                top.update(eligible_statistics(pool, i, cand_all, ok, w))
                top.update(list_statistics(pool, i, np.r_[top["strict"], top["taken"]], w))
                res = dict(full=top)
                if cfg.get("suffix_screen", True) and pool.suffix is not None and pool.pos[i] >= pool.suffix_start:
                    sc = pool.suffix
                    sok = pool.eligible(i, sc, mask, w)
                    scand = sc[sok]
                    stop = top_with_ties(scand, d[m, scand], k, rtol)
                    stop["d"] = d[m]
                    stop.update(eligible_statistics(pool, i, sc, sok, w))
                    stop.update(list_statistics(pool, i, np.r_[stop["strict"], stop["taken"]], w))
                    cond = conditional_row(d[m, scand], target)
                    mass = np.full(3, math.nan)
                    if cond["status"] == 0:
                        mass = np.bincount(pool.origin(i, scand, w), weights=cond["q"], minlength=3)
                    stop.update(mass=mass, status=cond["status"], beta=cond["beta"], dmin=cond["dmin"])
                    res["screen"] = stop
                out[mem, mask] = res
            if "window" in cfg["masks"] and "none" in cfg["masks"] and "screen" in out[mem, "none"]:
                out[mem, "window"]["screen"]["removed"] = out[mem, "none"]["screen"]["mass"][2]
        return out

    def compare_query(self, model, reading, centring, name, i, d, cfg):
        rep = self.rep
        scope = f"{model}/{reading}_{centring}"
        mine = self.row_results(i, d, cfg)
        refs = cfg.get("references", ["local"])
        expected_dir = self.root / "audit_a" / "expected" / model / f"{reading}_{centring}"
        for (mem, mask), res in mine.items():
            label = dict(input=name, memory=mem, mask=mask, query=int(i))
            F = self.npz(_setting_path(self.root, model, reading, centring, name, mem, mask))
            E = self.npz(expected_dir / name / f"{mem}_{mask}.npz")
            if F is None and E is None:
                rep.require("presence.setting_file", scope, False, label)
                continue
            if F is not None and E is not None:
                self.compare_copies(scope, label, F, E, i)
            src = F if F is not None else E
            row = i if F is not None else int(np.flatnonzero(np.asarray(E["query"]) == i)[0])
            for prefix, part in (("", "full"), ("screen_", "screen")):
                if part not in res:
                    continue
                if prefix and prefix + "query" not in src:
                    rep.require("presence.screen_rows", scope, False, label)
                    continue
                srow = row if not prefix else int(np.flatnonzero(np.asarray(src["screen_query"]) == i)[0])
                ref_rows = []
                for ref in refs:
                    if ref == "local":
                        ref_rows.append(mine[cfg["memories"][0], mask][part])
                        continue
                    R = self.npz(_setting_path(self.root, model, reading, centring, ref, mem, mask))
                    if R is None:
                        R = self.npz(expected_dir / ref / f"{mem}_{mask}.npz")
                        rr = int(np.flatnonzero(np.asarray(R["query"]) == i)[0]) if R is not None else None
                    else:
                        rr = i
                    if R is None:
                        ref_rows.append(None)
                        continue
                    if prefix:
                        rr = int(np.flatnonzero(np.asarray(R["screen_query"]) == i)[0])
                    ref_rows.append(stored_row(R, rr, prefix))
                self.compare_list(scope, dict(label, part=part), src, srow, prefix, res[part], ref_rows, refs)

    def compare_copies(self, scope, label, F, E, i):
        """audit_a/expected holds copies of the audit queries' rows; they must equal the census lists."""
        rows = {"": (int(np.flatnonzero(np.asarray(E["query"]) == i)[0]), i)}
        if "screen_query" in E and "screen_query" in F:
            hit_e = np.flatnonzero(np.asarray(E["screen_query"]) == i)
            hit_f = np.flatnonzero(np.asarray(F["screen_query"]) == i)
            if len(hit_e) and len(hit_f):
                rows["screen_"] = (int(hit_e[0]), int(hit_f[0]))
        for key in E.keys():
            prefix = "screen_" if key.startswith("screen_") else ""
            if key in (prefix + "query", prefix + "tie_indptr", prefix + "tie_indices") or key not in F:
                continue
            if prefix not in rows:
                continue
            e, f = rows[prefix]
            self.rep.same("copy.expected_equals_census", scope, np.asarray(E[key][e]), np.asarray(F[key][f]),
                          dict(label, field=key))
        for prefix, (e, f) in rows.items():
            if prefix + "tie_indptr" in E:
                ptr = E[prefix + "tie_indptr"]
                te = sorted(np.asarray(E[prefix + "tie_indices"][int(ptr[e]):int(ptr[e + 1])]).tolist())
                tf = stored_row(F, f, prefix)["ties"]
                self.rep.same("copy.expected_equals_census", scope, te, sorted([] if tf is None else tf.tolist()),
                              dict(label, field=prefix + "tie_indices"))

    def compare_list(self, scope, label, F, row, prefix, mine, ref_rows, refs):
        rep = self.rep
        part = "row." if not prefix else "screen."
        st = stored_row(F, row, prefix)
        my_members = np.r_[mine["strict"], mine["taken"]]
        rep.same(part + "membership", scope, set(st["list"].tolist()), set(my_members.tolist()), label)
        rep.same(part + "strict_members", scope, set(st["strict"].tolist()), set(mine["strict"].tolist()), label)
        rep.same(part + "t_all", scope, st["t_all"], mine["t_all"], label)
        rep.same(part + "t_in", scope, st["t_in"], mine["t_in"], label)
        rep.close(part + "cutoff_d64", scope, st["d64"], mine["d64"], TOL_DIST, label)
        stored_ties = st["ties"]
        rep.require(part + "tie_ids_stored", scope, stored_ties is not None, label)
        if stored_ties is not None:
            rep.same(part + "tie_ids", scope, set(stored_ties.tolist()), set(mine["ties"].tolist()), label)
        # strictly-inside members are stored in increasing distance
        dd = mine["d"][st["strict"]]
        rep.add(part + "strict_order", scope, bool(np.all(np.diff(dd) >= -TOL_DIST)), None, TOL_DIST,
                dict(label, distances=dd.tolist()[:8]))
        for key in ("origins", "eligible", "same_token"):
            if prefix + key in F:
                rep.same(part + key, scope, np.asarray(F[prefix + key][row]), mine[key], label)
        if prefix + "same_token_eligible" in F:
            rep.same(part + "same_token_eligible", scope, int(F[prefix + "same_token_eligible"][row]),
                     mine["same_token_eligible"], label)
        if prefix + "gap_median" in F:
            rep.close(part + "gap_median", scope, float(F[prefix + "gap_median"][row]), mine["gap_median"], 0.0, label)
        if prefix + "overlap_counts" in F:
            counts = np.asarray(F[prefix + "overlap_counts"][row])
            exp_ = np.asarray(F[prefix + "overlap_expected"][row])
            mx = np.asarray(F[prefix + "overlap_maximum"][row])
            for r, (ref, B) in enumerate(zip(refs, ref_rows)):
                lab = dict(label, reference=ref)
                if B is None or B["ties"] is None:
                    rep.require(part + "overlap_reference_available", scope, False, lab)
                    continue
                c, e, m = overlap(mine, B)
                rep.same(part + "overlap_counts", scope, counts[r], c, lab)
                rep.close(part + "overlap_expected", scope, exp_[r], e, TOL_DIST, lab)
                rep.same(part + "overlap_maximum", scope, float(mx[r]), float(m), lab)
        if prefix:
            stat = int(F["screen_status"][row])
            rep.same("screen.status", scope, stat, mine["status"], label)
            rep.close("screen.mass", scope, np.asarray(F["screen_mass"][row]), mine["mass"], TOL_DIST, label)
            if "screen_removed" in F and "removed" in mine:
                rep.close("screen.removed_short_range_mass", scope, float(F["screen_removed"][row]), mine["removed"],
                          TOL_DIST, label)
            elif label["mask"] == "window":
                rep.require("screen.removed_present", scope, False, label)
            if "screen_dmin" in F:
                rep.close("screen.dmin", scope, float(F["screen_dmin"][row]), mine["dmin"], TOL_DIST, label)
            if "screen_beta" in F and mine["status"] == 0:
                b = float(F["screen_beta"][row])
                rep.close("screen.beta_rel", scope, b, mine["beta"], TOL_BETA_REL, label, scale=max(abs(mine["beta"]), 1e-300))

    # ---------------------------------------------- 1% sample and aggregates
    def census_models(self):
        out = []
        for model in self.models:
            out.append((model, self.pool, [g for g in self.groups if (model, *g) in self.configs]))
        census = self.root / "census"
        if self.planned:
            # Section 5D: every stitched context, alone, for each approved corpus-mean (C) group, both models.
            stitched = [int(c["id"]) for c in self.layout.probe["contexts"] if c.get("kind") == "stitched"]
            for model in self.models:
                for c in stitched:
                    name = f"{model}_stitch{c:02d}"
                    groups = []
                    for g in self.groups:
                        if g[1] != "C":
                            continue
                        cfgp = census / name / f"{g[0]}_{g[1]}" / "group.json"
                        cfg = read_json(cfgp) if cfgp.is_file() else None
                        if self.rep.require("presence.stitched_group_complete", f"{name}/{g[0]}_{g[1]}",
                                            cfg is not None and cfg.get("complete") is True, dict(path=str(cfgp))):
                            self.configs[name, *g] = cfg
                            groups.append(g)
                    out.append((name, self.layout.stitch_pool(c), groups))
            return out
        if census.is_dir():
            for d in sorted(census.iterdir()):
                if d.is_dir() and "_stitch" in d.name and d.name.split("_stitch")[0] in self.models:
                    c = int(d.name.split("_stitch")[1])
                    groups = []
                    for g in GROUPS:
                        cfgp = d / f"{g[0]}_{g[1]}" / "group.json"
                        if self.planned and g not in self.groups:
                            continue                     # dropped by the plan: stale outputs are not audited
                        if cfgp.is_file():
                            cfg = read_json(cfgp)
                            self.configs[d.name, *g] = cfg
                            if cfg.get("complete"):
                                groups.append(g)
                    out.append((d.name, self.layout.stitch_pool(c), groups))
        return out

    def statistics(self):
        rep = self.rep
        for model, pool, groups in self.census_models():
            rng = np.random.default_rng(124)                    # the audit's own sample
            sample = np.sort(rng.choice(pool.n, size=max(1, math.ceil(pool.n / 100)), replace=False))
            if pool is self.pool:
                sample = np.union1d(sample, self.audit_rows)
            elig_cache = {}
            for reading, centring in groups:
                cfg = self.configs[model, reading, centring]
                scope = f"{model}/{reading}_{centring}"
                k, w = cfg["k"], cfg["window"]
                summary_path = self.root / "analysis" / "geometry" / model / f"{reading}_{centring}.json"
                summaries = {(s["input"], s["memory"], s["mask"]): s for s in read_json(summary_path)} \
                    if summary_path.is_file() else None
                if summaries is None:
                    rep.require("presence.geometry_summary", scope, False, dict(path=str(summary_path)))
                refs = cfg.get("references", ["local"])
                for name in cfg["inputs"]:
                    for mem in cfg["memories"]:
                        for mask in cfg["masks"]:
                            F = self.npz(_setting_path(self.root, model, reading, centring, name, mem, mask))
                            label = dict(input=name, memory=mem, mask=mask)
                            if not self.setting_complete(scope, label, F, pool, cfg, mask):
                                continue
                            ref_files = [self.npz(_setting_path(self.root, model, reading, centring,
                                                                name if r == "local" else r,
                                                                cfg["memories"][0] if r == "local" else mem, mask))
                                         for r in refs]
                            self.sample_rows(scope, label, pool, F, sample, k, w, mask, elig_cache, ref_files, refs, "")
                            if "screen_query" in F:
                                sq = np.asarray(F["screen_query"]).astype(np.int64)
                                srows = np.sort(np.random.default_rng(125).choice(
                                    len(sq), size=max(1, math.ceil(len(sq) / 100)), replace=False))
                                srows = np.union1d(srows, np.flatnonzero(np.isin(sq, self.audit_rows))) \
                                    if pool is self.pool else srows
                                self.sample_rows(scope, label, pool, F, srows, k, w, mask, elig_cache, ref_files,
                                                 refs, "screen_", sq)
                                # the screen overlaps gate the selection: check them on every screen row
                                self.overlap_formula(scope, label, F, ref_files, refs, np.arange(len(sq)), "screen_",
                                                     k, sq)
                            self.overlap_formula(scope, label, F, ref_files, refs, sample, "", k, None)
                            if summaries is not None:
                                s = summaries.get((name, mem, mask))
                                if not rep.require("presence.geometry_entry", scope, s is not None, label):
                                    continue
                                self.compare_aggregate(scope, label, F, s, k)
                self.files.clear()

    def setting_complete(self, scope, label, F, pool, cfg, mask):
        """Every member a census setting must hold, with the shapes the spec's quantities imply."""
        rep = self.rep
        if F is None:
            if self.planned:
                rep.require("presence.setting_file", scope, False, label)
            return False
        n, k = pool.n, cfg["k"]
        per_row = ["cutoff", "t_all", "t_in", "same_token", "same_token_eligible", "gap_median", "overlap_expected",
                   "overlap_maximum"]
        shapes = {"indices": (n, k), "origins": (n, 3), "eligible": (n, 3), "tie_indptr": (n + 1,),
                  "tie_indices": None, "overlap_counts": (n, len(cfg.get("references", ["local"])), 4)}
        shapes.update({key: (n,) for key in per_row})
        shapes["overlap_expected"] = shapes["overlap_maximum"] = (n, len(cfg.get("references", ["local"])))
        if pool.source is not None:     # a stitched text has no Q rows, hence no stored tie identities
            shapes.update({key: (n,) for key in ("stitch_observed", "stitch_baseline", "stitch_beyond")})
            del shapes["tie_indptr"], shapes["tie_indices"]
        if cfg.get("suffix_screen") and pool is self.pool:
            suffix_q = [q for q in self.layout.query if self.pool.pos[q] >= self.pool.suffix_start]
            m = len(suffix_q)
            for key, shape in list(shapes.items()):
                if key not in ("tie_indptr", "tie_indices"):
                    shapes["screen_" + key] = (m, *shape[1:]) if shape else None
            shapes.update(screen_tie_indptr=None, screen_tie_indices=None, screen_query=(m,), screen_mass=(m, 3),
                          screen_status=(m,), screen_beta=(m,), screen_dmin=(m,))
            if mask == "window":
                shapes["screen_removed"] = (m,)
        missing = [key for key in shapes if key not in F]
        wrong = [key for key, shape in shapes.items() if key in F and shape is not None
                 and tuple(np.shape(F[key])) != shape]
        ok = rep.require("integrity.setting_members", scope, not missing and not wrong,
                         dict(label, missing=missing, wrong_shape=wrong))
        if ok and "screen_query" in shapes:
            rep.same("integrity.screen_query_is_suffix_Q", scope, np.asarray(F["screen_query"]), suffix_q, label)
        return ok

    def overlap_formula(self, scope, label, F, ref_files, refs, rows, prefix, k, sq):
        """Expected and maximum overlap from the stored counts and cutoff sizes (section 4), on rows whose tie
        identities are not stored; the counts themselves are recomputed where identities exist (sample_rows)."""
        rows = np.asarray(rows, dtype=np.int64)
        counts = np.asarray(F[prefix + "overlap_counts"][rows]).astype(np.int64)
        a = np.asarray(F[prefix + "t_in"][rows]).astype(np.int64)
        a_all = np.asarray(F[prefix + "t_all"][rows]).astype(np.int64)
        ids = rows if sq is None else np.asarray(sq)[rows]
        for r, (ref, R) in enumerate(zip(refs, ref_files)):
            lab = dict(label, reference=ref)
            if not self.rep.require("formula.reference_available", scope, R is not None, lab):
                continue
            if prefix and not np.array_equal(np.asarray(R["screen_query"])[rows], ids):
                self.rep.require("formula.reference_rows_aligned", scope, False, lab)
                continue
            b = np.asarray(R[prefix + "t_in"][rows]).astype(np.int64)
            b_all = np.asarray(R[prefix + "t_all"][rows]).astype(np.int64)
            expected, maximum = overlap_values(counts[:, r], a, a_all, b, b_all, k)
            part = "formula." + prefix
            self.rep.rows(part + "overlap_expected", scope, F[prefix + "overlap_expected"][rows][:, r], expected,
                          TOL_DIST, lab, ids)
            self.rep.rows(part + "overlap_maximum", scope, F[prefix + "overlap_maximum"][rows][:, r], maximum, 0,
                          lab, ids)

    @staticmethod
    def sample_classes(F, rows, prefix, n):
        """Stored lists of `rows` as (row-number * (n+1) + member) keys: strict, tied (all), known-ties flag."""
        ind = np.asarray(F[prefix + "indices"][rows]).astype(np.int64)
        valid = ind >= 0
        t_in = np.asarray(F[prefix + "t_in"][rows]).astype(np.int64)
        t_all = np.asarray(F[prefix + "t_all"][rows]).astype(np.int64)
        n_strict = valid.sum(1) - t_in
        col = np.arange(ind.shape[1])[None, :]
        rid = np.broadcast_to(np.arange(len(rows))[:, None], ind.shape)
        strict = valid & (col < n_strict[:, None])
        taken = valid & ~strict
        s_keys = rid[strict] * (n + 1) + ind[strict]
        known = t_all == t_in
        t_keys = [rid[taken & known[:, None]] * (n + 1) + ind[taken & known[:, None]]]
        if prefix + "tie_indptr" in F:
            ptr = np.asarray(F[prefix + "tie_indptr"])
            ids = F[prefix + "tie_indices"]
            for r, row in enumerate(rows):
                if not known[r] and row + 1 < len(ptr) and ptr[row + 1] - ptr[row] == t_all[r]:
                    t_keys.append(r * (n + 1) + np.asarray(ids[int(ptr[row]):int(ptr[row + 1])]).astype(np.int64))
                    known[r] = True
        return dict(ind=ind, valid=valid, t_in=t_in, t_all=t_all, strict=s_keys, ties=np.concatenate(t_keys),
                    known=known)

    def sample_rows(self, scope, label, pool, F, rows, k, w, mask, elig_cache, ref_files, refs, prefix, sq=None):
        """Section 5A quantities of sampled rows, recomputed from the stored lists (all rows at once)."""
        rep = self.rep
        part = "sample." if not prefix else "sample.screen_"
        rows = np.asarray(rows, dtype=np.int64)
        q = rows if sq is None else np.asarray(sq)[rows]
        A = self.sample_classes(F, rows, prefix, pool.n)
        ind, valid = A["ind"], A["valid"]
        jj = np.where(valid, ind, 0)
        same = pool.text[jj] == pool.text[q][:, None]
        gap = np.abs(pool.pos[jj] - pool.pos[q][:, None])
        origin = np.where(valid, np.where(~same, 0, np.where(gap <= w, 2, 1)), -1)
        origins = np.stack([(origin == g).sum(1) for g in range(3)], axis=1)
        same_token = ((pool.tok[jj] == pool.tok[q][:, None]) & valid).sum(1)
        gaps = np.where(valid & same, gap, np.nan).astype(float)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                gap_median = np.nanmedian(gaps, axis=1) if gaps.shape[1] else np.full(len(rows), np.nan)
        key = (id(pool), prefix, mask)
        if key not in elig_cache:
            cand = pool.suffix if prefix else np.arange(pool.n)
            stats = [eligible_statistics(pool, int(i), cand, pool.eligible(int(i), cand, mask, w), w) for i in q]
            elig_cache[key] = (q.copy(), np.stack([s["eligible"] for s in stats]),
                               np.asarray([s["same_token_eligible"] for s in stats]))
        q_cached, eligible, same_token_eligible = elig_cache[key]
        rep.require(part + "sample_consistent", scope, np.array_equal(q_cached, q), label)
        ex = dict(label)
        rep.rows(part + "origins", scope, F[prefix + "origins"][rows], origins, 0, ex, q)
        rep.rows(part + "same_token", scope, F[prefix + "same_token"][rows], same_token, 0, ex, q)
        rep.rows(part + "gap_median", scope, F[prefix + "gap_median"][rows], gap_median, 0, ex, q)
        rep.rows(part + "eligible", scope, F[prefix + "eligible"][rows], eligible, 0, ex, q)
        if prefix + "same_token_eligible" in F:
            rep.rows(part + "same_token_eligible", scope, F[prefix + "same_token_eligible"][rows], same_token_eligible,
                     0, ex, q)
        rep.rows(part + "list_size", scope, valid.sum(1), np.minimum(k, eligible.sum(1)), 0, ex, q)
        rep.require(part + "ties_disjoint_from_strict", scope, not np.isin(A["strict"], A["ties"]).any(), label)
        if prefix + "overlap_counts" in F:
            stored_counts = np.asarray(F[prefix + "overlap_counts"][rows])
            stored_exp = np.asarray(F[prefix + "overlap_expected"][rows])
            stored_max = np.asarray(F[prefix + "overlap_maximum"][rows])
            for r, (ref, R) in enumerate(zip(refs, ref_files)):
                if R is None:
                    continue
                if prefix and not np.array_equal(np.asarray(R["screen_query"])[rows], q):
                    rep.require(part + "reference_rows_aligned", scope, False, dict(label, reference=ref))
                    continue
                B = A if R is F else self.sample_classes(R, rows, prefix, pool.n)
                use = A["known"] & B["known"]
                if not use.any():
                    continue
                N = pool.n + 1

                def count(x, y):
                    return np.bincount(x[np.isin(x, y)] // N, minlength=len(rows))

                ss, c1 = count(A["strict"], B["strict"]), count(A["strict"], B["ties"])
                c2, c3 = count(A["ties"], B["strict"]), count(A["ties"], B["ties"])
                counts = np.stack([ss, c1, c2, c3], axis=1)
                expected, maximum = overlap_values(counts, A["t_in"], A["t_all"], B["t_in"], B["t_all"], k)
                lab = dict(label, reference=ref)
                rep.rows(part + "overlap_counts", scope, stored_counts[use, r], counts[use], 0, lab, q[use])
                rep.rows(part + "overlap_expected", scope, stored_exp[use, r], expected[use], TOL_DIST, lab, q[use])
                rep.rows(part + "overlap_maximum", scope, stored_max[use, r], maximum[use], 0, lab, q[use])
        if not prefix and pool.source is not None and "stitch_observed" in F:
            obs, base = self.stitch_values(pool, q, ind, valid)
            rep.rows("sample.stitch_observed", scope, F["stitch_observed"][rows], obs, TOL_AGG, ex, q)
            rep.rows("sample.stitch_baseline", scope, F["stitch_baseline"][rows], base, TOL_AGG, ex, q)
            rep.rows("sample.stitch_beyond", scope, F["stitch_beyond"][rows], obs - base, TOL_AGG, ex, q)

    @staticmethod
    def stitch_values(pool, q, ind, valid):
        """Section 5D: same-source share of the list and its position baseline: b(lag) is the mean over node
        positions u with u + lag a node of 1[source(u) = source(u + lag)]; the query's baseline is the mean of
        b(j - i) over its neighbours j. One text, so pool index = position - first position."""
        if not hasattr(pool, "_baseline"):
            src = pool.source[pool.start - 1:pool.L]        # sources of the node positions
            n = len(src)
            base = np.empty(2 * n - 1)
            for lag in range(-(n - 1), n):
                a = src[max(0, -lag):n - max(0, lag)]      # u
                b = src[max(0, lag):n - max(0, -lag)]      # u + lag
                base[lag + n - 1] = np.mean(a == b)
            pool._baseline, pool._src = base, src
        n = len(pool._src)
        jj = np.where(valid, ind, 0)
        count = valid.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            obs = ((pool._src[jj] == pool._src[q][:, None]) & valid).sum(1) / count
            base = np.where(valid, pool._baseline[jj - q[:, None] + n - 1], 0.0).sum(1) / count
        return np.where(count > 0, obs, np.nan), np.where(count > 0, base, np.nan)

    def compare_aggregate(self, scope, label, F, s, k):
        """Means over queries of the stored per-query quantities (section 5A table)."""
        rep = self.rep

        def mean(a):
            a = np.asarray(a, dtype=float)
            if a.ndim == 2:
                return [mean(a[:, c]) for c in range(a.shape[1])]
            a = a[np.isfinite(a)]
            return float(a.mean()) if len(a) else None

        def share(num, den):
            num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
            if num.ndim == 2:
                den = den[:, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(den != 0, num / np.where(den != 0, den, 1), math.nan)

        origins, eligible = np.asarray(F["origins"]), np.asarray(F["eligible"])
        selected, eligible_total = origins.sum(1), eligible.sum(1)
        mine = dict(queries=len(selected), full_topk_queries=int((selected == k).sum()),
                    origin=mean(share(origins, selected)), chance_origin=mean(share(eligible, eligible_total)),
                    position_gap=mean(F["gap_median"]),
                    position_gap_defined=int(np.isfinite(np.asarray(F["gap_median"], dtype=float)).sum()),
                    same_token=mean(share(F["same_token"], selected)),
                    chance_same_token=mean(share(F["same_token_eligible"], eligible_total)))
        if "overlap_expected" in F:
            mine.update(overlap_expected=mean(np.asarray(F["overlap_expected"]) / k),
                        overlap_maximum=mean(np.asarray(F["overlap_maximum"]) / k))
        for key in ("stitch_observed", "stitch_baseline", "stitch_beyond"):
            if key in F:
                mine[key] = mean(F[key])
        if "screen_mass" in F:
            mine.update(screen_mass=mean(F["screen_mass"]), screen_origin=mean(np.asarray(F["screen_origins"]) / k),
                        screen_status=np.bincount(np.asarray(F["screen_status"]).astype(np.int64), minlength=4).tolist(),
                        screen_overlap_expected=mean(np.asarray(F["screen_overlap_expected"]) / k),
                        screen_overlap_maximum=mean(np.asarray(F["screen_overlap_maximum"]) / k))
            if "screen_removed" in F:
                mine["screen_removed"] = mean(F["screen_removed"])
        def floats(v):
            return np.asarray([np.nan if x is None else x for x in (v if isinstance(v, list) else [v])], dtype=float)

        for key, value in mine.items():
            lab = dict(label, field=key)
            if not rep.require("aggregate.geometry_key_present", scope, key in s, lab):
                continue
            if key in ("queries", "full_topk_queries", "position_gap_defined", "screen_status"):
                rep.same("aggregate.geometry_summary", scope, s[key], value, lab)
            else:
                rep.close("aggregate.geometry_summary", scope, floats(s[key]), floats(value), TOL_AGG, lab)

    # ------------------------------------------------------------- readout
    def readout(self):
        rep, pool = self.rep, self.pool
        controls = Controls(pool)
        for model in self.models:
            preds = self.load_predictions(model)
            prediction_json = self.root / "analysis" / "prediction" / f"{model}.json"
            stored_settings = {}
            if prediction_json.is_file():
                for s in read_json(prediction_json)["settings"]:
                    stored_settings[s["site"], s["reading"], s["centring"], s["memory"], s["mask"]] = s
            for reading, centring in self.groups:
                cfg = self.configs.get((model, reading, centring))
                if cfg is None:
                    continue
                scope = f"{model}/{reading}_{centring}"
                sites = [s for s in SPEC["readout_sites"] if f"site{s:02d}" in cfg["inputs"]]
                for s in SPEC["readout_sites"]:
                    if s not in sites:
                        rep.absent.append(dict(what=f"{scope}/readout site{s:02d}", reason="site not in census inputs"))
                setting_index = 0
                for site in sites:
                    for mem in cfg["memories"]:
                        for mask in cfg["masks"]:
                            setting = (site, reading, centring, mem, mask)
                            label = dict(site=site, memory=mem, mask=mask)
                            path = _readout_path(self.root, model, reading, centring, site, mem, mask)
                            F = self.npz(_setting_path(self.root, model, reading, centring, f"site{site:02d}", mem, mask))
                            if not path.is_file():
                                if self.planned:
                                    rep.require("presence.readout_file", scope, False, label)
                                else:
                                    rep.absent.append(dict(what=f"{scope}/{path.name}", reason="readout file missing"))
                                continue
                            if F is None:
                                continue
                            with np.load(path, allow_pickle=False) as R:
                                data = {key: R[key] for key in R.files}
                            self.readout_summary_check(scope, label, data, stored_settings.get(setting),
                                                       path.with_suffix(".json"))
                            key = setting_key(model, setting)
                            self.witness_check(scope, label, data, F, controls, cfg["window"], key)
                            docs = [int(x) for x in data["documents"]]
                            rep.same("readout.documents", scope, docs, sorted(set(pool.texts)), label)
                            doc = docs[(setting_index + sum(map(ord, model))) % len(docs)]
                            setting_index += 1
                            self.document_check(scope, dict(label, document=doc), data, F, controls, cfg["window"],
                                                preds, key, doc)
                            self.files.pop(F.path, None)

    def load_predictions(self, model):
        pool = self.pool
        parts = {k: [] for k in ("top20_ids", "top20_probs", "top1_id", "top20_mass")}
        for c in pool.texts:
            with np.load(self.root / "pred" / model / f"c{c:02d}.npz") as f:
                for key in parts:
                    parts[key].append(f[key][pool.start:pool.stop])      # row u = position u
        out = {key: np.concatenate(v) for key, v in parts.items()}
        out["top20_probs"] = out["top20_probs"].astype(np.float64)
        out["top20_mass"] = out["top20_mass"].astype(np.float64)
        return out

    def readout_summary_check(self, scope, label, data, stored_entry, json_path):
        rep = self.rep
        fields = [str(f) for f in data["fields"]]
        reasons = [str(r) for r in data["reasons"]]
        strata = [str(s) for s in data["strata"]]
        if self.spec_parameters:
            self.spec_check("min_matched", scope, int(data["min_matched"]), SPEC["min_matched"])
            self.spec_check("bootstrap_draws", scope, int(data["bootstrap_draws"]), SPEC["bootstrap_draws"])
        mine = readout_summary(data["by_document"], fields, reasons, int(data["min_matched"]),
                               int(data["bootstrap_seed"]), int(data["bootstrap_draws"]))
        stored = read_json(json_path) if json_path.is_file() else None
        for where, s in (("json", stored), ("analysis", stored_entry)):
            lab = dict(label, source=where)
            if not rep.require("presence.readout_summary", scope, s is not None, lab):
                continue
            rep.same("readout.eligible", scope, s["eligible"], mine["eligible"], lab)
            rep.close("readout.coverage", scope, s["coverage"], mine["coverage"], TOL_AGG, lab)
            for key in ("score", "top1_neighbour", "top1_control"):
                rep.close("readout." + key, scope, np.nan if s[key] is None else s[key],
                          np.nan if mine[key] is None else mine[key], TOL_AGG, dict(lab, field=key))
            a = np.asarray(s["bootstrap_percentiles"] or [np.nan] * 3, dtype=float)
            b = np.asarray(mine["percentiles"] or [np.nan] * 3, dtype=float)
            rep.close("readout.bootstrap_percentiles", scope, a, b, TOL_AGG, lab)
            for n, name in enumerate(strata):
                ss = s["strata"][name]
                status = "absent" if mine["absent"][n] else "covered" if mine["covered"][n] else "uncovered"
                rep.same("readout.stratum_status", scope, ss["status"], status, dict(lab, stratum=name))
                rep.close("readout.stratum_values", scope,
                          [ss["pairs"], ss["matched"], np.nan if ss["mean_advantage"] is None else ss["mean_advantage"]],
                          [mine["pairs"][n], mine["matched"][n], mine["mean_advantage"][n]], TOL_AGG,
                          dict(lab, stratum=name))

    def witness_check(self, scope, label, data, F, controls, w, key):
        """Replay the controls of the audit queries from their keyed seeds; check every 5C constraint."""
        rep, pool = self.rep, self.pool
        strata = [str(s) for s in data["strata"]]
        stored = {}
        for qq, jj, cc, rr, ww, ss in zip(*(np.asarray(data[f]) for f in ("query", "neighbour", "control", "reason",
                                                                         "weight", "stratum"))):
            stored[int(qq), int(jj)] = (int(cc), int(rr), float(ww), int(ss))
        expected_pairs = {}
        for i in self.layout.audit:
            st = stored_row(F, i)
            if not rep.require("readout.witness_ties_available", scope, st["ties"] is not None, dict(label, query=i)):
                continue
            pairs = expand(st)
            excluded = frozenset({i} | {j for j, _ in pairs})
            for j, weight in pairs:
                expected_pairs[i, j] = (weight, excluded)
        rep.same("readout.witness_pairs", scope, set(stored), set(expected_pairs), label)
        keys = [p for p in expected_pairs if p in stored]
        if not keys:
            return
        got = np.asarray([stored[p] for p in keys], dtype=float)             # control, reason, weight, stratum
        drawn = np.asarray([controls.draw(key, i, j, expected_pairs[i, j][1]) for i, j in keys], dtype=float)
        weight = np.asarray([expected_pairs[p][0] for p in keys])
        qi = np.asarray([i for i, _ in keys])
        nj = np.asarray([j for _, j in keys])
        names = [f"{ORIGINS[o]}/{'frequent' if controls.frequency[j] >= 20 else 'rare'}"
                 for o, j in zip(pool.origin_pairs(qi, nj, w), nj)]
        my_stratum = np.asarray([strata.index(nm) for nm in names], dtype=float)
        ids = qi * (pool.n + 1) + nj
        rep.rows("readout.witness_weight", scope, got[:, 2], weight, TOL_AGG, label, ids)
        rep.rows("readout.control_reason", scope, got[:, 1], drawn[:, 1], 0, label, ids)
        rep.rows("readout.control_replay", scope, got[:, 0], drawn[:, 0], 0, label, ids)
        rep.rows("readout.stratum", scope, got[:, 3], my_stratum, 0, label, ids)
        # every section 5C constraint on the stored controls themselves
        cc = got[:, 0].astype(np.int64)
        matched = (got[:, 1] == 0) & (cc >= 0)
        c = np.where(matched, cc, 0)
        neighbour_keys = np.asarray(sorted(i * (pool.n + 1) + j for i, j in expected_pairs))
        same_text = pool.text[qi] == pool.text[nj]
        ok = ((pool.text[c] == pool.text[nj]) & (pool.tok[c] == pool.tok[nj]) & (c != qi) & (c != nj)
              & ~np.isin(qi * (pool.n + 1) + c, neighbour_keys)
              & (~same_text | (lag_bin(np.abs(pool.pos[c] - pool.pos[qi])) == lag_bin(np.abs(pool.pos[nj] - pool.pos[qi])))))
        rep.rows("readout.control_constraints", scope, ok[matched], np.ones(matched.sum()), 0, label, ids[matched])
        rep.rows("readout.unmatched_has_no_control", scope, cc[~matched], -np.ones((~matched).sum()), 0, label,
                 ids[~matched])

    def document_check(self, scope, label, data, F, controls, w, preds, key, doc):
        """Recompute one document's stored contributions from the stored lists, replayed controls and predictions."""
        rep, pool = self.rep, self.pool
        fields = [str(f) for f in data["fields"]]
        reasons = [str(r) for r in data["reasons"]]
        strata = [str(s) for s in data["strata"]]
        col = {f: n for n, f in enumerate(fields)}
        reason_col = [col["matched"]] + [col[r] for r in reasons[1:]]
        sums = np.zeros((len(strata), len(fields)))
        raw = np.zeros((len(strata), len(reasons)), dtype=np.int64)
        ids, probs, top1, mass = preds["top20_ids"], preds["top20_probs"], preds["top1_id"], preds["top20_mass"]
        for i in self.layout.query:
            if int(pool.text[i]) != doc:
                continue
            st = stored_row(F, i)
            if not rep.require("readout.document_ties_available", scope, st["ties"] is not None, dict(label, query=i)):
                return
            pairs = expand(st)
            excluded = frozenset({i} | {j for j, _ in pairs})
            js = np.asarray([j for j, _ in pairs], dtype=np.int64)
            ws = np.asarray([x for _, x in pairs])
            drawn = [controls.draw(key, i, int(j), excluded) for j in js]
            ctrl = np.asarray([c for c, _ in drawn], dtype=np.int64)
            why = np.asarray([r for _, r in drawn], dtype=np.int64)
            origin = pool.origin(i, js, w)
            names = [f"{ORIGINS[o]}/{'frequent' if controls.frequency[j] >= 20 else 'rare'}" for o, j in zip(origin, js)]
            sidx = np.asarray([strata.index(nm) for nm in names], dtype=np.int64)
            ii = np.full(len(js), i)
            o_n = prediction_overlap(ids[ii], probs[ii], ids[js], probs[js])
            m = why == 0
            o_c = np.zeros(len(js))
            if m.any():
                o_c[m] = prediction_overlap(ids[ii[m]], probs[ii[m]], ids[ctrl[m]], probs[ctrl[m]])
            for n in range(len(js)):
                s = sidx[n]
                raw[s, why[n]] += 1
                sums[s, reason_col[why[n]]] += ws[n]
                sums[s, col["mass_query"]] += ws[n] * mass[i]
                sums[s, col["mass_neighbour"]] += ws[n] * mass[js[n]]
                if why[n] == 0:
                    c = ctrl[n]
                    sums[s, col["advantage"]] += ws[n] * (o_n[n] - o_c[n])
                    sums[s, col["top1_neighbour"]] += ws[n] * float(top1[i] == top1[js[n]])
                    sums[s, col["top1_control"]] += ws[n] * float(top1[i] == top1[c])
                    sums[s, col["mass_control"]] += ws[n] * mass[c]
        d = [int(x) for x in data["documents"]].index(doc)
        rep.close("readout.document_contributions", scope, data["by_document"][d], sums, TOL_DIST, label)
        rep.same("readout.document_raw_counts", scope, data["raw_counts"][d], raw, label)

    # ----------------------------------------------------------- selection
    def selection(self):
        rep = self.rep
        path = self.root / "analysis" / "stage_b_selection.json"
        if not path.is_file():
            if self.planned or self.spec_parameters:
                rep.require("presence.selection", "selection", False, dict(path=str(path)))
            else:
                rep.absent.append(dict(what="analysis/stage_b_selection.json", reason="missing"))
            return
        stored = read_json(path)
        sites, required = list(stored["sites"]), int(stored["required_sites"])
        readout_sites, thr = list(stored["readout_sites"]), stored["thresholds"]
        models = list(stored["table"][0]["models"]) if stored["table"] else list(self.models)
        if self.spec_parameters:
            self.spec_check("selection sites", "selection", sites, SPEC["sites"])
            self.spec_check("required sites", "selection", required, SPEC["required_sites"])
            self.spec_check("readout sites", "selection", readout_sites, SPEC["readout_sites"])
            self.spec_check("thresholds", "selection", thr, SPEC["thresholds"])
        rep.same("selection.models", "selection", sorted(models), sorted(self.models))
        rows = []
        for (reading, centring), mem, mask in [(g, m, k) for g in GROUPS for m in CANDIDATE_MEMORIES
                                               for k in ("none", "window")]:
            construction = dict(reading=reading, centring=centring, memory=mem, mask=mask)
            row = dict(construction=construction, models={}, worst=[])
            for model in models:
                cfg = self.configs.get((model, reading, centring)) if (reading, centring) in self.groups else None
                by_site = {}
                for site in sites:
                    F = None
                    if cfg is not None and f"site{site:02d}" in cfg["inputs"]:
                        F = self.npz(_setting_path(self.root, model, reading, centring, f"site{site:02d}", mem, mask))
                    by_site[site] = self.site_screen(F, cfg, mask, thr) if F is not None and "screen_mass" in F else None
                    if by_site[site] is not None:
                        row["worst"] += [by_site[site]["max"]["random"], by_site[site]["max"]["embedding"]]
                passing = sum(1 for v in by_site.values() if v is not None and v["passed"])
                row["models"][model] = dict(sites=by_site, passing=passing, passed=passing >= required)
            row["passed"] = all(m["passed"] for m in row["models"].values())
            vals = [x for x in row["worst"] if x is not None and not math.isnan(x)]
            row["worst"] = max(vals) if vals else None
            rows.append(row)
        self.files.clear()
        # compare the table
        stored_rows = {tuple(sorted(r["construction"].items())): r for r in stored["table"]}
        rep.same("selection.table_rows", "selection", len(stored["table"]), len(rows))
        for row in rows:
            key = tuple(sorted(row["construction"].items()))
            lab = dict(row["construction"])
            srow = stored_rows.get(key)
            if not rep.require("selection.row_present", "selection", srow is not None, lab):
                continue
            for model in models:
                sm = srow["models"].get(model)
                if not rep.require("selection.model_present", "selection", sm is not None, dict(lab, model=model)):
                    continue
                for site in sites:
                    mine, theirs = row["models"][model]["sites"][site], sm["sites"].get(str(site))
                    sl = dict(lab, model=model, site=site)
                    if not rep.same("selection.site_has_data", "selection", theirs is not None, mine is not None, sl):
                        continue
                    if mine is None:
                        continue
                    rep.same("selection.site_conditions", "selection", theirs["conditions"], mine["conditions"], sl)
                    rep.same("selection.site_passed", "selection", theirs["passed"], mine["passed"], sl)
                    short = theirs["removed"] if row["construction"]["mask"] == "window" else theirs["query_mass"]["near"]
                    got = [theirs["query_mass"]["other"], short] + [theirs["maximum_overlap"][r] for r in
                                                                    ("local", "random", "embedding")] + \
                          [theirs["expected_overlap"][r] for r in ("local", "random", "embedding")]
                    want = [mine["other"], mine["short"]] + [mine["max"][r] for r in ("local", "random", "embedding")] + \
                           [mine["exp"][r] for r in ("local", "random", "embedding")]
                    rep.close("selection.site_values", "selection",
                              [np.nan if v is None else v for v in got], [np.nan if v is None else v for v in want],
                              TOL_AGG, sl)
                rep.same("selection.passing_sites", "selection", sm["passing"], row["models"][model]["passing"],
                         dict(lab, model=model))
                rep.same("selection.model_passed", "selection", sm["passed"], row["models"][model]["passed"],
                         dict(lab, model=model))
            rep.same("selection.candidate_passed", "selection", srow["passed"], row["passed"], lab)
            rep.close("selection.worst_adversary_overlap", "selection",
                      np.nan if srow["worst_adversary_overlap"] is None else srow["worst_adversary_overlap"],
                      np.nan if row["worst"] is None else row["worst"], TOL_AGG, lab)

        def rank_key(r):
            c = r["construction"]
            return (INTERVENTION.index((c["centring"], c["mask"])), c["reading"] != "cosine",
                    math.inf if r["worst"] is None else r["worst"], MEMORY_A[c["memory"]])

        passing = sorted([r for r in rows if r["passed"]], key=rank_key)
        M = passing[0]["construction"] if passing else None
        Lc = dict(M, memory="none") if M else None
        X, common = None, []
        self.x_candidates, self.x_scored = [], []
        if passing:
            X, common = self.choose_x(passing, passing[0], models, readout_sites, rank_key)
            # the evidence behind X, not only the winner (section 5 / E3: every number behind the choice)
            eligible_rows = {id(r) for r in self.x_candidates}
            for row in passing[1:]:
                srow = stored_rows[tuple(sorted(row["construction"].items()))]
                pred = srow.get("prediction", {})
                rep.same("selection.X_eligibility", "selection", bool(pred.get("eligible")), id(row) in eligible_rows,
                         dict(row["construction"]))
            for row, score, p5 in self.x_scored:
                pred = stored_rows[tuple(sorted(row["construction"].items()))].get("prediction", {})
                rep.close("selection.X_evidence", "selection",
                          [np.nan if pred.get(k) is None else pred[k] for k in ("common_score", "common_bootstrap_p5")],
                          [score, np.nan if p5 is None else p5], TOL_AGG, dict(row["construction"]))
        rep.same("selection.recommend_stage_b", "selection", stored["recommend_stage_b"], bool(passing))
        rep.same("selection.M", "selection", stored["M"], M)
        rep.same("selection.L", "selection", stored["L"], Lc)
        rep.same("selection.X", "selection", stored["X"], X)
        if passing:
            rep.same("selection.common_strata", "selection", stored.get("common_strata", []), common)
            R = dict(reading="cosine", centring="C", memory="none", mask="none")
            solve = [R] + ([Lc] if Lc != R else []) + [M] + ([X] if X is not None else [])
            rep.same("selection.solve", "selection", stored.get("solve"), solve)
        else:
            rep.same("selection.solve", "selection", stored.get("solve"), [])

    def site_screen(self, F, cfg, mask, thr):
        """Section 5 conditions 1-4 at one site, from the stored screen statistics."""
        k = cfg["k"]
        refs = cfg.get("references", ["local"])
        if sorted(refs) != sorted(("local", "random", "embedding")):
            return None                     # condition 4 cannot be evaluated without both adversaries
        mass = np.asarray(F["screen_mass"], dtype=float)

        def finite_mean(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            return float(a.mean()) if len(a) else None

        other = finite_mean(mass[:, 0])
        short = finite_mean(F["screen_removed"]) if mask == "window" and "screen_removed" in F else \
            (finite_mean(mass[:, 2]) if mask == "none" else None)
        mx = {r: float(np.mean(np.asarray(F["screen_overlap_maximum"])[:, n] / k)) for n, r in enumerate(refs)}
        ex = {r: float(np.mean(np.asarray(F["screen_overlap_expected"])[:, n] / k)) for n, r in enumerate(refs)}
        for r in ("local", "random", "embedding"):
            mx.setdefault(r, None)
            ex.setdefault(r, None)
        limit = thr["removed_mass_max"] if mask == "window" else thr["unmasked_near_mass_max"]
        cond = dict(other_mass=other is not None and other >= thr["other_mass_min"],
                    short_mass=short is not None and short <= limit,
                    local=mx["local"] is not None and mx["local"] <= thr["overlap_max"],
                    random=mx["random"] is not None and mx["random"] <= thr["overlap_max"],
                    embedding=mx["embedding"] is not None and mx["embedding"] <= thr["overlap_max"])
        return dict(conditions=cond, passed=all(cond.values()), other=other, short=short, max=mx, exp=ex)

    def choose_x(self, passing, m_row, models, readout_sites, rank_key):
        """Section 5: at most one additional example X, on the strata covered in all eligible candidates."""
        candidates = []
        for row in passing:
            if row is m_row:
                continue
            c = row["construction"]
            studies = {}
            for model in models:
                for site in readout_sites:
                    path = _readout_path(self.root, model, c["reading"], c["centring"], site, c["memory"], c["mask"])
                    if not path.is_file():
                        studies = None
                        break
                    with np.load(path, allow_pickle=False) as f:
                        by_doc = f["by_document"]
                        s = readout_summary(by_doc, [str(x) for x in f["fields"]], [str(x) for x in f["reasons"]],
                                            int(f["min_matched"]), int(f["bootstrap_seed"]), int(f["bootstrap_draws"]))
                        info = (by_doc, [str(x) for x in f["fields"]], [str(x) for x in f["strata"]],
                                int(f["bootstrap_seed"]), int(f["bootstrap_draws"]), s["covered"])
                    if not s["eligible"]:
                        studies = None
                        break
                    studies[model, site] = info
                if studies is None:
                    break
            if studies:
                candidates.append((row, studies))
        self.x_candidates = [row for row, _ in candidates]
        self.x_scored = []
        if not candidates:
            return None, []
        common = np.logical_and.reduce([info[5] for _, st in candidates for info in st.values()])
        strata_names = next(iter(candidates[0][1].values()))[2]
        names = [s for s, u in zip(strata_names, common) if u]
        if not common.any():
            return None, names
        scored = []
        for row, studies in candidates:
            scores, boots = [], []
            for by_doc, fields, _, seed, draws, _ in studies.values():
                col = {f: n for n, f in enumerate(fields)}
                total = by_doc.sum(0)[common]
                scores.append(float(np.mean(total[:, col["advantage"]] / total[:, col["matched"]])))
                D = by_doc.shape[0]
                resample = np.random.default_rng(seed).integers(0, D, size=(draws, D))
                b = np.empty(draws)
                for n in range(draws):
                    t = by_doc[resample[n]].sum(0)[common]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        b[n] = np.mean(t[:, col["advantage"]] / t[:, col["matched"]])
                boots.append(b)
            boot = np.mean(boots, axis=0)
            valid = boot[np.isfinite(boot)]
            scored.append((row, float(np.mean(scores)), float(np.percentile(valid, 5)) if len(valid) else None))
        self.x_scored = scored
        best = min(scored, key=lambda x: (-x[1], rank_key(x[0])))
        if best[2] is not None and best[2] > 0:
            return best[0]["construction"], names
        return None, names

    # ---------------------------------------------------------------- run
    def run(self):
        t0 = time.perf_counter()
        self.plan()
        self.configuration()
        log(f"models {self.models}, groups {self.groups}, rows {self.rows}")
        self.recompute()
        self.files.clear()
        log("census statistics")
        self.statistics()
        self.files.clear()
        log("readout")
        self.readout()
        self.files.clear()
        log("selection")
        self.selection()
        checks = self.rep.result()
        passed = bool(checks) and all(c["passed"] for c in checks)
        return dict(passed=passed, spec="SPEC v3.2 section 12.4", root=str(self.root), rows=self.rows,
                    audit_queries=[dict(text=c, position=p) for c, p in self.layout.audit_pairs[:self.rows]],
                    models=self.models, groups=[list(g) for g in self.groups], planned=self.planned,
                    spec_parameters=self.spec_parameters, seconds=time.perf_counter() - t0,
                    failed_checks=[dict(name=c["name"], scope=c["scope"], failed=c["failed"]) for c in checks
                                   if not c["passed"]],
                    checks=checks, absent=self.rep.absent, notes=self.rep.notes,
                    conventions_trusted=CONVENTIONS, not_reproduced=NOT_REPRODUCED)


def run(root: Path, out: Path, rows: int | None = None, *, spec_parameters: bool = True) -> dict:
    """Run the Stage A audit on `root`; write the JSON report to `out` and return it.

    rows: number of audit queries (in probe order) whose complete rows are recomputed; None = all.
    spec_parameters=False turns deviations from the spec's sizes and thresholds into notes (small fixtures only).
    """
    try:
        report = Audit(root, rows, spec_parameters).run()
    except Exception as error:              # an audit that cannot run has failed
        import traceback
        report = dict(passed=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(),
                      checks=[], not_reproduced=NOT_REPRODUCED, conventions_trusted=CONVENTIONS)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(report), indent=1, allow_nan=False))
    tmp.replace(out)
    return report


def time_one_row(root: Path, model: str) -> float:
    """Seconds for one complete explicit-lag audit row: the first audit query of the first text, hidden state 8
    (or the deepest even state present), all memories, both readings and both centrings, with masks, top-64
    lists and suffix-screen conditionals. Reads only probe/probe.json and acts/<model>/. The per-site set-up
    (loading, centring, diagonals) is added amortised over the audit queries of one site."""
    layout = Layout(root)
    pool = layout.pool
    i = next(layout.pool.index(c, p) for c, p in layout.audit_pairs if c == pool.texts[0])
    states = np.load(Path(root) / "acts" / model / f"c{pool.texts[0]:02d}.npy", mmap_mode="r").shape[0]
    site = 8 if states > 8 else max(s for s in range(2, states, 2))
    cfg = dict(k=64, window=64, target=64, tie_rtol=1e-12, memories=list(MEMORY_A), masks=list(MASKS))
    alphas = np.asarray(list(MEMORY_A.values()))
    audit = Audit.__new__(Audit)
    audit.pool = pool
    t0 = time.perf_counter()
    X = load_input(root, model, pool, f"site{site:02d}")
    setup, row = time.perf_counter() - t0, 0.0
    r, t = pool.rank[int(pool.text[i])], int(pool.pos[i])
    for centring in ("C", "O"):
        t0 = time.perf_counter()
        Y, norms = centred(X, centring)
        state = {}
        for reading in ("dot", "cosine"):
            sq = norms ** 2 if reading == "dot" else (norms / norms) ** 2
            diag = explicit_diagonals(sq, alphas)
            pd = diag[:, :, pool.start - 1:].reshape(len(alphas), -1)
            state[reading] = (diag, pd, pd.mean(axis=1))
        setup += time.perf_counter() - t0
        t0 = time.perf_counter()
        G = gram_block(Y, r, t)
        for reading in ("dot", "cosine"):
            if reading == "cosine":
                G /= norms[r, :t][:, None, None]
                G /= norms[None, :, :]
            diag, pd, scale = state[reading]
            K = explicit_lag_row(G, t, alphas, pool.start).reshape(len(alphas), -1)
            d = distances_from_kernel(K, diag[:, r, t - 1], pd, reading, scale)
            Audit.row_results(audit, i, d, cfg)
        del G, Y
        row += time.perf_counter() - t0
    n_rows = max(1, len(layout.audit))
    log(f"time_one_row({model}, site {site}): row {row:.2f}s, per-site set-up {setup:.2f}s (amortised over {n_rows})")
    return row + setup / n_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--test-parameters", action="store_true",
                        help="small fixtures only: deviations from the spec's sizes become notes")
    parser.add_argument("--time-row", metavar="MODEL", help="print time_one_row(root, MODEL) and exit")
    args = parser.parse_args(argv)
    if args.time_row:
        print(time_one_row(args.root, args.time_row))
        return 0
    report = run(args.root, args.out, args.rows, spec_parameters=not args.test_parameters)
    failed = [f"{c['name']} [{c['scope']}]" for c in report.get("checks", []) if not c["passed"]]
    print(json.dumps(dict(passed=report["passed"], failed=failed[:50], error=report.get("error")), indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
