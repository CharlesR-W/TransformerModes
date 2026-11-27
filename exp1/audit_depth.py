"""Independent audit of the depth embedding (SPEC_DEPTH.md v3.1 section 8; site encoding per SPEC_DEPTH_ADDENDUM.md).

Written by a different model family from the pipeline, with its own numpy/scipy code computed from the spec's
definitions. Nothing here imports or copies exp1/depth.py, depth_dm.py, spacetime.py, stage_b.py or pack_depth.py;
those files were read only to learn the release's file layout.

    OPENBLAS_NUM_THREADS=3 TMPDIR=/var/tmp nice -n 5 uv run --with numpy --with scipy python exp1/audit_depth.py STAGE

STAGE is one of
  manifest    section 3 (seeds, counts, panels disjoint from sets, cohorts, references, paths) and token ids
  inputs      section 2 (synthetic ids regenerated, shuffle maps) and the raw activations: hashes against each unit's
              input_files.json, layer-0 identity of every manifest row with its token id, shuffled rows against their
              originals, separator rows
  cell        heavy, per (model, set) of doc00/d32/rand: P_l rebuilt from raw activations (section 4 graph,
              canonical distances, Sinkhorn), the joint solve (structured operator, shift-invert Lanczos), the
              residual / mean-zero / orthonormality of the release's float64 vectors, the float32 cast, the
              independent count of eigenvalues (block LDL inertia), t* as the crossing, own placement (section 5) of
              every panel token and a sample of path tokens, and band-level orthogonal alignment with no rescaling
              (--model, --set select one; default all six, resumable)
  facts       section 6 per-embedding and per-placement measures of all 22 units x 2 strengths, from release data
  aggregates  cross.json, permutations.json, gaussian.json, refit_agreement.json recomputed with own code
  pack        the site pack (--pack DIR): re-encoding, scales, saturation, inventory, rows/texts/facts
  report      assemble OUT/report.json (in the form site/build_site.py's depth_problems requires)
  all         everything above in order (cell for all six), then report

Outputs go to OUT (default /data/tm_depth_modal/audit_depth): parts/*.json (one per stage / cell, reused on rerun unless
--redo), cache/ (own eigenvectors), report.json. "passed" is true only if every required check passed and every
required stage (12 cells, facts, aggregates, pack with inventory) is present.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, eigsh

# ------------------------------------------------------------------ constants, typed from SPEC_DEPTH.md v3.1
LAYERS = 17
WIDTH = 2048
KNN = 15                 # section 4: 15th smallest distance
FLOOR = 1e-6             # section 4: non-edge weight
NEAR = 1e-9              # section 4: pairs within this of a radius are recomputed canonically
SINKHORN_GATE = 1e-10    # section 4: max row-sum error
TIE = 0.01               # section 4: near-tied when eigenvalues differ by under 1% of the lower one
KEEP = 64                # section 4: modes kept
GUARD0, GUARD_MAX = 72, 96
KS = (8, 16, 64)
COND_MAX = 1e8           # section 6: withheld when the tridiagonal condition number exceeds 1e8
OWN_MS_MIN = 1e-12       # section 6: withheld where the set's mean squared coordinate is under 1e-12
RANK_REL = 1e-6          # section 6: singular values above 1e-6 of the largest
STRENGTHS = ("tstar", "tstar_div16")
MODELS = ("pythia", "olmo")
SETS = ("doc00", "doc01", "doc02", "doc04", "pairA", "pairB", "d04", "d08", "d16", "d32", "rand")
AUDITED = ("doc00", "d32", "rand")
PANELS = ("natural", "random", "shuffled")
COHORTS = ("in_set", "seen_sequence_new_position", "unseen_sequence")
SOURCES = (0, 1, 2, 4)
SET_TABLE = {                                   # section 3 table: contexts, tokens per context
    "doc00": ([0], 2047), "doc01": ([1], 2047), "doc02": ([2], 2047), "doc04": ([4], 2047),
    "pairA": ([0, 1], 1024), "pairB": ([2, 4], 1024), "d04": ([0, 1, 2, 4], 512),
    "d08": (list(range(8)), 256), "d16": (list(range(16)), 128), "d32": (list(range(32)), 64),
    "rand": (["r0", "r1", "r2", "r3"], 512),
}
LADDER = ("doc00", "pairA", "d04", "d08", "d16", "d32")
# tolerances of section 8
TOL_ROW_TV = 1e-8
TOL_RESID = 1e-7
TOL_MEAN0 = 1e-9
TOL_ORTHO = 1e-8
TOL_ALIGN = 1e-4
REL_TOL, ABS_TOL = 1e-6, 1e-9     # "floating values to 1e-9 + 1e-6 |value|"

RELEASE = Path("/data/tm_depth_modal")
OUT = Path("/data/tm_depth_modal/audit_depth")
ACTS = {"pythia": Path("/data/tm_acts/pythia_dl/pythia"), "olmo": Path("/data/tm_acts/olmo_dl/olmo")}
SYNTH = Path("/data/tm_acts/depth_acts")
PROBE = Path("/data/tm_acts/probe/probe.json")
HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / "SPEC_DEPTH.md"
ADDENDUM = HERE.parent / "SPEC_DEPTH_ADDENDUM.md"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        return jsonable(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1, allow_nan=False))
    tmp.replace(path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ckey(ctx):
    return f"c{ctx:02d}" if isinstance(ctx, int) else ctx


SYNTH_OVERRIDE = None    # --synth-acts: directory <model>/<ctx>.npy preferred for synthetic contexts when present


def act_file(model, ctx):
    if isinstance(ctx, int):
        return ACTS[model] / f"{ckey(ctx)}.npy"
    if SYNTH_OVERRIDE is not None and (SYNTH_OVERRIDE / model / f"{ctx}.npy").is_file():
        return SYNTH_OVERRIDE / model / f"{ctx}.npy"
    return SYNTH / model / f"{ctx}.npy"


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ------------------------------------------------------------------ check bookkeeping
class Checks:
    """A list of check records: name, scope, passed, worst value, tolerance, required, detail."""

    def __init__(self):
        self.items = []

    def add(self, name, scope, passed, worst=None, tol=None, required=True, **detail):
        self.items.append(dict(name=name, scope=scope, passed=bool(passed), worst=worst, tolerance=tol,
                               required=bool(required), **detail))
        return bool(passed)

    def close(self, name, scope, got, want, tol_abs=ABS_TOL, tol_rel=REL_TOL, required=True, **detail):
        """Element-wise |got - want| <= tol_abs + tol_rel |want|; None/NaN must match None/NaN."""
        g = np.asarray(_floats(got), dtype=float)
        w = np.asarray(_floats(want), dtype=float)
        if g.shape != w.shape:
            return self.add(name, scope, False, None, f"{tol_abs} + {tol_rel}|v|", required,
                            reason=f"shape {g.shape} vs {w.shape}", **detail)
        gn, wn = np.isnan(g), np.isnan(w)
        bad_nan = int((gn ^ wn).sum())
        ok = ~(gn | wn)
        dev = np.abs(g[ok] - w[ok])
        lim = tol_abs + tol_rel * np.abs(w[ok])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(lim > 0, dev / np.where(lim > 0, lim, 1), np.where(dev > 0, np.inf, 0.0))
        worst = float(ratio.max()) if dev.size else 0.0
        nfail = int((dev > lim).sum()) + bad_nan
        ex = None
        if nfail:
            idx = np.flatnonzero(((np.abs(g - w) > tol_abs + tol_rel * np.abs(w)) & ~(gn | wn)).ravel() | (gn ^ wn).ravel())[:3]
            ex = [dict(index=int(i), got=_num(g.ravel()[i]), want=_num(w.ravel()[i])) for i in idx]
        return self.add(name, scope, nfail == 0, worst, f"{tol_abs} + {tol_rel}|v| (worst = max dev / limit)",
                        required, compared=int(g.size), failed=nfail, examples=ex, **detail)

    def same(self, name, scope, got, want, required=True, **detail):
        ok = jsonable(got) == jsonable(want)
        return self.add(name, scope, ok, None, "exact", required,
                        **({} if ok else dict(got=_short(got), want=_short(want))), **detail)

    def gate(self, name, scope, value, tol, required=True, **detail):
        value = float(value)
        return self.add(name, scope, math.isfinite(value) and value < tol, value, tol, required, **detail)


def _num(x):
    x = float(x)
    return None if math.isnan(x) else x


def _floats(x):
    if isinstance(x, (list, tuple)):
        return [_floats(v) for v in x]
    if x is None:
        return float("nan")
    return x


def _short(x):
    s = json.dumps(jsonable(x))
    return s if len(s) < 400 else s[:400] + "..."


# ------------------------------------------------------------------ activations
class Acts:
    """Unit vectors of manifest rows at one layer, from the raw float32 arrays (mmap), in float64."""

    def __init__(self, model):
        self.model = model
        self.maps = {}

    def arr(self, ctx):
        k = ckey(ctx)
        if k not in self.maps:
            a = np.load(act_file(self.model, ctx), mmap_mode="r", allow_pickle=False)
            if a.shape != (LAYERS, 2048, WIDTH) or a.dtype != np.float32:
                raise ValueError(f"unexpected activation array {act_file(self.model, ctx)}: {a.shape} {a.dtype}")
            self.maps[k] = a
        return self.maps[k]

    def raw(self, rows, layer):
        out = np.empty((len(rows), WIDTH), dtype=np.float64)
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[ckey(r[0])].append((i, int(r[1])))
        for k, pairs in groups.items():
            ii = np.array([p[0] for p in pairs])
            pp = np.array([p[1] for p in pairs])
            order = np.argsort(pp)
            out[ii[order]] = self.arr(k if k[0] != "c" else int(k[1:]))[layer, pp[order]]
        return out

    def unit(self, rows, layer):
        x = self.raw(rows, layer)
        nrm = np.linalg.norm(x, axis=1)
        if not np.all(np.isfinite(nrm) & (nrm > 0)):
            raise ValueError("zero or non-finite activation norm")
        x /= nrm[:, None]
        return np.ascontiguousarray(x)


# ------------------------------------------------------------------ section 4: graph and Sinkhorn (own implementation)
def kth(d, k=KNN):
    return np.partition(d, k - 1, axis=1)[:, k - 1].copy()


def site_graph(U):
    """Construction graph at one layer from unit vectors U (n, w) float64: returns distances D (canonical near the
    radii, self = inf), final radii r, edge mask E (bool, symmetric, no diagonal), recomputed pair count."""
    n = len(U)
    D = U @ U.T
    np.subtract(1.0, D, out=D)
    iu = np.triu_indices(n, 1)
    D[(iu[1], iu[0])] = D[iu]                      # one value per pair (the upper one)
    np.fill_diagonal(D, np.inf)
    prov = kth(D)
    cand = np.abs(D - prov[:, None]) <= NEAR
    cand |= np.abs(D - prov[None, :]) <= NEAR
    cand = np.triu(cand | cand.T, 1)
    ii, jj = np.nonzero(cand)
    for i, j in zip(ii.tolist(), jj.tolist()):
        v = 1.0 - np.dot(U[i], U[j])
        D[i, j] = v
        D[j, i] = v
    r = kth(D)
    E = D <= r[:, None]
    E |= D <= r[None, :]
    np.fill_diagonal(E, False)
    return D, r, E, int(len(ii))


def sinkhorn(W, newton_from=1e-4, max_fixed=20000):
    """Positive v with v_i (W v)_i = 1 (W symmetric, nonnegative, zero diagonal): fixed point v <- v / sqrt(v W v),
    then Newton on F(v) = v (W v) - 1 until it stops improving."""
    v = 1.0 / np.sqrt(W.sum(axis=1))
    for _ in range(max_fixed):
        r = v * (W @ v)
        err = np.abs(r - 1).max()
        if err < newton_from:
            break
        v = v / np.sqrt(r)
    best = (np.abs(v * (W @ v) - 1).max(), v)
    for _ in range(30):
        wv = W @ v
        F = v * wv - 1
        J = v[:, None] * W
        J.flat[:: len(v) + 1] += wv
        v2 = v - np.linalg.solve(J, F)
        if not np.all(v2 > 0):
            break
        e2 = np.abs(v2 * (W @ v2) - 1).max()
        if e2 >= best[0]:
            break
        best = (e2, v2)
        v = v2
        if e2 < 1e-15:
            break
    return best[1], float(best[0])


def build_site(U):
    """Everything section 4 defines at one layer. Returns P (symmetric, doubly stochastic), r, v, stats."""
    n = len(U)
    D, r, E, ncand = site_graph(U)
    ties = (D == r[:, None]).sum(axis=1) > 1
    deg = E.sum(axis=1)
    del D
    W = np.where(E, 1.0, FLOOR)
    np.fill_diagonal(W, 0.0)
    v, err = sinkhorn(W)
    P = W
    P *= v[:, None]
    P *= v[None, :]
    P += P.T
    P *= 0.5
    rowerr = float(np.abs(P.sum(axis=1) - 1).max())
    return P, r, v, E, dict(degree_median=float(np.median(deg)), degree_max=int(deg.max()),
                            tie_share=float(ties.mean()), err=rowerr, recomputed_pairs=ncand)


def place_rows(Uq, Us, r_sites, v_sites, mask=None):
    """Section 5 rows of new points Uq (m, w) against a construction (Us, frozen radii r_sites, scalings v_sites).
    mask: index of a construction token to exclude per query, or -1. Returns rows (m, n), nearest cosine (m,)."""
    D = Uq @ Us.T
    np.subtract(1.0, D, out=D)
    if mask is not None:
        for i, j in enumerate(mask):
            if j >= 0:
                D[i, j] = np.inf
    prov = kth(D)
    for i in range(len(Uq)):
        di = D[i]
        cand = np.flatnonzero((np.abs(di - prov[i]) <= NEAR) | (np.abs(di - r_sites) <= NEAR))
        q = Uq[i]
        for j in cand.tolist():
            di[j] = 1.0 - np.dot(q, Us[j])
    rx = kth(D)
    E = (D <= rx[:, None]) | (D <= r_sites[None, :])
    W = np.where(E, 1.0, FLOOR)
    if mask is not None:
        for i, j in enumerate(mask):
            if j >= 0:
                W[i, j] = 0.0
    W *= v_sites[None, :]
    W /= W.sum(axis=1, keepdims=True)
    nearest = 1.0 - D.min(axis=1)
    return W, nearest


# ------------------------------------------------------------------ the joint operator (structured; never dense)
def path_degree(nl=LAYERS):
    d = np.full(nl, 2.0)
    d[0] = d[-1] = 1.0
    return d


def apply_L(P, c, t, F):
    """L_t F for F (L, n, m): blockdiag((I - P_l) / (2 c_l)) + t (D_path x I)."""
    out = np.empty_like(F)
    deg = path_degree(len(c))
    for l in range(len(c)):
        out[l] = (F[l] - P[l] @ F[l]) / (2 * c[l]) + t * deg[l] * F[l]
        if l:
            out[l] -= t * F[l - 1]
        if l < len(c) - 1:
            out[l] -= t * F[l + 1]
    return out


def depth_only_eigs(t, nl=LAYERS):
    return t * 4 * np.sin(np.arange(nl) * np.pi / (2 * nl)) ** 2


def diag_block(Pl, cl, t, degl, sigma):
    n = len(Pl)
    S = Pl * (-1.0 / (2 * cl))
    S.flat[:: n + 1] += 1.0 / (2 * cl) + t * degl - sigma
    return S


class BlockCholesky:
    """(L_t - sigma I)^{-1} for sigma below the spectrum: block LDL^T along depth with Cholesky Schur complements."""

    def __init__(self, P, c, t, sigma):
        nl, n, _ = P.shape
        self.t, self.nl = t, nl
        deg = path_degree(nl)
        self.f = []
        sinv = None
        for l in range(nl):
            S = diag_block(P[l], c[l], t, deg[l], sigma)
            if sinv is not None:
                S -= (t * t) * sinv
                sinv = None
            cf, _ = sla.cho_factor(S, lower=True, overwrite_a=True, check_finite=False)
            self.f.append(cf)
            if l < nl - 1:
                inv, info = sla.lapack.dpotri(cf, lower=1)
                if info:
                    raise RuntimeError(f"dpotri info {info}")
                sinv = np.tril(inv) + np.tril(inv, -1).T
                del inv

    def solve(self, B):
        nl, t = self.nl, self.t
        W = np.empty_like(B)
        for l in range(nl):
            y = B[l] if l == 0 else B[l] + t * W[l - 1]
            W[l] = sla.cho_solve((self.f[l], True), y, check_finite=False)
        X = np.empty_like(B)
        X[-1] = W[-1]
        for l in range(nl - 2, -1, -1):
            X[l] = W[l] + t * sla.cho_solve((self.f[l], True), X[l + 1], check_finite=False)
        return X


def centre_layers(F):
    return F - F.mean(axis=1, keepdims=True)


def lowest_modes(P, c, t, k, seed=11, tol=1e-12):
    """k lowest eigenpairs of L_t on functions with zero mean at every layer (shift-invert Lanczos on the
    projected inverse, then Rayleigh-Ritz on L_t itself). Returns vals (k,), F (L, n, k), residuals (k,)."""
    nl, n, _ = P.shape
    sigma = -1e-3
    fac = BlockCholesky(P, c, t, sigma)

    def op(x):
        f = centre_layers(x.reshape(nl, n, 1))
        return centre_layers(fac.solve(f)).ravel()

    v0 = centre_layers(np.random.default_rng(seed).standard_normal((nl, n, 1))).ravel()
    A = LinearOperator((nl * n, nl * n), matvec=op, dtype=np.float64)
    mu, V = eigsh(A, k=k, which="LA", tol=tol, ncv=min(nl * n - 1, 2 * k + 40), v0=v0, maxiter=100000)
    del fac
    V = centre_layers(V.reshape(nl, n, k))
    Q, _ = np.linalg.qr(V.reshape(nl * n, k))
    Q = Q.reshape(nl, n, k)
    LQ = apply_L(P, c, t, Q)
    H = np.einsum("lni,lnj->ij", Q, LQ)
    vals, Y = np.linalg.eigh((H + H.T) / 2)
    F = np.ascontiguousarray((Q.reshape(-1, k) @ Y).reshape(nl, n, k))
    res = np.sqrt((((LQ.reshape(-1, k) @ Y).reshape(nl, n, k) - F * vals) ** 2).sum(axis=(0, 1)))
    return vals, F, res, dict(ritz_from_inverse=(sigma + 1 / np.sort(mu)[::-1]).tolist())


def inertia_count(P, c, t, sigma, reverse=False):
    """Number of eigenvalues of L_t (full space) below sigma, by Sylvester's law on the block LDL^T factorisation
    (eigendecomposition of each Schur complement); minus the depth-only sector's known eigenvalues below sigma."""
    nl, n, _ = P.shape
    order = range(nl - 1, -1, -1) if reverse else range(nl)
    deg = path_degree(nl)
    neg, smallest = 0, np.inf
    sinv = None
    for l in order:
        S = diag_block(P[l], c[l], t, deg[l], sigma)
        if sinv is not None:
            S -= (t * t) * sinv
            sinv = None
        w, V = sla.eigh(S, overwrite_a=True, check_finite=False, driver="evd")
        del S
        neg += int((w < 0).sum())
        smallest = min(smallest, float(np.abs(w).min()))
        if (reverse and l > 0) or (not reverse and l < nl - 1):
            sinv = (V / w) @ V.T
        del V
    depth = int((depth_only_eigs(t, nl) < sigma).sum())
    return neg - depth, dict(full=neg, depth_only=depth, smallest_pivot_eig=smallest)


def bands_of(vals):
    """Runs of adjacent modes whose eigenvalues differ by under 1% of the lower one (1-based first, last)."""
    vals = np.asarray(vals)
    link = np.diff(vals) < TIE * vals[:-1]
    out, a = [], 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or not link[i - 1]:
            if i - a > 1:
                out.append((a + 1, i))
            a = i
    return out, link


def partition_modes(bands, upto):
    """Complete groups (bands and singletons) covering 1..upto; a band crossing upto is returned separately."""
    groups, crossing = [], None
    k = 1
    bmap = {b[0]: b for b in bands}
    while k <= upto:
        if k in bmap:
            b = bmap[k]
            if b[1] <= upto:
                groups.append(b)
            else:
                crossing = b
                break
            k = b[1] + 1
        else:
            groups.append((k, k))
            k += 1
    return groups, crossing


def tri_matrix(c, t, lam):
    nl = len(c)
    T = np.diag(1.0 / (2 * np.asarray(c)) + t * path_degree(nl) - lam)
    T -= t * (np.eye(nl, k=1) + np.eye(nl, k=-1))
    return T


def procrustes(A, B):
    """Orthogonal R minimising |A R - B| (no rescaling)."""
    U, s, Vt = np.linalg.svd(A.T @ B)
    return U @ Vt, s


# ------------------------------------------------------------------ stage: manifest (section 3)
def stage_manifest(rel, ck):
    m = read_json(rel / "manifest.json")
    probe = read_json(PROBE)
    inputs = read_json(SYNTH / "inputs.json")
    docs = probe["contexts"][:32]
    ids = {ckey(i): list(d["token_ids"]) for i, d in enumerate(docs)}
    for k, v in inputs["contexts"].items():
        ids[k] = list(v["ids"])
    sc = "manifest"
    ck.same("manifest.seed", sc, m.get("seed"), 20261128)
    ck.same("manifest.token_fields", sc, m.get("token_fields"), ["context", "position", "token_id"])
    # contexts
    want_ctx = [ckey(i) for i in range(32)] + [f"r{i}" for i in range(5)] + [f"s{i}" for i in range(4)]
    ck.same("manifest.context_keys", sc, list(m["contexts"]), want_ctx)
    bad = [k for k in want_ctx[:32] if m["contexts"][k] != dict(context=int(k[1:]), kind="natural", subset=docs[int(k[1:])]["subset"])]
    ck.add("manifest.natural_context_records", sc, not bad, failed=bad)
    bad = [k for k in want_ctx[32:37] if m["contexts"][k] != dict(context=k, kind="random")]
    ck.add("manifest.random_context_records", sc, not bad, failed=bad)
    bad = []
    for n, src in enumerate(SOURCES):
        k = f"s{n}"
        rec = m["contexts"][k]
        if not (rec.get("kind") == "shuffled" and rec.get("context") == k and rec.get("source_context") == src
                and rec.get("perm") == inputs["contexts"][k]["perm"]):
            bad.append(k)
    ck.add("manifest.shuffled_context_records", sc, not bad, failed=bad)
    # every row's token id is the text's id at that position (position 1..2047 -> index p-1)
    def check_rows(rows, where):
        wrong = [r for r in rows if not (1 <= r[1] <= 2047) or ids[ckey(r[0])][r[1] - 1] != r[2]]
        ck.add("manifest.row_token_ids", where, not wrong, compared=len(rows), failed=len(wrong), examples=wrong[:3])
    for g in ("sets", "panels", "paths"):
        for name, rows in m[g].items():
            check_rows(rows, f"{g}.{name}")
    # panels: composition, uniqueness
    def per_ctx(rows):
        out = defaultdict(list)
        for r in rows:
            out[ckey(r[0])].append(r[1])
        return out
    pan = {k: per_ctx(v) for k, v in m["panels"].items()}
    ck.same("panel.natural.composition", sc, {k: len(v) for k, v in pan["natural"].items()}, {ckey(i): 32 for i in range(32)})
    ck.same("panel.random.composition", sc, {k: len(v) for k, v in pan["random"].items()},
            {"r0": 128, "r1": 128, "r2": 128, "r3": 128, "r4": 256})
    ck.same("panel.shuffled.composition", sc, {k: len(v) for k, v in pan["shuffled"].items()}, {f"s{i}": 256 for i in range(4)})
    for k, v in m["panels"].items():
        keys = [(ckey(r[0]), r[1]) for r in v]
        ck.add("panel.unique_rows", k, len(set(keys)) == len(keys))
    for name, (ctxs, per) in SET_TABLE.items():
        rows = m["sets"][name]
        pc = per_ctx(rows)
        ck.same("set.composition", name, {k: len(v) for k, v in pc.items()}, {ckey(c): per for c in ctxs})
        keys = [(ckey(r[0]), r[1]) for r in rows]
        ck.add("set.unique_rows", name, len(set(keys)) == len(keys))
        if per == 2047:
            ck.add("set.all_positions", name, all(sorted(v) == list(range(1, 2048)) for v in pc.values()))
        else:
            panel_keys = {(ckey(r[0]), r[1]) for v in m["panels"].values() for r in v}
            clash = [k for k in keys if k in panel_keys]
            ck.add("set.disjoint_from_panels", name, not clash, clashes=len(clash), examples=clash[:3])
        ck.add("set.storage_sorted_by_context_then_position", name,
               keys == sorted(keys, key=lambda k: (want_ctx.index(k[0]), k[1])), required=False)
    # regenerate the draws from the spec's description (assumption: Generator.choice(pop, n, replace=False), sorted)
    rng = np.random.default_rng(20261128)
    pos = np.arange(1, 2048)
    picked, regen_p = {}, {}
    for kind, plan in (("natural", [(ckey(i), 32) for i in range(32)]),
                       ("random", [(f"r{i}", 128 if i < 4 else 256) for i in range(5)]),
                       ("shuffled", [(f"s{i}", 256) for i in range(4)])):
        for k, n in plan:
            picked[k] = np.sort(rng.choice(pos, n, replace=False))
            regen_p.setdefault(kind, {})[k] = picked[k].tolist()
    ok_p = all(regen_p[kind][k] == sorted(v) for kind in pan for k, v in pan[kind].items())
    regen_s, ok_s = {}, True
    for name, (ctxs, per) in SET_TABLE.items():
        for c in ctxs:
            k = ckey(c)
            if per == 2047:
                sel = pos
            else:
                sel = np.sort(rng.choice(np.setdiff1d(pos, picked[k]), per, replace=False))
            if sorted(per_ctx(m["sets"][name])[k]) != sel.tolist():
                ok_s = False
    ck.add("draws.reproduced_from_seed (choice without replacement, sorted)", sc, ok_p and ok_s,
           panels=ok_p, sets=ok_s, note="the spec names the seed and order, not the call; this is the natural reading")
    # cohorts, recomputed from the definitions
    for name, (ctxs, per) in SET_TABLE.items():
        set_keys = {(ckey(r[0]), r[1]) for r in m["sets"][name]}
        ctx_keys = {ckey(c) for c in ctxs}
        for kind, rows in m["panels"].items():
            want = [COHORTS[0] if (ckey(r[0]), r[1]) in set_keys else COHORTS[1] if ckey(r[0]) in ctx_keys else COHORTS[2]
                    for r in rows]
            ck.same("cohorts", f"{name}.{kind}", m["cohorts"][name][kind], want)
            # shuffled tokens must never be "in set" or "seen" (a shuffled text is always an unseen sequence)
            if kind == "shuffled":
                ck.add("cohorts.shuffled_always_unseen", f"{name}", all(c == COHORTS[2] for c in m["cohorts"][name][kind]))
        cnt = {kind: {c: m["cohorts"][name][kind].count(c) for c in COHORTS} for kind in PANELS}
        ck.add("cohorts.counts", name, True, required=False, counts=cnt)
    # references (section 5)
    for name, (ctxs, per) in SET_TABLE.items():
        ref = m["references"][name]
        if name == "rand":
            ok = ref.get("panel") == "random" and ref.get("cohort") == COHORTS[1] and ref.get("unseen_sequence") == COHORTS[2]
        elif per == 2047:
            ok = ref.get("panel") == "natural" and ref.get("cohort") == COHORTS[2] and "other documents" in ref.get("label", "")
        else:
            ok = ref.get("panel") == "natural" and ref.get("cohort") == COHORTS[1]
        ck.add("references", name, ok, record=ref)
    # paths: documents 0, 1, 2, 4 and s0..s3 in full reading order
    ck.same("paths.keys", sc, sorted(m["paths"]), sorted(["c00", "c01", "c02", "c04", "s0", "s1", "s2", "s3"]))
    bad = [k for k, v in m["paths"].items() if [r[1] for r in v] != list(range(1, 2048)) or any(ckey(r[0]) != k for r in v)]
    ck.add("paths.full_reading_order", sc, not bad, failed=bad)
    # synthetic_inputs block equals inputs.json's
    ck.same("manifest.synthetic_inputs", sc, m.get("synthetic_inputs"),
            {k: inputs[k] for k in ("seed_random", "seed_shuffle", "allowed_ids_sha256", "sha256")})
    # 992 natural-panel tokens outside document 0, placed (never in set) in every ladder set
    outside = [i for i, r in enumerate(m["panels"]["natural"]) if r[0] != 0]
    ck.same("ladder.outside_doc00_count", sc, len(outside), 992)
    for name in (*LADDER, "rand"):
        coh = m["cohorts"][name]["natural"]
        ck.add("ladder.outside_doc00_never_in_set", name, all(coh[i] != COHORTS[0] for i in outside))
    # probe: model inputs are the separator then the ids, for both models
    sep = {mm: probe["models"][mm]["eot"] for mm in MODELS}
    bad = [ckey(i) for i, d in enumerate(docs) if not all(d["model_input_ids"][mm] == [sep[mm]] + d["token_ids"] for mm in MODELS)]
    ck.add("probe.model_input_ids_are_separator_then_ids", "32 documents", not bad, failed=bad)
    return m


# ------------------------------------------------------------------ stage: inputs (section 2) and activations
def stage_inputs(rel, ck, hash_acts=True):
    probe = read_json(PROBE)
    inputs = read_json(SYNTH / "inputs.json")
    m = read_json(rel / "manifest.json")
    docs = probe["contexts"][:32]
    sc = "inputs"
    allowed = sorted({t for d in docs for t in d["token_ids"]})
    ck.same("inputs.allowed_count", sc, len(allowed), 9490)
    enc = lambda a: hashlib.sha256(np.asarray(a, dtype="<i4").tobytes()).hexdigest()   # assumption: int32 LE bytes
    ck.add("inputs.allowed_sha256 (int32 little-endian bytes)", sc, enc(allowed) == inputs["allowed_ids_sha256"],
           required=False, note="the encoding is not stated in the spec")
    ck.same("inputs.seeds", sc, [inputs["seed_random"], inputs["seed_shuffle"]], [20261126, 20261127])
    draws = np.random.default_rng(20261126).choice(np.asarray(allowed), size=(5, 2047))   # with replacement
    for i in range(5):
        k = f"r{i}"
        ck.same("inputs.random_ids_regenerated", k, inputs["contexts"][k]["ids"], draws[i].tolist())
        ck.add("inputs.sha256", k, enc(inputs["contexts"][k]["ids"]) == inputs["sha256"][k], required=False)
    for n, src in enumerate(SOURCES):
        k = f"s{n}"
        rec = inputs["contexts"][k]
        perm = np.random.default_rng(20261127 + n).permutation(2047)
        ck.same("inputs.shuffle_perm_regenerated (1-based)", k, rec["perm"], (perm + 1).tolist())
        ck.same("inputs.shuffle_source", k, rec["source_context"], src)
        ck.same("inputs.shuffle_ids_follow_perm", k, rec["ids"], [docs[src]["token_ids"][p - 1] for p in rec["perm"]])
        ck.add("inputs.sha256", k, enc(rec["ids"]) == inputs["sha256"][k], required=False)
    # the extraction's own provenance for the synthetic contexts
    for model in MODELS:
        p = SYNTH / model / "provenance.json"
        ck.add("inputs.provenance_present", model, p.is_file(), required=False)
    # raw activations: hashes against every unit's input_files.json
    if hash_acts:
        seen = {}
        for model in MODELS:
            for name in SETS:
                want = read_json(rel / "units" / model / name / "input_files.json")["activations"]
                bad = []
                for k, h in want.items():
                    f = act_file(model, int(k[1:]) if k[0] == "c" else k)
                    if (model, k) not in seen:
                        seen[(model, k)] = sha256_file(f)
                    if seen[(model, k)] != h:
                        bad.append(k)
                ck.add("activations.sha256_match_unit_inputs", f"{model}/{name}", not bad, failed=bad, compared=len(want))
    # layer-0 identity: every row at layer 0 equals the embedding of its token id (the same vector at every
    # occurrence in every context); synthetic rows match natural rows of the same id; shuffled rows equal their
    # originals; position 0 is the same separator vector in every context.
    ids = {ckey(i): d["token_ids"] for i, d in enumerate(docs)}
    for k, v in inputs["contexts"].items():
        ids[k] = v["ids"]
    for model in MODELS:
        acts = Acts(model)
        ref, sep, bad_rows, bad_sep, checked = {}, None, [], [], 0
        layer0 = {}
        for k in list(ids):
            a = np.asarray(acts.arr(int(k[1:]) if k[0] == "c" else k)[0])
            layer0[k] = a
            rows = a[1:]
            hashes = [hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in rows]
            for p, (t, h) in enumerate(zip(ids[k], hashes), start=1):
                checked += 1
                if ref.setdefault(t, h) != h:
                    bad_rows.append((k, p, t))
            hs = hashlib.blake2b(a[0].tobytes(), digest_size=16).digest()
            if sep is None:
                sep = hs
            elif hs != sep:
                bad_sep.append(k)
        # distinct ids should have distinct embeddings (otherwise this test could not tell rows apart)
        dup = len(ref) - len(set(ref.values()))
        ck.add("activations.layer0_row_is_embedding_of_its_id", model, not bad_rows, compared=checked,
               failed=len(bad_rows), examples=bad_rows[:5], distinct_ids=len(ref), ids_sharing_a_vector=dup)
        ck.add("activations.separator_row_identical", model, not bad_sep, failed=bad_sep)
        for n, src in enumerate(SOURCES):
            k = f"s{n}"
            perm = np.asarray(inputs["contexts"][k]["perm"])
            same = np.array_equal(layer0[k][1:], layer0[ckey(src)][perm])
            ck.add("activations.shuffled_layer0_equals_original_at_perm", f"{model}/{k}", same)
            # and layer 16 must differ (a real re-run, not a copy)
            a16s = np.asarray(acts.arr(k)[16, 1:])
            a16o = np.asarray(acts.arr(src)[16])[perm]
            ck.add("activations.shuffled_layer16_differs_from_original", f"{model}/{k}",
                   not np.array_equal(a16s, a16o), required=False)
        del layer0


# ------------------------------------------------------------------ stage: one audited cell
def rng_sample(n, size, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size, replace=False))


def stage_cell(rel, out, model, name, ck, redo=False):
    """Heavy checks for one (model, set) at both strengths. Writes parts/cell_<model>_<name>.json."""
    t0 = time.time()
    m = read_json(rel / "manifest.json")
    rows = m["sets"][name]
    n = len(rows)
    udir = rel / "units" / model / name
    sites = np.load(udir / "sites.npz")
    unit = read_json(udir / "unit.json")
    facts = {s: read_json(udir / s / "facts.json") for s in STRENGTHS}
    acts = Acts(model)
    sc = f"{model}/{name}"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # --- section 4 rebuild, all 17 layers (needed for the solve); compare with the release
    P = np.empty((LAYERS, n, n))
    my_r = np.empty((LAYERS, n))
    my_v = np.empty((LAYERS, n))
    stats = []
    audit = np.load(udir / "audit_rows.npz")
    arows, aidx, alayers = audit["rows"], audit["indices"], audit["layers"]
    ck.same("audit_rows.layers", sc, alayers.tolist(), [0, 8, 16])
    ck.same("audit_rows.count", sc, len(aidx), 64)
    tv_audit, tv_full, edge_diff = [], [], []
    site_vals = np.empty((LAYERS, 65))
    site_vecs = np.empty((LAYERS, n, 65), dtype=np.float32)
    for l in range(LAYERS):
        U = acts.unit(rows, l)
        Pl, r, v, E, st = build_site(U)
        P[l], my_r[l], my_v[l] = Pl, r, v
        stats.append(st)
        # implied release matrix: release radii and scalings with this audit's distances
        D = 1.0 - U @ U.T
        iu = np.triu_indices(n, 1)
        D[(iu[1], iu[0])] = D[iu]
        np.fill_diagonal(D, np.inf)
        cand = np.triu((np.abs(D - sites["radii"][l][:, None]) <= NEAR) | (np.abs(D - sites["radii"][l][None, :]) <= NEAR), 1)
        for i, j in zip(*np.nonzero(cand)):
            D[i, j] = D[j, i] = 1.0 - np.dot(U[i], U[j])
        # Radii agree only to rounding (this audit normalises differently, by design), and a token's 15th
        # neighbour sits exactly at its radius; so an edge counts as differing only when it is decided by more
        # than 1e-12 under the release's radii. The scaling comparison below uses this audit's own edges.
        rr = sites["radii"][l]
        clear = (np.abs(D - rr[:, None]) > 1e-12) & (np.abs(D - rr[None, :]) > 1e-12)
        Er = (D <= rr[:, None]) | (D <= rr[None, :])
        np.fill_diagonal(Er, False)
        edge_diff.append(int(((Er != E) & clear).sum()))
        Wr = np.where(E, 1.0, FLOOR)
        np.fill_diagonal(Wr, 0.0)
        Wr *= sites["sinkhorn_scaling"][l][:, None] * sites["sinkhorn_scaling"][l][None, :]
        tv_full.append(float(0.5 * np.abs(Wr - Pl).sum(axis=1).max()))
        del D, Er, Wr, E, U
        if l in (0, 8, 16):
            j = [0, 8, 16].index(l)
            tv = 0.5 * np.abs(arows[j] - Pl[aidx]).sum(axis=1)
            tv_audit.append(float(tv.max()))
        w, V = sla.eigh(Pl, subset_by_index=[n - 65, n - 1], check_finite=False)
        site_vals[l] = w[::-1]
        site_vecs[l] = V[:, ::-1]
        del V
        log(f"{sc} layer {l}: deg median {st['degree_median']} max {st['degree_max']} tie {st['tie_share']:.3f} "
            f"err {st['err']:.2g} recomputed {st['recomputed_pairs']} edge-diff {edge_diff[-1]} rss {peak_rss_gb():.2f} GB")
    ck.gate("rows.P_rebuilt_vs_audit_rows_TV (layers 0, 8, 16; 64 rows)", sc, max(tv_audit), TOL_ROW_TV, per_layer=tv_audit)
    ck.gate("rows.P_own_edges_with_release_scalings_TV (all rows, all layers)", sc, max(tv_full), TOL_ROW_TV,
            required=False, per_layer=tv_full)
    ck.same("rows.edges_decided_by_release_radii_equal_own_edges (count differing by more than 1e-12)", sc, edge_diff, [0] * LAYERS, required=False)
    ck.close("sites.radii", sc, my_r, sites["radii"], tol_abs=1e-12, tol_rel=0)
    ck.close("sites.sinkhorn_scaling", sc, my_v, sites["sinkhorn_scaling"], tol_abs=0, tol_rel=1e-8)
    ck.gate("sites.own_sinkhorn_row_error", sc, max(s["err"] for s in stats), SINKHORN_GATE)
    for key in ("degree_median", "degree_max", "tie_share"):
        ck.same(f"sites.{key}", sc, [s[key] for s in stats], [s[key] for s in unit["sites"]])
    ck.gate("sites.release_row_error_under_gate", sc, max(s["err"] for s in unit["sites"]), SINKHORN_GATE)
    ck.close("sites.top_eigenvalue_is_one", sc, site_vals[:, 0], np.ones(LAYERS), tol_abs=1e-10, tol_rel=0)
    my_c = (1 - site_vals[:, 1:9]).mean(axis=1) / 2          # 1 - eig(M) = (1 - eig(P)) / 2
    ck.close("sites.c", sc, my_c, sites["c"])
    ck.close("sites.eigenvalues (P, nontrivial 1..64)", sc, site_vals[:, 1:65], sites["eigenvalues"])
    rel_site = np.load(udir / "site_modes.npy").astype(np.float64)
    sres = np.empty((LAYERS, 64))
    for l in range(LAYERS):
        V = rel_site[l]
        sres[l] = np.linalg.norm(P[l] @ V - V * sites["eigenvalues"][l], axis=0)
    ck.gate("sites.release_site_modes_residual (float32 vectors)", sc, float(sres.max()), 1e-5, required=False)
    ck.same("unit.n", sc, unit["n"], n)
    del rel_site

    # --- the joint solves
    results = {}
    mine = {}
    for s in STRENGTHS:
        f = facts[s]
        t = f["t"]
        ck.same("facts.t", f"{sc}/{s}", t, unit["tstar"] if s == "tstar" else unit["tstar"] / 16)
        G = len(f["guard_eigenvalues"])
        K = min(G + 8, 104)
        cf = cache / f"{model}_{name}_{s}_modes.npz"
        if cf.is_file() and not redo:
            z = np.load(cf)
            vals, F, res = z["vals"], z["F"], z["res"]
        else:
            log(f"{sc}/{s}: solving {K} modes, t={t:.6g}")
            ts = time.time()
            vals, F, res, _ = lowest_modes(P, my_c, t, K)
            np.savez(cf, vals=vals, F=F, res=res)
            log(f"{sc}/{s}: solved in {time.time() - ts:.0f} s, max residual {res.max():.2g}, rss {peak_rss_gb():.2f} GB")
        mine[s] = (vals, F)
        scs = f"{sc}/{s}"
        ck.gate("solve.own_residual", scs, float(res.max()), TOL_RESID, required=False)
        ck.close("solve.eigenvalues_vs_guard_eigenvalues", scs, vals[:G], f["guard_eigenvalues"])
        ck.same("facts.eigenvalues_are_first_64_guard", scs, f["eigenvalues"], f["guard_eigenvalues"][:KEEP])
        # release float64 vectors with the structured operator
        F64 = np.load(udir / s / "modes_f64.npy")
        lam = np.asarray(f["eigenvalues"])
        R = apply_L(P, my_c, t, F64) - F64 * lam
        rres = np.sqrt((R ** 2).sum(axis=(0, 1)))
        del R
        ck.gate("solve.release_f64_residual |L f - lambda f|", scs, float(rres.max()), TOL_RESID)
        ck.gate("solve.release_f64_mean_zero_every_layer (max |sum_i f_l(i)|)", scs, float(np.abs(F64.sum(axis=1)).max()), TOL_MEAN0)
        Gm = np.einsum("lni,lnj->ij", F64, F64)
        ck.gate("solve.release_f64_orthonormal (max |F'F - I|)", scs, float(np.abs(Gm - np.eye(KEEP)).max()), TOL_ORTHO)
        F32 = np.load(udir / s / "modes.npy")
        ck.add("release.float32_equals_cast_of_float64", scs, F32.dtype == np.float32 and np.array_equal(F32, F64.astype(np.float32)))
        ck.gate("facts.residual_list_under_gate", scs, max(f["residual"]), TOL_RESID)
        ck.gate("facts.residual_list_vs_own_recomputation (both rounding-sized; max |difference|)", scs,
                float(np.abs(np.asarray(f["residual"][:KEEP]) - rres).max()), 1e-10, required=False)
        # bands from own eigenvalues vs the release's
        mb, _ = bands_of(vals)
        rb, _ = bands_of(f["guard_eigenvalues"])
        ck.same("bands.release_list_recomputed_from_guard", scs, [[b["first"], b["last"]] for b in f["bands"]], [list(b) for b in rb])
        mine_upto = [list(b) if b[1] < G else [b[0], min(b[1], G)] for b in mb if b[0] <= G]
        ck.same("bands.own_eigenvalues_give_same_bands (up to the release's guard)", scs, mine_upto, [list(b) for b in rb], required=False)
        last_release_band = rb[-1] if rb else None
        if last_release_band and last_release_band[1] == G and G < GUARD_MAX:
            ck.add("solve.guard_extended_while_last_band_reaches_end", scs, False, note=f"band {last_release_band} reaches G={G} < 96")
        else:
            ck.add("solve.guard_extended_while_last_band_reaches_end", scs, True, guard=G)
        # t* as the crossing: lowest mean-zero eigenvalue equals 4 t sin^2(pi / 34) at t*
        if s == "tstar":
            dep = depth_only_eigs(t)[1]
            g = vals[0] - dep
            f1 = F[:, :, 0]
            dl = float(((f1[1:] - f1[:-1]) ** 2).sum())               # d lambda_1 / dt (Hellmann-Feynman)
            slope = dl - 4 * math.sin(math.pi / 34) ** 2
            t_new = t - g / slope
            ck.close("tstar.crossing (lambda_1 vs 4 t* sin^2(pi/34))", scs, vals[0], dep)
            ck.close("tstar.own_newton_estimate", scs, t_new, t, required=True, gap_1_2=float(vals[1] - vals[0]))
        # independent count at resolved boundaries
        counts = {}
        guard = np.asarray(f["guard_eigenvalues"])
        for k in KS:
            if guard[k] - guard[k - 1] < TIE * guard[k - 1]:
                counts[str(k)] = dict(countable=False, reason="boundary inside a near-tied band")
                ck.add("count.eigenvalues_below_midpoint", f"{scs}/k={k}", True, required=False, countable=False,
                       note="boundary inside a band: not countable (reported, not a failure)")
                continue
            mid = 0.5 * (guard[k - 1] + guard[k])
            cpath = out / "parts" / f"count_{model}_{name}_{s}_{k}.json"
            if cpath.is_file() and not redo:
                cnt, info = read_json(cpath)
            else:
                ts = time.time()
                cnt, info = inertia_count(P, my_c, t, mid)
                info["seconds"] = time.time() - ts
                write_json(cpath, [cnt, info])
            dist = float(np.abs(depth_only_eigs(t) - mid).min())
            counts[str(k)] = dict(countable=True, midpoint=mid, count=cnt, **info, nearest_depth_only_distance=dist)
            ck.same("count.eigenvalues_below_midpoint", f"{scs}/k={k}", cnt, k, midpoint=mid, **info)
            log(f"{scs}: count below midpoint after {k} = {cnt} ({info})")
        results[s] = dict(counts=counts, release_residual_max=float(rres.max()), own_residual_max=float(res.max()),
                          K=K, G=G, own_bands=mb, release_bands=rb)
        del F64, F32
    del P

    # --- own placement (section 5) of every panel token and a sample of path tokens, both strengths
    want = {}
    for kind in PANELS:
        want[("panels", kind)] = m["panels"][kind]
    path_sample = {}
    if name in ("d32", "rand"):
        for j, (k, prow) in enumerate(sorted(m["paths"].items())):
            idx = rng_sample(len(prow), 64, 20261126 + 100 + j)
            path_sample[k] = idx
            want[("paths", k)] = [prow[i] for i in idx]
    key_index = {(ckey(r[0]), r[1]): i for i, r in enumerate(rows)}
    replay_idx = rng_sample(n, 32, 777)                     # own-code self-test: construction tokens, self masked
    want[("replay", "own")] = [rows[i] for i in replay_idx]
    qrows, qtag, qmask = [], [], []
    for key, rr in want.items():
        for i, r in enumerate(rr):
            j = key_index.get((ckey(r[0]), r[1]), -1)
            qrows.append(r)
            qtag.append((key, i, j))
            qmask.append(j if key[0] == "replay" else -1)
    nq = len(qrows)
    placed = np.array([j < 0 or key[0] == "replay" for key, i, j in qtag])
    pidx = np.flatnonzero(placed)
    Bc = {s: np.zeros((LAYERS, nq, KEEP)) for s in STRENGTHS}
    nearest = np.ones((LAYERS, nq))
    replay_tv = np.zeros(LAYERS)
    BLOCK = 512
    for l in range(LAYERS):
        Us = acts.unit(rows, l)
        for a in range(0, len(pidx), BLOCK):
            ii = pidx[a:a + BLOCK]
            Uq = acts.unit([qrows[i] for i in ii], l)
            msk = np.array([qmask[i] for i in ii])
            W, near = place_rows(Uq, Us, my_r[l], my_v[l], mask=msk)
            nearest[l, ii] = near
            for s in STRENGTHS:
                Bc[s][l, ii] = W @ mine[s][1][l, :, :KEEP] / (2 * my_c[l])
            rp = msk >= 0
            if rp.any():
                # the construction row: rebuilt P row of that token (from my_r/my_v and canonical distances)
                Ui = Us[msk[rp]]
                Ds = 1.0 - Ui @ Us.T
                Pi = np.zeros((len(Ui), len(Us)))
                for q, j in enumerate(msk[rp]):
                    d = Ds[q].copy()
                    d[j] = np.inf
                    cnd = np.flatnonzero((np.abs(d - my_r[l][j]) <= NEAR) | (np.abs(d - my_r[l]) <= NEAR))
                    for jj in cnd.tolist():
                        d[jj] = 1.0 - np.dot(Us[j], Us[jj])
                    e = (d <= my_r[l][j]) | (d <= my_r[l])
                    w = np.where(e, 1.0, FLOOR) * my_v[l] * my_v[l][j]
                    w[j] = 0.0
                    Pi[q] = w
                replay_tv[l] = max(replay_tv[l], float(0.5 * np.abs(W[rp] - Pi).sum(axis=1).max()))
            del W, Uq
        del Us
        log(f"{sc}: placed layer {l} ({len(pidx)} rows), rss {peak_rss_gb():.2f} GB")
    ck.gate("own_code.replay_rows_equal_construction_rows (TV)", sc, float(replay_tv.max()), 1e-9, required=False)

    for s in STRENGTHS:
        scs = f"{sc}/{s}"
        vals, F = mine[s]
        t = facts[s]["t"]
        Y = np.full((LAYERS, nq, KEEP), np.nan)
        for k in range(KEEP):
            T = tri_matrix(my_c, t, vals[k])
            Y[:, pidx, k] = np.linalg.solve(T, Bc[s][:, pidx, k])
        for q, (key, i, j) in enumerate(qtag):
            if not placed[q]:
                Y[:, q] = F[:, j, :KEEP]
        rsel = np.array([q for q, (key, i, j) in enumerate(qtag) if key[0] == "replay"])
        rj = np.array([qtag[q][2] for q in rsel])
        rep_err = float(np.abs(Y[:, rsel] - F[:, rj, :KEEP]).max() / np.sqrt((F[:, :, :KEEP] ** 2).mean()))
        ck.gate("own_code.replay_coordinates_equal_own_modes (max err / rms)", scs, rep_err, 1e-7, required=False)
        # band-level alignment with no rescaling, fitted on the construction tokens
        rel32 = np.load(udir / s / "modes.npy").astype(np.float64)
        release_bands = [tuple(b) for b in results[s]["release_bands"]]
        own_bands = [tuple(b) for b in results[s]["own_bands"]]
        groups, crossing = partition_modes(release_bands, KEEP)
        if partition_modes(own_bands, KEEP) != (groups, crossing):
            # merge the two partitions (a rotation within a larger group is still orthogonal with no rescaling)
            merged = []
            for b in sorted(set(release_bands) | set(own_bands)):
                if merged and b[0] <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], b[1]))
                else:
                    merged.append(b)
            groups, crossing = partition_modes(merged, KEEP)
        rel_panels = {kind: np.load(udir / "panels" / f"{kind}.npz") for kind in PANELS}
        rel_paths = {k: np.load(udir / "paths" / f"{k}.npz") for k in path_sample}
        align = []
        worst = dict(construction=0.0, panels=0.0, paths=0.0)
        for a, b in groups:
            A = F[:, :, a - 1:b].reshape(-1, b - a + 1)
            B = rel32[:, :, a - 1:b].reshape(-1, b - a + 1)
            Rm, sv = procrustes(A, B)
            rms = float(np.sqrt((B ** 2).mean()))
            e_con = float(np.abs(A @ Rm - B).max() / rms)
            e_pan = 0.0
            for kind in PANELS:
                qs = [q for q, (key, i, j) in enumerate(qtag) if key == ("panels", kind)]
                Yq = Y[:, qs, a - 1:b] @ Rm
                Rq = rel_panels[kind][s][:, :, a - 1:b].astype(np.float64)
                e_pan = max(e_pan, float(np.abs(Yq - Rq).max() / rms))
            e_path = 0.0
            for k, idx in path_sample.items():
                qs = [q for q, (key, i, j) in enumerate(qtag) if key == ("paths", k)]
                Yq = Y[:, qs, a - 1:b] @ Rm
                Rq = rel_paths[k][s][:, idx, a - 1:b].astype(np.float64)
                e_path = max(e_path, float(np.abs(Yq - Rq).max() / rms))
            worst["construction"] = max(worst["construction"], e_con)
            worst["panels"] = max(worst["panels"], e_pan)
            worst["paths"] = max(worst["paths"], e_path)
            align.append(dict(first=a, last=b, construction=e_con, panels=e_pan, paths=e_path if path_sample else None,
                              procrustes_sv_min=float(sv.min())))
        aligned_modes = sum(b - a + 1 for a, b in groups)
        ck.gate("coords.band_alignment_construction (max err / band rms)", scs, worst["construction"], TOL_ALIGN,
                aligned_modes=aligned_modes, groups=len(groups), not_aligned=crossing)
        ck.gate("coords.band_alignment_panels (same rotation, all panel tokens)", scs, worst["panels"], TOL_ALIGN,
                aligned_modes=aligned_modes)
        if path_sample:
            ck.gate("coords.band_alignment_path_sample (64 tokens per path)", scs, worst["paths"], TOL_ALIGN, required=False)
        # the band that crosses mode 64 (not alignable by the spec's rule): is the release's part of it inside
        # the span of own modes of the whole band (own guard)?
        if crossing:
            a = crossing[0]
            hi = min(crossing[1], len(vals))
            A = F[:, :, a - 1:hi].reshape(-1, hi - a + 1)
            B = rel32[:, :, a - 1:KEEP].reshape(-1, KEEP - a + 1)
            proj = A @ (A.T @ B)
            rms = float(np.sqrt((B ** 2).mean()))
            ck.gate("coords.crossing_band_in_own_span (informational; spec: not aligned)", scs,
                    float(np.abs(proj - B).max() / rms), TOL_ALIGN, required=False, band=crossing, own_span=[a, hi],
                    own_band_resolved=hi < len(vals))
        # placed tokens of withheld modes must be NaN in the release; others finite
        wh = np.asarray(facts[s]["extension"]["withheld"])
        fin_ok = all(np.all(np.isfinite(rel_panels[kd][s][:, :, ~wh])) and np.all(np.isnan(rel_panels[kd][s][:, :, wh]))
                     for kd in PANELS)
        ck.add("release.panel_finite_except_withheld", scs, fin_ok)
        # in-set panel rows are exact copies of the release modes
        for kind in PANELS:
            z = rel_panels[kind]
            ins = z["in_set_index"]
            exp = np.array([key_index.get((ckey(r[0]), r[1]), -1) for r in m["panels"][kind]])
            ck.same("release.panel_in_set_index", f"{scs}/{kind}", ins.tolist(), exp.tolist())
            sel = np.flatnonzero(ins >= 0)
            if sel.size:
                ck.add("release.in_set_rows_copy_modes", f"{scs}/{kind}",
                       np.array_equal(z[s][:, sel], np.load(udir / s / "modes.npy")[:, ins[sel]]))
        results[s]["alignment"] = align
        results[s]["crossing_band"] = crossing
        del rel32
    # nearest cosine (support) vs the release
    worst_n = 0.0
    for kind in PANELS:
        qs = [q for q, (key, i, j) in enumerate(qtag) if key == ("panels", kind)]
        rn = np.load(udir / "panels" / f"{kind}.npz")["nearest_cosine"]
        worst_n = max(worst_n, float(np.abs(nearest[:, qs] - rn).max()))
    ck.gate("support.nearest_cosine_vs_own (all panel tokens, all layers)", sc, worst_n, 1e-12)
    write_json(out / "parts" / f"cell_{model}_{name}_detail.json",
               dict(stats=stats, results=results, seconds=time.time() - t0, peak_rss_gb=peak_rss_gb(),
                    tv_audit=tv_audit, tv_full=tv_full, edge_diff=edge_diff))
    log(f"{sc}: cell done in {time.time() - t0:.0f} s, peak rss {peak_rss_gb():.2f} GB")


# ------------------------------------------------------------------ stage: facts (section 6) for every unit
def embedding_fields(f, rows):
    """Section 6 per-mode measures of modes f (L, n, k) (float64)."""
    w = (f ** 2).sum(axis=1)                       # (L, k) depth weight
    tot = w.sum(axis=0)
    p = (f ** 2).sum(axis=0) / tot                 # (n, k)
    support = 1.0 / (f.shape[1] * (p ** 2).sum(axis=0))
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[ckey(r[0])].append(i)
    if len(groups) > 1:
        between = np.zeros(f.shape[2])
        for ii in groups.values():
            mu = f[:, ii].mean(axis=1)             # (L, k)
            between += len(ii) * (mu ** 2).sum(axis=0)
        share = between / tot
    else:
        share = None
    weak = w < 1e-6
    nrm = np.sqrt(w)
    adj = np.einsum("lnk,lnk->lk", f[1:], f[:-1]) / (nrm[1:] * nrm[:-1])
    adj_ok = ~(weak[1:] | weak[:-1])
    cnt = adj_ok.sum(axis=0)
    adj_mean = np.array([adj[adj_ok[:, k], k].mean() if cnt[k] else np.nan for k in range(f.shape[2])])
    end = np.einsum("nk,nk->k", f[0], f[-1]) / (nrm[0] * nrm[-1])
    end[weak[0] | weak[-1]] = np.nan
    return dict(depth_weight=w, token_support=support, document_share=share, weak_slice=weak,
                adjacent_cos_mean=adj_mean, adjacent_cos_count=cnt, endpoint_cos=end)


def placement_measures(Y, own, cohort_idx, k):
    """Section 6 cohort measures for the top k modes: Y (L, m, >=k) placed coordinates, own (L, n, >=k)."""
    y = Y[:, cohort_idx, :k].astype(np.float64)
    o = own[:, :, :k].astype(np.float64)
    own_ms = (o ** 2).mean(axis=(1, 2))
    coh_ms = (y ** 2).mean(axis=(1, 2))
    live = own_ms >= OWN_MS_MIN
    rr = np.where(live, np.sqrt(coh_ms / np.where(live, own_ms, 1)), np.nan)
    mean_vec = y.mean(axis=1)                       # (L, k)
    rms_radius = np.sqrt(k * own_ms)               # root mean square over tokens of the k-dim norm
    off = np.where(live, np.linalg.norm(mean_vec, axis=1) / np.where(live, rms_radius, 1), np.nan)
    pooled_rr = math.sqrt(coh_ms.mean() / own_ms.mean()) if own_ms.mean() >= OWN_MS_MIN else None
    pooled_off = math.sqrt(float(np.mean(off ** 2))) if live.all() else None
    return dict(radius_ratio=rr, centre_offset=off, pooled_radius_ratio=pooled_rr, pooled_centre_offset=pooled_off,
                own_mean_square=own_ms)


def check_extremes(ck, scope, f, ext):
    """The release's 8 lowest / highest tokens must carry the 8 lowest / highest values (ties allowed)."""
    bad = 0
    for l in ("0", "8", "16"):
        x = f[int(l)]
        srt = np.sort(x, axis=0)
        for k in range(x.shape[1]):
            lo = np.asarray(ext[l]["low"][k])
            hi = np.asarray(ext[l]["high"][k])
            if not (np.array_equal(x[lo, k], srt[:8, k]) and np.array_equal(x[hi, k], srt[::-1][:8, k])
                    and len(set(lo.tolist())) == 8 and len(set(hi.tolist())) == 8):
                bad += 1
    ck.add("facts.extremes (8 lowest / highest at layers 0, 8, 16)", scope, bad == 0, failed_modes=bad)


def stage_facts(rel, ck, models=MODELS, sets=SETS):
    m = read_json(rel / "manifest.json")
    for model in models:
        for name in sets:
            udir = rel / "units" / model / name
            unit = read_json(udir / "unit.json")
            sites = np.load(udir / "sites.npz")
            rows = m["sets"][name]
            sc = f"{model}/{name}"
            ck.same("unit.n", sc, unit["n"], len(rows))
            ck.same("unit.reference", sc, unit["reference"], m["references"][name])
            want_ds = {f"panels/{k}": dict(count=len(m["panels"][k]), in_set=m["cohorts"][name][k].count("in_set")) for k in PANELS}
            if name in ("d32", "rand"):
                key_index = {(ckey(r[0]), r[1]) for r in rows}
                for k, pr in m["paths"].items():
                    want_ds[f"paths/{k}"] = dict(count=len(pr), in_set=sum((ckey(r[0]), r[1]) in key_index for r in pr))
            ck.same("unit.datasets", sc, unit["datasets"], want_ds)
            ck.same("unit.missing", sc, unit["missing"], dict(panels=[], paths=[]))
            ck.same("unit.sites_count", sc, len(unit["sites"]), LAYERS)
            ck.gate("unit.sinkhorn_row_error (release)", sc, max(s["err"] for s in unit["sites"]), SINKHORN_GATE)
            done = read_json(udir / "done.json")
            bad = [p for p, h in done["files"].items() if sha256_file(udir / p) != h]
            ck.add("release.files_match_done_json", sc, not bad, failed=bad[:5], compared=len(done["files"]))
            own = {}
            for s in STRENGTHS:
                scs = f"{sc}/{s}"
                f = read_json(udir / s / "facts.json")
                modes = np.load(udir / s / "modes.npy")
                own[s] = modes
                g = np.asarray(f["guard_eigenvalues"])
                ck.same("facts.t", scs, f["t"], unit["tstar"] if s == "tstar" else unit["tstar"] / 16)
                ck.add("facts.guard_sorted_and_sized", scs, bool(np.all(np.diff(g) >= 0)) and len(g) in (72, 80, 88, 96), G=len(g))
                ck.same("facts.eigenvalues_are_first_64_guard", scs, f["eigenvalues"], g[:KEEP].tolist())
                ck.close("facts.gaps (lambda_{k+1} - lambda_k)", scs, [f["gaps"][str(k)] for k in KS], [g[k] - g[k - 1] for k in KS])
                rb, link = bands_of(g)
                ck.same("facts.bands", scs, [[b["first"], b["last"]] for b in f["bands"]], [list(b) for b in rb])
                ck.same("facts.bands.unresolved", scs, [b["unresolved"] for b in f["bands"]],
                        [bool(b[1] == len(g) and link[-1]) for b in rb])
                ck.same("facts.bands.complete_in_release", scs, [b["complete_in_release"] for b in f["bands"]],
                        [b[1] <= KEEP for b in rb])
                ck.same("facts.cuts_band", scs, f["cuts_band"], {str(k): bool(link[k - 1]) for k in KS})
                last = rb[-1] if rb else None
                ck.add("facts.guard_extension_rule", scs, not (last and last[1] == len(g) and len(g) < GUARD_MAX), G=len(g), last_band=last)
                ck.gate("facts.max_residual", scs, f["max_residual"], TOL_RESID)
                ck.same("facts.max_residual_is_max_of_list", scs, f["max_residual"], max(f["residual"]))
                ck.same("release.modes_shape", scs, list(modes.shape), [LAYERS, len(rows), KEEP])
                ck.add("release.modes_finite", scs, bool(np.all(np.isfinite(modes))))
                # fields from the float32 release
                mf = embedding_fields(modes.astype(np.float64), rows)
                ff = f["fields"]
                ck.close("facts.depth_weight", scs, ff["depth_weight"], mf["depth_weight"])
                ck.close("facts.token_support", scs, ff["token_support"], mf["token_support"])
                if mf["document_share"] is None:
                    ck.same("facts.document_share (n/a for one document)", scs, ff["document_share"], None)
                else:
                    ck.close("facts.document_share", scs, ff["document_share"], mf["document_share"])
                ck.same("facts.weak_slice", scs, ff["weak_slice"], mf["weak_slice"].tolist())
                ck.close("facts.adjacent_cos_mean", scs, ff["adjacent_cos_mean"], mf["adjacent_cos_mean"])
                ck.same("facts.adjacent_cos_count", scs, ff["adjacent_cos_count"], mf["adjacent_cos_count"].tolist())
                ck.close("facts.endpoint_cos", scs, ff["endpoint_cos"], mf["endpoint_cos"])
                check_extremes(ck, scs, modes.astype(np.float64), ff["extremes"])
                # extension diagnostics
                ext = f["extension"]
                Ts = [tri_matrix(sites["c"], f["t"], lam) for lam in f["eigenvalues"]]
                sv = np.array([np.linalg.svd(T, compute_uv=False) for T in Ts])
                cond = sv[:, 0] / sv[:, -1]
                ck.close("facts.extension.smallest_sv", scs, ext["smallest_sv"], sv[:, -1])
                ck.close("facts.extension.condition", scs, ext["condition"], cond)
                ck.same("facts.extension.withheld (condition > 1e8)", scs, ext["withheld"], (cond > COND_MAX).tolist())
                mr = np.asarray(_floats(ext["max_relative_residual"]), dtype=float)
                ck.gate("facts.extension.max_relative_residual (no gate stated; 1e-8 assumed)", scs,
                        float(np.nanmax(mr)) if np.isfinite(mr).any() else 0.0, 1e-8, required=False)
                # placement measures per panel and cohort
                wh = np.asarray(ext["withheld"])
                for kind in PANELS:
                    z = np.load(udir / "panels" / f"{kind}.npz")
                    Y = z[s]
                    near = z["nearest_cosine"]
                    coh = np.asarray(m["cohorts"][name][kind])
                    fp = f["placement"][kind]
                    present = [c for c in COHORTS if (coh == c).any()]
                    ck.same("facts.placement.cohort_keys", f"{scs}/{kind}", sorted(k for k in fp if k in COHORTS), sorted(present))
                    groups = {c: np.flatnonzero(coh == c) for c in present}
                    for c, ii in groups.items():
                        compare_measures(ck, f"{scs}/{kind}/{c}", fp[c], Y, modes, near, ii, wh, f["cuts_band"])
                    if kind == "shuffled":
                        bs = fp["unseen_sequence"].get("by_source_document")
                        set_ctx = {r[0] for r in rows}
                        pr = m["panels"][kind]
                        want_keys = sorted({r[0] for r in pr})
                        ck.same("facts.placement.by_source_document.keys", f"{scs}/{kind}", sorted(bs), want_keys)
                        for sk in want_keys:
                            src = m["contexts"][sk]["source_context"]
                            ck.same("facts.placement.by_source_document.source", f"{scs}/{sk}",
                                    [bs[sk]["source_context"], bs[sk]["source_document_in_set"]], [src, src in set_ctx])
                            ii = np.array([i for i, r in enumerate(pr) if r[0] == sk])
                            compare_measures(ck, f"{scs}/{kind}/by_source/{sk}", bs[sk]["measures"], Y, modes, near, ii, wh, f["cuts_band"])
            del own


def compare_measures(ck, scope, rec, Y, modes, near, ii, withheld, cuts):
    ck.same("facts.placement.count", scope, rec["count"], len(ii))
    ck.close("facts.placement.nearest_cosine_median", scope, rec["nearest_cosine_median"], np.median(near[:, ii], axis=1))
    for k in (8, 64):
        r = rec[str(k)]
        ck.same("facts.placement.cuts_band", f"{scope}/k={k}", r["cuts_band"], cuts[str(k)])
        if np.any(withheld[:k]):
            ck.same("facts.placement.unavailable_when_withheld", f"{scope}/k={k}", r.get("unavailable"), "withheld_mode")
            continue
        pm = placement_measures(Y, modes, ii, k)
        for key in ("radius_ratio", "centre_offset", "own_mean_square"):
            ck.close(f"facts.placement.{key}", f"{scope}/k={k}", r[key], pm[key])
        for key in ("pooled_radius_ratio", "pooled_centre_offset"):
            ck.close(f"facts.placement.{key}", f"{scope}/k={k}", r[key], pm[key])


# ------------------------------------------------------------------ overlaps (section 6), own implementation
def frames(X, k):
    """Per-layer and stacked orthonormal frames of the first k columns, columns centred over rows (per layer)."""
    X = np.asarray(X[:, :, :k], dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    out = []
    for M in list(X) + [X.reshape(-1, k)]:
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        rank = int((sv > RANK_REL * sv[0]).sum()) if sv.size and sv[0] > 0 else 0
        out.append((U[:, :k] if rank == k else None, rank))
    return out


def ov_from(fa, fb, k, perm=None):
    res = []
    for (ua, ra), (ub, rb) in zip(fa, fb):
        if ra == rb == k:
            if perm is not None:
                ub = ub[perm] if len(ub) == len(perm) else ub.reshape(-1, len(perm), k)[:, perm].reshape(-1, k)
            res.append(dict(overlap=float(((ua.T @ ub) ** 2).sum() / k), rank_a=ra, rank_b=rb))
        else:
            res.append(dict(overlap=None, rank_a=ra, rank_b=rb))
    return res


def compare_scores(ck, scope, rec_scores, A, B, cuts_a, cuts_b, wh_a=None, wh_b=None):
    """rec_scores: {k: {layers: [...], stacked: {...}} or unavailable}; recompute from A, B (L, rows, >=k)."""
    for k in KS:
        r = rec_scores[str(k)]
        cut = bool(cuts_a[str(k)] or cuts_b[str(k)])
        if not (np.all(np.isfinite(A[:, :, :k])) and np.all(np.isfinite(B[:, :, :k]))):
            ck.same("overlap.unavailable_when_withheld", f"{scope}/k={k}", r.get("unavailable"), "withheld_mode")
            continue
        mine = ov_from(frames(A, k), frames(B, k), k)
        theirs = list(r["layers"]) + [r["stacked"]]
        ck.same("overlap.ranks (rank requirement)", f"{scope}/k={k}",
                [[x["rank_a"], x["rank_b"], x["overlap"] is None] for x in theirs],
                [[x["rank_a"], x["rank_b"], x["overlap"] is None] for x in mine])
        ck.close("overlap.values", f"{scope}/k={k}", [x["overlap"] for x in theirs], [x["overlap"] for x in mine])
        ck.same("overlap.cuts_band_flag", f"{scope}/k={k}", sorted({bool(x.get("cuts_band")) for x in theirs}), [cut])


def stage_aggregates(rel, ck, sample_every=1, gaussian_sets=SETS):
    m = read_json(rel / "manifest.json")

    def U(model, name):
        return rel / "units" / model / name

    facts = {(mo, n, s): read_json(U(mo, n) / s / "facts.json") for mo in MODELS for n in SETS for s in STRENGTHS}
    cuts = lambda mo, n, s: facts[(mo, n, s)]["cuts_band"]
    modes = lambda mo, n, s: np.load(U(mo, n) / s / "modes.npy")
    panel = lambda mo, n, kind, s: np.load(U(mo, n) / "panels" / f"{kind}.npz")[s]
    outside = np.array([i for i, r in enumerate(m["panels"]["natural"]) if r[0] != 0])
    # ---- cross.json
    cross = read_json(rel / "cross.json")
    ck.same("cross.skipped", "cross", cross["skipped"], [])
    recs = cross["records"]
    want_keys = set()
    for mo in MODELS:
        for a, b in zip(LADDER + ("d32",), LADDER[1:] + ("rand",)):
            for s in STRENGTHS:
                want_keys.add(("ladder", mo, a, b, s, "natural_outside_doc00", None))
    for n in SETS:
        for s in STRENGTHS:
            want_keys.add(("models", None, n, None, s, "construction", "in_set"))
            for kind in PANELS:
                for c in COHORTS:
                    if c in m["cohorts"][n][kind]:
                        want_keys.add(("models", None, n, None, s, kind, c))
            if n in ("d32", "rand"):
                want_keys.add(("models_natural_comparison", None, n, None, s, "natural_outside_doc00", None))
    got_keys = []
    for r in recs:
        if r["kind"] == "ladder":
            got_keys.append(("ladder", r["model"], r["set_a"], r["set_b"], r["strength"], r["panel"], None))
        else:
            got_keys.append((r["kind"], None, r["set"], None, r["strength"], r["panel"], r.get("cohort")))
    ck.add("cross.records_complete_and_unique", "cross", set(got_keys) == want_keys and len(got_keys) == len(set(got_keys)),
           missing=[list(k) for k in sorted(want_keys - set(got_keys), key=str)][:10],
           extra=[list(k) for k in sorted(set(got_keys) - want_keys, key=str)][:10], records=len(recs))
    for i, r in enumerate(recs):
        if i % sample_every:
            continue
        s = r["strength"]
        if r["kind"] == "ladder":
            a, b = r["set_a"], r["set_b"]
            sc = f"cross/ladder/{r['model']}/{a}->{b}/{s}"
            ck.same("cross.count", sc, r["count"], len(outside))
            A = panel(r["model"], a, "natural", s)[:, outside]
            B = panel(r["model"], b, "natural", s)[:, outside]
            compare_scores(ck, sc, r["scores"], A, B, cuts(r["model"], a, s), cuts(r["model"], b, s))
            # sizes: the same rows' cohort measures in each set
            for side, nm in (("sizes_a", a), ("sizes_b", b)):
                f = facts[(r["model"], nm, s)]
                z = np.load(U(r["model"], nm) / "panels" / "natural.npz")
                compare_measures(ck, f"{sc}/{side}", r[side], z[s], modes(r["model"], nm, s), z["nearest_cosine"],
                                 outside, np.asarray(f["extension"]["withheld"]), f["cuts_band"])
        else:
            n = r["set"]
            sc = f"cross/{r['kind']}/{n}/{s}/{r['panel']}/{r.get('cohort')}"
            if r["panel"] == "construction":
                A, B = modes("pythia", n, s), modes("olmo", n, s)
                ck.same("cross.count", sc, r["count"], len(m["sets"][n]))
            elif r["kind"] == "models_natural_comparison":
                A, B = panel("pythia", n, "natural", s)[:, outside], panel("olmo", n, "natural", s)[:, outside]
                ck.same("cross.count", sc, r["count"], len(outside))
            else:
                ii = np.flatnonzero(np.asarray(m["cohorts"][n][r["panel"]]) == r["cohort"])
                A, B = panel("pythia", n, r["panel"], s)[:, ii], panel("olmo", n, r["panel"], s)[:, ii]
                ck.same("cross.count", sc, r["count"], len(ii))
            compare_scores(ck, sc, r["scores"], A, B, cuts("pythia", n, s), cuts("olmo", n, s))
    log("aggregates: cross done")
    # ---- permutations.json
    perm = read_json(rel / "permutations.json")
    ck.same("permutations.seed_and_draws", "permutations", [perm["seed"], perm["n_draws"]], [20261129, 20])
    ck.same("permutations.skipped", "permutations", perm["skipped"], [])
    exact_draws, n_rec = 0, 0
    for i, r in enumerate(perm["records"]):
        n, s, kind, c = r["set"], r["strength"], r["panel"], r["cohort"]
        sc = f"permutations/{n}/{s}/{kind}/{c}"
        if kind == "construction":
            rr, ii, coh = m["sets"][n], np.arange(len(m["sets"][n])), ["in_set"] * len(m["sets"][n])
            A, B = modes("pythia", n, s), modes("olmo", n, s)
        elif c == "outside_doc00":
            rr, ii = m["panels"]["natural"], outside
            coh = [m["cohorts"][n]["natural"][j] for j in ii]
            A, B = panel("pythia", n, "natural", s)[:, ii], panel("olmo", n, "natural", s)[:, ii]
        else:
            rr = m["panels"][kind]
            ii = np.flatnonzero(np.asarray(m["cohorts"][n][kind]) == c)
            coh = [c] * len(ii)
            A, B = panel("pythia", n, kind, s)[:, ii], panel("olmo", n, kind, s)[:, ii]
        tid = [rr[j][2] for j in ii]
        # groups of rows sharing (token id, cohort), in order of first occurrence
        groups = defaultdict(list)
        for q, key in enumerate(zip(tid, coh)):
            groups[key].append(q)
        groups = [np.asarray(g) for g in groups.values() if len(g) > 1]
        movable = sum(len(g) for g in groups) / len(ii)
        ck.same("permutations.count", sc, r["count"], len(ii))
        ck.close("permutations.movable_share", sc, r["movable_share"], movable)
        ck.same("permutations.uninformative_flag (movable < 10%)", sc, r["within_id_uninformative"], movable < 0.1)
        if i % sample_every:
            continue
        # own reading of the draws: one Generator per row population, 20 within-group draws then 20 free draws
        g = np.random.default_rng(20261129)
        draws = {"within_id_and_cohort": [], "free": []}
        for _ in range(20):
            p = np.arange(len(ii))
            for grp in groups:
                p[grp] = g.permutation(grp)
            draws["within_id_and_cohort"].append(p)
        for _ in range(20):
            draws["free"].append(g.permutation(len(ii)))
        for k in KS:
            rec = r["scores"][str(k)]
            cut = bool(cuts("pythia", n, s)[str(k)] or cuts("olmo", n, s)[str(k)])
            if "unavailable" in rec:
                ck.add("permutations.unavailable_only_when_withheld", f"{sc}/k={k}",
                       not (np.all(np.isfinite(A[:, :, :k])) and np.all(np.isfinite(B[:, :, :k]))))
                continue
            fa, fb = frames(A, k), frames(B, k)
            mine = ov_from(fa, fb, k)
            theirs = list(rec["matched"]["layers"]) + [rec["matched"]["stacked"]]
            ck.close("permutations.matched", f"{sc}/k={k}", [x["overlap"] for x in theirs], [x["overlap"] for x in mine])
            ck.same("permutations.matched_cuts_band", f"{sc}/k={k}", sorted({bool(x.get("cuts_band")) for x in theirs}), [cut])
            for name in ("within_id_and_cohort", "free"):
                vals = np.array([[np.nan if x["overlap"] is None else x["overlap"] for x in ov_from(fa, fb, k, p)] for p in draws[name]])
                rd = np.asarray(_floats(rec[name]["draws"]), dtype=float)
                n_rec += 1
                same = rd.shape == vals.shape and np.allclose(rd, vals, rtol=1e-6, atol=1e-9, equal_nan=True)
                exact_draws += same
                # summary statistics of the recorded draws
                with np.errstate(all="ignore"):
                    ck.close(f"permutations.{name}.mean_of_recorded_draws", f"{sc}/k={k}", rec[name]["mean"], np.mean(rd, axis=0))
                    ck.close(f"permutations.{name}.min_max_of_recorded_draws", f"{sc}/k={k}",
                             [rec[name]["minimum"], rec[name]["maximum"]], [np.min(rd, axis=0), np.max(rd, axis=0)])
                if not same:
                    # not the same draws: then the recorded mean must at least lie inside own 20-draw range +- spread
                    with np.errstate(all="ignore"):
                        lo, hi = np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)
                        span = np.maximum(hi - lo, 1e-12)
                        inside = np.all((np.asarray(_floats(rec[name]["mean"]), float) >= lo - span) &
                                        (np.asarray(_floats(rec[name]["mean"]), float) <= hi + span) | np.isnan(lo))
                    ck.add(f"permutations.{name}.recorded_mean_consistent_with_own_draws", f"{sc}/k={k}", bool(inside), required=False)
    ck.add("permutations.draws_reproduced_exactly_from_own_reading", "permutations", exact_draws == n_rec, required=False,
           reproduced=exact_draws, compared=n_rec,
           note="the spec gives the seed and the grouping, not the draw procedure; inexact reproduction is not a defect")
    log("aggregates: permutations done")
    # ---- gaussian.json
    gs = read_json(rel / "gaussian.json")
    ck.same("gaussian.skipped", "gaussian", gs["skipped"], [])
    ck.same("gaussian.records_sets", "gaussian", sorted(r["set"] for r in gs["records"]), sorted(SETS))
    for r in gs["records"]:
        n = r["set"]
        sc = f"gaussian/{n}"
        A = np.load(U("pythia", n) / "site_modes.npy")
        B = np.load(U("olmo", n) / "site_modes.npy")
        ck.same("gaussian.count", sc, r["count"], len(m["sets"][n]))
        for k in KS:
            mine = ov_from(frames(A, k), frames(B, k), k)[:-1]
            ck.close("gaussian.real_per_layer_overlap", f"{sc}/k={k}", [x["overlap"] for x in r["scores"][str(k)]["real"]],
                     [x["overlap"] for x in mine])
        if n not in gaussian_sets:
            continue
        rows = m["sets"][n]
        uniq, inv = np.unique([x[2] for x in rows], return_inverse=True)
        gm = []
        for member in range(2):
            gx = np.stack([np.random.default_rng([20261201, member, int(t)]).standard_normal(WIDTH) for t in uniq])[inv]
            gx /= np.linalg.norm(gx, axis=1, keepdims=True)
            P, *_ = build_site(np.ascontiguousarray(gx))
            w, V = sla.eigh(P, subset_by_index=[len(P) - 65, len(P) - 1], check_finite=False)
            gm.append(V[:, ::-1][:, 1:65][None])
            del P, V
        for k in KS:
            mine = ov_from(frames(gm[0], k), frames(gm[1], k), k)[0]
            ck.close("gaussian.reference_overlap (assumed seeding: default_rng([20261201, member, id]))", f"{sc}/k={k}",
                     r["scores"][str(k)]["gaussian"]["overlap"], mine["overlap"], tol_abs=1e-6, tol_rel=1e-4, required=False,
                     note="eigenvector sets of near-tied P eigenvalues may rotate between solvers; looser tolerance")
        log(f"aggregates: gaussian {n} done")
    # ---- refit_agreement.json
    ra = read_json(rel / "refit_agreement.json")
    ck.same("refit.skipped", "refit", ra["skipped"], [])
    natural_08 = [r for r in m["panels"]["natural"] if r[0] in range(8)]
    for mo in MODELS:
        rdir = rel / "refits" / mo / "d08"
        rows = read_json(rdir / "rows.json")
        ck.same("refit.rows (d08 set then natural panel of documents 0..7)", mo, rows, m["sets"]["d08"] + natural_08)
        ck.same("refit.count_held", mo, len(natural_08), 256)
        done = read_json(rdir / "done.json")
        bad = [p for p, h in done["files"].items() if sha256_file(rdir / p) != h]
        ck.add("refit.files_match_done_json", mo, not bad, failed=bad[:5])
        b = np.load(rdir / "tstar" / "refit_modes.npy")
        rf = read_json(rdir / "tstar" / "facts.json")
        a = modes(mo, "d08", "tstar").astype(np.float64)
        held = panel(mo, "d08", "natural", "tstar").astype(np.float64)
        hidx = [i for i, r in enumerate(m["panels"]["natural"]) if r[0] in range(8)]
        held = held[:, hidx]
        nb = len(m["sets"]["d08"])
        ck.same("refit.shape", mo, list(b.shape), [LAYERS, nb + 256, KEEP])
        ck.gate("refit.max_residual (release)", mo, rf["max_residual"], TOL_RESID)
        for k in KS:
            A = a[:, :, :k].reshape(-1, k)
            Bs = b[:, :nb, :k].reshape(-1, k)
            Rm, _ = procrustes(A, Bs)
            shared = 1 - ((A @ Rm - Bs) ** 2).sum() / (Bs ** 2).sum()
            H = held[:, :, :k].reshape(-1, k)
            Bh = b[:, nb:, :k].reshape(-1, k)
            placed = 1 - ((H @ Rm - Bh) ** 2).sum() / (Bh ** 2).sum()
            rec = ra["records"][mo][str(k)]
            ck.close("refit.agreement", f"{mo}/k={k}", [rec["shared"], rec["placed"]], [shared, placed])
            ck.same("refit.cuts_band", f"{mo}/k={k}", rec["cuts_band"],
                    bool(facts[(mo, "d08", "tstar")]["cuts_band"][str(k)] or rf["cuts_band"][str(k)]))
    log("aggregates: refit done")


# ------------------------------------------------------------------ stage: the site pack
def encode(x, scale):
    """The addendum's rule: q = sign(x) round-half-even(127 sqrt(|x| / scale)), clipped; scale 0 -> 0; NaN -> -128."""
    x = np.asarray(x, dtype=np.float32).astype(np.float64)
    s = np.broadcast_to(np.asarray(scale, dtype=np.float64)[:, None, :], x.shape)
    fin = np.isfinite(x)
    q = np.zeros(x.shape)
    live = fin & (s > 0)
    q[live] = np.sign(x[live]) * np.rint(127.0 * np.sqrt(np.abs(x[live]) / s[live]))
    q = np.clip(q, -127, 127)
    q[~fin] = -128
    return q.astype(np.int8)


def stage_pack(rel, pack, ck):
    m = read_json(rel / "manifest.json")
    probe = read_json(PROBE)
    inputs = read_json(SYNTH / "inputs.json")
    meta = read_json(pack / "meta.json")
    sc = "pack"
    # inventory: every file on disk is listed with its size and hash, and nothing else is listed
    on_disk = {p.relative_to(pack).as_posix(): p for p in sorted(pack.rglob("*")) if p.is_file()}
    listed = meta.get("files", {})
    extra = sorted(set(on_disk) - set(listed) - {"meta.json"})
    missing = sorted(set(listed) - set(on_disk))
    wrong = sorted(k for k in listed if k in on_disk and (sha256_file(on_disk[k]) != listed[k]["sha256"]
                                                         or on_disk[k].stat().st_size != listed[k]["bytes"]))
    ck.add("pack.inventory_matches_disk", sc, not (extra or missing or wrong), extra=extra[:5], missing=missing[:5], wrong=wrong[:5])
    total = sum(p.stat().st_size for p in on_disk.values())
    ck.gate("pack.total_bytes_under_150MB", sc, total, 150_000_000, bytes=total)
    ck.same("pack.complete", sc, meta.get("complete"), True)
    for key in ("missing_units", "incomplete_units", "stale_units", "contexts_without_ids"):
        ck.same(f"pack.{key}", sc, meta.get(key), [])
    ck.same("pack.aggregates_present", sc, meta.get("aggregates"), {k: True for k in ("cross", "permutations", "gaussian", "refit_agreement")})
    ck.same("pack.manifest_sha256", sc, meta.get("manifest_sha256"), sha256_file(rel / "manifest.json"))
    ck.same("pack.spec_sha256_is_SPEC_DEPTH.md", sc, meta.get("spec_sha256"), sha256_file(SPEC))
    # rows.json and texts.json against the manifest, probe and inputs
    rows = read_json(pack / "rows.json")
    cidx = {c: i for i, c in enumerate(rows["contexts"])}
    ck.same("pack.rows.contexts", sc, rows["contexts"], list(m["contexts"]))
    for g in ("sets", "panels"):
        for k, v in m[g].items():
            ck.same(f"pack.rows.{g}", k, rows[g][k], [[cidx[ckey(r[0])], r[1]] for r in v])
    for s_, kinds in m["cohorts"].items():
        for kind, v in kinds.items():
            ck.same("pack.rows.cohorts", f"{s_}.{kind}", [rows["cohort_labels"][x] for x in rows["cohorts"][s_][kind]], v)
    texts = read_json(pack / "texts.json")
    vocab = {}
    for d in probe["contexts"][:32]:
        for t, s_ in zip(d["token_ids"], d["token_strings"]):
            vocab.setdefault(str(t), s_)
    ck.same("pack.texts.vocab_strings_from_probe", sc, texts["vocab"], dict(sorted(vocab.items(), key=lambda x: int(x[0]))))
    for i, d in enumerate(probe["contexts"][:32]):
        ck.same("pack.texts.ids", ckey(i), texts["contexts"][ckey(i)]["ids"], d["token_ids"])
    for k, v in inputs["contexts"].items():
        ck.same("pack.texts.ids", k, texts["contexts"][k]["ids"], v["ids"])
        if k[0] == "s":
            ck.same("pack.texts.shuffle_map", k, [texts["contexts"][k]["source_context"], texts["contexts"][k]["perm"]],
                    [v["source_context"], v["perm"]])
    # hover text: every manifest row's token string, through rows.json + texts.json, is the probe's string at that position
    bad = 0
    tstr = {}
    for i, d in enumerate(probe["contexts"][:32]):
        tstr[ckey(i)] = d["token_strings"]
    for g in ("sets", "panels"):
        for k, v in rows[g].items():
            for ci, pos in v:
                c = rows["contexts"][ci]
                sid = texts["contexts"][c]["ids"][pos - 1]
                if c in tstr and texts["vocab"][str(sid)] != tstr[c][pos - 1]:
                    bad += 1
    ck.add("pack.hover_token_string_equals_probe_string", sc, bad == 0, failed=bad)
    # per-unit files: re-encode the float32 release with the stated rule
    sat_total = 0
    fact_vals = read_json(pack / "facts.json")
    for model in MODELS:
        for name in SETS:
            udir = rel / "units" / model / name
            unit = read_json(udir / "unit.json")
            ck.close("pack.facts.tstar", f"{model}/{name}", fact_vals[f"tstar.{model}.{name}"]["value"], unit["tstar"], tol_abs=0, tol_rel=0)
            for s in STRENGTHS:
                scs = f"{model}/{name}/{s}"
                rec = read_json(pack / "u" / model / name / f"{s}.json")
                f = read_json(udir / s / "facts.json")
                files = rec["files"]
                modes = np.load(udir / s / "modes.npy")
                srcs = {"set": modes}
                for kind in PANELS:
                    srcs[f"panel-{kind}"] = np.load(udir / "panels" / f"{kind}.npz")[s]
                if name in ("d32", "rand"):
                    for k in m["paths"]:
                        srcs[f"path-{k}"] = np.load(udir / "paths" / f"{k}.npz")[s]
                ck.same("pack.unit_file_tags", scs, sorted(files), sorted(srcs))
                set_scale = None
                for tag in ["set"] + sorted(t for t in srcs if t != "set"):
                    x = np.asarray(srcs[tag][:, :, :16], dtype=np.float32)
                    mag = np.where(np.isfinite(x), np.abs(x), 0).astype(np.float64)
                    scale = mag.max(axis=1)
                    if tag == "set":
                        set_scale = scale
                    else:
                        scale = np.maximum(scale, set_scale)
                    fr = files[tag]
                    ck.add("pack.scales_exact", f"{scs}/{tag}", np.array_equal(np.asarray(fr["scales"], dtype=np.float64), scale))
                    q = encode(x, scale)
                    data = np.fromfile(pack / fr["path"], dtype=np.int8)
                    ok = data.size == q.size and np.array_equal(data, q.ravel())
                    ck.add("pack.bytes_reencode_exactly", f"{scs}/{tag}", ok, shape=fr["shape"], mismatched=int((data != q.ravel()).sum()) if data.size == q.size else "size")
                    ck.same("pack.shape_and_rows", f"{scs}/{tag}", [fr["shape"], fr["rows"]], [list(q.shape), x.shape[1]])
                    sat_total += int((np.abs(q.astype(np.int16)) == 127).sum())
                # per-mode facts quoted by the tab equal the release's facts (themselves recomputed by stage facts)
                ck.close("pack.mode.eigenvalue", scs, [md["eigenvalue"] for md in rec["modes"]], f["eigenvalues"][:16], tol_abs=0, tol_rel=0)
                ck.same("pack.mode.withheld", scs, [md["withheld"] for md in rec["modes"]], f["extension"]["withheld"][:16])
                bands, _ = bands_of(f["guard_eigenvalues"])
                want_band = [next(([x[0], x[1]] for x in bands if x[0] <= k <= x[1]), None) for k in range(1, 17)]
                ck.same("pack.mode.band (first, last)", scs, [None if md["band"] is None else md["band"][:2] for md in rec["modes"]],
                        want_band)
                for key in ("depth_weight", "weak_slice", "condition", "document_share"):
                    want = [None if f["fields"].get(key) is None and key == "document_share" else
                            (np.asarray(f["fields"][key])[:, j].tolist() if key in ("depth_weight", "weak_slice") else
                             (f["extension"][key][j] if key == "condition" else f["fields"][key][j])) for j in range(16)]
                    ck.same(f"pack.mode.{key}", scs, [md[key] for md in rec["modes"]], want)
                ck.same("pack.mode.extremes", scs, [md["extremes"] for md in rec["modes"]],
                        [{l: {"low": f["fields"]["extremes"][l]["low"][j], "high": f["fields"]["extremes"][l]["high"][j]}
                          for l in ("0", "8", "16")} for j in range(16)])
                ck.same("pack.cuts_band", scs, rec["cuts_band"], f["cuts_band"])
                for key in ("token_support", "adjacent_cos_mean", "endpoint_cos"):
                    ck.close(f"pack.mode.{key}", scs, [md[key] for md in rec["modes"]], f["fields"][key][:16], tol_abs=0, tol_rel=0)
    ck.same("pack.saturated_entries", sc, meta.get("saturated_entries"), sat_total)
    check_pack_facts(ck, rel, m, fact_vals, meta, total)
    check_pack_aggregates(ck, rel, read_json(pack / "aggregates.json"))
    files = {f"{pack.name}/{k}": sha256_file(p) for k, p in on_disk.items()}
    return dict(algorithm="sha256", files=files, count=len(files),
                digest=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest())


def check_pack_facts(ck, rel, m, F, meta, total_bytes):
    """Every fact the tab quotes, from the manifest / release / spec; unknown keys are failures (not verified)."""
    probe = read_json(PROBE)
    nat = [k for k, v in m["contexts"].items() if v["kind"] == "natural"]
    rnd = [k for k, v in m["contexts"].items() if v["kind"] == "random"]
    shf = [k for k, v in m["contexts"].items() if v["kind"] == "shuffled"]
    used = sorted({ckey(r[0]) for r in m["sets"]["rand"]})
    want = dict(layers=LAYERS, last_layer=LAYERS - 1, blocks=LAYERS - 1, modes_kept=KEEP, modes_shown=16, context=8,
                quant_steps=127, knn=KNN, tie_pct=TIE * 100, condition=COND_MAX, texts=32, random_sequences=5,
                shuffled_sequences=4, text_tokens=2047, first_random="r0", last_random="r4", first_shuffled="s0",
                last_shuffled="s3", rand_contexts=used, unseen_random=[c for c in rnd if c not in used],
                shuffled_sources=list(SOURCES), paths=list(m["paths"]), path_count=len(m["paths"]),
                units_expected=22, units_packed=22, cap_megabytes=150.0,
                **{f"panel.{k}": len(v) for k, v in m["panels"].items()}, **{"panel.natural_per_text": 32},
                **{f"aggregate.{k}": True for k in ("cross", "permutations", "gaussian", "refit_agreement")},
                **{"model.pythia": "Pythia-1b", "model.olmo": "OLMo-1B"})
    for name in SETS:
        want[f"set.{name}.n"] = len(m["sets"][name])
        want[f"set.{name}.texts"] = len(SET_TABLE[name][0])
    for mo in MODELS:
        for name in SETS:
            want[f"tstar.{mo}.{name}"] = read_json(rel / "units" / mo / name / "unit.json")["tstar"]
    ck.same("pack.facts.texts_kinds", "pack", [len(nat), len(rnd), len(shf)], [32, 5, 4])
    verified, unknown = 0, []
    for k, rec in F.items():
        if k in want:
            ok = jsonable(rec["value"]) == jsonable(want[k]) or (isinstance(want[k], float) and isinstance(rec["value"], (int, float))
                                                                  and abs(rec["value"] - want[k]) <= 1e-12 * max(1, abs(want[k])))
            ck.add("pack.facts.value", k, ok, got=_short(rec["value"]), want=_short(want[k]), text=rec.get("text"))
            verified += 1
        elif k == "axis_pct":
            ck.add("pack.facts.value (display setting, not a measurement)", k, True, required=False, value=rec["value"])
        elif k == "text_colours":
            ck.add("pack.facts.value (display setting, not a measurement)", k, True, required=False, value=rec["value"])
        elif k == "site_megabytes":
            # bytes of the data files written (meta.bytes), within the 0.1 MB the text shows
            ck.add("pack.facts.site_megabytes", k, abs(rec["value"] - meta["bytes"] / 1e6) < 1e-9 and meta["bytes"] <= total_bytes,
                   value=rec["value"], meta_bytes=meta["bytes"], disk_bytes=total_bytes)
        else:
            unknown.append(k)
    ck.add("pack.facts.every_fact_verified", "pack", not unknown, unverified=unknown, verified=verified,
           note="facts this audit has no rule for (e.g. added after this code was written) are listed and fail")


def check_pack_aggregates(ck, rel, agg):
    """The pack's compact aggregates equal the release aggregates (which stage aggregates recomputed)."""
    cross = read_json(rel / "cross.json")["records"]
    ck.same("pack.aggregates.cross.count", "pack", len(agg["cross"]["records"]), len(cross))
    bad = 0
    for a, r in zip(agg["cross"]["records"], cross):
        for k in KS:
            s, sr = a["scores"][str(k)], r["scores"][str(k)]
            if "unavailable" in sr:
                bad += s.get("unavailable") != sr["unavailable"]
                continue
            same = (s["layers"] == [x["overlap"] for x in sr["layers"]] and s["stacked"] == sr["stacked"]["overlap"]
                    and s["ranks"] == [[x["rank_a"], x["rank_b"]] for x in (*sr["layers"], sr["stacked"])]
                    and s["cuts_band"] == bool(sr["stacked"].get("cuts_band")))
            bad += not same
        for key in ("kind", "model", "set", "set_a", "set_b", "strength", "panel", "cohort", "count"):
            bad += a.get(key) != r.get(key)
    ck.add("pack.aggregates.cross_equals_release", "pack", bad == 0, mismatches=bad)
    perm = read_json(rel / "permutations.json")["records"]
    bad = 0
    for a, r in zip(agg["permutations"]["records"], perm):
        bad += [a[k] for k in ("set", "panel", "cohort", "strength", "count", "movable_share", "within_id_uninformative")] != \
               [r[k] for k in ("set", "panel", "cohort", "strength", "count", "movable_share", "within_id_uninformative")]
        for k in KS:
            s, sr = a["scores"][str(k)], r["scores"][str(k)]
            if "unavailable" in sr:
                continue
            for name in ("within_id_and_cohort", "free"):
                bad += any(jsonable(s[name][q]) != jsonable(sr[name][q]) for q in ("mean", "minimum", "maximum"))
            bad += s["matched"]["stacked"] != sr["matched"]["stacked"]["overlap"]
    ck.add("pack.aggregates.permutations_equal_release", "pack", bad == 0 and len(agg["permutations"]["records"]) == len(perm), mismatches=bad)
    gs = read_json(rel / "gaussian.json")["records"]
    bad = sum(a["scores"][str(k)]["gaussian"] != r["scores"][str(k)]["gaussian"]["overlap"]
              or a["scores"][str(k)]["real"] != [x["overlap"] for x in r["scores"][str(k)]["real"]]
              for a, r in zip(agg["gaussian"]["records"], gs) for k in KS)
    ck.add("pack.aggregates.gaussian_equals_release", "pack", bad == 0 and len(agg["gaussian"]["records"]) == len(gs), mismatches=bad)
    ck.same("pack.aggregates.refit_equals_release", "pack", agg["refit_agreement"], read_json(rel / "refit_agreement.json"))


# ------------------------------------------------------------------ stage: entry D1 and the tab's quoted facts
def _parse_shown(text):
    """Numbers in a fact's displayed text (thousands commas, % and the minus sign handled)."""
    import re
    t = text.replace("−", "-").replace(",", "")
    return [(float(x), len(x.split(".")[1]) if "." in x else 0) for x in re.findall(r"-?\d+(?:\.\d+)?", t)], t.strip().endswith("%")


def stage_d1(rel, pack, build, ck):
    """Recompute every number of entry D1 (the list d1_fact_list.tsv names) from the release with own code, and the
    chart data; check each value the build wrote and the text it shows; test the entry's qualitative claims."""
    m = read_json(rel / "manifest.json")
    bx = Path(build) / "assets" / "x"
    built = {**{f"x.depth.{k}": v for k, v in read_json(bx / "depth.facts.json").items()},
             **{f"x.d1-depth.{k}": v for k, v in read_json(bx / "d1-depth.facts.json").items()}}
    chart = read_json(bx / "d1-depth.json")
    listed = [ln.split("\t")[0] for ln in (Path(rel) / "audit_depth" / "d1_fact_list.tsv").read_text().splitlines()[1:] if ln.strip()]
    U = lambda mo, n, s="tstar": read_json(rel / "units" / mo / n / s / "facts.json")
    unit = lambda mo, n: read_json(rel / "units" / mo / n / "unit.json")
    cross = read_json(rel / "cross.json")["records"]
    perm = read_json(rel / "permutations.json")
    gs = {r["set"]: r for r in read_json(rel / "gaussian.json")["records"]}
    refit = read_json(rel / "refit_agreement.json")["records"]
    NM = {"pythia": "Pythia-1b", "olmo": "OLMo-1B"}
    DOC = SETS[:-1]
    med16 = lambda f, key: float(np.median(np.asarray(_floats(f["fields"][key][:16]), float)))
    E = {}                                           # expected value per fact key (own derivation)
    # --- design and counts
    E.update({"x.depth.model.pythia": "Pythia-1b", "x.depth.model.olmo": "OLMo-1B", "x.depth.layers": LAYERS,
              "x.depth.blocks": LAYERS - 1, "x.depth.texts": 32, "x.d1-depth.sets": len(SETS),
              "x.depth.set.d32.n": len(m["sets"]["d32"]), "x.depth.set.doc00.n": len(m["sets"]["doc00"]),
              "x.depth.set.d08.n": len(m["sets"]["d08"]), "x.depth.unseen_random": ["r4"],
              "x.depth.panel.natural": 1024, "x.depth.panel.random": 768, "x.depth.panel.shuffled": 1024,
              "x.depth.shuffled_sources": list(SOURCES), "x.depth.first_shuffled": "s0", "x.depth.last_shuffled": "s3",
              "x.d1-depth.k": 16, "x.depth.knn": KNN, "x.depth.last_layer": LAYERS - 1, "x.d1-depth.kp": 8,
              "x.d1-depth.k8": 8, "x.d1-depth.k64": 64, "x.depth.tie_pct": 1.0, "x.depth.modes_shown": 16,
              "x.depth.modes_kept": KEEP, "x.d1-depth.draws": 20, "x.d1-depth.weak_pct": 0.1,
              "x.d1-depth.q4.mid_layer": 8, "x.d1-depth.q1.units": 22, "x.d1-depth.q2.steps": 12,
              "x.d1-depth.ladder_tokens": 992, "x.d1-depth.q4.nat.rand.n": 992, "x.d1-depth.refit.other_n": 4})
    for s in ("doc00", "pairA", "d32"):
        E[f"x.d1-depth.set.{s.lower()}.each"] = SET_TABLE[s][1]
    for s in ("pairA", "pairB", "d04", "d08", "d16", "d32", "rand"):
        E[f"x.d1-depth.set.{s.lower()}.ctx"] = [ckey(c) for c in SET_TABLE[s][0]]
    # --- question 1
    q1 = {(mo, n): U(mo, n) for mo in MODELS for n in SETS}
    q1d = {(mo, n): U(mo, n, "tstar_div16") for mo in MODELS for n in SETS}
    adj = {k: med16(f, "adjacent_cos_mean") for k, f in q1.items()}
    end = {k: med16(f, "endpoint_cos") for k, f in q1.items()}
    absend = {k: float(np.median(np.abs(np.asarray(_floats(f["fields"]["endpoint_cos"][:16]), float)))) for k, f in q1.items()}
    sup = {k: med16(f, "token_support") for k, f in q1.items()}
    out_key = max(end, key=lambda k: abs(end[k]))
    rest = [v for k, v in end.items() if k != out_key]
    E.update({"x.d1-depth.q1.adj.min": min(adj.values()), "x.d1-depth.q1.adj.max": max(adj.values()),
              "x.d1-depth.q1.end.min": min(rest), "x.d1-depth.q1.end.max": max(rest), "x.d1-depth.q1.end.out": end[out_key],
              "x.d1-depth.q1.end.out_who": f"{NM[out_key[0]]}'s {out_key[1]}",
              "x.d1-depth.q1.absend.min": min(absend.values()), "x.d1-depth.q1.absend.max": max(absend.values()),
              "x.d1-depth.q1.support.min": min(sup.values()), "x.d1-depth.q1.support.max": max(sup.values())})
    doc = {k: med16(f, "document_share") for k, f in q1.items() if f["fields"]["document_share"] is not None}
    E["x.d1-depth.q1.doc.pairs"] = max(doc[(mo, n)] for mo in MODELS for n in ("pairA", "pairB"))
    E["x.d1-depth.q1.doc.d32.pythia"] = doc[("pythia", "d32")]
    E["x.d1-depth.q1.doc.d32.olmo"] = doc[("olmo", "d32")]
    E["x.d1-depth.q1.doc.rand.min"] = min(doc[(mo, "rand")] for mo in MODELS)
    E["x.d1-depth.q1.doc.rand.max"] = max(doc[(mo, "rand")] for mo in MODELS)
    inband = {}
    for k, f in q1.items():
        bands, _ = bands_of(f["guard_eigenvalues"])
        inband[k] = sum(any(a <= j <= b for a, b in bands) for j in range(1, 17))
    E["x.d1-depth.q1.band.modes"] = sum(inband.values())
    E["x.d1-depth.q1.band.total"] = 16 * len(q1)
    E["x.d1-depth.q1.band.units"] = sum(v > 0 for v in inband.values())
    adj16 = {k: med16(f, "adjacent_cos_mean") for k, f in q1d.items()}
    E["x.d1-depth.q1.adj16.min"], E["x.d1-depth.q1.adj16.max"] = min(adj16.values()), max(adj16.values())
    E["x.d1-depth.q1.depth.max"] = max(float(np.max(np.asarray(q1[(mo, "d32")]["fields"]["depth_weight"])[:, :16])) for mo in MODELS)
    # --- question 2 (ladder, t*, k = 16, stacked)
    lad = {(r["model"], r["set_a"], r["set_b"]): r for r in cross if r["kind"] == "ladder" and r["strength"] == "tstar"}
    steps = list(zip(LADDER, LADDER[1:])) + [("d32", "rand")]
    for a, b in steps:
        if (a, b) != ("d32", "rand"):
            E[f"x.d1-depth.q2.pythia.{a.lower()}_{b.lower()}"] = lad[("pythia", a, b)]["scores"]["16"]["stacked"]["overlap"]
    E["x.d1-depth.q2.pythia.d32_rand"] = lad[("pythia", "d32", "rand")]["scores"]["16"]["stacked"]["overlap"]
    E["x.d1-depth.q2.olmo.d32_rand"] = lad[("olmo", "d32", "rand")]["scores"]["16"]["stacked"]["overlap"]
    ol = [lad[("olmo", a, b)]["scores"]["16"]["stacked"]["overlap"] for a, b in steps[:-1]]
    E["x.d1-depth.q2.olmo.min"], E["x.d1-depth.q2.olmo.max"] = min(ol), max(ol)
    cuts16 = [(mo, a, b) for mo in MODELS for a, b in steps if lad[(mo, a, b)]["scores"]["16"]["stacked"]["cuts_band"]]
    E["x.d1-depth.q2.cut_count"] = len(cuts16)
    E["x.d1-depth.q2.cut_list"] = [f"{NM[mo]} {a} → {b}" for mo, a, b in cuts16]
    # radii of the same 992 tokens in each set: own recomputation from the panels (stage aggregates verified sizes_*)
    outside = np.array([i for i, r in enumerate(m["panels"]["natural"]) if r[0] != 0])
    rr, rr_cut = {}, []
    for mo in MODELS:
        for n in (*LADDER, "rand"):
            z = np.load(rel / "units" / mo / n / "panels" / "natural.npz")
            pm = placement_measures(z["tstar"], np.load(rel / "units" / mo / n / "tstar" / "modes.npy"), outside, 8)
            rr[(mo, n)] = pm["pooled_radius_ratio"]
            if q1[(mo, n)]["cuts_band"]["8"]:
                rr_cut.append((mo, n))
            # the cross.json sizes the generator reads (not hash-tied): must equal this recomputation
            for r in cross:
                if r["kind"] == "ladder" and r["model"] == mo and r["strength"] == "tstar":
                    for side, nm in (("sizes_a", r["set_a"]), ("sizes_b", r["set_b"])):
                        if nm == n:
                            ck.close("d1.cross_ladder_sizes_vs_own (k=8 pooled radius ratio)", f"{mo}/{n}/{side}",
                                     r[side]["8"]["pooled_radius_ratio"], pm["pooled_radius_ratio"])
    for mo in MODELS:
        vals = [rr[(mo, n)] for n in LADDER if (mo, n) not in rr_cut]
        E[f"x.d1-depth.q2.rr.{mo}.min"], E[f"x.d1-depth.q2.rr.{mo}.max"] = min(vals), max(vals)
        E[f"x.d1-depth.q2.rr.{mo}.d32"], E[f"x.d1-depth.q2.rr.{mo}.rand"] = rr[(mo, "d32")], rr[(mo, "rand")]
    E["x.d1-depth.q2.rr.cut_list"] = [f"{NM[mo]} {n}" for mo, n in rr_cut]
    # --- question 3
    P = {(r["set"], r["panel"], r["cohort"], r["strength"]): r for r in perm["records"]}
    own = {n: P[(n, "construction", "in_set", "tstar")]["scores"]["16"] for n in SETS}
    stable = [n for n in SETS if not own[n]["matched"]["stacked"]["cuts_band"]]
    E["x.d1-depth.q3.own.min"] = min(own[n]["matched"]["stacked"]["overlap"] for n in stable)
    E["x.d1-depth.q3.own.max"] = max(own[n]["matched"]["stacked"]["overlap"] for n in stable)
    E["x.d1-depth.q3.stable"] = len(stable)
    E["x.d1-depth.q3.cut_list"] = [n for n in SETS if n not in stable]
    fixed = {n: len({r[2] for r in m["sets"][n]}) / len(m["sets"][n]) for n in SETS}
    E["x.d1-depth.q3.fixed.min"] = min(fixed[n] for n in DOC)
    E["x.d1-depth.q3.fixed.max"] = max(fixed[n] for n in DOC)
    E["x.d1-depth.q3.fixed.rand"] = fixed["rand"]
    ds = [n for n in DOC if n in stable]
    same = {n: own[n]["within_id_and_cohort"]["mean"][-1] for n in SETS}
    E["x.d1-depth.q3.sameid.min"], E["x.d1-depth.q3.sameid.max"] = min(same[n] for n in ds), max(same[n] for n in ds)
    gap = [own[n]["matched"]["stacked"]["overlap"] - same[n] for n in ds]
    E["x.d1-depth.q3.gap.min"], E["x.d1-depth.q3.gap.max"] = min(gap), max(gap)
    E["x.d1-depth.q3.free.max"] = max(own[n]["free"]["mean"][-1] for n in SETS)
    E["x.d1-depth.q3.rand.sameid"] = same["rand"]
    E["x.d1-depth.q3.rand.own"] = own["rand"]["matched"]["stacked"]["overlap"]
    real = {n: [np.nan if x["overlap"] is None else x["overlap"] for x in gs[n]["scores"]["16"]["real"]] for n in SETS}
    E["x.d1-depth.q3.layer.l0.min"], E["x.d1-depth.q3.layer.l0.max"] = min(real[n][0] for n in SETS), max(real[n][0] for n in SETS)
    E["x.d1-depth.q3.layer.l1.min"], E["x.d1-depth.q3.layer.l1.max"] = min(real[n][1] for n in SETS), max(real[n][1] for n in SETS)
    above = [n for n in SETS if np.nanmax(real[n][1:]) > real[n][0]]
    E["x.d1-depth.q3.layer.above_count"] = len(above)
    E["x.d1-depth.q3.layer.d32.l13.min"], E["x.d1-depth.q3.layer.d32.l13.max"] = np.nanmin(real["d32"][1:4]), np.nanmax(real["d32"][1:4])
    E["x.d1-depth.q3.layer.d32.late.min"], E["x.d1-depth.q3.layer.d32.late.max"] = np.nanmin(real["d32"][4:]), np.nanmax(real["d32"][4:])
    gref = {n: gs[n]["scores"]["16"]["gaussian"]["overlap"] for n in SETS}
    E["x.d1-depth.q3.layer.ref.min"], E["x.d1-depth.q3.layer.ref.max"] = min(gref[n] for n in DOC), max(gref[n] for n in DOC)
    E["x.d1-depth.q3.layer.d32.ref"], E["x.d1-depth.q3.layer.rand.ref"] = gref["d32"], gref["rand"]
    E["x.d1-depth.q3.layer.rand.l0"] = real["rand"][0]
    E["x.d1-depth.q3.layer.rand.late.min"], E["x.d1-depth.q3.layer.rand.late.max"] = np.nanmin(real["rand"][4:]), np.nanmax(real["rand"][4:])
    # --- question 4 (placement measures recomputed from the panels, k = 8)
    for mo in MODELS:
        for n, kind, coh in (("d32", "random", "unseen_sequence"), ("d32", "natural", "seen_sequence_new_position"),
                             ("d32", "shuffled", "unseen_sequence"), ("rand", "natural", "unseen_sequence"),
                             ("rand", "random", "seen_sequence_new_position")):
            z = np.load(rel / "units" / mo / n / "panels" / f"{kind}.npz")
            ii = np.flatnonzero(np.asarray(m["cohorts"][n][kind]) == coh)
            pm = placement_measures(z["tstar"], np.load(rel / "units" / mo / n / "tstar" / "modes.npy"), ii, 8)
            E[f"x.d1-depth.q4.{mo}.{n}.{kind}.{coh}.rr"] = pm["pooled_radius_ratio"]
            E[f"x.d1-depth.q4.{mo}.{n}.{kind}.{coh}.co"] = pm["pooled_centre_offset"]
            if n == "rand":
                E[f"x.d1-depth.q4.near.{mo}.{n}.{kind}.{coh}"] = float(np.median(z["nearest_cosine"][8, ii]))
    for kind, coh in (("construction", "in_set"), ("natural", "seen_sequence_new_position"),
                      ("shuffled", "unseen_sequence"), ("random", "unseen_sequence")):
        E[f"x.d1-depth.q4.agree.d32.{kind}.{coh}.v"] = P[("d32", kind, coh, "tstar")]["scores"]["16"]["matched"]["stacked"]["overlap"]
    rec = P[("d32", "random", "unseen_sequence", "tstar")]
    E["x.d1-depth.q4.agree.d32.random.unseen_sequence.ref"] = rec["scores"]["16"]["within_id_and_cohort"]["mean"][-1]
    E["x.d1-depth.q4.agree.d32.random.unseen_sequence.mov"] = rec["movable_share"]
    E["x.d1-depth.q4.nat.rand.v"] = P[("rand", "natural", "outside_doc00", "tstar")]["scores"]["16"]["matched"]["stacked"]["overlap"]
    E["x.d1-depth.q4.nat.d32.v"] = P[("d32", "natural", "outside_doc00", "tstar")]["scores"]["16"]["matched"]["stacked"]["overlap"]
    E["x.d1-depth.q4.nat.rand.ref"] = P[("rand", "natural", "outside_doc00", "tstar")]["scores"]["16"]["within_id_and_cohort"]["mean"][-1]
    agree_rows = [("d32", "construction", "in_set"), ("d32", "natural", "seen_sequence_new_position"), ("d32", "shuffled", "unseen_sequence"),
                  ("d32", "random", "unseen_sequence"), ("rand", "construction", "in_set"), ("rand", "random", "seen_sequence_new_position"),
                  ("rand", "random", "unseen_sequence"), ("rand", "natural", "unseen_sequence"), ("rand", "shuffled", "unseen_sequence")]
    E["x.d1-depth.q4.weak_count"] = sum(P[(n, k, c, "tstar")]["within_id_uninformative"] for n, k, c in agree_rows)
    # --- refit
    for mo in MODELS:
        for k in KS:
            E[f"x.d1-depth.refit.{mo}.{k}.placed"] = refit[mo][str(k)]["placed"]
        E[f"x.d1-depth.refit.{mo}.16.shared"] = refit[mo]["16"]["shared"]
    oc = [(mo, k) for mo in MODELS for k in (8, 64) if refit[mo][str(k)]["cuts_band"]]
    E["x.d1-depth.refit.other_cut"] = len(oc)
    E["x.d1-depth.refit.cut_list"] = [f"{NM[mo]} at {k}" for mo, k in oc]
    # --- caveats
    E["x.d1-depth.bands.k16.n"] = len(cross)
    E["x.d1-depth.bands.k16.cut"] = sum(bool(r["scores"]["16"].get("stacked", {}).get("cuts_band", r["scores"]["16"].get("cuts_band"))) for r in cross)
    E["x.d1-depth.bands.k64.cut"] = sum(bool(r["scores"]["64"].get("stacked", {}).get("cuts_band", r["scores"]["64"].get("cuts_band"))) for r in cross)
    for s in STRENGTHS:
        recs = [r for r in cross if r["kind"] == "models" and r["strength"] == s]
        nulls = [(r["set"], sum(x["overlap"] is None for x in r["scores"]["16"]["layers"])) for r in recs]
        E[f"x.d1-depth.undef.{s}.count"] = sum(c for _, c in nulls)
        E[f"x.d1-depth.undef.{s}.total"] = 17 * len(recs)
        sets_null = sorted({n for n, c in nulls if c}, key=SETS.index)
        if s == "tstar":
            E["x.d1-depth.undef.tstar.sets"] = sets_null
        else:
            E["x.d1-depth.undef.tstar_div16.nsets"] = len(sets_null)
    for mo in MODELS:
        ts = {n: unit(mo, n)["tstar"] for n in SETS}
        E[f"x.d1-depth.tstar.{mo}.min"], E[f"x.d1-depth.tstar.{mo}.max"] = min(ts.values()), max(ts.values())
    # --- compare: every listed fact is built, and its value equals the own derivation; its text shows that value
    ck.same("d1.listed_fact_count", "d1", len(listed), 167)
    missing = [k for k in listed if k not in built]
    ck.add("d1.listed_facts_present_in_build", "d1", not missing, missing=missing)
    unchecked = []
    for k in listed:
        if k not in built:
            continue
        b = built[k]
        if k not in E:
            unchecked.append(k)
            continue
        want = E[k]
        if isinstance(want, (list, str)):
            # list-valued facts: the value as a list, or the text as own rendering ("a to d" for a contiguous run
            # of contexts, otherwise "a, b and c")
            bv = b["value"]
            items = want if isinstance(want, list) else [want]
            def render(xs):
                xs = [str(x) for x in xs]
                if len(xs) > 2 and all(x[0] in "cr" for x in xs):
                    nums = [int(x[1:]) for x in xs]
                    if nums == list(range(nums[0], nums[0] + len(nums))) and len({x[0] for x in xs}) == 1:
                        return f"{xs[0]} to {xs[-1]}"
                return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else "none"
            import re as _re
            parsed = [x.strip() for x in _re.split(r",\s*|\s+and\s+", b["text"].strip()) if x.strip()]
            ok = (jsonable(bv) == jsonable(want) or b["text"].strip() == render(items) or parsed == [str(x) for x in items]
                  or (not items and b["text"].strip() in ("", "none")))
            ck.add("d1.fact_value", k, ok, got=_short(bv), want=_short(want), text=b["text"], own_rendering=render(items) if items else "")
            continue
        bv = b["value"]
        ck.close("d1.fact_value", k, bv, want)
        nums, pct = _parse_shown(b["text"])
        if nums and isinstance(bv, (int, float)):
            shown, dec = nums[0]
            v = bv * 100 if pct and abs(bv) <= 1.5 and not k.endswith("tie_pct") else bv
            ok = abs(shown - v) <= 0.5 * 10 ** -dec + 1e-9 * max(1, abs(v))
            ck.add("d1.fact_text_is_rounded_value", k, ok, text=b["text"], value=bv)
    ck.add("d1.every_listed_fact_derived_independently", "d1", not unchecked, required=False, not_derived=unchecked,
           note="listed facts without an own derivation here are compared only for presence")
    # --- chart data
    rows11 = list(SETS)
    for pi, key in enumerate(("token_support", "endpoint_cos", "document_share")):
        for si, mo in enumerate(MODELS):
            ser = chart["q1"]["panels"][pi]["series"][si]
            vals, lo, hi = [], [], []
            for n in rows11:
                x = q1[(mo, n)]["fields"][key]
                if x is None:
                    vals.append(None); lo.append(None); hi.append(None); continue
                a = np.asarray(_floats(x[:16]), float)
                vals.append(float(np.median(a))); lo.append(float(np.percentile(a, 25))); hi.append(float(np.percentile(a, 75)))
            ck.close("d1.chart.q1", f"{key}/{mo}", [ser["values"], ser["lo"], ser["hi"]], [vals, lo, hi])
    for pi, mo in enumerate(MODELS):
        z = np.asarray(q1[(mo, "d32")]["fields"]["depth_weight"])[:, :16].T
        ck.close("d1.chart.q1_depth", mo, chart["q1_depth"]["panels"][pi]["z"], z)
        bands, _ = bands_of(q1[(mo, "d32")]["guard_eigenvalues"])
        marks = [("≈" in r) for r in chart["q1_depth"]["panels"][pi]["rows"]]
        ck.same("d1.chart.q1_depth.band_marks", mo, marks, [any(a <= j <= b for a, b in bands) for j in range(1, 17)])
    for si, mo in enumerate(MODELS):
        ser = chart["q2"]["series"][si]
        ck.close("d1.chart.q2", mo, ser["values"], [lad[(mo, a, b)]["scores"]["16"]["stacked"]["overlap"] for a, b in steps])
        ck.same("d1.chart.q2.hollow", mo, ser["hollow"], [bool(lad[(mo, a, b)]["scores"]["16"]["stacked"]["cuts_band"]) for a, b in steps])
    q3s = chart["q3"]["series"]
    ck.close("d1.chart.q3", "matched", q3s[0]["values"], [own[n]["matched"]["stacked"]["overlap"] for n in SETS])
    ck.same("d1.chart.q3.hollow", "matched", q3s[0]["hollow"], [bool(own[n]["matched"]["stacked"]["cuts_band"]) for n in SETS])
    ck.close("d1.chart.q3", "within_id", [q3s[1]["values"], q3s[1]["lo"], q3s[1]["hi"]],
             [[own[n]["within_id_and_cohort"][q][-1] for n in SETS] for q in ("mean", "minimum", "maximum")])
    ck.close("d1.chart.q3", "free", q3s[2]["values"], [own[n]["free"]["mean"][-1] for n in SETS])
    lay = chart["q3_layers"]["series"]
    named = {"d32 (32 texts)": "d32", "rand (random tokens)": "rand"}
    others = [s for s in lay if s["name"] not in named]
    for s in lay:
        if s["name"] in named:
            ck.close("d1.chart.q3_layers", named[s["name"]], s["y"], real[named[s["name"]]])
    other_sets = [n for n in SETS if n not in ("d32", "rand")]
    ck.close("d1.chart.q3_layers", "other sets (as a set of curves)", sorted(map(tuple, (o["y"] for o in others))),
             sorted(tuple(real[n]) for n in other_sets))
    ck.close("d1.chart.q3_layers.reflines", "d32, rand", [r["y"] for r in chart["q3_layers"]["reflines"]], [gref["d32"], gref["rand"]])
    q4rows = [("d32", "natural", "seen_sequence_new_position"), ("d32", "shuffled", "unseen_sequence"), ("d32", "random", "unseen_sequence"),
              ("rand", "random", "seen_sequence_new_position"), ("rand", "random", "unseen_sequence"), ("rand", "natural", "unseen_sequence"),
              ("rand", "shuffled", "unseen_sequence")]
    pms = {}
    for mo in MODELS:
        for n, kind, coh in q4rows:
            z = np.load(rel / "units" / mo / n / "panels" / f"{kind}.npz")
            ii = np.flatnonzero(np.asarray(m["cohorts"][n][kind]) == coh)
            pms[(mo, n, kind, coh)] = placement_measures(z["tstar"], np.load(rel / "units" / mo / n / "tstar" / "modes.npy"), ii, 8)
    for si, mo in enumerate(MODELS):
        ck.close("d1.chart.q4.radius", mo, chart["q4"]["panels"][0]["series"][si]["values"],
                 [pms[(mo, *r)]["pooled_radius_ratio"] for r in q4rows])
        ck.close("d1.chart.q4.shift", mo, chart["q4"]["panels"][1]["series"][si]["values"],
                 [pms[(mo, *r)]["pooled_centre_offset"] for r in q4rows])
    qa = chart["q4_agree"]["series"]
    ck.close("d1.chart.q4_agree", "matched", qa[0]["values"], [P[(n, k, c, "tstar")]["scores"]["16"]["matched"]["stacked"]["overlap"] for n, k, c in agree_rows])
    ck.close("d1.chart.q4_agree", "within_id", [qa[1]["values"], qa[1]["lo"], qa[1]["hi"]],
             [[P[(n, k, c, "tstar")]["scores"]["16"]["within_id_and_cohort"][q][-1] for n, k, c in agree_rows] for q in ("mean", "minimum", "maximum")])
    ck.same("d1.chart.q4_agree.hollow (uninformative)", "within_id", qa[1]["hollow"], [P[(n, k, c, "tstar")]["within_id_uninformative"] for n, k, c in agree_rows])
    ck.close("d1.chart.q4_agree", "free", qa[2]["values"], [P[(n, k, c, "tstar")]["scores"]["16"]["free"]["mean"][-1] for n, k, c in agree_rows])
    # --- qualitative claims of the prose, tested on own numbers
    ck.add("d1.claim: overlap drops from layer 0 to layer 1 in every set", "q3", all(real[n][1] < real[n][0] for n in SETS))
    ck.add("d1.claim: some later layer exceeds layer 0 in all sets but d32", "q3", sorted(set(SETS) - set(above)) == ["d32"], above=above)
    ck.add("d1.claim: d32 -> rand is among the two largest changes (two smallest overlaps) in both models", "q2",
           all(sorted(range(6), key=lambda i: [lad[(mo, a, b)]["scores"]["16"]["stacked"]["overlap"] for a, b in steps][i])[:2].count(5)
               for mo in MODELS))
    ck.add("d1.claim: no ladder step keeps the same space (all step overlaps < 0.9)", "q2",
           all(lad[(mo, a, b)]["scores"]["16"]["stacked"]["overlap"] < 0.9 for mo in MODELS for a, b in steps), required=False)
    ck.add("d1.claim: in rand every group sits at or just above its word-piece reference", "q4",
           all(0 <= P[("rand", k, c, "tstar")]["scores"]["16"]["matched"]["stacked"]["overlap"] - P[("rand", k, c, "tstar")]["scores"]["16"]["within_id_and_cohort"]["mean"][-1] < 0.05
               for n, k, c in agree_rows if n == "rand"), required=False,
           diffs=[P[("rand", k, c, "tstar")]["scores"]["16"]["matched"]["stacked"]["overlap"] - P[("rand", k, c, "tstar")]["scores"]["16"]["within_id_and_cohort"]["mean"][-1] for n, k, c in agree_rows if n == "rand"])
    ck.add("d1.claim: in d32 the models agree most on tokens of the set's own kind", "q4",
           max(range(4), key=lambda i: qa[0]["values"][i]) in (0, 1), required=False)
    lo2 = {mo: sorted(SETS, key=lambda n: unit(mo, n)["tstar"])[:2] for mo in MODELS}
    ck.add("d1.claim: the two lowest t* in each model are doc02 and pairB", "caveats",
           all(sorted(v) == ["doc02", "pairB"] for v in lo2.values()), lowest=lo2)
    ck.add("d1.claim: random tokens in d32 sit closer to the centre (radius < 1) with a larger shift than the reference", "q4",
           all(pms[(mo, "d32", "random", "unseen_sequence")]["pooled_radius_ratio"] < 1 and
               pms[(mo, "d32", "random", "unseen_sequence")]["pooled_centre_offset"] > pms[(mo, "d32", "natural", "seen_sequence_new_position")]["pooled_centre_offset"]
               for mo in MODELS), required=False)
    ck.add("d1.claim: the placed d08 tokens score about as well as the shared tokens (|placed - shared| < 0.03 at 16)", "refit",
           all(abs(refit[mo]["16"]["placed"] - refit[mo]["16"]["shared"]) < 0.03 for mo in MODELS), required=False,
           diffs={mo: refit[mo]["16"]["placed"] - refit[mo]["16"]["shared"] for mo in MODELS})
    # the expected share of tokens a within-group random permutation leaves in place is (number of groups) / tokens,
    # because a uniform permutation of g items has one fixed point on average; groups here are (id, cohort) = ids
    ck.add("d1.fixed_share_expectation (distinct ids / tokens = groups / tokens for the construction cohort)", "q3", True,
           required=False, values=fixed)
    # how the spec's 'uninformative' flag (movable < 10%) relates to the expected fixed share, for every record
    rows_info = []
    for r in perm["records"]:
        if r["strength"] != "tstar":
            continue
        n, kind, coh = r["set"], r["panel"], r["cohort"]
        if kind == "construction":
            ids, cohs = [x[2] for x in m["sets"][n]], ["in_set"] * len(m["sets"][n])
        elif coh == "outside_doc00":
            ids = [m["panels"]["natural"][i][2] for i in outside]
            cohs = [m["cohorts"][n]["natural"][i] for i in outside]
        else:
            ii = np.flatnonzero(np.asarray(m["cohorts"][n][kind]) == coh)
            ids, cohs = [m["panels"][kind][i][2] for i in ii], [coh] * len(ii)
        groups = len(set(zip(ids, cohs)))
        rows_info.append((n, kind, coh, r["movable_share"], groups / len(ids), r["within_id_uninformative"]))
    hi_fixed = [x for x in rows_info if x[4] > 0.75 and not x[5]]
    ck.add("d1.uninformative_flag_vs_expected_fixed_share (records with > 75% expected in place but not flagged)", "q3/q4",
           not hi_fixed, required=False, records=[dict(set=a, panel=b, cohort=c, movable=d, expected_fixed=e) for a, b, c, d, e, _ in hi_fixed])


# ------------------------------------------------------------------ driver
def run_stage(out, stage, fn, *a, redo=False, **kw):
    path = out / "parts" / f"{stage}.json"
    if path.is_file() and not redo:
        log(f"{stage}: reusing {path}")
        return read_json(path)
    ck = Checks()
    t0 = time.time()
    extra = None
    try:
        extra = fn(*a, ck, **kw)
        error = None
    except Exception as e:                      # a stage that cannot run has failed; record why
        import traceback
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        log(f"{stage}: ERROR {error}")
    part = dict(stage=stage, checks=ck.items, error=error, seconds=time.time() - t0,
                inventory=extra if isinstance(extra, dict) and extra.get("algorithm") == "sha256" else None)
    if error is None or ck.items:
        write_json(path, part)
    nfail = sum(1 for c in ck.items if c["required"] and not c["passed"])
    log(f"{stage}: {len(ck.items)} checks, {nfail} required failed{'; ERROR' if error else ''} ({time.time() - t0:.0f} s)")
    return part


def load_avg_ok(limit=10.0, wait=60, tries=60):
    for _ in range(tries):
        if os.getloadavg()[0] <= limit:
            return True
        log(f"load average {os.getloadavg()[0]:.1f} > {limit}; waiting")
        time.sleep(wait)
    return False


def assemble(out, pack):
    parts = {p.stem: read_json(p) for p in sorted((out / "parts").glob("*.json"))
             if not p.stem.startswith("count_") and not p.stem.endswith("_detail")}
    required = ["manifest", "inputs", "facts", "aggregates", "d1", "pack"] + [f"cell_{mo}_{n}" for mo in MODELS for n in AUDITED]
    absent = [r for r in required if r not in parts]
    errors = [k for k, v in parts.items() if v.get("error")]
    checks = [dict(c, stage=k) for k, v in parts.items() for c in v["checks"]]
    failed = [c for c in checks if c["required"] and not c["passed"]]
    inv = parts.get("pack", {}).get("inventory")
    passed = bool(checks) and not failed and not absent and not errors and inv is not None
    report = dict(passed=passed, spec="SPEC_DEPTH.md v3.1 section 8, with SPEC_DEPTH_ADDENDUM.md section 2 for the site encoding",
                  spec_sha256=sha256_file(SPEC), addendum_sha256=sha256_file(ADDENDUM) if ADDENDUM.is_file() else None,
                  auditor="independent audit (different model family), exp1/audit_depth.py", audit_code_sha256=sha256_file(Path(__file__)),
                  numpy=np.__version__, scipy=scipy.__version__, release=str(RELEASE), pack=str(pack) if pack else None,
                  generated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  absent_stages=absent, stage_errors={k: parts[k]["error"] for k in errors},
                  summary=dict(checks=len(checks), required=sum(c["required"] for c in checks), required_failed=len(failed),
                               informational_failed=sum(1 for c in checks if not c["required"] and not c["passed"])),
                  failed_checks=[dict(name=c["name"], scope=c["scope"], stage=c["stage"], worst=c["worst"], tolerance=c["tolerance"]) for c in failed],
                  checks=checks)
    if inv is not None:
        report["inventory"] = inv
    write_json(out / "report.json", report)
    log(f"report: passed={passed}; {len(checks)} checks, {len(failed)} required failed; absent stages {absent}; errors {errors}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("manifest", "inputs", "cell", "facts", "aggregates", "d1", "pack", "report", "all"))
    ap.add_argument("--release", type=Path, default=RELEASE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--pack", type=Path, default=None, help="site pack directory (e.g. /data/tm_depth_modal/site_depth)")
    ap.add_argument("--build", type=Path, default=Path("/var/tmp/tm_audit_build"), help="dev site build holding assets/x/d1-depth*.json")
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--set", dest="name", choices=AUDITED)
    ap.add_argument("--redo", action="store_true", help="recompute even if the stage's part file exists")
    ap.add_argument("--no-hash", action="store_true", help="inputs stage: skip hashing the 82 activation files")
    ap.add_argument("--synth-acts", type=Path, default=OUT / "acts_fetch",
                    help="preferred copies of synthetic-context activations (fetched from the Volume; the local mirror "
                         "/data/tm_acts/depth_acts has zero-filled holes in s0..s3); every file is hash-checked by stage inputs")
    a = ap.parse_args()
    global SYNTH_OVERRIDE
    SYNTH_OVERRIDE = a.synth_acts if a.synth_acts and a.synth_acts.is_dir() else None
    rel, out = a.release.resolve(), a.out.resolve()
    (out / "parts").mkdir(parents=True, exist_ok=True)
    try:
        os.nice(max(0, 5 - os.getpriority(os.PRIO_PROCESS, 0)))
    except OSError:
        pass
    st = a.stage
    if st in ("manifest", "all"):
        run_stage(out, "manifest", stage_manifest, rel, redo=a.redo)
    if st in ("inputs", "all"):
        run_stage(out, "inputs", stage_inputs, rel, redo=a.redo, hash_acts=not a.no_hash)
    if st in ("cell", "all"):
        cells = [(mo, n) for mo in MODELS for n in AUDITED if (a.model in (None, mo)) and (a.name in (None, n))]
        for mo, n in cells:
            if (out / "parts" / f"cell_{mo}_{n}.json").is_file() and not a.redo:
                log(f"cell {mo}/{n}: reusing")
                continue
            load_avg_ok()
            run_stage(out, f"cell_{mo}_{n}", stage_cell, rel, out, mo, n, redo=a.redo)
    if st in ("facts", "all"):
        run_stage(out, "facts", stage_facts, rel, redo=a.redo)
    if st in ("aggregates", "all"):
        load_avg_ok()
        run_stage(out, "aggregates", stage_aggregates, rel, redo=a.redo)
    if st in ("d1", "all") and a.build.is_dir():
        run_stage(out, "d1", stage_d1, rel, a.pack, a.build, redo=a.redo)
    if st in ("pack", "all") and a.pack:
        run_stage(out, "pack", stage_pack, rel, a.pack.resolve(), redo=a.redo)
    if st in ("report", "all"):
        assemble(out, a.pack.resolve() if a.pack else None)


if __name__ == "__main__":
    main()
