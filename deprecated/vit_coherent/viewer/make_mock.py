"""MOCKUP data generator: writes a synthetic dataset in the SCHEMA.md v1 layout to data/_mock/.

For viewer development only.  Nothing here is a result; every name says MOCK.  The geometry is
planted (superclass / class / nuisance latents mixed with a depth-dependent weight) and the
spectral objects are computed on that planted geometry (kNN self-tuned graph, symmetric
normalisation, eigsh), so modes are smooth on the layout and the links between objects are
self-consistent; attributions, cross-validated values, selection criteria and the like are random.

    uv run --with numpy --with scipy --with pillow python -I make_mock.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "_mock"
rng = np.random.default_rng(0)
N, C, SC, L, H, NN, M, D = 10_000, 100, 12, 12, 6, 1536, 64, 384
TS = [1, 4, 16, 64]
RS = [8, 16, 32, 64]
OFF = [0, 8, 24, 56]
PAIRS = [(a, b) for a in range(1, 13) for b in range(a + 1, 13)]
DECL = [(1, 3), (3, 5), (1, 5), (1, 6), (6, 12), (1, 12), (6, 9), (9, 12), (3, 6), (3, 9)]
TRIPLES = [(1, 3, 5), (1, 6, 12), (6, 9, 12), (3, 6, 9)]
KFULL = ["st15"]
KALL = ["st15", "vmf14"]  # kernels for which the mock writes per-layer files
KERNELS = [
    ("st7", "self-tuned", {"k": 7}), ("st15", "self-tuned", {"k": 15}), ("st30", "self-tuned", {"k": 30}),
    ("vmf14", "vMF", {"kappa": 14.2857}), ("vmf5", "vMF", {"kappa": 5}), ("vmf30", "vMF", {"kappa": 30}),
]
manifest = []


def w(rel, arr, dtype):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    a = np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<"))
    a.tofile(p)
    manifest.append({"path": rel, "dtype": np.dtype(dtype).name, "shape": list(a.shape), "bytes": p.stat().st_size})


def wj(rel, obj):
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(obj)))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    cls = np.repeat(np.arange(C), 100)
    c2s = rng.integers(0, SC, C)
    c2s[:SC] = np.arange(SC)
    sup = c2s[cls]
    # planted latents
    Zs = rng.normal(size=(SC, 24)) * 2.6
    Zc = Zs[c2s] + rng.normal(size=(C, 24)) * 1.3
    Z = Zc[cls] + rng.normal(size=(N, 24)) * 1.0
    nu_k = rng.integers(0, 6, N)
    Un = rng.normal(size=(6, 10)) * 2.2
    U = Un[nu_k] + rng.normal(size=(N, 10)) * 0.9
    wts = 1 / (1 + np.exp(-(np.arange(1, 13) - 5.5) / 1.6))
    views = []
    for l in range(L):
        x = np.concatenate([wts[l] * Z, (1 - wts[l]) * U, 0.6 * rng.normal(size=(N, 6))], 1)
        x = x - x.mean(0)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        views.append(x.astype(np.float32))

    # images
    w("images/class.u8", cls, np.uint8)
    w("images/superclass.u8", sup, np.uint8)
    w("images/source.u8", np.tile(np.r_[np.zeros(50), np.ones(50)], C), np.uint8)
    w("images/half.u8", rng.integers(0, 2, N), np.uint8)
    make_thumbs(cls, sup, nu_k)

    # aligned "UMAP" (mock: planted 2-d anchors, not UMAP)
    ang_s = np.linspace(0, 2 * np.pi, SC, endpoint=False)
    As = np.c_[np.cos(ang_s), np.sin(ang_s)] * 6
    Ac = As[c2s] + rng.normal(size=(C, 2)) * 1.4
    An = rng.normal(size=(6, 2)) * 5
    eps = rng.normal(size=(N, 2))
    um = np.stack([wts[l] * Ac[cls] + (1 - wts[l]) * An[nu_k] + eps * (0.9 - 0.4 * wts[l]) for l in range(L)])
    w("cls/umap.f32", um, np.float32)

    spectra = {}
    modes = {}
    for k in KALL:
        modes[k], spectra[k] = [], []
        for l in range(L):
            psi, lam = spectral(views[l], kind=k)
            modes[k].append(psi)
            spectra[k].append(lam)
            w(f"cls/{k}/modes/L{l+1:02d}.f16", psi.T * 100, np.float16)
            print("modes", k, l + 1, file=sys.stderr)
        lam = np.array(spectra[k])
        full = np.concatenate([np.ones((L, 1)), lam, np.sort(rng.uniform(-0.03, 1, (L, N - 65)) * lam[:, -1:] * 0.9, 1)[:, ::-1]], 1)
        w(f"cls/{k}/spectrum.f32", full, np.float32)
        w(f"cls/{k}/spectrum_alpha1.f32", np.concatenate([np.ones((L, 1)), lam * 0.97], 1), np.float32)
        w(f"cls/{k}/evals.f32", lam, np.float32)
        S = np.zeros((L, M, L, 4))
        for a in range(L):
            for b in range(L):
                G = (modes[k][a].T @ modes[k][b]) ** 2  # unit vectors -> squared cosines
                for ti, t in enumerate(TS):
                    S[a, :, b, ti] = G @ (spectra[k][b] ** (2 * t))
        w(f"cls/{k}/smooth.f32", S, np.float32)

    k = "st15"
    hards = seba_and_flows(k, modes[k], spectra[k])
    deltas(k, modes, spectra, PAIRS, full=True)
    deltas("vmf14", modes, spectra, DECL, full=False)
    neurons_heads_attr(k, views, modes[k], spectra[k], hards)

    # mp view: umap + modes for st15 only, 13 layers (layer 0 = a nuisance-only copy)
    mpu = np.concatenate([um[:1] * 0.8 + rng.normal(size=(1, N, 2)) * 0.5, um * 1.05], 0)
    w("mp/umap.f32", mpu, np.float32)
    lam_mp = []
    for l in range(13):
        psi, lam = spectral(views[max(l - 1, 0)] + 0.02 * rng.normal(size=views[0].shape).astype(np.float32), "st15")
        w(f"mp/st15/modes/L{l:02d}.f16", psi.T * 100, np.float16)
        lam_mp.append(lam)
    w("mp/st15/evals.f32", np.array(lam_mp), np.float32)

    # weights
    w("weights/wout_ls.f16", rng.normal(size=(L, NN, D)) * 0.02, np.float16)
    # attention exemplars
    idx = np.concatenate([np.sort(rng.choice(100, 15, replace=False)) + 100 * c for c in range(C)])
    w("attn/idx.i16", idx, np.int16)
    yy, xx = np.mgrid[0:16, 0:16]
    mx = np.zeros((L, 1500, H, 2), np.float32)
    for l in range(L):
        A = np.zeros((1500, H, 256), np.uint8)
        for h in range(H):
            cy, cx = rng.uniform(3, 13, (2, 1500))
            sig = 1.5 + 3 * (h / H) + 0.2 * l
            g = np.exp(-((yy[None] - cy[:, None, None]) ** 2 + (xx[None] - cx[:, None, None]) ** 2) / (2 * sig**2))
            g = g.reshape(1500, 256) + rng.uniform(0, 0.15, (1500, 256))
            A[:, h] = np.round(255 * g / g.max(1, keepdims=True))
        w(f"attn/L{l+1:02d}.u8", A, np.uint8)
        mx[l, :, :, 0] = rng.uniform(0.02, 0.3, (1500, H))
        mx[l, :, :, 1] = rng.uniform(0.0, 0.5, (1500, H))
    w("attn/max.f32", mx, np.float32)

    summaries(spectra)
    meta = {
        "tag": "_mock", "MOCKUP": True, "model": "dinov2_vits14 (MOCKUP: synthetic data)", "N": N, "N_heldout": N,
        "views": {"cls": {"layers": list(range(1, 13))}, "mp": {"layers": list(range(0, 13))}},
        "kernels": [{"id": i, "family": f, "param": p,
                     "formula": "K_ij = exp(-d_ij^2/(s_i s_j))" if f == "self-tuned" else "K_ij = exp(kappa (cos_ij - 1))"}
                    for i, f, p in KERNELS],
        "kernels_full": KFULL, "default_kernels": KFULL, "t": TS, "seba_r": RS, "seba_offsets": OFF,
        "pairs": PAIRS, "declared_pairs": DECL, "triples": TRIPLES, "n_modes": M, "n_delta": 8,
        "n_attr_modes": 16, "n_q": 32,
        "classes": [{"id": c, "wnid": f"n{c:08d}", "name": f"mock class {c}", "superclass": int(c2s[c])} for c in range(C)],
        "superclasses": [{"id": s, "name": f"mock super {s}", "wnid": f"n9{s:07d}"} for s in range(SC)],
        "mode_scale": 100.0, "noise_snr": 10.0, "seed": 0, "dataset": "MOCKUP (planted latents)", "kernel_ids": [x[0] for x in KERNELS],
        "delta_pairs": {"cls": {"st15": PAIRS, "vmf14": DECL}}, "reconstruction": {"max_abs_err": 1e-12}, "readout_vs_hub": 3e-6,
    }
    wj("meta.json", meta)
    wj("manifest.json", manifest)
    print("wrote", OUT, sum(m["bytes"] for m in manifest) / 1e6, "MB", file=sys.stderr)


def knn_graph(x, k=15):
    idx = np.zeros((N, k), np.int64)
    dist = np.zeros((N, k), np.float32)
    for s in range(0, N, 1000):
        cs = x[s : s + 1000] @ x.T
        cs[np.arange(len(cs)), np.arange(s, s + len(cs))] = -2
        j = np.argpartition(-cs, k, axis=1)[:, :k]
        cj = np.take_along_axis(cs, j, 1)
        o = np.argsort(-cj, 1)
        idx[s : s + 1000] = np.take_along_axis(j, o, 1)
        dist[s : s + 1000] = np.sqrt(np.maximum(2 - 2 * np.take_along_axis(cj, o, 1), 0))
    return idx, dist


_GRAPH = {}


def spectral(x, kind):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    key = (x.shape, float(x[0, 0]), float(x[-1, -1]))
    if key not in _GRAPH:
        _GRAPH.clear()
        _GRAPH[key] = knn_graph(x, 30)
    idx, dist = _GRAPH[key]
    if kind.startswith("st"):
        s = dist[:, 14]
        wv = np.exp(-dist**2 / (s[:, None] * s[idx]))
    else:
        wv = np.exp(14.29 * (1 - dist**2 / 2 - 1))
    Wm = csr_matrix((wv.ravel(), (np.repeat(np.arange(N), idx.shape[1]), idx.ravel())), shape=(N, N))
    Wm = (Wm + Wm.T) / 2
    d = np.asarray(Wm.sum(1)).ravel()
    Dm = csr_matrix((1 / np.sqrt(d), (np.arange(N), np.arange(N))))
    P = Dm @ Wm @ Dm
    lam, V = eigsh(P, k=65, which="LA", v0=np.ones(N))
    o = np.argsort(-lam)
    lam, V = lam[o][1:], V[:, o][:, 1:]
    V = V - V.mean(0)
    V /= np.linalg.norm(V, axis=0)
    V *= np.sign(V[np.abs(V).argmax(0), np.arange(V.shape[1])])
    return V, lam * 0.995


def seba_and_flows(k, modes, spectra):
    hards = np.zeros((L, 4, N), np.int8)
    sj = {"r": RS, "offsets": OFF, "layers": []}
    for l in range(L):
        mem = np.zeros((120, N))
        sj["layers"].append({"layer": l + 1, "r": {}})
        for ri, r in enumerate(RS):
            E = modes[l][:, : r - 1] * spectra[l][: r - 1] * np.sqrt(N)
            cen = E[rng.choice(N, r, replace=False)]
            sqd = lambda c_: (E**2).sum(1)[:, None] - 2 * E @ c_.T + (c_**2).sum(1)[None]
            for _ in range(12):
                d2 = sqd(cen)
                a = d2.argmin(1)
                for j in range(r):
                    if (a == j).any():
                        cen[j] = E[a == j].mean(0)
            d2 = np.maximum(sqd(cen), 0)
            m = np.exp(-(d2 - d2.min(1, keepdims=True)) / (np.median(d2) * 0.08)).T
            m /= m.max(1, keepdims=True)
            m[m < 0.02] = 0
            mem[OFF[ri] : OFF[ri] + r] = m
            h = m.argmax(0)
            h[m.max(0) < 0.1] = -1
            hards[l, ri] = h
            sj["layers"][-1]["r"][str(r)] = {"gap": float(spectra[l][r - 2] - spectra[l][r - 1]), "split_half_min_cos": rng.uniform(0.3, 0.95),
                            "split_half_mean_cos": rng.uniform(0.7, 0.99), "dist_from_span": rng.uniform(0.05, 0.4),
                            "neg_mass": rng.uniform(0.0, 0.15), "coverage": float((h >= 0).mean()),
                            "sizes": [int((h == j).sum()) for j in range(r)]}
        w(f"cls/{k}/seba/L{l+1:02d}.u8", np.round(mem * 255), np.uint8)
        w(f"cls/{k}/seba/L{l+1:02d}_hard.i8", hards[l], np.int8)
    wj(f"cls/{k}/seba.json", sj)
    fj = {"pairs": PAIRS, "r": RS, "H_b_given_a": {}, "H_a_given_b": {}, "MI": {}, "baseline": {}}
    for ri, r in enumerate(RS):
        cnt = np.zeros((len(PAIRS), r + 1, r + 1), np.uint16)
        hba, hab = [], []
        for p, (a, b) in enumerate(PAIRS):
            ha, hb = hards[a - 1, ri].astype(int), hards[b - 1, ri].astype(int)
            ha[ha < 0] = r
            hb[hb < 0] = r
            np.add.at(cnt[p], (ha, hb), 1)
            hba.append(cond_H(cnt[p]))
            hab.append(cond_H(cnt[p].T))
        fj["H_b_given_a"][str(r)] = hba
        fj["H_a_given_b"][str(r)] = hab
        fj["MI"][str(r)] = list(rng.uniform(0.5, 3, len(PAIRS)))
        w(f"cls/{k}/flows/counts_r{r:02d}.u16", cnt, np.uint16)
        fj["baseline"][str(r)] = [{"layer": l, "H_B_given_A": rng.uniform(0.2, 0.9) * np.log2(r) / 3, "H_A_given_B": rng.uniform(0.2, 0.9) * np.log2(r) / 3,
                                   "H_A": np.log2(r) * 0.9, "H_B": np.log2(r) * 0.9, "r_used": r} for l in range(1, 13)]
    wj(f"cls/{k}/flows.json", fj)
    return hards


def cond_H(c):
    c = c.astype(float)
    rows = c.sum(1, keepdims=True)
    p = c / c.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(p * np.log2(np.where(c > 0, c / rows, 1)))
    return float(h)


def deltas(k, modes, spectra, pairs, full):
    res = {}
    for a, b in pairs:
        Pa, Pb = modes[k][a - 1], modes[k][b - 1]
        Q, _ = np.linalg.qr(np.concatenate([Pa, Pb], 1))
        A_ = Q.T @ Pa
        B_ = Q.T @ Pb
        vec = np.zeros((4, 16, N))
        out = {kk: [] for kk in ["evals", "cv", "cv_evalA", "cv_discarded", "cv_subspace_cos", "unsupported", "terms", "net", "path", "smooth", "all_vals_extreme"]}
        dattr = np.full((4, 16, 480), np.nan)
        dtop = np.full((4, 16, 32, 2), np.nan)
        for ti, t in enumerate(TS):
            Dl = B_ @ np.diag(spectra[k][b - 1] ** (2 * t)) @ B_.T - A_ @ np.diag(spectra[k][a - 1] ** (2 * t)) @ A_.T
            ev, V = np.linalg.eigh(Dl)
            rows = list(range(len(ev) - 1, len(ev) - 9, -1)) + list(range(8))
            F = Q @ V[:, rows]
            F *= np.sign(F[np.abs(F).argmax(0), np.arange(16)])
            vec[ti] = F.T * 100
            sm = np.stack([(((modes[k][l].T @ F) ** 2) * (spectra[k][l] ** (2 * t))[:, None]).sum(0) for l in range(L)], 1)
            terms = np.diff(sm, axis=1)
            mask = np.zeros(11, bool)
            mask[a - 1 : b - 1] = True
            e = ev[rows]
            cvA = e * rng.uniform(0.9, 1.1, 16)
            cv = cvA * rng.uniform(-0.15, 1.05, 16)
            out["evals"].append(e)
            out["cv"].append(cv)
            out["cv_evalA"].append(cvA)
            out["cv_discarded"].append(rng.uniform(0, 0.3, 16))
            out["cv_subspace_cos"].append([np.sort(rng.uniform(0.2, 1, 8))[::-1], np.sort(rng.uniform(0.2, 1, 8))[::-1]])
            out["unsupported"].append((cv < 0.25 * np.abs(cvA)) | (np.sign(cv) != np.sign(cvA)))
            out["terms"].append(terms)
            out["net"].append(terms[:, mask].sum(1))
            out["path"].append(np.abs(terms[:, mask]).sum(1))
            out["smooth"].append(sm)
            out["all_vals_extreme"].append([float(ev.min()), float(ev.max())])
            if full:
                for j in range(16):
                    lay = b if j < 8 else a
                    row = np.full(480, np.nan)
                    row[0] = rng.normal(0, 0.05)
                    row[1 : 1 + 6 * lay] = rng.normal(0, 0.04, 6 * lay)
                    row[73 : 73 + lay] = rng.normal(0.05, 0.08, lay)
                    row[85:469] = rng.normal(size=D) / np.sqrt(D)
                    row[469:474] = [rng.uniform(0.3, 0.95), rng.uniform(0.1, 0.9), rng.uniform(0.01, 0.2), lay, 10 ** rng.uniform(-3, 1)]
                    dattr[ti, j] = row
                    cols = rng.choice(lay * NN, 32, replace=False)
                    vals = np.sort(np.abs(rng.standard_cauchy(32)) * 0.004)[::-1] * rng.choice([-1, 1], 32)
                    dtop[ti, j, :, 0], dtop[ti, j, :, 1] = cols, vals
        key = f"a{a:02d}_b{b:02d}"
        if full:
            w(f"cls/{k}/delta_attr/{key}.f32", dattr, np.float32)
            w(f"cls/{k}/delta_attr/{key}_top.f32", dtop, np.float32)
        res[key] = out
        w(f"cls/{k}/delta/{key}.f16", vec, np.float16)
    triples = []
    for abc in TRIPLES:
        for t in TS:
            for sg in "+-":
                triples.append({"abc": abc, "t": t, "sign": sg, "cos_ac_ab": np.sort(rng.uniform(0, 1, 8))[::-1],
                                "cos_ac_bc": np.sort(rng.uniform(0, 1, 8))[::-1], "cos_ac_sum": np.sort(rng.uniform(0.3, 1, 8))[::-1]})
    wj(f"cls/{k}/delta.json", {"pairs": pairs, "results": res, "triples": triples})
    print("deltas", k, file=sys.stderr)


def neurons_heads_attr(k, views, modes, spectra, hards):
    rngs = np.zeros((L, NN, 2), np.float32)
    stats = np.zeros((L, NN, 2), np.float32)
    attrj = {"layers": []}
    for l in range(L):
        Wm = rng.normal(size=(views[l].shape[1], NN)) * 2.5
        pre = views[l] @ Wm + rng.normal(-0.5, 0.5, NN)
        g = pre * 0.5 * (1 + np.tanh(0.7978845608 * (pre + 0.044715 * pre**3)))
        lo, hi = g.min(0), g.max(0)
        q = np.round((g - lo) / np.where(hi > lo, hi - lo, 1) * 255).astype(np.uint8)
        w(f"cls/neurons/L{l+1:02d}.u8", q, np.uint8)
        rngs[l, :, 0], rngs[l, :, 1] = lo, hi
        stats[l, :, 0], stats[l, :, 1] = g.mean(0), g.std(0)
        gc = g - g.mean(0)
        gc /= np.linalg.norm(gc, axis=0) + 1e-12
        load = modes[l].T @ gc
        w(f"cls/{k}/load/L{l+1:02d}.f16", load, np.float16)
        ranks = np.argsort(np.argsort(g, 0), 0).astype(np.float64) + 1
        au = np.full((120, NN), np.nan)
        for ri, r in enumerate(RS):
            for j in range(r):
                m = hards[l, ri] == j
                n1 = m.sum()
                if 0 < n1 < N:
                    au[OFF[ri] + j] = (ranks[m].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (N - n1))
        w(f"cls/{k}/auroc/L{l+1:02d}.f16", au, np.float16)
        hn = np.abs(views[l] @ rng.normal(size=(views[l].shape[1], H))) + rng.uniform(0.1, 0.5, H)
        w(f"cls/heads/L{l+1:02d}.f16", hn.T, np.float16)
        A = np.full((16, 1 + 72 + 12 * NN), np.nan)
        for i in range(16):
            A[i, 0] = rng.normal(0, 0.05)
            A[i, 1 : 1 + 6 * (l + 1)] = rng.normal(0, 0.05, 6 * (l + 1)) * (1 + np.arange(6 * (l + 1)) / 20)
            for lp in range(l + 1):
                base = 73 + NN * lp
                A[i, base : base + NN] = rng.standard_cauchy(NN) * 0.0006 * (lp + 1) / (l + 1)
            A[i, 73 + NN * l : 73 + NN * (l + 1)] += load[i] * 0.004
            A[i] /= np.nansum(A[i])
        w(f"cls/{k}/attr/L{l+1:02d}.f32", A, np.float32)
        wc = rng.normal(0, 0.04, (16, 12, NN))
        wc[:, l + 1 :] = np.nan
        wc[:, l] += load[:16] * 0.5
        w(f"cls/{k}/writecos/L{l+1:02d}.f16", wc, np.float16)
        attrj["layers"].append({"layer": l + 1, "r2_probe": rng.uniform(0.4, 0.97, 16), "r2_heldout": rng.uniform(0.2, 0.9, 16),
                                "ridge_lambda": 10 ** rng.uniform(-3, 1, 16), "total_cov": rng.uniform(0.01, 0.2, 16),
                                "w_res": rng.normal(size=(16, D)) / np.sqrt(D)})
        print("neurons", l + 1, file=sys.stderr)
    w("cls/neurons/range.f32", rngs, np.float32)
    w("cls/neurons/stats.f32", stats, np.float32)
    wj(f"cls/{k}/attr.json", attrj)
    q = {}
    for a, b in DECL:
        q[f"a{a:02d}_b{b:02d}"] = {"sv": np.r_[1.0, np.sort(rng.uniform(0, 1, 32))[::-1]], "mean_evals": np.sort(rng.uniform(0, 1, 32))[::-1],
                                   "cos_q_right_vs_Pa": np.sort(rng.uniform(0, 1, 8))[::-1], "cos_q_left_vs_Pb": np.sort(rng.uniform(0, 1, 8))[::-1]}
    wj(f"cls/{k}/q.json", q)


def summaries(spectra):
    sel = {"views": {"cls": {}}, "validation": {"cls": {}}}
    for kid, fam, p in KERNELS:
        lam_k = spectra.get(kid, spectra["st15"])
        per = []
        for l in range(L):
            lam = np.r_[1.0, lam_k[l]]
            per.append({
                "stability": {str(r): np.sort(rng.uniform(0.2, 1, r))[::-1] for r in [8, 16, 32]},
                "stability_mean": {str(r): rng.uniform(0.4, 0.98) for r in [8, 16, 32]},
                "n_extended": {"A": int(rng.integers(20, 64)), "B": int(rng.integers(20, 64))},
                "eff_n_modes": rng.uniform(5, 6000, 64), "eff_n_proj": {str(r): rng.uniform(50, 8000) for r in [8, 16, 32]},
                "spectrum": {"lambda": lam, "min": rng.uniform(-0.2, 0) if fam == "self-tuned" else 0.0,
                             "neg_frac": rng.uniform(0, 0.02) if fam == "self-tuned" else 0.0, "gaps": -np.diff(lam),
                             "decay_r": {"0.5": int((lam > 0.5).sum()), "0.1": int((lam > 0.1).sum())}},
                "alpha1": {"lambda": lam * 0.98, "eff_n_modes": rng.uniform(5, 6000, 64)},
                "stability_capped": {str(r): False for r in [8, 16, 32]},
            })
        sel["views"]["cls"][kid] = {"per_layer": per}
        sel["validation"]["cls"][kid] = {"knn_acc": np.sort(rng.uniform(0.1, 0.85, L)), "eta2": rng.uniform(0, 0.8, (L, 64))}
    sel["default_kernels"] = KFULL
    sel["clear_cut"] = True
    sel["order"] = [x[0] for x in KERNELS]
    sel["scores"] = {kid: {"pass": list(rng.uniform(0, 1, 12) > 0.4), "score": int(rng.integers(3, 12)), "mean_stab16": rng.uniform(0.5, 0.95),
                           "n_stable": int(rng.integers(3, 12)), "n_nonlocal": int(rng.integers(3, 12))} for kid, *_ in KERNELS}
    sel["reasoning"] = "MOCKUP: no reasoning, synthetic numbers."
    sel["rule"] = "MOCKUP rule text."
    wj("selection.json", sel)
    ctl = {}
    for kid in KFULL:
        c = {"layers": [1, 6, 12], "noise_snr": 10.0, "t": TS, "per_layer": []}
        for l in [1, 6, 12]:
            rec = {"layer": l, "same_eigs_max_abs_eig": 1e-15}
            for nm, sc in [("identical", 1e-13), ("rotation", 1e-12), ("noise", 0.1)]:
                rec[nm] = {"max_abs_eig": list(rng.uniform(0.2, 1, 4) * sc), "top8": [], "bottom8": []}
            if l < 12:
                rec[f"adjacent_a{l:02d}_b{l+1:02d}_max_abs_eig"] = list(rng.uniform(0.1, 0.6, 4))
            c["per_layer"].append(rec)
        ctl[kid] = c
    wj("controls.json", ctl)
    curve = lambda lo, hi: np.clip(lo + (hi - lo) / (1 + np.exp(-(np.arange(L) - 5) / 1.5)) + rng.normal(0, 0.01, L), 0, 1)
    wj("decodability.json", {
        "cls": {"ridge": {"fine": curve(0.1, 0.82), "super": curve(0.3, 0.92)}, "knn": {"fine": curve(0.08, 0.8), "super": curve(0.28, 0.9)}},
        "mp": {"ridge": {"fine": np.r_[0.05, curve(0.15, 0.78)], "super": np.r_[0.2, curve(0.35, 0.9)]},
               "knn": {"fine": np.r_[0.04, curve(0.1, 0.74)], "super": np.r_[0.18, curve(0.3, 0.88)]}}})
    checks = [{"name": f"MOCK check {i}", "kernel": "st15" if i % 2 else "vmf", "t": [None, 1, 4, 64][i % 4], "value": float(rng.uniform(0, 1e-9)),
               "threshold": 1e-9, "relation": "<", "passed": i != 5, "note": "MOCKUP" if i == 5 else ""} for i in range(14)]
    wj("unit_tests.json", {"MOCKUP": True, "config": {"n_groups": 64, "N": 1920}, "checks": checks, "measured": {"mock": {"t1": [0.1, 0.2]}},
                           "n_checks": len(checks), "n_failed": 1, "all_passed": False})
    wj("mlp.json", {"MOCKUP": True, "decodability": {"A": curve(0.9, 0.6)[:5].tolist(), "B": curve(0.9, 0.55)[:5].tolist(), "y": curve(0.2, 0.99)[:5].tolist()},
                    "clusters": {"n_states_r16_coverage": rng.uniform(0.5, 1, 5).tolist()}, "entropies": {"H_next_given_prev": rng.uniform(0, 2, 4).tolist()}})
    wj("cifar_decodability.json", {"MOCKUP": True, "sites": ["stem"] + [f"b{i}" for i in range(1, 10)] + ["pool"],
                                    "flattened": {"ridge": curve(0.3, 0.88)[:11].tolist(), "knn": curve(0.25, 0.8)[:11].tolist()},
                                    "pooled": {"ridge": curve(0.35, 0.91)[:11].tolist(), "knn": curve(0.3, 0.9)[:11].tolist()}})


def make_thumbs(cls, sup, nu):
    from PIL import Image, ImageDraw

    hues = np.linspace(0, 1, SC, endpoint=False)
    import colorsys

    for c in range(C):
        sheet = Image.new("RGB", (640, 640), "white")
        dr = ImageDraw.Draw(sheet)
        for j in range(100):
            i = 100 * c + j
            r, cc = divmod(j, 10)
            x0, y0 = cc * 64, r * 64
            col = colorsys.hsv_to_rgb(hues[sup[i]], 0.55, 0.5 + 0.08 * nu[i])
            dr.rectangle([x0, y0, x0 + 63, y0 + 63], fill=tuple(int(255 * v) for v in col))
            ang = (c * 37) % 180
            for s in range(-64, 128, 10):
                dx = int(64 * np.cos(np.radians(ang)))
                dr.line([x0 + s, y0, x0 + s + dx, y0 + 63], fill=(255, 255, 255), width=1)
            dr.rectangle([x0 + 4, y0 + 22, x0 + 59, y0 + 42], fill="white")
            dr.text((x0 + 7, y0 + 25), f"M c{c}", fill="black")
        p = OUT / "thumbs" / f"sheet_{c:03d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(p, quality=80)


if __name__ == "__main__":
    main()
