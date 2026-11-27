"""The int8 rule of pack_depth.py (SPEC_DEPTH.md section 8, square-root companding with one scale per layer and
mode), and the code parameters it quotes.

    cd exp1 && TMPDIR=/var/tmp uv run --with numpy --with pytest python -m pytest -q --basetemp=/var/tmp/tm_pytest_pd test_pack_depth.py
"""
import math

import numpy as np

import pack_depth as pd


def reference(a, floor=None):
    """The stated rule, written out element by element, independently of pack_depth.quantise."""
    v = np.asarray(a[:, :, :16], dtype=np.float32).astype(np.float64)
    L, M, K = v.shape
    scales = np.zeros((L, K))
    for l in range(L):
        for k in range(K):
            col = v[l, :, k]
            fin = col[np.isfinite(col)]
            scales[l, k] = np.abs(fin).max() if fin.size else 0.0
            if floor is not None:
                scales[l, k] = max(scales[l, k], floor[l][k])
    out = np.empty(v.shape, dtype=np.int8)
    for (l, i, k), x in np.ndenumerate(v):
        s = scales[l, k]
        if not math.isfinite(x):
            out[l, i, k] = -128
        elif s == 0:
            out[l, i, k] = 0
        else:
            c = round(127 * math.sqrt(abs(x) / s))      # Python round is half-to-even, as numpy.rint
            out[l, i, k] = int(math.copysign(min(127, c), x)) if c else 0
    return out, scales.tolist()


def decode(q, scales):
    s = np.asarray(scales)[:, None, :]
    return np.sign(q) * (q.astype(np.float64) / 127) ** 2 * s


def sample(seed=0):
    rng = np.random.default_rng(seed)
    a = (rng.standard_t(2, (4, 60, 20)) * 0.01).astype(np.float32)   # heavy tails, like the real modes
    a[:, :, 4] = 0                       # a mode that is identically zero
    a[2, :, 6] = 0                       # a mode that is zero at one layer only
    a[:, 7, 9] = np.nan                  # a withheld coordinate
    a[:, :, 11] = np.nan                 # a withheld mode
    return a


def test_rule_matches_reference():
    a = sample()
    q, scales, sat = pd.quantise(a)
    r, rs = reference(a)
    assert q.dtype == np.int8 and q.shape == (4, 60, 16) and q.flags.c_contiguous
    assert np.array_equal(q, r) and scales == rs
    assert all(float(np.float32(s)) == s for row in scales for s in row)
    assert np.all(q[:, :, 4] == 0) and np.all(q[2, :, 6] == 0) and np.all(q[:, :, 11] == -128) and np.all(q[:, 7, 9] == -128)
    assert sat == int((np.abs(q.astype(np.int16)) == 127).sum())
    assert sat >= 4 * 13                 # each live (layer, mode) reaches 127 at its largest value


def test_decoding_error_and_reencoding():
    a = sample(1)
    q, scales, _ = pd.quantise(a)
    x = a[:, :, :16].astype(np.float64)
    ok = q != -128
    dec = decode(q, scales)
    s = np.broadcast_to(np.asarray(scales)[:, None, :], x.shape)
    u = np.sqrt(np.divide(np.abs(x), s, out=np.zeros_like(x), where=s > 0))
    bound = s * (2 * u / 254 + 1 / 254 ** 2)          # half a step in the square-root domain
    assert np.all(np.abs(dec - x)[ok] <= bound[ok] * (1 + 1e-9) + 1e-300)
    q2, _, _ = pd.quantise(np.where(ok, dec, np.nan).astype(np.float32), floor=scales)
    assert np.array_equal(q2[ok], q[ok])               # decoded values encode back to the same codes
    small = ok & (np.abs(x) > s * 1e-4) & (np.abs(x) < s / 1000)
    assert small.any() and np.all(q[small] != 0)                       # companding keeps small values off zero (step near 0: scale / 127^2)


def test_floor_from_the_set_file_and_symmetry():
    a = sample(2)
    q, scales, _ = pd.quantise(a)
    b = (a * np.float32(0.5)).astype(np.float32)        # a "panel" inside the set's range
    qb, sb, _ = pd.quantise(b, floor=scales)
    rb, rsb = reference(b, floor=scales)
    assert np.array_equal(qb, rb) and sb == rsb and sb == scales
    c = (a * np.float32(3)).astype(np.float32)          # a "panel" beyond it: its own maxima win, nothing clips
    qc, sc, _ = pd.quantise(c, floor=scales)
    assert all(sc[l][k] >= scales[l][k] for l in range(4) for k in range(16))
    assert np.array_equal(qc, reference(c, floor=scales)[0])
    qn, _, _ = pd.quantise(-a)
    fin = q != -128
    assert np.array_equal(qn[fin], -q[fin])


def test_code_parameters_are_read_from_the_pipeline():
    p = pd.code_parameters()
    assert p["knn"] == 15 and p["tie"] == 0.01 and p["condition"] == 1e8
