"""Local §9.1 checks; use CENSUS_DEVICE=cuda to repeat the very same cases."""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import census as c
import kernels
import probe
import readout as r
import select_b as s
import synthetic


@pytest.fixture(scope="session")
def device():
    torch.set_num_threads(min(2, torch.get_num_threads()))
    value = torch.device(os.environ.get("CENSUS_DEVICE", "cpu"))
    if value.type == "cuda":
        if not torch.cuda.is_available():
            pytest.fail("CUDA was requested but is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
    return value


def on(x, device):
    return torch.as_tensor(x, device=device)


@pytest.mark.parametrize("reading,centring", tuple(itertools.product(("cosine", "dot"), ("C", "O"))))
def test_streamed_kernel_both_oracles(reading, centring, device):
    rng = np.random.default_rng(43)
    x = rng.normal(size=(3, 11, 7))
    pool = c.Pool(rng.integers(0, 30, size=x.shape[:2]), (0, 3, 8), 4, 8)
    alphas = tuple(c.MEMORIES.values())
    y, mu = c.centre(x, centring, reading, device)
    y0 = x - (x.mean((0, 1), keepdims=True) if centring == "C" else x.mean(1, keepdims=True))
    if reading == "cosine":
        y0 /= np.linalg.norm(y0, axis=2, keepdims=True)
    np.testing.assert_allclose(y.cpu(), y0, atol=1e-15, rtol=0)
    assert mu.numel() == (7 if centring == "C" else 21)
    state = c.Kernel(y, pool, reading, alphas)
    raw, dist = [], []
    for doc in range(3):
        for lo in range(0, pool.length, 3):
            data = state.tile(doc, lo, min(lo + 3, pool.length), raw_output=True)
            if data is not None:
                _, k, d = data
                raw.append(torch.stack(k).cpu().numpy())
                dist.append(torch.stack(d).cpu().numpy())
    raw, dist = np.concatenate(raw, axis=1), np.concatenate(dist, axis=1)
    nodes = [(doc, t) for doc in range(3) for t in range(3, 11)]
    for a_i, a in enumerate(alphas):
        explicit = np.zeros((pool.n, 11, 7))
        for i, (doc, t) in enumerate(nodes):
            explicit[i, :t + 1] = y0[doc, :t + 1][::-1] * np.sqrt(a ** np.arange(t + 1))[:, None]
        vectors = explicit.reshape(pool.n, -1)
        product = vectors @ vectors.T
        loops = np.array([[sum(a ** lag * np.dot(y0[u, p - lag], y0[v, q - lag])
                                for lag in range(min(p, q) + 1)) for v, q in nodes] for u, p in nodes])
        diag = np.diag(loops)
        if reading == "cosine":
            unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            direct_distance = 1 - loops / np.sqrt(diag[:, None] * diag)
            explicit_dist = 1 - unit @ unit.T
        else:
            direct_distance = ((diag[:, None] + diag) / 2 - loops) / diag.mean()
            explicit_dist = ((vectors[:, None] - vectors[None, :]) ** 2).sum(2) / (2 * diag.mean())
        for got, expected in [(raw[a_i], product), (raw[a_i], loops), (dist[a_i], direct_distance),
                              (dist[a_i], explicit_dist), (state.diag[a_i].cpu(), diag)]:
            np.testing.assert_allclose(got, expected, atol=1e-12, rtol=0)


def test_random_token_vectors_are_id_keyed_and_zero_ties_are_exact(device):
    rng = np.random.default_rng(7)
    tokens = np.array([[8, 9, 9, 514], [8, 9, 514, 9]])
    x = c.random_vectors(tokens, width=9)
    np.testing.assert_array_equal(x, rng.standard_normal((515, 9))[tokens])
    np.testing.assert_array_equal(c.random_vectors(np.array([514]), 9)[0], x[0, 3])
    for centring, reading in itertools.product(("C", "O"), ("cosine", "dot")):
        y, mu = c.centre(x, centring, reading, device)
        codes = tuple(on(v.flatten(), device) for v in c._duplicate_labels(tokens, mu.cpu().numpy()))
        k = c.Kernel(y, c.Pool(tokens, (0, 1), 1, 2), reading, tuple(c.MEMORIES.values()), duplicate_labels=codes)
        _, _, distances = k.tile(0, 0, 4)
        assert distances[0][1, 2].item() == 0  # current-token repetition
        for d in distances:
            assert d[0, 4].item() == 0 and d[1, 5].item() == 0  # shared prefixes


def test_kernel_guards_fail_closed(tmp_path, device):
    with pytest.raises(ValueError, match="near-zero"):
        c.centre(np.ones((2, 5, 3)), "C", "cosine", device)
    rng = np.random.default_rng(0)
    x = np.tile(rng.normal(size=(1, 5, 3)), (2, 1, 1))
    p = c.Pool(np.arange(10).reshape(2, 5), (0, 1), 1, 2)
    with pytest.raises(ValueError, match="off-diagonal"):
        c.run_group(tmp_path, "model", p, {"site04": x}, "dot", "C", device=device, k=2, pair_count=0)
    assert not (tmp_path / "census").exists()


def test_origin_document_counts_and_bootstrap_against_queries():
    # Documents have the same number of pool nodes but not the same numbers
    # of neighbours or of rows with any neighbour. Pooling pairs would give
    # a different answer from averaging the eligible query shares.
    counts = np.array([[8, 0, 0], [0, 2, 0], [0, 0, 0], [0, 0, 8], [3, 1, 0], [0, 0, 0]])
    selected = counts.sum(1)
    shares = np.divide(counts, selected[:, None], where=selected[:, None] > 0, out=np.full(counts.shape, np.nan))
    data, interval = c._origin_document_data(counts, shares, selected, (0, 4, 7))
    assert data["counts"] == [[8, 2, 0], [0, 0, 8], [3, 1, 0]]
    assert data["query_counts"] == [2, 2, 2] and data["queries_with_neighbours"] == [2, 1, 1]
    assert interval["seed"] == r.BOOTSTRAP_SEED and interval["draws"] == 200
    doc = shares.reshape(3, 2, 3)
    truth = [np.nanmean(doc[rows].reshape(-1, 3), axis=0) for rows in r.bootstrap_indices(3)]
    np.testing.assert_allclose([interval["p5"], interval["p95"]], np.percentile(truth, [5, 95], axis=0), atol=1e-15)
    reconstructed = np.array(data["share_sums"]).sum(0) / sum(data["queries_with_neighbours"])
    np.testing.assert_allclose(reconstructed, np.nanmean(shares, axis=0), atol=1e-15)
    _, empty = c._origin_document_data(np.zeros_like(counts), np.full_like(shares, np.nan), np.zeros_like(selected), (0, 4, 7))
    assert empty["valid_draws"] == 0 and empty["p5"] is None and empty["p95"] is None


@pytest.mark.parametrize("reading", ("cosine", "dot"))
def test_saved_display_distances_against_explicit_history(tmp_path, reading, device):
    rng = np.random.default_rng(59)
    x = rng.normal(size=(2, 12, 7))
    p = c.Pool(rng.integers(0, 10, size=x.shape[:2]), (2, 4), 3, 8)
    c.run_group(tmp_path, "toy", p, {"site04": x}, reading, "O", device=device, k=5, window=2,
                memories=("none", "8"), screen=False, pair_count=0, row_block=4)
    y = x - x.mean(1, keepdims=True)
    if reading == "cosine":
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
    for memory, mask in itertools.product(("none", "8"), c.MASKS):
        with np.load(c.setting_path(tmp_path, "toy", reading, "O", "site04", memory, mask)) as f:
            raw = f["display_distances"]
            assert raw.dtype == np.float32 and raw.shape == (p.n, 8)
            rows = [(i, t) for i in range(len(p.texts)) for t in range(p.start - 1, p.length)]
            a = c.MEMORIES[memory]
            feat = np.zeros((p.n, p.length, x.shape[-1]))
            for j, (text, t) in enumerate(rows):
                feat[j, :t + 1] = y[text, :t + 1][::-1] * np.sqrt(a ** np.arange(t + 1))[:, None]
            feat = feat.reshape(p.n, -1)
            if reading == "cosine":
                feat /= np.linalg.norm(feat, axis=1, keepdims=True)
                oracle = 1. - feat @ feat.T
            else:
                s = (feat ** 2).sum(1).mean()
                oracle = ((feat[:, None] - feat[None, :]) ** 2).sum(-1) / (2 * s)
            j = f["indices"]
            expected = np.full_like(raw, np.nan)
            expected[:, :j.shape[1]] = np.where(j >= 0, oracle[np.arange(p.n)[:, None], j], np.nan)
            np.testing.assert_array_equal(raw, expected)


def test_masks_origins_and_chance(device):
    text, pos, token = [on(a, device) for a in ([0, 0, 0, 0, 1, 1], [100, 104, 164, 165, 8, 108], [1, 1, 4, 1, 6, 1])]
    q = on([0], device)
    j = torch.arange(6, device=device)
    origin, gap = c.origin_codes(text, pos, q, j)
    assert origin.tolist() == [[2, 2, 2, 1, 0, 0]]
    d = on([[0., .1, .08, .5, .04, .2]], device).double()
    expected = [([1, 2, 3, 4, 5], [4, 2, 1], [1, 0, 2], [2, 1, 2], 34., 1, 3),
                ([3, 4, 5], [4, 5, 3], [2, 1, 0], [2, 1, 0], 65., 2, 2),
                ([4, 5], [4, 5, -1], [2, 0, 0], [2, 0, 0], None, 1, 1)]
    for mask, (eligible, indices, origins, counts, median, same, chance_same) in zip(c.MASKS, expected):
        ok = c.eligible_mask(origin, q, j, mask, ntexts=2)
        assert torch.where(ok)[1].tolist() == eligible
        top = c.top_with_ties(d.masked_fill(~ok, torch.inf), 3)
        assert top.indices[0].tolist() == indices
        f = c.statistics(top, origin, gap, ok, q, j, token)
        assert f["origins"].tolist() == [origins] and f["eligible"].tolist() == [counts]
        assert f["same_token"].item() == same and f["same_token_eligible"].item() == chance_same
        if median is None:
            assert torch.isnan(f["gap_median"]).all()
        else:
            assert f["gap_median"].item() == median
        arrays = f | {"overlap_expected": torch.ones(1, 1), "overlap_maximum": torch.ones(1, 1)}
        g = c.geometry_summary({key: value.cpu().numpy() for key, value in arrays.items()}, 3)
        assert g["chance_origin"] == [n / len(eligible) for n in counts]
        assert g["chance_same_token"] == chance_same / len(eligible)
        assert g["origin"] == [i / sum(origins) for i in origins]
    with pytest.raises(ValueError, match="more than one text"):
        c.eligible_mask(origin, q, j, "other_texts", ntexts=1)


@pytest.mark.parametrize("mask", c.MASKS)
def test_exact_huge_ties_under_each_mask(mask, device):
    n = 1024
    text = on([0, 0, 0, *([1] * n)], device)
    pos = on([10, 28, 900, *range(1, n + 1)], device)
    q, j = on([0], device), torch.arange(n + 3, device=device)
    origin, _ = c.origin_codes(text, pos, q, j)
    ok = c.eligible_mask(origin, q, j, mask, ntexts=2)
    a = on(np.r_[3., 3., 3., np.ones(64), np.full(n - 64, 4.)][None], device)
    a[0, 3:66] = .25  # exactly one of A's selected is at its cutoff
    b = on(np.r_[3., 3., 3., np.ones(n)][None], device)
    at, bt = (c.top_with_ties(d.masked_fill(~ok, torch.inf), 64) for d in (a, b))
    assert at.indices.tolist() == bt.indices.tolist() == [list(range(3, 67))]
    assert (at.t_all.item(), at.t_in.item(), bt.t_all.item(), bt.t_in.item()) == (1, 1, 1024, 64)
    assert bt.cutoff.item() == 1.
    count = c.overlap_counts(at, bt).cpu().numpy()
    count_v = (at.t_in, at.t_all, bt.t_in, bt.t_all)
    expected, maximum = c.overlap_values(count, *(v.cpu().numpy() for v in count_v))
    assert count.tolist() == [[0, 63, 0, 1]]
    assert expected.item() / 64 == .0625
    assert maximum.item() / 64 == 1


def test_tolerance_cutoff_and_both_overflow(device):
    a = on(np.array([[4., 1. + 4e-13, 3., 1., 1. - 4e-13, .5, 1. + 4e-12]]), device)
    b = on(np.array([[.25, 1., 1., 1., 4., 1., 8.]]), device)
    ta, tb = (c.top_with_ties(d, 3) for d in (a, b))
    assert ta.indices.tolist() == [[5, 1, 3]] and tb.indices.tolist() == [[0, 1, 2]]
    assert ta.t_all.item() == 3 and ta.t_in.item() == 2
    counts = c.overlap_counts(ta, tb).cpu().numpy()
    e, m = c.overlap_values(counts, *[n.cpu().numpy() for n in (ta.t_in, ta.t_all, tb.t_in, tb.t_all)])
    assert counts.tolist() == [[0, 1, 0, 2]]
    assert e == pytest.approx([.5 + 2 * (2 / 3) * .5]) and m.tolist() == [2]


def test_matcher_hand_cases_and_replay():
    text = np.array([0] * 10 + [1] * 4)
    pos = np.array([10, 11, 14, 18, 21, 75, 76, 400, 600, 900, 2, 5, 7, 9])
    ids = np.array([99, 5, 5, 7, 5, 6, 6, 6, 8, 8, 9, 7, 9, 9])
    match = r.Matcher(text, pos, ids)
    setting = (4, "cosine", "C", "8", "none")
    q = np.repeat([0, 0, 0, 1, 4], [6, 3, 1, 1, 2])
    j = np.array([1, 3, 5, 8, 10, 11, 1, 2, 10, 6, 0, 2, 3])
    block = np.repeat(np.arange(5), [6, 3, 1, 1, 2])
    controls, reasons = match.batch("pythia", setting, q, j, block)
    again, same = match.batch("pythia", setting, q, j, block)
    np.testing.assert_array_equal(again, controls)
    np.testing.assert_array_equal(same, reasons)
    assert controls[0] == 2 and reasons[1] == 1 and reasons[6] == reasons[7] == 3
    assert reasons[5] == reasons[10] == reasons[12] == 1 and reasons[11] == 2
    for i, peer, row, control, reason in zip(q, j, block, controls, reasons):
        others = np.flatnonzero((text == text[peer]) & (ids == ids[peer]) & (np.arange(len(ids)) != peer))
        near = [h for h in others if text[peer] != text[i] or (pos[h] != pos[i] and
                r.lag_bin(abs(pos[i] - pos[h])) == r.lag_bin(abs(pos[i] - pos[peer])))]
        available = sorted(set(near) - {i, *j[block == row]}, key=lambda h: pos[h])
        expected = 0 if available else 3 if near else 2 if len(others) else 1
        assert reason == expected
        if not reason:
            seed = r.keyed_seed("pythia", setting, text[i], pos[i], text[peer], pos[peer])
            rank = r._uniform_rank(np.array([seed]), np.array([len(available)])).item()
            assert control == available[rank]
        else:
            assert control == -1


def test_predictions_contributions_and_document_bootstrap():
    text = np.repeat([4, 9], 5)
    pos = np.tile(np.arange(1, 6), 2)
    token = np.repeat([3, 6], 5)
    pids = np.array([[1, 3], [1, 5], [2, 5], [4, 6], [4, 7], [1, 6], [7, 2], [1, 8], [3, 4], [1, 3]])
    probs = np.tile([.5, .2], (len(pos), 1))
    pred = r.Predictions(pids, probs, pids[:, 0], probs.sum(1))
    assert pred.overlap(np.array([0, 0, 0]), np.array([1, 2, 8])).tolist() == [.5, 0., .2]
    q = np.array([0, 1, 5, 6])
    indices = np.full((10, 2), -1)
    indices[q] = [[5, 6], [5, 7], [0, 1], [1, 2]]
    # Query 0 has one strict neighbour (5), and three equally likely cutoff IDs.
    ties = np.array([6, 7, 8, 7, 1, 2])
    ptr = np.array([0, 3, 4, 4, 4, 4, 5, 6, 6, 6, 6])
    t_in, t_all = np.zeros(10, int), np.diff(ptr)
    t_in[q] = 1
    sums, raw, witness = r.contributions(q, q, indices, ptr, ties, t_in, t_all, r.Matcher(text, pos, token), pred,
                                           "m", (4, "cosine", "C", "2", "none"), record_all=True)
    assert raw.sum() == 10 and sums[..., r.F["matched"]].sum() == pytest.approx(8)
    assert sums[:, 0, r.F["matched"]].tolist() == pytest.approx([4., 4.])
    assert sums[..., r.F["mass_query"]].sum() == pytest.approx(.7 * 8)
    covered = np.array([True, False, False, False, False, False])
    samples = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    boots = r.bootstrap_scores(sums, covered, samples)

    def overlap(i, j):
        a, b = [dict(zip(pids[k], probs[k])) for k in (i, j)]
        return sum(min(a[v], b[v]) for v in a.keys() & b.keys())

    pair_adv = np.array([overlap(i, j) - overlap(i, k) for i, j, k in
                         zip(witness["query"], witness["neighbour"], witness["control"])])
    pair_doc = text[witness["query"]] == 9
    weight = witness["weight"]
    decomp = [(pair_adv[pair_doc == i] * weight[pair_doc == i]).sum() for i in (0, 1)]
    np.testing.assert_allclose(sums[:, 0, r.F["advantage"]], decomp, atol=1e-15, rtol=0)
    direct = [np.average(np.concatenate([pair_adv[pair_doc == i] for i in d]),
                         weights=np.concatenate([weight[pair_doc == i] for i in d])) for d in samples]
    np.testing.assert_allclose(boots, direct, atol=1e-15, rtol=0)
    summary = r.summarize(sums, min_matched=2)
    assert summary["eligible"] and summary["strata"]["near/rare"]["status"] == "absent"
    assert r.summarize(sums, min_matched=200)["strata"]["other/rare"]["status"] == "uncovered"
    sums[0, 2, r.F["all_candidates_excluded"]] = 1
    assert not r.summarize(sums, min_matched=2)["eligible"]


def test_suffix_conditional_dense_oracle(device):
    rng = np.random.default_rng(14)
    d = rng.uniform(.05, 1, (5, 151))
    d[rng.random(d.shape) < .1] = np.inf
    beta0, dmin0, logz0, logp0 = kernels._cond_rows(d, 64, False, renyi=2)
    p, status, beta, dmin, logz = c.cond_rows(on(d, device))
    assert status.tolist() == [0] * 5
    for a, b in ((p, np.exp(logp0)), (beta, beta0), (dmin, dmin0), (logz, logz0)):
        np.testing.assert_allclose(a.cpu(), b, atol=1e-12, rtol=0)
    checks = np.r_[np.array([[0.] * 65 + [1.] * 86, [0.] * 64 + [1.] * 87]), d[:2]]
    checks[2, :90] = np.inf
    checks[3] = np.r_[[0.] * 64, [np.inf] * 87]
    probs, status, beta, *_ = c.cond_rows(on(checks, device))
    assert status.tolist() == [2, 2, 1, 0]
    assert torch.count_nonzero(probs[:3]) == 0
    assert beta[-1].item() == 0  # uniform target reached at beta=0
    assert probs[-1, 0].item() == 1 / 64


def test_stitched_baseline_hand_and_run(tmp_path, device):
    source = np.array([0, 0, 1, 1, 0])
    np.testing.assert_equal(c.position_baseline(source), [1., .5, 0., .5, 1., .5, 0., .5, 1.])
    spec = dict(contexts=[dict(id=9, token_ids=[1, 2, 3, 4, 5],
                               source_positions=[None, [0, 1], [0, 2], [1, 1], [1, 2], [0, 3]])],
                node_sets=dict(one_text={"9": [[9, 1, 6]]}))
    pool = c.pool_from_probe(spec, stitch=9)
    np.testing.assert_equal(pool.source, source)
    x = np.array([[[0., .1], [.01, .1], [3., 3.], [4., 4.], [8., 8.]]])
    summary = c.run_group(tmp_path, "stitch", pool, {"site04": x}, "dot", "C", k=1, screen=False,
                          masks=("none", "window"), row_block=2, memories=("none",), device=device)
    assert summary["masks"] == ["none", "window"]
    with np.load(c.setting_path(tmp_path, "stitch", "dot", "C", "site04", "none", "none")) as f:
        assert f["indices"][0, 0] == 1
        assert f["stitch_observed"][0] == 1 and f["stitch_baseline"][0] == .5 and f["stitch_beyond"][0] == .5
    with pytest.raises(ValueError, match="more than one text"):
        c.run_group(tmp_path, "stitch", pool, {"site04": x}, "dot", "C", device=device)


def test_token_identity_for_two_short_texts():
    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs.get("add_special_tokens") is False
            return {"A short text.": [5, 9, 11], "  Another one!": [12, 5]}[text]

        def convert_ids_to_tokens(self, ids):
            return [str(i) for i in ids]

    first, second = Tokenizer(), Tokenizer()
    assert probe.assert_token_identity("A short text.", first, second) == [5, 9, 11]
    assert probe.assert_token_identity("  Another one!", first, second) == [12, 5]
    second.convert_ids_to_tokens = lambda ids: ["wrong"] * len(ids)
    with pytest.raises(probe.TokenIdentityError, match="strings"):
        probe.assert_token_identity("  Another one!", first, second)
    if fixtures := os.environ.get("TM_EXP1_HF_FIXTURES"):
        transformers = pytest.importorskip("transformers")
        real = [transformers.AutoTokenizer.from_pretrained(Path(fixtures) / name, local_files_only=True,
                                                            trust_remote_code=False) for name in ("pythia", "olmo")]
        for text in ("A short text.", "  Another one!"):
            assert probe.assert_token_identity(text, *real)


def _write_probe(root, tokens, start, suffix):
    length = tokens.shape[1]
    probe = dict(contexts=[dict(id=i, kind="pool", token_ids=doc.tolist()) for i, doc in enumerate(tokens)],
                 node_sets=dict(census_pool=[[i, start, length + 1] for i in range(len(tokens))],
                                suffix=[[i, suffix, length + 1] for i in range(len(tokens))],
                                query_sample=[[i, pos] for i in range(len(tokens)) for pos in (start, suffix, length)],
                                audit_queries=[[i, suffix] for i in range(len(tokens))]))
    c.write_json(root / "probe" / "probe.json", probe)
    return c.pool_from_probe(probe)


def _write_preds_and_acts(root, model, pool, x, guesses):
    for i, ctx in enumerate(pool.texts):
        act = np.zeros((5, pool.length + 1, x.shape[-1]), dtype=np.float32)
        act[4, 1:] = x[i]
        c.atomic(root / "acts" / model / f"c{ctx:02d}.npy", np.save, act)
        g = np.r_[0, guesses[i]]
        ids = np.column_stack((g, g + 30))
        c.atomic(root / "pred" / model / f"c{ctx:02d}.npz", np.savez, top20_ids=ids, top1_id=g,
                 top20_probs=np.tile([.5, .25], (len(g), 1)), top20_mass=np.full(len(g), .75))


def test_small_synthetic_end_to_end_and_audit_files(tmp_path, device):
    gen = dict(texts=3, positions=24, width=10, sites=2, vocab=7)
    data = synthetic.generate(5, "planted", **gen)
    strong = synthetic.generate(5, "strong", **gen)
    drift = synthetic.generate(5, "drift", **gen)
    aa, bb = (synthetic.generate(5, v, **gen) for v in ("noise_a", "noise_b"))
    np.testing.assert_equal(data.states[0], strong.states[0])
    np.testing.assert_equal(drift.tokens, data.tokens)
    np.testing.assert_equal(aa.regime, bb.regime)
    assert not np.array_equal(aa.states, bb.states)
    pool = _write_probe(tmp_path, data.tokens, 2, 10)
    for variant, model in [(data, "pythia"), (strong, "olmo")]:
        x = variant.states[1]
        _write_preds_and_acts(tmp_path, model, pool, x, variant.predictors)
        inputs = {"site04": x, "random": c.random_vectors(pool.tokens, width=10), "embedding": variant.embeddings[data.tokens]}
        c.run_group(tmp_path, model, pool, inputs, "cosine", "C", k=4, target=4, window=2, row_block=9, device=device)
        report = r.run_readout(tmp_path, model, sites=(4,), min_matched=1)
        assert len(report) == 15
        with np.load(report[0]) as f:
            assert f["query"].shape[0] >= len(pool.audit) * 4
            assert f["by_document"][:, :, 1:5].sum() == pytest.approx(len(pool.query) * 4)
        base = tmp_path / "audit_a" / "expected" / model / "cosine_C"
        with np.load(base / "site04_pairs.npz") as f:
            assert f["distance"].shape == (5, 1000)
            y, _ = c.centre(x, "C", "cosine")
            v = y[:, pool.start - 1:].flatten(0, 1).numpy()
            np.testing.assert_allclose(f["distance"][0], 1 - (v[f["query"]] * v[f["candidate"]]).sum(1), atol=1e-12)
        with np.load(base / "random" / "none_none.npz") as f:
            assert f["query"].tolist() == list(pool.audit)
            np.testing.assert_equal(np.diff(f["tie_indptr"]), f["t_all"])
            np.testing.assert_equal(np.diff(f["screen_tie_indptr"]), f["screen_t_all"])
        with np.load(c.setting_path(tmp_path, model, "cosine", "C", "site04", "8", "none")) as a, \
             np.load(c.setting_path(tmp_path, model, "cosine", "C", "site04", "8", "window")) as b:
            np.testing.assert_equal(b["screen_removed"], a["screen_mass"][:, 2])
            assert (b["screen_mass"][:, 2] == 0).all()
    decision = s.select(tmp_path, sites=(4,), readout_sites=(4,), required_sites=1)
    assert len(decision["table"]) == 32
    assert sum(row["complete"] for row in decision["table"]) == 8
    assert (tmp_path / "analysis" / "stage_b_selection.json").exists()


def test_select_order_tie_blocking_x_and_empty(tmp_path):
    def fixture(model, reading, centring, memory, mask, site, worst=.35, tied=False):
        p = c.setting_path(tmp_path, model, reading, centring, c.input_name(site), memory, mask)
        c.atomic(p, np.savez, screen_mass=[[.66, .25, .09 if mask == "none" else 0.]],
                 screen_status=[0], screen_origins=[[45, 14, 5]], screen_removed=[.3],
                 screen_overlap_expected=[[20, 4 if tied else 64 * worst, 10]],
                 screen_overlap_maximum=[[20, 64 if tied else 64 * worst, 10]])

    for model, (reading, centring) in itertools.product(("pythia", "olmo"), c.PRIORITY):
        path = tmp_path / "census" / model / f"{reading}_{centring}" / "group.json"
        c.write_json(path, dict(inputs=["site04", "site08"], memories=list(c.MEMORIES), masks=list(c.MASKS), k=64))
        for site in (4, 8):
            for memory, mask, worst, tie in (("2", "none", .35, True), ("8", "none", .25, False),
                                             ("64", "none", .35, False), ("whole", "none", .4, False),
                                             ("2", "window", .05, False)):
                fixture(model, reading, centring, memory, mask, site, worst, tie)
                by_doc = np.zeros((2, 6, len(r.FIELDS)))
                by_doc[:, 0, r.F["matched"]] = 201
                by_doc[:, 0, r.F["advantage"]] = 20 if (site == 4 and reading == "cosine" and centring == "C" and memory == "64") else 2
                if memory == "whole":  # cannot alter the compared weights
                    by_doc[:, 1, r.F["matched"]] = by_doc[:, 1, r.F["advantage"]] = 400
                c.atomic(r.readout_path(tmp_path, model, reading, centring, site, memory, mask), np.savez,
                         by_document=by_doc, documents=np.arange(2), min_matched=200)
    result = s.select(tmp_path, sites=(4, 8), readout_sites=(4,), required_sites=2)
    assert result["M"] == {"memory": "8", "mask": "none", "reading": "cosine", "centring": "C"}
    assert result["L"] == s.R and result["solve"].count(s.R) == 1
    assert result["X"]["memory"] == "64" and result["common_strata"] == ["other/rare"]
    blocked = next(row for row in result["table"] if row["construction"] == dict(result["M"], memory="2"))
    assert not blocked["passed"]
    failed = blocked["models"]["pythia"]["sites"]["4"]
    assert failed["expected_overlap"]["random"] == .0625 and failed["maximum_overlap"]["random"] == 1
    empty = s.select(tmp_path / "empty", sites=(4,), required_sites=1)
    assert not empty["recommend_stage_b"] and empty["owner_override_only"] == [s.R, s.FALLBACK]


def test_atomic_setting_publication_and_capped_tail(tmp_path, device):
    rng = np.random.default_rng(42)
    p = c.Pool(rng.integers(8, size=(2, 12)), (0, 1), 1, 8, (8,), (8,))
    raw = rng.normal(size=(2, 12, 6))

    def stop(path):
        assert path.is_file()
        raise RuntimeError("simulated cap during publication")

    with pytest.raises(RuntimeError, match="simulated cap"):
        c.run_group(tmp_path, "pilot", p, {"site04": raw}, "cosine", "C", k=4, target=4,
                    window=3, on_setting=stop, device=device)
    path = c.setting_path(tmp_path, "pilot", "cosine", "C", "site04", "none", "none")
    with np.load(path) as f:
        assert f["indices"].shape == (p.n, 4)
    assert not json.loads((path.parent.parent / "group.json").read_bytes())["complete"]
    assert not path.with_name("none_window.npz").is_file()
    assert (tmp_path / "audit_a" / "expected" / "pilot" / "cosine_C" / "site04" / "none_none.npz").is_file()


def test_synthetic_runner_publishes_small_summaries(tmp_path, device):
    result = synthetic.run_synthetic(tmp_path, seeds=(3,), variants=("drift",), groups=(c.PRIORITY[0],),
                                     generator_options=dict(texts=2, positions=17, width=7, sites=2), start=2, device=device)
    assert len(result) == 1 and result[0]["complete"] and not result[0]["lists_saved"]
    summaries = json.loads((tmp_path / "analysis" / "geometry" / "synthetic_drift_seed3" / "cosine_C.json").read_bytes())
    assert len(summaries) == 4 * 5 * 3
    assert (tmp_path / "analysis" / "synthetic" / "synthetic_drift_seed3.npz").exists()


def test_timescales_against_brute_force(tmp_path):
    rng = np.random.default_rng(22)
    x = rng.normal(size=(2, 8, 3))
    answer = r.autocorrelation(x, max_lag=6, width_block=2)
    y, _ = c.centre(x, "C", "cosine", "cpu")
    z = y.numpy()
    rho = [(z[:, d:] * z[:, :-d]).sum(-1).mean() for d in range(1, 7)]
    background = (z[0, :, None] * z[1, None, :]).sum(-1).mean()
    np.testing.assert_allclose(answer["rho"], rho, atol=1e-13, rtol=0)
    assert answer["background"] == pytest.approx(background, abs=1e-13)
    assert answer["half_lag"] == next((d for d, v in enumerate(rho, 1) if v - background <= (rho[0] - background) / 2), None)
    n = np.array([[[[1, 2]], [[3, 6]]], [[[9, 10]], [[11, 12]]]])
    attn = tmp_path / "sums.npz"
    np.savez(attn, context_ids=[4, 9], layer_ids=[1], query_count=n * 5,
             sink_mass_sum=n, non_sink_mass_sum=n * 4, distance_numerator_sum=n * 12)
    head = r.attention_summary(attn, [4])[0]
    assert head["sink"]["heads"] == [.2, .2] and head["distance"]["quartiles"] == [3., 3., 3.]


def test_optional_tiny_real_pythia(tmp_path, device):
    if not os.environ.get("CENSUS_TINY_REAL"):
        pytest.skip("optional: set CENSUS_TINY_REAL=1 with locally cached Pythia-70m and Transformers")
    transformers = pytest.importorskip("transformers")
    import extract
    try:
        name, revision = "EleutherAI/pythia-70m", "a39f36b100fe8a5377810d56c3f4789b9c53ac42"
        options = dict(local_files_only=True, trust_remote_code=False)
        if fixtures := os.environ.get("TM_EXP1_HF_FIXTURES"):
            name = Path(fixtures) / "pythia70m"
        else:
            options["revision"] = revision
        tokenizer = transformers.AutoTokenizer.from_pretrained(name, **options)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            name, attn_implementation="eager", use_safetensors=True, dtype=torch.float32, **options).eval().to(device)
    except OSError as error:
        pytest.skip(f"optional local Pythia-70m is unavailable: {error}")
    texts = ("The house stood at a crossroads where the forest opened out toward the lake. " * 3,
             "Flowers appeared between stones and the old tree leaned against the wall. " * 3,
             "The gardener went to see what had grown under the cedar since the last heavy rain. " * 3)
    seq = [tokenizer.encode(text, add_special_tokens=False)[:19] for text in texts]
    pool = _write_probe(tmp_path, np.array(seq), 4, 13)
    ctx = [dict(id=i, token_ids=doc) for i, doc in enumerate(seq)]
    meta = extract.extract_contexts(model, ctx, tokenizer.eos_token_id, tmp_path, "pythia", query_start=4,
                                    query_split=13, causal_split=9, model_revision=revision)
    assert meta["causal"]["passed"]
    inputs = {"site04": c.load_site(tmp_path, "pythia", pool, 4), "embedding": c.load_site(tmp_path, "pythia", pool, 0),
              "random": c.random_vectors(pool.tokens, model.config.hidden_size)}
    c.run_group(tmp_path, "pythia", pool, inputs, "cosine", "C", k=4, target=4, row_block=16, device=device)
    assert len(r.run_readout(tmp_path, "pythia", sites=(4,), min_matched=1)) == 15
    assert len(s.select(tmp_path, models=("pythia",), sites=(4,), readout_sites=(4,), required_sites=1)["table"]) == 32
