"""§5C prediction concordance and §10 descriptive reference timescales."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from census import (MASKS, MEMORIES, PRIORITY, atomic, centre, input_name, load_probe,
                    load_site, pool_from_probe, setting_path, write_json)

SITES = (4, 8, 12, 16)
REASONS = ("matched", "no_same_token_in_text", "empty_lag_bin", "all_candidates_excluded")
STRATA = tuple(f"{origin}/{freq}" for origin in ("other", "far", "near") for freq in ("rare", "frequent"))
FIELDS = ("advantage", "matched", *REASONS[1:], "top1_neighbour", "top1_control",
          "mass_query", "mass_neighbour", "mass_control")
F = {name: i for i, name in enumerate(FIELDS)}
BOOTSTRAP_SEED = 23
MIN_MATCHED = 200


def _mix(x):
    """SplitMix64 finaliser, including its Weyl increment, arithmetic mod 2**64."""
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9e3779b97f4a7c15)
        z = (z ^ (z >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        z = (z ^ (z >> 27)) * np.uint64(0x94d049bb133111eb)
        return z ^ (z >> 31)


def keyed_seed(model, setting, query_text, query_pos, neighbour_text, neighbour_pos):
    """SHA256 of compact JSON [model, list(setting)], then mix each u64 field.

    Use the first 8 digest bytes as a little-endian integer. The setting is
    (site, reading, centring, memory, mask). No row index or order enters the key.
    """
    encoded = json.dumps([model, list(setting)], separators=(",", ":"), ensure_ascii=True).encode("ascii")
    x = np.uint64(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little"))
    for field in (query_text, query_pos, neighbour_text, neighbour_pos):
        x = _mix(x ^ np.asarray(field, dtype=np.uint64))
    return x


def _uniform_rank(seed, n):
    count = np.maximum(n, 1).astype(np.uint64)
    with np.errstate(over="ignore"):
        threshold = (-count) % count
    bits = _mix(seed)
    bad = bits < threshold
    while np.any(bad):
        bits[bad] = _mix(bits[bad])
        bad = bits < threshold
    return (bits % count).astype(np.int64)


def lag_bin(gap):
    return np.searchsorted([8, 64, 512], gap, side="left")


class Matcher:
    """Sorted text/token/position buckets; draw uniformly after exclusions.

    No per-pair Python search or rejection through neighbours: counts and
    selected ranks come from interval arithmetic on sorted excluded positions.
    """
    def __init__(self, text, position, token):
        self.text = np.asarray(text)
        self.position = np.asarray(position)
        token = np.asarray(token)
        self.nodes = np.lexsort((self.position, token, self.text))
        self.coord = np.argsort(self.nodes)
        groups = np.r_[True, (np.diff(token[self.nodes]) != 0) | (np.diff(self.text[self.nodes]) != 0)]
        self.boundaries = np.r_[np.flatnonzero(groups), len(token)]
        self.bucket = (np.cumsum(groups) - 1)[self.coord]
        self.stride = int(self.position.max()) + 1
        self.search = np.repeat(np.arange(len(self.boundaries) - 1), np.diff(self.boundaries)) * self.stride + self.position[self.nodes]
        self.fixed = len(token) + 1
        _, inv, sizes = np.unique(token, return_inverse=True, return_counts=True)
        self.frequency = sizes[inv]
        self.docs, self.doc_index = np.unique(self.text, return_inverse=True)

    def batch(self, model, setting, q, j, block_row):
        """q and j are expanded pairs, grouped by block_row; each j occurs once per i."""
        if not len(j):
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8)
        b = self.bucket[j]
        t, p = self.text, self.position
        same = t[q] == t[j]
        start, stop = self.boundaries[b], self.boundaries[b + 1]
        base = b * self.stride
        bins = lag_bin(np.abs(p[q] - p[j]))
        edges = np.asarray([0, 8, 64, 512, self.stride + 512])
        lo, hi = edges[bins] + 1, edges[bins + 1]

        def find(pos):
            return np.searchsorted(self.search, base + np.clip(pos, 0, self.stride - 1), side="left")

        # The right edges are exclusive; at stride they are the bucket end.
        def right(pos):
            return np.where(pos >= self.stride, stop, find(pos))

        l0, r0 = np.where(same, find(p[q] - hi), start), np.where(same, right(p[q] - lo + 1), stop)
        l1, r1 = np.where(same, right(p[q] + lo), stop), np.where(same, right(p[q] + hi + 1), stop)
        qkey = block_row * self.fixed
        excluded = np.unique(np.r_[qkey + self.coord[j], qkey + self.coord[q]])
        el0, er0 = np.searchsorted(excluded, qkey + l0), np.searchsorted(excluded, qkey + r0)
        el1, er1 = np.searchsorted(excluded, qkey + l1), np.searchsorted(excluded, qkey + r1)
        n0, n1 = (r0 - l0) - (er0 - el0), (r1 - l1) - (er1 - el1)
        total = n0 + n1
        reason = np.select([stop - start <= 1, r0 - l0 + r1 - l1 <= 1, total == 0], [1, 2, 3], default=0).astype(np.int8)
        seeds = keyed_seed(model, setting, t[q], p[q], t[j], p[j])
        rank = _uniform_rank(seeds, total)
        use_first = rank < n0
        low, high = np.where(use_first, el0, el1), np.where(use_first, er0, er1)
        original = low.copy()
        index = np.where(use_first, l0 + rank, l1 + rank - n0)
        key = qkey + index
        while np.any(low < high):
            mid = (low + high) // 2
            advance = (low < high) & (excluded[np.minimum(mid, len(excluded) - 1)] - (mid - original) <= key)
            low = np.where(advance, mid + 1, low)
            high = np.where(advance, high, mid)
        control = np.full(len(j), -1, dtype=np.int32)
        matched = reason == 0
        control[matched] = self.nodes[(index + low - original)[matched]]
        return control, reason


class Predictions:
    def __init__(self, ids, probabilities, top1, mass):
        ids = np.asarray(ids, dtype=np.int64)
        self.ids = ids
        self.prob = np.asarray(probabilities, dtype=np.float64)
        self.top1 = np.asarray(top1)
        self.mass = np.asarray(mass, dtype=np.float64)
        self.stride = int(ids.max()) + 1
        order = np.argsort(ids, axis=1)
        self.keys = (np.take_along_axis(ids, order, axis=1) + np.arange(len(ids))[:, None] * self.stride).reshape(-1)
        self.sorted_prob = np.take_along_axis(self.prob, order, axis=1).reshape(-1)

    def overlap(self, q, j):
        # int64: pool indices arrive as int32 and j * stride overflows it past index 42,713 at full size, which
        # made the lookup miss and the overlap 0 for the last quarter of the pool (found by the audit, 2026-11-23).
        key = np.asarray(j, dtype=np.int64)[:, None] * self.stride + self.ids[q]
        at = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
        return (np.minimum(self.prob[q], self.sorted_prob[at]) * (self.keys[at] == key)).sum(1)


def load_predictions(root, model, pool):
    fields = ("top20_ids", "top20_probs", "top1_id", "top20_mass")
    parts = []
    for c in pool.texts:
        with np.load(Path(root) / "pred" / model / f"c{c:02d}.npz") as data:
            parts.append([data[key][pool.start:] for key in fields])
    return Predictions(*(np.concatenate(x, axis=0) for x in zip(*parts)))


def _expand(query, indices, indptr, ties, t_in, t_all):
    rows, cols = np.where(indices[query] >= 0)
    n = (indices[query] >= 0).sum(1) - t_in[query]
    strict = cols < n[rows]
    sizes = t_all[query]
    if np.any(indptr[query + 1] - indptr[query] != sizes):
        raise ValueError("stored Q tie IDs do not match recorded cutoff sizes")
    t_row = np.repeat(np.arange(len(query)), sizes)
    offsets = np.arange(len(t_row)) - np.repeat(np.r_[0, sizes.cumsum()[:-1]], sizes)
    tied_indices = ties[indptr[query[t_row]] + offsets]
    row = np.r_[rows[strict], t_row]
    j = np.r_[indices[query[rows[strict]], cols[strict]], tied_indices]
    w = np.r_[np.ones(strict.sum()), t_in[query[t_row]] / t_all[query[t_row]]]
    ordered = np.lexsort((j, row))
    return row[ordered], j[ordered], w[ordered]


def contributions(query, audit, indices, indptr, ties, t_in, t_all, matcher, pred, model, setting,
                  *, window=64, batch_size=64, record_all=False):
    """Return weighted document contributions, raw pair counts, and control witness arrays."""
    query = np.asarray(query, dtype=np.int64)
    sums = np.zeros((len(matcher.docs), len(STRATA), len(FIELDS)))
    raw = np.zeros((len(matcher.docs), len(STRATA), len(REASONS)), dtype=np.int64)
    records = []
    for start in range(0, len(query), batch_size):
        block = query[start:start + batch_size]
        row, j, weight = _expand(block, indices, indptr, ties, t_in, t_all)
        q = block[row]
        control, reason = matcher.batch(model, setting, q, j, row)
        same_text = matcher.text[q] == matcher.text[j]
        near = np.abs(matcher.position[q] - matcher.position[j]) <= window
        strata = 2 * (same_text.astype(np.int64) + (same_text & near)) + (matcher.frequency[j] >= 20)
        doc = matcher.doc_index[q]
        key = (doc, strata)
        np.add.at(raw, (*key, reason), 1)
        np.add.at(sums, (*key, F["matched"] + reason), weight)
        for field, ids in (("mass_query", q), ("mass_neighbour", j)):
            np.add.at(sums[..., F[field]], key, weight * pred.mass[ids])
        m = reason == 0
        mq, mj, mc, w = q[m], j[m], control[m], weight[m]
        mkey = (doc[m], strata[m])
        for field, value in (("advantage", pred.overlap(mq, mj) - pred.overlap(mq, mc)),
                             ("top1_neighbour", pred.top1[mq] == pred.top1[mj]),
                             ("top1_control", pred.top1[mq] == pred.top1[mc]),
                             ("mass_control", pred.mass[mc])):
            np.add.at(sums[..., F[field]], mkey, w * value)
        keep = np.ones(len(q), dtype=bool) if record_all else np.isin(q, audit)
        if np.any(keep):
            records.append((q[keep], j[keep], control[keep], reason[keep], weight[keep], strata[keep]))
    witness = {}
    for n, field in enumerate(("query", "neighbour", "control", "reason", "weight", "stratum")):
        witness[field] = np.concatenate([r[n] for r in records]) if records else np.empty(0)
    return sums, raw, witness


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan), where=denominator != 0)


def bootstrap_indices(n, draws=200, seed=BOOTSTRAP_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(draws, n))


def bootstrap_scores(by_doc, covered, samples=None):
    """Resample documents; retain the reference strata, not their bootstrap weights."""
    if samples is None:
        samples = bootstrap_indices(len(by_doc))
    if not np.any(covered):
        return np.full(len(samples), np.nan)
    total = by_doc[samples].sum(1)[:, covered, :]
    return _ratio(total[..., F["advantage"]], total[..., F["matched"]]).mean(1)


def _nullable(x):
    return float(x) if np.isfinite(x) else None


def summarize(by_doc, *, min_matched=MIN_MATCHED, samples=None):
    total = by_doc.sum(0)
    matched = total[:, F["matched"]]
    pairs = total[:, F["matched"]:F["all_candidates_excluded"] + 1].sum(1)
    ratio = _ratio(matched, pairs)
    covered = (matched >= min_matched) & (ratio >= .5) & (pairs > 0)
    absent = pairs == 0
    score = _ratio(total[:, F["advantage"]], matched)
    overall = float(matched.sum() / pairs.sum()) if pairs.sum() else 0.
    samples_scores = bootstrap_scores(by_doc, covered, samples)
    good = samples_scores[np.isfinite(samples_scores)]
    top_neighbour = _ratio(total[:, F["top1_neighbour"]], matched)
    top_control = _ratio(total[:, F["top1_control"]], matched)
    return dict(eligible=bool(np.any(covered) and np.all(covered | absent) and overall >= .5), coverage=overall,
                score=float(score[covered].mean()) if covered.any() else None,
                top1_neighbour=float(top_neighbour[covered].mean()) if covered.any() else None,
                top1_control=float(top_control[covered].mean()) if covered.any() else None,
                bootstrap_percentiles=np.percentile(good, [5, 50, 95]).tolist() if len(good) else None,
                bootstrap_valid=int(len(good)),
                strata={name: dict(status="absent" if absent[i] else "covered" if covered[i] else "uncovered",
                                   pairs=float(pairs[i]), coverage=_nullable(ratio[i]), mean_advantage=_nullable(score[i]),
                                   **{field: float(total[i, idx]) for field, idx in F.items()},
                                   top1_neighbour_rate=_nullable(top_neighbour[i]),
                                   top1_control_rate=_nullable(top_control[i]),
                                   mean_query_retained_mass=_nullable(total[i, F["mass_query"]] / pairs[i]) if pairs[i] else None,
                                   mean_neighbour_retained_mass=_nullable(total[i, F["mass_neighbour"]] / pairs[i]) if pairs[i] else None,
                                   mean_control_retained_mass=_nullable(total[i, F["mass_control"]] / matched[i]) if matched[i] else None)
                        for i, name in enumerate(STRATA)})


def readout_path(root, model, reading, centring, site, memory, mask):
    return Path(root) / "readout" / model / f"{reading}_{centring}" / input_name(site) / f"{memory}_{mask}.npz"


def run_readout(root, model, *, groups=PRIORITY, sites=SITES, min_matched=MIN_MATCHED, record_all=False,
                output_root=None):
    """Consume every completed model setting in groups/sites; no inference."""
    root = Path(root)
    results_root = Path(output_root) if output_root is not None else root
    probe_bytes = (root / "probe" / "probe.json").read_bytes()
    probe_sha256 = hashlib.sha256(probe_bytes).hexdigest()
    pool = pool_from_probe(json.loads(probe_bytes))
    matcher = Matcher(pool.text, pool.position, pool.ids)
    pred = load_predictions(root, model, pool)
    out, summaries = [], []
    for reading, centring in groups:
        base = results_root / "census" / model / f"{reading}_{centring}"
        if not (base / "group.json").exists():
            continue
        config = json.loads((base / "group.json").read_bytes())
        if config.get("probe_sha256") not in (None, probe_sha256):
            raise ValueError("census and readout use different probes")
        if config.get("lists_saved") is False:
            continue
        for site in sites:
            if input_name(site) not in config["inputs"]:
                continue
            for memory in config["memories"]:
                for mask in config["masks"]:
                    setting = (site, reading, centring, memory, mask)
                    path = setting_path(results_root, model, reading, centring, input_name(site), memory, mask)
                    if not path.is_file():
                        continue
                    with np.load(path) as data:
                        fields = ("indices", "tie_indptr", "tie_indices", "t_in", "t_all")
                        if not pool.query:
                            raise ValueError("prediction readout requires fixed Q")
                        sums, raw, witness = contributions(pool.query, pool.audit, *(data[k] for k in fields), matcher, pred,
                                                           model, setting, window=config["window"], record_all=record_all)
                    summary = summarize(sums, min_matched=min_matched)
                    summaries.append(dict(site=site, reading=reading, centring=centring, memory=memory, mask=mask, **summary))
                    dest = readout_path(results_root, model, reading, centring, site, memory, mask)
                    atomic(dest, np.savez, by_document=sums, raw_counts=raw, documents=matcher.docs,
                           fields=FIELDS, strata=STRATA, reasons=REASONS, min_matched=min_matched,
                           bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=200, model=model,
                           site=site, reading=reading, centring=centring, memory=memory, mask=mask,
                           seed_algorithm="sha256/splitmix64-v1", **witness)
                    write_json(dest.with_suffix(".json"), summary)
                    out.append(dest)
    write_json(results_root / "analysis" / "prediction" / f"{model}.json", dict(model=model, settings=summaries))
    return out


def attention_summary(path, texts):
    with np.load(path) as f:
        use = np.isin(f["context_ids"], texts)
        fields = ("sink_mass_sum", "distance_numerator_sum", "non_sink_mass_sum", "query_count")
        sink, num, mass, count = (f[k][use].sum((0, 1)) for k in fields)  # document, range
        layers = f["layer_ids"]
    sink_mean, distance = _ratio(sink, count), _ratio(num, mass)

    def report(x):
        return dict(heads=[_nullable(n) for n in x], quartiles=[_nullable(n) for n in np.nanpercentile(x, [25, 50, 75])])

    return [dict(layer=int(layer), sink=report(sink_mean[i]), distance=report(distance[i])) for i, layer in enumerate(layers)]


@torch.no_grad()
def autocorrelation(x, centring="C", max_lag=1024, width_block=128):
    y, _ = centre(x, centring, "cosine", "cpu")
    n, t, width = y.shape
    limit = min(t - 1, max_lag)
    count = n * (t - np.arange(1, limit + 1))
    # Zero pad for linear, not circular, autocorrelation. Sum powers over
    # channels before inverse FFT; chunk width to bound peak RSS.
    pad = 1 << (2 * t - 2).bit_length()
    power = y.new_zeros(pad // 2 + 1)
    for lo in range(0, width, width_block):
        f = torch.fft.rfft(y[:, :, lo:lo + width_block].transpose(1, 2), n=pad, dim=-1)
        power += (f.real.square() + f.imag.square()).sum((0, 1))
    corr = torch.fft.irfft(power, n=pad)[1:limit + 1].numpy() / count
    sums = y.sum(1)
    # Dot products of sums are exactly the sum over cross-text position pairs.
    between_count = n * (n - 1) * t * t
    background = float((sums.sum(0).square().sum() - sums.square().sum()) / between_count) if n > 1 else None
    half = None
    if limit and background is not None:
        below = np.flatnonzero(corr - background <= (corr[0] - background) / 2)
        half = int(below[0] + 1) if len(below) else None
    return dict(rho=corr.tolist(), within_count=count.tolist(), background=background, between_count=between_count,
                half_lag=half, censored=bool(limit and background is not None and half is None), max_lag=limit)


def run_timescales(root, model, *, sites=tuple(range(2, 17, 2))):
    root = Path(root)
    pool = pool_from_probe(load_probe(root))
    residual = {}
    for site in sites:
        x = load_site(root, model, pool, site)
        residual[str(site)] = {c: autocorrelation(x, c) for c in ("C", "O")}
    report = dict(model=model, attention=attention_summary(root / "attn" / f"{model}.npz", pool.texts),
                  residual=residual, censored_count={c: sum(row[c]["censored"] for row in residual.values()) for c in ("C", "O")})
    write_json(root / "analysis" / f"{model}_timescales.json", report)
    return report
