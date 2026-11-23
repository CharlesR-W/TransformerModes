"""Stage A viewer export. No model loads, recomputation of results, or network.

The manifest describes planar typed arrays. For each field in layout order,
round the byte offset up to the item size and read a C-order array of shape
(rows, columns), or (rows,) when columns is omitted. All multi-byte values
are little endian. Gzip is a file format here, not HTTP Content-Encoding.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

SITES = (4, 8, 12, 16)
MEMORIES = {"none": 0., "2": .5, "8": .875, "64": .984375, "whole": 1.}
GROUPS = (("cosine", "C"), ("cosine", "O"), ("dot", "C"), ("dot", "O"))
ACCOUNTINGS = (("C", "none"), ("C", "window"), ("O", "window"))
ORIGINS = ("other_texts", "same_text_far", "same_text_near")
DISPLAY_K = 8
MISSING = 65535
SCHEMA = "stage-a-viewer-v1"
DISTANCE_BY_READING = {
    "cosine": dict(quantity="cosine distance", units="dimensionless", similarity="1 - distance"),
    "dot": dict(quantity="half squared history-feature distance divided by the pool mean K(t,t)",
                units="dimensionless, scaled by mean K(t,t)"),
}


def _json_value(path):
    return json.loads(path.read_bytes())


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _key(group):
    return "_".join(group)


def _display_vocab(probe):
    lookup = {}
    for c in (*probe["contexts"], *probe.get("stitch_sources", [])):
        tokens, strings = c["token_ids"], c["token_strings"]
        if len(tokens) != len(strings) or any(not isinstance(s, str) for s in strings):
            raise ValueError("probe display strings do not align with IDs")
        for token, display in zip(tokens, strings):
            if lookup.setdefault(token, display) != display:
                raise ValueError(f"conflicting probe display strings for token {token}")
    return lookup


def _prediction_display(root, probe, predicted, vocab):
    """Supplement the probe from the already-cached pinned Pythia tokenizer."""
    known = {str(token): vocab[token] for token in sorted(predicted) if token in vocab}
    missing = sorted(predicted.difference(vocab))
    meta = dict(status="probe_only", extra_ids=0)
    if not missing:
        return known, meta
    pythia = probe.get("models", {}).get("pythia", {})
    if not pythia.get("id") or not pythia.get("revision"):
        return known, dict(meta, status="no_pinned_tokenizer_in_probe")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return known, dict(meta, status="transformers_unavailable")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            pythia["id"], revision=pythia["revision"], cache_dir=str(root / "hf"),
            local_files_only=True, trust_remote_code=False, use_fast=True, token=False)
    except OSError:
        return known, dict(meta, status="pinned_tokenizer_not_cached")
    replacement = probe.get("invalid_token_display", "�")
    for token in missing:
        if tokenizer.convert_ids_to_tokens(token) is None:
            continue  # A padded output-head ID need not exist in the tokenizer.
        display = tokenizer.decode([token], clean_up_tokenization_spaces=False)
        # Match probe.token_display: collapse any invalid UTF-8 fragment,
        # otherwise preserve the exact single-ID decode, including whitespace.
        known[str(token)] = replacement if "�" in display else display
    return known, dict(status="pinned_pythia_tokenizer", model_id=pythia["id"], revision=pythia["revision"],
                       extra_ids=len(known) - (len(predicted) - len(missing)))


class _Pack:
    def __init__(self, folder):
        self.folder = folder
        self.files = {}
        self.layouts = {
            "neighbours": [
                dict(name="index", dtype="<u2", columns=DISPLAY_K, units="zero-based index in named candidate pool",
                     missing=MISSING),
                dict(name="origin_counts", dtype="|u1", columns=3, units="selected neighbours",
                     column_labels=list(ORIGINS), share_denominator="same-row sum; undefined when zero"),
                dict(name="tie_flags", dtype="|u1", units="bit field",
                     bits={"0-3": "number of displayed ranks in tied cutoff (trailing valid ranks)",
                           "4": "cutoff has at least two tied candidates, even if not displayed",
                           "5": "cutoff tie has unselected candidates", "6-7": "reserved (zero)"}),
            ]
        }
        # Preserve the census float32 exactly, rather than rounding to the
        # float16 assumed by the spec's original bandwidth estimate.
        self.layouts["neighbours_with_distance"] = self.layouts["neighbours"] + [
            dict(name="distance", dtype="<f4", columns=DISPLAY_K, units="dimensionless distance (see setting reading)",
                 missing="NaN", aligned_to="index")]

    def _put(self, name, raw, encoding, dtype, shape, units, *, source=None):
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.compress(raw, compresslevel=3, mtime=0)
        path.write_bytes(data)
        row = dict(encoding=encoding, dtype=dtype, shape=shape, units=units, compression="gzip",
                   bytes=len(data), decoded_bytes=len(raw))
        if source is not None:
            row["source"] = source
        self.files[name] = row
        return name

    def json(self, name, value, *, units, source=None):
        shape = [len(value)] if isinstance(value, list) else []
        return self._put(name, _json_bytes(value), "json", "json", shape, units, source=source)

    def arrays(self, name, layout, data, *, units):
        n = len(next(iter(data.values())))
        raw = bytearray()
        for field in self.layouts[layout]:
            dtype = np.dtype(field["dtype"])
            value = np.asarray(data[field["name"]], dtype=dtype, order="C")
            expected = (n, field["columns"]) if "columns" in field else (n,)
            if value.shape != expected:
                raise ValueError(f"{name}: layout expects {expected}, found {value.shape}")
            raw.extend(b"\0" * (-len(raw) % dtype.itemsize))
            raw.extend(value.tobytes())
        return self._put(name, raw, "typed_arrays", layout, [n], units)


def _pool(root, dataset, ranges, contexts):
    """Verify the NPZ's index space against the actual probe, before packing."""
    file = root / "census" / dataset / "pool.npz"
    slices = {}
    offset = 0
    with np.load(file, allow_pickle=False) as f:
        text, pos, token = (f[k] for k in ("text", "position", "token"))
    for c, start, stop in ranges:
        if c in slices or not 1 <= start < stop <= len(contexts[c]["token_ids"]) + 1:
            raise ValueError("invalid census ranges in probe")
        end = offset + stop - start
        sl = slice(offset, end)
        if not (len(text[sl]) == stop - start and np.all(text[sl] == c)
                and np.array_equal(pos[sl], np.arange(start, stop))
                and np.array_equal(token[sl], contexts[c]["token_ids"][start - 1:stop - 1])):
            raise ValueError(f"{file}: census indices disagree with probe")
        slices[c] = sl
        offset = end
    if offset != len(text) or len(pos) != offset or len(token) != offset or offset > MISSING:
        raise ValueError(f"{file}: pool cannot be represented by uint16 with a missing sentinel")
    return slices, offset


def _predictions(root, pack, model, c, length, predicted):
    source = root / "pred" / model / f"c{c:02d}.npz"
    if not source.is_file():
        return None
    with np.load(source, allow_pickle=False) as f:
        ids = f["top20_ids"][1:, :3]
        probs = f["top20_probs"][1:, :3]
    if (ids.ndim != 2 or probs.shape != ids.shape or len(ids) != length or not 1 <= ids.shape[1] <= 3
            or ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids > MISSING))
            or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1))):
        raise ValueError(f"invalid prediction arrays in {source}")
    if not np.array_equal(probs, probs.astype("<f4")):
        raise ValueError(f"{source}: predictions are not losslessly representable as float32")
    predicted.update(int(v) for v in np.unique(ids))
    n = ids.shape[1]
    layout = f"predictions_top{n}"
    if layout not in pack.layouts:
        pack.layouts[layout] = [dict(name="token_id", dtype="<u2", columns=n, units="model vocabulary ID"),
                                dict(name="probability", dtype="<f4", columns=n, units="full-softmax probability")]
    path = pack.arrays(f"predictions/{model}/c{c:02d}.bin.gz", layout,
                       dict(token_id=ids, probability=probs), units="one row per position starting at 1")
    pack.files[path]["source"] = source.relative_to(root).as_posix()
    return path


def _neighbour_planes(file, n):
    with np.load(file, allow_pickle=False) as f:
        top, origins, t_in, t_all = (f[k] for k in ("indices", "origins", "t_in", "t_all"))
        distance = f["display_distances"] if "display_distances" in f else None
    if (top.ndim != 2 or top.shape[0] != n or not 1 <= top.shape[1] <= 64 or origins.shape != (n, 3)
            or t_in.shape != (n,) or t_all.shape != (n,)
            or any(v.dtype.kind not in "iu" for v in (top, origins, t_in, t_all))
            or np.any((top < -1) | (top >= n)) or np.any((origins < 0) | (origins > 64))):
        raise ValueError(f"invalid census arrays in {file}")
    selected = (top >= 0).sum(1)
    if (np.any(selected != origins.sum(1)) or np.any((top >= 0) != (np.arange(top.shape[1]) < selected[:, None]))
            or np.any((t_in < 0) | (t_in > selected) | (t_in > t_all) | (t_all > n))
            or np.any((t_in == 0) != (t_all == 0))):
        raise ValueError(f"inconsistent cutoff or origin counts in {file}")
    shown = np.minimum(selected, DISPLAY_K)
    tied_shown = np.maximum(0, shown - (selected - t_in))
    flags = np.where(t_all > 1, tied_shown + 16 + 32 * (t_all > t_in), 0)
    indices = np.full((n, DISPLAY_K), MISSING, dtype="<u2")
    prefix = top[:, :DISPLAY_K]
    indices[:, :prefix.shape[1]] = np.where(prefix >= 0, prefix, MISSING)
    planes = dict(index=indices, origin_counts=origins, tie_flags=flags)
    if distance is not None:
        valid = indices != MISSING
        if (distance.dtype.kind != "f" or distance.dtype.itemsize != 4 or distance.shape != indices.shape
                or not np.isfinite(distance[valid]).all() or not np.isnan(distance[~valid]).all()):
            raise ValueError(f"{file}: invalid display distances or padding")
        planes["distance"] = distance
    return planes


def _viewer(root, pack, probe, contexts, models, approved, probe_sha256):
    """Yield choices only for actual saved Stage A cosine settings."""
    node_sets = probe["node_sets"]
    corpus = node_sets["census_pool"]
    pools = {"corpus": [list(r) for r in corpus]}
    stitch_ranges = {c: node_sets["one_text"][str(c)] for c, row in contexts.items() if row["kind"] == "stitched"}
    for c, ranges in stitch_ranges.items():
        if len(ranges) != 1 or ranges[0][0] != c:
            raise ValueError("a stitch must be its own candidate pool")
        pools[f"stitch{c:02d}"] = [list(r) for r in ranges]

    choices = []
    for model in models:
        all_pools = [(model, "corpus", corpus), *((f"{model}_stitch{c:02d}", f"stitch{c:02d}", ranges)
                                                  for c, ranges in stitch_ranges.items())]
        for dataset, pool, ranges in all_pools:
            base = root / "census" / dataset
            validated = None
            for centring, mask in ACCOUNTINGS:
                if ("cosine", centring) not in approved or (pool != "corpus" and centring != "C"):
                    continue
                folder = base / f"cosine_{centring}"
                group_json = folder / "group.json"
                if not group_json.is_file():
                    continue
                group = _json_value(group_json)
                if group.get("probe_sha256") not in (None, probe_sha256):
                    raise ValueError(f"{group_json}: census and probe hashes differ")
                if group.get("lists_saved") is False or mask not in group["masks"]:
                    continue
                for site in SITES:
                    name = f"site{site:02d}"
                    if name not in group["inputs"]:
                        continue
                    for memory, alpha in MEMORIES.items():
                        file = folder / name / f"{memory}_{mask}.npz"
                        if memory not in group["memories"] or not file.is_file():
                            continue
                        if validated is None:
                            validated = _pool(root, dataset, ranges, contexts)
                        slices, n = validated
                        planes = _neighbour_planes(file, n)
                        available = "distance" in planes
                        layout = "neighbours_with_distance" if available else "neighbours"
                        distance_meta = dict(DISTANCE_BY_READING["cosine"], available=available)
                        if not available:
                            distance_meta["reason"] = "rank distances not stored in census"
                        files = {}
                        for c, sl in slices.items():
                            path = f"neighbours/{model}/c{c:02d}/cosine_{centring}/{name}/{memory}_{mask}.bin.gz"
                            files[str(c)] = pack.arrays(path, layout, {k: v[sl] for k, v in planes.items()},
                                                        units="one row per census token in increasing position")
                        choices.append(dict(model=model, site=site, reading="cosine", centring=centring, memory=memory,
                                            alpha=alpha, mask=mask, candidate_pool=pool, files=files, distance=distance_meta,
                                            source=file.relative_to(root).as_posix()))
    return pools, choices


def _figure_geometry(root, models, contexts, approved):
    base = root / "analysis" / "geometry"
    kinds = {model: ("pool", model, None) for model in models}
    kinds.update({f"{m}_stitch{c:02d}": ("stitched", m, c) for m in models for c, row in contexts.items()
                  if row["kind"] == "stitched"})
    if base.is_dir():
        for p in sorted(base.iterdir()):
            if p.is_dir() and re.fullmatch(r"synthetic_(strong|planted|drift|noise_a|noise_b)_seed\d+", p.name):
                kinds[p.name] = ("synthetic", None, None)
    result = []
    for dataset, (kind, model, text) in kinds.items():
        for reading, centring in GROUPS:
            if (reading, centring) not in approved:
                continue
            path = base / dataset / f"{reading}_{centring}.json"
            if path.is_file():
                result.append(dict(dataset=dataset, kind=kind, model=model, text=text, reading=reading, centring=centring,
                                   source=path.relative_to(root).as_posix(), rows=_json_value(path)))
    return result


def _selection(root, approved):
    path = root / "analysis" / "stage_b_selection.json"
    if not path.is_file():
        return None, None, []
    data = _json_value(path)
    for key in ("M", "L", "R", "X"):
        choice = data.get(key)
        if choice is not None and (choice["reading"], choice["centring"]) not in approved:
            raise ValueError(f"stored selection {key} refers to a group excluded by the Stage A plan")
    table, source_rows = [], []
    for i, value in enumerate(data["table"]):
        key = value["construction"]
        in_plan = (key["reading"], key["centring"]) in approved
        # Retain the prespecified candidate table's empty placeholders, but
        # do not publish experimental numbers from stale, unapproved groups.
        has_sites = any(s is not None for m in value["models"].values() for s in m["sites"].values())
        if in_plan or not (has_sites or value["passed"]):
            table.append(value)
            source_rows.append(i)
    return path.relative_to(root).as_posix(), dict(data, table=table), source_rows


def _figures(root, pack, models, contexts, approved):
    geometry = _figure_geometry(root, models, contexts, approved)
    timing, prediction = [], []
    for model in models:
        path = root / "analysis" / f"{model}_timescales.json"
        if path.is_file():
            timing.append(dict(model=model, source=path.relative_to(root).as_posix(), data=_json_value(path)))
        path = root / "analysis" / "prediction" / f"{model}.json"
        if path.is_file():
            all_rows = _json_value(path)["settings"]
            valid = [(i, row) for i, row in enumerate(all_rows) if (row["reading"], row["centring"]) in approved]
            prediction.append(dict(model=model, source=path.relative_to(root).as_posix(),
                                   source_rows=[i for i, _ in valid], rows=[r for _, r in valid]))
    lead_whiskers = []
    for group in geometry:
        if group["kind"] != "pool" or group["reading"] != "cosine":
            continue
        for i, row in enumerate(group["rows"]):
            bs = row.get("origin_document_bootstrap")
            if bs is not None and row["input"] == "site08" and row["mask"] in ("none", "window"):
                lead_whiskers.append(dict(model=group["model"], reading=group["reading"], centring=group["centring"],
                                           memory=row["memory"], mask=row["mask"], source=group["source"], source_row=i,
                                           p5=None if bs["p5"] is None else bs["p5"][0],
                                           p95=None if bs["p95"] is None else bs["p95"][0],
                                           seed=bs["seed"], draws=bs["draws"], valid_draws=bs["valid_draws"]))
    e1 = dict(lead_selector=dict(kind="pool", input="site08", reading="cosine", masks=["none", "window"]),
              origin_order=list(ORIGINS), overlap_reference_order=["local", "random", "embedding"],
              screen_status_order=["calibrated", "too few eligible", "tied minima", "unbracketed"],
              geometry=geometry, timescales=timing, other_text_share_document_bootstrap=lead_whiskers or None)
    e2 = dict(prediction=prediction)
    source, selection, source_rows = _selection(root, approved)
    lead = []
    if selection is not None:
        for role in ("M", "L", "R"):
            use = selection.get(role)
            if use is None:
                continue
            for group in geometry:
                if group["kind"] != "pool" or any(group[k] != use[k] for k in ("reading", "centring")):
                    continue
                for i, row in enumerate(group["rows"]):
                    if (row["input"] == "site08" and row["mask"] == use["mask"] and row["memory"] == use["memory"]
                            and "screen_origin" in row):
                        lead.append(dict(role=role, model=group["model"], reading=use["reading"], centring=use["centring"],
                                         source=group["source"], source_row=i, row=row))
    e3 = dict(source=source, selection=selection, source_rows=source_rows, suffix_lead=lead)
    return {name: pack.json(f"figures/{name}.json.gz", data, units="pipeline summaries with source paths")
            for name, data in (("e1", e1), ("e2", e2), ("e3", e3))}


def run(root, out=None):
    """Create out/manifest.json and one self-contained generation of data files.

    Default out is root/site. Returns the in-memory manifest. The separate
    site build must still require a passing audit_a/report.json before release.
    """
    root = Path(root)
    out = Path(out) if out is not None else root / "site"
    probe_bytes = (root / "probe" / "probe.json").read_bytes()
    probe_sha256 = hashlib.sha256(probe_bytes).hexdigest()
    probe = json.loads(probe_bytes)
    contexts = {r["id"]: r for r in probe["contexts"]}
    if len(contexts) != len(probe["contexts"]) or any(not isinstance(c, int) or c < 0 for c in contexts):
        raise ValueError("probe context IDs must be unique nonnegative integers")
    models = list(probe.get("models", {})) or sorted(p.name for p in (root / "pred").iterdir() if p.is_dir())
    if not models or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", m) for m in models):
        raise ValueError("invalid or absent model IDs")
    vocab = _display_vocab(probe)
    plan_path = root / "analysis" / "stage_a_plan.json"
    plan = _json_value(plan_path) if plan_path.is_file() else None
    if plan is not None and plan.get("probe_sha256") not in (None, probe_sha256):
        raise ValueError("Stage A plan and probe hashes differ")
    approved = frozenset(map(tuple, plan["approved_groups"])) if plan is not None else frozenset(GROUPS)
    if not approved.issubset(GROUPS):
        raise ValueError("Stage A plan contains an unknown reading/centring group")

    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    prior = _json_value(manifest_path) if manifest_path.is_file() else None
    if prior is not None and (not isinstance(prior, dict) or prior.get("schema") != SCHEMA):
        raise ValueError(f"refusing to replace an unrelated manifest at {manifest_path}")
    folder = Path(tempfile.mkdtemp(prefix="stage-a-data-", dir=out))
    try:
        pack = _Pack(folder)
        texts = []
        predicted = set()
        for c, row in contexts.items():
            payload = {k: row[k] for k in ("id", "kind", "token_ids", "token_strings", "text", "subset", "file_index",
                                            "text_sha256", "schedule", "segments") if k in row}
            length = len(row["token_ids"])
            payload["position_start"] = 1
            source = "probe/probe.json"
            text_file = pack.json(f"texts/c{c:02d}.json.gz", payload, units="probe text and tokens, positions from 1", source=source)
            preds = {model: _predictions(root, pack, model, c, length, predicted) for model in models}
            texts.append(dict(id=c, kind=row["kind"], positions=[1, length + 1], text_file=text_file,
                              predictions={k: v for k, v in preds.items() if v is not None}))
        known, tokenizer_info = _prediction_display(root, probe, predicted, vocab)
        vocabulary = pack.json("texts/prediction_display.json.gz", known, units="model vocabulary ID to display string",
                               source="probe/probe.json")
        pools, choices = _viewer(root, pack, probe, contexts, models, approved, probe_sha256)
        figures = _figures(root, pack, models, contexts, approved)
        distance_meta = dict(DISTANCE_BY_READING["cosine"], unit="cosine distance", by_reading=DISTANCE_BY_READING,
                             order="strict neighbours by distance then (text, position); the selected cutoff group by (text, position)",
                             available=any(c["distance"]["available"] for c in choices),
                             complete=bool(choices) and all(c["distance"]["available"] for c in choices))
        if not distance_meta["available"]:
            distance_meta["reason"] = "rank distances not stored in census"
        descriptor = dict(encoding="json", dtype="json", shape=[], units="manifest", compression="none")
        manifest = dict(schema=SCHEMA, data_directory=folder.name, manifest_file=dict(path="manifest.json", **descriptor),
                        relative_paths="asset paths are relative to data_directory; source paths are relative to experiment root",
                        byte_order="little", field_alignment="align each plane to dtype itemsize; C-order, no final padding",
                        probe_sha256=probe_sha256, models=models, model_metadata=probe.get("models", {}),
                        corpus=probe.get("corpus"), description=probe.get("description"),
                        plan_source=plan_path.relative_to(root).as_posix() if plan_path.is_file() else None,
                        approved_groups=[list(g) for g in GROUPS if g in approved],
                        dropped_groups=[list(g) for g in GROUPS if g not in approved],
                        viewer_sites=list(SITES), viewer_k=DISPLAY_K, origins=list(ORIGINS),
                        candidate_pools=pools, candidate_ranges="[text, start inclusive, stop exclusive]; flatten in listed order",
                        texts=texts, settings=choices, layouts=pack.layouts, files=pack.files, figures=figures,
                        token_card=dict(history_tokens=24, history_ends_at="the selected or neighbour position (inclusive)",
                                        finite_memory_weight="1 - alpha**24", whole_prefix_weight="min(24, position)/position",
                                        neighbour_distance=distance_meta,
                                        prediction_display=dict(file=vocabulary, found_ids=len(known),
                                                                missing_ids=len(predicted) - len(known), fallback="numeric vocabulary ID",
                                                                tokenizer=tokenizer_info)),
                        removed_short_range_mass=dict(scope="suffix query sample aggregate only", figure="e1", field="screen_removed"),
                        required_before_release="audit_a/report.json must attest a passing independent audit",
                        asset_bytes=sum(f["bytes"] for f in pack.files.values()),
                        decoded_asset_bytes=sum(f["decoded_bytes"] for f in pack.files.values()))
        # The data directory is immutable once referenced; publish one pointer
        # last so a failed pack leaves any earlier manifest usable.
        with tempfile.NamedTemporaryFile(dir=out, delete=False) as stream:
            pending = Path(stream.name)
            try:
                stream.write(_json_bytes(manifest))
                stream.close()
                pending.replace(manifest_path)
            finally:
                pending.unlink(missing_ok=True)
    except BaseException:
        shutil.rmtree(folder)
        raise
    if prior is not None:
        old = prior.get("data_directory", "")
        if re.fullmatch(r"stage-a-data-[a-z0-9_]+", old) and old != folder.name:
            shutil.rmtree(out / old, ignore_errors=True)
    return manifest
