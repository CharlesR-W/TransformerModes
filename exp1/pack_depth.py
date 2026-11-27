"""Site data for the "Depth embedding" tab (SPEC_DEPTH.md section 8), from the release written by exp1/depth.py.

    TMPDIR=/var/tmp uv run --with numpy python exp1/pack_depth.py                 # pack every complete unit
    TMPDIR=/var/tmp uv run --with numpy python exp1/pack_depth.py --release       # refuse unless everything is there

Reads ROOT (default /data/tm_depth: manifest.json, units/<model>/<set>/, and the aggregates cross.json,
permutations.json, gaussian.json, refit_agreement.json when they exist), the probe (token strings) and the synthetic
inputs (ids of r0..r4 and s0..s3). NumPy only; nothing is recomputed: every number written is read from the release.

A unit is packed when its done.json exists and every file read hashes to the value done.json lists (each file is
read once into memory, verified, and parsed from those bytes, so a unit being rewritten while this runs is skipped,
not half read; a unit's files join the pack only when all of them are written). Only the documented layout is read.
A unit whose done.json code or spec hash differs from the current exp1/depth.py, depth_dm.py, spacetime.py and
SPEC_DEPTH.md is packed but flagged stale (depth.py's own release() would ignore it); --release refuses it. Absent,
unfinished, skipped and stale units are listed by name on stdout and in meta.json. Run again as units complete.

Output OUT (default ROOT/site_depth), replaced as a whole only after the size check passes:

  meta.json      schema, layers, modes, models, strengths, sets, panels, paths, per-unit status, missing units,
                 aggregates present, quantisation rule, and the inventory {path: {bytes, sha256}} of every other file
  facts.json     {name: {value, text, source}}: every number the tab's text quotes (site/build_site.py turns
                 {{fact:x.depth.<name>}} into its text)
  rows.json      context keys, construction-set and panel rows as [context index, position], cohorts per set and panel
  texts.json     token strings by id, and for every context its kind, Pile subset, ids, and (shuffled) source map
  aggregates.json  cross-model, ladder, permutation, Gaussian and refit summaries, or null for each not yet computed
  u/<model>/<set>/<strength>.json   per-mode facts (eigenvalue, band, withheld, depth weights, token support,
                 document share, depth cosines, extreme tokens) and the int8 files with their scales
  u/<model>/<set>/<strength>/set.bin, panel-<kind>.bin, path-<context>.bin

Each .bin is int8, C order, shape (17, rows, 16) = (layer, token, mode) in manifest row order, modes 1..16 of the
release's float32 coordinates, with one scale per (layer, mode), stored as scales[layer][mode]:
  set file:          scale[l][k] = the largest finite |value| of mode k at layer l over the set's own rows;
  panel, path file:  scale[l][k] = the larger of the set file's scale[l][k] and the file's own largest finite |value|
                     at (l, k), so a placed token is never clipped.
Every scale is the |value| of a float32 entry, so it is a float32 number, stored exactly. Square-root companding:
  q = sign(x) * round-half-even(127 * sqrt(|x| / scale)), clipped to [-127, 127], computed in float64 from the float32
      x in this order (|x| / scale, then sqrt, then * 127, then round); where scale is 0, q = 0;
  a withheld (NaN) coordinate stores -128;
  decoding: x = sign(q) * (q / 127)^2 * scale.

The bytes written are measured; the pack refuses above 150 MB (SPEC_DEPTH.md section 8; 1 MB = 10^6 bytes).
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import io
import json
import os
import tempfile
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / "SPEC_DEPTH.md"
CODE = ("depth.py", "depth_dm.py", "spacetime.py")
SCHEMA = "depth-site-v1"
SITE_MODES = 16                # modes shipped per embedding, panel and path (SPEC_DEPTH.md section 8)
CONTEXT = 8                    # tokens of context shown on each side of a hovered token (section 8)
CAP_BYTES = 150_000_000        # the size cap (section 8)
QUANT_STEPS = 127              # int8 levels on each side of zero (section 8's rule)
AXIS_QUANTILE = 0.99           # the tab's default axis limits hold this share of each mode's coordinates (site/js/depth.js)
TEXT_COLOURS = 8               # categorical colours the tab uses at a time (site/css/depth.css --dp-cat-1..8)
MODELS = (("pythia", "Pythia-1b"), ("olmo", "OLMo-1B"))
STRENGTHS = (("tstar", "t*"), ("tstar_div16", "t*/16"))
SETS = ("doc00", "doc01", "doc02", "doc04", "pairA", "pairB", "d04", "d08", "d16", "d32", "rand")
PANELS = ("natural", "random", "shuffled")
COHORTS = ("in_set", "seen_sequence_new_position", "unseen_sequence")
EXTREME_LAYERS = ("0", "8", "16")
AGGREGATES = ("cross", "permutations", "gaussian", "refit_agreement")


class PackError(RuntimeError):
    pass


def log(*a):
    print("[pack_depth]", *a, flush=True)


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def json_bytes(obj):
    """depth.py's canonical JSON (for hash comparisons with its done.json)."""
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash():
    return hashlib.sha256(json_bytes({p: file_hash(HERE / p) for p in CODE})).hexdigest()


def write_json(path, obj, compact=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":") if compact else None,
                      indent=None if compact else 1)
    path.write_text(text + "\n", encoding="utf-8")


def finite(x):
    """JSON-safe: NumPy -> Python; non-finite -> None."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    if isinstance(x, np.ndarray):
        return finite(x.tolist())
    if isinstance(x, np.generic):
        return finite(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ------------------------------------------------------------------ code parameters quoted by the page
def code_parameters():
    """Parameters the primer quotes, read from the pipeline's source (never typed here)."""
    dm_src = (HERE / "depth_dm.py").read_text(encoding="utf-8")
    knn = None
    for node in ast.walk(ast.parse(dm_src)):
        if isinstance(node, ast.FunctionDef) and node.name == "radius_kernel":
            names = [a.arg for a in node.args.args]
            defaults = dict(zip(names[-len(node.args.defaults):], node.args.defaults))
            knn = ast.literal_eval(defaults["k"])
    src = (HERE / "depth.py").read_text(encoding="utf-8")
    tie = re.search(r"np\.diff\(vals\) < ([0-9.e-]+) \* vals\[:-1\]", src)
    cond = re.search(r"\(cond > ([0-9.e+]+)\)", src)
    if knn is None or not tie or not cond:
        raise PackError("could not read the neighbour count, tie rule or condition limit from the pipeline source")
    return dict(knn=int(knn), tie=float(tie[1]), condition=float(cond[1]))


# ------------------------------------------------------------------ quantisation (the stated rule; the audit re-encodes)
def quantise(a, floor=None):
    """float32 (L, M, >=16) -> (int8 (L, M, 16), scales (L, 16) as lists, |q| = 127 count). The module docstring
    states the rule; floor is the set file's scales (L, 16), which a panel or path file's scales may not go below."""
    v = np.asarray(a[:, :, :SITE_MODES], dtype=np.float32).astype(np.float64)
    ok = np.isfinite(v)
    mag = np.where(ok, np.abs(v), 0.0)
    scales = mag.max(axis=1) if v.shape[1] else np.zeros((v.shape[0], v.shape[2]))
    if floor is not None:
        scales = np.maximum(scales, np.asarray(floor, dtype=np.float64))
    s = np.broadcast_to(scales[:, None, :], v.shape)
    live = ok & (s > 0)
    u = np.sqrt(np.divide(mag, s, out=np.zeros_like(v), where=live))
    code = np.clip(np.sign(v) * np.rint(QUANT_STEPS * u), -QUANT_STEPS, QUANT_STEPS)
    q = np.where(ok, np.where(live, code, 0.0), -128).astype(np.int8)
    # every scale is the |value| of a float32 entry, so the float64 here is that float32 exactly
    assert np.all(scales.astype(np.float32).astype(np.float64) == scales)
    return np.ascontiguousarray(q), scales.tolist(), int((np.abs(q.astype(np.int16)) == QUANT_STEPS).sum())


# ------------------------------------------------------------------ inputs: manifest, probe, synthetic ids
class Inputs:
    def __init__(self, root: Path, probe_path: Path, inputs_path: Path):
        self.root = root
        self.manifest = read_json(root / "manifest.json")
        self.probe = read_json(probe_path)
        self.synth = read_json(inputs_path) if inputs_path.is_file() else None
        m = self.manifest
        self.contexts = list(m["contexts"])
        self.cindex = {c: i for i, c in enumerate(self.contexts)}
        vocab, ids = {}, {}
        for i, c in enumerate(self.probe["contexts"][:32]):
            key = f"c{i:02}"
            if m["contexts"][key]["context"] != i:
                raise PackError(f"manifest context {key} is not probe context {i}")
            ids[key] = list(c["token_ids"])
            for t, s in zip(c["token_ids"], c["token_strings"]):
                if vocab.setdefault(t, s) != s:
                    raise PackError(f"token id {t} has two strings in the probe")
        for key in self.contexts:
            if key in ids:
                continue
            if self.synth is None:
                ids[key] = None
                continue
            seq = list(self.synth["contexts"][key]["ids"])
            missing = sorted(set(seq) - vocab.keys())
            if missing:
                raise PackError(f"{key}: ids without a string in the probe: {missing[:5]}")
            ids[key] = seq
        self.ids, self.vocab = ids, vocab
        # every manifest row's token id must be the text's id at that position
        def check(rows, where):
            for ctx, pos, tid in rows:
                seq = ids.get(ctx if isinstance(ctx, str) else f"c{ctx:02}")
                if seq is not None and tid is not None and seq[pos - 1] != tid:
                    raise PackError(f"{where}: row ({ctx}, {pos}) has id {tid}, the text has {seq[pos - 1]}")
        for name, rows in m["sets"].items():
            check(rows, f"set {name}")
        for name, rows in m["panels"].items():
            check(rows, f"panel {name}")
        for name, rows in m["paths"].items():
            check(rows, f"path {name}")

    def key(self, ctx):
        return ctx if isinstance(ctx, str) else f"c{ctx:02}"

    def rows_compact(self, rows):
        return [[self.cindex[self.key(c)], int(p)] for c, p, _ in rows]


# ------------------------------------------------------------------ one unit
class Stale(Exception):
    pass


class Unit:
    def __init__(self, root: Path, model: str, name: str, current_code: str, current_spec: str):
        self.dir = root / "units" / model / name
        self.model, self.name = model, name
        done = self.dir / "done.json"
        if not done.is_file():
            raise FileNotFoundError("no done.json (not finished or not started)")
        self.done = read_json(done)
        # A unit built by other pipeline code or spec is still packed (the layout is what is read), but flagged;
        # --release and the site's release build refuse it.
        self.stale = [w for w, ok in (("pipeline code (exp1/depth.py, depth_dm.py, spacetime.py)", self.done.get("code_sha256") == current_code),
                                      ("SPEC_DEPTH.md", self.done.get("spec_sha256") == current_spec)) if not ok]
        if hashlib.sha256(json_bytes(self.done["files"])).hexdigest() != self.done.get("outputs_sha256"):
            raise Stale("done.json's file list does not match its own outputs hash")
        self.read = {}

    def bytes(self, rel):
        """A file's bytes, read once and verified against done.json; everything packed is parsed from these bytes,
        so a unit that is replaced while packing is detected rather than half read."""
        listed = self.done["files"].get(rel)
        if listed is None:
            raise Stale(f"{rel} is not among the unit's finished outputs")
        try:
            data = (self.dir / rel).read_bytes()
        except FileNotFoundError as e:
            raise Stale(f"{rel} disappeared while packing (the unit is being rebuilt)") from e
        h = hashlib.sha256(data).hexdigest()
        if h != listed:
            raise Stale(f"{rel} changed since done.json was written (the unit is being rebuilt)")
        self.read[rel] = h
        return data

    def has(self, rel):
        return rel in self.done["files"]

    def json(self, rel):
        return json.loads(self.bytes(rel))

    def npy(self, rel):
        return np.load(io.BytesIO(self.bytes(rel)), allow_pickle=False)

    def npz(self, rel):
        with np.load(io.BytesIO(self.bytes(rel)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


def band_of(bands, k):
    for b in bands:
        if b["first"] <= k <= b["last"]:
            return [int(b["first"]), int(b["last"]), bool(b.get("unresolved", False))]
    return None


def strength_record(facts, strength_label, tstar):
    f = facts["fields"]
    ext = facts["extension"]
    weight = np.asarray(f["depth_weight"], dtype=np.float64)          # (17, 64)
    weak = np.asarray(f["weak_slice"], dtype=bool)
    share = f.get("document_share")
    modes = []
    for j in range(SITE_MODES):
        k = j + 1
        modes.append(dict(
            k=k, eigenvalue=facts["eigenvalues"][j], band=band_of(facts["bands"], k),
            withheld=bool(ext["withheld"][j]), condition=ext["condition"][j],
            depth_weight=weight[:, j].tolist(), weak_slice=weak[:, j].tolist(),
            token_support=f["token_support"][j], document_share=None if share is None else share[j],
            adjacent_cos_mean=f["adjacent_cos_mean"][j], endpoint_cos=f["endpoint_cos"][j],
            extremes={l: {"low": f["extremes"][l]["low"][j], "high": f["extremes"][l]["high"][j]} for l in EXTREME_LAYERS}))
    return finite(dict(
        strength_label=strength_label, t=facts["t"], tstar=tstar, max_residual=facts["max_residual"],
        eigenvalues=facts["eigenvalues"][:SITE_MODES], gaps=facts["gaps"], cuts_band=facts["cuts_band"],
        bands=facts["bands"], modes_total=len(facts["eigenvalues"]), modes=modes))


def pack_unit(unit: Unit, inp: Inputs, out: Path, plan):
    """Write one unit's files into out; return (status record, list of written relative paths, saturated count)."""
    m = inp.manifest
    meta = unit.json("unit.json")
    if meta.get("set") != unit.name or meta.get("model") != unit.model or meta.get("refit"):
        raise PackError(f"{unit.dir}: unit.json names {meta.get('model')}/{meta.get('set')}")
    rows = m["sets"][unit.name]
    keys = {(inp.key(c), p): i for i, (c, p, _) in enumerate(rows)}
    if meta["n"] != len(rows):
        raise PackError(f"{unit.dir}: unit.json n = {meta['n']}, manifest set has {len(rows)} rows")
    panels = [k for k in PANELS if unit.has(f"panels/{k}.npz")]
    paths = [k for k in m["paths"] if unit.has(f"paths/{k}.npz")]
    written, saturated, records, files = [], 0, {}, {s: {} for s, _ in STRENGTHS}
    base = Path("u") / unit.model / unit.name

    def emit(strength, tag, arr, rel_rows, floor=None):
        nonlocal saturated
        q, scales, sat = quantise(arr, floor)
        saturated += sat
        rel = base / strength / f"{tag}.bin"
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        q.tofile(out / rel)
        written.append(rel.as_posix())
        files[strength][tag] = dict(path=rel.as_posix(), shape=list(q.shape), scales=scales, rows=rel_rows,
                                    withheld_modes=[j + 1 for j in range(q.shape[2]) if np.all(q[:, :, j] == -128)])

    for strength, label in STRENGTHS:
        facts = unit.json(f"{strength}/facts.json")
        modes = unit.npy(f"{strength}/modes.npy")
        if modes.shape[0] != plan["layers"] or modes.shape[1] != len(rows) or modes.shape[2] < SITE_MODES:
            raise PackError(f"{unit.dir}/{strength}/modes.npy has shape {modes.shape}")
        if not np.all(np.isfinite(modes[:, :, :SITE_MODES])):
            raise PackError(f"{unit.dir}/{strength}/modes.npy has non-finite construction coordinates")
        records[strength] = strength_record(facts, label, meta["tstar"])
        emit(strength, "set", modes, len(rows))
        del modes
    for collection, names in (("panels", panels), ("paths", paths)):
        for name in names:
            z = unit.npz(f"{collection}/{name}.npz")
            idx = z["in_set_index"]
            prow = m[collection][name]
            expect = np.array([keys.get((inp.key(c), p), -1) for c, p, _ in prow])
            if not np.array_equal(idx, expect) or any(z[s].shape[:2] != (plan["layers"], len(prow)) for s, _ in STRENGTHS):
                raise PackError(f"{unit.dir}/{collection}/{name}.npz does not match the manifest rows")
            if collection == "panels" and not np.array_equal(idx >= 0, np.asarray(m["cohorts"][unit.name][name]) == COHORTS[0]):
                raise PackError(f"{unit.dir}/panels/{name}.npz: in-set rows differ from the manifest cohorts")
            for s, _ in STRENGTHS:
                emit(s, f"{collection[:-1]}-{name}", z[s], len(prow), floor=files[s]["set"]["scales"])
            del z
    for strength, _ in STRENGTHS:
        records[strength]["files"] = files[strength]
        rel = base / f"{strength}.json"
        write_json(out / rel, records[strength])
        written.append(rel.as_posix())
    status = dict(status="packed", panels=panels, paths=paths, tstar=meta["tstar"],
                  missing=dict(panels=[k for k in PANELS if k not in panels],
                               paths=[k for k in m["paths"] if k not in paths] if unit.name in ("d32", "rand") else []),
                  release_missing=meta.get("missing"), done_outputs_sha256=unit.done["outputs_sha256"],
                  done_code_sha256=unit.done.get("code_sha256"), done_spec_sha256=unit.done.get("spec_sha256"),
                  stale=[f"built with other {w} than the current" for w in unit.stale],
                  sites=[{k: s[k] for k in ("degree_median", "degree_max", "tie_share")} for s in meta["sites"]],
                  read=unit.read)
    return status, written, saturated, records


# ------------------------------------------------------------------ aggregates (structures of depth.py cross/controls)
def overlap_value(entry):
    return None if entry is None else entry.get("overlap")


def scores_compact(scores):
    out = {}
    for k, s in (scores or {}).items():
        if "unavailable" in s:
            out[k] = dict(unavailable=s["unavailable"], cuts_band=bool(s.get("cuts_band")))
            continue
        out[k] = dict(layers=[overlap_value(x) for x in s["layers"]], stacked=overlap_value(s["stacked"]),
                      ranks=[[x["rank_a"], x["rank_b"]] for x in (*s["layers"], s["stacked"])],
                      cuts_band=bool(s["stacked"].get("cuts_band")))
    return out


def aggregates(root: Path):
    out, present = {}, {}
    p = root / "cross.json"
    if p.is_file():
        c = read_json(p)
        recs = []
        for r in c["records"]:
            keep = {k: r[k] for k in ("kind", "model", "set", "set_a", "set_b", "strength", "panel", "cohort", "count") if k in r}
            keep["scores"] = scores_compact(r["scores"])
            recs.append(keep)
        out["cross"] = dict(records=recs, skipped=c.get("skipped", []))
    p = root / "permutations.json"
    if p.is_file():
        c = read_json(p)
        recs = []
        for r in c["records"]:
            sc = {}
            for k, s in r["scores"].items():
                if "unavailable" in s:
                    sc[k] = dict(unavailable=s["unavailable"], cuts_band=bool(s.get("cuts_band")))
                    continue
                matched = scores_compact({k: s["matched"]})[k]
                sc[k] = dict(matched=matched, **{name: {q: s[name][q] for q in ("mean", "minimum", "maximum")}
                                                 for name in ("within_id_and_cohort", "free") if name in s})
            recs.append(dict(set=r["set"], panel=r["panel"], cohort=r["cohort"], strength=r["strength"], count=r["count"],
                             movable_share=r["movable_share"], within_id_uninformative=r["within_id_uninformative"], scores=sc))
        out["permutations"] = dict(seed=c.get("seed"), n_draws=c.get("n_draws"), records=recs, skipped=c.get("skipped", []))
    p = root / "gaussian.json"
    if p.is_file():
        c = read_json(p)
        recs = [dict(set=r["set"], count=r["count"],
                     scores={k: dict(gaussian=overlap_value(s["gaussian"]), real=[overlap_value(x) for x in s["real"]])
                             for k, s in r["scores"].items()}) for r in c["records"]]
        out["gaussian"] = dict(seed=c.get("seed"), records=recs, skipped=c.get("skipped", []), note=c.get("note"))
    p = root / "refit_agreement.json"
    if p.is_file():
        out["refit_agreement"] = read_json(p)
    for name in AGGREGATES:
        present[name] = name in out
        out.setdefault(name, None)
    return finite(out), present


# ------------------------------------------------------------------ facts quoted by the page
class Facts:
    def __init__(self):
        self.f = {}

    def add(self, name, value, text, source):
        if name in self.f:
            raise PackError(f"fact {name} twice")
        self.f[name] = dict(value=finite(value), text=text, source=source)


def fmt_int(n):
    return f"{n:,}"


def fmt_sig(x, digits=3):
    return f"{x:.{digits}g}"


def build_facts(inp: Inputs, plan, params, units, packed_bytes, aggregates_present):
    F = Facts()
    m = inp.manifest
    F.add("layers", plan["layers"], str(plan["layers"]), "units/*/*/tstar/modes.npy shape[0]")
    F.add("last_layer", plan["layers"] - 1, str(plan["layers"] - 1), "layers - 1")
    F.add("blocks", plan["layers"] - 1, str(plan["layers"] - 1), "layers - 1 (layer 0 is the embedding output)")
    F.add("modes_kept", plan["modes_kept"], str(plan["modes_kept"]), "units/*/*/tstar/modes.npy shape[2]")
    F.add("modes_shown", SITE_MODES, str(SITE_MODES), "exp1/pack_depth.py SITE_MODES (SPEC_DEPTH.md section 8)")
    F.add("context", CONTEXT, str(CONTEXT), "exp1/pack_depth.py CONTEXT (SPEC_DEPTH.md section 8)")
    F.add("quant_steps", QUANT_STEPS, str(QUANT_STEPS), "exp1/pack_depth.py QUANT_STEPS (SPEC_DEPTH.md section 8)")
    F.add("axis_pct", AXIS_QUANTILE * 100, f"{AXIS_QUANTILE * 100:g}%", "exp1/pack_depth.py AXIS_QUANTILE (read by site/js/depth.js from meta.json)")
    F.add("text_colours", TEXT_COLOURS, str(TEXT_COLOURS), "exp1/pack_depth.py TEXT_COLOURS (site/css/depth.css)")
    F.add("knn", params["knn"], str(params["knn"]), "exp1/depth_dm.py radius_kernel default k")
    F.add("tie_pct", params["tie"] * 100, f"{params['tie'] * 100:g}%", "exp1/depth.py bands()")
    exp = int(round(math.log10(params["condition"])))
    sup = str(exp).translate(str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻"))
    F.add("condition", params["condition"], f"10{sup}" if 10 ** exp == params["condition"] else f"{params['condition']:g}",
          "exp1/depth.py extension_config()")
    natural = [c for c, v in m["contexts"].items() if v["kind"] == "natural"]
    rand = [c for c, v in m["contexts"].items() if v["kind"] == "random"]
    shuf = [c for c, v in m["contexts"].items() if v["kind"] == "shuffled"]
    F.add("texts", len(natural), str(len(natural)), "manifest.json contexts (kind natural)")
    F.add("random_sequences", len(rand), str(len(rand)), "manifest.json contexts (kind random)")
    F.add("shuffled_sequences", len(shuf), str(len(shuf)), "manifest.json contexts (kind shuffled)")
    F.add("text_tokens", len(next(iter(m["paths"].values()))), fmt_int(len(next(iter(m["paths"].values())))),
          "manifest.json paths (tokens per text)")
    F.add("first_random", rand[0], rand[0], "manifest.json contexts")
    F.add("last_random", rand[-1], rand[-1], "manifest.json contexts")
    F.add("first_shuffled", shuf[0], shuf[0], "manifest.json contexts")
    F.add("last_shuffled", shuf[-1], shuf[-1], "manifest.json contexts")
    used = sorted({inp.key(c) for c, _, _ in m["sets"]["rand"]})
    unused = [c for c in rand if c not in used]
    F.add("rand_contexts", used, f"{used[0]} to {used[-1]}" if len(used) > 1 else used[0], "manifest.json sets.rand")
    F.add("unseen_random", unused, ", ".join(unused) if unused else "none", "manifest.json contexts minus sets.rand")
    src = [m["contexts"][c]["source_context"] for c in shuf]
    F.add("shuffled_sources", src, ", ".join(str(s) for s in src[:-1]) + f" and {src[-1]}", "manifest.json contexts[s*].source_context")
    for k, rows in m["panels"].items():
        F.add(f"panel.{k}", len(rows), fmt_int(len(rows)), f"manifest.json panels.{k}")
    per_doc = {}
    for c, _, _ in m["panels"]["natural"]:
        per_doc[inp.key(c)] = per_doc.get(inp.key(c), 0) + 1
    vals = sorted(set(per_doc.values()))
    F.add("panel.natural_per_text", vals[0] if len(vals) == 1 else None, fmt_int(vals[0]) if len(vals) == 1 else "varies",
          "manifest.json panels.natural")
    for name in SETS:
        rows = m["sets"][name]
        ctxs = sorted({inp.key(c) for c, _, _ in rows}, key=inp.cindex.get)
        F.add(f"set.{name}.n", len(rows), fmt_int(len(rows)), f"manifest.json sets.{name}")
        F.add(f"set.{name}.texts", len(ctxs), str(len(ctxs)), f"manifest.json sets.{name}")
    paths = list(m["paths"])
    F.add("paths", paths, ", ".join(paths), "manifest.json paths")
    F.add("path_count", len(paths), str(len(paths)), "manifest.json paths")
    F.add("units_expected", len(MODELS) * len(SETS), str(len(MODELS) * len(SETS)), "exp1/pack_depth.py MODELS x SETS")
    n_packed = sum(u["status"] == "packed" for u in units.values())
    F.add("units_packed", n_packed, str(n_packed), "this pack")
    F.add("site_megabytes", packed_bytes / 1e6, f"{packed_bytes / 1e6:.1f} MB", "bytes written by this pack")
    F.add("cap_megabytes", CAP_BYTES / 1e6, f"{CAP_BYTES / 1e6:g} MB", "exp1/pack_depth.py CAP_BYTES (SPEC_DEPTH.md section 8)")
    for key, label in MODELS:
        F.add(f"model.{key}", label, label, "probe.json models (display name)")
    for key, u in units.items():
        if u["status"] == "packed":
            model, name = key.split("/")
            F.add(f"tstar.{model}.{name}", u["tstar"], fmt_sig(u["tstar"]), f"units/{key}/unit.json tstar")
    for name, ok in aggregates_present.items():
        F.add(f"aggregate.{name}", ok, "computed" if ok else "not computed yet", f"{name}.json present")
    return F.f


# ------------------------------------------------------------------ main
def model_names(probe):
    names = {}
    for key, label in MODELS:
        ident = probe["models"][key]["id"].split("/")[-1]
        if not ident.lower().startswith(label.lower().split("-")[0].lower()):
            raise PackError(f"probe model {key} is {ident}, expected {label}")
        names[key] = label
    return names


def check_out(out: Path, root: Path, inputs):
    """OUT may never be, contain or sit inside the release (or hold an input); an existing OUT must be a prior pack."""
    for p in (root, *inputs):
        p = p.resolve()
        if out == p or out in p.parents:
            raise PackError(f"refusing --out {out}: it is or contains the input {p}")
    for inner in (root / "units", root / "refits"):
        if out == inner or inner in out.parents:
            raise PackError(f"refusing --out {out}: inside the release's {inner.name}/")
    if out.exists():
        try:
            ok = out.is_dir() and read_json(out / "meta.json").get("schema") == SCHEMA
        except (OSError, ValueError):
            ok = False
        if not ok and (out.is_file() or any(out.iterdir())):
            raise PackError(f"refusing --out {out}: it exists and is not a previous output of this script")


def pack(root: Path, out: Path, probe: Path, inputs: Path, release: bool):
    check_out(out, root, (probe, inputs, SPEC, HERE))
    inp = Inputs(root, probe, inputs)
    model_names(inp.probe)
    params = code_parameters()
    current_code, current_spec = code_hash(), file_hash(SPEC)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        return _pack(root, out, inp, params, current_code, current_spec, release, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)       # gone already after a successful pack (moved to OUT)


def _pack(root, out, inp, params, current_code, current_spec, release, tmp):
    units, written, saturated, plan = {}, [], 0, {"layers": None, "modes_kept": None}
    for model, _ in MODELS:
        for name in SETS:
            key = f"{model}/{name}"
            try:
                unit = Unit(root, model, name, current_code, current_spec)
                shape = np.load(unit.dir / "tstar/modes.npy", mmap_mode="r", allow_pickle=False).shape   # checked again from verified bytes
                if plan["layers"] is None:
                    plan.update(layers=int(shape[0]), modes_kept=int(shape[2]))
                stage = Path(tempfile.mkdtemp(prefix=".unit-", dir=tmp))
                try:
                    status, files, sat, _ = pack_unit(unit, inp, stage, plan)
                    for rel in files:          # a unit's files join the pack only once all of them are written
                        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                        os.replace(stage / rel, tmp / rel)
                finally:
                    shutil.rmtree(stage, ignore_errors=True)
                units[key], saturated = status, saturated + sat
                written += files
                log(f"{key}: packed ({len(files)} files; panels {', '.join(status['panels']) or 'none'}"
                    f"{'; paths ' + ', '.join(status['paths']) if status['paths'] else ''})")
            except FileNotFoundError as e:
                units[key] = dict(status="missing", reason=str(e))
            except Stale as e:
                units[key] = dict(status="skipped", reason=str(e))
                log(f"{key}: SKIPPED: {e}")
    if plan["layers"] is None:
        raise PackError("no complete unit to pack")
    m = inp.manifest
    write_json(tmp / "rows.json", dict(
        contexts=inp.contexts, fields=["context_index", "position"],
        sets={k: inp.rows_compact(v) for k, v in m["sets"].items()},
        panels={k: inp.rows_compact(v) for k, v in m["panels"].items()},
        cohort_labels=list(COHORTS),
        cohorts={s: {k: [COHORTS.index(c) for c in v] for k, v in kinds.items()} for s, kinds in m["cohorts"].items()},
        references=m["references"]))
    texts = {}
    for c in inp.contexts:
        meta = m["contexts"][c]
        texts[c] = dict(kind=meta["kind"], subset=meta.get("subset"), ids=inp.ids[c],
                        **({"source_context": meta["source_context"], "perm": meta.get("perm")} if meta["kind"] == "shuffled" else {}))
    write_json(tmp / "texts.json", dict(vocab={str(k): v for k, v in sorted(inp.vocab.items())}, contexts=texts,
                                        invalid_display=inp.probe.get("invalid_token_display")))
    agg, present = aggregates(root)
    write_json(tmp / "aggregates.json", agg)
    written += ["rows.json", "texts.json", "aggregates.json"]
    data_bytes = sum((tmp / p).stat().st_size for p in written)
    missing = [k for k, u in units.items() if u["status"] != "packed"]
    incomplete = [k for k, u in units.items() if u["status"] == "packed" and (u["missing"]["panels"] or u["missing"]["paths"])]
    no_ids = [c for c in inp.contexts if inp.ids[c] is None]
    stale = [k for k, u in units.items() if u.get("stale")]
    agg_skipped = {k: len(v.get("skipped") or []) for k, v in agg.items() if isinstance(v, dict) and "skipped" in v}
    complete = (not missing and not incomplete and not no_ids and not stale and all(present.values()) and not any(agg_skipped.values()))
    facts = build_facts(inp, plan, params, units, data_bytes, present)
    write_json(tmp / "facts.json", facts, compact=False)
    written.append("facts.json")
    inventory = {p: dict(bytes=(tmp / p).stat().st_size, sha256=file_hash(tmp / p)) for p in sorted(written)}
    meta = dict(
        schema=SCHEMA, generated=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        source_root=str(root), spec_sha256=current_spec, code_sha256=current_code,
        manifest_sha256=file_hash(root / "manifest.json"), layers=plan["layers"], modes_kept=plan["modes_kept"],
        modes_shown=SITE_MODES, context=CONTEXT, axis_quantile=AXIS_QUANTILE, parameters=params,
        models=[dict(key=k, name=v) for k, v in MODELS], strengths=[dict(key=k, label=v) for k, v in STRENGTHS],
        sets=[dict(key=s, n=len(m["sets"][s]),
                   contexts=sorted({inp.key(c) for c, _, _ in m["sets"][s]}, key=inp.cindex.get),
                   reference=m["references"][s]) for s in SETS],
        panels=[dict(key=k, n=len(m["panels"][k])) for k in PANELS], paths=list(m["paths"]),
        units={k: {kk: vv for kk, vv in u.items() if kk != "read"} for k, u in units.items()},
        missing_units=missing, incomplete_units=incomplete, aggregates=present, aggregates_skipped=agg_skipped,
        contexts_without_ids=no_ids, stale_units=stale, complete=complete,
        quantisation=("int8, C order (layer, row, mode), modes 1..16, square-root companded, one scale per (layer, mode) in "
                      "files.<tag>.scales[layer][mode]: set file scale = max finite |value| over its rows; panel/path file scale = "
                      "max(set file scale, own max finite |value|); every scale a float32 value, exact. "
                      "q = sign(x) * round-half-even(127 * sqrt(|x| / scale)) in float64 (divide, sqrt, multiply, round), "
                      "clipped to [-127, 127]; scale 0 -> q = 0; NaN (withheld) -> -128; x = sign(q) * (q / 127)^2 * scale"),
        saturated_entries=saturated, bytes=data_bytes, cap_bytes=CAP_BYTES, files=inventory)
    write_json(tmp / "meta.json", meta, compact=False)
    total = sum(p.stat().st_size for p in tmp.rglob("*") if p.is_file())
    unlisted = {p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file()} - set(inventory) - {"meta.json"}
    if unlisted:
        raise PackError(f"files written but not inventoried: {sorted(unlisted)[:5]}")
    n_packed = len(units) - len(missing)
    log(f"measured {total:,} bytes ({total / 1e6:.2f} MB) for {n_packed} of {len(units)} units; cap {CAP_BYTES / 1e6:g} MB")
    if total > CAP_BYTES:
        raise PackError(f"site data is {total / 1e6:.1f} MB, above the {CAP_BYTES / 1e6:g} MB cap; nothing written")
    full = projection(root, inp, inventory, units)
    log(f"projection for all {len(units)} units with every panel and path: {full / 1e6:.1f} MB")
    if missing:
        log(f"MISSING: {len(missing)} of {len(units)} units not packed:")
        by_reason = {}
        for k in missing:
            by_reason.setdefault(units[k]["reason"], []).append(k)
        for reason, ks in by_reason.items():
            log(f"  {reason}: {', '.join(ks)}")
    if incomplete:
        log("PACKED WITHOUT SOME PLACEMENTS: " + "; ".join(
            f"{k} (no panels {units[k]['missing']['panels']}, no paths {units[k]['missing']['paths']})" for k in incomplete))
    for name, ok in present.items():
        if not ok:
            log(f"aggregate {name}.json: not computed yet")
        elif agg_skipped.get(name):
            log(f"aggregate {name}.json: {agg_skipped[name]} skipped entries (computed before every unit was there)")
    if stale:
        log("STALE (packed, but built with other pipeline code or spec than the current; not releasable): " + "; ".join(
            f"{k} ({', '.join(units[k]['stale'])})" for k in stale))
    if no_ids:
        log(f"no token ids for {', '.join(no_ids)} (synthetic inputs file missing)")
    if release and not complete:
        raise PackError("--release: the release is incomplete (see the lines above); nothing written")
    old = None
    if out.exists():
        old = Path(tempfile.mkdtemp(prefix=f".{out.name}.old-", dir=out.parent))
        os.replace(out, old / "x")
    os.replace(tmp, out)
    if old:
        shutil.rmtree(old)
    log(f"wrote {out} ({'complete' if complete else 'partial'}); int8 entries at |q| = 127 (the largest value at their layer): {saturated:,}")
    return meta


def projection(root, inp, inventory, units):
    """Bytes for 22 complete units: int8 bytes are exact from the shapes; JSON per unit taken as the largest packed."""
    m = inp.manifest
    per_row = None
    for k, u in units.items():
        if u["status"] == "packed":
            name = k.split("/")[1]
            per_row = inventory[f"u/{k}/tstar/set.bin"]["bytes"] / len(m["sets"][name])
            break
    json_max = max((v["bytes"] for p, v in inventory.items() if p.startswith("u/") and p.endswith(".json")), default=0)
    shared = sum(v["bytes"] for p, v in inventory.items() if not p.startswith("u/"))
    total = shared
    for model, _ in MODELS:
        for name in SETS:
            rows = len(m["sets"][name]) + sum(len(r) for r in m["panels"].values())
            if name in ("d32", "rand"):
                rows += sum(len(r) for r in m["paths"].values())
            total += len(STRENGTHS) * (rows * per_row + json_max)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("/data/tm_depth"), help="release root written by exp1/depth.py")
    ap.add_argument("--out", type=Path, help="output directory, replaced (default ROOT/site_depth)")
    ap.add_argument("--probe", type=Path, default=Path("/data/tm_acts/probe/probe.json"))
    ap.add_argument("--inputs", type=Path, default=Path("/data/tm_acts/depth_acts/inputs.json"))
    ap.add_argument("--release", action="store_true", help="refuse unless all 22 units, every placement and every aggregate are present")
    a = ap.parse_args()
    root = a.root.resolve()
    try:
        pack(root, (a.out or root / "site_depth").resolve(), a.probe, a.inputs, a.release)
    except (PackError, OSError, KeyError, ValueError) as e:
        ap.exit(1, f"pack_depth failed: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
