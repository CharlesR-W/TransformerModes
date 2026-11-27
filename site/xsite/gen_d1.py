"""D1: the depth embedding of tokens (SPEC_DEPTH.md), summarised for the Experiments tab.

    run_depth(dd, release, out)     # called by site/build_site.py when the site is built with --depth

Reads only (never recomputes a result):
  dd/        the copied pack of exp1/pack_depth.py: meta.json, aggregates.json (cross-model, ladder, permutation,
             Gaussian and refit summaries) and u/<model>/<set>/<strength>.json (per-mode facts of modes 1..16);
  release/   the release written by exp1/depth.py, for one thing the pack does not carry: the placement measures of
             section 6 (units/<model>/<set>/tstar/facts.json, key "placement"). Each file read is checked against
             the unit's done.json, and done.json against the outputs hash the pack's meta.json recorded, so these
             numbers come from exactly the unit outputs the audited pack was made from.

Writes out/d1-depth.json (chart payloads) and out/d1-depth.facts.json (every number the entry quotes, each with the
file and key it comes from). Summaries made here, and only here, are: medians and quartiles over modes 1..16 of
the per-mode facts (question 1), and minima and maxima over the sets a sentence names. Every qualitative claim the
entry's prose makes about these numbers is checked below (claim(...)); a build fails if the data contradict one,
so a corrected release cannot leave a sentence that is no longer true.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from xlib import Out, read_json

XID = "d1-depth"
K = "16"                      # modes compared in questions 2 and 3 (the first 16: the modes the viewer carries)
KP = "8"                      # placement measures exist for the top 8 and top 64 (section 6); 8 is used
T, T16 = "tstar", "tstar_div16"
LADDER = ("doc00", "pairA", "d04", "d08", "d16", "d32")
DOC_SETS = ("doc00", "doc01", "doc02", "doc04", "pairA", "pairB", "d04", "d08", "d16", "d32")
MULTI = ("pairA", "pairB", "d04", "d08", "d16", "d32", "rand")
COLOUR = {"pythia": "pythia", "olmo": "olmo"}
COH = {"in_set": "in the set", "seen_sequence_new_position": "new positions", "unseen_sequence": "unseen"}


class ClaimError(ValueError):
    pass


def claim(ok, what):
    if not ok:
        raise ClaimError(f"D1: the entry says {what}, and the data no longer show it; revise content/experiments/d1-depth.html")


def canon(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run_depth(dd: Path, release: Path, out: Path):
    dd, release = Path(dd), Path(release)
    meta = read_json(dd / "meta.json")
    agg = read_json(dd / "aggregates.json")
    o = Out(XID, out)
    models = [(m["key"], m["name"]) for m in meta["models"]]
    sets = [s["key"] for s in meta["sets"]]
    setinfo = {s["key"]: s for s in meta["sets"]}
    name = dict(models)

    def unit(m, s, t):
        return read_json(dd / "u" / m / s / f"{t}.json")

    def placement(m, s):
        """units/<m>/<s>/tstar/facts.json "placement", verified against done.json and the pack's record of it."""
        udir = release / "units" / m / s
        done = read_json(udir / "done.json")
        if sha(canon(done["files"])) != done.get("outputs_sha256") or done["outputs_sha256"] != meta["units"][f"{m}/{s}"]["done_outputs_sha256"]:
            raise ValueError(f"{udir}/done.json is not the unit output the depth pack was made from")
        raw = (udir / "tstar" / "facts.json").read_bytes()
        if sha(raw) != done["files"].get("tstar/facts.json"):
            raise ValueError(f"{udir}/tstar/facts.json differs from its done.json")
        return json.loads(raw)["placement"]

    P = agg["permutations"]["records"]
    C = agg["cross"]["records"]

    def perm(s, panel, cohort, t=T):
        r = [x for x in P if x["set"] == s and x["panel"] == panel and x["cohort"] == cohort and x["strength"] == t]
        if len(r) != 1:
            raise ValueError(f"aggregates.json permutations: {len(r)} records for {s}/{panel}/{cohort}/{t}")
        return r[0]

    def psrc(s, panel, cohort, path, t=T):
        return f"aggregates.json permutations.records[set={s}, panel={panel}, cohort={cohort}, strength={t}].{path}"

    def f2(name_, v, src, d=2):
        return o.fact(name_, v, d, source=src)

    # ------------------------------------------------------------------ the scope line
    o.fact("sets", len(sets), source="meta.json sets")
    o.fact("models", len(models), source="meta.json models")
    o.fact("k", int(K), source="gen_d1.py K: questions 2 and 3 compare the first 16 modes")
    o.fact("kp", int(KP), source="gen_d1.py KP: placement measures (section 6) exist for the top 8 and top 64; 8 is used")
    o.fact("draws", agg["permutations"]["n_draws"], source="aggregates.json permutations.n_draws")
    def listing(ctx):
        nums = [int(c[1:]) for c in ctx]
        if len(ctx) > 2 and nums == list(range(nums[0], nums[0] + len(ctx))):
            return f"{ctx[0]} to {ctx[-1]}"
        return ctx[0] if len(ctx) == 1 else ", ".join(ctx[:-1]) + " and " + ctx[-1]
    for s in sets:
        o.fact(f"set.{s.lower()}.ctx", listing(setinfo[s]["contexts"]), source=f"meta.json sets[{s}].contexts")
    for s in sets:
        o.fact(f"set.{s.lower()}.each", setinfo[s]["n"] // len(setinfo[s]["contexts"]), source=f"meta.json sets[{s}].n / len(contexts)")
    o.fact("k8", 8, source="the smallest of the three mode counts the release reports (SPEC_DEPTH.md section 6: k = 8, 16, 64)")
    o.fact("k64", 64, source="the largest of the three mode counts the release reports (SPEC_DEPTH.md section 6: k = 8, 16, 64)")
    o.fact("weak_pct", 0.10, pct=True, source="SPEC_DEPTH.md section 7, control 4 (a): a same-id reference is called uninformative where under 10% of tokens can move (the release's within_id_uninformative)")
    counts = {r["count"] for r in C if r["kind"] == "ladder"}
    if len(counts) != 1:
        raise ValueError(f"aggregates.json cross: ladder records on different numbers of tokens {sorted(counts)}")
    o.fact("ladder_tokens", counts.pop(), source="aggregates.json cross.records[kind=ladder].count (the same in every ladder record)")

    # ------------------------------------------------------------------ question 1: the modes, per set (t*, modes 1..16)
    rows1 = [s for s in sets]
    bands_in = band_units = modes_total = 0

    def check_modes(md, where, keys=("token_support", "endpoint_cos", "adjacent_cos_mean")):
        if [x["k"] for x in md] != list(range(1, int(K) + 1)):
            raise ValueError(f"{where}: expected modes 1..{K}, got {[x['k'] for x in md]}")
        for x in md:
            for key in keys:
                if not (isinstance(x[key], (int, float)) and np.isfinite(x[key])):
                    raise ValueError(f"{where}: mode {x['k']} has no finite {key}")
            if x["document_share"] is not None and not np.isfinite(x["document_share"]):
                raise ValueError(f"{where}: mode {x['k']} has a non-finite document_share")
    q1 = {key: {m: [] for m, _ in models} for key in ("support", "end", "doc")}
    summary = {}
    for m, _ in models:
        for s in sets:
            u = unit(m, s, T)
            md = u["modes"]
            check_modes(md, f"u/{m}/{s}/{T}.json")
            bands_in += sum(x["band"] is not None for x in md)
            band_units += any(x["band"] is not None for x in md)
            modes_total += len(md)
            vals = {"support": [x["token_support"] for x in md], "end": [x["endpoint_cos"] for x in md],
                    "adj": [x["adjacent_cos_mean"] for x in md],
                    "doc": [x["document_share"] for x in md] if md[0]["document_share"] is not None else None}
            md16 = unit(m, s, T16)["modes"]
            check_modes(md16, f"u/{m}/{s}/{T16}.json", keys=("adjacent_cos_mean",))
            adj16 = [x["adjacent_cos_mean"] for x in md16]
            vals["absend"] = [abs(x["endpoint_cos"]) for x in md]
            rec = {}
            for key, v in vals.items():
                if v is None:
                    rec[key] = None
                    continue
                a = np.asarray(v, dtype=np.float64)
                rec[key] = (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))
            rec["adj16"] = float(np.median(np.asarray(adj16, dtype=np.float64)))
            summary[m, s] = rec
            for key in ("support", "end", "doc"):
                q1[key][m].append(rec[key])
    src_u = "u/<model>/<set>/tstar.json modes[0..15].{} (median over the 16 modes)"
    o.fact("q1.band.modes", bands_in, source="u/<model>/<set>/tstar.json modes[0..15].band not null, count over models and sets")
    o.fact("q1.band.total", modes_total, source="u/<model>/<set>/tstar.json modes[0..15], count over models and sets")
    o.fact("q1.band.units", band_units, source="embeddings (model, set) at t* with at least one of modes 1..16 in a near-tied band")
    o.fact("q1.units", len(summary), source="models times sets")
    adj = [summary[k]["adj"][0] for k in summary]
    adj16 = [summary[k]["adj16"] for k in summary]
    f2("q1.adj.min", min(adj), src_u.format("adjacent_cos_mean") + ", minimum over models and sets")
    f2("q1.adj.max", max(adj), src_u.format("adjacent_cos_mean") + ", maximum over models and sets")
    f2("q1.adj16.min", min(adj16), "u/<model>/<set>/tstar_div16.json modes[0..15].adjacent_cos_mean (median over the 16 modes), minimum")
    f2("q1.adj16.max", max(adj16), "u/<model>/<set>/tstar_div16.json modes[0..15].adjacent_cos_mean (median over the 16 modes), maximum")
    claim(min(adj) > 0.85, "a mode changes little from one layer to the next at t* (median adjacent-layer cosine above 0.85 everywhere)")
    claim(all(summary[k]["adj16"] < summary[k]["adj"][0] for k in summary), "in every embedding the layer-to-layer cosine is lower at t*/16 than at t*")
    sup = [summary[k]["support"][0] for k in summary]
    o.fact("q1.support.min", min(sup), pct=True, source=src_u.format("token_support") + ", minimum")
    o.fact("q1.support.max", max(sup), pct=True, source=src_u.format("token_support") + ", maximum")
    claim(max(sup) < 0.5, "a typical mode is carried by a minority of the set's tokens (median token support under 50% everywhere)")
    ends = sorted(((summary[k]["end"][0], k) for k in summary), key=lambda x: -abs(x[0]))
    (out_v, (out_m, out_s)), rest = ends[0], [v for v, _ in ends[1:]]
    f2("q1.end.min", min(rest), src_u.format("endpoint_cos") + ", minimum over models and sets except the one largest in size")
    f2("q1.end.max", max(rest), src_u.format("endpoint_cos") + ", maximum over models and sets except the one largest in size")
    f2("q1.end.out", out_v, src_u.format("endpoint_cos").replace("<model>/<set>", f"{out_m}/{out_s}"))
    o.fact("q1.end.out_who", f"{name[out_m]}'s {out_s}", source="the model and set of q1.end.out")
    claim(max(abs(v) for v in rest) < 0.2 and abs(out_v) > 0.3, "the median signed cosine between a mode's layer-0 and layer-16 slices is near zero in all but one embedding, and far from zero in that one")
    ab = [summary[k]["absend"][0] for k in summary]
    f2("q1.absend.min", min(ab), src_u.format("endpoint_cos") + " taken in absolute value (median over the 16 modes of |endpoint_cos|), minimum over models and sets")
    f2("q1.absend.max", max(ab), src_u.format("endpoint_cos") + " taken in absolute value (median of |endpoint_cos|), maximum over models and sets")
    doc = lambda m, s: summary[m, s]["doc"][0]
    pairs = [doc(m, s) for m, _ in models for s in ("pairA", "pairB")]
    f2("q1.doc.pairs", max(pairs), src_u.format("document_share") + " for pairA and pairB, maximum over both models", 3)
    for m, _ in models:
        f2(f"q1.doc.d32.{m}", doc(m, "d32"), src_u.format("document_share").replace("<model>/<set>", f"{m}/d32"))
    f2("q1.doc.rand.min", min(doc(m, "rand") for m, _ in models), src_u.format("document_share") + " for rand (between-sequence share), minimum over both models", 3)
    f2("q1.doc.rand.max", max(doc(m, "rand") for m, _ in models), src_u.format("document_share") + " for rand (between-sequence share), maximum over both models", 3)
    claim(3 * max(pairs) < min(doc(m, "d32") for m, _ in models), "modes of the 32-document set separate documents far more than those of the pairs")
    claim(all(doc(m, "d04") < doc(m, "d32") for m, _ in models), "the share is larger in d32 than in d04 in both models")

    labels = {s: f"{s} · {len(setinfo[s]['contexts'])} {'random seq.' if s == 'rand' else 'text' if len(setinfo[s]['contexts']) == 1 else 'texts'}" for s in sets}

    def dots_panel(title, sub, key, xr, fmt):
        series = []
        for m, nm in models:
            v = q1[key][m]
            series.append(dict(name=nm, color=COLOUR[m], values=[None if r is None else r[0] for r in v],
                               lo=[None if r is None else r[1] for r in v], hi=[None if r is None else r[2] for r in v]))
        return dict(title=title, sub=sub, rows=[labels[s] for s in rows1], series=series, xrange=xr, fmt=fmt, legend=False)
    o.chart("q1", dict(cols=3, mobile_stack=True, panels=[
        dots_panel("How many tokens carry a mode", "token support: 100% = spread evenly over the set's tokens", "support", [0, 0.5], "pct"),
        dots_panel("First layer against last", "signed cosine between a mode's values at the first and last layer", "end", [-1, 1], "2f"),
        dots_panel("How much of a mode separates texts", "document share; not defined for one text", "doc", [0, 1], "2f")]))

    # depth weight of modes 1..16 in d32 (supporting)
    hp = []
    for m, nm in models:
        u = unit(m, "d32", T)
        z = [x["depth_weight"] for x in u["modes"]]
        hp.append(dict(title=nm, sub="d32 · t* · ≈ marks modes in a near-tied group", rows=[f"{x['k']}{' ≈' if x['band'] else ''}" for x in u["modes"]], cols=list(range(meta["layers"])),
                       z=z, zrange=[0, float(max(max(r) for r in z))], rowlabel="mode", collabel="layer", zlabel="depth weight (share of the mode at that layer)", fmt="2f"))
    o.chart("q1_depth", dict(cols=2, mobile_stack=True, panels=hp))
    dmax = max(max(max(r) for r in p["z"]) for p in hp)
    o.fact("q1.depth.max", dmax, 1, pct=True, source="u/<model>/d32/tstar.json modes[0..15].depth_weight, the largest entry over both models")
    claim(dmax < 0.3, "in d32 no mode sits mostly in one layer")

    # ------------------------------------------------------------------ question 2: the ladder (within a model, t*, k = 16)
    lad = [r for r in C if r["kind"] == "ladder" and r["strength"] == T]
    steps = list(zip(LADDER, LADDER[1:])) + [("d32", "rand")]
    q2rows, series, cuts, every = [], [], [], {}
    for a, b in steps:
        q2rows.append(f"{a} → {b}")
    for m, nm in models:
        vals, hol = [], []
        for a, b in steps:
            r = [x for x in lad if x["model"] == m and x["set_a"] == a and x["set_b"] == b]
            if len(r) != 1:
                raise ValueError(f"aggregates.json cross: {len(r)} ladder records for {m} {a}->{b}")
            sc = r[0]["scores"][K]
            v = None if "unavailable" in sc else sc["stacked"]
            vals.append(v)
            hol.append(bool(sc.get("cuts_band")))
            every[m, a, b] = v
            src = f"aggregates.json cross.records[kind=ladder, model={m}, set_a={a}, set_b={b}, strength=tstar].scores.16.stacked"
            f2(f"q2.{m}.{a.lower()}_{b.lower()}", v, src)
            if sc.get("cuts_band"):
                cuts.append(f"{nm} {a} → {b}")
        series.append(dict(name=nm, color=COLOUR[m], values=vals, hollow=hol))
    o.chart("q2", dict(rows=q2rows, series=series, xrange=[0, 1], fmt="2f", xlabel="overlap of the first 16 modes",
                       hollow_note="the cutoff at 16 modes is flagged (near-tied modes)"))
    o.fact("q2.cut_count", len(cuts), source="aggregates.json cross.records[kind=ladder, strength=tstar].scores.16.cuts_band, count")
    o.fact("q2.steps", len(steps) * len(models), source="ladder steps times models")
    o.fact("q2.cut_list", ", ".join(cuts) if cuts else "none", source="aggregates.json cross.records[kind=ladder, strength=tstar].scores.16.cuts_band")
    olmo = [every["olmo", a, b] for a, b in steps if b != "rand" and f"OLMo-1B {a} → {b}" not in cuts and every["olmo", a, b] is not None]
    f2("q2.olmo.min", min(olmo), "aggregates.json cross.records[kind=ladder, model=olmo, strength=tstar].scores.16.stacked, minimum over the document steps (doc00 to d32)")
    f2("q2.olmo.max", max(olmo), "aggregates.json cross.records[kind=ladder, model=olmo, strength=tstar].scores.16.stacked, maximum over the document steps")
    # the panel's size in each set of the ladder (SPEC_DEPTH.md section 6, question 2: "sizes are compared separately, by the
    # radius ratios of the same rows"); the pack drops these, so they are read from the release's cross.json, whose ladder
    # overlaps must equal the pack's (the pack's aggregates.json is made from this file)
    raw = read_json(release / "cross.json")
    rlad = [r for r in raw["records"] if r["kind"] == "ladder" and r["strength"] == T]
    radius, rcut, rflag = {}, [], set()
    for m, nm in models:
        for a, b in steps:
            r = [x for x in rlad if x["model"] == m and x["set_a"] == a and x["set_b"] == b]
            if len(r) != 1 or abs(r[0]["scores"][K]["stacked"]["overlap"] - every[m, a, b]) > 1e-12:
                raise ValueError(f"{release}/cross.json: ladder record {m} {a}->{b} does not match the pack's aggregates.json")
            for side, s in (("sizes_a", a), ("sizes_b", b)):
                z = r[0][side][KP]
                v = z["pooled_radius_ratio"]
                if (m, s) in radius and abs(radius[m, s] - v) > 1e-12:
                    raise ValueError(f"cross.json: two different panel radii for {m}/{s}")
                radius[m, s] = v
                if z["cuts_band"]:
                    rflag.add((m, s))
                    if f"{nm} {s}" not in rcut:
                        rcut.append(f"{nm} {s}")
    for m, _ in models:
        for s in (*LADDER, "rand"):
            f2(f"q2.rr.{m}.{s.lower()}", radius[m, s], f"cross.json records[kind=ladder, model={m}, strength=tstar, set_a={s} (sizes_a) or set_b={s} (sizes_b)].8.pooled_radius_ratio")
        ld = [radius[m, s] for s in LADDER if (m, s) not in rflag]
        f2(f"q2.rr.{m}.min", min(ld), f"the same, minimum over doc00 to d32 for {m}, flagged values (cuts_band) left out")
        f2(f"q2.rr.{m}.max", max(ld), f"the same, maximum over doc00 to d32 for {m}, flagged values left out")
        claim(not {(m, "d32"), (m, "rand")} & rflag, "the d32 and rand panel radii are not flagged")
        claim(abs(radius[m, "d32"] - radius[m, "rand"]) < 0.05, "the panel's radius is about the same in d32 and in rand, although those spaces overlap little")
    o.fact("q2.rr.cut_list", ", ".join(rcut) or "none", source="cross.json records[kind=ladder, strength=tstar].sizes_a/sizes_b.8.cuts_band")
    allv = [v for v in every.values() if v is not None]
    claim(max(allv) < 0.75, "neighbouring sets of the ladder share only part of their space (every step under 0.75)")
    claim(all(every[m, "d32", "rand"] < every[m, "d16", "d32"] for m, _ in models), "the step from d32 to rand shares less than the step into d32")
    claim(all(sorted(v for (mm, a, b), v in every.items() if mm == m).index(every[m, "d32", "rand"]) <= 1 for m, _ in models),
          "the step from d32 to rand is one of the two lowest in each model")

    # ------------------------------------------------------------------ question 3: between models (t*, k = 16)
    own, sameid, free, gap, mov, cut3 = {}, {}, {}, {}, {}, []
    s3 = []
    for s in sets:
        r = perm(s, "construction", "in_set")
        sc = r["scores"][K]
        if "unavailable" in sc:
            raise ValueError(f"{s}: between-model overlap unavailable at k = 16")
        own[s] = sc["matched"]["stacked"]
        sameid[s] = (sc["within_id_and_cohort"]["mean"][-1], sc["within_id_and_cohort"]["minimum"][-1], sc["within_id_and_cohort"]["maximum"][-1])
        free[s] = (sc["free"]["mean"][-1], sc["free"]["minimum"][-1], sc["free"]["maximum"][-1])
        mov[s] = r["movable_share"]
        if sc["matched"]["cuts_band"]:
            cut3.append(s)
        s3.append(s)
    rows_ = read_json(dd / "rows.json")
    texts = read_json(dd / "texts.json")["contexts"]
    fixed = {}
    for s in sets:
        ids = [texts[rows_["contexts"][c]]["ids"][p - 1] for c, p in rows_["sets"][s]]
        if len(ids) != setinfo[s]["n"]:
            raise ValueError(f"rows.json: {s} has {len(ids)} rows, meta.json says {setinfo[s]['n']}")
        fixed[s] = len(set(ids)) / len(ids)
    o.fact("q3.fixed.min", min(fixed[s] for s in DOC_SETS), pct=True, source="rows.json sets.<set> with texts.json ids: distinct word pieces / tokens, minimum over the document sets (a random reordering within a group leaves one token in place on average)")
    o.fact("q3.fixed.max", max(fixed[s] for s in DOC_SETS), pct=True, source="the same, maximum over the document sets")
    o.fact("q3.fixed.rand", fixed["rand"], pct=True, source="the same for rand")
    stable = [s for s in sets if s not in cut3]
    claim(len(cut3) == 1, "exactly one set's own-token comparison at 16 modes cuts a near-tied group")
    stable_docs = [s for s in stable if s in DOC_SETS]
    src3 = lambda path: psrc("<set>", "construction", "in_set", f"scores.16.{path}")
    f2("q3.own.min", min(own[s] for s in stable), src3("matched.stacked") + ", minimum over sets whose first 16 modes cut no group")
    f2("q3.own.max", max(own[s] for s in stable), src3("matched.stacked") + ", maximum over the same sets")
    o.fact("q3.stable", len(stable), source="sets with scores.16.matched.cuts_band false")
    o.fact("q3.cut_list", ", ".join(cut3) if cut3 else "none", source=src3("matched.cuts_band"))
    f2("q3.sameid.min", min(sameid[s][0] for s in stable_docs), src3("within_id_and_cohort.mean[-1]") + " (all layers stacked), minimum over the document sets that cut no group")
    f2("q3.sameid.max", max(sameid[s][0] for s in stable_docs), src3("within_id_and_cohort.mean[-1]") + ", maximum over the same sets")
    g = [own[s] - sameid[s][0] for s in stable_docs]
    f2("q3.gap.min", min(g), "q3: matched.stacked − within_id_and_cohort.mean[-1], minimum over the same document sets")
    f2("q3.gap.max", max(g), "q3: matched.stacked − within_id_and_cohort.mean[-1], maximum over the same document sets")
    f2("q3.free.max", max(free[s][0] for s in sets), src3("free.mean[-1]") + " (mean of the draws), maximum over all sets", 3)
    o.fact("q3.mov.min", min(mov[s] for s in DOC_SETS), pct=True, source=psrc("<set>", "construction", "in_set", "movable_share") + ", minimum over the document sets")
    o.fact("q3.mov.max", max(mov[s] for s in DOC_SETS), pct=True, source=psrc("<set>", "construction", "in_set", "movable_share") + ", maximum over the document sets")
    f2("q3.rand.own", own["rand"], psrc("rand", "construction", "in_set", "scores.16.matched.stacked"), 3)
    f2("q3.rand.sameid", sameid["rand"][0], psrc("rand", "construction", "in_set", "scores.16.within_id_and_cohort.mean[-1]"), 3)
    o.fact("q3.rand.mov", mov["rand"], pct=True, source=psrc("rand", "construction", "in_set", "movable_share"))
    claim(all(own[s] > sameid[s][2] for s in stable_docs), "in every document set the models agree more than the whole range of same-id swaps")
    claim(min(sameid[s][0] for s in stable_docs) > 10 * max(free[s][0] for s in sets), "free swaps leave almost nothing next to same-id swaps")
    claim(all(sameid[s][0] > 0.5 * own[s] for s in stable_docs), "in the document sets much of the overlap (over half) survives reordering within word pieces")
    claim(mov["rand"] < 0.25 and own["rand"] - sameid["rand"][0] < 0.01, "in rand the same-id reference is close to the matched value and few tokens can move")
    o.chart("q3", dict(rows=[labels[s] for s in s3], xrange=[0, 1], fmt="2f", hollow_note="the cutoff at 16 modes is flagged (near-tied modes)",
                       xlabel="overlap of the first 16 modes", series=[
                           dict(name="the two models, same tokens", color="ink", values=[own[s] for s in s3], hollow=[s in cut3 for s in s3]),
                           dict(name=f"reordered within each word piece (mean; line: range of {agg['permutations']['n_draws']})", color="far", marker="diamond",
                                values=[sameid[s][0] for s in s3], lo=[sameid[s][1] for s in s3], hi=[sameid[s][2] for s in s3]),
                           dict(name=f"reordered freely (mean of {agg['permutations']['n_draws']})", color="mute", marker="square", values=[free[s][0] for s in s3])]))

    # per-layer modes (each layer embedded on its own) beside two random vectors per token id
    G = {r["set"]: r["scores"][K] for r in agg["gaussian"]["records"]}
    gsrc = lambda s, path: f"aggregates.json gaussian.records[set={s}].scores.16.{path}"
    d32 = G["d32"]["real"]
    f2("q3.layer.l0.min", min(G[s]["real"][0] for s in sets), gsrc("<set>", "real[0]") + ", minimum over sets")
    f2("q3.layer.l0.max", max(G[s]["real"][0] for s in sets), gsrc("<set>", "real[0]") + ", maximum over sets")
    f2("q3.layer.d32.l0", d32[0], gsrc("d32", "real[0]"))
    f2("q3.layer.d32.l13.min", min(d32[1:4]), gsrc("d32", "real[1..3]") + ", minimum")
    f2("q3.layer.d32.l13.max", max(d32[1:4]), gsrc("d32", "real[1..3]") + ", maximum")
    f2("q3.layer.d32.late.min", min(d32[4:]), gsrc("d32", "real[4..16]") + ", minimum")
    f2("q3.layer.d32.late.max", max(d32[4:]), gsrc("d32", "real[4..16]") + ", maximum")
    f2("q3.layer.d32.ref", G["d32"]["gaussian"], gsrc("d32", "gaussian"))
    f2("q3.layer.ref.min", min(G[s]["gaussian"] for s in DOC_SETS), gsrc("<set>", "gaussian") + ", minimum over the document sets")
    f2("q3.layer.ref.max", max(G[s]["gaussian"] for s in DOC_SETS), gsrc("<set>", "gaussian") + ", maximum over the document sets")
    f2("q3.layer.rand.ref", G["rand"]["gaussian"], gsrc("rand", "gaussian"), 3)
    f2("q3.layer.rand.l0", G["rand"]["real"][0], gsrc("rand", "real[0]"))
    f2("q3.layer.rand.late.min", min(G["rand"]["real"][4:]), gsrc("rand", "real[4..16]") + ", minimum")
    f2("q3.layer.rand.late.max", max(G["rand"]["real"][4:]), gsrc("rand", "real[4..16]") + ", maximum")
    claim(min(G["rand"]["real"][0], *G["rand"]["real"][4:]) > 5 * G["rand"]["gaussian"], "in rand the models agree far above the random-vector reference at layer 0 and from layer 4 on")
    l1 = [G[s]["real"][1] for s in sets]
    ref32 = G["d32"]["gaussian"]
    claim(all(abs(v - ref32) < 0.15 for v in d32[4:]) and max(d32[1:4]) < ref32 / 2, "in d32 the per-layer agreement stays around the random-vector reference from layer 4 and falls far below it at layers 1 to 3")
    claim(max(d32[1:4]) < 0.5 * d32[0] and min(d32[4:]) > max(d32[1:4]), "in d32 the per-layer agreement drops after layer 0 and recovers from layer 4")
    claim(all(G[s]["real"][1] < G[s]["real"][0] for s in sets), "in every set the per-layer agreement is lower at layer 1 than at layer 0")
    above = [s for s in sets if max(G[s]["real"][1:]) > G[s]["real"][0]]
    claim(set(sets) - set(above) == {"d32"}, "in every set but d32 some later layer agrees more than layer 0")
    o.fact("q3.layer.above_count", len(above), source=gsrc("<set>", "real") + ": sets in which some layer after 0 exceeds layer 0")
    o.fact("q3.layer.l1.min", min(l1), 2, source=gsrc("<set>", "real[1]") + ", minimum over sets")
    o.fact("q3.layer.l1.max", max(l1), 2, source=gsrc("<set>", "real[1]") + ", maximum over sets")
    lay = list(range(meta["layers"]))
    lines = [dict(name="other sets", y=G[s]["real"], color="mute2", width=1) for s in sets if s not in ("d32", "rand")]
    lines = lines[:1] + [dict(name="", y=x["y"], color="mute2", width=1, label=False) for x in lines[1:]]
    lines += [dict(name=f"d32 ({len(setinfo['d32']['contexts'])} texts)", y=d32, color="ink", width=2.4, dots=True),
              dict(name="rand (random tokens)", y=G["rand"]["real"], color="accent", width=2.4, dots=True)]
    o.chart("q3_layers", dict(x=lay, xlabel="layer (each embedded on its own)", ylabel="overlap of the two models' first 16 modes", yrange=[0, 1], fmt="2f",
                              series=[x for x in lines if x["name"]] + [x for x in lines if not x["name"]],
                              reflines=[dict(y=G["d32"]["gaussian"], label="d32: random dictionaries"),
                                        dict(y=G["rand"]["gaussian"], label="rand: random dictionaries")]))

    # ------------------------------------------------------------------ question 4: placed tokens of other kinds (t*)
    pl = {(m, s): placement(m, s) for m, _ in models for s in ("d32", "rand")}
    q4rows = [("d32", "natural", "seen_sequence_new_position", "new positions (ref.)"),
              ("d32", "shuffled", "unseen_sequence", "shuffled texts"),
              ("d32", "random", "unseen_sequence", "random tokens"),
              ("rand", "random", "seen_sequence_new_position", "new positions (ref.)"),
              ("rand", "random", "unseen_sequence", "unseen sequence r4"),
              ("rand", "natural", "unseen_sequence", "document tokens"),
              ("rand", "shuffled", "unseen_sequence", "shuffled texts")]
    for s in ("d32", "rand"):
        ref = setinfo[s]["reference"]
        claim(any(r[0] == s and r[1] == ref["panel"] and r[2] == ref["cohort"] for r in q4rows), f"the reference row of {s} is its own same-kind held-out cohort")
    rr, co = {m: [] for m, _ in models}, {m: [] for m, _ in models}
    for m, nm in models:
        for s, panel, coh, _ in q4rows:
            meas = pl[m, s][panel][coh][KP]
            src = f"units/{m}/{s}/tstar/facts.json placement.{panel}.{coh}.8"
            claim(not meas.get("cuts_band"), f"the top 8 modes of {m}/{s} are a stable space for placement")
            rr[m].append(meas["pooled_radius_ratio"])
            co[m].append(meas["pooled_centre_offset"])
            f2(f"q4.{m}.{s}.{panel}.{coh}.rr", meas["pooled_radius_ratio"], src + ".pooled_radius_ratio")
            f2(f"q4.{m}.{s}.{panel}.{coh}.co", meas["pooled_centre_offset"], src + ".pooled_centre_offset")
    for m, _ in models:
        claim(rr[m][2] < rr[m][0] and co[m][2] > co[m][0] and rr[m][1] < rr[m][0], "in d32 random and shuffled tokens sit nearer the centre than the reference, random ones off-centre")
        claim(co[m][5] > co[m][3] and co[m][5] > co[m][4], "in rand document tokens are more off-centre than random tokens of either kind")
    for m, _ in models:
        claim(all(0.9 < rr[m][i] < 1.1 and co[m][i] < 0.1 for i in (0, 3)), "tokens of the set's own kind land like the set (radius ratio within 10% of 1, little shift)")
        claim(0.9 < rr[m][5] < 1.1, "document tokens placed into rand keep about the usual radius")
    # how close a placed token's nearest construction token is (section 6 "support"), at the middle layer
    mid_l = (meta["layers"] - 1) // 2
    o.fact("q4.mid_layer", mid_l, source="the middle layer, (layers - 1) // 2")
    for m, _ in models:
        for s, panel, coh, _ in q4rows:
            v = pl[m, s][panel][coh]["nearest_cosine_median"][mid_l]
            f2(f"q4.near.{m}.{s}.{panel}.{coh}", v, f"units/{m}/{s}/tstar/facts.json placement.{panel}.{coh}.nearest_cosine_median[{mid_l}]")
        a = pl[m, "rand"]["natural"]["unseen_sequence"]["nearest_cosine_median"][mid_l]
        b = pl[m, "rand"]["random"]["seen_sequence_new_position"]["nearest_cosine_median"][mid_l]
        claim(a < b - 0.1, "in rand, document tokens have a less similar nearest construction token than new positions of rand's sequences")
    rowgroups = [{"from": 0, "label": f"into d32, built from {len(setinfo['d32']['contexts'])} texts"}, {"from": 3, "label": "into rand, built from random tokens"}]
    o.chart("q4", dict(cols=2, mobile_stack=True, panels=[
        dict(title="Radius, relative to the set's own tokens", sub="root mean square distance from the set's centre; 1 = the same as the set's own", rows=[r[3] for r in q4rows], rowgroups=rowgroups,
             xrange=[0, 1.2], fmt="2f", refline=dict(x=1, label="1"), legend=False, series=[dict(name=nm, color=COLOUR[m], values=rr[m]) for m, nm in models]),
        dict(title="Shift of the group's average", sub="distance of the group's mean from the set's centre, per layer, in units of the set's radius", rows=[r[3] for r in q4rows], rowgroups=rowgroups,
             xrange=[0, 0.7], fmt="2f", legend=False, series=[dict(name=nm, color=COLOUR[m], values=co[m]) for m, nm in models])]))

    # between-model agreement on the same rows (k = 16), beside the references
    q4a = [("d32", "construction", "in_set", "the set's own tokens"), *[(r[0], r[1], r[2], r[3]) for r in q4rows[:3]],
           ("rand", "construction", "in_set", "the set's own tokens"), *[(r[0], r[1], r[2], r[3]) for r in q4rows[3:]]]
    vals, lo_, hi_, mid, fr, weak = [], [], [], [], [], []
    for s, panel, coh, _ in q4a:
        r = perm(s, panel, coh)
        sc = r["scores"][K]
        claim(not sc["matched"]["cuts_band"], f"the first 16 modes of {s} are stable for the agreement of {panel}/{coh}")
        vals.append(sc["matched"]["stacked"])
        mid.append(sc["within_id_and_cohort"]["mean"][-1]); lo_.append(sc["within_id_and_cohort"]["minimum"][-1]); hi_.append(sc["within_id_and_cohort"]["maximum"][-1])
        fr.append(sc["free"]["mean"][-1])
        weak.append(r["within_id_uninformative"])
        tag = f"q4.agree.{s}.{panel}.{coh}"
        f2(tag + ".v", sc["matched"]["stacked"], psrc(s, panel, coh, "scores.16.matched.stacked"), 3)
        f2(tag + ".ref", sc["within_id_and_cohort"]["mean"][-1], psrc(s, panel, coh, "scores.16.within_id_and_cohort.mean[-1]"), 3)
        o.fact(tag + ".mov", r["movable_share"], pct=True, source=psrc(s, panel, coh, "movable_share"))
    o.chart("q4_agree", dict(rows=[r[3] for r in q4a], hollow_note="under 10% of this group can be reordered, so this reference says little",
                             rowgroups=[{"from": 0, "label": "in d32"}, {"from": 4, "label": "in rand"}], xrange=[0, 1], fmt="2f",
                             xlabel="overlap of the first 16 modes", series=[
                                 dict(name="the two models, same tokens", color="ink", values=vals),
                                 dict(name="reordered within each word piece (mean; line: range)", color="far", marker="diamond", values=mid, lo=lo_, hi=hi_, hollow=weak),
                                 dict(name="reordered freely", color="mute", marker="square", values=fr)]))
    claim(all(0 <= v - r < 0.03 for (s, *_), v, r in zip(q4a, vals, mid) if s == "rand"), "in rand every group sits at or just above its same-word-piece reference")
    claim(vals[0] - max(vals[2:4]) > 0.2, "in d32 the models agree much less on shuffled and random tokens than on the set's own")
    o.fact("q4.weak_count", sum(weak), source="rows of the q4_agree chart whose within_id_uninformative is true")
    o.fact("q4.weak_list", ", ".join(f"{r[3]} in {r[0]}" for r, w in zip(q4a, weak) if w) or "none",
           source="aggregates.json permutations.records[...].within_id_uninformative (under 10% of tokens can move)")
    # the spec's own comparison: the 992 natural-panel tokens outside c00, in rand and in d32
    for s in ("d32", "rand"):
        r = perm(s, "natural", "outside_doc00")
        sc = r["scores"][K]
        f2(f"q4.nat.{s}.v", sc["matched"]["stacked"], psrc(s, "natural", "outside_doc00", "scores.16.matched.stacked"))
        f2(f"q4.nat.{s}.ref", sc["within_id_and_cohort"]["mean"][-1], psrc(s, "natural", "outside_doc00", "scores.16.within_id_and_cohort.mean[-1]"))
        o.fact(f"q4.nat.{s}.n", r["count"], source=psrc(s, "natural", "outside_doc00", "count"))
        if s == "rand":
            claim(0 < sc["matched"]["stacked"] - sc["within_id_and_cohort"]["mean"][-1] < 0.05, "inside rand the natural panel is barely above its same-id reference")
    claim(perm("rand", "natural", "outside_doc00")["scores"][K]["matched"]["stacked"] < perm("d32", "natural", "outside_doc00")["scores"][K]["matched"]["stacked"],
          "the natural panel agrees less across models inside rand than inside d32")

    # ------------------------------------------------------------------ checks and caveats
    rf = agg["refit_agreement"]["records"]
    for m, _ in models:
        for k in ("8", "16", "64"):
            for part in ("placed", "shared"):
                f2(f"refit.{m}.{k}.{part}", rf[m][k][part], f"aggregates.json refit_agreement.records.{m}.{k}.{part}")
    o.fact("refit.cut_list", ", ".join(f"{name[m]} at {k}" for m, _ in models for k in ("8", "16", "64") if rf[m][k]["cuts_band"]) or "none",
           source="aggregates.json refit_agreement.records.<model>.<k>.cuts_band")
    claim(all(abs(rf[m][k]["placed"] - rf[m][k]["shared"]) < 0.05 for m, _ in models for k in ("8", "16", "64")), "placed tokens agree with the refit about as well as the shared tokens")
    claim(not any(rf[m]["16"]["cuts_band"] for m, _ in models), "the refit's 16-mode comparison is a stable space")
    other = [(m, k) for m, _ in models for k in ("8", "64")]
    o.fact("refit.other_cut", sum(rf[m][k]["cuts_band"] for m, k in other), source="aggregates.json refit_agreement.records.<model>.{8,64}.cuts_band, count")
    o.fact("refit.other_n", len(other), source="models times the two other mode counts")
    n = {k: 0 for k in ("8", "16", "64")}
    c = dict(n)
    for r in C:
        for k in n:
            sc = r["scores"][k]
            n[k] += 1
            c[k] += bool(sc.get("cuts_band"))
    for k in n:
        o.fact(f"bands.k{k}.cut", c[k], source=f"aggregates.json cross.records[*].scores.{k}.cuts_band, count (both strengths, all kinds)")
        o.fact(f"bands.k{k}.n", n[k], source=f"aggregates.json cross.records[*].scores.{k}, count")
    claim(c["64"] > n["64"] / 2, "a cutoff at 64 modes is flagged in most comparisons")
    for t in (T, T16):
        undefined, total, where = 0, 0, set()
        for r in C:
            if r["kind"] != "models" or r["strength"] != t or "unavailable" in r["scores"][K]:
                continue
            lv = r["scores"][K]["layers"]
            nul = sum(v is None for v in lv)
            undefined, total = undefined + nul, total + len(lv)
            if nul:
                where.add(r["set"])
        src = f"aggregates.json cross.records[kind=models, strength={t}].scores.16.layers"
        o.fact(f"undef.{t}.count", undefined, source=src + ", null entries (a rank below 16 at that layer)")
        o.fact(f"undef.{t}.total", total, source=src + ", all entries")
        ws = [s for s in sets if s in where]
        o.fact(f"undef.{t}.sets", ", ".join(ws[:-1]) + (" and " if len(ws) > 1 else "") + (ws[-1] if ws else "none"), source="the sets of the records with a null entry")
        o.fact(f"undef.{t}.nsets", len(ws), source="the number of sets of the records with a null entry")
        if t == T:
            claim(where == {"doc02", "pairB"}, "at t* the undefined per-layer overlaps are exactly in doc02 and pairB")
        else:
            claim(undefined > total / 4, "at t*/16 a large share of per-layer overlaps are undefined")
    for m, nm in models:
        ts = sorted(((meta["units"][f"{m}/{s}"]["tstar"], s) for s in sets))
        f2(f"tstar.{m}.min", ts[0][0], f"meta.json units.{m}/{ts[0][1]}.tstar (the smallest)")
        o.fact(f"tstar.{m}.min_set", ts[0][1], source="the set of the smallest t*")
        o.fact(f"tstar.{m}.second_set", ts[1][1], source="the set of the second smallest t*")
        f2(f"tstar.{m}.max", ts[-1][0], f"meta.json units.{m}/{ts[-1][1]}.tstar (the largest)", 1)
        claim({ts[0][1], ts[1][1]} == {"doc02", "pairB"}, "the two smallest crossing strengths are those of doc02 and pairB, in both models")
    o.chart("card", dict(answer="Two models build similar spaces on the same tokens; much of the overlap survives reordering occurrences within each word piece.",
                         numbers=["{{fact:x.d1-depth.q3.own.min}}–{{fact:x.d1-depth.q3.own.max}} between models",
                                  "{{fact:x.d1-depth.q3.sameid.min}}–{{fact:x.d1-depth.q3.sameid.max}} after reordering within word pieces (document sets)",
                                  "{{fact:x.d1-depth.q3.free.max}} at most after free reordering (mean of {{fact:x.d1-depth.draws}})"]))
    o.write()


if __name__ == "__main__":
    import sys
    run_depth(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
