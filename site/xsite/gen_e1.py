"""E1: who are a token's nearest neighbours under memory? Reads analysis/geometry (census summaries written by
exp1/census.py), analysis/<model>_timescales.json and the stitched and synthetic census rows. Writes chart payloads
and facts for content/experiments/e1-neighbours.html. Nothing is recomputed except means of stored values.

    python site/xsite/gen_e1.py DATA_DIR OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from xlib import ACC_LABEL, ACCOUNTINGS, Data, MEMORIES, MEMORY_LABELS, ORIGINS, Out

XID = "e1-neighbours"
LEAD_SITE = 8


def stack_panels(d, datasets, reading, field, site_want, *, accountings=ACCOUNTINGS, whiskers=True, removed=True, title=None):
    """One panel per (dataset, accounting): a stacked bar per memory of the three origin shares in `field`."""
    panels, sources = [], []
    for ds, ds_label in datasets:
        site = d.site_or_nearest(ds, site_want, reading, "C")
        for centring, mask in accountings:
            values, lo, hi, notes, src = [], [], [], [], None
            for mem in MEMORIES:
                row, where = d.geo_row(ds, reading, centring, f"site{site:02d}" if site is not None else "", mem, mask) if site is not None else (None, None)
                v = row.get(field) if row else None
                values.append(None if v is None else [float(x) for x in v[:3]])
                bs = (row or {}).get("origin_document_bootstrap") if whiskers else None
                lo.append(bs["p5"][0] if bs and bs.get("p5") else None)
                hi.append(bs["p95"][0] if bs and bs.get("p95") else None)
                notes.append(f"{100 * row['screen_removed']:.0f}%" if removed and mask == "window" and row and row.get("screen_removed") is not None else None)
                src = src or where
            layer = "no data" if site is None else f"layer {site}" + ("" if site == site_want else f" ({site_want} not computed)")
            panels.append(dict(title=f"{ds_label} · {ACC_LABEL[centring, mask]}", sub=layer, values=values,
                               whisker=[[a, b] if a is not None else None for a, b in zip(lo, hi)] if whiskers else None,
                               zero_near=mask == "window", notes=notes if any(notes) else None,
                               note_label="removed" if any(notes) else None))
            sources.append(src)
    return dict(categories=MEMORY_LABELS, xlabel="memory", segments=ORIGINS, segcolors=["other", "far", "near"],
                ylabel="share of the 64 neighbours", cols=len(accountings), panels=panels, title=title), sources


def run(data: Path, out: Path):
    d = Data(data)
    o = Out(XID, out)
    models = [(m, d.model_name(m)) for m in d.models]

    # lead: site 8, cosine, census pool
    lead, _ = stack_panels(d, models, "cosine", "origin", LEAD_SITE)
    o.chart("lead", lead)
    screen, _ = stack_panels(d, models, "cosine", "screen_mass", LEAD_SITE, whiskers=False)
    screen["ylabel"] = "share of the walk weight (suffix sample)"
    o.chart("screen", screen)
    dot, _ = stack_panels(d, models, "dot", "origin", LEAD_SITE)
    o.chart("dot", dot)

    # the same kernel on text-only inputs, beside the models (mean over all measured layers); corpus mean, no mask
    tpanels = []
    for m, name in models:
        sites = d.sites(m)
        vals = []
        for mem in MEMORIES:
            rs = [d.geo_row(m, "cosine", "C", f"site{s:02d}", mem, "none")[0] for s in sites]
            rs = [r["origin"] for r in rs if r and r.get("origin")]
            vals.append(np.mean(rs, axis=0).tolist() if rs else None)
        tpanels.append(dict(title=f"{name}, its layers", sub=f"mean of {len(sites)} layers", values=vals, whisker=None, zero_near=False))
        o.fact(f"m{d.models.index(m) + 1}.layers", len(sites), source=f"analysis/geometry/{m}/cosine_C.json inputs site*")
        for mem, v in zip(MEMORIES, vals):
            o.fact(f"m{d.models.index(m) + 1}.layermean.mem_{mem}.near", v[2] if v else None, pct=True,
                   source=f"analysis/geometry/{m}/cosine_C.json origin[2], mask none, mean over inputs site*")
            o.fact(f"m{d.models.index(m) + 1}.layermean.mem_{mem}.own", v[1] + v[2] if v else None, pct=True,
                   source=f"analysis/geometry/{m}/cosine_C.json origin[1]+origin[2], mask none, mean over inputs site*")
    for inp, label in (("random", "random token vectors"), ("embedding", "embedding output")):
        for m, name in models:
            if inp == "random" and m != models[0][0]:
                continue                               # the same vectors for both models (one per token id)
            vals = [(d.geo_row(m, "cosine", "C", inp, mem, "none")[0] or {}).get("origin") for mem in MEMORIES]
            tpanels.append(dict(title=label if inp == "random" else f"{label}, {name}", sub="same kernel, same texts", values=vals, whisker=None, zero_near=False))
            key = "random" if inp == "random" else f"emb{d.models.index(m) + 1}"
            for mem, v in zip(MEMORIES, vals):
                o.fact(f"{key}.mem_{mem}.near", v[2] if v else None, pct=True, source=f"analysis/geometry/{m}/cosine_C.json input {inp}, origin[2], mask none")
                o.fact(f"{key}.mem_{mem}.other", v[0] if v else None, pct=True, source=f"analysis/geometry/{m}/cosine_C.json input {inp}, origin[0], mask none")
    o.chart("textonly", dict(categories=MEMORY_LABELS, xlabel="memory", segments=ORIGINS, segcolors=["other", "far", "near"],
                             ylabel="share of the 64 neighbours", cols=len(tpanels), panels=tpanels))

    # adversary overlap and overlap with memory none, corpus mean, no mask
    panels = []
    for m, name in models:
        site = d.site_or_nearest(m, LEAD_SITE)
        rows = [d.geo_row(m, "cosine", "C", f"site{site:02d}", mem, "none")[0] if site else None for mem in MEMORIES]
        get = lambda i: [r["overlap_expected"][i] if r and len(r.get("overlap_expected", [])) > i else None for r in rows]
        panels.append(dict(title=name, sub=f"layer {site} · corpus mean", x=list(range(5)), xticklabels=MEMORY_LABELS, xlabel="memory",
                           ylabel="neighbours shared (of 64)", yrange=[0, 1], fmt="pct",
                           series=[dict(name="random token vectors", y=get(1), dots=True, color="ink"),
                                   dict(name="embedding output", y=get(2), dots=True, color="ink", dash=True),
                                   dict(name="same model, memory none", y=get(0), dots=True, color="mute", dash="dot")]))
    o.chart("adversary", dict(panels=panels, shared=False))

    # same current token, by mask (other-texts-only lives here)
    panels = []
    for m, name in models:
        site = d.site_or_nearest(m, LEAD_SITE)
        series = []
        for mask, label, dash in (("none", "every token eligible", None), ("window", "window", True), ("other_texts", "other texts only", "dot")):
            ys = [(d.geo_row(m, "cosine", "C", f"site{site:02d}", mem, mask)[0] or {}).get("same_token") if site else None for mem in MEMORIES]
            series.append(dict(name=label, y=ys, dots=True, color="ink", dash=dash))
        chance = [(d.geo_row(m, "cosine", "C", f"site{site:02d}", mem, "none")[0] or {}).get("chance_same_token") if site else None for mem in MEMORIES]
        series.append(dict(name="chance (every token eligible)", y=chance, color="mute2", width=1.5))
        panels.append(dict(title=name, sub=f"layer {site} · corpus mean", x=list(range(5)), xticklabels=MEMORY_LABELS, xlabel="memory",
                           ylabel="neighbours with the same current token", yrange=[0, 1], fmt="pct", series=series))
    o.chart("same_token", dict(panels=panels))

    # stitched controls: share beyond position
    panels, beyond = [], {}
    for m, name in models:
        series = []
        for n_c, c in enumerate(d.stitched()):
            ds = f"{m}_stitch{c:02d}"
            site = d.site_or_nearest(ds, LEAD_SITE)
            schedule = d.asset(next(t for t in d.manifest["texts"] if t["id"] == c)["text_file"]).get("schedule", "")
            ys = [(d.geo_row(ds, "cosine", "C", f"site{site:02d}", mem, "none")[0] or {}).get("stitch_beyond") if site else None for mem in MEMORIES]
            series.append(dict(name=f"text {c} ({schedule} segments)", y=ys, dots=True, dash=True if schedule == "short" else None,
                               color="ink" if n_c < 2 else "mute"))
            for mem, y in zip(MEMORIES, ys):
                beyond.setdefault((m, mem), []).append(y)
        panels.append(dict(title=name, sub="corpus mean, every token eligible", x=list(range(5)), xticklabels=MEMORY_LABELS, xlabel="memory",
                           ylabel="same-source share minus position baseline", fmt="pct", refline=dict(y=0, label="position alone"),
                           series=series))
    o.chart("stitched", dict(panels=panels))

    # synthetic census: planted regimes as neighbours (corpus mean, no mask, last site, mean over seeds)
    variants = ["strong", "planted", "drift", "noise_a", "noise_b"]
    vlabel = {"strong": "strong regime", "planted": "weak regime", "drift": "position drift", "noise_a": "noise field A", "noise_b": "noise field B"}
    geo_dir = d.root / "analysis" / "geometry"
    syn_panels = []
    for v in variants:
        names = sorted(p.name for p in geo_dir.glob(f"synthetic_{v}_seed*")) if geo_dir.is_dir() else []
        if not names:
            continue
        site = max(d.sites(names[0]))
        values = []
        for mem in MEMORIES:
            rs = [d.geo_row(n, "cosine", "C", f"site{site:02d}", mem, "none")[0] for n in names]
            rs = [r["origin"] for r in rs if r and r.get("origin")]
            values.append(np.mean(rs, axis=0).tolist() if rs else None)
        syn_panels.append(dict(title=vlabel[v], sub=f"{len(names)} seed{'s' * (len(names) != 1)} · site {site}", values=values, whisker=None, zero_near=False))
    done = d.root / "analysis" / "synthetic" / "_done.json"
    groups = __import__("json").loads(done.read_text()).get("groups") if done.is_file() else None
    o.fact("synthetic.groups", "; ".join(f"{r} reading, {'corpus' if c == 'C' else 'own-text'} mean" for r, c in groups) if groups else "not recorded",
           source="analysis/synthetic/_done.json groups")
    o.fact("synthetic.runs", len(list(geo_dir.glob("synthetic_*"))) if geo_dir.is_dir() else 0, source="analysis/geometry/synthetic_*")
    o.chart("synthetic", dict(categories=MEMORY_LABELS, xlabel="memory", segments=ORIGINS, segcolors=["other", "far", "near"],
                              ylabel="share of the 64 neighbours", cols=min(5, max(1, len(syn_panels))), panels=syn_panels))

    # timescales (section 10): attention distance per layer and residual half-similarity lag per site
    panels = []
    for m, name in models:
        path = d.root / "analysis" / f"{m}_timescales.json"
        if not path.is_file():
            continue
        ts = __import__("json").loads(path.read_text())
        layers = [a["layer"] for a in ts["attention"]]
        x = sorted(set(layers) | {int(s) for s in ts["residual"]})
        att = {a["layer"]: a["distance"]["quartiles"] for a in ts["attention"]}
        med = [att.get(l, [None] * 3)[1] for l in x]
        band = [[att.get(l, [None] * 3)[0] for l in x], [att.get(l, [None] * 3)[2] for l in x]]
        res = {c: [ts["residual"].get(str(l), {}).get(c, {}).get("half_lag") for l in x] for c in ("C", "O")}
        panels.append(dict(title=name, x=x, xlabel="layer", ylabel="tokens (log scale)", yscale="log", fmt="3g",
                           reflines=[dict(y=2, label="2"), dict(y=8, label="8"), dict(y=64, label="64")],
                           series=[dict(name="attention distance, median over heads", y=med, band=band, dots=True, join=True, color="ink"),
                                   dict(name="residual half-similarity lag, corpus mean", y=res["C"], dots=True, join=True, color="mute"),
                                   dict(name="residual half-similarity lag, own-text mean", y=res["O"], dots=True, join=True, dash=True, color="mute")]))
        o.fact(f"m{d.models.index(m) + 1}.censored", int(ts["censored_count"]["C"]), source=f"analysis/{m}_timescales.json censored_count.C")
    o.chart("timescales", dict(panels=panels))

    # facts: per model (m1, m2 in manifest order), corpus mean / window / own-text at the lead layer
    for i, (m, name) in enumerate(models, 1):
        k = f"m{i}"
        site = d.site_or_nearest(m, LEAD_SITE)
        o.fact(f"{k}.name", name, source="site/manifest.json model_metadata")
        o.fact(f"{k}.layer", site if site is not None else "none", source="analysis/geometry: the lead layer or the nearest computed")
        for mem in MEMORIES:
            mk = f"mem_{mem}"
            for (centring, mask), ak in ((("C", "none"), "c"), (("C", "window"), "cw"), (("O", "none"), "o"), (("O", "window"), "ow")):
                row, where = d.geo_row(m, "cosine", centring, f"site{site:02d}", mem, mask) if site else (None, None)
                src = f"{where} origin" if where else "absent"
                orig = row.get("origin") if row else None
                o.fact(f"{k}.{ak}.{mk}.other", orig[0] if orig else None, pct=True, source=src)
                o.fact(f"{k}.{ak}.{mk}.own", orig[1] + orig[2] if orig else None, pct=True, source=src + " [1]+[2]")
                o.fact(f"{k}.{ak}.{mk}.near", orig[2] if orig else None, pct=True, source=src)
                o.fact(f"{k}.{ak}.{mk}.chance_other", (row or {}).get("chance_origin", [None])[0], pct=True, source=src.replace("origin", "chance_origin"))
                o.fact(f"{k}.{ak}.{mk}.gap", (row or {}).get("position_gap"), digits=0, source=src.replace("origin", "position_gap"))
                mass = (row or {}).get("screen_mass")
                o.fact(f"{k}.{ak}.{mk}.screen_other", mass[0] if mass else None, pct=True, source=src.replace("origin", "screen_mass[0]"))
                if mask == "window":
                    o.fact(f"{k}.{ak}.{mk}.removed", (row or {}).get("screen_removed"), pct=True, source=src.replace("origin", "screen_removed"))
                if (centring, mask) == ("C", "none"):
                    ov = (row or {}).get("overlap_expected")
                    o.fact(f"{k}.{ak}.{mk}.adv", max(ov[1], ov[2]) if ov and len(ov) > 2 else None, pct=True,
                           source=src.replace("origin", "max(overlap_expected[1], overlap_expected[2])"))
                    o.fact(f"{k}.{ak}.{mk}.local", ov[0] if ov else None, pct=True, source=src.replace("origin", "overlap_expected[0]"))
                    bs = (row or {}).get("origin_document_bootstrap")
                    o.fact(f"{k}.{ak}.{mk}.other_p5", bs["p5"][0] if bs and bs.get("p5") else None, pct=True, source=src + " bootstrap p5")
                    o.fact(f"{k}.{ak}.{mk}.other_p95", bs["p95"][0] if bs and bs.get("p95") else None, pct=True, source=src + " bootstrap p95")
            ys = [y for y in beyond.get((m, mem), []) if y is not None]
            o.fact(f"{k}.stitch.{mk}", float(np.mean(ys)) if ys else None, pct=True,
                   source=f"analysis/geometry/{m}_stitchNN/cosine_C.json stitch_beyond, mean over the stitched texts")
    o.fact("stitched_texts", len(d.stitched()), source="site/manifest.json texts (kind stitched)")
    o.fact("models", len(models), source="site/manifest.json models")
    o.chart("card", dict(answer=("With memory, a token's nearest neighbours move into its own text and next to it "
                                 "({{fact:x.e1-neighbours.m1.c.mem_none.own}} to {{fact:x.e1-neighbours.m1.c.mem_64.own}} in its own text, memory none to 64); "
                                 "random token vectors through the same kernel mostly do not."), numbers=[
        "Own text at memory 64: {{fact:x.e1-neighbours.m1.c.mem_64.own}} ({{fact:x.e1-neighbours.m1.name}})",
        "Within 64 positions at memory 64: {{fact:x.e1-neighbours.m1.c.mem_64.near}}",
        "Within 64 positions at memory 64, random token vectors: {{fact:x.e1-neighbours.random.mem_64.near}}"]))
    o.write()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
