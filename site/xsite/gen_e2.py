"""E2: are neighbours prediction-concordant beyond matched controls? Reads analysis/prediction/<model>.json
(exp1/readout.py summaries: concordance score, coverage, bootstrap percentiles, strata). Writes chart payloads and
facts for content/experiments/e2-concordance.html.

    python site/xsite/gen_e2.py DATA_DIR OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

from xlib import ACC_LABEL, ACCOUNTINGS, Data, MEMORIES, MEMORY_LABELS, Out, read_json

XID = "e2-concordance"
LEAD_SITE = 8
STRATA = [("other/rare", "other texts · rare token"), ("other/frequent", "other texts · frequent token"),
          ("far/rare", "same text far · rare"), ("far/frequent", "same text far · frequent"),
          ("near/rare", "within 64 · rare"), ("near/frequent", "within 64 · frequent")]


def run(data: Path, out: Path):
    d = Data(data)
    o = Out(XID, out)
    lead, cov_panels, top_panels = [], [], []
    for i, m in enumerate(d.models, 1):
        name = d.model_name(m)
        path = d.root / "analysis" / "prediction" / f"{m}.json"
        rows = read_json(path)["settings"] if path.is_file() else []
        sites = sorted({r["site"] for r in rows if r["reading"] == "cosine"})
        site = min(sites, key=lambda s: (abs(s - LEAD_SITE), s)) if sites else None
        idx = {(r["reading"], r["centring"], r["site"], r["memory"], r["mask"]): (j, r) for j, r in enumerate(rows)}
        get = lambda c, mk, mem: idx.get(("cosine", c, site, mem, mk), (None, None))
        layer = "no readout" if site is None else f"layer {site}" + ("" if site == LEAD_SITE else f" ({LEAD_SITE} not computed)")
        series = []
        for c, mk in ACCOUNTINGS:
            vals, lo, hi, hollow = [], [], [], []
            for mem in MEMORIES:
                _, r = get(c, mk, mem)
                bp = (r or {}).get("bootstrap_percentiles")
                vals.append((r or {}).get("score"))
                lo.append(bp[0] if bp else None)
                hi.append(bp[2] if bp else None)
                hollow.append(bool(r) and not r.get("eligible"))
            series.append(dict(name=ACC_LABEL[c, mk], values=vals, lo=lo, hi=hi, hollow=hollow,
                               color="corpus" if c == "C" else "own", marker="circle" if mk == "none" else "diamond"))
        lead.append(dict(title=name, sub=layer, categories=MEMORY_LABELS, xlabel="memory", ylabel="concordance score",
                         refline=dict(y=0, label="no advantage"), series=series, fmt="3f"))
        # coverage by stratum (corpus mean, no mask)
        z = []
        for key, _ in STRATA:
            z.append([((get("C", "none", mem)[1] or {}).get("strata", {}).get(key) or {}).get("coverage") for mem in MEMORIES])
        cov_panels.append(dict(title=name, sub=f"{layer} · corpus mean", rows=[s for _, s in STRATA], cols=MEMORY_LABELS, z=z, zrange=[0, 1],
                               zlabel="share of pairs with a matched control", rowlabel="", collabel="memory", fmt="pct", ml=170))
        top_panels.append(dict(title=name, sub=f"{layer} · corpus mean", x=list(range(5)), xticklabels=MEMORY_LABELS, xlabel="memory",
                               ylabel="same top-1 prediction", fmt="pct",
                               series=[dict(name="neighbour", y=[(get("C", "none", mem)[1] or {}).get("top1_neighbour") for mem in MEMORIES], dots=True, color="ink"),
                                       dict(name="its matched control", y=[(get("C", "none", mem)[1] or {}).get("top1_control") for mem in MEMORIES], dots=True, dash=True, color="ink")]))
        k = f"m{i}"
        o.fact(f"{k}.name", name, source="site/manifest.json model_metadata")
        o.fact(f"{k}.layer", site if site is not None else "none", source=f"analysis/prediction/{m}.json: lead layer or nearest")
        for c, mk in ACCOUNTINGS:
            ak = {"C": "c", "O": "o"}[c] + ("w" if mk == "window" else "")
            for mem in MEMORIES:
                j, r = get(c, mk, mem)
                src = f"analysis/prediction/{m}.json settings[{j}]" if r else "absent"
                bp = (r or {}).get("bootstrap_percentiles")
                base = f"{k}.{ak}.mem_{mem}"
                o.fact(f"{base}.score", (r or {}).get("score"), digits=3, source=src + " score")
                o.fact(f"{base}.p5", bp[0] if bp else None, digits=3, source=src + " bootstrap_percentiles[0]")
                o.fact(f"{base}.p95", bp[2] if bp else None, digits=3, source=src + " bootstrap_percentiles[2]")
                o.fact(f"{base}.coverage", (r or {}).get("coverage"), pct=True, source=src + " coverage")
                o.fact(f"{base}.eligible", ("eligible" if r.get("eligible") else "ineligible") if r else None, source=src + " eligible")
                if (c, mk) == ("C", "none"):
                    o.fact(f"{base}.top1", (r or {}).get("top1_neighbour"), pct=True, source=src + " top1_neighbour")
                    o.fact(f"{base}.top1_control", (r or {}).get("top1_control"), pct=True, source=src + " top1_control")
        # means over all readout layers (cosine, corpus mean, no mask), and how often the within-64 strata were covered
        cn = [r for r in rows if r["reading"] == "cosine" and r["centring"] == "C" and r["mask"] == "none"]
        lsites = sorted({r["site"] for r in cn})
        o.fact(f"{k}.readout_layers", ", ".join(map(str, lsites)) if lsites else "none", source=f"analysis/prediction/{m}.json sites")
        for mem in MEMORIES:
            rs = [r for r in cn if r["memory"] == mem]
            src = f"analysis/prediction/{m}.json cosine C none memory {mem}, mean over layers {lsites}"
            mean = lambda key: (sum(r[key] for r in rs if r.get(key) is not None) / len([r for r in rs if r.get(key) is not None])) if any(r.get(key) is not None for r in rs) else None
            sc, t1, t0 = mean("score"), mean("top1_neighbour"), mean("top1_control")
            o.fact(f"{k}.mean.mem_{mem}.score", sc, digits=3, source=src + " score")
            o.fact(f"{k}.mean.mem_{mem}.top1", t1, pct=True, source=src + " top1_neighbour")
            o.fact(f"{k}.mean.mem_{mem}.top1_control", t0, pct=True, source=src + " top1_control")
            o.fact(f"{k}.mean.mem_{mem}.ratio", (t1 / t0) if t1 is not None and t0 else None, digits=1, source=src + " top1_neighbour / top1_control")
            o.fact(f"{k}.mean.mem_{mem}.coverage", mean("coverage"), pct=True, source=src + " coverage")
            pairs = sum(v["pairs"] for r in rs for v in r["strata"].values())
            near = sum(v["pairs"] for r in rs for key, v in r["strata"].items() if key.startswith("near/") and v["status"] != "covered")
            o.fact(f"{k}.mean.mem_{mem}.near_left_out", near / pairs if pairs else None, pct=True,
                   source=src + " strata near/*: pairs in uncovered strata over all pairs")
        near_cov = sum(r["strata"][s]["status"] == "covered" for r in cn for s in ("near/rare", "near/frequent"))
        o.fact(f"{k}.near_covered", near_cov, source=f"analysis/prediction/{m}.json cosine C none: covered near strata")
        o.fact(f"{k}.near_strata", 2 * len(cn), source=f"analysis/prediction/{m}.json cosine C none: near strata")
        o.fact(f"{k}.all_eligible", sum(bool(r["eligible"]) for r in rows), source=f"analysis/prediction/{m}.json eligible, every setting")
        o.fact(f"{k}.all_settings", len(rows), source=f"analysis/prediction/{m}.json settings")
        elig = [r for r in rows if r["reading"] == "cosine" and r["site"] == site]
        o.fact(f"{k}.eligible_count", sum(bool(r["eligible"]) for r in elig), source=f"analysis/prediction/{m}.json eligible, cosine, lead layer")
        o.fact(f"{k}.settings_count", len(elig), source=f"analysis/prediction/{m}.json settings, cosine, lead layer")
    o.chart("lead", dict(panels=lead))
    o.chart("coverage", dict(panels=cov_panels))
    o.chart("top1", dict(panels=top_panels))
    o.chart("card", dict(answer=("With no memory a token's neighbours share the model's top prediction {{fact:x.e2-concordance.m1.mean.mem_none.ratio}} times "
                                 "as often as matched controls; the advantage shrinks with memory to about nothing at the whole prefix."), numbers=[
        "Same top prediction, no memory: {{fact:x.e2-concordance.m1.mean.mem_none.top1}} vs {{fact:x.e2-concordance.m1.mean.mem_none.top1_control}} for controls ({{fact:x.e2-concordance.m1.name}})",
        "Score, no memory → whole prefix: {{fact:x.e2-concordance.m1.mean.mem_none.score}} → {{fact:x.e2-concordance.m1.mean.mem_whole.score}}",
        "Pairs within 64 positions left out (no covered control) at memory 64: {{fact:x.e2-concordance.m1.mean.mem_64.near_left_out}}"]))
    o.write()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
