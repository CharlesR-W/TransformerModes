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
        elig = [r for r in rows if r["reading"] == "cosine" and r["site"] == site]
        o.fact(f"{k}.eligible_count", sum(bool(r["eligible"]) for r in elig), source=f"analysis/prediction/{m}.json eligible, cosine, lead layer")
        o.fact(f"{k}.settings_count", len(elig), source=f"analysis/prediction/{m}.json settings, cosine, lead layer")
    o.chart("lead", dict(panels=lead))
    o.chart("coverage", dict(panels=cov_panels))
    o.chart("top1", dict(panels=top_panels))
    o.chart("card", dict(answer="On hold: the independent audit failed its check of this readout; not yet verified.", placeholder=True, numbers=[
        "Score at memory none: {{fact:x.e2-concordance.m1.c.mem_none.score}} ({{fact:x.e2-concordance.m1.name}})",
        "Score at memory 64: {{fact:x.e2-concordance.m1.c.mem_64.score}}",
        "Coverage at memory 64: {{fact:x.e2-concordance.m1.c.mem_64.coverage}}"]))
    o.write()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
