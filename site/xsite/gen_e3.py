"""E3: what does the screen select for Stage B, and why? Reads analysis/stage_b_selection.json (exp1/select_b.py)
and the suffix-screen rows of analysis/geometry. Shows stored decisions; never re-ranks.

    python site/xsite/gen_e3.py DATA_DIR OUT_DIR
"""
from __future__ import annotations

import sys
from pathlib import Path

from xlib import ACC_LABEL, Data, ORIGINS, Out, read_json

XID = "e3-selection"
LEAD_SITE = 8
COND = [("other_mass", "1 elsewhere"), ("short_mass", "2 not nearby"), ("local", "3 memory changes it"),
        ("random", "4 not random vectors"), ("embedding", "4 not embedding")]
MEM_WORD = {"none": "none", "2": "2", "8": "8", "64": "64", "whole": "whole"}


def describe(c):
    if not c:
        return "none"
    return f"memory {MEM_WORD[c['memory']]}, {c['reading']} reading, {ACC_LABEL[c['centring'], c['mask']]}" + (", no mask" if c["mask"] == "none" else "")


def run(data: Path, out: Path):
    d = Data(data)
    o = Out(XID, out)
    path = d.root / "analysis" / "stage_b_selection.json"
    sel = read_json(path) if path.is_file() else None
    dropped = set(d.dropped())
    if sel is None:
        o.chart("table", dict(columns=[dict(label="Construction")], rows=[], note="No selection file in this data directory."))
        o.chart("lead", dict(categories=[], segments=ORIGINS, panels=[]))
        o.fact("recommend", "not available", source="analysis/stage_b_selection.json absent")
        for k in ("m", "l", "x", "x_reason", "passing", "required", "sites"):
            o.fact(k, "not available", source="analysis/stage_b_selection.json absent")
        o.chart("card", dict(answer="No selection file yet.", placeholder=True, numbers=["—", "—", "—"]))
        o.write()
        return
    models = [m for m in d.models]
    sites = sel["sites"]
    roles = {}
    for role in ("M", "L", "R", "X"):
        if sel.get(role):
            roles.setdefault(tuple(sorted(sel[role].items())), []).append(role)
    columns = [dict(label="Construction")] + [dict(label=d.model_name(m), sub=f"sites passing all four conditions, of {len(sites)}") for m in models] + [dict(label="Result")]
    rows = []
    for i, row in enumerate(sel["table"]):
        c = row["construction"]
        cells = [dict(text=describe(c))]
        group_dropped = (c["reading"], c["centring"]) in dropped
        for m in models:
            mm = row["models"].get(m)
            if mm is None or group_dropped:
                cells.append(dict(text="dropped by the cap" if group_dropped else "—", muted=True))
                continue
            observed = sum(v is not None for v in mm["sites"].values())
            counts = mm["condition_site_counts"]
            cells.append(dict(text=f"{mm['passing']} of {len(sites)}", strong=mm["passed"],
                              small=" · ".join(f"{label}: {counts.get(key, 0)}" for key, label in COND),
                              note=None if mm["complete"] else f"only {observed} of {len(sites)} sites computed"))
        tags = roles.get(tuple(sorted(c.items())), [])
        result = "passed" if row["passed"] else ("not computed" if group_dropped else "did not pass")
        if row.get("primary_rank"):
            result += f" · rank {row['primary_rank']}"
        if tags:
            result += " · chosen as " + " and ".join(tags)
        worst = row.get("worst_adversary_overlap")
        cells.append(dict(text=result, strong=bool(tags), tag=bool(tags),
                          small=None if worst is None else f"worst text-only overlap {100 * worst:.1f}% (ranking input)"))
        rows.append(dict(cells=cells, source=f"analysis/stage_b_selection.json table[{i}]", cls="chosen" if tags else ("muted" if group_dropped else "")))
    o.chart("table", dict(columns=columns, rows=rows,
                          note=f"A construction passes if, for every model, at least {sel['required_sites']} of the {len(sites)} layers pass all conditions at once. "
                               "The small line counts, per condition (numbered as in What was computed), the layers where that condition alone held."))
    # every per-model, per-layer value the conditions and the ranking used, as stored
    pc = lambda v: "—" if v is None else f"{100 * v:.1f}%"
    srows = []
    for i, row in enumerate(sel["table"]):
        c = row["construction"]
        for m in models:
            for site in sites:
                v = ((row["models"].get(m) or {}).get("sites") or {}).get(str(site))
                if v is None:
                    srows.append(dict(cells=[dict(text=describe(c)), dict(text=d.model_name(m)), dict(text=str(site))] + [dict(text="not computed", muted=True)] + [dict(text="")] * 5,
                                      cls="muted", source=f"analysis/stage_b_selection.json table[{i}].models.{m}.sites.{site}"))
                    continue
                qm, mo, eo, cond = v.get("query_mass") or {}, v.get("maximum_overlap") or {}, v.get("expected_overlap") or {}, v.get("conditions") or {}
                mark = lambda k: "✓" if cond.get(k) else "✗"
                near = f"removed {pc(v.get('removed'))}" if c["mask"] == "window" else f"near {pc(qm.get('near'))}"
                srows.append(dict(cells=[dict(text=describe(c)), dict(text=d.model_name(m)), dict(text=str(site)),
                                         dict(text=f"{pc(qm.get('other'))} {mark('other_mass')}"), dict(text=f"{near} {mark('short_mass')}"),
                                         dict(text=f"{pc(mo.get('local'))} {mark('local')}", small=f"expected {pc(eo.get('local'))}"),
                                         dict(text=f"{pc(mo.get('random'))} {mark('random')}", small=f"expected {pc(eo.get('random'))}"),
                                         dict(text=f"{pc(mo.get('embedding'))} {mark('embedding')}", small=f"expected {pc(eo.get('embedding'))}"),
                                         dict(text="passes all" if v.get("passed") else "—", strong=bool(v.get("passed")),
                                              small=f"{v.get('rows', '?')} rows; status {v.get('status_counts')}")],
                                  source=f"analysis/stage_b_selection.json table[{i}].models.{m}.sites.{site}"))
    o.chart("sites", dict(columns=[dict(label="Construction"), dict(label="Model"), dict(label="Layer"), dict(label="1 elsewhere", sub="walk weight on other texts"),
                                   dict(label="2 not nearby", sub="window: removed weight; else weight within 64"),
                                   dict(label="3 vs memory none", sub="largest overlap ties allow"), dict(label="4 vs random vectors", sub="largest overlap"),
                                   dict(label="4 vs embedding", sub="largest overlap"), dict(label="Layer result", sub="rows used; screen status counts")],
                          rows=srows, note="✓ condition held at that layer, ✗ it did not. Status counts: calibrated, too few eligible, tied minima, unbracketed rows."))
    # lead: suffix-screen walk-weight shares for R, L, M at the lead layer
    panels = []
    picks = [(r, sel.get(r)) for r in ("R", "L", "M") if sel.get(r)]
    for m in models:
        site = d.site_or_nearest(m, LEAD_SITE, "cosine", "C")
        values, alts = [], []
        for role, c in picks:
            row, _ = d.geo_row(m, c["reading"], c["centring"], f"site{site:02d}", c["memory"], c["mask"]) if site else (None, None)
            values.append([float(x) for x in row["screen_mass"][:3]] if row and row.get("screen_mass") else None)
            alts.append([float(x) for x in row["screen_origin"][:3]] if row and row.get("screen_origin") else None)
        panels.append(dict(title=d.model_name(m), sub=f"layer {site}" + ("" if site == LEAD_SITE else f" ({LEAD_SITE} not computed)"),
                           values=values, alt=alts, alt_label="top-64 shares on the same rows", zero_near=False, whisker=None))
    o.chart("lead", dict(categories=[f"{r}: {describe(c)}" for r, c in picks], short=[r for r, _ in picks], xlabel="construction",
                         segments=ORIGINS, segcolors=["other", "far", "near"], ylabel="share of the walk weight (suffix sample)",
                         cols=len(models), panels=panels))
    o.fact("recommend", "a construction passed" if sel["recommend_stage_b"] else "no construction passed", source="analysis/stage_b_selection.json recommend_stage_b")
    # per-candidate facts: what each condition gave, so the answer can quote the table
    import numpy as np
    for i, row in enumerate(sel["table"]):
        c = row["construction"]
        k = f"cand.{c['reading']}_{c['centring'].lower()}_{c['memory']}_{c['mask']}"
        src = f"analysis/stage_b_selection.json table[{i}]"
        o.fact(f"{k}.passed", "passed" if row["passed"] else "did not pass", source=src + " passed")
        o.fact(f"{k}.rank", row.get("primary_rank"), source=src + " primary_rank", missing="unranked")
        o.fact(f"{k}.worst", row.get("worst_adversary_overlap"), pct=True, source=src + " worst_adversary_overlap")
        for j, m in enumerate(models, 1):
            mm = row["models"].get(m) or {}
            o.fact(f"{k}.m{j}.passing", mm.get("passing"), source=src + f" models.{m}.passing")
            for cond, _ in COND:
                o.fact(f"{k}.m{j}.{cond}", (mm.get("condition_site_counts") or {}).get(cond), source=src + f" models.{m}.condition_site_counts.{cond}")
    # the chosen construction in the full census and on the screen's rows, mean over the measured layers
    M = sel.get("M")
    for j, m in enumerate(models, 1):
        vals = {}
        if M:
            rs = [d.geo_row(m, M["reading"], M["centring"], f"site{s:02d}", M["memory"], M["mask"])[0] for s in sites]
            rs = [r for r in rs if r]
            for field in ("origin", "screen_origin", "screen_mass"):
                v = [r[field][0] for r in rs if r.get(field)]
                vals[field] = float(np.mean(v)) if v else None
        src = f"analysis/geometry/{m}/{M['reading']}_{M['centring']}.json M rows, mean over layers" if M else "no M"
        o.fact(f"m.m{j}.census_other", vals.get("origin"), pct=True, source=src + " origin[0]")
        o.fact(f"m.m{j}.screen_top_other", vals.get("screen_origin"), pct=True, source=src + " screen_origin[0]")
        o.fact(f"m.m{j}.screen_mass_other", vals.get("screen_mass"), pct=True, source=src + " screen_mass[0]")
        o.fact(f"m{j}.name", d.model_name(m), source="site/manifest.json model_metadata")
    o.fact("m", describe(sel.get("M")), source="analysis/stage_b_selection.json M")
    o.fact("l", describe(sel.get("L")), source="analysis/stage_b_selection.json L")
    o.fact("r", describe(sel.get("R")), source="analysis/stage_b_selection.json R")
    o.fact("x", describe(sel.get("X")), source="analysis/stage_b_selection.json X")
    o.fact("x_reason", sel.get("X_reason") or sel.get("reason") or "—", source="analysis/stage_b_selection.json X_reason / reason")
    o.fact("passing", sum(bool(r["passed"]) for r in sel["table"]), source="analysis/stage_b_selection.json table[].passed")
    o.fact("candidates", len(sel["table"]), source="analysis/stage_b_selection.json table")
    o.fact("required", sel["required_sites"], source="analysis/stage_b_selection.json required_sites")
    o.fact("sites", len(sites), source="analysis/stage_b_selection.json sites")
    o.fact("dropped", len(dropped), source="site/manifest.json dropped_groups")
    o.chart("card", dict(answer="The fixed rule picked {{fact:x.e3-selection.m}}, the first passing candidate in its order.", numbers=[
        "Screen: {{fact:x.e3-selection.recommend}}", "Candidates passing: {{fact:x.e3-selection.passing}} of {{fact:x.e3-selection.candidates}}",
        "Chosen: {{fact:x.e3-selection.m}}"]))
    o.write()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
