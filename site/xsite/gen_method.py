"""Method tab data: the independent audit as a table, the spending ledger, the probe's documents, and the free local
diagnostic of SPEC section 0 (exp1/local_pilot.json, in the repository, not part of the fetch).

    python site/xsite/gen_method.py DATA_DIR OUT_DIR
"""
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path

from xlib import SITE, Data, Out, read_json

XID = "method"
FAMILIES = OrderedDict([
    ("row", "Complete distance rows of the audit queries, recomputed from the activations by the explicit lag sum (sites 4 and 12, every memory, reading and centring): the 64 neighbours, the cutoff distance, ties, origins, same-token counts and overlaps."),
    ("screen", "The same rows on the Stage-B-matched suffix screen: membership, walk weights (the conditional row), the short-range mass a window removed, overlaps."),
    ("pairs", "1,000 random token pairs per pass: kernel values, distances and diagonals against values the census saved before it discarded its distance blocks."),
    ("sample", "Census statistics recomputed from the stored neighbour lists for a 1% sample of tokens, including the stitched texts' position baseline."),
    ("aggregate", "The published per-setting summaries recomputed from per-token values."),
    ("formula", "Expected and maximum overlaps recomputed from the stored tie counts."),
    ("copy", "The audit's copies of the census files equal the census files."),
    ("readout", "Matched controls replayed from their keyed seeds; every matching constraint; unmatched reasons; per-document contributions, scores, coverage and bootstrap."),
    ("selection", "The Stage B choice recomputed from the stored site values: conditions, passing sites, M, L and X."),
    ("presence", "Every promised file and summary exists for every approved setting."),
    ("layout", "Stored index layouts agree with the probe (texts, positions, tokens, query sample, audit queries)."),
    ("probe", "The query sample and audit queries have the structure the spec fixes."),
    ("integrity", "Internal consistency of the stored sets."),
    ("plan", "Settings were run in the prespecified priority order."),
    ("config", "Only the five memories of the spec appear."),
    ("audit", "How many audit queries had complete rows recomputed."),
])


def _charge(row):
    if row.get("actual_usd") is not None:
        return row["actual_usd"]
    return row.get("estimated_usd", row.get("reserved_usd", 0) + row.get("startup_allowance_usd", 0))


def run(data: Path, out: Path):
    d = Data(data)
    o = Out(XID, out)
    # ---------------------------------------------------------- audit
    path = d.root / "audit_a" / "report.json"
    report = read_json(path) if path.is_file() else None
    if report is None:
        o.chart("audit", dict(columns=[dict(label="Audit")], rows=[], note="No audit report in this data directory."))
        for k in ("passed", "checks", "failed", "compared", "rows", "seconds", "spec_parameters", "models", "groups", "absent",
                  "inventory_digest", "inventory_count"):
            o.fact(f"audit.{k}", "missing", source="audit_a/report.json absent")
        for key in ("not_reproduced", "conventions", "absent"):
            o.chart(key, dict(columns=[dict(label="—")], rows=[]))
    else:
        fam = OrderedDict((k, dict(names=set(), scopes=0, compared=0, failed=0, tol=set(), worst=None)) for k in FAMILIES)
        for c in report["checks"]:
            f = fam.setdefault(c["name"].split(".")[0], dict(names=set(), scopes=0, compared=0, failed=0, tol=set(), worst=None))
            f["names"].add(c["name"])
            f["scopes"] += 1
            f["compared"] += int(c.get("compared") or 0)
            f["failed"] += int(c.get("failed") or 0)
            if c.get("tolerance") is not None:
                f["tol"].add(str(c["tolerance"]))
            w = c.get("worst_deviation")
            if isinstance(w, (int, float)):
                f["worst"] = w if f["worst"] is None else max(f["worst"], w)
        rows = []
        for k, f in fam.items():
            if not f["scopes"]:
                continue
            tol = sorted(f["tol"], key=lambda t: (not re.match(r"^[0-9.e-]+$", t), t))
            rows.append(dict(cells=[dict(text=k), dict(text=FAMILIES.get(k, "")),
                                    dict(text=f"{f['scopes']:,} ({len(f['names'])} kind{'s' * (len(f['names']) != 1)})"),
                                    dict(text=f"{f['compared']:,}"), dict(text=", ".join(tol)),
                                    dict(text="—" if f["worst"] is None else f"{f['worst']:.1e}"),
                                    dict(text="passed" if f["failed"] == 0 else f"{f['failed']:,} failed", strong=True, good=f["failed"] == 0, bad=f["failed"] > 0)],
                             detail=sorted(f["names"])))
        o.chart("audit", dict(columns=[dict(label="Family"), dict(label="What it recomputes"), dict(label="Checks"), dict(label="Values compared"),
                                       dict(label="Tolerance"), dict(label="Worst deviation"), dict(label="Result")], rows=rows,
                              note="Tolerances: a number is the largest allowed difference; exact = identical; required = must exist."))
        one = lambda label, items: dict(columns=[dict(label=label)], rows=[dict(cells=[dict(text=str(x))]) for x in items])
        o.chart("not_reproduced", one("What the audit does not reproduce", report.get("not_reproduced", [])))
        o.chart("conventions", one("Conventions the audit takes from the pipeline (and checks where it can)", report.get("conventions_trusted", [])))
        o.chart("absent", one("Absent, with the reason", [f"{a['what']}: {a['reason']}" for a in report.get("absent", [])] + list(report.get("notes", []))))
        src = "audit_a/report.json"
        o.fact("audit.passed", "passed" if report.get("passed") else "FAILED", source=src + " passed")
        o.fact("audit.checks", len(report["checks"]), source=src + " checks")
        o.fact("audit.failed", sum(not c["passed"] for c in report["checks"]), source=src + " checks[].passed")
        o.fact("audit.compared", sum(int(c.get("compared") or 0) for c in report["checks"]), source=src + " checks[].compared")
        o.fact("audit.rows", report.get("rows") if isinstance(report.get("rows"), int) else len(report.get("audit_queries", [])), source=src + " rows")
        o.fact("audit.seconds", round(report.get("seconds") or 0), source=src + " seconds")
        o.fact("audit.spec_parameters", "the spec's parameters" if report.get("spec_parameters") else "fixture parameters (not the spec's)", source=src + " spec_parameters")
        o.fact("audit.models", ", ".join(d.model_name(m) for m in report.get("models", [])), source=src + " models")
        o.fact("audit.groups", len(report.get("groups", [])), source=src + " groups")
        o.fact("audit.absent", len(report.get("absent", [])), source=src + " absent")
        inv = report.get("inventory") if isinstance(report.get("inventory"), dict) else {}
        o.fact("audit.inventory_digest", inv.get("digest") or "none: this report has no file inventory", source=src + " inventory.digest")
        o.fact("audit.inventory_count", (inv.get("count") if inv.get("count") is not None else len(inv.get("files", {}))) if inv else 0,
               source=src + " inventory.count")
    # ---------------------------------------------------------- ledger
    rows = [read_json(p) for p in sorted((d.root / "ledger" / "attempts").glob("*/*.json"))] if (d.root / "ledger").is_dir() else []
    by_unit = OrderedDict()
    for r in rows:
        u = by_unit.setdefault(r["unit"], dict(usd=0.0, attempts=0, failed=0, seconds=0.0))
        u["usd"] += _charge(r)
        u["attempts"] += 1
        u["failed"] += r.get("status") not in ("finished", "succeeded", "completed", "ok", "done")
        u["seconds"] += float(r.get("billed_seconds_estimate") or r.get("seconds") or 0)
    o.chart("ledger", dict(columns=[dict(label="Unit"), dict(label="Attempts"), dict(label="Not finished"), dict(label="Billed time"), dict(label="Charged")],
                           rows=[dict(cells=[dict(text=k), dict(text=str(u["attempts"])), dict(text=str(u["failed"])),
                                             dict(text=f"{u['seconds'] / 60:.0f} min"), dict(text=f"${u['usd']:.2f}")]) for k, u in by_unit.items()],
                           note="Charged = the provider bill where reconciled, else billed seconds × the published rate (SPEC section 11)."))
    o.fact("ledger.spent", sum(u["usd"] for u in by_unit.values()), digits=2, source="ledger/attempts/*/*.json, sum of charges")
    o.fact("ledger.attempts", len(rows), source="ledger/attempts/*/*.json")
    o.fact("ledger.failed", sum(u["failed"] for u in by_unit.values()), source="ledger/attempts/*/*.json status")
    o.fact("ledger.cap", 8, source="SPEC.md section 11 (the Stage A hard cap)")
    # ---------------------------------------------------------- probe documents (analysis/probe_summary.json, with
    # how each text begins from the packed texts) and the rejection counts
    probe = d.probe or {}
    psrc = "analysis/probe_summary.json" if d.probe else "site/texts (probe summary absent: development build)"
    light = {c["id"]: c for c in probe.get("contexts") or []}
    by_index = {s.get("file_index"): s for s in probe.get("stitch_sources") or []}
    docs, subsets = [], OrderedDict()
    for t in d.manifest["texts"]:
        p = d.asset(t["text_file"])
        c = {**p, **light.get(t["id"], {})}           # the probe summary wins where both have a field
        words = re.sub(r"\s+", " ", p.get("text", "")).strip()
        first = words[:70] + ("…" if len(words) > 70 else "")
        rank = c.get("eligible_shuffle_rank")
        if t["kind"] == "stitched":
            segs = c.get("segments") or []
            srcs = OrderedDict()
            for g in segs:
                info = by_index.get(g.get("source_file_index"), {})
                srcs.setdefault(g["source"], f"{g['source']}: {info.get('subset') or 'subset unknown'}, line {g.get('source_file_index', '?')}")
            docs.append(dict(cells=[dict(text=str(t["id"])), dict(text=f"stitched ({c.get('schedule')} segments)"), dict(text="—"), dict(text="—"),
                                    dict(text=f"{len(segs)} segments; " + "; ".join(srcs.values())), dict(text=first)], source=psrc))
        else:
            subsets[c.get("subset")] = subsets.get(c.get("subset"), 0) + 1
            docs.append(dict(cells=[dict(text=str(t["id"])), dict(text=c.get("subset") or "—"), dict(text=str(c.get("file_index", "—"))),
                                    dict(text="—" if rank is None else f"{rank:,}"), dict(text=(c.get("text_sha256") or "")[:12]), dict(text=first)], source=psrc))
    o.chart("documents", dict(columns=[dict(label="Text"), dict(label="Subset"), dict(label="Line in test.jsonl"), dict(label="Shuffle rank"),
                                       dict(label="SHA-256 (start) / segments"), dict(label="Begins")], rows=docs,
                              note="Shuffle rank: the document's place in the seeded shuffle of eligible documents; accepted texts are the first that pass every rule."))
    counts = probe.get("counts") or {}
    reasons = [("forbidden_text", "special tokens"), ("ascii", "under 95% ASCII"), ("too_short", "under 2,048 tokens"), ("whitespace", "mostly whitespace"),
               ("internal_32gram", "repeats itself"), ("shared_32gram", "shares a passage"), ("subset_cap", "subset full"), ("stitch_same_subset", "stitch pair same subset")]
    def crow(name, c, strong=False):
        return dict(cells=[dict(text=name, strong=strong), dict(text=f"{c.get('walked', 0):,}"), dict(text=f"{c.get('accepted_pool', 0):,}"),
                           dict(text=f"{c.get('accepted_stitch_source', 0):,}")] + [dict(text=f"{c.get(k, 0):,}") for k, _ in reasons])
    crows = [crow("all subsets", counts["total"], True)] if counts.get("total") else []
    crows += [crow(sub, c) for sub, c in (counts.get("by_subset") or {}).items() if c.get("walked") or c.get("accepted_pool") or c.get("accepted_stitch_source")]
    o.chart("rejections", dict(columns=[dict(label="Subset"), dict(label="Examined"), dict(label="Accepted"), dict(label="Stitch sources")]
                               + [dict(label=label) for _, label in reasons], rows=crows,
                               note="Documents examined in shuffled order until 32 texts and 4 stitch sources were accepted; each rejection is counted under its first failed rule."
                               if crows else "Rejection counts are not available in this build."))
    total = counts.get("total") or {}
    o.fact("probe.examined", total.get("walked"), source=psrc + " counts.total.walked", missing="not recorded")
    o.fact("probe.eligible", total.get("eligible"), source=psrc + " counts.total.eligible", missing="not recorded")
    o.fact("probe.texts", sum(t["kind"] != "stitched" for t in d.manifest["texts"]), source="site/manifest.json texts")
    o.fact("probe.subsets", len(subsets), source=psrc + " contexts[].subset")
    o.fact("probe.description", d.manifest.get("description") or "—", source="site/manifest.json description")
    o.fact("probe.sha", (probe.get("probe_sha256") or d.manifest.get("probe_sha256") or "—")[:16], source=psrc + " probe_sha256")
    # ---------------------------------------------------------- SPEC section 0 diagnostic (exp1/local_pilot.json)
    pilot_path = SITE.parent / "exp1" / "local_pilot.json"
    if pilot_path.is_file():
        pilot = read_json(pilot_path)
        cols = [("corpus mean", "all tokens", "same_text"), ("own-text mean", "all tokens", "same_text"),
                ("own-text mean", "not within 64 positions", "same_text"), ("corpus mean", "all tokens", "pred_match")]
        prows = []
        for mem in ("none", "8", "64", "whole"):
            cells = [dict(text=mem)]
            for c, rule, key in cols:
                v = [r[key] for r in pilot["rows"] if r["memory"] == mem and r["centring"] == c and r["rule"] == rule]
                cells.append(dict(text=f"{min(v):.2f}–{max(v):.2f}" if key == "same_text" else f"{min(v):.3f}–{max(v):.3f}"))
            prows.append(dict(cells=cells))
        chance = [r["chance"] for r in pilot["rows"] if r["rule"] == "all tokens"]
        o.chart("pilot", dict(columns=[dict(label="Memory"), dict(label="Own-text share, corpus mean"), dict(label="Own-text mean removed"),
                                       dict(label="… and no neighbour within 64"), dict(label="Same top-1 prediction (unmatched)")], rows=prows,
                              note=f"Ranges over the three layers. Chance own-text share {min(chance):.3f}."))
        o.fact("pilot.model", pilot["model"].split("/")[-1], source="exp1/local_pilot.json model")
        o.fact("pilot.tokens", pilot["T"] - 1, source="exp1/local_pilot.json T")
        o.fact("pilot.burn", pilot["burn"], source="exp1/local_pilot.json burn")
    o.write()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
