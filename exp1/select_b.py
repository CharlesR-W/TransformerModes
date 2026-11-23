"""Apply the prespecified Stage B screen, never launch Stage B."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from census import MEMORIES, PRIORITY, SITES, input_name, setting_path, write_json
from readout import F, SITES as READOUT_SITES, STRATA, bootstrap_indices, bootstrap_scores, readout_path, summarize

INTERVENTION = (("C", "none"), ("C", "window"), ("O", "none"), ("O", "window"))
R = dict(reading="cosine", centring="C", memory="none", mask="none")
FALLBACK = dict(reading="cosine", centring="C", memory="64", mask="window", label="did not pass the screen")


def _mean(a):
    a = np.asarray(a)
    return float(a[np.isfinite(a)].mean()) if np.isfinite(a).any() else None


def screen_site(path, k, mask):
    with np.load(path) as data:
        if "screen_mass" not in data or "screen_overlap_maximum" not in data:
            return None
        mass = data["screen_mass"]
        status = data["screen_status"]
        expected, maximum = (data["screen_overlap_" + v] / k for v in ("expected", "maximum"))
        if maximum.shape[1] != 3 or not len(status):
            return None
        removed = data["screen_removed"] if mask == "window" else None
        origin_counts = data["screen_origins"]
    origin_names = ("other", "far", "near")
    masses = {g: _mean(mass[:, i]) for i, g in enumerate(origin_names)}
    top = {g: _mean(origin_counts[:, i] / k) for i, g in enumerate(origin_names)}
    refs = ("local", "random", "embedding")
    exp_share = {r: _mean(expected[:, i]) for i, r in enumerate(refs)}
    max_share = {r: _mean(maximum[:, i]) for i, r in enumerate(refs)}
    short = _mean(removed) if removed is not None else masses["near"]

    def upper(n, limit):
        return n is not None and n <= limit

    checks = dict(other_mass=masses["other"] is not None and masses["other"] >= .25,
                  short_mass=upper(short, .5 if mask == "window" else .25), local=upper(max_share["local"], .5),
                  random=upper(max_share["random"], .5), embedding=upper(max_share["embedding"], .5))
    return dict(passed=all(checks.values()), conditions=checks, rows=len(status),
                status_counts=np.bincount(status, minlength=4).tolist(),
                query_mass=masses, top_origin=top, removed=short if removed is not None else None,
                removed_valid_rows=int(np.isfinite(removed).sum()) if removed is not None else None,
                expected_overlap=exp_share, maximum_overlap=max_share)


def _construction(reading, centring, memory, mask):
    return dict(reading=reading, centring=centring, memory=memory, mask=mask)


def _rank_key(row):
    fields = row["construction"]
    return (INTERVENTION.index((fields["centring"], fields["mask"])), fields["reading"] != "cosine",
            row["worst_adversary_overlap"], MEMORIES[fields["memory"]])


def _read_concordance(root, row, models, sites):
    field = row["construction"]
    studies, details = {}, {}
    for model, site in itertools.product(models, sites):
        key = f"{model}/{site}"
        path = readout_path(root, model, field["reading"], field["centring"], site, field["memory"], field["mask"])
        if not path.is_file():
            details[key] = dict(eligible=False, reason="missing readout")
            continue
        with np.load(path) as f:
            by_doc = f["by_document"]
            docs = f["documents"]
            summary = summarize(by_doc, min_matched=int(f["min_matched"]))
        details[key] = summary
        if summary["eligible"]:
            covered = np.array([summary["strata"][s]["status"] == "covered" for s in STRATA])
            studies[key] = (by_doc, docs, covered)
    eligible = bool(details) and len(studies) == len(details)
    row["prediction"] = dict(eligible=eligible, settings=details)
    return studies if eligible else None


def _example(root, rows, m, models, sites):
    candidates = []
    for row in rows:
        studies = _read_concordance(root, row, models, sites)
        if row is not m and studies:
            candidates.append((row, studies))
    if not candidates:
        return None, [], "no eligible passing alternative"
    vals = [v for _, studies in candidates for v in studies.values()]
    by_doc, docs, _ = vals[0]
    if any(not np.array_equal(d, docs) for _, d, _ in vals):
        raise ValueError("cannot use a shared document bootstrap with different Q documents")
    common = np.logical_and.reduce([c for _, _, c in vals])
    names = [s for s, use in zip(STRATA, common) if use]
    if not names:
        return None, [], "no strata covered in every eligible comparison"
    draws = bootstrap_indices(len(by_doc))
    for row, studies in candidates:
        scores, boots, detail = [], [], {}
        for key, (by_doc, _, _) in studies.items():
            total = by_doc.sum(0)[common]
            values = total[:, F["advantage"]] / total[:, F["matched"]]
            score = float(values.mean())
            scores.append(score)
            boots.append(bootstrap_scores(by_doc, common, draws))
            detail[key] = dict(score=score, stratum_means=dict(zip(names, values.tolist())))
        boot = np.stack(boots).mean(0)
        valid = boot[np.isfinite(boot)]
        row["prediction"].update(common_score=float(np.mean(scores)), common_sites=detail,
                                 common_bootstrap_p5=float(np.percentile(valid, 5)) if len(valid) else None,
                                 common_bootstrap_valid=len(valid), common_bootstrap_draws=len(boot))
    best = min((r for r, _ in candidates), key=lambda r: (-r["prediction"]["common_score"], _rank_key(r)))
    fifth = best["prediction"]["common_bootstrap_p5"]
    if fifth is not None and fifth > 0:
        return best["construction"], names, "screened for prediction concordance"
    return None, names, "highest common-stratum score did not have a positive bootstrap fifth percentile"


def select(root, *, models=("pythia", "olmo"), sites=SITES, readout_sites=READOUT_SITES, required_sites=5):
    root = Path(root)
    if not models or not sites or not 1 <= required_sites <= len(sites):
        raise ValueError("invalid model/site screen")
    configs = {}
    for model, (reading, centring) in itertools.product(models, PRIORITY):
        path = root / "census" / model / f"{reading}_{centring}" / "group.json"
        if path.exists():
            configs[model, reading, centring] = json.loads(path.read_bytes())
    rows = []
    for (reading, centring), mem, mask in itertools.product(PRIORITY, tuple(MEMORIES)[1:], ("none", "window")):
        row = dict(construction=_construction(reading, centring, mem, mask), models={})
        worst_adversaries = []
        for model in models:
            by_site = {}
            config = configs.get((model, reading, centring))
            for site in sites:
                path = setting_path(root, model, reading, centring, input_name(site), mem, mask)
                if config and input_name(site) in config["inputs"] and mem in config["memories"] and mask in config["masks"] and path.exists():
                    data = screen_site(path, config["k"], mask)
                else:
                    data = None
                by_site[str(site)] = data
                if data is not None:
                    worst_adversaries.extend(data["maximum_overlap"][r] for r in ("random", "embedding"))
            observed = [x for x in by_site.values() if x is not None]
            passing = sum(s["passed"] for s in observed)
            complete = len(observed) == len(sites)
            condition_counts = {key: sum(site["conditions"][key] for site in observed)
                                for key in ("other_mass", "short_mass", "local", "random", "embedding")}
            row["models"][model] = dict(sites=by_site, passing=passing, complete=complete,
                                        condition_site_counts=condition_counts, passed=complete and passing >= required_sites)
        row["complete"] = all(m["complete"] for m in row["models"].values())
        row["passed"] = all(m["passed"] for m in row["models"].values())
        row["worst_adversary_overlap"] = max((a for a in worst_adversaries if a is not None), default=None)
        rows.append(row)
    passing = sorted([r for r in rows if r["passed"]], key=_rank_key)
    result = dict(recommend_stage_b=bool(passing), R=R, M=None, L=None, X=None, common_strata=[],
                  required_sites=required_sites, sites=list(sites), readout_sites=list(readout_sites),
                  thresholds=dict(other_mass_min=.25, removed_mass_max=.5, unmasked_near_mass_max=.25, overlap_max=.5),
                  table=rows)
    if passing:
        best = passing[0]
        for rank, r in enumerate(passing, 1):
            r["primary_rank"] = rank
        m = best["construction"]
        result.update(M=m, L=dict(m, memory="none"))
        x, common, reason = _example(root, passing, best, models, readout_sites)
        result.update(X=x, common_strata=common, X_reason=reason)
        result["solve"] = [R] + ([result["L"]] if result["L"] != R else []) + [m] + ([x] if x is not None else [])
    else:
        result.update(reason="no candidate met the prespecified screen on available complete settings",
                      solve=[], owner_override_only=[R, FALLBACK])
    write_json(root / "analysis" / "stage_b_selection.json", result)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    args = cli.parse_args()
    outcome = select(args.root)
    print(json.dumps({k: outcome[k] for k in ("recommend_stage_b", "M", "L", "X")}))
