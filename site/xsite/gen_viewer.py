"""Viewer side data: text labels for the Text menu, and the window's removed short-range mass per viewer setting
(a suffix-sample aggregate from analysis/geometry; never per token). Facts used by the primer.

    python site/xsite/gen_viewer.py DATA_DIR OUT_DIR
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from xlib import Data, MEMORIES, Out


def _words(text, n=6):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    s = " ".join(words[:n])
    return s if len(s) <= 46 else s[:44].rstrip() + "…"


def run(data: Path, out: Path):
    d = Data(data)
    o = Out("viewer", out)
    m = d.manifest
    texts = {}
    for t in m["texts"]:
        payload = d.asset(t["text_file"])
        if t["kind"] == "stitched":
            label = f"{t['id']} · stitched, {payload.get('schedule', '?')} segments"
            short = f"Stitched text {t['id']}"
            subset = None
        else:
            subset = payload.get("subset")
            label = f"{t['id']} · {subset} · “{_words(payload.get('text', ''))}”" if subset else f"Text {t['id']}"
            short = f"Text {t['id']}"
        texts[str(t["id"])] = dict(label=label, short=short, subset=subset)
    removed = {}
    for model in d.models:
        for centring in ("C", "O"):
            for site in m["viewer_sites"]:
                for memory in MEMORIES:
                    row, _ = d.geo_row(model, "cosine", centring, f"site{site:02d}", memory, "window")
                    if row is not None and row.get("screen_removed") is not None:
                        removed[f"{model}|{centring}|{site}|{memory}"] = row["screen_removed"]
    o.chart("texts", texts)
    o.chart("removed", removed)
    corpus = m["candidate_pools"]["corpus"]
    starts = sorted({r[1] for r in corpus})
    o.fact("burn_in", starts[0] - 1 if len(starts) == 1 else None, source="site/manifest.json candidate_pools.corpus: first analysed position - 1",
           missing="varies by text")
    o.fact("first_position", starts[0], source="site/manifest.json candidate_pools.corpus")
    o.fact("texts", sum(t["kind"] != "stitched" for t in m["texts"]), source="site/manifest.json texts (kind pool)")
    o.fact("stitched", sum(t["kind"] == "stitched" for t in m["texts"]), source="site/manifest.json texts (kind stitched)")
    o.fact("pool_tokens", sum(b - a for _, a, b in corpus), source="site/manifest.json candidate_pools.corpus")
    o.fact("text_tokens", max(t["positions"][1] - 1 for t in m["texts"]), source="site/manifest.json texts[].positions")
    names = [d.model_name(x) for x in d.models]
    o.fact("models", " and ".join(names) if len(names) < 3 else ", ".join(names), source="site/manifest.json models, model_metadata")
    o.fact("shown", m["viewer_k"], source="site/manifest.json viewer_k")
    o.write()


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
