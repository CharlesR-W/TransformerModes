"""Generator helpers (adapted from DepthEmbedding site/xsite/xlib.py): facts, chart payloads, reading the fetched data.

A generator is `run(data: Path, out: Path)`: it reads the fetched Stage A directory (analysis/, audit_a/, ledger/,
site/) and writes `out/<id>.json` (chart payloads) and `out/<id>.facts.json` (every number the prose may quote).
Pages insert facts with `{{fact:x.<id>.<name>}}`; the build fails on an unknown token.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]
PALETTE = json.loads((SITE / "palette.json").read_text(encoding="utf-8"))
MEMORIES = ["none", "2", "8", "64", "whole"]
MEMORY_LABELS = ["none", "2", "8", "64", "whole"]
ORIGINS = ["other texts", "same text, far", "same text, within 64"]
ACCOUNTINGS = [("C", "none"), ("C", "window"), ("O", "none"), ("O", "window")]
ACC_LABEL = {("C", "none"): "corpus mean", ("C", "window"): "corpus mean + window",
             ("O", "none"): "own-text mean", ("O", "window"): "own-text mean + window",
             ("C", "other_texts"): "corpus mean, other texts only", ("O", "other_texts"): "own-text mean, other texts only"}


def clean(x):
    """JSON-safe data: NumPy -> Python, non-finite chart values -> null (a gap)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def fmt(value, digits=None, pct=False):
    """Format a finite number (pct takes a fraction), or pass a plain string through."""
    if isinstance(value, str):
        if digits is not None or pct:
            raise ValueError("digits and pct are only for numerical facts")
        return value
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"expected a finite number or a string, got {value!r}")
    if type(value) is int and digits is None and not pct:
        return f"{value:,}"
    number = value * 100 if pct else value
    if digits is None:
        digits = 0 if pct else 3
    text = f"{number:,.{digits}f}"
    if float(text.replace(",", "")) == 0:
        text = text.lstrip("-")
    return text + ("%" if pct else "")


def _write_json(path, obj, **kw):
    text = json.dumps(obj, allow_nan=False, ensure_ascii=False, **kw) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        name = Path(f.name)
        try:
            f.write(text)
            f.close()
            os.chmod(name, 0o644)
            os.replace(name, path)
        finally:
            name.unlink(missing_ok=True)


class Out:
    """Collect an entry's chart records and its facts."""

    def __init__(self, xid, out: Path):
        if not re.fullmatch(r"[a-z][a-z0-9-]*", xid):
            raise ValueError(f"id must be a lower-case slug: {xid!r}")
        self.id, self.out, self.charts, self.facts = xid, Path(out), {}, {}

    def chart(self, key, obj):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", key) or key in self.charts or not isinstance(obj, dict):
            raise ValueError(f"expected a unique JSON object key, got {key!r}")
        self.charts[key] = clean(obj)
        return key

    def fact(self, name, value, digits=None, pct=False, *, source, missing="n/a", tier=None):
        """Register a value and its display text. A value of None is recorded as missing and shown as `missing`."""
        if not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", name) or name in self.facts:
            raise ValueError(f"expected a unique lower-case dotted fact name, got {name!r}")
        value = clean(value)
        text = missing if value is None else fmt(value, digits, pct)
        self.facts[name] = {"value": value, "text": text, "source": source, **({"tier": tier} if tier else {})}
        return text

    def write(self):
        _write_json(self.out / f"{self.id}.json", self.charts, separators=(",", ":"))
        _write_json(self.out / f"{self.id}.facts.json", self.facts, indent=1)


# ------------------------------------------------------------------ reading the fetched data
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Data:
    """The fetched Stage A directory. Only reads; never recomputes a result."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = read_json(self.root / "site" / "manifest.json")
        self.models = list(self.manifest["models"])
        plan = self.root / "analysis" / "stage_a_plan.json"
        self.plan = read_json(plan) if plan.is_file() else None

    def model_name(self, model):
        meta = self.manifest.get("model_metadata", {}).get(model, {})
        s = str(meta.get("id") or model).split("/")[-1]
        if s.lower().startswith("pythia-"):
            return "Pythia-" + s[7:]
        if s.lower().startswith("olmo-"):
            return "OLMo-" + re.sub(r"-0724-hf$|-hf$", "", s[5:], flags=re.I)
        return s

    @lru_cache(maxsize=None)
    def geometry(self, dataset, reading, centring):
        path = self.root / "analysis" / "geometry" / dataset / f"{reading}_{centring}.json"
        return read_json(path) if path.is_file() else None

    def geo_row(self, dataset, reading, centring, input_name, memory, mask):
        rows = self.geometry(dataset, reading, centring) or []
        for i, r in enumerate(rows):
            if r["input"] == input_name and r["memory"] == memory and r["mask"] == mask:
                return r, f"analysis/geometry/{dataset}/{reading}_{centring}.json[{i}]"
        return None, None

    def sites(self, dataset, reading="cosine", centring="C"):
        rows = self.geometry(dataset, reading, centring) or []
        return sorted({int(r["input"][4:]) for r in rows if r["input"].startswith("site")})

    def site_or_nearest(self, dataset, want, reading="cosine", centring="C"):
        have = self.sites(dataset, reading, centring)
        return min(have, key=lambda s: (abs(s - want), s)) if have else None

    def asset(self, path):
        """Decode one packed asset (JSON only) by its manifest path."""
        spec = self.manifest["files"][path]
        raw = gzip.decompress((self.root / "site" / self.manifest["data_directory"] / path).read_bytes())
        if len(raw) != spec["decoded_bytes"]:
            raise ValueError(f"{path}: size differs from the manifest")
        return json.loads(raw)

    @property
    def probe(self):
        """analysis/probe_summary.json (light probe fields), or None when absent (allowed only in --dev builds)."""
        path = self.root / "analysis" / "probe_summary.json"
        return read_json(path) if path.is_file() else None

    def stitch_subsets(self, text_payload):
        """{source label: subset} for a stitched text, from its segments and the probe summary's stitch sources."""
        by_index = {s.get("file_index"): s.get("subset") for s in ((self.probe or {}).get("stitch_sources") or [])}
        out = {}
        for seg in text_payload.get("segments") or []:
            out.setdefault(seg["source"], by_index.get(seg.get("source_file_index")))
        return out

    def stitched(self):
        return [t["id"] for t in self.manifest["texts"] if t["kind"] == "stitched"]

    def dropped(self):
        return [tuple(g) for g in self.manifest.get("dropped_groups", [])]


def share(v):
    return None if v is None else float(v)
