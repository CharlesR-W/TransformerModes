"""Build the interactive viewer from a SCHEMA.md (v1) data directory.

    uv run --with numpy --with markdown python -I build_viewer.py      # real data: ../data/dinov2s -> out/dinov2s
    uv run --with numpy python -I build_viewer.py --data ../data/_mock # MOCKUP data -> out/_mock (page says MOCKUP)

Output `out/<tag>/` is self-contained for viewing over HTTP (the laptop's page server, or
`python3 -m http.server -d out/<tag>` locally): `index.html` (app inlined), `vendor/` (KaTeX,
Plotly), `d/` (data).  What the build does to the data directory:
  - copies (hardlinks when possible) every binary listed in its manifest that exists;
  - transposes `cls/neurons/Lxx.u8` [N,1536] into neuron-major chunks `cls/neurons_t/Lxx_cYY.u8`
    [128, N] so one neuron's activations load as a 1.3 MB file;
  - splits `<v>/<k>/delta.json` into one JSON per pair (`<v>/<k>/delta/aXX_bYY.json`) and keeps only a
    summary (eigenvalues, cross-validated values, unsupported flags, net, path) in `d/index.json`;
  - gathers every small JSON (meta, selection, controls, decodability, seba, flows, attr without w_res,
    q, unit_tests, mlp, the CIFAR decodability) into `d/index.json`;
  - includes the verdict (viewer/verdict.md, else explore/vit_coherent/VERDICT.md, or --verdict) near the top;
    a MOCKUP build never includes it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # explore/vit_coherent
CHUNK = 128


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:  # malformed JSON is reported on the page, not fatal
        print(f"warning: cannot parse {p}: {e}", file=sys.stderr)
        return {"_error": f"cannot parse {p.name}: {e}"}


def strip_key(obj, key):
    if isinstance(obj, dict):
        return {k: strip_key(v, key) for k, v in obj.items() if k != key}
    if isinstance(obj, list):
        return [strip_key(v, key) for v in obj]
    return obj


def pick(arg, data: Path, name: str) -> Path:
    if arg:
        return Path(arg)
    return data / name if (data / name).exists() else ROOT / "data" / name


def md_to_html(md: str) -> str:
    """Minimal Markdown (headings, paragraphs, lists, **bold**, *em*, `code`, links); $...$ is left for KaTeX."""
    try:
        import markdown  # type: ignore

        return markdown.markdown(md, extensions=["tables"])
    except Exception:
        pass
    out, para, lst = [], [], []

    def inline(s):
        s = html.escape(s, quote=False)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"(?<![\*\w])\*(?!\s)(.+?)\*", r"<i>\1</i>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        s = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', s)
        return s

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if lst:
            out.append("<ul>" + "".join("<li>" + inline(x) + "</li>" for x in lst) + "</ul>")
            lst.clear()

    for line in md.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush()
        elif s.startswith("#"):
            flush()
            n = len(s) - len(s.lstrip("#"))
            out.append(f"<h{min(n + 2, 6)}>{inline(s[n:].strip())}</h{min(n + 2, 6)}>")
        elif re.match(r"^\s*[-*] ", s):
            if para:
                flush()
            lst.append(re.sub(r"^\s*[-*] ", "", s))
        elif lst and s.startswith("  "):
            lst[-1] += " " + s.strip()
        else:
            para.append(s.strip())
    flush()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dinov2s"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--cifar", default=str(ROOT.parent / "cifar_coherent" / "data" / "r20s0" / "decodability.json"))
    ap.add_argument("--verdict", default=None, help="verdict Markdown (default: viewer/verdict.md, else explore/vit_coherent/VERDICT.md)")
    ap.add_argument("--mlp", default=None, help="mlp.json (default: data dir, else explore/vit_coherent/data/)")
    ap.add_argument("--unit-tests", default=None, help="unit_tests.json (default: data dir, else explore/vit_coherent/data/)")
    ap.add_argument("--skip-data", action="store_true", help="rebuild index.html only (data already in place)")
    a = ap.parse_args()
    data = Path(a.data).resolve()
    meta = load_json(data / "meta.json")
    if meta is None:
        sys.exit(f"no meta.json in {data}")
    tag = meta.get("tag", data.name)
    mock = bool(meta.get("MOCKUP")) or data.name.startswith("_mock")
    cifar = Path(a.cifar)
    if mock and not cifar.exists():
        cifar = data / "cifar_decodability.json"
    out = Path(a.out) if a.out else HERE / "out" / tag
    d = out / "d"
    notes = []

    manifest = load_json(data / "manifest.json") or []
    if isinstance(manifest, dict):
        manifest = manifest.get("files", list(manifest.values()))
    have = {}
    if not a.skip_data:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for ent in manifest:
        rel = ent["path"]
        src = data / rel
        if not src.exists():
            notes.append(f"listed in manifest but missing: {rel}")
            continue
        have[rel] = ent.get("shape")
        if a.skip_data:
            continue
        if re.fullmatch(r"cls/neurons/L\d\d\.u8", rel):
            arr = np.fromfile(src, np.uint8)
            n_img = int(meta["N"])
            arr = arr.reshape(n_img, -1)
            nn = arr.shape[1]
            for c0 in range(0, nn, CHUNK):
                rel2 = rel.replace("neurons/", "neurons_t/").replace(".u8", f"_c{c0 // CHUNK:02d}.u8")
                (d / rel2).parent.mkdir(parents=True, exist_ok=True)
                np.ascontiguousarray(arr[:, c0 : c0 + CHUNK].T).tofile(d / rel2)
        else:
            link_or_copy(src, d / rel)
    # thumbnails are not in the binary manifest
    thumbs = sorted((data / "thumbs").glob("sheet_*.jpg"))
    if not a.skip_data:
        for t in thumbs:
            link_or_copy(t, d / "thumbs" / t.name)
    if not thumbs:
        notes.append("no thumbnails (thumbs/sheet_XXX.jpg); image strips show index numbers")

    idx = {
        "tag": tag, "mock": mock, "built": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "meta": meta, "have": have, "neuron_chunk": CHUNK, "n_thumb_sheets": len(thumbs),
        "selection": load_json(data / "selection.json"), "controls": load_json(data / "controls.json"),
        "decodability": load_json(data / "decodability.json"), "unit_tests": load_json(pick(a.unit_tests, data, "unit_tests.json")),
        "mlp": load_json(pick(a.mlp, data, "mlp.json")), "cifar_decodability": load_json(cifar),
        "cifar_path": str(cifar), "views": {}, "notes": notes,
    }
    for v in ("cls", "mp"):
        vd = data / v
        if not vd.exists():
            continue
        V = idx["views"][v] = {"kernels": {}}
        for kd in sorted(p for p in vd.iterdir() if p.is_dir() and p.name not in ("neurons", "heads")):
            k = kd.name
            K = V["kernels"][k] = {}
            for name in ("seba", "flows", "q"):
                j = load_json(kd / f"{name}.json")
                if j is not None:
                    K[name] = j
            j = load_json(kd / "attr.json")
            if j is not None:
                K["attr"] = strip_key(j, "w_res")
            dj = load_json(kd / "delta.json")
            if dj is not None:
                summ = {}
                for key, r in (dj.get("results") or {}).items():
                    summ[key] = {f: r.get(f) for f in ("evals", "cv", "cv_evalA", "unsupported", "net", "path")}
                    if not a.skip_data:
                        p = d / v / k / "delta" / f"{key}.json"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.write_text(json.dumps(sanitize(r), separators=(",", ":"), allow_nan=False))
                K["delta"] = {"pairs": dj.get("pairs"), "summary": summ, "triples": dj.get("triples"),
                              "controls": dj.get("controls"), "extra": {kk: vv for kk, vv in dj.items()
                                                                        if kk not in ("pairs", "results", "triples", "controls")}}
    vpath = Path(a.verdict) if a.verdict else (HERE / "verdict.md" if (HERE / "verdict.md").exists() else ROOT / "VERDICT.md")
    if mock and not a.verdict:
        vpath = HERE / "_no_verdict_for_mock.md"
    idx["verdict_html"] = md_to_html(vpath.read_text()) if vpath.exists() else None

    (d / "index.json").write_text(json.dumps(sanitize(idx), separators=(",", ":"), allow_nan=False))

    # vendor + page
    vend = out / "vendor"
    if vend.exists():
        shutil.rmtree(vend)
    shutil.copytree(HERE / "vendor", vend)
    css = (HERE / "src" / "style.css").read_text()
    js = "\n".join(p.read_text() for p in sorted((HERE / "src" / "js").glob("*.js")))
    page = (HERE / "src" / "index.html").read_text()
    title = ("MOCKUP · " if mock else "") + "ViT smoothness across depth · " + tag
    page = page.replace("__TITLE__", html.escape(title)).replace("/*__CSS__*/", css).replace("//__JS__", js)
    (out / "index.html").write_text(page)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out / 'index.html'}  ({size / 1e6:.1f} MB incl. data)  mock={mock}  notes={len(notes)}")
    for n in notes[:20]:
        print("  note:", n)


def sanitize(o):
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize(v) for v in o]
    return o


if __name__ == "__main__":
    main()
