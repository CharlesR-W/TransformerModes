"""Check one Experiments fragment against a build's facts and chart payloads.

    python3 site/check_fragment.py e1-neighbours [--build build/site] [--strict]

Errors: bad structure, unknown fact tokens, missing chart payloads, figures without a 'Plotted:' caption, scripts or
styles in a fragment, duplicate ids, images without alt text, literal colours. Warnings (errors with --strict): answer
placeholders left, more than five number tiles or two supporting figures or six caveats.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from build_x import XDIR, BuildError, load_facts, manifest, read_entry

HERE = Path(__file__).resolve().parent


def check(entry, xdir):
    facts = load_facts(xdir)
    text, parsed, card = read_entry(entry, xdir, facts)
    errors, warnings, ids = [], [], set()
    for tag, attrs, line in parsed.tags:
        if tag in ("script", "style", "link"):
            errors.append(f"line {line}: no <{tag}> in fragments")
        if tag != "figure" and "data-live" in attrs:
            errors.append(f"line {line}: data-live belongs on a figure")
        if "id" in attrs:
            if attrs["id"] in ids:
                errors.append(f"line {line}: duplicate id {attrs['id']!r}")
            ids.add(attrs["id"])
        if tag == "img" and "alt" not in attrs:
            errors.append(f"line {line}: image needs alt text")
        if re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", attrs.get("style") or ""):
            errors.append(f"line {line}: literal colour; use palette tokens")
    for _, line, caption in parsed.figures:
        if not caption.strip():
            errors.append(f"line {line}: figure needs a caption")
    for name in ("x-scope", "x-answer", "x-numbers", "x-caveats"):
        if not any(name in (a.get("class") or "").split() for _, a, _ in parsed.tags):
            errors.append(f"missing .{name}")
    tiles = len(re.findall(r'<div(?: class="x-num-text")?><b>', text.split('class="x-numbers"', 1)[-1].split("</div>\n</div>", 1)[0]))
    if not 3 <= tiles <= 5:
        warnings.append(f"{tiles} number tiles (3-5 expected)")
    support = text.split('-figures">', 1)[-1].split("<h3", 1)[0].count("<figure") if '-figures">' in text else 0
    if support > 2:
        warnings.append(f"{support} supporting figures (at most 2)")
    caveats = text.split('class="x-caveats"', 1)[-1].split("</ul>", 1)[0].count("<li") if "x-caveats" in text else 0
    if caveats > 6:
        warnings.append(f"{caveats} caveats (at most 6)")
    if "x-placeholder" in text:
        warnings.append("answer placeholder still present")
    if card is None:
        warnings.append("no overview card in the generator output")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("id")
    ap.add_argument("--build", type=Path, default=HERE.parent / "build" / "site")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    try:
        entry = next((e for e in manifest() if e["id"] == a.id), None)
        if entry is None:
            raise BuildError(f"unknown entry {a.id!r} (see {XDIR / 'index.json'})")
        errors, warnings = check(entry, a.build / "assets" / "x")
    except (BuildError, OSError) as exc:
        errors, warnings = [str(exc)], []
    for level, msgs in (("error", errors), ("warning", warnings)):
        for m in msgs:
            print(f"{a.id}: {level}: {m}")
    print(f"{a.id}: {len(errors)} errors, {len(warnings)} warnings")
    return int(bool(errors or (a.strict and warnings)))


if __name__ == "__main__":
    raise SystemExit(main())
