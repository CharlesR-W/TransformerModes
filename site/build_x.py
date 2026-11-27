"""Experiments tab and page-fragment checks (adapted from DepthEmbedding site/build_x.py, without embedding figures).

Entries are listed in content/experiments/index.json; each is content/experiments/<id>.html, a
<section class="experiment" id="x-<id>"> whose one h2 is the title in index.json. Generators (xsite/gen_*.py) write
assets/x/<id>.json (chart payloads; key "card" for the overview card) and assets/x/<id>.facts.json into the build.
Every {{fact:x.<id>.<name>}} token anywhere on the page is replaced by the fact's text; an unknown token fails.
"""
from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
XDIR = HERE / "content" / "experiments"
ID = re.compile(r"[a-z][a-z0-9-]*")
FACT = re.compile(r"\{\{\s*fact:([^{}]+)\}\}")
LIVE_KINDS = ("lines", "heatmap", "bars", "dots", "hist", "stack", "points", "table", "agreement", "paired")
GROUPS = {"stage-a": "Stage A: the neighbour census", "depth": "Depth embedding"}
# An entry with "requires": "<name>" is included only when the build wrote assets/x/<name>.facts.json (D1 needs the
# depth pack: build_site.py --depth). The lede addition _lede_<name>.html follows the same rule.
REQUIRES = {"depth"}
SHOW_CARDS = False


class BuildError(ValueError):
    pass


def require(condition, where, message):
    if not condition:
        raise BuildError(f"{where}: {message}")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BuildError(f"{path}: {e}") from e


def manifest():
    path = XDIR / "index.json"
    entries = read_json(path)
    require(isinstance(entries, list), path, "expected an ordered list of entries")
    seen = set()
    for e in entries:
        xid = e.get("id")
        require(isinstance(xid, str) and ID.fullmatch(xid) and xid not in seen, path, f"bad or repeated id {xid!r}")
        require(isinstance(e.get("title"), str) and e["title"].strip(), path, f"{xid}: missing title")
        require(e.get("group") in GROUPS, path, f"{xid}: unknown group {e.get('group')!r}")
        require(e.get("requires") in (None, *REQUIRES), path, f"{xid}: unknown requirement {e.get('requires')!r}")
        seen.add(xid)
    return entries


def load_facts(xdir: Path):
    facts = {}
    for path in sorted(Path(xdir).glob("*.facts.json")):
        xid = path.name[:-len(".facts.json")]
        for k, v in read_json(path).items():
            facts[f"x.{xid}.{k}"] = v
    return facts


def fact_sub(text, facts, where, *, plain=False):
    def replace(m):
        key = m[1].strip()
        record = facts.get(key)
        require(isinstance(record, dict) and isinstance(record.get("text"), str), where, f"unknown fact {key!r}")
        return record["text"] if plain else html.escape(record["text"])

    result = FACT.sub(replace, text)
    require(not re.search(r"\{\{\s*fact\b", result), where, "malformed fact token")
    return result


def external(url):
    url = re.sub(r"[\t\r\n]", "", url).strip()
    return bool(re.match(r"(?:[a-z][a-z0-9+.-]*:|[/\\]{2})", url, re.I)) and not url.startswith(("data:", "mailto:"))


def check_css(text, where):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    for m in re.finditer(r"(?:@import\s+(?:url\(\s*)?|url\(\s*)['\"]?([^'\"\s;)]+)", text, re.I):
        require(m[1].startswith("data:") or not external(m[1]), where, f"external CSS URL {m[1]!r}")


class Markup(HTMLParser):
    """The subset of HTML needed for fragment validation and local resources."""

    def __init__(self, text, where):
        super().__init__(convert_charrefs=True)
        self.where, self.tags, self.h2, self.figures = where, [], [], []
        self.in_h2 = self.in_style = self.in_caption = False
        self.style, self.figure = "", None
        self.feed(text)
        self.close()
        check_css(self.style, where)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        self.tags.append((tag, a, line))
        rel = (a.get("rel") or "").lower().split()
        dest = a.get("src") if tag in ("script", "img", "iframe") else a.get("href") if tag == "link" and (
            "stylesheet" in rel or "modulepreload" in rel or "icon" in rel) else None
        if dest:
            require(not external(dest), f"{self.where}:{line}", f"external resource {dest!r}")
        if a.get("style"):
            check_css(a["style"], f"{self.where}:{line}")
        if tag == "style":
            self.in_style = True
        if tag == "h2":
            self.in_h2 = True
            self.h2.append("")
        if tag == "figure":
            self.figure = [a, line, ""]
            self.figures.append(self.figure)
        if tag == "figcaption" and self.figure:
            self.in_caption = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False
            self.style += "\n"
        if tag == "h2":
            self.in_h2 = False
        if tag in ("figure", "figcaption"):
            self.in_caption = False
        if tag == "figure":
            self.figure = None

    def handle_data(self, data):
        if self.in_h2:
            self.h2[-1] += data
        if self.in_style:
            self.style += data
        if self.in_caption:
            self.figure[2] += data


def ref_target(ref, xid, payload, where):
    url, _, key = (ref or "").partition("#")
    require(url == f"assets/x/{xid}.json" and key, where, f"expected assets/x/{xid}.json#<key>, got {ref!r}")
    obj = payload
    for part in key.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    require(isinstance(obj, dict), where, f"no chart object at {ref!r} (run the generator)")
    return obj


def check_live_shape(obj, kind, where):
    required = {"lines": ("x", "series"), "heatmap": ("rows", "cols", "z"), "bars": ("categories", "series"),
                "dots": ("rows", "series"), "hist": ("edges", "series"), "stack": ("categories", "segments"),
                "points": ("categories", "series"), "table": ("columns", "rows"),
                "agreement": ("categories", "series", "segments", "strip", "groups"), "paired": ("rows", "series")}[kind]
    panels = obj.get("panels", [obj])
    shared = {k: v for k, v in obj.items() if k != "panels"}
    require(isinstance(panels, list) and all(isinstance(p, dict) and all(k in p or k in shared for k in required) for p in panels),
            where, f"{kind} payload requires {', '.join(required)} (in each panel for small multiples)")


def read_entry(entry, xdir, facts):
    xid = entry["id"]
    where = XDIR / f"{xid}.html"
    payload_path = Path(xdir) / f"{xid}.json"
    payload = read_json(payload_path) if payload_path.exists() else {}
    text = fact_sub(where.read_text(encoding="utf-8"), facts, where)
    parsed = Markup(text, where)
    require(parsed.tags and parsed.tags[0][0] == "section" and parsed.tags[0][1].get("id") == f"x-{xid}"
            and "experiment" in (parsed.tags[0][1].get("class") or "").split(), where,
            f'fragment must start with <section class="experiment" id="x-{xid}">')
    titles = [" ".join(h.split()) for h in parsed.h2]
    require(titles == [entry["title"]], where, f"expected one h2 matching {entry['title']!r}, got {titles!r}")
    for a, line, caption in parsed.figures:
        loc = f"{where}:{line}"
        if "live" in (a.get("class") or "").split() or "data-live" in a:
            kind = a.get("data-live")
            require(kind in LIVE_KINDS, loc, f"invalid live figure kind {kind!r}")
            check_live_shape(ref_target(a.get("data-src"), xid, payload, loc), kind, loc)
        require(caption.strip().startswith("Plotted:") or "x-table" in (a.get("class") or ""), loc,
                "a figure caption starts with 'Plotted:'")
    card = payload.get("card")
    if card is not None:
        require(isinstance(card, dict) and isinstance(card.get("answer"), str) and isinstance(card.get("numbers"), list)
                and 3 <= len(card["numbers"]) <= 5, where, "card needs an answer and 3-5 number strings")
        card = dict(card, answer=fact_sub(card["answer"], facts, where, plain=True),
                    numbers=[fact_sub(n, facts, where, plain=True) for n in card["numbers"]])
    return text, parsed, card


def available(entry, xdir):
    need = entry.get("requires")
    return need is None or (Path(xdir) / f"{need}.facts.json").is_file()


def experiments_html(xdir, facts):
    entries = []
    for e in manifest():
        if not available(e, xdir):
            continue
        src = XDIR / f"{e['id']}.html"
        require(src.is_file(), src, "experiment fragment missing")
        text, _, card = read_entry(e, xdir, facts)
        entries.append((e, text, card))
    lede = XDIR / "_lede.html"
    parts = [fact_sub(lede.read_text(encoding="utf-8"), facts, lede) if lede.is_file() else ""]
    for need in sorted(REQUIRES):
        extra = XDIR / f"_lede_{need}.html"
        if extra.is_file() and (Path(xdir) / f"{need}.facts.json").is_file():
            parts.append(fact_sub(extra.read_text(encoding="utf-8"), facts, extra))
    # The landing (_lede.html) now carries its own three findings, each linking to its entry, so the "At a glance"
    # cards are no longer drawn; each generator's card is still validated by read_entry. Set SHOW_CARDS to bring them back.
    cards = {g: [] for g in GROUPS}
    for e, _, c in (entries if SHOW_CARDS else []):
        if c is None:
            continue
        nums = "".join(f"<li>{html.escape(n)}</li>" for n in c["numbers"])
        answer = html.escape(c["answer"])
        if c.get("placeholder"):
            answer = f'<span class="x-placeholder">{answer}</span>'
        cards[e["group"]].append(f'<article class="x-card"><h4><a href="#experiments/x-{e["id"]}">{html.escape(e["title"])}</a></h4>'
                                 f'<p class="x-card-answer"><strong>{answer}</strong></p><ul class="x-card-numbers">{nums}</ul></article>')
    if any(cards.values()):
        parts.append('<section class="x-glance" aria-labelledby="x-glance-title"><h2 id="x-glance-title">At a glance</h2>')
        for g, title in GROUPS.items():
            if cards[g]:
                parts.append(f'<section class="x-card-group"><h3>{title}</h3><div class="x-cards">' + "\n".join(cards[g]) + "</div></section>")
        parts.append("</section>")
    parts.extend(text for _, text, _ in entries)
    return "\n".join(parts), [e for e, _, _ in entries]
