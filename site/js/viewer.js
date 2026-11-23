// The Viewer tab: one text at a time, every token coloured by where its 64 nearest neighbours come from, a token
// card listing its nearest neighbours as text, and a position strip. Data: d/manifest.json (exp1/pack_site.py) and
// assets/x/viewer.json (site/xsite/gen_viewer.py: text labels, model names, the window's removed mass).
// State lives in the hash: #viewer?text=3&model=pythia&layer=8&memory=8&acc=C-window&pos=1234
import { manifest, asset, json, poolIndex } from "./data.js";
import { h, showTip, hideTip } from "./charts.js";

const ACC = [
  { key: "C-none", centring: "C", mask: "none", label: "Corpus mean", words: "corpus mean, every other token eligible" },
  { key: "C-window", centring: "C", mask: "window", label: "Corpus mean + window", words: "corpus mean, no neighbour within 64 positions in the same text" },
  { key: "O-window", centring: "O", mask: "window", label: "Own-text mean + window", words: "own-text mean removed, no neighbour within 64 positions in the same text" },
];
const MEMS = ["none", "2", "8", "64", "whole"];
const MEM_WORDS = { none: "no memory (the current token only)", "2": "memory 2", "8": "memory 8", "64": "memory 64", whole: "the whole prefix" };
const PAST = 24, SHOW = 8, NEAR = 64;
const nf = new Intl.NumberFormat("en");
const pct = v => `${Math.round(100 * v)}%`;
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ------------------------------------------------------------ colour: OKLab mix of paper, far and near tints
const toLin = c => (c <= .04045 ? c / 12.92 : ((c + .055) / 1.055) ** 2.4);
const fromLin = c => (c <= .0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - .055);
function hexLab(hex) {
  const v = parseInt(hex.replace("#", ""), 16), [r, g, b] = [(v >> 16) & 255, (v >> 8) & 255, v & 255].map(x => toLin(x / 255));
  const l = Math.cbrt(.4122214708 * r + .5363325363 * g + .0514459929 * b), m = Math.cbrt(.2119034982 * r + .6806995451 * g + .1073969566 * b),
    s = Math.cbrt(.0883024619 * r + .2817188376 * g + .6299787005 * b);
  return [.2104542553 * l + .7936177850 * m - .0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + .4505937099 * s, .0259040371 * l + .7827717662 * m - .8086757660 * s];
}
function labRgb([L, A, B]) {
  const l = (L + .3963377774 * A + .2158037573 * B) ** 3, m = (L - .1055613458 * A - .0638541728 * B) ** 3, s = (L - .0894841775 * A - 1.2914855480 * B) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + .2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - .3413193965 * s, -.0041960863 * l - .7034186147 * m + 1.7076147010 * s];
  return `rgb(${rgb.map(c => Math.round(255 * Math.min(1, Math.max(0, fromLin(c))))).join(",")})`;
}
class Palette {
  constructor() { this.load(); }
  load() {
    this.paper = hexLab(css("--t-paper") || "#fcfcfb"); this.far = hexLab(css("--t-far") || "#92b9eb"); this.near = hexLab(css("--t-near") || "#f69b7a");
    this.cache = new Map();
  }
  // shares of other texts / same text far / same text near: the paper colour moved towards the far and near tints by
  // their shares; the tints have equal OKLab lightness, so lightness depends only on the own-text share
  mix(f, n) {
    const key = Math.round(f * 1000) * 1001 + Math.round(n * 1000);
    let c = this.cache.get(key);
    if (!c) { const p = this.paper; c = labRgb(p.map((x, i) => x + f * (this.far[i] - x) + n * (this.near[i] - x))); this.cache.set(key, c); }
    return c;
  }
}

function modelName(id, fallback) {
  const s = String(id || fallback).split("/").pop();
  if (/^pythia-/i.test(s)) return "Pythia-" + s.slice(7);
  if (/^olmo-/i.test(s)) return "OLMo-" + s.slice(5).replace(/-0724-hf$/i, "").replace(/-hf$/i, "");
  return s;
}
const showTok = s => s.replace(/\n/g, "↵").replace(/\t/g, "⇥");

export class Viewer {
  constructor(root) {
    this.root = root;
    this.state = null;
    this.pal = new Palette();
    this.ready = this.init().catch(e => { console.error("viewer:", e); root.replaceChildren(h("div", { class: "live-error", text: `Viewer not loaded: ${e.message}` })); });
  }

  // ---------------------------------------------------------------- set-up
  async init() {
    const [m, aux] = await Promise.all([manifest(), json("assets/x/viewer.json")]);
    this.m = m; this.aux = aux;
    this.pools = Object.fromEntries(Object.entries(m.candidate_pools).map(([k, r]) => [k, poolIndex(r)]));
    this.settings = new Map();
    for (const s of m.settings) this.settings.set([s.model, s.site, s.centring, s.mask, s.memory, s.candidate_pool].join("|"), s);
    this.texts = m.texts.map(t => ({ ...t, pool: t.kind === "stitched" ? `stitch${String(t.id).padStart(2, "0")}` : "corpus", info: aux.texts[String(t.id)] || {} }));
    this.models = m.models;
    this.build();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { this.pal.load(); this.legendTriangle(); this.paint(); });
  }

  setting(st = this.state) {
    const acc = ACC.find(a => a.key === st.acc), t = this.text(st.text);
    return acc && t ? this.settings.get([st.model, st.layer, acc.centring, acc.mask, st.memory, t.pool].join("|")) : undefined;
  }
  text(id) { return this.texts.find(t => t.id === +id); }

  build() {
    const r = this.root;
    r.replaceChildren();
    const seg = (name, label, values, words) => {
      const g = h("div", { class: "vw-seg", role: "radiogroup", "aria-label": label });
      for (const v of values) {
        const b = h("button", { type: "button", role: "radio", "aria-checked": "false", "data-v": String(v), text: words ? words(v) : String(v) });
        b.addEventListener("click", () => { if (!b.disabled) this.change({ [name]: name === "layer" ? +v : String(v) }); });
        g.append(b);
      }
      return g;
    };
    const field = (label, help, ctl) => h("div", { class: "vw-field" }, h("span", { class: "vw-lab" }, label, help ? h("button", { type: "button", class: "vw-help", "aria-label": `What is ${label.toLowerCase()}?`, title: "Explain", text: "?", onclick: () => this.explain(help) }) : null), ctl);
    this.selText = h("select", { "aria-label": "Text", class: "vw-text-select" });
    for (const t of this.texts) this.selText.append(h("option", { value: String(t.id), text: t.info.label || `Text ${t.id}` }));
    this.selText.addEventListener("change", () => this.change({ text: +this.selText.value, pos: null }));
    this.segModel = seg("model", "Model", this.models, v => modelName(this.m.model_metadata?.[v]?.id, v));
    this.segLayer = seg("layer", "Layer", this.m.viewer_sites);
    this.segMem = seg("memory", "Memory", MEMS);
    this.selAcc = h("select", { "aria-label": "Accounting" });
    for (const a of ACC) this.selAcc.append(h("option", { value: a.key, text: a.label }));
    this.selAcc.addEventListener("change", () => this.change({ acc: this.selAcc.value }));
    this.note = h("p", { class: "vw-note", "aria-live": "polite" });
    const bar = h("div", { class: "vw-bar" },
      h("div", { class: "vw-controls" },
        field("Text", "texts", this.selText), field("Model", "models", this.segModel), field("Layer", "layer", this.segLayer),
        field("Memory", "memory", this.segMem), field("Accounting", "accounting", this.selAcc)),
      this.note);
    // legend
    this.tri = h("canvas", { class: "vw-tri", width: 132, height: 116, "aria-hidden": "true" });
    this.legendText = h("div", { class: "vw-legend-text" });
    this.removed = h("p", { class: "vw-removed" });
    const legend = h("div", { class: "vw-legend", "aria-label": "Colour key" }, h("div", { class: "vw-tri-wrap" }, this.tri,
      h("span", { class: "vw-corner c-other", text: "other texts" }), h("span", { class: "vw-corner c-far", text: "own text, far" }), h("span", { class: "vw-corner c-near", text: "own text, near" })),
      this.legendText);
    // text, strip, card
    this.textEl = h("div", { class: "vw-text", tabindex: "-1", "aria-label": "The text, one coloured span per token" });
    this.status = h("div", { class: "vw-status", "aria-live": "polite" });
    this.strip = h("canvas", { class: "vw-strip-canvas", height: 38, "aria-label": "Shares along position; click to jump" });
    this.stripCap = h("div", { class: "vw-strip-cap" });
    const stripBox = h("div", { class: "vw-strip" }, this.stripCap, this.strip);
    this.card = h("aside", { class: "vw-card", "aria-label": "Token card", hidden: "" });
    this.hint = h("div", { class: "vw-hint" }, h("b", { text: "Click any coloured token" }), " to list its nearest neighbours as text, with the stretch of past each comparison weighs and the model's predictions beside them. ",
      h("span", { class: "mouse-only", text: "Arrow keys step through tokens; Esc closes the card." }));
    this.details = h("details", { class: "vw-details" }, h("summary", { text: "Details: this text, this setting, and what the viewer leaves out" }), h("div", { class: "vw-details-body" }));
    this.details.addEventListener("toggle", () => { if (this.details.open) this.fillDetails(); });
    const side = h("div", { class: "vw-side" }, legend, this.removed, this.hint, this.card);
    r.append(bar, h("div", { class: "vw-main" }, h("div", { class: "vw-col" }, this.status, this.textEl, stripBox), side), this.details);

    this.textEl.addEventListener("click", e => {
      const s = e.target.closest(".tk"); if (!s || s.classList.contains("burn")) return;
      this.select(+s.dataset.p, { scroll: false });
    });
    this.textEl.addEventListener("pointerover", e => {
      const s = e.target.closest(".tk"); if (!s || e.pointerType === "touch") return;
      const p = +s.dataset.p, t = this.tip(p);
      if (t) showTip(e.clientX, e.clientY, t);
    });
    this.textEl.addEventListener("pointermove", e => { const s = e.target.closest(".tk"); if (s && e.pointerType !== "touch") { const t = this.tip(+s.dataset.p); if (t) showTip(e.clientX, e.clientY, t); } });
    this.textEl.addEventListener("pointerleave", hideTip);
    document.addEventListener("keydown", e => {
      if (document.body.dataset.tab !== "viewer" || e.target.closest("input, select, textarea") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "Escape" && this.state?.pos != null) { this.select(null); e.preventDefault(); }
      if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && this.state?.pos != null && this.cur) {
        const { start, stop } = this.cur.range, p = Math.min(stop - 1, Math.max(start, this.state.pos + (e.key === "ArrowRight" ? 1 : -1)));
        if (p !== this.state.pos) this.select(p, { scroll: true });
        e.preventDefault();
      }
    });
    this.strip.addEventListener("click", e => {
      if (!this.cur) return;
      const r = this.strip.getBoundingClientRect(), L = this.cur.text.token_ids.length;
      const p = Math.max(1, Math.min(L, Math.round((e.clientX - r.left) / r.width * L + .5)));
      const span = this.spans[p - 1];
      if (span) { span.scrollIntoView({ block: "center", behavior: "smooth" }); span.classList.add("flash"); setTimeout(() => span.classList.remove("flash"), 1200); }
    });
    let raf = 0;
    window.addEventListener("scroll", () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; if (document.body.dataset.tab === "viewer") this.drawStrip(); }); }, { passive: true });
    new ResizeObserver(() => this.drawStrip()).observe(this.strip);
    this.legendTriangle();
  }

  explain(id) {
    const d = document.getElementById("primer"); if (!d) return;
    d.open = true;
    const el = document.getElementById(`primer-${id}`) || d;
    el.scrollIntoView({ block: "start", behavior: "smooth" });
    el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1400);
  }

  // ---------------------------------------------------------------- state
  defaults() {
    const first = this.texts.find(t => t.kind !== "stitched") || this.texts[0];
    return { text: first.id, model: this.models[0], layer: this.m.viewer_sites.includes(8) ? 8 : this.m.viewer_sites[0], memory: "8", acc: "C-none", pos: null };
  }
  hashFor(st) {
    const q = new URLSearchParams({ text: st.text, model: st.model, layer: st.layer, memory: st.memory, acc: st.acc });
    if (st.pos != null) q.set("pos", st.pos);
    return `#viewer?${q}`;
  }
  async applyParams(q) {
    await this.ready; if (!this.m) return;
    const d = this.state || this.defaults(), g = k => q.get(k);
    const st = {
      text: this.text(g("text")) ? +g("text") : d.text, model: this.models.includes(g("model")) ? g("model") : d.model,
      layer: this.m.viewer_sites.includes(+g("layer")) ? +g("layer") : d.layer, memory: MEMS.includes(g("memory")) ? g("memory") : d.memory,
      acc: ACC.some(a => a.key === g("acc")) ? g("acc") : d.acc, pos: g("pos") != null && g("pos") !== "" ? +g("pos") : null,
    };
    if (this.lastHash !== location.hash && this.state && st.text !== this.state.text && this.state.pos != null) this.trail = { ...this.state };
    await this.show(st, { scroll: true });
  }
  async layout() { await this.ready; if (this.m && !this.state) await this.show(this.defaults()); else this.drawStrip(); }
  change(part) { this.trail = null; this.show({ ...(this.want || this.state), ...part, pos: part.text != null ? null : (this.want || this.state).pos }, { scroll: part.text != null }); }

  // fall back to the nearest choice that has data, saying so
  resolve(st) {
    const notes = [], t = this.text(st.text);
    const has = s => !!this.setting(s);
    if (!has(st)) {
      const accs = ACC.map(a => a.key), tries = [];
      // keep the model first, then the nearest layer, then the memory, then the accounting
      for (const model of [st.model, ...this.models]) for (const layer of [st.layer, ...[...this.m.viewer_sites].sort((a, b) => Math.abs(a - st.layer) - Math.abs(b - st.layer))])
        for (const acc of [st.acc, ...accs]) for (const memory of [st.memory, ...MEMS]) tries.push({ ...st, acc, layer, model, memory });
      const ok = tries.find(has);
      if (ok) {
        if (ok.acc !== st.acc) notes.push(`${ACC.find(a => a.key === st.acc).label} is not computed for ${t.kind === "stitched" ? "a stitched text" : "this text"}`);
        if (ok.layer !== st.layer) notes.push(`layer ${st.layer} has no data for ${modelName(this.m.model_metadata?.[ok.model]?.id, ok.model)}`);
        if (ok.model !== st.model) notes.push(`${modelName(this.m.model_metadata?.[st.model]?.id, st.model)} has no data for this text`);
        if (ok.memory !== st.memory) notes.push(`memory ${st.memory} has no data here`);
        return { st: ok, note: notes.length ? `Showing the nearest available view: ${notes.join("; ")}.` : "" };
      }
      return { st, note: "No neighbour data for this text." };
    }
    return { st, note: "" };
  }

  // Load first, then commit: this.state and this.cur always describe the same loaded snapshot. While a new view
  // loads, the old text is greyed and not clickable, the card is closed, and a failed load restores the old controls.
  async show(want, { scroll = false } = {}) {
    const { st, note } = this.resolve(want);
    const token = (this.loading = {}), prev = this.state;
    this.want = st;
    this.cardReq = {};                                        // any card still loading belongs to the old view
    this.syncControls(note, st);
    const t = this.text(st.text), s = this.setting(st);
    if (!s) { this.status.textContent = "No neighbour data for this choice."; this.syncControls("", prev || st); return; }
    this.textEl.classList.add("stale"); this.card.hidden = true; this.clearMarks();
    this.status.textContent = "Loading…";
    let text, rows;
    try {
      [text, rows] = await Promise.all([asset(t.text_file), asset(s.files[String(t.id)])]);
      const r = this.pools[s.candidate_pool].byText.get(t.id);
      if (!r || rows.rows !== r.stop - r.start) throw new Error(`neighbour rows (${rows.rows}) do not match the text's census range`);
    } catch (e) {
      if (token !== this.loading) return;
      this.want = null;
      console.error("viewer data:", e);
      this.textEl.classList.remove("stale");
      if (prev && this.cur) { this.syncControls("", prev); if (prev.pos != null) await this.select(prev.pos, { keepHash: true }); }
      this.status.textContent = `Could not load that view (${e.message}); still showing the previous one.`;
      return;
    }
    if (token !== this.loading) return;                       // a later change won
    this.want = null;
    const range = this.pools[s.candidate_pool].byText.get(t.id);
    const fresh = !this.cur || this.cur.textId !== t.id;
    this.state = st;                                          // commit state and data together
    this.cur = { textId: t.id, text, rows, range, setting: s, t };
    const url = this.hashFor(st);
    if (location.hash !== url) history.replaceState(null, "", url);
    this.lastHash = location.hash;
    document.title = `${t.info.short || "Text " + t.id} · Viewer · TransformerModes`;
    if (fresh) this.renderText();
    this.paint();
    this.textEl.classList.remove("stale");
    this.status.textContent = "";
    this.legendWords();
    if (this.details.open) this.fillDetails();
    if (st.pos != null && (st.pos < range.start || st.pos >= range.stop)) st.pos = null;
    if (st.pos != null) await this.select(st.pos, { scroll, keepHash: true });
    else { this.card.hidden = true; this.hint.hidden = false; this.clearMarks(); if (scroll && fresh) window.scrollTo({ top: 0 }); }
    this.prefetch();
  }

  syncControls(note, st = this.state) {
    const t = this.text(st.text);
    this.selText.value = String(st.text);
    this.selText.title = t.info.label || `Text ${t.id}`;
    const mark = (g, v, ok) => g.querySelectorAll("button").forEach(b => { b.setAttribute("aria-checked", String(b.dataset.v === String(v))); b.disabled = !ok(b.dataset.v); });
    mark(this.segModel, st.model, v => [...this.settings.values()].some(s => s.model === v && s.candidate_pool === t.pool));
    mark(this.segLayer, st.layer, v => !!this.setting({ ...st, layer: +v }));
    mark(this.segMem, st.memory, v => !!this.setting({ ...st, memory: v }));
    this.selAcc.value = st.acc;
    [...this.selAcc.options].forEach(o => { o.disabled = !this.setting({ ...st, acc: o.value }); });
    this.note.textContent = note || "";
    this.note.hidden = !note;
  }

  // ---------------------------------------------------------------- text and colours
  renderText() {
    const { text, range, t } = this.cur, toks = text.token_strings, frag = document.createDocumentFragment();
    const segStarts = new Set((text.segments || []).slice(1).map(s => s.position_start));
    this.spans = [];
    for (let p = 1; p <= toks.length; p++) {
      const s = document.createElement("span");
      s.className = p < range.start || p >= range.stop ? "tk burn" : "tk";
      if (segStarts.has(p)) s.classList.add("seg-start");
      s.dataset.p = p;
      const str = toks[p - 1];
      if (str.includes("\n")) {
        const parts = str.split("\n");
        parts.forEach((part, i) => { if (part) s.append(part); if (i < parts.length - 1) { s.append(h("span", { class: "nl", text: "↵" }), "\n"); } });
      } else s.textContent = str;
      frag.append(s); this.spans.push(s);
    }
    this.textEl.replaceChildren(frag);
    this.textEl.classList.toggle("stitched", t.kind === "stitched");
  }
  counts(p) {
    const { rows, range } = this.cur, i = p - range.start;
    if (i < 0 || i >= rows.rows) return null;
    const oc = rows.origin_counts.data;
    return [oc[3 * i], oc[3 * i + 1], oc[3 * i + 2]];
  }
  paint() {
    if (!this.cur || !this.spans) return;
    const { range } = this.cur;
    for (let p = range.start; p < range.stop; p++) {
      const span = this.spans[p - 1], c = this.counts(p), n = c[0] + c[1] + c[2];
      if (!n) { span.style.backgroundColor = ""; span.classList.add("none"); continue; }
      span.classList.remove("none");
      span.style.backgroundColor = this.pal.mix(c[1] / n, c[2] / n);
    }
    this.drawStrip();
  }
  tip(p) {
    if (!this.cur) return null;
    const c = this.counts(p);
    const head = `position ${nf.format(p)}`;
    if (!c) return `<div class="tt-head">${head}</div><div class="tt-row"><span class="tt-k">burn-in: not analysed</span></div>`;
    const n = c[0] + c[1] + c[2];
    if (!n) return `<div class="tt-head">${head}</div><div class="tt-row"><span class="tt-k">no eligible neighbours</span></div>`;
    const row = (k, v, cls) => `<div class="tt-row"><span class="tt-k"><i class="sw ${cls}"></i>${k}</span><span class="tt-v">${v} of ${n}</span></div>`;
    return `<div class="tt-head">${head}</div>${row("same text, within 64", c[2], "near")}${row("same text, far", c[1], "far")}${row("other texts", c[0], "other")}`;
  }

  legendTriangle() {
    const cv = this.tri, ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const A = [W / 2, 6], B = [8, H - 6], C = [W - 8, H - 6];    // other (top), far (left), near (right)
    const img = ctx.createImageData(W, H), det = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]);
    const win = this.state && ACC.find(a => a.key === this.state.acc)?.mask === "window";
    const col = {};
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const l1 = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / det, l2 = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / det, l3 = 1 - l1 - l2;
      if (l1 < -.01 || l2 < -.01 || l3 < -.01) continue;
      const near = win ? 0 : Math.max(0, l3), far = Math.max(0, l2) + (win ? Math.max(0, l3) : 0);
      const k = Math.round(far * 20) + "," + Math.round(near * 20);
      const c = col[k] || (col[k] = this.pal.mix(Math.round(far * 20) / 20, Math.round(near * 20) / 20).match(/\d+/g).map(Number));
      const o = 4 * (y * W + x); img.data[o] = c[0]; img.data[o + 1] = c[1]; img.data[o + 2] = c[2]; img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    ctx.strokeStyle = css("--line") || "#ccc"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(...A); ctx.lineTo(...B); ctx.lineTo(...C); ctx.closePath(); ctx.stroke();
    this.tri.parentElement?.classList.toggle("win", !!win);
  }
  legendWords() {
    const st = this.state, acc = ACC.find(a => a.key === st.acc), stitched = this.cur.t.kind === "stitched";
    const parts = [h("p", {}, h("b", { text: "Background: where a token's 64 nearest neighbours are." }), " Darker means more of them in the token's own text; paper means all of them in other texts. ",
      "The hue splits the own-text part: ", h("b", { class: "k-far", text: "blue" }), " far away in it, ", h("b", { class: "k-near", text: "orange" }), " within 64 positions; an even split is greyish mauve.")];
    if (acc.mask === "window") parts.push(h("p", { class: "vw-rule" }, h("b", { text: "Window rule: " }), "no neighbour may sit within 64 positions in the same text, so orange cannot appear. That is imposed, not found."));
    if (stitched) parts.push(h("p", { class: "vw-rule" }, h("b", { text: "Stitched text: " }), "it is compared only with itself (two documents cut into alternating segments), so no neighbour can come from another text. Dashed marks show where a segment starts."));
    parts.push(h("p", { class: "vw-small" }, "Near means within 64 positions. Grey text: burn-in, not analysed. Hatched: no eligible neighbour."));
    this.legendText.replaceChildren(...parts);
    this.legendTriangle();
    const key = [st.model, acc.centring, st.layer, st.memory].join("|"), rm = this.aux.removed?.[key];
    this.removed.hidden = acc.mask !== "window" || stitched;
    this.removed.replaceChildren(...(rm == null ? [h("b", { text: "Removed by the window: " }), "not available for this setting."] :
      [h("b", { text: `Removed by the window: ${pct(rm)}` }), ` of a row's walk weight, on average, sat on same-text tokens within 64 positions before the rule removed them (sampled suffix queries of all texts, ${modelName(this.m.model_metadata?.[st.model]?.id, st.model)}, layer ${st.layer}, ${MEM_WORDS[st.memory]}). It is an average, not shown per token.`]));
  }

  // ---------------------------------------------------------------- strip (shares along position; also a minimap)
  drawStrip() {
    const cv = this.strip; if (!this.cur || !cv.isConnected) return;
    const dpr = window.devicePixelRatio || 1, W = Math.max(100, cv.clientWidth), H = 38;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const { range, text } = this.cur, L = text.token_ids.length, X = p => (p - 1) / L * W, cw = W / L;
    const colour = { other: css("--o-other"), far: css("--o-far"), near: css("--o-near") }, top = 4, hh = H - 12;
    ctx.fillStyle = css("--t-burn"); ctx.fillRect(0, top, X(range.start), hh);
    // bin to at most one column per pixel
    const bins = Math.min(range.stop - range.start, Math.floor(W));
    const per = (range.stop - range.start) / bins;
    for (let b = 0; b < bins; b++) {
      let o = 0, f = 0, n = 0, k = 0;
      for (let p = range.start + Math.floor(b * per); p < range.start + Math.floor((b + 1) * per); p++) { const c = this.counts(p), s = c[0] + c[1] + c[2]; if (s) { o += c[0] / s; f += c[1] / s; n += c[2] / s; k++; } }
      const x = X(range.start + b * per), w = Math.max(cw, per * cw) + .4;
      if (!k) continue;
      let y = top + hh;
      for (const [v, key] of [[o / k, "other"], [f / k, "far"], [n / k, "near"]]) { const hgt = v * hh; ctx.fillStyle = colour[key]; ctx.fillRect(x, y - hgt, w, hgt); y -= hgt; }
    }
    // segment starts (stitched) and the viewport bracket
    ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1;
    for (const s of (text.segments || []).slice(1)) { ctx.beginPath(); ctx.setLineDash([2, 2]); ctx.moveTo(X(s.position_start) + .5, top); ctx.lineTo(X(s.position_start) + .5, top + hh); ctx.stroke(); }
    ctx.setLineDash([]);
    const vis = this.visibleRange();
    if (vis) { ctx.strokeStyle = css("--fg"); ctx.lineWidth = 1.5; ctx.strokeRect(X(vis[0]) + .75, 1, Math.max(3, X(vis[1] + 1) - X(vis[0]) - 1.5), H - 7); }
    if (this.state.pos != null) { ctx.fillStyle = css("--fg"); const x = X(this.state.pos + .5); ctx.beginPath(); ctx.moveTo(x - 4, H); ctx.lineTo(x + 4, H); ctx.lineTo(x, H - 6); ctx.fill(); }
    this.stripCap.replaceChildren(h("span", { text: "Along the text: share of the 64 neighbours from " }), h("i", { class: "sw other" }), " other texts ", h("i", { class: "sw far" }), " same text, far ", h("i", { class: "sw near" }), " same text, within 64",
      h("span", { class: "vw-strip-tail", text: " · click to jump" }));
  }
  visibleRange() {
    if (!this.spans?.length) return null;
    const r = this.textEl.getBoundingClientRect(), top = Math.max(r.top, 0), bot = Math.min(r.bottom, innerHeight);
    if (bot <= top) return null;
    const find = y => { let lo = 0, hi = this.spans.length - 1; while (lo < hi) { const mid = (lo + hi) >> 1; if (this.spans[mid].getBoundingClientRect().bottom < y) lo = mid + 1; else hi = mid; } return lo; };
    return [find(top) + 1, find(bot) + 1];
  }

  // ---------------------------------------------------------------- token card
  clearMarks() { this.textEl.querySelectorAll(".sel, .nb").forEach(s => { s.classList.remove("sel", "nb"); delete s.dataset.rank; }); }
  async select(p, { scroll = false, keepHash = false } = {}) {
    if (!this.cur) return;
    this.clearMarks();
    if (p == null) { this.state.pos = null; this.card.hidden = true; this.hint.hidden = false; history.replaceState(null, "", this.hashFor(this.state)); this.lastHash = location.hash; this.drawStrip(); return; }
    this.state.pos = p;
    if (!keepHash) { history.replaceState(null, "", this.hashFor(this.state)); this.lastHash = location.hash; }
    const span = this.spans[p - 1];
    span?.classList.add("sel");
    if (scroll && span) span.scrollIntoView({ block: innerWidth <= 960 ? "start" : "center" });   // on a phone the card covers the lower part
    this.drawStrip();
    await this.renderCard(p);
  }

  async predictions(textId) {
    const t = this.text(textId), path = t?.predictions?.[this.state.model];
    return path ? asset(path) : null;
  }
  predRow(pred, p) {
    if (!pred) return null;
    const i = p - 1, k = pred.token_id.columns;
    if (i < 0 || i >= pred.rows) return null;
    return [...Array(k).keys()].map(j => ({ id: pred.token_id.data[i * k + j], p: pred.probability.data[i * k + j] }));
  }
  tokName(id) { const s = this.vocab?.[String(id)]; return s == null ? `#${id}` : `“${showTok(s)}”`; }

  async renderCard(p) {
    const { rows, range, setting, t } = this.cur, st = this.state, acc = ACC.find(a => a.key === st.acc);
    const i = p - range.start, k = rows.index.columns, card = this.card;
    const req = (this.cardReq = {}), snapshot = this.cur;
    const c = this.counts(p), n = c[0] + c[1] + c[2];
    const pool = this.pools[setting.candidate_pool];
    const nb = [];
    for (let r = 0; r < k; r++) {
      const j = rows.index.data[i * k + r];
      if (j === this.m.layouts.neighbours[0].missing) continue;
      const [nt, np] = pool.locate(j);
      nb.push({ rank: r + 1, j, text: nt, pos: np, d: rows.distance ? rows.distance.data[i * k + r] : null });
    }
    const flags = rows.tie_flags.data[i], tiedShown = flags & 15, cutTie = !!(flags & 16), leftOut = !!(flags & 32);
    // bits 0-3 count the shown ranks inside the tied cutoff group; they are the last valid shown ranks
    nb.forEach((x, r) => { x.tied = cutTie && r >= nb.length - tiedShown; });
    // load what the card needs: neighbour texts, predictions, display strings
    const texts = new Map([[t.id, this.cur.text]]);
    const need = [...new Set(nb.map(x => x.text))].filter(id => !texts.has(id));
    card.hidden = false; this.hint.hidden = true;
    if (!card.firstChild || card.dataset.p !== String(p)) card.replaceChildren(h("p", { class: "vw-card-loading", text: "Loading neighbours…" }));
    card.dataset.p = p;
    let predQ, predN = new Map();
    try {
      const [loaded, vocab, pq] = await Promise.all([Promise.all(need.map(id => asset(this.text(id).text_file))), this.vocab ? this.vocab : asset(this.m.token_card.prediction_display.file), this.predictions(t.id)]);
      need.forEach((id, x) => texts.set(id, loaded[x]));
      this.vocab = vocab; predQ = pq;
      const pn = await Promise.all([...texts.keys()].map(id => this.predictions(id)));
      [...texts.keys()].forEach((id, x) => predN.set(id, pn[x]));
    } catch (e) { console.error("token card:", e); if (req === this.cardReq && snapshot === this.cur) card.replaceChildren(h("p", { class: "live-error", text: `Card not loaded: ${e.message}` })); return; }
    if (req !== this.cardReq || snapshot !== this.cur) return;

    // same-text neighbours are marked in the text too
    nb.forEach(x => { if (x.text === t.id) { const s = this.spans[x.pos - 1]; if (s) { s.classList.add("nb"); s.dataset.rank = x.rank; } } });

    const a = setting.alpha, whole = st.memory === "whole";
    const weight = s => (whole ? 1 : s === 0 ? 1 : a ** s);
    const share = whole ? Math.min(PAST, p) / p : 1 - a ** PAST;
    const tokText = (id, q) => texts.get(id).token_strings[q - 1];
    const past = (id, q, mark) => {
      const box = h("div", { class: "vw-past" + (mark ? " q" : "") }), inner = h("div", { class: "vw-past-in" });
      for (let s = Math.min(PAST - 1, q - 1); s >= 0; s--) {
        const w = weight(s), el = h("span", { class: "pt" + (s === 0 ? " cur" : ""), "data-lag": s, text: showTok(tokText(id, q - s)) });
        el.style.setProperty("--w", w.toFixed(3));
        inner.append(el);
      }
      box.append(inner);
      return box;
    };
    const qPreds = this.predRow(predQ, p) || [], qIds = new Set(qPreds.map(x => x.id));
    const preds = (list, isQuery) => {
      if (!list) return h("span", { class: "vw-preds none", text: "predictions not stored" });
      return h("span", { class: "vw-preds" }, h("span", { class: "vw-pl", text: isQuery ? "it predicts next " : "predicts " }),
        ...list.flatMap((x, j) => [j ? " · " : "", h(!isQuery && qIds.has(x.id) ? "b" : "span", { class: !isQuery && qIds.has(x.id) ? "shared" : "", title: `vocabulary id ${x.id}${!isQuery && qIds.has(x.id) ? "; also among the clicked token's top 3" : ""}` }, `${this.tokName(x.id)} ${pct(x.p)}`)]));
    };
    const where = x => {
      if (x.text === t.id) {
        const d = x.pos - p, near = Math.abs(d) <= NEAR;
        const words = `same text, ${nf.format(Math.abs(d))} position${Math.abs(d) === 1 ? "" : "s"} ${d < 0 ? "earlier" : "later"}`;
        const src = t.kind === "stitched" ? this.sourceOf(this.cur.text, x.pos) : null, qsrc = t.kind === "stitched" ? this.sourceOf(this.cur.text, p) : null;
        return [h("i", { class: `sw ${near ? "near" : "far"}` }), words, src ? h("span", { class: "vw-src", text: ` · source ${src}${src === qsrc ? " (same source)" : ""}` }) : null];
      }
      const info = this.text(x.text)?.info || {};
      return [h("i", { class: "sw other" }), `text ${x.text}, position ${nf.format(x.pos)}`, info.subset ? h("span", { class: "vw-src", text: ` · ${info.subset}` }) : null];
    };

    const modelLabel = modelName(this.m.model_metadata?.[st.model]?.id, st.model);
    const head = h("div", { class: "vw-card-head" },
      h("div", { class: "vw-card-title" }, h("b", { class: "vw-tok", text: `“${showTok(tokText(t.id, p))}”` }), h("span", { text: ` text ${t.id}, position ${nf.format(p)}` })),
      h("button", { type: "button", class: "vw-close", "aria-label": "Close the card", text: "×", onclick: () => this.select(null) }));
    const sub = h("p", { class: "vw-card-sub", text: `${modelLabel} · layer ${st.layer} · ${MEM_WORDS[st.memory]} · ${acc.words}` });
    const bar = h("div", { class: "vw-bar3", role: "img", "aria-label": `${c[2]} within 64, ${c[1]} far, ${c[0]} other texts` });
    if (n) for (const [v, cls] of [[c[0], "other"], [c[1], "far"], [c[2], "near"]]) if (v) bar.append(h("i", { class: cls, style: { flexGrow: v } }));
    const counts = h("div", { class: "vw-counts" }, n ? bar : null,
      h("p", {}, ...(n ? [h("b", { text: `Its ${n} nearest neighbours: ` }), `${c[2]} in the same text within 64 positions, ${c[1]} in the same text further away, ${c[0]} in other texts.`]
        : [h("b", { text: "No eligible neighbours" }), " under this accounting."])));
    const notes = [];
    if (n && n < 64) notes.push(`Only ${n} tokens were eligible, so the list is shorter than 64.`);
    if (cutTie) notes.push(`The 64th place is tied${leftOut ? "; some tied tokens were left out, chosen by (text, position)" : ""}. ${tiedShown ? `Marked ≈ below: the shown ranks inside that tie.` : "The tie lies beyond the ranks shown."}`);
    if (!rows.distance) notes.push("Distances were not stored for this setting; ranks only.");
    const ruler = h("div", { class: "vw-ruler" }, h("span", { class: "vw-ruler-lab" }, h("b", { text: "Underline = memory weight of each earlier token" }), ` (aligned at the end, so lags line up from the right; a weight, not its contribution to the similarity). The last ${Math.min(PAST, p)} tokens carry ${pct(share)} of the weight${whole ? " of this token's whole prefix; at the whole prefix two pasts of different lengths share only their overlapping lags" : ""}.`));
    const list = h("ol", { class: "vw-nbs" });
    const qrow = h("li", { class: "vw-nb vw-q" }, h("div", { class: "vw-nb-head" }, h("span", { class: "vw-rank", text: "" }), h("b", { text: "The clicked token" })), past(t.id, p, true), preds(qPreds, true));
    list.append(qrow);
    for (const x of nb) {
      const sim = x.d == null || !Number.isFinite(x.d) ? null : 1 - x.d;
      const href = this.hashFor({ ...st, text: x.text, pos: x.pos });
      const li = h("li", { class: "vw-nb", "data-text": x.text, "data-pos": x.pos, "data-rank": x.rank },
        h("div", { class: "vw-nb-head" }, h("span", { class: "vw-rank", text: x.rank }),
          h("span", { class: "vw-sim", title: sim == null ? "distance not stored" : `cosine distance ${x.d.toFixed(5)}`, text: sim == null ? "—" : sim.toFixed(3) }),
          x.tied ? h("span", { class: "vw-tie", title: "tied at the cutoff", text: "≈" }) : null,
          h("span", { class: "vw-where" }, ...where(x)),
          h("a", { class: "vw-open", href, title: x.text === t.id ? "Go to this token and open its card" : "Open this token in its own text", text: x.text === t.id ? "go →" : "open →" })),
        past(x.text, x.pos), preds(this.predRow(predN.get(x.text), x.pos), false));
      li.addEventListener("pointerenter", () => { if (x.text === t.id) this.spans[x.pos - 1]?.classList.add("hot"); });
      li.addEventListener("pointerleave", () => this.spans[x.pos - 1]?.classList.remove("hot"));
      li.querySelector(".vw-open").addEventListener("click", () => { this.trail = { ...st }; });
      list.append(li);
    }
    const back = this.trail && (this.trail.text !== st.text || this.trail.pos !== p) ? h("a", { class: "vw-back", href: this.hashFor(this.trail), text: `← back to text ${this.trail.text}, position ${nf.format(this.trail.pos)}` }) : null;
    const listHead = h("p", { class: "vw-list-head" }, h("b", { text: `The ${nb.length} nearest` }), ` of the ${n}, with similarity (1 − cosine distance between the two memory-weighted pasts). `, h("span", { class: "vw-small", text: "Bold prediction = also in the clicked token's top 3." }));
    card.replaceChildren(...[head, back, sub, counts, ...notes.map(t => h("p", { class: "vw-tienote", text: t })), listHead, ruler, list].filter(Boolean));
    card.dataset.p = p;
    // hovering any earlier token highlights the same lag in every row
    card.querySelectorAll(".vw-past").forEach(b => {
      b.addEventListener("pointerover", e => { const s = e.target.closest(".pt"); if (!s) return; card.querySelectorAll(".pt.lag").forEach(x => x.classList.remove("lag")); card.querySelectorAll(`.pt[data-lag="${s.dataset.lag}"]`).forEach(x => x.classList.add("lag")); });
      b.addEventListener("pointerleave", () => card.querySelectorAll(".pt.lag").forEach(x => x.classList.remove("lag")));
      requestAnimationFrame(() => { b.scrollLeft = b.scrollWidth; });
    });
  }
  sourceOf(text, p) { const s = (text.segments || []).find(g => p >= g.position_start && p < g.position_stop); return s ? s.source : null; }

  // ---------------------------------------------------------------- details
  fillDetails() {
    if (!this.cur) return;
    const body = this.details.querySelector(".vw-details-body"), { t, text, range, setting, rows } = this.cur, st = this.state;
    let o = 0, f = 0, n = 0, q = 0, short = 0;
    for (let p = range.start; p < range.stop; p++) { const c = this.counts(p), s = c[0] + c[1] + c[2]; if (s) { o += c[0] / s; f += c[1] / s; n += c[2] / s; q++; } if (s < 64) short++; }
    const meta = this.m.model_metadata?.[st.model] || {};
    const dl = pairs => h("dl", { class: "vw-dl" }, ...pairs.flatMap(([k, v]) => [h("dt", { text: k }), h("dd", {}, v)]));
    const segs = text.segments ? h("table", { class: "vw-segs" }, h("thead", {}, h("tr", {}, ...["Segment", "Source", "Positions", "Source positions"].map(x => h("th", { text: x })))),
      h("tbody", {}, ...text.segments.map((s, i) => h("tr", {}, h("td", { text: i + 1 }), h("td", { text: s.source }), h("td", { text: `${s.position_start}–${s.position_stop - 1}` }), h("td", { text: `${s.source_start + 1}–${s.source_stop}` }))))) : null;
    body.replaceChildren(
      h("section", {}, h("h3", { text: "This text" }), dl([
        ["Text", `${t.id}${t.kind === "stitched" ? ` (stitched, ${text.schedule} segments)` : ""}`], ["Subset", text.subset || (t.kind === "stitched" ? "two sources, A and B (see segments)" : "—")],
        ["File index", text.file_index ?? "—"], ["SHA-256 of the document", h("code", { text: text.text_sha256 || "—" })],
        ["Tokens", `${nf.format(text.token_ids.length)}; positions ${range.start}–${range.stop - 1} are analysed, earlier ones are burn-in`]]), segs),
      h("section", {}, h("h3", { text: "This setting" }), dl([
        ["Model", `${meta.id || st.model}${meta.revision ? ` @ ${String(meta.revision).slice(0, 10)}` : ""}`], ["Layer", `hidden state ${st.layer} (block output)`],
        ["Memory", `${MEM_WORDS[st.memory]}, weight a = ${setting.alpha} per step back`], ["Accounting", ACC.find(a => a.key === st.acc).words],
        ["Reading", "cosine: each centred token vector scaled to unit length, then the memory-weighted past scaled to unit length"], ["Candidates", `${nf.format(this.pools[setting.candidate_pool].size)} tokens (${setting.candidate_pool === "corpus" ? "every analysed token of every text" : "this stitched text only"})`],
        ["Stored file", h("code", { text: setting.source })]])),
      h("section", {}, h("h3", { text: "This text under this setting" }), dl([
        ["Mean shares over its tokens", q ? `${pct(n / q)} same text within 64 · ${pct(f / q)} same text far · ${pct(o / q)} other texts` : "—"],
        ["Tokens with fewer than 64 eligible neighbours", nf.format(short)]])),
      h("section", {}, h("h3", { text: "Not in the viewer" }), h("p", {}, "The dot reading (loud tokens count more), the other-texts-only accounting, own-text mean without the window, and layers 2, 6, 10 and 14 are summarised in ",
        h("a", { href: "#experiments/x-e1-neighbours", text: "Experiments, E1" }), ". The viewer shows the 8 nearest of the 64 for each token; the colours use all 64.")));
  }

  prefetch() {
    const st = this.state, i = MEMS.indexOf(st.memory), t = this.text(st.text);
    const go = () => { for (const mem of [MEMS[i + 1], MEMS[i - 1]]) { if (!mem) continue; const s = this.setting({ ...st, memory: mem }); if (s) asset(s.files[String(t.id)]).catch(() => {}); } };
    (window.requestIdleCallback || (f => setTimeout(f, 400)))(go);
  }
}
