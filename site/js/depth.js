// "Depth embedding" tab (depth_tab.html), adapted from DepthEmbedding's embedding viewer (site/js/embedding.js) with
// tokens in place of images. Data: dd/ = the output of exp1/pack_depth.py (see its docstring for every file).
// Hash state: #depth?m=<pythia|olmo|both>&s=<set>&t=<strength>&x=<mode>&y=<mode>&c=<colour>&o=<overlay>&fix=<0|1>&l=<layer>
const $ = id => document.getElementById(id);
const DD = "dd/";
const cache = new Map();
function once(key, fn) { if (!cache.has(key)) { const p = fn().catch(e => { cache.delete(key); throw e; }); cache.set(key, p); } return cache.get(key); }
const getJSON = path => once(path, async () => { const r = await fetch(DD + path); if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`); return r.json(); });
const getBin = (path, bytes) => once(path, async () => {
  const r = await fetch(DD + path); if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  const b = await r.arrayBuffer(); if (b.byteLength !== bytes) throw new Error(`${path}: ${b.byteLength} bytes, expected ${bytes}`);
  return new Int8Array(b);
});
const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function hexRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = [...h].map(c => c + c).join(""); return [0, 2, 4].map(o => parseInt(h.slice(o, o + 2), 16)); }
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const esc = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const minus = s => s.replace(/^-/, "−");
function num(x, d = 3) {
  if (x == null || !isFinite(x)) return "–";
  if (x === 0) return "0";
  const a = Math.abs(x);
  return minus(a >= 1000 ? Math.round(x).toLocaleString("en") : a >= 0.001 ? (+x.toPrecision(d)).toString() : x.toExponential(d - 1));
}
const int = n => n.toLocaleString("en");
const LAYERS_PER_SECOND = 2.2;
// default axis limits: each axis holds the share meta.axis_quantile (exp1/pack_depth.py) of its mode's construction-token
// coordinates; the rest are drawn on the frame's edge as open circles and counted on the plot
let QUANT = 0.99;
const LIMITS = {
  q: () => `fit ${100 * QUANT}% of the coordinates, each axis to its own mode (the rest drawn on the edge and counted)`,
  all: () => "fit every token, each axis to its own mode",
  one: () => "fit every token, one scale for both axes (the plain view)",
};
function quantile(sorted, q) { return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))] : 0; }
// per axis: the largest |value| and the QUANT quantile of |value|, at each layer and over all layers
function axisLimits(setCol, overCol, L) {
  const M = setCol.v.length / L, out = { max: { L: new Float64Array(L), all: 0 }, q: { L: new Float64Array(L), all: 0 } }, pooled = [];
  for (let l = 0; l < L; l++) {
    const a = [];
    for (let i = 0; i < M; i++) { const v = Math.abs(setCol.v[l * M + i]); if (isFinite(v)) a.push(v); }
    a.sort((x, y) => x - y); pooled.push(...a);
    out.q.L[l] = Math.max(quantile(a, QUANT), 1e-12);
    out.max.L[l] = Math.max(setCol.mx[l], overCol ? overCol.mx[l] : 0, 1e-12);
  }
  pooled.sort((x, y) => x - y);
  out.q.all = Math.max(quantile(pooled, QUANT), 1e-12); out.max.all = Math.max(...out.max.L);
  return out;
}

// a token string as shown: newlines and tabs made visible, nothing else changed
function tokText(s) { return s.replace(/\n/g, "↵").replace(/\t/g, "⇥"); }

// int8 coordinates of one file, square-root companded with one scale per (layer, mode):
// value = sign(q) (q / 127)^2 scale[l][k]; -128 = withheld (NaN). See exp1/pack_depth.py.
class Coords {
  constructor(q, f) {
    this.q = q; [this.L, this.M, this.K] = f.shape; this.scales = f.scales; this.withheld = new Set(f.withheld_modes || []);
    this.f = new Float64Array(this.L * this.K);
    for (let l = 0; l < this.L; l++) for (let k = 0; k < this.K; k++) this.f[l * this.K + k] = f.scales[l][k] / (127 * 127);
  }
  get(l, i, k) { const v = this.q[(l * this.M + i) * this.K + k]; return v === -128 ? NaN : (v < 0 ? -v * v : v * v) * this.f[l * this.K + k]; }
  // (L * M) arrays of one mode, and the largest |value| at each layer
  column(k) {
    const { L, M } = this, out = new Float32Array(L * M), mx = new Float64Array(L);
    for (let l = 0; l < L; l++) for (let i = 0; i < M; i++) { const v = this.get(l, i, k); out[l * M + i] = v; if (Math.abs(v) > mx[l]) mx[l] = Math.abs(v); }
    return { v: out, mx };
  }
}

export class DepthView {
  constructor(root) {
    this.root = root;
    this.st = { m: "pythia", s: null, t: "tstar", x: 1, y: 2, c: "text", o: "none", fix: true, sq: false, lim: "q", l: 0 };
    this.panels = []; this.hl = []; this.hover = null; this.playing = false; this.lastHash = null; this.tableModel = null; this.extK = null; this.agreeK = "16";
    this.ready = this.init().catch(e => { console.error(e); $("dp-status").textContent = `The depth data could not be loaded: ${e.message}`; this.failed = true; });
  }

  // ------------------------------------------------ data that every view needs
  async init() {
    const [meta, rows, texts] = await Promise.all([getJSON("meta.json"), getJSON("rows.json"), getJSON("texts.json")]);
    this.meta = meta; this.rows = rows; this.texts = texts;
    if (meta.axis_quantile > 0 && meta.axis_quantile <= 1) QUANT = meta.axis_quantile;
    this.ctxKeys = rows.contexts;
    this.ctx = rows.contexts.map(k => ({ key: k, ...texts.contexts[k] }));
    this.models = meta.models; this.modelName = Object.fromEntries(meta.models.map(m => [m.key, m.name]));
    this.strengths = meta.strengths; this.sets = meta.sets; this.setBy = Object.fromEntries(meta.sets.map(s => [s.key, s]));
    this.unit = (m, s) => meta.units[`${m}/${s}`];
    this.has = (m, s) => this.unit(m, s)?.status === "packed";
    const first = this.sets.find(s => s.key === "d32" && this.models.some(m => this.has(m.key, s.key))) || this.sets.find(s => this.models.some(m => this.has(m.key, s.key)));
    this.st.s = first ? first.key : this.sets[0].key;
    this.st.m = this.models.find(m => this.has(m.key, this.st.s))?.key || this.models[0].key;
    this.buildControls(); this.bind(); this.primerMemory();
  }
  label(ctxKey) {
    const c = this.texts.contexts[ctxKey];
    if (c.kind === "natural") return `${ctxKey} · ${c.subset}`;
    if (c.kind === "random") return `${ctxKey} · random tokens`;
    return `${ctxKey} · document ${c.source_context}, shuffled`;
  }
  idAt(ci, p) { const ids = this.ctx[ci].ids; return ids ? ids[p - 1] : null; }
  str(id) { return id == null ? "?" : this.texts.vocab[id] ?? "?"; }

  // ------------------------------------------------ controls
  seg(el, name, items, value) {
    el.replaceChildren(...items.map(it => {
      const l = document.createElement("label"), i = document.createElement("input"), s = document.createElement("span");
      i.type = "radio"; i.name = name; i.value = it.value; i.checked = it.value === value; i.disabled = !!it.disabled;
      s.textContent = it.label; if (it.title) l.title = it.title; l.append(i, s); return l;
    }));
  }
  shownModels() { return this.st.m === "both" ? this.models.map(m => m.key) : [this.st.m]; }
  buildControls() {
    const st = this.st;
    this.seg($("dp-model"), "dp-model", [...this.models.map(m => ({ value: m.key, label: m.name })), { value: "both", label: "Both" }], st.m);
    $("dp-set").replaceChildren(...this.sets.map(s => {
      const here = this.models.filter(m => this.has(m.key, s.key)).map(m => m.name);
      const o = new Option(`${s.key} · ${s.contexts.length === 1 ? "1 text" : `${s.contexts.length} ${s.contexts[0][0] === "r" ? "sequences" : "texts"}`}, ${int(s.n)} tokens${here.length === this.models.length ? "" : here.length ? ` (${here.join(", ")} only so far)` : " (not computed yet)"}`, s.key);
      o.disabled = !here.length; return o;
    }));
    this.seg($("dp-strength"), "dp-strength", this.strengths.map(s => ({ value: s.key, label: s.label })), st.t);
    $("dp-lim").replaceChildren(...Object.entries(LIMITS).map(([k, v]) => new Option(v(), k)));
    this.seg($("dp-agree-k"), "dp-agree-k", ["8", "16", "64"].map(k => ({ value: k, label: `k = ${k}` })), this.agreeK);
  }
  syncControls() {
    const st = this.st, set = this.setBy[st.s];
    this.root.querySelectorAll("input[name=dp-model]").forEach(r => { r.checked = r.value === st.m; });
    this.root.querySelectorAll("input[name=dp-strength]").forEach(r => { r.checked = r.value === st.t; });
    $("dp-set").value = st.s;
    const texts = set.contexts.length, per = Math.round(set.n / texts);
    $("dp-set-hint").textContent = texts === 1 ? `All ${int(set.n)} tokens of ${this.label(set.contexts[0])}.`
      : `${int(per)} tokens from each of ${texts} ${set.contexts[0][0] === "r" ? "random sequences" : "texts"} (${set.contexts[0]}–${set.contexts.at(-1)}), ${int(set.n)} in all.`;
    // axes: the modes of the first shown model that is packed, with its flags
    const rec = this.panels.find(p => p.rec)?.rec;
    for (const [id, a] of [["dp-ax", "x"], ["dp-ay", "y"]]) {
      const sel = $(id), n = this.meta.modes_shown;
      sel.replaceChildren(...Array.from({ length: n }, (_, j) => {
        const md = rec?.modes[j], flags = md ? this.flagText(md, true) : "";
        return new Option(`${j + 1}${flags ? ` · ${flags}` : ""}`, j + 1);
      }));
      sel.value = st[a];
    }
    $("dp-axes-hint").innerHTML = this.axesHint();
    $("dp-fix").checked = st.fix; $("dp-sq").checked = st.sq; $("dp-lim").value = st.lim;
    const hasPanel = st.o.startsWith("panel:");
    [...$("dp-color").options].forEach(o => { o.disabled = o.value === "cohort" && !hasPanel; });
    $("dp-color").value = st.c;
    $("dp-color-hint").textContent = ({
      text: "Which text each token comes from. Click a name in the key to pick it out.",
      pos: "Where in its text the token sits: pale = near the start, deep = near the end.",
      freq: "How many times this token's id occurs in the construction set (log scale).",
      cohort: "Each placed token's group: in the set, a new position of a text in the set, or a text the set never saw.",
    })[st.c];
    const ov = $("dp-over"), models = this.shownModels();
    const avail = kind => models.some(m => this.has(m, st.s) && (kind.startsWith("panel:") ? this.unit(m, st.s).panels.includes(kind.slice(6)) : this.unit(m, st.s).paths.includes(kind.slice(5))));
    const opts = [new Option("none (construction set only)", "none")];
    const g1 = document.createElement("optgroup"); g1.label = "Panels (the same tokens placed into every set)";
    for (const p of this.meta.panels) { const v = `panel:${p.key}`, o = new Option(`panel: ${p.key} (${int(p.n)} tokens)${avail(v) ? "" : " · not computed yet"}`, v); o.disabled = !avail(v); g1.append(o); }
    const g2 = document.createElement("optgroup"); g2.label = "Paths (one text in reading order; d32 and rand only)";
    for (const c of this.meta.paths) { const v = `path:${c}`, o = new Option(`path: ${this.label(c)}${avail(v) ? "" : " · not available here"}`, v); o.disabled = !avail(v); g2.append(o); }
    ov.replaceChildren(...opts, g1, g2); ov.value = st.o;
    $("dp-over-hint").textContent = st.o === "none" ? "Overlay tokens placed into this embedding from outside the set; the set's own tokens then turn grey."
      : st.o.startsWith("panel:") ? "Rings mark panel tokens that are in the set (their coordinates are the set's own)."
      : "One text, every token joined in reading order. Rings mark tokens that are also in the set.";
  }
  axesHint() {
    const st = this.st, out = [];
    for (const p of this.panels.filter(p => p.rec)) {
      for (const [a, k] of [["x", st.x], ["y", st.y]]) {
        const md = p.rec.modes[k - 1], f = this.flagText(md, false);
        if (f) out.push(`${this.models.length > 1 && this.shownModels().length > 1 ? `${this.modelName[p.model]}: ` : ""}${a} = mode ${k}: ${f}.`);
      }
    }
    return out.map(esc).join("<br>");
  }
  flagText(md, short) {
    const f = [];
    if (md.band) f.push(short ? `≈ band ${md.band[0]}–${md.band[1]}` : `near-tied with mode${md.band[1] - md.band[0] > 1 ? "s" : ""} ${[...Array(md.band[1] - md.band[0] + 1)].map((_, i) => md.band[0] + i).filter(k => k !== md.k).join(", ")} (band ${md.band[0]}–${md.band[1]}${md.band[1] > this.meta.modes_shown ? `, which continues past mode ${this.meta.modes_shown}` : ""}); its own axis is not individually meaningful`);
    if (md.withheld) f.push(short ? "× withheld" : "withheld from placement (placed tokens have no coordinate on it)");
    return f.join(short ? " " : "; ");
  }
  bind() {
    const st = this.st;
    const re = (f, reload = true) => async e => { f(e); this.clamp(); this.saveHash(); if (reload) await this.load(); else { this.prepare(); this.syncControls(); this.drawAll(); } };
    this.root.querySelectorAll("#dp-model").forEach(el => el.addEventListener("change", re(e => { st.m = e.target.value; })));
    $("dp-strength").addEventListener("change", re(e => { st.t = e.target.value; }));
    $("dp-set").addEventListener("change", re(() => { st.s = $("dp-set").value; this.hl = []; }));
    $("dp-ax").addEventListener("change", re(() => { st.x = +$("dp-ax").value; this.extK = st.x; }, false));
    $("dp-ay").addEventListener("change", re(() => { st.y = +$("dp-ay").value; }, false));
    $("dp-fix").addEventListener("change", re(() => { st.fix = $("dp-fix").checked; }, false));
    $("dp-lim").addEventListener("change", re(() => { st.lim = $("dp-lim").value; }, false));
    $("dp-sq").addEventListener("change", re(() => { st.sq = $("dp-sq").checked; }, false));
    $("dp-color").addEventListener("change", re(() => { st.c = $("dp-color").value; this.hl = []; }, false));
    $("dp-over").addEventListener("change", re(() => {
      const was = st.o; st.o = $("dp-over").value; this.hl = [];
      if (st.o.startsWith("panel:") && !was.startsWith("panel:")) st.c = "cohort";
      else if (st.o.startsWith("path:") && !was.startsWith("path:")) st.c = "pos";
    }));
    $("dp-depth").addEventListener("input", () => { this.stop(); st.l = +$("dp-depth").value; this.draw(); });
    $("dp-depth").addEventListener("change", () => this.saveHash());
    $("dp-play").addEventListener("click", () => this.togglePlay());
    $("dp-jump").addEventListener("click", () => { const c = $("dp-controls"); c.scrollIntoView({ block: "start" }); c.focus({ preventScroll: true }); });
    $("dp-back").addEventListener("click", () => { this.root.querySelector(".dp-stage").scrollIntoView({ block: "start" }); $("dp-play").focus({ preventScroll: true }); });
    $("dp-modes-model").addEventListener("change", e => { this.tableModel = e.target.value; this.drawModes(); this.drawExtremes(); });
    $("dp-agree-k").addEventListener("change", e => { this.agreeK = e.target.value; this.drawAgreement(); });
    document.addEventListener("keydown", e => {
      if (document.body.dataset.tab !== "depth" || !this.panels.length || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target instanceof Element ? e.target : document.body;
      if (t.closest("select, textarea, input[type=radio], input[type=text], input[type=range]:not(#dp-depth)")) return;
      if (e.key === " ") { if (t.closest("input, button, summary, a")) return; e.preventDefault(); this.togglePlay(); }
      else if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); this.stepLayer(e.key === "ArrowRight" ? 1 : -1); }
    });
    window.addEventListener("scroll", () => { $("dp-tip").style.display = "none"; }, { passive: true });
    let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { if (document.body.dataset.tab === "depth") this.drawAll(); }, 120); });
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => { if (this.panels.length) { this.prepare(); this.drawAll(); } });
  }
  primerMemory() {
    const d = $("dp-primer"); if (!d) return;
    let pref = null;
    try { pref = localStorage.getItem("tm-dp-primer"); } catch (e) { /* storage unavailable */ }
    // a phone starts with it folded; only the reader's own clicks are remembered (the toggle event of a fold made
    // here would otherwise be stored as a choice and fold it on a wide screen too)
    let auto = false;
    if (pref === "closed" || (pref == null && innerWidth < 700)) { auto = d.open; d.open = false; }
    d.addEventListener("toggle", () => {
      if (auto) { auto = false; return; }
      try { localStorage.setItem("tm-dp-primer", d.open ? "open" : "closed"); } catch (e) { /* ignore */ }
    });
  }
  clamp() {
    const st = this.st, K = this.meta.modes_shown, L = this.meta.layers;
    if (!this.setBy[st.s]) st.s = this.sets[0].key;
    if (!["both", ...this.models.map(m => m.key)].includes(st.m)) st.m = this.models[0].key;
    if (!this.strengths.some(s => s.key === st.t)) st.t = this.strengths[0].key;
    st.x = Math.max(1, Math.min(K, Math.round(+st.x) || 1)); st.y = Math.max(1, Math.min(K, Math.round(+st.y) || 2));
    if (st.x === st.y) st.y = st.x === 1 ? 2 : 1;
    st.l = Math.max(0, Math.min(L - 1, +st.l || 0));
    const ok = this.shownModels().some(m => this.has(m, st.s) && (st.o === "none" || (st.o.startsWith("panel:") ? this.unit(m, st.s).panels.includes(st.o.slice(6)) : this.unit(m, st.s).paths.includes(st.o.slice(5)))));
    if (!ok) st.o = "none";
    if (!["text", "pos", "freq", "cohort"].includes(st.c)) st.c = "text";
    if (!(st.lim in LIMITS)) st.lim = "q";
    if (st.c === "cohort" && !st.o.startsWith("panel:")) st.c = "text";
  }

  // ------------------------------------------------ hash state
  saveHash() {
    const st = this.st, h = new URLSearchParams({ m: st.m, s: st.s, t: st.t, x: st.x, y: st.y, c: st.c, o: st.o, fix: +st.fix, sq: +st.sq, lim: st.lim, l: (+st.l).toFixed(2) });
    this.lastHash = "#depth?" + h.toString();
    if (document.body.dataset.tab === "depth") history.replaceState(history.state, "", this.lastHash);
  }
  async show(query) {
    await this.ready; if (this.failed) return;
    if (query && [...query.keys()].length && location.hash !== this.lastHash) {
      const st = this.st;
      for (const k of ["m", "s", "t", "c", "o", "lim"]) if (query.has(k)) st[k] = query.get(k);
      for (const k of ["x", "y", "l"]) if (query.has(k) && isFinite(+query.get(k))) st[k] = +query.get(k);
      if (query.has("fix")) st.fix = query.get("fix") !== "0";
      if (query.has("sq")) st.sq = query.get("sq") === "1";
      this.clamp(); await this.load(); this.lastHash = location.hash; return;
    }
    if (!this.panels.length) { this.clamp(); await this.load(); this.saveHash(); return; }
    this.drawAll();
  }

  // ------------------------------------------------ loading
  async load() {
    const st = this.st, token = this.loadToken = (this.loadToken || 0) + 1;
    $("dp-status").textContent = "loading…";
    const got = await Promise.all(this.shownModels().map(async model => {
      if (!this.has(model, st.s)) return { model, missing: "not computed yet" };
      try {
        const base = `u/${model}/${st.s}/${st.t}`, rec = await getJSON(`${base}.json`);
        const f = rec.files.set, set = new Coords(await getBin(f.path, f.shape.reduce((a, b) => a * b)), f);
        let over = null, overRows = null, overKind = null;
        if (st.o !== "none") {
          const tag = st.o.replace(":", "-"), of = rec.files[tag];
          if (of) {
            over = new Coords(await getBin(of.path, of.shape.reduce((a, b) => a * b)), of);
            overKind = st.o.split(":")[0];
            if (overKind === "panel") overRows = this.rows.panels[st.o.slice(6)];
            else { const ci = this.ctxKeys.indexOf(st.o.slice(5)); overRows = Array.from({ length: over.M }, (_, i) => [ci, i + 1]); }
          }
        }
        return { model, rec, set, over, overRows, overKind };
      } catch (e) { console.error(e); return { model, missing: `failed to load (${e.message})` }; }
    }));
    if (token !== this.loadToken) return;
    this.panels = got;
    if (!this.tableModel || !got.some(p => p.model === this.tableModel && p.rec)) this.tableModel = got.find(p => p.rec)?.model || null;
    if (this.extK == null || this.extK > this.meta.modes_shown) this.extK = st.x;
    this.prepare(); this.renderPanels(); this.syncControls(); this.drawAll();
    $("dp-status").textContent = this.statusText();
    this.root.dataset.state = "ready";
  }
  statusText() {
    const m = this.meta, packed = Object.values(m.units).filter(u => u.status === "packed").length, total = Object.keys(m.units).length;
    return packed < total ? `This build carries ${packed} of ${total} embeddings (model × construction set); the others are not computed yet.` : "";
  }

  // per panel: the two plotted columns, scales, rows, colours
  prepare() {
    const st = this.st, rows = this.rows.sets[st.s];
    this.setRows = rows;
    // how often each id occurs in the construction set (for the frequency colouring and the hover card)
    this.idCount = new Map();
    for (const [ci, p] of rows) { const id = this.idAt(ci, p); this.idCount.set(id, (this.idCount.get(id) || 0) + 1); }
    this.maxCount = Math.max(1, ...this.idCount.values());
    this.pal = this.palette();
    const coloured = this.colouredContexts();
    this.slotOf = coloured.slot; this.textCtx = coloured.list;
    for (const p of this.panels) {
      if (!p.rec) continue;
      const kx = st.x - 1, ky = st.y - 1;
      p.sx = p.set.column(kx); p.sy = p.set.column(ky);
      p.ox = p.over ? p.over.column(kx) : null; p.oy = p.over ? p.over.column(ky) : null;
      p.limX = axisLimits(p.sx, p.ox, p.set.L); p.limY = axisLimits(p.sy, p.oy, p.set.L);
      if (p.over) {
        const keys = new Set(rows.map(([ci, pos]) => ci * 4096 + pos));
        p.overIn = p.overRows.map(([ci, pos]) => keys.has(ci * 4096 + pos));
        p.overCohort = p.overKind === "panel" ? this.rows.cohorts[st.s][st.o.slice(6)] : null;
      }
      p.setCols = rows.map(([ci, pos]) => this.colour(ci, pos, null));
      p.overCols = p.over ? p.overRows.map(([ci, pos], i) => this.colour(ci, pos, p.overCohort ? p.overCohort[i] : null)) : null;
    }
  }
  palette() {
    const cat = [1, 2, 3, 4, 5, 6, 7, 8].map(i => hexRgb(cssv(`--dp-cat-${i}`) || "#888888"));
    const ramp = [2, 3, 4, 5, 6, 7].map(i => hexRgb(cssv(`--seq-${i}`) || "#3987e5"));
    return { cat, ramp, grey: hexRgb(cssv("--mute-2") || "#8d8c86"), fg: cssv("--fg"), mute: cssv("--mute"), line: cssv("--line"), surface: cssv("--surface"), accent: cssv("--accent") };
  }
  rampAt(t) {
    const r = this.pal.ramp, x = Math.max(0, Math.min(1, t)) * (r.length - 1), i = Math.min(r.length - 2, Math.floor(x)), a = x - i;
    return r[i].map((c, j) => Math.round(c + a * (r[i + 1][j] - c)));
  }
  // the texts whose tokens are coloured by "text": the contexts of the points in focus (overlay if shown, else the set)
  colouredContexts() {
    const st = this.st, src = st.o.startsWith("panel:") ? this.rows.panels[st.o.slice(6)] : st.o.startsWith("path:") ? [[this.ctxKeys.indexOf(st.o.slice(5)), 1]] : this.rows.sets[st.s];
    const list = [...new Set(src.map(r => r[0]))].sort((a, b) => a - b), slot = new Map();
    if (this.hl.length) this.hl.slice(-8).forEach((ci, j) => slot.set(ci, j));
    else list.slice(0, 8).forEach((ci, j) => slot.set(ci, j));
    return { list, slot };
  }
  colour(ci, pos, cohort) {
    const st = this.st;
    if (st.c === "pos") return this.rampAt((pos - 1) / 2046);
    if (st.c === "freq") { const n = this.idCount.get(this.idAt(ci, pos)) || 0; return n ? this.rampAt(Math.log(n) / Math.log(Math.max(2, this.maxCount))) : this.pal.grey; }
    if (st.c === "cohort") return cohort == null ? this.pal.grey : this.pal.cat[cohort];
    const s = this.slotOf.get(ci); return s == null ? this.pal.grey : this.pal.cat[s];
  }
  isFaded(ci) { return this.st.c === "text" && this.hl.length > 0 && !this.hl.includes(ci); }

  // ------------------------------------------------ plots
  renderPanels() {
    const wrap = $("dp-plots"); wrap.innerHTML = ""; wrap.dataset.n = this.panels.length;
    $("dp-unmatched").hidden = this.panels.length < 2;
    for (const p of this.panels) {
      const fig = document.createElement("figure"), cap = document.createElement("figcaption"), cv = document.createElement("canvas");
      fig.className = "dp-panel"; fig.append(cap, cv); wrap.append(fig); p.fig = fig; p.cap = cap; p.cv = cv;
      if (!p.rec) { cv.hidden = true; cap.innerHTML = `<b>${esc(this.modelName[p.model])}</b> · ${esc(this.st.s)}: ${esc(p.missing)}`; fig.classList.add("dp-missing"); continue; }
      cv.tabIndex = 0; cv.setAttribute("aria-label", `${this.modelName[p.model]}: scatter of the tokens of ${this.st.s} in modes ${this.st.x} and ${this.st.y}`);
      this.bindPlot(p);
    }
    this.renderStrips();
  }
  fit(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); return { g, w, h };
  }
  frameAt(p, s = this.st.l) {
    const L = p.set.L, l0 = Math.max(0, Math.min(L - 2, Math.floor(s))), a = Math.max(0, Math.min(1, s - l0));
    return { l0, a, ...this.limitsAt(p, l0, a) };
  }
  // the axis limits at (fractional) layer l0 + a, as chosen under Axes
  limitsAt(p, l0, a) {
    const st = this.st, l1 = Math.min(p.set.L - 1, l0 + 1), pick = A => st.fix ? A.all : (1 - a) * A.L[l0] + a * A.L[l1];
    if (st.lim === "one") { const R = Math.max(pick(p.limX.max), pick(p.limY.max)); return { Rx: R, Ry: R, equal: true }; }
    const k = st.lim === "all" ? "max" : "q";
    return { Rx: pick(p.limX[k]), Ry: pick(p.limY[k]), equal: false };
  }
  drawAll() {
    if (document.body.dataset.tab !== "depth" || !this.panels.length) return;
    this.legend(); this.draw(); this.drawStrips(); this.drawOcc(); this.drawModes(); this.drawExtremes(); this.drawAgreement();
  }
  draw() {
    if (document.body.dataset.tab !== "depth") return;
    for (const p of this.panels) if (p.rec) this.drawPanel(p);
    const s = this.st.l, lc = Math.round(s), near = Math.abs(s - lc) < 0.02, L = this.meta.layers;
    $("dp-layerhead").textContent = near ? `Layer ${lc}` : `Layer ${Math.floor(s)} → ${Math.floor(s) + 1}`;
    $("dp-layersub").textContent = near ? (lc === 0 ? "the input vectors" : `output of block ${lc} of ${L - 1}`) : "drawn in between, for the eye only";
    $("dp-depth").value = s; $("dp-depth").setAttribute("aria-valuetext", $("dp-layerhead").textContent);
    document.querySelectorAll(".dp-strip figure").forEach(f => f.classList.toggle("on", +f.dataset.l === lc));
    this.drawOccCursor();
  }
  // screen position of a coordinate pair, and whether it lies beyond the axis limits (then it is put on the frame's
  // edge); with the square-root scale each coordinate is drawn at sign(v) sqrt(|v| R)
  // pad: the margin around the drawn points, one number or {t, b, l, r}
  projector(w, h, { Rx, Ry, equal }, pad = 22) {
    const m = typeof pad === "number" ? { t: pad, b: pad, l: pad, r: pad } : pad;
    let Sx = (w - m.l - m.r) / 2 / Rx, Sy = (h - m.t - m.b) / 2 / Ry;
    if (equal) Sx = Sy = Math.min(Sx, Sy);
    const cx = m.l + (w - m.l - m.r) / 2, cy = m.t + (h - m.t - m.b) / 2, sq = this.st.sq;
    const f = (v, R) => { const c = Math.max(-R, Math.min(R, v)); return sq ? Math.sign(c) * Math.sqrt(Math.abs(c) * R) : c; };
    return (x, y) => [cx + Sx * f(x, Rx), cy - Sy * f(y, Ry), Math.abs(x) > Rx * (1 + 1e-9) || Math.abs(y) > Ry * (1 + 1e-9)];
  }
  drawPanel(p) {
    const st = this.st, { g, w, h } = this.fit(p.cv), P = this.pal;
    g.fillStyle = P.surface; g.fillRect(0, 0, w, h);
    // the frame sits inside margins that hold the text (top: axis names; bottom: limits and the count beyond the frame),
    // so no label is ever drawn over a token; tokens are drawn 8 px inside the frame
    const FT = 20, FB = 36, FS = 12, fr = this.frameAt(p), { l0, a } = fr, M = p.set.M, L = p.set.L, l1 = Math.min(L - 1, l0 + 1);
    const proj = this.projector(w, h, fr, { t: FT + 8, b: FB + 8, l: FS + 8, r: FS + 8 });
    g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); const [x0, y0] = proj(0, 0);
    g.moveTo(FS, y0); g.lineTo(w - FS, y0); g.moveTo(x0, FT); g.lineTo(x0, h - FB); g.stroke();
    g.strokeRect(FS + 0.5, FT + 0.5, w - 2 * FS - 1, h - FT - FB - 1);
    g.fillStyle = P.mute; g.font = "11px Inter, system-ui, sans-serif";
    g.textAlign = "left"; g.fillText(`↑ mode ${st.y}`, x0 + 4, FT - 6);
    g.textAlign = "right"; g.fillText(`mode ${st.x} →`, w - FS, FT - 6);
    const lerp = (col, i) => { const v0 = col.v[l0 * col.v.length / L + i], v1 = col.v[l1 * col.v.length / L + i]; return (1 - a) * v0 + a * v1; };
    const r = Math.max(1.6, Math.min(3, w / 300)), over = !!p.over;
    p.scr = new Float32Array(M * 2); p.oscr = over ? new Float32Array(p.over.M * 2) : null;
    p.cl = new Uint8Array(M); p.ocl = over ? new Uint8Array(p.over.M) : null;
    const dot = (x, y, rad, clipped, col, alpha) => {
      g.beginPath(); g.arc(x, y, rad, 0, 7);
      if (clipped) { g.strokeStyle = rgba(col, Math.max(alpha, 0.5)); g.lineWidth = 1.2; g.stroke(); } else { g.fillStyle = rgba(col, alpha); g.fill(); }
    };
    const order = this.order(M);
    // construction tokens: coloured, or grey under an overlay
    const faded = [], front = [];
    for (const i of order) { (this.isFaded(this.setRows[i][0]) ? faded : front).push(i); }
    for (const pass of [faded, front]) for (const i of pass) {
      const [sx, sy, c] = proj(lerp(p.sx, i), lerp(p.sy, i)); p.scr[2 * i] = sx; p.scr[2 * i + 1] = sy; p.cl[i] = c;
      if (over) { dot(sx, sy, r * 0.8, c, P.grey, 0.28); continue; }
      dot(sx, sy, c ? r + 0.6 : r, c, p.setCols[i], pass === faded ? 0.1 : 0.82);
    }
    if (over) {
      const N = p.over.M, ok = new Uint8Array(N);
      for (let i = 0; i < N; i++) { const x = lerp(p.ox, i), y = lerp(p.oy, i); if (isFinite(x) && isFinite(y)) { const [sx, sy, c] = proj(x, y); p.oscr[2 * i] = sx; p.oscr[2 * i + 1] = sy; p.ocl[i] = c; ok[i] = 1; } else { p.oscr[2 * i] = p.oscr[2 * i + 1] = NaN; } }
      p.overOk = ok;
      if (p.overKind === "path") {
        g.lineWidth = 1;
        for (let i = 1; i < N; i++) { if (!ok[i] || !ok[i - 1]) continue; g.strokeStyle = rgba(p.overCols[i], 0.5); g.beginPath(); g.moveTo(p.oscr[2 * i - 2], p.oscr[2 * i - 1]); g.lineTo(p.oscr[2 * i], p.oscr[2 * i + 1]); g.stroke(); }
      }
      for (const i of this.order(N)) {
        if (!ok[i]) continue;
        const f = this.isFaded(p.overRows[i][0]);
        dot(p.oscr[2 * i], p.oscr[2 * i + 1], p.ocl[i] ? r + 0.6 : r, p.ocl[i], p.overCols[i], f ? 0.1 : 0.9);
        if (p.overIn[i] && !f) { g.strokeStyle = P.fg; g.lineWidth = 0.9; g.beginPath(); g.arc(p.oscr[2 * i], p.oscr[2 * i + 1], r + 1.6, 0, 7); g.stroke(); }
      }
    }
    // what the frame shows, and how many tokens lie beyond it
    const nClip = p.cl.reduce((t, v) => t + v, 0), nOclip = p.ocl ? p.ocl.reduce((t, v) => t + v, 0) : 0;
    const limTxt = `${fr.equal ? `axes ±${num(fr.Rx, 2)}` : `x ±${num(fr.Rx, 2)}, y ±${num(fr.Ry, 2)}`}${st.lim === "q" ? ` (${100 * QUANT}% of coordinates${st.fix ? ", over all layers" : ", at this layer"})` : st.fix ? "" : " (this layer)"}${st.sq ? " · square-root scale" : ""}`;
    g.font = "11px Inter, system-ui, sans-serif"; g.textAlign = "left"; g.fillStyle = P.mute;
    const narrow = w < 480;
    g.fillText(narrow ? limTxt.replace(" of coordinates", "").replace(", over all layers", "").replace(", at this layer", " per layer") : limTxt, FS, h - FB + 14);
    if (nClip || nOclip) {
      g.fillStyle = P.fg;
      g.fillText(narrow ? `${int(nClip)}${over ? ` + ${int(nOclip)} placed` : ""} beyond the frame: open circles on its edge`
        : `${int(nClip)} of ${int(M)} tokens${over ? ` and ${int(nOclip)} placed` : ""} beyond the frame, drawn on its edge as open circles`, FS, h - 6);
    }
    p.clipText = nClip || nOclip ? `${int(nClip)}${over ? ` + ${int(nOclip)} placed` : ""} beyond the frame` : "";
    // the hovered token, and the same token in the other model's plot
    if (this.hover) {
      const hv = this.hover, arr = hv.kind === "set" ? p.scr : p.oscr;
      if (arr && isFinite(arr[2 * hv.i])) {
        g.strokeStyle = hv.panel === p ? P.fg : P.accent; g.lineWidth = 2; g.beginPath(); g.arc(arr[2 * hv.i], arr[2 * hv.i + 1], r + 4, 0, 7); g.stroke();
      }
    }
    // caption
    const tstar = p.rec.tstar, t = p.rec.t, s = this.strengths.find(x => x.key === st.t);
    const flags = [st.x, st.y].map(k => p.rec.modes[k - 1]).filter(md => md.band || md.withheld).map(md => `<span class="dp-flagtxt" title="${esc(this.flagText(md, false))}">mode ${md.k} ${esc(this.flagText(md, true))}</span>`);
    const missingOver = st.o !== "none" && !p.over ? ` · <span class="dp-flagtxt">${esc(st.o)} not available for this model yet</span>` : "";
    const withheldOver = p.over && [st.x, st.y].some(k => p.over.withheld.has(k)) ? ` · <span class="dp-flagtxt">placed tokens have no coordinate on a withheld mode</span>` : "";
    const cap = `<b>${esc(this.modelName[p.model])}</b> · ${esc(st.s)} · ${esc(s.label)}${st.t === "tstar" ? ` = ${num(tstar)}` : ` = ${num(t)} (t* = ${num(tstar)})`}${flags.length ? ` · ${flags.join(" · ")}` : ""}${missingOver}${withheldOver}`;
    if (p.capHtml !== cap) { p.cap.innerHTML = cap; p.capHtml = cap; }
  }
  // a fixed scrambled drawing order, so no text is always drawn on top
  order(n) {
    this._ord = this._ord || new Map();
    if (!this._ord.has(n)) this._ord.set(n, [...Array(n).keys()].sort((x, y) => ((x * 7919) % n) - ((y * 7919) % n)));
    return this._ord.get(n);
  }
  bindPlot(p) {
    const cv = p.cv;
    const at = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener("pointermove", e => { if (e.pointerType === "mouse") this.hoverAt(p, ...at(e), e); });
    cv.addEventListener("pointerup", e => { if (e.pointerType !== "mouse") this.hoverAt(p, ...at(e), e); });
    cv.addEventListener("pointerleave", e => { if (e.pointerType === "mouse") { this.hover = null; $("dp-tip").style.display = "none"; this.draw(); } });
  }
  hoverAt(p, x, y, e) {
    let best = null, bd = 14 * 14;
    const scan = (arr, kind, rows) => { if (!arr) return; for (let i = 0; i < arr.length / 2; i++) { if (this.isFaded(rows[i][0])) continue; const q = (arr[2 * i] - x) ** 2 + (arr[2 * i + 1] - y) ** 2; if (q < bd) { bd = q; best = { kind, i }; } } };
    if (p.over) scan(p.oscr, "over", p.overRows);      // placed tokens first; a construction token wins only when nearer
    scan(p.scr, "set", this.setRows);
    const prev = this.hover;
    this.hover = best ? { ...best, panel: p } : null;
    if (!best) { $("dp-tip").style.display = "none"; if (prev) this.draw(); return; }
    if (!prev || prev.i !== best.i || prev.kind !== best.kind || prev.panel !== p) this.draw();
    this.showTip(p, best, e);
  }
  // the hover card: the token with CONTEXT tokens each side, its text, position, id count, and placement group
  tokenCard(ci, pos, extra) {
    const c = this.ctx[ci], ids = c.ids, n = this.meta.context, a = Math.max(1, pos - n), b = Math.min(ids ? ids.length : pos, pos + n);
    let ctx = "";
    if (ids) for (let q = a; q <= b; q++) { const s = esc(tokText(this.str(ids[q - 1]))); ctx += q === pos ? `<mark>${s}</mark>` : `<span>${s}</span>`; }
    const id = this.idAt(ci, pos), cnt = this.idCount.get(id) || 0;
    const src = c.kind === "shuffled" && c.perm ? ` · originally position ${int(c.perm[pos - 1])} of c${String(c.source_context).padStart(2, "0")}` : "";
    return `<div class="dp-tipctx">${a > 1 ? "…" : ""}${ctx}${ids && b < ids.length ? "…" : ""}</div>`
      + `<div class="dp-tipmeta"><b>${esc(this.label(c.key))}</b> · position ${int(pos)} of ${int(ids ? ids.length : 0)}${src}</div>`
      + `<div class="dp-tipmeta">token id ${id ?? "?"} · occurs ${int(cnt)} time${cnt === 1 ? "" : "s"} in ${esc(this.st.s)}${extra ? ` · ${extra}` : ""}</div>`;
  }
  showTip(p, best, e) {
    const st = this.st, rows = best.kind === "set" ? this.setRows : p.overRows, [ci, pos] = rows[best.i];
    let extra = best.kind === "set" ? "in the construction set" : "";
    const clipped = (best.kind === "set" ? p.cl : p.ocl)?.[best.i];
    if (best.kind === "over") {
      if (p.overCohort) extra = ["in the set", "placed: new position of a text in the set", "placed: a text the set never saw"][p.overCohort[best.i]];
      else extra = p.overIn[best.i] ? "also in the set" : "placed";
    }
    const coords = this.panels.filter(q => q.rec).map(q => {
      const col = best.kind === "set" ? [q.sx, q.sy] : [q.ox, q.oy];
      if (!col[0]) return "";
      const { l0, a } = this.frameAt(q), L = q.set.L, M = col[0].v.length / L, l1 = Math.min(L - 1, l0 + 1);
      const v = c => (1 - a) * c.v[l0 * M + best.i] + a * c.v[l1 * M + best.i];
      return `${this.panels.length > 1 ? `${esc(this.modelName[q.model])}: ` : ""}mode ${st.x} = ${num(v(col[0]))}, mode ${st.y} = ${num(v(col[1]))}`;
    }).filter(Boolean);
    const tip = $("dp-tip");
    tip.innerHTML = this.tokenCard(ci, pos, extra) + `<div class="dp-tipmeta dp-tipco">${clipped ? "<b>Beyond the axis limits</b>: drawn on the frame's edge; its coordinates are<br>" : ""}${coords.join("<br>")}${Math.abs(st.l - Math.round(st.l)) > 0.02 ? " (between layers)" : ` (layer ${Math.round(st.l)})`}</div>`;
    tip.style.display = "block";
    const host = tip.offsetParent?.getBoundingClientRect() || { left: 0, top: 0, width: innerWidth };
    const tw = Math.min(340, host.width - 8);
    tip.style.left = Math.max(4, Math.min(host.width - tw - 4, e.clientX - host.left + 14)) + "px"; tip.style.top = (e.clientY - host.top + 16) + "px";
  }

  // ------------------------------------------------ key above the plot
  legend() {
    const st = this.st, el = $("dp-legend"), P = this.pal, what = $("dp-what");
    const over = st.o === "none" ? "" : st.o.startsWith("panel:") ? ` Overlaid: the <b>${esc(st.o.slice(6))}</b> panel, placed into this embedding.` : ` Overlaid: the path of <b>${esc(this.label(st.o.slice(5)))}</b>, every token in reading order.`;
    const limits = st.lim === "q" ? ` Each axis is scaled to hold ${100 * QUANT}% of its mode's coordinates${st.fix ? " over all layers" : " at each layer"}; tokens beyond are drawn on the frame's edge as open circles and counted on the plot (choose <i>fit every token</i> under Axis limits for the plain view).`
      : st.lim === "all" ? " Each axis is scaled to its own mode's largest value." : " One scale for both axes, set by the largest value.";
    what.innerHTML = `Each dot is one token of <b>${esc(st.s)}</b> (${int(this.setBy[st.s].n)} tokens), at its coordinates in modes <b>${st.x}</b> (across) and <b>${st.y}</b> (up).${limits}${over}`;
    const items = [];
    const sw = c => `<i style="background:${rgba(c, 1)}"></i>`;
    if (st.o !== "none") items.push(`<span class="dp-leg static"><i style="background:${rgba(P.grey, 0.5)}"></i>the set's own tokens</span>`);
    if (st.c === "text") {
      for (const ci of this.textCtx) {
        const s = this.slotOf.get(ci), on = this.hl.includes(ci);
        items.push(`<button type="button" class="dp-leg${this.hl.length && !on ? " off" : ""}" data-ci="${ci}" aria-pressed="${on}" title="${esc(this.label(this.ctxKeys[ci]))}: click to pick out">${sw(s == null ? P.grey : P.cat[s])}${esc(this.ctxKeys[ci])}</button>`);
      }
      if (this.hl.length) items.push(`<button type="button" class="dp-leg dp-clear" data-clear="1">show all</button>`);
      const grey = this.textCtx.filter(ci => !this.slotOf.has(ci)).length;
      if (grey) items.push(`<span class="dp-legnote">Grey: ${grey} more text${grey > 1 ? "s" : ""} (8 colours at a time). Click names to colour and pick them out.</span>`);
    } else if (st.c === "cohort") {
      ["in the set", "new position of a text in the set", "a text the set never saw"].forEach((t, j) => items.push(`<span class="dp-leg static">${sw(P.cat[j])}${t}</span>`));
    } else {
      const ramp = `<i class="dp-ramp" style="background:linear-gradient(90deg,${P.ramp.map(c => rgba(c, 1)).join(",")})"></i>`;
      items.push(st.c === "pos" ? `<span class="dp-leg static">position 1 ${ramp} ${int(2047)}</span>`
        : `<span class="dp-leg static">occurs once ${ramp} ${int(this.maxCount)} times</span>${st.o !== "none" ? `<span class="dp-leg static">${sw(P.grey)}id not in the set</span>` : ""}`);
    }
    if (st.o.startsWith("path:")) items.push(`<span class="dp-leg static"><i class="dp-ringkey"></i>also in the set</span>`);
    if (st.o.startsWith("panel:")) items.push(`<span class="dp-leg static"><i class="dp-ringkey"></i>in the set (ringed)</span>`);
    el.innerHTML = items.join("");
    el.querySelectorAll("button.dp-leg").forEach(b => b.addEventListener("click", () => {
      if (b.dataset.clear) this.hl = [];
      else { const ci = +b.dataset.ci, j = this.hl.indexOf(ci); if (j >= 0) this.hl.splice(j, 1); else { this.hl.push(ci); if (this.hl.length > 8) this.hl.shift(); } }
      this.prepare(); this.drawAll();
    }));
  }

  // ------------------------------------------------ every layer at a glance
  renderStrips() {
    const box = $("dp-strips"); box.innerHTML = "";
    for (const p of this.panels.filter(q => q.rec)) {
      const row = document.createElement("div"); row.className = "dp-striprow";
      if (this.panels.length > 1) { const h = document.createElement("div"); h.className = "dp-striplabel"; h.textContent = this.modelName[p.model]; row.append(h); }
      const strip = document.createElement("div"); strip.className = "dp-strip"; strip.style.setProperty("--n", p.set.L);
      for (let l = 0; l < p.set.L; l++) {
        const f = document.createElement("figure"), cv = document.createElement("canvas"), cap = document.createElement("figcaption");
        f.dataset.l = l; cap.textContent = `${l}`; f.title = `layer ${l}`; f.append(cv, cap); strip.append(f);
        f.addEventListener("click", () => { this.stop(); this.st.l = l; this.draw(); this.saveHash(); });
      }
      row.append(strip); box.append(row); p.strip = strip;
    }
  }
  drawStrips() {
    for (const p of this.panels.filter(q => q.rec && q.strip)) {
      [...p.strip.children].forEach((f, l) => {
        const { g, w, h } = this.fit(f.querySelector("canvas")), P = this.pal;
        g.fillStyle = P.surface; g.fillRect(0, 0, w, h);
        const pj = this.projector(w, h, this.limitsAt(p, l, 0), 3), M = p.set.M;
        for (let i = 0; i < M; i++) {
          const x = p.sx.v[l * M + i], y = p.sy.v[l * M + i];
          const [sx, sy] = pj(x, y);
          g.fillStyle = p.over ? rgba(P.grey, 0.3) : rgba(p.setCols[i], this.isFaded(this.setRows[i][0]) ? 0.08 : 0.75); g.fillRect(sx - 0.8, sy - 0.8, 1.6, 1.6);
        }
        if (p.over) { const N = p.over.M; for (let i = 0; i < N; i++) { const x = p.ox.v[l * N + i], y = p.oy.v[l * N + i]; if (!isFinite(x) || !isFinite(y)) continue; const [sx, sy] = pj(x, y); g.fillStyle = rgba(p.overCols[i], 0.8); g.fillRect(sx - 0.9, sy - 0.9, 1.8, 1.8); } }
      });
    }
  }

  // ------------------------------------------------ depth weight bars over the slider
  drawOcc() {
    const cv = $("dp-occ"), { g, w, h } = this.fit(cv), P = this.pal, L = this.meta.layers, ps = this.panels.filter(p => p.rec), st = this.st;
    g.fillStyle = P.surface; g.fillRect(0, 0, w, h);
    if (!ps.length) return;
    const pad = 8, X = l => pad + (w - 2 * pad) * l / (L - 1), bw = Math.max(3, Math.min(14, (w - 2 * pad) / L - 4)), each = bw / ps.length;
    const occ = ps.map(p => Array.from({ length: L }, (_, l) => p.rec.modes[st.x - 1].depth_weight[l] + p.rec.modes[st.y - 1].depth_weight[l]));
    const mx = Math.max(1e-9, ...occ.flat());
    g.font = "10px Inter, system-ui, sans-serif"; g.fillStyle = P.mute;
    occ.forEach((o, j) => { g.globalAlpha = j ? 0.75 : 0.4; g.fillStyle = P.fg; for (let l = 0; l < L; l++) { const bh = (h - 13) * o[l] / mx; g.fillRect(X(l) - bw / 2 + j * each, h - 12 - bh, each - (ps.length > 1 ? 1 : 0), bh); } });
    g.globalAlpha = 1; g.fillStyle = P.mute;
    for (let l = 0; l < L; l++) if (l % (w < 420 ? 4 : 2) === 0 || l === L - 1) { g.textAlign = "center"; g.fillText(String(l), X(l), h - 1); }
    $("dp-occ-cap").textContent = `Bars: share of modes ${st.x} and ${st.y} at each layer (each mode's shares add up to 1)` + (ps.length > 1 ? ` · ${ps.map((p, j) => `${this.modelName[p.model]} ${j ? "dark" : "light"}`).join(", ")}` : "");
    this.occBase = g.getImageData(0, 0, cv.width, cv.height); this.drawOccCursor();
  }
  drawOccCursor() {
    if (!this.occBase) return;
    const cv = $("dp-occ"), g = cv.getContext("2d"); g.putImageData(this.occBase, 0, 0);
    const dpr = window.devicePixelRatio || 1; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = cv.clientWidth, h = cv.clientHeight, pad = 8, x = pad + (w - 2 * pad) * this.st.l / (this.meta.layers - 1);
    g.strokeStyle = this.pal.accent; g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h - 12); g.stroke();
  }
  togglePlay() {
    if (this.playing) return this.stop();
    const L = this.meta.layers; if (this.st.l >= L - 1) this.st.l = 0;
    this.playing = true; this.lastT = 0; $("dp-play").textContent = "❚❚ Pause"; $("dp-play").setAttribute("aria-label", "Pause");
    requestAnimationFrame(t => this.tick(t));
  }
  tick(t) {
    if (!this.playing) return;
    const L = this.meta.layers;
    if (this.lastT) this.st.l = Math.min(L - 1, this.st.l + (t - this.lastT) / 1000 * LAYERS_PER_SECOND);
    this.lastT = t; this.draw();
    if (this.st.l >= L - 1) { this.stop(); return; }
    requestAnimationFrame(x => this.tick(x));
  }
  stop() { if (!this.playing) return; this.playing = false; $("dp-play").textContent = "▶ Play"; $("dp-play").setAttribute("aria-label", "Play through the layers"); this.saveHash(); }
  stepLayer(d) { this.stop(); const s = this.st.l, L = this.meta.layers; this.st.l = Math.max(0, Math.min(L - 1, d > 0 ? Math.floor(s + 1e-6) + 1 : Math.ceil(s - 1e-6) - 1)); this.draw(); this.saveHash(); }

  // ------------------------------------------------ mode table and extreme tokens
  drawModes() {
    const ps = this.panels.filter(p => p.rec);
    this.seg($("dp-modes-model"), "dp-modes-model", ps.map(p => ({ value: p.model, label: this.modelName[p.model] })), this.tableModel);
    $("dp-modes-model").hidden = ps.length < 2;
    const p = ps.find(q => q.model === this.tableModel) || ps[0], tb = $("dp-modes");
    if (!p) { tb.innerHTML = ""; $("dp-modes-hint").textContent = "No embedding loaded."; return; }
    const st = this.st, L = this.meta.layers, rec = p.rec, mx = Math.max(...rec.modes.flatMap(m => m.depth_weight));
    const single = this.setBy[st.s].contexts.length === 1;
    $("dp-modes-hint").innerHTML = `${esc(this.modelName[p.model])}, ${esc(st.s)}, ${esc(this.strengths.find(s => s.key === st.t).label)}. Each row is one mode; the cells are its <b>depth weight</b> at layers 0 to ${L - 1} (darker = more of the mode there; an even spread would be ${num(1 / L, 2)} per layer). <b>Token support</b>: the share of tokens the mode effectively uses (1 = spread evenly over all of them). <b>Text share</b>: the part of the mode that is a difference between whole texts${single ? " (not applicable to a one-text set)" : ""}. Click a row to list its extreme tokens below; x and y put it on an axis.`;
    const head = `<thead><tr><th>mode</th><th class="dp-num">eigenvalue</th><th>depth weight, layer 0 → ${L - 1}</th><th class="dp-num">token support</th><th class="dp-num">text share</th><th>flags</th><th><span class="dp-vh">axis</span></th></tr></thead>`;
    const body = rec.modes.map(md => {
      const cells = md.depth_weight.map((v, l) => `<i style="background:${rgba(this.rampAt(v / mx), 1)}" title="layer ${l}: ${num(v)}${md.weak_slice[l] ? " (almost empty)" : ""}"></i>`).join("");
      const cls = [md.k === st.x || md.k === st.y ? "on" : "", md.k === this.extK ? "sel" : ""].join(" ");
      const flags = [md.band ? `<span class="dp-flag tie" title="${esc(this.flagText(md, false))}">≈ ${md.band[0]}–${md.band[1]}</span>` : "", md.withheld ? `<span class="dp-flag wh" title="withheld from placement">× withheld</span>` : ""].join(" ");
      return `<tr class="${cls}" data-k="${md.k}"><td><b>${md.k}</b>${md.k === st.x ? ' <span class="dp-axtag">x</span>' : ""}${md.k === st.y ? ' <span class="dp-axtag">y</span>' : ""}</td><td class="dp-num">${md.eigenvalue.toFixed(3)}</td><td><span class="dp-heat" style="--n:${L}">${cells}</span></td><td class="dp-num">${md.token_support.toFixed(3)}</td><td class="dp-num">${md.document_share == null ? "–" : md.document_share.toFixed(2)}</td><td>${flags}</td><td class="dp-axbtns"><button type="button" data-ax="x" ${md.k === st.x ? "disabled" : ""}>x</button><button type="button" data-ax="y" ${md.k === st.y ? "disabled" : ""}>y</button></td></tr>`;
    }).join("");
    tb.innerHTML = head + `<tbody>${body}</tbody>`;
    tb.querySelectorAll("tbody tr").forEach(tr => tr.addEventListener("click", e => {
      const k = +tr.dataset.k, ax = e.target.closest("button")?.dataset.ax;
      if (ax) { this.st[ax] = k; if (this.st.x === this.st.y) this.st[ax === "x" ? "y" : "x"] = k === 1 ? 2 : 1; this.clamp(); this.saveHash(); this.prepare(); this.syncControls(); this.drawAll(); return; }
      this.extK = k; this.drawModes(); this.drawExtremes();
    }));
  }
  drawExtremes() {
    const ps = this.panels.filter(p => p.rec), p = ps.find(q => q.model === this.tableModel) || ps[0], box = $("dp-ext");
    if (!p) { box.innerHTML = ""; return; }
    const md = p.rec.modes[(this.extK || 1) - 1], rows = this.setRows;
    $("dp-ext-head").textContent = `Tokens at the extremes of mode ${md.k} · ${this.modelName[p.model]}, ${this.st.s}`;
    $("dp-ext-hint").innerHTML = `The ${md.extremes["0"].low.length} construction tokens with the lowest and the highest value of mode ${md.k} at layers ${Object.keys(md.extremes).join(", ")}. Which end is "low" is arbitrary (a mode's sign is). Hover a token for its full context.`
      + (md.band ? ` <span class="dp-flagtxt">Mode ${md.k} is in a near-tied band (modes ${md.band[0]}–${md.band[1]}): these lists are not individually meaningful, only the band is.</span>` : "");
    const chip = i => {
      const [ci, pos] = rows[i], ids = this.ctx[ci].ids, s = q => (q >= 1 && q <= ids.length ? esc(tokText(this.str(ids[q - 1]))) : "");
      return `<li data-i="${i}" tabindex="0"><span class="dp-cx">${s(pos - 2)}${s(pos - 1)}</span><mark>${s(pos)}</mark><span class="dp-cx">${s(pos + 1)}${s(pos + 2)}</span><span class="dp-src">${esc(this.ctxKeys[ci])}</span></li>`;
    };
    box.innerHTML = Object.entries(md.extremes).map(([l, e]) => `<section class="dp-extcol"><h4>Layer ${l}</h4><div class="dp-extend"><span class="dp-hint">low end</span><ol>${e.low.map(chip).join("")}</ol></div><div class="dp-extend"><span class="dp-hint">high end</span><ol>${e.high.map(chip).join("")}</ol></div></section>`).join("");
    box.querySelectorAll("li").forEach(li => {
      const show = e => { const [ci, pos] = rows[+li.dataset.i], tip = $("dp-tip"); tip.innerHTML = this.tokenCard(ci, pos, "in the construction set"); tip.style.display = "block"; const host = tip.offsetParent.getBoundingClientRect(), r = li.getBoundingClientRect(); tip.style.left = Math.max(4, Math.min(host.width - 344, r.left - host.left)) + "px"; tip.style.top = (r.bottom - host.top + 6) + "px"; };
      li.addEventListener("pointerenter", show); li.addEventListener("focus", show);
      li.addEventListener("pointerleave", () => { $("dp-tip").style.display = "none"; }); li.addEventListener("blur", () => { $("dp-tip").style.display = "none"; });
    });
  }

  // ------------------------------------------------ agreement charts (cross.json, permutations.json, gaussian.json)
  async drawAgreement() {
    const box = $("dp-agree"), st = this.st, k = this.agreeK;
    this.seg($("dp-agree-k"), "dp-agree-k", ["8", "16", "64"].map(x => ({ value: x, label: `k = ${x}` })), k);
    let agg;
    try { agg = await getJSON("aggregates.json"); } catch (e) { agg = {}; }
    const s = this.strengths.find(x => x.key === st.t).label, L = this.meta.layers, layers = [...Array(L).keys()];
    const charts = [];
    // 1. between models, the set's own tokens, layer by layer, beside the two permutation references
    {
      const title = `Between the two models, on the tokens of ${st.s} (${s}), layer by layer`;
      const why = "needs permutations.json (or cross.json), written when the run's last stage finishes";
      const perm = agg.permutations?.records.find(r => r.set === st.s && r.panel === "construction" && r.cohort === "in_set" && r.strength === st.t);
      const cross = agg.cross?.records.find(r => r.kind === "models" && r.set === st.s && r.strength === st.t && r.panel === "construction");
      const sc = perm?.scores[k] ?? (cross ? { matched: cross.scores[k] } : null);
      if (!sc) charts.push(this.emptyChart(title, agg.permutations || agg.cross ? `no record for ${st.s} at ${s}` : why));
      else if (sc.unavailable || sc.matched?.unavailable) charts.push(this.emptyChart(title, `unavailable at k = ${k}: a mode in the first ${k} is withheld`));
      else {
        const series = [{ name: "same tokens (matched)", values: [...sc.matched.layers, sc.matched.stacked], ranks: sc.matched.ranks, slot: 1, width: 2 }];
        for (const [key, name, slot] of [["within_id_and_cohort", "references: each token swapped with another occurrence of the same id", 2], ["free", "references: tokens swapped freely", 0]]) {
          const r = sc[key]; if (r) series.push({ name, values: r.mean, band: [r.minimum, r.maximum], slot, dash: slot === 0 });
        }
        const note = [sc.matched.cuts_band ? `the first ${k} modes cut a near-tied band, so this k is not a stable space` : "",
          perm?.within_id_uninformative ? "too few of these tokens share an id with another for the same-id reference to say much (the pipeline flags it as uninformative)" : "",
          perm ? `${perm.movable_share != null ? `${num(100 * perm.movable_share, 2)}% of tokens can move under the same-id swap; ` : ""}references: mean and range over ${agg.permutations.n_draws} draws` : "references not computed yet"].filter(Boolean).join(". ");
        charts.push(this.lineChart(title, layers, series, { stacked: true, note }));
      }
    }
    // 2. per-layer modes (each layer embedded on its own) beside independent Gaussian vectors per token id
    {
      const title = `Between the two models, each layer embedded on its own, beside random vectors per token id (${st.s})`;
      const g = agg.gaussian?.records.find(r => r.set === st.s);
      if (!g) charts.push(this.emptyChart(title, agg.gaussian ? `no record for ${st.s}` : "needs gaussian.json, written when the run's last stage finishes"));
      else {
        const sc = g.scores[k];
        charts.push(this.lineChart(title, layers, [{ name: "the two models (per-layer modes)", values: sc.real, slot: 1, width: 2 },
          { name: "two independent random vectors per token id", values: layers.map(() => sc.gaussian), slot: 0, dash: true }],
          { note: "The per-layer modes use no coupling between layers, so this chart is the same at both strengths. The random reference gives every occurrence of an id the same vector, so it is one value for all layers." }));
      }
    }
    // 3. within one model, along the ladder of construction sets, on the natural-panel tokens outside c00
    {
      const title = `Within each model, from one set to the next (${s}), on the natural panel's tokens outside c00`;
      const recs = agg.cross?.records.filter(r => r.kind === "ladder" && r.strength === st.t) || [];
      if (!agg.cross) charts.push(this.emptyChart(title, "needs cross.json, written when the run's last stage finishes"));
      else if (!recs.length) charts.push(this.emptyChart(title, `no ladder records at ${s}`));
      else {
        const pairs = [...new Map(recs.map(r => [`${r.set_a}→${r.set_b}`, 1])).keys()];
        const rec = (m, pr) => recs.find(x => x.model === m.key && `${x.set_a}→${x.set_b}` === pr)?.scores[k];
        const series = this.models.map((m, j) => ({ name: m.name, slot: j + 1, width: 2,
          values: pairs.map(pr => { const v = rec(m, pr); return v && !v.unavailable ? v.stacked : null; }),
          ranks: pairs.map(pr => rec(m, pr)?.ranks?.at(-1)), flags: pairs.map(pr => !!rec(m, pr)?.cuts_band) }));
        const anyCut = series.some(x => x.flags.some(Boolean));
        charts.push(this.lineChart(title, pairs, series, { categorical: true, note: `All layers stacked. Each step compares two embeddings of the same model on the same placed tokens.${anyCut ? " Open circles (†): the first k modes cut a near-tied band in one of the two sets, so that space is not stable at this k." : ""}` }));
      }
    }
    // 4. between models, by panel and placement group (all layers stacked), beside the two permutation references
    {
      const title = `Between the two models, by panel and group, on ${st.s} (${s}), all layers stacked`;
      const recs = (agg.cross?.records || []).filter(r => r.kind === "models" && r.set === st.s && r.strength === st.t);
      if (!agg.cross) charts.push(this.emptyChart(title, "needs cross.json (and permutations.json for the references), written when the run's last stage finishes"));
      else if (!recs.length) charts.push(this.emptyChart(title, `no record for ${st.s} at ${s}`));
      else {
        const COH = { in_set: "in the set", seen_sequence_new_position: "new position of a text in the set", unseen_sequence: "a text the set never saw" };
        const rows = recs.map(r => {
          const perm = agg.permutations?.records.find(q => q.set === r.set && q.panel === r.panel && q.cohort === r.cohort && q.strength === r.strength);
          const sc = r.scores[k], ps = perm?.scores[k], last = a => a ? a[a.length - 1] : null;
          const ref = (key) => ps && !ps.unavailable && ps[key] ? { mean: last(ps[key].mean), lo: last(ps[key].minimum), hi: last(ps[key].maximum) } : null;
          return { label: r.panel === "construction" ? `the set's own tokens (${int(r.count)})` : `${r.panel} panel: ${COH[r.cohort] || r.cohort} (${int(r.count)})`,
            value: sc && !sc.unavailable ? sc.stacked : null, unavailable: sc?.unavailable, cuts: sc?.cuts_band,
            refs: [ref("within_id_and_cohort"), ref("free")], weak: perm?.within_id_uninformative };
        });
        charts.push(this.dotChart(title, rows, `Dot: the two models on the same tokens. Bars: the same statistic after swapping tokens with another occurrence of the same id within the group (coloured) or freely (grey), range over ${agg.permutations?.n_draws ?? "the"} draws, with the mean marked.${rows.some(r => r.weak) ? " ◦ marks a group where too few tokens share an id for the same-id reference to say much." : ""}${rows.some(r => r.cuts) ? " † the first k modes cut a near-tied band in one model." : ""}`));
      }
    }
    // 5. the natural panel (outside c00) placed into rand beside the same tokens placed into d32: agreement between models
    {
      const title = `Between the two models: the natural panel placed into rand, beside the same tokens placed into d32 (${s})`;
      const pick = set => (agg.cross?.records || []).find(r => r.kind === "models_natural_comparison" && r.set === set && r.strength === st.t);
      const ref = set => agg.permutations?.records.find(r => r.set === set && r.panel === "natural" && r.cohort === "outside_doc00" && r.strength === st.t)?.scores[k];
      const a = pick("rand"), b = pick("d32");
      if (!agg.cross) charts.push(this.emptyChart(title, "needs cross.json, written when the run's last stage finishes"));
      else if (!a || !b) charts.push(this.emptyChart(title, `needs both rand and d32 in both models at ${s}`));
      else {
        const series = [];
        for (const [rec, set, slot] of [[a, "rand", 1], [b, "d32", 2]]) {
          const sc = rec.scores[k];
          if (!sc || sc.unavailable) continue;
          series.push({ name: `placed into ${set}`, values: [...sc.layers, sc.stacked], ranks: sc.ranks, slot, width: 2, flags: sc.cuts_band ? [...sc.layers.map(() => false), true] : null });
          const r = ref(set);
          if (r && !r.unavailable && r.within_id_and_cohort) series.push({ name: `${set}: same-id swap reference`, values: r.within_id_and_cohort.mean, band: [r.within_id_and_cohort.minimum, r.within_id_and_cohort.maximum], slot, dash: true });
        }
        if (!series.length) charts.push(this.emptyChart(title, `unavailable at k = ${k}: a mode in the first ${k} is withheld`));
        else charts.push(this.lineChart(title, layers, series, { stacked: true, note: `${series.some(x => x.flags) ? "Open circle (†): the first k modes cut a near-tied band, so that space is not stable at this k. " : ""}The same ${int(a.count)} natural-panel tokens (every document but c00), placed into an embedding built from random tokens and into one built from documents; each line compares the two models' spaces on them.` }));
      }
    }
    // 6. agreement with a refit (SPEC_DEPTH.md control 3): placed tokens against the same tokens solved inside the embedding
    {
      const title = "Agreement with a refit (the d08 control, t*)";
      const rf = agg.refit_agreement;
      if (!rf) charts.push(this.emptyChart(title, "needs refit_agreement.json, written when the run's last stage finishes"));
      else {
        const ks = ["8", "16", "64"], models = this.models.filter(m => rf.records?.[m.key]);
        const f = document.createElement("figure"); f.className = "dp-chart dp-refit";
        const cell = v => v == null ? "–" : num(v, 3);
        f.innerHTML = `<figcaption>${esc(title)}</figcaption><div class="dp-tablewrap"><table class="dp-modes dp-reftab"><thead><tr><th>model</th>${ks.map(x => `<th class="dp-num">k = ${x}<br><span class="dp-hint">placed · shared</span></th>`).join("")}</tr></thead><tbody>`
          + models.map(m => `<tr><td><b>${esc(m.name)}</b></td>${ks.map(x => { const r = rf.records[m.key][x]; return `<td class="dp-num">${r ? `${cell(r.placed)} · ${cell(r.shared)}${r.cuts_band ? " †" : ""}` : "–"}</td>`; }).join("")}</tr>`).join("")
          + `</tbody></table></div><p class="dp-hint">Tokens placed into the embedding, compared with the same tokens when they are added to the set and the embedding is solved again; one rotation over all layers is fitted on the shared tokens. 1 = the same coordinates after that rotation. <i>Placed</i>: the added tokens; <i>shared</i>: the set's own tokens. The refit moves the graph, scalings and t* too, so this is not an upper bound or a known answer.${models.some(m => ks.some(x => rf.records[m.key][x]?.cuts_band)) ? " † the first k modes cut a near-tied band." : ""}${rf.skipped?.length ? ` Not computed for: ${esc(rf.skipped.join(", "))}.` : ""}</p>`;
        charts.push(f);
      }
    }
    box.replaceChildren(...charts);
  }
  // one row per group: the matched value as a dot, references as range bars with the mean marked; x from 0 to 1
  dotChart(title, rows, note) {
    const f = document.createElement("figure"); f.className = "dp-chart dp-dotchart";
    const W = 560, rowH = 40, padT = 8, padB = 24, H = padT + padB + rows.length * rowH, X0 = 12, X1 = W - 12;
    const X = v => X0 + (X1 - X0) * v, col = slot => slot ? `var(--dp-cat-${slot})` : "var(--mute)";
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
    for (const v of [0, 0.5, 1]) svg += `<line class="dp-grid" x1="${X(v)}" x2="${X(v)}" y1="${padT}" y2="${H - padB}"/><text class="dp-tick" x="${X(v)}" y="${H - padB + 14}" text-anchor="${v === 0 ? "start" : v === 1 ? "end" : "middle"}">${v}</text>`;
    rows.forEach((r, i) => {
      const y = padT + i * rowH + 30;
      svg += `<line class="dp-grid" x1="${X0}" x2="${X1}" y1="${y}" y2="${y}"/><text class="dp-rowlab" x="${X0}" y="${y - 11}">${esc(r.label)}${r.weak ? " ◦" : ""}${r.cuts ? " †" : ""}</text>`;
      r.refs.forEach((ref, j) => { if (!ref) return; const c = col(j === 0 ? 2 : 0); svg += `<line x1="${X(ref.lo)}" x2="${X(ref.hi)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="5" opacity=".3"/><line x1="${X(ref.mean)}" x2="${X(ref.mean)}" y1="${y - 4}" y2="${y + 4}" stroke="${c}" stroke-width="1.5"/>`; });
      svg += r.value == null ? `<text class="dp-tick" x="${X(0.5)}" y="${y + 3}" text-anchor="middle">${r.unavailable ? `unavailable (${esc(r.unavailable.replace(/_/g, " "))})` : "not defined: rank below k"}</text>`
        : `<circle cx="${X(r.value)}" cy="${y}" r="4" fill="${col(1)}"><title>${esc(r.label)}: ${num(r.value)}</title></circle>`;
    });
    svg += "</svg>";
    const legend = `<span class="dp-leg static"><i style="background:var(--dp-cat-1)"></i>the two models, same tokens</span><span class="dp-leg static"><i class="dp-linekey" style="--c:var(--dp-cat-2)"></i>same-id swap reference</span><span class="dp-leg static"><i class="dp-linekey" style="--c:var(--mute)"></i>free swap reference</span>`;
    f.innerHTML = `<figcaption>${esc(title)}</figcaption>${svg}<div class="dp-legend">${legend}</div><p class="dp-hint">${esc(note)}</p>`;
    return f;
  }
  emptyChart(title, why) {
    const f = document.createElement("figure"); f.className = "dp-chart dp-chart-empty";
    f.innerHTML = `<figcaption>${esc(title)}</figcaption><div class="dp-chart-none"><b>Not computed yet.</b> <span>${esc(why)}.</span></div>`;
    return f;
  }
  // a small SVG line chart, y from 0 to 1; series {name, values (null = gap), slot (0 = neutral), band [lo, hi], dash, width}
  lineChart(title, xs, series, opt = {}) {
    const f = document.createElement("figure"); f.className = "dp-chart";
    const W = 560, H = 210, padL = 34, padR = opt.stacked ? 64 : 12, padT = 10, padB = 26, n = xs.length;
    const X = i => padL + (W - padL - padR) * (n === 1 ? 0.5 : i / (n - 1)), Y = v => padT + (H - padT - padB) * (1 - v);
    const col = s => s.slot ? `var(--dp-cat-${s.slot})` : "var(--mute)";
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
    for (const v of [0, 0.5, 1]) svg += `<line class="dp-grid" x1="${padL}" x2="${W - padR}" y1="${Y(v)}" y2="${Y(v)}"/><text class="dp-tick" x="${padL - 6}" y="${Y(v) + 3}" text-anchor="end">${v}</text>`;
    const every = opt.categorical ? 1 : n > 9 ? 2 : 1;
    xs.forEach((x, i) => { if (i % every === 0 || i === n - 1) svg += `<text class="dp-tick" x="${X(i)}" y="${H - padB + 14}" text-anchor="${opt.categorical && n > 1 ? (i === 0 ? "start" : i === n - 1 ? "end" : "middle") : "middle"}">${esc(x)}</text>`; });
    if (!opt.categorical) svg += `<text class="dp-tick" x="${(padL + W - padR) / 2}" y="${H - 2}" text-anchor="middle">layer</text>`;
    const path = vals => { let d = "", pen = false; vals.forEach((v, i) => { if (v == null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; }); return d; };
    for (const s of series) {
      const vals = s.values.slice(0, n);
      if (s.band) { const lo = s.band[0].slice(0, n), hi = s.band[1].slice(0, n); let d = ""; hi.forEach((v, i) => { d += `${i ? "L" : "M"}${X(i)},${Y(v ?? 0)}`; }); for (let i = n - 1; i >= 0; i--) d += `L${X(i)},${Y(lo[i] ?? 0)}`; svg += `<path d="${d}Z" fill="${col(s)}" opacity=".14"/>`; }
      svg += `<path d="${path(vals)}" fill="none" stroke="${col(s)}" stroke-width="${s.width || 1.5}" ${s.dash ? 'stroke-dasharray="4 3"' : ""}/>`;
      vals.forEach((v, i) => {
        if (v == null) return;
        if (s.flags?.[i]) svg += `<circle cx="${X(i)}" cy="${Y(v)}" r="3.4" fill="var(--surface)" stroke="${col(s)}" stroke-width="1.6"/>`;
        else if (n < 12 || opt.categorical) svg += `<circle cx="${X(i)}" cy="${Y(v)}" r="2.6" fill="${col(s)}"/>`;
      });
      if (opt.stacked && s.values.length > n && s.values[n] != null) {
        const v = s.values[n], x = W - padR + 34;
        svg += s.flags?.[n] ? `<circle cx="${x}" cy="${Y(v)}" r="3.6" fill="var(--surface)" stroke="${col(s)}" stroke-width="1.6"/>` : `<circle cx="${x}" cy="${Y(v)}" r="3.4" fill="${col(s)}"/>`;
        if (s.band) svg += `<line x1="${x}" x2="${x}" y1="${Y(s.band[0][n])}" y2="${Y(s.band[1][n])}" stroke="${col(s)}" stroke-width="5" opacity=".2"/>`;
      }
    }
    if (opt.stacked) svg += `<text class="dp-tick" x="${W - padR + 34}" y="${H - padB + 14}" text-anchor="middle">all layers</text>`;
    svg += `<line class="dp-cross" x1="0" x2="0" y1="${padT}" y2="${H - padB}" visibility="hidden"/><rect class="dp-hit" x="${padL}" y="${padT}" width="${W - padL - padR + (opt.stacked ? 60 : 0)}" height="${H - padT - padB}" fill="transparent"/></svg>`;
    const legend = series.map(s => `<span class="dp-leg static"><i class="dp-linekey${s.dash ? " dash" : ""}" style="--c:${col(s)}"></i>${esc(s.name)}${s.band ? " (range shaded)" : ""}</span>`).join("");
    // a missing value is an overlap that is not defined: one side's first k modes have rank below k there
    const gaps = series.some(s => !s.band && s.values.slice(0, opt.stacked ? n + 1 : n).some(v => v == null));
    if (gaps) opt = { ...opt, note: `${opt.note ? opt.note + " " : ""}Gaps: the overlap is not defined there, because one model's first k modes have rank below k at that layer (a mode is almost absent from it); hover shows the ranks.` };
    f.innerHTML = `<figcaption>${esc(title)}</figcaption>${svg}<div class="dp-legend">${legend}</div>${opt.note ? `<p class="dp-hint">${esc(opt.note)}</p>` : ""}<div class="dp-chart-tip" hidden></div>`;
    const el = f.querySelector("svg"), cross = f.querySelector(".dp-cross"), tip = f.querySelector(".dp-chart-tip");
    f.querySelector(".dp-hit").addEventListener("pointermove", e => {
      const r = el.getBoundingClientRect(), sx = (e.clientX - r.left) * W / r.width;
      const stackedZone = opt.stacked && sx > W - padR + 10, i = stackedZone ? n : Math.max(0, Math.min(n - 1, Math.round((sx - padL) / ((W - padL - padR) / Math.max(1, n - 1)))));
      const x = stackedZone ? W - padR + 34 : X(i);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
      tip.hidden = false; tip.innerHTML = `<b>${stackedZone ? "all layers stacked" : opt.categorical ? esc(xs[i]) : `layer ${xs[i]}`}</b><br>` + series.map(s => `${esc(s.name)}: ${s.values[i] == null ? `not defined${s.ranks?.[i] ? ` (ranks ${s.ranks[i][0]} and ${s.ranks[i][1]})` : ""}` : num(s.values[i])}${s.flags?.[i] ? " †" : ""}${s.band ? ` (${num(s.band[0][i])}–${num(s.band[1][i])})` : ""}`).join("<br>");
      tip.style.left = Math.min(r.width - 220, Math.max(0, x * r.width / W + 10)) + "px";
    });
    f.querySelector(".dp-hit").addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
    return f;
  }
}
