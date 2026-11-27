// Live charts: <figure class="live" data-live="lines|heatmap|stack|points|table|bars|dots|hist|agreement|paired"
// data-src="assets/x/<id>.json#<key>" [data-height="240"]><div class="live-body"></div><figcaption>…</figcaption></figure>.
// The payload is one chart object, or {"panels": [{title, sub, …chart…}], cols, shared} for small multiples, which share
// one legend drawn above them. Colours are named by role and read from tokens.css, so both themes work:
// other / far / near (where neighbours come from), corpus / own (the centring), ink / mute (neutral series).
import { h, lineChart, heatmap, barChart, dotChart, histChart, stackChart, pointChart, tableChart, agreementChart, pairedChart } from "./charts.js?v=8f14246563";

const css = (v, d) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || d;
// pythia / olmo: the two models in entry D1, the colours the Depth embedding tab gives them (css/depth.css)
const ROLE = { other: "--o-other", far: "--o-far", near: "--o-near", corpus: "--c-corpus", own: "--c-own", ink: "--fg", mute: "--mute", mute2: "--mute-2", wash: "--line", accent: "--accent", pythia: "--dp-cat-1", olmo: "--dp-cat-2" };
const colour = (name, i = 0) => name && ROLE[name] ? css(ROLE[name], "#888") : name && name.startsWith("#") ? name : css(["--fg", "--mute", "--mute-2"][i % 3], "#555");
const hexRgb = x => { const v = parseInt(x.replace("#", ""), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
const rgb = c => `rgb(${c.map(Math.round).join(",")})`;
function ramp(stops, t) { const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)); return rgb(mix(stops[i], stops[i + 1], x - i)); }
const seq = t => ramp([...Array(8).keys()].map(i => hexRgb(css(`--seq-${i}`, "#2a78d6"))), t);

const jsonCache = new Map();
function getJSON(url) {
  if (!jsonCache.has(url)) jsonCache.set(url, fetch(url).then(r => { if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r.json(); }).catch(e => { jsonCache.delete(url); throw e; }));
  return jsonCache.get(url);
}
async function fromSrc(src) {
  const [url, path] = src.split("#");
  const obj = path ? path.split(".").reduce((o, k) => (o == null ? o : o[k]), await getJSON(url)) : await getJSON(url);
  if (obj == null) throw new Error(`${src}: not found`);
  return obj;
}
const paint = list => list.map((s, i) => ({ ...s, color: colour(s.color, i) }));

const RENDER = {
  lines(box, o, H) {
    const lab = o.xticklabels;
    return lineChart(box, { ...o, xfmt: lab ? (v => lab[o.x.indexOf(v)] ?? String(v)) : undefined, xticks: o.xticks || (lab ? o.x : undefined),
      ytitle: o.ylabel, height: H || o.height || 230, series: paint(o.series).map(s => ({ ...s, width: s.width || 2 })), legend: false });
  },
  heatmap(box, o, H) { return heatmap(box, { ...o, height: H, zfmt: o.fmt === "pct" ? (v => `${Math.round(100 * v)}%`) : undefined, color: t => seq(t) }); },
  stack(box, o, H) { return stackChart(box, { ...o, height: H || o.height || 210, colors: (o.segcolors || []).map(c => colour(c)) }); },
  agreement(box, o, H) { return agreementChart(box, { ...o, height: H, series: paint(o.series), colors: o.segcolors.map(c => colour(c)) }); },
  paired(box, o) { return pairedChart(box, { ...o, series: paint(o.series) }); },
  points(box, o, H) { return pointChart(box, { ...o, height: H || o.height || 230, series: paint(o.series) }); },
  table(box, o) { return tableChart(box, o); },
  bars(box, o, H) { return barChart(box, { ...o, ytitle: o.ylabel, height: H || o.height || 220, series: paint(o.series) }); },
  dots(box, o) { return dotChart(box, { ...o, series: paint(o.series) }); },
  hist(box, o, H) { return histChart(box, { ...o, ytitle: o.ylabel, height: H || o.height || 200, series: paint(o.series) }); },
};

// one legend for a whole figure: segments (stack), series (lines, points)
function legendFor(kind, o) {
  const first = o.panels ? o.panels[0] : o, items = [];
  if (!first) return null;
  if (kind === "stack" || kind === "agreement") {
    const segs = o.segments || first.segments || [], cols = o.segcolors || first.segcolors || [];
    segs.forEach((s, i) => items.push(h("span", { class: "k" }, h("i", { class: "box", style: { background: colour(cols[i]) } }), s)));
    if ((o.panels || [o]).some(p => !p.companion && p.whisker && p.whisker.some(Boolean))) items.push(h("span", { class: "k" }, h("i", { class: "whisk" }), `5–95% over documents (${segs[0]})`));
    if ((o.panels || [o]).some(p => p.zero_near)) items.push(h("span", { class: "k" }, h("i", { class: "hatch", style: { "--c": colour(cols[cols.length - 1]) } }), `${segs[segs.length - 1]}: zero by rule`));
    if (first.companion) {
      items.push(h("span", { class: "k" }, h("i", { class: "box", style: { width: "13px", background: colour("ink") } }), "Wide: model"));
      items.push(h("span", { class: "k" }, h("i", { class: "box", style: { width: "6px", opacity: ".6", background: colour("ink") } }), "Narrow: same tokens, random vectors"));
    }
  }
  if (kind === "dots" && o.panels && o.legend !== false)          // small multiples of dot charts share one legend above them
    (first.series || []).forEach((s, i) => { if (s.name) items.push(h("span", { class: "k" }, h("i", { class: `mk ${s.marker || "circle"}`, style: { "--c": colour(s.color, i) } }), s.name)); });
  if (["lines", "points", "agreement", "paired"].includes(kind)) {
    (first.series || []).forEach((s, i) => {
      if (!s.name) return;                     // unnamed series (e.g. a bundle of grey lines named once) carry no key
      const c = colour(s.color, i);
      const key = kind === "paired" ? h("i", { class: `mk ${i ? "square" : "circle hollow"}`, style: { "--c": c } })
        : kind === "points" ? h("i", { class: `mk ${s.marker || "circle"}`, style: { "--c": c } })
        : h("i", { class: "ln", style: { background: s.dash ? `repeating-linear-gradient(90deg, ${c} 0 ${s.dash === "dot" ? 2 : 4}px, transparent ${s.dash === "dot" ? 2 : 4}px ${s.dash === "dot" ? 5 : 7}px)` : c } });
      items.push(h("span", { class: "k" }, key, s.name));
    });
    if (kind === "points" && (o.panels || [o]).some(p => p.series.some(s => (s.hollow || []).some(Boolean)))) items.push(h("span", { class: "k" }, h("i", { class: "mk circle hollow", style: { "--c": colour("mute") } }), "hollow: fails the completeness rule (E3)"));
    if (kind === "points") items.push(h("span", { class: "k" }, h("i", { class: "whisk" }), "5–95% over documents"));
    if ((first.series || []).some(s => s.band)) items.push(h("span", { class: "k" }, h("i", { class: "band" }), "band: middle half over heads"));
  }
  return items.length ? h("div", { class: "chart-legend fig-legend" }, ...items) : null;
}

const onResize = (elm, fn) => { let w = elm.clientWidth; new ResizeObserver(() => { if (Math.abs(elm.clientWidth - w) > 4) { w = elm.clientWidth; fn(); } }).observe(elm); };

export async function render(fig) {
  if (fig.__live) return; fig.__live = true;
  const kind = fig.dataset.live, src = fig.dataset.src, H = +fig.dataset.height || null;
  let body = fig.querySelector(":scope > .live-body");
  if (!body) { body = h("div", { class: "live-body" }); fig.insertBefore(body, fig.querySelector(":scope > figcaption")); }
  try {
    const fn = RENDER[kind];
    if (!fn) throw new Error(`unknown live chart type “${kind}”`);
    const o = await fromSrc(src);
    body.replaceChildren();
    const lg = kind === "table" ? null : legendFor(kind, o);
    if (lg) body.append(lg);
    if (Array.isArray(o.panels)) {
      if (!o.panels.length) { body.append(h("p", { class: "chart-sub", text: "No data in this build." })); return; }
      const grid = h("div", { class: `chart-panels${o.mobile_stack ? " mobile-stack" : ""}`, style: { "--cols": o.cols || Math.min(o.panels.length, 2) } });
      body.append(grid);
      const boxes = o.panels.map(p => { const b = h("div", { class: "chart-box" }); grid.append(h("div", { class: "chart-panel" }, p.title ? h("p", { class: "chart-title", text: p.title }) : null, p.sub ? h("p", { class: "chart-sub", text: p.sub }) : null, b)); return b; });
      const shared = ["categories", "segments", "segcolors", "ylabel", "xlabel", "short", "maxAgreement"];
      const reserveTop = o.panels.some(p => p.notes || p.zero_near);
      const draw = () => o.panels.forEach((p, i) => {
        const q = { ...p, reserveTop };
        for (const k of shared) if (q[k] == null && o[k] != null) q[k] = o[k];
        fn(boxes[i], q, H);
      });
      draw(); onResize(body, draw);
    } else {
      const box = h("div", { class: "chart-box" }); body.append(box);
      const draw = () => fn(box, o, H);
      draw(); if (kind !== "table") onResize(body, draw);
    }
    if (kind === "agreement") body.append(h("p", { class: "chart-sub coverage-key", text: "Groups used are listed under each memory. Far = same text, more than 64 tokens away. Frequent (freq.) = the neighbour's token occurs 20 or more times among the analysed tokens." }));
    if (o.ylabel && ["stack", "paired", "agreement"].includes(kind)) body.prepend(h("p", { class: "chart-sub chart-ytitle", text: `${kind === "paired" ? "" : "↑ "}${o.ylabel}` }));
  } catch (e) {
    console.error("live chart:", src, e);
    body.replaceChildren(h("div", { class: "live-error", text: `Chart not drawn: ${e.message}` }));
  }
}

// redraw on a theme change (colours are read when drawing)
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  document.querySelectorAll("figure.live").forEach(f => { if (f.__live) { f.__live = false; render(f); } });
});
