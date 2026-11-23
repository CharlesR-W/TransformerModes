// Live charts in Experiments entries: <figure class="live" data-live="lines|heatmap|bars|dots|hist"
// data-src="assets/x/<id>.json#<key>" [data-height="280"]><div class="live-body"></div><figcaption>…</figcaption></figure>.
// The payload at #key (a dotted path) is a chart object (field lists in charts.js and site/xsite/AUTHORING.md), or
// {"panels": [{"title": …, …chart…}, …], "shared": true} for small multiples (shared: one value range for all).
import { h, lineChart, heatmap, barChart, dotChart, histChart } from "./charts.js";

const css = (v, d) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || d;
const hexRgb = x => { const v = parseInt(x.replace("#", ""), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; };
const mix = (a, b, t) => a.map((x, i) => x + (b[i] - x) * t);
const rgb = c => `rgb(${c.map(Math.round).join(",")})`;
function ramp(stops, t) { const x = Math.max(0, Math.min(1, t)) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)); return rgb(mix(stops[i], stops[i + 1], x - i)); }
const P = {
  set: i => css(`--c-set-${i % 8}`, "#2a78d6"),
  cls: i => css(`--c-class-${i % 10}`, "#2a78d6"),
  ord: t => ramp([...Array(8).keys()].map(i => hexRgb(css(`--c-ord-${i}`, "#2a78d6"))), t),
  seq: t => ramp([...Array(8).keys()].map(i => hexRgb(css(`--c-seq-${i}`, "#2a78d6"))), t),
  div: v => { const neg = hexRgb(css("--c-neg", "#2166ac")), mid = hexRgb(css("--c-mid", "#d3d2cc")), pos = hexRgb(css("--c-pos", "#b2182b")), a = Math.max(-1, Math.min(1, v)); return rgb(a < 0 ? mix(mid, neg, Math.pow(-a, .8)) : mix(mid, pos, Math.pow(a, .8))); },
  mute: () => css("--mute", "#6b6b70"),
};

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
function colourSeries(o, list) {
  const n = list.length;
  return list.map((s, i) => ({ ...s, color: s.color || (s.control ? P.mute() : o.colors === "ordinal" ? P.ord(n > 1 ? i / (n - 1) : 1) : o.colors === "classes" ? P.cls(i) : P.set(i)) }));
}

// categorical x labels (e.g. sites): the first and last are always labelled; the rest are thinned evenly so that no
// two labels collide, dropping any that would crowd the last one. An explicit o.xticks (x values) wins.
function categoricalTicks(x, lab, W) {
  const n = x.length; if (n < 2) return x.slice();
  const lw = Math.max(...lab.map(t => String(t).length)) * 6.6 + 10, gap = (Math.max(160, W) - 70) / (n - 1);
  const step = Math.max(1, Math.ceil(lw / gap));
  return x.filter((_, i) => i === 0 || i === n - 1 || (i % step === 0 && n - 1 - i >= step));
}

const RENDER = {
  lines(box, o, H) {
    const lab = o.xticklabels, W = box.clientWidth || 600;
    return lineChart(box, { x: o.x, xscale: o.xscale, xfmt: lab ? (v => lab[o.x.indexOf(v)] ?? String(v)) : (v => (+(+v).toPrecision(4)).toString()),
      xticks: o.xticks || (lab ? categoricalTicks(o.x, lab, W) : undefined), xname: o.xname || o.xlabel || "", xlabel: o.xlabel, ytitle: o.ylabel || o.ytitle,
      yrange: o.yrange, fmt: o.fmt, height: H || o.height || 260, refline: o.refline, shade: o.shade, band: o.band, bandLabel: o.band_label || o.bandLabel,
      cursor: o.cursor, legend: o.legend, series: colourSeries(o, o.series).map(s => ({ ...s, dash: s.dash ?? (s.control ? true : undefined), width: s.width || 2 })) });
  },
  heatmap(box, o, H) {
    return heatmap(box, { ...o, height: H, color: o.diverging ? (t => P.div(2 * t - 1)) : (t => P.seq(t)) });
  },
  bars(box, o, H) {
    return barChart(box, { ...o, ytitle: o.ylabel || o.ytitle, height: H || o.height || 220, series: colourSeries(o, o.series) });
  },
  dots(box, o) {
    return dotChart(box, { ...o, series: colourSeries(o, o.series) });
  },
  hist(box, o, H) {
    return histChart(box, { ...o, ytitle: o.ylabel || o.ytitle, height: H || o.height || 200, series: colourSeries(o, o.series) });
  },
};

const onResize = (elm, fn) => { let w = elm.clientWidth; new ResizeObserver(() => { if (Math.abs(elm.clientWidth - w) > 4) { w = elm.clientWidth; fn(); } }).observe(elm); };

export async function render(fig) {
  if (fig.__live) return; fig.__live = true;
  const kind = fig.dataset.live, src = fig.dataset.src, H = +fig.dataset.height || null;
  let body = fig.querySelector(":scope > .live-body");
  if (!body) { body = h("div", { class: "live-body" }); fig.insertBefore(body, fig.querySelector(":scope > figcaption")); }
  try {
    const fn = RENDER[kind];
    if (!fn) throw new Error(`unknown live chart type “${kind}”`);
    if (!src) throw new Error("no data-src");
    const o = await fromSrc(src);
    body.replaceChildren();
    if (Array.isArray(o.panels)) {
      const grid = h("div", { class: "chart-panels", style: { "--n": o.panels.length } });
      body.append(grid);
      const boxes = o.panels.map(p => { const b = h("div", { class: "chart-box" }); grid.append(h("div", { class: "chart-panel" }, p.title ? h("p", { class: "chart-title", text: p.title }) : null, p.sub ? h("p", { class: "chart-sub", text: p.sub }) : null, b)); return b; });
      // shared ranges: one y (or x for dot plots) range over all panels
      if (o.shared) {
        const vals = o.panels.flatMap(p => (p.series || []).flatMap(s => [...(s.y || s.values || []), ...(s.lo || []), ...(s.hi || [])])).filter(Number.isFinite);
        const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * .06 || .05;
        o.panels.forEach(p => { if (kind === "dots") p.xrange ??= [lo - pad, hi + pad]; else if (kind !== "heatmap") p.yrange ??= [Math.min(0, lo - pad), hi + pad]; });
      }
      const draw = () => o.panels.forEach((p, i) => fn(boxes[i], { ...p, colors: p.colors || o.colors }, H));
      draw(); onResize(body, draw);
    } else {
      const box = h("div", { class: "chart-box" }); body.append(box);
      const draw = () => fn(box, o, H);
      draw(); onResize(body, draw);
    }
  } catch (e) {
    console.error("live chart:", src, e);
    body.replaceChildren(h("div", { class: "live-error", text: `Chart not drawn: ${e.message}` }));
  }
}
