// Minimal SVG charts for the Experiments tab (ported and trimmed from the Coherent Modes site): thin marks, hairline
// axes, crosshair + tooltip. Line, heatmap, grouped bars, dot plot ("ladder"), step histogram. No dependencies.
const NS = "http://www.w3.org/2000/svg";
export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
export function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs) {
    const v = attrs[k]; if (v == null) continue;
    if (k === "style" && typeof v === "object") { for (const [p, x] of Object.entries(v)) { if (p.startsWith("--")) e.style.setProperty(p, x); else e.style[p] = x; } }
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "text") e.textContent = v;
    else e.setAttribute(k, v);
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
}
const css = (v, d) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || d;
const INK = () => css("--fg", "#16161a"), MUTE = () => css("--mute", "#6b6b70");

// ---------- scales and ticks
export function scale(kind, d0, d1, r0, r1) {
  let f, inv;
  if (kind === "log") {
    const a = Math.log10(d0), b = Math.log10(d1);
    f = v => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0);
    inv = p => Math.pow(10, a + (p - r0) / (r1 - r0) * (b - a));
  } else {
    f = v => r0 + (v - d0) / ((d1 - d0) || 1) * (r1 - r0);
    inv = p => d0 + (p - r0) / ((r1 - r0) || 1) * (d1 - d0);
  }
  f.inv = inv; f.kind = kind; f.domain = [d0, d1];
  return f;
}
export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + .5) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toFixed(10));
  return out;
}
// value formats: "int" | "pct" (x100, %) | "<n>f" | "<n>g"; default 3 significant figures
export function fmtOf(f) {
  const m = typeof f === "string" && f.match(/^(\d)([fg])$/);
  if (f === "int") return v => Math.round(v).toLocaleString("en");
  if (f === "pct") return v => (+(100 * v).toPrecision(3)).toString() + "%";
  if (m) return m[2] === "f" ? (v => v.toFixed(+m[1])) : (v => (+v.toPrecision(Math.max(1, +m[1]))).toString());
  return v => Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en") : (+v.toPrecision(3)).toString();
}
const tickFmt = f => f === "pct" ? (v => (+(100 * v).toFixed(1)).toString() + "%") : f ? fmtOf(f) : (v => (+v.toFixed(3)).toString());

// ---------- tooltip (the page's one #tooltip element)
const tip = () => document.getElementById("tooltip");
export function showTip(x, y, content) {
  const t = tip(); if (!t) return;
  if (typeof content === "string") t.innerHTML = content; else t.replaceChildren(content);
  t.hidden = false;
  const r = t.getBoundingClientRect(), W = window.innerWidth, H = window.innerHeight;
  let lx = x + 14, ly = y + 14;
  if (lx + r.width > W - 8) lx = x - r.width - 14;
  if (ly + r.height > H - 8) ly = y - r.height - 14;
  t.style.left = Math.max(8, lx) + "px"; t.style.top = Math.max(8, ly) + "px";
}
export function hideTip() { const t = tip(); if (t) t.hidden = true; }
export function tipRows(head, rows) {
  const d = h("div");
  if (head) d.append(h("div", { class: "tt-head", text: head }));
  for (const r of rows) {
    const k = h("span", { class: "tt-k" });
    if (r.color) k.append(h("span", { class: "tt-line", style: { "--sw": r.color } }));
    k.append(document.createTextNode(r.name));
    d.append(h("div", { class: "tt-row" }, k, h("span", { class: "tt-v", text: r.value })));
  }
  return d;
}
function legend(series, groups, extra) {
  const lg = h("div", { class: "chart-legend" });
  series.forEach((s, si) => {
    const key = h("i", { style: { "--sw": s.color } });
    if (s.dash) key.style.background = s.dash === "dot" ? `repeating-linear-gradient(90deg, ${s.color} 0 2px, transparent 2px 5px)` : `repeating-linear-gradient(90deg, ${s.color} 0 4px, transparent 4px 7px)`;
    if (s.box) Object.assign(key.style, { width: "10px", height: "10px", borderRadius: "2px", background: s.control ? "transparent" : s.color, boxShadow: s.control ? `inset 0 0 0 1.5px ${s.color}` : "" });
    if (s.marker) Object.assign(key.style, { width: "8px", height: "8px", borderRadius: s.marker === "circle" ? "50%" : "1px", background: s.hollow ? "transparent" : s.color, boxShadow: s.hollow ? `inset 0 0 0 1.5px ${s.color}` : "", transform: s.marker === "diamond" ? "rotate(45deg)" : "" });
    const k = h("span", { class: "k" }, key, s.name);
    if (groups) {
      k.addEventListener("pointerenter", () => groups.forEach((p, j) => p && p.setAttribute("opacity", j === si ? 1 : .15)));
      k.addEventListener("pointerleave", () => groups.forEach(p => p && p.setAttribute("opacity", 1)));
    }
    lg.append(k);
  });
  if (extra) lg.append(extra);
  return lg;
}
function frame(container, o, defH) {
  container.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" });
  container.append(wrap);
  const W = Math.max(240, container.clientWidth || 480), H = o.height || defH;
  return { wrap, W, H };
}
function yAxis(svg, m, W, H, Y, lo, hi, yfmt, title) {
  const g = el("g", { class: "grid" }, svg), yt = niceTicks(lo, hi, Math.max(3, Math.round(H / 55)));
  for (const v of yt) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, svg).textContent = yfmt(v); }
  if (title) el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)` }, svg).textContent = title;
}
const yTitleFits = (t, H) => t && t.length * 6.4 <= H - 50;

// ---------- line chart
// o: x, series [{name, y, color, dash (true | "dot"), width, dots, noline, band:[lo[],hi[]]}], yrange, xlabel, ytitle,
//    height, xfmt, yfmt, xticks, refline {y,label}, shade [x0,x1], band [lo[],hi[]], bandLabel, legend, xname, cursor (x)
export function lineChart(container, o) {
  const { wrap, W, H } = frame(container, o, 240);
  const yRot = yTitleFits(o.ytitle, H);
  if (o.ytitle && !yRot) wrap.before(h("p", { class: "chart-sub chart-ytitle", text: o.ytitle }));
  const m = { l: o.ml ?? (yRot ? 58 : 44), r: o.mr ?? 14, t: 10, b: o.xlabel ? 38 : 24 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.ytitle || "chart" }, wrap);
  const xs = o.x, x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = scale(o.xscale === "log" ? "log" : "linear", x0, x1, m.l, W - m.r);
  let ylo, yhi;
  const logY = o.yscale === "log";
  if (o.yrange) [ylo, yhi] = o.yrange;
  else {
    const all = o.series.flatMap(s => [...s.y, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]).concat(o.band ? [...o.band[0], ...o.band[1]] : [])
      .concat((o.reflines || []).map(r => r.y)).concat(o.refline ? [o.refline.y] : []).filter(v => Number.isFinite(v) && (!logY || v > 0));
    ylo = Math.min(...all); yhi = Math.max(...all);
    if (logY) { ylo = Math.pow(10, Math.floor(Math.log10(ylo || 1))); yhi = Math.pow(10, Math.ceil(Math.log10(yhi || 10))); }
    else { const pad = (yhi - ylo) * .06 || .05; ylo -= pad; yhi += pad; }
  }
  const Y = scale(logY ? "log" : "linear", ylo, yhi, H - m.b, m.t);
  const yfmt = o.yfmt || tickFmt(o.fmt), xfmt = o.xfmt || (v => String(v));
  if (logY) {
    const g = el("g", { class: "grid" }, svg);
    for (let e = Math.round(Math.log10(ylo)); e <= Math.round(Math.log10(yhi)); e++) { const v = Math.pow(10, e); el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); el("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, svg).textContent = v >= 1 ? String(v) : v.toPrecision(1); }
    if (yRot && o.ytitle) el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(12,${(m.t + H - m.b) / 2}) rotate(-90)` }, svg).textContent = o.ytitle;
  } else yAxis(svg, m, W, H, Y, ylo, yhi, yfmt, yRot ? o.ytitle : null);
  const ax = el("g", { class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
  const xt = (o.xticks || niceTicks(x0, x1, Math.max(3, Math.round(W / 90)))).filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  for (const v of xt) { el("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax); el("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, svg).textContent = xfmt(v); }
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  if (o.shade) el("rect", { x: X(o.shade[0]), y: m.t, width: Math.max(2, X(o.shade[1]) - X(o.shade[0])), height: H - m.b - m.t, class: "shade" }, svg);
  for (const r of [...(o.refline ? [o.refline] : []), ...(o.reflines || [])]) {
    if (!Number.isFinite(r.y) || r.y < ylo || r.y > yhi) continue;
    el("line", { x1: m.l, x2: W - m.r, y1: Y(r.y), y2: Y(r.y), class: "ref" }, svg);
    if (r.label) el("text", { x: r.at === "left" ? m.l + 4 : W - m.r - 2, y: Y(r.y) - 4, "text-anchor": r.at === "left" ? "start" : "end", class: "reflab" }, svg).textContent = r.label;
  }
  const bandD = b => {
    let d = "";
    xs.forEach((xv, i) => { if (Number.isFinite(b[1][i])) d += (d ? "L" : "M") + X(xv) + "," + Y(Math.min(yhi, b[1][i])); });
    for (let i = xs.length - 1; i >= 0; i--) if (Number.isFinite(b[0][i])) d += "L" + X(xs[i]) + "," + Y(Math.max(ylo, b[0][i]));
    return d ? d + "Z" : "";
  };
  let bandNote = null;
  if (o.band) { el("path", { d: bandD(o.band), fill: INK(), opacity: .075 }, svg); bandNote = o.bandLabel || "shaded: reference band"; }
  const groups = o.series.map(s => {
    const gg = el("g", {}, svg);
    if (s.band) el("path", { d: bandD(s.band), fill: s.color, opacity: .12 }, gg);
    let d = "", pen = false;
    xs.forEach((xv, i) => { const v = s.y[i]; if (!Number.isFinite(v)) { if (!s.join) pen = false; return; } d += (pen ? "L" : "M") + X(xv).toFixed(1) + "," + Y(v).toFixed(1); pen = true; });
    const p = el("path", { d, fill: "none", stroke: s.noline ? "none" : s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, gg);
    if (s.dash) p.setAttribute("stroke-dasharray", s.dash === "dot" ? "1.5 3.5" : "5 4");
    if (s.label !== false && o.directLabels) { let last = -1; s.y.forEach((v, i) => { if (Number.isFinite(v)) last = i; }); if (last >= 0) el("text", { x: X(xs[last]) + 6, y: Y(s.y[last]) + 3.5, "text-anchor": "start", class: "dlab" }, gg).textContent = s.short || s.name; }
    const dr = s.dots === true ? 3 : +s.dots || 0;
    if (dr) xs.forEach((xv, i) => { if (Number.isFinite(s.y[i])) el("circle", { cx: X(xv), cy: Y(s.y[i]), r: dr, fill: s.dash ? css("--surface", "#fff") : s.color, stroke: s.dash ? s.color : css("--surface", "#fff"), "stroke-width": 1.5 }, gg); });
    return gg;
  });
  if (o.cursor != null) el("line", { x1: X(o.cursor), x2: X(o.cursor), y1: m.t, y2: H - m.b, class: "cursor" }, svg);
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: INK(), "stroke-width": 1, opacity: 0 }, svg);
  const dots = o.series.map(s => el("circle", { r: 4, fill: s.color, stroke: css("--surface", "#fff"), "stroke-width": 2, opacity: 0 }, svg));
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  const nearest = px => { let bi = 0, bd = Infinity; xs.forEach((xv, i) => { const d = Math.abs(X(xv) - px); if (d < bd) { bd = d; bi = i; } }); return bi; };
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), i = nearest(ev.clientX - r.left);
    cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i])); cross.setAttribute("opacity", .3);
    o.series.forEach((s, si) => { const v = s.y[i], d = dots[si]; if (Number.isFinite(v)) { d.setAttribute("cx", X(xs[i])); d.setAttribute("cy", Y(v)); d.setAttribute("opacity", 1); } else d.setAttribute("opacity", 0); });
    const vf = o.fmt ? fmtOf(o.fmt) : yfmt;
    const rows = o.series.map(s => ({ name: s.name, value: Number.isFinite(s.y[i]) ? vf(s.y[i]) + (s.band && Number.isFinite(s.band[0][i]) ? ` (${vf(s.band[0][i])} to ${vf(s.band[1][i])})` : "") : "n/a", color: s.color, v: s.y[i] }));
    if (rows.length > 6) rows.sort((a, b) => (b.v ?? -1e9) - (a.v ?? -1e9));
    showTip(ev.clientX, ev.clientY, tipRows((o.xname ? o.xname + " " : "") + xfmt(xs[i]), rows));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); dots.forEach(d => d.setAttribute("opacity", 0)); hideTip(); });
  const note = bandNote ? h("span", { class: "k mute" }, h("i", { class: "band" }), bandNote) : null;
  if (o.legend !== false && o.series.length > 1) wrap.append(legend(o.series, groups, note));
  else if (note) wrap.append(h("div", { class: "chart-legend" }, note));
  return { svg, X, Y };
}

// ---------- heatmap (rows x cols), optional slider over a leading axis
// o: rows, cols, z ([r][c] or [k][r][c]), zrange, color(t)->css, zfmt, zlabel, rowlabel, collabel, slider {label, values}
export function heatmap(container, o) {
  container.replaceChildren();
  const ctrl = h("div", { class: "chart-controls" }), wrap = h("div", { class: "chart-wrap" });
  container.append(ctrl);
  if (o.zlabel) container.append(h("p", { class: "chart-sub", text: "colour: " + o.zlabel }));
  container.append(wrap);
  const has3 = !!o.slider;
  let k = has3 ? o.slider.values.length - 1 : 0;
  if (has3) {
    const val = h("b", { text: String(o.slider.values[k]) });
    const inp = h("input", { type: "range", min: 0, max: o.slider.values.length - 1, step: 1, value: k, "aria-label": o.slider.label || "slice" });
    inp.addEventListener("input", () => { k = +inp.value; val.textContent = String(o.slider.values[k]); draw(); });
    ctrl.append(h("label", {}, (o.slider.label || "") + " "), inp, val);
  } else ctrl.remove();
  const nr = o.rows.length, nc = o.cols.length;
  const draw = () => {
    wrap.replaceChildren();
    const Wc = Math.max(260, container.clientWidth || 480), ml = o.ml ?? (o.rowlabel ? 76 : 60), mt = 8;
    const cw = Math.max(6, Math.min(44, (Wc - ml - 70) / nc)), ch = o.cellH || Math.max(12, Math.min(26, cw));
    const colTxt = o.cols.map(String), rot = colTxt.some(t => t.length * 6.1 > cw - 3), lmax = Math.max(...colTxt.map(t => t.length)) * 6.1;
    const mb = (rot ? 30 + Math.min(110, lmax * .72) : 30) + (o.collabel ? 16 : 0);
    const W = Math.min(Wc, ml + nc * cw + 96), H = mt + nr * ch + mb;
    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.zlabel || "heatmap" }, wrap);
    const Z = has3 ? o.z[k] : o.z, [zl, zh] = o.zrange || [0, 1], zfmt = o.zfmt || fmtOf(o.fmt);
    for (let r = 0; r < nr; r++) {
      el("text", { x: ml - 6, y: mt + r * ch + ch / 2 + 3.5, "text-anchor": "end" }, svg).textContent = o.rows[r];
      for (let c = 0; c < nc; c++) {
        const v = Z[r][c];
        const rect = el("rect", { x: ml + c * cw + .5, y: mt + r * ch + .5, width: cw - 1, height: ch - 1, rx: 1.5, fill: Number.isFinite(v) ? o.color((v - zl) / (zh - zl)) : css("--line-2", "#efeee9") }, svg);
        rect.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(null, [
          { name: o.rowlabel || "row", value: String(o.rows[r]) }, { name: o.collabel || "column", value: String(o.cols[c]) },
          { name: o.zlabel || "value", value: Number.isFinite(v) ? zfmt(v) : "n/a" }])));
        rect.addEventListener("pointerleave", hideTip);
      }
    }
    const every = Math.ceil(nc / Math.max(1, Math.floor(nc * cw / (rot ? 13 : 34))));
    for (let c = 0; c < nc; c += every) {
      const x = ml + c * cw + cw / 2, y = mt + nr * ch + (rot ? 8 : 14);
      const t = el("text", rot ? { x, y, "text-anchor": "end", "dominant-baseline": "middle", transform: `rotate(-45 ${x} ${y})` } : { x, y, "text-anchor": "middle" }, svg);
      t.textContent = colTxt[c];
    }
    if (o.collabel) el("text", { x: ml + nc * cw / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.collabel;
    if (o.rowlabel) el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ttl", transform: `translate(11,${mt + nr * ch / 2}) rotate(-90)` }, svg).textContent = o.rowlabel;
    const bx = ml + nc * cw + 16, bh = Math.min(140, nr * ch), id = "g" + Math.random().toString(36).slice(2);
    const lg = el("linearGradient", { id, x1: 0, x2: 0, y1: 1, y2: 0 }, el("defs", {}, svg));
    for (let i = 0; i <= 8; i++) el("stop", { offset: i / 8, "stop-color": o.color(i / 8) }, lg);
    el("rect", { x: bx, y: mt, width: 10, height: bh, fill: `url(#${id})`, rx: 2 }, svg);
    [[zh, mt + 4], [zl, mt + bh]].forEach(([v, y]) => { el("text", { x: bx + 14, y, "text-anchor": "start" }, svg).textContent = zfmt(v); });
  };
  draw();
  return { redraw: draw };
}

// ---------- grouped vertical bars
// o: categories, series [{name, values, err [[lo,hi]], color, control}], yrange, ytitle, xlabel, refline {y,label}, fmt, height
export function barChart(container, o) {
  const { wrap, W, H } = frame(container, o, 220);
  const yRot = yTitleFits(o.ytitle, H);
  if (o.ytitle && !yRot) wrap.before(h("p", { class: "chart-sub chart-ytitle", text: o.ytitle }));
  const nc = o.categories.length, ns = o.series.length;
  const labW = Math.max(...o.categories.map(c => String(c).length)) * 6.2;
  const m = { l: yRot ? 58 : 44, r: 10, t: 10, b: 0 };
  const bandW = (W - m.l - m.r) / nc, rot = labW > bandW - 4;
  m.b = (rot ? 14 + Math.min(80, labW * .72) : 22) + (o.xlabel ? 16 : 0);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.ytitle || "bar chart" }, wrap);
  const all = o.series.flatMap(s => [...s.values, ...(s.err || []).flat()]).filter(Number.isFinite);
  let [ylo, yhi] = o.yrange || [Math.min(0, ...all), Math.max(...all) * 1.06 || 1];
  const Y = scale("linear", ylo, yhi, H - m.b, m.t), yfmt = tickFmt(o.fmt), vfmt = fmtOf(o.fmt);
  yAxis(svg, m, W, H, Y, ylo, yhi, yfmt, yRot ? o.ytitle : null);
  el("line", { x1: m.l, x2: W - m.r, y1: Y(Math.max(ylo, 0)) + .5, y2: Y(Math.max(ylo, 0)) + .5 }, el("g", { class: "axis" }, svg));
  const inner = Math.min(bandW * .78, ns * 26), bw = inner / ns;
  const groups = o.series.map(() => el("g", {}, svg));
  o.categories.forEach((c, ci) => {
    const x0 = m.l + ci * bandW + (bandW - inner) / 2;
    o.series.forEach((s, si) => {
      const v = s.values[ci]; if (!Number.isFinite(v)) return;
      const x = x0 + si * bw, y = Y(Math.max(v, ylo)), base = Y(Math.max(0, ylo));
      const rect = el("rect", { x: x + .5, y: Math.min(y, base), width: Math.max(1, bw - 1.5), height: Math.max(.5, Math.abs(base - y)), rx: 1.5,
        fill: s.control ? "none" : s.color, stroke: s.control ? s.color : "none", "stroke-width": s.control ? 1.3 : 0, "stroke-dasharray": s.control ? "3 2" : null }, groups[si]);
      const e = s.err && s.err[ci];
      if (e && Number.isFinite(e[0])) el("line", { x1: x + bw / 2, x2: x + bw / 2, y1: Y(e[0]), y2: Y(e[1]), stroke: INK(), "stroke-width": 1 }, groups[si]);
      rect.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(String(c), [{ name: s.name, value: vfmt(v) + (e && Number.isFinite(e[0]) ? ` (${vfmt(e[0])} to ${vfmt(e[1])})` : ""), color: s.color }])));
      rect.addEventListener("pointerleave", hideTip);
    });
    const x = m.l + ci * bandW + bandW / 2, y = H - m.b + (rot ? 8 : 14) - (o.xlabel ? 0 : 0);
    el("text", rot ? { x, y, "text-anchor": "end", "dominant-baseline": "middle", transform: `rotate(-45 ${x} ${y})` } : { x, y, "text-anchor": "middle" }, svg).textContent = c;
  });
  if (o.refline) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(o.refline.y), y2: Y(o.refline.y), class: "ref" }, svg);
    if (o.refline.label) el("text", { x: W - m.r - 4, y: Y(o.refline.y) - 5, "text-anchor": "end" }, svg).textContent = o.refline.label;
  }
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  if (ns > 1 && o.legend !== false) wrap.append(legend(o.series.map(s => ({ ...s, box: true })), groups));
  return { svg, Y };
}

// ---------- dot plot / "ladder": one row per category, value on x
// o: rows, rowgroups [{label, from, to}], series [{name, values, lo, hi, marker, hollow, color}], xrange, xlabel, refline {x,label}, fmt
export function dotChart(container, o) {
  container.replaceChildren();
  const wrap = h("div", { class: "chart-wrap" }); container.append(wrap);
  const W = Math.max(240, container.clientWidth || 480), rowH = 22, nr = o.rows.length;
  const gaps = (o.rowgroups || []).length, labW = Math.min(W * .45, Math.max(...o.rows.map(r => String(r).length)) * 6.6 + 12);
  const m = { l: labW, r: 14, t: 8, b: o.xlabel ? 36 : 22 };
  const rowY = [], H0 = m.t; let y = H0;
  o.rows.forEach((_, i) => { if ((o.rowgroups || []).some(g => g.from === i)) y += 18; rowY.push(y + rowH / 2); y += rowH; });
  const H = y + m.b;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.xlabel || "dot plot" }, wrap);
  const all = o.series.flatMap(s => [...s.values, ...(s.lo || []), ...(s.hi || [])]).filter(Number.isFinite);
  let [x0, x1] = o.xrange || [Math.min(...all), Math.max(...all)];
  if (!o.xrange) { const pad = (x1 - x0) * .06 || .05; x0 -= pad; x1 += pad; }
  const X = scale("linear", x0, x1, m.l, W - m.r), xfmt = tickFmt(o.fmt), vfmt = fmtOf(o.fmt);
  const g = el("g", { class: "grid" }, svg);
  for (const v of niceTicks(x0, x1, Math.max(3, Math.round((W - m.l) / 80)))) {
    el("line", { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, g);
    el("text", { x: X(v), y: H - m.b + 14, "text-anchor": "middle" }, svg).textContent = xfmt(v);
  }
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  for (const rg of o.rowgroups || []) el("text", { x: 4, y: rowY[rg.from] - rowH / 2 - 5, "text-anchor": "start", class: "ttl" }, svg).textContent = rg.label;
  o.rows.forEach((r, i) => {
    el("line", { x1: m.l, x2: W - m.r, y1: rowY[i], y2: rowY[i], class: "row" }, svg);
    el("text", { x: m.l - 8, y: rowY[i] + 3.5, "text-anchor": "end" }, svg).textContent = r;
  });
  if (o.refline) {
    el("line", { x1: X(o.refline.x), x2: X(o.refline.x), y1: m.t, y2: H - m.b, class: "ref" }, svg);
    if (o.refline.label) el("text", { x: X(o.refline.x) + 4, y: m.t + 9, "text-anchor": "start" }, svg).textContent = o.refline.label;
  }
  const ns = o.series.length, off = i => ns > 1 ? (i - (ns - 1) / 2) * Math.min(5, 12 / ns) : 0;
  const groups = o.series.map((s, si) => {
    const gg = el("g", {}, svg);
    s.values.forEach((v, i) => {
      if (!Number.isFinite(v)) return;
      const cy = rowY[i] + off(si), cx = X(v);
      if (s.lo && Number.isFinite(s.lo[i])) el("line", { x1: X(s.lo[i]), x2: X(s.hi[i]), y1: cy, y2: cy, stroke: s.color, "stroke-width": 1.4, opacity: .7 }, gg);
      const st = s.hollow ? { fill: css("--surface", "#fff"), stroke: s.color, "stroke-width": 1.6 } : { fill: s.color, stroke: css("--surface", "#fff"), "stroke-width": 1.2 };
      const mk = s.marker || "circle", r = 4.5;
      const shape = mk === "diamond" ? el("path", { d: `M${cx},${cy - r * 1.25}L${cx + r * 1.1},${cy}L${cx},${cy + r * 1.25}L${cx - r * 1.1},${cy}Z`, ...st }, gg)
        : mk === "triangle" ? el("path", { d: `M${cx},${cy - r * 1.2}L${cx + r * 1.1},${cy + r * .8}L${cx - r * 1.1},${cy + r * .8}Z`, ...st }, gg)
        : mk === "square" ? el("rect", { x: cx - r * .9, y: cy - r * .9, width: r * 1.8, height: r * 1.8, ...st }, gg)
        : el("circle", { cx, cy, r, ...st }, gg);
      shape.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(String(o.rows[i]), [{ name: s.name, value: vfmt(v) + (s.lo && Number.isFinite(s.lo[i]) ? ` (${vfmt(s.lo[i])} to ${vfmt(s.hi[i])})` : ""), color: s.color }])));
      shape.addEventListener("pointerleave", hideTip);
    });
    return gg;
  });
  if (ns > 1 && o.legend !== false) wrap.append(legend(o.series.map(s => ({ ...s, marker: s.marker || "circle" })), groups));
  return { svg, X };
}

// ---------- overlaid step histograms
// o: edges [B+1], series [{name, counts [B], color, control}], norm "share" | "count", xlabel, ytitle, xticklabels, fmt
export function histChart(container, o) {
  const { wrap, W, H } = frame(container, o, 200);
  const yRot = yTitleFits(o.ytitle, H);
  if (o.ytitle && !yRot) wrap.before(h("p", { class: "chart-sub chart-ytitle", text: o.ytitle }));
  const m = { l: yRot ? 58 : 44, r: 12, t: 10, b: o.xlabel ? 38 : 24 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.xlabel || "histogram" }, wrap);
  const E = o.edges, B = E.length - 1;
  const ys = o.series.map(s => { const tot = s.counts.reduce((a, b) => a + b, 0) || 1; return s.counts.map(c => o.norm === "count" ? c : c / tot); });
  const yhi = Math.max(...ys.flat()) * 1.08 || 1;
  const X = scale("linear", E[0], E[B], m.l, W - m.r), Y = scale("linear", 0, yhi, H - m.b, m.t);
  const fmt = o.norm === "count" ? "int" : (o.fmt || "pct");
  yAxis(svg, m, W, H, Y, 0, yhi, tickFmt(fmt), yRot ? o.ytitle : null);
  const ax = el("g", { class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
  if (o.xticklabels) E.slice(0, B).forEach((e, i) => { el("text", { x: X((E[i] + E[i + 1]) / 2), y: H - m.b + 15, "text-anchor": "middle" }, svg).textContent = o.xticklabels[i] ?? ""; });
  else for (const v of niceTicks(E[0], E[B], Math.max(3, Math.round(W / 90)))) { el("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax); el("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, svg).textContent = (+v.toPrecision(4)).toString(); }
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  const vfmt = fmtOf(fmt);
  const groups = o.series.map((s, si) => {
    const gg = el("g", {}, svg), y = ys[si];
    let d = `M${X(E[0])},${Y(0)}`;
    for (let i = 0; i < B; i++) d += `L${X(E[i])},${Y(y[i])}L${X(E[i + 1])},${Y(y[i])}`;
    d += `L${X(E[B])},${Y(0)}`;
    if (!s.control) el("path", { d: d + "Z", fill: s.color, opacity: .1 }, gg);
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.8, "stroke-dasharray": s.control ? "4 3" : null, "stroke-linejoin": "round" }, gg);
    return gg;
  });
  const hit = el("rect", { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const r = svg.getBoundingClientRect(), v = X.inv(ev.clientX - r.left), i = Math.max(0, Math.min(B - 1, E.findIndex((e, k) => k < B && v >= e && v < E[k + 1])));
    const lab = o.xticklabels ? o.xticklabels[i] : `${(+E[i].toPrecision(4))} to ${(+E[i + 1].toPrecision(4))}`;
    showTip(ev.clientX, ev.clientY, tipRows(lab, o.series.map((s, si) => ({ name: s.name, value: vfmt(ys[si][i]) + (o.norm !== "count" ? ` (${s.counts[i]})` : ""), color: s.color }))));
  });
  hit.addEventListener("pointerleave", hideTip);
  if (o.series.length > 1 && o.legend !== false) wrap.append(legend(o.series.map(s => ({ ...s, dash: s.control })), groups));
  return { svg, X, Y };
}

// ---------- stacked shares (one bar per category, segments bottom to top), for origin shares
// o: categories, values [[s0, s1, s2] | null], segments (names), colors (css colours), whisker [[lo, hi] | null] on the
//    top of the first segment, zero_near (the last segment is zero by rule: a hatched cap), notes (text above bars),
//    note_label, alt [[...] | null] + alt_label (shown in the tooltip), ylabel, xlabel, height, short (axis labels)
export function stackChart(container, o) {
  const { wrap, W, H } = frame(container, o, 200);
  const nc = o.categories.length, labels = o.short || o.categories;
  const m = { l: 40, r: 6, t: o.notes || o.zero_near || o.reserveTop || Number.isFinite(o.chance) ? 26 : 10, b: o.xlabel ? 36 : 22 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.ylabel || "stacked shares" }, wrap);
  const Y = scale("linear", 0, 1, H - m.b, m.t);
  yAxis(svg, m, W, H, Y, 0, 1, v => `${Math.round(v * 100)}%`, null);
  const band = (W - m.l - m.r) / nc, bw = Math.min(30, band * .62), gap = 1.5, surf = css("--surface", "#fff"), ink = INK();
  const pf = v => `${Math.round(100 * v)}%`, tf = v => Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : "n/a";  // tf: tooltips, one decimal
  let hatchId = null;
  if (o.zero_near) {
    hatchId = "h" + Math.random().toString(36).slice(2);
    const pat = el("pattern", { id: hatchId, width: 4, height: 4, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, el("defs", {}, svg));
    el("rect", { width: 4, height: 4, fill: surf }, pat);
    el("line", { x1: 0, y1: 0, x2: 0, y2: 4, stroke: o.colors[o.colors.length - 1], "stroke-width": 2 }, pat);
  }
  o.categories.forEach((c, i) => {
    const span = o.companion ? Math.min(bw + 14, band * .76) : bw;
    const wide = o.companion ? span * .65 : span, x = m.l + i * band + (band - span) / 2, v = o.values[i];
    const tx = m.l + i * band + band / 2;
    el("text", { x: tx, y: H - m.b + 14, "text-anchor": "middle" }, svg).textContent = labels[i];
    if (!v) {
      el("rect", { x: x + .5, y: Y(1), width: bw - 1, height: Y(0) - Y(1), fill: "none", stroke: MUTE(), "stroke-dasharray": "3 3", rx: 2 }, svg);
      el("text", { x: tx, y: Y(.5), "text-anchor": "middle", class: "na" }, svg).textContent = "n/a";
      return;
    }
    const g = el("g", { class: "stack" }, svg);
    const draw = (values, x, w, opacity) => {
      let acc = 0;
      values?.forEach((share, k) => {
        if (!(share > 0)) return;
        const y0 = Y(acc), y1 = Y(acc + share);
        el("rect", { x, y: y1, width: w, height: Math.max(.5, y0 - y1 - (acc + share < .999 ? gap : 0)), fill: o.colors[k], rx: 1.5, opacity }, g);
        acc += share;
      });
    };
    draw(v, x, wide);
    const companion = o.companion?.[i];
    if (companion) draw(companion, x + wide + 2, span - wide - 2, .60);
    if (o.zero_near) el("rect", { x, y: Y(1) - 7, width: bw, height: 5, fill: `url(#${hatchId})`, stroke: o.colors[o.colors.length - 1], "stroke-width": .6, rx: 1 }, g);
    const w = o.whisker && o.whisker[i];
    if (!o.companion && w && Number.isFinite(w[0])) {
      const cx = x + bw / 2;
      el("line", { x1: cx, x2: cx, y1: Y(w[0]), y2: Y(w[1]), stroke: ink, "stroke-width": 1.3 }, g);
      for (const e of w) el("line", { x1: cx - 4, x2: cx + 4, y1: Y(e), y2: Y(e), stroke: ink, "stroke-width": 1.3 }, g);
    }
    const note = o.notes && o.notes[i];
    if (note) el("text", { x: tx, y: m.t - 12, "text-anchor": "middle", class: "note" }, svg).textContent = note;
    const hit = el("rect", { x: m.l + i * band, y: m.t - 10, width: band, height: H - m.b - m.t + 10, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", ev => {
      const rows = o.segments.map((name, k) => ({ name, value: tf(v[k]) + (o.alt && o.alt[i] ? ` (${o.alt_label || "alt"}: ${tf(o.alt[i][k])})` : "") + (companion ? ` (random vectors: ${tf(companion[k])})` : ""), color: o.colors[k] }));
      if (w && Number.isFinite(w[0])) rows.push({ name: `${o.segments[0]}, 5–95% over documents`, value: `${tf(w[0])} to ${tf(w[1])}` });
      if (note) rows.push({ name: o.note_label || "note", value: note });
      if (o.zero_near) rows.push({ name: o.segments[o.segments.length - 1], value: "zero by rule" });
      showTip(ev.clientX, ev.clientY, tipRows(`${o.xname || ""}${o.categories[i]}`, rows));
    });
    hit.addEventListener("pointerleave", hideTip);
  });
  if (Number.isFinite(o.chance)) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(o.chance), y2: Y(o.chance), stroke: surf, "stroke-width": 2.5, "pointer-events": "none" }, svg);
    el("line", { x1: m.l, x2: W - m.r, y1: Y(o.chance), y2: Y(o.chance), stroke: ink, "stroke-width": 1, "stroke-dasharray": "4 3", "pointer-events": "none" }, svg);
    el("text", { x: W - m.r, y: 15, "text-anchor": "end", class: "ttl" }, svg).textContent = `If drawn at random, grey would reach ${pf(o.chance)}`;
  }
  if (o.notes && o.note_label) el("text", { x: m.l - 4, y: m.t - 12, "text-anchor": "end", class: "note" }, svg).textContent = o.note_label;
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  return { svg };
}

const chartPct = v => Number.isFinite(v) ? `${Number((v * 100).toFixed(2))}%` : "n/a";
function markTip(node, label, rows) {
  el("title", {}, node).textContent = `${label}: ${rows.map(r => `${r.name} ${r.value}`).join("; ")}`;
  node.addEventListener("pointermove", e => showTip(e.clientX, e.clientY, tipRows(label, rows)));
  node.addEventListener("pointerleave", hideTip);
}

// group labels under each memory: two lines ("other texts" / "frequent") when the column is wide enough, else "other, freq."
function groupLines(name, band) {
  const [where, freq] = name.split(": ");
  if (band >= 80) return [{ Other: "other texts", Far: "same text, far", Near: "same text, near" }[where] || where,
    freq === "both" ? "rare + frequent" : freq];
  return [`${({ Other: "other", Far: "far", Near: "near" }[where] || where)}, ${({ frequent: "freq.", rare: "rare", both: "both" }[freq] || freq)}`];
}
// Agreement lines and the coverage stacks share the same memory positions.
export function agreementChart(container, o) {
  const { wrap, W } = frame(container, {}, 145);
  const plotH = Math.max(116, Math.min(W < 400 ? 150 : Infinity, o.height || 145));
  const m = { l: 37, r: 5, t: 9, b: 0 }, baseline = plotH;
  const bandW = (W - 37 - 5) / o.categories.length;
  const nLines = Math.max(1, ...o.groups.map(gs => gs.flatMap(g => groupLines(g, bandW)).length));
  const barTop = baseline + 34, barH = 68, H = barTop + barH + 46 + 12 * nLines;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${o.title}: top-1 agreement and weighted pair shares by memory` }, wrap);
  const Y = scale("linear", 0, o.maxAgreement, baseline, m.t), band = (W - m.l - m.r) / o.categories.length;
  const X = i => m.l + (i + .5) * band;
  const surf = css("--surface", "#fff");
  yAxis(svg, m, W, plotH, Y, 0, o.maxAgreement, chartPct, null);
  el("text", { x: m.l, y: baseline + 21, class: "ttl" }, svg).textContent = "Share of all pairs";
  for (const [x, label] of [[barTop + 4, "100%"], [barTop + barH, "0%"]])
    el("text", { x: m.l - 6, y: x, "text-anchor": "end" }, svg).textContent = label;
  for (const series of o.series) {
    let d = "", pen = false;
    series.y.forEach((v, i) => {
      if (!Number.isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${X(i)},${Y(v)}`; pen = true;
    });
    el("path", { d, fill: "none", stroke: series.color, "stroke-width": 2, "stroke-dasharray": series.dash ? "5 4" : null }, svg);
    series.y.forEach((v, i) => {
      if (Number.isFinite(v)) el("circle", { cx: X(i), cy: Y(v), r: 3.2, fill: series.dash ? surf : series.color, stroke: series.color, "stroke-width": 1.2 }, svg);
    });
  }
  o.categories.forEach((label, i) => {
    const x = X(i), w = Math.min(40, band * .68), shares = o.strip[i];
    let y = barTop + barH;
    if (shares) shares.forEach((v, j) => {
      if (!v) return;
      y -= v * barH;
      el("rect", { x: x - w / 2, y, width: w, height: v * barH, fill: o.colors[j] }, svg);
    });
    else el("text", { x, y: y - 20, "text-anchor": "middle" }, svg).textContent = "n/a";
    const tick = barTop + barH + 16;
    el("text", { x, y: tick, "text-anchor": "middle", class: "ttl" }, svg).textContent = label;
    o.groups[i].flatMap(name => groupLines(name, band).map((line, k) => [line, k])).forEach(([line, k], j) => {
      el("text", { x, y: tick + 14 + j * 12, "text-anchor": "middle", class: k ? "coverage-group sub" : "coverage-group" }, svg).textContent = line;
    });
    const rows = [...o.series.map(s => ({ name: s.name, value: chartPct(s.y[i]), color: s.color })),
      ...o.segments.map((name, j) => ({ name, value: chartPct(shares?.[j]), color: o.colors[j] })),
      { name: "Groups used", value: o.groups[i].join("; ") || "none" }];
    markTip(el("rect", { x: x - band / 2, y: m.t, width: band, height: H - m.t - 12, fill: "transparent" }, svg), `Memory ${label}`, rows);
  });
  el("text", { x: m.l + (W - m.l - m.r) / 2, y: H - 3, "text-anchor": "middle", class: "ttl" }, svg).textContent = "Memory · groups used below each label";
}

// Each row connects two shares measured on the same screen sample.
export function pairedChart(container, o) {
  // rows with chosen: true are a different setting: drawn after a gap, under their own note (row.note)
  const step = 28, gapH = 30, top = o.head ? 24 : 13;
  const ys = []; let y0 = top;
  o.rows.forEach((row, i) => { if (row.chosen && !(o.rows[i - 1] || {}).chosen) y0 += gapH; ys.push(y0 + step / 2); y0 += step; });
  const bottom = y0;
  const { wrap, W, H } = frame(container, {}, bottom + 25);
  const l = 76, r = 20, X = scale("linear", 0, 1, l, W - r), surf = css("--surface", "#fff");
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${o.title}: ${o.ylabel}` }, wrap);
  const g = el("g", { class: "grid" }, svg);
  for (const v of [0, .25, .5, .75, 1]) {
    el("line", { x1: X(v), x2: X(v), y1: top, y2: bottom }, g);
    el("text", { x: X(v), y: bottom + 15, "text-anchor": "middle" }, svg).textContent = chartPct(v);
  }
  if (o.head) el("text", { x: 0, y: top - 8, class: "ttl" }, svg).textContent = o.head;
  o.rows.forEach((row, i) => {
    const y = ys[i], [a, b] = row.values;
    if (row.chosen && !(o.rows[i - 1] || {}).chosen) {
      el("line", { x1: 0, x2: W - r, y1: y - step / 2 - gapH + 6, y2: y - step / 2 - gapH + 6, class: "row" }, svg);
      if (row.note) el("text", { x: 0, y: y - step / 2 - 6, class: "ttl" }, svg).textContent = row.note;
    }
    el("text", { x: l - 9, y: y + 4, "text-anchor": "end", class: row.chosen ? "ttl" : null }, svg).textContent = row.label;
    if (Number.isFinite(a) && Number.isFinite(b)) el("line", { x1: X(a), x2: X(b), y1: y, y2: y, stroke: MUTE(), opacity: .65 }, svg);
    if (Number.isFinite(a)) el("circle", { cx: X(a), cy: y, r: 4.5, fill: surf, stroke: o.series[0].color, "stroke-width": 2 }, svg);
    if (Number.isFinite(b)) el("rect", { x: X(b) - 4, y: y - 4, width: 8, height: 8, rx: 1, fill: o.series[1].color }, svg);
    if (!Number.isFinite(a) && !Number.isFinite(b)) el("text", { x: X(.5), y: y + 4, "text-anchor": "middle" }, svg).textContent = "n/a";
    markTip(el("rect", { x: 0, y: y - step / 2, width: W, height: step, fill: "transparent" }, svg),
      row.chosen ? (row.note || row.label) : `${o.head ? o.head.replace(/,? by memory.*$/, "") : "Cosine"}, memory ${row.label}`, o.series.map((s, j) => ({ name: s.name, value: chartPct(row.values[j]), color: s.color })));
  });
}

// ---------- points with whiskers over categories (vertical), several series offset within each category
// o: categories, series [{name, values, lo, hi, hollow [bool], color, marker}], refline {y, label}, ylabel, xlabel, fmt, yrange
export function pointChart(container, o) {
  const { wrap, W, H } = frame(container, o, 230);
  const titled = o.ylabel && yTitleFits(o.ylabel, H);
  const m = { l: titled ? 62 : 46, r: 8, t: 16, b: o.xlabel ? 36 : 22 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.title || o.ylabel || "points" }, wrap);
  const all = o.series.flatMap(s => [...s.values, ...(s.lo || []), ...(s.hi || [])]).concat(o.refline ? [o.refline.y] : []).filter(Number.isFinite);
  let [lo, hi] = o.yrange || (all.length ? [Math.min(...all), Math.max(...all)] : [-.1, .1]);
  if (!o.yrange) { const pad = (hi - lo) * .08 || .02; lo -= pad; hi += pad; }
  const Y = scale("linear", lo, hi, H - m.b, m.t), vf = fmtOf(o.fmt);
  yAxis(svg, m, W, H, Y, lo, hi, tickFmt(o.fmt), titled ? o.ylabel : null);
  if (o.refline && o.refline.y >= lo && o.refline.y <= hi) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(o.refline.y), y2: Y(o.refline.y), class: "ref" }, svg);
    if (o.refline.label) el("text", { x: m.l + 3, y: Y(o.refline.y) + 12, "text-anchor": "start", class: "reflab" }, svg).textContent = o.refline.label;
  }
  const nc = o.categories.length, ns = o.series.length, band = (W - m.l - m.r) / nc, step = Math.min(12, band * .7 / Math.max(1, ns));
  o.categories.forEach((c, i) => el("text", { x: m.l + (i + .5) * band, y: H - m.b + 14, "text-anchor": "middle" }, svg).textContent = c);
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 3, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  const surf = css("--surface", "#fff");
  o.series.forEach((s, si) => {
    const g = el("g", {}, svg);
    s.values.forEach((v, i) => {
      const cx = m.l + (i + .5) * band + (si - (ns - 1) / 2) * step;
      if (!Number.isFinite(v)) return;
      if (s.lo && Number.isFinite(s.lo[i])) el("line", { x1: cx, x2: cx, y1: Y(s.lo[i]), y2: Y(s.hi[i]), stroke: s.color, "stroke-width": 1.4, opacity: .8 }, g);
      const hollow = s.hollow && s.hollow[i], r = 4.2, cy = Y(v);
      const st = hollow ? { fill: surf, stroke: s.color, "stroke-width": 1.7 } : { fill: s.color, stroke: surf, "stroke-width": 1.2 };
      const shape = s.marker === "diamond" ? el("path", { d: `M${cx},${cy - r * 1.3}L${cx + r * 1.15},${cy}L${cx},${cy + r * 1.3}L${cx - r * 1.15},${cy}Z`, ...st }, g) : el("circle", { cx, cy, r, ...st }, g);
      shape.addEventListener("pointermove", ev => showTip(ev.clientX, ev.clientY, tipRows(`${o.xname || "memory "}${c(i)}`, [{ name: s.name, value: vf(v) + (s.lo && Number.isFinite(s.lo[i]) ? ` (5–95%: ${vf(s.lo[i])} to ${vf(s.hi[i])})` : "") + (hollow ? " · ineligible" : ""), color: s.color }])));
      shape.addEventListener("pointerleave", hideTip);
    });
  });
  function c(i) { return o.categories[i]; }
  return { svg };
}

// ---------- a plain table from data: columns [{label, sub}], rows [{cells: [{text, small, strong, muted, note, good, bad, tag}], cls, detail}]
export function tableChart(container, o) {
  container.replaceChildren();
  const wrap = h("div", { class: "x-tablewrap" }), t = h("table", { class: "x-datatable" });
  t.append(h("thead", {}, h("tr", {}, ...o.columns.map(c => h("th", {}, c.label, c.sub ? h("small", { text: c.sub }) : null)))));
  const tb = h("tbody");
  for (const r of o.rows) {
    const tr = h("tr", { class: r.cls || null, title: r.source || null });
    r.cells.forEach((c, k) => {
      const td = h(k === 0 ? "th" : "td", { scope: k === 0 ? "row" : null, class: [c.muted ? "muted" : "", c.good ? "good" : "", c.bad ? "bad" : "", c.tag ? "tag" : ""].filter(Boolean).join(" ") || null });
      td.append(c.strong ? h("b", { text: c.text }) : document.createTextNode(c.text));
      if (c.small) td.append(h("small", { text: c.small }));
      if (c.note) td.append(h("small", { class: "warn", text: c.note }));
      if (k === 0 && r.detail && r.detail.length) td.append(h("details", { class: "x-cell-more" }, h("summary", { text: `${r.detail.length} check${r.detail.length === 1 ? "" : "s"}` }), h("code", { text: r.detail.join(" ") })));
      tr.append(td);
    });
    tb.append(tr);
  }
  t.append(tb);
  wrap.append(t);
  container.append(wrap);
  if (o.note) container.append(h("p", { class: "chart-sub x-table-note", text: o.note }));
  if (!o.rows.length) container.append(h("p", { class: "chart-sub", text: "No rows." }));
  return { table: t };
}
