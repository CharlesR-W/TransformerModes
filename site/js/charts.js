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
  if (o.yrange) [ylo, yhi] = o.yrange;
  else {
    const all = o.series.flatMap(s => [...s.y, ...(s.band ? [...s.band[0], ...s.band[1]] : [])]).concat(o.band ? [...o.band[0], ...o.band[1]] : []).filter(Number.isFinite);
    ylo = Math.min(...all); yhi = Math.max(...all);
    const pad = (yhi - ylo) * .06 || .05; ylo -= pad; yhi += pad;
  }
  const Y = scale("linear", ylo, yhi, H - m.b, m.t);
  const yfmt = o.yfmt || tickFmt(o.fmt), xfmt = o.xfmt || (v => String(v));
  yAxis(svg, m, W, H, Y, ylo, yhi, yfmt, yRot ? o.ytitle : null);
  const ax = el("g", { class: "axis" }, svg);
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b + .5, y2: H - m.b + .5 }, ax);
  const xt = (o.xticks || niceTicks(x0, x1, Math.max(3, Math.round(W / 90)))).filter(v => v >= x0 - 1e-9 && v <= x1 + 1e-9);
  for (const v of xt) { el("line", { x1: X(v), x2: X(v), y1: H - m.b, y2: H - m.b + 4 }, ax); el("text", { x: X(v), y: H - m.b + 15, "text-anchor": "middle" }, svg).textContent = xfmt(v); }
  if (o.xlabel) el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "ttl" }, svg).textContent = o.xlabel;
  if (o.shade) el("rect", { x: X(o.shade[0]), y: m.t, width: Math.max(2, X(o.shade[1]) - X(o.shade[0])), height: H - m.b - m.t, class: "shade" }, svg);
  if (o.refline) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(o.refline.y), y2: Y(o.refline.y), class: "ref" }, svg);
    if (o.refline.label) el("text", { x: m.l + 6, y: Y(o.refline.y) - 5, "text-anchor": "start" }, svg).textContent = o.refline.label;
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
    const dr = s.dots === true ? 3 : +s.dots || 0;
    if (dr) xs.forEach((xv, i) => { if (Number.isFinite(s.y[i])) el("circle", { cx: X(xv), cy: Y(s.y[i]), r: dr, fill: s.color, stroke: "#fff", "stroke-width": 1.5 }, gg); });
    return gg;
  });
  if (o.cursor != null) el("line", { x1: X(o.cursor), x2: X(o.cursor), y1: m.t, y2: H - m.b, class: "cursor" }, svg);
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: INK(), "stroke-width": 1, opacity: 0 }, svg);
  const dots = o.series.map(s => el("circle", { r: 4, fill: s.color, stroke: "#fff", "stroke-width": 2, opacity: 0 }, svg));
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
      const st = s.hollow ? { fill: css("--surface", "#fff"), stroke: s.color, "stroke-width": 1.6 } : { fill: s.color, stroke: "#fff", "stroke-width": 1.2 };
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
