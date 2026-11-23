'use strict';
/* ---------- DOM + math helpers ---------- */
const $ = (s, el = document) => el.querySelector(s);
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const TEXCACHE = new Map();
function tex(s, display) {
  const key = (display ? 'D' : 'I') + s;
  if (TEXCACHE.has(key)) return TEXCACHE.get(key);
  let out;
  try { out = katex.renderToString(s, {displayMode: !!display, throwOnError: false, strict: 'ignore',
    macros: {'\\R': '\\mathbb{R}', '\\one': '\\mathbf{1}'}}); }
  catch (e) { out = '<code>' + esc(s) + '</code>'; }
  TEXCACHE.set(key, out);
  return out;
}
/* HTML string in which $...$ (inline) and $$...$$ (display) are rendered with KaTeX. */
function T(s) {
  return String(s).replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => tex(m, true))
                  .replace(/\$([^$]+?)\$/g, (_, m) => tex(m, false));
}
function H(s) { const d = document.createElement('span'); d.innerHTML = T(s); return d; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
function fmt(v, d) {
  if (v == null || Number.isNaN(v)) return '–';
  if (d != null) return Number(v).toFixed(d);
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(3);
}
const fmtInt = v => Number(v).toLocaleString('en-US');
const pad2 = n => String(n).padStart(2, '0');
function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }

/* ---------- float16 decoding (lookup table) ---------- */
const F16LUT = (() => {
  const t = new Float32Array(65536);
  for (let x = 0; x < 65536; x++) {
    const s = x >> 15, e = (x >> 10) & 31, f = x & 1023;
    let v = e === 0 ? f * 5.960464477539063e-8 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * Math.pow(2, e - 15);
    t[x] = s ? -v : v;
  }
  return t;
})();
function decodeF16(buf) {
  const u = new Uint16Array(buf), out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) out[i] = F16LUT[u[i]];
  return out;
}

/* ---------- colour ---------- */
function hex2rgb(x) { const n = parseInt(x.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
function makeLUT(stops) { // stops: [[pos 0..1, hex]...] -> Uint8Array 256*3
  const lut = new Uint8Array(256 * 3), S = stops.map(([p, c]) => [p, hex2rgb(c)]);
  for (let i = 0; i < 256; i++) {
    const x = i / 255; let j = 0;
    while (j < S.length - 2 && x > S[j + 1][0]) j++;
    const [p0, c0] = S[j], [p1, c1] = S[j + 1], u = clamp((x - p0) / (p1 - p0 || 1), 0, 1);
    for (let k = 0; k < 3; k++) lut[3 * i + k] = Math.round(c0[k] + u * (c1[k] - c0[k]));
  }
  return lut;
}
const DIV_STOPS = [[0, '#0d366b'], [0.2, '#2a78d6'], [0.38, '#86b6ef'], [0.5, '#e4e1da'], [0.62, '#f2a49d'], [0.8, '#e34948'], [1, '#8f1d1c']];
const SEQ_STOPS = [[0, '#e4e1da'], [0.12, '#cde2fb'], [0.4, '#6da7ec'], [0.7, '#256abf'], [1, '#0d366b']];
const LUT_DIV = makeLUT(DIV_STOPS), LUT_SEQ = makeLUT(SEQ_STOPS);
const gradCSS = stops => 'linear-gradient(90deg,' + stops.map(([p, c]) => c + ' ' + (p * 100) + '%').join(',') + ')';
const GRAD_DIV = gradCSS(DIV_STOPS), GRAD_SEQ = gradCSS(SEQ_STOPS);
const CAT8 = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const NEUTRAL = [217, 214, 207];      // points outside a highlighted set
const HL = '#eb6834';                  // highlight (class / image set)
function lutColor(lut, u) { const i = clamp(Math.round(u * 255), 0, 255) * 3; return 'rgb(' + lut[i] + ',' + lut[i + 1] + ',' + lut[i + 2] + ')'; }
const divColor = (v, m) => lutColor(LUT_DIV, 0.5 + 0.5 * clamp(v / (m || 1), -1, 1));
const seqColor = (v, lo, hi) => lutColor(LUT_SEQ, (v - lo) / ((hi - lo) || 1));

/* ---------- tooltip ---------- */
const TIP = {
  el: null,
  show(html, x, y) {
    const el = this.el || (this.el = document.getElementById('tip'));
    el.innerHTML = html; el.hidden = false;
    const r = el.getBoundingClientRect();
    let px = x + 14, py = y + 14;
    if (px + r.width > innerWidth - 6) px = x - r.width - 14;
    if (py + r.height > innerHeight - 6) py = y - r.height - 14;
    el.style.left = Math.max(4, px) + 'px'; el.style.top = Math.max(4, py) + 'px';
  },
  hide() { if (this.el) this.el.hidden = true; },
};

/* ---------- thumbnails (SCHEMA: sheet i//100, 10x10 tiles of 64 px) ---------- */
function thumbStyle(i, size) {
  const s = size / 64, sheet = Math.floor(i / 100), j = i % 100, r = Math.floor(j / 10), c = j % 10;
  return `background-image:url(d/thumbs/sheet_${String(sheet).padStart(3, '0')}.jpg);background-size:${640 * s}px ${640 * s}px;` +
    `background-position:${-c * 64 * s}px ${-r * 64 * s}px;width:${size}px;height:${size}px`;
}
function thumbHTML(i, size = 64, extra = '') {
  return `<div class="th" data-img="${i}" style="${thumbStyle(i, size)}">${extra}</div>`;
}

/* ---------- small stats ---------- */
function centredUnit(v) { // returns Float32Array centred, unit norm (or null if constant)
  const n = v.length; let m = 0; for (let i = 0; i < n; i++) m += v[i]; m /= n;
  const o = new Float32Array(n); let s = 0;
  for (let i = 0; i < n; i++) { const x = v[i] - m; o[i] = x; s += x * x; }
  s = Math.sqrt(s); if (!(s > 0)) return null;
  for (let i = 0; i < n; i++) o[i] /= s;
  return o;
}
function topk(v, k, desc = true, mask) {
  const idx = [];
  for (let i = 0; i < v.length; i++) if (!mask || mask[i]) idx.push(i);
  idx.sort((a, b) => desc ? v[b] - v[a] : v[a] - v[b]);
  return idx.slice(0, k);
}
function seededSample(arr, k, seed = 1) { // deterministic random sample
  const a = arr.slice(); let s = seed * 2654435761 >>> 0;
  for (let i = a.length - 1; i > 0; i--) { s = (s * 1664525 + 1013904223) >>> 0; const j = s % (i + 1); [a[i], a[j]] = [a[j], a[i]]; }
  return a.slice(0, k);
}
function quantile(arr, q) { const a = Float32Array.from(arr).filter(x => !Number.isNaN(x)).sort(); return a.length ? a[Math.min(a.length - 1, Math.floor(q * a.length))] : NaN; }
