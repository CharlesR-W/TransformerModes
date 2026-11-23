/* ---------- UMAP scatter: 10k points on a canvas, colour-binned batched drawing ---------- */
class Scatter {
  constructor(wrap, opts = {}) {
    this.wrap = wrap; this.opts = opts;
    this.cv = h('canvas'); wrap.append(this.cv);
    this.ctx = this.cv.getContext('2d');
    this.xy = null; this.from = null; this.anim = 0;
    this.bounds = null; this.zoom = {s: 1, x: 0, y: 0};
    this.col = null;        // Uint8Array N*3 colours
    this.order = null;      // Uint32Array draw order
    this.bins = null;       // array of {css, idx[]} in draw order
    this.labels = [];       // [{x,y,text}] in data coords
    this.focus = null; this.ringSet = null;
    this.r = opts.r || 2.1;
    this.grid = null;
    const btns = h('div', {class: 'umapbtns'}, h('button', {title: 'reset zoom (or double-click)', onclick: () => this.resetZoom()}, 'reset zoom'));
    wrap.append(btns);
    new ResizeObserver(() => this.resize()).observe(wrap);
    this.cv.addEventListener('mousemove', e => this.onMove(e));
    this.cv.addEventListener('mouseleave', () => { TIP.hide(); this.hover = null; this.opts.onHover && this.opts.onHover(null); });
    this.cv.addEventListener('wheel', e => this.onWheel(e), {passive: false});
    this.cv.addEventListener('mousedown', e => this.onDown(e));
    this.cv.addEventListener('dblclick', () => this.resetZoom());
    this.resize();
  }
  resize() {
    const r = this.wrap.getBoundingClientRect(), d = devicePixelRatio || 1;
    if (!r.width) return;
    this.W = r.width; this.H = r.height; this.cv.width = Math.round(r.width * d); this.cv.height = Math.round(r.height * d);
    this.ctx.setTransform(d, 0, 0, d, 0, 0); this.draw();
  }
  setBounds(b) { this.bounds = b; }
  setPoints(xy, animate = true) {
    if (this.xy === xy) return;
    if (this.xy && animate && this.xy.length === xy.length) {
      this.from = this.cur ? Float32Array.from(this.cur) : this.xy; this.xy = xy; this.t0 = performance.now();
      const step = () => {
        const u = Math.min(1, (performance.now() - this.t0) / 520), e = u < .5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
        const c = this.cur || (this.cur = new Float32Array(xy.length));
        for (let i = 0; i < xy.length; i++) c[i] = this.from[i] + (xy[i] - this.from[i]) * e;
        this.draw(c, u < 1);
        if (u < 1) this.anim = requestAnimationFrame(step); else { this.cur = null; this.draw(); }
      };
      cancelAnimationFrame(this.anim); this.anim = requestAnimationFrame(step);
    } else { this.xy = xy; this.cur = null; this.draw(); }
  }
  /* values + type -> colour bins. type 'div' (lo=-m..hi=m), 'seq', 'set' (mask), 'cat' (int labels, -1 none). */
  setColor(spec) {
    const n = N, bins = new Map(); const add = (key, css, i, z) => { let b = bins.get(key); if (!b) { b = {css, z, idx: []}; bins.set(key, b); } b.idx.push(i); };
    const neut = `rgb(${NEUTRAL})`;
    if (!spec || spec.type === 'none') {
      for (let i = 0; i < n; i++) add('n', 'rgb(150,160,172)', i, 0);
    } else if (spec.type === 'div' || spec.type === 'seq') {
      const lut = spec.type === 'div' ? LUT_DIV : LUT_SEQ, lo = spec.lo, hi = spec.hi, v = spec.vals, Q = 48;
      for (let i = 0; i < n; i++) {
        let u = (v[i] - lo) / ((hi - lo) || 1); if (Number.isNaN(u)) { add('nan', neut, i, -1); continue; }
        u = clamp(u, 0, 1); const q = Math.round(u * Q), li = Math.round(q / Q * 255) * 3;
        const z = spec.type === 'div' ? Math.abs(q - Q / 2) : q;
        const inset = spec.mask ? spec.mask[i] : 1;
        add(q + (inset ? '' : 'o'), `rgb(${lut[li]},${lut[li + 1]},${lut[li + 2]})`, i, z + (inset ? 1000 : 0));
      }
    } else if (spec.type === 'set') {
      for (let i = 0; i < n; i++) spec.mask[i] ? add('in', spec.color || HL, i, 1) : add('out', neut, i, 0);
    } else if (spec.type === 'cat') {
      for (let i = 0; i < n; i++) { const c = spec.vals[i]; c < 0 || c >= spec.colors.length ? add('u', '#ebe8e2', i, -1) : add('c' + c, spec.colors[c], i, 1 + (spec.mask && spec.mask[i] ? 10 : 0)); }
    }
    this.bins = [...bins.values()].sort((a, b) => a.z - b.z);
    this.draw();
  }
  setLabels(lbls) { this.labels = lbls || []; this.draw(); }
  setFocus(i, ring) { this.focus = i; this.ringSet = ring || null; this.draw(); }
  tx() { // data -> screen
    const b = this.bounds, pad = 14, W = this.W, H = this.H;
    const sx = (W - 2 * pad) / (b[1] - b[0]), sy = (H - 2 * pad) / (b[3] - b[2]), s = Math.min(sx, sy);
    const ox = pad + (W - 2 * pad - s * (b[1] - b[0])) / 2, oy = pad + (H - 2 * pad - s * (b[3] - b[2])) / 2;
    const z = this.zoom;
    return [s * z.s, ox * z.s + z.x - b[0] * s * z.s, oy * z.s + z.y + b[3] * s * z.s]; // y flipped
  }
  draw(xy, moving) {
    xy = xy || this.cur || this.xy;
    const c = this.ctx; if (!this.W) return;
    c.clearRect(0, 0, this.W, this.H);
    if (!xy || !this.bounds) { c.fillStyle = '#898781'; c.font = '13px system-ui'; c.fillText(this.msg || 'loading…', 16, 24); return; }
    const [s, ox, oy] = this.tx(), r = this.r * Math.min(2.2, Math.pow(this.zoom.s, 0.35)), TAU = Math.PI * 2;
    const bins = this.bins || [{css: 'rgb(150,160,172)', idx: Array.from({length: N}, (_, i) => i)}];
    for (const b of bins) {
      c.fillStyle = b.css; c.beginPath();
      for (const i of b.idx) { const X = ox + s * xy[2 * i], Y = oy - s * xy[2 * i + 1]; c.moveTo(X + r, Y); c.arc(X, Y, r, 0, TAU); }
      c.fill();
    }
    if (this.ringSet) { c.strokeStyle = '#0b0b0b'; c.lineWidth = 1; c.beginPath();
      for (const i of this.ringSet) { const X = ox + s * xy[2 * i], Y = oy - s * xy[2 * i + 1]; c.moveTo(X + r + 2, Y); c.arc(X, Y, r + 2, 0, TAU); } c.stroke(); }
    if (this.focus != null) {
      const X = ox + s * xy[2 * this.focus], Y = oy - s * xy[2 * this.focus + 1];
      c.lineWidth = 3; c.strokeStyle = 'white'; c.beginPath(); c.arc(X, Y, 7, 0, TAU); c.stroke();
      c.lineWidth = 1.6; c.strokeStyle = '#0b0b0b'; c.beginPath(); c.arc(X, Y, 7, 0, TAU); c.stroke();
    }
    if (!moving && this.labels.length) {
      c.font = '600 11px system-ui'; c.textAlign = 'center'; c.textBaseline = 'middle';
      for (const L of this.labels) { const X = ox + s * L.x, Y = oy - s * L.y; c.lineWidth = 3; c.strokeStyle = 'rgba(255,255,255,.9)'; c.strokeText(L.text, X, Y); c.fillStyle = L.color || '#2b2a27'; c.fillText(L.text, X, Y); }
    }
    if (!moving) this.buildGrid(xy, s, ox, oy);
  }
  buildGrid(xy, s, ox, oy) {
    const G = 12, cols = Math.ceil(this.W / G) + 1, rows = Math.ceil(this.H / G) + 1, cells = new Map();
    for (let i = 0; i < N; i++) {
      const X = ox + s * xy[2 * i], Y = oy - s * xy[2 * i + 1];
      if (X < -G || Y < -G || X > this.W + G || Y > this.H + G) continue;
      const k = Math.floor(Y / G) * cols + Math.floor(X / G); let a = cells.get(k); if (!a) cells.set(k, a = []); a.push(i);
    }
    this.grid = {G, cols, cells, s, ox, oy, xy};
  }
  pick(mx, my) {
    const g = this.grid; if (!g) return null; let best = null, bd = 100;
    const cx = Math.floor(mx / g.G), cy = Math.floor(my / g.G);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const a = g.cells.get((cy + dy) * g.cols + cx + dx); if (!a) continue;
      for (const i of a) { const X = g.ox + g.s * g.xy[2 * i], Y = g.oy - g.s * g.xy[2 * i + 1], d = (X - mx) ** 2 + (Y - my) ** 2; if (d < bd) { bd = d; best = i; } }
    }
    return best;
  }
  pos(e) { const r = this.cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
  onMove(e) {
    if (this.drag) return;
    const [mx, my] = this.pos(e), i = this.pick(mx, my);
    this.hover = i;
    if (i == null) { TIP.hide(); this.opts.onHover && this.opts.onHover(null); return; }
    this.opts.onHover && this.opts.onHover(i);
    TIP.show(this.opts.tip ? this.opts.tip(i) : 'image ' + i, e.clientX, e.clientY);
  }
  onWheel(e) {
    e.preventDefault(); const [mx, my] = this.pos(e), f = Math.exp(-e.deltaY * 0.0015), z = this.zoom;
    const ns = clamp(z.s * f, 1, 40), k = ns / z.s; z.x = mx - (mx - z.x) * k; z.y = my - (my - z.y) * k; z.s = ns;
    if (ns === 1) { z.x = 0; z.y = 0; }
    this.draw();
  }
  onDown(e) {
    const [x0, y0] = this.pos(e), z0 = {...this.zoom}; let moved = false;
    const mv = ev => { const [x, y] = this.pos(ev); if (Math.abs(x - x0) + Math.abs(y - y0) > 3) moved = true;
      if (moved && this.zoom.s > 1) { this.drag = true; this.zoom.x = z0.x + x - x0; this.zoom.y = z0.y + y - y0; this.draw(); TIP.hide(); } };
    const up = ev => { removeEventListener('mousemove', mv); removeEventListener('mouseup', up); this.drag = false;
      if (!moved) { const [x, y] = this.pos(ev), i = this.pick(x, y); if (i != null && this.opts.onClick) this.opts.onClick(i, ev); } };
    addEventListener('mousemove', mv); addEventListener('mouseup', up);
  }
  resetZoom() { this.zoom = {s: 1, x: 0, y: 0}; this.draw(); }
}

const UMAPB = {};   // view -> global bounds over all layers (aligned UMAP: fixed frame)
async function umapLayer(view, l) {
  const U = await DB.umap(view); if (!U) return null;
  const li = DB.li(view, l);
  if (!UMAPB[view]) { let a = Infinity, b = -Infinity, c = Infinity, d = -Infinity;
    for (let i = 0; i < U.length; i += 2) { const x = U[i], y = U[i + 1]; if (x < a) a = x; if (x > b) b = x; if (y < c) c = y; if (y > d) d = y; }
    UMAPB[view] = [a, b, c, d]; }
  return U.subarray(li * N * 2, (li + 1) * N * 2);
}
/* colour spec from selection data */
function specFrom(sd) {
  if (!sd || sd.missing) return {type: 'none'};
  if (sd.type === 'set') return {type: 'set', mask: sd.mask};
  return {type: sd.type, vals: sd.vals, lo: sd.lo, hi: sd.hi, mask: sd.type === 'seq' && sd.mask ? sd.mask : null};
}
function legendHTML(sd, extra = '') {
  if (!sd) return `<span class="note">grey: nothing selected. Click a mode, coarse state, neuron, head or image.</span>`;
  if (sd.missing) return `<span class="err">${esc(sd.missing)}</span>`;
  if (sd.type === 'set') return `<span><span class="sw" style="background:${HL}"></span>${sd.label}</span>${extra}`;
  const g = sd.type === 'div' ? GRAD_DIV : GRAD_SEQ;
  return `<span>${sd.label}</span><span class="num">${fmt(sd.lo)}</span><span class="ramp" style="background:${g}"></span><span class="num">${fmt(sd.hi)}</span>` +
    `<span class="note">${sd.type === 'div' ? 'colour range ±99.5th pct of |value|, clipped' : '0.5–99.5th pct, clipped'}</span>${extra}`;
}
let LAB = null; // {cls, sup}, loaded at start
function imgTip(i, sd) {
  const c = LAB.cls[i], s = LAB.sup ? LAB.sup[i] : null;
  let v = '';
  if (sd && sd.vals) v = `<div><span class="k">value</span> <b>${fmt(sd.vals[i])}</b></div>`;
  if (sd && sd.mask) v += `<div><span class="k">in set</span> <b>${sd.mask[i] ? 'yes' : 'no'}</b></div>`;
  return `<div class="row">${thumbHTML(i, 96)}<div><div><b>image ${i}</b></div><div>${esc(className(c))}</div>` +
    (s != null ? `<div class="k">${esc(superName(s))}</div>` : '') + v + `<div class="k" style="margin-top:3px">click: inspect this image</div></div></div>`;
}
