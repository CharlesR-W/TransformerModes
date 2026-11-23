/* ---------- global state, selection, URL hash ---------- */
const S = {
  tab: 'layer', view: 'cls', k: null, l: 9, t: 1, r: 16,
  sel: null,        // primary object (colours the UMAP, drives components and detail)
  img: null,        // focused image (independent of sel)
  color: 'sel',     // UMAP colour: sel | state | super
  cblock: null,     // block shown in the neuron grid (null = follow the object's layer)
  cmetric: 'auto',  // components metric
  a: 1, b: 12,      // block view pair
  dsign: 'pos',     // block view triangle shows top + or top - eigenvalue
};
const HIST = [];
let GEN = 0;

function selKey(s) {
  if (!s) return '';
  switch (s.kind) {
    case 'mode': return `mode:${s.view}:${s.l}:${s.i}`;
    case 'state': return `state:${s.view}:${s.l}:${s.c}`;
    case 'neuron': return `neuron:${s.l}:${s.n}`;
    case 'head': return `head:${s.l}:${s.h}`;
    case 'dmode': return `dmode:${s.view}:${s.a}:${s.b}:${s.j}`;
    case 'class': return `class:${s.c}`;
    case 'super': return `super:${s.s}`;
  }
  return '';
}
function parseSel(x) {
  if (!x) return null; const p = x.split(':'); const n = i => +p[i];
  switch (p[0]) {
    case 'mode': return {kind: 'mode', view: p[1], l: n(2), i: n(3)};
    case 'state': return {kind: 'state', view: p[1], l: n(2), c: n(3)};
    case 'neuron': return {kind: 'neuron', l: n(1), n: n(2)};
    case 'head': return {kind: 'head', l: n(1), h: n(2)};
    case 'dmode': return {kind: 'dmode', view: p[1], a: n(2), b: n(3), j: n(4)};
    case 'class': return {kind: 'class', c: n(1)};
    case 'super': return {kind: 'super', s: n(1)};
  }
  return null;
}
function writeHash() {
  const q = new URLSearchParams({tab: S.tab, v: S.view, k: S.k, l: S.l, t: S.t, r: S.r, a: S.a, b: S.b});
  if (S.sel) q.set('sel', selKey(S.sel));
  if (S.img != null) q.set('img', S.img);
  if (S.color !== 'sel') q.set('c', S.color);
  try { history.replaceState(null, '', '#' + q.toString()); } catch (e) {}
}
function readHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (q.get('tab')) S.tab = q.get('tab');
  if (q.get('v') && IDX.views[q.get('v')]) S.view = q.get('v');
  if (q.get('k') && kernelsFor(S.view).includes(q.get('k'))) S.k = q.get('k');
  for (const f of ['l', 't', 'r', 'a', 'b']) if (q.get(f) != null && !Number.isNaN(+q.get(f))) S[f] = +q.get(f);
  if (q.get('sel')) S.sel = parseSel(q.get('sel'));
  if (q.get('img')) S.img = +q.get('img');
  if (q.get('c')) S.color = q.get('c');
}
function kernelsFor(view) { return Object.keys((IDX.views[view] || {}).kernels || {}).length ? Object.keys(IDX.views[view].kernels)
  : (META.kernels || []).map(k => k.id); }

let rafPending = false;
function invalidate() { if (rafPending) return; rafPending = true; requestAnimationFrame(() => { rafPending = false; GEN++; renderAll(); }); }
function setS(patch) {
  Object.assign(S, patch);
  const L = DB.layers(S.view); if (!L.includes(S.l)) S.l = L.includes(9) ? 9 : L[L.length - 1];
  if (!RS().includes(S.r)) S.r = RS()[1] || RS()[0];
  if (S.sel && S.sel.kind === 'state' && S.sel.c >= S.r) S.sel = null;
  writeHash(); invalidate();
}
function select(sel, opts = {}) {
  if (selKey(sel) === selKey(S.sel)) return;
  if (S.sel) { HIST.push(S.sel); if (HIST.length > 40) HIST.shift(); }
  S.sel = sel;
  if (!opts.keepBlock) S.cblock = null;
  S.cmetric = 'auto';
  writeHash(); invalidate();
}
function back() { if (!HIST.length) return; S.sel = HIST.pop(); S.cblock = null; S.cmetric = 'auto'; writeHash(); invalidate(); }
function focusImg(i) { S.img = (S.img === i ? null : i); writeHash(); invalidate(); }

/* Layer at which the selected object lives (for components, detail). */
function selLayer(s) {
  if (!s) return S.l;
  if (s.kind === 'dmode') return s.j < 8 ? s.b : s.a; // smooth endpoint
  return s.l != null ? s.l : S.l;
}

/* Human-readable name (HTML with KaTeX) of a selection. */
function selName(s, short) {
  if (!s) return 'nothing selected';
  const vw = s.view === 'mp' ? ' (mean-patch)' : '';
  switch (s.kind) {
    case 'mode': return T(`mode $\\psi_{${s.i + 1}}^{(${s.l})}$`) + (short ? '' : ` of layer ${s.l}${vw}`);
    case 'state': return T(`coarse state $${s.c}$`) + ` at layer ${s.l}${vw} (r = ${S.r})`;
    case 'neuron': return `MLP neuron ${s.n} of block ${s.l}`;
    case 'head': return `head ${s.h} of block ${s.l}`;
    case 'dmode': return T(`${s.j < 8 ? 'became smooth' : 'stopped being smooth'} $${s.j < 8 ? '+' : '-'}${(s.j % 8) + 1}$ of $\\Delta_{${S.t}}^{${s.a}\\to ${s.b}}$`) + (short ? '' : vw);
    case 'class': return `class ${s.c}: ${esc(className(s.c))}`;
    case 'super': return `superclass ${s.s}: ${esc(superName(s.s))}`;
  }
  return '?';
}
const className = c => ((META.classes || [])[c] || {}).name || ('class ' + c);
const superName = s => ((META.superclasses || [])[s] || {}).name || ('superclass ' + s);

/* The selected object as data on the N sites.
   Returns {vals, mask, type: 'div'|'seq'|'set', lo, hi, label} or {missing: 'reason'}. */
async function selData(s) {
  if (!s) return null;
  try {
    if (s.kind === 'mode') {
      if (!DB.hasModes(s.view, S.k, s.l)) return {missing: `no modes stored for ${kLabel(S.k)} at layer ${s.l} (${s.view})`};
      const M = await DB.modes(s.view, S.k, s.l), vals = M.subarray(s.i * N, (s.i + 1) * N);
      const m = quantile(vals.map(Math.abs), 0.995);
      return {vals, type: 'div', lo: -m, hi: m, label: T(`$\\sqrt N\\,\\psi_{${s.i + 1}}^{(${s.l})}(x)$`)};
    }
    if (s.kind === 'state') {
      if (!DB.hasSeba(s.view, S.k, s.l)) return {missing: `no SEBA memberships for ${kLabel(S.k)} at layer ${s.l}`};
      const ri = RS().indexOf(S.r), row = ROFF()[ri] + s.c;
      const [mem, hard] = await Promise.all([DB.seba(s.view, S.k, s.l), DB.hard(s.view, S.k, s.l)]);
      const vals = new Float32Array(N), mask = new Uint8Array(N);
      for (let i = 0; i < N; i++) { vals[i] = mem[row * N + i] / 255; mask[i] = hard && hard[ri * N + i] === s.c ? 1 : 0; }
      return {vals, mask, type: 'seq', lo: 0, hi: 1, label: T(`membership $m_{${s.c}}(x)$ (scaled to max 1)`)};
    }
    if (s.kind === 'neuron') {
      if (!DB.hasNeurons(s.l)) return {missing: 'no neuron activations for block ' + s.l};
      const vals = await DB.neuron(s.l, s.n);
      let lo = Infinity; for (const v of vals) lo = Math.min(lo, v);
      return {vals, type: 'seq', lo, hi: quantile(vals, 0.995), label: T(`$g_{${s.l},${s.n}}(x)$`)};
    }
    if (s.kind === 'head') {
      const Hh = await DB.heads(s.l); if (!Hh) return {missing: 'no head norms for block ' + s.l};
      const vals = Hh.subarray(s.h * N, (s.h + 1) * N);
      return {vals, type: 'seq', lo: quantile(vals, 0.005), hi: quantile(vals, 0.995), label: T(`$\\|A_{${s.l},${s.h}}(x)\\|_2$`)};
    }
    if (s.kind === 'dmode') {
      if (!DB.hasDelta(s.view, S.k, s.a, s.b)) return {missing: `Δ not stored for ${kLabel(S.k)}, pair ${s.a}→${s.b}`};
      const V = await DB.deltaVec(s.view, S.k, s.a, s.b), ti = TS().indexOf(S.t), off = (ti * 16 + s.j) * N;
      const vals = V.subarray(off, off + N), m = quantile(vals.map(Math.abs), 0.995);
      return {vals, type: 'div', lo: -m, hi: m, label: T(`$\\sqrt N f(x)$`)};
    }
    if (s.kind === 'class' || s.kind === 'super') {
      const lab = await DB.labels(), arr = s.kind === 'class' ? lab.cls : lab.sup, v = s.kind === 'class' ? s.c : s.s;
      const mask = new Uint8Array(N); for (let i = 0; i < N; i++) mask[i] = arr && arr[i] === v ? 1 : 0;
      return {mask, type: 'set', label: s.kind === 'class' ? 'images of this class (label: validation/display only)' : 'images of this superclass (label: validation/display only)'};
    }
  } catch (e) { console.error(e); return {missing: String(e.message || e)}; }
  return null;
}

/* Diffusion time t, judged from the data per kernel with lmax = largest nonconstant eigenvalue of P over all layers:
   lmax^(2t) < 1e-6 -> 'degenerate' (R^t ~ 0 on nonconstant functions; values ~0, eigenvectors numerical noise);
   lmax^(2t) < 0.01 -> 'attenuated' (strongly attenuated: values small; check the cross-validated values). */
function lamMax(view, k) {
  const pl = ((((IDX.selection || {}).views || {})[view] || {})[k] || {}).per_layer || [];
  let m = 0; for (const p of pl) { const l = p && p.spectrum && p.spectrum.lambda; if (l && l.length > 1) m = Math.max(m, Math.abs(l[1])); }
  return m || null;
}
function tState(t, view = S.view, k = S.k) {
  const m = lamMax(view, k); if (m == null) return null; const v = Math.pow(m, 2 * t);
  return v < 1e-6 ? {kind: 'degenerate', v, m} : v < 0.01 ? {kind: 'attenuated', v, m} : null;
}
