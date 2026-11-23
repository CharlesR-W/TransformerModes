/* ---------- data layer: lazy, cached reads of the SCHEMA.md files under d/ ---------- */
let IDX = null, META = null, N = 0;
const DB = {
  cache: new Map(), lru: [],
  has(rel) { return IDX && Object.prototype.hasOwnProperty.call(IDX.have, rel); },
  async raw(rel) {
    if (this.cache.has(rel)) { const p = this.cache.get(rel); this._touch(rel); return p; }
    const p = fetch('d/' + rel).then(r => { if (!r.ok) throw new Error(rel + ': HTTP ' + r.status); return r.arrayBuffer(); });
    this.cache.set(rel, p); this._touch(rel);
    p.catch(() => this.cache.delete(rel));
    return p;
  },
  _touch(rel) { // keep at most ~70 large binaries in memory
    const i = this.lru.indexOf(rel); if (i >= 0) this.lru.splice(i, 1); this.lru.push(rel);
    while (this.lru.length > 70) { const old = this.lru.shift(); this.cache.delete(old); this.dec.delete(old); }
  },
  dec: new Map(),
  async typed(rel, kind) { // kind: f32 f16 u8 i8 u16 i16
    if (this.dec.has(rel)) { this._touch(rel); return this.dec.get(rel); }
    const buf = await this.raw(rel);
    const out = kind === 'f32' ? new Float32Array(buf) : kind === 'f16' ? decodeF16(buf) : kind === 'u8' ? new Uint8Array(buf)
      : kind === 'i8' ? new Int8Array(buf) : kind === 'u16' ? new Uint16Array(buf) : kind === 'i16' ? new Int16Array(buf) : null;
    this.dec.set(rel, out);
    return out;
  },
  async json(rel) {
    if (this.cache.has('J' + rel)) return this.cache.get('J' + rel);
    const p = fetch('d/' + rel).then(r => { if (!r.ok) throw new Error(rel + ': HTTP ' + r.status); return r.json(); });
    this.cache.set('J' + rel, p); p.catch(() => this.cache.delete('J' + rel));
    return p;
  },
  // ---- typed accessors ----
  layers(view) { return (META.views && META.views[view] && META.views[view].layers) || (view === 'cls' ? [1,2,3,4,5,6,7,8,9,10,11,12] : [0,1,2,3,4,5,6,7,8,9,10,11,12]); },
  li(view, l) { return this.layers(view).indexOf(l); },          // array axis index of depth l
  sfx(l) { return 'L' + pad2(l); },
  async umap(view) { return this.has(`${view}/umap.f32`) ? this.typed(`${view}/umap.f32`, 'f32') : null; },
  hasModes(view, k, l) { return this.has(`${view}/${k}/modes/${this.sfx(l)}.f16`); },
  async modes(view, k, l) { return this.typed(`${view}/${k}/modes/${this.sfx(l)}.f16`, 'f16'); }, // [64,N] x100
  async evals(view, k) { return this.has(`${view}/${k}/evals.f32`) ? this.typed(`${view}/${k}/evals.f32`, 'f32') : null; },
  async spectrum(view, k) { return this.has(`${view}/${k}/spectrum.f32`) ? this.typed(`${view}/${k}/spectrum.f32`, 'f32') : null; },
  async smooth(view, k) { return this.has(`${view}/${k}/smooth.f32`) ? this.typed(`${view}/${k}/smooth.f32`, 'f32') : null; },
  hasSeba(view, k, l) { return this.has(`${view}/${k}/seba/${this.sfx(l)}.u8`); },
  async seba(view, k, l) { return this.typed(`${view}/${k}/seba/${this.sfx(l)}.u8`, 'u8'); },       // [120,N]
  async hard(view, k, l) { return this.has(`${view}/${k}/seba/${this.sfx(l)}_hard.i8`) ? this.typed(`${view}/${k}/seba/${this.sfx(l)}_hard.i8`, 'i8') : null; }, // [4,N]
  async labels() {
    const [c, s] = await Promise.all([this.typed('images/class.u8', 'u8'),
      this.has('images/superclass.u8') ? this.typed('images/superclass.u8', 'u8') : null]);
    return {cls: c, sup: s};
  },
  async half() { return this.has('images/half.u8') ? this.typed('images/half.u8', 'u8') : null; },
  async source() { return this.has('images/source.u8') ? this.typed('images/source.u8', 'u8') : null; },
  hasNeurons(l) { return this.has(`cls/neurons/${this.sfx(l)}.u8`); },
  async neuronRange() { return this.has('cls/neurons/range.f32') ? this.typed('cls/neurons/range.f32', 'f32') : null; },
  async neuronStats() { return this.has('cls/neurons/stats.f32') ? this.typed('cls/neurons/stats.f32', 'f32') : null; },
  async neuronChunk(l, c) { return this.typed(`cls/neurons_t/${this.sfx(l)}_c${pad2(c)}.u8`, 'u8'); },
  async neuron(l, n) { // dequantised activations g_{l,n}(x), Float32Array N
    const C = IDX.neuron_chunk, ch = await this.neuronChunk(l, Math.floor(n / C)), rg = await this.neuronRange();
    const off = (n % C) * N, ri = ((l - 1) * META_NN() + n) * 2, lo = rg ? rg[ri] : 0, hi = rg ? rg[ri + 1] : 1;
    const out = new Float32Array(N); const sc = (hi - lo) / 255;
    for (let i = 0; i < N; i++) out[i] = lo + ch[off + i] * sc;
    return out;
  },
  async heads(l) { return this.has(`cls/heads/${this.sfx(l)}.f16`) ? this.typed(`cls/heads/${this.sfx(l)}.f16`, 'f16') : null; }, // [6,N]
  async load(k, l) { return this.has(`cls/${k}/load/${this.sfx(l)}.f16`) ? this.typed(`cls/${k}/load/${this.sfx(l)}.f16`, 'f16') : null; },     // [64,1536]
  async auroc(k, l) { return this.has(`cls/${k}/auroc/${this.sfx(l)}.f16`) ? this.typed(`cls/${k}/auroc/${this.sfx(l)}.f16`, 'f16') : null; }, // [120,1536]
  async attr(k, l) { return this.has(`cls/${k}/attr/${this.sfx(l)}.f32`) ? this.typed(`cls/${k}/attr/${this.sfx(l)}.f32`, 'f32') : null; },
  async writecos(k, l) { return this.has(`cls/${k}/writecos/${this.sfx(l)}.f16`) ? this.typed(`cls/${k}/writecos/${this.sfx(l)}.f16`, 'f16') : null; },
  async weights() { return this.has('weights/wout_ls.f16') ? this.typed('weights/wout_ls.f16', 'f16') : null; },
  pairKey(a, b) { return 'a' + pad2(a) + '_b' + pad2(b); },
  hasDelta(view, k, a, b) { return this.has(`${view}/${k}/delta/${this.pairKey(a, b)}.f16`); },
  async deltaVec(view, k, a, b) { return this.typed(`${view}/${k}/delta/${this.pairKey(a, b)}.f16`, 'f16'); }, // [4,16,N]
  async deltaInfo(view, k, a, b) { return this.json(`${view}/${k}/delta/${this.pairKey(a, b)}.json`); },
  hasDeltaAttr(k, a, b) { return this.has(`cls/${k}/delta_attr/${this.pairKey(a, b)}.f32`); },
  async deltaAttr(k, a, b) { return this.typed(`cls/${k}/delta_attr/${this.pairKey(a, b)}.f32`, 'f32'); },        // [4,16,480]
  async deltaTop(k, a, b) { const rel = `cls/${k}/delta_attr/${this.pairKey(a, b)}_top.f32`; return this.has(rel) ? this.typed(rel, 'f32') : null; }, // [4,16,32,2]
  async flows(view, k, r) { const rel = `${view}/${k}/flows/counts_r${pad2(r)}.u16`; return this.has(rel) ? this.typed(rel, 'u16') : null; },
  async attnIdx() { return this.has('attn/idx.i16') ? this.typed('attn/idx.i16', 'i16') : null; },
  async attn(l) { return this.has(`attn/${this.sfx(l)}.u8`) ? this.typed(`attn/${this.sfx(l)}.u8`, 'u8') : null; },
  async attnMax() { return this.has('attn/max.f32') ? this.typed('attn/max.f32', 'f32') : null; },
};
const META_NN = () => 1536;
const NH = 6, NNEUR = 1536, NM = 64;
const RS = () => META.seba_r || [8, 16, 32, 64];
const ROFF = () => META.seba_offsets || [0, 8, 24, 56];
const TS = () => META.t || [1, 4, 16, 64];
function kInfo(k) { return (META.kernels || []).find(x => x.id === k) || {id: k}; }
function kFull(k) { return (META.kernels_full || []).includes(k); }
function kTex(k) {
  const ki = kInfo(k), p = ki.param || {};
  if ((ki.family || '').startsWith('self') || k.startsWith('st')) return `K^{\\mathrm{st}}(k{=}${p.k != null ? p.k : k.slice(2)})`;
  return `K^{\\mathrm{vmf}}(\\kappa{=}${p.kappa != null ? (+p.kappa).toFixed(p.kappa % 1 ? 2 : 0) : k.slice(3)})`;
}
function kLabel(k) {
  const ki = kInfo(k), p = ki.param || {};
  if (k.startsWith('st')) return `K-st k=${p.k != null ? p.k : k.slice(2)}`;
  return `K-vmf κ=${p.kappa != null ? (+p.kappa).toFixed(p.kappa % 1 ? 1 : 0) : k.slice(3)}`;
}

/* browser-side statistics over the 8-bit activations */
async function blockLoadings(l, f) { // cos(centred g_{l,n}, centred f) for all n of block l; f any function on sites
  const fu = centredUnit(f); if (!fu) return null;
  const st = await DB.neuronStats(), rg = await DB.neuronRange(), C = IDX.neuron_chunk;
  const out = new Float32Array(NNEUR);
  const chunks = await Promise.all(Array.from({length: NNEUR / C}, (_, c) => DB.neuronChunk(l, c)));
  for (let n = 0; n < NNEUR; n++) {
    const ch = chunks[Math.floor(n / C)], off = (n % C) * N;
    const ri = ((l - 1) * NNEUR + n) * 2, sc = (rg[ri + 1] - rg[ri]) / 255;
    let s = 0; for (let i = 0; i < N; i++) s += ch[off + i] * fu[i];
    const sd = st ? st[ri + 1] : NaN; // sum_i (lo + q sc - mean) fu_i = sc * sum q fu  (fu centred)
    out[n] = sd > 0 ? (sc * s) / (sd * Math.sqrt(N)) : NaN;
  }
  return out;
}
async function blockAUROC(l, mask) { // AUROC of g_{l,n} for a set, from 8-bit histograms (ties counted 1/2)
  let n1 = 0; for (let i = 0; i < N; i++) n1 += mask[i]; const n0 = N - n1;
  if (!n1 || !n0) return null;
  const C = IDX.neuron_chunk, out = new Float32Array(NNEUR);
  const chunks = await Promise.all(Array.from({length: NNEUR / C}, (_, c) => DB.neuronChunk(l, c)));
  const h1 = new Float64Array(256), h0 = new Float64Array(256);
  for (let n = 0; n < NNEUR; n++) {
    const ch = chunks[Math.floor(n / C)], off = (n % C) * N; h1.fill(0); h0.fill(0);
    for (let i = 0; i < N; i++) (mask[i] ? h1 : h0)[ch[off + i]]++;
    let below = 0, s = 0; for (let q = 0; q < 256; q++) { s += h1[q] * (below + 0.5 * h0[q]); below += h0[q]; }
    out[n] = s / (n1 * n0);
  }
  return out;
}
function vecCos(a, b) { const au = centredUnit(a), bu = centredUnit(b); if (!au || !bu) return NaN; let s = 0; for (let i = 0; i < au.length; i++) s += au[i] * bu[i]; return s; }
const ORDER_CACHE = new Map();
function rankOf(v, key) { // ascending ranks (1-based, ties broken by index), cached by key when given
  if (key && ORDER_CACHE.has(key)) return ORDER_CACHE.get(key);
  const idx = Array.from(v.keys()).sort((x, y) => v[x] - v[y]), rk = new Float64Array(v.length);
  for (let r = 0; r < idx.length; r++) rk[idx[r]] = r + 1;
  if (key) { ORDER_CACHE.set(key, rk); if (ORDER_CACHE.size > 200) ORDER_CACHE.delete(ORDER_CACHE.keys().next().value); }
  return rk;
}
function vecAUROC(v, mask, key) {
  const rk = rankOf(v, key); let n1 = 0, rs = 0;
  for (let i = 0; i < v.length; i++) if (mask[i]) { n1++; rs += rk[i]; }
  const n0 = v.length - n1; return n1 && n0 ? (rs - n1 * (n1 + 1) / 2) / (n1 * n0) : NaN;
}
function aurocPerState(v, lab, r) { // AUROC of v for each hard state 0..r-1 with one sort
  const rk = rankOf(v), n1 = new Float64Array(r), rs = new Float64Array(r);
  for (let i = 0; i < v.length; i++) { const c = lab[i]; if (c >= 0 && c < r) { n1[c]++; rs[c] += rk[i]; } }
  return Array.from({length: r}, (_, c) => { const a = n1[c], b = v.length - a; return a && b ? (rs[c] - a * (a + 1) / 2) / (a * b) : NaN; });
}

/* SCHEMA v2: Delta-mode attribution row (ti, j) -> {baseline, heads[72], mlpSum[12], w_res[384], r2p, r2h, cov, layer, top[[col, v]]} */
async function dmodeAttr(k, a, b, ti, j) {
  if (!DB.hasDeltaAttr(k, a, b)) return null;
  const A = await DB.deltaAttr(k, a, b), o = (ti * 16 + j) * 480, row = A.subarray(o, o + 480);
  const T2 = await DB.deltaTop(k, a, b), top = [];
  if (T2) { const o2 = (ti * 16 + j) * 64; for (let q = 0; q < 32; q++) { const c = T2[o2 + 2 * q], v = T2[o2 + 2 * q + 1]; if (!Number.isNaN(c)) top.push([Math.round(c), v]); } }
  return {baseline: row[0], heads: row.subarray(1, 73), mlpSum: Array.from(row.subarray(73, 85)), w_res: row.subarray(85, 469),
    r2p: row[469], r2h: row[470], cov: row[471], layer: Math.round(row[472]), lambda: row[473], top};
}
let NATT_ = null; const NATT = () => NATT_ || META.n_attn || 1500;
