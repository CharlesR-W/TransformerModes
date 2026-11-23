/* ---------- components panel: 12x6 heads, 1536 neurons of one block, ranked list ---------- */
const METRIC_DEF = {
  attr: {name: 'direct attribution', tex: String.raw`\mathrm{attr}_c=\frac{\operatorname{cov}_x\!\big(w\cdot J_l(x)\,c(x),\,f(x)\big)}{\operatorname{cov}_x(V_l w,\,f)}`,
    where: String.raw`$f\approx V_l w$ closed-form ridge fit (GCV $\lambda$) at the object's layer $l$; $c$ = a head output $A_{l',h}$, a neuron term $M_{l',n}=g_{l',n}\,(\mathrm{LS}\odot W_{\mathrm{out}}[:,n])$ or the baseline; fractions of the total, summing to 1 over all components.`},
  write: {name: 'write cosine', tex: String.raw`\cos\!\big(\mathrm{LS}_{l'}\odot W_{\mathrm{out}}^{(l')}[:,n],\; w_{\mathrm{res}}\big),\quad w_{\mathrm{res}}=\operatorname{mean}_x J_l(x)^{\top}w`,
    where: 'weight-level, raw residual coordinates; neurons only (no weight-level read direction exists for the MLP input).'},
  load: {name: 'loading', tex: String.raw`\cos\!\big(\tilde g_{l',n},\,\tilde f\big)`,
    where: String.raw`tilde = centred over the $N$ probe images; $g$ = post-GELU activation at CLS (heads: $\|A_{l',h}(x)\|_2$).`},
  auroc: {name: 'selectivity (AUROC)', tex: String.raw`\mathrm{AUROC}\big(g_{l',n}\;;\;\mathbf 1[x\in\text{set}]\big)`,
    where: 'probability that a member scores above a non-member (0.5 = no selectivity); heads use their output norm.'},
};

class Components {
  constructor(root) {
    this.root = root; this.tok = 0;
    root.innerHTML = '';
    this.hd = h('div', {class: 'hd'}, h('span', {class: 't'}, 'Components at the CLS position'), h('span', {class: 's', id: 'comp-sub'}));
    this.metricBar = h('div', {class: 'ctl'});
    this.defEl = h('div', {class: 'def'});
    this.r2El = h('div');
    this.headsCv = h('canvas', {class: 'gridcv'});
    this.pick = h('div', {class: 'blockpick'});
    this.neurHd = h('div', {class: 'note'});
    this.neurCv = h('canvas', {class: 'gridcv'});
    this.scale = h('div', {class: 'legend'});
    this.rank = h('div', {class: 'rank'});
    this.status = h('div', {class: 'loading'});
    root.append(this.hd, h('div', {class: 'flexwrap', style: {gap: '6px 12px', alignItems: 'center'}}, this.metricBar, this.status), this.defEl, this.r2El,
      h('div', {class: 'note', style: {marginTop: '4px'}}, 'Attention heads (rows h0–h5) × block'), this.headsCv,
      h('div', {class: 'note', style: {marginTop: '6px'}}, 'MLP block shown below (bar: share of |metric| per block, where cheap to compute)'), this.pick,
      this.neurHd, this.neurCv, this.scale, h('div', {class: 'note', style: {marginTop: '6px', fontWeight: 600}}, 'Largest |value|'), this.rank);
    for (const [cv, kind] of [[this.headsCv, 'head'], [this.neurCv, 'neuron']]) {
      cv.addEventListener('mousemove', e => this.onMove(e, kind));
      cv.addEventListener('mouseleave', () => TIP.hide());
      cv.addEventListener('click', e => this.onClick(e, kind));
    }
    this.cw = 0;
  }
  metricsFor(s) {
    if (!s) return [];
    const set = s.kind === 'state' || s.kind === 'class' || s.kind === 'super';
    const out = [];
    if (s.kind === 'mode' && s.view === 'cls' && s.i < 16 && DB.has(`cls/${S.k}/attr/${DB.sfx(s.l)}.f32`)) out.push('attr', 'write');
    if (s.kind === 'dmode' && s.view === 'cls' && DB.hasDeltaAttr(S.k, s.a, s.b)) out.push('attr', 'write');
    out.push(set ? 'auroc' : 'load');
    return out;
  }
  async render() {
    const tok = ++this.tok, s = S.sel;
    const W = Math.max(300, Math.floor(this.root.getBoundingClientRect().width - 26));
    if (S.view !== 'cls' && s && s.kind !== 'neuron' && s.kind !== 'head') {
      this.status.textContent = ''; this.metricBar.innerHTML = '';
      this.defEl.innerHTML = '<span class="na">Components are linked at the CLS position only (PLAN §4); the mean-patch view has no component panel.</span>';
      this.clear(W); return;
    }
    const ms = this.metricsFor(s);
    let m = S.cmetric === 'auto' || !ms.includes(S.cmetric) ? ms[0] : S.cmetric;
    this.metricBar.innerHTML = '';
    if (ms.length) {
      const seg = h('div', {class: 'seg small'});
      for (const x of ms) seg.append(h('button', {class: x === m ? 'active' : '', onclick: () => { S.cmetric = x; this.render(); }}, METRIC_DEF[x].name));
      this.metricBar.append(h('label', null, 'colour by'), seg);
    }
    if (!s) { this.defEl.innerHTML = '<span class="na">Select a mode, coarse state, smoothness-change mode, neuron or head to colour the components.</span>'; this.r2El.innerHTML = ''; this.clear(W); return; }
    const d = METRIC_DEF[m];
    this.defEl.innerHTML = `<b>${d.name}</b>: ${tex(d.tex)}<br>${T(d.where)}`;
    const lo = selLayer(s);
    let lb = S.cblock || (s.kind === 'neuron' || s.kind === 'head' ? s.l : Math.min(Math.max(lo, 1), 12));
    this.status.textContent = 'computing…';
    let res;
    const ck = [selKey(s), m, lb, S.k, S.t, S.r, S.view].join('|');
    if (this.cacheKey === ck && this.res) { res = this.res; }
    else try { res = await this.compute(s, m, lb, lo); this.cacheKey = ck; } catch (e) { console.error(e); this.status.textContent = 'error: ' + e.message; return; }
    if (tok !== this.tok) return;
    this.status.textContent = res.note || '';
    this.res = res; this.lb = lb; this.m = m;
    // R^2 badge (attribution only)
    if (m === 'attr' && res.r2) {
      const lowp = res.r2[0] < 0.3;
      const ab = res.absSum, canc = ab != null && ab > 3;
      this.r2El.innerHTML = `<span class="r2 ${lowp ? 'low' : ''}">fit ${tex('R^2')} probe ${fmt(res.r2[0], 2)} · held-out (Nyström) ${fmt(res.r2[1], 2)}</span>` +
        (ab != null ? ` <span class="ctlnum ${canc ? 'warn' : ''}" title="fractions sum to 1; the sum of their absolute values is 1 when nothing cancels">${tex('\\sum_c|\\mathrm{attr}_c|')} = ${fmt(ab, ab > 10 ? 0 : 2)}${res.absNote || ''}</span>` : '') +
        (lowp ? ' <span class="note">low R²: the linear readout explains little of f; attribution is of the fit, not of f.</span>' : '') +
        `<span class="note"> attribution is of the fit ${tex('V_l w')} at layer ${res.atLayer}, frozen-scale linearisation of the final LayerNorm.</span>`;
    } else this.r2El.innerHTML = m === 'attr' ? '<span class="err">no fit R² stored</span>' : '';
    // block chooser
    this.pick.innerHTML = '';
    const bm = res.blockBars ? Math.max(...res.blockBars.filter(x => !Number.isNaN(x)), 1e-12) : 0;
    for (let b = 1; b <= 12; b++) {
      const v = res.blockBars ? res.blockBars[b - 1] : NaN;
      const btn = h('button', {class: b === lb ? 'active' : '', title: `show block ${b}'s MLP neurons` + (Number.isNaN(v) ? '' : ` (${fmt(v)})`),
        onclick: () => { S.cblock = b; this.render(); }},
        h('i', {style: {height: (Number.isNaN(v) ? 0 : Math.max(1, 20 * v / bm)) + 'px', background: res.blockBarColor || '#b9cde2'}}), String(b));
      btn.style.width = (W / 12 - 2) + 'px';
      this.pick.append(btn);
    }
    this.neurHd.innerHTML = `MLP neurons of block <b>${lb}</b> (48 × 32, index order n = 48·row + col)` + (res.neurNote ? ' · ' + res.neurNote : '');
    this.draw(W);
    this.drawRank();
  }
  clear(W) { this.res = null; this.pick.innerHTML = ''; this.rank.innerHTML = ''; this.scale.innerHTML = ''; this.neurHd.innerHTML = ''; this.draw(W); }
  async compute(s, m, lb, lo) {
    const out = {heads: new Float32Array(72).fill(NaN), neur: new Float32Array(NNEUR).fill(NaN), base: NaN, kind: m};
    const ti = TS().indexOf(S.t);
    if (m === 'attr' && s.kind === 'mode') {
      const A = await DB.attr(S.k, s.l), C = 1 + 72 + 12 * NNEUR, row = A.subarray(s.i * C, (s.i + 1) * C);
      out.base = row[0]; out.heads.set(row.subarray(1, 73)); out.neur.set(row.subarray(73 + NNEUR * (lb - 1), 73 + NNEUR * lb));
      out.blockBars = Array.from({length: 12}, (_, b) => { let t = 0, any = false; for (let n = 0; n < NNEUR; n++) { const v = row[73 + NNEUR * b + n]; if (!Number.isNaN(v)) { t += Math.abs(v); any = true; } } return any ? t : NaN; });
      out.allRow = row; out.atLayer = s.l;
      out.absSum = 0; for (const v of row) if (!Number.isNaN(v)) out.absSum += Math.abs(v);
      const aj = (((IDX.views.cls || {}).kernels || {})[S.k] || {}).attr; const L = perLayer(aj, s.l);
      if (L) out.r2 = [L.r2_probe[s.i], L.r2_heldout[s.i]];
      if (lb > s.l) out.neurNote = 'block above the readout layer: not applicable';
    } else if (m === 'attr' && s.kind === 'dmode') {
      const at = await dmodeAttr(S.k, s.a, s.b, ti, s.j);
      if (!at) throw new Error('no attribution stored for this Δ mode');
      out.base = at.baseline; out.heads.set(at.heads);
      out.blockBars = at.mlpSum.map(v => Math.abs(v)); out.blockBarColor = '#c9b9a8';
      for (const [col, v] of at.top) { const b = Math.floor(col / NNEUR) + 1; if (b === lb) out.neur[col % NNEUR] = v; }
      out.top = at.top; out.r2 = [at.r2p, at.r2h]; out.atLayer = at.layer;
      out.absSum = Math.abs(at.baseline || 0); for (const v of at.heads) if (!Number.isNaN(v)) out.absSum += Math.abs(v); for (const v of at.mlpSum) if (!Number.isNaN(v)) out.absSum += Math.abs(v);
      out.absNote = ' (MLP counted per block: a lower bound)';
      out.neurNote = 'only the 32 largest |attribution| neurons are stored; others blank'; out.mlpSum = at.mlpSum;
    } else if (m === 'write' && s.kind === 'mode') {
      const Wc = await DB.writecos(S.k, s.l), off = s.i * 12 * NNEUR;
      out.neur.set(Wc.subarray(off + (lb - 1) * NNEUR, off + lb * NNEUR));
      out.blockBars = Array.from({length: 12}, (_, b) => { let mx = NaN; for (let n = 0; n < NNEUR; n++) { const v = Math.abs(Wc[off + b * NNEUR + n]); if (!Number.isNaN(v)) mx = Number.isNaN(mx) ? v : Math.max(mx, v); } return mx; });
      out.note = 'heads: no single write direction (n/a)';
    } else if (m === 'write' && s.kind === 'dmode') {
      const at = await dmodeAttr(S.k, s.a, s.b, ti, s.j);
      const Wt = await DB.weights(); if (!at || !Wt) throw new Error('needs delta_attr w_res and weights/wout_ls.f16');
      const w = at.w_res; let wn = 0; for (const x of w) wn += x * x; wn = Math.sqrt(wn);
      out.blockBars = [];
      for (let b = 1; b <= 12; b++) { let mx = NaN;
        for (let n = 0; n < NNEUR; n++) { const o = ((b - 1) * NNEUR + n) * 384; let dt = 0, nn = 0;
          for (let k = 0; k < 384; k++) { const x = Wt[o + k]; dt += x * w[k]; nn += x * x; }
          const v = b > at.layer ? NaN : dt / (Math.sqrt(nn) * wn || 1); if (b === lb) out.neur[n] = v; if (!Number.isNaN(v)) mx = Number.isNaN(mx) ? Math.abs(v) : Math.max(mx, Math.abs(v)); }
        out.blockBars.push(mx); }
      out.note = 'computed in the browser from w_res and LayerScale ⊙ W_out';
    } else if (m === 'load') {
      const sd = await selData(s); if (!sd || !sd.vals) throw new Error('no function on sites for this selection');
      const pre = s.kind === 'mode' && s.view === 'cls' && lb === s.l ? await DB.load(S.k, s.l) : null;
      if (pre) { out.neur.set(pre.subarray(s.i * NNEUR, (s.i + 1) * NNEUR)); out.neurNote = 'precomputed loadings'; }
      else if (DB.hasNeurons(lb)) { out.neur.set(await blockLoadings(lb, sd.vals)); out.neurNote = 'computed in the browser from 8-bit activations'; }
      for (let b = 1; b <= 12; b++) { const Hh = await DB.heads(b); if (!Hh) continue; for (let hh = 0; hh < 6; hh++) out.heads[(b - 1) * 6 + hh] = vecCos(Hh.subarray(hh * N, (hh + 1) * N), sd.vals); }
    } else if (m === 'auroc') {
      const sd = await selData(s); if (!sd || !sd.mask) throw new Error('no set for this selection');
      const ri = RS().indexOf(S.r);
      const pre = s.kind === 'state' && s.view === 'cls' && lb === s.l ? await DB.auroc(S.k, s.l) : null;
      if (pre) { out.neur.set(pre.subarray((ROFF()[ri] + s.c) * NNEUR, (ROFF()[ri] + s.c + 1) * NNEUR)); out.neurNote = 'precomputed AUROC'; }
      else if (DB.hasNeurons(lb)) { out.neur.set(await blockAUROC(lb, sd.mask)); out.neurNote = 'computed in the browser from 8-bit activations (ties ½)'; }
      for (let b = 1; b <= 12; b++) { const Hh = await DB.heads(b); if (!Hh) continue; for (let hh = 0; hh < 6; hh++) out.heads[(b - 1) * 6 + hh] = vecAUROC(Hh.subarray(hh * N, (hh + 1) * N), sd.mask, 'head' + b + '.' + hh); }
    }
    // colour scale: symmetric about 0 (or 0.5 for AUROC)
    const c0 = m === 'auroc' ? 0.5 : 0; let mx = 0;
    for (const v of out.heads) if (!Number.isNaN(v)) mx = Math.max(mx, Math.abs(v - c0));
    for (const v of out.neur) if (!Number.isNaN(v)) mx = Math.max(mx, Math.abs(v - c0));
    if (!Number.isNaN(out.base)) mx = Math.max(mx, Math.abs(out.base));
    out.c0 = c0; out.mx = mx || 1;
    return out;
  }
  geom(W) { const cw = Math.floor(W / 12), nc = Math.max(4, Math.floor(W / 48) - 1); return {cw, hh: 13, nc, ng: nc + 1}; }
  draw(W) {
    W = W || this.W; this.W = W; const g = this.geom(W), d = devicePixelRatio || 1, res = this.res;
    // heads: 6 rows x 12 blocks (+ MLP-sum row for Δ attribution, + baseline)
    const rows = 6 + (res && res.mlpSum ? 1 : 0) + (res && !Number.isNaN(res.base) ? 1 : 0);
    const HW = g.cw * 12, HH = rows * (g.hh + 1) + 1;
    this.headsCv.width = HW * d; this.headsCv.height = HH * d; this.headsCv.style.width = HW + 'px'; this.headsCv.style.height = HH + 'px';
    const c = this.headsCv.getContext('2d'); c.setTransform(d, 0, 0, d, 0, 0); c.clearRect(0, 0, HW, HH);
    const col = v => Number.isNaN(v) ? null : divColor(v - (res ? res.c0 : 0), res ? res.mx : 1);
    c.font = '10px system-ui'; c.textBaseline = 'middle';
    for (let b = 0; b < 12; b++) for (let hh = 0; hh < 6; hh++) {
      const v = res ? res.heads[b * 6 + hh] : NaN, x = b * g.cw, y = hh * (g.hh + 1);
      const cc = col(v); c.fillStyle = cc || '#f3f1ec'; c.fillRect(x + 1, y + 1, g.cw - 2, g.hh);
      if (S.sel && S.sel.kind === 'head' && S.sel.l === b + 1 && S.sel.h === hh) { c.strokeStyle = '#0b0b0b'; c.lineWidth = 1.5; c.strokeRect(x + 1.5, y + 1.5, g.cw - 3, g.hh - 1); }
    }
    let y = 6 * (g.hh + 1);
    if (res && res.mlpSum) { for (let b = 0; b < 12; b++) { const v = res.mlpSum[b]; c.fillStyle = col(v == null ? NaN : v) || '#f3f1ec'; c.fillRect(b * g.cw + 1, y + 1, g.cw - 2, g.hh); }
      c.fillStyle = '#52514e'; c.fillText('MLP Σ', 2, y + g.hh / 2 + 1); y += g.hh + 1; }
    if (res && !Number.isNaN(res.base)) { c.fillStyle = col(res.base); c.fillRect(1, y + 1, g.cw * 2 - 2, g.hh); c.fillStyle = '#52514e'; c.fillText('baseline', g.cw * 2 + 4, y + g.hh / 2 + 1); }
    // neurons: 48 x 32
    const NW = 48 * g.ng, NHt = 32 * g.ng;
    this.neurCv.width = NW * d; this.neurCv.height = NHt * d; this.neurCv.style.width = NW + 'px'; this.neurCv.style.height = NHt + 'px';
    const q = this.neurCv.getContext('2d'); q.setTransform(d, 0, 0, d, 0, 0); q.clearRect(0, 0, NW, NHt);
    for (let n = 0; n < NNEUR; n++) {
      const v = res ? res.neur[n] : NaN, x = (n % 48) * g.ng, yy = Math.floor(n / 48) * g.ng;
      q.fillStyle = col(v) || '#f1efea'; q.fillRect(x, yy, g.nc, g.nc);
    }
    if (S.sel && S.sel.kind === 'neuron' && S.sel.l === this.lb) { const n = S.sel.n; q.strokeStyle = '#0b0b0b'; q.lineWidth = 1.5; q.strokeRect((n % 48) * g.ng - 1, Math.floor(n / 48) * g.ng - 1, g.nc + 2, g.nc + 2); }
    if (this.hl != null) { const n = this.hl; q.strokeStyle = HL; q.lineWidth = 2; q.strokeRect((n % 48) * g.ng - 1.5, Math.floor(n / 48) * g.ng - 1.5, g.nc + 3, g.nc + 3); }
    if (res) {
      const lab = res.kind === 'auroc' ? `AUROC ${fmt(0.5 - res.mx, 2)}` : fmt(-res.mx), lab2 = res.kind === 'auroc' ? fmt(0.5 + res.mx, 2) : fmt(res.mx);
      this.scale.innerHTML = `<span class="num">${lab}</span><span class="ramp" style="background:${GRAD_DIV}"></span><span class="num">${lab2}</span>` +
        `<span class="note">symmetric range = largest |value| shown${res.kind === 'auroc' ? ' around 0.5' : ''}; light = not applicable / not stored</span>`;
    }
    this.g = g;
  }
  drawRank() {
    const res = this.res; this.rank.innerHTML = ''; if (!res) return;
    const items = [];
    for (let i = 0; i < 72; i++) if (!Number.isNaN(res.heads[i])) items.push({kind: 'head', l: Math.floor(i / 6) + 1, h: i % 6, v: res.heads[i]});
    if (res.allRow) { for (let b = 0; b < 12; b++) for (let n = 0; n < NNEUR; n++) { const v = res.allRow[73 + b * NNEUR + n]; if (!Number.isNaN(v)) items.push({kind: 'neuron', l: b + 1, n, v}); } }
    else if (res.top) { for (const [col, v] of res.top) items.push({kind: 'neuron', l: Math.floor(col / NNEUR) + 1, n: col % NNEUR, v}); }
    else for (let n = 0; n < NNEUR; n++) if (!Number.isNaN(res.neur[n])) items.push({kind: 'neuron', l: this.lb, n, v: res.neur[n]});
    if (!Number.isNaN(res.base)) items.push({kind: 'base', v: res.base});
    items.sort((a, b) => Math.abs(b.v - res.c0) - Math.abs(a.v - res.c0));
    const rmx = Math.max(1e-12, ...items.slice(0, 12).map(it => Math.abs(it.v - res.c0)));
    for (const it of items.slice(0, 12)) {
      const name = it.kind === 'head' ? `head ${it.l}.${it.h}` : it.kind === 'neuron' ? `neuron ${it.l}.${it.n}` : 'baseline';
      const u = clamp((it.v - res.c0) / rmx, -1, 1), w = Math.abs(u) * 50;
      const bar = `<div class="bar1"><b></b><i style="left:${u < 0 ? 50 - w : 50}%;width:${w}%;background:${divColor(it.v - res.c0, rmx)}"></i></div>`;
      const row = h('div', {class: 'rr', html: `<span>${name}</span>${bar}<span style="text-align:right">${fmt(it.v)}</span>`});
      if (it.kind !== 'base') { row.onclick = () => select(it.kind === 'head' ? {kind: 'head', l: it.l, h: it.h} : {kind: 'neuron', l: it.l, n: it.n});
        row.onmouseenter = () => { if (it.kind === 'neuron' && it.l === this.lb) { this.hl = it.n; this.draw(); } };
        row.onmouseleave = () => { this.hl = null; this.draw(); }; }
      this.rank.append(row);
    }
    this.rank.append(h('div', {class: 'note', style: {marginTop: '3px'}}, res.allRow ? 'ranked over all heads, all blocks’ neurons and the baseline' : res.top ? 'ranked over heads and the stored top-32 neurons' : `ranked over heads and block ${this.lb}'s neurons`, ' · click to select'));
  }
  hit(e, kind) {
    const r = (kind === 'head' ? this.headsCv : this.neurCv).getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, g = this.g; if (!g) return null;
    if (kind === 'head') { const b = Math.floor(x / g.cw), hh = Math.floor(y / (g.hh + 1)); return b >= 0 && b < 12 && hh >= 0 && hh < 6 ? {l: b + 1, h: hh} : null; }
    const cx = Math.floor(x / g.ng), cy = Math.floor(y / g.ng), n = cy * 48 + cx; return cx >= 0 && cx < 48 && n >= 0 && n < NNEUR ? {l: this.lb, n} : null;
  }
  onMove(e, kind) {
    const o = this.hit(e, kind); if (!o) { TIP.hide(); return; }
    const res = this.res, v = res ? (kind === 'head' ? res.heads[(o.l - 1) * 6 + o.h] : res.neur[o.n]) : NaN;
    TIP.show(`<b>${kind === 'head' ? `head ${o.h} of block ${o.l}` : `neuron ${o.n} of block ${o.l}`}</b><br>` +
      (res ? `${METRIC_DEF[res.kind].name}: <b>${Number.isNaN(v) ? 'n/a' : fmt(v)}</b>` : '') + '<br><span class="k">click: select (UMAP shows its activation)</span>', e.clientX, e.clientY);
  }
  onClick(e, kind) { const o = this.hit(e, kind); if (!o || (kind === 'neuron' && !this.lb)) return; select(kind === 'head' ? {kind: 'head', l: o.l, h: o.h} : {kind: 'neuron', l: o.l, n: o.n}, {keepBlock: true}); S.cblock = null; }
}
