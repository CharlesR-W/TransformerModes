/* ---------- Block view: Delta_t^{a->b} ---------- */
const BX = {built: false};
const DELTA_DEF = String.raw`$\Delta_t^{a\to b}=R_b^t-R_a^t$ (symmetric, annihilates constants; $R_l^t=P_l^{2t}$, Sinkhorn $P_l$, shared uniform measure). Top positive eigenfunctions: <b>became smooth</b>; most negative: <b>stopped being smooth</b>. Telescoping identity: $f^\top\Delta^{a\to b}f=\sum_{l=a}^{b-1}f^\top\Delta^{l\to l+1}f$; net $=\sum$, path $=\sum|\cdot|$ (path $\gg|$net$|$ exposes cancellation).`;
const CV_DEF = String.raw`Cross-validated value: $\phi$ = eigenfunction of $\Delta_A$ (built on half A), expanded in the endpoint's half-A eigenbasis ($|\lambda|\ge0.05$), Sinkhorn-consistently extended to half B, re-centred and normalised; reported $\phi^\top\Delta_B\phi$ next to its half-A eigenvalue. <b>Unsupported</b>: cv $<0.25\,|\mathrm{eval}_A|$ or sign flip (projection-dependent).`;

function buildBlock() {
  const root = $('#tab-block'); root.innerHTML = '';
  BX.tri = h('div', {class: 'plot'}); BX.triSeg = h('div', {class: 'seg small'}); BX.triNote = h('div', {class: 'note'});
  BX.trip = h('div');
  const left = h('div', {style: {display: 'flex', flexDirection: 'column', gap: '12px'}},
    h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Pick a block a → b'), h('span', {class: 'r'}, BX.triSeg)), BX.tri, BX.triNote),
    h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't', html: 'Definitions'})), defBlock(T(DELTA_DEF)), defBlock(T(CV_DEF)), caveat(T(CAVEAT_DMODE))));
  BX.head = h('div', {class: 'hd'}); BX.lists = h('div', {class: 'dmcols'}); BX.spec = h('div', {class: 'note'});
  BX.right = h('div', {class: 'card'}, BX.head, BX.spec, BX.lists);
  BX.detail = h('div', {style: {marginTop: '12px'}});
  root.append(h('div', {class: 'bx'}, left, h('div', null, BX.right)), BX.detail);
  BX.built = true;
}

function deltaSummary(view, k) { return ((((IDX.views[view] || {}).kernels || {})[k] || {}).delta) || null; }

async function renderBlock() {
  if (!BX.built) buildBlock();
  const gen = GEN, view = S.view, k = S.k, ti = TS().indexOf(S.t), L = DB.layers(view);
  const D = deltaSummary(view, k);
  // triangle
  BX.triSeg.innerHTML = '';
  for (const [x, lab] of [['pos', 'top + eigenvalue'], ['neg', 'most − eigenvalue'], ['sup', 'supported share']])
    BX.triSeg.append(h('button', {class: S.dsign === x ? 'active' : '', onclick: () => setS({dsign: x})}, lab));
  if (!D) { BX.tri.innerHTML = `<div class="na">no Δ results for ${kLabel(k)} (${view})</div>`; BX.lists.innerHTML = ''; BX.detail.innerHTML = ''; return; }
  const z = L.map(() => L.map(() => null)), txt = L.map(() => L.map(() => ''));
  for (const [key, r] of Object.entries(D.summary || {})) {
    const m = key.match(/a(\d+)_b(\d+)/); if (!m || !r.evals) continue;
    const a = +m[1], b = +m[2], ia = L.indexOf(a), ib = L.indexOf(b); if (ia < 0 || ib < 0) continue;
    const e = r.evals[ti]; let v;
    if (S.dsign === 'pos') v = e[0]; else if (S.dsign === 'neg') v = e[8];
    else { const u = r.unsupported ? r.unsupported[ti] : null; v = u ? u.filter(x => !x).length / u.length : null; }
    z[ib][ia] = v; txt[ib][ia] = `${a}→${b}`;
  }
  let mx = 0; for (const row of z) for (const v of row) if (v != null) mx = Math.max(mx, Math.abs(v));
  const cs = S.dsign === 'neg' ? [[0, '#0d366b'], [0.5, '#86b6ef'], [1, '#eef3f8']] : S.dsign === 'pos' ? [[0, '#fbeeec'], [0.5, '#f2a49d'], [1, '#8f1d1c']] : SEQ_STOPS;
  const shapes = [];
  const ia = L.indexOf(S.a), ib = L.indexOf(S.b);
  if (ia >= 0 && ib >= 0) shapes.push({type: 'rect', x0: S.a - 0.5, x1: S.a + 0.5, y0: S.b - 0.5, y1: S.b + 0.5, line: {color: '#0b0b0b', width: 2}});
  for (const [a, b] of (META.declared_pairs || [])) shapes.push({type: 'rect', x0: a - 0.42, x1: a + 0.42, y0: b - 0.42, y1: b + 0.42, line: {color: '#ffffff', width: 1, dash: 'dot'}});
  plot(BX.tri, [{type: 'heatmap', x: L, y: L, z, text: txt, zmin: S.dsign === 'neg' ? -mx : S.dsign === 'pos' ? 0 : 0, zmax: S.dsign === 'neg' ? 0 : S.dsign === 'pos' ? mx : 1,
    colorscale: cs, xgap: 2, ygap: 2, hoverongaps: false, colorbar: {thickness: 9, len: 0.8},
    hovertemplate: '%{text}: %{z:.3f}<extra></extra>'}],
    {height: 340, margin: {l: 40, r: 6, t: 8, b: 36}, xaxis: {title: {text: 'a (from)'}, dtick: 1, showgrid: false}, yaxis: {title: {text: 'b (to)'}, dtick: 1, showgrid: false}, shapes});
  if (!BX.triWired && BX.tri.on) { BX.tri.on('plotly_click', ev => { const p = ev.points[0]; if (p.z == null) return; setS({a: +p.x, b: +p.y}); }); BX.triWired = true; }
  BX.triNote.innerHTML = T(`Cell $(a,b)$: ${S.dsign === 'pos' ? 'largest eigenvalue of' : S.dsign === 'neg' ? 'most negative eigenvalue of' : 'share of the 16 stored modes not marked unsupported for'} $\\Delta_{${S.t}}^{a\\to b}$, ${kLabel(k)}. Dotted outline = declared pair. Click a cell.`) +
    (D.pairs && D.pairs.length < 60 ? ' <b>Only declared pairs are stored for this kernel.</b>' : '');
  // pair validity
  const key = DB.pairKey(S.a, S.b), R = (D.summary || {})[key];
  BX.head.innerHTML = `<span class="t">${T(`$\\Delta_{${S.t}}^{${S.a}\\to${S.b}}$`)} · ${kLabel(k)} · ${view === 'mp' ? 'mean-patch' : 'CLS'}</span><span class="s">16 stored eigenpairs; cards: eigenvalue, cross-validated value, net vs path; click a card</span>`;
  if (!R) { BX.lists.innerHTML = `<div class="na">pair ${S.a}→${S.b} not stored for this kernel. Pick a coloured cell.</div>`; BX.spec.innerHTML = ''; BX.detail.innerHTML = ''; return; }
  const ev = R.evals[ti], cv = R.cv ? R.cv[ti] : null, cvA = R.cv_evalA ? R.cv_evalA[ti] : null, un = R.unsupported ? R.unsupported[ti] : null, net = R.net ? R.net[ti] : null, path = R.path ? R.path[ti] : null;
  const nz = noiseControl(k, ti);
  BX.spec.innerHTML = `${un ? un.filter(x => x).length : '?'} of 16 marked unsupported by cross-validation (hatched).` +
    (nz ? ` <span class="ctlnum">noise control, ${kLabel(k)}, t = ${S.t}: max |Δ| = ${nz.vals.map(([l, v]) => `${fmt(v, 3)} (L${l})`).join(', ')}</span> <span class="note">= max |eigenvalue| of Δ between a layer's view and the same view after an isotropic perturbation at SNR ${fmt(nz.snr)} (noise power ${fmt(1 / nz.snr, 2)}, RMS about ${fmt(100 / Math.sqrt(nz.snr), 0)}% of signal), tested at layers ${nz.vals.map(x => x[0]).join(', ')}; cards whose |μ| is not above the largest of these are marked</span>` : '');
  const V = DB.hasDelta(view, k, S.a, S.b) ? await DB.deltaVec(view, k, S.a, S.b) : null;
  const info = DB.has(`${view}/${k}/delta/${key}.f16`) ? await DB.deltaInfo(view, k, S.a, S.b).catch(() => null) : null;
  if (gen !== GEN) return;
  const col = (j0, title) => {
    const c = h('div', null, h('div', {style: {fontWeight: 700, fontSize: '13px', margin: '2px 0 4px'}, html: title}));
    for (let j = j0; j < j0 + 8; j++) {
      const isSel = S.sel && S.sel.kind === 'dmode' && S.sel.a === S.a && S.sel.b === S.b && S.sel.j === j && S.sel.view === view;
      const u = un ? un[j] : false;
      let mini = '';
      if (V) { const f = V.subarray((ti * 16 + j) * N, (ti * 16 + j + 1) * N); mini = topk(f, 4).map(i => thumbHTML(i, 34)).join('') + '<span class="note">…</span>' + topk(f, 4, false).map(i => thumbHTML(i, 34)).join(''); }
      const terms = info && info.terms ? info.terms[ti][j] : null;
      const card = h('div', {class: 'dmcard' + (u ? ' unsup' : '') + (isSel ? ' sel' : ''), html:
        `<div class="top"><span class="mu">${j < 8 ? '+' : '−'}${(j % 8) + 1}: ${tex('\\mu')} = ${fmt(ev[j])}</span>` +
        (cv ? `<span>cv ${fmt(cv[j])}${cvA ? ` <span class="note">(A: ${fmt(cvA[j])})</span>` : ''}</span>` : '') +
        (un ? `<span class="badge ${u ? 'un' : 'ok'}">${u ? 'unsupported by CV' : 'CV-supported'}</span>` : '') +
        (nz && Math.abs(ev[j]) <= nz.max ? `<span class="ctlnum warn" title="noise control max |Δ| ${fmt(nz.max, 3)}">|μ| ≤ noise control ${fmt(nz.max, 2)}</span>` : '') +
        (net ? `<span class="note">net ${fmt(net[j])} · path ${fmt(path[j])}</span>` : '') + `</div>` +
        `<div class="mini">${terms ? termsSpark(terms, S.a, S.b) : ''}<span style="display:inline-flex;gap:2px;align-items:center">${mini}</span></div>`});
      card.onclick = e => { if (e.target.closest('.th')) return; select({kind: 'dmode', view, a: S.a, b: S.b, j}); };
      c.append(card);
    }
    wireThumbs(c);
    return c;
  };
  BX.lists.innerHTML = '';
  BX.lists.append(col(0, T(`Became smooth ($\\mu>0$): smooth at ${S.b}, less so at ${S.a}`)), col(8, T(`Stopped being smooth ($\\mu<0$): smooth at ${S.a}, less so at ${S.b}`)));
  renderBlockDetail(gen, info);
}
function termsSpark(terms, a, b) { // inline SVG bars of f^T Delta^{l->l+1} f, l = 1..11
  const W = 110, Hh = 28, n = terms.length, bw = W / n; let m = 1e-9; for (const v of terms) if (v != null) m = Math.max(m, Math.abs(v));
  let s = `<svg width="${W}" height="${Hh}" style="flex:none"><line x1="0" x2="${W}" y1="${Hh / 2}" y2="${Hh / 2}" stroke="#c3c2b7" stroke-width="1"/>`;
  terms.forEach((v, i) => { if (v == null) return; const l = i + 1, inr = l >= a && l <= b - 1, hh = Math.abs(v) / m * (Hh / 2 - 1);
    s += `<rect x="${i * bw + 1}" width="${bw - 2}" y="${v > 0 ? Hh / 2 - hh : Hh / 2}" height="${Math.max(0.5, hh)}" fill="${v > 0 ? (inr ? '#c0392f' : '#ecc3bf') : (inr ? '#1c5cab' : '#c3d6ec')}"><title>l=${l}→${l + 1}: ${fmt(v)}</title></rect>`; });
  return s + '</svg>';
}

async function renderBlockDetail(gen, info) {
  const el = BX.detail, s = S.sel;
  if (!s || s.kind !== 'dmode' || s.a !== S.a || s.b !== S.b || s.view !== S.view) {
    el.innerHTML = '<div class="card"><div class="na">Click a card above to see that mode on the UMAPs of both endpoints, its per-layer terms, cross-validation and component attribution.</div></div>';
    BX.comp = null; return;
  }
  const sd = await selData(s); if (gen !== GEN) return;
  CUR_SD = sd;
  el.innerHTML = '';
  const ti = TS().indexOf(S.t), sm = info && info.smooth ? info.smooth[ti][s.j] : null, L = DB.layers(s.view);
  const sAt = l => sm ? sm[L.indexOf(l)] : NaN;
  // endpoints side by side
  const wa = h('div', {class: 'umapwrap'}), wb = h('div', {class: 'umapwrap'});
  const top = h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't', html: selName(s)}),
    h('span', {class: 'r'}, h('button', {class: 'kbd', onclick: () => setS({tab: 'layer', l: selLayer(s)})}, 'open in Layer explorer'))),
    defBlock(dmodeDef(s)),
    h('div', {class: 'sidebyside'},
      h('div', null, h('div', {class: 'note', html: T(`UMAP of layer ${s.a} (from) · $s_{${s.a}}^{${S.t}}(f)$ = <b>${fmt(sAt(s.a))}</b>`)}), wa),
      h('div', null, h('div', {class: 'note', html: T(`UMAP of layer ${s.b} (to) · $s_{${s.b}}^{${S.t}}(f)$ = <b>${fmt(sAt(s.b))}</b>`)}), wb)),
    h('div', {class: 'legend', html: legendHTML(sd, ' <span class="note">same colour scale on both; hover a point to see its image</span>')}));
  el.append(top);
  for (const [w, l] of [[wa, s.a], [wb, s.b]]) {
    const sc = new Scatter(w, {tip: i => imgTip(i, sd), onClick: i => focusImg(i), r: 1.8});
    const xy = await umapLayer(s.view, l); sc.setBounds(UMAPB[s.view]); sc.setPoints(xy, false); sc.setColor(specFrom(sd)); sc.setFocus(S.img);
  }
  // charts + strips + components
  const charts = h('div', {class: 'card'}); const strips = h('div', {class: 'card'}); const comp = h('div', {class: 'card'});
  el.append(h('div', {class: 'lx2', style: {gridTemplateColumns: 'minmax(0,1.05fr) minmax(0,0.85fr) minmax(470px,1.1fr)'}}, charts, strips, comp));
  await renderDmodeCharts(charts, s, gen, false, info);
  strips.innerHTML = `<div class="hd"><span class="t">Images</span></div><div class="strips">${stripsHTML([{label: T('largest $f$'), idx: topk(sd.vals, 15)}, {label: T('smallest $f$'), idx: topk(sd.vals, 15, false)}])}</div>`;
  wireThumbs(strips);
  if (s.view === 'cls') { BX.comp = new Components(comp); BX.comp.render(); } else comp.innerHTML = '<div class="na">Components are linked at the CLS position only.</div>';
  // sub-block relation and product operator
  const extra = h('div', {class: 'two', style: {marginTop: '12px'}}); el.append(extra);
  renderTriples(extra, s); renderQ(extra, s);
}

async function renderDmodeCharts(el, s, gen, compact, info) {
  info = info || await DB.deltaInfo(s.view, S.k, s.a, s.b).catch(() => null);
  if (gen !== GEN) return;
  const ti = TS().indexOf(S.t), L = DB.layers(s.view);
  if (compact) el.innerHTML = `<div class="hd"><span class="t">Per-layer terms of this mode</span></div>`;
  else el.innerHTML = `<div class="hd"><span class="t">Where along depth the change happens</span></div>`;
  if (!info) { el.append(h('div', {class: 'na'}, 'no per-pair Δ JSON')); return; }
  const terms = info.terms[ti][s.j], net = info.net[ti][s.j], path = info.path[ti][s.j];
  el.append(defBlock(T(`$f^\\top\\Delta_{${S.t}}^{l\\to l+1}f$ for $l=1..${terms.length}$; dark bars are the terms of the identity ($a\\le l<b$), light ones context. net = ${fmt(net)}, path = ${fmt(path)}` + (Math.abs(net) > 0 && path / Math.abs(net) > 2 ? ' <b>(path ≫ |net|: cancellation)</b>' : ''))));
  const p1 = h('div', {class: 'plot'}); el.append(p1);
  const xs = terms.map((_, i) => L.includes(0) ? i : i + 1);
  plot(p1, [{type: 'bar', x: xs.map(l => `${l}→${l + 1}`), y: terms, marker: {color: terms.map((v, i) => { const l = xs[i], inr = l >= s.a && l <= s.b - 1; return v > 0 ? (inr ? '#c0392f' : '#ecc3bf') : (inr ? '#1c5cab' : '#c3d6ec'); })},
    hovertemplate: '%{x}: %{y:.4f}<extra></extra>'}], {height: 200, margin: {t: 8}, yaxis: {title: {text: 'f^T Δ^{l→l+1} f'}, zeroline: true}, xaxis: {tickfont: {size: 10}}});
  if (info.smooth) {
    const p2 = h('div', {class: 'plot'}); el.append(defBlock(T(`Smoothness $s_l^{${S.t}}(f)$ of the same $f$ at every layer (the terms above are its successive differences):`)), p2);
    plot(p2, [{x: L, y: info.smooth[ti][s.j], mode: 'lines+markers', line: {color: tColor(S.t), width: 2}, marker: {size: 6}, name: 't = ' + S.t, hovertemplate: 'layer %{x}: %{y:.3f}<extra></extra>'}],
      {height: 180, margin: {t: 8}, xaxis: {dtick: 1, title: {text: 'layer l'}}, yaxis: {title: {text: 's_l^t(f)'}, range: [0, 1.02]},
        shapes: [s.a, s.b].map(x => ({type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: {color: '#898781', width: 1, dash: 'dot'}}))});
  }
  if (!compact && info.cv) {
    const r = (f) => info[f] ? fmt(Array.isArray(info[f][ti][s.j]) ? NaN : info[f][ti][s.j]) : '–';
    const sub = info.cv_subspace_cos ? info.cv_subspace_cos[ti][s.j < 8 ? 0 : 1] : null;
    el.append(h('div', {html: `<table class="doc"><tr><th>eigenvalue (all N)</th><th>cv value ${tex('\\phi^\\top\\Delta_B\\phi')}</th><th>half-A eigenvalue</th><th>discarded energy</th><th>status</th></tr>` +
      `<tr><td class="num">${fmt(info.evals[ti][s.j])}</td><td class="num">${r('cv')}</td><td class="num">${r('cv_evalA')}</td><td class="num">${r('cv_discarded')}</td>` +
      `<td>${info.unsupported ? (info.unsupported[ti][s.j] ? '<span class="badge un">unsupported by CV</span>' : '<span class="badge ok">CV-supported</span>') : '–'}</td></tr></table>` +
      (sub ? `<div class="note">principal cosines of the extended leading ${s.j < 8 ? '+' : '−'} 8-subspaces (A→B vs B): ${sub.map(x => fmt(x, 2)).join(', ')}</div>` : '')}));
  }
}

function renderTriples(el, s) {
  const D = deltaSummary(s.view, S.k), tr = (D && D.triples) || [];
  const rel = tr.filter(x => x.t === S.t && ((x.abc[0] === s.a && x.abc[2] === s.b) || (x.abc[0] === s.a && x.abc[1] === s.b) || (x.abc[1] === s.a && x.abc[2] === s.b)));
  const c = h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Long block vs its sub-blocks')),
    defBlock(T('Principal cosines between the leading 8-dim $\\pm$ eigenspaces of $\\Delta^{a\\to c}$ and those of $\\Delta^{a\\to b}$, $\\Delta^{b\\to c}$, and of $\\Delta^{a\\to b}+\\Delta^{b\\to c}$, for the declared triples. Shown for triples involving the current pair at t = ' + S.t + '.')));
  if (!rel.length) c.append(h('div', {class: 'na'}, 'the current pair is not part of a declared triple (1,3,5), (1,6,12), (6,9,12), (3,6,9)'));
  else {
    const rows = rel.map(x => `<tr><td>${x.abc.join('→')}</td><td>${x.sign}</td><td class="mono">${(x.cos_ac_ab || []).map(v => fmt(v, 2)).join(' ')}</td><td class="mono">${(x.cos_ac_bc || []).map(v => fmt(v, 2)).join(' ')}</td><td class="mono">${(x.cos_ac_sum || []).map(v => fmt(v, 2)).join(' ')}</td></tr>`).join('');
    c.append(h('div', {html: `<table class="doc"><tr><th>a→b→c</th><th>±</th><th>cos(ac, ab)</th><th>cos(ac, bc)</th><th>cos(ac, ab+bc)</th></tr>${rows}</table>`}));
  }
  el.append(c);
}
function renderQ(el, s) {
  const K = (((IDX.views[s.view] || {}).kernels || {})[S.k] || {}), q = K.q && (K.q[DB.pairKey(s.a, s.b)] || (K.q.pairs || {})[DB.pairKey(s.a, s.b)]);
  const c = h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Product operator (comparison)')),
    defBlock(T(`$Q^{a\\to b}=P_bP_a$, top 32 singular values, next to the predecessor's mean operator $\\frac{1}{b-a+1}\\sum_{l=a}^{b}P_l$ (top 32 nonconstant eigenvalues). Stored for declared pairs.`)));
  el.append(c);
  if (!q) { c.append(h('div', {class: 'na'}, 'not stored for this pair')); return; }
  const p = h('div', {class: 'plot'}); c.append(p);
  const sv = q.sv.length > (q.mean_evals || []).length ? q.sv.slice(1) : q.sv; // drop sigma_0 = 1 (constant) when stored
  const idx = sv.map((_, i) => i + 1);
  plot(p, [{x: idx, y: sv, name: 'σ_k(Q), k ≥ 1', mode: 'lines+markers', line: {color: CAT8[0]}}, {x: idx, y: q.mean_evals, name: 'λ_k(mean operator), nonconstant', mode: 'lines+markers', line: {color: CAT8[1], dash: 'dash'}}],
    {height: 220, margin: {t: 26}, xaxis: {title: {text: 'k'}}, yaxis: {title: {text: 'value'}}});
  const cq = Object.entries(q.cos_q_modes || q).filter(([k2]) => k2.startsWith('cos_q') || q.cos_q_modes);
  if (cq.length) c.append(h('div', {class: 'note', html: 'principal cosines, top-8 singular subspaces of Q vs endpoint modes: ' + cq.map(([k2, v]) => `<b>${esc(k2.replace('cos_q_', ''))}</b>: ${(Array.isArray(v) ? v : [v]).map(x => fmt(x, 2)).join(' ')}`).join(' · ')}));
}

/* noise control (controls.json, SCHEMA v2) for kernel k at time index ti: [[layer, max|eig|]...] */
function noiseControl(k, ti) {
  const c = (IDX.controls || {})[k]; if (!c || !c.per_layer) return null;
  const vals = c.per_layer.filter(r => r.noise && r.noise.max_abs_eig).map(r => [r.layer, r.noise.max_abs_eig[ti]]);
  return vals.length ? {vals, max: Math.max(...vals.map(x => x[1])), snr: c.noise_snr} : null;
}
