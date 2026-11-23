/* ---------- Layer explorer ---------- */
const LX = {built: false};
function viewDefTex(view) {
  return view === 'mp'
    ? String.raw`$v_l(x)=\mathrm{norm}\big(\operatorname{mean}_{p}\,\mathrm{LN}(h_l(x))_p - m_l\big)$, mean of patch tokens (secondary view, $l\in\{0..12\}$)`
    : String.raw`$v_l(x)=\dfrac{\mathrm{LN}(h_l(x))_{\mathrm{CLS}}-m_l}{\|\mathrm{LN}(h_l(x))_{\mathrm{CLS}}-m_l\|}$, $h_l$ = residual stream after block $l$, $m_l$ = probe mean`;
}
const CAVEAT_VIEW = 'CLS (and mean-patch) are <b>partial views</b> of the network state: CLS at $l$ is not a function of CLS at $l-1$ (patch tokens write into it), so no data-processing monotonicity is expected; statements are about the geometry of the view, not about information in the network.';

function buildLayer() {
  const root = $('#tab-layer'); root.innerHTML = '';
  // UMAP card
  LX.colorSeg = h('div', {class: 'seg small'});
  LX.umapSub = h('span', {class: 's'});
  LX.umapWrap = h('div', {class: 'umapwrap'});
  LX.legend = h('div', {class: 'legend'});
  LX.viewDef = defBlock('');
  const umapCard = h('div', {class: 'card'},
    h('div', {class: 'hd'}, h('span', {class: 't', id: 'lx-umap-t'}), LX.umapSub, h('span', {class: 'r'}, h('span', {class: 'ctl'}, h('label', null, 'colour'), LX.colorSeg))),
    LX.umapWrap, LX.legend, LX.viewDef);
  LX.sc = new Scatter(LX.umapWrap, {
    tip: i => imgTip(i, LX.hoverSD || CUR_SD),
    onClick: i => focusImg(i),
  });
  // objects card
  LX.modeHd = h('div', {class: 'def'}); LX.modeCols = h('div', {class: 'colhd'}); LX.modeList = h('div', {class: 'objlist tall'});
  LX.stateHd = h('div', {class: 'def'}); LX.stateCols = h('div', {class: 'colhd'}); LX.stateList = h('div', {class: 'objlist'});
  const objCard = h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't', id: 'lx-obj-t'}, 'Objects of this layer')),
    h('div', {style: {fontWeight: 650, fontSize: '13px'}}, 'Modes'), LX.modeHd, LX.modeCols, LX.modeList,
    h('div', {style: {fontWeight: 650, fontSize: '13px', marginTop: '10px'}}, 'Coarse states'), LX.stateHd, LX.stateCols, LX.stateList);
  // components card
  const compCard = h('div', {class: 'card'});
  LX.comp = new Components(compCard);
  // detail row
  LX.detail = h('div', {class: 'card'}); LX.side = h('div', {style: {display: 'flex', flexDirection: 'column', gap: '12px'}});
  LX.inspector = h('div', {class: 'card'}); LX.sidePlot = h('div', {class: 'card'});
  LX.side.append(LX.inspector, LX.sidePlot);
  root.append(h('div', {class: 'lx'}, umapCard, objCard, compCard), h('div', {class: 'lx2'}, LX.detail, LX.side));
  LX.built = true;
}

async function renderLayer() {
  if (!LX.built) buildLayer();
  const gen = GEN, view = S.view, l = S.l, k = S.k;
  $('#lx-umap-t').innerHTML = `UMAP of the layer-${l} view${view === 'mp' ? ' (mean-patch)' : ' (CLS)'}`;
  LX.umapSub.textContent = 'aligned UMAP (umap-learn AlignedUMAP, cosine, 15 neighbours); a 2-d placement for browsing, distances not preserved; axes fixed across layers';
  LX.viewDef.innerHTML = T(viewDefTex(view)) + `<div class="caveat" style="margin-top:4px">${T(CAVEAT_VIEW)}</div>`;
  // colour mode buttons
  LX.colorSeg.innerHTML = '';
  for (const [x, lab] of [['sel', 'selection'], ['state', `coarse states (r=${S.r})`], ['super', 'superclass']])
    LX.colorSeg.append(h('button', {class: S.color === x ? 'active' : '', onclick: () => setS({color: x})}, lab));
  // points
  UMAPB[view] || await umapLayer(view, l);
  const xy = await umapLayer(view, l);
  if (gen !== GEN) return;
  if (!xy) { LX.sc.msg = 'no UMAP for this view'; LX.sc.xy = null; LX.sc.draw(); }
  else { LX.sc.setBounds(UMAPB[view]); if (LX.lastView !== view) { LX.sc.xy = null; LX.lastView = view; } LX.sc.setPoints(xy); }
  // selection data -> colour
  const sd = await selData(S.sel); if (gen !== GEN) return;
  CUR_SD = sd;
  await colourUMAP(sd, gen);
  if (gen !== GEN) return;
  LX.sc.setFocus(S.img);
  renderObjects(gen, sd);
  LX.comp.render();
  renderDetail(gen, sd);
  renderInspector(gen);
}

async function colourUMAP(sd, gen) {
  const view = S.view, l = S.l;
  LX.sc.setLabels([]);
  if (S.color === 'state') {
    const hard = DB.hasSeba(view, S.k, l) ? await DB.hard(view, S.k, l) : null;
    if (gen !== GEN) return;
    if (!hard) { LX.sc.setColor({type: 'none'}); LX.legend.innerHTML = '<span class="err">no coarse states for this kernel/layer</span>'; return; }
    const ri = RS().indexOf(S.r), lab = hard.subarray(ri * N, (ri + 1) * N), r = S.r;
    const selc = S.sel && S.sel.kind === 'state' && S.sel.l === l ? S.sel.c : null;
    const colors = r <= 8 ? CAT8.slice(0, r) : Array.from({length: r}, (_, c) => c === selc ? '#2a78d6' : '#b8b4ab');
    LX.sc.setColor({type: 'cat', vals: lab, colors, mask: selc != null ? lab.map(x => x === selc ? 1 : 0) : null});
    // centroid labels (direct labels; the categorical palette is only used for r <= 8)
    const xy = await umapLayer(view, l), sx = new Float64Array(r), sy = new Float64Array(r), cnt = new Float64Array(r);
    for (let i = 0; i < N; i++) { const c = lab[i]; if (c >= 0) { sx[c] += xy[2 * i]; sy[c] += xy[2 * i + 1]; cnt[c]++; } }
    LX.sc.setLabels(Array.from({length: r}, (_, c) => cnt[c] ? {x: sx[c] / cnt[c], y: sy[c] / cnt[c], text: String(c), color: c === selc ? '#0d366b' : '#2b2a27'} : null).filter(Boolean));
    LX.legend.innerHTML = `hard coarse states $c_{${l}}(x)$, r = ${r}` .replace(/\$(.+?)\$/g, (_, m) => tex(m)) +
      (r <= 8 ? CAT8.slice(0, r).map((c, j) => ` <span><span class="sw" style="background:${c}"></span>${j}</span>`).join('') : ' · grey with numbers at centroids (more than 8 states: no categorical palette)') +
      ' · <span class="sw" style="background:#ebe8e2"></span>unassigned · click a state in the list to highlight';
    return;
  }
  if (S.color === 'super') {
    const nS = (META.superclasses || []).length;
    if (LAB.sup && nS && nS <= 8) {
      LX.sc.setColor({type: 'cat', vals: LAB.sup, colors: CAT8.slice(0, nS)});
      LX.legend.innerHTML = 'superclass (label: display/validation only): ' + CAT8.slice(0, nS).map((c, j) => `<span><span class="sw" style="background:${c}"></span>${esc(superName(j))}</span>`).join(' ');
    } else {
      const sel = S.sel && S.sel.kind === 'super' ? S.sel.s : null;
      LX.sc.setColor(sel != null ? {type: 'set', mask: LAB.sup.map(x => x === sel ? 1 : 0)} : {type: 'none'});
      LX.legend.innerHTML = '';
      const pick = h('select', {onchange: e => select({kind: 'super', s: +e.target.value})}, h('option', {value: ''}, 'pick a superclass…'),
        ...(META.superclasses || []).map((x, j) => h('option', {value: j, selected: j === sel}, `${j}: ${x.name}`)));
      LX.legend.append(h('span', null, `${nS} superclasses (more than 8: one at a time) `), pick, h('span', {class: 'note'}, ' labels are for display/validation only'));
    }
    return;
  }
  LX.sc.setColor(specFrom(sd));
  const where = S.sel && S.sel.kind !== 'class' && S.sel.kind !== 'super' && selLayer(S.sel) !== l && S.sel.kind !== 'dmode'
    ? ` <span class="caveat" style="padding:0 6px">object lives at layer ${selLayer(S.sel)}; shown on the layer-${l} UMAP</span>` : '';
  LX.legend.innerHTML = legendHTML(sd, where);
  if (S.sel && (S.sel.kind === 'class' || S.sel.kind === 'super')) {
    const pick = h('select', {onchange: e => select({kind: 'class', c: +e.target.value})}, ...(META.classes || []).map((x, j) => h('option', {value: j, selected: S.sel.kind === 'class' && j === S.sel.c}, `${j}: ${x.name}`)));
    LX.legend.append(pick);
  }
}

/* objects list: modes and coarse states of layer S.l, with a context column driven by the selection */
async function renderObjects(gen, sd) {
  const view = S.view, l = S.l, k = S.k;
  $('#lx-obj-t').textContent = `Objects of layer ${l}`;
  LX.modeHd.innerHTML = T(`$P_l = D K D$ (Sinkhorn, $P\\one=\\one$) with $${kTex(k)}$; $P_l\\psi_i=\\lambda_i\\psi_i$, top 64 nonconstant, $\\|\\psi_i\\|=1$ on the $N$ images.`);
  LX.stateHd.innerHTML = T(`SEBA on $\\mathrm{span}\\{\\one,\\psi_1..\\psi_{r-1}\\}$, r = ${S.r}; hard $c_l(x)=\\arg\\max_j m_j(x)$, unassigned if $\\max_j m_j(x)<0.1$ (vectors scaled to max 1).`);
  const ev = await DB.evals(view, k); if (gen !== GEN) return;
  const li = DB.li(view, l), lam = ev ? Array.from(ev.subarray(li * NM, (li + 1) * NM)) : null;
  // context: what does the selection say about each mode / state?
  let ctxMode = null, ctxModeName = '', ctxState = null, ctxStateName = '';
  const selPl = (((IDX.selection || {}).views || {})[view] || {})[k];
  const effN = selPl ? (perLayer(selPl, l, view) || {}).eff_n_modes : null;
  try {
    if (sd && sd.vals && DB.hasModes(view, k, l) && !(S.sel.kind === 'mode' && S.sel.l === l)) {
      const M = await DB.modes(view, k, l), fu = centredUnit(sd.vals);
      if (S.sel.kind === 'neuron' && S.sel.l === l && view === 'cls' && DB.has(`cls/${k}/load/${DB.sfx(l)}.f16`)) {
        const Ld = await DB.load(k, l); ctxMode = Array.from({length: NM}, (_, i) => Ld[i * NNEUR + S.sel.n]);
      } else if (fu) ctxMode = Array.from({length: NM}, (_, i) => { let s = 0; for (let x = 0; x < N; x++) s += M[i * N + x] * fu[x]; return s / Math.sqrt(N); });
      ctxModeName = T('$\\cos(\\tilde f,\\psi_i)$ with the selection');
    }
    const hard = DB.hasSeba(view, k, l) ? await DB.hard(view, k, l) : null;
    if (hard && sd && (sd.vals || sd.mask) && !(S.sel.kind === 'state' && S.sel.l === l)) {
      const ri = RS().indexOf(S.r), lab = hard.subarray(ri * N, (ri + 1) * N);
      if (sd.vals && S.sel.kind === 'neuron' && S.sel.l === l && view === 'cls' && DB.has(`cls/${k}/auroc/${DB.sfx(l)}.f16`)) {
        const A = await DB.auroc(k, l); ctxState = Array.from({length: S.r}, (_, c) => A[(ROFF()[ri] + c) * NNEUR + S.sel.n]); ctxStateName = 'AUROC of the selection for the state';
      } else if (sd.vals) { ctxState = aurocPerState(sd.vals, lab, S.r); ctxStateName = 'AUROC of the selection for the state'; }
      else { ctxState = Array.from({length: S.r}, (_, c) => { let a = 0, b = 0; for (let i = 0; i < N; i++) if (lab[i] === c) { a++; b += sd.mask[i]; } return a ? b / a : NaN; }); ctxStateName = 'share of the state inside the selected set'; }
    }
  } catch (e) { console.error(e); }
  if (gen !== GEN) return;
  // modes
  LX.modeCols.innerHTML = `<span></span><span>${T('$\\lambda_i$ of $P_l$')}</span><span>${ctxMode ? ctxModeName : effN ? T('localisation $1/\\sum_x\\psi_i(x)^4$') : ''}</span>`;
  LX.modeList.innerHTML = '';
  const selI = S.sel && S.sel.kind === 'mode' && S.sel.l === l && S.sel.view === view ? S.sel.i : null;
  for (let i = 0; i < NM; i++) {
    const lv = lam ? lam[i] : NaN;
    let c3 = '';
    if (ctxMode) { const v = ctxMode[i], w = Math.abs(v) * 50; c3 = `<div class="bar1"><b></b><i style="left:${v < 0 ? 50 - w : 50}%;width:${w}%;background:${divColor(v, 1)}"></i><span>${fmt(v, 2)}</span></div>`; }
    else if (effN) { const v = effN[i], w = clamp(Math.log10(Math.max(v, 1)) / Math.log10(N) * 100, 0, 100); c3 = `<div class="bar1" title="effective number of images ${fmt(v)} (orange: below 10, the per-mode threshold of the pre-registered selection rule)"><i style="left:0;width:${w}%;background:${v < 10 ? '#e3a07a' : '#9ab7d3'}"></i><span>${fmt(v)}</span></div>`; }
    const row = h('div', {class: 'orow' + (i === selI ? ' sel' : ''), html:
      `<span class="lab">${tex(`\\psi_{${i + 1}}`)}${i < 16 && view === 'cls' && kFull(k) ? '' : ''}</span>` +
      `<div class="bar1"><i style="left:0;width:${clamp(lv, 0, 1) * 100}%;background:#86b6ef"></i><span>${fmt(lv, 3)}</span></div>${c3}`});
    row.onclick = () => select({kind: 'mode', view, l, i});
    row.onmouseenter = () => previewObj({kind: 'mode', view, l, i});
    row.onmouseleave = () => previewObj(null);
    LX.modeList.append(row);
  }
  if (selI != null && LX._scrolledTo !== selKey(S.sel)) { const el = LX.modeList.children[selI]; if (el) LX.modeList.scrollTop = el.offsetTop - LX.modeList.clientHeight / 2; LX._scrolledTo = selKey(S.sel); }
  // states
  const hard = DB.hasSeba(view, k, l) ? await DB.hard(view, k, l) : null;
  if (gen !== GEN) return;
  LX.stateList.innerHTML = ''; LX.stateCols.innerHTML = '';
  if (!hard) { LX.stateList.innerHTML = `<div class="na" style="padding:6px">no SEBA states stored for ${kLabel(k)} (only kernels_full have them)</div>`; return; }
  const ri = RS().indexOf(S.r), lab = hard.subarray(ri * N, (ri + 1) * N), size = new Array(S.r).fill(0); let un = 0;
  for (let i = 0; i < N; i++) lab[i] >= 0 ? size[lab[i]]++ : un++;
  const mxs = Math.max(...size, 1);
  LX.stateCols.innerHTML = `<span></span><span>hard-set size (of ${fmtInt(N)})</span><span>${ctxState ? ctxStateName : ''}</span>`;
  const selC = S.sel && S.sel.kind === 'state' && S.sel.l === l && S.sel.view === view ? S.sel.c : null;
  for (let c = 0; c < S.r; c++) {
    let c3 = '';
    if (ctxState) { const v = ctxState[c]; if (ctxStateName.startsWith('AUROC')) { const u = v - 0.5, w = Math.abs(u) * 100; c3 = `<div class="bar1"><b></b><i style="left:${u < 0 ? 50 - w : 50}%;width:${w}%;background:${divColor(u, 0.5)}"></i><span>${fmt(v, 2)}</span></div>`; }
      else c3 = `<div class="bar1"><i style="left:0;width:${(v || 0) * 100}%;background:${HL}"></i><span>${fmt(v, 2)}</span></div>`; }
    const sw = S.r <= 8 ? `<span class="sw" style="background:${CAT8[c]}"></span>` : '';
    const row = h('div', {class: 'orow' + (c === selC ? ' sel' : ''), html: `<span class="lab">${sw}${c}</span><div class="bar1"><i style="left:0;width:${size[c] / mxs * 100}%;background:#b9cde2"></i><span>${fmtInt(size[c])}</span></div>${c3}`});
    row.onclick = () => select({kind: 'state', view, l, c});
    row.onmouseenter = () => previewObj({kind: 'state', view, l, c});
    row.onmouseleave = () => previewObj(null);
    LX.stateList.append(row);
  }
  LX.stateList.append(h('div', {class: 'note', style: {padding: '3px 6px'}}, `unassigned: ${fmtInt(un)} images`));
  const bl = sameLayerBaseline(view, k, l, S.r);
  LX.stateHd.innerHTML += ` <span class="ctlnum ${un / N > 0.3 ? 'warn' : ''}">coverage ${fmt(1 - un / N, 2)}</span>` +
    (bl ? ` <span class="ctlnum warn" title="same-layer split-half baseline (DEVIATIONS 12): hard states from half-A- vs half-B-built operators on held-out images">split-half ${T('$H(c^B|c^A)$')} = ${fmt(bl.H_B_given_A, 2)} bits, ${T('$H(c^A|c^B)$')} = ${fmt(bl.H_A_given_B, 2)} bits</span>` : '');
}

/* hover preview of a list object on the UMAP (does not change the selection) */
const previewObj = debounce(async o => {
  if (S.tab !== 'layer') return;
  if (!o) { if (!LX.hoverSD) return; LX.hoverSD = null; colourUMAP(CUR_SD, GEN); return; }
  const sd = await selData(o); if (!sd || sd.missing) return;
  LX.hoverSD = sd; LX.sc.setColor(specFrom(sd));
  LX.legend.innerHTML = '<b>preview</b> · ' + legendHTML(sd);
}, 40);

/* detail: definition, image strips */
async function renderDetail(gen, sd) {
  const s = S.sel, el = LX.detail;
  if (!s) { el.innerHTML = `<div class="hd"><span class="t">Selection</span></div><div class="na">Nothing selected. Click a mode or coarse state in the list, a head or neuron in the components panel, or use the Block and Depth tabs. Hover any list row to preview it on the UMAP. Clicking a UMAP point inspects that image without changing the selection.</div>`; renderSide(gen, null); return; }
  if (sd && sd.missing) { el.innerHTML = `<div class="hd"><span class="t">${selName(s)}</span></div><div class="err">${esc(sd.missing)}</div>`; renderSide(gen, s); return; }
  let defs = '', groups = [];
  if (s.kind === 'mode') {
    defs = T(`Eigenfunction of the layer-${s.l} operator: $P_{${s.l}}\\psi_{${s.i + 1}} = \\lambda\\psi_{${s.i + 1}}$, kernel $${kTex(S.k)}$, unit norm on the $N$ images, sign: largest $|\\psi|$ positive. Values shown are $\\sqrt N\\psi$ (RMS 1).`);
    groups = [{label: T('largest $\\psi$'), idx: topk(sd.vals, 12)}, {label: T('smallest $\\psi$'), idx: topk(sd.vals, 12, false)}];
  } else if (s.kind === 'state') {
    defs = T(`SEBA vector $m_{${s.c}}$ on $\\mathrm{span}\\{\\one,\\psi_1..\\psi_{${S.r - 1}}\\}$ of layer ${s.l}, clipped at 0, scaled to max 1; the hard set is $\\{x: c_{${s.l}}(x)=${s.c}\\}$. A measured regime, not a premise: see SEBA diagnostics in Settings.`);
    const members = []; for (let i = 0; i < N; i++) if (sd.mask[i]) members.push(i);
    groups = [{label: T('largest membership $m_j(x)$'), idx: topk(sd.vals, 12)}, {label: `random members of the hard set (${fmtInt(members.length)} images)`, idx: seededSample(members, 12, s.c + 7)}];
  } else if (s.kind === 'neuron') {
    defs = T(`$g_{${s.l},${s.n}}(x)$ = post-GELU activation of MLP neuron ${s.n} in block ${s.l} at the CLS position (a function on sites). Its residual write is $M_{${s.l},${s.n}}(x)=g(x)\\,(\\mathrm{LS}\\odot W_{\\mathrm{out}}[:,${s.n}])$.`);
    const order = topk(sd.vals, N); const q = [0.5, 0.75, 0.9, 0.95, 0.99].map(p => order[Math.floor((1 - p) * (N - 1))]);
    groups = [{label: 'largest activation', idx: order.slice(0, 12), vals: order.slice(0, 12).map(i => sd.vals[i])},
      {label: 'at activation quantiles 50 / 75 / 90 / 95 / 99 %', idx: q, vals: q.map(i => sd.vals[i])}];
  } else if (s.kind === 'head') {
    defs = T(`$\\|A_{${s.l},${s.h}}(x)\\|_2$: norm of head ${s.h}'s output to CLS in block ${s.l} (after output projection and LayerScale), the head's scalar function on sites used for colouring. Exact per head because the projection is linear.`);
    groups = [{label: T('largest $\\|A(x)\\|$'), idx: topk(sd.vals, 12)}, {label: T('smallest $\\|A(x)\\|$'), idx: topk(sd.vals, 12, false)}];
  } else if (s.kind === 'dmode') {
    defs = dmodeDef(s);
    groups = [{label: T('largest $f$'), idx: topk(sd.vals, 12)}, {label: T('smallest $f$'), idx: topk(sd.vals, 12, false)}];
  } else if (s.kind === 'class' || s.kind === 'super') {
    defs = 'Label-defined set: shown for validation and display only; labels never define a kernel or basis.';
    const m = []; for (let i = 0; i < N; i++) if (sd.mask[i]) m.push(i);
    groups = [{label: `random members (${m.length})`, idx: seededSample(m, 24, 3)}];
  }
  if (gen !== GEN) return;
  el.innerHTML = `<div class="hd"><span class="t">${selName(s)}</span>${HIST.length ? '' : ''}</div><div class="def">${defs}</div>` +
    (s.kind === 'dmode' ? `<div class="caveat">${T(CAVEAT_DMODE)}</div>` : '') + `<div class="strips">${stripsHTML(groups)}</div>`;
  wireThumbs(el);
  renderSide(gen, s, sd);
}
const CAVEAT_DMODE = 'Endpoint smoothness-change modes are <b>not identified operations</b>: they depend on the declared resolution (kernel parameter and $t$), and a block that rearranges and then restores has $\\Delta=0$.';
function dmodeDef(s) {
  return T(`Eigenfunction $f$ of $\\Delta_{${S.t}}^{${s.a}\\to${s.b}} = R_{${s.b}}^{${S.t}} - R_{${s.a}}^{${S.t}}$ (with $R_l=P_l^2$, $R_l^t=P_l^{2t}$), ` +
    (s.j < 8 ? `the ${s.j + 1}${['st', 'nd', 'rd'][s.j] || 'th'} largest positive eigenvalue: <b>became smooth</b> between layers ${s.a} and ${s.b}.` : `the ${s.j - 7}${['st', 'nd', 'rd'][s.j - 8] || 'th'} most negative eigenvalue: <b>stopped being smooth</b> between layers ${s.a} and ${s.b}.`) +
    ` Its eigenvalue is $f^\\top\\Delta f = s_{${s.b}}^{t}(f) - s_{${s.a}}^{t}(f)$.`);
}

/* side panel: smoothness profile (modes, Δ modes), flows (states), write cosines (neurons), attention maps (heads) */
async function renderSide(gen, s, sd) {
  const el = LX.sidePlot; el.innerHTML = '';
  if (!s) { el.innerHTML = `<div class="hd"><span class="t">Smoothness profile</span></div><div class="def">${T(SMOOTH_DEF)}</div><div class="na">Select a mode to see its smoothness profile over depth and diffusion time.</div>`; return; }
  if (s.kind === 'mode') {
    const Sm = await DB.smooth(s.view, S.k); if (gen !== GEN) return;
    el.innerHTML = `<div class="hd"><span class="t">Smoothness profile of ${tex(`\\psi_{${s.i + 1}}^{(${s.l})}`)} over depth</span></div><div class="def">${T(SMOOTH_DEF + ` Here $S[a,i,b,t]=s_b^t(\\psi_i^{(a)})$ with $a=${s.l}$, $i=${s.i + 1}$.`)}</div>`;
    if (!Sm) { el.append(h('div', {class: 'na'}, 'no smooth.f32 for this kernel')); return; }
    const L = DB.layers(s.view), nL = L.length, a = DB.li(s.view, s.l), pl = h('div', {class: 'plot'}); el.append(pl);
    const tr = TS().map((t, ti) => ({x: L, y: L.map((_, b) => Sm[((a * NM + s.i) * nL + b) * 4 + ti]), name: 't = ' + t, mode: 'lines+markers',
      line: {color: tColor(t), width: 2}, marker: {size: 6}, hovertemplate: `t=${t}, layer %{x}: %{y:.3f}<extra></extra>`}));
    let ymx = 0; for (const q of tr) for (const v of q.y) if (v > ymx) ymx = v; ymx = Math.min(1.02, Math.max(0.05, ymx * 1.12));
    plot(pl, tr, {height: 260, xaxis: {title: {text: 'layer b'}, dtick: 1}, yaxis: {title: {text: 's<sub>b</sub><sup>t</sup>(ψ)'}, range: [0, ymx]}, legend: {y: 1.18},
      shapes: [{type: 'line', x0: s.l, x1: s.l, y0: 0, y1: 1, yref: 'paper', line: {color: '#b04a4a', width: 1, dash: 'dot'}}],
      annotations: [{x: s.l, y: 0, yref: 'paper', text: 'own layer', showarrow: false, font: {size: 10, color: '#b04a4a'}, yanchor: 'bottom', xanchor: 'left', xshift: 3}]});
    el.append(h('div', {class: 'note'}, `Colour = diffusion time t (light → dark = 1 → 64). The dotted line marks the mode's own layer, where s = λ^(2t). y-axis scaled to this mode's largest value (${fmt(ymx / 1.12, 3)}); smoothness is in [0, 1].`));
  } else if (s.kind === 'dmode') {
    await renderDmodeCharts(el, s, gen, true);
  } else if (s.kind === 'state') {
    const bls = [s.l - 1, s.l, s.l + 1].map(l => [l, sameLayerBaseline(s.view, S.k, l, S.r)]).filter(x => x[1]);
    el.innerHTML = `<div class="hd"><span class="t">Flows of state ${s.c} to adjacent layers</span></div><div class="def">${T(FLOW_DEF)}</div>` +
      (bls.length ? `<div><span class="ctlnum warn">same-layer split-half baseline ${T('$H(c^B|c^A)$')}: ${bls.map(([l, b]) => `L${l} ${fmt(b.H_B_given_A, 2)}`).join(' · ')} bits (r = ${S.r})</span></div>` : '');
    const w = h('div', {class: 'alluv'}); el.append(w);
    const L = DB.layers(s.view), i0 = Math.max(0, DB.li(s.view, s.l) - 1), i1 = Math.min(L.length - 1, DB.li(s.view, s.l) + 1);
    await drawAlluvial(w, s.view, L.slice(i0, i1 + 1), {width: Math.max(320, el.getBoundingClientRect().width - 26), height: 300, trace: sd.mask, gen});
  } else if (s.kind === 'neuron') {
    el.innerHTML = `<div class="hd"><span class="t">Write cosines of neuron ${s.l}.${s.n} with mode readout directions</span></div><div class="def">${T(METRIC_DEF.write.tex ? `$${METRIC_DEF.write.tex}$ for the top-16 modes $\\psi_i^{(l)}$ of every layer $l\\ge ${s.l}$ (kernel ${kLabel(S.k)}). Weight-level: where this neuron writes, relative to each mode's ridge readout.` : '')}</div>`;
    const pl = h('div', {class: 'plot'}); el.append(pl);
    const z = [], ys = [];
    for (let l = s.l; l <= 12; l++) { const Wc = await DB.writecos(S.k, l); if (!Wc) continue; ys.push('layer ' + l); z.push(Array.from({length: 16}, (_, i) => Wc[(i * 12 + (s.l - 1)) * NNEUR + s.n])); }
    if (gen !== GEN) return;
    if (!z.length) { pl.innerHTML = '<div class="na">no write cosines stored for this kernel</div>'; return; }
    let mx = 0; for (const r of z) for (const v of r) if (!Number.isNaN(v)) mx = Math.max(mx, Math.abs(v));
    plot(pl, [{type: 'heatmap', z, y: ys, x: Array.from({length: 16}, (_, i) => 'ψ' + (i + 1)), zmin: -mx, zmax: mx, colorscale: DIV_STOPS.map(([p, c]) => [p, c]),
      colorbar: {thickness: 10, len: 0.9}, hovertemplate: '%{y}, %{x}: %{z:.3f}<extra></extra>', xgap: 1, ygap: 1}],
      {height: 60 + 22 * z.length, margin: {l: 64, t: 10}, yaxis: {autorange: 'reversed'}});
    el.append(h('div', {class: 'note'}, `colour range ±${fmt(mx)} (largest |value|). Click a cell column in the objects list to select that mode.`));
    pl.on && pl.on('plotly_click', ev => { const p = ev.points[0]; select({kind: 'mode', view: 'cls', l: +String(p.y).split(' ')[1], i: p.pointIndex[1]}); });
  } else if (s.kind === 'head') {
    el.innerHTML = `<div class="hd"><span class="t">Attention of head ${s.l}.${s.h}: CLS → patches</span></div><div class="def">${T('$a_{\\mathrm{CLS}\\to p}$ over the 16×16 patches for exemplar images (1,500-image display subset), shown as an overlay scaled to its own max; the number is the CLS→CLS weight.')}</div>`;
    const box = h('div'); el.append(box);
    const [ix, A, mxA] = await Promise.all([DB.attnIdx(), DB.attn(s.l), DB.attnMax()]); if (ix) NATT_ = ix.length;
    if (gen !== GEN) return;
    if (!ix || !A) { box.innerHTML = '<div class="na">no attention maps stored</div>'; return; }
    const pos = new Map(Array.from(ix, (v, j) => [v, j]));
    const cand = Array.from(ix).sort((p, q) => sd.vals[q] - sd.vals[p]).slice(0, 12);
    box.innerHTML = `<div class="sl note">exemplars with the largest ${tex('\\|A(x)\\|')} (hover: hide overlay)</div><div class="imgs" style="display:flex;flex-wrap:wrap;gap:3px">` +
      cand.map(i => thumbHTML(i, 76, '<canvas width="16" height="16"></canvas>' + (mxA ? `<span class="tv">cls ${fmt(mxA[((s.l - 1) * NATT() + pos.get(i)) * 12 + s.h * 2 + 1], 2)}</span>` : ''))).join('') + '</div>';
    box.querySelectorAll('.th').forEach(t => { const i = +t.dataset.img, j = pos.get(i), cv = t.querySelector('canvas'), c = cv.getContext('2d'), im = c.createImageData(16, 16);
      for (let p = 0; p < 256; p++) { const u = A[(j * 6 + s.h) * 256 + p] / 255, li = Math.round(u * 255) * 3; im.data[4 * p] = LUT_SEQ[li]; im.data[4 * p + 1] = LUT_SEQ[li + 1]; im.data[4 * p + 2] = LUT_SEQ[li + 2]; im.data[4 * p + 3] = Math.round(40 + 215 * u); }
      c.putImageData(im, 0, 0); cv.style.imageRendering = 'pixelated'; t.onmouseover = () => cv.style.opacity = 0; t.onmouseout = () => cv.style.opacity = ''; });
    wireThumbs(box);
  } else {
    el.innerHTML = `<div class="hd"><span class="t">Label set</span></div><div class="na">Labels validate; nothing on this page is defined by them.</div>`;
  }
}
const SMOOTH_DEF = 'Smoothness of a unit, centred $f$ at layer $l$ and time $t$: $s_l^t(f)=f^\\top R_l^t f=\\|P_l^t f\\|^2\\in[0,1]$, $R_l=P_l^2$.';
const FLOW_DEF = 'Hard coarse states $c_a(x)$ → $c_b(x)$ of the same images; ribbon width = number of images; the highlighted ribbons follow the selected set. Soft memberships are not used for flows.';

/* image inspector */
async function renderInspector(gen) {
  const el = LX.inspector, i = S.img;
  if (i == null) { el.innerHTML = '<div class="hd"><span class="t">Image</span></div><div class="na">Click a point on the UMAP or a thumbnail to inspect an image: its labels (validation only), its value under the selection, and its coarse state at every layer.</div>'; return; }
  const c = LAB.cls[i], su = LAB.sup ? LAB.sup[i] : null, src = await DB.source(), half = await DB.half();
  let val = CUR_SD && CUR_SD.vals ? `<div>value under the selection: <b>${fmt(CUR_SD.vals[i])}</b></div>` : '';
  el.innerHTML = `<div class="hd"><span class="t">Image ${i}</span><span class="r"><button class="kbd" id="img-x">close</button></span></div>` +
    `<div class="flexwrap">${thumbHTML(i, 128)}<div class="small"><div><b>${esc(className(c))}</b> <a href="#" id="img-cls">(select class)</a></div>` +
    (su != null ? `<div class="note">${esc(superName(su))} <a href="#" id="img-sup">(select)</a></div>` : '') +
    `<div class="note">${src ? (src[i] ? 'ImageNet train split' : 'ImageNet val split') : ''}${half ? ` · half ${half[i] ? 'B' : 'A'}` : ''}</div>${val}` +
    `<div style="margin-top:6px" id="img-path" class="note">coarse state at each layer (r = ${S.r}):</div></div></div>`;
  $('#img-x', el).onclick = () => focusImg(i);
  $('#img-cls', el).onclick = e => { e.preventDefault(); select({kind: 'class', c}); };
  if ($('#img-sup', el)) $('#img-sup', el).onclick = e => { e.preventDefault(); select({kind: 'super', s: su}); };
  const path = $('#img-path', el), ri = RS().indexOf(S.r), row = h('div', {class: 'flexwrap', style: {gap: '3px', marginTop: '3px'}});
  for (const l of DB.layers(S.view)) {
    if (!DB.hasSeba(S.view, S.k, l)) continue;
    const hd = await DB.hard(S.view, S.k, l); if (gen !== GEN) return;
    const st = hd[ri * N + i];
    row.append(h('button', {class: 'kbd', title: `layer ${l}: state ${st < 0 ? 'unassigned' : st}`, style: {cursor: st < 0 ? 'default' : 'pointer', fontWeight: l === S.l ? 700 : 400},
      onclick: () => st >= 0 && select({kind: 'state', view: S.view, l, c: st})}, `${l}:${st < 0 ? '–' : st}`));
  }
  path.append(row);
}

function sameLayerBaseline(view, k, l, r) { // flows.json.baseline[r] = [{layer, H_B_given_A, H_A_given_B, ...}]
  const fl = ((((IDX.views[view] || {}).kernels || {})[k] || {}).flows) || {}; const b = (fl.baseline || {})[String(r)];
  return Array.isArray(b) ? b.find(x => x.layer === l) || null : null;
}
