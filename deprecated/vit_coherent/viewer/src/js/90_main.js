/* ---------- main: init, control bar, routing ---------- */
function renderCtls() {
  const el = $('#ctls'); el.innerHTML = '';
  const vs = h('div', {class: 'seg small'});
  for (const [v, lab] of [['cls', 'CLS'], ['mp', 'mean-patch']]) vs.append(h('button', {class: S.view === v ? 'active' : '', disabled: !IDX.views[v], title: v === 'mp' ? 'secondary view (variant)' : 'primary view',
    onclick: () => { const ks = kernelsFor(v); setS({view: v, k: ks.includes(S.k) ? S.k : ks[0], sel: null}); }}, lab));
  const ks = h('select', {onchange: e => setS({k: e.target.value})});
  const dflt = META.default_kernels || [];
  for (const k of kernelsFor(S.view)) ks.append(h('option', {value: k, selected: k === S.k}, kLabel(k) + (dflt.includes(k) ? ' (default)' : '') + (kFull(k) ? '' : ' (reduced set)')));
  const L = DB.layers(S.view), ld = h('div', {class: 'layerdots'});
  for (const l of L) ld.append(h('button', {class: l === S.l ? 'active' : '', onclick: () => setS({l}), title: 'layer ' + l + ' (← → keys)'}, String(l)));
  const play = h('button', {class: 'kbd', title: 'step through all layers', onclick: () => {
    if (PLAYT) { clearInterval(PLAYT); PLAYT = null; renderCtls(); return; }
    PLAYT = setInterval(() => { const i = L.indexOf(S.l); setS({l: L[(i + 1) % L.length]}); }, 1100); renderCtls(); }}, PLAYT ? '❚❚' : '▶');
  const ts = h('div', {class: 'seg small'});
  for (const t of TS()) { const st = tState(t);
    const why = st ? `largest nonconstant λ of P over layers = ${fmt(st.m, 3)} for ${kLabel(S.k)}, so λ^(2t) = ${fmt(st.v)}: ` +
      (st.kind === 'degenerate' ? 'numerically degenerate; smoothness and Δ ≈ 0 and eigenvectors are numerical noise' : 'strongly attenuated; values are small, check the cross-validated values') : `diffusion time t = ${t}`;
    ts.append(h('button', {class: (t === S.t ? 'active' : '') + (st ? ' ' + (st.kind === 'degenerate' ? 'degen' : 'atten') : ''), onclick: () => setS({t}), title: why}, String(t))); }
  const rs = h('div', {class: 'seg small'}); for (const r of RS()) rs.append(h('button', {class: r === S.r ? 'active' : '', onclick: () => setS({r})}, String(r)));
  el.append(...[h('span', {class: 'ctl'}, h('label', null, 'operator'), vs, ks),
    h('span', {class: 'ctl'}, h('label', null, 'layer'), ld, play),
    h('span', {class: 'ctl', title: 'diffusion time: R^t = P^{2t}'}, h('label', {html: tex('t')}), ts),
    h('span', {class: 'ctl', title: 'number of SEBA coarse states'}, h('label', {html: tex('r')}), rs),
    tState(S.t) ? h('span', {class: 'caveat', style: {margin: 0}, html: tState(S.t).kind === 'degenerate'
      ? `t = ${S.t} is numerically degenerate for ${kLabel(S.k)}: max ${tex('\\lambda^{2t}')} = ${fmt(tState(S.t).v)}; values ≈ 0, eigenvectors are noise`
      : `t = ${S.t} is strongly attenuated for ${kLabel(S.k)}: max ${tex('\\lambda^{2t}')} = ${fmt(tState(S.t).v)}; values are small, check the cross-validated values`}) : null].filter(Boolean));
  // selection chip
  const sc = $('#selchip'); sc.innerHTML = '';
  if (S.sel) sc.append(h('span', {class: 'what', html: selName(S.sel, true) + (S.sel.l != null && S.sel.kind !== 'neuron' && S.sel.kind !== 'head' && S.sel.kind !== 'mode' ? '' : '')}),
    h('button', {disabled: !HIST.length, onclick: back, title: 'previous selection (Backspace)'}, '← back'), h('button', {onclick: () => select(null), title: 'Esc'}, 'clear'));
  else sc.append(h('span', {class: 'note'}, 'nothing selected'));
  document.querySelectorAll('#tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === S.tab));
  document.querySelectorAll('section.tab').forEach(s => s.classList.toggle('active', s.dataset.tab === S.tab));
}
let PLAYT = null;

async function renderAll() {
  renderCtls();
  try {
    if (S.tab === 'layer') await renderLayer();
    else if (S.tab === 'block') await renderBlock();
    else if (S.tab === 'depth') await renderDepth();
    else if (S.tab === 'diag') await renderDiag();
    else if (S.tab === 'defs') renderDefs();
  } catch (e) { console.error(e); }
}

function header() {
  const m = META, dflt = m.default_kernels || [];
  $('#title').textContent = (IDX.mock ? 'MOCKUP · ' : '') + 'Smoothness across depth in a pretrained ViT (DINOv2 ViT-S/14)';
  if (IDX.mock) $('#mockbar').hidden = false;
  const pills = [
    IDX.mock ? ['MOCKUP data', 'mock'] : null,
    [`${esc(m.model || 'dinov2_vits14')}, frozen; 12 blocks, width 384, 6 heads, MLP 1536`, ''],
    [`N = ${fmtInt(m.N || N)} probe images (${esc(m.dataset || 'ImageNet-100')})`, ''],
    ['views: CLS (primary), mean-patch: partial views', 'warn'],
    [`operator: Sinkhorn P = DKD, R = P²; default kernel ${dflt.map(kLabel).join(', ') || '?'} (label-free selection)`, ''],
    ['exploratory side study: no claim enters THEORY.md', 'warn'],
  ].filter(Boolean);
  $('#pills').innerHTML = pills.map(([t, c]) => `<span class="pill ${c}">${t}</span>`).join('');
  $('#titleside').innerHTML = `Keys: <span class="kbd">←</span><span class="kbd">→</span> layer · <span class="kbd">↑</span><span class="kbd">↓</span> next mode/state · <span class="kbd">Backspace</span> back · <span class="kbd">Esc</span> clear.<br>Built ${esc(IDX.built)} from <span class="mono">data/${esc(IDX.tag)}</span>.`;
  const v = $('#verdict');
  if (IDX.verdict_html) {
    let open = false; try { open = localStorage.getItem('vc-verdict-open') === '1'; } catch (e) {}
    v.className = 'verdict' + (open ? '' : ' collapsed');
    v.innerHTML = `<div class="vhd"><span class="label">Verdict (orchestrator)</span><button class="kbd" id="vtog">${open ? 'collapse' : 'show full verdict'}</button></div><div class="vbody">${T(IDX.verdict_html)}</div>`;
    $('#vtog').onclick = () => { const o = v.classList.toggle('collapsed'); $('#vtog').textContent = o ? 'show full verdict' : 'collapse'; try { localStorage.setItem('vc-verdict-open', o ? '0' : '1'); } catch (e) {} };
  }
  else { v.className = 'verdict empty'; v.innerHTML = '<span class="label">Verdict</span> · not written yet. This slot is filled from <span class="mono">explore/vit_coherent/viewer/verdict.md</span> when the page is built.'; }
  $('#foot').innerHTML = `Data: <span class="mono">explore/vit_coherent/data/${esc(IDX.tag)}</span> (SCHEMA.md v2). Math by KaTeX, charts by Plotly (both vendored); UMAP and component grids are canvas. ` +
    (IDX.notes && IDX.notes.length ? `<details><summary>${IDX.notes.length} build notes (missing or skipped pieces)</summary><ul>${IDX.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul></details>` : 'No missing files at build time.');
}

function keys(e) {
  if (e.target.closest('input,select,textarea')) return;
  const L = DB.layers(S.view);
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { const i = L.indexOf(S.l) + (e.key === 'ArrowRight' ? 1 : -1); if (i >= 0 && i < L.length) setS({l: L[i]}); e.preventDefault(); }
  else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    const d = e.key === 'ArrowDown' ? 1 : -1, s = S.sel;
    if (s && s.kind === 'mode') { const i = clamp(s.i + d, 0, NM - 1); S.sel = {...s, i}; writeHash(); invalidate(); e.preventDefault(); }
    else if (s && s.kind === 'state') { const c = clamp(s.c + d, 0, S.r - 1); S.sel = {...s, c}; writeHash(); invalidate(); e.preventDefault(); }
    else if (s && s.kind === 'dmode') { const j = clamp(s.j + d, 0, 15); S.sel = {...s, j}; writeHash(); invalidate(); e.preventDefault(); }
    else if (s && s.kind === 'neuron') { const n = clamp(s.n + d, 0, NNEUR - 1); S.sel = {...s, n}; writeHash(); invalidate(); e.preventDefault(); }
  }
  else if (e.key === 'Backspace') { back(); e.preventDefault(); }
  else if (e.key === 'Escape') { if (S.img != null) focusImg(S.img); else select(null); }
}

(async function init() {
  try {
    IDX = await fetch('d/index.json').then(r => r.json());
  } catch (e) { document.querySelector('main').innerHTML = `<div class="card err">Cannot load d/index.json (${esc(e.message)}). Open this page over HTTP (the laptop page server or <span class="mono">python3 -m http.server</span>), not file://.</div>`; return; }
  META = IDX.meta; N = META.N || 10000;
  const dflt = (IDX.selection && IDX.selection.default_kernels) || META.default_kernels || META.kernels_full || [];
  if (!META.default_kernels) META.default_kernels = dflt;
  S.k = dflt[0] || kernelsFor('cls')[0];
  if (!IDX.views.cls && IDX.views.mp) S.view = 'mp';
  readHash();
  if (!kernelsFor(S.view).includes(S.k)) S.k = kernelsFor(S.view)[0];
  LAB = await DB.labels();
  header();
  document.querySelectorAll('#tabs button').forEach(b => b.onclick = () => { S.tab = b.dataset.tab; writeHash(); invalidate(); window.scrollTo({top: 0}); });
  addEventListener('keydown', keys);
  addEventListener('hashchange', () => { const before = location.hash; readHash(); invalidate(); });
  addEventListener('resize', debounce(() => invalidate(), 200));
  setS({});
})();
