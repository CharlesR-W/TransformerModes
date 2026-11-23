/* ---------- Settings & diagnostics ---------- */
const GX = {built: false, r: '16', tab: 'sel'};
function sparkSVG(a, w = 90, hgt = 20) {
  const v = a.filter(x => typeof x === 'number' && !Number.isNaN(x)); if (v.length < 2) return '';
  const lo = Math.min(...v), hi = Math.max(...v), sx = w / (a.length - 1);
  const pts = a.map((x, i) => typeof x === 'number' ? `${(i * sx).toFixed(1)},${(hgt - 2 - (x - lo) / ((hi - lo) || 1) * (hgt - 4)).toFixed(1)}` : null).filter(Boolean).join(' ');
  return `<svg width="${w}" height="${hgt}" style="vertical-align:middle"><polyline points="${pts}" fill="none" stroke="#2a78d6" stroke-width="1.5"/></svg>`;
}
function jsonHTML(o, depth = 0) {
  if (o == null) return '–';
  if (typeof o === 'number') return fmt(o);
  if (typeof o === 'boolean') return o ? '<span class="status pass">✓ true</span>' : '<span class="status fail">✗ false</span>';
  if (typeof o === 'string') return esc(o);
  if (Array.isArray(o)) {
    if (!o.length) return '[]';
    if (o.every(x => typeof x === 'number' || x == null)) return (o.length > 3 ? sparkSVG(o) + ' ' : '') + `<span class="mono">${o.length <= 16 ? o.map(x => fmt(x)).join(', ') : `${o.length} values, ${fmt(Math.min(...o.filter(x => x != null)))} … ${fmt(Math.max(...o.filter(x => x != null)))}`}</span>`;
    if (o.every(x => x && typeof x === 'object' && !Array.isArray(x))) {
      const keys = [...new Set(o.flatMap(x => Object.keys(x)))];
      return `<table><tr>${keys.map(k => `<th>${esc(k)}</th>`).join('')}</tr>${o.slice(0, 200).map(r => `<tr>${keys.map(k => `<td>${jsonHTML(r[k], depth + 1)}</td>`).join('')}</tr>`).join('')}</table>` + (o.length > 200 ? `<div class="note">${o.length - 200} more rows</div>` : '');
    }
    return o.slice(0, 50).map(x => jsonHTML(x, depth + 1)).join('<br>');
  }
  const rows = Object.entries(o).map(([k, v]) => `<tr><th>${esc(k)}</th><td>${jsonHTML(v, depth + 1)}</td></tr>`).join('');
  const t = `<table>${rows}</table>`;
  return depth > 1 && Object.keys(o).length > 6 ? `<details><summary>${Object.keys(o).length} entries</summary>${t}</details>` : t;
}

function buildDiag() {
  const root = $('#tab-diag'); root.innerHTML = '';
  GX.sel = h('div'); GX.spec = h('div'); GX.seba = h('div'); GX.ctl = h('div'); GX.dec = h('div'); GX.ut = h('div'); GX.mlp = h('div');
  root.append(
    h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Kernel selection: label-free criteria per candidate and layer'), h('span', {class: 's', id: 'gx-sel-s'})), GX.sel),
    h('div', {class: 'two', style: {marginTop: '12px'}},
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Spectrum of P at the current layer')), GX.spec),
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'SEBA diagnostics (coarse states are a measured regime)')), GX.seba)),
    h('div', {class: 'card', style: {marginTop: '12px'}}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Decodability by layer (PLAN §5)')), GX.dec),
    h('div', {class: 'two', style: {marginTop: '12px'}},
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Controls for Δ (PLAN §3)')), GX.ctl),
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Operator unit tests on planted geometry (PLAN §6a)')), GX.ut)),
    h('div', {class: 'card', style: {marginTop: '12px'}}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Trained residual MLP (PLAN §6b, measured, not asserted)')), GX.mlp));
  GX.built = true;
}

async function renderDiag() {
  if (!GX.built) buildDiag();
  const view = S.view, sel = IDX.selection, kids = (META.kernels || []).map(k => k.id), L = DB.layers(view), dflt = (sel && sel.default_kernels) || META.default_kernels || [];
  $('#gx-sel-s').textContent = `${view === 'mp' ? 'mean-patch' : 'CLS'} view · default chosen: ${dflt.map(kLabel).join(', ') || '–'}`;
  // ---- selection criteria small multiples
  GX.sel.innerHTML = '';
  const V = sel && sel.views && sel.views[view];
  if (!V) { GX.sel.innerHTML = '<div class="na">no selection.json for this view</div>'; }
  else {
    const rseg = h('div', {class: 'seg small'});
    for (const r of ['8', '16', '32']) rseg.append(h('button', {class: GX.r === r ? 'active' : '', onclick: () => { GX.r = r; renderDiag(); }}, 'r = ' + r));
    GX.sel.append(h('div', {class: 'flexwrap', style: {alignItems: 'center'}}, h('span', {class: 'ctl'}, h('label', null, 'subspace size for stability'), rseg),
      h('span', {class: 'note', html: 'Lines = kernel candidates (thick = chosen default); x = layer. Rule: the default is the candidate that is stable and non-localised at most layers; if none is, that is the finding. No label-based number enters selection.'})));
    if (sel.rule || sel.reasoning) GX.sel.append(h('div', {class: 'def', html: (sel.rule ? `<b>Rule:</b> ${esc(sel.rule)}<br>` : '') + `<b>Reasoning (compute agent, verbatim):</b> ${esc(sel.reasoning || '–')}` +
      (sel.clear_cut != null ? `<br><b>clear-cut:</b> ${sel.clear_cut ? 'yes' : 'no (both top candidates kept as defaults)'}` : '')}));
    if (sel.scores) {
      const ord = sel.order || Object.keys(sel.scores);
      const rows = ord.filter(k => sel.scores[k]).map(k => { const sc = sel.scores[k];
        return `<tr><td>${kLabel(k)}${dflt.includes(k) ? ' <b>(default)</b>' : ''}</td><td class="num">${sc.score ?? '–'}</td>` +
          (sc.pass || []).map(p => `<td style="text-align:center;color:${p ? '#2f6b45' : '#b04a4a'}">${p ? '✓' : '✗'}</td>`).join('') +
          `<td class="num">${fmt(sc.mean_stab16, 3)}</td><td class="num">${sc.n_stable ?? '–'}</td><td class="num">${sc.n_nonlocal ?? '–'}</td></tr>`; }).join('');
      GX.sel.append(h('details', {open: true}, h('summary', null, 'Rule scores per candidate (pass = stable and non-localised at that layer)'),
        h('div', {html: `<table class="doc"><tr><th>kernel</th><th>score</th>${L.map(l => `<th>${l}</th>`).join('')}<th>mean stab. r=16</th><th># stable</th><th># non-localised</th></tr>${rows}</table>`})));
    }
    const grid = h('div', {class: 'three'}); GX.sel.append(grid);
    const series = (fn) => kids.filter(k => V[k]).map(k => { const pl = V[k].per_layer || []; return {x: L.slice(0, pl.length), y: pl.map(p => { try { const v = fn(p); return v == null || Number.isNaN(v) ? null : v; } catch (e) { return null; } }),
      name: kLabel(k), mode: 'lines+markers', line: {color: KCOL(k), width: dflt.includes(k) ? 3.2 : 1.6}, marker: {size: dflt.includes(k) ? 6 : 4}}; });
    const med = a => { const s = a.filter(x => x != null).slice().sort((p, q) => p - q); return s.length ? s[Math.floor(s.length / 2)] : null; };
    const panels = [
      ['(a) split-half stability', `mean principal cosine between half-A and half-B top-${GX.r} extended subspaces (held-out images); 1 = same subspace`, p => p.stability_mean[GX.r], [0, 1.02], 'linear'],
      ['(a′) worst principal cosine', `smallest of the ${GX.r} principal cosines`, p => { const a = p.stability[GX.r]; return a[a.length - 1]; }, [0, 1.02], 'linear'],
      ['(b) localisation of modes', 'median effective number of images $1/\\sum_x\\psi(x)^4$ over the first 16 modes (log); small = localised on few images', p => med(p.eff_n_modes.slice(0, 16)), null, 'log'],
      ['(b′) localisation of the top-r projector', `effective number for the top-${GX.r} projector (log)`, p => p.eff_n_proj[GX.r], null, 'log'],
      ['(c) negative part of the spectrum', '$\\sum\\lambda_-^2/\\sum\\lambda^2$ of P (K-st need not be PSD; downstream uses $R=P^2$)', p => p.spectrum.neg_frac, null, 'linear'],
      ['(c′) spectral decay', 'number of eigenvalues of P above 0.1', p => p.spectrum.decay_r['0.1'], null, 'linear'],
    ];
    for (const [t, d, fn, rg, ty] of panels) {
      const pl = h('div', {class: 'plot'});
      grid.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}}, t), h('div', {class: 'def', html: T(d)}), pl));
      plot(pl, series(fn), {height: 230, margin: {t: 6, b: 34}, showlegend: t.startsWith('(a) '), legend: {y: -0.25, orientation: 'h'}, xaxis: {dtick: 1, title: {text: 'layer'}}, yaxis: {type: ty, range: rg && ty === 'linear' ? rg : undefined}});
    }
    // alpha=1 comparison and label validation
    const g2 = h('div', {class: 'three', style: {marginTop: '8px'}}); GX.sel.append(g2);
    const po = sel.pca_overlap && sel.pca_overlap[view], pp = h('div', {class: 'plot'});
    g2.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}}, '(d) overlap of modes with PCA'),
      h('div', {class: 'def', html: T('mean principal cosine between the span of the top-16 nonconstant modes of $P_l$ and the span of the view\'s top-16 principal-component scores; 1 = the modes span the same space as PCA (the predecessor\'s other failure mode).') +
        (sel.pca_overlap && sel.pca_overlap.definition ? ` <details><summary>definition (compute agent)</summary>${esc(sel.pca_overlap.definition)}</details>` : '')}), pp));
    if (po) plot(pp, kids.filter(k => po[k]).map(k => ({x: po[k].map(r => r.layer), y: po[k].map(r => r.mean16), name: kLabel(k), mode: 'lines+markers',
      line: {color: KCOL(k), width: dflt.includes(k) ? 3.2 : 1.6}, marker: {size: dflt.includes(k) ? 6 : 4}, showlegend: false,
      customdata: po[k].map(r => r.mean8), hovertemplate: kLabel(k) + ', layer %{x}: mean cos16 %{y:.3f}, mean cos8 %{customdata:.3f}<extra></extra>'})),
      {height: 220, margin: {t: 6}, xaxis: {dtick: 1, title: {text: 'layer'}}, yaxis: {range: [0, 1.02]}});
    else pp.innerHTML = '<div class="na">no pca_overlap block in selection.json</div>';
    const pa = h('div', {class: 'plot'}), pv = h('div', {class: 'plot'});
    g2.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}}, 'Comparison: Coifman–Lafon α = 1 (not usable downstream)'),
      h('div', {class: 'def', html: T(`median effective number of images over modes 1..16: α = 1 operator (dashed) vs Sinkhorn (solid), log scale. Its stationary vector is $\\sqrt{\\text{degree}}$, not uniform, so it cannot be compared or multiplied across layers; split-half stability is not computed for it (DEVIATIONS 9).`)}), pa),
      h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}}, 'Validation only (labels, after the fact)'),
        h('div', {class: 'def', html: 'weighted-kNN class accuracy under each kernel; <b>not used for selection</b>.'}), pv));
    const sa = series(p => med(p.eff_n_modes.slice(0, 16))), sb = series(p => p.alpha1 && p.alpha1.eff_n_modes ? med(p.alpha1.eff_n_modes.slice(0, 16)) : null).map(t => ({...t, line: {...t.line, dash: 'dash'}, name: t.name + ' α=1', showlegend: false}));
    plot(pa, [...sa.map(t => ({...t, showlegend: false})), ...sb], {height: 220, margin: {t: 6}, xaxis: {dtick: 1, title: {text: 'layer'}}, yaxis: {type: 'log'}});
    const val = sel.validation && sel.validation[view];
    if (val) plot(pv, kids.filter(k => val[k]).map(k => ({x: L, y: val[k].knn_acc, name: kLabel(k), mode: 'lines+markers', line: {color: KCOL(k), width: dflt.includes(k) ? 3 : 1.5}, showlegend: false})),
      {height: 220, margin: {t: 6}, xaxis: {dtick: 1, title: {text: 'layer'}}, yaxis: {title: {text: 'kNN accuracy'}, range: [0, 1]}});
    else pv.innerHTML = '<div class="na">no validation block</div>';
  }
  // ---- spectrum at layer S.l
  GX.spec.innerHTML = '';
  if (V) {
    GX.spec.append(defBlock(T(`Eigenvalues $\\lambda_0=1\\ge\\lambda_1\\ge\\dots$ of $P_{${S.l}}$ (first 65) for every candidate; dashed = α = 1 operator of the current kernel. Negative eigenvalues are possible for K-st; downstream objects use $R=P^2$.`)));
    const pl = h('div', {class: 'plot'}); GX.spec.append(pl); const li = DB.li(view, S.l);
    const tr = kids.filter(k => V[k] && V[k].per_layer[li]).map(k => ({y: V[k].per_layer[li].spectrum.lambda, name: kLabel(k), mode: 'lines', line: {color: KCOL(k), width: k === S.k ? 2.6 : 1.3}}));
    const a1 = V[S.k] && V[S.k].per_layer[li] && V[S.k].per_layer[li].alpha1; if (a1) tr.push({y: a1.lambda, name: kLabel(S.k) + ' α=1', mode: 'lines', line: {color: KCOL(S.k), dash: 'dash', width: 1.5}});
    plot(pl, tr, {height: 260, xaxis: {title: {text: 'index'}}, yaxis: {title: {text: 'λ'}}, legend: {y: 1.25}});
    const rows = kids.filter(k => V[k] && V[k].per_layer[li]).map(k => { const s = V[k].per_layer[li].spectrum; return `<tr><td>${kLabel(k)}</td><td class="num">${fmt(s.min)}</td><td class="num">${fmt(s.neg_frac)}</td><td class="num">${s.decay_r ? s.decay_r['0.5'] : '–'}</td><td class="num">${s.decay_r ? s.decay_r['0.1'] : '–'}</td></tr>`; }).join('');
    GX.spec.append(h('div', {html: `<table class="doc"><tr><th>kernel</th><th>min λ</th><th>neg. fraction</th><th>#λ &gt; 0.5</th><th>#λ &gt; 0.1</th></tr>${rows}</table>`}));
  }
  // ---- SEBA diagnostics for S.k, all layers, r = S.r
  const sj = ((((IDX.views[view] || {}).kernels || {})[S.k] || {}).seba);
  if (!sj) GX.seba.innerHTML = `<div class="na">no seba.json for ${kLabel(S.k)}</div>`;
  else {
    const f = ['gap', 'split_half_min_cos', 'split_half_mean_cos', 'dist_from_span', 'neg_mass', 'coverage'];
    const rows = L.map(l => { const pl = perLayer(sj, l, view) || {}; const d = perR(pl.r || pl, S.r); return d ? `<tr><td>${l}</td>${f.map(k => `<td class="num">${fmt(d[k], 3)}</td>`).join('')}<td class="num">${d.sizes ? Math.min(...d.sizes) + '–' + Math.max(...d.sizes) : '–'}</td></tr>` : ''; }).join('');
    GX.seba.innerHTML = `<div class="def">${T(`${kLabel(S.k)}, r = ${S.r}. gap = $\\lambda_{r-1}-\\lambda_r$; split-half = principal cosines at $r-1$; dist = distance of pre-clip SEBA vectors from the input span; neg. mass = share of $L^1$ mass removed by clipping; coverage = share assigned.`)}</div>` +
      `<table class="doc"><tr><th>layer</th><th>gap</th><th>split-half min cos</th><th>split-half mean cos</th><th>dist from span</th><th>neg. mass</th><th>coverage</th><th>set sizes</th></tr>${rows}</table>`;
  }
  // ---- decodability
  GX.dec.innerHTML = '';
  const dj = IDX.decodability, cj = IDX.cifar_decodability;
  GX.dec.append(defBlock(T('Closed-form ridge one-vs-rest (GCV $\\lambda$) and DINO-protocol weighted kNN ($k=20$, $T=0.07$), fit on the 10,000 probe images, tested on the held-out 10,000; fine (100) and superclass labels. Hue = classifier, dash = label granularity.')));
  const g3 = h('div', {class: 'three'}); GX.dec.append(g3);
  const decPlot = (title, d, xs, extra) => { const pl = h('div', {class: 'plot'}); g3.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}, html: title}), pl, extra || ''));
    if (!d) { pl.innerHTML = '<div class="na">not available</div>'; return; }
    const tr = []; for (const [ci, clf] of ['ridge', 'knn'].entries()) for (const lab of ['fine', 'super']) if (d[clf] && d[clf][lab]) tr.push({x: xs || d[clf][lab].map((_, i) => i + 1), y: d[clf][lab], name: `${clf === 'knn' ? 'kNN' : 'ridge'} · ${lab}`, mode: 'lines+markers', line: {color: CAT8[ci], dash: lab === 'super' ? 'dash' : 'solid', width: 2}, marker: {size: 5}});
    plot(pl, tr, {height: 260, xaxis: {title: {text: 'layer'}, dtick: 1}, yaxis: {title: {text: 'held-out accuracy'}, range: [0, 1]}, legend: {y: 1.2}}); };
  decPlot('ViT-S/14 · CLS view', dj && dj.cls, DB.layers('cls'));
  decPlot('ViT-S/14 · mean-patch view', dj && dj.mp, DB.layers('mp'));
  // CIFAR predecessor: {sites, views: {flat|gap: {site: {ridge_acc, knn_acc, knn15_dist, pair_dist, ...}}}, protocol}
  const pc = h('div', {class: 'plot'}), pc2 = h('div', {class: 'plot'});
  g3.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}}, 'Predecessor: ResNet-20 / CIFAR-10, flattened vs spatially averaged block outputs'), pc,
    h('div', {class: 'def', html: T('Distance concentration: 15th-nearest-neighbour cosine distance over the typical pair distance (near 1 = every point equally far, the diagnosed failure of the flattened view).')}), pc2,
    h('div', {class: 'note', html: `${cj && cj.protocol ? esc(cj.protocol) + ' ' : ''}from <span class="mono">${esc((IDX.cifar_path || '').replace(/^.*explore\//, 'explore/'))}</span>`})));
  if (!cj) pc.innerHTML = '<div class="na">CIFAR decodability not available yet (written by another agent).</div>';
  else if (cj.views && cj.sites) {
    const vn = {flat: 'flattened', gap: 'spatially averaged', pooled: 'spatially averaged'}, tr = [], tr2 = [];
    Object.entries(cj.views).forEach(([v, d], vi) => {
      const xs = cj.sites.filter(st => d[st]), c = vi === 0 ? CAT8[1] : CAT8[0];
      tr.push({x: xs, y: xs.map(st => d[st].ridge_acc), name: `${vn[v] || v} · ridge`, mode: 'lines+markers', line: {color: c, width: 2}, marker: {size: 5}});
      tr.push({x: xs, y: xs.map(st => d[st].knn_acc), name: `${vn[v] || v} · kNN`, mode: 'lines+markers', line: {color: c, width: 2, dash: 'dot'}, marker: {size: 5}});
      tr2.push({x: xs, y: xs.map(st => d[st].knn15_dist != null && d[st].pair_dist ? d[st].knn15_dist / d[st].pair_dist : null), name: vn[v] || v, mode: 'lines+markers', line: {color: c, width: 2}, marker: {size: 5}, showlegend: false});
    });
    plot(pc, tr, {height: 260, xaxis: {title: {text: 'site'}, type: 'category'}, yaxis: {title: {text: 'test accuracy'}, range: [0, 1]}, legend: {y: 1.24}});
    plot(pc2, tr2, {height: 170, margin: {t: 6}, xaxis: {type: 'category'}, yaxis: {title: {text: 'd₁₅NN / d_pair'}, range: [0, 1.05]}});
  } else pc.innerHTML = `<div class="json">${jsonHTML(cj)}</div>`;
  // ---- controls (SCHEMA v2: {k: {layers, noise_snr, t, per_layer: [{layer, identical:{max_abs_eig[t]}, rotation, noise, adjacent_*}]}})
  const cj2 = IDX.controls, ti = TS().indexOf(S.t);
  let ch = `<div class="def">${T('Max $|\\text{eigenvalue}|$ of $\\Delta_t$ between a layer\'s operator and the operator of a modified copy of the same view: identical (must be 0 to numerical precision), randomly rotated (must be 0: only the dot product enters), and after an isotropic perturbation at SNR 10 (noise power 0.1, RMS about 32% of signal; how much $\\Delta$ a perturbation that is not computation produces), tested at layers 1, 6, 12, next to the adjacent-layer $\\Delta$ for scale. t = ' + S.t + ' (global control).')}</div>`;
  if (!cj2) ch += '<div class="na">no controls.json</div>';
  else {
    const rows = [];
    for (const [k, c] of Object.entries(cj2)) for (const r of (c && c.per_layer) || []) {
      const g = nm => r[nm] && r[nm].max_abs_eig ? r[nm].max_abs_eig[ti] : null;
      const adj = Object.entries(r).filter(([kk]) => kk.startsWith('adjacent_')).map(([kk, v]) => `${kk.replace('adjacent_', '').replace('_max_abs_eig', '')}: ${fmt(Array.isArray(v) ? v[ti] : v)}`).join('<br>');
      rows.push(`<tr><td>${kLabel(k)}</td><td class="num">${r.layer}</td><td class="num">${fmt(g('identical'))}</td><td class="num">${fmt(g('rotation'))}</td><td class="num">${fmt(g('noise'))}</td><td class="num">${adj || '–'}</td></tr>`);
    }
    const snr = Object.values(cj2)[0] && Object.values(cj2)[0].noise_snr;
    ch += rows.length ? `<table class="doc"><tr><th>kernel</th><th>layer</th><th>identical</th><th>rotation</th><th>noise${snr ? ` (SNR ${fmt(snr)})` : ''}</th><th>adjacent-layer Δ (reference)</th></tr>${rows.join('')}</table>`
      : `<div class="json">${jsonHTML(cj2)}</div>`;
  }
  const rec = META.reconstruction, rvh = META.readout_vs_hub;
  if (rec || rvh) ch += `<details><summary>Exactness of the component decomposition</summary><div class="def">Reconstruction of $v_l$ from the sum of components (asserted before attribution) and deviation of the float64 forward from the hub model's readout (DEVIATIONS 5).</div><div class="json">${jsonHTML({reconstruction: rec, readout_vs_hub: rvh})}</div></details>`.replace(/\$(.+?)\$/g, (_, m) => tex(m));
  const us = sel && sel.uncentred_sensitivity;
  if (us) ch += `<details><summary>Sensitivity: uncentred view</summary><div class="json">${jsonHTML(us)}</div></details>`;
  GX.ctl.innerHTML = ch;
  // ---- unit tests (format of the §6a agent: {config, checks: [{name, kernel, t, value, threshold, relation, passed, note}], measured, n_checks, n_failed, all_passed})
  const ut = IDX.unit_tests;
  if (!ut) GX.ut.innerHTML = '<div class="na">unit_tests.json not available yet (written by another agent).</div>';
  else {
    const list = Array.isArray(ut) ? ut : Array.isArray(ut.checks) ? ut.checks : Array.isArray(ut.tests) ? ut.tests : null;
    let u = `<div class="def">${T('Point clouds on a sphere built by hand (64 fine groups, map $y:64\\to8$). Thresholds fixed in PLAN §6a before running; failures are reported and the object reconsidered.')}</div>`;
    if (ut.n_checks != null) u += `<div style="margin:4px 0">${ut.all_passed ? '<span class="status pass">✓ all passed</span>' : `<span class="status fail">✗ ${ut.n_failed} failed</span>`} <span class="note">${ut.n_checks} checks</span></div>`;
    if (list) {
      const failed = list.filter(t => (t.passed ?? t.pass) === false);
      const row = t => { const ok = t.passed ?? t.pass ?? t.ok; return `<tr><td>${esc(t.name ?? t.id ?? '')}${t.note ? `<div class="note">${esc(t.note)}</div>` : ''}</td><td>${esc(t.kernel ?? '')}</td><td class="num">${t.t ?? ''}</td>` +
        `<td class="num">${typeof (t.value ?? t.measured) === 'number' ? fmt(t.value ?? t.measured) : jsonHTML(t.value ?? t.measured)}</td><td class="num">${esc(t.relation ?? '')} ${typeof t.threshold === 'number' ? fmt(t.threshold) : esc(JSON.stringify(t.threshold ?? ''))}</td>` +
        `<td>${ok == null ? '–' : ok ? '<span class="status pass">✓ pass</span>' : '<span class="status fail">✗ fail</span>'}</td></tr>`; };
      const head = '<tr><th>check</th><th>kernel</th><th>t</th><th>value</th><th>threshold</th><th>result</th></tr>';
      if (failed.length) u += `<div class="small" style="font-weight:650;margin-top:4px">Failed checks</div><table class="doc">${head}${failed.map(row).join('')}</table>`;
      u += `<details ${failed.length ? '' : 'open'}><summary>All ${list.length} checks</summary><div style="max-height:420px;overflow:auto"><table class="doc">${head}${list.map(row).join('')}</table></div></details>`;
      if (ut.measured) u += `<details><summary>Measured quantities (reported, not asserted)</summary><div class="json" style="max-height:420px;overflow:auto">${jsonHTML(ut.measured)}</div></details>`;
      if (ut.config) u += `<details><summary>Configuration</summary><div class="json">${jsonHTML(ut.config)}</div></details>`;
    } else u += `<div class="json">${jsonHTML(ut)}</div>`;
    GX.ut.innerHTML = u;
  }
  // ---- MLP (format of the §6b agent)
  renderMLP(GX.mlp, IDX.mlp);
}

function renderMLP(el, mj) {
  el.innerHTML = `<div class="def">4 residual blocks, width 256, trained on ${tex('y=(A+B)\\bmod 8')} with ${tex('x=E_A[A]+E_B[B]+\\nu')} (Gaussian nuisance). Here the view is the complete state. No recovery threshold: the network may keep A and B, and the result is reported as found.</div>`;
  if (!mj) { el.innerHTML += '<div class="na">mlp.json not available yet (written by another agent).</div>'; return; }
  const cfg = mj.config || {}, tr = mj.training || {};
  el.innerHTML += `<div class="note">${esc(cfg.task || '')} · ${esc(cfg.train || '')} · probe N = ${cfg.N_probe ?? '?'} · held-out y accuracy ${fmt(tr.heldout_y_acc, 3)}</div>`;
  const views = cfg.views || Object.keys(mj.decodability || {});
  const g = h('div', {class: 'three'}); el.append(g);
  // decodability per view
  const p1 = h('div', {class: 'plot'});
  g.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}}, 'Decodability per view'), h('div', {class: 'def', html: 'held-out accuracy; solid = ridge on the cosine view, dotted = DINO-protocol kNN. Hue = target.'}), p1));
  const tg = ['A', 'B', 'y', 'AB'], dec = mj.decodability || {}, traces = [];
  tg.forEach((t, i) => { if (!views.some(v => dec[v] && dec[v][t])) return;
    traces.push({x: views, y: views.map(v => dec[v] && dec[v][t] ? dec[v][t].ridge_cos_view : null), name: t + ' ridge', mode: 'lines+markers', line: {color: CAT8[i], width: 2}});
    traces.push({x: views, y: views.map(v => dec[v] && dec[v][t] ? dec[v][t].knn_dino : null), name: t + ' kNN', mode: 'lines+markers', line: {color: CAT8[i], width: 1.5, dash: 'dot'}, showlegend: false}); });
  plot(p1, traces, {height: 250, xaxis: {type: 'category', title: {text: 'view'}}, yaxis: {range: [0, 1.05], title: {text: 'accuracy'}}, legend: {y: 1.25}});
  // smoothness of A, B, y contrasts per view at time t
  const ks = Object.keys(mj.kernels || {}), k0 = ks[0];
  const p2 = h('div', {class: 'plot'}), kseg = h('div', {class: 'seg small'});
  GX.mk = ks.includes(GX.mk) ? GX.mk : k0;
  for (const k of ks) kseg.append(h('button', {class: k === GX.mk ? 'active' : '', onclick: () => { GX.mk = k; renderMLP(el, mj); }}, (cfg.kernels || {})[k] || k));
  g.append(h('div', null, h('div', {class: 'flexwrap', style: {alignItems: 'center'}}, h('span', {style: {fontWeight: 650, fontSize: '13px'}}, 'Smoothness of the planted contrasts'), kseg),
    h('div', {class: 'def', html: T(`mean $s^t(f)$ over the contrasts of A, B and y (band: min–max), t = ${S.t} (global control; nearest stored t used).`)}), p2));
  const K = (mj.kernels || {})[GX.mk] || {}, sm = K.smoothness || {};
  const tt = (cfg.t || [1, 4, 16, 64]).reduce((b, x) => Math.abs(x - S.t) < Math.abs(b - S.t) ? x : b, (cfg.t || [S.t])[0]);
  const vs = views.filter(v => sm[v]), t2 = [];
  ['A', 'B', 'y'].forEach((t, i) => { const get = (v, f) => { const o = sm[v] && sm[v][String(tt)] && sm[v][String(tt)][t]; return o ? o[f] : null; };
    t2.push({x: vs, y: vs.map(v => get(v, 'max')), mode: 'lines', line: {width: 0}, showlegend: false, hoverinfo: 'skip'});
    t2.push({x: vs, y: vs.map(v => get(v, 'min')), mode: 'lines', line: {width: 0}, fill: 'tonexty', fillcolor: CAT8[i] + '22', showlegend: false, hoverinfo: 'skip'});
    t2.push({x: vs, y: vs.map(v => get(v, 'mean')), name: t, mode: 'lines+markers', line: {color: CAT8[i], width: 2}}); });
  plot(p2, t2, {height: 250, xaxis: {type: 'category', title: {text: 'view'}}, yaxis: {range: [0, 1.02], title: {text: 's^t'}}, legend: {y: 1.2}});
  // Delta from input + SEBA + flows tables
  const p3 = h('div');
  g.append(h('div', null, h('div', {style: {fontWeight: 650, fontSize: '13px'}, html: T('$\\Delta_t^{x\\to h}$ from the input')}), p3));
  const dfi = K.delta_from_input || {};
  p3.innerHTML = `<table class="doc"><tr><th>view</th><th>top + eig.</th><th>most − eig.</th><th># + &gt; 0.05</th><th># − &lt; −0.05</th></tr>` +
    Object.entries(dfi).map(([v, o]) => { const r = o[String(tt)] || {}; return `<tr><td>${esc(v)}</td><td class="num">${fmt((r.pos_vals || [])[0])}</td><td class="num">${fmt((r.neg_vals || [])[0])}</td><td class="num">${r['n_pos_gt_0.05'] ?? '–'}</td><td class="num">${r['n_neg_lt_-0.05'] ?? '–'}</td></tr>`; }).join('') + `</table><div class="note">t = ${tt}, ${esc((cfg.kernels || {})[GX.mk] || GX.mk)}</div>`;
  const sb = K.seba || {}, fl = K.flows || {};
  const rs = Object.keys(sb);
  let tb = '';
  for (const r of rs) tb += `<div style="font-weight:650;font-size:12.5px;margin-top:6px">SEBA, r = ${r}</div><table class="doc"><tr><th>view</th><th>coverage</th><th>states used</th><th>H(A | state)</th><th>H(B | state)</th><th>gap</th><th>neg. mass max</th></tr>` +
    Object.entries(sb[r]).map(([v, o]) => `<tr><td>${esc(v)}</td><td class="num">${fmt(o.coverage, 3)}</td><td class="num">${o.n_states_used ?? '–'}</td><td class="num">${fmt(o.H_A_given_state, 3)}</td><td class="num">${fmt(o.H_B_given_state, 3)}</td><td class="num">${fmt(o.gap)}</td><td class="num">${fmt(o.neg_mass_max, 3)}</td></tr>`).join('') + '</table>';
  for (const r of Object.keys(fl)) tb += `<div style="font-weight:650;font-size:12.5px;margin-top:6px">Hard-state flows, r = ${r} (bits)</div><table class="doc"><tr><th>a → b</th><th>H(c_b | c_a)</th><th>H(c_a | c_b)</th><th>MI</th></tr>` +
    Object.entries(fl[r]).map(([pr, o]) => `<tr><td>${esc(pr)}</td><td class="num">${fmt(o.H_b_given_a, 3)}</td><td class="num">${fmt(o.H_a_given_b, 3)}</td><td class="num">${fmt(o.MI, 3)}</td></tr>`).join('') + '</table>';
  el.append(h('div', {class: 'two', style: {marginTop: '8px'}}, h('div', {html: tb.split('Hard-state flows')[0]}), h('div', {html: tb.includes('Hard-state flows') ? '<div style="font-weight:650;font-size:12.5px;margin-top:6px">Hard-state flows' + tb.split('Hard-state flows').slice(1).join('Hard-state flows') : ''})));
  el.append(h('details', null, h('summary', null, 'raw mlp.json'), h('div', {class: 'json', style: {maxHeight: '420px', overflow: 'auto'}, html: jsonHTML(mj)})));
}
