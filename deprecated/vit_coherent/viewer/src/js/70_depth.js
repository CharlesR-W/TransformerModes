/* ---------- Depth: alluvial flows of hard coarse states, entropies, smoothness profiles ---------- */
const DX = {built: false};
async function hardLabels(view, layers) {
  const ri = RS().indexOf(S.r), out = [];
  for (const l of layers) { const hd = DB.hasSeba(view, S.k, l) ? await DB.hard(view, S.k, l) : null; out.push(hd ? hd.subarray(ri * N, (ri + 1) * N) : null); }
  return out;
}
async function drawAlluvial(wrap, view, layers, o) {
  const labs = await hardLabels(view, layers); if (o.gen != null && o.gen !== GEN) return;
  if (labs.some(x => !x)) { wrap.innerHTML = '<div class="na">no hard states for some of these layers (only kernels_full have SEBA states)</div>'; return; }
  const r = S.r, U = r, K = r + 1, W = o.width, Hh = o.height, nc = layers.length, nodeW = Math.min(14, Math.max(8, W / nc / 8)), padT = 18, padB = 16;
  const cnt = labs.map(lab => { const c = new Array(K).fill(0); for (let i = 0; i < N; i++) c[lab[i] < 0 ? U : lab[i]]++; return c; });
  const tr = o.trace ? labs.map(lab => { const c = new Array(K).fill(0); for (let i = 0; i < N; i++) if (o.trace[i]) c[lab[i] < 0 ? U : lab[i]]++; return c; }) : null;
  const flows = []; for (let t = 0; t < nc - 1; t++) { const F = new Map(), A = labs[t], B = labs[t + 1], Ft = new Map();
    for (let i = 0; i < N; i++) { const a = A[i] < 0 ? U : A[i], b = B[i] < 0 ? U : B[i], key = a * K + b; F.set(key, (F.get(key) || 0) + 1); if (o.trace && o.trace[i]) Ft.set(key, (Ft.get(key) || 0) + 1); }
    flows.push({F, Ft}); }
  // ordering: column 0 by state id; then barycentre of incoming flows, forward then backward
  const order = [Array.from({length: r}, (_, c) => c)];
  for (let t = 1; t < nc; t++) {
    const pos = new Map(order[t - 1].map((c, j) => [c, j])), bary = new Array(r).fill(0), wsum = new Array(r).fill(0);
    for (const [key, v] of flows[t - 1].F) { const a = Math.floor(key / K), b = key % K; if (a === U || b === U) continue; bary[b] += v * pos.get(a); wsum[b] += v; }
    order.push(Array.from({length: r}, (_, c) => c).sort((x, y) => (wsum[x] ? bary[x] / wsum[x] : 1e9) - (wsum[y] ? bary[y] / wsum[y] : 1e9)));
  }
  const gap = Math.min(4, Math.max(0.6, 0.22 * (Hh - padT - padB) / K)), avail = Hh - padT - padB - gap * K, k = avail / N;
  const mL = 30, mR = 30, colX = t => nc === 1 ? W / 2 : mL + t * (W - mL - mR - nodeW) / (nc - 1);
  const nodeY = []; // nodeY[t][state] = y0
  for (let t = 0; t < nc; t++) { const y = {}; let acc = padT; for (const c of [...order[t], U]) { y[c] = acc; acc += cnt[t][c] * k + gap; } nodeY.push(y); }
  const svgNS = 'http://www.w3.org/2000/svg';
  let s = `<svg xmlns="${svgNS}" width="${W}" height="${Hh}">`;
  // ribbons
  const minC = Math.max(1, Math.round(N * (o.minShare || 0.001)));
  for (let t = 0; t < nc - 1; t++) {
    const x0 = colX(t) + nodeW, x1 = colX(t + 1), xm = (x0 + x1) / 2, outOff = {}, inOff = {};
    const items = [...flows[t].F.entries()].map(([key, v]) => ({a: Math.floor(key / K), b: key % K, v, vt: flows[t].Ft.get(key) || 0}));
    const posA = c => nodeY[t][c], posB = c => nodeY[t + 1][c];
    items.sort((p, q) => posA(p.a) - posA(q.a) || posB(p.b) - posB(q.b));
    for (const it of items) { it.ya = posA(it.a) + (outOff[it.a] || 0); outOff[it.a] = (outOff[it.a] || 0) + it.v * k; }
    items.sort((p, q) => posB(p.b) - posB(q.b) || posA(p.a) - posA(q.a));
    for (const it of items) { it.yb = posB(it.b) + (inOff[it.b] || 0); inOff[it.b] = (inOff[it.b] || 0) + it.v * k; }
    const band = (ya, yb, hgt, fill, op, extra = '') => `<path d="M${x0},${ya}C${xm},${ya} ${xm},${yb} ${x1},${yb}L${x1},${yb + hgt}C${xm},${yb + hgt} ${xm},${ya + hgt} ${x0},${ya + hgt}Z" fill="${fill}" fill-opacity="${op}" ${extra}/>`;
    for (const it of items) { if (it.v < minC) continue;
      const un = it.a === U || it.b === U;
      s += band(it.ya, it.yb, it.v * k, un ? '#d9d6cf' : '#9ab7d3', tr ? 0.28 : 0.45, `data-f="${t},${it.a},${it.b},${it.v},${it.vt}"`); }
    if (tr) for (const it of items) { if (!it.vt) continue; s += band(it.ya, it.yb, it.vt * k, '#2a78d6', 0.85, `data-f="${t},${it.a},${it.b},${it.v},${it.vt}"`); }
  }
  // nodes
  for (let t = 0; t < nc; t++) {
    const x = colX(t);
    s += `<text x="${x + nodeW / 2}" y="11" text-anchor="middle" font-size="11" fill="#52514e" font-weight="600">layer ${layers[t]}</text>`;
    const bl = o.baseline !== false ? sameLayerBaseline(view, S.k, layers[t], S.r) : null;
    if (bl) s += `<text x="${x + nodeW / 2}" y="${Hh - 1}" text-anchor="middle" font-size="9.5" fill="#8a4a2a"><title>same-layer split-half baseline H(c^B|c^A), bits</title>base ${fmt(bl.H_B_given_A, 2)}</text>`;
    for (const c of [...order[t], U]) {
      const y = nodeY[t][c], hgt = cnt[t][c] * k; if (hgt <= 0) continue;
      const sel = S.sel && S.sel.kind === 'state' && S.sel.l === layers[t] && S.sel.c === c;
      s += `<g class="node" data-n="${t},${c}"><rect x="${x}" y="${y}" width="${nodeW}" height="${Math.max(0.8, hgt)}" fill="${c === U ? '#ebe8e2' : '#52708f'}" ${sel ? 'stroke="#0b0b0b" stroke-width="2"' : ''}/>`;
      if (tr && tr[t][c]) s += `<rect x="${x}" y="${y}" width="${nodeW}" height="${tr[t][c] * k}" fill="#2a78d6"/>`;
      if (hgt > 11 && c !== U) s += `<text x="${x + nodeW + 2}" y="${y + Math.min(hgt / 2, 9) + 3}" font-size="9.5" fill="#2b2a27">${c}</text>`;
      s += '</g>';
    }
  }
  s += '</svg>';
  wrap.innerHTML = s;
  wrap.querySelectorAll('.node').forEach(g => { const [t, c] = g.dataset.n.split(',').map(Number);
    g.onmousemove = e => TIP.show(`<b>layer ${layers[t]}, ${c === U ? 'unassigned' : 'state ' + c}</b><br>${fmtInt(cnt[t][c])} images` + (tr ? `<br>of the selected set: ${fmtInt(tr[t][c])}` : '') + (c === U ? '' : '<br><span class="k">click: select this state</span>'), e.clientX, e.clientY);
    g.onmouseleave = () => TIP.hide();
    g.onclick = () => { if (c !== U) select({kind: 'state', view, l: layers[t], c}); }; });
  wrap.querySelectorAll('path[data-f]').forEach(p => { const [t, a, b, v, vt] = p.dataset.f.split(',').map(Number);
    p.onmousemove = e => TIP.show(`layer ${layers[t]} ${a === U ? 'unassigned' : 'state ' + a} → layer ${layers[t + 1]} ${b === U ? 'unassigned' : 'state ' + b}<br><b>${fmtInt(v)}</b> images (${fmt(100 * v / cnt[t][a], 1)}% of the source)` + (tr ? `<br>selected set: ${fmtInt(vt)}` : ''), e.clientX, e.clientY);
    p.onmouseleave = () => TIP.hide(); });
}

function buildDepth() {
  const root = $('#tab-depth'); root.innerHTML = '';
  DX.alluv = h('div', {class: 'alluv'}); DX.alluvNote = h('div', {class: 'legend'});
  DX.ent = h('div', {class: 'plot'}); DX.fm = h('div', {class: 'plot'}); DX.fmHd = h('div', {class: 'def'});
  DX.sp = h('div', {class: 'plot'}); DX.ov = h('div', {class: 'plot'}); DX.spHd = h('div', {class: 'def'});
  root.append(
    h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Coarse-state flows across depth'), h('span', {class: 's', id: 'dx-al-s'})), defBlock(T(FLOW_DEF + ' Nodes ordered by the barycentre of their incoming flows; the grey node at the bottom of each column is "unassigned"; ribbons below 0.1% of N are not drawn. Under each column: the same-layer split-half baseline $H(c^B\\mid c^A)$ in bits (assignment noise; compare with the entropies below).')), DX.alluvNote, DX.alluv),
    h('div', {class: 'two', style: {marginTop: '12px'}},
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Conditional entropies between adjacent layers')),
        defBlock(T('$H(c_{l+1}\\mid c_l)$ and $H(c_l\\mid c_{l+1})$ in bits over hard states (unassigned counted as a state), next to the same-layer split-half baseline $H(c^B\\mid c^A)$: assignments from half-A-built vs half-B-built operators compared on held-out images, so assignment noise is not read as depth change.')), DX.ent),
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Flow matrix a → b')), DX.fmHd, DX.fm)),
    h('div', {class: 'two', style: {marginTop: '12px'}},
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Smoothness profiles of one layer\'s modes')), DX.spHd, DX.sp),
      h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Overview: leading distinctions of every layer')),
        defBlock(T('Cell $(a,b)$: mean of $S[a,i,b,t]=s_b^t(\\psi_i^{(a)})$ over the leading 8 modes $i$ of layer $a$. Read a row: at which depths layer $a$\'s leading distinctions are smooth. Click a row to show its profiles on the left.')), DX.ov)));
  DX.built = true;
}

async function renderDepth() {
  if (!DX.built) buildDepth();
  const gen = GEN, view = S.view, L = DB.layers(view);
  $('#dx-al-s').textContent = `${kLabel(S.k)} · r = ${S.r} · ${view === 'mp' ? 'mean-patch' : 'CLS'} · all layers`;
  const sd = await selData(S.sel); if (gen !== GEN) return;
  const trace = sd && sd.mask ? sd.mask : null;
  DX.alluvNote.innerHTML = trace ? `<span><span class="sw" style="background:#2a78d6"></span>followed set: ${selName(S.sel)} (${fmtInt(trace.reduce((a, b) => a + b, 0))} images)</span>` :
    '<span class="note">Select a coarse state, class or superclass to follow its images through all layers (blue). Click a node to select that state.</span>';
  await drawAlluvial(DX.alluv, view, L, {width: Math.max(600, DX.alluv.getBoundingClientRect().width || 1200), height: S.r > 16 ? 640 : 520, trace, gen});
  if (gen !== GEN) return;
  // entropies
  const K = (((IDX.views[view] || {}).kernels || {})[S.k] || {}), fl = K.flows;
  if (fl) {
    const prs = (fl.pairs || META.pairs || []).map(p => p.join(',')), rk = String(S.r);
    const Hba = (fl.H_b_given_a || {})[rk] || [], Hab = (fl.H_a_given_b || {})[rk] || [];
    const xs = [], hba = [], hab = [];
    for (let t = 0; t < L.length - 1; t++) { const pi = prs.indexOf(L[t] + ',' + L[t + 1]); xs.push(L[t] + 0.5); hba.push(pi >= 0 ? Hba[pi] : null); hab.push(pi >= 0 ? Hab[pi] : null); }
    const bl = ((fl.baseline || {})[rk] || []), blAt = l => bl.find(x => x.layer === l) || {};
    plot(DX.ent, [
      {x: xs, y: hba, name: 'H(c<sub>l+1</sub> | c<sub>l</sub>)', mode: 'lines+markers', line: {color: CAT8[0], width: 2}},
      {x: xs, y: hab, name: 'H(c<sub>l</sub> | c<sub>l+1</sub>)', mode: 'lines+markers', line: {color: CAT8[1], width: 2}},
      {x: L, y: L.map(l => blAt(l).H_B_given_A ?? null), name: 'baseline H(c<sup>B</sup> | c<sup>A</sup>), same layer', mode: 'markers', marker: {color: '#52514e', symbol: 'line-ew-open', size: 16, line: {width: 2}}},
      {x: L, y: L.map(l => blAt(l).H_A_given_B ?? null), name: 'baseline H(c<sup>A</sup> | c<sup>B</sup>)', mode: 'markers', marker: {color: '#a9a59c', symbol: 'line-ew-open', size: 11, line: {width: 2}}}],
      {height: 280, xaxis: {title: {text: 'layer (adjacent-layer transitions plotted between integers)'}, dtick: 1}, yaxis: {title: {text: 'bits'}, rangemode: 'tozero'}, legend: {y: 1.24}});
  } else DX.ent.innerHTML = '<div class="na">no flows.json for this kernel</div>';
  // flow matrix for (a, b)
  const Ccnt = await DB.flows(view, S.k, S.r), pairs = META.pairs || [];
  const pi = pairs.findIndex(([a, b]) => a === S.a && b === S.b);
  DX.fmHd.innerHTML = T(`Row-normalised contingency $F_{cd}=\\Pr[c_{${S.b}}=d\\mid c_{${S.a}}=c]$ for the Block-tab pair (${S.a} → ${S.b}); last row/column = unassigned. Change the pair in the Block tab.`);
  if (Ccnt && pi >= 0) {
    const K1 = S.r + 1, z = [];
    for (let a = 0; a < K1; a++) { let s = 0; const row = []; for (let b = 0; b < K1; b++) s += Ccnt[(pi * K1 + a) * K1 + b]; for (let b = 0; b < K1; b++) row.push(s ? Ccnt[(pi * K1 + a) * K1 + b] / s : null); z.push(row); }
    const lab = Array.from({length: K1}, (_, i) => i === S.r ? 'u' : String(i));
    plot(DX.fm, [{type: 'heatmap', z, x: lab, y: lab, zmin: 0, zmax: 1, colorscale: SEQ_STOPS, xgap: S.r > 32 ? 0 : 1, ygap: S.r > 32 ? 0 : 1, colorbar: {thickness: 9},
      hovertemplate: `state %{y} at ${S.a} → state %{x} at ${S.b}: %{z:.2f}<extra></extra>`}],
      {height: 330, margin: {l: 40, t: 8}, xaxis: {title: {text: `state at layer ${S.b}`}, type: 'category', tickfont: {size: 9}}, yaxis: {title: {text: `state at layer ${S.a}`}, type: 'category', autorange: 'reversed', tickfont: {size: 9}}});
  } else DX.fm.innerHTML = '<div class="na">no flow counts for this pair/kernel/r</div>';
  // smoothness profiles
  const Sm = await DB.smooth(view, S.k); if (gen !== GEN) return;
  if (!Sm) { DX.sp.innerHTML = DX.ov.innerHTML = '<div class="na">no smooth.f32 for this kernel</div>'; return; }
  const nL = L.length, ti = TS().indexOf(S.t), a = DB.li(view, S.l), nI = 32;
  let zcap = 0; for (let q = ti; q < Sm.length; q += 4) if (Sm[q] > zcap) zcap = Sm[q];
  zcap = zcap > 0 ? zcap : 1; const capNote = ` Colour range 0 to ${fmt(zcap, 3)} = the largest value of S at this kernel and t (fixed across layers; smoothness itself is in [0, 1]).`;
  DX.spHd.innerHTML = T(`$S[a,i,b,t]=s_b^t(\\psi_i^{(a)})$ for the first ${nI} modes of layer $a=${S.l}$ (layer control), t = ${S.t}. Click a row to select the mode.` + capNote);
  const z = [], y = [];
  for (let i = 0; i < nI; i++) { y.push('ψ' + (i + 1)); z.push(L.map((_, b) => Sm[((a * NM + i) * nL + b) * 4 + ti])); }
  plot(DX.sp, [{type: 'heatmap', z, x: L, y, zmin: 0, zmax: zcap, colorscale: SEQ_STOPS, colorbar: {thickness: 9, title: {text: 's', side: 'right'}}, hovertemplate: '%{y} of layer ' + S.l + ' at layer %{x}: %{z:.3f}<extra></extra>', xgap: 1, ygap: 1}],
    {height: 460, margin: {l: 44, t: 8}, xaxis: {title: {text: 'layer b'}, dtick: 1}, yaxis: {autorange: 'reversed', tickfont: {size: 9}},
      shapes: [{type: 'rect', x0: S.l - 0.5, x1: S.l + 0.5, y0: -0.5, y1: nI - 0.5, line: {color: '#b04a4a', width: 1}}]});
  if (!DX.spW && DX.sp.on) { DX.sp.on('plotly_click', ev => { const p = ev.points[0]; select({kind: 'mode', view: S.view, l: S.l, i: p.pointIndex[0]}); }); DX.spW = true; }
  const zo = L.map((_, aa) => L.map((_, b) => { let s = 0; for (let i = 0; i < 8; i++) s += Sm[((aa * NM + i) * nL + b) * 4 + ti]; return s / 8; }));
  plot(DX.ov, [{type: 'heatmap', z: zo, x: L, y: L, zmin: 0, zmax: zcap, colorscale: SEQ_STOPS, colorbar: {thickness: 9}, hovertemplate: 'modes of layer %{y} at layer %{x}: %{z:.3f}<extra></extra>', xgap: 1, ygap: 1}],
    {height: 460, margin: {l: 44, t: 8}, xaxis: {title: {text: 'layer b (where smoothness is measured)'}, dtick: 1}, yaxis: {title: {text: 'layer a (whose modes)'}, dtick: 1, autorange: 'reversed'},
      shapes: [{type: 'rect', x0: L[0] - 0.5, x1: L[nL - 1] + 0.5, y0: S.l - 0.5, y1: S.l + 0.5, line: {color: '#b04a4a', width: 1}}]});
  if (!DX.ovW && DX.ov.on) { DX.ov.on('plotly_click', ev => setS({l: +ev.points[0].y})); DX.ovW = true; }
}
