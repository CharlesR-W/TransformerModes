/* ---------- tolerant accessors for per-layer JSON, Plotly house style ---------- */
function perLayer(obj, l, view = 'cls') {
  if (!obj) return null;
  const tryK = o => o == null ? null : (o['L' + pad2(l)] ?? o[String(l)] ?? o[l] ?? null);
  if (Array.isArray(obj)) return obj[view === 'mp' ? l : l - 1] ?? null;
  if (obj.layers) { const r = Array.isArray(obj.layers) ? obj.layers[view === 'mp' ? l : l - 1] : tryK(obj.layers); if (r != null) return r; }
  if (obj.per_layer) return Array.isArray(obj.per_layer) ? obj.per_layer[view === 'mp' ? l : l - 1] : tryK(obj.per_layer);
  return tryK(obj);
}
function perR(obj, r) { return obj == null ? null : (obj[String(r)] ?? obj['r' + r] ?? obj['r' + pad2(r)] ?? null); }

const PAXIS = {gridcolor: '#eeebe5', zerolinecolor: '#d3cdc3', linecolor: '#c3c2b7', ticks: 'outside', tickcolor: '#c3c2b7',
  title: {font: {size: 12}}, automargin: true, tickfont: {size: 11}};
function PLAY(extra) {
  const base = {margin: {l: 48, r: 12, t: 26, b: 40}, paper_bgcolor: 'white', plot_bgcolor: 'white',
    font: {family: 'system-ui,-apple-system,Segoe UI,Helvetica,Arial,sans-serif', size: 12, color: '#2b2a27'},
    xaxis: {...PAXIS}, yaxis: {...PAXIS}, legend: {orientation: 'h', y: 1.12, x: 0, font: {size: 11}}, hovermode: 'closest', height: 260,
    hoverlabel: {bgcolor: 'white', bordercolor: '#d3cdc3', font: {color: '#2b2a27'}}};
  return deepMerge(base, extra || {});
}
function deepMerge(a, b) {
  const o = Array.isArray(a) ? a.slice() : {...a};
  for (const k of Object.keys(b)) o[k] = b[k] && typeof b[k] === 'object' && !Array.isArray(b[k]) && a[k] && typeof a[k] === 'object' ? deepMerge(a[k], b[k]) : b[k];
  return o;
}
const PCONF = {displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'], displayModeBar: 'hover'};
function plot(el, traces, layout) { try { Plotly.react(el, traces, PLAY(layout), PCONF); } catch (e) { el.innerHTML = '<span class="err">plot error: ' + esc(e.message) + '</span>'; console.error(e); } }
const T_COLORS = ['#9ec5f4', '#5598e7', '#1c5cab', '#0d366b']; // t = 1,4,16,64: one-hue ordinal ramp (light = short time)
const tColor = t => T_COLORS[Math.max(0, TS().indexOf(t))] || '#256abf';
const KCOL = k => { const all = (META.kernels || []).map(x => x.id); return CAT8[Math.max(0, all.indexOf(k)) % 8]; };
function card(title, sub, ...kids) {
  return h('div', {class: 'card'}, h('div', {class: 'hd'}, h('span', {class: 't', html: title}), sub ? h('span', {class: 's', html: sub}) : null), ...kids);
}
function defBlock(html) { return h('div', {class: 'def', html}); }
function caveat(html) { return h('div', {class: 'caveat', html}); }
function stripsHTML(groups) { // [{label, idx:[], vals?:[]}]
  return groups.map(g => `<div class="strip"><div class="sl">${g.label}</div><div class="imgs">` +
    (g.idx.length ? g.idx.map((i, k) => thumbHTML(i, g.size || 64, g.vals ? `<span class="tv">${fmt(g.vals[k])}</span>` : '')).join('') : '<span class="na">none</span>') + '</div></div>').join('');
}
function wireThumbs(el) {
  el.querySelectorAll('.th[data-img]').forEach(t => {
    const i = +t.dataset.img;
    t.onclick = () => focusImg(i);
    t.onmouseenter = e => TIP.show(imgTip(i, CUR_SD), e.clientX, e.clientY);
    t.onmousemove = e => TIP.show(imgTip(i, CUR_SD), e.clientX, e.clientY);
    t.onmouseleave = () => TIP.hide();
    if (i === S.img) t.classList.add('cur');
  });
}
let CUR_SD = null; // data of the current selection (for tooltips)
