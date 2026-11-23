/* ---------- Definitions (all objects on the page, PLAN notation) ---------- */
const DEFS = [
  ['Sites, views', [
    ['sites $X$', 'N = 10,000 ImageNet-100 images (100 per class; 50 val + 50 train), standard eval transform. Labels and superclasses are stored for validation and display only.'],
    ['CLS view', viewDefTex('cls') + '; $l\\in\\{1..12\\}$ (layer 0 excluded: constant).'],
    ['mean-patch view', 'secondary view, same pipeline, $l\\in\\{0..12\\}$.'],
    ['caveat', CAVEAT_VIEW],
    ['distance', '$d_{ij}^2=2-2\\cos(v_i,v_j)$, the only geometric assumption (dot product on the view).'],
  ]],
  ['Per-layer operator', [
    ['self-tuned kernel', '$K^{\\mathrm{st}}_{ij}=\\exp\\!\\big(-d_{ij}^2/(s_is_j)\\big)$, $s_i$ = distance to the $k$-th nearest neighbour, $k\\in\\{7,15,30\\}$. Resolution declared in neighbour count; not guaranteed PSD.'],
    ['vMF kernel', '$K^{\\mathrm{vmf}}_{ij}=\\exp\\!\\big(\\kappa(\\cos_{ij}-1)\\big)$, $\\kappa\\in\\{1/0.07,5,30\\}$ ($1/0.07$ = DINOv2 kNN-eval temperature). Resolution declared in cosine units.'],
    ['normalisation', 'Sinkhorn: $P=DKD$, $P\\one=\\one$; one shared uniform stationary measure across layers. Coifman–Lafon $\\alpha=1$ is computed only for comparison.'],
    ['two-step operator', '$R_l=P_l^2$ (symmetric, bistochastic, PSD); resolution $t\\in\\{1,4,16,64\\}$: $R_l^t=P_l^{2t}$.'],
    ['modes', '$P_l\\psi_i^{(l)}=\\lambda_i\\psi_i^{(l)}$, top 64 nonconstant eigenpairs, $\\|\\psi\\|=1$ on the $N$ images, largest $|\\psi|$ positive.'],
    ['localisation', 'effective number of images $1/\\sum_x\\psi(x)^4$ (per mode, and for the top-$r$ projector).'],
    ['split-half stability', 'principal cosines between the top-$r$ subspaces built on disjoint halves A, B and extended to held-out images by the Sinkhorn-consistent Nyström extension $\\psi(x)=\\sum_j p(x,j)\\psi(x_j)/\\lambda$, $p(x,j)=K(x,x_j)d_j/\\sum_kK(x,x_k)d_k$ (modes with $|\\lambda|<0.05$ not extended).'],
  ]],
  ['Coarse states', [
    ['SEBA', 'sparse basis of $\\mathrm{span}\\{\\one,\\psi_1..\\psi_{r-1}\\}$, $r\\in\\{8,16,32,64\\}$, clipped at 0, each vector scaled to max 1: memberships $m_j(x)$.'],
    ['hard state', '$c_l(x)=\\arg\\max_jm_j(x)$; unassigned if $\\max_jm_j(x)<0.1$.'],
    ['flows', 'contingency of $(c_a(x),c_b(x))$ over images; row-normalised $F_{cd}=\\Pr[c_b=d\\mid c_a=c]$; $H(c_b\\mid c_a)$, $H(c_a\\mid c_b)$ in bits, with a same-layer split-half baseline.'],
  ]],
  ['Across depth', [
    ['smoothness', SMOOTH_DEF],
    ['smoothness profile', '$S[a,i,b,t]=s_b^t(\\psi_i^{(a)})$.'],
    ['endpoint smoothness-change operator', '$\\Delta_t^{a\\to b}=R_b^t-R_a^t$; top positive eigenfunctions: <b>became smooth</b>; most negative: <b>stopped being smooth</b>.'],
    ['caveat', CAVEAT_DMODE],
    ['telescoping', '$f^\\top\\Delta^{a\\to c}f=\\sum_{l=a}^{c-1}f^\\top\\Delta^{l\\to l+1}f$ (exact); net $=\\sum$, path $=\\sum|\\cdot|$.'],
    ['cross-validated value', CV_DEF],
    ['product operator', '$Q^{a\\to b}=P_bP_a$, top 32 singular triples (comparison), and the predecessor\'s mean operator.'],
  ]],
  ['Components (ViT-specific variant, CLS position)', [
    ['residual decomposition', '$h_l=e+\\sum_{l\'\\le l}\\big[\\sum_hA_{l\',h}+b^{\\mathrm{attn}}_{l\'}+\\sum_nM_{l\',n}+b^{\\mathrm{mlp}}_{l\'}\\big]$, $M_{l\',n}(x)=g_{l\',n}(x)\\,(\\mathrm{LS}\\odot W_{\\mathrm{out}}[:,n])$, $g$ = post-GELU activation at CLS.'],
    ['view map', '$v_l(x)=J_l(x)h_l(x)+o_l(x)$, $J_l(x)u=\\gamma\\odot(u-\\bar u)/(\\sigma_l(x)\\rho_l(x))$: exact at the actual scales; the frozen-scale approximation is only in treating $J_l(x)$ as independent of the attributed component.'],
    ['head function', '$\\|A_{l,h}(x)\\|_2$, norm of head $h$\'s output to CLS (colouring only).'],
    ['loading', METRIC_DEF.load.tex.replace(/^/, '$') + '$; ' + METRIC_DEF.load.where],
    ['selectivity', '$' + METRIC_DEF.auroc.tex + '$; ' + METRIC_DEF.auroc.where],
    ['direct attribution', '$' + METRIC_DEF.attr.tex + '$; ' + METRIC_DEF.attr.where + ' Always shown with the fit\'s $R^2$ (probe and Nyström-extended held-out).'],
    ['write cosine', '$' + METRIC_DEF.write.tex + '$; ' + METRIC_DEF.write.where],
  ]],
  ['Decodability', [
    ['ridge, kNN', 'closed-form ridge one-vs-rest (GCV $\\lambda$) and weighted kNN ($k=20$, $T=0.07$), fit on the probe set, tested on the held-out 10,000; fine and superclass labels; CLS and mean-patch.'],
  ]],
];
let DEFS_DONE = false;
function renderDefs() {
  if (DEFS_DONE) return; DEFS_DONE = true;
  const root = $('#tab-defs');
  root.append(h('div', {class: 'card defs'}, h('div', {class: 'hd'}, h('span', {class: 't'}, 'Definitions of every object on this page'), h('span', {class: 's'}, 'notation of explore/vit_coherent/PLAN.md (v2)')),
    ...DEFS.map(([g, items]) => h('div', null, h('h3', null, g), h('dl', null, ...items.flatMap(([k, v]) => [h('dt', null, k), h('dd', {html: T(v)})]))))));
}
