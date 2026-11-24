// Page shell (adapted from the DepthEmbedding site): hash router (#viewer?<state>, #experiments/<entry>, #method/<id>),
// maths, heading anchors, and lazy live charts in Experiments and Method (js/live.js).
const TABS = ["viewer", "experiments", "method"];
let viewer = null;

function resolveHash() {
  const raw = decodeURIComponent(location.hash.slice(1));
  const qi = raw.indexOf("?"), path = qi < 0 ? raw : raw.slice(0, qi), qs = qi < 0 ? "" : raw.slice(qi + 1);
  let [tab, id] = path ? path.split("/") : ["experiments", null];
  if (!TABS.includes(tab)) {       // a bare #id: the tab that holds it, else the viewer
    const e = tab && document.getElementById(tab), p = e && e.closest(".tab-panel");
    id = p ? tab : null; tab = p ? p.dataset.tab : "viewer";
  }
  const panel = document.getElementById(`tab-${tab}`);
  const anchor = (id && (panel.querySelector(`#${CSS.escape(id)}`) || (tab === "experiments" && panel.querySelector(`#x-${CSS.escape(id)}`)))) || null;
  return { tab, id, anchor, query: new URLSearchParams(qs) };
}

async function route() {
  const { tab, id, anchor, query } = resolveHash();
  document.body.dataset.tab = tab;
  document.querySelectorAll(".tab-panel").forEach(p => p.hidden = p.dataset.tab !== tab);
  document.querySelectorAll(".tablist a").forEach(a => a.setAttribute("aria-selected", String(a.dataset.tab === tab)));
  if (tab === "viewer") {
    if (!viewer) {
      const { Viewer } = await import("./viewer.js");
      viewer = window.__viewer = new Viewer(document.getElementById("vw"));
      barHeight();
    }
    if ([...query.keys()].length && location.hash !== viewer.lastHash) await viewer.applyParams(query);
    else await viewer.layout();
    if (anchor) scrollToAnchor(anchor);
    document.getElementById("vw").dataset.state = "ready";
    return;
  }
  document.title = `${tab[0].toUpperCase()}${tab.slice(1)} · TransformerModes`;
  if (tab === "experiments") { xpView(id, anchor); figures(); renderVisibleSoon(); return; }
  figures(); renderVisibleSoon();
  xp.cur = undefined;
  if (anchor) scrollToAnchor(anchor); else window.scrollTo(0, 0);
}
function barHeight() {
  const set = () => { const b = document.querySelector(".vw-bar"); if (b) document.documentElement.style.setProperty("--bar-h", b.offsetHeight + "px"); };
  const wait = () => { const b = document.querySelector(".vw-bar"); if (b) { new ResizeObserver(set).observe(b); set(); } else setTimeout(wait, 100); };
  wait();
}

// ---------------------------------------------------- Experiments: the overview, or one entry at a time
const xp = { list: null, cur: undefined, nav: null };
const escH = t => String(t).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
function xpEntries() {
  if (xp.list) return xp.list;
  const panel = document.getElementById("tab-experiments");
  const text = el => { if (!el) return ""; const c = el.cloneNode(true); c.querySelectorAll(".anchor").forEach(a => a.remove()); return c.textContent.trim(); };
  xp.list = [...panel.querySelectorAll(".xp > section.experiment[id]")].map(s => ({ id: s.id, el: s, title: text(s.querySelector(":scope > h2")),
    card: panel.querySelector(`.x-card h4 a[href="#experiments/${s.id}"]`)?.closest(".x-card") }));
  return xp.list;
}
function xpNav() {
  if (xp.nav) return xp.nav;
  const nav = document.createElement("nav");
  nav.className = "x-subnav"; nav.setAttribute("aria-label", "Experiment entries"); nav.hidden = true;
  nav.innerHTML = `<div class="x-subnav-inner"><a class="x-back" href="#experiments"><span aria-hidden="true">←</span> All experiments</a>`
    + `<span class="x-where"></span><span class="x-pn"><a class="x-prev"></a><a class="x-next"></a></span></div>`;
  document.getElementById("tab-experiments").prepend(nav);
  return (xp.nav = nav);
}
function xpView(id, anchor) {
  const panel = document.getElementById("tab-experiments"), art = panel.querySelector(".xp"), list = xpEntries(), nav = xpNav();
  const sec = anchor?.closest(".xp > section.experiment") || null;
  const view = id === "all" && !anchor ? "all" : sec ? "entry" : "landing";
  const prev = xp.cur, entry = list.find(e => e.el === sec);
  art.dataset.view = panel.dataset.view = document.body.dataset.xview = view;
  list.forEach(e => e.el.classList.toggle("x-current", e.el === sec));
  nav.hidden = view !== "entry";
  if (entry) {
    const i = list.indexOf(entry), pv = list[i - 1], nx = list[i + 1];
    nav.querySelector(".x-where").innerHTML = `<b>Stage A</b> <span>${i + 1} of ${list.length}</span>`;
    for (const [a, e, dir] of [[nav.querySelector(".x-prev"), pv, "Previous"], [nav.querySelector(".x-next"), nx, "Next"]]) {
      a.hidden = !e;
      if (!e) continue;
      a.href = `#experiments/${e.id}`; a.title = `${dir}: ${e.title}`;
      const t = escH(e.title);
      a.innerHTML = dir === "Previous" ? `<span aria-hidden="true">‹</span> <span class="x-pn-t">${t}</span><span class="x-pn-s">Previous</span>` : `<span class="x-pn-t">${t}</span><span class="x-pn-s">Next</span> <span aria-hidden="true">›</span>`;
    }
    document.title = `${entry.title} · Experiments · TransformerModes`;
  }
  xp.cur = sec;
  if (anchor && anchor !== sec) scrollToAnchor(anchor);
  else if (view === "entry") { if (prev !== sec) window.scrollTo({ top: 0, behavior: "instant" }); }
  else window.scrollTo({ top: 0, behavior: "instant" });
}
window.addEventListener("hashchange", route);

function scrollToAnchor(anchor) {
  for (let d = anchor.closest("details"); d; d = d.parentElement.closest("details")) d.open = true;
  const go = () => anchor.scrollIntoView({ block: "start" });
  requestAnimationFrame(go);
  document.fonts?.ready.then(() => setTimeout(go, 50));
}

// ---------------------------------------------------- live charts render when they come near the viewport
let io = null;
function figures() {
  if (io) return;
  io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting) { io.unobserve(e.target); renderFigure(e.target); }
  }, { rootMargin: "600px 0px" });
  document.querySelectorAll("#tab-experiments figure.live, #tab-method figure.live").forEach(f => {
    const d = f.closest("details");
    if (d && !d.open) d.addEventListener("toggle", () => { if (d.open) renderFigure(f); }, { once: true });
    else io.observe(f);
  });
}
// Also draw every visible figure soon after load, not only when scrolled near, so full-page captures and printing
// show them (figures in hidden tabs or closed <details> wait, since they have no width yet).
function renderVisibleSoon() {
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 200));
  idle(() => document.querySelectorAll("#tab-experiments figure.live, #tab-method figure.live").forEach(f => {
    if (!f.dataset.rendered && f.offsetParent !== null && !f.closest("details:not([open])")) { io?.unobserve(f); renderFigure(f); }
  }));
}
window.addEventListener("beforeprint", () => document.querySelectorAll("figure.live").forEach(f => { if (f.offsetParent !== null && !f.closest("details:not([open])")) renderFigure(f); }));
async function renderFigure(f) {
  if (f.dataset.rendered) return; f.dataset.rendered = "1";
  try { await (await import("./live.js")).render(f); } catch (e) { console.error("figure:", e); }
}

function maths() {
  window.renderMathInElement(document.body, {
    delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\[", right: "\\]", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }],
    ignoredClasses: ["no-math", "tooltip", "vw"], throwOnError: false,
  });
}
function anchors() {
  document.querySelectorAll(".prose h2[id], .prose h3[id], .xp h3[id], .xp section.experiment[id] > h2").forEach(hd => {
    const id = hd.id || hd.parentElement.id;
    const a = document.createElement("a"); a.className = "anchor"; a.href = `#${hd.closest(".tab-panel").dataset.tab}/${id}`;
    a.textContent = "#"; a.setAttribute("aria-label", "Link to this section");
    hd.prepend(a);
  });
}
// the primer stays folded once a reader folds it (a per-browser convenience; works without storage too)
function primerMemory() {
  const d = document.getElementById("primer"); if (!d) return;
  let pref = null;
  try { pref = localStorage.getItem("tm-primer"); } catch (e) { /* storage unavailable */ }
  // on a phone the primer is several screens tall, so it starts folded unless the reader opened it before
  if (pref === "closed" || (pref == null && innerWidth < 700)) d.open = false;
  d.addEventListener("toggle", () => { try { localStorage.setItem("tm-primer", d.open ? "open" : "closed"); } catch (e) { /* ignore */ } });
}
function init() {
  anchors(); primerMemory();
  if (window.renderMathInElement) maths(); else window.addEventListener("load", maths);
  route();
}
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
