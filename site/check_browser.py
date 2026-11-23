"""Drive a served build in headless Chrome (playwright-cli) and fail on anything wrong.

    python3 -m http.server 8817 --bind 127.0.0.1 -d build/site      # in another process
    TMPDIR=/var/tmp uv run --with numpy python site/check_browser.py http://127.0.0.1:8817/ /var/tmp/shots [--build build/site]

Both viewports (1440x900, 390x844) and every tab: console errors, page errors, external requests, HTTP errors, broken
images, charts that did not draw, and horizontal page overflow are failures. The viewer: every control is changed and
restored, the primer and Details open, a token is clicked, a neighbour is opened in its own text and the browser goes
back. Token cards are compared with neighbour rows decoded here independently (numpy, not the site's JavaScript):
counts, each shown neighbour's (text, position), the tied-cutoff marks, and the clicked token's predicted ids. Dark-mode
screenshots of the viewer and one entry. Screenshots go to OUT_DIR; look at them.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

HERE = Path(__file__).resolve().parent
ACC = {("C", "none"): "C-none", ("C", "window"): "C-window", ("O", "window"): "O-window"}


def decode(build: Path, manifest, path):
    """Independent reader of the manifest contract (exp1/pack_site.py docstring)."""
    spec = manifest["files"][path]
    raw = gzip.decompress((build / "d" / manifest["data_directory"] / path).read_bytes())
    assert len(raw) == spec["decoded_bytes"], path
    if spec["encoding"] == "json":
        return json.loads(raw)
    n, off, out = spec["shape"][0], 0, {}
    for f in manifest["layouts"][spec["dtype"]]:
        dt = np.dtype(f["dtype"])
        off += -off % dt.itemsize
        cols = f.get("columns", 1)
        out[f["name"]] = np.frombuffer(raw, dt, n * cols, off).reshape(n, cols) if "columns" in f else np.frombuffer(raw, dt, n, off)
        off += n * cols * dt.itemsize
    assert off == len(raw), path
    return out


def expected_cases(build: Path, limit=4):
    """Up to `limit` ordinary cases (one per setting, middle of a text) plus one row whose shown ranks include a tied
    cutoff, searched among the memory-none settings, where ties occur."""
    m = json.loads((build / "d" / "manifest.json").read_text())
    pools = {}
    for name, ranges in m["candidate_pools"].items():
        flat = [(t, p) for t, a, b in ranges for p in range(a, b)]
        pools[name] = (flat, {t: (a, b) for t, a, b in ranges})
    texts = {t["id"]: t for t in m["texts"]}
    cases, seen_tie = [], False
    for s in m["settings"]:
        if (s["centring"], s["mask"]) not in ACC:
            continue
        flat, ranges = pools[s["candidate_pool"]]
        if len(cases) >= limit and (seen_tie or s["memory"] != "none"):
            continue
        for tid, path in s["files"].items():
            tid = int(tid)
            a, b = ranges[tid]
            rows = decode(build, m, path)
            flags = rows["tie_flags"]
            tie_rows = np.flatnonzero((flags & 16) & ((flags & 15) > 0))
            want_tie = not seen_tie and len(tie_rows)
            if not want_tie and len(cases) >= limit:
                continue
            r = int(tie_rows[0]) if want_tie else (b - a) // 2
            pos = a + r
            idx = [int(j) for j in rows["index"][r] if j != m["layouts"]["neighbours"][0]["missing"]]
            k = int(flags[r] & 15) if flags[r] & 16 else 0
            pred_path = texts[tid]["predictions"].get(s["model"])
            preds = decode(build, m, pred_path)["token_id"][pos - 1].tolist() if pred_path else []
            cases.append(dict(hash=f"#viewer?text={tid}&model={s['model']}&layer={s['site']}&memory={s['memory']}&acc={ACC[s['centring'], s['mask']]}&pos={pos}",
                              pos=pos, counts=rows["origin_counts"][r].tolist(), neighbours=[list(flat[j]) for j in idx],
                              tied=list(range(len(idx) - k + 1, len(idx) + 1)) if k else [], preds=preds, source=s["source"]))
            seen_tie = seen_tie or bool(want_tie)
            if len(cases) >= limit + 1 or (len(cases) >= limit and seen_tie):
                return cases
            if not want_tie and len(cases) < limit:
                break
    return cases


CHECK_JS = r"""
async page => {
  const {base, dir, cases} = CONFIG;
  const origin = new URL(base).origin, issues = new Set(), shots = [];
  let expectFailure = false;      // set while a load failure is provoked on purpose
  page.on('console', msg => { if (msg.type() === 'error' && !expectFailure) issues.add(`console: ${msg.text()}`); });
  page.on('pageerror', err => issues.add(`page error: ${err.message}`));
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.origin !== origin && url.protocol !== 'data:' && url.protocol !== 'about:') { issues.add(`external request: ${url.href}`); await route.abort(); }
    else await route.continue();
  });
  page.on('response', r => { if (r.status() >= 400 && !expectFailure) issues.add(`HTTP ${r.status()}: ${r.url()}`); });
  const shot = async (name, full = false) => { const path = `${dir}/${name}.png`; await page.screenshot({path, fullPage: full, timeout: 30000}); shots.push(path); };
  const test = async (label, fn) => { try { await fn(); } catch (err) { issues.add(`${label}: ${err.message.split('\n')[0]}`); } };
  const go = async hash => { await page.goto('about:blank'); await page.goto(base + hash, {waitUntil: 'load'}); };
  const overflow = async label => {
    const w = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
    if (w[0] > w[1] + 1) issues.add(`${label}: horizontal page overflow (${w[0]} > ${w[1]})`);
  };
  const viewerReady = async () => {
    await page.waitForFunction(() => window.__viewer && window.__viewer.cur && document.querySelectorAll('#vw .tk').length > 0 && !document.querySelector('.vw-status')?.textContent, null, {timeout: 20000});
  };
  const cardReady = async pos => {
    await page.waitForFunction(p => { const c = document.querySelector('.vw-card'); return c && !c.hidden && c.dataset.p === String(p) && c.querySelector('.vw-nb[data-rank]'); }, pos, {timeout: 20000});
  };
  const figures = async scope => {
    const opened = await scope.locator('details').evaluateAll(ds => ds.map(d => { const o = d.open; d.open = true; return o; }));
    const figs = scope.locator('figure.live');
    for (let i = 0, n = await figs.count(); i < n; i++) {
      const f = figs.nth(i), label = await f.getAttribute('data-src');
      await test(label, async () => {
        await f.scrollIntoViewIfNeeded();
        await page.waitForFunction(el => { const b = el.querySelector('.live-body'); return b && (b.querySelector('svg text, table, .chart-sub') || b.querySelector('.live-error')); }, await f.elementHandle(), {timeout: 15000});
        const err = await f.locator('.live-error').count();
        if (err) issues.add(`${label}: ${await f.locator('.live-error').innerText()}`);
      });
    }
    const broken = await scope.locator('img').evaluateAll(imgs => imgs.filter(im => !im.complete || !im.naturalWidth).map(im => im.src));
    for (const u of broken) issues.add(`broken image: ${u}`);
    await scope.locator('details').evaluateAll((ds, o) => ds.forEach((d, i) => { d.open = o[i]; }), opened);
  };

  for (const [size, width, height] of [['desktop', 1440, 900], ['mobile', 390, 844]]) {
    await page.setViewportSize({width, height});
    await page.emulateMedia({colorScheme: 'light'});
    // ---------------- viewer
    await test(`${size} viewer`, async () => {
      await go('#viewer');
      await viewerReady();
      await overflow(`${size} viewer`);
      await shot(`viewer-${size}`);
      // the primer folds and unfolds
      await page.locator('#primer > summary').click(); await page.locator('#primer > summary').click();
      // every control: switch to another enabled value and back
      for (const name of ['Model', 'Layer', 'Memory']) {
        const g = page.getByRole('radiogroup', {name, exact: true});
        const on = await g.locator('[aria-checked="true"]').getAttribute('data-v');
        const other = g.locator('button[aria-checked="false"]:not([disabled])').first();
        if (!await other.count()) continue;
        await other.click(); await viewerReady();
        await g.locator(`button[data-v="${on}"]`).click(); await viewerReady();
      }
      for (const name of ['Accounting', 'Text']) {
        const sel = page.getByRole('combobox', {name, exact: true});
        const old = await sel.inputValue();
        const other = await sel.evaluate(s => [...s.options].find(o => !o.disabled && o.value !== s.value)?.value ?? null);
        if (other === null) continue;
        await sel.selectOption(other); await viewerReady();
        await sel.selectOption(old); await viewerReady();
      }
      // a view that fails to load leaves the previous one in place, with its controls
      if (size === 'desktop') {
        const g = page.getByRole('radiogroup', {name: 'Memory', exact: true});
        const on = await g.locator('[aria-checked="true"]').getAttribute('data-v');
        const target = on === 'whole' ? '64' : 'whole';
        expectFailure = true;
        await page.route(`**/*_${target === 'whole' ? 'whole' : '64'}_*.bin.gz`, r => r.abort());
        await page.route(`**/${target}_*.bin.gz`, r => r.abort());
        await g.locator(`button[data-v="${target}"]`).click();
        await page.waitForFunction(() => /Could not load/.test(document.querySelector('.vw-status')?.textContent || ''), null, {timeout: 10000});
        const now = await g.locator('[aria-checked="true"]').getAttribute('data-v');
        const state = await page.evaluate(() => window.__viewer.state.memory);
        if (now !== on || state !== on) issues.add(`failed load: controls show ${now}, state ${state}, expected ${on}`);
        if (await page.locator('.vw-text.stale').count()) issues.add('failed load: text left greyed out');
        await page.unrouteAll({behavior: 'ignoreErrors'});
        await page.route('**/*', async route => {
          const url = new URL(route.request().url());
          if (url.origin !== origin && url.protocol !== 'data:' && url.protocol !== 'about:') { issues.add(`external request: ${url.href}`); await route.abort(); }
          else await route.continue();
        });
        expectFailure = false;
      }
      // click a token, open a neighbour in its own text, come back
      const tok = page.locator('#vw .tk:not(.burn):not(.none)').nth(30);
      await tok.scrollIntoViewIfNeeded(); await tok.click();
      const pos = +(await tok.getAttribute('data-p'));
      await cardReady(pos);
      await overflow(`${size} viewer with card`);
      await shot(`viewer-card-${size}`);
      const link = page.locator('.vw-card .vw-nb[data-rank] .vw-open').first();
      const [nt, np] = [await page.locator('.vw-card .vw-nb[data-rank]').first().getAttribute('data-text'), +(await page.locator('.vw-card .vw-nb[data-rank]').first().getAttribute('data-pos'))];
      await link.click();
      await cardReady(np);
      const cur = await page.evaluate(() => [window.__viewer.state.text, window.__viewer.state.pos]);
      if (String(cur[0]) !== nt || cur[1] !== np) issues.add(`${size} neighbour link went to ${cur}, expected ${nt}, ${np}`);
      await page.goBack(); await cardReady(pos);
      await page.keyboard.press('Escape');
      // Details
      await page.locator('.vw-details > summary').click();
      await page.waitForFunction(() => document.querySelector('.vw-details-body').innerText.length > 100, null, {timeout: 5000}).catch(() => issues.add(`${size} Details is nearly empty`));
      await page.locator('.vw-details > summary').click();
    });
    // ---------------- token cards against independently decoded rows
    if (size === 'desktop') for (const c of cases) await test(`card ${c.hash}`, async () => {
      await go(c.hash);
      await viewerReady(); await cardReady(c.pos);
      const got = await page.evaluate(() => {
        const card = document.querySelector('.vw-card');
        const label = card.querySelector('.vw-bar3')?.getAttribute('aria-label') || '';
        const m = label.match(/(\d+) within 64, (\d+) far, (\d+) other/);
        return { counts: m ? [+m[3], +m[2], +m[1]] : null,
                 nbs: [...card.querySelectorAll('.vw-nb[data-rank]')].map(li => [+li.dataset.text, +li.dataset.pos]),
                 tied: [...card.querySelectorAll('.vw-nb[data-rank]')].filter(li => li.querySelector('.vw-tie')).map(li => +li.dataset.rank),
                 preds: [...card.querySelectorAll('.vw-q .vw-preds [title]')].map(s => +(s.getAttribute('title').match(/vocabulary id (\d+)/) || [])[1]) };
      });
      const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
      if (!eq(got.counts, c.counts)) issues.add(`card ${c.hash}: counts ${JSON.stringify(got.counts)} != stored ${JSON.stringify(c.counts)}`);
      if (!eq(got.nbs, c.neighbours)) issues.add(`card ${c.hash}: neighbours ${JSON.stringify(got.nbs)} != stored ${JSON.stringify(c.neighbours)}`);
      if (!eq(got.tied, c.tied)) issues.add(`card ${c.hash}: tie marks ${JSON.stringify(got.tied)} != stored ${JSON.stringify(c.tied)}`);
      if (!eq(got.preds, c.preds)) issues.add(`card ${c.hash}: predictions ${JSON.stringify(got.preds)} != stored ${JSON.stringify(c.preds)}`);
    });
    // ---------------- experiments and method
    for (const tab of ['experiments', 'method']) await test(`${size} ${tab}`, async () => {
      await go(`#${tab}`);
      const panel = page.locator(`#tab-${tab}`);
      await panel.waitFor({state: 'visible'});
      await page.waitForTimeout(300);
      await overflow(`${size} ${tab}`);
      await page.evaluate(() => window.scrollTo(0, 0));
      await shot(`${tab}-${size}`);
      if (tab === 'method') { await figures(panel); await overflow(`${size} method after figures`); }
      if (tab === 'experiments') {
        const ids = await panel.locator('section.experiment').evaluateAll(els => els.map(e => e.id));
        for (const id of ids) await test(`${size} ${id}`, async () => {
          await go(`#experiments/${id}`);
          const entry = page.locator(`#${id}`);
          await entry.waitFor({state: 'visible'});
          const shown = await panel.locator('section.experiment').evaluateAll(els => els.filter(e => e.getBoundingClientRect().height > 0).length);
          if (shown !== 1) issues.add(`${size} ${id}: ${shown} entries visible`);
          await figures(entry);
          await overflow(`${size} ${id}`);
          await page.evaluate(() => window.scrollTo(0, 0));
          await page.waitForTimeout(200);
          await shot(`${id}-${size}`, size === 'desktop');
        });
      }
    });
  }
  // ---------------- dark mode
  await page.setViewportSize({width: 1440, height: 900});
  await page.emulateMedia({colorScheme: 'dark'});
  await test('dark viewer', async () => {
    const c = cases[0]; await go(c.hash); await viewerReady(); await cardReady(c.pos);
    await page.locator('.vw-text').scrollIntoViewIfNeeded();
    await shot('viewer-card-dark');
  });
  await test('dark entry', async () => {
    await go('#experiments/x-e1-neighbours'); const e = page.locator('#x-e1-neighbours'); await e.waitFor({state: 'visible'}); await figures(e);
    await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForTimeout(200); await shot('e1-dark');
  });
  return '@@JSON@@' + JSON.stringify({issues: [...issues], shots}) + '@@END@@';
}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base_url")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--build", type=Path, default=HERE.parent / "build" / "site", help="the served build (for independent decoding)")
    ap.add_argument("--session", default=f"tm-check-{os.getpid()}")
    a = ap.parse_args()
    base = a.base_url.rstrip("/") + "/"
    u = urlsplit(base)
    if u.scheme not in ("http", "https") or u.fragment or u.query:
        ap.error("base_url must be an http(s) directory URL")
    dest = a.out_dir.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    cases = expected_cases(a.build)
    js = CHECK_JS.replace("CONFIG", json.dumps(dict(base=base, dir=str(dest), cases=cases)), 1)
    cmd = ["playwright-cli", f"-s={a.session}"]

    def run(*args, timeout=60, check=True):
        r = subprocess.run(cmd + list(args), cwd=dest, capture_output=True, text=True, timeout=timeout)
        if check and r.returncode:
            raise RuntimeError(f"playwright-cli {args[0]} failed:\n{r.stdout}\n{r.stderr}")
        return r.stdout

    try:
        try:
            run("open", "about:blank", "--browser=chrome")
            out = run("run-code", js, "--raw", timeout=900)
        finally:
            run("close", check=False)
        res = json.loads(out.strip())
        if not isinstance(res, str) or not res.startswith("@@JSON@@"):
            raise RuntimeError(f"unexpected browser output: {out[-2000:]}")
        res = json.loads(res[len("@@JSON@@"):-len("@@END@@")])
    except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as exc:
        ap.exit(1, f"browser check failed: {exc}\n")
    print(f"{len(cases)} token cards compared with independently decoded rows (one with a tied cutoff: "
          f"{any(c['tied'] for c in cases)})")
    for issue in res["issues"]:
        print(f"error: {issue}")
    print(f"{len(res['shots'])} screenshots, {len(res['issues'])} errors: {dest}")
    return int(bool(res["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
