"""CPU synthetic scale/truncation benchmark. See --help. Outputs only bench_numerics.json.

Use BLAS/OMP thread settings consistently; the CLI passes its --threads setting
on to isolated case processes, and uses Linux sampled RSS (not Python allocations).
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter

import numpy as np

ALPHA = 1 - 1/64
T, BURN, WIDTH, SITES = 2047, 256, 64, 8
CUTS = (128, 256, 512, 1024)
FLOORS = (0., 1e-6)
GIB = 2**30
MIB = 2**20
HERE = Path(__file__).resolve().parent


def synthetic_site(contexts, t, site, width=WIDTH, seed=83):
    """Shared planted cluster runs (roughly 73 tokens) plus slow smooth drift.

    Cluster order and phase vary by context; site-specific perturbations rotate
    the latent signals and add independent noise. Centre per site, then unit rows.
    """
    common = np.random.default_rng(seed)
    rng = np.random.default_rng(seed + 1009 + 6299*site)
    cluster = common.standard_normal((28, width))
    cluster *= common.uniform(0.8, 1.3, (28, 1))
    drift = common.standard_normal((6, width))
    pos = np.arange(t)
    x = np.empty((contexts, t, width))
    for c in range(contexts):
        order = common.permutation(len(cluster))
        labels = order[((pos + int(common.integers(73))) // 73) % len(cluster)]
        phase = common.uniform(-np.pi, np.pi, 3)
        theta = pos[:, None] * (2*np.pi / np.array([371, 661, 1217])) + phase
        theta += site * np.array([0.025, -0.045, 0.013])
        drift_coeff = np.column_stack((np.sin(theta), np.cos(theta)))
        x[c] = 1.25 * cluster[labels] + 0.27 * drift_coeff @ drift
    x += 0.62 * rng.standard_normal(x.shape)
    x = x + 0.16 * (x @ (rng.standard_normal((width, width)) / np.sqrt(width)))
    x -= x.mean(axis=(0, 1), keepdims=True)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x


def mode_drift(ref_t, ref_w, ref_u, t, w, u, k=16):
    """Principal angles via orthogonal projection, resolving angles near zero."""
    q = np.linalg.qr(ref_u[..., :k].reshape(-1, k), mode='reduced')[0]
    z = np.linalg.qr(u[..., :k].reshape(-1, k), mode='reduced')[0]
    sine = np.linalg.svd(z - q @ (q.T @ z), compute_uv=False).clip(0., 1.)
    rel = w[:k] / ref_w[:k] - 1
    return dict(t_relative=float(t/ref_t - 1), eigenvalue_max_relative=float(np.abs(rel).max()),
                eigenvalue_relative=rel.tolist(), subspace_mean_sin2=float(np.mean(sine**2)),
                subspace_max_sin=float(sine.max()), principal_angles_rad=np.arcsin(sine).tolist())


def dense_calibrate(ms):
    import spacetime as st
    lam, vec = st.site_eigs(ms)
    scales = (1 - lam[:, 1:9]).mean(axis=1)
    t = st.balance_direct(ms, scales, eigs=(lam, vec))
    return scales, t


def dense_modes(ms, scales, t, k=24):
    import spacetime as st
    w, u = st.modes(ms, scales, t, k=k)
    return w, u, float(st.residual(ms, scales, t, w, u).max())


def _memory_field(path, field):
    with open(path) as f:
        for line in f:
            if line.startswith(field + ':'):
                return int(line.split()[1]) * 1024
    raise RuntimeError(f'missing Linux memory statistic: {field}')


def _rss():
    return _memory_field('/proc/self/status', 'VmRSS')


class Stages:
    """Stage wall time summed over repeated visits; peak absolute sampled RSS."""
    def __init__(self, limit):
        self.base = _rss()
        self.limit = limit
        self.stats = {}

    @contextmanager
    def measure(self, name):
        rec = self.stats.setdefault(name, dict(seconds=0., peak_rss_bytes=0, peak_growth_bytes=0))
        stop = Event()
        start_rss = _rss()

        def sample():
            rss = _rss()
            rec['peak_rss_bytes'] = max(rec['peak_rss_bytes'], rss)
            rec['peak_growth_bytes'] = max(rec['peak_growth_bytes'], rss - start_rss)
            if rss > self.limit:
                print(f'RSS limit exceeded in {name}: {rss/MIB:.0f} MiB', file=sys.stderr, flush=True)
                os._exit(75)

        def poll():
            while not stop.wait(0.01):
                sample()

        sample()
        th = Thread(target=poll, daemon=True)
        th.start()
        t0 = perf_counter()
        try:
            yield
        finally:
            rec['seconds'] += perf_counter() - t0
            stop.set()
            th.join()
            sample()


def _solve(walks, stats):
    import solve
    with stats.measure('calibrate'):
        cal = solve.calibrate(walks)
    with stats.measure('solve'):
        result = solve.solve_modes(walks, cal)
    return result


def _result(result, walks):
    cal, checks = result.calibration, result.checks
    return dict(t_star=cal.t_star, scales=cal.scales.tolist(), eigenvalues=result.values.tolist(),
                residual_max=float(checks['resid'].max()), balance_error=float(checks['balance_check']),
                cutoff_gaps=checks['gaps'], site_matvecs=cal.site_matvecs, root_matvecs=cal.root_matvecs,
                solve_matvecs=result.matvecs, walk_bytes=sum(w.nbytes for w in walks),
                union_nnz=sum(w.base.nnz - w.u.size for w in walks),
                sinkhorn_iterations=[w.sinkhorn_iterations for w in walks],
                sinkhorn_error_max=max(w.sinkhorn_error for w in walks))


def scale_case(n, k_nb, limit):
    from graph import build_walk
    from similarity import memory_neighbours
    m = n // (T - BURN + 1)
    assert n == m*(T - BURN + 1)
    stats = Stages(limit)
    walks = []
    for s in range(SITES):
        x = synthetic_site(m, T, s)
        with stats.measure('kernel'):
            nb = memory_neighbours(x, ALPHA, k_nb, BURN)
        with stats.measure('graph'):
            walks.append(build_walk(nb.indices[0], nb.distances[0]))
        del nb, x
    result = _solve(walks, stats)
    row = dict(n=n, contexts=m, k_nb=k_nb, baseline_rss_bytes=stats.base,
               stages=stats.stats, **_result(result, walks))
    if (n, k_nb) == (1792, 256):
        with stats.measure('dense_materialize'):
            ms = np.stack([w.dense() for w in walks])
        with stats.measure('dense_calibrate'):
            scales, t = dense_calibrate(ms)
        with stats.measure('dense_solve'):
            vals, vecs, residual = dense_modes(ms, scales, t)
        row['same_walk_dense'] = dict(t_star=t, residual_max=residual,
            scale_max_relative=float(np.abs(result.calibration.scales/scales - 1).max()),
            **mode_drift(t, vals, vecs, result.calibration.t_star, result.values, result.modes))
    return row


def truncation_case(limit):
    from graph import build_walk
    from kernels import ea_kernel
    from similarity import memory_kernel_dense, memory_neighbours
    n = T - BURN + 1
    stats = Stages(limit)
    ms = np.empty((SITES, n, n))
    rows = []
    for s in range(SITES):
        x = synthetic_site(1, T, s)
        with stats.measure('dense_kernel'):
            # Only in the small-N oracle; delete raw K before fitting.
            raw, dist = memory_kernel_dense(x, ALPHA, BURN)
            del raw
        with stats.measure('dense_graph'):
            ker = ea_kernel(1 - dist[0], perp=64, renyi=2)
            if not ker['err'] < 1e-10:
                raise RuntimeError(f"dense Sinkhorn residual {ker['err']:g}")
            ms[s] = 0.5 * ker['P']
            ms[s].flat[::n+1] += 0.5
        del dist, ker['P']
        nb = memory_neighbours(x, ALPHA, max(CUTS), BURN)
        ids = nb.indices[0].numpy().copy()
        ds = nb.distances[0].numpy().copy()
        del nb, x
        logq = -ker['beta'][:, None] * ((1 - (1 - ds)) - ker['dmin'][:, None]) - ker['logz'][:, None]
        masses = np.exp(logq).cumsum(axis=1)[:, np.asarray(CUTS) - 1]
        rows.append((ids, ds, 1 - masses))
    with stats.measure('dense_calibrate'):
        scales, t = dense_calibrate(ms)
    with stats.measure('dense_solve'):
        vals, vecs, residual = dense_modes(ms, scales, t)
    del ms
    ref = dict(t_star=t, scales=scales.tolist(), eigenvalues=vals.tolist(), residual_max=residual,
               cutoff_gap_16=float(vals[16] - vals[15]), stages=stats.stats)
    out = []
    for ci, k_nb in enumerate(CUTS):
        lost = np.concatenate([x[2][:, ci] for x in rows])
        for floor in FLOORS:
            walks = [build_walk(ids[:, :k_nb], ds[:, :k_nb], eta=floor) for ids, ds, _ in rows]
            import solve
            result = solve.solve_modes(walks)
            out.append(dict(k_nb=k_nb, eta=floor, lost_mass_mean=float(lost.mean()),
                            lost_mass_p95=float(np.quantile(lost, .95)), lost_mass_max=float(lost.max()),
                            **_result(result, walks),
                            **mode_drift(t, vals, vecs, result.calibration.t_star, result.values, result.modes)))
            del walks, result
    return dict(n=n, dense_reference=ref, rows=out)


def _worker(args):
    import torch
    torch.set_num_threads(args.threads)
    limit = int(args.rss_budget_gib * GIB)
    if args.worker[0] == 'scale':
        row = scale_case(int(args.worker[1]), int(args.worker[2]), limit)
    else:
        row = truncation_case(limit)
    row['process_hwm_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    row['torch_version'] = torch.__version__
    print(json.dumps(row, allow_nan=False))


def _project(scale):
    projected = []
    for k_nb in (256, 1024):
        subset = sorted((r for r in scale if r['k_nb'] == k_nb and 'stages' in r), key=lambda r: r['n'])
        if not subset:
            continue
        largest = subset[-1]
        r = 57344 / largest['n']
        data = dict(k_nb=k_nb, from_n=largest['n'], target_n=57344,
                    walk_gib=largest['walk_bytes'] * r / GIB, stages={})
        for stage in ('graph', 'calibrate', 'solve'):
            s = largest['stages'][stage]
            base = largest['baseline_rss_bytes']
            data['stages'][stage] = dict(seconds=s['seconds'] * r,
                rss_gib=(base + (s['peak_rss_bytes'] - base) * r)/GIB)
            if len(subset) >= 2:
                prev = subset[-2]
                slope = np.log(s['seconds']/prev['stages'][stage]['seconds'])/np.log(largest['n']/prev['n'])
                data['stages'][stage]['last_observed_time_exponent'] = float(slope)
                data['stages'][stage]['power_law_seconds'] = float(s['seconds'] * r**slope)
        projected.append(data)
    return projected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threads', type=int, default=1, help='BLAS and torch thread limit per worker')
    ap.add_argument('--seconds', type=int, default=870, help='total case time budget (default: 14.5 min)')
    ap.add_argument('--rss-budget-gib', type=float, default=3.75)
    ap.add_argument('--worker', nargs='+', help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        _worker(args)
        return
    start = perf_counter()
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = str(args.threads)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    results = dict(config=dict(alpha=ALPHA, sites=SITES, t=T, burn_in=BURN, width=WIDTH, threads=args.threads,
                               cpu=platform.machine(), python=platform.python_version(), numpy=np.__version__,
                               rng_seed=83, rss_budget_gib=args.rss_budget_gib,
                               projection_model='fixed union degree/iterations, constant runtime RSS + O(N)',
                               rss_method='10-ms sampled absolute Linux RSS plus process high-water mark'), scale=[])

    def run(kind, *values, estimate=None):
        avail = _memory_field('/proc/meminfo', 'MemAvailable')/GIB
        budget = min(args.rss_budget_gib, max(0., avail - 0.4))
        if estimate is not None and estimate > budget:
            return dict(skipped=f'estimated {estimate:.2f} GiB > safe budget {budget:.2f} GiB '
                                f'({avail:.2f} GiB available, 0.4 GiB held back)')
        remain = args.seconds - (perf_counter() - start)
        if remain <= 1:
            return dict(skipped='total time budget exhausted')
        cmd = [sys.executable, str(Path(__file__).resolve()), '--threads', str(args.threads),
               '--rss-budget-gib', str(budget), '--worker', kind, *map(str, values)]
        try:
            run = subprocess.run(cmd, env=env, timeout=remain, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return dict(skipped=f'worker hit remaining wall-clock budget ({remain:.0f} s)')
        if run.returncode:
            return dict(skipped=f'worker exit {run.returncode}', detail=run.stderr.strip())
        return json.loads(run.stdout)

    print('N      K_nb stage seconds / peak MiB     kernel       graph       cal        solve', flush=True)
    for n, k in [(1792, 256), (1792, 1024), (7168, 256), (7168, 1024), (14336, 256), (14336, 1024)]:
        if (n, k) == (7168, 256):
            # Measure the small dense comparison first; then prioritise the
            # truncation study before the expensive scaling cases.
            print('Running truncation study', flush=True)
            results['truncation'] = run('truncation', estimate=1.65)
            print('Truncation worker: ' + ('complete' if 'rows' in results['truncation'] else
                                           str(results['truncation'])), flush=True)
        print(f'Running N={n}, K_nb={k}', flush=True)
        # Worst-case 2Nk union with seven sites retained and a conservative
        # transient budget for incoming top-k and CSR construction at site 8.
        est = 0.45 + (7 * 24 + 84)*n*k/GIB + 4*n*SITES*96/GIB
        if n == 14336 and k == 1024:
            prev = next((r for r in results['scale'] if r['n'] == n//2 and r['k_nb'] == k and 'stages' in r), None)
            if prev:
                peak = max(v['peak_rss_bytes'] for v in prev['stages'].values())
                # Refine the estimate using the observed union density and
                # allocation pattern. The worker still has an RSS watchdog.
                est = min(est, (2*peak - prev['baseline_rss_bytes'])/GIB + 0.2)
        r = dict(n=n, k_nb=k)
        r.update(run('scale', n, k, estimate=est))
        results['scale'].append(r)
        if 'stages' in r:
            print(f'{n:<7}{k:<7}' + ' '.join(f"{r['stages'][s]['seconds']:8.2f}/{r['stages'][s]['peak_rss_bytes']/MIB:.0f}"
                  for s in ('kernel', 'graph', 'calibrate', 'solve')), flush=True)
            if 'same_walk_dense' in r:
                print('Dense oracle, same walks: ' + ' '.join(f"{s}: {r['stages'][s]['seconds']:.2f} s / "
                    f"{r['stages'][s]['peak_rss_bytes']/MIB:.0f} MiB" for s in
                    ('dense_materialize', 'dense_calibrate', 'dense_solve')), flush=True)
        else:
            print(n, k, r['skipped'], r.get('detail', ''), flush=True)
    if 'rows' in results['truncation']:
        print('K_nb eta       mean lost q   p95 lost q   rel. t*    max rel. eig16  subspace sin²', flush=True)
        for r in results['truncation']['rows']:
            print(f"{r['k_nb']:4d} {r['eta']:8.0e}  {r['lost_mass_mean']:11.6f}  {r['lost_mass_p95']:10.6f}"
                  f" {r['t_relative']:+10.6f} {r['eigenvalue_max_relative']:14.6f} {r['subspace_mean_sin2']:14.6f}", flush=True)
    else:
        print('Truncation:', results['truncation'], flush=True)
    results['cpu_projection'] = _project(results['scale'])
    positions = 32*2047
    results['gpu_flops_per_site'] = dict(gram=2*positions**2*2048,
            recurrence_per_alpha=2*32**2*(2047-1)**2,
            cosine_distance_per_alpha=3*57344**2)
    results['wall_seconds'] = perf_counter() - start
    output = HERE / 'bench_numerics.json'
    output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {output} in {results["wall_seconds"]:.1f} s', flush=True)


if __name__ == '__main__':
    main()
