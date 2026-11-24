"""Local, checkpointed Stage B known-answer gate (SPEC_B.md v3.3).

No model extraction, network, or paid execution. Use a new output directory
for a new configuration. A failed K0 checkpoint is terminal, not retried.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy import linalg

import stage_b as b

SPEC = 'B v3.3'
FROZEN_SEEDS = (70, 71, 72)
DEVELOPMENT_SEEDS = tuple(range(1170, 1175))
VG_GRID = (6., 9., 13.5, 20.25)
DESIGN_PREDICTION_MIN, DESIGN_PAST_MAX = .55, .27


@dataclass(frozen=True)
class SyntheticParameters:
    """Section 9's fixture; only v_g can change during design calibration."""
    n: int = 1024
    sequence_length: int = 192
    chunk_length: int = 64
    width: int = 64
    transition_x: float = .05
    emission_alpha: float = .85
    initial_hidden: str = 'uniform'
    sequence_seed_offset: int = 0
    belief_coordinates: int = 3
    symbol_coordinates: int = 3
    distractor_coordinates: int = 4
    belief_mean_squared_norm: float = 2.
    symbol_mean_squared_norm: float = 1.
    v_g: float = VG_GRID[0]
    noise_scale: float = .07
    loading_seed: int = 31
    loading_qr: str = 'positive diag(R)'
    belief_columns: tuple = (1, 3)  # one-based, inclusive
    distractor_columns: tuple = (4, 7)
    symbol_columns: tuple = (8, 10)
    sites: tuple = (('absent', 0, 0), ('absent', 0, 0), ('symbol-only', 0, 1),
                    ('present', 1, 1), ('present', 1, 1), ('present', 1, 1),
                    ('scrambled', 1, 1), ('symbol-only', 0, 1))
    scramble_seed: int = 33
    distractor_noise_seed_offset: int = 100
    distractor_noise_draw_order: str = 'g, then full epsilon at sites 0 to 7 in order'
    sequence_draw_order: str = 'initial hidden state, then transition and emission draws per step'
    gate_input_record: int = 64

    def source_matrices(self):
        k = self.belief_coordinates
        t = np.full((k, k), self.transition_x)
        e = np.full((k, self.symbol_coordinates), (1 - self.emission_alpha) / (self.symbol_coordinates-1))
        np.fill_diagonal(t, 1 - (k-1)*self.transition_x)
        np.fill_diagonal(e, self.emission_alpha)
        return t, e


DEFAULT_PARAMETERS = SyntheticParameters()
ROLES = tuple(role for role, _, _ in DEFAULT_PARAMETERS.sites)
PRESENT = tuple(i for i, role in enumerate(ROLES) if role == 'present')
ABSENT_SCRAMBLED = tuple(i for i, role in enumerate(ROLES) if role in ('absent', 'scrambled'))
SYMBOL_ONLY = tuple(i for i, role in enumerate(ROLES) if role == 'symbol-only')
SCRAMBLED, = (i for i, role in enumerate(ROLES) if role == 'scrambled')
T, EMISSION = DEFAULT_PARAMETERS.source_matrices()


def generator_seeds(seeds, parameters=DEFAULT_PARAMETERS):
    return dict(sequences={s + parameters.sequence_seed_offset for s in seeds},
                distractor_noise={s + parameters.distractor_noise_seed_offset for s in seeds},
                loadings={parameters.loading_seed}, scramble={parameters.scramble_seed})


def forward_filter(symbols, parameters=DEFAULT_PARAMETERS):
    """Uniform prior at time -1. Transition, emit, then Bayes-normalise."""
    symbols = np.asarray(symbols)
    k, m = parameters.belief_coordinates, parameters.symbol_coordinates
    if (symbols.ndim != 2 or symbols.dtype.kind not in 'iu'
            or (symbols < 0).any() or (symbols >= m).any()):
        raise ValueError('expected a (sequence, time) symbol array')
    t, e = parameters.source_matrices()
    prior = np.full((len(symbols), k), 1/k)
    post = np.empty((*symbols.shape, k))
    for j, y in enumerate(symbols.T):
        prior = (prior @ t) * e.T[y]
        prior /= prior.sum(1, keepdims=True)
        post[:, j] = prior
    return post


def source(n, seed, parameters=DEFAULT_PARAMETERS):
    rng = np.random.default_rng(seed + parameters.sequence_seed_offset)
    hidden = rng.integers(0, parameters.belief_coordinates, n)
    symbols = np.empty((n, parameters.sequence_length), dtype=np.int64)
    ct, ce = (m.cumsum(1)[:, :-1] for m in parameters.source_matrices())
    for j in range(parameters.sequence_length):
        hidden = (rng.random(n)[:, None] >= ct[hidden]).sum(1)
        symbols[:, j] = (rng.random(n)[:, None] >= ce[hidden]).sum(1)
    return symbols, forward_filter(symbols, parameters)


def _cut_index(parameters):
    return parameters.gate_input_record - parameters.chunk_length - 1


def next_symbol(beliefs, parameters=DEFAULT_PARAMETERS):
    t, e = parameters.source_matrices()
    return (beliefs[:, _cut_index(parameters), :] @ t) @ e


def within_symbol(belief, symbol):
    residual = belief.copy()
    for y in range(3):
        selected = symbol == y
        if selected.any():
            residual[selected] -= belief[selected].mean(0)
    u, s, _ = linalg.svd(residual, full_matrices=False)
    if sum(s > s[0] * 1e-10) != 2:
        raise ArithmeticError('within-symbol belief does not have rank 2')
    return u[:, :2]


def _variance(x, total):
    x = x - x.reshape(-1, x.shape[-1]).mean(0)
    return x * np.sqrt(total / np.mean(np.sum(x*x, axis=-1)))


def synthetic_layers(symbols, beliefs, seed, parameters=DEFAULT_PARAMETERS):
    """Draw once per site in the frozen order, regenerating the same draws at each v_g."""
    p = parameters
    n = len(symbols)
    rng = np.random.default_rng(seed + p.distractor_noise_seed_offset)
    rot = np.random.default_rng(p.loading_seed)
    donor = np.random.default_rng(p.scramble_seed).permutation(n).astype(np.int64)
    c = _variance(beliefs[:, -p.chunk_length:], p.belief_mean_squared_norm)
    e = _variance(np.eye(p.symbol_coordinates)[symbols[:, -p.chunk_length:]], p.symbol_mean_squared_norm)
    g = _variance(rng.standard_normal((n, p.distractor_coordinates)), p.v_g)

    def cols(first_last):
        first, last = first_last
        return slice(first-1, last)

    for role, beta, eta in p.sites:
        basis, r = linalg.qr(rot.standard_normal((p.width, p.symbol_columns[1])), mode='economic')
        basis *= np.where(r.diagonal() < 0, -1., 1.)
        x = p.noise_scale * rng.standard_normal((n, p.chunk_length, p.width))
        x += (g @ basis[:, cols(p.distractor_columns)].T)[:, None, :]
        if beta:
            x += beta * ((c[donor] if role == 'scrambled' else c) @ basis[:, cols(p.belief_columns)].T)
        if eta:
            x += eta * ((e[donor] if role == 'scrambled' else e) @ basis[:, cols(p.symbol_columns)].T)
        yield x


def capture(u, q):
    return float(np.linalg.norm(u.T @ q)**2 / 2)


def _site_capture(m, q, donor=None, unrelated=None):
    w, u = b.mz_eigh(m, 16)
    return (dict(eigenvalues=w, recipient=capture(u, q), scrambled_donor=capture(u, donor) if donor is not None else None,
                 unrelated_donor=capture(u, unrelated) if unrelated is not None else None), u)


def _prediction(beliefs, parameters):
    return b.build_walk(b.hellinger_distances(next_symbol(beliefs, parameters)))


def _prediction_capture(beliefs, q, parameters):
    walk = _prediction(beliefs, parameters)
    if walk.s is None:
        raise ArithmeticError(f'prediction Sinkhorn failed: {walk.report["reasons"]}')
    read, modes = _site_capture(walk.s, q)
    return walk, read, modes


def _past_capture(x, q, parameters):
    walk = b.build_walk(b.chunk_distances(x[:, :parameters.gate_input_record], views=('cut',))['cut'])
    if walk.s is None:
        raise ArithmeticError(f'layer Sinkhorn failed: {walk.report["reasons"]}')
    a = b.lazy_walk(walk.s)
    m = a @ a
    read, modes = _site_capture((m + m.T)/2, q)
    return walk, read, modes


@dataclass(frozen=True)
class Config:
    n: int
    seeds: tuple[int, ...]
    smoke: bool = False
    count: int = 99
    gains: int = 19
    parameters: SyntheticParameters = DEFAULT_PARAMETERS
    parameter_sha256: str | None = None
    calibration: bool = False

    def manifest(self):
        official = not self.smoke and self.seeds == FROZEN_SEEDS
        root = Path(__file__).resolve().parent
        sources = ('gate_b.py', 'stage_b.py', 'test_stage_b.py', 'test_spacetime.py', 'spacetime.py', 'kernels.py')
        scope = ('development calibration' if self.calibration else 'smoke (not the gate)' if self.smoke else
                 'confirmatory' if official else 'non-confirmatory (not the gate)')
        return dict(spec=SPEC, checkpoint_format=2, n=self.n, seeds=list(self.seeds), scope=scope,
                    parameters=asdict(self.parameters), parameter_sha256=self.parameter_sha256,
                    count_relabellings=self.count, gain_relabellings=self.gains,
                    roles=ROLES, thresholds=dict(K0_past_max=.3, K0_prediction_min=.5, K1_gain_min=.2,
                    K1_count_min=1, K2_K3_K4_gain_max=.05, K2_K3_K4_count_sum_max=1,
                    K5_true_rank=1, K5_unrelated_nonfirst_min=2, K6_spectrum=1e-12,
                    K6_cost=1e-10, K6_projector=1e-8),
                    sources=dict(T=T, emission=EMISSION), dependencies=dict(numpy=np.__version__, scipy=scipy.__version__),
                    code_sha256={f: sha256((root/f).read_bytes()).hexdigest() for f in sources})


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return 'inf' if x > 0 else '-inf' if x < 0 else 'nan'
    return x


class Store:
    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = _jsonable(config.manifest())
        manifest_path = self.root / 'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError('checkpoint manifest differs; use a new output directory')
        else:
            self.write('manifest', self.manifest)
        self.used = {}

    def write(self, name, value):
        path = self.root / f'{name}.json'
        self._write_text(path, json.dumps(_jsonable(value), indent=2, allow_nan=False) + '\n')

    @staticmethod
    def _write_text(path, text):
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_bytes(text.encode('utf-8'))
        tmp.replace(path)

    def write_parameters(self, name, parameters, *, status, **extra):
        self.write(name, dict(spec=SPEC, status=status, synthetic=asdict(parameters), **extra))
        path = self.root / f'{name}.json'
        digest = sha256(path.read_bytes()).hexdigest()
        self._write_text(path.with_suffix('.json.sha256'), f'{digest}  {path.name}\n')
        return digest

    def step(self, name, work, *, evidence=False):
        path = self.root / f'{name}.json'
        bundle = self.root / f'{name}.npz'
        if path.exists():
            if evidence and not bundle.is_file():
                raise ValueError(f'missing checkpoint evidence: {bundle}')
            out = json.loads(path.read_text())
            print(f'{name}: resumed ({out["seconds"]:.3f}s originally)', flush=True)
        else:
            print(f'{name}: started', flush=True)
            t = time.perf_counter()
            out = work()
            if evidence:
                out, arrays = out
                tmp = bundle.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    np.savez(f, **arrays)
                tmp.replace(bundle)
            out = _jsonable(dict(seconds=time.perf_counter()-t, **out))
            self.write(name, out)
            print(f'{name}: {out["seconds"]:.3f}s', flush=True)
        self.used[name] = out
        return out

    def load_evidence(self, name):
        with np.load(self.root / f'{name}.npz', allow_pickle=False) as f:
            return dict(f)

    def finish(self, status, **extra):
        self.write('report', dict(status=status, config=self.manifest, steps=self.used, **extra))
        print(f'{self.manifest["scope"]}: {status}', flush=True)


def _draw_record(n, seed, parameters):
    symbols, beliefs = source(n, seed, parameters)
    cut = _cut_index(parameters)
    q = within_symbol(beliefs[:, cut], symbols[:, cut])
    return symbols, beliefs, q


def run_k0(config, seed):
    parameters = config.parameters
    symbols, beliefs, q = _draw_record(config.n, seed, parameters)
    ps, pred, uf = _prediction_capture(beliefs, q, parameters)
    walks, stats, u0 = [], [], []
    for x in synthetic_layers(symbols, beliefs, seed, parameters):
        walk, read, u = _past_capture(x, q, parameters)
        walks.append(walk.s)
        stats.append(dict(graph=walk.report, past_capture=read))
        u0.append(u)
    past_max = max(stats[l]['past_capture']['recipient'] for l in PRESENT)
    passed = past_max <= .3 and pred['recipient'] >= .5
    return dict(passed=passed, present_past_max=past_max, prediction=pred, chance=16/(config.n-1),
                prediction_graph=ps.report, sites=stats), dict(symbols=symbols, beliefs=beliefs,
                layer_walks=np.stack(walks), sf=ps.s, prediction_u=uf, past_u=np.stack(u0))


def read_frozen_parameters(path):
    """Verify the committed companion SHA-256 before parsing any parameters."""
    path = Path(path)
    raw = path.read_bytes()
    checksum = path.with_suffix(path.suffix + '.sha256').read_text().split()
    actual = sha256(raw).hexdigest()
    if checksum != [actual, path.name]:
        raise ValueError(f'frozen parameter SHA-256 mismatch: {path}')
    evidence = json.loads(raw)
    if (not isinstance(evidence, dict) or evidence.get('spec') != SPEC or evidence.get('status') != 'frozen'
            or evidence.get('development_seeds') != list(DEVELOPMENT_SEEDS)):
        raise ValueError('expected parameters frozen by successful v3.3 development calibration')
    synthetic = evidence.get('synthetic')
    if not isinstance(synthetic, dict) or synthetic.get('v_g') not in VG_GRID:
        raise ValueError('frozen v_g is outside the declared grid')
    parameters = replace(DEFAULT_PARAMETERS, v_g=float(synthetic['v_g']))
    if synthetic != _jsonable(asdict(parameters)):
        raise ValueError('frozen fixture changes a parameter other than v_g')
    return parameters, actual


def _calibration_prediction(seed, parameters):
    _, beliefs, q = _draw_record(parameters.n, seed, parameters)
    walk, read, _ = _prediction_capture(beliefs, q, parameters)
    return dict(seed=seed, prediction_only_capture=read['recipient'], eigenvalues=read['eigenvalues'], graph=walk.report)


def _calibration_past(seed, parameters):
    symbols, beliefs, q = _draw_record(parameters.n, seed, parameters)
    sites = []
    for i, x in enumerate(synthetic_layers(symbols, beliefs, seed, parameters)):
        if i in PRESENT:
            walk, read, _ = _past_capture(x, q, parameters)
            sites.append(dict(site=i, capture=read['recipient'], eigenvalues=read['eigenvalues'], graph=walk.report))
    return dict(seed=seed, v_g=parameters.v_g, sites=sites, past_only_max=max(s['capture'] for s in sites))


def calibrate(output):
    """Isolated K0-only design path. It cannot run the analytic or gate suites."""
    parameters = DEFAULT_PARAMETERS
    config = Config(parameters.n, DEVELOPMENT_SEEDS, count=0, gains=0, parameters=parameters, calibration=True)
    store = Store(output, config)
    design_sha = store.write_parameters('design_parameters', parameters, status='design only',
                                        development_seeds=DEVELOPMENT_SEEDS, grid=VG_GRID)
    summary = dict(design_parameter_sha256=design_sha, prediction_minimum_required=DESIGN_PREDICTION_MIN,
                   past_maximum_allowed=DESIGN_PAST_MAX, prediction_captures=[], grid=[])
    try:
        for seed in DEVELOPMENT_SEEDS:
            row = store.step(f'{seed}-prediction', lambda: _calibration_prediction(seed, parameters))
            summary['prediction_captures'].append(dict(seed=seed, capture=row['prediction_only_capture']))
        summary['prediction_minimum'] = min(row['capture'] for row in summary['prediction_captures'])
        if summary['prediction_minimum'] < DESIGN_PREDICTION_MIN:
            store.finish('fixture uninformative at design', reason='prediction-only minimum below 0.55', **summary)
            return 2
        for v_g in VG_GRID:
            candidate = replace(parameters, v_g=v_g)
            rows = [store.step(f'vg-{v_g:g}-{seed}', lambda: _calibration_past(seed, candidate))
                    for seed in DEVELOPMENT_SEEDS]
            max_past = max(row['past_only_max'] for row in rows)
            summary['grid'].append(dict(v_g=v_g, past_only_max=max_past, by_seed=[
                dict(seed=row['seed'], past_only_max=row['past_only_max'],
                     captures=[dict(site=s['site'], capture=s['capture']) for s in row['sites']]) for row in rows]))
            if max_past <= DESIGN_PAST_MAX:
                digest = store.write_parameters('parameters', candidate, status='frozen',
                                                 development_seeds=DEVELOPMENT_SEEDS)
                store.finish('parameters frozen', chosen_v_g=v_g, parameter_file='parameters.json',
                             parameter_sha256=digest, **summary)
                return 0
        store.finish('fixture uninformative at design', reason='no grid point had maximum past-only capture at most 0.27', **summary)
        return 2
    except Exception as exc:
        store.finish('operational or numerical failure', error=f'{type(exc).__name__}: {exc}', **summary)
        raise


def _reference_sets(config, seed, symbols):
    named = (('random', None, 90+seed, 52), ('symbol', symbols, 95+seed, 51))
    ps, meta = {}, {}
    for name, group, gain_seed, count_seed in named:
        ps[name], meta[name] = {}, {}
        for task, count, s in [('gains', config.gains, gain_seed), ('counts', config.count, count_seed)]:
            ps[name][task], meta[name][task] = b.relabellings(config.n, count, np.random.default_rng(s), groups=group)
            meta[name][task]['seed'] = s
    return ps, meta


def site_control(a, s, q, q_scrambled, q_unrelated, refs):
    product = b.product_walk(a, s)
    observed, u = _site_capture(product, q, q_scrambled, q_unrelated)
    spectrum = b.mz_eigh(product, min(64, len(a)-1), vectors=False)
    data, evidence = dict(observed=observed, references={}), dict(observed_u=u)
    for name, perms in refs.items():
        reads, modes = [], []
        for p in perms['gains']:
            stat, u = _site_capture(b.product_walk(a, s, p), q, q_scrambled, q_unrelated)
            reads.append(stat)
            modes.append(u)
        best = max(r['recipient'] for r in reads)
        maxima, err = zip(*(b.null_maximum(a, s, p) for p in perms['counts']), strict=True)
        data['references'][name] = dict(captures=reads, max_recipient_capture=best,
                                        gain=observed['recipient']-best,
                                        count=b.count_from_spectra(spectrum, maxima, err))
        evidence[f'{name}_u'] = np.stack(modes)
    return data, evidence


def _depth_record(run, q, k):
    return dict(values=run.values, **run.report,
                recipient_belief_energy=[float(np.linalg.norm(q.T @ g[:, :k])**2) for g in run.vectors])


def _projection_error(u, v):
    dot = np.sum(u*v, axis=0)
    # Frobenius norm of the difference of rank-one projectors; this form
    # avoids cancellation under perfect equivariance.
    return np.linalg.norm(u - v*np.where(dot < 0, -1., 1.), axis=0) * np.sqrt(1+np.abs(dot))


def _singles(w, separation):
    neighbour_gaps = np.abs(np.diff(w))
    return (np.r_[np.inf, neighbour_gaps] >= separation) & (np.r_[neighbour_gaps, np.inf] >= separation)


def k6_flat(n=80, *, seed=6, permutations=None, diagnose_reused=False):
    """Exact flat layer, analytic separable spectrum, and production solves.

    Use two well-separated prediction directions plus a tied complement.
    Complete graph + sign contrasts is zero-diagonal when n is a multiple
    of four; otherwise its diagonal is allowed in this constructed fixture.
    """
    flat = (np.ones((n, n)) - np.eye(n)) / (n-1)
    j = np.arange(n)
    features = np.stack([np.cos(np.pi * (j+.5)/n), np.cos(2*np.pi * (j+.5)/n)], axis=1) * np.sqrt(2/n)
    if n % 4 == 0:
        features = np.stack([(-1.)**j, (-1.)**(j//2)], axis=1)/np.sqrt(n)
    s = flat + (features * [.21, .09]) @ features.T
    if n % 4 == 0:
        # Subtract .30/n from the diagonal and redistribute it off-diagonal.
        s += .30/(n-1) * (np.ones((n, n))/n - np.eye(n))
        np.fill_diagonal(s, 0.)
    nsites = 8
    a = b.lazy_walk(flat)
    partner = b.prepare_partner([flat]*nsites)
    observed = b.solve_product(partner, s)
    e = b.energy(partner.past, observed)
    w, u = b.mz_eigh(b.product_walk(a, s))
    site_isolated = _singles(w, 1e-7)
    # Guard against a reused solver silently returning too few members of
    # an exactly tied eigenspace. Flat blocks separate in depth and space.
    a_mz = (n-2)/(2*(n-1))
    temporal = b.spacetime.temporal_eigs(nsites, partner.t)
    pexact = np.sort((np.ones(n-1)[:, None]+temporal).ravel())
    grid = (1-w[:, None])/partner.c[0]+temporal
    order = np.argsort(grid, axis=None)
    exact = grid.ravel()[order]
    isolated_depth = _singles(exact, 1e-7)
    spatial, depth = np.unravel_index(order[isolated_depth], grid.shape)
    _, envelopes = linalg.eigh(b.spacetime.path_lap(np.eye(nsites)))

    def full_isolated(u):
        return (envelopes[:, None, depth] * u[None, :, spatial]).reshape(nsites*n, -1)
    reused_diagnostic = None
    if diagnose_reused:
        # Reproduce a skipped multiplicity in the unmodified, single-vector
        # solver at a rounded balance differing only at the 1e-13 scale.
        t_test = 6.568535592272027 if n == 80 else partner.t
        ms = np.stack([b.product_walk(a, s)] * nsites)
        old_w, old_v = b.spacetime.modes(ms, partner.c, t_test, k=partner.modes)
        old_resid = b.spacetime.residual(ms, partner.c, t_test, old_w, old_v)
        want = np.sort(((1-w[:, None])/partner.c[0]+b.spacetime.temporal_eigs(nsites, t_test)).ravel())[:len(old_w)]
        reused_diagnostic = dict(t=t_test, values=old_w, exact=want, residuals=old_resid,
                                 ordering_error=float(np.max(abs(old_w-want))))
    past_error = float(np.max(np.abs(partner.past.values-pexact[:partner.modes])))
    product_error = float(np.max(np.abs(observed.values-exact[:partner.modes])))
    depth_isolated = isolated_depth[:partner.modes]
    u_depth = observed.vectors.reshape(nsites*n, -1)[:, depth_isolated]
    if permutations is None:
        permutations = b.relabellings(n, 3, np.random.default_rng(seed))[0]
    comparisons, wrong = [], []
    for p in permutations[:3]:
        wp, up = b.mz_eigh(b.product_walk(a, s, p))
        eq = float(np.max(np.abs(w-wp)))
        pe = float(np.max(_projection_error(u[p][:, site_isolated], up[:, site_isolated]), initial=0.))
        run = b.solve_product(partner, s, p)
        ep = b.energy(partner.past, run)
        error = None if e is None or ep is None else abs(ep-e)
        ud = run.vectors.reshape(nsites*n, -1)[:, depth_isolated]
        dep = float(np.max(_projection_error(observed.vectors[:, p, :].reshape(nsites*n, -1)[:, depth_isolated], ud), initial=0.))
        all_dep = float(np.max(_projection_error(full_isolated(u[p]), full_isolated(up)), initial=0.))
        wrong.append(run)
        comparisons.append(dict(site_spectral_error=eq, site_projector_error=pe, depth_projector_error=dep,
                                all_isolated_depth_projector_error=all_dep, energy=ep, energy_error=error,
                                solve=_depth_record(run, features, partner.k)))
    count = b.site_count(a, s, permutations)
    delta = b.increment_reference((partner.past, partner.past), (observed, observed), [(run, run) for run in wrong])
    passed = (bool(site_isolated.any() and u_depth.shape[1]) and past_error <= 1e-10 and product_error <= 1e-10
              and all(c['site_spectral_error'] <= 1e-12 and c['energy_error'] is not None and c['energy_error'] <= 1e-10
                      and max(c['site_projector_error'], c['depth_projector_error'], c['all_isolated_depth_projector_error']) <= 1e-8
                      for c in comparisons)
              and count['count'] == 0 and partner.past.report['picture_status'] == 'unresolved'
              and delta['status'] == 'withheld' and delta['reason'] == 'unresolved')
    return dict(passed=passed, n=n, c=partner.c, t=partner.t, flat_a_mz=a_mz, site_spectrum=w,
                site_isolated=np.flatnonzero(site_isolated), depth_isolated=np.flatnonzero(isolated_depth),
                solver_window_isolated=np.flatnonzero(depth_isolated),
                exact_past_spectrum_error=past_error, exact_product_spectrum_error=product_error,
                past_lowest_multiplicity=n-1, past=_depth_record(partner.past, features, partner.k),
                observed=_depth_record(observed, features, partner.k), energy=e, comparisons=comparisons,
                count=count, reused_solver_diagnostic=reused_diagnostic,
                delta={**delta, 'display': 'unresolved' if delta.get('reason') == 'unresolved' else delta['status']})


def _conditions(true, unrelated):
    def gain(sites, l, kind):
        return sites[l]['references'][kind]['gain']

    def count(sites, l, kind):
        return sites[l]['references'][kind]['count']['count']

    k1 = all(min(gain(true, l, k) for k in ('random', 'symbol')) >= .2 and count(true, l, 'symbol') >= 1 for l in PRESENT)
    k2_sum = sum(count(true, l, 'symbol') for l in SYMBOL_ONLY)
    k2 = k2_sum <= 1 and all(gain(true, l, 'symbol') <= .05 for l in SYMBOL_ONLY)
    k3_sum = sum(count(true, l, 'random') for l in ABSENT_SCRAMBLED)
    k3 = k3_sum <= 1 and all(gain(true, SCRAMBLED, kind) <= .05 for kind in ('random', 'symbol'))
    k4_sum = sum(count(unrelated, l, 'random') for l in range(8))
    k4 = k4_sum <= 1 and all(gain(unrelated, l, k) <= .05 for l in range(8) for k in ('random', 'symbol'))
    return dict(K1=k1, K2=k2, K3=k3, K4=k4, K2_symbol_count_sum=k2_sum,
                K3_random_count_sum=k3_sum, K4_random_count_sum=k4_sum)


def remaining(config, seed, store):
    name = f'{seed}-K0'
    rec = store.load_evidence(name)
    cut = _cut_index(config.parameters)
    symbols = rec['symbols'][:, cut]
    q = within_symbol(rec['beliefs'][:, cut], symbols)
    scramble = np.random.default_rng(config.parameters.scramble_seed).permutation(config.n)
    q_scrambled = q[scramble]
    layers = [b.Walk(s, stats['graph']) for s, stats in zip(rec['layer_walks'], store.used[name]['sites'], strict=True)]
    refs, meta = _reference_sets(config, seed, symbols)
    store.step(f'{seed}-relabellings', lambda: meta)

    def unrelated_input():
        y, beliefs, qf = _draw_record(config.n, 80+seed, config.parameters)
        sf = _prediction(beliefs, config.parameters)
        if sf.s is None:
            raise ArithmeticError(f'unrelated Sinkhorn failed: {sf.report["reasons"]}')
        recipient, u = _site_capture(sf.s, q, q_scrambled, qf)
        return dict(graph=sf.report, prediction_capture=recipient), dict(symbols=y, beliefs=beliefs, sf=sf.s, q=qf, u=u)

    unrel_name = f'{seed}-unrelated'
    unrel = store.step(unrel_name, unrelated_input, evidence=True)
    other = store.load_evidence(unrel_name)
    futures = dict(true=b.Walk(rec['sf'], store.used[name]['prediction_graph']), unrelated=b.Walk(other['sf'], unrel['graph']))
    cells = {}
    for label, sf in futures.items():
        qf = other['q'] if label == 'unrelated' else None
        cells[label] = [store.step(f'{seed}-{label}-site-{l}',
                        lambda l=l: site_control(b.lazy_walk(layers[l].s), sf.s, q, q_scrambled, qf, refs), evidence=True)
                        for l in range(8)]

    def partner_input():
        try:
            pt = b.prepare_partner(layers)
        except b.Withheld as exc:
            return dict(status='withheld', reason=str(exc)), {}
        past = _depth_record(pt.past, q, pt.k)
        # Past-only recipient/donor captures are reported before depth mixing.
        past['local_scrambled_donor_capture'] = [capture(u, q_scrambled) for u in rec['past_u']]
        past['local_unrelated_donor_capture'] = [capture(u, other['q']) for u in rec['past_u']]
        return dict(status='computed' if pt.past.report['accepted'] else 'withheld', c=pt.c, t=pt.t, past=past), dict(
            lazy=pt.lazy, past_values=pt.past.values, past_vectors=pt.past.vectors, c=pt.c, t=pt.t)

    pt_name = f'{seed}-past-depth'
    pr = store.step(pt_name, partner_input, evidence=True)
    ranks = {}
    if pr['status'] == 'computed':
        pt = store.load_evidence(pt_name)
        partner = b.Partner(pt['lazy'], pt['c'], float(pt['t']), b.Depth(pt['past_values'], pt['past_vectors'], pr['past']), 8, 16, 24, False)
        for label, sf in futures.items():
            def depth_job(p=None):
                try:
                    run = b.solve_product(partner, sf, p, own_units=p is None)
                except b.Withheld as exc:
                    return dict(status='withheld', energy=None, reason=str(exc)), {}
                e = b.energy(partner.past, run)
                return dict(status='computed' if e is not None else 'withheld', energy=e,
                            **_depth_record(run, q, partner.k)), dict(modes=run.vectors)

            obs = store.step(f'{seed}-{label}-depth', depth_job, evidence=True)
            values = [store.step(f'{seed}-{label}-depth-null-{i:02d}',
                                  lambda p=p: depth_job(p), evidence=True)['energy']
                      for i, p in enumerate(refs['symbol']['gains'])]
            ranks[label] = b.energy_reference(obs['energy'], values)
    else:
        ranks = {label: dict(status='withheld', p=None, reason='past-only partner inadmissible') for label in futures}
    flat = store.step(f'{seed}-K6', lambda: k6_flat(config.n, permutations=refs['random']['counts']))
    return store.step(f'{seed}-K1-K6', lambda: dict(**_conditions(cells['true'], cells['unrelated']),
                K5=ranks, K6=flat['passed']))


def run_fixtures(store):
    # These check functions do not import pytest; the CLI needs only numpy/SciPy.
    import test_stage_b
    import test_spacetime

    suites = [('stage-b', {n: f.__wrapped__ for n, f in vars(test_stage_b).items()
                           if n.startswith('test_') and hasattr(f, '__wrapped__')}),
              ('reused-spacetime', {n: f for n, f in vars(test_spacetime).items() if n.startswith('test_')})]
    for prefix, checks in suites:
        for name, fixture in sorted(checks.items()):
            def check():
                try:
                    return dict(passed=True, measurements=fixture())
                except Exception as exc:
                    return dict(passed=False, error=f'{type(exc).__name__}: {exc}')
            result = store.step(f'fixture-{prefix}-{name[5:]}', check)
            if not result['passed']:
                return False
    return True


def run(config, output, only):
    if only != 'fixtures':
        if set(config.seeds) & set(DEVELOPMENT_SEEDS):
            raise ValueError('development seeds may only be used by --calibrate')
        if set(config.seeds) & set(FROZEN_SEEDS) and (
                config.seeds != FROZEN_SEEDS or config.parameter_sha256 is None or config.smoke):
            raise ValueError('confirmatory seeds require the verified frozen file and all three seeds together')
    store = Store(output, config)
    try:
        if not run_fixtures(store):
            store.finish('fixture failure')
            return 1
        if only == 'fixtures':
            store.finish('fixtures passed')
            return 0
        for seed in config.seeds:
            suitability = store.step(f'{seed}-K0', lambda: run_k0(config, seed), evidence=True)
            if not suitability['passed']:
                store.finish('fixture uninformative', stopped_at_seed=seed)
                return 2
        if only == 'k0':
            store.finish('K0 passed; remaining checks not run')
            return 0
        results = [remaining(config, seed, store) for seed in config.seeds]
        k5_true = all(r['K5']['true'].get('rank') == 1 for r in results)
        k5_nr = sum(r['K5']['unrelated'].get('rank', 1) > 1 for r in results)
        accepted = all(all(x['K5'][label]['status'] == 'computed' for label in ('true', 'unrelated')) for x in results)
        k5 = accepted and k5_true and k5_nr >= 2
        passed = k5 and all(all(r[f'K{i}'] for i in (1, 2, 3, 4, 6)) for r in results)
        status = ('withheld (inadmissible K5 walk or solve)' if not accepted else 'gate passed' if passed and store.manifest['scope'] == 'confirmatory'
                  else 'diagnostic finished (not the gate)' if store.manifest['scope'] != 'confirmatory' else 'gate conditions not met')
        store.finish(status, K5_passed=k5, K5_true_rank_one=k5_true, K5_unrelated_nonfirst=k5_nr,
                     frozen_conditions_met=passed)
        return 0 if passed else 1
    except Exception as exc:
        store.finish('operational or numerical failure', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', choices=('fixtures', 'k0', 'all'), help='default: all')
    ap.add_argument('--calibrate', action='store_true', help='run only v3.3 development calibration after separate fixtures')
    ap.add_argument('--parameters', type=Path, help='frozen parameters.json with a matching parameters.json.sha256')
    ap.add_argument('--seeds', type=int, nargs='+')
    ap.add_argument('--smoke-n', type=int, help='diagnostic only; smaller N, one explicit non-frozen seed, reduced null loops')
    args = ap.parse_args()
    if args.calibrate:
        if any(x is not None for x in (args.only, args.parameters, args.seeds, args.smoke_n)):
            ap.error('--calibrate accepts only --out')
        return calibrate(args.out)
    only = args.only or 'all'
    seeds = tuple(FROZEN_SEEDS if args.seeds is None else sorted(args.seeds))
    if args.smoke_n is not None:
        if not 65 <= args.smoke_n < DEFAULT_PARAMETERS.n or len(seeds) != 1 or seeds[0] in FROZEN_SEEDS:
            ap.error('smoke requires 65<=N<1024 and one explicit non-frozen seed')
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        ap.error('seeds must be distinct and non-negative')
    if set(seeds) & set(DEVELOPMENT_SEEDS):
        ap.error('development seeds may only be used by --calibrate')
    if only != 'fixtures' and set(seeds) & set(FROZEN_SEEDS):
        if seeds != FROZEN_SEEDS or args.parameters is None:
            ap.error('confirmatory runs require all seeds 70 71 72 and --parameters from successful calibration')
    try:
        parameters, digest = (DEFAULT_PARAMETERS, None) if args.parameters is None else read_frozen_parameters(args.parameters)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    cfg = (Config(parameters.n, seeds, parameters=parameters, parameter_sha256=digest) if args.smoke_n is None else
           Config(args.smoke_n, seeds, True, 5, 3, parameters, digest))
    return run(cfg, args.out, only)


if __name__ == '__main__':
    raise SystemExit(main())
