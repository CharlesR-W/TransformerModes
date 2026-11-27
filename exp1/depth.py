"""Local SPEC_DEPTH v3.1 pipeline. Run `python depth.py --help` for commands."""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

# Set before importing NumPy, also when invoked without the documented wrapper.
os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS'] = os.environ.get('TM_DEPTH_BLAS_THREADS', '3')
import numpy as np
import scipy

import depth_dm as dm
import spacetime as st

HERE = Path(__file__).resolve().parent
SPEC = HERE.parent / 'SPEC_DEPTH.md'
ROOT = Path('/data/tm_acts')
PROBE = Path(os.environ.get('TM_DEPTH_PROBE', ROOT / 'probe/probe.json'))
SYNTH = Path(os.environ.get('TM_DEPTH_SYNTH', ROOT / 'depth_acts'))
ACTS_ROOT = os.environ.get('TM_DEPTH_ACTS')  # When set, activation directories are ACTS_ROOT/<model>.
ACTS = {m: Path(ACTS_ROOT) / m if ACTS_ROOT else ROOT / f'{m}_dl' / m for m in ('pythia', 'olmo')}
MODELS = tuple(ACTS)
MODE = ('tstar', 'tstar_div16')
AUDITED = {'doc00', 'd32', 'rand'}
SOURCES = (0, 1, 2, 4)
PATHS = (*SOURCES, 's0', 's1', 's2', 's3')
SETS = {
    **{f'doc{c:02}': ((c,), 2047) for c in SOURCES},
    'pairA': ((0, 1), 1024), 'pairB': ((2, 4), 1024), 'd04': (SOURCES, 512),
    'd08': (tuple(range(8)), 256), 'd16': (tuple(range(16)), 128),
    'd32': (tuple(range(32)), 64), 'rand': (tuple(f'r{i}' for i in range(4)), 512),
}
COHORTS = ('in_set', 'seen_sequence_new_position', 'unseen_sequence')
KS = (8, 16, 64)
LAYERS, WIDTH, KEEP, BLOCK = 17, 2048, 64, 512
GATE_LIMIT, STOP_LIMIT = 4_500_000_000, 5_500_000_000


def read_json(path):
    with Path(path).open() as f:
        return json.load(f)


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def json_bytes(obj):
    return (json.dumps(jsonable(obj), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(json_bytes(obj))
    tmp.replace(path)


def digest(obj):
    return hashlib.sha256(json_bytes(obj)).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def context_name(ctx):
    return f'c{ctx:02}' if isinstance(ctx, int) else ctx


def act_path(model, ctx):
    return (ACTS[model] if isinstance(ctx, int) else SYNTH / model) / (context_name(ctx) + '.npy')


def row_key(row):
    return row[0], row[1]


def make_manifest(probe, inputs=None):
    ids = {i: c['token_ids'] for i, c in enumerate(probe['contexts'][:32])}
    if len(ids) != 32 or any(len(v) != 2047 for v in ids.values()):
        raise ValueError('expected 32 documents with 2,047 token ids each')
    contexts = {context_name(i): {'context': i, 'kind': 'natural', 'subset': probe['contexts'][i]['subset']}
                for i in ids}
    allowed = set(x for seq in ids.values() for x in seq)
    if inputs is not None and (inputs['seed_random'], inputs['seed_shuffle']) != (20261126, 20261127):
        raise ValueError('synthetic seeds do not match the spec')
    for c in [*(f'r{i}' for i in range(5)), *(f's{i}' for i in range(4))]:
        data = None if inputs is None else inputs['contexts'][c]
        ids[c] = [None] * 2047 if data is None else data['ids']
        if data is not None and (len(ids[c]) != 2047 or not set(ids[c]) <= allowed):
            raise ValueError(f'invalid synthetic ids: {c}')
        contexts[c] = {'context': c, 'kind': 'random' if c[0] == 'r' else 'shuffled'}
        if c[0] == 's':
            source = SOURCES[int(c[1:])]
            perm = None if data is None else data['perm']
            if data is not None and (data['source_context'] != source or sorted(perm) != list(range(1, 2048))
                                     or ids[c] != [ids[source][p - 1] for p in perm]):
                raise ValueError(f'invalid shuffled source map: {c}')
            contexts[c].update(source_context=source, perm=perm)

    rng = np.random.default_rng(20261128)
    positions = np.arange(1, 2048)
    picked = {}

    def draw(ctx, n, exclude=False):
        pop = np.setdiff1d(positions, picked[ctx], assume_unique=True) if exclude else positions
        return np.sort(rng.choice(pop, n, replace=False))

    def rows(c, ps):
        return [[c, int(p), ids[c][p - 1]] for p in ps]

    panels = {'natural': [], 'random': [], 'shuffled': []}
    for kind, choices in (
        ('natural', [(c, 32) for c in range(32)]),
        ('random', [(f'r{i}', 128 if i < 4 else 256) for i in range(5)]),
        ('shuffled', [(f's{i}', 256) for i in range(4)]),
    ):
        for ctx, n in choices:
            picked[ctx] = draw(ctx, n)
            panels[kind].extend(rows(ctx, picked[ctx]))

    sets, cohorts, references = {}, {}, {}
    for name, (ctxs, n) in SETS.items():
        sets[name] = [r for c in ctxs for r in rows(c, positions if n == 2047 else draw(c, n, True))]
        keys = {row_key(r) for r in sets[name]}
        cohorts[name] = {kind: [COHORTS[0] if row_key(r) in keys else COHORTS[1] if r[0] in ctxs
                                else COHORTS[2] for r in panel] for kind, panel in panels.items()}
        if name == 'rand':
            references[name] = dict(panel='random', cohort=COHORTS[1], unseen_sequence=COHORTS[2],
                                    label='new positions of r0–r3; r4 is an unseen sequence')
        elif n == 2047:
            references[name] = dict(panel='natural', cohort=COHORTS[2], label='other documents (own document entirely in set)')
        else:
            references[name] = dict(panel='natural', cohort=COHORTS[1], label='new positions of construction documents')
    return dict(schema=1, seed=20261128, token_fields=['context', 'position', 'token_id'],
                contexts=contexts, sets=sets, panels=panels, cohorts=cohorts, references=references,
                paths={context_name(c): rows(c, positions) for c in PATHS},
                synthetic_inputs=None if inputs is None else {
                    k: inputs[k] for k in ('seed_random', 'seed_shuffle', 'allowed_ids_sha256', 'sha256')})


def manifest_now(out):
    syn = SYNTH / 'inputs.json'
    m = make_manifest(read_json(PROBE), read_json(syn) if syn.exists() else None)
    write_json(out / 'manifest.json', m)
    return m


def grouped(rows):
    groups = defaultdict(list)
    for i, (ctx, pos, _) in enumerate(rows):
        groups[ctx].append((i, pos))
    return [(ctx, *np.array(pairs, dtype=np.intp).T) for ctx, pairs in groups.items()]


def load_layer(model, groups, layer, n):
    x = np.empty((n, WIDTH), dtype=np.float64)
    for ctx, ii, pp in groups:
        mmap = np.load(act_path(model, ctx), mmap_mode='r', allow_pickle=False)
        if mmap.shape != (LAYERS, 2048, WIDTH) or mmap.dtype != np.float32:
            raise ValueError(f'invalid activation array: {act_path(model, ctx)}')
        x[ii] = mmap[layer, pp]
        del mmap
    return dm.unit_vectors(x)


def build_sites(model, rows):
    """Keep exactly one stack each of vectors, walks and full site eigensystems."""
    n = len(rows)
    x = np.empty((LAYERS, n, WIDTH))
    ms = np.empty((LAYERS, n, n))
    eigenvalues = np.empty((LAYERS, n))
    eigenvectors = np.empty_like(ms)
    kernels, stats = [], []
    selection = grouped(rows)
    for l in range(LAYERS):
        x[l] = load_layer(model, selection, l, n)
        ker = dm.radius_kernel(x[l])
        p = ker.pop('P')
        ms[l] = p
        del p
        ms[l] *= .5
        ms[l].flat[::n + 1] += .5
        vals, vecs = np.linalg.eigh(ms[l])
        eigenvalues[l], eigenvectors[l] = vals[::-1], vecs[:, ::-1]
        del vecs
        kernels.append(ker)
        stats.append({k: ker[k] for k in ('err', 'degree_median', 'degree_max', 'tie_share')})
    c = (1 - eigenvalues[:, 1:9]).mean(axis=1)
    tstar = st.balance_direct(ms, c, eigs=(eigenvalues, eigenvectors))
    # P and M have the same eigenvectors. Retain the nonconstant site modes only.
    site_modes = np.ascontiguousarray(eigenvectors[:, :, 1:KEEP + 1])
    del eigenvectors
    return x, ms, kernels, stats, c, tstar, site_modes, 2 * eigenvalues[:, 1:KEEP + 1] - 1


def bands(vals):
    links = np.diff(vals) < .01 * vals[:-1]
    endpoints = [0, *(np.flatnonzero(~links) + 1), len(vals)]
    groups = [dict(first=int(a + 1), last=int(b), unresolved=bool(b == len(vals) and links[-1]),
                   complete_in_release=bool(b <= KEEP)) for a, b in zip(endpoints, endpoints[1:]) if b - a > 1]
    return groups, {str(k): bool(links[k - 1]) for k in KS}


def solve(ms, c, tstar, strength):
    t = tstar if strength == MODE[0] else tstar / 16
    solver = st.Solver(ms, c, t)
    for k in (72, 80, 88, 96):
        vals, vec = st.modes(ms, c, t, k=k, tol=1e-10, solver=solver)
        if vals[-1] - vals[-2] >= .01 * vals[-2] or k == 96:
            break
        del vals, vec
    del solver
    res = st.residual(ms, c, t, vals, vec)
    if not np.all(np.isfinite(res) & (res < 1e-7)):
        raise RuntimeError(f'joint solve residual {res.max():.3g} at {strength}')
    groups, cuts = bands(vals)
    return dict(vecs=np.ascontiguousarray(vec[:, :, :KEEP]), vals=vals[:KEEP].copy(), c=c, t=t,
                tstar=tstar, guard_eigenvalues=vals, residual=res, bands=groups, cuts_band=cuts)


def embedding_measures(f, rows):
    weights = np.einsum('lnk,lnk->lk', f, f)
    p = np.einsum('lnk,lnk->nk', f, f) / weights.sum(axis=0)
    weak = weights < 1e-6
    adjacent = np.einsum('lnk,lnk->lk', f[1:], f[:-1]) / np.sqrt(weights[1:] * weights[:-1]).clip(1e-300)
    adjacent[weak[1:] | weak[:-1]] = np.nan
    count = np.isfinite(adjacent).sum(axis=0)
    end = np.einsum('nk,nk->k', f[0], f[-1]) / np.sqrt(weights[0] * weights[-1]).clip(1e-300)
    end[weak[0] | weak[-1]] = np.nan
    groups = grouped(rows)
    between = None if len(groups) == 1 else sum(len(i) * f[:, i].mean(axis=1)**2 for _, i, _ in groups).sum(axis=0) / weights.sum(axis=0)
    extremes = {}
    for l in (0, 8, 16):
        ordered = np.argsort(f[l], axis=0, kind='stable')
        extremes[str(l)] = {'low': ordered[:8].T, 'high': ordered[-8:][::-1].T}
    return dict(depth_weight=weights, token_support=1 / (len(rows) * (p**2).sum(axis=0)),
                document_share=between, weak_slice=weak,
                adjacent_cos_mean=np.nansum(adjacent, axis=0) / np.where(count, count, np.nan),
                adjacent_cos_count=count, endpoint_cos=end, extremes=extremes)


def t_matrices(jm):
    l = len(jm['c'])
    return (np.diag(1 / (2 * jm['c'])) + jm['t'] * st.path_lap(np.eye(l)))[None] - jm['vals'][:, None, None] * np.eye(l)


def extension_config(jm):
    mat = t_matrices(jm)
    sv = np.linalg.svd(mat, compute_uv=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond = sv[:, 0] / sv[:, -1]
    blocked = ~np.isfinite(cond) | (cond > 1e8)
    return dict(matrix=mat, smallest_sv=sv[:, -1], condition=cond, withheld=blocked,
                max_relative_residual=np.where(blocked, np.nan, 0.0))


def extend(rows, jm, config, *, track=True):
    """Upstream joint_forward equation, with batched RHS and a residual per trajectory."""
    b = (rows @ jm['vecs']) / (2 * jm['c'][:, None, None])
    y = np.full_like(b, np.nan)
    for j, mat in enumerate(config['matrix']):
        if config['withheld'][j]:
            continue
        y[:, :, j] = sol = np.linalg.solve(mat, b[:, :, j])
        if track:
            denom = np.linalg.norm(b[:, :, j], axis=0)
            err = np.linalg.norm(mat @ sol - b[:, :, j], axis=0) / np.where(denom > 0, denom, 1)
            config['max_relative_residual'][j] = max(config['max_relative_residual'][j], err.max(initial=0))
    return y


def algebra_block(rows, jm, config):
    """Use the verbatim upstream joint_forward on solvable modes for this check."""
    keep = ~config['withheld']
    if keep.all():
        return dm.joint_forward(rows, jm)[0]
    if not keep.any():
        return np.empty((len(rows), rows.shape[1], 0))
    return dm.joint_forward(rows, {**jm, 'vecs': jm['vecs'][:, :, keep], 'vals': jm['vals'][keep]})[0]


def fitted_rows(ms, ii, out):
    for l, m in enumerate(ms):
        out[l] = m[ii]
        out[l] *= 2
        out[l, np.arange(len(ii)), ii] -= 1
    return out


def squared_rel(actual, target):
    return np.sum((actual - target)**2, axis=(0, 1)), np.sum(target**2, axis=(0, 1))


def relative(num, den):
    return np.sqrt(np.divide(num, den, out=np.zeros_like(num), where=den > 0))


def replay_and_algebra(work, model, rows, x, ms, kernels, solutions, configs, audited):
    n = len(rows)
    rng = np.random.default_rng(20261130)
    ii = rng.choice(n, 256, replace=False)
    results = {s: {} for s in solutions}
    totals = {s: np.zeros((2, KEEP)) for s in solutions}
    block = np.empty((LAYERS, 256, n))
    for first in range(0, n, 256):
        ix = np.arange(first, min(first + 256, n))
        ps = fitted_rows(ms, ix, block[:, :len(ix)])
        for s, jm in solutions.items():
            keep = ~configs[s]['withheld']
            y = algebra_block(ps, jm, configs[s])
            n2, d2 = squared_rel(y, jm['vecs'][:, ix][:, :, keep])
            totals[s][:, keep] += [n2, d2]
    for s, (a, b) in totals.items():
        per = relative(a, b)
        overall = float(relative(a.sum(), b.sum()))
        results[s]['algebra'] = dict(relative_l2=overall, max_mode_relative_l2=float(per.max()),
                                     checked_modes=int((~configs[s]['withheld']).sum()),
                                     passed=bool(b.sum() > 0 and max(overall, per.max()) < 1e-8))
    for size, sequence in ((256, ii), (7, rng.permutation(ii))):
        tv = np.zeros(LAYERS)
        totals = {s: np.zeros((2, KEEP)) for s in solutions}
        for first in range(0, 256, size):
            ix = sequence[first:first + size]
            replay = block[:, :len(ix)]
            selection = grouped([rows[j] for j in ix])
            for l in range(LAYERS):
                q = load_layer(model, selection, l, len(ix))
                replay[l], _ = dm.radius_rows(q, x[l], kernels[l], mask_self=ix)
                p = 2 * ms[l, ix]
                p[np.arange(len(ix)), ix] -= 1
                tv[l] = max(tv[l], .5 * np.abs(replay[l] - p).sum(axis=1).max())
            for s, jm in solutions.items():
                keep = ~configs[s]['withheld']
                y = extend(replay, jm, configs[s])
                n2, d2 = squared_rel(y[:, :, keep], jm['vecs'][:, ix][:, :, keep])
                totals[s][:, keep] += [n2, d2]
        for s, (a, b) in totals.items():
            per = relative(a, b)
            overall = float(relative(a.sum(), b.sum()))
            results[s][f'replay_{size}'] = dict(relative_l2=overall, max_mode_relative_l2=float(per.max()),
                max_total_variation=float(tv.max()), layer_max_total_variation=tv, indices=sequence,
                checked_modes=int((~configs[s]['withheld']).sum()),
                passed=bool(b.sum() > 0 and tv.max() < 1e-9 and max(overall, per.max()) < 1e-7))
    if audited:
        ix = ii[:64]
        vals = np.empty((3, len(ix), n))
        for j, l in enumerate((0, 8, 16)):
            vals[j] = 2 * ms[l, ix]
            vals[j, np.arange(len(ix)), ix] -= 1
        np.savez(work / 'audit_rows.npz', rows=vals, indices=ix, layers=np.array([0, 8, 16]))
    write_json(work / 'checks.json', results)
    for s, checks in results.items():
        if not all(ch['passed'] for ch in checks.values()):
            raise RuntimeError(f'algebra or replay failed at {s}; see {work / "checks.json"}')
    return results


def placement_measures(y, own, nearest, indices, withheld, cuts):
    a = y[:, indices]
    out = {'count': len(indices), 'nearest_cosine_median': np.median(nearest[:, indices], axis=1)}
    for k in (8, 64):
        cut = cuts[str(k)]
        if np.any(withheld[:k]):
            out[str(k)] = dict(cuts_band=cut, unavailable='withheld_mode')
            continue
        v = a[:, :, :k].astype(np.float64, copy=False)
        denom = np.square(own[:, :, :k].astype(np.float64, copy=False)).mean(axis=(1, 2))
        numer = np.square(v).mean(axis=(1, 2))
        mean2 = np.square(v.mean(axis=1)).sum(axis=1) / k
        live = denom >= 1e-12
        rad = np.sqrt(numer / np.where(live, denom, np.nan))
        off = np.sqrt(mean2 / np.where(live, denom, np.nan))
        out[str(k)] = dict(cuts_band=cut, radius_ratio=rad, centre_offset=off,
            pooled_radius_ratio=float(np.sqrt(numer.mean() / denom.mean())) if denom.mean() >= 1e-12 else None,
            pooled_centre_offset=float(np.sqrt(np.square(off).mean())) if live.all() else None,
            own_mean_square=denom)
    return out


def place(model, rows, in_set, x, kernels, solutions, configs):
    n_site = x.shape[1]
    y = {s: np.empty((LAYERS, len(rows), KEEP), dtype=np.float32) for s in solutions}
    nearest = np.ones((LAYERS, len(rows)), dtype=np.float64)
    known = np.flatnonzero(in_set >= 0)
    other = np.flatnonzero(in_set < 0)
    for s, jm in solutions.items():
        y[s][:, known] = jm['vecs'][:, in_set[known]]
    if not len(other):
        return y, nearest
    rows_block = np.empty((LAYERS, min(BLOCK, len(other)), n_site))
    for first in range(0, len(other), BLOCK):
        ii = other[first:first + BLOCK]
        sel = grouped([rows[i] for i in ii])
        pr = rows_block[:, :len(ii)]
        for l in range(LAYERS):
            query = load_layer(model, sel, l, len(ii))
            pr[l], nearest[l, ii] = dm.radius_rows(query, x[l], kernels[l])
        for s, jm in solutions.items():
            y[s][:, ii] = extend(pr, jm, configs[s])
    return y, nearest


def refit_rows(manifest):
    return [r for r in manifest['panels']['natural'] if r[0] in range(8)]


def unit_plan(manifest, model, name, refit=False):
    own = manifest['sets'][name]
    if refit:
        return own + refit_rows(manifest), {}, {}, {}
    if any(r[2] is None or not act_path(model, r[0]).is_file() for r in own):
        return None
    def available(rows):
        return all(r[2] is not None and act_path(model, r[0]).is_file() for r in rows)
    panels = {k: r for k, r in manifest['panels'].items() if available(r)}
    paths = {k: r for k, r in manifest['paths'].items() if available(r)} if name in ('d32', 'rand') else {}
    missing = {'panels': sorted(set(manifest['panels']) - panels.keys()),
               'paths': sorted(set(manifest['paths']) - paths.keys()) if name in ('d32', 'rand') else []}
    return own, panels, paths, missing


def peak_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def watch_memory():
    while True:
        if peak_bytes() > STOP_LIMIT:
            print(f'ABORT: peak memory {peak_bytes()} exceeded {STOP_LIMIT} bytes', file=sys.stderr, flush=True)
            os._exit(75)
        time.sleep(.15)


def log(msg):
    print(msg, flush=True)


def provenance(manifest, model, name, plan, refit):
    own, panels, paths, missing = plan
    contexts = dict.fromkeys(row[0] for group in (own, *panels.values(), *paths.values()) for row in group)
    stats = {context_name(ctx): file_hash(act_path(model, ctx)) for ctx in contexts}
    needed_contexts = {context_name(ctx): manifest['contexts'][context_name(ctx)] for ctx in contexts}
    return dict(model=model, name=name, refit=refit, activations=stats, rows=own, panels=panels, paths=paths,
                cohorts={k: manifest['cohorts'][name][k] for k in panels}, missing=missing,
                context_metadata=needed_contexts, reference=manifest['references'][name], probe_sha256=file_hash(PROBE),
                synthetic_inputs=manifest['synthetic_inputs'] if any(isinstance(c, str) for c in contexts) else None)


def code_hash():
    return digest({p: file_hash(HERE / p) for p in ('depth.py', 'depth_dm.py', 'spacetime.py')})


def inventory(directory):
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in ('done.json', 'run.json')}


def unit_dir(out, model, name, refit=False):
    return out / ('refits' if refit else 'units') / model / name


def done_matches(dest, hashes):
    try:
        done = read_json(dest / 'done.json')
        return all(done[k] == v for k, v in hashes.items()) and done['outputs_sha256'] == digest(inventory(dest))
    except (OSError, ValueError, KeyError):
        return False


def build_unit(work, manifest, model, name, plan, refit):
    rows, panels, paths, missing = plan
    x, ms, kernels, site_stat, c, tstar, layer_vecs, site_vals = build_sites(model, rows)
    log(f'{model}/{name}{" refit" if refit else ""}: built {len(rows)} tokens; t*={tstar:.9g}')
    np.save(work / 'site_modes.npy', layer_vecs.astype(np.float32))
    del layer_vecs
    np.savez(work / 'sites.npz', c=c, eigenvalues=site_vals, radii=np.array([ker['rk'] for ker in kernels]),
             sinkhorn_scaling=np.array([ker['v'] for ker in kernels]))
    solutions, configs, facts = {}, {}, {}
    for s in (MODE[:1] if refit else MODE):
        jm = solutions[s] = solve(ms, c, tstar, s)
        val = jm['guard_eigenvalues']
        log(f'{model}/{name} {s}: {len(val)} modes, max residual {jm["residual"].max():.3g}')
        cfg = configs[s] = extension_config(jm)
        sdir = work / s
        sdir.mkdir()
        f32 = jm['vecs'].astype(np.float32)
        np.save(sdir / 'modes.npy', f32)
        if name in AUDITED and not refit:
            np.save(sdir / 'modes_f64.npy', jm['vecs'])
        # The refit is an off-release control; keep its original precision for the rotation.
        if refit:
            np.save(sdir / 'refit_modes.npy', jm['vecs'])
        facts[s] = dict(t=jm['t'], eigenvalues=jm['vals'], guard_eigenvalues=val,
                        gaps={str(k): val[k] - val[k - 1] for k in KS}, bands=jm['bands'],
                        cuts_band=jm['cuts_band'], residual=jm['residual'], max_residual=float(jm['residual'].max()),
                        placement={}, fields=embedding_measures(f32.astype(np.float64), rows))
        del f32
    replay_and_algebra(work, model, rows, x, ms, kernels, solutions, configs, name in AUDITED and not refit)
    del ms
    log(f'{model}/{name}: algebra and both replay block sizes passed')
    key_index = {row_key(r): i for i, r in enumerate(rows)}
    datasets = {}
    for collection, content in (('panels', panels), ('paths', paths)):
        for label, rr in content.items():
            idx = np.array([key_index.get(row_key(r), -1) for r in rr], dtype=np.int32)
            y, nearest = place(model, rr, idx, x, kernels, solutions, configs)
            (work / collection).mkdir(exist_ok=True)
            datasets[f'{collection}/{label}'] = dict(count=len(rr), in_set=int((idx >= 0).sum()))
            arrays = {**y, 'in_set_index': idx}
            if collection == 'panels':
                arrays['nearest_cosine'] = nearest
                cohort = np.asarray(manifest['cohorts'][name][label])
                for s in solutions:
                    modes = np.load(work / s / 'modes.npy')
                    groups = facts[s]['placement'][label] = {}
                    for ch in COHORTS:
                        pick = np.flatnonzero(cohort == ch)
                        if len(pick):
                            groups[ch] = placement_measures(y[s], modes, nearest, pick, configs[s]['withheld'],
                                                            solutions[s]['cuts_band'])
                    if label == 'shuffled':
                        seen = {r[0] for r in rows}
                        by_source = groups['unseen_sequence']['by_source_document'] = {}
                        for ctx, ii, _ in grouped(rr):
                            source = manifest['contexts'][ctx]['source_context']
                            by_source[ctx] = dict(source_context=source, source_document_in_set=source in seen,
                                measures=placement_measures(y[s], modes, nearest, ii, configs[s]['withheld'],
                                                           solutions[s]['cuts_band']))
            np.savez(work / collection / (label + '.npz'), **arrays)
            log(f'{model}/{name}: wrote {collection}/{label} ({len(rr)} rows)')
    for s in solutions:
        cfg = configs[s]
        facts[s]['extension'] = {k: v for k, v in cfg.items() if k != 'matrix'}
        write_json(work / s / 'facts.json', facts[s])
    write_json(work / 'unit.json', dict(model=model, set=name, refit=refit, n=len(rows), tstar=tstar, sites=site_stat,
               reference=manifest['references'][name], datasets=datasets, missing=missing))
    if refit:
        write_json(work / 'rows.json', rows)


def run_unit(out, model, name, *, refit=False, force=False):
    started = time.monotonic()
    threading.Thread(target=watch_memory, daemon=True).start()
    m = read_json(out / 'manifest.json')
    plan = unit_plan(m, model, name, refit)
    if plan is None:
        log(f'SKIP {model}/{name}: missing construction inputs or activations')
        return
    dest = unit_dir(out, model, name, refit)
    record = provenance(m, model, name, plan, refit)
    hashes = dict(inputs_sha256=digest(record), spec_sha256=file_hash(SPEC), code_sha256=code_hash())
    if not force and done_matches(dest, hashes):
        log(f'SKIP {dest}: all four hashes match')
        return
    missing = plan[-1]
    if any(missing.values()):
        log(f'SKIP unavailable placements for {model}/{name}: {json_bytes(missing).decode().strip()}')
    work = dest.with_name(dest.name + '.work')
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # Keep the manifest as the sole row table (refit adds its appended row table).
    write_json(work / 'input_files.json', {k: record[k] for k in ('activations', 'probe_sha256', 'synthetic_inputs')})
    build_unit(work, m, model, name, plan, refit)
    if code_hash() != hashes['code_sha256'] or file_hash(SPEC) != hashes['spec_sha256']:
        raise RuntimeError('code or specification changed during unit')
    files = inventory(work)
    write_json(work / 'done.json', {**hashes, 'outputs_sha256': digest(files), 'files': files})
    output_bytes = sum(p.stat().st_size for p in work.rglob('*') if p.is_file())
    telemetry = dict(model=model, set=name, refit=refit, peak_rss_bytes=peak_bytes(),
                     wall_seconds=time.monotonic() - started, output_bytes=output_bytes,
                     missing=missing, numpy=np.__version__, scipy=scipy.__version__)
    if telemetry['peak_rss_bytes'] > STOP_LIMIT:
        raise RuntimeError('peak memory crossed the unit stop threshold')
    write_json(work / 'run.json', telemetry)
    if dest.exists():
        shutil.rmtree(dest)
    work.replace(dest)
    log(json_bytes(telemetry).decode().strip())


def overlap_frame(a, k):
    """Per-layer frames plus a frame after stacking separately centred layers."""
    a = np.asarray(a[:, :, :k], dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    frames = []
    for x in (*a, a.reshape(-1, k)):
        u, sv, _ = np.linalg.svd(x, full_matrices=False)
        rank = int(np.count_nonzero(sv > 1e-6 * sv[0])) if sv.size else 0
        frames.append((u if rank == k else None, rank))
    return frames


def frame_scores(a, b, k, perm=None, cuts=False):
    result = []
    for (u, ra), (v, rb) in zip(a, b):
        value = None
        if ra == rb == k:
            if perm is not None:
                v = v[perm] if len(v) == len(perm) else v.reshape(-1, len(perm), k)[:, perm].reshape(-1, k)
            value = float(np.square(u.T @ v).sum() / k)
        result.append(dict(overlap=value, rank_a=ra, rank_b=rb, cuts_band=bool(cuts)))
    return dict(layers=result[:-1], stacked=result[-1])


def overlap(a, b, k, cuts=False):
    if a.shape[:2] != b.shape[:2] or a.shape[-1] < k or b.shape[-1] < k:
        raise ValueError('overlap requires the same rows and enough columns')
    if not np.all(np.isfinite(a[:, :, :k])) or not np.all(np.isfinite(b[:, :, :k])):
        return dict(unavailable='withheld_mode', cuts_band=bool(cuts))
    return frame_scores(overlap_frame(a, k), overlap_frame(b, k), k, cuts=cuts)


def npz_key(path, key):
    with np.load(path, allow_pickle=False) as f:
        return f[key]


class ReleaseUnit:
    def __init__(self, path, manifest):
        self.path = path
        self.meta = read_json(path / 'unit.json')
        self.rows = manifest['sets'][self.meta['set']]

    def modes(self, s):
        return np.load(self.path / s / 'modes.npy', mmap_mode='r')

    def facts(self, s):
        return read_json(self.path / s / 'facts.json')

    def panels(self, kind, s):
        file = self.path / 'panels' / f'{kind}.npz'
        return npz_key(file, s) if file.is_file() else None


def release(out, manifest, model, name):
    dest = unit_dir(out, model, name)
    try:
        done = read_json(dest / 'done.json')
        if done['code_sha256'] == code_hash() and done['spec_sha256'] == file_hash(SPEC):
            return ReleaseUnit(dest, manifest)
    except (OSError, ValueError, KeyError):
        pass
    return None


def filtered_radii(unit, name, strength, manifest, ii):
    file = unit.path / 'panels/natural.npz'
    v = unit.modes(strength)
    with np.load(file, allow_pickle=False) as panel:
        facts = unit.facts(strength)
        return placement_measures(panel[strength], v, panel['nearest_cosine'], ii,
                                  facts['extension']['withheld'], facts['cuts_band'])


def compare(a, b, fa, fb):
    return {str(k): overlap(a, b, k, cuts=fa['cuts_band'][str(k)] or fb['cuts_band'][str(k)]) for k in KS}


def cross_stage(out, m):
    records = []
    skipped = []
    ladder = ['doc00', 'pairA', 'd04', 'd08', 'd16', 'd32', 'rand']
    natural = np.array([i for i, r in enumerate(m['panels']['natural']) if r[0] != 0])
    units = {(model, s): release(out, m, model, s) for model in MODELS for s in SETS}
    for model in MODELS:
        for a_name, b_name in zip(ladder, ladder[1:]):
            a, b = units[model, a_name], units[model, b_name]
            if not a or not b:
                skipped.append(['ladder', model, a_name, b_name]); continue
            for s in MODE:
                panel_a, panel_b = a.panels('natural', s), b.panels('natural', s)
                if panel_a is None or panel_b is None:
                    skipped.append(['ladder_panel', model, a_name, b_name, s]); continue
                records.append(dict(kind='ladder', model=model, set_a=a_name, set_b=b_name, strength=s,
                    panel='natural_outside_doc00', count=len(natural),
                    scores=compare(panel_a[:, natural], panel_b[:, natural], a.facts(s), b.facts(s)),
                    sizes_a=filtered_radii(a, a_name, s, m, natural), sizes_b=filtered_radii(b, b_name, s, m, natural)))
    for name in SETS:
        a, b = (units[model, name] for model in MODELS)
        if not a or not b:
            skipped.append(['models', name]); continue
        for s in MODE:
            fa, fb = a.facts(s), b.facts(s)
            records.append(dict(kind='models', set=name, strength=s, panel='construction', cohort='in_set',
                                count=len(a.rows), scores=compare(a.modes(s), b.modes(s), fa, fb)))
            for panel, rr in m['panels'].items():
                aa, bb = a.panels(panel, s), b.panels(panel, s)
                if aa is None or bb is None:
                    skipped.append(['models_panel', name, s, panel]); continue
                cohorts = np.asarray(m['cohorts'][name][panel])
                for ch in COHORTS:
                    ii = np.flatnonzero(cohorts == ch)
                    if ii.size:
                        records.append(dict(kind='models', set=name, strength=s, panel=panel, cohort=ch, count=len(ii),
                                            scores=compare(aa[:, ii], bb[:, ii], fa, fb)))
                if panel == 'natural' and name in ('d32', 'rand'):
                    records.append(dict(kind='models_natural_comparison', set=name, strength=s,
                                        panel='natural_outside_doc00', count=len(natural),
                                        scores=compare(aa[:, natural], bb[:, natural], fa, fb)))
    write_json(out / 'cross.json', dict(records=records, skipped=skipped))
    log(f'cross: {len(records)} records, {len(skipped)} skipped; see {out / "cross.json"}')


def permutation_groups(ids, cohorts):
    groups = defaultdict(list)
    for i, key in enumerate(zip(ids, cohorts)):
        groups[key].append(i)
    return [np.asarray(ii, dtype=np.intp) for ii in groups.values() if len(ii) > 1]


def permutations(rng, n, groups):
    base = np.arange(n)
    within, free = [], []
    for _ in range(20):
        p = base.copy()
        for g in groups:
            p[g] = rng.permutation(g)
        within.append(p)
    for _ in range(20):
        free.append(rng.permutation(base))
    return dict(within_id_and_cohort=within, free=free)


def permutation_scores(a, b, perms, fa, fb):
    result = {}
    for k in KS:
        key = str(k)
        cuts = fa['cuts_band'][key] or fb['cuts_band'][key]
        if not np.all(np.isfinite(a[:, :, :k])) or not np.all(np.isfinite(b[:, :, :k])):
            result[key] = dict(unavailable='withheld_mode', cuts_band=cuts); continue
        af, bf = overlap_frame(a, k), overlap_frame(b, k)
        actual = frame_scores(af, bf, k, cuts=cuts)
        stat = result[key] = dict(matched=actual)
        for name, draws in perms.items():
            vals = np.full((len(draws), len(af)), np.nan)
            # Permuting rows changes neither centring nor singular values; reuse the SVD frames.
            for i, perm in enumerate(draws):
                score = frame_scores(af, bf, k, perm=perm, cuts=cuts)
                vals[i] = [v['overlap'] if v['overlap'] is not None else np.nan for v in (*score['layers'], score['stacked'])]
            stat[name] = dict(mean=vals.mean(axis=0), minimum=vals.min(axis=0), maximum=vals.max(axis=0), draws=vals)
    return result


def permutation_stage(out, m):
    records, skipped = [], []
    for name in SETS:
        a, b = (release(out, m, model, name) for model in MODELS)
        kinds = [('construction', 'in_set', np.arange(len(m['sets'][name])), m['sets'][name])]
        for kind, rr in m['panels'].items():
            cohorts = np.asarray(m['cohorts'][name][kind])
            kinds.extend((kind, ch, np.flatnonzero(cohorts == ch), rr) for ch in COHORTS if (cohorts == ch).any())
        if name in ('d32', 'rand'):
            rr = m['panels']['natural']
            kinds.append(('natural', 'outside_doc00', np.array([i for i, r in enumerate(rr) if r[0] != 0]), rr))
        for kind, ch, ii, rr in kinds:
            cohorts = [m['cohorts'][name][kind][i] if kind != 'construction' else 'in_set' for i in ii]
            groups = permutation_groups([rr[i][2] for i in ii], cohorts)
            # Each named row population has 20 draws from the declared seed;
            # the identical permutations are reused at the two strengths.
            perms = permutations(np.random.default_rng(20261129), len(ii), groups)
            movable = sum(len(g) for g in groups) / len(ii)
            if not a or not b:
                skipped.append([name, kind, ch]); continue
            for s in MODE:
                ac, bc = ((a.modes(s), b.modes(s)) if kind == 'construction' else
                          (a.panels(kind, s), b.panels(kind, s)))
                if ac is None or bc is None:
                    skipped.append([name, kind, ch, s]); continue
                result = permutation_scores(ac[:, ii], bc[:, ii], perms, a.facts(s), b.facts(s))
                records.append(dict(set=name, panel=kind, cohort=ch, strength=s, count=len(ii), movable_share=movable,
                                    within_id_uninformative=movable < .1, scores=result))
        log(f'permutation references: {name}')
    write_json(out / 'permutations.json', dict(seed=20261129, n_draws=20, rng='same initial seed per named row population',
                                               records=records, skipped=skipped))


def gaussian_site_modes(rows, member):
    # One fixed independent dictionary per member, consistently keyed by id across sets.
    unique, inverse = np.unique([r[2] for r in rows], return_inverse=True)
    gx = np.empty((len(unique), WIDTH))
    for j, token_id in enumerate(unique):
        gx[j] = np.random.default_rng([20261201, member, int(token_id)]).standard_normal(WIDTH)
    p = dm.radius_kernel(dm.unit_vectors(gx[inverse]))['P']
    vals, vecs = np.linalg.eigh(p)
    return np.ascontiguousarray(vecs[:, -2:-KEEP - 2:-1])[None]


def gaussian_stage(out, m):
    records, skipped = [], []
    for name, rr in m['sets'].items():
        a, b = (release(out, m, model, name) for model in MODELS)
        if not a or not b:
            skipped.append(name); continue
        a, b = (np.load(u.path / 'site_modes.npy', mmap_mode='r') for u in (a, b))
        g1, g2 = (gaussian_site_modes(rr, j) for j in range(2))
        scores = {}
        for k in KS:
            scores[str(k)] = dict(gaussian=overlap(g1, g2, k)['layers'][0], real=overlap(a, b, k)['layers'])
        records.append(dict(set=name, count=len(rr), scores=scores))
        log(f'Gaussian per-layer reference: {name}')
    write_json(out / 'gaussian.json', dict(seed=20261201, draws=1, records=records, skipped=skipped,
                                           note='The fixed id-only dictionaries yield the same walk at all 17 layers.'))


def refit_score(a, b, held, b_held):
    u, _, vt = np.linalg.svd(a.T @ b)
    rotate = u @ vt
    def score(v, w):
        if not np.all(np.isfinite(v)):
            return None
        return float(1 - np.square(v @ rotate - w).sum() / np.square(w).sum())
    return dict(shared=score(a, b), placed=score(held, b_held))


def refit_stage(out, m):
    results, skipped = {}, []
    ii = np.array([i for i, r in enumerate(m['panels']['natural']) if r[0] in range(8)])
    for model in MODELS:
        base = release(out, m, model, 'd08')
        refit = unit_dir(out, model, 'd08', True) / MODE[0]
        if base is None or not (refit / 'refit_modes.npy').is_file():
            skipped.append(model); continue
        b = np.load(refit / 'refit_modes.npy', mmap_mode='r')
        a, held = base.modes(MODE[0]), base.panels('natural', MODE[0])
        if held is None:
            skipped.append(model); continue
        fa, fb = base.facts(MODE[0]), read_json(refit / 'facts.json')
        results[model] = {}
        for k in KS:
            reshape = lambda x: np.array(x[:, :, :k], dtype=np.float64).reshape(-1, k)
            key = str(k)
            results[model][key] = dict(**refit_score(reshape(a), reshape(b[:, :len(base.rows)]),
                reshape(held[:, ii]), reshape(b[:, len(base.rows):])), cuts_band=fa['cuts_band'][key] or fb['cuts_band'][key])
    write_json(out / 'refit_agreement.json', dict(records=results, skipped=skipped))


def child(out, task, model=None, name=None, *, force=False):
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), '--out', str(out), task]
    if model is not None:
        argv.extend(('--model', model, '--set', name))
    if force:
        argv.append('--force')
    subprocess.run(argv, check=True)


def gates(out, force):
    cases = [('pythia', 'd32', False), ('olmo', 'd08', True)]
    results = []
    for model, name, refit in cases:
        child(out, '_refit' if refit else '_unit', model, name, force=force)
        path = unit_dir(out, model, name, refit)
        run = read_json(path / 'run.json')
        results.append(dict(**run, complete_placement=not any(run['missing'].values()),
                            checks=read_json(path / 'checks.json'), under_memory_limit=run['peak_rss_bytes'] < GATE_LIMIT))
    write_json(out / 'gate.json', dict(cases=results, limit_bytes=GATE_LIMIT))
    if not all(r['under_memory_limit'] for r in results):
        raise RuntimeError('local gate failed the 4.5 GB peak memory requirement')
    if not results[0]['complete_placement']:
        log('INCOMPLETE placement gate: synthetic panels or paths unavailable; document-only units can still run')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('manifest', 'unit', 'gate', 'cross', 'controls', 'all',
                                      '_unit', '_refit', '_cross', '_permutations', '_gaussian', '_refit_scores'))
    p.add_argument('--out', type=Path, default=Path(os.environ.get('TM_DEPTH_OUT', '/data/tm_depth')))
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--set', dest='name', choices=SETS)
    p.add_argument('--force', action='store_true', help='rebuild selected unit or gate even if hashes match')
    return p


def main():
    args = parser().parse_args()
    if args.command in ('unit', '_unit', '_refit') and (args.model is None or args.name is None):
        parser().error('unit requires --model and --set')
    if args.command == '_refit' and args.name != 'd08':
        parser().error('refit is defined only for d08')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    if priority < 5:
        os.nice(5 - priority)
    task = args.command
    if task in ('_unit', '_refit'):
        return run_unit(out, args.model, args.name, refit=task == '_refit', force=args.force)
    stages = {'_cross': cross_stage, '_permutations': permutation_stage, '_gaussian': gaussian_stage,
              '_refit_scores': refit_stage}
    if task in stages:
        threading.Thread(target=watch_memory, daemon=True).start()
        return stages[task](out, read_json(out / 'manifest.json'))
    with (out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        m = manifest_now(out)
        if task == 'manifest':
            log(f'{out / "manifest.json"}: {len(m["sets"])} sets; synthetic ids {"present" if m["synthetic_inputs"] else "unavailable"}')
        elif task == 'unit':
            child(out, '_unit', args.model, args.name, force=args.force)
        elif task == 'gate':
            gates(out, args.force)
        else:
            if task == 'all':
                gates(out, False)
                for model in MODELS:
                    for name in SETS:
                        child(out, '_unit', model, name)
            if task in ('all', 'cross'):
                child(out, '_cross')
            if task in ('all', 'controls'):
                for model in MODELS:
                    child(out, '_refit', model, 'd08')
                for stage in ('_refit_scores', '_permutations', '_gaussian'):
                    child(out, stage)


if __name__ == '__main__':
    main()
