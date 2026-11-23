"""§9.3 planted and noise controls. No external data or models are needed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from census import PRIORITY, Pool, atomic, input_name, random_vectors, run_group

VARIANTS = ("strong", "planted", "drift", "noise_a", "noise_b")


@dataclass
class Synthetic:
    seed: int
    variant: str
    tokens: np.ndarray
    regime: np.ndarray
    states: np.ndarray  # (site, text, position, feature)
    embeddings: np.ndarray  # (token, feature); row indexed by token ID
    predictors: np.ndarray  # per-token regime-dependent pseudo forecasts (fixture use)
    alpha: np.ndarray
    beta: np.ndarray
    gamma: float


def generate(seed=0, variant="planted", *, texts=8, positions=1024, width=64, sites=4, vocab=256,
             switch_probability=1 / 128, zipf_exponent=1.1, beta=.5, noise_rms=.25):
    """Draw one control field, sharing token/regime paths across variants.

    Positions are 1..positions; site 0 has no regime loading. Unspecified
    parameters: unit random template directions and a linear unit-direction
    drift from -1 to 1; truncated Zipf; a switch chooses a different state
    uniformly. Drift adds to planted; the two noise variants are pure noise.
    The extra predictor field is only for optional integration fixtures.
    """
    if (variant not in VARIANTS or min(texts, positions, width, sites, vocab) < 1
            or not 0 <= switch_probability <= 1 or noise_rms <= 0):
        raise ValueError("invalid synthetic parameters")
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    graph, lexical, basis, typical, noise_a, noise_b = streams
    regime = np.empty((texts, positions), dtype=np.int8)
    regime[:, 0] = graph.integers(4, size=texts)
    switch = graph.random((texts, positions - 1)) < switch_probability
    step = graph.integers(1, 4, size=(texts, positions - 1)) * switch
    regime[:, 1:] = (regime[:, :1] + step.cumsum(1)) % 4
    weight = np.arange(1, vocab + 1, dtype=float) ** (-zipf_exponent)
    tokens = lexical.choice(vocab, size=(texts, positions), p=weight / weight.sum())

    def directions(n):
        rows = basis.standard_normal((n, width))
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    r, e, g = directions(4), directions(vocab), directions(1)[0]
    alpha = np.full(sites, 1. if variant == "strong" else .25)
    alpha[0] = 0.
    beta = np.full(sites, beta, dtype=float)
    gamma = .5 if variant == "drift" else 0.
    draw = {"noise_a": noise_a, "noise_b": noise_b}.get(variant, typical)
    out = draw.standard_normal((sites, texts, positions, width)) * (noise_rms / np.sqrt(width))
    if variant.startswith("noise_"):
        alpha[:] = beta[:] = 0.
    else:
        out += alpha[:, None, None, None] * r[regime]
        out += beta[:, None, None, None] * e[tokens]
        out += gamma * np.linspace(-1., 1., positions)[None, None, :, None] * g
    # Toy predictions for an end-to-end fixture, not reported on the site.
    predictors = (tokens + regime) % vocab
    return Synthetic(seed, variant, tokens, regime, out, e, predictors, alpha, beta, gamma)


def run_synthetic(root, *, seeds=range(5), variants=VARIANTS, groups=PRIORITY, device="cpu", row_block=128,
                  start=256, generator_options=None, persist_lists=False):
    """Run §5A on reusable synthetic geometry; save latent labels for Stage B."""
    root = Path(root)
    groups = tuple(groups)
    seeds, variants = tuple(seeds), tuple(variants)
    if tuple(g for g in PRIORITY if g in groups) != groups:
        raise ValueError("groups must follow cosine C/O, dot C/O priority")
    results = []
    for group_idx, (reading, centring) in enumerate(groups):
        for seed in seeds:
            random_by_shape = None
            for variant in variants:
                data = generate(seed, variant, **(generator_options or {}))
                name = f"synthetic_{variant}_seed{seed}"
                texts, length = data.tokens.shape
                pool = Pool(data.tokens, tuple(range(texts)), start, max(start, length - 511))
                if random_by_shape is None:
                    random_by_shape = random_vectors(data.tokens, width=data.states.shape[-1])
                inputs = {"random": random_by_shape, "embedding": data.embeddings[data.tokens]}
                inputs.update({input_name(s): v for s, v in enumerate(data.states)})
                if group_idx == 0:
                    atomic(root / "analysis" / "synthetic" / f"{name}.npz", np.savez,
                           regime=data.regime, token=data.tokens, alpha=data.alpha, beta=data.beta, gamma=data.gamma,
                           variant=variant, seed=seed, generator_options=str(generator_options or {}))
                results.append(run_group(root, name, pool, inputs, reading, centring, device=device, row_block=row_block,
                                         screen=False, pair_count=0, validate_models=False, persist_lists=persist_lists))
    return results
