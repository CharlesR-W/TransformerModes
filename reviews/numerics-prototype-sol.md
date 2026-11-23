Implemented the numerical core and wrote the measurements to [bench_numerics.json](/home/ubuntu/code/TransformerModes/exp1/bench_numerics.json). **All 23 tests passed in 47.8 seconds**, including the unchanged `test_spacetime.py` suite. Nothing was committed or pushed.

The main finding is that **the sparse solver reproduces the dense solver on the same graph; neighbour truncation does not reproduce the dense graph.** Even at \(K_{\rm nb}=1024\), discarding only 3.25% of the conditional mass on average shifted \(t^*\) by 15% and produced a 16-mode subspace dissimilarity of 0.187 with the default floor.

## Numerical agreement

- **Memory kernel.** The explicit history feature map agreed with the dense recurrence within \(1.43\times10^{-13}\) on raw \(K_a\) and \(1.71\times10^{-13}\) on distances across the small cosine and dot tests. At \(N=1792\), the streamed cosine kernel returned exactly the same top-256 neighbour sets as the dense reference, with maximum kept-entry errors of \(7.11\times10^{-14}\) on raw \(K_a\) and \(1.44\times10^{-15}\) on distances.
- **Graph rule.** With all neighbours and no floor, the maximum conditional-weight error was \(5.07\times10^{-16}\) for participation ratio and \(9.30\times10^{-16}\) for Shannon perplexity. The maximum lazy-walk entry errors against `ea_kernel` were \(5.93\times10^{-15}\) and \(3.13\times10^{-14}\), respectively. Union-kNN matched its dense oracle within \(1.05\times10^{-14}\). At \(N=448,\ K_{\rm nb}=256,\ \eta=10^{-6}\), the sparse walk matched an independently assembled **truncated, floored** dense walk within \(5.28\times10^{-15}\); against the **untruncated, unfloored** walk, the maximum entry drift was \(7.33\times10^{-3}\) and the maximum row L1 drift was 0.171.
- **Through-depth solve.** These comparisons use identical walks in the sparse and dense solvers. The sparse result contains 24 float64 modes of shape \((L,N,24)\).

| \(N\) | \(L\) | Dense \(t^*\) | Relative \(t^*\) error | Max relative eigenvalue error, 24 modes | Largest \(\sin\theta\), first 16 | Max unprojected residual, sparse / dense |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 4 | 0.578354117 | −5.99e−15 | 2.49e−14 | 4.99e−14 | 1.81e−14 / 2.15e−13 |
| 1,792 | 8 | 2.84696835 | +3.33e−15 | 1.60e−14 | 3.93e−14 | 1.80e−13 / 1.34e−13 |

The solve checks unprojected residuals, balance at \(t^*\), the scalar lower bound, energy accounting, occupancy and recurrent-strength totals, finiteness, orthogonality, and slice means. Paper-transition checks and monotonicity across a coupling grid do not apply to this single-coupling solve.

## Truncation at \(N=1792,\ L=8\)

The dense, untruncated PR64 reference had no floor, \(t^*=5.3575711\), and a gap of 0.03792 between modes 16 and 17. Lost mass is measured on the **original dense conditional rows**, before refitting on the retained support or applying the floor. The eigenvalue column is the maximum absolute relative drift across the first 16; the JSON contains all 16 individual drifts.

| \(K_{\rm nb}\) | \(\eta\) | Mean lost mass | P95 lost mass | Relative \(t^*\) drift | Max eigenvalue drift | \(1-\operatorname{mean}\cos^2\theta\) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0 | 11.16% | 15.68% | −75.37% | 75.37% | 0.5671 |
| 128 | 1e−6 | 11.16% | 15.68% | −67.35% | 67.35% | 0.5289 |
| 256 | 0 | 9.32% | 13.20% | −52.88% | 52.88% | 0.4400 |
| 256 | 1e−6 | 9.32% | 13.20% | −50.04% | 50.04% | 0.3803 |
| 512 | 0 | 6.79% | 9.73% | −32.06% | 32.06% | 0.3154 |
| 512 | 1e−6 | 6.79% | 9.73% | −30.77% | 30.77% | 0.3153 |
| 1,024 | 0 | 3.25% | 4.76% | −15.65% | 15.65% | 0.1882 |
| 1,024 | 1e−6 | 3.25% | 4.76% | −15.00% | 15.00% | 0.1873 |

**Recommendation:** use \(K_{\rm nb}=1024\) as a starting point for a sensitivity sweep on short real texts, not as a validated production cutoff. Do not choose 256 on the strength of its speed or mean lost mass. None of the tested widths reproduced the dense low space on this synthetic problem. Truncation removes edges and refits each bandwidth; symmetrisation, Sinkhorn balancing, and site rescaling then change the operator further. A small mass error is not a spectral guarantee.

## CPU benchmark

Width 64, planted cluster runs and slow drift, \(L=8\), \(a=63/64\), one CPU thread per worker. Each cell is **seconds / peak process RSS in MiB**. Times sum over all sites where applicable. RSS includes the runtime and any walks already resident; the columns are not additive.

| \(N\) | \(K_{\rm nb}\) | Kernel | Graph | Site spectra + \(t^*\) | 24-mode solve |
|---:|---:|---:|---:|---:|---:|
| 1,792 | 256 | 1.27 / 338 | 2.73 / 348 | 6.34 / 369 | 2.60 / 371 |
| 1,792 | 1,024 | 1.95 / 502 | 10.45 / 535 | 21.03 / 555 | 10.46 / 545 |
| 7,168 | 256 | 10.62 / 622 | 11.28 / 637 | 34.77 / 658 | 18.65 / 668 |
| 7,168 | 1,024 | 13.70 / 1,076 | 44.28 / 1,311 | 103.50 / 1,053 | 50.14 / 1,106 |
| 14,336 | 256 | — | — | — | Timed out with 191 s left in the run budget |
| 14,336 | 1,024 | — | — | — | Not started; time budget exhausted |

On the **same** \(N=1792,\ K_{\rm nb}=256\) walks, the dense oracle took 15.46 s for calibration and 9.36 s for the solve, plus 1.00 s to materialise the walks. Sparse calibration plus solve took **8.94 s versus 24.82 s**, a **2.8× speedup** excluding materialisation. Their sampled peak RSS was 371 versus 1,003 MiB. The separate dense **untruncated** reference took 1.41 s for the kernel, 89.67 s for graph construction, 18.42 s for calibration, and 9.62 s for the solve.

The completed pass took **870.1 s (14 min 30 s)**. The largest completed worker high-water mark was 1,316 MiB. Both 14,336-node rows were dropped for **time**, not for an observed memory failure. An earlier pass also hit the time budget, and a bookkeeping bug overwrote its completed truncation results; that bug was fixed and the reported pass reran the sweep.

## Full-size projection and conventions

For \(N=57{,}344,\ L=8\), the two time estimates below are linear scaling from \(N=7168\) and a power law fitted to the two measured sizes. They are **not confidence intervals**. RSS is projected as a fixed runtime baseline plus a term linear in \(N\).

| \(K_{\rm nb}\) | Graph: s / GiB | Site spectra + \(t^*\): s / GiB | 24-mode solve: s / GiB | Resident walks alone |
|---:|---:|---:|---:|---:|
| 256 | 90–95 / 3.37 | 278–446 / 3.53 | 149–359 / 3.62 | 1.48 GiB |
| 1,024 | 354–386 / 8.63 | 828–1,130 / 6.62 | 401–526 / 7.03 | 5.76 GiB |

The graph estimate is more credible than the solve estimate: at fixed retained degree, CSR storage and each sparse matvec scale roughly with \(LNK_{\rm nb}\), but Sinkhorn and Lanczos iteration counts, reciprocal-edge rates, and memory bandwidth can change. The projections assume no swapping and do not budget for keeping width-2048 activations resident on CPU. At 256, memory is already marginal on the shared VPS; at 1,024, **the walks alone exceed the 4 GiB budget**.

For the GPU kernel at \(M=32,\ T=2047,\ d=2048\), the streamed Gram products still evaluate all \(65{,}504^2\) pairs: \(2(32\cdot2047)^2\cdot2048=\mathbf{17.575}\) TFLOP per site, about **141 TFLOP across eight sites**. Recurrence and retained cosine-distance arithmetic add about 8.57 and 9.87 GFLOP per site per alpha. This is an operation count, not a GPU timing. Top-k selection, data movement, and the current Python-driven row recurrence can add substantial overhead; CUDA was not tested.

- **Floor:** with symmetric affinity \(A\), use \(W=(1-\eta)A+\eta(\mathbf1\mathbf1^\top-I)\) before balancing. This preserves unit kNN edges and applies the same off-diagonal floor to the adaptive rules. The diagonal correction stays in CSR; the floor becomes one rank-one term after scaling. It is not dropped by slice centring. At full size the default floor adds **0.0573 unscaled mass per row**, so whether to retain a fixed per-pair floor or a fixed total floor is a modelling decision.
- **History and ties:** burn-in is one-based and inclusive; earlier tokens still contribute history. Returned similarities are raw \(K_a\); cosine-normalised similarities are `1 - distance`. Self-neighbours are excluded. Exact cutoff ties have no guaranteed ordering. If tied minima make the requested effective count unattainable, the graph builder raises rather than silently changing the target.
- **Limits:** zero-floor graphs can be disconnected or fail Sinkhorn; calibration rejects disconnected or numerically indistinguishable-from-disconnected sites. Individual modes need not agree inside a degenerate band, and a cutoff that splits such a band does not define a unique comparison subspace. The remaining open decisions are the acceptable spectral drift on real text, the resulting neighbour budget, and the floor’s scaling with \(N\).

Core files: [similarity.py](/home/ubuntu/code/TransformerModes/exp1/similarity.py), [graph.py](/home/ubuntu/code/TransformerModes/exp1/graph.py), [solve.py](/home/ubuntu/code/TransformerModes/exp1/solve.py). Validation: [kernels.py](/home/ubuntu/code/TransformerModes/exp1/kernels.py), [test_numerics.py](/home/ubuntu/code/TransformerModes/exp1/test_numerics.py), [bench_numerics.py](/home/ubuntu/code/TransformerModes/exp1/bench_numerics.py). The dense files and `SPEC.md` were not changed by this work.
