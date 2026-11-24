# TransformerModes, experiment 1, Stage B: independent chunks, through depth, and the next-token prediction

Spec B v2, 2026-11-24. History: v1 (design); v2 (review 1, `reviews/spec-b-v1-astra.md`, 11 findings; responses in
section 18). Replaces the provisional Stage B of `SPEC.md` (sections 6-8, 16). Inherits unchanged from `SPEC.md`:
models, revisions, tokenizers, separators and forward-pass settings (section 2), the corpus file (section 3) and the
site machinery (section 12). Exploratory. Sources: the owner's direction (2026-11-24), Stage A's results,
`exp1/spacetime.py` and `exp1/kernels.py` (DepthEmbedding's dense operator and graph rule, the oracle, unchanged).

**The question.** Does matching each chunk to its own next-token prediction change the through-depth picture more
than attaching the same prediction geometry to the wrong chunks, and how much of the result is already visible from
the cut token or from the prediction alone?

**What it does not claim.** That anything found is a minimal predictive state or a Hankel rank; where in the model a
prediction is computed; that the structure is used by the model (no outcome is held out for that).

**In one paragraph.** We cut each of 2,048 Pile test documents once at a random position between tokens 256 and 1,023,
let Pythia-1b and OLMo-1B read the real document up to the cut, and keep the last 64 tokens' hidden states at 8 layers
(the **chunk**) and the model's next-token distribution at the cut. No two chunks share a document, so Stage A's chains
along a text cannot occur. Chunks are compared by three summaries of their states (the state at the cut, the primary
one; the recent past; the chunk average) and by how alike their next-token distributions are. The **product walk**
(a step in the layer's geometry, a step in the prediction geometry, a step back) charges a pattern for prediction
roughness after layer smoothing. Its appearance alone is never evidence: with a layer that prefers no pair of chunks
the product reproduces the prediction's own patterns exactly (P10). Evidence comes only from quantities that depend on
the correct pairing, compared with the same prediction geometry attached to wrong chunks (random relabellings, and
relabellings that keep the cut token). A planted hidden-Markov source is the gate before spend. Projected about $5.60,
proposed Stage B cap $8, about four hours.

## 0. What Stage A taught

Stage A (site E1-E3) listed each token's 64 nearest tokens in 32 long texts. With memory, neighbours move into the
token's own text and next to it (Pythia, layer 8: own-text share 59% with no memory, 98% at memory 64), a property of
the residual stream rather than the kernel (random token vectors: 9% within 64 positions at memory 64, against 51%).
Prediction concordance with matched controls falls with memory (0.133, 0.063, 0.017, 0.004 for none, 8, 64, whole
prefix). Participation-ratio rows were flat (whole prefix: half a row's walk weight in other texts, 0.6% of its 64
nearest). The owner judged the window and the own-text mean to be fixes for the sampling and asked for independent
chunks and for a construction using the model's prediction. Stage B therefore uses one chunk per document, nothing
masked, no per-document or position-based correction, fixed-length chunks, and the prediction inside the operator
with exact controls. **Single long texts** stay with Stage A's token viewer (Stage B links to it): on one text,
neighbouring positions share both pasts and predictions, which would bring the chain back into both views.

## 1. Questions and prespecified tests

| # | Question | Statistic and reference | Status |
|---|---|---|---|
| Q1 | At each layer, does pairing chunks with their own predictions keep more smoothness than wrong pairings? | n_l: eigenvalues of the product walk exceeding the largest leading eigenvalue among 200 wrong pairings (section 8.1), random and cut-token-preserving | per cell calibrated (1 in 201 under the null); depth shape and comparison differences exploratory |
| Q2a | Through depth, is agreeing with the prediction cheaper under the correct pairing? | energy cost E (section 8.2) against 50 random and 50 cut-token-preserving wrong pairings, state at the cut, each model | prespecified: 4 tests |
| Q2b | Does correct pairing raise the two models' agreement more than the same wrong pairing applied to both? | increment Δ (section 8.3) against the same 50 + 50 wrong pairings; prediction-only agreement beside it | prespecified: 2 tests |
| Q3 | Does the construction recover a planted belief where it is present, within the current symbol, and nothing where it is absent, symbol-only or scrambled? | section 9, frozen conditions | gate before the main spend |

Everything else (where in depth the count peaks, whether the recent past or average changes it, what modes look like,
attribute shares) is **exploratory**, shown with two half-sample curves for the primary comparison, and called
**indeterminate** when the halves disagree in sign or a count is censored at 64.

## 2. Probe and extraction

**Corpus.** As Stage A (`monology/pile-uncopyrighted`, revision `3be90335…`, `test.jsonl.zst`; subset =
`meta.pile_set_name`). Stage A's 36 documents are excluded by SHA-256 of the text.

**Eligibility** (failures counted by reason): no forbidden string; at least 95% ASCII; at least 1,024 Pythia tokens;
the token assertion of `SPEC.md` section 2 on the full text (`probe.assert_token_identity`); full-text SHA-256 not
already accepted.

**Cut and chunk.** Positions count the separator as 0. Eligible documents are shuffled with `default_rng(1)`; each in
turn gets a cut u uniform on {256, ..., 1,023} from `default_rng(2)` (one draw per document examined). The chunk is
positions u-63 to u. Reject if more than half of the chunk's tokens are whitespace-only, or if the chunk shares a
32-token id sequence with an accepted chunk. The first **2,048** accepted are the probe, i = 0..2,047 in accepted
order; **halves** are even and odd i. The cut range does not depend on length or subset before rejection; the
rejections depend on the cut, so the accepted cut distribution is reported by subset and by length decile. Saved per
chunk: source line index, text hash, length, subset, u, token ids of positions u-95..u+1.

**Extraction (a narrow new mode; `extract.py`'s paths return all positions or no probabilities).** Each model reads
`[its separator] + tokens 1..u` in one forward pass of the full causal model, one document per batch, `SPEC.md` section
2 settings (float32, eval, eager attention, TF32 off, pinned versions). Hooks keep only block outputs at sites
{2, 4, ..., 16} for positions u-63..u, sliced on the device. The **native** logits at the last position are read from
the model's own output and converted to a full softmax in float64. Hook equality with `output_hidden_states` is checked
on the first document. **Check (fails closed, every chunk):** the exported final norm and head (numpy, float64)
applied to the stored pre-norm layer-16 cut state reproduce the native softmax to 1e-6 absolute.

| Stored per model | Shape, type | Size |
|---|---|---|
| chunk states | (2,048, 8, 64, 2,048) float32; lag 0 = the cut | 8.6 GB |
| next-token distribution at the cut | (2,048, 50,304) float64 | 0.82 GB |
| final norm parameters, output head, input embedding | numpy | 0.82 GB |
| per chunk: top-20 ids and probabilities, entropy; provenance | small | |

## 3. Three summaries and the prediction geometry

x_{i,l}(s) is chunk i's state at site l and lag s (s = 0 at the cut, 63 at the chunk's start); layer 0 is the input
embedding, recomputed from token ids. **Centring:** one global mean μ_l per model and site, over every chunk and lag
of the sample analysed, as in Stage A's cosine construction. The owner's objection was to per-document and
position-based fixes; none is used. Each token vector is then scaled to unit length: y = (x - μ)/||x - μ|| (the cosine
reading; the dot reading is deferred). Every summary is a weighted Gram matrix over lags:

K_W(i, j) = Σ_{s, s'} W_{ss'} ⟨y_i(s), y_j(s')⟩,  distance d = 1 - K_W(i, j)/sqrt(K_W(i, i) K_W(j, j))

| Summary | W | In words |
|---|---|---|
| **state at the cut** (primary) | W_00 = 1 | the contextualised state where the model predicts |
| recent past, memory 8 | W_ss = a^s, a = 7/8 | the owner's end-aligned kernel K_a over the last tokens, lag by lag; truncation at 64 lags drops a^64 = 1.9 × 10^-4 of the weight |
| chunk average | W_ss' = 1/64² | the mean of the 64 contextualised states |

These are three summaries of states that each already depend on the whole prefix. They are not an ablation of order or
a test of extra memory: the average pools contextualised states, it does not shuffle tokens and rerun the model.
Checks (fail closed): no ||x - μ|| below 1e-6 × median; the chunk average's pooled norm sqrt(K_W(i, i)) not below
1e-3 × its median; no off-diagonal distance below 1e-9 except in the layer-0 state at the cut, where chunks with the
same cut token are identical by construction and handled by the tie rule (section 4).

**The prediction geometry.** p_i is the model's next-token distribution at chunk i's cut: a **one-step predictive
distribution**, the empty-continuation block of a row of Golowich, Liu and Shetty's extended logit matrix. Since
sqrt(p_i) has unit length, ⟨sqrt(p_i), sqrt(p_j)⟩ = Σ_v sqrt(p_i(v) p_j(v)) is the Bhattacharyya coefficient and d_F =
1 - BC = Σ_v (sqrt(p_i(v)) - sqrt(p_j(v)))²/2 is the squared Hellinger distance, computed from the squared differences
(no cancellation). Square roots rather than logits: the distance is fixed by the distribution, bounded, and weighted
toward probable tokens. Not centred. Near-identical predictions are legitimate and handled by the tie rule.

## 4. Graph rule (one, fixed)

**Participation ratio 64**, dense, from `kernels.py`: q(j|i) ∝ exp(-β_i (d_ij - min_j d_ij)) with 1/Σ_j q(j|i)² = 64;
W = (q + q^T)/2; symmetric Sinkhorn to 1e-11; S = the resulting symmetric doubly stochastic walk (zero diagonal);
A = (I + S)/2. Dense and floor-free: every pair has positive weight, so there are no floor-defined components.

**Tie rule (permutation-equivariant).** Equal distances get equal weight because q depends only on distance. The tied
minimum set of row i is T_i = {j : d_ij - min_j d_ij ≤ 1e-12}. If |T_i| ≥ 64 the target is unattainable and the row is
the β → ∞ limit, uniform on T_i; such rows are counted and reported, never treated as failures. Identical predictions
for every chunk therefore give the complete graph, the case of P5.

**Reported for every walk:** conditional tail τ_q (mean mass outside each row's 64 largest q), final-walk tail τ_S
(the same for rows of S), achieved participation ratios, Sinkhorn error, the first 12 nontrivial eigenvalues and the
gap after the 16th. A first nontrivial gap below 1e-3 is flagged as a near-disconnection, with the sizes and subsets
of the two sides of its sign split. kNN is not run (section 16).

## 5. The product walk

S_l is layer l's walk for the summary in use, A_l = (I + S_l)/2, S_F the prediction walk (not lazy). Per layer:

- **past only** (the paired partner): M_l = A_l²;
- **with the prediction:** M_l = A_l S_F A_l;
- **wrong pairing** Π (a relabelling of chunks): M_l^Π = A_l Π S_F Π^T A_l, the same prediction geometry on the wrong
  chunks; one Π is applied at every site, in every summary and in both models (common relabellings).

By P2 the product charges a function for its roughness on the layer (two lazy steps) plus the prediction graph's
roughness of the layer-smoothed function: **prediction roughness after layer smoothing**. It is not an intersection
projector, and its modes are not "what the layer knows about the future".

| # | Statement | Status |
|---|---|---|
| P1 | M = A S_F A is symmetric, entrywise nonnegative, doubly stochastic, -I ≤ M ≤ I; I - M is positive semidefinite, which is all `spacetime.py` needs | theorem |
| P2 | f^T (I - M) f = f^T (I - A²) f + (Af)^T (I - S_F)(Af); hence M ≤ A² | theorem |
| P3 | λ_k(M) ≤ λ_k(A²) for all k; λ_k(M) ≤ λ_k(S_F) where λ_k(S_F) ≥ 0 (Ostrowski) | theorem |
| P4 | M has eigenvalue 1 on mean-zero functions only if both graphs share a disconnection | theorem |
| P5 | S_F = I gives A². S_F the complete graph gives M = -A²/(N - 1) on mean-zero functions. If S_F commutes with A, M has their eigenvectors with eigenvalues ((1 + μ)/2)² ν | theorem |
| P6 | With A and S_F replaced by orthogonal projections onto centred feature spans (plus the constant), the nontrivial eigenvalues of M are the squared sample canonical correlations of the two feature sets | theorem |
| P7 | The output is a function of the prefix, so the one-step prediction is the same at every layer; n_l describes the layer's geometry at this graph resolution, not where the prediction is computed. Layer 16 is an endpoint reference, not a ceiling: the norm-head-softmax map need not preserve neighbourhoods | theorem (first clause); reading |
| P8 | Eigenvectors of M_l whose eigenvalues exceed the wrong-pairing null are smooth functions of a variable common to both views; n_l counts resolvable shared directions (harmonics included), not a dimension | hypothesis (alternating diffusion: Lederman and Talmon 2018; Talmon and Wu 2019, under conditional independence of the views given the common variable, unverified here); tested on the known answer |
| P9 | At equal site units c_l and coupling t, L_with - L_past = blockdiag(A_l (I - S_F) A_l / c_l) ⪰ 0, so every through-depth eigenvalue rises (Weyl), and the energy cost E ≥ 0 | theorem |
| P10 | If every A_l commutes with Π (a layer that prefers no pair of chunks, A_l = αI + βJ; or a layer that is a function of the cut token, with Π keeping the cut token), the wrong-pairing product has exactly the site spectra, through-depth spectrum and (relabelled) modes of the correct one: n_l = 0 and E, Δ show no effect. With a flat layer the product's modes are the prediction walk's eigenvectors whatever the layer represents | theorem |

P1-P6 were checked numerically on random walks at N = 60 (scratch, to 1e-15); all ten are tested in section 10.

**Motivation, conditional.** For a stationary process, canonical correlations between long enough past and future
windows give, under minimality and full-rank conditions on the population past-future Hankel operator, the state of a
minimal linear realisation (Akaike 1975); Golowich, Liu and Shetty find extended logit matrices of language models
approximately low rank. That is why a past-against-future comparison is natural. Our view is one-step, sampled and
nonlinear, so no state or rank claim is made.

**Degenerate cases.** Prediction walk nearly uninformative: M ≈ 0 on mean-zero functions, n_l = 0. Prediction
geometry dominated by frequent next tokens: shown by the Hellinger token decomposition (section 8.4). Layer flat: P10.

**Not taken.** A lazy prediction walk: it adds an explicit A²/2 past-only term to M. The prediction walk squared:
Sinkhorn walks with zero diagonal have large negative eigenvalues (-0.58 at N = 60), which squaring makes look smooth.
The prediction as a ninth site at the end of the chain: no per-layer statement. **Multi-step prediction rows** (the
next-token distributions after a fixed set of probe continuations) and the four-state source that would have
illustrated them: deferred together (section 16).

## 6. References beside every result

| Reference | Built as | Answers |
|---|---|---|
| random wrong pairing | Π uniform; 200 for counts, 50 for through-depth; common across sites, summaries, models | is the correct pairing special at all? |
| cut-token-preserving wrong pairing | Π uniform within groups of chunks sharing the exact cut token id; chunks with a unique cut token stay fixed (keep their own prediction, so this reference is conservative). **Movable coverage** (share of chunks in groups of 2 or more) and the share of moved chunks whose partner shares the subset are printed | is it more than the cut token? |
| prediction only | the prediction walk's top 16 eigenvectors V_F (the flat-layer limit, P10) | how much of a picture is the prediction's own shape |
| past only | A_l², the paired partner | what the layer shows without the prediction |

## 7. Through-depth operator

`spacetime.py` unchanged: sites l in {2, 4, ..., 16}, L = 8; L = blockdiag((I - M_l)/c_l) + t (D_path ⊗ I) on functions
with mean-zero slices; 24 modes computed with `modes(ms, c, t)`, 20 checked, 16 read.

**Units and coupling are those of the past-only partner**, for the past-only run, the correct product and every wrong
pairing: c_l = mean of the first 8 nontrivial 1 - λ_k(A_l²), t = t* from `balance_direct` on the A_l². The paired
difference is then at fixed units and coupling, and P9 applies. The product's own fitted c_l and t* are computed and
reported, not used. **c_l wrapper** (`solve_b.site_scales_mz`; `spacetime.site_spectrum` subtracts the constant
projector, leaving a zero that can be chosen over genuine negative eigenvalues): full `eigh` of M_l, remove the one
eigenvector with |⟨v, 1/sqrt(N)⟩| > 1 - 1e-10 (asserting eigenvalue 1 to 1e-10), take the 8 largest of the rest. The
resulting c and the stored decomposition are passed to `modes` and `balance_direct`; `spacetime.py` is not edited.

**Runs per model.** Full sample (2,048): for each summary, past only and with the prediction; for the state at the cut,
50 random and 50 cut-token-preserving wrong pairings (eigenvalues and modes); for the other two, 20 random wrong
pairings (exploratory). Halves (1,024): past only and with the prediction, state at the cut.

**Acceptance (fails closed):** residuals of the first 20 modes below 1e-8; for past-only runs |λ_1/(t*·4 sin²(π/16)) -
1| < 1e-6; occupancy sums to 16; slices mean zero, modes orthonormal to 1e-9; finite outputs; Sinkhorn error below
1e-10. **Cutoff rule:** with g_K = (λ_{K+1} - λ_K)/λ_K, read K = 16 if g_16 ≥ 0.01, else the largest K in 8..15 with
g_K ≥ 0.01, else the low space is **unresolved**: spectra are shown, modes are not.

## 8. Measured quantities

**8.1 Per-layer count.** For each model, summary and layer (0, 2, ..., 16): the top 64 eigenvalues of M_l on mean-zero
functions (full `eigh`), and for each of 200 common wrong pairings the leading eigenvalue of M_l^Π, matrix-free
(Lanczos on x ↦ A_l Π S_F Π^T A_l x with mean-zero projection, tolerance 1e-10, convergence asserted). **n_l = the
number of eigenvalues of the product exceeding (1 + 1e-9) × the largest of the 200 null leading eigenvalues**, censored
at 64: "eigenvalues exceeding the largest of 200 wrong-pairing maxima, at this graph resolution". Under exchangeable
pairing the true leading eigenvalue beats all 200 with probability 1/201, so a cell shows n_l ≥ 1 by chance at most
that often; over the 54 cells (2 models × 3 summaries × 9 layers) about 0.27 such cells are expected under a complete
null. A cell's count does not identify each counted direction as a separate effect. The same with cut-token-preserving
relabellings for the state at the cut (n_l^tok); at layer 0 the state at the cut is a function of the cut token, so
n_0^tok = 0 exactly (P10), a built-in check. Halves: state at the cut, both references.

**8.2 Energy cost (prespecified).** State at the cut, each model: E = Σ_{k≤16} λ_k(L_with) - Σ_{k≤16} λ_k(L_past),
units and coupling of the past-only run; E_Π the same for each wrong pairing. p = (1 + #{E_Π ≤ E})/51, one-sided
(smallest attainable 0.02), separately for random and cut-token-preserving relabellings: four tests. Summed eigenvalues
are used, so a near-tie inside the 16 does not matter.

**8.3 Cross-model increment (prespecified).** X = [G_2 ... G_16] (N × 128), G_l the 16-mode slices at site l.
Alignment A(X, X') = ||X^T X'||_F²/(||X^T X||_F ||X'^T X'||_F). Δ = A(X^Pythia_with, X^OLMo_with) - A(X^Pythia_past,
X^OLMo_past); Δ_Π the same with Π applied to **both** models' prediction walks, which keeps prediction-prediction
agreement and breaks only layer-prediction pairing. p = (1 + #{Δ_Π ≥ Δ})/51 for each reference: two tests. Shown
beside it: prediction-only agreement A(V_F^Pythia, V_F^OLMo); total agreement A(X^Pythia_with, X^OLMo_with) against 200
free chunk relabellings; how prediction-shaped each picture is, A(X_with, V_F) against the wrong-pairing values
A(X_with^Π, Π V_F).

**8.4 Descriptive.** **Occupancy** o_l = ||G_l||_F²/K by layer. **Hellinger token decomposition:** h(v) =
Σ_{ij} S_F,ij (sqrt(p_i(v)) - sqrt(p_j(v)))²/2 divided by Σ_{ij} S_F,ij d_F(i, j), the share of the prediction walk's
distance carried by token v (computed as p(v)·1 - sqrt(p(v))^T S_F sqrt(p(v)) per token); the top 20 tokens and the
share carried by tokens that are some chunk's top-1. **Attribute shares** of a space X: ||Q_Z^T X||_F²/||X||_F² with
chance rank(Z)/(N - 1), for cut token (ids with 20 or more chunks; coverage printed), subset, cut position (4 cosine
harmonics), predicted top-1 token, prediction entropy (Legendre 1-4 of its rank). They overlap and omit rare tokens;
they describe, they do not validate. Describing a prediction-built basis by predicted token is not evidence of
predictive function.

## 9. Known answer before spend (gate)

**Source.** Mess3 (Marzen and Crutchfield 2017): hidden states {0, 1, 2}; move from i to j with probability 1 - 2x if
j = i, else x; on arrival in j emit j with probability α, each other symbol (1 - α)/2; x = 0.05, α = 0.85. The belief b
is computed exactly by forward filtering from the uniform start; the next-symbol distribution is (b^T T) E, with T the
3 × 3 transition matrix and E the 3 × 3 emission matrix above (both are fixtures in `test_b.py`). N = 1,024
independent sequences of 192 symbols, cut at the last; chunk = the last 64 positions. **Seeds 70, 71, 72** (not the
reviewer's 60-62). The current symbol explains about 80% of belief variance, so recovery is scored on the
**within-symbol belief** c_w: the belief minus its mean within the current-symbol class (2 dimensions).

**Synthetic layers** (width 64, 8 sites): x_l(i, s) = β_l R_l c(i, s) + Q_l g_i + η_l E_l e(i, s) + 0.07 ε, with c the
centred belief scaled to total variance 2, g_i a per-sequence distractor in R^4 (constant along the chunk) with total
variance 6, e the one-hot current symbol scaled to total variance 1, ε standard normal, R_l, Q_l, E_l fixed random
orthonormal columns per site (`default_rng(31)`). Schedule: sites 0-1 **absent** (β = η = 0); site 2 and 7
**symbol-only** (β = 0, η = 1); sites 3-5 **present** (β = η = 1); site 6 **scrambled** (belief and symbol of donor
π(i)). Futures: the true one; an **unrelated** one (each sample's future from an independent sequence, its donor).

**Run:** the production path of sections 3-8, state at the cut, PR64 with the tie rule (predictions tie often here:
long common suffixes give near-identical posteriors), 200 random and 200 **symbol-preserving** wrong pairings (the
analogue of cut-token-preserving). **Capture** of c_w by a site walk = ||U^T Q||_F²/2, U the walk's top 16 nontrivial
eigenvectors (orthonormal, full rank), Q an orthonormal basis of c_w; chance 16/1,023 ≈ 0.016.

**Pass conditions (frozen before any run; every seed; a failure stops the stage and is reported, nothing is tuned):**

| # | Where | Condition |
|---|---|---|
| K0 | present sites | fixture informative: past-only capture ≤ 0.3 and prediction-only capture ≥ 0.5. If K0 fails the fixture is redesigned with the owner; it is not a result about the method |
| K1 | present sites | product capture ≥ 0.5; n_l ≥ 2 against random and ≥ 1 against symbol-preserving relabelling |
| K2 | symbol-only sites | product capture ≤ 0.15; n_l against symbol-preserving relabelling summed over both sites ≤ 1 (against random, reported: legitimately positive) |
| K3 | absent and scrambled sites | n_l against random summed over the three sites ≤ 1; at the scrambled site, capture of recipient and donor c_w ≤ 0.15 |
| K4 | present sites, one random and one symbol-preserving wrong pairing | product capture of recipient and of donor c_w ≤ 0.15 |
| K5 | unrelated future, every site | n_l against random summed over 8 sites ≤ 1; capture of recipient and of future-donor c_w ≤ 0.15 |
| K6 | flat-layer fixture (separate, exact) | S_l = the complete graph at every site: the product's top 16 eigenvectors span V_F (largest principal-angle sine < 1e-8 when V_F's 16/17 gap exceeds 1e-6); n_l = 0; E and Δ identical to every wrong pairing |

**Reported, not gated:** through-depth energy cost with its p; occupancy by site; belief energy ||Q^T G_l||_F² at
non-present sites (depth leakage through the coupling); past-unit and own c_l, t*.

Cost: about 30 minutes on 8 cores; on the VPS if it has capacity, else the first Modal unit ($0.32).

## 10. Other checks before spend (VPS)

- **Summaries:** the three kernels against explicit feature maps (1e-12); a **reversal fixture** (a chunk of
  orthogonal states and its reverse: equal averages, aligned similarity zero); pooled-norm check; layer-0 recomputation.
- **Product and references:** P1-P6 and P9-P10 on random walks; shared-cut, commuting, complete-graph and flat-layer
  cases; a token-function layer under token-preserving Π (identical spectra); the `site_scales_mz` wrapper on an
  all-negative mean-zero spectrum (M = -A²/(N - 1)), where `site_spectrum` would wrongly pick 0; Stage A's pre-B
  operator fixtures (`SPEC.md` 9.1) through the product with S_F = I.
- **Tie rule:** rows with 63, 64 and 100 tied minima; equivariance under relabelling with ties.
- **Count and tests:** walks sharing exactly three smooth directions give n = 3; independent walks give n ≥ 1 in about
  1 of 201 seeds (checked on 400 seeds); the matrix-free leading eigenvalue equals dense `eigh` (1e-10).
- **Readouts:** alignment, attribute shares and the Hellinger decomposition on hand cases; probe rules on toy
  documents (full-text de-duplication, cut draw, halves); `test_spacetime.py` unchanged.

## 11. The site

Added to the existing site (`SPEC.md` 12: facts tokens, no hand-typed numbers, live charts, `check_browser.py`).
**Presentation rule:** a product picture is never shown as "what the layer knows about the future"; wherever it appears
its wrong-pairing and prediction-only references are beside it.

### 11.1 The chunk viewer (new tab **Through depth**)

It leads with one chunk, not a cloud.

| Control | Values | Why |
|---|---|---|
| Chunk | random, or by index | the object of study |
| Model | Pythia / OLMo | same chunks, two models |
| Compare chunks by | state at the cut / recent past / chunk average | the three summaries |
| Layer | 0, 2, ..., 16 | depth |

Panels: (1) the chunk: 64 tokens, cut token marked, 32 earlier tokens greyed, subset, cut position, the actual next
token, the model's top 5 predictions; (2) side by side, its 8 nearest chunks in the layer's geometry and its 8 nearest
in the prediction geometry, each a one-line snippet with its distance, so a reader can judge "near" by reading; (3) the
layer's spectrum: the product's top 64 eigenvalues against the largest of the 200 wrong-pairing values (random and
cut-token-preserving), with n_l. A closed **Modes** section (past only / with the prediction; colour by attribute; depth
slider) shows a scatter on joint modes 1 and 2 only when both are separated from their neighbours by g ≥ 0.01, local
to the setting and labelled so; otherwise it says "modes 1-2 not resolved" and shows none. Data: about 8 MB.

**Primer additions:** chunk and cut; the three summaries; the prediction geometry; the product as prediction roughness
after layer smoothing, and why its pictures need the references; wrong pairings; the count.

### 11.2 Experiments entries

| Entry | Lead figure: x · y · facets · uncertainty | Sentence a reader should be able to say |
|---|---|---|
| E4 Does a layer's geometry pair with the model's prediction beyond chance? | x = layer · y = n_l · colour = summary; filled marker where n_l^tok ≥ 1 as well (state at the cut) · panels = model · thin lines: halves (state at the cut). Supporting: spectra against null maxima at layers 0, 8, 16; Hellinger token shares; attribute shares of the counted directions | "At layer [l], [n] eigenvalues exceed every wrong pairing ([m] beyond the cut token); [the halves agree / indeterminate]; at layer 0 the cut token alone gives [k]." |
| E5 Through depth, does the correct pairing matter? | x = summary · y = energy cost E · dot: correct pairing; strips: 50 random and 50 cut-token-preserving wrong pairings (20 random for the exploratory summaries) · panels = model. Tiles: the six prespecified p-values; Δ against Δ_Π; prediction-only agreement; prediction-shapedness against wrong pairings. Supporting: occupancy, past only against with | "Agreeing with its own prediction costs less than [x of 50] wrong pairings ([y of 50] that keep the cut token); the models' agreement rises by [Δ] against [Δ_Π]; prediction alone agrees [a]." |
| E6 Does the construction find a planted belief, and only where it is? | x = synthetic site, shaded by role · y = capture of the within-symbol belief · lines: product, past only, prediction only, wrong pairing · seeds thin. Tiles: K0-K6 pass or fail | "Where the belief is planted the product captures [x] of its within-symbol part (past only [y], wrong pairing [z]); at symbol-only, absent and scrambled sites it captures [w]." |

Each entry follows the template (question, short answer with one bold sentence, lead figure, 3-5 tiles, "what was
computed", at most six caveats, decisions in a closed `<details>`). Method gains a Stage B section with a worked
six-chunk example of A, S_F, M and a wrong pairing that a reader can recompute by hand.

## 12. Verification

**Three inventories.** (1) **Inputs** on the Volume: chunk states, distributions, exported weights, the probe file
(ordered hashes, cuts, source indices), seeds and the SHA-256 of every relabelling, conventions. (2) **Full-precision
evidence** on the Volume: float64 walks for the audited cells, all eigenvalues and null maxima, float64 modes, saved
distance rows, the gate's synthetic truth (beliefs, scramble, donors). (3) **Public release**: int8 slices, facts,
spectra, viewer data, audit report. The audit checks (1) → (2) at full precision and (2) → (3) for every headline fact
and one packed viewer card; a hash fixes the evidence, it does not show the statistic was right.

**Independent audit (U5):** numpy and the standard library, importing nothing from the repository, written by a
different agent. Both models:

1. **Prediction:** 64 chunks (`default_rng(41)`): softmax from the stored layer-16 cut state through the exported norm
   and head, against the stored native distribution (1e-6).
2. **Summaries:** 16 chunks (`default_rng(42)`): complete distance rows at sites 4 and 12 and layer 0, by explicit lag
   sums, for the state at the cut and one case each of the recent-past and average paths (1e-9).
3. **Rebuild:** state at the cut, all 8 sites and the prediction walk at full N with its own bisection, tie rule and
   Sinkhorn (walk entries 1e-9); its own products, c_l (with the constant removed) and t*; the published 20 modes
   satisfy the eigen-equation (residual ≤ 1e-7), mean-zero, orthonormal.
4. **Low-end ordering (independent):** the full dense operator (16,384 × 16,384) of the primary product and its partner,
   all eigenvalues by `eigvalsh`; remove one copy of each known slice-constant eigenvalue t·4 sin²(jπ/16), j = 0..7;
   the lowest 20 remaining equal the published ones (1e-8), so no lower mode was skipped. A small complete-spectrum
   fixture (N = 50, L = 3) checks this procedure first.
5. **Counts and tests:** n_l at sites 4 and 12 from its own eigenvalues and replayed relabellings; E and Δ for 5 of the
   wrong pairings; p-values from the stored values.
6. **Gate:** an independent forward filter from section 9's matrices for seed 70, against the stored beliefs (1e-12),
   and K1-K5 recomputed from the stored eigenvectors.
7. **Inventory** of the release by SHA-256, as `audit_a.py`; the site build refuses unless the report passes. The
   report lists what it does not reproduce (extraction, the exploratory summaries' through-depth runs).

## 13. Compute plan

**Separate Stage B ledger and cap** (`/ledger/stage_b/`, its own `STAGE_B_CAP`; the Stage A ledger is untouched).
Before dispatch each unit reserves (its limit + 3 minutes of start-up) × rate, written durably to the Volume. The
completion audit (U4, U5 and one retry of U5) is reserved before any exploratory work. After an external termination,
`modal_app.py reconcile-b`, run from the VPS, charges every reservation without a completion record at its full amount
unless Modal's usage shows less. At most two attempts per unit, never automatic. Each stage has its own coordinator,
limit = longest unit limit + 30 minutes. Rates as `SPEC.md` 11: L4 + 4 cores + 32 GiB $1.04/h; 16 cores + 64 GiB
$1.27/h; 8 cores + 32 GiB $0.63/h; 2 cores + 8 GiB $0.16/h. All work after extraction is CPU with numpy.

| Unit | Hardware | Work | Expected | Limit | Spend |
|---|---|---|---|---|---|
| U-HMM | 8 cores | section 9 (unless run on the VPS) | 30 min | 50 min | $0.32 |
| U0 probe | 2 cores | stream, filter, tokenise, cut, de-duplicate | 30 min | 50 min | $0.08 |
| U1 extract ×2 | L4 | 2,048 forward passes; first 64 documents timed, then re-admitted | 15 min | 20 min, then from rate | $0.62 |
| P pilot | 16 cores | Pythia, state at the cut: 9 walks, 200 nulls at one site, one paired solve, 5 wrong-pairing solves, one recent-past audit row, hashing 100 files, peak memory; outputs reused | 20 min | 35 min | $0.47 |
| U2 walks and counts ×2 | 16 cores | 27 walks, 54 counts with nulls, halves, Hellinger decomposition, neighbour lists | 25 min | from pilot | $1.06 |
| U3 solves ×2 | 16 cores | 6 paired, 100 + 40 wrong-pairing, 4 half solves | 30 min | from pilot | $1.27 |
| U4 readout | 16 cores | tests, alignments, attributes, facts, viewer data | 20 min | 40 min | $0.42 |
| U5 audit | 16 cores | section 12 | 45 min | 75 min | $0.95 |
| coordinators, start-ups | | | | | $0.40 |

**Projected about $5.60; proposed cap $8.** Wall-clock about four hours. Not a verified total: the largest uncertainty
is U3 (about 150 solves per model at N = 2,048; each wrong pairing needs products, eight inversions and a Lanczos
solve, about 10 s by arithmetic, unmeasured), then U1's extraction rate (2,048 passes of 257-1,024 tokens). Anchors:
Stage A's extraction of 36 × 2,048 tokens took 1-5 minutes per model on an L4; a dense 1,792-node graph took 90 s on
one thread.

**Tiers, defined before launch** (each complete, audited and labelled on the page; the stop rule after the pilot picks
the first that fits under the cap with the audit holds):

| Tier | Content | Page label |
|---|---|---|
| 1 | everything above | none |
| 2 | exploratory summaries keep their counts and paired runs but lose their 20 wrong pairings; half solves dropped (half counts kept) | "wrong-pairing runs only for the state at the cut" |
| 3 | state at the cut only: counts, paired runs, 50 + 50 wrong pairings, both models, audit | "the recent past and chunk average were not run" |

If tier 3 does not fit, stop and report the pilot's numbers.

**Stage A's lessons.** Limits come from a measured pilot with margin, or are the table's for small fixed work. Every
long unit commits per step (per walk, per solve, per document batch) with timing in the ledger, and resumes. Nothing
optional runs before critical outputs (primary before exploratory; facts before viewer data). Every index and key is
int64. Before launch the owner confirms the workspace's remaining billing limit exceeds the cap. Fetch with
`modal volume get tm-exp1 /chunks/<dir>` (release and audit only, about 15 MB). The VPS was overloaded while this was
written (swap full, root disk down to 80 MB free): check before fetching and building.

## 14. Code

New in `exp1/`: `probe_b.py`, `extract_b.py` (narrow mode), `views_b.py` (summaries, prediction geometry, tie rule;
calls `kernels.py`), `product_b.py` (products, matrix-free nulls, counts), `solve_b.py` (`site_scales_mz`, runs
`spacetime.py` unchanged), `readout_b.py`, `hmm_check.py`, `audit_b.py` (independent), `test_b.py` (section 10).
`modal_app.py` gains the Stage B ledger, `STAGE_B_WORKERS` and `reconcile-b`. Site: `xsite/gen_e4.py`, `gen_e5.py`,
`gen_e6.py`, `gen_depth.py`.

## 15. Decisions made on the owner's behalf

| # | Decision | Chosen | Alternative not taken |
|---|---|---|---|
| 1 | Size | 2,048 chunks, one per document | 4,096 (about $9) |
| 2 | Cut | positions 256-1,023 of documents of at least 1,024 tokens | anywhere in any document (position tied to length) |
| 3 | Subsets | natural mix, reported | a per-subset cap |
| 4 | Summaries | state at the cut (primary), recent past at memory 8, chunk average | memory 64 inside the chunk (nearly the average at 64 lags) |
| 5 | Centring | one global mean per model and site (Stage A's cosine) | none (then cosine is dominated by the shared mean direction) |
| 6 | Reading | cosine | dot (deferred) |
| 7 | Prediction geometry | Hellinger distance of the one-step distribution | centred logits; top-k overlap |
| 8 | Product | A_l S_F A_l; partner A_l² | lazy prediction walk; S_F²; prediction as an end site |
| 9 | Units for the pair | the partner's c_l and t* for both, so the difference is at fixed coupling | each run its own fitted units |
| 10 | References | random and cut-token-preserving wrong pairings, prediction only, same relabelling in both models | free relabelling only |
| 11 | Graph rule | PR64 with an equivariant tie rule, fixed | kNN15, or a switch chosen in the pilot |
| 12 | Uncertainty | two disjoint halves for the primary count | bootstrap (duplicates break the graph rule) |
| 13 | Viewer | one chunk first; scatter only on resolved modes | animated pooled PCA (undefined: the pooled scatter of the slices is the identity) |
| 14 | Compute | CPU numpy throughout after extraction | a GPU port |

## 16. Deferred

Multi-step prediction rows (probe continuations, about $0.60) with a source built so that two beliefs agree for one
step and differ for two; a sentence-embedding baseline on the same chunks (owner's list); a training-checkpoint axis;
the dot reading; kNN; 4,096 chunks; a held-out outcome for predictive utility (observed next token with matched
controls); a tiny transformer trained on Mess3; an order-only contrast; a one-text version; out-of-sample extension.

## 17. Open questions for the owner

1. **Tier:** if the pilot does not fit tier 1, are tiers 2 and 3 acceptable as defined?
2. **Claim:** agreement beyond the cut token (tested by the token-preserving reference) is the strongest claim made.
   Enough?
3. **Billing:** the workspace's remaining Modal limit this cycle is above $8?
4. **Recent past at memory 8** rather than 64: acceptable?

## 18. Response to review 1

| # | Finding | Response | Where |
|---|---|---|---|
| 1 | Product modes reproduce the prediction under a flat layer | Accepted. Described as prediction roughness after layer smoothing; P10 states the flat-layer and token-function cases; K6 is a mandatory fixture; only pairing-dependent statistics are evidence; prediction-only and wrong-pairing references beside every picture; the false v1 claim that relative units undo halving is removed. Adapted on units: rather than only reporting each run's own c_l and t*, the pair shares the partner's, so the difference is at fixed coupling (P9); own values reported | 5, 6, 7, 9, 11 |
| 2 | Hankel conclusion overstated | Accepted. P6 ends at sample canonical correlations; realisation result is conditional motivation; "one-step predictive distribution"; layer 16 an endpoint reference | 3, 5 |
| 3 | Gate fails on ties, passes on the symbol | Accepted. Documents, identical vectors, roundoff and graph ties distinguished; equivariant tie rule; within-symbol belief scored against symbol-only sites, prediction-only, random and symbol-preserving wrong pairings; donor scored; local gating, leakage reported; capture uses full-rank orthonormal eigenvectors; new seeds, frozen. K4 strict occupancy ordering removed | 3, 4, 9 |
| 4 | Four-state source has the wrong known answer | Accepted; removed and deferred with the multi-step row | 5, 16 |
| 5 | The shuffle tests a narrower null | Accepted. Named literally; spectrum and threshold shown; common relabellings; six prespecified tests; family rate for the 54 cells stated; other patterns exploratory, "indeterminate" defined | 1, 8 |
| 6 | Nuisance agreement and circularity | Accepted. Full-text de-duplication and cut distributions; cut-token-preserving reference with coverage; same wrong relabelling in both models for the increment; prediction-only agreement; Hellinger token decomposition; no utility claim | 2, 6, 8 |
| 7 | Summaries do not isolate order; centring contradiction | Accepted. Three summaries of contextualised states; reversal fixture; pooled-norm check; one sentence on global centring | 3, 10 |
| 8 | Flat-tail switch; `site_spectrum` trap | Accepted. One rule, PR64 (dense, no floor, so floor sensitivity does not arise); final-walk tails, gaps, near-disconnection report; cutoff rule with "unresolved"; `site_scales_mz` wrapper with the all-negative test, `spacetime.py` unedited | 4, 7, 10 |
| 9 | Pooled PCA axes undefined | Accepted. Chunk-first viewer with neighbours and spectrum; scatter only on resolved joint modes | 11 |
| 10 | Pilot does not bound the work | Accepted. Separate ledger and cap, durable reservations with start-up, audit reserved first, reconciliation outside the coordinator, end-to-end pilot slice, matrix-free nulls, narrow extraction with native logits, re-admission after 64 documents, tiers defined. $5.60 is a projection, not a bound | 2, 13 |
| 11 | Audit can certify a wrong answer | Accepted. Three inventories; independent low-end ordering by full `eigvalsh`; complete-spectrum fixture; lag and averaging paths and one viewer card audited; gate filter re-derived; worked example on the page | 11, 12 |
| Size | Smallest Stage B worth running | Adopted as the plan | throughout |
