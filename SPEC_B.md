# TransformerModes, experiment 1, Stage B: independent chunks, through depth, and the next-token prediction

Spec B v3.1, 2026-11-24. History: v1 (design); v2 (review 1, `reviews/spec-b-v1-astra.md`); v3 (review 2,
`reviews/spec-b-v2-astra.md`); v3.1 (the corrections of `reviews/spec-b-v3-astra.md`, applied by the orchestrator:
toy sizes for the exhaustive fixture, smallest gaps and the solver hand-off, coordinator schedule, moved fraction
before extraction, inertia guards, spot-check wording); responses in section 18. Replaces the provisional Stage B of `SPEC.md` (sections 6-8,
16). Inherits from `SPEC.md`: models, revisions, tokenizers, separators, forward-pass settings (section 2), the corpus
file (section 3), the site machinery (section 12). Exploratory. Sources: the owner's direction (2026-11-24), Stage A's
results, `exp1/spacetime.py` and `exp1/kernels.py` (DepthEmbedding's operator and graph rule, unchanged).

**The question.** Through depth, is the prediction constraint cheaper when each chunk carries its own next-token
prediction than when the same prediction geometry is attached to the wrong chunks, beyond exact cut-token identity on
this sample; and how much of the picture is already visible from the prediction alone?

**What it does not claim.** A minimal predictive state or a Hankel rank; where in the model a prediction is computed;
that the structure is used by the model (no outcome is held out); agreement beyond textual nuisance in general.

**In one paragraph.** We cut each of 2,048 Pile test documents once at a random position between tokens 256 and 1,023,
let Pythia-1b and OLMo-1B read the real document up to the cut, and keep the last 64 tokens' hidden states at 8 layers
(the **chunk**) and the next-token distribution at the cut. No two chunks share a document, so Stage A's chains along a
text cannot occur. Chunks are compared by three summaries of their states (the state at the cut, primary; the recent
past; the chunk average) and by how alike their next-token distributions are. The **product walk** (a step in the
layer's geometry, a step in the prediction geometry, a step back) charges a pattern for prediction roughness after
layer smoothing. Its pictures alone are never evidence: with a layer that prefers no pair of chunks they are the
prediction's own patterns (P10). Evidence is pairing-dependent only, against the same prediction geometry attached to
wrong chunks. Approval comes in three stages with their own maxima (section 13): the gate ($1.15, or free on the VPS),
extraction and timing pilot ($3.90), completion at the tier the pilot supports (indicative $6.60-8.32, recomputed from
measurements). Expected total about $5.30.

## 0. What Stage A taught

Stage A (site E1-E3): with memory, a token's nearest neighbours move into its own text and next to it (Pythia, layer 8:
own-text share 59% with no memory, 98% at memory 64), a property of the residual stream (random token vectors: 9%
within 64 positions at memory 64, against 51%). Prediction concordance with matched controls falls with memory (0.133,
0.063, 0.017, 0.004 for none, 8, 64, whole prefix). Participation-ratio rows were flat (whole prefix: half a row's walk
weight in other texts, 0.6% of its 64 nearest). The owner judged the window and own-text mean to be fixes for the
sampling and asked for independent chunks and a construction using the prediction. Stage B: one chunk per document,
nothing masked, no per-document or position-based correction, fixed-length chunks, the prediction inside the operator
with exact controls. **Single long texts** stay with Stage A's viewer (linked): on one text neighbouring positions
share both pasts and predictions, which would bring the chain back into both views.

## 1. The primary test, secondary values, exploratory readings

**Primary test (one decision).** Statistic: the energy cost E of section 8.2 for the state at the cut. Reference: 199
cut-token-preserving wrong pairings, common to both models. Per model p = (1 + #{Π : E_Π ≤ E + ε_E})/200, one-sided
(small E is the direction of interest; smallest attainable p = 0.005). **Decision** (intersection-union, valid at
level 0.05 without correction): "the prediction constraint is cheaper for the correct pairing, beyond exact cut-token
identity on this sample" if p ≤ 0.05 in Pythia **and** in OLMo; otherwise "not shown". Predeclared alternatives:
**uninformative token reference** if the mean fraction of chunks actually moved by the 199 relabellings is below 0.3
(no primary decision is then made); **withheld** if a primary walk is inadmissible (section 4) or a primary solve
fails acceptance (section 7).

**Secondary raw values** (printed with their p, no "any success" claim, no correction): E against 49 random wrong
pairings per model; the cross-model increment Δ (section 8.3) against the 199 token-preserving and the 49 random
pairings, with its sign reported separately (a positive Δ and a Δ only less negative than the wrong pairings are
different outcomes).

**Exploratory, descriptive:** per-layer counts n_l (section 8.1), their depth shape, differences between summaries,
mode pictures, attribute shares. Shown with two half-sample curves for the state at the cut; called **indeterminate**
where the halves disagree in sign or a count is censored at 64.

**Gate (Q3):** the planted hidden-Markov check of section 9, run and passed before any other spend.

## 2. Probe and extraction

**Corpus.** As Stage A (`monology/pile-uncopyrighted`, revision `3be90335…`, `test.jsonl.zst`; subset =
`meta.pile_set_name`); Stage A's 36 documents excluded by SHA-256 of the text. **Eligibility** (failures counted by
reason): no forbidden string; at least 95% ASCII; at least 1,024 Pythia tokens; the token assertion of `SPEC.md`
section 2 on the full text (`probe.assert_token_identity`); full-text SHA-256 not already accepted.

**Cut and chunk.** Positions count the separator as 0. Eligible documents are shuffled with `default_rng(1)`; each gets
a cut u uniform on {256, ..., 1,023} from `default_rng(2)` (one draw per document examined). Chunk = positions u-63 to
u. Reject if more than half of the chunk's tokens are whitespace-only or the chunk shares a 32-token id sequence with an
accepted chunk. The first **2,048** accepted are the probe (i = 0..2,047); **halves** are even and odd i. Rejections
depend on the cut, so the accepted cut distribution is reported by subset and length decile. Saved per chunk: source
line, text hash, length, subset, u, ids of positions u-95..u+1.

**Extraction (a narrow new mode).** Each model reads `[its separator] + tokens 1..u` in one forward pass of the full
causal model, one document per batch, `SPEC.md` section 2 settings (float32, eval, eager attention, TF32 off, pinned
versions). Hooks keep block outputs at sites {2, 4, ..., 16} for positions u-63..u, sliced on the device. The
**native** logits at the last position are read from the model's own output and turned into a full softmax in float64.
Hook equality with `output_hidden_states` is checked on the first document. **Check (fails closed, every chunk):** the
exported final norm and head, applied in numpy float64 to the stored pre-norm layer-16 cut state, reproduce the native
softmax to 1e-6 absolute. Stored per model: chunk states (2,048, 8, 64, 2,048) float32, 8.6 GB; distributions
(2,048, 50,304) float64, 0.82 GB; final norm, head and input embedding, 0.82 GB; top-20 ids and probabilities,
entropy, provenance.

## 3. Three summaries and the prediction geometry

x_{i,l}(s) is chunk i's state at site l and lag s (s = 0 at the cut); layer 0 is the input embedding, recomputed from
ids. **Centring:** one global mean μ_l per model and site, over every chunk and lag of the sample analysed, as in Stage
A's cosine construction; the owner's objection was to per-document and position-based fixes, and none is used. Then
y = (x - μ)/||x - μ|| (cosine reading; dot deferred). Each summary is a weighted Gram matrix over lags,
K_W(i, j) = Σ_{s,s'} W_{ss'} ⟨y_i(s), y_j(s')⟩, with distance d = 1 - K_W(i, j)/sqrt(K_W(i, i) K_W(j, j)):

| Summary | W | In words |
|---|---|---|
| **state at the cut** (primary) | W_00 = 1 | the contextualised state where the model predicts |
| recent past, memory 8 | W_ss = a^s, a = 7/8 | the owner's end-aligned kernel K_a, lag by lag; truncation at 64 lags drops a^64 = 1.9 × 10^-4 of the weight |
| chunk average | W_ss' = 1/64² | the mean of the 64 contextualised states |

Three summaries of states that each already depend on the whole prefix; not an ablation of order or of memory (the
average pools contextualised states; it does not shuffle tokens and rerun the model). Checks (fail closed): no ||x - μ||
below 1e-6 × median; the average's pooled norm not below 1e-3 × its median; no off-diagonal distance below 1e-9 except
in the layer-0 state at the cut, where chunks sharing a cut token are identical by construction (tie rule, section 4).

**The prediction geometry.** p_i is the model's next-token distribution at chunk i's cut: a **one-step predictive
distribution** (the empty-continuation block of a row of Golowich, Liu and Shetty's extended logit matrix). With
r_i = sqrt(p_i), the Bhattacharyya coefficient is BC = ⟨r_i, r_j⟩ and d_F = 1 - BC is the squared Hellinger distance.
**Computation:** BC as a float64 Gram matrix of R (2,048 × 50,304, 0.82 GB), in row blocks; pairs with d_F < 1e-6 are
recomputed directly as Σ_v (r_i(v) - r_j(v))²/2, in blocks, to avoid cancellation. A broadcast over pairs and tokens
(1.7 TB) is never formed. Not centred. Near-identical predictions are legitimate and go to the tie rule.

## 4. Graph rule (one, fixed) and admissibility

**Participation ratio 64, no floor**, through a tie-aware row builder `views_b.pr_rows`: a wrapper around the arithmetic
of `kernels._cond_rows` with its own tests, not the unchanged oracle. Untied rows: q(j|i) ∝ exp(-β_i (d_ij - min_j
d_ij)) with 1/Σ_j q(j|i)² = 64 by bisection, identical to `_cond_rows` (tested to 1e-12). **Tie convention:** the
tied-minimum set is T_i = {j : d_ij - min_j d_ij ≤ 1e-12}; if |T_i| ≥ 64 the row is uniform on T_i and zero elsewhere
(an explicit convention: the β → ∞ limit when the ties are exact; with |T_i| = 64 the target is met, above 64 it is
exceeded); such rows are counted and reported. Equal distances get equal weight, so the rule is permutation-equivariant.
Then W = (q + q^T)/2, symmetric Sinkhorn to 1e-11 (`kernels.sinkhorn_sym`), S the walk, A = (I + S)/2.

The walk is therefore **not** guaranteed positive everywhere: tied rows are zero off their tie sets and untied rows can
underflow. Connectivity is tested, not assumed. **Reported for every walk:** post-tie support sizes; tied-row count;
Sinkhorn row residual; number of connected components of W > 0 (`scipy.sparse.csgraph`); conditional tail τ_q and
final-walk tail τ_S (mass outside each row's 64 largest entries); the raw distance scale (median and 5th-95th
percentiles of all d, of each row's nearest and of its 64th distance); the first 12 nontrivial eigenvalues; the gap after
the 16th. **Admissibility.** A *primary* walk (state at the cut, sites 2-16, and the prediction walk) with more than
one component, or a first nontrivial gap below 1e-8, makes the primary solve inadmissible: the primary decision is
withheld, the secondary through-depth values are withheld, and counts are still shown for connected cells. A
disconnected *descriptive* walk (any layer-0 walk, any exploratory summary) keeps its count, labelled "disconnected,
k components", and loses its through-depth panels. Stage A's flat rows are a reason to report τ and the spectrum, not
for another rule; kNN is deferred.

## 5. The product walk

S_l is layer l's walk for the summary in use, A_l = (I + S_l)/2, S_F the prediction walk (not lazy). Per layer:
**past only** (partner) M_l = A_l²; **with the prediction** M_l = A_l S_F A_l; **wrong pairing** Π: M_l^Π = A_l Π S_F
Π^T A_l. One Π is applied at every site, in every summary and in both models (common relabellings). By P2 the product
charges prediction roughness after layer smoothing; it is not an intersection projector and its modes are not "what the
layer knows about the future".

| # | Statement | Status |
|---|---|---|
| P1 | M = A S_F A is symmetric, entrywise nonnegative, doubly stochastic, -I ≤ M ≤ I; I - M ⪰ 0, all `spacetime.py` needs | theorem |
| P2 | f^T (I - M) f = f^T (I - A²) f + (Af)^T (I - S_F)(Af); hence M ⪯ A² | theorem |
| P3 | λ_k(M) ≤ λ_k(A²) for all k; λ_k(M) ≤ λ_k(S_F) where λ_k(S_F) ≥ 0 (Ostrowski) | theorem |
| P4 | M has eigenvalue 1 on mean-zero functions only if both graphs share a disconnection | theorem |
| P5 | S_F = I gives A²; S_F the complete graph gives M = -A²/(N - 1) on mean-zero functions; if S_F commutes with A, M has their eigenvectors with eigenvalues ((1 + μ)/2)² ν | theorem |
| P6 | With A and S_F replaced by orthogonal projections onto centred feature spans (plus the constant), the nontrivial eigenvalues of M are the squared sample canonical correlations of the two feature sets | theorem |
| P7 | The output is a function of the prefix, so the one-step prediction is the same at every layer; n_l describes the layer's geometry at this graph resolution, not where the prediction is computed; layer 16 is an endpoint reference, not a ceiling | theorem (first clause); reading |
| P8 | Eigenvectors whose eigenvalues exceed the wrong-pairing null are smooth functions of a variable common to both views; n_l counts resolvable shared directions, not a dimension | hypothesis (alternating diffusion: Lederman and Talmon 2018; Talmon and Wu 2019, under conditional independence given the common variable, unverified here); tested on the known answer |
| P9 | At equal c_l > 0 and t, L_with - L_past = blockdiag(A_l (I - S_F) A_l / c_l) ⪰ 0, so by min-max no ordered through-depth eigenvalue decreases, and E ≥ 0 | theorem |
| P10 | With U = I_L ⊗ Π: if every A_l commutes with Π, then L^Π = U L U^T. Spectra are equal, E_Π = E (correct and wrong costs equal, not zero), and whole spectral projectors are equivariant (an arbitrary choice within a tied eigenspace need not be). For Δ both models must commute with the same Π. Cases: a flat layer (A_l = αI + βJ, any Π, where the product's modes are the prediction walk's eigenvectors whatever the layer represents); a layer that is a function of the cut token, under token-preserving Π | theorem |

P1-P6 were checked numerically at N = 60 (scratch, to 1e-15); all are tested in section 10.

**Motivation, conditional.** For a stationary process, canonical correlations between long enough past and future
windows give, under minimality and full-rank conditions on the population past-future Hankel operator, the state of a
minimal linear realisation (Akaike 1975); Golowich, Liu and Shetty find extended logit matrices of language models
approximately low rank. Our view is one-step, sampled and nonlinear: no state or rank claim.

**Degenerate cases.** Prediction walk uninformative: M ≈ 0 on mean-zero functions. Prediction dominated by frequent
tokens: the Hellinger token decomposition (8.4). Flat layer: P10. **Not taken:** a lazy prediction walk (adds an
explicit A²/2 past-only term); S_F² (zero-diagonal Sinkhorn walks have large negative eigenvalues, -0.58 at N = 60,
which squaring makes look smooth); the prediction as an end site (no per-layer statement). Multi-step prediction rows
are deferred (section 16).

## 6. References

| Reference | Built as | Answers |
|---|---|---|
| cut-token-preserving wrong pairing | Π uniform within groups sharing the exact cut-token id; singletons stay. Printed per sample: group-size distribution, coverage (share in groups of 2 or more), **mean fraction actually moved** (expected (N - d)/N for d distinct ids), share of moved chunks whose partner shares the subset | is it more than exact cut-token identity? |
| random wrong pairing | Π uniform | is the correct pairing special at all? |
| prediction only | the prediction walk's top 16 eigenvectors V_F (the flat-layer limit) | how much of a picture is the prediction's own shape |
| past only | A_l², the paired partner | what the layer shows without the prediction |

Relabelling streams: token-preserving `default_rng(51)`, random `default_rng(52)`; each Π's SHA-256 is stored. The
references test exchangeability of the paired two-model prediction records relative to the fixed layer records (within
cut-token groups for the token version); subset and formatting associations can remain beyond the cut token.

## 7. Through-depth operator

`spacetime.py` unchanged: sites l in {2, 4, ..., 16}, L = 8; L = blockdiag((I - M_l)/c_l) + t (D_path ⊗ I) on functions
with mean-zero slices; 24 modes from `modes(ms, c, t)`, 20 checked, 16 read. **Units and coupling are the past-only
partner's** for the past-only run, the correct product and every wrong pairing: c_l = mean of the 8 **smallest**
mean-zero gaps, that is of 1 - λ_k(A_l²) over the 8 largest nonconstant eigenvalues λ_k, t = t* from `balance_direct`
on the A_l². The pair is thus compared at fixed units and
coupling (P9); the product's own fitted c_l and t* are reported, not used. **Mean-zero sector everywhere:** the constant
is shifted strictly below the spectrum, M ↦ M - 3·11^T/N (its eigenvalue goes from 1 to -2; all others are in [-1, 1]
and unchanged), and the largest algebraic eigenvalues are taken. `solve_b.site_scales_mz` computes c_l this way (so
`spacetime.site_spectrum`, which can pick its leftover zero over genuine negative eigenvalues, is not used).
**Hand-off to the reused solver:** `spacetime.balance_direct` expects an unshifted decomposition with the constant
first and slices off column zero, so `solve_b` passes it the original (unshifted) past walks and lets it decompose
them itself, or restores the constant-first eigenpair before handing over a cached decomposition; a sorted shifted
decomposition is never passed (it would discard the leading nonconstant direction and keep the constant). Tested on
a non-flat fixture with a unique leading direction, since the flat fixture hides this error. `spacetime.py` is not
edited.

**Runs per model** (tier 1): full sample, each summary, past only and with the prediction; state at the cut: 199
token-preserving and 49 random wrong pairings (eigenvalues and modes); recent past and average: 19 random wrong pairings
each, so every product picture shown has a wrong-pairing reference; halves (1,024): past only and with the prediction,
state at the cut.

**Acceptance (fails closed):** residuals of the first 20 modes below 1e-8; past-only |λ_1/(t*·4 sin²(π/16)) - 1| <
1e-6; occupancy sums to 16; mean-zero slices, orthonormal modes to 1e-9; finite. **Cutoff:** g_K = (λ_{K+1} -
λ_K)/λ_K. For pictures, K = 16 if g_16 ≥ 0.01, else the largest K in 8..15 with g_K ≥ 0.01, else **unresolved**
(spectra shown, modes not). For the increment Δ, K = 16 is fixed; Δ is **withheld** unless g_16 ≥ 0.01 in every
required run (both models' past-only and correct products, and every wrong pairing of the reference used). Individual
wrong pairings are never dropped. E needs no exclusion: a sum of the lowest 16 eigenvalues is defined at a tie.

## 8. Measured quantities

**8.1 Per-layer count (exploratory).** For each model, summary and layer (0, 2, ..., 16): the top 64 mean-zero
eigenvalues of M_l (dense `eigh` of the shifted matrix), and for each of **99** random relabellings the leading
mean-zero eigenvalue of M_l^Π, matrix-free (Lanczos for the largest algebraic eigenvalue of x ↦ A_l Π S_F Π^T A_l x -
3·(1^T x/N) 1, tolerance 1e-10, residual stored and asserted; jobs may be batched across relabellings). With λ* the
largest null value and ε = 1e-8 + 1e-8 |λ*| (at least 100 times the largest stored residual, asserted), **n_l = the
number of eigenvalues exceeding λ* + ε**, censored at 64: "eigenvalues exceeding the largest of 99 wrong-pairing
maxima, at this graph resolution". Under exchangeable pairing a cell shows n_l ≥ 1 by chance with probability at most
1/100; over the 54 full-sample random-reference cells about 0.54 such cells are expected under a complete null. This is
an expected count, not a familywise rate; the counts make no family claim. Also: 99 token-preserving relabellings for
the state at the cut at sites 2-16 (at layer 0 the state at the cut is a function of the cut token, so n_0^tok = 0 by
P10, checked with 9 relabellings); halves (N = 1,024), state at the cut, random reference. **Null eigensolve count**
(tier 1, both models): full random 2 × 27 × 99 = 5,346; full token 2 × (8 × 99 + 9) = 1,602; half 2 × 2 × 9 × 99 =
3,564 at N = 1,024. Total 10,512 leading-eigenvalue jobs, each kind timed in the pilot.

**8.2 Energy cost (primary for the state at the cut).** E = Σ_{k≤16} λ_k(L_with) - Σ_{k≤16} λ_k(L_past), past-only
units and coupling; E_Π the same with M_l^Π. Ties: ε_E = 1e-8 (1 + |E|); a wrong pairing with E_Π ≤ E + ε_E counts
against the correct one (conservative). Decision rule: section 1.

**8.3 Cross-model increment (secondary).** X = [G_2 ... G_16] (N × 128), G_l the 16-mode slices at site l. Alignment
A(X, X') = ||X^T X'||_F²/(||X^T X||_F ||X'^T X'||_F). Δ = A(X^Pythia_with, X^OLMo_with) - A(X^Pythia_past,
X^OLMo_past); Δ_Π the same with Π applied to **both** models' prediction walks, which keeps prediction-prediction
agreement and breaks only layer-prediction pairing. p = (1 + #{Δ_Π ≥ Δ - ε_Δ})/(B + 1), ε_Δ = 1e-8, B = 199 or 49.
Beside it: prediction-only agreement A(V_F^Pythia, V_F^OLMo); total agreement against 199 free chunk relabellings;
prediction-shapedness A(X_with, V_F) against A(X_with^Π, Π V_F).

**8.4 Descriptive.** Occupancy o_l = ||G_l||_F²/K. **Hellinger token decomposition:** h(v) = Σ_ij S_F,ij (r_i(v) -
r_j(v))²/2 over Σ_ij S_F,ij d_F(i, j), computed as Σ_i p_i(v) - Σ_i r_i(v) (S_F R)_{iv} from one product S_F R; the
top 20 tokens and the share carried by tokens that are some chunk's top-1. **Attribute shares** ||Q_Z^T X||_F²/||X||_F²
with chance rank(Z)/(N - 1): cut token (ids with 20 or more chunks, coverage printed), subset, cut position (4 cosine
harmonics), predicted top-1, entropy (Legendre 1-4 of its rank). They overlap, omit rare tokens, describe and do not
validate.

## 9. Known answer before spend (stage i, the gate)

**Source.** Mess3 (Marzen and Crutchfield 2017): states {0, 1, 2}; move i → j with probability 1 - 2x if j = i, else x;
on arrival in j emit j with probability α, each other symbol (1 - α)/2; x = 0.05, α = 0.85. T and E (3 × 3) are
fixtures in `test_b.py`; the belief b is filtered exactly from the uniform start; next-symbol distribution (b^T T) E.
N = 1,024 independent sequences of 192 symbols, cut at the last; chunk = the last 64 positions. **Confirmatory seeds
70, 71, 72** (never the reviewer's 60-62). Recovery is scored on the **within-symbol belief** c_w (belief minus its mean
within the current-symbol class), because the current symbol explains about 80% of belief variance.

**Synthetic layers** (width 64, 8 sites): x_l(i, s) = β_l R_l c(i, s) + Q_l g_i + η_l E_l e(i, s) + 0.07 ε; c the
centred belief at total variance 2; g_i a per-sequence distractor in R^4, constant along the chunk, total variance 6;
e the one-hot current symbol at total variance 1; ε standard normal; R_l, Q_l, E_l fixed random orthonormal columns
(`default_rng(31)`). Sites 0-1 **absent** (β = η = 0); sites 2 and 7 **symbol-only** (β = 0, η = 1); 3-5 **present**;
6 **scrambled** (belief and symbol of donor π(i), π from `default_rng(33)`). **Unrelated-future control:** every
synthetic record (sequences, layers, distractors, noise, π) is held fixed; only the futures are replaced, by those of an
independent set of 1,024 sequences from `default_rng(80 + seed)`, whose beliefs are the donors.

**Capture** of c_w by a site walk = ||U^T Q||_F²/2, U the top 16 mean-zero eigenvectors, Q an orthonormal basis of c_w;
chance 16/1,023 ≈ 0.016. **Gain** at a site = capture under the given pairing minus the largest capture among 19
wrong pairings of the named kind (random, `default_rng(90 + seed)`; symbol-preserving, `default_rng(95 + seed)`).
Counts use 99 relabellings.

**Order.** (1) analytic, tie and calibration fixtures (section 10); (2) **K0 alone**; (3) only if K0 passes, the null
loops and K1-K6. **If K0 fails:** stop with "fixture uninformative", spend nothing further, do not adjust and retry; a
redesign gets a new spec version, fresh confirmatory seeds and a new gate-only authorisation. It is not a result about
the method.

**Frozen conditions (every seed).** The gate requires gains, not absolute leakage limits: a lazy layer can carry
donor-prediction modes without fault (P10), so donor captures are reported raw.

| # | Where | Condition |
|---|---|---|
| K0 | present sites | fixture suitable: past-only capture ≤ 0.3 and prediction-only capture ≥ 0.5 |
| K1 | present sites | gain ≥ 0.2 against both kinds of wrong pairing; n_l ≥ 1 against symbol-preserving relabelling |
| K2 | symbol-only sites | gain against symbol-preserving pairings ≤ 0.05; n_l against symbol-preserving relabelling summed over both sites ≤ 1 |
| K3 | absent and scrambled sites | n_l against random relabelling summed over the three ≤ 1; scrambled site gain ≤ 0.05 |
| K4 | unrelated future, every site | gain ≤ 0.05 against both kinds; n_l against random summed over 8 sites ≤ 1 |
| K5 | primary test machinery | E against 19 symbol-preserving pairings: rank 1 of 20 in every seed (true future); not rank 1 in at least two seeds (unrelated future) |
| K6 | flat-layer fixture (exact) | S_l the complete graph at every site: site spectra of correct and wrong products equal (1e-12); E = E_Π (1e-10); the product projector onto each isolated eigenvalue equals the relabelled one (1e-8); n_l = 0; the past-only lowest eigenvalue has multiplicity N - 1, so the past-only space and Δ are reported **unresolved**, and the run must say so |

**Reported, not gated:** raw donor and recipient captures; occupancy; belief energy ||Q^T G_l||_F² at non-present
sites (depth leakage); past-unit and own c_l, t*.

**Where it can run.** About 1 hour on 4 free cores and under 2 GB of memory (N = 1,024 matrices are 8 MB; the null
loops dominate: 8 sites × 2 futures × 3 references × 99 jobs per seed at about 0.2 s each), plus 10 minutes of
fixtures. On the VPS when its load allows (free); otherwise the stage (i) Modal unit.

## 10. Fixtures and checks before spend (VPS, free; under 15 minutes, under 1 GB)

- **Summaries:** the three kernels against explicit feature maps (1e-12); a **reversal fixture** (a chunk of orthogonal
  states and its reverse: equal averages, aligned similarity zero); pooled-norm check; layer-0 recomputation.
- **Product and references:** P1-P6, P9, P10 on random walks; shared-cut, commuting and flat-layer cases; a token-
  function layer under token-preserving Π (equal spectra); `site_scales_mz` on an all-negative mean-zero spectrum
  (M = -A²/(N - 1)), where `site_spectrum` picks 0; Stage A's pre-B operator fixtures (`SPEC.md` 9.1) with S_F = I.
- **Complete-prediction fixture through the whole path:** identical predictions → complete-graph S_F; observed and null
  counts, E and Δ computed by the production code: n_l = 0, E_Π = E within ε_E (p = 1), Δ withheld or p = 1.
- **Tie builder:** rows with 63, 64 and 100 exact minima; near ties at 1e-13; underflowing rows; post-tie support,
  Sinkhorn residual, component count and gap on a constructed two-component case (inadmissible as primary).
- **Calibration by exhaustive ranking:** N = 6, L = 2, independent random walks; all 720 relabellings enumerated.
  This toy has only 5 mean-zero directions per site and 10 through-depth modes, so it uses its own sizes, stated
  here: scale rank r = 2 (c_l from the 2 smallest gaps), K = 2 for E and Δ with an isolated 2/3 boundary required,
  direct dense spectra. For each labelling taken as "observed": the rank of the **leading product eigenvalue**
  among the other 719 (the count has no p-value of its own; its exceedance rule is what is tested), and the p-values
  of E and Δ, all with the production tie conventions; "rank 1" in K5 uses the same conservative energy-tie rule.
  Their distributions must be uniform or conservative at every level.
- **Matrix-free null:** leading mean-zero eigenvalue equals dense `eigh` of the shifted matrix (1e-10), including a
  spectrum whose mean-zero part is entirely negative.
- **Hellinger:** blocked Gram with direct recomputation against a direct double loop on 200 × 5,000 distributions.
- **Readouts:** alignment, attribute shares, token decomposition on hand cases; probe rules on toy documents;
  `test_spacetime.py` unchanged.

## 11. The site

Added to the site (`SPEC.md` 12 conventions). **Presentation rule:** a product picture never appears without its
wrong-pairing and prediction-only references beside it, and is never described as what the layer knows about the
future. Panels a tier does not support are suppressed, with the tier's label.

**Chunk viewer (new tab Through depth), one chunk first.** Controls: Chunk (random or by index); Model; Compare chunks
by (state at the cut / recent past / chunk average); Layer (0, 2, ..., 16). Panels: (1) the chunk (64 tokens, cut token
marked, 32 earlier greyed, subset, position, actual next token, top 5 predictions); (2) its 8 nearest chunks in the
layer's geometry beside its 8 nearest in the prediction geometry, one-line snippets with distances, and the share of
the row's walk mass those 8 carry (so a broad row is seen as broad); (3) the layer's spectrum: the product's top 64
eigenvalues against the largest of the wrong-pairing values, with n_l. A closed **Modes** section shows a scatter on
joint modes 1 and 2 only when both are separated by g ≥ 0.01, local to the setting, with its references; otherwise
"modes 1-2 not resolved". Data about 8 MB. **Primer additions:** chunk and cut; the summaries; the prediction
geometry; the product as prediction roughness after layer smoothing, and why its pictures need references; wrong
pairings; the count.

| Entry | Lead figure: x · y · facets · uncertainty | Sentence a reader should be able to say |
|---|---|---|
| E4 Does a layer's geometry pair with the model's prediction beyond chance? (exploratory) | x = layer · y = n_l · colour = summary; filled where n_l^tok ≥ 1 too (state at the cut) · panels = model · thin lines: halves. Supporting: spectra against null maxima at layers 0, 8, 16; raw distance scales and tails; token decomposition | "At layer [l], [n] eigenvalues exceed every wrong pairing ([m] beyond the cut token); the halves [agree / disagree: indeterminate]." |
| E5 Is the prediction constraint cheaper for the correct pairing? (primary) | x = model · y = energy cost E · dot: correct pairing; strips: 199 token-preserving and 49 random wrong pairings · tiles: the decision (or "uninformative token reference" / "withheld"), each model's p, moved fraction and coverage, secondary Δ values with sign | "For the state at the cut the correct pairing is cheaper than [x of 199] token-preserving wrong pairings in Pythia and [y] in OLMo: [shown / not shown] beyond exact cut-token identity on this sample." |
| E6 Does the construction find a planted belief, and only where it is? | x = synthetic site, shaded by role · y = capture of the within-symbol belief · lines: product, past only, prediction only, best wrong pairing · seeds thin. Tiles: K0-K6 | "Where the belief is planted the product's capture exceeds every wrong pairing by [g]; at symbol-only, absent and scrambled sites the gain is [h]." |

Entries follow the template (question, short answer with one bold sentence, lead figure, 3-5 tiles, "what was
computed", at most six caveats, decisions in a closed `<details>`). Method gains a Stage B section with a six-chunk
worked example of A, S_F, M and a wrong pairing that a reader can recompute.

## 12. Verification

**Three inventories.** (1) **Inputs** (Volume): chunk states, distributions, exported weights, the probe file (ordered
hashes, cuts, sources), every relabelling's hash and stream, conventions. (2) **Full-precision evidence** (Volume):
float64 walks for audited cells, all eigenvalues, null maxima with residuals, float64 modes, saved distance rows, the
gate's synthetic truth. (3) **Release**: int8 slices, facts, spectra, viewer data, audit report. The audit checks
(1) → (2) at full precision and (2) → (3) for every headline fact and one packed viewer card.

**Independent audit (U5):** numpy and the standard library, importing nothing from the repository, written by a
different agent. Both models:

1. **Prediction:** 64 chunks (`default_rng(41)`): softmax from the stored layer-16 cut state through the exported norm
   and head against the stored native distribution (1e-6).
2. **Summaries:** 16 chunks (`default_rng(42)`): complete distance rows at sites 4 and 12 and layer 0 by explicit lag
   sums, for the state at the cut and one case each of the recent-past and average paths (1e-9); the Hellinger rows.
3. **Rebuild:** state at the cut, 8 sites and the prediction walk at full N with its own bisection, tie convention and
   Sinkhorn (1e-9); its own products, c_l (shifted constant) and t*; the published 20 modes satisfy the eigen-equation
   (residual ≤ 1e-7), mean-zero, orthonormal.
4. **No skipped modes (inertia):** for the primary correct product and its partner in each model, with σ = (λ_16 +
   λ_17)/2 when g_16 ≥ 1e-6: the number of negative eigenvalues of L - σI from its own block LDL^T elimination (the
   inertia of the 8 Schur complements, each by `eigvalsh`), minus the number of slice-constant eigenvalues t·4
   sin²(jπ/16) below σ (exactly known), must equal 16; with σ' = λ_1 - max(1e-6, 10 × the stored residual of mode 1)
   it must equal 0. If g_16 < 1e-6 the check is reported unresolved. The elimination is guarded: if any
   intermediate Schur complement has an eigenvalue within 1e-8 (relative) of zero, a second shift (σ moved by a
   quarter of the gap) is used; if the count is still unstable, ordering is reported **unverified**, never passed. A complete-spectrum fixture (N = 50, L = 3, dense) validates the procedure first.
5. **Tests:** n_l at sites 4 and 12 recomputed from its own dense eigenvalues against **all** stored null maxima; 5
   relabellings replayed independently, which checks those five null values (a spot check, stated as such, not a
   recomputation of the maximum over 99); E and Δ for 5 wrong pairings; every p from the stored values with the tie
   conventions.
6. **Gate:** an independent forward filter for seed 70 against stored beliefs (1e-12); K0-K6 from stored eigenvectors.
7. **Inventory** of the release by SHA-256, as `audit_a.py`; the build refuses unless the report passes. The report
   lists what it does not reproduce (extraction; exploratory through-depth runs). A hash fixes the evidence; it does
   not show a statistic was right.

## 13. Compute and staged approval

Dependencies: numpy and SciPy (pinned; SciPy for `eigsh` and `csgraph`); torch and transformers for extraction only.
**Separate Stage B ledger** (`/ledger/stage_b/`), Stage A's untouched. Each unit reserves (limit + 3 minutes start-up) ×
rate, written durably before dispatch; the coordinator refuses a unit if spent + outstanding reservations would exceed
the stage maximum; completion holds (U4, U5, one U5 retry) are reserved before any exploratory work. After an external
termination, `modal_app.py reconcile-b` (run from the VPS) charges each reservation without a completion record in full
unless Modal's usage shows less. At most two attempts per unit, never automatic. Each stage's coordinator has a limit
covering its whole sequential schedule including one retry of its longest unit (stage (ii): probe, both extractions,
pilot, retry: 50 + 45 + 45 + 45 min plus start-ups, so 4 h), reserved inside the stage maximum (0.25 core, 1 GiB,
$0.02/h); if a retry cannot fit in the remaining maximum the stage stops rather than exceed it. **Before any model
extraction**, U0 computes the fraction of chunks moved by the 199 cut-token-preserving relabellings on the accepted
probe; if it is below 30% the coordinator stops and the owner is told that the primary decision is unavailable on
this probe before paying for extraction. Rates (`SPEC.md` 11): L4 + 4 cores + 32 GiB $1.04/h; 16 cores
+ 64 GiB $1.27/h; 8 cores + 32 GiB $0.63/h; 2 cores + 8 GiB $0.16/h. Limits after the pilot: 1.5 × measured + 10 min.

**Three authorisations, each with its own maximum, each brought to the owner with the previous stage's results;
passing one confers no permission for the next. Maxima are reservations, not expected runtimes.**

| Stage | Units (limit) | Expected | Maximum (reservations) |
|---|---|---|---|
| (i) gate | U-HMM 8 cores (50 min); one retry only if the worker died without a result; coordinator | $0.32 (free on the VPS) | 2 × 53 min × $0.63/h + $0.03 = **$1.15** |
| (ii) extraction and timing pilot | U0 2 cores (50 min) $0.14; U1 × 2 on L4 (45 min each, aborted if the rate after 64 documents projects over 40 min) $1.66; pilot 16 cores (45 min) $1.02; retry hold for the largest unit $1.02; coordinator $0.06 | $1.12 | **$3.90** |
| (iii) completion | U2 × 2, U3 × 2, U4, U5 + retry, at the tier the pilot supports | tier 1 $3.90 | indicative, below; recomputed from the pilot |

**The pilot** (stage ii, Pythia, outputs reused) times each kind of work: the 9 state-at-the-cut walks and the
prediction walk (blocked Hellinger Gram), one recent-past walk (64-lag Gram), one chunk-average walk; one cell's 99
full-size and 99 half-size null jobs; one paired solve with balance; 10 wrong-pairing solves; one audit walk rebuild
and one inertia check; hashing 200 files; peak memory.

**Tiers** (each complete, audited and labelled; the stop rule after the pilot offers the first whose stage-(iii)
maximum the owner accepts):

| Tier | Outputs | References | Audit | Page label | Indicative max (expected) |
|---|---|---|---|---|---|
| 1 | counts for 3 summaries (halves for the state at the cut); through-depth for 3 summaries | primary 199 token + 49 random; 19 random for each other summary | full (section 12) | none | U2 2 × $1.12, U3 2 × $1.33, U4 $0.91, U5 2 × $1.23, coordinator $0.05: **$8.32** ($3.90) |
| 2 | counts for 3 summaries; through-depth for the state at the cut only | primary 199 + 49 | full; no exploratory through-depth to audit | "through-depth runs only for the state at the cut" | U3 2 × $1.12: **$7.90** ($3.60) |
| 3 | state at the cut only: counts, halves, through-depth | primary 199 + 49 | state at the cut only | "primary comparison only: does not complete the three-summary request" | U2 2 × $0.66, U3 2 × $1.08, U5 2 × $1.08: **$6.60** ($2.80) |

Worst case payable on stop: after stage (i) $1.15; after (ii) $5.05; after (iii) at tier 1 $13.37. If no tier is
accepted the product is the pilot report, not an unaudited Stage B. Largest uncertainty: the wrong-pairing solves
(about 290 per model at N = 2,048, about 5 s each by arithmetic) and the 10,512 null jobs; both are timed in the pilot.

**Stage A's lessons.** Pilot-measured limits; per-step commits with timing and resumption; primary before exploratory,
facts before viewer data; int64 indices and keys; owner confirms the workspace billing limit before each stage; fetch
with `modal volume get tm-exp1 /chunks/<dir>` (about 15 MB); check the VPS's load and disk first (swap was full and the
root disk fell to 80 MB free while this was written).

## 14. Code

New in `exp1/`: `probe_b.py`, `extract_b.py` (narrow mode), `views_b.py` (summaries, blocked Hellinger, `pr_rows`),
`product_b.py` (products, shifted matrix-free nulls, counts, ranks), `solve_b.py` (`site_scales_mz`, runs `spacetime.py`
unchanged), `readout_b.py`, `hmm_check.py`, `audit_b.py` (independent), `test_b.py` (section 10). `modal_app.py` gains
the Stage B ledger, stage maxima, `STAGE_B_WORKERS` and `reconcile-b`. Site: `xsite/gen_e4.py`, `gen_e5.py`,
`gen_e6.py`, `gen_depth.py`.

## 15. Decisions made on the owner's behalf

| # | Decision | Chosen | Alternative not taken |
|---|---|---|---|
| 1 | Size | 2,048 chunks, one per document | 4,096 (about twice the cost) |
| 2 | Cut | positions 256-1,023 of documents of at least 1,024 tokens | anywhere (position tied to length) |
| 3 | Summaries | state at the cut (primary), recent past at memory 8, chunk average | recent past at memory 64 |
| 4 | Centring and reading | one global mean per model and site; cosine | no centring; dot (deferred) |
| 5 | Prediction geometry | Hellinger distance of the one-step distribution | centred logits; top-k overlap |
| 6 | Product and partner | A_l S_F A_l; A_l²; partner's units and coupling for both | lazy prediction walk; S_F²; each run its own units |
| 7 | Primary test | E, state at the cut, 199 token-preserving pairings, both models at 0.05 (intersection-union) | six tests with a family correction (needs about 300 relabellings each) |
| 8 | Count | 99 relabellings, additive tie margin, mean-zero sector by shift | 200 everywhere (twice the null jobs) |
| 9 | Graph rule | PR64, no floor, tie-aware wrapper, connectivity tested | kNN15; a pilot-chosen switch |
| 10 | Gate | gains over wrong pairings; K0 first; stop if unsuitable | absolute leakage limits |
| 11 | Uncertainty | two disjoint halves for the primary count | bootstrap (duplicates break the rule) |
| 12 | Viewer | one chunk first; scatter only on resolved modes | animated pooled PCA (undefined) |
| 13 | Approval | three stages with reservation maxima | one cap for the whole stage |

## 16. Deferred

Multi-step prediction rows (probe continuations, about $0.60) with a source in which two beliefs agree for one step and
differ for two; a sentence-embedding baseline (owner's list); a training-checkpoint axis; the dot reading; kNN; 4,096
chunks; a held-out outcome for predictive utility; a tiny transformer trained on Mess3; an order-only contrast; a
one-text version; out-of-sample extension.

## 17. Open questions for the owner

1. **Stage (i):** run the gate on the VPS when it is idle (free, about 1 hour) or on Modal (at most $1.15)?
2. **Tier 3** is a primary-comparison-only result that does not complete the three-summary request. Acceptable if the
   pilot shows tiers 1-2 do not fit?
3. **The claim** "cheaper for the correct pairing beyond exact cut-token identity on this sample" is the strongest made.
   Enough?
4. **Billing:** the workspace's remaining Modal limit covers each stage's maximum?
5. **Recent past at memory 8** rather than 64: acceptable?

## 18. Responses to reviews

**Review 1** (`spec-b-v1-astra.md`)

| # | Finding | Response | Where |
|---|---|---|---|
| 1 | Product modes reproduce the prediction under a flat layer | Accepted: prediction roughness after layer smoothing; P10; K6; pairing-dependent evidence only; references beside every picture. Adapted: the pair shares the partner's units and coupling (P9); own values reported | 5-7, 9, 11 |
| 2 | Hankel conclusion overstated | Accepted: P6 ends at sample CCA; conditional motivation; one-step distribution; endpoint reference | 3, 5 |
| 3 | Gate fails on ties, passes on the symbol | Accepted: tie convention; within-symbol belief; symbol-only sites; donors; new seeds | 4, 9 |
| 4 | Four-state source wrong | Accepted: removed, deferred | 5, 16 |
| 5 | Shuffle tests a narrower null | Accepted: named literally; common relabellings; exploratory labels; indeterminate | 1, 8 |
| 6 | Nuisance agreement and circularity | Accepted: full-text de-duplication; token-preserving reference; same Π in both models; prediction-only; token decomposition; no utility claim | 2, 6, 8 |
| 7 | Summaries and centring | Accepted: contextualised summaries; reversal fixture; global centring stated | 3, 10 |
| 8 | Flat-tail switch; `site_spectrum` trap | Accepted: one rule; gaps and cutoff rule; shifted constant | 4, 7 |
| 9 | Pooled PCA undefined | Accepted: chunk-first viewer | 11 |
| 10 | Pilot does not bound the work | Accepted: separate ledger, reservations, reconciliation, narrow extraction, tiers | 2, 13 |
| 11 | Audit can certify a wrong answer | Accepted: inventories; skipped-mode check; worked example | 12 |

**Review 2** (`spec-b-v2-astra.md`)

| # | Finding | Response | Where |
|---|---|---|---|
| 1 | P10 and K6 overstated; Δ undefined at ties | Accepted. P9 "does not decrease"; P10 restated with U = I ⊗ Π, equal costs not zero, whole projectors, both models; K6 tests spectra, E = E_Π, isolated projectors, and requires "unresolved" for the flat past-only space and Δ; Δ at fixed K = 16, withheld unless every required boundary is resolved, no shuffle dropped; opening question reworded | top, 5, 7, 9 |
| 2 | Count threshold sign; artificial zero in the null | Accepted. Additive margin ε = 1e-8 + 1e-8 |λ*| above 100 × residuals; constant shifted to -2, largest algebraic; complete-prediction fixture through the whole path; tie conventions in E and Δ ranks | 7, 8, 10 |
| 3 | No family decision | Accepted. One primary test (E, state at the cut, 199 token-preserving pairings, intersection-union over two models at 0.05); others secondary raw values; moved fraction, group sizes, coverage printed; "uninformative token reference" predeclared; wording "beyond exact cut-token identity on this sample" | 1, 6, 8 |
| 4 | Gate: fixture, property and code failures confused | Accepted. Fixtures, then K0 alone; stop-and-version policy; gains over 19 + 19 wrong pairings with raw donor captures; exhaustive-ranking calibration (N = 6) instead of 400 seeds; fixed records and streams in the unrelated-future control stated | 9, 10 |
| 5 | Tie rule contradicts positivity | Accepted. Tie-aware wrapper with its own tests; support, residual, components and gap tested; primary versus descriptive admissibility and what is withheld; raw distance scales | 4, 10 |
| 6 | Admission arithmetic; null count; Hellinger; SciPy; ordering audit | Accepted. Three stage authorisations with reservation maxima and per-tier arithmetic; nulls cut to where a test or a displayed threshold uses them (99 per count; 10,512 jobs, counted); blocked Hellinger Gram; SciPy declared; dense ordering replaced by an inertia count at a guarded boundary with residuals and a full-spectrum fixture; each tier's outputs, references and audit defined, unsupported panels suppressed; tier 3 labelled | 8, 10, 12, 13 |
| 7 (r1 #7 residue) | "nearly the average at 64 lags" | Deleted | 15 |
