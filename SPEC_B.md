# TransformerModes, experiment 1, Stage B: independent chunks, through depth, and the prediction

Spec B v1, 2026-11-24. Replaces the provisional Stage B of `SPEC.md` (sections 6-8 and 16). Inherits unchanged from
`SPEC.md`: models, revisions, tokenizers, separators and the forward-pass rules (section 2), the corpus file (section
3), the site machinery (section 12) and the ledger and reservation mechanism (section 11). Exploratory. Sources: the
owner's direction (orchestrator's brief, 2026-11-24), Stage A's published results, `exp1/spacetime.py` and
`exp1/kernels.py` (DepthEmbedding's dense operator and graph rules, used unchanged as the oracle).

**In one paragraph.** We cut each of 4,096 Pile test documents once at a random position between tokens 256 and
1,023, let Pythia-1b and OLMo-1B read the real document up to the cut, and keep the last 64 tokens' hidden states at 8
layers (the **chunk**) and the model's full next-token distribution at the cut. No two chunks share a document, so
Stage A's chains along a text and its document blobs cannot occur, with no window and no mean removal. Two chunks are
compared in three ways (the state at the cut; the recent past, lag by lag with memory 8; the average over the whole
chunk) and, separately, by how alike the model's predictions after them are. Each comparison gives a graph and a walk
at every layer; the **product walk** (a step in the layer's geometry, a step in the prediction geometry, a step back)
keeps only patterns that are smooth in both. We ask, per layer, how many directions the layer shares with the
prediction beyond chance, and we run DepthEmbedding's through-depth operator with and without the prediction step.
A planted hidden-Markov source with a known belief state is the gate before spend. Projected about $8.75, proposed cap
$12; about seven hours of wall-clock.

## 0. What Stage A taught, and what changes

Stage A (`SPEC.md` v3.2, site E1-E3) listed each token's 64 nearest tokens in 32 long texts. With memory, neighbours
move into the token's own text and next to it (Pythia, layer 8, corpus mean: own-text share 59% with no memory, 98% at
memory 64, 51% within 64 positions). This is the residual stream, not the kernel: random token vectors through the same
kernel put 9% of their neighbours within 64 positions at memory 64. Prediction concordance against matched controls
falls with memory (Pythia, mean of four layers: 0.133, 0.063, 0.017, 0.004 for memory none, 8, 64, whole prefix).
Participation-ratio rows were very flat: at the whole prefix half of a row's walk weight lay in other texts while 0.6%
of its 64 nearest did. The owner judged the window and mean removal to be fixes for a problem of the sampling, and
asked for independent chunks and for a construction using the model's predictive state.

| Stage A problem | Stage B answer |
|---|---|
| Neighbours are neighbouring positions of one text | one chunk per document; no two nodes share a text |
| Window and own-text mean are rules that impose results | not used; nothing is masked |
| Whole-prefix similarity depends on prefix length | every chunk has 64 tokens; the dot reading's dependence on position is reported (section 7) |
| Flat participation-ratio rows | flatness measured on chunks in the pilot; a prestated switch to DepthEmbedding's kNN rule (section 4) |
| Concordance was a readout beside the geometry | the prediction enters the operator itself, with an exact past-only partner and a relabelled-prediction null |

**The single-long-text half of the owner's request** is served by Stage A's token viewer, and Stage B adds only a link
to it. Running the product construction on the tokens of one text would bring back what Stage B removes: neighbouring
positions share both their pasts and their predictions, so the chain along the text would dominate both views. A
one-text version is deferred (section 15).

## 1. Questions

| # | Question | Measured | Could come out |
|---|---|---|---|
| Q1 | At which layers does a chunk's geometry share directions with the model's prediction geometry beyond chance, and does it depend on how chunks are compared? | the count n_l of section 7, per layer, comparison and model, against 20 relabellings of the prediction | rising with depth, flat, or peaked; memory raising or lowering it at early layers |
| Q2 | When the through-depth operator must also agree with the prediction, where in depth does its low space live, what is it about, and do the two models agree on it more than on the past-only low space? | occupancy by layer; captured shares of five chunk attributes; cross-model alignment, each against its null | moves toward late layers or not; about the next token, the subset or position; models agree more, less or the same |
| Q3 | Does the construction find a planted belief state where it is present and report nothing where it is absent or scrambled, when the past-only walk does not? | section 8, prestated pass conditions | pass or fail; a failure stops the stage before the main spend |

The claims are descriptive geometry of two models on one probe. Nothing causal is claimed.

## 2. Probe and extraction

**Corpus.** As Stage A: `monology/pile-uncopyrighted` at revision `3be90335…`, `test.jsonl.zst`; `meta.pile_set_name`
is the **subset**. The 36 Stage A documents are excluded by SHA-256 of their text.

**Eligibility** (each failure counted by reason): no forbidden string (`SPEC.md` section 2); at least 95% ASCII
characters; at least 1,024 Pythia tokens; the token assertion of `SPEC.md` section 2 on the full text (identical ids in
both tokenizers, no id in {0, 1, 50277, 50278, 50279}, identical token strings), reusing `probe.assert_token_identity`.

**Cut and chunk.** Positions count the model's separator as 0, so document token k sits at position k. Eligible
documents are shuffled with `default_rng(1)`; for each in turn a cut u is drawn uniformly from {256, ..., 1,023} with
`default_rng(2)` (one draw per document examined, accepted or not). The **chunk** is positions u-63 to u. A document is
rejected if more than half of its chunk's tokens are whitespace-only, or if its chunk shares any 32-token id sequence
with an accepted chunk. The first **4,096** accepted documents are the probe, in their accepted order, indexed
i = 0..4,095. **Halves:** even and odd i, 2,048 each, the two replicates. The cut range does not depend on document
length, so cut position is independent of length and subset by construction. Subsets are not capped: the sample is
"chunks from random places in the first 1,024 tokens of mostly ASCII Pile test documents of at least 1,024 tokens",
and the page says so with the subset table.

**Forward pass.** Each model reads `[its separator] + document tokens 1..u` (u + 1 tokens, 257 to 1,024), one
document per batch, with `SPEC.md` section 2's settings (float32, eval, eager attention, TF32 off, pinned versions),
reusing `extract.forward`. The hook-equality check runs on the first document; Stage A's causal check is not repeated
(each context is read once, whole).

**Stored per model** (`/chunks/` on Volume `tm-exp1`), only what the kernels and the audit need:

| Item | Shape and type | Size | Used by |
|---|---|---|---|
| chunk states at sites l in {2, 4, ..., 16} | (4,096, 8, 64, 2,048) float32, lag 0 = the cut | 17.2 GB | all comparisons |
| next-token distribution at the cut | (4,096, 50,304) float32, full softmax | 0.82 GB | prediction geometry |
| final norm parameters and output head | numpy arrays | 0.41 GB | audit (recomputes the distribution) |
| input embedding matrix | numpy array | 0.41 GB | layer 0 (recomputed from token ids) |
| per chunk: cut u, subset, token ids of positions u-95..u+1, top-20 ids and probabilities, entropy | small | 6 MB | viewer, attributes |
| provenance | JSON | | as Stage A |

**Assertion (fails closed, every chunk):** the distribution recomputed from the stored layer-16 cut state through the
pinned final norm and head equals the model's own softmax to 1e-5 absolute.

## 3. Comparing two chunks

Notation: x_{i,l}(s) in R^2048 is chunk i's hidden state at site l and lag s (s = 0 at the cut, s = 63 at the chunk's
start); layer 0 is the embedding output, recomputed from token ids. μ_l is the mean of x_{i,l}(s) over every chunk and
lag of the sample being analysed (the full sample or one half). **Readings** as in Stage A: **cosine** uses
y = (x - μ)/||x - μ||, so each token counts equally; **dot** uses y = x - μ, so loud tokens count more.

The three **comparisons** are one formula with three lag-weight matrices W (64 × 64):

K_W(i, j) = Σ_{s, s'} W_{s s'} ⟨y_i(s), y_j(s')⟩

| Comparison | W | In words | The owner's original item |
|---|---|---|---|
| state at the cut | W_00 = 1, else 0 | the hidden state where the model predicts | "just the local residual vector" |
| recent past, memory 8 | W_ss = a^s, a = 7/8, else 0 | the last few tokens, compared lag by lag with fading weight | the end-aligned kernel K_a, "smart EMA" |
| chunk average | W_ss' = 1/64² | the mean state over the 64 tokens; order ignored | "a simple running mean"; "big aggregated blocks" |

The recent past is the owner's K_a(t, t') = Σ_s a^s ⟨y_{t-s}, y_{t'-s}⟩ truncated at 64 lags, which drops a^64 =
1.9 × 10^-4 of its weight. Distances: cosine d = 1 - K_ij/sqrt(K_ii K_jj); dot d = [(K_ii + K_jj)/2 - K_ij]/k̄, k̄ the
mean of K_ii. Gram matrices are computed in float64 from the float32 states. Fail closed if any ||x - μ|| is below
1e-6 times the median, or any off-diagonal distance is below 1e-9 (exact duplicates). The state at the cut at layer 0
is the current token's embedding, so hundreds of chunks tie exactly; that one view is not built (section 7 says what
replaces it).

**The prediction geometry.** With p_i the model's next-token distribution at chunk i's cut, the vector sqrt(p_i) has
unit length, and ⟨sqrt(p_i), sqrt(p_j)⟩ = Σ_v sqrt(p_i(v) p_j(v)) is the Bhattacharyya coefficient BC. The distance
d_F = 1 - BC is the squared Hellinger distance: the cosine reading applied to square-root probabilities, so the only
geometric assumption is again a dot product. It is a row of the Golowich-Liu-Shetty extended logit matrix (histories
against continuations) restricted to the empty continuation, in square-root coordinates. Square roots rather than
logits: Hellinger distance is fixed by the distribution, bounded, and weighted toward probable tokens, whereas centred
logit vectors are dominated by the 50,000 improbable tokens. No centring.

## 4. Graph rule

Unchanged from `kernels.py`. **Participation ratio** (PR64): q(j|i) ∝ exp(-β_i (d_ij - min_j d_ij)) with β_i set so
that 1/Σ_j q(j|i)² = 64; W = (q + q^T)/2; symmetric Sinkhorn to tolerance 1e-11; `ea_kernel(C = 1 - d, 64, renyi=2)`.
**kNN15:** union 15 nearest neighbours, weight 1, floor 1e-6, Sinkhorn (`knn_kernel`, DepthEmbedding's shipped rule).
Either gives a symmetric doubly stochastic walk S with zero diagonal. The lazy walk is A = (I + S)/2.

**Flatness, decided in the pilot and fixed before the main run.** τ = the mean over rows of the conditional mass
outside each row's 64 largest q(j|i). On Pythia at site 8 (full sample, cosine) for the four views (three comparisons
and the prediction): if τ > 0.5 for two or more of them, **every** view in Stage B uses kNN15; otherwise PR64. τ is
reported for every view of the main run either way. The known-answer check (section 8) must pass under both rules, so
the switch cannot be chosen to pass it.

## 5. The product walk

For each layer l and comparison, S_l is the layer's walk and A_l = (I + S_l)/2. S_F is the prediction geometry's walk
(non-lazy). Two site walks enter the through-depth operator, one per run:

- **past only:** M_l = A_l A_l (the paired partner; same eigenvectors as DepthEmbedding's walk A_l, squared eigenvalues);
- **with the prediction:** M_l = A_l S_F A_l (a step in the layer's geometry, a step in the prediction geometry, a step
  back).

The two differ only in the middle factor, so any difference between their pictures is the prediction's doing. A null
partner replaces S_F by Π S_F Π^T, Π a random relabelling of the chunks (`default_rng(51)` onward): the same prediction
geometry attached to the wrong chunks.

**What is proved, what holds under conditions, what is a hypothesis.** P1-P6 were checked numerically by the designer on random
Sinkhorn walks at N = 60 (a scratch script, not kept; every check held to 1e-15) and are tested again in section 9.

| # | Statement | Status |
|---|---|---|
| P1 | M = A S_F A is symmetric, entrywise nonnegative and doubly stochastic, with -I ≤ M ≤ I. So I - M is positive semidefinite and (I - M)/c_l is a valid site energy for `spacetime.py`, which needs M ≤ I, not M ≥ 0 | theorem (S_F symmetric doubly stochastic; A lazy) |
| P2 | f^T (I - M) f = f^T (I - A²) f + (A f)^T (I - S_F)(A f): the layer's two-step roughness of f plus the prediction graph's roughness of the layer-smoothed function. A function is cheap only if it is smooth in both | theorem (identity) |
| P3 | λ_k(M) ≤ λ_k(A²) for every k; λ_k(M) ≤ λ_k(S_F) wherever λ_k(S_F) ≥ 0 (Ostrowski's congruence theorem). The prediction step can only remove smoothness | theorem |
| P4 | M has eigenvalue 1 on mean-zero functions only if the two graphs share a disconnection (a nonconstant function constant on the components of both). If either graph is connected, so is the product, as `balance_direct` requires | theorem |
| P5 | Limits. S_F = I gives the past-only walk. S_F = the complete graph gives M = -A²/(N - 1) on mean-zero functions: nothing survives. If S_F commutes with A (for example S_F = S_l), M has their eigenvectors with eigenvalues ((1 + μ)/2)² ν: the product only reweights | theorem |
| P6 | Linear limit. If A and S_F are replaced by orthogonal projections onto centred feature spans (plus the constant), the nontrivial eigenvalues of M are the squared sample canonical correlations of the two feature sets. With past and future features this is canonical-correlation analysis of past against future. Its number of nonzero correlations is the rank of the past-future cross-covariance, which for a linear Gaussian process is the dimension of the minimal state (Akaike 1975; the Hankel singular values) | theorem |
| P7 | The output is a function of the prefix, so the predictive state is the same object at every layer. A layer's n_l says where it is laid out in that layer's geometry at this graph resolution, not where it is computed | theorem (first clause); reading (second) |
| P8 | Eigenvectors of M_l above the relabelled-prediction null are smooth functions of a variable common to the layer's geometry and the prediction geometry, and n_l counts resolvable shared directions, which grows with that variable's dimension and with the graph's resolution (a two-dimensional variable has many harmonics). It is a count, not a dimension | hypothesis. Alternating-diffusion theory (Lederman and Talmon 2018; Talmon and Wu 2019) proves the analogue for manifold-valued common variables when the two views are conditionally independent given it; whether that holds here is unknown. Tested on a known answer (section 8) |

**At the last layer.** For the state at the cut, layer 16 and the prediction are tied by a fixed map (final norm, head,
softmax), so their graphs are close but not equal; where they coincide, P5 says the product adds nothing beyond
reweighting. n_16 for that comparison is a ceiling check, not a finding.

**Degenerate cases and how they show.** (i) Prediction rows nearly flat or the prediction uninformative: M ≈ 0 on
mean-zero functions, n_l = 0, and the through-depth low space has no preferred layer; the top 16 eigenvalues of S_F
are printed beside every result. (ii) Prediction geometry carved up by frequent next tokens (chunks predicting ",",
"\n", " the"): S_F is nearly block-structured by top-1 prediction and the shared directions say mainly "which next
token"; this is real predictive structure, and its size is reported as the predicted-token share (section 7).
(iii) A layer's rows flat: the same at that layer. (iv) Exact ties: excluded by the duplicate check.

**Not taken.** A lazy prediction walk (I + S_F)/2: an unrelated prediction then leaves A²/2, the past picture at half
strength, which relative site units restore (at N = 60: top nontrivial eigenvalue 0.404 against A²/2's 0.411), so it
would not report nothing. S_F² (positive semidefinite, and exactly the Gram matrix of Lederman and Talmon's alternating
diffusion S_F A): Sinkhorn walks with zero diagonal have large negative eigenvalues (down to -0.58 at N = 60), which
squaring turns into false smoothness. The prediction as a ninth site at the end of the depth chain: it charges
disagreement with the prediction at one end only, gives no per-layer statement, and nearly duplicates layer 16 there.

**The multi-step version is deferred.** The proper predictive-state row is the extended logit matrix: the next-token
distributions after the history followed by each of a fixed set of J probe continuations, concatenated in square-root
coordinates (J + 1 blocks, each weighted 1/(J + 1)). It needs J extra short forward passes per chunk and a choice of
continuation set (for example the J most frequent tokens of the probe). One step already separates histories over
50,304 outcomes; the known-answer check includes a source built so that one step misses part of the belief state
(section 8), which measures what is lost before anyone pays for more. Projected later cost with J = 16: about $0.60.

## 6. Through-depth operator

Exactly `spacetime.py`, unchanged: sites l in {2, 4, ..., 16} (L = 8); L_t = blockdiag((I - M_l)/c_l) + t (D_path ⊗ I)
on functions whose slices have mean zero; c_l = the mean of site l's first 8 nontrivial 1 - λ_k(M_l) (relative site
units); t = t* from `balance_direct`, where the first eigenvalue on mean-zero slices equals the first temporal
eigenvalue t·4 sin²(π/(2L)); 24 modes computed, 20 checked, 16 read. Layer 0 is not a site (it enters only section
7's per-layer count, as a text-only reference).

**Solves, per model.** Cosine reading only: 3 comparisons × {past only, with the prediction} on the full sample (6),
the same on each half (12, at one eighth of the cost each), and one null solve (state at the cut, with the relabelled
prediction). The dot reading is not solved through depth (section 14). **Size:** at N = 4,096 in float64 one matrix
is 134 MB; the 8 walks, 8 inverses and the doubled eigenvector stack of `site_eigs` come to about 4.3 GB, well inside
a 64 GiB unit. The same dense solver has run at N = 8,000 × 12 sites in DepthEmbedding; the size was chosen so that it
runs unchanged on CPU, with each half (2,048) about DepthEmbedding's probe size.

**Acceptance per solve (fails closed):** residuals ||L f - λ f|| of the first 20 modes below 1e-8; |λ_1/(t*·4
sin²(π/16)) - 1| < 1e-6; occupancy sums to 16; slices mean zero and modes orthonormal to 1e-9; finite outputs; Sinkhorn
row error below 1e-10 for every S; for PR64 the achieved participation ratio of every row within 1e-6 of 64 (a target
made unattainable by tied minima fails the run). Stored and shown: the first 12 site gaps 1 - λ_k(M_l) per site and
the gaps after modes 12, 16 and 20.

## 7. Measured quantities

**Per-layer count n_l (Q1).** For each model, comparison, reading and layer (0, 2, ..., 16; site 0 only for the
recent past and the chunk average), the top 64 eigenvalues λ_1 ≥ ... ≥ λ_64 of M_l = A_l S_F A_l on mean-zero
functions, and the largest eigenvalue of each of 20 null products A_l Π S_F Π^T A_l. **n_l = the number of λ_k above
the largest of the 20 null values**, censored at 64. When nothing is shared the true pairing is just one more
relabelling, so its first eigenvalue beats all 20 with probability 1/21: n_l ≥ 1 by chance about once in 21. This is parallel analysis (Horn 1965) with the prediction relabelled
instead of the data, and by P6 its linear analogue counts canonical correlations of past and future above noise. Also
stored: the 64 eigenvalues, the 20 null maxima, and the same eigenvalues of A_l² and S_F for scale. **Replicates:** the
two halves, each computed alone (n_l depends on N, so halves sit lower; the shape across depth is what replicates).

**The low space (Q2).** G_l (N × 16) holds the first 16 modes' slices at site l, X = [G_2 ... G_16] (N × 128).
**Occupancy** o_l = ||G_l||_F²/16, the share of the low space at layer l (sums to 1). **Alignment** of two low spaces,
A(X, X') = ||X^T X'||_F²/(||X^T X||_F ||X'^T X'||_F) in [0, 1], unchanged by any rotation of modes; for two models on
the same chunks (cross-model), and for past only against with the prediction (same model). **Null:** 200 relabellings
of the chunks in one of the two, 99th percentile. With independent chunks there is no text or position structure that
a relabelling breaks unfairly, so this is the right null for "do the two pictures order the same chunks alike".

**Attributes, for description only, never in a kernel.** Each is a centred matrix Z with orthonormal basis Q_Z; the
**captured share** of a low space is ||Q_Z^T X||_F²/||X||_F², printed with its chance level rank(Z)/(N - 1).

| Attribute | Columns |
|---|---|
| subset | indicators of subsets with at least 20 chunks, the rest pooled |
| cut position | cos(π j (u - 256 + 1/2)/768), j = 1..4 |
| current token (at the cut) | indicators of ids with at least 20 chunks |
| predicted next token | indicators of the model's top-1 ids with at least 20 chunks |
| prediction entropy | Legendre polynomials of degree 1-4 of the entropy's rank, scaled to [-1, 1] |

These replace Stage A's shares by position, document and token. They are applied to the through-depth low spaces, to
each layer's shared directions (the top n_l eigenvectors of M_l) and to the prediction walk's top 16 eigenvectors. The
current-token share of the prediction walk stands in for the missing layer-0 state at the cut: it says how much of the
prediction geometry the current token alone lays out.

**Dot-reading check.** The Spearman correlation between each chunk's self-similarity K_ii (dot reading) and its cut
position, per comparison and layer. Stage A's length artefact cannot arise from the kernel here, since every chunk has
64 tokens; whatever correlation remains comes from the hidden states and is reported.

## 8. Known answer before spend (gate)

**Sources.** (a) **Mess3** (Marzen and Crutchfield 2017): hidden states {0, 1, 2}; from state i move to j with
probability 1 - 2x if j = i, else x; on arrival in j emit symbol j with probability α, each other symbol (1 - α)/2;
x = 0.05, α = 0.85. The belief b (posterior over the hidden state) is computed exactly by filtering from the uniform
start, and the next-symbol distribution is an invertible affine function of b. (b) **Four-state cycle**: states
A, B, C, D; advance to the next state with probability 0.9, else stay; on arrival A and C emit 0 or 1 with equal
probability, B emits 0 with probability 0.95, D emits 1 with probability 0.95. The next-symbol distribution sees b_A
and b_C only through their sum; the next two symbols separate them.

**Samples and synthetic layers.** N = 1,024 independent sequences of 192 symbols per seed, cut at the last symbol, chunk
= the last 64 positions; three seeds (`default_rng(60 + seed)`). Eight synthetic sites of width 64: at site l, sample
i and lag s,

x_l(i, s) = β_l R_l c(i, s) + Q_l g_i + η_l E_l e(i, s) + 0.22 ε,

with c the belief minus its mean, scaled to total variance 1; g_i a per-sequence distractor in R^4, standard normal,
scaled to total variance 6 (a "topic" that is constant along the chunk, like Stage A's documents); e the one-hot
current symbol, total variance 1 where present; ε standard normal noise; R_l, Q_l, E_l fixed random matrices with
orthonormal columns, different at each site (`default_rng(31)`). Schedule: sites 0-1 **absent** (β = η = 0); sites
2-5 and 7 **present** (β = η = 1); site 6 **scrambled** (belief and symbol of sample π(i), π a fixed relabelling).
The belief is about 9% of each present site's variance. The current symbol is kept out of absent sites so that
"absent" means absent: a layer holding only the current token would legitimately share part of the prediction.

**Run.** The production code paths of sections 3-7 (state at the cut for the gate; the other comparisons reported),
under both PR64 and kNN15. Futures: the true next-symbol distribution; a **negative control** in which each sample's
future comes from an unrelated Mess3 sequence; for the cycle source, one-step and two-step futures.

**Pass conditions (frozen; every seed and both rules; a failure stops the stage and is reported with its numbers,
and no parameter is tuned to pass).** "Captured" means the share of the belief's centred coordinates inside the column
span of the 16-mode slice G_l (or of the top 16 eigenvectors of a walk).

| # | Condition |
|---|---|
| K1 | Mess3, true future: n_l ≥ 2 at every present site; the sum of n_l over the three absent and scrambled sites ≤ 1 |
| K2 | the past-only walk's top 16 eigenvectors capture ≤ 0.5 of the belief at every present site (the belief is minor), while the product walk's capture ≥ 0.8 |
| K3 | through depth: with the prediction, the slices capture ≥ 0.8 of the belief at every present site and at least 0.3 more than past only; at the scrambled site they capture ≤ 0.2 of the scrambled belief c∘π |
| K4 | occupancy with the prediction: each absent and scrambled site holds less of the low space than every present site |
| K5 | negative control: n_l ≤ 1 summed over all sites; capture of the belief ≤ 0.2 at every site |
| — | cycle source (reported, not gated): capture of the part of b_A - b_C not explained by the one-step-visible coordinates, with one-step against two-step futures. Expected well under 0.2 with one step and above 0.6 with two |

Cost: about 30 minutes on 8 cores; on the VPS when it has capacity (free), otherwise the first Modal unit ($0.32), ahead
of every other unit.

## 9. Other checks before spend (VPS, no Modal)

- **Comparisons:** the three kernels against explicit vectorised feature maps, both readings (1e-12); the layer-0
  recomputation from ids against an embedding lookup; the duplicate check catching a planted duplicate.
- **Product walk:** P1-P6 on random Sinkhorn walks; shared-cut and commuting cases by construction; an analytic case
  in which a group indicator is an eigenvector of A (eigenvalue α) and of S_F (eigenvalue φ), giving eigenvalue α²φ
  in M; Stage A's pre-B operator fixtures (`SPEC.md` section 9.1: the two-group analytic fixture with constant depth
  envelope, near-cut fixtures with two and nine groups, equivariance under relabelling) run through the product with
  S_F = I and with an S_F sharing the split.
- **Count:** a constructed pair of walks sharing exactly three smooth directions gives n = 3; a pair sharing none
  gives n ≤ 1 in at least 19 of 20 seeds.
- **Torch port** of `_cond_rows`, `sinkhorn_sym`, `knn_kernel` and the eigensolves for the GPU unit: equal to the numpy
  oracle on CPU in float64 (q, W, S to 1e-12; eigenvalues to 1e-10) at N = 300, masked and unmasked; repeated on CUDA
  at full size in the pilot (1e-10).
- **Readouts:** captured shares, alignment and its null on hand cases; probe rules (eligibility, cut draw, de-duplication,
  halves) on toy documents; `test_spacetime.py` unchanged.

## 10. The site

Added to the existing site (`SPEC.md` section 12 conventions: facts tokens, no hand-typed numbers, live charts,
`check_browser.py`, closed `<details>` for extra views). One new tab, **Through depth**, beside Viewer (Stage A's tokens),
Experiments and Method; Stage A's Viewer gains one link to it and it links back for single long texts.

### 10.1 The chunk viewer

Modelled on DepthEmbedding's embedding tab: each dot is a chunk, placed by its coordinates in the 16-mode low space at
the current depth (axes: the first two principal components of the low space pooled over sites, a fit, named on the
page); a depth slider with Play and occupancy bars above it; interpolated frames between sites for the eye only.

| Control | Values | Why it is there |
|---|---|---|
| Model | Pythia / OLMo | two models on the same chunks; a chunk keeps its place in the list across models |
| Compare chunks by | state at the cut / recent past / chunk average | the three comparisons are the owner's three kernels |
| Agree with the prediction | off (past only) / on | the product construction; off is its exact partner |
| Colour | mode 1-16; subset; cut position; predicted next token (8 most common, rest grey); prediction entropy | modes, and the attributes that describe them |

Depth is the slider in the plot, not a bar control: four bar controls in all. Cosine reading only; dot-reading
results are in Experiments. Switching a control shows "mode numbers and signs are local to one picture" and the
alignment A between the two pictures. **Chunk card (click):** the 64 tokens with the cut token marked and 32 earlier
tokens greyed for context, subset, cut position, the actual next token, the model's top 5 predictions with
probabilities; its 8 nearest chunks under the current comparison at the current layer and its 8 nearest in the
prediction geometry, each as a one-line snippet with its distance, so a reader can check by reading whether "near"
means alike. **Mode card:** depth weights, the five captured shares with chance ticks. **Data** (lazy, per model and
setting): low-space slices int8, 8 sites × 4,096 × 16 × 12 settings = 6.3 MB; texts 1.6 MB; neighbour lists
2 models × 25 views (3 comparisons × 8 sites, and the prediction) × 4,096 × 8 × 4 bytes = 6.6 MB.

**Primer additions** (top of the tab, collapsible, one screen): chunk and cut; the three comparisons; the prediction
geometry (how alike two chunks' next-token distributions are); the product walk in one sentence; past only as its
partner; relabelling as the null; shared directions; modes, depth and occupancy as in DepthEmbedding.

### 10.2 Experiments entries

| Entry | Lead figure: x · y · facets · uncertainty | The sentence a reader should be able to say |
|---|---|---|
| E4 Where in depth does a chunk's geometry agree with the model's prediction? | x = layer (0 = embedding output, 2-16) · y = directions shared with the prediction beyond chance (n_l, 0-64) · colour = comparison · panels = model · thin lines: each half alone; dotted line at 1: what chance gives about once in 21. Supporting: the 64 eigenvalues beside the null maxima at layers 4, 8, 12, 16; the dot reading; captured shares of the shared directions; the self-similarity check | "At layer [l] the state at the cut shares [n] directions with the prediction geometry ([m] at layer 2); the recent past [raises / lowers / leaves] this at early layers; the two models agree in shape." |
| E5 What does the through-depth picture look like when it must agree with the prediction? | x = layer (2-16) · y = share of the 16-mode low space (occupancy) · solid: with the prediction, dashed: past only · columns = comparison, rows = model · thin lines: halves. Number tiles: cross-model alignment with and without (with the null), alignment of the two pictures within a model, the largest captured share. Supporting: captured shares by attribute; per-layer cross-model alignment; the null solve | "With the prediction the low space sits [mostly at layers x-y] instead of [z]; Pythia and OLMo agree [a] with it against [b] without (chance [c]); it is mostly about [attribute]." |
| E6 Does the construction find a planted predictive state, and only where it is? | x = synthetic layer (0-7), shaded absent / present / scrambled · y = share of the planted belief captured by the low space at that layer · solid: with the prediction, dashed: past only · panels: true future, unrelated future · thin lines: seeds, both graph rules. Tiles: n_l per site. Supporting: the cycle source, one step against two | "Where the belief is planted the construction captures [x] of it against [y] for the past-only walk; where it is absent or scrambled, and with an unrelated future, it reports [nothing / n]." |

Each entry follows the site's template (question, two or three sentence answer with one bold sentence, lead figure,
3-5 number tiles, a "what was computed" table, at most six caveats, decisions in a closed `<details>`). The Method
tab gains a Stage B section carrying sections 2-8 of this spec, the subset table and the decisions.

## 11. Verification

**Independent audit (U5)**, numpy and the standard library only, importing nothing from this repository, written by a
different agent from the pipeline's. For both models:

1. **Predictions:** for 64 chunks (`default_rng(41)`), the next-token distribution recomputed from the stored layer-16
   cut state through the exported final norm and head, against the stored distribution (1e-5 absolute).
2. **Comparisons:** for 16 audit chunks (`default_rng(42)`), complete distance rows of all three comparisons, both
   readings, at sites 4 and 12 and layer 0, by the explicit lag sum from the stored states (and from ids and the
   embedding at layer 0), against rows the pipeline saved before discarding (1e-9); 1,000 random pairs per view.
3. **One full rebuild:** state at the cut, cosine, all 8 sites and the prediction walk at full N: distances, the
   graph rule by its own bisection (or its own kNN), its own Sinkhorn; every walk entry against the published walk
   (1e-9). Then its own 8 products A_l S_F A_l (1e-9), its own c_l (1e-9), and the **eigen-equation**: the published 20
   modes satisfy ||L f - λ f|| ≤ 1e-7 under its own operator at the published t*, with mean-zero slices,
   orthonormality (1e-9) and λ_1 = t*·4 sin²(π/16) (1e-6).
4. **Count:** n_l at sites 4 and 12 for that comparison, from its own eigenvalues and the 20 nulls replayed from their
   stored seeds; equal.
5. **Readouts:** occupancy, alignments, captured shares and the facts behind every E4-E6 headline recomputed from the
   published slices and attributes (1e-9).
6. **Inventory:** SHA-256 of every published file and a digest of the map, as `audit_a.py`. The site build refuses
   unless the report passes and the inventory matches the data. The report states what it does not reproduce
   (extraction itself, the dot reading's walks, the other comparisons' through-depth solves).

**What a reader can click:** any chunk, to read its text, its predictions and its nearest chunks in both geometries;
E6, the planted check with its frozen conditions; the audit report from the Method tab. `check_browser.py` drives every
control, a chunk click, Play and both viewports, failing on console errors or external requests; screenshots are
inspected before deploying.

## 12. Compute plan

Modal app and Volume `tm-exp1`, prefix `/chunks/`; Stage B workers behind the existing `--approve-stage-b` flag. Rates
as `SPEC.md` section 11: A100-80GB + 8 cores + 64 GiB $3.39/h; L4 + 4 cores + 32 GiB $1.04/h; 16 cores + 64 GiB
$1.27/h; 8 cores + 32 GiB $0.63/h; 2 cores + 8 GiB $0.16/h; about 2 minutes of start-up per attempt.

| Unit | Count | Hardware | Work | Expected | Limit | Spend |
|---|---|---|---|---|---|---|
| U-HMM | 1 | 8 cores | section 8 (skipped if run on the VPS) | 30 min | 50 min | $0.32 |
| U0 probe | 1 | 2 cores | stream, filter, tokenise both ways, cut, de-duplicate | 30 min | 50 min | $0.08 |
| U1 extract | 2 | L4 | 4,096 forward passes, chunk states, distributions, assertion | 25 min | from pilot | $0.87 |
| P2 pilot | 1 + 1 | A100; 16 cores | CUDA repeat of the port tests; site 8, four views at full N: flatness τ and per-view time; one full solve; the audit's one-walk rebuild timed | 15; 25 min | 30; 40 min | $1.38 |
| U2 views | 2 | A100 | every view: Gram, graph, Sinkhorn, eigensolve, products, 20 nulls, nearest-8 lists, audit rows; halves; dot reading last | 20 min | from pilot | $2.26 |
| U3 solves | 2 | 16 cores | 6 + 12 half + 1 null solves with `spacetime.py` | 45 min | from pilot | $1.91 |
| U4 readout | 1 | 16 cores | counts, alignments and nulls, captured shares, facts, viewer data | 25 min | 45 min | $0.53 |
| U5 audit | 1 | 16 cores | section 11 | 45 min | 75 min | $0.95 |
| coordinators, start-ups | | | | | | $0.45 |

**Projected about $8.75; proposed cap $12**, which leaves room for one failed attempt of any one unit. Wall-clock about
seven hours. **Largest uncertainty:** U3, the dense CPU solves at N = 4,096 (8.5 full-solve equivalents per model,
dominated by eight dense eigensolves of each product), then U1's extraction rate, which is 36 times Stage A's tokens.
Stage A's measured anchors: extraction of 36 contexts of 2,048 tokens took 1-5 minutes per model on an L4; dense
linear algebra at N = 1,792 × 8 sites took about 25 s on one CPU thread, plus 90 s to build a graph on one thread.

**Pilot and stop rule.** U1's first attempt extracts the first 64 documents under a 20-minute limit and records seconds
per document; U1 resumes from there with limit = 1.5 × measured rate × remaining + 10 minutes. After P2: projected
total = spent + 1.2 × (measured per-view time × view count + measured solve time × solve count) + U4 and U5 holds. If
above $12: drop the dot reading (about $0.50), then the half-sample solves (about $0.40), then run the through-depth
solves on the even half only (N = 2,048); if still above, stop and report the pilot's numbers. The flatness rule of
section 4 is applied from P2's τ and recorded before U2 starts.

**Stage A's lessons, absorbed.**

| Lesson (README run log) | Rule here |
|---|---|
| Untimed work expired a limit (synthetic census) | every unit's limit comes from a measured pilot with margin, or is the table's limit for small fixed work |
| Coordinator outlived by a restarted worker | each stage has its own coordinator, limit = longest unit's limit + 30 minutes |
| Restart repeated finished work | every long unit commits per step (per view in U2, per solve in U3, per document batch in U1) with timing in the ledger, and resumes |
| Audit finished, then timed out hashing | the inventory is hashed in parallel and timed in P2; nothing optional runs before a critical output (order: cosine full, halves, dot; facts before viewer data before supporting figures) |
| int32 overflow in a lookup key | every index and key is int64; token ids are cast before any arithmetic |
| Workspace billing limit terminated all containers | before launch the owner confirms the workspace's remaining limit exceeds the cap; the coordinator records any termination as a failed, charged attempt |
| App fetch took 25 minutes | fetch with `modal volume get tm-exp1 /chunks/<dir>`; only `analysis`, `audit`, `site` and the ledger |
| Attempts per unit | at most two attempts per unit, never automatic; reservations as Stage A |

While this spec was written the VPS root disk fell to 80 MB free (2.8 GB an hour later) and swap was full; check
both before fetching (about 40 MB of site data) and building.

## 13. Code

New in `exp1/`: `probe_b.py`, `extract_b.py` (wraps `extract.forward`), `views_b.py` (comparisons, prediction
geometry, graph rules via the torch port), `torch_kernels.py` (the port; `kernels.py` stays the oracle), `product_b.py`
(products, counts, nulls), `solve_b.py` (runs `spacetime.py` unchanged), `readout_b.py`, `hmm_check.py`, `audit_b.py`
(independent), `test_b.py` (section 9). `modal_app.py` gains `STAGE_B_WORKERS`. Site: `xsite/gen_e4.py`, `gen_e5.py`,
`gen_e6.py`, `gen_depth.py` and the Through depth tab.

## 14. Decisions made on the owner's behalf

| # | Decision | Chosen | Alternative not taken |
|---|---|---|---|
| 1 | Nodes | one chunk per document, 4,096 documents | several chunks per document; 2,048 documents (about $5, open question 1) |
| 2 | Cut | uniform on positions 256-1,023, documents of at least 1,024 tokens | cut anywhere in a document of any length (position then tied to length and subset) |
| 3 | Subsets | natural mix, uncapped, reported | a per-subset cap (Stage A's 4 of 32) |
| 4 | Chunk length | 64 tokens (the owner's example) | two lengths |
| 5 | Recent past | memory 8 inside the chunk, lag by lag | memory 64 truncated at 64 lags (weights 1 to 0.37), nearly the chunk average |
| 6 | Order contrast | the three comparisons as a ladder (cut only, recent past, whole chunk) | the same weights aligned against averaged, which isolates order alone (deferred) |
| 7 | Prediction geometry | Hellinger distance between next-token distributions | dot product of centred logits; top-k overlap |
| 8 | Product | A_l S_F A_l with lazy layer walk, non-lazy prediction walk | lazy prediction walk; S_F²; the prediction as an end site (section 5) |
| 9 | Partner and null | past only A_l²; prediction relabelled | DepthEmbedding's A_l as the partner |
| 10 | Per-layer number | count above the relabelled null (parallel analysis) | the sum of top eigenvalues; subspace angles between the two walks' top 16 (blind where the shared structure is minor in the layer) |
| 11 | Graph rule | PR64, switched once to kNN15 by the prestated flatness rule | Shannon perplexity (the owner's suggestion; produced two-item modes in DepthEmbedding) |
| 12 | Readings | cosine throughout; dot for the per-layer count only | dot through depth as well |
| 13 | Sites | layers 2, 4, ..., 16; layer 0 as text-only reference in the count | all 16 layers |
| 14 | Compute | graphs on one GPU (torch port tested against the oracle); solves on CPU with `spacetime.py` unchanged | all on CPU (about $2 more); a GPU port of the solver |
| 15 | Uncertainty | two disjoint halves as replicates | bootstrap (resampling duplicates chunks, which breaks the graph rule) |
| 16 | Multi-step prediction | deferred; its loss measured on the cycle source | a small version now |

## 15. Deferred

- **The multi-step extended logit row** (section 5): J probe continuations per chunk; about $0.60 at J = 16.
- **A sentence-embedding baseline** (the owner's list): the same chunks embedded by a SentenceTransformers model, as a
  further comparison in the count and as a past-only walk, to say what a retrieval-style similarity of chunks shares
  with the prediction.
- **A training-checkpoint axis** (Pythia's public checkpoints): the count and the low space against training step.
- A tiny transformer trained on Mess3, in place of synthetic layers; the order-only contrast; other chunk lengths and
  memories; a one-text version of the product; out-of-sample extension (Nyström) of new chunks; the dot reading
  through depth; a GPU port of the solver; the prediction as an end site, as a comparison.

## 16. Open questions for the owner

1. **Size.** 4,096 chunks (projected $8.75, cap $12) or 2,048 (about $5, cap $7)? The question can be answered with
   2,048; 4,096 gives the attribute readouts more power, and each replicate half is then as large as DepthEmbedding's
   whole probe.
2. **The recent past at memory 8.** It keeps the three comparisons distinct; at memory 64 inside a 64-token chunk the
   lag-by-lag comparison would nearly equal the chunk average. Acceptable?
3. **Billing.** Please confirm the Modal workspace's remaining limit for this cycle is above the cap.
4. **Next step if E4 is interesting:** the multi-step prediction row (about $0.60) or the sentence-embedding baseline
   first?
