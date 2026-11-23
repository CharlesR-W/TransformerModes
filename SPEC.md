# TransformerModes, experiment 1: memory kernels on two language models, through depth

Spec v2.1, 2026-11-23. v1 earlier the same day; v2 answered review 1 (`reviews/spec-v1-astra.md`, see the last
section); v2.1 adds the owner's answers and replaces dense numerics by a sparse, every-token design. Exploratory.
Sources: `BRIEF.md`; DepthEmbedding `SPEC_SPACETIME.md`, `spacetime.py`, `spacetime_stage.py`, `test_spacetime.py`,
`invert/NOTES.md`, `invert/dm.py`, `opt/REPORT.md`, `opt/sparse_core.py`, `site/xsite/AUTHORING.md`;
`deprecated/lm_coherent/SPEC.md`. Model, corpus and price facts checked on huggingface.co and modal.com, 2026-11-23.

**In one paragraph.** We take 32 long documents from the Pile test split, run Pythia-1b and OLMo-1B once on the first
2,047 tokens of each, and store every residual and the models' next-token predictions. A token's similarity to another
compares their recent pasts aligned at the end, with a memory length (none, 8 tokens, 64 tokens, the whole prefix)
and two readings. Every token after a 255-token burn-in is a node. Each similarity gives two graphs at each of 8
layers: the **one-text graph** (the 1,792 tokens of one text) and the **all-texts graph** (all 57,344 tokens of the
32 texts together). DepthEmbedding's through-depth operator ties the layers; its 16 lowest modes are the picture. A
text-only picture built from the same tokens with the same memory is the adversary, and a fixed rule names a winner or
says that none exists. The viewer shows the text coloured by a mode. Projected spend about $25.

## 0. Why this is the smart way to do it

Nothing is spent until the owner has seen this section. Each line: the choice, what it beats, by how much.

**Experimental design**
- **One forward pass per model, everything stored** (20 GB of residuals): every kernel, graph and control is
  recomputed from the store without the models. Rerunning models per variant would multiply the GPU work by the 8
  constructions and make the two models' inputs harder to keep identical.
- **Every token is a node.** The exact kernel already compares every pair of tokens, so sampling 1 token in 7 (v1)
  saved nothing in the kernel and cost sampling noise, a resample design, half-coloured texts and an out-of-sample
  extension. Now the one-text and all-texts pictures are compared on all 1,792 tokens of a text.
- **A text-only adversary and a "no winner" outcome.** The construction is chosen by how much the two models agree
  beyond what the tokens alone produce with the same memory (random vectors per token id; each model's embedding
  output). Choosing the nicest-looking picture, or raw cross-model agreement, would reward surface text statistics
  that both models share trivially (review 1). The adversary adds about 10% to the compute.
- **Controls with known answers run first and gate spending**: an exact planted positive, two independent noise fields
  that must not agree, and an equivariance check, all for cents on synthetic data.
- **Few constructions, one coupling.** 8 similarities (4 memories × 2 readings), coupling fixed at the calibrated
  value t*; sweeps of width, coupling and graph recipes are deferred or kept to one small comparison.
- **A capped timing pilot (about $1) gates the rest**, with fallbacks fixed in advance (section 11).

**Numerical methods** (timings at N = 2,000 tokens × 12 layers from DepthEmbedding `opt/REPORT.md`)

| Step | Chosen | Instead of | Gain |
|---|---|---|---|
| Memory kernel | exact recursion along diagonals of each Gram block, float64 on one GPU (about 1.4·10^15 flops: minutes) | truncated windows or a random sketch (approximate); 16 CPU cores (about an hour) | exact; ~20x faster |
| Graph storage | each row keeps its 512 nearest tokens (sparse) plus a uniform floor (one rank-one term) | dense N × N walks | at N = 57,344: about 0.5 GB per layer instead of 26 GB; dense would need over 400 GB for walks and inverses |
| Sinkhorn scaling | sparse fixed point | dense Newton | 49 s to 0.19 s at N = 2,000 |
| Layer scales | 9 leading eigenpairs per layer | full eigendecompositions | 197 s to 3-7 s at N = 2,000; full ones are infeasible at 57,344 |
| Coupling t* | bracketed root of one-mode solves | bisection with full solves | 38 s to 5-8 s; t* equal to 4e-14 |
| 24 modes | direct Lanczos on the sparse operator, on the GPU | dense block inversion | 23 s to 2 s at N = 2,000; the dense path scales as N³ (days at 57,344), the sparse one about linearly |
| Correctness | `spacetime.py` copied byte for byte as the dense reference; the sparse path must reproduce it on the same operator | trusting a new solver | eigenvalues and t* to 1e-8, subspaces to sin θ ≤ 1e-6 (section 6) |

The one thing the sparse design changes is the graph rule: keeping 512 neighbours per row is part of its definition.
How far that moves the picture from the dense rule is measured before the main run, and the number of neighbours
doubles if the drift is not small (section 5). Net effect: every token instead of 1 in 7, for about half v1's
projected $47.

## 1. Questions and measured quantities

| # | Question | Measured quantity (section 7) | Could come out |
|---|---|---|---|
| Q1 | Does memory give a picture the two models share beyond what the text alone gives? | remainder share; cross-model agreement on the remainder minus agreement with text-only pictures | memory helps, hurts, a middle length is best, or nothing beats text alone |
| Q2 | Does it matter whether each token counts equally (cosine) or loud tokens count more (dot)? | the same numbers per reading; agreement between readings | readings agree or differ |
| Q3 | Does a text's picture change when other texts are in the graph? (descriptive) | agreement on the text's tokens between graphs built on 1, 4, 16 and 32 texts, including two disjoint sets of companions | stable, or reorganised by the company it keeps |
| Q4 | Which of three configured graph recipes keeps the joint modes spread over many tokens? | worst two-token mass and spread of the first 16 joint modes | the earlier finding against Shannon perplexity repeats on tokens or not |
| Q5 | Over what distances do the models attend directly, and do residuals stay similar? (descriptive) | direct attention distance; residual autocorrelation | reference scales only, not an effective context length |

**Functional readout** (beside the decision, not in it): do tokens close in the picture share the model's next-token
predictions beyond tokens matched on the current token and document?

## 2. Models and tokens

| | Pythia | OLMo |
|---|---|---|
| Hugging Face id, revision | `EleutherAI/pythia-1b`, `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` | `allenai/OLMo-1B-0724-hf`, `d7cbab742d80589e714b1a2d7f838dcd21cbe143` |
| Shape | 16 blocks, width 2,048, parallel attention and MLP | 16 blocks, width 2,048, non-parametric layer norm |
| Context limit | 2,048 | 4,096 |
| End-of-text token | `<|endoftext|>` = id 0 | `<|endoftext|>` = id **50279**; id 0 is `|||IP_ADDRESS|||` |
| Extra tokens | none above 50,276 | 50277 `|||EMAIL_ADDRESS|||`, 50278 `|||PHONE_NUMBER|||` |
| Training data | the Pile (test split held out) | Dolma 1.7 (the Pile test split is not guaranteed held out) |

Ordinary text gets the same ids from both tokenizers, but **the separator differs**: each model reads its own
end-of-text token at position 0, and identity is asserted on positions 1-2,047. Display decoding passes
`clean_up_tokenization_spaces=False` (OLMo's default is `True`).

**Token assertion (U0, fails closed).** (i) Documents containing `<|endoftext|>`, `<|padding|>`,
`|||EMAIL_ADDRESS|||`, `|||PHONE_NUMBER|||`, `|||IP_ADDRESS|||` or NUL are rejected before tokenisation. (ii) Both
tokenizers encode each accepted text to identical ids. (iii) No text id is in {0, 1, 50277, 50278, 50279}. (iv) Every
id in the probe has the same `convert_ids_to_tokens` string under both. A failure stops the run.

**Forward pass (U1).** float32, `attn_implementation="eager"`, TF32 off, one context per batch, `transformers==4.57.6`,
`torch==2.5.1` (if OLMo does not load, the nearest working pair is pinned and recorded). Residuals by forward hooks on
each decoder layer; hooks for layers 1-15 asserted equal to `output_hidden_states`. Hidden state ℓ = 0 is the
embedding output, ℓ = 1..16 the block outputs. Also stored per position: top-20 next-token ids and probabilities
(softmax of the model's own logits over the full vocabulary) and the full-distribution entropy.

**Causal checks (hard).** Two equal-shape inputs: context 0, and context 0 with positions 1,024-2,047 replaced by
context 1's tokens. Residuals at positions 0-1,023 agree to 1e-5 relative at every layer; at position 1,024 they differ
by more than 1e-3 relative at every layer. A truncated 1,024-token run is recorded per layer as a diagnostic only
(eager attention and OLMo's layer norm compute in float32 internally, so `.double()` is not a float64 reference).

## 3. Probe

**Corpus (owner's answer: anything in the Pile is fine, all text shown).** Pile test split,
`monology/pile-uncopyrighted` revision `3be90335b66f24456a5d6659d9c8d208c0357119`, `test.jsonl.zst` (337,771,483
bytes), streamed in file order; `meta.pile_set_name` is the document's **subset**. Pythia never trained on it; OLMo
may have seen similar text.

**Selection.** Eligible: at least 95% ASCII characters, at least 2,048 Pythia tokens, no forbidden strings; shuffled
with `numpy.random.default_rng(0).permutation`. A document's **context** is its first 2,047 tokens, read from the start,
so no document is split. Reject if: whitespace-only tokens exceed 30%; more than 10% of positions u ≥ 32 repeat a
32-gram that occurred earlier in the context; any 32-gram occurs in an already accepted document; the subset already
has 4 accepted texts. Rejection counts by reason and subset are stored. The first 32 accepted are the **texts**, in
acceptance order (random); at least 8 subsets. Four more accepted documents, never among the 32, make the stitched
controls (section 9.2). The page describes the probe as **a capped sample of the beginnings of long, mostly ASCII
documents from the Pile test split**, not as random text in general; the subset table is on the Method tab.

**Input and nodes.** `[that model's end-of-text] + context`, T = 2,048, positions u = 0..2,047; the residual at
u ≥ 1 has read text tokens 1..u. Position 0 is identical in every context, carries the attention sink and is
**excluded from everything**. **Burn-in** b = 256: the nodes of a text are positions 256..2,047 (1,792 tokens). At
memory 64 the weight before position 1 is at most (1 - 1/64)^256 = 1.8% of the total; the whole prefix never reaches
a steady state, which is measured, not hidden.

| Graph | Nodes | N |
|---|---|---|
| **one-text graph** | the 1,792 tokens of one text (32 texts and 4 stitched controls) | 1,792 |
| **all-texts graph** | the 1,792 tokens of each of the 32 texts together | 57,344 |
| **companion graphs** (Q3; texts 0 and 1 only) | text c with 3 others; text c with 15 others (set H1); text c with the other 16 (set H2) | 7,168; 28,672; 30,464 |

Companions are drawn with `default_rng(100 + c)`; the 3 are a subset of H1. The viewer opens on text 0; example figures
use texts 0 and 1.

**Stored** (Volume `tm-exp1`): `probe/probe.json` (subsets, ids, display strings, stitched tables, companion sets,
rejections, assertions); `acts/<model>/c<cc>.npy`, float32 (17, 2048, 2048), 36 contexts, 20.5 GB in all;
`pred/<model>.npz`; `attn/<model>.npz` (section 10).

## 4. Similarity (the kernel family)

**Sites.** L = 8 hidden states ℓ ∈ {2, 4, ..., 16}, the same in both models. **Centring.** μ_ℓ = mean of x_{c,u,ℓ}
over the 32 texts and u = 1..2,047, float64, one per model and site, used by every graph, so the similarity of two
tokens never depends on which graph they are in (one-text graphs are labelled as using a corpus mean).

| Reading | Token vector y_u | Similarity | Distance d given to the graph rule |
|---|---|---|---|
| cosine (each token counts equally) | (x_u - μ)/‖x_u - μ‖ | ĉ = K(t,t')/sqrt(K(t,t)K(t',t')) | 1 - ĉ |
| dot (loud tokens count more) | x_u - μ | K(t,t') | [(K(t,t) + K(t',t'))/2 - K(t,t')] / s |

s = mean of K(t,t) over all nodes (one scalar per model, site and memory), so dot distances are in units of the mean
squared norm and comparable across graphs.

**Memory kernel** (the owner's definition). With G(u,u') = ⟨y_u, y_u'⟩ and y_u = 0 for u ≤ 0,

    K_a(t, t') = Σ_{s ≥ 0} a^s G(t - s, t' - s) = G(t, t') + a K_a(t - 1, t' - 1).

It is the dot product of φ(t) = (a^{s/2} y_{t-s})_{s≥0}: two pasts laid side by side **aligned at their ends** and
compared lag by lag; not the dot product of running means. Facts the primer states:
- For two unrelated noise texts the expected similarity is zero at every pair of positions, but chance agreement
  persists along a diagonal (K(t+1, t'+1) and K(t, t') share history, correlation a at steady state), so graph entries
  are dependent even without real autocorrelation (measured, section 9.1).
- At the whole prefix (a = 1) prefixes of different lengths are compared over the shorter one, and even for noise the
  spread of ĉ depends on position (variance about 1/(d·max(t,t')) for unit tokens in dimension d).
- **What the readings isolate.** With unit tokens K(t,t) = (1 - a^t)/(1 - a), so for memories none, 8 and 64 after
  burn-in the Euclidean distance of unit-token pasts is m(1 - ĉ) to within 2%: there the readings differ only in token
  normalisation. At the whole prefix the dot distance of unit tokens is (t + t')/2 - sqrt(tt')ĉ, so the readings also
  differ in prefix normalisation; whole-prefix comparisons are two combined geometry choices.

**Memory grid.** m = memory in tokens, a = 1 - 1/m: none (a = 0), 8 (a = 0.875), 64 (a = 0.984375), whole (a = 1).
8 constructions = 4 memories × 2 readings. **"Smart EMA" here** is the measurement of section 10; a kernel with
model-derived lag weights is deferred (a general lag profile needs a convolution along diagonals, not this
one-step recursion).

**Text-only adversaries**, on exactly the same nodes with the same readings, memories, graph rule and one site (no
layers): (i) **random token vectors**: a fixed Gaussian vector in dimension 2,048 per token id (`default_rng(7)`),
the same for both models; (ii) **embedding output**: each model's hidden state ℓ = 0. At memory none both are exact
functions of the current token (already in the nuisance span of section 7.3) and are not built.

**Loud tokens.** Kept under the dot reading. Reported per model and site: share of centred squared norm in the top 1%
of tokens, and the 10 most frequent token strings among tokens above 5 times the median norm.

**Exact computation, streamed on one GPU (U2).** First the diagonals K(t,t) of every text by the same recursion on
squared norms. Then for each text c and site: the Gram of its 2,047 tokens against all 65,504 tokens of the 32 texts
(one float64 matmul), the recursion `K[t, 1:] = G[t, 1:] + a * K[t-1, :-1]` within each text's column block, row by
row, for the three nonzero memories at once; conversion to distances with the diagonals; and for each node row its
K_nb nearest nodes (ties by index), both among all 57,344 nodes and among its own text's 1,792. Only these lists are
stored (int32 indices, float32 distances computed in float64); nothing of size N × N is kept. The companion graphs get
their lists from the same row blocks restricted to their texts' columns; the four stitched texts from their own blocks.
U2 then calibrates the participation-ratio widths on the kept lists and makes a second streamed pass for the
discarded mass of section 5. Cost: about 1.76·10^13 flops per site, reading and
model for the Grams, 1.4·10^15 in all with the adversary inputs, minutes on an A100 in float64 (L4 has no usable
float64 rate). Not used: the closed form a^t Σ a^{-u} g(u), which overflows. Tests: recursion against φ and a direct
double loop, 1e-12.

**Checks** (fail closed): no token with ‖x_u - μ‖ < 1e-6 × the median; smallest off-diagonal distance in every graph
above 1e-9 (adversary graphs exempt; their near-ties are reported).

## 5. Graph rule

**Definition (default recipe: participation ratio, k = 64, K_nb = 512).** For node i with kept neighbour set N_i
(its K_nb nearest nodes in the graph), q(j | i) = exp(-β_i (d_ij - min_{j' ∈ N_i} d_ij'))/Z_i for j ∈ N_i, with β_i set
by bisection so that 1/Σ_j q(j | i)² = k (achieved within 1e-3 of k, fail closed). S = (q + q^T)/2. W = S + η(11^T - I)
with η = 10^-4/(N - 1): a uniform floor holding 10^-4 of each row's mass, which keeps every graph connected and is
the single rank-one term. Symmetric Sinkhorn scaling d with d_i (W d)_i = 1 to 1e-11 (sparse fixed point; Newton steps
solved by conjugate gradients if it stalls), P = D W D, lazy walk M = (I + P)/2. In storage M = sparse part + rank-one
part, the form of `opt/sparse_core.py`'s `Walk`. The per-row calibration is `_cond_rows` from DepthEmbedding
`invert/dm.py`, applied to the kept distances (`mask_self=False`; self excluded from N_i).

**Truncation is part of the rule, and its drift is measured before the main run (V3).** (i) Discarded mass: for every
row of the all-texts graph, Σ_{j ∉ N_i} exp(-β_i(d_ij - min))/Z_i, by a second streamed pass; required median ≤ 1e-3
and maximum ≤ 1e-2 at every site. (ii) Against the dense, untruncated, floor-free participation-ratio rule
(`ea_kernel(renyi=2)`) solved with the dense reference: texts 0-3 (N = 1,792) and the 4-text companion graph of text 0
(N = 7,168), Pythia, cosine, memories none and 64: t* and the first 16 eigenvalues within 1% and alignment of the
16-mode pictures (section 7.2) at least 0.99. If either fails, K_nb = 1,024 for every graph and V3 is repeated; if it
fails again, the rule stays as defined and the page states the drift.

**Disconnected texts.** If, without the floor, the kept-neighbour graph of a layer falls into several components
(texts that never appear in each other's 512 nearest), its lowest modes are set by the floor and are document
indicators. The number of components per layer is reported, and such constructions fail the remainder gate of
section 8 through their document share.

**Why participation ratio, against the owner's "Shannon perplexity is perhaps gentler".** It is the same per-point
Gaussian construction with the participation ratio in place of exp(entropy). Before symmetrisation and Sinkhorn, a
participation ratio of k bounds every conditional weight by 1/sqrt(k) = 0.125, whereas a Shannon perplexity of 64
allows about 0.6 on one neighbour; in DepthEmbedding (joint modes, perplexity 400, N = 8,000) Shannon gave modes carried
by two images. The bound does not carry through Sinkhorn, so the shipped joint modes are checked (P2).

**Recipe comparison (P2), on the joint modes the viewer ships.** Recipes on the same kept lists: participation ratio
k = 64; Shannon perplexity 64 (`_cond_rows` with renyi=1); union 15-nearest-neighbours with weight 1, floor 10^-6
(the DepthEmbedding rule). Pythia, cosine reading: the all-texts graph at memory 64 and texts 0-3 at memories none
and 64, 9 graphs × 16 joint modes. Per mode, w_i = Σ_ℓ f_ℓ(i)²/Σ_{ℓ,j} f_ℓ(j)² (depth-aggregated token mass);
**two-token mass** = the largest w_i + w_j; **spread** = 1/Σ_i w_i². Localised if two-token mass > 0.1 or spread < 20.
Reported separately for the all-texts and one-text graphs, with each recipe's effective width (exp of mean row
entropy of P). **Rule:** the recipe with the fewest localised modes over the 144 is used after P2; ties go to
participation ratio, then nearest neighbours. Widths are not matched, so the conclusion is about these configured
recipes only.

## 6. Through-depth operator and solvers

The operator is DepthEmbedding's: L_t = blockdiag((I - M_ℓ)/c_ℓ) + t (D_path ⊗ I), solved on functions whose slice
at every layer has mean zero. `spacetime.py`, `spacetime_stage.py` and `test_spacetime.py` are copied from commit
`8dd7a66` unchanged and their tests must pass unchanged.

| Choice | Value |
|---|---|
| Sites | the 8 hidden states in depth order, path with free ends, the same tokens at every layer |
| Units | relative: c_ℓ = mean of the first 8 nontrivial 1 - λ_k of M_ℓ |
| Coupling | t = t* only: the coupling at which the first between-layer eigenvalue equals the first eigenvalue on slice-mean-zero functions (it equates two eigenvalues, not two energy terms) |
| Modes | 24 computed, 20 checked, 16 read; readings also at K = 12 and 20 (section 8) |
| Precision | float64 |

**Solvers.** One-text graphs (N = 1,792) and the dense checks: the copied `spacetime.py` on the densified walk
(`modes`, `balance_direct` with the full site eigensystem passed in, `site_scales('own')`). All-texts and companion
graphs: the sparse path of `opt/sparse_core.py`, copied with its arithmetic unchanged and its array calls made
switchable between numpy/scipy and CuPy (`cupyx.scipy.sparse`), run on the GPU: `site_spectra` (9 leading eigenpairs
per layer) for c_ℓ, `balance_root` (bracketed t*), `normal_modes` (direct Lanczos on the sparse operator with the
between-layer-only sector pushed above the target) for 24 modes, `ritz` for the values. If Lanczos needs more than
20,000 operator applications for 24 modes, `power_modes` (the polynomial filter of the report) is used instead and
recorded.

**The dense reference must agree with the sparse path on the same operator (V1, V2; hard).** V1: the synthetic
all-texts graph (8 texts × 768 tokens × 4 layers). V2: the real one-text graph of text 0 (Pythia, cosine, memory 64),
and the 4-text companion graph of text 0. Required: t* and the first 20 eigenvalues to 1e-8 relative; for each
cluster of eigenvalues closer than 1e-8, the largest principal-angle sine to 1e-6; CPU (scipy) and GPU (CuPy) sparse
paths both checked.

**Acceptance per solve:** residuals of the first 20 modes below 1e-8; at t*, |λ_1/(t*·4 sin²(π/16)) - 1| < 1e-6;
occupancy and recurrent strengths sum to K; finite outputs; Sinkhorn row error below 1e-10. Stored: relative gaps
(λ_{K+1} - λ_K)/λ_K at K = 12, 16, 20; a small gap at 16 means the 16-dimensional space itself may be unstable, which
is why the gates of section 8 are also checked at 12 and 20.

## 7. Readings

### 7.1 The picture as a matrix on tokens

G_ℓ = the N × K matrix of the first K modes' slices at site ℓ (K = 16 unless stated); X = [G_1 ... G_8] (N × 8K).
**R** = X X^T = Σ_ℓ G_ℓ G_ℓ^T (trace K) combines token functions shared by several modes or layers and weights each by
how much of the picture uses it (four depth envelopes of one function give 4uu^T). R is never formed: every reading
below uses the factor X. Everything is unchanged by rotating the K modes. **Effective rank** r_eff = number of
eigenvalues of R above 1% of the largest; U_eff their eigenvectors.

### 7.2 Agreement

A(R_1, R_2) = tr(R_1 R_2)/sqrt(tr(R_1²) tr(R_2²)) = ‖X_1^T X_2‖²/(‖X_1^T X_1‖ ‖X_2^T X_2‖) (linear kernel alignment;
the mean cos² of principal angles for equal-weight projectors). After restriction to a subset of tokens (one text of
the all-texts graph), both factors are centred on that subset first and the retained trace is reported; below 5%
retained, the value is "no signal". **Null:** 200 permutations of one picture's token labels (within texts for
multi-text graphs), mean and 95th percentile. **Document bootstrap** (all-texts graph): 200 draws of the 32 texts with
replacement on the fixed graphs, 5th-95th percentiles, labelled as conditional on these graphs.

### 7.3 What the picture is made of

Nuisance span Z, orthonormalised by QR in this order on mean-zero functions:

| Family | All-texts graph | One-text graph |
|---|---|---|
| document | the 32 text indicators, centred: 31 dimensions | none |
| position | p_j(t) = cos(π j (t - 255.5)/1,792), j = 1..16, the same function of position in every text, residualised against documents | the same 16 functions (position and slow drift of topic cannot be told apart; labelled so) |
| common token | indicators of ids seen at least 20 times among the nodes, residualised against the above | the same within the text |

Shares s_doc = tr(Q_doc^T R Q_doc)/K, s_pos, s_tok likewise; **remainder share** s_rem = 1 - s_doc - s_pos - s_tok;
chance = each family's dimension over N - 1. **Remainder picture** R̃ from X̃ = (I - Π_Z) X; its effective rank is
(tr R̃)²/tr(R̃²). The remainder is "outside these columns", not "beyond the obvious": previous tokens, formatting,
rare tokens and fast position patterns remain, which is what the adversaries are for. Per-mode shares (viewer) use
one mode's unit slice at one site.

### 7.4 Other readings

| Reading | Definition |
|---|---|
| spread, two-token mass | section 5, per mode; localised if spread < 20 or mass > 0.1 |
| captured share of a grouping F | ‖U_eff^T Q_F‖²/r_F for orthonormal centred indicators Q_F (r_F columns); chance r_eff/(N - 1) |
| same-document walk share | per token and site, Σ_{j in its text, j ≠ i} P_ℓ[i,j]; chance 1,791/57,343 = 0.031 |
| cross-model agreement | A(R̃_Pythia, R̃_OLMo), same tokens and Z |
| text agreement | A(R̃_model, R̃_adversary), each model and adversary |
| cross-reading agreement | A(R_cosine, R_dot), same model and memory |
| one-text vs all-texts, text c | A between the one-text picture and the all-texts picture restricted to c (centred), on all 1,792 tokens, with the retained trace |
| companion series, texts 0 and 1 | the same between the graphs on 1, 4, 16 (H1), 16 (H2) and 32 texts; **H1 vs H2** compares text c's picture in two graphs whose other texts are disjoint |
| realised radius | per token, Σ_j q(j|i) d_ij (the row's mean distance), median over text c in each graph: a fixed k is not a fixed radius |
| prediction overlap | o(i,j) = Σ_{v in both top-20 sets} min(p_i(v), p_j(v)), a lower bound on the overlap of two next-token distributions |

**Functional readout.** Picture distance D(i,j) = ‖X_i - X_j‖². For each token i its 10 nearest tokens j by D; for
each j a control j' (`default_rng(11)`) with the same current token id as j and the same same-text relation to i.
Readout = mean of o(i,j) - o(i,j'), each model with its own predictions and picture; pairs without a control are
skipped and the coverage reported; document bootstrap. Not in the decision: the next-token distribution is read from
the last layer alone, so as a criterion it would favour pictures weighted toward late layers. It is shown beside the
decision and the E2 answer states its sign.

## 8. Choosing the construction the viewer ships

Applied by `decide.py` inside U7, recomputed by the audit (U8), shown on the page with every number. A construction
is **supported** if all of these hold for both models:

1. **Numerical:** every solve passed acceptance.
2. **Not mostly nuisance:** s_rem ≥ 0.25 in the all-texts graph and in the median one-text graph; remainder effective
   rank ≥ 4.
3. **Shared beyond text:** cross-model agreement minus the largest text agreement (over both models and both
   adversaries; 0 at memory none, where the adversaries are the removed common-token span) is at least 0.05, in the
   all-texts graph and in the median one-text graph.
4. **Above chance:** cross-model agreement above its null 95th percentile.
5. **Not a cutoff artefact:** conditions 2-4 also hold with K = 12 and K = 20.

**Score** = mean over (all-texts, median one-text) of the margin in 3; highest supported score wins; a runner-up within
0.05 is broken toward the cosine reading, then the shorter memory. Reported, not used: the paired difference from the
same reading at memory none (document bootstrap, conditional) and the functional readout.

**No supported winner** is a stated outcome: the viewer then ships the **reference construction**, fixed now as cosine
reading with memory 64 (counterpart: memory none), labelled "reference, not supported by the criteria", and E2 says
so. With a winner, its counterpart is memory none (or, if the winner is none, the best supported construction with
memory and the same reading, else cosine 64). The score measures shared descriptive geometry beyond a text-only
picture; usefulness for interpretation is only what the functional readout shows.

## 9. Controls

### 9.1 Synthetic (pipeline), run with P2's recipe, before any main unit

Same code path end to end, sparse path included. 8 texts of 1,024 positions (position 0 excluded), burn-in 256 (768
tokens per text, all-texts N = 6,144), width 64, 4 sites, seeds 1-5; harmonics cos(π j (t - 255.5)/768). At position u
of text c and site ℓ,

    x = α_ℓ r_{z(u)} + β_ℓ e_{w(u)} + γ g(u) + ε,

z(u) ∈ {1..4} a hidden regime (Markov chain per text, switching probability 1/128; r_z random unit vectors shared
across texts; α_ℓ = α·(0, 1, 1, 1)); w(u) a token id, independent Zipf(1.1) on 500 ids, e_w random unit vectors,
β_ℓ = β·(1, 0.7, 0.5, 0.3); g(u) = cos(πu/1,024) g_a + sin(πu/1,024) g_b, the same in every text; ε ~ N(0, I/64).

| Variant | α, β, γ | Purpose |
|---|---|---|
| strong | 1, 1, 0 | exact positive |
| planted | 0.25, 1, 0 | power at realistic strength (per-token same-regime cosine about 0.03 against a spread of 0.125) |
| drift | 0.25, 1, 0.5 | position share |
| noise, fields 1 and 2 | 0, 0, 0 | two independently drawn noise fields on the same tokens |

**Hard checks (stop before the main units).** H1: strong, memories 8 and 64, cosine: captured share of the regime span
≥ 0.99 in ≥ 4 of 5 seeds. H2: noise field 1 against field 2, every memory: agreement of their all-texts pictures below
the null 99th percentile in ≥ 4 of 5 seeds. H3: relabelling all tokens by one permutation permutes the picture exactly
(1e-8). V1 (section 6) runs on the same data.

**Measurements (reported).** Power: planted, captured share of the regimes against memory, beside a regime grouping
from an independent Markov chain with the same persistence (95th percentile over 50 draws). Leakage through depth: at
site 0, where regimes are absent, the captured share of the site-0 walk alone against the joint picture's site-0 slice
(coupling is designed to carry structure there; the difference is the leakage). Token dominance: s_tok by memory.
Position: s_pos by memory in planted and drift, and in noise field 1 the share of each token's 10 nearest within ±m
positions and the spread of ĉ by position quartile (the flat-norm white-noise check).

### 9.2 Stitched real texts (expected answers, not gated)

Two source pairs (A, B), (C, D), each from different subsets. Each pair makes two texts with alternating segments of
random lengths (seeded): **long**, uniform in [384, 640]; **short**, uniform in [48, 80]; ids concatenated, never
re-tokenised across a join. One-text graphs only. Measured for every construction and both models: captured share of
the A/B indicator; the **position-only baseline** (share of the indicator inside the 16 harmonics; about 0.9 for a few
long segments); and the **share beyond position** (indicator residualised against the harmonics, then captured).
Expected: share beyond position above its null at none and 8 in all four texts, lower at 64 than at 8 for the short
schedule; the whole prefix is reported without a prediction.

## 10. Direct attention distance and residual autocorrelation (Q5)

From U1's attention probabilities, per model, layer and head, over query positions 256..2,047 of the 32 texts: sink
mass A[i,0] and mean direct attention distance Σ_{j≥1} A[i,j](i - j)/Σ_{j≥1} A[i,j]; per layer median and interquartile
range over heads. From residuals, per model and site: ρ(δ) = mean cosine of centred unit vectors δ apart within a text,
δ = 1..1,024, and the background ρ_∞ between tokens of different texts; the half-similarity lag is the smallest δ with
ρ(δ) - ρ_∞ ≤ (ρ(1) - ρ_∞)/2, reported as censored ("> 1,024") with a count of texts when there is no crossing. Neither
is an effective context length (direct attention ignores routes through values, MLPs and earlier layers; residual
similarity can persist through shared vocabulary or topic); the page draws them as reference scales beside the
memory grid.

## 11. Compute plan

Modal app `tm-exp1`, Volume `tm-exp1`, per the `modal` skill (detached coordinator, one worker per unit). Rates on
2026-11-23: A100-80GB $0.000694/s, L4 $0.000164/s, CPU $0.0000131 per physical core-second, memory $0.00000222 per
GiB-second; volumes free to 1 TiB. The VPS runs tests and builds the site. Estimates below are from flop counts and
the report's measurements scaled to N; the pilot replaces them with measurements.

| Unit | Count | Hardware | Work | Each | Spend |
|---|---|---|---|---|---|
| U0 probe | 1 | 2 cores | download, filter, tokenise both ways, assert | 20 min | $0.1 |
| U1 extract | 2 (model) | L4, 4 cores, 32 GiB | 36 forward passes, hooks, predictions, attention statistics, causal checks | 25 min | $0.5 |
| U2 kernel lists | 2 (model) | A100-80GB, 8 cores, 64 GiB | diagonals, streamed Grams, memory recursion, K_nb lists for all graphs, second pass for discarded mass | 25 min | $2.4 |
| P1 timing pilot | 2 | as U3 and U4 | one all-texts unit and one one-text group, Pythia, cosine, memory 64: stage timings, peak memory, billed cost | 45 min cap | $1 |
| V (checks) | 1 | 16 cores, 96 GiB + A100 for 15 min | V1-V3 dense reference comparisons | 60 min | $2 |
| P2 recipes | 1 | A100-80GB | Shannon and nearest-neighbour graphs and joint solves (participation ratio reuses P1) | 20 min | $1 |
| U5 synthetic | 1 GPU + 30 CPU | A100; 8 cores | 5 variants × 5 seeds; all-texts graphs on the GPU, one-text graphs dense | 40 min; 3 min | $2.2 |
| U3 all-texts | 16 (model × reading × memory) | A100-80GB, 8 cores, 64 GiB | 8 walks, sparse Sinkhorn, site spectra, t*, 24 modes, adversary walks, readings, viewer neighbours | 12 min | $6 |
| U3c companions | 8 (model × text × construction pair) | A100-80GB | 3 companion graphs × 2 constructions (cosine none and 64) | 15 min | $3 |
| U4 one-text | 24 (model × reading × group of 6 texts) | 8 cores, 32 GiB | 4 memories × (8 walks + dense solve), adversaries | 20 min | $5 |
| U7 analysis | 1 | 16 cores, 64 GiB | readings, nulls, bootstraps, readout, decision, packaging | 60 min | $1.3 |
| U8 audit | 1 | 16 cores, 64 GiB | independent checker (section 12.4) | 60 min | $1.3 |

**Projected spend about $26; budget $40.** The largest uncertainty is the number of Lanczos applications at
N = 57,344 × 8 (estimated 20,000-50,000 at about 3 ms each on an A100, from 8 layers × about 40 million stored
entries per layer); P1 measures it. Order: U0 → U1 → U2 → P1 ‖ V ‖ P2 → budget check → U5 (P2's recipe) → hard checks
H1-H3, V1-V3 → U3 ‖ U3c ‖ U4 → U7 → U8 → site. Wall-clock about 4 hours, mostly waiting between gates.

**Budget check after P1 (fixed now).** Projected total = spent so far + measured unit costs × remaining unit counts
+ 20%. If above $40: (i) switch the all-texts solves to `power_modes`; (ii) K_nb stays at 512 even if V3 asked for
1,024, and the page states the measured drift; (iii) if still above, stop and report the measured numbers.

**Volume layout.** `/hf`, `/probe`, `/acts/<model>`, `/pred`, `/attn`, `/nb/<model>/<reading>/<input>_<memory>.npz`
(kept lists, about 36 GB in all), `/pilot`, `/checks`, `/synthetic`, `/all/<model>/<reading>_<memory>.npz` (float32
slices (8, 57,344, 24), values, site spectra, c_ℓ, t*, gaps, acceptance, components, per-token same-text walk shares,
top-10 neighbours per token and site, adversary modes, worked-token rows), `/one/<model>/<reading>/c<cc>.npz`,
`/companions`, `/analysis`, `/audit`, `/site`. Only `/analysis`, `/audit` and `/site` are fetched. Run ids, spend and
status go in the README.

## 12. The site

Public repo `CharlesR-W/TransformerModes`, GitHub Pages; machinery copied from DepthEmbedding (`build_site.py`,
`xsite/xlib.py` facts and `{{fact:...}}` tokens, `check_fragment.py`, `check_browser.py` adapted, live charts, palette,
CSS; not the CIFAR embedding figure). Tabs: **Viewer**, **Experiments**, **Method**; a one-line banner on each (both
models, 16 blocks, width 2,048, the 8 sites marked).

### 12.1 Viewer

**Main view** (desktop: text about 65% of the width, side column right; phone: one column, no horizontal scroll).

| Control | Values | Why |
|---|---|---|
| Text | 32 texts (subset, first words) + 4 stitched controls | the text is the object read |
| Graph | one-text graph / all-texts graph (all-texts disabled for stitched texts) | the owner asked for both |
| Model | Pythia / OLMo | the two models are the replication; tokens align |
| Memory | shipped construction / its counterpart (section 8) | local versus memory is the point of the kernel |
| Layer | 8 stops, 2-16 | each mode has a slice at every layer |
| Mode | 1-16 | which of the picture's functions colours the text |

A caption line under the controls names the reading, recipe and coupling, and whether the construction is the
supported winner or the reference.
- **Text.** Tokens inline with whitespace kept; every token after burn-in coloured by the mode's unit slice at the
  layer (diverging palette, range ±99th percentile of |value| over the mode's layers, fixed across layers; the legend
  says so); burn-in tokens grey. If the mode's depth weight at this layer is under 1%, the text is faded with a note.
  Stitched texts show joins and A/B labels.
- **Mode card.** Depth weight over 8 layers (bars), the four shares as one stacked bar with chance ticks, spread.
  Whenever Model, Memory or Graph changes: "mode numbers and signs are local to this graph" and the alignment between
  the picture just left and the new one on this text's 1,792 tokens (precomputed), so a reader sees whether the change
  is real before comparing colours.
- **Token card** (click a token). Token, position, value; its 10 nearest tokens under the similarity at this layer
  (ĉ for cosine, scaled d for dot; text and position; in the all-texts graph from any text), each with the last up to 24
  tokens ending there, right-aligned under the clicked token's own, and the share of memory weight those 24 tokens
  carry (31.5% at memory 64; 1.2% of a whole prefix at the end of a text); in the all-texts graph, the token's
  same-text walk share.

**One disclosure, "Details":** a strip chart of the value against position (120 px), a scatter of mode 1 against the
selected mode, and the **worked token**: for 3 prespecified tokens in each of texts 0 and 1 (`default_rng(13)`), the
10 nearest by raw similarity beside the 10 largest walk weights P_ℓ[i,j], and the local eigen-equation at this layer
evaluated in the browser from the stored sparse row, the floor term and float32 slices:
(f_ℓ(i) - Σ_j M_ℓ[i,j] f_ℓ(j))/c_ℓ + t(deg_ℓ f_ℓ(i) - f_{ℓ-1}(i) - f_{ℓ+1}(i)) against λ f_ℓ(i), residual printed. The
state is kept in the URL hash; the primer's open state in `localStorage` (try/catch).

**Data (lazy per text).** `d/manifest.json` (60 KB); `d/text/c<cc>.json` (ids, display strings, 50 KB each);
`d/modes/<model>/<mem>/<graph>/c<cc>.i8` (int8 unit slices (8, 16, 1,792) with float32 scales, 230 KB each; 272 files,
63 MB); `d/info/...json` (eigenvalues, depth weights, shares, spreads, toggle alignments, 15 KB each); `d/nb/...`
(top-10 neighbours per token and site, uint16 index within the graph and float16 value, scaled so values are of order
one, 570 KB each, 155 MB); `d/worked/...` (sparse rows and slices of the 6 worked tokens, 3 MB). About 230 MB in all;
opening a text fetches about 0.5 MB per graph-model-memory choice, a click 0.6 MB.

### 12.2 Primer (top of the Viewer tab, collapsible, about one screen)

**Position** (token number; the first 255 are burn-in). **Layer** (which block's output; 8 of 16). **Memory** (a
drawing of two pasts aligned at their right ends with fading weights; chance agreements persist along the alignment;
the whole prefix compares pasts of different lengths, so it depends on position even for structureless text).
**Readings** (for memories up to 64 they differ only in whether tokens are normalised). **The two graphs** (the
one-text graph links tokens of one text; the all-texts graph links all tokens of the 32 texts, so a token's neighbours
may come from other texts; Sinkhorn scaling gives every token the same total link weight, it does not make one text
behave like the whole). **Mode** (a pattern over tokens and layers that changes little between similar tokens and
between neighbouring layers; numbered from the smoothest; numbers and signs are local to one graph). **Coupling**
(fixed where the first between-layer and first within-layer eigenvalues meet). **Share bar** (which text, position,
common token, the rest; chance marks; the rest still contains previous tokens and formatting). **Checking** (click a
token; open Details for a worked token).

### 12.3 Experiments tab entries

Each follows `_template.html` (question, scope, 2-3 sentence answer with one bold sentence, lead figure, 3-5 tiles,
"What was computed" table, at most two supporting figures, at most six caveats, closed `<details>`); numbers are
generated facts.

| Entry | Lead figure: x · y · facets · uncertainty | The sentence a reader should be able to say |
|---|---|---|
| E1 Does the pipeline recover what it should, and only that? | x = memory · y = captured share of the regime span · series: strong (thin), planted (solid), same-persistence reference (dashed grey) · band: min-max over 5 seeds · chance line. Supporting: stitched texts, x = memory, y = share beyond position, long and short schedules, both models, null line | "A strong planted regime is recovered exactly, independent noise fields share nothing, the realistic regime appears with memory [as measured], and on real stitched texts the switch between documents is [visible / not visible] beyond position." |
| E2 Which similarity, if any, gives a picture both models share beyond the text? | two panels sharing rows; rows = 8 constructions · left x = remainder share, Pythia circle, OLMo diamond, gate 0.25 · right x = cross-model agreement (filled) and largest text agreement (hollow), whiskers = document bootstrap (conditional) · winner or reference highlighted, failing rows grey with the failed condition named. Supporting: stacked shares; functional readout with intervals | "[Winner] is supported: the models agree on its remainder at [x], [y] above the best text-only picture, and close tokens share predictions [z] beyond matched controls", or "No construction beats the text-only pictures by 0.05; the viewer shows the reference construction." |
| E3 Does a text's picture change when other texts are in the graph? | x = number of texts in the graph (1, 4, 16, 32) · y = alignment with the 32-text picture on the text's tokens · series: texts 0 and 1 × memory none and 64 (hue = memory) · facets: model · H2 as a second marker at 16; retained trace and realised radius in a strip below. Supporting: one-text vs all-texts alignment for all 32 texts by memory (band 10-90%) | "Text [c]'s picture with 15 other texts matches its picture with a disjoint 16 at [x], and its own one-text picture matches the all-texts one at [y]: [the company matters / little]." |
| E4 Which graph recipe keeps the joint modes spread out? | x = recipe, grouped all-texts / one-text · y = localised modes among the first 16 joint modes · effective width above each bar | "[Recipe] gives [n] localised joint modes against [n] and [n]; it is used. This compares three configured recipes, not the rules in general." |
| E5 How far do the models attend directly, and how long do residuals stay similar? | x = layer · y = tokens (log) · series: median direct attention distance (solid), residual half-similarity lag (dashed, censored marked), hue = model · reference lines at 8 and 64 · band: interquartile range over heads | "Direct attention reaches about [x] tokens and residual similarity halves after about [y]: reference scales beside the memory grid, not effective context lengths." |

Tiles: E1 H1-H3 and V1-V3; E2 outcome, margin, remainder share, readout; E3 H1-H2 alignment, retained trace,
same-text walk share against 0.031; E4 the three counts; E5 sink mass. E2's "More figures": cross-reading agreement,
layer-matched agreement, cutoff checks at 12 and 20, loud-token statistics, components per layer. The Method tab
carries sections 0 and 2-10 in page form, the subset table, decisions and deferred items.

### 12.4 How a reader can check there is nothing odd going on

- **The dense reference.** V1-V2 show the sparse solver reproducing the byte-for-byte copied dense code on the same
  operator; V3 shows how far the 512-neighbour rule is from the dense rule. Both are tiles in E1.
- **Controls** on the page; the stitched texts in the viewer with joins drawn.
- **Every mode** shows its shares with chance marks and its spread; localised modes are flagged.
- **Clicking a token** lists, as text, the tokens most similar to it and the ends of the pasts compared, with the
  fraction of the memory shown; Details shows raw neighbours beside actual walk weights and the eigen-equation
  evaluated in the browser.
- **The decision table** shows all 8 constructions with every condition.
- **Audit (U8)**, beside the raw arrays on Modal, numpy only and importing nothing from the pipeline: recomputes every
  decision input for all 8 constructions and both adversaries from the float32 slices and the probe tables (same
  permutation and bootstrap seeds), then the ranking and outcome, to 1e-6; for 50 query tokens per graph (the 16
  all-texts graphs, one-text graphs of texts 0-3) recomputes distances to **all** tokens of the graph from the
  activations by the explicit lag sum and checks stored K_nb and top-10 membership and values. The site build fails if
  its report is missing or failing.
- `check_browser.py` drives headless Chrome over every control, a token click, Details and both viewports, failing on
  console errors or external requests; screenshots are inspected before deploying.

## 13. Code and tests

New: `probe.py`, `extract.py`, `kernel_gpu.py` (diagonals, streamed Grams, recursion, K_nb lists, discarded mass),
`graph.py` (the rule of section 5 on kept lists; sparse Sinkhorn; walk as sparse plus rank one), `sparse_solve.py`
(from `opt/sparse_core.py`, arithmetic unchanged, array module switchable), `kernels.py` (from `invert/dm.py` at
`8dd7a66`, unchanged: `_cond_rows`, `ea_kernel`, `ea_oos_rows`, `sinkhorn_sym`, `knn_kernel`, `perplexity`),
`stage.py`, `readings.py`, `synthetic.py`, `decide.py`, `audit.py` (independent), `modal_app.py` (from the `modal`
skill), `test_tm.py`, `test_kernels.py` (masked-row and duplicate-bound tests from `invert/test_dm.py`). Copied
unchanged: `spacetime.py`, `spacetime_stage.py`, `test_spacetime.py`. `test_tm.py` covers: the recursion against φ
and a double loop, a ∈ {0, 0.5, 1}; GPU lists equal to a dense CPU computation on a small input; the unit-token
identity of section 4; the rule with K_nb = N - 1 and η = 0 equal to `ea_kernel(renyi=2)`; sparse-plus-rank-one walk
equal to its dense form; sparse solver equal to `spacetime.py` (small V1); shares summing to 1, (1, 0, 0, 0) for a
centred document indicator and (0, 1, 0, 0) for a document-residualised harmonic; A(R, R) = 1 and rotation invariance;
equivariance H3; the token assertion on two short texts. All run on the VPS in under ten minutes (CuPy paths on
Modal).

## 14. Decisions made on the owner's behalf

| # | Decision | Chosen | Alternative not taken |
|---|---|---|---|
| 1 | Context | first 2,047 tokens of a document of at least 2,048 | shorter contexts; windows inside documents |
| 2 | Texts | 32, at most 4 per subset, burn-in 256 | uniform among long documents (favours papers, law, code) |
| 3 | Nodes | every token after burn-in, in both graphs | sampling (v1) |
| 4 | Sites | 8 even layers | all 16 |
| 5 | Memory grid | none, 8, 64, whole | lengths derived from the models (deferred) |
| 6 | Readings | cosine of unit-token pasts; scaled Euclidean of raw-token pasts | mixed readings |
| 7 | Centring | one corpus mean per model and site for every graph | per graph, per text, per position |
| 8 | Graph rule | participation ratio k = 64 on 512 kept neighbours, floor 10^-4 of row mass, unless P2 or V3 says otherwise | dense rule (infeasible at 57,344); Shannon perplexity; k-NN |
| 9 | Coupling | t* only | DepthEmbedding's six-stop slider |
| 10 | Nuisance span | texts, 16 global position harmonics, ids seen ≥ 20 times | all token identities; other numbers of harmonics |
| 11 | Adversaries | random token vectors; each model's embedding output | formatting-only or n-gram baselines |
| 12 | Decision | margin over text agreement ≥ 0.05 with gates; fixed reference if none | resampling stability as a gate (no valid null) |
| 13 | Functional readout | beside the decision | inside the rule (favours late layers) |
| 14 | Q3 design | companions for texts 0 and 1 at 1, 4, 16 (two disjoint halves), 32 texts, two constructions | all texts and constructions (cost); split halves without shared tokens (nothing to compare on) |
| 15 | Precision | float32 forward; float64 kernels and solvers; float32 storage of distances and slices | float64 forward |

## 15. Deferred

Model-derived lag weights (the "smart EMA" kernel); the recurrent random sketch and the leave-one-text-out Nystrom
comparison (`ea_oos_rows` is copied for it); attention-weighted links between positions across layers; a
training-checkpoint axis and the model as a coupled axis; coupling robustness, the coupling slider, native units and
the Trower et al. matrix; a participation-ratio width sweep; graph-specific or position-wise centring;
cross-text-only neighbours; mixed readings; clipping loud tokens; k-means set readouts; longer contexts (OLMo reads
4,096); more models. **Owner's TODO, explicitly not for now:** (a) a baseline from a sentence-embedding model
(SentenceTransformers) or a vector-database style embedding of the same contexts, "whatever the smart version of that
is"; (b) the Hankel predictive-state construction "like from LingenLearning" (recorded as dictated; reference to be
confirmed).

## 16. Departures from the brief

- **Separator and token identity.** The end-of-text token is id 0 in Pythia and 50279 in OLMo, where id 0 is a PII
  placeholder; each model reads its own separator and identity is asserted on positions 1-2,047.
- **Every token is a node** (N = 57,344 in the all-texts graph), not 512 sampled per text; this needs the sparse graph
  rule and solver of sections 5-6 and is cheaper than the sampled dense design.
- **No CPU kernel computation**; the kernel runs streamed on one GPU in float64; nothing N × N is stored.
- **Storage.** All 17 hidden states in float32.
- **Centring settled** as one corpus mean for every graph.
- **Same-text dominance** is measured (same-text walk share, components, document share, E3), not corrected.
- **"Smart EMA"** is a descriptive measurement here.
- **Comparisons** use alignment of R through its factor, with centring after restriction and a no-signal rule.
- **Viewer** keeps two two-way toggles (Graph, Memory), each an explicit request, with detail behind one disclosure.
- **Brief correction:** identical tokens give identical vectors only at the embedding output, not at early layers.

## 17. Response to review 1

| # | Objection | Response | Where |
|---|---|---|---|
| 1 | Noise control tests the wrong null | Accepted. Two independently drawn noise fields must not agree (hard H2); the A/B resample design and its stability gate are gone (every token is now a node, so there is no node resample); regime power against independent labels of the same persistence; synthetic runs use P2's recipe | 9.1, 8 |
| 2 | Winner rule can select an obvious picture | Accepted. Text-only adversaries (random token vectors, embedding output) with the same memory; the margin over them is the score; remainder effective-rank gate; "no supported winner" with a fixed reference construction; bootstrap labelled conditional; functional next-token readout reported beside; claim stated as shared descriptive geometry | 4, 7.3, 7.4, 8 |
| 3 | A/B is not a ceiling; Q3 cannot show unbiased sampling | Accepted; superseded by the every-token design. Q3 is now "does a text's picture change with its company": companion graphs on 1, 4, 16 and 32 texts, two disjoint 16-text companions (H1 vs H2), retained trace and a no-signal rule, realised radius reported. Generalisation to a new text awaits the leave-one-text-out test | 3, 7.2, 7.4, E3 |
| 4 | Cost estimate has no margin | Accepted, and the numerics replaced: sparse walks with 512 kept neighbours and a rank-one floor, sparse Sinkhorn, partial spectra, bracketed t*, GPU Lanczos (`opt/sparse_core.py`), validated against the copied dense code (V1-V2) with the rule's drift measured (V3). Capped timing pilot with a fixed budget check; projected $26, budget $40 | 0, 5, 6, 11 |
| 5 | Primer noise claim false; readings not isolated | Accepted, with an exact refinement: up to memory 64 the readings differ only in token normalisation (to 2%); at the whole prefix they are two combined choices. Diagonal dependence stated; white-noise position check added; "any lag profile at no cost" removed | 4, 9.1, 12.2 |
| 6 | Targets mix invariants and predictions; S512 passes on position | Accepted. Hard checks reduced to an exact positive, independent noise and equivariance; the rest reported; site-0 leakage shown as site-only against coupled; stitches reuse each source pair across random-length schedules, with a position-only baseline and a share beyond position; no prediction at whole prefix | 9.1, 9.2 |
| 7 | Cutoff and token-space errors | Accepted. Gaps and gates at K = 12, 16, 20; R described as weighted; captured share on a declared effective rank; chance dimensions 31 and 0; document-residualised harmonics; t* described as equating eigenvalues | 6, 7, 12.2 |
| 8 | Recipe pilot does not test shipped joint modes | Accepted. P2 compares recipes on the first 16 joint modes (depth-aggregated two-token mass and spread), all-texts and one-text reported apart, no tolerance margin, conclusion limited to configured recipes | 5 |
| 9 | Checker and click-to-verify not implementable | Accepted. U8 audit beside the raw arrays recomputes every decision input and the ranking and checks neighbour lists against all tokens for sampled queries; per-token walk shares stored; worked tokens with walk weights and an in-browser eigen-equation check; memory fraction shown; dot values scaled before float16 | 11, 12.1, 12.4 |
| 10 | Mode numbers invite false comparisons | Accepted. "Mode numbers and signs are local" plus the alignment between pictures on every toggle; strip, scatter and worked token behind one disclosure; all-texts graph disabled for stitched texts | 12.1 |
| 11 | Q5 overreaches; causal check overstated | Accepted. E5 renamed and described as reference scales with background and censoring; causal check by equal-shape inputs and a current-token check; truncated run kept as a diagnostic | 2, 10 |
| 12 | Publishing long excerpts | Answered by the owner: anything in the Pile is fine and all text is shown. Kept from the objection: the probe is described as a capped sample of the beginnings of long, mostly ASCII documents, not as natural text in general | 3 |
