# TransformerModes, experiment 1: memory kernels on two language models, through depth

Spec v1, 2026-11-23. Exploratory. Written from `BRIEF.md` (owner's request and answers fixed there); sources read:
DepthEmbedding `SPEC_SPACETIME.md`, `spacetime.py`, `spacetime_stage.py`, `test_spacetime.py`, `invert/NOTES.md`,
`invert/dm.py`, `opt/REPORT.md`, `site/xsite/AUTHORING.md`; `deprecated/lm_coherent/SPEC.md` and the Mamba READMEs.
Facts about the models and the corpus were checked against huggingface.co on 2026-11-23 (section 2).

**In one paragraph.** We take 32 natural documents, run Pythia-1b and OLMo-1B on the first 2,047 tokens of each, and
store every residual. A token's similarity to another token compares their recent pasts aligned at the end, with a
memory length m (none, 8 tokens, 64 tokens, the whole prefix) and two readings (every token counts equally, or
loud tokens count more). Each of these 8 similarities is turned into a graph on tokens, one graph per layer, for one
text alone and for all 32 texts pooled; DepthEmbedding's through-depth operator ties the layers together and its 16
lowest modes are the picture. Pre-stated label-free criteria pick one similarity for the viewer, which shows the text
itself coloured by a mode. Projected spend about $50 on Modal, about 4.5 hours of compute.

## 1. Questions and measured quantities

Each question can come out either way; section 7 defines every quantity named here.

| # | Question | Measured quantity | Could come out |
|---|---|---|---|
| Q1 | Does memory make the token picture less about the obvious and more shared between two different models? | for each memory m: remainder share of the 16-mode picture; agreement between Pythia and OLMo on that remainder | memory helps, hurts, or a middle length is best |
| Q2 | Does it matter whether each token counts equally (cosine) or loud tokens count more (dot)? | the same two numbers for both readings; agreement between the readings | readings agree, or dot is dominated by a few loud tokens |
| Q3 | Is one text's own picture what the pooled picture shows on that text? | agreement on shared tokens, against the agreement of two resampled pooled graphs | close to that ceiling (the text behaves like a sample of the pool) or far below it |
| Q4 | Which graph rule avoids modes carried by a handful of tokens? | number of localised modes per rule | the prior finding against Shannon perplexity repeats on tokens or not |
| Q5 | How far back do the two models look, layer by layer? (descriptive) | attention distance; lag at which residual similarity halves | sets the scale for a model-derived memory later |

The **decision** (which similarity the viewer ships) follows the rule of section 8, fixed before any real data is read.

## 2. Models and tokens

| | Pythia | OLMo |
|---|---|---|
| Hugging Face id, revision | `EleutherAI/pythia-1b`, `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` | `allenai/OLMo-1B-0724-hf`, `d7cbab742d80589e714b1a2d7f838dcd21cbe143` |
| Shape | 16 blocks, width 2,048, parallel attention and MLP, rotary on 25% of head dims | 16 blocks, width 2,048, non-parametric layer norm, rotary |
| Context limit | 2,048 | 4,096 |
| End-of-text token | `<|endoftext|>` = id 0 | `<|endoftext|>` = id **50279**; id 0 is `|||IP_ADDRESS|||` |
| Extra tokens | none beyond 50,276 | 50277 `|||EMAIL_ADDRESS|||`, 50278 `|||PHONE_NUMBER|||`, 50279 |
| Training data | the Pile (test split held out) | Dolma 1.7 (shares sources with the Pile; the Pile test split is **not** guaranteed held out) |

Both tokenizers are `GPTNeoXTokenizer` on the same BPE. Ordinary text therefore gets the same ids, but **the separator
differs**, so each model reads its own end-of-text token at position 0 and the identity of the two token sequences is
asserted on positions 1 to 2,047 only. OLMo's tokenizer config has `clean_up_tokenization_spaces=True`; every decode
for display passes `False`.

**Token assertion (U0, fails closed).** (i) Documents containing any of `<|endoftext|>`, `<|padding|>`,
`|||EMAIL_ADDRESS|||`, `|||PHONE_NUMBER|||`, `|||IP_ADDRESS|||` or NUL are rejected before tokenisation. (ii) For every
accepted document, both tokenizers encode the full text to identical id sequences. (iii) No text id is in
{0, 1, 50277, 50278, 50279}. (iv) For every id that occurs in the probe, `convert_ids_to_tokens` returns the same
string under both tokenizers. A failure stops the run and is reported; nothing is patched.

**Forward pass (U1).** float32 weights and activations, `attn_implementation="eager"`, TF32 off, batch of one context,
no padding; `transformers==4.57.6`, `torch==2.5.1` (the versions the deprecated studies ran; if OLMo does not load,
the nearest working pair is pinned and recorded). Residuals are read by forward hooks on each decoder layer (first
element of the output if it is a tuple), not from `output_hidden_states`, whose last entry is taken after the final
norm in some versions; the hook outputs for layers 1 to 15 are asserted equal to the corresponding
`output_hidden_states` entries. Hidden state ℓ = 0 is the embedding output, ℓ = 1..16 the block outputs.

## 3. Probe

**Corpus.** The Pile test split, `monology/pile-uncopyrighted` revision `3be90335b66f24456a5d6659d9c8d208c0357119`,
file `test.jsonl.zst` (337,771,483 bytes), streamed in file order. Each record's `meta.pile_set_name` is its
**subset** (ArXiv, Github, FreeLaw, ...).

**Eligibility.** At least 95% of characters ASCII; Pythia token count of the full text at least 2,048; the forbidden
strings of section 2 absent. Eligible documents are shuffled with `numpy.random.default_rng(0).permutation`.

**Acceptance**, walking the shuffled list. A document's **context** is its first 2,047 tokens, read from the start of
the document (never a window inside it), so no document is split. Reject if: whitespace-only tokens exceed 30% of the
context; more than 10% of positions u ≥ 32 repeat, as their preceding 32 ids, a 32-gram that already occurred earlier
in the same context (long verbatim repetition makes near-duplicate tokens); any 32-gram of the context occurs in an
already accepted document (shared boilerplate across documents); the subset already has 4 accepted pool documents.
Rejection counts by reason and by subset are stored.

**Counts.** The first M = 32 accepted documents are the **pool** (probe order = acceptance order, which is random).
At most 4 per subset, so at least 8 subsets; the subset table is shown on the Method tab. Walking on, 4 more accepted
documents make the two **stitched control contexts** (section 9.2); they are never in the pool.

**Input sequence.** `[end-of-text of that model] + context`, length T = 2,048. Position u = 0..2,047; the residual
at u ≥ 1 has read text tokens 1..u and predicts token u + 1. Position 0 is identical in every context, carries the
attention sink and an outsized norm, and is **excluded from everything**: centring, memory sums and nodes.

**Burn-in.** b = 256. Nodes are positions 256..2,047 (1,792 per context). At memory 64, the weight the sum would put
before position 1 is at most (1 - 1/64)^256 = 1.8% of the total; at memory 8 it is negligible. The whole-prefix
kernel never reaches a steady state; that is a property of the kernel and is measured (section 7.3).

**Node sets.**

| Graph | Nodes | N |
|---|---|---|
| **single context** ("this text alone") | all positions 256..2,047 of one context; one graph per pool context (32) and per stitched context (2) | 1,792 |
| **pooled A** ("all 32 texts") | in each pool context c, 256 positions drawn uniformly without replacement from 256..2,047 with `default_rng(1000 + c)` | 8,192 |
| **pooled B** (resample, for stability only) | the first 128 of A's draw in each context, plus 128 drawn from the rest with `default_rng(2000 + c)` | 8,192 |

Every pooled node is also a node of its context's single graph, and A and B share 4,096 nodes (128 per context), so
every comparison in section 7 is on shared nodes and needs no out-of-sample extension. The later Nystrom question is
a leave-one-context-out extension on exactly these files (deferred). Node sets do not depend on the model, so both
models' graphs have identical nodes.

**Contexts shown first.** The viewer opens on pool context 0 (probe order, fixed before any result); figures that show
an example text use contexts 0 and 1.

**Stored (Volume `tm-exp1`, section 11).** `probe/probe.json`: documents (file index, subset, character and token
counts), token ids, per-token display strings (single-id decode, clean-up off; an id that is not valid UTF-8 alone is
shown as `·` with its id), node sets A and B, stitched segment tables, rejection counts, assertion results.
`acts/<model>/c<cc>.npy`: float32 array (17, 2048, 2048), hidden states ℓ = 0..16 at every position, for all 34
contexts (285 MB each, 19.4 GB in total). `attn/<model>.npz`: attention statistics (section 10). Everything downstream
is recomputable from these without the models.

## 4. Similarity (the kernel family)

**Sites.** L = 8 sites, hidden states ℓ ∈ {2, 4, 6, 8, 10, 12, 14, 16}, the same in both models. The embedding output
(ℓ = 0) is not a site: it is a function of the token alone and would put exact duplicates in the graph.

**Centring.** μ_ℓ = mean of x_{c,u,ℓ} over the 32 pool contexts and u = 1..2,047 (65,504 vectors, float64), one per
model and site. The same μ_ℓ is used for every graph, pooled and single, so **the similarity of two given tokens never
depends on which graph they are in**; only the node set and the graph rule's normalisation differ. Graph-specific
centring is deferred (section 14).

**Token vectors** at position u, for the two readings:

| Reading | Plain name | Token vector y_u | Similarity of positions t, t' | Distance d(t, t') given to the graph rule |
|---|---|---|---|---|
| cosine | each token counts equally | (x_u - μ) / ‖x_u - μ‖ | ĉ = K(t,t') / sqrt(K(t,t) K(t',t')) | 1 - ĉ |
| dot | loud tokens count more | x_u - μ | K(t,t') | (K(t,t) + K(t',t')) / 2 - K(t,t') |

**Memory kernel** (the owner's definition, fixed). With G(u, u') = ⟨y_u, y_u'⟩ and y_u = 0 for u ≤ 0 (so position 0
and anything before it contribute nothing),

    K_a(t, t') = Σ_{s ≥ 0} a^s G(t - s, t' - s),   equivalently   K_a(t, t') = G(t, t') + a K_a(t - 1, t' - 1).

It is the dot product of the feature vectors φ(t) = (a^{s/2} y_{t-s})_{s ≥ 0}: the two pasts are laid side by side
**aligned at their ends** and compared lag by lag. It is not the dot product of running means (that would compare
every lag with every lag). Two consequences the primer states: an end-aligned sum of independent noise has no
position structure, so any smoothness along the text comes from residuals that really are similar at nearby
positions; and for a = 1 two prefixes of different lengths are compared over the shorter one, so whole-prefix
similarity depends on the two positions even for text with no positional structure.

**Memory grid.** m = memory length in tokens, a = 1 - 1/m.

| Label | a | Meaning |
|---|---|---|
| none | 0 | the token's own residual (DepthEmbedding's kernel: at a = 0 the cosine reading is its centred cosine, the dot reading its `dot_effective_cosine` distance, except that centring is over the corpus) |
| 8 | 0.875 | about the last 8 tokens |
| 64 | 0.984375 | about the last 64 tokens |
| whole | 1 | the whole prefix from position 1 |

8 constructions = 4 memories × 2 readings. **"Smart EMA" in this experiment** is a measurement, not a kernel: the
models' own look-back distances are measured (section 10) and drawn as reference marks next to the grid. A kernel with
lag weights taken from them (the filter takes any lag profile w_s at no extra cost) is deferred until those
measurements exist.

**Loud tokens.** Under the dot reading, tokens with large centred norm dominate the distances for about m tokens
after them. They are kept (that is what the reading means). Reported per model and site: the share of total centred
squared norm carried by the top 1% of tokens, and the 10 most frequent token strings among tokens whose norm exceeds
5 times the median.

**Exact computation.** For each pair of contexts (c, c'), c ≤ c', and site: G = Y_c Y_c'^T (2,047 × 2,047, float64
from float32 activations), then the recursion over t = 1..2,047, row by row,
`K[t, 1:] = G[t, 1:] + a * K[t-1, :-1]`, `K[t, 0] = G[t, 0]` (indices of text positions), vectorised over pairs and
memories; a = 1 is the same recursion (a cumulative sum along diagonals). It is exact, and stable for 0 ≤ a ≤ 1
because every weight is at most 1. The closed form a^t Σ a^{-u} g(u) is not used (it overflows). Blocks for c > c'
are transposes. The node Gram is read off at the node rows and columns. No GPU is needed: per site and reading the
528 blocks cost about 7·10^13 flops. Tests (section 13): the recursion against the explicit feature concatenation φ
and against a direct double loop, to 1e-12 relative.

**Checks per graph** (fail closed): no token with ‖x_u - μ‖ < 1e-6 × the median; smallest off-diagonal distance
above 1e-9 (no exact duplicates; the probe filters make them impossible by design).

## 5. Graph rule

**Default: participation-ratio rule, k = 64** (copied from DepthEmbedding `invert/dm.py`, `ea_kernel(renyi=2)`). For
node i, q(j | i) = exp(-β_i (d_ij - min_{j'≠i} d_ij')) / Z_i over j ≠ i, with β_i set by bisection so that the
participation ratio 1 / Σ_j q(j | i)^2 equals k. Then W = (q + q^T)/2 with zero diagonal, symmetric Sinkhorn
scaling to a doubly stochastic P (`sinkhorn_sym`, tolerance 1e-11, float64), P ← (P + P^T)/2, and the lazy walk
M = (I + P)/2. The same k = 64 is used for the single and pooled graphs (the same number of effective neighbours,
not the same share of the population).

**Why not Shannon perplexity, which the owner suggested as gentler.** It is the same construction with
exp(entropy) = k in place of the participation ratio, and it is equally gentle in spirit (smooth, per-point width).
In DepthEmbedding it produced modes carried by two images (46 of the first 64 joint modes with over 10% of their
squared norm on two images at N = 8,000), and the reason is structural: a row of Shannon perplexity 64 over 8,192
nodes can put about 0.6 of its weight on one close neighbour, whereas a participation ratio of k bounds every
conditional weight by 1/sqrt(k) = 0.125. Token data have near-duplicates (repeated phrases, code), so the risk is
higher, not lower. The participation-ratio rule also has an exact out-of-sample row (`ea_oos_rows`), which the later
Nystrom work needs. TorchDR is not used: its affinities calibrate Shannon entropy, and the tested in-house code
covers both rules.

**Pilot comparison (P2, section 11), decided automatically.** Rules: participation ratio k = 64; Shannon perplexity
64 (`ea_kernel(renyi=1)`); union k-nearest-neighbours, k = 15, floor 1e-6, Sinkhorn (`knn_kernel`). Pythia, cosine
reading, memories none and 64, all 8 sites, single-site walks (no through-depth), on pooled A and pool contexts 0-3:
80 (site, memory, graph) cells. In each cell the first 64 nontrivial eigenvectors of M_ℓ are taken and a mode is
**localised** if its spread (section 7.4) is below 20 tokens. Also stored: each rule's effective width (exp of the
mean row entropy of P), and the largest single weight in any row. **Rule:** the default is the rule with the fewest
localised modes in total; a difference under 1% of the 5,120 examined modes is a tie, broken in the order
participation ratio, nearest neighbours, Shannon. The pilot also runs k = 32 and 128 (participation ratio, memory
64, pooled A, 8 sites) and reports the agreement of their 16-mode spaces with k = 64's; this is reported, not
decided on.

## 6. Through-depth operator

`spacetime.py` is copied from DepthEmbedding commit `8dd7a66` byte for byte, with `spacetime_stage.py` (needed by
its last test) and `test_spacetime.py`; the tests must pass unchanged. Nothing in it is redesigned.

| Choice | Value |
|---|---|
| Sites | the 8 hidden states of section 4, in depth order, path with free ends |
| Walks | M_ℓ from section 5, same node set at every site (a token at one layer is linked to itself at the next) |
| Units | relative site units, c_ℓ = mean of the first 8 nontrivial 1 - λ_k of M_ℓ (`site_scales('own')`) |
| Coupling | t = t* only (`balance_direct`, with the full site spectra passed in); no slider. For the chosen construction only, t*/4 and 4t* are also solved (pooled A, both models) as a robustness check |
| Modes | 24 computed (`modes`, block-elimination `Solver`, Lanczos on the inverse), 16 stored and read |
| Precision | float64 throughout |

**Acceptance per solve** (all must pass before a unit is marked complete; the logic of `spacetime_stage.accept`):
unprojected residual of each of the 16 modes below 1e-8; at t*, |λ_1 / (t* · 4 sin²(π/16)) - 1| < 1e-6; occupancy
and recurrent strengths each sum to 16 (1e-8, 1e-3); the scalar lower bound below λ_1; all outputs finite; Sinkhorn
row error below 1e-10 at every site; every walk connected (second eigenvalue below 1 - 1e-12). The relative gap
(λ_17 - λ_16)/λ_16 is stored: a small gap means only the 16-dimensional space, not mode 16, is meaningful.

**Dense size.** At N = 8,192 and L = 8 the walks, inverses and site eigenvectors take about 13 GB together; the solve
fits in 96 GB on 16 cores, as DepthEmbedding's N = 8,000, L = 12 solves did.

## 7. Readings

### 7.1 The picture as a matrix on tokens

G_ℓ = the N × 16 matrix of the 16 modes' slices at site ℓ (columns orthonormal over all 8N entries).
R = Σ_ℓ G_ℓ G_ℓ^T (N × N, trace 16) counts each token function once however many modes or layers reuse it.
U = the top 16 eigenvectors of R (the picture's 16 main token functions, orthonormal). Everything below is unchanged
by rotating the 16 modes, so near-ties between modes cannot change it.

### 7.2 Agreement between two pictures on the same tokens

A(R_1, R_2) = tr(R_1 R_2) / sqrt(tr(R_1²) tr(R_2²)) ∈ [0, 1] (linear centred kernel alignment; mode slices are
already mean-zero). For two 16-dimensional spaces with equal weights it equals the mean cos² of their principal
angles; unlike principal angles it weights directions by how strongly the picture holds them, so restrictions to a
few hundred tokens that lose rank do not produce noise-driven values. **Null:** 200 random permutations of one
picture's token labels (within each document for pooled graphs, within the text for single graphs); reported as
mean and 95th percentile. **Interval** for pooled numbers: 200 bootstrap draws of the 32 documents with
replacement (a drawn document brings all its nodes); 5th and 95th percentiles.

### 7.3 What the picture is made of

Z = the span of three families of functions on the graph's nodes, orthonormalised by QR in this order (a column whose
residual norm is below 1e-8 is dropped; dimensions are recorded):

| Family | Pooled graph | Single-context graph |
|---|---|---|
| document | indicators of the 32 documents (includes the constant) | the constant |
| position | p_j(t) = cos(π j (t - 255.5)/1,792), j = 1..16, the same function of position in every document (wavelengths down to 224 tokens) | the same 16 functions on this text (here position and slow drift of topic cannot be told apart, and the label says so) |
| common token | indicators of token ids occurring at least 20 times among the graph's nodes | the same rule within the text |

Shares of the picture: s_doc = tr(Q_doc^T R Q_doc)/16, s_pos and s_tok likewise with each family's new orthonormal
columns, and the **remainder share** s_rem = 1 - s_doc - s_pos - s_tok: the part of the picture not predictable from
which document, where in it, or which common token. Chance levels (for a random 16-dimensional picture) are the
families' dimensions over N - 1 and are drawn on every share. Per-mode shares (viewer) use the unit slice u of one
mode at one site in place of R/16. The **remainder part** of a picture is R̃ = (I - Π_Z) R (I - Π_Z).

### 7.4 Other readings

| Reading | Definition | Chance or reference |
|---|---|---|
| spread of a mode at a site | 1 / Σ_i u_i⁴ for its unit slice u: how many tokens carry it; localised if below 20 | N for a flat function |
| captured share of a grouping F | ‖U^T Q_F‖_F² / r_F, Q_F an orthonormal basis (r_F columns) of F's centred indicators | 16/(N - 1) |
| same-document walk share (pooled, per site) | (1/N) Σ_i Σ_{j in i's document, j ≠ i} P_ℓ[i, j]: how much of a token's walk step stays in its own text | 255/8,191 = 0.031 |
| stability | A(R_A, R_B) restricted to the 4,096 tokens shared by pooled A and B | permutation null |
| cross-model agreement | A(R̃_Pythia, R̃_OLMo) on the same nodes (Z depends only on tokens, so it is the same for both) | permutation null |
| cross-reading agreement | A(R_cosine, R_dot), same model and memory | permutation null |
| single vs pooled, text c | on the 128 tokens of c shared by A and B, both pictures restricted and centred on those tokens: A(H R_A H, H R_single(c) H) | **ceiling:** A(H R_A H, H R_B H) on the same tokens; permutation null |
| layer-matched agreement | A computed from G_ℓ G_ℓ^T alone, per site | permutation null |
| depth weight, eigenvalue, gap | from `spacetime.py` (`depth_weight`, values, λ_17 - λ_16) | |

Single-context numbers are summarised as median and 10th-90th percentiles over the 32 pool texts.

## 8. Choosing the construction the viewer ships

Fixed now, applied by `decide.py` to the stored readings, and shown on the page as a table with every number:

1. **Numerical gate.** Every solve of the construction passed acceptance (section 6).
2. **Not mostly obvious.** Remainder share s_rem ≥ 0.25 for both models, in pooled A and in the median single graph.
3. **Reproducible.** Stability ≥ 0.5 for both models.
4. **Score** = mean of (cross-model agreement on pooled A, median over texts of cross-model agreement on single
   graphs). The highest score wins. If the runner-up is within 0.05, prefer the cosine reading (the inherited
   geometry), then the shorter memory (simpler to read).
5. If no construction passes 2 and 3, the highest score is shipped anyway and the viewer and the answer of entry E2
   say plainly that no construction passed.

Why cross-model agreement is the score: two independently trained models with different data and architecture
details agreeing on the same tokens is label-free replication; agreement that only reflects document, position or
common tokens is removed first (R̃), because both models know those trivially. The viewer's second memory option is
"none" if the winner has memory; if the winner is "none", it is the best-scoring construction with memory and the
same reading.

## 9. Controls

### 9.1 Synthetic, with known answers (pipeline control)

Same code path end to end (similarity, graph rule, through-depth, readings). M_s = 8 contexts of 1,024 positions
(position 0 excluded), burn-in 256 (768 nodes per text), pooled A and B with 256 and 128-shared per text
(N = 2,048), width 64, 4 sites, 5 seeds. At position u of context c and site ℓ:

    x = α_ℓ r_{z(u)} + β_ℓ e_{w(u)} + γ g(u) + ε

- z(u) ∈ {1..4}: a hidden **regime**, a Markov chain per context switching with probability 1/128 per token;
  r_1..r_4 random unit vectors shared by all contexts; α_ℓ = α · (0, 1, 1, 1): absent at the first site.
- w(u): a **token** id drawn independently from a Zipf(1.1) law on 500 ids; e_w random unit vectors;
  β_ℓ = β · (1, 0.7, 0.5, 0.3).
- g(u) = cos(πu/1024) g_a + sin(πu/1024) g_b: a **position drift**, the same function of position in every context.
- ε: independent N(0, I/64).

Section 7 applies with the sizes changed: 768 node positions per text, so the position harmonics are
cos(π j (t - 255.5)/768); common tokens are ids seen at least 20 times. Variants: **planted** (α = 0.25, β = 1, γ = 0), **drift** (planted with γ = 0.5), **noise** (α = β = γ = 0; regimes
still drawn as labels). Why α = 0.25 works: per token, two same-regime tokens have mean cosine about 0.03 against a
noise spread of about 1/sqrt(64) = 0.125, invisible; the end-aligned sum keeps the mean and shrinks the spread by
about sqrt(2m), to 0.011 at m = 64. Targets (cosine reading, median over seeds; the regime span has 3 dimensions):

| # | Known answer | Pass if |
|---|---|---|
| K1 | regimes need memory | captured share of the regimes, pooled A: ≥ 0.8 at m = 64, ≤ 0.3 at none |
| K2 | regimes are absent at the first site | regime occupancy ‖Q_reg^T G_0‖² ≤ 0.1 × its maximum over sites (m = 64) |
| K3 | tokens dominate without memory | common-token share s_tok ≥ 0.3 at none, ≤ 0.1 at m = 64 |
| K4 | nothing planted, nothing found | noise variant: captured share of the (unused) regime labels and stability both at or below their permutation 95th percentiles, every memory |
| K5 | the memory sum creates no position structure | planted: s_pos ≤ 0.05 at none, 8, 64; drift: s_pos ≥ 0.3 at whole. (Planted at whole is reported, not gated: section 4 predicts the kernel's own position dependence.) |

α, β, γ may be adjusted once, on seed 0 in the test suite, before any real-data unit runs; the final values are
recorded and seeds 1-5 then run unchanged on Modal. A failed target on seeds 1-5 stops the run before U3.

### 9.2 Stitched real contexts (real activations, expected answer)

Two extra contexts from four fresh accepted documents (A, B from different subsets for each). **S512**: A tokens
1-512, B 1-512, A 513-1,024, B 513-1,023. **S64**: A and B alternate in 64-token segments (A continues where it
stopped each time). Token ids are concatenated, never re-tokenised across a join. Measured on their single graphs:
captured share of the A/B indicator, for every construction and both models. Expected, stated now: S512 ≥ 0.5 at
none, 8 and 64, lower at whole; S64 ≥ 0.5 at none and 8, lower at 64 than at 8, near chance at whole (a memory
longer than the segments blurs them). These are predictions about the models as well as the method, so they are
reported, not gated; a failure is shown and discussed. The viewer lists both contexts with their joins marked.

### 9.3 Numerical and data checks

Token assertion (section 2). Causal indexing, context 0: a float64 forward pass on positions 0..1,023 alone matches
the full float64 pass at those positions to 1e-10 relative at every layer; the float32 discrepancy is recorded per
layer as a hazard. Hook outputs equal `output_hidden_states` (section 2). Acceptance per solve (section 6). Kernel spot
check by the independent checker (section 12.5).

## 10. Timescales the models use (Q5; what "smart EMA" becomes)

From the eager attention probabilities in U1, per model, layer and head, over query positions i = 256..2,047 of the 32
pool contexts: sink mass A[i, 0]; mean look-back Σ_{j≥1} A[i, j](i - j) / Σ_{j≥1} A[i, j]. Per layer: median over
heads of the mean over queries, and the interquartile range over heads. From the residuals, per model and site:
ρ(δ) = mean cosine of centred unit token vectors δ apart within a text, δ = 1..1,024; the **half-similarity lag** is
the smallest δ with ρ(δ) ≤ ρ(1)/2 (ρ of tokens in different texts is about 0 after centring; recorded). Stored in
`attn/<model>.npz` and `analysis/timescales.json`.

## 11. Compute plan

Modal app `tm-exp1`, Volume `tm-exp1`, following the `modal` skill (detached coordinator, one worker per unit,
parts named by what they measured). Rates from modal.com/pricing on 2026-11-23: CPU $0.0000131 per physical
core-second, memory $0.00000222 per GiB-second, L4 $0.000164/s; volumes free to 1 TiB. Nothing heavy runs on the VPS;
the VPS runs the unit tests (small sizes) and builds the site from fetched summaries.

| Unit | Count | Hardware | Work | Estimate each | Spend |
|---|---|---|---|---|---|
| U0 probe | 1 | 2 cores, 8 GiB | download, filter, tokenise both ways, assert | 15 min | $0.05 |
| U1 extract | 2 (model) | L4, 4 cores, 32 GiB | 34 forward passes, hooks, attention statistics, float64 causal check | 20 min | $0.4 |
| P2 pilot | 8 | 16 cores, 64 GiB | 6 (rule × memory) cells of section 5; k = 32/128; one full through-depth solve for timing | 25-45 min | $5 |
| U5 synthetic | 30 (variant × seed × reading) | 8 cores, 32 GiB | all 4 memories, pooled A and B, 8 single texts | 5 min | $1.6 |
| U3 pooled | 16 (model × reading × memory) | 16 cores, 96 GiB | node Grams on A ∪ B, 16 walks, 2 through-depth solves, neighbours | 70 min | $29 |
| U4 single | 136 (model × reading × context) | 8 cores, 32 GiB | 4 memories × (8 walks, 1 solve), neighbours | 4.5 min | $6.4 |
| U6 coupling | 2 (model) | 16 cores, 96 GiB | chosen construction, pooled A, t*/4 and 4t* | 70 min | $3.6 |
| U7 analysis | 1 | 16 cores, 64 GiB | all readings, nulls, bootstraps, decision, site packaging, kernel spot check | 45 min | $1 |

**Projected spend about $47; ceiling $80** (a retry of any stage fits). Wall-clock: U0-U1 35 min; pilot 45 min;
U3, U4, U5 in parallel about 80 min; U6 70 min after the decision; U7 45 min: **about 4.5 hours**, plus site work.

**Order and gates.** U0 → U1 → (P2 and U5 together) → automatic rule decision and synthetic gate → U3, U4 → `decide`
→ U6 → U7. **Pilot timing fallback, fixed now:** if the pilot's full pooled solve projects U3 above $40, pooled n
drops to 192 per context (N = 6,144; A and B share 96) for every pooled run, and the page says so.

**Volume layout.** `/hf` (model and dataset cache); `/probe/probe.json`; `/acts/<model>/c<cc>.npy`;
`/attn/<model>.npz`; `/pilot/<cell>.npz`; `/synthetic/<variant>_s<seed>_<reading>.npz`;
`/pooled/<model>/<reading>_<memory>.npz` (for A and B: float32 mode slices (8, N, 24), values, site spectra (first 33),
c_ℓ, t*, gaps, acceptance record, per-site same-document walk shares, top-10 kernel neighbours per node per site as
uint16 indices and float32 similarities; about 30 MB); `/single/<model>/<reading>/c<cc>.npz` (the same per memory;
about 5 MB); `/coupling/<model>.npz`; `/analysis/*.json` (every reading of section 7, the decision, controls);
`/site/` (packaged viewer data, section 12.2). Activations stay on the Volume; only `/analysis` and `/site` are
fetched. Run ids, spend and status go in the README Status section.

## 12. The site

Public repo `CharlesR-W/TransformerModes`, GitHub Pages, machinery copied from DepthEmbedding (`build_site.py`,
`xsite/xlib.py` with `Out.fact` and `{{fact:...}}` tokens, `check_fragment.py`, `check_browser.py`, the live chart
renderer, palette and CSS; the CIFAR embedding figure is not copied). Three tabs: **Viewer**, **Experiments**,
**Method**. Every tab carries a one-line banner: both models, 16 blocks, width 2,048, the 8 sites marked.

### 12.1 Viewer layout

Desktop: two control bars, then the text (left, about 65% of the width) with a strip chart under it, and a side
column (right) with the mode card, a small scatter and the token card. Phone (390 px): one column in the order
controls, text, strip, mode card, token card, scatter; no horizontal scroll.

- **Text.** The context as inline tokens with whitespace and newlines kept, each token's background coloured by the
  selected mode's unit slice at the selected layer (diverging palette; range ±(99th percentile of |value| over all
  layers of that mode), fixed across layers, values beyond drawn at the end colours; the legend says so). Burn-in
  tokens (positions 1-255) are grey italic, not coloured. In the pooled graph only that text's 256 node tokens are
  coloured; the others are plain, and a note says "1 token in 7 is in the pooled graph". If the mode's depth weight
  at this layer is under 1%, the whole text is faded with the note "this mode barely lives at this layer". Stitched
  texts show their joins as thin bars with A/B labels in the margin.
- **Strip chart** under the text (full width, 120 px): x = position 0-2,047, y = the same values; a line through the
  single graph's 1,792 values or dots at the pooled nodes; the clicked token marked. Smooth curves here are smooth
  functions of position, visible at a glance.
- **Mode card.** Eigenvalue; depth weight over the 8 layers as small bars; the four shares (document, position,
  common token, remainder) as one stacked bar with chance ticks; spread in tokens; "the sign of a mode is arbitrary".
- **Scatter** (240 px): all nodes of the graph with x = mode 1 and y = the selected mode (mode 2 when mode 1 is
  selected), at the selected layer; single graph coloured by position (sequential ramp), pooled graph grey with this
  text's nodes in the accent colour. Hovering a point highlights its token; clicking selects it.
- **Token card** (after clicking a token or a point). The token, its position, its value; then its 10 nearest tokens
  under the similarity itself (before any graph), at this layer: the value (ĉ for the cosine reading, the distance d
  for the dot reading, nearest = smallest d in both), text name and position, and the
  last up to 24 tokens ending at that neighbour, right-aligned under the clicked token's own last 24 tokens (the
  kernel compares pasts aligned at the end, so this is what it compared). In the pooled graph neighbours come from all
  32 texts, with the token's same-document walk share at this layer.

**Controls (6).**

| Control | Values | Why it is needed |
|---|---|---|
| Text | 32 pool texts (subset and first words) + 2 stitched controls | the text is the object being read |
| Graph | this text alone / all 32 texts | the owner asked for both; toggling shows whether the two pictures look alike |
| Model | Pythia / OLMo | the two models are the replication; tokens align, so the same text is shown |
| Memory | the chosen construction / its counterpart (section 8) | the local-versus-memory contrast is the scientific point of the kernel |
| Layer | 8 stops, 2-16 | each mode has a slice at every layer |
| Mode | 1-16, each button with a tiny share bar | which of the picture's functions colours the text |

Reading (cosine or dot), coupling and graph rule are fixed and named in a one-line operator caption under the bars;
other choices live in the Experiments tab. The current state is kept in the URL hash so Experiments entries can link
to a view. The primer's open or closed state is remembered in `localStorage` inside try/catch.

### 12.2 Viewer data (lazy, per text)

| File | Content | Size |
|---|---|---|
| `d/manifest.json` | texts, subsets, joins, sites, constructions shipped, node table of pooled A, chosen operator caption | 60 KB |
| `d/text/c<cc>.json` (34) | token ids and display strings | 50 KB each, 1.7 MB |
| `d/modes/<model>/<mem>/single/c<cc>.i8` (136) | unit slices, int8, (8 sites, 16 modes, 1,792), with a float32 scale per site and mode in the info file | 230 KB each, 31 MB |
| `d/modes/<model>/<mem>/pooled.i8` (4) | the same for 8,192 nodes | 1 MB each, 4 MB |
| `d/info/<model>/<mem>/<graph>[/c<cc>].json` | eigenvalues, depth weights, shares, spreads, scales, gap | 15 KB each, 2 MB |
| `d/nb/<model>/<mem>/single/c<cc>.bin` (136) | top-10 neighbours per node per site, uint16 index and float16 similarity | 570 KB each, 78 MB |
| `d/nb/<model>/<mem>/pooled.bin` (4) | the same for pooled nodes | 2.6 MB each, 10.5 MB |

About 130 MB in all; opening one text fetches about 1.5 MB, a token click about 0.6 MB more.

### 12.3 Primer (top of the Viewer tab, collapsible, about one screen)

One short paragraph or small drawing per index the reader meets: **position** (token number from the start of the
document; the first 255 are burn-in); **layer** (which block's output; 8 of 16 shown); **memory** (a drawing of two
pasts aligned at their right ends with fading weights; "none" is the token alone; why whole-prefix similarity depends
on position); **the two readings** (one sentence, and which one the viewer uses); **the two graphs** (one text, or 256
tokens from each of 32 texts; Sinkhorn scaling removes how densely tokens are sampled); **a mode** (a pattern over
tokens and layers that changes little between similar tokens and between neighbouring layers; numbered from the
smoothest; the sign is arbitrary; the depth bars say where it lives); **coupling** (fixed at the value where
changing across layers and roughness within a layer first trade evenly; one sentence); **the share bar** (what part
of the picture is document, position, common token, or none of these, with chance marks); **how to check** (click a
token and read what it was compared with).

### 12.4 Experiments tab entries

Each follows `_template.html`: question, scope, 2-3 sentence answer with one bold sentence, lead figure, 3-5
number tiles, "What was computed" table, at most two supporting figures, at most six caveats, closed `<details>` for
decisions and extra figures. Numbers are facts from generators, never typed by hand.

| Entry (question) | Lead figure: x · y · facets · uncertainty | The sentence a reader should be able to say |
|---|---|---|
| E1 Does the method find what we planted, and nothing else? | x = memory (none, 8, 64, whole) · y = captured share of the planted regimes · series: planted (solid), noise (dashed grey, control), drift (dotted); chance line · facets: none (cosine; dot in More figures) · band: min-max over 5 seeds. Supporting: stitched texts, x = memory, y = captured share of the A/B indicator, S512 and S64, both models, chance line | "Regimes lasting about 128 tokens are found with 64-token memory and not from single tokens, noise gives nothing, and on real text 64-token alternations blur once the memory reaches 64." |
| E2 Which similarity should the viewer use? | two panels sharing rows; rows = the 8 constructions · left x = remainder share, Pythia circle and OLMo diamond, gate line 0.25 · right x = score (cross-model agreement of remainders), whiskers = 5-95% bootstrap over documents, hollow marker = null 95th percentile · chosen row highlighted, excluded rows grey with the failed gate named. Supporting: stacked share bars per construction and model; stability per construction | "[Chosen] wins: [x]% of its picture is beyond document, position and common tokens in both models, and the models agree on that part at [y] against a null of [z]." |
| E3 Does one text's own picture look like the pooled picture of that text? | x = memory · y = agreement on the text's shared tokens · series: single vs pooled (solid), pooled A vs B (dashed, ceiling), null (dotted grey) · facets: model · band: 10-90% over 32 texts · chosen reading | "At [memory], a text's own picture matches the pooled one about as well as / far less well than two pooled samples match each other, so the single text is / is not like a sample of the pool." |
| E4 Which graph rule: participation ratio, Shannon perplexity or nearest neighbours? | x = rule, grouped by memory (none, 64) · y = localised modes among the 5,120 examined · effective width printed above each bar · no band (counts over all cells) | "Shannon perplexity gives [n] modes carried by fewer than 20 tokens against [n] for participation ratio, so [rule] is used." |
| E5 How far back do the two models look? | x = layer 2-16 · y = tokens (log) · series: median look-back of attention (solid) and half-similarity lag of residuals (dashed), hue = model; reference lines at 8 and 64 · band: interquartile range over heads | "Middle layers look back about [x] tokens and residuals stay similar for about [y], so the 64-token memory sits [inside / beyond] the models' own range." |

Tiles: E1 targets K1-K5 passed; E2 chosen construction, its remainder share, score, stability; E3 median agreement
and ceiling, same-document walk share against 0.031; E4 the three counts; E5 attention sink mass. E2's "More figures"
holds the cross-reading agreement, layer-matched agreement, the coupling check (agreement of t*/4 and 4t* with t*),
and the loud-token statistics. The Method tab carries sections 2-10 of this spec in page form, the subset table,
the decisions table and the deferred list.

### 12.5 How a reader can check there is nothing odd going on

- The synthetic control with its five known answers, and the stitched texts, are on the page; the stitched texts are
  also in the viewer with their joins drawn, so the reader can see whether colours follow the joins.
- Every mode shows what it is made of (document, position, common token, remainder) with chance marks, and how many
  tokens carry it; localised modes are flagged.
- Clicking any token lists, as text, the tokens it is most similar to and the pasts that were compared, with the
  similarity value; nothing between the residuals and that list is hidden.
- The decision table shows all 8 constructions with every gate number, so the choice can be re-done by eye.
- Facts are written at build time from `/analysis`. `site/xsite/check_numbers.py` is written separately from the
  pipeline (numpy only, importing nothing from it) and recomputes from the stored mode slices and the probe tables:
  every share, stability, cross-model agreement and the decision for the chosen construction, and K1-K5; it fails
  the build on any difference above 1e-6. On Modal (inside U7) it also recomputes the similarity for 200 random node
  pairs per construction directly from the activations by the explicit lag sum, against the stored neighbour values.
- `check_browser.py` drives headless Chrome over every control, a token click and both viewports, and fails on
  console errors or external requests; screenshots are inspected before deploying.

## 13. Code and tests

New files: `probe.py` (U0), `extract.py` (U1), `similarity.py` (centring, recursion, node Grams, distances),
`kernels.py` (copied from DepthEmbedding `invert/dm.py` at `8dd7a66`: `_cond_rows`, `ea_kernel`, `ea_oos_rows`,
`sinkhorn_sym`, `knn_kernel`, `perplexity`, unchanged), `stage.py` (walks, through-depth solve, acceptance,
neighbours), `readings.py` (section 7), `synthetic.py`, `decide.py`, `modal_app.py` (from the `modal` skill's
`modal_job.py`), `test_tm.py`, `test_kernels.py` (the masked-row and duplicate-bound tests from `invert/test_dm.py`).
Copied unchanged: `spacetime.py`, `spacetime_stage.py`, `test_spacetime.py`. Site: `site/` as in section 12.

`test_tm.py` must cover: the recursion against φ and a double loop for a ∈ {0, 0.5, 1} (1e-12); block transposition;
the a = 0 readings against plain centred cosine and half squared Euclidean distance; node-Gram assembly against a
direct computation; share decomposition sums to 1 and gives (1, 0, 0, 0) for a document indicator and (0, 1, 0, 0)
for a position harmonic; A(R, R) = 1 and invariance to rotating the modes; the token assertion on two short real
texts (tokenizers only); synthetic K1-K5 at seed 0, small size, on the VPS. All tests run on the VPS in under ten
minutes.

## 14. Decisions made on the owner's behalf

| # | Decision | Chosen | Alternative not taken | Check |
|---|---|---|---|---|
| 1 | Corpus | Pile test split (held out for Pythia only) | a Dolma sample, or a mix | caveat on the page |
| 2 | Context | first 2,047 tokens of a document with at least 2,048 | shorter T to admit more documents; windows inside documents | subset table |
| 3 | Sizes | 32 documents, 256 pooled tokens each (N = 8,192), burn-in 256 | 24 × 512 (N ≈ 12k): above the proven dense size | pilot timing |
| 4 | Diversity | at most 4 documents per subset | uniform among long documents (favours papers, law, code) | subset table |
| 5 | Filters | ASCII 95%, whitespace 30%, internal and shared 32-gram repetition | no filters (near-duplicate tokens) | rejection counts |
| 6 | Sites | 8 even layers | all 16 | E2 layer-matched figure |
| 7 | Memory grid | none, 8, 64, whole | lengths derived from the models | E5 marks |
| 8 | Readings | cosine of unit-token pasts; Euclidean on raw-token pasts | mixed readings (cosine of raw pasts, Euclidean on unit tokens) | cross-reading agreement |
| 9 | Centring | one corpus mean per model and site, for every graph | centring over each graph's nodes, or per document | deferred |
| 10 | Graph rule | participation ratio, k = 64, same k in both graphs | Shannon perplexity (owner's suggestion); k-NN; k growing with N | pilot E4 |
| 11 | Coupling | t* only, relative units | the six-stop slider of DepthEmbedding | U6 check |
| 12 | Obvious parts | documents, 16 position harmonics, tokens seen ≥ 20 times | all token identities (span too large to remove); other J | shares with chance marks |
| 13 | Default rule | gates 0.25 and 0.5, score = cross-model remainder agreement | rank by stability, or by one model alone | decision table |
| 14 | Pooled colouring | only the 256 sampled tokens of a text are coloured | extend pooled modes to every token (out-of-sample) | deferred |
| 15 | Precision | float32 forward pass, float64 for all kernel and operator maths | float64 forward pass | causal check hazard |
| 16 | Controls | stitched texts from fresh documents, outside the pool | stitch pool documents | none needed |

## 15. Deferred

Lag weights taken from the models (attention look-back or residual autocorrelation: the "smart EMA" kernel); the
recurrent random sketch and the leave-one-context-out Nystrom comparison; attention-weighted links between positions
across layers; a training-checkpoint axis and the model as a coupled axis; the coupling slider, native units and the
Trower et al. matrix; graph-specific, per-document or position-wise centring; cross-document-only neighbours; mixed
readings and any clipping of loud tokens; post-hoc sets (k-means readouts) from DepthEmbedding; a width sweep of k
beyond the pilot; longer contexts (OLMo reads 4,096); more models.

## 16. Departures from the brief

- **Separator and token identity.** The brief says the same text gives identical ids. It does for ordinary text, but
  the end-of-text token is id 0 in Pythia and id 50279 in OLMo, where id 0 means `|||IP_ADDRESS|||` and three PII
  placeholder tokens exist. Each model reads its own separator at position 0; the identity assertion covers
  positions 1-2,047 and excludes documents with the placeholder strings.
- **Sizes.** 32 documents × 256 pooled tokens (N = 8,192) rather than 24-32 × 512 (about 12k): the dense solver is
  proven at 8,000 and its cost grows with N³; there is a further fallback to 192 per document.
- **No GPU for kernels.** The cross-document Grams and memory recursions are cheap on CPU inside the solve units, so
  only extraction uses a GPU and no large kernel matrices are stored.
- **Storage.** All 17 hidden states in float32 (19.4 GB), not 8 layers in float16: float16 loses precision on loud
  tokens, and the other layers cost little and serve later choices of sites.
- **Centring settled** as one corpus mean shared by every graph, so pooled and single graphs differ only in their
  nodes; the brief left it open.
- **Same-document dominance** is measured (walk share, document share, E3) rather than corrected: Sinkhorn corrects
  sampling density, but a single text is not only a denser sample of the pool, it may cover a different region, which
  no density correction removes.
- **"Smart EMA"** becomes the timescale measurement of E5; the kernel built from it is deferred.
- **Comparisons use kernel alignment of R** instead of principal angles of single bases: it needs no mode matching,
  weights layers by how much of the picture they hold, and equals the mean cos² of principal angles in the
  equal-weight case.
- **Token identity** is measured on tokens seen at least 20 times, not all tokens (removing all would remove most of
  a single text's dimensions). Minor correction: identical tokens give identical vectors only at the embedding output,
  which is not a site; after the first block a residual depends on its context.
- **Viewer** has two two-way toggles (Graph and Memory) where the brief proposed at most one: each answers an
  explicit request of the owner (both graphs; local versus memory), and the total is six controls.
