# TransformerModes, experiment 1: memory kernels on two language models

Spec v3.2, 2026-11-23. History: v1; v2 (review 1); v2.1 (owner's answers); v3 (review 2, numerics prototype
`reviews/numerics-prototype-sol.md` with `exp1/bench_numerics.json`, local diagnostic `exp1/local_pilot.json`);
v3.1 (review 3, `reviews/spec-v3-astra.md`); v3.2 (the three Stage A fixes of `reviews/spec-v3.1-astra.md`, applied by
the orchestrator: tie-proof selection overlap, stitched baseline, audit queries drawn from Q, audit reserved first). Exploratory. Sources: `BRIEF.md`; DepthEmbedding `spacetime.py`,
`spacetime_stage.py`, `test_spacetime.py`, `SPEC_SPACETIME.md`, `invert/dm.py`, `invert/NOTES.md`; `exp1/`
(`similarity.py`, `graph.py`, `solve.py`, `kernels.py`, `test_numerics.py`). **The owner is asked to approve Stage A
only.** Stage B (sections 6-8) is provisional, pending Stage A's results and the list in section 16.

**In one paragraph.** We run Pythia-1b and OLMo-1B once on the first 2,047 tokens of 32 long Pile documents and store
every residual and next-token prediction. A token's similarity to another compares their recent pasts aligned at the
end, with a memory length (none, 2, 8, 64 tokens, or the whole prefix). **Stage A, the neighbour census** (hard cap $8,
projected about $6.80): for every token, who are its 64 nearest tokens under each memory and each way of accounting for
which text a token comes from: other texts, the same text far away, or the same text nearby? On a fixed sample, do
neighbours share the model's next-token predictions more than matched controls? No eigen-solves; the result is a page,
figures and a viewer in which clicking a token lists its neighbours as text, verified by its own independent audit.
**Stage B** (separate approval, provisional): DepthEmbedding's through-depth modes for the local kernel and the one
memory construction Stage A's screen supports, dense and exact on one GPU. **Stage B is not worth funding if** no memory
construction passes the geometric screen of section 5 in both models (no candidate met the prespecified screen, which
by itself says nothing more than that); the census is then the result.

## 0. Why this is the smart way to do it

Labels: **[here]** measured in this repo; **[sibling]** measured in DepthEmbedding; **[derived]** by calculation;
**[pilot]** to be measured before the stage's main spend.

**The evidence that shaped Stage A** [here]. A free CPU diagnostic on Pythia-160m: four WikiText articles, two Python
standard-library files and two licence texts (not Pile documents), 1,023 tokens each, nodes from position 128, layers
3/6/9, cosine reading, each token's 64 nearest neighbours. Ranges are over the three layers.

| Memory | Own-text share, corpus mean (chance 0.125) | Own-text mean removed | ... and no neighbour within 64 positions | Same top-1 prediction (unmatched) |
|---|---|---|---|---|
| none | 0.43-0.58 | 0.31-0.40 | 0.25-0.31 | 0.11-0.15 |
| 8 | 0.61-0.77 | 0.41-0.53 | 0.30-0.36 | 0.065-0.075 |
| 64 | 0.84-0.93 | 0.56-0.74 | 0.29-0.36 | 0.033-0.035 |
| whole | 0.94-0.96 | 0.82-0.88 | 0.22-0.24 | 0.025-0.026 |

The last column is an unmatched diagnostic, not Stage A's matched advantage; its references (random pairs 0.011, pairs
sharing the current token 0.295) were printed by the script (`/var/tmp/tm_local_pilot2.log`), not stored in the JSON.
In the earlier run (`exp1/local_pilot_v1.json`), 75-84% of whole-prefix neighbours lay within 32 positions in the same
text. **What it shows:** with memory, nearest neighbours are usually nearby in the same text; we have not measured the
dense walk or its modes on these inputs. **What it changed:** the experiment measures neighbours first, under a small
set of accounting variants (section 4), before paying for modes. Memory 2 is added as interpolation (the rise from none
to 8 is 34-51% of the rise to the whole prefix), not because of a finding.

**Design choices.**
- One forward pass per model, everything stored (21 GB): every kernel and control is recomputed without the models
  [derived].
- Every post-burn-in token enters the census; the all-texts spectral graph of Stage B uses only the final 512 tokens
  per text [derived from GPU memory, section 6].
- Geometric counts for every token; the expensive prediction readout on a prespecified, stratified sample shared by all
  candidates and both models [derived: about 3.5 billion directed neighbour pairs exist in all].
- Stage B's graph is screened in Stage A on the same tokens (final 512 per text) and the same dense conditional rows it
  will use, so a candidate is not chosen on a different graph from the one solved.
- Text-only adversaries (random vectors per token id; each model's embedding output): Stage A reports how much their
  neighbours overlap the models'; Stage B projects both models off the same tested text-feature span. This removes
  agreement inside that span only, not every surface statistic.

**Numerics.**

| Step | Chosen | Evidence |
|---|---|---|
| Memory kernel | exact recursion along diagonals of each Gram block, streamed, float64 | matches the explicit feature map to 1.4e-13; streamed top-256 sets identical to dense [here] |
| Stage A cost | about 1.4·10^15 float64 flops of Gram products, then masks, tie-aware top-64, suffix rows, matching and I/O | flops [derived]; wall time and cost [pilot] |
| Stage B graph | dense participation-ratio rule and `spacetime.py`, ported to CuPy, the copied CPU code as oracle | the dense solver ran at N = 8,000 × 12 layers [sibling]; at N = 1,792 × 8 the dense CPU path took 90 s to build the graph, 15 s to calibrate and 9 s to solve, one thread [here] |
| Not chosen: truncated rows | keeping each row's nearest neighbours only | on a synthetic graph at N = 1,792, keeping 1,024 neighbours loses 3.25% of the conditional mass on average yet moves t* about 15% and the 16-mode space by 0.19 (1 - mean cos²) [here] |
| Stage B size | 16,384 tokens | fits one 80 GB GPU by calculation [derived]; the fit and the cost are pilot questions [pilot] |

**Costs** [derived; the pilots replace them]: Stage A about $6.80 under a hard $8 cap enforced by reservations (section
11). Stage B about $21, approved separately with Stage A's results.

## 1. Questions

| # | Stage | Question | Could come out |
|---|---|---|---|
| Q1 | A | Under each memory and accounting, are a token's nearest neighbours in other texts, its own text far away, or its own text nearby? | memory makes neighbours local, or not; accounting changes that, or not |
| Q2 | A | On a fixed query sample, do neighbours share the model's top-20 next-token predictions more than controls matched on text, current token and distance? | more, less or the same, by memory |
| Q3 | A | Do the models' neighbours differ from what text alone gives (two text-only adversaries)? | overlap high or low |
| Q4 | A | Over what distances do the models attend directly, and how long do residuals stay similar? (descriptive) | reference scales only |
| Q5 | B | For the local kernel and the selected memory construction, do the two models share through-depth structure outside the tested text features, and is it more with memory? | four outcomes, section 8 |
| Q6 | B | Does a text's picture change when other texts are in the graph? (two focal texts, illustrative) | stable or reorganised |

## 2. Models, tokens, extraction

| | Pythia | OLMo |
|---|---|---|
| Id, revision | `EleutherAI/pythia-1b`, `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` | `allenai/OLMo-1B-0724-hf`, `d7cbab742d80589e714b1a2d7f838dcd21cbe143` |
| Shape | 16 blocks, width 2,048 | 16 blocks, width 2,048 |
| End-of-text token | id 0 | id **50279** (id 0 is `|||IP_ADDRESS|||`; 50277-50278 are PII placeholders) |
| Training data | the Pile (test split held out) | Dolma 1.7 (not guaranteed held out) |

Ordinary text gets the same ids from both tokenizers; each model reads its own separator at position 0. **Token
assertion (fails closed):** documents containing `<|endoftext|>`, `<|padding|>`, the three placeholder strings or NUL
are rejected; both tokenizers give identical ids for each accepted text; no text id is in {0, 1, 50277, 50278, 50279};
every probe id has the same `convert_ids_to_tokens` string under both. Display decoding passes
`clean_up_tokenization_spaces=False`.

**Forward pass.** float32, `model.eval()`, eager attention, TF32 off, one context per batch, `transformers==4.57.6`,
`torch==2.5.1` (nearest working pair pinned and recorded if OLMo fails). Residuals by hooks on each decoder layer
(block outputs; hidden state 16 is before the final norm), asserted equal to `output_hidden_states` for layers 1-15;
hidden state 0 is the embedding output. **Causal check (hard):** context 0, and context 0 with positions 1,024-2,047
replaced by a suffix whose token at 1,024 differs (found by search, recorded); residuals at positions 0-1,023 agree to
1e-5 relative at every layer and differ by more than 1e-3 at 1,024. A truncated run is a recorded diagnostic only.

**Saved by extraction** (so Stage A is verifiable on its own and later choices need no rerun):
- **Provenance:** model, tokenizer and library versions actually loaded; eval, dtype, attention and TF32 settings;
  the hook-equality and causal-check results, including the replacement suffix.
- **Activations:** float32 (17, 2,048, 2,048) per context, all hidden states at every position, 36 contexts.
- **Predictions:** per position, the top-20 ids and their full-softmax probabilities, the entropy of the full
  distribution and the retained top-20 mass. Exact full distributions can be regenerated from hidden state 16 and the
  pinned final norm and output head, so logits are not stored.
- **Attention summaries** (eager probabilities, never stored whole): per document, layer and head, and per query range
  (positions 256-1,535 and 1,536-2,047): the sum of sink mass A[i,0], the sums Σ_{j≥1} A[i,j](i - j) and
  Σ_{j≥1} A[i,j], and the query count, so per-head means and their medians are recomputable.

## 3. Probe, node sets and query samples

**Corpus** (owner: anything in the Pile is fine, all text shown). `monology/pile-uncopyrighted` revision
`3be90335b66f24456a5d6659d9c8d208c0357119`, `test.jsonl.zst` (337,771,483 bytes); `meta.pile_set_name` is the
**subset**. **Selection:** at least 95% ASCII, at least 2,048 Pythia tokens, no forbidden strings; shuffled with
`default_rng(0)`; the context is a document's first 2,047 tokens. Reject if whitespace-only tokens exceed 30%, more than
10% of positions u ≥ 32 repeat an earlier 32-gram of the context, any 32-gram occurs in an accepted document, or the
subset already has 4 texts. The first 32 accepted are the **texts** (random order, at least 8 subsets); four more make
the stitched controls (section 9.2). The page calls this **a capped sample of the beginnings of long, mostly ASCII Pile
test documents**. **Recorded per document:** file index, SHA-256 of the text, subset, the accepted ids including each
model's separator id; per stitched text, its segment schedule and, for every position, the source document and source
position.

**Positions.** Input `[that model's end-of-text] + context`, positions 0..2,047; position 0 (constant, attention sink)
is excluded from everything (centring, memory sums, nodes); a node's history includes every position from 1. Tokens are
identified by the immutable pair (text, position).

| Set | Tokens | Size | Used by |
|---|---|---|---|
| census pool | positions 256..2,047 of the 32 texts | 57,344 | Stage A: queries and candidates |
| suffix | positions 1,536..2,047 of the 32 texts | 16,384 | Stage A screen; Stage B all-texts graph |
| query sample Q | per text 64 positions in the suffix and 32 in 256..1,535, drawn with `default_rng(21)` | 3,072 | Stage A screen (suffix part) and prediction readout |
| audit queries | per text 1 position drawn from that text's 64 suffix queries in Q, `default_rng(22)` (so every audit query has production matches and suffix-screen records) | 32 | Stage A audit |
| one-text graph | positions 256..2,047 of one text | 1,792 | Stage B |
| matched one-text graph | the suffix of one text | 512 | Stage B |
| companion graphs | the suffix of focal text c (0 or 1) with that of 3, 15 (H1) or a disjoint 15 (H2) other texts | 2,048; 8,192; 8,192 | Stage B |

Q, the audit queries and the companion sets (`default_rng(100 + c)`) are drawn by `probe.py` and stored in
`probe/probe.json` before any kernel is computed.

## 4. Similarity

**Sites.** L = 8 hidden states ℓ ∈ {2, 4, ..., 16}, the same in both models.

**Centrings answer different questions** and are shown side by side, neither as the corrected version of the other:

| Centring | Token vector | Question it answers |
|---|---|---|
| corpus mean (C) | x_u - μ_ℓ, μ_ℓ the mean over the 32 texts, positions 1..2,047 | which passages does the state at this token resemble, as the model holds it? |
| own-text mean (O) | x_u - μ_{ℓ,c}, μ_{ℓ,c} the mean of text c over positions 1..2,047 | a retrospective within-document contrast: which passages resemble it once each whole text's average is removed? It uses tokens after the query, removes genuine topic information along with offsets, and gives two positions of the same text a covariance of -Σ/T (Σ the token covariance, T the text length) that positions in different texts do not share |

**Exclusion masks are rules, not findings:**

| Mask | Eligible neighbours of token i | What it imposes |
|---|---|---|
| none | every other token of the pool | nothing |
| window | no token of the same text within w = 64 positions (the Theiler window of recurrence analysis; the memory kernel is a weighted recurrence plot) | the near share is **zero by definition**; it forbids short offsets but does not make two long histories independent. The page shows the short-range mass the mask removed (section 5) |
| other texts | tokens of other texts only | the other-text share is **one by definition**; shown in Experiments only, never solved in Stage B, disabled for a stitched text analysed alone |

Both centrings depend only on the token's own text, so the similarity of two tokens never depends on which graph they
are in.

**Readings.**

| Reading | y_u | Similarity | Distance d |
|---|---|---|---|
| cosine (each token counts equally) | centred, then unit length | ĉ = K(t,t')/sqrt(K(t,t)K(t',t')) | 1 - ĉ |
| dot (loud tokens count more) | centred | K(t,t') | [(K(t,t) + K(t',t'))/2 - K(t,t')]/s, s = mean K(t,t) over the census pool |

**Memory kernel** (the owner's definition). With G(u,u') = ⟨y_u, y_u'⟩ and y_u = 0 for u ≤ 0,
K_a(t,t') = Σ_{s≥0} a^s G(t-s, t'-s) = G(t,t') + a K_a(t-1, t'-1): two pasts laid side by side **aligned at their ends**
and compared lag by lag (not the dot product of running means). Memory m ∈ {none (a = 0), 2 (a = 0.5), 8 (0.875),
64 (0.984375), whole (a = 1)}, a = 1 - 1/m. For the primer: chance agreement persists along a diagonal (K(t,t') and
K(t+1,t'+1) share history, correlation a for unrelated noise); at the whole prefix pasts of different lengths are
compared over the shorter, so similarity depends on position even for structureless text; for memories up to 64 after
burn-in the readings differ approximately only in token normalisation, at the whole prefix also in prefix
normalisation.

**Text-only adversaries**, same tokens, centrings, readings, memories and masks: (i) **random token vectors**, a fixed
Gaussian vector in dimension 2,048 per token id (`default_rng(7)`), model-free; (ii) **embedding output**, each model's
hidden state 0.

**Exact computation** (`exp1/similarity.py`, extended). Diagonals K(t,t) first. Then per text c: the Gram of its 2,047
tokens against all 65,504 tokens of the 32 texts (one float64 matmul), the recursion `K[t, 1:] = G[t, 1:] + a *
K[t-1, :-1]` within each text's column block for all memories at once, distances from the diagonals, the three masks.
The closed form a^t Σ a^{-u} g(u) is not used (it overflows). Checks (fail closed): no token with ‖x - μ‖ < 1e-6 ×
median; smallest off-diagonal model distance above 1e-9.

**Ties (production path).** Two candidates are tied if their distances differ by at most 1e-12 relative. The top-64 set
is every candidate strictly closer than the 64th distance d_64 plus the tied candidates at d_64 needed to reach 64,
chosen by (text, position); stored with it: d_64, the number of candidates tied at d_64 (t_all) and how many were taken
(t_in). Every overlap between two neighbour sets is the **expected overlap under uniformly random tie-breaking**,
E|A ∩ B| = Σ_j P(j ∈ A) P(j ∈ B), with P = 1 strictly inside, t_in/t_all at the cutoff, 0 outside. This matters for the
adversaries at memory none, where hundreds of tokens can share a current token. `torch.topk`, which leaves tie
membership unspecified, is replaced by this rule.

Expected overlap is a **descriptive** statistic: when one set's cutoff ties contain the other set, it is small although
every neighbour is shared (a model picking 64 of 1,024 tied occurrences of one token against an adversary tying all
1,024 gives 64/1,024 = 0.06). The selection conditions of section 5 therefore use the **maximum overlap compatible with
the cutoff ties**. With A_s, B_s the strictly-inside sets, A_t, B_t the tied sets at each cutoff and a, b the numbers
still to take from them, and c1 = |A_s ∩ B_t|, c2 = |A_t ∩ B_s|, c3 = |A_t ∩ B_t|:
max|A ∩ B| = |A_s ∩ B_s| + max over z = 0..min(c3, a, b) of [z + min(c1, b - z) + min(c2, a - z)].
The four counts |A_s ∩ B_s|, c1, c2, c3 are computed and stored while the rows are live, for every pair of settings
whose overlap is used (each setting against memory none and against each adversary); full tie identities are stored for
the audit queries.

## 5. Stage A: the neighbour census

**Settings.** 2 models × 2 readings × 2 centrings × 5 memories × 3 masks × (8 sites + 2 adversary inputs). One kernel
pass per (model, reading, centring, input) serves all memories and masks: 80 passes.

**A. Geometric census, every token of the census pool, every setting.**

| Quantity | Definition (means over queries; chance computed exactly from eligible counts) |
|---|---|
| origin shares | fractions of the top-64 in other texts / same text more than 64 positions away / same text within 64 |
| position gap | median |Δ position| of same-text neighbours |
| same current token | fraction of the top-64 with the query's current token id |
| adversary overlap | expected overlap with each adversary's top-64 (same memory, reading, centring, mask), divided by 64 |
| overlap with local | expected overlap with the top-64 at memory none (same reading, centring, mask), divided by 64 |

**B. Stage-B-matched screen** (while the distance rows are live; no eigen-solve, nothing N × N stored). Queries: the
2,048 suffix queries of Q; candidates: the 16,384 suffix tokens (the Stage B all-texts graph). For each query row, the
dense participation-ratio conditional of Stage B: q(j|i) ∝ exp(-β_i(d_ij - min_j d_ij)) over eligible j ≠ i, β_i by
bisection so that 1/Σ_j q(j|i)² = 64 (`_cond_rows` arithmetic; rows whose target is unattainable because of tied
minima are counted and excluded). Reported: **top-64 origin shares** and **conditional-mass origin shares**
Σ_{j in origin} q(j|i) on these rows; for the window mask, the **short-range mass removed**: the conditional mass the
unmasked row (calibrated separately) puts on same-text tokens within 64 positions.

**C. Prediction readout on the query sample Q** (sites 4, 8, 12, 16; every setting of the 8 model sites). For each
query i in Q and each neighbour j in its census top-64 (ties expanded as in section 4), a control j' drawn with a keyed
seed (hash of model, setting, i, j) from the census pool with j's text, j's current token id and, if j is in i's text,
the same bin of |Δ position| (1-8, 9-64, 65-512, over 512), excluding i, j and i's other neighbours. o(i,j) =
Σ_{v in both top-20 sets} min(p_i(v), p_j(v)); pair advantage = o(i,j) - o(i,j'). **Strata:** neighbour origin (3) ×
frequency of j's token in the pool (at least 20 occurrences, or fewer): 6 strata. A pair without an eligible control is
**unmatched** (reason recorded: no same-token token in that text, empty lag bin, all candidates excluded). A stratum is
**covered** if at least 50% of its pairs and at least 200 pairs are matched; an empty stratum is **absent** and carries
no weight. **Concordance score** = the mean over covered strata of each stratum's mean advantage, with equal weights;
a setting is **ineligible** for ranking if any non-absent stratum is uncovered or overall coverage is below 50%. Stored
per document: per stratum, the sum of advantages, matched and unmatched counts by reason, and the retained top-20 mass;
the bootstrap (200 draws of documents) resamples these contributions. Also reported: top-1 match beside its matched
control (readable form). The model's own predictions are used for each model.

**D. Stitched controls** (each stitched text analysed alone; masks none and window; centring C). Same-source share of the
top-64 against its **position baseline**: for each signed lag Δ, b(Δ) = the mean, over all query positions u of that
stitched text with u and u + Δ both nodes, of 1[source(u) = source(u + Δ)] under its segment schedule; the baseline of
query i is the mean of b(j - i) over its neighbours j. (Conditioning on i as well would leave one position per lag, the
neighbour itself, and make observed minus baseline identically zero.) Observed minus baseline is the share beyond
position.

**E. Synthetic census.** The variants of section 9.3 run through A, so the page can show what a planted regime looks
like as neighbours.

**Stage A's result stands alone.** It says, for two real models, what memory and accounting do to a token's
neighbourhood and whether neighbours are prediction-concordant, verifiable by clicking any token.

### Choosing Stage B's construction (fixed now)

Candidates: memory ∈ {2, 8, 64, whole} × reading × centring × mask ∈ {none, window}: 32. On the **Stage-B-matched
screen**, a candidate **passes** if for both models at 5 or more of the 8 sites:
1. its conditional mass from other texts is at least 0.25;
2. for the window mask, the short-range mass removed is at most 0.5 (the rule did not remove most of the row);
   for no mask, the conditional mass within 64 positions in the same text is at most 0.25;
3. its top-64 overlap with memory none (same reading, centring, mask) is at most 0.5;
4. its overlap with each adversary is at most 0.5.

In 3 and 4 the overlap at a site is the mean over suffix queries of (maximum overlap compatible with ties)/64
(section 4), so a candidate cannot pass through tie-breaking; the expected overlap is shown beside it.

**Primary memory construction M:** among passing candidates, the least intervention first (C/none, then C/window, then
O/none, then O/window), then the cosine reading, then the memory with the smallest adversary overlap (the maximum over
models, sites and the two adversaries of the site value in 4), then the shorter memory. Its
**local counterpart L** is memory none with M's reading, centring and mask. **Stage B solves** R (memory none, cosine,
C, no mask: DepthEmbedding's construction), L (if different from R) and M. **At most one additional example X:** among
passing candidates other than M whose concordance is eligible, the one with the highest concordance score (candidates
are compared on the strata covered in all of them, so absent strata cannot change the weights) if its
bootstrap 5th percentile is above 0; X is described as "screened for prediction concordance" and treated as descriptive
in Stage B (no difference test). The concordance score is never used to choose M and never presented as validation.
**If nothing passes,** Stage B is not recommended (see the first paragraph); if the owner still wants it, it solves R and
the reference construction (cosine, C, window, memory 64) labelled "did not pass the screen".

## 6. Stage B: graph rule, operator and solver (provisional)

**Graph rule:** participation ratio, k = 64, dense and exact (`kernels.py`: `_cond_rows`, `ea_kernel(renyi=2)`,
`sinkhorn_sym`, from DepthEmbedding `invert/dm.py` at `8dd7a66`), on eligible pairs (excluded pairs get d = +∞, hence
zero weight); W = (q + q^T)/2; symmetric Sinkhorn (float64, 1e-11); M = (I + P)/2; no truncation, no floor;
`ea_kernel` is passed C = 1 - d. Achieved participation ratio is checked per row. Participation ratio rather than the
owner's suggested Shannon perplexity: before Sinkhorn it caps any conditional weight at 1/sqrt(k) = 0.125 where a
Shannon perplexity of 64 allows about 0.6, and Shannon gave two-image joint modes in DepthEmbedding [sibling]. Shannon
and union 15-nearest-neighbours are compared once on R's all-texts graph, off the critical path, reported only.

**Operator:** L_t = blockdiag((I - M_ℓ)/c_ℓ) + t(D_path ⊗ I) on functions with mean-zero slices; c_ℓ = mean of the first
8 nontrivial 1 - λ_k of M_ℓ, reusing the full site spectra; t = t* (`balance_direct`), where the first between-layer
eigenvalue t·4 sin²(π/(2L)) meets the first eigenvalue on mean-zero slices (two eigenvalues, not two energy terms); 24
modes computed, 20 checked, 16 read. Every temporal formula uses L.

**GPU route:** `spacetime_gpu.py` and `kernels_gpu.py` are the copied CPU modules with the array module switched to
CuPy; ARPACK stays in scipy on the CPU, one vector moving per operator application. The CPU modules stay unchanged as the
oracle. **G1 (hard, pre-B):** on the operator fixtures of section 9.1, a masked nonuniform synthetic graph and the
one-text graph of text 0 under L, GPU and CPU agree: q, W and P entries to 1e-12; c_ℓ, t*, first 24 eigenvalues to 1e-10
relative; complete eigenvalue clusters (with guard modes) to principal-angle sine 1e-8. One-text and matched graphs run
on the CPU oracle itself.

**Memory at N = 16,384, float64** [derived; one matrix 2.15 GB]: graph building, one site at a time, about 32 GB
transient plus up to 17.2 GB of finished walks; calibration: walks 17.2 + the site eigenvectors, which the copied
`site_eigs` holds twice while stacking (34.4) = 51.5 GB before workspace; solve: walks + inverses + set-up about 43 GB.
Device-level high-water is measured in the B-pilot.

**Acceptance per solve:** residuals of the first 20 modes below 1e-8; |λ_1/(t*·4 sin²(π/(2L))) - 1| < 1e-6; occupancy
and recurrent strengths sum to K; finite outputs; Sinkhorn row error below 1e-10. Stored and shown: the first 12 site
gaps 1 - λ_k per layer and joint gaps at K = 12, 16, 20; a construction whose site gaps reveal a near-disconnection that
sets the first displayed modes or the unit c_ℓ is not shipped as a winner, and the clusters' text composition is shown.

## 7. Stage B readings (provisional)

G_ℓ = N × K slices of the first K modes (K = 16 unless stated); X = [G_1 ... G_L]; R = X X^T (never formed).
**Alignment** A(R_1, R_2) = ‖X_1^T X_2‖²/(‖X_1^T X_1‖ ‖X_2^T X_2‖). **Common projection:** Z = text indicators (31
dimensions in the all-texts graph) and position harmonics cos(π j (t - t_0 + 1/2)/n) with wavelength at least 224 tokens
(J = 4 for n = 512, 16 for 1,792), residualised against texts; F = indicators of every current-token id plus the first
32 nontrivial eigenvectors of each adversary walk. For comparing **models within one construction**, both are projected
off Z + F of that construction; for the **difference between M and L**, both sides are projected off the **union** of
the two constructions' spans (section 16). Reported for every remainder: retained energy, effective rank (tr R̃)²/tr(R̃²),
exhaustion dim/N, and alignment against a reference calibrated with the projection reapplied (section 16).

**Q6:** on the focal text's 512 tokens, alignment between the matched one-text, full one-text (restricted), companion
(4, H1, H2) and all-texts pictures, for R and M; focal occupancy and retained signal reported separately (no signal
below 0.05 retained); realised radius Σ_j q(j|i) d_ij per graph. Illustrative, not an estimate. **Per mode:** depth
weight, shares, spread 1/Σ_i w_i², two-token mass.

## 8. Stage B decision (provisional)

M and L are each **supported** if, for both models, in the all-texts graph and the median one-text graph: acceptance
passed, no disqualifying near-disconnection, retained energy ≥ 0.1, effective rank ≥ 4, exhaustion ≤ 0.5, cross-model
agreement above the calibrated reference, and the same at K = 12 and 20. **Increment:** M's cross-model agreement minus
L's, both on the union projection, with a paired document bootstrap. **Outcomes:** (i) M supported with a demonstrated
increment (5th percentile above 0): the viewer ships M with L; (ii) both supported, no demonstrated increment; (iii) only
L supported; (iv) neither. The viewer always ships M and L, labelled by outcome, and names the failed condition. The
claim is shared descriptive geometry outside the tested text features.

## 9. Controls

### 9.1 Hard checks

**Before Stage A (on the VPS, free; then repeated on CUDA at the start of the A-pilot):**
- **Kernel:** streamed kernel against the explicit feature map and a direct double loop, both readings, all memories
  (1e-12); the diagonals.
- **Masks:** on a toy pool with known texts and positions, the three masks give exactly the eligible sets by hand; the
  other-texts mask is refused for a single stitched text.
- **Ties:** a pool with planted exact ties at the cutoff (hundreds of equal distances): top-64 membership follows the
  rule of section 4, t_all and t_in are right, and expected and maximum overlaps equal their hand values, including
  the case of 64 picked from 1,024 tied against a set tying all 1,024 (expected 0.0625, maximum 1: fails condition 4);
  same under all three masks.
- **Matching:** controls have j's text, token id and lag bin; never i, j or i's neighbours; replaying a keyed seed gives
  the same control; unmatched reasons are counted correctly on constructed cases.
- **Statistics:** origin shares and chance levels on a toy pool by hand; per-document contributions sum to the totals;
  the bootstrap from contributions equals one recomputed from pairs on a small case.
- **Suffix screen:** conditional rows equal `kernels._cond_rows(renyi=2)` on the same distances (1e-12); unattainable
  targets from tied minima are detected.
- **Stitched baseline** on a constructed schedule by hand, including a case where observed and baseline differ; the
  token assertion on two short texts.

**Before Stage B (moved from v3; pre-B gate with G1):**
- **Analytic operator fixture,** memory none, L ∈ {4, 8} identical sites: two groups of n = 48 tokens with vectors
  e_i + c·g_A (or g_B), all orthonormal, so each conditional row has n - 1 equal near and n equal far entries; a
  participation ratio of 64 is feasible because n - 1 < 64 < 2n - 1. The two row weights are solved from the
  target; Sinkhorn is trivial by symmetry; the group indicator is an eigenvector of M of known eigenvalue, giving scaled
  energy h. Required to 1e-10: the **ground state** is the indicator with a **constant** depth envelope (j = 0) at
  eigenvalue h; the first non-constant harmonic, cos(π(l + 1/2)/L), sits at h + t·4 sin²(π/(2L)).
- **Near-cut fixtures** on directly constructed doubly stochastic walks with a prescribed crossing weight ε: two groups
  (one fragile direction: the first gap is about ε and the first mode is the cut); nine groups (eight fragile gaps,
  setting c_ℓ): both are flagged by the gap report.
- **Equivariance:** permuting node labels after features permutes q, P and the joint modes, compared as whole clusters.
- **Decision fixture:** constructed inputs with literally zero retained signal fail the energy gate; each of the four
  outcomes is produced by a constructed case.

### 9.2 Stitched real texts (expected answers, not gated)

Two source pairs (A, B), (C, D), each from different subsets; each pair makes two texts alternating segments of random
length, **long** uniform in [384, 640] and **short** in [48, 80], ids concatenated. Stage A: section 5 D. Stage B
(one-text graphs): captured share of the A/B indicator, the position-only baseline (its share inside the harmonics) and
the share beyond position. Expected: share beyond position above zero at none, 2 and 8; lower at 64 than at 8 for the
short schedule; no prediction at the whole prefix.

### 9.3 Synthetic (measurements, reported, not gated)

8 texts of 1,024 positions, width 64, 4 sites, 5 seeds. x = α_ℓ r_{z(u)} + β_ℓ e_{w(u)} + γ g(u) + ε with a hidden
4-state regime (switch probability 1/128, absent at site 0), Zipf tokens, an optional position drift, Gaussian noise.
Variants: strong (α = 1), planted (α = 0.25), drift (γ = 0.5), two independently drawn noise fields. Stage A reports
their census; Stage B reports captured regime share against a same-persistence reference, site-0 leakage, and the
noise fields' alignment by position quartile against redrawn fields on the same index geometry.

## 10. Direct attention distance and residual autocorrelation (Q4, Stage A)

From the attention sums of section 2: per model, layer and head, the mean sink mass and mean direct attention distance
over query positions 256..2,047; per layer the median and interquartile range over heads. Per model and site: ρ(δ) =
mean cosine of centred unit vectors δ apart within a text (δ ≤ 1,024), background ρ_∞ between texts, and the
half-similarity lag (smallest δ with ρ(δ) - ρ_∞ ≤ (ρ(1) - ρ_∞)/2; censored "> 1,024" with a count when absent). Neither
is an effective context length; the page draws them beside the memory grid as reference scales.

## 11. Compute plan

Modal app `tm-exp1`, Volume `tm-exp1`, per the `modal` skill. Rates (modal.com, 2026-11-23): A100-80GB $0.000694/s,
L4 $0.000164/s, CPU $0.0000131 per physical core-second, memory $0.00000222 per GiB-second. Hourly: A100 + 8 cores +
64 GiB **$3.39**; L4 + 4 cores + 32 GiB $1.04; 16 cores + 64 GiB $1.27; 2 cores + 8 GiB $0.16. Each attempt adds about 2
minutes of start-up.

**Spending control (Stage A cap $8, hard).** A ledger on the Volume records every attempt's billed seconds × rate.
Before any unit starts, the coordinator **reserves** its timeout × rate; it refuses to start a unit if spent + reserved
would exceed $8. Failed attempts are charged. Settings run in priority order, so the cap can only remove the last ones:
(1) cosine reading, both centrings; (2) dot reading, corpus mean; (3) dot reading, own-text mean. Settings dropped by
the cap are listed on the page. The readout and the Stage A audit (with start-up and one retry each) are reserved
**before** any setting beyond (1) is dispatched, so the cap can never remove the verification. The A-pilot times one
complete explicit-lag audit row; if 32 full rows do not fit the audit's reservation, fewer rows are audited (never
fewer models or masks) and the page says how many.

**Stage A ledger** [derived; the A-pilot replaces it]:

| Unit | Count | Hardware | Work | Time each | Spend |
|---|---|---|---|---|---|
| U0 probe | 1 | 2 cores, 8 GiB | download, filter, tokenise both ways, assert, query samples | 20 min | $0.06 |
| U1 extract | 2 | L4 | 36 forward passes, hooks, predictions, attention sums, causal check | 25 min | $0.94 |
| A-pilot | 1 + 1 | A100; 16 cores | CUDA repeat of the census tests; one shard (Pythia, cosine, C, site 8, all memories and masks): pass, screen, list I/O; then the readout, statistics and bootstrap for that shard on CPU | 15 min; 10 min | $1.21 |
| UA census | 2 (model) | A100 | the remaining 39 passes per model: geometric census, suffix screen, tie records, list output (about 35 GB) | 25 min | $3.05 |
| UA readout | 1 | 16 cores, 64 GiB | prediction readout (about 94 million pairs), statistics, bootstrap, stitched and synthetic census, timescales, viewer data | 40 min | $0.89 |
| UA audit | 1 | 16 cores, 64 GiB | section 12.4 | 30 min | $0.68 |

Projected **about $6.83**; cap $8, which leaves room for one failed attempt of a small unit but not of a census unit.
**Stop rule after the A-pilot:** projected total = spent + (measured shard cost × remaining shards) + readout and audit
(scaled from the pilot) + 10%; if above $8, priority group (3) is dropped, then (2); if still above, stop and report the
pilot's numbers. Wall-clock about 2 hours.

**Stage B ledger** (provisional, for the second approval): G1 and B-pilot (one all-texts solve, device high-water,
one CPU one-text group benchmarked) $3.5; all-texts graphs for R, L, M (and X) × 2 models, 15 min each, $5.8-7.7;
companions for R and M, $4.5; one-text groups on CPU, $2.3; synthetic $1.2; recipe comparison $0.8 (dropped first if
short); analysis $1.0; operator audit $1.0. About $21, cap to be set at approval.

**Volume.** `/hf`, `/probe`, `/acts`, `/pred`, `/attn`, `/census/<model>/<reading>_<centring>/` (per input and
memory and mask: top-64 lists with tie records, d_64, eligible counts by origin; suffix-screen rows' origin masses and
removed short-range mass; about 35 GB), `/readout` (per-document stratum contributions, control ids or keyed seeds,
unmatched reasons), `/audit_a` (expected values saved before distance blocks are discarded), `/ledger`, `/b` (Stage B),
`/analysis`, `/site`. Only `/analysis`, `/audit_a`, `/ledger` and `/site` are fetched.

## 12. The site

Public repo `CharlesR-W/TransformerModes`, GitHub Pages; machinery copied from DepthEmbedding (`build_site.py`,
`xsite/xlib.py` facts and `{{fact:...}}` tokens, `check_fragment.py`, `check_browser.py` adapted, live charts, palette,
CSS). Tabs: **Viewer**, **Experiments**, **Method**; a one-line banner (both models, 16 blocks, width 2,048, sites
marked). Stage A publishes the viewer (neighbours) and entries E1-E3; Stage B adds modes and E4-E5.

### 12.1 Viewer

Every offered choice has data behind it.

| Control | Values (Stage A) | Added by Stage B |
|---|---|---|
| Text | 32 texts + 4 stitched | |
| Model | Pythia / OLMo | |
| Layer | 4, 8, 12, 16 | (modes at the same four layers; the depth bar shows all 8) |
| Memory | none, 2, 8, 64, whole | |
| Accounting | corpus mean; corpus mean + window; own-text mean + window (cosine reading) | M's and L's exact setting if not among these, including the dot reading |
| Colour | where the neighbours come from | mode 1-16 for R, L and M |
| Graph | | one-text / all-texts (all-texts disabled for stitched texts) |

Other-texts-only, the remaining accounting choices and the dot reading are shown in Experiments, not the viewer; the
caption under the controls says so. A stitched text offers only the corpus-mean accountings without the other-texts
mask (the only ones computed for it, section 5D). The short-range mass a window removed exists only for the sampled
suffix queries and is shown as a labelled aggregate, never as a per-token value.

- **Text** inline with whitespace kept. **Neighbour colouring:** each token's share of its **64** neighbours from
  other texts (sequential palette 0-1), burn-in grey; for the window accounting, a
  marker shows how much short-range mass the mask removed. **Mode colouring** (Stage B): the mode's unit slice at the
  layer (diverging, range fixed across layers); faded with a note when the depth weight there is under 1%.
- **Token card** (click): the **8 nearest** eligible tokens under the current setting (value, text, position; tied
  cutoff marked), each with the last up to 24 tokens ending there, right-aligned under the clicked token's own, with the
  share of memory weight those 24 tokens carry (1 - a^24: 100% at 2, 96% at 8, 31.5% at 64; 24/t at the whole prefix);
  the token's origin shares over all 64; its top-3 predictions beside its neighbours'.
- **Mode card** (Stage B): depth weights, stacked shares with chance ticks, spread; on any change of Model, Memory,
  Accounting or Graph: "mode numbers and signs are local to this graph" with the alignment between the two pictures on
  this text's shared tokens.
- **Details** (one disclosure): strip chart of the coloured value against position; Stage B adds a scatter and the
  worked token (raw nearest beside the largest walk weights; the local block equation evaluated in the browser).

**Data** (lazy per text and setting): texts 36 × 50 KB; neighbour lists for 2 models × 4 layers × 5 memories × 3
accountings = 120 settings, 8 neighbours per token as uint16 index + float16 value (cosine distances, order one) plus a
byte of origin share and a tie flag: 36 texts × 1,792 tokens × 120 × 34 bytes ≈ **263 MB**, about 61 KB per text and
setting; Stage B adds up to 2 settings × 2 models × 4 layers of lists (≈ 18 MB) and int8 mode slices at 4 layers
(≈ 40 MB).

### 12.2 Primer (top of the Viewer tab, collapsible, one screen)

**Position** (the first 255 tokens are burn-in). **Layer** (block outputs 4, 8, 12, 16 of 16). **Memory** (two pasts
aligned at their right ends with fading weights; chance agreements persist along the alignment; the whole prefix depends
on position even for structureless text). **Accounting** (corpus mean: the state as the model holds it; own-text mean: a
retrospective contrast that removes each whole text's average, topic included; the window: a rule that forbids
neighbours within 64 positions, so "nearby" is zero by construction). **Neighbour origin** (other texts, same text far,
same text near). **Checking** (click a token: its neighbours are listed as text, with the stretch of past each
comparison weighs). Stage B adds: the two graphs, modes (numbers and signs local to one graph), coupling, the share bar.

### 12.3 Experiments entries

| Entry | Lead figure: x · y · facets · uncertainty | The sentence a reader should be able to say |
|---|---|---|
| E1 (A) Who are a token's nearest neighbours under memory? | x = memory (none, 2, 8, 64, whole) · y = share of the 64 neighbours, stacked: other texts / same text far / same text within 64 · facets: columns = corpus mean, own-text mean (side by side, different questions), each without and with the window (window columns hatched in the near band: "zero by rule", with the removed short-range mass printed); rows = models · site 8, cosine, census pool · whisker: document bootstrap on the other-text share. Supporting: the Stage-B-matched screen as conditional-mass shares; adversary overlap by memory | "Under the corpus mean, memory makes [x]% of a token's neighbours its own text, [y]% within 64 positions; the window removes [z] of the conditional mass; text-only vectors share [w] of the neighbours." |
| E2 (A) Are neighbours prediction-concordant beyond matched controls? | x = memory · y = concordance score (stratified top-20 overlap advantage) · series: accounting · facets: model · whiskers: document bootstrap; zero line; ineligible settings hollow and labelled. Supporting: coverage by stratum; top-1 match beside its matched control | "On the fixed query sample, neighbours share [more / less] of the model's predictions than matched controls as memory grows; coverage is [c]." |
| E3 (A) What does the screen select for Stage B, and why? | table, not a chart: the 32 candidates × the four conditions per model, the ordering that chose M, L, and X if any. Lead figure: E1's lead restricted to the suffix screen for M, L and R | "The screen chose [M] because [...]" or "No memory construction passed: [condition] failed for [...]; Stage B is not recommended." |
| E4 (B) Do the models share structure outside the tested text features, and is it more with memory? | rows R, L, M (X descriptive) · x = cross-model agreement of the remainder, calibrated reference marked · panels all-texts, median one-text | "[Outcome (i)-(iv)]." |
| E5 (B) Does a text's picture change with its company? | x = texts in the graph (1, 4, 16 H1, 16 H2, 32) · y = alignment with the 32-text picture on the focal suffix · texts 0 and 1 × R and M | "With disjoint companions the picture of text [c] agrees at [x]." |

The timescales of section 10 are a supporting figure of E1 (reference lines at 2, 8, 64); the stitched controls are
E1 tiles. The Method tab carries sections 0, 2-5 and 9-10, the subset table and the decisions.

### 12.4 How a reader can check there is nothing odd going on (Stage A)

- **Stage A audit (UA audit)**, numpy only, importing nothing from the pipeline: for the 32 audit queries, at sites 4
  and 12, both models, both readings and both centrings, recomputes the **complete** distance row against the census
  pool by the explicit lag sum from the activations (all memories), then the three masks, the top-64 with ties
  (membership, d_64, t_all, t_in), the suffix-screen conditional masses and removed short-range mass; replays the matched
  controls of those queries from their keyed seeds and checks every matching constraint; recomputes all census
  statistics from the stored lists for a 1% sample of tokens and the readout from stored contributions; and compares
  1,000 random pairs per pass with values the census saved before discarding its distance blocks. Tolerance 1e-9 on
  distances, exact on memberships and counts. Its report is fetched with the ledger; the Stage A site build fails if it
  is missing or failing. It states what it does not reproduce.
- Clicking any token shows its neighbours as text, the stretch of past compared and its predictions beside theirs.
- The selection table (E3) shows every number behind the choice for Stage B.
- `check_browser.py` drives headless Chrome over every control, a token click, Details and both viewports, failing on
  console errors or external requests; screenshots are inspected before deploying.

Stage B adds its own operator audit (rebuild one walk independently and check the block equation; recompute decision
inputs from slices), its G1 tiles and the worked token.

## 13. Code and tests

In `exp1/`: `similarity.py` (extended for masks, the tie rule and the suffix screen), `kernels.py`, `spacetime.py`,
`spacetime_stage.py`, `test_spacetime.py`, `test_numerics.py`; `graph.py` and `solve.py` (sparse, shelved). New for
Stage A: `probe.py`, `extract.py`, `census.py`, `readout.py`, `select_b.py`, `audit_a.py` (independent), `modal_app.py`
(with the ledger), `test_census.py` (section 9.1, pre-A). For Stage B: `spacetime_gpu.py`, `kernels_gpu.py`,
`stage_b.py`, `readings.py`, `decide.py`, `audit_b.py`, `test_stage_b.py` (pre-B fixtures). All tests run on the VPS
in under ten minutes; the census tests are repeated on CUDA in the A-pilot.

## 14. Decisions made on the owner's behalf

| # | Decision | Chosen | Alternative not taken |
|---|---|---|---|
| 1 | Staging | census first; modes only for what its screen supports | modes for every construction |
| 2 | Texts | 32, first 2,047 tokens, at most 4 per subset | windows inside documents; uniform among long documents |
| 3 | Census | every post-burn-in token, 64 neighbours, all settings | sampled tokens |
| 4 | Prediction readout | fixed stratified sample of 3,072 queries, four sites | every token (about 3.5 billion pairs) |
| 5 | Sites | 8 even layers (4 in the viewer) | all 16 |
| 6 | Memory grid | none, 2, 8, 64, whole | lengths derived from the models |
| 7 | Accounting | corpus or own-text mean × no mask, 64-position window, other texts | position-wise centring; windows scaled with memory |
| 8 | Stage B choice | geometric screen on Stage B's own tokens and rows, least intervention first; concordance adds at most one example | ranking by concordance |
| 9 | All-texts graph | final 512 tokens of 32 texts (16,384), dense | 16 × 1,024; truncated rows (measured drift) |
| 10 | Graph rule | participation ratio k = 64, fixed | Shannon perplexity (owner's suggestion) or k-NN as decisive |
| 11 | Coupling | t* only | DepthEmbedding's slider |
| 12 | Viewer | cosine, 4 layers, 3 accountings, 8 neighbours shown | every setting (over 1 GB) |

## 15. Deferred

- **Truncated rows.** `graph.py`, `solve.py` and their tests stay. Measured: keeping 1,024 of 1,792 neighbours moves t*
  about 15% and the 16-mode space by 0.19 on a synthetic graph, because participation-ratio rows have long, nearly flat
  tails when distances concentrate. Idea: calibrate on the exact dense row inside the streaming pass, keep the top
  neighbours exactly, and treat the dropped tail as a uniform low-rank term instead of refitting on the kept support.
- Model-derived lag weights (the "smart EMA" kernel); the recurrent random sketch; leave-one-text-out Nystrom
  (`ea_oos_rows` is copied); attention-weighted links across layers; a training-checkpoint axis; the model as a coupled
  axis; coupling robustness; a width sweep; windows scaled with memory; mixed readings; clipping loud tokens; k-means
  readouts; longer contexts; more models; a second memory construction paired with its own local counterpart.
- **Owner's TODO, not for now:** (a) a sentence-embedding (SentenceTransformers) or vector-database style baseline on
  the same contexts, "whatever the smart version of that is"; (b) the Hankel predictive-state construction of Golowich,
  Liu and Shetty, *Sequences of Logits Reveal the Low Rank Structure of Language Models* (CoherentModes notes: "lingen").

## 16. Open before Stage B approval

Recorded from review 3; each must be settled, with Stage A's numbers, before Stage B is requested.

1. **One memory construction, exactly paired** (finding 6): M with L as in section 5; M2 is cut; X is descriptive.
2. **Same projection on both sides of the difference** (finding 6): the union of M's and L's Z + F spans for the
   increment test, with retained energy and rank shown for both; check exhaustion and singleton-token fractions from the
   Stage A token tables before authorising; at memory none the adversary pictures are not purely token functions under
   own-text centring or the window, so L gets its own adversary eigenvectors too.
3. **Four outcomes** (finding 6), as in section 8, and wording "outside the tested text features" throughout.
4. **Calibrated cross-model reference** (finding 7): permutations or surrogate factors with the common projection
   reapplied after permuting; calibration checked on a text-only control and on independent synthetic fields; the null
   stated exactly. The `decide.py` hard test is deterministic (zero retained signal fails).
5. **GPU memory and solver checks** (finding 8): device-level high-water including the doubled eigenvector stack; G1
   with a masked candidate; achieved participation ratio per row, rejecting targets made unattainable by tied minima;
   the first 12 gaps reported, not a single flag.
6. **Timing** (finding 8): one end-to-end GPU unit and one CPU one-text group benchmarked (the measured dense CPU graph
   build is 90 s per 1,792-token graph on one thread); the audit included in the projection; the recipe comparison and
   the second focal text dropped first if short.
7. **Operator fixtures** of section 9.1 (pre-B) passing.

## 17. Departures from the brief

Each model reads its own separator (id 0 in Pythia, 50279 in OLMo); staged design with a neighbour census first; every
post-burn-in token in the census, the final 512 per text in the spectral graph; accounting variants answer the owner's
"possibly accounting somehow for that"; "smart EMA" is a descriptive measurement; identical tokens give identical
vectors only at the embedding output.

## 18. Responses to reviews

**Review 1:** noise null, winner rule, cost, primer, invariants, cutoff, recipe pilot, audit, mode numbers and Q5 all
accepted; the A/B ceiling superseded; publication answered by the owner. **Review 2:** 1 (common projection; margin
removed), 4 (analytic fixtures, L-parameterised formulas), 6 (retained signal, equal companions), 7 (matched controls),
8 (ledger, stage caps) accepted; 2, 3 and 5 dissolved by the dense design, their residues tracked below.

**Review 3** (`reviews/spec-v3-astra.md`):

| # | Finding | Response | Where |
|---|---|---|---|
| 1 | Analytic fixture wrong; near-cut infeasible | Accepted. Ground state = constant envelope, first harmonic tested separately; n = 48 satisfies n - 1 < 64 < 2n - 1; near cuts on directly constructed walks with 2 and 9 groups; operator fixtures moved to the pre-B gate; pre-A tests target the census code (kernel, masks, ties, matching, statistics, suffix rows) | 9.1 |
| 2 | Screen is not Stage B's graph; masks impose results | Accepted. Full census kept as the primary result; a Stage-B-matched screen on suffix queries and suffix candidates reports top-64 and dense conditional-mass origin shares; the window's near share and the other-texts share are labelled as imposed, with the removed short-range mass shown; selection uses the matched screen | 4, 5, 12.3 |
| 3 | Concordance ranks candidates on different populations | Accepted, reviewer's simpler option: M chosen on the geometric screen; concordance (fixed sample, strata, equal weights, coverage rules, ineligible and absent cases defined) may add at most one example, "screened for prediction concordance" | 5 |
| 4 | Centring, masks and ties generate results | Accepted. C and O labelled as different questions, side by side; −Σ/T covariance stated; stitched position baseline from the empirical lag distribution and schedule; other-texts disabled for a stitched text; expected overlap under random tie-breaking with stored tie counts; tie and all-mask tests on the production path (`torch.topk` replaced) | 4, 5, 9.1 |
| 5 | Unpriced readout; no Stage A audit | Accepted. Geometric counts for every token, readout on a stratified sample; bootstrap from per-document contributions; matching, overlap, aggregation and I/O timed in the A-pilot; standalone Stage A audit; reservations, failed attempts and priority-ordered settings under the $8 cap; ledger re-derived | 5, 11, 12.4 |
| 6 | "Memory adds" can mislead | Accepted for Stage B, recorded: one M with exact L, union projection, fourth outcome, M2 cut | 5, 8, 16 |
| 7 | Permutation reference too easy after projection | Accepted, recorded for Stage B | 16 |
| 8 | GPU memory and timing not bounds | Accepted, recorded for Stage B; memory figure corrected to include the doubled eigenvector stack | 6, 16 |
| §0 | Owner-facing claims | Adopted as listed: diagnostic sample named, unmatched column, log cited, chain claim replaced, m = 2 as interpolation, census versus spectral graph, narrower projection claim, CPU timings exclude graph building (now stated), 3% is a mean, fit and costs are pilot questions | 0 |
| Save | What Stage A should save | Adopted: provenance and hashes, separator ids, stitch maps, full-softmax top-20 and retained mass, attention sums and counts, tie and eligibility records, keyed control seeds, per-document paired contributions, audit expected values, ledger | 2, 3, 11 |
| Viewer | Offered choices without data | Reconciled: 4 layers and 8 shown neighbours in both controls and budget; Stage B adds M's and L's exact settings, dot included | 12.1 |
