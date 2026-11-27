# Depth embedding of tokens: spec v3.1 (2026-11-26)

Owner's request: build for language-model tokens the embedding DepthEmbedding built for images, for several
choices of which tokens it is built from, in both models, with Nystrom placement of tokens from one kind of input
into an embedding built from another. Stage B (`SPEC_B.md`) is untouched and still paused. v3.1 answers reviews 1 to 3
(`reviews/depth_spec_review{1,2,3}.md`); section 11 lists what changed.

Words used here. A **token** is one position of one text; it has one activation vector per layer (17 layers:
the embedding output and 16 blocks). A **construction set** is the set of tokens an embedding is built from.
**Placement** computes where a token outside the construction set lands in that embedding (Nystrom extension).
Indices: `l` layer (0..16), `i, j` tokens of the construction set (N of them), `k` mode (1..64), `x` a placed token.

## 1. Questions

1. What do the leading joint modes of the depth-coupled operator look like on tokens, per construction set?
2. How does the embedding of a fixed panel of tokens change along a ladder of construction sets?
3. Do Pythia-1b and OLMo-1B agree on the same token occurrences, beyond what the token's id gives?
4. Where do tokens of one kind (documents, shuffled documents, random tokens) land in an embedding built from
   another kind, compared with held-out tokens of the embedding's own kind?

Everything here is descriptive. No measure is a pass/fail test of a hypothesis, and pages say so.

## 2. Data

- **Existing:** Stage A activations, Modal Volume `tm-exp1`, `acts/<model>/c<cc>.npy`, float32 `(17, 2048, 2048)`
  = (layer, position, width). Position 0 is the model's separator and is never a token here; positions 1..2047 are
  the text's 2,047 tokens. Contexts 0..31 are the 32 Pile texts; both models have identical token ids. Local copy:
  `/data/tm_acts/{pythia_dl/pythia,olmo_dl/olmo}`.
- **New extraction:** nine contexts per model, same format, to an isolated root `acts_depth/<model>/`:
  - `r0..r4`: 2,047 ids each, drawn independently and uniformly **with replacement** from the sorted list of
    distinct ids that occur in the 32 texts (9,490 ids), `numpy.random.default_rng(20261126)`, one call of
    `choice` of shape (5, 2047). These are random tokens with a uniform, not natural, frequency profile.
  - `s0..s3`: documents 0, 1, 2, 4 with their 2,047 tokens permuted, `default_rng(20261127 + n).permutation`.
    The permutation is saved, so every shuffled occurrence is paired with its original. This is the contrast that
    keeps a text's token composition and changes only order.
  The id arrays and their SHA-256 are written to `acts_depth/inputs.json` before any model runs and are the same
  for both models. Extraction is a narrow adapter around Stage A's pinned forward path (`extract.py`): same
  separator, dtype, eval and attention settings, same provenance record. The frozen ids are fed directly (no
  decoding to text and re-encoding, which arbitrary id sequences need not survive); it asserts that every id is
  in the allowed list, that both tokenizers give the same token string for each id, and that the model input
  arrays equal the frozen arrays behind each model's separator; it refuses any output root but `acts_depth/`.
- **Spend:** one L4 unit per model (Stage A's ledger: 36 contexts took 58.8 s for Pythia and 283.5 s for OLMo on
  an L4), launched directly from the VPS with no billed coordinator. Each unit has a 600 s time limit and Stage
  A's 120 s startup allowance; at most one retry per unit and none beyond the reservation. Reserved before launch
  at the repo's recorded rate for an L4 with 4 CPU and 16 GB ($0.00025192 per second): 4 attempts x 720 s =
  $0.73. The launcher refuses if the reservation exceeds $1.00 and records it and the actual seconds in
  `ledger/depth.json`. Owner's go-ahead: 2026-11-26, "Okay go ahead please", in reply to the settings message that
  named this extraction as the only Modal use at under $1; maximum $1.00.

## 3. Construction sets and panels (positions fixed by seed, shared by both models)

Panels are drawn first and are never in any multi-document or random construction set.

- **Natural panel:** 32 positions from each of the 32 documents (1,024 tokens).
- **Random panel:** 128 positions from each of `r0..r3` (512), plus 256 positions of `r4`, a sequence no
  construction set uses (768 in all).
- **Shuffled panel:** 256 positions from each of `s0..s3` (1,024).

| Construction set | Documents | Tokens |
|---|---|---|
| `doc00`, `doc01`, `doc02`, `doc04` | one each (PubMed Central, Github, Pile-CC, USPTO) | all 2,047 (so they contain their own panel tokens) |
| `pairA`, `pairB` | {0, 1}, {2, 4} | 1,024 per document |
| `d04` | {0, 1, 2, 4} | 512 per document |
| `d08`, `d16`, `d32` | contexts 0..7, 0..15, 0..31 | 256, 128, 64 per document |
| `rand` | `r0..r3` | 512 per sequence |

Sampling is uniform without replacement from the positions 1..2047 left after the panel (the single-document
sets are the exception: they take all 2,047 positions), `default_rng(20261128)`, drawn in the order panels then
the rows of this table. Synthetic contexts have their own keys (`r0`..`r4`, `s0`..`s3`), never a document number. A manifest `depth/manifest.json` lists every token of every
set and panel by (context, position, token id) and is the key every later step and the audit join on.

The ladder is a description of these named samples. Sets differ in document mix and in tokens per document, not
only in the number of documents.

## 4. Construction

Code: `exp1/depth_dm.py`, copied from `DepthEmbedding/invert/dm.py` at commit `6c11546b` (file hash recorded in the
header), plus `exp1/spacetime.py` (byte-identical copy). One declared departure, below.

Per layer `l`: normalise each token's activation to unit length in float64 (**origin at zero, no mean removed**)
and store the normalised vectors; cosine matrix `C_l`; `d = 1 - C`. Token `i`'s radius `r_i` is its 15th smallest
distance to the other tokens (itself excluded first). `i` and `j` are joined if `d_ij <= r_i` or `d_ij <= r_j`
(edge 1, otherwise 1e-6, diagonal 0). This is a **15th-neighbour-radius union graph**: ties at the radius are all
kept, so a token whose id occurs many times can have far more than 15 neighbours at layer 0 (one id occurs 196
times in document 1). Per layer and set the build records the realised degrees (median, maximum) and the share of
tokens with a tie at their radius.

*One distance, however it is batched.* A matrix product's rounding depends on how it is blocked, and a cosine that
rounds to the other side of a radius flips an edge (review 2 measured 16 to 22 of 512 replayed rows failing with
7-row blocks). So: cosines come from a matrix product of the stored normalised vectors, and every pair whose
distance is within 1e-9 of either token's radius is recomputed as the plain dot product of the two stored vectors
(`numpy.dot` of two contiguous float64 1-D arrays, the same for either order and any batch). In two passes:
provisional radii from the matrix product; recompute the pairs within 1e-9 of either token's provisional radius;
take the final radii from the corrected distances; then compare with plain `<=` (the 1e-9 only selects pairs to
recompute, it is not added to the threshold). Placement does the same against the construction's frozen final
radii and the new token's own radius. Construction and placement share this one routine. Review 3 tested it on
real activations at block sizes 1, 7 and 256: no differing edge.

Symmetric Sinkhorn to a doubly stochastic `P_l` (maximum row-sum error under 1e-10 or the unit fails); lazy walk `M_l = (I + P_l)/2`. Scale `c_l` = mean of the first 8 nontrivial values of
`1 - eig(M_l)`. Operator on functions of (layer, token) with mean zero at every layer:

`L_t = blockdiag((I - M_l)/c_l) + t (D_path (x) I_N)`, `D_path` the path-graph Laplacian over the 17 layers.

`t = t*` and `t*/16`, where `t*` is `spacetime.balance_direct`'s value: the coupling at which the first
depth-only eigenvalue (a function constant across tokens at each layer), `4 t sin^2(pi / 34)`, equals the lowest
eigenvalue on functions with zero token mean at every layer (a crossing, not an optimum). 72 lowest modes are computed at each `t` (tolerance 1e-10); 64 are kept and the eigenvalue gaps at 8, 16
and 64 are recorded. Two adjacent modes are **near-tied** when their eigenvalues differ by under 1% of the lower
one; a run of near-tied modes is a band, shown and audited as a space, and its members' individual axes and
extreme tokens are marked as not individually meaningful. The 1% rule is a display convention, not a proof of
degeneracy. If a band straddles mode 8, 16 or 64, every top-k quantity at that k is marked "cuts a band" and is not
presented as a stable space. If the last band reaches mode 72 the solve is extended by 8 modes at a time (to at
most 96) until it ends, or the band is marked unresolved. Each layer's own top 64 nontrivial eigenvectors of `P_l` are kept too ("per-layer modes").
Site spectra and `t*` are computed once per model and set and reused for both strengths.

**Two departures from the image method.** (1) The origin is zero, not the probe's mean. (2) Ties: DepthEmbedding's
construction takes exactly 15 forward neighbours (ties broken by storage order) while its rule for a new point
keeps every tie on the reverse side. Repeated tokens give exact ties at layer 0, and review 1 measured 170 of 512
rows differing between the two paths. Here both paths use the radius rule above, so a token's row does not depend
on storage order and replaying a construction token reproduces its row. The pilots in section 9 used the upstream
rule and are labelled so.

Choices and the alternative, from the pilots in section 9: 15 nearest, not a perplexity rule; the zero origin, not
a mean (a design choice: the zero origin removes nothing and does not change when the construction sample
changes, which matters when tokens of one kind are placed in an embedding of another; the pilot counts under 15
nearest were similar at both origins except OLMo's layer 0, but mode agreement between origins was not measured); two coupling strengths
(the owner asked for one; `t*/16` is kept as the single alternative); no memory kernel (the owner dropped it).

## 5. Placement

For a token `x` and a construction set: at each layer, cosines of `x` to the set's tokens; `x`'s radius is its
15th smallest distance to them; `x` is joined to `j` if `d_xj <= r_x` or `d_xj <= r_j`; weights times the set's
Sinkhorn scalings, row-normalised (`depth_dm.rows`, with the radius rule). Then `joint_forward`: per mode, one
17 x 17 tridiagonal solve along depth. Two consequences the primer states: the construction's scalings are frozen
(the graph is not rebuilt with `x` in it), and a placed coordinate at one layer depends on `x`'s activations at
every layer. Every `x` gets a row, however far it is from the set; section 6's support measure is what says how
far. Placement runs in blocks of at most 512 tokens; a block's rows are reused for both strengths.

All three panels are placed into every construction set. For each (set, panel token) the manifest records a
cohort: **in set** (panel tokens of a single-document set's own document; their coordinates are the mode
coordinates, never used in a held-out score), **new position of a seen sequence**, or **unseen sequence**. This
Summaries are keyed by destination, source panel and cohort; a shuffled sequence is always an unseen *sequence*
even when its source document is in the set, and the key says which. This gives each destination a same-kind
held-out reference: natural panel tokens of the set's own documents for
document sets (for a single-document set none exist, so its reference is the natural panel of the *other*
documents and is labelled so), `r0..r3` panel tokens for `rand`, with `r4` as the unseen-sequence reference.

For the viewer only (never in a score): every token of documents 0, 1, 2, 4 and of `s0..s3` placed into `d32`
and `rand`, drawn as paths in reading order; tokens that are in the set are flagged.

## 6. Measures (all written as build facts; none typed into pages)

**Per embedding and `t`.** Eigenvalues, gaps, `t*`, maximum residual. Per mode `f` (unit norm over all entries):
- depth weight `w_l = sum_i f_l(i)^2`;
- token support `1 / (N sum_i p_i^2)` with `p_i = sum_l f_l(i)^2 / sum_l w_l` (1 if spread evenly over tokens);
- document share: `sum_l` (between-document sum of squares of `f_l`) `/ sum_l w_l`; not applicable to one document;
- cosine between the mode's slices at adjacent layers (mean) and between layers 0 and 16; a slice with
  `w_l < 1e-6` is flagged and excluded;
- the 8 tokens at each extreme at layers 0, 8, 16.

**Per placement cohort, set and `t`,** for the top 8 and the top 64 modes, per layer and with all layers pooled:
- radius ratio: `sqrt(mean squared coordinate of the cohort / mean squared coordinate of the set's own tokens)`,
  raw coordinates about the construction's origin;
- centre offset: per layer, the norm of the cohort's mean coordinate vector divided by the set's root-mean-square
  radius at that layer; pooled = root mean square of the per-layer values (so opposite shifts do not cancel);
- both are withheld where the set's mean squared coordinate is under 1e-12;
- support: the cohort's median cosine to its nearest construction token at each layer.

Once per embedding, `t` and mode: the tridiagonal matrix's smallest singular value and condition number, and the
largest relative residual `|T y - b| / |b|` over placed tokens. A mode whose condition number exceeds 1e8 is
withheld from every placement measure and flagged in the viewer; a top-k placement measure that contains a
withheld mode is reported as unavailable, not recomputed with another mode. A zero right-hand side has residual 0.

**Subspace overlap** `ov_k(A, B)`, used below. `A`, `B` are coordinates of the same rows in two embeddings, first
`k` modes, columns centred over the rows, float64, unquantised. SVD each; keep left singular vectors with singular
value above 1e-6 of the largest (ranks `ra`, `rb` are reported). `ov = |Ua' Ub|_F^2 / k`, reported only when `ra = rb = k` and
otherwise as unavailable with both ranks. Per layer: columns centred over that layer's rows. Stacked: each layer's
rows centred separately, then stacked. It compares spaces, not individual axes and not sizes; mode 3 in two
embeddings is not claimed to be the same direction, and pages label axes so. k = 8, 16, 64 throughout.

Written out, with `X_m^S` the coordinates of rows `X` in model `m`'s embedding built from set `S`:

- **Question 2 (within a model, spaces):** `ov_k(E_m^S, E_m^S')` for each adjacent pair of the ladder
  `doc00`→`pairA`→`d04`→`d08`→`d16`→`d32`, where `E` is the 992 natural-panel tokens outside document 0, which are
  placed (never copied) in every one of these sets; and `ov_k(E_m^d32, E_m^rand)`. Sizes are compared separately,
  by the radius ratios of the same rows.
- **Question 3 (between models):** `ov_k(S_pythia^S, S_olmo^S)` on each set's own tokens, and
  `ov_k(X_pythia^S, X_olmo^S)` for each panel cohort `X`.
- **Question 4:** the cohort measures above, each beside the same-kind held-out reference of that destination;
  and the cross-model agreement of the natural panel in a random-token embedding beside the same in a document
  embedding: `ov_k(E_pythia^rand, E_olmo^rand)` beside `ov_k(E_pythia^d32, E_olmo^d32)`.

## 7. Controls

1. **Algebra:** `joint_forward` applied to the fitted `P_l` reproduces the modes, relative error under 1e-8.
2. **End-to-end replay:** for every set and model, 256 construction tokens are placed from their activations with
   themselves masked, once in blocks of 256 and once in blocks of 7 with the queries reordered; their rows must
   match `P_l`'s rows (total variation under 1e-9, not to be relaxed) and their coordinates the modes (relative
   error under 1e-7). A fixture with repeated identical vectors, and one with rows and columns jointly reordered,
   must give the same result up to the reordering.
3. **Agreement with a refit** (`d08`, both models, `t*`): 256 natural-panel tokens of documents 0..7 are added to
   the set and the embedding is solved again. One rotation over all layers is fitted on the 2,048 shared tokens
   (top k modes, k = 8, 16, 64); reported: `1 - sum|aR - b|^2 / sum|b|^2` for the placed tokens and for the shared
   tokens. It measures agreement with a refit whose mean-free graph, scalings and `t*` all moved; it is not an
   upper bound and not a known answer.
4. **Question 3's references, same statistic, same rows:** (a) one model's token occurrences permuted **within
   token id and cohort** (each occurrence keeps its whole depth trajectory; never exchanging a copied token with
   a placed one), 20 draws from `default_rng(20261129)`, reported as mean and range with the share of tokens that
   can move, and called uninformative where under 10% can; (b) occurrences permuted freely, 20 draws; (c) per-layer modes only: two independent Gaussian vectors
   per token id (width 2,048) through section 4's per-layer construction, beside the models' per-layer-mode
   overlap. (a) asks whether matching the actual occurrence adds agreement beyond its id; it does not say the
   extra agreement is semantic.
5. `test_spacetime.py` passes on the copied code; a planted two-cluster fixture through section 4 separates the
   clusters in mode 1 at every layer.

## 8. Outputs, viewer, audit

- **Release data** (Volume `depth/`, off the site): float64 diagnostics, float32 coordinates for 64 modes, rows
  for the audited samples, the manifest.
- **Site data:** 16 modes per embedding, panel and path, int8 per mode with the scale stored, loaded on demand.
  The build measures the bytes this tab adds to the site and refuses above 150 MB (review 2 counts 94 MB before
  scales and metadata).
- **New tab "Depth embedding"**, adapted from DepthEmbedding's `embedding_tab.html` / `embedding.js`: a scatter of
  the construction set's tokens in two chosen modes at one layer; a depth slider with Play (depth is the time
  axis; a token is one point throughout); selectors for model, construction set, coupling strength; colour by
  document, position in the text, or how often the token id occurs in the set; hover shows the token with 8
  tokens of context each side; a panel or a text path can be overlaid; two models side by side with axes labelled
  as not matched. Below: per mode its depth weight and extreme tokens, and the overlap charts with their
  references.
- **Experiments tab entry D1** (E4 to E6 are reserved by `SPEC_B.md`), written from the build facts by Opus 5.5:
  one figure per question in section 1.
- **Audit** by an agent of a different model family than the pipeline's author, own numpy-only code, joining on
  the manifest. For `doc00`, `d32` and `rand` in each model, at both strengths:
  - rebuild 64 sampled rows of `P_l` at layers 0, 8, 16 from raw activations (total variation under 1e-8);
  - the solve, on the **float64** eigenvectors the pipeline keeps for these audited cells: residual
    `|L f - lambda f|` under 1e-7 with the structured operator (never the dense 34,816-square matrix), mean zero
    at every layer (1e-9), orthonormal (1e-8); and an independent count of eigenvalues below the midpoint after
    each of modes 8, 16 and 64 whose boundary is not inside a band (inertia of the shifted block-tridiagonal
    factorisation, as `SPEC_B.md` section 13 describes, with the depth-only sector's known eigenvalues
    subtracted) equal to 8, 16 and 64, so that a skipped mode is caught; a boundary inside a band is reported as
    not countable;
  - the float32 release equals the cast of those float64 vectors exactly;
  - coordinates, amplitudes included: for each complete band, one orthogonal matrix with no rescaling is fitted on
    the construction tokens between the audit's own solve and the release, then applied unchanged to the placed
    tokens of each panel; largest error under 1e-4 of that band's root-mean-square coordinate over the
    construction tokens, for both; an unresolved band is reported, not aligned;
  - every number the pages show, recomputed from the release data: counts and keys exactly; floating values to
    `1e-9 + 1e-6 |value|`, except quantities that are themselves rounding-sized (residuals, errors), which need
    only be under their stated gate;
  - int8 site data: re-encoding the float32 release with the stated rule (round to nearest of `value / scale *
    127`, clipped to [-127, 127]) reproduces the published bytes exactly; scales exact; saturated entries counted;
  - the published inventory. The site's release build refuses without a passing audit whose inventory matches.

## 9. Pilots this spec rests on (local, free, 2026-11-24 to 26; not for publication)

Files in `/data/tm_acts/pilot/`. Per-layer modes, about 2,048 tokens per set, top 64 modes; "small" = modes whose
participation over tokens is under 2%; "doc" = modes with document share over 0.5.

| Cell (layers 4, 8, 12, 16; sets of 1, 8, 32 documents) | Pythia small | OLMo small |
|---|---|---|
| Shannon perplexity 50, set mean | 7 to 23 | 5 to 14 |
| Participation ratio 50, set mean | 1 to 10 | 0 to 4 |
| 15 nearest, set mean | 0 to 4 | 0 to 2 |
| 15 nearest, zero origin | 0 to 6 | 0 to 3 |

All of these used the upstream tie rule (exactly 15 forward neighbours). Layer 0 under 15 nearest: Pythia 7 to 10
small modes at the zero origin and 9 to 13 with the set mean; OLMo 8 to 18 and 29 to 45. Document modes with 32
documents, 15 nearest: at most 2 at layers 4 and 8; at layer 12, Pythia 4 (zero origin) and 5 (set mean), OLMo 1;
at layer 16, Pythia 11 and 9, OLMo 18 and 13. In the first, incomplete grid (Pythia only, `docmodes.json`): a
single global width gave 37 to 64 small modes; the zero origin with the Shannon rule 22 to 48, and with the
multiplied per-token rule 30 to 60.

Depth-coupled pilot (8 documents x 256 tokens, set mean, upstream tie rule, `t*`, 64 modes, both models): solve 84
to 92 s on 3 threads; the algebra check passed (3e-14); agreement with a refit 0.99 / 0.96 (top 8) and 0.91 / 0.90
(top 64); tokens of other documents at 0.69 to 0.77 of the set's radius; cross-model top-16 overlap 0.76 with
layers stacked, 0.71 for the layer-0 slice of the joint modes, 0.005 with tokens permuted freely. Mean
adjacent-layer cosine of a mode was about 0.95 while the median cosine between its layer-0 and layer-16 slices was
0.05 (Pythia) and 0.09 (OLMo): a typical mode changes slowly from layer to layer and has a median end-to-end
cosine near zero.

## 10. Where it runs

Solves, placements and controls run on the VPS at low priority, 3 threads, one process at a time, resumable per
(model, set): each unit writes its outputs and a done-marker holding the hashes of its inputs, this spec, the code
and its outputs, and a rerun skips a unit only when all four match. Before the full run two units are run alone
with peak memory, wall time and output bytes recorded: the largest placement (`d32` with all three panels and all
eight text paths, one model) and the largest solve (the `d08` refit, N = 2,304); the full run starts only if both
peaks are under 4.5 GB, and any unit that passes 5.5 GB stops. Estimate from the pilot, to be replaced by the
gate's measurements: 22 solve-and-place units of 5 to 8 minutes, two refits, the references of control 4 (overlaps,
plus 17 site eigendecompositions per set and draw for the Gaussian reference), packing, and the audit's own twelve
joint solves under the same memory stop: 4 to 5 hours in all. The only Modal spend is section 2's.

## 11. Changes from v1 (review 1's finding numbers)

Tie rule unified and replay made an end-to-end control (1). Within-id permutation reference on the same joint
statistic (2). Fixed panels, within-model ladder comparison, d32 against rand (3). Cohorts and same-kind held-out
references for every destination, an unseen random sequence, saved shuffle map (4). Radius ratio defined, centre
offset and support added, solve diagnostics recorded (5). Rank-revealing overlap, guard modes and gaps, unmatched
axes, k-means colouring dropped (6). Token support separated from depth weight; end-to-end depth cosine added;
the v1 claim that depth shows little at `t*` was wrong and is withdrawn, so three strengths became two (7). Refit
renamed and frozen; audit checks rows, eigenvalues and amplitudes (8). Pilot table regenerated from the JSON and
the zero origin crossed with 15 nearest, which changed the origin choice (9). Site tier cut to 16 modes with a
measured cap (10). Blocked placement, cached spectra, measured gate before the full run, structured audit (11).
Random ids, adapter, isolated root and reserved worst-case spend specified (12). Source commit pinned, entry
renamed D1, `t*` described as a crossing (13).

Changes from v2 (review 2): one distance routine with pairs near a radius recomputed canonically, replay gated at
two block sizes; the graph named for what it is, with realised degrees recorded (1, first additional). Ladder
compared on 992 always-placed tokens; every overlap written with both arguments; references permuted within id and
cohort (3, second additional). Keys by destination, panel and cohort (4). Smallest singular value, residual
normalisation, withholding rule, pooled centre offset that cannot cancel (5). Overlap requires full rank on both
sides; near-tied bands defined for display and audit (6). Audit given tolerances, an independent mode count and a
band-level alignment without rescaling (8). Pilot prose corrected and labelled as upstream-rule; the claim that the
two origins give nearly the same modes withdrawn, the zero origin kept as a design choice (9). Site cap is on what
this tab adds (10). Gate covers the largest placement and the largest solve; resume keyed on hashes (11).
Reservation priced with startup, id checks spelled out, the owner's go-ahead quoted (12). `t*` worded as the
crossing it is (13).

Changes from v3 (review 3, which closed everything else): fixed cutoffs that cut a near-tied band are marked and
not presented as stable spaces, guard modes extend until the last band ends, counts only at resolved boundaries;
the audit's solver gate applies to kept float64 vectors and the float32 and int8 forms are checked as exact casts
and encodings; the two distance passes written out; withheld modes make a top-k measure unavailable.
