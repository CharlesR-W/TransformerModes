# TransformerModes: design brief for the first experiment

This brief is from the orchestrating agent to the designer. It records what the owner asked for (fixed), what the
owner has answered (fixed), and the orchestrator's analysis (proposals: accept, improve or overrule them with reasons).

## 1. The owner's request, verbatim in substance

- New project `~/code/TransformerModes`, split from CoherentModes. Old Pythia / Mamba / DINO studies are copied to
  `deprecated/` and are out of scope.
- "Run some experiments on normal LLMs akin to the CIFAR ones", examining "a different similarity kernel that I think
  makes sense":
  1. the dot product of the whole sequence's vectorized activations, "probably we can just use a simple running mean
     to avoid storing";
  2. just the local residual vector;
  3. "maybe some smart EMA or similar, perhaps based on context window".
- "Probably we should look at both cosine and dot product. kNN is fine but perhaps a Shannon perplexity rule is gentler."
- "One set where the graph is calculated along a single long context, and another where we aggregate all of the
  contexts (possibly accounting somehow for that? Unsure). Since this will itself change the embedding."
- "Drop for now the multi-site averaged Laplacian stuff and instead just use the joint spatiotemporal one"
  (DepthEmbedding's through-depth operator). "At least two LLMs on the same probe set, probably Pythia and OLMo. So
  we'll be using many of the techniques from DepthEmbedding/ as well."
- "Avoid having too many views: the Mamba page is unreadable because of this. Where we need to fiddle with a bunch of
  indices that I may not be aware of, add an explainer in the primer section aiming to make this intuitive, or settle
  on one and leave others for later."
- "Feel free to investigate several ways of doing this to find what is the most principled and functional as a
  potential interp tool." "Try hard to make it easily readable and understandable and visually intuitive, while making
  it easy enough to verify there's no crazy bullshit going on."

## 2. The owner's answers to clarifying questions (fixed)

- **Kernel = end-aligned prefixes.** With `G[t,t'] = <x_t, x_t'>` the local Gram matrix of one layer's residuals,
  `K_a[t,t'] = sum_{s>=0} a^s G[t-s, t'-s]`, i.e. `K_a[t,t'] = G[t,t'] + a K_a[t-1,t'-1]`. a = 0 is the local
  kernel, a = 1 the whole prefix, 0 < a < 1 an EMA with memory about 1/(1-a) tokens. Items are token positions.
- **Models: `EleutherAI/pythia-1b` and `allenai/OLMo-1B-0724-hf`.** Both are 16 layers x 2048 wide with the GPT-NeoX
  tokenizer (vocab 50,304; OLMo adds a few special tokens), so the same text should give identical token ids and
  nodes align token-for-token and layer-for-layer. This must be asserted, not assumed.
- **Probe text: the orchestrator's judgement**, with this guidance: "I want one where the probe set is pretty
  natural/random and another where it is context specific. I imagine doing some sort of sparse/Nystrom magic later and
  seeing if they are similar (like if the context-specific one looks like an unbiased sample from the true one).
  Take care to optimize this step since it should make life much simpler. Splitting a document into many contexts is
  the least desirable, because it muddies things."
- **Repo: public `CharlesR-W/TransformerModes` with a GitHub Pages site like DepthEmbedding's.** Finished experiments
  go in the site's Experiments tab and are deployed without asking.

## 3. The orchestrator's analysis (proposals)

### 3.1 Probe design: one extraction serves both sets

Proposal. Draw M independent documents (M about 24 to 32), each from a different source document, each one context of
the full length T = 2,048 (position 0 = the document separator, 2,047 text tokens), at random from a held-out natural
corpus (the Pile test split was used before; it is held out for Pythia, not guaranteed for OLMo, which trained on
Dolma: state it). Run both models once and store the residuals at the chosen layers for every position (fp16 or fp32,
about 1.6 GB per model at 8 layers). Everything downstream is recomputable from that store without the models.

- **Pooled ("natural") graph**: nodes = n positions sampled uniformly at random from each context (n about 512), so
  N about 12k.
- **Single-context graph**: nodes = all positions of one context. Any of the M contexts can play this role, so the
  page shows one or two chosen in advance by seed and the statistics are reported over all M as replicates.
- Because every single context is also a member of the pool, "does the context-specific picture look like a sample
  from the pooled one" is a direct comparison on shared nodes, and the later Nystrom question is leave-one-context-out:
  build the pooled graph without context c, extend to c's tokens, compare with c's own graph. No document is split
  into several contexts.

Open points for the designer: T (2,048 vs shorter; equal lengths matter for the whole-prefix kernel and for position
effects), M and n, whether to sample documents uniformly among those long enough (biased toward papers, code, books)
or with a per-source cap, burn-in positions to drop, and whether a planted or stitched control context is worth one
extra item on the page (the skill in use requires at least one control with a known answer; a synthetic-activation
planted test of the pipeline may be enough).

### 3.2 Computing the kernel exactly

For two contexts c, c', `K_a` on the T x T block is a one-sided geometric filter along the diagonals of the Gram block
`X_c X_c'^T` (a conv1d over the skewed matrix, or a row recursion; cumulative sum for a = 1). All M(M+1)/2 blocks x
layers x a few values of a is minutes on one GPU, after which node rows are sampled. So no approximation is needed at
this size. A scalable alternative for later (not for now unless you see a reason): a recurrent random sketch
`z_t = P x_t + sqrt(a) U z_{t-1}` with U random orthogonal, whose dot products are unbiased for `K_a`; then every
token has one vector and Nystrom / approximate neighbours are ordinary.

### 3.3 Hazards I expect

- **First-position and outlier norms.** Position 0 (and sometimes delimiter tokens) has an enormous residual norm in
  these models. Under the dot product with a = 1 every prefix contains it and all prefixes become parallel. Prefix
  sums should start after position 0, and per-token normalisation is one candidate reading of "cosine".
- **What "cosine vs dot" means here.** Candidates: (i) dot: Euclidean distance in the weighted-prefix feature space,
  d^2 = K[t,t] + K[t',t'] - 2K[t,t']; (ii) cosine of prefixes: K / sqrt(K[t,t] K[t',t']); (iii) mean of per-token
  cosines: normalise each x first, then filter. Pick the readings worth running and say why.
- **Centring.** Uncentred LM residuals share a large mean direction. The mean can be taken over the pool or over the
  node set, and that choice is part of "aggregating changes the embedding".
- **Trivial position modes.** If residuals carry a slowly varying component (topic, position drift), K[t, t+delta] is
  high for small delta, the single-context graph is close to a chain along the text, and the leading modes are
  position harmonics. The whole-prefix kernel is the most exposed. The page must show how much of the low space is a
  smooth function of position, with a control, rather than let the reader mistake it for structure.
- **Same-context dominance in the pool.** Tokens of one context share context-specific components, so pooled
  neighbours may mostly be same-context. The principled handle the existing pipeline already has is density
  normalisation: Sinkhorn (doubly stochastic) scaling removes sampling density to first order (Coifman-Lafon alpha = 1;
  Marshall & Coifman; Landa, Coifman & Kluger), which is exactly the statement "a single context is a non-uniform
  sample of the same geometry". Alternatives: cross-context-only neighbours, per-context centring. Decide what is
  default, what is a measured comparison and what is deferred.
- **Exact duplicates.** At the embedding and early layers with a = 0, identical tokens give identical vectors (Pythia
  has no positional embedding in the residual; rotary). Choose sites and tie handling accordingly.
- **Memory length.** a = 1 probably washes out in a 2,047-token context; a = 0 is tied to token identity. "Smart EMA
  based on context window" could mean: a fixed small grid of memory lengths; a per-layer length read off the
  residual stream's own autocorrelation; or lag weights taken from the model's measured attention-distance profile at
  that layer (label-free, model-derived, transformer-specific). The filter generalises from geometric weights to any
  lag profile w_s at no cost.

### 3.4 Graph rule

"Shannon perplexity rule": per-point Gaussian bandwidths set so each row has a fixed entropy. The existing pipeline
ends in a symmetric doubly stochastic lazy walk. Symmetric entropic affinities (Van Assel et al., SNEkhorn, NeurIPS
2023) give a symmetric doubly stochastic matrix with per-row entropy constraints; the TorchDR package implements
`EntropicAffinity`, `SymmetricEntropicAffinity`, `SinkhornAffinity` on GPU. The owner prefers reputable packages over
rewrites. Check that it fits (dense N about 12k, float64 behaviour) before relying on it; kNN + Sinkhorn from
`DepthEmbedding/coherent.py` is the comparison rule.

### 3.5 Operator

DepthEmbedding's `spacetime.py`: `L_t = blockdiag(H_l) + t (D_path (x) I)`, identity link between a token at one layer
and the same token at the next, solved in the sector of mean-zero slices, relative site units, coupling t* as the
default ruler. Reuse the module and its tests rather than rewriting. Sites: a subset of the 16 block outputs (8 even
layers?) is probably more readable than all 16.

Deferred, to be listed as such: attention-weighted links between positions across layers; a checkpoint (training
time) axis; a joint operator with the model as a coupled axis; the random-sketch and Nystrom work.

### 3.6 How to settle on one kernel

The owner invites investigating several constructions but wants few views. Proposal: the investigation lives in a
small number of Experiments-tab entries with quantitative, mostly label-free criteria, and the main viewer ships one
default construction (at most one two-way toggle, local vs prefix, if that contrast is the scientific point).
Candidate criteria:

- a planted synthetic test of the pipeline with a known answer (for example hidden regimes that a lagged window
  identifies and single vectors do not);
- agreement between the two models on the same tokens (principal angles of low spaces at matched layers), which is
  label-free replication;
- stability under node resampling and between pooled and single-context graphs on shared nodes;
- share of the low space explained by token identity and by smooth functions of position (triviality controls);
- validation only: document source, and cheap token-level annotations if any are worth it.

### 3.7 Page

Site machinery copied from DepthEmbedding (tabs, Experiments tab with build-time facts, browser checks). The viewer
should make the text itself the main object: tokens coloured by a mode or set at a chosen layer, next to an embedding
scatter linked to it, with the two models side by side since their tokens align. A primer section explains every
index the reader meets (token position, layer/site, memory length, coupling, mode number). Count the controls and
justify each one.

## 4. Fixed commitments inherited from CoherentModes (apply to the construction)

- Every object in the construction ends in an SVD/eigendecomposition of a sample-estimable operator; nothing is
  trained on top of the model inside the construction. Fits (k-means, Procrustes, UMAP) are allowed as readouts and
  must be called fits.
- The only geometric assumption is the dot product on each view (cosine on normalised activations).
- No labels, probes or task distributions define a kernel or a basis; labels are for validation and colouring.
- No sparsity assumption.
- Transformer-specific choices are allowed in this repo but must be named as such.

## 5. Constraints

- Orchestration box has no GPU: all model and kernel computation on Modal (the `modal` skill conventions: detached
  coordinator, one worker per unit, outputs on a Volume). State projected spend and wall-clock in the spec.
- Model, dataset and package versions pinned; float64 where the old studies needed it (Sinkhorn, eigen-solves).

## 6. Facts from a survey of DepthEmbedding (verify anything you rely on)

- Walks there: centre over the probe, normalise rows, cosine Gram; union kNN k = 15 (weight 1, floor 1e-6, zero
  diagonal); symmetric log-domain Sinkhorn; lazy walk M = (I + P)/2 (`coherent.py`,
  `upstream/cifar_coherent/sweep_math.py:final_kernel_bank`). Other rules exist upstream (perplexity 100/300, global
  kappa) but only kNN shipped.
- **Prior evidence against the plain Shannon-perplexity rule** (`invert/NOTES.md`): in the out-of-sample work "the
  Shannon-perplexity kernel produces modes carried by two images and was rejected". What worked was a
  **participation-ratio kernel**: per-point Gaussian width chosen so each row's participation ratio is k (an order-2
  Renyi perplexity), symmetrised, then Sinkhorn; its out-of-sample row reproduces a masked probe item's row exactly,
  and Nystrom-type extension reproduced 0.94-0.98 of the first 16 modes. The owner has now suggested Shannon
  perplexity as "perhaps gentler" and may not have this result in mind: weigh it, and say plainly in the spec what you
  chose and why.
- Operator defaults: relative site units (c_l = mean of the site's first 8 nontrivial 1 - lambda_k), coupling t = t*
  (first temporal eigenvalue equals first eigenvalue on mean-zero slices), 24 modes computed, 16 stored.
- Solver: the dense block-elimination `Solver` in `spacetime.py` has been run at N = 8,000 x L = 12, K = 256 on Modal
  (16 cores, 96 GB). A sparse CSR + rank-one Lanczos prototype (`opt/REPORT.md`) is faster but was never put in the
  pipeline; its projections to N = 50,000 are unmeasured.
- Site: three tabs (Viewer, Experiments, Method). Experiments entries follow `site/xsite/AUTHORING.md`: a generator
  writes chart specs and a facts file, prose uses `{{fact:...}}` tokens, no measured number is typed by hand,
  `check_numbers.py` recomputes headline facts independently, `check_browser.py` drives headless Chrome and fails on
  console errors or external requests. Deploy is a single orphan commit force-pushed to `gh-pages`.
- The DepthEmbedding viewer has grown about 25 controls. That is the pattern the owner is complaining about.
- Owner's page preferences recorded there: read by skimming, "must not look long at a glance"; each entry is a
  question, a 2-3 sentence answer with one bold sentence, a lead figure, 3-5 number tiles, a "what was computed" table
  (quantity, equation, symbols in words, code location); captions start "Plotted:" and say only what is drawn; plain
  words in labels, not variable names; post-hoc fits are called fits; at most six caveats; exploratory, stated once,
  not hedged after; "be loose but never false"; text never takes a data colour; extra views go in closed `<details>`.
