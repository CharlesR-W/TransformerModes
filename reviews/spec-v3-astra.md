**Stage A is worth doing. I would approve the census under a hard $8 cap after a few changes, but not approve v3 exactly as written.** The census has a useful answer even if Stage B never runs: it can show how memory changes which passages a residual is compared with, and whether those passages have similar next-token predictions. The earlier objections to spending on a large, unvalidated spectral sweep no longer apply to that question.

The remaining blockers for Stage A are narrower: one mandatory pre-spend test has the wrong answer; the shortlist can reward properties imposed by the mask or by which pairs are matchable; and the pilot does not time a potentially dominant analysis step. These can be fixed without changing the models, kernel, or probe. **Stage B should remain a separate approval.** In particular, its current test of “memory adds” can return yes even if the underlying model pictures have not changed.

Line references below are to `/home/ubuntu/code/TransformerModes/SPEC.md`, v3.

### Status of the eight review-2 findings

| # | Status | Reason |
|---|---|---|
| 1. No-memory baseline and alignment margin | **Partly resolved** | Projecting both models off a common span is a genuine improvement. But the span changes between *constructions* being compared; own-text centring and a position mask can also give a no-memory text-only graph structure beyond current-token indicators. |
| 2. Impossible truncation check | **Resolved** | It is gone. The prototype provides a positive reason to shelve truncation, not merely a concern that it might matter. |
| 3. Floor-defined modes | **Resolved for the primary recipe** | The dense PR walk has no floor. The separate kNN comparison is labelled. The remaining issue is detection of near-disconnections in a dense walk, not the old floor problem. |
| 4. Statistical predictions used as exact gates | **Partly resolved** | Planted recovery and noise agreement are now measurements, and the temporal formula uses the correct \(L\). The replacement analytic fixture specifies the wrong first mode. A stochastic text-only counterexample is also not a deterministic gate without further construction. |
| 5. Two solvers can agree on the same wrong graph | **Partly resolved** | Comparing \(q,W,P\), scales, and isolated eigenspaces against the CPU oracle is the right design. Correct the independent fixture and exercise a masked, nonuniform graph in G1; the named real G1 construction is unmasked. |
| 6. No-signal denominator and unequal companions | **Resolved** | Focal occupancy and energy surviving centring now have separate denominators. Both companion groups contain 15 other texts, and the result is correctly called illustrative. |
| 7. Functional controls not matched | **Resolved for the original descriptive readout** | Actual destination text, token ID, and within-text lag bin are now matched, and the score is accurately named top-20 overlap. Using it to *rank candidates* introduces a new common-support problem. |
| 8. Cost ledger and pilot | **Partly resolved** | The hourly rates and separate approvals are credible. The unbenchmarked matched-pair analysis, timing of the audit, and dense graph-building cost still limit what the totals establish. |

### Fix before Stage A

**1. Blocking — the mandatory analytic test will reject a correct implementation.** §9.1, lines 315–326; §0, lines 51–52.

For identical layers, let the centred group indicator have single-site scaled energy \(h\). Its depth harmonics have eigenvalues
\[
h+t\,4\sin^2\!\left(\frac{j\pi}{2L}\right),\qquad j=0,\ldots,L-1.
\]
The **first** mode has \(j=0\): the indicator with a **constant** depth envelope. Line 321 requires the \(j=1\) cosine envelope. That harmonic has strictly higher energy at \(t>0\). The cited sibling result says this correctly.

Also specify a feasible group size. With \(n\) tokens per group the conditional has \(n-1\) equally near neighbours and \(n\) equally far ones. For an interior PR64 solution one needs \(n-1<64<2n-1\). In this two-distance fixture, changing their separation merely rescales the fitted inverse bandwidth; it does **not** independently dial the between-group probability to an arbitrarily tiny value. Finally, a two-group near-cut has one small nontrivial gap, not eight. It should not be expected to trip line 259’s eighth-gap test.

**Change:** use memory none, specify \(n\), solve for the two actual row weights, and test the constant-envelope ground state and the first nonconstant harmonic separately. For the near-cut test, directly construct a walk with prescribed crossing weight. Use two groups to test detection of one fragile direction and at least nine to test a fragile eight-gap scale. These Stage B fixtures can instead be moved to the pre-B gate; the pre-A tests should target the census code that A actually runs.

**2. Blocking for freezing the shortlist — the screen is not a test of the graph Stage B will build.** §§3–6, lines 115–126, 181–184, 205–219, 223–227.

There are two distinct mismatches. Stage A ranks 64 neighbours among **57,344** positions; Stage B uses the **last 512 positions per document**, among 16,384. Merely restricting Stage A’s *queries* to the suffix does not restrict the candidate neighbours to it. Whole-prefix memory is especially sensitive to this change. And “64 nearest” is not the same as a dense conditional with *participation ratio* 64. For example, over 16,383 candidates, a valid PR64 row with one heavy entry and equal remaining entries puts only **12.8% of its mass in the top 64**. The prototype’s measured spectral drift after removing even much smaller tail masses makes this distinction consequential.

The screen also gives every windowed candidate a near-neighbour share of zero **by definition**. That is a legitimate design choice to forbid local recurrences, not evidence that the residuals naturally connect distant passages. Other-texts-only similarly forces its displayed other-text share to one.

**Change:** keep the full census as the primary descriptive result. While the distance rows are live, compute a cheap **Stage-B-matched screen** on prespecified queries: both query and candidate in the 512-token suffix; report top-64 origin shares and dense *conditional-mass* origin shares on those rows. No eigensolve or full stored matrix is needed. Show how much short-range mass was removed by the mask, and distinguish a rule-imposed exclusion from an observed result. If the budget does not support this extra screen, call the shortlist exploratory and do not describe rejected candidates as unsupported for the dense operator.

**3. Blocking for using the functional score as a rank — candidates are evaluated on different populations.** §5, lines 186–219; §12.3, lines 468–470.

For a fixed query-neighbour pair, the revised control is sensible: it asks whether the neighbour is more predictive than an eligible non-neighbour with the same destination text and token ID, and similar within-text lag. But rare IDs and small lag bins frequently have no alternative, especially after excluding all 64 selected neighbours. A construction can win by producing mostly unmatchable unusual pairs and doing well on the few frequent-token pairs left. Reporting overall coverage does not make its mean advantage comparable with a candidate covering quite different queries and token types. Resampling documents on fixed lists will not remove this bias. Nor does a positive interval, after screening 32 candidates across eight correlated sites, constitute independent confirmation.

It **is** sensible to use the model’s predictions to choose an example of a *prediction-related neighbourhood*. It is not a neutral way to choose an interpretability tool: the final model output is now part of the selection objective, so a model-shared pattern unrelated to next-token overlap can be excluded by design.

**Change:** retain the functional readout. Prespecify a common query sample, minimum coverage, coverage broken out by text, neighbour origin and token frequency, and a paired/common-support comparison or fixed stratum weights for ranking. Define ineligible and zero-coverage cases. Describe any selected result as “screened for prediction concordance,” not independently validated by that same score. My simpler preference is to choose one memory candidate on the geometric screen and use prediction concordance to choose at most one *additional* example. That preserves a path for an interesting construction that does not optimise the readout.

**4. Should-fix before A — own-text centring, temporal exclusion and tied text controls can themselves generate the result being displayed.** §§4–5, lines 134–149, 166–176, 189–199; §12.2, lines 451–455.

Own-text centring subtracts a mean computed from the **entire** text, including tokens after the query. It removes genuine topic information as well as unwanted offsets. Even with IID vectors of covariance \(\Sigma\), the covariance of two distinct demeaned positions in the same text is \(-\Sigma/T\); positions in different texts do not acquire that covariance. This is a valid *retrospective within-document contrast*, not a recovery of the model’s causal state. A window of 64 forbids short offsets; it does not make two whole-prefix histories independent. For the stitched controls, “predicted by position alone” also needs an actual definition: the same-source rate rises automatically for near neighbours inside long segments.

The text-only adversaries have an additional problem. If a current token occurs hundreds of times, many no-memory neighbours can be exactly tied. Choosing the first 64 by `(text, position)` gives a reproducible but artificial preference for early documents. A small intersection with those 64 IDs is not evidence against a lexical explanation. The existing `exp1/similarity.py:135–138` uses `torch.topk`; its documentation and `test_numerics.py:91–99` explicitly leave tie membership unspecified. The proposed tie policy is **not** covered by the previous successful tests.

**Change:** label C and O as different questions, not raw versus corrected truth. Show ordinary and masked results side by side; disable other-texts-only for a stitch analysed in isolation. Define the stitched position baseline using the empirical lag distribution and segment schedule. For adversary comparisons, count ties at the cutoff and use fractional/expected overlap across tied candidates, or report an overlap interval. Add exact-tie and all-three-mask tests for the production GPU path. These changes do not require a third centring or another memory sweep.

**5. Blocking for the $8 cap and a verified Stage A page — the expensive part not covered by the pilot is not the Gram multiply.** §§5, 11, 12.4, lines 181–199, 365–378, 397–400, 485–492.

I reproduce the order of magnitude of the stated Gram count and the hourly hardware rates. But the eight model sites alone generate
\[
2\ \text{models}\times2\ \text{readings}\times2\ \text{centrings}
\times5\ \text{memories}\times3\ \text{masks}\times8\ \text{sites}
\times57{,}344\times64
=\mathbf{3.52\ billion}
\]
directed neighbour pairs. Each functional measurement requires finding a matched control and comparing two top-20 lists for both pairs. UA2 allows 30 minutes for this **plus** summaries, bootstrap, timescales, synthetic census and packaging. The A-pilot measures GPU list generation, not that workload. The concrete all-input list count is about **35.2 GB** before stitches or compression; streaming is feasible, loading many expanded ID/probability arrays at once is not. Dropping two dot memories also saves less than a proportional share of total cost because their Gram is shared with the remaining memories.

Separately, Stage A is supposed to stand alone and publish before B. The only independent audit is a **Stage B** unit, and the build fails if its report is missing. The user should not have to fund B to verify A.

**Change:** benchmark matching, overlap computation, aggregation and output I/O on one representative shard **inside the A-pilot**. Compute geometric counts for every token, but prespecify a stratified sample of query positions for the functional readout if necessary, shared across candidates and models. Bootstrap stored per-document contributions, not billions of individual pairs afresh. Split the audit: A gets its own independently recomputed sample of full distance rows, masks, matches and statistics before publication; B gets the operator audit only if approved. Put both components, failed attempts and a real stop/reservation mechanism under the cap. **With these changes and the corrected gate, I have no blocking objection to Stage A itself.**

### Can wait until Stage B approval

**6. Blocking for the claim “memory adds” — the compared remainders and the required local counterparts are not necessarily the same.** §§4, 5, 7–8, lines 166–168, 213–218, 271–284, 297–311.

The common projection is valid for comparing Pythia with OLMo **within one construction**. But memory none removes \(Z\) and current-token indicators; memory \(>0\) also removes up to 96 adversary directions. Identical underlying model pictures could score higher for “memory” just because its extra projection removed directions on which the models disagreed. A paired bootstrap would reproduce that difference faithfully; it would not attribute it correctly. Conversely, the extra 96 columns can push a short-text projection above the 50% exhaustion gate and disqualify memory purely by dimensionality.

Also, at memory none the **embedding input** is a function of token ID, but its **picture** need not be: own-text centring makes the feature depend on token and text, and a same-text window makes graph eligibility depend on position. Additive token and document indicators do not remove all such interactions. Finally, R′ matches M1 only. M2 can have another reading or accounting, win Stage B, and have no local counterpart. The three outcomes omit “a memory construction passes, but is not demonstrably better than local,” and references to M1 are undefined if no memory passes the A screen.

**Change:** select **one** primary memory with its exactly matched local counterpart. For the *difference* test, apply the **same union of nuisance/text features to both sides** and show retained energy and rank on both. Allow “both supported, no demonstrated increment”; distinguish it from “memory unsupported.” If M2 is retained, either pair it properly or declare it descriptive. State “outside the tested text features,” never “surface text statistics cannot win.” Use the Stage A token tables to calculate projection ranks and singleton fractions before authorising B.

**7. Should-fix — the cross-model permutation reference can be too easy after projection.** §§7–9, lines 265–284, 297–311, 323–337.

Both projected pictures live in the same, potentially small, irregular subspace. Removing every token indicator makes a singleton token’s residual row zero in **both** models. Permuting one finished picture freely within a document moves these zeros onto other token types and generally takes it *out* of the common residual subspace. Even independent random directions in a shared \(d\)-dimensional allowed space have expected squared cosine \(1/d\), not \(1/(N-1)\). The current reference can therefore call agreement “above chance” partly for obeying the same projection. Shared position geometry can compound this.

**Change:** calibrate with independent fields or surrogate factors subjected to the same indexing, projection and restriction; if using permutations, reapply the common projection and check calibration on text-only and independent controls. State precisely the null being tested. Make the hard `decide.py` example deterministic—for example, literally zero retained signal should fail the energy gate. “Adversary plus independent noise must always fail a 95th-percentile statistical test” is not an exact invariant.

**8. Should-fix — the GPU route is plausible, but 49 GB and the B timetable are not demonstrated upper bounds.** §§0, 6, 11, lines 56–65, 240–261, 380–395.

One \(16{,}384^2\) FP64 matrix is indeed 2.15 GB; eight walks or eight inverses are about 17.2 GB. There is no evident mathematical obstacle to the dense port. However, the copied `spacetime.py:165–171` accumulates a list of site eigenvectors **then stacks it**, transiently holding two eigenvector collections. With walks, that alone is about **51.5 GB**, before libraries, allocator caches or other temporaries. CuPy/cuSOLVER workspace and pool retention matter. G1 should include a masked candidate and verify *achieved PR*, not only agreement with the CPU routines: `kernels.py:39–92` does not reject an unattainable target caused by tied minima. The eighth-gap check can identify a vanishing eight-gap scale but miss a single cut that determines the first displayed mode; report the first several gaps, not just the binary flag.

The measured one-thread **untruncated** \(N=1,792\) reference took about **90 s building the graph, 18 s calibrating and 10 s solving**. Thirty-six large one-text graphs alone would take about 71 serial minutes at that rate, before the matched, adversary and stitched work. Eight cores could make a 25-minute group plausible with explicit parallelism; the GPU R pilot does not measure that. The six-query audit is an improvement, but the CPU all-texts walk rebuild is another substantial dense operation.

**Change:** benchmark one CPU group as well as one end-to-end GPU unit before releasing B. Reuse the full site spectra for the scales; avoid the stack duplication or measure it explicitly; record device-level high-water in addition to the pool’s; test PR, excluded-edge zeros and an independently assembled masked walk. Include the audit in the timing projection. If short of budget, drop the full-size recipe comparison or second focal-text companion before weakening the primary oracle or matched memory comparison.

### What §0 can tell the owner

The diagnostic **does** justify doing a census first. I checked the saved rows: whole-prefix corpus-mean own-text fractions were **0.944, 0.946 and 0.957** across the three layers, versus **0.434, 0.511 and 0.581** for no memory. The earlier `exp1/local_pilot_v1.json` contains the 75–84% within-32 figures. The prototype also supports the decision not to call truncated rows a numerically faithful substitute for the dense rule: at 1,024 retained neighbours the **mean** lost mass was 3.25%, \(t^*\) moved about 15%, and the subspace dissimilarity was 0.187 **on the tested synthetic graph**.

Four edits would make the owner-facing account accurate:

- Name the diagnostic sample: **four WikiText articles, two Python standard-library files and two licence texts**, not eight sampled Pile documents. The final prediction column is an unmatched diagnostic, not Stage A’s matched advantage. The script prints the random baseline but does not store it in `local_pilot.json`.
- Replace “the graph is nearly a chain” and “modes … would show” with **“nearest neighbours are usually nearby in the same text; we have not measured the dense walk or its modes on these inputs.”** The own-text fraction’s rise from none to 8 accounts for about **34%, 38% and 51%** of its total rise to whole-prefix across the layers. Trying \(m=2\) is sensible interpolation, not a demonstrated finding that most change happens below 8.
- Replace “every token is a node” with **“every post-burn-in token enters the census; the all-texts spectral graph uses only the final 512 per text.”** Replace “surface text statistics cannot win” with the narrower claim about the specified projection.
- Distinguish measured CPU numerics from proposed GPU validation. The 15 s and 9 s CPU figures are calibration and solve **on already-built walks**; they exclude dense graph construction. The reported truncation loss is an average, not 3% in every row. The 80 GB fit and the costs are pilot questions, not existing measurements.

### What Stage A should save now

1. **Extraction provenance:** exact model/tokenizer/library revisions actually used, model eval and precision settings, file indices and content hashes, accepted token IDs including each separator, source-to-stitched position mapping, and passed hook/causal checks. Keep raw block outputs, including the pre-final-normalisation last block.
2. **Prediction and attention evidence:** top-20 IDs **and full-softmax probabilities**, entropy, retained top-20 mass, and document/head/query-range counts and sums for the attention summaries. Saving raw activations is enough for most later changes. Exact vocabulary probabilities can be regenerated from the final residual and pinned final norm/output head; there is no need to store every logit merely as insurance.
3. **Census and control evidence:** immutable query/candidate IDs; eligible counts by origin, cutoff distances and tie counts; prespecified suffix queries and suffix-restricted results; matched control IDs or independently replayable keyed seeds; unmatched reasons and denominators by document/stratum. Store per-document *paired* contributions. A pooled percentage, scalar seed or top-64 list over the larger pool cannot reconstruct all of this cheaply.
4. **A standalone audit and spending record:** save expected values for sampled pairs/rows before discarding distance blocks, raw worked examples, actual settings dropped by the cap, timings, memory and billed cost. A should produce a verifiable result even if the owner declines B.

For a first experiment I would keep the two-stage design, the fixed dense primary recipe, and the text-first viewer. I would cut M2 before cutting its local comparator, sample the expensive functional readout before reducing geometric coverage, and move the alternative graph recipes out of the critical path. The viewer contract also needs one reconciliation: it offers eight layers and ten neighbours but budgets files for four layers and eight neighbours, and a dot-reading winner cannot be the advertised shipped mode if the viewer only offers cosine (§12.1, lines 411–419, 442–447). Ship fewer choices if necessary; make every offered choice genuine.

**Unverified and blockers:** I inspected `/home/ubuntu/code/TransformerModes/exp1/local_pilot.py`, both pilot JSONs, `exp1/bench_numerics.json`, the numerical test and kernel sources, and `/home/ubuntu/code/TransformerModes/reviews/numerics-prototype-sol.md`. I did not rerun model extraction, the benchmark suite, or any paid job; I made no repository edits. I could not independently recover the printed 0.011 random-pair baseline from the cited JSON, verify real-data tie rates, matched coverage or projection exhaustion, or establish CuPy peak memory, wall time or billed cost. Those are measurements for the respective pilots. No tool failure blocked the review.
