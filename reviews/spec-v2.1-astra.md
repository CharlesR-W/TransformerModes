**Recommendation: do not approve the $26 full run on the strength of §0 as written.** The move to every-token graphs is scientifically attractive: it removes a sampling confound, lets the reader see the same 1,792 tokens in both views, and makes the one-text versus all-texts comparison much easier to explain. A sparse graph defined in advance, with a dense oracle for the *same* operator, is also a sound plan. I would keep both changes.

I found four problems that change the go/no-go decision even if the pending prototype matches the dense solver perfectly:

- The no-memory candidate gets a text-baseline score of zero by assumption, although its text-only signal is **not** removed by the specified nuisance span.
- The truncation check, read literally, counts the excluded token itself and **must fail every row** of the default graph.
- Two claimed “known answers” are statistical predictions, and the four-layer synthetic run is given the **eight-layer** acceptance formula.
- The priced production rows do not add up at the specified rates. In particular, U3 alone costs **$10.84, not $6**, even if its 12-minute runtime is right.

This is a review of `/home/ubuntu/code/TransformerModes/SPEC.md` v2.1. Section and line numbers below refer to that file. I treated the owner’s publication decision and graph terminology as settled.

### Disposition of the twelve original objections

| # | Status | Re-check |
|---|---|---|
| 1. Wrong noise null | **Partly resolved** | The invalid same-field A/B gate is gone, independent fields are used, and the selected recipe is tested. H2 still compares a structured prefix experiment with an individual-token permutation null; independent fields need not be exchangeable over positions. |
| 2. Winner can reward surface text | **Partly resolved** | The no-winner outcome, effective-rank check, fixed-graph bootstrap label and separate functional readout are real improvements. The zero baseline at memory none and the interpretation of the new margin are material errors. |
| 3. A/B is not a ceiling | **Partly resolved** | The ceiling and unbiased-sampling inference are gone. The replacement question is sound. Its 5% retained-trace rule can invalidate even perfect agreement, and one of the two “16-text” graphs contains 17 texts. |
| 4. Cost lacks support | **Partly resolved** | A capped pilot and explicit stop are the right response. The budget arithmetic, pilot coverage and fallback do not yet support the advertised total. |
| 5. Noise structure and meaning of the two readings | **Resolved** | The diagonal dependence, whole-prefix length effect and cost of arbitrary lag profiles are now correctly explained. Keep **“approximately”** in the primer for memory 64: a 2% distance bound is not an assurance of identical nearest-neighbour ordering. |
| 6. Controls confuse invariants and predictions | **Partly resolved** | The revised stitches are much better: same source pair, random lengths, position-only baseline, residualised label, no whole-prefix prediction. H1 is still described as an exact positive when it is not one. |
| 7. Spectral cutoff and token-space summaries | **Resolved for the original errors** | The interpretation of the 16/17 gap, weighting of repeated token directions, chance dimensions, captured-share rank and meaning of \(t^*\) are corrected. Floor-induced multiplicity creates a new issue below. |
| 8. Pilot did not test shipped joint modes | **Resolved for the original objection** | P2 now tests joint modes with the right two-token statistic and limits its conclusion to the configured recipes. The new sparse rule still needs a check of artificial connectivity, not just localisation. |
| 9. Audit and click-through cannot verify the result | **Partly resolved** | The audit now runs beside the arrays, checks all candidates, and checks neighbour membership rather than just stored neighbour values. The worked-token design is good. The cost of the literal audit is not established, and mode slices alone cannot verify numerical acceptance. |
| 10. Mode toggles invite false comparisons | **Resolved in design** | The warning, invariant agreement on the same text, disabled impossible graph choice, and single Details disclosure address this well. The shared-token agreement must use the corrected retained-signal rule. |
| 11. Q5 and causal precision | **Resolved in substance** | Direct attention and residual similarity are correctly presented as descriptive, not causal context lengths; the equal-shape suffix test is appropriate. Choose a replacement token *known to differ* at the cut rather than assuming the two texts differ at position 1,024. |
| 12. Public text | **Resolved by the owner** | No publication objection in this review. The description of what population was sampled is also improved. |

### Material findings

**1. Blocking — memory none has an unearned advantage, and the margin is not a measure of shared information after controlling for text.** §§4, 7.3–8, lines 170–173, 289–300, 325–347; E2, line 514.

“Function of the current token” is not the same as “in the span of indicators of IDs seen at least 20 times.” For example, a function distinguishing two rare identifiers in a single program can be entirely lexical and remain in the remainder. By skipping the adversaries and inserting zero at memory none, the rule charges every memory candidate for its text alignment but never charges the no-memory candidate. Neither the effective-rank gate nor checking three cutoffs fixes that asymmetry.

There is a second, independent problem. Subtracting the largest **model-to-adversary** alignment from **model-to-model** alignment is a contrast between similarities, not a partial alignment. For example, let both models have exactly the same remainder \(B+S\), the adversary reproduce \(B\), and \(B,S\) be orthogonal in Frobenius inner product. If \(\|B\|=4\|S\|\), the margin is \(1-4/\sqrt{17}=0.030\): it fails even though the part not in the adversary is replicated *perfectly*. Conversely, two models can encode the same text-only pattern that neither of the two adversaries captures and score highly. Comparing a one-site adversary to a depth-weighted picture adds a further difference in the weighting of directions.

**Change:** give memory none a real baseline. A clean option is an analytic token-type equality representation or a token-type-level calculation, not a nearest-neighbour graph with hundreds of exact ties. For every memory, if the question is agreement *outside the tested adversaries*, project both model factors off the **same** prespecified text-feature span, then report residual energy, effective rank and alignment. Check that the projection has not exhausted the token space. Alternatively keep the present margin as a heuristic, apply it symmetrically, and call it “agreement advantage over these two recipes”—not evidence of information beyond text. In either case run a text-only counterexample through `decide.py` before approval. Name the actual failed gate in E2; no winner does not necessarily mean no margin exceeded 0.05.

**2. Blocking — V3, as written, cannot pass, and its fallback does not establish dense equivalence.** §§4–5, lines 178–187, 196–211; §11, lines 427–432.

The numerator sums over \(j\notin N_i\). Since self is explicitly excluded from \(N_i\), that includes \(i\). With PR \(k=64\), the largest conditional weight is at least \(\sum_j q_j^2=1/64\). The self-distance is zero, no larger than the nearest kept distance, so its extrapolated weight is **at least \(1/64=0.015625\)**. That alone violates the maximum threshold of 0.01 in every row, at either 512 or 1,024 neighbours.

After excluding self, the stated quantity is still **tail mass divided by kept mass**, at the *truncated* bandwidth. If that ratio is \(r\), the fraction lost at that bandwidth is \(r/(1+r)\). The fully calibrated dense rule uses a different bandwidth. Small row tails are useful evidence but not a guarantee about the lowest eigenspace: a very small omitted edge can be the only bridge between large regions. V3’s direct dense comparisons are valuable for the named small Pythia/cosine cases, not for every production graph. It tests PR even if P2 ultimately selects a different recipe. And the 1,024 fallback requires new lists and tail checks: 512-entry files do not contain neighbours 513–1,024.

**Change:** specify \(j\ne i,\ j\notin N_i\), identify the denominator and bandwidth, and separate three conclusions: “the implementation is correct,” “this sparse recipe resembles the dense recipe on these cases,” and “this is the sparse recipe we chose irrespective of dense drift.” A failed V3 can be compatible with the third, not the second. Benchmark or provision the 1,024 rerun, and specify what evidence accompanies the recipe actually selected.

**3. Blocking for the interpretation of the modes — the floor can define the picture without the document gate detecting it.** §0, lines 44–53; §5, lines 196–216; §§6–8, lines 242–266, 328–337.

The sparse-plus-rank-one representation is legitimate. The interpretation at lines 213–216 is not generally true. The added off-diagonal entries sum to \(10^{-4}\) per row **before scaling**. The symmetrised \(S\) need not have row sums one. After Sinkhorn the floor contributes \(\eta d_i(\sum_jd_j-d_i)\) in row \(i\), not a fixed fraction. Components can be formatting classes, parts of texts, or groups spanning several texts; they need not be documents. One weakly connected cut can matter as much as an exactly disconnected one.

There is an instructive exact case. Take a regular, symmetric, row-stochastic \(S\) with \(b\) disconnected components, identical at every layer. Every centred function constant on each component has lazy-walk gap
\[
1-\lambda(M)=\frac{N\eta}{2[1+(N-1)\eta]} .
\]
There are \(b-1\) such directions. If \(b\ge9\), the eight gaps setting the layer’s unit \(c_\ell\) can **all be floor gaps**. Dividing by \(c_\ell\) then makes the artificial connectivity the ruler for the entire operator. If a cutoff splits the resulting multiplicity, the solver can return a numerically excellent but arbitrary slice of that band. A remainder threshold of 0.25 does not exclude this example when the components do not coincide with the document indicators. Counting components alone also misses near-disconnections.

**Change:** for each candidate compute the **fraction of each low mode’s spatial energy due to floor edges**; this is cheap from \(d\), the mode slices, and \(\eta\), without a dense matrix. Report which components actually contain which tokens, not just their count. For finalists with appreciable floor energy or very small gaps, repeat with a prespecified smaller/larger floor or larger neighbour list and compare the *space and decision*. Mark a floor-defined or unresolved band as such rather than presenting its numbered modes as model structure. Put the exact regular-component example in the solver tests.

**4. Blocking — H1–H3 are not all invariants, and the synthetic acceptance formula has the wrong layer count.** §§6, 9.1, lines 257–266, 351–380; E1, line 513.

- **H1:** “strong” still has Gaussian noise with full support, token effects, no regime at the first site, and Markov regime changes inside the memory window. At switch probability \(1/128\), about **39.5%** of 64-token histories contain a switch. There is no exact result that a thresholded recurrent subspace must capture 99% of the *current* regime at both memories. Four successes out of five would be a power result; failure would not establish a numerical bug.
- **H2:** independent white-noise *fields* are the correct improvement. But both runs still share prefix lengths and the same sequence geometry. The spec itself derives position-dependent cosine variance at whole-prefix; for unit tokens the expected unscaled dot distance between unrelated prefixes is \((t+t')/2\). A within-text permutation destroys the shared position structure. Independence of the fields does **not** imply alignment below that permutation threshold. Finding position geometry shared by independent inputs would be an important result, not a reason to stop.
- **H3:** permuting token **order before** the memory calculation changes the scientific input and should change the answer. Permuting node **labels after** memory features are defined should not. Ties broken by current array index can also violate equivariance. A tied spectral cutoff need not give an equivariant individual projector of that rank.
- **Acceptance:** line 263 hard-codes \(4\sin^2(\pi/16)\). The formula is \(4\sin^2(\pi/(2L))\); for the four-site synthetic run the ratio of the correct value to the specified value is **3.84776**, not a rounding discrepancy. A correct implementation will fail that gate if it follows the text.

**Change:** use a genuinely analytic fixture for hard recovery—for instance an explicitly constructed feasible block walk with known invariant subspace—and retain “strong” as a power measurement. Calibrate a structured noise reference by redrawing whole fields with the same index geometry, or report H2 by position rather than gating it. Define H3 using immutable token IDs and predecessor links, and compare a complete isolated band. Parameterise every temporal formula by \(L\).

**5. Should-fix — V1/V2 are useful solver checks, but they can agree on the same wrongly constructed graph.** §§5–6, lines 196–203, 248–266; §13, lines 545–556.

Densifying a misassembled sparse walk and solving it two ways can pass the dense comparison, residual, trace, and row-sum checks. There is a concrete porting trap: `/home/ubuntu/code/DepthEmbedding/opt/sparse_core.py:38–68` uses \((1-\eta)A+\eta(11^\top-I)\) for a **binary off-edge-floor** recipe. The new weighted PR recipe requires \(S+\eta(11^\top-I)\). `Walk` is reusable; its kNN constructor is not the PR constructor unchanged. Likewise `site_spectra` at lines 328–342 computes bottom eigenvalues and a Frobenius statistic in addition to the needed leading spectrum. “Arithmetic unchanged” is not a specification for either port.

**Change:** add a tiny **independent dense weights-to-walk oracle** with finite truncation, an asymmetric directed neighbour list, nonconstant Sinkhorn factors and a deliberately visible floor. Compare \(q,S,P\), random-vector products, site scales and the two energy terms before comparing eigenpairs. Test both PR and binary recipe conventions. Include a near-cut and a cluster spanning ranks 20/21; compare complete clusters with guard eigenpairs, not an arbitrary partial cluster. Run these cheap tests before any large dense V job. The existing \(K_{\rm nb}=N-1,\eta=0\) check is worth keeping but exercises neither finite truncation nor the floor.

**6. Should-fix — the companion design asks the right question, but its no-signal rule and graph labels can give the wrong answer.** §§3, 7.2, 7.4, lines 122–129, 280–285, 313–315; E3, line 515.

If “5% retained” is the displayed centred restricted trace divided by the original trace, an ideally balanced all-texts picture puts only **\(1/32=3.125\%\)** of its energy on each text *before* centring. It could be a perfect copy of the one-text pattern on every text and be reported as “no signal” on all of them. If the intended denominator is the uncentred restriction, write that down. Also, after reserving the focal text there are 31 companions. A set of 15 and “the other 16” yield graphs of **16 and 17 texts**, as the node counts on line 126 correctly imply. The 17-text result should not be a second marker at 16.

**Change:** report two numbers separately: focal-text occupancy relative to its \(1/32\) node share, and the fraction of **that text’s own restricted energy** surviving centring. Use the latter and an explicit small-norm guard for the no-signal decision. Use two disjoint groups of 15 companions, leaving one unused, for a matched-size comparison. With two focal texts and one draw each, call H1/H2 illustrative composition sensitivity, not a population estimate. No additional resampling campaign is needed.

**7. Should-fix — the functional readout is useful but the current control does not match document or overlapping history as claimed.** §1, lines 66–67; §7.4, lines 316–323; E2, line 514.

For a neighbour from text B of a query in text A, the control can currently come from **any** non-A text. Prediction overlap can therefore improve simply because B is a related subject or shares vocabulary with A. For a within-text neighbour, the control can be far away while the selected neighbour is adjacent, sharing almost the same observed prefix. Both are potentially useful descriptive findings, but neither supports the stronger “beyond matched document/context” reading. Also \(o(i,j)\) is a valid lower bound on full-distribution overlap; the **difference of two such lower bounds** is not a lower bound on the difference in full overlap.

**Change:** exclude self, the selected neighbour, and invalid control duplicates explicitly. Match the control to the neighbour’s **actual document and current token ID**, and for within-text pairs match a coarse lag bin or report adjacent/near and distant pairs separately. Report coverage by construction and the fraction of top-20 probability mass; call the quantity “top-20 overlap advantage.” If exact matching leaves too few pairs, that is a useful limit to report. Keeping this readout outside the winner rule is sound.

**8. Blocking for spend approval — the $26 estimate and $1 pilot are not supported even before uncertain performance is considered.** §0, lines 20–54; §11, lines 403–438; §12.4, lines 533–539.

Using the spec’s own rates and nominal runtimes:

| Item | Spec | Recalculation |
|---|---:|---:|
| U3: 16 × 12 min, A100 + 8 cores + 64 GiB | $6 | **$10.84** |
| U3c: 8 × 15 min | $3 | **$5.00 for GPU alone** |
| U2: 2 × 25 min, A100 + 8 cores + 64 GiB | $2.4 | **$2.82** |
| P1: one U3 and one U4 worker, both using the 45-min cap | about $1 | **up to $3.02** |

U3c also says “8 units” and “3 companion graphs × 2 constructions” per unit. There are 2 models × 2 focal texts × 2 constructions × 3 graphs = **24 solves**, not 48; specify the sharding before pricing it. The kept-list estimate needs similar accounting: uncompressed int32 index plus float32 distance for the stated all-texts **and** one-text main lists is about **64 GB**, before companion and adversary lists, rather than 36 GB. Compression or sharing might reduce that, but neither is costed.

Several expensive operations sit outside the timed representative solve. U4 allows **50 seconds per eight-site dense construction** including graph work and adversaries; it has 576 such solves. V3 calls for multiple dense four-text comparisons. U8’s literal direct-lag audit is not a token-sized spot check: with 50 queries, 57,344 candidates, width 2,048 and roughly 1,000 lags, whole-prefix alone is about \(1.17\times10^{13}\) FLOPs **per site/model/reading**. And a GPU sparse matvec with eight layers × 40 million entries must read about **3.84 GB** of FP64 values and int32 indices: 3 ms requires **1.28 TB/s** before other work. That is a possible benchmark target, not a transferable measurement. The polynomial fallback is not guaranteed faster; `/home/ubuntu/code/DepthEmbedding/opt/sparse_core.py:286–308` performs multiple underlying operator applications per reported polynomial call.

**Change:** do not respond by abandoning sparse graphs. Correct the ledger, then stage approvals: free correctness tests; a separately capped pilot with an explicit maximum bill; full run only if the remaining cost plus contingency fits. Time a real one-text group, one all-texts unit **including** calibration and \(t^*\), representative list generation, and a small benchmark of the audit algorithm. Charge fallbacks, dense validation and failed attempts. Count underlying sparse matvecs, not solver callbacks. If U8 is expensive, independently compute *one full row* for fewer prespecified queries across representative geometries, and use inexpensive random-pair spot checks elsewhere. Recompute decision metrics from slices, but say separately which numerical acceptance records are independently reproduced.

### What §0 can honestly tell the owner

**Supported:** extracting and reusing activations avoids eight repeated extraction campaigns; every-token graphs eliminate node-sampling noise and partly coloured texts; the end-aligned recursion is exact in its mathematical definition; sparse storage with an explicitly retained rank-one floor can represent the **chosen sparse walk** without a further approximation; dense-oracle testing and a capped pilot are appropriate safeguards. The new term “all-texts graph” and the concrete node counts explain the design well.

**Not yet supported:** “~20× faster,” “about half” the previous cost, adversaries adding 10%, controls running first “for cents,” and sparse runtime being approximately linear for this experiment. The cited `/home/ubuntu/code/DepthEmbedding/opt/REPORT.md:4–23,37–53,64–67` measured **CPU** prototypes at \(N=2{,}000\), principally binary 15-neighbour walks; its fast mode timing is for **16**, not 24, modes. It explicitly did not benchmark a GPU, a Gaussian kernel, or a full larger-\(N\) stage. The 38-second prior calibration was not a benchmark of repeated full-solve bisection as the §0 table implies. Fixed-degree storage is approximately linear; iteration counts need not be. The present schedule also spends on real extraction, lists, and real checks **before** the synthetic gates.

I would replace the speedup table’s last column with three labels: **measured in sibling experiment**, **derived for this design**, and **to be measured before approval**. Add one sentence the owner needs: *“The 512-neighbour graph is a different scientific operator, not just a faster way of computing the dense one; we will show whether the colours depend on the artificial connections between otherwise separate groups.”* Until the pending benchmarks and corrected gates exist, §0 supports the design direction, not the numerical price promise.

### Where to simplify

Keep the text viewer, one worked token, one no-memory contrast and the no-winner outcome. For a first experiment I would cut work in this order:

1. Treat PR as the **fixed primary recipe**. Use the small joint-mode comparison to report what Shannon and kNN do, not to change the construction after some validation and budgeting have already run. This removes a branch from the controls, cost gate and interpretation.
2. Replace a large stochastic “exact” gate with one tiny analytic end-to-end fixture, one near-disconnected fixture, and the existing empirical planted/noise results clearly marked as measurements.
3. Keep the two focal-text companion examples, but use equal companion counts. Do not add more focal texts until the retained-signal diagnostic works.
4. Reduce the *number* of costly audited rows before weakening their independence. A reader gains more from one fully checked row and one demonstrably non-floor-driven mode than from hundreds of largely redundant spot checks.

**Not verified or blocked:** I did not run the pending sparse prototype or use its results, run either model, access probe activations, start Modal, benchmark the GPU, or build the site. Actual cutoff gaps, graph components, post-Sinkhorn floor energy, achievable neighbour widths, matched-readout coverage, runtime and billed total remain unknown. The runtime figures above are arithmetic checks or bandwidth requirements, not performance measurements. I used only read-only source inspection and very light scalar calculations in this round. No tool failed or blocked the re-check.
