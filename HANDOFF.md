# Handoff (2026-11-24)

Written at the owner's request when work paused. It says what exists, what was found, what is half-done, and what
the owner has and has not approved. Read this first, then `README.md` (status and the Stage A run log).

## 1. Where things stand

| Part | State |
|---|---|
| **Stage A** (neighbour census, Pythia-1b and OLMo-1B, 32 Pile texts) | Run, audited (1,242 checks, none failed), published: <https://charlesr-w.github.io/TransformerModes/>. Spec as run: `SPEC.md` v3.2. |
| **Stage B** (independent chunks, three ways of comparing them, a product with the model's next-token prediction, through depth) | Designed and reviewed three times: `SPEC_B.md` v3.3. Gate and numerical core written (`exp1/stage_b.py`, `gate_b.py`, `test_stage_b.py`; 29 tests pass). **Nothing has been run and nothing has been spent.** |
| Deprecated studies (Pythia-160m, Mamba, DINO) | Copied to `deprecated/` with banners; not runnable from here. Originals are still in CoherentModes, untouched. |
| Repo | Public, `main` pushed. The site is the `gh-pages` branch (one orphan commit, force-pushed). |

The owner paused here to review the outputs and decide how to proceed. Do not launch anything on Modal without a
new go-ahead from the owner.

## 2. What the owner has decided (do not re-litigate)

- Models: `EleutherAI/pythia-1b` and `allenai/OLMo-1B-0724-hf` on the same texts. Corpus: the Pile test split,
  all text may be shown ("if it's in the Pile it's fine").
- The kernel: end-aligned pasts with decaying weights, `K_a(t,t') = sum_s a^s <y_{t-s}, y_{t'-s}>`. Memory as a decay
  rate is what was meant; it is not a hard cut-off.
- **No exclusion windows and no mean-removal "fixes".** The owner called the window a hack. Measure and show; do
  not intervene. (One global mean per model and layer, the ordinary cosine construction, stays.)
- Few views, plain words (the owner did not understand "pooled"), every index explained in a primer, easy to verify.
- Roles: Opus 5.5 for design, judgement and user-facing text; GPT-6.2 Astra for iterative critique; GPT (Sol) for
  implementation; no Gemini writer for now.
- Publishing: public repo and Pages site; finished experiments go in the Experiments tab and are deployed without
  asking.
- Stage B direction: independent chunks ("big aggregated blocks", done the smart way) and something "like or
  using" the Hankel / logit-matrix predictive state "which makes sense in this through-layer context".
- Stage B approvals given so far: **stage (i), the known-answer gate, on Modal with a hard maximum of $1.15**; and
  that the **smallest tier is an acceptable result** if the timing pilot forces it. Stages (ii) extraction and pilot
  (maximum $3.90) and (iii) completion have **not** been approved; each needs its own request with the previous
  stage's results.
- Later, not now (`README.md`): a SentenceTransformers / vector-database style baseline; the multi-step Hankel
  version with probe continuations (Golowich, Liu and Shetty's "lingen" paper).

## 3. What Stage A found

Numbers are on the site and in `/var/tmp/tm_release/analysis/` (the fetched, audited outputs; see section 7).

- **Memory pulls a token's nearest neighbours into its own text and next to it.** Layer 8, cosine: own-text share
  59% with no memory to 98% at memory 64 in Pythia (62% to 99% in OLMo); at the whole prefix about 92% of
  neighbours are within 64 positions. The two models agree closely throughout.
- **It is the model, not the kernel**: random vectors per token id through the identical kernel put 9% of
  neighbours within 64 positions at memory 64 where the models put about 50% (the kernel pairs lag with lag;
  neighbouring positions share no vectors). At the whole prefix the kernel alone already gives 34%.
- **Memory makes neighbours less predictive.** Against controls matched on text, current token and distance band,
  the top-20 overlap advantage is 0.133 with no memory, 0.063 at memory 8, 0.017 at 64, 0.004 at the whole prefix
  (top-1 match 23% against 10% with no memory). Neighbours within 64 positions could never be matched and are
  outside this score (47% of pairs at memory 64).
- **The participation-ratio graph has very flat rows** on tokens: half a row's walk weight can lie in other texts
  while under 1% of its 64 nearest do. "Walk weight" and "the 64 nearest" are different pictures of one graph.
- **The dot-product reading at the whole prefix is dominated by prefix length** (text-only vectors reproduce it).
- The pre-registered rule for Stage B picked dot product, memory 64; the owner overruled it (see `SPEC_B.md`).

Things I told the owner that turned out wrong and were corrected (the corrected versions are what the docs say):
"the graph is nearly a chain" (only nearest-neighbour shares were measured); that the screen differed from the
census because it used the last 512 tokens (it is the flat walk weight); that the product walk keeps only patterns
smooth in both views (a flat layer passes every prediction pattern through); that it "reduces to the Hankel SVD"
(what holds is sample canonical correlations with a one-step prediction).

## 4. Stage B: exactly where it stopped

Design (`SPEC_B.md` v3.3): 2,048 Pile documents, one random cut each; the node is the model's state at the cut. Three
comparisons of two chunks (state at the cut; lag by lag, memory 8; average over the last 64 states), cosine,
participation ratio 64 with no floor. Layer walk `A_l` (lazy), prediction walk `S_F` (Hellinger affinity of
next-token distributions, not lazy), product `M_l = A_l S_F A_l`, through-depth operator unchanged with units and
coupling fixed at the past-only partner's. One primary test: the energy cost of agreeing with the prediction, state
at the cut, against 199 wrong pairings that keep the cut token, p <= 0.05 required in both models. A product mode's
appearance is never evidence; only pairing-dependent quantities are. Three separately approved stages with maxima.

Next steps, in order, when the owner says go:

1. **Commit the frozen gate code** (done with this handoff: `exp1/stage_b.py`, `gate_b.py`, `test_stage_b.py`).
   `SPEC_B.md` section 9 requires the generator code, grid and stopping rule to be committed before any development
   seed runs.
2. **Run the fixtures and the K0-only calibration locally (free)**, from that commit:

   ```bash
   cd exp1
   TMPDIR=/var/tmp uv run --offline --with numpy --with scipy --with pytest \
     python -m pytest -q --basetemp=/var/tmp/tm_pytest_b test_stage_b.py test_spacetime.py   # then rm -rf the basetemp
   TMPDIR=/var/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --offline --with numpy --with scipy \
     python gate_b.py --calibrate --out calibration_b_v33
   ```

   Development seeds are 1170 to 1174. It looks only at K0's quantities and either writes
   `calibration_b_v33/parameters.json` (+ `.sha256`) and reports "parameters frozen", or exits 2 with "fixture
   uninformative at design". **Do not adjust anything in response.** If it is uninformative, the fixture needs a
   redesign (new spec version, fresh confirmatory seeds, a new gate-only approval from the owner). The designer's own
   expectation, from reasoning only, is that the prediction-only capture (minimum 0.55 required) is the likelier
   failure. A smoke run at N = 96 on a non-gate seed returned "uninformative"; that size says little.
3. **Commit the calibration outputs**, then run the confirmatory seeds 70, 71, 72 **once**, on Modal under the
   approved $1.15 maximum (the gate has no Modal unit yet: add one to `exp1/modal_app.py` with a separate Stage B
   ledger as `SPEC_B.md` section 13 describes; about 8 to 22 minutes per seed by the implementer's estimate):

   ```bash
   python gate_b.py --only all --seeds 70 71 72 --parameters calibration_b_v33/parameters.json --out confirmatory_b_v33
   ```

   Exit 0 "gate passed"; exit 1 "gate conditions not met" or a withheld solve; exit 2 "fixture uninformative".
   None of these authorises extraction.
4. Bring the gate's result to the owner with the stage (ii) request (extraction and timing pilot, maximum $3.90).
   Stage (ii) needs code that does not exist yet: the Stage B probe (one cut per document, full-text hash
   de-duplication, the moved-fraction check before extraction), a narrow extraction mode that reads the native
   last-position logits, the pilot, the Stage B audit (to be written by a different agent than the pipeline's
   author), and the site's "Through depth" tab and entries E4 to E6.

Not done for Stage B, so that nobody assumes otherwise: no seed of any kind has been run; the gate's thresholds
(gain >= 0.2 where the belief is planted, <= 0.05 elsewhere) have never been exercised; all costs are arithmetic.

## 5. Things that will bite

- **The Modal workspace has a billing-cycle spend limit** that terminates every container when reached (it stopped
  the census once; the owner raised it). A stalled run with no error and no containers is probably this. Check with
  a trivial launch: it raises `ResourceExhaustedError: workspace billing cycle spend limit reached`.
- **Time limits caused most wasted spend in Stage A** (five failed attempts). Every unit's limit must come from a
  measured pilot; the coordinator must outlast its units; long units must commit per step and print per-step
  timings; nothing optional may sit in front of critical outputs.
- **`modal_app.py fetch` downloads one file at a time** (over 25 minutes for the viewer data). Use
  `modal volume get tm-exp1 <dir> <local>` (27 seconds), into a fresh empty directory, and verify with
  `audit_a.verify_inventory`.
- **Integer widths**: pool index times vocabulary size overflows int32 (the bug the audit caught in `readout.py`).
- **`spacetime.modes` can skip members of an exactly repeated eigenspace** while reporting tiny residuals. Residuals
  never certify ordering; `stage_b.py` has an exact branch for identical sites and `SPEC_B.md` plans an inertia
  check. The same code is in DepthEmbedding; it only matters when eigenvalues tie exactly.
- **The viewer's tie marks were only ever exercised on planted cases**: no real viewer row has a tied cutoff.
- **The Stage A ledger's figures are runtime estimates**, never reconciled against Modal's bill ($10.31 estimated;
  cap raised $8, $11, $13 by the owner). The two-attempt rule was overridden twice with the owner's approval.
- **This VPS**: 4 cores, 8 GB, swap usually full, disk at 95%, shared with other sessions' jobs. The audit's
  large-pool test writes about 2.4 GB of pytest temp files: always pass `--basetemp` and delete it. Agents have
  stalled here under load (two did); restart with a fresh agent and a complete brief rather than resuming a very
  long one.
- **Background-job notices can mislead.** "Completed (exit code 0)" has several times been a wrapper's exit, a
  `tail`'s status or a wait loop dying on its own bug. Read the job's own output before reporting a result.

## 6. Layout

| Path | What it is |
|---|---|
| `SPEC.md`, `SPEC_B.md` | Stage A as run; Stage B as designed. `BRIEF.md` is the original design brief with the owner's request. |
| `reviews/` | Every independent review (GPT-6.2 Astra) of both specs, and the numerics prototype report. |
| `exp1/` | Pipeline: `probe.py`, `extract.py`, `census.py`, `readout.py`, `select_b.py`, `pack_site.py`, `audit_a.py` (independent; numpy only), `modal_app.py` (units, ledger, coordinator), `synthetic.py`; Stage B: `stage_b.py`, `gate_b.py`. Reused from DepthEmbedding unchanged: `spacetime.py`, `spacetime_stage.py`, `kernels.py`. Sparse prototype kept for later: `similarity.py`, `graph.py`, `solve.py`. `runs/` holds launch records. |
| `site/` | Site sources and `build_site.py` (a release build refuses unless the audit passed and the file inventory matches). `site/dev_fixture.py` builds a small local fixture. |
| `deprecated/` | The old studies. |
| `build/`, `data/` | Not in git. |

Tests. State at this commit: `test_stage_b.py` and `test_spacetime.py` were run on the final Stage B code (29
passed). The Stage A suites last ran together before the readout fix's follow-up edits (102 passed, 2 failed, 1
skipped; the two failures were test expectations, fixed, and the two affected suites then passed, 33 passed); the
whole set has not been re-run in one go since. The audit suite needs about 5 minutes and 2.4 GB of temp space:

```bash
cd exp1 && TMPDIR=/var/tmp uv run --with torch==2.5.1 --extra-index-url https://download.pytorch.org/whl/cpu \
  --with transformers==4.57.6 --with numpy --with scipy --with zstandard --with pytest --with modal==1.6.2 \
  python -m pytest -q --basetemp=/var/tmp/tm_pytest test_audit_a.py test_modal_app.py test_probe_extract.py \
  test_census.py test_pack_site.py test_numerics.py test_spacetime.py test_stage_b.py
```

Site: `uv run --with numpy python site/build_site.py <data dir> --out build/site`, then copy `build/site/` into an
empty directory, commit it as one orphan commit on `gh-pages` and force-push (see DepthEmbedding's README).

## 7. Data and scratch

- **Modal Volume `tm-exp1`** (about 60 GB): `probe/`, `acts/<model>/c<cc>.npy` (all 17 hidden states of 36
  contexts, float32), `pred/`, `attn/`, `census/` (top-64 lists with tie records for every token, layer, memory and
  mask), `readout/`, `analysis/`, `site/`, `audit_a/`, `ledger/`. Everything in Stage A is recomputable from `acts/`
  without the models. Deleting it is the owner's call.
- **On this VPS, safe to delete** (the owner was told): `/var/tmp/tm_release` (the audited fetch the site was built
  from; refetchable), `/var/tmp/tm_real_dev` (a partial fetch a safety check would not let the agent remove),
  `/var/tmp/tm_real_dev2`, `/var/tmp/tm_deploy`, `/var/tmp/tm-site` (builds and screenshots), `/var/tmp/tm_diag`
  (2.1 GB of downloads from the audit diagnosis), `/var/tmp/tm_probe`, `/var/tmp/tm_frozen`.

## 8. CoherentModes

Left untouched, including its 65 uncommitted changes. Its `explore/ncs_report` build still reads the directories
that were copied here, which is why they were copied and not moved. Removing the originals there, once its tree is
clean, is the owner's decision.
