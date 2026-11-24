# TransformerModes

Spectral pictures of what language models do to text: token positions are the points, a similarity between tokens'
recent pasts makes a graph at every layer, and one operator ties the layers together. Split out of
[CoherentModes](https://github.com/CharlesR-W/CoherentModes) on 2026-11-23; it reuses the through-depth operator of
[DepthEmbedding](https://github.com/CharlesR-W/DepthEmbedding).

## Status

- **Experiment 1, Stage A** (`SPEC.md` v3.2, code in `exp1/`, site in `site/`): the neighbour census on Pythia-1b and
  OLMo-1B over the same 32 Pile texts is complete and passed its independent audit (1,242 checks, 32 full rows
  recomputed from raw activations). The spec went through four rounds of independent review (`reviews/`).
- **Stage B** (through-depth modes) is provisional and has not been approved or run.
- `deprecated/`: the earlier Pythia-160m, Mamba and DINO studies, copied from CoherentModes for reference. Not
  maintained; see `deprecated/README.md`.

### Stage A run log (Modal app and Volume `tm-exp1`, 2026-11-23)

Spend figures are the ledger's estimates from runtimes and published rates, not reconciled against Modal's bill. The
ledger's total is about $10.31. The cap was $8 at approval, raised by the owner to $11 after the timing pilot and to
$13 for the re-run after the audit's finding.

| Stage | Run id | Outcome |
|---|---|---|
| probe | `20261123-190845-ad04be6d` | done: 180,378 documents streamed, 23,350 eligible, 32 pool texts and 4 stitched contexts; token-identity check passed |
| extract | `20261123-192555-9ff86eda` | Pythia done; OLMo failed before any model work (a Hugging Face downloader log file left open on the Volume blocked `volume.reload()` in a reused container) |
| extract | `20261123-193404-7017cd5b` | OLMo done after disabling that downloader; hook equality exact and causal check passed for both models |
| A-pilot | `20261123-200736-7e80ef52` | done: 63 s per census pass on an A100 (the estimate was about 20 s), 5.9 GB peak; one audit row 4.3 s |
| census | `20261123-201624-a982d428` | stopped after 27 minutes: the Modal workspace reached its billing-cycle spend limit and every container was terminated; five of eight groups were saved. The two worker attempts were charged by hand for their elapsed time |
| census | `20261123-225108-bbdc3492` | done: resumed from the saved groups; all four settings groups for both models |
| readout | `20261123-231754-192a4b7e` | failed: a 40-minute limit expired inside the synthetic census (100 runs, about 34 minutes, which the pilot had not timed) |
| readout | `20261124-000059-65414cee` | failed: the worker was restarted 20 minutes in and repeated the synthetic census; the coordinator's 65-minute limit expired first. Charged by hand |
| readout | `20261124-011236-1cc2737b` | done: steps made resumable, synthetic census reduced to the first group (cosine, corpus mean: 25 runs). Selection 7 min, synthetic 10 min, packing 2 min |
| audit | `20261124-013611-6d8af28f` | failed: finished every check, then hit a 30-minute limit while hashing the published files, so no report was written |
| audit | `20261124-021002-ef20273f` | **audit failed**: 8 of 1,242 checks, all `readout.document_contributions`. Cause: an int32 overflow in `readout.Predictions.overlap` zeroed the top-20 prediction overlap for neighbours and controls at pool indices above 42,713 |
| readout | `20261124-033722-05537416` | done: prediction readout recomputed with the one-line fix (int64 keys); concordance scores rose about 1.4x, top-1 rates and the selection unchanged |
| audit | `20261124-034712-f75d9d17` | **audit passed**: 1,242 checks, none failed; inventory of 4,423 published files |

Two things the run taught that are not in the spec: the app's own `fetch` downloads one file at a time and takes
over 25 minutes for the viewer data (use `modal volume get tm-exp1 <dir>` instead, 27 seconds); and the
participation-ratio rows have very flat tails on these activations, so "walk weight" and "the 64 nearest" give very
different pictures of the same graph.

## Later (owner's list)

- **A retrieval-style baseline.** Embed the same contexts with a sentence-embedding model (SentenceTransformers or
  the embedding model behind a vector database) and run the same graph construction on those vectors, as an outside
  reference for what a similarity between contexts usually means.
- **Hankel / predictive-state construction.** Compare each token's history by what it predicts about the future, in
  the manner of Golowich, Liu and Shetty, *Sequences of Logits Reveal the Low Rank Structure of Language Models*
  (the "lingen" paper): the extended logit matrix of histories against continuations.
