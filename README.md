# TransformerModes

Spectral pictures of what language models do to text: token positions are the points, a similarity between tokens'
recent pasts makes a graph at every layer, and one operator ties the layers together. Split out of
[CoherentModes](https://github.com/CharlesR-W/CoherentModes) on 2026-11-23; it reuses the through-depth operator of
[DepthEmbedding](https://github.com/CharlesR-W/DepthEmbedding).

## Status

- **Experiment 1** (`SPEC.md` v3.2, code in `exp1/`): memory kernels on Pythia-1b and OLMo-1B, on the same 32 texts.
  The spec went through four rounds of independent review (`reviews/`). Stage A, the neighbour census, is running on
  Modal; Stage B (through-depth modes) is provisional and needs its own approval.
- `deprecated/`: the earlier Pythia-160m, Mamba and DINO studies, copied from CoherentModes for reference. Not
  maintained; see `deprecated/README.md`.

### Stage A run log (Modal app and Volume `tm-exp1`)

Spend figures are the ledger's estimates from runtimes and published rates, not reconciled against Modal's bill.

| Stage | Run id | Outcome | Estimated spend |
|---|---|---|---|
| probe | `20261123-190845-ad04be6d` | done: 180,378 documents streamed, 23,350 eligible, 32 pool texts and 4 stitched contexts; token-identity check passed | $0.05 |
| extract | `20261123-192555-9ff86eda` | Pythia done; OLMo failed before any model work (a Hugging Face downloader log file left open on the Volume blocked `volume.reload()` in a reused container) | $0.18 |
| extract | `20261123-193404-7017cd5b` | OLMo done after disabling that downloader; hook equality exact and causal check passed for both models | $0.12 |
| A-pilot | `20261123-200736-7e80ef52` | done: 63 s per census pass on an A100 (estimate was about 20 s), 5.9 GB peak; one audit row 4.3 s | $0.37 |
| census | `20261123-201624-a982d428` | running: all four settings groups approved | |

The cap for Stage A was $8 at approval and was raised to $11 by the owner after the pilot, so that the dot reading
would not be dropped (expected total about $7.30; the remainder is retry holds and a 20% margin).

## Later (owner's list)

- **A retrieval-style baseline.** Embed the same contexts with a sentence-embedding model (SentenceTransformers or
  the embedding model behind a vector database) and run the same graph construction on those vectors, as an outside
  reference for what a similarity between contexts usually means.
- **Hankel / predictive-state construction.** Compare each token's history by what it predicts about the future, in
  the manner of Golowich, Liu and Shetty, *Sequences of Logits Reveal the Low Rank Structure of Language Models*
  (the "lingen" paper): the extended logit matrix of histories against continuations.
