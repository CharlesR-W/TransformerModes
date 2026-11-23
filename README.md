# TransformerModes

Spectral pictures of what language models do to text: token positions are the points, a similarity between tokens'
recent pasts makes a graph at every layer, and one operator ties the layers together. Split out of
[CoherentModes](https://github.com/CharlesR-W/CoherentModes) on 2026-11-23; it reuses the through-depth operator of
[DepthEmbedding](https://github.com/CharlesR-W/DepthEmbedding).

## Status

- **Experiment 1** (`SPEC.md`, code in `exp1/`): memory kernels on Pythia-1b and OLMo-1B, on the same 32 texts.
  Spec under review; nothing has been run and nothing has been spent.
- `deprecated/`: the earlier Pythia-160m, Mamba and DINO studies, copied from CoherentModes for reference. Not
  maintained; see `deprecated/README.md`.

## Later (owner's list)

- **A retrieval-style baseline.** Embed the same contexts with a sentence-embedding model (SentenceTransformers or
  the embedding model behind a vector database) and run the same graph construction on those vectors, as an outside
  reference for what a similarity between contexts usually means.
- **Hankel / predictive-state construction.** Compare each token's history by what it predicts about the future, in
  the manner of Golowich, Liu and Shetty, *Sequences of Logits Reveal the Low Rank Structure of Language Models*
  (the "lingen" paper): the extended logit matrix of histories against continuations.
