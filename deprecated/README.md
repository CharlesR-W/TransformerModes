# Deprecated studies

**Status: deprecated (2026-11-23).** These four directories were copied from
[CoherentModes](https://github.com/CharlesR-W/CoherentModes) (`explore/`, commit `4c932fa8`) when this repo was
created. They are kept for reference and are not maintained, built or deployed from here. The active work in this
repo is described in the top-level `README.md`.

| Directory | Study | Model |
|---|---|---|
| `lm_coherent/` | Coherent sets over 1,000 text snippets | Pythia-160m, 5 training checkpoints |
| `mamba_coherent/` | The same snippets, three views of the state | Mamba-130m |
| `mamba_contexts/` | 8,000 token contexts | Mamba-130m |
| `vit_coherent/` | CLS token through depth on ImageNet-100 | DINOv2 ViT-S/14 |

The directory names are unchanged so that the studies' references to each other still resolve.

## What does not work from here

- **Shared modules.** All four import `coherent.py`, `seba.py`, `schema_v2.py`, `construction.py`, `sweep_math.py`
  and `site/analysis.py` from CoherentModes' `explore/cifar_coherent/` through `../cifar_coherent` paths. That
  directory was not copied. `seba.py` is GPL-3.0-or-later.
- **Data.** No run outputs were copied. The Pythia and Mamba outputs live on the Modal volumes `cm-lm-coherent`,
  `cm-mamba-coherent` and `cm-mamba-contexts`; the DINO outputs (about 1.3 GB) are on `cm-vit-coherent` and in the
  CoherentModes working tree on the VPS.
- **Pages.** The built Mamba pages are included as single HTML files (`mamba_coherent/page/mamba_coherent.html`,
  `mamba_contexts/page/mamba_contexts.html`). The Pythia page and the DINO site are build outputs and were not
  copied. `vit_coherent/viewer/` holds the older DINO viewer's sources, which CoherentModes does not track.
- **The joint report.** CoherentModes' `explore/ncs_report/` reads these directories in place and was left there,
  as were the originals.
