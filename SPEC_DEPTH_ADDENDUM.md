# Addendum to SPEC_DEPTH.md v3.1 (2026-11-26)

`SPEC_DEPTH.md` is left byte-for-byte as the run saw it, because every unit's completion marker records its hash.
Two things changed after v3.1 was reviewed. Neither changes anything that is measured. Neither was re-reviewed.

## 1. Where the solves ran (replaces section 10)

The VPS was overloaded by other work: one unit that took 11 minutes on a quiet machine did not finish in two hours.
The owner approved Modal on 2026-11-26 ("modal is fine"). `exp1/modal_depth_run.py` ran the same `exp1/depth.py`
on the Volume's data: a one-unit pilot (`pythia/d32`), then the other 21 units in parallel, then `cross` and
`controls`; one preemptible container each, 4 CPU, 16 GB, 4 BLAS threads. Worst-case reservation at the repo's
recorded rates $4.63 under a $5.00 maximum. Ledger: `ledger/depth_run.json` on the Volume, mirrored at
`/data/tm_depth_run_ledger.json`. The local gate runs of section 10 were made before the move and are superseded
by the pilot's receipt. Outputs: Volume `depth/`, fetched to `/data/tm_depth_modal/`.

## 2. Site encoding (replaces the "Site data" bullet of section 8 and the int8 check in the audit)

16 modes per embedding, panel and path, loaded on demand, 8 bits per value, one scale per (layer, mode), with
square-root companding: `code = round(127 sign(x) sqrt(|x| / scale))`, clipped to [-127, 127], decoded as
`sign(code) (code / 127)^2 scale`. A set file's scale is the largest absolute finite value of that mode at that
layer over the set's own tokens; a panel or path file uses the larger of that and its own maximum, so placed
tokens never clip. The exact rule, including withheld values, is in the header of `exp1/pack_depth.py` and in the
pack's `meta.json`. Reason: with linear int8 and one scale per mode (v3.1), a mode's few far-out tokens left most
tokens within one or two steps of zero. The audit checks that re-encoding the float32 release with this rule
reproduces the published bytes exactly.
