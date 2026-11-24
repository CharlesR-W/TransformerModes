# TransformerModes: instructions for agents

Start with `HANDOFF.md` (what exists, what was found, what is half-done, what the owner has and has not approved),
then `README.md` (status and the Stage A run log). The owner's global instructions in `~/.claude/CLAUDE.md` apply and
take precedence over this file.

- **Spend:** nothing runs on Modal without the owner's go-ahead for that stage and its maximum. Stage B is approved
  only for its gate (maximum $1.15). The Modal workspace has a billing-cycle limit that stops all containers.
- **Specs are authoritative** (`SPEC.md` for Stage A as run, `SPEC_B.md` for Stage B). A substantive change is
  critiqued by a different model family before it is acted on; reviews go in `reviews/`.
- **Inherited commitments** (from CoherentModes): every object in a construction ends in an eigendecomposition or SVD
  of a sample-estimable operator; nothing is trained inside the construction (fits are readouts and are called
  fits); the only geometry is a dot product on each view; no labels or probes define a kernel or basis; no sparsity
  assumption.
- **The owner's standing preferences here:** no exclusion windows or mean-removal fixes (measure and show instead);
  few views; plain words; every index explained; numbers on pages come from build-time facts, never typed by hand;
  results are published only with a passing independent audit whose file inventory matches what is published.
- **Report what happened.** A background job's "completed, exit 0" is not a result: read its own output. Say what
  was not run or not verified.
- **This machine** is small and shared: pass `--basetemp` to pytest and delete it; fetch Volume data with
  `modal volume get`, into a fresh directory.
