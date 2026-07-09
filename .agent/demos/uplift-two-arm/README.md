# Handoff-Pack Uplift Experiment

This shelf is for current two-arm experiments that test whether bounded
Polylogue handoff packets improve continuation reconstruction compared with a
raw archive reference.

The shelf is current, not append-only. Regenerate or replace `current/` when a
better protocol or cleaner run exists.

## Current run (2026-07-09, n=5 pilot)

Successor to the n=1 `jxe`/`jxe.2` pilot (raw-ref 8/10 vs handoff-pack 5/10,
diagnostic-negative, attributed to packet staleness). See `current/report.md`
for the full write-up: n=5 pairs, handoff-pack wins 4/5, directional but
explicitly **not** a publishable result (protocol calls for n>=12-20). One
pair's blind was compromised and is documented rather than discarded; one
pair is a genuine counterexample where the pack arm asserted a false fact
with high confidence. Cold-reader gate: PASS.
