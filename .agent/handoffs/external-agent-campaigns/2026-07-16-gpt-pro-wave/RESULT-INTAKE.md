# GPT Pro result intake — 2026-07-17

This campaign now retains the valid downloaded result packages under each
workload's `results/<job>/r01/raw` directory.  The raw member is the immutable
received artifact; it is deliberately kept as received rather than rewritten
into source files. Each workload index records the SHA-256, byte size,
snapshot, package state, and the result of local validation.

## What this does and does not mean

- It makes the received work durable and reviewable in the campaign.
- It does **not** make the packages queryable Polylogue archive sessions:
  current import detection classifies these ZIP/Markdown/PATCH artifacts as
  `unknown-export` and produces no session records.  `polylogue-hs3y` owns the
  product capability to preserve external result packages as queryable work
  evidence.
- A package is not accepted as product code just because it has a clean patch
  application against its stated snapshot.  It must be reconciled with current
  master, reviewed, and locally verified.
- Test Diet packages are retained as drafts only.  Their dispatch gate reports
  that `foundation-receipt.json` is missing, so they must not authorize the
  16-worker fan-out or deletion decisions.

## Exceptions

`beads-01` / `polylogue-config-closure.zip` was rejected: it contains a
zero-byte patch and a 162 MB copied repository snapshot.  The original download
is retained outside this repository at the operator's download location and is
identified in `beads/results/index.json`; it is intentionally not committed as
campaign evidence.

`beads-02` arrived as a standalone `PATCH.diff`, not the required result ZIP.
The raw patch is retained with an explicit incomplete-package status.  Its
bounded declaration foundation was locally integrated through PR #3004; the
later overlapping `beads-03` and `beads-04` packages remain inputs for their
own reconciliation work.

Browser-visible result summaries that failed to produce a valid download are
correlated in Bead notes but are not represented here as accepted artifacts.
