Title: "Exact FTS identity ledger: rowid↔block_id↔source_hash↔recipe binding, drift magnitude gauges, and trigger-coherence property tests (1xc.12)"

Result ZIP: `misc-03-fts-identity-r01.zip`

## Mission

Implement bead `polylogue-1xc.12` (P1 — read its full record). The FTS5
index (`messages_fts`, contentless, synchronized by insert/delete/update
triggers over `blocks.search_text`) has a keystone identity assumption:
`messages_fts.rowid == blocks.rowid == docsize.id`. SQLite ROWID REUSE
breaks the safety of count-based checks: after delete+insert, a ghost FTS
row can bind to a DIFFERENT block while counts still agree. Search results
then cite the wrong content — a silent correctness hole in the archive's
most-used read path. Current freshness reporting is also too boolean:
operators need drift MAGNITUDE.

Build (bead design is authority):

1. **`messages_fts_identity` ledger** in the index tier: keyed by rowid
   with UNIQUE block_id, plus source_hash and recipe_id; maintained by the
   SAME trigger events as the contentless FTS table; rebuilt atomically by
   the repair/rebuild path. Consume the storage-neutral DerivationKey
   VALUE SHAPE from `polylogue-wmsc` (a parallel job drafts it — mirror
   its subject/source/recipe/output-contract field semantics, keep an
   FTS-owned ledger and lifecycle; state the interface assumption if the
   wmsc result is absent from your inputs).
2. **Exact freshness state**: O(1)-readable state rows covering missing,
   excess, identity mismatch, source mismatch, recipe mismatch, last check
   time, repair generation (bead AC #1-2). Exact checks JOIN on rowid AND
   confirm block_id — never count parity alone.
3. **Drift gauges**: magnitude metrics from the freshness state exposed
   through the existing status/observability path (find the ops/OTLP
   metric emission seam — `daemon/` observability wiring; Prometheus-style
   naming per the bead).
4. **Metamorphic trigger-coherence property tests**: Hypothesis state
   machine over arbitrary block mutations (insert/update/delete/full
   session replace/rowid-reuse-inducing delete+insert sequences) asserting
   FTS ∥ docsize ∥ identity-ledger coherence after every step — the
   rowid-reuse ghost case must be a named regression that FAILS on
   current master's count-only logic (prove the vulnerability first,
   fixture it, then show the ledger catches it).
5. **Schema window discipline**: this is an index-tier addition — edit
   canonical DDL + bump the index schema version (rebuild route, NO
   migration helper; `devtools lab policy schema-versioning` enforces
   this). The bead says implement "in the next batched index-schema
   window": check the snapshot for other pending index-tier bumps and note
   batching opportunities for the integrator rather than assuming you own
   the version bump timing.

## Constraints

- Triggers are hot-path: measure and state the write-amplification cost of
  ledger maintenance (per-block insert/delete adds one ledger row op) and
  keep it O(1) per block event.
- The FTS repair path (`fts` convergence stage + rebuild) must rebuild the
  ledger in the same transaction boundary as the FTS rebuild — find the
  current repair implementation (convergence stages + storage FTS module)
  and integrate, don't parallel-build.
- Do not change the tokenizer or FTS schema shape (`unicode61`, no porter
  — pinned decision).

## Deliverable emphasis

HANDOFF.md: DDL + version-bump diff, trigger design with cost analysis,
the freshness-state row schema, gauge names/semantics, the rowid-reuse
vulnerability demonstration (this is the story: show the ghost, then the
catch), property-test design, and batching notes for the index-window
integrator.
