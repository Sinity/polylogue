Title: "Order-independent lineage writes and branch-point safety: close the three production-shaped state-machine failures (866e)"

Result ZIP: `lin-01-lineage-law-r01.zip`

## Mission

Implement bead `polylogue-866e` (P0, in_progress — read its FULL record and
notes; PR #2922 already landed canonical sibling ordering and
missing-branch-cut safety, so verify current master state first and build
the REMAINING law). Stateful property testing found three production-shaped
failures on clean master: (1) repeated full replacement with sibling
variants plus child-before-parent ingestion can retain OLDER sibling text
as primary; (2) deleting a referenced parent branch point can leave a child
whose modeled prefix exceeds the surviving parent and CRASH composition;
(3) the general invariant — equivalent lineage histories must converge to
identical composed reads under ANY arrival/replacement order — does not
hold universally.

The write-path authority map (bead design, verify line numbers against the
snapshot): `polylogue/storage/sqlite/archive_tiers/write.py` — full-replace
normalization ~235–455, batch link normalization ~1858–1889,
composed-signature/prefix alignment ~3167–3454, stale branch-point repair
~3478–3594. Link resolution:
`storage/sqlite/queries/session_links.py` ~175–341. Independent read
oracle: `storage/sqlite/queries/message_query_reads.py` ~74+.

Deliver:

1. **Reproduce first**: run the saved Hypothesis failure classes
   (`POLYLOGUE_HYPOTHESIS_REUSE_FAILURES=1` against
   `tests/property/test_write_path_state_machine.py`) on the snapshot
   baseline; commit each class as a named deterministic transition fixture
   with the three oracles the AC demands (physical-row, link-row,
   composed-read).
2. **The law**: make lineage write transitions order-independent —
   equivalent histories (same final logical content, any arrival order:
   child-first, parent-replaced-later, sibling-variant races) converge to
   identical composed reads; transitions atomic and rollback-safe; missing/
   dangling relations remain TYPED, readable states (LineageCompleteness
   vocabulary — the Diet architecture decision
   `02-lineage-composition-and-snapshots.md` in the snapshot's testdiet
   context is the ratified contract: canonical divergent tails + typed
   edges, one deferred read snapshot, incomplete lineage explicit, never
   fabricated content).
3. **Branch-point safety**: parent deletion/replacement paths guarantee a
   child's modeled prefix never exceeds surviving parent content — degrade
   to explicit truncated-lineage state instead of crashing composition.
   Remember the load-bearing constraint: `branch_point_message_id` is
   deliberately NOT a FK (full-replace deletes+reinserts parent messages;
   SET NULL would destroy the splice) — preserve that.
4. Tests: the state machine extended with the adversarial transitions
   (sibling variant replacement, parent-delete-then-reingest, quarantined
   cycles), each named class asserting through all three oracles; plus the
   composed-read equivalence property over permuted histories.

## Constraints

- Content-hash idempotency must survive: equal material stays idempotent;
  your normalization cannot change import identity semantics
  (`pipeline/ids.py`, `core/hashing.py`).
- index.db is rebuildable: if a fix requires reinterpreting stored rows,
  the canonical DDL + rebuild is the route, never in-place migration.
- This is the hottest storage hotspot: keep the diff surgical; every
  changed write-path function needs its named mutation-killing test.

## Deliverable emphasis

HANDOFF.md: per-failure-class root cause + fix mechanism, the equivalence
law statement as implemented, oracle design, what PR #2922 had already
covered vs what you added, composition-degradation semantics table, and
residual risks for the integrator's live-archive validation.
