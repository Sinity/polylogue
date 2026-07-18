# Source, Bead, and History Evidence

## Snapshot authority

The supplied project-state archive contained a Git bundle and captured repository tree. Reconstruction produced:

```text
commit: 536a53efac0cbe4a2473ad379e4db49ef3fce74d
branch recorded by snapshot: master
subject: fix(repair): harden raw authority convergence (#3046)
commit date: 2026-07-17 18:55:47 +0200
pre-change worktree: clean
```

The implementation was developed on local branch `feature/assertions/judgment-transaction-r01`; the patch remains based directly on the snapshot commit.

Repository instructions inspected included `CLAUDE.md`, `AGENTS.md`, and `TESTING.md`. Source traversal covered the storage writer and lifecycle, facade and shared payloads, root Click registration, query-first verbs, MCP declarations/handlers, status and daemon projections, generated product workflow registries, relevant tests, and Git history.

## Lifecycle authority

History identifies the already-merged lifecycle authority as:

```text
5aa34e6c5d231c952529174febe99b2a58f4da07
feat(assertions): add reviewed candidate judgment flow (#2791)
2026-07-13 02:53:55 +0200
```

Current source at the snapshot already contained candidate transition, judgment receipt, promotion, exact-retry/conflict, and bulk SAVEPOINT logic. The implementation therefore wraps and calls that machinery rather than reconstructing it.

## Bead evidence

### `polylogue-41ow`

The bead records a reproduced race in `upsert_assertion`: a non-atomic existing-status `SELECT` followed by a later upsert allowed an automated candidate refresh to overwrite an operator acceptance. The decided design requires one `BEGIN IMMEDIATE` preservation/write transaction, explicit conflicts for competing judgments, a second-process/connection proof, canonical busy-timeout profile, and rollback on failure.

The bead's completeness comment also notes inconsistent per-helper `PRAGMA foreign_keys` calls. The patch normalizes the actual shared assertion writer through `_configure_assertion_write_connection`, while preserving harmless existing inline calls.

### `polylogue-37t.12`

The bead explicitly states that PR #2791's lifecycle and bulk judgment storage are settled authority. Residual work is bounded evidence disclosure, queue health/retention reporting, one root `polylogue judge` public lifecycle, MCP policy preservation, and a real-route proof.

Its acceptance criteria require:

- candidate/injection lifecycle invariants;
- partial-success bulk semantics with per-item SAVEPOINTs;
- root judge as the sole CLI workflow;
- bounded age/source/claim/evidence disclosure with at most five previews;
- healthy-empty versus stalled/debt status and visible 60-day retention;
- review/admin-only judgment capability;
- real-route CLI/MCP/active-claim proof.

The patch addresses those residuals without changing the lifecycle schema or creating another queue.

### `polylogue-mrxt`

The bead requires a genuine operator-authored production action and rejects fixtures, direct SQL creation, or automatic promotion as a valid canary. The snapshot has no live archive or operator credentials, so the patch does not fabricate completion. It adds the missing stable retry key and supplies an exact production-route script whose mutation steps are CLI actions; SQL is used only afterward to verify durable receipts.

The bead comment also identifies `upsert_comparative_judgment_assertion` and `list_comparative_judgments` as built but uncalled. A current-source search confirms their only occurrences are their own definitions/docstrings in `user_write.py`. This patch does not wire that separate comparative mechanism into the canary because doing so would create a parallel path contrary to the canonical-lifecycle mission.

## Relevant source findings

### Race location

Before the patch, `upsert_assertion` made its terminal-status preservation decision before obtaining an immediate writer reservation. The write happened through `INSERT ... ON CONFLICT`, so a second connection could commit a judgment in the gap. The repaired source obtains the writer slot first and keeps the read/policy/write/readback within that scope.

### Nested SQLite behavior

A SQLite SAVEPOINT inside a deferred transaction does not reserve the global writer slot by itself. The patch's zero-row `UPDATE` is therefore intentional: it upgrades the caller's transaction before any status-preservation read while leaving commit/rollback ownership with the caller.

### Identity inconsistency

`upsert_recall_pack` formerly generated default identity from name plus payload, making a changed named pack append rather than update. `upsert_annotation` needed the opposite distinction: exact retries should converge, but changed free-form note text should append. The chosen rules are stable name for recall packs and target/body content identity for annotations without explicit ids.

### Evidence projection

The shared review payload was the correct extension point because CLI and MCP already consume it. Resolving evidence in the facade keeps storage envelopes free of presentation concerns and allows each failed ref to degrade independently.

### Queue telemetry

Candidate rows live in `user.db`, while producer/scheduler/debt evidence lives in operations storage. The queue product therefore reads both tiers without changing either. Empty state is not inferred as healthy from absence alone.

### Cross-tier debt lock

Broader facade execution exposed `database source_debt is locked` during `DETACH` on a long-lived connection. The related read now uses a dedicated canonical read-only connection whose close releases attached state and outstanding statements together. This is included because queue/status composition exercises the same archive-debt surface.

## Contradictions and resolutions

1. Older generated documentation and current snapshot code exposed `mark candidates`; later bead notes and the merged root-command decision require root `judge`. The later authority wins. Behavior was ported first, then the duplicate was removed and generators/tests were updated.
2. The mission points to dogfood investigation files under `.agent/scratch/dogfood-2/investigations/` if present. That directory contained no files in the reconstructed snapshot. The bead's reproduced interleaving and current source were sufficient to build a real two-connection regression.
3. The live canary bead requires a genuine operator event, but the execution environment had no live archive. The correct resolution is an executable operator script plus production-route test coverage, not a fabricated row.
4. The result package does not claim a complete repository-wide pass. Focused suites completed; two broad aggregates hit the command ceiling and the terminal PTY suite has an external Tokio stderr fault.

## Package evidence

`PATCH.diff` was generated from the working tree against the exact snapshot base and passed `git apply --check` in a clean detached worktree. At generation time:

```text
changed files: 36
insertions: 1744
deletions: 601
patch lines: 3915
patch bytes: 189695
patch SHA-256: a72c7e066521721cec2de5784d3e8071f2604496c5bb27ea2bb90b7699e7ec76
```

No supplied project-state archive is included in the package.
