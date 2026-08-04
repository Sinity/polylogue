# Blob-reference liveness closure audit

Date: 2026-08-04

Scope: `polylogue-0v4tn` and merged PR #3705.

## Decision

PR #3705 delivered the guarded source-tier reconciliation actuator, but it did not itself apply that actuator to the live archive. The live archive is still red for I3, so this change makes I3 a direct rebuild preflight gate. It does not claim that the live bookkeeping repair is complete.

## Acceptance matrix

| Requirement | Evidence | Status |
| --- | --- | --- |
| Set-based `raw_payload` referent check | `BLOB_REF_LIVENESS_JOIN` maps `raw_payload` to `raw_sessions.raw_id`; the classifier uses a `LEFT JOIN` per mapped ref type. | Satisfied |
| Set-based attachment referent check | `attachment` maps to the parent `raw_sessions.raw_id`, not retired index-tier attachment identity. The archive verifier now imports the same canonical mapping. | Satisfied |
| Read-only default | `blob-reference-liveness` calls the reconciler without `--apply`, which opens `source.db` with `mode=ro`. | Satisfied |
| Offline, fresh verified backup, and transaction safety for apply | Apply rejects a running daemon, checkpoints and validates the source backup before and inside `BEGIN IMMEDIATE`, fsyncs a prepared receipt, checks the exact delete count, runs `quick_check`, and commits or rolls back as one transaction. | Satisfied |
| Receipt and blob safety | The receipt records the exact candidate digest before mutation and terminal state after it. The only actuator mutation is `DELETE FROM blob_refs`; it contains no blob-store unlink or file deletion. | Satisfied |
| Direct reindex gate | `rebuild_index_from_source` now runs `blob-refs-liveness` against the actual archive root before it acquires a rebuild lease or creates generation state. | Satisfied |
| Predicate anti-vacuity | The classifier fixture has a live and orphan ref for every mapped type and asserts the exact orphan matrix. Inverting the `LEFT JOIN ... IS NULL` predicate reports the four live rows instead and fails the assertion. | Satisfied |

## Live I3 evidence

Read-only command, run against `/realm/db/polylogue/source.db`:

```sql
SELECT 'raw_payload' AS ref_type, COUNT(*) AS orphaned
FROM blob_refs AS b
LEFT JOIN raw_sessions AS r ON r.raw_id = b.ref_id
WHERE b.ref_type = 'raw_payload' AND r.raw_id IS NULL
UNION ALL
SELECT 'attachment' AS ref_type, COUNT(*) AS orphaned
FROM blob_refs AS b
LEFT JOIN raw_sessions AS r ON r.raw_id = b.ref_id
WHERE b.ref_type = 'attachment' AND r.raw_id IS NULL
ORDER BY ref_type;
```

Result:

```text
attachment|0
raw_payload|73427
```

No live apply, source backup creation, blob deletion, namespace quarantine, GC run, or raw-authority operation occurred. A future offline operator pass must use `polylogue ops maintenance blob-reference-liveness --apply` with a fresh verified source backup and a new receipt path. The preflight gate will continue to reject direct reindexing until I3 reports zero unwaived orphan refs.

## Adversarial review

One independent read-only review found that the original apply test mocked the offline guard. This change adds a regression test that simulates a live daemon and proves refusal occurs before receipt creation or `blob_refs` mutation. Source tracing then found and repaired the verifier's stale attachment referent mapping and the missing direct-reindex preflight.
