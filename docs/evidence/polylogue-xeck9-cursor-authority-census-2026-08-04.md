# Cursor authority census, 2026-08-04

## Scope and safety boundary

This report records the read-only production census for `polylogue-xeck9`.
It intentionally contains no source paths, raw IDs, session IDs, titles, hashes, or payload excerpts. No production database, cursor, raw row, accepted head, blob, or daemon state was changed while collecting it.

The investigation covered only the raw-frontier readiness projection:

```text
polylogue.storage.raw_retention.raw_frontier_integrity_projection()
  -> raw_frontier_integrity_snapshot(source.db read-only)
     -> _active_index_raw_authority(index.db read-only)
     -> _ops_cursor_byte_offsets(ops.db read-only)
     -> _check_cursor_ahead_of_accepted(...)
```

Accepted heads remain the comparison authority. The projection joins each accepted head's `accepted_raw_id` to `source.raw_sessions.source_path`, then compares an active `ops.ingest_cursor.byte_offset` with every accepted byte frontier on that path. Semantic heads are intentionally not byte-comparable.

## Census

The direct read-only projection reported:

| Fact | Count |
| --- | ---: |
| Active non-excluded cursors with byte offsets | 20,041 |
| Byte-comparable cursor paths | 16,879 |
| Cursor/head byte comparisons | 16,881 |
| True cursor-ahead rows | 1 |
| True ahead comparisons | 1 |
| Incomparable cursor/head authority rows | 727 |
| Semantic-only accepted-head paths, deliberately outside byte comparison | 2,435 |

The one proven violation belongs to an `unknown-export` cursor. Its committed cursor offset is 11,166,810 bytes and its sole accepted byte head is a byte-proven full raw at frontier 11,166,556, a difference of 254 bytes. The head's source and index generations agree at zero. This is a real cursor-ahead condition, not an incomparable state.

The 727 incomparables classify as follows:

| Typed evidence state | Count | Meaning |
| --- | ---: | --- |
| `source_raws_without_accepted_head` | 725 | Source-tier raw evidence exists for the cursor path, but no accepted index head is present. |
| `cursor_path_absent_from_source` | 2 | The active cursor path has no current source-tier raw row. |

The existing status DTO renders these separately from the proven violation: `cursor_ahead_count=1`, `cursor_authority_gap_count=727`, `cursor_ahead_status="violated"`, and `overall_status="violated"`. The reason string reports both populations. Since a violation dominates unknown evidence, incomparability cannot make this result green.

## Root-cause conclusion

No demonstrated defect was found in the cursor write path or comparison predicate on the current branch. The comparison is strict `cursor_offset > accepted_frontier`, consumes the durable accepted head, and its regression coverage already exercises the real `raw_frontier_integrity_snapshot()` route. The production defect was at the readiness consumers: raw convergence and reindex did not consume this proof before selecting source rows. Reclassifying the true violation as incomparable or comparing against a non-accepted raw would contradict the current projection and its existing tests.

The live cursor is ahead of its accepted full-head frontier. That is production reconciliation work, not evidence for a code change. The 725 source-backed incomparables are deferred authority states that must remain visible. The 2 source-absent cursor paths likewise must remain explicit until their durable history is reconciled; this report does not authorize a cursor reset or source-row edit.

## Safe reconciliation contract, not performed

There is no cursor-specific dry-run/apply actuator that can safely repair this condition without re-running the real ingest path. Do not use raw-authority frontier inspection as a cursor repair shortcut: it records census observations, while daemon convergence applies only executable proof-backed plans under its writer coordinator, and neither path reconciles the cursor condition. The safe sequence for an operator is:

1. Stop or confirm quiescence of the daemon, then capture a backup plan and an initial read-only full status receipt:

   ```bash
   polylogue ops maintenance backup-plan --output-format json > /realm/tmp/work/polylogue-xeck9-backup-plan.json
   polylogue ops status --json --full > /realm/tmp/work/polylogue-xeck9-before.json
   ```

2. Run the targeted reconciliation through the normal ingest/materialization route for the affected source, with its normal backup gate. The operator must retain the daemon/ingest receipt and the exact source selection externally; this report intentionally does not publish the private path.

3. Capture the post-run status using the same read-only command and compare these fields: `cursor_ahead_count`, `cursor_authority_gap_count`, `cursor_ahead_samples`, `cursor_authority_gap_samples`, and `overall_status`.

4. Accept the run only if the receipt proves the accepted head advanced to cover the cursor, or the cursor did not advance and an explicit retry/deferred state remains. The postcondition must preserve accepted-head authority, leave source rows untouched outside normal ingest, report every remaining incomparable row, and show `cursor_ahead_count=0`. A nonzero incomparable count remains justified only when its typed evidence state is retained in the receipt.

The two read-only commands are intentionally published but were not executed in this lane because the task forbids production mutation and the backup planner exposes archive metadata. The normal ingest/reconciliation action was also not performed.

## Reproduction commands used

```bash
env PYTHONPATH="$PWD" .venv/bin/python - <<'PY'
from pathlib import Path
from polylogue.storage.raw_retention import raw_frontier_integrity_projection

projection = raw_frontier_integrity_projection(
    Path("/home/sinity/.local/share/polylogue"),
    {"available": True, "lost_source_evidence_count": 0, "lost_source_evidence_samples": []},
)
print(projection.to_dict())
PY
```

The report records only the privacy-safe aggregate and fixed numeric evidence from that output.
