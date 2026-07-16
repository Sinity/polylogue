# 033. polylogue-4be — Restore drill: prove the backups restore, quarterly

Priority/type/status: **P2 / task / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Three backup layers exist (btrbk, polylogue-sqlite-backup, source-tier doctrine); none has ever been restore-tested. An untested backup is a hypothesis, and this one carries the entire project's irreplaceable asset.

## Existing design note

A devtools lane (or ops command): restore the latest backup set to a scratch root, run integrity_check per tier, run a 10-query battery (counts, one find, one read, one insight read), compare counts against the live archive within expected-lag tolerance, record timing + result as an ops artifact. Quarterly cadence via the operator's existing timer infrastructure (sinnix-side systemd timer calling the lane; alert on failure through the daemon health surface). First run is the bead; the timer makes it standing.

## Acceptance criteria

One full restore executed from real backups with the battery green and timing recorded; the lane is invocable as one command; the quarterly timer is wired sinnix-side; a deliberately corrupted scratch restore fails loudly.

## Static mechanism / likely defect

The bead states there are multiple backup layers but no restore test. The restore drill should prove latest backup set can produce an archive that passes integrity checks and basic user queries within expected lag.

## Source anchors to inspect first

- `polylogue/storage/blob_store.py:352` — detect_orphans only compares disk hashes to caller-supplied referenced IDs.
- `polylogue/storage/blob_store.py:387` — cleanup_orphans deletes caller-supplied hashes and lacks lease/ref/generation checks.
- `polylogue/storage/blob_gc.py:163` — _has_active_lease exists in the safer GC path.
- `polylogue/storage/blob_gc.py:307` — run_blob_gc is the safer planner/executor path to route destructive cleanup through.
- `polylogue/storage/blob_gc.py:393` — GC generation/age gate exists in run_blob_gc.
- `polylogue/browser_capture/models.py:22` — BrowserCaptureAttachment has metadata and possible data fields; acquisition policy must be explicit.
- `polylogue/browser_capture/server.py:259` — Capture POST writes payloads to spool after admission checks.
- `polylogue/archive/attachment/models.py` — Normalized attachment model should carry acquired/missing/recoverable state.
- `polylogue/storage/blob_store.py` — Byte storage API and hash validation live here.

## Implementation plan

1. Implementation shape:
2. 1. Add `polylogue ops restore-drill` or `devtools restore-drill`.
3. 2. Locate latest configured backup set; restore to a scratch root, never over live archive.
4. 3. Run `PRAGMA integrity_check` for each SQLite tier.
5. 4. Run a 10-query battery: session count, message count, blob-reference debt, one `find`, one `read`, one insight/profile read, one attachment lookup, one usage/cost summary, one tag/user-state read, one health/status command.
6. 5. Compare counts to live archive with an expected-lag tolerance and record differences.

## Tests to add

- synthetic backup restored to scratch and query battery passes.
- corrupted DB/file fails with clear failure.
- drill refuses to write into live archive path.
- count-lag tolerance behaves as declared.

## Verification commands

- ``devtools test tests/unit/operations/test_restore_drill*.py -k 'restore or backup or corruption'` plus one real restore drill artifact.`

## Pitfalls

- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
