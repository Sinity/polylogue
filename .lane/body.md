## Summary

Record each blob-GC member outcome durably immediately after its locked recheck and filesystem action. Update the durable-intent regression to verify that a mid-batch crash preserves completed and pending member states separately.

## Problem

Blob GC previously held all member outcomes in memory until the unlink loop ended. A crash after an unlink but before the batch bookkeeping commit left physical loss represented only as a pending intent, while the generation receipt could later summarize work without per-member completion evidence.

## Solution

Commit each terminal member outcome before continuing to the next member, reacquiring the control-tier writer lock for each iteration. Preserve the connection-local hook matcher generation across these bookkeeping commits so bounded batches do not rebuild it per member.

## Verification

- `./.venv/bin/python -m devtools test tests/unit/storage/test_blob_gc_durable_intent.py` — 19 passed; two inherited failures in migration parity and authorization coverage.
- `./.venv/bin/python -m devtools test tests/unit/storage/test_blob_gc.py tests/unit/storage/test_blob_gc_durable_intent.py -k 'not authorized and not v33'` — 37 blob-GC tests passed; two deselected.
- `nix develop --command devtools verify --quick` — formatting, lint, and mypy passed; `render all` failed on the inherited duplicate documentation entry `docs/artifact-publication.md`.

## Residual risk

The quick gate remains red on the inherited docs-surface duplicate. Existing unrelated failures remain in the focused durable-intent file.
*** Add File: /realm/worktrees/packet-polylogue-vhsno/.lane/close-reason.md
 +Blob GC now commits each terminal member outcome immediately after its protected filesystem action, and the regression proves a mid-batch crash cannot leave a completed unlink unaccounted for.
