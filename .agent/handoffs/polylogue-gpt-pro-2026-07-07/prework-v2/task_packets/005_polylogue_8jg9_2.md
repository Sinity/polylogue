# 005. polylogue-8jg9.2 — Blob-GC lease/orphan concurrency test (the acquire->commit race)

Priority/type/status: **P2 / task / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test. #818 has real orphan-detection bugs. Add a test that runs run_blob_gc concurrently with a mid-flight write holding a lease and asserts the leased blob is never reclaimed.

## Acceptance criteria

A test acquires a lease, starts GC, and asserts the leased blob survives; a released-lease orphan is reclaimed; sweep_orphaned_blob_leases clears a SIGKILLed writer's lease past ORPHAN_LEASE_MAX_AGE_S. Verify: the new pytest under tests/unit/storage.

## Static mechanism / likely defect

Design direction: internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test. #818 has real orphan-detection bugs. Add a test that runs run_blob_gc concurrently with a mid-flight write holding a lease and asserts the leased blob is never reclaimed.

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

1. internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test.
2. #818 has real orphan-detection bugs.
3. Add a test that runs run_blob_gc concurrently with a mid-flight write holding a lease and asserts the leased blob is never reclaimed.

## Tests to add

- Acceptance proof: A test acquires a lease, starts GC, and asserts the leased blob survives
- Acceptance proof: a released-lease orphan is reclaimed
- Acceptance proof: sweep_orphaned_blob_leases clears a SIGKILLed writer's lease past ORPHAN_LEASE_MAX_AGE_S.
- Acceptance proof: Verify: the new pytest under tests/unit/storage.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
