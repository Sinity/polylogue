# 004. polylogue-8jg9.4 — ops doctor cleanup_orphans can delete an in-flight leased blob (the real #818)

Priority/type/status: **P1 / bug / open**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

run_blob_gc is lease/ref/generation-safe, but the ops-doctor path (BlobStore.detect_orphans/cleanup_orphans) compares disk against caller-supplied ids only — VERIFIED LIVE 2026-07-06: blob_store.py contains zero references to pending_blob_refs/blob_refs/gc_generations. If the doctor caller passes only raw_sessions raw-ids (per the R&D audit), a blob acquired-but-not-yet-committed is classified orphan and deleted — the exact race the lease design exists to close. Fix: make cleanup_orphans consult leases + blob_refs + the generation age gate, or hard-gate the doctor path behind run_blob_gc. First step: verify what the live doctor caller passes as db_referenced_ids. NOT optional; independent of the 8jg9.2 concurrency test which should then cover this path too.

## Acceptance criteria

A leased-uncommitted blob survives ops doctor cleanup in a fixture race; doctor path either delegates to run_blob_gc or applies all three invariants; 8jg9.2 test extended to the doctor path. Verify: fixture race test.

## Static mechanism / likely defect

The safe `run_blob_gc` path already consults leases/refs/generations, but `BlobStore.cleanup_orphans` deletes hashes supplied by a simpler disk-vs-ID detector without consulting active leases or blob refs.

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

1. Convert destructive doctor/orphan cleanup to call the lease-aware GC planner, or make `cleanup_orphans` preview-only unless passed a verified GC plan token/object.
2. If keeping `cleanup_orphans`, add lease/ref/generation-age checks inside it so no caller can bypass safety.
3. Rename direct disk orphan detection to `preview_orphan_candidates` or similar to remove false safety aura.
4. Update ops-doctor output to show protected-by-lease/protected-by-ref/protected-by-generation counts.

## Tests to add

- Fixture: write a staged blob, acquire operation lease, run ops-doctor cleanup; file survives.
- Fixture: unreferenced old blob with no lease is deleted only when dry_run false and GC generation gate passes.
- Race fixture: cleanup plan computed before lease; lease acquired before delete; delete is skipped.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
