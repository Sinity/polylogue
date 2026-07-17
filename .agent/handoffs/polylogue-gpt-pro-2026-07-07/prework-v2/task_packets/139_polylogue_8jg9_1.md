# 139. polylogue-8jg9.1 — Standing backlog-hygiene invariant lint (bd devloop gate)

Priority/type/status: **P2 / task / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Backlog structure trails filing unless an invariant lint enforces it (the 2026-07-03 session needed a 41-agent sweep to recover). The backlog equivalent of automagic-invariants: violations fail a gate instead of accumulating until an archaeology session.

## Existing design note

This session needed a 41-agent sweep because structure trails filing. Make it self-maintaining: a bd-level lint in the devloop that fails on (a) any P0/P1 bead with null acceptance (respecting notes sidecars), (b) any decision-type bead open past a 'Status: adopted/decided' line, (c) any active bead with no area:* label, (d) any orphan (no epic parent) older than N days, (e) any blocks-edge pointing at a closed bead (false-block). The backlog equivalent of automagic-invariants.

## Acceptance criteria

The lint runs in the devloop and fails on a seeded violation of each of the 5 classes; a clean backlog passes; wired into devtools verify or a bd hook. Verify: seed one violation per class, assert non-zero exit.

## Static mechanism / likely defect

Issue description localizes the mechanism: Backlog structure trails filing unless an invariant lint enforces it (the 2026-07-03 session needed a 41-agent sweep to recover). The backlog equivalent of automagic-invariants: violations fail a gate instead of accumulating until an archaeology session. Design direction: This session needed a 41-agent sweep because structure trails filing. Make it self-maintaining: a bd-level lint in the devloop that fails on (a) any P0/P1 bead with null acceptance (respecting notes sidecars), (b) any decision-type bead open past a 'Status: adopted/decided' line, (c) any active bead with no area:* label, (d) any orphan (no epic parent) older than N days, (e) any blocks-edge pointing at a closed bead…

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

1. This session needed a 41-agent sweep because structure trails filing.
2. Make it self-maintaining: a bd-level lint in the devloop that fails on (a) any P0/P1 bead with null acceptance (respecting notes sidecars), (b) any decision-type bead open past a 'Status: adopted/decided' line, (c) any active bead with no area:* label, (d) any orphan (no epic parent) older than N days, (e) any blocks-edge pointing at a closed bead (false-block).
3. The backlog equivalent of automagic-invariants.

## Tests to add

- Acceptance proof: The lint runs in the devloop and fails on a seeded violation of each of the 5 classes
- Acceptance proof: a clean backlog passes
- Acceptance proof: wired into devtools verify or a bd hook.
- Acceptance proof: Verify: seed one violation per class, assert non-zero exit.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
