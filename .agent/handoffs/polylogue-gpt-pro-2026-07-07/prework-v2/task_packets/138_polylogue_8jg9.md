# 138. polylogue-8jg9 — Operational resilience: recoverable, restorable, survives daemon death and deploy

Priority/type/status: **P2 / epic / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **epic-needs-child-closure**.

## What the bead says

WHY: an archive whose pitch is durable evidence must itself survive incidents — daemon death mid-write, bad deploys, disk loss. Durable tiers (source.db/user.db) are irreplaceable; a restore path that has never been drilled is a hope, not a capability. ENABLES: trusting the archive as system-of-record; the backup-manifest gate that durable-tier migrations (60i5) already assume. MEMBER BEADS: polylogue-4be (backup-restore + quarterly restore drill), polylogue-peo (daemon-death recovery), polylogue-s8q (deploy trust; parked P4 while prod polylogued is inactive). Epic closes when a restore drill has actually run against a copy of the live archive and daemon-death recovery is regression-tested.

## Existing design note

Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home. This is the 'does the system survive an incident' capability, distinct from security (forgetting on purpose) and from 1xc (correct at scale).

## Acceptance criteria

A quarterly restore drill proves backups restore (4be); daemon crash mid-convergence recovers without stranding debt (peo, ties 1xc.3/1xc.4); deployed state is provable via deployment-smoke when prod is re-activated (s8q). Verify: devtools workspace deployment-smoke --json + a restore-drill artifact.

## Static mechanism / likely defect

Issue description localizes the mechanism: WHY: an archive whose pitch is durable evidence must itself survive incidents — daemon death mid-write, bad deploys, disk loss. Durable tiers (source.db/user.db) are irreplaceable; a restore path that has never been drilled is a hope, not a capability. ENABLES: trusting the archive as system-of-record; the backup-manifest gate that durable-tier migrations (60i5) already assume. MEMBER BEADS: polylogue-4be (backup-restore + quarterly restore drill), polylogue-peo (daemon-death recovery), polylogue-s8q (deploy trust… Design direction: Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home. This is the 'does the system survive an incident' capability, distinct from security (forgetting on purpose) and from 1xc (correct at scale).

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

1. Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home.
2. This is the 'does the system survive an incident' capability, distinct from security (forgetting on purpose) and from 1xc (correct at scale).

## Tests to add

- Acceptance proof: A quarterly restore drill proves backups restore (4be)
- Acceptance proof: daemon crash mid-convergence recovers without stranding debt (peo, ties 1xc.3/1xc.4)
- Acceptance proof: deployed state is provable via deployment-smoke when prod is re-activated (s8q).
- Acceptance proof: Verify: devtools workspace deployment-smoke --json + a restore-drill artifact.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
