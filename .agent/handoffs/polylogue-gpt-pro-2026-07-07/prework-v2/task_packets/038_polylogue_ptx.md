# 038. polylogue-ptx — Browser-capture posting channel: un-gate, with attachments

Priority/type/status: **P2 / feature / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Operator decision 2026-07-03: UN-GATE. Agents may drive web chats through the posting channel (agent-private Chrome profile posture per the ambient control model). Scope for this bead: bring the parked worktree branch to production quality, enable the channel, and add attachment support (files/images posted alongside text). Trajectory beyond this bead (separate beads): user-drivable posting from the webui, and harness remote-control lanes (Claude Code remote control, Codex analogue). Residual risk accepted: same-account ToS/rate exposure; mitigation is attributability — everything the channel posts is itself captured in the archive.

## Existing design note

The channel exists on the worktree branch (worktree-agent-aa5375b510cb4aa5d era work): extension->receiver POSTING path, previously operator-gated OFF. To ship: rebase/land the branch, flip the gate default for the agent-private profile, add attachment upload (multipart through the receiver -> provider web upload flows; store posted attachments as acquired blobs so the archive keeps what was sent), and cover with the deterministic capture smoke pattern. Verify the 0mu freshness fix (newest-wins) is in place first so posted-then-captured sessions do not get clobbered by DOM fallback.

## Acceptance criteria

1. The parked worktree posting branch (extension->receiver POSTING path, worktree-agent-aa5375b510cb4aa5d era) is landed at production quality. 2. The posting gate default is flipped ON for the agent-private Chrome profile. 3. Attachment upload works: files/images are posted via multipart through the receiver into provider web upload flows, and posted attachments are stored as acquired blobs so the archive retains exactly what was sent. 4. The 0mu freshness fix (newest-wins) is confirmed in place FIRST, so a posted-then-captured session is not clobbered by DOM fallback. 5. The channel is covered by the deterministic capture smoke pattern. Verify: deterministic capture smoke passes; a live agent-private post with an attachment lands both the message and the attachment blob in the archive (recorded); `devtools test` selection on the receiver/capture path.

## Static mechanism / likely defect

Issue description localizes the mechanism: Operator decision 2026-07-03: UN-GATE. Agents may drive web chats through the posting channel (agent-private Chrome profile posture per the ambient control model). Scope for this bead: bring the parked worktree branch to production quality, enable the channel, and add attachment support (files/images posted alongside text). Trajectory beyond this bead (separate beads): user-drivable posting from the webui, and harness remote-control lanes (Claude Code remote control, Codex analogue). Residual risk accepted: same-a… Design direction: The channel exists on the worktree branch (worktree-agent-aa5375b510cb4aa5d era work): extension->receiver POSTING path, previously operator-gated OFF. To ship: rebase/land the branch, flip the gate default for the agent-private profile, add attachment upload (multipart through the receiver -> provider web upload flows; store posted attachments as acquired blobs so the archive keeps what was sent), and cover with th…

## Source anchors to inspect first

- `polylogue/browser_capture/models.py:22` — BrowserCaptureAttachment has metadata and possible data fields; acquisition policy must be explicit.
- `polylogue/browser_capture/server.py:259` — Capture POST writes payloads to spool after admission checks.
- `polylogue/archive/attachment/models.py` — Normalized attachment model should carry acquired/missing/recoverable state.
- `polylogue/storage/blob_store.py` — Byte storage API and hash validation live here.
- `polylogue/daemon/http.py:983` — _check_auth_logic uses direct equality and allows all when token is unset.
- `polylogue/daemon/http.py:1037` — _check_auth currently accepts query-string access_token broadly.
- `polylogue/daemon/http.py:1294` — do_GET dispatches without central Host/Origin admission.
- `polylogue/daemon/http.py:1301` — _check_cross_origin applies only to POST and allows absent Origin.
- `polylogue/browser_capture/receiver.py:45` — BrowserCaptureReceiverConfig defaults auth_token to None.
- `polylogue/browser_capture/server.py:54` — _origin_allowed accepts absent Origin.
- `polylogue/browser_capture/server.py:68` — _check_token accepts every request when auth_token is None and uses direct equality otherwise.
- `polylogue/browser_capture/server.py:47` — Only per-request max body exists; add spool file/count/bytes governor.
- `polylogue/storage/blob_store.py:352` — detect_orphans only compares disk hashes to caller-supplied referenced IDs.
- `polylogue/storage/blob_store.py:387` — cleanup_orphans deletes caller-supplied hashes and lacks lease/ref/generation checks.
- `polylogue/storage/blob_gc.py:163` — _has_active_lease exists in the safer GC path.
- `polylogue/storage/blob_gc.py:307` — run_blob_gc is the safer planner/executor path to route destructive cleanup through.
- `polylogue/storage/blob_gc.py:393` — GC generation/age gate exists in run_blob_gc.

## Implementation plan

1. The channel exists on the worktree branch (worktree-agent-aa5375b510cb4aa5d era work): extension->receiver POSTING path, previously operator-gated OFF.
2. To ship: rebase/land the branch, flip the gate default for the agent-private profile, add attachment upload (multipart through the receiver -> provider web upload flows
3. store posted attachments as acquired blobs so the archive keeps what was sent), and cover with the deterministic capture smoke pattern.
4. Verify the 0mu freshness fix (newest-wins) is in place first so posted-then-captured sessions do not get clobbered by DOM fallback.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: The parked worktree posting branch (extension->receiver POSTING path, worktree-agent-aa5375b510cb4aa5d era) is landed at production quality.
- Acceptance proof: 2.
- Acceptance proof: The posting gate default is flipped ON for the agent-private Chrome profile.
- Acceptance proof: 3.
- Acceptance proof: Attachment upload works: files/images are posted via multipart through the receiver into provider web upload flows, and posted attachments are stored as acquired blobs so the archive retains exactly what was sent.
- Acceptance proof: 4.
- Acceptance proof: The 0mu freshness fix (newest-wins) is confirmed in place FIRST, so a posted-then-captured session is not clobbered by DOM fallback.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
