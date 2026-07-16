# 035. polylogue-83u.3 — Preserve uploaded attachment bytes in live browser capture

Priority/type/status: **P1 / feature / open**. Lane: **01-blob-attachment-integrity**. Release: **B-storage-byte-integrity**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

chatgpt-dom-v1 records the attachment chip (name + DOM text), not the uploaded bytes (byte_count=0) — the bytes live on provider servers. Needs a capture-side acquisition path. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Capture-side: the DOM adapter cannot see upload bytes (they live on provider servers). Options to evaluate in the extension/receiver: (a) intercept the upload request body at capture time (webRequest/fetch hook) and spool alongside the DOM capture; (b) re-fetch provider attachment URLs while the authenticated session is live, before spooling. Either way bytes join the capture payload as inline attachment content and flow through the same ParsedAttachment.inline_bytes -> blob path as the embedded-payload bead. VERIFY extension architecture constraints (MV3 service worker, receiver contract) before choosing; keep the receiver contract versioned.

## Acceptance criteria

- Extension/receiver architecture constraints (MV3 service worker lifecycle, receiver contract) are documented in the PR before choosing between (a) intercepting the upload request body at capture time (webRequest/fetch hook) and (b) re-fetching provider attachment URLs while the authenticated session is live.
- The chosen path spools uploaded attachment bytes into the capture payload as inline attachment content, flowing through the same ParsedAttachment.inline_bytes -> blob store path as the embedded-payload bead; a captured attachment now has a real blob_hash and nonzero byte_count (was byte_count=0 for chatgpt-dom-v1).
- The receiver contract is versioned for the new payload field.
- A deterministic capture smoke covers a capture-with-attachment producing a stored blob (`devtools test <capture smoke>` or a capture-regression artifact).

## Static mechanism / likely defect

Browser capture payload models can carry attachment bytes, but live capture currently treats many attachments as metadata or extracted text only. Future evidence needs bytes or honest unavailable state.

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

1. Define acquisition policy for live browser attachments: inline bytes, fetch URL, defer with tokenized acquisition job, or mark unavailable.
2. Persist acquired bytes via blob store with content hash and attachment row link.
3. Do not store raw bytes for private/unsupported items without explicit policy; store classified missing/unavailable state.
4. Expose attachment acquisition state in capture status and archive read payloads.

## Tests to add

- Live capture fixture with inline_base64 persists blob and resolves hash.
- Fixture with only provider URL marks deferred/unfetched, not missing-lost.
- Quota/security tests prove attachment bytes obey receiver token and spool governor.

## Verification commands

- ``devtools test tests/unit/sources/test_browser_capture*.py tests/unit/browser_capture -k 'attachment or inline_base64 or blob'` plus browser-extension smoke for capture-with-attachment.`

## Pitfalls

- Do not delete or compress before byte references are classified and lease/ref safety is proven.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
