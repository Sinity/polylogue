# 18. polylogue-83u.3 — Acquire uploaded attachment bytes in live browser capture

Priority: **P1**  
Lane: **attachment-integrity**  
Readiness: **needs-architecture-note then code**

Depends on packet(s): polylogue-kwsb.1, polylogue-83u.4

## Why this is urgent / critical-path

The DOM adapter currently captures attachment chips but not bytes. For live browser capture, this is the last moment when bytes or authenticated provider handles may still be available.

## Static diagnosis / likely mechanism

Static mechanism:
- Browser capture schema already has byte-carrying fields: `inline_base64`, `content_base64`, `data` (`polylogue/browser_capture/models.py:22-35`).
- Parser already converts those fields to `ParsedAttachment.inline_bytes` (`polylogue/sources/parsers/browser_capture.py:83-132`).
- ChatGPT DOM adapter currently records only name/url/provider_meta (`browser-extension/src/content/chatgpt.js:98-108`), no inline bytes.
So the storage path exists; the missing piece is extension-side acquisition.

## Implementation plan

Implementation shape:
1. First document the MV3 architecture choice in the PR: service-worker lifecycle, content-script/main-world access, permissions, receiver contract, payload size limits.
2. Prefer an extension-side acquisition path. Receiver-side refetch lacks page cookies/session unless the extension supplies bytes or an authenticated fetch result.
3. For uploaded local files, prototype a content/main-world hook on file input/change and/or fetch/FormData submission that captures File bytes into an in-memory bounded cache keyed by name/size/lastModified/provider id.
4. When a turn attachment chip is captured, attach `inline_base64`, `size_bytes`, `mime_type`, and `sha256`/provider_meta acquisition details if a cached file matches.
5. Bump receiver schema version and validate max payload/attachment sizes.
6. Reuse the existing parser `inline_bytes -> blob` path; do not widen content hashing or invent synthetic hashes.
7. If upload-body interception is impossible for a provider, classify as `unfetched` with reason and file a provider-specific follow-up.

## Test plan

Tests:
- JS/unit or browser smoke: synthetic File upload leads to capture payload with inline_base64 and size.
- receiver/model accepts versioned payload and rejects oversized inline content.
- parser turns payload into `ParsedAttachment.inline_bytes` and stored blob has true SHA-256 and nonzero byte_count.
- legacy payload without bytes still parses as unfetched/metadata-only.

## Verification command / proof

`devtools test tests/unit/sources/test_browser_capture*.py tests/unit/browser_capture -k 'attachment or inline_base64 or blob'` plus browser-extension smoke for capture-with-attachment.

## Pitfalls

Do not choose receiver-side refetch without proving cookies/auth can flow safely. Do not put raw bytes into durable transcript JSON beyond the existing transport-only inline path.

## Files/functions to inspect or touch

- `browser-extension/src/content/chatgpt.js:98-108`
- `browser-extension/src/background.js`
- `polylogue/browser_capture/models.py:22-35`
- `polylogue/sources/parsers/browser_capture.py:83-132`
- `polylogue/browser_capture/server.py`
- `polylogue/storage/blob_store.py`
