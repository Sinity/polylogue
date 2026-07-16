# 19. polylogue-83u.2 — Acquire bytes for non-inline sources while live handles are open

Priority: **P2**  
Lane: **attachment-integrity**  
Readiness: **ready-now after census/classification**

Depends on packet(s): polylogue-83u.4

## Why this is urgent / critical-path

Some attachment bytes still exist at source time but are currently bypassed. Those are capture/parser bugs, not irrecoverable gaps.

## Static diagnosis / likely mechanism

Bead design identifies three live-handle boundaries: Drive downloads inside iterator scope, export-zip members while zipfile is open, and local paths guarded by a source-root allowlist. Existing parser model already supports `ParsedAttachment.inline_bytes`.

## Implementation plan

Implementation shape:
1. Drive: restore/download asset bytes with `DriveSourceClient.download_bytes` inside the iterator/lifetime of the source handle.
2. Export ZIP: resolve attachment member and read it while the `ZipFile` is still open; attach as inline bytes.
3. Local path: accept a transport-only `local_source_path`, canonicalize with `realpath`, require it to stay under declared source-root allowlist, reject symlink/`..` escapes before opening.
4. Non-live handles remain `unfetched` with `source_url/source_path`; no synthetic hash.
5. Assert acquired attachment blob hash equals true SHA-256 of bytes and session content hash remains stable for otherwise-identical content.

## Test plan

Tests:
- Drive fixture downloads inside iterator and stores true blob.
- ZIP fixture stores member bytes before close.
- local allowlisted file stores true blob.
- local path escape is rejected and no read happens.
- closed/non-live handle stays unfetched.
- ingest idempotency/content hash unchanged except attachment acquisition state.

## Verification command / proof

`devtools test tests/unit/sources/ -k 'attachment and (drive or zip or local or inline_bytes)'`

## Pitfalls

Do not let source-local paths become arbitrary file read. The allowlist/realpath check is part of the feature, not a hardening add-on.

## Files/functions to inspect or touch

- `polylogue/sources/parsers/drive*.py`
- `polylogue/sources/parsers/drive_support_attachments.py`
- `polylogue/sources/parsers/base_models.py:179-212`
- `source client / zip importer modules`
- `blob write path `_acquire_attachment_blob``
