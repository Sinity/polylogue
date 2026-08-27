Summary

- Retire the finite attachment acquisition-debt maintenance command and its CLI tests.
- Keep attachment byte/reference coverage in the archive verification path under neutral names.
- Preserve the existing source-specific acquisition, true-hash CAS publication, and typed failure behavior.

Problem

The temporary attachment debt report exposed rebuildable index history as a maintenance product. Current source acquisition and archive verification already own the relevant byte/reference checks, so the public command duplicated a transitional route.

Solution

The internal index projection is now `AttachmentCoverageReport` / `scan_attachment_coverage`, and archive verification exposes it as `attachment-coverage`. The old command registration, implementation, CLI tests, and stale references were removed.

Verification

- `bash nix/devtools-wrapper.sh test tests/unit/storage/test_attachment_acquisition.py tests/unit/sources/test_drive_attachment_fetch.py tests/unit/security/test_attachment_security.py`: 71 passed, 1 warning.
- Focused consolidated suite: 349 passed, 2 warnings.
- Follow-up verification suite: 175 passed, 2 warnings.
- `bash nix/devtools-wrapper.sh verify oracle-integrity`: passed; 1159 modules scanned, 0 new reachability issues.
- `bash nix/devtools-wrapper.sh render devtools-reference --check`: sync OK.
- `git diff --check`: passed.
- `bash nix/devtools-wrapper.sh verify --quick`: blocked because `ruff format` is missing from the managed environment.

Residual risk

The packet names `tests/unit/storage/test_attachment_reacquisition.py`, which is absent from this checkout. The requested live clean-generation census receipt was not available through the local tooling, and the full verify result could not be captured; no claim is made for either.
