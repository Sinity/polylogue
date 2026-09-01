Summary

SQLite source snapshots now bind source continuity to deterministic schema and logical-row identity. The immutable backup remains the parser input, while source drift during backup is rejected. Existing durable append-frontier resynthesis and ambiguous-frontier fallback remain on the rebased master base.

Problem

Filesystem metadata and SQLite page layout change during ordinary commits, checkpoints, and vacuuming. Using those observations as source continuity evidence can create needless revisions or accept a snapshot that does not represent one logical source state.

Solution

Compute the logical revision before and after the online backup, and compare it with the staged backup before publication. Add coverage for equivalent SQLite databases with different insertion order.

Verification

`nix develop --accept-flake-config --command devtools test tests/unit/sources/test_source_snapshot.py tests/unit/sources/parsers/test_codex_state.py -k 'sqlite or snapshot or logical'` passed: 23 tests.

`nix develop --accept-flake-config --command devtools test tests/unit/sources/test_source_snapshot.py::test_sqlite_cut_manifest_hashes_the_published_backup_bytes` passed: 1 test.

`nix develop --accept-flake-config --command devtools test tests/unit/pipeline -k 'sqlite or acquire'` passed: 14 tests.

`nix develop --accept-flake-config --command devtools test tests/property -k 'append or source'` passed: 4 tests.

`nix develop --accept-flake-config --command devtools verify --quick` passed all checks.

`nix develop --accept-flake-config --command devtools verify` refused because the compatible native testmon graph is absent.

Residual risk

The full affected verification corpus was not run because the required native testmon graph is unavailable. The broader append selection had three inherited failures caused by stale derived schema identity before append processing.
