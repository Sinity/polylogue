Summary

Rebased the inherited packet onto current origin/master and removed the duplicate inline tool-outcome derivation added by the unfinished lane. Current master already contains the typed outcome implementation, parser evidence handling, deliberate unknown reasons, and schema lifecycle declaration.

Problem

The inherited lane replay introduced a second `_derive_tool_outcomes` helper in `storage/sqlite/archive_tiers/write.py` and duplicated the `ToolOutcome` import. Keeping both implementations would leave the write choke point ambiguous and stale.

Solution

Retained the canonical `polylogue.sources.tool_outcomes.derive_tool_outcomes` route from current master and removed the duplicate writer helper and import. Rebase conflict resolution preserved current-master parser, schema, fixture, and assertion behavior.

Verification

- `nix develop --command devtools test tests/unit/storage/test_tool_outcome.py tests/unit/storage/test_archive_tiers_archive.py`: 75 passed, 1 failed. The failure is the inherited `test_read_open_rejects_stale_index_with_generation_and_lifecycle_action`, which raises current `SchemaSkewError` instead of the test's older `SchemaVersionMismatchError` expectation.
- `nix develop --command devtools verify --quick`: success, exit 0. Ruff format, Ruff check, mypy, rendering, layering, patterns, CI commands, JavaScript tests, documentation commands, schema checks, oracle integrity, consumer reachability, definition closure, timestamp doctrine, insight honesty, schema promotion, and privacy registry all passed.
- `git fetch origin && git rebase origin/master`: success. Product code is now identical to current origin/master; the only branch diff is the required uncommitted `.lane` publication text.

Residual risk

The focused archive test retains the inherited exception-name mismatch. Full corpus convergence was not run in this lane.

LANE-BRANCH: feature/packet/polylogue-xd0ha
LANE-COMMIT: eab0c2434d8253ff687a5a8b81424e8592e605e2
LANE-QUICK: green
LANE-CLASSIFICATION: inherited test expectation mismatch: current read-open path raises SchemaSkewError
LANE-CLASSIFICATION: packet implementation already present on origin/master; lane commit removes the stale duplicate helper
