## Summary

Strengthen the reusable source manifest validator with member identity checks, content hashes, authenticated consumption receipts, append-prefix validation, mutable SQLite logical evidence, manifest integrity verification, regular-file enumeration, and fail-closed collision and replacement handling.

## Problem

The existing validator classified append logs by size, accepted same-byte namespace replacement as unchanged, treated a plain hash map as acquisition proof, and invoked SQLite evidence without comparing it to a baseline. Its manifest digest was not checked during recheck.

## Solution

The validator now requires stable device/inode identity and exact bytes for unchanged members. Append transitions prove the original prefix, SQLite transitions compare caller-provided logical snapshots, and spool disappearance requires an authenticated receipt naming a sealed generation. Baseline integrity is verified before any recheck. File-backed export and SQLite roots are supported while symlinks, non-regular members, duplicate identities, root failures, and unexplained replacements block.

The change is limited to the existing maintenance validator and its focused tests. No archive, backup, retention, campaign, or repair state is written. The prior hook topology helper continues to use the canonical declaration function. Duplicate root/helper deletion: no additional duplicate lists were found in the scoped product path. Maintained production LOC changed from 293 to 361, a net increase of 68 lines, with 46 additional test lines.

## Verification

- `nix develop --command devtools test tests/unit/sources/test_source_laws.py tests/unit/sources/test_hook_spool.py tests/unit/maintenance/test_source_manifest_continuity.py` passed: 181 tests.
- `nix develop --command devtools verify oracle-integrity` passed: 1,156 modules scanned, 0 type-checking-only routes, 30 baselined.
- `nix develop --command devtools verify --quick` passed after rebase: format, lint, mypy, rendering, layering, patterns, command and schema checks, oracle integrity, reachability, and privacy registry all passed.
- `nix develop --command devtools verify` refused with exit 2 because the compatible native Testmon graph was unavailable. `devtools why` reported `native_testmon_graph_unavailable` and the absent seed `polylogue-312644d25c43e826db4ddd199302808ba5d4948efd84fefb2f91f13b833fa337`.
- The branch rebased cleanly onto current `origin/master` at commit `e3d94ae34`.

## Residual risk

External backup/runtime evidence was not available to this worktree, so the external source-survival predicate remains unresolved. The validator accepts provider-specific logical snapshot callbacks, but does not itself implement SQLite transaction copying or source-seal persistence. Those remain coordinator-owned integration work.
