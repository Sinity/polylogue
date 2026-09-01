Summary

Preserves deferred Claude cursor authority and covers duplicate installed daemon routes.

Problem

A deferred Claude cursor recomputed semantic authority from rewritten bytes and accepted the new prefix as an append baseline. Route validation rejected duplicate installed bindings without a regression test.

Solution

Retain semantic authority only when the current prefix at the stored boundary exactly matches the existing frontier. Add the installed-binding duplicate route test.

Verification

- `nix develop --accept-flake-config --command devtools test tests/unit/sources/test_live_deferred_append_dedup.py`: 7 passed.
- `nix develop --accept-flake-config --command devtools test tests/unit/daemon/test_route_contracts.py`: 128 passed.
- `nix develop --accept-flake-config --command devtools verify --quick`: exit 0.

Residual risk

The complete corpus and live daemon were not run.

LANE-BRANCH: fix/codex-triage-rest
LANE-COMMIT: 28687fca5
LANE-QUICK: green
LANE-CLASSIFICATION: fixed
