## Summary

Remove the session-insight materialization ledger and converge derived session rows through the generic `derived` stage.

## Problem

The ledger duplicated derived freshness state and could drift from normal convergence.

## Solution

Remove the marker schema, repair path, readiness flags, and private periodic drain. Use source-bound provenance, migrate legacy `insights` debt to `derived`, and require the v85 semantic rebuild boundary.

## Verification

`nix develop --accept-flake-config --command devtools verify --quick` passed.

The affected insight, storage, daemon, and source selection initially reported 1,068 passed. The remaining packet fixture failure was fixed and its direct regression test passed. The three residual failures reproduce on `origin/master`: two schema-status tests and `test_git_commit_adapter_raises_explicitly_for_non_repository`.

## Residual risk

`devtools verify atlas` is blocked by stale citation commits in multiple atlas sheets, including files outside this packet.
