## Summary

Remove the session-insight materialization ledger and converge derived session rows through the generic `derived` stage.

## Problem

The ledger duplicated derived freshness state and could drift from normal convergence.

## Solution

Remove the marker schema, repair path, readiness flags, and private periodic drain. Use source-bound provenance and migrate legacy `insights` debt to `derived`. Remove stale tests for the retired lifecycle and regenerate schema disposition evidence.

## Verification

`uv run devtools verify --quick` passed all gates, including schema-versioning. The packet command `uv run devtools test tests/unit/storage/insights tests/unit/insights` failed before collection because the first path does not exist. The corrected broad selection was killed by the host pytest queue after unrelated failures; it has no completion receipt.

## Residual risk

The broad focused selection needs a fresh managed run if full storage and insight test coverage is required.
