## Summary

- Record each secret-scan sweep outcome in the ops stage-event ledger.
- Add a medium daemon-health check for the latest sweep state.
- Cover failed persistence and health degradation with regression tests.

## Problem

The periodic secret-scan loop logged exceptions and retried without leaving an
observable state. Persistent failures therefore looked healthy to daemon health
and operators.

## Solution

Persist completed and retryable failed outcomes as `maintenance.secret_scan_sweep`
stage events. The medium health tier reads the latest event and reports a failed
sweep as an error while preserving the retry path.

## Verification

- `uv run devtools test tests/unit/daemon/test_secret_scan_sweep.py tests/unit/daemon/test_health_contract.py tests/unit/daemon/test_health_check_paths.py` — 72 passed.
- `uv run devtools verify --quick` — Ruff and mypy passed; render gate is blocked by the pre-existing duplicate `docs/artifact-publication.md` documentation entry.

## Residual risk

If `ops.db` itself is unavailable or locked beyond the persistence timeout, the
outcome cannot be recorded and is logged; the existing next-tick retry remains.
