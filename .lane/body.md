## Summary

- Preserve failed convergence stage exceptions so convergence debt records failures instead of deliberate deferrals.
- Bound foreground embedding with the daemon session, message, time, and monthly cost limits.
- Recover unfinished embedding catch-up receipts at startup and prioritize due convergence debt.

## Problem

Stage wrappers returned `False` after catching real FTS, embedding, insight, raw recovery, and standing-query exceptions while those stages declared false results pending. This classified failures as deferred and suppressed failure handling. Foreground embedding also bypassed backlog budgets. Catch-up receipts outside terminal states were never recovered, and a limit applied before due filtering could starve older due debt.

## Solution

Unexpected stage exceptions now reach the converger's failed-state path, while bounded work and transient SQLite contention retain pending behavior. Foreground embedding enforces the shared limits and monthly spend accounting. Startup marks unfinished catch-up receipts interrupted. Debt selection orders due rows before future retries.

## Verification

- `bash nix/devtools-wrapper.sh test tests/unit/daemon/test_convergence_stages.py tests/unit/daemon/test_embedding_convergence_progress.py tests/unit/daemon/test_daemon_cli.py -x` -> `187 passed`
- `bash nix/devtools-wrapper.sh test tests/unit/sources/test_live_catchup_planning.py tests/unit/daemon/test_catch_up_observability.py tests/unit/storage/test_embedding_generations.py -x` -> `43 passed`
- `bash nix/devtools-wrapper.sh verify --quick` -> blocked because `ruff` is missing from the environment.

## Residual risk

The full quick gate remains unverified until the managed environment provides `ruff`. The focused tests cover the affected daemon, cursor, catch-up, and embedding lifecycle paths.
