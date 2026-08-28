Summary

Confirm the daemon startup blob-publication reconciliation baseline is resolved.

Problem

The named test previously expected startup reconciliation to delete both a missing blob receipt and a receipt whose blob remains durably referenced. That expectation conflicts with the blob-authority contract: referenced receipts remain for explicit abandonment. Commit `b908947c4` corrected the expectation; the current `origin/master` contains that correction and the product route is green.

Solution

No product-code change is required in this lane. Preserve the existing reconciliation behavior and corrected regression expectation.

Verification

- `nix develop --command bash nix/devtools-wrapper.sh test tests/unit/daemon/test_daemon_cli.py::test_reconcile_blob_publications_clears_terminal_receipts_at_startup` — 1 passed.
- `nix develop --command bash nix/devtools-wrapper.sh test tests/unit/daemon/test_daemon_cli.py` — 129 passed.
- `nix develop --command bash nix/devtools-wrapper.sh verify --quick` — run after rebasing; result recorded in the lane report.

Residual risk

The original baseline report predates the upstream expectation correction. No broader full-corpus claim is made by the focused checks.
