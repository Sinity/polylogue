Summary

Expose the durable Sinex publication ledger through daemon status, direct CLI status, and MCP archive status. Add a transport-independent status reader so operators can inspect lag, retries, receipts, rejection, and primary blocking without configuring a transport.

Problem

`PublicationService.status()` existed but had no external status consumer. Publication backlog and lag were therefore invisible when Sinex mode was enabled. Status reads also should not require deployment transport composition.

Solution

Extract `publication_status()` as the source-tier ledger reader and reuse it from `PublicationService`. Add the resulting secret-safe payload to daemon and direct status, compact CLI output, and MCP archive status. Add a test proving mirror status reads the durable ledger without invoking transport.

Verification

`PATH="$PWD/.venv/bin:$PATH" bash nix/devtools-wrapper.sh test tests/unit/sinex tests/unit/daemon/test_daemon_status.py tests/unit/mcp/test_status_scope_coordination.py` -> `121 passed, 4 warnings in 20.18s`.

`PATH="$PWD/.venv/bin:$PATH" bash nix/devtools-wrapper.sh verify --quick` after rebasing onto `origin/master` -> `.cache/verify/current-run.json` reports `"status": "success"` for all 17 steps.

Residual risk

The current Polylogue repository has no stable Python client or deployment-registered endpoint for the Sinex Rust material, JetStream, DurableEmissionReceipt, and RawEnvelopeSettlement APIs. This change exposes local durable obligation state only. It does not claim live Sinex transport, material confirmation, aggregate settlement, or cross-repository killpoint proof.
