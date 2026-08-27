Summary

Add write-gated MCP tools for typed agent work events and decisions. Events use the existing raw-and-parsed append ingest seam and are stored in session_events.

Problem

Live agents had no mutation surface for recording tool runs, subagent spawns, decisions, or artifact changes. The existing observed-event substrate could read archived evidence but could not accept these events during a session.

Solution

Add record_work_event and emit_decision declarations and handlers under the existing write capability. Add deterministic event identities, typed event-kind validation, facade methods, and an ArchiveStore append path using source_index=-1. Reposting an event reuses the existing idempotent raw/index ingest behavior. Add declaration, envelope, and archive behavior coverage.

Verification

`bash nix/devtools-wrapper.sh test tests/unit/mcp/test_tool_declarations.py tests/unit/mcp/test_envelope_contracts.py` passed: 31 passed.

`bash nix/devtools-wrapper.sh test tests/unit/mcp/test_work_event_tools.py` passed: 1 passed.

`bash nix/devtools-wrapper.sh render openapi && bash nix/devtools-wrapper.sh render cli-output-schemas && bash nix/devtools-wrapper.sh render all --check` passed: generated surfaces synchronized and site links resolved.

`bash nix/devtools-wrapper.sh verify --quick` reached `ruff format` and was blocked because the `ruff` executable is unavailable in the managed environment.

Residual risk

The quick static gate could not complete until the managed environment provides ruff. Full corpus verification was not run.
