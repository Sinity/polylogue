Summary

- Expose a read-only `DeclarationRegistryProtocol` for generic derivation.
- Validate completeness edges against registered producer declarations.
- Derive MCP smoke arguments and envelope classifications from live tool declarations.
- Add a regression proving unknown producer edges produce an actionable diagnostic.

Problem

MCP smoke arguments and envelope classifications were maintained in separate test lists, so a tool declaration could drift from its executable examples or response contract.

Solution

The MCP declaration’s executable example and output classification now provide both test projections. Registry validation also checks that each completeness edge names a registered declaration or producer.

Verification

- `./.venv/bin/python -m devtools test tests/unit/declarations/test_registry.py tests/unit/declarations/test_synthetic_domain.py tests/unit/mcp/test_tool_declarations.py tests/unit/mcp/test_tool_discovery.py tests/unit/mcp/test_envelope_contracts.py` — 51 passed.
- `PATH="$PWD/.venv/bin:$PATH" ./.venv/bin/python -m devtools verify --quick` — all checks passed.
- `./.venv/bin/python -m devtools render all --check` — generated surfaces synchronized.

Residual risk

The broader epic still requires migration of additional declaration families and the `devtools new tool` scaffold; those are separate slices and are not claimed here.
