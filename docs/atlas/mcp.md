# MCP

## Area boundary

The live MCP surface is a ten-tool operation algebra. Six read tools are always available; four privileged dispatchers appear only under independent capability flags (`polylogue/mcp/declarations/registry.py:1-6`; `polylogue/mcp/declarations/models.py:11-40`).

## Tool inventory

| Tool | Gate | Role |
| --- | --- | --- |
| `query` | read/default | Execute terminal query pages and projections (`polylogue/mcp/declarations/registry.py:65-79`) |
| `read` | read/default | Read a stable archive URI or ref through a declared view (`polylogue/mcp/declarations/registry.py:80-93`) |
| `get` | read/default | Resolve one exact object identity (`polylogue/mcp/declarations/registry.py:94-107`) |
| `explain` | read/default | Explain grammar, capabilities, refs, semantics, or recovery (`polylogue/mcp/declarations/registry.py:108-121`) |
| `context` | read/default | Compile bounded policy-gated context with receipts (`polylogue/mcp/declarations/registry.py:122-135`) |
| `status` | read/default | Report archive authority and readiness (`polylogue/mcp/declarations/registry.py:136-149`) |
| `write` | `write` | Dispatch declared mutations (`polylogue/mcp/declarations/registry.py:150-166`) |
| `judge` | `judge` | Decide assertion candidates (`polylogue/mcp/declarations/registry.py:167-180`) |
| `run` | `write` | Execute saved query or recipe refs (`polylogue/mcp/declarations/registry.py:181-194`) |
| `maintenance` | `maintenance` | Preview, execute, inspect, and rebuild (`polylogue/mcp/declarations/registry.py:195-210`) |

`write`, `judge`, and `maintenance` are independent booleans, not a role ladder. `run` shares the `write` gate (`polylogue/mcp/declarations/models.py:16-40`; `tests/unit/mcp/test_tool_declarations.py:26-41`).

## Declaration and discovery path

1. `_CUTOVER_TOOL_ROWS` declares each name, discovery text, registrar, capability, verb, result semantics, schema source, example, output kind, and operation owner (`polylogue/mcp/declarations/registry.py:65-210`).
2. `_cutover_declaration` lowers each row into the shared declaration kernel, including handler binding, output contract, and discovery completeness edge (`polylogue/mcp/declarations/registry.py:214-264`).
3. Import-time registry validation rejects duplicate or incomplete declarations (`polylogue/mcp/declarations/registry.py:267-281`).
4. `build_server` wraps FastMCP in `DeclaredToolRegistrar`, registers handlers, then requires exact capability-visible parity before adding resources and prompts (`polylogue/mcp/server.py:55-97`).
5. The registrar rejects undeclared handlers, capability violations, wrong implementation modules, discovery-text drift, duplicates, missing handlers, and extras (`polylogue/mcp/declarations/adapter.py:47-84`; `polylogue/mcp/declarations/adapter.py:97-149`).

## `EXPECTED_TOOL_NAMES`

- Production-visible names come from `declared_tool_names(capabilities)`, which filters declarations through capability checks (`polylogue/mcp/declarations/registry.py:294-308`).
- Test infrastructure derives `EXPECTED_TOOL_NAMES` from the all-capabilities declaration set rather than maintaining a second copied list (`tests/infra/mcp.py:21-30`).
- The six-name read baseline remains frozen independently, so deleting both a handler and its declaration cannot self-authorize a public surface contraction (`tests/infra/mcp.py:19-30`; `tests/unit/mcp/test_tool_declarations.py:18-23`).
- Every registered tool must also appear in `TOOL_CONTRACT`, and stale classifications fail (`tests/unit/mcp/test_envelope_contracts.py:94-122`).

## Operation to contract flow


verified: 4abb7a80bca2160d27fdc799891305cf02b680ff 2026-08-25
