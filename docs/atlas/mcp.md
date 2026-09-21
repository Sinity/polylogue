# MCP

## Area boundary

The live MCP surface is a twelve-tool operation algebra. Six read tools are always available; six privileged tools appear only under independent capability flags (`polylogue/mcp/declarations/registry.py:70-274`; `polylogue/mcp/declarations/models.py:11-40`).

## Tool inventory

| Tool | Gate | Role |
| --- | --- | --- |
| `query` | read/default | Execute terminal query pages and projections; also carries the discriminated `session_operation` request (`polylogue/mcp/declarations/registry.py:71-91`) |
| `read` | read/default | Read a stable archive URI or ref through a declared view (`polylogue/mcp/declarations/registry.py:92-112`) |
| `get` | read/default | Resolve one exact object identity (`polylogue/mcp/declarations/registry.py:113-127`) |
| `explain` | read/default | Explain grammar, capabilities, refs, semantics, or recovery, and answer shared query completions via `subject="completions"` (`polylogue/mcp/declarations/registry.py:128-143`) |
| `context` | read/default | Compile bounded policy-gated context with receipts (`polylogue/mcp/declarations/registry.py:144-158`) |
| `status` | read/default | Report archive authority and readiness (`polylogue/mcp/declarations/registry.py:159-175`) |
| `write` | `write` | Dispatch declared mutations; the named destructive operations fail closed without `confirm=true` (`polylogue/mcp/declarations/registry.py:176-193`) |
| `record_work_event` | `write` | Append a typed live-agent event (`polylogue/mcp/declarations/registry.py:194-208`) |
| `emit_decision` | `write` | Append a decision event with evidence references (`polylogue/mcp/declarations/registry.py:209-223`) |
| `judge` | `judge` | Decide assertion candidates (`polylogue/mcp/declarations/registry.py:224-238`) |
| `run` | `write` | Execute saved query or recipe refs (`polylogue/mcp/declarations/registry.py:239-255`) |
| `maintenance` | `maintenance` | Rebuild derived insights and inspect or adjudicate operation recovery (`polylogue/mcp/declarations/registry.py:256-273`) |

`write`, `judge`, and `maintenance` are independent booleans, not a role ladder. `run`, `record_work_event`, and `emit_decision` all share the `write` gate, so enabling `write` exposes four tools beyond the read baseline (`polylogue/mcp/declarations/models.py:16-40`; `tests/unit/mcp/test_tool_declarations.py:27-49`).

`record_work_event` and `emit_decision` carry `target_visible=False`, which removes them from the *target* transaction algebra (`TARGET_DEFAULT_READ_ALGEBRA`/`PRIVILEGED_ALGEBRA`) while leaving them fully live and registered. Do not read a target-algebra projection as the live tool count (`polylogue/mcp/declarations/registry.py:386-397`).

## Declaration and discovery path

1. `_CUTOVER_TOOL_ROWS` declares each name, discovery text, registrar, capability, verb, result semantics, schema source, example, output kind, and operation owner (`polylogue/mcp/declarations/registry.py:70-274`).
2. `_cutover_declaration` lowers each row into the shared declaration kernel, including handler binding, output contract, and the discovery completeness edge to `tests.infra.mcp.EXPECTED_TOOL_NAMES` (`polylogue/mcp/declarations/registry.py:277-339`).
3. Import-time registry validation rejects duplicate names and incomplete declarations, raising at module import (`polylogue/mcp/declarations/registry.py:342-355`).
4. `build_server` wraps MCPServer in `DeclaredToolRegistrar`, registers handlers, then calls `finalize()` to require exact capability-visible parity before adding resources and prompts (`polylogue/mcp/server.py:48-90`).
5. The registrar rejects undeclared handlers, capability violations, wrong implementation modules, discovery-text drift, and duplicates at registration, then missing handlers and extras at `finalize()` (`polylogue/mcp/declarations/adapter.py:47-105`; `polylogue/mcp/declarations/adapter.py:107-149`).

## `EXPECTED_TOOL_NAMES`

- Production-visible names come from `declared_tool_names(capabilities)`, which filters declarations through capability checks (`polylogue/mcp/declarations/registry.py:372-384`).
- Test infrastructure derives `EXPECTED_TOOL_NAMES` from the all-capabilities declaration set rather than maintaining a second copied list (`tests/infra/mcp.py:25-30`).
- The six-name read baseline remains frozen independently, so deleting both a handler and its declaration cannot self-authorize a public surface contraction (`tests/infra/mcp.py:18`; `tests/unit/mcp/test_tool_declarations.py:19-24`).
- Every registered tool must also appear in `TOOL_CONTRACT`, and stale classifications fail (`tests/unit/mcp/test_envelope_contracts.py:85-113`).

## Operation to contract flow

Session operations have individual request/result/error schemas generated from
owner models (`polylogue/operations/session_contracts.py:1-80`). The existing
`query` dispatcher accepts these as a discriminated `session_operation`
request. `polylogue.operations.session_reads.execute_session_operation` owns
execution; the MCP and machine CLI are adapters. Other MCP operations retain
the tool-level contracts above. See [session operations](../session-operations.md) for paging,
original-source fallback, and clock semantics.

Public filters in these contracts are `origin`-typed (`polylogue/core/enums.py:86`);
`RawOrigin` is a separate narrow literal for raw-source reads. No MCP request
field takes a `Provider` (`polylogue/operations/session_contracts.py:9-16`).

## Insight projections

MCP insight projections still bypass `analysis/registry.py`, so a descriptor
added to the insight registry does not automatically reach MCP. Check the MCP
projection explicitly when adding or renaming an insight.

## DISCREPANCIES

None recorded for the agent manual. The tool count and the `maintenance`
confirmation gate are both derived: `declared_tool_names` is the sole authority
for the tool surface (`polylogue/mcp/declarations/registry.py:377-388`) and the
manual renders its list and spelled count from it, while the gate is declared
once as a `ConfirmationGate` on the maintenance contract
(`polylogue/agent_integration/spec.py:289-293`) and rendered from there
(`devtools/render_agent_manual.py:146-170`).

`record_work_event` and `emit_decision` carry `target_visible=False`, so they
have no target transaction while remaining live write-gated tools
(`polylogue/mcp/declarations/registry.py:199-227`). Outside the target algebra
is not the same as not a tool; the manual lists them.
