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

An MCP operation flows from the public operation name to its dispatcher verb and then to the tool contract. Session operations declare per-operation request, result, and error schemas from owner models (`polylogue/operations/session_contracts.py`). The `query` tool dispatches their discriminated `session_operation` request to `polylogue.operations.session_reads.execute_session_operation`; MCP and machine CLI are adapters. Other operations use the tool-level contracts above. See [session operations](../session-operations.md) for paging, original-source fallback, and clock semantics.

Public filters in these contracts are `origin`-typed (`polylogue/core/enums.py:86`);
`RawOrigin` is a separate narrow literal for raw-source reads. No MCP request
field takes a `Provider` (`polylogue/operations/session_contracts.py:9-16`).

## Capability evidence

`explain(subject="capability")` reads only canonical session and message
counters through `Polylogue.storage_counts()`. It does not hydrate recent
sessions or compute archive statistics breakdowns. If an archive tier is
unavailable or its schema is refused, declarations and profile identities
remain available with unknown counts and freshness. The shared outcome is
`degraded` with `archive_counts_unavailable`; a successful empty archive has
measured zero counts. Cancellation and other read errors retain their normal
operation error behavior.

Capability pages carry public `OriginSpec` evidence once in page-level
`evidence.origins`; individual declarations reference the shared snapshot.
Origins whose declaration excludes public filtering are omitted. Session
counts do not prove field values, and message counts do not prove blocks or
actions: only matching canonical observations are reported, with unknown for
unmeasured declarations.

Freshness uses the injected archive's standing FTS query binding and
aggregate-only convergence debt projection. This measured scope certifies
query binding and debt evidence, not whole-archive materializer readiness.
An unavailable binding stays unknown without an exact inspection fallback;
outstanding debt produces `stale_or_degraded` item/page evidence and the
shared degraded outcome. A measured current binding with no debt permits
`request-current`. The count reducer remains unchanged.

## Read-view discovery

`explain(subject="capability")` returns `read_view_profile_ids`: the `view_id`
of every executable viewport profile, from `Polylogue.list_read_view_profiles()`.
The list is the whole declared catalog, independent of the paged
query-capability `items` and their `total`; `limit` and `offset` apply to the
query declarations only. Full profile metadata stays on its own facade and
daemon route (`/api/read-view-profiles`). `read_views` remains the distinct
session-list projection vocabulary. The self-inspection continuity scenario
compares every profile identity against its fixture oracle.

## Evidence pages and runtime configuration

`read("delegation:subtree:<id>", limit=...)` selects nodes in the canonical
storage reader. Its nested subtree payload carries the whole `node_count`
and `max_depth`, page `limit` and `offset`, and `next_offset` plus an opaque
`continuation`. Follow the continuation to enumerate the full subtree. The
cursor seeks by depth, session identity and traversal path; its shared query
frame refuses a changed delegation relation. Object references cover the
returned nodes only. The Python `resolve_ref` and daemon `/api/refs/resolve`
routes accept the same page arguments and run the shared resolution plan.

`context(recipient_ref=...)` passes its limit and offset to the durable user
reader. That reader counts and selects summary columns in one SQLite
snapshot, without reading or decoding `context_image_json`. The facade
returns a counted summary page; `get_context_delivery` remains the exact,
recipient-scoped image read. Receipt pages preserve `next_offset` and observe
the current ledger on each request.

Both `status(scope="sinex")` and the Sinex section of
`status(scope="archive")` use `sinex_mode` from the injected runtime config
projection, together with that projection's source-tier path.

## Insight projections

Registry-backed projections use the insight descriptor's fetch/payload contract and forward the requested offset to descriptors that declare pagination, after their session filters and before their result window. For descriptors that declare `query`, `expression` is that projection's text search, without a second DSL parser. For example, `query(projection="threads", expression="strong", limit=1, offset=1)` selects the second strong work thread. Reference expressions retain their existing reference-pipeline precedence. Descriptors without a query field retain their declared filters.
Session-list projections and the sessions/origin-recent resources carry `unit="sessions"` and the shared object-shaped terminal `outcome`. The typed session-list adapter preserves its owner's verdict and every named coverage gap; advanced listing uses the same outcome owner as other row envelopes. The discriminated `session_operation` family retains its separately declared owner result schema.
`postmortem` and `pathologies` call their analysis facades and attach the shared
terminal `outcome` at the MCP operation boundary. An empty, complete scope is
`empty`; a truncated scope or missing profiles/digests is `degraded` even when
it produced no findings. A nonempty postmortem also names its unavailable
`longest_tool_gap` measurement rather than implying complete coverage.

## Indeterminate mutations

A mutation whose receipt was lost returns `code="indeterminate"`,
`retryable=false`, and the original `request_id`. Both privileged daemon calls
and facade-backed writes use the same typed exception serializer. Recover the
existing daemon operation by that identity before considering a new mutation;
`daemon_required` is reserved for a call that found no resident writer. MCP
cancellation continues to cancel the originally submitted request, not a replay.

## DISCREPANCIES

None recorded for the agent manual. The tool count and the `maintenance`
confirmation gate are both derived: `declared_tool_names` is the sole authority
for the tool surface (`polylogue/mcp/declarations/registry.py:377-388`) and the
manual renders its list and spelled count from it, while the gate is declared
once as a `ConfirmationGate` on the maintenance contract
(`polylogue/agent_integration/spec.py:353-357`) and rendered from there
(`devtools/render_agent_manual.py:146-170`).

`record_work_event` and `emit_decision` carry `target_visible=False`, so they
have no target transaction while remaining live write-gated tools
(`polylogue/mcp/declarations/registry.py:199-227`). Outside the target algebra
is not the same as not a tool; the manual lists them.

MCP insight maintenance forwards an explicit session-ID selection to the sealed daemon planner. Omission selects the full scope; an empty list remains an empty explicit scope. Orchestration `get` preserves its owner’s terminal verdict and gaps in the shared object-shaped outcome envelope, while leaving the evidence fields unchanged.

MCP messages-view authority measures elapsed time from the read operation
boundary with the shared monotonic authority clock, including the transcript
window read. Serialization preserves that authority value.

Typed session pages retain their framed request through every adapter. When
the MCP budget shortens a page, its executable continuation advances from the
returned prefix under the same operation, projection, filters and archive frame.
An oversized messages row returns `MCPMessageFragmentPayload`: ASCII JSON row
bytes with original message identity, row offset, result ref, byte offset and
exact total. Follow the returned read arguments; concatenate contiguous fragments
then JSON-decode once. `fragment_offset` is accepted only with a bound messages
continuation. A retry repeats the same fragment; the last fragment advances one
row. Fragments never masquerade as complete messages or successful empty pages.
The registered signature supplies its input schema; the typed payload owns the
fragment result schema. Session-operation errors retain their contract and log
their canonical error code as failed calls. Blackboard query pages count and
select active notes in one User snapshot rather than enumerate a finite prefix.
