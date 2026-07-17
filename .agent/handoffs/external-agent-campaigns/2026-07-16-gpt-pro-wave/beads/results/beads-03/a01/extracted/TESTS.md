# Beads 03 test and verification record

## Test design

The tests are split into four independent proof layers.

1. **Fixed declaration parity.** A committed 104-row declaration witness is compared to the production registry. This prevents the runtime server and expected inventory from drifting together unnoticed.
2. **Live FastMCP wire parity.** The real admin server is listed through FastMCP and compared to a committed witness containing description, input schema, output schema, annotations, and required role. A separate schema witness remains compatible with existing consumers.
3. **Registration anti-bypass.** Every one of the 56 migrated handlers must carry the exact declaration object installed by `register_declared_tool`; direct `@mcp.tool()` registration or a wrapper that bypasses the adapter fails.
4. **Real operations route.** A seeded archive is read through registered FastMCP handlers, not descriptor methods: search reaches ArchiveStore/FTS, ObjectRef resolution reaches the Polylogue facade/archive, topology reaches lineage reads, and usage timeline reaches the insight registry/facade operation.

Existing contract/discovery/envelope tests remain in place and continue to exercise pagination, continuation, error, absence, role, dynamic signature, and output-shape behavior.

## Production dependencies exercised

### `search`

Route: FastMCP handler -> `server_tools.search` -> `archive_search_payload` / canonical query request -> ArchiveStore/FTS.

Planted oracle: only the parent session contains `declaration route needle`; expected total is 1 and the returned canonical session ID is independently known from `ArchiveStore.write_parsed`.

### `resolve_ref`

Route: FastMCP handler -> `Polylogue.resolve_ref` -> ObjectRef parsing and ArchiveStore point read.

Planted oracle: `session:<parent_id>` resolves to the planted parent, preserves its canonical normalized ref and object-ref list; `session:declaration-missing` remains a typed unresolved payload with no fabricated object.

### `get_session_topology`

Route: FastMCP handler -> `Polylogue.get_session_topology` -> real archive lineage reads.

Planted oracle: the child was written with the parent's provider identity, so root and ordered thread are known from input facts.

### `usage_timeline`

Route: dynamic insight registration -> insight registry -> real Polylogue usage-timeline operation.

Planted oracle: the archive has no usage rows, so the typed envelope is exactly `{"total": 0, "items": []}`.

## Anti-vacuity mutations

- Delete or rename a declaration row: the fixed 104-row witness and exact inventory test fail.
- Change a declaration role: role-scoped discovery fails against the fixed declaration witness.
- Restore a direct decorator or bypass the declaration adapter: the migrated-handler marker test fails.
- Add a generic `*args/**kwargs` wrapper or change a signature: the live FastMCP input/output schema witness fails.
- Change a description, annotation, output model, or role: the full wire-contract witness fails.
- Remove the FTS/query call or return a stubbed search payload: the seeded search assertion fails.
- Normalize ObjectRefs differently or collapse absence into an exception/error: the real resolve-ref assertions fail.
- Remove parent traversal or sort topology incorrectly: the planted root/thread assertions fail.
- Bypass the insight registry/facade for the generic timeline tool: the real-route test or handler declaration marker fails.
- Reintroduce a hidden page cap, wrong cursor, wrong offset origin, or not-found exception: the existing search/message/envelope regression selections fail.
- Change any generated witness without running its renderer: `render mcp-contract-fixtures --check` fails.

## Commands and results

### Formatting, lint, and typing

```text
.venv/bin/ruff format --check <17 changed Python files>
17 files already formatted

.venv/bin/ruff check <17 changed Python files>
All checks passed!

.venv/bin/mypy polylogue/mcp/declarations \
  polylogue/mcp/server_tools.py \
  polylogue/mcp/server_insight_tools.py \
  devtools/render_mcp_contract_fixtures.py \
  tests/unit/mcp/test_read_tool_declarations.py
Success: no issues found in 8 source files
```

### Declaration, wire, discovery, and legacy MCP contracts

```text
pytest -q tests/unit/mcp/test_read_tool_declarations.py \
  tests/unit/mcp/test_envelope_contracts.py
43 passed in 10.76s

pytest -q tests/unit/mcp/test_tool_discovery.py
69 passed in 19.17s

pytest -q \
  tests/unit/mcp/test_tool_contracts.py::TestReadViewProfilesTool \
  tests/unit/mcp/test_tool_contracts.py::TestInsightTools \
  tests/unit/mcp/test_tool_contracts.py::TestArchiveTools \
  tests/unit/mcp/test_tool_contracts.py::TestStatsTool \
  tests/unit/mcp/test_tool_contracts.py::TestGetSessionSummaryTool \
  tests/unit/mcp/test_tool_contracts.py::TestQueryExplanationTool \
  tests/unit/mcp/test_tool_contracts.py::TestQueryCompletionsTool \
  tests/unit/mcp/test_tool_contracts.py::TestActionAffordancesTool
36 passed in 18.86s

pytest -q \
  tests/unit/mcp/test_tool_contracts.py::TestQueryTools \
  tests/unit/mcp/test_tool_contracts.py::test_mcp_search_params_match_query_spec
32 passed in 17.60s

pytest -q tests/unit/mcp/test_tool_contracts.py::TestMutationTools
30 passed in 12.21s
```

The three bounded `test_tool_contracts.py` lanes collect all 98 tests in that file and passed. They were split because a combined command exceeded the execution window despite continuing to report passing progress.

```text
pytest -q \
  tests/unit/mcp/test_server_surfaces.py::TestServerSurfaceRegistration \
  tests/unit/mcp/test_server_surfaces.py::TestArchiveGenericToolSurfaces
29 passed in 13.75s
```

Focused pagination/absence selection after the final ObjectRef test change:

```text
pytest -q \
  tests/unit/mcp/test_tool_contracts.py::TestQueryTools::test_search_total_and_cursor_reflect_more_native_hits \
  tests/unit/mcp/test_tool_contracts.py::TestGetSessionSummaryTool::test_get_messages_rejects_unknown_offset_from \
  tests/unit/mcp/test_tool_contracts.py::TestGetSessionSummaryTool::test_get_not_found \
  tests/unit/mcp/test_tool_contracts.py::TestGetSessionSummaryTool::test_get_messages_large_filtered_offset_returns_actionable_note \
  tests/unit/mcp/test_tool_contracts.py::TestMutationTools::test_summary_not_found
5 passed in 4.06s
```

Final direct run of the new real-route/declaration file:

```text
pytest -q tests/unit/mcp/test_read_tool_declarations.py
8 passed in 4.38s
```

### Generated surfaces and docs reachability

```text
python -m devtools render mcp-contract-fixtures --check
render mcp-contract-fixtures: sync OK

python -m devtools render mcp-tool-index --check
render mcp-tool-index: sync OK

python -m devtools render devtools-reference --check
render devtools-reference: sync OK: docs/devtools.md

python -m devtools render topology-projection --check
Wrote docs/plans/topology-target.yaml with 1011 rows. TBD: 9.

python -m devtools render topology-status --check
exit 0

python -m devtools verify docs-coverage
docs-coverage: every public CLI command, MCP tool, config key, and stable route is reachable

pytest -q \
  tests/unit/devtools/test_command_catalog.py \
  tests/unit/devtools/test_generated_surfaces.py \
  tests/unit/devtools/test_render_devtools_reference.py \
  tests/unit/devtools/test_lineage_validation.py
18 passed in 2.71s
```

The topology projection command rewrites its output even when passed `--check`; a second run produced identical file hashes. The generated changes include the four new declaration modules and current-source line-count refreshes.

### Exact before/after discovery and wire comparison

Using the installed FastMCP SDK, a clean base worktree and the patched worktree were listed directly for every role.

```text
read   66 tools: ordered discovery exact
write  95 tools: ordered discovery exact
review 97 tools: ordered discovery exact
admin 104 tools: ordered discovery exact
```

A pre-change surface capture with SHA-256 `f37109eaf1b3fe37c97da43e32a75cbe38133f2cb4be00e8bd5a27c84b40e336` was also compared by name. For every role, descriptions, input schemas, output schemas, and annotations were exactly equal.

### Patch/apply proof

Against a fresh detached worktree at `f654480cadb7cc4c194704e24dfd483199547b35`:

```text
git apply --check --binary PATCH.diff
pass

git apply --index --binary PATCH.diff
pass

git diff --cached --check
pass

regenerated staged binary diff compared with PATCH.diff
exact byte equality: pass

PYTHONPATH=<clean-patched-worktree> pytest -q tests/unit/mcp/test_read_tool_declarations.py
8 passed in 4.45s

PYTHONPATH=<clean-patched-worktree> python -m devtools render mcp-contract-fixtures --check
render mcp-contract-fixtures: sync OK
```

### Aggregate timeout disclosure

A one-shot command containing all changed MCP/devtools suites was attempted. It emitted 118 passing progress markers and no failure, then was terminated by the execution timeout before completion. No aggregate success is claimed; the relevant files/classes were completed in the bounded lanes above.

## Not run / unverified

- Full repository test suite.
- `devtools verify --quick` or `--all` under the repository's canonical Nix environment.
- Live daemon, browser, extension, real archive, secrets, or deployed MCP server.
- Incident-scale memory/cancellation/disconnect/resource-cleanup replay.
- Cold-model discovery trials.
- Verification against the unavailable accepted `beads-02` result package.
- A generated Python semantic-operation parity matrix; current source has no such renderer/artifact and `polylogue-s1kr` remains open.
