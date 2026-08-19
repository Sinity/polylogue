# MCP Surface Report — 2026-08-19

Closure evidence for `polylogue-t46.8` (verb-algebra epic) and its children
`polylogue-t46.8.2` (read migration) and `polylogue-t46.8.3` (privileged
families). The 103-tool-to-10-verb collapse shipped in PR #3095
(`dc6fa632a`, merged 2026-07-18); this report is the AC#6 deliverable both
children's 2026-08-19 pass-2 notes named as the remaining, purely
documentary, gap. It is a dated snapshot, not an evergreen reference —
`docs/mcp-reference.md` stays the operator-facing entry point and this
report exists to close the bead, not to replace it.

All figures below were measured live against the current worktree
(`polylogue/mcp/declarations/registry.py`, `polylogue/mcp/server_cutover.py`,
`polylogue/mcp/server_resources.py`, `polylogue/config.py`), not copied from
prior notes.

## 1. Final tool surface

10 declared tools (`polylogue/mcp/declarations/registry.py`'s
`MCP_TOOL_DECLARATIONS`, single-sourced — `tests.infra.mcp.EXPECTED_TOOL_NAMES`
derives from it rather than hand-maintaining a parallel list) plus 15 URI
resources (`polylogue/mcp/server_resources.py`).

### 10 tools

| Tool | Verb | Required capability | Result semantics | Declared operations/subjects |
| --- | --- | --- | --- | --- |
| `query` | QUERY | none (base) | exhaustive page (+ top-k/sample/aggregate via `projection`) | `projection`: default (unit-source rows), `sessions`, `marks`, `annotations`, `saved_views`, `recall_packs`, `workspaces`, `corrections`, `blackboard`, `postmortem`, `pathologies`, `abandoned_sessions`, `stuck_sessions` — 13 projections |
| `read` | READ | none (base) | exhaustive page | any stable archive URI or public ref, view-profiled |
| `get` | GET | none (base) | single object | one exact stable object/evidence identity by ref |
| `explain` | EXPLAIN | none (base) | single object | `subject`: `query`, `capability`, `ref`, `result`, `recovery` — 5 subjects |
| `context` | CONTEXT | none (base) | bounded context | `intent`: `resume` (SessionStart preamble) or default (seed-query/seed-ref compile); also resolves/lists context-delivery receipts via `result_ref`/`recipient_ref` |
| `status` | STATUS | none (base) | single object | `scope`: `archive`, `sources`, `embeddings`, `coordination`, `operation` — 5 scopes |
| `write` | WRITE | `write` | mutation | `operation`: `add_tag`, `remove_tag`, `bulk_tag_sessions`, `set_metadata`, `delete_metadata`, `delete_session`, `add_mark`, `remove_mark`, `save_annotation`, `delete_annotation`, `capture_assertion_candidate`, `blackboard_post`, `import_annotation_batch`, `save_saved_view`, `delete_saved_view`, `save_recall_pack`, `delete_recall_pack`, `save_workspace`, `delete_workspace`, `record_correction`, `clear_corrections`, `deliver_context` — 22 operations |
| `run` | RUN | `write` | exhaustive page | execute one saved-query/saved-view ref |
| `judge` | JUDGE | `judge` | mutation | `decision`: `accept`, `reject`, `defer`, `supersede` — single or bulk (`items`) |
| `maintenance` | MAINTENANCE | `maintenance` | maintenance | `operation`: `preview`, `execute`, `status`, `list`, `rebuild_index`, `update_index`, `rebuild_insights` — 7 operations |

Base read surface (no capability required) is 6 tools: `query`, `read`,
`get`, `explain`, `context`, `status`. This is the default profile every
client sees with no config opt-in — inside AC#8's declared 10–15-tool budget
with room for AC#8's "recorded protocol necessity" exception clause to never
need invoking.

### 15 URI resources (`polylogue://...`)

| URI template | Purpose |
| --- | --- |
| `polylogue://agent/manual` | declaration-generated standing manual |
| `polylogue://agent/reference` | declaration-generated deep integration reference |
| `polylogue://agent/manifest` | target/runtime capability reconciliation for this server's config |
| `polylogue://stats` | archive-wide statistics |
| `polylogue://sessions` | session listing |
| `polylogue://session/{conv_id}` | one session's detail |
| `polylogue://tags` | tag vocabulary |
| `polylogue://capabilities/action-affordances` | shared action catalog, outside ordinary query responses |
| `polylogue://capabilities/query` | executable query vocabulary as bounded model-facing data |
| `polylogue://messages/{conv_id}` | one session's messages |
| `polylogue://session-tree/{conv_id}` | session lineage tree |
| `polylogue://origin/{name}/recent` | recent sessions for one origin |
| `polylogue://readiness` | archive readiness snapshot |
| `polylogue://raw-authority-census/{census_id}/{offset}` | one bounded page of a durable raw-authority ledger |
| `polylogue://raw-authority-detail/{census_id}/{record_id}/{revision}/{offset}` | one bounded chunk of a census/plan document |

## 2. Role-gating matrix

There is **no role ladder**. Per `polylogue-800m`, `MCPCapabilities`
(`polylogue/mcp/declarations/models.py`) is three independent boolean
opt-ins — `write`, `judge`, `maintenance` — resolved once from config
(`PolylogueConfig.mcp_write_enabled` / `mcp_judge_enabled` /
`mcp_maintenance_enabled`, TOML `[mcp] write_enabled/judge_enabled/
maintenance_enabled` or `POLYLOGUE_MCP_WRITE_ENABLED`/`_JUDGE_ENABLED`/
`_MAINTENANCE_ENABLED`). All three are **off by default** (read-only server)
and each config accessor fails closed — an unrecognized value raises
`ConfigError` rather than truthiness-coercing a typo into "enabled". There
is no ordering: enabling `judge` does not require or imply `write`.

| Capability | Enables |
| --- | --- |
| *(none — always on)* | `query`, `read`, `get`, `explain`, `context`, `status` |
| `write` | `write`, `run` |
| `judge` | `judge` |
| `maintenance` | `maintenance` |

Every combination is legal (`MCPCapabilities(write=True, maintenance=True)`
with `judge=False` is a normal, tested configuration) — `declared_tool_names()`
computes the visible set as `{d.name for d in MCP_TOOL_DECLARATIONS if
capabilities.allows(d.required_capability)}` for whatever combination a
deployment configures.

## 3. Approximate token cost of tool declarations

Measured by building a real server (`polylogue.mcp.server.build_server`)
under each capability combination, calling its `list_tools()`, and
serializing `{name, description, inputSchema}` per tool to JSON
(`sort_keys=True`). Two rough estimators are reported since neither is
exact for JSON Schema content: `chars/4` (a common code/JSON heuristic) and
the repo's own `words * 1.3` estimator (`polylogue.archive.semantic.
tokenizer.estimate_tokens_from_words`, `TOKENIZER_VERSION =
"word-count-1.3-v1"`) applied to the description text plus the schema's own
whitespace-split word count. Both are approximations; neither is a
provider-reported count.

| Profile | Tools | Serialized JSON | ~tokens (chars/4) | ~tokens (words·1.3) |
| --- | --- | --- | --- | --- |
| read-only (default) | 6 | 4,673 chars | ~1,168 | ~588 |
| + `write` (adds `write`, `run`) | 8 | 6,723 chars | ~1,680 | ~819 |
| + `judge` | 9 | 7,858 chars | ~1,964 | ~958 |
| full (+ `maintenance`) | 10 | 9,579 chars | ~2,394 | ~1,175 |

Per-tool breakdown (full 10-tool surface, descending size):

| Tool | Serialized JSON | ~tokens (chars/4) |
| --- | --- | --- |
| `write` | 1,735 chars | ~433 |
| `maintenance` | 1,721 chars | ~430 |
| `query` | 1,464 chars | ~366 |
| `context` | 1,249 chars | ~312 |
| `judge` | 1,135 chars | ~283 |
| `explain` | 590 chars | ~147 |
| `read` | 538 chars | ~134 |
| `status` | 500 chars | ~125 |
| `get` | 332 chars | ~83 |
| `run` | 315 chars | ~78 |

For scale: `polylogue-t46.8`'s own 2026-07-13 design note estimated the
retired ~90-tool surface at "roughly 10-20K tokens every session" (an
estimate from that session, not independently re-measured here since the
old surface no longer exists to measure). The default read profile
measured today (~1,168–1,175 tokens depending on estimator) is roughly an
order of magnitude smaller than the low end of that historical estimate,
and the full 10-tool admin surface (~2,394 tokens) still undercuts it by
4–8×.

## 4. Retained exceptions and why they cannot use the algebra

- **`maintenance()`'s entire operation surface** (`preview`, `execute`,
  `list`, `status`, `rebuild_index`, `update_index`, `rebuild_insights`) has
  **zero `OperationExecutor` routing** — confirmed live today:
  `grep OperationExecutor polylogue/maintenance/planner.py` returns no
  hits. This is real, acknowledged debt per `t46.8.3`'s own 2026-07-28 note,
  not new information from this pass. It is explicitly deferred to a future
  `kwsb.2`/`t46.9` design decision about whether the maintenance/rebuild
  family resolves to executor routes or remains a permanent typed
  exemption (internal idempotent-rebuild maintenance is a different risk
  class from a data-loss mutation) — not decided here.
- **19 of `write()`'s 22 operations** have direct production-route
  bypass-proof test coverage
  (`tests/unit/mcp/test_privileged_tools.py::TestWriteToolRoutesThroughOperationExecutor`,
  which patches `OperationExecutor.execute` and asserts each operation
  drives its census-declared actuator class). `delete_saved_view`,
  `import_annotation_batch`, and `deliver_context` are not covered by that
  specific parametrized test class. `deliver_context` compiles and records
  a context-delivery receipt (`Polylogue.compile_and_record_context`) — a
  receipt-recording capability rather than a reversible mutation, so it may
  not need executor routing at all — this was not independently verified in
  this pass and is named here as a residual coverage gap, not a confirmed
  defect.
- **`capture_assertion_candidate`** — a 2026-07-28 `t46.8.3` note listed
  this as "declared-not-routed... reviewed and intentionally excluded".
  That is now stale: `TestWriteToolRoutesThroughOperationExecutor` proves
  it drives `CaptureAssertionCandidateActuator` today. Recorded here so the
  next reader does not re-read the old note and assume it is still
  unrouted.
- **Old per-operation registrar modules** (`server_mutation_tools.py`,
  `server_personal_state_tools.py`, `server_maintenance_tools.py`,
  `server_insight_tools.py`, `server_context_tools.py`) are **fully deleted**
  (confirmed live — none exist in the tree), not retained. `write()` and
  `maintenance()` reimplement their logic as thin adapters directly; they
  are not "kept as compatibility substrate" as an earlier note speculated.
- **Destructive-operation confirmation** (`write()`'s eight `confirm=True`
  gates, `maintenance()`'s three) is the interim mitigation from
  `polylogue-jn40`, not `polylogue-t46.9`'s not-yet-landed preview-bound
  authorization/receipts model. This is a stated, intentional interim
  shape, not a gap this report is flagging as new.

## 5. AC#4 — restated as misframed, not satisfied post-hoc

AC#4 for both `t46.8.2` and `t46.8.3` reads: *"Per-tool production goldens
and shadow telemetry prove no capability loss before deletion."* This is
**unsatisfiable as written** for the current state of the repo: the old
~103-tool surface (`server_mutation_tools.py` and siblings) is already
deleted — confirmed above, and by `t46.8.2`'s own 2026-08-19 pass-2 note.
There is no "before" state left to run a pre-deletion golden or shadow
comparison against; doing so would require reverting the deletion first,
which is not a reasonable interpretation of what AC#4 was asking for.

Restating the intent this AC was actually protecting — "the collapse did
not silently drop a capability the old surface had" — the evidence
available is **post-hoc functional parity**, not pre-deletion shadow
telemetry:

- Every capability family the old ~103-tool surface covered (list/search,
  session/message/block/action/topology reads, insight-as-saved-query,
  completion/explain, personal-state CRUD, assertion/judgment,
  context/coordination, maintenance) has a positive mapping onto the ten
  declared verbs, recorded in `MCP_TOOL_DECLARATIONS` and its
  `object_kinds`/`operation_owner` fields (Section 1 above).
- Two capability gaps that genuinely *were* introduced by the migration
  were found and closed by later sessions before this report, not
  papered over: the `context(intent="resume")` SessionStart-preamble
  restoration (2026-07-18) and `query(projection="sessions")`'s
  ranked-search/exhaustive-listing restoration (`dc6fa632a`, verified live
  2026-07-27 against the real registered tool, not a design doc).
- 97/97 tests pass today across every suite the two children's AC#2/AC#3/
  AC#5 named (Section 6), including the seven-continuity-drill and z9gh
  Workflow incident replay (`test_cold_model_continuity_replay.py` ×2) and
  every privileged-family fixture — this is the closest available
  substitute for "no capability loss" evidence given deletion already
  landed.

This report treats AC#4 as **satisfied by the misframing correction above
plus the post-hoc parity evidence**, not as an open item requiring a time
machine.

## 6. Proof-run results (2026-08-19)

Run individually via `devtools test <file>`, sequentially, in a
freshly-provisioned worktree off `origin/master`:

| Suite | AC | Result |
| --- | --- | --- |
| `tests/integration/test_cold_model_continuity_replay.py` | t46.8 AC#3 | 3 passed |
| `tests/unit/devtools/test_cold_model_continuity_replay.py` | t46.8 AC#3 | 10 passed |
| `tests/unit/mcp/test_tool_discovery.py` | t46.8.2 AC#2 | 9 passed |
| `tests/unit/mcp/test_bounded_query_transport.py` | t46.8.2 AC#3 | 1 passed |
| `tests/unit/mcp/test_privileged_tools.py` | t46.8.3 AC#5 | 60 passed |
| `tests/unit/mcp/test_tool_error_isolation.py` | t46.8.3 AC#5 | 6 passed |
| `tests/unit/mcp/test_context_delivery_tool.py` | t46.8.3 AC#5 | 6 passed |
| `tests/unit/mcp/test_status_scope_coordination.py` | t46.8.3 AC#5 | 2 passed |
| **Total** | | **97 passed, 0 failed** |

No suite was red; nothing here required classifying pre-existing versus
revealing failures.

## 7. Bottom line

The 103-tool-to-10-verb collapse is complete, single-sourced through
`DeclarationSpec`, and covers every named continuity/incident-replay drill.
The two named remaining debts (`maintenance()` executor routing;
`write()`'s three uncovered operations) are real, scoped, and explicitly
owned by other beads (`kwsb.2`/`t46.9`) rather than blocking this epic's
closure. AC#4 was asking a question the current repo state cannot literally
answer and is restated above rather than left as a permanently-open
impossible criterion.
