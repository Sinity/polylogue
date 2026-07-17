# Polylogue one-read contract cut

Source package inspected: `/mnt/data/polylogue_extract/polylogue/work/polylogue`.

Bead graph inspected: `/mnt/data/polylogue_extract/polylogue/polylogue-beads-export.jsonl`. The extracted worktree does not contain `.beads/issues.jsonl`; the release-gate label `delivery:C-read-evidence-contract` is present on the exported bead records.

This cut treats the target read law as: every user-facing read lowers to `Query construction -> filtering -> projection -> aggregation -> rendering`, with the surface owning only option parsing, preset selection, and delivery side effects. Logic moves downward into archive/query, API/facade, surfaces, rendering, or generated contract registries. No surface should import or mirror storage internals; the layering baseline is the no-backward-import rule in `docs/plans/layering.yaml:1-6`, with explicit surface restrictions for CLI/MCP/API and storage rules at `docs/plans/layering.yaml:13-26`.

## Existing algebra that is real today

The query side is substantial and reusable. `SessionQuerySpec` recognizes the public filter vocabulary in `polylogue/archive/query/spec.py:199-244`, builds specs from parameter maps in `polylogue/archive/query/spec.py:306-361`, stores canonical selection intent plus `with_units` projection attachments in `polylogue/archive/query/spec.py:439-495`, and exposes `from_params`, `to_plan`, `list`, `list_summaries`, `count`, and `build_filter` in `polylogue/archive/query/spec.py:497-600`. `SessionQueryPlan` is the canonical immutable execution state at `polylogue/archive/query/plan.py:85-140`, and its terminal operations delegate to archive execution at `polylogue/archive/query/plan.py:258-281`. The fluent `SessionFilter` is an adapter over the same plan, not an alternate model, as shown by `polylogue/archive/filter/filters.py:1-7`, `polylogue/archive/filter/filters.py:35-72`, and `polylogue/archive/filter/filters.py:112-164`.

The archive executor is also farther along than several surfaces assume. It translates plan fields into one ArchiveStore kwarg set in `polylogue/archive/query/archive_execution.py:127-196`; it owns list/search/semantic/hybrid dispatch in `polylogue/archive/query/archive_execution.py:345-385`; it attaches `with <units>` rows after selection in `polylogue/archive/query/archive_execution.py:408-430`; it exposes list/list_summaries/count and paired search hits in `polylogue/archive/query/archive_execution.py:433-544`; and it already contains reciprocal-rank fusion for the hybrid lane in `polylogue/archive/query/archive_execution.py:538-618`.

The projection/render side exists as a contract object but is not yet an executor. `polylogue/surfaces/projection_spec.py:1-7` names the intended Query × Projection × Render algebra. It defines evidence families, body policy, render format, render destination, selection, projection, render, and the composed `QueryProjectionSpec` at `polylogue/surfaces/projection_spec.py:18-146`. Executable read views are mapped to evidence families in `polylogue/surfaces/projection_spec.py:149-161`, and `projection_from_views` composes named read views into a spec in `polylogue/surfaces/projection_spec.py:189-271`. The important caveat is in its own docstring: it is “a contract builder, not an executor” at `polylogue/surfaces/projection_spec.py:213-218`.

The query DSL is real and broad, but terminal projection parity is incomplete. The shared DSL front door is documented in `polylogue/archive/query/expression.py:1-58`; pipeline and unit parsing live around `polylogue/archive/query/expression.py:1636-2042`; terminal unit queries currently return unit rows and explicitly tell callers to use `query_units`, the API, or a CLI terminal-unit query instead of `SessionQuerySpec` at `polylogue/archive/query/expression.py:2039-2042`.

## Inventory

Legend: `Q` = query construction, `F` = filtering, `P` = projection, `A` = aggregation, `R` = rendering. “Surface-owned” means logic implemented at that surface rather than delegated. “Delegated” names the current lower layer, when one is used.

| Surface | Entry point / path | Surface-owned today | Delegated today | Classification | Anchors |
|---|---|---:|---|---|---|
| CLI | Root query mode and hidden `find` verb dispatch | Q preset parsing and verb dispatch | `RootModeRequest.query_spec()` and archive query executor | trivially-routable | `polylogue/cli/click_app.py:392-449`, `polylogue/cli/query.py:24-38`, `polylogue/cli/root_request.py:99-115`, `polylogue/cli/archive_query.py:145-161` |
| CLI | `archive_query` list/search/count stdout path | Q/F/P/A/R all mixed: `with <units>` parsing, filter kwarg extraction, daemon fast-path selection, count/stats/list/search/session payloads, stream/file output | Some calls into `SessionQuerySpec`, `ArchiveStore`, and `archive_search_hits` patterns, but not one contract path | needs-contract-extension for `with_units`/stats; otherwise trivially-routable | `polylogue/cli/archive_query.py:164-767`, `polylogue/cli/archive_query.py:781-852`, `polylogue/cli/archive_query.py:1516-1524`, `polylogue/cli/archive_query.py:1571-1656`, `polylogue/cli/archive_query.py:1765-2003`, `polylogue/cli/archive_query.py:2114-2241` |
| CLI | Inline hybrid/search fusion in `archive_query._query_hits` | Retrieval-lane dispatch and RRF ranking | `ArchiveStore.search_summaries`; vector provider direct call | trivially-routable: archive executor already has equivalent lane logic | `polylogue/cli/archive_query.py:781-852`, `polylogue/archive/query/archive_execution.py:538-618` |
| CLI | Daemon session-page fast path from root query | Q/F/R routing logic and manual query-param mapping | daemon HTTP `/api/sessions` | trivially-routable once daemon path is canonical | `polylogue/cli/archive_query.py:855-1040`, especially support exclusions at `polylogue/cli/archive_query.py:962-993` and mapping at `polylogue/cli/archive_query.py:995-1040` |
| CLI | `select` query action | P/R owns select-row shape; Q/F delegates | `request.query_spec()`, `spec.build_filter`, list_summaries/list | already-conformant for Q/F; row rendering needs `x7d` | `polylogue/cli/select.py:169-193`, `polylogue/cli/select.py:254-261` |
| CLI | `read` dispatcher and `--spec` | Q/P/R preset composition, direct-ref JSON bypass, destination dispatch | `projection_from_views`; read-view handlers; query-spec resolver | needs-contract-extension: ProjectionSpec is built but not the executor | `polylogue/cli/query_verbs.py:934-996`, `polylogue/cli/query_verbs.py:1000-1110`, `polylogue/cli/query_verbs.py:1122-1317`, `polylogue/cli/query_verbs.py:587-644` |
| CLI | `read --view summary/transcript` | Exact-file fast path and terminal formatting | `execute_query_request` for non-exact query mode | trivially-routable | `polylogue/cli/read_views/standard.py:75-107` |
| CLI | `read --view dialogue` | P/R body windowing and dialogue JSON/markdown payload | exact session via `env.polylogue.get_session` and `ContentProjectionSpec.prose_only()` | needs-contract-extension: body/window/policy should be ProjectionSpec | `polylogue/cli/read_views/standard.py:109-158`, `polylogue/cli/read_views/standard.py:165-229` |
| CLI | `read --view messages` | P/R paginated message envelope, JSON/NDJSON/plain rendering, file output envelope | `Polylogue.get_messages_paginated`, `iter_messages` | needs-contract-extension: message/raw pagination is not SelectionSpec/ProjectionSpec | `polylogue/cli/read_views/messages.py:37-172`, `polylogue/cli/messages.py:21-107` |
| CLI | `read --view raw` | P/R raw payload envelope and JSON/YAML rendering | `Polylogue.get_raw_artifacts_for_session` | needs-contract-extension: raw artifacts are an evidence family but no executor | `polylogue/cli/read_views/messages.py:175-217`, `polylogue/cli/messages.py:110-161` |
| CLI | `read --view context` and `context-image` | Context-specific option mapping and markdown/JSON rendering | `compose_context_preamble`, `env.polylogue.context_image_payload`, `compile_context` | already contract-aware but not fully conformant; executor gap | `polylogue/cli/read_views/context.py:42-88`, `polylogue/api/archive.py:2208-2604` |
| CLI | `read --view temporal` | A/P/R event aggregation and markdown/JSON rendering | Selection via `request.query_spec()`; message/action queries via store | needs-contract-extension: temporal is a projection/aggregate family | `polylogue/cli/read_views/standard.py:278-332`, `polylogue/cli/read_views/standard.py:368-446` |
| CLI | `read --view chronicle` | P/A/R edge-window payload assembly | Selection via exact ref or `request.query_spec()` | needs-contract-extension: chronicle edge windows are not a projection executor | `polylogue/cli/read_views/chronicle.py:45-121` |
| CLI | `read --view neighbors` | P/R neighbor options, JSON/plain envelope | `env.polylogue.neighbor_candidates` | needs-contract-extension: neighbor limit/window are ProjectionSpec fields but payload projection is bespoke | `polylogue/cli/read_views/neighbors.py:54-111`, `polylogue/surfaces/projection_spec.py:107-108` |
| CLI | `read --view correlation` | Very thin dispatch; output controlled in insight view | `polylogue.insights.correlation_view.run_correlation_view` | needs-contract-extension: correlation is an evidence family but not executed by the contract | `polylogue/cli/read_views/correlation.py:25-42` |
| CLI | Query-set read for `read --all` non-summary and query-set dialogue | P/R owns multi-session formatting and file delivery | `env.polylogue.list_sessions_for_spec`, `project_query_results`, `format_session` | trivially-routable after projection executor exists | `polylogue/cli/read_views/query_set.py:30-93`, `polylogue/cli/query_set_read.py:29-70` |
| CLI | Streaming markdown file export | Q/F/P/R direct raw SQL over `index.db`, ref resolution, prefix-edge check, block filtering | None meaningful; bypasses substrate read/render | trivially-routable, high priority offender | `polylogue/cli/read_views/streaming_markdown.py:17-59`, `polylogue/cli/read_views/streaming_markdown.py:62-162`, `polylogue/cli/read_views/streaming_markdown.py:185-210` |
| CLI | `analyze --count`, facets, postmortem, portfolio | A/P/R boolean mode dispatch; some query target resolution | Count routes through `_execute_query_verb`; postmortem/portfolio via API bundles | needs-contract-extension: aggregate/derived-view spec | `polylogue/cli/query_verbs.py:2005-2140`, `polylogue/api/archive.py:1825-1898`, `polylogue/api/archive.py:2009-2077` |
| Daemon HTTP | Route table and HTTP query-param normalization | Q option parsing and URL contract | `_build_query_spec_params` yields spec-compatible maps | already-conformant as parsing shell, but routes below diverge | `polylogue/daemon/http.py:228-276`, `polylogue/daemon/http.py:912-980` |
| Daemon HTTP | `/api/sessions`, normal non-split path | Q/P/R manual list envelope; F delegates | `SessionQuerySpec.from_params`, `compile_expression_into`, `spec.build_filter`, `spec.count` | trivially-routable | `polylogue/daemon/http.py:1691-1804` |
| Daemon HTTP | `/api/sessions`, search envelope path | P/R envelope assembly | `build_search_envelope_for_spec` | already-conformant for search semantics | `polylogue/daemon/http.py:1805-1832`, `polylogue/surfaces/search_envelope.py` tested by `tests/unit/surfaces/test_search_envelope_contract.py:1-12` |
| Daemon HTTP | `/api/sessions`, split-archive fast path | Q/F/A/P/R manual filter merging, DSL split, direct `ArchiveStore.search_summaries/list_summaries/count`, manual payload | Only low-level ArchiveStore | trivially-routable; explicit `4p1.1` offender | `polylogue/daemon/http.py:1834-2158`; mirror-comment at `polylogue/daemon/http.py:1900-1903`; search/list/count at `polylogue/daemon/http.py:2038-2048`, `polylogue/daemon/http.py:2128-2149` |
| Daemon HTTP | `/api/sessions/:id` detail | P/R manual session detail and messages | `poly.get_session`; split path uses `ArchiveStore.read_summary/read_session` | trivially-routable after exact-session projection executor | `polylogue/daemon/http.py:2213-2357`, `polylogue/daemon/http.py:2371-2391` |
| Daemon HTTP | `/api/sessions/:id/messages` | P/R manual paginated message envelope | `poly.get_messages_paginated`; split path loads full session and slices | needs-contract-extension for message paging | `polylogue/daemon/http.py:3111-3182` |
| Daemon HTTP | `/api/sessions/:id/read` | P/R view switch, HTTP envelope wrapping | messages/raw/context/context-image/neighbors/correlation helpers | needs-contract-extension: named read view should be a spec preset | `polylogue/daemon/http.py:2942-3068` |
| Daemon HTTP | `/api/sessions/:id/raw`, `/api/raw_artifacts/:id` | P/R raw envelope | `poly.get_raw_artifacts_for_session` | needs-contract-extension | `polylogue/daemon/http.py:2398-2425`, `polylogue/daemon/http.py:3255-3269` |
| Daemon HTTP | `/api/query-units` | Q/P/R terminal-unit request and envelope | `query_unit_envelope` | needs-contract-extension: unit rows are outside `SessionQueryPlan` | `polylogue/daemon/http.py:2705-2760` |
| Daemon HTTP | `/api/facets` | A/P/R HTTP shape and deferred-family choice | `SessionQuerySpec.from_params`, `poly.facets(spec)` | needs-contract-extension: AggregateSpec/FacetSpec | `polylogue/daemon/http.py:3271-3298`, `polylogue/api/archive.py:3723-3819` |
| Daemon HTTP | `/api/read-view-profiles`, `/api/query-completions`, `/api/refs/resolve`, `/api/action-affordances` | R only for HTTP envelopes | shared registries/facade | already-conformant support surfaces | `polylogue/daemon/http.py:2818-2864`, `polylogue/daemon/http.py:2866-2879`, `polylogue/archive/viewport/profiles.py:1-6`, `polylogue/archive/query/completions.py` |
| Daemon HTTP | Similarity, cost, provenance, topology, parent-chain, attachments, insights, stack/compare | Mostly P/A/R panel envelopes and delivery; some direct ArchiveStore in split workspace | API/facade/insight/topology/provenance helpers; similarity still duplicates vec KNN | genuinely-special for panel envelopes, with contract extensions for evidence families; similarity trivially-routable | routes at `polylogue/daemon/http.py:257-276`; attachments at `polylogue/daemon/http.py:1484-1579`; cost at `polylogue/daemon/http.py:2432-2454`; insights at `polylogue/daemon/http.py:2461-2571`; provenance/topology at `polylogue/daemon/http.py:2577-2668`; similar at `polylogue/daemon/http.py:3075-3104`; stack/compare at `polylogue/daemon/http.py:3189-3248`; duplicate KNN at `polylogue/daemon/similarity.py:172-230` |
| Daemon web shell | Browser reader JS and DOM renderers | R and client navigation state; query-param construction | daemon HTTP API | genuinely-special delivery shell; must consume canonical server payloads | `polylogue/daemon/web_shell.py:650-715`, `polylogue/daemon/web_shell.py:751-770`, `polylogue/daemon/web_shell.py:1250-1490`; tests at `tests/unit/daemon/test_web_reader.py:663-830`, `tests/visual/test_reader_dom_smoke.py:534-615` |
| MCP | `MCPSessionQueryRequest` and `build_query_spec` | Q normalizer/aliases | `SessionQuerySpec.from_params`, `compile_expression_into` | already-conformant query construction | `polylogue/mcp/query_contracts.py:24-63`, `polylogue/mcp/query_contracts.py:66-140` |
| MCP | `search` and `list_sessions` tools | P/R MCP envelopes, cursor hack | spec build; `archive_search_payload`/`archive_session_list_payload` | trivially-routable after shared execution result | `polylogue/mcp/server_tools.py:128-181`, `polylogue/mcp/server_tools.py:284-305`, `polylogue/mcp/archive_support.py:254-379` |
| MCP | `query_units` tool | Q/P/R terminal-unit validation and payload | `archive_query_unit_payload`, query-unit envelope | needs-contract-extension | `polylogue/mcp/server_tools.py:183-271`, `polylogue/mcp/archive_support.py:382-401` |
| MCP | Legacy `archive_list_sessions` / `archive_search_sessions` | Q/P/A/R wrapper semantics and totals | `poly.archive_list_sessions`, `poly.archive_search_sessions`, count methods | trivially-routable; duplicate list semantics | `polylogue/mcp/server_tools.py:362-563`, `polylogue/api/archive.py:3399-3606` |
| MCP | `get_session`, `get_messages`, `raw_artifacts` | P/R MCP detail/message/raw envelopes | direct `ArchiveStore` or API raw helper | needs-contract-extension for exact-session/message/raw read | `polylogue/mcp/server_tools.py:346-360`, `polylogue/mcp/server_tools.py:853-933`, `polylogue/mcp/archive_support.py:404-455` |
| MCP | `compile_context`, `build_context_image`, `compose_context_preamble` | P/R context envelopes; preamble git/session-start enrichment | `Polylogue.compile_context`, `context_image_payload`, `context_preamble_payload` | context-image is needs-contract-extension; preamble enrichment genuinely-special | `polylogue/mcp/server_tools.py:307-344`, `polylogue/mcp/server_context_tools.py:33-162`, `polylogue/api/archive.py:2271-2604` |
| MCP | `facets`, `stats`, `get_stats_by`, insight tools, topology/logical/tree/neighbors/assertions/profiles | A/P/R tool-specific envelopes | API/facade or insight helpers; some ArchiveStore.stats direct | needs-contract-extension for aggregates/derived views; some genuinely-special panels | `polylogue/mcp/server_tools.py:606-690`, `polylogue/mcp/server_tools.py:737-851`, `polylogue/mcp/server_tools.py:1004-1043`, `polylogue/mcp/server_insight_tools.py:65-117`, `polylogue/mcp/server_insight_tools.py:510-557`, `polylogue/mcp/server_insight_tools.py:1099-1178` |
| Python API | `list_sessions_for_spec` and `search_session_hits` | none beyond facade method | `spec.list()` and `search_hits_for_plan(spec.to_plan())` | already-conformant | `polylogue/api/archive.py:3200-3241` |
| Python API | Legacy session query/search/count/list endpoints | Q/F/A/P helpers duplicate spec-to-ArchiveStore mapping | Some build `SessionQuerySpec`; direct `_archive_*` helpers otherwise | trivially-routable | `polylogue/api/archive.py:520-630`, `polylogue/api/archive.py:3288-3606`, `polylogue/api/archive.py:4037-4125` |
| Python API | Exact session, messages, raw, actions | P/R detail/material-origin filtering and pagination | `ArchiveStore`/repository helpers; `ContentProjectionSpec` for session | needs-contract-extension | `polylogue/api/archive.py:1757-1773`, `polylogue/api/archive.py:3909-4035`, `polylogue/api/archive.py:3119-3155` |
| Python API | Context image, context preamble, query units, resolve ref, read-view profiles | P/R service payloads | `compile_context`, context/query-unit registries | already contract-aware, executor gap remains | `polylogue/api/archive.py:2208-2604`, `polylogue/api/archive.py:2512-2516`, `polylogue/api/archive.py:2606-2750` |
| Python API | Bundles, facets, stats, neighbors/correlation | A/P/R derived views and panels | API/insights/facade helpers; direct stats methods | needs-contract-extension: AggregateSpec/DerivedViewSpec | `polylogue/api/archive.py:1825-1898`, `polylogue/api/archive.py:2009-2077`, `polylogue/api/archive.py:3254-3263`, `polylogue/api/archive.py:3702-3819`, `polylogue/api/archive.py:4195-4251` |
| Rendering | `format_session` and renderer modules | R canonical session render dispatch | projection content via `ContentProjectionSpec`; block renderers | already-conformant render substrate, but not wired to QueryProjectionSpec | `polylogue/rendering/formatting.py:1-12`, `polylogue/rendering/formatting.py:49-84`, `polylogue/rendering/renderers/html.py:22-61`, `polylogue/rendering/core_markdown.py:80-125`, `polylogue/rendering/blocks.py:16-32` |
| Rendering / web / CLI | Session-to-HTML paths | R duplicated across canonical rendering, web shell JS, CLI read/export path | Mixed | trivially-routable, tracked by `polylogue-7le` | `polylogue/rendering/formatting.py:130-139`, `polylogue/rendering/renderers/html.py:22-61`, `polylogue/daemon/web_shell.py:1250-1490`, `polylogue/cli/read_view_handlers.py:52-149` |
| Generated docs / product contracts | Read-view profiles, CLI schemas, OpenAPI/product workflows | R/docs generation and declared presets | profile/workflow registries | already-conformant contract registry, should become gate | `polylogue/archive/viewport/profiles.py:1-6`, `polylogue/archive/viewport/profiles.py:39-76`, `polylogue/archive/viewport/profiles.py:77-161`, `polylogue/archive/viewport/profiles.py:238-332`, `polylogue/product/workflows.py:103-230`, `docs/devtools.md:90-105` |

## Contract gaps and decisions

1. Selection execution exists, but surfaces still remap filters. The same fields are mapped in `archive_execution._plan_filter_kwargs`, API `_archive_query_kwargs`, daemon split-archive, MCP `archive_query_filters`, and CLI `archive_query`. Recommendation: delete surface-side mappings and expose one `execute_session_query(spec, *, mode)` result object from archive/query or the API facade. Keep the current spec vocabulary; do not add a second filter model.

2. `QueryProjectionSpec` is declarative only. The CLI, daemon, MCP, and API all build or expose projection specs, but they still call bespoke handlers for message paging, raw artifacts, chronicle, temporal, neighbors, correlation, and context. Recommendation: add a `read_projection` executor below surfaces. The executor takes `SessionQuerySpec | exact refs`, `ProjectionSpec`, and `RenderSpec`, then returns typed payload(s) before rendering. Do not make daemon/MCP import CLI read-view handlers.

3. `with <units>` lives in `SessionQuerySpec` but is deliberately absent from `SessionQueryPlan`. The current archive executor attaches units after list/list_summaries, while terminal unit queries use a separate query-unit envelope. Recommendation: extend `ProjectionSpec` with `attached_units` and `unit_fields`, and reserve terminal unit pipelines for a `UnitProjectionSpec` that can be rendered by the same executor. Preserve existing `with messages/actions/files` syntax; do not delete the capability.

4. Messages/raw paging is outside the algebra. `ProjectionSpec` has `body_limit`/`body_offset`, but session message endpoints use `limit`, `offset`, `full`, `tail`, `material_origin`, `message_type`, and raw-artifact envelopes. Recommendation: add `MessageWindowSpec` or formal message-window fields to `ProjectionSpec`, including `tail`, `material_origin`, `message_type`, and block/body policy. Keep raw artifacts as `EvidenceFamily.RAW` with a bounded preview policy; do not route raw through generic session JSON.

5. Body policy is split between `ProjectionSpec` and `ContentProjectionSpec`. CLI dialogue, `format_session`, context, and transcript rendering use `ContentProjectionSpec.prose_only()` or ad hoc block exclusions. Recommendation: make `ProjectionSpec.body_policy` compile to `ContentProjectionSpec` inside the executor, then retire direct surface usage except in substrate renderers.

6. Render format and destination are duplicated. `RenderSpec` supports markdown/json/ndjson/html/obsidian/org/yaml/plaintext/csv and terminal/stdout/browser/clipboard/file, while CLI query output and read views still carry `--json`, `--format`, `--to`, `--out`, and bespoke file output rules. Recommendation: keep aliases as compatibility presets, but route all dialect/destination choices through `RenderSpec`. This is `jnj.3` plus the read executor.

7. Aggregate reads are not first-class. Counts, stats, facets, `stats_by`, postmortem, portfolio, topology panels, and insights are not expressible as `ProjectionSpec` today. Recommendation: add `AggregateSpec` and `DerivedViewSpec` under the Projection axis rather than overloading `EvidenceFamily`. Facets/stats remain named aggregate presets; expensive panels declare materialization policy. This aligns with `jnj.2`, `fnm.1`, and `5wp`.

8. Retrieval-lane vocabulary is internally inconsistent. The boundary vocabulary explicitly excludes `semantic` (`QUERY_RETRIEVAL_LANES = auto/dialogue/actions/hybrid`), but archive execution and tests still talk about a resolved `semantic` lane, and CLI carries a semantic alias through `similar_text`. Recommendation: preserve user-facing `similar_text` and the legacy CLI `--semantic` alias, but keep `semantic` out of the public `retrieval_lane` enum unless product policy changes. Make “resolved lane” a response field, not an input field.

9. Direct object refs bypass read-view semantics. CLI `read session:...` currently resolves and prints a JSON ref envelope only, while `find id:... then read --view messages` follows read-view handlers. Recommendation: implement `jnj.4`: exact session refs become `SelectionSpec.refs`, message/block/assertion refs stay on the resolver surface unless a projection family is selected.

10. Daemon web shell rendering is a delivery shell, not a contract exception. Its DOM state, route-state chips, and interactive panels are genuinely web-specific, but HTML/message block rendering should come from the same rendering substrate. Recommendation: keep the SPA as a consumer of canonical payloads and renderer fragments; consolidate session/block HTML under `rendering/` (`7le`, `ap7`).

11. Support matrix is missing. The DSL has registries and pipeline stages, but the product does not generate a matrix of which clauses, units, and terminal stages are legal. Recommendation: implement `fnm.11` before extending unit projections further; make generated docs fail when a registered clause lacks matrix coverage.

## Ordered collapse plan

### PR 1 — Record the one-read law and land this inventory (`polylogue-4p1`)

Add a doctrine entry to `docs/architecture-spine.md` and a checked inventory manifest under `docs/plans/one-read-contract-cut.*`. The manifest should use the four classifications in this report and include the owner bead for every non-conformant row. This PR changes no behavior.

Verification: `tests/unit/surfaces/test_projection_spec.py`, `tests/unit/daemon/test_route_contracts.py`, `tests/unit/product/test_query_action_workflows.py`, `tests/unit/devtools/test_generated_surfaces.py`, and `tests/unit/architecture/test_surface_storage_boundary.py`. The architecture boundary already asserts public surfaces route through operations/facade rather than storage in `tests/unit/architecture/test_surface_storage_boundary.py:1-8` and fails on direct forbidden imports at `tests/unit/architecture/test_surface_storage_boundary.py:73-86`.

### PR 2 — Route daemon split-archive `/api/sessions` through `SessionQuerySpec` (`polylogue-4p1.1`)

Keep the split archive fast path and the response envelope, but replace manual filter merging with `SessionQuerySpec.from_params` plus `compile_expression_into` and the canonical plan/filter kwarg translator. The path can still use direct ArchiveStore opening for performance; the rule is that filter semantics come from the spec, not the route.

Verification: extend daemon tests to enumerate every spec filter field; run `tests/unit/daemon/test_web_reader.py`, `tests/unit/daemon/test_route_contracts.py`, `tests/unit/test_cross_surface_agreement.py`, and `tests/unit/test_read_surface_coherence.py`. The cross-surface test already asserts repository, facade, CLI, MCP, and daemon HTTP agree on query IDs at `tests/unit/test_cross_surface_agreement.py:155-203`.

### PR 3 — Create one archive query execution result and migrate list/search/count surfaces (`polylogue-t46.3`)

Add a shared executor that returns `{items, hits, total, next_cursor, resolved_lane, query_plan}` from a `SessionQuerySpec` and a mode. Migrate API legacy `_archive_*` helpers first, then daemon split/normal session lists, then MCP `archive_support`, then CLI `archive_query`. Keep old method names as wrappers until parity is green.

Verification: `tests/infra/surfaces.py` already provides Repository, Facade, CLI, MCP, and Daemon adapters (`tests/infra/surfaces.py:63-77`, `tests/infra/surfaces.py:147-175`, `tests/infra/surfaces.py:557-578`). Run `tests/unit/test_cross_surface_agreement.py`, `tests/unit/surfaces/test_search_envelope_contract.py`, `tests/unit/surfaces/test_search_cursor_pagination.py`, `tests/unit/mcp/test_tool_contracts.py`, and daemon web-reader session-list tests.

### PR 4 — Normalize retrieval lanes and delete CLI local hybrid fusion (`polylogue-t46.3` slice)

Move CLI semantic alias handling into the query-spec builder or a compatibility normalizer. Remove `_query_hits` local RRF and call the shared execution result. Preserve output shape and response `surface/resolved_lane` fields.

Verification: `tests/unit/test_read_surface_coherence.py:216-274` for retrieval-lane validation, `tests/unit/surfaces/test_search_cursor_pagination.py:261-272` for RRF ordering semantics, and MCP search evidence tests at `tests/unit/mcp/test_tool_contracts.py:563-592`.

### PR 5 — Unify root query row projection (`polylogue-x7d`)

Introduce a row value object for session list/search/select rows: title budget, snippet budget, normalized single-line text, ids, score/rank, refs, and stable JSON keys. Migrate `archive_query`, `query_output`, and `select` to this helper.

Verification: add a golden that renders the same seeded rows through root query JSON, table, and `select`. Preserve generated CLI output schemas. Existing row-shape pressure comes from `tests/infra/surfaces.py:343-365`, which already normalizes multiple historical result shapes.

### PR 6 — Delete CLI `streaming_markdown.py` raw SQL path (`polylogue-t46.5`)

Expose a substrate iterator for transcript/dialogue markdown file export. It may stream messages to keep memory bounded, but it must read via API/archive execution and render through `rendering/core_markdown.py` and `rendering/blocks.py`. Delete direct SQL helpers.

Verification: `tests/unit/cli/test_streaming_markdown_read_view.py:62-119`, plus a new forked/prefix-sharing fixture comparing file export bytes to stdout/substrate transcript bytes. Run rendering snapshots.

### PR 7 — Make `ProjectionSpec` executable for exact-session read views (`polylogue-4p1.2`, new)

Add the first `read_projection` executor for exact session refs and the core families `sessions`, `messages`, `blocks`, and `raw`. Route CLI `read --view summary/transcript/dialogue/messages/raw`, daemon `/api/sessions/:id/read`, daemon `/messages`, MCP `get_session/get_messages/raw_artifacts`, and API exact-session/message/raw methods through it. Keep legacy envelopes as format adapters.

Verification: `tests/unit/surfaces/test_projection_spec.py`, `tests/unit/daemon/test_web_reader.py:1052-1163`, `tests/unit/mcp/test_tool_contracts.py:795-980`, CLI messages/raw tests, and `tests/visual/test_reader_dom_smoke.py:534-615`.

### PR 8 — Collapse per-view flags into ProjectionSpec/RenderSpec (`polylogue-jnj.1`)

Represent neighbor/correlation/context options in ProjectionSpec/RenderSpec, not in CLI option bags. Compile `ProjectionSpec.body_policy` to `ContentProjectionSpec`. Keep legacy flags as aliases that fill the spec.

Verification: `tests/unit/surfaces/test_projection_spec.py:21-157`, read-view profile payload tests, daemon route contracts, MCP `list_read_view_profiles` tests at `tests/unit/mcp/test_tool_contracts.py:1049-1068`, and generated CLI reference/OpenAPI checks.

### PR 9 — Generate the DSL support matrix and close unit clause parity (`polylogue-fnm.11`)

Generate a matrix from expression registries and enforce it in docs/devtools. Add sessions `| count`/`| group-by` and date clauses inside unit `where` expressions. Improve errors with unit/stage context.

Verification: parser tests, generated support-matrix diff, query-unit CLI/API/MCP tests, and `tests/unit/archive/test_with_units_projection.py:39-132`.

### PR 10 — Promote `with_units` and terminal unit rows into ProjectionSpec (`polylogue-4p1.3`, new; coordinates with `fnm.2` and `fnm.6`)

Add `ProjectionSpec.attached_units`, `unit_fields`, and terminal `UnitProjectionSpec`. Route CLI `with <units>`, API `query_units`, daemon `/api/query-units`, and MCP `query_units` through one executor. Keep terminal-unit output envelopes intact while sharing selection/projection.

Verification: `tests/unit/archive/test_with_units_projection.py:198-326`, daemon query-unit route tests, MCP query-unit tests, API `query_units`, and generated support matrix.

### PR 11 — Add AggregateSpec/DerivedViewSpec for facets/stats/bundles (`polylogue-4p1.4`, new; coordinates with `t46.6`, `fnm.1`, `jnj.2`, `5wp`)

Delete dead CLI stats aggregators, route origin/date/tool/work-kind grouping through API `stats_by`, and define named aggregate presets for count, facets, stats, stats_by, postmortem, portfolio, pathologies, and insights. Expensive derived panels declare materialization policy instead of pretending to be simple projections.

Verification: `tests/unit/daemon/test_web_reader.py:762-830` for facets, `tests/unit/mcp/test_facets_tool_contract.py:15-50`, API stats/facets tests, and new aggregate parity tests across CLI/daemon/MCP/API.

### PR 12 — Delegate daemon similarity KNN (`polylogue-t46.4`)

Replace `polylogue/daemon/similarity.py` direct sqlite-vec KNN and aggregation with `SqliteVecProvider.query_by_session` or a substrate rollup added below surfaces. The daemon keeps disabled/unavailable/not_embedded status envelopes only.

Verification: seeded session-similarity parity between `/api/sessions/:id/similar` and provider output; grep confirms `_knn_for_embedding`, `_aggregate_hits`, and `_l2_to_cosine_similarity` are gone.

### PR 13 — Normalize render dialect/destination and HTML paths (`polylogue-jnj.3`, `polylogue-7le`, `polylogue-ap7`, `polylogue-1lm`)

Make `RenderSpec` the source of truth for `--format`, `--json`, `--to`, `--out`, layout, and timestamp policy. Consolidate block/message HTML under `rendering/`; web shell consumes canonical payloads/fragments. Add semantic transcript renderer registry after HTML consolidation.

Verification: `tests/unit/rendering/test_output_snapshots.py`, `tests/unit/architecture/test_static_rendering_contracts.py:10-13`, daemon visual smoke tests, generated schemas, and one before/after semantic transcript demo fixture.

### PR 14 — Collapse MCP tool fanout after goldens (`polylogue-t46.8`)

Only after the shared executors exist, turn the many MCP read tools into declared presets/resources over the same algebra. Keep old tool names as wrappers for one release cycle, with per-tool equivalence goldens before deletion.

Verification: `tests/unit/mcp/test_tool_contracts.py`, `tests/integration/test_mcp.py`, generated MCP tool schemas, and a wrapper-to-preset parity suite.

## Risk register

| Risk | Where it appears | Compatibility decision |
|---|---|---|
| JSON envelopes differ by surface | CLI list/search envelopes, daemon `/api/sessions`, MCP paginated payloads, API typed results | Keep current public envelopes as adapters around one execution result until a versioned envelope migration exists. Do not force SearchEnvelope everywhere in one PR. |
| Totals differ or are unknown | CLI search envelopes historically allow unknown total; MCP list has estimate vs count; daemon split path computes count separately | Define `total_kind = exact | estimated | unknown` internally. Preserve public fields and add optional metadata only where schemas allow. |
| Cursor semantics drift | CLI `_build_cursor`, MCP offset/cursor hack, SearchCursor tests | Use shared cursor object from archive execution. Keep opaque cursor format pinned by `tests/unit/surfaces/test_search_cursor_pagination.py`. |
| Retrieval lane naming breaks users | Public lanes exclude `semantic`; resolved execution can say semantic; CLI has semantic alias | Keep alias input outside `retrieval_lane`; return resolved lane separately. Reject `retrieval_lane=semantic` unless a new product decision changes the enum. |
| Hybrid ranking changes | CLI local RRF vs archive executor RRF | Compare seeded hybrid result order before deletion. Archive executor becomes source of truth. |
| Direct `read session:REF` changes shape | Current direct ref prints resolver JSON only | Implement as compatibility flag first: default view remains summary JSON for direct refs unless explicit `--view`; then migrate to `jnj.4` with golden warnings. |
| Streaming markdown byte changes | Raw SQL path omits prefix-sharing and filters blocks differently | Add byte goldens for normal and forked sessions. Treat forked-session fix as intentional improvement; keep non-forked bytes stable. |
| HTML output changes | rendering HTML, CLI HTML, web shell JS render | Decide canonical export HTML vs interactive SPA rendering. The SPA can differ structurally, but block rendering semantics must match snapshot fixtures. |
| Raw/provenance privacy regression | raw artifacts, provenance previews, import explain, web routes | Keep raw preview bounded and opt-in. Route through `EvidenceFamily.RAW` only with explicit raw projection. Preserve visual smoke privacy assertions. |
| Facets/aggregates become slower or stale | daemon facets, stats, insights panels | AggregateSpec must carry materialization policy and `staleness` metadata for derived views. Do not recompute expensive cross-session panels synchronously without a budget. |
| Context-image omissions change | `compile_context` omits unsupported views and emits caveats | Projection executor must preserve `omitted` and `caveats` arrays; adding support for a formerly omitted view is a versioned improvement. |
| Surface layering violation during migration | Temptation to import CLI handlers into daemon/MCP | Move logic down. Guard with `tests/unit/architecture/test_surface_storage_boundary.py` and add a sibling rule forbidding cross-surface read handler imports. |

## Proposed bead-field updates, append-only

### `polylogue-4p1` — Decision: one read algebra

Append to Design:

```text
Contract cut inventory delivered from the source package: docs/plans/one-read-contract-cut.md (or equivalent manifest). The current realization is partial: SessionQuerySpec/SessionQueryPlan/archive_execution are executable; QueryProjectionSpec exists in polylogue/surfaces/projection_spec.py but is still a builder, not the read executor. The doctrine must therefore name two gates: (1) no new surface-side selection/filter mapping, and (2) no new read view without either an existing ProjectionSpec/RenderSpec value or a bead adding that value below surfaces.
```

Append to Acceptance Criteria:

```text
Inventory rows classify CLI, daemon HTTP/web shell, MCP, Python API, rendering, and generated docs as already-conformant / trivially-routable / needs-contract-extension / genuinely-special, with file:line anchors. Every needs-contract-extension row links to jnj/fnm/t46 or a new 4p1.x bead. Devtools or architecture tests verify that surfaces do not import other surface read handlers to share logic.
```

### `polylogue-4p1.1` — Route daemon split-archive fast path through `SessionQuerySpec.from_params`

Append to Design:

```text
Use _build_query_spec_params only as HTTP parsing. After parsing, construct SessionQuerySpec.from_params and compile_expression_into exactly like the non-split path. The split path may still open ArchiveStore directly for performance, but the SQL-pushable kwarg set must come from the canonical plan translator in archive/query/archive_execution.py rather than a daemon-local mirror.
```

Append to Acceptance Criteria:

```text
A table-driven test iterates every SessionQuerySpec filter field recognized by spec.py and proves /api/sessions split archive returns the same ids and total as the normal facade path. The manual “must mirror public params” block is gone. Cursor/limit/sample/has_* fields have explicit parity cases.
```

### `polylogue-t46.3` — Unify list/search query-spec execution

Append to Design:

```text
Create one execution result object for list/search/count over SessionQuerySpec: rows/hits, exact-or-estimated total, next cursor, resolved lane, and plan metadata. API legacy archive_* methods, daemon /api/sessions, MCP archive_support, and CLI archive_query become adapters around that result. Keep public response envelopes unchanged during migration.
```

Append to Acceptance Criteria:

```text
Cross-surface fixture covers root CLI find/read summary, MCP search/list_sessions/archive_* wrappers, daemon /api/sessions split and non-split, and Python API query_sessions/count_sessions for identical filters. Grep shows no surface-local SessionQuerySpec->ArchiveStore kwarg mapper outside the shared executor.
```

### `polylogue-t46.4` — Delegate daemon session-similarity KNN

Append to Design:

```text
The daemon keeps only HTTP/status projection: disabled, unavailable, not_embedded, ready. All vector KNN, per-message fanout, session aggregation, and score conversion live in SqliteVecProvider or another storage/search provider helper. If matched-message count is a daemon requirement, add it to the provider return type rather than recomputing in daemon.
```

Append to Acceptance Criteria:

```text
Seeded archive test compares /api/sessions/{id}/similar ranking, score order, and matched-message count to provider query_by_session output. _knn_for_embedding, _archive_knn_for_embedding, _aggregate_hits, and _l2_to_cosine_similarity are deleted from daemon/similarity.py.
```

### `polylogue-t46.5` — Delete streaming markdown SQL path

Append to Design:

```text
Replace cli/read_views/streaming_markdown.py with a streaming renderer over substrate session/message iteration. The renderer must use the same block renderer and body policy compiler as normal transcript/dialogue output, while still writing incrementally for large sessions.
```

Append to Acceptance Criteria:

```text
For a forked/resumed prefix-sharing session, read --view transcript --to file and stdout/substrate transcript are byte-identical after normalizing destination-only headers. Non-forked current goldens remain stable. The file contains no sqlite3 SQL over sessions/messages/blocks.
```

### `polylogue-t46.6` — Fix referenced_path divergence and delete dead CLI stats aggregators

Append to Design:

```text
Treat this as the first AggregateSpec cleanup slice. referenced_path matching semantics come only from the canonical query plan/runtime matching. CLI stats boolean/semantic grouping functions are removed or converted to wrappers around API stats_by/facets. No aggregate query should select a different session set than the base query.
```

Append to Acceptance Criteria:

```text
Two-term referenced_path parity test covers CLI semantic-stats, API query, and archive plan execution. query_stats/query_semantic dead grouping helpers have no callers. stats_by/facets output goldens cover origin/date/tool/work-kind buckets.
```

### `polylogue-jnj.1` — Collapse read per-view flags into ProjectionSpec/RenderSpec

Append to Design:

```text
ProjectionSpec is not enough until it executes. Sequence this bead after the exact-session read_projection executor lands. First migrate neighbor/context/correlation/temporal/chronicle option bags into ProjectionSpec fields or named preset metadata. Then make legacy CLI flags aliases that fill the spec. RenderSpec owns format/layout/destination; ReadViewInvocation owns only delivery-time plumbing until it can be deleted.
```

Append to Acceptance Criteria:

```text
read --spec for every executable read view serializes all effective options, including neighbor window/limit, context selector, body offset/limit, timestamp policy, render layout, destination, and output path. The same spec can be executed through CLI and daemon /api/sessions/{id}/read with equivalent payloads before rendering.
```

### `polylogue-x7d` — Unify root query row rendering contracts

Append to Design:

```text
The row helper consumes the shared query execution result, not ArchiveStore rows directly. It returns both machine fields and display fields so CLI table, CLI JSON, select, daemon rows, and MCP summaries can share truncation and normalization without sharing envelope schemas.
```

Append to Acceptance Criteria:

```text
Seeded overlong title/snippet fixture proves identical truncation budgets across archive_query, query_output, select, and daemon/MCP row adapters where those fields appear. Generated JSON schemas remain backward-compatible.
```

### `polylogue-fnm.11` — Pipeline/clause parity and support matrix

Append to Design:

```text
Generate the support matrix from the same registries that parse/lower the DSL. Mark each clause by unit, stage, and terminal projection status. The matrix should distinguish selection clauses, aggregate stages, attached-unit projections, and terminal UnitProjectionSpec rows.
```

Append to Acceptance Criteria:

```text
Generated docs include sessions count/group-by, date clauses in unit where, and terminal-unit projection support. Unsupported combinations fail with unit/stage/caret diagnostics and nearest alternatives. Query unit parity tests cover CLI, API, daemon, and MCP.
```

### `polylogue-t46.8` — MCP surface collapse

Append to Design:

```text
Do not delete named MCP tools until the shared query/projection/aggregate executors exist. First define each read tool as a named preset over QuerySpec x ProjectionSpec x RenderSpec or AggregateSpec/DerivedViewSpec. Old tools become wrappers that record parity goldens against the preset.
```

Append to Acceptance Criteria:

```text
Every MCP read tool has a preset declaration or a genuinely-special exemption. Wrapper and preset outputs are byte-equivalent for seeded positive and degraded cases. Tool schema generation marks deprecated wrappers only after parity is green.
```

### `polylogue-jnj.2` — Analyze boolean modes become named projections

Append to Design:

```text
Introduce AggregateSpec for count/stats/facets/stats_by and DerivedViewSpec for postmortem/portfolio/pathologies/insights. Analyze modes are named presets over those specs, not bespoke boolean branches in CLI.
```

Append to Acceptance Criteria:

```text
CLI analyze --count/--facets, daemon /api/facets, MCP facets, and Python API facets/count share one aggregate executor. Existing response envelopes remain stable as adapters.
```

### `polylogue-jnj.3` — Output dialect normalization

Append to Design:

```text
RenderSpec owns format, destination, layout, timestamp policy, and output path. --json is an alias for --format json; --to/--out affect destination only. Query output and read output both lower to RenderSpec before rendering.
```

Append to Acceptance Criteria:

```text
Each public read/query verb can print its effective RenderSpec. format x destination tests cover terminal/stdout/file/browser for markdown/json/ndjson/html where supported.
```

### `polylogue-jnj.4` — Direct `read session:REF` uses read-view semantics

Append to Design:

```text
Represent direct session refs as SelectionSpec.refs and execute the requested read-view projection. Keep raw resolver JSON as an explicit resolver mode for message/block/assertion refs and for compatibility flags.
```

Append to Acceptance Criteria:

```text
read session:X --view messages and find id:X then read --view messages produce byte-identical payloads before destination rendering. read session:X --format json no longer bypasses read-view profiles unless --resolve-ref is explicitly selected.
```

### `polylogue-7le` — Consolidate session->HTML paths

Append to Design:

```text
Canonical block/message/session HTML rendering lives in rendering/. The web shell may wrap canonical fragments in interactive DOM, but it does not own block-type rendering semantics. CLI HTML export and daemon/web HTML previews consume RenderSpec(format=html).
```

Append to Acceptance Criteria:

```text
No duplicated block-type HTML dispatch remains outside rendering/. CLI HTML export and web-shell HTML fragment snapshots match canonical renderer output for seeded message/block types.
```

### `polylogue-ap7` — Semantic transcript rendering

Append to Design:

```text
Build the semantic renderer registry below surfaces and feed it from ProjectionSpec body/block selections. The registry must serve both CLI and web; providers supply normalized tool/block metadata, not surface-specific render hints.
```

Append to Acceptance Criteria:

```text
Edit, Bash, Task, WebFetch/WebSearch, MCP tools, and unknown tools have seeded CLI and web snapshots. Unknown tools preserve current fallback rendering.
```

### `polylogue-1lm` — Composable transcript views

Append to Design:

```text
Selector x transform x budget is a deepening of ProjectionSpec, not a new transcript-only path. Reuse the DSL predicate grammar for selectors, compile adjacency selectors in the read_projection executor, and make named presets visible in read-view profiles and completions.
```

Append to Acceptance Criteria:

```text
prose/dialogue/skeleton/decisions/forensic/reboot/compact-export presets execute through the same read_projection executor in CLI, daemon, MCP, and API.
```

### `polylogue-5wp` — Insights as declared derived views

Append to Design:

```text
DerivedViewSpec registry records compute path, materialization policy, staleness key, and consumers. Session insight panels and list hot rows become declared derived views consumed by the read algebra; write-effects maintain materialized policies.
```

Append to Acceptance Criteria:

```text
Every current insight table/panel is classified as query-time/materialized/hybrid. At least one panel routes through DerivedViewSpec with parity snapshots, and list surfaces consume the new session_stats hot row rather than recomputing.
```

### New bead proposal: `polylogue-4p1.2` — Execute exact-session read projections

```text
Title: Execute exact-session read projections through QueryProjectionSpec
Labels: area:surface, area:query, delivery:C-read-evidence-contract, lane:read-contracts, refactor
Description: QueryProjectionSpec currently serializes read intent but does not execute it. Exact-session read views still fan out through CLI handlers, daemon handlers, MCP direct ArchiveStore reads, and API helper methods. Add a read_projection executor for sessions/messages/blocks/raw that takes SelectionSpec.refs or a bounded SessionQuerySpec plus ProjectionSpec/RenderSpec and returns typed payloads before surface envelope adaptation.
Acceptance: CLI summary/transcript/dialogue/messages/raw, daemon /api/sessions/{id}/read and /messages, MCP get_session/get_messages/raw_artifacts, and Python API exact-session/message/raw helpers can all be routed through the executor behind compatibility envelopes. Parity tests prove id order, body policy, message pagination, and raw artifact shape remain stable.
```

### New bead proposal: `polylogue-4p1.3` — Attach and render query units through ProjectionSpec

```text
Title: Promote with-units and terminal unit rows into the Projection axis
Labels: area:query, area:surface, delivery:C-read-evidence-contract, lane:read-contracts
Description: with <units> is represented on SessionQuerySpec but not SessionQueryPlan, while terminal unit queries use separate query-unit APIs. Add attached_units/unit_fields to ProjectionSpec and a UnitProjectionSpec for terminal unit rows so CLI, daemon, API, and MCP query-unit output shares one executor.
Acceptance: with messages/actions/files, field allowlists, and terminal unit queries produce identical rows across CLI/API/daemon/MCP. Existing query-unit envelopes remain stable as adapters. Support matrix documents every unit projection form.
```

### New bead proposal: `polylogue-4p1.4` — AggregateSpec and DerivedViewSpec

```text
Title: Make counts, facets, stats, bundles, and insights declared aggregate/derived reads
Labels: area:query, area:insights, area:surface, delivery:C-read-evidence-contract, lane:read-contracts
Description: Aggregates and panels currently live as CLI analyze branches, daemon/MCP/API endpoints, and insight helpers. Add AggregateSpec for count/facets/stats/stats_by and DerivedViewSpec for postmortem/portfolio/pathologies/topology/cost/insight panels with explicit materialization policy and staleness metadata.
Acceptance: CLI analyze, daemon facets, MCP facets/stats, Python API count/facets/stats, and one insight panel share an aggregate/derived executor. Public envelopes are compatibility adapters. Expensive derived views declare materialization policy.
```

### New bead proposal: `polylogue-4p1.5` — Render/delivery compatibility registry

```text
Title: Centralize render format, destination, and legacy envelope compatibility
Labels: area:surface, area:rendering, delivery:C-read-evidence-contract, lane:read-contracts
Description: RenderSpec names format/destination/layout, but public surfaces still own dialect aliases and envelope variants. Add a compatibility registry that maps each public command/route/tool to its RenderSpec defaults and envelope adapter. This lets the shared executors return one internal payload while preserving public shapes until versioned migrations.
Acceptance: CLI --format/--json/--to/--out, daemon route format params, MCP tool response envelopes, and API typed returns are declared in one registry. Generated docs/schemas render from the registry. At least CLI read and daemon read use the registry in production.
```
