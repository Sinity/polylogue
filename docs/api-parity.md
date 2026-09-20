[← Back to README](../README.md)

# CLI / MCP / Python operation parity

<!-- GENERATED FILE: edit polylogue/api/parity.py, then run `devtools render api-parity`. -->

Parity is governed by stable operation identities, not by name-shaped
reflection. Each row below is one semantic operation; each cell is either a
live binding that must resolve or an intentional absence with its reason.
MCP rows are derived from the MCP declaration registry, so a new tool adds a
row here automatically.

Drift check: `devtools verify api-parity --check` (also run by
`devtools verify --quick` as `gate api-parity`).

## Operations

| Operation | Summary | CLI | MCP | Python |
| --- | --- | --- | --- | --- |
| `api.embedding_preflight` | Report whether the embedding backend is usable before a semantic read. | `polylogue ops embed preflight` | _absent_: embedding readiness is reported inside the MCP `status` envelope, not as its own tool | `archive.embedding_preflight()` |
| `api.embedding_status` | Report embedding coverage and staleness for the active archive. | `polylogue ops embed status` | _absent_: embedding coverage is reported inside the MCP `status` envelope, not as its own tool | `archive.embedding_status()` |
| `api.import_annotation_batch` | Import a durable typed annotation batch under a declared schema version. | `polylogue annotations import` | `write` | `await archive.import_annotation_batch()` |
| `mcp.tool.context` | Compile a policy-gated bounded context image with receipts. | `polylogue context` | `context` | `await archive.context_image_payload()` |
| `mcp.tool.emit_decision` | Record a typed decision event with evidence references for a session. | _absent_: decision emission is an MCP/library transaction; the CLI exposes `note`, not decision records | `emit_decision` | `await archive.emit_decision()` |
| `mcp.tool.explain` | Explain parser grammar, capabilities, refs, result semantics, or recovery. | _absent_: query-grammar explanation is served by the generated discovery surfaces (`polylogue manual`, docs/search.md); no dedicated CLI verb owns it | `explain` | `await archive.explain_query_expression()` |
| `mcp.tool.get` | Resolve one exact stable object or evidence identity. | `polylogue read` | `get` | `await archive.resolve_ref()` |
| `mcp.tool.judge` | Accept, reject, defer, or supersede assertion candidates without collapsing candidate state. | `polylogue judge` | `judge` | `await archive.judge_assertion_candidates()` |
| `mcp.tool.maintenance` | Rebuild session insights and inspect or adjudicate operation recovery. rebuild_insights and recovery_adjudicate require confirm=true and fail closed without it. | _absent_: insight rebuild is owned by the daemon convergence route (`polylogued run`); the CLI exposes inspection (`ops insights status`), never a rebuild verb | `maintenance` | `await archive.rebuild_insights()` |
| `mcp.tool.query` | Execute a parser-owned terminal query page or resume its q2 continuation. | `polylogue find …` (root query mode) | `query` | `await archive.query_units()` |
| `mcp.tool.read` | Read a stable archive URI or public ref through a declared view. | `polylogue read` | `read` | `await archive.resolve_ref()` |
| `mcp.tool.record_work_event` | Record a typed live-agent work event for a session. | _absent_: agent work-event recording is an MCP/library transaction; no interactive CLI verb owns it | `record_work_event` | `await archive.record_work_event()` |
| `mcp.tool.run` | Execute a saved query or governed recipe ref. | `polylogue select` | `run` | _absent_: `run` executes a saved query or governed recipe ref; the facade exposes the underlying query operation instead of a ref-execution wrapper |
| `mcp.tool.status` | Report compact archive authority and readiness status. | `polylogue status` | `status` | `await archive.stats()` |
| `mcp.tool.write` | Apply a declared mutation operation after shared authorization. Destructive operations (delete_session, remove_tag, remove_mark, delete_metadata, delete_annotation, delete_saved_view, delete_recall_pack, delete_workspace) require confirm=true and fail closed without it. | `polylogue mark` | `write` | _absent_: `write` is a capability-gated transaction dispatcher; each of its operations is its own semantic operation with its own facade callable |

## Classification of the public Python facade

Every public callable on `polylogue.api.Polylogue` (165 at
render time) is either bound by an operation above or listed here as an
explicit exclusion. An unclassified callable fails the parity gate.

| Category | Reason | Callables |
| --- | --- | --- |
| `agent-coordination` | agent coordination reads/writes surfaced through `polylogue agents`, which owns their own contract | `compile_and_record_context`, `compile_context`, `correlate_claude_agent_dispatches`, `correlate_hermes_context_deliveries`, `get_agent_policies`, `get_context_delivery`, `get_effective_context`, `get_hook_event_summary_for_session`, `get_session_orchestration`, `list_blackboard_notes`, `list_context_deliveries`, `list_context_injection_ledger`, `post_blackboard_note`, `record_context_delivery`, `record_manual_continuation` |
| `aggregate` | aggregates reached through the `query` operation's aggregate result semantics | `aggregate_sessions`, `archive_count_sessions`, `archive_debt`, `cost_outlook`, `count_sessions`, `diagnose_query_miss`, `export_otel`, `facets`, `get_index_status`, `get_stats_by`, `health_check`, `hermes_integration_health`, `list_command_shape_usage`, `list_read_view_profiles`, `origin_usage_report`, `pathology_report`, `query_completions`, `query_sessions`, `session_usage_reconciliation`, `storage_stats`, `tool_call_latency_distribution`, `workflow_shape_distribution` |
| `assertion-review` | assertion candidate review helpers behind the `judge` operation's queue | `assertion_candidate_queue_health`, `capture_assertion_candidate`, `judge_assertion_candidate`, `list_assertion_candidate_reviews`, `list_assertion_candidates`, `list_assertion_claim_payloads`, `list_assertion_claims`, `list_comparative_judgments`, `record_comparative_judgment` |
| `context-pack` | bounded packet builders composed from the `context` operation rather than carrying their own identity | `compare_sessions`, `context_preamble_payload`, `correlate_sessions`, `find_abandoned_sessions`, `find_resume_candidates`, `find_similar_sessions_by_metadata`, `list_sessions_for_spec`, `neighbor_candidate_payloads`, `neighbor_candidates`, `portfolio_bundle`, `postmortem_bundle`, `resume_brief`, `search_similar_sessions`, `session_correlation_payload`, `topic_pack` |
| `ingest` | ingest stages owned by `polylogued run`; the CLI/MCP parity target is the daemon, not the facade call | `compact_lineage`, `explain_import`, `parse_file`, `parse_sources`, `reconcile_codex_spawn_edges`, `reconcile_hermes_session_lifecycle`, `regenerate_private_fable_packet` |
| `insight-projection` | descriptor-driven insight registry (polylogue/analysis/registry.py) owns cross-surface parity for these | `export_insight_bundle`, `find_stuck_session_latency_profile_insights`, `get_session_insight_status`, `get_session_latency_profile_insight`, `get_session_profile_insight`, `get_session_profile_record`, `get_thread_insight`, `insight_readiness_report`, `insight_rigor_audit`, `list_archive_coverage_insights`, `list_archive_debt_insights`, `list_cost_rollup_insights`, `list_session_cost_insights`, `list_session_latency_profile_insights`, `list_session_profile_insights`, `list_session_tag_rollup_insights`, `list_thread_insights`, `list_tool_episode_insights`, `list_tool_usage_insights`, `list_usage_timeline_insights` |
| `lifecycle` | construction, teardown, and lazy builders; they carry no cross-surface operation identity | `close`, `filter`, `iter_messages`, `open` |
| `read-detail` | object-level reads reached through the `read`/`get` operations' refs rather than their own operation id | `archive_get_session`, `bulk_get_messages`, `get_actions_batch`, `get_ancestors`, `get_descendants`, `get_file_edits`, `get_logical_session`, `get_messages_paginated`, `get_raw_artifacts_for_session`, `get_session`, `get_session_events`, `get_session_stats`, `get_session_summary`, `get_session_topology`, `get_session_tree`, `get_sessions`, `get_siblings`, `get_thread`, `get_web_content_constructs`, `list_sessions`, `list_summaries`, `read_transcript_window`, `search`, `search_envelope`, `search_session_hits` |
| `user-state` | durable user.db overlays whose parity target is the annotation/marker family, not a single operation | `add_mark`, `add_tag`, `bulk_tag_sessions`, `clear_corrections`, `create_recall_pack`, `delete_annotation`, `delete_correction`, `delete_metadata`, `delete_recall_pack`, `delete_session`, `delete_session_safe`, `delete_view`, `delete_workspace`, `get_annotation`, `get_metadata`, `get_recall_pack`, `get_setting`, `get_view`, `get_workspace`, `join_typed_annotations`, `list_annotations`, `list_corrections`, `list_marks`, `list_recall_packs`, `list_settings`, `list_tags`, `list_views`, `list_workspaces`, `record_correction`, `remove_mark`, `remove_tag`, `save_annotation`, `save_view`, `save_workspace`, `set_metadata`, `update_metadata` |
