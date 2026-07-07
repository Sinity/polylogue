---
created: 2026-06-28
purpose: Implementation map for the DSL `with <units>` projection clause (collapses context-pack into the algebra)
status: active — ready to implement; map from background agent a68dadb848e1876ec
read-with: context-pack-gratuitous-2026-06-28.md, dsl-vs-flags-audit-2026-06-28.md, MASTER-BACKLOG §3
---

# `with <units>` projection — implementation map

Goal: `sessions where repo:x with assertions, actions` returns each selected session WITH its related units
attached → makes context-packs expressible as queries; retires compile_context/ContextSpec/mcp context_pack DTOs.

## CENTRAL RISK (not the grammar — the dual execution/render paths)
- Spec/plan path (Python API, MCP) honors `SessionQuerySpec`. archive_execution.py.
- **CLI list path (`cli/archive_query.py`) is RAW-PARAMS — reconstructs filters from `request.params`, never
  consults `query_spec()`.** So a spec-only `with_units` SILENTLY NO-OPS through `find … then read --all`.
  Must wire `request.query_spec().with_units` into archive_query.py AND attach in archive_execution.py, keep
  BOTH renderers in sync. (Also: boolean_predicate has the same dual-path exposure — latent.)
- Secondary risk: split `with` OUTSIDE Lark (like `|`), respecting quotes/parens, or a quoted FTS term
  containing "with" mis-splits.

## Vertical slice (`with assertions`), ordered
1. PARSE (expression.py): add `_split_with_clause(expr)->(head, tuple[str,...])` modeled on
   `_split_pipeline_stages` (:1373, top-level scan respecting quotes/parens :1385-1408). Call at top of
   `compile_expression()` (:2566) and `parse_expression_ast()` (:1861). Validate unit tokens against the
   descriptor registry (`terminal_query_unit`/`query_unit_descriptor` in metadata.py); raise
   ExpressionCompileError on unknown. Apply predicate to `head` as today. (Grammar `_QUERY_GRAMMAR` :560-631;
   STRUCT_UNIT terminal :606; unit sources matched outside Lark via terminal_query_source_pairs.)
2. SPEC (spec.py): add `with_units: tuple[str,...] = ()` after boolean_predicate (:483). Set in BOTH return
   branches of `compile_expression()` (:2577 boolean, :2604 compact) and in `compile_expression_into()`
   replace(...) (:2638). LEAVE SessionQueryPlan untouched (projection is post-selection, not filter/sort/limit).
   Optionally add "with" to _RECOGNIZED_PARAMS (:200-244) only if MCP/daemon param surface should accept it.
3. FETCH HELPER (new, unit_results.py or sibling): `fetch_attached_units(archive, session_ids, units) ->
   dict[unit, dict[session_id, tuple[payload,...]]]`. For assertion: build `session.id:(id1|id2|…)` predicate,
   call `ArchiveStore.query_assertions(predicate, ...)` ONCE per page (archive.py:3960; LEFT JOINs assertions
   on `target_ref='session:'||session_id` :4009), map rows via `AssertionQueryRowPayload.from_row`
   (surfaces/payloads.py:1262), bucket by session. Drive method+payload off `query_unit_descriptor(unit)`
   (metadata.py QUERY_UNIT_DESCRIPTORS :702-812) so other units are free. Generic dispatch proven by
   unit_results.py:_execute_rows_terminal (:242-273).
4. EXECUTE+ATTACH (spec path) archive_execution.py: list_summaries_archive (:399-418)/list_archive (:421-440);
   after `archive_rows`/`_archive_summaries` (:336-376), call helper for `spec.with_units`, attach. (list_archive
   already re-reads full sessions via archive.read_session :437.) Thread with_units from caller (plan won't carry).
5. RENDER (CLI raw path) archive_query.py: read `request.query_spec().with_units` in
   `_execute_archive_query_stdout` (:146-676); fetch via helper using page_summaries ids; inject `attached_units`
   key in `_summary_payload`/`_emit_list` (:1305-1329); extend `_summary_line` for markdown/plaintext.
6. VERIFY: devtools test tests/unit/cli + parser tests; mypy --strict; dogfood on dev archive
   `find 'repo:polylogue' with assertions then read --all --format json`.

## Result model
SessionSummary/Session (archive/session/domain_models.py:30-46,63-83). Add optional
`attached_units: dict[str, tuple[payload,...]] = {}` (or small AttachedUnits model). Unit payloads exist +
JSON-ready: AssertionQueryRowPayload (payloads.py:1262), ActionQueryRowPayload (:1163), etc. Use
build_query_unit_envelope (:2021) shape. On CLI raw path, simplest = extra key in _summary_payload dict (no
domain-model change needed there).

## REUSE (don't reimplement) — this IS what we're collapsing
compile_context (api/archive.py:2162+) loops session_ids → recovery_digest → get_session →
_compile_recovery_context_from_digest (:1942-1976) → list_assertion_claims(target_ref="session:<id>") (:1993).
context/compiler.py compile_recovery_context (:145-210), ContextImage/Segment (:32-90); context/pack.py;
context/assertion_claims.py. `--include-assertions` flag (compiler.py:70; api:1947,2278) + compile_context/
ContextSpec retire once `with` covers them.

## Generalization
Steps 3-5 are unit-agnostic (metadata.py maps every unit→sql_query_method+payload_model). Only per-unit nuance:
session-scoping join — query_assertions joins target_ref; query_actions/query_messages scope via
session_filters/message joins. Confirm each query_* accepts a session.id predicate before enabling that unit.

## Key files
expression.py, spec.py, unit_results.py, metadata.py; cli/archive_query.py, cli/query_verbs.py,
cli/read_view_handlers.py, cli/root_request.py; storage/sqlite/archive_tiers/archive.py; surfaces/payloads.py;
api/archive.py, context/compiler.py.
