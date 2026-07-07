---
created: "2026-06-29T11:43:33+02:00"
purpose: "Audit Polylogue's Claude Code and Codex field semantics against provider code and local corpus samples"
status: "active"
project: "polylogue"
---

# Agent Runtime Field Semantics

## Scope

Investigated current Polylogue parsing/storage of Claude Code and Codex runtime
sessions, with ChatGPT / Claude.ai / Gemini checked as contrast cases. Evidence
came from current Polylogue code, inactive upstream Claude Code and Codex
codebases under `/realm/project/_inactive`, and recent local corpus samples from
`~/.claude/projects` and `~/.codex/sessions`.

## Findings

- Polylogue's three message axes are useful but currently overloaded in places:
  `role` is provider/API channel, `message_type` is artifact shape, and
  `material_origin` is authorship/provenance.
- The current `classify_material_origin` correctly refuses to infer
  `human_authored` from runtime `role=user` alone. Chat-export sessions are
  upgraded at session validation time because their user channel is structurally
  cleaner.
- Claude Code upstream has a direct human-turn predicate:
  `type === 'user' && !isMeta && toolUseResult === undefined`. Other upstream
  comments add that content-level `tool_result` is sometimes more reliable than
  `toolUseResult` for subagents.
- Polylogue mostly catches Claude Code tool results because recent local rows
  with top-level `toolUseResult` also carry `message.content` `tool_result`
  blocks. It does not preserve top-level `toolUseResult` metadata beyond the
  block text / `is_error`.
- Current Claude Code parser appends many non-message sidecar records as empty
  `role=unknown` messages: `attachment`, `mode`, `last-prompt`, `ai-title`,
  `bridge-session`, `permission-mode`, `pr-link`, and empty `system` records.
- Codex upstream `RolloutItem` includes `session_meta`, `response_item`,
  `inter_agent_communication`, `compacted`, `turn_context`, and `event_msg`.
  `EventMsg::UserMessage` is used by Codex metadata sync for first user message
  / title; `EventMsg::TokenCount` is used for total token usage.
- Current Codex parser materializes `response_item.message`,
  `function_call`, and `function_call_output`; many other first-class variants
  (`custom_tool_call`, `custom_tool_call_output`, `tool_search_call/output`,
  `web_search_call`, `reasoning`, `agent_message`) are mostly stored only as
  compact events or ignored by durable typed surfaces.
- In a recent Codex sample, most `event_msg.agent_message` text exactly matched
  `response_item.message` assistant rows, but a non-trivial tail was unpaired.
  `event_msg.user_message` mostly matched `response_item.message role=user`.

## Conclusions

The right direction is not to collapse all providers into one fake role model.
Keep provider truth and add explicit cross-provider semantic projections:

- raw provider channel (`role`) should remain honest;
- model-visible/user-visible/human-authored/tool-result/context/protocol should
  be separate facts, not inferred from one enum;
- runtime `role=user` should stay unknown unless backed by provider-native
  human-turn evidence or a hook/assertion;
- provider sidecars should become typed events/facets, not empty messages.

## Implemented Slice

Patched the immediate parser semantics:

- Claude Code real user prompts now become `human_authored` when the native
  predicate holds: `type=user`, not `isMeta`, no `toolUseResult`, no
  content-level `tool_result`, and no non-human `origin.kind`.
- Claude Code non-human origins become protocol/runtime rather than vague
  user-message rows.
- Claude Code sidecar rows with no message payload are skipped instead of
  becoming empty `role=unknown` messages.
- Claude Code continuation summaries are runtime context, and compaction
  summaries are generated context packs.
- Codex user response messages and unpaired `event_msg.user_message` rows become
  `human_authored` after context filtering.
- Codex `event_msg.user_message` / `agent_message` rows are deduped against
  already-materialized `response_item.message` rows.
- Codex custom/tool-search/web/local-shell call variants now materialize through
  the existing tool-use/tool-result block model where their payloads carry
  enough structure.

Focused verification:

```bash
devtools test tests/unit/sources/test_parsers_claude_code_artifacts.py \
  tests/unit/sources/test_parsers_codex.py \
  tests/unit/sources/test_codex_event_stream_contract.py \
  tests/unit/sources/test_tool_result_role_reclassification.py \
  tests/unit/storage/test_archive_tiers_write.py -q
# 144 passed
```

Recent local sample after the patch (20 newest files each):

- Claude: `user/message/human_authored=35`, `tool/message/tool_result=2658`,
  no prompt-shaped `unknown` bucket in the sample.
- Codex: `user/message/human_authored=1229`,
  `user/context/runtime_context=112`, no prompt-shaped `unknown` bucket in the
  sample.

## Follow-on Surface and Schema Slice

Moved the corrected semantics into useful recall/readiness surfaces:

- MCP `get_messages` now normalizes enum-backed `role`, `message_type`, and
  `material_origin` values before filtering. Agents can reliably ask for
  `material_origin=human_authored`, `material_origin=runtime_context`,
  `message_type=context`, etc. over archive-backed sessions.
- Added MCP contract coverage for the human/context/assistant split.
- Centralized the session-insight materialization ledger vocabulary in
  `SESSION_INSIGHT_MATERIALIZATION_TYPES`.
- Added status/readiness fields for run projection tables:
  `session_runs`, `session_observed_events`, and
  `session_context_snapshots`.
- Repair assessment now counts missing materialization-ledger rows as real
  row debt, including the run-projection rows. This fixes the old mismatch
  where a ready flag could be false while repair saw zero row debt.
- Public insight readiness now exposes run-projection artifacts as
  `session_runs`, `session_observed_events`, and
  `session_context_snapshots`, with aliases `run-projection`,
  `observed-events`, and `context-snapshots`.

Focused verification after the surface/schema slice:

```bash
devtools test tests/unit/storage/test_session_insight_status_descriptors.py \
  tests/unit/storage/test_repair.py \
  tests/unit/api/test_facade_contracts.py::test_archive_tiers_api_session_insight_status_reads_index_tier \
  tests/unit/cli/test_insights.py::test_insights_status_json -q
# 26 passed

devtools test tests/unit/storage/test_session_insight_status_descriptors.py \
  tests/unit/storage/test_repair.py \
  tests/unit/storage/test_session_insight_refresh.py::test_apply_session_insight_session_updates_async_preserves_thread_roots_for_children \
  tests/unit/api/test_facade_contracts.py::test_archive_tiers_api_session_insight_status_reads_index_tier \
  tests/unit/mcp/test_tool_contracts.py::TestGetSessionTool::test_get_messages_filters_role_type_and_material_origin \
  tests/unit/sources/test_parsers_claude_code_artifacts.py \
  tests/unit/sources/test_parsers_codex.py \
  tests/unit/sources/test_codex_event_stream_contract.py \
  tests/unit/storage/test_archive_tiers_write.py -q
# 167 passed before the final public-readiness missing-count aggregation;
# the narrower 26-test rerun covers that final aggregation.
```

Next schema/insight cleanup candidates:

- Decide whether "insights" should be renamed in code-facing schema to
  "derived read models" / "recovery projections" while keeping public command
  vocabulary stable.
- Add a public readiness spec for latency profiles or deliberately classify
  latency as a derived metric rather than an insight product.
- Continue separating stored evidence/inference/enrichment payload columns from
  public "insight" product envelopes so the schema says what is source-derived
  versus model/heuristic-derived.

## Devloop Anchor Correction

2026-06-30 correction: the separate `/realm/tmp/polylogue-dev/archive` archive
was retired to avoid split-brain evidence. The canonical active archive for this
devloop is `/home/sinity/.local/share/polylogue`, index schema v18, with one
devloop daemon and no prod daemon. Current direct counts at the time of this
correction are 13,116 sessions, 3,947,844 messages, 4,050,065 blocks, 13,382
session runs, 1,854,045 observed events, and 13,382 context snapshots.

The older larger session/message count in this note came from a pre-dedup or
stale archive view. Do not reuse it as current evidence, and do not reintroduce
`/realm/tmp/polylogue-dev/archive` as a live database root.

Historical devloop archive recipe, retained only as provenance:

```bash
POLYLOGUE_ARCHIVE_ROOT=/realm/tmp/polylogue-dev/archive
XDG_DATA_HOME=/realm/tmp/polylogue-dev/xdg
```

Historical counts there were 4,302 sessions, 1,380,539 messages, and
4,302 session profiles. `polylogued status` under that env reported 4,110 tracked
cursor files, 0 failed, 26 excluded, and health warning rather than the
production archive's critical stale-maintenance state.

The current repo `devtools workspace dev-loop --json` suggests a newer
branch-local archive root at `/realm/project/polylogue/.local/dev-archive`, but
that root is only partially present and its `index.db` does not currently have a
`sessions` table. Until it is prepared/launched with the dev-loop command, use
the `/realm/tmp/polylogue-dev/archive` environment for claims about the existing
devloop dataset.

Follow-up implemented: `devtools workspace dev-loop` now probes the selected
branch-local archive read-only and emits `archive_status` with `schema_ready`,
`user_version`, `session_count`, and `message_count`. It warns when the selected
archive root is only a directory or has an `index.db` without the `sessions`
table. Live smoke:

```bash
devtools workspace dev-loop --json | jq '{dev_archive_root, archive_status, warnings}'
# .local/dev-archive => schema_ready=false, sessions_table_exists=false, warning emitted

devtools workspace dev-loop --archive-root /realm/tmp/polylogue-dev/archive --json \
  | jq '{dev_archive_root, archive_status, warnings}'
# /realm/tmp/polylogue-dev/archive => schema_ready=true, user_version=18,
# session_count=4302, message_count=1380539, warnings=[]
```
