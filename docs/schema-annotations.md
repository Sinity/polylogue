# Schema Semantic Annotations

Schema elements under `schemas/providers/<provider>/` carry
`x-polylogue-semantic-role` annotations that describe the semantic
role of each JSON path (e.g., `message_role`, `session_title`,
`message_timestamp`).

## Review process

1. Run schema inference: `devtools schema-generate <provider>`
2. Review generated annotations for correctness
3. Create or update `pins.json` in the provider directory to reject
   known-wrong annotations (see `claude-code/pins.json` for format)
4. Re-run inference — `select_best_roles()` will respect pin overrides
5. Promote reviewed annotations: `devtools schema-promote <provider>`
6. Verify: `python devtools/verify_schema_annotations.py`

## Verification

```bash
python devtools/verify_schema_annotations.py
```

Exits 0 when all shipped providers have at least one annotated element.

## Session Timing Measures

Session profile timing intentionally exposes two different archive-local
measures:

- `engaged_duration_ms`: message-clustered wall clock. It sums phase intervals
  separated by no more than the current five-minute phase idle threshold. This
  does not measure human attention or foreground focus.
- `phase_idle_threshold_ms`: the idle threshold recorded on every materialized
  session phase. It is currently 300000ms and explains why message-clustered
  wall clock split or dropped long gaps.
- `tool_active_duration_ms`: paired provider tool windows. It sums
  timestamped provider tool-call start/output pairs. This does not infer
  duration for unpaired, untimestamped, or provider-opaque work.

Worked examples:

- One 12-minute Bash call followed by a response: `engaged_duration_ms` is
  near zero when the message gap exceeds the phase threshold;
  `tool_active_duration_ms` is about 720000.
- Ten operator messages six minutes apart: both values are near zero because
  the gaps exceed the message-clustering threshold and there are no tools.
- Four two-minute tool calls with short replies between them: both values are
  about eight minutes because phase boundaries stay open and the tool pairs are
  timestamped.

## Session Shape Measures

`workflow_shape` and `terminal_state` live on `session_profiles` as
construct-valid archive signals, not intent labels.

- `workflow_shape` is derived from message/tool counts and tool categories. It
  distinguishes chat, exploration, agentic loops, subagent dispatch, and batch
  review. It does not measure quality, importance, or productivity.
- `terminal_state` is derived from the final meaningful message and provider
  tool-event pairing. It distinguishes clean finishes from unanswered user
  turns, trailing errors, and pending tool calls. It does not decide whether the
  operator should resume the thread.

Both fields carry confidence plus JSON evidence so downstream readers can
inspect the rule input instead of treating the labels as opaque semantics.

## Session Latency Measures

`session_latency_profiles` separates response and provider-tool timing from
the broader session profile:

- `median_tool_call_ms`, `p90_tool_call_ms`, and `max_tool_call_ms` summarize
  timestamped provider tool-call start/output pairs.
- `stuck_tool_count` counts provider tool starts left open past the fixed
  stuck threshold when the session end is known.
- `median_agent_response_ms` summarizes user-to-assistant message gaps.
- `median_user_response_ms` summarizes assistant-to-user message gaps with a
  cap on long idle pauses.
- `tool_call_count_by_category_json` carries category counts from the same
  session-profile tool classifier.

These measures do not infer hidden provider work, correctness, human attention,
or operator productivity. Missing provider timestamps lower coverage; they do
not create synthetic latency rows.
