# Real-data cases

Each becomes an example directory under `examples/sources/<format>/` (input
plus expected output) or a law in `laws.md`, in the slice that brings its provider:

| Case | Where | Bead |
|---|---|---|
| Codex `response_item` payloads in both shapes (the older and the current one) parse to the same message | `examples/sources/codex/response-item-shapes/` | `sinnix-ir7p` (d2-import) |
| empty tool-call IDs never pair; an unpaired call stays `pending`/`unknown` | `examples/sources/claude-code/empty-tool-ids/`, `codex/empty-tool-ids/`; law 13 | `polylogue-1tyq5.187` |
| an explicit inactive-path flag wins over inferred order | `examples/sources/chatgpt-export/explicit-inactive/` | `polylogue-1tyq5.187` |
| a Codex state table with a column named `rowid` | `examples/sources/codex/state-rowid-column/` | `polylogue-1tyq5.187` |
| usage events without a timestamp still count; totals never depend on page size | laws 14 and 15 | `polylogue-1tyq5.187` |
| a ZIP member name of any length is kept whole | `examples/sources/chatgpt-export/long-member-name/` | `polylogue-1tyq5.187` |
| a session larger than the parameter limit | law 35 | `polylogue-1tyq5.187` |
| AI Studio parent chains with mixed time-zone offsets order by instant | `examples/sources/aistudio-drive/mixed-offsets/` | `polylogue-54a31` |
| a Gemini CLI prompt in `displayContent` only | `examples/sources/gemini-cli/display-only-prompt/` | `polylogue-54a31` |
| byte-identical files at two paths | law 40 | `polylogue-54a31` |
| a Grok export sample | `examples/sources/grok-export/`; law 13 | `polylogue-5yig1` |
| a growing transcript whose two side files have identical bytes | `examples/sources/claude-code/grown-with-shared-side-file/`; law 2 | `polylogue-8cra7` |
| a zero-byte session file | `examples/sources/claude-code/empty-file/`; law 37 | `polylogue-bt3c4` |
| Hermes databases with integer session keys, parsed live and from the stored copy | `examples/sources/hermes/integer-session-keys/` | `polylogue-ckoyg` |
| a ChatGPT reply authored by a tool; media nested in a Claude Code tool result | `examples/sources/chatgpt-export/tool-authored-reply/`, `claude-code/nested-result-media/` | `polylogue-e58ps` |
| strings holding U+2028, U+2029 and U+0085: one record | `examples/sources/claude-code/unicode-line-separators/` | `polylogue-fl6ld` |
| a claude.ai capture tool result with `exit_code` 7 and `is_error` null keeps outcome `error` and the code | `examples/sources/claude-ai-capture/exit-code-without-flag/` | `polylogue-hiw9v` |
| a Claude Code file carrying two or three session IDs | `examples/sources/claude-code/several-session-ids/` | `polylogue-kqqxf` |
| a search with a tab or newline between words: separate terms | query reference-model cases (section 5) | `polylogue-mfr9d` |
| a duplicated record (same ID, same content) is one message | `examples/sources/claude-code/duplicate-record/` | `polylogue-va53x` |
| Gemini, Antigravity and Hermes keep stderr, display text and their own IDs | `examples/sources/gemini-cli/stderr-and-display/`, `antigravity/`, `hermes/` | `polylogue-znuts` |
| decoy files under provider locations (an analysis JSONL under `projects/`, `subagents/agent-*.meta.json`, an Antigravity `*.md.metadata.json`, a Gemini CLI JSONL that passes a Claude record check, a SQLite file, an empty AI Studio chunk wrapper; also a `workflows/wf_*.json` run snapshot, a relationship-index JSONL with a `conversation` field, a bare `tool_use_id` record, `tool-results/hook-*` files (hook events, not tool output), a Gemini CLI checkpoint stub, a Hermes skill-template array under `~/.hermes`, and raw bytes that are a SQLite database, refused before JSON detection): outcome "not a session" | `examples/sources/claude-code/decoys/`, `antigravity/decoys/`, `aistudio-drive/empty-envelope/` (rules R16, R18) | `polylogue-21qj`, `6bebe`, `ioz7`, `t83e`, `eo81`, `hbtj2`, `mvcbi` |
| a Claude Code tool call re-emitted under a new message with the same `tool_use_id` | `examples/sources/claude-code/reemitted-tool-call/`; law 60 (rule R59) | `polylogue-xnkf` |
| Claude Code rewrites record one while appending the rest | `examples/sources/claude-code/first-record-rewritten/`; law 21 (rule R46) | `polylogue-ps055` |
| a provider time years after the collection time | `examples/sources/claude-code/future-timestamp/` (rule R9) | `polylogue-dgr8c`, `m5djs` |
| integers beyond 2^53, a non-finite number | `examples/sources/claude-code/large-numbers/` (rule R15) | `polylogue-7pjy3`, `pirai` |
| all-digit timestamps in seconds and in milliseconds | `examples/sources/codex/epoch-units/` (rule R32) | `polylogue-z3sv`, `xj7ru` |
| a project directory reached through a symlink | `examples/sources/claude-code/symlinked-project-dir/` (rule R26) | `polylogue-m706z` |
| signature-only thinking blocks with an empty body | `examples/sources/claude-code/signature-only-thinking/` | `polylogue-r39b` |
| October 2025 Codex rollouts that the Python parser read as zero messages | `examples/sources/codex/2025-10-layout/`; law 49 | `polylogue-i415` |
| `models/gemini-2.5-pro`-style model names, priced after normalization | `examples/sources/aistudio-drive/model-resource-names/` (rule R33) | `polylogue-6j9c` |
| a ChatGPT capture larger than 8 MiB | `examples/sources/browser-capture/large-capture/` | `polylogue-mvq8` |

Cases carried over from the Python test suite (`evidence/deep-tests.md`,
section 4). They come from redacted real fixtures or live-corpus
measurements; the source column names the Python test. The 125 Python
fixture files are synthetic or redacted and seed `examples/sources/` after a
privacy read (`plan/migration.md`).

| Case | Where | Source |
|---|---|---|
| a resume after automatic compaction stores the replayed prefix once, as copied messages | `claude-code/acompact-resume/` (slice 1) | `test_claude_code_normalization_laws.py::test_acompact_resume_replayed_prefix_is_stored_once_and_composed` |
| a subagent that compacts itself stays whole and never takes the main session's prefix | `claude-code/subagent-self-compaction/` | `…::test_subagent_self_compaction_stays_whole_and_never_composes_main_prefix` |
| a subagent collected before its parent resolves as a fresh subagent once the parent arrives | `claude-code/subagent-before-parent/` | `…::test_subagent_child_arriving_before_parent_resolves_as_spawned_fresh` |
| an instruction the harness generated for an agent is `written_by: harness` | `claude-code/generated-instruction/`; law 98 | `…::test_agent_artifact_topology_prevents_generated_instruction_from_becoming_human` |
| compaction summary in both wire shapes, `compact_boundary` records, microcompaction | `claude-code/compaction-shapes/` | `test_compaction.py` |
| a compaction boundary's preserved-segment anchor wins over the logical parent | `claude-code/compaction-preserved-anchor/` | `test_compaction.py::test_preserved_segment_anchor_wins_over_logical_parent` |
| a file of only `file-history-snapshot`/`progress` records is not a session; a session whose first 32 records are checkpoints still is | `claude-code/checkpoint-only/`, `claude-code/checkpoint-prefix/` | `test_artifact_taxonomy.py` |
| a subagent's tool-result side files sit at the session level | `claude-code/subagent-side-files/` | `test_dispatch_payloads.py` |
| `agent-name` sets the title; `pr-link` keeps its PR fields; `frame-link` also comes count-only (105 of 234) | `claude-code/sidecar-records/` | `test_claude_code_sidecar_evidence.py` |
| `queue-operation` enqueue carries the drafted prompt; dequeue, remove and popAll carry none | `claude-code/queue-operations/` | `…::test_queue_operation_enqueue_persists_content_dequeue_does_not` |
| two records asserting different `forkedFrom` sessions: no lineage parent is adopted | `claude-code/conflicting-forked-from/` (rule R76) | `…::test_conflicting_forked_from_claims_adopt_no_lineage_parent` |
| a tool result listing more than 50 filenames keeps all of them, in order | `claude-code/many-filenames/` | `…::test_tool_use_result_preserves_complete_filename_evidence…` |
| `thinkingMetadata` on a user record; a negative `maxThinkingTokens` is omitted | `claude-code/thinking-metadata/` | `test_claude_code_unread_wire_fields.py` |
| `requestId`, `apiBlockIndex` and `advisorModel` stay with their turn's usage | `claude-code/usage-fields/` | `test_claude_code_unread_wire_fields.py` |
| `hookAdditionalContext` on a system record is context the model saw | `claude-code/hook-additional-context/` | `…::test_hook_additional_context_becomes_runtime_context_content` |
| a `gitBranch` in Claude Code's `claude/…` namespace is never a title | `claude-code/auto-branch-name/` | `test_assembly.py::test_git_branch_claude_prefix_skipped` |
| a measured numeric zero stays zero, not unknown | `claude-code/zero-counters/` | `test_counter_and_coordinate_contracts.py::test_claude_numeric_zero_remains_measured` |
| a tool output before its call is kept and paired; a call with no output stays unpaired | `codex/out-of-order-output/` (slice 2) | `test_codex_event_stream_contract.py` |
| a second `session_meta` is a continuation only with structural evidence | `codex/second-session-meta/` | `test_codex_event_stream_contract.py` |
| code-mode `exec`: a call site that never ran; one run twice; repeated transport IDs; a patch inside its transport span | `codex/code-mode-children/` | `test_codex_event_stream_contract.py` |
| a `compacted` record's replacement history is stored whole; a lone surrogate in it survives | `codex/compacted-history/` | `test_codex_compaction_and_turn_context.py` |
| `turn_context` instructions start a new revision only when they change | `codex/turn-context-revisions/` | `…::test_the_turn_context_branch_is_reachable_and_numbers_revisions` |
| a `state_5.sqlite` title that repeats the opening prompt (166 of 2,771) ranks below a real title | `codex/state-title-echo/` | `test_assembly.py::test_state_title_downgrades_when_it_echoes_the_first_prompt` |
| reading `state_5.sqlite` while Codex holds a write transaction | law 111 | `parsers/test_codex_state.py::test_snapshot_reads_consistent_state_while_writer_holds_a_transaction` |
| ChatGPT: a timing-only node keeps its generation duration; an equal legacy duration is a duplicate | `chatgpt-export/timing-only-node/` (slice 4) | `test_chatgpt_normalization_survivors.py` |
| ChatGPT ZIP: an upper-case `.ZIP` suffix; assets bound to pointer attachments, also one directory down; metadata siblings skipped | `chatgpt-export/zip-shapes/` | `test_assembly_chatgpt.py`, `test_dispatch_payloads.py` |
| ChatGPT shared page (`chatgpt.com/share/<id>`): a flat `messages` list | `chatgpt-export/shared-page/` | `test_dispatch_payloads.py` |
| ChatGPT `codex.json`: each Codex Cloud task is its own session | `chatgpt-export/codex-cloud-tasks/` | `test_dispatch_payloads.py` |
| ChatGPT: one attachment referenced several times is one blob with several references | `chatgpt-export/duplicate-attachment-occurrences/` | fixture `native-duplicate-attachment-occurrences-v1.json` |
| every key in a committed ChatGPT export is mapped or excluded | law 112 | `test_chatgpt_field_ledger.py` |
| claude.ai: a conversation-level file stays unreferenced; `memories.json` is account memory; `chat_messages: []` (47 measured) is not a session | `claude-ai-export/conversation-files/`, `memories/`, `empty-chat/` | `test_claude_ai_conversation_level_attachments.py`, `test_dispatch_payloads.py` |
| claude.ai: sibling variant continuations both survive | `claude-ai-export/sibling-variants/` | `test_claude_web_normalization.py` |
| browser capture is detected before the weaker ChatGPT mapping shape | `browser-capture/detector-order/` | `test_chatgpt_normalization_survivors.py` |
| Grok: two `grok.json` exports in different folders are two sessions; a reordered re-export is one; a malformed entry is no session | `grok-export/identity/` (slice 5) | `parsers/test_grok.py` |
| AI Studio: an empty `chunkedPrompt` or mistyped `chunks` is no session; two id-less turns at one time citing different files are two turns | `aistudio-drive/edge-shapes/` | `test_gemini_drive_normalization_laws.py` |
| AI Studio: role-less chunks skipped; text plus parts not repeated; per-part `thoughtSignature` kept | `aistudio-drive/chunk-shapes/` | `test_gemini_chunked_prompt_contract.py` |
| Hermes: a finish stamped before its start pairs by correlation; duplicate markers count once; an unended scope is unpaired | `hermes/lifecycle/` | `parsers/test_hermes_lifecycle.py`, `parsers/test_hermes_spans.py` |
| Hermes: two installs reusing one raw session ID stay two sessions; `state.db` and ATOF agree on the install key | `hermes/two-installs/` | `parsers/test_hermes_spans.py`, `parsers/test_hermes_state.py` |
| Hermes ATIF: any `ATIF`-prefixed schema version kept; tool arguments never copied into span events | `hermes/atif-versions/` | `parsers/test_hermes_spans.py` |
| Antigravity: a retried section reusing a message ID gives one active leaf | `antigravity/duplicate-message-id/` | `parsers/test_antigravity.py` |
| OpenTelemetry: a non-GenAI trace is no session; a conversation keeps its ID across an upgrade | `otel/resource-identity/` | `parsers/test_otel_genai.py` |
| hook payloads in camelCase and snake_case generations detect the same provider and paste evidence | `hooks/payload-generations/` (slice 2) | `test_hook_producer.py`, `test_hook_paste_enrichment.py` |
| a hook event never creates a session; a nested transcript is rejected | law 113 | `test_hook_carriers.py`, `test_hook_producer.py` |
| a `parent_id` names its nearest earlier occurrence when IDs repeat | `chatgpt-export/repeated-parent-id/` | `test_active_branch.py` |
| a reordered export resolves the conversation by identity; `<container>:<member>` names a member only inside a real ZIP | `chatgpt-export/member-addressing/` | `test_export_member_addressing.py` |

Cases from the conceptual and write-path audits (`evidence/conceptual-*.md`,
`evidence/deep-writes.md`), measured on the operator's corpus:

| Case | Where | Source |
|---|---|---|
| a split response interleaved with its own tool result (3,603 of 3,661 continuation records in a sample) | `claude-code/split-response-interleaved/`; law 31 | conceptual-identity 6 |
| parallel tool results under one folded response are not branches (651 in 21 of 80 files) | `claude-code/parallel-results/`; law 99 | conceptual-model-fit 1 |
| a fork subagent repeating the spawning response under a new uuid (40 of 61 forks) | `claude-code/fork-repeats-spawn/`; law 25 | conceptual-model-fit 3 |
| a hook paste joined by time window records `matched_by: time_window` | `claude-code/paste-time-window/` (rule R76) | deep-writes A2 |
| invalid UTF-8 in a record: parsed with U+FFFD, counted as `invalid_utf8` with its offset | `claude-code/invalid-utf8/` | deep-writes A1 |
| Codex prompt echo 1 ms apart from its event pairs (31% of typed prompts duplicated otherwise) | `codex/prompt-echo-1ms/`; law 4 | conceptual-identity 7 |
| Codex repeated `token_count` events (7.7% repeats) | `codex/repeated-token-count/`; law 96 | conceptual-construct-validity 1 |
| a Codex `Reasoning` item repeated with growing `summary_text` | `codex/progressive-reasoning/`; law 48 | conceptual-identity 8 |
| ChatGPT "branch in new chat" sharing node IDs across conversations (20 groups in one export) | `chatgpt-export/branch-in-new-chat/`; law 11 | conceptual-identity 5 |
| Hermes JSON `session_start` in local time without an offset (2 h from `state.db`) | `hermes/local-time/`; law 105 | conceptual-time T7 |
| a Claude Code `async_launched` Agent result is a launch acknowledgement: the call waits for the task notification for its outcome (56% of `Agent` results) | `claude-code/async-agent-launch/` | conceptual-construct-validity 13 |
| a Gemini CLI `subagent` chat sharing its `sessionId` with a `main` chat gets a `subagent` link from it | `gemini-cli/subagent-same-session-id/` | conceptual-model-fit 14 |
| a compaction summary (`isCompactSummary`) is the compaction event's text, never a message; `microcompact_boundary` is event `microcompaction` | `claude-code/compaction-shapes/`; law 34 | conceptual-model-fit 13 |
| Claude Code `Notification` with `idle_prompt` is not a wait on the operator | `hooks/idle-notification/`; law 85 | conceptual-construct-validity 15 |
