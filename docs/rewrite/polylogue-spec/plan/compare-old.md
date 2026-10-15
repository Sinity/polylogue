# Comparing the new parsers with the old ones

`xtask compare-old` checks the new parsers against the Python parsers on the
operator's real files. The old parsers' output is a regression reference, not an oracle for source completeness. Every difference needs an explanation: a new-parser bug, an old-parser defect demonstrated against retained evidence, or an intended change listed below. This file says how to produce the old
output, how to map it onto the new shape, and how to compare the two.

The comparison happens at **parser output**, before any database, on both
sides. That sidesteps the old archive's hashed message and session IDs
entirely: the old parsers emit provider-native IDs (`provider_message_id`,
`provider_session_id`), and hashing only happened later in the old writer.

## 1. Dumping the old parsers' output

Entry point (old repository, `polylogue/sources/source_parsing.py`):

```python
parse_one_source_path(path, file_mtime=None, source_name=ORIGIN,
                      sidecar_data={}, capture_raw=False)
# yields (raw, ParsedSession) pairs; touches no database
```

The old tests call it the same way (`tests/unit/sources/test_location_binding.py`).
`ParsedSession`, `ParsedMessage` and `ParsedContentBlock` are pydantic models
in `polylogue/sources/parsers/base_models.py`.

The dump script lives in the new repository at `xtask/compare-old/dump_old.py`,
about 50 lines, and runs inside the old repository's devshell. It writes one
JSON line per session in a neutral shape, holding only the fields being
compared:

```json
{"file": "...", "provider": "claude-code", "id": "<provider_session_id>",
 "kind": "standard|subagent|prompt_suggestion|...", "parent": "...", "branch_point": "...",
 "title": "...", "cwd": ["..."], "branch": "...", "commit": "...",
 "events": ["message_usage", "claude_session_environment", "..."],
 "messages": [{"id": "<uuid>", "role": "user|assistant|tool|system",
               "type": "...", "material_origin": "human_authored|...",
               "timestamp": "...", "parent_message_provider_id": "...",
               "model_name": "...", "input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_write_tokens": 0, "stop_reason": "...",
               "blocks": [{"type": "text|thinking|tool_use|tool_result|image|code|document",
                           "text": "...", "tool_name": "...", "tool_id": "...", "tool_input": {},
                           "is_error": false, "exit_code": 0, "outcome_unknown_reason": "..."}]}]}
```

A file that raises an exception produces `{"file", "error"}` and the run
continues. A working draft of the script is kept beside the session scratch
(`plan/scripts/dump_old.py` in this directory); copy it into the new repository unchanged.

**Tried on real data (2026-10-11, old repo at `494cf00036`).** The command:

```console
$ cd /realm/project/polylogue/repo
$ direnv exec . python -I dump_old.py claude-code FILE...
```

- On one main session file and one subagent file, it produced 44 messages
  (18 assistant, 17 tool, 9 user) with 17 tool calls, 12 of them ok and 5
  errors. The subagent came out as a `prompt_suggestion` session with
  parent set.
- On 100 random real Claude Code files (26 MB), it took 30 s wall time, of
  which about 20 s was devshell startup, and peaked at 130 MB. It produced
  94 sessions and 7,125 messages, with 0 errors. The 6 files without a
  session are among the cases to explain.

## 2. Mapping the old shape onto the new one

The new side is the export format (`spec/export-format.md`), produced by the
new parser for the same file. Each old record maps across like this:

| Old | New | Rule |
|---|---|---|
| `provider` + `id` | `session.id` = `<origin>:<native id>` | main sessions: `claude-code:<id>`. Subagents: old `<parent session id>:agent-<agentId>`; new `claude-code:<parent session id>/<agentId>`: the old parent ID, `/`, and the part after `:agent-`. |
| `kind` `standard` / `primary` | `kind` `main`; a resumed session also gets a `continuation` link (`how: resume`) to the session it continues | |
| `kind` `subagent` / `prompt_suggestion` | `kind` `subagent` | the old prompt-suggestion kind is the agent type in the `subagent` link's `attrs` |
| `kind` `temporary` | `temporary: true` | |
| `parent`, `branch_point` | the `subagent` link's `parent` and `branch_point` | |
| message `id` (native uuid) | the message's native ID (compared as a field, not by splitting the new ID) | equal for every Claude Code message, since both sides use `uuid` |
| `role` `tool` | `role` `user` with `written_by` `harness` | the old parser gave tool-result records their own role |
| `material_origin` `human_authored` | `written_by` `human` | |
| `assistant_authored` | `model` | |
| `operator_command`, `runtime_protocol`, `runtime_context`, `tool_result`, `generated_context_pack`, `generated_analysis_pack` | `harness` | `operator_command` is flagged below |
| `unknown` | any value | listed as a difference, never matched silently |
| block `tool_use` | `tool_call` (`tool` = `tool_name`, `call_id` = `tool_id`, `input` = `tool_input`) | |
| block `tool_result` `is_error: true` / `false` | `outcome` `error` / `ok` | |
| `is_error: null` with `outcome_unknown_reason` | `outcome` `unknown` (or `pending`) with a reason | the reason values are compared as categories, not text |
| block `code` / `document` | `text` / `file` | the old parser split code blocks out of text; the new one does not |
| token fields | `usage.input/output/cache_read/cache_write` | |
| `events` | `events[].kind` | compared as counts per kind, through a short table of old-to-new kind names kept in the script |
| old link kinds `resume`, `fork`, `continuation`, `handoff` | `resume` and `fork` are `how` values of a `continuation` link (and query shorthands; `link:fork` means a forked continuation or a copied subagent); `continuation` is a kind; `handoff` is a `reference` link with `how: briefing` or `ping` | the old names are not kept as aliases: the new archive starts fresh, and nothing typed against the old one needs to keep working (`decisions/lineage.md`) |

## 3. Comparison rules

**Must be equal** for every session both sides found:

- the set of session IDs per input file, after the ID mapping;
- per session: the set of message IDs on the active path and their order,
  roles (mapped), `written_by` (mapped), and timestamps;
- per message: the block kind sequence; text block contents, compared by
  hash; tool names, call IDs and inputs;
- tool outcomes, where the old side recorded `is_error`;
- token counts and model names per message;
- `cwd`, `branch`, `commit`, and the subagent parent and branch point.

**Expected differences**, each from a deliberate new rule. The report counts
them separately so they cannot hide a bug:

| Difference | New rule |
|---|---|
| Old subagent IDs `<parent>:agent-<id>` | `spec/sources/claude-code.md`, identity |
| Tool results in `role=tool` messages | `spec/model.md`, roles |
| Text from side files (`tool-results/<id>.txt`) present in the new output, a placeholder in the old | `spec/sources/claude-code.md`, tool calls. The old dump runs without sidecar data. |
| `exit_code` filled from the output text (`Exit code N`) where the old parser left it empty | `spec/sources/claude-code.md`, tool calls |
| `unknown` with reason `no_result` for calls never answered (shown as `pending` while the session is live, computed at read time) | `spec/model.md`, tool outcome |
| Empty thinking blocks that carry only a signature, kept as empty thinking blocks | `spec/sources/claude-code.md`, record types |
| Messages marked `inherited_from` in resumed sessions; the old parser had no such mark at parse time (its writer cut the copied prefix later) | `decisions/lineage.md` |
| Code blocks: the old side split them out of text, the new side keeps them in the text | `spec/model.md`, blocks |
| Titles: the new side prefers `custom-title`, then `ai-title`, then `summary` | `spec/sources/claude-code.md`, identity |
| Events the new side does not create (the old `message_usage` per message, `claude_session_environment`) | usage lives on the message; environment fields live on the session |

**Typed slash commands** (decided 2026-10-11 from principle: `written_by`
is who produced the content). The command line the person typed
(`<command-name>/clear</command-name>` with its arguments) is
`written_by: human`; the text the harness expands it into (command or skill
body, `<local-command-stdout>`) is `harness`. This matches the old parser
for the typed part, so no difference is expected there.

**Reporting.** Per provider, the report gives:
- input files, and sessions and messages found by each side;
- for each difference kind, a count and up to five examples, each named by
  file path, session ID and message ID only.

Message text never appears in the report. Text is compared by BLAKE3, so
the report can be pasted into a bead or a PR. Full detail goes to a private
JSON file under the run directory, for local inspection.

```text
claude-code   files 16,047   sessions old 16,031 new 16,040   messages old 2,291,004 new 2,291,340
  equal                                   15,880 sessions
  only new: sessions                       9   (e.g. ~/.claude/projects/-x/abc.jsonl  claude-code:abc)
  expected: subagent id form           11,398
  expected: side-file text              4,212 blocks
  expected: exit code from text         8,031 blocks
  flagged:  operator_command → harness  1,904 messages
  DIFF:     tool outcome                   12   (e.g. … claude-code:… :<uuid>  old ok, new error)
```

(The numbers are illustrative.)

### Intended differences found by the audit of the Python code

Places where the old parser output differs from the new by design, so
`xtask compare-old` does not flag them (`audit/spec-additions.md` §11).
Rows where the rewrite behaves the same were removed; the other rows keep
their numbers:

| # | Python behavior | Rewrite behavior |
|---|---|---|
| 1 | Large or precise JSON numbers become floats | integer or exact number text (R15) |
| 2 | Lone surrogates kept exactly in text | U+FFFD in parsed text, counted |
| 3 | Block IDs content-based `<message>:b:<sha256>:<n>` | content-list index (`model.md`) |
| 4 | Codex usage = latest cumulative total | sum of per-request counts; the total only as a check |
| 5 | A known outcome may carry no reason | `error` may carry `denied`, `interrupted`, `timed_out` |
| 6 | `has_paste` guessed from length and code fences | paste only from hook or `history.jsonl` events |
| 7 | Timestamp sort key guesses milliseconds by magnitude | unit declared per wire field (R32) |
| 8 | Missing value shown as "unknown" | "—" |
| 9 | `excise` re-plans against live state at execution | sends exactly the scopes its preview showed, one `excision` row each |
| 10 | A continuation is refused when the archive changed | continues by key with a note (R52) |
| 11 | An index change that would orphan a note's reference is refused | the note shows a missing target, counted in `status` |
| 12 | Exports outrank native-API captures | native capture first (`sources/README.md`, several routes) |
| 14 | claude.ai duplicate IDs stay separate messages | measured first; split into `<id>~<n>` only if the repeats are distinct messages (`sources/chat-exports.md`, claude.ai) |
| 17 | A sandbox link with no name match joined to its message's first file | no bytes, with a reason (R76) |
| 18 | `with` capped at 200 rows per session, 5,000 per page, 2,000 characters; pages over 2,500 sessions refused | no caps (`query.md`, bounds per session) |
| 19 | At most six conditions explained by `--why`, none in boolean queries | every condition |
| 20 | Briefing tokens as the maximum of 1.3 × words and bytes / 4 | bytes / 4 |
| 21 | A short ID ambiguous in the index resolved through notes | an error listing the candidates (R76) |
| 22 | A garbage or negative `limit` on stack and compare becomes 50 | refused (R48) |
| 25 | Antigravity trajectory sessions | parsed from `~/.gemini/antigravity-cli/conversations/<id>.db` (10 real files here; `sources/others.md`, Antigravity); compare messages per step |
| 26 | Hermes lifecycle events joined by an unescaped `LIKE`, which can cross installs | an exact qualified join (`sources/others.md`, Hermes) |
| 27 | `parent:` pages can return fewer children than the limit | full pages from one SQL statement (R27) |

Sources: audit P24-r2:A1, audit P24-r2:missed.

## 4. Running it for slice 1

1. **Dump the old output once**, in parallel batches, to a private directory
   outside every repository:

   ```console
   $ out=/realm/tmp/work/compare-old/claude-code; mkdir -p "$out"
   $ find ~/.claude/projects -name '*.jsonl' -print0 \
       | xargs -0 -n 200 -P 8 sh -c \
         'direnv exec /realm/project/polylogue/repo python -I dump_old.py claude-code "$@" \
            > "'"$out"'/$(date +%s%N).jsonl"' _
   ```

   Estimated time, from the throughput above: about 1 s per 1–3 MB of input
   per process, plus devshell start per batch. For 19.7 GB across 8
   processes that is roughly one to two hours, run once, as a queued job.
2. **Run the new parser** over the same files: `xtask compare-old
   claude-code --old "$out"`. It parses each file with the new parser in
   memory and writes no archive.
3. **Read the report.** Each DIFF line is either fixed in the new parser or
   added to the expected-differences table above, with the rule it follows.
   Slice 1 is accepted when no unexplained DIFF lines remain.

The old dump is generated once and kept until the Python version is retired.
It is private data: it holds message text, so it lives under `/realm/tmp/work`
and never in a repository.

## Source conservation beyond parser parity

Compare retained bytes and revisions separately from parser output. For Codex, use the version-aware reconciliation rules in `spec/sources/codex.md`: catalog discovery, inherited ordinal boundaries and paginated projection content are separate checks. Include synthetic inherited-only, unregistered-readable, stalled-checkpoint, optional-null and active-writer cases. Record decoder/schema versions, as-of times, digests and unsupported populations. A matching old-parser output cannot close evidence the old parser ignored, and a reduced raw record count cannot establish a lost turn. Keep private corpus receipts outside Git.
