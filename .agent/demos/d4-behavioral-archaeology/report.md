# D4 "Behavioral Archaeology": Six DSL Queries, Rapid Fire

## Claim

Six DSL queries answer engineering-lead questions about AI coding sessions that no chat UI transcript view answers from a saved transcript alone, using only existing product primitives (`polylogue` CLI query DSL), with no bespoke scripts.

## Corpus

The fresh private-data-free run used the deterministic `polylogue demo seed` fixture at `run:d4-behavioral-archaeology:20260809T061905Z`. It contained 19 sessions, 71 messages, and 2 synthetic message embeddings. The command receipt was captured at commit `d23f5dd27a0bda0a9c0a4306ef98c898d4d920ce` on 2026-08-09.

## Method

The seed and verification steps ran against a throwaway private synthetic archive. Each of the six PROMPT.md commands then ran once through the `polylogue` CLI, followed by the required `polylogue --explain` route for Q1. `run.log` records each exact command, output, exit code, and output digest. The packet validator cross-checks those structured fields. The older 2026-07-09 receipt at `fdd5ea848` is retained only as historical context.

## Findings

1. **SEQ thrash-loop hunt**: `seq(action:shell -> action:shell)` finds 3 sessions with consecutive shell-tool calls: `codex-session:demo-receipts`, `claude-code-session:63705dcc-...`, and `codex-session:demo-00`. The current command uses `then select --format json`. The older `then select --json` spelling appears only in clearly labeled historical-receipt fields.
2. **Tool call volume**: the current run reports 26 successful actions: Bash 9, Read 8, exec_command 3, and one each for Edit, Task, Write, apply_patch, read_file, and run_check.
3. **Which tools break**: the current run reports 7 failed actions: Bash 4, exec_command 2, and Edit 1.
4. **Semantic probe across providers**: `near:"flaky async test"` returns one result, a Claude Code session with the synthetic fixture message "I will inspect the generated fixture and adjust the next command." Only 2 of 71 messages have synthetic embeddings, so this result does not establish complete semantic coverage.
5. **Time-scoped session population**: `since:2y` lists 13 of 19 sessions. This demonstrates time filtering only. It does not reproduce the severity or resumability scoring of `find_abandoned_sessions`.
6. **Query piped into `read`**: `find 'origin:codex-session' then read --first --view messages` renders `codex-session:demo-receipts`, including a failed clock test command with exit code 1, a conflicting success claim, an `apply_patch`, and a later successful test result.

## Specimens

See `evidence.ndjson` for the cited session, action, embedding, and current-run references. The current command and output mapping is in `packet.json.provenance.current_run` and is independently bound to `run.log` by output digests.

## Counterexamples

The original D4 work also recorded a real defect: the bare command `polylogue find "sessions where seq(action:shell -> action:shell)"` ignored the explicit predicate and returned the full session set. That historical comparison remains documented as `polylogue-70qb`. It is not presented as a newly executed command in this receipt, and the retired `then select --json` alias is retained only as historical receipt text.

## Limits

- This is a deterministic seeded demo corpus, not the live archive. The 19-session and 71-message counts are illustrative and do not estimate production prevalence, archive scale, or provider-wide behavior.
- The semantic result is bounded by 2 synthetic embeddings out of 71 messages. It does not prove semantic search completeness.
- The time query demonstrates filtering only. It does not replicate `find_abandoned_sessions` severity or resumability scoring.
- The packet records a current run and a historical receipt. The historical receipt is not evidence for the current code SHA.

## Non-claims

- The fixture counts do not estimate production prevalence, archive scale, or provider-wide behavior.
- The packet does not prove that semantic search is complete when embeddings are sparse.
- The packet does not prove that a chat UI could never implement equivalent features.

## Reproduce

```bash
polylogue demo seed --root /path/to/private-synthetic-demo-archive --force
export POLYLOGUE_ARCHIVE_ROOT=/path/to/private-synthetic-demo-archive
export POLYLOGUE_FORCE_PLAIN=1

polylogue find 'sessions where seq(action:shell -> action:shell)' then select --format json
polylogue 'actions where exit_code:>=0 | group by tool | count'
polylogue 'actions where is_error:true | group by tool | count'
polylogue find 'near:"flaky async test"'
polylogue find 'since:2y'
polylogue find 'origin:codex-session' then read --first --view messages

# --explain demonstration
polylogue --explain find 'sessions where seq(action:shell -> action:shell)'
```

See `run.log` for the exact current output of every command above. The packet's current-run receipt identifies the exact code SHA, UTC timestamp, production route, private synthetic archive class, command mapping, and output hashes.
