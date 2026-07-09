# D4 "Behavioral Archaeology": Six DSL Queries, Rapid Fire

## Claim

Six DSL queries answer engineering-lead questions about AI coding sessions
that no chat UI transcript view can answer — using only existing product
primitives (`polylogue` CLI query DSL), no bespoke scripts.

## Corpus

`polylogue demo seed` fixture world: 11 sessions, 43 messages, deterministic
and private-data-free.

## Method

Each query run once via the `polylogue` CLI against the seeded archive; raw
output captured in `run.log`. Query 1 additionally demonstrated with
`--explain` to show the parsed AST.

## Findings

1. **SEQ thrash-loop hunt** — `seq(action:shell -> action:shell)` finds
   sessions with back-to-back shell-tool calls (a real repeated-attempt
   signal): 2 of 11 sessions match (`claude-code-session:63705dcc-...`,
   `codex-session:demo-00`), confirmed via `then select --json`.
2. **Tool call volume** — grouping actions by tool: Bash 9, Read 8, Task 1,
   Write 1, exec_command 1. Bash dominates this fixture's tool usage.
3. **Which tools break** — grouping failed actions (`is_error:true`) by
   tool: Bash 4, exec_command 1. Read/Write/Task never fail in this corpus.
4. **Semantic probe across providers** — `near:"flaky async test"` returns
   0 results despite a session literally titled "Debugging flaky async
   pipeline tests" existing in the corpus. This is itself an honest finding,
   not a failure to hide: the seeded fixture has only 2 of 43 messages with
   embeddings materialized (`SELECT COUNT(*) FROM message_embeddings_meta`
   on `embeddings.db`), so semantic search correctly returns nothing rather
   than a coincidental keyword-adjacent false match. A live archive with
   full embedding coverage would find this session; the demo corpus's
   sparse coverage is exactly the failure mode `near:` is designed to
   report honestly (no result) rather than fabricate.
5. **Time-scoped session population** — `since:2y` selects 9 of the 11
   sessions (2 predate the 2-year window), demonstrating the same kind of
   filter an "abandoned in this repo this quarter" query would use as its
   time bound. (Actual abandonment classification — `find_abandoned_sessions`
   — is a dedicated MCP/insight tool with its own severity scoring, not a
   raw DSL predicate; this query demonstrates the time-filtering mechanism
   that tool composes with, not abandonment scoring itself.)
6. **Query piped into `read`** — `find 'origin:codex-session' then read
   --first --view messages` resolves the query to a session and renders its
   messages directly, including a real captured tool error (`exit_code: 4`,
   "file or directory not found: tests/missing_test.py") and the agent's own
   next-step response — exactly the kind of "click through to the turn"
   affordance a chat UI can't give you from a saved transcript.

## Specimens

See `evidence.ndjson` for the cited session/query refs.

## Counterexamples

**A real defect surfaced while authoring query 1.** The bare command
`polylogue find "sessions where seq(action:shell -> action:shell)"` (no
`then` verb) returns `mode: list, total: 11` — the FULL unfiltered session
set — while the identical predicate via `then select --json` correctly
returns only the 2 matching sessions. This is not SEQ-specific: bare `find
"sessions where origin:codex-session"` also returns the unfiltered total
(11), while the equivalent compact form `find "origin:codex-session"`
(no `sessions where` prefix) correctly filters to 5. The explicit boolean-
query entry form appears to be silently ignored specifically in bare-`find`
(no trailing `then <verb>`) mode. Filed as polylogue-70qb rather than
worked around or hidden — this IS the demo's thesis in action: a DSL query
surfaced a real product defect a chat transcript view never could.

## Limits

- This is the seeded, deterministic demo corpus (11 sessions), not the live
  archive. Absolute counts are small and illustrative, not representative
  of production session volume or failure rates.
- Query 4's zero-result outcome is a property of this fixture's sparse
  embedding coverage (2/43 messages), not a claim that semantic search is
  broken.
- Query 5 demonstrates time-filtering only; it does not replicate
  `find_abandoned_sessions`'s severity/resumability scoring.
- The polylogue-70qb defect (bare-find ignoring `sessions where` predicates)
  means any operator running these exact commands without `then select`/
  `then read` should expect an unfiltered list, not the intended filtered
  result, until that bug is fixed.

## Reproduce

```bash
polylogue demo seed --root /path/to/demo-archive --force
export POLYLOGUE_ARCHIVE_ROOT=/path/to/demo-archive
export POLYLOGUE_FORCE_PLAIN=1

polylogue find 'sessions where seq(action:shell -> action:shell)' then select --json
polylogue 'actions where exit_code:>=0 | group by tool | count'
polylogue 'actions where is_error:true | group by tool | count'
polylogue find 'near:"flaky async test"'
polylogue find 'since:2y'
polylogue find 'origin:codex-session' then read --first --view messages

# --explain demonstration
polylogue --explain find 'sessions where seq(action:shell -> action:shell)'
```

See `run.log` for the exact recorded output of every command above.
