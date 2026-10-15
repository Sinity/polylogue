# Slices

Each slice is usable on its own, built in final form, and verified before the
next starts (`plan/method.md`). The order puts daily use first: by the end of
slice 4 the operator can search and read all agent sessions, live, from the
terminal and from agents, and the old Python archive's data is all in the
new archive, so its copies can be deleted (`plan/migration.md`; no Python
code is carried over, and the old daemon is already stopped).

**The schema is whole from slice 1** (rule R63): slice 1 creates every table
of all three files, and each slice writes only the tables its code fills
(slice 1's list is in `plan/slice-01.md`). The check that every stored column
has a reader (`plan/testing.md`, drift checks) runs as a release check once
every slice is built.

**Slices 5–12 are provisional** (`polylogue-sinex-joint/PLAN-2026-10-12.md`):
their rows and the spec sections only they use stay as written and are not
refined further until the slice is next; building slices 1–4 will change
what they need. The next two slices are refined; slice 13 (Polylogue on
Sinex) stays at the level of the joint contract until slices 1–4 exist.

| # | Slice | Delivers | Size estimate (product code) |
|---|---|---|---|
| 1 | **Claude Code, end to end** | The spool files the reset set aside, put back so convergence stores them (`plan/migration.md`). Every revision of every input kept, with change classification (unchanged, appended, head rewritten with the rest appended, rewritten, deleted), observations with full metadata, and pieces for appends (`spec/storage/blobs.md`). The daemon's Unix socket (newline-delimited JSON) with the slice-1 request subset, every command and the MCP proxy as its clients. Every Claude Code session and subagent file read, stored, parsed and indexed; search, `read`, `status`, `export` with its manifest, JSON output, `import` (including `--dry-run`), `input cat`; an MCP server with `search` and `read`; lineage marks, compaction events and per-session facts (counts, tokens, cost from the price table). The daemon rescans periodically (file watching comes in slice 2); `--until-idle` exits once caught up. Unreadable locations retried and shown as unavailable; an empty archive reports ok; the writer lock on the archive's resolved path; deadlines, panic capture per parse task, fair stepping and stall reporting (`spec/convergence.md`, `spec/behavior.md`); `status` figures as measured, measured absent or unmeasured, and models with no price. CI on free runners. Mining the old parsers' review findings for real data shapes (`evidence/beads/beads-switchover.md`). Also from 2026-10-12: the per-record parse framework (law 120), change detection with the racy-unchanged guard and stored-byte hashes (law 121), status totals as writer-kept counters (law 122), read-back sentences (`read_as`) and logical days for the slice's time words. Credential masking of CLI, export and MCP output, with `--reveal` (the web reader's in slice 10). | 12–16k lines |
| 2 | **Live daemon and Codex** | File watching (appends parse incrementally from slice 1), the systemd unit; delta encoding for rewrites, joining quiet pieces, and the `replaced` and `renamed` change classes (`spec/storage/blobs.md`); the `hook` command, hook spool ingestion and `install-agent` (records what it will change before writing); briefings injected by hooks (directory briefing at session start, what changed on resume, the session's own briefing after compaction, takeovers for exact session references; `spec/surfaces/mcp-and-hooks.md`), the `brief` verb and MCP tool, and recorded deliveries with the `deliveries` unit; the Codex provider, including its SQLite databases as snapshots stored as page patches, its pre-rewrite rollouts from the old archive as earlier revisions, every revision parsed, and `inherited_from` on Codex fork copies; `follow` with live-state lines; live state (`spec/model.md`): the process identity and job variables in hook lines, the daemon's process sampler and spool, live facts on sessions, the `processes` unit, peers, `polylogue running`, `read --view live`, the `status` live list and the briefing line about peer agents; the "running now" and "waiting on you" blocks in the overview; when a source location disappeared and reappeared; the agent-use test of the shipped skill and tools (`plan/testing.md`). The Morrow archive gains a hook spool with one delivered `prompt_reference` briefing to #4 naming #2 (`examples/archive/FACTS.md`, "Hook spool (slice 2)"). Also: provider database snapshots through `sqlite_dbpage` (law 117); CPU and peak memory on agent processes; the machine section from Sinex when configured (joint #18); the `ending` field from records and normalized commands (law 97), which briefings need. | 6–9k |
| 3 | **Query language and facts** | The full query language (`spec/query.md`) with the reference-model tests: every unit and field, including `turns`, `output:`, `command:` and `path:` (the tool-output and trigram indexes arrive here) and `evidence_until:`; facts beyond slice 1's (subscription-window usage, tool outcome and command statistics); `analyze`; saved-query definitions with unit and denominator behind the shipped figures; `polylogue sql` over the versioned `v1_` views; `statusline`; advisory hooks before tool calls; `setting` (time zone, subscription plan); completion and the fuzzy picker. | 6–9k |
| 4 | **Web chats, and the old archive's data brought in** | ChatGPT and claude.ai/Claude Design exports via `import`; merging one conversation from several routes; attachments, including files and URLs sessions link to; Polylogue's own export format as a source (re-import changes nothing); the browser capture spool with polite fetching, page scripts ignored, chunked large attachments, download retries that never block each other, and per-message archive state in the provider's page; the ChatGPT field table and sandbox attachments; capture-state words in the extension and the native host's reply naming the stored hash; the purchased embedding vectors imported; the old archive's raw inputs fed back in as ordinary inputs and checked against the old parsers (`plan/compare-old.md`) before its copies are deleted; the Python client Lynchpin reads Polylogue through (`spec/surfaces/extension-and-clients.md`, placed here when outlining: no slice had it). | 6–9k |
| 5 | **Remaining providers** (provisional) | AI Studio from Drive (OAuth), including the 11 chats copied from the old local cache; Gemini CLI (whole-session JSON and per-turn JSONL); Gemini app conversations from Takeout My Activity; Hermes (`state.db` as database snapshots); Antigravity (every file stored; brain documents, `state.vscdb` and `conversation_summaries.db` snapshots and the `antigravity-cli` trajectory databases parsed; encrypted `.pb` IDE conversations once the language server's step call is tried); Grok web; Grok CLI; Meta Muse Code once a verified native file exists; Claude Code Web runs. | 5–8k |
| 6 | **Lineage and context** (provisional) | `inherited_from` for the remaining providers (Claude Code has it from slice 1, Codex from slice 2); every link kind; the `context` view; delegation cards; agent configuration in force (instruction files, skills, settings) linked to sessions; resume candidates (`continue`); takeovers that follow lineage to the newest successor; replaying the hand-made handoffs to measure briefings; lexical `recall`; claude.ai compaction summaries; the compaction lifecycle (pre-compaction snapshot, dropped references, restoration); run grouping from harness evidence; live tasks (the AgentCTL job record as a companion input, `bd` claim commands the session ran) and child processes linked to tool calls. | 4–6k |
| 7 | **Notes and memory** (provisional) | Tags, notes, claims and decisions on any reference; `judge`, including `--blind`; goals, questions, decisions and blocked work with their states; corrections tied to what they correct; a triage inbox; which evidence supports which claim kind (`spec/analysis.md` §10); backlinks; decision pointers on prompts; session outcome; hiding and `excise`; inline markers; saved queries and kept results; `status --agents`; the change feed (`changes`), which the Lynchpin client (slice 4) gains then; `cite`; `import --notes`. Also: checks on claims, rubrics and annotation passes judged on blind samples (`spec/user-data.md`; laws 118, 119; joint #16); workflows 20 and 21 (`spec/workflows.md`). | 4–6k |
| 8 | **Schema inference and generated tests** (provisional) | The offline schema command, the synthetic input generator, the laws on generated archives, drift reporting in `status`; `polylogue demo` from the Morrow world with the claim-versus-record case (`spec/presentation.md`); the README with executed command blocks; tool-specific synthetic content. Acceptance includes a person other than the operator installing it and answering one real question about their own sessions from the README alone. | 3–5k |
| 9 | **Semantic search** (provisional) | Embeddings (Voyage or local), `near:` and hybrid search, cost preflight; semantic hits in `recall`; `embeddings.db` migrations; rejected texts recorded once and listed; vector reuse within a documented model family. | 2–3k |
| 10 | **Web reader** (provisional) | Loopback web UI: search, session list, session reader with live tail, the day page (also `polylogue day`), cost and agent-activity views, the live page (`/live`); `polylogue open`; keyboard and screen-reader access; one table of value states shared with the terminal; live updates over the change feed. Acceptance includes a smoke test in a real browser over the Morrow archive: pages render, show no private paths, and take their data from the JSON API. Also: the session report (workflow 18) and saving an annotated excerpt as one HTML file (workflow 19). | 5–8k (+ frontend) |
| 11 | **Agent analytics** (provisional) | Views of what agents run and how it goes, time analysis, failure statistics over time, cost per outcome; git correlation (commits, branches and PRs linked to sessions; the `changes` view); shipped saved views (repeated failing commands, abandoned with a question, most expensive failures, edits never committed, reverted outcomes, context efficiency, subagents with no resulting commit, prompt friction counted not classified); watched queries with opt-in notices, a minimum change size and a cooldown; signals (rules as `signal` notes, deliveries, the interrupt budget; `spec/user-data.md`, signals; law 140); the work graph as a view over runs, links and notes; export profiles. Signals also feed the directory briefing's watching section (lines agents choose, `spec/user-data.md`). | 3–5k |
| 12 | **Workbench** (provisional) | Pages, workspaces and investigations; pings by address (a ping to a `working` session waits until it is `idle` or `waiting`); roles; extension in-page features (summonable panel, selection capture, archive search in the page), reverse organization and posting; reproduction checks and session replay; lessons and proposed instruction improvements; variants (translations, summaries) beside the original; shared findings that cite archive references; comparisons and rankings. The visual timeline (a page or session as of a past moment) comes here; the `evidence_until:` field exists from slice 3. Also: the project story page (workflow 17), excerpts with page layout, and temporal joins across units (`spec/query.md`, planned extensions). | to be designed with the operator after slice 10, once `questions.md` #47 (converse with agents or direct them) is answered; each part needs a consumer who meets it unasked (`vision/README.md`, the consumer test), so a view-only part such as investigations waits for an agent-read or pushed form |
| 13 | **On Sinex** (right after slice 4) | Polylogue on Sinex for the operator (`decisions/sinex.md`), the same work as Sinex's slice 6, done together. In order: the `polylogue-sinex` adapter (not in the default build), publishing over Sinex's ingress by asking what Sinex holds and sending the rest; finite fixed-basis shadow qualification against isolated Sinex endpoints through the native publisher; the identity bridge; the shared example-archive contract test passing in both repositories; rebuilding the local index from Sinex, proved on the real corpus; two archives combined through Sinex (law 23); only then `primary` mode; backfill of the existing corpus after the proof; excision as tombstones and crypto-shredding in Sinex (after slice 7, which brings excision); publishing notes likewise once slice 7 has them; ambient Sinex context in session views. Also Beads, OTel and eval exports, the Codex AppServer event stream, A2A task streams once a real producer exists, and Beads issues as nodes of the work graph. | designed jointly with Sinex's slice 6 |

Total for slices 1–11: roughly 55–85k lines of Rust, against 640k lines of
Python today. The estimate is design information, not a budget; a slice that
grows well past it gets a second look in review.

**Later, built on first real use:** formal experiments with arms, judge
calibration and other research features the Beads plan. They stay in the
spec because it describes the full product, and are built when a real study
needs them. Decided by the spec authors from the operator's wish for a
full-featured spec and `plan/method.md` (2026-10-11); the operator may
overrule.

The visual timeline waits for the workbench because it is presentation over
data that exists from slice 3; daily use comes first. Decided by the spec
authors (2026-10-11); the operator may overrule.

Sources: `evidence/beads/beads-switchover.md`, "Sources of `plan/slices.md`".

## Why this order

- **Claude Code first**: the operator's largest and most-used source, with
  the richest structure (subagents, compaction, tool calls). Getting it right
  exercises most of the model.
- **Live before breadth**: the daemon and hooks make the archive current,
  which is what makes it trustworthy to use daily.
- **Query and facts before more providers**: agents and the operator need to
  ask real questions early; more providers then plug into a working whole.
- **Web chats with migration**: the old archive can be switched off once its
  sources are covered, its raw inputs are stored, and its paid vectors are
  carried over.
- **On Sinex right after slice 4**: once Polylogue on its own SQLite files is
  in daily use, slice 13 is done together with Sinex's slice 6, before slices 5–12.
  Neither project waits on the other before that point.
- **Workbench last**: it builds on everything else and needs the operator's
  input on details. Whether daily use (slices 1–4) should come before the
  reader and the workbench is the operator's question (`questions.md`).

## Acceptance for every slice

- All spec examples marked with a slice built so far pass (they are tests).
- The laws whose slice is among the slices built so far pass on the example archive
  (slice 13, built right after slice 4, runs the laws of slices 1–4 and 13;
  slices 5–12 then also run 13's; slice 13's parts that need notes or
  excision, slice 7, are tagged "after slice 7" and run from then on)
  and, from slice 8, on archives from the inferred-schema generator
  (`plan/testing/laws.md`).
- A run on the operator's real files: every input of the slice's providers is
  parsed or has a stated reason; counts agree with the old archive or the
  difference is explained; time and memory are within `spec/behavior.md`.
- On that same real archive, meaning as well as counts: a sample of
  sessions, drawn at random per provider, read against their source files
  for authorship, tool pairing and outcome, and lineage, because counts that
  agree between two paths sharing one bug prove nothing; and ordinary use:
  appends while live, a daemon restart mid-round, and a second round that
  changes nothing (`polylogue-6yg68`, which judged one real archive on five
  independent outcomes rather than a count of closed checks).
- The operator has used it.

Slice 1 is specified in full in `plan/slice-01.md`.
