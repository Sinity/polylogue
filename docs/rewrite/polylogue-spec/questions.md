# Questions for the operator

Product choices that nothing in the evidence decides. Each has a default;
if you do nothing, work proceeds on it.

The other 44 questions that used to be here were settled by the spec
authors from the definitions, principles, your dated rulings, the Beads
designs and real data (`evidence/questions-resolution.md` has the reasoning
for each). Each settled answer is recorded where it applies; the table below
says where, and `decisions/README.md` says who decided each. Question numbers are kept so references elsewhere
stay valid.

## Open

**6 and 7. Order of work.** Daily use first: Claude Code, the live daemon
with Codex and hook briefings, the query language, web chats (slices 1–4),
then the web reader (slice 10) and the workbench of pages and pings
(slice 12). The evidence says pages and pings become the main working
surface eventually, so the question is only whether to reach them sooner.
*Default: as listed.* Stakes: weeks of ordering; the data model is the same
either way.

**47. Conversing with agents, or directing them?** Your own May question,
never answered (decision record `polylogue-v1-substrate-shape`, 2026-05-24):
do you want to be *in conversations with* agents, as one voice among several
in shared threads, or to *direct* agents, each a persistent identity you
call into a task and whose history accumulates? The first leads to a
dialogue substrate (threads, inboxes, notifications that wake an agent), the
second to agent identities with their own history and a director's console.
It shapes the workbench and pings (slice 12), not slices 1–4. *Default: the
record's provisional answer, directing (agent identities first, shared
threads deferred).* Stakes: slice 12's design.

**48. Should Polylogue back an agent's memory tool?** Your May decision
record `session-resume-reality-check` recommended wiring Anthropic's Memory
tool to a Polylogue-backed store, so an agent's `/memories` are queryable
across sessions, machines and agents; the spec later listed that as out of
scope (a spec-author decision, 2026-10-11), on the ground that it would make
Polylogue part of the harness. But agents already write notes to Polylogue
through MCP, so a memory-tool adapter would be a thin layer mapping the
tool's view, create, replace and delete onto notes, not a second store.
*Default: build that adapter after notes exist (slice 7 or later); until
then Polylogue reads harness memory files as inputs.* Stakes: one adapter;
the data model is unchanged.

## Settled

Rows 3 and 45 are the operator's own rulings; the rest were decided by the
spec authors.

| # | Question | Answer | Recorded in |
|---|---|---|---|
| 1 | Lineage storage | full message lists, copied messages marked, text stored once | `decisions/lineage.md` |
| 2 | Database files | three: `archive.db`, `index.db`, `embeddings.db` | `decisions/storage-files.md` |
| 3 | CLI writes and the daemon | every read and write goes through the daemon's socket (warm caches, one writer); no standalone mode; decided by the operator 2026-10-11 | `decisions/process.md` |
| 4 | Growing files | keep every revision | `decisions/raw-retention.md` |
| 5 | Repository | same repository, long-lived branch | `decisions/repository.md` |
| 8 | Who starts pinged agents | Polylogue resumes the addressed session itself | `spec/user-data.md`, pings |
| 9 | Agent edits | apply directly, attributed and revisable | `spec/user-data.md` |
| 10 | Marker syntax | `::kind(args): body` and `@[address] text` | `spec/user-data.md`, markers |
| 11 | Research features | kept as later features, built on first real use | `plan/slices.md` |
| 12 | Antigravity | every file stored; brain documents, state snapshots and CLI trajectory stores parsed; IDE conversations (encrypted) once the language server's step call is tried | `spec/sources/others.md` |
| 13 | Origin names | short names | `spec/model.md`, origins |
| 14 | Embeddings provider | Voyage first, a local model later | `spec/analysis.md` |
| 15 | Web reader port | stale: each archive already has its own port | `spec/surfaces/web.md` |
| 16 | Price table | shipped table plus a local override file | `spec/analysis.md` |
| 17 | Page naming | "page" is the one container; layouts are presentation | `spec/user-data.md` |
| 18 | Timeline and evidence-cutoff view | `evidence_until:` from slice 3; the visual timeline with the workbench | `plan/slices.md` |
| 19 | Voice to pages | Sinnix owns the gateway; it writes through Polylogue's normal write path | `spec/user-data.md` |
| 20 | Pages on Sinex | nothing to move: pages are notes, which backed mode keeps in Sinex | `decisions/sinex.md` |
| 21 | Secrets | keep the `has:secret` flag; flagged text is never sent to embedding providers | `spec/behavior.md` |
| 22 | Agents and excision | agents cannot excise | `spec/user-data.md`, `spec/storage/archive-db.md` |
| 23, 25 | When `primary` ships; privacy | right after the rebuild proof; privacy is Sinex's Q25 | `decisions/sinex.md` |
| 24 | Old archive data | restore spools in place; re-collect AI Studio from Drive | `plan/migration.md` |
| 26 | Extension transport | native messaging | `spec/surfaces/extension-and-clients.md` |
| 27 | Parallel workers | one checkout with the enforced git rule; fleets use worktrees | `plan/method.md` |
| 28 | `CHECK` constraints | none; enums are checked on write | `spec/storage/README.md` |
| 29 | Per-request timing records | not stored; tracing spans and an in-memory slow-query window | `spec/behavior.md` |
| 30, 43 | Notifications and health alerts | Polylogue shows changes; Sinnix pushes notices | `spec/surfaces/cli.md`, `spec/behavior.md` |
| 31 | Query language changes | accepted as listed | `spec/query.md` |
| 32 | SQL views | `polylogue sql` over versioned `v1_` views | `spec/storage/index-db.md` |
| 33 | Excision | kept | `spec/storage/archive-db.md` |
| 34 | Build and test history | kept in the shared xtask core | `plan/workspace.md` |
| 35 | Database snapshots | page snapshots, not logical exports | `decisions/raw-retention.md` |
| 36 | Benchmark baselines | the xtask history is the baseline | `plan/workspace.md` |
| 37 | Backing a harness memory tool | reopened as #48 (above), after the operator's May record recommended it | `spec/purpose.md` |
| 38 | Hook durability | measured 2026-10-11: `fdatasync` p99 5.3 ms on the spool's filesystem, under the 10 ms threshold, so every recorded hook event is synced; the synchronous briefing hook's delivery line is not, so a briefing never waits on a sync | `spec/sources/README.md`, hooks |
| 39 | IDs with lone surrogates | `~u:` plus the hex of the bytes; text keeps U+FFFD | `spec/model.md`, identity |
| 40 | Active path without a marked leaf | unknown, never guessed | `spec/model.md` |
| 41 | Repeated claude.ai IDs | measured 2026-10-12: no repeats in the four real exports, so the general fallback rule applies | `spec/sources/chat-exports.md` |
| 42 | Vectors of multi-block messages | report the unmatched; re-embedding them is acceptable | `plan/migration.md` |
| 44 | Searching unparsed files | no second read path; `status` shows what is waiting | `spec/behavior.md` |
| 45 | Live agents | tracked: live state from hook and process evidence, peers, live tasks; designed fresh, the Python coordination code is not ported; the operator asked for it 2026-10-11 | `spec/model.md`, Live state |
| 46 | Automatic judging | none | `spec/user-data.md` |
