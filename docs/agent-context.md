# Agent context

Packs, budgets, briefs, takeover, recall, deliveries and advisories use the
[agent contract](agents.md) and the shared [reading pipeline](reading.md).

## Packs

A pack turns a selection into context for a recipient and a purpose: `polylogue QUERY pack [NAME] [--for SESSION|operator] [--purpose TEXT] [--budget N]`, or MCP `brief`. The selection is a query, a kept `result:`, a page, or references. The session-start brief, takeover and recall are **named packs**: fixed section lists over fixed selections, built by the same builder, with the same fitting, trust rendering, entry map and delivery record.

A pack holds:
- passages as written, and the assertions on them with author and judgment state;
- open questions and problems, and whatever a current assertion `blocks`;
- **contrary assertions:** current operator or operator-accepted agent assertions with a `challenges` or `corrects` link to an included item precede its text; other authors' contrary assertions are counted with references and inspection commands, without inlining their bodies;
- the repository state of the selected sessions ("HEAD is N commits past <recorded commit>");
- applicable methods, as nominations with their assessment ([assertions.md#memory](assertions.md#memory));
- the **entry map**, last: what is whole, what is shortened, what can be expanded and with which command, what was left out and why.

A pack for continuing work lets its recipient recover the goal, the constraints, the approaches already rejected, the current candidate, the limits of what is known and the next useful check.

Sending a pack creates a delivery ([Deliveries](#deliveries)); preparing one without sending creates a delivery in state `prepared`.

**Fitting.**
- Figures (counts, totals) are computed over every candidate before fitting, so they do not depend on the budget.
- Candidate entries are fitted across all sections before expanding any passage. Each entry keeps its references and source attribution outside a losslessly JSON-quoted source prefix, on a datamarked `┃` line. Useful first-line prefixes share the remaining space before further expansion in stable order; a last expansion is shortened on a UTF-8 boundary, with its quote closed. The entry map counts whole, shortened and omitted candidates and names their expansion commands.
- A section whose read failed or ran past its deadline renders "not available (reason)", never "none".
- **Stable prefixes:** section order is fixed, items within a section are ordered by a stable key, and the manual comes first, so a later delivery to the same session appends a delta and earlier text keeps its prompt-cache reuse.
- Large relevant documents are legitimate context; a table of contents is never mandatory.
- No model calls.

## Budgets

A budget is a request-envelope field (`budget_tokens` on the socket, HTTP and MCP, `--budget N` on the CLI; a token counts as 4 bytes), with a default per named pack. Defaults are product values, not settings, adjusted by measured exploration saved.

| Pack | Default budget |
|---|---|
| brief at `startup` or `clear` | 10,000 tokens plus the manual |
| brief at `resume` | 4,000 tokens |
| brief at `compact` | 10,000 tokens plus the manual and the operator's prompts, which are never cut |
| takeover | 10,000 tokens per prompt, shared by every session the prompt names |
| recall | 10,000 tokens |

A missing budget uses the named default. JSON `budget_tokens: null` or CLI `--budget none` lifts the token limit; a numeric budget, including zero, remains explicit. A budget too small for the required header, entry map and one candidate entry is refused; compaction prompts remain outside the fitted budget.

Each harness's inline cap is declared with its hooks. A pack larger than the cap is injected as whole sections up to the cap, then a line naming the command that prints the whole delivery (`polylogue delivery:<id> read`); the delivery stores every byte, and text is never silently cut.

## Brief

The session-start brief is injected by the `SessionStart` hook (and Hermes's first `pre_llm_call`):

| Source | Content |
|---|---|
| `startup`, `clear` | the brief of the payload's `cwd` |
| `resume` | what changed since the last session-start, resume or compaction delivery this session received with outcome `delivered` |
| `compact` | the session's own brief: every prompt its operator typed (`written_by: human`), verbatim and never cut; the latest todo list and plan; assertions on the session; failures not later fixed; what the compaction dropped ([sessions.md#compaction-boundaries](sessions.md#compaction-boundaries)) |
| `fork` | nothing |

**"Here" is the repository, not the directory:** the current directory, its checkout root, every worktree sharing its Git common directory, the project home, and every repository name recorded for sessions there. Sessions in sibling worktrees are labelled with their worktree.

**Sections, in order.** Each item carries its reference and age (`3 d ago`); each section names what it left out and the command that shows it.

1. Header ([Trust rendering](agents.md#trust-rendering)).
2. The agent manual ([Agent manual](agents.md#agent-manual)), on `startup`, `clear` and `compact` only.
3. **Assertions:** the operator's assertions and operator-accepted agent assertions (decisions, lessons, handoffs, problems, goals, questions, findings) scoped to the repository, the directory and the listed sessions. Unjudged agent, script and unattributed assertions are counted, with the query that lists them, and not inlined. The way to have them inlined is judgment: by the operator, or by a calibrated annotation pass.
4. **Other agents working in this repository now** ([Live sessions](agents.md#live-sessions)): worktree, live state, task, last event; the caller excluded; "⚠ 2 agents in this worktree" when a peer shares the worktree.
5. **Observed task claims here:** last-observed claims with no observed close or release, without an age cutoff. Their observation time and unknown external tracker state are labelled; Polylogue does not infer verified current activity.
6. **Recent sessions**, newest first, until the budget is full: title; time and harness; tool-call status; commits, pull requests and files changed; the task it still holds open; the latest observed prose request with its recorded authorship and message reference, before the stored model answer; the first lines of the last answer, datamarked.
7. **Plan and todo list** of the most recent session, as stored, with the reference of the tool call that set it.
8. **Claims** made here since the last brief, with author, judgment state and any check result.
9. **Open items here** ([measures.md#catalog](measures.md#catalog)): todo and plan items left open, sessions ending on an unanswered prompt, calls without results in ended sessions, with what later resolved any of them.
10. **Pending signal notices** addressed to this session or in its directory scope, without an age cutoff; any recorded outcome excludes a notice, including an uncertain or failed send. Records with unknown scope or a historical signal-effect state are counted with an inspection command; no global target or unsent state is inferred ([Notices to sessions](session-control.md#notices-to-sessions)).
11. **Pages here:** the table of contents of pages whose members include this session's role or which target this repository ([pages.md#agents-and-pages](pages.md#agents-and-pages)).
12. **Code may have moved:** "HEAD is N commits past <recorded commit>" or "<recorded commit> is not in HEAD's history".
13. **Machine section**, in primary mode only: read from a stored Sinex context pack under a 100 ms deadline, within a declared share of the budget. Standalone, the section is absent.
14. **Left out:** each group with the command that lists it.

**Latency.** The daemon answers a brief within 300 ms p95 warm and 1 s p95 cold, at any directory (`/` and a directory nobody worked in included); the hook allows 1.5 s including connection. A brief's reads are bounded by what it reports, not by how much work happened under its directory:
- **Scope index.** The daemon keeps every main session's working directories, repository, activity, stored message count, live facts and task in memory, read once at start and brought forward from the change feed after each commit. Which sessions are under the directory, which work there now and which hold a task there costs no index read.
- **Session summaries.** What a brief says about a session (its last answer, files changed, commits, pull requests, an unanswered prompt, an unfinished tool call) is composed when the brief is read, from the session's rows through their indexes, so metadata entries need no session-body read. Only fitted expansions read the summary, the latest active nonduplicate prose request and eligible session assertions; direct scoped assertions and pending notices likewise select metadata before reading their fitted bodies.
- **Deadline.** Sections run cheapest first (assertions, other agents, open tasks, recent sessions, notices) and stop a fifth of the remaining time, at most a second, before the request's deadline. A section that meets it is left out with reason `deadline`, the brief names it and the command that prints the whole brief (`polylogue brief --cwd DIR --deadline none`), and the hook injects the sections that are ready. A deadline never yields an empty brief while a section has items.
- **Delivery off the answer path.** The answer does not wait for the writer: the delivery record is queued, the hook hands its `delivered` outcome over within 20 ms and exits, and the writer applies both in order, a repeated request ID once. A full writer queue never makes a brief wait: the record waits in the daemon, in order with its outcome event, and goes to the writer after its next commit. An outcome is recorded even when the hook has already exited. Until a delivery's record commits, the daemon remembers it as the recipient's latest brief, so a `resume` or `compact` brief right after it reports only what changed since.

On a whole-brief timeout, the hook injects one unreferenced line: "Polylogue context timed out; continue with the current task and local instructions." It exits 0. The original preparation is marked `not_sent:timeout` only when the daemon positively confirms its existing queued preparation and its bytes were never emitted. Unconfirmed acceptance remains unresolved; the failure notice does not invent a delivery. If output was emitted but its acknowledgement was lost, its outcome remains unknown for reconciliation, without resending.

**Interfaces:** the hook; `polylogue brief [--cwd DIR] [--session SESSION] [--source startup|resume|compact] [--budget N]`; MCP `brief`.

## Takeover

A prompt (`UserPromptSubmit`, Gemini `BeforeAgent`, Hermes `pre_llm_call`) that contains session references gets a takeover of those sessions. A reference is an origin-prefixed ID, a bare native ID or unique prefix of a parsed origin, or a provider conversation URL, resolved by the one reference parser ([references.md](references.md)).

- **Exact or nothing.** An ambiguous prefix injects its candidates, never a guess. Plain words never trigger a lookup. A session not in the archive injects "session <ref> is not in the archive (yet)", and a provider URL not yet captured also names `polylogue backfill start`. A session never takes itself over.
- **Lineage.** A resumed or continued session is taken over through its newest successor, and the takeover says so.

**Content, in order:**
1. **Goal:** the first prompt, verbatim, labelled with its `written_by`; a session an orchestrator started shows its first message as "written by <parent session>, not the operator".
2. **Effective context at the end:** read through the `context --at` view, so what precedes the last compaction appears as the compaction summary plus what the boundary kept ([reading.md#views](reading.md#views)).
3. **State:** latest todo and plan; what was left open (unanswered prompt, unfinished tool call, open todo, what it waits on, open assertions); open claims with author and judgment state.
4. **Last exchange:** the last operator prompt and the last answer.
5. **Files touched**, most edited first, with branch, recorded commit and how far HEAD has moved.
6. **Assertions** on the session and its lineage.
7. **Failed tool calls** of the last exchange, by their recorded status.
8. **Provider link** for web-chat sessions, so an agent can point the operator at the conversation.
9. **Left out**, with read commands (`polylogue session:<id> read --view dialogue`, `--view actions`, `--view context --at <message>`).

**Several references share one budget per prompt.** One or two sessions each get a full takeover, split evenly. Three or more inject an index: one line per session (title, harness, cwd, last activity, open state, reference) and "`polylogue REF pack takeover` for any of these"; the budget decides how many lines, and the rest are counted. This also bounds prompts that harnesses write themselves, without detecting them.

**Interfaces:** the hook; `polylogue REF pack takeover`; MCP `brief` with a session reference.

## Recall

`polylogue [QUERY] pack recall TEXT` (MCP `brief` with `task`) answers "what do I already know that bears on this". It fuses ranked lists by reciprocal rank fusion ([search.md#recall](search.md#recall)): messages holding every word in prose, every word in tool output, similarity to the whole text when vectors exist, and sessions in the caller's repository.

- An optional query restricts the scope.
- Hits are grouped by session, each with its assertions (author, judgment), its `written_by`, and a datamarked snippet.
- The answer states how it searched (indexes, hits, sessions, more). With vectors off and no exact hit: "no exact matches; semantic search is not enabled, so related work under other wording may exist".
- It pages by cursor with no fixed cap; `--keep` keeps the result as `result:<id>`.
- Recall is not injected at the first prompt; it is added there only when its precision on real first prompts is shown to help.
- The first page answers within the `search` latency class ([interfaces.md#latency-classes](interfaces.md#latency-classes)), from indexes, never by scanning message bodies.

## Deliveries

Every hand-over of prepared content to a model or a person is a **delivery**: briefs, takeovers, recalls, packs, advisories, notices to sessions, pings, live-control acts and chat actions. A delivery records:
- `delivery:<id>`; sender (`hook`, `mcp`, `cli`, `web`, `signal`, `ping`, `chat`); event (`session_start`, `resume`, `compaction`, `takeover`, `recall`, `pack`, `advice`, `notice`, `ping`, `control`, `chat`);
- its computed author; recipient kind and reference (session, operator, web conversation); `cwd`;
- the delivered bytes, `refs`, `left_out`, `content_hash`, and its outcome events.

**Outcomes** (one vocabulary for deliveries and notices):

| Outcome | Meaning |
|---|---|
| `prepared` | no send attempted yet |
| `delivered` | handed over and acknowledged: the hook returned the text, the MCP or CLI answer was returned, the harness command exited 0 |
| `not_sent:<reason>` | `timeout`, `empty`, `failed(<first stderr line>)`, `unsupported`, `session_missing` |
| `unknown` | a send was attempted and its acknowledgement was lost |

**A delivery acknowledges a hand-over.** A harness may persist a tool result outside
its inline output, apply its own output cap, or later compact the context. The
`delivered` outcome alone does not establish that the complete result remains in
the model's current context. The [context view](reading.md#the-context-view)
reconstructs what the captured harness records show. A caller can read any
passage again.

- The delivery reference appears in the delivered text.
- **Uncertain sends are reconciled, never resent.** For `unknown`, the daemon looks for the delivery's reference line in the recipient's transcript and records `delivered` when it finds it.
- High-frequency deliveries (advisories) are recorded once per session, advisory and target.
- Deliveries are user state: they survive rebuilds, travel in `read --to` bundles, and in primary mode go to Sinex as `polylogue.delivery` ([primary-mode.md](primary-mode.md)).

**Delivery use.** A derivation over the recipient session records `used` when a later message written by the recipient's model, or a tool call it made, contains the delivery's ID or one of its `refs`; that message or call is the record of use. There is no window: it runs on the recipient's new messages through the one convergence path ([measures.md#rules](measures.md#rules)). `used` shows the content was referred to, not that it was understood or accepted, and it is never a session link.

**Query:** the `deliveries` unit, with fields `id`, `sender`, `event`, `author`, `recipient`, `recipient_kind`, `cwd`, `outcome`, `used`, `at`, `bytes`, `refs`. For example `polylogue 'deliveries event:session_start outcome:prepared since:1d'` lists lost briefs.

## Advisories

One kind of advisory exists: a prompt naming a file path (a token with `/` or a file extension) that an operator-judged decision or correction targets gets one pointer line.

- Only current, unexpired decisions and corrections with the operator’s latest judgment of that revision accepting them trigger it. Archived assertion text is not injected: the pointer names the file and a ready `read` command.
- A relative file target requires an explicit matching cwd or repository scope. Absolute targets match normalized absolute prompt paths. Selection performs no filesystem or Git probes.
- At most one advisory appears per prompt, with a 10-minute cooldown per recipient session, kept in daemon memory. The receiving session may not have been captured yet: history matching uses its supplied identity, not a captured-session prefix lookup. A hook or MCP advisory preparation’s assertion revision and file target are recorded in its `refs`; that pair is offered only once to that session, including when delivery is unknown. After the cooldown, a different pair may trigger; a newly judged assertion revision is a distinct pair.
- Hooks call the typed `advisory PROMPT --cwd PATH --session SESSION` operation. Its existing delivery event is `advice`. Preparation records `prepared`; only hook output that was written is acknowledged as `delivered`. An uncertain send is not retried.
- Advisories never block: preparation is capped at 20 ms, intersected with the caller’s bound, or the hook says nothing. Cold or incomplete selection also fails open. In primary mode, a pending or uncertain operation in the local producer spool suppresses optional advice until confirmed application removes it; this uses loaded spool metadata without waiting or scanning bodies. Prompts without filepath tokens make no advisory request; explicit archived-session references take precedence and keep takeover behavior.
