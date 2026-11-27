# Agent context

How Polylogue packages context for an agent: the one `brief` verb and its forms, the brief for a directory that hooks inject at session start, the brief of a session (its takeover), of a search (its recall) and of a page (a context bundle), briefs for the operator, budgets, deliveries, and advisories. Briefs use the [trust rendering](agents.md#trust-rendering) of the agent pages and the shared [reading pipeline](reading.md); sending text to a session is `send` ([session-control.md](session-control.md#send)), and continuing a session's work in another harness is `resume --in` ([session-control.md](session-control.md#resume)).

## Brief

One verb packages context under a budget, and the selection's form, never its row count, chooses what is packaged. Help and completion describe the forms as "brief of a session: take it over" and "brief of a search: recall", and `polylogue help takeover` or `help recall` lands on `brief`.

| Line | Form | What is packaged |
|---|---|---|
| `polylogue brief` (`--cwd DIR` for another directory) | the brief for a directory | the operator's notes here, other agents working here now, open work, what recent sessions did, failing checks with no later pass, notices ([below](#the-brief-for-a-directory)) |
| `polylogue SESSION brief`, where SESSION is a session reference, `latest`, `self`, or a row number whose row is a session | the brief of a session: its takeover | goal, effective context, state, the last exchange, files touched, failed calls, what it left open and what was left out ([below](#the-brief-of-a-session)) |
| `polylogue 'WORDS' brief`, where the query holds only search words or quoted phrases (no field, unit or transform), a single plain word included | the brief of a search: its recall | what is already known: hits from several ranked searches fused, grouped by session, with their annotations and judgments, the operator's decisions first ([below](#the-brief-of-a-search)) |
| `polylogue page:NAME brief` | the brief of a page | the page's contents view, open items and decision box, then its text with its citations and its slots' targets in budgeted depth; a page from the `bundle` template is a named context bundle ([pages.md](pages.md#pages-elsewhere)) |
| `polylogue SELECTION brief`, any other selection | the brief of rows | what happened in these rows, decisions and findings, open ends, their annotations and judgments |

- **The first line of every brief names its form**: `brief of a session: takeover of codex:019d0f2a`. A session chosen by a query rather than a reference gets the brief of rows, and its footer offers the takeover line (`polylogue codex:019d0f2a brief`), so the form never depends on how many rows matched.
- **`--for SESSION`** names who will receive the brief, so what that session has already been delivered is left out. A context bundle is built for its recipient the same way: `polylogue page:release-context brief --for codex:019d0f2a`. **`--cwd DIR`** sets the directory of the first form and the ranking of the third (default the current directory). **`--budget N|none`** sets the budget ([Budgets](#budgets)). **`--to`** writes the brief to a destination such as the clipboard ([reading.md](reading.md#formats-and-destinations)): `polylogue codex:019d0f2a brief --to clipboard`.
- **`--for operator`** writes the brief for the operator to read rather than for a model ([Briefs for the operator](#briefs-for-the-operator)).
- **Briefs a caller asks for itself record no delivery.** The hooks' injected briefs (session start, a prompt naming an archived session, compaction) are deliveries ([Deliveries](#deliveries)). Handing a brief to another session is `send`, and starting a briefed session in another harness is `resume --in`.
- Over MCP, `brief` is a step of every tool; with no selection it needs `cwd`:

```json
polylogue_read {"arguments": {"steps": [{"verb": "brief", "cwd": "/home/op/morrow"}]}}
polylogue_read {"arguments": {"selection": "codex:019d0f2a", "steps": [{"verb": "brief"}]}}
polylogue_read {"arguments": {"selection": "flaky attachment test", "steps": [{"verb": "brief"}]}}
polylogue_read {"arguments": {"selection": "page:release-context", "steps": [{"verb": "brief", "for": "codex:019d0f2a"}]}}
```

### What every brief holds

- passages as written, with the annotations on them and each one's author and judgment state;
- open questions and problems, and whatever a current annotation `blocks`;
- **contrary annotations:** current operator or operator-accepted agent annotations with a `challenges` or `corrects` link to an included item come before its text; other authors' contrary annotations are counted with references and the lines that read them, without their bodies;
- the repository state of the sessions it covers ("HEAD is N commits past <recorded commit>");
- **the left-out list**, last: what is whole, what is shortened, what can be read further and with which line, what was left out and why.

A brief for continuing work lets its recipient recover the goal, the constraints, the approaches already rejected, the current candidate, the limits of what is known and the next useful check.

**Fitting.**
- Figures (counts, totals) are computed over every candidate before fitting, so they do not depend on the budget.
- Candidate entries are fitted across all sections before any passage is expanded. Each entry carries its references and source outside a JSON-quoted source prefix, on a datamarked `┃` line. Useful first-line prefixes share the remaining space before further expansion in stable order; a last expansion is shortened on a UTF-8 boundary, with its quote closed. The left-out list counts whole, shortened and omitted candidates and names the lines that read them.
- A section whose read failed or ran past its deadline renders "not available (reason)", never "none".
- **Stable prefixes:** section order is fixed, items within a section are ordered by a stable key, and the skill file comes first, so a later delivery to the same session adds a difference and earlier text keeps its prompt-cache reuse.
- Large relevant documents are legitimate context; a table of contents is never mandatory.
- No model calls.

### The brief for a directory

`polylogue brief` packages what an agent starting work in a directory needs. The `SessionStart` hook (and Hermes's first `pre_llm_call`) injects it, by source:

| Source | Content |
|---|---|
| `startup`, `clear` | the brief for the payload's `cwd` |
| `resume` | what changed since the last session-start, resume or compaction delivery this session received with outcome `delivered` |
| `compact` | the session's own brief: every prompt its operator typed (`written_by: human`), verbatim and never cut; the latest todo list and plan; annotations on the session; failures not later fixed; what the compaction dropped ([sessions.md](sessions.md#compaction-boundaries)) |
| `fork` | nothing |

A directory with no recorded sessions says `No session is recorded for this directory: DIR.` This is distinct from a scope index that has not loaded.

**"Here" is the repository, not the directory:** the current directory, its repository root, every worktree sharing its Git directory, the project home, and every repository name recorded for sessions there. Sessions in sibling worktrees are labelled with their worktree.

**Sections, in order.** Each item carries its reference and age (`3 d ago`); each section names what it left out and the line that shows it.

1. Header ([trust rendering](agents.md#trust-rendering)).
2. The skill file ([agents.md](agents.md)), on `startup`, `clear` and `compact` only.
3. **Annotations:** the operator's annotations and operator-accepted agent annotations (corrections, decisions, lessons, handoffs, problems, goals, questions, findings) scoped to the repository, the directory and the listed sessions. Unjudged agent, script and unattributed annotations are counted, with the query that lists them, and not inlined; judgment is what inlines them ([judging.md](judging.md)).
4. **Pages in full:** the text of pages here whose `brief` is `full` ([below](#pages-here)).
5. **Other agents working in this repository now** ([live sessions](live-sessions.md#live-state)): directory, live state and wait, task, uncommitted-edit count, last event; the caller left out; "⚠ 2 agents in this worktree" when a peer shares the worktree.
6. **Observed task claims here:** last-observed claims with no observed close or release, without an age cutoff. Their observation time and the unknown state of the outside tracker are labelled; Polylogue does not infer that the work is still active.
7. **Recent sessions**, newest first, until the budget is full: title; time and harness; tool-call status; commits, pull requests and files changed; the task it still holds open; the latest observed prose request with its recorded authorship and message reference, before the stored model reply; the first lines of the last model message, datamarked.
8. **Plan and todo list** of the most recent session, as stored, with the reference of the tool call that set it.
9. **Claims** made here since the last brief, with author, judgment state and any check result.
10. **Methods that apply here:** method annotations whose conditions all hold in this repository (`applies:here` is `applicable`), each with its name and the line that reads it; unassessed ones are counted with the query that lists them ([annotations.md](annotations.md#memory)).
11. **Pages here:** the listing of pages here whose `brief` is `contents` ([below](#pages-here)).
12. **Open items here** ([measures.md](measures.md#catalog)): todo and plan items left open, sessions ending on an unanswered prompt, calls without results in ended sessions, with what later resolved any of them.
13. **Pending notices** addressed to this session or in its directory scope, without an age cutoff; any recorded outcome removes a notice, including an uncertain or failed send. Records with unknown scope or an older signal state are counted with the line that reads them; no global target or unsent state is inferred ([notices to sessions](session-control.md#notices-to-sessions)).
14. **Code may have moved:** "HEAD is N commits past <recorded commit>" or "<recorded commit> is not in HEAD's history".
15. **Machine section**, in primary mode only: read from a stored Sinex context under a 100 ms deadline, within a declared share of the budget. Standalone, the section is absent.
16. **Left out:** each group with the line that lists it.

### Pages here

The pages that bear on a session are those whose `scope` links name this repository or directory, and those that name the session's role as a member ([pages.md](pages.md#members)). Agents use a page when it is in their scope or named to them. Each page's `brief` setting (`note --as page --brief full` when the page is made, or the `brief` key of its [working copy](page-working-copy.md#the-format); [pages.md](pages.md#pages-elsewhere)) decides how it enters the brief for a directory:

- **`full`**: the page's text goes in as section 4, after the operator's annotations and before recent sessions, with each paragraph's citations as footnotes and the [trust rendering](agents.md#trust-rendering) of authored text. Text an agent wrote follows the rule of section 3: an accepted revision is inlined, and where a later revision is proposed the accepted text is carried with `1 proposed change`; text with no accepted revision is counted with the line that reads the page. A page that does not fit the budget is given as its contents view plus the line that reads it whole (`polylogue page:polylogue/how-we-land read`).
- **`contents`** (the default): one line in the Pages here section (name, title, template, last change), followed by its open items, each with its reference, what changed on the page since this role last looked, `N sources changed` when cited sources changed ([page-text.md](page-text.md#when-a-source-changes)), and the line that reads the page's contents (`polylogue page:flaky-ci read --view contents`).
- **`off`**: the page is left out of the brief for a directory; `page:NAME brief` still gives its own brief.

Where the recipient's role wrote text on a page here, the page's line also reports what became of it: `your 6 paragraphs on page:flaky-ci: 4 accepted, 1 rejected, 1 proposed` ([pages.md](pages.md#authorship-and-proposals)). Listed pages are ordered by their last change, newest first, and the budget decides how many are listed; the rest are counted with the query that lists them (`polylogue 'pages where scope:here'`). When no page matches, the block is absent.

**Latency.** Every form of `brief` replies within the brief latency class ([latency.md](latency.md#latency-classes)), at any directory (`/` and a directory nobody worked in included); the hook shares one 1.5 s deadline across Git observation, mailbox/permission requests and context, including connection. Multiple takeover requests share the remaining allowance; an expired deadline starts no further request. Delivery acknowledgements keep their separate 20 ms allowance. A brief's reads are bounded by what it reports, not by how much work happened under its directory:
- **Scope index.** Until its first read finishes, the brief says "scope index has not loaded" and names `polylogue status`; it answers with the sections available and never waits for that read, including with `--deadline none`. The daemon holds every main session's working directories, repository, activity, stored message count, live facts and task in memory, read once at start and brought forward from the change feed after each commit. Which sessions are under the directory, which work there now and which hold a task there costs no index read.
- **Session summaries.** What a brief says about a session (its last model message, files changed, commits, pull requests, an unanswered prompt, an unfinished tool call) is composed when the brief is read, from the session's rows through their indexes, so metadata entries need no session-body read. Only fitted expansions read the summary, the latest active prose request that is not a duplicate, and the session's annotations; scoped annotations and pending notices likewise select metadata before reading their fitted bodies.
- **Deadline.** Sections run cheapest first (annotations, other agents, open tasks, recent sessions, notices) and stop a fifth of the remaining time, at most a second, before the request's deadline. A section that meets it is left out with reason `deadline`, the brief names it and the line that prints the whole brief (`polylogue brief --cwd ~/morrow --deadline none`), and the hook injects the sections that are ready. A deadline never yields an empty brief while a section has items.
- **Delivery off the reply path.** The reply does not wait for the writer: delivery records enter the archive writer's queue without waiting for room. The hook hands its `delivered` outcome over within 20 ms and exits, and the writer commits both in order, a repeated request ID once. An outcome is recorded even when the hook has already exited. A `resume` or `compact` brief reads the recipient's latest delivered brief from the writer's pending deliveries and committed rows, so a brief right after a delivery reports only what changed since.

Every session-start attempt except a fork records whether context was given in a `SessionContext` hook spool event. When none was given, it names the cause: an unreachable daemon, the archive path, timeout, refusal, unusable answer, exchange failure, a missing working directory, empty context or failed output. `polylogue 'events where kind:hook_session_context'` lists these events with their details. Status counts starts without context by cause since the last start given context, and names the earliest failure in that interval. A wrong archive path is reported as such, rather than as a stopped daemon. When a spool write fails, one native journal entry names the failed write ([capture](capture.md#hooks)).

Status counts delivery outcomes in the archive writer's pending deliveries. Shutdown stops producers and the writer finishes every accepted write before it stops ([operations](operations.md#the-daemon)).

On a whole-brief timeout, the hook injects one unreferenced line: "Polylogue context timed out; continue with the current task and local instructions." It exits 0. The preparation is marked `not_sent:timeout` only when the daemon confirms its queued preparation and its bytes were never emitted. An unconfirmed acceptance stays unresolved; the failure line does not invent a delivery. If output was emitted but its acknowledgement was lost, its outcome stays `unknown` for reconciliation, without resending.

The hook asks for the brief through `internal hook` with the payload's source, directory and session, so the recipient is the `--for` session.

### The brief of a session

A prompt (`UserPromptSubmit`, Gemini `BeforeAgent`, Hermes `pre_llm_call`) that contains session references gets the brief of those sessions injected. A reference is an origin-prefixed ID, a native ID or a unique prefix of at least eight characters, or a provider conversation URL, resolved by the one reference parser ([references.md](references.md)).

- **Exact or nothing.** An ambiguous prefix injects its candidates, never a guess. Plain words never trigger a lookup. A session not in the archive injects "session <ref> is not in the archive (yet)"; a provider URL not yet captured also names `polylogue import chatgpt` (or the provider's name). A session never receives its own takeover.
- **Lineage.** A resumed or continued session is taken over through its newest successor, and the brief says so.
- **Prompt times.** The prompts section opens with `Times are in ZONE.`, naming the logical-day zone. Each prompt carries its stored message time with the offset at that instant, or `unknown time`, in session order even when the clock moves backward.

**Content, in order:**
1. **Goal:** the first prompt, verbatim, labelled with its `written_by`; a session an orchestrator started shows its first message as "written by <parent session>, not the operator".
2. **Effective context at the end:** read through the [context view](transcripts.md#the-context-view), so what precedes the last compaction appears as the compaction summary plus what the boundary kept.
3. **State:** latest todo and plan; what was left open (unanswered prompt, unfinished tool call, open todo, what it waits on, open questions); open claims with author and judgment state.
4. **Last exchange:** the last operator prompt and the last model message.
5. **Files touched**, most edited first, with branch, recorded commit and how far HEAD has moved.
6. **Annotations** on the session and its lineage.
7. **Failed tool calls** of the last exchange, by their recorded status.
8. **Provider link** for web-chat sessions, so an agent can point the operator at the conversation.
9. **Left out**, with the lines that read it (`polylogue codex:019d0f2a read --view dialogue`, `polylogue codex:019d0f2a read --view changes`, `polylogue codex:019d0f2a read --view context`).

**Several references in one prompt share one budget.** One or two sessions each get a full takeover, split evenly. Three or more inject an index: one line per session (title, harness, directory, last activity, open state, reference) and "`polylogue REF brief` for any of these"; the budget decides how many lines, and the rest are counted. This also bounds prompts that harnesses write themselves, without detecting them.

### The brief of a search

`polylogue 'WORDS' brief` replies to "what is already known that bears on this". It fuses ranked lists ([search.md](search.md#recall)): messages holding every word in prose, every word in tool output, similarity to the whole text when vectors exist, and sessions in the `--cwd` repository.

- Hits are grouped by session, each with its annotations (author, judgment), its `written_by`, and a datamarked snippet; the operator's decisions come first.
- The reply states how it searched (indexes, hits, sessions, more). With vectors off and no exact hit it says so, in the words of [search.md](search.md#recall).
- It pages by cursor with no fixed cap.
- The brief of a search is not injected at the first prompt; it is added there only when its precision on real first prompts is shown to help.
- The first page arrives within the search latency class, from indexes, never by scanning message bodies.

### Briefs for the operator

`polylogue brief --for operator` (and `--for operator` on any form) writes the same brief for the operator to read in a terminal or the web rather than for a model to take in:
- no skill file, no trust header and no datamarking of quoted text; the wording addresses the operator, and items read as a list to act on;
- nothing is left out as already delivered, and no left-out list of agent context is printed; each shortened section ends with the line that reads it whole;
- unjudged agent annotations are listed, not only counted, each with its `judge` line, since the operator is who judges them; questions waiting on the operator and held web-chat turns come first, each with its `answer` line;
- no delivery is recorded.

## Budgets

A budget is a request field (`budget_tokens` on the socket, HTTP and MCP, `--budget N` on the CLI), with a default per form; the defaults are in [reading.md](reading.md#paging-and-budgets). They are product values, not settings, adjusted by measured exploration saved. The startup manual and protected operator prompts at compaction count toward the budget and stay whole. If the required header cannot fit, the brief asks for a larger budget. The hook's brief of several sessions named in one prompt shares 10,000 tokens among them.

A missing budget uses the form's default. JSON `budget_tokens: null` or CLI `--budget none` lifts the token limit; a number, zero included, stays explicit. A budget too small for the required header, left-out list and one candidate entry is refused.

Each harness's inline cap is declared with its hooks. A brief larger than the cap is injected as whole sections up to the cap, then a line naming the line that prints the whole delivery (`polylogue delivery:01JD9K2P read`); the delivery stores every byte, and text is never silently cut.

## Deliveries

A **delivery** records each time prepared content is handed to a session or a person: the hooks' injected briefs, advisories, notices to sessions, sent messages, live-session acts, the first turn of a session Polylogue starts, and turns written into web chats. A delivery records:
- `delivery:<id>`; sender (`hook`, `mcp`, `signal`, `send`, `act`); event (`session_start`, `resume`, `compaction`, `session_brief`, `search_brief`, `signal`, `advice`, `message`, `notice`, `control`, `first_turn`);
- its computed author; recipient kind and reference (session, role instance, web conversation, or the operator when a brief names no recipient session; the operator has no reference); `cwd`;
- the delivered bytes, `refs`, `left_out` (each an `item` and its `reason`), `content_hash`, the read position it was prepared at (`prepared_position`), and its outcome events (each with its `outcome`).

**Outcomes** (one vocabulary for deliveries and notices):

| Outcome | Meaning |
|---|---|
| `prepared` | no send attempted yet |
| `delivered` | handed over and acknowledged: the hook returned the text, the hosted connection accepted it, the harness command exited 0 |
| `failed(<first stderr line>)` | a handed-over run that then exited unsuccessfully, recorded after `delivered` |
| `not_sent:<reason>` | `timeout`, `empty`, `failed(<first stderr line>)`, `unsupported`, `session_missing`, `session_ended`, `session_stale`, `session_not_live`, `directory_unknown`, `directory_missing(DIR)`, `already_answered` |
| `unknown` | a send was attempted and its acknowledgement was lost |

**A delivery acknowledges a handing over.** A harness may store a tool result outside its inline output, apply its own output cap, or later compact the context. The `delivered` outcome alone does not show that the whole text remains in the model's current context. The [context view](transcripts.md#the-context-view) reconstructs what the captured harness records show, and a caller can read any passage again.

- The delivery reference appears in the delivered text.
- **Uncertain sends are reconciled, never resent.** For `unknown`, the daemon looks for the delivery's reference line in the recipient's transcript and records `delivered` when it finds it.
- A delivery records the archive write counter of the read its content was prepared from (`prepared_position`), and the next brief's delta starts there, whether the delivery is still queued or committed: annotations committed after the brief was prepared remain in the next brief. A delivery recorded without it (written before it was kept) starts its delta at its own write counter.
- High-frequency deliveries (advisories) are recorded once per session, advisory and target.
- Deliveries are user state: they survive rebuilds, travel in `read --to` bundles, and in primary mode go to Sinex as `polylogue.delivery` ([primary-mode.md](primary-mode.md)).

**Delivery use.** A derivation over the recipient session records `used` when a later message written by the recipient's model, or a tool call it made, contains the delivery's ID or one of its `refs`; that message or call is the record of use. There is no window: it runs on the recipient's new messages through the one derivation path ([measures.md](measures.md#rules)). `used` shows the content was referred to, not that it was understood or accepted, and it is never a session link.

**Query:** the `deliveries` unit, with fields `id`, `sender`, `event`, `author`, `recipient`, `recipient_kind`, `cwd`, `outcome`, `used`, `at`, `bytes`, `refs`. For example `polylogue 'deliveries event:session_start outcome:prepared since:1d'` lists lost briefs.

## Advisories

One kind of advisory exists: a prompt naming a file path (a token with `/` or a file extension) that an operator-judged decision or correction targets gets one pointer line.

- Only current, unexpired decisions and corrections whose latest operator judgment of that revision accepts them trigger it. Annotation text is not injected: the pointer names the file and a ready `read` line.
- A relative file target needs an explicit matching directory or repository scope. An absolute target matches a normalized prompt path token, a relative token being resolved against the prompt's working directory first. Choosing an advisory makes no filesystem or Git probes.
- At most one advisory appears per prompt, with a 10-minute cooldown per recipient session, held in daemon memory. The receiving session may not have been captured yet: history matching uses its supplied identity, not a lookup of captured sessions. The annotation revision and file target of an advisory are recorded in its delivery's `refs`; that pair is offered only once to that session, including when delivery is unknown. After the cooldown a different pair may trigger; a newly judged revision is a different pair.
- Hooks call the hidden `internal advisory` operation with the prompt, directory and session ([interfaces.md](interfaces.md#operation-table)). Its delivery event is `advice`. Preparation records `prepared`; only hook output that was written is acknowledged as `delivered`. An uncertain send is not retried.
- Advisories never block: preparation is capped at 20 ms, intersected with the caller's bound, or the hook says nothing. Cold or incomplete selection also fails open. Prompts without file path tokens make no advisory request; archived-session references take precedence and get the brief of a session.
