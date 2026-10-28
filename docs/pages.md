# Pages

A page is a place where work is organised: a group of sessions, messages, assertions, queries and other pages, arranged so the current state of an understanding can be read without scrolling back through transcripts. People and agents both work through pages, and pages are optional: nothing in Polylogue requires one. This page owns the page model, its layouts, members and roles as records, how agents work through pages, time and checkpoints, investigations, experiments, and the ping record. Everything a page holds is an assertion ([assertions.md](assertions.md)); there are no page tables. How a ping is delivered, and how role instances are started, is [agents.md](agents.md#pings); the page views are [web.md](web.md).

## Model

- **A page holds references, not copies.** Its content is the archive itself, shown in place, so a page never goes stale against the sessions it cites and reading it never calls a model.
- **`page`** is an assertion keyed by the page's name, with value `{title, layout, template?}`, one current per name. A page can contain pages.
- **`slot`** is an assertion linked `slot_of page:<root>`. Its `about` link is what it shows: any reference ([references.md](references.md#forms)), including a session, a range of messages, a block, a span, an assertion, a variant, a kept result, a file at a commit, an outside `url:`, a Sinex `sx:` reference or a nested page. Its value is `{section, pos, mode, heading?, query?, caption?, x?, y?, w?, h?}`.
  - A slot shows one of: a reference, a live query (`query` holds query text or `query:<hash>`), a heading, or a nested page.
  - `pos` is a fractional position, so inserting a slot never rewrites its siblings.
  - `mode` is `follow` (show the current revision) or `fixed` (show one exact revision, stored as `@rev`).
  - `x`, `y`, `w` and `h` are read only by the board layout.
- **Edits are ordinary assertion edits.** Editing a slot's text revises the assertion it shows; moving a slot revises the slot; removing a slot withdraws the slot only, and what it showed survives. A move and a text edit by two writers both survive; two incompatible edits of one thing are a conflict ([assertions.md](assertions.md#currency)).
- **Placing changes only the page.** One assertion can appear on several pages.

## Layouts

One set of slots renders in four layouts:
- **linear**: read in `pos` order, like a wiki article;
- **board**: placed by `x`, `y`, `w`, `h`;
- **wiki**: pages linked through slots that show other pages;
- **table**: slots as rows, with their kinds and states.

A live-query slot shows the query's current answer. A slot that cites a kept result shows the stored rows, marked "stored rows, not re-run".

## Members and roles

- **Members.** A `member` assertion has an alias as its key (`@reviewer`), a session as `about`, optionally `role_of role:<root>`, and `member_of page:<root>` when it belongs to one page; without a page it is a global alias. A page-local member can still be pinged from anywhere.
- **Roles.** A `role` assertion has a name as its key and as its value the role's instructions (a SOUL.md-style text, or a `file:` reference) and the harness that runs it (`claude`, `codex`, `hermes`). A session is an instance of a role, or of no role, in which case it is just itself. A role may run several sessions at once.
- **Alias resolution.** An alias resolves to a concrete session when a ping is sent. It follows `continuation` and `resume` links to the newest successor, never across a fork ([sessions.md](sessions.md#session-links)). Rebinding the alias later never retargets an earlier ping.

## Agents and pages

- **Table of contents.** `page:X toc` is an agent's starting point on a page, the way `AGENTS.md` is for a repository: title, members, each slot with its reference and one line (title, kind, author, state), the page's open items, and the commands that read further. It is generated on request, never maintained by hand.
- **Check-out and check-in.**
  - `page:X checkout DIR` writes the page and its slots' assertion texts as Markdown files, each with a header naming the object IDs and their base revisions;
  - `checkin DIR` turns changed files into revisions with those bases; a file whose base moved is refused as `stale`, and the others commit;
  - a deleted file is not a withdrawal unless `--withdraw-missing` is given;
  - the files are a working copy, not a second store.
- **Agent edits apply directly,** attributed and reversible, with no proposal step, because agents have the same powers as the operator ([assertions.md](assertions.md#authorship)).
- **Scaffolds.** At the start of a task an agent can set up a page's sections and fill them while it works, by `assert` writes or by markup in its own output ([assertions.md](assertions.md#markup)); markup aimed at a page appears there live, and success is silent. Templates are pages: `assert page NAME --template T` copies the template's heading slots. Shipped templates:
  - `investigation`: Goal, Assumptions, Alternatives considered, Plan, Evidence, Findings, Open questions, Decision;
  - `experiment`: Hypothesis, Units, Arms, Assignment, Exposure, Outcomes, Analysis, Decision;
  - `project-understanding`: Overview, Components, Decisions, Open questions, Recent work;
  - `plan`: Goal, Assumptions, Alternatives considered, Plan, Steps, Checks, Expected artifacts, Decision.

  Empty sections render as "empty — add". A filled "Checks" section is not a test result.

**Page verbs:**

| Verb | Acts on | Arguments |
|---|---|---|
| `add REF…` | a page | `--section`, `--fixed`, `--query Q`, `--heading H`, `--pos P`, `--x`, `--y` |
| `move POS` | a slot | — |
| `withdraw` | a slot | `--reason` |
| `toc`, `checkout DIR`, `checkin DIR`, `keep`, `analyze` | a page | `--withdraw-missing`; `--at T` for `read` |
| `ping TEXT` | a session or alias | `--page`, `--about REF`, `--reply-to` |

## Time

- `page:X read --at T` renders the page from the revisions current at T, through the as-of read of [assertions.md](assertions.md#currency).
- **Checkpoint.** `page:X keep` writes one page revision in which every `follow` slot is fixed at its current revision and every live-query slot is replaced by a newly kept result. `page:X@<rev>` then shows exactly what was there.

## Investigations

An investigation is a page made from the `investigation` template whose subject (a slot in section `subject`) is a `question`; it may start as a question plus a useful selection of sessions and messages. There is no separate investigation record. Four facts stay separate on it: whether a reference still resolves to the text it cited, whether something supports a claim (`supports` assertions), the editorial choice to place it (the slot), and standing authority (an accepted judgment of a `lesson` or `method`). An intact reference is not evidence that a claim is true.

## Experiments

An experiment is a page made from the `experiment` template, for real comparisons only.
- Its assignment is a `decision` on the page with value `{units: result:<id>, arms, method: list | seeded:<seed>, assignment: {unit: arm}}`, written before any exposure. A revision of the assignment after any assigned unit's exposure is refused.
- Outcomes are `outcome` assertions or judgments per unit. Unresolved units stay in the denominator: `page:X analyze` reports, per arm, the interval `[satisfied/n, (satisfied+unresolved)/n]`, never a point estimate that drops unresolved units.
- Polylogue's own output uses causal words only about a page with a frozen assignment.

## Ping records

A ping is an explicit address with text. It is how one session, or the operator, asks another session to do something.

- **A ping is an assertion** of kind `ping`: `about` is the recipient session (resolved from an alias when sent), `scope` is the page it came from, and its value is `{text, subject?, reply_to?}`, where `subject` is an exact reference and `reply_to` a page and section. Its author is the sender.
- **Typing an exact address sends it.** `@[<reference or @alias>] text` in a message written by the operator or an agent is a ping; so are the CLI `ping` verb, the MCP operation and the web composer. A bare `@reviewer` without brackets is ordinary text, so nobody pings by accident.
- **Once per ping.** A ping written as markup has a derived ID ([assertions.md](assertions.md#markup)), so re-parsing, re-importing, quoting or translating the text never sends it again; deliberately sending the same text again is a new ping.
- **Delivery and replies** (live state, resume, starting a role instance, queueing, the reply link) are [agents.md](agents.md#pings). Each attempt is recorded as a delivery.

## Laws

- **page-holds-references**: the Morrow investigation page built from the template, with a message slot, a kept result, a live query and a `::finding(page=…, section=Findings)` marker, stores one page assertion, heading slots, reference slots and the finding, and no copied text.
- **slot-edits**: moving a slot changes one slot revision; removing the Evidence slot keeps the assertion it showed.
- **checkout-checkin**: `checkout`, editing one file and `checkin` writes one revision; when that file's base has moved, it is refused `stale` and the other files commit.
- **page-as-of**: `page:X read --at T` before the finding was written shows Findings empty.
- **page-checkpoint**: after `page:X keep`, `page:X@<rev>` reproduces the page exactly after later edits to its slots and the archive.
- **toc-generated**: `page:X toc` lists every slot with its reference and one line, and changes when a slot is added, with no stored table of contents.
- **alias-follows-successor**: a member alias whose session was resumed resolves to the resumed session, never across a fork, and an earlier ping keeps its original recipient.
- **ping-explicit-address**: `@[@reviewer] text` creates a ping; `@reviewer text` creates none.
- **ping-once**: re-ingesting a message that holds `@[…]` creates no second ping and sends nothing.
- **experiment-interval**: an arm with 1 satisfied, 1 unsatisfied and 2 unresolved units of 4 reports [1/4, 3/4]; revising the assignment after an exposure is refused.
