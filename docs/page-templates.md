# Page templates and shipped pages

A template gives a new page its sections and query cells; shipped pages are the pages every archive starts with. Investigations and experiments are pages from their templates. The page model is [pages.md](pages.md).

## Templates

A template is a page named `template/<name>`. Instantiating it (`--template T` on `note --as page` or `save page:NAME`, or `template:` in a new working copy's header) copies its heading and query slots into the new page, each copy linked `derived_from` its template slot at that revision, and records the template's exact revision on the new page. Copies never follow later template edits, so there is one source of structure per page. **Parameters** are the saved-query parameters ([saved-queries.md](saved-queries.md#query-objects)): a template declares `params: {repo: null, since: "30d"}`, its query slots use `$repo` and `$since`, and a page binds them with `--param repo=morrow`. Empty sections render `empty — add`; a filled Checks section is not a test result.

## Shipped pages and templates

Shipped pages and templates are built into the binary and resolved by name, with no write at daemon start. Revising one writes an ordinary page under the same name, which then wins; the built-in stays readable as `page:NAME@builtin`.

| Template | Sections (t: text written by people or agents; q: live query cell; k: kept) |
|---|---|
| `investigation` | Goal (t), Assumptions (t), Alternatives considered (t), Plan (t), Evidence (excerpts, q), Findings (t), Open questions (t), Decision (the decision box) |
| `decision-record` | Context (t, excerpts), Options (t), Decision (the decision box), Consequences (t), Revisit when (the decision's `reverse_if` check, shown holding or not) |
| `plan` | Goal, Assumptions, Alternatives considered, Plan, Steps, Checks, Expected artifacts, Decision |
| `project-understanding` | Overview (t), Components (t), Decisions (q: decisions scoped to `$repo`), Open questions (q), Recent work (q: sessions in `$repo` since `$since`) |
| `guide` | Summary (t), Steps (t), Rules (t, each citing the ruling it states), Exceptions (t), Changes (q: revisions of this page's text); made for `brief: full` |
| `report` | Question (t), Method (t, the queries as cells), Results (k: figures, charts), Findings (t, cited), Caveats (t: coverage, unknowns) |
| `comparison` | Question (t), Answers (k: the compared rows), Side by side (`compare` of the kept answers), Verdicts (q: comparisons on those rows), Decision |
| `reading-list` | Items (cards, each `unread` until the reader's seen mark passes it), Notes (t) |
| `day`, `week` | parameter `day` or `week`: Sessions (q, by repository and harness), Decisions (q), Findings (q), Failing checks with no later pass (q), Cost (q, unknown costs hatched), Notices (q), Account (t, cited prose) |
| `project-story`, `feature-story` | `$repo` or `$feature`: per logical week, sessions, what happened (cited lines), decisions, outcomes from operator-sampled passes |
| `experiment` | Hypothesis, Units, Arms, Assignment, Exposure, Outcomes, Analysis, Decision |
| `workspace` | the web's saved tabs and columns ([web.md](web.md#workspaces)) |
| `bundle` | a context bundle: the slots a recipient should read, with the page's title as its purpose and the manifest the brief prints |

| Page | What it is |
|---|---|
| `page:home` | the orientation screen: bare `polylogue` is this page rendered for a terminal ([terminal.md](terminal.md#the-orientation-screen)); sections Waiting, Running, Recent here, Pages changed since you looked (`edits` with `since:seen`), Open items (`open`) |
| `page:index` | every page grouped by name prefix and tag, with recent edits and wanted pages |
| `page:pinned` | what `save page:pinned` collects; pinning a row is saving it here |
| `page:cost` | the Cost view as a page ([web-views.md](web-views.md#cost)) |

Adding a section to a shipped page is an ordinary write that makes the page the operator's: `polylogue questions save page:home --live --section 'Open questions'`.

## Investigations

An investigation is a page from the `investigation` template whose subject is a `question`; it may start as a question and a selection of sessions and messages (on the web, **New page from these**). There is no separate investigation record. Four facts stay separate on it: whether a citation still matches the text it cited (its state), whether something supports a claim (`supports` links), the editorial choice to place it (the slot), and standing acceptance (an accepted judgment of a `lesson` or `method`). An intact citation does not show that a claim is true; it shows what the claim rests on.

## Experiments

An experiment is a page from the `experiment` template, for real comparisons only.

- Its assignment is a `decision` on the page with value `{units: result:<id>, arms, method: list|seeded:<seed>, assignment: {unit: arm}}`, written before any exposure. A revision of the assignment after any assigned unit's exposure is refused.
- Outcomes are `outcome` annotations or judgments per unit. `polylogue 'slots where page:exp-1 | measure experiment'` reports, per arm, the interval `[satisfied/n, (satisfied+unresolved)/n]`, never a point estimate that drops unresolved units ([measures.md](measures.md#experiment)).
- Polylogue's own output uses causal words only about a page with a fixed assignment.

## Pages in use

Each of these is a page from a shipped template, built with the ordinary verbs on the Morrow sample archive.

- **A project's living summary.** `polylogue page:morrow/architecture note 'How Morrow is put together' --as page --template project-understanding --param repo=morrow --scope repo:morrow` makes Overview, Components and the live Decisions, Open questions and Recent work. Asked with `polylogue @reviewer send 'write the Overview and Components from the last month of sessions, citing each claim' --page morrow/architecture --section Overview`, an agent reads `--view contents`, gathers citation lines with `polylogue 'messages where "zone" repo:morrow' read --format citation`, writes the sections into a working copy and checks it in. The page reads with each paragraph footnoted and marked proposed until judged, and the header counts `14 citations · 0 sources changed · 2 need citations · 6 proposals`.
- **A decision record.** `polylogue page:catalog-store edit` on a new name opens a working copy listing the templates; `template: decision-record` gives Context, Options, Decision, Consequences and Revisit when. A footnote that quotes a session is pinned to its exact span on check-in. The decision itself is `polylogue page:catalog-store note 'The catalog stays SQLite' --as decision --section Decision --evidence message:codex:01a20000-0005:msg_01a20000a11`, and the decision box shows it with its revisit check.
- **A report.** `polylogue page:report/first-build note 'First build after the index rewrite' --as page --template report`; figures go on kept (`polylogue 'sessions where repo:morrow since:30d | measure time' save page:report/first-build --kept --as chart --section Results`), and the prose states numbers as values fixed when written ([page-text.md](page-text.md#values-in-sentences)).
- **A guide agents read at session start.** `polylogue page:morrow/how-we-land note 'How changes land in Morrow' --as page --template guide --scope repo:morrow --brief full`. Every session that starts in `repo:morrow` gets its text in the brief, every rule citing the message that set it. An agent that finds it wrong edits it, and its revision shows as a tracked change until the operator accepts it; briefs carry the accepted text with `1 proposed change`.
- **A weekly account.** `polylogue page:week/2026-W46 note 'Week 46' --as page --template week --param week=2026-W46`. The live sections fill from the archive; the Account is prose an agent writes from them, every sentence cited. The operator corrects a sentence with `edit` or rejects it; `polylogue 'pages where template:week'` lists every week, and `polylogue page:week/2026-W46 read --at 2026-11-23` shows the account as it stood.
- **An investigation notebook.** From a search in the workbench, **New page from these** with template `investigation` places the marked hits in Evidence; with the page active, each further `+` adds an excerpt ([web-pages.md](web-pages.md#citing-while-reading)). Agents add findings by markup in their own output (`::finding(page=flaky-ci, section=Findings, evidence=tool:codex:01a20000-0005#call_7): …`).
- **A comparison.** `polylogue '@1,2,3' save page:retry-design --template comparison --section Answers` places three sessions and keeps them as the page's comparison set, shown side by side, with verdicts recorded by `judge`.
- **A reading list.** `polylogue @1-6 save page:reading/agents --as card` places six sessions as cards, each `unread` until the reader's seen mark passes it; `polylogue page:reading/agents read --view changes --from seen` shows what is new.
- **The home page.** Bare `polylogue` is `page:home`; `polylogue questions save page:home --live --section 'Open questions'` adds a section to it.

## Laws

| Law | Property |
|---|---|
| template-params | `project-story` with `--param repo=morrow` and with `--param repo=other` gives two pages whose live slots answer different populations; editing the template changes neither |
| experiment-interval | an arm with 1 satisfied, 1 unsatisfied and 2 unresolved of 4 reports [1/4, 3/4]; revising the assignment after an exposure is refused |
