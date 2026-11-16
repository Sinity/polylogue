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
| `page:home` | the orientation screen: bare `polylogue` is this page rendered for a terminal ([terminal.md](terminal.md#the-orientation-screen)); sections Waiting, Running, Recent here, Pages changed since you looked (`edits since:seen`), Open items (`open`) |
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

## Laws

| Law | Property |
|---|---|
| template-params | `project-story` with `--param repo=morrow` and with `--param repo=other` gives two pages whose live slots answer different populations; editing the template changes neither |
| experiment-interval | an arm with 1 satisfied, 1 unsatisfied and 2 unresolved of 4 reports [1/4, 3/4]; revising the assignment after an exposure is refused |
