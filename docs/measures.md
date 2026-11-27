# Derived data and measures

Everything Polylogue computes from parsed sessions: the rules every derived product follows, the catalog of products, and the measures over them (usage and cost, quota and plans, time, tool behaviour, comparison, reports, rankings, agreement and experiments). The parsed rows these products read are [sessions.md](sessions.md); query grammar and paging are [query.md](query.md), `group by` and `agg` are [transforms.md](transforms.md); kept results, figures and findings are [saved-queries.md](saved-queries.md) and judgment notes [annotations.md](annotations.md), and how rankings and agreement are used in judging [judging.md](judging.md); semantic ranking is [search.md](search.md#semantic-ranking); how figures appear in a read is [reading.md](reading.md); chart and page layout is [web.md](web.md); the brief is [agent-context.md](agent-context.md#brief).

Measurement addresses four questions for someone who runs many agents on subscriptions: how much of the subscription is left and when it runs out; what something cost, and would have cost at API prices; where an agent's time went and how agents behave; what one set of sessions looks like beside another. Every figure is computed from records, never guessed from prose, and says what it counts and what it could not count.

A measure is the last transform of a query, `| measure NAME`, so every measure, report, day and cost figure is a query: `polylogue 'repo:sinex since:30d | measure usage'`. Options follow the name after colons, and the whole is one query scalar: quote an option containing commas, for example `sessions | measure "contrast:origin,words"`. `help transforms` lists the measures with their options.

## Principles

- **The outlook leads for a subscription user.** The current window, its burn and when it runs out come first in status, the overview, the cost page and `| measure outlook`. API-equivalent money follows, labelled as a comparison.
- **Unknown is never zero.** Every sum states its known part and how many items it could not count, in the items' own unit. A partial value is a lower bound ("at least"); a missing time is never placed.
- **One definition per figure, one function per rule.** One cost function prices a response; every other cost is a sum of its results. Surfaces never recompute.
- **Denominators and grain are part of the reply.** Every rate names its denominator, every count names its unit, and figures from different definitions are labelled not comparable.
- **Computed, not interpreted.** Counts, sums, durations, joins and statistics over recorded structure are in scope. Labels inferred from prose (work kind, phases, success, abandonment) are not, because they would claim what the records do not show.
- **No umbrella category.** Users see fields, units, measures and reports. The code's name for a stored or computed product is *derivation*, and the current generated reference page is [Derived data](derived.md).
- **Measurement is the query language plus named computations.** Anything a pipeline can express is a pipeline. A measure exists only for what pipelines cannot express: windows, forecasts, survival, level changes, transitions, ranking.
- **Every number reaches every surface the same way,** through the operation table; `--explain` prints the pipeline behind a report.
- **Precise accounting with existing libraries.** Use established libraries for token counting and cost tracking.
- **Reports are general.** No report is specific to one person's own tools or setup.

## Rules

### Validity

Each product declares one validity kind, which decides the evidence it needs before it ships.

| Kind | What it is | Evidence required before it ships |
|---|---|---|
| **exact** | a count, sum, join or ordering over recorded fields; its name states its definition | properties checked by property tests, and hand-checked values on Morrow |
| **parameterized** | exact once a declared constant is fixed | as exact, plus the constant printed wherever the value is shown, and a sensitivity figure (the value at the constant and at one other value on a real-shaped sample) |
| **extracted** | read from tool output or command text by declared per-tool patterns | coverage (share of candidates recognized) and precision and recall on a hand-checked real sample, with `n` and date, recorded in the declaration |
| **judged** | a label, segment or ranking standing for something the records do not state | agreement with operator-labelled examples or a structural ground truth, with `n`, recorded before shipping; without it the product is not exposed |

Labelled samples are private judgment notes and kept results in the operator's archive; only the figures appear in declarations and docs.

### Grain and storage

Analyses are computed when their data arrives, and at read only when that cannot be done accurately.

- **Stored grains are row and session only.** A stored product is a column on a parsed row (`message`, `tool_call`, `event`), or a session column or session-keyed table.
- **Session-grain values are stored as their data arrives** when a query filters, sorts or groups on them across the archive, or when they are costly per read (graph walks, interval unions).
- **Grouped figures are computed at read** from stored session rows: logical session, day, week, repository, project. Grouping stored rows is exact, so a late import, a re-parse or a newly arrived link changes them with nothing stale to invalidate. A grouped figure measured too slow at read may be stored as well, keyed by session (for example by session and UTC quarter hour), so it changes with its session; nothing is keyed by day, repository or root.
- **Read-time products** are those that depend on the clock (live state, windows, forecasts, "quiet for 6 hours"), on a setting (logical day, plan) or on an arbitrary selection (aggregates, comparisons, rankings), and per-turn or per-pair computations no archive-wide query sorts on.
- **Stored products read only parsed rows,** never annotations, settings or the clock, so a note or a setting change never makes a stored product stale.
- **Storing a product does not make it citable.** Citation is served by [kept results](#citation-and-figures). Whether a product that is neither stored nor a field is best served as a named query is decided per product.

**Two storage forms:**

1. **Row columns,** computed with their row when it is composed, by the same function the derivation uses.
2. **Session columns and session-keyed tables:**

   ```sql
   CREATE TABLE <product> (
     session_pk INTEGER NOT NULL REFERENCES session(pk) ON DELETE CASCADE,
     <natural key columns>,
     <typed value columns>,                 -- NULL means unknown, never 0
     <sum>_unknown INTEGER NOT NULL DEFAULT 0,
     PRIMARY KEY (session_pk, <natural key columns>)
   ) STRICT, WITHOUT ROWID;
   ```

   Typed columns for every value a query filters, sorts or groups on; JSON only for display detail; no per-row copies of constants or versions.

### Convergence

Derived data converges through one path, with no separate pass and no refresh, rebuild or recompute command.

- **Two grains, one mechanism each.** A value of one row (a normalized command, a message's cost) is a function of that row and is written with it, by the same function, whenever the row is composed. A value of one session (its counts, spans, time by party, completion values and first prompt, live facts, active path, duplicates, title, commit and pull-request links, logical-session root, the place of each transcript entry among the others) is a *session summary*: a function of the session's shown rows, computed by the summary worker. There is no third way: no value of a session is kept current by subtracting what a changed row contributed and adding what it contributes now, none is kept by a trigger, and none is computed in the transaction that writes the rows. A value kept by difference is right only while every path that changes a row reports the change; a value computed from the rows is right whenever it is computed.
- **Marks.** The code that assembles a session's rows marks the session due in the same transaction, with the earliest place in the session's row order that its changes reached, before or after them; a summary whose version moved marks every session. The worker finds its work through the marks, never by scanning sessions. A mark is ordinary pending work: a crash leaves marks behind and the next start works through them.
- **Cross-session causes** mark the sessions they reach in their own transaction: a link change marks the logical-session root of the child's subtree; a changed repository lookup marks the sessions at or under its directory.
- **Folds.** A summary whose value depends on the rows in order is a fold over the session's messages and events in row order (a row's place, then messages before events, then its ID). One ordered read serves every fold: the worker starts from the folds' state stored after the nearest row before the earliest change, reads the rows after it and stores the state again after the last row and at every 512th row, so an append of k rows reads k rows and a change before the end starts from the nearest earlier state. Folding the rows after a state gives what folding every row from the start gives; a fold that cannot continue from its state (a message timed before what its state already settled) starts the session again from its first row. A run reads the session's rows a page at a time, so what it holds does not grow with the session, and a fold's state holds nothing a table already holds: the commit and pull-request links fold finds the links of the rows it reads, and its write replaces only those found after where the run started.
- **The worker.** The summary worker runs beside the writer and never in the writer's transactions. It takes due sessions, newest first, in two waves: first the summaries that mark rows of the session (duplicates, the active path) and those with causes of their own (lineage), whose changed rows move where the folds start; then the folds, the session's own columns from its records and the folds, and its [transcript lines](transcripts.md#transcript-lines) from the last line the earliest change can reach. A session a reader waits for is computed before the next session of the pass in progress. It reads on its own read connection, and the writer only writes the result, and only while the session's rows and marks are as they were when its inputs were read. A session that changed meanwhile keeps its mark and its earlier values and is summarized by the next pass.
- **A build derives as it composes.** In a build of the index a session's rows are together when they are composed, so its derived values are computed then, on every core, and loaded with the rows; what needs other sessions' rows is derived after the load and before the index is installed. Upkeep and a build give the same derived values ([architecture.md](architecture.md#one-path)).
- **Reads of due sessions:** an aggregate counts due sessions as `pending` and never shows their old values as current; a transcript read of one session asks the summary worker for that session and waits for it, bounded by the read's limit, so a single-session reply shows the last commit.
- **Versions:** rule edits bump `Rules(n)`; shipped tables (prices, command classes) are versioned by content hash, so a forgotten bump cannot ship. Changing a derivation's version re-derives exactly that derivation (a fold's, every fold from the session's first row); there is no index-wide version. A golden-output test per derivation fails when output changes without a version change.
- **Re-parse:** a chain whose reading is replaced or removed changes the shown rows of its sessions, which marks them due like any other change.
- Ongoing cost is per changed session, never per archive.

### The declaration

Every derivation declares its construct beside its code, so the code is the catalog:

```rust
struct Summary {
    name: &'static str,
    version: Version,                  // Rules(n) | Content(hash of a shipped table)
    kind: Kind,                        // Row (a value of one row, written with it) | Marks (rows of the
                                       //   session, first wave) | Fold | AfterFolds (from records and folds)
                                       //   | Lines (the transcript lines, second wave)
    construct: Construct,
}
trait Fold {                           // a summary over the session's rows in order
    type State: Serialize + DeserializeOwned + Default;
    const NAME: &'static str;
    const VERSION: i64;
    fn step(state: &mut Self::State, row: &SessionRow, cx: &Context, out: &mut Output) -> Result<Flow>;
    fn end(state: &Self::State, cx: &Context, out: &mut Output) -> Result<()>;
}
struct Construct {
    represents: &'static str,          // what it measures and why someone wants it
    reads: &'static [&'static str],
    unit: &'static str,
    denominator: Option<&'static str>,
    unknown: &'static str,             // how unknown and partial inputs count
    params: &'static [Param],          // printed wherever the value is shown
    validity: Validity,                // Exact | Parameterized | Extracted | Judged
    evidence: &'static [Evidence],     // test names; recorded figures with n and date
}
```

- Read-time fields, units, measures and aggregates carry the same `Construct` in their registries.
- The current [derived-data reference](derived.md) is rendered from the declarations and registries; a test fails when a stored table has no declaration or a declaration has no entry.
- A registration test refuses an `Extracted` or `Judged` declaration without a recorded figure, and an `Exact` or `Parameterized` one without a named test.
- `help fields [UNIT]` and `help --all` serve the catalog to people and agents through the CLI and MCP `help`. Each field's help is its `represents` sentence; its unknown word is its `unknown` rule.

### Citation and figures

- A kept result records the query hash, resolved bounds, rows and the version of every derivation it read. Reading it later shows "computed under prices `<hash>`, now `<hash>`" when they differ.
- A **figure** is a saved `| measure` or `| agg` query with a declared denominator query ([saved-queries.md](saved-queries.md#figures)). Shipped figure definitions (cost, tokens by token kind, tool error rate, turn latency) are saved queries.
- Every figure a measure or report returns names its definition (measure or pipeline text), population, unit, and its unknown, partial, pending and untimed counts.

### The default unit

[Logical sessions](sessions.md#logical-sessions) are the default unit for listings, counts and reports. A report's `sessions` column counts logical sessions, with a `physical` column beside it; per-session products are stored per physical session and grouped by root at read; `--physical` counts physical sessions. Money and tokens sum over every physical session once, whatever the grouping.

## Catalog

Store: **row** (a column on a parsed row), **session** (stored at session grain), **read** (computed at read). Compositions (every form of the brief, and the `overview`, `context` and `tree` views) are not products: they compose these at read and are stored only as deliveries.

| Product | Fields or unit | Store | Validity |
|---|---|---|---|
| Shown version of repeated turns | `duplicate_of`, `duplicate_reason` | row | extracted |
| Active path | `on_active_path` | row | exact |
| Session columns from records | kind, title, directories, branch, harness version, start | session | exact |
| Display title | `title` when none is recorded | session | exact |
| Working directories and repository | `session_directory`; `repo:`, `project` | session | extracted |
| Logical session | `root`, `depth`, `root_pending` | session | exact |
| Session counts | messages, prompts, words, tool calls, errors, … | session | exact |
| Message cost | `cost` on a response | row | exact |
| Session cost | `session_cost`: `cost`, `cost_with_parents`, `tree_cost`, `subagent_cost`, `cost_complete` | session | exact |
| Usage over a selection | `measure usage` | read | exact |
| Responses behind usage | `measure responses` | read | exact |
| Quota readings | `quota_reading` | row | extracted |
| Subscription windows, burn, forecast, spend at past limit hits | `measure quota` | read | parameterized |
| Billing-cycle outlook | `measure outlook` | read | parameterized |
| Daily series, baselines, level changes, rhythm | `group by day(at)`, `measure daily`, `measure rhythm` | read | parameterized |
| Context pressure | `context_tokens` | row, read | exact |
| Cache economics | cache shares and costs | read | exact |
| Tool-call result time and duration | `result_at`, `duration`, `reported_duration` | row | exact |
| Session span | `session_span`: `wall_span`, `clustered_time`, `ends_with`, … | session | parameterized |
| Where the time went | `session_time`; `waiting_model`, `waiting_human`; `measure time` | session | parameterized |
| Turn fields | the `turns` unit | read | exact |
| Concurrency | sessions active at once | read | parameterized |
| Normalized command and command class | `command_norm`, `command_class` | row | extracted |
| Outcome counts, failure-rate bounds, transitions | `agg failure_rate`, `measure outcomes`, `measure transitions` | read | exact |
| Reruns | retry chains, fail-then-pass, edits between checks | read | exact |
| Failing checks | `measure failing` | read | exact |
| Survival | `measure survival` | read | parameterized |
| Operator interventions | row marks and events; sums | row, read | extracted |
| Service interruptions | `api_error` events | row, read | extracted |
| Cost of a failure | per fail-then-pass pair | read | exact |
| Commit and pull-request links | `commit_link`, `pull_request_link`; `commit:`, `pr:` | session | extracted |
| Landed work | landed and not landed commits | read | exact |
| Delegations | per `spawn` link | read | exact |
| Live state | rank, waiting state, open call, task, peers | read | exact |
| Hook coverage | `measure hooks` | read | exact |
| Open items | the `open_items` unit | read | exact or extracted per item kind |
| Repeated instructions | repeated sentences | read | exact |
| Rework | re-edits, re-reads | read | parameterized |
| Related sessions | `related:REF` | read | judged |
| Facet counts with IDF and lift | `measure facets` | read | exact |
| Contrast | `measure contrast` | read | exact |
| Distributions and correlation | `p50`, `p90`, `histogram`, `corr` | read | exact |
| Rankings over comparisons | `measure ranking` | read | parameterized |
| Agreement of a pass | `measure agreement` | read | exact |
| Experiment intervals | `measure experiment` | read | exact |
| Reports | queries joining the above | read | exact |
| Embeddings | vectors in the embeddings store | row | judged |
| Format drift | `measure drift` ([archive.md](archive.md#format-drift)) | read | exact |

Activity sessions, presence, agency, interruption and resumption, cross-day episodes and day narratives are Sinex's higher layer ([Sinex higher-layer](https://github.com/Sinity/sinex/blob/master/docs/higher-layer.md)); Polylogue supplies sessions, task linkage and session spans as its events.

## Session structure

**Shown version of repeated turns.** Two routes without shared turn IDs (a Gemini page capture and its Takeout export) hold one turn twice; without this every count, cost and search hit for those sessions doubles. Prompts match on whitespace-normalized text occurring once in each input, in the same order, with the same roles after; a repeated prompt ("continue") stays shown from both. The export's message is shown; the capture's message names it in `duplicate_of` with `duplicate_reason`. No match means shown. Unit: a message mark. A session derivation; the marked message is excluded from every population. Evidence: precision and recall on 50 matched and 50 unmatched hand-checked turns.

**Active path.** The path from the provider's newest marked leaf to the root. With no valid marker the value is NULL (unknown, read as on the path). Abandoned branches were run and billed, so usage counts both paths while message counts follow the active one. Evidence: Morrow #7.

**Session columns from records.** Kind, provider title, working directories, branch, harness version and start, merged from every input's session record by order-independent rules. The result is independent of input order.

**Display title.** For a session whose records name no title: the first line of the first human prose prompt. The repository is shown beside it, never inside it. NULL when there is no human prose prompt; never written out as a recorded title.

**Working directories and repository.** One row per (session, directory) in first-use order: the directory, the longest repository lookup at or above it (by whole path components), the repository identity (the first of its root commits; the roots are common `git-identity`'s set: every root reachable from `HEAD`, sorted, shallow-clone boundaries left out, the same set the session hook records and Sinex attributes by), a display name, and `project` (the repository name, else the directory under `/<…>/project/<name>`, else NULL). A directory with no lookup has NULL identity and counts as "no repository", never as another repository. A lookup is written only from a read that answered and names a remote or a root: a read that fails, runs past its deadline or names neither leaves the last written lookup in place. `repo:X` matches any directory of the session; `group by project` counts a two-project session under each. Cross-session cause: a new or changed repository lookup. Evidence: coverage on a real archive (share of agent sessions with an identity).

**Logical session.** `root` is the session reached by walking parent links of kind `continuation`, `fork`, `spawn` and `resume` to a session with no such parent; `depth` counts the steps; `reference` links never join. A pending link (the parent is absent or only a stub) leaves root and depth unknown with `root_pending` set. Multiple recorded parent identities and cycles also leave root and depth unknown, with a typed conflict or cycle issue. Alternative branch points to the same parent remain distinct links and have the same structural root and depth. Links come only from source markers ([sessions.md](sessions.md#session-links)). Stored because `group by root` and logical counts run over the whole archive. The store owns one indexed ancestry walk for derivation, the current reader and root/depth query fields. Link mutations and endpoint arrival or stub resolution mark only affected descendants in the same transaction; a read while those marks are pending computes the current value through that same walk. The root is independent of link arrival order; Morrow #2 and #3 share root #2, and #5 and #6 share #5. The physically copied #4 has no recorded provider relationship and remains its own root.

**Session counts.** Messages, human prompts, human commands, words, words per author, thinking messages, compactions, tool calls, errors and calls by action, over the session's own rows. Stored as session columns, because they are fields sorted and grouped across the archive. A threshold such as "substantive" messages is stated in the query (`messages where words>=3`), never stored.

## Cost and usage

### Token kinds and populations

- **Five token kinds** on every response a model wrote, as reported: uncached input; cache read; cache write (with its 1-hour part when reported); output excluding reasoning; reasoning. NULL means not reported, never 0. Reasoning is priced at the output rate; the token total is all five token kinds.
- **Counted tokens where no usage is recorded.** For sessions whose source records no usage (web chats), tokens of the visible text are counted with the tokenizer library for the model's family where one is published, otherwise with the shared [token estimator](reading.md#paging-and-budgets). Counted tokens are their own state (`counted`, shown `≈`), name the tokenizer, are a lower bound of billed input (hidden system prompts and tools are not visible), are never summed with reported tokens without that label, and price only to the `estimated` money kind.
- **Populations,** one per figure, named in the result:

  | Population | Contents |
  |---|---|
  | `own` (default) | each recorded response once across the archive: the shown version only (`duplicate_of` NULL), on and off the active path |
  | `with_parents` | a session's own responses plus those of the sessions it composes from ([reading.md](transcripts.md#composed-transcripts)); per session only, never summed across sessions |
  | `tree` | a session's own responses plus those of every session below it through `spawn` links |
  | `recorded_reports` | usage a source reports without naming its response (Codex `token_count`); its own unit and denominator: validated, unconfirmed and untimed reports |

- Display filters from annotations (tags, `hidden`) never change a population; a measure states how many hidden sessions it includes.
- A source that cannot assign usage to responses makes the response count and response-based totals of that scope unknown; its recorded reports are still summed as a known part.
- **Codex completeness by reconciliation:** `token_count` reports are validated per request (the running total grew by exactly the last counts). A session whose validated reports cover its final running total, with no unconfirmed report, has complete usage and an API-equivalent cost at grain "usage reports"; otherwise it is partial, with the gap in tokens.
- **Time bounds** select response times (`timed_by: source_record_time`), not session activity. Untimed rows are counted as `untimed` and excluded from bounded sums.
- A measure that counts messages (`usage`, `outcomes`, `failing`, `transitions`, `daily`, `rhythm`) counts only the messages inside the query's top-level `since:` and `until:`, and a selected session's figures include its whole logical session; each response still counts once.
- `measure responses` lists the responses `measure usage` sums over the same selection and bounds, one row each (ID, session, time, model, tokens per token kind, API-equivalent and provider-reported cost), in time order with untimed responses last, paged like any listing; `responses:MODEL` narrows the rows to one model and `responses:unknown` to responses naming none.
- `measure usage` returns: population, `timed_by`, sessions, responses (known, and items with unknown boundaries), tokens per token kind (known, and unknown responses), money by kind and by model, unpriced models with counts, recorded reports, the price catalog hash and label, and untimed items.
- A provider's own total for a child (Claude `toolUseResult.totalTokens`) is kept for comparison only and is never added. A token kind the provider does not charge is NULL, adds nothing and does not make the response partial (`n/a`).

### The price catalog

- One catalog ships with Polylogue: the providers' list prices, maintained by hand with a source note per row, as `prices.toml`. The user may override rows in `$XDG_CONFIG_HOME/polylogue/prices.toml`; an override replaces the shipped row with the same model and effective date.
- A row has `name` (normalized model), `effective` (applies at or after), `input`, `cached_input`, `cache_write`, `cache_write_1h`, `output` (USD per million tokens, held exactly as integer pico-USD per token, so `0.01875` is 18 750; a rate finer than that is refused); optional long-context rates above a prompt size; optional modes and tiers; an optional per-search price; and `source` (URL and the date it was read).
- **Model names are normalized per origin,** beside its parser (prefixes, aliases, date suffixes). The catalog matches exact names; a versionless alias (`opus`) stays unpriced, never guessed.
- **The version is the content hash** of the effective shipped table plus overrides; a different hash reprices through the derive path with no re-parse. A human label is shown beside it, never used as the key.
- No row on or before the response time, a tier or mode without a rate, or a cache token kind with tokens but no rate leaves the cost unknown; zero tokens need no rate. Cache writes with no reported lifetime use the 5-minute rate and are counted as `assumed_cache_lifetime`.
- A malformed override file is refused with line and field; the shipped table stays in force and status shows the error. Status and usage reports list every model seen with no price, with counts and newest time.
- The unit `prices` lists the rows with their sources (`polylogue 'prices where name:claude-opus-4-1'`); `--why` on it names the row a model resolves to at a time and why; `status` shows the active hash.

### Money

- **One kind per figure,** never added across kinds:

  | Kind | Meaning |
  |---|---|
  | `api_equivalent` | list price of the recorded tokens; the default cost |
  | `provider_reported` | a total the harness or provider recorded |
  | `subscription_share` | a plan's price for a cycle spread over its responses in proportion to their API-equivalent cost |
  | `manual` | an amount the user entered (a `cost` annotation) |
  | `estimated` | computed from incomplete or counted data, with what it assumed |

- A separate type per kind makes adding across kinds a type error.
- **Integer nano-USD** in storage, every sum and every price computation: a response's cost is the tokens of each token kind times their pico-USD rates and tier multipliers in 128-bit integers, rounded once to the nano-USD. Decimal dollars in input (query literals, provider-reported amounts) convert exactly. JSON carries an amount as `{nusd, usd}`, `usd` being the exact decimal dollars; a daily series of cost counts nano-USD (`unit: nusd`). Rounding for people happens once, after summing, in text and HTML: whole cents from one cent up, three significant digits below a cent, so a known small cost never reads as `$0.00`.
- **Completeness travels with the value:** `{nusd, complete, priced, unpriced}`, one of `complete`, `partial` (a lower bound) or `unknown` (null). Query fields `cost`, `cost_complete`, `unpriced_responses`; `agg sum(cost)` reports `partial_cost` and `unknown_cost`.
- A provider-reported zero is a measured zero. Each response of a multi-model session is priced under its own model. A session view shows API-equivalent and provider-reported side by side, with their difference labelled "not in transcript".

**Message cost.** One function prices a response from its token counts by kind and its model's catalog row at its time. Unknown when any billed token kind is unreported or the model is unpriced. A row column whose version is the catalog hash. Every stored message cost equals the function applied to its usage.

**Session cost.** `own` (the sum of message cost over the own population), `own_unpriced`, `complete` (no unpriced response; for Codex, reconciled), `tree` (own plus every session below through `spawn` links), `tree_unpriced` and `tree_missing_children` (children a spawning call names that are not in the archive, with those calls' references). A partial value shows "at least". A child belongs to exactly one tree. Cross-session cause: a change in a session marks the tree cost of each ancestor. Fields `cost`, `cost_with_parents` (the `with_parents` population: what the session shows, inherited responses included), `tree_cost`, `subagent_cost` (tree minus own, partial if either is) and `cost_complete`; every assistant message also carries its own cost and tokens per token kind as fields. Continuations, forks and resumes are not trees: a logical session's cost is the sum of its sessions' own costs, once each. Stored so that sorting by cost and tree sums are indexed reads.

### Subscription quota

- **Quota readings.** One row per distinct reading: origin, account, limit, window length, used percent, reset time, status, plan, credits, observed time, source row. Sources: Codex `rate_limits` on `token_count` events; Claude Code refusals (`quotaLimits`, 100% only when rejected); Claude Code status-line readings (`rate_limits.five_hour` and `seven_day`) through a status-line hook tap, which is official and local and needs no credential. A reading copied into a forked session is not new (deduplicated by report ID). A relative reset without an observed time cannot be placed. Written with its source row. The status snapshot reads each limit's newest reading within the longest window after each commit, so `status --line` shows it with no read at answer time ([operations.md#status](operations.md#status)).
- **Windows** are computed at read per account and limit: *reported* (a reading gives its reset), *projected* (whole window lengths from the last reported reset, for fixed schedules), *inferred* (a limit that starts at the first response after the previous window ended, labelled `≈`). Window usage: start, end, source, usage by token kind, API-equivalent cost and active sessions; "at least" when it holds unknown usage.
- **Burn and forecast:** percent burn per hour between the first and last readings at least 5 minutes apart; the exhaustion time if before the reset, else "lasts to reset", labelled as a linear projection; token and money burn per hour; projected tokens at reset. A reading lower than the previous one restarts burn; a reset earlier than its reading is stale, ignored and counted; quota at 0% or 100% at the first reading never divides by zero.
- **Spend at past limit hits:** when no reading in the window carries a percent, the API-equivalent spend in earlier windows of the same limit at the moment a rejection was recorded, with `n` and range; absent when `n = 0`. The name says what is measured, not a capacity.
- Each rejected reading is an event (`events where kind:rate_limit`). `| measure quota` lists past windows newest first, with usage, peak percent and whether the limit was hit.
- Untimed reports are listed as `unplaced` and affect only windows they could belong to. Accounts are separated by recorded identity, else stated as one.
- **Lead position:** status and the overview show, per subscribed account, one line per active limit: used percent or tokens, reset, burn, exhaustion or "lasts to reset", with inferred values marked.

### Plans and billing cycles

- **Plan setting** `plan.<provider>` (archive scope, shared across hosts): the plan name in the provider's terms (`pro`, `max-5x`, `max-20x`; `plus`, `pro`; a bare `max` is refused as ambiguous), an optional monthly price (else the shipped plan table, with its source), the cycle anchor day and an optional budget, dated from when it was set; a new value starts a new span. The plan table never states allowances the provider does not publish.
- A session counts against a subscription only when its records show it: a quota reading in its files, or the plan setting in force at its time; otherwise its billing is `unknown`.
- **Cycle outlook:** a calendar month from the anchor day (days 29–31 clamp to the month's end; UTC dates). Projections per unit (API-equivalent, provider-reported, each token kind), each method named: `linear` (used ÷ elapsed × total) and `trailing-7d-mean`. Each carries coverage, missing days and untimed or incomplete usage. Observations after the as-of instant are excluded, so the outlook is reproducible.
- **Subscription value:** per cycle, API-equivalent cost against the plan price, as a ratio; `subscription_share` derives from it. An optional user budget gives used and projected ratios, a breach day and overage at a user rate, labelled "your budget". A plan changed mid-cycle splits the cycle.
- Interfaces: `| measure outlook[:PLAN[:METHOD]]`, the cost report, the web cost page.

### Daily series and rhythm

- A daily series of any summable metric over the one logical day (the shared `DaySetting`, [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md)): logical sessions, physical sessions, prompts, responses, tool calls, tool errors with denominator, tokens per token kind, API-equivalent cost, recorded-report tokens and cost, clustered time, or any field through `group by day(at) | agg sum:F`.
- Each day carries `complete`, `unknown_items` and `untimed_items` for that metric; untimed items go to an `unknown day` row. A session crossing midnight contributes by each item's own time.
- `baseline_7d` is computed only over 7 complete previous days. **Level changes** are found by binary segmentation with a BIC-style penalty, minimum segment 7 days, over runs of at least 14 complete days, each with means before and after, and labelled "level change", never "trend".
- **Rhythm:** a 7 × 24 grid by logical weekday and clock hour, of any countable unit.
- Calendar and this-day views read the same series. Nothing per day is stored, so a setting change re-buckets at the next read. A 23- or 25-hour day is one day. One daily path serves outlooks, `| measure daily`, the web cost and calendar pages and reports by day. A day's activity and usage are queries with a time condition (`polylogue 'sessions since:today | measure usage'`); the same day across years uses `month_day` ([query.md](query.md#units-and-fields)). The records of a month by logical day are `polylogue 'messages where since:2026-11-01 until:2026-11-30 active:* | group by day(at) | count'` (any timed unit: `events`, `annotations`); the known cost of a selection per day is `| measure daily:api_cost`; the responses behind a selection's usage, with their tokens and cost, are `polylogue 'repo:morrow since:2026-11-09 until:2026-11-09 | measure responses'`, narrowed to one model with `responses:MODEL` or to responses naming none with `responses:unknown`; records with no representable time are `messages where at:unknown`.

### Context pressure and cache economics

- **Context pressure.** `context_tokens` per response is uncached input + cache read + cache write, as reported (a row column; NULL when a token kind is unreported). Per session at read: the maximum, the series, responses above declared fractions of the model's context window (from the catalog), compactions with tokens before and after (from the boundary event), and the operator prompts before each compaction. Evidence: Morrow #4 (182,000 → 14,500).
- **Cache economics.** Per selection: the cache-read share of input tokens; the cost of cache writes (5-minute and 1-hour) and of the reads that followed in the same session; reads per write. Labelled API-equivalent, since on subscriptions cache reads are not billed.

## Time

**Tool-call result time and duration.** `result_at` is the time of the message holding the call's result; `duration = result_at − call time` when both exist and the result is strictly later, else NULL with reason `same_timestamp`, `result_before_call` or `no_result`. `reported_duration` is the harness's own figure, kept apart. A background call ends at the notification naming its task. Times from two clocks are never subtracted unless the source declares them one clock. Row columns, written when the writer pairs the result with its call.

**Session span.** One stored row per session:

| Column | Definition | Unknown |
|---|---|---|
| `first_at`, `last_at`, `time_source` | earliest and latest own message time (events when there are no messages), with time quality | NULL; never the import time |
| `wall_ms` | `last_at − first_at` | NULL with no timed item |
| `clustered_ms` | union of intervals between consecutive own active-path messages at most 5 minutes apart, extended across any tool call to its result | NULL with fewer than two timed messages; `unordered_gaps` counted |
| `tool_sum_ms`, `tool_union_ms` | sum and union of tool-call durations | `tool_untimed` counted |
| `first_failure_at`, `prompts_before_failure` | first tool call with outcome `error`; human prose prompts up to it | NULL when none failed |
| `first_check_pass_at` | first `ok` run of a `test`-class command | NULL |
| `ends_with`, `last_stop_reason` | kind of the last own item: `human_prompt`, `human_command`, `model_text`, `tool_call_open`, `tool_result`, `harness`; the last model message's recorded stop reason | `unknown` for an empty or untimed session |

`ends_with` describes the last item and never claims success, failure or abandonment. Only times of quality `intrinsic` or `captured` ([sessions.md](sessions.md#times)) enter these clocks; others are counted apart. Fields `wall_span`, `clustered_time`, `tool_time`, `tool_time_union`, `ends_with`, `last_stop_reason`. Parameterized: the 5-minute gap is printed in field help and read headers, with a sensitivity figure for 5 against 10 minutes.

**Where the time went.** Every instant of a session's wall span is attributed to exactly one party: `tool`, keyed by normalized command for shell calls and by action otherwise; `model` (from a human prompt or a tool result to the next model message); `human` (from a model message ending a turn to the next human prompt; unknown, with no value rather than 0, for a session none of whose model messages states a stop reason, since its provider reports no turn end, and that time is unattributed); `unattributed` (missing or reversed times), never merged into another party. Overlapping tool calls share their union proportionally, and the overlap is reported. Unit: milliseconds per party and key; denominator: the session's `wall_ms`. Stored as `session_time(session, party, key, ms)`. Fields `waiting_model`, `waiting_human`; `| measure time`; the read header "75 min wall, 41 min clustered: cargo test 12m, edits 6m, model 18m, you 5m". Evidence: Morrow #2 (10:00–11:15).

**Turn fields** (the `turns` unit, at read): `latency` (last prompt to first model message), `reply_gap` (turn end to next human prompt), `attempts` (replies under the last prompt, branches included), `burst`, tool calls, errors, tokens and cost, `interrupted` (the harness's interrupt record), `corrected` (a current correction annotation), and `unanswered` (the turn ended the session with no reply). Per-session latency medians are aggregates (`turns where session:X | agg p50(latency)`).

**Concurrency.** For each instant, the number of sessions whose clustered intervals contain it, split by main and spawned sessions and by repository; per hour, the time-weighted mean and the maximum. Sessions without intervals are counted apart. Parameterized by the clustering gap.

**Censoring.** A call or session with no observed end is unfinished as of its last observation; its duration is censored and shown apart, never averaged as ended.

## Tool behaviour

**Normalized command and command class.** `command_norm` is the shell command normalized for comparison across sessions. Inline function headers and calls through a defined function name or shell variable are removed, leaving the commands in the function body; `command_class` comes from one declared table mapping normalized prefixes and subcommands, per tool family (cargo, nix, make and just, npm and pnpm, pytest, go, git, gh, bd, read-only probes), to `test`, `build`, `lint`, `format`, `vcs`, `package`, `probe` or `other`. A call with no command has NULL. The table is general. It is Polylogue's one command classifier, labelled **extracted**, and it serves only the analysis of agent tool use (outcome counts, reruns, failing checks, survival, cost of a failure, `group by command_class`). No brief section, advisory, notice or verification state reads it, because deciding what a command means from its name is brittle. A row column whose version is the table's hash. **Precision evidence** is required before any class-based figure ships and is shown with it: coverage weighted by time, and precision on the 200 most frequent normalized commands, checked by hand. Each class-based figure is labelled "by command class (extracted)".

**Submitted shell stages.** `sessions | measure shell-stages` extracts the normalized stages in recorded shell calls over the selected sessions and message times. A supported wrapper contributes the stages inside it. Each occurrence counts, including repeated commands and conditional branches. Each stage also reports its distinct call and session counts. The report gives the total recorded shell calls and how many yielded no recognized stages, including missing command text and unsupported scripts. The counts describe submitted text. A call's outcome and timing belong to the call as a whole.

**Outcome counts, failure-rate bounds, transitions.** Outcomes `ok`, `error`, `unknown` (with reason) and `pending` (at read). `| agg failure_rate` returns `[error/n, (error + unknown + pending)/n]` with all counts; `n = 0` is unmeasured. The harness verdict and `exit_code != 0` are two definitions, never mixed. `| measure outcomes` reports the counts. Directly-follows transitions between calls (`| measure transitions`), with counts after failure and common three-call sequences, are paged, never cut at a fixed count. An origin that records no tool calls shows "no tool data", never 0. A wrapper call has its own outcome.

**Reruns.** In one session and working directory, ordered by time: *retry chains* (consecutive failing runs of one normalized command, with the edits between them counted); *fail-then-pass* (a failing run followed by an `ok` run of the same command and arguments, with the edits and minutes between; named for what it records, not a claimed fix); *edits between checks* (file edits between consecutive check-class runs). Untimed calls are counted apart. Evidence: Morrow #2 (`cargo test` exit 101, one edit to `src/parse.rs`, then ok) and #4 (`cargo clippy` exit 1, then ok).

**Failing checks.** Per repository, the check-class runs whose latest run failed with no later `ok` run of the same normalized command, with the failures since the last success. Exit code 1 from `grep`, `egrep`, `fgrep`, `rg`, `ag`, `which`, `test`, `[`, or `git grep` means no match: that run counts as neither success nor failure and does not resolve an earlier failure. Other nonzero exit codes remain failures, including `cargo test` exit code 101. Read by `| measure failing` and reports only, labelled extracted through the command class.

**Survival.** `| measure survival` gives Kaplan-Meier estimates with Greenwood error over session spans, with one option naming the event: `survival:failure` (the default; time to the first failed tool call, in minutes), `survival:failure-turns` (the same, in human prompts) and `survival:unanswered` (time to ending on an unanswered prompt: `ends_with: human_prompt` and quiet for 6 hours at reading, censored otherwise). Time to the first passing test-class run is, labelled "first ok run of a test command", never "solved". Live sessions are censored at now; untimed sessions are unmeasured.

**Operator interventions.** Per turn and session: interrupts (harness interrupt records), rejected tool calls (with the operator's typed reason when present), prompts queued while the agent was busy (Claude Code `queue-operation` enqueue records), permission answers (hook events) and corrections (correction annotations, at read). Each marker is a row flag or event kind set by the parser, declared per harness version beside it; sums are computed at read. Rates are per clustered agent hour and per 100 model turns, each named; they are figures in reports and on the web cost and contrast pages, not brief content. A harness that records none of the markers reports "not recorded" from its origin declaration. Evidence: a recorded example of each marker per harness, and marker counts by harness version, so a version that stopped writing one shows as a gap rather than a quiet month.

**Service interruptions.** Events of kind `api_error` with the provider's error kind (overload, rate-limit refusals, API errors), and per session the wall time from each error to the next successful model message. They feed spend at past limit hits.

**Cost of a failure.** For each fail-then-pass pair: the API-equivalent cost and the time per party between the failing and the passing run, with unpriced counts.

## Sessions, work and comparison

**Commit and pull-request links.** From `git commit` output, commit trailers, Codex git records, `gh pr` output and pull-request link records; session-keyed, behind the `commit:` and `pr:` filters. A commit several calls report keeps the earliest call in session order as its source, and a pull request's URL is its normal `https://github.com/<owner>/<repo>/pull/<number>`, so neither depends on the order the calls arrived in. Evidence: precision and recall of commit hashes against `git log` for one month of sessions.

**Landed work.** For each linked commit, whether it is reachable from the repository's default branch at the time of reading, and each pull request's state from a captured forge record; per session and repository, commits landed and not yet landed. Read-time, cached per commit hash with the default-branch head it was checked against. Polylogue reads the repository itself.

**Delegations.** Per `spawn` link: the parent and its spawning call, the child, the instruction, the model requested and used, the child's final message and the parent's tool result, the outcome, the child's cost and wall time, `retry_of`, `result_use`, and how the link was made. Read-time; [agents.md](agents.md) and the delegation card read it.

**Live state.** From the live columns the writer holds and the time of reading: rank, waiting state, open call, live task, peers. Never stored, because it depends on the clock.

**Hook coverage.** `| measure hooks`: per harness and hook event, whether the event is declared in the harness's settings, whether it was seen recently, and when it was last seen, so a hook that stopped firing shows as a gap. How hooks are installed is [installing.md](installing.md#installing).

**Open items.** Per repository: tasks claimed and not closed (task linkage, [agents.md](agents.md)); todo and plan items whose last recorded state is not completed; sessions ending on an unanswered prompt; calls without results in ended sessions; background tasks never reported. Each item has `opened_by`, `state` (`open`, `resolved`, `superseded`) and `resolved_by` (the later structural record: the explicit task close or release, the later todo state, the reply to the prompt). Read-time, so a later arrival resolves an item with nothing to invalidate. Every form of the brief reads this one unit. Each item kind declares its own validity.

**Repeated instructions.** Human prose prompts split into sentences and normalized (case, whitespace, punctuation, numbers); a sentence of at least six words seen in at least three logical sessions is listed with its sessions, first and last dates and repositories. Only normalized matches count.

**Rework.** *Re-edit*: a repository-relative path edited in a session and again in a different logical session within 7 days. *Re-read*: a path read again in one session with no edit between. Labelled a count of edits, never wasted work, because a planned follow-up is also a re-edit.

**Related sessions.** One query term `related:REF`, ranked by a declared weighted sum over named reasons, each reported in `why` with its value: same logical session; same task; shared edited files (count and Jaccard); same repository; shared attachment identity; same display title; start times within 24 hours; an explicit link (shown, never needed). Every eligible session is scored before paging. Semantic similarity is `similar:session:REF` ([search.md](search.md#semantic-ranking)); `fuse` combines them. Judged: the term is exposed only with a recorded recall at 10 over held-out pairs joined by a continuation link or one task (each held-out reason removed from scoring), followed by a second figure from operator-judged rankings when they exist.

**Facet counts with IDF and lift.** Per facet family, `group by F | count` over the selection beside the same count under the unit's defaults alone; IDF = ln(global sessions ÷ global sessions with the value); lift = selected share ÷ global share. A logical session counts once per value that any of its physical sessions carries; a `physical:true` listing counts physical sessions one by one. Values are paged per family. Default families: origin, model, repository, project, tag, logical month, `ends_with`; deep families add every directory's repository, author, block kind, action and flags. An empty selection returns zero counts with outcome `empty`.

**Contrast.** `| measure contrast[:FIELDS]` renders chosen fields of two or more sessions with a "varies" mark per field; reading the sessions themselves side by side is the `compare` view ([reading.md](reading.md)). With a second population in parentheses, `| measure contrast (QUERY)`, it puts the two populations side by side, each component (cost per session median and p90, failure bounds, clustered time, prompts, the `ends_with` distribution) with its own denominator and unknown count, labelled with both populations: never one score, and no component can be hidden.

**Distributions and correlation.** Aggregates over any numeric field: `p50(F)`, `p90(F)`, `max(F)`, `histogram(F)` (equal-width or explicit edges, with the unknown count), `corr(X, Y)` (Pearson r with pairs, missing pairs, and a reason below 2 pairs or at zero variance). Percentiles are nearest-rank, one definition. Unknowns are excluded and counted; zeros take part.

**Reports.** A report is a query: plain pipelines and measures over a selection, grouped by a key, which `--explain` prints. Keys: `origin`, `model`, `day`, `week`, `month`, `repo`, `project`, `root`, `tool`, `action`, `command_class` (extracted). Rows carry unknown, partial and pending counts and a `total` row computed before rounding; a two-model session counts under each model and once in the total. Counting is by item time within `since:` and `until:`.

| Report | Query |
|---|---|
| sessions, prompts, tool calls and API-equivalent cost per key | `'repo:sinex since:30d \| group by origin \| agg count(), sum(prompts), sum(tool_calls), sum(cost)'` |
| cost | `\| measure outlook` for the quota and plan outlook first, then `\| measure usage`: per key the five token kinds, API-equivalent and provider-reported; recorded reports apart |
| tools | `\| measure outcomes`: calls, ok, error, unknown, pending, failure bounds, exit ≠ 0 |
| time | `\| measure time`: wall span, clustered time, tool time by command, waiting for model, waiting for human |
| windows | `\| measure quota`: window history |
| two selections | `\| measure contrast (QUERY)` |

An empty selection gives zero counts with outcome `empty`; a failed component gives `degraded` with that component's error while the other columns stay. Any report can be saved like any query (`save NAME`, [saved-queries.md](saved-queries.md#save)); a kept report records its pipelines, scope, clock, unknowns and derivation versions, and renders to a file through `read --to` like any kept result ([reading.md](reading.md#formats-and-destinations)).

**Embeddings.** A vector per message or chunk, used as a ranking score, never as a filter. The derive consumer embeds new or changed text while model calls are on (`polylogue set embed.calls on`), within the spend budget the settings carry ([search.md](search.md#semantic-ranking)); turning calls on is the user's one authorization. Model calls are off by default; stored and purchased vectors are kept and used. Judged: recall at 10 on known pairs before it is relied on. Search behaviour and providers are search.md's.

## Judging and experiments

### Rankings and agreement

**Ranking.** `| measure ranking` reads current comparisons ([judging.md](judging.md#the-better-of-several-rows)); a selection holding several dimensions is ranked per dimension. Each comparison adds pairwise outcomes: `prefer:X` is a win for X over each other row; an order is a win for each row over every row after it; `tie` is half a win each way for every pair; `incomparable`, `abstain` and `insufficient_evidence` add none and are counted per row. A row's win rate is its wins over its pairwise outcomes, with that count. Bradley-Terry strengths are fitted by maximum likelihood with a weak prior, so a row that won every comparison still gets a finite strength; the prior is the declared constant and is printed. Each rank shows its strength with an interval and the number of comparisons behind it; rows the comparisons do not connect are ranked in separate groups, never against each other. **Uncertain pairs** are neighbouring ranks whose intervals overlap, fewest direct comparisons first, each given as a ready `judge` line that selects the two rows with the dimension.

**Agreement.** `| measure agreement` compares a pass's `rubric_value`s with a reference judge: the operator unless the option names another judge, in the forms `judged_by:` takes ([query.md](query.md#units-and-fields)). A value's reference item is the reference judge's current judgment of that exact revision (accept agrees, reject disagrees) or the judge's own value for the same row and field under the same rubric revision (equal values agree). **Excluded:** abstentions, deferrals and values with no reference item. **Incompatible:** a reference item about another revision of the value or under another rubric revision. **Conflicting:** two current reference items that differ. Agreement is agreeing items over compared items, overall and per field, with the compared count as its denominator and the three other counts beside it.

### Experiment

`polylogue 'slots where page:exp-1 | measure experiment'` reads the page's assignment and the outcomes of its units ([page-templates.md](page-templates.md#experiments)). For each arm, with n the units assigned to it, satisfied the units whose current outcome meets the declared outcome, and unresolved the units with no current outcome, an abstention or a conflict, the result is the interval `[satisfied/n, (satisfied+unresolved)/n]`. Unresolved units stay in the denominator: there is no point estimate that drops them. An arm with n = 0 is unmeasured. Units whose recorded exposure differs from their assignment are counted per arm. Causal words ("caused", "because of the arm") appear only for a page whose assignment was written before any exposure and never revised after it; otherwise the reply speaks of association.
