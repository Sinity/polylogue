# Derived data and measures

Everything Polylogue computes from parsed sessions: the rules every derived product follows, the catalog of products, and the measures over them (usage and cost, quota and plans, time, tool behaviour, comparison, reports). The parsed rows these products read are [sessions.md](sessions.md); query grammar, `group by`, `agg` and paging are [query.md](query.md); kept results, findings and judgment notes are [query.md](query.md#query-objects) and [assertions.md](assertions.md); semantic ranking is [search.md](search.md#semantic-ranking); how figures appear in a read is [reading.md](reading.md); chart and page layout is [web.md](web.md); the brief and packs are [agents.md](agents.md).

Measurement answers four questions for someone who runs many agents on subscriptions: how much of the subscription is left and when it runs out; what something cost, and would have cost at API prices; where an agent's time went and how agents behave; what one set of sessions looks like beside another. Every figure is computed from records, never guessed from prose, and says what it counts and what it could not count.

## Principles

- **The outlook leads for a subscription user.** The current window, its burn and when it runs out come first in status, the overview, the cost page and `analyze --cost`. API-equivalent money follows, labelled as a comparison.
- **Unknown is never zero.** Every sum states its known part and how many items it could not count, in the items' own unit. A partial value is a lower bound ("at least"); a missing time is never placed.
- **One definition per figure, one function per rule.** One cost function prices a response; every other cost is a sum of its results. Surfaces never recompute.
- **Denominators and grain are part of the answer.** Every rate names its denominator, every count names its unit, and figures from different definitions are labelled not comparable.
- **Computed, not interpreted.** Counts, sums, durations, joins and statistics over recorded structure are in scope. Labels inferred from prose (work kind, phases, success, abandonment) are not, because they would claim what the records do not show. Ruling, 2026-03-27: "I think 'confidence' stuff is menaningless slop" (cc-queued:14ea730a-aa86-433a-9bba-8be973977d5c@2026-03-27T08:47).
- **No umbrella category.** Users see fields, units, measures and reports. The code's name for a stored or computed product is *derivation*, and the generated reference page is "Derived data" (`reference/derived.md`). Ruling, 2026-08-31: "aggressively dissolve the insights concept … there shouldn't be some soprt of separte step called 'insights' in the convergence" (cc-queued:62b956ba-3ea3-4693-8d71-9a36321c37a9@2026-08-31T05:41).
- **Measurement is the query language plus named computations.** Anything a pipeline can express is a pipeline. A measure exists only for what pipelines cannot express: windows, forecasts, survival, level changes, transitions, ranking.
- **Every number reaches every surface the same way,** through the command table; `explain` prints the pipeline behind a report.
- **Precise accounting with existing libraries.** Ruling, 2026-05-04: "we want to have precise tokenization, token counts, cost tracking, model etc." (cc:c1cf89f2-c4ff-48de-9459-599c2e8d04ff:c6cfe879-70d5-42e4-8913-989d114ce0ff); token counting and cost tracking "should actually be pretty much solved problems" (ruling 2026-06-27, cc:1944721d-4531-41be-8093-13454b62ebd6:cb4115e5-9b91-4d6d-938d-bfe842e10c14).
- **Reports are general.** No report is specific to one person's own tools or setup.

## Rules

### Validity

Each product declares one validity kind, which decides the evidence it needs before it ships. Ruling, 2026-10-27: "construct validity very important, and the way things are converged, avoiding the issues of old codebase where these things were some weid separate stage" (cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:2f5be8f0-e636-4b96-860c-1c1ca4f3e930).

| Kind | What it is | Evidence required before it ships |
|---|---|---|
| **exact** | a count, sum, join or ordering over recorded fields; its name states its definition | laws checked by property tests, and hand-checked values on Morrow |
| **parameterized** | exact once a declared constant is fixed | as exact, plus the constant printed wherever the value is shown, and a sensitivity figure (the value at the constant and at one other value on a real-shaped sample) |
| **extracted** | read from tool output or command text by declared per-tool patterns | coverage (share of candidates recognized) and precision and recall on a hand-checked real sample, with `n` and date, recorded in the declaration |
| **judged** | a label, segment or ranking standing for something the records do not state | agreement with operator-labelled examples or a structural ground truth, with `n`, recorded before shipping; without it the product is not exposed |

Labelled samples are private judgment notes and kept results in the operator's archive; only the figures appear in declarations and docs.

### Grain and storage

Analyses are computed when their data arrives, and at read only when that cannot be done accurately. Ruling, 2026-06-27: "we are supposed to materialize such analyses into proper schema we have. I mean, basically compute these when we read the underlying data. not dynamically (ofc if it cant be done (accurately, correctly))" (cc-queued:1944721d-4531-41be-8093-13454b62ebd6@2026-06-27T17:03).

- **Stored grains are row and session only.** A stored product is a column on a parsed row (`message`, `tool_call`, `event`), or a session column or session-keyed table.
- **Session-grain values are stored as their data arrives** when a query filters, sorts or groups on them across the archive, or when they are costly per read (graph walks, interval unions).
- **Grouped figures are computed at read** from stored session rows: logical session, day, week, repository, project. Grouping stored rows is exact, so a late import, a re-parse or a newly arrived link changes them with nothing stale to invalidate. A grouped figure measured too slow at read may be stored as well, keyed by session (for example by session and UTC quarter hour), so it changes with its session; nothing is keyed by day, repository or root.
- **Read-time products** are those that depend on the clock (live state, windows, forecasts, "quiet for 6 hours"), on a setting (logical day, plan) or on an arbitrary selection (aggregates, comparisons, rankings), and per-turn or per-pair computations no archive-wide query sorts on.
- **Stored products read only parsed rows,** never assertions, settings or the clock, so a note or a setting change never makes a stored product stale.
- **Storing a product does not make it citable.** Citation is served by [kept results](#citation-and-figures). Whether a product that is neither stored nor a field is best served as a named query is decided per product. Ruling, 2026-10-27, on that rule: "this might not be universal neccessarily" (cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:109d583d-02c8-43d4-9ea0-0db79c3f5ed4).

**Two storage forms:**

1. **Row columns,** computed by the writer with their row, in the chain's transaction, by the same function the derivation uses.
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

Derived data converges through one path, with no separate stage and no backfill, refresh, rebuild or recompute command.

- Row values are written with their rows. Session products are recomputed for exactly the sessions whose rows or derivation version changed.
- **Marks:** `derive_state(name, session, version, from_changed)`. A session is behind for a derivation when its mark is missing, its version differs, or the session changed after the mark.
- **Cross-session causes** delete the affected marks in their own transaction: a link change marks the logical-session root of the child's subtree and the session cost of its ancestors; a repository lookup marks the working-directories product of sessions under its path.
- **Recompute:** the derive consumer takes behind sessions, live ones first, then newest first; it computes in parallel from a read snapshot, and the writer commits each session's rows with its new mark. A crash leaves marks behind, which the ordinary open path treats as pending work.
- **Reads of behind sessions:** an aggregate counts behind sessions as `pending` and never shows their old values as current; a read of one session computes its products on the spot with the same function, so a single-session answer is never stale.
- **Versions:** rule edits bump `Rules(n)`; shipped tables (prices, command classes) are versioned by content hash, so a forgotten bump cannot ship. Changing a derivation's version re-derives exactly that derivation; there is no index-wide version. A golden-output test per derivation fails when output changes without a version change.
- **Re-parse:** `ON DELETE CASCADE` from parsed rows removes their products; a retracted chain changes its session, which marks every derivation behind.
- Ongoing cost is per changed session, never per archive.

### The declaration

Every derivation declares its construct beside its code, so the code is the catalog:

```rust
struct Derivation {
    name: &'static str,
    version: Version,                  // Rules(n) | Content(hash of a shipped table)
    after: &'static [&'static str],
    run: fn(&Transaction<'_>, i64, &Inputs) -> Result<()>,
    finish: Option<fn(&Transaction<'_>) -> Result<()>>,
    dirtied_by: &'static [Cause],      // SessionChanged implicit; LinkChanged{Descendants|Ancestors}; RepoLookupChanged
    construct: Construct,
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
- The generated page `reference/derived.md` is rendered from the declarations and registries; a test fails when a stored table has no declaration or a declaration has no entry.
- A registration test refuses an `Extracted` or `Judged` declaration without a recorded figure, and an `Exact` or `Parameterized` one without a named test.
- `explain derived` and `explain fields --unit U` serve the catalog to agents through CLI and MCP. Each field's help is its `represents` sentence; its unknown word is its `unknown` rule.

### Citation and figures

- A kept result records the query hash, resolved bounds, rows and the version of every derivation it read. Reading it later shows "computed under prices `<hash>`, now `<hash>`" when they differ.
- A **figure** is a saved `| measure` or `| agg` query with a declared denominator query ([query.md](query.md#query-objects)). Shipped figure definitions (cost, tokens by lane, tool error rate, turn latency) are saved queries.
- Every figure a measure or report returns names its definition (measure or pipeline text), population, unit, and its unknown, partial, pending and untimed counts.

### The default unit

[Logical sessions](sessions.md#logical-sessions) are the default unit for listings, counts and reports. A report's `sessions` column counts logical sessions, with a `physical` column beside it; per-session products are stored per physical session and grouped by root at read; `--physical` counts physical sessions. Money and tokens sum over every physical session once, whatever the grouping.

## Catalog

Store: **row** (a column on a parsed row), **session** (stored at session grain), **read** (computed at read). Compositions (the brief, takeover, packs, recall, and the `summary`, `context` and `links` views) are not products: they compose these at read and are stored only as deliveries.

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
| Session cost | `session_cost`: `cost`, `tree_cost`, `subagent_cost`, `cost_complete` | session | exact |
| Usage over a selection | `measure usage` | read | exact |
| Quota readings | `quota_reading` | row | extracted |
| Subscription windows, burn, forecast, spend at past limit hits | `measure windows` | read | parameterized |
| Billing-cycle outlook | `measure outlook` | read | parameterized |
| Daily series, baselines, level changes, rhythm | `group by day(at)` | read | parameterized |
| Context pressure | `context_tokens` | row, read | exact |
| Cache economics | cache shares and costs | read | exact |
| Tool-call result time and duration | `result_at`, `duration`, `reported_duration` | row | exact |
| Session span | `session_span`: `wall_span`, `clustered_time`, `ends_with`, … | session | parameterized |
| Where the time went | `session_time`; `waiting_model`, `waiting_human` | session | parameterized |
| Turn fields | the `turns` unit | read | exact |
| Concurrency | sessions active at once | read | parameterized |
| Normalized command and command class | `command_norm`, `command_class` | row | extracted |
| Outcome counts, failure-rate bounds, transitions | `agg failure_rate` | read | exact |
| Reruns | retry chains, fail-then-pass, edits between checks | read | exact |
| Failing checks | `measure failing` | read | exact |
| Survival | `measure survival` | read | parameterized |
| Operator interventions | row marks and events; sums | row, read | extracted |
| Service interruptions | `api_error` events | row, read | extracted |
| Cost of a failure | per fail-then-pass pair | read | exact |
| Commit and pull-request links | `commit_link`, `pr_link`; `commit:`, `pr:` | session | extracted |
| Landed work | landed and not landed commits | read | exact |
| Delegations | per `spawn` link | read | exact |
| Live state | rank, waiting state, open call, task, peers | read | exact |
| Open items | the `open_items` unit | read | exact or extracted per item kind |
| Repeated instructions | repeated sentences | read | exact |
| Rework | re-edits, re-reads | read | parameterized |
| Related sessions | `related:REF` | read | judged |
| Facet counts with IDF and lift | facets | read | exact |
| Compare | session and selection compare | read | exact |
| Distributions and correlation | `p50`, `p90`, `histogram`, `corr` | read | exact |
| Reports | `analyze` | read | exact |
| Embeddings | vectors in the embeddings store | row | judged |
| Format drift | the `drift` unit ([archive.md](archive.md#format-drift)) | read | exact |

Activity sessions, presence, agency, interruption and resumption, cross-day episodes and day narratives are Sinex's higher layer ([Sinex higher-layer](https://github.com/Sinity/sinex/blob/master/docs/higher-layer.md)); Polylogue supplies sessions, task linkage and session spans as its events.

## Session structure

**Shown version of repeated turns.** Two routes without shared turn IDs (a Gemini page capture and its Takeout export) hold one turn twice; without this every count, cost and search hit for those sessions doubles. Prompts match on whitespace-normalized text occurring once in each input, in the same order, with the same roles after; a repeated prompt ("continue") stays shown from both. The native route's message is shown; the other names it in `duplicate_of` with `duplicate_reason`. No match means shown. Unit: a message mark. A session derivation; the marked message is excluded from every population. Evidence: precision and recall on 50 matched and 50 unmatched hand-checked turns.

**Active path.** The path from the provider's newest marked leaf to the root. With no valid marker the value is NULL (unknown, read as on the path). Abandoned branches were run and billed, so usage counts both paths while message counts follow the active one. Evidence: Morrow #7.

**Session columns from records.** Kind, provider title, working directories, branch, harness version and start, merged from every input's session record by order-independent rules. Law: the result is independent of input order.

**Display title.** For a session whose records name no title: the first line of the first human prose prompt. The repository is shown beside it, never inside it. NULL when there is no human prose prompt; never written out as a recorded title.

**Working directories and repository.** One row per (session, directory) in first-use order: the directory, the longest repository lookup at or above it (by whole path components), the repository identity (its root commit), a display name, and `project` (the repository name, else the directory under `/<…>/project/<name>`, else NULL). A directory with no lookup has NULL identity and counts as "no repository", never as another repository. `repo:X` matches any directory of the session; `group by project` counts a two-project session under each. Cross-session cause: a new or changed repository lookup. Evidence: coverage on a real archive (share of agent sessions with an identity).

**Logical session.** `root` is the session reached by walking parent links of kind `continuation`, `fork`, `spawn` and `resume` to a session with no such parent; `depth` counts the steps; `reference` links never join. A pending link (the parent is not in the archive) stops the walk with `root_pending` set, instead of a guess. Links come only from source markers ([sessions.md](sessions.md#session-links)). Stored because `group by root` and logical counts run over the whole archive. Laws: the root is independent of link arrival order; Morrow #2, #3 and #4 share root #2, and #5 and #6 share #5.

**Session counts.** Messages, human prompts, human commands, words, words per author, thinking messages, compactions, tool calls, errors and calls by action, over the session's own rows. Stored as session columns, because they are fields sorted and grouped across the archive. A threshold such as "substantive" messages is stated in the query (`messages where words>=3`), never stored.

## Cost and usage

### Token lanes and populations

- **Five lanes** on every response a model wrote, as reported: uncached input; cache read; cache write (with its 1-hour part when reported); output excluding reasoning; reasoning. NULL means not reported, never 0. Reasoning is priced at the output rate; the token total is all five lanes.
- **Counted tokens where no usage is recorded.** For sessions whose source records no usage (web chats), tokens of the visible text are counted with the tokenizer library for the model's family where one is published, otherwise with the shared [token estimator](reading.md#budgets). Counted tokens are their own state (`counted`, shown `≈`), name the tokenizer, are a lower bound of billed input (hidden system prompts and tools are not visible), are never summed with reported tokens without that label, and price only to the `estimated` money kind.
- **Populations,** one per figure, named in the result:

  | Population | Contents |
  |---|---|
  | `own` (default) | each recorded response once across the archive: the shown version only (`duplicate_of` NULL), on and off the active path |
  | `with_parents` | a session's own responses plus those of the sessions it composes from ([reading.md](reading.md#composed-transcripts)); per session only, never summed across sessions |
  | `tree` | a session's own responses plus those of every session below it through `spawn` links |
  | `recorded_reports` | usage a source reports without naming its response (Codex `token_count`); its own unit and denominator: validated, unconfirmed and untimed reports |

- Display filters from assertions (tags, `hidden`) never change a population; a measure states how many hidden sessions it includes.
- A source that cannot assign usage to responses makes the response count and response-based totals of that scope unknown; its recorded reports are still summed as a known part.
- **Codex completeness by reconciliation:** `token_count` reports are validated per request (the running total grew by exactly the last counts). A session whose validated reports cover its final running total, with no unconfirmed report, has complete usage and an API-equivalent cost at grain "usage reports"; otherwise it is partial, with the gap in tokens.
- **Time bounds** select response times (`timed_by: source_record_time`), not session activity. Untimed rows are counted as `untimed` and excluded from bounded sums.
- `measure usage` returns: population, `timed_by`, sessions, responses (known, and items with unknown boundaries), tokens per lane (known, and unknown responses), money by kind and by model, unpriced models with counts, recorded reports, the price catalog hash and label, and untimed items.
- A provider's own total for a child (Claude `toolUseResult.totalTokens`) is evidence only and is never added. A lane the provider does not charge is NULL, adds nothing and does not make the response partial (`n/a`).

### The price catalog

- One catalog ships with Polylogue: the providers' list prices, maintained by hand with a source note per row, as `prices.toml`. The user may override rows in `$XDG_CONFIG_HOME/polylogue/prices.toml`; an override replaces the shipped row with the same model and effective date.
- A row has `name` (normalized model), `effective` (applies at or after), `input`, `cached_input`, `cache_write`, `cache_write_1h`, `output` (USD per million tokens); optional long-context rates above a prompt size; optional modes and tiers; an optional per-search price; and `source` (URL and the date it was read).
- **Model names are normalized per origin,** beside its parser (prefixes, aliases, date suffixes). The catalog matches exact names; a versionless alias (`opus`) stays unpriced, never guessed.
- **The version is the content hash** of the effective shipped table plus overrides; a different hash reprices through the derive path with no re-parse. A human label is shown beside it, never used as the key.
- No row on or before the response time, a tier or mode without a rate, or a cache lane with tokens but no rate leaves the cost unknown; zero tokens need no rate. Cache writes with no reported lifetime use the 5-minute rate and are counted as `assumed_cache_lifetime`.
- A malformed override file is refused with line and field; the shipped table stays in force and status shows the error. Status and usage reports list every model seen with no price, with counts and newest time.
- `polylogue prices [--model NAME]` lists rows, sources and the active hash, or the row a model resolves to and why.

### Money

- **One kind per figure,** never added across kinds:

  | Kind | Meaning |
  |---|---|
  | `api_equivalent` | list price of the recorded tokens; the default cost |
  | `provider_reported` | a total the harness or provider recorded |
  | `subscription_share` | a plan's price for a cycle spread over its responses in proportion to their API-equivalent cost |
  | `manual` | an amount the user entered (a `cost` assertion) |
  | `estimated` | computed from incomplete or counted data, with what it assumed |

- A separate type per kind makes adding across kinds a type error.
- **Integer nano-USD** in storage and every sum; JSON carries `usd` with the exact value and `nusd`; rounding happens once, after summing, in text and HTML.
- **Completeness travels with the value:** `{nusd, complete, priced, unpriced}`, one of `complete`, `partial` (a lower bound) or `unknown` (null). Query fields `cost`, `cost_complete`, `unpriced_responses`; `agg sum:cost` reports `partial_cost` and `unknown_cost`.
- A provider-reported zero is a measured zero. Each response of a multi-model session is priced under its own model. A session view shows API-equivalent and provider-reported side by side, with their difference labelled "not in transcript".

**Message cost.** One function prices a response from its lanes and its model's catalog row at its time. Unknown when any billed lane is unreported or the model is unpriced. A row column whose version is the catalog hash. Law: every stored message cost equals the function applied to its usage.

**Session cost.** `own` (the sum of message cost over the own population), `own_unpriced`, `complete` (no unpriced response; for Codex, reconciled), `tree` (own plus every session below through `spawn` links), `tree_unpriced` and `tree_missing_children` (children a spawning call names that are not in the archive, with those calls' references). A partial value shows "at least". A child belongs to exactly one tree. Cross-session cause: a change in a session marks the tree cost of each ancestor. Fields `cost`, `tree_cost`, `subagent_cost` (tree minus own, partial if either is) and `cost_complete`. Continuations, forks and resumes are not trees: a logical session's cost is the sum of its sessions' own costs, once each. Stored so that sorting by cost and tree sums are indexed reads.

### Subscription quota

- **Quota readings.** One row per distinct reading: origin, account, limit, window length, used percent, reset time, status, plan, credits, observed time, source row. Sources: Codex `rate_limits` on `token_count` events; Claude Code refusals (`quotaLimits`, 100% only when rejected); Claude Code status-line readings (`rate_limits.five_hour` and `seven_day`) through a status-line hook tap, which is official and local and needs no credential. A reading copied into a forked session is not new (deduplicated by report ID). A relative reset without an observed time cannot be placed. Written with its source row and indexed on (origin, limit, time), so the status line reads it within its budget.
- **Windows** are computed at read per account and limit: *reported* (a reading gives its reset), *projected* (whole window lengths from the last reported reset, for fixed schedules), *inferred* (a limit that starts at the first response after the previous window ended, labelled `≈`). Window usage: start, end, source, usage by lane, API-equivalent cost and active sessions; "at least" when it holds unknown usage.
- **Burn and forecast:** percent burn per hour between the first and last readings at least 5 minutes apart; the exhaustion time if before the reset, else "lasts to reset", labelled as a linear projection; token and money burn per hour; projected tokens at reset. A reading lower than the previous one restarts burn; a reset earlier than its reading is stale, ignored and counted; quota at 0% or 100% at the first reading never divides by zero.
- **Spend at past limit hits:** when no reading in the window carries a percent, the API-equivalent spend in earlier windows of the same limit at the moment a rejection was recorded, with `n` and range; absent when `n = 0`. The name says what is measured, not a capacity.
- Each rejected reading is an event (`events where kind:rate_limit`). `measure windows` and `analyze --windows` list past windows newest first, with usage, peak percent and whether the limit was hit.
- Untimed reports are listed as `unplaced` and affect only windows they could belong to. Accounts are separated by recorded identity, else stated as one.
- **Lead position:** status and the overview show, per subscribed account, one line per active limit: used percent or tokens, reset, burn, exhaustion or "lasts to reset", with inferred values marked.

### Plans and billing cycles

- **Plan setting** `plan.<provider>`: the plan name in the provider's terms (`pro`, `max-5x`, `max-20x`; `plus`, `pro`; a bare `max` is refused as ambiguous), an optional monthly price (else the shipped plan table, with its source), the cycle anchor day and an optional budget, dated from when it was set; a new value starts a new span. The plan table never states allowances the provider does not publish.
- A session counts against a subscription only with evidence: a quota reading in its files, or the plan setting in force at its time; otherwise its billing is `unknown`.
- **Cycle outlook:** a calendar month from the anchor day (days 29–31 clamp to the month's end; UTC dates). Projections per unit (API-equivalent, provider-reported, each lane), each method named: `linear` (used ÷ elapsed × total) and `trailing-7d-mean`. Each carries coverage, missing days and untimed or incomplete usage. Observations after the as-of instant are excluded, so the outlook is reproducible.
- **Subscription value:** per cycle, API-equivalent cost against the plan price, as a ratio; `subscription_share` derives from it. An optional user budget gives used and projected ratios, a breach day and overage at a user rate, labelled "your budget". A plan changed mid-cycle splits the cycle.
- Interfaces: `measure outlook[:PLAN[:METHOD]]`, `analyze --cost`, the web cost page.

### Daily series and rhythm

- A daily series of any summable metric over the one logical day (the shared `DaySetting`, [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md)): logical sessions, physical sessions, prompts, responses, tool calls, tool errors with denominator, tokens per lane, API-equivalent cost, recorded-report tokens and cost, clustered time, or any field through `group by day(at) | agg sum:F`.
- Each day carries `complete`, `unknown_items` and `untimed_items` for that metric; untimed items go to an `unknown day` row. A session crossing midnight contributes by each item's own time.
- `baseline_7d` is computed only over 7 complete previous days. **Level changes** are found by binary segmentation with a BIC-style penalty, minimum segment 7 days, over runs of at least 14 complete days, each with means before and after, and labelled "level change", never "trend".
- **Rhythm:** a 7 × 24 grid by logical weekday and clock hour, of any countable unit.
- Calendar and this-day views read the same series. Nothing per day is stored, so a setting change re-buckets at the next read. A 23- or 25-hour day is one day. One daily path serves outlooks, the cost and timeline pages and `analyze --by day`.

### Context pressure and cache economics

- **Context pressure.** `context_tokens` per response is uncached input + cache read + cache write, as reported (a row column; NULL when a lane is unreported). Per session at read: the maximum, the series, responses above declared fractions of the model's context window (from the catalog), compactions with tokens before and after (from the boundary event), and the operator prompts before each compaction. Evidence: Morrow #4 (182,000 → 14,500).
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

**Where the time went.** Every instant of a session's wall span is attributed to exactly one lane: `tool`, keyed by normalized command for shell calls and by action otherwise; `model` (from a human prompt or a tool result to the next model message); `human` (from a model message ending a turn to the next human prompt); `unattributed` (missing or reversed times), never folded into another lane. Overlapping tool calls share their union proportionally, and the overlap is reported. Unit: milliseconds per lane and key; denominator: the session's `wall_ms`. Stored as `session_time(session, lane, key, ms)`. Fields `waiting_model`, `waiting_human`; `analyze --time --by K`; the read header "75 min wall, 41 min clustered: cargo test 12m, edits 6m, model 18m, you 5m". Evidence: Morrow #2 (10:00–11:15).

**Turn fields** (the `turns` unit, at read): `latency` (last prompt to first model message), `reply_gap` (turn end to next human prompt), `attempts` (replies under the last prompt, branches included), `burst`, tool calls, errors, tokens and cost, `interrupted` (the harness's interrupt record), `corrected` (a current correction assertion), and `unanswered` (the turn ended the session with no answer). Per-session latency medians are aggregates (`turns where session:X | agg p50:latency`).

**Concurrency.** For each instant, the number of sessions whose clustered intervals contain it, split by main and spawned sessions and by repository; per hour, the time-weighted mean and the maximum. Sessions without intervals are counted apart. Parameterized by the clustering gap.

**Censoring.** A call or session with no observed end is unfinished as of its last observation; its duration is censored and shown apart, never averaged as ended.

## Tool behaviour

**Normalized command and command class.** `command_norm` is the shell command normalized for comparison across sessions; `command_class` comes from one declared table mapping normalized prefixes and subcommands, per tool family (cargo, nix, make and just, npm and pnpm, pytest, go, git, gh, bd, read-only probes), to `test`, `build`, `lint`, `format`, `vcs`, `package`, `probe` or `other`. A call with no command has NULL. The table is general. It is Polylogue's one command classifier, labelled **extracted**, and it serves only the analysis of agent tool use (outcome counts, reruns, failing checks, survival, cost of a failure, `analyze --by command_class`). No brief section, advisory, notice or verification state reads it, because deciding what a command means from its name is brittle. A row column whose version is the table's hash. **Precision evidence** is required before any class-based figure ships and is shown with it: coverage weighted by time, and precision on the 200 most frequent normalized commands, checked by hand. Each class-based figure is labelled "by command class (extracted)".

**Outcome counts, failure-rate bounds, transitions.** Outcomes `ok`, `error`, `unknown` (with reason) and `pending` (at read). `| agg failure_rate` returns `[error/n, (error + unknown + pending)/n]` with all counts; `n = 0` is unmeasured. The harness verdict and `exit_code != 0` are two definitions, never mixed. Directly-follows transitions between calls, with counts after failure and common three-call sequences, are paged, never cut at a fixed count. An origin that records no tool calls shows "no tool data", never 0. A wrapper call has its own outcome.

**Reruns.** In one session and working directory, ordered by time: *retry chains* (consecutive failing runs of one normalized command, with the edits between them counted); *fail-then-pass* (a failing run followed by an `ok` run of the same command and arguments, with the edits and minutes between; named for what it records, not a claimed fix); *edits between checks* (file edits between consecutive check-class runs). Untimed calls are counted apart. Evidence: Morrow #2 (`cargo test` exit 101, one edit to `src/parse.rs`, then ok) and #4 (`cargo clippy` exit 1, then ok).

**Failing checks.** Per repository, the check-class runs whose latest run failed with no later `ok` run of the same normalized command, with the failures since the last success. Read by `measure failing` and `analyze` only, labelled extracted through the command class.

**Survival.** Kaplan-Meier estimates with Greenwood error over session spans: time to the first failed tool call (in minutes and prompts); time to ending on an unanswered prompt (`ends_with: human_prompt` and quiet for 6 hours at reading, censored otherwise); time to the first passing test-class run, labelled "first ok run of a test command", never "solved". Live sessions are censored at now; untimed sessions are unmeasured.

**Operator interventions.** Per turn and session: interrupts (harness interrupt records), rejected tool calls (with the operator's typed reason when present), prompts queued while the agent was busy (Claude Code `queue-operation` enqueue records), permission answers (hook events) and corrections (correction assertions, at read). Each marker is a row flag or event kind set by the parser, declared per harness version beside it; sums are computed at read. Rates are per clustered agent hour and per 100 model turns, each named; they are measures in `analyze` and on the web cost and compare pages, not brief content. A harness that records none of the markers reports "not recorded" from its origin declaration. Evidence: a recorded example of each marker per harness, and marker counts by harness version, so a version that stopped writing one shows as a gap rather than a quiet month.

**Service interruptions.** Events of kind `api_error` with the provider's error kind (overload, rate-limit refusals, API errors), and per session the wall time from each error to the next successful model message. They feed spend at past limit hits.

**Cost of a failure.** For each fail-then-pass pair: the API-equivalent cost and the time lanes between the failing and the passing run, with unpriced counts.

## Sessions, work and comparison

**Commit and pull-request links.** From `git commit` output, commit trailers, Codex git records, `gh pr` output and pull-request link records; session-keyed, behind the `commit:` and `pr:` filters. Evidence: precision and recall of commit hashes against `git log` for one month of sessions.

**Landed work.** For each linked commit, whether it is reachable from the repository's default branch at the time of reading, and each pull request's state from a captured forge record; per session and repository, commits landed and not yet landed. Read-time, cached per commit hash with the default-branch head it was checked against. Polylogue reads the repository itself.

**Delegations.** Per `spawn` link: the parent and its spawning call, the child, the instruction, the model requested and used, the child's final answer and the parent's tool result, the outcome, the child's cost and wall time, `retry_of`, `result_use`, and how the link was made. Read-time; [agents.md](agents.md) and the delegation card read it.

**Live state.** From the live columns the writer keeps and the time of reading: rank, waiting state, open call, live task, peers. Never stored, because it depends on the clock.

**Open items.** Per repository: tasks claimed and not closed (task linkage, [agents.md](agents.md)); todo and plan items whose last recorded state is not completed; sessions ending on an unanswered prompt; calls without results in ended sessions; background tasks never reported. Each item has `opened_by`, `state` (`open`, `resolved`, `superseded`) and `resolved_by` (the later structural evidence: the close command, the later todo state, the answer). Read-time, so a later arrival resolves an item with nothing to invalidate. The brief and takeover read this one unit. Each item kind declares its own validity.

**Repeated instructions.** Human prose prompts split into sentences and normalized (case, whitespace, punctuation, numbers); a sentence of at least six words seen in at least three logical sessions is listed with its sessions, first and last dates and repositories. Only normalized matches count.

**Rework.** *Re-edit*: a repository-relative path edited in a session and again in a different logical session within 7 days. *Re-read*: a path read again in one session with no edit between. Labelled a count of edits, never wasted work, because a planned follow-up is also a re-edit.

**Related sessions.** One query term `related:REF`, ranked by a declared weighted sum over named reasons, each reported in `why` with its value: same logical session; same task; shared edited files (count and Jaccard); same repository; shared attachment identity; same display title; start times within 24 hours; an explicit link (shown, never needed). Every eligible session is scored before paging. Semantic similarity is `similar:session:REF` ([search.md](search.md#semantic-ranking)); `fuse` combines them. Judged: the term is exposed only with a recorded recall at 10 over held-out pairs joined by a continuation link or one task (each held-out reason removed from scoring), followed by a second figure from operator-judged rankings when they exist.

**Facet counts with IDF and lift.** Per facet family, `group by F | count` over the selection beside the same count under the unit's defaults alone; IDF = ln(global sessions ÷ global sessions with the value); lift = selected share ÷ global share. A session counts once per value. Values are paged per family. Default families: origin, model, repository, project, tag, logical month, `ends_with`; deep families add every directory's repository, author, block kind, action and flags. An empty selection returns zero counts with outcome `empty`.

**Compare.** Session compare renders chosen fields of two or more sessions with a "varies" mark per field. Selection compare puts two populations side by side, each component (cost per session median and p90, failure bounds, clustered time, prompts, the `ends_with` distribution) with its own denominator and unknown count, labelled with both populations: never one score, and no component can be hidden.

**Distributions and correlation.** Aggregates over any numeric field: `p50`, `p90`, `max`, `histogram:F(bins)` (equal-width or explicit edges, with the unknown count), `corr:X:Y` (Pearson r with pairs, missing pairs, and a reason below 3 pairs or at zero variance). Percentiles are nearest-rank, one definition. Unknowns are excluded and counted; zeros take part.

**Reports.** `polylogue QUERY analyze [--by KEY[,KEY]] [--cost] [--tools] [--time] [--windows] [--compare QUERY]`. `analyze` is the report verb and `measure NAME` the named computation. Each report is ordinary pipelines joined on the key, and `explain` prints them. Keys: `origin`, `model`, `day`, `week`, `month`, `repo`, `project`, `root`, `tool`, `action`, `command_class` (extracted). Rows carry unknown, partial and pending counts and a `total` row computed before rounding; a two-model session counts under each model and once in the total. Counting is by item time within `since:` and `until:`.

| Report | Columns |
|---|---|
| default | logical sessions, physical sessions, prompts, tool calls, API-equivalent cost (with unknown and partial) |
| `--cost` | the quota and plan outlook first, then per key the five lanes, API-equivalent and provider-reported; recorded reports apart |
| `--tools` | calls, ok, error, unknown, pending, failure bounds, exit ≠ 0 |
| `--time` | wall span, clustered time, tool time by command, waiting for model, waiting for human |
| `--windows` | window history |
| `--compare Q2` | selection compare |

An empty selection gives zero counts with outcome `empty`; a failed component gives `degraded` with that component's error while the other columns stay. Any report can be saved and kept; a kept report records its pipelines, scope, clock, unknowns and derivation versions, and renders to a file through [`read --to`](reading.md#writing-out) like any kept result.

**Embeddings.** A vector per message or chunk, used as a ranking score, never as a filter. The derive consumer embeds new or changed text while the `embeddings` setting is on; the setting carries the spend budget ([search.md](search.md#semantic-ranking)) and is the user's one authorization. Model calls are off by default; stored and purchased vectors are kept and used. Judged: recall at 10 on known pairs before it is relied on. Search behaviour and providers are search.md's.

## Laws

Oracles: Morrow facts computed by hand ([testing.md](testing.md#morrow)), synthetic series, property tests.

- **every-product-declared**: every stored table has a declaration and an entry in the derived-data reference; every `Extracted` or `Judged` declaration has a recorded figure; every `Exact` or `Parameterized` one names a test.
- **stored-products-read-parsed-rows-only**: each derivation run with the assertion, setting and clock inputs empty and full produces the same output.
- **version-bump-required**: changing a derivation's rules without bumping its version fails its golden test; changing a shipped table changes its hash.
- **late-import-regroups**: a late import of a session in a past week changes that week's grouped figures at the next read, with no stale value served.
- **late-link-converges**: a link arriving late updates the logical-session root of the child's subtree and the session cost of its ancestors in the link's transaction.
- **behind-sessions-pending**: an aggregate over a selection with behind sessions reports them as `pending`; a single-session read of a behind session equals the value after derivation.
- **derive-crash-leaves-marks**: a crash during derivation leaves marks behind and no partial product visible.
- **morrow-cost**: archive own API-equivalent cost is $5.86 known, with 4 session totals unknown; #4's 18 inherited messages add nothing.
- **duplicate-route-sums-unchanged**: adding a duplicate route of a session leaves every own sum unchanged; an off-path branch's response is in own usage and not in message counts.
- **usage-equals-sum**: `sessions where Q | agg sum:cost` equals `measure usage` over Q in exact nano-USD; every stored message cost equals the cost function of its usage.
- **prices-effective-dated**: rows dated 2026-09-01 and 2026-10-01 price a 2026-09-30 response by the first and a 2026-10-01 response by the second; an override changes the hash and reprices exactly the affected messages; every shipped row has a source.
- **money-kinds-never-added**: no code adds two money kinds (separate types); a provider-reported zero is a measured zero.
- **tree-cost**: the tree cost of Morrow #2 is 3.01; a tree with a missing child reports `tree_missing_children ≥ 1` with the spawning call's reference.
- **counted-tokens-labelled**: counted tokens are never summed with reported tokens without the `counted` label, and price only to `estimated`.
- **codex-reconciles**: a Codex fixture whose reports reconcile has complete usage; one with an unconfirmed report is partial, with its gap.
- **codex-windows**: a Codex fixture gives reported windows, burn between readings 30 minutes apart, and exhaustion inside the window.
- **status-line-burn**: readings of 15% then 19% five minutes apart give 48% per hour and its exhaustion time.
- **spend-at-limit-hits**: a refusal-only fixture shows token burn, and spend at past limit hits from two earlier rejected windows with `n = 2`.
- **windows-converge**: a late import re-derives later inferred windows; a reset drop restarts burn; a first reading of 0% or 100% never divides by zero.
- **cycle-outlook**: linear and trailing-mean projections handle anchor day 31 and zero elapsed time; subscription share over two cycles sums back to the plan price; a mid-cycle plan change splits the cycle; observations after the as-of instant are excluded.
- **time-lanes-sum-to-wall**: Morrow #2 has a 75-minute wall span, and its time lanes match the hand-computed split and sum exactly to it.
- **long-tool-call-counts**: a 12-minute shell call counts 12 minutes of tool time where message clustering drops that gap.
- **overlapping-calls**: two overlapping 10-minute calls give sum 20, union 15, overlap 5.
- **tool-results-not-human**: a tool-result row never counts as human activity.
- **daily-series**: Morrow `analyze --by day` equals its golden; a step series (10 days at 1, 10 at 5) gives one level change at day 11; an incomplete day breaks baselines on both sides; a daylight-saving spring day is one row.
- **morrow-tools**: Morrow has 37 calls with 2 errors, and fail-then-pass pairs for `cargo test` (one edit to `src/parse.rs`) and `cargo clippy`.
- **failure-rate-bounds**: `agg failure_rate` over 70 ok, 10 error and 20 unknown gives `[0.10, 0.30]`; an exit-code-only failure counts only under the exit definition.
- **survival-steps**: Kaplan-Meier steps and censoring match hand values; time to pass uses the test class.
- **one-command-classifier**: no other table or pattern list maps command text to a class or a check, and no brief, advisory or notice code path reads `command_class`.
- **class-figures-labelled**: every figure grouped by `command_class` carries the extracted label and its precision figure; with no precision recorded, the figure is refused with that reason.
- **root-order-independent**: logical-session roots are independent of link arrival order; a `reference` link never changes a root; a resume joins its parent's logical session.
- **logical-by-default**: report `sessions` counts logical sessions by default and physical sessions with `--physical`; money is the same either way.
- **facets-count-once**: a session counts once per value; tag case folds; the IDF of a value present everywhere is 0.
- **unknown-never-zero**: a compare of Claude Code against Codex reports Codex cost as unknown or reconciled, never 0; an origin without tool calls shows "no tool data"; a harness that records no intervention markers shows "not recorded".
- **nearest-rank**: `agg p50` equals the reference nearest-rank median; `corr` reports pairs and missing pairs.
- **related-scores-everything**: `related:` scores every eligible session before paging and reports each reason in `why`; it is not exposed until its recall figure is recorded.
