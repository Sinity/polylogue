[← Back to Docs](README.md)

# Search & Query Reference

Polylogue uses a query-first grammar over archive units. Bare tokens are
full-text terms, field clauses narrow the selected unit, explicit
`sessions/messages/actions/blocks/assertions/runs/observed-events/context-snapshots where ...`
forms opt into Boolean predicates,
and trailing CLI verbs render or mutate the selected session set. The same
query semantics — filters, retrieval lanes, ranking policy, and typed response
payloads — apply across the CLI, MCP, Python API, and daemon HTTP surfaces.

Quick links:

- [Retrieval Lanes](#retrieval-lanes) — `dialogue`, `actions`, `hybrid`,
  `semantic`, and how `auto` elevates.
- [Terminal Unit Queries](#terminal-unit-queries) — `messages/actions/blocks/
  assertions/runs/observed-events/context-snapshots where ...` row results.
- [Ranking Policy](#ranking-policy) — current `mixed-bm25-rrf-vector`
  policy and version contract.
- [SearchEnvelope Contract](#searchenvelope-contract) — the typed
  response shape shared across surfaces ([#1266](https://github.com/Sinity/polylogue/issues/1266)).
- [FTS5 Syntax](#fts5-syntax) — boolean, phrase, and prefix queries.
- [Facets (Scoped vs Global)](#facets-scoped-vs-global) — aggregate
  counts with both views ([#1269](https://github.com/Sinity/polylogue/issues/1269)).

## Grammar

The canonical parser is the Lark grammar in
`polylogue/archive/query/expression.py`. It has two entry shapes that lower
through the same typed AST and query planner:

```text
compact-query      ::= compact-clause*
boolean-query      ::= ["sessions" "where"] predicate
unit-query         ::= ("messages" | "actions" | "blocks" | "assertions" | "runs" | "observed-events" | "context-snapshots") "where" predicate
pipeline-query     ::= unit-query ("|" pipeline-stage)+
                     | "sessions" "where" predicate "|" unit-query ("|" pipeline-stage)*

compact-clause     ::= field-clause
                     | quoted-text | bare-text
                     | "-" quoted-text | "-" bare-text
                     | count-comparison | count-range
                     | date-comparison | date-range

predicate          ::= predicate "OR" predicate
                     | predicate "AND" predicate
                     | "NOT" predicate
                     | "(" predicate ")"
                     | field-clause
                     | count-comparison | count-range
                     | date-comparison | date-range
                     | fts-leaf | semantic-leaf
                     | "exists" structural-unit "(" predicate ")"
                     | "seq" "(" sequence-step "->" sequence-step+ ")"

structural-unit    ::= "message" | "action" | "block" | "assertion"
pipeline-stage     ::= "sort" "by" "time" ["asc" | "desc"]
                     | "limit" integer
                     | "offset" integer
```

Compact queries select sessions:

```bash
polylogue repo:polylogue tag:active -"stale plan" "query envelope"
polylogue origin:(codex-session|claude-code-session) messages:>=10 words:<=2000
polylogue near:id:codex-session:abc123
```

Explicit Boolean session queries also select sessions:

```bash
polylogue sessions where '(repo:polylogue OR repo:sinex) AND NOT tag:stale'
polylogue sessions where 'exists message(role:assistant AND text:timeout)'
polylogue sessions where 'seq(action:file_edit -> action:shell)'
```

Unit queries select terminal rows instead of sessions:

```bash
polylogue messages where 'role:assistant AND text:timeout'
polylogue actions where 'session.repo:polylogue AND action:file_edit AND path:polylogue/archive'
polylogue blocks where 'type:code AND text:sqlite'
polylogue assertions where 'kind:decision AND status:active AND text:review'
polylogue runs where 'role:subagent AND status:completed AND agent:Explore'
polylogue observed-events where 'delivery_state:acted_on AND text:#2100'
polylogue context-snapshots where 'boundary:session_start AND session.repo:polylogue'
```

Pipeline stages can either decorate a direct terminal row query or scope one
through a session source stage:

```bash
polylogue 'messages where role:assistant | sort by time desc | limit 10'
polylogue 'actions where action:file_edit | limit 20 | offset 40'
polylogue 'sessions where repo:polylogue AND origin:claude-code-session | messages where role:assistant'
polylogue 'sessions where origin:(codex-session|claude-code-session) | actions where action:file_edit'
polylogue 'sessions where repo:polylogue | messages where role:assistant | limit 10 | offset 20'
polylogue 'sessions where repo:polylogue | messages where role:assistant | sort by time desc | limit 10'
```

When a pipeline starts with `sessions where ...`, the left stage is lowered into
`session.<field>` predicates on the terminal unit query, so it uses the same row
executor as direct `messages/actions/...` pipelines. For now the session stage
accepts field/count/date predicates only; FTS, semantic, `exists`, lineage, and
sequence stages are typed errors until they have dedicated unit-changing
lowerers. Terminal pipeline stages currently support `sort by time [asc|desc]`,
`limit N`, and `offset N` for SQL-backed terminal rows (`messages`, `actions`,
`blocks`, `assertions`). Query-string limits narrow the surface limit instead
of expanding caller/API caps, and query-string offsets are added to the caller
offset. Runtime-transform terminal rows (`runs`, `observed-events`,
`context-snapshots`) reject sort stages until they have streaming lowerers;
they must not collect every projected row in memory merely to sort a page.

`--explain --format json` reports terminal pipelines as a `unit_source` AST with
ordered `pipeline_stages`. Session-scoped pipelines include a `session_scope`
stage carrying the original session predicate, followed by any terminal
`sort`/`limit`/`offset` stages. Consumers should use that stage list rather than
inferring pipeline behavior from incidental `limit`, `offset`, or `sort`
fields.

Unsupported forms raise typed `ExpressionCompileError`s and must not broaden
into looser full-text search. In particular, reserved unit prefixes such as
`messages where` are errors when malformed; they are not treated as ordinary
text terms.

### DSL Fields

| Field | Meaning | Example |
|-------|---------|---------|
| `repo` | Repository substring | `repo:polylogue` |
| `origin` | Source origin | `origin:claude-code-session` |
| `tag` | User/session tag | `tag:review` |
| `path` | Referenced file path substring | `path:polylogue/cli` |
| `cwd` | Working-directory prefix | `cwd:/realm/project` |
| `tool` | Tool name used in the session | `tool:bash` |
| `action` | Semantic action category | `action:file_edit` |
| `has` | Content presence (`paste`, `tools`, `thinking`, or stored type) | `has:paste` |
| `id` | Session id or prefix | `id:codex-session:abc` |
| `title` | Session title substring | `title:refactor` |
| `since` / `until` | Session time bounds, ISO or relative | `since:7d` |
| `contains` | Exact content substring filter | `contains:sqlite` |
| `near` | Vector similarity from text or a session id | `near:"semantic search"` / `near:id:<session>` |
| `lane` | Retrieval lane | `lane:dialogue` |
| `lineage` | Sessions sharing topology with a seed | `lineage:id:<session>` |

Field values support quoted strings and in-field alternatives:

```bash
polylogue 'origin:(codex-session|claude-code-session) title:"query DSL"'
polylogue 'tool:bash AND NOT tag:stale'
```

Negation is supported for fields that are semantically safe to negate, such as
`origin`, `tag`, `tool`, and `action`.

### Comparisons And Ranges

Readable comparisons are supported for message counts, word counts, and dates:

```bash
polylogue 'messages >= 5 AND messages <= 20'
polylogue 'words between 100 and 500'
polylogue 'date between 2026-06-01 and 2026-06-17'
```

Compact count syntax is equivalent where available:

```bash
polylogue messages:>=5 words:<=500
```

### Structural Predicates

`exists <unit>(...)` keeps the selected unit as `sessions` but requires at least
one child row matching the nested predicate.

| Unit | Accepted fields |
|------|-----------------|
| `message` | `action`, `command`, `output`, `path`, `role`, `text`, `tool`, `type`, `words` |
| `action` | `action`, `command`, `output`, `path`, `text`, `tool`, `type` |
| `block` | `action`, `command`, `path`, `text`, `tool`, `type` |
| `assertion` | `author`, `author_kind`, `author_ref`, `body`, `context`, `evidence`, `key`, `kind`, `scope`, `scope_ref`, `status`, `target`, `target_ref`, `text`, `value`, `visibility` |

Structural units also accept `session.<field>` predicates for the owning
session fields, such as `session.repo`, `session.origin`,
`session.tag`, `session.title`, `session.date`, `session.since`, and
`session.until`. Count and date session fields accept compact comparison
prefixes such as `session.messages:>=2`, `session.words:<=500`, and
`session.date:>=2026-01-02`. This lets a unit query carry its session scope
inline instead of splitting selection between the query string and parallel
parameters.

Examples:

```bash
polylogue sessions where 'exists action(session.repo:polylogue AND tool:bash AND text:pytest)'
polylogue sessions where 'exists block(type:code AND text:timeout)'
```

### FTS And Semantic Leaves

Inside Boolean predicates, `~` marks an explicit FTS leaf and `semantic:` /
`near:text:` mark semantic-vector leaves:

```bash
polylogue sessions where '~"null pointer" AND repo:polylogue'
polylogue sessions where 'semantic:"query compiler failure"'
polylogue sessions where 'near:text:timeout'
```

Semantic leaves require embeddings to be configured and available. When vectors
are unavailable, the query fails with a typed semantic/vector availability
error instead of falling back to broad lexical search.

## Terminal Unit Queries

Most query expressions select sessions. Explicit unit sources select terminal
rows instead:

```bash
polylogue --format json messages where role:assistant AND text:timeout
polylogue --format ndjson actions where session.repo:polylogue AND action:file_edit AND path:polylogue/archive
polylogue --format yaml blocks where type:code AND text:sqlite
polylogue --format json assertions where kind:decision AND status:active AND text:review
polylogue --format json runs where role:subagent AND status:completed AND agent:Explore
polylogue --format json observed-events where delivery_state:acted_on AND text:#2100
polylogue --format json context-snapshots where boundary:session_start AND session.repo:polylogue
polylogue --format json messages where text:timeout | group by role | count
```

The row shape is the shared `QueryUnitEnvelope` used by CLI JSON/NDJSON/YAML,
Python `Polylogue.query_units()`, MCP `query_units`, and daemon
`GET /api/query-units?expression=...`. Plain and CSV CLI output are
transport-specific renderings of the same message/action/block/assertion/run/
observed-event/context-snapshot row payloads.
Aggregate pipelines over SQL-backed terminal rows use the sibling
`QueryUnitAggregateEnvelope` and currently support `group by FIELD | count`
over closed unit fields such as message role, action tool/action, block type,
assertion kind/status, and owning-session origin/repo.
Those surfaces share the same session-scoping filters for the row source
where applicable, such as origin, tag, repo, title, date bounds, message-type
and tool/paste/thinking feature filters.

Use `session.<field>` inside the expression when the unit rows should be scoped
by their owning session:

```bash
polylogue messages where session.origin:claude-code-session AND role:assistant
polylogue actions where session.repo:polylogue AND action:file_edit
polylogue blocks where session.since:7d AND session.words:<=500 AND type:code
polylogue assertions where session.repo:polylogue AND kind:caveat
polylogue runs where session.repo:polylogue AND role:subagent AND status:completed
polylogue observed-events where session.origin:codex-session AND object_ref:github-review
polylogue context-snapshots where session.messages:>=2 AND session.date:>=2026-01-02 AND boundary:subagent_start
```

`runs`, `observed-events`, and `context-snapshots` are runtime-transform row
sources. They return projected evidence from existing recovery/run-projection
transforms, not durable SQL table rows, so they are terminal unit sources only;
they do not act as `exists run(...)`, `exists observed-event(...)`, or
`exists context-snapshot(...)` session selectors.

Session filters such as `--origin`, `--tag`, `--repo`, `--since`, and `--until`
still narrow the owning sessions before rows are returned. Session-only actions
and result-shaping modes that do not have row semantics yet, such as
`delete`, `open`, `stats`, `count`, `--cursor`, and custom sort/reverse modes,
are rejected instead of silently coercing row queries back to session queries.

## Assertion Candidate Judgment

Transform and recovery jobs may emit assertion rows with `status:candidate`.
Candidate rows are private, carry `context_policy.inject=false`, and keep
`promotion_required=true` until an operator makes an explicit judgment. They
can be inspected like any other assertion row:

```bash
polylogue assertions where 'status:candidate AND target:session:codex-session:abc123' --format json
polylogue ops state candidates list --target-ref session:codex-session:abc123 --format json
```

The judgment workflow lives under `polylogue ops state candidates` and writes
back into the same `user.db` assertion substrate:

```bash
polylogue ops state candidates accept assertion:candidate-id --reason "confirmed by transcript" --format json
polylogue ops state candidates reject assertion:candidate-id --reason "unsupported by evidence" --format json
polylogue ops state candidates defer assertion:candidate-id --reason "needs another source" --format json
polylogue ops state candidates supersede assertion:candidate-id --kind summary --body "replacement claim" --format json
```

`accept` and `supersede` create an active assertion whose evidence includes the
candidate assertion ref and whose `supersedes` lineage points at the candidate.
`reject` and `defer` preserve a judgment assertion with the reason and leave a
durable lifecycle record. No candidate assertion is injected into compiled
context unless a later surface asks for candidates explicitly.

Pending candidate judgments also appear in the operator debt cockpit:

```bash
polylogue ops debt list --kind assertion-candidate --only-actionable --format json
```

## Public Ref Resolution

Query/read payloads carry public object and evidence refs so agents and the web
shell can jump from a row, work packet, or assertion back to the exact archive
object it cites. Resolve refs through the shared resolver rather than turning
them into broad text search:

```bash
polylogue read session:codex-session:abc123 --format json
polylogue read message:codex-session:abc123:m1 --format json
polylogue read block:codex-session:abc123:m1:0 --format json
polylogue read assertion:assertion-id --format json
```

The same `PublicRefResolutionPayload` is exposed by
`Polylogue.resolve_ref()`, MCP `resolve_ref`, and daemon
`GET /api/refs/resolve?ref=...`. The resolver supports session, message,
block, assertion, and runtime projection refs (`run`, `observed-event`,
`context-snapshot`) when the addressed object exists. Unsupported or missing
refs return a bounded unresolved payload with caveats; they never widen into a
session search.

## Outbound OTel Projection

Polylogue has two telemetry directions that should not be confused:

- inbound OTLP receiver routes store external telemetry in `ops.db` for local
  correlation;
- outbound OTel projection exports Polylogue archive evidence as an
  observability-shaped view.

Outbound projection is not archive authority. Trace/span ids are stable export
ids, while Polylogue refs remain the navigation surface back to canonical
sessions, messages, runs, context snapshots, observed events, assertions, and
evidence refs.

The Python API exposes the first bounded projection surface:

```python
payload = await archive.export_otel(
    source_ref="session:codex-session:abc123",
    expressions=(
        "runs where session.id:codex-session:abc123",
        "observed-events where session.id:codex-session:abc123",
        "context-snapshots where session.id:codex-session:abc123",
    ),
)
```

`OtelProjectionPayload` currently emits OTLP-like JSON (`format="otlp-json"`)
over existing query-unit row payloads. Runs and actions become spans;
messages, observed events, and context snapshots become log/event records.
Tool outputs and absolute local paths are omitted by default; the payload
instead carries output length/presence, redaction flags, and refs that clients
can resolve deliberately when the operator wants to inspect source evidence.

## Filters

### Identity and content

| Flag | Description |
|------|-------------|
| `--id`, `-i` | Session ID (exact or prefix match) |
| `--contains`, `-c` | FTS term (repeatable = AND) |
| `--exclude-text` | Exclude sessions matching this term |
| `--title` | Title contains substring |
| `--origin`, `-o` | Include origins (comma = OR) |
| `--exclude-origin` | Exclude origins |
| `--repo`, `-r` | Filter by repository name |
| `--referenced-path` | File path contains substring (repeatable = AND) |
| `--cwd-prefix` | Working directory starts with this prefix |

### Content type

| Flag | Description |
|------|-------------|
| `--has-tool-use` | Only sessions with tool calls |
| `--has-thinking` | Only sessions with reasoning/thinking blocks |
| `--has-paste` | Only sessions with pasted content |
| `--typed-only` | Only typed (non-pasted) content |
| `--has`, `--has-type` | Filter by content type: `thinking`, `tools`, `summary`, `attachments` |

### Message stats

| Flag | Description |
|------|-------------|
| `--min-messages` | Minimum message count |
| `--max-messages` | Maximum message count |
| `--min-words` | Minimum word count |
| `--message-type` | Filter by message type |

### Semantic actions

| Flag | Description |
|------|-------------|
| `--action` | Require semantic action: `file_read`, `file_write`, `file_edit`, `shell`, `search`, `web`, `agent`, `subagent`, `git` (repeatable = AND) |
| `--exclude-action` | Exclude semantic action (repeatable = AND) |
| `--action-sequence` | Require ordered action subsequence (comma-separated) |
| `--action-text` | Text match within action evidence (repeatable = AND) |
| `--tool` | Require normalized tool name (repeatable = AND) |
| `--exclude-tool` | Exclude normalized tool name (repeatable = AND) |

### Time and scope

| Flag | Description |
|------|-------------|
| `--since` | Only sessions on or after this date/time |
| `--until` | Only sessions on or before this date/time |
| `--limit`, `-n` | Maximum results |
| `--offset` | Start offset |
| `--latest` | Newest-first sort |
| `--sort` | Sort order |
| `--reverse` | Reverse sort direction |
| `--sample` | Random sample of N sessions |

### Tags

| Flag | Description |
|------|-------------|
| `--tag`, `-t` | Include tags (comma = OR, supports `key:value`) |
| `--exclude-tag` | Exclude tags |

### Retrieval

| Flag | Description |
|------|-------------|
| `--retrieval-lane` | Query lane: `auto`, `dialogue`, `actions`, `hybrid` |
| `--similar` | Semantic similarity query (requires embeddings) |

### Output modifiers

| Flag | Description |
|------|-------------|
| `--no-code-blocks` | Strip code blocks from output |
| `--no-tool-calls` | Strip tool call blocks |
| `--no-tool-outputs` | Strip tool result blocks |
| `--no-file-reads` | Strip file read blocks |
| `--prose-only` | Show only authored prose text |
| `--dialogue-only` | Show only user/assistant messages |
| `--message-role` | Filter by role (`user`, `assistant`, `system`, `tool`) |

## Verbs

Verbs determine the action applied to the matched session set.

| Verb | Description |
|------|-------------|
| `read --all` | List/export matched sessions with metadata |
| `analyze --count` | Print count of matched sessions |
| `analyze --by ...` | Grouped statistics (`origin`, `month`, `year`, `day`, `action`, `tool`, `repo`, `work-kind`) |
| `read` | Display session content through read views |
| `read --to browser` | Open session in browser |
| `read --all --format ...` | Export matched sessions |
| `read --view messages` | Show individual messages |
| `read --view raw` | Show raw (unparsed) session data |
| `select` | Select and print a single field |
| `delete` | Delete matched sessions (requires `--dry-run` confirmation) |

## Retrieval Lanes

Lane selection lives on the query as `retrieval_lane`. The resolved value
appears in the `SearchEnvelope.retrieval_lane` field on every response so
consumers can tell what actually ran (which matters because `auto` may
elevate — see below).

| Lane | Description | Score kind |
|------|-------------|------------|
| `auto` | Surface left the lane to the planner. May elevate to `hybrid` when embeddings are enabled and an FTS query is present (see [Auto Elevation](#auto-elevation)). | depends on chosen lane |
| `dialogue` | FTS5 over message text (`messages_fts` virtual table, `unicode61` tokenizer). Default lexical lane. | `bm25` |
| `actions` | FTS5 over tool-use/tool-result block text in `messages_fts`. Targets tool/file/shell evidence rather than prose. Public ranked-hit payloads currently carry action rank/evidence without a numeric action BM25 score. | `null` |
| `hybrid` | Reciprocal Rank Fusion combining FTS5 and vector similarity (requires embeddings). | `rrf` |
| `semantic` | Pure vector similarity over Voyage-4 embeddings via sqlite-vec. Triggered by `--similar` or `--semantic`. | `vector_distance` |

Implementation: `polylogue/storage/search_providers/fts5.py`,
`polylogue/storage/search_providers/hybrid.py`,
`polylogue/storage/search_providers/sqlite_vec_support.py`.

### Lane Semantics

#### `dialogue` (FTS5 lexical)

- Backed by SQLite FTS5's BM25 implementation against `messages_fts`.
- Tokenizer is `unicode61` — case-insensitive for ASCII, Unicode-aware
  tokenization. **Porter stemming is not available** in this SQLite build,
  so `refactor` and `refactoring` are distinct tokens. Use prefix queries
  (`refactor*`) when you want morphological breadth.
- Raw score is **BM25**: lower is better in SQLite FTS5, values are
  typically negative, and they are **not comparable across queries**.
- Match evidence: `matched_terms`, `snippet`, `match_surface="message"`,
  `message_id`, and `target_ref` point at the hit message.

#### `actions` (FTS5 over action blocks)

- Same FTS5 mechanics as `dialogue`, but the query is restricted to
  `tool_use` and `tool_result` blocks inside `messages_fts`. The normalized
  `actions` view remains the structured action surface for filters and
  analytics.
- Current public action-lane hits preserve rank and action match surface
  but do not expose the underlying action FTS BM25 score in the shared
  `SearchEnvelope`; consumers should treat `score_kind=null` as the
  contract for action-only hits until the action evidence path is widened.
- Useful when you remember an action ("the session where I edited
  `connection_profile.py`") rather than its prose.

#### `hybrid` (RRF fusion)

- Runs both `dialogue` (FTS5) and `semantic` (vector) lanes, then fuses
  with **Reciprocal Rank Fusion** at `k=60`:
  `fused_score = Σ 1 / (k + rank_in_lane)`.
- Tie-breaking is deterministic: descending fused score, then ascending
  `session_id`. This makes cursor and offset pagination stable
  across runs even when scores tie.
- Reported `score_kind` is `"rrf"`. Higher fused scores indicate stronger
  cross-lane consensus.
- **Lane contributions** (per-lane rank and per-lane RRF contribution)
  are preserved end-to-end on each hit's `score_components`
  ([#1267](https://github.com/Sinity/polylogue/issues/1267)): every lane
  that contributed adds a `<lane>_rank` (1-based rank within that lane)
  and a matching `<lane>_rrf` (the `1 / (k + rank)` contribution that was
  summed into the fused score). Lane names are `text` (FTS5 dialogue),
  `action` (FTS5 action blocks), and `vector` (semantic). A hit that
  appeared only in the lexical lane carries `{text_rank, text_rrf}` and
  nothing else; a hit that survived both lanes carries the full
  `(text|action|vector)_(rank|rrf)` set, so consumers can show "ranked
  high in both lexical and semantic lanes" without re-running the
  search.

#### `semantic` (vector-only)

- Pure k-nearest-neighbor over Voyage-4 1024-dim embeddings via
  sqlite-vec's `vec0` virtual table.
- Triggered by `--similar <text>` or `--semantic` (which promotes the
  positional query string into `similar_text`; no FTS leg runs).
- Score kind is `"vector_distance"` — lower means closer in embedding
  space. Like BM25, distances are not directly comparable across
  different query embeddings.
- Requires embeddings to be enabled and populated; see
  [docs/architecture.md § Embedding Pipeline](architecture.md#embedding-pipeline).
- Use `polylogue ops embed status` to check whether vector retrieval is disabled,
  missing an API key, pending backlog catch-up, partially usable, or complete.
  `polylogue ops embed status --detail` performs exact pending-message and
  retrieval-band accounting; the default status path stays cheap and reports
  the latest persisted catch-up run.

### Embedding Activation And Catch-Up

Semantic search stays unavailable until embeddings are both enabled and
materialized. The activation path is deliberately bounded:

1. `polylogue ops embed status` shows config state, key presence, coverage,
   configured model/dimension, monthly cost cap, backlog, latest catch-up
   progress, and `next_action` (`code`, `reason`, `command`) for automation.
2. `polylogue ops embed preflight --max-sessions 10` estimates the next
   bounded window without contacting Voyage. Use `--format json` for the
   scriptable form: it reports the exact window, pricing assumptions,
   effective cost cap, and a ready-to-run `backfill_args` list for the same
   bounded catch-up slice.
3. `polylogue ops embed enable --yes` enables the daemon stage when a Voyage key is
   already configured, or `polylogue ops embed enable --voyage-api-key ...` records
   the key and enables the stage.
4. `polylogue ops embed backfill --max-sessions 10` runs an explicit bounded
   catch-up batch; after enablement, `polylogued` also processes bounded daemon
   batches for new or stale sessions.

`--max-messages` is a hard message-count window and uses live message counts
rather than potentially stale `session_stats`. `--max-sessions` may
use materialized session stats so small first batches remain fast on large
archives.

MCP clients should use `embedding_status` for readiness/next-action state and
`embedding_preflight` for the same no-provider-call cost window. Both tools are
read-only and return the canonical JSON payloads used by the CLI, so agents do
not need to scrape terminal output before deciding whether semantic search is
actually usable.

Python API clients use the same contracts through
`Polylogue.embedding_status(detail=False)` and
`Polylogue.embedding_preflight(...)`.

### Auto Elevation

When `retrieval_lane=auto` (the default), the planner picks a concrete
lane based on archive state and the query shape:

| Condition | Resolved lane |
|-----------|---------------|
| `--lexical` flag set | `dialogue` (forced FTS-only, even with embeddings enabled) |
| `--semantic` flag set, or `--similar <text>` given | `semantic` (vector-only) |
| Embeddings retrieval-ready + FTS query present | `hybrid` |
| Otherwise | `dialogue` |

The elevation rule lives in
`polylogue/cli/query.py:_maybe_elevate_to_hybrid`. The resolved lane is
always echoed back in `SearchEnvelope.retrieval_lane` so callers do not
have to re-derive what ran.

Two ergonomic overrides on the root CLI surface ([#1217](https://github.com/Sinity/polylogue/issues/1217)):

- `--lexical` — force the FTS-only lane; useful when you want
  deterministic keyword matches regardless of embedding state.
- `--semantic` — promote the query string into a vector-only similarity
  probe; no FTS leg, no boolean operators applied.

## Ranking Policy

Every `SearchEnvelope` declares its `ranking_policy` and
`ranking_policy_version`. The current policy identifier is
`mixed-bm25-rrf-vector` (version `1`):

- `dialogue` orders hits by FTS5 BM25 (lower is better; raw scores are
  usually negative).
- `actions` orders through the action FTS read model, but the
  public action-lane hit payload does not currently expose a numeric
  action score.
- `hybrid` fuses dialogue + action + semantic lanes with RRF at `k=60`
  and orders by fused score, breaking ties on `(−fused_score,
  session_id)`.
- `semantic` orders by ascending vector distance.

Consumers should pin the `ranking_policy_version` they validate against
and treat any change as a contract event. The version is intentionally
exposed so external pipelines can detect ordering shifts without diffing
raw scores. See `docs/openapi/search.yaml` (`x-polylogue-ranking-policy`)
for the machine-readable declaration.

## SearchEnvelope Contract

All ranked surfaces (CLI `--format json`, MCP `search`/`list_sessions`,
daemon `GET /api/sessions?query=…`, Python API) return the typed
`SearchEnvelope` defined in `polylogue/surfaces/payloads.py` and emitted
via the daemon under
[`docs/openapi/search.yaml`](openapi/search.yaml) ([#1266](https://github.com/Sinity/polylogue/issues/1266)).

| Field | Meaning |
|-------|---------|
| `hits` | Ordered list of `SessionSearchHitPayload`. Each hit carries a `session` summary plus a `match` evidence block. |
| `total` | Total matching sessions, or `null` when the lane cannot compute it cheaply. |
| `limit` / `offset` | Applied page size and row offset. Offset-based pagination is **best-effort** for ranked results. |
| `next_offset` | Convenience offset pointer; only set when more results are likely. |
| `next_cursor` | Opaque keyset cursor encoding rank, score, session id, and resolved retrieval lane. **Preferred** for stable rank-first pagination across pages — pass it back unchanged in the next request. |
| `query` | The FTS query text actually applied after CLI/MCP/HTTP coercion. Empty when no FTS query was given. |
| `sort` | Applied explicit sort field (`"date"`, `"messages"`, `"words"`, etc.) or `null` to preserve the lane's natural rank order. Ranked search will not silently fall back to date sort. |
| `retrieval_lane` | Resolved lane that actually ran (`dialogue` / `actions` / `hybrid` / `semantic` / `auto`). |
| `ranking_policy` / `ranking_policy_version` | Declared ordering semantics; see above. |
| `diagnostics` | Optional `QueryMissDiagnosticsPayload` when the query produced zero hits but filters were applied. |

Each hit's `match` (a `SessionSearchMatchPayload`) carries:

| Field | Meaning |
|-------|---------|
| `rank` | 1-based position in the result list. |
| `retrieval_lane` | Lane that produced this hit (matches envelope, unless future per-hit attribution differs). |
| `match_surface` | Indexed surface that matched — e.g. `message`, `action`, `hybrid`, `semantic`, or `attachment`. |
| `score` | Raw lane score (semantics depend on `score_kind`). |
| `score_kind` | One of `"bm25"`, `"rrf"`, `"vector_distance"`, or `null` for identity-only lanes. **Always check this before comparing or ordering by `score` directly.** |
| `score_components` | Map of per-component contributions explaining the rank. Dialogue (FTS5) hits carry `{"bm25_raw": <relevance>}`; hybrid hits carry per-lane `<lane>_rank` and `<lane>_rrf` entries summed into `score` ([#1267](https://github.com/Sinity/polylogue/issues/1267)). Identity-only lanes (e.g. attachment) carry `{}`. |
| `lane_rank` / `lane_contribution` | Primary lane rank and contribution when the backend can identify one. For hybrid, this is the strongest contributing RRF lane; all lane details still live in `score_components`. |
| `raw_score` | Backend-native score before public interpretation. For FTS this is BM25 relevance; for hybrid it is the fused RRF score. |
| `matched_terms` | FTS terms that triggered the match. |
| `snippet` | Highlighted excerpt around the match (FTS5 snippet). |
| `message_id` / `target_ref` / `anchor` | Stable identifiers pointing the reader at the matching message or sub-block. |
| `actions` | Per-target reader action availability (open, copy-link, etc.). |

### Score-kind cheatsheet

- `bm25` — lower = better match in SQLite FTS5. Values are typically negative.
  **Never display raw BM25 as a percent or compare across queries**;
  rank position is the durable signal.
- `rrf` — higher = better; bounded by `Σ 1/(k+1)` over contributing
  lanes. Per-lane decomposition lives in `score_components` as
  `<lane>_rank` / `<lane>_rrf` pairs, so consumers can explain "this
  hit appeared in both lexical and semantic lanes" without re-running
  the query (#1267).
- `vector_distance` — lower = closer in embedding space. Not comparable
  across different query embeddings.
- `null` — identity-bearing match (e.g. attachment identity lane); no
  numeric score, only rank.

### Per-hit explanations ([#1267](https://github.com/Sinity/polylogue/issues/1267))

Every ranked hit carries deterministic why-this-matched evidence on its
`match` payload. The exact field set depends on the lane that produced
the hit:

| Lane | `matched_terms` | `score_kind` | `score_components` |
|------|-----------------|--------------|--------------------|
| `dialogue` (FTS5 over messages) | tokenized query terms (lowercased, FTS5 operators stripped) | `bm25` | `{"bm25_raw": <relevance>}` |
| `actions` (FTS5 over action blocks) | tokenized query terms | `null` today | `{}` today; action rank is preserved, but action BM25 is not part of the public hit evidence contract yet |
| `hybrid` (RRF fusion) | tokenized query terms | `rrf` | `<lane>_rank` and `<lane>_rrf` for every contributing lane (`text` / `action` / `vector`); `score` equals the sum of `*_rrf` |
| `semantic` (vector-only) | the query string passed to `--similar` / `--semantic` (single term) | `vector_distance` | `{}` (raw distance lives in `score`) |
| `attachment` (identity lookup) | the matched identifier (single term) | `null` | `{}` (identity hits have no numeric rank) |

Tokenization for `matched_terms` strips FTS5 boolean operators
(`AND` / `OR` / `NOT` / `NEAR`), quote/colon/paren punctuation, and
trailing `*` prefix markers, then deduplicates case-insensitively. The
result is the literal set of tokens a reader can expect to see
highlighted in the `snippet`.

Hybrid `score_components` are the load-bearing surface: each
contributing lane adds two entries that explain its share of the fused
score. For example, a hit at lexical rank 1 and vector rank 2 carries:

```json
"score_components": {
  "text_rank": 1.0,
  "text_rrf": 0.0163934426,
  "vector_rank": 2.0,
  "vector_rrf": 0.0161290323
},
"score": 0.0325224749
```

Consumers can read `score_components` directly to render a "matched in
both lanes" badge or to debug ranking drift without re-running the
search.

### Pagination

For ranked queries, prefer `next_cursor` over `offset`. Cursor
pagination encodes the rank tie-breaker
(`(rank, score, session_id, retrieval_lane)`) and is stable under
archive growth between page fetches. Offset pagination is supported for
non-ranked list paths and as a best-effort fallback for ranked paths.

The cursor is an opaque URL-safe base64 token (a versioned JSON
envelope; see :class:`polylogue.surfaces.payloads.SearchCursor`).
Consumers MUST treat it as opaque and pass it back unchanged.

```bash
# Page 1
polylogue "sqlite" read --all --format json --limit 25
# Read .next_cursor from the response, then ask for page 2:
polylogue "sqlite" read --all --format json --limit 25 \
    --cursor "$NEXT_CURSOR"
```

MCP search and the daemon `/api/sessions` endpoint accept the same
`cursor` parameter; the Python API exposes `Polylogue.search_envelope(
query, cursor=...)`. The cursor carries the retrieval lane it was
minted in, so a `dialogue` cursor passed back to a `hybrid` request is
rejected up-front rather than silently changing ranking policy
mid-walk.

Stability guarantees (#1268):

- **No duplicates**: any hit returned on page N is filtered out on page
  N+1 even when new rows were ingested between requests.
- **No gaps**: any hit that sorts strictly after the anchor (under the
  lane's natural ordering: BM25 lower-is-better, RRF higher-is-better,
  vector distance lower-is-better) survives the cursor trim.
- **Restart-stable**: cursors are self-contained tokens with no
  server-side state; they survive daemon restart.

## FTS5 Syntax

The `messages_fts` virtual table uses SQLite's FTS5 with the `unicode61`
tokenizer. Prefix queries use `*`:

```bash
polylogue "refactor*"
```

Phrase queries use quotes:

```bash
polylogue '"null pointer exception"'
```

Boolean operators combine terms:

```bash
polylogue "refactor AND schema NOT test"
```

Column filters restrict matches:

```bash
polylogue 'text:css {session_id claude-code}: refactor'
```

## Output Formats

| Format | Description |
|--------|-------------|
| `markdown` | Default -- formatted markdown with syntax-highlighted code blocks |
| `json` | Full session as JSON |
| `jsonl` | One JSON object per line (used by `bulk-export`) |
| `yaml` | YAML representation |
| `plaintext` | Plain text, no formatting |
| `html` | HTML with Pygments syntax highlighting |
| `obsidian` | YAML frontmatter + markdown body |
| `org` | Org-mode format |
| `csv` | Messages as rows |

Set format with `-f` / `--format` on a verb:

```bash
polylogue "sqlite locking" read --all --format json
polylogue --since yesterday bulk-export --format jsonl
```

## Facets (Scoped vs Global)

Facets summarize the archive as aggregate counts. Polylogue exposes the
same shape across daemon HTTP (`GET /api/facets`), MCP (`facets`), CLI
(`polylogue facets`), and the Python API (`Polylogue.facets`); see
[#1269](https://github.com/Sinity/polylogue/issues/1269) (slice D of
[#873](https://github.com/Sinity/polylogue/issues/873)).

A facets response carries both views explicitly:

- `scoped` — counts rolled from the current query/filter set. Empty
  buckets if the filter chain narrows away every value.
- `global` — counts over the unfiltered archive. Always populated.
- `scoped_to_query` — `true` whenever any filter narrowed the view.
- `idf` — inverse-document-frequency per facet value, computed against
  the global universe. Higher = rarer = stronger signal; near zero =
  value appears in almost every session. Disable with
  `--no-idf` on the CLI.

Top-level fields (`origins`, `tags`, `total_sessions` etc.)
mirror the *active* view (scoped when filtered, global otherwise) for
backward compatibility with surfaces written before #1269. Consumers
that need both views should read `scoped` and `global` directly.

```bash
polylogue facets                     # global only (no filters)
polylogue facets -o chatgpt-export   # scoped to ChatGPT exports + global side-by-side
polylogue facets -q "vector store"   # scoped to FTS hits
polylogue facets -f json --no-idf    # FacetsResponse, no IDF weighting
```

## Empty Result Diagnostics

When a query returns no results:

1. Check origin spelling: `polylogue --origin claude-code-session read --all` (not `claude_code`)
2. Expand the time window: `--since 2024-01` instead of `--since yesterday`
3. Verify the archive has data: `polylogue analyze --count` (no filters)
4. Check FTS index health: `polylogued status` shows `fts_readiness`
5. Run `polylogue ops doctor` for schema and index integrity
6. If using `--similar`, ensure embeddings are built (check `polylogue ops embed status --detail` for embedding readiness/coverage)
