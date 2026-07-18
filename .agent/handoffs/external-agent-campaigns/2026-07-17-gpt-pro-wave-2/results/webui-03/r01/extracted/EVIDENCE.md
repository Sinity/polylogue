# WebUI v2 search vertical — source and decision evidence

## Evidence authority

The implementation used the extracted repository at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d` as code authority, followed by repository instructions, current Beads records, and history. The worktree was detached at that commit, which is contained by `master`, `origin/master`, and the listed feature refs. No live deployment state was inferred from source.

## Source inspected

### Repository and architecture instructions

- `AGENTS.md`: substrate owns meaning; surfaces project; new production modules require topology declarations; generated files must be rendered, not hand-maintained.
- `CLAUDE.md`: generated-surface commands, topology/layering/degrade-loudly verification, and the note that aggregate render output must be inspected rather than inferred from one status line.
- `docs/architecture-spine.md`: surfaces do not reimplement semantic behavior.
- `docs/search.md`: FTS5, parser fields/booleans, semantic/hybrid retrieval, facets, snippets, and search payload behavior.

### Query owners

- `polylogue/archive/query/expression.py`: real Lark grammar, parser diagnostics, field and `near:` lowering.
- `polylogue/archive/query/spec.py`: immutable request surface and retrieval-lane vocabulary.
- `polylogue/archive/query/plan.py`, `plan_execution.py`, `archive_execution.py`: canonical planning, ranked/filter execution, and count behavior.
- `polylogue/archive/query/search_hits.py`: ranked session-hit projection.
- `polylogue/archive/query/transaction.py`: query identity, result refs, page progression, execution control, and opaque `q1` continuation.
- `polylogue/core/dates.py`: relative date interpretation.

### Storage and API owners

- `polylogue/storage/sqlite/archive_tiers/archive.py`: contentless FTS over `blocks.search_text`, distinct count, summary ordering, message/block tables, and archive read projections.
- `polylogue/api/archive.py`: `Polylogue.search_session_hits`, facets, and query-miss diagnostics.
- `polylogue/storage/fts/*`: readiness/freshness invariant.
- `polylogue/daemon/embedding_readiness.py`: semantic readiness state.
- `polylogue/surfaces/payloads.py`: canonical `session:<id>` and `message:<id>` refs.

### Daemon and current shell

- `polylogue/daemon/http.py`: production route dispatch, authentication, archive-root resolution, and response helpers.
- `polylogue/daemon/route_contracts.py`: declared route/auth/response registry used by tests and OpenAPI.
- `polylogue/daemon/fts_status.py`: FTS readiness payload.
- `polylogue/daemon/web_shell.py`: current search/facet UI and legacy deletion candidates.
- `polylogue/daemon/web_shell_reader.py`: reader route/anchor and credential integration; not a wholesale deletion candidate.

### Tests and examples

- `tests/unit/cli/test_query_expression.py`: exact parser-proven DSL strings and source lines.
- route/security/storage/query/date tests named in `TESTS.md`.
- current Preact/Vite Bead decisions; no production WebUI v2 scaffold was present in source.

## Beads adjudication

### `polylogue-z9gh.9.1` — shared query transaction

Observed requirement: one bounded production read boundary must own complete continuation state, exact/qualified coverage, stable query/result refs, useful evidence, and eventually keyset/snapshot/spool semantics. Surfaces must not own private cursors or totals.

Applied decision: this vertical reuses and hardens `QueryTransactionRequest`, `QueryContinuation`, `QueryResultPage`, and `QueryCoverage`. It does not claim mutation-stable paging; that remains the Bead’s cross-surface work.

### `polylogue-4p1` — Query × Projection × Render

Observed requirement: canonical selection semantics are separate from content projection and render delivery.

Applied decision: `SessionQuerySpec`/plan remains Query; `WebSearchResponse` is the compact browser Projection; JSON and semantic HTML are two render/delivery forms consuming the same projection. No second query model is introduced.

### `polylogue-t46.8` and `polylogue-t46.8.2` — protocol-native read algebra

Observed requirement: HTTP and MCP should converge on query/read/get/explain transactions backed by the shared bounded query owner, not parallel list/search semantics.

Applied decision: the browser utility emits the transport-neutral query request field names. It does not invent an HTTP-only expression object or duplicate MCP execution.

### `polylogue-bby.11` — WebUI architecture v2

Observed decision: TypeScript + Preact + Vite, committed reproducible assets, generated OpenAPI client, shared tokens, and a strangler migration.

Observed contradiction: this snapshot contains no landed canonical asset renderer/static-dist owner/generated client/token module.

Applied decision: add the requested source and tests, define the exact integration seam, and do not create a second scaffold or commit assets into an invented directory.

### `polylogue-1ilk` — browser verification

Observed requirement: Vitest component lane and Playwright e2e/visual lane against the daemon.

Applied decision: implement the Vitest lane now. Mark live Playwright/accessibility/visual verification unverified because the scaffold/live daemon lane is absent.

## History inspected

Relevant history included:

- `9163d0134` — bounded agent-facing archive reads;
- `fd7b35492` — interruptible/admission-controlled archive execution;
- `876358610` — bounded aggregate execution repair;
- `c1f7704fa` — bounded cockpit aggregate routes;
- `0e0cddaee` (from prior history inspection) — first-party web credential bootstrap;
- current base `536a53efa` — raw-authority convergence hardening.

Inference supported by that history: this vertical should compose existing controlled reads and first-party credential hooks rather than create another executor, fetch protocol, or auth mechanism.

## Contradictions found and resolved

### Distinct count versus block-level rows

Observed fact: `count_search_sessions` counted distinct session ids. The prior `search_summaries` SQL selected matching blocks and applied `LIMIT/OFFSET` before any session deduplication.

Consequence: a session with multiple matching blocks could appear twice, exact totals described a different cardinality from page rows, and continuation could skip a logical session.

Resolution: page grouped session matches first, then select one evidence block for each paged session. A real FTS regression fixture proves two pages contain two distinct sessions even when the first has two matching messages.

### Initially inferred message-ref shape versus source owner

Observed fact: the stable payload owner uses `message:<message_id>`, not `message:<session_id>:<message_id>`.

Resolution: Python projection, TypeScript fixtures, runtime validators, and route tests use and enforce the canonical form.

### Relative dates versus continuation replay

Observed fact: parsing `since:7d` against wall-clock time on every page changes the logical query while retaining a nominal continuation.

Resolution: persist a timezone-aware `as_of` in complete request identity and pass it as the relative-date base on every compile.

### Exact semantic totals versus provider bounds

Observed fact: semantic/hybrid execution can be bounded even when an unbounded count path returns some candidate rows.

Resolution: semantic/hybrid coverage is qualified. A full page remains conservatively continuable; no exhaustive claim is made.

### Offset as false evidence

Observed fact found during contract refinement: an empty request at offset 100 does not prove that 100 matching rows exist. A manually altered or overshot token could otherwise inflate `at_least` coverage.

Resolution: offset contributes to observed coverage only when the page contains returned evidence. A continuing page is forbidden from being empty.

### Annotation-only protocol validation

Observed fact: Python dataclass annotations did not reject booleans/floats at runtime, and Base64 decode errors were not all normalized to `ValueError`.

Resolution: shared request and continuation decoding now use strict runtime checks, reject coercion, catch malformed encoding/Unicode/JSON, and preserve the typed error boundary.

### Client runtime shape versus server invariants

Observed fact: the initial TypeScript guard checked field presence but would hydrate malformed provenance, page continuation, timestamps, time buckets, or page sizes.

Resolution: the browser decoder mirrors the load-bearing server invariants and has direct rejection tests.

### Unsupported `semantic` selector

Observed fact: the canonical lane enum offered to this surface is `auto`, `dialogue`, `actions`, `hybrid`; pure vector intent is represented by `near:`.

Resolution: no client-only `semantic` option is exposed.

### “One bounded read” wording versus actual execution

Observed fact: page, count, facets, four time buckets, and evidence use separate controlled reads.

Resolution: final documentation states this explicitly. The evidence projection itself is now one batched SQL read, but the whole response is not a single SQLite snapshot.

## Source-supported decisions

- SSR and JSON must share one typed response to prevent dictionary drift.
- FTS convergence must withhold results rather than render a misleading empty state.
- Missing embeddings must not silently degrade a `near:` query to lexical search.
- selected zero-count facets must remain visible so the user can undo an empty selection.
- continuation append must preserve the server’s row order exactly.
- continuation failure must preserve the already completed page.
- browser back/forward must restore from the daemon rather than use locally cached semantic state.
- errors returned to the browser must be typed and sanitized; detailed exceptions remain logged.

## Unresolved uncertainty

- how the eventual shared transaction will choose among keyset, snapshot re-execution, and spool for every query class;
- how the canonical WebUI scaffold will name its final asset manifest and generated client imports;
- production-scale cost of four time-bucket counts and facet recomputation;
- browser credential timing and accessibility behavior in the deployed daemon;
- package inclusion until Nix/wheel/static-dist owners are available.

No claim is made for those points.
