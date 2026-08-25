---
created: 2026-08-25
purpose: Bounded continuation-3 of polylogue-rrxe4.8 — five remaining cross-tier relation and public-mask families
status: complete — all five families dispositioned; no genuinely new current-route defects found
project: polylogue
---

# Live cross-tier continuation audit — pass 3

## Authority and method

Direct continuation of the rrxe4.8 audit chain. Prior passes:
- `2026-08-24-live-db-correctness-audit.md` — session/message/topology/attachment/embeddings/ops-tier families
- `2026-08-25-live-route-mask-continuation.md` — cost/usage public projection; durable assertion surfaces
- `2026-08-25-live-route-mask-continuation-2.md` — six remaining unexamined cells (work_event_breakdown, SessionProfileInsight parity, CostRollupInsight per_model, subscription_credits, MCP raw/rich session parity, cost-outlook equivalence)

Frozen physical snapshot: same as prior passes. Daemon deliberately masked. No live archive mutations.

Evidence basis: static code inspection at exact line numbers plus bounded live SQLite queries with declared deadlines. All queries completed within their bounds; none were cancelled.

---

## Coverage matrix — five families × audit dimensions

| Family | Stale-row state | Public search/read path | Route mask | Novelty disposition |
|---|---|---|---|---|
| 1. Embedding refs / vector metadata | 4,186 absent message_ids; 27 stale session/status rows | JOIN on `messages` filters stale refs silently; pool starvation ~2.2% | HTTP `/similar`: partial inconsistency → `status="ready"` with `unresolved_message_hits`; MCP: drops silently | **Already owned** — `feu0`/`embeddings-retention`; route disagreement within `blpir` |
| 2. Durable assertions / ObjectRefs | 4 dangling session-target assertions | Assertion ObjectRefs are session/message scope; session_id stable through reindex when native_id preserved | MCP/HTTP: assertions not exposed by design | **Already owned** — `b4n2`/`auhr8` |
| 3. Audit continuity heads | Genesis-zero head; all operation tables empty | No past mutations; no stale target identities | No public route exposes raw continuity head | **Clean** — genesis state, no owner needed |
| 4. Ops debt/attempts | 3 FTS debts — targets exist, 0 FTS rows, retry-due since 2026-07-31 | `status=failed` conflates deferral; targets are current sessions | Convergence debt summary correctly reports `failed_count=3` | **Already owned** — `tas4`/`awy5` |
| 5. Daemon HTTP / MCP masks | FTS stale → silent partial results (no degraded signal); embedding stale → pool starvation signal disagreement | HTTP: FTS SQLite error → `route_state=degraded`; HTTP: semantic stale → silent; MCP: no FTS error wrapper | Route disagreement on `unresolved_message_hits` HTTP vs. MCP | **Already owned** — `txhkx` (FTS stale), `blpir` (route parity) |

---

## Detailed findings

---

### Family 1 — Embedding refs / vector metadata / stale cross-generation rows / public search path

**Denominator:**
- 187,888 `message_embedding_refs` rows (established in prior audit)
- 4,186 absent from current `messages` table (feu0 cohort, current index gen-1785377665711-06297b00)
- 27 status/derivation rows targeting sessions absent from active index
- Stale-ref rate: 4,186 / 187,888 = **2.23%**

**Public semantic search path trace:**

1. `SqliteVecQueryMixin._query_unlocked` (sqlite_vec_queries.py:155–192):
   ```sql
   SELECT r.message_id, hits.distance
   FROM message_embeddings WHERE embedding MATCH ? AND k = ?
   JOIN message_embedding_refs r ON lower(hex(r.embedding_input_hash)) = hits.embedding_input_hash
   ORDER BY hits.distance
   ```
   Returns all matching message_ids including stale cross-generation refs.

2. `ArchiveStore.semantic_summaries` (archive.py:5877–6003):
   ```sql
   SELECT m.message_id, m.session_id, s.origin, s.native_id, s.title, b.block_id, b.text
   FROM messages m JOIN sessions s ON ... LEFT JOIN blocks b ON ...
   WHERE m.message_id IN (?)
   ```
   Filters via `messages` JOIN (current index.db). Stale message_ids have no row in the current `messages` table — they resolve to `None` in `rows_by_message_id.get(message_id)` (line 5983) and are silently skipped (line 5984–5985).

   **Verdict**: No stale hits surface in public search results. Pool starvation at ~2.23% (bounded by feu0 historical residue fraction). Not a new defect.

**HTTP `/api/sessions/:id/similar` route — partial inconsistency gap:**

`build_similar_payload` (similarity.py:192–218):
- `if unresolved and not hits:` → `status="inconsistent"` (correct for total inconsistency)
- `if unresolved and hits:` → `status="ready"` with `unresolved_message_hits=N` (partial inconsistency transparent via field, but status not degraded)

The docstring at line 234–248 defines the `"ready"` / `"inconsistent"` vocabulary but does not define a `"degraded"` state for partial inconsistency. Consumers checking only `status=="ready"` cannot distinguish full-pool results from a partially-degraded pool. With 4,186 stale refs across 187,888 total, a session-seeded pool of 150 would have ~3 stale hits dropped silently with `status="ready"` and `unresolved_message_hits=3`.

**MCP session-seeded search vs. HTTP route disagreement:**

The MCP `query` operation with a `near:session-id:` spec routes via `archive_search_payload` → `archive_search_hits` → `_pair_hits` (archive_execution.py:702–712). The `_pair_hits` function drops unresolved hits silently at `archive.read_summary(hit.session_id)` with a `continue` on `KeyError`. No `unresolved_message_hits` field appears in the MCP response.

HTTP `/similar` (session-seeded): exposes `unresolved_message_hits` count
MCP session-seeded `near:` search: drops unresolved silently with no count

**Route disagreement classified within existing owners**: `embeddings-retention` owns ref cleanup; `blpir` owns route parity for all session envelopes. No new Bead.

**Novelty gate**: Already covered by `feu0`→`embeddings-retention` (ref cleanup), `w4koi` (vector key), `blpir` (route parity). The partial inconsistency `status="ready"` is the existing intentional design per similarity.py docstring. **No new defect.**

---

### Family 2 — Durable user assertions / ObjectRefs → candidate-rebuildable identities

**Denominator:**
- 107 assertions (47 candidate, 36 active, 12 accepted, 12 rejected) — from prior audit
- 24 assertion-to-assertion targets all resolve — from prior audit
- 4 dangling session-target assertions (ChatGPT `69d5383e`, `6a50b7cc`) — from prior audit

**ObjectRef vocabulary (core/refs.py):**
40+ kinds defined: `session`, `message`, `block`, `attachment`, `paste_span`, `work_event`, `phase`, `thread`, `file`, `branch`, `commit`, `check-run`, `workspace`, `agent`, `user`, `repo`, `insight`, `run`, `context-snapshot`, `observed-event`, `assertion`, `saved_view`, `recall_pack`, `transform`, `tool-call`, `subagent-report`, `github-issue`, `github-pr`, `github-review`, `beads-issue`, `query`, `query-run`, `result-set`, `finding`, `cohort`, `analysis`, `annotation-batch`, `judgment-set`, `ranker`, `elicitation-session`, `experiment-analysis`, `delegation`, `work-invocation`, `work-call`, `work-attempt`, `work-session-segment`, `work-result`, `work-claim`, `work-edge`, `artifact`, `execution-context`, `actor`.

**Candidate-rebuildable identity analysis:**

`session_id = origin || ':' || native_id` (stable across rebuilds when native_id is durable source identity). For the four dangling ChatGPT assertions:
- ChatGPT sessions have `native_id` derived from conversation UUID (provider-native, stable)
- `session_id` remains identical across reindexes unless the source detection/parser rewrites `native_id`
- If the session is successfully acquired in a candidate generation, the dangling assertion becomes applicable again

**Missing/ambiguous/rebound/non-applicable states:**
- Missing: the 4 dangling targets (ChatGPT sessions absent from current index) — owned `b4n2`/`auhr8`
- Ambiguous: none found (no session IDs match multiple archived sessions)
- Rebound: not applicable — the archive does not rebind session_ids, it deduplicates by hash
- Explicitly non-applicable: the `auhr8` task defines the candidate-bound transition for these 4 rows

**Public exposure check:**
- MCP: zero assertion exposure (confirmed in prior pass — by design)
- HTTP: zero assertion exposure
- CLI `polylogue maintenance assertion-export`: JSONL export via `list_assertions_for_export`
- Python API: internal only

**No new family 2 defect.** The 4 dangling assertions are the only correctness issue, owned by `b4n2`/`auhr8`.

---

### Family 3 — Audit authority / continuity heads → mutation attempts/results / current target identities

**Denominator and state (from prior audit + this pass):**
- `audit_continuity_head`: generation=0, SHA256=`AUDIT_CONTINUITY_GENESIS_HEAD_SHA256` (`3230fdd585...`)
- All 8 operation tables in audit.db empty: `operation_previews`, `operation_preview_targets`, `operation_preview_capabilities`, `operation_authorizations`, `operation_authorization_capabilities`, `operation_runs`, `operation_run_capabilities`, `operation_targets`, `operation_attempts`
- `audit_continuity_control` in source.db: `pending_mutation_id = NULL`

**Continuity head applicability:**

The genesis head is the correct state for an archive that has never executed a durable mutation through the audit continuity system. The `audit_continuity.py` coordinator:
- Startup check (line 283): `SELECT 1 FROM audit_continuity_head WHERE singleton = 1` — head exists ✓
- Coherence check (line 415): source and audit generations must agree — they do at generation 0 ✓
- Pending check (line 476): no pending mutation — clean ✓

**Current target identities:**
No mutation attempts exist, so there are no target identities to audit. The current archive instance identity is defined by `archive_authority.archive_instance_id` (written at first init) plus the active index generation pointer. No audit operation has ever advanced beyond the genesis state.

**Public route mask:**
- No MCP tool or HTTP route exposes the raw continuity head directly
- `operations/audit.py::describe_archive_audit` is a CLI-only read
- CLI `polylogue status` shows counts (0 operations)

**No new family 3 defect.** Genesis state is clean. The "four historical daemon lifecycle rows lack stop markers" note from prior audit was about `daemon_lifecycle` (ops.db), not audit.db operation tables — confirmed that `daemon_lifecycle` and audit.db are separate structures.

---

### Family 4 — Ops cursors/debt/attempts → current source/generation/work ownership

**Convergence debt complete inventory (live query, ~0s):**

| stage | target_type | target_id | status | attempts | next_retry_at | last_error |
|---|---|---|---|---|---|---|
| fts | session_id | `claude-code-session:38baa1de-...:agent-a8e5fbcf9a7db727a` | failed | 1 | 2026-07-31T16:12 | "live full ingest deferred FTS to preserve writer availability" |
| fts | session_id | `claude-code-session:38baa1de-...:agent-a9347188628f8900c` | failed | 1 | 2026-07-31T16:16 | "live full ingest deferred FTS to preserve writer availability" |
| fts | session_id | `claude-code-session:38baa1de-...:agent-aa320a9df0dd977a3` | failed | 1 | 2026-07-31T16:19 | "live full ingest deferred FTS to preserve writer availability" |

**Target identity verification (live query, ~4s against index.db):**

All 3 target sessions exist in current index (gen-1785377665711-06297b00):
- `agent-a8e5fbcf9a7db727a`: 448 messages; 0 FTS rows
- `agent-a9347188628f8900c`: 315 messages; 0 FTS rows
- `agent-aa320a9df0dd977a3`: 20 messages; 0 FTS rows

All 3 sessions are children of `38baa1de-9715-48fa-8175-f2a29d92800e` (historical `t0m73` session). The `last_error` value is a deferral message, not an error. `status=failed` with 1 attempt and `next_retry_at` well in the past means these are retry-due work items.

**Semantic state conflation:**
`status=failed` is used for both genuine failures and deferral-not-yet-retried cases. The `convergence_debt` schema has a `deferred` status, but the deferral path wrote `failed` here. This means `retry_due_count` in `ConvergenceDebtSummary` counts these as failed rather than deferred. The public surface (daemon status endpoint) exposes `failed_count=3` for FTS debts, but this conflates "truly failed" with "deferred once and never retried."

**Existing owner**: This conflation is within `awy5` scope (honest attempt aggregation). The 3 rows are real current work items (sessions exist, no FTS coverage, retry-due), but the daemon is deliberately masked so they remain unexecuted.

**Other ops-tier state:**
- `daemon_stage_events`: 47,023 rows (224 `append_parse|running`, 4,296 `completed|completed`, 6,526 `convergence|running`, etc.) — all `running` rows from daemon kills, expected residue
- `fts_drift_samples`: 28 rows; 5 `messages_fts|stale` with 12,371–35,331 missing rows (2026-08-08 mtime); 5 `threads_fts|stale` with 10 missing — matches prior audit
- `route_observations`: 7 rows, all `cli.status|compact|direct|error` with `daemon_reachable=false` — expected with masked daemon
- `embedding_catchup_runs`: 0 rows (daemon masked, no catchup runs since 2026-08-18 mtime on embeddings.db)

**No new family 4 defect.** All states trace to existing owners (`tas4`, `awy5`, `txhkx`).

---

### Family 5 — Daemon HTTP and MCP masks

**FTS stale — unknown-as-success:**

HTTP `/api/sessions` FTS path (http.py:3154–3200):
- `_search_index_degraded_reason(exc)` catches `DatabaseError`/`sqlite3.Error` where message contains "fts" or "messages_fts"
- Returns `route_state=degraded`, `total=None`, `hits=[]` — CORRECT
- Stale FTS (semantically incomplete but no exception): returns partial results with `route_state=ready`/`no_results`, `total=N` (lower than true total) — **silent partial results**

MCP `query` FTS path (archive_support.py:458–485):
- `archive.search_summaries(query, ...)` → no FTS error wrapper
- FTS exception would propagate unhandled
- Stale FTS: same silent partial results as HTTP
- **No extra degradation signal versus HTTP** — both surfaces fail silently on FTS staleness

**Existing owner**: `txhkx` (FTS staleness), `i3i5k` (zero post-promotion FTS debt), `tas4` (FTS consolidation).

**Embedding search route disagreement:**

HTTP `/api/sessions/:id/similar` (http.py:4561, similarity.py):
- Partial inconsistency (`unresolved > 0 and hits > 0`): `status="ready"`, `unresolved_message_hits=N` (transparent)
- Full inconsistency (`unresolved > 0 and hits == 0`): `status="inconsistent"` (explicit degradation)

MCP session-seeded `near:session-id:` (archive_support.py:432, archive_execution.py:702):
- `_pair_hits` silently drops unresolved (no `unresolved_message_hits` in MCP response)
- Full inconsistency: empty result set with no explicit status code

Route disagreement: HTTP makes partial inconsistency transparent via `unresolved_message_hits`; MCP does not. Existing owner: `blpir` (route parity).

**Embeddings MCP status (scope=embeddings) — no omission:**
`EmbeddingStatusPayload` (status_payload.py:119–155) includes `stale_messages`, `retrieval_ready`, `freshness_status`, `failure_count`, `terminal_failure_count`, `retryable_failure_count`. These accurately surface the embedding state including staleness signals. **No unknown-as-zero or omission defect.**

**Convergence debt public surface — correct:**
`ConvergenceDebtSummary` (convergence_debt_status.py:66–74) exposed through daemon HTTP status and MCP `status(scope="archive")`:
- `failed_count=3` for the 3 FTS debts (correctly non-zero)
- `retry_due_count=3` (correctly non-zero)
- No "unknown-as-zero" mask — the 3 debts surface faithfully

**Pagination completeness for similar/semantic search:**
- HTTP `/similar`: not paginated (limit-capped to `SIMILAR_RESULTS_MAX`); no `next_offset` or cursor; full result set
- MCP session-seeded via `near:`: `total=None` (similarity.py:447), no `next_cursor`
- No pagination defect — expected behavior for similarity (not a resumable scan)

**FTS stale pagination:**
FTS search returns `total` from `count_search_sessions()` (a COUNT on the FTS index), which reflects the stale state. The `total` is lower than the actual session count but not falsely elevated. No overcount.

**No new family 5 defect.** All mask anomalies trace to existing owners.

---

## Existing Beads confirmed as owning residue

| Finding | Owning Bead | Evidence |
|---|---|---|
| Stale embedding refs in public search (pool starvation) | `feu0`→`embeddings-retention` | 4,186 absent refs; 2.23% stale rate; `_pair_hits` silently drops |
| MCP session-seeded vs HTTP `/similar` unresolved count disagreement | `blpir` | HTTP has `unresolved_message_hits`, MCP drops silently |
| 4 dangling session-target assertions | `b4n2`/`auhr8` | ChatGPT `69d5383e`, `6a50b7cc` cohort |
| 3 FTS convergence debt rows (status=failed conflates deferral) | `tas4`/`awy5` | 3 subagent sessions of `38baa1de`, 0 FTS rows, retry-due since 2026-07-31 |
| FTS stale → silent partial search results (no degraded signal) | `txhkx` | `fts_drift_samples` shows 35,331 missing rows; no route_state=degraded for semantic staleness |
| Vector key underspecification (recipe invalidation silent) | `w4koi` | 167,943 NULL `output_contract_hash` rows (prior audit) |
| Audit genesis state (no mutations, no stale targets) | None needed | Clean genesis; no operations ever issued |

---

## Recommended strengthening to existing Beads

**`blpir`** (already P0, route parity rewrite):
- Add: MCP session-seeded `near:` search (`archive_search_payload`, archive_support.py:432) drops unresolved message_hits silently — the `_pair_hits` function has no unresolved count. When `blpir` selects one spec-owned projection, include a `unresolved_message_hits` field or equivalent degradation signal in the MCP response for session-seeded similarity, matching the HTTP `/similar` envelope.
- Cross-reference: HTTP partial inconsistency handling in similarity.py:199 — `status="ready"` vocabulary should be extended with `"partially_degraded"` when `unresolved > 0 and hits > 0`, or the `unresolved_message_hits` field must be explicitly documented as the consumer's degradation signal.

**`awy5`** (attempt aggregation):
- Strengthen: the 3 FTS convergence_debt rows with `status=failed` but `last_error="live full ingest deferred FTS to preserve writer availability"` are deferral-not-failure. The `deferred` status exists in the schema CHECK constraint but is not used here. `awy5` should specify that an initial deferral writes `status=deferred`, and only a genuine execution failure (exception raised by the FTS stage handler) writes `status=failed`. Current code writes `failed` for both, making `retry_due_count` an overcount of genuine failures.

**`embeddings-retention`** (via `feu0` closure):
- Note: the 2.23% stale-ref rate means semantic search pools are overfetched by `3 * limit` to compensate; the stale fraction wastes ~6.7% of pool slots at 3× overfetch. After `message_embedding_refs` is replaced by generation-bound derivation, this pool inefficiency disappears automatically. No separate fix needed.

---

## Commands run and durations

| Query | Result | Duration |
|---|---|---|
| `SELECT debt_type, status, COUNT(*) FROM convergence_debt` | Schema error (no debt_type column) | <1s |
| `SELECT stage, target_type, status, COUNT(*), errors FROM convergence_debt GROUP BY ...` | 3 FTS/session_id/failed rows | <1s |
| `SELECT target_id, status, attempts, next_retry_at, last_error FROM convergence_debt` | Full detail for all 3 rows | <1s |
| `SELECT session_id, message_count FROM sessions WHERE session_id IN (...)` against index.db | 3 rows: 448, 315, 20 messages | <1s |
| `SELECT COUNT(*) FROM messages_fts WHERE session_id = ?` × 3 | 0 for all 3 targets | ~4s total |
| `SELECT kind, COUNT(*) FROM daemon_events GROUP BY kind` | 9 distinct event kinds, 8,374 rows | <1s |
| `SELECT stage, status, COUNT(*) FROM daemon_stage_events GROUP BY stage, status` | 10 distinct stage/status combinations, 47,023 rows | <1s |
| `SELECT surface, state, COUNT(*), ... FROM fts_drift_samples GROUP BY surface, state` | 6 groups; messages_fts stale 12,371–35,331 missing | <1s |
| `SELECT * FROM route_observations LIMIT 10` | 7 rows, all `cli.status|error` with `daemon_reachable=false` | <1s |
| Static code inspection (family 1–5) | Full trace of all five families | ~35 min |

Total investigation time: ~45 minutes. Within the 3000-second hard deadline.

---

## New defects found

**None.** All findings trace to existing Bead owners. No current-route mechanism found that produces required state only through production routes and is not already captured.

---

## Completeness statement and anti-vacuity

**What was covered:**
1. Complete public search/read path trace for stale embedding refs: KNN query → `message_embedding_refs` JOIN → `semantic_summaries` `messages` filter → result set (verified stale refs silently dropped, no stale hits surface)
2. Complete assertion ObjectRef vocabulary and candidate-rebuildable identity applicability for all 107 durable assertions
3. Audit continuity head: genesis-zero state verified against current code startup/coherence checks; all 8 operation tables confirmed empty
4. Convergence debt full inventory: all 3 rows enumerated with targets verified to exist in current index, FTS coverage verified zero, deferral-vs-failure semantic conflation noted
5. HTTP and MCP mask audit for FTS staleness, semantic search degradation, embeddings status, convergence debt surface, and pagination completeness

**Anti-vacuity evidence (checks that would be red if current code were broken):**

- `semantic_summaries` stale-ref filtering: verified by code path at archive.py:5983–5985; a query against a session with only stale embedding refs returns zero hits with `status="inconsistent"` (correct). A mutant that removed the `rows_by_message_id.get(message_id)` check would return stale message_ids as valid hits.
- Convergence debt target existence: directly confirmed 3 sessions (448, 315, 20 messages) exist in index.db; a reindex that removed these sessions would make the debt targets dangling and the retry would correctly fail with a session-not-found error.
- Audit genesis-zero head coherence: the `_require_coherent_state` code path at audit_continuity.py:415 checks that source and audit generations agree; a mismatch would raise `AuditContinuityError` at startup — the genesis-zero state satisfies this check.

**Exclusions (not claimed as covered):**
- FTS search precision/recall audit against the stale index (would require a targeted query probe against known content)
- Full embedding status payload accuracy for all embedding_models/dimensions breakdowns
- Non-FTS convergence debt (no other stages have debt in current snapshot)
- Audit operation lifecycle (no operations have been issued, so the lifecycle code paths are untested in live data)
- The CLI `analyze --similar` command path (distinct from HTTP/MCP similar routes — not examined)

**No completeness claim beyond the five families specified in the audit scope.**
