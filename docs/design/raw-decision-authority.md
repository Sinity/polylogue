# Addressable raw decisions without repeated census snapshots (polylogue-gen6d)

The raw-authority ledger records, for every bounded reconciliation pass, which
replay plans existed, which were selected, and what happened to each. It
re-records that membership **in full, every pass**, so the durable tier grows
with the number of passes rather than with the number of decisions. This
document selects the replacement shape and states, field by field, what is
retained, migrated, reconstructed, or retired.

**It changes no code and mutates no durable archive.** It is the design record
for polylogue-gen6d; the implementation is owned by polylogue-6kur.3 (raw
materialization and aggregate refresh moving into canonical derivations) and
the kernel work in polylogue-bp12n.1. Every file:line below was read at
`5218889` and is the state of the tree at that commit.

Read `docs/internals.md` (schema regimes) and
`docs/design/convergence-simplification-inventory.md` (what the daemon
redesign already committed to deleting) first; this document assumes the
census → replay → materialize pipeline is familiar.

## 1. The problem, dated

Every figure in this section is a number **recorded in the code at the time it
was measured**, quoted with its source. None was re-measured for this document:
the task's scope forbids live archive access and this worktree contains no
archive. Current measurements are a separate, explicitly unrun item (§10).

| Figure | Recorded at | Source |
|---|---|---|
| 6,039 MB across two tables and seven indexes, 89% of a durable tier whose actual evidence (`raw_sessions` + `blob_refs`) was 52 MB, growing ~990 MB/day | polylogue-wkc6 | `polylogue/storage/raw_authority.py:51-59` |
| 99.98% of plan rows ever written carried the `carried_forward` outcome — "nothing happened" | polylogue-wkc6 | `polylogue/storage/raw_authority.py:47-49` |
| `residual_json` + `post_residual_json` average ~144 KB per header; headers alone accrue ~28 MB/day at ~97 censuses/day | polylogue-wkc6 | `polylogue/storage/raw_authority.py:70-73` |
| 41 of the most recent ~112 censuses stuck holding plan rows, ~100k rows/hour, `fixed_point=0` on all 256 censuses ever run | polylogue-z7ko / f4z9 / wkc6 | `polylogue/storage/raw_authority.py:977-979` |
| The correlated-subquery form of the orphan-plan sweep measured ~26 billion ops (>1h) on the live archive; the current form completes in ~0.3s | undated in source | `polylogue/storage/raw_retention.py` (comment above `_PURELY_ORPHANED_AUTHORITY_PLANS_SQL`) |

Two bounded retention windows already exist and are enforced inside
`prune_raw_authority_census_history` (`raw_authority.py:930-1070`), called from
every census write (`raw_authority.py:1269`):

- `RAW_AUTHORITY_CENSUS_PLAN_RETENTION = 8` (`raw_authority.py:60`) — how many
  censuses keep their per-plan membership rows.
- `RAW_AUTHORITY_CENSUS_HEADER_RETENTION = 256` (`raw_authority.py:74`) — how
  many headers survive; a header with any blocker row is exempt
  (`raw_authority.py:1029-1034`), and a surviving header's
  `predecessor_census_id` is nulled at the cut (`raw_authority.py:1043-1061`).

**This is the load-bearing fact for every retirement decision below: history
older than those windows is already gone today.** A design that stops writing
per-pass membership is not losing evidence the archive currently keeps; it is
declining to write rows the next 8 passes would delete anyway. The windows are
also why "move the machinery behind an adapter" fails as an approach: an
adapter over per-pass membership preserves the write cost, the retention
sweep, and the nulled-predecessor ambiguity — the three things that actually
hurt.

What one **unchanged** pass writes today, traced through
`record_raw_authority_census` (`raw_authority.py:1073-1286`):

- Converged archive (no candidates): 1 header row, 0 plan rows, 0 post-plan
  rows, plus the retention sweep. Constant, and already cheap.
- A pass over a non-empty backlog where nothing is selected: 1 header + N
  `raw_authority_census_plans` rows, every one carrying
  `outcome_status='carried_forward'` and the literal reason
  `"bounded scheduler carried this complete plan forward unchanged"`
  (`raw_authority.py:1251-1256`), + N `raw_authority_census_post_plans` rows
  when the lifecycle completes in the same call (`raw_authority.py:1261-1268`).
  N is the count of tracked authority components.

The second shape is the one the dated figures describe, and it is the one this
design removes.

## 2. What must survive

The decisions below are consulted to classify a component as terminally
admitted / rejected / ambiguous rather than replayable. Any redesign that
loses one of them makes the daemon either replay a settled component forever
or treat an unsettled one as done. Verified in
`_raw_replay_plan_outcome` (`polylogue/storage/raw_convergence.py:4608-4772`)
and its readers:

| Durable fact | Where it lives today | Disposition |
|---|---|---|
| `raw_revision_applications.decision` (`deferred`/`ambiguous`/`superseded`/…) | index tier | retain (untouched) |
| `raw_session_memberships.decision` | index tier | retain (untouched) |
| `raw_sessions.parse_error`, `parsed_at_ms`, `revision_authority`, byte-chain columns | source tier | retain (untouched) |
| `raw_membership_census.status` (`complete`/`failed`/`non_session`) | source tier | retain |
| `raw_authority_parser_census.parser_fingerprint` + `SUPERSEDED_MEMBERSHIP_FINGERPRINTS` (`raw_authority.py:42-44`) | source tier | retain |
| `raw_revision_heads.accepted_raw_id` (+ frontier kind/value) | index tier | retain (untouched) |
| Open/resolved authorization blockers (`raw_authority_blockers.resolved_at_ms`); the fail-closed gate is `unresolved_raw_replay_blockers` (`raw_authority.py:900-925`), which refuses the whole pass (`raw_convergence.py:5307-5316`) | source tier | retain, re-keyed (§4) |
| "This component was attempted and made zero typed progress, so stop reselecting it" — the hjpx no-progress rule (`raw_convergence.py:4608-4645`), read back by `raw_replay_plan_no_progress_plan_ids` (`raw_authority.py:839-885`) | **only** in `raw_authority_census_plans.outcome_status` | **migrate to an addressable decision** |
| "This plan's preconditions moved after it was planned" — `rejected_stale` (`reject_stale_raw_replay_plan`, `raw_authority.py:2248-2314`; `reject_invalid_raw_replay_application`, `:2314-2379`) | `raw_authority_census_plans` + a blocker | **migrate** (decision) / retain (blocker) |
| "This component's payload exceeds the daemon parse envelope" — `raw_replay_plan_deferred_for_envelope` (`raw_authority.py:799-838`) | `raw_authority_census_plans` | **reconstruct** — it is a property of `raw_sessions.blob_size` against the current envelope, not a decision |
| "When was this plan last attempted" — `raw_replay_plan_last_attempts` (`raw_authority.py:765-793`), backed by `idx_raw_authority_census_plans_attempts` | `raw_authority_census_plans` | **retire from durable**; fairness position is disposable (ops tier) |

Everything else `raw_authority_censuses` / `raw_authority_census_plans` /
`raw_authority_census_post_plans` holds is per-pass bookkeeping about work that
did not happen.

The direction is already committed in the tree, one phase earlier:
`RawAuthorityVerdict` (`polylogue/core/enums.py:820-850`) is the closed
five-value vocabulary that downstream consumers read "instead of reaching into
the fragmented multi-table bookkeeping", and `raw_authority_verdicts`
(`archive_tiers/source.py:512-524`) is explicitly **not a source of truth** but
a cache keyed by `cohort_fingerprint`, "invalidated by content, not by elapsed
time". This design finishes that move for the write path that
`polylogue/archive/raw_authority_verdict.py:9-19` names as deliberately out of
scope for its own phase.

## 3. Selected design: addressable decisions, disposable iteration

Three rules, in decreasing order of authority.

**(1) A decision is addressed by the content it was made about, not by the
pass that made it.** A replay plan already carries a content digest over its
inputs, witnesses and preconditions — `RawReplayPlan.input_digest`, built in
`build_raw_replay_plan` (`raw_authority.py:631-745`, digest at `:728-738`).
That digest is the decision's address. Recording the same decision again is an
idempotent no-op, so **a pass that changes nothing writes nothing**; and when
the inputs move, the digest moves with them, so a decision can never outlive
the evidence it was made about. Stale-plan detection stops being a postflight
comparison between two passes and becomes a structural property of the key.

**(2) Iteration is disposable.** Which components a bounded pass looked at, in
what order, how many it selected, and when it last attempted one are
scheduling facts. They belong in `ops.db` (disposable tier), bounded by a row
cap, and losing them costs one redundant attempt, never a wrong answer. This
is the same rule the derivation kernel states for its `PassCursor`
(`polylogue/daemon/derivation.py`): a position hint may move where a pass
starts looking and may never certify a key.

**(3) Pending work is derived by inspecting output, never stored.** The set of
components still to do is `required` minus `valid`, computed from
`raw_sessions` + the index-tier receipts at inspection time. There is no
census header to publish, no membership to conserve, and therefore no
interrupted-census recovery path to write.

### 3.1 Durable interface (source tier, additive)

```sql
-- Migration NNN (additive-durable). One row per durable decision, addressed
-- by the plan digest the decision was made about.
CREATE TABLE IF NOT EXISTS raw_replay_decisions (
    plan_input_digest   TEXT PRIMARY KEY CHECK(length(plan_input_digest) = 64),
    decision            TEXT NOT NULL CHECK(decision IN (
                            'terminal_no_progress', 'rejected_stale', 'terminal_classified'
                        )),
    input_raw_ids_json  TEXT NOT NULL CHECK(json_valid(input_raw_ids_json)),
    logical_keys_json   TEXT NOT NULL CHECK(json_valid(logical_keys_json)),
    parser_fingerprint  TEXT NOT NULL,
    reason              TEXT NOT NULL,
    evidence_json       TEXT NOT NULL CHECK(json_valid(evidence_json)),
    decided_at_ms       INTEGER NOT NULL CHECK(decided_at_ms >= 0)
) STRICT;

CREATE INDEX IF NOT EXISTS idx_raw_replay_decisions_logical
ON raw_replay_decisions(logical_keys_json);
```

Python interface (module `polylogue/storage/raw_authority.py`, replacing the
census writers):

```python
def record_raw_replay_decision(conn, decision: RawReplayDecision) -> bool:
    """Insert one addressable decision. Returns False when it already existed.

    ``INSERT ... ON CONFLICT(plan_input_digest) DO NOTHING`` followed by a
    read-back equality check, exactly as ``record_raw_authority_census``
    already does for ``raw_authority_plans`` (raw_authority.py:1178-1199): a
    same-digest row with different content is a collision, not an update.
    """

def raw_replay_decisions_for(conn, digests: Sequence[str]) -> dict[str, RawReplayDecision]:
    """Bounded lookup for one page of candidate components."""
```

`decision` is closed and each member has positive authority to exist:

- `terminal_no_progress` — the hjpx rule (`raw_convergence.py:4608-4645`). No
  other durable home; without it a component that cannot progress is
  reselected forever.
- `rejected_stale` — the plan's preconditions moved under it. Retained because
  it is paired with a blocker that gates the whole pass.
- `terminal_classified` — the component is settled by evidence that lives
  elsewhere (`raw_membership_census.status='failed'`, a current-fingerprint
  `ambiguous` verdict, a non-transient `parse_error`). This member is
  **optional and denormalizing**: the classifier can re-derive it from the
  retained evidence in §2 on every pass. It exists only so a large archive can
  skip the join; it is never read as authority when the underlying evidence
  disagrees. Decided: implement it, but only as a cache read behind the same
  evidence check, mirroring `raw_authority_verdicts`.

### 3.2 Disposable interface (ops tier, no migration chain)

```sql
-- ops.db. Deleting this file loses scheduling position and telemetry only.
CREATE TABLE IF NOT EXISTS raw_replay_attempts (
    plan_input_digest  TEXT PRIMARY KEY,
    attempts           INTEGER NOT NULL CHECK(attempts >= 0),
    last_attempt_ms    INTEGER NOT NULL,
    last_status        TEXT NOT NULL,
    last_reason        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_replay_pass_receipts (
    pass_id            TEXT PRIMARY KEY,
    started_at_ms      INTEGER NOT NULL,
    completed_at_ms    INTEGER,
    parser_fingerprint TEXT NOT NULL,
    cursor             TEXT,
    discovered         INTEGER NOT NULL DEFAULT 0,
    inspected          INTEGER NOT NULL DEFAULT 0,
    selected           INTEGER NOT NULL DEFAULT 0,
    executed           INTEGER NOT NULL DEFAULT 0,
    pending            INTEGER NOT NULL DEFAULT 0,
    failed             INTEGER NOT NULL DEFAULT 0,
    quiescent          INTEGER NOT NULL DEFAULT 0
);
```

Both are capped (`raw_replay_attempts` by digest — one row per live component,
deleted with the component; `raw_replay_pass_receipts` by a fixed newest-N
window). Neither is ever read as authority. The "unchanged pass writes
nothing" law is asserted against `source.db` only, and is stated that way in
the fixture (§7).

### 3.3 Bounded traversal

`build_raw_replay_plans` (`raw_authority.py:748-764`) returns the whole
component set today. It becomes a page contract with a resumable position,
the same shape the derivation kernel declares:

```python
def iter_replay_components(
    conn: sqlite3.Connection, *, cursor: str | None, limit: int
) -> ComponentPage:
    """One page of authority components, ordered by logical source key.

    ``next_cursor`` is the last key of the page; the next call resumes with
    ``WHERE logical_source_key > ?  ORDER BY logical_source_key LIMIT ?``.
    No sort or count over the whole relation, and no page larger than
    ``limit``.
    """
```

A pass then: pull a page → build each candidate's plan digest → look up
`raw_replay_decisions` for the page → inspect the output relation for the
rest → compute outside the writer lease → publish one component per
transaction. The bounded-pass budget, the resume cursor, and the
prerequisite gating are the kernel's (polylogue-hg6zd); this document only
fixes the adapter's side of the contract.

### 3.4 Explicitly rejected

- **A generic durable correctness ledger** ("every decision any subsystem ever
  makes, in one table"). Rejected: it recreates the second source of truth the
  tier-durability rule exists to prevent, and nothing in §2 needs a decision
  that is not already addressable by the evidence it concerns.
- **Keeping bounded census history behind an adapter.** Rejected: it preserves
  every cost in §1 — the per-pass write, the retention sweep, and the nulled
  predecessor chain — and moves only the call sites.
- **Persisting a `dry_run` mode.** Rejected, with positive authority: all three
  production call sites of `converge_materialization` pass `dry_run=False`
  (`daemon/cli.py:1330`, `daemon/cli.py:1432`,
  `daemon/convergence_stages.py:1076`), and no CLI or devtools command reaches
  it at all. Under this design a preview *is* an inspection pass, and an
  inspection pass writes nothing.
- **Storing `fixed_point`.** Rejected: derived. "Nothing pending at the
  current parser fingerprint" is a query over the same inspection the pass
  already runs. Note the dated observation that `fixed_point=0` held on all
  256 censuses ever run (`raw_authority.py:977-979`) — the stored flag has
  never once been true.

## 4. Disposition, field by field

Legend: **retain** (keep as-is) · **migrate** (durable meaning moves to the new
shape) · **reconstruct** (recomputable from retained durable evidence; do not
persist) · **retire** (delete, with the authority named).

### `raw_authority_censuses` (source, `archive_tiers/source.py:386-425`) — RETIRE

| Column | Disposition | Authority / replacement |
|---|---|---|
| `census_id`, `sequence_no`, `predecessor_census_id` | retire | Identity of a pass, not of a decision. The chain is already cut by header compaction (`raw_authority.py:1043-1061`) and `NULL` is deliberately ambiguous between "genesis" and "truncated", so it cannot be walked as history today. |
| `scope_json`, `residual_json`, `post_residual_json` | retire | ~144 KB/header of quarantined-raw id arrays (`raw_authority.py:70-73`). Replacement: the same ids are derivable by inspection; the operator-facing count moves to the ops pass receipt. |
| `inventory_digest`, `residual_digest`, `post_inventory_digest`, `post_residual_digest` | retire | Exist to compare one pass against the next. With addressable decisions there is no cross-pass comparison to make. |
| `parser_fingerprint` | migrate | Per-decision field on `raw_replay_decisions`, where it gates the same superseded-fingerprint rule (`raw_authority.py:42-44`). |
| `mode` | retire | Closed set `census|dry_run|apply`; `dry_run` has no production caller, and `census` vs `apply` is now the presence or absence of work. |
| `lifecycle_status`, `completed_at_ms`, `postflight_at_ms` | retire | The `planned` state exists only because planning and applying are separated by a pass boundary; §5 removes the boundary. |
| `quiescent`, `fixed_point` | reconstruct | Derived per inspection (§3.4). |
| `plan_count`, `executable_plan_count`, `residual_plan_count`, `post_plan_count` | retire | Counters over a membership set that is no longer written. Equivalent counters live in the ops pass receipt. |
| `created_at_ms` | retire | Pass telemetry; ops receipt. |

Product readers that must migrate: `read_raw_authority_census`
(`raw_authority.py:459-608`) behind CLI `raw-authority-census`
(`polylogue/cli/commands/maintenance/_raw_identity.py:40-78`) and MCP resource
`polylogue://raw-authority-census/{census_id}/{offset}`
(`polylogue/mcp/server_resources.py:373-392`);
`_raw_authority_detail_document` (`raw_authority.py:303-356`) behind
`raw-authority-detail` and its MCP resource (`server_resources.py:394-410`);
`raw_materialization_readiness_snapshot` (`storage/archive_readiness.py:475-576`)
feeding `polylogue status`, `polylogue paths`, and `daemon/status.py:257-259`.
See §6.

### `raw_authority_plans` (source, `source.py:427-436`) — MIGRATE / RETIRE

| Column | Disposition | Authority / replacement |
|---|---|---|
| `plan_id` | retire | `f"raw-replay:{input_digest}"` (`raw_authority.py:738`) — a prefix on the digest. |
| `input_digest` | migrate | Becomes `raw_replay_decisions.plan_input_digest`, the decision's address. |
| `input_raw_ids_json`, `logical_keys_json` | migrate | Carried on the decision row so a decision remains readable when its component is gone. |
| `authority_witness_json`, `source_preconditions_json`, `index_preconditions_json` | retire | Positive authority: their sole consumer is `validate_raw_replay_plan` (`raw_authority.py:1289-1304`), which re-derives the plan from current evidence and compares — a check that exists only to detect drift across the plan→apply pass boundary. §5 applies within one transaction against the live frontier CAS (`RawCASFrontierError`, six guarded `UPDATE`s in `raw_convergence.py`), which is a stronger check and is not reconstructible-dependent. `index_preconditions_json` additionally snapshots the **rebuildable** index tier, so it is not durable evidence in the first place. |
| `created_at_ms` | retire | Telemetry. |

### `raw_authority_census_plans` (source, `source.py:438-461`) — RETIRE membership, MIGRATE three read-backs

| Column | Disposition | Authority / replacement |
|---|---|---|
| `census_id`, `ordinal`, `selected` | retire | Per-pass membership and selection order — the rows that are 99.98% `carried_forward` (`raw_authority.py:47-49`). |
| `outcome_status` = `terminal` (no-progress) | migrate | `raw_replay_decisions.decision='terminal_no_progress'`. |
| `outcome_status` = `rejected_stale` | migrate | `raw_replay_decisions.decision='rejected_stale'` + its retained blocker. |
| `outcome_status` = `deferred` | reconstruct | `raw_replay_plan_deferred_for_envelope` (`raw_authority.py:799-838`) compares the component's payload size against the current envelope; both sides are live facts. |
| `outcome_status` = `executed` | reconstruct | The durable proof of execution is `raw_sessions.parsed_at_ms` plus the index-tier application receipt — which `_raw_replay_plan_outcome` already re-derives (`raw_convergence.py:4736-4747`). |
| `outcome_status` = `retryable`, `carried_forward` | retire | "Nothing happened yet." Re-derived by inspection. |
| `reason`, `next_action` | migrate (terminal/stale only) | `raw_replay_decisions.reason`; for non-terminal rows these are fixed literals assigned at insert (`raw_authority.py:1251-1256`). |
| `application_receipt_json` | reconstruct | `raw_replay_application_receipt` (`raw_authority.py:1307-1394`) recomputes it from `raw_sessions`/`raw_session_memberships`/index receipts. |
| `outcome_recorded`, `recorded_at_ms` | retire | Conservation bookkeeping for the planned→finalized lifecycle (`finalize_raw_authority_census`, `raw_authority.py:1689-1699`) that §5 removes; the attempt timestamp moves to the disposable `raw_replay_attempts`. |
| `idx_raw_authority_census_plans_status`, `idx_raw_authority_census_plans_attempts` | retire | Deleted with the table. |

### `raw_authority_census_post_plans` (source, `source.py:463-469`) — RETIRE

Whole table. Positive authority: its only reads are inside
`finalize_raw_authority_census`'s postflight subset invariant
(`raw_authority.py:1692-1768`), which exists to prove that the plan set the
pass published still holds after the apply. With one component published per
transaction there is no published set to conserve.

### `raw_authority_blockers` (source, `source.py:471-486`) — RETAIN, re-keyed

Retained in full: it is the durable, fail-closed **authorization** record, and
`unresolved_raw_replay_blockers` (`raw_authority.py:900-925`) refuses an entire
pass while any is open (`raw_convergence.py:5307-5316`). One migration change:
`census_id TEXT NOT NULL REFERENCES raw_authority_censuses(census_id)` must
become a non-FK `observed_pass_id TEXT` plus the new
`plan_input_digest TEXT NOT NULL`, because its current parent table is
retired. `plan_id` becomes the digest. The partial unique index on open
blockers per plan (`source.py:484-486`) is retained, re-expressed on the
digest. Readers — `describe_raw_authority_blocker` (`:1929`),
`list_unresolved_raw_authority_blockers` (`:1976`),
`resolve_raw_authority_blocker` (`:2071`),
`auto_resolve_stale_plan_blockers` (`:2180`), CLI `raw-authority-blockers`,
and `archive_readiness.py:594` — keep their shape.

### Retained without change

`raw_membership_census`, `raw_authority_parser_census`,
`raw_authority_verdicts`, `raw_legacy_append_resynthesis_receipts`,
`raw_sessions` and its byte-chain columns, and the index-tier
`raw_revision_applications` / `raw_revision_heads`. These carry evidence, not
pass bookkeeping.

Two notes recorded rather than acted on, because retiring either needs
authority this task does not have:

- `raw_legacy_append_resynthesis_receipts` has **one writer and no production
  reader** (`revision_governance.py:1037` writes it; the only read is its own
  accessor at `:1143-1157`, called from one test). It is nevertheless a
  byte-proof against a live filesystem — `source_prefix_sha256`, `source_size`,
  `source_mtime_ns`, `source_ctime_ns` cannot be re-derived once the source
  file moves. **Do not retire.** Wire a reader or record it as deliberately
  write-only; that is not this task.
- `raw_authority_parser_census.censused_at_ms` is written as the SQL literal
  `0` by both production writers (`revision_governance.py:2345-2352`,
  `sources/revision_backfill.py:1189-1196`) and read by nothing. Reported, not
  changed here.

## 5. Transactions

### 5.1 Unchanged pass

```
open source.db read-only
  iter_replay_components(cursor, limit)      -- one page, no lock
  raw_replay_decisions_for(page digests)     -- one indexed lookup
  inspect output relation for the remainder  -- raw_sessions + index receipts
  every candidate settled -> no write transaction is ever opened
close
                            durable writes: 0    ops writes: 1 capped receipt
```

The law: **a second pass over unchanged inputs opens no write transaction
against `source.db`.** Today the same pass writes a header and, over a
non-empty backlog, N+N membership rows.

### 5.2 Publishing one component

```
outside the writer lease:
  build plan -> digest; compute the replacement (parse, classify)

BEGIN IMMEDIATE                                  (source.db, write lease held)
  UPDATE raw_sessions SET <frontier> WHERE <exact prior values>
      rowcount != 1  -> RawCASFrontierError -> ROLLBACK, stays pending
  record_revision_application_sync(...)          (index tier receipt)
  INSERT OR IGNORE INTO raw_replay_decisions ... (only for a durable-terminal
                                                  or rejected-stale outcome)
COMMIT
```

- **Crash before COMMIT.** Nothing durable. The next pass re-inspects, finds
  the component pending, and re-derives the identical plan digest from the
  identical bytes — so it makes the identical decision. There is no `planned`
  census to recover, which deletes
  `recover_interrupted_raw_authority_censuses` (`raw_authority.py:1834-1899`)
  and the entire interrupted-lifecycle branch.
- **Crash after COMMIT.** The decision is durable and its key is the content it
  was made about. Re-recording it is `DO NOTHING`. Re-inspection sees the
  output valid and does nothing.
- **The gap today.** `record_raw_authority_census` and
  `finalize_raw_authority_census` are two separate transactions
  (`raw_authority.py:1095` and `:1680`), with per-plan outcome writes between
  them, each on its own connection — which is why the `planned` lifecycle and
  its recovery function exist at all.

### 5.3 Legitimate empty output

A component whose correct materialization is **zero sessions** — a non-session
artifact — is recorded by `raw_membership_census.status='non_session'`
(`source.py:348`). Inspection must classify that as settled, not as "never
parsed". This is the raw-side twin of the derivation kernel's empty-output law:
a valid empty result is not missing work, and the failure mode it prevents is
an artifact that is re-parsed on every pass forever.

## 6. Consumer migration map

| Consumer | file:line | Change |
|---|---|---|
| CLI `raw-authority-census` | `cli/commands/maintenance/_raw_identity.py:40-78` | Reads the ops pass receipt + decisions instead of a census header; `census_id` argument becomes a pass id. Operator-visible shape change. |
| CLI `raw-authority-detail` | `_raw_identity.py:83-120` | Census detail document retired; blocker and plan detail keep their routes against the digest. |
| CLI `raw-authority-blockers`, `raw-authority-frontier` | `_raw_identity.py:12-38` | Unchanged apart from the blocker's re-keyed parent. |
| MCP resources `polylogue://raw-authority-census/...`, `.../raw-authority-detail/...` | `mcp/server_resources.py:373-410` | Same shape change as their CLI twins; the query-handle helpers `raw_authority_census_query_handle` / `raw_authority_detail_query_handle` (`raw_authority.py:227-250`) follow. |
| `raw_materialization_readiness_snapshot` | `storage/archive_readiness.py:475-576` | `authority_pending_census_count` becomes a pending-component count from inspection; the frontier `state_counts` read out of `post_residual_json` move to the pass receipt. Field names on `DaemonStatus` (`daemon/status.py:257-259`) are kept where the meaning survives. |
| Daemon materialization pass | `storage/raw_convergence.py:5199-6194` (`_converge_raw_materialization`), entry `:5140` | Census record/finalize calls replaced by per-component publication; the pass keeps its bounded budget and gains the resume cursor. |
| Frontier reconciler | `storage/raw_reconciler.py:1327, 1771, 1844, 1947` | Same replacement; frontier previews stop writing a census. |
| `_delete_orphaned_raw_authority_plans` | `storage/raw_retention.py:2055-2106` | Deleted with `raw_authority_plans`; its three-way anti-join has no subject. |
| `prune_raw_authority_census_history` | `raw_authority.py:930-1070` | Deleted; there is no per-pass history to compact. |
| `reset_raw_authority_census_ledger`, `prune_orphaned_index_revision_seeds` | `raw_authority.py:2382`, `:2425` | Already unreferenced at this HEAD; deleted with the tables they count. |

## 7. Fixture sketches

Synthetic and deterministic (`tests/infra/` builders, `frozen_clock`), no live
archive. Each names the mutation that makes it red.

**F1 — an unchanged pass writes nothing durable.**
`tests/unit/storage/test_raw_decision_authority.py`

```python
def test_a_second_pass_over_unchanged_inputs_opens_no_durable_write(archive):
    converge_raw_materialization(archive.root)            # first pass settles
    before = source_db_change_counter(archive.root)       # PRAGMA data_version
    report = converge_raw_materialization(archive.root)
    assert source_db_change_counter(archive.root) == before
    assert report.executed == 0 and report.quiescent
```
Anti-vacuity: re-introduce any per-pass row (a header, a carried-forward
membership row) and the change counter moves.

**F2 — decisions survive index + ops deletion.** (gen6d AC2)

```python
def test_admitted_rejected_and_ambiguous_decisions_survive_index_and_ops_loss(tmp_path):
    archive = synthetic_archive(cohorts=[
        admitted_cohort("logical-a"),        # one proven full revision
        superseded_cohort("logical-b"),      # a later revision displaces an earlier
        ambiguous_cohort("logical-c"),       # two unorderable revisions
    ])
    converge_raw_materialization(archive.root)
    before = snapshot_decisions(archive.root)      # raw_replay_decisions + frontier

    (archive.root / "index.db").unlink()
    (archive.root / "ops.db").unlink()
    converge_raw_materialization(archive.root)     # reconverge from source alone

    assert snapshot_decisions(archive.root) == before
    for raw_id, blob_hash in archive.expected_bytes.items():
        assert blob_store(archive.root).read(blob_hash) == archive.payload(raw_id)
    assert source_frontier(archive.root, "logical-b").accepted_raw_id == before.heads["logical-b"]
```
Anti-vacuity: put any of the three decisions in `index.db` or `ops.db` and the
post-deletion snapshot differs; drop the blob-ref liveness join and the byte
assertion fails.

**F3 — crash before and after publication.**

```python
@pytest.mark.parametrize("crash_at", ["before_commit", "after_commit"])
def test_a_crash_around_publication_converges_to_one_decision(archive, crash_at):
    with crashing_writer(archive, at=crash_at):
        with pytest.raises(SimulatedCrash):
            converge_raw_materialization(archive.root)
    recovered = converge_raw_materialization(archive.root)
    assert decision_rows(archive.root) == [expected_decision]
    assert recovered.executed == (1 if crash_at == "before_commit" else 0)
```
Anti-vacuity: split the frontier CAS and the decision insert into two
transactions and `before_commit` leaves a decision with no frontier move.

**F4 — legitimate empty output settles once.**

```python
def test_a_non_session_artifact_is_settled_not_reparsed(archive):
    archive.admit(non_session_artifact())
    first = converge_raw_materialization(archive.root)
    second = converge_raw_materialization(archive.root)
    assert first.executed == 1 and second.executed == 0
    assert membership_census(archive.root).status == "non_session"
```
Anti-vacuity: treat "zero materialized sessions" as "not yet parsed" and the
second pass re-executes.

**F5 — bounded traversal.**

```python
def test_one_pass_over_ten_thousand_components_costs_its_page(archive_factory):
    archive = archive_factory(components=10_000)
    report = converge_raw_materialization(archive.root, budget=Budget(page=8, publication=1))
    assert report.work.discovered == 8 and report.executed == 1
```
Anti-vacuity: restore the whole-set `build_raw_replay_plans` call and
`discovered` becomes 10,000.

## 8. Migration order

Durable-tier changes are additive and numbered, one `PRAGMA user_version` step
at a time, behind a verified backup. The destructive step is **separate,
last, and gated on explicit operator consent** — this is the copy-forward rule
in `docs/internals.md`, not a preference.

1. **M1 — additive DDL (source).** `CREATE TABLE raw_replay_decisions` + its
   index; `ALTER TABLE raw_authority_blockers ADD COLUMN plan_input_digest`
   and `ADD COLUMN observed_pass_id` (both nullable at this step). No reader
   changes. Migration-safety class: `additive-no-backup` for the new table;
   the blocker columns are additive.
2. **M2 — bounded backfill (same migration chain, next number).** Populate
   `raw_replay_decisions` from the surviving plan window:
   `raw_authority_census_plans.outcome_status IN ('terminal','rejected_stale')`
   joined to `raw_authority_plans.input_digest`; populate the blockers' new
   columns from their `plan_id`. Bounded by the existing 8-census retention
   window — anything older is already deleted today (§1), which is stated in
   the migration comment so the bound is not mistaken for data loss.
3. **M3 — readers (code).** New lookup path behind the decisions table; ops
   tables created by DDL (disposable tier, no chain). Both ledgers are live;
   the census tables are still written. Reversible by reverting code.
4. **M4 — writers (code).** The pass publishes per component (§5.2) and stops
   calling `record_raw_authority_census` / `finalize_raw_authority_census` /
   `record_raw_replay_outcome` / `prune_raw_authority_census_history`. The
   four tables remain present and inert. **Surface migration lands here** (§6):
   CLI and MCP census routes switch. A full archive reconvergence is not
   required — no derived schema identity moves, because none of these tables
   is in a derived tier. (`devtools schema closure <file>` must be run on each
   touched file at implementation time and the result reported; membership is
   a property of the import graph, not of the directory.)
5. **M5 — destructive (separate migration, explicit consent, verified backup).**
   `DROP TABLE raw_authority_census_post_plans`, `raw_authority_census_plans`
   (with `idx_raw_authority_census_plans_status`,
   `idx_raw_authority_census_plans_attempts`), `raw_authority_censuses`,
   `raw_authority_plans` — in that order, children first, with
   `PRAGMA foreign_keys = ON` so the declared FKs verify the order.
   Preconditions: M4 shipped; a current measurement recorded (§10); zero rows
   in `raw_authority_census_plans` with an unmigrated terminal outcome.

Exact code deletion targets at M4/M5, all verified present at this HEAD in
`polylogue/storage/raw_authority.py`: `raw_authority_census_query_handle:227`,
`raw_authority_detail_query_handle:236`, `_raw_authority_census_ref:251`,
`_raw_authority_detail_ref:277`, `_raw_authority_detail_document:303`,
`read_raw_authority_detail:413`, `read_raw_authority_census:459`,
`build_raw_replay_plans:748` (replaced by `iter_replay_components`),
`raw_replay_plan_last_attempts:765`,
`raw_replay_plan_deferred_for_envelope:799` (moves to a live size check),
`raw_replay_plan_no_progress_plan_ids:839` (reads the decisions table),
`prune_raw_authority_census_history:930`, `record_raw_authority_census:1073`,
`validate_raw_replay_plan:1289`, `record_raw_replay_outcome:1572`,
`_raw_replay_plan_from_row:1603`, `_raw_authority_census_receipt:1615`,
`latest_raw_authority_census_receipt:1648`,
`finalize_raw_authority_census:1669`,
`recover_interrupted_raw_authority_censuses:1834`,
`reject_stale_raw_replay_plan:2248` (becomes a decision write),
`reject_invalid_raw_replay_application:2314`,
`reset_raw_authority_census_ledger:2382`,
`prune_orphaned_index_revision_seeds:2425`; constants
`RAW_AUTHORITY_CENSUS_PLAN_RETENTION:60`,
`RAW_AUTHORITY_CENSUS_HEADER_RETENTION:74`,
`RAW_AUTHORITY_CENSUS_QUERY_PREFIX:45`,
`RAW_AUTHORITY_DETAIL_QUERY_PREFIX:46`,
`RAW_AUTHORITY_DETAIL_CHUNK_CHARS:47`; and in
`polylogue/storage/raw_retention.py`,
`_delete_orphaned_raw_authority_plans:2055` with
`_PURELY_ORPHANED_AUTHORITY_PLANS_SQL`.

## 9. Decisions closed by this document

Every choice the implementation would otherwise have to make is fixed here:
decision address (plan input digest), decision vocabulary (three members,
§3.1), tier for iteration (ops, disposable), traversal contract
(`iter_replay_components`, keyset paged), publication granularity (one
component per transaction), stale detection (the live frontier CAS, not a
stored precondition snapshot), blocker re-keying, backfill bound (the existing
8-census window), migration order (M1–M5 with the destructive step gated), and
surface migration (§6). The `terminal_classified` member is deliberately
included as a cache-behind-evidence, not as authority.

## 10. Measurement that is still owed

No current measurement was taken: the task forbids live archive access and
this worktree has none. Before M5 runs, one measurement must be recorded and
**labelled as current, next to the dated figures in §1**, not merged with
them:

```
SELECT 'censuses', COUNT(*) FROM raw_authority_censuses
UNION ALL SELECT 'plans', COUNT(*) FROM raw_authority_plans
UNION ALL SELECT 'census_plans', COUNT(*) FROM raw_authority_census_plans
UNION ALL SELECT 'post_plans', COUNT(*) FROM raw_authority_census_post_plans
UNION ALL SELECT 'open_blockers', COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL;
```

plus the per-table page counts from `dbstat`. The one helper that already
computes the first half, `reset_raw_authority_census_ledger`
(`raw_authority.py:2382-2422`), is read-only when `dry_run=True` and has no
caller; it is a deletion target, so the measurement belongs in a devtools
archive subcommand or a scratch script, run once, and quoted with its date.
