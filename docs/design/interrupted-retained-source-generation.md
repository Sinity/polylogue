# Interrupted retained source generations

Status: **policy superseded** (polylogue-fzbzk, 2026-09-22). Originally decided under polylogue-xt5ga. This is a design record only. It authorizes no live-archive mutation, source or audit migration, replay, or blob deletion.

> **Read the correction first.** The evidence in this record stands: the
> reproduction, the enumeration of durable references a stopped ingest leaves
> behind, and the liveness-owner analysis are all still accurate in kind. Its
> *chosen policy* is not. Policy A -- durable abandonment plus a release ledger
> -- is superseded, and the bounded successor it specifies must not be built
> from this page. See [Correction](#correction-2026-09-22-policy-a-is-superseded)
> at the end.

## Question

An accepted retained-input ingest can stop after durable acceptance and after some source publication, but before `IngestExecution.finalize`. How should that generation remain truthful, and how may its retained bytes eventually stop protecting the blob store?

## Reproduction and evidence

The route is explicit in `polylogue/operations/daemon_ingest.py:693-769`: acceptance creates the source generation and audit authority; enumeration and materialization publish source evidence; a stop calls `mark_unknown`, then `fence`; a resumed request returns the audit-derived indeterminate state and never replays from current source or index state (`:721-732`). `mark_unknown` records an unknown operation attempt (`:671-679`), while `fence` only records `machine_requests.stop_reason` and revokes unstarted authorizations (`polylogue/operations/audit.py:1729-1754`). Nothing in that path disposes of the accepted source generation.

I reproduced the post-interruption rows in memory with the current `SOURCE_DDL` and `AUDIT_DDL`. The synthetic input had one accepted item and one admitted raw record, then stopped at `deadline`. The read-only output was:

```text
source_generations: [{source_generation_id: 'g-interrupted', sealed_at_ms: None}]
source_items: [{source_generation_id: 'g-interrupted', source_item_id: 'item-1', disposition: 'admitted', outcome_code: 'success', raw_id: 'raw-1', blob_hash: '1111111111111111111111111111111111111111111111111111111111111111'}]
source_item_raw_members: [{source_generation_id: 'g-interrupted', source_item_id: 'item-1', record_coordinate: 'record:0', raw_id: 'raw-1', raw_blob_hash: '1111111111111111111111111111111111111111111111111111111111111111'}]
audit machine_requests: [{request_id: 'req-1', artifact_ref: 'g-interrupted', stop_reason: 'deadline', stopped_at_ms: 102}]
audit operation_runs: [{operation_id: 'op-1', status: 'interrupted', unknown_count: 1, unknown_reason: 'deadline'}]
liveness 1111111111111111111111111111111111111111111111111111111111111111: live ('source.db.raw_sessions', 'source.db.source_items', 'source.db.blob_refs')
generation projection: ['1111111111111111111111111111111111111111111111111111111111111111']
```

The exact durable evidence left without a terminal generation disposition is therefore:

* `source_generations(g-interrupted)`, still unsealed. Its current schema has `sealed_at_ms` but no abandoned or released state (`polylogue/storage/sqlite/archive_tiers/source.py:54-61`).
* `source_items(item-1)`, including the retained input `blob_hash`, item disposition, and the enumeration witnesses. The item schema permits `pending` and `interrupted` outcomes, but has no terminal abandonment state (`polylogue/storage/sqlite/archive_tiers/source.py:63-101`).
* `source_item_raw_members(record:0)`, whose `raw_blob_hash` is a second durable content reference (`polylogue/storage/sqlite/archive_tiers/source.py:181-191`), plus the joined `raw_sessions(raw-1)` row and its `blob_refs(raw_payload, raw-1)` receipt.
* `machine_requests(req-1)` and `machine_request_parts`, fenced with `stop_reason=deadline`, `operation_previews(preview-1)`, `operation_authorizations(auth-1)` revoked by the fence, and `operation_runs(op-1)` finalized as unknown. Those audit rows correctly prove an indeterminate attempt; they do not own source-byte release.

The liveness owners observed by the canonical descriptor are `source.db.source_items`, `source.db.raw_sessions`, and `source.db.blob_refs` (`polylogue/storage/blob_liveness.py:53-67`). `project_live_blob_hashes(..., source_generation_id=...)` scopes raw rows through source membership and scopes source items directly (`polylogue/storage/blob_liveness.py:334-378`). Thus the generation is not an SQL orphan. It is a retained, nonterminal owner with no bounded disposition, and every retry or restart continues to see those bytes as live.

## Decision: explicit abandonment, then release through the existing GC owner

Choose policy A: record **durable abandonment** for the exact accepted generation, preserve its provenance, and release its generation-owned liveness only through an exact, restartable release ledger. The release is an accounting transition. Physical unlink remains the existing two-phase blob-GC operation, which commits exact member intent before unlink and resumes pending members after a crash (`polylogue/storage/blob_gc.py:539-582`, `:750-763`).

Truthfulness rules:

1. Abandonment never deletes `source_generations`, `source_items`, `source_item_raw_members`, `raw_sessions`, `blob_refs`, or the original audit rows. The source row continues to prove that a payload was accepted and records its hash. The abandonment receipt says that the payload is intentionally no longer replayable once release closes. Reads that need the bytes must report the missing/released evidence explicitly.
2. The release plan is computed from the pinned source and index snapshots and stores the exact `(generation, owner kind, owner ref, blob hash)` set. A raw or attachment hash is released only when the final liveness check proves that no other generation or tier still owns it. Shared `raw_sessions` and `blob_refs` therefore remain live when another owner needs them.
3. A retry with the same generation, manifest digest, and release-plan digest returns the existing terminal receipt. A different manifest or release set for the same generation is a conflict, never a second replay or a new request. Restart resumes only `pending` release members; it does not rediscover a new set from the filesystem.
4. The original ingest operation remains `unknown` and the machine request remains fenced. The successor operation is a separate, explicitly authorized adjudication of that known indeterminate effect. It must not turn `unknown` into `completed` or infer a successful ingest from current index rows.

## Bounded successor

The successor is one daemon-owned mutation, `mutate-abandon-retained-source-generation`, version 1, with effect identity `source-generation-abandon:v1:<source_generation_id>:<manifest_digest>:<release_plan_digest>` and an idempotency key over the same tuple.

The implementation owns these seams:

* `polylogue/storage/sqlite/migrations/source/048_abandon_retained_source_generation.sql` adds `source_generation_dispositions` (one row per generation, immutable manifest digest, `abandoned_at_ms`, reason, original `request_id`, adjudication `operation_id`, release-plan digest, and state) and `source_generation_release_members` (the exact owner/ref/hash set with `pending`, `released`, `retained`, or `blocked` outcome). Fresh `SOURCE_DDL` and its schema identity must include the same objects. The migration is additive and numbered; it must run only through the durable migration runner.
* `polylogue/storage/blob_liveness.py` extends the source-item and raw-membership predicates to consult the disposition and release-member tables. Absence of the new table on an older archive remains a blocker or the current live behavior, never an assumption that bytes are releasable. The descriptor must continue to include direct `source_items`, `raw_sessions`, and typed `blob_refs` owners.
* `polylogue/operations/specs.py` declares the operation and `polylogue/operations/mutation_actuators.py` provides preview, authorization, exact-plan inspection, and apply. The actuator writes through the daemon's `DaemonWriteCoordinator` and the existing source/audit continuity transaction. It does not open a second direct-SQL cleanup route.
* Audit uses the existing `operation_previews`, `operation_authorizations`, `operation_runs`, `operation_targets`, `operation_events`, and `machine_requests` tables. The source disposition and release-member rows are the domain receipt; the audit operation binds the actor, original request, exact plan, and terminal outcome.
* After the release ledger closes, the actuator hands hashes to the existing bounded GC generation route. It never unlinks a blob itself. `gc_generation_members` remains the physical deletion recovery authority and can record `skipped_still_live` when a concurrent owner appears.

Write authority is the daemon writer lease. The preview must refuse a sealed generation, a missing or conflicting original machine request, a completed ingest receipt, an already adjudicated generation, an unreadable source/index tier, or a stale archive identity. Before the first durable migration or release operation, require a verified backup receipt covering `source.db`, `audit.db`, and every candidate blob. Keep that backup and a runtime commit matching the source and audit schema versions as rollback artifacts. If the backup, continuity head, namespace marker, or schema identity does not match, refuse without changing durable state. This decision does not run that migration or operation.

## Rejected policy and prohibited shortcuts

Policy B, deliberate indefinite retention with only a visibility report, is rejected. It leaves `source_items` and the joined raw/blob-ref owners live forever, so normal GC cannot reclaim bytes and repeated interrupted generations accumulate without a terminal bound. An audit `unknown` row alone is not a retention policy.

The successor must not silently delete source rows, clear `blob_hash` values, replay the accepted input under a new ingest request, infer completion from rebuildable index state, issue direct SQL cleanup outside the daemon writer, or introduce an `ops.db` or campaign-only ledger. Every abandonment and every release member must be durable, idempotent, auditable, and restartable in the source/audit authority tiers.

## Focused verification plan

The decision was verified with read-only source inspection and the in-memory synthetic reproduction above. The successor's focused tests are named here for implementation:

* `tests/unit/operations/test_abandon_retained_source_generation.py`: accepted generation interrupted after partial enumeration produces one immutable abandonment receipt; a duplicate request is a no-op and a changed manifest conflicts.
* `tests/unit/storage/test_blob_liveness.py`: generation-scoped liveness includes the interrupted source item and joined raw/blob-ref owners, excludes only a released generation-owned member, and retains a hash shared by another generation or index attachment.
* `tests/unit/storage/test_blob_gc_durable_intent.py`: a release handoff creates exact pending members, survives interruption, and delegates physical deletion to the existing GC member-intent protocol.
* `tests/unit/operations/test_machine_receipts.py`: restart reads the original unknown ingest receipt plus the separate abandonment receipt and never returns a completed ingest or replays source bytes.

No live archive, full corpus, schema migration, or cleanup operation was run for this decision.

## Correction 2026-09-22: Policy A is superseded

The decision above rejected deliberate retention (policy B) on the grounds that
interrupted generations keep their bytes live forever and accumulate without a
terminal bound. For an archival application that is not a defect. Preserving
accepted original evidence is what the product is for, and unbounded growth of
*retained accepted input* is the cost of that purpose, not a leak in it.

The distinctions the challenge draws, and this correction accepts:

| The observed fact | What it does not establish |
| --- | --- |
| the client stopped waiting | that publication stopped |
| the attempt was interrupted | that the accepted input is disposable |
| interpretation is unfinished | that the source must be released before local work is retried |
| the operator requests deletion | anything about interruption recovery; that invokes a separate retention policy |

So the immediate policy does not need `source_generation_dispositions` and
`source_generation_release_members` merely to make stopped ingests collectible.
What it needs is an explicit statement that **accepted originals remain
retained**, and that the unfinished attempt stays visible as unfinished.

### The retention default

1. An interrupted accepted generation **stays retained**. Its
   `source_generations`, `source_items`, `source_item_raw_members`,
   `raw_sessions` and `blob_refs` rows are unchanged, its bytes stay live to the
   blob-liveness descriptor, and no operation exists to release them on the
   grounds of interruption.
2. The original attempt **stays indeterminate**. `mark_unknown` and `fence`
   already record that honestly, and truthfulness rule 4 above is retained
   verbatim: nothing may turn `unknown` into `completed` or infer a successful
   ingest from rebuildable index state.
3. The generation is **inspectable, not silently indefinite**. A retained
   unfinished generation should be readable through the existing operation and
   source surfaces rather than discoverable only by SQL. That is a read-side
   gap, not a new durable table.
4. A later deliberate **retention limit or purge is not waived**. It may well
   justify the abandonment/release machinery this record specified. It should be
   judged against its own use case with its own measured accumulation evidence,
   and not arrive inside the fresh start disguised as interruption recovery.

### What that retires

Superseded: the "Decision: explicit abandonment, then release through the
existing GC owner" section's choice of policy A, the whole "Bounded successor"
section, and the rejection of policy B in "Rejected policy and prohibited
shortcuts". The *prohibited shortcuts* in that same section are retained --
they describe what any future disposal route must not do, and nothing about
this correction licenses direct SQL cleanup, silent row deletion, cleared
`blob_hash` values, or an `ops.db` ledger.

Retained and still accurate: the question, the reproduction, the enumeration of
durable references, the liveness-owner analysis, and truthfulness rule 4. The
focused verification plan is superseded along with the successor it tests.

### Three head facts this record states wrongly

Re-resolved at `9ef655cb8`; the record and its bead both predate these.

* **`polylogue/storage/sqlite/migrations/source/048_abandon_retained_source_generation.sql` is the wrong slot number, twice over.** The directory is not empty -- it holds `002_excision_policy_projections.sql` and its frozen `002.train.json` -- so the numbered route *is* the live regime, and the next source slot is `003`, not `048`.
* **`SOURCE_SCHEMA_VERSION` no longer exists.** The durable tier modules deliberately declare no `*_SCHEMA_VERSION` of their own. The source tier's durable target is `SOURCE_TIER_VERSION`, currently `2`, above `ARCHIVE_FORMAT_FLOOR_VERSION = 1`. Any statement of the form "applied to a version-47 archive" is unevaluable at this head.
* **A note circulated on polylogue-fzbzk claiming `migrations/{source,user,audit}/` each contain only `__init__.py` is stale.** Only `audit/` does. `source/` and `user/` each carry one numbered train.

None of these change the correction: the contracted scope needs no migration at
all, which is also why the stale seam stopped mattering.
