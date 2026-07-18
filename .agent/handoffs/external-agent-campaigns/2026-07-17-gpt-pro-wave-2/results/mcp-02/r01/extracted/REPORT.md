# Authority design for the six-tool MCP cutover

`ARCH-04` denotes `.agent/handoffs/external-agent-campaigns/2026-07-16-gpt-pro-wave/testdiet/context/testsuite_diet/architecture/04-destructive-and-authentication-boundaries.md`.

## Status and answer

This is an analysis/adjudication report, not a patch. It projects the already-decided executable-operation contract into the six-tool endpoint `query`, `read`, `write`, `judge`, `run`, `operate`. The controlling architecture says every CLI, API, MCP, daemon, maintenance, and internal-repair route must submit an `OperationRequest` to one executable `OperationExecutor`, with storage guards retained only as defense in depth (`ARCH-04:12-40`; bead `polylogue-t46.9`).

The source does not implement that boundary yet. `OperationSpec` remains descriptive metadata, not an executable declaration: it lacks a stable semantic version, capability, resolver, handler, confirmation policy, durable idempotency/conflict policy, and receipt schema (`polylogue/operations/specs.py:12-62`). The declared `delete-session` spec even says permanent deletion/confirm/dry-run while setting `previewable=False` (`polylogue/operations/specs.py:613-631`). `ArchiveWriteGateway` is an ingest commit/effect gateway, not authority; the only production construction is ingest (`polylogue/archive/write_gateway.py:1-96`; `polylogue/pipeline/services/ingest_batch/_core.py:1216-1217`; bead `polylogue-a7xr.18`).

The current MCP role model also differs from the assignment shorthand. Source defines a monotonic `read < write < review < admin` ladder (`polylogue/mcp/declarations/models.py:11-18`), and registration gates judgment at `review` (`polylogue/mcp/server_tools.py:1424-1434`). The recommended adjudication is not to invent a second policy system: retain role names as deployable capability profiles, but make `assertion:judge` an explicit non-ordinal capability. `write` does not imply it; `review` and `admin` do. This preserves the canonical judgment lane in bead `polylogue-37t.12` while allowing the task's read/write/admin shorthand to remain true for the general archive ladder.

## Evidence authority and scope

The snapshot reports HEAD `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. The visible working-tree source compared equal to the bundled Git clone at that commit; the archive metadata reported a dirty snapshot and included an extra `browser-extension/package-lock.json`, so exact pre-archive dirty provenance was not reconstructed. No live daemon, browser, deployment, user archive, destructive operation, concurrency test, or integration test was run. Conclusions are static source/tracker adjudication.

Repository rules make the intended ownership clear: Polylogue is local and single-writer; substrate owns semantics and surfaces are leaf adapters (`CLAUDE.md:3-7,17-39`). `source.db` and `user.db` are durable, with user assertions irreplaceable; index/embeddings are rebuildable and ops state is disposable (`CLAUDE.md:89-110`). The daemon owns writes and its main process is the sole SQLite writer (`CLAUDE.md:140-160`). These facts determine destructive classes and receipt/recovery obligations below.

## Six-tool endpoint and capability profiles

The generated declaration artifact currently proposes seven default read transactions plus four privileged transactions named `write`, `judge`, `run`, and `maintenance` (`docs/generated/mcp-equivalence.json`, `target_algebra`). The assignment is stricter: six tools. Therefore the final projection is:

| Tool | Purpose | Discovery/visibility | Effective authorization |
| --- | --- | --- | --- |
| `query` | Execute bounded typed queries, aggregates, graph traversal, explanation/discovery plans. | All authenticated read profiles. | `archive:read`; folds current query/get/graph/explain/status query forms. |
| `read` | Resolve stable refs, bounded object/context/receipt projections. | All authenticated read profiles. | `archive:read`; folds current read/context/get/status projections. |
| `write` | Apply user-owned reversible assertions and submit lifecycle requests. | write, review, admin profiles. | Operation-specific write capability such as `user:write`, `annotation:write`, or `assertion:candidate`. |
| `judge` | Perform canonical candidate judgment transactions. | review and admin profiles only. | `assertion:judge`; never inferred from ordinary write or caller-supplied `actor_ref`. |
| `run` | Execute saved queries or governed recipes. | write, review, admin profiles; read-only saved query remains reachable through query/read. | `run:execute` plus the union of every expanded step capability. Saved content grants none. |
| `operate` | Inspect, preview, apply, resume, and reconcile maintenance/ingest/reset/excision operations. | admin/system profiles; redacted public status may project through read. | Operation-specific admin/system capability; destructive operations additionally require bound confirmation. |

Recommended profiles are explicit capability sets, not an independent RBAC model:

| Profile | Capabilities |
| --- | --- |
| read | `archive:read` |
| write | read + `user:write`, `annotation:write`, `assertion:candidate`, `excision:request`, `run:execute` |
| review | write + `assertion:judge` |
| admin | review + `operate:*`, including reset/excision/raw-authority/migration/backup capabilities |
| system actor | Explicit narrowly-scoped internal capabilities; no discoverable MCP role and no authority inherited from archive text, prompts, resources, or recipes. |

Transport/runtime supplies `principal_id`, capability set/fingerprint, receiver/archive identity, and credential state. Request `actor_ref` may be retained only as provenance. The architecture explicitly says read/write/admin/system capabilities are inputs to the executor and prompts/resources/saved recipes/archive text never acquire authority (`ARCH-04:68-79`).

## Operation invariant

Every mutating path follows exactly:

`resolve → authorize → preview → bound confirmation when required → apply → durable receipt → postflight/reconcile`

Resolution occurs once logically: preview and apply invoke the same resolver and canonicalization rules. Apply re-resolves under the mutation transaction/lease and compares the complete target-set and effect digests. Any change returns `preview_stale` before mutation (`ARCH-04:55-57`). Daemon write coordination, SQLite transactions, `ArchiveWriteGateway`, write effects, excision guards, blob leases, and tier-specific checks remain behind the executor; none substitutes for public authority (`ARCH-04:12-20`; bead `polylogue-t46.9`).

### Classification legend

| Code | Meaning |
| --- | --- |
| R0 | Nonmutating execution. |
| R1 | Additive or rebuildable/reversible effect; no durable evidence loss. |
| R2 | Durable user-state transition reversible by a declared inverse, before-image, tombstone, or supersession. History may remain immutable. |
| D1 | Destructive to rebuildable/derived state. Reconstructable, but target binding is mandatory. |
| D2 | Scoped irreversible deletion of durable evidence/bytes, including excision and blob/raw cleanup. |
| D3 | Broad or live authority repair/reset/migration that can irreversibly rewrite durable truth. |

### Preview profiles

| Code | Preview obligation |
| --- | --- |
| P0 | Validation/disclosure only; no apply effect. |
| P1 | Exact resolved refs and complete target digest, count, normalized request, affected tiers, before/after diff or expected rows, current generation, inverse/compensator or rebuild route. |
| PJ | Candidate/evidence projection, current status and generation, proposed decision/reason, injection authorization, replacement/supersession result. |
| PR | Immutable expansion of saved ref/version into a typed query or operation DAG, step capabilities, bounds, complete target/effect union, checkpoints/compensation. |
| PD | P1 plus destructive class, full target-set and effect digests, bounded disclosed samples, counts/bytes by tier, lineage/dependents, rebuild source, postflight obligations. |
| PB | PD plus every durable path/hash, backup/recovery status, writer/quiescence/lease state, archive/receiver identity, and explicit operator authorization scope. |

### Confirmation profiles

| Code | Requirement |
| --- | --- |
| C0 | No interactive confirmation. Capability, request intent, idempotency key, and durable receipt remain mandatory. |
| CAS | Explicit intent plus expected generation/current-state digest; no preview token. A changed generation is an explicit conflict. |
| CB | Opaque token bound to immutable preview hash, principal, capability fingerprint, archive identity, operation/spec version, request/target/effect digests, nonce, and expiry. Required for scoped D1/D2 destructive effects. |
| CB+ | CB plus explicit admin/operator authorization of the preview receipt digest. Required for D3 broad destruction or live authority repair. |

### Receipt and retry profiles

`RW` records operation/spec version, principal/capability fingerprint, archive identity, request/idempotency key and digest, target/effect digests, affected tiers, before/after generations, outcome, before-image/inverse or rebuild obligation, timestamps, and postflight state. `RJ` adds candidate generation, decision/reason, injection authorization, judgment/result refs, and per-item outcomes. `RR` adds saved ref/version, expanded plan digest, step receipts, cursor/checkpoint, and aggregate outcome. `RO`/`RD` add tier/path/hash counts, bytes, generation/checkpoints, backup/recovery facts, convergence debt, and postflight checks; `RD` is the destructive audit receipt.

`I1` means the same idempotency key plus identical canonical request returns the same receipt; the same key with a different request returns `idempotency_conflict`. Natural duplicate add/remove operations return explicit no-op receipts. `I2` adds expected-generation/CAS. `IJ` makes an exact judgment retry idempotent while any changed decision/reason/replacement/injection intent conflicts. `IR` resumes only from durable completed-step receipts. `ID` never reapplies a destructive effect after a recorded receipt; interrupted cross-tier work resumes/reconciles from a durable operation journal.

Common pre-apply failures are `request_invalid`, `operation_unknown`, `spec_version_unsupported`, `capability_denied`, `target_not_found`, `target_ambiguous`, `preview_required`, `confirmation_required`, `confirmation_invalid`, `confirmation_expired`, `principal_mismatch`, `capability_fingerprint_mismatch`, `archive_mismatch`, `spec_mismatch`, `preview_stale`, `expected_generation_mismatch`, `idempotency_conflict`, and `busy_retryable`. Apply/postflight failures are `policy_refused`, `apply_failed_rolled_back`, `partial_failure`, `indeterminate_effect`, `receipt_write_failed`, `postflight_failed`, and `postflight_pending`. A missing durable receipt after an uncertain D2/D3 effect is never reported as success.

## Decision-complete verb matrix

The matrix includes current semantic mutations plus the declared saved-query/recipe `run` lane and required operation lifecycle. Legacy names are compatibility aliases; the semantic operation ids below become the authority. “Deletion” of user assertions is classified R2 because the unified `user.db` model uses assertion status/supersession rather than evidence excision (`CLAUDE.md:99-107`); the executor must not remove the interim confirm until it can emit a restorable before-image and expose the inverse. `session.prune` is D1 because the current store deletes only rebuildable index rows and intentionally leaves user overlays (`polylogue/storage/sqlite/archive_tiers/archive.py:6328-6348`). `session.excise` is the distinct D2 operation that destroys evidence (`ARCH-04:59-66`).

| Family | Operation id | Object kinds | Capability/profile | Class | Preview | Confirm | Receipt | Retry | Specific failures | Primary evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| write | assertion.candidate.capture | assertion-candidate, evidence-ref, object-ref | `assertion:candidate` (write/review/admin/system) | R2 reversible durable assertion | P1 | C0 | RW: candidate ref, source/evidence refs, policy `{inject:false}`, generation | I1 | `evidence_invalid`; `candidate_policy_violation` | `polylogue/api/archive.py:2441-2465`; bead `polylogue-37t.15` |
| write | annotation.batch.import | annotation-schema, annotation-batch, assertion | `annotation:write` (write/review/admin) | R2 reversible by compensating/tombstoning imported assertions | P1 mandatory for overwrite or bulk | C0 | RW + schema version, batch provenance, accepted/rejected counts, assertion refs | I1 by batch/idempotency key | `schema_conflict`; `batch_partial_validation` | `polylogue/api/archive.py:1872-1901`; `CLAUDE.md:108-110` |
| write | blackboard.note.post | blackboard-note, object-ref | `user:write` (write/review/admin) | R2 reversible durable assertion | P1 for overwrite; otherwise request echo | C0 | RW + note ref, targets, author provenance, context policy | I1 | `target_invalid`; `context_policy_violation` | `polylogue/api/archive.py:6172-6237`; `CLAUDE.md:99-107` |
| write | tag.add | tag, session | `user:write` (write/review/admin) | R2 reversible assertion | P1 | C0 | RW + normalized tag, assertion ref, before/after active generation | I1; duplicate add is no-op receipt | `tag_invalid`; `session_ambiguous` | `polylogue/api/archive.py:5177-5209` |
| write | tag.remove | tag, session | `user:write` (write/review/admin) | R2 reversible tombstone | P1 | C0 after before-image/restore exists; interim jn40 boolean only | RW + removed assertion ref and restorable prior row | I1; absent tag is no-op receipt | `tag_not_present` (non-error no-op); `restore_unavailable` during migration | `polylogue/api/archive.py:5210-5230`; bead `polylogue-jn40` |
| write | tag.bulk_add | tag, session-set | `user:write` (write/review/admin) | R2 reversible assertions | P1 mandatory: complete target digest, count, normalized tags | C0 | RW + per-session outcomes and complete target digest | I1; stable per-item outcomes | `target_too_broad`; `partial_validation` | `polylogue/api/archive.py:5322-5365` |
| write | mark.add | mark, object-ref | `user:write` (write/review/admin) | R2 reversible assertion | P1 | C0 | RW + mark kind, target ref, assertion ref | I1; duplicate add is no-op receipt | `mark_invalid`; `object_not_found` | `polylogue/api/archive.py:5480-5507` |
| write | mark.remove | mark, object-ref | `user:write` (write/review/admin) | R2 reversible tombstone | P1 | C0 after before-image/restore exists; interim jn40 boolean only | RW + removed assertion and restorable prior row | I1; absent mark is no-op receipt | `mark_not_present` (non-error no-op); `restore_unavailable` during migration | `polylogue/api/archive.py:5508-5565`; bead `polylogue-jn40` |
| write | annotation.save | annotation, object-ref, annotation-schema | `annotation:write` (write/review/admin) | R2 reversible/CAS durable assertion | P1: prior body/schema/generation and proposed diff | CAS for overwrite; C0 for create | RW + annotation ref, schema version, previous/current generation, before-image | I2 | `schema_conflict`; `expected_generation_mismatch` | `polylogue/api/archive.py:5566-5643` |
| write | annotation.delete | annotation | `annotation:write` (write/review/admin) | R2 reversible tombstone, not excision | P1: exact annotation, target, schema, current generation | C0 once restore receipt exists; interim jn40 gate | RW + prior assertion row and restore handle | I2; already deleted is exact no-op | `annotation_not_found`; `expected_generation_mismatch` | `polylogue/api/archive.py:5644-5654`; bead `polylogue-jn40` |
| write | saved_view.save | saved-query | `user:write` (write/review/admin) | R2 reversible/CAS durable assertion | P1: normalized query, prior version, authority-free executable content | CAS for overwrite; C0 create | RW + view ref/version/query digest/before-image | I2 | `query_invalid`; `expected_generation_mismatch` | `polylogue/api/archive.py:5655-5692`; architecture `ARCH-04:76-79` |
| write | saved_view.delete | saved-query | `user:write` (write/review/admin) | R2 reversible tombstone | P1: view ref/name/query digest/current generation | C0 after restore exists; interim jn40 gate | RW + prior row/restore handle | I2 | `view_not_found`; `expected_generation_mismatch` | `polylogue/api/archive.py:5693-5710`; bead `polylogue-jn40` |
| write | recall_pack.save | recall-pack, object-ref-set | `user:write` (write/review/admin) | R2 reversible/CAS durable assertion | P1: bounded payload summary, ref set digest, prior version | CAS for overwrite; C0 create | RW + pack ref/version/payload digest/before-image | I2 | `payload_invalid`; `expected_generation_mismatch` | `polylogue/api/archive.py:5938-5975` |
| write | recall_pack.delete | recall-pack | `user:write` (write/review/admin) | R2 reversible tombstone | P1: pack label, payload/ref digest, generation | C0 after restore exists; interim jn40 gate | RW + prior row/restore handle | I2 | `pack_not_found`; `expected_generation_mismatch` | `polylogue/api/archive.py:5976-6001`; bead `polylogue-jn40` |
| write | workspace.save | reader-workspace, saved-query/ref-set | `user:write` (write/review/admin) | R2 reversible/CAS durable assertion | P1: layout/query/ref changes, prior version | CAS for overwrite; C0 create | RW + workspace ref/version/content digest/before-image | I2 | `workspace_invalid`; `expected_generation_mismatch` | `polylogue/api/archive.py:6002-6071` |
| write | workspace.delete | reader-workspace | `user:write` (write/review/admin) | R2 reversible tombstone | P1: workspace content/ref digest and generation | C0 after restore exists; interim jn40 gate | RW + prior row/restore handle | I2 | `workspace_not_found`; `expected_generation_mismatch` | `polylogue/api/archive.py:6072-6088`; bead `polylogue-jn40` |
| write | metadata.set | session-metadata, session | `user:write` (write/review/admin) | R2 reversible/CAS assertion | P1: key, typed old/new value, generation | CAS for overwrite; C0 create | RW + key, old/new value, assertion ref, before-image | I2 | `metadata_key_invalid`; `expected_generation_mismatch` | `polylogue/api/archive.py:5259-5290` |
| write | metadata.delete | session-metadata, session | `user:write` (write/review/admin) | R2 reversible tombstone | P1: key, old value, generation | C0 after restore exists; interim jn40 gate | RW + prior value/assertion/restore handle | I2 | `metadata_not_found`; `expected_generation_mismatch` | `polylogue/api/archive.py:5291-5321`; bead `polylogue-jn40` |
| write | correction.record | correction, session/object-ref | `user:write` (write/review/admin) | R2 reversible/CAS durable assertion | P1: prior active correction, new body/value, scope, generation | CAS for replacement; C0 create | RW + correction ref, supersedes chain, context policy, before-image | I2 | `correction_conflict`; `scope_invalid` | `polylogue/api/archive.py:6089-6148`; `CLAUDE.md:99-107` |
| write | correction.delete | correction, session | `user:write` (write/review/admin) | R2 reversible tombstone | P1: exact correction kind/body/generation | C0 after restore exists | RW + prior assertion/restore handle | I2 | `correction_not_found`; `expected_generation_mismatch` | `polylogue/api/archive.py:6149-6160` |
| write | correction.clear | correction-set, session | `user:write` (write/review/admin) | R2 reversible bulk tombstone | P1 mandatory: full assertion-set digest, count, kinds, generations | C0 only when receipt can restore every row; otherwise CB during migration | RW + per-row before-images and outcomes | I2; exact retry returns same receipt | `target_set_changed`; `restore_unavailable` | `polylogue/api/archive.py:6161-6171`; bead `polylogue-jn40` |
| write | session.suppress | suppression-assertion, session | `user:write` (write/review/admin) | R2 reversible visibility policy; evidence retained | P1 mandatory: resolved session, visibility impact, durable suppression row, no evidence deletion | C0 with explicit intent and before-image | RW + suppression assertion, prior visibility, inverse operation | I2 | `session_ambiguous`; `suppression_policy_conflict` | `polylogue/cli/commands/reset.py:206-219,381-448`; architecture `ARCH-04:59-66` |
| write | lifecycle.excision_request.submit | excision-request, session | `excision:request` (write/review/admin) | R2 reversible request creation; no bytes destroyed | P1: resolved requested subject, reason, lineage-cascade choice, policy queue | C0 | RW + request ref/status/actor provenance/subject digest | I1 | `subject_ambiguous`; `request_duplicate_conflict` | `polylogue/cli/commands/excise.py:164-183`; `polylogue/security/lifecycle.py` |
| judge | assertion.candidate.decide | assertion-candidate, judgment, resulting assertion | `assertion:judge` (review/admin; explicit system only by policy) | R2 durable lifecycle transition; semantically irreversible history, result supersedable | PJ mandatory: candidate/evidence, current status/generation, decision, injection/replacement intent | CAS; explicit request intent; no confirm token | RJ: candidate generation, decision/reason, inject authorization, judgment/result refs, per-item outcome | IJ: exact retry succeeds; changed retry/conflicting judgment fails | `candidate_not_pending`; `judgment_conflict`; `injection_not_authorized`; `replacement_invalid` | `polylogue/storage/sqlite/archive_tiers/user_write.py:1624-1887`; beads `polylogue-37t.12`, `polylogue-41ow`, `polylogue-303r.5` |
| run | saved_query.execute | saved-query, query-plan, result-set | `run:execute` plus `archive:read` (write/review/admin; read may use query directly) | R0 nonmutating | PR: ref/version, expanded typed plan, bounds/result semantics | C0 | RR: saved ref/version, plan digest, result-set/continuation, resource accounting | IR; same ref/version/args reuses deterministic result receipt when available | `saved_ref_not_found`; `saved_ref_stale`; `query_budget_exceeded` | `docs/generated/mcp-equivalence.json` target `run`; bead `polylogue-t46.8.3` |
| run | recipe.execute | recipe, operation-plan DAG, step receipts | `run:execute` AND union of every underlying operation capability | Class is maximum class of expanded steps; saved content grants no authority | PR mandatory: immutable expanded DAG, step inputs, capabilities, target/effect union, rollback/checkpoint plan | Inherited: C0/CAS for non-destructive; one CB for D1/D2 or CB+ for D3 bound to the whole expanded effect | RR + per-step receipts, checkpoint/cursor, aggregate target/effect digest | IR; resume only from durable completed-step receipts; changed recipe/version is new request | `recipe_cycle`; `step_capability_denied`; `recipe_changed`; `partial_failure`; `compensation_failed` | architecture `ARCH-04:76-79`; bead `polylogue-t46.8.3` |
| operate | operation.list/status/preview | operation-spec, operation-receipt, maintenance-plan | `operate:inspect` (admin/system; public projections may be read-redacted) | R0 nonmutating | P0; preview itself uses the operation resolver and disclosure policy | C0 | No apply receipt; preview is immutable PreviewReceipt when execution may follow | Read/idempotent | `operation_unknown`; `receipt_not_found`; `disclosure_denied` | `polylogue/mcp/server_maintenance_tools.py:57-109,165-244` |
| operate | archive.ingest_or_reprocess | raw-session, source artifact, parsed session, derived projections | `operate:ingest` (admin/system) | R1 additive/update; source durable, derived rebuildable | P1 for explicit reprocess scope; ordinary watched ingest may use standing bounded authorization | C0 | RO: source/raw digest, parser/version, session refs, tier effects, convergence obligations | I1 by content hash/source identity; changed content creates new effect identity | `provider_ambiguous`; `source_changed`; `parse_failed`; `convergence_pending` | `CLAUDE.md:112-120,140-160`; `polylogue/api/ingest.py:29-65` |
| operate | maintenance.repair | session-insight, message-type backfill, derived rows | `operate:maintenance` (admin/system) | R1 non-destructive/rebuildable repair | P1 mandatory for manual operation; target/count/current debt/expected rows | C0 | RO: target digest, repaired/skipped/failed counts, generation/postflight debt | I1/ID; retry only remaining debt | `target_unsupported`; `repair_incomplete`; `convergence_pending` | `polylogue/maintenance/targets.py:158-188`; `polylogue/maintenance/planner.py:425-651` |
| operate | maintenance.cleanup.derived | orphaned message/index rows, empty rebuildable sessions | `operate:maintenance` (admin/system) | D1 destructive but rebuildable | PD mandatory: complete target digest/count, affected derived tiers, rebuild path | CB | RD: deleted identities/counts, target/effect digest, rebuild/postflight obligations | ID; exact retry returns receipt and never widens target | `preview_stale`; `cleanup_scope_widened`; `postflight_failed` | `polylogue/maintenance/targets.py:189-211`; architecture `ARCH-04:48` |
| operate | maintenance.cleanup.durable_or_blob | orphaned attachment/blob, orphan blob-ref debt, superseded raw snapshot | `operate:maintenance` + durable-delete capability (admin; system may apply only an operator-bound request) | D2 scoped irreversible evidence/bytes deletion | PB mandatory: every durable/blob identity digest, bytes, leases/references, recovery/backup status; archive-wide or authority-rewriting scope is D3 | CB | RD + deleted blob/raw hashes, byte counts, lease/reference proof, audit receipt | ID with durable checkpoints; no re-delete | `reference_race`; `lease_active`; `backup_required`; `indeterminate_effect` | `polylogue/storage/raw_retention.py:1146-1248`; `polylogue/storage/blob_integrity.py:1296-1370`; `polylogue/sources/live/batch.py:2256-2284`; `polylogue/maintenance/targets.py:212-234`; `CLAUDE.md:158-160` |
| operate | index.update | session, index rows, FTS/insight debt | `operate:derived` (admin/system) | R1 rebuildable incremental convergence | P1: session set, current generations, expected surfaces | C0 | RO + updated/skipped refs, generation and remaining debt | I1 | `session_not_found`; `source_not_materialized`; `convergence_pending` | `polylogue/api/archive.py:4985-5010` |
| operate | derived.rebuild_generation | index/FTS/insight/embedding generation | `operate:derived` (admin/system) | R1 while building side-by-side; old generation retained | PD: source scope, expected row counts/bytes, generation id, resource budget | C0 for side-by-side build | RO + new generation, source/plan digest, validation results; no promotion yet | ID by generation/idempotency key | `generation_exists`; `source_changed`; `validation_failed`; `resource_exhausted` | `polylogue/cli/commands/maintenance/_rebuild_index.py:365-486` |
| operate | derived.promote_generation | generation pointer, index tier | `operate:derived` (admin/system) | R1 pointer switch; previous generation retained for rollback | PD: candidate/active generations, validation summary, expected active generation | CAS; no interactive token if rollback generation retained | RO + old/new generation, CAS value, rollback handle | I2 | `active_generation_changed`; `candidate_invalid`; `rollback_unavailable` | `polylogue/cli/commands/maintenance/_rebuild_index.py:435-486` |
| operate | derived.reset_or_prune | index/FTS/insight/embedding rows or tier | `operate:reset` (admin/system) | D1 destructive rebuildable | PD mandatory: exact target/tier digest, row/byte estimates, rebuild source and obligations | CB | RD + removed identities/counts, rebuild recipe, convergence debt | ID; target digest fixed | `preview_stale`; `rebuild_source_missing`; `partial_tier_delete` | `polylogue/storage/sqlite/archive_tiers/archive.py:6328-6348`; architecture `ARCH-04:48` |
| operate | embedding.backfill_or_reconcile | embedding vectors/status/run ledger | `operate:derived` (admin/system) | R1/D1 rebuildable | P1: model/dimension, session set, orphan counts, expected writes/deletes | C0 for additive backfill; CB for explicit destructive reset | RO + model/version, vectors written/deleted, run ledger, remaining backlog | ID with cursor/run id | `model_mismatch`; `dimension_mismatch`; `backlog_changed`; `provider_failed` | `polylogue/storage/embeddings/reconcile.py:170-418`; `polylogue/daemon/embedding_backlog.py:118-359`; `polylogue/cli/commands/embed.py:607-788`; `polylogue/cli/commands/maintenance/_embeddings.py:58-110` |
| operate | raw.materialize_or_replay | raw-session, parsed/index rows, replay state | `operate:raw` (admin/system) | R1 source-preserving repair; derived outputs rebuildable | P1: raw ids, parser/schema versions, downstream replacements/debt | C0 | RO + raw/session refs, parser version, rows/effects, replay cursor | ID by raw content hash and replay state | `raw_not_found`; `parser_version_changed`; `replay_state_conflict` | `polylogue/maintenance/replay.py:131-256`; `polylogue/daemon/cli.py:610-651` |
| operate | raw.authority.repair | source.db raw authority, index projections, filesystem artifacts | `operate:raw-authority` (admin/system; explicit operator for live) | D3 live authority repair; can rewrite durable truth | PB mandatory: frontier, selected authority, all overwritten/deleted hashes/paths, backup and quiescence proof | CB+ | RD + before/after authority frontier, backups, paths/hashes, checkpoints and postflight | ID with durable journal; resume/reconcile only | `authority_ambiguous`; `archive_live`; `backup_missing`; `partial_failure`; `indeterminate_effect` | `polylogue/cli/commands/maintenance/_raw_identity.py:35-55`; `polylogue/daemon/cli.py:690-752` |
| operate | session.prune | session, index/derived projections | `operate:reset` (admin/system) | D1 destructive rebuildable; user overlays preserved | PD mandatory: resolved ids, all derived projections, explicit statement that durable user/source evidence remains | CB | RD + deleted derived refs/counts and rebuild obligations | ID | `session_ambiguous`; `preview_stale`; `source_missing_for_rebuild` | `polylogue/storage/sqlite/archive_tiers/archive.py:6328-6348` |
| operate | session.excise | session, raw bytes, source/index/user/embedding/ops rows, blobs, lineage | `operate:excision` (admin plus explicit operator; system only with separately authorized request) | D2 scoped irreversible evidence destruction | PB mandatory: full cross-tier/blob/lineage target digest, bytes, cascade, non-resurrection guards, postflight; broad archive-wide scope is D3 | CB | RD: durable audit receipt, all deleted hashes/rows/bytes, checkpoints, non-resurrection/postflight proof | ID; exact retry returns prior receipt; partial state reconciled from journal | `lineage_dependents`; `preview_stale`; `guard_missing`; `partial_failure`; `non_resurrection_failed` | `polylogue/security/excision.py:1-60,577-642`; beads `polylogue-layg`, `polylogue-t46.9` |
| operate | archive.reset.derived | index/embeddings/ops derived tiers | `operate:reset` (admin/system) | D1 broad destructive rebuildable | PB mandatory: tier/path/bytes/table counts, source rebuild readiness, daemon quiescence | CB | RD + removed tier artifacts, backup if any, rebuild plan/debt | ID with checkpoints | `archive_live`; `rebuild_source_missing`; `partial_failure` | `polylogue/cli/commands/reset.py:454-542`; architecture `ARCH-04:48` |
| operate | archive.reset.durable | source.db, user.db, blobs, full archive root | `operate:reset-durable` (admin + explicit operator) | D3 broad irreversible destruction | PB mandatory: every durable path/tier/blob, byte counts, backups, recovery impossibility, receiver/archive identity | CB+ | RD + exact deleted paths/hashes/bytes, authorization, checkpoints, residual inventory | ID; never infer success without receipt reconciliation | `archive_live`; `backup_missing`; `path_changed`; `partial_failure`; `indeterminate_effect` | `polylogue/cli/commands/reset.py:454-542`; `CLAUDE.md:89-110` |
| operate | blob.gc | blob, blob-ref, gc-generation, lease | `operate:blob-gc` (admin; system may apply only an operator-bound request) | D2 irreversible blob deletion | PB: unreferenced hash set, bytes, snapshot/reference check, active leases, generation; archive-wide or authority-rewriting scope is D3 | CB | RD + hash/byte counts, skipped reasons, lease/reference proofs, gc generation | ID by GC generation and hash set | `lease_active`; `reference_race`; `snapshot_changed`; `unlink_failed` | `CLAUDE.md:158-160`; `polylogue/storage/blob_gc.py:338-509`; `polylogue/cli/commands/maintenance/_blob_gc.py:42-80` |
| operate | tier.migrate | SQLite tier, schema version, backup manifest | `operate:migrate` (admin/system) | D3 when copy-forward/drop/rewrite touches durable tier; otherwise R1 schema migration | PB for destructive migration: source/target versions, backup manifest, object counts, disk budget, rollback path | CB+ for destructive durable migration; CAS/C0 for additive migration with verified rollback | RD + versions, backup manifest digest, migrations applied, checks/postflight | ID by tier/from/to version; resume from migration ledger only | `version_unsupported`; `backup_missing`; `disk_space`; `integrity_check_failed` | `polylogue/cli/commands/maintenance/_migrate_tier.py:45-60`; architecture `ARCH-04:49` |
| operate | archive.backup | archive tier set, backup artifact/manifest | `operate:backup` (admin/system) | R1 additive external artifact | P1: included tiers, destination, free space, consistency/quiescence mode | C0 unless overwriting an existing backup (then CAS) | RO + manifest, tier hashes/sizes, consistency point, destination | I1 by manifest/destination policy | `destination_exists`; `archive_busy`; `integrity_check_failed`; `short_write` | `polylogue/cli/commands/backup.py:13-67`; `polylogue/daemon/backup.py:513-640` |
| operate | database.vacuum | SQLite tier/file | `operate:maintenance` (admin/system) | R1 logical-state-preserving physical rewrite | P1: tier/path, live-writer state, free disk estimate, pre-integrity status, rollback/backup condition | C0 under quiescent non-destructive policy; any repair variant that can drop/rewrite durable truth is a separate D3 operation with CB+ | RO + before/after sizes, integrity result, elapsed/resource data | ID; retry only after checking prior file integrity | `archive_live`; `disk_space`; `integrity_check_failed` | `polylogue/cli/shared/check_support.py:73-80` |
| operate | startup_or_convergence.repair | FTS, lineage, insights, embeddings, watched-query baselines/result sets, candidate findings, convergence debt | `system:converge` with bounded declared scope plus each underlying durable-write capability | R1/D1 rebuildable repair plus R2 durable candidate emission | P1 generated internally: debt ids, bounded target digest, budget, durable candidate/result-set effects, and expected repairs; operator preview for manual broad run | C0 for R1/R2 additive effects; CB for every D1 effect. A declared standing system policy may mint/consume that CB only within its fixed bounds | RO/system receipt + debt cleared/created, durable candidate/result-set refs, target/effect digest, remaining work | ID with debt/cursor/query/candidate identifiers | `standing_scope_exceeded`; `writer_busy`; `candidate_conflict`; `repair_incomplete`; `convergence_pending` | `polylogue/daemon/convergence_standing_queries.py:77-105,143-228,231-300`; `polylogue/daemon/convergence_stages.py:117-190,361-526,1201-1448,1750-1778`; architecture `ARCH-04:76-79` |

## Uniform bound-preview confirmation protocol

### Preview creation

For any operation requiring CB or CB+, the executor resolves and authorizes first, then persists an immutable `PreviewReceipt`. A system-bound CB for a predeclared D1 policy follows the same record and validation path; standing authorization does not become C0. Canonicalization is exactly RFC 8785 JSON Canonicalization Scheme (JCS) over UTF-8 JSON. Reject duplicate object names and non-finite numbers; normalize instants to RFC 3339 UTC strings before JCS; represent semantic sets as arrays sorted by resolver-defined canonical object reference. The canonical content is hashed as:

```text
preview_hash = "sha256:" + sha256(canonical_json(preview_payload_without_preview_hash_or_token))
```

The preview contains at least:

```json
{
  "protocol": "polylogue.operation-confirm/v1",
  "operation": {"id": "session.excise", "spec_version": "1.0.0"},
  "principal": {"id": "local:operator", "capability_fingerprint": "sha256:..."},
  "archive": {"archive_id": "...", "fileset_id": "...", "receiver_id": "..."},
  "request_digest": "sha256:...",
  "target": {"set_digest": "sha256:...", "count": 1, "disclosed": ["session:..."]},
  "effect": {"digest": "sha256:...", "class": "D2", "tiers": ["source", "index", "user", "embeddings", "ops", "blobs"]},
  "generations": {"user": 41, "index": "gen-..."},
  "postflight": ["non_resurrection", "no_dangling_blob_refs", "derived_absence"],
  "issued_at": "...",
  "expires_at": "...",
  "nonce": "...",
  "preview_hash": "sha256:..."
}
```

The complete target set participates in `target.set_digest` even when disclosure is bounded or redacted. `effect.digest` covers the exact ordered/canonical effect plan, destructive class, tier/path/hash identities, and postflight obligations. Preview records live in a short-lived server-side confirmation store (an ops-tier ledger is appropriate); restart may invalidate unused previews. Store only a hash of the opaque 256-bit token. The token itself is returned once to the authorized principal.

### Apply validation

Apply accepts both `preview_hash` and the opaque token. The executor verifies protocol, token hash, single-use state, expiry, principal, capability fingerprint, archive/fileset/receiver identity, operation id/spec version, request digest, target-set digest, effect digest, and required confirmation strength. It then re-resolves targets under the write lease/transaction and recomputes target/effect digests. Any mismatch is `preview_stale`; no mutation starts. CB+ also verifies a separately recorded explicit operator authorization of that preview digest. The apply schema is unchanged: a CB+ token is issued only after that authorization, and the server-side token record carries the authorization principal, method, time, and strength.

After successful apply the token is consumed. The operation receipt stores the preview hash and confirmation strength, not the secret token. A network retry with the same idempotency key and request digest returns the existing operation receipt even though the token is consumed. A new idempotency key cannot reuse a consumed token.

### MCP tool-schema shape

Authority-bearing identity/capabilities are runtime context, never caller fields. `actor_ref`, when preserved, belongs under provenance and does not authorize.

```json
{
  "name": "write",
  "input": {
    "operation": "tag.add",
    "target": {"refs": ["session:..."]},
    "input": {"tags": ["decision"]},
    "mode": "preview",
    "idempotency_key": "uuid-or-stable-client-key",
    "expected_generation": null,
    "confirmation": null,
    "provenance": {"actor_ref": "user:local"}
  }
}
```

```json
{
  "name": "operate",
  "input": {
    "operation": "session.excise",
    "scope": {"refs": ["session:..."]},
    "input": {"reason": "operator request", "cascade_lineage": false},
    "mode": "apply",
    "idempotency_key": "...",
    "expected_generation": null,
    "confirmation": {"preview_hash": "sha256:...", "token": "opcf_..."}
  }
}
```

`judge` uses a typed list of `{candidate_ref, expected_generation, decision, reason, inject, replacement_*}` items and never takes a destructive confirmation token. `run` takes `{ref, expected_version, args, mode, idempotency_key, confirmation}`; the executor expands the immutable ref/version before authorization. `operate` additionally supports nonmutating lifecycle modes `list`, `status`, `preview`, and mutating `apply`/`reconcile`, but an operation spec constrains which modes are valid.

Tool schemas should use a discriminated operation registry so invalid per-operation fields fail before preview. A single untyped arbitrary dictionary would recreate surface-local policy. `OperationSpec` owns the stable operation id/version, input schema, capability, class, resolver/disclosure, preview/confirmation policy, handler, tiers, idempotency/conflict policy, receipt projection, and recovery/postflight obligations (`ARCH-04:22-40`).

### Migration from `polylogue-jn40`

1. Land the ten interim fail-closed booleans exactly as bead `polylogue-jn40` requests: `delete_annotation`, `delete_saved_view`, `delete_recall_pack`, `delete_workspace`, `delete_metadata`, `remove_tag`, `remove_mark`, `maintenance_execute`, `rebuild_index`, and `rebuild_session_insights`. This is a temporary asymmetry patch, not proof of authority.
2. Add executable specs, PreviewReceipt, confirmation/idempotency ledger, OperationExecutor, typed failures, and receipts. Do not let `confirm=true` reach handlers as authority.
3. During compatibility, `confirm=false` on a destructive legacy MCP call returns a preview plus `confirmation_required`. Remote MCP `confirm=true` does not silently mint and consume a token. A local interactive CLI adapter may translate `--yes` only after it has rendered/obtained the exact bound preview and submits its token.
4. Dual-emit old and new response fields only long enough for route-parity tests and telemetry. Record use of boolean adapters.
5. Remove booleans when every surface invokes the executor. Remove confirmation entirely from reversible tag/mark and soft user-state tombstones only after exact before-image/restore receipts and inverse routes exist. Use CB for scoped D1/D2 destructive effects and CB+ for D3 broad/live effects.

The architecture is explicit: a boolean is compatibility only and never executor proof (`ARCH-04:44-57`).

## Adjudicated contradictions and transition rules

| Observed conflict | Adjudication |
| --- | --- |
| Task shorthand says `read ⊂ write ⊂ admin`; source has `review` between write and admin. | Treat roles as capability profiles. Preserve `review` for deployment/discovery compatibility, but judgment authorization is the distinct `assertion:judge` capability; ordinary write cannot judge. |
| Generated target algebra has 7 read transactions and calls privileged admin tool `maintenance`; assignment requires six tools. | Fold get/graph/explain/context/status forms into `query` or `read`; rename the privileged lifecycle tool to `operate`. The generated artifact is evidence of migration intent, not final authority. |
| `delete_session_safe` sounds permanent; low-level store deletes only index rows and preserves user overlays. | Retire ambiguous `delete_session`. Compatibility maps to explicit `session.prune`; suppression is a separate write operation. Only `session.excise` means irreversible evidence destruction. |
| jn40 treats reversible remove/delete verbs like destructive maintenance. | Keep booleans as immediate fail-closed mitigation. Final class follows data semantics: assertion tombstones are R2 and need receipts/inverses, not permanent interactive confirmation. |
| Daemon write coordinator and ArchiveWriteGateway look like central chokepoints. | Keep both behind OperationExecutor. Coordinator serializes and gateway commits/effects; neither owns principal authorization, target disclosure, preview binding, or public receipts. |
| Raw-retention code calls superseded snapshot deletion “compaction” and checks source/active-revision authority. | Classify the current effect D2: it deletes durable raw rows and blob bytes. Source existence and reference checks remain defense-in-depth, not proof of byte-exact reconstructability or authority. Reclassify to R1/D1 only after an executable spec proves exact reconstruction and records that proof. |
| Current judgment code supports exact retry but has a SELECT/upsert race. | Fix bead `polylogue-41ow` with one immediate user-tier transaction before routing the lane; executor CAS is additive, not a substitute for storage atomicity. |

## Required invariants and acceptance proof

Implementation is complete only when:

- Every declared mutation has exactly one executable `OperationSpec`; every surface and system actor reaches its handler only through `OperationExecutor`.
- A production-route bypass test fails when any adapter calls a storage/API actuator directly, and static inventory rejects unclassified mutation entry points. This must be a real-route test, not a declaration allowlist (`ARCH-04:128-138`; bead `polylogue-t46.9`).
- Delete/prune/excise/reset invoked through CLI/API/MCP/daemon/internal maintenance produces the same operation/spec id, canonical request, target digest, authorization result, effect digest, and receipt projection.
- Changing principal, capability set, archive/fileset/receiver identity, operation version, expiry, request, target set, or effect plan after preview fails before mutation.
- Reversible writes execute without needless interactive confirmation and emit restorable receipts; judgment exact retries are idempotent and changed concurrent judgments conflict.
- D2/D3 operations have durable checkpoints and reconciliation; no uncertain partial result is labeled success.
- Prompts, resources, recipes, `actor_ref`, and archive text cannot grant capability or satisfy confirmation.

## Limitations and value of another iteration

This pass is decision-complete for the authority model and broad static census. It did not execute mutation routes, force the `polylogue-41ow` two-connection race, inspect a live archive's exact user-tier tombstone rows, or prove every maintenance target's byte-level reversibility. The highest-value next iteration is implementation-coupled: generate the executable operation registry, make the static bypass detector consume it, and run real-route parity/failure-injection tests on the first session suppress/prune/excise vertical slice. A purely narrative second pass has low value unless used as an adversarial review against code changed after this snapshot.
