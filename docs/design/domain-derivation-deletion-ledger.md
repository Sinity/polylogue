# Domain derivation adoption ledger

The production daemon uses the ordered-domain kernel selected by `polylogue-04r9f`. SQL and output identity remain with each domain. This ledger measures the change from `a0187cfc1fed2486d85fee20f9c99bfc041443d6` and distinguishes adopted producers from surviving predecessors.

## Ownership changes

| Responsibility | Removed owner | Current owner | Remaining work |
| --- | --- | --- | --- |
| FTS session membership | FTS stages, startup repair, `fts_identity_convergence`, `dangling_repair`, `drift_sampling`, `freshness` | `storage/fts/derivation.py`, composed by `operations/fts_derivation.py` and the shared daemon owner | Canonical triggers, `messages_fts_identity` and the FTS refresh guard remain intentionally. |
| FTS orphan membership | `fts_orphan_audit` and independent repair loops | One low-cadence orphan partition in the FTS adapter | Global residue discovery is bounded, but inspecting its full binding and replacing residue still costs a global scan. |
| Session counters | Duplicate append/full-replace arithmetic in the parsed writer | `storage/derived/session/summary.py`, shared by canonical writes, inspection and derivation publication | No generic completion table is needed: the session row represents valid zero counters. |
| Session profiles | Generic profile stage | Existing session-profile adapter, preceded by the summary adapter | Excess retirement consumes no deleted summary row. Selected profile scope and archive sweep share the daemon owner. |
| Embeddings | Daemon embedding stage and backlog execution loop | `storage/embeddings/derivation.py` and `daemon/embedding_owner.py` | Manual CLI backfill and older session materialization still execute their own routes. Their full retirement is incomplete. |
| Raw-to-logical membership | Generic raw parse stage | Existing raw-observation adapter and daemon raw owner | Source admission, bounded parse preparation, and raw membership evidence retain their independent owners. |
| Readiness | FTS freshness receipts, stage-derived insight status, optimistic embedding attempt counters | Domain inspection in FTS, session summary/profile and embedding status projections | Other products still have stage-specific telemetry. No operation receipt certifies these migrated outputs. |
| Generic scheduling | Generic DAG sorting and redundant migrated callbacks | Ordered registry, bounded required/excess discovery, disposable cursor, shared compute adapter and writer bridge | `ConvergenceStage`, `FileState`, `SessionState`, callback variants and barriers remain for the products below. |
| Remaining product execution | None | Sinex publication, raw-authority cache warming, attachment acquisition, Claude workflow, delegation evidence and standing queries still use `convergence_stages.py` | Move each surviving responsibility to its product owner before removing the generic stage engine. Renaming or relocating callbacks would not establish shared derivation ownership. |
| Disposable stage debt | Migrated domains no longer consult it as correctness authority | `sources/live/convergence_debt*`, daemon retry/status and surviving stage callers | The table and retry surface remain; complete stage/debt deletion is not delivered. |
| Verification | Removed FTS ledger/repair-shape tests, obsolete migrated-stage assertions | Common kernel laws, production adapter laws, restart, property and fault tests | Whole-domain differential and managed workload acceptance remains separate from selected green tests. |

## Measurement

The table below counts added and deleted physical lines with `git diff --numstat` against the named base. It includes replacements within files, so gross deletion is not a claim that every deleted line represented a distinct retired capability. Domain SQL is counted separately. There are no file renames; semantic moves are explicitly accounted for in the ownership table rather than assigned an unverifiable line count.

| Scope | Added | Deleted | Net |
| --- | ---: | ---: | ---: |
| Docs and devtools | 80 | 479 | -399 |
| Other production | 416 | 234 | +182 |
| Daemon | 659 | 2749 | -2090 |
| Domain storage | 1779 | 1903 | -124 |
| Tests | 1816 | 6073 | -4257 |
| Total, excluding this ledger | 4750 | 11438 | -6688 |

The reduction is real across daemon, FTS storage, readiness consumers and tests. It is also incomplete: the surviving generic stage engine prevents a claim that the repository has only one convergence execution model. No credit is taken for its eventual deletion.

## Schema and generation contract

This is an additive-derived change for embedding reference identity and a removal of rebuildable FTS freshness structures. No new generic durable object or source/user/audit migration is introduced. Session summary and FTS derivation source participate in the derived schema identity closure. Their changes require daemon reconvergence after landing; landing during a rebuild invalidates that build. The embedding adapter itself is outside that closure, while its schema and write dependencies retain their own identity rules.

Embedding references bind exact message semantics. Metadata binds the complete recipe and output contract. A computed replacement replaces the vector and its metadata atomically, even when a recipe change preserves the provider request address. Missing physical vectors cannot be certified by surviving metadata. Inspection and publication refuse retired frames.

## Module depth and limits

The kernel's interface carries bounded discovery, authoritative classification, explicit prerequisites, lease-free computation and revalidated publication. FTS, embeddings, raw membership and session partitions share these behaviors while retaining domain SQL. Removing the kernel would duplicate those rules across production owners. The retained per-pass state has no durable readiness authority.

Independent architectural inspection reaffirmed the ordered-domain choice. It identified concrete corrections for inspection-budget progress, empty-session census, physical embedding replacement, frame certification and orphan failure isolation. Those corrections stay with their existing owners. Kernel line count alone does not justify replacing the design with separate runners or a generic artifact framework.

The kernel still has substantial contract/type material and bounded scheduling logic. Complete predecessor deletion, generation-wide rebuild equality, and production-scale writer-hold/queue-memory evidence are not established by this ledger. Performance evidence belongs to `polylogue-bp12n.4` and `polylogue-bp12n.5`.
