# Bead readiness audit: implementation cluster

Audit date: 2026-08-03

Audit base: `25434d0f0` (`refactor(sources): unify Claude Code eager and streaming parsers (#3691)`)

Beads evidence: coordinator export `/realm/project/polylogue/.beads/issues.jsonl`; `bd --readonly --directory /realm/project/polylogue comments <id> --json` for the four records with separate comments

Scope: specification readiness only. This audit makes no production change and no Beads mutation.

## Result

| Readiness state | Count |
| --- | ---: |
| EXECUTION-READY | 0 |
| EXECUTION-READY WITH PACKET | 3 |
| DESIGN-BLOCKED | 14 |
| EVIDENCE-BLOCKED | 4 |
| DEPENDENCY-BLOCKED | 6 |
| MISFRAMED/REDUNDANT | 10 |
| Total | 37 |

The safe next Luna wave is `polylogue-tw4ar`, `polylogue-io8np`, and `polylogue-rrxe4`. These lanes are disjoint at the production-write boundary. Review the existing unmerged `polylogue-yazae` commit `c22418c3f` before starting `rrxe4`, then merge that fixture work first if it is accepted. Do not dispatch `polylogue-a7xr.23`, `polylogue-a7xr.25`, `polylogue-6e7m`, or `polylogue-nas1` independently of the content/identity Sol lane.

## Per-Bead audit

The evidence column cites the decisive part of the full record, including design, acceptance criteria, notes, and comments where present. Parent-child edges are shown for context but are not treated as blockers.

| Bead | Status | Full-content evidence | Current source anchors | Dependencies | Smallest missing artifact or dispatch action |
| --- | --- | --- | --- | --- | --- |
| `polylogue-1fijp` | EVIDENCE-BLOCKED | The design fixes the admission arms and the operator note narrows re-acquisition to opportunistic reads. PRs #3668, #3687, and #3688 landed the chokepoint, Drive structural classification, and Antigravity routing. The latest note says the remaining direct writers are structurally distinct and AC (e) still requires 72 hours with zero new quarantines. | `storage/sqlite/archive_tiers/raw_admission.py::admit_raw_observation`; `revision_governance.py::admit_raw_and_parsed_result`; direct `write_source_raw_session` sites in `revision_governance.py`, live batch paths, archive ingest, and repair | Parent `aggz` is closed | A 72-hour live receipt plus an explicit AC rewrite that distinguishes fresh observation admission from copy-forward, post-parse identity binding, and content-addressed replay. No new Luna implementation lane yet. |
| `polylogue-taj0o` | MISFRAMED/REDUNDANT | The note said Stage 2 remained. Current HEAD is PR #3691 and implements exactly Stage 2: one `_claude_code_multiway_parse`, fallback-id primary selection, per-session sidecar accumulation, and deletion of both old grouping paths and Claude-specific reconciliation. | `sources/dispatch.py::_claude_code_multiway_parse`; `sources/parsers/claude/code_parser.py::_SessionAccumulator`; commit `25434d0f0` | None | Reconcile the open Bead against PR #3691. Any residual live old-versus-new archive comparison belongs to reindex verification, not another parser rewrite. |
| `polylogue-rrxe4` | EXECUTION-READY WITH PACKET | Design and AC fix the production seam, one canonical equivalence comparator, four separate metamorphic properties, Hypothesis state-machine use, anti-vacuity mutation, and the focused selector. Both blocking inputs are now closed. | `tests/infra/convergence_harness.py`; `tests/infra/pathology_composer.py`; `maintenance/archive_verification.py::ARCHIVE_VERIFICATION_CHECKS`; existing state-machine precedents under `tests/property` and `tests/unit/storage` | `amrpx` closed; `t0m73` closed | Packet P1 below. Review and merge the existing `yazae` fixture commit first if it will be consumed. |
| `polylogue-yazae` | MISFRAMED/REDUNDANT | The record specifies a production-ingest zoo, queryable manifest, registry and canary consumers, and a conventions rule. An unmerged worktree commit `c22418c3f` already adds `tests/infra/pathology_zoo.py` and its tests, but its two-file diff does not cover the registry, canary, or conventions AC. A fresh lane would duplicate existing work. | Existing HEAD has `tests/infra/pathology_composer.py`; unmerged `c22418c3f` adds `tests/infra/pathology_zoo.py` and `tests/infra/test_pathology_zoo.py` | `amrpx` closed | Review/cherry-pick or reject `c22418c3f`, then narrow this Bead to the missing consumer wiring and conventions rule. Do not regenerate the builder. |
| `polylogue-ey4ro` | DEPENDENCY-BLOCKED | The design fixes five valid instrument kinds, anti-vacuity, `docs/plans/red-backlog.json`, and closure workflow. AC requires zoo, registry binding, dual-path equivalence, and hermeticity gaps to land or remain tracked. | `maintenance/archive_verification.py`; `.agent/CONVENTIONS.md`; missing `docs/plans/red-backlog.json`; test and campaign infrastructure | `rrxe4` open; `yazae` open; `t0m73` closed | Land or reconcile `rrxe4` and `yazae`, then produce one current gate census from `818fy` as the mapping input. |
| `polylogue-wwph1` | DESIGN-BLOCKED | The taxonomy and forcing classes are detailed, but the canonical prompt named in the record is absent from this checkout. The design puts the final report under ignored `.agent/scratch/` while AC also requires committed rerunnable scripts. Those durability rules conflict. | `.agent/CONVENTIONS.md` states `.agent/scratch/` is gitignored; `.agent/scratch/2026-08-03-root-cause-audit-prompt-v2.md` is absent; `maintenance/archive_verification.py` is the graduation target | None | Decide the durable home for enumeration scripts and the final coverage ledger, and decide whether each class pass is one PR or campaign-local scratch plus a single final report. |
| `polylogue-tw4ar` | EXECUTION-READY WITH PACKET | Migration 024 and fingerprint invalidation are landed. The remaining design fixes a bounded `DaemonConverger` stage, content-keyed stale detection, `false_means_pending`, typed-visible append skips, and a cache-hit proof. | `storage/raw_authority_verdict_cache.py`; `storage/raw_authority_verdict_projection.py`; `daemon/convergence_stages.py::make_default_convergence_stages`; `daemon/convergence.py::ConvergenceStage` | None | Packet P2 below. |
| `polylogue-ds4b4` | MISFRAMED/REDUNDANT | The record asks for a third verdict-aware GC check. PR #3625 proved a stronger existing invariant: every live `raw_sessions` or `blob_refs` reference protects its blob for every `RawAuthorityVerdict`, and verdict classification is deliberately not a deletion trigger. Adding verdict logic to GC would create a weaker duplicate safety owner. | `storage/blob_gc.py::run_blob_gc_report`; `tests/unit/storage/test_blob_gc_raw_authority_verdict_invariant.py`; commit `a584cc62e` | Recorded blockers `tw4ar` and `w6hql` are irrelevant to the stronger row-reference proof | Reframe item 4 as satisfied by the existing row-reference invariant, or file a separate retirement-policy Bead if a future path deletes raw rows and needs a retained-successor proof. |
| `polylogue-w6hql` | DEPENDENCY-BLOCKED | Its own design says it is now an umbrella. Enum, derivation, projection, and cache table are landed; closure requires cache convergence, append-cohort coverage, consumer migration, and retirement of fragmented writers. | `core/enums.py::RawAuthorityVerdict`; `archive/raw_authority_verdict.py`; `storage/raw_authority_verdict_projection.py`; `storage/raw_authority_verdict_cache.py`; source migration 024 | `lb39z` closed; `tw4ar` open; `lr6dx` open | Land `tw4ar`, then specify and land append verdict semantics before `lr6dx` removes old readers and writers. No umbrella implementation lane. |
| `polylogue-zok3` | DESIGN-BLOCKED | The record correctly separates query predicates, view parameters, rendering, and globals, but it explicitly leaves three incompatible public shapes: per-view subcommands, a structured `--view`, or DSL projections. | `cli/query_verbs.py`; `cli/read_view_handlers.py::READ_VIEW_HANDLERS`; `archive/viewport/profiles.py::READ_VIEW_PROFILES`; `surfaces/projection_spec.py` | None | One public syntax decision shared with `4n8k` and `jnj.1`, plus a complete 34-flag classification table. |
| `polylogue-4n8k` | DESIGN-BLOCKED | The full description is an agenda, not a decision. It asks each of roughly 18 views to be classified as projection, rendering, or action and asks which DSL extension to use. Those answers change grammar and public CLI compatibility. | Same read-algebra hotspot as `zok3`; grammar in `archive/query/expression.py`; view profiles in `archive/viewport/profiles.py` | None | A checked view-by-view classification and target spelling. Recommended default: units and evidence families are projections, encoding/destination are render concerns, and named views survive only as presets. |
| `polylogue-a7xr.23` | DESIGN-BLOCKED | The record chooses content-defined chunking, but it does not define chunk parameters, manifest/transaction schema, raw reconstruction, GC ownership, or a safe transition from `revision_kind`. Current code uses `revision_kind` across more than forty source and storage modules. The claim that forks do not share root history also cannot support identity decisions here. | `storage/blob_store.py`; `storage/sqlite/archive_tiers/source.py`; revision governance and raw retention modules; broad `revision_kind` consumers | Parent `a7xr`; content/identity Sol lane owns the broader design | A content-addressed raw representation design with chunker versioning, reconstruction atomicity, reference/GC model, and an explicit statement of which revision semantics remain after storage dedup. |
| `polylogue-cijx.2` | DESIGN-BLOCKED | The design chooses root-commit SHA as repository identity and says forks are safe because they do not share it. Real forks commonly share the same root commit, so that premise cannot distinguish fork from mirror. Current code has already moved to normalized remote identity plus `repo_checkouts` and skips bare directories, leaving the proposed root-commit cutover unresolved. | `storage/sqlite/archive_tiers/write.py::_write_repo_edges`, `repo_identity_key`; `archive/session/repo_identity.py`; index tables `repos`, `repo_checkouts`, `session_repos`; `sources/emitter.py::_append_repo_identity_evidence` | Parent `cijx` | Decide the repository equivalence relation for forks, mirrors, remote-less repos, and remote changes. Acceptance evidence must include two forks sharing a root commit and one mirror sharing all history. |
| `polylogue-6e7m` | MISFRAMED/REDUNDANT | AC requires provider titles only in storage and a read-time structural label from repository, file shape, and message count. That implementation already exists and is exercised through real summary reads. | `insights/session_label.py::session_structural_label_for_session`; `tests/unit/insights/test_session_label.py`; `tests/unit/storage/test_title_source_queryable.py`; original implementation in `5e23e6abf` | None; content/identity Sol lane must preserve this contract | Reconcile as landed. A live collision-rate rerun can be a verification receipt, not another implementation. |
| `polylogue-e98k` | DESIGN-BLOCKED | PR #3637 landed startup/rebuild observability and the computed Polylogue-side budget. Its body explicitly deferred AC 1, one declared value driving both Polylogue profiles and Sinnix cgroup limits. The Bead's optional-env proposal still omits allocation ratios and headroom policy. | `storage/sqlite/connection_profile.py::mapped_bytes_budget`; `core/metrics.py` cgroup readers; daemon and rebuild call sites; Sinnix `modules/services/polylogue.nix` outside this worktree | Cross-repo Sinnix contract, not represented by a dependency edge | Decide which repo owns the budget, how it allocates mmap/cache/read concurrency, and how `MemoryHigh`/`MemoryMax` headroom derives from it. Then split live canary and generation-retention checks from the cross-repo config change. |
| `polylogue-6kur` | MISFRAMED/REDUNDANT | PR #3661 landed the safe FK-impossible cull. The notes correctly prohibit touching `empty_sessions` and expand the remainder into convergence parity, raw-authority drain deletion, and relocation to blob GC/raw retention. Those are separate owner modules and decisions, not the original safe subset. | `storage/repair.py`; `maintenance/preview.py`; `maintenance/targets.py`; commit `ca41b5463` | `ne6k` decision for empty sessions; raw-authority drain/retirement work | Reconcile the landed safe subset, then split remaining targets by owning invariant. Do not use this umbrella for a second broad repair rewrite. |
| `polylogue-a7xr.26` | EVIDENCE-BLOCKED | The note fixes the intended result, deletion of the aiosqlite implementation while preserving async signatures, but explicitly requires a realistic wrapper benchmark and says to stop if the result exceeds roughly 2x. | `storage/sqlite/async_sqlite.py::SQLiteBackend`; `async_sqlite_archive.py`; `async_sqlite_raw.py`; sync archive/repository engine; async consumers in pipeline and repository | Parent `a7xr` | A reproducible realistic batch-read benchmark, with workload, archive size, concurrency, baseline, wrapper prototype, and result. |
| `polylogue-a7xr.25` | DESIGN-BLOCKED | The operator chose the ride-the-rebuild retention rule, but the record does not fix how an event references one or more blocks. `session_events` has `source_message_id` and `payload_json`, no block-ref field or relation. "No DDL change" conflicts with "reference block ids otherwise" unless refs remain encoded inside JSON. | `storage/sqlite/archive_tiers/write.py::_write_session_events`, `_SESSION_EVENTS_REDUNDANT_TYPES`; `session_events` DDL; Codex event/block lowering | Parent `a7xr`; content/identity Sol lane owns the representation; rebuild campaign `818fy/xselt` owns application | Decide between a typed event-to-block relation and minimal reference JSON, including one-to-many mappings, unresolved refs, and consumer hydration. |
| `polylogue-io8np` | EXECUTION-READY WITH PACKET | AC requires one bounded topology and parent-chain envelope builder used by MCP and daemon HTTP, with the same node limit. Current code still has independent MCP unbounded tuples and daemon bounded dictionaries. | `daemon/topology_http.py::build_topology_envelope`, `build_parent_chain_envelope`; `mcp/payloads.py::session_topology_payload`; `mcp/server_cutover.py` topology route | None | Packet P3 below. |
| `polylogue-mjupn` | MISFRAMED/REDUNDANT | The finding is real, but it assumes CLI read views, MCP `read(view=topology)`, and MCP `get(projection=...)` are one vocabulary before `4n8k`/`jnj.1` decides that product contract. A shared table now would freeze the disputed model. | `cli/read_view_handlers.py`; `mcp/server_cutover.py` read/get branches; `surfaces/projection_spec.py`; `archive/viewport/profiles.py` | Read-algebra design cluster | Merge the evidence into `4n8k`/`jnj.1`; implement one registry only after the public projection boundary is decided. |
| `polylogue-nbls5` | DESIGN-BLOCKED | AC explicitly permits either deleting dead scaffolding and correcting docs or wiring MCP. Current architecture exposes ten role-gated dispatcher tools, so adding eleven generated tools would contradict the consolidated MCP surface. The record does not choose a dispatcher operation instead. | `mcp/insight_tool_contracts.py::InsightListToolSpec`; `insights/registry.py::INSIGHT_REGISTRY`; `mcp/server_cutover.py::_INSIGHT_PROJECTIONS`; MCP declarations/contracts | None | Choose the supported MCP route. Recommended default: add registry names behind the existing `query` dispatcher, then delete per-tool scaffolding and correct claims. |
| `polylogue-oj4oo` | MISFRAMED/REDUNDANT | The execution recipe chose canonical `OperationStatus`. PR #3657 moved it to `core/enums.py`, aliases `BackfillStatus`, generator-tied ops DDL, and removed the comment-only embedding translation. | `core/enums.py::OperationStatus`; `maintenance/planner.py::BackfillStatus`; `operations/operation_status.py`; ops DDL/write helpers; commit `0a39fe098` | None | Reconcile as landed. |
| `polylogue-lm62x` | DESIGN-BLOCKED | The record itself asks whether query runs, route observations, and workflow surfaces need different granularity. The value sets represent execution transport, route transport, and product workflow, so a blind enum merge could erase valid distinctions. | `storage/sqlite/archive_tiers/ops.py` query/route tables; `archive/query/production_evaluator.py::Surface`; `product/workflows.py::WorkflowSurface`; `operations/route_observation.py` | None | Name and define the axes, then choose whether they share a transport enum plus a broader workflow enum. Required evidence is a join/use-site census, not member-set equality. |
| `polylogue-jglh` | DESIGN-BLOCKED | AC delegates a semantic judgment for every pair. Several listed pairs are only coincidentally equal today, including completeness/exactness and provider-specific versus public fidelity. The Bead does not record those judgments. | Listed enum/Literal definitions across `core`, `archive/viewport`, `storage`, `sources`, `schemas`, `insights`, and `surfaces` | None | A pair-by-pair axis matrix: shared meaning, direction of dependency, canonical home, compatibility effect, and explicit keep-split rationale where applicable. |
| `polylogue-mzp8` | DESIGN-BLOCKED | The InvalidationReason merge is fixed, but CostBasis still permits merge or rename and the other two pairs only say "rename the narrower one." Those names and axes are public typing/import contracts. | `maintenance/invalidation.py`; `maintenance/preview.py`; `archive/semantic/pricing.py`; `archive/semantic/cost_records.py`; `agent_integration/installer.py`; `operations/specs.py`; measurement modules | None | Choose canonical names and value mappings for all four pairs, including compatibility/import policy. Recommended default: merge InvalidationReason and distinctly name pricing-catalog basis versus recorded-cost authority. |
| `polylogue-fbkr` | EVIDENCE-BLOCKED | The record gives two valid outcomes for manual frontier apply and reset: delete them if automatic convergence covers every safe case, or promote them to explicit consent if genuinely destructive judgment remains. The deciding coverage proof is absent. | `cli/commands/maintenance/_raw_identity.py`; `product/raw_authority.py::apply_frontier`; daemon automatic frontier application; `maintenance/raw_authority_reset.py` with test-only callers | Raw-authority Phase 1 drain is closed; retirement work remains | An executable-plan coverage receipt comparing automatic daemon application with every manual plan shape, plus a live post-drain reset-need check. |
| `polylogue-5vft` | MISFRAMED/REDUNDANT | This combines three independent changes. Disposable-tier bootstrap and periodic preflight recovery landed in #3133, while `--only-missing` still forbids promotion and managed reset still refuses. Current daemon architecture also uses acquire-only degraded mode and resumable bulk rebuilds, which the old self-heal design predates. | `maintenance/rebuild_index.py::validate_rebuild_index_request`; `cli/commands/reset.py::_archive_index_targets`; `daemon/cli.py::_periodic_schema_preflight_recheck`; `daemon/bulk_rebuild.py` | None | Split into: effective-full selection promotion, managed-generation reset/replace consent, and derived-mismatch bulk-rebuild routing. Reconcile the already-landed ops bootstrap/recheck slice. |
| `polylogue-1lm` | DEPENDENCY-BLOCKED | Its selector/transform/budget algebra is specific, including adjacency and resolvable omission refs, but it explicitly depends on `jnj.1` and says it must follow the shared projection normalizer. | `archive/semantic/content_projection.py::ContentProjectionSpec`; `surfaces/projection_spec.py::ProjectionSpec`; context compiler and rendering paths | `jnj.1` open; parent `4p1`; related `ap7` | Land the read-algebra ownership decision and normalizer first. Then refresh the stale, absent prework packet against current symbols. |
| `polylogue-jnj.1` | DESIGN-BLOCKED | Its design lists unresolved ownership pairs: ProjectionSpec versus ContentProjectionSpec, RenderFormat versus formatting registries, destination ownership, and profiles versus handlers. It explicitly says to decide ownership before extending anything. | `surfaces/projection_spec.py`; `archive/semantic/content_projection.py`; `archive/viewport/profiles.py`; `cli/read_view_handlers.py`; rendering formatters | Parent `jnj` | One concern-to-owner matrix and hard-cut migration order. This decision must absorb `4n8k`, `zok3`, and `mjupn` evidence. |
| `polylogue-jnj.2` | DESIGN-BLOCKED | Description says facets becomes a real verb. Design allows either a top-level command or a facets projection. The query-first command floor deliberately keeps a small verb set, so this is a public product choice. | CLI analyze/facets commands and tests; root request/filter mapping; shared query transaction from `z9gh.9.1` | `z9gh.9.1` closed; parent `jnj` | Choose verb versus named projection and define JSON envelope parity for exact-ID selection. Recommended default: a named analyze projection over the canonical relation, with any top-level spelling generated as the same operation rather than a second filter owner. |
| `polylogue-jnj.9` | DESIGN-BLOCKED | The separate comment proves `config --show-layers` already satisfies most read/list scope. Remaining set/get semantics do not say which layer is writable, how Nix-managed values behave, or whether secrets may be written. | `cli/commands/config.py`; `config.py::effective_config_payload`, inventory and five-layer resolver; generated `docs/configuration.md` gap | Parent `jnj` | A writable-layer and secret policy. Recommended default: user-config writes only, refuse deployment/site/env-owned keys with an exact edit target, and generate docs from inventory. |
| `polylogue-jnj.10` | MISFRAMED/REDUNDANT | The description asks for `polylogue syntax`; design proposes `help query` or `find --help-syntax`, completion installation in init/Nix/Homebrew, saved-query examples, zero-result hints, and success hints. AC is a generic query-parity template that does not verify these products. | Completion code and tests under `cli`; generated CLI reference; agent integration; distribution files outside this repo for Nix/Homebrew | Parent `jnj`; unnamed saved-view and empty-result coordination | Split into syntax-card generation, completion installation/distribution, and TTY-only teaching hints. Replace the unrelated AC before dispatch. |
| `polylogue-f7zw` | DEPENDENCY-BLOCKED | The design and AC fully specify one language-neutral corpus, edge vectors, mutation failures, digest, and lockstep update. The deliverable explicitly requires identical fixtures and tests in both Polylogue and Sinex, while current Sinex has no counterpart. | `tests/fixtures/material_protocol/v1/small-session`; `docs/material-protocol-v1.md`; Python encoder; no matching Sinex fixture/encoder test | Parent `303r`; named counterpart `sinex-4j2.1` is not represented as a Beads edge here | Land or jointly schedule the Sinex encoder/test counterpart and choose the shared corpus source of truth. |
| `polylogue-7aw` | DEPENDENCY-BLOCKED | Design fixes OriginSpec-based config evidence, content-addressed revisions, partial ExecutionContextRef resolution, and honest cohort claims. It consumes the still-open OriginSpec and actor/context contracts instead of defining parallel ones. | `sources/origin_specs.py`; `core/refs.py::{ActorRef,ExecutionContextRef}`; current work-evidence queries; agent integration and hook evidence | Parent `2qx` open; related `h6r` open; `37t.10` remains consumer | Land the relevant `2qx` OriginSpec extension contract and `h6r` execution-context identity before assigning storage and resolver files. |
| `polylogue-37t.8` | EVIDENCE-BLOCKED | PR #2827 landed safe Claude/Codex routing, explicit unsupported results, and `continue --exec`. The note says the only remainder is a manual real-session reopen receipt. | `archive/resume_routing.py::route_resume`; `cli/query_verbs.py` continue route; unit tests | Parent `37t` | One operator-visible dogfood receipt for a real Claude and Codex session, including command and observed reopened harness. |
| `polylogue-37t.7` | MISFRAMED/REDUNDANT | The design names the retired bespoke devloop as first consumer, contrary to current repository policy. It also combines product wiring with a separate chaos-drill experiment and leaves SessionStart versus explicit CLI injection as an alternative. | `devtools workspace failure-context`; `.cache/verify` receipts; `api/archive.py::compile_context`; current agent SessionStart integration | Parent `37t` | Rewrite around current `devtools verify` receipts and the installed agent integration. Split context construction from session-cut recovery experiments, and choose explicit versus automatic injection. |
| `polylogue-nas1` | DEPENDENCY-BLOCKED | The ontology is settled: provider-native resume topology and context delivery are orthogonal, and heuristics cannot create resume edges. The chosen representation explicitly reuses `1vpm.6` work/context graph and `37t.22` receipts. | `session_links` write/queries; `storage/sqlite/archive_tiers/context_delivery_write.py`; context delivery surfaces; `core/refs.py` | `7s57` closed; `37t.22` closed; `1vpm.6` open; content/identity Sol lane owns reconciliation | Land the provider-neutral graph relation from `1vpm.6`, then let the content/identity lane fix the join and provenance contract. No standalone topology schema. |

## Content/identity Sol lane reconciliation

The content/identity lane owns `a7xr.23`, `a7xr.25`, `6e7m`, and `nas1` as one design problem. At audit time its worktree branch had no committed design document yet, so this audit records boundaries rather than guessing its conclusion.

### No-duplication boundaries

| Bead | This audit's boundary | Sol lane obligation |
| --- | --- | --- |
| `a7xr.23` | No CDC implementation packet. The current Bead lacks a raw manifest, transaction, reconstruction, and GC design. | State whether CDC changes storage only or also retires revision semantics. Define chunker versioning, chunk refs, reconstruction, atomic publication, and migration/rebuild policy. |
| `a7xr.25` | No writer patch until event-to-block references have a declared representation. | Define the identity of an event-to-block reference, including one-to-many mappings, inherited/composed messages, unresolved refs, and whether the representation is typed DDL or minimal JSON. Preserve current `_SESSION_EVENTS_REDUNDANT_TYPES` evidence. |
| `6e7m` | Treat the read-time structural label as landed. Do not introduce a stored derived title. | Preserve `sessions.title` as provider evidence and `insights/session_label.py` as a read projection. Any new identity design must keep provider title provenance and avoid stale serialized labels. |
| `nas1` | No new `session_links` type for context use and no time/tool-name inference. | Compose provider-native resume assertions with `37t.22` delivery receipts through the `1vpm.6` graph. Define unresolved and abandoned deliveries without fabricating successor topology. |

### Cross-Bead invariants

1. Content identity, storage deduplication, and topology evidence are separate axes. Equal chunks do not prove session identity or resume topology.
2. Provider evidence remains immutable input. Read-time labels and context-assistance joins are projections.
3. Rebuildable index changes use canonical DDL and a declared `IndexDeltaDeclaration`; durable raw-byte or user-evidence changes require the durable-tier migration and backup regime.
4. The Sol lane should cite the current implementations that already landed for `6e7m` and partial event filtering instead of proposing replacements from the old Bead snapshots.

## Implementation packets

### P1: `polylogue-rrxe4` convergence-property loop

**Owned files and symbols**

- `tests/infra/convergence_harness.py`: extend the existing production adapter, without a second convergence state machine.
- `tests/infra/archive_equivalence.py`: add `canonical_archive_facts` and `assert_archives_equivalent`, owning canonical row comparison modulo declared generation/timestamp fields.
- Four new property modules under `tests/property/` for order invariance, incremental-versus-bulk, idempotence, and append-prefix consistency.
- Narrow fixture adapters consuming `tests/infra/pathology_composer.py` and, after review, `tests/infra/pathology_zoo.py`.

**Avoided files**

- Do not change `polylogue/daemon/convergence.py`, production stage semantics, `storage/sqlite/archive_tiers/write.py`, or `maintenance/archive_verification.py` merely to make the harness pass.
- Do not edit the in-flight zoo builder until commit `c22418c3f` has been reviewed and ownership transferred.
- Do not normalize away semantically meaningful differences in the comparator.

**Production route**

Corpus program to real raw/parsed input, production ingest writer, production `DaemonConverger` stages, then `ARCHIVE_VERIFICATION_CHECKS`. Bulk and trickle arms must call the same product routes with different schedules.

**Anti-vacuity test**

Use a historical deferred-tail or parent-arrival fixture. A mutation that disables session-link resolution/retry or makes one order skip the divergent tail must make order invariance fail while the unmodified production route passes. The comparator must also fail when one persisted message/block/action row is removed from one arm.

**Focused verification**

```text
devtools test -k convergence_property
devtools test tests/unit/daemon/test_convergence_restart_law.py
devtools verify
```

**Commit boundary**

One test-infrastructure commit containing the comparator, harness extension, and four property modules. Production fixes discovered by a red property belong in separate Beads/PRs.

**Merge ordering**

Review and merge `c22418c3f` first if the zoo becomes an input. This packet can run in parallel with P2 and P3 because it owns test infrastructure only.

### P2: `polylogue-tw4ar` verdict-cache convergence

**Owned files and symbols**

- `polylogue/storage/raw_authority_verdict_cache.py`: add `find_stale_raw_authority_verdict_cohorts`, `refresh_raw_authority_verdict_cohorts`, and a `RawAuthorityVerdictRefreshResult` carrying refreshed, skipped-append, and remaining counts.
- `polylogue/daemon/convergence_stages.py`: add `make_raw_authority_verdict_cache_stage` and register it in `make_default_convergence_stages`.
- `tests/unit/storage/test_raw_authority_verdict_cache.py` and `tests/unit/daemon/test_convergence_stages.py`: real stage-interface coverage.

**Avoided files**

- Do not edit migration 024 or durable DDL unless current schema is proven insufficient.
- Do not change `derive_raw_authority_verdict`, append-cohort semantics, blob GC, or fragmented-table retirement.
- Do not run classification in a worker process. The daemon main process remains the sole SQLite writer.

**Production route**

`DaemonConverger` invokes the registered stage. `check` discovers cohort keys missing from `raw_authority_verdicts` or mismatched by content fingerprint. `execute` calls the existing projection/cache write path in a bounded batch. Remaining work returns pending through `false_means_pending` and convergence debt.

**Anti-vacuity test**

Seed a cached cohort, mutate its `raw_sessions` membership/blob hash through production writers, and assert the stage detects the fingerprint change and refreshes the cache. A second stage pass must hit the cache without invoking `project_raw_authority_verdicts`. Seed an append cohort and assert a typed skipped count rather than an exception.

**Focused verification**

```text
devtools test -k verdict_cache
devtools test -k convergence
devtools verify
```

**Commit boundary**

One feature commit for the registered stage, helper query, and tests. Append verdict semantics and old-table retirement remain separate commits under `w6hql`/`lr6dx`.

**Merge ordering**

Land before any `w6hql` consumer cutover or `lr6dx` retirement. It is disjoint from P1 and P3.

### P3: `polylogue-io8np` shared topology envelopes

**Owned files and symbols**

- `polylogue/insights/topology_envelope.py`: add `build_topology_envelope` and `build_parent_chain_envelope`, and own node/edge shaping, `DEFAULT_NODE_LIMIT`, `MAX_NODE_LIMIT`, readiness, and parent-chain derivation.
- Migrate `polylogue/daemon/topology_http.py::build_topology_envelope` and `build_parent_chain_envelope` to thin adapters or re-exports.
- Migrate `polylogue/mcp/payloads.py::session_topology_payload` and the MCP topology route to the shared builder and expose the same node limit.
- Update focused daemon tests and add `tests/unit/mcp/test_topology_payload.py` for cross-surface parity.

**Avoided files**

- Do not change topology query semantics, `session_links`, lineage composition, or HTTP routing beyond accepting/passing the common limit.
- Do not fold this into the unresolved general read-view registry work.
- Preserve the typed MCP payload contract or regenerate its schema deliberately if the shared envelope adds truncation fields.

**Production route**

`Polylogue.get_session_topology` returns one `SessionTopology`; both daemon HTTP and MCP pass it through the same bounded projection. Parent-chain uses the same shared node/edge vocabulary.

**Anti-vacuity test**

Build a topology larger than `DEFAULT_NODE_LIMIT` with an edge whose parent is dropped. Both HTTP and MCP must return identical kept node/edge ids, truncation count, cycle/unresolved state, and no dangling edge. Removing the shared bound or restoring the MCP local constructor must fail the parity test.

**Focused verification**

```text
devtools test tests/unit/daemon/test_topology_endpoint.py tests/unit/daemon/test_topology_stack.py
devtools test tests/unit/mcp/test_topology_payload.py
devtools verify
```

**Commit boundary**

One refactor commit containing the shared builder, both adapters, and parity tests. No topology storage changes.

**Merge ordering**

Independent of P1 and P2. Merge before any later MCP projection-registry work so that work consumes one topology envelope.

## Design agendas

Every row below is a blocking decision. Size or difficulty is not the reason for any DESIGN-BLOCKED classification.

| Bead | Exact decision question | Alternatives | Recommended default | Affected contracts | Acceptance evidence required |
| --- | --- | --- | --- | --- | --- |
| `wwph1` | Where do durable campaign scripts and the final coverage ledger live? | Ignored `.agent/scratch`; tracked `docs/plans` plus `devtools`; one final tracked report with transient scratch checkpoints | Track rerunnable enumerators under `devtools` or `.agent/scripts` and the final ledger under `docs/plans`; keep raw candidate dumps in scratch | Agent conventions, campaign reproducibility, registry graduation | Fresh clone can rerun one class and reproduce denominator/judgment counts without the missing prompt file |
| `zok3` | What public syntax owns view-specific parameters? | Per-view subcommands; structured `--view`; DSL projection/profile syntax | DSL projection plus typed profile parameters, with hard removal of inapplicable flat flags | CLI grammar, help, completions, generated docs | One full 34-flag classification and equivalent command examples for transcript, neighbors, correlation, and context |
| `4n8k` | Which current views are projections, renderings, presets, or distinct actions? | Preserve views; move all to units; classify individually | Individual classification, retaining named views only as presets over Query x Projection x Render | Query grammar, view registry, render profiles, MCP/read parity | Table for every current view plus round-trip/parity fixtures for each moved family |
| `a7xr.23` | Does CDC replace only whole-file blob storage or also revision authority semantics? | Storage-only chunk manifests; full removal of revision model; hybrid interim prefix retirement | Storage-only CDC first. Keep provenance/revision semantics until every authority consumer has an explicit replacement | Durable source tier, blob GC, admission, replay, rebuild | Golden reconstruction, crash atomicity, chunker-version migration, corpus savings and wall-clock measurement |
| `cijx.2` | What makes two observations the same repository when forks and mirrors share history? | Root commit; normalized remote; repository UUID/assertion; composite evidence cluster | Evidence cluster with root history plus explicit remote/forge identity, preserving mirror aliases and fork distinctions | Repo query, checkout observations, file evidence, reindex semantics | Fixtures for mirror, fork with shared root, remote rename, remote-less checkout, and reused path interval |
| `e98k` | Which configuration owns the memory budget and how are profile/cgroup shares derived? | Sinnix-owned budget exported to Polylogue; Polylogue-owned budget imported by Nix; warning-only independent constants | Sinnix deployment budget with an explicit Polylogue env contract and documented deterministic profile/headroom ratios | Polylogue config, SQLite profiles, systemd limits, rebuild canary | Unit derivation tests in both repos and live journal/cgroup receipt during a canary rebuild |
| `a7xr.25` | How does a session event reference all blocks that carry its lowered payload? | Minimal refs inside JSON; nullable single block id; normalized event-to-block relation | Normalized index-tier event-to-block relation, because one event can map to zero, one, or many blocks | Index DDL, writer, event readers, reprocess/reindex | One-to-many, unresolved, inherited-prefix, and consumer round-trip fixtures; size/write-volume census |
| `nbls5` | How are registry insights exposed through the consolidated MCP tool set? | Eleven generated tools; no MCP support; existing dispatcher operation | Existing `query` dispatcher with registry-backed projection names; delete dead tool-generation scaffolding | MCP declarations, capability roles, tool contracts, registry docs | Discovery plus one call for every registry entry through the same dispatcher, with no new top-level tools |
| `lm62x` | Are query surface, route transport, and workflow surface one axis? | One broad enum; transport enum plus workflow enum; three documented axes | A shared transport enum for physical call routes and a distinct broader workflow channel enum | ops DDL, evaluator, route observations, product workflows | Producer/consumer/join census and cross-table fixture showing lossless correlation |
| `jglh` | Which identical member sets represent the same semantic axis? | Collapse every pair; keep every pair; pair-by-pair decision | Pair-by-pair decision with shared definitions moved to the lowest legal layer | Layering, public types, DDL helpers, payload schemas | One matrix row per listed pair and tests that semantic mappings, not spellings, remain correct |
| `mzp8` | What are the canonical names and mappings for same-name divergent vocabularies? | Merge values; rename one axis; create a shared supertype | Merge InvalidationReason; rename catalog pricing basis and recorded cost authority; rename installer and metric-specific types | Imports, serialized values, schemas, docs | Call-site census, explicit old-to-new mapping, mypy and generated schema checks |
| `jnj.1` | Which abstraction owns semantic inclusion, data projection, rendering, destination, and presets? | Extend current `ProjectionSpec`; prefer `ContentProjectionSpec`; create a normalizer over both | One normalized request with separate selection, content projection, and render/profile components; existing types become adapters then disappear | CLI/MCP/API/web/export/context | Concern-to-owner matrix, one canonical serialized spec, and cross-surface parity fixture |
| `jnj.2` | Is facets a top-level verb or a named projection over analyze/query? | New verb; named projection; generated alias | Named projection over the canonical relation, with any convenience spelling calling that exact operation | CLI command floor, filter scope, JSON envelope | Exact-ID and full-filter positive/empty cases proving no scope broadening |
| `jnj.9` | Which config layer may `set` mutate and how are managed/secret keys handled? | Write user config; edit winning layer; generate deployment instructions only | Mutate user config only. Refuse managed/site/env and secret writes with a precise edit target | Config resolver, CLI, docs generator, Nix ownership | Tests for user write, env override, Nix-managed refusal, unknown key, secret redaction, and generated docs |

## Dependency DAG

```text
Testing and verification
  amrpx [closed] + t0m73 [closed]
    -> yazae existing commit review/narrowing
    -> rrxe4 [packet P1]
  yazae + rrxe4 + t0m73
    -> ey4ro

Raw authority
  tw4ar [packet P2]
    -> w6hql umbrella completion
  append-verdict design + tw4ar
    -> lr6dx retirement
  ds4b4 item 4
    -> superseded by existing row-reference GC invariant

Read algebra
  4n8k + zok3 + mjupn evidence
    -> jnj.1 ownership decision
  jnj.1
    -> 1lm
  jnj.1 decision
    -> jnj.2 implementation shape
  io8np [packet P3]
    -> later MCP projection-registry work

Content and identity
  content/identity Sol design
    -> a7xr.23
    -> a7xr.25
    -> nas1
  1vpm.6
    -> nas1
  6e7m [already landed]
    -> preserve as invariant in Sol design

Interop and context
  2qx OriginSpec + h6r execution context
    -> 7aw
  Sinex material-protocol counterpart
    -> f7zw

Evidence gates
  1fijp implementation [landed]
    -> 72-hour live receipt
  a7xr.26
    -> wrapper benchmark
  fbkr
    -> automatic-versus-manual plan coverage receipt
  37t.8 implementation [landed]
    -> manual reopen receipts
```

## Shared hotspots and lane serialization

| Hotspot | Beads | Rule |
| --- | --- | --- |
| `storage/sqlite/archive_tiers/write.py` and index DDL | `a7xr.25`, `cijx.2`, residual `1fijp` interpretation | Serialize under the content/identity design. No concurrent implementation lanes. |
| CLI read algebra and grammar | `4n8k`, `zok3`, `mjupn`, `jnj.1`, `jnj.2`, `1lm` | One design lane, then one migration branch. Do not split by Bead because every item touches the same public contract and handlers. |
| `core/enums.py`, ops DDL, generated schemas | `lm62x`, `jglh`, `mzp8`; `oj4oo` already landed | Resolve all vocabulary agendas first, cluster non-overlapping definitions, and serialize shared-file edits. |
| Raw-authority cache/convergence | `tw4ar`, `w6hql`, `lr6dx` | Land P2 first. Append semantics second. Consumer/write-path retirement last. |
| MCP payload and projection dispatch | `io8np`, `nbls5`, `mjupn` | P3 may land now. Hold dispatcher/registry work until the read-algebra decision; rebase it after P3. |
| Test pathology infrastructure | `yazae`, `rrxe4`, `ey4ro` | Review existing yazae commit first. P1 next. Red-backlog mapping last. |

## Safe parallel plan

Wave 0 is reconciliation work, not a Luna implementation lane:

1. Review `c22418c3f` for `yazae` against its production-route and manifest AC.
2. Reconcile landed open Beads `taj0o`, `6e7m`, `oj4oo`, and `ds4b4` in coordinator state.
3. Wait for the content/identity Sol document before dispatching its four covered Beads.

Wave 1 can use three Luna lanes in parallel:

1. P2 `tw4ar`, daemon/storage cache convergence.
2. P3 `io8np`, shared topology envelopes.
3. P1 `rrxe4`, convergence property harness, after the `yazae` review determines whether to consume its builder.

Wave 2 begins only after Wave 1 and design decisions:

1. `ey4ro` after `rrxe4` and narrowed `yazae` land.
2. Raw-authority append verdict coverage, then `lr6dx`, then close the `w6hql` umbrella.
3. One read-algebra branch covering `4n8k`/`zok3`/`mjupn`/`jnj.1`, followed by `1lm` and the chosen `jnj.2` surface.
4. MCP registry exposure after the read-algebra decision and P3.

## Corrections to prior readiness claims

Several records carry old delivery labels such as `readiness=A-implementation-ready` or point to snapshot packets under `.agent/handoffs/...` that are absent from this checkout. Those labels are not current specifications and were not accepted as readiness evidence.

- `taj0o`: a title-level audit would call it open parser work. Full notes plus current history show PR #3691 already landed the entire remaining stage.
- `6e7m`: its open title suggests missing title design. Current source already implements its decisive AC as a read-time structural label and keeps provider evidence in storage.
- `oj4oo`: the open vocabulary title is stale after PR #3657.
- `ds4b4`: its title suggests a missing GC invariant. Full source and the anti-vacuity suite show the requested verdict-specific check would duplicate the stronger existing row-reference invariant.
- `37t.8`: the implementation is landed; only manual evidence remains.
- `e98k`: many AC and detailed design text do not make it fully ready. The already-landed PR explicitly deferred the one-source-of-truth cross-repo contract, and the remaining scaling/headroom policy is unspecified.
- `jnj.1`, `1lm`, `7aw`, `37t.7`, and `37t.8`: old prework-packet pointers were generated from `8a975a40` and the referenced files are absent. Current symbols and current architecture control this audit.
- `37t.7`: the old `A-implementation-ready` label also names the retired devloop as first consumer, so it cannot be dispatched without reframing.
- `yazae`: the open Bead is no longer greenfield because an unmerged implementation commit exists. Dispatch must begin with review, not another builder.

## Residual uncertainty

The content/identity Sol worktree had not committed its design document when this audit was written. Its eventual decisions may change the readiness of `a7xr.23`, `a7xr.25`, and `nas1`, but they must preserve the no-duplication boundaries above. The coordinator export was slightly older than this worktree's local `.beads/issues.jsonl`; the coordinator export was used by instruction, and separate comments were read with explicit read-only coordinator-directory commands. No live archive mutation or live 72-hour measurement was performed.
