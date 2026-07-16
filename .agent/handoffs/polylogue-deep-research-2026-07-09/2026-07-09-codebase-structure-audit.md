---
created: 2026-07-09
purpose: Combined read-only evidence artifact for three sibling audit-lane beads (polylogue-9e5 epic)
status: complete
project: polylogue
covers:
  - polylogue-9e5.14 (facade decomposition map)
  - polylogue-9e5.15 (dead-code / script-silo sweep — audit half only)
  - polylogue-9e5.5 (exhaustive table read/write matrix -> dead-table kill list)
---

# Codebase structure audit — 2026-07-09

Read-only census work for three sibling beads under the `polylogue-9e5` "audit
lane" epic. No product code was modified. Each section below was produced by
an independent research pass (rg call-census, vulture/coverage/affordance
intersection, and a table read/write matrix script respectively); method,
gotchas, and full data are preserved per section so the census is defensible
and re-runnable.

---

## 1. polylogue-9e5.14 — Facade decomposition map

### Headline finding

The premise that `polylogue/api/archive.py`'s facade is "a thin composition
root over the 10-mixin `SessionRepository`" does **not** hold structurally
today. Of 102 public methods on the real facade class (`PolylogueArchiveMixin`),
only **2** (`get_messages_paginated`, `iter_messages`) literally delegate via
`self.repository.*` into the named `storage/repository/` mixins. The other
~90 non-alias methods call `ArchiveStore` directly — a separate synchronous
`sqlite3`-based class in `polylogue/storage/sqlite/archive_tiers/archive.py`
(opened per-call via `with ArchiveStore.open_existing(self.config) as archive:
archive.<name>(...)`). `RepositoryWriteMixin.save_parsed_session` itself also
bottoms out in `ArchiveStore`, so `ArchiveStore` really is the shared
canonical engine — but the facade duplicates the `open_existing` boilerplate
call-by-call rather than routing through the repository layer at all. A large
sub-cluster (tags/metadata/marks/annotations/views/recall-packs/workspaces/
corrections/blackboard — ~38 methods) has **no equivalent among the 10 named
mixin files**; they are 1:1 proxies onto `ArchiveStore`'s own `user.db`-backed
CRUD surface, a third parallel implementation. "Move to mixin" is therefore
not literally available for this cluster without first re-plumbing them onto
the async `SessionRepository`/`SQLiteBackend` stack — a materially larger
change than a mechanical relocation.

### Exact counts

- `polylogue/api/archive.py` defines **4 classes**: `SessionNotFoundError`
  (exception, 0 methods); `_ArchiveInsightExportOperations` (private duck-typed
  Protocol adapter, 7 unprefixed methods, instantiated once inside
  `export_insight_bundle`, referenced nowhere else in the repo);
  `_ArchiveNeighborRuntime` (same pattern, 14 unprefixed methods, instantiated
  once inside `neighbor_candidates`); **`PolylogueArchiveMixin`** — the real
  facade, **102 public methods**, 18 private. 7 + 14 + 102 = 123 unprefixed-name
  methods total in the file, close to the bead's "~126" estimate (the two
  private adapters' method names are dictated by the Protocol they satisfy,
  not by any real external caller — some even collide in name with
  `PolylogueArchiveMixin` methods, e.g. `list_session_profile_insights`, `get`,
  `search`, `count` — worth de-confusing separately).
- `polylogue.api.Polylogue` = `PolylogueArchiveMixin` + `PolylogueEmbeddingsMixin`
  + `PolylogueInsightsMixin` + `PolylogueIngestMixin` (`polylogue/api/__init__.py:44`)
  — `archive.py`'s class is one of four facade mixins composing the public
  `Polylogue` object, not the whole thing.

### Mixin roster (current, from `polylogue/storage/repository/__init__.py`)

`SessionRepository` composes exactly 10 mixin classes:

| Mixin class | File(s) |
| --- | --- |
| `RepositoryArchiveReadMixin` (= `RepositoryArchiveSessionMixin` + `RepositoryArchiveQueryMixin` + `RepositoryArchiveTreeMixin` + `RepositoryArchiveSearchMixin`) | `archive/reads.py` (sub-mixins in `archive/sessions.py`, `archive/queries.py`, `archive/tree.py`, `archive/search.py`) |
| `RepositoryInsightProfileReadMixin` | `insight/profile_reads.py` |
| `RepositoryInsightRunProjectionReadMixin` | `insight/run_projection_reads.py` |
| `RepositoryInsightTimelineReadMixin` | `insight/timeline_reads.py` |
| `RepositoryInsightThreadReadMixin` | `insight/thread_reads.py` |
| `RepositoryInsightSummaryReadMixin` | `insight/summary_reads.py` |
| `RepositoryInsightTopologyReadMixin` | `insight/topology_reads.py` |
| `RepositoryRawMixin` | `raw/repository_raw.py` |
| `RepositoryWriteMixin` (uses `archive/writes/metadata.py::metadata_read_modify_write`, `archive/writes/sessions.py::delete_session_via_backend` as plain helper functions) | `archive/repository_writes.py` |
| `RepositoryVectorMixin` | `vectors/repository_vectors.py` |

This matches CLAUDE.md's "archive reads, archive writes, raw, vectors, + six
insight readers" — the layout has grown internal sub-files since that summary
was written but the 10-mixin identity still holds.

### Full decomposition table (102 methods)

| Method | Consumers | Delegates to | Verdict |
|---|---|---|---|
| `get_session` | CLI, MCP, TUI, daemon, tests, internal (api/context/insights/storage) | `ArchiveStore.read_session`/`resolve_session_id` | keep-on-facade (core primitive, too high-fanout to relocate) |
| `explain_import` | MCP, daemon, tests | `sources/import_explain.py` | keep-on-facade (pure API-boundary translation) |
| `postmortem_bundle` | CLI, MCP | `ArchiveStore.list_summaries` + `self.repository.get_session_profiles_batch` + `insights/postmortem.py` | keep-on-facade (multi-source orchestration) |
| `pathology_report` | MCP | `ArchiveStore.list_summaries` + `insights/pathology.py` | keep-on-facade |
| `materialize_pathology_assertions` | **none found** | `ArchiveStore.list_summaries` + `insights/pathology.py` | **deprecate** — zero consumers anywhere, including tests |
| `portfolio_bundle` | CLI | `ArchiveStore.list_summaries` + `self.repository.get_session_profiles_batch` + `insights/portfolio.py`+`postmortem.py` | keep-on-facade |
| `list_assertion_claims` | tests only (internal base for `list_assertion_claim_payloads`) | module fn `_archive_list_assertion_claims` (opens `user.db` via config directly) | keep-on-facade as plumbing; consider narrowing visibility |
| `list_assertion_claim_payloads` | MCP, daemon, internal (context) | `self.list_assertion_claims` + `surfaces/payloads.py` | keep-on-facade |
| `list_assertion_candidates` | CLI, tests | `self.list_assertion_claim_payloads` | keep-on-facade (alias) |
| `list_assertion_candidate_reviews` | CLI, tests | `user_write.py` + `surfaces/payloads.py` | keep-on-facade |
| `judge_assertion_candidate` | CLI, tests | `_archive_judge_assertion_candidate` + `surfaces/payloads.py` | keep-on-facade |
| `compile_context` | CLI, MCP, tests | `context/compiler.py` + 5 other facade methods | keep-on-facade — the paradigm case for genuine 240-line orchestration |
| `list_read_view_profiles` | MCP, tests | `archive/viewport.py` | keep-on-facade |
| `context_image_payload` | CLI, MCP, daemon, tests | `self.compile_context` + `context/compiler.py` | keep-on-facade (alias/lens) |
| `context_preamble_payload` | daemon only | `context/preamble.py` | keep-on-facade (single real consumer) |
| `explain_query_expression` | MCP, tests | `archive/query/expression.py` | keep-on-facade |
| `query_units` | devtools, tests | `ArchiveStore` + `archive/query/unit_results.py` | keep-on-facade (real DSL execution) |
| `export_otel` | tests only | `self.query_units` + `telemetry/otel_projection.py` | tests-only externally; low-priority deprecate candidate |
| `resolve_ref` | CLI, MCP, daemon, tests | `ArchiveStore` + 5 internal `_resolve_*_object_ref` helpers | keep-on-facade |
| `query_completions` | MCP, tests | `archive/query/completions.py` | keep-on-facade |
| `get_sessions` | tests, internal (api/insights) | `self.get_session` loop | keep-on-facade (alias/batch) |
| `get_actions` | **tests only** | `self.get_session` + `_actions_for_session` | **deprecate** candidate — no CLI/MCP/daemon/TUI/devtools consumer |
| `get_actions_batch` | CLI, tests | `self.get_sessions` | keep-on-facade (alias) |
| `list_sessions` | TUI, devtools, tests, internal | `ArchiveStore.list_summaries`+`read_session` | keep-on-facade |
| `list_summaries` | CLI (11 files), MCP, daemon, tests, internal (heavy) | `ArchiveStore.list_summaries` | keep-on-facade — heaviest-used method in the census |
| `list_sessions_for_spec` | CLI, devtools, tests, internal | `SessionQuerySpec.list()` + `storage/search_providers.py` | keep-on-facade |
| `search_session_hits` | devtools, internal | `archive/query/search_hits.py` + `storage/search_providers.py` | keep-on-facade (real internal fan-out feeding `search_envelope`) |
| `diagnose_query_miss` | daemon, internal | `archive/query/miss_diagnostics.py` | keep-on-facade |
| `storage_stats` | **TUI only** | `ArchiveStore.stats` | keep-on-facade (real consumer, just not CLI/MCP/daemon) |
| `search` | CLI, daemon, devtools, tests, heaviest internal fan-out in file | `ArchiveStore.search_summaries` | keep-on-facade |
| `search_envelope` | CLI, tests | `ArchiveStore.read_summary` + query/spec modules + `surfaces/payloads.py` | keep-on-facade — 110-line real orchestrator |
| `archive_count_sessions` | MCP, tests | `ArchiveStore.count_sessions` | keep-on-facade (64-line payload builder) |
| `archive_list_sessions` | MCP, tests | `ArchiveStore.list_summaries` | keep-on-facade (72-line payload builder) |
| `archive_search_sessions` | MCP, tests | `ArchiveStore.search_summaries` | keep-on-facade (70-line payload builder) |
| `archive_get_session` | MCP, tests | `ArchiveStore.read_session`/`resolve_session_id` | ArchiveStore-thin-wrapper |
| `get_session_insight_status` | internal, tests | `ArchiveStore.session_insight_status` | ArchiveStore-thin-wrapper, no external surface |
| `get_session_profile_insight` | MCP, internal, tests | `ArchiveStore.get_session_profile_insight` | ArchiveStore-thin-wrapper |
| `get_session_profile_record` | daemon, internal, tests | `ArchiveStore.get_session_profile_record` | ArchiveStore-thin-wrapper |
| `list_session_profile_insights` | internal, tests | `ArchiveStore.list_session_profile_insights` | ArchiveStore-thin-wrapper (name collides with the private adapter class's method of the same name — different impl) |
| `filter` | daemon, internal, tests | `archive/filter/filters.py::SessionFilter` + `storage/search_providers.py` | keep-on-facade — documented public Filter-Chain entry point |
| `provider_usage_report` | CLI, MCP, daemon | `storage/usage.py` | keep-on-facade |
| `stats` | CLI, MCP, devtools, internal, tests | `ArchiveStore` (multi-call) + `operations/` | keep-on-facade |
| `facets` | CLI, MCP, daemon, tests | `ArchiveStore` + `archive/query/facets.py` + `surfaces/payloads.py` | keep-on-facade — 97-line orchestrator |
| `health_check` | internal, tests | `_archive_health_report` | keep-on-facade (thin true alias) |
| `rebuild_insights` | MCP, tests (17 test call sites) | `ArchiveStore` rebuild path | keep-on-facade |
| `resume_brief` | MCP, tests | `insights/resume.py` | keep-on-facade |
| `find_resume_candidates` | CLI, MCP, internal, tests | `insights/resume.py` | keep-on-facade |
| `insight_readiness_report` | CLI, internal, tests | `ArchiveStore.insight_readiness_report` | ArchiveStore-thin-wrapper, real CLI consumer |
| `archive_debt` | MCP, daemon, tests | `operations/archive_debt.py` | keep-on-facade (documented shared payload; no literal CLI call site found by this name — verify before treating as no-CLI) |
| `insight_rigor_audit` | MCP, tests | `ArchiveStore.audit_insight_rigor` | ArchiveStore-thin-wrapper |
| `get_messages_paginated` | CLI, daemon, devtools, internal, tests | `self.repository.get_messages_paginated` (**true mixin delegation**) + `self.get_session` (material_origin branch) | **move-to-mixin:`archive/sessions.py`** for the repo-backed path; facade keeps the post-filter |
| `iter_messages` | CLI, internal | `self.repository.iter_messages` (**true mixin delegation**) + `self.get_session` (material_origin branch) | **move-to-mixin:`archive/sessions.py`** — already literally delegates |
| `bulk_get_messages` | internal, tests | `self.get_session` loop + `_archive_message_matches` | keep-on-facade; no CLI/MCP/daemon consumer — down-scope visibility candidate |
| `get_raw_artifacts_for_session` | CLI, MCP, daemon, tests | `ArchiveStore.raw_artifacts_for_session` | ArchiveStore-thin-wrapper |
| `query_sessions` | internal, tests | `archive/query/spec.py` + `ArchiveStore` | keep-on-facade; internal-only |
| `count_sessions` | CLI, MCP, daemon, devtools, tests, internal | `archive/query/spec.py` + `ArchiveStore` | keep-on-facade |
| `export_insight_bundle` | CLI, tests | `insights/export_bundles.py` + `ArchiveStore` (via `_ArchiveInsightExportOperations`) | keep-on-facade |
| `get_session_summary` | MCP, internal, tests | `ArchiveStore.read_summary`/`resolve_session_id` | ArchiveStore-thin-wrapper |
| `get_session_stats` | internal, tests | `ArchiveStore.read_summary`/`resolve_session_id` | ArchiveStore-thin-wrapper, no external surface |
| `get_stats_by` | CLI, MCP, devtools, internal, tests | `ArchiveStore.stats_by` | ArchiveStore-thin-wrapper |
| `get_index_status` | MCP, internal, tests | `ArchiveStore.index_status` | ArchiveStore-thin-wrapper |
| `update_index` | MCP, internal, tests | `ArchiveStore.rebuild_index` | ArchiveStore-thin-wrapper |
| `neighbor_candidates` | CLI, MCP, internal, tests | `archive/session/neighbor_candidates.py` + `ArchiveStore` (via `_ArchiveNeighborRuntime`) | keep-on-facade |
| `neighbor_candidate_payloads` | daemon only | `self.neighbor_candidates` + `surfaces/payloads.py` | keep-on-facade (single real consumer) |
| `session_correlation_payload` | daemon only | `self.get_session` + `insights/session_commit.py` | keep-on-facade (single real consumer, 50-line logic) |
| `get_session_tree` | MCP, internal, tests | `ArchiveStore.get_session_tree` | ArchiveStore-thin-wrapper |
| `list_tags` | MCP, internal, tests | `ArchiveStore.list_user_tags` | ArchiveStore-thin-wrapper |
| `delete_session` | CLI, internal, tests | `self.delete_session_safe` | keep-on-facade (alias, docstring says centralizes idempotency) |
| `delete_session_safe` | CLI, MCP, daemon, tests | `ArchiveStore.delete_sessions`/`resolve_session_id` + `surfaces/payloads.py` | keep-on-facade |
| `add_tag` / `remove_tag` / `bulk_tag_sessions` | CLI/MCP/internal/tests (varies) | `ArchiveStore.{add,remove}_user_tags` | ArchiveStore-thin-wrapper — **tags group** |
| `get_metadata` / `update_metadata` / `set_metadata` / `delete_metadata` | MCP/CLI/internal/tests (varies) | `ArchiveStore.*_user_metadata` | ArchiveStore-thin-wrapper — **metadata group** |
| `add_mark` / `remove_mark` / `list_marks` | CLI, MCP, daemon, tests | `ArchiveStore.{add,remove,list}_mark(s)` + `core/user_state_targets.py` | ArchiveStore-thin-wrapper — **marks group** (no mixin covers this cluster at all) |
| `save_annotation` / `get_annotation` / `list_annotations` / `delete_annotation` | CLI/MCP/daemon/tests (varies; `get_annotation` has no CLI/MCP) | `ArchiveStore.*_annotation(s)` | ArchiveStore-thin-wrapper — **annotations group** |
| `save_view` / `get_view` / `get_view_by_name` / `list_views` / `delete_view` | MCP/daemon/tests (varies) | `ArchiveStore.*_view(s)` | ArchiveStore-thin-wrapper — **views group**; `get_view_by_name` is **tests-only (1 call site)** → **deprecate** candidate |
| `create_recall_pack` / `get_recall_pack` / `list_recall_packs` / `delete_recall_pack` | MCP/daemon/tests (varies; `get_recall_pack` has no CLI/MCP) | `ArchiveStore.*recall_pack(s)` + `_build_recall_pack_payload` | ArchiveStore-thin-wrapper — **recall-pack group** |
| `save_workspace` / `get_workspace` / `list_workspaces` / `delete_workspace` | MCP/daemon/tests (varies; `get_workspace` has no CLI/MCP) | `ArchiveStore.*workspace(s)` + payload builders | keep-on-facade for `save_workspace` (real assembly logic); rest ArchiveStore-thin-wrapper — **workspace group** |
| `record_correction` / `list_corrections` / `delete_correction` / `clear_corrections` | MCP/internal/tests (no CLI/daemon) | `ArchiveStore.*correction(s)` | ArchiveStore-thin-wrapper — **corrections group** |
| `post_blackboard_note` / `list_blackboard_notes` | MCP, tests | `ArchiveStore.*blackboard_note(s)` + `archive/blackboard.py` encode/decode | keep-on-facade (real encode/decode logic, 40-65 lines each) — **blackboard group** |

### Closing tally

- **~60 keep-on-facade** — genuine orchestration or legitimate facade-to-facade
  aliasing, or single-daemon-consumer methods that are still real (not dead).
- **2 move-to-mixin** (`get_messages_paginated`, `iter_messages` → `archive/sessions.py`)
  — the only methods that literally already delegate to `self.repository.*`.
- **~35 "ArchiveStore-thin-wrapper"** (tags/metadata/marks/annotations/views/
  recall-packs/workspaces/corrections clusters) — legitimately consumed, but
  each repeats `with ArchiveStore.open_existing(...)` boilerplate for ~1 line
  of real work; the consolidation opportunity is collapsing that boilerplate,
  not deleting the methods, and there is no mixin file today that owns this
  surface (a `user.db`/`ArchiveStore`-CRUD mixin does not exist among the 10).
- **3 explicit deprecate candidates** (zero or tests-only consumers):
  - `materialize_pathology_assertions` — zero call sites anywhere, including tests.
  - `get_actions` — tests-only.
  - `get_view_by_name` — tests-only, 1 call site.
- **2 borderline / follow-up-worthy**: `export_otel` (tests-only) and
  `bulk_get_messages` (tests + one internal caller) — not as clear-cut as the
  three above but worth individual review.

### Follow-up bead proposals

1. **Facade cluster consolidation: collapse repeated `ArchiveStore.open_existing` boilerplate for the ~35 thin-wrapper methods** (tags/metadata/marks/annotations/views/recall-packs/workspaces/corrections/blackboard). Scope: a single context-manager helper or a dedicated repository mixin for `ArchiveStore`-backed user-state CRUD, cutting per-call boilerplate without changing behavior. This is the concrete, low-risk slice of the `polylogue-1fp` decomposition that this census makes obviously actionable.
2. **Delete the 3 zero/tests-only-consumer methods** (`materialize_pathology_assertions`, `get_actions`, `get_view_by_name`) plus review the 2 borderline ones (`export_otel`, `bulk_get_messages`). Small, mechanical, testmon-verifiable PR.

---

## 2. polylogue-9e5.15 — Dead-code and script-silo sweep (audit half)

### Method

- **Signal 1 (vulture)**: not installed as a project dependency; no `.vulture`
  config or `[tool.vulture]` pyproject section exists. Run via `uvx vulture`
  (v2.16, no project install needed) from repo root.
  - `--min-confidence 80` → 21 hits, all "unused variable" (not
    function/method) — no usable signal at this threshold.
  - `--min-confidence 60` → 1543 hits; filtered to function/method/class/
    property = **660 candidates** (variable/attribute hits dropped as
    near-100% false positives from unpacking/kwargs).
- **Signal 2 (coverage)**: reused the existing fresh artifact
  `.cache/coverage/coverage.json` (generated 2026-07-09T08:47 UTC, coverage.py
  7.14.1, branch data, 872 files) — not regenerated. Parsed per-file
  `functions` dict directly (already function-scoped): 8,856 named
  functions/methods with ≥1 statement, **892 (10.1%) show 0% coverage**.
- **Signal 3 (affordance usage)**: `.agent/demos/agent-affordance-usage/
  surface-inventory.csv` (generated 2026-07-05 — **4 days stale** relative to
  today; noted explicitly, treated as still roughly valid since registered
  CLI/MCP surfaces don't churn day-to-day). 34 `cli_command` + 59 `mcp_tool`
  entries classified `kill` (zero observed invocations in the captured
  window).
- **Intersection**: joined vulture hits and coverage 0%-functions on
  `(file, short_name)`; mapped each affordance-CSV `kill` entry to its
  underlying Click-command function via targeted `rg`, then checked each
  against vulture and coverage. Any ambiguous name match (common short names)
  was manually verified with `rg` call-site inspection rather than trusted
  blindly.

### Vulture false-positive classes excluded (verified individually, not assumed)

- Lark DSL transformer methods in `archive/query/expression.py` (dispatched by
  grammar rule name, e.g. `count_leaf`, `semantic_bare_leaf`).
- Pydantic `@field_validator`/`@model_validator` methods (`browser_capture/models.py`
  `coerce_provider_meta`; `verification/manifests/models.py` `_check_*` family)
  — invoked by Pydantic on instantiation, zero direct call sites, correctly excluded.
- Click commands dynamically built from `INSIGHT_REGISTRY` via
  `_build_insight_command` (`cli/commands/insights.py`) — no static per-command
  symbol exists for vulture to see at all; this whole `analyze insights *`
  cluster is structurally invisible to signal 1.
- MCP tool functions (`mcp/server_*.py`) registered via `@mcp.tool()` — the
  decorator application counts as a "reference" to vulture, so these are
  essentially never flagged regardless of real invocation count. This is why
  signals 1 and 3 barely overlap on the MCP side by construction.

### Intersected kill-list (≥2 of 3 sources — the real candidates)

**3-way (vulture + coverage + affordance-usage):**

| Symbol | Location |
|---|---|
| `self_command` | `polylogue/cli/commands/agents.py:72` |
| `current_command` | `polylogue/cli/commands/agents.py:86` |
| `conflicts_command` | `polylogue/cli/commands/agents.py:93` |
| `overlap_command` | `polylogue/cli/commands/agents.py:100` |
| `handoff_command` | `polylogue/cli/commands/agents.py:107` |
| `reject_mark_candidate_command` | `polylogue/cli/query_verbs.py:1834` |
| `defer_mark_candidate_command` | `polylogue/cli/query_verbs.py:1859` |
| `supersede_mark_candidate_command` | `polylogue/cli/query_verbs.py:1886` |

(`status_command`/`work_item_command` in the same `agents` group are
affordance-kill but test-covered, so they drop to single-source — not
arbitrary, they're the group's canonical/default subcommands, invoked via
`ctx.invoke` in tests.)

**2-way (vulture + affordance-usage; test-covered but not really invoked):**
`accept_mark_candidate_command`, `list_mark_candidates_command`,
`review_mark_candidates_command` (all `cli/query_verbs.py`); `pace_command`,
`tools_command`, `turns_command`, `usage_command` (`cli/commands/diagnostics.py`);
`script_command`, `tour_command` (`cli/commands/demo.py`).

**2-way (vulture + coverage; genuinely orphaned library code, manually verified
zero call sites, not framework-dispatched):**

| Symbol | Location | Note |
|---|---|---|
| `to_evidence_input` (×6 classes) | `polylogue/archive/semantic/models.py:57,100,134,165,202,233` | Zero call sites anywhere; not a Pydantic validator; looks like an abandoned "evidence" integration point. |
| `_matches_referenced_path`, `_matches_action_terms`, `_matches_tool_terms`, `_matches_action_sequence`, `_matches_action_text_terms`, `_sort_generic`, `_candidate_record_query_for`, `_search_limit` | `polylogue/archive/query/plan.py:187-255` (`SessionQueryPlan` methods) | Each is a 1-statement pass-through to an equivalently-named free function called directly elsewhere — 8 dead wrapper methods. |
| `_merge_tuples`, `_canonicalize_with_units`, `_source_where_unit` | `polylogue/archive/query/expression.py:2457,1622,1396` | Zero callers verified. |
| `path_only_sidecars` | `polylogue/archive/artifact_taxonomy/support.py:33` | 1-statement, 0% coverage, vulture-flagged. |
| `get_eager` (×2) | `polylogue/api/archive.py:1356`, `polylogue/storage/repository/archive/sessions.py:110` | Same-name pair, both flagged — likely a dead shared pattern. |

**Total defensible kill-list: ~30 symbols** across the verified 3-way/2-way categories.

### Single-source review list (not kill candidates — individual review only)

- **Affordance-only**: all 11 `analyze insights {cost-rollups,costs,coverage,
  debt,phases,profiles,tags,threads,timeline,tool-usage,usage-timeline,
  work-events}` — dynamically registry-generated, structurally invisible to
  vulture; would need registry-level usage data, not per-symbol, to corroborate.
- **Affordance-only**: `continue` / bare `mark` — directly `add_command`-registered
  and partially covered (80.9%/72.1%), zero real invocation per the CSV but no
  static-analysis corroboration.
- **Affordance-only**: most of the 59 `mcp_tool` kill entries — MCP wrapper
  closures are structurally invisible to vulture; a deeper cross-check would
  need to trace into each tool's underlying implementation call chain (not
  done exhaustively for all 59 — natural follow-up).
- **Vulture+coverage 2-way but affordance signal actively contradicts (do NOT
  kill)**: `space_command`, `workload_command` (`cli/commands/diagnostics.py`,
  `diagnostics space`/`diagnostics workload`) — the affordance CSV explicitly
  classifies both as `keep`, `operator_only_caveat=True` — a positive "keep"
  vote, not silence. Included here to show the intersection method surfaces
  contradictions rather than papering over them.
- ~150 of the raw vulture+coverage 2-way matches not individually itemized
  (e.g. most of `config.py`'s notification/health properties, `insights/
  archive.py` accessors, `daemon/http.py` `_handle_*` methods, various
  `storage/repository/*` accessors) — plausible but not individually
  call-site-verified in this pass; several resemble the Protocol/dataclass-variant
  pattern seen in `to_evidence_input`/`coerce_provider_meta` and deserve the
  same manual check before any deletion.

### `scripts/` verdict

`scripts/cost_accounting_demo.py` is **not orphaned** — it is actively
referenced from `README.md:380,389,391` and `docs/cost-model.md:107,113,128`
as the canonical, documented, hand-checkable reproduction of the Codex
double-billing cost-accounting fix. The **tf2.2 precedent** (closed
2026-07-03, deleted `scripts/agent_forensics.py` after folding its function
into registered `polylogue analyze` surfaces) is structurally similar but that
script had no doc mooring holding it in place — this one does.

**Recommendation: keep as-is; do not delete.** If the standing goal is
`scripts/` folding to zero regardless, the correct move is a **fold** (migrate
into a `.agent/demos/<name>/` shelf entry matching the existing convention,
update the two doc references), not a bare deletion — this is a documentation-
migration decision the dead-code evidence doesn't by itself compel.

### Follow-up bead proposals

1. **Delete the ~30-symbol intersected kill-list** (3-way + verified 2-way
   entries above), batched as one mechanical-sweep PR per the batching
   doctrine, testmon + layering/topology gates as the verification net. This
   is the "execution half" the epic's own note says must be a separate
   tracked bead from this audit.
2. **Land a devtools lane wrapping the vulture+coverage+affordance
   intersection** (per the bead's own AC #5) so this sweep is re-runnable
   yearly instead of one-off shell archaeology; land a committed vulture
   allowlist for the identified false-positive classes (Lark transformer
   dispatch, Pydantic validators, registry-built Click commands, MCP tool
   closures) next to it.

---

## 3. polylogue-9e5.5 — Exhaustive table read/write matrix

### Method

1. Extracted every `CREATE TABLE`/`CREATE VIRTUAL TABLE` name from the 5
   canonical DDL files
   (`polylogue/storage/sqlite/archive_tiers/{source,index,embeddings,user,ops}.py`)
   via regex script. Result: **58 CREATE TABLE statements, 57 unique names**
   (`otlp_spans` is genuinely defined twice — once in `source.py`, once in
   `ops.py` — see below). Tier breakdown: source.db=7, index.db=37,
   embeddings.db=3, user.db=2, ops.db=9. Up from the bead's earlier "~54" estimate.
2. For each table, ran two `rg` passes over `polylogue/` (excluding the 5 DDL
   files and `migrations/`) to classify READ (`FROM`/`JOIN <table>`) and WRITE
   (`INSERT INTO`/`UPDATE`/`DELETE FROM <table>`), bucketing `tests/` matches
   separately as test-only.
3. Verdict: **keep** (runtime read + write) / **dead** (neither, outside
   DDL/migrations/tests) / **zombie** (write, no read) / **mystery** (read, no
   write) / **test-only-wired** (only ever touched from `tests/`).

### Gotchas hit and fixed (material — change the verdicts)

1. **Glob exclusion bug.** `!*/archive_tiers/*.py` silently excluded nothing
   (a single-segment glob doesn't match the 3-segment prefix
   `storage/sqlite/archive_tiers/...` when walking from `polylogue/`).
   `index.py` embeds real `INSERT`/`DELETE` statements inside `CREATE TRIGGER`
   bodies (FTS sync triggers) — with the broken glob these leaked into
   "runtime" evidence and falsely marked `blocks_command_trigram` as "keep."
   Fixed with `!**/archive_tiers/{source,index,embeddings,user,ops}.py` +
   `!**/migrations/**`.
2. `INSERT OR REPLACE INTO` / `INSERT OR IGNORE INTO` don't match a naive
   `INSERT\s+INTO` regex — pervasive in this codebase. Missing this produced
   false zombie/mystery verdicts for `history_sidecars`, `session_events`,
   `session_agent_policies`, `session_working_dirs`, `attachment_native_ids`
   until fixed.
3. Trailing-whitespace requirement after `UPDATE <table>` broke on
   end-of-line table names (common in this multi-line SQL style) — fixed by
   relying on `\b` instead.
4. **Dynamic (f-string) table names** — a meaningful fraction of writes route
   through generic helpers where the table name is a Python variable, invisible
   to literal-text regex. Hand-traced and applied as manual overrides:
   `write.py:621-634` `_clear_session_projection_rows` (loop over `blocks`,
   `attachment_refs`, `paste_spans`, `session_events`,
   `session_provider_usage_events`, `session_agent_policies`,
   `session_working_dirs`, `session_repos`, `session_commits`,
   `session_reported_costs`, `session_model_usage`); another `write.py:3920`
   delete-by-message_id loop; `rebuild.py` `_PER_SESSION_INSIGHT_TABLES` tuple;
   `insights/session/storage.py` `build_insert_sql`/`_delete_where_in`;
   `daemon/fts_automerge.py` `_FTS_SURFACES`; `cli/commands/status.py`
   `_ARCHIVE_TIER_TABLES` → `_archive_table_counts()` (the sole reader of
   `otlp_telemetry`, see below).
5. Manually spot-checked match context for short/common-word table names
   (`repos`, `threads`, `attachments`, `gc_generations`) — all genuine SQL
   usage, no false positives.

### Full matrix (58 rows)

| Table | Tier | Read | Write | Verdict |
|---|---|---|---|---|
| raw_sessions | source.db | runtime | runtime | keep |
| blob_refs | source.db | runtime | runtime | keep |
| gc_generations | source.db | runtime | runtime | keep |
| raw_artifacts | source.db | runtime | runtime | keep |
| raw_hook_events | source.db | runtime | runtime | keep |
| otlp_spans | source.db | runtime | runtime | keep |
| history_sidecars | source.db | runtime | runtime | keep |
| fts_freshness_state | index.db | runtime | runtime | keep |
| sessions | index.db | runtime | runtime | keep |
| messages | index.db | runtime | runtime | keep |
| blocks | index.db | runtime | runtime | keep |
| web_content_constructs | index.db | runtime | runtime | keep |
| messages_fts | index.db | runtime | runtime | keep |
| blocks_command_trigram | index.db | none* | none* | **dead* (see nuance below)** |
| session_events | index.db | runtime | runtime | keep |
| session_agent_policies | index.db | runtime | runtime | keep |
| session_links | index.db | runtime | runtime | keep |
| threads | index.db | runtime | runtime | keep |
| threads_fts | index.db | runtime | runtime | keep |
| thread_sessions | index.db | runtime | runtime | keep |
| session_working_dirs | index.db | runtime | runtime | keep |
| repos | index.db | runtime | runtime | keep |
| session_repos | index.db | runtime | runtime | keep |
| session_commits | index.db | none | runtime | **zombie** |
| attachments | index.db | runtime | runtime | keep |
| attachment_refs | index.db | runtime | runtime | keep |
| attachment_native_ids | index.db | runtime | runtime | keep |
| paste_spans | index.db | runtime | runtime | keep |
| price_catalogs | index.db | runtime | runtime | keep |
| model_prices | index.db | none | runtime | **zombie** |
| session_reported_costs | index.db | none | runtime | **zombie** |
| session_model_usage | index.db | runtime | runtime | keep |
| session_provider_usage_events | index.db | runtime | runtime | keep |
| session_tags | index.db | runtime | runtime | keep |
| insight_materialization | index.db | runtime | runtime | keep |
| session_work_events | index.db | runtime | runtime | keep |
| session_work_events_fts | index.db | runtime | runtime | keep |
| session_phases | index.db | runtime | runtime | keep |
| session_latency_profiles | index.db | runtime | runtime | keep |
| session_profiles | index.db | runtime | runtime | keep |
| session_tag_rollups | index.db | runtime | runtime | keep |
| session_runs | index.db | runtime | runtime | keep |
| session_observed_events | index.db | runtime | runtime | keep |
| session_context_snapshots | index.db | runtime | runtime | keep |
| message_embeddings | embeddings.db | runtime | runtime | keep |
| message_embeddings_meta | embeddings.db | runtime | runtime | keep |
| embedding_status | embeddings.db | runtime | runtime | keep |
| assertions | user.db | runtime | runtime | keep |
| user_settings | user.db | none | none | **dead** |
| ingest_cursor | ops.db | runtime | runtime | keep |
| ingest_attempts | ops.db | runtime | runtime | keep |
| convergence_debt | ops.db | runtime | runtime | keep |
| cursor_lag_samples | ops.db | runtime | runtime | keep |
| daemon_stage_events | ops.db | runtime | runtime | keep |
| daemon_events | ops.db | runtime | runtime | keep |
| embedding_catchup_runs | ops.db | runtime | runtime | keep |
| otlp_spans (2nd def.) | ops.db | runtime | runtime | keep |
| otlp_telemetry | ops.db | runtime (diagnostic-only) | runtime | keep, but thin (see below) |

**Totals: 53 keep (1 of which is diagnostic-only-thin), 3 zombie, 2 dead (incl. 1 with a trigger-write nuance), 0 mystery, 0 test-only-wired.**

### Dead / zombie / thin tables — detail

- **`user_settings`** (user.db) — **dead**, reproduces `polylogue-at44` exactly:
  zero Python reference anywhere outside DDL/migration except one unrelated
  filename string (`"user_settings.pb"` in
  `polylogue/archive/artifact_taxonomy/runtime.py:55`).
- **`model_prices`** (index.db) — **zombie**. Written by `pricing_seed.py:85`
  (seeded from the curated `PRICING`/LiteLLM catalog), never read back —
  cost computation resolves per-model prices from the in-process Python
  catalog instead of round-tripping through this table. Sibling
  `price_catalogs` genuinely is read (`write.py:3254`).
- **`session_reported_costs`** (index.db) — **zombie**. Written via
  `write.py:3185` and cleared via the dynamic-DELETE list, never read.
- **`session_commits`** (index.db) — **zombie**. Written via `write.py:3372`
  plus dynamic DELETE. `polylogue/insights/session_commit.py:454-464` has a
  `persist_session_commits()` function whose own docstring says
  *"placeholder... the inline write in the MCP tool is used for the initial
  implementation"* with a no-op body (`del edges, repo_id`) — an explicit
  in-code admission this is unfinished. The MCP tool `correlate_session(s)`
  computes commit-correlation live from git history each call and never reads
  the persisted table.
- **`blocks_command_trigram`** (index.db) — flagged dead by the strict
  Python-code definition, but with a real caveat: it's an FTS5 external-content
  trigram index over `blocks.tool_detail_text`, kept live by three native
  SQLite triggers (`index.py:375-403`) that fire on every INSERT/UPDATE/DELETE
  on `blocks` — so it carries an ongoing write-amplification cost via
  DB-engine mechanics, even though zero Python code ever queries it. The DDL's
  own comment block documents a benchmarked 900x+ speedup (26s vs 0.15s) for
  the intended query shape — nobody built that query. This is "zombie via
  trigger, dead at the application layer," a distinct flavor worth its own
  follow-up: either wire the trigram-accelerated path into the query DSL, or
  drop the triggers and stop paying the write cost.
- **`otlp_telemetry`** (ops.db) — technically alive (not the prior claim of
  "0 readers" — see below) but the single reader is `polylogue status`'s
  generic `_archive_table_counts()` diagnostic row-count; nothing ever reads
  the actual `payload`/`signal_type` columns back out. Classified "keep" but
  flagged as thin/diagnostic-only, not a substantive consumer.

### Verification of the bead's prior claims

- **"`otlp_telemetry` has 0 readers"** — does **not** currently hold; it has
  exactly one reader (the generic per-tier row-count diagnostic in `polylogue
  status`), which is a shallow, non-substantive read. The bead's earlier
  spot-check likely predates that status-command wiring, or didn't count a
  dynamic/generic reader. Treat as "alive but barely," not a clean dead-table.
- **"4 tables with exactly 1 reference"** — doesn't reproduce as stated
  against the current tree (the exact count is sensitive to the DDL/migration
  self-mention exclusion bug described above). The closest fresh equivalent:
  **5 tables with minimal/zero live wiring today** — the 3 zombies
  (`model_prices`, `session_reported_costs`, `session_commits`, each exactly
  one non-DDL write site, zero reads), `user_settings` (zero total), and
  `blocks_command_trigram` (zero Python-level references, all activity is
  DDL-embedded triggers) — 4 of 5 concentrated in index.db, 1 in user.db, none
  in source.db/embeddings.db/ops.db. Supersede the stale "4 tables, exactly 1
  reference" framing with this set of 5.
- **Calibration check**: `user_settings` reproduces cleanly as dead, matching
  `polylogue-at44`'s description exactly — confirms the classification method
  is sound before trusting the new findings above.

### Follow-up bead proposals

1. **Drop `model_prices` and `session_reported_costs`** in the next index.db
   schema-bump window (per-tier migration batching, `polylogue-60i5`) — both
   are pure write-cost with zero consumers and no in-code acknowledgment of
   future use, unlike `session_commits`.
2. **Resolve `session_commits`/`blocks_command_trigram` as design decisions,
   not schema drops**: either wire `persist_session_commits()`'s stub into a
   real reader (replacing the MCP tool's live git-recompute path) or remove
   the table; either wire the trigram index into the query DSL's LIKE/search
   path or drop its triggers. Both carry an explicit "someone started this and
   didn't finish" signature worth a design decision, not a silent drop.

---

## Cross-cutting notes

- Bead 9e5.14's finding that most of `PolylogueArchiveMixin` bypasses the
  10-mixin `SessionRepository` in favor of direct `ArchiveStore` access is a
  structural precondition for bead 9e5.5's table matrix: `ArchiveStore` (in
  `storage/sqlite/archive_tiers/archive.py`) is the actual point of contact
  with most tables, which is why the read/write census in section 3 had to
  trace through `write.py`/`user_write.py`/etc. rather than the mixin files
  alone.
- All three sections independently re-confirmed the `polylogue-at44`
  (`user_settings`) precedent as a calibration check before trusting new
  findings — the method is validated against a known-true result in each case.
- No product code, schema, or bead state was modified to produce this
  artifact. Scratch scripts used for the table-matrix extraction are at
  `/tmp/claude-1000/-realm-project-polylogue/3b0038ec-234c-44b8-bb88-fe222b44fe0f/scratchpad/{extract_tables.py,classify_tables_v2.py,results_v2.json,final_matrix.txt}`
  (session-scoped, not committed — the bead's AC #5 for a re-runnable devtools
  lane is a separate follow-up, not satisfied by this scratch copy).
