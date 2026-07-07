---
created: "2026-06-29T14:24:21+02:00"
purpose: "Capture dev-loop findings about insight construct validity and simplification targets"
status: "active"
project: "polylogue"
---

# Insights Construct Validity

## Context

During the devloop, Hermes forensics suggested a repeated-tool/stall-style
detector. A first implementation attempt added `repeated_tool` as another
`PathologyKind`, then ran it against `/realm/tmp/polylogue-dev/archive`.

## Findings

- Naive consecutive identical `tool_finished` detection produced 613
  `repeated_tool` findings in the first 500 dev archive sessions. Samples were
  mostly normal repeated read/edit actions on the same file.
- Restricting to failed/error/timeout summaries dropped that to 18/500 and
  surfaced plausible failed read/edit loops against stale worktree paths, but
  it still had not proved enough utility to deserve product vocabulary.
- The right direction is not another named pathology. First extract/report
  neutral, evidence-backed patterns and validate whether a report is useful.
  Only then promote a semantic label.
- Existing `session_phases.kind` was a clearer correctness bug: storage already
  ignored the field because phases are time-gap intervals, but the CLI/API/MCP
  registry/docs still exposed the dead kind filter and rendered
  `inference.kind`.

## Outcome

- Do not commit the unvalidated `repeated_tool` pathology detector.
- Commit `fix(insights): drop dead phase kind surface` instead: it removes the
  unsupported phase-kind query/CLI/doc surface and renders phase index instead.

## Next Targets

- Classify every registered insight as evidence, projection, inference,
  enrichment, assertion, or aggregate. Anything that mixes these needs a reason
  or a simplification patch.
- For materialized tables, ask whether the table saves real cost or only
  preserves old architecture. If recomputation is cheap and exact, prefer
  on-demand composition.
- Audit `session_profiles`: it currently mixes evidence, inference,
  enrichment, and query-denormalization.
- Audit `pathology`: `wasted_loop` may be construct-invalid if it only means
  normal failed-test iteration. Reports must demonstrate usefulness before
  headline fields or candidate assertions lean on it.

## 2026-06-29 Follow-up

- The existing `wasted_loop` pathology became useful only after two fixes:
  API dogfooding had to respect an explicit dev archive root instead of
  silently following the ambient XDG `db_path`, and the detector had to count
  consecutive structured failure streaks instead of every failure in a run.
- Dev archive smoke after the fix: `pathology_report(limit=500)` over
  `/realm/tmp/polylogue-dev/archive` analyzed 500 of 4,302 matched sessions and
  returned 65 `wasted_loop` findings. The top finding was a genuine 94-event
  failure streak in an isolated Claude Code worktree, mostly failed shell/read
  attempts around missing temp files and stash/repro commands.
- The failed `repeated_tool` experiment remains invalid. Repetition alone is
  normal agent work; structured outcome streaks are the minimum viable evidence
  for a failure-loop construct.

## 2026-06-29 Structured Event Tightening

- The first useful `wasted_loop` fix still had a smell: it excluded cancelled
  Read/Edit/etc. failures by matching `ObservedEvent.summary` prefixes. That
  preserved the report's behavior but kept the construct dependent on prose.
- Better primitive: carry tool execution metadata (`tool_name`, `tool_id`,
  `command`, `handler_kind`, `status`) from `RecoveryEvent` into
  `ObservedEvent`. This uses evidence already extracted from structured tool
  blocks and requires no storage DDL bump because materialized observed events
  store the full model in `payload_json`.
- The detector now treats `test_failed` as diagnostic and treats
  `command_failed` as diagnostic only when `handler_kind` is command-like
  (`shell`, `git`, `github`, `test`) or when a legacy payload has a concrete
  command. Bare generic/file-read failures no longer count by summary string.
- Methodological rule: pathology detectors must depend on typed predicates
  from the run projection. If a detector needs string-prefix deny/allow lists
  over summaries, the projection is missing a primitive or the detector is not
  ready to be product vocabulary.

## 2026-06-29 Phase Contract Tightening

- Dev archive evidence (`/realm/tmp/polylogue-dev/archive/index.db`): 7,463
  `session_phases` rows; every row had `inference_json.confidence == 0.0` and
  `support_level == weak`; 4,310 rows had `fallback_inference == true`.
- The public rigor matrix still claimed phase-kind classification, but the
  phase payload has no label/kind/summary. The useful part is the evidence
  interval: message range, timing provenance, word count, tool counts.
- Change direction: expose `SessionPhaseInsight` from storage as
  `semantic_tier="evidence"` with no inference payload. Keep the storage
  columns for compatibility until a schema cleanup; consumers that still receive
  legacy/in-memory phase inference tolerate it, but product docs no longer
  advertise it as semantic classification.

## 2026-06-29 Phase Cleanup Follow-through

- Commit `80e9d7bd2` made `ArchiveStore`'s direct phase insight reader match the
  record path: phase insights now surface evidence payloads only. The
  compatibility `session_phases.inference_json` column still exists, but the
  public `SessionPhaseInsight` no longer hydrates it as
  `inference`/`inference_provenance`.
- Retrieval-band and derived-status accounting no longer count phase rows as
  inference-retrieval support. Phase rows remain phase readiness/evidence
  artifacts; work-event/profile inference rows remain inference retrieval.
- Commit `ef3ea94d0` removed avoidable `harness="unknown"` in run projection
  for local agent origins. `gemini-cli-session`, `hermes-session`, and
  `antigravity-session` map to existing `RunHarness="local"` while preserving
  exact `provider_origin`.
- Commit `baac72b14` stopped profile support from using legacy phase confidence.
  Phase presence/duration can still support evidence and engaged-duration
  calculations, but phase confidence cannot upgrade profile inference support.
  `ALL_PHASES_HEURISTIC` remains only as a historical enum value so old payloads
  validate; new profile inference payloads should not emit it.

## Remaining Construct-Validity Targets

- `session_profiles` still mixes evidence, inference, enrichment, aggregate
  counters, and query denormalization. Continue decomposing by public contract
  before attempting schema removal.
- Several compatibility names still contain `phase_inference_*`. Do not rename
  storage/status keys casually; treat it as a schema/status contract cleanup with
  a fresh rebuild plan and migration notes.
- Audit `workflow_shape` and `terminal_state`: if the classifier emits
  `"unknown"` where a typed predicate already exists, fix the predicate; if not,
  make the support/evidence level explicit rather than pretending the field is
  informative.

## 2026-06-29 Terminal-State Boundary Fix

- Commit `99033d652` fixed a construct-validity bug in `terminal_state`.
  Previously any historical tool/action output containing an error marker could
  force `error_left`, even when the session later had a clean assistant finish.
- The corrected predicate keeps pending tools as the strongest boundary signal,
  then reports tool/event error evidence only when there is no later meaningful
  text boundary or when the final assistant message is itself error-like.
- This preserves `terminal_state` as a final observable archive-shape signal,
  not a summary of whether any error ever occurred during the session. Historical
  failures belong in run/observed-event/pathology reports, not the terminal
  boundary label.

## 2026-06-29 Low-Tool Dialogue Shape Fix

- `workflow_shape` still used `unknown` for long sessions with real
  user/assistant dialogue but no tools because `chat` was capped at eight
  messages. That was defensive rather than evidence-based: the observed shape
  is low-tool dialogue, not unknowable.
- The classifier now emits `chat` for any session with user or assistant text
  and negligible tool density, with slightly lower confidence for sustained
  dialogue than for short chats. `unknown` remains available for genuinely
  non-informative records such as empty/protocol-only sessions.

## 2026-06-29 Phase Derived-Model Name Cleanup

- Public readiness/maintenance derived-model maps still used
  `session_phase_inference` even though the product contract is now
  `session_phases` as deterministic interval evidence.
- The safe cleanup is key-level, not storage-level: keep legacy
  `phase_inference_*` snapshot fields until a schema/status cleanup, but expose
  the derived model as `session_phases` in the derived status map, readiness
  checks, repair aggregation, and maintenance preview model inventory.

## 2026-06-29 Goal Outcome Mapping Fix

- Goal-session enrichment (`/goal ...`) still mapped obsolete terminal-state
  names (`completed`, `error`, `timed_out`, `stuck`) even though profiles now
  emit boundary labels such as `clean_finish`, `error_left`, `question_left`,
  and `tool_left`.
- This made `goal_outcome` silently disappear for sessions with clear terminal
  evidence. The mapping now treats clean finish as completed, trailing error as
  failed, unanswered/pending boundaries as abandoned, and the reserved
  `agent_hanging` boundary as timed out.
- Follow-through: `goal_text` and `goal_outcome` now participate in enrichment
  search text and the profile-insights docs list them with the rest of the
  merged-tier enrichment fields. Otherwise the field would be computed but
  awkward to discover through profile search.

## 2026-06-29 Tool-Left Overcount Fix

- Dev archive evidence (`/realm/tmp/polylogue-dev/archive/index.db`) showed
  3,406 / 4,302 profiles marked `terminal_state="tool_left"`. Sampling showed
  Claude Code sessions with paired `tool_use` / `tool_result` blocks and final
  assistant text still got `pending_tool_count` equal to roughly half their
  historical actions.
- Root cause: `terminal_state` treated every semantic action with no inline
  `output_text` as pending. That is invalid for providers whose results are
  separate messages/blocks rather than embedded on the action.
- Fix direction: pending tool state should come from explicit provider event
  pairing and block-level `tool_use`/`tool_result` pairing, not from historical
  action rows missing inline output. A direct block-id estimate over the dev
  archive found only 116 of the 3,406 stored `tool_left` profiles had unpaired
  identified tool-use ids; 3,290 had no unpaired identified block id.

## 2026-06-29 Work-Event Public Name Cleanup

- Public derived-status, readiness, repair, and maintenance-preview inventories still exposed `session_work_event_inference` / `session_work_event_inference_fts`, mirroring the older phase naming problem. That label overstates the construct: the table is the materialized work-event row model, with inference payloads inside rows where needed.
- Cleanup kept internal runtime/dataclass counters stable but renamed public derived model keys to `session_work_events` and `session_work_events_fts`. Legacy keys are now intentionally absent from repair aggregation, matching the phase cleanup: callers should use the actual derived table/model identity rather than the inference mechanism.
- Stale refresh test also expected `terminal_state` inside `evidence_payload_json`; corrected it to assert the profile column and absence from evidence payload, preserving the evidence/inference/enrichment split.

## 2026-06-29 Direct Status Readiness Silo Cleanup

- Direct archive status had its own work/phase/thread readiness logic that only checked missing `insight_materialization` rows. That could report ready when materialization existed but row counts were stale, orphaned, or mismatched.
- The direct status path now feeds from `session_insight_status_sync(..., verify_freshness=True)` and exposes expected/stale/orphan/mismatch evidence for work events, phases, threads, and profile rows.
- While wiring this, canonical status itself had a latent schema mismatch: stale work/phase queries read `materializer_version`/`source_sort_key` from `session_work_events` and `session_phases`, but those tables do not have those columns in the live dev archive. Staleness belongs to the per-session `insight_materialization` ledger; row tables only support row counts and orphan checks.

## 2026-06-29 Archive Facade Readiness Delegation

- `ArchiveStore.session_insight_status()` was another hand-rolled readiness implementation. It counted `session_tags` as `tag_rollup_count`, set `expected_tag_rollup_count = tag_rows`, and treated tag rollups as ready when any session tag existed.
- The facade now delegates to the same canonical `session_insight_status_sync()` used by status and repair paths. This removes a silo and makes API/CLI/readiness agree on `session_tag_rollups`, work/phase freshness, thread freshness, and materializer-version checks.

## 2026-06-29 Blocker Enrichment Boundary

- Raw blocker-text extraction was treating any authored user text with words like "failed", "cannot", or "traceback" as a blocker. That is valid as a text band, but not as an enrichment/resume claim: an initial problem statement should not become "Resolve blocker" after the session ended cleanly.
- `SessionEnrichmentPayload.blockers` is now gated by terminal boundary posture. It is populated only for unresolved endings (`error_left`, `question_left`, `tool_left`, `agent_hanging`). Clean sessions still retain the initial problem as intent/outcome context, not as an unresolved blocker.

## 2026-06-29 Aggregate Origin Breakdown Cleanup

- Tag rollups, archive coverage, and thread payloads still exposed aggregate maps as `provider_breakdown` even though the values came from source/session origin evidence. Worse, the SQLite coverage/tag helpers converted `sessions.origin` back through `Provider`, producing values like `CODEX` under what should be source-origin semantics.
- Cleanup direction: canonical serialized/read-model fields are `origin_breakdown` / `origins`; old `provider_breakdown` / `providers` are accepted as legacy input or exposed only as compatibility properties. New aggregate writers now preserve origin tokens such as `codex-session` and `claude-code-session` in nested breakdown maps.
- Rigor audit fields now check `origin_breakdown`, and registry projection keeps legacy `provider_breakdown` input mapped to origin payloads while passing canonical `origin_breakdown` through unchanged.
- Verified with focused API/MCP/registry/rigor tests and `devtools verify --quick`.

## 2026-06-29 Wasted-Loop Predicate Tightening

- The `wasted_loop` pathology detector was still too broad: any sequence of diagnostic failures without an intervening structured success counted, even when the failures were different commands/checks. That reports normal evidence-first debugging as pathology.
- The predicate now requires repeated failures of the same normalized diagnostic signature (command when present, else tool/handler signature). Mixed `pytest` / `mypy` / `devtools verify` failures no longer imply a stuck loop by themselves.
- Detector version bumped to 4 because cached/report comparisons need to distinguish the narrower rule.

## 2026-06-29 Archive Readiness Mismatch

- `polylogue config paths --format json` reported `archive_ready=true` for the live archive while `polylogued run` refused the watcher with `index.db:11!=18`. The command was only checking file presence, not tier schema versions.
- This directly explains a confusing devloop failure: raw Codex JSONLs parsed cleanly with `polylogue import --explain`, but `polylogue find id:...` missed them and a temporary daemon could not materialize imports because the active index tier is stale for the running checkout.
- Fix direction: path/config readiness must expose physical layout and schema readiness separately. `final_shape_ready` remains physical; `archive_schema_ready` and per-tier `archive_tier_versions` carry user_version truth; `archive_ready` is false when schema mismatches.

## 2026-06-29 Workload Probe Schema Gate

- User correctly challenged the idea of adding a permanent raw-source markdown
  preview/export path: manual import/preview is an emergency workaround, while
  the real product invariant is seamless daemon convergence into the archive.
- Live evidence: both requested Codex JSONL files exist under
  `/home/sinity/.codex/sessions/2026/06/29/`, direct parser explain accepts
  them, but neither `source.db.raw_sessions`, `index.db.sessions`, nor
  `ops.db.ingest_cursor` has rows for their IDs/paths. `polylogued` cannot
  converge them because schema preflight blocks the live watcher on
  `index.db:11!=18`.
- `ops diagnostics workload` already reported `schema_mismatch:index`, but
  still computed derived readiness from stale index tables and showed green
  surface subclaims. That is construct-invalid: current-code readiness cannot
  be derived from a tier whose schema contract is known wrong.
- Fix direction: when source/index schema mismatches, derived readiness is
  `checked=false` with reason `schema_mismatch:<tiers>` and no surface
  readiness map. When user/index schema mismatches, user-overlay orphan checks
  are similarly unchecked. This makes the report point at the true convergence
  blocker instead of mixing stale health claims into a red archive preflight.
- Separate cleanup target: `polylogue ops reset --database` currently deletes
  `source.db` along with `index.db`, `embeddings.db`, and `ops.db`, while
  archive docs/plan classify `source.db` as durable evidence and `index.db` as
  rebuildable. The safer operator flow for this incident is index-tier rebuild,
  not broad database reset that drops raw evidence.

## 2026-06-29 Index-Only Reset Surface

- Follow-up to the archive mismatch: add `polylogue ops reset --index` as the
  command-level expression of the documented "move the mismatched index tier
  aside" recovery path.
- Scope is deliberately narrow: it deletes only `index.db`, `index.db-wal`, and
  `index.db-shm`. It preserves `source.db`, `embeddings.db`, `ops.db`, and
  `user.db`. `--database` is left as the existing broad reset until that
  behavior can be redesigned without surprising existing users.
- Docs now use `polylogue ops reset --index && polylogued run` for index-tier
  schema bumps, matching the durability model where `source.db` remains the raw
  evidence authority.

## 2026-06-29 Daemon Status Schema Gate

- After fixing `config paths` and workload diagnostics, daemon status still
  computed `archive_ready` from source/index file presence. That would report a
  complete-but-stale archive as ready through `polylogued status` and component
  readiness.
- Fix direction: daemon status now carries per-tier expected/current
  `user_version`, `version_status`, `archive_schema_ready`, and
  `schema_mismatches`. Physical `active_store=archive_file_set` remains true
  when source/index files are present, but `archive_ready=false` and component
  readiness is `blocked` when schema mismatches exist.
- Metrics received the same semantic split: `polylogue_archive_active_store`
  remains physical, while `polylogue_archive_storage_ready{state="archive_runtime"}`
  is gated by layout blockers including schema mismatch.

## 2026-06-29 Route-Readiness Naming

- `polylogue status` also exposes a static catalog report under
  `archive_runtime_paths`. Its `archive_runtime_ready` flag never meant live
  archive convergence; it meant "primary code paths route to split-tier archive
  APIs." That is a valid engineering signal, but the name collides with actual
  archive runtime/schema readiness and invites exactly the wrong operator
  inference during a stale-index incident.
- Fix direction: introduce the preferred `archive_routing_ready` field and
  render the human label as "Archive routing paths". Keep
  `archive_runtime_ready` as a compatibility alias for existing consumers, with
  comments making the predicate explicit.

## 2026-06-29 Import Guidance and Convergence

- `polylogue import PATH` is not a direct manual ingestion path; it stages into
  the daemon inbox and asks the daemon to schedule the file. That shape is
  acceptable because it preserves the convergence invariant, but the success
  text must point the operator at convergence/readiness evidence rather than a
  generic analysis command.
- Fix direction: accepted import output now points to live daemon progress,
  `polylogued status`, and `polylogue status --full`. This keeps the operator
  loop as "get daemon/archive convergence working" instead of "try another
  ad-hoc import/read command."

## 2026-06-29 Codex Usage Alias Harmonization

- Codex `token_count` events already accepted provider aliases such as
  `cached_input_tokens`, `cached_tokens`, and camelCase token count fields, but
  per-message usage parsing accepted a narrower set. That made source evidence
  survive in `session_provider_usage_events` while the same vocabulary could be
  dropped from `messages.cache_read_tokens` / input-output columns.
- Fix direction: `_token_usage` now accepts the same relevant Codex aliases for
  per-message rows where they map cleanly to message columns. This is a
  correctness-preserving harmonization, not a fabricated estimate.

## 2026-06-29 Read Probes Must Not Bootstrap

- Readiness and reader routes had two side-effect/readiness conflations:
  `ArchiveStore.open_existing(read_only=True)` bootstrapped missing tiers, and
  `_open_readiness_probe_connection` explicitly initialized an archive before
  opening a read probe. That can turn "archive absent" into "empty archive is
  reachable" from a status/read path.
- Fix direction: read-only archive opens never bootstrap; readiness probes open
  strictly read-only and report unavailable when the DB is absent. The web
  reader route gate now requires current source/index tier versions, so a stale
  `index.db` file does not enable archive reader routes.

## 2026-06-29 Timeline Insight Query Contract

- Goodall found that async timeline insight readers still queried legacy column
  names (`event_index`, `heuristic_label`, `start_time`, `materialized_at`,
  `phase_index`) while the canonical DDL stores compact deterministic columns
  (`position`, `work_event_type`, `started_at_ms`, `ended_at_ms`) and keeps
  materialization metadata in `insight_materialization`.
- Fix direction: repair the read side as a projection from canonical DDL into
  the current runtime record model. This avoids widening schema around stale
  names while preserving API compatibility for callers that consume
  `SessionWorkEventRecord` / `SessionPhaseRecord`.
- Residual design debt: phase records still carry inference-shaped payload
  fields even though the rigor contract treats phases as deterministic timing
  evidence. That is a follow-up construct-validity cleanup, not part of the
  reader-contract repair.
