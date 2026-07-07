---
created: "2026-06-29T19:01:02+02:00"
purpose: "Organize the recovery/legacy-flag cleanup, audit findings, and demo-oriented workload"
status: "active"
project: "polylogue"
---

# Recovery, DSL, And Demo Workload

## Current Direction

The operator intent is decisive cleanup, not compatibility accumulation:

- Delete public recovery-as-view/tool/route surfaces.
- Keep disaster/repair recovery terminology only for storage, backup, WAL,
  daemon interruption, blob restoration, and corruption repair.
- Move handoff/successor-context behavior to `ContextImage`, query units, read
  views, and explicit query recipes.
- Treat weird global flags as legacy surface area. DSL/query descriptors should
  own selection; read views should own their own presentation options.
- Produce useful demos from the same primitives rather than separate forensics
  silos.

## Active Slice

Implemented in the dirty tree so far:

- Removed public `read --view recovery`, `read --report`, HTTP
  `/api/sessions/:id/recovery`, public `RecoveryReadPayload`, and MCP
  `get_recovery_report` / `get_recovery_work_packet`.
- Moved `continue` toward `ContextSpec(read_views=("messages",),
  unit_queries=(runs, observed-events, context-snapshots, actions))`.
- Added `ContextSpec.unit_queries` and query-unit context segments.
- Removed public root/read/MCP message projection flags from the edited
  surfaces (`message_role`, `material_origin`, `message_type`,
  `no_tool_outputs`, `no_tool_calls`, `no_code_blocks`, `no_file_reads`,
  `prose_only`, `dialogue_only`).
- Removed hidden `compile_context` support for `read_views=("recovery",
  "work-packet")`; those view names now produce unsupported omissions instead
  of invoking recovery transforms.

## Live Archive / Daemon Baseline

2026-06-30 correction: do not use the old 16K-session / 5.7M-message count as
the current archive baseline. That was a pre-dedup/stale-index shape and counts
duplication that fork/continuation dedup resolved.

Current active archive at `/home/sinity/.local/share/polylogue`, probed
2026-06-30 03:05 CEST:

- `source.db.raw_sessions`: 2,397 rows, latest acquisition
  2026-06-30T03:04:58.856+02:00.
- `index.db.sessions`: 2,390 rows, latest indexed update
  2026-06-30T02:55:55.095+02:00.
- `index.db.messages`: 159,956 rows.

Runtime state at that probe:

- Installed/prod `polylogued.service`: inactive; stale against current schema.
- Repo-local dev daemon `polylogued-devloop.service`: active, PID 2829052,
  running `.venv/bin/polylogued run` from `/realm/project/polylogue`.
- The daemon is still doing source-root catch-up/reconciliation; its large file
  count should be read as watcher/cursor reconciliation work, not as proof that
  the active archive is the obsolete 16K/5.7M corpus.

Verification status:

- `python -m compileall` passed after the main edits.
- `devtools render all` passed before the latest test fixes.
- A broad focused pytest lane was killed externally because of RAM pressure.
  Treat that as interrupted verification, not code failure. Use narrow nodes
  until the host is calmer.
- After batching fixes, these focused checks passed:
  - MCP surfaces/contracts/discovery/envelopes: `234 passed`.
  - CLI/read/completion/snapshots plus route/OpenAPI/view-profile batches:
    `399 passed`, `247 passed`, `276 passed` across the narrow serialized
    lanes.
  - `devtools render all --check` passed after rendering pages.
  - `compile_context` smoke after hidden compatibility removal: `3 passed`.

## Subagent Findings To Preserve

### Issue Mining

High-leverage issue clusters:

- #2006 / #2177 / #1844 / #2317: query DSL and contract-owned surfaces should
  replace flags and parallel dispatch.
- #1883 / #2384 / #2459: recovery as public noun should disappear; the useful
  evidence/handoff behavior belongs under context, assertions, and work-event
  substrates.
- #2467 / #2471 / #2472 / #2476 / #2478: lineage and effective context are the
  correct answer to "what did the model see?", not projection toggles.
- #2246 / #2473: schema/assertion/aggregate correctness must be fixed before
  trusting demos and MCP aggregate output.
- #2316 / #2480 / #2481 / #2460 / #2461: provider/Codex/Claude/run/cost
  harmonization remains critical for honest analytics.
- #2482 / #2196 / #2304: topic/time/lineage demos are valuable once they are
  built on query/context primitives, not a new silo.

### Recovery Audit

Delete now:

- README/docs that advertise removed recovery MCP tools.
- Generated product workflow ids/docs that still say `find-then-recovery` or
  `read --view recovery`.
- Topology entry for deleted `polylogue/cli/read_views/recovery.py`.
- CLI special-case `_effective_read_output_format(... view == "recovery")`.
- Tests/completions still expecting recovery as read view.
- MCP test mocks for removed recovery tools.

Rename/design:

- Public API `recovery_*` methods and context compiler recovery-centered names.
- Internal `insights/transforms.py` recovery digest/report/work-packet names.
- Run projection parameter names like `recovery_events`.
- Artifact/operation/benchmark catalogs around `recovery_digest`.

Keep as real recovery:

- Blob restore/planning, daemon interruption, FTS startup repair, backup/WAL,
  corruption, and maintenance docs/tests.

### Insight Construct Validity

Priority bugs:

- `session_runs` conflates run identity with projection membership. Fix by
  keying `(session_id, run_ref)` or splitting run nodes from memberships.
- Run projection is omitted from incremental refresh/delete, causing stale
  query-unit rows.
- `session_phases` claims evidence-only semantics but still carries
  inference/confidence payloads.
- `SessionEvidencePayload` stores classifier-shaped support data.
- Bespoke run-projection read facades duplicate query units.

### Provider Harmonization

Priority bugs:

- Content hash excludes stored semantic/source fields, so parser fixes for
  model/token/outcome/material-origin can be skipped.
- Codex marks runtime-root user messages as human-authored without positive
  evidence.
- Local-agent `total_tokens` is mapped to output tokens.
- Gemini token mapping and branch metadata are inconsistent/dropped.
- Structured execution outcomes are only partially normalized.

## Prioritized Next Work

1. Stabilize the current public cleanup with narrow tests and regenerated docs.
2. Remove remaining recovery public/docs/test traces from README, product
   workflows, topology, CLI completion, and MCP test fixtures.
3. Remove/rename `compile_context` support for `read_views={"recovery",
   "work-packet"}` so hidden compatibility does not survive.
4. Continue legacy flag deletion inward: `QueryOutputSpec`, archive query
   projection helpers, CLI machine-error suggestions, docs/search/export.
5. Fix query-unit ref extraction and unit-query-only context recipes enough to
   make context packs a credible replacement.
6. Fix #2473 aggregate capping before using MCP aggregate output as demo proof.
7. Produce demos under `/realm/inbox/demos_polylogue`:
   - current devloop temporal timeline using query units and run projection;
   - context-image successor handoff from query-unit recipe;
   - honest provider-field audit report showing source-backed vs inferred
     fields;
   - topic/time/lineage report once it can use query/context primitives.

## Demo Artifacts Produced

`/realm/inbox/demos_polylogue` now has:

- `01-real-archive-temporal-devloops/`: private aggregate CSVs over
  `session_runs`, `session_observed_events`, and
  `session_context_snapshots`, plus cardinality JSON and a README with caveats.
- `02-query-unit-surface/`: captured public CLI JSON examples for `runs`,
  `observed-events`, and `context-snapshots` query-unit expressions.
- `04-synthetic-demo-shape/`: reproducible `polylogue demo seed` archive,
  verify output, and a query-unit example over the synthetic archive.

Observed demo lesson: query-unit rows are usable and expressive, but live
observed-event/context-snapshot probes over the large archive are slow enough
to justify a query-unit performance follow-up before making them the headline
operator workflow.

## Demo Guardrails

Demos should show capability without lying:

- Use query units and context images as the primary algebra.
- Label known construct-validity gaps explicitly.
- Avoid bespoke reports unless they are only artifacts; permanent behavior
  should land in query/context/insight primitives.
- Prefer outputs that can be regenerated from commands and copied to
  `/realm/inbox/demos_polylogue`.

## 2026-06-29 19:53 Slice: Root CLI Projection Backdoors

User reported another agent killed a broad test run because of RAM pressure.
Treat that as host pressure, not a failing product signal. Verification should
stay serialized and focused until the working set shrinks.

Implemented slice:

- `QueryOutputSpec` no longer carries deleted output projection concepts:
  `dialogue_only`, `message_roles`, `material_origins`, `transform`, or
  `content_projection`.
- Root query `project_query_results` is identity; selection must come from the
  query spec/algebra instead of post-selection output mutation.
- `archive_query` no longer parses hidden root projection params, no longer
  has transform-on-read branches, and no longer has session-envelope projection
  helpers for root reads/streams.
- `RootModeRequest.has_output_mode()` no longer treats removed message
  projection flags as route triggers.
- Dead `apply_transform` / `strip-tools` / `strip-thinking` / `strip-all`
  query-action code and tests were removed after confirming no production
  caller remained.
- CLI stream rendering now renders a preselected message list and no longer
  carries `dialogue_only`, message-role, material-origin, or stats-based
  selected-role projection state.
- CLI tests now assert full unprojected root reads/streams; dead helper tests
  were removed rather than turned into ceremonial compatibility checks.

Verification:

- `python -m compileall -q polylogue/cli/query_contracts.py polylogue/cli/query.py polylogue/cli/root_request.py polylogue/cli/archive_query.py tests/unit/cli/test_query_exec_laws.py tests/unit/cli/test_query_support_runtime.py tests/unit/cli/test_archive_query.py tests/unit/cli/test_click_app.py`
- `devtools test tests/unit/cli/test_query_exec_laws.py tests/unit/cli/test_query_support_runtime.py tests/unit/cli/test_archive_query.py tests/unit/cli/test_click_app.py -q` -> `250 passed in 52.20s`
- After deleting dead `apply_transform`: same focused command -> `246 passed
  in 52.07s`
- After simplifying stream rendering:
  `devtools test tests/unit/cli/test_query_exec_laws.py tests/unit/cli/test_query_support_runtime.py tests/unit/cli/test_archive_query.py tests/unit/cli/test_click_app.py tests/unit/cli/test_query_fmt.py -q`
  -> `297 passed in 50.01s`
- Focused stream-output rerun:
  `devtools test tests/unit/cli/test_query_support_runtime.py tests/unit/cli/test_query_fmt.py -q`
  -> `56 passed in 1.23s`

## 2026-06-29 20:07 Slice: API Recovery Report Facade Removal

Implemented:

- Removed `PolylogueArchiveMixin.recovery_report` and
  `PolylogueArchiveMixin.recovery_work_packet`.
- Removed the private `_compile_recovery_context_from_digest` and
  `_recovery_work_packet_from_digest` API bridge.
- Removed the assertion-to-work-packet bridge helpers that only served that
  public recovery facade.
- Updated facade tests so digest/topology coverage remains, while report and
  work-packet facade expectations disappear.
- Fixed `test_get_session_returns_none_for_unknown_id` to initialize an empty
  archive before opening read-only archive tiers; the old version was relying
  on a now-deleted extra call and failed independently when run alone.

Verification:

- `python -m compileall -q polylogue/api/archive.py tests/unit/api/test_facade_contracts.py`
- `devtools test tests/unit/api/test_facade_contracts.py::test_recovery_digest_compiles_seeded_session tests/unit/api/test_facade_contracts.py::test_recovery_digest_resolves_subagent_child_links tests/unit/api/test_facade_contracts.py::test_get_session_returns_none_for_unknown_id tests/unit/api/test_facade_contracts.py::test_compile_context_builds_message_segments_from_refs_and_query -q`
  -> `4 passed in 41.14s`

## 2026-06-29 20:12 Slice: Context Recovery Wrapper Removal

Implemented:

- Removed `RecoveryContextCompilation`, `compile_recovery_context`, and
  `context_image_from_recovery` from `polylogue.context.compiler`.
- Removed the lazy package exports for those deleted recovery-context symbols.
- Deleted recovery-specific context compiler tests and replaced the file with
  focused coverage for remaining primitives: unit-query-only specs,
  query-unit refs, context snapshot fingerprints, and explicit seed
  validation.

Verification:

- `python -m compileall -q polylogue/context tests/unit/context/test_compiler.py`
- `devtools test tests/unit/context/test_compiler.py -q` -> `5 passed in
  0.78s`

## 2026-06-29 20:26 Slice: Internal Session Digest Vocabulary

Implemented:

- Renamed the deterministic recovery/digest transform vocabulary to session
  digest/session report/successor context:
  `compile_session_digest`, `SessionDigest`, `render_session_report`,
  `SuccessorContextPacket`, and `session_digest_v0`.
- Made the archive facade helper private (`_session_digest`) so the digest is
  no longer a standalone public async API surface. Public consumers compose it
  through existing postmortem/pathology/portfolio/context flows.
- Removed the last internal `read_views=("recovery", ...)` claims from
  successor-context packet scopes; those packets now record ordinary
  message-backed source material instead of inventing a recovery view.
- Renamed artifact/operation/benchmark catalog nodes from recovery digest/report
  to session digest/report and retagged the benchmark domain as
  `session-analysis`.
- Updated readiness/status transform scope from `recovery` to
  `session-analysis`.
- Kept operational recovery language intact where it refers to real daemon,
  blob, backup, WAL, corruption, and interruption recovery.

Verification:

- `python -m compileall -q polylogue/insights polylogue/api/archive.py polylogue/artifacts/runtime.py polylogue/operations/specs.py polylogue/readiness/capability.py polylogue/core/refs.py polylogue/mcp/server_context_tools.py devtools tests/unit tests/benchmarks/test_session_digest.py`
- `env POLYLOGUE_PYTEST_WORKERS=0 devtools test tests/unit/insights/test_transforms.py tests/unit/insights/test_run_projection_materialization.py tests/unit/api/test_facade_contracts.py::test_session_digest_compiles_seeded_session tests/unit/api/test_facade_contracts.py::test_session_digest_resolves_subagent_child_links tests/unit/api/test_facade_contracts.py::test_compile_context_builds_message_segments_from_refs_and_query tests/unit/core/test_artifact_graph.py tests/unit/operations/test_specs.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/core/test_readiness_capability.py tests/unit/cli/test_status.py tests/benchmarks/test_session_digest.py -q`
  -> first run: one route-catalog failure because `session_digest` was still a
  public async facade helper; after privatizing it, rerun passed:
  `103 passed in 31.41s`.
- `devtools render all --check` initially found generated drift in
  `docs/test-quality-workflows.md`, `.cache/site`, and `AGENTS.md`; after
  `devtools render quality-reference --output docs/test-quality-workflows.md`,
  `devtools render pages`, and `devtools render agents --input CLAUDE.md --output AGENTS.md`,
  rerun passed.

## 2026-06-30 Provider Harmonization Finding: Message Usage Nullability

Finding:

- Parser docs say per-message token usage is populated only when the raw record
  carries usage info, otherwise `None`.
- `ParsedMessage` and `Message` still default `input_tokens`, `output_tokens`,
  `cache_read_tokens`, and `cache_write_tokens` to `0`.
- `index.db.messages` defines those four columns as `INTEGER NOT NULL DEFAULT 0`.
- Codex token-count events preserve absence/missing model at the event level,
  but ordinary message rows cannot distinguish provider-reported zero from no
  provider per-message usage at all.

Why it matters:

- This is exactly the “do not lie about underlying datasets while harmonizing”
  issue. Rollups can coalesce missing to zero for arithmetic, but source-derived
  message rows should be able to represent unsupported/absent usage distinctly.

Candidate next slice:

- Schema bump: make `messages.{input_tokens,output_tokens,cache_read_tokens,cache_write_tokens}` nullable with
  `CHECK(column IS NULL OR column >= 0)`.
- Change parser DTO/domain fields to `int | None`.
- Coalesce only in aggregate/rollup code where arithmetic semantics require it.
- Add tests proving absent provider usage stays NULL while explicit zero remains
  zero, especially for Codex/Claude Code.

Do not do this in the middle of the active archive rebuild; it would invalidate
the current schema-v18 convergence run and force another full rebuild.

## 2026-06-30 Browser-Capture Dogfood: Project `a` / Sinex Misalignment

Trigger:

- Operator challenged the manual CDP extraction of the ChatGPT project `a`
  session `6a413f0b-7e7c-83ed-b8ed-84004812cf6a`: this should be done by the
  real browser-capture extension/receiver path, not by a bespoke export script.

Findings:

- The first manual fallback happened because no receiver was listening on
  `127.0.0.1:8765` and the Polylogue extension was not loaded in the live Chrome
  target list.
- Loading `/realm/project/polylogue/browser-extension` into live Chrome and
  starting the receiver produced the real native browser-capture artifact:
  `/home/sinity/.local/share/polylogue/browser-capture/chatgpt/6a413f0b-7e7c-83ed-b8ed-84004812cf6a-5a87b2936bf2.json`.
- That artifact is a proper `browser_llm_session` envelope from the extension,
  source `browser-extension`, with 266 normalized turns and 437 raw ChatGPT
  backend mapping nodes. It is not just visible DOM text.
- The export bundle in `/realm/inbox/chatgpt-project-a-exports` was updated so
  `a-sinex-misalignment-analysis--chatgpt-6a413f0b.*` points at the native
  extension/backend capture rather than the earlier manual CDP fallback.
- The session is now archived, not only spooled:
  `archive-state` reports `raw_row_exists=true`, `indexed_session_exists=true`,
  and `indexed_message_count=266`.

Daemon/version trap:

- A detached installed/Nix `polylogued` process was serving HTTP but refusing to
  run the watcher because it expected `index.db` schema version 10 while the
  active repo/archive are at version 18. This made `polylogue import PATH`
  truthfully record an import event but never process it.
- Correct devloop daemon is the repo-local
  `/realm/project/polylogue/.venv/bin/polylogued run`.
- The repo-local daemon is now running as transient user unit
  `polylogued-devloop.service` via `systemd-run --user`, with receiver/API on
  ports 8765/8766.

Code cleanup discovered and fixed:

- Live insight refresh hit a stale cleanup bug:
  `build_run_projection() got an unexpected keyword argument 'recovery_events'`.
- Fixed the call sites to use `session_digest_events`.
- Focused verification:
  `devtools test tests/unit/insights/test_transforms.py -k 'run_projection_harness_uses_origin_predicate or sparse'`
  -> `6 passed, 19 deselected`.
- Restarted the repo-local daemon after the patch; fresh logs showed successful
  `insights: archive refreshed ...` lines without the stale recovery keyword
  traceback.

Design gap:

- Current browser extension supports automatic capture for open supported tabs,
  plus popup/background `captureSupportedTabs`.
- It does not support "sync all conversations in ChatGPT project
  `g-p-6a40343a1f9881918dee375ded0971a4-a`" unless those conversations are
  opened or another acquisition mode enumerates project conversation IDs.
- A correct permanent implementation should be a source acquisition capability:
  project ref -> authenticated ChatGPT project/conversation enumeration ->
  normal browser-capture/session envelopes -> standard ingest/convergence. It
  should not be an export-only script or a recovery-style silo.

## 2026-06-30 Conductor Prompt Internalization

Source:

- `/realm/inbox/download/conductor-polylogue.md`

Operational frame:

- Treat this session as the conductor of the Polylogue dogfood/demo loop, not a
  single-detail implementation worker. Pick capability-producing slices,
  delegate implementation when useful, verify on the live archive, and leave an
  inspectable artifact every loop.
- The issue set is a parts bin, not the plan. Defect hunting is justified only
  when it unlocks or protects a demonstrable capability.
- The primary architectural lens is algebraic substrate over silos: collapse
  special-purpose recovery/context/export paths into general query, projection,
  acquisition, and rendering composition.
- The flagship capability target is a real topic-lineage context pack, starting
  with Sinex-related chatlogs and then Polylogue-related chatlogs: seed/query
  log, candidates, classified set, timeline, compact model-ready pack,
  open-questions, and rejected/false-start notes.
- The composed current state: browser capture is proven against a live ChatGPT
  project `a` session, but project-wide ChatGPT sync is not yet a first-class
  acquisition mode; daemon bringup must use the repo-local `.venv/bin/polylogued`
  rather than the stale installed wrapper; session digest cleanup is partly done
  but still needs broader audit for remaining recovery/silo vocabulary and
  unread materializations.

Immediate conductor-grade next slice:

- Build or prototype the Sinex topic-lineage pack on the live archive using
  existing query primitives first. Do not invent a bespoke silo unless the
  general query/rendering substrate is demonstrably insufficient; record those
  insufficiencies as DSL/product requirements. The artifact should live under
  `/realm/inbox/demos_polylogue/topic-lineage-sinex/` and be useful enough to
  feed to a strong model.
