# HANDOFF — Budgeted per-component status snapshots

## Operator summary

This package implements `polylogue-20d.17` against the supplied Polylogue snapshot. The change replaces request-time whole-status assembly with one shared, typed component-snapshot protocol used by daemon/archive status and agent coordination. The daemon refreshes components independently outside request handlers; CLI, HTTP and MCP consumers select retained components and compatibility fields from memory. A slow, failed or hung collector therefore changes only its own lifecycle state and cannot delay healthy components.

The implementation covers the default and explicit exact status routes, preserves the established top-level compatibility envelope, exposes stable self-describing lifecycle metadata for WebUI job `webui-05`, and makes no-daemon direct mode descriptor-only. The exact raw-materialization, archive-debt and raw-replay diagnostics are manual, collection-limited, cooperatively cancellable and fingerprint-scoped resumable jobs rather than synchronous request probes.

## Authoritative snapshot identity

- Supplied project-state manifest: `polylogue-manifest.json`, generated `2026-07-17T180950Z`.
- Manifest branch: `master`.
- Manifest commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
- Commit subject: `fix(repair): harden raw authority convergence (#3046)`.
- Manifest dirty flag: `true`.
- Manifest branch-delta patch, branch-delta log and branch-delta file list: zero bytes.
- Applying the supplied tracked working-tree overlay to the named commit produced no tracked source delta. The implementation patch is therefore based on the named commit.
- Implementation work branch: `perf-01-status-snapshots-r02`; the patch itself is commit-independent and was clean-applied to a detached worktree at the authoritative commit.

## Evidence inspected

The pass inspected the full `polylogue-20d.17` Beads record and the related records `polylogue-703`, `polylogue-s7ae.8`, `polylogue-20d.14`, `polylogue-cuxz`, and `polylogue-feqr`. It followed the production routes through `daemon/status.py`, `daemon/status_snapshot.py`, `daemon/cli.py`, `daemon/http.py`, `cli/commands/status.py`, `cli/commands/agents.py`, coordination payload/building code, MCP tools/resources/prompts, the MCP call-log outbox, archive-readiness/debt providers, SQLite candidate paging, package entry points and generated topology/equivalence declarations.

Relevant history inspected included:

```text
5ef9f5a4d feat(devtools): execute typed authority cohorts (#3021)
ef17859b3 fix(status): omit archive debt from periodic snapshots (#3002)
fa6691b61 fix(status): bound periodic readiness classification (#3001)
885f2938d fix(daemon): bound periodic status snapshots (#2999)
b9431a05a fix(extension): skip unauthenticated receiver probes (#2998)
d17ded51b feat(storage): unify raw authority frontier convergence (#2962)
ad3eb5e77 fix: make silent exception swallows loud across evidence surfaces (#2963)
682b29bf3 refactor(insights): delete prose-keyword heuristics for terminal state and activity labels (#2960)
3556de6ed fix(cli): report failed status components as explicit unknown entries (#2959)
7bf2bb5f3 chore(beads): operator decisions + xnws narration pilot traces (#2958)
9665db1f0 chore(beads): record exception-swallowing census on polylogue-xnws
f9445d454 chore(beads): close completed index v37 fast-forward (#2951)
```

The focused history showed successive attempts to bound whole snapshots (`#2999`, `#3001`, `#3002`), compact coordination projections (`#2656`, `#2784`, `#2809`) and explicit unknown status entries (`#2959`). The current mechanism keeps the useful compatibility projections but moves scheduling and freshness to per-component declarations.

## Protocol types

All consumers share `polylogue/core/status_snapshots.py`.

`StatusComponentState` is the closed state family `fresh | stale | refreshing | timed_out | unavailable | degraded`. `StatusCostClass` is `cheap | bounded | expensive`. `StatusDetailClass` is `summary | standard | detail | exact`. `StatusPrivacyClass` is `public | operator | sensitive`. `StatusRefreshTrigger` is `interval | source_fingerprint | manual`.

`StatusCollectionContext` is the immutable collector input. It carries `component_id`, dependency values, selected detail class, deadline, ISO start time, an optional work-unit `collection_limit`, an optional previous resumable value, and a cooperative `cancel_requested` callback.

`StatusCollectionResult` is the collector output. It carries a JSON value, one collector-terminal state (`fresh`, `degraded`, or `unavailable`), optional observation time, reason, evidence references and detail reference. Scheduler-owned lifecycle transitions add `refreshing`, `timed_out` and age-derived `stale`.

`StatusComponentSpec` is the complete declaration for one independently scheduled fact family. Its fields are `component_id`, collector, dependencies, cost class, detail class, deadline, refresh interval, staleness policy, refresh trigger, optional source-fingerprint provider, privacy class, owned projection fields, detail reference and default-enabled flag. Declaration validation rejects self/duplicate dependencies, non-positive deadlines/staleness, negative intervals and source-fingerprint triggers without providers.

`StatusSnapshot` is one immutable observation. It records component identity, lifecycle state, optional value, observation/start/finish timestamps, corresponding monotonic timestamps, source fingerprint, reason, evidence refs, detail ref and generation attempt.

`StatusSnapshotRegistry` validates the dependency graph, expands requested dependencies, schedules collector generations, tracks deadlines, retains last-good and resumable evidence, and produces immutable projections. Read methods (`project`, `component_values`, `snapshot`) hold only the registry lock and never invoke a collector or fingerprint provider.

## Scheduler behavior and failure semantics

The daemon starts `_periodic_status_snapshot_refresh` alongside its existing periodic loops. Every second it moves daemon and coordination scheduler work into `asyncio.to_thread`, so filesystem or process fingerprint calculation cannot block the daemon event loop. The registries then start one daemon thread per due component generation. Request handlers never join those threads.

A collector generation becomes `refreshing` immediately. Projection reads mark it `timed_out` after its declared deadline and set its cooperative cancellation event. The registry keeps the worker generation identity until the worker exits, preventing scheduler ticks from creating an unbounded replacement-thread pile. A late result from a timed-out generation cannot overwrite the timeout. A useful late partial value may be retained only as resumable diagnostic state.

Fresh evidence older than `stale_after_s` projects as `stale` without mutating the stored observation. A collector exception becomes `degraded` when fresh last-good evidence exists, otherwise `unavailable`. Timed-out, refreshing, unavailable and valueless degraded components can contribute retained last-good compatibility fields, but the lifecycle projection still reports the current non-fresh state. Only a genuinely fresh result becomes `last_good`; degraded evidence is never mislabeled as last-good.

Source fingerprints are sampled by scheduler calls, not reads. A changed fingerprint refreshes the component even when its interval has not expired. A dependency finishing after a dependent component also makes that dependent due. Resume values are reused only while their source fingerprint remains unchanged.

## Daemon/archive component inventory

| Component | Owned compatibility facts | Dependencies | Class | Deadline | Refresh / invalidation | Stale after | Detail reference |
|---|---|---|---|---:|---|---:|---|
| `runtime` | `daemon_liveness`<br>`daemon_lifecycle`<br>`component_state`<br>`live`<br>`browser_capture`<br>`watcher_roots`<br>`browser_capture_active` | none | `cheap` / `standard` / `public` | 0.15 s | 1 s scheduler tick; 2 s interval | 10 s | `polylogued status --format json` |
| `writer` | `daemon_write_coordinator` | none | `cheap` / `standard` / `public` | 0.1 s | 1 s scheduler tick; 2 s interval | 10 s | `polylogued health --format json` |
| `storage` | `db_path`<br>`db_size_bytes`<br>`wal_size_bytes`<br>`blob_dir_size_bytes`<br>`disk_free_bytes`<br>`archive_storage` | none | `bounded` / `standard` / `public` | 0.5 s | fingerprint of every tier DB/WAL/SHM plus `blobs/` | 45 s | `polylogue ops readiness --deep` |
| `fts` | `fts_readiness` | none | `bounded` / `standard` / `public` | 0.4 s | `index.db` + WAL fingerprint | 45 s | `polylogue ops readiness --deep` |
| `insights` | `insight_freshness` | none | `bounded` / `standard` / `public` | 0.4 s | `index.db` + WAL fingerprint | 45 s | `polylogue ops readiness --deep` |
| `raw_materialization` | `raw_materialization_readiness` | `storage` | `expensive` / `standard` / `public` | 1.5 s | `source.db`, `index.db`, `ops.db` + WAL fingerprint; dependency refresh | 120 s | `polylogue status --exact-archive-readiness --json --full` |
| `raw_materialization_exact` | `raw_materialization_readiness_exact` | none | `expensive` / `exact` / `public` | 10 s | manual request; same DB/WAL/SHM fingerprint scopes resume state; default disabled | 300 s | `polylogue status --exact-archive-readiness --json --full` |
| `raw_frontier` | `raw_frontier_integrity` | `raw_materialization` | `expensive` / `standard` / `public` | 1.75 s | `source.db`, `index.db`, `ops.db` + WAL fingerprint; dependency refresh | 120 s | `polylogue ops readiness --deep` |
| `live_cursor` | `live_cursor`<br>`failing_files` | none | `bounded` / `standard` / `public` | 0.75 s | `ops.db` + WAL fingerprint | 30 s | `polylogued health --format json` |
| `live_ingest` | `live_ingest_attempts`<br>`memory` | none | `bounded` / `standard` / `public` | 0.75 s | `ops.db` + WAL fingerprint | 30 s | `polylogued health --format json` |
| `convergence` | `convergence`<br>`cursor_lag`<br>`catchup` | `live_ingest` | `expensive` / `standard` / `public` | 3 s | `index.db` + `ops.db` WAL fingerprint; dependency refresh | 60 s | `polylogued health --format json` |
| `raw_failures` | `raw_parse_failures`<br>`raw_validation_failures`<br>`raw_quarantined`<br>`raw_maintenance_failures`<br>`raw_detection_warnings`<br>`raw_failure_samples` | none | `bounded` / `standard` / `public` | 0.75 s | `source.db` + `ops.db` WAL fingerprint | 60 s | `polylogued health --format json` |
| `embeddings` | `embedding_readiness` | none | `expensive` / `standard` / `public` | 2 s | `index.db` + `embeddings.db` WAL fingerprint | 120 s | `polylogue ops embed status --json --detail` |
| `mcp_delivery` | `mcp_call_delivery` | none | `expensive` / `standard` / `public` | 1 s | 1 s scheduler tick; 5 s interval; dispatcher updates retained counts on submit/drain | 30 s | `MCP readiness_check` |
| `health` | `health` | none | `bounded` / `standard` / `public` | 0.75 s | 1 s scheduler tick; 10 s interval | 45 s | `polylogued health --format json` |
| `last_ingestion` | `last_ingestion_batch`<br>`last_event_id` | none | `cheap` / `standard` / `public` | 0.25 s | `ops.db` + WAL fingerprint | 30 s | `polylogue events --limit 1` |
| `claim_guard` | `claim_guard`<br>`archive_storage` | `storage`, `fts`, `raw_materialization`, `raw_frontier`, `live_ingest` | `cheap` / `standard` / `public` | 0.25 s | dependency refresh of storage/FTS/raw/frontier/live-ingest; 5 s interval | 30 s | `polylogue ops readiness --deep` |
| `archive_debt` | `archive_debt` | none | `expensive` / `exact` / `public` | 10 s | manual request; source/index/ops DB+WAL fingerprint scopes resume state; default disabled | 300 s | `polylogue archive-debt --limit 50` |
| `raw_replay` | `raw_replay_backlog` | none | `expensive` / `exact` / `public` | 10 s | manual request; source/index/ops DB+WAL fingerprint scopes resume state; default disabled | 300 s | `polylogue ops readiness --deep` |

## Agent-coordination component inventory

| Component | Owned compatibility facts | Dependencies | Class | Deadline | Refresh / invalidation | Stale after | Detail reference |
|---|---|---|---|---:|---|---:|---|
| `repo` | `repo` | none | `bounded` / `standard` / `operator` | 0.75 s | Git worktree/admin/common-dir HEAD, branch ref, index, packed-refs and cwd metadata | 15 s | `polylogue agents status --detail --format json` |
| `process` | `self`<br>`peers`<br>`resource_episodes` | `repo` | `bounded` / `standard` / `operator` | 0.9 s | identity-only `/proc` digest (PID, Name/PPid/NSpid, cmdline, cgroup, cwd) | 15 s | `polylogue agents self --detail --format json` |
| `work_item` | `work_item` | `repo` | `bounded` / `standard` / `operator` | 0.75 s | `.beads` DB/WAL/SHM, issues JSONL and config fingerprint | 30 s | `bd list --status=in_progress --json` |
| `beads` | `beads` | `repo` | `expensive` / `standard` / `operator` | 1.5 s | `.beads` DB/WAL/SHM, issues JSONL and config fingerprint | 60 s | `polylogue agents work-item --detail --format json` |
| `archive` | `archive` | `process` | `bounded` / `standard` / `operator` | 1 s | source/index/user DB/WAL/SHM fingerprint | 60 s | `polylogue agents status --detail --format json` |
| `handoff_evidence` | `handoff`<br>`session_trees`<br>`activity_episodes`<br>`subagent_exchanges`<br>`proof_refs`<br>`context_flow_refs` | `repo`, `process`, `archive` | `expensive` / `standard` / `operator` | 1.5 s | scratch/handoff files plus index/user DB/WAL/SHM fingerprint | 90 s | `polylogue agents handoff --detail --format json` |
| `derived_overlap` | `overlaps`<br>`advisories` | `repo`, `process`, `work_item`, `archive`, `handoff_evidence` | `cheap` / `standard` / `operator` | 0.25 s | dependency refresh; 2 s interval | 15 s | `polylogue agents conflicts --detail --format json` |

## Request-path wiring

The following production routes now perform one bounded transport read and/or an in-memory projection only:

- `polylogue status` calls `fetch_daemon_snapshot('/api/status')`; when the daemon is absent it calls `direct_status_snapshot_payload`.
- `polylogued status` is dispatched by the lightweight `polylogue.daemon.entrypoint` and avoids importing the full long-running daemon composition.
- `polylogue agents …` calls `coordination_snapshot_payload`; no-daemon fallback emits the same explicit unavailable component contract without importing Git/process/Beads/archive collectors.
- `GET /api/status` selects components/detail class and calls `get_status_snapshot_payload`.
- `GET /api/agents/coordination` calls `get_coordination_snapshot_payload`.
- MCP `readiness_check`, `polylogue://readiness`, default/detail `embedding_status`, `agent_coordination`, and `agent_coordination_brief` project through `operations/status_client.py`.
- MCP outbox delivery pressure is an independent `mcp_delivery` component. Local MCP fallback reads a lock-protected retained counter; it does not scan the outbox directory in the request thread.

`fresh=1` schedules eligible selected components in the background and immediately returns the current projection. Component selection expands dependencies before scheduling. It is not a synchronous freshness guarantee.

The legacy rich CLI helpers and the rich daemon status builder remain for explicit diagnostics and compatibility, but the default routes above do not enter them. This is the minimal convergence required by `polylogue-20d.17`; deleting the now-dominated code is deliberately left to `polylogue-703` after independent local certification.

## Direct mode

No-daemon status computes only cheap process-local and filesystem descriptor evidence: archive root, active DB descriptor sizes/free space, runtime/config metadata and explicit component declarations. Expensive archive, embedding, debt, raw-replay, Beads, process-tree and handoff components project `unavailable` with reason `daemon not running`, their deadline/staleness policy, and detail reference. Direct coordination similarly emits typed fallbacks and does not instantiate local collectors.

## Exact diagnostics

`polylogue status --exact-archive-readiness` requests the manual components `raw_materialization_exact`, `archive_debt`, and `raw_replay` with a caller-selected work-unit limit. The HTTP request queues them and returns their current lifecycle state; it does not wait.

Raw-materialization exact classification pages lexical raw-gap rows, observes cancellation at safe boundaries, accumulates category counters/cursors and resumes on a later refresh while the source fingerprint is unchanged. Raw replay uses that same bounded classifier rather than the unbounded weighted replay planner. Archive debt pages assertion candidates and raw-materialization rows one work unit at a time, treats each remaining aggregate provider as one bounded probe, caps retained rows independently, and returns `collection_progress` for resumption. All three exact components are disabled by default, have 10-second scheduler deadlines and 300-second stale policies.

## JSON contract for `webui-05`

Daemon/archive payloads preserve their established top-level fields and add:

```json
{
  "status_snapshot": {
    "protocol": "polylogue.status-snapshot.v1",
    "scope": "daemon",
    "state": "fresh|stale|refreshing|timed_out|unavailable|degraded",
    "captured_at": "ISO-8601",
    "age_s": 0.0,
    "refresh_error": null,
    "component_count": 19,
    "state_counts": {"fresh": 16, "unavailable": 3},
    "state_evidence": {}
  },
  "status_components": {
    "component_id": {
      "component_id": "component_id",
      "state": "timed_out",
      "reason": "collector exceeded 1.5s deadline",
      "observed_at": null,
      "refresh_started_at": "ISO-8601",
      "refresh_finished_at": "ISO-8601",
      "refresh_duration_s": 1.5,
      "age_s": null,
      "source_fingerprint": "…",
      "evidence_refs": [],
      "cost_class": "expensive",
      "detail_class": "standard",
      "deadline_s": 1.5,
      "refresh_interval_s": 30.0,
      "stale_after_s": 120.0,
      "refresh_trigger": "source_fingerprint",
      "privacy_class": "public",
      "dependencies": ["storage"],
      "projection_fields": ["raw_materialization_readiness"],
      "detail_ref": "polylogue status --exact-archive-readiness --json --full",
      "last_good": {
        "observed_at": "ISO-8601",
        "age_s": 18.2,
        "source_fingerprint": "…",
        "refresh_duration_s": 0.42,
        "evidence_refs": [],
        "value": null
      },
      "value": null
    }
  }
}
```

Compact status metadata intentionally omits current and last-good values; compatibility facts remain at their established top-level names. This prevents duplicated payload growth while keeping lifecycle and field ownership self-describing. Full registry projections can include values for internal diagnostics.

Coordination `AgentCoordinationPayload` adds `status_snapshot` with protocol, scope, overall state, generation time, state counts and compact component entries containing `state`, reason, observation/start/finish timestamps, duration, age, deadline, stale policy, detail reference and compact last-good timing. Compact coordination keeps the existing `projection.byte_budget`, `serialized_bytes`, `total_counts` and `omitted_counts` contract. The measured final compact payload is 6,076 bytes, below the 8 KiB acceptance boundary.

## Latency evidence and reasoning

The exact base commit and current implementation were measured as separate local subprocess trees with an isolated empty archive and no daemon:

| Surface | Base p50 | Base p95 | Current final p50 | Current final p95 | Result |
|---|---:|---:|---:|---:|---|
| `polylogue status --json` | 2,092.335 ms | 2,536.968 ms | 453.822 ms | 558.476 ms | approximately 4.6× lower p95 |
| `polylogued status --format json` | 1,731.175 ms | 2,146.949 ms | 251.667 ms | 363.913 ms | approximately 5.9× lower p95 |

The final 600-sample seeded warm production-projection campaign recorded:

| Projection | p50 | p95 | max | Serialized bytes |
|---|---:|---:|---:|---:|
| daemon randomized component selection | 0.259 ms | 0.352 ms | 0.393 ms | 7,956 |
| compact coordination randomized view/limit | 0.363 ms | 0.683 ms | 0.722 ms | 6,076 |
| descriptor-only direct status | 0.217 ms | 0.880 ms | 1.132 ms | 15,120 |

The performance change is structural: request cost is now bounded transport + lock-protected selection + compatibility assembly + serialization. Archive size and an individual collector's runtime no longer sit on the request critical path.

## Changed files

The patch changes 47 files (7,928 insertions, 666 deletions). Complete list:

- `docs/generated/mcp-equivalence.json`
- `docs/plans/degrade-loudly-allowlist.yaml`
- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`
- `polylogue/cli/commands/agents.py`
- `polylogue/cli/commands/status.py`
- `polylogue/coordination/__init__.py`
- `polylogue/coordination/payloads.py`
- `polylogue/coordination/status_snapshot.py`
- `polylogue/core/status_snapshots.py`
- `polylogue/daemon/__init__.py`
- `polylogue/daemon/cli.py`
- `polylogue/daemon/entrypoint.py`
- `polylogue/daemon/http.py`
- `polylogue/daemon/status.py`
- `polylogue/daemon/status_snapshot.py`
- `polylogue/mcp/call_log.py`
- `polylogue/mcp/declarations/registry.py`
- `polylogue/mcp/server_prompts.py`
- `polylogue/mcp/server_resources.py`
- `polylogue/mcp/server_tools.py`
- `polylogue/operations/__init__.py`
- `polylogue/operations/archive_debt.py`
- `polylogue/operations/status_client.py`
- `polylogue/readiness/capability.py`
- `polylogue/storage/archive_readiness.py`
- `polylogue/storage/sqlite/archive_tiers/user_write.py`
- `pyproject.toml`
- `tests/benchmarks/test_status_snapshots.py`
- `tests/conftest.py`
- `tests/unit/cli/commands/test_status.py`
- `tests/unit/cli/test_status.py`
- `tests/unit/coordination/test_status_snapshot.py`
- `tests/unit/core/test_status_snapshots.py`
- `tests/unit/daemon/test_daemon_cli.py`
- `tests/unit/daemon/test_daemon_entrypoint.py`
- `tests/unit/daemon/test_daemon_events_endpoint.py`
- `tests/unit/daemon/test_daemon_http_contracts.py`
- `tests/unit/daemon/test_daemon_status.py`
- `tests/unit/mcp/test_agent_coordination.py`
- `tests/unit/mcp/test_embedding_status_tool.py`
- `tests/unit/mcp/test_envelope_contracts.py`
- `tests/unit/mcp/test_mcp_call_log.py`
- `tests/unit/mcp/test_server_surfaces.py`
- `tests/unit/operations/test_archive_debt.py`
- `tests/unit/operations/test_status_client.py`
- `tests/unit/storage/test_archive_readiness.py`

## Acceptance matrix

| Requirement | Result | Evidence |
|---|---|---|
| One shared protocol for daemon/archive and coordination | Met | Both inventories are `StatusComponentSpec` declarations scheduled by `StatusSnapshotRegistry`. |
| No default request path rebuilds the rich whole | Met | Route tests monkeypatch legacy collectors/builders to raise; CLI/HTTP/MCP lanes pass. Explicit named-source and detail commands remain separate diagnostics. |
| Stalled component cannot delay healthy components | Met | Deterministic hanging/timeout/late-return tests; warm projection p95 remains below 1 ms. |
| Explicit state, timestamps, age, last-good, deadline and detail ref | Met | Stable `status_components` and coordination `status_snapshot.components` contracts. |
| Direct no-daemon story is bounded and honest | Met | Descriptor-only fallback tests fail if SQLite readiness/debt/embedding collectors are invoked. |
| Source-keyed invalidation | Met | Git linked-worktree, Beads, archive/process/handoff fingerprint and dependency invalidation tests. |
| Exact diagnostics opt-in, bounded, cancellable, resumable | Met for the three status exact components | Work limits are passed to collection, cancellation is checked, progress resumes only under the same fingerprint. Aggregate SQLite probes remain protected by scheduler deadlines. |
| CLI mixed-state rendering | Met | CLI-level test covers fresh/stale/timed-out/unavailable and last-good/detail metadata. |
| Interactive daemon/coordination budgets | Met in local deterministic and isolated subprocess evidence | Final p95 values are sub-second; compact coordination is 6,076 bytes. Operator live-daemon/archive dogfood remains unverified. |
| Randomized production-like evidence | Partially met | Committed 600-sample benchmark and local artifact include seed, Git head, archive state, p50/p95/max and bytes. No access was available to the operator's active archive/daemon/MCP client for live artifact refs. |
| Generated topology/equivalence and repository policy gates | Met | Render checks, degradation policy and clock hygiene pass in the implementation tree; targeted generated checks pass after clean application. |

## Apply order

1. Check out `536a53efac0cbe4a2473ad379e4db49ef3fce74d` with a clean working tree.
2. Run `git apply --check PATCH.diff`.
3. Run `git apply PATCH.diff`.
4. Run the changed-file Ruff and strict-mypy commands from `TESTS.md`.
5. Run the focused scheduler, daemon/HTTP, CLI and MCP lanes from `TESTS.md`.
6. Rebuild/install the wheel so the changed `polylogued` console-script entry point is installed.
7. Restart the daemon and verify `/api/status`, `/api/agents/coordination`, `polylogued status`, `polylogue status`, MCP readiness and MCP coordination against the operator archive.

The package's patch was itself applied successfully to a detached clean worktree at the named commit and passed the clean-application smoke lane.

## `polylogue-703` residuals

- `cli/commands/status.py` still contains old rich direct-status helpers below the now-returning snapshot command path. They are not used by default status and should be deleted only after a dedicated public/import compatibility audit.
- `daemon/status.py` remains the rich diagnostic/compatibility collector; default status no longer calls it. A later convergence can move remaining shared facts into components and narrow this module.
- Health/workload diagnostics outside the mission still have their own purpose-specific assemblies. They should consume component fragments where semantic parity exists rather than forcing one universal payload.
- The legacy whole-snapshot test seam in `daemon/status_snapshot.py` remains for existing tests/startup diagnostics.
- Snapshot persistence was not added. Current evidence and resumable progress are memory-only; an `ops.db` disposable-tier implementation can be considered if restart continuity becomes an operator requirement.

## Risks and limitations

No access was available to the operator's deployed daemon, active archive, browser, MCP client, secrets, NixOS service or current worktree. Live dogfood and deployment restart checks are therefore unverified. Local subprocess measurements use an isolated empty archive; warm measurements use seeded retained production projections.

Cancellation is cooperative. A collector blocked in an uncooperative native or OS call cannot be force-killed safely; the scheduler still times it out, prevents replacement-thread accumulation and keeps requests independent. Resume state is memory-only and lost on daemon restart. Source fingerprint calculation is bounded by scheduler-side work and moved off the event loop, but a very large `/proc` or filesystem metadata scan can delay that component's refresh decision without delaying requests.

The unrestricted whole-repository mypy run reports five errors in unchanged modules (`schemas/code_detection/tree_sitter.py`, `archive/query/predicate.py`, `storage/embeddings/materialization.py`, and `storage/raw_retention.py`). Every changed production Python file passes strict mypy. The complete unrestricted repository test suite was not run; affected production routes and shared mechanisms were covered by focused lanes totaling hundreds of tests.

## Value of another iteration

Against this supplied snapshot, another implementation iteration is likely to add only small repair value: operator live dogfood, deployment-specific tuning, or deletion of independently certified `polylogue-703` residual code. A substantial second pass would be justified only by new live evidence showing a route that bypasses the snapshot protocol, a collector whose scheduler-side fingerprint is itself operationally expensive, or a product decision to persist snapshots/resume cursors in `ops.db`.
