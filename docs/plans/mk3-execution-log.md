# MK3 Execution Log And Agent Coordination Scratchpad

This file is the durable coordination scratchpad for MK3 and daemon-workbench
execution. It is intentionally practical: use it to record active slices,
subagent lanes, test choices, resource constraints, and handoff evidence. Issue
bodies remain the source of acceptance criteria; this log records execution
state and cross-agent coordination.

## Current Coordination State

Updated: 2026-05-15

| Lane | Issue owners | Status | Next action | Verification lane |
|------|--------------|--------|-------------|-------------------|
| Runtime convergence and host proof | #845, #996, #999, #829, #869 | Ready for deployment/proof slice | Ensure packaged daemon is latest through Sinnix, run real archive convergence, record residual workload | service status, daemon logs, real archive convergence report, targeted daemon tests |
| Reader contract spine | #859, #873, #839, #864, #1022, #1041, #1027 | Ready to start | Define TargetRef/message anchors and typed reader envelopes | daemon HTTP contract tests, CLI/MCP/API parity tests, privacy tests |
| Reader evidence shell | #848, #865, #956, #993 | Blocked on enough contract stability for full UI, but evidence harness can advance | Expand synthetic reader DOM/browser smoke toward MK3 states | `pytest tests/unit/daemon/test_web_reader.py`, future browser lane, `devtools verify --quick` |
| Paste/attachment/provenance | #839, #864, #848, #993 | Blocked on message envelope/provenance vocabulary | Implement paste-span projection MVP after contract spine | focused storage/payload tests, daemon API tests, visual state tests |
| Topology/workspace | #866, #993, #848, #865 | Substrate can start once source identity is stable | Materialize topology edges before graph UI | topology storage tests, parser fixtures, visual graph states |
| User state/advanced panels | #867, #993, #1019, #995 | Active MCP parity slice | Expose marks and saved views through write-role MCP tools; keep annotations/message targets and CLI parity open | MCP mutation tests, saved-view roundtrip tests |
| Verification throughput | #1026, #997, #998, #594, #590, #1012 | Active testmon default slice | Keep pytest-testmon seeded, reduce outlier runtime, and remove remaining full-suite reflexes | `devtools verify`, `devtools verify --seed-testmon --skip-slow`, focused regression tests |

## Active Slice Notes

### 2026-05-15 - Reader workspace shell and recall-pack contract cleanup

Target:

- Turn the `/w/stack` and `/w/compare` shell routes into real reader workspace
  modes, not only daemon routes that serve the generic shell.
- Treat `/` and `/c/{id}` as canonical reader routes after auditing the daemon
  and visual-evidence surfaces.
- Remove the pre-release recall-pack `conversation_ids` write shim; recall
  packs now accept typed target `items` as the single mutation contract.

Coordination:

- Branch: `feature/feat/reader-workspace-ui`.
- Read-only sidecars audited route compatibility and visual test coverage.
- Focused verification target: daemon reader contracts, CLI/MCP user-state
  contracts, storage user-state contract, visual DOM lane, quick gate.

Owned files:

- `polylogue/daemon/web_shell.py`
- `polylogue/api/archive.py`
- `polylogue/daemon/user_state_http.py`
- `polylogue/cli/commands/user_state.py`
- `polylogue/mcp/server_mutation_tools.py`
- reader user-state and visual tests

Outcome:

- `/` remains the canonical reader/search route and `/c/{id}` remains the
  canonical single-conversation deep link; neither is a compatibility alias.
- `/w/stack` and `/w/compare` now hydrate stack/compare workspace views from
  `/api/stack` and `/api/compare` instead of only serving the generic shell.
- Workspace save/restore and recall-pack creation controls are present in the
  shell and write canonical typed target items.
- Recall-pack mutations no longer accept the legacy `conversation_ids` input
  path through daemon, CLI, MCP, or the facade; the stored
  `conversation_ids_json` remains only as a derived resolved-conversation
  index.
- Removed the daemon HTTP `_is_localhost` test-surface alias; tests now pin the
  shared `is_loopback_host` helper directly.

Verification:

- `pytest -q tests/unit/storage/test_user_state_contracts.py::test_recall_pack_items_resolve_and_degrade_explicitly tests/unit/daemon/test_web_reader.py::TestReaderUserState::test_recall_packs_roundtrip_cited_conversations tests/unit/daemon/test_web_reader.py::TestReaderUserState::test_recall_pack_rejects_conversation_ids_compat_input tests/unit/cli/test_user_state_command.py::test_user_state_recall_pack_save_passes_typed_items tests/unit/mcp/test_user_state_tools.py::test_recall_pack_tools_roundtrip_typed_payloads tests/unit/daemon/test_daemon_http_security.py::TestIsLocalhost`
- `pytest -q tests/unit/daemon/test_web_reader.py::TestReaderSearchState::test_root_returns_html_with_required_regions tests/unit/daemon/test_web_reader.py::TestReaderWorkspaceRoutes::test_workspace_shell_routes_are_unauthenticated tests/visual/test_reader_dom_smoke.py::test_reader_search_shell_dom_evidence tests/visual/test_reader_dom_smoke.py::test_reader_stack_workspace_dom_evidence tests/visual/test_reader_dom_smoke.py::test_reader_compare_workspace_dom_evidence`
- `pytest -q tests/visual/test_reader_dom_smoke.py`
- `pytest -q tests/unit/cli/test_user_state_command.py tests/unit/mcp/test_user_state_tools.py tests/unit/storage/test_user_state_contracts.py`
- `pytest -q tests/unit/daemon/test_daemon_http_security.py`
- `pytest -q -n0 tests/unit/daemon/test_web_reader.py`
- `pytest -q tests/unit/mcp/test_tool_schema_witness.py tests/unit/mcp/test_user_state_tools.py::test_recall_pack_tools_roundtrip_typed_payloads`
- `pytest -q --tb=short --ignore=tests/integration -n 4`
- `ruff format --check <changed python files>`
- `ruff check <changed python files>`
- `python -m mypy <changed python files>`
- `devtools render-all --check`
- `devtools verify --quick`

Resource note:

- One `devtools verify` attempt reached pytest and was killed by earlyoom while
  a concurrent NixOS rebuild was also running; the reduced-worker pytest
  command above completed the same non-integration pytest set with 6422 passed.

## Subagent Parallelization Model

Use subagents when the work can be split by owned files and verified without
waiting on another agent's next line of code. Each subagent should comment on
its issue with scope before implementation and commit every successful
milestone in its branch/worktree.

Do not default to worktrees. Use the lightest coordination mode that preserves
clarity:

- **Same-context helper**: read-only research, issue-thread synthesis, test
  failure classification, review of a proposed patch. The main agent owns all
  writes.
- **Same-branch serialized worker**: small implementation slice with disjoint
  files, reviewed and committed by the main agent before another worker touches
  adjacent files.
- **Worktree worker**: long-running or mechanically large implementation that
  needs independent commits, or any slice whose write set would block the main
  checkout for too long.

| Subagent lane | Best for | Owns | Avoids | First check |
|---------------|----------|------|--------|-------------|
| Contract worker | TargetRef, reader payloads, API envelopes | `polylogue/surfaces/`, `polylogue/daemon/http.py`, `polylogue/archive/query/`, daemon/API tests | Visual harness and CSS-heavy reader layout | `pytest tests/unit/daemon/ -q -k "api or reader or conversation"` |
| Source/provenance worker | Typed source fields, provider-meta graduation, Antigravity source shape | parsers, archive models, schema/provider docs, provenance tests | Reader JS/layout and user-state tables | parser fixture tests plus schema roundtrip |
| Visual evidence worker | Synthetic reader fixture, DOM/browser smoke, evidence manifests | `tests/unit/daemon/test_web_reader.py`, possible `tests/visual/`, `devtools/lab_scenario.py`, visual docs | Storage schema and archive semantics | targeted reader smoke lane |
| Topology worker | topology edge DDL, ingest repair, topology operations | storage DDL, ingest enrichment, archive operations, topology tests | Reader graph UI until read model exists | `pytest tests/unit/storage/test_session_topology.py -q` or new equivalent |
| User-state worker | marks, annotations, saved views, recall packs | user-state DDL/repository/API, daemon endpoints, user-state tests | topology schema and reader layout | targeted user-state storage/API tests |
| Verification worker | affected tests, pytest evidence artifacts, slow-test reduction | `devtools/verify.py`, test config, verification manifests | feature semantics | `devtools verify --quick`, targeted pytest duration captures |
| Packaging/deployment worker | Sinnix input, Nix package/service, deployment docs | Sinnix flake/module files, packaging docs, daemon service evidence | Polylogue runtime changes unless needed for packaging | `nix flake check`/targeted Nix build plus service smoke |

Serialise work when two lanes must edit the same shared files:

- `polylogue/daemon/http.py`: contract, user-state, and reader evidence lanes.
- storage DDL/schema bootstrap: topology, user-state, source/provenance lanes.
- `docs/execution-plan.md`: one planning owner at a time.
- generated docs: run render commands after all doc edits in a branch.

## Autonomous Execution Launch Checklist

Before starting an autonomous execution wave, write a short entry below with:

- target wave and issue owner;
- chosen coordination mode for each worker;
- owned files and avoided files;
- expected first commit or handoff artifact;
- first focused test for each worker;
- broad gate to run once the wave is assembled.

Default launch shape:

1. Same-context helper summarizes issue comments and existing code paths.
2. Main agent chooses the first blocking implementation slice.
3. Same-branch serialized workers handle small disjoint edits when useful.
4. Worktree workers are reserved for topology/schema, verification tooling, or
   packaging/deployment branches that can progress independently.
5. Main agent integrates, runs focused checks, updates this log, then escalates
   to `devtools verify --quick` and PR checks.

## Verification Economy

Default inner loop:

1. Run the narrow test for the touched subsystem.
2. Run static/generated checks once the slice is coherent.
3. Run `devtools verify --quick` before push.
4. Run `devtools verify` before PR readiness; it uses pytest-testmon affected
   selection after a successful seed.

Resource rules:

- Do not run full pytest repeatedly while the host is under IO pressure or low
  memory. Use focused tests and `devtools verify --quick` until the branch is
  near merge.
- Run `devtools verify --seed-testmon --skip-slow` after checkout,
  dependency/test-harness changes, or when default verify reports a missing
  seed. Do not use a hand-maintained affected-test router.
- For browser/visual lanes, keep fast DOM/contract smoke separate from the
  heavier browser screenshot lane.
- For schema/storage changes, run focused storage/parser tests before broad
  verification; OOM or timeout evidence should become an issue comment, not a
  silent retry loop.
- For packaging/deployment changes, run Nix/service checks after code tests so
  failures are easier to attribute.

## Execution Entries

### 2026-05-15 - Pytest-testmon default affected verification

Target:

- #1059 affected-test default path, replacing the homegrown file/import router
  with pytest-testmon.

Outcome:

- `devtools verify` now runs the static/generated gates plus
  `pytest --testmon --testmon-forceselect` for affected per-test selection
  under the default scale-tier marker filter.
- `devtools verify --seed-testmon --skip-slow` is the explicit full
  non-integration seed/update path. It writes `.testmondata` and
  `.cache/testmon/seed.json`; default verify refuses ad hoc or missing seed
  state instead of falling back to the full suite.
- `devtools verify --all` / `--full` is the explicit full non-integration
  diagnostic. `--affected` and `_affected_test_files()` were removed.
- The verify runner now stops after the first failed step, preventing format or
  lint failures from cascading into expensive pytest runs.
- Full/seed pytest worker count defaults to 16, affected testmon runs default
  to `-n 0` single-process to avoid xdist collection skew under
  `--testmon-forceselect`, and both can be adjusted with
  `POLYLOGUE_PYTEST_WORKERS`.
- The extra `devtools` worktree-result replay cache was removed. Every
  invocation now reaches pytest-testmon, which owns package/Python/source
  invalidation. The later #1549 cleanup removed Polylogue's hand-maintained
  changed-file invalidator; explicit full collection is now limited to
  `--seed-testmon` and `--all` / `--full`.

Measured locally:

- Missing seed: `devtools verify --json` exits 2 with seed guidance.
- Seed: `POLYLOGUE_PYTEST_WORKERS=4 devtools verify --seed-testmon --skip-slow
  --json` ran 6424 tests, pytest 209.05s, total 231.42s.
- Default after seed: `devtools verify --json` ran 5 testmon-selected tests,
  pytest 3.60s, total 24.04s.
- Default after editing the verify tests: `devtools verify --json` ran 17
  testmon-selected tests, pytest 7.54s, total 32.57s.

Verification:

- `pytest -q tests/unit/devtools/test_verify.py`
- `ruff check devtools/verify.py tests/unit/devtools/test_verify.py pyproject.toml`
- `mypy --strict devtools/verify.py tests/unit/devtools/test_verify.py`
- `POLYLOGUE_PYTEST_WORKERS=4 devtools verify --seed-testmon --skip-slow --json`
- `devtools verify --json`

### 2026-05-15 - Health aggregation evidence

Target:

- #999 daemon health tier contract and aggregation evidence.

Outcome:

- Added a deterministic health aggregation contract test that patches the fast,
  medium, and expensive tier runners, requests fast+medium, and asserts the
  expensive tier is excluded while worst-severity aggregation and tier summary
  counts are correct.
- The test records `daemon.health.aggregate` evidence with requested tiers,
  overall status, alert count, tier summary, severities, and
  `expensive_tier_excluded`.

Artifact example:

- `.cache/verification/evidence/daemon.health.aggregate-*.json`

Verification:

- `pytest -q tests/unit/daemon/test_health_contracts.py`
- `ruff check tests/unit/daemon/test_health_contracts.py`
- `mypy --strict tests/unit/daemon/test_health_contracts.py`

### 2026-05-15 - Notification dispatch evidence

Target:

- #999 notification dispatch verifiability without adding a speculative
  non-log production backend.

Outcome:

- Added tests for the actual notification dispatch contract: alert batches are
  delivered to the selected backend with config, unknown configured backend
  names fail loudly, and backend exceptions propagate to the daemon periodic
  health loop's existing catch boundary.
- The dispatch route records `daemon.notifications.dispatch` evidence with
  alert count, severities, backend name, call count, and config-forwarding fact.
- Rate-limit/dedup and any non-log transport remain open product work; this
  slice pins the current dispatch boundary first.

Artifact example:

- `.cache/verification/evidence/daemon.notifications.dispatch-*.json`

Verification:

- `pytest -q tests/unit/daemon/test_notifications.py`
- `ruff check tests/unit/daemon/test_notifications.py`
- `mypy --strict tests/unit/daemon/test_notifications.py`

### 2026-05-15 - Backup restore/read evidence

Target:

- #999 backup verification acceptance criterion: backup must be readable, not
  merely created.

Outcome:

- Added a daemon backup contract test that seeds a real archive database, runs
  `backup_archive`, opens the produced backup through the read-only SQLite path,
  runs `PRAGMA integrity_check`, and verifies restored conversation/message
  rows can be queried.
- The test records bounded `daemon.backup_restore.read` evidence with backup
  filename, DB size, warning count, integrity status, row counts, and
  `restore_query_ok`.

Artifact example:

- `.cache/verification/evidence/daemon.backup_restore.read-*.json`

Verification:

- `pytest -q tests/unit/daemon/test_backup.py`
- `ruff check tests/unit/daemon/test_backup.py`
- `mypy --strict tests/unit/daemon/test_backup.py`
- `ruff format --check tests/unit/daemon/test_backup.py`

### 2026-05-15 - Daemon status and convergence evidence artifacts

Target:

- #999 runtime observability evidence through ordinary pytest contract tests.

Coordination:

- Main-agent implementation; the slice only marks existing daemon status and
  convergence-debt tests with bounded evidence recording.

Outcome:

- `polylogued status --format json` now records a contract evidence artifact
  containing daemon component status, live source availability counts, browser
  capture status, stdout sample, and exit code.
- Convergence-debt retry tests now record source-path and conversation-subject
  retry evidence: debt before/after, retry count, and source cursor cleanup.
- Artifacts are written through `tests/infra/contract_evidence.py`, so the
  mechanism remains pytest-native and does not create a parallel proof layer.

Artifact examples:

- `.cache/verification/evidence/daemon.status.json-*.json`
- `.cache/verification/evidence/daemon.convergence_debt.source_retry-*.json`
- `.cache/verification/evidence/daemon.convergence_debt.conversation_retry-*.json`

Verification:

- `pytest -q tests/unit/daemon/test_daemon_cli.py::test_polylogued_status_json_reports_daemon_components tests/unit/daemon/test_daemon_cli.py::test_drain_convergence_debt_retries_due_items_without_source_failure tests/unit/daemon/test_daemon_cli.py::test_drain_convergence_debt_retries_conversation_subjects_without_source_lookup`
- `ruff check tests/unit/daemon/test_daemon_cli.py`
- `mypy --strict tests/unit/daemon/test_daemon_cli.py`

### 2026-05-15 - Verification suppressions reality scan

Target:

- #1062 suppression/exception discipline as a concrete source scan, not only
  an empty registry expiry lint.

Coordination:

- Main-agent implementation; no subagent needed because the write set is
  limited to `verify-suppressions` and its unit tests.

Outcome:

- `verify-suppressions` now discovers real source-level exception
  mechanisms across `polylogue/`, `tests/`, and `devtools/`: `pytest.skip`,
  `pytest.xfail`, `pytest.mark.skip/skipif/xfail`, `# noqa`,
  `# type: ignore[...]`, and coverage ignores.
- Discovery uses Python AST/tokenization rather than text grep so fixture
  strings do not inflate the report.
- `--enforce-discovered` blocks unregistered discovered exceptions; default
  mode reports the current backlog so existing repo debt is visible without
  hiding behind an empty `docs/plans/suppressions.yaml`.

Observed baseline:

- `verify-suppressions --json`: 175 discovered source exceptions,
  all currently unregistered; `blocking=False` without
  `--enforce-discovered`.

Verification:

- `pytest -q tests/unit/devtools/test_verify_suppressions.py`
- `ruff check devtools/verify_suppressions.py tests/unit/devtools/test_verify_suppressions.py`
- `mypy --strict devtools/verify_suppressions.py tests/unit/devtools/test_verify_suppressions.py`
- `verify-suppressions --json`

### 2026-05-15 - MK3 design pack dissolved into tracker

Outcome:

- `docs/design/mk3/` committed as source evidence.
- `docs/execution-plan.md` now owns MK3 sequencing through active lanes,
  dependency gates, PR candidates, and wave handoffs.
- MK3 issue anchors added to #848, #865, #993, #866, #867, #957, #859, #839,
  #864, and #956.

Verification:

- `devtools render-docs-surface --check`
- `devtools render-all --check`
- `devtools verify --quick`
- GitHub checks on #1043

Next:

- Use the near-term PR candidates in `docs/execution-plan.md` to dispatch
  non-overlapping workers.
- Keep this log updated when a lane starts, blocks, hands off, or changes
  verification strategy.

### 2026-05-15 - Wave 1 TargetRef reader contract launch

Target:

- Wave 1 / #859 contract spine, with #848/#865 reader consumers.
- First implementation slice: stable `TargetRef` payloads, deterministic
  conversation/message anchors, and explicit action availability metadata for
  unsupported reader actions.

Coordination:

- Main branch: `feature/feat/targetref-reader-contract`.
- Same-context read-only helpers: inspect daemon payload builders, existing
  tests, and visual evidence docs.
- No worktree workers for this slice; the editable surface is small and
  overlaps in `polylogue/daemon/http.py`, so implementation remains serialized.

Owned files:

- `polylogue/surfaces/payloads.py`
- `polylogue/daemon/http.py`
- `tests/unit/daemon/test_web_reader.py`
- `docs/visual-evidence.md`
- `docs/plans/mk3-execution-log.md`

Avoided files:

- Storage schema and rebuilds.
- Topology/user-state/provenance persistence.
- Web-shell visual layout beyond consuming the enriched JSON contract.

First gates:

- `pytest -q tests/unit/daemon/test_web_reader.py -k "shared or conversation or privacy"`
- `pytest -q tests/unit/daemon/test_web_reader.py`
- `devtools verify --quick`

Outcome:

- Added the shared `TargetRef` and reader action availability contracts.
- Exposed conversation/message target refs and deterministic anchors from the
  daemon reader list, detail, search-hit, and messages endpoints.
- Extended the reader smoke lane to pin enriched target/action payloads and
  privacy checks across ordinary reader JSON endpoints.
- Fixed stale topology projection ownership for
  `polylogue/storage/source_conversations.py`, which surfaced during the
  affected gate.

Verification:

- `pytest -q tests/unit/daemon/test_web_reader.py -k "shared or conversation or privacy"`
- `pytest -q tests/unit/daemon/test_web_reader.py`
- `pytest -q tests/unit/daemon/`
- `pytest -q tests/unit/devtools/test_topology_projection_witness.py`
- `devtools verify --quick`
- `devtools verify --affected --skip-slow`

### 2026-05-15 - Reader visual lane lab-scenario wiring

Target:

- #865 operator-facing command for the new fast reader visual/DOM lane.

Outcome:

- Added `devtools lab-scenario run reader-visual-smoke` as the named command
  wrapper around `pytest -q tests/visual`.
- `devtools lab-scenario list --json` now distinguishes showcase scenarios
  from the reader visual lane instead of pretending the visual lane has
  showcase tier-0 baselines.
- Added unit coverage for listing and command dispatch with report output.

Verification:

- `pytest -q tests/unit/devtools/test_lab_list_subcommands.py tests/unit/devtools/test_lab_surface.py -k "lab_scenario"`
- `devtools lab-scenario run reader-visual-smoke --json`

### 2026-05-15 - Reader user-state API launch

Target:

- #867 initial daemon reader API for durable conversation user state.

Coordination:

- Main branch: `feature/feat/reader-user-state-api`.
- Serialized implementation. The write surface overlaps `polylogue/daemon/http.py`
  and the existing user-state query helpers, so no worker lane is useful.

Owned files:

- `polylogue/daemon/http.py`
- `polylogue/storage/sqlite/queries/conversations_identity.py`
- `tests/unit/daemon/test_web_reader.py`
- `docs/plans/mk3-execution-log.md`

Outcome:

- Exposed conversation-target marks through `/api/user/marks`.
- Exposed saved views through `/api/user/saved-views`, with strict
  `ConversationQuerySpec` validation before storing query JSON.
- Exposed recall packs through `/api/user/recall-packs`, preserving cited
  conversation IDs and payload JSON.
- Fixed existing mark/saved-view/recall write helpers so writes commit across
  request boundaries.
- Kept annotations, message/range targets, UI controls, CLI parity, and MCP
  parity open under #867 rather than pretending this slice completes the issue.

Verification:

- `pytest -q tests/unit/daemon/test_web_reader.py -k "UserState or marks or saved_views or recall_packs"`
- `pytest -q tests/unit/daemon/test_web_reader.py`
- `pytest -q tests/visual`

### 2026-05-15 - Reader user-state controls launch

Target:

- #867 follow-on reader UI controls for the durable user-state APIs merged in
  #1050.

Coordination:

- Main branch: `feature/feat/reader-user-state-controls`.
- Serialized implementation. The write surface is the single-file static reader
  shell plus synthetic visual fixtures, so subagent parallelism would add merge
  friction without useful throughput.

Owned files:

- `polylogue/daemon/web_shell.py`
- `tests/visual/conftest.py`
- `tests/visual/test_reader_dom_smoke.py`
- `docs/plans/mk3-execution-log.md`

Outcome:

- Reader shell loads `/api/user/marks` and `/api/user/saved-views` alongside
  conversations.
- Conversation list and detail header display star/pin/archive state from the
  durable archive tables.
- Detail header and Notes panel can toggle conversation marks through the
  daemon API.
- Notes panel can save the current search/provider view and open saved views
  using canonical saved-view query JSON.
- Visual fixtures now seed durable marks and saved views so DOM/API evidence
  proves the surface is wired to archive state, not local browser state.
- Kept annotations, message/range targets, CLI parity, and MCP parity open
  under #867.

Verification:

- `ruff format --check polylogue/daemon/web_shell.py tests/visual/conftest.py tests/visual/test_reader_dom_smoke.py`
- `ruff check polylogue/daemon/web_shell.py tests/visual/conftest.py tests/visual/test_reader_dom_smoke.py`
- `mypy polylogue/daemon/web_shell.py tests/visual/conftest.py tests/visual/test_reader_dom_smoke.py`
- `pytest -q tests/visual tests/unit/daemon/test_web_reader.py -k "reader or UserState or saved_views or marks"`

### 2026-05-15 - MCP user-state parity launch

Target:

- #867 write-role MCP parity for durable conversation marks and saved views.

Coordination:

- Main branch: `feature/feat/mcp-user-state-tools`.
- Serialized implementation. The slice touches one MCP registration module,
  typed MCP payloads, and focused MCP contract tests.

Owned files:

- `polylogue/mcp/server_mutation_tools.py`
- `polylogue/mcp/payloads.py`
- `tests/unit/mcp/test_tool_contracts.py`
- `docs/plans/mk3-execution-log.md`

Outcome:

- Added MCP list/add/remove tools for conversation marks.
- Added MCP list/save/delete tools for saved views.
- Saved-view MCP writes validate query JSON through strict
  `ConversationQuerySpec` construction before storing canonical JSON.
- Mark mutations resolve conversation IDs through the existing query store
  before calling the shared Polylogue facade.
- Kept annotations, message/range targets, recall-pack MCP shape, and CLI
  parity open under #867.

Verification:

- `pytest -q tests/unit/mcp/test_tool_contracts.py -k "mark or saved_view"`
- `pytest -q tests/unit/mcp/test_user_state_tools.py tests/unit/mcp/test_envelope_contracts.py tests/unit/mcp/test_tool_schema_witness.py`
- `pytest -q tests/unit/mcp`
- `devtools verify --quick`

### 2026-05-15 - Wave 1 reader query smoke closure

Target:

- #865 reader smoke follow-up adjacent to #859 TargetRef contracts.
- Replace skipped query/FTS assertions with executable search and no-results
  reader checks.

Coordination:

- Main branch: `feature/test/reader-query-smoke`.
- Serialized same-branch implementation; no helper needed because the editable
  surface is confined to reader smoke, shared search-match payloads, and the
  daemon search-hit envelope.

Outcome:

- Seed the synthetic reader archive with the message FTS virtual table/triggers
  so query endpoints exercise the same readiness path as runtime archives.
- Unskipped query facets and no-result query smoke tests.
- Added positive `/api/conversations?query=...` assertions for conversation
  target refs and match-level message target refs/anchors/actions.
- Documented the now-realized query smoke coverage in `docs/visual-evidence.md`.

First gates:

- `pytest -q tests/unit/daemon/test_web_reader.py -k "query or search"`
- `pytest -q tests/unit/daemon/test_web_reader.py`

Verification:

- `pytest -q tests/unit/daemon/test_web_reader.py -k "query or search"`
- `pytest -q tests/unit/daemon/test_web_reader.py`
- `devtools verify --quick`
- `pytest -q tests/unit/daemon/`

### 2026-05-15 - Wave 2 reader visual DOM evidence launch

Target:

- #865 visual/DOM evidence lane for the MK3 reader, consumed by #848.
- First executable visual-evidence slice: browserless DOM and API evidence
  envelopes against the daemon-served reader with synthetic archive data.

Coordination:

- Main branch: `feature/test/reader-visual-dom-evidence`.
- Serialized implementation. The new `tests/visual/` package, route support,
  and docs touch a compact shared surface; no worktree worker is useful here.

Owned files:

- `polylogue/daemon/http.py`
- `tests/visual/`
- `docs/visual-evidence.md`
- `docs/plans/mk3-execution-log.md`

Avoided files:

- Browser binary packaging.
- Full web-shell product redesign.
- Real archive fixtures or screenshots containing operator data.
- Storage schema beyond synthetic test seeding.

First gates:

- `pytest -q tests/visual`
- `pytest -q tests/unit/daemon/test_web_reader.py`
- `devtools verify --quick`

Outcome:

- Added `tests/visual/` as a dedicated browserless visual/DOM evidence lane.
- The lane boots the production daemon HTTP server against synthetic archives
  and writes JSON evidence envelopes for search/list, conversation/detail,
  query/no-results/facets, empty archive, degraded FTS, and privacy safety.
- Added `/c/{id}` web-shell serving so conversation deep links can be verified
  directly without a browser rewrite.
- Normalized daemon message `message_type` serialization to stable enum values
  in reader detail and paginated message endpoints.
- Documented the fast lane split and the remaining browser screenshot gate in
  `docs/visual-evidence.md`.

Verification:

- `pytest -q tests/visual`
- `pytest -q tests/unit/daemon/test_web_reader.py`
- `ruff format --check polylogue/daemon/http.py tests/visual`
- `ruff check polylogue/daemon/http.py tests/visual`
- `mypy tests/visual polylogue/daemon/http.py`
- `devtools verify --quick`
- `devtools verify --affected --skip-slow`

### 2026-05-15 - #867 target-aware annotations substrate launch

Target:

- #867 durable reader user-state substrate beyond conversation-only marks.
- First complete target-aware slice: conversation/message marks plus
  conversation/message annotations through storage, facade, daemon HTTP, and MCP.

Coordination:

- Main branch: `feature/feat/user-state-annotations-targets`.
- Parallelized read-only reconnaissance into storage/API and CLI/surface lanes;
  implementation stayed serialized because schema, facade, daemon, and MCP
  payloads form one contract surface.

Owned files:

- `polylogue/storage/sqlite/schema_ddl*.py`
- `polylogue/storage/sqlite/schema_bootstrap.py`
- `polylogue/storage/sqlite/queries/conversations_identity.py`
- `polylogue/storage/repository/archive/repository_writes.py`
- `polylogue/api/archive.py`
- `polylogue/daemon/user_state_http.py`
- `polylogue/mcp/server_mutation_tools.py`
- `polylogue/mcp/payloads.py`
- focused storage/daemon/MCP/visual tests and MCP tool-schema witness

Avoided files:

- CLI parity; keep for the next low-conflict `user-state` command slice.
- Recall-pack export redesign; keep for a separate typed export/degraded-target
  slice after annotations exist.
- Non-conversation/message targets; those still depend on broader target
  identity work.
- Full reader layout changes; this slice only flips annotate action
  availability now that the endpoint contract exists.

First gates:

- `pytest -q tests/unit/storage/test_user_state_contracts.py ... tests/unit/mcp/test_user_state_tools.py`
- `ruff format --check <changed files>`
- `ruff check <changed files>`
- `python -m mypy <changed files>`

### 2026-05-15 - #867 CLI user-state parity launch

Target:

- #867 operator parity for the durable marks, annotations, and saved-view
  substrate.
- Add a low-conflict CLI surface without mixing user-state CRUD into query
  modifier execution.

Coordination:

- Same branch: `feature/feat/user-state-annotations-targets`.
- Informed by the read-only CLI reconnaissance lane. Implementation stayed in
  one new command module plus registration/docs/tests.

Owned files:

- `polylogue/cli/commands/user_state.py`
- `polylogue/cli/click_command_registration.py`
- `devtools/render_cli_reference.py`
- `docs/cli-reference.md`
- focused CLI tests

Avoided files:

- Query-mode mutation plumbing; this slice uses an explicit `user-state` root
  group.
- Recall-pack CLI create/export; that remains tied to the next recall-pack
  redesign slice.

Outcome:

- Added lazy `polylogue user-state`.
- Added `marks list/add/remove` with conversation/message target options.
- Added `annotations list/save/delete`.
- Added `saved-views list/save/delete`, including query-spec validation and
  canonical query JSON.
- Regenerated CLI reference docs for the new command.

Verification:

- `pytest -q tests/unit/cli/test_user_state_command.py tests/unit/cli/test_click_app.py::TestCliMetadata::test_all_subcommands_registered`
- `ruff format --check polylogue/cli/commands/user_state.py polylogue/cli/click_command_registration.py devtools/render_cli_reference.py tests/unit/cli/test_user_state_command.py tests/unit/cli/test_click_app.py`
- `ruff check polylogue/cli/commands/user_state.py polylogue/cli/click_command_registration.py devtools/render_cli_reference.py tests/unit/cli/test_user_state_command.py tests/unit/cli/test_click_app.py`
- `python -m mypy polylogue/cli/commands/user_state.py polylogue/cli/click_command_registration.py devtools/render_cli_reference.py tests/unit/cli/test_user_state_command.py tests/unit/cli/test_click_app.py`

Outcome:

- Bumped archive schema to v13.
- Migrated legacy conversation-only `user_marks` into target-aware
  `(target_type, target_id, mark_type)` rows while preserving old data.
- Added durable `user_annotations` table for conversation/message targets.
- Added target-aware mark and annotation operations through SQL helpers,
  repository writes, and `Polylogue`.
- Added daemon `/api/user/annotations` list/get/save/delete routes and widened
  `/api/user/marks` to conversation/message targets.
- Added MCP `list_annotations`, `save_annotation`, and `delete_annotation`,
  and widened mark tools to accept target arguments.
- Regenerated the MCP tool-schema witness.
- Added storage tests for rebuild, target validation, and content-hash
  exclusion.

Verification:

- `pytest -q tests/unit/storage/test_user_state_contracts.py tests/unit/storage/test_user_state_contracts.py tests/unit/daemon/test_web_reader.py tests/unit/mcp/test_user_state_tools.py tests/unit/mcp/test_envelope_contracts.py tests/unit/mcp/test_tool_schema_witness.py tests/visual/test_reader_dom_smoke.py`
- `ruff format --check <changed files>`
- `ruff check <changed files>`
- `python -m mypy <changed files>`

### 2026-05-15 - #867 recall-pack target evidence launch

Target:

- Complete the next recall-pack slice without another schema rebuild by
  making the stored payload typed and self-describing.
- Preserve daemon compatibility for legacy `conversation_ids` while allowing
  explicit conversation/message/mark/annotation items.
- Add CLI and MCP parity so recall packs are not daemon-only user state.

Coordination:

- Branch: `feature/feat/recall-pack-target-evidence`.
- Read-only sidecar confirmed existing recall packs only surfaced through the
  async facade, storage helpers, and daemon `/api/user/recall-packs`; CLI and
  MCP parity were absent.
- Implementation kept storage table shape stable and centralized target
  resolution in the `Polylogue` facade.

Owned files:

- `polylogue/api/archive.py`
- `polylogue/daemon/user_state_http.py`
- `polylogue/cli/commands/user_state.py`
- `polylogue/mcp/payloads.py`
- `polylogue/mcp/server_mutation_tools.py`
- user-state storage, daemon, CLI, and MCP tests

Outcome:

- Recall-pack saves now normalize legacy `conversation_ids` and explicit
  `items` into a payload with `schema_version`, `items`, `resolved_count`, and
  `degraded_count`.
- Conversation, message, mark, and annotation items resolve to stable target
  evidence where possible.
- Missing conversations/messages/marks/annotations degrade explicitly with
  `status` and `disabled_reason`.
- Unsupported future target types are preserved as unsupported degraded items
  instead of being silently dropped.
- Added `polylogue user-state recall-packs list/save/delete`.
- Added MCP `list_recall_packs`, `save_recall_pack`, and
  `delete_recall_pack`, with envelope classification and regenerated schema
  witness.

Verification:

- `pytest -q -n0 tests/unit/storage/test_user_state_contracts.py tests/unit/daemon/test_web_reader.py::TestReaderUserState::test_recall_packs_roundtrip_cited_conversations tests/unit/cli/test_user_state_command.py tests/unit/mcp/test_user_state_tools.py`
- `ruff check polylogue/api/archive.py polylogue/cli/commands/user_state.py polylogue/mcp/payloads.py polylogue/mcp/server_mutation_tools.py tests/unit/storage/test_user_state_contracts.py tests/unit/daemon/test_web_reader.py tests/unit/cli/test_user_state_command.py tests/unit/mcp/test_user_state_tools.py`
- `python -m mypy polylogue/api/archive.py polylogue/cli/commands/user_state.py polylogue/mcp/payloads.py polylogue/mcp/server_mutation_tools.py`
- `pytest -q -n0 tests/unit/mcp/test_tool_schema_witness.py tests/unit/mcp/test_envelope_contracts.py tests/unit/cli/test_click_app.py::TestCliMetadata::test_all_subcommands_registered tests/unit/mcp/test_user_state_tools.py tests/unit/cli/test_user_state_command.py`
- `devtools render-all --check`

### 2026-05-15 - verification artifact conversion start

Target:

- Stop treating proof/catalog rows as the verification authority.
- Realize the first reusable pytest artifact mechanism for contract tests.
- Reframe planning docs and issues around pytest/coverage/benchmark/CI/runtime
  artifacts.

Owned files:

- `tests/infra/contract_evidence.py`
- `tests/infra/test_contract_evidence.py`
- `tests/unit/cli/test_json_envelope_contract.py`
- `pyproject.toml`
- `tests/conftest.py`
- verification planning docs under `docs/` and `docs/plans/`

Outcome:

- Added `record_contract_evidence`, an explicit opt-in pytest fixture that
  writes bounded JSON artifacts and exposes artifact paths through pytest
  `record_property`.
- Added redaction/truncation for repo/home paths and obvious secret assignment
  strings before evidence reaches disk.
- Registered the `contract` pytest marker.
- Wired the CLI JSON envelope matrix to emit `cli.json_envelope` evidence.
- Wired MCP surface-registration contract tests to emit evidence for tools,
  resources, resource templates, and prompts.
- Marked proof-era manifests as transitional inventories, not verification
  closure.
- Updated #1058/#1059/#1060/#1062/#1063/#1064/#594/#997/#999 issue bodies so
  each points at standard mechanisms and concrete owner surfaces.

Verification:

- `pytest -q tests/infra/test_contract_evidence.py tests/unit/cli/test_json_envelope_contract.py tests/unit/mcp/test_server_surfaces.py::TestServerSurfaceRegistration::test_server_surface_contract`
- `ruff check tests/infra/contract_evidence.py tests/infra/test_contract_evidence.py tests/conftest.py tests/unit/cli/test_json_envelope_contract.py tests/unit/mcp/test_server_surfaces.py`
- `mypy --strict tests/infra/contract_evidence.py tests/infra/test_contract_evidence.py tests/conftest.py tests/unit/cli/test_json_envelope_contract.py tests/unit/mcp/test_server_surfaces.py`
- `devtools verify-manifests`
- `devtools render-all --check`
