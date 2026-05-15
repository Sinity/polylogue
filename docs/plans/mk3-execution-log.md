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
| User state/advanced panels | #867, #993, #1019, #995 | Active reader-control slice | Consume durable user-state APIs from the reader shell; keep annotations/message targets and CLI/MCP parity open | visual DOM evidence, daemon user-state API tests, saved-view roundtrip tests |
| Verification throughput | #1026, #997, #998, #594, #590, #1012 | Ready to start independently | Add affected-test workflow and reduce outlier runtime | `devtools verify --affected --skip-slow`, durations capture, focused regression tests |

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
| Verification worker | affected tests, proof routing, slow-test reduction | `devtools/verify.py`, test config, proof manifests | feature semantics | `devtools verify --quick`, targeted pytest duration captures |
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
4. Run full `devtools verify` before PR readiness when the PR changes runtime
   semantics, unless the PR explicitly records why a focused gate is sufficient.

Resource rules:

- Do not run full pytest repeatedly while the host is under IO pressure or low
  memory. Use focused tests and `devtools verify --quick` until the branch is
  near merge.
- Prefer `devtools verify --affected --skip-slow` for small changes once the
  affected-test workflow is implemented.
- For browser/visual lanes, keep fast DOM/contract smoke separate from the
  heavier browser screenshot lane.
- For schema/storage changes, run focused storage/parser tests before broad
  verification; OOM or timeout evidence should become an issue comment, not a
  silent retry loop.
- For packaging/deployment changes, run Nix/service checks after code tests so
  failures are easier to attribute.

## Execution Entries

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

- Storage schema and migrations.
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
