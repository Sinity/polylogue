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
| User state/advanced panels | #867, #993, #1019, #995 | Blocked on TargetRef/query serialization | Add TargetRef-based marks/annotations first | user-state CRUD tests, content-hash exclusion tests, saved-view roundtrip tests |
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
