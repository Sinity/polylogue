# Execution Plan

Living sequencing plan for Polylogue subsystem maturation. This document is a
coordination map, not a replacement for issue acceptance criteria. See
`docs/architecture-spine.md` for the target shape and `docs/design/mk3/` for
the current reader/workbench design source material.

## How To Use This Plan

This plan answers three operational questions:

1. What should start next without creating avoidable rework?
2. Which issue owns the durable acceptance criteria?
3. What executable evidence has to exist before a wave can claim completion?

Issue bodies remain authoritative for scope. This document keeps the backlog
coherent across issues by naming dependency order, handoff artifacts, and
cross-issue exit gates. When an issue is split or closed, update this plan in
the same PR or issue-maintenance pass.

The coordination scratchpad for active execution is
`docs/plans/mk3-execution-log.md`. Use it for subagent lane ownership, current
status, test choices, resource constraints, and handoff notes.

## Landed

| Subsystem | Status | Closed by |
|-----------|--------|-----------|
| Archive substrate (acquisition, parsing, persistence, FTS, blob store) | Done | Multiple PRs |
| Content hash idempotency | Done | #838, #421 |
| Session insights baseline (profiles, work events, phases, threads) | Landed; rigor hardening remains | Multiple PRs, #1019 |
| Cost/subscription tracking and outlook slice | Landed; product forecasting remains | #938, #943, #995 |
| Daemon convergence architecture | Landed; production proof and residual workload remain under #845/#996/#999 | #847, #854 |
| Browser extension + receiver | Done | #937, #940 |
| Identity ledger (stable IDs across re-ingestion) | Done | #775, #963 |
| MCP server (35+ tools, read/write/admin roles) | Done | Multiple PRs |
| CLI context-pack | Done | #968 |
| Paste detection | Done | #947, #966 |
| Root artifact cleanup | Done | #954, #966 |
| Verification baseline (format, lint, mypy, topology, manifests) | Done | #944 |
| Type tightening (SubjectRef.kind enum) | Done | #421, #968 |
| CLI mutation flags (--add-tag/--remove-tag) | Done; broader mutation contracts remain under #862 | #967 |
| Manifest-only session prevention | Done | #945 |
| MCP error sanitization and metadata validation | Done | #948 |
| CLI format matrix coverage | Done | #949 |

## Active Lanes

| Lane | Live owners | Next artifact | Why now |
|------|-------------|---------------|---------|
| Runtime convergence and host evidence | #845, #996, #999, #829 | Packaged daemon rollout plus production-corpus convergence report | The daemon must converge over the real archive; bounded status/health evidence is the blocking deployment artifact. |
| Source and semantic vocabulary | #1022, #839, #864, #1041, #1027 | Source-family contract, typed context/protocol fields, and provider/schema distribution evidence | MK3 reader contracts need typed source/provenance facts before UI can stop scraping provider metadata. |
| Shared read contracts | #859, #873, #860, #848 | TargetRef/message-envelope/query/facet/status contracts shared by CLI, MCP, API, and daemon HTTP | This is the smallest useful MK3 start because later UI work consumes these envelopes. |
| Reader evidence and product shell | #848, #865, #956, #993 | Single-session reader slice with executable DOM/browser evidence | MK3 must become a verified runtime reader, not only design screenshots. |
| Topology and durable user state | #866, #867, #993 | Canonical topology read model and TargetRef-based marks/annotations/saved views | Multi-chat workspace, compare, recall packs, and topology panels need these substrates. |
| Insight and cost rigor | #1019, #994, #995 | Readiness/confidence matrix for insight, cost, similarity, and derived panels | Advanced panels must show real availability, stale, inferred, and partial states. |
| Verification and throughput | #1026, #997, #998, #594, #590, #1012 | Affected-test default, pytest evidence artifacts, benchmark/coverage reports, and inherited-failure cleanup | Broad changes are too expensive and too easy to misclaim without a faster, more precise gate. |
| Distribution and user-facing polish | #869, #953, #952, #958 | Turnkey install/service/docs/CLI package path | Package and docs work should follow runtime/read-surface contracts so it does not document unstable behavior. |

## Dependency Gates

Use these gates to decide whether a PR is ready to start, not only whether it
is ready to merge.

| Gate | Required before | Owner issues | Handoff |
|------|-----------------|--------------|---------|
| G1: typed source/provenance vocabulary | Rich reader chips, provenance panels, paste/attachment states | #1022, #839, #864, #1041, #1027 | Shared source-family names and typed fields that replace provider-meta scraping. |
| G2: shared read envelope | Reader shell, visual smoke, advanced panels | #859, #873, #860 | Versioned list/detail/message/facet/status envelopes and query serialization. |
| G3: executable reader evidence | Closing #848/#993 rows that claim runtime UI behavior | #865 | Synthetic-data DOM/browser lane with nonblank, private-safe, stateful assertions. |
| G4: topology read model | Stack, compare, lineage rail, topology explorer | #866 | Ancestor/descendant/sibling/root APIs with unresolved/late-repair/cycle evidence. |
| G5: TargetRef user state | Marks, annotations, saved views, recall packs, durable workspaces | #867 | Idempotent CRUD and content-hash exclusion for session/message targets first. |
| G6: runtime deployment evidence | Distribution docs and system service guidance | #845, #996, #999, #829, #869, #953 | Packaged daemon installed through Sinnix, real archive convergence report, health/status evidence. |
| G7: faster verification | Large MK3/daemon waves | #1026, #997, #998, #594, #590, #1012 | Affected-test default, contract evidence artifacts, and inherited failure cleanup so broad verification is credible. |

## Near-Term PR Candidates

These are deliberately shaped as coherent PR-sized slices. They can run in
parallel when their write surfaces do not overlap.

| Candidate | Owner issues | Primary files | Acceptance evidence |
|-----------|--------------|---------------|------------------|
| Define `TargetRef` and message-anchor contract | #859, #848, #867 | `polylogue/surfaces/`, `polylogue/daemon/http.py`, `polylogue/archive/query/`, daemon tests | Session/message targets serialize deterministically; reader endpoints expose anchors; unsupported targets return explicit disabled reasons. |
| Replace reader/provider metadata scraping with typed source fields | #1022, #839, #864 | source parsers, archive models, payloads, schema docs | Header/search chips read typed fields; provider-meta fallback is either removed or marked transitional with tests. |
| Make reader visual smoke MK3-ready | #865, #848, #956 | `tests/unit/daemon/test_web_reader.py`, possible `tests/visual/`, `devtools/lab_scenario.py` | Synthetic reader evidence covers list/search, session detail, empty/no-results, privacy, degraded state, and nonblank output. |
| Implement paste-span projection MVP | #839, #864, #848, #993 | message/storage projections, payloads, reader API/tests | Explicit span fixture and whole-message fallback fixture render distinct states; typed-only/paste-only copy disables when boundaries are unknown. |
| Materialize topology edge substrate | #866 | storage DDL, ingest enrichment, archive operations, parser fixtures | Child-before-parent unresolved edge, late repair, cycle quarantine, and sibling/ancestor read tests pass. |
| Add TargetRef-based marks and annotations | #867, #862 | user-state DDL/repository/API, daemon endpoints, reader tests | Idempotent CRUD works for session/message targets; content hash unchanged; reimport preservation tested where identity evidence exists. |
| Prove packaged daemon convergence | #845, #996, #999, #829, #869 | Sinnix input update, package/service config, daemon docs | Installed service is latest merged Polylogue, real archive converges, health/status and residual workload are recorded. |
| Reduce verification wall time | #1026, #997, #998, #594/#590 | `devtools/verify.py`, test config, pytest artifacts | default `devtools verify` runs affected tests; full non-slow baseline is measured explicitly and outliers have owners. |

## Parallel Execution Rules

Parallelize only where ownership is clean enough that agents can commit useful
work independently. Each worker gets an issue owner, owned files, avoided files,
first verification command, and expected handoff note in
`docs/plans/mk3-execution-log.md`.

Use the lightest coordination mode that fits:

- **Same-context helper** for read-only exploration, issue triage, test-output
  classification, and narrowly scoped code review. The main agent keeps write
  ownership and folds the findings into the current branch.
- **Same-branch serialized worker** for small patches where the write set is
  disjoint and the main agent can review immediately before the next worker
  starts.
- **Worktree worker** only when the task is long-running, risky, mechanically
  large, or likely to need independent commits while the main branch keeps
  moving.

Good parallel lanes:

- contract worker: TargetRef, reader payloads, query/status envelopes;
- source/provenance worker: typed source fields, parser/schema evidence;
- visual evidence worker: synthetic reader DOM/browser lane;
- topology worker: edge substrate and topology read model;
- user-state worker: marks, annotations, saved views, recall packs;
- verification worker: affected-test workflow, pytest evidence artifacts, and report cleanup;
- packaging/deployment worker: Sinnix input, package/service rollout evidence.

Serialize work when lanes touch the same shared file family:

- `polylogue/daemon/http.py` for contract, user-state, and reader evidence
  changes;
- storage DDL/schema bootstrap for topology, user-state, and source/provenance
  changes;
- `docs/execution-plan.md` and generated docs;
- any branch that is already waiting on CI for merge readiness.

Before dispatching an autonomous wave, record in the execution log:

- lane and issue owner;
- coordination mode: same-context, same-branch serialized, or worktree;
- exact owned and avoided files;
- first verification command;
- handoff artifact expected from the worker.

## Verification Economy

Use narrow evidence first, then broad gates at publication boundaries:

1. Run the subsystem test for the files touched.
2. Run static/generated checks once the slice is coherent.
3. Run `devtools verify --quick` before push.
4. Run full `devtools verify` before PR readiness when runtime semantics
   changed. For docs-only changes, generated docs and quick verification are
   sufficient unless the PR claims runtime behavior.

Avoid wasting RAM and wallclock:

- Do not repeatedly run full pytest while the host is under IO pressure, low
  memory, or active daemon catch-up. Capture that condition in the execution log
  and use focused tests until pressure clears.
- Keep browser screenshot evidence out of the quick loop; pair it with a fast
  DOM/contract smoke lane.
- For schema/storage changes, run focused storage/parser tests before broad
  gates.
- For packaging/deployment, run code checks first, then Nix/service checks, so
  failures are attributable.
- Keep pytest-testmon seeded with `devtools verify --seed-testmon --skip-slow`.
  The normal `devtools verify` path runs affected tests; use `devtools verify
  --all` only as an explicit full non-integration diagnostic or release/CI
  parity check.

## MK3 Integration Waves

MK3 is an archive workbench direction layered over the existing archive,
daemon, API, reader, and verification issues. The design pack is evidence and
detail; the work below closes through the named issue owners.

### Wave 1 - Reader Contract Spine

Owner issues: #859, #873, #839, #864, #1022, #1041, #1027, with #848
consuming.

Start condition:

- No broad frontend work is needed first. This wave can start immediately if it
  keeps the first TargetRef scope to session/message and treats missing
  substrate as disabled/partial states.

Scope:

- Define `TargetRef` vocabulary for session/message first, with disabled
  reasons for content block, attachment, paste span, raw artifact, topology
  edge, saved view, and workspace targets.
- Enrich reader list/detail/message/facet/status envelopes through shared
  query/status contracts instead of browser-only filters.
- Surface message anchors, content-block summaries, paste/attachment counts,
  provenance status, search diagnostics, and source-centered vocabulary.

Exit evidence:

- daemon HTTP contract tests for list/search/detail/messages/facets/status;
- CLI/MCP/API parity checks where the same query spec is used;
- privacy tests proving ordinary reader endpoints do not leak absolute local
  paths.

Handoff to next wave:

- #848 can render a message list without provider-specific frontend inference.

### Wave 2 - Single-Session Reader

Owner issues: #848 and #865.

Start condition:

- Wave 1 exposes stable enough list/detail/message/status payloads for the
  reader to avoid hand-rolled filters and metadata scraping.

Scope:

- Implement MK3 message cards, copy actions, action rail, folds, keyboard
  baseline, header chip order, and inspector tabs.
- Cover loading, empty, no-results, raw-only, huge transcript, degraded FTS,
  auth failure, privacy/local-only, copied, denied, selected, and search-match
  states.
- Keep derived/provenance/user-state panels statusful when substrate is missing
  instead of inventing frontend-only data.

Exit evidence:

- #865 browser/DOM lane covers list/search, session detail, degraded
  states, nonblank rendering, and private-path safety;
- fast DOM/contract smoke remains separate from heavier browser screenshot
  evidence.

Handoff to next wave:

- Runtime reader has a stable single-session baseline that paste,
  attachment, topology, and user-state panels can plug into without rewriting
  the shell.

### Wave 3 - Paste, Attachments, And Provenance

Owner issues: #839, #864, #848, #993.

Start condition:

- Wave 1 message envelopes exist; Wave 2 reader has fold/action/panel slots for
  large content and provenance.

Scope:

- Add paste-span projection or equivalent reader contract with explicit,
  heuristic, and whole-message-fallback states.
- Add attachment cards/library states: available, missing, remote-only,
  unsupported, too large, quarantined, shared, and path-redacted.
- Drill into raw artifacts, source runs, hook events, parser evidence, and
  blob/attachment links through sanitized provenance APIs.

Exit evidence:

- fixtures for explicit paste spans, heuristic paste fallback, missing
  attachment, quarantined/unsupported attachment, and provenance redaction;
- typed-only/paste-only copy controls disable with specific reasons when spans
  are unavailable.

Handoff to next wave:

- Paste and attachment objects can become workspace/topology/provenance targets
  without changing their identity vocabulary.

### Wave 4 - Topology And Multi-Chat Workspace

Owner issues: #866, #993, #848, #865.

Start condition:

- Topology starts in substrate as soon as source/provenance identity is stable
  enough; stack/compare UI waits for the reader baseline.

Scope:

- Materialize canonical topology edges for unresolved native parents, late
  repair, confidence/provenance, cycles, continuations, forks, sidechains, and
  subagents.
- Render branch chips, lineage rail, topology explorer, parent-chain stack,
  compare-with-parent, tabs, stack, compare, and timeline from shared topology
  operations.
- Start workspace persistence in URLs and graduate to durable user-state
  storage when #867 is ready.

Exit evidence:

- storage tests for unresolved parent, late repair, ambiguous parent, cycle
  quarantine, subagent/sidechain cluster, and sibling/ancestor reads;
- visual evidence for resolved parent, unresolved parent, subagent cluster,
  fork compare, too-many-lanes, and missing topology states.

Handoff to next wave:

- Workspaces can persist references to topology nodes/edges through TargetRef
  rather than URL-only session IDs.

### Wave 5 - Durable User State And Advanced Panels

Owner issues: #867, #993, #1019, #995.

Start condition:

- Session/message TargetRef is stable. Saved-view work also needs shared
  query serialization from Wave 1.

Scope:

- Implement marks, annotations, saved views, recall packs, and workspaces as
  durable archive state keyed by `TargetRef`, not browser state.
- Roundtrip saved views through canonical query specs.
- Surface insights, cost, similarity, and provenance panels with readiness,
  freshness, confidence, missing-data, and disabled/unconfigured states.

Exit evidence:

- idempotent CRUD and content-hash exclusion tests for user metadata;
- saved-view query roundtrip tests;
- visual smoke for marked, annotated, pending mutation, failed mutation, stale
  insight, partial cost, and unconfigured similarity states.

Handoff to next wave:

- Reader and realtime code can refer to persisted workspaces and saved views
  rather than browser-local state.

### Wave 6 - Realtime And Operational Workbench

Owner issues: #957, #999, #845, #996, #829.

Start condition:

- Runtime daemon health/status contracts are stable enough to report progress,
  pressure, stale data, and maintenance without reducing daemon scope.

Scope:

- Stream or poll typed archive, session, message, insight, FTS, and
  snapshot events with coalescing/backpressure semantics.
- Expose status, maintenance, capture, replay, first-run, stale,
  disconnected, and degraded states without shrinking daemon convergence scope.
- Keep packaging/systemd responsible for service placement, restart, and
  operator control; application code reports health and progress.

Exit evidence:

- event contract tests for append/update/snapshot/stale/disconnected states;
- reader/list DOM smoke for appended messages, stale state, disconnected
  fallback, maintenance running/failed, and first-run setup;
- production-corpus convergence report after deployment through Sinnix.

Handoff to next wave:

- Distribution docs can describe real service behavior and operational states,
  not source-checkout-only development assumptions.

### Wave 7 - Assurance, Docs, And Distribution

Owner issues: #865, #952, #953, #958, #997, #998, #594, #590.

Start condition:

- Enough runtime reader/workbench behavior exists that screenshots, packaging,
  and verification reports document product reality rather than intended designs.

Scope:

- Cover every primary MK3 view and degraded state with executable visual
  evidence.
- Use real daemon-reader screenshots and videos in README/docs, not design
  mockups.
- Polish CLI output/completions/color as the companion control plane.
- Package the daemon/web reader/service setup for Nix/PyPI/container/homebrew
  surfaces without source-checkout assumptions.
- Route verification reports to pytest, coverage, benchmark, CI, and runtime
  evidence artifacts instead of metadata-only rows.

Exit evidence:

- `devtools verify --quick`;
- full `devtools verify` before PR readiness;
- package/build checks when #953 changes distribution surfaces;
- issue/PR acceptance matrix marking each MK3 row implemented, deferred to a
  precise child issue, or rejected with rationale.

## Backlog Hygiene Rules

- Do not close a broad tracker because a shell exists. Close only when every
  row is implemented, split to a named child issue, or explicitly rejected.
- Do not add new MK3-only planning docs unless the same PR folds the actionable
  sequence back into this file and the owning GitHub issues.
- Do not leave TODO comments or skipped tests without an issue number.
- When an issue is realized, close it and update the `Active Lanes` or
  `Dependency Gates` row that referenced it.
- When deployment or production validation changes the plan, record concrete
  dates and observed evidence in the issue or PR, not only in chat.

## Frozen / Parked

| Subsystem | Freeze reason | Unfreeze condition |
|-----------|--------------|-------------------|
| TUI dashboard | Lower priority than web reader | Web reader reaches stable MK3 workbench |
| Site publication | Lower priority than web reader | Reader docs completion |
