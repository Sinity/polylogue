# Execution Plan

Living sequencing plan for Polylogue subsystem maturation. This document is a
coordination map, not a replacement for issue acceptance criteria. See
`docs/architecture-spine.md` for the target shape and `docs/design/mk3/` for
the current reader/workbench design source material.

## Landed

| Subsystem | Status | Closed by |
|-----------|--------|-----------|
| Archive substrate (acquisition, parsing, persistence, FTS, blob store) | Done | Multiple PRs |
| Content hash idempotency | Done | #838, #421 |
| Session insights baseline (profiles, work events, phases, threads) | Landed; rigor hardening remains | Multiple PRs, #1019 |
| Cost/subscription tracking and outlook slice | Landed; product forecasting remains | #938, #943, #995 |
| Daemon convergence architecture | Landed; production proof and residual workload remain | #847, #854, #845, #1036 |
| Browser extension + receiver | Done | #937, #940 |
| Identity ledger (stable IDs across re-ingestion) | Done | #775, #963 |
| MCP server (35+ tools, read/write/admin roles) | Done | Multiple PRs |
| CLI context-pack | Done | #968 |
| Paste detection | Done | #947, #966 |
| Root artifact cleanup | Done | #954, #966 |
| Verification baseline (format, lint, mypy, topology, manifests) | Done | #944 |
| Type tightening (SubjectRef.kind enum) | Done | #421, #968 |
| CLI mutation flags (--add-tag/--remove-tag) | Done | #862, #967 |
| Manifest-only conversation prevention | Done | #945 |
| MCP error sanitization and metadata validation | Done | #948 |
| CLI format matrix coverage | Done | #949 |

## In Flight

| Substream | Blocked by | Next artifact |
|-----------|-----------|---------------|
| Daemon convergence proof and residual workload (#845, #1036) | Deployment via Sinnix for latest merged daemon changes | Production-corpus convergence report and packaged rollout |
| Source vocabulary and local-agent sources (#1022) | — | Public source-family contract and filter/completion parity |
| Insight rigor and downstream contracts (#1019) | — | Product-by-product evidence/inference/readiness matrix |
| MK3 archive workbench integration (#848, #865, #866, #867, #956, #957, #993) | Design pack now committed under `docs/design/mk3/`; active work dissolved into the issues below | Contract-first reader slice: TargetRef, message anchors, enriched reader payload, and visual smoke matrix |
| Web reader realtime (#957) | — | SSE streaming channel from daemon to reader |
| Reconciliation ledger (#944) | Open residual owners stay precise | Closeout matrix mapping stale closure claims to owner issues |

## Queued

Dependency order (items in each tier are parallelizable):

1. **Storage hardening**: blob GC (#818), FTS bloat reduction (#817), daemon safety (#771)
2. **Archive semantics**: context/protocol artifact storage (#839), provider_meta graduation (#864), source vocabulary (#1022)
3. **Read surfaces**: MK3 contract slices (#848/#993), search explainability (#873), reader marks (#867), session lineage (#866), insight rigor (#1019)
4. **Operational surfaces**: maintenance/replay planner (#996), daemon health notifications (#999), read-surface SLOs (#872)
5. **Verification depth**: systematic test architecture (#997), verifiability dashboard/traceability (#998), evidence quality (#594/#590)
6. **Product polish**: CLI polish (#958), docs revamp (#952), broad distribution (#953), webui advanced functionality (#993)

## MK3 Integration Waves

MK3 is an archive workbench direction layered over the existing archive,
daemon, API, reader, and verification issues. The design pack is evidence and
detail; the work below closes through the named issue owners.

### Wave 1 - Reader Contract Spine

Owner issues: #859, #873, #839, #864, #1022, with #848 consuming.

Scope:

- Define `TargetRef` vocabulary for conversation/message first, with disabled
  reasons for content block, attachment, paste span, raw artifact, topology
  edge, saved view, and workspace targets.
- Enrich reader list/detail/message/facet/status envelopes through shared
  query/status contracts instead of browser-only filters.
- Surface message anchors, content-block summaries, paste/attachment counts,
  provenance status, search diagnostics, and source-centered vocabulary.

Exit proof:

- daemon HTTP contract tests for list/search/detail/messages/facets/status;
- CLI/MCP/API parity checks where the same query spec is used;
- privacy tests proving ordinary reader endpoints do not leak absolute local
  paths.

### Wave 2 - Single-Conversation Reader

Owner issues: #848 and #865.

Scope:

- Implement MK3 message cards, copy actions, action rail, folds, keyboard
  baseline, header chip order, and inspector tabs.
- Cover loading, empty, no-results, raw-only, huge transcript, degraded FTS,
  auth failure, privacy/local-only, copied, denied, selected, and search-match
  states.
- Keep derived/provenance/user-state panels statusful when substrate is missing
  instead of inventing frontend-only data.

Exit proof:

- #865 browser/DOM lane covers list/search, conversation detail, degraded
  states, nonblank rendering, and private-path safety;
- fast DOM/contract smoke remains separate from heavier browser screenshot
  evidence.

### Wave 3 - Paste, Attachments, And Provenance

Owner issues: #839, #864, #848, #993.

Scope:

- Add paste-span projection or equivalent reader contract with explicit,
  heuristic, and whole-message-fallback states.
- Add attachment cards/library states: available, missing, remote-only,
  unsupported, too large, quarantined, shared, and path-redacted.
- Drill into raw artifacts, source runs, hook events, parser evidence, and
  blob/attachment links through sanitized provenance APIs.

Exit proof:

- fixtures for explicit paste spans, heuristic paste fallback, missing
  attachment, quarantined/unsupported attachment, and provenance redaction;
- typed-only/paste-only copy controls disable with specific reasons when spans
  are unavailable.

### Wave 4 - Topology And Multi-Chat Workspace

Owner issues: #866, #993, #848, #865.

Scope:

- Materialize canonical topology edges for unresolved native parents, late
  repair, confidence/provenance, cycles, continuations, forks, sidechains, and
  subagents.
- Render branch chips, lineage rail, topology explorer, parent-chain stack,
  compare-with-parent, tabs, stack, compare, and timeline from shared topology
  operations.
- Start workspace persistence in URLs and graduate to durable user-state
  storage when #867 is ready.

Exit proof:

- storage tests for unresolved parent, late repair, ambiguous parent, cycle
  quarantine, subagent/sidechain cluster, and sibling/ancestor reads;
- visual evidence for resolved parent, unresolved parent, subagent cluster,
  fork compare, too-many-lanes, and missing topology states.

### Wave 5 - Durable User State And Advanced Panels

Owner issues: #867, #993, #1019, #995.

Scope:

- Implement marks, annotations, saved views, recall packs, and workspaces as
  durable archive state keyed by `TargetRef`, not browser state.
- Roundtrip saved views through canonical query specs.
- Surface insights, cost, similarity, and provenance panels with readiness,
  freshness, confidence, missing-data, and disabled/unconfigured states.

Exit proof:

- idempotent CRUD and content-hash exclusion tests for user metadata;
- saved-view query roundtrip tests;
- visual smoke for marked, annotated, pending mutation, failed mutation, stale
  insight, partial cost, and unconfigured similarity states.

### Wave 6 - Realtime And Operational Workbench

Owner issues: #957, #999, #845, #996, #829.

Scope:

- Stream or poll typed archive, conversation, message, insight, FTS, and
  snapshot events with coalescing/backpressure semantics.
- Expose status, maintenance, capture, replay, first-run, stale,
  disconnected, and degraded states without shrinking daemon convergence scope.
- Keep packaging/systemd responsible for service placement, restart, and
  operator control; application code reports health and progress.

Exit proof:

- event contract tests for append/update/snapshot/stale/disconnected states;
- reader/list DOM smoke for appended messages, stale state, disconnected
  fallback, maintenance running/failed, and first-run setup;
- production-corpus convergence report after deployment through Sinnix.

### Wave 7 - Assurance, Docs, And Distribution

Owner issues: #865, #952, #953, #958, #997, #998, #594, #590.

Scope:

- Cover every primary MK3 view and degraded state with executable visual
  evidence.
- Use real daemon-reader screenshots and videos in README/docs, not design
  mockups.
- Polish CLI output/completions/color as the companion control plane.
- Package the daemon/web reader/service setup for Nix/PyPI/container/homebrew
  surfaces without source-checkout assumptions.
- Route proof/verification reports to executable checks instead of
  metadata-only artifacts.

Exit proof:

- `devtools verify --quick`;
- full `devtools verify` before PR readiness;
- package/build checks when #953 changes distribution surfaces;
- issue/PR acceptance matrix marking each MK3 row implemented, deferred to a
  precise child issue, or rejected with rationale.

## Frozen / Parked

| Subsystem | Freeze reason | Unfreeze condition |
|-----------|--------------|-------------------|
| TUI dashboard | Lower priority than web reader | Web reader reaches stable MK3 workbench |
| Site publication | Lower priority than web reader | Reader docs completion |
