# Packet quality audit

This pass deliberately includes both line-localized bug packets and architecture/spec packets. Use this file to choose the right execution mode.

## Depth classes

- **source-localized**: mechanism was directly localized to source paths/functions by this pass or preserved from v1. Coding can usually start with a failing test.
- **bead-localized-from-export**: the Beads description/design already contains the implementation mechanism; this pass turned it into ordered steps and anchors. Coding can start after local inspection.
- **anchored-contract-prework**: source seams are known, but the packet still needs a design decision or contract cut before coding.
- **epic-checklist**: do not code as one PR; use as closure/splitting gate.
- **spec-only**: keep out of urgent implementation until sharpened.

## Counts

- bead-localized-from-export: 124
- anchored-contract-prework: 40
- epic-checklist: 22
- source-localized: 7
- spec-only: 1

## By lane

### 00-trust-floor

- anchored-contract-prework: 8
- source-localized: 6
- bead-localized-from-export: 3
- epic-checklist: 1

### 01-blob-attachment-integrity

- anchored-contract-prework: 4
- bead-localized-from-export: 2
- epic-checklist: 2

### 02-usage-cost-honesty

- bead-localized-from-export: 6
- anchored-contract-prework: 4
- source-localized: 1
- epic-checklist: 1

### 03-lineage-compaction-truth

- bead-localized-from-export: 5
- anchored-contract-prework: 3
- epic-checklist: 2

### 04-read-contract-query-render

- bead-localized-from-export: 15
- anchored-contract-prework: 6
- epic-checklist: 2

### 05-analysis-provenance-citations

- bead-localized-from-export: 10
- epic-checklist: 1

### 06-agent-context-coordination

- bead-localized-from-export: 22
- anchored-contract-prework: 4
- epic-checklist: 4
- spec-only: 1

### 07-content-variants

- bead-localized-from-export: 4
- epic-checklist: 1

### 08-scale-performance-live

- bead-localized-from-export: 20
- anchored-contract-prework: 6
- epic-checklist: 2

### 09-embeddings-retrieval

- bead-localized-from-export: 7
- epic-checklist: 1

### 10-analytics-experiments

- bead-localized-from-export: 13
- anchored-contract-prework: 2
- epic-checklist: 2

### 11-interoperability-origin

- bead-localized-from-export: 8
- anchored-contract-prework: 1
- epic-checklist: 1

### 12-external-legibility-demos

- bead-localized-from-export: 8
- anchored-contract-prework: 2
- epic-checklist: 2

### 99-horizon-or-general

- bead-localized-from-export: 1

## Immediate coding subset

- `polylogue-s7ae.6` — Classify the 74%-aborted full verify from the coordination commit before deploy (source-localized)
- `polylogue-37t.15` — Single agent-write chokepoint in upsert_assertion: non-user authors => CANDIDATE + inject:false, always (source-localized)
- `polylogue-kwsb.1` — Daemon/capture security hardening: Host/Origin gate, receiver token, spool governor (source-localized)
- `polylogue-8jg9.4` — ops doctor cleanup_orphans can delete an in-flight leased blob (the real #818) (source-localized)
- `polylogue-cpf.5` — Temporal provenance laundering: aggregates collapse to provider_ts; propagate the weakest source (source-localized)
- `polylogue-cpf.6` — Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit (source-localized)
- `polylogue-f2qv.5` — Version-gate provider-usage projection so it self-heals like session_profiles (source-localized)
- `polylogue-5hf` — Provider token accounting: honest cross-provider usage ledger (bead-localized-from-export)
- `polylogue-ivsc` — Classify Codex state_5 token drift outside lineage replay (bead-localized-from-export)
- `polylogue-xy95` — Speed up provider usage full stale diagnostics (bead-localized-from-export)
- `polylogue-a7xr.2` — Converger and repair disagree on session_profile staleness for NULL-sort-key sessions (bead-localized-from-export)
- `polylogue-a7xr.3` — message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x (bead-localized-from-export)
- `polylogue-a7xr.5` — FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies (bead-localized-from-export)
- `polylogue-a7xr.6` — parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb) (bead-localized-from-export)
- `polylogue-ptx` — Browser-capture posting channel: un-gate, with attachments (bead-localized-from-export)
- `polylogue-4p1` — Decision: one read algebra — Query x Projection x Render as the only read contract (bead-localized-from-export)
- `polylogue-t46.8` — MCP surface collapse: ~96 tools -> verb algebra (query/get/explain/context/assert/maintenance...) (bead-localized-from-export)
- `polylogue-jnj.1` — Collapse read per-view flags into ProjectionSpec/RenderSpec algebra (bead-localized-from-export)
- `polylogue-jnj.5` — Route ops reset --session/--source through the mutation contract (bead-localized-from-export)
- `polylogue-x7d` — Unify root query row rendering contracts (bead-localized-from-export)
- `polylogue-fnm.11` — Pipeline/clause parity across units + generated support matrix (bead-localized-from-export)
- `polylogue-fnm.10` — fields/select stage with parent-field projection (first real Transform) (bead-localized-from-export)
- `polylogue-svfj` — Block content-hash citation anchors: blocks.content_hash + resolver with typed drift states (bead-localized-from-export)
- `polylogue-rxdo.1` — ObjectRef expansion: query, query-run, result-set, finding, cohort, analysis, annotation-batch kinds (bead-localized-from-export)
- `polylogue-rxdo.2` — Content-addressed query identity + durable user.db queries/query_names/result_sets/query_edges (bead-localized-from-export)
- `polylogue-rxdo.3` — Query-run + result-relation telemetry in ops.db; refs on every query envelope (bead-localized-from-export)
- `polylogue-rxdo.4` — AssertionKind.FINDING: claims with evidence, reusing the candidate->judge lifecycle verbatim (bead-localized-from-export)
- `polylogue-3tl.16` — Public claims ledger: every README/launch claim carries a status and an evidence ref (bead-localized-from-export)
- `polylogue-d1y` — polylogue hooks install: one-command harness wiring + hook liveness monitoring (bead-localized-from-export)
- `polylogue-pj8` — Agent query cookbook: MCP prompts + skill recipes as the discoverability layer (bead-localized-from-export)
- `polylogue-t8t` — Agent workflow catalog: walk the seven core flows end-to-end, fix what breaks (bead-localized-from-export)
- `polylogue-rii.1` — Agent work-event write-leg -> session_events -> materialized read-models (bead-localized-from-export)
- `polylogue-x4s` — Express devloop state in Polylogue substrate (dogfood target) (bead-localized-from-export)
- `polylogue-0v9p` — Language detection and preference facts for variant selection (bead-localized-from-export)
- `polylogue-arso` — Content variant substrate: refs, nodes, alignment, storage (bead-localized-from-export)
- `polylogue-4ts.5` — Compaction boundary-range columns + effective-context derivation (bead-localized-from-export)
- `polylogue-gjg.1` — compaction_events + compaction_loss_items derived tables; identity survives rebuild + re-ingest (bead-localized-from-export)
- `polylogue-gjg.2` — Pre-compaction snapshot capture: hook payload when available, manifest-of-refs otherwise, honesty ladder always (bead-localized-from-export)
- `polylogue-20d.14` — Interactive SLO tier: named latency budgets, continuously measured, regression-gated (bead-localized-from-export)
- `polylogue-20d.13` — Daemon push channel: SSE events for live UIs instead of polling (bead-localized-from-export)
- `polylogue-20d.6` — Live full-ingest catch-up latency + WAL shape (bead-localized-from-export)
- `polylogue-20d.5` — Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL (bead-localized-from-export)
- `polylogue-20d.2` — Defer heavy imports off the CLI startup path (bead-localized-from-export)
- `polylogue-20d.1` — CLI->daemon fast path over UDS (persistent hot process) (bead-localized-from-export)
- `polylogue-20d.10` — Runtime post-filter efficiency: memoize semantic facts; lower matchers onto the actions view (bead-localized-from-export)
- `polylogue-o21` — Extension-point ergonomics: declare-once registries, scaffolds, actionable completeness errors (bead-localized-from-export)
- `polylogue-da1` — Provider format-drift sentinel: detect upstream export-shape changes from live ingest (bead-localized-from-export)
- `polylogue-7aw` — Ingest agent configuration as a source family (skills, CLAUDE.md, hooks) (bead-localized-from-export)
- `polylogue-ox0` — Codex deep integration: state DBs as authoritative source + AppServer live lane (bead-localized-from-export)
- `polylogue-t0p` — Capture the rest of Claude Code: todos, file-history, prompt history, debug artifacts (bead-localized-from-export)
- `polylogue-mhx.1` — Provider abstraction: one OpenAI-compatible embedding client, local and cloud, model registry in meta (bead-localized-from-export)
- `polylogue-mhx.2` — Embedding target policy: what gets a vector, at what granularity, at what cost (bead-localized-from-export)
- `polylogue-mhx.3` — Retrieval quality eval lane: measure FTS vs vector vs hybrid before believing any of them (bead-localized-from-export)
- `polylogue-0k6` — Embedding changed-text full-replace regression vs split embeddings.db metadata (bead-localized-from-export)
- `polylogue-0ns` — Bound archive embedding work within large sessions (bead-localized-from-export)
- `polylogue-1vpm.1` — Delegation derived unit: materializer + query unit + delegation-card projection (bead-localized-from-export)
- `polylogue-1vpm.2` — Episode unit: tables, 4-signal scorer with false-merge floor, assertion-calibrated (bead-localized-from-export)
- `polylogue-1vpm.4` — Turn-pair unit with prompt-burst semantics (no double-claimed answers) (bead-localized-from-export)
- `polylogue-cfk` — Re-run two-arm uplift with freshness-fixed packs (n>=3 pairs, then n=12-20) (bead-localized-from-export)
- `polylogue-212.8` — The honesty anti-demo: a tempting finding that emits verdict not_supported (bead-localized-from-export)
- `polylogue-212.1` — Post-hoc forensic Q&A demo: questions a tracer cannot answer (bead-localized-from-export)
- `polylogue-212.2` — D1 'The receipts': claim-vs-evidence on a real PR (bead-localized-from-export)
- `polylogue-212.3` — D2 'Where did the money actually go': cost by outcome (bead-localized-from-export)
- `polylogue-212.4` — D4 'Behavioral archaeology': six DSL queries, rapid fire (bead-localized-from-export)
- `polylogue-3tl.4` — Findings publishing lane: campaign artifacts on the docs site (bead-localized-from-export)
- `polylogue-3tl.7` — Release is a decision: proven install matrix across package managers and OSes (bead-localized-from-export)
- `polylogue-38x` — Reconcile archived audit residue against current source (bead-localized-from-export)
- `polylogue-9e5.25` — Review zero-use MCP surfaces from affordance usage artifact (bead-localized-from-export)
- `polylogue-9e5.26` — Review zero-use CLI surfaces from affordance usage artifact (bead-localized-from-export)
- `polylogue-9e5.27` — Speed up live affordance usage surface inventory (bead-localized-from-export)
- `polylogue-8jg9.1` — Standing backlog-hygiene invariant lint (bd devloop gate) (bead-localized-from-export)
- `polylogue-a7xr.11` — Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug (bead-localized-from-export)
- `polylogue-bby.8` — Web reader perceived performance: virtualized list, streamed search, optimistic navigation (bead-localized-from-export)
- `polylogue-opc` — Self-tracing: the daemon's own spans land in its own archive (bead-localized-from-export)
- `polylogue-oxz` — Performance instrumentation doctrine: slow-query log, phase timings, logging discipline (bead-localized-from-export)
- `polylogue-6wnh` — Bound thread refresh cost for large Codex appends (bead-localized-from-export)
- `polylogue-ma2` — Add FK-supporting index for web_content_constructs message cleanup (bead-localized-from-export)
- `polylogue-th0` — Interactive-surface test harness: pty flows, completions, fuzzy pickers (bead-localized-from-export)
- `polylogue-yeq` — Advanced verification lanes: metamorphic DSL, daemon chaos, API-contract walks (bead-localized-from-export)
- `polylogue-fnm.12` — User-defined query macros: named, composable DSL shorthands in user.db (bead-localized-from-export)
- … 23 more
