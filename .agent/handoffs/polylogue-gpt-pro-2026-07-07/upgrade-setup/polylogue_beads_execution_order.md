# Polylogue Beads execution order

Generated from `polylogue-beads-export.jsonl` and `polylogue-beads.json` in the uploaded archive.

Scope: open and in-progress issue beads only. Closed beads are historical/completed and are not scheduled for execution. Memory records are directive context, not executable beads.

Ordering rules used here: hard `blocks` dependencies first; blocker state inherited through parent-child hierarchy; child beads before parent/epic closure; ties by in-progress first, priority, type, then ID. `related`, `relates-to`, `discovered-from`, and `supersedes` are informational, not hard blockers.

Counts: 492 issue beads total; 397 open/in-progress scheduled here (396 open, 1 in progress); 95 closed; Beads summary says 317 ready and 79 blocked.

## Full ordered table

| # | Wave | ID | P | Type | Status | Parent(s) | Hard prerequisites still open | Bead |
| ---: | ---: | --- | ---: | --- | --- | --- | --- | --- |
| 1 | 0 | `polylogue-s7ae.6` | 1 | task | in_progress | polylogue-s7ae | — | Classify the 74%-aborted full verify from the coordination commit before deploy |
| 2 | 0 | `polylogue-8jg9.4` | 1 | bug | open | polylogue-8jg9 | — | ops doctor cleanup_orphans can delete an in-flight leased blob (the real #818) |
| 3 | 0 | `polylogue-9e5.28` | 1 | bug | open | polylogue-9e5 | — | Rigor audit iterates contracts, not the registry: uncovered number-bearing products vanish from audit |
| 4 | 0 | `polylogue-9e5.29` | 1 | bug | open | polylogue-9e5 | — | Number-over-empty gates: quantitative fields need field-level evidence contracts |
| 5 | 0 | `polylogue-9e5.30` | 1 | bug | open | polylogue-9e5 | — | Prose-mined forensic fields must carry text_derived provenance in the payload model |
| 6 | 0 | `polylogue-cpf.5` | 1 | bug | open | polylogue-cpf | — | Temporal provenance laundering: aggregates collapse to provider_ts; propagate the weakest source |
| 7 | 0 | `polylogue-cpf.6` | 1 | bug | open | polylogue-cpf | — | Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit |
| 8 | 0 | `polylogue-kwsb.1` | 1 | bug | open | polylogue-kwsb | — | Daemon/capture security hardening: Host/Origin gate, receiver token, spool governor |
| 9 | 0 | `polylogue-37t.15` | 1 | task | open | polylogue-37t | — | Single agent-write chokepoint in upsert_assertion: non-user authors => CANDIDATE + inject:false, always |
| 10 | 0 | `polylogue-83u.4` | 1 | task | open | polylogue-83u | — | Classify the 39,586 missing referenced blobs in the production backup |
| 11 | 0 | `polylogue-cfk` | 1 | task | open | — | — | Re-run two-arm uplift with freshness-fixed packs (n>=3 pairs, then n=12-20) |
| 12 | 0 | `polylogue-0v9p` | 1 | feature | open | polylogue-4smp | — | Language detection and preference facts for variant selection |
| 13 | 0 | `polylogue-83u.3` | 1 | feature | open | polylogue-83u | — | Preserve uploaded attachment bytes in live browser capture |
| 14 | 0 | `polylogue-arso` | 1 | feature | open | polylogue-4smp | — | Content variant substrate: refs, nodes, alignment, storage |
| 15 | 0 | `polylogue-bby.11` | 1 | feature | open | polylogue-bby | — | Webui architecture v2: the stack that can carry the ambition |
| 16 | 0 | `polylogue-d1y` | 1 | feature | open | polylogue-s7ae | — | polylogue hooks install: one-command harness wiring + hook liveness monitoring |
| 17 | 0 | `polylogue-pj8` | 1 | feature | open | polylogue-s7ae | — | Agent query cookbook: MCP prompts + skill recipes as the discoverability layer |
| 18 | 1 | `polylogue-s7ae.2` | 1 | task | open | polylogue-s7ae | polylogue-pj8 | Pre-deployment MCP and hook coordination batch |
| 19 | 2 | `polylogue-ahqd` | 1 | task | open | polylogue-s7ae | polylogue-s7ae.2 | Observe MCP write adoption after role rollout |
| 20 | 1 | `polylogue-rlsb` | 1 | feature | open | polylogue-4smp | polylogue-arso | Variant-aware projection, query, and reader render profiles |
| 21 | 2 | `polylogue-d4zk` | 1 | feature | open | polylogue-4smp | polylogue-arso, polylogue-rlsb | User and agent UX for creating, reviewing, and messaging about variants |
| 22 | 3 | `polylogue-4smp` | 1 | epic | open | — | — | Content variants: language-aware transformed archive objects with alignment |
| 23 | 0 | `polylogue-1xc.12` | 2 | bug | open | polylogue-1xc | — | FTS drift gauges + metamorphic coherence tests; rowid-reuse requires block_id check |
| 24 | 0 | `polylogue-20d.4` | 2 | bug | open | polylogue-20d | — | CLI structured-query routing parity with daemon (#1860): no FTS gate for non-FTS queries |
| 25 | 0 | `polylogue-4ts.3` | 2 | bug | open | polylogue-4ts | — | Distinguish subagent auto-compaction from main-session acompact |
| 26 | 0 | `polylogue-4ts.4` | 2 | bug | open | polylogue-4ts | — | Wrap lineage composition reads in a single read transaction |
| 27 | 0 | `polylogue-4ts.6` | 2 | bug | open | polylogue-4ts | — | Lineage composition silently truncates transcripts; surface a completeness signal |
| 28 | 0 | `polylogue-a7xr.1` | 2 | bug | open | polylogue-a7xr | — | Sweep remaining sqlite3 connection leaks: 'with sqlite3.connect()' commits but never closes |
| 29 | 0 | `polylogue-a7xr.2` | 2 | bug | open | polylogue-a7xr | — | Converger and repair disagree on session_profile staleness for NULL-sort-key sessions |
| 30 | 0 | `polylogue-a7xr.3` | 2 | bug | open | polylogue-a7xr | — | message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x |
| 31 | 0 | `polylogue-a7xr.6` | 2 | bug | open | polylogue-a7xr | — | parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb) |
| 32 | 0 | `polylogue-f2qv.1` | 2 | bug | open | polylogue-f2qv | — | Per-model token rollup double-count: session totals partitioned once (#2472) |
| 33 | 0 | `polylogue-f2qv.5` | 2 | bug | open | polylogue-f2qv | — | Version-gate provider-usage projection so it self-heals like session_profiles |
| 34 | 0 | `polylogue-jnj.5` | 2 | bug | open | polylogue-kwsb | — | Route ops reset --session/--source through the mutation contract |
| 35 | 0 | `polylogue-peo` | 2 | bug | open | polylogue-8jg9 | — | Daemon death leaves no trace: crash forensics + heartbeat sentinel + restart policy |
| 36 | 0 | `polylogue-rsad` | 2 | bug | open | — | — | MCP agent ergonomics: oversized responses, boilerplate affordances, metadata-only summaries |
| 37 | 0 | `polylogue-tsk` | 2 | bug | open | — | — | Resume ranking keys on workflow shapes the classifier never emits |
| 38 | 0 | `polylogue-xy95` | 2 | bug | open | polylogue-f2qv | — | Speed up provider usage full stale diagnostics |
| 39 | 0 | `polylogue-0k6` | 2 | task | open | polylogue-mhx | — | Embedding changed-text full-replace regression vs split embeddings.db metadata |
| 40 | 0 | `polylogue-0ns` | 2 | task | open | polylogue-mhx | — | Bound archive embedding work within large sessions |
| 41 | 0 | `polylogue-1vpm.1` | 2 | task | open | polylogue-1vpm | — | Delegation derived unit: materializer + query unit + delegation-card projection |
| 42 | 0 | `polylogue-1vpm.2` | 2 | task | open | polylogue-1vpm | — | Episode unit: tables, 4-signal scorer with false-merge floor, assertion-calibrated |
| 43 | 0 | `polylogue-1vpm.4` | 2 | task | open | polylogue-1vpm | — | Turn-pair unit with prompt-burst semantics (no double-claimed answers) |
| 44 | 0 | `polylogue-1xc.8` | 2 | task | open | polylogue-1xc | — | Schema rebuild-safety scenario |
| 45 | 0 | `polylogue-20d.10` | 2 | task | open | polylogue-20d | — | Runtime post-filter efficiency: memoize semantic facts; lower matchers onto the actions view |
| 46 | 0 | `polylogue-20d.14` | 2 | task | open | polylogue-20d | — | Interactive SLO tier: named latency budgets, continuously measured, regression-gated |
| 47 | 1 | `polylogue-20d.15` | 2 | task | open | polylogue-20d | polylogue-20d.14 | Bulk ingest throughput + resource envelope: parallel parse, batched writes, bounded RSS/IO |
| 48 | 2 | `polylogue-b5l` | 1 | task | open | — | polylogue-20d.15 | Blue-green index rebuilds: fresh-first without downtime |
| 49 | 0 | `polylogue-20d.2` | 2 | task | open | polylogue-20d | — | Defer heavy imports off the CLI startup path |
| 50 | 0 | `polylogue-20d.5` | 2 | task | open | polylogue-20d | — | Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL |
| 51 | 0 | `polylogue-20d.6` | 2 | task | open | polylogue-20d | — | Live full-ingest catch-up latency + WAL shape |
| 52 | 0 | `polylogue-212.1` | 2 | task | open | polylogue-212 | — | Post-hoc forensic Q&A demo: questions a tracer cannot answer |
| 53 | 0 | `polylogue-212.2` | 2 | task | open | polylogue-212 | — | D1 'The receipts': claim-vs-evidence on a real PR |
| 54 | 0 | `polylogue-212.3` | 2 | task | open | polylogue-212 | — | D2 'Where did the money actually go': cost by outcome |
| 55 | 0 | `polylogue-212.4` | 2 | task | open | polylogue-212 | — | D4 'Behavioral archaeology': six DSL queries, rapid fire |
| 56 | 0 | `polylogue-212.8` | 2 | task | open | polylogue-212 | — | The honesty anti-demo: a tempting finding that emits verdict not_supported |
| 57 | 3 | `polylogue-27m` | 2 | task | open | polylogue-kwsb | polylogue-b5l | Excision and secret hygiene: the archive can forget on purpose |
| 58 | 0 | `polylogue-37t.14` | 2 | task | open | polylogue-37t | — | Recursive-safety substrate: citation anchors, provenance edges, grounding verdicts (closed-loop/cycle/drift) |
| 59 | 0 | `polylogue-38x` | 2 | task | open | — | — | Reconcile archived audit residue against current source |
| 60 | 0 | `polylogue-3tl.12` | 2 | task | open | polylogue-3tl | — | README de-meta / de-persuasion pass with reproducible capability claims |
| 61 | 0 | `polylogue-3tl.13` | 2 | task | open | polylogue-3tl | — | Reconcile schema-versioning docs + retire superseded execution-plan.md |
| 62 | 0 | `polylogue-3tl.16` | 2 | task | open | polylogue-3tl | — | Public claims ledger: every README/launch claim carries a status and an evidence ref |
| 63 | 0 | `polylogue-3tl.4` | 2 | task | open | polylogue-3tl | — | Findings publishing lane: campaign artifacts on the docs site |
| 64 | 0 | `polylogue-3tl.7` | 2 | task | open | polylogue-3tl | — | Release is a decision: proven install matrix across package managers and OSes |
| 65 | 0 | `polylogue-3tl.9` | 2 | task | open | polylogue-3tl | — | Docs-and-visuals ownership: coverage lint + regenerable visuals as a standing devloop gate |
| 66 | 1 | `polylogue-3uw` | 2 | task | open | — | polylogue-d1y | Capture-completeness: the instrument's coverage error as a standing measure |
| 67 | 0 | `polylogue-3v1.1` | 2 | task | open | polylogue-3v1 | — | Multiple concurrent browser-capture extension instances: attribution, dedup, spool safety |
| 68 | 1 | `polylogue-3v1` | 2 | task | open | polylogue-jlme | — | Capture extension reliability + status UX: spool health, completeness, gap visibility |
| 69 | 0 | `polylogue-4be` | 2 | task | open | polylogue-8jg9 | — | Restore drill: prove the backups restore, quarterly |
| 70 | 0 | `polylogue-4p1.1` | 2 | task | open | polylogue-4p1 | — | Route daemon split-archive fast path through SessionQuerySpec.from_params |
| 71 | 1 | `polylogue-4p1` | 2 | task | open | — | — | Decision: one read algebra — Query x Projection x Render as the only read contract |
| 72 | 0 | `polylogue-60i5` | 2 | task | open | — | — | Durable-tier batch coordination: one user v4->v5 and one source v2->v3 migration window |
| 73 | 0 | `polylogue-6il` | 2 | task | open | — | — | devloop-integration --json --check consumed by devloop-review |
| 74 | 0 | `polylogue-6wnh` | 2 | task | open | polylogue-1xc | — | Bound thread refresh cost for large Codex appends |
| 75 | 0 | `polylogue-83u.6` | 2 | task | open | polylogue-83u | — | Attachment acquisition census by origin and byte volume |
| 76 | 0 | `polylogue-8jg9.1` | 2 | task | open | polylogue-8jg9 | — | Standing backlog-hygiene invariant lint (bd devloop gate) |
| 77 | 0 | `polylogue-8jg9.2` | 2 | task | open | polylogue-8jg9 | — | Blob-GC lease/orphan concurrency test (the acquire->commit race) |
| 78 | 0 | `polylogue-9e5.1` | 2 | task | open | polylogue-9e5 | — | Assertion-layer adoption audit: is the flywheel used or aspirational? |
| 79 | 0 | `polylogue-9e5.19` | 2 | task | open | polylogue-9e5 | — | Storage-layer correctness scenario family |
| 80 | 0 | `polylogue-9e5.24` | 2 | task | open | polylogue-9e5 | — | Sink MCP analysis primitives into insights/ + api facade; delete surface-side math |
| 81 | 0 | `polylogue-9e5.25` | 2 | task | open | polylogue-9e5 | — | Review zero-use MCP surfaces from affordance usage artifact |
| 82 | 0 | `polylogue-9e5.26` | 2 | task | open | polylogue-9e5 | — | Review zero-use CLI surfaces from affordance usage artifact |
| 83 | 0 | `polylogue-9e5.27` | 2 | task | open | polylogue-9e5 | — | Speed up live affordance usage surface inventory |
| 84 | 0 | `polylogue-9e5.3` | 2 | task | open | polylogue-9e5 | — | Column honesty audit: null/unknown density for key semantic columns |
| 85 | 0 | `polylogue-9e5.4` | 2 | task | open | polylogue-9e5 | — | Get->modify->put race audit across daemon/CLI/MCP writers |
| 86 | 0 | `polylogue-9l5.15` | 2 | task | open | polylogue-9l5 | — | Triage frontier: worth_reviewing_score + TRIAGED lifecycle — an inbox that empties |
| 87 | 0 | `polylogue-a7xr.5` | 2 | task | open | polylogue-a7xr | — | FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies |
| 88 | 0 | `polylogue-ap7` | 2 | task | open | polylogue-fnm | — | Semantic transcript rendering: tool-call-aware, provider-agnostic, shared CLI/web renderer registry |
| 89 | 0 | `polylogue-cpf.4` | 2 | task | open | polylogue-cpf | — | Enforce degrade-loudly: sweep silent soft-failure paths to carry a signal |
| 90 | 0 | `polylogue-exb` | 2 | task | open | — | — | Layering: substrate rings import the api facade (6 sites, 2 private-symbol reaches) |
| 91 | 0 | `polylogue-f2qv.2` | 2 | task | open | polylogue-f2qv | — | Codex disjoint-lane normalizer: decompose cached/uncached and reasoning/completion with a regression guard |
| 92 | 0 | `polylogue-f2qv.4` | 2 | task | open | polylogue-f2qv | — | Single pricing source of truth: LiteLLM catalog, drop tokencost, last-path-segment match |
| 93 | 0 | `polylogue-fnm.11` | 2 | task | open | polylogue-fnm | — | Pipeline/clause parity across units + generated support matrix |
| 94 | 0 | `polylogue-ivsc` | 2 | task | open | polylogue-f2qv | — | Classify Codex state_5 token drift outside lineage replay |
| 95 | 0 | `polylogue-jnj.1` | 2 | task | open | polylogue-jnj | — | Collapse read per-view flags into ProjectionSpec/RenderSpec algebra |
| 96 | 1 | `polylogue-1lm` | 2 | task | open | — | polylogue-jnj.1 | Composable transcript views: selector x transform x budget algebra |
| 97 | 0 | `polylogue-lio` | 2 | task | open | — | — | Align cross-repo devloop contract on beads (Sinex parity) |
| 98 | 0 | `polylogue-ma2` | 2 | task | open | polylogue-1xc | — | Add FK-supporting index for web_content_constructs message cleanup |
| 99 | 0 | `polylogue-mhx.3` | 2 | task | open | polylogue-mhx | — | Retrieval quality eval lane: measure FTS vs vector vs hybrid before believing any of them |
| 100 | 0 | `polylogue-ox0` | 2 | task | open | polylogue-fs1 | — | Codex deep integration: state DBs as authoritative source + AppServer live lane |
| 101 | 0 | `polylogue-rxdo.1` | 2 | task | open | polylogue-rxdo | — | ObjectRef expansion: query, query-run, result-set, finding, cohort, analysis, annotation-batch kinds |
| 102 | 0 | `polylogue-rxdo.2` | 2 | task | open | polylogue-rxdo | — | Content-addressed query identity + durable user.db queries/query_names/result_sets/query_edges |
| 103 | 0 | `polylogue-rxdo.3` | 2 | task | open | polylogue-rxdo | — | Query-run + result-relation telemetry in ops.db; refs on every query envelope |
| 104 | 0 | `polylogue-rxdo.4` | 2 | task | open | polylogue-rxdo | — | AssertionKind.FINDING: claims with evidence, reusing the candidate->judge lifecycle verbatim |
| 105 | 1 | `polylogue-rxdo.7` | 2 | task | open | polylogue-rxdo | polylogue-37t.15, polylogue-rxdo.1 | Annotation substrate: schema registry, annotation batches, JSONL import surface, typed value predicates |
| 106 | 0 | `polylogue-svfj` | 2 | task | open | — | — | Block content-hash citation anchors: blocks.content_hash + resolver with typed drift states |
| 107 | 1 | `polylogue-bby.15` | 2 | task | open | polylogue-bby | polylogue-svfj | Evidence basket -> citable report -> verified export (cockpit core loop) |
| 108 | 0 | `polylogue-t0p` | 2 | task | open | polylogue-l4kf | — | Capture the rest of Claude Code: todos, file-history, prompt history, debug artifacts |
| 109 | 0 | `polylogue-t46.3` | 2 | task | open | polylogue-t46 | — | Unify list/search query-spec->ArchiveStore execution across CLI, MCP, and daemon web |
| 110 | 0 | `polylogue-t46.4` | 2 | task | open | polylogue-t46 | — | Delegate daemon session-similarity KNN to SqliteVecProvider.query_by_session |
| 111 | 0 | `polylogue-t46.5` | 2 | task | open | polylogue-t46 | — | Route CLI transcript/dialogue file export through substrate read+render; delete streaming_markdown SQL path |
| 112 | 0 | `polylogue-t46.6` | 2 | task | open | polylogue-t46 | — | Fix referenced_path OR-vs-AND filter divergence and delete dead CLI stats aggregators |
| 113 | 0 | `polylogue-t46.8` | 2 | task | open | polylogue-t46 | — | MCP surface collapse: ~96 tools -> verb algebra (query/get/explain/context/assert/maintenance...) |
| 114 | 0 | `polylogue-t8t` | 2 | task | open | polylogue-s7ae | — | Agent workflow catalog: walk the seven core flows end-to-end, fix what breaks |
| 115 | 0 | `polylogue-th0` | 2 | task | open | — | — | Interactive-surface test harness: pty flows, completions, fuzzy pickers |
| 116 | 0 | `polylogue-x4s` | 2 | task | open | polylogue-rii | — | Express devloop state in Polylogue substrate (dogfood target) |
| 117 | 0 | `polylogue-x7d` | 2 | task | open | polylogue-jnj | — | Unify root query row rendering contracts |
| 118 | 0 | `polylogue-xgw` | 2 | task | open | polylogue-1xc | — | Archive schema hygiene for evidence-cockpit read paths |
| 119 | 0 | `polylogue-yeq` | 2 | task | open | — | — | Advanced verification lanes: metamorphic DSL, daemon chaos, API-contract walks |
| 120 | 0 | `polylogue-a7xr.10` | 2 | chore | open | polylogue-a7xr | — | Kill-or-adopt the search-provider lane: production bypasses the abstraction it should use |
| 121 | 0 | `polylogue-a7xr.11` | 2 | chore | open | polylogue-a7xr | — | Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug |
| 122 | 0 | `polylogue-20d.1` | 2 | feature | open | polylogue-20d | — | CLI->daemon fast path over UDS (persistent hot process) |
| 123 | 1 | `polylogue-20d.12` | 2 | feature | open | polylogue-20d | polylogue-20d.1 | Daemon result cache + post-ingest warming: precomputed answers, cursor-keyed invalidation |
| 124 | 0 | `polylogue-20d.13` | 2 | feature | open | polylogue-20d | — | Daemon push channel: SSE events for live UIs instead of polling |
| 125 | 0 | `polylogue-37t.1` | 2 | feature | open | polylogue-37t | — | Assertions: consumer wiring + lifecycle tightening for unified overlays |
| 126 | 0 | `polylogue-37t.12` | 2 | feature | open | polylogue-37t | — | Judgment queue: operator bulk review/accept/reject of candidate assertions |
| 127 | 1 | `polylogue-37t.11` | 1 | feature | open | polylogue-37t | polylogue-37t.15, polylogue-37t.12 | Context scheduler: one arbiter for everything that enters an agent's context |
| 128 | 2 | `polylogue-s7ae.3` | 1 | feature | open | polylogue-s7ae | polylogue-37t.11 | Coordination messages and subtle scheduler-mediated advisories |
| 129 | 3 | `polylogue-s7ae.5` | 1 | task | open | polylogue-s7ae | polylogue-37t.11, polylogue-s7ae.3 | Live proof: two agents, separate worktrees, one repo — overlap, message, context, handoff |
| 130 | 1 | `polylogue-37t.4` | 2 | task | open | polylogue-37t | polylogue-37t.12 | SessionStart preamble opt-in rollout (polylogue + sinnix repos) |
| 131 | 0 | `polylogue-37t.2` | 2 | feature | open | polylogue-37t | — | Inline annotation protocol: agent-authored structure in plain prose |
| 132 | 0 | `polylogue-37t.3` | 2 | feature | open | polylogue-37t | — | Reboot-with-refs: session self-compaction protocol |
| 133 | 0 | `polylogue-37t.6` | 2 | feature | open | polylogue-37t | — | Session-aware devshell entry: surface what the last agent session left behind |
| 134 | 0 | `polylogue-37t.7` | 2 | feature | open | polylogue-37t | — | Close the failure loop: verify postmortem -> next session's context seed |
| 135 | 0 | `polylogue-37t.8` | 2 | feature | open | polylogue-37t | — | Resume routing: map a session to the harness invocation that reopens it |
| 136 | 0 | `polylogue-4ts.5` | 2 | feature | open | polylogue-4ts | — | Compaction boundary-range columns + effective-context derivation |
| 137 | 1 | `polylogue-gjg.1` | 2 | task | open | polylogue-gjg | polylogue-d1y, polylogue-4ts.5 | compaction_events + compaction_loss_items derived tables; identity survives rebuild + re-ingest |
| 138 | 1 | `polylogue-gjg.2` | 2 | task | open | polylogue-gjg | polylogue-d1y, polylogue-4ts.5 | Pre-compaction snapshot capture: hook payload when available, manifest-of-refs otherwise, honesty ladder always |
| 139 | 2 | `polylogue-gjg.3` | 2 | task | open | polylogue-gjg | polylogue-d1y, polylogue-gjg.1, polylogue-gjg.2, polylogue-4ts.5 | Deterministic loss-forensics: 4-tier structural diff + lost-but-later-needed ranking |
| 140 | 0 | `polylogue-5hf` | 2 | feature | open | polylogue-f2qv | — | Provider token accounting: honest cross-provider usage ledger |
| 141 | 0 | `polylogue-7aw` | 2 | feature | open | — | — | Ingest agent configuration as a source family (skills, CLAUDE.md, hooks) |
| 142 | 1 | `polylogue-h6r` | 2 | task | open | — | polylogue-7aw | Agent identity: a stable who-did-this tuple for every session |
| 143 | 0 | `polylogue-7xv.1` | 2 | feature | open | polylogue-7xv | — | Work-trace + reproduction harness: verify a session repo-work from a clean worktree |
| 144 | 0 | `polylogue-83u.2` | 2 | feature | open | polylogue-83u | — | Attachment byte acquisition for non-inline sources (Drive/zip/local) |
| 145 | 0 | `polylogue-90y` | 2 | feature | open | polylogue-jlme | — | In-page overlay: Polylogue presence on chat sites — archive state, context, assertion capture |
| 146 | 0 | `polylogue-9l5.1` | 2 | feature | open | polylogue-9l5 | — | Outcome-conditioned analytics: cost/duration/retries/tools by structural success |
| 147 | 0 | `polylogue-9l5.2` | 2 | feature | open | polylogue-9l5 | — | Cross-provider comparative analytics |
| 148 | 0 | `polylogue-9l5.6` | 2 | feature | open | polylogue-9l5 | — | tool-episodes projection: call + result + outcome + context + next action |
| 149 | 0 | `polylogue-avg` | 2 | feature | open | — | — | Fold devloop claim-guard vocabulary upstream into ops status/readiness |
| 150 | 0 | `polylogue-bby.8` | 2 | feature | open | polylogue-bby | — | Web reader perceived performance: virtualized list, streamed search, optimistic navigation |
| 151 | 0 | `polylogue-da1` | 2 | feature | open | — | — | Provider format-drift sentinel: detect upstream export-shape changes from live ingest |
| 152 | 0 | `polylogue-dmp` | 2 | feature | open | — | — | polylogue note: zero-friction memory capture from the terminal |
| 153 | 0 | `polylogue-f2qv.3` | 2 | feature | open | polylogue-f2qv | — | Dual cost view: API-list-equivalent and subscription-credit reported separately |
| 154 | 1 | `polylogue-fnm.1` | 2 | feature | open | polylogue-fnm | polylogue-fnm.11 | Aggregates beyond count (sum/avg/min/max/percentiles) |
| 155 | 0 | `polylogue-fnm.10` | 2 | feature | open | polylogue-fnm | — | fields/select stage with parent-field projection (first real Transform) |
| 156 | 0 | `polylogue-fnm.12` | 2 | feature | open | polylogue-fnm | — | User-defined query macros: named, composable DSL shorthands in user.db |
| 157 | 0 | `polylogue-fnm.13` | 2 | feature | open | polylogue-fnm | — | Set-algebra over query results: union/intersect/except between queries |
| 158 | 0 | `polylogue-fnm.14` | 2 | feature | open | polylogue-fnm | — | find <query> \| compact: token-budgeted corpus-compaction projection with drop manifest |
| 159 | 0 | `polylogue-fnm.2` | 2 | feature | open | polylogue-fnm | — | Projection predicates/windows + render/layout stages on attached units |
| 160 | 0 | `polylogue-fnm.4` | 2 | feature | open | polylogue-fnm | — | Shell completion + fuzzy selection as read-only projections of the grammar registries |
| 161 | 0 | `polylogue-fnm.6` | 2 | feature | open | polylogue-fnm | — | Wire the terminal stage to projections: \| read / \| context-image |
| 162 | 0 | `polylogue-fs1.4` | 2 | feature | open | polylogue-fs1 | — | Report: polylogue forensics for Hermes sessions |
| 163 | 0 | `polylogue-kph` | 2 | feature | open | polylogue-s7ae | — | Provenance-carrying PRs: attach the authoring session's postmortem bundle |
| 164 | 0 | `polylogue-mhx.1` | 2 | feature | open | polylogue-mhx | — | Provider abstraction: one OpenAI-compatible embedding client, local and cloud, model registry in meta |
| 165 | 1 | `polylogue-37t.5` | 2 | feature | open | polylogue-37t | polylogue-mhx.1 | Local embedding lane via OpenAI-compatible provider (LiteLLM gateway) |
| 166 | 0 | `polylogue-mhx.2` | 2 | feature | open | polylogue-mhx | — | Embedding target policy: what gets a vector, at what granularity, at what cost |
| 167 | 1 | `polylogue-mhx.4` | 2 | feature | open | polylogue-mhx | polylogue-mhx.2 | Semantic recall leg in context compilation: the memory actually retrieves |
| 168 | 0 | `polylogue-o21` | 2 | feature | open | — | — | Extension-point ergonomics: declare-once registries, scaffolds, actionable completeness errors |
| 169 | 1 | `polylogue-2qx` | 2 | feature | open | polylogue-l4kf | polylogue-o21 | OriginSpec: one package per origin, dispatch order derived from declared strictness |
| 170 | 0 | `polylogue-opc` | 2 | feature | open | polylogue-20d | — | Self-tracing: the daemon's own spans land in its own archive |
| 171 | 0 | `polylogue-oxz` | 2 | feature | open | polylogue-20d | — | Performance instrumentation doctrine: slow-query log, phase timings, logging discipline |
| 172 | 0 | `polylogue-p5g` | 2 | feature | open | polylogue-37t | — | polylogue judge: interactive candidate triage in the terminal |
| 173 | 0 | `polylogue-ptx` | 2 | feature | open | polylogue-bby | — | Browser-capture posting channel: un-gate, with attachments |
| 174 | 0 | `polylogue-rii.1` | 2 | feature | open | polylogue-rii | — | Agent work-event write-leg -> session_events -> materialized read-models |
| 175 | 1 | `polylogue-rii.2` | 2 | feature | open | polylogue-rii | polylogue-rii.1 | Materialize hook events + OTLP spans into queryable evidence |
| 176 | 0 | `polylogue-yrx` | 2 | feature | open | polylogue-bby | — | Session changes view: per-session diff/changelog composed from edit evidence |
| 177 | 1 | `polylogue-f2qv` | 2 | epic | open | — | — | Provider usage & cost honesty: disjoint token lanes, one pricing source, dual cost view |
| 178 | 2 | `polylogue-jlme` | 2 | epic | open | — | — | Capture extension: reliability, coverage, and in-page presence |
| 179 | 0 | `polylogue-at44` | 3 | bug | open | — | — | user_settings table is dead: DDL + migration 004 exist, zero runtime read/write helpers |
| 180 | 0 | `polylogue-mhx.7` | 3 | bug | open | polylogue-mhx | — | Two live vec0 DDL definitions: unify to one canonical embeddings table-creation path |
| 181 | 0 | `polylogue-tf0e` | 3 | bug | open | — | — | Generic-messages parser fallback drops available created_at/updated_at |
| 182 | 0 | `polylogue-xnkf` | 3 | bug | open | — | — | actions view fans out on duplicate tool_ids: one logical action becomes up to NxM rows |
| 183 | 0 | `polylogue-013x` | 3 | task | open | — | — | search_text excludes Write-tool file bodies (tool_input.$.content) — undocumented coverage gap |
| 184 | 0 | `polylogue-0dz` | 3 | task | open | — | — | Chunked/streaming read-package layout for huge exports |
| 185 | 2 | `polylogue-1hj` | 3 | task | open | polylogue-s7ae | polylogue-d1y, polylogue-37t.4 | Blackboard as agent comms: cross-session messages that actually arrive |
| 186 | 4 | `polylogue-s7ae` | 1 | epic | open | — | — | Agent coordination substrate: evidence-backed multi-agent work without tracker lock-in |
| 187 | 0 | `polylogue-1vpm.3` | 3 | task | open | polylogue-1vpm | — | Generic artifact edges: produced/consumed/mentioned/reported_by/derived_from across sessions, runs, delegations |
| 188 | 0 | `polylogue-1vpm.5` | 3 | task | open | polylogue-1vpm | — | Correction-edge runtime query: resolve correction assertions to corrected blocks/tools/models |
| 189 | 1 | `polylogue-1vpm` | 2 | epic | open | — | — | Work-graph units: delegation, episode, artifact edges — the derived units between lineage and analysis |
| 190 | 0 | `polylogue-20d.11` | 3 | task | open | polylogue-20d | — | Read-profile mmap tuning: raise READ_MMAP, lower double-buffering cache |
| 191 | 0 | `polylogue-20d.16` | 3 | task | open | polylogue-20d | — | Performance/throughput scenario family |
| 192 | 0 | `polylogue-20d.7` | 3 | task | open | polylogue-20d | — | EQP sweep + dbstat census on a live-archive copy |
| 193 | 1 | `polylogue-iec` | 2 | task | open | — | polylogue-20d.7 | Schema optimization audit: storage shape earns its bytes and its reads |
| 194 | 1 | `polylogue-20d.8` | 3 | task | open | polylogue-20d | polylogue-20d.10 | Bound claim-vs-evidence regen latency (43s on live archive) |
| 195 | 2 | `polylogue-20d` | 2 | epic | open | — | — | Interactive performance: the front door answers in interactive time |
| 196 | 0 | `polylogue-212.5` | 3 | task | open | polylogue-212 | — | D5 'The session that watched itself': live capture proof |
| 197 | 1 | `polylogue-212.6` | 3 | task | open | polylogue-212 | polylogue-tsk | D8 'Pick up where I left off': abandoned-session triage to live continuation |
| 198 | 0 | `polylogue-212.9` | 3 | task | open | polylogue-212 | — | Fable-as-Foreman: subagent-delegation rhetoric report (the X-post demo) |
| 199 | 1 | `polylogue-212.7` | 2 | task | open | polylogue-212 | polylogue-212.9 | Demo Finding Packet contract + prompt runner + registry manifest |
| 200 | 2 | `polylogue-212` | 2 | epic | open | — | — | Demo portfolio: construct-valid demos (D1/D2/D4/D5/D8 + post-hoc forensic Q&A) |
| 201 | 0 | `polylogue-37t.13` | 3 | task | open | polylogue-37t | — | Revisit beads<->assertions boundary once beads-history ingestion (7fj) lands |
| 202 | 0 | `polylogue-37t.16` | 3 | task | open | polylogue-37t | — | Claim-kind -> allowed grounding-class compatibility registry |
| 203 | 0 | `polylogue-37t.17` | 3 | task | open | polylogue-37t | — | Read-access log + memory-utility analytics: which injected memories earn their tokens |
| 204 | 1 | `polylogue-3tl.10` | 3 | task | open | polylogue-3tl | polylogue-cfk | Launch kit: announcement artifacts prepared so publication is one decision |
| 205 | 0 | `polylogue-3tl.14` | 3 | task | open | polylogue-3tl | — | Fix FTS + entry-point doc drift: internals.md describes external-content FTS; CLAUDE.md misdescribes operations/archive.py |
| 206 | 0 | `polylogue-3tl.15` | 3 | task | open | polylogue-3tl | — | Anti-grep proof card: the "why not grep ~/.claude" answer, grounded in one finding |
| 207 | 0 | `polylogue-3tl.3` | 3 | task | open | polylogue-3tl | — | Claim-vs-evidence leaderboard variant (multi-model, incl. open models) |
| 208 | 0 | `polylogue-3tl.8` | 3 | task | open | polylogue-3tl | — | GitHub surface polish: the repo page itself is a landing page |
| 209 | 2 | `polylogue-4ts.7` | 3 | task | open | polylogue-4ts | polylogue-2qx | Physical session identity collision beneath origin collapse: same native_id, two source families, one row |
| 210 | 3 | `polylogue-4ts` | 1 | epic | open | — | — | Session lineage truth: shared content stored once, counted once, composed correctly |
| 211 | 0 | `polylogue-5en` | 3 | task | open | — | — | Branch-local daemon/web/extension dev loops: verify remaining AC and close out |
| 212 | 0 | `polylogue-703` | 3 | task | open | polylogue-t46 | — | One status assembly: daemon/status.py, cli/commands/status.py, and workload diagnostics converge |
| 213 | 0 | `polylogue-7le` | 3 | task | open | polylogue-t46 | — | Consolidate the three session->HTML paths |
| 214 | 0 | `polylogue-8jg9.3` | 3 | task | open | polylogue-8jg9 | — | SLO samples + idle-vs-stalled verdict: steady-state observability over convergence |
| 215 | 0 | `polylogue-9e5.10` | 3 | task | open | polylogue-9e5 | — | Resume/context efficacy eval (observational) |
| 216 | 0 | `polylogue-9e5.11` | 3 | task | open | polylogue-9e5 | — | Test-suite economics: coverage vs fix-density map |
| 217 | 0 | `polylogue-9e5.12` | 3 | task | open | polylogue-9e5 | — | Schema-inference ROI: load-bearing or gold-plated? |
| 218 | 0 | `polylogue-9e5.13` | 3 | task | open | polylogue-9e5 | — | Doc-vs-code drift diff -> one docs-correction PR |
| 219 | 0 | `polylogue-9e5.14` | 3 | task | open | polylogue-9e5 | — | Facade decomposition map: which of api/archive.py's ~126 methods each surface uses |
| 220 | 1 | `polylogue-1fp` | 3 | task | open | polylogue-t46 | polylogue-exb, polylogue-9e5.14 | Facade decomposition: split api/archive.py into per-capability protocols |
| 221 | 0 | `polylogue-9e5.15` | 3 | task | open | polylogue-9e5 | — | Dead-code and script-silo sweep: coverage-informed removal audit |
| 222 | 1 | `polylogue-9e5.16` | 3 | task | open | polylogue-9e5 | polylogue-9e5.14 | Python API parity: the library surface audited against CLI/MCP capabilities |
| 223 | 0 | `polylogue-9e5.18` | 3 | task | open | polylogue-9e5 | — | Wire atheris fuzz targets into CI |
| 224 | 0 | `polylogue-9e5.20` | 3 | task | open | polylogue-9e5 | — | Flakiness tracking + quarantine lane |
| 225 | 0 | `polylogue-9e5.21` | 3 | task | open | polylogue-9e5 | — | Mock-depth measurement |
| 226 | 0 | `polylogue-9e5.22` | 3 | task | open | polylogue-9e5 | — | Per-module coverage tracking (beyond aggregate floor) |
| 227 | 0 | `polylogue-9e5.23` | 3 | task | open | polylogue-9e5 | — | Extend coverage-manifest schema to accept bead: owners (retire gh#590 issue-refs) |
| 228 | 0 | `polylogue-9e5.5` | 3 | task | open | polylogue-9e5 | — | Exhaustive table read/write matrix -> dead-table kill list |
| 229 | 0 | `polylogue-9e5.6` | 3 | task | open | polylogue-9e5 | — | Hash-boundary census: classify every digest producer/consumer |
| 230 | 0 | `polylogue-9e5.7` | 3 | task | open | polylogue-9e5 | — | Daemon loop interaction model: lock/starvation map for the ~9 concurrent loops |
| 231 | 0 | `polylogue-9e5.8` | 3 | task | open | polylogue-9e5 | — | Provider->Origin completion map: sequenced retirement plan |
| 232 | 0 | `polylogue-9e5.9` | 3 | task | open | polylogue-9e5 | — | Heuristic accuracy benchmark: keyword classifiers vs hand-labeled truth |
| 233 | 1 | `polylogue-b0b.1` | 2 | task | open | polylogue-b0b | polylogue-9e5.3, polylogue-9e5.9 | Fix substring false-positives in work-event keyword classifier + inventory activity-type label as heuristic-tier |
| 234 | 2 | `polylogue-b0b` | 2 | task | open | — | polylogue-9e5.3, polylogue-9e5.9 | Replace remaining keyword outcome/pathology heuristics with structural evidence |
| 235 | 0 | `polylogue-9jsi` | 3 | task | open | — | — | Polish search recall: pl_fold write/query symmetry + remove_diacritics 2 + measured trigram lane |
| 236 | 0 | `polylogue-9l5.19` | 3 | task | open | polylogue-9l5 | — | Thinking-vs-doing drift: experimental coverage-gated measure of reasoning share vs tool-active share |
| 237 | 1 | `polylogue-9l5.13` | 2 | task | open | polylogue-9l5 | polylogue-9l5.19 | activity_spans materializer: edit/test/build/idle/delegate intervals with evidence tiers |
| 238 | 2 | `polylogue-9l5.14` | 3 | task | open | polylogue-9l5 | polylogue-9l5.13 | Efficiency measure pack v1: scorecard vector over spans/episodes/delegations — no magic score |
| 239 | 1 | `polylogue-9l5.7.1` | 3 | task | open | polylogue-9l5.7 | polylogue-9l5.19 | Tag Layer-0 substrate insight payloads with evidence_tier so consumers can read rule-heuristic vs structural confidence |
| 240 | 2 | `polylogue-9l5.7` | 2 | feature | open | polylogue-9l5 | polylogue-9l5.19 | Statistics substrate + measure registry: uncertainty primitives with construct-validity metadata |
| 241 | 3 | `polylogue-9l5.8` | 2 | feature | open | polylogue-9l5 | polylogue-9l5.7 | Temporal analytics: trends, rolling baselines, changepoint detection |
| 242 | 3 | `polylogue-stc` | 2 | feature | open | polylogue-9l5 | polylogue-9l5.7 | Experiment hosting: declared arms, preregistered metrics, paired analysis, agent-buildable |
| 243 | 0 | `polylogue-9yz` | 3 | task | open | polylogue-fnm | — | Named bounded-dialogue layout for operator-readable windows |
| 244 | 0 | `polylogue-a7xr.13` | 3 | task | open | polylogue-a7xr | — | api/contracts write-surface shadow adapters verify copies, not surfaces — delete or re-anchor |
| 245 | 0 | `polylogue-a7xr.16` | 3 | task | open | polylogue-a7xr | — | Table-drive the hand-aligned column triplicates in archive_tiers write/read hot core |
| 246 | 0 | `polylogue-a7xr.4` | 3 | task | open | polylogue-a7xr | — | One percentile implementation: three algorithms across five copies skew operator-facing stats |
| 247 | 0 | `polylogue-a7xr.7` | 3 | task | open | polylogue-a7xr | — | Role synonym vocabulary maintained by hand in two directions + normalize_role name collision |
| 248 | 0 | `polylogue-a7xr.8` | 3 | task | open | polylogue-a7xr | — | Index-tier sibling-path derivation pasted ~7x with divergent existence rules |
| 249 | 0 | `polylogue-bby.1` | 3 | task | open | polylogue-bby | — | Workbench responsive under slow/missing routes |
| 250 | 2 | `polylogue-bby.10` | 3 | task | open | polylogue-bby | polylogue-20d.12, polylogue-20d.13 | Timeline and firehose: the archive as a scrubbable stream |
| 251 | 0 | `polylogue-c9y` | 3 | task | open | polylogue-a7xr | — | Package topology legibility: boundary doctrine for the 28-package tree + insights/analytics vocabulary |
| 252 | 0 | `polylogue-ca4` | 3 | task | open | — | — | Decision: DuckDB as the optional OLAP engine over the archive |
| 253 | 0 | `polylogue-cpf.1` | 3 | task | open | polylogue-cpf | — | Doctrine lint: reject TEXT timestamps in new durable DDL |
| 254 | 0 | `polylogue-cpf.2` | 3 | task | open | polylogue-cpf | — | Doctrine: writer-class docstring convention + layering check |
| 255 | 0 | `polylogue-cpf.3` | 3 | task | open | polylogue-cpf | — | Doctrine: injected-context trust deny-lexicon tripwire fixture |
| 256 | 1 | `polylogue-cpf` | 2 | epic | open | — | — | Land the six doctrines: time, writers, finding-provenance, degraded-modes, non-goals, injected-context trust |
| 257 | 0 | `polylogue-dab.1` | 3 | task | open | polylogue-dab | — | Drop payload_json/search_text duplication from run-projection tables; hydrate from typed columns |
| 258 | 1 | `polylogue-dab` | 3 | task | open | polylogue-a7xr | — | Stop materializing run-projection cache rows; drop DDL after parity |
| 259 | 0 | `polylogue-dx1` | 3 | task | open | — | — | Decision: daemon HTTP substrate — hand-rolled BaseHTTPRequestHandler vs ASGI |
| 260 | 0 | `polylogue-f94` | 3 | task | open | — | — | Kill-or-commit the TUI (~373 lines of skeletal Textual screens) |
| 261 | 0 | `polylogue-fie` | 3 | task | open | polylogue-1xc | — | Decision: archive scaling doctrine — keep everything, optimize the ceilings |
| 262 | 3 | `polylogue-gjg.4` | 3 | task | open | polylogue-gjg | polylogue-d1y, polylogue-gjg.3, polylogue-4ts.5 | compaction_forgot + compaction_reground surfaces; re-grounding packs survive the next compaction |
| 263 | 4 | `polylogue-gjg` | 2 | epic | open | — | polylogue-d1y, polylogue-4ts.5 | Compaction lifecycle: pre-compaction snapshot, loss forensics, post-compaction re-grounding |
| 264 | 3 | `polylogue-h10` | 3 | task | open | polylogue-9l5 | polylogue-h6r, polylogue-9l5.7 | Prediction and calibration tracking: agents scored on what they said would happen |
| 265 | 0 | `polylogue-jnj.10` | 3 | task | open | polylogue-jnj | — | Make the completion system and DSL discoverable at point of use |
| 266 | 0 | `polylogue-jnj.11` | 3 | task | open | polylogue-jnj | — | Extend the fzf pattern from select to ambiguous-result moments |
| 267 | 0 | `polylogue-jnj.12` | 3 | task | open | polylogue-jnj | — | Empty-result guidance: 0 hits explains itself |
| 268 | 0 | `polylogue-jnj.13` | 3 | task | open | polylogue-jnj | — | Bare-invocation triage: status + five most recent sessions with one-key open |
| 269 | 0 | `polylogue-jnj.14` | 3 | task | open | polylogue-jnj | — | Bare single-token query-first dispatch: prefer subcommand-typo error over silent search |
| 270 | 0 | `polylogue-jnj.2` | 3 | task | open | polylogue-jnj | — | analyze boolean modes -> named projections; facets becomes a real verb |
| 271 | 0 | `polylogue-jnj.3` | 3 | task | open | polylogue-jnj | — | Output dialect normalization (--format/--json + --to/--out) |
| 272 | 0 | `polylogue-jnj.4` | 3 | task | open | polylogue-jnj | — | Direct `read session:REF` uses read-view semantics |
| 273 | 0 | `polylogue-jnj.6` | 3 | task | open | polylogue-jnj | — | Demo/import surface separation (import --demo -> polylogue demo) |
| 274 | 0 | `polylogue-jnj.8` | 3 | task | open | polylogue-jnj | — | Rationalize root onboarding, tutorial, and reader launcher |
| 275 | 1 | `polylogue-mhx.6` | 3 | task | open | polylogue-mhx | polylogue-mhx.3 | Embedding storage/spend efficiency: quantization, matryoshka, and scoped drain |
| 276 | 0 | `polylogue-pf1` | 3 | task | open | polylogue-a7xr | — | Sync/async divergence: diff the twin backends against the '10 known divergences' list |
| 277 | 1 | `polylogue-hiu` | 2 | task | open | — | polylogue-exb, polylogue-pf1 | Collapse storage twins onto the sync core behind an async adapter boundary |
| 278 | 0 | `polylogue-rii.3` | 3 | task | open | polylogue-rii | — | Ingest fidelity: parser fingerprints, byte-fidelity bands, unparsed-key census, round-trip bar |
| 279 | 2 | `polylogue-rvh` | 3 | task | open | polylogue-37t | polylogue-mhx.4 | Lesson reinforcement scheduling: judged memory on a forgetting curve |
| 280 | 1 | `polylogue-rxdo.5` | 3 | task | open | polylogue-rxdo | polylogue-37t.15, polylogue-rxdo.2, polylogue-rxdo.4 | StandingQueryStage: watched queries re-evaluated on convergence; deltas become candidate findings |
| 281 | 1 | `polylogue-rxdo.6` | 3 | task | open | polylogue-rxdo | polylogue-rxdo.2, polylogue-fnm.13 | DSL reference operands: from query:/result-set:/cohort: as provenance-preserving AST nodes |
| 282 | 2 | `polylogue-rxdo.8` | 3 | task | open | polylogue-rxdo | polylogue-rxdo.3, polylogue-rxdo.7 | Analysis recipes as DB-native runtime objects; YAML as import/export serialization only |
| 283 | 3 | `polylogue-rxdo` | 2 | epic | open | — | — | Analysis provenance: queries, result-sets, findings, analyses as first-class objects |
| 284 | 0 | `polylogue-t46.1` | 3 | task | open | polylogue-t46 | — | Replace showcase QA with demo-driven CLI and visual tests |
| 285 | 0 | `polylogue-t46.7` | 3 | task | open | polylogue-t46 | — | Move compose_context_preamble git enrichment into context/preamble.py |
| 286 | 2 | `polylogue-t46` | 2 | epic | open | — | — | Contracts own surfaces: delete parallel dispatch and the QA middle layer |
| 287 | 0 | `polylogue-ttu` | 3 | task | open | polylogue-3tl | — | Docs information architecture: tiered index, orphan sweep, stale-doc triage |
| 288 | 0 | `polylogue-utf` | 3 | task | open | — | — | Devtools surface economy: usage-ranked consolidation of the 67-command catalog |
| 289 | 0 | `polylogue-y4c` | 3 | task | open | polylogue-w8db | — | Configuration doctrine: great defaults, DB-backed runtime prefs, Nix module parity |
| 290 | 1 | `polylogue-6kh` | 2 | feature | open | polylogue-w8db | polylogue-fnm.12, polylogue-y4c | Query-scope preferences bundle: default time window, scope filters, logical fold |
| 291 | 0 | `polylogue-ze5` | 3 | task | open | polylogue-37t | — | Decision: user.db vocabulary — separate epistemic records from workspace state |
| 292 | 0 | `polylogue-1a9` | 3 | chore | open | polylogue-a7xr | — | Remove dead session-commit stubs + unused web-construct row + stale fuzz README |
| 293 | 0 | `polylogue-48h` | 3 | chore | open | polylogue-a7xr | — | Consolidate SQLite introspection helpers (10 copies of _table_exists and friends) |
| 294 | 0 | `polylogue-5dx` | 3 | chore | open | — | — | Dependency leverage policy: [analytics]/[ml] extras, evaluated adoptions |
| 295 | 0 | `polylogue-6bu` | 3 | chore | open | polylogue-3tl | — | Docs-site verification lane (pages cache, link integrity) |
| 296 | 0 | `polylogue-6l6` | 3 | chore | open | polylogue-3tl | — | Docs/theming/release-proof/control-plane polish |
| 297 | 1 | `polylogue-a7xr.12` | 3 | chore | open | polylogue-a7xr | polylogue-a7xr.11 | neighbor_candidates needs a 4-method protocol, not the 20-method SessionQueryRuntimeStore |
| 298 | 0 | `polylogue-a7xr.14` | 3 | chore | open | polylogue-a7xr | — | Collapse the one-operation operations-contract framework to concrete Import models |
| 299 | 0 | `polylogue-a7xr.15` | 3 | chore | open | polylogue-a7xr | — | payloads.py: generic from_row for the 74 identical-name copy lines (keeps typed wire contract) |
| 300 | 0 | `polylogue-a7xr.9` | 3 | chore | open | polylogue-a7xr | — | Mechanical helper dedup sweep: scalar coercion quadruplet, _table_exists x40, provenance vocab x6, title/tags mixin |
| 301 | 0 | `polylogue-bby.6` | 3 | chore | open | polylogue-bby | — | Interaction debt: replace window.prompt(); de-drift the JS renderer |
| 302 | 0 | `polylogue-jnj.7` | 3 | chore | open | polylogue-jnj | — | Provider-token leakage cleanup in public CLI help |
| 303 | 0 | `polylogue-jsy` | 3 | chore | open | polylogue-kwsb | — | Harden blob hash validation + drop misleading symlink check |
| 304 | 4 | `polylogue-kwsb` | 2 | epic | open | — | — | Security & privacy: the archive can forget on purpose and never leaks secrets |
| 305 | 0 | `polylogue-0aj` | 3 | feature | open | polylogue-a7xr | — | Declared write-effects chain: post-commit effects as registry entries |
| 306 | 1 | `polylogue-5wp` | 2 | task | open | — | polylogue-0aj | Insights as declared derived views: dissolve the stage, materialize by policy |
| 307 | 1 | `polylogue-1jc` | 3 | feature | open | polylogue-w8db | polylogue-20d.14, polylogue-y4c | Learned defaults: the archive proposes your configuration as judged candidates |
| 308 | 0 | `polylogue-1xc.10` | 3 | feature | open | polylogue-1xc | — | Design spike: express session insights + aggregates as declared derived views over a single refresh engine |
| 309 | 1 | `polylogue-1xc` | 1 | epic | open | — | — | Scale-hardening: bugs that only bite on real-scale archives |
| 310 | 1 | `polylogue-2n6` | 3 | feature | open | — | polylogue-37t.8 | Harness remote-control lane: drive Claude Code / Codex sessions from Polylogue surfaces |
| 311 | 0 | `polylogue-30h` | 3 | feature | open | — | — | Display titles: synthesize when the stored title is a first-prompt echo |
| 312 | 1 | `polylogue-37t.10` | 3 | feature | open | polylogue-37t | polylogue-37t.12 | Setup evolution via judged candidates: hooks/context-specs/cookbook changes proposed as evidence-linked assertions |
| 313 | 0 | `polylogue-37t.18` | 3 | feature | open | polylogue-37t | — | Second-brain entity graph: structural-vs-candidate mention split, backlinks, topic co-occurrence |
| 314 | 0 | `polylogue-37t.19` | 3 | feature | open | polylogue-37t | — | Semantic notification policy: route CONTENT signals through the existing fan-out, fatigue-controlled |
| 315 | 1 | `polylogue-37t.20` | 3 | feature | open | polylogue-37t | polylogue-37t.15 | Cross-project recall(task_hint) MCP tool: most-similar prior sessions + their lessons across all repos |
| 316 | 1 | `polylogue-37t.21` | 3 | feature | open | polylogue-37t | polylogue-37t.15 | Prompt/meta-workflow distillery: induce parametrized meta-prompts from high-value past sessions |
| 317 | 0 | `polylogue-37t.9` | 3 | feature | open | polylogue-37t | — | Agent self-experimentation rail: PROMPT_EVAL writer + context-spec variation + background candidate passes |
| 318 | 1 | `polylogue-3gd.1` | 3 | feature | open | polylogue-3gd | polylogue-d1y, polylogue-pj8 | polylogue doctor + adoption telemetry: why-zero-usage diagnosis with a relevance control |
| 319 | 2 | `polylogue-3gd` | 1 | feature | open | polylogue-37t | polylogue-d1y, polylogue-pj8 | Activation layer: the agent-side setup that makes the substrate get used at all |
| 320 | 3 | `polylogue-37t` | 2 | epic | open | — | — | Agent context/memory loop: declared claims -> judgment -> preamble -> reboot |
| 321 | 0 | `polylogue-3tl.6` | 3 | feature | open | polylogue-3tl | — | Publish the normalized session model as a versioned interchange schema |
| 322 | 1 | `polylogue-3xx` | 3 | feature | open | polylogue-w8db | polylogue-y4c | Verb-behavior and ops preferences bundle: confirmations, judge defaults, copy formats, spend, quiesce |
| 323 | 0 | `polylogue-45i` | 3 | feature | open | polylogue-3tl | — | Datasette lane: the archive as an explorable SQLite exhibit |
| 324 | 0 | `polylogue-4822` | 3 | feature | open | — | — | Curated polylogue.sdk + frozen public models: the external-consumer boundary lynchpin needs |
| 325 | 0 | `polylogue-7fj` | 3 | feature | open | polylogue-rii | — | Ingest beads issue history as a Polylogue evidence source |
| 326 | 1 | `polylogue-4c0` | 2 | task | open | polylogue-rii | polylogue-7fj | Beads-native work loop: session<->bead cross-links and archive-rendered work history |
| 327 | 2 | `polylogue-rii` | 2 | epic | open | — | — | Live substrate intake: agents write work-events; evidence materializes in-loop |
| 328 | 1 | `polylogue-7xv` | 3 | feature | open | polylogue-l4kf | — | Native git/repo awareness: session-to-commit/branch/repo correlation in Polylogue |
| 329 | 0 | `polylogue-83u.5` | 3 | feature | open | polylogue-83u | — | Blob store zstd compression (36GB -> est 5-8GB) |
| 330 | 1 | `polylogue-83u` | 1 | epic | open | — | — | Attachment & blob evidence integrity: bytes exist, are honest, and stay affordable |
| 331 | 3 | `polylogue-9l5.10` | 3 | feature | open | polylogue-9l5 | polylogue-9l5.7 | Process mining: workflow motifs, transition models, bottleneck discovery |
| 332 | 3 | `polylogue-9l5.11` | 3 | feature | open | polylogue-9l5 | polylogue-9l5.7 | Predictive advisories: calibrated classical models on structural labels |
| 333 | 3 | `polylogue-9l5.12` | 3 | feature | open | polylogue-9l5 | polylogue-9l5.7 | Information-theoretic and graph measures: redundancy, diversity, tree shapes |
| 334 | 0 | `polylogue-9l5.3` | 3 | feature | open | polylogue-9l5 | — | Pathology epidemiology: corpus-level rates and trends |
| 335 | 0 | `polylogue-9l5.4` | 3 | feature | open | polylogue-9l5 | — | Token-economy analytics: cache-lane and attention accounting |
| 336 | 0 | `polylogue-9l5.5` | 3 | feature | open | polylogue-9l5 | — | Ship opinionated saved views as product defaults |
| 337 | 3 | `polylogue-9l5.9` | 3 | feature | open | polylogue-9l5 | polylogue-9l5.7 | Survival analysis: session duration, abandonment hazard, time-to-outcome |
| 338 | 1 | `polylogue-b1n` | 3 | feature | open | polylogue-bby | polylogue-ptx | WebUI-driven posting: operator drives web chats from the workbench |
| 339 | 0 | `polylogue-bby.12` | 3 | feature | open | polylogue-bby | — | Session replay: play a session back the way it happened |
| 340 | 0 | `polylogue-bby.13` | 3 | feature | open | polylogue-bby | — | The day page: a daily narrative the operator actually reads |
| 341 | 0 | `polylogue-bby.2` | 3 | feature | open | polylogue-bby | — | Query completions + expression explain in the web search box |
| 342 | 0 | `polylogue-bby.3` | 3 | feature | open | polylogue-bby | — | Aggregate analytics views in the web UI |
| 343 | 0 | `polylogue-bby.4` | 3 | feature | open | polylogue-bby | — | Live session tailing as a first-class mode |
| 344 | 0 | `polylogue-bby.5` | 3 | feature | open | polylogue-bby | — | Long-session navigation: phases/windows/minimap |
| 345 | 2 | `polylogue-bfv` | 3 | feature | open | — | polylogue-d1y, polylogue-20d.1, polylogue-20d.12 | Advisory hooks: archive-informed PreToolUse/UserPromptSubmit responses |
| 346 | 0 | `polylogue-fnm.3` | 3 | feature | open | polylogue-fnm | — | SEQ modifiers: within:<duration> and {n,} occurrence counts |
| 347 | 0 | `polylogue-fnm.5` | 3 | feature | open | polylogue-fnm | — | topic-pack: staged multi-channel topic-lineage retrieval |
| 348 | 0 | `polylogue-fnm.7` | 3 | feature | open | polylogue-fnm | — | Generalized child-count predicates: count(unit where ...) comparisons |
| 349 | 0 | `polylogue-fnm.8` | 3 | feature | open | polylogue-fnm | — | Lineage scope operator: logical: prefix expands predicates across the session family |
| 350 | 0 | `polylogue-fs1.10` | 3 | feature | open | polylogue-fs1 | — | Spec-cards: sessions as portable benchmark items (leakage-gated export) |
| 351 | 0 | `polylogue-fs1.2` | 3 | feature | open | polylogue-fs1 | — | Importer: NeMo Relay ATOF/ATIF runtime spans |
| 352 | 0 | `polylogue-fs1.3` | 3 | feature | open | polylogue-fs1 | — | Per-source coverage/fidelity declaration for Hermes imports |
| 353 | 0 | `polylogue-fs1.5` | 3 | feature | open | polylogue-fs1 | — | Export: Atropos/eval JSONL downstream of the canonical archive |
| 354 | 2 | `polylogue-fs1.6` | 3 | feature | open | polylogue-fs1 | polylogue-37t.5 | Fully-sovereign loop demo: local Hermes -> archive -> local embeddings -> judged memory -> injection, air-gapped |
| 355 | 0 | `polylogue-fs1.7` | 3 | feature | open | polylogue-fs1 | — | Upstream native integration: lifecycle hooks / polylogue-hook support in the open-source Hermes agent |
| 356 | 0 | `polylogue-fs1.9` | 3 | feature | open | polylogue-fs1 | — | Polylogue->Sinex derived agent-trace event emitter |
| 357 | 0 | `polylogue-jnj.9` | 3 | feature | open | polylogue-jnj | — | Intentional runtime/deployment configuration surface |
| 358 | 0 | `polylogue-l4kf.1` | 3 | feature | open | polylogue-l4kf | — | polylogue-export origin + CIF envelope: import(export(A)) is a content-hash no-op |
| 359 | 1 | `polylogue-mhx.5` | 3 | feature | open | polylogue-mhx | polylogue-mhx.2 | Semantic analytics surfaces: topics/clustering, novelty, near-duplicate assist |
| 360 | 2 | `polylogue-mhx` | 2 | epic | open | — | — | Embedding substrate: provider-general, honest lifecycle, retrieval that earns its cost |
| 361 | 0 | `polylogue-scd` | 3 | feature | open | polylogue-jnj | — | Cross-surface handoff: polylogue open + copy-as-command everywhere |
| 362 | 0 | `polylogue-uiw` | 3 | feature | open | polylogue-l4kf | — | Origin breadth: enumerate the target set + generic openai-chat-shape detector |
| 363 | 0 | `polylogue-wmj` | 3 | feature | open | polylogue-l4kf | — | OTel GenAI trace export lane |
| 364 | 1 | `polylogue-0cg` | 3 | feature | open | polylogue-l4kf | polylogue-wmj | OTel GenAI semantic-conventions ingest: any instrumented agent framework becomes an origin |
| 365 | 0 | `polylogue-y0b` | 3 | feature | open | polylogue-3tl | — | Generated codebase atlas: the grok report as a rendered, drift-checked doc |
| 366 | 2 | `polylogue-3tl` | 1 | epic | open | — | — | External legibility: a stranger can understand, run, and cite Polylogue |
| 367 | 1 | `polylogue-y8w` | 3 | feature | open | polylogue-w8db | polylogue-y4c | Reading preferences bundle: per-scope views, fold budgets, rows, pager |
| 368 | 2 | `polylogue-w8db` | 2 | epic | open | — | — | Configuration doctrine + DB-backed runtime preferences |
| 369 | 1 | `polylogue-yp0` | 3 | feature | open | polylogue-a7xr | polylogue-9e5.7 | Daemon internal event bus: loops subscribe, polling retires |
| 370 | 2 | `polylogue-9e5` | 3 | epic | open | — | — | Audit lane: read-only analyses producing evidence artifacts |
| 371 | 0 | `polylogue-9l5.18` | 3 | epic | open | polylogue-9l5 | — | Missing data-model units: entity-mention, world-effect, verification-run, project, topic-cluster, cross-origin-thread |
| 372 | 2 | `polylogue-a7xr` | 3 | epic | open | — | — | Substrate consolidation: kill the storage twins and split the god-modules |
| 373 | 1 | `polylogue-jnj` | 3 | epic | open | — | — | Product surface algebra: one rule per concern across CLI/config/onboarding |
| 374 | 0 | `polylogue-s8q` | 4 | bug | open | polylogue-8jg9 | — | Make deployed Polylogue state trustworthy; captures queryable |
| 375 | 1 | `polylogue-8jg9` | 2 | epic | open | — | — | Operational resilience: recoverable, restorable, survives daemon death and deploy |
| 376 | 0 | `polylogue-2jj` | 4 | task | open | — | — | IssueBench: real issues as coding-agent effectiveness benchmarks |
| 377 | 3 | `polylogue-9l5.16` | 4 | task | open | polylogue-9l5 | polylogue-9l5.7 | Trajectory Quality Index: reward-shaping composite, never truth |
| 378 | 3 | `polylogue-9l5.17` | 4 | task | open | polylogue-9l5 | polylogue-9l5.7 | Model-drift observatory: candidate changepoints with validity gates, never causal claims |
| 379 | 4 | `polylogue-9l5` | 2 | epic | open | — | — | Outcome-grounded analytics: the archive answers 'so what' questions |
| 380 | 2 | `polylogue-c36` | 4 | task | open | — | polylogue-20d.15 | Native-compilation probe: mypyc first, only where profiles demand it |
| 381 | 0 | `polylogue-fs1.8` | 4 | task | open | polylogue-fs1 | — | Nous Chat browser-capture adapter |
| 382 | 3 | `polylogue-fs1` | 2 | epic | open | — | — | Hermes bridge: state.db + runtime spans -> canonical evidence -> forensics/eval export |
| 383 | 1 | `polylogue-gqx` | 4 | task | open | — | polylogue-20d.13, polylogue-scd | Desktop presence spike: Polylogue in the operator's ambient environment |
| 384 | 0 | `polylogue-lu1` | 4 | task | open | — | — | Ambient theming: terminal respects the environment, webui gains a theme system |
| 385 | 0 | `polylogue-wohv` | 4 | task | open | — | — | messages_fts UNINDEXED columns are write-only noise in a contentless table: drop or annotate |
| 386 | 0 | `polylogue-4g5` | 4 | feature | open | polylogue-l4kf | — | Expose the archive as an HPI module and Promnesia source |
| 387 | 0 | `polylogue-611` | 4 | feature | open | polylogue-l4kf | — | Grok (xAI) conversation export importer |
| 388 | 1 | `polylogue-7k7` | 4 | feature | open | polylogue-l4kf | polylogue-fs1.5 | Research-tooling export lane: inspect-ai / Docent formats |
| 389 | 0 | `polylogue-ale` | 4 | feature | open | polylogue-l4kf | — | External link archival: sessions cite URLs; the evidence should not rot |
| 390 | 0 | `polylogue-bby.14` | 4 | feature | open | polylogue-bby | — | Pinboard: workspaces as a spatial surface |
| 391 | 3 | `polylogue-bby` | 2 | epic | open | — | — | Web workbench: from result list to evidence cockpit |
| 392 | 2 | `polylogue-fnm.9` | 4 | feature | open | polylogue-fnm | polylogue-fnm.1, polylogue-fnm.7 | Pipeline-as-subquery composition |
| 393 | 3 | `polylogue-fnm` | 2 | epic | open | — | — | Query DSL: one grammar owns query semantics; compose instead of multiplying verbs |
| 394 | 1 | `polylogue-l4kf.2` | 4 | feature | open | polylogue-l4kf | polylogue-l4kf.1 | Federation: .well-known/ai-sessions manifest + selective content-hash sync |
| 395 | 1 | `polylogue-l4kf.3` | 4 | feature | open | polylogue-l4kf | polylogue-rxdo.4 | Outbound provenance: git notes (refs/notes/polylogue) + PR/issue citation footers + SARIF pathology export |
| 396 | 0 | `polylogue-r47` | 4 | feature | open | polylogue-l4kf | — | Obsidian/PKM export profile: sessions and findings as wiki-linked Markdown |
| 397 | 2 | `polylogue-l4kf` | 3 | epic | open | — | — | Ecosystem interop + origin breadth: more sources in, two-way citable export out |

## Per-bead intent and done state

### 1. `polylogue-s7ae.6` — Classify the 74%-aborted full verify from the coordination commit before deploy

P1 task · in_progress · parent: polylogue-s7ae
Intent: Commit 32ff31651 (coordination substrate, ~1376 LOC) merged with only verify --quick + focused tests green; the full devtools verify was aborted at 74% with scattered unclassified failures.
Execution shape: Commit 32ff31651 shipped ~1376 LOC with only verify --quick + focused tests green; full devtools verify was aborted at 74% with unclassified scattered failures.
Done when: A full devtools verify run is recorded; every failure classified (coordination-caused fixed; pre-existing referenced); s7ae deploy-clean.

### 2. `polylogue-8jg9.4` — ops doctor cleanup_orphans can delete an in-flight leased blob (the real #818)

P1 bug · open · parent: polylogue-8jg9
Intent: run_blob_gc is lease/ref/generation-safe, but the ops-doctor path (BlobStore.detect_orphans/cleanup_orphans) compares disk against caller-supplied ids only — VERIFIED LIVE 2026-07-06: blob_store.py contains zero references to pending_blob_refs/blob_refs/gc_generations.
Done when: A leased-uncommitted blob survives ops doctor cleanup in a fixture race; doctor path either delegates to run_blob_gc or applies all three invariants; 8jg9.2 test extended to the doctor path.

### 3. `polylogue-9e5.28` — Rigor audit iterates contracts, not the registry: uncovered number-bearing products vanish from audit

P1 bug · open · parent: polylogue-9e5
Intent: _RIGOR_MATRIX covers ~5 of 11 number-bearing insight products; audit.py iterates declared contracts, so a product with NO contract silently disappears from the audit instead of showing as uncovered — cost/coverage/tool/debt surfaces escape entirely.
Done when: One audit row per registered product or a justified exemption; monkeypatching a contract out yields uncovered, not omission; policy gate fails on uncovered number products.

### 4. `polylogue-9e5.29` — Number-over-empty gates: quantitative fields need field-level evidence contracts

P1 bug · open · parent: polylogue-9e5
Intent: Products can emit 0.0 (a number) when backing rows are empty/NULL — a rendered zero is a claim, and absent evidence must render as None/uncovered, never zero.
Execution shape: Anchor files: polylogue/insights/rigor.py (RigorContract ~L45, RigorVersionField ~L37), polylogue/insights/audit.py (insight_rigor_audit surface), polylogue/insights/confidence.py.
Done when: Property tests generate all-NULL rows and assert None/uncovered, never 0.0; every number-bearing contract declares denominator+provenance; a rendered insight cannot carry a quantitative claim over empty backing rows.

### 5. `polylogue-9e5.30` — Prose-mined forensic fields must carry text_derived provenance in the payload model

P1 bug · open · parent: polylogue-9e5
Intent: transforms.py mines commit SHAs / decisions / caveats / test-pass counts from prose into forensic bundles while the no-regex-over-prose rule only structurally holds for the exit-code axis; the recovery-digest incident (#2482) fixed one renderer, not the type system.
Done when: Digest from prose containing SHA+decision marks those fields text_derived while exit-code outcome stays raw_evidence; policy test fails on a forensic conclusion rendered from text-derived fields without caveat.

### 6. `polylogue-cpf.5` — Temporal provenance laundering: aggregates collapse to provider_ts; propagate the weakest source

P1 bug · open · parent: polylogue-cpf
Intent: classify_aggregate_hwm_source (temporal_source.py) launders weak timestamp provenance into provider_ts, so freshness/staleness surfaces look better-grounded than they are.
Done when: Table-driven tests over every TemporalSource pair (weakest wins); provider_ts + fallback_date aggregate emits fallback_date; leaf audit reports unjustifiable provider_ts paths.

### 7. `polylogue-cpf.6` — Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit

P1 bug · open · parent: polylogue-cpf
Intent: (1) core/dates.py:37 sets RELATIVE_BASE = datetime.now(tz=utc) PER CALL inside parse_date (verified live 2026-07-06 — the earlier 'frozen at import' claim was wrong; a long-lived daemon does NOT drift).
Done when: parse_date and query lowering accept an injected clock; since:7d under frozen_clock is deterministic and shifts only with the injected clock; no direct datetime.now in query-time parsing outside the seam (lint or grep gate); audit table enumerates every sort_key_ms/COALESCE ordering+window path with a fixed/safe/synthetic verdict; timeless sessions appear with time_confidence=synthetic instead of vanishing or pinning to 1970.

### 8. `polylogue-kwsb.1` — Daemon/capture security hardening: Host/Origin gate, receiver token, spool governor

P1 bug · open · parent: polylogue-kwsb
Intent: Three confirmed holes (red-team, multiple independent confirmations): (1) DNS REBINDING reads the whole archive — GET routes have no Host check and Origin is checked only on POST and skipped when absent, so a malicious page resolving to 127.0.0.1 can read loopback HTTP; fix = ONE central Host/Origin allowlist middleware before dispatch (must admit the web shell own-origin — breaking same-origin shell is the named risk).
Execution shape: All three holes live in polylogue/daemon/http.py: the Origin check exists only on the POST path (~L1305 headers.get Origin, skipped when absent) while GET routes (_static_get_routes ~L228, _parameterized_get_routes ~L257) have no Host/Origin gate at all — that is the DNS-rebinding read hole.
Done when: Cross-origin GET with foreign Host is refused; unauthenticated capture POST refused; forged-token POST refused; web shell + extension keep working (fixture proof); spool bounded.

### 9. `polylogue-37t.15` — Single agent-write chokepoint in upsert_assertion: non-user authors => CANDIDATE + inject:false, always

P1 task · open · parent: polylogue-37t
Intent: Known live hole (R&D-confirmed): blackboard_post lets an agent write author_kind=agent rows that land status=ACTIVE — an agent claim can self-inject as authoritative TODAY.
Execution shape: coerce_agent_authored(assertion) applied inside upsert_assertion (both storage twins — sync archive_tiers AND async mixins, per the twins trap); terminal-judged detection via existing judgment rows; deterministic-detector carve-out only via an explicit allowlist argument, not author_kind sniffing.
Done when: blackboard_post as agent lands candidate+inject:false; a rejected candidate re-upserted by an agent stays rejected; user-authored writes unaffected; both storage paths covered.

### 10. `polylogue-83u.4` — Classify the 39,586 missing referenced blobs in the production backup

P1 task · open · parent: polylogue-83u
Intent: Backup verifier warns 'referenced blobs missing: 39586'.
Execution shape: Refine this from recovery-only into an executable classification/product issue.
Done when: Every one of the 39,586 missing referenced blobs classified by the blob-reference-debt classifier (table, ref type, origin, recoverability); direct-file-recoverable subset restored via blob-reference-restore-direct with SHA-256 verification; the irrecoverable remainder documented with counts and the recovery-vs-accept decision recorded.

### 11. `polylogue-cfk` — Re-run two-arm uplift with freshness-fixed packs (n>=3 pairs, then n=12-20)

P1 task · open
Intent: Successor to campaign polylogue-jxe, which closed diagnostic-negative: raw-ref arm 8/10 vs handoff-pack 5/10, with the loss attributed to packet staleness (pack generated before later devloop work; raw-ref arm found newer beads/archive evidence).
Execution shape: Protocol: identical paired two-arm design as jxe.2/jxe.3 (same scoring rubric, blinded arms, committed comparison artifact under .agent/demos/uplift-two-arm/, cold-reader gate).
Done when: n>=3 paired runs completed under the recorded protocol (fresh continuation-time pack + live query vs raw-ref + live query); per-pair scores + paired analysis committed under .agent/demos/uplift-two-arm/; cold-reader gate on the comparison artifact; result recorded in the bead (positive, negative, or ambiguous -> three-arm follow-up decision).

### 12. `polylogue-0v9p` — Language detection and preference facts for variant selection

P1 feature · open · parent: polylogue-4smp
Intent: Why: agents should translate when useful, but the archive first needs honest language facts.
Execution shape: Add a language fact layer at block grain where practical, with message/session rollups derived from children.
Done when: Block/message/session language facts exist with confidence and provenance.

### 13. `polylogue-83u.3` — Preserve uploaded attachment bytes in live browser capture

P1 feature · open · parent: polylogue-83u
Intent: chatgpt-dom-v1 records the attachment chip (name + DOM text), not the uploaded bytes (byte_count=0) — the bytes live on provider servers.
Execution shape: Capture-side: the DOM adapter cannot see upload bytes (they live on provider servers).
Done when: - Extension/receiver architecture constraints (MV3 service worker lifecycle, receiver contract) are documented in the PR before choosing between (a) intercepting the upload request body at capture time (webRequest/fetch hook) and (b) re-fetching provider attachment URLs while the authenticated session is live.

### 14. `polylogue-arso` — Content variant substrate: refs, nodes, alignment, storage

P1 feature · open · parent: polylogue-4smp
Intent: Why: translations and other transformed content need a first-class substrate over existing public refs.
Execution shape: Implement typed models and storage for ContentVariant, VariantNode, and VariantAlignment.
Done when: Canonical types, storage DDL, repository/API read/write methods, and public ref resolution exist for variant and variant-node refs.

### 15. `polylogue-bby.11` — Webui architecture v2: the stack that can carry the ambition

P1 feature · open · parent: polylogue-bby
Intent: The roadmap now on the reader (mission control, timeline+firehose, replay, pinboard, day page, command palette, semantic renderers, SSE-live everything) cannot be built in JS-in-Python-strings, and shouldn't be built three views deep before the foundation is chosen.
Execution shape: (1) STACK DECISION with rationale: TypeScript + Preact + Vite.
Done when: Scaffold merged: typed generated API client, SSE/cache module, tokens, palette, routing; list + reader views reach parity with the old SPA on the seeded corpus (including the bby.7 ref walk) and the old SPA's list/reader are retired; devtools render webui reproduces byte-identical committed dist in CI; a coding agent added one new view (the judge queue) purely against the scaffold docs — the agent-buildability proof.

### 16. `polylogue-d1y` — polylogue hooks install: one-command harness wiring + hook liveness monitoring

P1 feature · open · parent: polylogue-s7ae
Intent: Hooks are the highest-fidelity capture channel (event-granularity, 100% coverage vs ~79% post-hoc per docs/hooks.md) and the enabling substrate for context injection — yet wiring them is manual settings.json surgery per harness, per machine, per event type (16 Claude Code events, 6 Codex), and NOTHING notices when they stop firing (harness update, moved script, broken PATH): capture silently degrades to post-hoc JSONL discovery.
Execution shape: (1) INSTALL: 'polylogue hooks install [--harness claude-code|codex] [--events recommended|all|<list>]' — idempotently merges the polylogue-hook entries into the harness settings (respects existing hooks, writes the minimal diff, --dry-run shows it); 'polylogue hooks status' shows wired-vs-recommended per harness with the exact missing entries; uninstall symmetric.
Done when: On a clean settings.json, hooks install --harness claude-code --events recommended wires the starter set; a second run produces zero diff.

### 17. `polylogue-pj8` — Agent query cookbook: MCP prompts + skill recipes as the discoverability layer

P1 feature · open · parent: polylogue-s7ae
Intent: Agents use what is in their face and skip what requires invention (jgp doctrine).
Execution shape: Three thin layers over existing capability, no new query machinery: (1) MCP prompts: register ~6 intent-named prompts (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the right tool-call sequences with cwd/repo prefilled — prompts are the MCP-native discoverability channel.
Done when: - ~6 intent-named MCP prompts are registered (resume-context, postmortem-last, decisions-about, unacknowledged-failures, sessions-touching-file, cost-of) that expand to the correct tool-call sequences with cwd/repo prefilled; total prompt count stays small (curation, not another catalog).

### 18. `polylogue-s7ae.2` — Pre-deployment MCP and hook coordination batch

P1 task · open · parent: polylogue-s7ae
Prerequisites: polylogue-pj8.
Intent: Why: the coordination program will require MCP prompt/tool updates and harness/hook rollout.
Execution shape: Audit and implement the deployment-sensitive pieces in one pass: Polylogue MCP tools/prompts for coordination views, server tool contract registration, CLI/openapi/generated schema updates, Sinnix MCP registry implications if needed, Beads git hook health detection/reporting, and hook-mediated coordination source points.
Done when: MCP prompt/tool surface for coordination is implemented or explicitly delegated to the envelope bead with no remaining predeploy MCP code gaps.

### 19. `polylogue-ahqd` — Observe MCP write adoption after role rollout

P1 task · open · parent: polylogue-s7ae
Prerequisites: polylogue-s7ae.2.
Intent: Why: polylogue-27p made full/evidence/browser agent profiles write-capable and mutation rows author-attributed, but the current Codex process predates the Home Manager activation.
Done when: A freshly launched full/evidence/browser agent session performs benign record_correction, add_tag, and blackboard_post MCP calls; the resulting archive rows carry the authoring session ref; an affordance-usage artifact/report shows those write calls; lean profile remains read-only.

### 20. `polylogue-rlsb` — Variant-aware projection, query, and reader render profiles

P1 feature · open · parent: polylogue-4smp
Prerequisites: polylogue-arso.
Intent: Why: translated or simplified content should be selectable, readable, exported, and queried through the existing read algebra, not through bespoke translation flags or export modes.
Execution shape: Extend the existing Query x Projection x Render algebra without adding a new overlay abstraction.
Done when: CLI/API/MCP/daemon query/read paths can request variants through ProjectionSpec and existing query/projection stages.

### 21. `polylogue-d4zk` — User and agent UX for creating, reviewing, and messaging about variants

P1 feature · open · parent: polylogue-4smp
Prerequisites: polylogue-arso, polylogue-rlsb.
Intent: Why: the operator wants agents to translate at will and wants to view/interact with those translations.
Execution shape: Build UX over existing addressing and coordination messages.
Done when: A user can address an object ref and request a variant-producing action through CLI/MCP and at least one web/in-page UX path.

### 22. `polylogue-4smp` — Content variants: language-aware transformed archive objects with alignment

P1 epic · open
Intent: Why: agents should be able to translate source content, annotations, and other addressable Polylogue objects for the operator, and the reader/export/query surfaces should let the operator view and interact with those translations without confusing transformed text with original evidence.
Execution shape: Core model: ContentVariant(target_ref, kind, source_language, target_language, status, coverage, composition_policy, author_ref, evidence_refs, staleness/supersession, metadata).
Done when: A typed content-variant model exists over public refs without treating variants as assertions.

### 23. `polylogue-1xc.12` — FTS drift gauges + metamorphic coherence tests; rowid-reuse requires block_id check

P2 bug · open · parent: polylogue-1xc
Intent: FTS readiness is too boolean: operators need drift MAGNITUDE and tests need to prove trigger coherence under arbitrary block mutation.
Execution shape: Anchor files: polylogue/storage/sqlite/archive_tiers/index.py ~L307-318 (messages_fts_ai/ad/au triggers; threads_fts ~L449+), polylogue/daemon/fts_startup.py (startup repair), polylogue/storage/archive_readiness.py (current boolean readiness).
Done when: Rowid-reuse test fails unless block_id equality checked; hypothesis op-sequence tests green; gauges scrape without table scans; ledger-vs-exact agreement periodically verified.

### 24. `polylogue-20d.4` — CLI structured-query routing parity with daemon (#1860): no FTS gate for non-FTS queries

P2 bug · open · parent: polylogue-20d
Intent: The daemon discriminates structured-only queries from FTS queries (http.py ~:1789-1793); the CLI calls the search path unconditionally, so structured filters pay the FTS readiness gate.
Execution shape: The daemon discriminates structured-only queries from FTS queries (polylogue/daemon/http.py ~:1789-1793); the CLI calls the search path unconditionally so structured filters pay the FTS readiness gate.
Done when: - The CLI search-vs-list site branches on structured-only vs FTS (spec.query_terms/contains_terms), mirroring the daemon http.py discriminator; structured-only queries no longer pass through the FTS readiness gate.

### 25. `polylogue-4ts.3` — Distinguish subagent auto-compaction from main-session acompact

P2 bug · open · parent: polylogue-4ts
Intent: agent-acompact-* also fires for Task-subagent self-compaction (~39/187 files <90% overlap; 9 at 0%) — parser assigns wrong parent; composition prepends the wrong transcript.
Execution shape: Code-confirmed (gh#2471): the agent-acompact-* prefix classifier assigns parent=main-session unconditionally, but ~39/187 such files are Task-subagent self-compactions (<90% content overlap with the main session; 9 at 0%).
Done when: 1.

### 26. `polylogue-4ts.4` — Wrap lineage composition reads in a single read transaction

P2 bug · open · parent: polylogue-4ts
Intent: Composition uses multiple autocommit SELECTs; a concurrent parent re-ingest between reads yields a torn transcript.
Execution shape: Code-confirmed (gh#2476): get_messages / read_archive_session_envelope / _composed_db_signatures compose via multiple autocommit SELECTs (edge read -> recursive parent read -> own read); a parent re-ingest between reads yields a torn transcript.
Done when: 1.

### 27. `polylogue-4ts.6` — Lineage composition silently truncates transcripts; surface a completeness signal

P2 bug · open · parent: polylogue-4ts
Intent: storage/sqlite/queries/message_query_reads.py:get_messages composes a prefix-sharing child's full logical transcript (parent prefix up to branch_point + child tail).
Done when: get_messages / read_archive_session_envelope returns (or the envelope carries) a completeness indicator; a depth>64 chain and a dangling-branch-point session both report lineage_complete=false with a reason, and the depth-limit hit is logged.

### 28. `polylogue-a7xr.1` — Sweep remaining sqlite3 connection leaks: 'with sqlite3.connect()' commits but never closes

P2 bug · open · parent: polylogue-a7xr
Intent: Python's sqlite3 Connection context manager commits/rolls back the TRANSACTION on __exit__ but does NOT close the connection — a well-known trap.
Done when: Every 'with sqlite3.connect(...)' in non-test polylogue/ either closes via contextlib.closing/try-finally or is justified; a ResourceWarning-as-error test run over the coordination-envelope and user_state_resolver hot paths shows no leaked connections.

### 29. `polylogue-a7xr.2` — Converger and repair disagree on session_profile staleness for NULL-sort-key sessions

P2 bug · open · parent: polylogue-a7xr
Intent: VERIFIED LIVE 2026-07-06 (divergence audit): daemon/convergence_stages.py:829-836 and storage/repair.py:566-584 encode DIFFERENT staleness predicates for the same derived rows.
Execution shape: One session_profile_stale_predicate(sessions_alias, profile_alias) -> str SQL-fragment builder in storage/insights/session/runtime.py (next to SESSION_INSIGHT_MATERIALIZATION_TYPES); both convergence_stages.py and repair.py compose their queries from it; repair's UNION arms for session_latency_profiles reuse the same fragment with the lp alias.
Done when: rg shows exactly one definition of the staleness predicate; a fixture with sort_key_ms NULL + source_sort_key set is classified identically by a convergence pass and an ops repair pass (regression test asserting agreement); no repair churn on a converged archive (idempotence test: repair immediately after convergence selects zero rows).

### 30. `polylogue-a7xr.3` — message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x

P2 bug · open · parent: polylogue-a7xr
Intent: VERIFIED LIVE 2026-07-06: storage/message_type_backfill.py:54-64 claims (comment) to concatenate block text in position order, but its GROUP_CONCAT has no inner ORDER BY — SQLite GROUP_CONCAT is unordered, so the #839 classifier can receive scrambled prose.
Execution shape: message_prose_sql(alias, *, separator, block_types, min_chars) fragment builder next to archive_embeddable_message_where (the factoring pattern already proven there); backfill gains ordered concatenation via correlated subquery (SELECT GROUP_CONCAT(text, sep) FROM (SELECT text FROM blocks WHERE message_id=m.message_id AND ...
Done when: One builder; backfill output for a multi-block fixture is position-ordered (regression test with 3+ blocks inserted out of order); block_type filter applied on the classifier path; embeddings selection output unchanged (golden).

### 31. `polylogue-a7xr.6` — parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb)

P2 bug · open · parent: polylogue-a7xr
Intent: Divergence audit: identical _parse_archive_datetime copies in context/selection.py:285, mcp/archive_support.py:492, cli/read_views/standard.py:232, api/archive.py:514, archive/query/archive_execution.py:113 (naive stays naive; empty string raises) vs a DIVERGENT copy in storage/insights/session/rebuild.py:763 (empty->None; naive FORCED to UTC).
Execution shape: core/timestamps.py is the designated home (docstring: unified timestamp parsing, all operations UTC): add parse_archive_datetime() with the rebuild copy's UTC-forcing semantics (matches the module contract) + iso_from_epoch_ms(); delete all copies.
Done when: One definition each; all six+five sites import core/timestamps; a test asserts the parsed value is ALWAYS tz-aware UTC; no naive-vs-aware comparison remains reachable (grep + focused tests).

### 32. `polylogue-f2qv.1` — Per-model token rollup double-count: session totals partitioned once (#2472)

P2 bug · open · parent: polylogue-f2qv
Intent: PROBLEM.
Done when: On a synthetic two-model session, per-model rollups partition the session total exactly (sum of per-model == session total, no model row holds the full total); a regression test locks this.

### 33. `polylogue-f2qv.5` — Version-gate provider-usage projection so it self-heals like session_profiles

P2 bug · open · parent: polylogue-f2qv
Intent: PROBLEM: session_model_usage (provider token/cost rollup) is materialized once at ingest (polylogue/storage/sqlite/archive_tiers/write.py:618) and is NOT in the insight rebuild path (absent from storage/insights/session/rebuild.py and insights/registry.py).
Done when: 1) Provider-usage rollups carry a materializer version and a stale check reachable from the periodic session-insight convergence loop.

### 34. `polylogue-jnj.5` — Route ops reset --session/--source through the mutation contract

P2 bug · open · parent: polylogue-kwsb
Intent: Identity resets tombstone directly before the preview/confirmation branch — a typo mutates suppression state without dry-run or JSON evidence.
Execution shape: Audit-confirmed: ops reset --session/--source tombstones BEFORE the preview/confirmation branch in the reset command implementation (cli/commands/ reset path).
Done when: - `polylogue ops reset --session <ref>` and `--source <ref>` print a dry-run of the exact target rows (origin/native_id + counts) BEFORE any tombstone write; no mutation occurs without `--yes` (code path confirmed: tombstone no longer runs before the preview/confirmation branch — grep the reset command implementation).

### 35. `polylogue-peo` — Daemon death leaves no trace: crash forensics + heartbeat sentinel + restart policy

P2 bug · open · parent: polylogue-8jg9
Intent: During read-only serving (run --no-watch --no-source-catchup --no-browser-capture) the daemon terminated twice within minutes; the log simply stops mid-work (last lines: routine embed/insights progress), no traceback, no 'Killed', nothing in ops.db.
Execution shape: (1) faulthandler.enable() + SIGTERM/SIGINT handlers that log signal + active thread stacks to the run log AND an ops.db daemon_lifecycle row (started/stopped/signal/last_heartbeat) before exit; an atexit sentinel distinguishes clean stop from vanish.
Done when: 1.

### 36. `polylogue-rsad` — MCP agent ergonomics: oversized responses, boilerplate affordances, metadata-only summaries

P2 bug · open
Intent: Field report from a Sinex-side agent doing design archaeology over the archive (2026-07-06).

### 37. `polylogue-tsk` — Resume ranking keys on workflow shapes the classifier never emits

P2 bug · open
Intent: Fables architecture pass: find_resume_candidates ranks on workflow-shape labels that the current shape classifier no longer emits — the ranking silently degrades to its fallback terms.
Execution shape: find_resume_candidates ranks on workflow-shape labels the current shape classifier no longer emits, so ranking silently falls back to its remaining terms.
Done when: 1.

### 38. `polylogue-xy95` — Speed up provider usage full stale diagnostics

P2 bug · open · parent: polylogue-f2qv
Intent: During polylogue-4ts.2, polylogue analyze usage --origin codex-session --detail full --limit 20 --format json entered D-state and had to be terminated.
Execution shape: Profile provider_usage_report_from_connection(detail='full', origin='codex-session') by stage.
Done when: On the active archive, the Codex full usage diagnostic either completes within an agreed interactive budget or exposes separately selectable expensive sections; no D-state wait in the normal stale-rollup path; tests cover reasoning-only rows and the optimized stale-rollup result.

### 39. `polylogue-0k6` — Embedding changed-text full-replace regression vs split embeddings.db metadata

P2 task · open · parent: polylogue-mhx
Intent: Changed-text reindexing for the same message_id needs an explicit full-replace regression against split embeddings.db metadata (index-tier rows cleared, embeddings tier not).
Execution shape: Step 1 — QUANTIFY on the live archive (fables analysis 9): count sessions whose updated_at_ms postdates embedding_status.last_embedded_at_ms with unchanged message counts — the concrete stale-vector population the original bug produced; record the number in the bead on completion (it doubles as the fix's impact statement).
Done when: 1.

### 40. `polylogue-0ns` — Bound archive embedding work within large sessions

P2 task · open · parent: polylogue-mhx
Intent: Why: while verifying live daemon convergence on 2026-07-04, a forced embedding debt drain could run longer than the outer daemon session window because _embed_archive_sessions_sync checks _DAEMON_EMBED_STOP_AFTER_SECONDS only between sessions, while embed_archive_session_sync can process a very large session internally.
Execution shape: Make archive embedding bounded within a single large session so a forced debt drain cannot exceed the daemon window.
Done when: 1.

### 41. `polylogue-1vpm.1` — Delegation derived unit: materializer + query unit + delegation-card projection

P2 task · open · parent: polylogue-1vpm
Intent: First-class delegations rows in index.db (derived, recomputable, extractor-versioned): delegation identity prefers (parent_session_id, tool_use_block_id) — never prompt text (identical prompts are different delegations).
Done when: Fixtures: Claude Task pair, acompact exclusion, Codex spawn, unresolved child, no false subagent from forked_from_id; delegations where parent.repo:X and status:failed works; card renders bounded (full prompts only under explicit opt-in); index bump batched.

### 42. `polylogue-1vpm.2` — Episode unit: tables, 4-signal scorer with false-merge floor, assertion-calibrated

P2 task · open · parent: polylogue-1vpm
Intent: episodes / episode_members / episode_edges in index.db.
Done when: Deliberately under-stitches on first corpus (polylogue repo work first — strongest evidence density); zero candidate-only merges in default render; edge evidence auditable; operator decisions survive rebuild; episodes where member.origin:chatgpt and member.origin:claude-code returns cross-tool episodes.

### 43. `polylogue-1vpm.4` — Turn-pair unit with prompt-burst semantics (no double-claimed answers)

P2 task · open · parent: polylogue-1vpm
Intent: Per-turn latency/cost/correction-rate needs a prompt->answer relation, and the naive pairing law (each prompt -> MIN(next assistant)) is WRONG: two human messages before one answer both claim it.
Done when: human->human->assistant yields ONE pair with burst_size=2; tool rows skipped; trailing burst abandoned=true; latency NULL-safe; turn-pairs where answer_model:X works cross-surface.

### 44. `polylogue-1xc.8` — Schema rebuild-safety scenario

P2 task · open · parent: polylogue-1xc
Intent: scenario-coverage.yaml gap 'schema-rebuild-safety' orphaned on gh#590.
Done when: A rebuild-safety scenario resets a derived tier and rebuilds from source, asserting byte/row parity + no user.db loss; a durable additive migration round-trips behind the backup gate.

### 45. `polylogue-20d.10` — Runtime post-filter efficiency: memoize semantic facts; lower matchers onto the actions view

P2 task · open · parent: polylogue-20d
Intent: matches_action_sequence, matches_referenced_path, and category matching each call _actions_for(session) -> build_session_semantic_facts (runtime_matching.py:20-25) — full semantic-fact construction over a hydrated session, no memoization across the three matchers, applied as list-comprehension post-filter (runtime_filters.py:188-189).
Execution shape: Minimal fix: memoize facts per session within a filter pass (functools cache keyed per pass, or attach _semantic_facts to the Session object).
Done when: 1.

### 46. `polylogue-20d.14` — Interactive SLO tier: named latency budgets, continuously measured, regression-gated

P2 task · open · parent: polylogue-20d
Intent: 'Snappy' needs numbers or it regresses silently.
Execution shape: (1) BUDGETS (starting points, tune with evidence): daemon-served query p50<100ms/p95<400ms; completion round trip <50ms (it is keystroke-path); status/facets from cache <30ms; webui first meaningful paint <300ms warm daemon; cold CLI (no daemon) <700ms after 20d.2 import deferral; ingest-to-searchable <10s from JSONL write (measures the whole hook->ingest->FTS->cache-invalidate chain).
Done when: slo-catalog.yaml contains the interactive tier with the stated budgets.

### 47. `polylogue-20d.15` — Bulk ingest throughput + resource envelope: parallel parse, batched writes, bounded RSS/IO

P2 task · open · parent: polylogue-20d
Prerequisites: polylogue-20d.14.
Intent: Live evidence 2026-07-03: the full index rebuild replayed 16,725 raw rows at 12-15 rows/s whole-run (5/s when it hit big sessions) — 20-40 minutes of archive downtime for an operation the fresh-first doctrine treats as routine.
Execution shape: (1) MEASURE first on a live-archive copy: where do the 12-15 rows/s go (parse vs store vs FTS vs insights — the attempt rows record stage timings); bench ingest-throughput gives the synthetic baseline.
Done when: Full replay of a live-archive copy sustains >=100 raw rows/s whole-run on the operator machine and finishes <5 min; rebuild prints live rows/s and ETA.

### 48. `polylogue-b5l` — Blue-green index rebuilds: fresh-first without downtime

P1 task · open
Prerequisites: polylogue-20d.15.
Intent: Operator re-think directive (2026-07-03): the fresh-first doctrine's CORRECTNESS half is right (no in-place migration chains, derived tiers rebuild from source) but its OPERATIONAL half is wrong — a schema bump currently means 'ops reset --index && polylogued run' with the archive degraded for the whole rebuild (observed live: 20-40 min at 12-15 rows/s, every surface confused meanwhile).
Execution shape: (1) MECHANISM: index tier gets generation-suffixed files (index.g42.db); a tiny pointer (ops.db row + symlink for external readers) names the active generation.
Done when: A schema bump on the seeded corpus (and then the live archive) completes with zero failed queries: reads served continuously from the old generation, swap under 100ms write pause, delta replayed, old generation reaped.

### 49. `polylogue-20d.2` — Defer heavy imports off the CLI startup path

P2 task · open · parent: polylogue-20d
Intent: ~2s import tax per invocation; also the residual cold cost when the daemon path is absent.
Execution shape: Measure first: python -X importtime -c 'from polylogue.cli.click_app import main' 2>&1 | sort -t'|' -k2 -rn | head -30.
Done when: - `python -X importtime -c 'from polylogue.cli.click_app import main'` shows surfaces/payloads and api/archive no longer imported on the `polylogue <cmd> --help` path.

### 50. `polylogue-20d.5` — Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL

P2 task · open · parent: polylogue-20d
Intent: Residue of the streaming-export slice: lineage-composed transcript streaming falls back to the eager path; read --view messages --full --to file lacks a true writer/iterator renderer; material-origin-filtered pagination is eager until SQL owns the predicate.
Execution shape: Three eager fallbacks to close (prior-audit evidence, re-locate): (1) lineage-composed transcript streaming falls back to the eager path — extend the streaming writer landed in a9dc3f274 to composed (parent-prefix + tail) reads; (2) read --view messages --full --to file lacks a true writer/iterator renderer — same pattern; (3) material-origin-filtered message pagination hydrates eagerly until SQL owns the predicate — push material_origin into th…
Done when: - Lineage-composed transcript streaming uses the streaming writer (extend the a9dc3f274 pattern) for composed (parent-prefix + tail) reads — no eager full-materialization fallback remains (grep the composed read path).

### 51. `polylogue-20d.6` — Live full-ingest catch-up latency + WAL shape

P2 task · open · parent: polylogue-20d
Intent: 0.2 files/s full-ingest chunks; parse_s ~274s for 50 small files.
Execution shape: Live evidence (gh#2391): full-ingest chunks ~0.2 files/s; 50 small files -> parse_s ~274s while convergence <2s; WAL ballooned during a 50-file chunk.
Done when: - RE-MEASURE first (recent daemon backoff commits changed the shape): bounded catch-up run + stage timings + `polylogue ops diagnostics workload` before/after are captured and the baseline recorded.

### 52. `polylogue-212.1` — Post-hoc forensic Q&A demo: questions a tracer cannot answer

P2 task · open · parent: polylogue-212
Intent: The category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live — when did the bad assumption first enter; which file churned before the regression; what evidence did the agent cite for a design choice; which prior failed attempts resemble today's failure.
Execution shape: A category-separation demo: take one completed multi-hour coding-agent session and answer post-hoc questions live, when the bad assumption first entered; which file churned before the regression; what evidence the agent cited for a design choice; which prior failed attempts resemble today's.
Done when: 1.

### 53. `polylogue-212.2` — D1 'The receipts': claim-vs-evidence on a real PR

P2 task · open · parent: polylogue-212
Intent: Pick a merged agent-authored PR; resolve PR -> authoring session via session_commits/session_repos; get_postmortem_bundle; render two columns: claimed (PR-body sentences: 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration — drillable to the raw tool_result block).
Execution shape: A demo: pick a merged agent-authored PR, resolve PR->authoring session via session_commits/session_repos, run get_postmortem_bundle, and render two columns, claimed (PR-body sentences like 'tests pass') vs observed (actions rows: the pytest invocation, exit_code, duration, drillable to the raw tool_result block).
Done when: 1.

### 54. `polylogue-212.3` — D2 'Where did the money actually go': cost by outcome

P2 task · open · parent: polylogue-212
Intent: Five-axis cost basis shown honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot nobody else can do: cost by outcome — '$N this month; X% spent in sessions that ended abandoned or with a failing final action; five most expensive failures, click through to the exact turn.' Needs the outcome-conditioned join (action outcome fields bead); instruments otherwise exist (cost_rollups, session_costs, terminal-state profiles, per-origin exact/estimate labels rendered…
Execution shape: A demo: show the five-axis cost basis honestly (provider-reported exact vs catalog-priced with stated coverage), then the pivot no chat UI can do, cost by outcome: total monthly spend, the % spent in sessions that ended abandoned or with a failing final action, and the five most expensive failures each drillable to the exact turn.
Done when: 1.

### 55. `polylogue-212.4` — D4 'Behavioral archaeology': six DSL queries, rapid fire

P2 task · open · parent: polylogue-212
Intent: Each answers a question an engineering lead would ask, each impossible in any chat UI: SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); near:'race condition' semantic probe across providers; abandoned-in-this-repo-this-quarter; then pipe straight into read.
Execution shape: A demo: six DSL queries, each answering a question an engineering lead would ask and each impossible in a chat UI, SEQ thrash-loop hunt; failure-rate by model; which tools break (observed-event outcomes by tool); a `near:'race condition'` semantic probe across providers; abandoned-in-this-repo-this-quarter; then a query piped straight into `read`.
Done when: 1.

### 56. `polylogue-212.8` — The honesty anti-demo: a tempting finding that emits verdict not_supported

P2 task · open · parent: polylogue-212
Intent: Ship a demo whose SUCCESS is refusal: attempt a tempting claim (e.g.
Execution shape: Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is not_supported.
Done when: Anti-demo packet passes the packet lint with not_supported verdict; report names each missing capability with the bead ref that would supply it; included in the registry manifest and the public mini-portfolio.

### 57. `polylogue-27m` — Excision and secret hygiene: the archive can forget on purpose

P2 task · open · parent: polylogue-kwsb
Prerequisites: polylogue-b5l.
Intent: Keep-everything is doctrine; 'cannot remove that API key I pasted in 2025' is a bug in that doctrine's shadow.
Execution shape: Excision mechanics: source.db rows get a redacted_at + reason tombstone with the payload replaced by a hash-of-removed marker (content hash boundary: the session's content_hash is recomputed and the old hash recorded on the tombstone so idempotency does not resurrect it); derived tiers rebuild the affected session (blue-green machinery makes this cheap); blobs: reference-counted delete with GC-lease discipline; embeddings: delete rows by message…
Done when: Excising a seeded session's message removes it from every tier (verified by grep across source/index/FTS/embeddings/blob refs) and leaves the tombstone; re-ingesting the original source does not resurrect it; the secret scanner flags a seeded fake credential as a candidate without logging its value; retro scan on the live archive produces a bounded candidate list.

### 58. `polylogue-37t.14` — Recursive-safety substrate: citation anchors, provenance edges, grounding verdicts (closed-loop/cycle/drift)

P2 task · open · parent: polylogue-37t
Intent: THE load-bearing safety invariant for a self-ingesting archive (browser capture auto-ingests the operator R&D chats; distilled findings become assertions; assertions can inject into future context — the recovery-digest fabrication class generalizes to: an agent claim laundered through other agent claims until it re-enters context as truth).
Execution shape: Derived verdict tier (fail-closed when absent) + durable policy mutation on quarantine (candidate + inject:false + quarantine_reason in context_policy; NO new AssertionStatus axis — reuse candidate machinery).
Done when: The laundering scenario is structurally blocked in a test: agent assertion citing only agent sessions/assertions never appears in compiled context; adding a git/tool-result/human-judgment anchor or judging it releases it; cycle + drift each independently quarantine; a transcript-only anchor cannot release a world-claim (compatibility matrix test).

### 59. `polylogue-38x` — Reconcile archived audit residue against current source

P2 task · open
Intent: Older archived audits under .agent/archive/conductor-history/2026-07-01 still contain valuable findings that are not all represented as executable Beads.
Execution shape: Run this as a source-grounded reconciliation pass, not as implementation by memory.
Done when: A current-source reconciliation table exists for every seed finding from the archived construct-validity/fanout/insights audits; every still-live finding is linked to an owning executable Beads issue or split into one; every stale/fixed finding cites current source or tests; no archived audit item in the seed list remains only as untriaged markdown; bd ready no longer depends on reading .agent/archive to discover these issues.

### 60. `polylogue-3tl.12` — README de-meta / de-persuasion pass with reproducible capability claims

P2 task · open · parent: polylogue-3tl
Intent: Raw-log 2026-07-04 18:16-18:21 (post-dates the closed 3tl.1 skim-ladder rewrite): strip the meta/persuasion register from the README, define agent-coined terms (judged notes, work phases, logical session) on first use, and make each capability claim reproducible on the operator's own archive (a command the reader can run).
Done when: README first screen names the category and four verbs without persuasion register; every coined term is defined at first use; each capability claim links a runnable `polylogue`/`devtools` command; a fresh no-context reader can reproduce >=2 claims.

### 61. `polylogue-3tl.13` — Reconcile schema-versioning docs + retire superseded execution-plan.md

P2 task · open · parent: polylogue-3tl
Intent: architecture-spine.md:34-37 lists 'in-place upgrade chains' as Rejected with no durable-additive carve-out, contradicting the shipped migrations and internals.md's own two-regime text (internals.md:284-289 is internally inconsistent).
Done when: architecture-spine + internals schema-versioning sections describe the two-regime model consistently; execution-plan.md is archived/removed and no doc calls it current; README points at Beads.

### 62. `polylogue-3tl.16` — Public claims ledger: every README/launch claim carries a status and an evidence ref

P2 task · open · parent: polylogue-3tl
Intent: Turn radical honesty into a product surface: every public claim (README, docs site, launch post, category one-liner) must be exactly one of proven (backed by a finding/proof artifact), capability (code exists, no measured-result claim), aspirational (roadmap only), or retired (no longer true).
Execution shape: A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date.
Done when: claims.yml exists and covers every quantitative/comparative claim in README + docs site; CI gate rejects unreferenced claims; each status has at least one real entry or an explicit none; the flight-recorder category claim itself is ledgered (initially capability, not proven).

### 63. `polylogue-3tl.4` — Findings publishing lane: campaign artifacts on the docs site

P2 task · open · parent: polylogue-3tl
Intent: The Pages pipeline already builds and deploys the docs site on master push; give campaign artifacts (claim-vs-evidence finding, forensics report) a publishing lane there — rendered report + reproduction instructions, regenerated from the seeded corpus so nothing private ships.
Execution shape: Publishing lane = a devtools render surface, not an ad-hoc workflow.
Done when: 1.

### 64. `polylogue-3tl.7` — Release is a decision: proven install matrix across package managers and OSes

P2 task · open · parent: polylogue-3tl
Intent: Release machinery exists (release-please, PyPI + Homebrew tap + GHCR + Nix flake wired in CI per the grok evidence) but 'wired' is not 'proven': nobody continuously verifies that a stranger's install actually works on each lane, so the first real user on each path is the test.
Execution shape: (1) INSTALL-MATRIX CI (scheduled weekly + pre-release, not per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, docker run from GHCR, nix run — on ubuntu + macos runners (arm+x86 where available); each job runs the same smoke: install -> polylogue demo seed -> one find -> one read -> version check.
Done when: 1.

### 65. `polylogue-3tl.9` — Docs-and-visuals ownership: coverage lint + regenerable visuals as a standing devloop gate

P2 task · open · parent: polylogue-3tl
Intent: The operator wants agents to comprehensively OWN external-facing docs and visual material, not touch them opportunistically.
Execution shape: (1) COVERAGE LINT (devtools verify docs-coverage): every public CLI command/verb, MCP tool, config key, and daemon route must be reachable from the docs tree (generated inventories make this a set diff, same pattern as the topology gate); new-surface-without-docs fails the lane with the exact missing entry named (actionable-error discipline from o21).
Done when: 1.

### 66. `polylogue-3uw` — Capture-completeness: the instrument's coverage error as a standing measure

P2 task · open
Prerequisites: polylogue-d1y.
Intent: Convergence legibility answers 'how converged is what we ingested'; nothing answers 'how much of what EXISTS did we ingest'.
Execution shape: Three evidence sources joined against the archive: hook events (SessionStart without a matching archived session after a grace window = a miss), watcher-root file inventories (files seen vs raw rows), extension capture-gap events (3v1).
Done when: Coverage renders per origin on the live archive with the known-miss list drillable to refs; a seeded missed-session scenario trips the health alert; findings' sample-frame stanzas can cite the coverage number for their window.

### 67. `polylogue-3v1.1` — Multiple concurrent browser-capture extension instances: attribution, dedup, spool safety

P2 task · open · parent: polylogue-3v1
Intent: Raw-log 2026-07-04 19:00: with agent-private + live Chrome both able to run the capture extension against the single loopback receiver, >1 instance can post concurrently.
Done when: Two simultaneous extension instances posting the same session produce one archived session (dedup by content hash); each capture carries an attributable instance id; concurrent spool writes never corrupt or interleave a spool file (test with 2 simulated posters).

### 68. `polylogue-3v1` — Capture extension reliability + status UX: spool health, completeness, gap visibility

P2 task · open · parent: polylogue-jlme
Intent: The extension (MV3, popup + badge, content bridges for chatgpt.com/claude.ai/grok.com/x.com) works end-to-end but its trust surface is thin: the operator cannot tell at a glance whether a given chat is fully captured, partially captured, or silently missed; receiver-down behavior (daemon stopped — which is common, it is loopback-only) and retry/spool state are invisible; gemini.google.com is absent from host_permissions entirely despite Gemini being a supported archive origin; and capture failu…
Execution shape: (1) PER-TAB TRUTH in popup + badge: for the active chat — captured-through timestamp/message-count vs what the page shows, pending-spool count, last receiver contact; badge encodes three states (green current / yellow spooled-waiting / red capture-error) instead of generic activity.
Done when: Badge shows current/spooled/error states per tab.

### 69. `polylogue-4be` — Restore drill: prove the backups restore, quarterly

P2 task · open · parent: polylogue-8jg9
Intent: Three backup layers exist (btrbk, polylogue-sqlite-backup, source-tier doctrine); none has ever been restore-tested.
Execution shape: A devtools lane (or ops command): restore the latest backup set to a scratch root, run integrity_check per tier, run a 10-query battery (counts, one find, one read, one insight read), compare counts against the live archive within expected-lag tolerance, record timing + result as an ops artifact.
Done when: One full restore executed from real backups with the battery green and timing recorded; the lane is invocable as one command; the quarterly timer is wired sinnix-side; a deliberately corrupted scratch restore fails loudly.

### 70. `polylogue-4p1.1` — Route daemon split-archive fast path through SessionQuerySpec.from_params

P2 task · open · parent: polylogue-4p1
Intent: polylogue/daemon/http.py:1970 documents that the split-archive fast path intentionally does not construct a SessionQuerySpec and hand-mirrors every public structured filter (has_paste, has_tool_use, has_thinking, repo, has_type, tool/exclude_tool, action/exclude_action/action_sequence/action_text, referenced_path, cwd_prefix, title, min/max_messages, min/max_words, plus the shared _filter_kw block).
Done when: The daemon split-archive list/search/count path derives all structured filters from a SessionQuerySpec built via from_params (no per-field re-read of HTTP params for filters the spec already models); a test enumerates SessionQuerySpec filter attributes and fails if the fast path drops any; the in-code 'must mirror those public params here' comment and its manual mirroring block are removed; render surfaces (openapi/cli-output-schemas) still veri…

### 71. `polylogue-4p1` — Decision: one read algebra — Query x Projection x Render as the only read contract

P2 task · open
Intent: The read side has N surfaces multiplying independently: CLI verbs with per-view flags, analyze boolean modes, MCP tools (~61 read tools, many being named parameterizations of the same underlying read), web routes, read-view profiles, read-package layouts.
Execution shape: Doctrine bead, deliverable is the recorded contract + a conformance inventory, not a rewrite: (1) Write the algebra down in docs/architecture-spine.md — the three spec types, their composition law, and the rule that MCP tools / analyze modes / CLI views / web routes are NAMED PRESETS over the algebra (a preset = a (Q,P,R) triple with defaults, registered via the declare-once machinery).
Done when: docs/architecture-spine.md gains a 'One read algebra' entry under Major Decisions naming SelectionSpec x ProjectionSpec x RenderSpec (QueryProjectionSpec) as the sole read contract and presets as named (S,P,R) triples, with projection_spec.py cited as the existing realization.

### 72. `polylogue-60i5` — Durable-tier batch coordination: one user v4->v5 and one source v2->v3 migration window

P2 task · open
Intent: Cross-cutting operational constraint (the single biggest insight across the R&D specs): MANY pending designs each want a durable-tier bump — user v5: recursive-safety columns, content-variants tables, s7ae coordination messages, config-engine settings, queries/result-sets/analyses (rxdo.2, rxdo.8); source v3: compaction snapshots, ingest-fidelity fingerprints, secret-redaction tombstones, zstd blob placement.
Execution shape: Process: when >=2 labeled beads are execution-ready, declare a window, write migrations NNN as one coherent set per tier, verify backup manifest, apply one step, run parity tests.
Done when: First batch window executed with a single user-tier migration covering all ready v5 consumers; no durable migration lands outside a declared window.

### 73. `polylogue-6il` — devloop-integration --json --check consumed by devloop-review

P2 task · open
Intent: Integration lane has no machine-readable contract; review cannot see branch role, ahead count, stale ledger, replay branches, PR URLs.
Execution shape: Add `--json` and `--check` to the devloop-integration lane so it emits a machine-readable contract carrying branch role, ahead count, stale-ledger state, replay branches, and PR URLs; mirror the existing devloop-status/readiness JSON envelope shape so review has one parser.
Done when: - `devloop-integration --json` emits a stable schema carrying branch role, ahead count, stale ledger, replay branches, and PR URLs, mirroring the devloop-status/readiness envelope shape.

### 74. `polylogue-6wnh` — Bound thread refresh cost for large Codex appends

P2 task · open · parent: polylogue-1xc
Intent: Current 3wb closure evidence shows the old 260s append.index.graph_resolve rebuild tail is not active on recent daemon appends, but the worst current graph_resolve sample is still dominated by append.index.graph_resolve.thread_refresh: 3.020976s of a 3.040429s graph_resolve step on a 340.8 MB Codex append.
Execution shape: Use /realm/tmp/polylogue-workload-graph-tail-499e26363.json as the initial evidence.
Done when: A focused benchmark or live diagnostic shows thread_refresh cost on giant Codex append/replay rows; either the implementation becomes incremental and the worst recent 340 MiB-class thread_refresh path is materially reduced, or the bead records why the current cost is the correct bounded floor with a guardrail that would catch a regression toward the 260s class.

### 75. `polylogue-83u.6` — Attachment acquisition census by origin and byte volume

P2 task · open · parent: polylogue-83u
Intent: Post-v13: measure acquired/unfetched/unavailable by origin and byte volume on the live archive.
Execution shape: Read-only attachment-acquisition census over the active archive (promoted from the 2026-07-04 notes sidecar).
Done when: 1.

### 76. `polylogue-8jg9.1` — Standing backlog-hygiene invariant lint (bd devloop gate)

P2 task · open · parent: polylogue-8jg9
Intent: Backlog structure trails filing unless an invariant lint enforces it (the 2026-07-03 session needed a 41-agent sweep to recover).
Execution shape: This session needed a 41-agent sweep because structure trails filing.
Done when: The lint runs in the devloop and fails on a seeded violation of each of the 5 classes; a clean backlog passes; wired into devtools verify or a bd hook.

### 77. `polylogue-8jg9.2` — Blob-GC lease/orphan concurrency test (the acquire->commit race)

P2 task · open · parent: polylogue-8jg9
Intent: internals.md documents the load-bearing lease model (pending_blob_refs + gc_generations bridging the acquire-blob -> write-DB-row commit window) but test-closure-matrix.yaml:179 admits it is not exercised by a dedicated test.
Done when: A test acquires a lease, starts GC, and asserts the leased blob survives; a released-lease orphan is reclaimed; sweep_orphaned_blob_leases clears a SIGKILLed writer's lease past ORPHAN_LEASE_MAX_AGE_S.

### 78. `polylogue-9e5.1` — Assertion-layer adoption audit: is the flywheel used or aspirational?

P2 task · open · parent: polylogue-9e5
Intent: Count assertions by kind/status/author_kind in the live user.db: are candidates being judged?
Execution shape: Pure read-only SELECT over user.db (promoted from the 2026-07-04 notes sidecar; open with SQLite URI mode=ro, never a write connection).
Done when: 1.

### 79. `polylogue-9e5.19` — Storage-layer correctness scenario family

P2 task · open · parent: polylogue-9e5
Intent: scenario-coverage.yaml gap 'storage-correctness' orphaned on gh#590.
Done when: A storage-correctness scenario family exists and runs via devtools lab lanes; it covers idempotent re-ingest, FTS trigger drift, and lineage composition; scenario-coverage.yaml references this bead, not gh#590.

### 80. `polylogue-9e5.24` — Sink MCP analysis primitives into insights/ + api facade; delete surface-side math

P2 task · open · parent: polylogue-9e5
Intent: server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentile), compare_sessions (:559 per-key set diff).
Execution shape: server_insight_tools.py implements analysis math directly in the MCP surface, unreachable from CLI/library: correlate_sessions (:826 Pearson r + metric-name->field map), find_similar_sessions metadata lane (:652 weighted heuristic), aggregate_sessions/workflow_shape_distribution/find_abandoned_sessions (:510/:233/:298 GROUP-BY + severity-rank + ISO-week), tool_call_latency_distribution (:131 nearest-rank percentile), compare_sessions (:559 per-k…
Done when: correlate/find_similar-metadata/aggregate/workflow_shape/find_abandoned/tool_call_latency/compare have api facade methods and the MCP tools call them (grep shows no math/GROUP-BY left in server_insight_tools.py); the severity map, similarity weights, and week-bucketing are defined once in insights/; a CLI or library caller produces byte-identical aggregates to the MCP tool for a fixture archive; devtools verify green.

### 81. `polylogue-9e5.25` — Review zero-use MCP surfaces from affordance usage artifact

P2 task · open · parent: polylogue-9e5
Intent: The current agent-affordance-usage demo classifies 59 MCP tools as zero captured agent use and non-operator surfaces.
Execution shape: Batch the review through contracts/surface-algebra rather than deleting isolated tools.
Done when: 1.

### 82. `polylogue-9e5.26` — Review zero-use CLI surfaces from affordance usage artifact

P2 task · open · parent: polylogue-9e5
Intent: The current agent-affordance-usage demo classifies 34 CLI commands as zero captured agent use and non-operator surfaces.
Execution shape: Batch the review through CLI surface algebra and command-inventory contracts.
Done when: 1.

### 83. `polylogue-9e5.27` — Speed up live affordance usage surface inventory

P2 task · open · parent: polylogue-9e5
Intent: After switching the default family report and inventory counts away from action-row materialization, the live .agent/demos/agent-affordance-usage regeneration still took roughly 88 seconds on the full archive.
Execution shape: Profile devtools workspace affordance-usage on the live archive with query-plan evidence.
Done when: 1.

### 84. `polylogue-9e5.3` — Column honesty audit: null/unknown density for key semantic columns

P2 task · open · parent: polylogue-9e5
Intent: For material_origin, tool_result_is_error/exit_code, message_type, branch_type, session_kind: null/unknown density per origin per month on the live archive.
Execution shape: Read-only column-honesty census over the live index.db (promoted from the 2026-07-04 notes sidecar; no product-code mutation).
Done when: 1.

### 85. `polylogue-9e5.4` — Get->modify->put race audit across daemon/CLI/MCP writers

P2 task · open · parent: polylogue-9e5
Intent: Sweep multi-step read-then-write sequences on separate connections: blob leases, ingest_cursor updates, embedding_status transitions, fts_freshness_state.
Execution shape: Static get->modify->put race audit across the shared-SQLite writers (promoted from the 2026-07-04 notes sidecar; static sweep first, not tests).
Done when: 1.

### 86. `polylogue-9l5.15` — Triage frontier: worth_reviewing_score + TRIAGED lifecycle — an inbox that empties

P2 task · open · parent: polylogue-9l5
Intent: A context-free frontier over all ~16K logical sessions (inverts the cwd-coupled find_resume_candidates): time-invariant worth_reviewing_score materialized with a decomposable breakdown (unresolved blockers, open questions, decision density, terminal state), collapsed by logical_session_id; inverted-U staleness applied at READ time (materialized staleness goes stale).
Execution shape: The score is a NAMED-FEATURE LINEAR COMBINATION with per-feature contributions visible in the payload (rigor doctrine: no opaque scores) — every feature is a computable structural signal that already exists or has an owning bead: unacknowledged_failure (tool_result_is_error=1 with no subsequent success of a normalized-same command in-session — the 'failed and moved on' signature), abnormal_termination (session ends inside a tool loop / no assist…
Done when: Frontier returns logical representatives with score breakdown + confidence; triage/snooze removes rows via runtime method; disposable clean-finish rows zero out while blocker sessions surface; demoted buckets visible.

### 87. `polylogue-a7xr.5` — FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies

P2 task · open · parent: polylogue-a7xr
Intent: Same class as the closed fts_freshness_state double-declaration, three more objects: trigger DDL for messages_fts/session_work_events_fts/threads_fts lives in BOTH storage/sqlite/archive_tiers/index.py (:307-324, :729-767, :449-464) and storage/fts/fts_lifecycle.py (:198-233+ as _BLOCKS/_SESSION_WORK_EVENT/_THREAD trigger DDL constants used by drop-and-recreate repair).
Execution shape: Move trigger DDL lists to storage/fts/sql.py (already holds FTS_INDEX_EXISTS_SQL) as the single source; archive_tiers/index.py composes its DDL script from them; fts_lifecycle imports them.
Done when: rg finds each trigger body in exactly one module; a drift test asserts fresh-DB and repair-path trigger text are identical (normalized); rebuild + repair smoke green.

### 88. `polylogue-ap7` — Semantic transcript rendering: tool-call-aware, provider-agnostic, shared CLI/web renderer registry

P2 task · open · parent: polylogue-fnm
Intent: Transcripts render today as generic message/block sequences — the same flat treatment for a prose paragraph, a 400-line Bash result, an Edit diff, and a Task dispatch.
Execution shape: (1) RENDERER REGISTRY keyed by (tool family, provider-normalized): Edit/Write -> syntax-highlighted diff/file cards (the yrx diff reconstruction is the data source — same computation, presentation here); Bash/shell -> terminal-styled block with command, collapsed output (first/last N lines, expand), exit badge from structural outcome; Read/Grep/Glob -> compact file-reference cards (path, range, match count) instead of dumped contents; Task/subag…
Done when: On the seeded corpus: Edit shows a highlighted diff, Bash shows exit-badged folded output, Task shows a linked subagent card — in BOTH web and CLI; unknown tools render as today; structure-parity snapshot tests green across backends; a before/after recording of one real session is committed as the demo asset (3tl.5 machinery).

### 89. `polylogue-cpf.4` — Enforce degrade-loudly: sweep silent soft-failure paths to carry a signal

P2 task · open · parent: polylogue-cpf
Intent: The cpf degraded-mode doctrine says "degrade loudly once", but a deep read (2026-07-05) found the codebase systematically degrades SILENTLY on derived/fallback/probe paths — robust (never crashes) but serves incomplete/stale data with no signal, which for a system-of-record is a construct-validity hole.
Done when: Each identified silent soft-fail path (probe fail-closed, lineage truncation, timeout-to-empty, fallback data-drop) either emits a typed degradation signal consumers can read, or logs-loudly-once; a reader/agent can distinguish "no data" from "degraded/timed-out/truncated".

### 90. `polylogue-exb` — Layering: substrate rings import the api facade (6 sites, 2 private-symbol reaches)

P2 task · open
Intent: The architecture says surfaces adapt over substrate, but the dependency arrow runs backwards in at least six places: storage/embeddings/preflight.py imports select_pending_embedding_session_window from polylogue.api; storage/embeddings/materialization.py and insights/correlation_view.py import api.sync.bridge.run_coroutine_sync; storage/repair.py imports the PRIVATE api.archive._rebuild_archive_session_insights; sources/live/batch.py imports the whole Polylogue facade; pipeline/run_stages.py im…
Execution shape: Per-site relocation, then close the gate: (1) _active_archive_root -> config/core (it is runtime-root resolution, nothing facade-y about it); (2) run_coroutine_sync -> a core/asyncbridge module (two substrate rings need it; the api.sync home is an accident of history); (3) _rebuild_archive_session_insights: repair orchestration needs the insight-rebuild primitive — move the primitive into insights/ or pipeline/ and have BOTH api and repair call…
Done when: The six inward imports are relocated (grep for 'polylogue.api' under storage/, sources/, insights/, pipeline/ returns nothing, including function-local).

### 91. `polylogue-f2qv.2` — Codex disjoint-lane normalizer: decompose cached/uncached and reasoning/completion with a regression guard

P2 task · open · parent: polylogue-f2qv
Intent: PROBLEM.
Done when: Synthetic Codex and Claude token_count payloads normalize into four disjoint labelled lanes that sum to reported totals; an invariant test asserts disjointness and that the naive input+output sum would double-count (7.69x-class guard).

### 92. `polylogue-f2qv.4` — Single pricing source of truth: LiteLLM catalog, drop tokencost, last-path-segment match

P2 task · open · parent: polylogue-f2qv
Intent: PROBLEM.
Done when: grep shows tokencost is removed from dependencies and imports; a single LiteLLM-backed resolver owns all model->rate lookups via last-path-segment match; a test asserts no second price table exists and that live-archive models resolve or are labelled unknown.

### 93. `polylogue-fnm.11` — Pipeline/clause parity across units + generated support matrix

P2 task · open · parent: polylogue-fnm
Intent: Live evidence: `sessions where origin:claude-code-session | count` fails with 'pipeline terminal stage must be an executable <unit>s where ...
Execution shape: (1) Build the support matrix FROM the registries (query_units + stage lowerers + clause grammar), not by hand: a generated docs/query-support-matrix.md (devtools render, drift-checked) showing units x pipeline stages x compact clauses.
Done when: docs/query-support-matrix.md is generated from registries and drift-checked by render all --check.

### 94. `polylogue-ivsc` — Classify Codex state_5 token drift outside lineage replay

P2 task · open · parent: polylogue-f2qv
Intent: After logical-session high-water token accounting, the live Codex reconciliation probe still shows 78 logical outside-tolerance threads.
Execution shape: Use /realm/tmp/polylogue-cost-reconciliation/codex-logical-probe-current-max100.json as the seed artifact.
Done when: The Codex reconciliation report distinguishes lineage replay residuals from external-state/accounting-grain drift; live active archive artifact explains the remaining outside-tolerance rows without implying replay double-counting; any adjusted pass/fail status is backed by tests and live evidence.

### 95. `polylogue-jnj.1` — Collapse read per-view flags into ProjectionSpec/RenderSpec algebra

P2 task · open · parent: polylogue-jnj
Intent: read exposes compact algebra (--projection/--render/--spec) alongside per-view flag clusters (--window-hours, --repo-path, --since-hours, --related-limit...).
Execution shape: Collapse read per-view flags into the existing Query x Projection x Render algebra, and use this bead to converge the duplication that would otherwise make export/variant work accrete another surface.
Done when: read --spec remains the visible contract for composed selection/projection/render state.

### 96. `polylogue-1lm` — Composable transcript views: selector x transform x budget algebra

P2 task · open
Prerequisites: polylogue-jnj.1.
Intent: 'Prose-only' is one point in a space the operator keeps requesting by example: user messages plus directly-adjacent agent replies (raw-log 07-02 — what the agent intended to report, minus the toil); tool outputs truncated from the middle beyond N lines (raw-log 06-23); decisions-only; tool-skeleton (calls + outcomes, no bodies); failure-slices; reboot-with-refs (37t.3); compact recaps for mass export ('every sinex-related chatlog in compact form for gptpro', 06-18).
Execution shape: (1) Extend ProjectionSpec (jnj.1) with typed selector predicates (reuse the DSL block-predicate grammar — no second filter language) and per-class TransformSpec; compile_context and renderers consume the same spec (4p1's Projection axis, deepened).
Done when: The three raw-log examples work as presets/inline specs on the live archive; compile_context and read share the machinery; presets visible to completions; omission markers always carry resolvable refs.

### 97. `polylogue-lio` — Align cross-repo devloop contract on beads (Sinex parity)

P2 task · open
Intent: Shared conductor conventions predate beads: Sinex still uses devloop-checkpoint --queue as its directive channel.
Execution shape: Shared conductor conventions predate beads; Sinex still uses `devloop-checkpoint --queue` as its directive channel.
Done when: 1.

### 98. `polylogue-ma2` — Add FK-supporting index for web_content_constructs message cleanup

P2 task · open · parent: polylogue-1xc
Intent: Live v24 rebuild evidence showed ChatGPT full-replace rows spending seconds in append.index.full_replace.delete_messages.
Execution shape: After polylogue-6h7 closes, add a canonical index on web_content_constructs(message_id) in the index tier DDL, bump INDEX_SCHEMA_VERSION once as part of a batched schema slice, document the re-ingest plan in docs/internals.md, and add a focused planner/DDL test proving message-id lookups no longer scan the table.
Done when: EXPLAIN QUERY PLAN for SELECT 1 FROM web_content_constructs WHERE message_id = ?

### 99. `polylogue-mhx.3` — Retrieval quality eval lane: measure FTS vs vector vs hybrid before believing any of them

P2 task · open · parent: polylogue-mhx
Intent: Every retrieval default (auto lane resolves lexical; hybrid RRF constants; --semantic promotion) was chosen by taste, not measurement.
Execution shape: devtools bench retrieval, campaign run/compare pattern: (1) Labeled set: ~50-100 (query -> expected sessions/messages) pairs; bootstrap from real usage — queries the operator/agents ran followed by which session they actually opened (the archive records its own MCP/CLI usage; affordance-usage machinery can mine query->open chains), then hand-verify.
Done when: 1.

### 100. `polylogue-ox0` — Codex deep integration: state DBs as authoritative source + AppServer live lane

P2 task · open · parent: polylogue-fs1
Intent: Codex writes far more than rollout JSONL: state_5.sqlite (authoritative session/thread state — already used read-only by cost reconciliation lpl), goals_1.sqlite (goal/plan state — unexplored), history.jsonl, hooks.json, prompts/, rules/, skills/ (agent-config, 7aw), shell_snapshots/.
Execution shape: Three sub-lanes, sequenced: (1) STATE-DB IMPORTER (mirror of fs1.1): state_5.sqlite as authoritative Codex source — threads, turns, token counters with the disjoint-lane semantics already encoded in memory, parent/fork relations for lineage (4ts cross-check); goals_1.sqlite explored and mapped (VERIFY schema — undocumented); rollout JSONL demotes to fallback.
Done when: state_5.sqlite importer materializes threads/turns/lineage on the live machine and reconciles with existing rollout-derived sessions (dedup by content, no double-count — token totals match lpl reconciliation); goals_1.sqlite mapped with a written schema note; app-server protocol spike artifact committed (capabilities, version, event vocabulary) with go/no-go for lanes 2 and 3.

### 101. `polylogue-rxdo.1` — ObjectRef expansion: query, query-run, result-set, finding, cohort, analysis, annotation-batch kinds

P2 task · open · parent: polylogue-rxdo
Intent: Verified live 2026-07-06: ObjectRefKind in core/refs.py is a closed Literal of 29 kinds with none of the analysis-object kinds; normalize_object_ref_text rejects unknown kinds, so nothing can target a query or result set today.
Execution shape: Add kinds: query, query-run, result-set, finding, cohort, analysis, annotation-batch to ObjectRefKind + the kind map + normalize paths in core/refs.py.
Done when: normalize_object_ref_text accepts the new kinds; resolve_ref returns typed pending payloads for them; user_audit + rendered schemas regenerated; existing ref tests extended.

### 102. `polylogue-rxdo.2` — Content-addressed query identity + durable user.db queries/query_names/result_sets/query_edges

P2 task · open · parent: polylogue-rxdo
Intent: query:<hash> keyed on the canonical planned AST AFTER macro expansion (mirrors content-hash idempotency: equivalent queries collapse).
Execution shape: Canonicalization: expand macros -> typed AST -> NFC strings -> sort commutative AND/OR children -> preserve non-commutative pipeline/seq/except/sort/limit order -> canonical field aliases -> include grain+lane+rank policy -> relative-time queries hash the DYNAMIC ast while runs store resolved absolute bounds -> sha256 over sorted-key compact JSON.
Done when: Same query text with reordered AND operands yields one query hash; @macro repoint does not change the hash of past runs; user-tier migration preserves all existing assertions (parity test); set-algebra grain is part of result-set identity so cross-grain member keys cannot collide.

### 103. `polylogue-rxdo.3` — Query-run + result-relation telemetry in ops.db; refs on every query envelope

P2 task · open · parent: polylogue-rxdo
Intent: Every COMMITTED query execution (CLI, MCP, daemon web, API) records an ops.db query_runs row (actor, surface, verb, request+lowered spec, archive epoch, timing, status, degraded state) and a result fingerprint + bounded sample refs; the query response envelope gains query_run_ref + result_set_ref + grain + count_precision.
Execution shape: ops.db is disposable so long-lived citations must not point here without promotion; expired query-run refs resolve to a typed expired-operational-ref payload, never silently vanish.
Done when: CLI --json and MCP query responses carry the three refs for the same committed query (parity test); routine preview typing produces zero rows; a promoted run survives ops.db reset.

### 104. `polylogue-rxdo.4` — AssertionKind.FINDING: claims with evidence, reusing the candidate->judge lifecycle verbatim

P2 task · open · parent: polylogue-rxdo
Intent: A finding is a durable claim (n, statistic, query_ref, result_set_ref, expected) produced by a detector/agent/analysis, stored as an assertion row so the ENTIRE existing lifecycle (candidate default, judge_assertion_candidate accept/reject/defer/supersede, judgment recorded as assertion, promotion flips inject gate) is reused with zero new lifecycle code — exactly the pathology pattern in user_write.py.
Execution shape: value_json schema polylogue.finding.v1: finding_kind (query-delta | finding-drift | measure | pathology | claim-vs-evidence), statistic {op,value,unit}, n, query_ref, result_set_ref, baseline/current refs, optional expected {measure,op,value}.
Done when: upsert_findings_as_assertions mirrors the pathology writer; findings appear in the judgment queue; finding refs resolve; regenerated schemas + user_audit pass; a re-run with identical inputs produces zero new rows.

### 105. `polylogue-rxdo.7` — Annotation substrate: schema registry, annotation batches, JSONL import surface, typed value predicates

P2 task · open · parent: polylogue-rxdo
Prerequisites: polylogue-37t.15, polylogue-rxdo.1.
Intent: The missing loop for external-agent analysis: export evidence pack -> agent labels rows under a declared schema -> import as candidate assertions -> query them back -> judge -> report.
Execution shape: Schemas connect to the 9l5.7 measure-registry discipline: a label is an operationalization with construct-validity metadata, not just a JSON key.
Done when: Roundtrip demo: export a bounded evidence pack, import 5 labeled rows as candidates, query them via assertions where with a typed value predicate, judge one active, render.

### 106. `polylogue-svfj` — Block content-hash citation anchors: blocks.content_hash + resolver with typed drift states

P2 task · open
Intent: THE anchor atom multiple programs stand on (webui cockpit citations, finding evidence refs rxdo.4, drift detection 37t.14, compaction loss anchors gjg.3, export citations).
Execution shape: Derived index-tier change: canonical DDL edit + version bump — batch with the next index window (gjg.1, ma2, 4ts.5, wohv all queue for the same bump).
Done when: Anchor created pre-re-ingest resolves post-re-ingest as ok or drifted_position (verified content); a fork replay resolves relocated_lineage with the inheritance edge cited; ambiguous returns candidates; hash_mismatch never auto-rewrites.

### 107. `polylogue-bby.15` — Evidence basket -> citable report -> verified export (cockpit core loop)

P2 task · open · parent: polylogue-bby
Prerequisites: polylogue-svfj.
Intent: The missing "report" end of the web workbench: select blocks/spans in the reader -> basket (content-hash anchors + quote + note + provenance of the query that surfaced it) -> live Markdown report draft with footnotes -> EXPORT GATE re-resolves every citation and blocks/flags by state (ok + drifted_position export with verified note; drifted_message/relocated need explicit promotion; ambiguous/missing block by default; quarantined blocks unless the report is explicitly forensic; hash_mismatch ha…
Execution shape: Three-pane cockpit flow (results | reader+graph | basket+draft); daemon API basket/report/verify routes collapse into service verbs when the t46/B8 contract lands.
Done when: Full loop on the seeded demo corpus: query -> basket 5 items -> draft renders footnotes -> re-ingest the corpus -> verify flags the drifted item and export annotates it; a deleted block blocks export with a typed reason.

### 108. `polylogue-t0p` — Capture the rest of Claude Code: todos, file-history, prompt history, debug artifacts

P2 task · open · parent: polylogue-l4kf
Intent: Session JSONL is one artifact among many that Claude Code writes, and the others answer questions transcripts cannot: ~/.claude/todos/*.json = the agent's live PLAN state per session (task lists with status — plan-vs-execution comparison becomes structural); file-history/ = pre-edit file snapshots (ground truth for the yrx changes view, catches what tool-log reconstruction misses); history.jsonl = the operator's prompt history across sessions (paste-detection and prompt-reuse analytics); histor…
Execution shape: Priority order by evidence value: (1) TODOS: small JSON, session-keyed, trivially parsed -> new artifact kind + a session-linked read model (plan items with status transitions when multiple snapshots exist); analytics: plan-completion rate by session outcome (a construct-valid 'did it do what it planned' measure — feeds claim-vs-evidence).
Done when: Todos and file-history ingest end-to-end from the live machine into artifact kinds with provenance; plan-vs-outcome measure registered (9l5.7) with tier=structural; yrx cross-check lane reports agreement rate between reconstructed and snapshot diffs; watcher covers the new roots; fidelity declared per artifact kind.

### 109. `polylogue-t46.3` — Unify list/search query-spec->ArchiveStore execution across CLI, MCP, and daemon web

P2 task · open · parent: polylogue-t46
Intent: Three surfaces re-map the query DSL/params to ArchiveStore filter args and each own pagination/total/cursor: daemon http.py:1902 _do_archive_list_sessions (+ the _do_archive_* fast-path family), mcp/archive_support.py:254-379 archive_session_list_payload/archive_search_payload, and cli/archive_query.py:674/787/815 _query_hits/_paginate_rows/_build_cursor.
Done when: CLI find, MCP archive_list_sessions/archive_search_sessions, and daemon /api/sessions return the same total and page boundaries for identical filters (parity test across the three surfaces); the per-surface spec->filter mapping and total/cursor logic is deleted in favor of one execution helper (grep shows no second query_terms/contains merge); the two MCP list surfaces converge to one total semantic; devtools verify green.

### 110. `polylogue-t46.4` — Delegate daemon session-similarity KNN to SqliteVecProvider.query_by_session

P2 task · open · parent: polylogue-t46
Intent: daemon/similarity.py re-implements session-seeded vector ranking (raw MATCH/k SQL over message_embeddings, per-session best-distance aggregation, matched-message count, L2->cosine in _l2_to_cosine_similarity:217) that storage/search_providers/sqlite_vec_queries.py:143 SqliteVecProvider.query_by_session already does -- the substrate file even comments that the daemon's _PER_MESSAGE_K mirrors it.
Done when: daemon _knn_for_embedding/_aggregate_hits/_l2_to_cosine_similarity are deleted; /api/similar ranking equals SqliteVecProvider.query_by_session ordering for a seed session (parity test); the sqlite_vec_queries comment about mirroring _PER_MESSAGE_K is removed because there is no longer a mirror; devtools verify green.

### 111. `polylogue-t46.5` — Route CLI transcript/dialogue file export through substrate read+render; delete streaming_markdown SQL path

P2 task · open · parent: polylogue-t46
Intent: cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_envelope + rendering/core_markdown + rendering/blocks.
Execution shape: cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_envelope + rendering/core_markdo…
Done when: streaming_markdown.py raw-SQL read helpers are deleted; transcript/dialogue --to file markdown for a prefix-sharing (forked/resumed) session composes the full lineage identically to stdout output (test compares file export bytes vs the substrate transcript for a forked session); block filtering (reasoning/prose) matches the substrate projection; devtools verify green.

### 112. `polylogue-t46.6` — Fix referenced_path OR-vs-AND filter divergence and delete dead CLI stats aggregators

P2 task · open · parent: polylogue-t46
Intent: cli/query_semantic.py:63 referenced_path_matches_slice uses any(term...) (OR-of-terms) for multi-term referenced_path while the substrate archive/query/runtime_matching.py:35 uses all(term...) (AND-of-terms), so the semantic-stats surface selects different sessions than the actual query filter -- a live correctness divergence.
Done when: A two-term referenced_path query returns the same session set from the semantic-stats surface and from the query filter (regression test); referenced_path_matches_slice/action_matches_slice and the dead query_stats aggregators are deleted (grep confirms callers gone); origin/date/tool/work-kind grouping goes through stats_by; devtools verify green.

### 113. `polylogue-t46.8` — MCP surface collapse: ~96 tools -> verb algebra (query/get/explain/context/assert/maintenance...)

P2 task · open · parent: polylogue-t46
Intent: The MCP surface (96 tools live) is a discovery burden and a maintenance trap (every tool = contract + names + regen).
Done when: Verb set + resources + prompts cover every retired tool proven by goldens; EXPECTED_TOOL_NAMES shrinks with equivalence evidence per deletion; discovery tests + contracts regenerated; no capability regression reported by the golden suite.

### 114. `polylogue-t8t` — Agent workflow catalog: walk the seven core flows end-to-end, fix what breaks

P2 task · open · parent: polylogue-s7ae
Intent: Affordances exist in pieces; nobody has walked the actual workflows end-to-end: (1) RESUME — arrive in repo, recover context, continue; (2) FORENSIC DEBUG — did a past session touch this; (3) PRIOR ART — has anything explored this approach; (4) DECISION LOOKUP — what did we decide and why; (5) POSTMORTEM WRITE — close the loop after failure (37t.7); (6) COST CHECK — what has this repo/task cost; (7) SELF-INSPECTION — agent reads its own live session mid-flight (raw-log 05-28; needs hook-fresh i…
Execution shape: (1) Each flow = a workflow registry entry (product/workflows.py REQUIRED_WORKFLOW_IDS) with intent, tool sequence, envelope shapes, round-trip token cost, latency budget (20d.14).
Done when: Seven registry entries; seven archived walk transcripts; every gap filed as a linked bead; rendered catalog lists measured tokens+latency per flow; self-inspection demonstrates an agent reading its own in-progress session.

### 115. `polylogue-th0` — Interactive-surface test harness: pty flows, completions, fuzzy pickers

P2 task · open
Intent: The suite (248k lines) is strong on units/properties/snapshots and blind on exactly the surfaces the UX program is now building: nothing drives a real pty, so fzf select flows, the coming judge TUI (p5g), bare-invocation triage (jnj.13), pager behavior, and terminal-width/color rendering are untested by construction; shell completions (fnm.4) have no correctness harness at all (a broken completion script fails silently forever); interactive-ambiguity moments (jnj.11) can regress without any red…
Execution shape: (1) PTY harness in tests/infra: pexpect (or pty+os primitives — decide by trying pexpect's reliability under pytest-xdist) driving the real CLI binary against the seeded corpus: send keys, assert on screen state with normalized snapshots (strip timing/colors via the existing syrupy terminal-snapshot conventions; explicit width matrix 80/120/200 since fzf layouts shift).
Done when: PTY harness runs 5+ golden flows green in CI serial lane; completion contract tests are registry-generated and fail when a unit is added without completion metadata; a deliberate fzf-flow regression (reordered candidates) is caught by the harness in a demonstration commit.

### 116. `polylogue-x4s` — Express devloop state in Polylogue substrate (dogfood target)

P2 task · open · parent: polylogue-rii
Intent: Raw-log 2026-07-03: 'perhaps devloops themselves could be expressed in sinex and/or polylogue and/or beads?'.
Execution shape: The full argument (fables devloop reading): the conductor's own memory is the silo it is fighting — ACTIVE-LOOP.md, OPERATING-LOG.md, HANDOFF-LATEST.md, EVENTS.jsonl live outside the archive while the product has the native home sitting unwired: handoff and run_state assertion kinds exist in the enum with NO writer (user_write.py has no helper for them), blackboard has an unresolved filter, and get_resume_brief/compose_context_preamble already d…
Done when: - First writers for the handoff/run_state assertion kinds are added to user_write.py (the kinds already exist in the enum with no writer — grep confirms the new helpers); devloop-handoff / devloop-focus dual-write into user.db while markdown stays authoritative.

### 117. `polylogue-x7d` — Unify root query row rendering contracts

P2 task · open · parent: polylogue-jnj
Intent: The bounded find fix had to patch three projection/rendering paths: archive_query root rows, query_output deterministic rows, and select rows.
Execution shape: Target shape: define a small shared row projection helper or value object for session list rows and search-hit rows, with explicit budgets (title 96 or table budget, snippet 320), single-line normalization, and separate full-read expansion.
Done when: - A shared row-projection helper/value object exists for session-list rows and search-hit rows with explicit budgets (title 96 / table budget, snippet 320), single-line normalization, and separate full-read expansion.

### 118. `polylogue-xgw` — Archive schema hygiene for evidence-cockpit read paths

P2 task · open · parent: polylogue-1xc
Intent: Targeted schema-extension/hygiene (unsafe joins, JSON checks, durable-row rules) — not a rewrite.
Execution shape: Targeted, additive schema-extension/hygiene for the read paths that feed the evidence-cockpit web workbench (bby) — NOT a rewrite.
Done when: - Each cockpit read-path hygiene item is enumerated in the PR (unsafe join, missing JSON check, durable-row rule) with the specific query/column it applies to; fixes are additive (CREATE INDEX / CHECK / ADD COLUMN), not a rewrite.

### 119. `polylogue-yeq` — Advanced verification lanes: metamorphic DSL, daemon chaos, API-contract walks

P2 task · open
Intent: Beyond coverage-by-example, three method upgrades the suite lacks: (1) METAMORPHIC testing for the query engine — the DSL has algebraic laws (filter commutativity, pipeline-stage composition vs post-hoc filtering, unit-count consistency between find and aggregate paths) that hold for ALL queries, not just the ones we thought to write; hypothesis can generate queries from the grammar and assert the laws, catching lowering bugs example tests never will (the sessions-vs-observed-events pipeline in…
Execution shape: (1) Metamorphic lane: a hypothesis strategy over the registry grammar (bounded depth, seeded corpus) + ~8 laws as properties: predicate-order invariance, LIMIT monotonicity, group-by-count sums equal ungrouped count, unit-where result parity with equivalent find-mode filters, measure composition associativity once 9l5.7 lands.
Done when: Metamorphic lane finds-or-proves: run against the current engine and either file real bugs or commit the laws as green properties; chaos lane demonstrates one seeded crash-recovery invariant per staged kill point; ref-walk lane covers 100% of list-emitting routes/tools and fails on a deliberately broken ref in a demonstration commit.

### 120. `polylogue-a7xr.10` — Kill-or-adopt the search-provider lane: production bypasses the abstraction it should use

P2 chore · open · parent: polylogue-a7xr
Intent: VERIFIED 2026-07-06: FTS5Provider/HybridSearchProvider/factories have zero production call sites — only their own tests import them.
Execution shape: If ADOPT: define RetrievalLane protocol from what production actually needs (query, candidate set, scores, lane metadata for the eval payload); implementations wrap the existing inline SQL (fts lane), SqliteVecProvider (dense lane), reciprocal_rank_fusion (hybrid), reranker (mhx.1's client); cli/archive_query.py:830-852 becomes lane dispatch; the current FTS5Provider/HybridSearchProvider bodies are salvage-or-delete per method (most likely delet…
Done when: A decision recorded WITH mhx.3 (adopt or kill, one paragraph of why); if adopt: all production retrieval flows through the lane interface, inline fusion in archive_query.py gone, mhx.3 bake-off consumes the lanes, goldens unchanged; if kill: zero references remain, mhx.3 notes it owns lane construction.

### 121. `polylogue-a7xr.11` — Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug

P2 chore · open · parent: polylogue-a7xr
Intent: VERIFIED 2026-07-06: 6 of 14 protocols in protocols.py have zero consumers anywhere (SessionReader, SearchStore, ArchiveMessageQueryStore, SemanticArchiveQueryStore, SessionSemanticStatsStore, SessionArchiveReadStore) — violating the module's own docstring rule ('only protocols with 2+ implementations earn their existence').
Execution shape: Delete the 6 unconsumed protocols, the repo list/list_summaries/count kwarg wrappers + iter_summary_pages, the two TypedDicts (pass SessionRecordQuery through at query_store_archive.py:70-84), and either the cursor field-spec entry or add the cursor field deliberately (decide with rxdo pagination needs — a real cursor concept may arrive with result-set pagination; if so, wire it properly instead of deleting).
Done when: protocols.py contains only consumed protocols (each with a named consumer in a comment); dead kwarg surface gone; cursor mapping resolved (deleted or actually wired); mypy strict green; ~600 LOC removed.

### 122. `polylogue-20d.1` — CLI->daemon fast path over UDS (persistent hot process)

P2 feature · open · parent: polylogue-20d
Intent: Route CLI queries through the already-hot daemon when available: skips import cost, warm SQLite page cache, shared readiness state.
Execution shape: Precedent: fast-status path (commands/status.py:950, click_app.py:214-221) already prefers the daemon — extend the pattern to the whole read surface.
Done when: - Fast-path read surface: `--verbose` prints `served-by: daemon (uds, <ms>)` and a warm daemon serves find/read/messages/facets within the 20d.14 interactive-tier budget (target 3.6-17s -> 0.3-0.5s wall).

### 123. `polylogue-20d.12` — Daemon result cache + post-ingest warming: precomputed answers, cursor-keyed invalidation

P2 feature · open · parent: polylogue-20d
Prerequisites: polylogue-20d.1.
Intent: The fast path (20d.1) makes the daemon reachable in milliseconds; this bead makes the daemon WORTH reaching: today every facets/status/aggregate request recomputes from SQLite (live evidence: /api/facets defers repos+action_types by default, was stuck 'loading...
Execution shape: (1) CACHE KEY: the archive ingest cursor (ops.db already tracks it) + query fingerprint.
Done when: bench slo (interactive tier): cached facets/status p50 <30ms on the seeded corpus with warm daemon.

### 124. `polylogue-20d.13` — Daemon push channel: SSE events for live UIs instead of polling

P2 feature · open · parent: polylogue-20d
Intent: Everything live currently polls: the webui polls for facets/status, live tailing (bby.4) would poll, CLI watch modes would poll.
Execution shape: (1) TRANSPORT: Server-Sent Events over the existing HTTP server — chunked responses work on BaseHTTPRequestHandler (one long-lived thread per subscriber; cap subscribers, loopback-only), so this is NOT blocked on the ASGI decision (dx1) — though if dx1 migrates, SSE gets cheaper; note the thread cost in the dx1 evidence.
Done when: A subscribed browser receives session.ingested within 2s of ingest commit on the seeded corpus.

### 125. `polylogue-37t.1` — Assertions: consumer wiring + lifecycle tightening for unified overlays

P2 feature · open · parent: polylogue-37t
Intent: Assertion substrate is the live path; remaining work is consumer wiring + lifecycle (promotion, staleness, expiry).
Execution shape: Wiring points (verify current state first): unwired AssertionKinds handoff/prompt_eval/highlight need first writers + surface registration (user_audit every-kind-has-a-surface invariant will force the surface entry; scope_ref/author_ref must use a registered ObjectRef kind — see #2383 memory).
Done when: - First writers exist for the three currently-unwired AssertionKinds (handoff, prompt_eval, highlight — present-but-writerless at core/enums.py:409/423/426) and each passes the user_audit every-kind-has-a-surface invariant with a registered ObjectRef scope_ref/author_ref.

### 126. `polylogue-37t.12` — Judgment queue: operator bulk review/accept/reject of candidate assertions

P2 feature · open · parent: polylogue-37t
Intent: WHY: 37t's loop names four stages (claims -> JUDGMENT -> preamble -> reboot) but only JUDGMENT has no owning bead.
Done when: MCP: 'judge_assertion_candidate' and 'list_assertion_candidates' tools exist on the operator/agent-write MCP role, listed in EXPECTED_TOOL_NAMES with TOOL_CONTRACT entries; a candidate written by an agent (author_kind='agent', status='candidate') is listable and can be accepted/rejected via the MCP tool, and an accepted candidate produces a new ACTIVE assertion (verify via list_assertion_claims statuses=active).

### 127. `polylogue-37t.11` — Context scheduler: one arbiter for everything that enters an agent's context

P1 feature · open · parent: polylogue-37t
Prerequisites: polylogue-37t.15, polylogue-37t.12.
Intent: The missing 40% of the OS-vision design, and the coherence fix for a real fragmentation risk: as of today SEVEN independent mechanisms want to write into agent context, each with its own budget rules — repo brief + resume delta (37t.4), semantic recall (mhx.4), SRS-due lessons (rvh), blackboard messages (1hj), PreToolUse/prompt advisories (bfv), compaction re-grounding (gjg), and the affordance-index pointer (pj8).
Execution shape: (1) SOURCE PROTOCOL: each injector registers (declare-once) as a ContextSource with: moment (session-start | pre-compact-resume | mid-session-advisory | on-demand), priority class (correctness > directives > recall > ambient), a propose() returning candidate items (each: content-or-ref, token cost, relevance score, source ref, expiry), and a degrade order (full -> ref-only -> drop).
Done when: ContextSource protocol + scheduler in context/compiler.py with deterministic assembly (property test: same inputs -> byte-identical context); 37t.4's sections migrated as the first two sources; budget invariants enforced (property test: never exceeds moment budget at any source combination); ledger rows written per injection and readable via CLI + MCP; cross-source dedup demonstrated (advisory suppresses same-ref blackboard item in a seeded scen…

### 128. `polylogue-s7ae.3` — Coordination messages and subtle scheduler-mediated advisories

P1 feature · open · parent: polylogue-s7ae
Prerequisites: polylogue-37t.11.
Intent: Why: multi-agent cooperation needs lightweight communication and awareness, but not a noisy chatroom or hardcoded workflow police.
Execution shape: Realize the coordination-specific parts of existing beads 1hj (blackboard as agent comms), bfv (advisory hooks), d1y (hook install/liveness), and 37t.11 (ContextSource scheduler/ledger).
Done when: Agents can post and receive scoped coordination messages with refs/provenance, using existing blackboard/user-state machinery where viable.

### 129. `polylogue-s7ae.5` — Live proof: two agents, separate worktrees, one repo — overlap, message, context, handoff

P1 task · open · parent: polylogue-s7ae
Prerequisites: polylogue-37t.11, polylogue-s7ae.3.
Intent: Realize the epic s7ae HEADLINE acceptance line ('A live proof demonstrates at least two agents on one repo with separate worktrees, visible overlap/resource awareness, a scoped coordination message, context injection, and a handoff packet') — currently owned by no child.
Done when: A committed, reproducible proof exists (run script + captured before/after JSON envelope artifacts under a devtools workspace path) demonstrating: two agents on one repo in separate worktrees; each envelope shows the other as a same-repo peer with overlap + resource-episode awareness; exactly one scoped coordination message posted and observed as delivered/addressed in the recipient's envelope; context injection recorded via the 37t.11 ledger; a…

### 130. `polylogue-37t.4` — SessionStart preamble opt-in rollout (polylogue + sinnix repos)

P2 task · open · parent: polylogue-37t
Prerequisites: polylogue-37t.12.
Intent: Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first (operator decision).
Execution shape: Wire compose_context_preamble into SessionStart hooks per-project via .claude/settings.json, polylogue + sinnix first.
Done when: - compose_context_preamble is wired into the existing SessionStart hook (upgrading sessionstart-polylogue-recall.sh, not adding a second hook) for polylogue + sinnix.

### 131. `polylogue-37t.2` — Inline annotation protocol: agent-authored structure in plain prose

P2 feature · open · parent: polylogue-37t
Intent: Agents write structured markers in prose; extraction at block enrichment turns them into candidate assertions with evidence refs.
Execution shape: PROTOCOL DESIGN (2026-07-03, generalizing the marker idea into a composable protocol): (1) SYNTAX: line-anchored sigil markers — '::kind(args): body' on its own line, inline '[[kind: body]]' for short spans.
Done when: - Final sigil chosen after a corpus collision scan: grep the live archive for candidate-prefix false positives and record the scan result in the PR.

### 132. `polylogue-37t.3` — Reboot-with-refs: session self-compaction protocol

P2 feature · open · parent: polylogue-37t
Intent: Agent reboots into a fresh session carrying all prose verbatim with every tool exchange collapsed to a one-line expandable ref — better than harness compaction because refs resolve via resolve_ref.
Execution shape: Flow: agent calls MCP compile_context with ContextSpec(seed_refs=[current session], purpose=continue) + a new prose_with_refs segment profile -> markdown with authored prose verbatim, tool_use/result collapsed to '-> [tool:Bash exit:0 4.2s] pytest ...
Done when: - compile_context with a new prose_with_refs segment profile emits authored prose verbatim while every tool_use/result collapses to a one-line '<ref:action:...>' marker.

### 133. `polylogue-37t.6` — Session-aware devshell entry: surface what the last agent session left behind

P2 feature · open · parent: polylogue-37t
Intent: On cd/direnv entry, print what the last agent session in this cwd left: unresolved blackboard blocker/question notes, the last session's terminal state, resume candidates for this directory.
Execution shape: On cd/direnv entry, print one bounded line summarizing what the last agent session in this cwd left behind: unresolved blackboard blocker/question notes (blackboard_list unresolved filter), the last session's terminal state, and resume candidates for this directory (find_resume_candidates, which already scores cwd at 0.15 weight).
Done when: 1.

### 134. `polylogue-37t.7` — Close the failure loop: verify postmortem -> next session's context seed

P2 feature · open · parent: polylogue-37t
Intent: workspace failure-context produces an envelope (testmon graph + git history + fixtures for a failing test); the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session.
Execution shape: `workspace failure-context` produces an envelope (testmon graph + git history + fixtures for a failing test) and the pytest supervisor produces a postmortem (.cache/verify/) — neither flows into the next agent session.
Done when: - A compile_context seed is constructed from the latest verify postmortem (.cache/verify/) + the `workspace failure-context` envelope (testmon graph + git history + fixtures), injectable via the SessionStart hook or an explicit `polylogue context --from-verify` entry point.

### 135. `polylogue-37t.8` — Resume routing: map a session to the harness invocation that reopens it

P2 feature · open · parent: polylogue-37t
Intent: Genuinely-missing item: nothing owns 'reopen this session in its harness' — claude --resume <id> vs the codex equivalent, per origin; plus detecting an already-open interactive session (the kitty/hyprland control plane can answer that on this machine, but keep that integration optional/pluggable).
Execution shape: Add a mapping from an archived session to the harness invocation that reopens it, per origin: `claude --resume <id>` for Claude Code, the Codex equivalent, etc.
Done when: 1.

### 136. `polylogue-4ts.5` — Compaction boundary-range columns + effective-context derivation

P2 feature · open · parent: polylogue-4ts
Intent: session_events boundary_start/end_position + boundary_message_id; get_effective_context(session, at_position) = what the model actually saw vs the full composed prefix.
Execution shape: Design from gh#2478 (code-grounded): add session_events.boundary_start_position/boundary_end_position (message range a compaction replaces, in the session's own position coordinate) + boundary_message_id (the materialized summary).
Done when: 1.

### 137. `polylogue-gjg.1` — compaction_events + compaction_loss_items derived tables; identity survives rebuild + re-ingest

P2 task · open · parent: polylogue-gjg
Prerequisites: polylogue-d1y, polylogue-4ts.5.
Intent: Promote compaction from a session_events count + lineage edge to an archived object.
Execution shape: Derived tier: edit canonical DDL + bump INDEX_SCHEMA_VERSION, batch with the next index bump window (ma2/4ts.5 rule).
Done when: Rebuild from source produces identical compaction_ids; a re-ingested session keeps its compaction rows; changed interpretation surfaces as event_content_hash delta not silent overwrite.

### 138. `polylogue-gjg.2` — Pre-compaction snapshot capture: hook payload when available, manifest-of-refs otherwise, honesty ladder always

P2 task · open · parent: polylogue-gjg
Prerequisites: polylogue-d1y, polylogue-4ts.5.
Intent: Two-level snapshotting with snapshot_source as a FIRST-CLASS honesty axis, not a footnote: precompact-hook (strongest — the actual assembled context payload, blob-stored content-addressed, claim: this WAS model context) > jsonl-boundary (manifest of composed-transcript message refs up to the boundary; claim limited to archive-composed transcript, NOT model context) > reconstructed-composed-context (weakest) > none (epidemiology only).
Execution shape: PreCompact hook wiring rides d1y (hooks install — existing gjg dependency).
Done when: A live compaction on the operator machine lands either a hook snapshot or a labeled jsonl-boundary manifest; every snapshot row carries source+confidence; no unlabeled reconstruction.

### 139. `polylogue-gjg.3` — Deterministic loss-forensics: 4-tier structural diff + lost-but-later-needed ranking

P2 task · open · parent: polylogue-gjg
Prerequisites: polylogue-d1y, polylogue-gjg.1, polylogue-gjg.2, polylogue-4ts.5.
Intent: The base retained/lost/transformed classifier is deterministic and structural — NO LLM in the base pass (LLM annotation may layer later as separate judgment rows).
Execution shape: Registered as a 9l5.7 measure (compaction-loss) with tier=structural and coverage gates; epidemiology = plain relation algebra over the two tables (rate by provider/trigger/session-length bucket, marked-decision loss rate, failed-tool-outcome loss rate, snapshot-coverage rate).
Done when: Classifier is pure + property-tested (same inputs => same items); ranking exposes per-component scores; epidemiology query renders with n + coverage footnotes.

### 140. `polylogue-5hf` — Provider token accounting: honest cross-provider usage ledger

P2 feature · open · parent: polylogue-f2qv
Intent: Coverage, caveats, cached-vs-uncached splits, reasoning tokens, current-window + cumulative session usage.
Execution shape: SCOPE.
Done when: Given a session/day/origin, the ledger returns per-lane token totals (cached/uncached input, reasoning/completion output), a coverage class and caveat set drawn from the documented vocabulary, and both API-equivalent and subscription-credit cost figures.

### 141. `polylogue-7aw` — Ingest agent configuration as a source family (skills, CLAUDE.md, hooks)

P2 feature · open
Intent: Treat agent configuration as a corpus polylogue versions, queries, and correlates with session outcomes: CLAUDE.md/AGENTS.md revisions, skills, hook configs — config-over-time x outcome-over-time is the continual-learning dataset (composes with the self-experimentation rail).
Execution shape: FULL SCOPE (upgraded from notes 2026-07-03; operator: we should have full control/understanding of CLAUDE.md regardless of replacement ambitions): (1) CAPTURE: CLAUDE.md files (global dots/claude tree incl.
Done when: Config artifacts ingest with content-hash versioning and watcher coverage on the live machine; a session from last week resolves to the exact CLAUDE.md/skill versions it ran under; skill-invocation report renders (which skills, frequency, outcome mix); one state-like global-CLAUDE.md section migrated to injected-equivalent with drill-verified parity and the static text retired; the classification of the operator's current global CLAUDE.md into t…

### 142. `polylogue-h6r` — Agent identity: a stable who-did-this tuple for every session

P2 task · open
Prerequisites: polylogue-7aw.
Intent: Half the analytics tower assumes 'per agent' partitions the schema cannot express: a model name is not an agent — the same model under different CLAUDE.md versions, skills, or MCP profiles is behaviorally a different worker.
Execution shape: agent_identity = (model_name, harness+version, config_state_ref, role) materialized per session as a derived column set: model/harness from session metadata (exists), config_state_ref from the 7aw config-artifact joins (the hash of the config set in force at session start), role from subagent dispatch context where present.
Done when: Identity tuple materialized for new sessions on the live machine (config ref resolving via 7aw); an identity-partitioned measure runs with the unknown fraction stated; h10's calibration curves key on identity, not bare model name.

### 143. `polylogue-7xv.1` — Work-trace + reproduction harness: verify a session repo-work from a clean worktree

P2 feature · open · parent: polylogue-7xv
Intent: Reproduction-first verification, NOT raw command-stream replay (rejected: schema does not guarantee action-level cwd/env/stdin/sandbox/tool-version/pre-post hashes; unsafe+brittle as the headline).
Done when: Fixture session (edits + failing-then-passing test) renders a work-trace with ordered actions/cwd/repo/commit/touched-paths/exit evidence; reproduction verify creates a worktree, applies target, runs verifiers, records attempt + assertions; unsafe classes are classified and not auto-run; proof card round-trips refs.

### 144. `polylogue-83u.2` — Attachment byte acquisition for non-inline sources (Drive/zip/local)

P2 feature · open · parent: polylogue-83u
Intent: Acquire bytes where the handle is live: Drive via DriveSourceClient.download_bytes inside the iterator scope (un-bypass download_assets); export-zip member resolution while the zipfile is open; local paths via transport-only local_source_path under a source-root allowlist + realpath-escape check.
Execution shape: Un-bypass byte acquisition at each live-handle boundary and deposit bytes onto ParsedAttachment.inline_bytes, then reuse the shipped true-SHA-256 blob write (the _acquire_attachment_blob path).
Done when: 1.

### 145. `polylogue-90y` — In-page overlay: Polylogue presence on chat sites — archive state, context, assertion capture

P2 feature · open · parent: polylogue-jlme
Intent: The extension currently only EXTRACTS; it could also PRESENT.
Execution shape: TASTE CONSTRAINTS FIRST (operator: 'must feel non-crappy'): shadow-DOM component, zero layout shift on the host page (fixed corner chip + slide-over panel, never inline injection into the chat column); respects prefers-color-scheme; one keyboard chord (e.g.
Done when: On chatgpt.com and claude.ai: chip+panel render with zero host-page layout shift in light and dark themes; selection pill appears only on text selection; saving a selection creates a candidate assertion whose evidence ref resolves to the exact archived message; per-site toggle and global kill work; panel shows relevant judged assertions when embeddings are enabled.

### 146. `polylogue-9l5.1` — Outcome-conditioned analytics: cost/duration/retries/tools by structural success

P2 feature · open · parent: polylogue-9l5
Intent: Group cost, duration, retry chains, and tool usage by structural outcome (exit_code/is_error terminal state), with per-origin coverage caveats.
Execution shape: Anchored examples (all one step from existing substrate): cost of failed vs clean sessions; failure-rate by model VERSION; retry cascade depth; 'sessions where >30% of tool calls errored' (needs the child-count DSL predicate or the relation directly).
Done when: 1.

### 147. `polylogue-9l5.2` — Cross-provider comparative analytics

P2 feature · open · parent: polylogue-9l5
Intent: The archive is the only place Claude/Codex/ChatGPT/Gemini work traces coexist normalized: same task-shape comparisons — failure rates, retry behavior, cost per completed session, tool-mix, session lengths — by origin/model with explicit coverage tiers per origin so partial provenance cannot masquerade as a finding.
Execution shape: The killer query shape: 'same repo, same month: Claude Code vs Codex — turns per task, $/session, tool-failure rate, subagent usage' — no lab and no observability vendor can run it; the archive is the only place these providers coexist normalized.
Done when: 1.

### 148. `polylogue-9l5.6` — tool-episodes projection: call + result + outcome + context + next action

P2 feature · open · parent: polylogue-9l5
Intent: Sidecar research (Sartre): affordance-usage and analyze tools stop at aggregate evidence.
Execution shape: New derived read model `tool_episodes` (rebuildable; registry pattern under polylogue/storage/insights/session/, registered in insights/registry.py so CLI+MCP inherit it).
Done when: 1.

### 149. `polylogue-avg` — Fold devloop claim-guard vocabulary upstream into ops status/readiness

P2 feature · open
Intent: The loop scripts guard claims better than the product does: devloop-status treats schema-version match as 'openable, not converged', gates convergence claims on raw-materialization debt being zero/classified, and blocks latency claims behind live_performance_proof_blocked.
Execution shape: Add a claim-guard section to `polylogue ops status`/readiness that derives 'what you may claim' (archive openable / converged / search-ready / perf-measurable) from the same signals the devloop scripts already use: schema-version match => openable-not-converged; raw-materialization debt zero/classified => converged; FTS freshness => search-ready; the live_performance_proof_blocked gate => perf-measurable.
Done when: - `polylogue ops status --json` exposes a claim-guard block with the four claim states (openable / converged / search-ready / perf-measurable), each derived from its documented signal.

### 150. `polylogue-bby.8` — Web reader perceived performance: virtualized list, streamed search, optimistic navigation

P2 feature · open · parent: polylogue-bby
Intent: Fluidity in the reader is perceived latency, not just server latency: the session list renders 16k+ rows into the DOM (scroll cost grows with archive size), search waits for full results before painting anything, clicking a session blocks on the full detail fetch, and every panel loads with spinners instead of skeletons.
Execution shape: Four standard techniques, applied to the existing SPA (sequence after bby.6 extracts the JS to real files — refactoring inline-string JS is not viable): (1) LIST VIRTUALIZATION: render only the viewport window of the session list (hand-rolled windowing is ~100 lines, no framework needed); constant DOM cost at any archive size.
Done when: Session list scrolls at 60fps with 20k sessions (virtualized DOM stays constant-size).

### 151. `polylogue-da1` — Provider format-drift sentinel: detect upstream export-shape changes from live ingest

P2 feature · open
Intent: Claude Code, Codex, ChatGPT, and Gemini change their export/JSONL shapes without notice.
Execution shape: Reuse the existing schema-inference machinery (schemas/ shape signatures) rather than building a new detector: at ingest, count records whose shape does not match the committed provider schema package, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry (the ops tier explicitly allows additive columns — no schema bump).
Done when: - At ingest, records whose shape does not match the committed provider schema package are counted, keyed by (origin, element kind, unseen-key signature), into ops.db telemetry via additive columns only (no schema bump — the ops tier explicitly allows additive columns).

### 152. `polylogue-dmp` — polylogue note: zero-friction memory capture from the terminal

P2 feature · open
Intent: Terminal-side ambient capture is missing: when the operator (or an agent in a shell) realizes something worth remembering, there is no one-liner to record it with provenance — the thought either interrupts flow for a heavier tool or evaporates.
Execution shape: (1) 'polylogue note "WAL contention was the real cause, not cache size"' writes a candidate NOTE assertion.
Done when: 1.

### 153. `polylogue-f2qv.3` — Dual cost view: API-list-equivalent and subscription-credit reported separately

P2 feature · open · parent: polylogue-f2qv
Intent: PROBLEM.
Done when: Cost surfaces return api_equivalent_usd and subscription_credit as distinct fields; a test asserts they differ correctly for a session with cache reads (subscription view < API view).

### 154. `polylogue-fnm.1` — Aggregates beyond count (sum/avg/min/max/percentiles)

P2 feature · open · parent: polylogue-fnm
Prerequisites: polylogue-fnm.11.
Intent: `group by X | count` is the only aggregate; cost/duration/token questions need sum/avg/percentiles to compose instead of spawning bespoke analyze modes.
Execution shape: Full target shape (fables ladder item 4): `| group by tool, session.origin | agg count, avg:duration_ms, p90:duration_ms, sum:tokens` — multi-field group by AND named aggregate list AND time bucketing `group by bucket:day(time)` (temporal-bucket machinery already exists in the temporal read view; reuse its bucket functions in the lowering).
Done when: - On the live archive `messages where ...

### 155. `polylogue-fnm.10` — fields/select stage with parent-field projection (first real Transform)

P2 feature · open · parent: polylogue-fnm
Intent: The upward-access ceiling: session.* fields work for FILTERING on every unit (~25 scoped fields, metadata.py:620-658) and for whitelisted group-by, but output shapes are frozen Pydantic payloads that hardcode exactly two parent fields (MessageQueryRowPayload carries origin+title, payloads.py:~1265-1280).
Execution shape: Land it as the first real Transform, fulfilling the QueryUnitTransformStage reservation (expression.py:376-386, 'never produced by the current parser').
Done when: - `messages where session.repo:polylogue AND text:timeout | fields text, occurred_at, session.title, session.repo` returns generic dict-shaped rows keyed by requested field name, emitted alongside (not replacing) the typed MessageQueryRowPayload.

### 156. `polylogue-fnm.12` — User-defined query macros: named, composable DSL shorthands in user.db

P2 feature · open · parent: polylogue-fnm
Intent: The highest-leverage runtime configurable found in the preference design pass: operators (and agents) repeat the same filter combinations constantly — 'my real coding sessions' = origin:claude-code-session + repo-scope + exclude-subagents + trailing-90d.
Execution shape: (1) Definition: 'polylogue config macro set mine "origin:claude-code-session exclude:subagents after:-90d"' (and MCP/webui equivalents); stored as typed user.db rows (the y4c settings registry), validated at definition time by compiling against the grammar — a macro that does not parse is refused with the caret error.
Done when: Define/list/delete macros via CLI+MCP; @macro composes inside find, unit-where, and pipeline queries on the live archive; invalid macro refused at definition with caret; explain shows expansion with provenance; cycle/depth guards tested; completions surface @-macros.

### 157. `polylogue-fnm.13` — Set-algebra over query results: union/intersect/except between queries

P2 feature · open · parent: polylogue-fnm
Intent: BRAINSTORM (2026-07-05, operator asked to explore syntax incl.
Done when: `polylogue find 'auth intersect week:2026-W01'` returns exactly the session_ids in both operand sets; `except` subtracts; `union` dedups.

### 158. `polylogue-fnm.14` — find <query> | compact: token-budgeted corpus-compaction projection with drop manifest

P2 feature · open · parent: polylogue-fnm
Intent: The R&D-flywheel enabler: package a queried cohort as a decision-dense, lineage-deduplicated digest for an external LLM, with an honest fidelity manifest.
Done when: Fixture with protocol/tool spam compacts to a digest excluding it with per-material_origin drop counts; failed->fix->verify fixture keeps the pair with refs; fork/resume fixture emits shared prefix once and reports duplicate-prefix omissions; 60k budget test proves the deterministic degradation order; every digest anchor round-trips to a source ref; context-image and compact remain separate payload shapes sharing helpers.

### 159. `polylogue-fnm.2` — Projection predicates/windows + render/layout stages on attached units

P2 feature · open · parent: polylogue-fnm
Intent: Declared predicates/windows on attached units (e.g.
Execution shape: Two layers: (1) predicates/windows on attached units — extend the with-stage parse (hand-parsed pipeline region, expression.py ~:1484-1601 where WITH_PROJECTION_SUPPORTED_UNITS is enforced) to accept per-unit bracket args (messages[role:user, last:20]); lower onto the existing exact-session-id fetch in attached_units.py (caps exist: _MAX_ROWS_PER_SESSION=200; field selection landed 867b1d048 — extend that payload, don't fork it).
Done when: - `...

### 160. `polylogue-fnm.4` — Shell completion + fuzzy selection as read-only projections of the grammar registries

P2 feature · open · parent: polylogue-fnm
Intent: Completion/query-builder metadata built on the same grammar+registries used by CLI/MCP/daemon/web — not a second parser.
Execution shape: Scope per gh#1844 minus what landed: query_completions MCP tool, projection-unit completions, and dynamic shell completions (polylogue config completions --shell) exist.
Done when: - A registry-diff snapshot test asserts that every grammar-reachable field/unit/pipeline-stage/read-view name (enumerated from metadata.py descriptors, read_view_registry, and operations.action_contracts.ACTION_CONTRACTS) appears in the completion payload, so new DSL work cannot silently miss completions.

### 161. `polylogue-fnm.6` — Wire the terminal stage to projections: | read / | context-image

P2 feature · open · parent: polylogue-fnm
Intent: QueryUnitTransformStage is reserved-never-parsed and terminal args are reserved for future actions.
Execution shape: The terminal args slot is explicitly reserved for this (expression.py:397-407 'future actions (read view, analyze mode, bundle kind)').
Done when: - `sessions where ...

### 162. `polylogue-fs1.4` — Report: polylogue forensics for Hermes sessions

P2 feature · open · parent: polylogue-fs1
Intent: Five-section per-session/per-corpus report, computed from the canonical archive (composition over existing primitives where possible): 1) session topology — parents, resumes, compactions, subagents, branches, long turns; 2) LLM/request economy — token lanes, cost, retry/fallback causes, model/provider shifts, cache-read amplification; 3) tool execution profile — durations, failures, approvals, repeated calls, parallel groups; 4) failure patterns — loops, stalls, empty-response retries, repeated…
Execution shape: Composition first: sections 1-3 and most of 4 should lower onto existing primitives — get_session_topology/logical session (topology), session_provider_usage_events + cost rollups (economy), actions/tool timing (tool profile), pathology detectors + structural outcomes (failure patterns), session_commits/git correlation (footprint).

### 163. `polylogue-kph` — Provenance-carrying PRs: attach the authoring session's postmortem bundle

P2 feature · open · parent: polylogue-s7ae
Intent: Sessions already link to repos and commits (session_repos, session_commits).
Execution shape: Wire CI or a gh hook to attach the authoring session's postmortem bundle to each PR, pairing PR-body claims with the in-session verification exit codes.
Done when: 1.

### 164. `polylogue-mhx.1` — Provider abstraction: one OpenAI-compatible embedding client, local and cloud, model registry in meta

P2 feature · open · parent: polylogue-mhx
Intent: Voyage is hardcoded (VOYAGE_API_URL/DEFAULT_MODEL/DEFAULT_DIMENSION constants).
Execution shape: (1) Config: [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute + batch size; keep the native Voyage path as one provider preset, default unchanged.
Done when: 1.

### 165. `polylogue-37t.5` — Local embedding lane via OpenAI-compatible provider (LiteLLM gateway)

P2 feature · open · parent: polylogue-37t
Prerequisites: polylogue-mhx.1.
Intent: Voyage is the only embedding provider; a local lane makes semantic search $0 and the whole loop air-gapped — and pairs with the Hermes bridge program for a fully local, zero-cloud stack.
Execution shape: Seam: VectorProvider protocol + Voyage constants in sqlite_vec_support.py.

### 166. `polylogue-mhx.2` — Embedding target policy: what gets a vector, at what granularity, at what cost

P2 feature · open · parent: polylogue-mhx
Intent: Today exactly one class is embedded: authored prose messages (user/assistant, human/assistant-authored material origin, positive word count — the v21 partial index).
Execution shape: Target classes, each with an explicit purpose, source text, and marginal cost: (1) authored prose messages [exists] — purpose: fine-grained hybrid search.
Done when: 1.

### 167. `polylogue-mhx.4` — Semantic recall leg in context compilation: the memory actually retrieves

P2 feature · open · parent: polylogue-mhx
Prerequisites: polylogue-mhx.2.
Intent: compose_context_preamble and compile_context currently select by explicit refs, recency, and policy — there is no semantic leg, so a judged lesson about 'SQLite WAL contention' never surfaces when the new session starts debugging a WAL issue unless someone remembers it exists.
Execution shape: (1) Query formation: at SessionStart the recall query is cheap context — repo, recent commit subjects, the resumed session's summary (on resume); mid-session (agent-invoked via MCP) it is the agent's stated intent.
Done when: SessionStart recall proposes items through the ContextSource protocol (37t.11) with visible why-fields (similarity, kind, judgment state, refs) — no opaque scores; judged assertions outrank candidates at equal similarity; recall stays within the preamble segment budget with refs-over-bodies; a seeded lesson about a distinctive topic surfaces when a session starts on that topic and does NOT surface on an unrelated repo (both directions tested); d…

### 168. `polylogue-o21` — Extension-point ergonomics: declare-once registries, scaffolds, actionable completeness errors

P2 feature · open
Intent: Every extension today is a scavenger hunt across parallel registration sites, each failing opaquely when missed — the accumulated tribal knowledge lives in bd memories: a new MCP tool needs EXPECTED_TOOL_NAMES + TOOL_CONTRACT + role gating + render openapi + render cli-output-schemas (four separate opaque failures); a new golden-path workflow must be in REQUIRED_WORKFLOW_IDS or CLI startup crashes with an unrelated error; a new AssertionKind breaks two renders plus the every-kind-has-a-surface…
Execution shape: Three legs, applied per extension point (MCP tool, CLI verb/command, DSL unit/stage, insight, origin, assertion kind, devtools command, workflow): (1) DECLARE-ONCE: each point gets a single declaration object carrying ALL metadata the parallel sites currently hold (name, contract, role gating, schema, docs blurb, owning surface) — the parallel lists become derivations: EXPECTED_TOOL_NAMES is generated FROM tool declarations, REQUIRED_WORKFLOW_ID…
Done when: - Slice 1 (size:S, unblocks dependents): a DeclarationSpec dataclass + registry protocol are defined and one pilot extension point (MCP tools) is migrated to declare-once, with a published pattern doc; dependents can build against the protocol immediately.

### 169. `polylogue-2qx` — OriginSpec: one package per origin, dispatch order derived from declared strictness

P2 feature · open · parent: polylogue-l4kf
Prerequisites: polylogue-o21.
Intent: Adding a provider/origin is the most common expansion Polylogue will ever do (uiw enumerates a dozen candidates; Grok, OTel-GenAI, Langfuse waiting), yet origin logic is smeared: looks_like + parser in sources/parsers/*, hand-ordered dispatch list in dispatch.py (strict-before-loose maintained by vigilance — the Hermes ordering caution exists precisely because one wrong insertion steals records), Provider enum in types.py, schema package elsewhere, usage-coverage entry elsewhere, fidelity/compl…
Execution shape: (1) An OriginSpec dataclass per origin, one package per origin under sources/origins/<name>/: detector (looks_like), parser entry, declared STRICTNESS TIER (exact-schema > record-path-validated > structural-sequence > loose-shape), schema-package pointer, capture-mode/fidelity declaration, usage-coverage class, pricing hints.

### 170. `polylogue-opc` — Self-tracing: the daemon's own spans land in its own archive

P2 feature · open · parent: polylogue-20d
Intent: Polylogue has an OTLP receiver and stores spans — and instruments itself with none.
Execution shape: (1) A tiny internal tracer (no opentelemetry-sdk dependency — spans are dicts posted to the in-process intake; the OTLP wire format only matters for external emitters): span(name, attrs) context manager instrumenting: HTTP route handlers (route, status, duration), converger stages (per session/batch), archive_query compile vs execute vs render phases, write_effects (per effect once 0aj lands), embed windows, cache lookups (20d.12).
Done when: Spans emitted for routes/stages/query-phases on the seeded corpus daemon; request-id ties a route span to its query-phase children; sampling caps enforced with drop counters visible in /metrics; ops traces --slow renders a span tree for a real slow request; retention pruning works.

### 171. `polylogue-oxz` — Performance instrumentation doctrine: slow-query log, phase timings, logging discipline

P2 feature · open · parent: polylogue-20d
Intent: Beyond spans, three instrumentation gaps and one doctrine gap: (a) no SLOW-QUERY LOG — SQLite statements over a threshold should be recorded with their text and, on demand, their EXPLAIN QUERY PLAN, or every perf regression starts from scratch (the EQP sweep 20d.7 is a snapshot; this is the continuous version); (b) CLI has no phase breakdown — 'polylogue --debug-timing find X' should print import/config/db-open/compile/execute/render wall per phase (the 1.6s floor was diagnosed by hand; it shou…
Execution shape: (1) SLOW-QUERY LOG: sqlite3 set_trace_callback (or the profile hook) on both connection profiles; statements >threshold (default 50ms, y4c-tunable) log normalized SQL + duration + connection profile into ops.db (bounded ring, not unbounded rows); 'ops slow-queries' renders top-N with optional EQP capture on a copy.
Done when: Slow-query log captures a seeded slow statement with duration + normalized SQL and bounded storage; --debug-timing prints the phase table and matches span data for daemon-served queries; log-doctrine page committed + print()-diagnostic lint wired into verify quick; webui beacons land in ops.db on the seeded workbench.

### 172. `polylogue-p5g` — polylogue judge: interactive candidate triage in the terminal

P2 feature · open · parent: polylogue-37t
Intent: The judgment gate is the heart of the memory model, but judging has no ergonomic surface: candidates accumulate (pathology findings, decision candidates, soon overlay/selection captures and setup improvements) and reviewing them means MCP calls or raw queries.
Execution shape: (1) 'polylogue judge' opens the pending-candidate queue in the established fzf pattern (jnj.11 machinery): list shows kind, age, source session, first line; preview pane renders the full candidate + its evidence refs (resolved excerpts, not bare ids).
Done when: - `polylogue judge` opens the pending-candidate queue in the established fzf pattern (jnj.11 machinery): the list shows kind, age, source session, and first line; the preview pane renders the full candidate plus its evidence refs as resolved excerpts (not bare ids).

### 173. `polylogue-ptx` — Browser-capture posting channel: un-gate, with attachments

P2 feature · open · parent: polylogue-bby
Intent: Operator decision 2026-07-03: UN-GATE.
Execution shape: The channel exists on the worktree branch (worktree-agent-aa5375b510cb4aa5d era work): extension->receiver POSTING path, previously operator-gated OFF.
Done when: 1.

### 174. `polylogue-rii.1` — Agent work-event write-leg -> session_events -> materialized read-models

P2 feature · open · parent: polylogue-rii
Intent: record_work_event/emit_decision write surface routed through the existing idempotent ingest seam (no parallel writer); flows into the run-projection read models.
Execution shape: Route through the existing idempotent ingest seam (write_raw_and_parsed / the daemon ingest path) — no parallel writer (gh#2459 body is code-grounded here).
Done when: - MCP tools record_work_event / emit_decision are registered with the mutation role: EXPECTED_TOOL_NAMES + TOOL_CONTRACT updated, role gating enforced, and `devtools render openapi && devtools render cli-output-schemas` regenerated with `devtools render all --check` clean.

### 175. `polylogue-rii.2` — Materialize hook events + OTLP spans into queryable evidence

P2 feature · open · parent: polylogue-rii
Prerequisites: polylogue-rii.1.
Intent: Hook events are captured as raw blobs but never materialized (~95% of hook-only signal invisible: tool annotations, pre-MCP output, permission decisions, cwd changes, subagent lifecycle); OTLP spans likewise.
Execution shape: Code-confirmed gaps (gh#2461, re-locate lines): archive/artifact_taxonomy/runtime.py:~209 classifies HOOK_EVENT with parse_as_session=False (stored as raw blobs, used only for paste enrichment); artifact_taxonomy/support.py:~82 looks_like_hook_event hardcodes provider in ('claude-code','codex').

### 176. `polylogue-yrx` — Session changes view: per-session diff/changelog composed from edit evidence

P2 feature · open · parent: polylogue-bby
Intent: A session's most important output is often 'what did it change' — and the archive already holds the evidence: Edit/Write/NotebookEdit tool_use blocks carry file paths and old/new content, Bash blocks carry git commands and commit SHAs in results, hook FileChanged events (when wired) add out-of-band edits.
Execution shape: (1) READ MODEL: a session_changes projection (derived, rebuildable) composed per session from structural evidence only: edit-tool blocks (path, old/new strings -> unified diff hunks; sequential edits to one file fold into a cumulative diff), Write blocks (full-file states), git evidence (commit SHAs from tool results + Claude-Session trailers via 7xv's mapping), test outcomes (exit codes from the keystone fields) — no prose mining, every row car…
Done when: For a seeded session containing edits and a commit: the Changes tab, read --view changes, and get_session_changes list the touched files with reconstructed diffs and the commit SHA; failed edits (is_error) are excluded; every row resolves to its block ref.

### 177. `polylogue-f2qv` — Provider usage & cost honesty: disjoint token lanes, one pricing source, dual cost view

P2 epic · open
Intent: WHY: token/cost accounting is a correctness surface with a track record of silent large errors (7.69x Codex inflation; per-model partition double-count #2472) — and cost numbers are exactly what operators quote publicly, so wrong numbers are reputational.
Execution shape: PROBLEM / DOCTRINE.
Done when: 1.

### 178. `polylogue-jlme` — Capture extension: reliability, coverage, and in-page presence

P2 epic · open
Intent: WHY: the MV3 browser-capture extension is a load-bearing acquisition surface (live chat capture feeding the spool/receiver) but had no owning epic: reliability, capture-state visibility, and in-page presence were scattered.
Execution shape: The browser-capture extension surface (spool health, capture-state UX, in-page overlay, concurrent-instance safety) had no owning epic.
Done when: Spool health + capture completeness are observable; per-chat capture-state indicator ships once (badge and in-page chip share one signal, not two); concurrent instances dedup by content hash.

### 179. `polylogue-at44` — user_settings table is dead: DDL + migration 004 exist, zero runtime read/write helpers

P3 bug · open
Intent: Verified live 2026-07-06: rg over polylogue/ finds user_settings only in the DDL (user.py), migration 004, and an unrelated filename string in artifact_taxonomy/runtime.py — no reader, no writer, table is empty and unwired.
Execution shape: Add get/set/list helpers in user_write.py + async twin (STORAGE TWINS trap: apply to both sync archive_tiers and async mixins or daemon/CLI diverge), a settings surface on the api facade, and first consumer: subscription_tier read by cost_compute (kills the hardcoded /21_700_000*20.0 Pro assumption).
Done when: Set+get subscription_tier via CLI/API; cost compute reads it with a sane default; both storage paths tested.

### 180. `polylogue-mhx.7` — Two live vec0 DDL definitions: unify to one canonical embeddings table-creation path

P3 bug · open · parent: polylogue-mhx
Intent: Verified 2026-07-06: message_embeddings vec0 DDL exists in BOTH storage/search_providers/sqlite_vec_runtime.py AND storage/sqlite/archive_tiers/embeddings.py — the R&D review reports incompatible metadata naming between them (+origin vs legacy +source_name).
Execution shape: Canonical site: storage/sqlite/archive_tiers/embeddings.py (the tier owner per architecture; search_providers/sqlite_vec_runtime.py becomes a consumer importing the DDL constant).
Done when: One canonical DDL site; drift test fails on divergence; metadata column naming decided (origin) and migrated per derived-tier regime.

### 181. `polylogue-tf0e` — Generic-messages parser fallback drops available created_at/updated_at

P3 bug · open
Intent: sources/dispatch.py:_generic_messages_session (710-729) hardcodes created_at=None, updated_at=None even when the source payload carries timestamps.
Done when: A generic `{messages, title, created_at, updated_at}` payload parses to a ParsedSession carrying those timestamps; date-range queries and reader display show them.

### 182. `polylogue-xnkf` — actions view fans out on duplicate tool_ids: one logical action becomes up to NxM rows

P3 bug · open
Intent: VERIFIED ON LIVE ARCHIVE 2026-07-06 (construct-validity hunt): the actions view (archive_tiers/index.py:328) pairs blocks on (tool_id, session_id) with no uniqueness or proximity constraint.
Execution shape: Fix at the view (derived tier — canonical DDL edit + index rebuild regime, batch per 60i5 doctrine): pair each tool_use with the FIRST tool_result sharing tool_id at the smallest position >= the use block's position that is not already claimed — in SQLite view terms, a correlated MIN(position) subquery on the result side plus DISTINCT on the use side; if the correlated form is too slow for hot paths, materialize the pairing at index time (action…
Done when: A fixture session with re-emitted (use,result) pairs sharing tool_id yields exactly one action row per logical use; the live sample session's action count drops accordingly (before/after recorded); aggregate goldens updated with the delta explained; empty-string guard in place.

### 183. `polylogue-013x` — search_text excludes Write-tool file bodies (tool_input.$.content) — undocumented coverage gap

P3 task · open
Intent: Construct-validity hunt 2026-07-06: blocks.search_text (archive_tiers/index.py:215, generated column) concatenates text + tool_name + tool_input $.command/$.file_path/$.path — but NOT $.content, so code an agent WROTE (Write/Edit tool bodies) is invisible to FTS unless it also appears in prose or a tool_result echo.
Execution shape: Decide deliberately, then document: option A (include) — extend the generated column with COALESCE(json_extract(tool_input,'$.content'),'') capped/truncated (Write bodies can be huge; FTS index size impact must be measured on the live archive first — a size probe belongs in the decision evidence), derived-tier regime: DDL edit + index rebuild, batch per 60i5; option B (exclude, document) — docs/search.md gains a searchable-content matrix (block…
Done when: docs/search.md contains the searchable-content matrix matching the live generated column (drift-checked by a test extracting the DDL expression); if include: rebuild plan executed + size delta recorded; a fixture proves a Write-tool body is findable (A) or that the documented workaround finds it (B).

### 184. `polylogue-0dz` — Chunked/streaming read-package layout for huge exports

P3 task · open
Intent: DEMO-RADAR open question after full-chatlog exports produced huge single JSON/Markdown files: move read-package full-transcript layouts toward a chunked/streaming layout (per-window files + index manifest).
Execution shape: Huge exports (multi-GiB Claude Code JSONL) already stream on INGEST; the READ side (read --all/session dumps, web payloads) still materializes whole sessions.
Done when: Reading the largest live session streams in bounded memory (measure RSS before/after); manifest + segments round-trip to identical content; web/CLI consume the same layout.

### 185. `polylogue-1hj` — Blackboard as agent comms: cross-session messages that actually arrive

P3 task · open · parent: polylogue-s7ae
Prerequisites: polylogue-d1y, polylogue-37t.4.
Intent: Raw-log 05-08, uncaptured: a groupchat-ish channel for agents, subagents, and operator.
Execution shape: (1) Extend blackboard rows: scope (repo | session-tree | broadcast | direct:session-ref), ttl, per-session delivered_at receipts.
Done when: Repo-scoped post appears in the next session's preamble and marks delivered; session-tree scope reaches a spawned subagent live; caps/ttl enforced; CLI+webui board surfaces work; delivery events queryable.

### 186. `polylogue-s7ae` — Agent coordination substrate: evidence-backed multi-agent work without tracker lock-in

P1 epic · open
Intent: Why: Polylogue should make concurrent agent work operational, not merely visible.
Execution shape: Core shape: add a reusable coordination envelope, not a web-only mission-control feature.
Done when: A typed coordination envelope exists and is queryable without assuming Beads.

### 187. `polylogue-1vpm.3` — Generic artifact edges: produced/consumed/mentioned/reported_by/derived_from across sessions, runs, delegations

P3 task · open · parent: polylogue-1vpm
Intent: One derived relation linking archive objects to artifacts with edge type + evidence refs + confidence + extractor version — replacing the temptation to special-case .agent/scratch, report markdown, evidence packs, PR summaries, or sidecars (raw_artifacts already proves artifact identity is a storage concern: source_path, artifact_kind, link_group_key, sidecar_agent_type; the missing piece is the graph edge).
Done when: Delegation/episode/gjg/rxdo artifact needs all satisfiable through this one relation (no per-program artifact tables); edges queryable from the DSL (artifact.kind/artifact.path fields on owning units).

### 188. `polylogue-1vpm.5` — Correction-edge runtime query: resolve correction assertions to corrected blocks/tools/models

P3 task · open · parent: polylogue-1vpm
Intent: Error-rate-per-tool and correction-density measures need correction assertions joined to what they corrected.
Done when: Each anchor grain resolves to exactly its honest field set; unresolved visible; policy check rejects persistent cross-tier views; measures over the edge respect anchor-grain caveats.

### 189. `polylogue-1vpm` — Work-graph units: delegation, episode, artifact edges — the derived units between lineage and analysis

P2 epic · open
Intent: Three convergent derived units that make "what work actually happened" queryable, sitting ABOVE within-provider lineage (session_links stays the leaf truth) and BELOW analysis runs.
Done when: delegations where / episodes where work as terminal units with set-algebra participation; fixtures prove no false subagent from bare forked_from_id and no acompact false-delegation; episode default render includes only linked+corroborated tiers; per-edge signal contributions auditable in evidence_json; operator stitch decisions round-trip as assertions and constrain rebuilds.

### 190. `polylogue-20d.11` — Read-profile mmap tuning: raise READ_MMAP, lower double-buffering cache

P3 task · open · parent: polylogue-20d
Intent: Readers get 32MiB cache / 128MiB mmap (connection_profile.py:84-87) against a 23GB index — mmap covers 0.5%.
Execution shape: Raise READ_MMAP_SIZE_BYTES to 2-4GiB; simultaneously LOWER cache_size on the read profile (SQLite's page cache double-buffers what mmap already maps).

### 191. `polylogue-20d.16` — Performance/throughput scenario family

P3 task · open · parent: polylogue-20d
Intent: Scenario family for perf/throughput regression: seed archives at three scales (demo-size, 10%-of-live sample shape, live-shape synthetic) via scenarios/ + corpus_seeded_db infra; measured flows = ingest batch, rebuild-index, hot find query set, read --all of largest session, convergence catch-up.
Done when: polylogue lab perf (or devtools equivalent) runs the family and diffs against baseline; one seeded regression (sleep injection) is caught; baselines refreshed with rationale in the same PR that changes them.

### 192. `polylogue-20d.7` — EQP sweep + dbstat census on a live-archive copy

P3 task · open · parent: polylogue-20d
Intent: Systematic plan audit: monkeypatch sqlite3 execute in a pytest session against a reflink copy (cp --reflink index.db /realm/tmp/eqp-copy.db), log EXPLAIN QUERY PLAN during a scripted tour of every CLI verb + MCP insight tool; grep for SCAN and USE TEMP B-TREE.
Execution shape: Method: monkeypatch sqlite3 execute in a pytest session against a reflink copy (cp --reflink index.db /realm/tmp/eqp-copy.db) logging EXPLAIN QUERY PLAN during a scripted tour of every CLI verb + MCP insight tool; grep SCAN and USE TEMP B-TREE.

### 193. `polylogue-iec` — Schema optimization audit: storage shape earns its bytes and its reads

P2 task · open
Prerequisites: polylogue-20d.7.
Intent: The schema grew by accretion; nobody has audited its SHAPE for cost since v1: column-level waste (JSON blobs where typed columns are read; denormalizations nothing reads — 9e5.3 is the value side, this is the storage side), missing hot denormalizations (session_stats; search_text possibly duplicating message text at scale), TEXT primary keys where INTEGER rowid joins would halve index size (measure before touching — may be a deliberate loss), STRICT tables (free type-safety the DDL predates), p…
Execution shape: Evidence then surgery, migrations-v2 as the vehicle.
Done when: Census artifact committed (live archive, bytes by table/index per tier); ranked list with measured deltas on top 3; one executed change with before/after size + latency; never-used indexes dropped or justified.

### 194. `polylogue-20d.8` — Bound claim-vs-evidence regen latency (43s on live archive)

P3 task · open · parent: polylogue-20d
Prerequisites: polylogue-20d.10.
Intent: Likely falls out of the action-unit outcome fields work (SQL-side pairing instead of Python row inspection).
Execution shape: Hinge: the pairing cost is Python-side row inspection; 1vpm action-unit outcome fields move it into SQL.

### 195. `polylogue-20d` — Interactive performance: the front door answers in interactive time

P2 epic · open
Intent: Cold CLI invocations pay ~2s of Python imports; some helps took 5-9s; find-then-select cold spikes; claim-vs-evidence regen 43s; ingest catch-up crawled at 0.2 files/s.
Execution shape: Front-door interactive-latency spine.
Done when: - The 20d.14 interactive SLO tier is defined in docs/plans/slo-catalog.yaml and runs green in `devtools bench slo` against the seeded corpus with a live daemon.

### 196. `polylogue-212.5` — D5 'The session that watched itself': live capture proof

P3 task · open · parent: polylogue-212
Intent: Live dev session with polylogued tailing; mid-session, query the archive for THIS session — messages typed a minute ago come back through MCP with ingest-cursor timestamps proving capture latency; end by generating the session's own postmortem before it ends.
Execution shape: The reflexive capture proof: run an agent session ABOUT polylogue while browser-capture + hooks record it, then produce the archive's account of that same session (timeline, tool calls, cost, claims) as a Demo Finding Packet (212.7 contract).
Done when: A committed packet under .agent/demos/ where the recorded session's evidence (tool timing, exit codes, cost) annotates the session's own narrative; regeneration instructions work cold.

### 197. `polylogue-212.6` — D8 'Pick up where I left off': abandoned-session triage to live continuation

P3 task · open · parent: polylogue-212
Prerequisites: polylogue-tsk.
Intent: The memory-product moment as a demo: find_abandoned_sessions surfaces real abandoned work ranked by resumability; get_resume_brief composes the evidence-cited brief (every line resolvable); the operator picks one and actually continues it in the harness.
Execution shape: Chain of existing primitives: find_abandoned_sessions -> get_resume_brief -> resume routing (37t.8 owns session->invocation mapping; until it lands, the demo ends with the composed `claude --resume <id>` command printed).

### 198. `polylogue-212.9` — Fable-as-Foreman: subagent-delegation rhetoric report (the X-post demo)

P3 task · open · parent: polylogue-212
Intent: The corpus demo idea that started as a tweet hook: extract every subagent delegation from the operator's Claude Code sessions and analyze the FOREMAN RHETORIC — how the orchestrator instructs its subagents.
Execution shape: Substrate path: (1) delegation extraction rides 1vpm.1 (delegations derived unit — parent_session_id x tool_use_block_id identity); interim: Task-tool tool_use blocks are queryable today via the actions view without waiting for the full unit.
Done when: A committed packet under .agent/demos/ with specimen gallery + aggregate tables over the live archive's real delegations; every aggregate number resolves to query + refs; the public-artifact variant passes the privacy review or is explicitly held private.

### 199. `polylogue-212.7` — Demo Finding Packet contract + prompt runner + registry manifest

P2 task · open · parent: polylogue-212
Prerequisites: polylogue-212.9.
Intent: Convert 212 from a shelf of named demos into a PORTFOLIO CONTRACT: every demo is an executable PROMPT.md handed to a coding agent, and every prompt emits the identical Demo Finding Packet: PROMPT.md, finding.yaml (five-part provenance stanza per 3tl.4: archive cursor, measure/query version, commit SHA, sample-frame predicate, run date), report.md (fixed section order: claim, corpus, method, findings, specimens, counterexamples, limits, reproduce), evidence.ndjson (one row per cited ref), querie…
Execution shape: Anchor: .agent/demos/ (existing shelf: agent-forensics, claim-vs-evidence, degraded-archive-proof, CURATED_CATALOG.md as the manifest seed).
Done when: Packet schema documented + validated by the runner; one existing demo (D1 receipts) re-emitted through the runner produces a conforming packet on the seeded corpus; registry manifest lint catches a missing packet.

### 200. `polylogue-212` — Demo portfolio: construct-valid demos (D1/D2/D4/D5/D8 + post-hoc forensic Q&A)

P2 epic · open
Intent: Ground rule for all: every displayed number resolves, on click or --explain, to structural evidence (outcome fields, usage events, provenance refs, raw bytes) — never regex over prose.
Execution shape: Portfolio contract (see 212.7): every demo = executable PROMPT.md emitting the uniform Demo Finding Packet; product primitives only, shell as glue; anti-demo (212.8) ships beside successes.
Done when: Each demo child (212.1 post-hoc forensic Q&A, 212.2 D1, 212.3 D2, 212.4 D4, 212.5 D5, 212.6 D8) ships in two variants: (a) a public seeded-corpus variant (seed 1843) reproducible with one documented command, and (b) a live-archive operator variant.

### 201. `polylogue-37t.13` — Revisit beads<->assertions boundary once beads-history ingestion (7fj) lands

P3 task · open · parent: polylogue-37t
Intent: Re-examine the polylogue-lnd boundary once polylogue-7fj ingests beads issue history as a Polylogue evidence source.
Done when: A short decision note (comment on this bead or a new decision bead) records, per seam, whether the lnd boundary held or was adjusted after 7fj landed; if close-reason->candidate-assertion composition is adopted, it is captured as its own execution-grade bead rather than done ad hoc.

### 202. `polylogue-37t.16` — Claim-kind -> allowed grounding-class compatibility registry

P3 task · open · parent: polylogue-37t
Intent: The generalized recovery-digest fix as a small declarative registry: each claim kind declares which anchor classes can prove it (pr_merged: external_pr | git_commit | tool_result, never agent_session/transcript-quote; assistant_said: raw transcript IS correct grounding; command_outcome: tool_result/exit-code keystone; decision_made: human_message | human_judgment).
Execution shape: Registry rows next to the measure-registry discipline (9l5.7 sibling — a claim kind is an operationalization with validity metadata).
Done when: Compatibility matrix is table-driven + tested; a transcript anchor sets compatible_claim=0 for pr_merged and 1 for assistant_said; verdicts change accordingly.

### 203. `polylogue-37t.17` — Read-access log + memory-utility analytics: which injected memories earn their tokens

P3 task · open · parent: polylogue-37t
Intent: The signal the context scheduler (37t.11) needs and cannot get today: a read-access log (ops.db — already multi-writer via daemon events) recording which assertions/memories/packs were injected, read, expanded, or ignored, with in-process debounce + decayed counters.
Done when: Injection + read events land in ops.db with debounce; a memory-utility report ranks injected-vs-used; scheduler ranking consumes attention WITHOUT context_inject events (test proves the exclusion); dead-memory candidates emitted, never auto-deleted.

### 204. `polylogue-3tl.10` — Launch kit: announcement artifacts prepared so publication is one decision

P3 task · open · parent: polylogue-3tl
Prerequisites: polylogue-cfk.
Intent: 'Announce widely' fails when composed under adrenaline on launch day.
Execution shape: Prepared artifacts, all in-repo (docs/launch/ or .github/), all public-safe: (1) Show HN post: title options + body (the one-command demo front and center; HN wants running code and honest limitations — the epistemics story IS the differentiator there); (2) blog/announcement post: the category-naming essay (system of record for AI work; the deleted-feature-that-guessed story; the finding as proof) — publishable on the pages site; (3) the compari…

### 205. `polylogue-3tl.14` — Fix FTS + entry-point doc drift: internals.md describes external-content FTS; CLAUDE.md misdescribes operations/archive.py

P3 task · open · parent: polylogue-3tl
Intent: Found during 2026-07-06 full-codebase grok (see .agent/scratch/codebase-grok-2026-07-06.md).
Execution shape: Rewrite the internals.md FTS5 Model bullet list to describe the contentless design accurately: contentless + contentless_delete=1 over blocks.search_text (VIRTUAL), rowid-keyed triggers, no snippet()/rebuild/integrity-check support, and point to the replacement machinery (fts_freshness_state ledger, docsize-vs-idx_blocks_search_text_populated comparison, per-session repair #1851, convergence stage) as the intended net.
Done when: internals.md FTS5 Model section matches archive_tiers/index.py DDL (contentless, blocks-based, trigger names, replacement repair machinery); CLAUDE.md operations row and MCP tool-registration gotcha corrected.

### 206. `polylogue-3tl.15` — Anti-grep proof card: the "why not grep ~/.claude" answer, grounded in one finding

P3 task · open · parent: polylogue-3tl
Intent: README/docs must answer the strongest skeptical reader directly: grep finds text; Polylogue resolves provider structure — paired tool calls/results, exit-code/is_error failure predicates, costs, lineage, typed units, evidence-backed derived claims — and can say whether the agent NOTICED the failure.
Execution shape: Target: README.md (skeptic section) + docs site page.
Done when: Cold reader can distinguish lexical search from the structured evidence model in one screen; card links one finding URL + one seeded reproduction command; doc-command lint passes; bare-query teaching removed or corrected to signalled forms.

### 207. `polylogue-3tl.3` — Claim-vs-evidence leaderboard variant (multi-model, incl. open models)

P3 task · open · parent: polylogue-3tl
Intent: Comparative multi-model variant of the finding: silent-proceed / unsupported-claim rates across models including the open models present in the archive, with cost/cache columns.
Execution shape: Reuse the claim-vs-evidence harness (campaign artifacts under .agent/demos/claim-vs-evidence) — the variant axis is MODEL: silent-proceed / unsupported-claim rates per model family over identical task classes, with cost + cache columns from f2qv-honest accounting.

### 208. `polylogue-3tl.8` — GitHub surface polish: the repo page itself is a landing page

P3 task · open · parent: polylogue-3tl
Intent: Strangers arrive at github.com/Sinity/polylogue before any docs site: the repo description, topics, social-preview card, pinned content, badge row, and issue-template experience ARE the first screen of the product.
Execution shape: One audit-and-set pass, most of it gh api-scriptable and therefore agent-executable: (1) repo description = the one-liner from the positioning analysis; topics (ai, sqlite, local-first, claude, chatgpt, archive, observability, agent-memory...) — topics drive GitHub search discovery; (2) social preview image: rendered card (name + category line + one screenshot) — the visual-tapes/atlas assets feed it; (3) badge row in README: CI, PyPI version, l…

### 209. `polylogue-4ts.7` — Physical session identity collision beneath origin collapse: same native_id, two source families, one row

P3 task · open · parent: polylogue-4ts
Prerequisites: polylogue-2qx.
Intent: Beneath the aggregate origin-collapse bug: session_id = origin:native_id means a gemini-export and a drive-takeout session sharing a native_id are ALREADY ONE PHYSICAL ROW — undetectable, un-splittable even by reparse, and aggregate lossy_grouping markers (2qx wiring) fix aggregation honesty ONLY, not identity.
Done when: Census artifact exists with confidence labels; design doc reviewed; fixture proves the target model.

### 210. `polylogue-4ts` — Session lineage truth: shared content stored once, counted once, composed correctly

P1 epic · open
Intent: Fork/resume/compaction share content; storage+aggregates ignored it.
Done when: Terminal state: shared content stored once (prefix dedup verified on live archive), counted once (4ts.2), composed correctly (read paths serve full logical transcripts across the branch-point matrix); external citation of archive counts uses logical grain with the physical figure footnoted.

### 211. `polylogue-5en` — Branch-local daemon/web/extension dev loops: verify remaining AC and close out

P3 task · open
Intent: Largely realized (devtools workspace dev-loop launcher is the devloop default).
Execution shape: Branch-local devloop exists for the daemon (dev-loop payload in daemon/http.py _dev_loop_payload; devloop-review warns on stale run dirs).
Done when: Each surface (daemon/web/extension/MCP) has a documented branch-local recipe that two concurrent branches can run without cross-talk; stale-run-dir detection warns in at least the daemon + web cases.

### 212. `polylogue-703` — One status assembly: daemon/status.py, cli/commands/status.py, and workload diagnostics converge

P3 task · open · parent: polylogue-t46
Intent: Status/health is computed at least three times: daemon/status.py (2,418 lines), cli/commands/status.py (1,892 lines, its own _table_exists and direct DB probing), and ops diagnostics workload.
Execution shape: Inventory first: diff the fact sets each of the three computes (they overlap ~60-80% by eyeball: archive tier presence/sizes, FTS readiness, counts, daemon liveness, embedding coverage, debt).

### 213. `polylogue-7le` — Consolidate the three session->HTML paths

P3 task · open · parent: polylogue-t46
Intent: Three independent renderers: rendering/renderers/html.py, the web shell's hand-rolled JS, and a third path (fables pass; re-verify inventory).
Execution shape: The three session->HTML paths to consolidate: (1) polylogue/rendering/core_messages.py + rendering/blocks.py (canonical block/message renderers), (2) daemon web shell (polylogue/daemon/web_shell*.py) which re-renders for the SPA, (3) the CLI read/export HTML view path (read_view_handlers.py --view html lane).
Done when: One HTML rendering entry point; web-shell and CLI HTML outputs diff-clean vs before (or intentionally improved with goldens updated); no duplicated block-type dispatch tables remain.

### 214. `polylogue-8jg9.3` — SLO samples + idle-vs-stalled verdict: steady-state observability over convergence

P3 task · open · parent: polylogue-8jg9
Intent: The honesty keystone for daemon observability: backlog>0 is a defect ONLY when work is offered and not draining — idle backlog and stalled backlog are different verdicts, and conflating them trains operators to ignore alerts.
Execution shape: Anchors: daemon_events + cursor-lag tables/samplers already exist (ops.db; daemon/cursor_lag_*.py modules) — this ADDS slo_samples (closed-set label enum, retention GC) + reducers (level/quantile/slope/ETA/burn-rate) as pure functions over those tables, and the idle-vs-stalled verdict: stalled = offered_work > 0 AND drain_rate == 0 over the window; idle = backlog with no offered work.
Done when: Stalled reported only when offered-and-not-draining; bulk import does not fire ingest SLO; reducers degrade honestly on cold start (level-only); retention bounds table size.

### 215. `polylogue-9e5.10` — Resume/context efficacy eval (observational)

P3 task · open · parent: polylogue-9e5
Intent: For sessions that actually were continuations: compare what get_resume_brief/compose_context_preamble WOULD have injected vs what the successor session actually had to rediscover (searches, re-reads of files the brief cites).
Execution shape: Observational (no A/B): join resume-shaped evidence the archive already has — get_resume_brief/compose_context_preamble usage (hook events, MCP call logs in ops.db), session_links resume chains, and outcome proxies (time-to-first-edit, early-tool-error rate, repeated-orientation queries) — comparing resumed-with-context vs resumed-bare sessions on the same repo.
Done when: A committed analysis artifact over the live archive: n per arm, the 3-4 outcome proxies with uncertainty, confounders section, and a verdict on whether the controlled cfk result generalizes.

### 216. `polylogue-9e5.11` — Test-suite economics: coverage vs fix-density map

P3 task · open · parent: polylogue-9e5
Intent: 248k test lines vs 229k product lines, yet the embedding-staleness defect was untested.
Execution shape: Map where tests earn their runtime: per-module (a) coverage percent, (b) historical fix-density (git log --grep fix -- <module> commit counts), (c) test wall-time share (.cache/verify pytest artifacts have per-test durations), (d) testmon selection frequency.
Done when: Committed matrix for every polylogue/ package; five concrete actions each with expected effect (minutes saved or risk covered); actions filed as beads or done inline.

### 217. `polylogue-9e5.12` — Schema-inference ROI: load-bearing or gold-plated?

P3 task · open · parent: polylogue-9e5
Intent: ~13.5k lines in schemas/: trace what actually consumes generated packages at runtime vs test-time, and whether drift detection ever fired on a real provider change.
Execution shape: Question: does the schemas/ package (Pydantic provider-record validation driving detect_provider tightness) earn its maintenance cost?
Done when: Per-provider verdict table committed with the three measurements; at least one simplify/tighten action executed or beaded; detector-order tests still green.

### 218. `polylogue-9e5.13` — Doc-vs-code drift diff -> one docs-correction PR

P3 task · open · parent: polylogue-9e5
Intent: Diff docs/ claims against established code truth: 'idempotent by content hash' phrasing vs the identity-hash fallback history, stale pathology docstring, blob_links naming, anything the deep-dive corrected that docs still assert.
Execution shape: One diff pass, one PR: for each doc in the Reference-docs table (CLAUDE.md lists them), extract checkable claims (file paths, command names, schema versions, table names, tool counts) and verify against live source mechanically where possible (paths exist, commands in --help, versions match constants).
Done when: Every reference doc swept; each stale claim fixed in the PR or beaded with reason; the extraction script committed so the sweep is repeatable.

### 219. `polylogue-9e5.14` — Facade decomposition map: which of api/archive.py's ~126 methods each surface uses

P3 task · open · parent: polylogue-9e5
Intent: Call-graph CLI/MCP/daemon usage of the facade's methods; propose split boundaries along observed clusters rather than guessing.
Execution shape: polylogue/api/archive.py is 5391 lines (verified).
Done when: Committed table covers 100% of the facade's public methods with consumer counts + verdicts; tests-only and zero-consumer methods explicitly listed (candidates for deletion).

### 220. `polylogue-1fp` — Facade decomposition: split api/archive.py into per-capability protocols

P3 task · open · parent: polylogue-t46
Prerequisites: polylogue-exb, polylogue-9e5.14.
Intent: api/archive.py is a 5,259-line, 126-method God-facade; every surface (CLI, MCP, daemon, devtools) imports the whole Polylogue object to use its own small slice.
Execution shape: Shape: capability protocols (QueryReads, SessionReads, InsightReads, AssertionWrites, MaintenanceOps, EmbeddingOps...) defined next to their implementations; the Polylogue facade becomes a thin composition root that constructs and hands out protocol views — kept for the public library API (docs promise it), but internal surfaces import their protocol, not the facade.

### 221. `polylogue-9e5.15` — Dead-code and script-silo sweep: coverage-informed removal audit

P3 task · open · parent: polylogue-9e5
Intent: 231k lines of product code accumulated through fast agent-driven iteration statistically carries dead weight, and nothing hunts it systematically: unreferenced functions/branches survive because tests import broadly and mypy checks reachability, not use.
Execution shape: (1) Three independent signal sources, intersected (each alone is too noisy): vulture (static unreferenced-symbol candidates), coverage data from the FULL suite run (devtools verify --all already produces it — lines never executed by any test), and the affordance-usage/tasks history for CLI/devtools entry points (registered but never invoked).
Done when: Audit artifact with the three-signal intersection committed; at least one batched deletion PR merged with net-negative diff and all gates green; scripts/ directory removed or reduced to zero Python; the lane is invocable via devtools and documented.

### 222. `polylogue-9e5.16` — Python API parity: the library surface audited against CLI/MCP capabilities

P3 task · open · parent: polylogue-9e5
Prerequisites: polylogue-9e5.14.
Intent: The async library API (api/__init__.py, documented in library-api.md) is a promised public surface, but nothing checks that it kept pace: capabilities added CLI-first or MCP-first (followup_class queries, logical sessions, corrections, context images, embeddings ops, observed-event pipelines) may or may not be reachable from the library, and library-api.md may describe an older shape.
Execution shape: (1) Generate the capability matrix: CLI commands (inventory) x MCP tools (EXPECTED_TOOL_NAMES) x facade methods (9e5.14 map) -> a three-column reachability table; every capability classified: all-surfaces / intentionally-surface-specific (document why) / drifted (fix or deprecate).
Done when: Capability matrix generated and committed as a rendered doc; every drifted row either fixed or documented as intentional; api-doc symbol check wired into verify; library-api.md accurate against the live facade.

### 223. `polylogue-9e5.18` — Wire atheris fuzz targets into CI

P3 task · open · parent: polylogue-9e5
Intent: tests/fuzz exists but runs nowhere.
Done when: Scheduled workflow green on a first run; a seeded crash (assert False target) produces an artifact + notification path; README-of-fuzz documents adding a target.

### 224. `polylogue-9e5.20` — Flakiness tracking + quarantine lane

P3 task · open · parent: polylogue-9e5
Intent: Flakiness is currently folklore (the 3.11 test_concurrent_reads_during_writes memory).
Done when: Flakiness ledger generated from existing artifacts; the known 3.11 flake appears in it; quarantine marker exists with lint requiring owner-bead ref; CI treats quarantined failures as warnings.

### 225. `polylogue-9e5.21` — Mock-depth measurement

P3 task · open · parent: polylogue-9e5
Intent: Measure where tests mock so deep they test the mocks: AST scan of tests/ counting patch targets per test, patch depth class (own-module boundary vs foreign-internal vs stdlib), and assert-on-mock ratio (asserts against Mock attrs vs real outputs).
Done when: Committed mock-depth report over tests/unit; three worst offenders converted to infra-backed tests with equal-or-better assertions; scan script re-runnable.

### 226. `polylogue-9e5.22` — Per-module coverage tracking (beyond aggregate floor)

P3 task · open · parent: polylogue-9e5
Intent: The 90% aggregate floor hides per-module holes (a 60% storage module offset by 99% rendering).
Done when: Per-package floors active in CI; lowering a module below its floor fails; worst-3 report visible in verify output; floors documented as ratchet policy.

### 227. `polylogue-9e5.23` — Extend coverage-manifest schema to accept bead: owners (retire gh#590 issue-refs)

P3 task · open · parent: polylogue-9e5
Intent: devtools/verify_manifests.py coverage_gaps require an `issue:` (GH decimal) or `suppression:` and a strict schema forbids extra keys, so a `bead:` owner is rejected.
Done when: coverage manifests reference bead owners instead of gh#590; devtools verify manifests passes with `bead:` fields; the 9 gaps show their bead id.

### 228. `polylogue-9e5.5` — Exhaustive table read/write matrix -> dead-table kill list

P3 task · open · parent: polylogue-9e5
Intent: Parse every SQL string in the repo; build writer/reader site counts for all ~54 tables.
Execution shape: Method (the at44 pattern generalized — user_settings was found dead exactly this way): for every table across the five tiers, classify READ (rg for SELECT/FROM in polylogue/ excluding migrations/DDL), WRITE (INSERT/UPDATE/DELETE), and DDL-only.
Done when: Committed matrix covers every CREATE TABLE across all five tiers; each dead/zombie row has a verdict (drop in next bump / keep with reason / wire like at44); re-runnable script so the matrix cannot rot.

### 229. `polylogue-9e5.6` — Hash-boundary census: classify every digest producer/consumer

P3 task · open · parent: polylogue-9e5
Intent: Enumerate every digest producer (core/hashing.py, write.py _hash_bytes sites, blob store, snapshot fingerprints, paste evidence) and every consumer of a content_hash column; classify each comparison as meaningful or vacuous.
Execution shape: Census every digest producer/consumer: content-hash identity (core/hashing.py NFC-normalized session hash), blob store SHA-256 (storage/blob_store.py, raw_id), attachment hashes (#2469 path), embeddings recipe/chunk hashes, FTS nothing, backup manifests, dolt/beads external.
Done when: Committed census covers every hashlib/sha call site in polylogue/ (rg-verified count matches); each row states inclusion contract + consumer; the register-or-fail lint runs in devtools verify --quick or documented as follow-up.

### 230. `polylogue-9e5.7` — Daemon loop interaction model: lock/starvation map for the ~9 concurrent loops

P3 task · open · parent: polylogue-9e5
Intent: daemon/cli.py runs ~9 concurrent while-True loops (FTS merge, WAL checkpoint, drive catchup, convergence, health...) against the same SQLite set as live ingest.
Execution shape: Map every long-lived daemon loop (convergence driver daemon/convergence.py, watcher/ingest loops in daemon/cli.py, fts_automerge.py, embedding catch-up, cursor-lag samplers, http server thread) against: which SQLite connection/lock class it holds, blocking vs async, backoff shape, and what starves it (the single-writer invariant means one hot loop can starve the rest).
Done when: Committed table covers every loop the daemon spawns (enumerated from daemon/cli.py startup, cross-checked against live polylogued thread/task dump); each starvation risk has evidence or an explicit not-reproducible note.

### 231. `polylogue-9e5.8` — Provider->Origin completion map: sequenced retirement plan

P3 task · open · parent: polylogue-9e5
Intent: Exact inventory of the 20+ remaining Provider imports, each classified wire-legitimate vs retirable, plus the non-injective GEMINI/DRIVE->AISTUDIO_DRIVE collapse consequences.
Execution shape: The sequenced retirement plan for provider vocabulary (tracked context: 9e5 epic; jnj.7 owns the CLI-help slice; 2qx owns OriginSpec).
Done when: Committed sequence doc lists every provider-token surface with class + flip step; no step breaks public output byte-compat (goldens); final state = Provider importable only from wire modules (layering lint enforces).

### 232. `polylogue-9e5.9` — Heuristic accuracy benchmark: keyword classifiers vs hand-labeled truth

P3 task · open · parent: polylogue-9e5
Intent: Hand-label ~100 sessions for work-event type and terminal state; score the keyword classifiers (extraction.py hard-coded confidences) against structural ground truth.
Execution shape: Operator direction (2026-07-03): standardize as a REPEATABLE lane, not a one-shot audit — automation/standardization, explicitly not a CI gate.

### 233. `polylogue-b0b.1` — Fix substring false-positives in work-event keyword classifier + inventory activity-type label as heuristic-tier

P2 task · open · parent: polylogue-b0b
Prerequisites: polylogue-9e5.3, polylogue-9e5.9.
Intent: polylogue/archive/session/extraction.py:_text_signal_from_lowered_text (241-262) matches keyword tables with naive `pattern in lowered_text`, so tokens embed into unrelated words: 'fix'->prefix/suffix, 'test'->latest/contest, 'plan'->explanation/airplane, 'data'->update/validate/metadata, 'spec'->respect/inspect, 'move'->remove, 'config'->reconfigured.
Done when: 1.

### 234. `polylogue-b0b` — Replace remaining keyword outcome/pathology heuristics with structural evidence

P2 task · open
Prerequisites: polylogue-9e5.3, polylogue-9e5.9.
Intent: Includes High-Value backlog: wherever detectors/insights still regex prose for outcomes, consume tool_result_is_error/exit_code instead, with per-origin coverage caveats where structure is absent.
Execution shape: Inventory first: rg detector/insight modules (insights/, schemas/code_detection/, pathology surfaces) for prose-pattern matching — regexes over message/block text that infer outcomes ('error', 'failed', 'fixed', success words).
Done when: Inventory table committed (module:line -> verdict); every CONVERT lands with a coverage caveat; every retained heuristic emits evidence_tier=text_derived; the fabrication fixture (prose claiming an event that structure contradicts) does not surface as fact on any public payload.

### 235. `polylogue-9jsi` — Polish search recall: pl_fold write/query symmetry + remove_diacritics 2 + measured trigram lane

P3 task · open
Intent: Real recall hole for the operator corpus: unicode61 alone does not fold the precomposed l-stroke, so latwo misses łatwo — and remove_diacritics CANNOT fix ł (not a combining-mark decomposition).
Done when: latwo/zrobilem hit seeded łatwo/zrobiłem; pl_fold idempotent + Python/SQL agree; DDL-site drift test; all MATCH builders normalized or explicitly out of scope; trigram off until benchmarked with size+precision report.

### 236. `polylogue-9l5.19` — Thinking-vs-doing drift: experimental coverage-gated measure of reasoning share vs tool-active share

P3 task · open · parent: polylogue-9l5
Intent: Early signal that a model got worse (or a harness got wasteful) for YOUR work: compare reasoning/thinking effort against tool-active time, trended by model family, repo, workflow shape, and month.
Execution shape: Candidate definitions, each emitted only where provider fields support it, else insufficient_evidence: thinking_token_share = reasoning_tokens/total_output_tokens (Codex output includes reasoning — see token-semantics memory; Claude thinking blocks where present); thinking_wall_share = model_thinking_duration_ms/session_wall_ms; tool_active_share = tool_duration_ms/session_wall_ms.
Done when: Measure registered with coverage gate semantics (per-provider availability matrix); emits insufficient_evidence rather than fabricating where fields are absent; a trend query by model/month works over the live archive; no composite score surface.

### 237. `polylogue-9l5.13` — activity_spans materializer: edit/test/build/idle/delegate intervals with evidence tiers

P2 task · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.19.
Intent: The missing bridge between raw structure and "so what": a derived queryable relation of time-bounded work spans composed OVER existing substrate (actions keystone fields, phases 5-min-gap intervals, weak work-event labels, observed events, run projection) — a normalizer/composer, not a new capture pipeline.
Done when: Seeded corpus produces spans with evidence refs; >threshold gaps are idle spans; structural test failure yields kind=test outcome=failed; activity-spans where session.repo:X | group by kind | sum duration_ms works (DSL terminal unit + fields registered as part of this bead); heuristic-classified spans carry the tier visibly.

### 238. `polylogue-9l5.14` — Efficiency measure pack v1: scorecard vector over spans/episodes/delegations — no magic score

P3 task · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.13.
Intent: Register over activity_spans + tool_episodes + delegations (1vpm.1): active_time_ratio, idle_gap_ratio, edit_test_cycle_count, failure_recovery_latency (failed test/command -> next structural success), silent_proceed_after_failure_rate (the hero-finding measure, productized), verification_after_edit_rate, tool_failure_rate, cost_per_structural_success, delegation_fanout/return_latency/used_result_rate/rework_rate, context_churn_ratio.
Done when: Each measure has a full registry row (construct/formula/frame/tier/confounds/suppress-when per 9l5.7); cross-origin comparison without coverage labels refuses; scorecard renders on the seeded corpus with footnotes.

### 239. `polylogue-9l5.7.1` — Tag Layer-0 substrate insight payloads with evidence_tier so consumers can read rule-heuristic vs structural confidence

P3 task · open · parent: polylogue-9l5.7
Prerequisites: polylogue-9l5.19.
Intent: The pre-existing Layer-0 payloads (WorkEvent, SessionInferencePayload, SessionEnrichmentPayload, workflow_shape, terminal_state) expose a bare `confidence: float` to MCP/API/dashboard consumers.
Done when: 1.

### 240. `polylogue-9l5.7` — Statistics substrate + measure registry: uncertainty primitives with construct-validity metadata

P2 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.19.
Intent: The keystone of the analytics tower.
Execution shape: (1) polylogue/analytics/stats.py: pure functions over sequences — wilson_interval(k,n), quantiles_with_ci (bootstrap or order-statistic CIs), two_proportion_test, mann_whitney (rank test avoids normality assumptions on latency/cost distributions), cliffs_delta effect size, histogram_buckets.
Done when: polylogue/analytics/stats.py exists with property tests (hypothesis: interval coverage on synthetic distributions).

### 241. `polylogue-9l5.8` — Temporal analytics: trends, rolling baselines, changepoint detection

P2 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: The archive spans years of daily work but has no time-axis analytics beyond day/week summaries: no trend ('is my silent-proceed rate improving?'), no baseline ('is today's cost anomalous vs my trailing month?'), no changepoint ('did failure rates shift when I switched models / enabled hooks / upgraded the harness?').
Execution shape: (1) Series builder as a DSL stage: any measure over any unit grouped into time buckets -> a typed series ('measure tool_failure_rate window week over 2026') — one series primitive, every measure gains a time axis for free (composability payoff).
Done when: A series stage composes with any registered measure on the seeded corpus.

### 242. `polylogue-stc` — Experiment hosting: declared arms, preregistered metrics, paired analysis, agent-buildable

P2 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: Generalize what cfk/jxe did by hand into substrate: an experiment is a first-class declared object — hypothesis, arms, assignment rule, PREREGISTERED metrics (declared before data collection, timestamped — the construct-validity teeth), sample-size intent, analysis plan — and the archive hosts its lifecycle: assignment, observation collection (sessions tagged to arms), paired/grouped analysis through the measure registry, and a cold-reader-gateable report.
Execution shape: (1) ExperimentSpec as a user.db artifact (assertion-adjacent, judgment-visible): hypothesis, arms (name + treatment description), assignment (manual | alternating | by-session-property), preregistered metrics (each a registry measure ref + direction + minimum-interesting-effect), planned n, analysis plan (paired vs unpaired, test choice via 9l5.7).
Done when: cfk's protocol is expressible as an ExperimentSpec and its analysis reproduces via experiment analyze.

### 243. `polylogue-9yz` — Named bounded-dialogue layout for operator-readable windows

P3 task · open · parent: polylogue-fnm
Intent: DEMO-RADAR open question: chatlog export currently uses a first-window bounded dialogue projection; make it a reusable named layout (read-view/render profile) — bounded operator-readable dialogue window with explicit elision markers — instead of an ad-hoc max_tokens cap in one workspace command.
Execution shape: Package the existing bounded-dialogue projection as a named read-view/render profile (list_read_view_profiles surface already exists — add profile id operator-dialogue): bounded window with explicit elision markers ([...

### 244. `polylogue-a7xr.13` — api/contracts write-surface shadow adapters verify copies, not surfaces — delete or re-anchor

P3 task · open · parent: polylogue-a7xr
Intent: Abstraction audit: api/contracts/ (8 files, 977 LOC) write-surface protocols (IngestSurface, MaintenanceSurface, TagMutationSurface, SessionDeleteSurface) have zero consumers outside the package; the adapters are constructed only in the two contract test files.
Execution shape: Preferred: re-anchor assert_implements on the ACTUAL facade/handler objects (the real CLI write path and MCP tool functions) so conformance means execution-path conformance — if that is not cheaply possible, delete the shadow layer and record the parity intent on the owning issue (#859 successor = t46 golden equivalence, which tests real surfaces).
Done when: Either assert_implements binds to real handler objects (test proves a signature drift in cli/commands/ fails the contract) or api/contracts/ is deleted with intent recorded; no shadow adapter remains that reimplements handler logic.

### 245. `polylogue-a7xr.16` — Table-drive the hand-aligned column triplicates in archive_tiers write/read hot core

P3 task · open · parent: polylogue-a7xr
Intent: Abstraction audit: archive_tiers/write.py:1420-1430 hand-aligns a 30-column messages INSERT to 30 placeholders, blocks tuple-yield at :1498-1504 must stay positionally synced by hand; archive.py:4898-5780 has 14 query_* methods with 388 hand-written row[col] accessors.
Execution shape: Derive column list + placeholder string + tuple order from the row dataclasses (dataclasses.fields()) with an escape hatch for expression columns (NULL literals, _sqlite_text coercions, JSON decoders).
Done when: One source of truth per table's column order; INSERT/SELECT built from it; crud + property tests green; a deliberate column reorder in the dataclass produces correct SQL (test).

### 246. `polylogue-a7xr.4` — One percentile implementation: three algorithms across five copies skew operator-facing stats

P3 task · open · parent: polylogue-a7xr
Intent: Divergence audit: _percentile exists 5x with three algorithms — linear interpolation (daemon/status.py:1306, daemon/live_ingest_attempt_progress.py:167, daemon/cursor_lag_baseline.py:320), nearest-rank q-in-[0,1] (insights/portfolio.py:107), nearest-rank p-in-[0,100] (archive/semantic/timing.py:37).
Execution shape: core/stats.py: percentile(sorted_values, q, *, method='linear'|'nearest') (core/metrics.py is host-metrics only — new module is right); timing.py's 0-100 scale becomes call-site conversion; five deletions.
Done when: One implementation; five sites import it; a small-sample fixture (n=5) yields identical p95 across status/portfolio/timing paths (test).

### 247. `polylogue-a7xr.7` — Role synonym vocabulary maintained by hand in two directions + normalize_role name collision

P3 task · open · parent: polylogue-a7xr
Intent: core/enums.py:127-134 Role.normalize maps synonyms->canonical; archive/message/roles.py:18-24 ROLE_SQL_VALUES holds the SAME sets inverted for SQL role-filter expansion.
Execution shape: ROLE_SYNONYMS: dict[Role, frozenset[str]] once in core/enums.py; Role.normalize iterates it; ROLE_SQL_VALUES becomes a derivation/re-export (~20 lines).
Done when: One synonym table; SQL expansion derived; rename done with call sites updated; coupling test in place.

### 248. `polylogue-a7xr.8` — Index-tier sibling-path derivation pasted ~7x with divergent existence rules

P3 task · open · parent: polylogue-a7xr
Intent: Seven daemon/CLI sites re-derive 'the index.db next to this anchor' with different fallback behavior — provenance.py:194 uses the path even when absent, fts_status.py:156 returns None when missing, embedding_backlog.py:60 builds a candidate list + requires a sessions table — while paths/_roots.py:76 already exports resolve_active_index_db_path.
Execution shape: Extend polylogue/paths with sibling_index_db(anchor, *, require_exists: bool) and sweep the seven sites (convergence_stages.py:868, similarity.py:324, embedding_backlog.py:60, fts_status.py:156, provenance.py:194, cli/commands/status.py:189, daemon/cli.py:182); embedding_backlog keeps its table probe locally on top of the resolved path.
Done when: One derivation; seven sites swept; a missing-index fixture yields the SAME verdict from every status surface (test).

### 249. `polylogue-bby.1` — Workbench responsive under slow/missing routes

P3 task · open · parent: polylogue-bby
Intent: Truthful partial/stale/error state instead of blocking or silently failing; fast initial results.

### 250. `polylogue-bby.10` — Timeline and firehose: the archive as a scrubbable stream

P3 task · open · parent: polylogue-bby
Prerequisites: polylogue-20d.12, polylogue-20d.13.
Intent: Raw-log 05-09, uncaptured: (1) a scrubbable TIMELINE of all AI activity — sessions as spans on a time axis (lane per origin/repo), zoomable months-to-minutes where zoom changes semantic density (year: heat; week: session spans; hour: message/tool events) — 'what was happening around X' as a navigable surface; (2) the FIREHOSE — everything at full detail from all live sessions, one merged auto-scrolling stream (mission control bby.9 shows structure; the firehose shows content).
Execution shape: (1) Timeline: virtualized canvas/SVG lanes fed by time-bucketed aggregates the daemon cache precomputes per zoom tier (existing read models map to tiers: day summaries -> session rows -> message pages) — never raw-scan on scroll; a scrub selection EMITS the equivalent DSL time-window predicate (a timeline selection IS a query — algebra, not a silo).
Done when: Timeline scrubs year-to-hour on the live archive without raw scans (cache hits verified); scrub selection produces the DSL window predicate; firehose renders merged live sessions via SSE; 60fps interaction on the operator machine.

### 251. `polylogue-c9y` — Package topology legibility: boundary doctrine for the 28-package tree + insights/analytics vocabulary

P3 task · open · parent: polylogue-a7xr
Intent: polylogue/ has 28 top-level subpackages plus 10 loose top-level modules, and the boundaries between several are folklore: archive/ vs storage/ vs operations/ vs maintenance/ (four places 'where does archive-adjacent logic go?' can land), surfaces/ vs rendering/ vs ui/ vs cli/ (presentation split four ways, with surfaces/payloads.py as a 2.9k-line God-module), core/ vs types.py vs protocols.py vs errors.py (four homes for shared types).
Execution shape: Doctrine-first, moves-second (moves are churn; only move what actively confuses): (1) Write the boundary doctrine into docs/architecture.md placement rules as decision procedure, not description: storage/=persistence+SQL, archive/=domain semantics over storage (query, refs, topology, write effects), operations/=multi-step operator workflows, maintenance/=repair+integrity — with three worked examples each; core/ absorbs types.py/protocols.py/erro…
Done when: Placement doctrine with decision procedure + examples committed; insights-vs-analytics rule recorded before the measure registry merges; core/ absorbs the loose type/protocol/error modules (imports updated, mypy green); topology-target.yaml reflects the target and verify topology passes; 'where does X go' for five hypothetical modules is answerable from the doc alone (reviewed as part of the PR).

### 252. `polylogue-ca4` — Decision: DuckDB as the optional OLAP engine over the archive

P3 task · open
Intent: The analytics tower will stress SQLite's analytical ceiling: window functions exist but columnar scans, percentile_cont, grouping sets, and vectorized aggregation over millions of block rows are where DuckDB is 10-100x.
Execution shape: Probe (devtools lab probe duckdb, mirroring the turso probe pattern): (1) attach the live index.db read-only via sqlite_scanner; run the 5 heaviest real analytics queries (tool-episode joins, cross-provider rollups, block-level scans for process mining) both ways; measure wall/RSS.
Done when: Probe artifact with the 5-query benchmark table (wall/RSS/parity) on the live archive committed under .local/ + summarized in the bead close reason; decision recorded; if adopted, one heavy measure demonstrates dual lowering with identical results on the seeded corpus.

### 253. `polylogue-cpf.1` — Doctrine lint: reject TEXT timestamps in new durable DDL

P3 task · open · parent: polylogue-cpf
Intent: WHY: TEXT timestamps in durable DDL re-introduce the exact ambiguity the four-time-kinds doctrine exists to kill (tz-unknown, lexicographic-vs-temporal sort divergence).
Execution shape: Time doctrine: UTC epoch-ms canon.
Done when: A test DDL adding a TEXT timestamp column fails the lint; existing INTEGER epoch-ms columns pass.

### 254. `polylogue-cpf.2` — Doctrine: writer-class docstring convention + layering check

P3 task · open · parent: polylogue-cpf
Intent: WHY: writer-class modules carry implicit invariants (single-writer, tier ownership, twin-sync) that new contributors/agents violate silently; a docstring convention + layering check makes the contract visible where the code is edited.
Execution shape: Writer-class doctrine: one writer-class per file, cross-tier interruption validity.
Done when: A file declaring two writer classes fails the check; single-class files pass.

### 255. `polylogue-cpf.3` — Doctrine: injected-context trust deny-lexicon tripwire fixture

P3 task · open · parent: polylogue-cpf
Intent: WHY: injected context (recall packs, preambles, assertions) is a prompt-injection surface — a deny-lexicon tripwire fixture set proves the trust boundary holds as the injection surfaces multiply (37t rollout makes this load-bearing).
Execution shape: Injected-context trust classes (OPERATOR/SYSTEM/QUOTED).
Done when: A fixture where QUOTED content contains an OPERATOR-style directive is caught by the tripwire test.

### 256. `polylogue-cpf` — Land the six doctrines: time, writers, finding-provenance, degraded-modes, non-goals, injected-context trust

P2 epic · open
Intent: The six doctrine texts were written in full in the 2026-07-03 design session (session transcript is the source — Claude-Session trailer on this commit); they cover the cheap-to-write, expensive-to-lack gaps: time semantics (three times, UTC epoch-ms canon, skew tolerance, duration honesty), writer classes (four classes, one writer-class per file, cross-tier interruption validity), finding provenance (five-part stanza, re-runs supersede, semantic version bumps flag stale findings), degraded-mode…
Execution shape: (1) Commit texts under docs/doctrine/ (or internals sections — match the ttu docs-IA tiering), adjusted to repo voice; link from architecture-spine.
Done when: Six doctrine documents committed and indexed; the three cheap lints wired (timestamp DDL check, provenance stanza gate, trust deny-lexicon fixture); architecture-spine links them; bd memory updated to point at doctrines instead of restating them.

### 257. `polylogue-dab.1` — Drop payload_json/search_text duplication from run-projection tables; hydrate from typed columns

P3 task · open · parent: polylogue-dab
Intent: Residual over-storage that dab does NOT cover.
Done when: Materialized run/observed-event/context-snapshot reads hydrate from typed columns with no payload_json read; payload_json column removed from all three tables' canonical DDL; snapshot/parity test proves identical ProjectedRun/ObservedEvent/ContextSnapshot output before/after for a subagent+compaction fixture; before/after index.db size delta recorded; devtools lab policy schema-versioning + verify layering green; index tier rebuilt from source e…

### 258. `polylogue-dab` — Stop materializing run-projection cache rows; drop DDL after parity

P3 task · open · parent: polylogue-a7xr
Intent: After readers are source-derived: remove readiness/repair debt, reroute relation reads, stop rebuild writes/prunes for the three cache tables, then bump schema to drop DDL.

### 259. `polylogue-dx1` — Decision: daemon HTTP substrate — hand-rolled BaseHTTPRequestHandler vs ASGI

P3 task · open
Intent: The daemon serves ~45 routes from a 3,870-line hand-rolled BaseHTTPRequestHandler stack (threaded, manual route tables, manual auth, hand-rolled Prometheus exposition) inside a 29k-line daemon ring.
Execution shape: Evidence-first decision, not a rewrite crusade: (1) enumerate concrete costs paid in the last 6 months attributable to the substrate (route bugs, the bby.7 class, polling-vs-push workarounds, per-handler async bridging boilerplate — grep git log for http.py fix churn); (2) prototype ONE route family on ASGI (starlette, uvicorn, in-process) behind the same auth + a compat proxy for a week of dogfood — measure latency delta, memory delta (uvicorn…

### 260. `polylogue-f94` — Kill-or-commit the TUI (~373 lines of skeletal Textual screens)

P3 task · open
Intent: DECIDED (operator, 2026-07-03): KILL.
Execution shape: Execution list (decision already made — KILL): delete polylogue/ui/tui/ (rg first for the actual module path), its command registration in cli/ (command_inventory + click registration), its tests, and the textual dependency from pyproject if nothing else imports it.

### 261. `polylogue-fie` — Decision: archive scaling doctrine — keep everything, optimize the ceilings

P3 task · open · parent: polylogue-1xc
Intent: Doctrine settled by operator (2026-07-03): NO retention/pruning — session data is important and worth the storage; keep-everything is permanent policy.
Execution shape: Deliverable is a measured decision, not implementation: (1) measure growth rate from ops.db ingest telemetry, extrapolate 12/24 months; (2) benchmark the two worst-scaling operations (full index rebuild, FTS rebuild) at synthetic 3x/10x via the scenario generator; (3) rank optimization levers by measured payoff: blob zstd (est 36GB->5-8GB, zero data loss), separating hot/cold FTS shards, incremental-rebuild investment; (4) record the doctrine in…

### 262. `polylogue-gjg.4` — compaction_forgot + compaction_reground surfaces; re-grounding packs survive the next compaction

P3 task · open · parent: polylogue-gjg
Prerequisites: polylogue-d1y, polylogue-gjg.3, polylogue-4ts.5.
Intent: CLI (compactions list/read, compaction forgot --top N, compaction inject --budget) + MCP tools compaction_forgot (ranked loss items WITH stable anchors — agents need refs, not prose) and compaction_reground (bounded token-budget context pack of top lost-but-later-referenced items).
Execution shape: MCP registration traps apply (EXPECTED_TOOL_NAMES + contract + regen).
Done when: forgot returns ranked items with anchors on a real compacted session; reground writes the handoff assertion; the assertion is injected only under the flag and its content survives a subsequent compaction.

### 263. `polylogue-gjg` — Compaction lifecycle: pre-compaction snapshot, loss forensics, post-compaction re-grounding

P2 epic · open
Prerequisites: polylogue-d1y, polylogue-4ts.5.
Intent: Compaction is where the OS-like context-management vision meets the harness's own memory management, and today Polylogue only observes its AFTERMATH (acompact lineage edges, v12).
Execution shape: (1) SNAPSHOT: wire the PreCompact hook (VERIFY availability/payload in current Claude Code — the hook catalog moves; Codex equivalent via app-server events ox0) to capture the full context state as an artifact (session-linked, blob-stored, content-addressed — dedup makes repeated compactions cheap).
Done when: PreCompact snapshots land for real compactions on the operator machine (or the JSONL-boundary fallback is implemented and labeled); the loss measure runs corpus-wide with tier=structural and renders an epidemiology table; re-grounding injects only under the flag and its arm comparison is defined as an ExperimentSpec; 37t epic description carries the handoff-triad map.

### 264. `polylogue-h10` — Prediction and calibration tracking: agents scored on what they said would happen

P3 task · open · parent: polylogue-9l5
Prerequisites: polylogue-h6r, polylogue-9l5.7.
Intent: Gwern/PredictionBook-lineage capability the archive is uniquely positioned for: sessions are full of predictions — explicit ('this should fix the test') and annotatable (::predict markers via the 37t.2 protocol) — and the archive knows OUTCOMES structurally (test passed, PR merged, command retried).
Execution shape: Two lanes, one ledger: (1) DECLARED via 37t.2: ::predict(p, horizon, resolver) with stated resolution criterion — highest validity, needs protocol adoption.
Done when: Ledger + measure registered; implicit lane produces curves from existing followup_class pairs with stated sample frames; declared lane round-trips one ::predict to a resolved row; reliability diagram renders in analyze and web.

### 265. `polylogue-jnj.10` — Make the completion system and DSL discoverable at point of use

P3 task · open · parent: polylogue-jnj
Intent: The completion system is a secret (polylogue config completions is buried; no install nudge) and DSL learnability needs a reference card, not just a tutorial: a one-screen `polylogue syntax` card generated from the grammar registries (fields, units, stages, operators, views — same source as completions, so it cannot drift), an install hint on bare invocation, and help epilogs pointing at it.
Execution shape: Three exposure channels for completions (ranked source: fables CLI audit): (a) polylogue init offers to install completions; (b) ship system-wide via the Nix/HM module (nix/hm-module.nix) and the Homebrew tap template — both distribution channels already exist; (c) one-time stderr hint when running interactively without completions installed.

### 266. `polylogue-jnj.11` — Extend the fzf pattern from select to ambiguous-result moments

P3 task · open · parent: polylogue-jnj
Intent: select has fzf; the ambiguous moments do not: multi-hit read REF, ambiguous session refs, multi-candidate resume.
Execution shape: One shared helper already exists: select shells out to fzf with tty detection and graceful fallback (cli/select.py:100-147).

### 267. `polylogue-jnj.12` — Empty-result guidance: 0 hits explains itself

P3 task · open · parent: polylogue-jnj
Intent: Today 0 hits is a dead end; the facet machinery + diagnose_query_miss can say why: which predicate zeroed the set, nearest non-empty relaxation, did FTS vs structured disagree, origin coverage note.
Execution shape: Algorithm (cheap, uses existing facet machinery): on 0 hits, re-run the compiled spec with each clause dropped one at a time and report counts — '0 results — without since:7d there are 42; without origin:codex there are 17'.

### 268. `polylogue-jnj.13` — Bare-invocation triage: status + five most recent sessions with one-key open

P3 task · open · parent: polylogue-jnj
Intent: No-arg polylogue on a tty currently shows status/stats — reasonable, but for a reader product the more inviting default is status PLUS the five most recent sessions with one-key open.
Execution shape: Bare 'polylogue' currently prints help via the strict command floor (cli/query_group.py _bare_root_error_message handles bare WORDS; bare NO-ARGS shows Click help).
Done when: Bare invocation renders triage in under 200ms on the live archive; falls back to help text when no archive exists; strict-floor bare-word behavior unchanged (polylogue foo still UsageError).

### 269. `polylogue-jnj.14` — Bare single-token query-first dispatch: prefer subcommand-typo error over silent search

P3 task · open · parent: polylogue-jnj
Intent: click_app.py routes positional args without a subcommand prefix to query mode (docstring line 4).
Done when: A bare single token that is a close subcommand match errors fast with a did-you-mean hint instead of running a full FTS search; multi-word bare input still searches; find and quoted single tokens still search.

### 270. `polylogue-jnj.2` — analyze boolean modes -> named projections; facets becomes a real verb

P3 task · open · parent: polylogue-jnj
Intent: analyze multiplexes count/facets/cost-outlook/postmortem/portfolio/grouped-stats/diagnostics behind mutually-exclusive booleans, and analyze --facets owns the full query-filter stack while top-level facets is narrower.
Execution shape: Order: (1) extend the top-level facets command (or a facets projection) to consume the full RootModeRequest filter stack — the gap is that analyze --facets owns the full query-filter surface while top-level facets is narrower (audit-confirmed); (2) move analyze's boolean modes (count/cost-outlook/postmortem/portfolio/grouped-stats/diagnostics) into named projections/subcommands sharing the query relation + render contracts; (3) delete the boolea…

### 271. `polylogue-jnj.3` — Output dialect normalization (--format/--json + --to/--out)

P3 task · open · parent: polylogue-jnj
Intent: Mixed --output/--output-format/--format/--to/--out and plain/text/plaintext families; standardize, remove aliases not buying value.
Execution shape: Anchors: polylogue/cli/query_output.py + query_output_contracts.py (query-side rendering), polylogue/cli/shared/formatting.py (plain/tty detection), scattered --json flags on verbs.
Done when: Every read verb accepts --format with identical semantics; --json is an alias; format x destination are independent; output-contract schemas regenerated (devtools render cli-output-schemas).

### 272. `polylogue-jnj.4` — Direct `read session:REF` uses read-view semantics

P3 task · open · parent: polylogue-jnj
Intent: Positional read session:...
Execution shape: Anchor: polylogue/cli/read_view_handlers.py (read --view semantics) vs the direct 'read session:REF' path in cli/query_group.py — the direct path bypasses read-view profile resolution, so the same ref renders differently depending on invocation shape.
Done when: Direct ref read and query-then-read produce byte-identical output for the same session and view; read-view profiles apply on both paths.

### 273. `polylogue-jnj.6` — Demo/import surface separation (import --demo -> polylogue demo)

P3 task · open · parent: polylogue-jnj
Intent: import --demo --wait --with-overlays mixes demo seeding, convergence wait, daemon scheduling; move demo convergence under polylogue demo.
Execution shape: Rule: 'import' ingests real user data; 'demo' owns synthetic/showcase flows entirely (polylogue demo seed/verify already exist).
Done when: import has no --demo flag; demo seed covers the flow; docs + output schemas regenerated; no other surface (MCP/daemon) references import-demo.

### 274. `polylogue-jnj.8` — Rationalize root onboarding, tutorial, and reader launcher

P3 task · open · parent: polylogue-jnj
Intent: Root CLI/onboarding/reader-launcher coherence; tutorial currently prints five status lines on a configured install.
Execution shape: Three surfaces to converge: root bare invocation (jnj.13 owns the triage screen), polylogue init/tutorial (currently prints five status lines on an already-configured install — make it state-aware: configured -> point at triage/cookbook; fresh -> guided init), and the reader launcher (web reader open command).

### 275. `polylogue-mhx.6` — Embedding storage/spend efficiency: quantization, matryoshka, and scoped drain

P3 task · open · parent: polylogue-mhx
Prerequisites: polylogue-mhx.3.
Intent: Two cost surfaces: storage (float32 1024-dim vectors; sqlite-vec supports int8 and bit quantization plus matryoshka prefix slicing) and spend surprises — live evidence 2026-07-03: an API-only inspection daemon run (--no-watch --no-source-catchup) silently drained pending embeddings and spent real Voyage dollars within minutes, because embedding_enabled=true makes ANY polylogued run a paid catch-up worker.
Execution shape: (1) DRAIN SCOPING: ambient catch-up runs only in the full daemon role — inspection/partial runs (--no-watch or any component-disabled run) default drain OFF with a --embed-catchup opt-in; startup prints the pending count + projected spend when drain is armed; each drain window writes its actual spend to embedding_catchup_runs (verify: may already record) and ops status shows cumulative spend this month vs embedding_max_cost_usd.

### 276. `polylogue-pf1` — Sync/async divergence: diff the twin backends against the '10 known divergences' list

P3 task · open · parent: polylogue-a7xr
Intent: The async backend self-documents ~10 known divergences from the sync path; nothing enforces the list stays complete.
Execution shape: Twins: async lane storage/sqlite/async_sqlite*.py vs sync lane storage/sqlite/archive_tiers/.
Done when: A committed twin-diff artifact classifies every write-path method; zero unexplained divergences (each is fixed or has an explicit rationale row); a test regenerates the diff and fails on new divergence.

### 277. `polylogue-hiu` — Collapse storage twins onto the sync core behind an async adapter boundary

P2 task · open
Prerequisites: polylogue-exb, polylogue-pf1.
Intent: DECIDED (delegated by operator 2026-07-03): direction B — sync core, async adapter.
Execution shape: Execution plan, each step independently shippable and mypy-strict-netted: (0) PREREQ pf1: reconcile the 10 documented divergences INTO the sync store — for each, the divergence diff decides which twin's behavior is canonical; the sync store becomes the single source of behavior BEFORE any wiring moves.
Done when: Per migrated mixin: bench ingest-throughput within noise of baseline and interactive read SLOs hold.

### 278. `polylogue-rii.3` — Ingest fidelity: parser fingerprints, byte-fidelity bands, unparsed-key census, round-trip bar

P3 task · open · parent: polylogue-rii
Intent: Import correctness must be visible, not inferred from ingest success: source-tier parser_fingerprint + decode_failure_class (durable, batch with source v3 window); derived raw_fidelity records with per-origin byte-fidelity ratio BANDS (ratio is diagnostic — the real bar is STRUCTURAL round-trip reconstruction equality), unparsed-key census (ranked ignored provider keys), misclassification tripwire (run other detectors post-parse), zero-message parse-success anomaly detector, and parser-fingerpr…
Execution shape: Split by tier regime: parser_fingerprint + decode_failure_class are DURABLE source-tier columns -> numbered source v3 migration, batched in the 60i5 window; raw_fidelity records (byte-ratio bands, unparsed-key census, round-trip equality) are DERIVED -> index-tier rebuild regime.
Done when: Unknown-key fixture reports census; fingerprint change enqueues reprocess; round-trip structural equality asserted for >=2 origins; ratio bands per-origin not absolute.

### 279. `polylogue-rvh` — Lesson reinforcement scheduling: judged memory on a forgetting curve

P3 task · open · parent: polylogue-37t
Prerequisites: polylogue-mhx.4.
Intent: Judged lessons inject by relevance and recency, but there is no consolidation model: a lesson injected once gets crowded out; always-inject burns budget on the internalized.
Execution shape: (1) Per-assertion scheduling state (user.db: last_injected, interval, ease) updated by the compiler on injection; SM-2-lite (no FSRS needed at N<1000).
Done when: Scheduling state updates on injection; a signatured lesson demonstrates interval reset on recurrence in a seeded scenario; due-ness visibly affects preamble composition under budget; Anki export produces a valid deck.

### 280. `polylogue-rxdo.5` — StandingQueryStage: watched queries re-evaluated on convergence; deltas become candidate findings

P3 task · open · parent: polylogue-rxdo
Prerequisites: polylogue-37t.15, polylogue-rxdo.2, polylogue-rxdo.4.
Intent: Fourth default convergence stage (after fts/embed/insights) using the existing session-scoped stage hooks (check_sessions/execute_sessions + false_means_pending).
Execution shape: Fits DaemonConverger without engine changes (ConvergenceStage already has session-scoped variants — verified in daemon/convergence.py).
Done when: Watch a query, ingest a matching session, observe exactly one query-delta candidate; re-ingest same content => zero new candidates; reset --index then converge => zero false drift; a promoted expected-count finding diverging emits one drift candidate targeting it.

### 281. `polylogue-rxdo.6` — DSL reference operands: from query:/result-set:/cohort: as provenance-preserving AST nodes

P3 task · open · parent: polylogue-rxdo
Prerequisites: polylogue-rxdo.2, polylogue-fnm.13.
Intent: Let saved analysis objects compose: from query-definition re-evaluates, from query-run uses its retained relation (typed error if not retained), from result-set uses the stored relation, from cohort re-evaluates (dynamic) or yields members (snapshot); all usable as set-algebra operands.
Execution shape: LALR pitfall (bd memory): any new terminal containing a colon must slot above FIELD_CLAUSE.4 priority or it is eaten as a field clause.
Done when: from result-set:<id> | group by model | count works and explain shows the ref lineage; grain mismatch errors with a suggestion; @macro semantics unchanged; cycles rejected.

### 282. `polylogue-rxdo.8` — Analysis recipes as DB-native runtime objects; YAML as import/export serialization only

P3 task · open · parent: polylogue-rxdo
Prerequisites: polylogue-rxdo.3, polylogue-rxdo.7.
Intent: Corpus-reviewed decision (defended against two runner-ups): recipes/runs must be DB objects because complex analyses are interactive DAGs, not static phase lists — the durable truth is what actually ran (which queries, which batches, which model, what got superseded), which YAML cannot record and assertions must not become (assertions are claims; recipes are procedure; runs are execution state — the user.py settings-vs-assertions comment already encodes this distinction).
Execution shape: user.db tables (batch with v5): analysis_recipes (definition_json, source_artifact_ref, version), analysis_runs (recipe ref, status, actor, archive_epoch, query_run_refs, annotation_batch_refs, artifact_refs, degraded).
Done when: recipe import -> run -> the run record cites its query runs and batches; re-run against a later epoch produces a diffable second run; YAML round-trips.

### 283. `polylogue-rxdo` — Analysis provenance: queries, result-sets, findings, analyses as first-class objects

P2 epic · open
Intent: THE convergent frontier from the 2026-07-05 R&D program (hit independently by swarm waves 2/4 and multiple GPT-Pro review branches).
Done when: A committed query returns stable query/query-run/result-set refs on every surface; assertions can target them; reset --index cannot destroy promoted/cited result sets; findings live in the existing candidate->judge lifecycle; the twelve recursive-loop failure modes recorded in child beads have guards.

### 284. `polylogue-t46.1` — Replace showcase QA with demo-driven CLI and visual tests

P3 task · open · parent: polylogue-t46
Intent: Keep demo, synthetic data, real CLI subprocess checks, visual tests; remove the miniature QA bureaucracy between those behaviors and ordinary tests.
Execution shape: Inventory what 'showcase QA' still is (rg showcase + the demo/QA command surfaces; the old qa CLI became demo per the stale-notes memory): keep = polylogue demo seed/verify (synthetic, private-data-free), real CLI subprocess checks (tests/ integration-style), visual tests (visual-tapes recordings per 3tl.5).
Done when: Zero bespoke QA-harness code remains; the removal PR contains the check->replacement mapping table; demo verify + devtools test cover every behavior the old layer claimed.

### 285. `polylogue-t46.7` — Move compose_context_preamble git enrichment into context/preamble.py

P3 task · open · parent: polylogue-t46
Intent: mcp/server_context_tools.py:127-159 shells out to git rev-parse / git log --oneline -5 in the surface to fill ContextPreambleProjectState, then re-validates, because the shared build_context_preamble_payload does not populate project state.
Done when: No git subprocess in server_context_tools.py; build_context_preamble_payload (or the preamble module) fills project state; MCP and a CLI/hook caller produce identical ContextPreambleProjectState for the same repo; devtools verify green.

### 286. `polylogue-t46` — Contracts own surfaces: delete parallel dispatch and the QA middle layer

P2 epic · open
Intent: Make existing contracts (query DSL, terminal units, refs, read-view profiles, action/route contracts, generated docs/schemas) the actual owners of behavior; delete hand-written parallel surfaces.
Execution shape: First slices (read-only audit 2026-07-02, Herschel): config aliases (daemon_host/daemon_port, top-level observability), hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle), status JSON compatibility aliases, origin->provider projection bridges in outputs, browser-capture old synthetic-ID recovery.
Done when: - For each listed first slice — config aliases (daemon_host/daemon_port, top-level observability); hidden/help + historical CLI aliases (find_help, -n, --full, demo-shelf --bundle); status JSON compatibility aliases; origin->provider projection bridges in outputs; browser-capture old synthetic-ID recovery — the parallel hand-written surface is DELETED in the same PR that makes the contract (DSL/registry/generated schema) the sole owner (grep con…

### 287. `polylogue-ttu` — Docs information architecture: tiered index, orphan sweep, stale-doc triage

P3 task · open · parent: polylogue-3tl
Intent: docs/ has 54 entries with 43 top-level markdown files and the README index links ~26 of them — a flat namespace where getting-started sits alphabetically between generate.md and glossary.md, and an unknown number of pages are orphaned (unreachable from any index) or stale (describing removed surfaces — the repo has archived whole CLI eras).
Execution shape: (1) Tier the tree: guide/ (stranger-facing: getting-started, installation, configuration, search, hooks), reference/ (generated: cli-reference, devtools, schemas, openapi, glossary), internals/ (architecture, internals, data-model, threat models, WAL/blob details), decisions+plans (spine, execution-plan, plans/*), retro/.
Done when: Every docs/*.md is reachable from the generated index or lives under docs/archive/ with a banner; render all --check fails on an unindexed new doc; the pages site nav reflects tiers; zero dead intra-doc links (link check already exists in 6bu — run it green).

### 288. `polylogue-utf` — Devtools surface economy: usage-ranked consolidation of the 67-command catalog

P3 task · open
Intent: devtools has 67 CommandSpecs across 7 groups with three-level nesting (lab probe X, bench mutation Y, workspace Z) and real overlap: verify has 12 subcommands including three lint-shaped ones; render has 13 where 'render all' is the only one anyone needs to remember; lab/bench/workspace accumulate one-off analysis commands that are demos wearing command names (claim-vs-evidence, cli-surface-audit, temporal-*).
Execution shape: Evidence-first, same method as 9e5.2 but scoped to devtools: (1) rank all 67 by real invocation count from devtools workspace tasks history + shell history (bd memories note the harness records runs); (2) classify: core loop (status/verify/test/render all) / registered-lane (lanes, campaigns — keep, they are registries) / one-off analyses that produced their artifact and are done (retire to git history or fold as a lane entry) / thin wrappers be…

### 289. `polylogue-y4c` — Configuration doctrine: great defaults, DB-backed runtime prefs, Nix module parity

P3 task · open · parent: polylogue-w8db
Intent: Config today is one polylogue.toml + env vars + a 1,660-line config.py, with no stated doctrine for what DESERVES a knob (the operator: no pointless configurability, great defaults) and no separation between deployment config (paths, ports, keys — file/env territory) and runtime preferences (default read view, injection budgets, saved defaults per verb, UI prefs — which belong in user.db where surfaces can read/write them live and they survive as part of the durable user tier, per the operator'…
Execution shape: (1) DOCTRINE (one page in internals): a knob must have >=2 legitimate values in real use, a stated default right for 90%, and an owner surface; anything else is code.
Done when: Doctrine page committed; key audit merged with the deletion list; user.db-backed prefs work end-to-end for at least default-read-view and injection-budget with live effect; 'config effective' names source per key; Nix module options are generated from the registry and render-checked; configuration.md generated.

### 290. `polylogue-6kh` — Query-scope preferences bundle: default time window, scope filters, logical fold

P2 feature · open · parent: polylogue-w8db
Prerequisites: polylogue-fnm.12, polylogue-y4c.
Intent: The highest-value query prefs: default trailing time window for bare queries (speed + relevance; all: to widen), default scope filters (exclude temporary sessions, subagent-physical rows), logical-session fold in list outputs, default sort and limit.
Execution shape: Implemented as an implicit macro layer: the resolved prefs compile to a predicate group prepended to bare queries (the fnm.12 expansion machinery — visible in explain as 'implicit-scope', overridable inline with all:/include:).
Done when: Bare find on the live archive runs windowed with the footer disclosure; all: widens; explain shows implicit-scope expansion; temporary/subagent exclusion + logical fold toggle work; defaults changeable at runtime with live effect.

### 291. `polylogue-ze5` — Decision: user.db vocabulary — separate epistemic records from workspace state

P3 task · open · parent: polylogue-37t
Intent: Operator question (2026-07-03): is 'assertions' the best term, and is the concept sufficient?
Execution shape: Recommendation (decide, then execute incrementally — no big-bang rename): (1) VOCABULARY: keep the storage table name (churn without benefit); introduce a typed CLASS field/derivation over kinds — epistemic | curation | workspace | comms — and use class-appropriate nouns in every surface: 'records' or 'notes/claims' for epistemic (external docs already translate to 'judged notes/memory' per 3tl.1), 'tags/marks', 'saved views', 'messages'.
Done when: Class taxonomy recorded in the vocabulary doctrine + glossary and every kind mapped; relations + revisions land as additive user.db migrations with the judge queue surfacing contradictions; surface nouns audited (no user-facing 'assertion' for non-epistemic classes); confidence field adopted by h10.

### 292. `polylogue-1a9` — Remove dead session-commit stubs + unused web-construct row + stale fuzz README

P3 chore · open · parent: polylogue-a7xr
Intent: Single surgical-renewal PR; targets enumerated on the issue.
Execution shape: Targets (gh#2477, code-confirmed): insights/session_commit.py persist_session_commits is a no-op ('del edges, repo_id') and session_commit_edge_to_row has no callers — delete both; storage/sqlite/archive_tiers/write.py ArchiveWebConstructRow is never instantiated (_write_web_constructs inserts inline) — delete the dataclass; tests/fuzz/README.md references polylogue.lib.timestamps (now polylogue.core.timestamps) — fix the doc.

### 293. `polylogue-48h` — Consolidate SQLite introspection helpers (10 copies of _table_exists and friends)

P3 chore · open · parent: polylogue-a7xr
Intent: Ten separate implementations of _table_exists exist across cli/commands/tutorial.py, cli/read_views/streaming_markdown.py, cli/commands/status.py, insights/readiness.py (async), operations/archive_debt.py, sources/live/hook_paste_enrichment.py, sources/live/convergence_debt_retry.py, storage/source_sessions.py, and storage/session_replacement.py (sync+async pair) — grown from the six the original grok counted.
Execution shape: One module: storage/sqlite/introspection.py with table_exists(conn, name, schema='main'), index_exists, column_exists — sync versions + thin async wrappers (or a single implementation taking both connection protocols; decide by reading the aiosqlite call sites).

### 294. `polylogue-5dx` — Dependency leverage policy: [analytics]/[ml] extras, evaluated adoptions

P3 chore · open
Intent: The analytics tower and adjacent programs want libraries; core must stay lean (pure-Python, fast cold start — 20d.2 is fighting import weight already).
Execution shape: (1) Policy in CONTRIBUTING/internals: core deps must survive 'needed on every invocation' scrutiny; analytics/ml capabilities import lazily behind extras with actionable ImportError ('pip install polylogue[analytics] for CIs and changepoints'); hand-rolled fallbacks only for primitives used in core paths (Wilson interval yes, PELT no).
Done when: Extras defined in pyproject with lazy-import seams; core test suite passes with no extras installed; policy table committed; 20d.2 receives the measured import-cost ranking of current core deps.

### 295. `polylogue-6bu` — Docs-site verification lane (pages cache, link integrity)

P3 chore · open · parent: polylogue-3tl
Intent: DEMO-RADAR 07-01: generated pages cache blocked integration pushes and produced false failures; PR #2500 fixed link generation.
Execution shape: Docs-site failure modes seen live: pre-push render all --check rebuilds gitignored .cache/site so a stale cache breaks push (run render pages first — recorded gotcha), and PR #2500 repaired link rot.
Done when: Broken internal link fails the lane with the offending page:anchor named; stale-cache false-failures documented with the one-command fix; lane runs in verify --quick or render all --check.

### 296. `polylogue-6l6` — Docs/theming/release-proof/control-plane polish

P3 chore · open · parent: polylogue-3tl
Intent: Externally inspectable + internally dogfoodable polish set.
Execution shape: Grab-bag polish set — split on claim if any slice grows: (a) docs theming pass (ui/theme.py tokens applied to docs site), (b) release-proof check (3tl.7 install matrix is the heavy half; this is the docs claim-consistency half — versions/commands in docs match pyproject), (c) control-plane doc currency (docs/devtools.md vs command_catalog.py drift — the doc-commands lint exists, extend to prose).
Done when: Each slice either done or split to its own bead; docs claims about version/commands verified against live surfaces (render checks green).

### 297. `polylogue-a7xr.12` — neighbor_candidates needs a 4-method protocol, not the 20-method SessionQueryRuntimeStore

P3 chore · open · parent: polylogue-a7xr
Prerequisites: polylogue-a7xr.11.
Intent: VERIFIED counts: archive/session/neighbor_candidates.py calls exactly 4 store methods (resolve_id, get, list_summaries_by_query, search_summary_hits) but is typed against the ~20-method SessionQueryRuntimeStore — forcing api/archive.py:1341-1747 _ArchiveNeighborRuntime to stub ~15 unneeded methods (~400 lines), re-implement the 18-kwarg trio a FOURTH time, and still need a cast at :4216 because it does not truly conform.
Execution shape: Define a 4-method NeighborStore protocol next to neighbor_candidates.py; retype the consumer; _ArchiveNeighborRuntime shrinks to ~60 lines; the cast disappears.
Done when: Adapter under 80 lines; no cast; neighbor_candidates behavior unchanged (existing tests); mypy strict green.

### 298. `polylogue-a7xr.14` — Collapse the one-operation operations-contract framework to concrete Import models

P3 chore · open · parent: polylogue-a7xr
Intent: Abstraction audit: operations/operation_contract.py (277 LOC of OperationRequest/Ack/FollowUp/Status generics) serves exactly ONE operation — OperationKind.IMPORT (9 prod uses); the other 9 of 10 enum members have zero prod uses and nothing dispatches on .kind (rendered once as a markdown label).
Execution shape: Collapse to concrete ImportRequest/ImportAck; reintroduce a base when a SECOND operation actually lands (rxdo query-runs or fs1.5 export may become that — check before deleting whether either is imminent; if yes, keep the base and delete only the unused enum members/statuses).
Done when: Wire envelope byte-identical (golden); one concrete model pair; unused enum members gone or each carries a consumer; devtools verify green.

### 299. `polylogue-a7xr.15` — payloads.py: generic from_row for the 74 identical-name copy lines (keeps typed wire contract)

P3 chore · open · parent: polylogue-a7xr
Intent: Abstraction audit (rg-verified): 30 hand-rolled from_row/from_* classmethods in surfaces/payloads.py where 74 of 74 x=row.y copy lines are identical-name — pure mechanical transcription across 2,921 LOC / 85 classes.
Execution shape: One generic from_row on the shared base (cls(**{f: getattr(row, f) for f in cls.model_fields if hasattr(row, f)})) with explicit overrides only where renames/defaults exist (title=row.session_title, material_origin='unknown' at :1275).
Done when: Identical wire output (goldens across list/search/read payloads); per-class field-parity test in place; ~400-500 LOC removed; render openapi/cli-output-schemas unchanged.

### 300. `polylogue-a7xr.9` — Mechanical helper dedup sweep: scalar coercion quadruplet, _table_exists x40, provenance vocab x6, title/tags mixin

P3 chore · open · parent: polylogue-a7xr
Intent: Bundle of zero-risk verbatim-copy consolidations from the divergence audit: (a) daemon scalar-coercion helpers (_required_str/_optional_str/_row_int/_row_float) copied across 5+ status modules with ALREADY-diverged signatures (row_float -> float vs float|None) while core/payload_coercion.py is the designated home; (b) _table_exists defined 40x (41 with the schema variant) — the codebase's single most duplicated function; (c) _range_timing_provenance/_date_provenance emitting the timestamped_ran…
Execution shape: (a) add row_int/row_float/required_str raising variants to core/payload_coercion.py, sweep daemon modules; (b) table_exists(conn, name, *, schema='main') + async twin in storage/sqlite/, mechanical sweep; (c) define once in archive/session/provenance.py with object|None signature, five deletions; (d) shared mixin for the title/tags precedence rules.
Done when: rg counts: one definition each for the swept helpers; mypy --strict green; no behavior goldens change.

### 301. `polylogue-bby.6` — Interaction debt: replace window.prompt(); de-drift the JS renderer

P3 chore · open · parent: polylogue-bby
Intent: window.prompt() for workspace/recall-pack naming; hand-rolled JS message rendering duplicating the Python renderers (drift risk — the renderer contract should be shared or snapshot-tested against the canonical renderer output).
Execution shape: Item list (fables web audit 5-6): (1) window.prompt() for workspace/recall-pack naming (web_shell_workspace.py:398,503) -> app-native modal; (2) saved queries (saved_query assertions) get a sidebar presence — doubles as the DSL example library in the UI; (3) canonical-URL projection back to provider UIs (chatgpt.com/c/..., claude.ai/chat/...) already exists in the model as a computed field — the reader should link out; (4) dark-only hardcoded pa…

### 302. `polylogue-jnj.7` — Provider-token leakage cleanup in public CLI help

P3 chore · open · parent: polylogue-jnj
Intent: Public filters use origin vocabulary; some analysis/maintenance help still advertises provider tokens.
Execution shape: The provider->origin retirement's CLI-help slice: rg for provider-vocabulary tokens in user-visible help/error strings (cli/ commands, click option help=, UsageError text).
Done when: No provider-family token appears in polylogue --help output tree or UsageError messages where an origin token is meant; docs/cli-reference.md regenerated.

### 303. `polylogue-jsy` — Harden blob hash validation + drop misleading symlink check

P3 chore · open · parent: polylogue-kwsb
Intent: Defense-in-depth (none currently exploitable): fullmatch 64-hex in blob_path/cleanup_orphans; drop the CWD symlink loop in sanitize_path; consider a zip aggregate budget.
Execution shape: Exact changes (gh#2483): (1) blob_store._VALID_HEX = re.compile(r'^[0-9a-f]+$') accepts trailing newline and any length -> use re.fullmatch(r'[0-9a-f]{64}', h) in blob_path + cleanup_orphans; (2) core/security.sanitize_path runs Path(v).is_symlink() per parsed attachment against the process CWD — meaningless since the path is never opened -> drop the symlink loop, keep traversal/control-char stripping; (3) decoder_zip has a per-entry 10GiB cap b…

### 304. `polylogue-kwsb` — Security & privacy: the archive can forget on purpose and never leaks secrets

P2 epic · open
Intent: WHY: a personal archive of ALL AI work is the most sensitive database on the machine — it must be able to forget on purpose (excision that provably removes bytes, not just rows) and must never leak (localhost daemon reachable from a hostile page, secrets in captured content).
Execution shape: No epic owned the security/privacy surface: excision (27m), reset-mutation safety (jnj.5, real bug at reset.py:260-277 where --session/--source tombstone before the preview:327 and --yes gate:331), and crawl-source permissions (jsy) were orphaned.
Done when: Excision (right-to-forget + secret redaction + blob excision) is execution-grade and shares one mutation-audit/dry-run/--yes contract with reset (jnj.5); the security-privacy-coverage.yaml gaps each have an owning bead or test; the MCP write/admin destructive path shares the same audit-row contract.

### 305. `polylogue-0aj` — Declared write-effects chain: post-commit effects as registry entries

P3 feature · open · parent: polylogue-a7xr
Intent: archive/write_effects.py:commit_archive_write_effects is the one good choke point in the write path — FTS repair, blob leases, cache invalidation already flow through it.
Execution shape: (1) Formalize what is already true: a WriteEffect protocol (name, phase: in-transaction | post-commit | async-deferred, ordering constraint, failure policy: abort | log-and-continue) and an ordered registry; commit_archive_write_effects walks the registry instead of inlining.

### 306. `polylogue-5wp` — Insights as declared derived views: dissolve the stage, materialize by policy

P2 task · open
Prerequisites: polylogue-0aj.
Intent: Session insights currently run as a dedicated convergence stage that materializes everything (profiles, work events, phases, threads, run projection) whether or not anything reads it — the run-projection rows already proved partially redundant (x5l source-derived CTEs; dab drops the cache rows).
Execution shape: (1) InsightSpec in the registry (o21): compute (SQL lowering and/or Python), policy (query-time | materialized | hybrid), staleness key, consumers — replaces insights/registry.py's implicit list.
Done when: InsightSpec registry with every current insight classified; at least one table demoted to query-time and session_stats added with list surfaces reading it; the standalone insights stage gone from convergence; rebuild produces identical read results (snapshot parity).

### 307. `polylogue-1jc` — Learned defaults: the archive proposes your configuration as judged candidates

P3 feature · open · parent: polylogue-w8db
Prerequisites: polylogue-20d.14, polylogue-y4c.
Intent: The archive records every polylogue invocation (its own dogfood telemetry + affordance usage), which means it can OBSERVE preference: the operator adds --view dialogue to 80% of codex reads; always re-sorts by recency; never opens temporary sessions from lists; always bumps --max-tokens on read.
Execution shape: (1) SIGNAL: invocation spans (20d.14 CLI telemetry) + affordance usage rows give (verb, flags, scope, count) aggregates; a detector runs as a low-frequency insight pass over trailing 30d with minimum support (n>=20) and dominance (>=70%) thresholds — both themselves y4c prefs.
Done when: Detector produces a correct suggestion from seeded telemetry (dominant flag pattern -> candidate with evidence aggregate); accepting in judge writes the scoped settings row and the new default takes effect; rejecting suppresses re-proposal; suggestions capped; deployment keys never proposed.

### 308. `polylogue-1xc.10` — Design spike: express session insights + aggregates as declared derived views over a single refresh engine

P3 feature · open · parent: polylogue-1xc
Intent: Longer-horizon refactor the operator gestured at ('insights as declared derived views').
Done when: 1) An ADR under docs/ (or thoughtspace) that inventories every current insight table, classifies per-session vs cross-session scope and its affected-scope function, and proposes (or explicitly rejects) a declared-derived-view registry with a single refresh engine.

### 309. `polylogue-1xc` — Scale-hardening: bugs that only bite on real-scale archives

P1 epic · open
Intent: Confirmed-severe set of code correct on small/clean fixtures but wrong at real scale (e.g.
Execution shape: Tier-1 confirmed-live items (gh#2465 checklist is authoritative; work it there): full insight rebuild runs as ONE transaction -> 6GB WAL + minutes-long write lock on the live archive — chunk the rebuild into bounded per-batch transactions with progress rows (storage/insights rebuild path); the run_ref global-PK collision class was fixed (#2464) — audit for siblings (any global PK derived from non-unique local coordinates).
Done when: Epic terminal state: every child closed and a scale-regression lane exists (seeded large-archive tier or live-copy probe) that would have caught each shipped bug class, wired into the optional lanes.

### 310. `polylogue-2n6` — Harness remote-control lane: drive Claude Code / Codex sessions from Polylogue surfaces

P3 feature · open
Prerequisites: polylogue-37t.8.
Intent: Operator direction 2026-07-03: analogous to web-chat posting, coding-agent sessions should eventually be drivable — Claude Code has remote-control affordances (claude.ai/code session URLs / SendMessage-style continuation), and Codex needs an analogous lane (possible without native remote control via terminal injection, but fiddlier).
Execution shape: Investigate-then-build, per-harness: (1) Claude Code — enumerate actual remote-control surfaces available on the current install (session URLs, CLI resume flags, any local IPC) before designing; prefer native resume (claude --resume <session-id>) via kitty terminal control (sinnix-kitty-control) as the floor that already works.

### 311. `polylogue-30h` — Display titles: synthesize when the stored title is a first-prompt echo

P3 feature · open
Intent: Session lists (web reader, CLI rows) render stored titles that are just the truncated first user prompt: 'do not launch any more subagents than these.
Execution shape: Structural synthesis, no LLM calls, no prose mining for facts: a display_title derivation that fires only when the stored title is a prompt-echo (title == prefix of first user message) — compose from what the archive knows structurally: repo/cwd basename + workflow shape + dominant tool family + first distinctive user line (skip boilerplate preambles by detecting repeated cross-session prefixes — the 'You are exploring...' template appears verba…

### 312. `polylogue-37t.10` — Setup evolution via judged candidates: hooks/context-specs/cookbook changes proposed as evidence-linked assertions

P3 feature · open · parent: polylogue-37t
Prerequisites: polylogue-37t.12.
Intent: The operator wants agents maintaining and self-improving the Polylogue setup itself — hook scripts, injection policies, cookbook recipes, MCP tool contracts.
Execution shape: Narrow write path, not a framework: (1) a setup_improvement candidate assertion kind (or reuse NOTE with a scope_ref to the config artifact — decide against the every-kind-has-a-surface test cost; registration-traps memory applies if a new kind is added).

### 313. `polylogue-37t.18` — Second-brain entity graph: structural-vs-candidate mention split, backlinks, topic co-occurrence

P3 feature · open · parent: polylogue-37t
Intent: Navigable knowledge graph over the archive: entities/entity_mentions/entity_topics + an entity_backlinks VIEW.
Execution shape: Storage (derived, index-tier — rebuild regime): entities(entity_id, kind, canonical_name), entity_mentions(entity_id, block_id, mention_kind: structural|candidate, extractor_version, confidence), entity_topics join, entity_backlinks VIEW over mentions.
Done when: Structural mentions resolve without gating; candidate mentions enter as recursive-safety candidates; backlinks VIEW works; a prose-fabrication fixture does NOT self-promote.

### 314. `polylogue-37t.19` — Semantic notification policy: route CONTENT signals through the existing fan-out, fatigue-controlled

P3 feature · open · parent: polylogue-37t
Intent: Wire-what-exists: the daemon already has a 5-backend notification fan-out carrying only OPS alerts.
Done when: A standing-query delta emits one Notice through the existing fan-out; token bucket suppresses a storm; snooze works; zero alerts on generated material; brief --since reads the ledger.

### 315. `polylogue-37t.20` — Cross-project recall(task_hint) MCP tool: most-similar prior sessions + their lessons across all repos

P3 feature · open · parent: polylogue-37t
Prerequisites: polylogue-37t.15.
Intent: The operator dream made concrete: recall(task_hint) returns the most-similar prior sessions plus their corrections/lessons/blockers across ALL repos, as a budgeted evidence pack.
Execution shape: task_hint is EXPLICIT — the calling agent states its intent (free text + optional repo/paths); no inference needed at this surface (that is the honest split vs SessionStart recall where mhx.4 forms the query from cheap context).
Done when: recall(task_hint) returns cross-repo ranked sessions with attached judged lessons and resolve-able refs; a hint about a topic with a known prior session surfaces it in top-3 on the live archive (spot fixture); candidate-vs-judged assertions visibly distinct in payload; tool registered with contract + discovery test.

### 316. `polylogue-37t.21` — Prompt/meta-workflow distillery: induce parametrized meta-prompts from high-value past sessions

P3 feature · open · parent: polylogue-37t
Prerequisites: polylogue-37t.15.
Intent: The operator stated dream — history is training data for HOW to work.
Execution shape: Pipeline: (1) COHORT: select high-value sessions by structural outcome (verify-success, low-correction, high-reuse) via the DSL — the selection query is part of each template's provenance; (2) INDUCE: an external-model pass (find|compact pack -> model -> annotation import per rxdo.7) proposes parametrized meta-prompts (params: repo, task-type, risk-tier); (3) LAND: PROMPT_TEMPLATE candidates in git-YAML (code-review lane, NOT user.db — recipes a…
Done when: Distillery produces parametrized templates from a session cohort as PROMPT_TEMPLATE candidates in git-YAML; A/B evaluator refuses a win below the evidence floor; each template cites the sessions it distilled from.

### 317. `polylogue-37t.9` — Agent self-experimentation rail: PROMPT_EVAL writer + context-spec variation + background candidate passes

P3 feature · open · parent: polylogue-37t
Intent: Raw-log-sourced: agents run methodological self-evals (same task, varied context specs via subagents — runnable today with compile_context spec variation), store judged observations as assertions (AssertionKind.PROMPT_EVAL exists with no writer, enums.py:426), and background LLM passes ('dreaming') write candidate assertions over the corpus.
Execution shape: Rail steps: (1) WRITER: an MCP/CLI surface that records a PROMPT_EVAL assertion (AssertionKind.PROMPT_EVAL exists at enums.py:426 with no writer) — payload: task ref, context-spec variant ids, outcome observations, evidence refs; lands CANDIDATE via the 37t.15 chokepoint like every agent write.

### 318. `polylogue-3gd.1` — polylogue doctor + adoption telemetry: why-zero-usage diagnosis with a relevance control

P3 feature · open · parent: polylogue-3gd
Prerequisites: polylogue-d1y, polylogue-pj8.
Intent: The substrate is worthless if agents do not use it — and the adoption signal is worthless if it false-alarms.
Done when: doctor reports liveness + zero-usage diagnosis with relevance control on a fixture matrix (used repo / configured-unused repo / irrelevant repo); adoption-rate insight computed from tool_use rows; archive-root mistake caught with actionable message.

### 319. `polylogue-3gd` — Activation layer: the agent-side setup that makes the substrate get used at all

P1 feature · open · parent: polylogue-37t
Prerequisites: polylogue-d1y, polylogue-pj8.
Intent: Operator directive (2026-07-03), verbatim intent: everything built here — assertions, annotation protocol, blackboard, judge/note verbs, MCP tools, remote control — is WASTED WORK if agents do not actually use it, and adoption is behavioral: models use what their context reminds them of.
Execution shape: (1) GLOBAL CLAUDE.md SECTION (sinnix-side, dots/claude/ — renders to all agents via render-agents): a substantial 'Polylogue substrate' chapter (~3-6K tokens of standing instruction) structured as: WHAT EXISTS (archive, memory, blackboard, protocol — one paragraph each with the mental model); WHEN-TRIGGERS (a trigger->action table: 'starting work in a repo -> read your preamble, it is evidence', 'about to re-derive how X works -> polylogue find/…
Done when: The substrate chapter exists in dots/claude and renders to Claude+Codex; trigger table, seven flows, and five protocol examples present; preamble cross-reference live; the weekly adoption report renders from affordance-usage with baseline captured BEFORE the chapter ships (so the delta is measurable); one month later the report shows material adoption movement or the chapter is revised (decay-watch is part of acceptance, not an afterthought).

### 320. `polylogue-37t` — Agent context/memory loop: declared claims -> judgment -> preamble -> reboot

P2 epic · open
Intent: The judged-memory loop: agents declare structured claims, the operator judges them, active claims compile into context preambles, and sessions reboot into compact evidence packs.
Execution shape: Epic spine: the loop closes when a declared claim travels end-to-end through all four named stages against the live archive.
Done when: - A seeded end-to-end scenario test demonstrates one claim flowing claims->judgment->preamble->reboot: an agent emits a declared marker (37t.2) that lands as a candidate assertion, the operator accepts it via the judgment queue, it appears as a ref in a compiled SessionStart preamble for the matching repo (37t.4/37t.11), and it survives a reboot-with-refs handoff (37t.3) resolvable via resolve_ref.

### 321. `polylogue-3tl.6` — Publish the normalized session model as a versioned interchange schema

P3 feature · open · parent: polylogue-3tl
Intent: There is no interchange format for agent sessions — every harness invents one, every archiver re-parses.
Execution shape: Scope: document the JSON export shape, not a new serialization.

### 322. `polylogue-3xx` — Verb-behavior and ops preferences bundle: confirmations, judge defaults, copy formats, spend, quiesce

P3 feature · open · parent: polylogue-w8db
Prerequisites: polylogue-y4c.
Intent: Remaining inventory classes: destructive-op confirmation level, judge queue default filter + batch size, copy-affordance default format, open target; ops: runtime-adjustable embedding spend budget, per-root watch debounce, alert-routing severity floor, daemon quiesce toggle, cache memory cap.
Execution shape: Same registry pattern; the ops keys route through the daemon event bus for live effect (quiesce = converger pause flag the status surfaces display; spend budget consumed by mhx.6 drain gates; severity floor consumed by the alert emitters).
Done when: Quiesce pauses ingest visibly and resumes; spend budget change takes effect without restart; confirmation level=never skips prompts in a seeded destructive flow while default prompts; judge opens with the configured filter.

### 323. `polylogue-45i` — Datasette lane: the archive as an explorable SQLite exhibit

P3 feature · open · parent: polylogue-3tl
Intent: index.db is already the artifact the SQLite/local-first crowd trusts most: a plain SQLite file.
Execution shape: (1) polylogue ops datasette (or docs-only recipe — decide by trying it): opens Datasette read-only against index.db (file:...?mode=ro; immutable mode avoids WAL contention with the daemon).

### 324. `polylogue-4822` — Curated polylogue.sdk + frozen public models: the external-consumer boundary lynchpin needs

P3 feature · open
Intent: Problem: downstream consumers (Lynchpin is the live example — raw sqlite + reimplemented models + a stale FROM conversations query) bypass the Python facade because the public boundary is broad, unversioned, and unstable — every internal module is reachable and nothing distinguishes supported surface from implementation detail.
Done when: Explicit public __all__ on polylogue.sdk + polylogue.models; stable DTO namespace with frozen models; capability/schema-version check API (consumer can ask: does this archive support X, which index version); SDK covers Lynchpin usage and Lynchpin drops its raw-sqlite path; layering lint forbids internal imports; examples import only the public namespace.

### 325. `polylogue-7fj` — Ingest beads issue history as a Polylogue evidence source

P3 feature · open · parent: polylogue-rii
Intent: Beads is Dolt-backed with full history plus interactions.jsonl and events export.
Execution shape: Beads is already a Dolt DB with full history (.beads/, bd dolt).
Done when: bd issue history for this repo ingests as sessions/events with stable native ids (idempotent re-ingest); one join query answers 'sessions that touched bead X' on the live archive.

### 326. `polylogue-4c0` — Beads-native work loop: session<->bead cross-links and archive-rendered work history

P2 task · open · parent: polylogue-rii
Prerequisites: polylogue-7fj.
Intent: Beads and the archive already observe the same work from two sides but never join: a bead's history (claims, closes, reasons) names no sessions; a session's transcript contains bd commands the archive does not structurally extract.
Execution shape: (1) EXTRACTION: bd invocations are shell tool calls with structured output — a block enricher recognizes bd claim/close/create/update and materializes session<->bead edge rows (bead id, verb, timestamp, session ref); zero heuristics, the commands are structural.
Done when: On the live archive: session<->bead edges materialize for the recent devloop sessions; the bead work-history envelope renders for a real closed bead with sessions, cost, and changes; a close-reason cross-check runs for one campaign bead and reports agreement; Stop-hook writes the session ref into bd notes on claim/close.

### 327. `polylogue-rii` — Live substrate intake: agents write work-events; evidence materializes in-loop

P2 epic · open
Intent: Invert the relationship for live agents: work lands in Polylogue as it happens (push), and the agent reads context/evidence back in-loop.
Execution shape: Invert the relationship for live agents: work lands in Polylogue as it happens (push) and the agent reads context/evidence back in-loop.
Done when: - The generic write-leg + intake seam scope is defined and split into child beads (rii.1 = the agent work-event write-leg); Hermes-specific ingestion is explicitly excluded and pointed at fs1.

### 328. `polylogue-7xv` — Native git/repo awareness: session-to-commit/branch/repo correlation in Polylogue

P3 feature · open · parent: polylogue-l4kf
Intent: Operator decision on polylogue-cuu (2026-07-03): Lynchpin is being dismantled; Polylogue owns session semantics and should potentially carry some git/repo awareness natively, with advanced cross-source correlation living in Sinex.
Execution shape: Start from evidence already in the archive, no new capture: (1) session->repo from cwd (map to repo root; sessions table already anchors cwd) and from git-command tool calls; (2) session->commit from SHAs observed in tool results (git commit/log/push output) plus Claude-session trailers in commit messages (Claude-Session: URLs and Co-Authored-By lines make the reverse link from repo history); (3) materialize as a derived read-model (insight regi…

### 329. `polylogue-83u.5` — Blob store zstd compression (36GB -> est 5-8GB)

P3 feature · open · parent: polylogue-83u
Intent: Content-addressed blobs are uncompressed JSON; zstd frames are self-identifying so no schema change is needed.
Execution shape: Address stays SHA-256 of UNCOMPRESSED bytes.

### 330. `polylogue-83u` — Attachment & blob evidence integrity: bytes exist, are honest, and stay affordable

P1 epic · open
Intent: Attachments are metadata-only by construction: 8,425 rows claim 8.4GB, 0 blobs exist, 56% zero-byte; blob_hash was synthetic until v13 made it honest-nullable with acquisition_status.
Done when: REFRAMED (operator 2026-07-04): the goal is to CAPTURE attachment bytes going forward, not miss-then-account.

### 331. `polylogue-9l5.10` — Process mining: workflow motifs, transition models, bottleneck discovery

P3 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: SEQ in the DSL queries KNOWN patterns; nothing DISCOVERS unknown ones.
Execution shape: (1) Event alphabet from structural evidence: (tool family, outcome-class) pairs from the actions view — coarse alphabet first (~20 symbols: shell-ok, shell-fail, edit, read, search, git, test-ok, test-fail, dispatch, ...); alphabet is a registry decision, documented, versioned (construct validity lives in the alphabet choice).
Done when: On the seeded corpus: DFG renders with edge counts/gaps; two-partition transition comparison outputs divergence + significance; top-10 motifs table with support; one motif converts to a runnable SEQ query.

### 332. `polylogue-9l5.11` — Predictive advisories: calibrated classical models on structural labels

P3 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: The honest ML niche beyond embeddings: supervised models where LABELS ARE STRUCTURAL (no human annotation, no LLM judging): abandonment early-warning (features from the first K turns -> P(never completes)), thrash-loop early-warning (feeds the advisory hooks bfv with a probabilistic signal), cost/duration forecasting for a session in progress, next-week cost forecast (upgrading cost_outlook from extrapolation to a proper interval forecast).
Execution shape: (1) Feature vectors from the substrate only: early-session structural signals (turn counts, tool mix, failure count, retry pattern, repo, model, hour) — a features module documented in the measure registry (each feature = a measure; confounds inherited).
Done when: One model (abandonment early-warning) trained via campaign on the live archive with time-split eval; model card shows calibration + Brier + AUC vs a base-rate baseline; beats baseline meaningfully or the bead closes with the negative result recorded.

### 333. `polylogue-9l5.12` — Information-theoretic and graph measures: redundancy, diversity, tree shapes

P3 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: Cheap, construct-transparent measures that add texture no rollup has: (1) REDUNDANCY via compression ratio — zstd ratio of session prose is a real measure of repetitiveness (context thrash shows up as low ratio-of-new-information; pairs with lineage dedup work); (2) DIVERSITY via entropy — tool-use distribution entropy per session/repo/model = workflow diversity (a model that only ever shell+edit's vs one using the full affordance set); (3) TREE SHAPES — subagent fan-out distributions, depth hi…
Execution shape: All four are afternoon-sized measures over existing tables, registered via 9l5.7 with explicit construct notes (compression ratio operationalizes 'textual redundancy', NOT 'quality'; entropy operationalizes 'affordance diversity', NOT 'skill' — the registry's construct field exists precisely to pin these).
Done when: Four measures registered with construct notes and tier labels; redundancy + entropy computed for the seeded corpus and queryable as session fields; fan-out/depth distributions render for a session tree; co-occurrence edges exported for the surface-economy audit.

### 334. `polylogue-9l5.3` — Pathology epidemiology: corpus-level rates and trends

P3 feature · open · parent: polylogue-9l5
Intent: Detectors are per-session and deterministic; the epidemiology is missing: pathology rates over time, by model, by repo, by tool — 'is thrash-looping getting better since the March harness change?' Materialize as an archive-level insight (registry pattern) so CLI/MCP get it for free.
Execution shape: Corpus-level layer over the per-session deterministic detectors: which models, repos, prompt styles, hours-of-day correlate with agent_hanging, question_left, thrash loops — 'a natural group-by away'.

### 335. `polylogue-9l5.4` — Token-economy analytics: cache-lane and attention accounting

P3 feature · open · parent: polylogue-9l5
Intent: Cache-lane accounting is disjoint and honest; nothing yet answers 'what fraction of apparent token throughput is cache-read amplification, per provider, over time' or 'context budget spent on tool results vs prose'.
Execution shape: Two derived metrics beg to exist, both structurally computable today: (1) context amplification — bytes re-read / bytes unique per session (how much of the context is churn); cache-lane accounting is disjoint and labeled, so cache-read amplification per provider over time is exact where provider-reported.

### 336. `polylogue-9l5.5` — Ship opinionated saved views as product defaults

P3 feature · open · parent: polylogue-9l5
Intent: The DSL pipeline is the analytics API; ship the canned questions as named saved views (thrash loops, abandoned-with-question, most-expensive-failures, tools-that-break, model-failure-rates) so the analytics are discoverable without learning the grammar first.
Execution shape: Two halves: (1) ship the defaults — named saved views for the canned questions (thrash loops, abandoned-with-question, most-expensive-failures, tools-that-break, model-failure-rates), listed in help + web; (2) parametrized saved queries (fables ladder item 9): saved_query assertions with $holes — `polylogue q thrash-loops repo=polylogue` substitutes into the stored expression.

### 337. `polylogue-9l5.9` — Survival analysis: session duration, abandonment hazard, time-to-outcome

P3 feature · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: Sessions end three ways — completed, abandoned, still-open — which is textbook censored-duration data, and nothing in the archive treats it that way: mean session length lies when abandonment is common (it averages over censoring).
Execution shape: (1) Duration events from structural evidence only: start = session start; terminal event = completed (structural terminal state) vs abandoned (find_abandoned_sessions criteria) vs censored (still open / archive edge).
Done when: KM estimator property-tested against a reference implementation on synthetic censored data.

### 338. `polylogue-b1n` — WebUI-driven posting: operator drives web chats from the workbench

P3 feature · open · parent: polylogue-bby
Prerequisites: polylogue-ptx.
Intent: Operator direction 2026-07-03 (with the ptx un-gate): posting should eventually be user-drivable from the web UI, not only agent-drivable.
Execution shape: Reuse the ptx receiver/extension posting path as the single write mechanism — the webui is a second CLIENT of the channel, not a second channel.

### 339. `polylogue-bby.12` — Session replay: play a session back the way it happened

P3 feature · open · parent: polylogue-bby
Intent: The forensic UX nothing else has: scrub through a session on its own event-time — messages and tool calls appear at recorded timestamps (or compressed 10x/60x), the ap7 semantic cards animating in sequence, with a parallel rail showing repo state advancing (commits from 7xv, file-change markers from yrx) and cost/token burn accumulating.
Execution shape: Pure derived view — zero new data: occurred_at_ms drives the event timeline (gaps compressed by a max-idle knob); the scrubber is the timeline component (bby.10) scoped to one session; playhead position = a message-position cursor, so pause = ordinary reader at that point (replay and reading are the same view in different time modes, not two views).
Done when: A real 200+ message session replays with correct event pacing at 10x, pause drops into the normal reader at the playhead, cost rail matches session totals at the end, and a replay recording of a seeded session is committed as a demo asset.

### 340. `polylogue-bby.13` — The day page: a daily narrative the operator actually reads

P3 feature · open · parent: polylogue-bby
Intent: Day summaries exist as data with no operator-facing face.
Execution shape: One view over existing read models (day summaries, session_stats/profiles, yrx aggregates, judge queue, blackboard) — the bead is composition + layout, not new computation; served from the daemon cache keyed by day+cursor (immutable for past days = cached forever).
Done when: Yesterday's page renders on the live archive with every number expanding to refs; past-day loads are cache-instant; CLI twin renders the same composition; open-loops section links straight into judge/resume actions.

### 341. `polylogue-bby.2` — Query completions + expression explain in the web search box

P3 feature · open · parent: polylogue-bby
Intent: The search box accepts the full DSL but exposes none of it: no completions, no explain, no error recovery.
Execution shape: /api/query-completions already exists as a route (http.py:258-283) and drives shell completion — the box just doesn't use it.

### 342. `polylogue-bby.3` — Aggregate analytics views in the web UI

P3 feature · open · parent: polylogue-bby
Intent: No aggregate view exists — the inspector is session-by-session while the substrate has rollups, facets, group-by pipelines, and cost relations.
Execution shape: An /analytics route with a handful of DSL-pipeline-backed panels: cost by origin/outcome over time, tool failure rates, workflow-shape mix.

### 343. `polylogue-bby.4` — Live session tailing as a first-class mode

P3 feature · open · parent: polylogue-bby
Intent: SSE granular topics + append-mode ingestion exist; the UI treats the archive as static.
Execution shape: SSE granular topics and append animations already exist; missing is a 'follow' toggle on an in-flight session: auto-scroll transcript tail + a running cost/latency ticker fed from the insights endpoints + a capture-latency chip from ingest-cursor timestamps.

### 344. `polylogue-bby.5` — Long-session navigation: phases/windows/minimap

P3 feature · open · parent: polylogue-bby
Intent: The substrate has temporal windows, phases, and work events; the reader has a flat scroll.
Execution shape: Minimap/timeline scrubber rendered from session_phases + temporal buckets: role-colored segments, error-marked via the keystone outcome fields (tool_result_is_error), click-to-jump; keep gg/G.

### 345. `polylogue-bfv` — Advisory hooks: archive-informed PreToolUse/UserPromptSubmit responses

P3 feature · open
Prerequisites: polylogue-d1y, polylogue-20d.1, polylogue-20d.12.
Intent: Hooks currently flow one direction: harness -> archive.
Execution shape: RESTRAINT IS THE DESIGN (jgp: restrained volume; an advisory layer that cries wolf gets disabled in a week): (1) PreToolUse advisory — fires ONLY on high-confidence, high-value matches: exact-command (normalized) with >=N structural failures in trailing window -> inject one line ('this command failed 4x this week: <ref>'); never blocks (advisory, not permission — PermissionRequest stays untouched); hard budget one advisory per tool call, with pe…

### 346. `polylogue-fnm.3` — SEQ modifiers: within:<duration> and {n,} occurrence counts

P3 feature · open · parent: polylogue-fnm
Intent: SEQ matches non-contiguous subsequences over ToolCategory with no time/count constraints.
Execution shape: Full SEQ v2 scope (fables ladder item 6) — three parts, span capture is the valuable one: (1) gap/adjacency modifiers ->[within:5m] and ->[next] (grammar: sequence_step/ARROW region expression.py:658-666; semantics in Python matcher runtime_matching.py:66-99 — track occurred_at_ms deltas; VERIFY hydrated Action carries a timestamp, else extend build_session_semantic_facts, and memoize it — 3 matchers call it uncached); (2) repetition SEQ((edit -…

### 347. `polylogue-fnm.5` — topic-pack: staged multi-channel topic-lineage retrieval

P3 feature · open · parent: polylogue-fnm
Intent: Composition flagship: seeds -> signal extraction -> embedding expansion -> time-neighbor -> topology -> classify -> timeline/context-pack/gaps.
Execution shape: Staged rounds (gh#2482 design, code-grounded primitives all exist): seeds via FTS + hybrid/similar_text vector; signal extraction from seed bodies (terms/files/branches/issues); embedding expansion; time-neighbor via discover_neighbor_candidates (6 channels); topology via get_session_topology/logical_session/topology_edges; LLM-snippet classify (optional, judged); emit timeline + context-pack + gaps.

### 348. `polylogue-fnm.7` — Generalized child-count predicates: count(unit where ...) comparisons

P3 feature · open · parent: polylogue-fnm
Intent: `sessions where count(action where is_error:true) >= 3` — quantified child predicates beyond EXISTS.
Execution shape: Target: `sessions where count(action where is_error:true) >= 5`.

### 349. `polylogue-fnm.8` — Lineage scope operator: logical: prefix expands predicates across the session family

P3 feature · open · parent: polylogue-fnm
Intent: `logical:` scope makes a predicate evaluate over the whole logical session (root + continuations/forks) via session_profiles.logical_session_id — 'sessions (logical) where terminal_state:abandoned AND exists message text:X anywhere in the family'.
Execution shape: Anchor: archive/query/expression.py — pipeline stages are hand-parsed OUTSIDE the Lark grammar (split on |, _parse_pipeline_unit_source ~L1949), so logical: needs no grammar change if implemented as a predicate-expansion pass; if it becomes a TERMINAL containing ':', it must slot above FIELD_CLAUSE.4 priority or it is eaten as a field clause (standing LALR trap).
Done when: logical:REF in any predicate position returns the recomposed logical session set; replayed prefix messages are not double-counted in downstream aggregates; grammar tests cover the terminal-priority trap.

### 350. `polylogue-fs1.10` — Spec-cards: sessions as portable benchmark items (leakage-gated export)

P3 feature · open · parent: polylogue-fs1
Intent: A session with intent + initial SHA + acceptance signal + final diff becomes a portable benchmark row WITHOUT transcript leakage: index.db spec_cards deriving task title/intent, repo/commit refs, outcome EVIDENCE TIER (pr_merged vs explicit_ref vs intent-only are different tiers — reproducibility gated on high-confidence commit attribution), verification command/result, completeness.
Done when: Deterministic rows on fixtures; tier separation tested; export leaks nothing by default; rebuild parity after reset --index.

### 351. `polylogue-fs1.2` — Importer: NeMo Relay ATOF/ATIF runtime spans

P3 feature · open · parent: polylogue-fs1
Intent: Import Hermes observer-layer trace exports as runtime span evidence: pre/post_api_request -> LLM request spans; pre/post_tool_call -> tool execution spans with duration/status; approval hooks -> high-risk decision points; subagent hooks -> delegation graph; error hooks -> retry/fallback taxonomy.
Execution shape: VERIFY first: current NeMo Relay plugin output shape in the Hermes repo (ATOF JSONL / ATIF JSON exported from observer hooks).

### 352. `polylogue-fs1.3` — Per-source coverage/fidelity declaration for Hermes imports

P3 feature · open · parent: polylogue-fs1
Intent: Declare what each source tier actually provided: hermes-state-db (authoritative), hermes-json-snapshot (fallback), hermes-atof/atif (spans), hermes-langfuse (optional), hermes-log-dir (degraded recovery).
Execution shape: Build on the existing provider-completeness taxonomy (devtools lab provider completeness) rather than a new mechanism.

### 353. `polylogue-fs1.5` — Export: Atropos/eval JSONL downstream of the canonical archive

P3 feature · open · parent: polylogue-fs1
Intent: Convert archived Hermes (and other agent) sessions into Atropos-compatible eval/RL trajectories: canonical archive -> eval JSONL, NOT bespoke snapshot->export (that shape is a one-off parser and duplicates Hermes's own NeMo Relay).
Execution shape: Downstream export only: canonical archive session -> Atropos/eval JSONL (trajectory = messages + tool calls + structural outcomes + terminal state).

### 354. `polylogue-fs1.6` — Fully-sovereign loop demo: local Hermes -> archive -> local embeddings -> judged memory -> injection, air-gapped

P3 feature · open · parent: polylogue-fs1
Prerequisites: polylogue-37t.5.
Intent: The demo only this pairing can do: Hermes (open weights) running locally behind the existing LiteLLM gateway (127.0.0.1:4000) -> sessions tailed into the archive (watcher already covers ~/.hermes/sessions) -> local embeddings -> judged assertions -> context injected into the next Hermes session.
Execution shape: Composition, not new machinery: every stage exists or is beaded — LiteLLM gateway (sinnix litellm.nix), hermes origin + watcher coverage, 37t.5 local embedding lane, assertion judgment + compose_context_preamble.

### 355. `polylogue-fs1.7` — Upstream native integration: lifecycle hooks / polylogue-hook support in the open-source Hermes agent

P3 feature · open · parent: polylogue-fs1
Intent: Claude Code and Codex offer fixed hook surfaces Polylogue adapts TO; the Hermes agent is open source, so integration can go the other direction — contribute generic lifecycle-hook support (or a polylogue-hook integration) upstream.
Execution shape: Sequence: (1) VERIFY current Hermes agent architecture before designing — it ships ~1,700 commits per minor release and v0.15 already collapsed run_agent.py into an orchestrator + 14 modules; the observer/middleware layer that exists (Langfuse, NeMo Relay ATOF/ATIF) is the natural attachment point, so the upstream PR may be 'generic lifecycle hook events on the observer bus' rather than anything Polylogue-named.

### 356. `polylogue-fs1.9` — Polylogue->Sinex derived agent-trace event emitter

P3 feature · open · parent: polylogue-fs1
Intent: Implements the polylogue-6mv boundary: Polylogue emits derived, privacy-preserving events onto the Sinex event stream instead of Sinex ingesting raw transcripts.
Done when: Emitter produces each of the six declared event kinds with only anchor + derived-fact fields, proven by a contract test that fails if any raw-text field is populated; a local emit->consume round-trip test shows events land on the Sinex stream and correlate via polylogue://session/<id>; emission is a no-op (no error) when Sinex is unreachable; `devtools verify` passes.

### 357. `polylogue-jnj.9` — Intentional runtime/deployment configuration surface

P3 feature · open · parent: polylogue-jnj
Intent: One coherent config surface across user/NixOS/daemon/CLI/MCP/web: ownership per layer, effective-value inspection, recovery from invalid config.
Execution shape: Today runtime/deployment config is env-var folklore (POLYLOGUE_ARCHIVE_ROOT, POLYLOGUE_FORCE_PLAIN, worker counts, pytest paths...).
Done when: config list shows every recognized key, its value, and winning layer; unknown-key set warns; docs page generated from the same inventory (no hand-maintained table).

### 358. `polylogue-l4kf.1` — polylogue-export origin + CIF envelope: import(export(A)) is a content-hash no-op

P3 feature · open · parent: polylogue-l4kf
Intent: Make the archive its own re-ingestable Origin: an export package (CIF-like envelope) carrying content_hash_algo, EMBEDDED ORIGINAL ORIGIN, source manifest, parser fingerprint, fidelity declaration, blob/hash inventory.
Done when: Round-trip fixture: identical session/message/block ids; transport provenance queryable; collision fixture quarantines; enum/mapping/parser-registry/docs/tests all include the new origin.

### 359. `polylogue-mhx.5` — Semantic analytics surfaces: topics/clustering, novelty, near-duplicate assist

P3 feature · open · parent: polylogue-mhx
Prerequisites: polylogue-mhx.2.
Intent: Vectors currently serve only point retrieval (--similar/--semantic/hybrid).
Execution shape: All three are read models over session vectors (emb-targets class 2), built as analyze projections, not scripts (compositionality rule): (1) `analyze topics [--period month]`: cluster session vectors (k-means or HDBSCAN — pick by silhouette on the live corpus, decide once), label clusters structurally (top TF-IDF terms from titles/summaries + dominant repo/origin — no LLM naming in v1), emit cluster rows with member refs; DSL integration: cluste…

### 360. `polylogue-mhx` — Embedding substrate: provider-general, honest lifecycle, retrieval that earns its cost

P2 epic · open
Intent: Current state: one hardcoded cloud provider (Voyage voyage-4, 1024-dim, constants in sqlite_vec_support.py), vec0 fixed-dimension tables, embedding targets limited to authored prose messages (v21 partial index), opt-in daemon catch-up with cost caps, hybrid RRF + --semantic/--similar surfaces, ops embed onboarding group.
Execution shape: Epic scope: provider/model generality (one OpenAI-compatible embedding client covering local and cloud), an explicit embedding-target policy (what gets a vector and why), retrieval quality measured rather than assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the lane (semantic recall in context compilation, clustering/topics, novelty detection).
Done when: 1.

### 361. `polylogue-scd` — Cross-surface handoff: polylogue open + copy-as-command everywhere

P3 feature · open · parent: polylogue-jnj
Intent: The CLI, webui, and MCP each dead-end at their own borders: a CLI result cannot jump to the richer web view ('polylogue open' does not exist); a webui row cannot be turned into the equivalent CLI command for scripting; an MCP payload ref requires manual reconstruction to inspect by hand.
Execution shape: (1) 'polylogue open <ref>' resolves any ref (session/message/assertion, or 'last') to the workbench deep link and opens it (xdg-open; prints the URL when headless; starts nothing — if the daemon is down it says so and prints the URL for later).

### 362. `polylogue-uiw` — Origin breadth: enumerate the target set + generic openai-chat-shape detector

P3 feature · open · parent: polylogue-l4kf
Intent: The detector registry makes each new origin mechanical, but nobody has enumerated the target set.
Execution shape: Enumerate candidates (aider chat history, Cline/Roo task logs, OpenHands trajectories, open-webui exports, LM Studio, llama.cpp server logs, custom scripts), rank by user-base x format-stability, then build ONE generic openai-chat-json detector covering the near-identical OpenAI chat-array shapes before writing any per-tool parser.

### 363. `polylogue-wmj` — OTel GenAI trace export lane

P3 feature · open · parent: polylogue-l4kf
Intent: Project Run/ObservedEvent/tool/subagent/context data to OTel trace/span/span-event form (GenAI semantic conventions) WITHOUT making OTel internal authority — export-only lane.
Execution shape: Export lane: archive sessions -> OTel GenAI semantic-convention spans (gen_ai.* attributes) so LangSmith/Langfuse/Phoenix-class consumers can read Polylogue evidence.
Done when: polylogue export --format otel-genai-jsonl produces spans that pass an OTel GenAI semantic-convention validator for a sample session set; tool pairs land as parent/child spans; cost attributes present where evidence exists.

### 364. `polylogue-0cg` — OTel GenAI semantic-conventions ingest: any instrumented agent framework becomes an origin

P3 feature · open · parent: polylogue-l4kf
Prerequisites: polylogue-wmj.
Intent: The daemon already has an OTLP receiver (/v1/traces).
Execution shape: Map GenAI spans/span-events to the normalized model: gen_ai.* attributes -> messages/blocks (prompt/completion events -> message rows; tool spans -> tool_use/tool_result blocks with structural outcomes from span status).

### 365. `polylogue-y0b` — Generated codebase atlas: the grok report as a rendered, drift-checked doc

P3 feature · open · parent: polylogue-3tl
Intent: Re-deriving 'what is this project, mechanically' from source took a frontier-model session a three-ring parallel read to produce ~15 numbers: LOC by ring, 54 tables across 5 tiers, 6 verbs / ~50 commands / ~392 flags, 126 facade methods, ~61 MCP tools, ~45 daemon routes, 10 origin families.
Execution shape: New devtools render atlas producing docs/atlas.md from source-of-truth registries, not from prose: verb_names.py + command_inventory for CLI counts, archive_tiers DDL for table counts per tier, MCP EXPECTED_TOOL_NAMES for tool count, daemon route table for route count, dispatch.py for origin families, cloc-style LOC by placement-rule ring (topology projection already classifies paths).

### 366. `polylogue-3tl` — External legibility: a stranger can understand, run, and cite Polylogue

P1 epic · open
Intent: Every finished artifact proves the substrate is honest; this program makes the project legible to someone with no context.
Done when: Terminal state: a stranger can (1) understand from the README's first screen, (2) run the one-command demo successfully, (3) cite a published finding URL.

### 367. `polylogue-y8w` — Reading preferences bundle: per-scope views, fold budgets, rows, pager

P3 feature · open · parent: polylogue-w8db
Prerequisites: polylogue-y4c.
Intent: Implementation bundle for the reading-class runtime prefs from the y4c inventory: per-scope default view preset (origin/repo/surface-scoped), per-block-type fold budgets as the default 1lm profile, timestamp style + timezone, row density, result-row column selection (x7d set), pager threshold, auto-read-on-single-hit.
Execution shape: Each pref = one registry key with scope-chain resolution (y4c machinery) + the consuming surface reading it: default view -> read view resolution (after 4pm fixes defaults); fold budgets -> 1lm preset selection; columns -> x7d row contract; auto-read -> find result cardinality check.
Done when: All listed keys settable/scopable with live effect; codex-origin skeleton-view default demonstrated; auto-read opens the single hit; snapshot tests cover two values per pref.

### 368. `polylogue-w8db` — Configuration doctrine + DB-backed runtime preferences

P2 epic · open
Intent: WHY: configuration semantics are scattered (env vars, hardcoded defaults, dead user_settings table) and nothing distinguishes deployment config from runtime preference from learned default.
Execution shape: Clearest missing-epic signal: y4c is a doctrine spine with four dependent implementation bundles, all orphaned.
Done when: Epic exists and owns y4c (spine) + 3xx, y8w, 6kh, 1jc.

### 369. `polylogue-yp0` — Daemon internal event bus: loops subscribe, polling retires

P3 feature · open · parent: polylogue-a7xr
Prerequisites: polylogue-9e5.7.
Intent: The daemon runs ~9 concurrent loops (the 9e5.7 lock/starvation audit maps them) coordinating through polling and shared tables: convergence checks for debt, embedding catch-up polls pending counts, FTS status re-derives, alerts re-scan.
Execution shape: (1) A small typed pub/sub (asyncio, in-process, no broker): events are frozen dataclasses — IngestCommitted(cursor, session_refs), CursorMoved, ConvergenceStateChanged, EmbeddingPending(count), BlobLeaseReleased...

### 370. `polylogue-9e5` — Audit lane: read-only analyses producing evidence artifacts

P3 epic · open
Intent: The follow-up analysis catalog from the fables deep-dive: each item is a bounded, read-only analysis a sidecar agent can run during wait windows (PROCESS.md already prescribes 3-4 bounded sidecar research agents when the backlog is thin or a long command runs).
Done when: Every child produces a READ-ONLY evidence artifact and never mutates product code.

### 371. `polylogue-9l5.18` — Missing data-model units: entity-mention, world-effect, verification-run, project, topic-cluster, cross-origin-thread

P3 epic · open · parent: polylogue-9l5
Intent: The ontology gaps identified across the R&D program — each is a derived unit that makes a class of questions queryable.
Done when: Each unit queryable as a terminal source with honest evidence tiers; candidate mentions gated by recursive-safety; project is the durable one (batched into user-v5); no name collisions with raw_artifacts/threads; TABLE-vs-VIEW matches the decision matrix.

### 372. `polylogue-a7xr` — Substrate consolidation: kill the storage twins and split the god-modules

P3 epic · open
Intent: WHY: internal duplication is where correctness quietly dies — the sync/async storage twins must be edited in pairs or daemon and CLI diverge (standing trap, see storage twins memory), god-modules resist review, and dead/double-declared tables (fts_freshness_state, 1ty) mislead every new reader.
Execution shape: Internal-debt cluster distinct from t46's surface focus: storage sync/async twins (hiu-adjacent), god-modules, dead tables, the fts_freshness_state double-declaration (1ty), and other consolidation debt (0aj, yp0, 48h, pf1, 1a9, dab, c9y).
Done when: Each twin/god-module has a consolidation bead with before/after; no duplicate schema declarations remain; devtools verify layering + schema-versioning stay green.

### 373. `polylogue-jnj` — Product surface algebra: one rule per concern across CLI/config/onboarding

P3 epic · open
Intent: The CLI grew per-view flags, boolean mode muxes, mixed output dialects, and onboarding remnants faster than its projection/render algebra.
Execution shape: Rule-per-concern map the children implement: output dialect (jnj.3), read-ref semantics (jnj.4), demo/import split (jnj.6), vocabulary hygiene (jnj.7), runtime config surface (jnj.9), bare-root behavior (jnj.13).

### 374. `polylogue-s8q` — Make deployed Polylogue state trustworthy; captures queryable

P4 bug · open · parent: polylogue-8jg9
Intent: Deployed-state contract skew.
Execution shape: Parked P4 while prod polylogued.service is inactive on sinnix — the trust problem: after a deploy, nothing proves the running daemon matches the repo state (version, schema versions, config) and recent captures are queryable.
Done when: (Provisional until unparked) After a deploy, one command reports version-match + schema-match + fresh-ingest-per-origin; a deliberate version skew is detected.

### 375. `polylogue-8jg9` — Operational resilience: recoverable, restorable, survives daemon death and deploy

P2 epic · open
Intent: WHY: an archive whose pitch is durable evidence must itself survive incidents — daemon death mid-write, bad deploys, disk loss.
Execution shape: Backup-restore (4be, quarterly restore drill), daemon-death recovery (peo), and deploy-trust (s8q, parked P4 while prod polylogued is inactive) had no home.
Done when: A quarterly restore drill proves backups restore (4be); daemon crash mid-convergence recovers without stranding debt (peo, ties 1xc.3/1xc.4); deployed state is provable via deployment-smoke when prod is re-activated (s8q).

### 376. `polylogue-2jj` — IssueBench: real issues as coding-agent effectiveness benchmarks

P4 task · open
Intent: Research lane (gpt-pro synthesis + raw-log agent-evals idea): closed beads/issues with their authoring sessions become benchmark tasks — time-to-first-patch, search depth, question count, spec-mismatch count, rework, context tokens to green; and the raw-log variant: agents experiment with their own setup (context/memory configurations) and store judged observations as assertions.
Execution shape: Real closed issues as a coding-agent benchmark: sample N closed polylogue GH issues with verifiable outcomes (merged PR + tests), reconstruct the pre-fix repo state (base commit before the fix PR), and package issue text + repo ref + the fix PR's test as SpecCards (fs1.10 schema — internal schema first, adapters second per the D07 doctrine).
Done when: (Vision — no fabricated AC) Requires: fs1.10 SpecCard schema landed; a first hand-built card set (~10 issues) proving the reconstruction recipe; leakage policy written.

### 377. `polylogue-9l5.16` — Trajectory Quality Index: reward-shaping composite, never truth

P4 task · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: WHY: one 0-1 trajectory score is useful for dashboards and RL reward-SHAPING.
Execution shape: Component subscores (each an existing/planned measure, coverage-gated): outcome_sub (structural success per 9l5.1), efficiency_sub (from the 9l5.14 scorecard vector), error_sub (tool-error + unacknowledged-failure rates), fragmentation_sub (phase-churn heuristic — LOWEST weight, labeled heuristic-tier), correction_sub (operator-correction density inverted).

### 378. `polylogue-9l5.17` — Model-drift observatory: candidate changepoints with validity gates, never causal claims

P4 task · open · parent: polylogue-9l5
Prerequisites: polylogue-9l5.7.
Intent: WHY: same task-shape across (model, month) reveals cost/turns/error drift — the "did this model get worse for my work" question.
Execution shape: Changepoint mechanics (classical, no ML): per (model_family, task-shape cohort) monthly series of median cost/turns/error-rate; candidate changepoint = rolling two-window median shift exceeding a MAD-scaled threshold, confirmed by permutation test (label-shuffle p<0.05); n_min per window enforced by the MeasureSpec coverage gate (REFUSE below floor, never extrapolate).

### 379. `polylogue-9l5` — Outcome-grounded analytics: the archive answers 'so what' questions

P2 epic · open
Intent: The archive answers 'so what' questions.
Execution shape: Epic: the archive answers 'so what' questions, layered over the Layer-0 substrate (profiles, work events, phases, threads, five-axis cost rollups, structural pathologies, followup_class, run projection, topology/logical sessions, tool timing, workflow shapes).
Done when: 1.

### 380. `polylogue-c36` — Native-compilation probe: mypyc first, only where profiles demand it

P4 task · open
Prerequisites: polylogue-20d.15.
Intent: Cython-or-similar consideration (operator ask), analyzed 2026-07-03.
Execution shape: (1) Profile bulk replay (py-spy during a 20d.15 run): if pure-Python transform is <20% of wall after parallel parse, close as not-worth-it (the honest likely outcome).
Done when: Profile artifact committed; decision recorded with numbers (close-as-no valid); if adopted: one compiled module behind an extra, matrix green, >=1.5x end-to-end documented.

### 381. `polylogue-fs1.8` — Nous Chat browser-capture adapter

P4 task · open · parent: polylogue-fs1
Intent: When Nous Chat web usage starts, add a provider adapter to the browser extension — mechanical work on the existing capture pattern (chatgpt/claude/gemini adapters).
Execution shape: Browser-capture adapter for Nous Chat (chat.nousresearch.com): a site adapter in browser-extension/ (MV3; existing chatgpt-dom adapter is the template) mapping the DOM/fetch shapes to the capture envelope, + origin routing so captured payloads land as hermes-session or a dedicated origin (decide with the fs1 epic owner — Hermes state.db import (fs1.1, done) may make DOM capture redundant except for web-only usage).
Done when: A live Nous Chat session captures end-to-end into the archive with correct origin + native ids; duplicate-vs-state.db ingest is reconciled (content-hash idempotency, no double sessions).

### 382. `polylogue-fs1` — Hermes bridge: state.db + runtime spans -> canonical evidence -> forensics/eval export

P2 epic · open
Intent: Positioning: Hermes acts; Polylogue remembers and explains what the agent did; Sinex knows what was happening on the machine around it.
Execution shape: Bridge Hermes's durable state.db (sessions, messages, tool calls, token/cache/reasoning/cost counters, parent sessions, compaction/archive/rewind state, FTS) plus observer/runtime spans into canonical Polylogue evidence — NOT the optional ~/.hermes/sessions/session_*.json snapshots (a compatibility adapter, likely off by default).
Done when: - Current Hermes internals are verified against the live build before building — the state.db schema (sessions, messages, tool calls, token/cache/reasoning counters, costs, parent sessions, compaction/archive/rewind, FTS) is confirmed and the verification note is recorded on the bead.

### 383. `polylogue-gqx` — Desktop presence spike: Polylogue in the operator's ambient environment

P4 task · open
Prerequisites: polylogue-20d.13, polylogue-scd.
Intent: Exploratory (operator: 'IDK if there's anything here'): the operator lives in Hyprland + kitty + noctalia-class widgets; Polylogue's ambient value could surface there without new heavy UI.
Execution shape: Timeboxed spike, sinnix-consumer posture: Polylogue ships stable substrate (SSE, deep links, status JSON — all already beaded); the desktop pieces are sinnix dotfile/module work CONSUMING that substrate, so most deliverables land in the sinnix repo with only gap-fixes landing here (e.g.
Done when: Spike report committed (bead close reason): each candidate prototyped-or-rejected with a reason; anything kept has its sinnix-side implementation merged and any polylogue-side gaps filed as beads.

### 384. `polylogue-lu1` — Ambient theming: terminal respects the environment, webui gains a theme system

P4 task · open
Intent: Raw-log 06-18/05-30: CLI colors should respond to the environment (pywal-class dynamic palettes) rather than hardcode a scheme; the webui needs a real theme-set — the operator wants element-level customization as a discovery path ('I don't know how to describe what I want').
Execution shape: (1) CLI: semantic use of the terminal's own 16-color palette (never hardcoded RGB) so pywal/terminal themes propagate free; NO_COLOR/FORCE_PLAIN respected (exists); one knob (color: auto|always|never) per y4c.
Done when: CLI renders correctly under a palette swap without restart (snapshot under two palettes); webui token file swaps themes globally including the overlay; demo recordings pin their theme.

### 385. `polylogue-wohv` — messages_fts UNINDEXED columns are write-only noise in a contentless table: drop or annotate

P4 task · open
Intent: Found during 2026-07-06 grok + operator design review of the contentless-FTS decision.
Execution shape: Two options.
Done when: Either the DDL carries an accurate comment (option a) or the columns are gone from table+triggers with an index-tier version bump batched alongside another index change (option b), and rg shows no reader referencing the removed columns.

### 386. `polylogue-4g5` — Expose the archive as an HPI module and Promnesia source

P4 feature · open · parent: polylogue-l4kf
Intent: The QS/local-first community (HPI, Promnesia, ActivityWatch users) is Polylogue's actual lineage and the friendliest audience — precisely the people who evangelize tools like this.
Execution shape: HPI module: a thin my.polylogue namespace package yielding typed session/message iterators over the read API (sync surface exists).

### 387. `polylogue-611` — Grok (xAI) conversation export importer

P4 feature · open · parent: polylogue-l4kf
Intent: BLOCKED: needs a real xAI export fixture before parser internals are frozen against a guessed shape.
Execution shape: The plumbing already reserves the seat (verified live): Origin.GROK_EXPORT in core/enums.py:50, source family grok-export in core/sources.py:126, provider_identity maps xai->grok.
Done when: A real Grok export ingests end-to-end (sessions/messages/blocks with native ids); detector fixture proves no other parser claims it and it claims no other fixture; parser props test added (protected file family).

### 388. `polylogue-7k7` — Research-tooling export lane: inspect-ai / Docent formats

P4 feature · open · parent: polylogue-l4kf
Prerequisites: polylogue-fs1.5.
Intent: Research groups analyze agent transcripts at benchmark scale with dedicated tools (Transluce Docent, inspect-ai eval logs, trajectory corpora).
Execution shape: Same downstream-of-canonical-archive pattern as the Atropos export (polylogue-fs1.5) — exports are projections, never a second source of truth.

### 389. `polylogue-ale` — External link archival: sessions cite URLs; the evidence should not rot

P4 feature · open · parent: polylogue-l4kf
Intent: Gwern-lineage hygiene: sessions cite external URLs constantly (docs, issues, papers) and they rot — a two-year-old session citing a 404 has lost its evidence.
Execution shape: (1) URL extraction is structural (link syntax in prose + WebFetch/WebSearch inputs); fetch politely (rate-limited, robots-aware, size-capped, opt-out domains), store readability-extracted text as blob + metadata (url, fetched_at, status, hash) — evidence preservation, not mirroring.
Done when: Opt-in flag archives cited pages with dedup and caps; URL inventory recorded regardless; archived copy reachable from a seeded session's link card; retro command archives a bounded batch.

### 390. `polylogue-bby.14` — Pinboard: workspaces as a spatial surface

P4 feature · open · parent: polylogue-bby
Intent: Workspaces/recall packs exist in user.db with no surface that makes them feel like a place.
Execution shape: Board = workspace rows + layout metadata (positions as workspace item payload — additive user.db migration); tiles are the existing card components (v2); query tiles re-run through the cache and show deltas since pinned.
Done when: Pin a session/message/query from the reader; arrange and persist layout; query tile live-updates via SSE; board exports to a read-package; layout survives daemon restart.

### 391. `polylogue-bby` — Web workbench: from result list to evidence cockpit

P2 epic · open
Intent: Web audit findings (fables session): the MK2 shell is a solid three-pane reader but hides the product's depth — the DSL, aggregates, live capture, and long-session structure are all invisible at the point of use.
Execution shape: Epic: evolve the MK2 three-pane web reader from a result list into an evidence cockpit that surfaces the product's depth at the point of use, the DSL/query algebra, aggregates, live-capture/daemon status, and long-session structure, all currently invisible in the shell.
Done when: 1.

### 392. `polylogue-fnm.9` — Pipeline-as-subquery composition

P4 feature · open · parent: polylogue-fnm
Prerequisites: polylogue-fnm.1, polylogue-fnm.7.
Intent: `sessions where in(messages where text:timeout | group by session | count >= 5)` — full subquery composition.
Execution shape: Pipeline-as-subquery: allow a pipeline result to feed an outer expression (sessions where id in (<pipeline>) or from result-set/query refs per rxdo.6).
Done when: An inner pipeline's session/unit set is consumable as an outer predicate operand (inline or via query:/result-set: ref); provenance records the composition; no quadratic re-execution (inner runs once).

### 393. `polylogue-fnm` — Query DSL: one grammar owns query semantics; compose instead of multiplying verbs

P2 epic · open
Intent: The Lark grammar in polylogue/archive/query/expression.py is THE query language; extend in place.
Execution shape: The Lark grammar in polylogue/archive/query/expression.py IS the query language — extend it in place, never as a parallel verb/flag path.
Done when: - New query semantics are added to the Lark grammar in polylogue/archive/query/expression.py (grep shows the grammar rule) rather than as a parallel verb or flag.

### 394. `polylogue-l4kf.2` — Federation: .well-known/ai-sessions manifest + selective content-hash sync

P4 feature · open · parent: polylogue-l4kf
Prerequisites: polylogue-l4kf.1.
Intent: WHY: local-first peers (second machine, trusted collaborator) should discover and exchange archive slices without a cloud service.
Execution shape: Manifest advertises archive id, supported origins, content_hash_algo, export profiles, freshness; never exposes private paths.

### 395. `polylogue-l4kf.3` — Outbound provenance: git notes (refs/notes/polylogue) + PR/issue citation footers + SARIF pathology export

P4 feature · open · parent: polylogue-l4kf
Prerequisites: polylogue-rxdo.4.
Intent: WHY: findings trapped in Polylogue-only reports have no reach; developer-native surfaces (git log --notes, GitHub code scanning) make evidence visible where work happens.
Execution shape: Anchors: a new cli/commands/cite.py (cite commit <sha> / cite pr) reading finding/query/session refs from the archive and writing (a) git notes under refs/notes/polylogue via subprocess git notes --ref=polylogue add (never touches working tree; push requires explicit --push with refspec), (b) a PR-footer text block to stdout for manual paste — NEVER auto-mutates GitHub.

### 396. `polylogue-r47` — Obsidian/PKM export profile: sessions and findings as wiki-linked Markdown

P4 feature · open · parent: polylogue-l4kf
Intent: The knowledgebase crowd (Obsidian, PKM, karlicoss lineage) navigates by wiki-links and frontmatter.
Execution shape: A render profile over the existing markdown renderer (rendering/renderers/), not a new exporter: --format obsidian on read/export paths, plus a bulk 'export to vault folder' recipe.

### 397. `polylogue-l4kf` — Ecosystem interop + origin breadth: more sources in, two-way citable export out

P3 epic · open
Intent: WHY: the cross-provider claim is only as strong as origin breadth — every AI surface the operator actually uses must land in the archive, and evidence must flow OUT as citable objects (two-way interop), or Polylogue is a roach motel.
Execution shape: A large orphan cluster is source-ingest breadth (t0p capture-rest-of-claude, uiw, 2qx OriginSpec, 0cg, 7xv, 611, ale) + two-way export (wmj, 7k7, r47, 4g5).
Done when: Each new origin has detector+parser+fixture+schema+docs (devtools lab provider completeness green); export paths are citable interchange, not bespoke dumps.

## End state after completing this order

- All open/in-progress Beads in this snapshot are closed with recorded evidence, leaving no blocked or ready executable Beads in the tracker for this tech-tree snapshot.
- The archive substrate is scale-hardened: index rebuilds are blue-green/fresh-first without degraded read windows; blob GC and ops doctor are lease/generation/reference safe; missing production-backup blob debt is classified/restored or explicitly accepted; FTS/search/read-model drift is measured and gated.
- Source evidence, durable user assertions, derived indexes, embeddings, and ops telemetry keep their tier boundaries: durable tiers migrate additively behind backups, derived tiers rebuild safely, disposable ops state stays disposable.
- Temporal, numeric, forensic, and provenance claims are evidence-contract driven: absent evidence is not rendered as zero, weakest temporal provenance propagates, text-derived forensic fields carry caveats, and the rigor audit iterates the registry rather than only declared contracts.
- Agent coordination is live: hooks are installable and health-checked, MCP/read-write roles are discoverable, write adoption is measured, candidate assertions flow through a single safe chokepoint, the scheduler governs context injection, and a two-agent/separate-worktree proof shows overlap awareness, scoped messaging, context injection, and handoff.
- Content variants are first-class archive objects: language detection/preferences, variant refs/nodes/alignment/storage, variant-aware query/projection/rendering, and UX for creating/reviewing/messaging variants all compose with the rest of the archive.
- Lineage truth is materialized: compaction boundaries, effective context, shared-content identity, subagent vs main-session compaction, completeness signals, physical identity collision handling, and compaction-loss/regrounding surfaces are all explicit and rebuild-stable.
- The read surface converges on one algebra: Query × Projection × Render owns CLI, daemon, MCP, web, Python API, and reader behavior; duplicate dispatch paths and surface-side math are deleted or routed through contracts.
- Interactive performance is productized: CLI startup is light, daemon fast paths/cache/push channels serve common queries in interactive SLOs, bulk ingest has bounded RSS/IO/WAL behavior, and latency regressions are continuously measured.
- The web workbench becomes an evidence cockpit: basket-to-report/export, session replay, timelines, day pages, live tailing, analytics views, long-session navigation, and citation anchors sit on the same contracts and provenance as CLI/MCP.
- The analytics layer answers outcome questions with calibrated uncertainty: work graph units, outcome/pathology measures, temporal trends, process mining, survival analysis, prediction/calibration, experiment hosting, and model-drift observatory all label construct limits rather than overclaiming truth.
- Interop/export breadth is expanded: OriginSpec owns ingestion strictness, additional origins/OTel/Hermes/Grok/etc. are importable where scheduled, Polylogue export/import is content-hash idempotent, and outbound citation/export lanes are available.
- External legibility is closed: README/category anchoring, one-command demo, install matrix, public claims ledger, proof cards, docs/visual coverage, findings publishing, and launch kit make the project understandable and citable by a stranger.