# Delivery manifest overview

Every active bead is in `delivery_manifest.csv/json`. This page gives the ordered high-signal view.

## A-trust-floor: Trust floor (59 beads)

- `polylogue-s7ae.6` P1 task — Classify the 74%-aborted full verify from the coordination commit before deploy  
  Lane: `verification-readiness`. Readiness: `A-implementation-ready`. Proof: full devtools verify log with failure classification table.
- `polylogue-8jg9.4` P1 bug — ops doctor cleanup_orphans can delete an in-flight leased blob (the real #818)  
  Lane: `blob-integrity`. Readiness: `B-local-inspection-needed`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-9e5.28` P1 bug — Rigor audit iterates contracts, not the registry: uncovered number-bearing products vanish from audit  
  Lane: `usage-cost-honesty`. Readiness: `B-local-inspection-needed`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-9e5.29` P1 bug — Number-over-empty gates: quantitative fields need field-level evidence contracts  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-9e5.30` P1 bug — Prose-mined forensic fields must carry text_derived provenance in the payload model  
  Lane: `temporal-provenance`. Readiness: `B-local-inspection-needed`. Proof: clock-seam regression tests and weakest-timestamp-source aggregate fixture.
- `polylogue-cpf.5` P1 bug — Temporal provenance laundering: aggregates collapse to provider_ts; propagate the weakest source  
  Lane: `temporal-provenance`. Readiness: `B-local-inspection-needed`. Proof: clock-seam regression tests and weakest-timestamp-source aggregate fixture.
- `polylogue-cpf.6` P1 bug — Temporal correctness: clock seam for relative-date parsing + targeted sort_key_ms audit  
  Lane: `temporal-provenance`. Readiness: `B-local-inspection-needed`. Proof: clock-seam regression tests and weakest-timestamp-source aggregate fixture.
- `polylogue-kwsb.1` P1 bug — Daemon/capture security hardening: Host/Origin gate, receiver token, spool governor  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-37t.15` P1 task — Single agent-write chokepoint in upsert_assertion: non-user authors => CANDIDATE + inject:false, always  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-83u.4` P1 task — Classify the 39,586 missing referenced blobs in the production backup  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-f2qv.1` P2 bug — Per-model token rollup double-count: session totals partitioned once (#2472)  
  Lane: `security-privacy`. Readiness: `B-local-inspection-needed`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-f2qv.5` P2 bug — Version-gate provider-usage projection so it self-heals like session_profiles  
  Lane: `usage-cost-honesty`. Readiness: `B-local-inspection-needed`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-jnj.5` P2 bug — Route ops reset --session/--source through the mutation contract  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-xy95` P2 bug — Speed up provider usage full stale diagnostics  
  Lane: `usage-cost-honesty`. Readiness: `A-implementation-ready`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-27m` P2 task — Excision and secret hygiene: the archive can forget on purpose  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-38x` P2 task — Reconcile archived audit residue against current source  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-9e5.1` P2 task — Assertion-layer adoption audit: is the flywheel used or aspirational?  
  Lane: `temporal-provenance`. Readiness: `A-implementation-ready`. Proof: clock-seam regression tests and weakest-timestamp-source aggregate fixture.
- `polylogue-9e5.19` P2 task — Storage-layer correctness scenario family  
  Lane: `evidence-honesty`. Readiness: `B-local-inspection-needed`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.
- `polylogue-9e5.24` P2 task — Sink MCP analysis primitives into insights/ + api facade; delete surface-side math  
  Lane: `evidence-honesty`. Readiness: `B-local-inspection-needed`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.
- `polylogue-9e5.25` P2 task — Review zero-use MCP surfaces from affordance usage artifact  
  Lane: `usage-cost-honesty`. Readiness: `A-implementation-ready`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-9e5.26` P2 task — Review zero-use CLI surfaces from affordance usage artifact  
  Lane: `usage-cost-honesty`. Readiness: `A-implementation-ready`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-9e5.27` P2 task — Speed up live affordance usage surface inventory  
  Lane: `usage-cost-honesty`. Readiness: `A-implementation-ready`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-9e5.3` P2 task — Column honesty audit: null/unknown density for key semantic columns  
  Lane: `evidence-honesty`. Readiness: `A-implementation-ready`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.
- `polylogue-9e5.4` P2 task — Get->modify->put race audit across daemon/CLI/MCP writers  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-cpf.4` P2 task — Enforce degrade-loudly: sweep silent soft-failure paths to carry a signal  
  Lane: `evidence-honesty`. Readiness: `B-local-inspection-needed`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.
- `polylogue-f2qv.2` P2 task — Codex disjoint-lane normalizer: decompose cached/uncached and reasoning/completion with a regression guard  
  Lane: `evidence-honesty`. Readiness: `B-local-inspection-needed`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.
- `polylogue-f2qv.4` P2 task — Single pricing source of truth: LiteLLM catalog, drop tokencost, last-path-segment match  
  Lane: `security-privacy`. Readiness: `B-local-inspection-needed`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-ivsc` P2 task — Classify Codex state_5 token drift outside lineage replay  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-5hf` P2 feature — Provider token accounting: honest cross-provider usage ledger  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-avg` P2 feature — Fold devloop claim-guard vocabulary upstream into ops status/readiness  
  Lane: `evidence-honesty`. Readiness: `A-implementation-ready`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.
- `polylogue-f2qv.3` P2 feature — Dual cost view: API-list-equivalent and subscription-credit reported separately  
  Lane: `usage-cost-honesty`. Readiness: `B-local-inspection-needed`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-f2qv` P2 epic — Provider usage & cost honesty: disjoint token lanes, one pricing source, dual cost view  
  Lane: `security-privacy`. Readiness: `A-implementation-ready`. Proof: negative Host/Origin/token/spool/security fixture suite.
- `polylogue-xnkf` P3 bug — actions view fans out on duplicate tool_ids: one logical action becomes up to NxM rows  
  Lane: `usage-cost-honesty`. Readiness: `D-horizon-ready`. Proof: usage/cost reconciliation report with disjoint lanes and empty-evidence tests.
- `polylogue-9e5.10` P3 task — Resume/context efficacy eval (observational)  
  Lane: `agent-write-safety`. Readiness: `D-horizon-ready`. Proof: candidate assertion write-path tests and rejected-candidate resurrection guard.
- `polylogue-9e5.11` P3 task — Test-suite economics: coverage vs fix-density map  
  Lane: `evidence-honesty`. Readiness: `D-horizon-ready`. Proof: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.

  …plus 24 more; see CSV/JSON for the complete list.

## B-storage-rebuild-bytes: Storage, rebuild, and byte-integrity floor (23 beads)

- `polylogue-83u.3` P1 feature — Preserve uploaded attachment bytes in live browser capture  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-1xc.12` P2 bug — FTS drift gauges + metamorphic coherence tests; rowid-reuse requires block_id check  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-peo` P2 bug — Daemon death leaves no trace: crash forensics + heartbeat sentinel + restart policy  
  Lane: `operational-resilience`. Readiness: `A-implementation-ready`. Proof: daemon crash/heartbeat fixture and backup restore drill log.
- `polylogue-1xc.8` P2 task — Schema rebuild-safety scenario  
  Lane: `storage-rebuild-scale`. Readiness: `B-local-inspection-needed`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-b5l` P1 task — Blue-green index rebuilds: fresh-first without downtime  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-4be` P2 task — Restore drill: prove the backups restore, quarterly  
  Lane: `operational-resilience`. Readiness: `A-implementation-ready`. Proof: daemon crash/heartbeat fixture and backup restore drill log.
- `polylogue-60i5` P2 task — Durable-tier batch coordination: one user v4->v5 and one source v2->v3 migration window  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-6wnh` P2 task — Bound thread refresh cost for large Codex appends  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-83u.6` P2 task — Attachment acquisition census by origin and byte volume  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-8jg9.1` P2 task — Standing backlog-hygiene invariant lint (bd devloop gate)  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-8jg9.2` P2 task — Blob-GC lease/orphan concurrency test (the acquire->commit race)  
  Lane: `blob-integrity`. Readiness: `B-local-inspection-needed`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-ma2` P2 task — Add FK-supporting index for web_content_constructs message cleanup  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-xgw` P2 task — Archive schema hygiene for evidence-cockpit read paths  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-83u.2` P2 feature — Attachment byte acquisition for non-inline sources (Drive/zip/local)  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-iec` P2 task — Schema optimization audit: storage shape earns its bytes and its reads  
  Lane: `blob-integrity`. Readiness: `A-implementation-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-8jg9.3` P3 task — SLO samples + idle-vs-stalled verdict: steady-state observability over convergence  
  Lane: `storage-rebuild-scale`. Readiness: `D-horizon-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-fie` P3 task — Decision: archive scaling doctrine — keep everything, optimize the ceilings  
  Lane: `blob-integrity`. Readiness: `D-horizon-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-1xc.10` P3 feature — Design spike: express session insights + aggregates as declared derived views over a single refresh engine  
  Lane: `storage-rebuild-scale`. Readiness: `D-horizon-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-1xc` P1 epic — Scale-hardening: bugs that only bite on real-scale archives  
  Lane: `storage-rebuild-scale`. Readiness: `A-implementation-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-83u.5` P3 feature — Blob store zstd compression (36GB -> est 5-8GB)  
  Lane: `blob-integrity`. Readiness: `D-horizon-ready`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-83u` P1 epic — Attachment & blob evidence integrity: bytes exist, are honest, and stay affordable  
  Lane: `blob-integrity`. Readiness: `B-local-inspection-needed`. Proof: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.
- `polylogue-s8q` P4 bug — Make deployed Polylogue state trustworthy; captures queryable  
  Lane: `storage-rebuild-scale`. Readiness: `D-horizon-ready`. Proof: large-corpus rebuild probe, blue-green generation swap proof, WAL/resource envelope report.
- `polylogue-8jg9` P2 epic — Operational resilience: recoverable, restorable, survives daemon death and deploy  
  Lane: `operational-resilience`. Readiness: `A-implementation-ready`. Proof: daemon crash/heartbeat fixture and backup restore drill log.

## C-read-evidence-contract: One read contract and first-class evidence objects (60 beads)

- `polylogue-4p1.1` P2 task — Route daemon split-archive fast path through SessionQuerySpec.from_params  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-4p1` P2 task — Decision: one read algebra — Query x Projection x Render as the only read contract  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-ap7` P2 task — Semantic transcript rendering: tool-call-aware, provider-agnostic, shared CLI/web renderer registry  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.11` P2 task — Pipeline/clause parity across units + generated support matrix  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-jnj.1` P2 task — Collapse read per-view flags into ProjectionSpec/RenderSpec algebra  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-1lm` P2 task — Composable transcript views: selector x transform x budget algebra  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-rxdo.1` P2 task — ObjectRef expansion: query, query-run, result-set, finding, cohort, analysis, annotation-batch kinds  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-rxdo.2` P2 task — Content-addressed query identity + durable user.db queries/query_names/result_sets/query_edges  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-rxdo.3` P2 task — Query-run + result-relation telemetry in ops.db; refs on every query envelope  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-rxdo.4` P2 task — AssertionKind.FINDING: claims with evidence, reusing the candidate->judge lifecycle verbatim  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-rxdo.7` P2 task — Annotation substrate: schema registry, annotation batches, JSONL import surface, typed value predicates  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-svfj` P2 task — Block content-hash citation anchors: blocks.content_hash + resolver with typed drift states  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-t46.3` P2 task — Unify list/search query-spec->ArchiveStore execution across CLI, MCP, and daemon web  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-t46.4` P2 task — Delegate daemon session-similarity KNN to SqliteVecProvider.query_by_session  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-t46.5` P2 task — Route CLI transcript/dialogue file export through substrate read+render; delete streaming_markdown SQL path  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-t46.6` P2 task — Fix referenced_path OR-vs-AND filter divergence and delete dead CLI stats aggregators  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-t46.8` P2 task — MCP surface collapse: ~96 tools -> verb algebra (query/get/explain/context/assert/maintenance...)  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-x7d` P2 task — Unify root query row rendering contracts  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.1` P2 feature — Aggregates beyond count (sum/avg/min/max/percentiles)  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.10` P2 feature — fields/select stage with parent-field projection (first real Transform)  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.12` P2 feature — User-defined query macros: named, composable DSL shorthands in user.db  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.13` P2 feature — Set-algebra over query results: union/intersect/except between queries  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.14` P2 feature — find <query> | compact: token-budgeted corpus-compaction projection with drop manifest  
  Lane: `read-contracts`. Readiness: `B-local-inspection-needed`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.2` P2 feature — Projection predicates/windows + render/layout stages on attached units  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.4` P2 feature — Shell completion + fuzzy selection as read-only projections of the grammar registries  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-fnm.6` P2 feature — Wire the terminal stage to projections: | read / | context-image  
  Lane: `read-contracts`. Readiness: `A-implementation-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-013x` P3 task — search_text excludes Write-tool file bodies (tool_input.$.content) — undocumented coverage gap  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-703` P3 task — One status assembly: daemon/status.py, cli/commands/status.py, and workload diagnostics converge  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-7le` P3 task — Consolidate the three session->HTML paths  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-1fp` P3 task — Facade decomposition: split api/archive.py into per-capability protocols  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-9jsi` P3 task — Polish search recall: pl_fold write/query symmetry + remove_diacritics 2 + measured trigram lane  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-9yz` P3 task — Named bounded-dialogue layout for operator-readable windows  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-jnj.10` P3 task — Make the completion system and DSL discoverable at point of use  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-jnj.11` P3 task — Extend the fzf pattern from select to ambiguous-result moments  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.
- `polylogue-jnj.12` P3 task — Empty-result guidance: 0 hits explains itself  
  Lane: `read-contracts`. Readiness: `D-horizon-ready`. Proof: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

  …plus 25 more; see CSV/JSON for the complete list.

## D-agent-context-coordination: Agent context, memory, and coordination (48 beads)

- `polylogue-d1y` P1 feature — polylogue hooks install: one-command harness wiring + hook liveness monitoring  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-pj8` P1 feature — Agent query cookbook: MCP prompts + skill recipes as the discoverability layer  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-s7ae.2` P1 task — Pre-deployment MCP and hook coordination batch  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-ahqd` P1 task — Observe MCP write adoption after role rollout  
  Lane: `agent-coordination`. Readiness: `B-local-inspection-needed`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-rsad` P2 bug — MCP agent ergonomics: oversized responses, boilerplate affordances, metadata-only summaries  
  Lane: `agent-coordination`. Readiness: `B-local-inspection-needed`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-tsk` P2 bug — Resume ranking keys on workflow shapes the classifier never emits  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-37t.14` P2 task — Recursive-safety substrate: citation anchors, provenance edges, grounding verdicts (closed-loop/cycle/drift)  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-6il` P2 task — devloop-integration --json --check consumed by devloop-review  
  Lane: `agent-substrate`. Readiness: `A-implementation-ready`. Proof: agent workflow catalog run and adoption telemetry report.
- `polylogue-lio` P2 task — Align cross-repo devloop contract on beads (Sinex parity)  
  Lane: `agent-substrate`. Readiness: `A-implementation-ready`. Proof: agent workflow catalog run and adoption telemetry report.
- `polylogue-t8t` P2 task — Agent workflow catalog: walk the seven core flows end-to-end, fix what breaks  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-x4s` P2 task — Express devloop state in Polylogue substrate (dogfood target)  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-37t.1` P2 feature — Assertions: consumer wiring + lifecycle tightening for unified overlays  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-37t.12` P2 feature — Judgment queue: operator bulk review/accept/reject of candidate assertions  
  Lane: `context-memory`. Readiness: `B-local-inspection-needed`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-37t.11` P1 feature — Context scheduler: one arbiter for everything that enters an agent's context  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-s7ae.3` P1 feature — Coordination messages and subtle scheduler-mediated advisories  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-s7ae.5` P1 task — Live proof: two agents, separate worktrees, one repo — overlap, message, context, handoff  
  Lane: `agent-coordination`. Readiness: `B-local-inspection-needed`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.4` P2 task — SessionStart preamble opt-in rollout (polylogue + sinnix repos)  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.2` P2 feature — Inline annotation protocol: agent-authored structure in plain prose  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-37t.3` P2 feature — Reboot-with-refs: session self-compaction protocol  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-37t.6` P2 feature — Session-aware devshell entry: surface what the last agent session left behind  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.7` P2 feature — Close the failure loop: verify postmortem -> next session's context seed  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.8` P2 feature — Resume routing: map a session to the harness invocation that reopens it  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-dmp` P2 feature — polylogue note: zero-friction memory capture from the terminal  
  Lane: `context-memory`. Readiness: `A-implementation-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-kph` P2 feature — Provenance-carrying PRs: attach the authoring session's postmortem bundle  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-p5g` P2 feature — polylogue judge: interactive candidate triage in the terminal  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-rii.1` P2 feature — Agent work-event write-leg -> session_events -> materialized read-models  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-rii.2` P2 feature — Materialize hook events + OTLP spans into queryable evidence  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-1hj` P3 task — Blackboard as agent comms: cross-session messages that actually arrive  
  Lane: `agent-coordination`. Readiness: `D-horizon-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-s7ae` P1 epic — Agent coordination substrate: evidence-backed multi-agent work without tracker lock-in  
  Lane: `agent-coordination`. Readiness: `A-implementation-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.13` P3 task — Revisit beads<->assertions boundary once beads-history ingestion (7fj) lands  
  Lane: `agent-coordination`. Readiness: `D-horizon-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.16` P3 task — Claim-kind -> allowed grounding-class compatibility registry  
  Lane: `agent-coordination`. Readiness: `D-horizon-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-37t.17` P3 task — Read-access log + memory-utility analytics: which injected memories earn their tokens  
  Lane: `context-memory`. Readiness: `D-horizon-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-rii.3` P3 task — Ingest fidelity: parser fingerprints, byte-fidelity bands, unparsed-key census, round-trip bar  
  Lane: `agent-coordination`. Readiness: `D-horizon-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.
- `polylogue-rvh` P3 task — Lesson reinforcement scheduling: judged memory on a forgetting curve  
  Lane: `context-memory`. Readiness: `D-horizon-ready`. Proof: context scheduler ledger fixture and candidate judgment queue proof.
- `polylogue-ze5` P3 task — Decision: user.db vocabulary — separate epistemic records from workspace state  
  Lane: `agent-coordination`. Readiness: `D-horizon-ready`. Proof: two-agent separate-worktree proof with before/after coordination envelopes.

  …plus 13 more; see CSV/JSON for the complete list.

## E-variants-preferences: Content variants, preferences, and transformed views (12 beads)

- `polylogue-0v9p` P1 feature — Language detection and preference facts for variant selection  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-arso` P1 feature — Content variant substrate: refs, nodes, alignment, storage  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-rlsb` P1 feature — Variant-aware projection, query, and reader render profiles  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-d4zk` P1 feature — User and agent UX for creating, reviewing, and messaging about variants  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-4smp` P1 epic — Content variants: language-aware transformed archive objects with alignment  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-at44` P3 bug — user_settings table is dead: DDL + migration 004 exist, zero runtime read/write helpers  
  Lane: `variants-preferences`. Readiness: `D-horizon-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-y4c` P3 task — Configuration doctrine: great defaults, DB-backed runtime prefs, Nix module parity  
  Lane: `variants-preferences`. Readiness: `D-horizon-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-6kh` P2 feature — Query-scope preferences bundle: default time window, scope filters, logical fold  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-1jc` P3 feature — Learned defaults: the archive proposes your configuration as judged candidates  
  Lane: `variants-preferences`. Readiness: `D-horizon-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-3xx` P3 feature — Verb-behavior and ops preferences bundle: confirmations, judge defaults, copy formats, spend, quiesce  
  Lane: `variants-preferences`. Readiness: `D-horizon-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-y8w` P3 feature — Reading preferences bundle: per-scope views, fold budgets, rows, pager  
  Lane: `variants-preferences`. Readiness: `D-horizon-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.
- `polylogue-w8db` P2 epic — Configuration doctrine + DB-backed runtime preferences  
  Lane: `variants-preferences`. Readiness: `A-implementation-ready`. Proof: mixed-language variant fixture with source/variant/alignment rendering in CLI/web/MCP.

## F-lineage-compaction: Lineage truth and compaction recovery (12 beads)

- `polylogue-4ts.3` P2 bug — Distinguish subagent auto-compaction from main-session acompact  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-4ts.4` P2 bug — Wrap lineage composition reads in a single read transaction  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-4ts.6` P2 bug — Lineage composition silently truncates transcripts; surface a completeness signal  
  Lane: `lineage-compaction`. Readiness: `B-local-inspection-needed`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-4ts.5` P2 feature — Compaction boundary-range columns + effective-context derivation  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-gjg.1` P2 task — compaction_events + compaction_loss_items derived tables; identity survives rebuild + re-ingest  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-gjg.2` P2 task — Pre-compaction snapshot capture: hook payload when available, manifest-of-refs otherwise, honesty ladder always  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-gjg.3` P2 task — Deterministic loss-forensics: 4-tier structural diff + lost-but-later-needed ranking  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-h6r` P2 task — Agent identity: a stable who-did-this tuple for every session  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-4ts.7` P3 task — Physical session identity collision beneath origin collapse: same native_id, two source families, one row  
  Lane: `lineage-compaction`. Readiness: `D-horizon-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-4ts` P1 epic — Session lineage truth: shared content stored once, counted once, composed correctly  
  Lane: `lineage-compaction`. Readiness: `B-local-inspection-needed`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-gjg.4` P3 task — compaction_forgot + compaction_reground surfaces; re-grounding packs survive the next compaction  
  Lane: `lineage-compaction`. Readiness: `D-horizon-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.
- `polylogue-gjg` P2 epic — Compaction lifecycle: pre-compaction snapshot, loss forensics, post-compaction re-grounding  
  Lane: `lineage-compaction`. Readiness: `A-implementation-ready`. Proof: branch/shared-prefix/compaction/truncation fixture matrix and regrounding proof.

## G-live-performance: Live intake, capture, daemon, and interactive performance (25 beads)

- `polylogue-20d.4` P2 bug — CLI structured-query routing parity with daemon (#1860): no FTS gate for non-FTS queries  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.10` P2 task — Runtime post-filter efficiency: memoize semantic facts; lower matchers onto the actions view  
  Lane: `live-substrate`. Readiness: `A-implementation-ready`. Proof: live-ingest fixture, event materialization proof, status/liveness report.
- `polylogue-20d.14` P2 task — Interactive SLO tier: named latency budgets, continuously measured, regression-gated  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.15` P2 task — Bulk ingest throughput + resource envelope: parallel parse, batched writes, bounded RSS/IO  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.2` P2 task — Defer heavy imports off the CLI startup path  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.5` P2 task — Finish streaming reads: composed transcripts, messages --full writer, origin-filtered pagination SQL  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.6` P2 task — Live full-ingest catch-up latency + WAL shape  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-3v1.1` P2 task — Multiple concurrent browser-capture extension instances: attribution, dedup, spool safety  
  Lane: `capture-reliability`. Readiness: `B-local-inspection-needed`. Proof: extension smoke, concurrent spool/dedup test, capture-gap event fixture.
- `polylogue-3v1` P2 task — Capture extension reliability + status UX: spool health, completeness, gap visibility  
  Lane: `capture-reliability`. Readiness: `A-implementation-ready`. Proof: extension smoke, concurrent spool/dedup test, capture-gap event fixture.
- `polylogue-th0` P2 task — Interactive-surface test harness: pty flows, completions, fuzzy pickers  
  Lane: `live-substrate`. Readiness: `A-implementation-ready`. Proof: live-ingest fixture, event materialization proof, status/liveness report.
- `polylogue-yeq` P2 task — Advanced verification lanes: metamorphic DSL, daemon chaos, API-contract walks  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.1` P2 feature — CLI->daemon fast path over UDS (persistent hot process)  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.12` P2 feature — Daemon result cache + post-ingest warming: precomputed answers, cursor-keyed invalidation  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.13` P2 feature — Daemon push channel: SSE events for live UIs instead of polling  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-90y` P2 feature — In-page overlay: Polylogue presence on chat sites — archive state, context, assertion capture  
  Lane: `capture-reliability`. Readiness: `A-implementation-ready`. Proof: extension smoke, concurrent spool/dedup test, capture-gap event fixture.
- `polylogue-opc` P2 feature — Self-tracing: the daemon's own spans land in its own archive  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-oxz` P2 feature — Performance instrumentation doctrine: slow-query log, phase timings, logging discipline  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-jlme` P2 epic — Capture extension: reliability, coverage, and in-page presence  
  Lane: `capture-reliability`. Readiness: `A-implementation-ready`. Proof: extension smoke, concurrent spool/dedup test, capture-gap event fixture.
- `polylogue-20d.11` P3 task — Read-profile mmap tuning: raise READ_MMAP, lower double-buffering cache  
  Lane: `interactive-performance`. Readiness: `D-horizon-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.16` P3 task — Performance/throughput scenario family  
  Lane: `interactive-performance`. Readiness: `D-horizon-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d.7` P3 task — EQP sweep + dbstat census on a live-archive copy  
  Lane: `live-substrate`. Readiness: `D-horizon-ready`. Proof: live-ingest fixture, event materialization proof, status/liveness report.
- `polylogue-20d.8` P3 task — Bound claim-vs-evidence regen latency (43s on live archive)  
  Lane: `interactive-performance`. Readiness: `D-horizon-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-20d` P2 epic — Interactive performance: the front door answers in interactive time  
  Lane: `interactive-performance`. Readiness: `A-implementation-ready`. Proof: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.
- `polylogue-5en` P3 task — Branch-local daemon/web/extension dev loops: verify remaining AC and close out  
  Lane: `capture-reliability`. Readiness: `D-horizon-ready`. Proof: extension smoke, concurrent spool/dedup test, capture-gap event fixture.
- `polylogue-dx1` P3 task — Decision: daemon HTTP substrate — hand-rolled BaseHTTPRequestHandler vs ASGI  
  Lane: `capture-reliability`. Readiness: `D-horizon-ready`. Proof: extension smoke, concurrent spool/dedup test, capture-gap event fixture.

## H-web-cockpit: Web evidence cockpit (18 beads)

- `polylogue-bby.11` P1 feature — Webui architecture v2: the stack that can carry the ambition  
  Lane: `web-evidence-cockpit`. Readiness: `A-implementation-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.15` P2 task — Evidence basket -> citable report -> verified export (cockpit core loop)  
  Lane: `web-evidence-cockpit`. Readiness: `A-implementation-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.8` P2 feature — Web reader perceived performance: virtualized list, streamed search, optimistic navigation  
  Lane: `web-evidence-cockpit`. Readiness: `A-implementation-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-ptx` P2 feature — Browser-capture posting channel: un-gate, with attachments  
  Lane: `web-evidence-cockpit`. Readiness: `A-implementation-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-yrx` P2 feature — Session changes view: per-session diff/changelog composed from edit evidence  
  Lane: `web-evidence-cockpit`. Readiness: `A-implementation-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.1` P3 task — Workbench responsive under slow/missing routes  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.10` P3 task — Timeline and firehose: the archive as a scrubbable stream  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.6` P3 chore — Interaction debt: replace window.prompt(); de-drift the JS renderer  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-30h` P3 feature — Display titles: synthesize when the stored title is a first-prompt echo  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-b1n` P3 feature — WebUI-driven posting: operator drives web chats from the workbench  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.12` P3 feature — Session replay: play a session back the way it happened  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.13` P3 feature — The day page: a daily narrative the operator actually reads  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.2` P3 feature — Query completions + expression explain in the web search box  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.3` P3 feature — Aggregate analytics views in the web UI  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.4` P3 feature — Live session tailing as a first-class mode  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.5` P3 feature — Long-session navigation: phases/windows/minimap  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby.14` P4 feature — Pinboard: workspaces as a spatial surface  
  Lane: `web-evidence-cockpit`. Readiness: `D-horizon-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.
- `polylogue-bby` P2 epic — Web workbench: from result list to evidence cockpit  
  Lane: `web-evidence-cockpit`. Readiness: `A-implementation-ready`. Proof: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## I-analytics-experiments: Analytics, experiments, and measured learning (30 beads)

- `polylogue-1vpm.1` P2 task — Delegation derived unit: materializer + query unit + delegation-card projection  
  Lane: `analytics-experiments`. Readiness: `B-local-inspection-needed`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-1vpm.2` P2 task — Episode unit: tables, 4-signal scorer with false-merge floor, assertion-calibrated  
  Lane: `analytics-experiments`. Readiness: `B-local-inspection-needed`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-1vpm.4` P2 task — Turn-pair unit with prompt-burst semantics (no double-claimed answers)  
  Lane: `analytics-experiments`. Readiness: `B-local-inspection-needed`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-3uw` P2 task — Capture-completeness: the instrument's coverage error as a standing measure  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.15` P2 task — Triage frontier: worth_reviewing_score + TRIAGED lifecycle — an inbox that empties  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.1` P2 feature — Outcome-conditioned analytics: cost/duration/retries/tools by structural success  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.2` P2 feature — Cross-provider comparative analytics  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.6` P2 feature — tool-episodes projection: call + result + outcome + context + next action  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-1vpm.3` P3 task — Generic artifact edges: produced/consumed/mentioned/reported_by/derived_from across sessions, runs, delegations  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-1vpm.5` P3 task — Correction-edge runtime query: resolve correction assertions to corrected blocks/tools/models  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-1vpm` P2 epic — Work-graph units: delegation, episode, artifact edges — the derived units between lineage and analysis  
  Lane: `analytics-experiments`. Readiness: `B-local-inspection-needed`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.19` P3 task — Thinking-vs-doing drift: experimental coverage-gated measure of reasoning share vs tool-active share  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.13` P2 task — activity_spans materializer: edit/test/build/idle/delegate intervals with evidence tiers  
  Lane: `analytics-experiments`. Readiness: `B-local-inspection-needed`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.14` P3 task — Efficiency measure pack v1: scorecard vector over spans/episodes/delegations — no magic score  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.7.1` P3 task — Tag Layer-0 substrate insight payloads with evidence_tier so consumers can read rule-heuristic vs structural confidence  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.7` P2 feature — Statistics substrate + measure registry: uncertainty primitives with construct-validity metadata  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.8` P2 feature — Temporal analytics: trends, rolling baselines, changepoint detection  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-stc` P2 feature — Experiment hosting: declared arms, preregistered metrics, paired analysis, agent-buildable  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-h10` P3 task — Prediction and calibration tracking: agents scored on what they said would happen  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.10` P3 feature — Process mining: workflow motifs, transition models, bottleneck discovery  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.11` P3 feature — Predictive advisories: calibrated classical models on structural labels  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.12` P3 feature — Information-theoretic and graph measures: redundancy, diversity, tree shapes  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.3` P3 feature — Pathology epidemiology: corpus-level rates and trends  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.4` P3 feature — Token-economy analytics: cache-lane and attention accounting  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.5` P3 feature — Ship opinionated saved views as product defaults  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.9` P3 feature — Survival analysis: session duration, abandonment hazard, time-to-outcome  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.18` P3 epic — Missing data-model units: entity-mention, world-effect, verification-run, project, topic-cluster, cross-origin-thread  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.16` P4 task — Trajectory Quality Index: reward-shaping composite, never truth  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5.17` P4 task — Model-drift observatory: candidate changepoints with validity gates, never causal claims  
  Lane: `analytics-experiments`. Readiness: `D-horizon-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.
- `polylogue-9l5` P2 epic — Outcome-grounded analytics: the archive answers 'so what' questions  
  Lane: `analytics-experiments`. Readiness: `A-implementation-ready`. Proof: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.

## J-embeddings-retrieval: Embeddings and semantic retrieval (11 beads)

- `polylogue-0k6` P2 task — Embedding changed-text full-replace regression vs split embeddings.db metadata  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-0ns` P2 task — Bound archive embedding work within large sessions  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.3` P2 task — Retrieval quality eval lane: measure FTS vs vector vs hybrid before believing any of them  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.1` P2 feature — Provider abstraction: one OpenAI-compatible embedding client, local and cloud, model registry in meta  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-37t.5` P2 feature — Local embedding lane via OpenAI-compatible provider (LiteLLM gateway)  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.2` P2 feature — Embedding target policy: what gets a vector, at what granularity, at what cost  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.4` P2 feature — Semantic recall leg in context compilation: the memory actually retrieves  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.7` P3 bug — Two live vec0 DDL definitions: unify to one canonical embeddings table-creation path  
  Lane: `embeddings-retrieval`. Readiness: `D-horizon-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.6` P3 task — Embedding storage/spend efficiency: quantization, matryoshka, and scoped drain  
  Lane: `embeddings-retrieval`. Readiness: `D-horizon-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx.5` P3 feature — Semantic analytics surfaces: topics/clustering, novelty, near-duplicate assist  
  Lane: `embeddings-retrieval`. Readiness: `D-horizon-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.
- `polylogue-mhx` P2 epic — Embedding substrate: provider-general, honest lifecycle, retrieval that earns its cost  
  Lane: `embeddings-retrieval`. Readiness: `A-implementation-ready`. Proof: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.

## K-interop-origin-export: Interop, origin breadth, and export (31 beads)

- `polylogue-ox0` P2 task — Codex deep integration: state DBs as authoritative source + AppServer live lane  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-t0p` P2 task — Capture the rest of Claude Code: todos, file-history, prompt history, debug artifacts  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-7aw` P2 feature — Ingest agent configuration as a source family (skills, CLAUDE.md, hooks)  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-7xv.1` P2 feature — Work-trace + reproduction harness: verify a session repo-work from a clean worktree  
  Lane: `origin-interop-export`. Readiness: `B-local-inspection-needed`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-da1` P2 feature — Provider format-drift sentinel: detect upstream export-shape changes from live ingest  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.4` P2 feature — Report: polylogue forensics for Hermes sessions  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-2qx` P2 feature — OriginSpec: one package per origin, dispatch order derived from declared strictness  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-tf0e` P3 bug — Generic-messages parser fallback drops available created_at/updated_at  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-0dz` P3 task — Chunked/streaming read-package layout for huge exports  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-7xv` P3 feature — Native git/repo awareness: session-to-commit/branch/repo correlation in Polylogue  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.10` P3 feature — Spec-cards: sessions as portable benchmark items (leakage-gated export)  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.2` P3 feature — Importer: NeMo Relay ATOF/ATIF runtime spans  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.3` P3 feature — Per-source coverage/fidelity declaration for Hermes imports  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.5` P3 feature — Export: Atropos/eval JSONL downstream of the canonical archive  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.6` P3 feature — Fully-sovereign loop demo: local Hermes -> archive -> local embeddings -> judged memory -> injection, air-gapped  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.7` P3 feature — Upstream native integration: lifecycle hooks / polylogue-hook support in the open-source Hermes agent  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.9` P3 feature — Polylogue->Sinex derived agent-trace event emitter  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-l4kf.1` P3 feature — polylogue-export origin + CIF envelope: import(export(A)) is a content-hash no-op  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-uiw` P3 feature — Origin breadth: enumerate the target set + generic openai-chat-shape detector  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-wmj` P3 feature — OTel GenAI trace export lane  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-0cg` P3 feature — OTel GenAI semantic-conventions ingest: any instrumented agent framework becomes an origin  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1.8` P4 task — Nous Chat browser-capture adapter  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-fs1` P2 epic — Hermes bridge: state.db + runtime spans -> canonical evidence -> forensics/eval export  
  Lane: `origin-interop-export`. Readiness: `A-implementation-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-4g5` P4 feature — Expose the archive as an HPI module and Promnesia source  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-611` P4 feature — Grok (xAI) conversation export importer  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-7k7` P4 feature — Research-tooling export lane: inspect-ai / Docent formats  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-ale` P4 feature — External link archival: sessions cite URLs; the evidence should not rot  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-l4kf.2` P4 feature — Federation: .well-known/ai-sessions manifest + selective content-hash sync  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-l4kf.3` P4 feature — Outbound provenance: git notes (refs/notes/polylogue) + PR/issue citation footers + SARIF pathology export  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-r47` P4 feature — Obsidian/PKM export profile: sessions and findings as wiki-linked Markdown  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.
- `polylogue-l4kf` P3 epic — Ecosystem interop + origin breadth: more sources in, two-way citable export out  
  Lane: `origin-interop-export`. Readiness: `D-horizon-ready`. Proof: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## L-external-legibility: External legibility, demos, and launch (29 beads)

- `polylogue-cfk` P1 task — Re-run two-arm uplift with freshness-fixed packs (n>=3 pairs, then n=12-20)  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.1` P2 task — Post-hoc forensic Q&A demo: questions a tracer cannot answer  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.2` P2 task — D1 'The receipts': claim-vs-evidence on a real PR  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.3` P2 task — D2 'Where did the money actually go': cost by outcome  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.4` P2 task — D4 'Behavioral archaeology': six DSL queries, rapid fire  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.8` P2 task — The honesty anti-demo: a tempting finding that emits verdict not_supported  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.12` P2 task — README de-meta / de-persuasion pass with reproducible capability claims  
  Lane: `docs-demos-launch`. Readiness: `B-local-inspection-needed`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.13` P2 task — Reconcile schema-versioning docs + retire superseded execution-plan.md  
  Lane: `docs-demos-launch`. Readiness: `B-local-inspection-needed`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.16` P2 task — Public claims ledger: every README/launch claim carries a status and an evidence ref  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.4` P2 task — Findings publishing lane: campaign artifacts on the docs site  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.7` P2 task — Release is a decision: proven install matrix across package managers and OSes  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.9` P2 task — Docs-and-visuals ownership: coverage lint + regenerable visuals as a standing devloop gate  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.5` P3 task — D5 'The session that watched itself': live capture proof  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.6` P3 task — D8 'Pick up where I left off': abandoned-session triage to live continuation  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.9` P3 task — Fable-as-Foreman: subagent-delegation rhetoric report (the X-post demo)  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212.7` P2 task — Demo Finding Packet contract + prompt runner + registry manifest  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-212` P2 epic — Demo portfolio: construct-valid demos (D1/D2/D4/D5/D8 + post-hoc forensic Q&A)  
  Lane: `docs-demos-launch`. Readiness: `A-implementation-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.10` P3 task — Launch kit: announcement artifacts prepared so publication is one decision  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.14` P3 task — Fix FTS + entry-point doc drift: internals.md describes external-content FTS; CLAUDE.md misdescribes operations/archive.py  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.15` P3 task — Anti-grep proof card: the "why not grep ~/.claude" answer, grounded in one finding  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.3` P3 task — Claim-vs-evidence leaderboard variant (multi-model, incl. open models)  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.8` P3 task — GitHub surface polish: the repo page itself is a landing page  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-ttu` P3 task — Docs information architecture: tiered index, orphan sweep, stale-doc triage  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-6bu` P3 chore — Docs-site verification lane (pages cache, link integrity)  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-6l6` P3 chore — Docs/theming/release-proof/control-plane polish  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl.6` P3 feature — Publish the normalized session model as a versioned interchange schema  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-45i` P3 feature — Datasette lane: the archive as an explorable SQLite exhibit  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-y0b` P3 feature — Generated codebase atlas: the grok report as a rendered, drift-checked doc  
  Lane: `docs-demos-launch`. Readiness: `D-horizon-ready`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.
- `polylogue-3tl` P1 epic — External legibility: a stranger can understand, run, and cite Polylogue  
  Lane: `docs-demos-launch`. Readiness: `B-local-inspection-needed`. Proof: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## M-substrate-consolidation: Substrate consolidation and codebase simplification (33 beads)

- `polylogue-a7xr.1` P2 bug — Sweep remaining sqlite3 connection leaks: 'with sqlite3.connect()' commits but never closes  
  Lane: `substrate-consolidation`. Readiness: `B-local-inspection-needed`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.2` P2 bug — Converger and repair disagree on session_profile staleness for NULL-sort-key sessions  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.3` P2 bug — message_type_backfill reconstructs prose unordered and unfiltered; message-prose SQL exists 5x  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.6` P2 bug — parse_archive_datetime: 6 copies, one with different tz semantics (naive/aware time bomb)  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.5` P2 task — FTS trigger DDL declared twice: archive_tiers/index.py vs fts_lifecycle repair copies  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-exb` P2 task — Layering: substrate rings import the api facade (6 sites, 2 private-symbol reaches)  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.10` P2 chore — Kill-or-adopt the search-provider lane: production bypasses the abstraction it should use  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.11` P2 chore — Prune protocols.py zero-consumer protocols + dead repo kwarg query surface + cursor mapping bug  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-o21` P2 feature — Extension-point ergonomics: declare-once registries, scaffolds, actionable completeness errors  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.13` P3 task — api/contracts write-surface shadow adapters verify copies, not surfaces — delete or re-anchor  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.16` P3 task — Table-drive the hand-aligned column triplicates in archive_tiers write/read hot core  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.4` P3 task — One percentile implementation: three algorithms across five copies skew operator-facing stats  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.7` P3 task — Role synonym vocabulary maintained by hand in two directions + normalize_role name collision  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.8` P3 task — Index-tier sibling-path derivation pasted ~7x with divergent existence rules  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-c9y` P3 task — Package topology legibility: boundary doctrine for the 28-package tree + insights/analytics vocabulary  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-dab.1` P3 task — Drop payload_json/search_text duplication from run-projection tables; hydrate from typed columns  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-dab` P3 task — Stop materializing run-projection cache rows; drop DDL after parity  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-f94` P3 task — Kill-or-commit the TUI (~373 lines of skeletal Textual screens)  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-pf1` P3 task — Sync/async divergence: diff the twin backends against the '10 known divergences' list  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-hiu` P2 task — Collapse storage twins onto the sync core behind an async adapter boundary  
  Lane: `substrate-consolidation`. Readiness: `A-implementation-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-utf` P3 task — Devtools surface economy: usage-ranked consolidation of the 67-command catalog  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-1a9` P3 chore — Remove dead session-commit stubs + unused web-construct row + stale fuzz README  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-48h` P3 chore — Consolidate SQLite introspection helpers (10 copies of _table_exists and friends)  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-5dx` P3 chore — Dependency leverage policy: [analytics]/[ml] extras, evaluated adoptions  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.12` P3 chore — neighbor_candidates needs a 4-method protocol, not the 20-method SessionQueryRuntimeStore  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.14` P3 chore — Collapse the one-operation operations-contract framework to concrete Import models  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.15` P3 chore — payloads.py: generic from_row for the 74 identical-name copy lines (keeps typed wire contract)  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr.9` P3 chore — Mechanical helper dedup sweep: scalar coercion quadruplet, _table_exists x40, provenance vocab x6, title/tags mixin  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-0aj` P3 feature — Declared write-effects chain: post-commit effects as registry entries  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-4822` P3 feature — Curated polylogue.sdk + frozen public models: the external-consumer boundary lynchpin needs  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-yp0` P3 feature — Daemon internal event bus: loops subscribe, polling retires  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-a7xr` P3 epic — Substrate consolidation: kill the storage twins and split the god-modules  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.
- `polylogue-wohv` P4 task — messages_fts UNINDEXED columns are write-only noise in a contentless table: drop or annotate  
  Lane: `substrate-consolidation`. Readiness: `D-horizon-ready`. Proof: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## N-horizon: Horizon and vision work (6 beads)

- `polylogue-ca4` P3 task — Decision: DuckDB as the optional OLAP engine over the archive  
  Lane: `horizon-spec`. Readiness: `D-horizon-ready`. Proof: decision memo or execution-grade spec with explicit pull-forward gate.
- `polylogue-2n6` P3 feature — Harness remote-control lane: drive Claude Code / Codex sessions from Polylogue surfaces  
  Lane: `horizon-spec`. Readiness: `D-horizon-ready`. Proof: decision memo or execution-grade spec with explicit pull-forward gate.
- `polylogue-2jj` P4 task — IssueBench: real issues as coding-agent effectiveness benchmarks  
  Lane: `horizon-spec`. Readiness: `D-horizon-ready`. Proof: decision memo or execution-grade spec with explicit pull-forward gate.
- `polylogue-c36` P4 task — Native-compilation probe: mypyc first, only where profiles demand it  
  Lane: `horizon-spec`. Readiness: `D-horizon-ready`. Proof: decision memo or execution-grade spec with explicit pull-forward gate.
- `polylogue-gqx` P4 task — Desktop presence spike: Polylogue in the operator's ambient environment  
  Lane: `horizon-spec`. Readiness: `D-horizon-ready`. Proof: decision memo or execution-grade spec with explicit pull-forward gate.
- `polylogue-lu1` P4 task — Ambient theming: terminal respects the environment, webui gains a theme system  
  Lane: `horizon-spec`. Readiness: `D-horizon-ready`. Proof: decision memo or execution-grade spec with explicit pull-forward gate.

