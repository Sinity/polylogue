# Release gates

### R0-normalize: Backlog normalization and readiness lint

Purpose: Make Beads itself delivery-grade: consistent ready counts, release/lane labels, missing acceptance criteria patched, and hidden blockers explicit.

Entry: Exported Beads snapshot available.

Exit: Every active bead has a delivery release, lane, readiness grade, proof lane, and either acceptance criteria or a deliberate horizon/spec status.

Active beads assigned: 0.

### A-trust-floor: Trust floor

Purpose: Prevent false evidence, unsafe byte deletion, unsafe daemon/capture access, and unsafe agent-written memory.

Entry: Backlog normalization generated and reviewed.

Exit: Full verification classified; security negative tests pass; missing bytes classified; numbers/time/prose-mined fields carry honest provenance; agent writes land as candidates.

Active beads assigned: 59.

### B-storage-rebuild-bytes: Storage, rebuild, and byte-integrity floor

Purpose: Make durable/derived storage and blob handling safe at real archive scale.

Entry: Trust-floor blob/security blockers either closed or explicitly isolated.

Exit: Blue-green derived rebuilds cannot show partial archives as ready; restore drill passes; blob refs resolve or carry classified missing state.

Active beads assigned: 23.

### C-read-evidence-contract: One read contract and first-class evidence objects

Purpose: Unify query/projection/render semantics and make query runs, result sets, findings, and citations addressable.

Entry: Trust-floor evidence-honesty bugs closed.

Exit: CLI, daemon, MCP, Python API, web, reports, and docs read through the same contract; content-hash citations expose drift states.

Active beads assigned: 60.

### D-agent-context-coordination: Agent context, memory, and coordination

Purpose: Make Polylogue useful while agents are working, with safe memory boundaries and scheduler-mediated context.

Entry: Daemon security and agent-write safety closed.

Exit: Hooks install; MCP roles/prompts discoverable; context scheduler emits ledgers; two-agent worktree proof exists.

Active beads assigned: 48.

### E-variants-preferences: Content variants, preferences, and transformed views

Purpose: Support translations, simplifications, summaries, and preferences without confusing transformed text with original evidence.

Entry: Evidence refs and agent-write safety available for source alignment and candidate review.

Exit: Variant refs/nodes/alignment/storage exist; reader/query/projection can show source, variant, and side-by-side views honestly.

Active beads assigned: 12.

### F-lineage-compaction: Lineage truth and compaction recovery

Purpose: Represent physical/logical sessions, shared prefixes, subagents, compactions, truncation, and regrounding truthfully.

Entry: Origin identity and read consistency contracts available.

Exit: Shared content is stored/counted once, compaction loss is queryable, and regrounding packs pass through the context scheduler.

Active beads assigned: 12.

### G-live-performance: Live intake, capture, daemon, and interactive performance

Purpose: Make capture/live surfaces observable, fast, bounded, and testable.

Entry: Security/capture safety controls landed.

Exit: Named SLOs and regression gates exist; daemon push/live cache paths work; capture reliability is visible and tested.

Active beads assigned: 25.

### H-web-cockpit: Web evidence cockpit

Purpose: Turn the web UI into an evidence workbench: baskets, reports, replay, timelines, live tail, long-session navigation.

Entry: Read/evidence contract and daemon push state available.

Exit: Evidence basket to citable export works; web UI shows stale/partial/degraded states instead of pretending readiness.

Active beads assigned: 18.

### I-analytics-experiments: Analytics, experiments, and measured learning

Purpose: Answer outcome/cost/process questions with sample frames, uncertainty, and construct-validity limits.

Entry: Evidence honesty, query identity, and measure registry foundations exist.

Exit: Measures are registered, experiments are first-class, analytics render caveats and evidence tiers.

Active beads assigned: 30.

### J-embeddings-retrieval: Embeddings and semantic retrieval

Purpose: Make vector search provider-general, budgeted, measurable, and useful for context retrieval.

Entry: Query contracts and context scheduler exist.

Exit: FTS/vector/hybrid quality evals exist; local/cloud providers share one interface; large sessions bound vector work.

Active beads assigned: 11.

### K-interop-origin-export: Interop, origin breadth, and export

Purpose: Import more origins through declared contracts and export citable evidence back out.

Entry: OriginSpec and evidence refs exist.

Exit: Each origin has detector/parser/fixtures/fidelity docs; Polylogue export/import is content-hash idempotent; outbound citations preserve provenance.

Active beads assigned: 31.

### L-external-legibility: External legibility, demos, and launch

Purpose: Make a stranger understand, run, verify, and cite Polylogue.

Entry: Claims ledger, demos, install proof, and evidence report substrate available.

Exit: README first screen is clear; one-command demo passes; public claims ledger covers launch claims; cold-reader proof passes.

Active beads assigned: 29.

### M-substrate-consolidation: Substrate consolidation and codebase simplification

Purpose: Remove duplicate storage/read/surface code after the contracts they should collapse onto are stable.

Entry: Read/evidence and storage/rebuild contracts stable enough to refactor around.

Exit: Storage twins collapse behind a clear boundary; public models are frozen; dead abstractions are deleted or adopted.

Active beads assigned: 33.

### N-horizon: Horizon and vision work

Purpose: Keep speculative or late-gated work visible without letting it distort the safety and proof spine.

Entry: Earlier foundation releases stable, or a decision memo explicitly pulls one item forward.

Exit: Each item either has an implementation-grade spec or remains parked with a decision record.

Active beads assigned: 6.

# Global rules baked into the setup


1. A bead is ready only when it has no active hard blocker, has acceptance criteria, has enough design/code-locality information to start, does not depend on an unmodeled product contract, and is PR-shaped.
2. Every number is a claim: counts, percentages, token totals, costs, latencies, and scores must carry denominators, evidence source, and absence semantics.
3. Every time has a source: provider time, capture time, file time, fallback time, synthetic time, and inferred time are different.
4. Agent-authored facts are candidates until judged by the operator/user.
5. Browser and daemon routes are hostile until Host, Origin, token, auth, and spool controls prove otherwise.
6. New origins are contracts, not scripts: detector, strictness, parser, fixtures, fidelity notes, and deterministic ambiguous dispatch are required.
7. Public claims need evidence refs or must be marked capability-only, unmeasured, stale, sampled, unknown, or not supported.
8. Every user-facing read path goes through the same Query × Projection × Render contract.
