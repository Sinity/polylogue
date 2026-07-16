# Proposed acceptance criteria fills

These are proposed patches for active beads whose `acceptance_criteria` field was empty in the source export. P2 entries are implementation-gate patches; P3/P4 entries are horizon/spec-gate patches unless their release owner pulls them forward.

## `polylogue-rsad` — MCP agent ergonomics: oversized responses, boilerplate affordances, metadata-only summaries

Priority: P2. Type: bug. Release: `D-agent-context-coordination`. Lane: `agent-coordination`. Confidence: `high`.

Oversized MCP responses are refused or summarized without flooding agent context: responses above the configured budget return metadata-only summaries, continuation handles, and explicit open/read-next affordances. Boilerplate affordances are removed from normal-size responses. Fixtures prove large result sets stay under budget, continuation handles resolve to the intended evidence, and the user-visible guidance names the next safe action.

## `polylogue-2qx` — OriginSpec: one package per origin, dispatch order derived from declared strictness

Priority: P2. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `high`.

OriginSpec is the sole origin-dispatch contract. Each origin declares detector, strictness, parser entry point, raw fixture, normalized fixture, schema mapping, and fidelity/completeness notes. Dispatch order is deterministic and tested against ambiguous fixtures. Adding a new origin without an OriginSpec fails a registry completeness check.

## `polylogue-37t.5` — Local embedding lane via OpenAI-compatible provider (LiteLLM gateway)

Priority: P2. Type: feature. Release: `J-embeddings-retrieval`. Lane: `embeddings-retrieval`. Confidence: `high`.

A local OpenAI-compatible embedding provider works through the same provider abstraction as cloud providers. Context retrieval can opt into the local provider without changing callers. Secrets and provider URLs are not logged. Interface-level fixtures prove local/cloud parity for request shape, error handling, disabled-provider behavior, and retrieval metadata.

## `polylogue-rii.2` — Materialize hook events + OTLP spans into queryable evidence

Priority: P2. Type: feature. Release: `D-agent-context-coordination`. Lane: `agent-coordination`. Confidence: `high`.

Hook events and OTLP spans materialize into queryable evidence tables with stable object refs, idempotent replay, parser fingerprints, and fixture coverage. Replaying the same input does not duplicate rows. Query surfaces can select the materialized events by session, repo/worktree, time, and evidence tier.

## `polylogue-fs1.4` — Report: polylogue forensics for Hermes sessions

Priority: P2. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `high`.

A Hermes forensic report regenerates from imported Hermes sessions and emits citable findings with coverage/fidelity caveats, raw evidence refs, and a single documented regeneration command. The report includes at least one happy-path fixture, one missing-field/degraded fixture, and one fidelity limitation that renders visibly instead of silently disappearing.

## `polylogue-scd` — Cross-surface handoff: polylogue open + copy-as-command everywhere

Priority: P3. Type: feature. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-scd` has an execution-grade design note before coding, lands behind the release gate `C-read-evidence-contract`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-3tl.10` — Launch kit: announcement artifacts prepared so publication is one decision

Priority: P3. Type: task. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-3tl.10` updates the public artifact and links every factual product claim to the claims ledger or marks it as capability-only/not-yet-measured. Docs/link verification and a cold-reader or demo-regeneration check cover the change. Verification artifact: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-3tl.8` — GitHub surface polish: the repo page itself is a landing page

Priority: P3. Type: task. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-3tl.8` updates the public artifact and links every factual product claim to the claims ledger or marks it as capability-only/not-yet-measured. Docs/link verification and a cold-reader or demo-regeneration check cover the change. Verification artifact: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-bfv` — Advisory hooks: archive-informed PreToolUse/UserPromptSubmit responses

Priority: P3. Type: feature. Release: `D-agent-context-coordination`. Lane: `agent-coordination`. Confidence: `medium`.

`polylogue-bfv` routes agent-authored material through candidate/judgment policy and scheduler-mediated context assembly. A ledger fixture shows included/excluded context with reasons, trust class, and budget. Rejected or stale material is not injected. Verification artifact: two-agent separate-worktree proof with before/after coordination envelopes.

## `polylogue-0aj` — Declared write-effects chain: post-commit effects as registry entries

Priority: P3. Type: feature. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-0aj` has an execution-grade design note before coding, lands behind the release gate `M-substrate-consolidation`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-yp0` — Daemon internal event bus: loops subscribe, polling retires

Priority: P3. Type: feature. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-yp0` declares a before/after measurement, an acceptable resource envelope, and a regression guard. The implementation fails loudly on stale/partial state and records phase timing where relevant. Verification artifact: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-48h` — Consolidate SQLite introspection helpers (10 copies of _table_exists and friends)

Priority: P3. Type: chore. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-48h` includes a before/after ownership map, preserves public behavior through parity tests, and deletes or redirects the old path with compatibility notes where needed. The refactor does not change evidence semantics unless a migration and release note say so. Verification artifact: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-703` — One status assembly: daemon/status.py, cli/commands/status.py, and workload diagnostics converge

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-703` declares a before/after measurement, an acceptable resource envelope, and a regression guard. The implementation fails loudly on stale/partial state and records phase timing where relevant. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-1fp` — Facade decomposition: split api/archive.py into per-capability protocols

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-1fp` includes a before/after ownership map, preserves public behavior through parity tests, and deletes or redirects the old path with compatibility notes where needed. The refactor does not change evidence semantics unless a migration and release note say so. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-dx1` — Decision: daemon HTTP substrate — hand-rolled BaseHTTPRequestHandler vs ASGI

Priority: P3. Type: task. Release: `G-live-performance`. Lane: `capture-reliability`. Confidence: `medium`.

A decision record for `polylogue-dx1` names the options considered, the chosen path, explicit non-goals, migration/rollback impact, and the release gate it affects. A minimal probe or code-reading appendix supports the decision. No product implementation ships under this bead until the decision record is linked from the relevant follow-up beads and `extension smoke, concurrent spool/dedup test, capture-gap event fixture` is updated if the decision changes a verification lane.

## `polylogue-mhx.6` — Embedding storage/spend efficiency: quantization, matryoshka, and scoped drain

Priority: P3. Type: task. Release: `J-embeddings-retrieval`. Lane: `embeddings-retrieval`. Confidence: `medium`.

`polylogue-mhx.6` runs through the provider-general embedding interface, has disabled-provider behavior, bounds work on large sessions, and records retrieval reason/evidence metadata. Quality is compared against FTS or a no-vector baseline before product claims are made. Verification artifact: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.

## `polylogue-mhx.5` — Semantic analytics surfaces: topics/clustering, novelty, near-duplicate assist

Priority: P3. Type: feature. Release: `J-embeddings-retrieval`. Lane: `embeddings-retrieval`. Confidence: `medium`.

`polylogue-mhx.5` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: FTS/vector/hybrid retrieval eval, provider abstraction tests, bounded-vector-work fixture.

## `polylogue-45i` — Datasette lane: the archive as an explorable SQLite exhibit

Priority: P3. Type: feature. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-45i` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-212.6` — D8 'Pick up where I left off': abandoned-session triage to live continuation

Priority: P3. Type: task. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-212.6` has an execution-grade design note before coding, lands behind the release gate `L-external-legibility`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-37t.10` — Setup evolution via judged candidates: hooks/context-specs/cookbook changes proposed as evidence-linked assertions

Priority: P3. Type: feature. Release: `D-agent-context-coordination`. Lane: `agent-coordination`. Confidence: `medium`.

`polylogue-37t.10` routes agent-authored material through candidate/judgment policy and scheduler-mediated context assembly. A ledger fixture shows included/excluded context with reasons, trust class, and budget. Rejected or stale material is not injected. Verification artifact: two-agent separate-worktree proof with before/after coordination envelopes.

## `polylogue-30h` — Display titles: synthesize when the stored title is a first-prompt echo

Priority: P3. Type: feature. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

`polylogue-30h` has an execution-grade design note before coding, lands behind the release gate `H-web-cockpit`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-utf` — Devtools surface economy: usage-ranked consolidation of the 67-command catalog

Priority: P3. Type: task. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-utf` includes a before/after ownership map, preserves public behavior through parity tests, and deletes or redirects the old path with compatibility notes where needed. The refactor does not change evidence semantics unless a migration and release note say so. Verification artifact: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-fs1.7` — Upstream native integration: lifecycle hooks / polylogue-hook support in the open-source Hermes agent

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-fs1.7` adds or updates an origin contract with detector, parser, raw fixture, normalized fixture, parser fingerprint, and fidelity/completeness notes. Ambiguous inputs are handled deterministically. The regression suite proves idempotent replay and visible degraded/missing-field behavior. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-fs1.6` — Fully-sovereign loop demo: local Hermes -> archive -> local embeddings -> judged memory -> injection, air-gapped

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-fs1.6` adds or updates an origin contract with detector, parser, raw fixture, normalized fixture, parser fingerprint, and fidelity/completeness notes. Ambiguous inputs are handled deterministically. The regression suite proves idempotent replay and visible degraded/missing-field behavior. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-2n6` — Harness remote-control lane: drive Claude Code / Codex sessions from Polylogue surfaces

Priority: P3. Type: feature. Release: `N-horizon`. Lane: `horizon-spec`. Confidence: `medium`.

`polylogue-2n6` has an execution-grade design note before coding, lands behind the release gate `N-horizon`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via decision memo or execution-grade spec with explicit pull-forward gate.

## `polylogue-b1n` — WebUI-driven posting: operator drives web chats from the workbench

Priority: P3. Type: feature. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-b1n` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-7xv` — Native git/repo awareness: session-to-commit/branch/repo correlation in Polylogue

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-7xv` has an execution-grade design note before coding, lands behind the release gate `K-interop-origin-export`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-y0b` — Generated codebase atlas: the grok report as a rendered, drift-checked doc

Priority: P3. Type: feature. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-y0b` updates the public artifact and links every factual product claim to the claims ledger or marks it as capability-only/not-yet-measured. Docs/link verification and a cold-reader or demo-regeneration check cover the change. Verification artifact: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-0cg` — OTel GenAI semantic-conventions ingest: any instrumented agent framework becomes an origin

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-0cg` adds or updates an origin contract with detector, parser, raw fixture, normalized fixture, parser fingerprint, and fidelity/completeness notes. Ambiguous inputs are handled deterministically. The regression suite proves idempotent replay and visible degraded/missing-field behavior. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-3tl.6` — Publish the normalized session model as a versioned interchange schema

Priority: P3. Type: feature. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-3tl.6` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-fie` — Decision: archive scaling doctrine — keep everything, optimize the ceilings

Priority: P3. Type: task. Release: `B-storage-rebuild-bytes`. Lane: `blob-integrity`. Confidence: `medium`.

A decision record for `polylogue-fie` names the options considered, the chosen path, explicit non-goals, migration/rollback impact, and the release gate it affects. A minimal probe or code-reading appendix supports the decision. No product implementation ships under this bead until the decision record is linked from the relevant follow-up beads and `leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof` is updated if the decision changes a verification lane.

## `polylogue-uiw` — Origin breadth: enumerate the target set + generic openai-chat-shape detector

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-uiw` adds or updates an origin contract with detector, parser, raw fixture, normalized fixture, parser fingerprint, and fidelity/completeness notes. Ambiguous inputs are handled deterministically. The regression suite proves idempotent replay and visible degraded/missing-field behavior. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-37t.9` — Agent self-experimentation rail: PROMPT_EVAL writer + context-spec variation + background candidate passes

Priority: P3. Type: feature. Release: `D-agent-context-coordination`. Lane: `context-memory`. Confidence: `medium`.

`polylogue-37t.9` routes agent-authored material through candidate/judgment policy and scheduler-mediated context assembly. A ledger fixture shows included/excluded context with reasons, trust class, and budget. Rejected or stale material is not injected. Verification artifact: context scheduler ledger fixture and candidate judgment queue proof.

## `polylogue-20d.11` — Read-profile mmap tuning: raise READ_MMAP, lower double-buffering cache

Priority: P3. Type: task. Release: `G-live-performance`. Lane: `interactive-performance`. Confidence: `medium`.

`polylogue-20d.11` declares a before/after measurement, an acceptable resource envelope, and a regression guard. The implementation fails loudly on stale/partial state and records phase timing where relevant. Verification artifact: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.

## `polylogue-9e5.9` — Heuristic accuracy benchmark: keyword classifiers vs hand-labeled truth

Priority: P3. Type: task. Release: `A-trust-floor`. Lane: `evidence-honesty`. Confidence: `medium`.

`polylogue-9e5.9` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: rigor-audit coverage report, evidence-contract tests, not-supported/unknown rendering fixture.

## `polylogue-9yz` — Named bounded-dialogue layout for operator-readable windows

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-9yz` has an execution-grade design note before coding, lands behind the release gate `C-read-evidence-contract`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-f94` — Kill-or-commit the TUI (~373 lines of skeletal Textual screens)

Priority: P3. Type: task. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-f94` includes a before/after ownership map, preserves public behavior through parity tests, and deletes or redirects the old path with compatibility notes where needed. The refactor does not change evidence semantics unless a migration and release note say so. Verification artifact: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-jnj.11` — Extend the fzf pattern from select to ambiguous-result moments

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-jnj.11` has an execution-grade design note before coding, lands behind the release gate `C-read-evidence-contract`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-jnj.12` — Empty-result guidance: 0 hits explains itself

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-jnj.12` has an execution-grade design note before coding, lands behind the release gate `C-read-evidence-contract`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-jnj.10` — Make the completion system and DSL discoverable at point of use

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-jnj.10` is expressed through the shared query grammar or an explicit decision record explains why not. CLI, daemon/MCP, docs, and generated support matrix agree on syntax, errors, and result shape. A metamorphic or parity fixture covers the new clause/transform. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-fnm.7` — Generalized child-count predicates: count(unit where ...) comparisons

Priority: P3. Type: feature. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-fnm.7` is expressed through the shared query grammar or an explicit decision record explains why not. CLI, daemon/MCP, docs, and generated support matrix agree on syntax, errors, and result shape. A metamorphic or parity fixture covers the new clause/transform. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-bby.6` — Interaction debt: replace window.prompt(); de-drift the JS renderer

Priority: P3. Type: chore. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-bby.6` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-bby.5` — Long-session navigation: phases/windows/minimap

Priority: P3. Type: feature. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-bby.5` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-bby.3` — Aggregate analytics views in the web UI

Priority: P3. Type: feature. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-bby.3` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-bby.4` — Live session tailing as a first-class mode

Priority: P3. Type: feature. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-bby.4` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-bby.2` — Query completions + expression explain in the web search box

Priority: P3. Type: feature. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-bby.2` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-bby.1` — Workbench responsive under slow/missing routes

Priority: P3. Type: task. Release: `H-web-cockpit`. Lane: `web-evidence-cockpit`. Confidence: `medium`.

The web behavior for `polylogue-bby.1` is backed by the shared API contract, handles loading/stale/error states explicitly, and has a seeded visual or interaction smoke test. Slow or missing daemon routes degrade visibly rather than rendering false emptiness. Verification artifact: web visual smoke, slow-route state fixture, basket-to-citable-export proof.

## `polylogue-9l5.5` — Ship opinionated saved views as product defaults

Priority: P3. Type: feature. Release: `I-analytics-experiments`. Lane: `analytics-experiments`. Confidence: `medium`.

`polylogue-9l5.5` has an execution-grade design note before coding, lands behind the release gate `I-analytics-experiments`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.

## `polylogue-9l5.3` — Pathology epidemiology: corpus-level rates and trends

Priority: P3. Type: feature. Release: `I-analytics-experiments`. Lane: `analytics-experiments`. Confidence: `medium`.

`polylogue-9l5.3` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.

## `polylogue-9l5.4` — Token-economy analytics: cache-lane and attention accounting

Priority: P3. Type: feature. Release: `I-analytics-experiments`. Lane: `analytics-experiments`. Confidence: `medium`.

`polylogue-9l5.4` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.

## `polylogue-3tl.3` — Claim-vs-evidence leaderboard variant (multi-model, incl. open models)

Priority: P3. Type: task. Release: `L-external-legibility`. Lane: `docs-demos-launch`. Confidence: `medium`.

`polylogue-3tl.3` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: one-command demo log, claims-ledger coverage report, install matrix, cold-reader proof.

## `polylogue-jsy` — Harden blob hash validation + drop misleading symlink check

Priority: P3. Type: chore. Release: `A-trust-floor`. Lane: `security-privacy`. Confidence: `medium`.

`polylogue-jsy` preserves byte integrity: before/after byte counts, SHA-256 roundtrip verification, missing-reference handling, and degraded-state rendering are recorded. The feature is blocked until missing blob debt is classified and no cleanup path can delete leased in-flight blobs. Verification artifact: negative Host/Origin/token/spool/security fixture suite.

## `polylogue-1a9` — Remove dead session-commit stubs + unused web-construct row + stale fuzz README

Priority: P3. Type: chore. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-1a9` updates the public artifact and links every factual product claim to the claims ledger or marks it as capability-only/not-yet-measured. Docs/link verification and a cold-reader or demo-regeneration check cover the change. Verification artifact: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-dab` — Stop materializing run-projection cache rows; drop DDL after parity

Priority: P3. Type: task. Release: `M-substrate-consolidation`. Lane: `substrate-consolidation`. Confidence: `medium`.

`polylogue-dab` declares a before/after measurement, an acceptable resource envelope, and a regression guard. The implementation fails loudly on stale/partial state and records phase timing where relevant. Verification artifact: layering/import graph diff, parity tests before/after refactor, public-model compatibility suite.

## `polylogue-jnj.8` — Rationalize root onboarding, tutorial, and reader launcher

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-jnj.8` updates the public artifact and links every factual product claim to the claims ledger or marks it as capability-only/not-yet-measured. Docs/link verification and a cold-reader or demo-regeneration check cover the change. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-jnj.2` — analyze boolean modes -> named projections; facets becomes a real verb

Priority: P3. Type: task. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-jnj.2` has an execution-grade design note before coding, lands behind the release gate `C-read-evidence-contract`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-jnj` — Product surface algebra: one rule per concern across CLI/config/onboarding

Priority: P3. Type: epic. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-jnj` has an execution-grade design note before coding, lands behind the release gate `C-read-evidence-contract`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-20d.7` — EQP sweep + dbstat census on a live-archive copy

Priority: P3. Type: task. Release: `G-live-performance`. Lane: `live-substrate`. Confidence: `medium`.

`polylogue-20d.7` declares a before/after measurement, an acceptable resource envelope, and a regression guard. The implementation fails loudly on stale/partial state and records phase timing where relevant. Verification artifact: live-ingest fixture, event materialization proof, status/liveness report.

## `polylogue-20d.8` — Bound claim-vs-evidence regen latency (43s on live archive)

Priority: P3. Type: task. Release: `G-live-performance`. Lane: `interactive-performance`. Confidence: `medium`.

`polylogue-20d.8` declares a before/after measurement, an acceptable resource envelope, and a regression guard. The implementation fails loudly on stale/partial state and records phase timing where relevant. Verification artifact: named SLO report, daemon hot-path benchmark, push/cache invalidation tests.

## `polylogue-fnm.5` — topic-pack: staged multi-channel topic-lineage retrieval

Priority: P3. Type: feature. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-fnm.5` runs through the provider-general embedding interface, has disabled-provider behavior, bounds work on large sessions, and records retrieval reason/evidence metadata. Quality is compared against FTS or a no-vector baseline before product claims are made. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-fnm.3` — SEQ modifiers: within:<duration> and {n,} occurrence counts

Priority: P3. Type: feature. Release: `C-read-evidence-contract`. Lane: `read-contracts`. Confidence: `medium`.

`polylogue-fnm.3` is expressed through the shared query grammar or an explicit decision record explains why not. CLI, daemon/MCP, docs, and generated support matrix agree on syntax, errors, and result shape. A metamorphic or parity fixture covers the new clause/transform. Verification artifact: CLI/daemon/MCP/Python/web query parity suite and content-hash citation drift fixture.

## `polylogue-83u.5` — Blob store zstd compression (36GB -> est 5-8GB)

Priority: P3. Type: feature. Release: `B-storage-rebuild-bytes`. Lane: `blob-integrity`. Confidence: `medium`.

`polylogue-83u.5` preserves byte integrity: before/after byte counts, SHA-256 roundtrip verification, missing-reference handling, and degraded-state rendering are recorded. The feature is blocked until missing blob debt is classified and no cleanup path can delete leased in-flight blobs. Verification artifact: leased-blob race fixture, blob-reference resolver report, SHA-256 restore/compression proof.

## `polylogue-fs1.5` — Export: Atropos/eval JSONL downstream of the canonical archive

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-fs1.5` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-fs1.3` — Per-source coverage/fidelity declaration for Hermes imports

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-fs1.3` adds or updates an origin contract with detector, parser, raw fixture, normalized fixture, parser fingerprint, and fidelity/completeness notes. Ambiguous inputs are handled deterministically. The regression suite proves idempotent replay and visible degraded/missing-field behavior. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-fs1.2` — Importer: NeMo Relay ATOF/ATIF runtime spans

Priority: P3. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `medium`.

`polylogue-fs1.2` adds or updates an origin contract with detector, parser, raw fixture, normalized fixture, parser fingerprint, and fidelity/completeness notes. Ambiguous inputs are handled deterministically. The regression suite proves idempotent replay and visible degraded/missing-field behavior. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-9l5.17` — Model-drift observatory: candidate changepoints with validity gates, never causal claims

Priority: P4. Type: task. Release: `I-analytics-experiments`. Lane: `analytics-experiments`. Confidence: `spec-gate`.

`polylogue-9l5.17` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.

## `polylogue-9l5.16` — Trajectory Quality Index: reward-shaping composite, never truth

Priority: P4. Type: task. Release: `I-analytics-experiments`. Lane: `analytics-experiments`. Confidence: `spec-gate`.

`polylogue-9l5.16` registers every emitted measure with sample frame, evidence tier, denominator, uncertainty/confound notes, and non-claim wording. Empty backing evidence renders unknown/not-supported, not zero. A seeded fixture demonstrates at least one supported finding and one deliberately unsupported result. Verification artifact: measure registry, sample-frame/uncertainty/confound rendering tests, experiment analysis fixture.

## `polylogue-l4kf.3` — Outbound provenance: git notes (refs/notes/polylogue) + PR/issue citation footers + SARIF pathology export

Priority: P4. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `spec-gate`.

`polylogue-l4kf.3` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-l4kf.2` — Federation: .well-known/ai-sessions manifest + selective content-hash sync

Priority: P4. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `spec-gate`.

`polylogue-l4kf.2` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-r47` — Obsidian/PKM export profile: sessions and findings as wiki-linked Markdown

Priority: P4. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `spec-gate`.

`polylogue-r47` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-7k7` — Research-tooling export lane: inspect-ai / Docent formats

Priority: P4. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `spec-gate`.

`polylogue-7k7` emits an export/interchange artifact that preserves stable object refs, evidence provenance, caveats, and content hashes. A roundtrip or consumer fixture proves no duplicate facts and no silent loss of missing/private blobs. Verification artifact: OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

## `polylogue-4g5` — Expose the archive as an HPI module and Promnesia source

Priority: P4. Type: feature. Release: `K-interop-origin-export`. Lane: `origin-interop-export`. Confidence: `spec-gate`.

`polylogue-4g5` has an execution-grade design note before coding, lands behind the release gate `K-interop-origin-export`, and records a focused proof artifact. Acceptance requires one seeded positive case, one degraded/empty case where applicable, docs or generated-surface updates for any public behavior, and verification via OriginSpec detector/parser/fixture/fidelity suite and content-hash export/import roundtrip.

