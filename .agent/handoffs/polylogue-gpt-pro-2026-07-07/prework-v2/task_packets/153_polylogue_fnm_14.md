# 153. polylogue-fnm.14 — find <query> | compact: token-budgeted corpus-compaction projection with drop manifest

Priority/type/status: **P2 / feature / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The R&D-flywheel enabler: package a queried cohort as a decision-dense, lineage-deduplicated digest for an external LLM, with an honest fidelity manifest. A projection/render preset over the read algebra (CompactProjectionSpec x layout:corpus-compaction-pack) — NOT a context subsystem: compile_context answers "what do I hand an agent to continue"; compact answers "what is the highest-value lowest-spam evidence digest of a COHORT" (cross-session ranking, lineage-family dedup, fairness strata, external manifest — shoving it into ContextImage would make ContextImage a second read algebra). Deterministic v1, no LLM summarization (destroys auditability before the manifest exists). Hard filter by material_origin (drop runtime_protocol/context, generated packs unless asked, successful unreferenced tool spam) then additive scoring with NAMED reasons in the manifest (authoredness, decision/outcome/error-fix signals, novelty-within-lineage-family, diversity bonus, redundancy/length penalties). Error->fix pairs kept as narrative units (command, structured failure, diagnosis, fix, verify — from actions keystone fields, never regex). Lineage dedup at logical-family grain: inherited prefix emitted ONCE per family with explicit markers; dangling branch point => physical fallback LOUDLY marked, never silent. Budget: stratified greedy water-fill (NOT pure knapsack — starves small sessions/minority providers; strata = lineage family/session/provider/evidence kind), reserve split ~3% header / 7% corpus map / 10% drop summary / 80% evidence, degradation order clip -> collapse-runs-to-deterministic-counts -> skeleton-only -> drop-with-manifest -> index-only-pack failure. Decision-density-biased, NOT tail-biased (context images are tail-biased for continuity; this is not that). Manifest is THE feature: drop counts by reason, per-session included/dropped tokens, stable anchors for every retained block (ties into block content-hash anchors when they land). Token proxy: word count with ~0.72 BPE derate (wave finding). Output envelope carries query_run/result_relation/pack refs when rxdo.3 lands, so external LLM outputs (auto-captured by browser extension) can attach back as annotation batches — closing the outsourced-cognition loop.

## Acceptance criteria

Fixture with protocol/tool spam compacts to a digest excluding it with per-material_origin drop counts; failed->fix->verify fixture keeps the pair with refs; fork/resume fixture emits shared prefix once and reports duplicate-prefix omissions; 60k budget test proves the deterministic degradation order; every digest anchor round-trips to a source ref; context-image and compact remain separate payload shapes sharing helpers. Verify: focused projection tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: The R&D-flywheel enabler: package a queried cohort as a decision-dense, lineage-deduplicated digest for an external LLM, with an honest fidelity manifest. A projection/render preset over the read algebra (CompactProjectionSpec x layout:corpus-compaction-pack) — NOT a context subsystem: compile_context answers "what do I hand an agent to continue"; compact answers "what is the highest-value lowest-spam evidence digest of a COHORT" (cross-session ranking, lineage-family dedup, fairness strata, external manifest — sh…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Identify the currently duplicated surface paths for this behavior.
2. Create/extend the shared contract object and route one surface at a time through it.
3. Add parity tests across CLI, daemon/API, MCP, and Python facade.
4. Delete dead surface-side code after parity is green.

## Tests to add

- Acceptance proof: Fixture with protocol/tool spam compacts to a digest excluding it with per-material_origin drop counts
- Acceptance proof: failed->fix->verify fixture keeps the pair with refs
- Acceptance proof: fork/resume fixture emits shared prefix once and reports duplicate-prefix omissions
- Acceptance proof: 60k budget test proves the deterministic degradation order
- Acceptance proof: every digest anchor round-trips to a source ref
- Acceptance proof: context-image and compact remain separate payload shapes sharing helpers.
- Acceptance proof: Verify: focused projection tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
