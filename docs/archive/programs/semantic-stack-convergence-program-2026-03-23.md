# Semantic Stack Convergence Program

Date: 2026-03-23
Status: executed implementation program
Role: canonical execution record for the semantic-stack convergence campaign

## Execution Outcome

This program is executed.

It landed:

1. a split harmonization boundary for typed adapters, fallback extraction, and provider-meta hydration
2. a canonical semantic facts layer shared by semantic proof and higher-order semantic products
3. session-analysis products converged onto those shared semantic facts instead of recomputing overlapping signals
4. semantic proof surfaces switched to the canonical semantic facts layer
5. a declared semantic surface-contract registry with reusable contract evaluation
6. operator exposure for semantic contract inventory through `polylogue check --semantic-contracts`
7. a named `semantic-stack` validation lane covering harmonization, facts, proof, QA, and contract inventory

See also:

- `schema-and-evidence-pipeline-convergence-program-2026-03-23.md`
- `core-architecture-convergence-program-2026-03-23.md`
- `multi-surface-semantic-proof-program-2026-03-22.md`
- `semantic-proof-and-showcase-proof-lanes-program-2026-03-22.md`
- `../planning-and-analysis-map-2026-03-21.md`

## One-Line Goal

Make Polylogue's semantic interpretation and semantic proof surfaces behave like
one coherent stack:

provider adapter -> harmonized semantic facts -> session/profile/tagging
products -> proof/export contracts -> operator/report surfaces.

## Why This Is Now The Main Frontier

The previous campaigns closed the major architecture and evidence-path work:

- query/storage/front-door convergence
- schema package authority
- artifact/cohort/proof control plane
- publication control plane
- site/repo-shape cleanup
- runtime/testing closure lanes
- schema-and-evidence pipeline convergence

What remains is now concentrated in the semantic layer:

- cross-provider harmonization still mixes typed adapter routing with fallback
  extraction and direct text/tool/reasoning helpers
- higher-order semantic products still compose multiple separate extraction
  passes over conversations
- semantic proof has become strong and broad, but its internal fact and surface
  policy machinery is now one of the biggest remaining architecture clusters
- render/export surfaces and semantic proof still express some of the same
  meaning-loss rules in different places

This is now the strongest follow-up because it sits directly on Polylogue's
product thesis: support all available metadata, make interpretation inspectable,
and avoid provider-specific hardcoding leaking into downstream semantics.

## Main Remaining Cluster

The biggest remaining semantic cluster spans:

- `polylogue/schemas/unified.py`
- `polylogue/lib/provider_semantics.py`
- `polylogue/lib/session_profile.py`
- `polylogue/lib/attribution.py`
- `polylogue/lib/work_events.py`
- `polylogue/lib/phases.py`
- `polylogue/lib/decisions.py`
- `polylogue/lib/tagging.py`
- `polylogue/rendering/semantic_proof.py`
- `polylogue/rendering/semantic_proof_facts.py`
- `polylogue/rendering/semantic_proof_surfaces.py`
- selected render/export surfaces such as `rendering/renderers/html.py`

## Program Thesis

Polylogue should have:

1. one canonical provider-neutral semantic message/facts layer
2. one harmonization boundary that turns provider-native payloads into those
   facts through typed adapters
3. one downstream semantic product layer for session profiles, work events,
   phases, decisions, and tags
4. one semantic proof engine that compares surfaces against the same canonical
   facts layer
5. one explicit registry of semantic preservation/loss contracts for export and
   read surfaces
6. one operator/report surface for semantic proof and semantic-profile evidence

## Non-Goals

This program is not:

- another schema package redesign
- a storage/backend campaign
- a site/template redesign
- a provider parser rewrite from scratch
- a UI/TUI campaign

Those may be good later. This one is specifically about converging the semantic
stack around one inspectable, typed semantic core.

## Architectural Rules

### 1. Harmonization Must Be One Boundary

Provider-native payload interpretation should happen once, at the semantic
adapter boundary. Downstream products should not keep re-deriving role/tool/
reasoning/content semantics from raw provider shapes.

### 2. Facts And Products Are Different Layers

Do not keep:

- message harmonization
- semantic fact extraction
- profile/work-event/decision assembly
- proof policy declarations

inside one implementation bucket merely because they all concern semantics.

### 3. Semantic Proof Must Compare Against Shared Facts

Semantic proof should compare output surfaces to the same canonical semantic
facts that power session profiles, tags, and downstream analysis products.

### 4. Surface Contracts Must Be Declared Once

Declared semantic loss or preservation rules for export/read surfaces should
live in one explicit contract registry, not be re-expressed ad hoc across proof
helpers and renderer assumptions.

### 5. Fallback Logic Must Be Narrow And Inspectable

Fallback extraction for malformed or partial provider records is still useful,
but it should be isolated, explicitly marked as fallback, and invisible to
normal typed-adapter paths.

## Execution Order

1. harmonization boundary cleanup
2. canonical semantic facts layer
3. session-analysis product convergence
4. semantic proof engine convergence
5. surface contract registry convergence
6. operator/report exposure convergence
7. named semantic-stack verification lane

## Step 1: Harmonization Boundary Cleanup

### Goal

Separate typed provider adapter routing, fallback extraction, and final
harmonized-message assembly into explicit layers.

### Current Problems

- `schemas/unified.py` currently acts as:
  - message model definition
  - provider adapter dispatcher
  - fallback extraction home
  - token/cost/text/tool extraction entrypoint
- `lib/provider_semantics.py` still owns a broad set of provider-specific text
  and content heuristics
- normal typed-adapter flow and malformed fallback flow live too close together

### Target Shape

Split the current stack into:

1. semantic models
2. provider adapter dispatch
3. fallback semantic extraction
4. shared provider extraction helpers

### Main Modules

- `polylogue/schemas/unified.py`
- `polylogue/lib/provider_semantics.py`
- new semantic adapter/fallback support modules under `polylogue/lib/` or
  `polylogue/schemas/`
- provider adapter modules under `polylogue/sources/providers/`

### Acceptance Criteria

- normal typed-adapter flow does not depend on fallback extraction helpers
- fallback paths are explicitly isolated and test-covered
- `extract_harmonized_message()` becomes a thin orchestrator over smaller seams

## Step 2: Canonical Semantic Facts Layer

### Goal

Introduce one typed, provider-neutral semantic facts layer derived from
harmonized conversations/messages.

### Current Problems

- session profile, work-event extraction, phase extraction, decisions, and
  semantic proof each compute overlapping counts and semantic signals
- there is no single canonical "semantic facts" object for one conversation
- proof and downstream products therefore risk drifting in what they consider
  meaningful semantics

### Target Shape

Create a typed conversation-level semantic facts layer that exposes:

- message/role/timestamp counts
- tool-use and reasoning counts
- content-block and attachment presence
- provider/model/cost/time metadata
- text-bearing vs non-renderable message distinctions
- branch/continuation/topology facts if needed

### Main Modules

- new semantic-facts module(s) under `polylogue/lib/`
- `polylogue/lib/models.py`
- `polylogue/lib/session_profile.py`
- `polylogue/rendering/semantic_proof_facts.py`

### Acceptance Criteria

- semantic proof input facts come from the canonical layer
- session profile/product code depends on the same facts layer
- repeated counting logic disappears from downstream products

## Step 3: Session-Analysis Product Convergence

### Goal

Make higher-order semantic products consume canonical facts rather than
re-reading conversation structures independently.

### Current Problems

- session profile orchestrates attribution, work events, phases, decisions,
  pricing, and tags with several independent passes
- tags and other products consume partially aggregated profile data rather than
  a shared semantic base

### Target Shape

Define a clear downstream product layer:

1. canonical semantic facts
2. derived semantic products:
   - attribution
   - work events
   - phases
   - decisions
   - session profile
   - tag inference

### Main Modules

- `polylogue/lib/session_profile.py`
- `polylogue/lib/attribution.py`
- `polylogue/lib/work_events.py`
- `polylogue/lib/phases.py`
- `polylogue/lib/decisions.py`
- `polylogue/lib/tagging.py`

### Acceptance Criteria

- session profile assembly becomes thinner and more obviously staged
- derived products reuse canonical facts instead of recomputing basic semantics
- downstream semantic products gain clearer typed inputs and tests

## Step 4: Semantic Proof Engine Convergence

### Goal

Split the proof engine into explicit layers and align it with the canonical
facts model.

### Current Problems

- `semantic_proof_surfaces.py` and `semantic_proof_facts.py` still hold a very
  large amount of mixed policy, extraction, and per-surface comparison logic
- surface-specific preservation rules are harder to inspect than they should be

### Target Shape

Split the proof engine into:

1. canonical input facts
2. surface output fact extractors
3. surface preservation/loss contracts
4. per-surface runners
5. suite orchestration and report formatting

### Main Modules

- `polylogue/rendering/semantic_proof.py`
- `polylogue/rendering/semantic_proof_facts.py`
- `polylogue/rendering/semantic_proof_surfaces.py`
- new semantic-proof contract/registry modules

### Acceptance Criteria

- suite orchestration is thin
- surface contracts are inspectable without reading giant mixed functions
- proof engine consumes the canonical semantic facts layer from Step 2

## Step 5: Surface Contract Registry Convergence

### Goal

Make render/export/read surfaces declare semantic preservation/loss expectations
once and reuse them everywhere.

### Current Problems

- semantic proof knows a lot about what each surface preserves or intentionally
  loses, but render/export code does not expose that contract directly
- this makes it harder to reason about intended lossiness vs accidental loss

### Target Shape

Introduce one registry of semantic surface contracts covering:

- canonical markdown
- export formats
- query summary/list/detail
- stream formats
- MCP read/detail surfaces
- site/read-model projections where relevant

### Main Modules

- new semantic contract registry modules under `polylogue/rendering/`
- selected render/export command and renderer modules
- `polylogue/rendering/semantic_proof_surfaces.py`

### Acceptance Criteria

- proof uses declared contracts instead of re-encoding surface intent inline
- operator/docs can expose the contract inventory directly if needed
- new surfaces can be added by declaring contracts, not by scattering policy

## Step 6: Operator And Report Exposure Convergence

### Goal

Make semantic proof and semantic-profile evidence easier to inspect through one
operator-facing shape.

### Current Problems

- semantic proof is operator-visible, but downstream semantic products are less
  intentionally exposed
- report/publication/QA surfaces may still project semantic information
  differently

### Target Shape

Route semantic evidence through a clearer operator surface, likely by exposing:

- semantic proof suite results
- per-provider or per-surface summaries
- semantic profile / semantic facts inspection hooks where useful

### Main Modules

- `polylogue/cli/commands/check.py`
- `polylogue/showcase/qa_runner.py`
- `polylogue/showcase/qa_report.py`
- `polylogue/publication.py`
- `polylogue/site/publication_support.py`

### Acceptance Criteria

- QA/publication/check consume the same typed semantic-report shapes
- semantic evidence is easier to inspect without reading internal code

## Step 7: Named Semantic-Stack Verification Lane

### Goal

Add one explicit verification lane for the semantic stack itself.

### Target Shape

The lane should prove:

- typed adapters and fallback boundaries behave as expected
- canonical semantic facts and session-profile aggregates stay aligned
- semantic proof surfaces remain consistent with declared contracts
- higher-order products do not drift from the semantic base layer

### Verification

The completion gate for this program should include:

- harmonization/provider adapter tests
- session profile / attribution / work-event / decision tests
- semantic proof suite tests
- showcase/report/CLI semantic proof tests
- at least one named validation lane for the semantic stack

## Expected Outcome

After this program, Polylogue should have one semantic interpretation story:

- provider-native data is harmonized once
- semantic facts are computed once
- downstream semantic products derive from that one facts layer
- proof compares surfaces to those same facts
- operator/report surfaces expose the same semantic truth rather than parallel
  approximations
