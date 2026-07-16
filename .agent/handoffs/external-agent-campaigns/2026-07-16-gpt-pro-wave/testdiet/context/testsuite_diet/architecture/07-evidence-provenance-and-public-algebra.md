---
created: 2026-07-16
purpose: Decide evidence-value semantics, provenance conservation, and stable public projections for L18-L20 and L25
status: recommended-decision
project: polylogue
---

# Evidence provenance and public algebra

## Decision

Adopt the existing `polylogue-cuxz.2` design as the shared epistemic value
protocol: `EvidenceValue[T]` is embedded in owning domain payloads and preserves
independent value, authority, provenance, temporal, coverage, freshness, and
confidence axes. It is not a table, lifecycle, or one confidence score.

Each domain retains one canonical fact model. CLI, MCP, Python API, HTTP, and
renderers project that model through declarations; surfaces may omit detail for
size/privacy but may not change meaning or strengthen evidence.

## EvidenceValue axes

| Axis | Required distinctions |
| --- | --- |
| `value_state` | `known`, `unknown`, `unavailable`, `skipped`, `not_applicable`, `redacted` |
| `value` | typed value when known; measured zero remains a real value |
| `measurement_authority` | structural, provider-reported, catalog-derived, rule-derived, model-derived, agent-declared, judged |
| evidence/definition | source refs and the versioned definition/recipe that produced the claim |
| temporal provenance | recorded, estimated, or unknown source/time confidence; observed frame |
| enumeration | census, sample, inferred/partial, or not applicable |
| frame/coverage | intended frame, observed denominator/coverage, exclusions and reasons |
| freshness/degradation | fresh, stale, timed-out, unavailable, degraded plus cause and last-good evidence |
| calibrated confidence | optional only with calibration/definition ref; never a naked generic float |

Fact-family declarations state which axes are mandatory and derive schemas,
adapters, renderer labels, examples, and completeness diagnostics. A family
cannot silently omit an axis it declared relevant.

## Composition laws

1. Removing evidence cannot increase authority, coverage, freshness, or
   calibrated confidence.
2. Known zero is not unknown, skipped, or unavailable.
3. Aggregate authority and temporal provenance are no stronger than the weakest
   required contribution unless the definition explicitly transforms them.
4. Totals name their grain and denominator; physical and logical sessions do
   not mix silently.
5. Missing price with exact tokens yields exact tokens and unknown cost, not
   zero cost or estimated tokens.
6. Contradictory sources remain an explicit conflict/debt set until a declared
   authority or judgment resolves them; order does not select a winner.
7. Census enumeration has no sampling interval. Sampled/incomplete frames must
   retain coverage and may not render as archive-wide facts.
8. Materialization time never masquerades as event time or recency.

## Domain ownership

Use the protocol for temporal values, outcomes, usage/price, phase/profile
inference, quota observations, query results, metrics, source freshness, and
status components. Computation and durability remain in their owning modules.

Do not make a generic fact store. A usage snapshot, status component, and
query-result receipt share epistemic axes but not identity, update cadence,
retention, or authority algorithms.

## Public vocabulary

`Origin` is the public source-origin identity for sessions, filters, and read
payloads. `Provider` remains legitimate at provider-wire, pricing-provider, and
embedding-provider boundaries. `Source` retains richer acquisition identity.

The public projection must never invert the non-injective provider-to-origin
mapping. Internal models still using provider-named storage fields pass through
one explicit boundary adapter (`project_origin_payload` during migration), and
new canonical domain models use origin/source vocabulary directly.

Other identities—material source, logical session, model, runtime, archive,
query result, assertion—remain separate rather than being encoded into origin.

## Surface algebra

Declare `Query × Projection × Render` independently:

- query selects logical facts;
- projection chooses documented fields/views and preserves evidence axes;
- render encodes the projection for text/JSON/MCP/HTTP without semantic repair.

Stable object/result refs carry identity and permit bounded detail retrieval.
The MCP and current web reader remain rewrite boundaries: preserve obligations
for the verb-algebra and replacement web design rather than adding more aliases
or adapter-local fact models now.

Output safety belongs to projection/render policy: attacker-controlled content
is data, never schema, tool description, instruction authority, HTML/script, or
terminal control. Redaction changes `value_state` and disclosure metadata; it
does not fabricate an empty known value.

## Competitive alternatives

| Alternative | Advantage | Why not chosen |
| --- | --- | --- |
| Scalar value plus nullable confidence | Compact | Collapses absence, authority, coverage, freshness, and calibration |
| One evidence tier badge | Easy rendering | Independent axes can coexist and propagate differently |
| Universal evidence/fact table | Central queries | Steals identity/lifecycle from domains and becomes an unbounded framework |
| Surface-specific payload models | Tailored APIs | Allows semantic drift and duplicated reconciliation |
| Provider everywhere, renamed at display | Minimal storage change | Provider and origin are different, non-injective vocabularies |
| Null/zero/sentinel conventions | Cheap | Recreates the exact unknown/zero/epoch bugs this contract addresses |
| One strongest-source winner for contradictions | Simple output | Hides disagreement and makes input order authoritative |

## Migration sequence

1. Land the protocol and `FactFamilySpec` on three canaries: exact tokens with
   unknown price, stale status with last-good evidence, and excluded source with
   known byte lag.
2. Repair temporal provenance from storage through public reads.
3. Migrate usage, outcome/inference, query-result, metric, and freshness
   families in domain-sized slices.
4. Generate cross-surface schemas/examples from declarations; remove duplicate
   vocabularies only after parity proof.
5. Complete public origin projection as owning models are retired; never perform
   a blind provider rename.

## Required proof

- each axis round-trips independently through real storage/domain-to-public
  adapters;
- known zero, unknown, skipped, unavailable, redacted, and not-applicable render
  differently on every surface;
- evidence removal and weaker temporal/coverage inputs cannot strengthen output;
- quantitative totals conserve contribution and weakest provenance;
- surface projections agree modulo declared privacy/detail omissions;
- injected archive strings cannot alter schema, tools, instructions, terminal,
  or HTML structure;
- mutation removal of authority, value state, definition ref, coverage, or
  temporal source fails production-route tests.

Primary evidence: `polylogue-cuxz`, `polylogue-cuxz.2`, `polylogue-f2qv.6`,
`polylogue-b2r9`, `polylogue-9e5.29`, `polylogue-9e5.30`, `polylogue-cpf.5`,
`polylogue-t46.8`; `polylogue/surfaces/projection_spec.py`,
`polylogue/insights/rigor.py`, temporal/evidence ancestry modules, and
`docs/provider-origin-identity.md`.
