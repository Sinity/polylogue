# 163. polylogue-cpf — Land the six doctrines: time, writers, finding-provenance, degraded-modes, non-goals, injected-context trust

Priority/type/status: **P2 / epic / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **epic-needs-child-closure**.

## What the bead says

The six doctrine texts were written in full in the 2026-07-03 design session (session transcript is the source — Claude-Session trailer on this commit); they cover the cheap-to-write, expensive-to-lack gaps: time semantics (three times, UTC epoch-ms canon, skew tolerance, duration honesty), writer classes (four classes, one writer-class per file, cross-tier interruption validity), finding provenance (five-part stanza, re-runs supersede, semantic version bumps flag stale findings), degraded-mode ladder (five rungs, degrade-loudly-once, sacred EVIDENCE-ONLY floor), the non-goals register (with revisit triggers), and injected-context trust classes (OPERATOR/SYSTEM/QUOTED). Landing them is mostly transcription + wiring the named enforcement hooks.

## Existing design note

(1) Commit texts under docs/doctrine/ (or internals sections — match the ttu docs-IA tiering), adjusted to repo voice; link from architecture-spine. (2) Enforcement hooks, each small: schema-audit check for TEXT timestamps in new DDL; writer-class docstring convention + layering check; provenance-stanza refusal in the findings lane (3tl.4); degraded-rung declaration in feature review; trust-class typing in the ContextSource protocol (37t.11 carries the implementation — this bead lands the doctrine text + the deny-lexicon tripwire test fixture). (3) Retire folklore: where a doctrine supersedes scattered notes (time comments, writer lore), point them at the doctrine.

## Acceptance criteria

Six doctrine documents committed and indexed; the three cheap lints wired (timestamp DDL check, provenance stanza gate, trust deny-lexicon fixture); architecture-spine links them; bd memory updated to point at doctrines instead of restating them.

## Static mechanism / likely defect

Issue description localizes the mechanism: The six doctrine texts were written in full in the 2026-07-03 design session (session transcript is the source — Claude-Session trailer on this commit); they cover the cheap-to-write, expensive-to-lack gaps: time semantics (three times, UTC epoch-ms canon, skew tolerance, duration honesty), writer classes (four classes, one writer-class per file, cross-tier interruption validity), finding provenance (five-part stanza, re-runs supersede, semantic version bumps flag stale findings), degraded-mode ladder (five rungs,… Design direction: (1) Commit texts under docs/doctrine/ (or internals sections — match the ttu docs-IA tiering), adjusted to repo voice; link from architecture-spine. (2) Enforcement hooks, each small: schema-audit check for TEXT timestamps in new DDL; writer-class docstring convention + layering check; provenance-stanza refusal in the findings lane (3tl.4); degraded-rung declaration in feature review; trust-class typing in the Conte…

## Source anchors to inspect first

- `polylogue/core/dates.py:10` — parse_date has no injected clock parameter.
- `polylogue/core/dates.py:37` — RELATIVE_BASE uses ambient datetime.now.
- `polylogue/archive/query/expression.py:2440` — Query grammar recognizes relative-date literals.
- `polylogue/archive/query/spec.py:498` — SessionQuerySpec.from_params is the central query-spec constructor.
- `polylogue/insights/temporal_source.py:66` — classify_profile_hwm_source promotes any updated_at to provider_ts.
- `polylogue/insights/temporal_source.py:97` — classify_aggregate_hwm_source currently collapses all non-empty source updates to provider_ts.
- `polylogue/insights/audit.py:173` — build_insight_rigor_audit_report is the audit entry point.
- `polylogue/insights/audit.py:194` — Current code iterates list_rigor_contracts, not the product registry.
- `polylogue/insights/audit.py:216` — Registry lookup is secondary and skipped for products without contracts.
- `polylogue/insights/rigor.py:85` — _RIGOR_MATRIX declares only a subset of registered products.
- `polylogue/insights/registry.py:294` — INSIGHT_REGISTRY is the universe the audit should iterate.

## Implementation plan

1. (1) Commit texts under docs/doctrine/ (or internals sections — match the ttu docs-IA tiering), adjusted to repo voice
2. link from architecture-spine.
3. (2) Enforcement hooks, each small: schema-audit check for TEXT timestamps in new DDL
4. writer-class docstring convention + layering check
5. provenance-stanza refusal in the findings lane (3tl.4)
6. degraded-rung declaration in feature review
7. trust-class typing in the ContextSource protocol (37t.11 carries the implementation — this bead lands the doctrine text + the deny-lexicon tripwire test fixture).

## Tests to add

- Acceptance proof: Six doctrine documents committed and indexed
- Acceptance proof: the three cheap lints wired (timestamp DDL check, provenance stanza gate, trust deny-lexicon fixture)
- Acceptance proof: architecture-spine links them
- Acceptance proof: bd memory updated to point at doctrines instead of restating them.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
