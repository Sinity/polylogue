# 099. polylogue-2qx — OriginSpec: one package per origin, dispatch order derived from declared strictness

Priority/type/status: **P2 / feature / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **blocked-hard**.

Hard blockers: polylogue-o21

## What the bead says

Adding a provider/origin is the most common expansion Polylogue will ever do (uiw enumerates a dozen candidates; Grok, OTel-GenAI, Langfuse waiting), yet origin logic is smeared: looks_like + parser in sources/parsers/*, hand-ordered dispatch list in dispatch.py (strict-before-loose maintained by vigilance — the Hermes ordering caution exists precisely because one wrong insertion steals records), Provider enum in types.py, schema package elsewhere, usage-coverage entry elsewhere, fidelity/completeness rows elsewhere. The generic openai-chat detector (uiw) raises the stakes: a loose detector in a hand-ordered list is a standing footgun.

## Existing design note

(1) An OriginSpec dataclass per origin, one package per origin under sources/origins/<name>/: detector (looks_like), parser entry, declared STRICTNESS TIER (exact-schema > record-path-validated > structural-sequence > loose-shape), schema-package pointer, capture-mode/fidelity declaration, usage-coverage class, pricing hints. (2) Dispatch order is DERIVED: sort by declared tier, then registration order within tier — the hand-ordered list in dispatch.py dies; a loose detector physically cannot outrank a strict one, and the uiw ordering caution becomes a type-level guarantee. Add a dispatch-order regression test that feeds every origin's fixtures through the full chain and asserts each is claimed by its own detector (catches tier misdeclarations). (3) Provider-completeness (devtools lab provider completeness) reads the specs instead of its own inventory — one origin, one place, and the completeness report's gaps become actionable-by-construction (ties the declare-once bead). (4) Auto-discovery via package scan now; the same spec shape is what a third-party entry-point mechanism would load later (the 'anyone adds a provider' gap from the ecosystem analysis) — design the spec as if external, ship it internal. Migration: mechanical, one origin per commit, mypy-netted; fixture suite is the behavior lock.

## Acceptance criteria gap

This active bead lacks acceptance criteria in the export. Add checkable acceptance criteria before coding unless this packet explicitly supplies a temporary gate.

## Static mechanism / likely defect

Origin dispatch is spread across detectors, parser bases, provider completeness, and preflight. The system needs one explicit OriginSpec per origin to avoid ambiguous importer behavior.

## Source anchors to inspect first

- `polylogue/sources/dispatch.py` — Current origin/source dispatch logic; target for OriginSpec consolidation.
- `polylogue/sources/import_preflight.py` — Preflight/readiness should report origin strictness and ambiguity.
- `polylogue/sources/provider_completeness.py` — Provider completeness is adjacent to OriginSpec readiness.
- `polylogue/sources/parsers/base.py` — Parser base contracts should be folded into OriginSpec.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.

## Implementation plan

1. Create OriginSpec with id, detector, strictness, parser, fixture set, normalized schema mapping, source material policy, fidelity declaration, docs link.
2. Generate dispatch order from strictness/specificity instead of hand-coded order.
3. Make preflight explain which specs matched, which lost, and why.
4. Backfill existing ChatGPT/Claude/Codex/Gemini/Antigravity/Hermes origins into specs.

## Tests to add

- Ambiguous fixture matches deterministic most-specific origin.
- Each OriginSpec has raw and normalized fixtures.
- Preflight output is actionable for unknown/format-drift files.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
