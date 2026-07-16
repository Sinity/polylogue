# 193. polylogue-3tl.9 — Docs-and-visuals ownership: coverage lint + regenerable visuals as a standing devloop gate

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The operator wants agents to comprehensively OWN external-facing docs and visual material, not touch them opportunistically. The repo already has the machinery pattern (render all --check, doc-commands linter, pages build, visual-tapes) but no coverage contract: nothing fails when a public surface ships undocumented, when a doc references a dead flag (doc-commands covers commands only), or when a screenshot/GIF rots against current UI. Docs drift is currently discovered by humans reading.

## Existing design note

(1) COVERAGE LINT (devtools verify docs-coverage): every public CLI command/verb, MCP tool, config key, and daemon route must be reachable from the docs tree (generated inventories make this a set diff, same pattern as the topology gate); new-surface-without-docs fails the lane with the exact missing entry named (actionable-error discipline from o21). (2) VISUAL FRESHNESS: every committed screenshot/GIF must be a visual-tapes artifact with a spec (3tl.5 machinery) — a render pass regenerates them against the seeded corpus; drift = the regen diff exceeds a perceptual threshold -> flagged for re-record. No hand-shot images in docs. (3) DEVLOOP GATE: the conductor End Gate gains a docs item — a slice that changed public surfaces is not closeable until docs-coverage passes (RUNBOOK/PROCESS edit + this bead). (4) Docs IA pass rides the atlas (y0b) + README (3tl.1); this bead is the ENFORCEMENT layer that keeps them true afterward. Extends 6bu (site link/cache checks) rather than replacing it — 6bu is transport health, this is content coverage.

## Acceptance criteria

1. New lane `devtools verify docs-coverage`: builds generated inventories of every public CLI command/verb, MCP tool, config key, and daemon route and fails when any is not reachable from the docs tree, naming the exact missing entry (actionable-error discipline, same set-diff pattern as the topology gate). Passes on the current tree. 2. Visual freshness: every committed screenshot/GIF is a visual-tapes artifact with a spec (3tl.5 machinery); a render pass regenerates them against the seeded corpus and flags any whose regen diff exceeds the perceptual threshold. A lint asserts no hand-shot images remain in the docs tree. 3. Devloop End Gate: a slice that changed a public surface is not closeable until docs-coverage passes (RUNBOOK/PROCESS updated). Verify: `devtools verify docs-coverage` green on HEAD; add a throwaway undocumented CLI verb locally and confirm the lane fails naming that exact verb.

## Static mechanism / likely defect

Issue description localizes the mechanism: The operator wants agents to comprehensively OWN external-facing docs and visual material, not touch them opportunistically. The repo already has the machinery pattern (render all --check, doc-commands linter, pages build, visual-tapes) but no coverage contract: nothing fails when a public surface ships undocumented, when a doc references a dead flag (doc-commands covers commands only), or when a screenshot/GIF rots against current UI. Docs drift is currently discovered by humans reading. Design direction: (1) COVERAGE LINT (devtools verify docs-coverage): every public CLI command/verb, MCP tool, config key, and daemon route must be reachable from the docs tree (generated inventories make this a set diff, same pattern as the topology gate); new-surface-without-docs fails the lane with the exact missing entry named (actionable-error discipline from o21). (2) VISUAL FRESHNESS: every committed screenshot/GIF must be a vi…

## Source anchors to inspect first

- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. (1) COVERAGE LINT (devtools verify docs-coverage): every public CLI command/verb, MCP tool, config key, and daemon route must be reachable from the docs tree (generated inventories make this a set diff, same pattern as the topology gate)
2. new-surface-without-docs fails the lane with the exact missing entry named (actionable-error discipline from o21).
3. (2) VISUAL FRESHNESS: every committed screenshot/GIF must be a visual-tapes artifact with a spec (3tl.5 machinery) — a render pass regenerates them against the seeded corpus
4. drift = the regen diff exceeds a perceptual threshold -> flagged for re-record.
5. No hand-shot images in docs.
6. (3) DEVLOOP GATE: the conductor End Gate gains a docs item — a slice that changed public surfaces is not closeable until docs-coverage passes (RUNBOOK/PROCESS edit + this bead).
7. (4) Docs IA pass rides the atlas (y0b) + README (3tl.1)

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: New lane `devtools verify docs-coverage`: builds generated inventories of every public CLI command/verb, MCP tool, config key, and daemon route and fails when any is not reachable from the docs tree, naming the exact missing entry (actionable-error discipline, same set-diff pattern as the topology gate).
- Acceptance proof: Passes on the current tree.
- Acceptance proof: 2.
- Acceptance proof: Visual freshness: every committed screenshot/GIF is a visual-tapes artifact with a spec (3tl.5 machinery)
- Acceptance proof: a render pass regenerates them against the seeded corpus and flags any whose regen diff exceeds the perceptual threshold.
- Acceptance proof: A lint asserts no hand-shot images remain in the docs tree.
- Acceptance proof: 3.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
