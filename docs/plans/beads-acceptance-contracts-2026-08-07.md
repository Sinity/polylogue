# Beads acceptance-contract wave — 2026-08-07

- Source export targets: 218
- Applied to this exact Beads snapshot: 218
- Refused as stale/nonempty: 0
- Structured contract key: `metadata.acceptance_contract_v1`
- Validator: `devtools lab policy acceptance-contracts --manifest docs/plans/beads-acceptance-contracts-2026-08-07.txt`

The validator does not judge natural-language prose. It validates a typed outcome/evidence/route/verification/anti-vacuity/safety/closure object, recomputes a scope-bearing source SHA-256 (lifecycle status and timestamps are deliberately excluded), requires an allowed confidence value, and requires the human-readable acceptance criteria to be an exact rendering of it. Live-operation contracts also require typed receipt verification. The `devtools lab policy acceptance-contracts` gate validates the committed manifest, and `lab policy bead-graph` rejects missing or invalid manifest contracts. `lane-brief` marks `confidence=planner-review` records as dispatch-blocked.

Sparse records remain marked `confidence=planner-review`; that flag is not permission for a Luna implementation worker to make architectural choices. It requires planner review before dispatch.
