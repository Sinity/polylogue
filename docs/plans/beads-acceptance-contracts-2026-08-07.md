# Beads acceptance-contract wave — 2026-08-07

- Source export targets: 218
- Applied to this exact Beads snapshot: 218
- Refused as stale/nonempty: 0
- Structured contract key: `metadata.acceptance_contract_v1`
- Validator: `python devtools/beads_acceptance_contracts.py --manifest docs/plans/beads-acceptance-contracts-2026-08-07.txt`

The validator does not judge natural-language prose. It validates a typed outcome/evidence/route/verification/anti-vacuity/safety/closure object, recomputes the source-snapshot SHA-256, and requires the human-readable acceptance criteria to be an exact rendering of it. The Bead graph gate validates every present contract, and `lane-brief` marks `confidence=planner-review` records as dispatch-blocked.

Sparse records remain marked `confidence=planner-review`; that flag is not permission for a Luna implementation worker to make architectural choices. It requires planner review before dispatch.
