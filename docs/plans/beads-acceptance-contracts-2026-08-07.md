# Beads acceptance-contract wave — 2026-08-07

- Source export targets: 218
- Canonical snapshot status: pending typed-carrier regeneration from live authority
- Dispatch status: blocked until the deterministic regeneration report is adjudicated
- Structured contract key: `metadata.acceptance_contract_v1`
- Validator: `devtools lab policy acceptance-contracts --manifest docs/plans/beads-acceptance-contracts-2026-08-07.txt`

The validator does not judge arbitrary natural-language process prose. It validates typed route authority, outcome/evidence/verification/anti-vacuity/safety/closure fields, a managed focused/default verification route for implementation and test contracts, and a positive live-operation receipt carrier with archive, operation, target, before-state, after-state, and result-status bindings. It recomputes a scope-bearing source SHA-256 with a stable dependency projection, plus a separate dependency digest. Lifecycle status and timestamps are deliberately excluded. Human-readable acceptance criteria must be an exact rendering of the structured contract, including the partial-closure successor rule. High and medium contracts with structurally truncated source spans are dispatch-blocked.

The `devtools lab policy acceptance-contracts` gate ratchets the committed manifest at 218 IDs and emits a sorted `regeneration_required` report in JSON mode. The report is the handoff to the later live-authority regeneration and adjudication step; this branch does not rewrite the 218 canonical records. `lab policy bead-graph` consumes the same manifest authority and rejects missing or invalid contracts. `lane-brief` validates the full current Bead record, source digest, dependency digest, and rendered criteria before dispatch. `polylogue-gvzkr` is currently classified as a read-only audit with per-table/per-column dispositions, so this lane does not rewrite it.

Sparse records remain marked `confidence=planner-review`; that flag is not permission for a Luna implementation worker to make architectural choices. It requires planner review before dispatch.
