# 192. polylogue-avg — Fold devloop claim-guard vocabulary upstream into ops status/readiness

Priority/type/status: **P2 / feature / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

The loop scripts guard claims better than the product does: devloop-status treats schema-version match as 'openable, not converged', gates convergence claims on raw-materialization debt being zero/classified, and blocks latency claims behind live_performance_proof_blocked. polylogue ops status should expose the same claim-guard vocabulary to ordinary users: a 'what you may claim' section (archive openable / converged / search-ready / perf-measurable) derived from the same signals, instead of leaving the discipline in a loop script. Then devloop-status consumes the product surface instead of computing its own (silo collapse).

## Existing design note

Add a claim-guard section to `polylogue ops status`/readiness that derives 'what you may claim' (archive openable / converged / search-ready / perf-measurable) from the same signals the devloop scripts already use: schema-version match => openable-not-converged; raw-materialization debt zero/classified => converged; FTS freshness => search-ready; the live_performance_proof_blocked gate => perf-measurable. Then have devloop-status consume the product surface instead of recomputing its own claim vocabulary (silo collapse).

## Acceptance criteria

- `polylogue ops status --json` exposes a claim-guard block with the four claim states (openable / converged / search-ready / perf-measurable), each derived from its documented signal. Verify: run the command and assert the block and derivations.
- An archive that is openable-but-not-converged reports converged=false with the raw-materialization reason string. Verify: test seeds unmaterialized raw debt and checks the reason.
- devloop-status calls the product surface and stops computing its own claim vocabulary. Verify: grep shows the duplicated claim logic removed from devloop-status.
- A parity test asserts the script's old computation and the product output agree over a fixed set of archive states.

## Static mechanism / likely defect

Issue description localizes the mechanism: The loop scripts guard claims better than the product does: devloop-status treats schema-version match as 'openable, not converged', gates convergence claims on raw-materialization debt being zero/classified, and blocks latency claims behind live_performance_proof_blocked. polylogue ops status should expose the same claim-guard vocabulary to ordinary users: a 'what you may claim' section (archive openable / converged / search-ready / perf-measurable) derived from the same signals, instead of leaving the discipline… Design direction: Add a claim-guard section to `polylogue ops status`/readiness that derives 'what you may claim' (archive openable / converged / search-ready / perf-measurable) from the same signals the devloop scripts already use: schema-version match => openable-not-converged; raw-materialization debt zero/classified => converged; FTS freshness => search-ready; the live_performance_proof_blocked gate => perf-measurable. Then have …

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. Add a claim-guard section to `polylogue ops status`/readiness that derives 'what you may claim' (archive openable / converged / search-ready / perf-measurable) from the same signals the devloop scripts already use: schema-version match => openable-not-converged
2. raw-materialization debt zero/classified => converged
3. FTS freshness => search-ready
4. the live_performance_proof_blocked gate => perf-measurable.
5. Then have devloop-status consume the product surface instead of recomputing its own claim vocabulary (silo collapse).

## Tests to add

- Acceptance proof: `polylogue ops status --json` exposes a claim-guard block with the four claim states (openable / converged / search-ready / perf-measurable), each derived from its documented signal.
- Acceptance proof: Verify: run the command and assert the block and derivations.
- Acceptance proof: An archive that is openable-but-not-converged reports converged=false with the raw-materialization reason string.
- Acceptance proof: Verify: test seeds unmaterialized raw debt and checks the reason.
- Acceptance proof: devloop-status calls the product surface and stops computing its own claim vocabulary.
- Acceptance proof: Verify: grep shows the duplicated claim logic removed from devloop-status.
- Acceptance proof: A parity test asserts the script's old computation and the product output agree over a fixed set of archive states.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
