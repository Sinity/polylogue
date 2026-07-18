# Lane D support — raw-authority crash/restart fixed-point proof harness

Work from the attached fresh Chisel archive. Produce an integration-ready
implementation package for a deterministic raw-authority repair proof harness.
The paired local lane owns the real archive clone, current preflight, and any
operator-authorized live action.

## Mission

Extend the existing raw-authority/reconciler test and proof infrastructure with
a compact synthetic topology that has multiple replay-plan components,
membership siblings, deferrals, and durable census receipts. Add a fault
injection matrix at real transaction boundaries: before an outcome commit,
after an outcome commit but before census finalization, and during a resumed
batch. The proof must establish, through the production reconciler and durable
ledger, that restart reaches a two-census quiescent fixed point and that every
immutable plan component is represented exactly once in a terminal/conserved
outcome.

Tests must also demonstrate that a deliberately broken conservation or
postcondition mutation fails the harness. Reuse the real repair APIs and
receipt/census storage; do not build a parallel model that merely asserts its
own simulated state.

## Boundary

Do not run against the live archive, add an operational bypass, or authorize an
apply. Do not reduce production limits merely to make the proof convenient.

## Required package

Return `HANDOFF.md`, `PATCH.diff`, `TESTS.md`, and `EVIDENCE.md`. Include the
fault matrix, exact source anchors, conservation law, and the local-only
preflight inputs still required before the live lane can make any decision.
