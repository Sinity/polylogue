# 01. polylogue-s7ae.6 — Classify the aborted full verification run before coordination deploy

Priority: **P1**  
Lane: **release-gate**  
Readiness: **ready-now / evidence-work**

## Why this is urgent / critical-path

The coordination commit was merged after quick/focused verification, while a full `devtools verify` run was aborted at 74%. Until each failure is classified as caused-by-coordination, pre-existing, flaky, or fixed, every coordination/deployment packet inherits unknown risk.

## Static diagnosis / likely mechanism

This is not primarily a code bug. It is a release-gate debt packet. The static implication is that any live deployment of coordination, scheduler, hook, or MCP surfaces must wait for a fresh full verify log with a failure-classification table. Do not let later packets cite a green quick lane as deploy-clean.

## Implementation plan

Create `docs/audits/coordination-full-verify-classification.md` or `.agent/reports/coordination-full-verify-classification.md` with: command, git sha, environment, start/end time, full output path, failure table, owner bead for each failure, and final deploy verdict. If a failure is coordination-caused, fix it in the same PR or file a blocker bead with a minimal repro. If it is pre-existing, cite the pre-existing bead/issue and show why coordination did not widen it.

## Test plan

No new product test is required unless full verify exposes a real regression. Add a tiny regression only for any failure fixed during the classification.

## Verification command / proof

Run `devtools verify` full. Preserve the log artifact. The gate only opens when every failure has a table row and every coordination-caused failure is fixed or explicitly blocks deployment.

## Pitfalls

Do not silently downgrade this to `verify --quick`. Do not close it on a partial run. The output of this packet is a release decision, not a vibes report.

## Files/functions to inspect or touch

- `devtools/verify*`
- `docs/audits/ or .agent/reports/`
