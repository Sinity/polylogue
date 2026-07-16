# 001. polylogue-s7ae.6 — Classify the 74%-aborted full verify from the coordination commit before deploy

Priority/type/status: **P1 / task / in_progress**. Lane: **00-trust-floor**. Release: **A-trust-floor**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Commit 32ff31651 (coordination substrate, ~1376 LOC) merged with only verify --quick + focused tests green; the full devtools verify was aborted at 74% with scattered unclassified failures. Until each failure is classified coordination-caused vs pre-existing, the deploy gate for s7ae stays closed — unclassified inherited failure state is exactly what the verification doctrine forbids shipping on.

## Existing design note

Commit 32ff31651 shipped ~1376 LOC with only verify --quick + focused tests green; full devtools verify was aborted at 74% with unclassified scattered failures. Before any deploy/switch, run full devtools verify and classify each failure coordination-caused vs pre-existing/flaky.

## Acceptance criteria

A full devtools verify run is recorded; every failure classified (coordination-caused fixed; pre-existing referenced); s7ae deploy-clean. Verify: devtools verify (full).

## Static mechanism / likely defect

Release-gate debt rather than one code defect: a full `devtools verify` stopped at 74%, so coordination deployment has unknown blast radius until every failing lane is classified.

## Source anchors to inspect first

- `polylogue/coordination/envelope.py` — Coordination envelope model exists; harden it as the shared payload.
- `polylogue/coordination/payloads.py` — Coordination payload types should stay small and evidence-ref oriented.
- `polylogue/coordination/rendering.py` — Rendered advisories should be scheduler-mediated, not chat spam.
- `tests/unit/coordination/test_envelope.py` — Existing envelope tests are the starting verification lane.
- `polylogue/mcp/server_prompts.py:219` — MCP prompt registration exists and can surface cookbook/roles.
- `polylogue/cli/commands/agents.py` — CLI agent commands are the operator-facing entry point.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:31` — ASSERTION_DEFAULT_STATUS is ACTIVE, so missing status currently means trusted active.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641` — upsert_blackboard_note passes author_kind and no explicit status into upsert_assertion.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — upsert_assertion is the single write chokepoint to patch.
- `polylogue/api/contracts/assertions.py` — Check public assertion request/response contract after changing default status behavior.

## Implementation plan

1. Run the exact full verification lane on the current commit and save raw logs under `.agent/reports/` or `docs/audits/`.
2. Build a failure-classification table: lane, command, failure signature, first bad commit if known, coordination-caused/pre-existing/flaky, owner bead, fix/defer decision.
3. For any coordination-caused failure, land a focused fix before deploying MCP/hook/coordination surfaces.
4. For pre-existing failures, create/update a bead and make the deployment verdict explicitly conditional.

## Tests to add

- The proof is the full verify log plus classification ledger; no code unit test is enough for this packet.

## Verification commands

- `Run `devtools verify` full. Preserve the log artifact. The gate only opens when every failure has a table row and every coordination-caused failure is fixed or explicitly blocks deployment.`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
