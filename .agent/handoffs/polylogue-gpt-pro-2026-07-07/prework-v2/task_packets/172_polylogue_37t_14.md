# 172. polylogue-37t.14 — Recursive-safety substrate: citation anchors, provenance edges, grounding verdicts (closed-loop/cycle/drift)

Priority/type/status: **P2 / task / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

THE load-bearing safety invariant for a self-ingesting archive (browser capture auto-ingests the operator R&D chats; distilled findings become assertions; assertions can inject into future context — the recovery-digest fabrication class generalizes to: an agent claim laundered through other agent claims until it re-enters context as truth). Substrate: assertion_citation_anchors (evidence ref resolved to typed anchor with grounding_class {human_message, human_judgment, tool_result, source_raw, git_commit, external_pr/issue/doc, agent_session, assertion, unknown} + COMPATIBLE_CLAIM bit + content hash pair + drift flag), assertion_provenance_edges (cites edges, quarantinable, reusing the TopologyEdgeStatus vocabulary but NOT the session_links table), assertion_grounding_verdicts (materialized by a converger stage: grounded | closed_loop | cycle | drifted | unknown via recursive CTE over the closure). KEY INSIGHT beyond the flag-level design: compatible_claim is what makes this construct-valid — a raw transcript proves "assistant SAID X", never X itself; a transcript anchor cannot ground a pr-merged claim. Closed-loop predicate: agent-authored claim whose closure reaches NO compatible non-drifted external grounding => quarantined from injection until an external citation or human judgment breaks the loop. Missing/uncomputable observed hash = unknown = NOT fresh (fail closed).

## Existing design note

Derived verdict tier (fail-closed when absent) + durable policy mutation on quarantine (candidate + inject:false + quarantine_reason in context_policy; NO new AssertionStatus axis — reuse candidate machinery). Cycle detection: recursive CTE with path-string membership, depth backstop; quarantined edges block auto-promotion. Drift: anchors store resolved-against hash; converger pass flags mismatches and forces inject:false + staleness.hash_drift. Promotion release function: accepted candidates do NOT inherit inject — injection is a second explicit decision (judge --accept --inject or set-policy). Too-tight guards to avoid: one compatible external anchor on SOME path suffices (not every citation); operator_command / human messages inside captured chats count as grounding for said-claims; deterministic detectors over structured tool-result evidence may rank as tool_result class.

## Acceptance criteria

The laundering scenario is structurally blocked in a test: agent assertion citing only agent sessions/assertions never appears in compiled context; adding a git/tool-result/human-judgment anchor or judging it releases it; cycle + drift each independently quarantine; a transcript-only anchor cannot release a world-claim (compatibility matrix test). Verify: focused tests over the CTE + scheduler gate fixture.

## Static mechanism / likely defect

Issue description localizes the mechanism: THE load-bearing safety invariant for a self-ingesting archive (browser capture auto-ingests the operator R&D chats; distilled findings become assertions; assertions can inject into future context — the recovery-digest fabrication class generalizes to: an agent claim laundered through other agent claims until it re-enters context as truth). Substrate: assertion_citation_anchors (evidence ref resolved to typed anchor with grounding_class {human_message, human_judgment, tool_result, source_raw, git_commit, external_… Design direction: Derived verdict tier (fail-closed when absent) + durable policy mutation on quarantine (candidate + inject:false + quarantine_reason in context_policy; NO new AssertionStatus axis — reuse candidate machinery). Cycle detection: recursive CTE with path-string membership, depth backstop; quarantined edges block auto-promotion. Drift: anchors store resolved-against hash; converger pass flags mismatches and forces inject…

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

1. Derived verdict tier (fail-closed when absent) + durable policy mutation on quarantine (candidate + inject:false + quarantine_reason in context_policy
2. NO new AssertionStatus axis — reuse candidate machinery).
3. Cycle detection: recursive CTE with path-string membership, depth backstop
4. quarantined edges block auto-promotion.
5. Drift: anchors store resolved-against hash
6. converger pass flags mismatches and forces inject:false + staleness.hash_drift.
7. Promotion release function: accepted candidates do NOT inherit inject — injection is a second explicit decision (judge --accept --inject or set-policy).

## Tests to add

- Acceptance proof: The laundering scenario is structurally blocked in a test: agent assertion citing only agent sessions/assertions never appears in compiled context
- Acceptance proof: adding a git/tool-result/human-judgment anchor or judging it releases it
- Acceptance proof: cycle + drift each independently quarantine
- Acceptance proof: a transcript-only anchor cannot release a world-claim (compatibility matrix test).
- Acceptance proof: Verify: focused tests over the CTE + scheduler gate fixture.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
