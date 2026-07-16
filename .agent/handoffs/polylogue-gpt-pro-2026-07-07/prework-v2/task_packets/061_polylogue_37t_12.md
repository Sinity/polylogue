# 061. polylogue-37t.12 — Judgment queue: operator bulk review/accept/reject of candidate assertions

Priority/type/status: **P2 / feature / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

WHY: 37t's loop names four stages (claims -> JUDGMENT -> preamble -> reboot) but only JUDGMENT has no owning bead. 37t.10 design already assumes 'the existing judgment queue lists setup candidates alongside memory candidates'; 37t.11's blocking security note states 'Candidate->judged promotion IS the QUOTED->OPERATOR transition' and that a source without a judgment gate cannot emit OPERATOR-class items. So the judgment surface is a hard prerequisite for those beads yet unbuilt. SUBSTRATE THAT ALREADY EXISTS (verify, do not rebuild): polylogue/storage/sqlite/archive_tiers/user_write.py -- judge_assertion_candidate(candidate_ref, decision in {accept,reject,defer,supersede}, reason, replacement_*) at :1245 records a JUDGMENT-kind assertion (enums.py AssertionKind.JUDGMENT) and, for accept/supersede, promotes via _promote_candidate_assertion at :1338 (candidate->active transition IS speced -- assertion_id_for_promoted_candidate at :263); list_assertion_candidates(:1158) returns CANDIDATE-status claims awaiting judgment; list_assertion_candidate_reviews(:1185) returns candidate rows with their latest judgment (ASSERTION_CANDIDATE_REVIEW_STATUSES at :1176). Async API wrappers exist: Polylogue.judge_assertion_candidate (api/archive.py:2181), list_assertion_claims (:2079). THE GAP (what this bead builds): an operator-ergonomic BULK review surface. (1) MCP: add a judge_assertion_candidate write tool + a list_assertion_candidates read tool to the agent-write/operator MCP role (mcp/server_tools.py currently exposes only read-only list_assertion_claims at :737 with statuses 'active,candidate'; there is no judge tool and no candidate-review tool). Register the tool name in EXPECTED_TOOL_NAMES + a TOOL_CONTRACT entry and mirror the every-tool-surface invariant (see #2436 memory: new MCP tool needs both). (2) CLI: a 'polylogue judge' verb group -- 'judge queue' (list pending candidates, grouped by kind/scope_ref/target_ref with evidence-ref preview), 'judge accept|reject|defer <ref...>' accepting MULTIPLE candidate refs in one call (the bulk gap), and an interactive/batch '--all-kind <kind>' or '--from-file' path so a batch of background-generated candidates is judged in one pass, not one MCP round-trip each. Route through the archive_routed 'user' path like list_assertion_claims (status.py:325). (3) The queue must expose per-candidate: kind, body/value preview, evidence_refs (resolvable via resolve_ref), scope_ref/target_ref, author_kind (agent vs user), created_at. PITFALLS: judge_assertion_candidate raises ValueError if the target is not CANDIDATE status (:1266) -- a bulk call must skip/report already-judged rows, not abort the batch; accept/supersede write a NEW active assertion (id from assertion_id_for_promoted_candidate) so bulk accept must be idempotent under re-run (judging an already-accepted candidate is a no-op, not a duplicate); a rejected candidate stays as a REJECTED row (not deleted) so rebuilds cannot resurrect it -- the queue read must default to statuses=(CANDIDATE,) only. Restrained-injection rule: the queue is a review surface, not a context injector; it does NOT wire into the preamble (that is 37t.4/37t.11 reading the resulting ACTIVE claims).

## Acceptance criteria

MCP: 'judge_assertion_candidate' and 'list_assertion_candidates' tools exist on the operator/agent-write MCP role, listed in EXPECTED_TOOL_NAMES with TOOL_CONTRACT entries; a candidate written by an agent (author_kind='agent', status='candidate') is listable and can be accepted/rejected via the MCP tool, and an accepted candidate produces a new ACTIVE assertion (verify via list_assertion_claims statuses=active). CLI: 'polylogue judge queue' lists pending candidates with kind/evidence-ref preview; 'polylogue judge accept <ref1> <ref2> ...' judges MULTIPLE candidates in one invocation and reports per-ref outcome (accepted / skipped-not-candidate); a bulk accept over a mix of candidate and already-accepted refs succeeds partially without aborting (idempotent). Rejected candidates remain queryable as REJECTED and never reappear in 'judge queue'. Tests: unit test that bulk accept of N candidates promotes exactly N to ACTIVE and writes N JUDGMENT rows; test that a second bulk accept of the same refs is a no-op; test that reject is durable across an index rebuild. Verify: devtools test tests/unit/mcp -k judge ; devtools test tests/unit/cli -k judge ; polylogue judge queue --format json.

## Static mechanism / likely defect

Design direction: WHY: 37t's loop names four stages (claims -> JUDGMENT -> preamble -> reboot) but only JUDGMENT has no owning bead. 37t.10 design already assumes 'the existing judgment queue lists setup candidates alongside memory candidates'; 37t.11's blocking security note states 'Candidate->judged promotion IS the QUOTED->OPERATOR transition' and that a source without a judgment gate cannot emit OPERATOR-class items. So the judgm…

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

1. WHY: 37t's loop names four stages (claims -> JUDGMENT -> preamble -> reboot) but only JUDGMENT has no owning bead.
2. 37t.10 design already assumes 'the existing judgment queue lists setup candidates alongside memory candidates'
3. 37t.11's blocking security note states 'Candidate->judged promotion IS the QUOTED->OPERATOR transition' and that a source without a judgment gate cannot emit OPERATOR-class items.
4. So the judgment surface is a hard prerequisite for those beads yet unbuilt.
5. SUBSTRATE THAT ALREADY EXISTS (verify, do not rebuild): polylogue/storage/sqlite/archive_tiers/user_write.py -- judge_assertion_candidate(candidate_ref, decision in {accept,reject,defer,supersede}, reason, replacement_*) at :1245 records a JUDGMENT-kind assertion (enums.py AssertionKind.JUDGMENT) and, for accept/supersede, promotes via _promote_candidate_assertion at :1338 (candidate->active transition IS speced -…
6. list_assertion_candidates(:1158) returns CANDIDATE-status claims awaiting judgment
7. list_assertion_candidate_reviews(:1185) returns candidate rows with their latest judgment (ASSERTION_CANDIDATE_REVIEW_STATUSES at :1176).

## Tests to add

- Acceptance proof: MCP: 'judge_assertion_candidate' and 'list_assertion_candidates' tools exist on the operator/agent-write MCP role, listed in EXPECTED_TOOL_NAMES with TOOL_CONTRACT entries
- Acceptance proof: a candidate written by an agent (author_kind='agent', status='candidate') is listable and can be accepted/rejected via the MCP tool, and an accepted candidate produces a new ACTIVE assertion (verify via list_assertion_claims statuses=active).
- Acceptance proof: CLI: 'polylogue judge queue' lists pending candidates with kind/evidence-ref preview
- Acceptance proof: 'polylogue judge accept <ref1> <ref2> ...' judges MULTIPLE candidates in one invocation and reports per-ref outcome (accepted / skipped-not-candidate)
- Acceptance proof: a bulk accept over a mix of candidate and already-accepted refs succeeds partially without aborting (idempotent).
- Acceptance proof: Rejected candidates remain queryable as REJECTED and never reappear in 'judge queue'.
- Acceptance proof: Tests: unit test that bulk accept of N candidates promotes exactly N to ACTIVE and writes N JUDGMENT rows
- Acceptance proof: test that a second bulk accept of the same refs is a no-op

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
