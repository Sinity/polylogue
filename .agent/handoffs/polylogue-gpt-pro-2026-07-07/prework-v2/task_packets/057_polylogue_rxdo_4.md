# 057. polylogue-rxdo.4 — AssertionKind.FINDING: claims with evidence, reusing the candidate->judge lifecycle verbatim

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

A finding is a durable claim (n, statistic, query_ref, result_set_ref, expected) produced by a detector/agent/analysis, stored as an assertion row so the ENTIRE existing lifecycle (candidate default, judge_assertion_candidate accept/reject/defer/supersede, judgment recorded as assertion, promotion flips inject gate) is reused with zero new lifecycle code — exactly the pathology pattern in user_write.py. CORRECTION to the corpus design it derives from: adding the enum member needs NO user-tier migration (AssertionKind column is schema-free TEXT by design); the real costs are the registration traps: render openapi + render cli-output-schemas regeneration, ASSERTION_CLAIM_KINDS inclusion, and a user_audit surface entry.

## Existing design note

value_json schema polylogue.finding.v1: finding_kind (query-delta | finding-drift | measure | pathology | claim-vs-evidence), statistic {op,value,unit}, n, query_ref, result_set_ref, baseline/current refs, optional expected {measure,op,value}. Defaults: status=candidate, visibility=private, context_policy={"inject":false,"promotion_required":true} — machine findings NEVER auto-inject (recursive-safety spine). Deterministic finding id = hash(claim key + target + value + sorted evidence refs + detector ref) so re-materialization cannot duplicate. Evidence laundering guard: findings carry source query/result refs so a report renderer can warn on circular evidence ancestry.

## Acceptance criteria

upsert_findings_as_assertions mirrors the pathology writer; findings appear in the judgment queue; finding refs resolve; regenerated schemas + user_audit pass; a re-run with identical inputs produces zero new rows. Verify: focused user_write tests + render all --check (grep out-of-sync).

## Static mechanism / likely defect

Issue description localizes the mechanism: A finding is a durable claim (n, statistic, query_ref, result_set_ref, expected) produced by a detector/agent/analysis, stored as an assertion row so the ENTIRE existing lifecycle (candidate default, judge_assertion_candidate accept/reject/defer/supersede, judgment recorded as assertion, promotion flips inject gate) is reused with zero new lifecycle code — exactly the pathology pattern in user_write.py. CORRECTION to the corpus design it derives from: adding the enum member needs NO user-tier migration (AssertionK… Design direction: value_json schema polylogue.finding.v1: finding_kind (query-delta | finding-drift | measure | pathology | claim-vs-evidence), statistic {op,value,unit}, n, query_ref, result_set_ref, baseline/current refs, optional expected {measure,op,value}. Defaults: status=candidate, visibility=private, context_policy={"inject":false,"promotion_required":true} — machine findings NEVER auto-inject (recursive-safety spine). Determ…

## Source anchors to inspect first

- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.

## Implementation plan

1. value_json schema polylogue.finding.v1: finding_kind (query-delta | finding-drift | measure | pathology | claim-vs-evidence), statistic {op,value,unit}, n, query_ref, result_set_ref, baseline/current refs, optional expected {measure,op,value}.
2. Defaults: status=candidate, visibility=private, context_policy={"inject":false,"promotion_required":true} — machine findings NEVER auto-inject (recursive-safety spine).
3. Deterministic finding id = hash(claim key + target + value + sorted evidence refs + detector ref) so re-materialization cannot duplicate.
4. Evidence laundering guard: findings carry source query/result refs so a report renderer can warn on circular evidence ancestry.

## Tests to add

- Acceptance proof: upsert_findings_as_assertions mirrors the pathology writer
- Acceptance proof: findings appear in the judgment queue
- Acceptance proof: finding refs resolve
- Acceptance proof: regenerated schemas + user_audit pass
- Acceptance proof: a re-run with identical inputs produces zero new rows.
- Acceptance proof: Verify: focused user_write tests + render all --check (grep out-of-sync).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
