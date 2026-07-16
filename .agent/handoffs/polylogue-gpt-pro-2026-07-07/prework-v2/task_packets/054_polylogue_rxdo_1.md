# 054. polylogue-rxdo.1 — ObjectRef expansion: query, query-run, result-set, finding, cohort, analysis, annotation-batch kinds

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Verified live 2026-07-06: ObjectRefKind in core/refs.py is a closed Literal of 29 kinds with none of the analysis-object kinds; normalize_object_ref_text rejects unknown kinds, so nothing can target a query or result set today. This is the narrow prerequisite for the whole analysis-provenance epic: refs first, resolvers stubbed (typed unresolved payload until tables land), tables second.

## Existing design note

Add kinds: query, query-run, result-set, finding, cohort, analysis, annotation-batch to ObjectRefKind + the kind map + normalize paths in core/refs.py. finding:<hash> resolves to the assertion row with kind=finding (assertion:<id> stays valid; finding is the public alias). resolve_ref dispatch gains stub branches returning typed unresolved payloads with reason=substrate-pending until the storage beads land. Registered-kind hygiene: each new kind needs a user_audit surface entry and regenerated render openapi + cli-output-schemas or the every-kind audit invariant fails (known registration trap). Do NOT bundle the @content-hash anchor suffix here — that belongs to the citation-anchor work (bby.11 block_content_hash).

## Acceptance criteria

normalize_object_ref_text accepts the new kinds; resolve_ref returns typed pending payloads for them; user_audit + rendered schemas regenerated; existing ref tests extended. Verify: devtools verify (testmon) + rg for the kind literals across surface schemas.

## Static mechanism / likely defect

Issue description localizes the mechanism: Verified live 2026-07-06: ObjectRefKind in core/refs.py is a closed Literal of 29 kinds with none of the analysis-object kinds; normalize_object_ref_text rejects unknown kinds, so nothing can target a query or result set today. This is the narrow prerequisite for the whole analysis-provenance epic: refs first, resolvers stubbed (typed unresolved payload until tables land), tables second. Design direction: Add kinds: query, query-run, result-set, finding, cohort, analysis, annotation-batch to ObjectRefKind + the kind map + normalize paths in core/refs.py. finding:<hash> resolves to the assertion row with kind=finding (assertion:<id> stays valid; finding is the public alias). resolve_ref dispatch gains stub branches returning typed unresolved payloads with reason=substrate-pending until the storage beads land. Register…

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

1. Add kinds: query, query-run, result-set, finding, cohort, analysis, annotation-batch to ObjectRefKind + the kind map + normalize paths in core/refs.py.
2. finding:<hash> resolves to the assertion row with kind=finding (assertion:<id> stays valid
3. finding is the public alias).
4. resolve_ref dispatch gains stub branches returning typed unresolved payloads with reason=substrate-pending until the storage beads land.
5. Registered-kind hygiene: each new kind needs a user_audit surface entry and regenerated render openapi + cli-output-schemas or the every-kind audit invariant fails (known registration trap).
6. Do NOT bundle the @content-hash anchor suffix here — that belongs to the citation-anchor work (bby.11 block_content_hash).

## Tests to add

- Acceptance proof: normalize_object_ref_text accepts the new kinds
- Acceptance proof: resolve_ref returns typed pending payloads for them
- Acceptance proof: user_audit + rendered schemas regenerated
- Acceptance proof: existing ref tests extended.
- Acceptance proof: Verify: devtools verify (testmon) + rg for the kind literals across surface schemas.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
