# 058. polylogue-rxdo.7 — Annotation substrate: schema registry, annotation batches, JSONL import surface, typed value predicates

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **blocked-hard**.

Hard blockers: polylogue-37t.15, polylogue-rxdo.1

## What the bead says

The missing loop for external-agent analysis: export evidence pack -> agent labels rows under a declared schema -> import as candidate assertions -> query them back -> judge -> report. Storage is ~75% ready (assertions table + upsert + judge lifecycle all real, verified); what is missing: (1) a general import surface (act kind / MCP tool / CLI assertions import) accepting JSONL rows with full assertion shape, defaulting status=candidate + inject:false for agent authors; (2) an annotation_schemas registry declaring value shape, target grain, required-evidence policy, abstain value — without it labels are queryable blobs, not analytical variables; (3) annotation_batch as the provenance container (schema id, source result ref, actor/model/prompt refs, counts, validation failures) — batches are containers, rows stay assertions; (4) typed JSON-path predicates over assertion values (value.score>=4), since substring match cannot express label analytics. Query-back gap is real: assertions are a DSL unit but MCP-list-only for rich shapes today.

## Existing design note

Schemas connect to the 9l5.7 measure-registry discipline: a label is an operationalization with construct-validity metadata, not just a JSON key. Trusted-schema auto-active is explicitly rejected for v1: ALL external-agent rows enter candidate (recursive-safety chokepoint in upsert_assertion — author_kind != user => CANDIDATE + inject:false — is a related but separate load-bearing bead in the safety program). Import validates refs against the archive, reports per-row failures, refuses rows without evidence refs when the schema demands them.

## Acceptance criteria

Roundtrip demo: export a bounded evidence pack, import 5 labeled rows as candidates, query them via assertions where with a typed value predicate, judge one active, render. Batch metadata queryable. Verify: integration-flavored focused test + MCP tool contract test (EXPECTED_TOOL_NAMES + contract + regen).

## Static mechanism / likely defect

Issue description localizes the mechanism: The missing loop for external-agent analysis: export evidence pack -> agent labels rows under a declared schema -> import as candidate assertions -> query them back -> judge -> report. Storage is ~75% ready (assertions table + upsert + judge lifecycle all real, verified); what is missing: (1) a general import surface (act kind / MCP tool / CLI assertions import) accepting JSONL rows with full assertion shape, defaulting status=candidate + inject:false for agent authors; (2) an annotation_schemas registry declaring… Design direction: Schemas connect to the 9l5.7 measure-registry discipline: a label is an operationalization with construct-validity metadata, not just a JSON key. Trusted-schema auto-active is explicitly rejected for v1: ALL external-agent rows enter candidate (recursive-safety chokepoint in upsert_assertion — author_kind != user => CANDIDATE + inject:false — is a related but separate load-bearing bead in the safety program). Import…

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

1. Schemas connect to the 9l5.7 measure-registry discipline: a label is an operationalization with construct-validity metadata, not just a JSON key.
2. Trusted-schema auto-active is explicitly rejected for v1: ALL external-agent rows enter candidate (recursive-safety chokepoint in upsert_assertion — author_kind != user => CANDIDATE + inject:false — is a related but separate load-bearing bead in the safety program).
3. Import validates refs against the archive, reports per-row failures, refuses rows without evidence refs when the schema demands them.

## Tests to add

- Acceptance proof: Roundtrip demo: export a bounded evidence pack, import 5 labeled rows as candidates, query them via assertions where with a typed value predicate, judge one active, render.
- Acceptance proof: Batch metadata queryable.
- Acceptance proof: Verify: integration-flavored focused test + MCP tool contract test (EXPECTED_TOOL_NAMES + contract + regen).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
