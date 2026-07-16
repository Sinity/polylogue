# 059. polylogue-bby.15 — Evidence basket -> citable report -> verified export (cockpit core loop)

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **blocked-hard**.

Hard blockers: polylogue-svfj

## What the bead says

The missing "report" end of the web workbench: select blocks/spans in the reader -> basket (content-hash anchors + quote + note + provenance of the query that surfaced it) -> live Markdown report draft with footnotes -> EXPORT GATE re-resolves every citation and blocks/flags by state (ok + drifted_position export with verified note; drifted_message/relocated need explicit promotion; ambiguous/missing block by default; quarantined blocks unless the report is explicitly forensic; hash_mismatch hard-fails). Storage v1 rides recall-pack machinery with an evidence_basket payload schema (items resolve/degrade counts already exist) — UI names it basket, storage adapter is an implementation detail; dedicated AssertionKinds (evidence_basket, report_draft) deliberately deferred until the shape settles (each new kind costs openapi/cli-schema regen + user_audit entry). Report exports emit Markdown/HTML + a citation manifest JSON.

## Existing design note

Three-pane cockpit flow (results | reader+graph | basket+draft); daemon API basket/report/verify routes collapse into service verbs when the t46/B8 contract lands. Depends on the block content-hash anchor substrate. Batch overlay endpoint (assertions/marks for a set of refs) serves the reader badges.

## Acceptance criteria

Full loop on the seeded demo corpus: query -> basket 5 items -> draft renders footnotes -> re-ingest the corpus -> verify flags the drifted item and export annotates it; a deleted block blocks export with a typed reason. Verify: integration-flavored test over the loop.

## Static mechanism / likely defect

Issue description localizes the mechanism: The missing "report" end of the web workbench: select blocks/spans in the reader -> basket (content-hash anchors + quote + note + provenance of the query that surfaced it) -> live Markdown report draft with footnotes -> EXPORT GATE re-resolves every citation and blocks/flags by state (ok + drifted_position export with verified note; drifted_message/relocated need explicit promotion; ambiguous/missing block by default; quarantined blocks unless the report is explicitly forensic; hash_mismatch hard-fails). Storage v… Design direction: Three-pane cockpit flow (results | reader+graph | basket+draft); daemon API basket/report/verify routes collapse into service verbs when the t46/B8 contract lands. Depends on the block content-hash anchor substrate. Batch overlay endpoint (assertions/marks for a set of refs) serves the reader badges.

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

1. Three-pane cockpit flow (results | reader+graph | basket+draft)
2. daemon API basket/report/verify routes collapse into service verbs when the t46/B8 contract lands.
3. Depends on the block content-hash anchor substrate.
4. Batch overlay endpoint (assertions/marks for a set of refs) serves the reader badges.

## Tests to add

- Acceptance proof: Full loop on the seeded demo corpus: query -> basket 5 items -> draft renders footnotes -> re-ingest the corpus -> verify flags the drifted item and export annotates it
- Acceptance proof: a deleted block blocks export with a typed reason.
- Acceptance proof: Verify: integration-flavored test over the loop.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
