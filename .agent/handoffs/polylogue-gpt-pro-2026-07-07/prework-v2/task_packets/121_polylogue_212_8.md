# 121. polylogue-212.8 — The honesty anti-demo: a tempting finding that emits verdict not_supported

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Ship a demo whose SUCCESS is refusal: attempt a tempting claim (e.g. minute-by-minute multi-source operator reconstruction) and emit the standard packet with verdict: not_supported, listing missing modalities, missing refs, and the exact query/evidence gap. Published BESIDE the successful demos, not hidden — this is the brand ("refuses rather than fabricates") made demonstrable, and it directly encodes the situation-brief praise for the honest deferral of the multi-source demo. Framing decision for operator in 212 notes: general "no unsupported number is published" vs concrete "multi-source reconstruction is not ready".

## Existing design note

Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is not_supported. Pick the tempting claim: minute-by-minute multi-source operator reconstruction (needs modalities the archive lacks). The packet lists missing modalities, missing refs, and the exact query/evidence that WOULD support it, using the same finding.yaml shape. Anchor: .agent/demos/<new-dir>/ + the insight_rigor_audit surface to enumerate what evidence exists vs required. The success criterion is the refusal being specific, not vague: every missing item names the unit/table/modality that would have to exist.

## Acceptance criteria

Anti-demo packet passes the packet lint with not_supported verdict; report names each missing capability with the bead ref that would supply it; included in the registry manifest and the public mini-portfolio. Verify: runner emits + lint passes.

## Static mechanism / likely defect

Issue description localizes the mechanism: Ship a demo whose SUCCESS is refusal: attempt a tempting claim (e.g. minute-by-minute multi-source operator reconstruction) and emit the standard packet with verdict: not_supported, listing missing modalities, missing refs, and the exact query/evidence gap. Published BESIDE the successful demos, not hidden — this is the brand ("refuses rather than fabricates") made demonstrable, and it directly encodes the situation-brief praise for the honest deferral of the multi-source demo. Framing decision for operator in 212… Design direction: Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is not_supported. Pick the tempting claim: minute-by-minute multi-source operator reconstruction (needs modalities the archive lacks). The packet lists missing modalities, missing refs, and the exact query/evidence that WOULD support it, using the same finding.yaml shape. Anchor: .agent/demos/<new-dir>/ + the insight_rigor_audit …

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

1. Depends on the 212.7 packet contract — this demo is one more packet whose verdict field is not_supported.
2. Pick the tempting claim: minute-by-minute multi-source operator reconstruction (needs modalities the archive lacks).
3. The packet lists missing modalities, missing refs, and the exact query/evidence that WOULD support it, using the same finding.yaml shape.
4. Anchor: .agent/demos/<new-dir>/ + the insight_rigor_audit surface to enumerate what evidence exists vs required.
5. The success criterion is the refusal being specific, not vague: every missing item names the unit/table/modality that would have to exist.

## Tests to add

- Acceptance proof: Anti-demo packet passes the packet lint with not_supported verdict
- Acceptance proof: report names each missing capability with the bead ref that would supply it
- Acceptance proof: included in the registry manifest and the public mini-portfolio.
- Acceptance proof: Verify: runner emits + lint passes.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
