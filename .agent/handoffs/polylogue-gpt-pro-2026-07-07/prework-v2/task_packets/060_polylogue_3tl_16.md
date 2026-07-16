# 060. polylogue-3tl.16 — Public claims ledger: every README/launch claim carries a status and an evidence ref

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Turn radical honesty into a product surface: every public claim (README, docs site, launch post, category one-liner) must be exactly one of proven (backed by a finding/proof artifact), capability (code exists, no measured-result claim), aspirational (roadmap only), or retired (no longer true). This is the discipline that keeps the flight-recorder positioning from becoming marketing fog — the product whose pitch is 'every metric resolves to bytes' cannot itself ship unresolvable claims. Complements 3tl.12 (README de-persuasion pass) by making the honesty machine-checkable instead of a one-time edit.

## Existing design note

A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date. README/docs quantitative claims link to a ledger entry by id. CI lint: a quantitative or comparative public claim without a ledger ref fails; a ledger entry with status=proven whose evidence ref does not resolve fails. Upgrade path: ledger entries become user.db findings once analysis provenance (rxdo) exists, so public claims share the same lifecycle as internal findings.

## Acceptance criteria

claims.yml exists and covers every quantitative/comparative claim in README + docs site; CI gate rejects unreferenced claims; each status has at least one real entry or an explicit none; the flight-recorder category claim itself is ledgered (initially capability, not proven). Verify: the CI lint run + a grep sweep of README claims against ledger ids.

## Static mechanism / likely defect

Issue description localizes the mechanism: Turn radical honesty into a product surface: every public claim (README, docs site, launch post, category one-liner) must be exactly one of proven (backed by a finding/proof artifact), capability (code exists, no measured-result claim), aspirational (roadmap only), or retired (no longer true). This is the discipline that keeps the flight-recorder positioning from becoming marketing fog — the product whose pitch is 'every metric resolves to bytes' cannot itself ship unresolvable claims. Complements 3tl.12 (README d… Design direction: A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date. README/docs quantitative claims link to a ledger entry by id. CI lint: a quantitative or comparative public claim without a ledger ref fails; a ledger entry with status=proven whose evidence ref does not resolve fails. Upgrad…

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

1. A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date.
2. README/docs quantitative claims link to a ledger entry by id.
3. CI lint: a quantitative or comparative public claim without a ledger ref fails
4. a ledger entry with status=proven whose evidence ref does not resolve fails.
5. Upgrade path: ledger entries become user.db findings once analysis provenance (rxdo) exists, so public claims share the same lifecycle as internal findings.

## Tests to add

- Acceptance proof: claims.yml exists and covers every quantitative/comparative claim in README + docs site
- Acceptance proof: CI gate rejects unreferenced claims
- Acceptance proof: each status has at least one real entry or an explicit none
- Acceptance proof: the flight-recorder category claim itself is ledgered (initially capability, not proven).
- Acceptance proof: Verify: the CI lint run + a grep sweep of README claims against ledger ids.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
