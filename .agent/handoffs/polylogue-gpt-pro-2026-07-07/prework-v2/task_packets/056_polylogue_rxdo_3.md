# 056. polylogue-rxdo.3 — Query-run + result-relation telemetry in ops.db; refs on every query envelope

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Every COMMITTED query execution (CLI, MCP, daemon web, API) records an ops.db query_runs row (actor, surface, verb, request+lowered spec, archive epoch, timing, status, degraded state) and a result fingerprint + bounded sample refs; the query response envelope gains query_run_ref + result_set_ref + grain + count_precision. Previews/keystrokes are NEVER persisted (ephemeral preview id only, unless a debug flag). This is what lets Polylogue analyze its own use (query-runs where actor_kind:agent and status:failed) and is a pure ops-tier change: no migration ceremony, disposable, high volume.

## Existing design note

ops.db is disposable so long-lived citations must not point here without promotion; expired query-run refs resolve to a typed expired-operational-ref payload, never silently vanish. Promotion path (pin/promote to user.db manifest) is the bridge to the durable bead. Envelope change is additive to SearchEnvelope for byte-compat. Wire at the shared execution chokepoint, not per-surface (t46 direction: contracts own surfaces).

## Acceptance criteria

CLI --json and MCP query responses carry the three refs for the same committed query (parity test); routine preview typing produces zero rows; a promoted run survives ops.db reset. Verify: focused envelope tests + parity test.

## Static mechanism / likely defect

Issue description localizes the mechanism: Every COMMITTED query execution (CLI, MCP, daemon web, API) records an ops.db query_runs row (actor, surface, verb, request+lowered spec, archive epoch, timing, status, degraded state) and a result fingerprint + bounded sample refs; the query response envelope gains query_run_ref + result_set_ref + grain + count_precision. Previews/keystrokes are NEVER persisted (ephemeral preview id only, unless a debug flag). This is what lets Polylogue analyze its own use (query-runs where actor_kind:agent and status:failed) an… Design direction: ops.db is disposable so long-lived citations must not point here without promotion; expired query-run refs resolve to a typed expired-operational-ref payload, never silently vanish. Promotion path (pin/promote to user.db manifest) is the bridge to the durable bead. Envelope change is additive to SearchEnvelope for byte-compat. Wire at the shared execution chokepoint, not per-surface (t46 direction: contracts own sur…

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

1. ops.db is disposable so long-lived citations must not point here without promotion
2. expired query-run refs resolve to a typed expired-operational-ref payload, never silently vanish.
3. Promotion path (pin/promote to user.db manifest) is the bridge to the durable bead.
4. Envelope change is additive to SearchEnvelope for byte-compat.
5. Wire at the shared execution chokepoint, not per-surface (t46 direction: contracts own surfaces).

## Tests to add

- Acceptance proof: CLI --json and MCP query responses carry the three refs for the same committed query (parity test)
- Acceptance proof: routine preview typing produces zero rows
- Acceptance proof: a promoted run survives ops.db reset.
- Acceptance proof: Verify: focused envelope tests + parity test.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
