# 053. polylogue-svfj — Block content-hash citation anchors: blocks.content_hash + resolver with typed drift states

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

THE anchor atom multiple programs stand on (webui cockpit citations, finding evidence refs rxdo.4, drift detection 37t.14, compaction loss anchors gjg.3, export citations). Today block identity is message_id:position — position shifts on re-ingest and fork replay, so any stored block citation can silently point at different content. Verified live: blocks table has NO content_hash (sessions and messages do). Add blocks.content_hash (32B, over canonical block EVIDENCE: type, text, tool_name, canonical tool_input, semantic/media/language, is_error, exit_code — deliberately EXCLUDING session/message/position/tool_id so the hash survives fork-position shift, re-ingest, and provider tool-id regeneration) + hash index. Anchor textual form uses the existing :: separator (session ids are colon-bearing): <session>::<message>::block@sha256:<hex>; structured form stored wherever durable. Resolver returns TYPED states, never guesses: ok | drifted_position | drifted_message | relocated_lineage (search lineage neighborhood, prefix-sharing preferred over spawned-fresh) | ambiguous (multiple hash hits => candidates listed, NOT a pick) | missing | quarantined | hash_mismatch (hard fail, never rewrite).

## Existing design note

Derived index-tier change: canonical DDL edit + version bump — batch with the next index window (gjg.1, ma2, 4ts.5, wohv all queue for the same bump). Writer computes at block write (BOTH storage twins). Boilerplate-duplicate ambiguity is expected (same prompt text N times) — the ambiguous state is the honest answer; position_hint + message hint disambiguate the common case. Empirical dup-rate check on the live archive is part of this bead (policy depends on it).

## Acceptance criteria

Anchor created pre-re-ingest resolves post-re-ingest as ok or drifted_position (verified content); a fork replay resolves relocated_lineage with the inheritance edge cited; ambiguous returns candidates; hash_mismatch never auto-rewrites. Verify: re-ingest fixture round-trip tests.

## Static mechanism / likely defect

Issue description localizes the mechanism: THE anchor atom multiple programs stand on (webui cockpit citations, finding evidence refs rxdo.4, drift detection 37t.14, compaction loss anchors gjg.3, export citations). Today block identity is message_id:position — position shifts on re-ingest and fork replay, so any stored block citation can silently point at different content. Verified live: blocks table has NO content_hash (sessions and messages do). Add blocks.content_hash (32B, over canonical block EVIDENCE: type, text, tool_name, canonical tool_input, se… Design direction: Derived index-tier change: canonical DDL edit + version bump — batch with the next index window (gjg.1, ma2, 4ts.5, wohv all queue for the same bump). Writer computes at block write (BOTH storage twins). Boilerplate-duplicate ambiguity is expected (same prompt text N times) — the ambiguous state is the honest answer; position_hint + message hint disambiguate the common case. Empirical dup-rate check on the live arch…

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

1. Derived index-tier change: canonical DDL edit + version bump — batch with the next index window (gjg.1, ma2, 4ts.5, wohv all queue for the same bump).
2. Writer computes at block write (BOTH storage twins).
3. Boilerplate-duplicate ambiguity is expected (same prompt text N times) — the ambiguous state is the honest answer
4. position_hint + message hint disambiguate the common case.
5. Empirical dup-rate check on the live archive is part of this bead (policy depends on it).

## Tests to add

- Acceptance proof: Anchor created pre-re-ingest resolves post-re-ingest as ok or drifted_position (verified content)
- Acceptance proof: a fork replay resolves relocated_lineage with the inheritance edge cited
- Acceptance proof: ambiguous returns candidates
- Acceptance proof: hash_mismatch never auto-rewrites.
- Acceptance proof: Verify: re-ingest fixture round-trip tests.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
