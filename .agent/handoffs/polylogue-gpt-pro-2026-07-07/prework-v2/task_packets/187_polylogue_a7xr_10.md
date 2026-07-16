# 187. polylogue-a7xr.10 — Kill-or-adopt the search-provider lane: production bypasses the abstraction it should use

Priority/type/status: **P2 / chore / open**. Lane: **11-interoperability-origin**. Release: **K-interop-origin**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

VERIFIED 2026-07-06: FTS5Provider/HybridSearchProvider/factories have zero production call sites — only their own tests import them. Production FTS is inline SQL (archive_tiers/archive.py:4545/4661/7668) and --retrieval-lane hybrid re-implements fusion inline at cli/archive_query.py:830-852. OPERATOR REFRAME (2026-07-06): non-use may indict the SURFACES, not the abstraction — a CLI module implementing retrieval semantics inline violates the substrate-owns-meaning rule, and mhx.3's four-lane bake-off (FTS / dense / hybrid / hybrid+rerank over identical chunks) is precisely the consumer a swappable retrieval-lane interface serves. So this is a KILL-OR-ADOPT decision, not a deletion: (ADOPT) redesign the lane interface FROM the live inline implementations (the inline SQL is the battle-tested semantics; the dead classes are unproven — adoption means moving proven SQL behind the interface, not resurrecting unproven classes as-is), route CLI/daemon/MCP retrieval through it, and mhx.3 gets its lanes for free; (KILL) delete the classes and accept inline retrieval per-surface, with mhx.3 building its own harness-local lanes. Decide WITH mhx.3 — whoever executes first owns the decision.

## Existing design note

If ADOPT: define RetrievalLane protocol from what production actually needs (query, candidate set, scores, lane metadata for the eval payload); implementations wrap the existing inline SQL (fts lane), SqliteVecProvider (dense lane), reciprocal_rank_fusion (hybrid), reranker (mhx.1's client); cli/archive_query.py:830-852 becomes lane dispatch; the current FTS5Provider/HybridSearchProvider bodies are salvage-or-delete per method (most likely delete — their tests test invented semantics, keep test_hybrid_laws property shapes if the fusion laws transfer). If KILL: delete classes+factories+SearchProvider protocol+dead tests; keep reciprocal_rank_fusion + hybrid_sessions helpers + the whole SqliteVecProvider vector half (live, 8 call sites) either way.

## Acceptance criteria

A decision recorded WITH mhx.3 (adopt or kill, one paragraph of why); if adopt: all production retrieval flows through the lane interface, inline fusion in archive_query.py gone, mhx.3 bake-off consumes the lanes, goldens unchanged; if kill: zero references remain, mhx.3 notes it owns lane construction. Either way devtools verify green.

## Static mechanism / likely defect

Issue description localizes the mechanism: VERIFIED 2026-07-06: FTS5Provider/HybridSearchProvider/factories have zero production call sites — only their own tests import them. Production FTS is inline SQL (archive_tiers/archive.py:4545/4661/7668) and --retrieval-lane hybrid re-implements fusion inline at cli/archive_query.py:830-852. OPERATOR REFRAME (2026-07-06): non-use may indict the SURFACES, not the abstraction — a CLI module implementing retrieval semantics inline violates the substrate-owns-meaning rule, and mhx.3's four-lane bake-off (FTS / dense /… Design direction: If ADOPT: define RetrievalLane protocol from what production actually needs (query, candidate set, scores, lane metadata for the eval payload); implementations wrap the existing inline SQL (fts lane), SqliteVecProvider (dense lane), reciprocal_rank_fusion (hybrid), reranker (mhx.1's client); cli/archive_query.py:830-852 becomes lane dispatch; the current FTS5Provider/HybridSearchProvider bodies are salvage-or-delete…

## Source anchors to inspect first

- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. If ADOPT: define RetrievalLane protocol from what production actually needs (query, candidate set, scores, lane metadata for the eval payload)
2. implementations wrap the existing inline SQL (fts lane), SqliteVecProvider (dense lane), reciprocal_rank_fusion (hybrid), reranker (mhx.1's client)
3. cli/archive_query.py:830-852 becomes lane dispatch
4. the current FTS5Provider/HybridSearchProvider bodies are salvage-or-delete per method (most likely delete — their tests test invented semantics, keep test_hybrid_laws property shapes if the fusion laws transfer).
5. If KILL: delete classes+factories+SearchProvider protocol+dead tests
6. keep reciprocal_rank_fusion + hybrid_sessions helpers + the whole SqliteVecProvider vector half (live, 8 call sites) either way.

## Tests to add

- Acceptance proof: A decision recorded WITH mhx.3 (adopt or kill, one paragraph of why)
- Acceptance proof: if adopt: all production retrieval flows through the lane interface, inline fusion in archive_query.py gone, mhx.3 bake-off consumes the lanes, goldens unchanged
- Acceptance proof: if kill: zero references remain, mhx.3 notes it owns lane construction.
- Acceptance proof: Either way devtools verify green.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
