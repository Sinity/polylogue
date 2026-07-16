# 042. polylogue-t46.4 — Delegate daemon session-similarity KNN to SqliteVecProvider.query_by_session

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

daemon/similarity.py re-implements session-seeded vector ranking (raw MATCH/k SQL over message_embeddings, per-session best-distance aggregation, matched-message count, L2->cosine in _l2_to_cosine_similarity:217) that storage/search_providers/sqlite_vec_queries.py:143 SqliteVecProvider.query_by_session already does -- the substrate file even comments that the daemon's _PER_MESSAGE_K mirrors it. Fix: build_similar_payload (http.py:3158) should call the facade/vec-provider session-similarity method and only project the payload; delete the daemon KNN/aggregation/L2->cosine copy. If the daemon needs a per-session rollup the provider does not expose, add it to the provider (substrate), not the surface.

## Acceptance criteria

daemon _knn_for_embedding/_aggregate_hits/_l2_to_cosine_similarity are deleted; /api/similar ranking equals SqliteVecProvider.query_by_session ordering for a seed session (parity test); the sqlite_vec_queries comment about mirroring _PER_MESSAGE_K is removed because there is no longer a mirror; devtools verify green.

## Static mechanism / likely defect

Design direction: daemon/similarity.py re-implements session-seeded vector ranking (raw MATCH/k SQL over message_embeddings, per-session best-distance aggregation, matched-message count, L2->cosine in _l2_to_cosine_similarity:217) that storage/search_providers/sqlite_vec_queries.py:143 SqliteVecProvider.query_by_session already does -- the substrate file even comments that the daemon's _PER_MESSAGE_K mirrors it. Fix: build_similar_pa…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. daemon/similarity.py re-implements session-seeded vector ranking (raw MATCH/k SQL over message_embeddings, per-session best-distance aggregation, matched-message count, L2->cosine in _l2_to_cosine_similarity:217) that storage/search_providers/sqlite_vec_queries.py:143 SqliteVecProvider.query_by_session already does -- the substrate file even comments that the daemon's _PER_MESSAGE_K mirrors it.
2. Fix: build_similar_payload (http.py:3158) should call the facade/vec-provider session-similarity method and only project the payload
3. delete the daemon KNN/aggregation/L2->cosine copy.
4. If the daemon needs a per-session rollup the provider does not expose, add it to the provider (substrate), not the surface.

## Tests to add

- Acceptance proof: daemon _knn_for_embedding/_aggregate_hits/_l2_to_cosine_similarity are deleted
- Acceptance proof: /api/similar ranking equals SqliteVecProvider.query_by_session ordering for a seed session (parity test)
- Acceptance proof: the sqlite_vec_queries comment about mirroring _PER_MESSAGE_K is removed because there is no longer a mirror
- Acceptance proof: devtools verify green.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
