# 110. polylogue-0k6 — Embedding changed-text full-replace regression vs split embeddings.db metadata

Priority/type/status: **P2 / task / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Changed-text reindexing for the same message_id needs an explicit full-replace regression against split embeddings.db metadata (index-tier rows cleared, embeddings tier not).

## Existing design note

Step 1 — QUANTIFY on the live archive (fables analysis 9): count sessions whose updated_at_ms postdates embedding_status.last_embedded_at_ms with unchanged message counts — the concrete stale-vector population the original bug produced; record the number in the bead on completion (it doubles as the fix's impact statement). Step 2 — regression: ingest fixture; re-ingest FULL-REPLACE variant with one message body changed at same position/count; assert (a) session selected by select_pending_archive_session_window, (b) after re-embed, message_embeddings_meta.content_hash matches the new hash and the old vector row is replaced not duplicated — the split-tier trap is index-tier rows cleared by full replace while embeddings.db metadata persists. If (b) fails, fix embedding_write.py to upsert by (session_id, position).

## Acceptance criteria

1. QUANTIFY step recorded: the count of sessions whose index-tier updated_at_ms postdates embedding_status.last_embedded_at_ms at unchanged message count is measured on the live archive and written into the bead as the fix's impact number. 2. Regression test: ingest a fixture, re-ingest a FULL-REPLACE variant with one message body changed at the same position/count, and assert (a) the session is re-selected by select_pending_archive_session_window and (b) after re-embed, message_embeddings_meta.content_hash matches the new hash with the old vector row REPLACED, not duplicated (the split-tier trap: index-tier rows cleared by full replace while embeddings.db metadata persists). 3. If (b) fails pre-fix, embedding_write.py upserts by (session_id, position). Verify: the new regression test fails on current main if the split-tier bug is live and passes after the fix (`devtools test` selection on the embeddings write path).

## Static mechanism / likely defect

Issue description localizes the mechanism: Changed-text reindexing for the same message_id needs an explicit full-replace regression against split embeddings.db metadata (index-tier rows cleared, embeddings tier not). Design direction: Step 1 — QUANTIFY on the live archive (fables analysis 9): count sessions whose updated_at_ms postdates embedding_status.last_embedded_at_ms with unchanged message counts — the concrete stale-vector population the original bug produced; record the number in the bead on completion (it doubles as the fix's impact statement). Step 2 — regression: ingest fixture; re-ingest FULL-REPLACE variant with one message body chan…

## Source anchors to inspect first

- `polylogue/api/embeddings.py` — Public embedding API seam.
- `polylogue/cli/commands/embed.py` — Operator embedding command and dry-run planner hooks.
- `polylogue/archive/query/retrieval.py` — Retrieval composition seam for FTS/vector/hybrid.
- `polylogue/archive/query/retrieval_search.py` — Search/retrieval runtime implementation.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Step 1 — QUANTIFY on the live archive (fables analysis 9): count sessions whose updated_at_ms postdates embedding_status.last_embedded_at_ms with unchanged message counts — the concrete stale-vector population the original bug produced
2. record the number in the bead on completion (it doubles as the fix's impact statement).
3. Step 2 — regression: ingest fixture
4. re-ingest FULL-REPLACE variant with one message body changed at same position/count
5. assert (a) session selected by select_pending_archive_session_window, (b) after re-embed, message_embeddings_meta.content_hash matches the new hash and the old vector row is replaced not duplicated — the split-tier trap is index-tier rows cleared by full replace while embeddings.db metadata persists.
6. If (b) fails, fix embedding_write.py to upsert by (session_id, position).

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: QUANTIFY step recorded: the count of sessions whose index-tier updated_at_ms postdates embedding_status.last_embedded_at_ms at unchanged message count is measured on the live archive and written into the bead as the fix's impact number.
- Acceptance proof: 2.
- Acceptance proof: Regression test: ingest a fixture, re-ingest a FULL-REPLACE variant with one message body changed at the same position/count, and assert (a) the session is re-selected by select_pending_archive_session_window and (b) after re-embed, message_embeddings_meta.content_hash matches the new hash with the old vector row REPLACED, not duplicated (the split-tier trap: index-tier rows cleared by full replace while embedding…
- Acceptance proof: 3.
- Acceptance proof: If (b) fails pre-fix, embedding_write.py upserts by (session_id, position).
- Acceptance proof: Verify: the new regression test fails on current main if the split-tier bug is live and passes after the fix (`devtools test` selection on the embeddings write path).

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
