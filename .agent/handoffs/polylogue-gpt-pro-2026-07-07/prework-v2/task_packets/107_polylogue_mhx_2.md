# 107. polylogue-mhx.2 — Embedding target policy: what gets a vector, at what granularity, at what cost

Priority/type/status: **P2 / feature / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Today exactly one class is embedded: authored prose messages (user/assistant, human/assistant-authored material origin, positive word count — the v21 partial index). That is the right floor but the wrong ceiling: session-level retrieval runs on message vectors (expensive, noisy), and assertions/memory — the content whose retrieval matters most for the context loop — have no vectors at all. Nobody has written down what SHOULD be embedded and why; this bead is that decision plus its implementation.

## Existing design note

Target classes, each with an explicit purpose, source text, and marginal cost: (1) authored prose messages [exists] — purpose: fine-grained hybrid search. (2) SESSION vectors [new]: one vector per session from derived text that already exists (title + profile summary + top-K salient authored lines); purpose: find_similar_sessions and neighbor candidates at 16k-session scale without scanning message vectors; near-zero marginal token cost because the text is already materialized in profiles. Storage: session_embeddings vec0 table in embeddings.db keyed by session_id + model identity. (3) ASSERTION vectors [new]: judged/candidate assertion bodies; purpose: semantic recall in context compilation (the emb-recall bead consumes this); tiny corpus, negligible cost, embed-on-write. (4) Explicit NON-targets, documented: tool payloads, generated context packs, protocol rows, reasoning dumps — material_origin filtering already excludes them; state it as policy so nobody 'completes' coverage by embedding noise. Config: per-class enable flags under [embedding.targets]; preflight and status report per-class counts/coverage separately. Rebuild: all classes rebuildable from index.db/user.db — tier-reset doctrine unchanged.

## Acceptance criteria

1. SESSION vectors implemented: a `session_embeddings` vec0 table in embeddings.db keyed by (session_id, model identity), populated from already-materialized derived text (title + profile summary + top-K salient authored lines) at near-zero new token spend; find_similar_sessions / neighbor candidates can run off it. 2. ASSERTION vectors implemented: judged/candidate assertion bodies embedded on-write into their own vec0 table. 3. Per-class enable flags exist under [embedding.targets]; `ops embed preflight` and `ops embed status --detail` report per-class pending/coverage counts SEPARATELY. 4. Non-targets (tool payloads, generated context packs, protocol rows, reasoning dumps) are documented as policy and a test asserts a tool_use block is never embedded. 5. All classes rebuild from index.db/user.db via tier reset (`ops reset --embeddings`). Verify: seed corpus, run backfill, `ops embed status --detail` shows three distinct class coverages; `devtools test` selection asserts the tool_use-never-embedded exclusion.

## Static mechanism / likely defect

Issue description localizes the mechanism: Today exactly one class is embedded: authored prose messages (user/assistant, human/assistant-authored material origin, positive word count — the v21 partial index). That is the right floor but the wrong ceiling: session-level retrieval runs on message vectors (expensive, noisy), and assertions/memory — the content whose retrieval matters most for the context loop — have no vectors at all. Nobody has written down what SHOULD be embedded and why; this bead is that decision plus its implementation. Design direction: Target classes, each with an explicit purpose, source text, and marginal cost: (1) authored prose messages [exists] — purpose: fine-grained hybrid search. (2) SESSION vectors [new]: one vector per session from derived text that already exists (title + profile summary + top-K salient authored lines); purpose: find_similar_sessions and neighbor candidates at 16k-session scale without scanning message vectors; near-zer…

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

1. Target classes, each with an explicit purpose, source text, and marginal cost: (1) authored prose messages [exists] — purpose: fine-grained hybrid search.
2. (2) SESSION vectors [new]: one vector per session from derived text that already exists (title + profile summary + top-K salient authored lines)
3. purpose: find_similar_sessions and neighbor candidates at 16k-session scale without scanning message vectors
4. near-zero marginal token cost because the text is already materialized in profiles.
5. Storage: session_embeddings vec0 table in embeddings.db keyed by session_id + model identity.
6. (3) ASSERTION vectors [new]: judged/candidate assertion bodies
7. purpose: semantic recall in context compilation (the emb-recall bead consumes this)

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: SESSION vectors implemented: a `session_embeddings` vec0 table in embeddings.db keyed by (session_id, model identity), populated from already-materialized derived text (title + profile summary + top-K salient authored lines) at near-zero new token spend
- Acceptance proof: find_similar_sessions / neighbor candidates can run off it.
- Acceptance proof: 2.
- Acceptance proof: ASSERTION vectors implemented: judged/candidate assertion bodies embedded on-write into their own vec0 table.
- Acceptance proof: 3.
- Acceptance proof: Per-class enable flags exist under [embedding.targets]
- Acceptance proof: `ops embed preflight` and `ops embed status --detail` report per-class pending/coverage counts SEPARATELY.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
