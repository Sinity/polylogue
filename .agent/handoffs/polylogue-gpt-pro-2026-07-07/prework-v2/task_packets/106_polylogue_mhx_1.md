# 106. polylogue-mhx.1 — Provider abstraction: one OpenAI-compatible embedding client, local and cloud, model registry in meta

Priority/type/status: **P2 / feature / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Voyage is hardcoded (VOYAGE_API_URL/DEFAULT_MODEL/DEFAULT_DIMENSION constants). Generality is one abstraction away: an OpenAI-compatible /v1/embeddings client pointed at a configurable base_url covers OpenAI, Voyage (has an OpenAI-compatible surface), and every local server (Ollama, llama.cpp, Infinity, LM Studio) — and the operator already runs a LiteLLM gateway at 127.0.0.1:4000 that bridges all of them, so local models need zero new protocol code.

## Existing design note

(1) Config: [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute + batch size; keep the native Voyage path as one provider preset, default unchanged. (2) Identity: message_embeddings_meta already records model — extend recorded identity to (base_url_host, model, dimension, model_revision?) so a mixed-provenance table is detectable and refused at query time (vectors from different models must never RRF together silently). (3) Dimension handling: vec0 tables are fixed-dim — dimension change = embeddings tier reset + re-embed with `ops embed preflight` cost estimate shown BEFORE any spend; wire the reset into ops reset --embeddings if not present. (4) Cost model: per-model $/1M-token entries resolve from the vendored LiteLLM catalog (single pricing source doctrine) with local models priced $0 but time-estimated (tokens/s probe) so preflight stays honest for the free lane. (5) Acceptance = 37t.5's demo: qwen3-class embedding model through the gateway, full backfill on the seeded corpus, --semantic queries return sane neighbors — that bead's scope collapses into verifying this abstraction end-to-end locally. VERIFY current sqlite-vec + Voyage API surfaces before freezing config names.

## Acceptance criteria

1. A single OpenAI-compatible /v1/embeddings client replaces the hardcoded Voyage path (sqlite_vec_support.py VOYAGE_API_URL / DEFAULT_MODEL / DEFAULT_DIMENSION constants), driven by [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute / batch_size; the native Voyage preset stays the unchanged default. 2. message_embeddings_meta records (base_url_host, model, dimension) and query time REFUSES to RRF vectors of differing model identity rather than mixing silently (test-asserted). 3. A dimension change triggers an embeddings-tier reset via `ops reset --embeddings` with an `ops embed preflight` cost estimate shown BEFORE any spend; local models price $0 with a tokens/s time estimate from the vendored LiteLLM catalog (single-pricing-source doctrine). 4. Acceptance demo (folds in 37t.5): a qwen3-class embedding model through the LiteLLM gateway (127.0.0.1:4000) backfills the seeded corpus and `polylogue find --semantic <q>` returns sane neighbors at $0. Verify: seeded corpus embedded via the gateway; `--semantic` returns relevant results; a mixed-model query raises/refuses; config names verified against the current sqlite-vec + Voyage surfaces before freezing.

## Static mechanism / likely defect

Issue description localizes the mechanism: Voyage is hardcoded (VOYAGE_API_URL/DEFAULT_MODEL/DEFAULT_DIMENSION constants). Generality is one abstraction away: an OpenAI-compatible /v1/embeddings client pointed at a configurable base_url covers OpenAI, Voyage (has an OpenAI-compatible surface), and every local server (Ollama, llama.cpp, Infinity, LM Studio) — and the operator already runs a LiteLLM gateway at 127.0.0.1:4000 that bridges all of them, so local models need zero new protocol code. Design direction: (1) Config: [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute + batch size; keep the native Voyage path as one provider preset, default unchanged. (2) Identity: message_embeddings_meta already records model — extend recorded identity to (base_url_host, model, dimension, model_revision?) so a mixed-provenance table is detectable and refused at query time (vectors from different m…

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

1. (1) Config: [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute + batch size
2. keep the native Voyage path as one provider preset, default unchanged.
3. (2) Identity: message_embeddings_meta already records model — extend recorded identity to (base_url_host, model, dimension, model_revision?) so a mixed-provenance table is detectable and refused at query time (vectors from different models must never RRF together silently).
4. (3) Dimension handling: vec0 tables are fixed-dim — dimension change = embeddings tier reset + re-embed with `ops embed preflight` cost estimate shown BEFORE any spend
5. wire the reset into ops reset --embeddings if not present.
6. (4) Cost model: per-model $/1M-token entries resolve from the vendored LiteLLM catalog (single pricing source doctrine) with local models priced $0 but time-estimated (tokens/s probe) so preflight stays honest for the free lane.
7. (5) Acceptance = 37t.5's demo: qwen3-class embedding model through the gateway, full backfill on the seeded corpus, --semantic queries return sane neighbors — that bead's scope collapses into verifying this abstraction end-to-end locally.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: A single OpenAI-compatible /v1/embeddings client replaces the hardcoded Voyage path (sqlite_vec_support.py VOYAGE_API_URL / DEFAULT_MODEL / DEFAULT_DIMENSION constants), driven by [embedding] provider_base_url / model / dimension / api_key(_env) / requests_per_minute / batch_size
- Acceptance proof: the native Voyage preset stays the unchanged default.
- Acceptance proof: 2.
- Acceptance proof: message_embeddings_meta records (base_url_host, model, dimension) and query time REFUSES to RRF vectors of differing model identity rather than mixing silently (test-asserted).
- Acceptance proof: 3.
- Acceptance proof: A dimension change triggers an embeddings-tier reset via `ops reset --embeddings` with an `ops embed preflight` cost estimate shown BEFORE any spend
- Acceptance proof: local models price $0 with a tokens/s time estimate from the vendored LiteLLM catalog (single-pricing-source doctrine).

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
