# 112. polylogue-37t.5 — Local embedding lane via OpenAI-compatible provider (LiteLLM gateway)

Priority/type/status: **P2 / feature / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **blocked-hard**.

Hard blockers: polylogue-mhx.1

## What the bead says

Voyage is the only embedding provider; a local lane makes semantic search $0 and the whole loop air-gapped — and pairs with the Hermes bridge program for a fully local, zero-cloud stack.

## Existing design note

Seam: VectorProvider protocol + Voyage constants in sqlite_vec_support.py. Config: [embedding] provider='openai-compatible', base_url, model, dimension in polylogue.toml; implement the OpenAI /v1/embeddings client shape once (LiteLLM gateway 127.0.0.1:4000 bridges to Ollama). Dimension: vec0 table is fixed float[1024] and EMBEDDING_DIMENSION asserted in meta CHECK — dimension becomes a tier-init parameter in embeddings.db meta; changing model/dimension => ops reset --embeddings + backfill (tier is designed expensive-rebuild; NO in-place migration); bump EMBEDDINGS_SCHEMA_VERSION. Cost preflight must branch on provider ($0 local), not hardcode Voyage constants. Eval before switching default: embed demo corpus + 200 live prose messages with both models; --similar top-10 overlap + a 20-query hand-relevance check.

## Acceptance criteria gap

This active bead lacks acceptance criteria in the export. Add checkable acceptance criteria before coding unless this packet explicitly supplies a temporary gate.

## Static mechanism / likely defect

Issue description localizes the mechanism: Voyage is the only embedding provider; a local lane makes semantic search $0 and the whole loop air-gapped — and pairs with the Hermes bridge program for a fully local, zero-cloud stack. Design direction: Seam: VectorProvider protocol + Voyage constants in sqlite_vec_support.py. Config: [embedding] provider='openai-compatible', base_url, model, dimension in polylogue.toml; implement the OpenAI /v1/embeddings client shape once (LiteLLM gateway 127.0.0.1:4000 bridges to Ollama). Dimension: vec0 table is fixed float[1024] and EMBEDDING_DIMENSION asserted in meta CHECK — dimension becomes a tier-init parameter in embeddi…

## Source anchors to inspect first

- `polylogue/coordination/envelope.py` — Coordination envelope model exists; harden it as the shared payload.
- `polylogue/coordination/payloads.py` — Coordination payload types should stay small and evidence-ref oriented.
- `polylogue/coordination/rendering.py` — Rendered advisories should be scheduler-mediated, not chat spam.
- `tests/unit/coordination/test_envelope.py` — Existing envelope tests are the starting verification lane.
- `polylogue/mcp/server_prompts.py:219` — MCP prompt registration exists and can surface cookbook/roles.
- `polylogue/cli/commands/agents.py` — CLI agent commands are the operator-facing entry point.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:31` — ASSERTION_DEFAULT_STATUS is ACTIVE, so missing status currently means trusted active.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641` — upsert_blackboard_note passes author_kind and no explicit status into upsert_assertion.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — upsert_assertion is the single write chokepoint to patch.
- `polylogue/api/contracts/assertions.py` — Check public assertion request/response contract after changing default status behavior.

## Implementation plan

1. Seam: VectorProvider protocol + Voyage constants in sqlite_vec_support.py.
2. Config: [embedding] provider='openai-compatible', base_url, model, dimension in polylogue.toml
3. implement the OpenAI /v1/embeddings client shape once (LiteLLM gateway 127.0.0.1:4000 bridges to Ollama).
4. Dimension: vec0 table is fixed float[1024] and EMBEDDING_DIMENSION asserted in meta CHECK — dimension becomes a tier-init parameter in embeddings.db meta
5. changing model/dimension => ops reset --embeddings + backfill (tier is designed expensive-rebuild
6. NO in-place migration)
7. bump EMBEDDINGS_SCHEMA_VERSION.

## Tests to add

- No-provider mode test.
- Changed-text invalidation test.
- Budget dry-run and large-session bound test.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
