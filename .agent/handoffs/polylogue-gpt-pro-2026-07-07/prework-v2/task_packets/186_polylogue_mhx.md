# 186. polylogue-mhx — Embedding substrate: provider-general, honest lifecycle, retrieval that earns its cost

Priority/type/status: **P2 / epic / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **epic-needs-child-closure**.

## What the bead says

Current state: one hardcoded cloud provider (Voyage voyage-4, 1024-dim, constants in sqlite_vec_support.py), vec0 fixed-dimension tables, embedding targets limited to authored prose messages (v21 partial index), opt-in daemon catch-up with cost caps, hybrid RRF + --semantic/--similar surfaces, ops embed onboarding group. Gaps this program owns: provider/model generality (local AND cloud through one abstraction), an explicit answer to WHAT gets embedded and why, retrieval quality measured instead of assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the whole lane — semantic recall in context compilation, clustering/topics, novelty detection. Doctrine anchor: embeddings.db is a rebuildable tier — model/dimension switches are tier resets with cost preflight, never in-place migrations (fresh-first). Existing beads folded in by dependency: 37t.5 (local lane) is the acceptance demo for provider generality; 0k6 (changed-text staleness) is lifecycle honesty.

## Existing design note

Epic scope: provider/model generality (one OpenAI-compatible embedding client covering local and cloud), an explicit embedding-target policy (what gets a vector and why), retrieval quality measured rather than assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the lane (semantic recall in context compilation, clustering/topics, novelty detection). Doctrine anchor: embeddings.db is a rebuildable tier, so model/dimension switches are tier resets with cost preflight, never in-place migrations. Delivered through child beads: mhx.1 (provider abstraction), mhx.2 (target policy), folded-in 37t.5 (local-lane acceptance demo), 0k6 (changed-text staleness), mhx.5 (semantic layer), 0ns (bounded per-session work).

## Acceptance criteria

1. All child beads (mhx.*, folded-in 37t.5, 0k6, 0ns) are closed (`bd show polylogue-mhx --json` shows no open children). 2. Provider generality demonstrated end-to-end: a local (qwen3-class) embedding model through the LiteLLM gateway (127.0.0.1:4000) backfills the seeded corpus and `polylogue find --semantic <q>` returns sane neighbors at $0 (mhx.1 acceptance). 3. Embedding-target classes (message, session, assertion) each report separate coverage via `polylogue ops embed status --detail`; documented non-targets (tool payloads, context packs, protocol rows, reasoning dumps) are test-asserted excluded (mhx.2). 4. Lifecycle honesty: a model/dimension switch triggers an embeddings-tier reset (`ops reset --embeddings`) with an `ops embed preflight` cost estimate shown before any spend; the changed-text staleness regression (0k6) passes; mixed-model vectors are refused rather than silently RRF'd. Verify: `bd show polylogue-mhx --json` children closed; the mhx.1/mhx.2 acceptance demos run on the seeded corpus.

## Static mechanism / likely defect

Issue description localizes the mechanism: Current state: one hardcoded cloud provider (Voyage voyage-4, 1024-dim, constants in sqlite_vec_support.py), vec0 fixed-dimension tables, embedding targets limited to authored prose messages (v21 partial index), opt-in daemon catch-up with cost caps, hybrid RRF + --semantic/--similar surfaces, ops embed onboarding group. Gaps this program owns: provider/model generality (local AND cloud through one abstraction), an explicit answer to WHAT gets embedded and why, retrieval quality measured instead of assumed, lifecy… Design direction: Epic scope: provider/model generality (one OpenAI-compatible embedding client covering local and cloud), an explicit embedding-target policy (what gets a vector and why), retrieval quality measured rather than assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the lane (semantic recall in context compilation, clustering/topics, novelty detection). Doctrine anchor: embed…

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

1. Epic scope: provider/model generality (one OpenAI-compatible embedding client covering local and cloud), an explicit embedding-target policy (what gets a vector and why), retrieval quality measured rather than assumed, lifecycle honesty (staleness, model switches, spend), and the advanced uses that justify the lane (semantic recall in context compilation, clustering/topics, novelty detection).
2. Doctrine anchor: embeddings.db is a rebuildable tier, so model/dimension switches are tier resets with cost preflight, never in-place migrations.
3. Delivered through child beads: mhx.1 (provider abstraction), mhx.2 (target policy), folded-in 37t.5 (local-lane acceptance demo), 0k6 (changed-text staleness), mhx.5 (semantic layer), 0ns (bounded per-session work).

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: All child beads (mhx.*, folded-in 37t.5, 0k6, 0ns) are closed (`bd show polylogue-mhx --json` shows no open children).
- Acceptance proof: 2.
- Acceptance proof: Provider generality demonstrated end-to-end: a local (qwen3-class) embedding model through the LiteLLM gateway (127.0.0.1:4000) backfills the seeded corpus and `polylogue find --semantic <q>` returns sane neighbors at $0 (mhx.1 acceptance).
- Acceptance proof: 3.
- Acceptance proof: Embedding-target classes (message, session, assertion) each report separate coverage via `polylogue ops embed status --detail`
- Acceptance proof: documented non-targets (tool payloads, context packs, protocol rows, reasoning dumps) are test-asserted excluded (mhx.2).
- Acceptance proof: 4.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
