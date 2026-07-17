# 109. polylogue-mhx.4 — Semantic recall leg in context compilation: the memory actually retrieves

Priority/type/status: **P2 / feature / open**. Lane: **09-embeddings-retrieval**. Release: **J-embeddings-retrieval**. Readiness: **blocked-hard**.

Hard blockers: polylogue-mhx.2

## What the bead says

compose_context_preamble and compile_context currently select by explicit refs, recency, and policy — there is no semantic leg, so a judged lesson about 'SQLite WAL contention' never surfaces when the new session starts debugging a WAL issue unless someone remembers it exists. This is the retrieval moment the entire memory thesis needs: relevant judged assertions + similar prior sessions, recalled by meaning, within budget, with refs.

## Existing design note

(1) Query formation: at SessionStart the recall query is cheap context — repo, recent commit subjects, the resumed session's summary (on resume); mid-session (agent-invoked via MCP) it is the agent's stated intent. (2) Retrieval: assertion vectors + session vectors (emb-targets) under a similarity floor + top-K cap; judged/active assertions rank above candidates; recency and repo-match as tiebreakers, all weights visible in the payload (no opaque scores — every recalled item carries why: similarity, kind, judgment state, refs). (3) Budget: recall competes inside the existing segment budget of the preamble (37t.4's ~600-token cap) — indices/refs over bodies per jgp; expandable via resolve_ref. (4) Fallback honesty: when embeddings are disabled/absent, the leg degrades to FTS-over-assertions and SAYS so in the payload (retrieval_lane field), never silently changing semantics. (5) Eval tie-in: add recall-scenario rows to the emb-eval labeled set (lesson X should surface for session-start context Y) so the leg's value is measured, feeding the uplift re-run instrumentation.

## Acceptance criteria

SessionStart recall proposes items through the ContextSource protocol (37t.11) with visible why-fields (similarity, kind, judgment state, refs) — no opaque scores; judged assertions outrank candidates at equal similarity; recall stays within the preamble segment budget with refs-over-bodies; a seeded lesson about a distinctive topic surfaces when a session starts on that topic and does NOT surface on an unrelated repo (both directions tested); degrades to silent no-op when embeddings are absent/stale. Verify: devtools test -k 'recall or context' + one live SessionStart observation with the ledger row showing the allocation.

## Static mechanism / likely defect

Issue description localizes the mechanism: compose_context_preamble and compile_context currently select by explicit refs, recency, and policy — there is no semantic leg, so a judged lesson about 'SQLite WAL contention' never surfaces when the new session starts debugging a WAL issue unless someone remembers it exists. This is the retrieval moment the entire memory thesis needs: relevant judged assertions + similar prior sessions, recalled by meaning, within budget, with refs. Design direction: (1) Query formation: at SessionStart the recall query is cheap context — repo, recent commit subjects, the resumed session's summary (on resume); mid-session (agent-invoked via MCP) it is the agent's stated intent. (2) Retrieval: assertion vectors + session vectors (emb-targets) under a similarity floor + top-K cap; judged/active assertions rank above candidates; recency and repo-match as tiebreakers, all weights vi…

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

1. (1) Query formation: at SessionStart the recall query is cheap context — repo, recent commit subjects, the resumed session's summary (on resume)
2. mid-session (agent-invoked via MCP) it is the agent's stated intent.
3. (2) Retrieval: assertion vectors + session vectors (emb-targets) under a similarity floor + top-K cap
4. judged/active assertions rank above candidates
5. recency and repo-match as tiebreakers, all weights visible in the payload (no opaque scores — every recalled item carries why: similarity, kind, judgment state, refs).
6. (3) Budget: recall competes inside the existing segment budget of the preamble (37t.4's ~600-token cap) — indices/refs over bodies per jgp
7. expandable via resolve_ref.

## Tests to add

- Acceptance proof: SessionStart recall proposes items through the ContextSource protocol (37t.11) with visible why-fields (similarity, kind, judgment state, refs) — no opaque scores
- Acceptance proof: judged assertions outrank candidates at equal similarity
- Acceptance proof: recall stays within the preamble segment budget with refs-over-bodies
- Acceptance proof: a seeded lesson about a distinctive topic surfaces when a session starts on that topic and does NOT surface on an unrelated repo (both directions tested)
- Acceptance proof: degrades to silent no-op when embeddings are absent/stale.
- Acceptance proof: Verify: devtools test -k 'recall or context' + one live SessionStart observation with the ledger row showing the allocation.

## Verification commands

- `devtools test <focused tests added for this bead>`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
