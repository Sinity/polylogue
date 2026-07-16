# 098. polylogue-20d.10 — Runtime post-filter efficiency: memoize semantic facts; lower matchers onto the actions view

Priority/type/status: **P2 / task / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

matches_action_sequence, matches_referenced_path, and category matching each call _actions_for(session) -> build_session_semantic_facts (runtime_matching.py:20-25) — full semantic-fact construction over a hydrated session, no memoization across the three matchers, applied as list-comprehension post-filter (runtime_filters.py:188-189). A broad query with SEQ or referenced_path hydrates every SQL-surviving candidate and builds facts up to 3x.

## Existing design note

Minimal fix: memoize facts per session within a filter pass (functools cache keyed per pass, or attach _semantic_facts to the Session object). Real fix: all three matchers' predicates (action category, affected path, sequence) are answerable from actions-view rows — fetch once per candidate set with a single WHERE session_id IN (...) query, group in Python, drop hydration entirely for candidates failing cheap predicates. The keystone columns (v16) and idx_blocks_type_tool (v20) exist for exactly this shape. Also push cheap structured clauses into SQL before hydration. SEQ span capture (DSL bead) builds on the same relation — coordinate.

## Acceptance criteria

1. Minimal fix: semantic facts are memoized per session within a single filter pass (no more than one build_session_semantic_facts per session per pass), eliminating the up-to-3x construction across matches_action_sequence / matches_referenced_path / category matching (runtime_matching.py, runtime_filters.py). 2. Real fix: the three matchers' predicates (action category, affected path, sequence) are answered from actions-view rows fetched once per candidate set with a single `WHERE session_id IN (...)` query, grouped in Python; candidates failing cheap predicates are dropped before hydration, and cheap structured clauses are pushed into SQL before hydration. 3. The keystone columns (index v16) and idx_blocks_type_tool (v20) are used for these predicates. Verify: instrumentation on a broad SEQ or referenced_path query shows fact builds reduced to <=1 per candidate and hydration limited to predicate-surviving candidates (before/after in the PR); `devtools test` selection on runtime_matching/runtime_filters asserts memoization and that filter results match the pre-change path.

## Static mechanism / likely defect

Issue description localizes the mechanism: matches_action_sequence, matches_referenced_path, and category matching each call _actions_for(session) -> build_session_semantic_facts (runtime_matching.py:20-25) — full semantic-fact construction over a hydrated session, no memoization across the three matchers, applied as list-comprehension post-filter (runtime_filters.py:188-189). A broad query with SEQ or referenced_path hydrates every SQL-surviving candidate and builds facts up to 3x. Design direction: Minimal fix: memoize facts per session within a filter pass (functools cache keyed per pass, or attach _semantic_facts to the Session object). Real fix: all three matchers' predicates (action category, affected path, sequence) are answerable from actions-view rows — fetch once per candidate set with a single WHERE session_id IN (...) query, group in Python, drop hydration entirely for candidates failing cheap predic…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.

## Implementation plan

1. Minimal fix: memoize facts per session within a filter pass (functools cache keyed per pass, or attach _semantic_facts to the Session object).
2. Real fix: all three matchers' predicates (action category, affected path, sequence) are answerable from actions-view rows — fetch once per candidate set with a single WHERE session_id IN (...) query, group in Python, drop hydration entirely for candidates failing cheap predicates.
3. The keystone columns (v16) and idx_blocks_type_tool (v20) exist for exactly this shape.
4. Also push cheap structured clauses into SQL before hydration.
5. SEQ span capture (DSL bead) builds on the same relation — coordinate.

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: Minimal fix: semantic facts are memoized per session within a single filter pass (no more than one build_session_semantic_facts per session per pass), eliminating the up-to-3x construction across matches_action_sequence / matches_referenced_path / category matching (runtime_matching.py, runtime_filters.py).
- Acceptance proof: 2.
- Acceptance proof: Real fix: the three matchers' predicates (action category, affected path, sequence) are answered from actions-view rows fetched once per candidate set with a single `WHERE session_id IN (...)` query, grouped in Python
- Acceptance proof: candidates failing cheap predicates are dropped before hydration, and cheap structured clauses are pushed into SQL before hydration.
- Acceptance proof: 3.
- Acceptance proof: The keystone columns (index v16) and idx_blocks_type_tool (v20) are used for these predicates.
- Acceptance proof: Verify: instrumentation on a broad SEQ or referenced_path query shows fact builds reduced to <=1 per candidate and hydration limited to predicate-surviving candidates (before/after in the PR)

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
