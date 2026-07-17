# 055. polylogue-rxdo.2 — Content-addressed query identity + durable user.db queries/query_names/result_sets/query_edges

Priority/type/status: **P2 / task / open**. Lane: **05-analysis-provenance-citations**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

query:<hash> keyed on the canonical planned AST AFTER macro expansion (mirrors content-hash idempotency: equivalent queries collapse). Mutable human names are a separate git-branch-style pointer table (name mutable, hash immutable). Durable result_sets rows are MANIFESTS (grain, corpus_epoch, member_count, membership merkle root, ordered_rank_hash, exactness, persistence class); exact members durable only for watch/pinned/finding/cohort persistence. query_edges (operand-of/refines/supersedes/derived-from/same-as) emitted from the planner/EXPLAIN nodes, never retrofitted by string parsing.

## Existing design note

Canonicalization: expand macros -> typed AST -> NFC strings -> sort commutative AND/OR children -> preserve non-commutative pipeline/seq/except/sort/limit order -> canonical field aliases -> include grain+lane+rank policy -> relative-time queries hash the DYNAMIC ast while runs store resolved absolute bounds -> sha256 over sorted-key compact JSON. Two hashes on result sets because membership equality is not rank equality (set algebra needs the first, UX drift the second). user.db v4->v5 additive migration: queries, query_names, result_sets, result_set_members, query_edges tables — MUST batch with the other pending user-v5 candidates (see the durable-batch coordination bead) behind a verified backup manifest. Migrate existing SAVED_QUERY assertions: compile+hash each into queries, repoint the assertion at query:<hash>. Guards to encode from the corpus review: macro identity instability (hash expanded AST, names carry supersedes), supersedes/derived-from DAG acyclicity check at insert.

## Acceptance criteria

Same query text with reordered AND operands yields one query hash; @macro repoint does not change the hash of past runs; user-tier migration preserves all existing assertions (parity test); set-algebra grain is part of result-set identity so cross-grain member keys cannot collide. Verify: focused tests on canonicalization + migration test + devtools verify.

## Static mechanism / likely defect

Issue description localizes the mechanism: query:<hash> keyed on the canonical planned AST AFTER macro expansion (mirrors content-hash idempotency: equivalent queries collapse). Mutable human names are a separate git-branch-style pointer table (name mutable, hash immutable). Durable result_sets rows are MANIFESTS (grain, corpus_epoch, member_count, membership merkle root, ordered_rank_hash, exactness, persistence class); exact members durable only for watch/pinned/finding/cohort persistence. query_edges (operand-of/refines/supersedes/derived-from/same-as) … Design direction: Canonicalization: expand macros -> typed AST -> NFC strings -> sort commutative AND/OR children -> preserve non-commutative pipeline/seq/except/sort/limit order -> canonical field aliases -> include grain+lane+rank policy -> relative-time queries hash the DYNAMIC ast while runs store resolved absolute bounds -> sha256 over sorted-key compact JSON. Two hashes on result sets because membership equality is not rank equ…

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

1. Canonicalization: expand macros -> typed AST -> NFC strings -> sort commutative AND/OR children -> preserve non-commutative pipeline/seq/except/sort/limit order -> canonical field aliases -> include grain+lane+rank policy -> relative-time queries hash the DYNAMIC ast while runs store resolved absolute bounds -> sha256 over sorted-key compact JSON.
2. Two hashes on result sets because membership equality is not rank equality (set algebra needs the first, UX drift the second).
3. user.db v4->v5 additive migration: queries, query_names, result_sets, result_set_members, query_edges tables — MUST batch with the other pending user-v5 candidates (see the durable-batch coordination bead) behind a verified backup manifest.
4. Migrate existing SAVED_QUERY assertions: compile+hash each into queries, repoint the assertion at query:<hash>.
5. Guards to encode from the corpus review: macro identity instability (hash expanded AST, names carry supersedes), supersedes/derived-from DAG acyclicity check at insert.

## Tests to add

- Acceptance proof: Same query text with reordered AND operands yields one query hash
- Acceptance proof: @macro repoint does not change the hash of past runs
- Acceptance proof: user-tier migration preserves all existing assertions (parity test)
- Acceptance proof: set-algebra grain is part of result-set identity so cross-grain member keys cannot collide.
- Acceptance proof: Verify: focused tests on canonicalization + migration test + devtools verify.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
