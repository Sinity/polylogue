# 040. polylogue-4p1.1 — Route daemon split-archive fast path through SessionQuerySpec.from_params

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

polylogue/daemon/http.py:1970 documents that the split-archive fast path intentionally does not construct a SessionQuerySpec and hand-mirrors every public structured filter (has_paste, has_tool_use, has_thinking, repo, has_type, tool/exclude_tool, action/exclude_action/action_sequence/action_text, referenced_path, cwd_prefix, title, min/max_messages, min/max_words, plus the shared _filter_kw block). This is a parallel implementation of build_query_spec_from_params (polylogue/archive/query/spec.py:498). A filter field added to the spec builder is silently absent from the daemon fast path until someone edits both sites. Collapse it: have the fast path build a SessionQuerySpec via from_params and read its lowered filter fields, keeping only the genuinely count/summary-specific plumbing (session_id passed separately) outside the spec. Prove parity with a test that enumerates SessionQuerySpec filter fields and asserts each is honored by the fast path.

## Acceptance criteria

The daemon split-archive list/search/count path derives all structured filters from a SessionQuerySpec built via from_params (no per-field re-read of HTTP params for filters the spec already models); a test enumerates SessionQuerySpec filter attributes and fails if the fast path drops any; the in-code 'must mirror those public params here' comment and its manual mirroring block are removed; render surfaces (openapi/cli-output-schemas) still verify.

## Static mechanism / likely defect

Design direction: polylogue/daemon/http.py:1970 documents that the split-archive fast path intentionally does not construct a SessionQuerySpec and hand-mirrors every public structured filter (has_paste, has_tool_use, has_thinking, repo, has_type, tool/exclude_tool, action/exclude_action/action_sequence/action_text, referenced_path, cwd_prefix, title, min/max_messages, min/max_words, plus the shared _filter_kw block). This is a parall…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. polylogue/daemon/http.py:1970 documents that the split-archive fast path intentionally does not construct a SessionQuerySpec and hand-mirrors every public structured filter (has_paste, has_tool_use, has_thinking, repo, has_type, tool/exclude_tool, action/exclude_action/action_sequence/action_text, referenced_path, cwd_prefix, title, min/max_messages, min/max_words, plus the shared _filter_kw block).
2. This is a parallel implementation of build_query_spec_from_params (polylogue/archive/query/spec.py:498).
3. A filter field added to the spec builder is silently absent from the daemon fast path until someone edits both sites.
4. Collapse it: have the fast path build a SessionQuerySpec via from_params and read its lowered filter fields, keeping only the genuinely count/summary-specific plumbing (session_id passed separately) outside the spec.
5. Prove parity with a test that enumerates SessionQuerySpec filter fields and asserts each is honored by the fast path.

## Tests to add

- Acceptance proof: The daemon split-archive list/search/count path derives all structured filters from a SessionQuerySpec built via from_params (no per-field re-read of HTTP params for filters the spec already models)
- Acceptance proof: a test enumerates SessionQuerySpec filter attributes and fails if the fast path drops any
- Acceptance proof: the in-code 'must mirror those public params here' comment and its manual mirroring block are removed
- Acceptance proof: render surfaces (openapi/cli-output-schemas) still verify.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
