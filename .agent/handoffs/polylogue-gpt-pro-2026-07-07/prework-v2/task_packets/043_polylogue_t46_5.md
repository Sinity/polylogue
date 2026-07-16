# 043. polylogue-t46.5 — Route CLI transcript/dialogue file export through substrate read+render; delete streaming_markdown SQL path

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

_No description in export._

## Existing design note

cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_envelope + rendering/core_markdown + rendering/blocks. It deliberately bails (returns False) on prefix-sharing sessions, so forked/resumed session file exports silently diverge. Fix: expose a streaming/iterator markdown render over the substrate read (add an iter/stream method on the facade or reuse get_messages_paginated) so standard.py:85/:119 use the same composition+block-filtering as the non-streaming path; delete streaming_markdown.py's SQL. Keep the no-buffering benefit by streaming from the paginated substrate read.

## Acceptance criteria

streaming_markdown.py raw-SQL read helpers are deleted; transcript/dialogue --to file markdown for a prefix-sharing (forked/resumed) session composes the full lineage identically to stdout output (test compares file export bytes vs the substrate transcript for a forked session); block filtering (reasoning/prose) matches the substrate projection; devtools verify green.

## Static mechanism / likely defect

Design direction: cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_en…

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. cli/read_views/streaming_markdown.py forks the whole read path for `read --view transcript/dialogue --to file` markdown exports: its own read-only index.db connection, ref resolution (_resolve_session_id), prefix-sharing lineage gating in raw SQL (_has_prefix_sharing_edge), _table_exists, and the message+block keyset join + block filtering -- duplicating api get_session/get_messages_paginated/read_archive_session_…
2. It deliberately bails (returns False) on prefix-sharing sessions, so forked/resumed session file exports silently diverge.
3. Fix: expose a streaming/iterator markdown render over the substrate read (add an iter/stream method on the facade or reuse get_messages_paginated) so standard.py:85/:119 use the same composition+block-filtering as the non-streaming path
4. delete streaming_markdown.py's SQL.
5. Keep the no-buffering benefit by streaming from the paginated substrate read.

## Tests to add

- Acceptance proof: streaming_markdown.py raw-SQL read helpers are deleted
- Acceptance proof: transcript/dialogue --to file markdown for a prefix-sharing (forked/resumed) session composes the full lineage identically to stdout output (test compares file export bytes vs the substrate transcript for a forked session)
- Acceptance proof: block filtering (reasoning/prose) matches the substrate projection
- Acceptance proof: devtools verify green.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
