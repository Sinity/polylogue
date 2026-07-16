# 158. polylogue-ap7 — Semantic transcript rendering: tool-call-aware, provider-agnostic, shared CLI/web renderer registry

Priority/type/status: **P2 / task / open**. Lane: **04-read-contract-query-render**. Release: **C-read-contract**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Transcripts render today as generic message/block sequences — the same flat treatment for a prose paragraph, a 400-line Bash result, an Edit diff, and a Task dispatch. The archive KNOWS what these are (typed blocks, tool names, structural outcomes, normalized across providers) and renders none of that knowledge. Ambitious target (operator directive): chatlogs that read the way the work actually happened — every standard tool call rendered semantically, across every origin, in both the web reader and the terminal.

## Existing design note

(1) RENDERER REGISTRY keyed by (tool family, provider-normalized): Edit/Write -> syntax-highlighted diff/file cards (the yrx diff reconstruction is the data source — same computation, presentation here); Bash/shell -> terminal-styled block with command, collapsed output (first/last N lines, expand), exit badge from structural outcome; Read/Grep/Glob -> compact file-reference cards (path, range, match count) instead of dumped contents; Task/subagent dispatch -> a subagent card with the dispatch prompt, status, and link into the tree (mission-control bby.9 integration); WebFetch/WebSearch -> link cards with domain + title; MCP tools -> server-badged generic cards with payload folding; unknown tools -> the current generic rendering (never worse than today). Registry follows declare-once (o21) and the origin-normalized tool vocabulary (the blocks already carry tool_name + normalized family via the v20 expression index work). (2) PROSE: markdown rendered properly in both surfaces (web: the shared JS renderer post-bby.6; CLI: rich when tty); reasoning/thinking blocks collapsed by default with token count; attachments inline in web (images) / cards in CLI. (3) ONE REGISTRY, TWO BACKENDS: renderer specs (what to extract, how to fold) are shared data; web and terminal implement the same spec — snapshot tests assert both backends agree on structure (bby.6's contract concern, solved by construction). (4) Message permalinks + per-block anchors in web (feeds scd copy-affordances). (5) Layout profiles: 'operator reading' (compact, folded) vs 'forensic' (everything, offsets) vs 'presentation' (demo-grade, for 3tl recordings) — RenderSpec presets per 4p1, not new flags.

## Acceptance criteria

On the seeded corpus: Edit shows a highlighted diff, Bash shows exit-badged folded output, Task shows a linked subagent card — in BOTH web and CLI; unknown tools render as today; structure-parity snapshot tests green across backends; a before/after recording of one real session is committed as the demo asset (3tl.5 machinery).

## Static mechanism / likely defect

Issue description localizes the mechanism: Transcripts render today as generic message/block sequences — the same flat treatment for a prose paragraph, a 400-line Bash result, an Edit diff, and a Task dispatch. The archive KNOWS what these are (typed blocks, tool names, structural outcomes, normalized across providers) and renders none of that knowledge. Ambitious target (operator directive): chatlogs that read the way the work actually happened — every standard tool call rendered semantically, across every origin, in both the web reader and the terminal. Design direction: (1) RENDERER REGISTRY keyed by (tool family, provider-normalized): Edit/Write -> syntax-highlighted diff/file cards (the yrx diff reconstruction is the data source — same computation, presentation here); Bash/shell -> terminal-styled block with command, collapsed output (first/last N lines, expand), exit badge from structural outcome; Read/Grep/Glob -> compact file-reference cards (path, range, match count) instead …

## Source anchors to inspect first

- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. (1) RENDERER REGISTRY keyed by (tool family, provider-normalized): Edit/Write -> syntax-highlighted diff/file cards (the yrx diff reconstruction is the data source — same computation, presentation here)
2. Bash/shell -> terminal-styled block with command, collapsed output (first/last N lines, expand), exit badge from structural outcome
3. Read/Grep/Glob -> compact file-reference cards (path, range, match count) instead of dumped contents
4. Task/subagent dispatch -> a subagent card with the dispatch prompt, status, and link into the tree (mission-control bby.9 integration)
5. WebFetch/WebSearch -> link cards with domain + title
6. MCP tools -> server-badged generic cards with payload folding
7. unknown tools -> the current generic rendering (never worse than today).

## Tests to add

- Acceptance proof: On the seeded corpus: Edit shows a highlighted diff, Bash shows exit-badged folded output, Task shows a linked subagent card — in BOTH web and CLI
- Acceptance proof: unknown tools render as today
- Acceptance proof: structure-parity snapshot tests green across backends
- Acceptance proof: a before/after recording of one real session is committed as the demo asset (3tl.5 machinery).

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit tests/integration -k "query or contract or citation or insight"`

## Pitfalls

- Do not broaden scope beyond the bead acceptance criteria; make a failing test first, then patch the smallest shared seam.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
