Title: "WebUI v2 vertical: search with snippets, provenance refs, facets, and lossless continuation paging"

Result ZIP: `webui-03-search-vertical-r01.zip`

## Mission

Build the search vertical for the WebUI v2 workspace (TypeScript + Preact +
Vite, daemon SSR + islands; if the webui-01 scaffold result is not among your
inputs, define the minimal scaffold interface you need and state it in
HANDOFF.md — do not build a second scaffold).

Polylogue search is layered: FTS5 (contentless, `unicode61`, over
`blocks.search_text`), semantic/vector (Voyage embeddings, `embeddings.db`),
and the Lark query DSL (`polylogue/archive/query/expression.py`) with fielded
predicates (`repo:x`, `since:7d`, `origin:codex-session`), booleans,
`near:"…"`, and unit pipelines. Read `docs/search.md` and the daemon's
existing search/query JSON routes in `polylogue/daemon/http.py` +
`web_shell_reader.py` before designing.

Deliver:

1. A search page: query input teaching the DSL progressively (placeholder
   examples that MUST be copied from strings that round-trip the real parser
   — extract them from repo tests or generate via the parser, never invent),
   result list with snippets, per-hit provenance (origin badge, session
   title/ref, message anchor deep-link into the webui-02 read page), and
   facet sidebar (origin, time bucket, repo) driven by server-computed facet
   JSON.
2. Continuation-based paging ONLY (opaque cursor from the shared
   QueryTransaction — inspect `polylogue/archive/query/transaction.py` for
   the page/continuation/result-ref vocabulary). No "load everything" mode;
   totals render as exact-vs-qualified per the transaction's coverage field.
3. Honest degraded states: FTS lagging convergence, embeddings absent, or
   query-parse errors each render distinctly; a parse error shows the
   parser's diagnostic and a corrected example, never a silent empty result.
4. SSR the first page (readable without JS); islands hydrate paging/facets.
5. Vitest tests: paging continuation flow, parse-error rendering, facet
   selection; Python route test asserting SSR skeleton + JSON contract
   fields the client depends on.

## Constraints

- Semantics live server-side: the client NEVER re-filters/re-ranks; it
  renders what the daemon returns (surfaces-project-only rule from
  `docs/architecture-spine.md`).
- Bead context to read in the snapshot's `.beads/issues.jsonl`:
  `polylogue-z9gh.9.1` (bounded resumable reads — your JSON contract),
  `polylogue-4p1` (Query × Projection × Render read algebra — align naming),
  `polylogue-t46.8` (the six-tool MCP surface being landed in parallel: the
  web JSON and MCP `query` tool should converge on the same request shape;
  design the client request builder so it could emit that shape).
- Zero CDN; sanitized fixtures only.

## Deliverable emphasis

HANDOFF.md: JSON contracts consumed (exact fields), the shared continuation/
request-builder utilities added for other verticals, DSL example corpus used
(with provenance: which repo test/file each example came from), degraded-state
matrix, and old `web_shell_*` routes/files this vertical supersedes (deletion
candidates listed, not deleted).
