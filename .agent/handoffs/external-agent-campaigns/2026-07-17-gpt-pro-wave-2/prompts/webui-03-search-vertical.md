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


---

## Context and authority

You are a long-running ChatGPT Pro engineering worker. A recent Polylogue
project-state archive will be attached. Retrieve and inspect it broadly; do not
assume attachment bytes consume your active prompt context. The attached
snapshot is the code authority. This prompt defines your mission. Repository
instructions and complete relevant Beads records define constraints and intent;
later Beads notes may supersede older descriptions. Current source wins when a
stale plan names paths or APIs that no longer exist.

Start by reporting the snapshot commit/branch/dirty-patch identity you found and
the source, tests, Beads, and history you inspected. Follow dependencies beyond
the obvious files when they affect the production route. Do not invent an API,
test helper, product contract, or parallel framework to make the task easy.

## Working contract

- Produce the largest internally coherent implementation draft that fits the
  mission. Prefer one real end-to-end behavior over disconnected scaffolding.
- Preserve Polylogue's substrate-first architecture and existing typed
  interfaces. Small production seams are allowed only when real production
  behavior needs observation or control.
- Write concrete production changes and real-route tests. A test must name the
  production dependency it exercises and the representative implementation
  mutation/removal that should make it fail.
- Do not delete existing tests or helpers. Identify proposed dominated
  deletions separately for independent local certification.
- Use your container and run meaningful self-contained checks when possible.
  Never claim access to the operator's live daemon, browser, archive, secrets,
  NixOS deployment, or current worktree. Mark those checks `unverified`.
- If the full scope is unsafe, complete the strongest coherent subset and make
  the remaining decisions and exact continuation steps explicit. Do not return
  placeholders, ellipses, pseudocode presented as code, or a generic plan in
  place of implementation.

## Deliverable

Create the exact `Result ZIP` named near the top of this prompt under
`/mnt/data/`. Do not include the supplied repository/project-state archive or
other copied inputs in the result. The finished ZIP must be attached to the
conversation through a working, user-clickable download link. Work left only
in an internal shell directory, temporary notebook, scattered sandbox files,
or prose is not delivered.

The ZIP must contain:

- `HANDOFF.md`: mission, snapshot identity, inspected evidence, mechanism,
  decisions, changed files, acceptance matrix, apply order, risks, and exact
  verification performed/remaining;
- `PATCH.diff`: one apply-ready unified diff against the named snapshot;
- `TESTS.md`: test design, production dependencies, anti-vacuity mutation,
  commands, and honest execution results;
- `EVIDENCE.md`: relevant source/Bead/history findings and any contradictions;
- `FILES/`: complete replacements only where they materially disambiguate the
  patch; omit it when unnecessary.

Before answering, reopen the ZIP, list and validate its members, compute its
SHA-256 and byte size, and confirm that `PATCH.diff` has no placeholders or
copied source snapshot. Your final chat response must begin with a substantive
operator-readable report of what you did and why. It must also state important
limitations, missing or unverified work, and how much additional value another
iteration could plausibly add—distinguishing a small repair from a substantial
second pass. Then report verification and risks and give a prominent working
link to the exact `/mnt/data/` ZIP. A bare download receipt is not acceptable.

## Continuation protocol

Do not perform a separate adversarial review unless the user explicitly asks
for one. If the user asks to **iterate** or **continue**, preserve valid prior
work, perform the highest-value remaining implementation/research pass, and
publish a new cohesive package revision with the same complete structure—not a
loose supplemental patch. Explain exactly what changed, what improved, what
still remains, and whether another iteration is likely to pay off.

If the user explicitly asks for an **adversarial review**, attack your prior
result against the original mission and current attached authority: search for
unsupported claims, invented or stale APIs, missing call sites, composition
failures, unsafe assumptions, vacuous tests, patch/apply defects, incomplete
acceptance criteria, and evidence that would falsify the design. Preserve work
that survives. Then repair every legitimate finding you can, regenerate the
entire cohesive package as the next revision, and report findings, repairs,
remaining disputes, and the value of another adversarial/implementation pass.
