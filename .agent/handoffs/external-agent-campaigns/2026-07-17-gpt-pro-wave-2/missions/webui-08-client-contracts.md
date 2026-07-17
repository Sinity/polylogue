Title: "WebUI v2 generated client contracts: one typed client from the daemon's declared read/operation surface"

Result ZIP: `webui-08-client-contracts-r01.zip`

## Mission

WebUI v2 verticals must not hand-write fetch calls against informally-known
JSON shapes — that recreates the marshaling triplication the repo is
actively killing. Build the generated-client layer: TypeScript types +
a thin typed client generated from the daemon's declared surface, so a
server-side contract change breaks the web build instead of production.

Ground truth to read first:

- `devtools render openapi` and its output (find the generated OpenAPI/
  schema artifacts in the snapshot — `devtools/` + `docs/` + `schemas/`);
  also `render cli-output-schemas` machinery, which shows how the repo
  already generates typed output contracts from Python models.
- The declaration direction: beads `polylogue-o21.1` (DeclarationSpec
  kernel: declare-once → generate registration/schemas/inventories/docs)
  and `polylogue-t46.8.1` (MCP declaration registry pilot at
  `polylogue/mcp/declarations/`). A parallel local lane is cutting MCP to a
  six-tool declared surface; the daemon HTTP JSON surface is expected to
  converge on the same declared read contract. Your generator should
  consume whatever declaration/schema artifact exists in the snapshot and
  be trivially retargetable when the declaration kernel lands.
- `polylogue/archive/query/transaction.py`: QueryResultPage/continuation/
  result-ref vocabulary — the paging envelope every read returns.

Deliver:

1. A generator (Node or Python script, run at build time, committed output):
   OpenAPI/declared-schema → TypeScript types + typed client functions with
   the continuation envelope modeled once (generic `Page<T>` with cursor,
   coverage exact/qualified, result refs). Deterministic output, diffable,
   with a `--check` mode for CI drift detection (mirror the repo's
   `render all --check` idiom).
2. Runtime client: same-origin fetch wrapper with typed errors (map the
   daemon's error envelope), deadline/abort support, and a continuation
   iterator utility (`for await (const page of client.query(...))`) that the
   verticals adopt.
3. Contract tests: golden-file tests pinning generated output for a fixture
   schema; a drift test that fails when the daemon schema and committed
   client diverge; unit tests for the continuation iterator including a
   truncation/qualified-total page.
4. Adoption notes for webui-02…06 (which hand-written calls each replaces).

## Constraints

- Do not invent a parallel schema source: consume the repo's generated
  artifacts; where a route lacks schema coverage, add the smallest schema
  declaration server-side following existing `render` machinery, and list
  every such addition.
- Zero runtime dependencies beyond what webui-01's scaffold establishes.

## Deliverable emphasis

HANDOFF.md: generator design, exact schema sources consumed, the Page/
continuation type contract (spelled fully), drift-check wiring for CI,
per-vertical adoption map, and how this retargets onto the o21.1/t46.8.1
declaration kernel when it lands.
