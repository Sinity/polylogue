Title: "Query discovery corpus: parser-valid examples classified by result semantics, generated from executable declarations"

Result ZIP: `mcp-04-discovery-corpus-r01.zip`

## Mission

Bead `polylogue-z9gh.3` (P0) documents the failure: shipped prompts and the
installed agent skill teach query expressions the parser REJECTS, and
discovery does not distinguish result semantics — exhaustive relations vs
top-k vs samples vs aggregates vs bounded context vs recursive pages — so
agents misread what they got. Build the corpus + generation machinery that
makes discovery provably truthful.

Ground truth (snapshot):

- The real grammar: `polylogue/archive/query/expression.py` (Lark DSL:
  fielded predicates, booleans, `near:"…"`, count/date ranges, `with
  <units>` projection, pipeline stages `sessions where … | group by … |
  count` over unit sources sessions/actions/messages/observed-events;
  pipeline stages are hand-parsed OUTSIDE the grammar, split on `|`;
  terminal priorities matter — any new terminal containing ':' must slot
  above FIELD_CLAUSE.4). Also the strict command floor (bead ref #1842
  semantics in `cli/query_group.py`).
- Existing teaching surfaces to audit for lies: MCP recipe prompts
  (`polylogue/mcp/` prompt registrations), `docs/search.md`,
  `docs/cli-reference.md` examples, the query_completions MCP tool, and
  any skill/manual text in the repo.
- The declaration direction: `z9gh.3` design — discovery/schemas/examples/
  errors/completions/docs generated from executable query declarations, not
  hand-maintained prose. Coordinate shape with `polylogue/mcp/declarations/`
  (t46.8.1 pilot) if present.

Deliver:

1. **The corpus**: 80–150 examples as structured data (one file, typed
   rows): expression string, unit source, what it answers (one sentence),
   result-semantics class (exhaustive / top-k / sample / aggregate /
   bounded-context / recursive-page), expected projection columns, cost
   class (selective vs corpus-scale), and 10–20 NEGATIVE examples (common
   invalid forms agents actually produce) each with the parser's actual
   diagnostic and the corrected form.
2. **The executable gate**: a test that parses EVERY positive example
   through the real Lark grammar + pipeline splitter (no mocks) and asserts
   the negative examples fail with the documented diagnostic class. This
   test is the anti-lying invariant: any teaching surface that imports its
   examples from this corpus cannot ship a rejected expression.
3. **Audit + rewiring**: audit every existing teaching surface against the
   corpus; produce the diff/PR that makes them import from it (or generate
   from it) instead of hand-written strings; list every currently-shipped
   invalid example found (each is evidence for z9gh.3).
4. **Result-semantics teaching**: for each semantics class, the exact
   phrasing discovery should attach (e.g. "top-k by relevance, not
   exhaustive; total is qualified") — aligned with QueryTransaction's
   coverage/total vocabulary (`archive/query/transaction.py`).

## Constraints

- No grammar changes. If an example SHOULD work but the grammar rejects it,
  record it as a grammar-gap finding (separate section), don't extend the
  grammar here.
- Corpus rows must be provider-neutral and privacy-safe.

## Deliverable emphasis

HANDOFF.md: corpus design, the shipped-invalid-examples census (file:line),
rewiring diff summary, grammar-gap findings, and how the corpus plugs into
z9gh.3's declaration-driven discovery + the six-tool `explain` tool.
