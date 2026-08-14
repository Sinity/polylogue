[← Back to README](../README.md)

# Query-Action Workflows

Polylogue’s CLI is query-first: select archive material with `find`, then apply
an explicit action. The live command and output contracts are in the generated
[CLI reference](../cli-reference.md); this page explains the few rules that
matter across workflows.

## Selection rules

- Exact `id:` and `session:` references are identity filters. A miss returns no
  target; it never broadens into full-text search.
- Actions that need one session reject ambiguous result sets until `--first`,
  `--all`, or another action-specific selector makes the intent explicit.
- Aggregate analysis keeps the matched result set as its scope and does not
  invent a selected session.
- Mutating and destructive actions expose their preview or confirmation guard
  before execution.

## Common paths

```bash
# Inspect normalized messages from one exact session.
polylogue find id:SESSION then read --view messages --limit 20

# Compile successor context from one selected session.
polylogue find id:SESSION then continue --format json

# Analyze a query result set without selecting one row.
polylogue find 'repo:polylogue pytest' then analyze --facets --format json

# Preview deletion; execution requires the separate confirmation flags.
polylogue find id:SESSION then delete --dry-run

# Review assertion candidates through the dedicated judgment surface.
polylogue judge --target-ref session:SESSION --review --format json
```

`mark` owns user overlays on selected sessions—tags, star, pin, archive state,
and notes. `judge` owns candidate-assertion decisions. Keeping those routes
separate prevents a session annotation from becoming model-claim authority.

## Executable evidence

The examples above are backed by demo-archive golden paths in
`polylogue/product/workflows.py`. They execute the real Click commands against
a seeded archive and validate human and JSON results in
`tests/unit/product/test_query_action_workflows.py`. The runtime action metadata
served to CLI, daemon, MCP, and browser clients comes from
`polylogue/operations/action_contracts.py`; this document is not a second
machine-readable registry.

Run the behavior suite with:

```bash
devtools test tests/unit/product/test_query_action_workflows.py
```
