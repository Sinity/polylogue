# Analyze projections

## Decision

`analyze` is a group of named read projections. Each projection owns its
parameters and calls the existing query, facet, cost, postmortem, or portfolio
service. The top-level `facets` command remains the archive-wide entry point;
`analyze facets` applies the same projection to the current query scope.

The named projections are `count`, `by`, `facets`, `cost-outlook`,
`postmortem`, and `portfolio`. The group with no projection retains the
default aggregate statistics view. A projection produces the same JSON and
human-readable payload as its service already defines. Projection selection is
therefore structural, while filtering remains owned by `RootModeRequest`.

## Execution contract

The query parser builds one `RootModeRequest` before the action runs. A named
projection changes only the terminal projection fields on that request. It does
not create a second filter parser, result relation, or surface-specific
aggregation path. `analyze facets` and `facets` share `Polylogue.facets` and
the `FacetsResponse` envelope. Empty scopes preserve the existing zero-result
or empty aggregate payload, including its diagnostics where the underlying
service supplies them.

The legacy flag forms remain accepted during this migration so saved query
workflows and generated help do not break at the parser boundary. They are
translated immediately into the named projection and are not separate
execution modes. New documentation and completion examples use named
projections.

## Proof

The focused proof covers a seeded positive scope and an empty scope for the
named `count`, `by`, and `facets` projections, plus the JSON envelope and
top-level facet parity. The managed CLI contract suite and generated CLI
reference check are the release-gate evidence. The broader daemon/MCP/Python
query parity and content-hash citation drift fixtures remain shared gate
checks because this change only selects existing read projections.
