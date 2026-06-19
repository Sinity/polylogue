# Query pipeline substrate

Owning issue: #2006.

## Purpose

Polylogue needs a richer query language without becoming its own query engine.
The surface language should lower onto the archive storage and search layers
already present in the project. The Lark grammar in
`polylogue/archive/query/expression.py` is the query grammar; compact field/text
clauses and explicit Boolean predicates are entry shapes in that grammar, not
separate floor/ceiling languages.

## Current state

The current implemented query substrate already handles compact session filters,
grouped Boolean predicates, message/action/block/assertion `exists`
predicates, ordered action sequences, FTS predicates, lineage predicates, a
semantic seed plus residual filter, and terminal row-producing
`messages/actions/blocks/assertions/runs/observed-events/context-snapshots
where ...` queries. Runtime-transform terminal rows for runs, observed events,
and context snapshots lower through the recovery/run projection rather than a
SQL table, so unsupported scoped session fields must fail closed instead of
broadening results.

## Design decision

Build a typed AST and lowering layer.

Predicate nodes: And, Or, Not, Leaf, Fts, Semantic, Structural, Sequence, Lineage, Relational.

Pipeline stages: source or filter, traverse, transform, aggregate, sort/limit,
and terminal action or view.

Implemented units: session, message, action, block, assertion, run, observed
event, context snapshot, lineage.

Target units still needing real lowerers: bundle/work packet, external work
refs, phase, thread, span.

## Surface ladder

Compact examples: repo filters, origin filters, tags, date filters, phrases, and the current `find QUERY then ACTION` shape.

Power examples: grouped conditions, numeric operators, semantic clauses, message predicates, sequence predicates, lineage predicates, and pipelines that change unit from sessions to messages or lineage and back.

## Implementation phases

These are coherent PR-sized implementation phases, not conceptual
micro-slices. Each phase should land useful executable queries and tests.

1. Keep compact and explicit Boolean syntax on the same Lark grammar and AST
   path.
2. Extend explain output to show terminal unit sources, unsupported
   unit/pipeline stages, and the concrete lowerer/execution legs selected.
3. Keep runtime-transform unit execution covered across CLI, Python API, MCP,
   daemon, docs, generated schemas, and completions as new fields are added.
4. Add traversal stages that change the active unit and lower to SQL/recursive
   CTEs or existing read models.
5. Add aggregation/sort/limit stages over supported units.
6. Lower terminal stages through existing read/analyze/bundle/action contracts.
7. Add completion/query-builder metadata from the same grammar, unit, field,
   operator, and action registries.

## Acceptance criteria

- The AST represents compact and explicit query syntax through one grammar.
- Unsupported forms fail with typed errors and do not broaden results.
- CLI, daemon, MCP, web, and completion can share the same parser and AST.
- Natural-language query tools target the AST.
- Query surfaces should name and call the grammar/AST/lowering path directly.
  There is no separate compatibility compiler or floor grammar.
