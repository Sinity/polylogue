# Query pipeline substrate

Owning issue: #2006.

## Purpose

Polylogue needs a richer query language without becoming its own query engine. The surface language should compile onto the archive storage and search layers already present in the project. The Lark grammar in `polylogue/archive/query/expression.py` is the query grammar; compact field/text clauses and explicit Boolean predicates are entry shapes in that grammar, not separate floor/ceiling languages.

## Current limitation

The current implemented query substrate already handles compact session filters, grouped Boolean predicates, message/action/block `exists` predicates, ordered action sequences, FTS predicates, lineage predicates, and a semantic seed plus residual filter. Remaining scope is broader unit coverage and pipeline execution: runs/events/assertions, aggregation, unit-changing traversal, terminal read/analyze/bundle stages, and shared completion/query-builder metadata.

## Design decision

Build a typed AST and lowering layer.

Predicate nodes: And, Or, Not, Leaf, Fts, Semantic, Structural, Sequence, Lineage, Relational.

Pipeline stages: source or filter, traverse, transform, aggregate, terminal action.

Implemented units: session, message, action, block, lineage.

Target units still needing real lowerers: run, observed event, assertion, context snapshot, bundle/work packet, external work refs, phase, thread, span.

## Surface ladder

Compact examples: repo filters, origin filters, tags, date filters, phrases, and the current `find QUERY then ACTION` shape.

Power examples: grouped conditions, numeric operators, semantic clauses, message predicates, sequence predicates, lineage predicates, and pipelines that change unit from sessions to messages or lineage and back.

## Implementation slices

1. Keep compact and explicit Boolean syntax on the same Lark grammar and AST path.
2. Extend explain output to show unsupported unit/pipeline stages, not just the lowered `SessionQuerySpec`.
3. Add run/event/assertion/context units only with real lowerers and fixtures.
4. Add traversal stages that change the active unit and lower to SQL/recursive CTEs or existing read models.
5. Add aggregation/sort/limit stages over supported units.
6. Lower terminal stages through existing read/analyze/bundle/action contracts.
7. Add completion/query-builder metadata from the same grammar, unit, field, operator, and action registries.

## Acceptance criteria

- The AST represents compact and explicit query syntax through one grammar.
- Unsupported forms fail with typed errors and do not broaden results.
- CLI, daemon, MCP, web, and completion can share the same parser and AST.
- Natural-language query tools target the AST.
- #1842 can ship its command-floor work without waiting for this full ceiling.
