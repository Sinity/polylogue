# Query pipeline substrate

Owning issue: #2006.

## Purpose

Polylogue needs a richer query ceiling without becoming its own query engine. The surface language should compile onto the archive storage and search layers already present in the project. The old `SessionQuerySpec` / query-expression compiler is the floor; #2006 owns the full DSL ceiling.

## Current limitation

The current session query shape is a flat conjunction across fields, with alternatives only inside one field. It is a good floor, but it cannot represent grouped conditions, message predicates, sequence patterns, lineage traversal, or mixed lexical and semantic constraints.

## Design decision

Build a typed AST and lowering layer.

Predicate nodes: And, Or, Not, Leaf, Fts, Semantic, Structural, Relational.

Pipeline stages: source or filter, traverse, transform, aggregate, terminal action.

Units: session, message, block, action, phase, thread, lineage, work packet, KV, span, run.

## Surface ladder

Floor examples: repo filters, origin filters, tags, date filters, phrases, and the current `find QUERY then ACTION` shape.

Ceiling examples: grouped conditions, numeric operators, semantic clauses, message predicates, sequence predicates, lineage predicates, and pipelines that change unit from sessions to messages or lineage and back.

## Implementation slices

1. Add typed AST objects that can represent today's grammar.
2. Preserve the current fast path for valid flat queries.
3. Add explain output that shows parse tree, plan, and unsupported stages.
4. Add grouped condition lowering for supported fields.
5. Add lexical and semantic result legs.
6. Add message predicates and sequence predicates.
7. Add traversal stages.
8. Add saved queries and macros after the AST stabilizes.

## Acceptance criteria

- The AST can represent the current grammar and future staged extensions.
- Unsupported forms fail with typed errors and do not broaden results.
- CLI, daemon, MCP, web, and completion can share the same parser and AST.
- Natural-language query tools target the AST.
- #1842 can ship its command-floor work without waiting for this full ceiling.
