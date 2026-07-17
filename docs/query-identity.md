# Query identity contract

Analysis provenance has three public object-reference forms:

- `query:<sha256>` — immutable identity of a canonical expanded planned AST.
- `query-run:qr_<id>` — one disposable operational execution.
- `result-set:<id>` — a promoted durable result-manifest.

The query digest is SHA-256 over compact, sorted-key JSON of the expanded typed
AST plus `grain`, `lane`, and `rank_policy`. Every string is NFC-normalized;
AND/OR children are sorted, while pipelines, sequence, `except`, sort, and
limit retain their supplied order. Field aliases are projected to their
canonical field token before hashing. Relative time remains dynamic in a query
identity; each query run records the resolved absolute time bounds instead.

The `query`, `query-run`, and `result-set` ObjectRef kinds are registered in
`polylogue.core.refs`, so they are valid `target_ref`, `scope_ref`, and
`author_ref` values without a special assertion path.
