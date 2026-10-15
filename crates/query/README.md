# Registered Session query subset

One pure binder produces typed `SessionQuery` plans for CLI preflight and daemon execution. The daemon binds again before acquiring its store view; surfaces render its answer without evaluating filters.

Supported forms:

```text
sessions
origin:claude-code kind:main
claude-code:opaque-session-id
sessions where origin:claude-code text:"shared phrase" refact*
chrono | sort by last_active desc | limit 10
```

Juxtaposition and `AND` require every condition. Words contain letters/digits; quoted values are phrases. `kind:*` includes separate subagent rows. An exact ID drops the implicit subagent exclusion. Without a text term, rows use newest known selected Message time, unknown last, then descending ID. Text membership uses the same order and explicitly reports canonical relevance ranking unavailable. Words may occur in separate Text blocks of one Message; a phrase must stay within one Text block. `limit` selects after complete membership evaluation and sorting and leaves the candidate count visible. A stage after limit is refused.

Unsupported syntax returns a position and typed non-success; it never becomes plain text. Other fields/units, `OR`/`NOT`, grouping, attachments, general sorting, semantic search, commands, unknown CLI flags and continuation are unregistered. `find` is only CLI intent syntax; it is not another query language. This binder implements the declared subset, not a complete grammar for future slices.

Execution belongs to the engine and store. It enumerates a pinned selected body catalog with direct retained admission qualifications, rather than certifying search completion from EOF or the matching row set. Unknown author rows are excluded from human/model prose and counted. Linked subagent expansion, inherited/active-path completeness, Session reductions and collection freshness remain unqualified. Hits name retained Messages; snippets require a separate full-context credential presentation contract. Empty known matches with incomplete population remain degraded. Limits and cost admissions are documented in the top-level README.
