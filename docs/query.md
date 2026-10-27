# Queries and reads

A query selects rows, then applies its stages in order. Conditions without
a unit select sessions. An explicit unit, such as `messages where …` or
`tool_calls where …`, selects those rows. The [grammar](../crates/query/src/query.pest),
[unit declarations](../crates/query/src/units.rs) and
[field declarations](../crates/query/src/fields.rs) own accepted syntax and
names. Use `polylogue explain` for the installed declarations and a query's
compiled explanation; `--why` also counts its conditions.

```text
origin:claude-code "chrono"
messages where written_by:human "chrono"
tool_calls where outcome:error | group by session | count
sessions where messages >= 10 | count
```

Juxtaposition means AND; `OR`, `NOT`, a leading `-` and parentheses combine
conditions. Quoting keeps spaces inside a value. `FIELD:*` includes known and unknown values; `FIELD:unknown` selects
unknown ones where the field supports these forms. Numeric comparisons and ranges use the field's declared type.
An unknown field or an invalid closed-set value is an error, not an empty
answer. IDs are opaque; a field naming a session resolves an exact ID before
a unique prefix and refuses ambiguity.

Full-text conditions search the unit's text. `near:"words"` requests semantic
ranking when embeddings are enabled. [Architecture](architecture.md#reading)
owns ranking, combining queries, total ordering and cursor behavior. A saved
query is a note named by `query:NAME`, optionally called with parameters as
`query:NAME(word=VALUE)`. [The compiler](../crates/query/src/compile.rs) checks
its definition and binds values, rather than interpolating SQL text.

A scope stage changes the listed unit within the selected sessions.
`group by`, `agg`, `count`, `sort by`, `where`, `limit`, `offset` and `distinct`
operate on the stage before them. `| matches` exposes matching child rows
with their `parent_id`; `exists`, `count(UNIT where …)` and `seq(… -> …)`
select through related rows. Supported combinations are declared and tested
in [Query](../crates/query/src) and [application query tests](../crates/polylogue/src/daemon/query/tests).
A parsed expression can still be refused when its unit or stage combination
is unsupported.

Dates and relative dates use the [logical day](vocabulary.md#queries-and-output),
including the configured zone and day start. Unknown times do not become
midnight or zero. A query answer includes its rows and count, with a next
cursor when there are more rows. Follow the cursor to read the answer;
`limit` deliberately changes the answer itself.

## Reading content

Session reads offer `full`, `dialogue`, `prompts` and `actions` views.
[Render](../crates/render/src/lib.rs) defines them: full keeps every block;
dialogue keeps text and thinking; prompts keeps text from user messages
that are neither harness nor model prompts; actions keeps tool calls and
results. Terminal, Markdown, HTML and JSON share that selection. Terminal
tool-result previews show the first 20 lines; Markdown carries the complete
selected result. Structured exchange is documented in [export-format.md](export-format.md).

`--with-parents` reads parent messages before a session's own messages.
Budgeted reads return masked text and structured block slices, preserving
source message IDs and block indices. [Web reader](web-reader.md) owns the
read-page contract, anchors and outlines. A budget bounds a page's output,
not the size of a decoded message.
