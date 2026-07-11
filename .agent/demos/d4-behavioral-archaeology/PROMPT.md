# D4 "Behavioral Archaeology": Six DSL Queries, Rapid Fire

Predeclaration receipt: `artifact:d4-behavioral-archaeology-predeclaration`.

Run these six queries against a seeded demo archive (`polylogue demo seed`).
Each answers a question an engineering lead would ask about their team's AI
coding sessions — each impossible to answer from a chat UI transcript view.
Product primitives only (`polylogue` CLI query DSL); no bespoke scripts.

1. **SEQ thrash-loop hunt** — repeated shell-tool calls in a row:
   `polylogue find "sessions where seq(action:shell -> action:shell)" then select --json`
2. **Tool call volume** — which tools are actually used:
   `polylogue "actions where exit_code:>=0 | group by tool | count"`
3. **Which tools break** — failure count by tool:
   `polylogue "actions where is_error:true | group by tool | count"`
4. **Semantic probe across providers**:
   `polylogue find 'near:"flaky async test"'`
5. **Time-scoped session population**:
   `polylogue find "since:2y"`
6. **Pipe a query straight into `read`**:
   `polylogue find "origin:codex-session" then read --first --view messages`

Then show `explain_query_expression` (CLI: `--explain`) once on query 1 to
prove the query means what it says — the parsed AST, not just prose.

## Note on this run

While authoring query 1 for this specific seeded fixture, running the exact
same predicate as a bare `find` (no `then` verb) vs `find ... then select`
produced DIFFERENT results — the bare form silently ignored the filter. This
is documented as a real finding in `report.md` (counterexamples) and filed
as its own bug (polylogue-70qb), not hidden. This
IS the point of the demo: a DSL query surfaces things a chat transcript
never could — including, in this case, a defect in the query surface
itself.
