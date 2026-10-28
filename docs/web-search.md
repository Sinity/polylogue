# Search operations for the workbench

Search uses the daemon's ordinary query request on the socket, HTTP and MCP. Facet counts are `group by FIELD | count` queries over the same filtered source, covering the whole population rather than the visible result page. Their distinct values can be paged independently. The query specification owns the completion protocol and the batched `facets` request.

The daemon's existing word completion includes declared closed-set values and observed message models. Values containing spaces, quotes or backslashes are quoted as query literals; an unquoted prefix can find a quoted value. Completion suggestions are distinct from facet counts. Model values use a covering index added when the archive opens, without rebuilding parsed rows. Whole-query cursor completion and warm value counts are specified in the query specification.

Save query records a name that runs current data. Keep results records every matching row in daemon order, including filtered semantic or mixed ranking. Both use the command-table writes at `/api/v1/save` and `/api/v1/keep`; Keep is also available through MCP. A retry reuses the first committed rows or saved-query revision for that request ID, including after the archive changes. Changing the intended name, query or author conflicts. Kept references accept quoted opaque IDs, including spaces, pipes and colons; a quoted `.matches` suffix is part of the ID, while unquoted `result:ID.matches` selects the recorded matches.

Live-state derivation reads the stored hook payload after preparation releases its parsed JSON. Waiting reasons, current tools and todo descriptions therefore survive the same preparation path as other event facts.

The opt-in `web::tests::search_measurements::search_measurements` test measures the first and 30 repeated daemon reads for live, text, grouped facet, semantic and mixed queries over 121 synthetic sessions. Run it through AgentCTL's measure operation. It uses a local synthetic embedding provider and reports that provider's response time separately.

## Requirements for the maud workbench

The workbench supplies typed answers to server-rendered maud pages and uses the shared TypeScript shell. Its required interactions are:

- A Running list ordered by `live_rank, last_event_at desc`, excluding ended sessions; a Recent list includes ended sessions. Rows open the shared reader at the latest message. Show recorded state and waiting evidence. Load notices independently and link to deliveries.
- Live updates that pause while hidden, refresh on return and after overlapping changes, and offer Apply updates while a row has focus. Preserve focus by session reference, including a focused row that leaves the result. Provide Pause updates and ordinary Refresh/Next links. Use the shared SSE feed planned for the shell.
- Search results rendered from the same query answer as other clients, with ranked excerpts, reference links, opaque Next cursors and Previous for visited pages. Delay completion requests and abort superseded requests.
- Facet controls from the shared field declarations for origin, repository, harness, model, time, live state and tags where applicable. Edit parser spans, quote inserted literals and reset the cursor. Preserve parentheses, ranking terms and sort clauses. Expand a leading saved query before editing; explain unsupported attached, combined, scoped, kept or transformed shapes. Never wrap a saved source as a condition if doing so changes its binding.
- Fetch whole-population facet counts after the first result page through the batched facets request. Page distinct choices; distinguish suggestions, deferred counts and unsupported fields from zero.
- Save/keep forms that keep input and request IDs after failure, use the shared command writes, and offer Keep only for table answers. No browser storage holds drafts or results.
