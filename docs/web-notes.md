# Note operations

`note read ID` reads one exact revision, including superseded and withdrawn revisions, with its targets, key, relations and expiry. It is also available through MCP and `GET /api/v1/note/read?id=ID`. Note lists and history use queries: `notes where first_note = "ID" current:*` includes all revisions; `current:true` selects the current unwithdrawn revision. Explicit `=` and `!=` preserve identity case; `:` keeps prefix matching. Query projections can include `id`, `key`, `target` and `judged`, and the declarations appear in `explain`.

`note add` accepts `--key`, `--relations` (JSON pairs of relation and note ID), and `--valid-until` (RFC 3339 with a UTC offset). A caller creating a revision supplies the metadata to preserve alongside `--revises`. Saved queries and signals can have no targets. Ordinary notes require explicit targets or `--query`.

`note accept` and `note reject` judge an exact agent revision. Acceptance records the operator's judgment on that revision; the revision stays the agent's, and briefs show it with its author and the judgment. A later revision needs its own judgment. `notes where agent:true judged:unknown` selects current unwithdrawn agent revisions needing review.

`note add --query QUERY` applies a note to every session, message, document or note returned when the write is applied. It uses the public query's semantic and mixed ranking inputs and follows every result page. Explicit targets annotate the selected references instead. The response reports the committed target count. To fix membership in advance, `keep QUERY` records the rows and `--query result:ID` annotates those rows; a target of `result:ID` describes the kept result itself. `keep` is available through the guarded HTTP adapter.

Bulk annotation stores the query intent with its resolved targets. Retrying an identical `request_id` returns the original note and target count even if the query population changed or became empty. Changed intent with the same ID is a conflict. Exported user state includes the query and resolved targets, so replay does not rerun the query.

Bulk annotation, `keep` and `save` share one request comparison in the daemon and writer. It excludes resolved targets, kept rows and read time, and the revision that `save` resolved by name. A retry uses the first committed resolution. Explicit targets and `note add --revises` remain part of the caller's content, including for saved-query notes; changing them with the same request ID is a conflict. The stored `SaveQuery` write distinguishes a revision resolved by `save` from one supplied through `note add`. `keep` is available through MCP and HTTP.

## Workbench rebuild requirements

The note pages and their JavaScript are deferred to the workbench build using server-rendered maud over typed answers and the shared TypeScript shell. The existing exact-reference `/n/{id}` reader remains. The replacement needs:

- A note list with target, kind, author and judgment filters, Needs review, and continuation links.
- An exact revision reader with targets, keys, judgments, paged revision history, a newer-revision link, operator revision editing, and Accept/Reject for the displayed agent revision. Preserve key, relations and expiry when editing.
- Add note from a message header, plus selected-row and all-match annotation forms. Preview the full count, distinguish current membership from kept rows, report committed counts, and preserve the original query when its displayed label is masked.
- Shared shell writes with stable request IDs, committed-record confirmation, lost-reply recovery, draft retention on refusal, and a new ID when content changes. Draft lifetime is the current page.
- Saved-query links that run current data and kept-result links that show recorded rows. Reading, paging and history work through ordinary links; writes use the shared shell and existing HTTP access guards.
