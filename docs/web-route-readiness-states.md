# Web route readiness states

The local web reader uses explicit route readiness payloads for panels where a
zero count is not enough evidence. A route can be ready with results, ready but
empty, ready with no scoped matches, or degraded because an underlying archive
component could not answer.

## `/api/sessions`

`GET /api/sessions` returns either the list envelope (`items`) or the ranked
search envelope (`hits`) and includes `route_state`:

- `ready`: the route completed and returned at least one session or hit.
- `empty`: the unfiltered archive list completed and the archive contains no
  sessions.
- `no_results`: the route completed but an active query or filter matched no
  sessions.
- `degraded`: the route could not compute search results because the message
  search index was unavailable. In this case `hits` is empty, `total` is `null`,
  `route_state.component` is `message_fts`, and `diagnostics.reasons` includes
  `search_index_degraded`.

The web shell consumes this field before rendering the sidebar empty state. A
failed or degraded sessions route renders a route-state notice with DOM markers
(`data-route-state-name="sessionList"` and `data-route-state="..."`) instead of
falling through to “No sessions in archive.” Empty and no-result states keep the
regular sidebar empty DOM marker (`data-sidebar-state="empty"` or
`data-sidebar-state="noresults"`).

## Facets first paint

`GET /api/facets` carries per-family readiness through `family_status` plus the
summary fields `complete_families`, `deferred_families`, `stale`, and
`budget_exceeded`. Deferred facet families are not rendered as failed or empty;
they remain explicit so the first paint can show cheap counts while the shell or
a caller can request expensive families separately.
