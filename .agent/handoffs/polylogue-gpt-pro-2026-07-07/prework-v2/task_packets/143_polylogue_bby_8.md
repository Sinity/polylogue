# 143. polylogue-bby.8 — Web reader perceived performance: virtualized list, streamed search, optimistic navigation

Priority/type/status: **P2 / feature / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Fluidity in the reader is perceived latency, not just server latency: the session list renders 16k+ rows into the DOM (scroll cost grows with archive size), search waits for full results before painting anything, clicking a session blocks on the full detail fetch, and every panel loads with spinners instead of skeletons. Even with a fast daemon the UI will feel sluggish until the client is engineered for perceived speed.

## Existing design note

Four standard techniques, applied to the existing SPA (sequence after bby.6 extracts the JS to real files — refactoring inline-string JS is not viable): (1) LIST VIRTUALIZATION: render only the viewport window of the session list (hand-rolled windowing is ~100 lines, no framework needed); constant DOM cost at any archive size. (2) SEARCH-AS-YOU-TYPE: debounced (~150ms) incremental search against the daemon (FTS is fast when ready), cancel in-flight requests on new keystrokes (AbortController), paint first page immediately with a 'more loading' tail — never a blank list while typing. (3) OPTIMISTIC NAVIGATION: clicking a session paints instantly from the list-row data (title/origin/date skeleton) while messages stream in; hover-prefetch the detail for the row under the cursor (the cache bead makes this nearly free). (4) CACHE-AND-REVALIDATE: client keeps a small LRU of visited sessions keyed by the archive cursor from the SSE channel — back-navigation is instant, invalidation is push-driven, stale is impossible by construction. Acceptance: interactions measured against the SLO tier budgets (first paint <300ms, list scroll 60fps at 20k sessions on the operator machine); the visual-tapes recording (3tl.5) doubles as the perceived-speed exhibit.

## Acceptance criteria

Session list scrolls at 60fps with 20k sessions (virtualized DOM stays constant-size). Search-as-you-type paints first results <300ms with a warm daemon and stale requests are cancelled. Back-navigation to a visited session renders instantly from client cache and revalidates via cursor.

## Static mechanism / likely defect

Issue description localizes the mechanism: Fluidity in the reader is perceived latency, not just server latency: the session list renders 16k+ rows into the DOM (scroll cost grows with archive size), search waits for full results before painting anything, clicking a session blocks on the full detail fetch, and every panel loads with spinners instead of skeletons. Even with a fast daemon the UI will feel sluggish until the client is engineered for perceived speed. Design direction: Four standard techniques, applied to the existing SPA (sequence after bby.6 extracts the JS to real files — refactoring inline-string JS is not viable): (1) LIST VIRTUALIZATION: render only the viewport window of the session list (hand-rolled windowing is ~100 lines, no framework needed); constant DOM cost at any archive size. (2) SEARCH-AS-YOU-TYPE: debounced (~150ms) incremental search against the daemon (FTS is f…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.

## Implementation plan

1. Four standard techniques, applied to the existing SPA (sequence after bby.6 extracts the JS to real files — refactoring inline-string JS is not viable): (1) LIST VIRTUALIZATION: render only the viewport window of the session list (hand-rolled windowing is ~100 lines, no framework needed)
2. constant DOM cost at any archive size.
3. (2) SEARCH-AS-YOU-TYPE: debounced (~150ms) incremental search against the daemon (FTS is fast when ready), cancel in-flight requests on new keystrokes (AbortController), paint first page immediately with a 'more loading' tail — never a blank list while typing.
4. (3) OPTIMISTIC NAVIGATION: clicking a session paints instantly from the list-row data (title/origin/date skeleton) while messages stream in
5. hover-prefetch the detail for the row under the cursor (the cache bead makes this nearly free).
6. (4) CACHE-AND-REVALIDATE: client keeps a small LRU of visited sessions keyed by the archive cursor from the SSE channel — back-navigation is instant, invalidation is push-driven, stale is impossible by construction.
7. Acceptance: interactions measured against the SLO tier budgets (first paint <300ms, list scroll 60fps at 20k sessions on the operator machine)

## Tests to add

- Acceptance proof: Session list scrolls at 60fps with 20k sessions (virtualized DOM stays constant-size).
- Acceptance proof: Search-as-you-type paints first results <300ms with a warm daemon and stale requests are cancelled.
- Acceptance proof: Back-navigation to a visited session renders instantly from client cache and revalidates via cursor.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
