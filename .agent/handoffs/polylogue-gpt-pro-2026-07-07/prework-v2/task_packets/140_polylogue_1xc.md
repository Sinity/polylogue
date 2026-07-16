# 140. polylogue-1xc — Scale-hardening: bugs that only bite on real-scale archives

Priority/type/status: **P1 / epic / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **epic-needs-child-closure**.

## What the bead says

Confirmed-severe set of code correct on small/clean fixtures but wrong at real scale (e.g. full insight rebuild = one transaction -> 6GB WAL + minutes-long write lock). Work the checklist on the issue; tier-1 items were observed live. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict.

## Existing design note

Tier-1 confirmed-live items (gh#2465 checklist is authoritative; work it there): full insight rebuild runs as ONE transaction -> 6GB WAL + minutes-long write lock on the live archive — chunk the rebuild into bounded per-batch transactions with progress rows (storage/insights rebuild path); the run_ref global-PK collision class was fixed (#2464) — audit for siblings (any global PK derived from non-unique local coordinates). General class to hunt: code correct on small/clean/distinct-id fixtures but wrong on real-scale shape (16K+ sessions, 5M+ messages, hash collisions, duplicate native ids, giant single artifacts like the 384MB Codex raw row). Add scale-tier tests where cheap (synthetic corpus generator exists).

## Acceptance criteria

Epic terminal state: every child closed and a scale-regression lane exists (seeded large-archive tier or live-copy probe) that would have caught each shipped bug class, wired into the optional lanes.

## Static mechanism / likely defect

Issue description localizes the mechanism: Confirmed-severe set of code correct on small/clean fixtures but wrong at real scale (e.g. full insight rebuild = one transaction -> 6GB WAL + minutes-long write lock). Work the checklist on the issue; tier-1 items were observed live. GH issue thread (body + comments) is input, not authority; this bead's scope statement wins where they conflict. Design direction: Tier-1 confirmed-live items (gh#2465 checklist is authoritative; work it there): full insight rebuild runs as ONE transaction -> 6GB WAL + minutes-long write lock on the live archive — chunk the rebuild into bounded per-batch transactions with progress rows (storage/insights rebuild path); the run_ref global-PK collision class was fixed (#2464) — audit for siblings (any global PK derived from non-unique local coordi…

## Source anchors to inspect first

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

1. Tier-1 confirmed-live items (gh#2465 checklist is authoritative
2. work it there): full insight rebuild runs as ONE transaction -> 6GB WAL + minutes-long write lock on the live archive — chunk the rebuild into bounded per-batch transactions with progress rows (storage/insights rebuild path)
3. the run_ref global-PK collision class was fixed (#2464) — audit for siblings (any global PK derived from non-unique local coordinates).
4. General class to hunt: code correct on small/clean/distinct-id fixtures but wrong on real-scale shape (16K+ sessions, 5M+ messages, hash collisions, duplicate native ids, giant single artifacts like the 384MB Codex raw row).
5. Add scale-tier tests where cheap (synthetic corpus generator exists).

## Tests to add

- Acceptance proof: Epic terminal state: every child closed and a scale-regression lane exists (seeded large-archive tier or live-copy probe) that would have caught each shipped bug class, wired into the optional lanes.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
