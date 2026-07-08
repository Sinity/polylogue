---
created: "2026-07-08T18:10:00+02:00"
purpose: "polylogue-srjq classification audit: every COALESCE(...sort_key_ms...) call site, fixed/safe/synthetic verdict"
status: "complete"
project: "polylogue"
---

# sort_key_ms COALESCE audit — 2026-07-08 (polylogue-srjq)

Split from polylogue-cpf.6 (the clock-seam half landed separately in
`122f28796`/#2566). This is the audit half: classify every
`COALESCE(...sort_key_ms...)`-shaped expression as **BUG** (a timeless
row's fallback value is observable in an ORDER BY or a WHERE-range
boundary, so it silently sorts to the epoch or vanishes from/appears in a
time-windowed query), **SAFE** (the fallback is never observable in
ordering/windowing — an aggregate feeding a display column, or a
self-cancelling drift check), or **SYNTHETIC-OK** (an intentional
placeholder already paired with an explicit degraded-provenance signal
elsewhere). No SYNTHETIC-OK sites were found — the repo has no existing
`time_confidence`/`is_synthetic`-style convention to pair with a fallback
today, which is itself a finding (see Follow-ups).

## Method

`grep -rn "sort_key_ms" polylogue --include="*.py" | grep -i coalesce` (67
hits) plus manual verification of every named file, including multi-line
SQL the single-line grep misses (found one extra site in
`archive.py:7139-7145`). 68 total sites across 9 files. Traced call sites
through `api/archive.py` → `daemon/http.py` → CLI/MCP tool registrations
to rank BUG sites by public-facing reach.

Two fallback shapes exist:

- **Shape A** — `COALESCE(..., sort_key_ms, ..., 0)`: terminal fallback is
  literal epoch (1970). The classic defect.
- **Shape B** — `COALESCE(..., source_sort_key_ms, materialized_at_ms)`
  (only in `session_insight_timeline_reads.py`): terminal fallback is a
  real, non-null timestamp (materialization time), not epoch — avoids the
  1970 defect but introduces an inverse false-freshness bias, a distinct
  and milder issue.

## Summary

| Classification | Count | Files |
|---|---|---|
| **BUG** | 26 | `query_builders.py`, `runtime.py`, `attachment_records.py`, `archive.py` |
| **SAFE** | 33 | `repair.py`, `status.py`, `rebuild.py`, `archive.py` (internal bookkeeping), `index.py` (schema def) |
| **SAFE — guarded staleness check** | 3 | `convergence_stages.py` (explicit `IS NULL`-guarded drift against a *different* column) |
| **SAFE — Shape B caveat** | 9 | `session_insight_timeline_reads.py` (avoids epoch, has inverse false-freshness bias — tracked as a follow-up) |
| **SYNTHETIC-OK** | 0 | no existing convention found |

## BUG sites (fix required)

### `polylogue/storage/search/query_builders.py`

| Line | Context | Consumer |
|---|---|---|
| 52, 113 | `COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms, 0)` in `ORDER BY ... sort_key DESC` (session_rank window + `LIMIT`) | public `search` MCP tool / CLI search |
| 67, 129 | same expression `>= ?` for the `--since` filter | public `search` since-filter |

### `polylogue/storage/search/runtime.py`

| Line | Context | Consumer |
|---|---|---|
| 118 | `COALESCE(m.occurred_at_ms, s.sort_key_ms, s.updated_at_ms, s.created_at_ms, 0) >= ?` | public search `since` filter |
| 133 | same chain, no trailing `,0` — NULL sorts last in `ORDER BY ... DESC` (same practical effect) | public search ranking |

### `polylogue/storage/sqlite/queries/attachment_records.py`

| Line | Context | Consumer |
|---|---|---|
| 195 | `COALESCE(m.occurred_at_ms, c.sort_key_ms, 0)` in `ORDER BY ... sort_key DESC` | attachment-identity search |
| 213 | same chain `>= ?` for `since` | attachment-identity search |

### `polylogue/storage/sqlite/archive_tiers/archive.py` (largest surface)

| Line | Function | Context |
|---|---|---|
| 1102 | `get_session_tree` | `ORDER BY COALESCE(sort_key_ms, created_at_ms, updated_at_ms, 0)` — no LIMIT, but a timeless sibling displays as "earliest" in a directly user-facing tree (`get_session_tree` MCP tool) |
| 1216, 1219, 1231 | `list_session_work_event_insights` | since_ms/until_ms range + DESC ORDER BY + LIMIT/OFFSET |
| 1279, 1282, 1294 | `list_session_phase_insights` | same shape, sibling insight |
| 1780 | `usage_timeline` base filter | `COALESCE(e.occurred_at_ms, s.sort_key_ms, 0) > 0` unconditionally drops timeless-session usage/cost from every bucket — understates spend with no signal |
| 4914, 4916 | `query_messages` | `sort=time` ORDER BY + LIMIT/OFFSET, public `query messages` CLI/MCP unit |
| 5163, 5166 | `query_actions` | same shape, `query actions` unit |
| 5247 | `query_session_actions` | same shape, session-scoped |
| 5307 | `query_session_action_occurrences` | same shape |
| 5360, 5361 | `query_files` (`MIN`/`MAX` aggregation) | feeds displayed `first_seen_ms`/`last_seen_ms` AND the `file` unit's time filter/ordering |
| 5437, 5438 | `query_session_files` | same shape, session-scoped, feeds `ORDER BY COALESCE(f.first_seen_ms, 0)` at 5449 |
| 5542, 5544 | `query_blocks` | same shape, `query blocks` unit |
| 7139-7145, 7148 | `_query_unit_time_expression` (message / action-block branches) | **highest priority**: generates the WHERE-boundary for the public `query` CLI's `time>=`/`time<=` field predicate, consumed by `_time_predicate_clause` — every user-typed time-range filter on `messages`/`actions`/`blocks` goes through this |

## SAFE sites (no fix needed)

- `repair.py:548,569,582`, `status.py:202,209,212,227,233,236,254,265,268,284,376,380` — self-cancelling equality-drift checks (`ABS(COALESCE(cached, 0) - COALESCE(live, 0)) > eps`) or hot-window gates that bias toward *more* repair coverage, not toward hiding staleness. Internal convergence/repair bookkeeping only, never a windowed user-facing listing.
- `rebuild.py:107` — full-archive rebuild sweep with no `LIMIT`; every session is eventually visited regardless of sort order.
- `archive.py:1802,1813` (post-filter display columns, guaranteed non-null by the 1780 filter already having run), `archive.py:2013,2016` (internal scan-cutoff optimization, dead fallback branch since `occurred_at_ms IS NOT NULL` is already required), `archive.py:4094` (self-cancelling drift check, sentinel `-1` on both sides) — no windowing/ordering effect.
- `index.py:87` — the `sort_key_ms` generated-column definition itself (produces the column, doesn't consume it).

## SAFE — guarded staleness check

- `convergence_stages.py:830,1439,1469` — the COALESCE only executes inside a branch explicitly gated by `sort_key_ms IS NOT NULL`; the NULL case is handled by an independent fallback signal (comparing `source_updated_at` text). Better-guarded than the equivalent unguarded checks in `status.py`/`repair.py`, and not the audited windowing pattern at all (diffs a cached `source_sort_key` against `sort_key_ms`, not user-facing time filtering).

## SAFE — Shape B caveat (follow-up, not a fix in this audit)

- `session_insight_timeline_reads.py:45,76,209,215,230,235,240,245,278,283,288,293,300` — terminal fallback is `im.materialized_at_ms` (real, non-null), not epoch, so no 1970 defect. Caveat: a genuinely old, timeless work-event/phase can appear to sort/filter as "recently occurring" since materialization time proxies for recency — the inverse bias, milder than the epoch case and not silently *dropping* data, but still worth an explicit `time_confidence` signal eventually (see Follow-ups).

## Verdict

Audit phase of polylogue-srjq is **done**: every one of the 68 sites has
an evidence-backed classification. The fix phase (26 BUG sites) is
substantial cross-cutting work — public search ranking, CLI query-unit
pagination/ordering, the central `time>=`/`time<=` predicate generator,
work-event/phase insight windowing, and usage/cost aggregation all need
coordinated changes plus a design decision on how "timeless" rows should
present (silently-safe-include vs. an explicit `time_confidence=synthetic`
surface field, which does not exist anywhere in the codebase today).
Deferred to scoped follow-up beads rather than rushed in this sitting —
see Follow-ups below and the bead notes on polylogue-srjq.

## Follow-ups (beads filed from this audit)

- **polylogue-srjq** stays open, design field updated with this table,
  scoped to the fix phase only (audit phase closed via this artifact).
- New beads for the fix phase, one per coherent subsystem so each stays
  PR-shaped: search ranking/since-filter (query_builders.py + runtime.py +
  attachment_records.py), CLI query-unit ordering + the central time
  predicate generator (archive.py query_messages/actions/blocks/files +
  `_query_unit_time_expression`), work-event/phase insight windowing
  (archive.py list_session_work_event_insights/list_session_phase_insights),
  usage_timeline's silent-drop base filter (archive.py:1780), and a
  standalone design bead for the `time_confidence` signal itself (does
  any consumer need to *see* degraded provenance, or is "don't silently
  exclude/mis-sort" sufficient) plus the Shape B false-freshness caveat in
  `session_insight_timeline_reads.py`.
