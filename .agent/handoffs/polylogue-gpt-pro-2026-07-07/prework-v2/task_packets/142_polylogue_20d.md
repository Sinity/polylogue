# 142. polylogue-20d — Interactive performance: the front door answers in interactive time

Priority/type/status: **P2 / epic / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **epic-needs-child-closure**.

## What the bead says

Cold CLI invocations pay ~2s of Python imports; some helps took 5-9s; find-then-select cold spikes; claim-vs-evidence regen 43s; ingest catch-up crawled at 0.2 files/s. WAL checkpoint + ANALYZE done (2026-07-03: index.db WAL=0, sqlite_stat1 present, v23). The CLI->daemon fast path is the structural attack; import deferral is the fallback for daemonless cold starts.

## Existing design note

Front-door interactive-latency spine. Mechanism ordering: 20d.14 states the named budgets first (evidence-tuned starting points); 20d.2 removes the ~2s import tax for the daemonless cold path; 20d.1 routes the hot path through the daemon over UDS; 20d.12 makes the daemon worth reaching (cursor-keyed result cache); 20d.13 replaces polling with SSE push; 20d.6/20d.15 own the live vs bulk ingest lanes; 20d.4/20d.5/20d.7/20d.8/20d.10/20d.11 are the direct-path and storage-profile fixes that keep the degraded mode fast. The epic's done-state ties to the 20d.14 budgets so 'interactive time' is a measured claim, not a vibe.

## Acceptance criteria

- The 20d.14 interactive SLO tier is defined in docs/plans/slo-catalog.yaml and runs green in `devtools bench slo` against the seeded corpus with a live daemon.
- On the operator machine, live measurement meets the daemon-served query, completion round-trip, cold-CLI, and ingest-to-searchable budgets named in 20d.14.
- No interactive read verb pays the old cold-import or FTS-gate penalties: the 20d.2 help-latency budget check and the 20d.4 structured-routing regression gate are in place and green.
- The evidence the epic cites (2s imports, 5-9s helps, 43s regen, 0.2 files/s ingest) is retired — each has an owning child whose acceptance names its budget.

## Static mechanism / likely defect

Issue description localizes the mechanism: Cold CLI invocations pay ~2s of Python imports; some helps took 5-9s; find-then-select cold spikes; claim-vs-evidence regen 43s; ingest catch-up crawled at 0.2 files/s. WAL checkpoint + ANALYZE done (2026-07-03: index.db WAL=0, sqlite_stat1 present, v23). The CLI->daemon fast path is the structural attack; import deferral is the fallback for daemonless cold starts. Design direction: Front-door interactive-latency spine. Mechanism ordering: 20d.14 states the named budgets first (evidence-tuned starting points); 20d.2 removes the ~2s import tax for the daemonless cold path; 20d.1 routes the hot path through the daemon over UDS; 20d.12 makes the daemon worth reaching (cursor-keyed result cache); 20d.13 replaces polling with SSE push; 20d.6/20d.15 own the live vs bulk ingest lanes; 20d.4/20d.5/20d.…

## Source anchors to inspect first

- `CONTRIBUTING.md:102` — Derived-tier schema changes require rebuild/blue-green planning.
- `AGENTS.md:168` — Agent guidance says schema mismatch should rebuild or blue-green-replace derived tiers.
- `polylogue/cli/commands/reset.py` — Current reset/rebuild commands are the operator path to replace derived tiers.
- `polylogue/daemon/convergence_stages.py` — Daemon convergence/readiness state should represent generation progress honestly.
- `polylogue/archive/query/spec.py:440` — SessionQuerySpec is the core read selection object.
- `polylogue/archive/query/spec.py:498` — from_params centralizes parameter lowering.
- `polylogue/cli/query_contracts.py:70` — QueryOutputSpec is current output/render contract seam.
- `polylogue/cli/archive_query.py:164` — CLI archive query execution remains a dense orchestration point.
- `polylogue/daemon/http.py:1734` — Daemon split-archive route already has from_params path for some cases.
- `polylogue/daemon/http.py:3296` — Another daemon path builds SessionQuerySpec from raw query params.
- `polylogue/mcp/query_contracts.py:54` — MCP query normalization has a distinct lowering path.
- `polylogue/storage/sqlite/archive_tiers/index.py:296` — messages_fts table DDL lives in archive_tiers/index.py.
- `polylogue/storage/fts/sql.py:11` — messages_fts DDL copy also exists in storage/fts/sql.py.
- `polylogue/storage/fts/fts_lifecycle.py:292` — ensure_fts_triggers_sync owns runtime repair/recreation.
- `polylogue/storage/fts/fts_lifecycle.py:512` — Thread FTS rebuild logic is duplicated in lifecycle repair path.
- `polylogue/daemon/convergence_stages.py:988` — daemon convergence has another repair/readiness path for archive messages_fts.

## Implementation plan

1. Front-door interactive-latency spine.
2. Mechanism ordering: 20d.14 states the named budgets first (evidence-tuned starting points)
3. 20d.2 removes the ~2s import tax for the daemonless cold path
4. 20d.1 routes the hot path through the daemon over UDS
5. 20d.12 makes the daemon worth reaching (cursor-keyed result cache)
6. 20d.13 replaces polling with SSE push
7. 20d.6/20d.15 own the live vs bulk ingest lanes

## Tests to add

- Acceptance proof: The 20d.14 interactive SLO tier is defined in docs/plans/slo-catalog.yaml and runs green in `devtools bench slo` against the seeded corpus with a live daemon.
- Acceptance proof: On the operator machine, live measurement meets the daemon-served query, completion round-trip, cold-CLI, and ingest-to-searchable budgets named in 20d.14.
- Acceptance proof: No interactive read verb pays the old cold-import or FTS-gate penalties: the 20d.2 help-latency budget check and the 20d.4 structured-routing regression gate are in place and green.
- Acceptance proof: The evidence the epic cites (2s imports, 5-9s helps, 43s regen, 0.2 files/s ingest) is retired — each has an owning child whose acceptance names its budget.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not implement the epic as a single broad PR; use it to close/split child work and verify the terminal invariant.
- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
