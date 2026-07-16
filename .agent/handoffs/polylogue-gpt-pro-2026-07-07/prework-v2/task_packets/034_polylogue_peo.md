# 034. polylogue-peo — Daemon death leaves no trace: crash forensics + heartbeat sentinel + restart policy

Priority/type/status: **P2 / bug / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

During read-only serving (run --no-watch --no-source-catchup --no-browser-capture) the daemon terminated twice within minutes; the log simply stops mid-work (last lines: routine embed/insights progress), no traceback, no 'Killed', nothing in ops.db. The web SPA degraded to 'Failed to fetch' everywhere with no daemon-down banner. One exit was code 144 (possibly external SIGTERM from the harness) — but that ambiguity IS the finding: when the daemon dies, nothing records why, and no surface says it is gone.

## Existing design note

(1) faulthandler.enable() + SIGTERM/SIGINT handlers that log signal + active thread stacks to the run log AND an ops.db daemon_lifecycle row (started/stopped/signal/last_heartbeat) before exit; an atexit sentinel distinguishes clean stop from vanish. (2) Startup writes a heartbeat row every periodic-loop tick; `polylogue` bare status and /healthz report heartbeat age, so 'daemon: running' claims are backed by a fresh heartbeat, not a pid file (bare status said 'Daemon: running' while it served a rebuild — verify what that check reads). (3) systemd unit: Restart=on-failure with backoff if not already set (check sinnix module). (4) Web SPA: liveness probe with visible 'daemon unreachable since T, retrying' banner instead of per-widget fetch failures (lands with bby.1). Postmortem for the two observed deaths belongs in the fix PR: reproduce serving + convergence under the same flags and capture what 144 actually was.

## Acceptance criteria

1. `faulthandler.enable()` is active at daemon start; SIGTERM/SIGINT handlers log the signal plus active thread stacks to the run log AND write an `ops.db` `daemon_lifecycle` row (started/stopped/signal/last_heartbeat) before exit; an atexit sentinel distinguishes clean stop from vanish. 2. A heartbeat row is written every periodic-loop tick; `polylogue` bare status and `/healthz` report heartbeat age, and a 'running' claim is backed by a fresh heartbeat rather than a pid file (verify what the bare-status check currently reads and correct it if pid-based). 3. The systemd unit has `Restart=on-failure` with backoff (check/patch the sinnix module). 4. The web SPA shows a visible 'daemon unreachable since T, retrying' banner instead of per-widget fetch failures (may land with bby.1). 5. Postmortem in the fix PR: the read-only serving death is reproduced under `run --no-watch --no-source-catchup --no-browser-capture` and what exit code 144 was is captured. Verify: send SIGTERM to a running daemon and confirm a `daemon_lifecycle` signal row plus a thread-stack log; `devtools test` selection on the lifecycle/heartbeat code; bare status reports a stale/absent heartbeat when the daemon is gone.

## Static mechanism / likely defect

Bead mechanism: under read-only serving flags the daemon terminated twice; logs stopped without traceback; exit code 144 was ambiguous. Current running checks may rely on pid/process rather than fresh heartbeat.

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

1. Implementation shape:
2. 1. Add `polylogue/daemon/lifecycle.py` to own run id, heartbeat row, signal logging, and stack dumps.
3. 2. At daemon start: enable `faulthandler`, create `ops.db daemon_lifecycle` table if absent, insert `started` row.
4. 3. Periodic loop writes heartbeat timestamp each tick.
5. 4. SIGTERM/SIGINT handlers log signal, active thread stack dump, heartbeat age, and lifecycle row before exiting/chain-calling.
6. 5. `atexit` writes clean stop; missing clean stop plus stale heartbeat means vanish.

## Tests to add

- direct lifecycle unit tests for start/heartbeat/clean stop rows.
- subprocess integration: start daemon fixture, send SIGTERM, assert signal row + stack log.
- stale heartbeat makes status/healthz report stale/down.
- clean shutdown distinct from vanish.

## Verification commands

- ``devtools test tests/unit/daemon/test_daemon_lifecycle*.py tests/unit/daemon/test_health*.py -k 'heartbeat or lifecycle or signal'` plus a manual/subprocess SIGTERM proof.`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
