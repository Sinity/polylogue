# 21. polylogue-peo — Add daemon crash forensics, heartbeat sentinel, and restart evidence

Priority: **P2**  
Lane: **operational-resilience**  
Readiness: **ready-now / lifecycle module**

## Why this is urgent / critical-path

A daemon that silently dies leaves the web UI and operator believing stale states. Crash/death evidence is an operational correctness requirement.

## Static diagnosis / likely mechanism

Bead mechanism: under read-only serving flags the daemon terminated twice; logs stopped without traceback; exit code 144 was ambiguous. Current running checks may rely on pid/process rather than fresh heartbeat.

## Implementation plan

Implementation shape:
1. Add `polylogue/daemon/lifecycle.py` to own run id, heartbeat row, signal logging, and stack dumps.
2. At daemon start: enable `faulthandler`, create `ops.db daemon_lifecycle` table if absent, insert `started` row.
3. Periodic loop writes heartbeat timestamp each tick.
4. SIGTERM/SIGINT handlers log signal, active thread stack dump, heartbeat age, and lifecycle row before exiting/chain-calling.
5. `atexit` writes clean stop; missing clean stop plus stale heartbeat means vanish.
6. `/healthz` and bare `polylogue` status should report heartbeat age and not claim running from pid alone.
7. Check/patch Sinnix systemd unit for `Restart=on-failure` with backoff; if outside repo, record exact sinnix patch prompt.
8. Web banner can be a follow-up/linked bby.1 if not in same PR, but API status must expose daemon-unreachable/stale-heartbeat data.

## Test plan

Tests:
- direct lifecycle unit tests for start/heartbeat/clean stop rows.
- subprocess integration: start daemon fixture, send SIGTERM, assert signal row + stack log.
- stale heartbeat makes status/healthz report stale/down.
- clean shutdown distinct from vanish.

## Verification command / proof

`devtools test tests/unit/daemon/test_daemon_lifecycle*.py tests/unit/daemon/test_health*.py -k 'heartbeat or lifecycle or signal'` plus a manual/subprocess SIGTERM proof.

## Pitfalls

Do not do heavy SQLite work inside an unsafe low-level handler outside Python’s normal signal-dispatch context. Keep stack dumps bounded so a death does not create huge logs.

## Files/functions to inspect or touch

- `polylogue/daemon/cli.py`
- `polylogue/daemon/http.py health routes`
- `polylogue/daemon/status.py`
- `ops.db helpers`
- `sinnix module for polylogued if available`
