# A4 — Daemon Lifecycle for a REQUIRED Resident Daemon

Design deliverable. Scope: autostart-on-first-client, socket discovery/naming,
single-instance enforcement, health/liveness, crash detection + auto-restart,
clean shutdown + idle timeout, and cold-start client behavior. READ-ONLY on the
codebase; this is a proposal grounded in the current start path.

Sibling swarm docs own adjacent surfaces: **A2** owns the wire protocol over the
socket; this doc only requires that *a socket exists, is discoverable, and
answers a `hello`/`ping`*. Prereq `t46` ("contracts own surfaces", thin client)
is assumed done.

---

## 1. What exists today (evidence)

**Start path.** `polylogued run` is a Click command
(`polylogue/daemon/cli.py:1400` `run_command`) that calls
`asyncio.run(run_daemon_services(...))` (`daemon/cli.py:1582`). This is the
entrypoint for the `polylogued.service` systemd **user** unit
(`daemon/cli.py:1512` docstring; sinnix `modules/services/polylogue.nix:201`
`programs.polylogued`, `:204` `autoStart`, `WantedBy=default.target`).

**Autostart is 100% delegated to systemd.** There is **no** code anywhere that
spawns the daemon when a client finds it missing (`rg` for `Popen`/`spawn.*daemon`
finds only pipeline process pools, never a daemon launcher). The CLI simply
degrades to daemonless direct-archive reads:
`click_app.py:212 _show_stats` → `show_fast_status(env)` → `except: pass` →
`print_summary` (`click_app.py:220`). Daemonless reads work; importing the
Python substrate (~240 ms floor) is the unavoidable cost.

**Single-instance enforcement (solid).** Pidfile at
`archive_root/daemon.pid` (`daemon/cli.py:891`). Before binding,
`_verify_pidfile` (`:79`) reads the PID, does `os.kill(pid, 0)`, and confirms
`/proc/<pid>/cmdline` contains `polylogued`; a stale pidfile is unlinked
(`:896-900`). Then `_acquire_pidfile` (`:703`) takes an **advisory
`fcntl.flock(LOCK_EX|LOCK_NB)`** on an fd held for process lifetime — this is
the real mutex; the PID text is advisory. On exit the fd closes, the pidfile is
unlinked (`:1182-1185`), and `atexit` (`:1566`, `_cleanup_pidfile`) is the
backstop.

**Discovery / naming.** Pidfile is per-archive-root. The read API is **TCP HTTP
on `127.0.0.1:8766`** (`daemon/cli.py:1483` default `--api-port`;
`DaemonAPIHTTPServer(ThreadingHTTPServer)` at `daemon/http.py:3871`,
`allow_reuse_address=True`). **There is no Unix socket today** — both the API
and browser-capture servers are `ThreadingHTTPServer` (TCP). Client discovery:
`POLYLOGUE_DAEMON_URL` env, else a `/proc/*/cmdline` scan for
`polylogued run --api-port N` (`status.py:136 _discover_polylogued_api_ports`).

**Health/liveness.** Two independent probes: pidfile liveness
(`status.py:1585 _check_daemon_liveness`) and HTTP `GET /healthz/live`
(`status.py:100 _daemon_live`, 200–499 = alive). `/healthz/*` and `/metrics`
are unauthenticated because they bind loopback (`daemon/http.py:1274` comment).
Rich readiness via `daemon_status_payload` (`status.py:2042`).

**Crash detection + restart.** Delegated to systemd. The unit override only sets
`IOAccounting`/`MemoryHigh=4G`/`MemoryMax=6G` (`polylogue.nix:238-241`); I did
**not** find an explicit `Restart=` — it inherits the systemd default (`no`)
unless the upstream `programs.polylogued` module sets it. **This is a gap.**

**Clean shutdown (thorough).** SIGTERM/Ctrl-C → asyncio cancels →
`run_daemon_services` `finally` (`daemon/cli.py:1131`): emit `shutdown_started`
lifecycle event, stop watcher, `await converger.stop()`, shut down HTTP servers
off-loop via a dedicated daemon thread (`_shutdown_server_if_serving:1241`,
carefully avoiding executor deadlock — #1877), cancel + drain component and
maintenance tasks with **5 s timeouts** (`_drain_tasks:1200`), mark interrupted
ingest attempts (`:1174`), release flock, unlink pidfile, emit
`shutdown_complete`.

**Idle timeout.** None. The daemon runs forever.

**Schema-degraded mode (preserve this).** A schema-CRITICAL preflight
(`daemon/cli.py:874 _check_schema_version_fast`) blocks the watcher but **keeps
HTTP/health surfaces serving** (`:1100-1114`) so the bad state is observable and
`set_degraded(...)` is recorded. This is a genuine third state, not just
up/down.

### Gap summary
| Concern | Today | Needed for target |
|---|---|---|
| Transport | TCP :8766 | UDS (thin/Go-able client, single-digit-ms) |
| Autostart on first client | none (systemd only) | race-free autostart |
| Client-side restart | none | crash detect + bounded respawn |
| Idle timeout | none | needed for client-spawned daemons |
| Discovery | `/proc` scan | deterministic per-archive socket |
| Cold-start UX | silent fallback | spawn+wait with warming state |

---

## 2. Recommendation in one line

**Systemd socket-activation is the primary autostart mechanism; a flock-guarded
client-spawn is the break-glass fallback. Both converge on one deterministic
per-archive-root UDS. The client never queues — it spawns/triggers, then waits
on socket-ready with a deadline, distinguishing "accepting connections" from
"reads are warm."**

Socket activation makes autostart-on-first-client *inherently race-free*: systemd
owns the listening socket from login, so the first `connect()` starts the daemon
with zero client-side spawn logic. Everything below is designed so the
break-glass path degrades to the same steady state.

---

## 3. Socket naming & discovery (per-archive-root, multi-archive)

One daemon per **resolved** archive root (`paths/_roots.py:107 archive_root()`,
overridable via `POLYLOGUE_ARCHIVE_ROOT`). Derive the socket deterministically:

```
runtime = $XDG_RUNTIME_DIR/polylogue        # tmpfs, 0700, auto-GC on logout
key     = sha256(realpath(archive_root))[:16]
socket  = runtime/<key>.sock
lock    = runtime/<key>.lock                 # flock target (see §4)
meta    = runtime/<key>.json                 # {pid, proto_version, archive_root, started_at, api_tcp_port?}
```

- **`$XDG_RUNTIME_DIR`, not the archive root.** It is tmpfs, user-private
  (0700), and cleared on logout — a dead socket never survives a reboot, and it
  is never on the (possibly networked/slow) archive volume. Fall back to
  `/run/user/$UID/polylogue`, then a `state_home()/run` dir if `XDG_RUNTIME_DIR`
  is unset (headless/cron).
- **Hash-keyed ⇒ multi-archive is free.** Two archives ⇒ two keys ⇒ two sockets
  ⇒ two daemons, no coordination. The default archive and a dev-loop isolated
  archive coexist automatically.
- **`meta.json` is the discovery + verification record** — it replaces the
  `/proc/*/cmdline` scan (`status.py:136`, delete it). A client reads `meta`,
  confirms `archive_root` matches and `proto_version` is compatible, then
  connects. The daemon writes it atomically (tmp+rename) after bind.
- **Env override** `POLYLOGUE_DAEMON_SOCKET` for tests/dev-loop pins an explicit
  path (mirrors today's `POLYLOGUE_DAEMON_URL`).
- **Keep a loopback TCP port too**, recorded in `meta.api_tcp_port`, for the web
  reader / MCP / browser-capture and any non-UDS consumer. UDS is the CLI/TUI
  fast path; TCP stays for HTTP surfaces. The daemon binds both.

---

## 4. Single-instance enforcement (race-free bind)

Keep the flock (it is correct) but combine it with a connect-probe so the classic
"stale socket file blocks bind" race is closed. On daemon start:

1. `flock(lock, LOCK_EX|LOCK_NB)` on `runtime/<key>.lock`. **EWOULDBLOCK ⇒ a
   live daemon owns this archive; exit 0 quietly** (this is the socket-activation
   double-start and the break-glass loser both landing here — not an error).
2. Holding the lock, `connect()` the existing `<key>.sock`:
   - connects ⇒ (shouldn't happen — we hold the lock) treat as live, exit.
   - `ECONNREFUSED`/`ENOENT` ⇒ stale socket file, `unlink` it.
3. `bind()` + `listen()` on `<key>.sock`, `chmod 0600`.
4. Atomically write `meta.json`. Release nothing — hold the flock for lifetime.

This makes bind exclusive *and* self-healing after a SIGKILL that left a socket
file behind. Under socket activation systemd owns the fd and step 1's flock is
still the single-writer guard for the maintenance/watcher singleton.

---

## 5. The lifecycle state machine

### Daemon states

```
        (bind fd from systemd, or self-bind after flock)
ABSENT ──spawn/activation──▶ ACCEPTING ──open archive──▶ WARMING ──ready──▶ READY
                                 │                          │                 │
                                 │ schema CRITICAL          │ idle N min      │ SIGTERM
                                 ▼                          │ (spawned only)  ▼
                              DEGRADED ◀───────schema───────┘             DRAINING
                                 │                                            │
                                 └──────────────── SIGTERM ───────────────▶ STOPPED
   any state, on fatal fault ─────────────────────────────────────────────▶ CRASHED
```

- **ACCEPTING** — socket bound, event loop up, answers `hello`/`ping`
  immediately. **Critical design point: the daemon must accept connections and
  answer liveness the instant it binds, before the 38 GB archive is opened.**
  Today FTS readiness + Drive catch-up run *before* the watcher starts
  (`daemon/cli.py:1007-1060`); that is fine, but the *listener* must be up first
  so a cold-start client isn't refused. Move socket bind ahead of
  `_ensure_fts_startup_readiness`.
- **WARMING** — archive opening: schema preflight, FTS startup readiness
  (`_ensure_fts_startup_readiness`), lineage readiness, automerge config, Drive
  catch-up (`daemon/cli.py:1012-1035`). Read ops in this window either
  block-with-deadline or return a typed `WARMING` status carrying a progress
  hint; **never silently hang**. `ping` returns `state=warming, progress=…`.
- **READY** — watcher + all periodic loops running (`daemon/cli.py:1037-1099`);
  reads are single-digit-ms warm. This is the composer's operating state.
- **DEGRADED** — schema-CRITICAL (`daemon/cli.py:874`, `set_degraded`). Listener
  + status/health serve; watcher/convergence refuse; reads served if the index
  is intact. Maps exactly to today's `watcher_blocked` branch — preserve it.
- **DRAINING** — the existing `finally` teardown (`daemon/cli.py:1131`), 5 s
  drain budget, mark-interrupted ingest, release flock, unlink socket + meta.
- **CRASHED** — process died without DRAINING; socket file may be stale
  (§4 self-heals it) and flock auto-released by the kernel.

### Client states

```
DISCOVER ──meta ok──▶ CONNECT ──refused/ENOENT──▶ AUTOSTART ──▶ WAIT_READY ──▶ ACTIVE
   │                     │                                          │             │
   │ no meta             │ connected                                │ deadline    │ EPIPE/ECONNRESET
   └──▶ AUTOSTART        └──────────────▶ ACTIVE                    ▼             ▼
                                                              FALLBACK        RECONNECT
                                                            (--no-daemon)    (bounded, §6)
```

---

## 6. Autostart-on-first-client (who launches, race-free)

**Primary — systemd socket activation (recommended).**
Ship `polylogued.socket` + `polylogued.service` (templated
`polylogued@<key>.socket` if the operator runs multiple archives). systemd binds
`$XDG_RUNTIME_DIR/polylogue/<key>.sock` at login (`WantedBy=sockets.target`) and
starts the service on the first `connect()`, passing the listening fd via
`LISTEN_FDS`/`LISTEN_PID`. The daemon detects `LISTEN_FDS` and does
`socket.fromfd(...)` instead of self-binding (§4 step 3 becomes fd inheritance;
flock step 1 still runs as the maintenance-singleton guard).

Why this is the right default: **the race is eliminated in the kernel.** N
clients connecting simultaneously to a cold archive all succeed — systemd starts
exactly one service, queues the connections on the socket backlog, and the
daemon accepts them once ACCEPTING. No client-side spawn lock, no thundering
herd. It also gives `Restart=on-failure` crash recovery, resource limits
(already set), and journald logging for free.

**Break-glass — client-spawn (no systemd; cloud sandbox, `--no-daemon`
inverse).** When `CONNECT` gets `ENOENT`/`ECONNREFUSED` and no systemd socket
exists:

1. `flock(runtime/<key>.lock, LOCK_EX|LOCK_NB)`.
   - **win** ⇒ this client owns the spawn. `Popen(["polylogued","run",...],
     start_new_session=True)` (detached; stdout/stderr to a rotating log under
     `state_home`/logs). Then drop into WAIT_READY. The spawned daemon takes the
     same flock for its lifetime, so release the *spawn* lock only after the
     daemon's own flock is observed (or just let the child re-take it — use a
     separate `.spawnlock` to avoid ambiguity).
   - **lose (EWOULDBLOCK)** ⇒ another client is already spawning; **do not
     spawn**. Go straight to WAIT_READY on the socket.
2. This flock is the single-winner election — it is the exact mechanism the
   daemon itself uses (`_acquire_pidfile`), reused client-side.

**Cross-cutting:** `POLYLOGUE_NO_AUTOSTART=1` (and the existing `--no-daemon`
break-glass) skip AUTOSTART entirely and force FALLBACK to direct-archive reads.
CI/cloud lanes set it.

---

## 7. Cold-start client behavior: spawn + wait, never queue

**Decision: spawn-and-wait, not client-side queue.** The client triggers the
daemon (connect→activation, or spawn) and then blocks on socket-ready with a
deadline, because queuing requests client-side duplicates the socket backlog
systemd already provides and complicates the thin (future Go) client.

WAIT_READY loop:
- Poll `connect()` + `hello` with exponential backoff (5 ms → 50 ms cap),
  overall deadline `POLYLOGUE_DAEMON_CONNECT_TIMEOUT` (default **10 s**;
  archive-open on a cold 38 GB tree can take seconds).
- If `hello` returns **ACCEPTING/WARMING**, the process is up but reads aren't
  warm. For a **read** op, wait for READY (or accept a `WARMING`-tagged
  best-effort result if the op supports it). For the **composer**, enter a
  distinct UI state: connection established, live-preview disabled, show
  "warming (FTS N%)"; flip to live preview on READY. This is the headline-UX
  contract — the composer must never appear frozen during a cold start.
- If the deadline elapses in ACCEPTING with no READY, surface a real error
  (daemon stuck warming) — do **not** silently fall back, or the operator loses
  the "daemon required" invariant. `--no-daemon` remains the explicit escape.
- **First paint budget:** print a one-line "starting polylogued…" to stderr only
  if WAIT_READY exceeds ~300 ms, so a warm daemon stays invisible and a cold
  start is honest.

---

## 8. Crash detection + auto-restart by the client

**Under systemd (primary): don't restart from the client.** Set
`Restart=on-failure`, `RestartSec=1s`, `StartLimitIntervalSec=30`,
`StartLimitBurst=5` on `polylogued.service` (close the §1 gap). Add
`WatchdogSec` + `sd_notify(WATCHDOG=1)` from the periodic loop so a wedged
event loop (not just a crashed process) is restarted. The client, on `EPIPE`/
`ECONNRESET`/`ECONNREFUSED` mid-session, re-enters CONNECT→WAIT_READY: because
socket activation re-arms, the next connect restarts the service. The client's
only job is *reconnect with backoff*, not *respawn*.

**Break-glass (client-spawned): bounded client respawn.** On a broken
connection to a socket that was live this session:
- RECONNECT: retry connect for a short window (systemd-less daemon may just be
  DRAINING).
- If the socket is gone and flock is free ⇒ the daemon died ⇒ re-run AUTOSTART
  (§6 break-glass), i.e. respawn.
- **Crash-loop breaker (mandatory):** track respawns in a `meta`-adjacent
  `restarts.json` (timestamp ring). If ≥3 spawns in 60 s, stop respawning and
  surface the daemon's last-log tail as an error. Infinite respawn against a
  deterministic startup crash (e.g. corrupt DB) is the failure mode to prevent —
  a schema-CRITICAL daemon reaches DEGRADED and *stays up*, so respawn is only
  for genuine process death, and the breaker catches deterministic death.

---

## 9. Clean shutdown + idle timeout

**Shutdown** is already good (`daemon/cli.py:1131` DRAINING). Additions:
- Add `sd_notify(STOPPING=1)` at DRAINING entry and `READY=1` at READY, so
  systemd's own state tracks the machine.
- On DRAINING, `unlink` `<key>.sock` and `<key>.json` (mirrors pidfile unlink at
  `:1185`) so a client never connects to a socket whose daemon is mid-teardown;
  the flock release is the authoritative "gone" signal.
- Keep `TimeoutStopSec` > the 5 s internal drain budget (recommend 15 s) so
  systemd doesn't SIGKILL mid-drain and leak the interrupted-ingest marking
  (`:1174`).

**Idle timeout** — asymmetric by launch mode:
- **systemd resident (autoStart=true): no idle timeout.** The daemon is REQUIRED
  and resident; shutting it down defeats the warm-cache purpose and fights
  socket activation. Leave it running.
- **Client-spawned (break-glass): idle timeout ON, default 30 min.** Track
  `last_activity` (any wire request) + open-connection count. When both "no
  connections" and "idle > timeout" hold, DRAIN and exit so an orphaned
  ad-hoc daemon (spawned by a one-off CLI call in a sandbox) doesn't linger
  forever. Under socket activation this is safe even resident: a subsequent
  connect re-activates it — but for the resident autostart case we disable it to
  keep the cache warm. Gate via `POLYLOGUE_DAEMON_IDLE_TIMEOUT_S` (0 = never;
  systemd unit sets 0).

---

## 10. Migration order (dependencies)

1. **Add UDS listener** to the daemon alongside the TCP server
   (`daemon/http.py`), bound *before* archive-open (§5 ACCEPTING). A2 defines the
   protocol framing; A4 needs only `hello`/`ping` + a `state` field.
2. **Deterministic socket/meta** (§3) + drop `/proc` discovery (`status.py:136`).
3. **flock+connect-probe bind** (§4), reusing `_acquire_pidfile` logic against
   the runtime lock.
4. **systemd `.socket` unit + `Restart`/`WatchdogSec`/`sd_notify`** in sinnix
   `modules/services/polylogue.nix`.
5. **Client CONNECT→AUTOSTART→WAIT_READY** in the thin client, replacing the
   silent `except: pass` fallback (`click_app.py:220`) with the explicit state
   machine + `--no-daemon`/`POLYLOGUE_NO_AUTOSTART` escapes.
6. **Break-glass spawn + crash-loop breaker + idle timeout** (§6/§8/§9) last —
   they only matter off-systemd (cloud, tests).

---

## 11. Open questions for the operator

1. **Socket activation vs. plain resident unit?** Socket activation is the
   cleanest race-free autostart, but it means the daemon is *sometimes* not
   running until first connect — is a truly always-warm daemon
   (`autoStart=true`, no activation, client just connects/reconnects) preferred
   on the primary workstation, with activation reserved for multi-archive/dev?
   My recommendation: resident `autoStart=true` on sinnix-prime for a
   permanently warm composer, socket-activation template for secondary archives.
2. **Cold-open blocking policy for reads issued in WARMING** — block up to the
   deadline, or return a partial `WARMING` result the composer renders as
   "indexing…"? Affects whether the very first query after boot feels slow or
   returns a placeholder.
3. **TCP retirement** — keep loopback TCP indefinitely for web/MCP/browser
   capture, or move those onto UDS too and drop TCP entirely? A4 assumes TCP
   stays for HTTP surfaces.
4. **`Restart=` may already be set** by the upstream `programs.polylogued` HM
   module (not in `modules/services/polylogue.nix`). Verify before adding a
   conflicting override.
