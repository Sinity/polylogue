# A3 — Daemonless audit & the "require the daemon" decision

Read-only design doc. Swarm2, 2026-07-05. Grounds in live source.

## TL;DR recommendation

**Do not drop daemonless. Drop the *duplicate* daemonless implementation.**

There are two different things people call "daemonless," and they have opposite
cost/benefit:

1. **Substrate-direct (library) daemonless** — `Polylogue` / `ArchiveStore`
   open `index.db`/`user.db`/`source.db` directly. This is load-bearing for the
   test suite, CI, cloud, MCP, lynchpin/oracle embedding, first-run bootstrap,
   and recovery-when-the-daemon-won't-start. **Non-negotiable; keep it.** It is
   not really "avoiding the daemon" — it is "Python code can open the archive,"
   which every one of ~thousands of tests does.

2. **CLI-client daemonless** — the `polylogue` CLI having a *second, full,
   independent* direct-DB read/write implementation, with the daemon as a mere
   accelerator it falls back off of. **This is the expensive one.** It is the
   source of the dual-path query gating, `_show_direct_status`, the fast/direct
   status branching, the `no_daemon` diagnostics tree, and much of the "CLI
   reaches into substrate 45×" problem the brief already flagged.

The move is not "require the daemon and reimplement nothing in the client." It
is: **make the daemon's request handler the single execution core, and let
`--no-daemon` / library / MCP / tests invoke that same core in-process.** One
handler, two transports (UDS vs in-process). The warm daemon is then *required
for speed and the composer*, and break-glass is a transport swap, not a parallel
codebase.

---

## Evidence: how daemonless works today

The daemon is **not in the read path**. Reads/writes open the archive SQLite
file set directly and rely on WAL (concurrent readers + single writer). The
daemon is a resident *writer + accelerator + web/metrics surface*, not a gateway.

Three surfaces already run fully daemonless against the same files:

- **CLI query** (`cli/query.py` → `cli/archive_query.py`): `execute_archive_query`
  opens `ArchiveStore` directly. No daemon check on the correctness path.
- **MCP** (`mcp/archive_support.py:29-34`): constructs `ArchiveStore` directly
  against the archive file set. The *entire* MCP tool surface is daemonless.
- **Python API** (`api/__init__.py`, `api/contracts/api_write_surface.py:77`):
  library-direct, explicitly labelled "no daemon."

The `_ARCHIVE_FACADE_ROUTES` map in `cli/commands/status.py:254-398` is easy to
misread as a daemon-vs-daemonless split. It is **not**. `archive_direct` vs
`archive_routed` both run **in-process against SQLite**; the split is only
"reaches raw SQLite" vs "goes through the ArchiveStore facade." Every entry is
daemonless. Same for `_ARCHIVE_CLI_ROUTES` (reset/blackboard, all `archive_direct`).

### What actually requires a running daemon (HTTP, no correctness fallback)

Only three CLI surfaces truly need the daemon, and only one is a hard dependency:

| Surface | File | Behavior without daemon |
| --- | --- | --- |
| `polylogue import` | `cli/commands/import_command.py:280-298` | **Hard fail.** POSTs `/api/ingest`; `URLError`/`OSError` → `fail(...)`. No fallback, by design (ingest must be daemon-serialized for observable convergence). Note a library-direct path *exists* (`Polylogue.parse_file`, used by `api_write_surface` and demo seed) but the CLI command does not use it. |
| `polylogue dashboard` | `cli/commands/dashboard.py:58` | Needs the daemon web UI (`/api/status` + served web reader). |
| Live ingest / watcher, browser-capture receiver, embedding catch-up, `/metrics`, periodic WAL checkpoint | `daemon/cli.py:797-828`, `daemon/metrics.py` | These *are* the daemon; no CLI equivalent exists or should. |

### What has a daemon fast-path but transparently falls back (daemon = pure optimization)

- **`ops status` / bare `polylogue`** (`status.py:1198` `show_fast_status`): tries
  `/api/status`, else `_daemon_live()` probe, else `_show_direct_status`
  (`status.py:1895`) — direct SQLite counts via `open_readonly_connection`.
- **Query/search session pages** (`archive_query.py:855-918`
  `_execute_daemon_session_page` + `_daemon_session_page_supported:921-950`):
  if the daemon is up **and** the query is a plain list/search page, reuse
  `/api/sessions`; otherwise `return False` → local `ArchiveStore`. Mutations,
  unit rows, stats, vector/hybrid, streaming, cursors always go local. This is
  ~100 lines of "is it safe to route to the daemon" gating that exists *only*
  because both paths must independently work.
- **tutorial / first-run** (`tutorial.py:148` `_daemon_http_alive`): best-effort
  probe feeding `diagnose_first_run`; degrades silently.

### The daemonless-assumption footprint (what a "require daemon" flip would touch)

- `cli/commands/status_diagnostics.py` — entire ~350-line first-run diagnostics
  module keyed off `daemon_alive`; the `no_daemon`/`stale_pidfile`/`locked_db`
  kinds exist to explain a healthy-archive-but-no-daemon state.
- `cli/commands/status.py` — `_show_direct_status`, `_direct_archive_counts`,
  `_fast_fts_doc_count`, `_show_daemon_status_unavailable`, the dual-timeout
  fast/full/direct branching.
- `cli/archive_query.py` — the daemon fast-path gate (above).
- Tests: ~20 files reference `daemonless`/`no_daemon`/`direct`; hot ones are
  `tests/unit/cli/test_status_diagnostics.py`, `test_status.py`,
  `test_query_exec_laws.py`, `test_import.py`, `tests/unit/daemon/test_web_reader.py`.
- Docs: `docs/cloud-agents.md` runs `polylogued run --no-api --no-watch
  --no-browser-capture` and, more importantly, runs pytest + direct archive ops
  against `/tmp/polylogue-archive` with **no daemon at all** (CI/cloud lane).

---

## Cost / benefit of making the daemon required (client contract)

### What simplifies

- **Delete the query dual-path** (`_execute_daemon_session_page`,
  `_daemon_session_page_supported`, the cascade of `return False`). The client
  always asks the core; "which path" disappears.
- **Collapse status** into one payload from the core. `_show_direct_status` and
  the daemon-vs-direct rendering divergence go away; `no_daemon` stops being a
  correctness state and becomes "warm cache/composer unavailable."
- **The 45 CLI→substrate reaches → 0.** A thin protocol client *cannot* reach
  into storage; this is exactly `t46` ("contracts own surfaces"), already a
  swarm prerequisite. Requiring the daemon and unifying on one core makes t46
  self-enforcing rather than a lint you can regress.
- **Write serialization.** Today a daemonless CLI `user.db` write (tags, marks,
  annotations, views, corrections, blackboard — all `archive_routed → user`)
  races the daemon's writes on the same file. Routing all mutations through the
  core-behind-the-daemon serializes them and removes a real concurrency surface.
- The composer / live-preview headline UX **only exists** with the warm daemon;
  a required daemon is the honest posture for that feature.

### What break-glass MUST still cover (the hard carve-outs)

These are why "daemon required" can be a **client/CLI** contract but never a
**substrate** contract:

1. **First-run bootstrap.** `polylogue init`, `ops maintenance archive-init`,
   and the *first* `polylogued run` all precede any daemon. You cannot require a
   daemon to create the archive the daemon needs.
2. **CI / cloud / tests.** The test harness and `docs/cloud-agents.md` construct
   `ArchiveStore`/`Polylogue` directly, thousands of times, with no daemon and
   no socket. A daemon-per-test-process is a non-starter. The Python substrate
   **must remain directly constructable**.
3. **Recovery when the daemon is the problem.** `locked_db`, `schema_mismatch`,
   `stale_pidfile`, `ops doctor --repair`, `ops reset --index`, `rebuild-index`
   must run *with the daemon stopped*. This is precisely the state where a
   "dial the socket first" client would be stuck.
4. **Library embedding.** lynchpin, the `oracle` digest, and MCP open the
   archive as a library. Same substrate the daemon uses.
5. **Scripting one-shots.** A piped `polylogue find ... | jq` should not fail
   because no daemon is warm.

### The cost of the flip

You must build the single-core request handler and make `--no-daemon` route
through it in-process. But that is **the same work `t46` + the A2 wire protocol
already require.** So dropping first-class *duplicate* daemonless is nearly free
once those land — and net-negative LOC.

---

## Recommendation (concrete)

1. **Unify on one execution core = the daemon's request handler.** The current
   sin is that "direct" and "daemon" are *different code*. Make them the same
   code with two transports.
2. **`--no-daemon` = in-process transport, not a reimplementation.** A single
   `polylogue --no-daemon <cmd>` runs the core in-process for that command (this
   is what MCP/library/tests already do implicitly). No parallel status
   renderer, no query dual-path.
3. **Warm daemon required for speed + the composer.** `complete()`/`preview()`
   single-digit-ms only over the resident socket. Break-glass is slower
   (cold-import ~240 ms floor per the brief), never less correct.
4. **Keep substrate-direct as an explicit, tested contract** (call it the
   "embedded core"): tests, CI, cloud, MCP, lynchpin, first-run, recovery all
   depend on it. Add a lint/test asserting the core is constructable without a
   socket, so this stays true.
5. **Delete on the way:** `_execute_daemon_session_page` gating,
   `_show_direct_status` divergence, and demote `no_daemon` from a correctness
   diagnostic to a "cold/composer-off" hint. Keep `locked_db`/`schema_mismatch`/
   `stale_pidfile` — those are recovery, not daemonless-mode, and matter more.
6. **`import`:** decide whether to keep it fallback-free (daemon-serialized for
   convergence integrity — current, defensible) or give it an in-process
   `Polylogue.parse_file` break-glass. Recommend keeping it daemon-required for
   *normal* use but wiring the library-direct path under `--no-daemon` so a
   daemon-down operator can still ingest one file for recovery.

## Open questions for the operator

- **`--no-daemon` semantics:** pure in-process core (simplest) vs auto-spawn a
  transient daemon (warms cache for the next call)? Recommend in-process; the
  resident daemon is a separate, deliberate `polylogued run`.
- **Writes without a daemon:** hard rule "all mutations over the socket," with
  `--no-daemon` mutations gated behind "daemon confirmed stopped"? This closes
  the user.db write-race but adds a preflight check to every write.
- **`import` fallback:** is daemon-serialized ingest a hard product invariant
  (observable convergence), or may recovery ingest go library-direct? This is a
  genuine product decision, not a cleanup.
