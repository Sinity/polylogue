# 097. polylogue-20d.1 — CLI->daemon fast path over UDS (persistent hot process)

Priority/type/status: **P2 / feature / open**. Lane: **08-scale-performance-live**. Release: **B/G-storage-live-performance**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Route CLI queries through the already-hot daemon when available: skips import cost, warm SQLite page cache, shared readiness state. Silent in-process fallback.

## Existing design note

Precedent: fast-status path (commands/status.py:950, click_app.py:214-221) already prefers the daemon — extend the pattern to the whole read surface. Transport: UDS at $XDG_RUNTIME_DIR/polylogue/daemon.sock (TCP stays for the browser); AF_UNIX HTTPServer subclass ~20 lines; instant-fail when down. Probe: socket exists -> connect (fails in microseconds) -> GET /api/health with 100ms budget; health payload carries {archive_root, index_schema_version, daemon_version, commit, started_at}; client compares against its own resolved config and silently falls back on mismatch — NON-NEGOTIABLE (live trap: POLYLOGUE_ARCHIVE_ROOT from .claude/settings.json pointed at /tmp while the real archive sat elsewhere). Thin client: new cli/daemon_client.py over stdlib http.client — no httpx, no payload models, no storage imports; send the RAW query string + flags (daemon owns compilation, which also delivers the #1860 structured-routing behavior the CLI lacks — the fast path fixes that bug for free); --format json renders via sys.stdout.write; table/plain imports only formatting helpers that render from payload dicts. Target: 3.6-17s -> 0.3-0.5s. Endpoints: REUSE /api/sessions, /api/query-units, /api/facets, /api/sessions/:id/read?view=, :id/messages; one new POST /api/cli/query accepting the root-request param dict (cli/root_request.py output) for the gaps so CLI flags never drift from the HTTP surface. Writes stay direct (user.db is a separate WAL, no contention); proxy reads only. Load isolation exists (the client-disconnect probe http.py:118-190 cancels server-side SQLite work on Ctrl-C); add a modest concurrent-read semaphore only if agent fan-out appears. Correctness: golden parity tests — byte-identical --format json between direct and proxied execution per read surface on the demo corpus. Escape hatches: --no-daemon, POLYLOGUE_DAEMON=off, --verbose prints 'served-by: daemon (uds, 41ms)'. Sequencing: subsumes the ~2s import tax, the cold-I/O tail, and the routing-parity bug; the direct path still needs the routing-parity + cached-stale-verdict fixes, but they shrink from 'the UX' to 'the degraded mode'.

## Acceptance criteria

- Fast-path read surface: `--verbose` prints `served-by: daemon (uds, <ms>)` and a warm daemon serves find/read/messages/facets within the 20d.14 interactive-tier budget (target 3.6-17s -> 0.3-0.5s wall). Verify: timed CLI run against a warm daemon; `devtools bench slo` interactive tier green.
- Golden parity: `--format json` output is byte-identical between direct and daemon-proxied execution for every read surface on the demo corpus. Verify: pytest golden-parity test.
- Config-mismatch safety (NON-NEGOTIABLE): with the daemon pointed at a different archive_root/index_schema_version/daemon_version than the client's resolved config, the client silently falls back to the in-process path. Verify: regression test seeding the POLYLOGUE_ARCHIVE_ROOT=/tmp mismatch trap.
- Escape hatches: `--no-daemon` and `POLYLOGUE_DAEMON=off` force the direct path; a daemon-down probe fails in microseconds (test).
- Writes never proxy: user.db operations always take the direct path (test/assertion).

## Static mechanism / likely defect

Issue description localizes the mechanism: Route CLI queries through the already-hot daemon when available: skips import cost, warm SQLite page cache, shared readiness state. Silent in-process fallback. Design direction: Precedent: fast-status path (commands/status.py:950, click_app.py:214-221) already prefers the daemon — extend the pattern to the whole read surface. Transport: UDS at $XDG_RUNTIME_DIR/polylogue/daemon.sock (TCP stays for the browser); AF_UNIX HTTPServer subclass ~20 lines; instant-fail when down. Probe: socket exists -> connect (fails in microseconds) -> GET /api/health with 100ms budget; health payload carries {ar…

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

1. Precedent: fast-status path (commands/status.py:950, click_app.py:214-221) already prefers the daemon — extend the pattern to the whole read surface.
2. Transport: UDS at $XDG_RUNTIME_DIR/polylogue/daemon.sock (TCP stays for the browser)
3. AF_UNIX HTTPServer subclass ~20 lines
4. instant-fail when down.
5. Probe: socket exists -> connect (fails in microseconds) -> GET /api/health with 100ms budget
6. health payload carries {archive_root, index_schema_version, daemon_version, commit, started_at}
7. client compares against its own resolved config and silently falls back on mismatch — NON-NEGOTIABLE (live trap: POLYLOGUE_ARCHIVE_ROOT from .claude/settings.json pointed at /tmp while the real archive sat elsewhere).

## Tests to add

- Acceptance proof: Fast-path read surface: `--verbose` prints `served-by: daemon (uds, <ms>)` and a warm daemon serves find/read/messages/facets within the 20d.14 interactive-tier budget (target 3.6-17s -> 0.3-0.5s wall).
- Acceptance proof: Verify: timed CLI run against a warm daemon
- Acceptance proof: `devtools bench slo` interactive tier green.
- Acceptance proof: Golden parity: `--format json` output is byte-identical between direct and daemon-proxied execution for every read surface on the demo corpus.
- Acceptance proof: Verify: pytest golden-parity test.
- Acceptance proof: Config-mismatch safety (NON-NEGOTIABLE): with the daemon pointed at a different archive_root/index_schema_version/daemon_version than the client's resolved config, the client silently falls back to the in-process path.
- Acceptance proof: Verify: regression test seeding the POLYLOGUE_ARCHIVE_ROOT=/tmp mismatch trap.
- Acceptance proof: Escape hatches: `--no-daemon` and `POLYLOGUE_DAEMON=off` force the direct path

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --quick`

## Pitfalls

- Do not optimize by hiding degraded/partial state; status must get more truthful, not prettier.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
