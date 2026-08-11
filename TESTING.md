# Testing

All commands below assume you are inside the project devshell. See
[CONTRIBUTING.md](CONTRIBUTING.md) for environment setup.

## Running Tests

```bash
# Normal repository verification
devtools verify

# First run after checkout, or when you intentionally want to refresh
# pytest-testmon's dependency database
devtools verify --seed-testmon --skip-slow

# Focused inner-loop runs — prefer `devtools test` over raw pytest. It runs the
# selection through the managed harness (repo env, single-process by default,
# live output, current-node progress artifacts, stall/runtime timeouts) and
# serializes overlapping runs from the same checkout so two suites do not race.
# Any pytest arguments go after the command name.
devtools test tests/unit/storage/test_hybrid_laws.py
devtools test -k "test_name"
devtools test tests/unit/pipeline -x
POLYLOGUE_PYTEST_WORKERS=8 devtools test tests/unit/storage   # override workers

# Raw pytest still works for ad-hoc needs the wrapper does not cover:
pytest -x --ignore=tests/integration
pytest tests/unit/storage/test_hybrid_laws.py

# Explicit full non-integration pytest diagnostic
devtools verify --all

# Full Nix/CI parity
nix flake check
```

### First-party browser credential journey

The browser security journey launches the production daemon against a fresh,
deterministic demo archive, then exercises list/read, user-state mutation, SSE
reconnect, credential lifecycle states, and secret-leak sentinels. Install the
locked Node dependencies once, install Chromium when the host has no compatible
system Chrome, and run the dedicated suite:

```bash
cd webui
npm ci
npm run install:e2e-browser  # CI uses install:e2e-browser:ci
npm run test:e2e
```

CI runs this journey in the `web-first-party-auth` job. Local NixOS development
uses the system Chrome path discovered by `webui/playwright.config.ts`, so the
browser install step is normally unnecessary after `npm ci`.

`devtools verify` uses pytest-testmon for per-test affected selection. The
seed command records `.cache/testmon/testmondata` plus
`.cache/testmon/seed.json`; those files are local generated state and are not
committed. If the seed is missing, the default command fails with setup
guidance instead of silently running the whole suite. Every seed writes
`.cache/testmon/seed-attempt.json` before work begins. An interrupted attempt
recovers its node ledger from the immutable run artifact when the outer process
could not finalize the receipt, then resumes only its unseen, failed, or changed
tests. Corrective code commits do not invalidate that attempt: pytest-testmon
owns dependency-change selection, while the Python and marker-policy identity
still prevents resuming against a different test corpus. `.cache/testmon/seed.json`
is published only after every originally selected node has a failure-free row
in the dependency database.

Plain focused `pytest` runs are single-process by default so small inner-loop
checks do not spawn a worker pool. `devtools verify` keeps pytest-testmon as
the affected-test selector and runs the selected default lane with an adaptive
worker pool (up to 12, override with `POLYLOGUE_PYTEST_WORKERS`) so
a stale or genuinely broad affected set cannot spend the full timeout in one
multi-GiB Python process. Because the default gate also applies marker filters
for scale tiers, it passes `--testmon-forceselect` so pytest-testmon still
selects affected tests instead of letting pytest marker selection expand the
run. Full diagnostic and seed runs use the same policy, which budgets roughly
768 MiB per worker, reserves host and tmpfs headroom, and reduces concurrency
when memory pressure is elevated.

Every collected test has a 120-second `pytest-timeout` budget. A test that
genuinely needs longer must declare the exception at the test site with
`@pytest.mark.timeout(<seconds>)`; a missing marker can never silently turn into
an unbounded wait. The signal method is the repository default so timeout
failures retain the responsible node and Python stacks in ordinary pytest
output.

Managed pytest temp databases pick their basetemp root through **one**
resolution order, shared by `tests/conftest.py` (direct `pytest` runs) and the
`devtools test`/`devtools verify` preflight
(`devtools.verify_runs.resolve_pytest_basetemp_root`) — there is no second,
independent placement policy that can silently disagree with this one:

1. `POLYLOGUE_PYTEST_BASETEMP_ROOT=/path` — an explicit operator override,
   still headroom-checked (see below), never silently downgraded.
2. `/dev/shm` (tmpfs) — the focused-run default, because measured SQLite fsync
   traffic makes it substantially faster when it clears the free-space
   requirement. Full-suite and seed-testmon runs use it only when
   `POLYLOGUE_PYTEST_TMPFS=1` is explicit.
3. `/realm/tmp/polylogue-pytest` (NVMe scratch) — the broad-run default, and
   the fallback when `/dev/shm` lacks headroom. Broad fixture trees have
   exceeded the supervised 2 GiB tmpfs ceiling while still making progress,
   so their normal route does not guess a future aggregate peak.
4. `/tmp/polylogue-pytest` — reachable **only** when `/realm/tmp` is not
   mounted at all (a genuine cloud sandbox, where `.claude/settings.json`
   sets this as `POLYLOGUE_PYTEST_BASETEMP_ROOT`). On a workstation with
   `/realm` mounted, that same cloud-sandbox env value is stripped before
   candidate selection runs, so it can never leak in as an accidental
   low-space `/tmp` placement — `/tmp` on a workstation is typically a small
   tmpfs shared by every concurrent agent lane, not scratch space.

Before committing to a root, each candidate is checked against a free-space
requirement (`POLYLOGUE_PYTEST_BASETEMP_MIN_FREE_MB`, default 1024 MiB). If
every reachable candidate is starved, the run refuses immediately — before
pytest starts collecting — naming every candidate checked, its free space,
and the requirement, instead of silently placing a basetemp somewhere that
fills up mid-run and surfaces as a bare `OSError: [Errno 28]` in an unrelated
command later.

One shared adaptive tmpfs budget is enforced across all xdist workers (512
MiB to 2 GiB) once a tmpfs root is chosen. Per-run `pytest-polylogue-*`
basetemps are removed at normal pytest shutdown, and pytest startup sweeps
stale per-run dirs from every known root (`/dev/shm`, `/realm/tmp/polylogue-pytest`,
`/tmp/polylogue-pytest`, plus any explicit configured root) — never based on
age alone: each managed basetemp carries a PID plus process-start identity,
and a directory whose exact owner process is still alive is never removed
regardless of age. A tree without a valid managed claim or whose owner cannot
be confirmed dead is never removed. The thirty-minute age threshold applies
only after a positive managed claim identifies a dead owner. The sweeper
restores owner-write permission only after a tree is adjudicated stale, so
published read-only fixture copies cannot leak tmpfs indefinitely. Shared
`pytest-polylogue-*-seeded-*` caches are never touched by the sweep — they
are shared, reused, and built once behind their own `.build.done` guard.

Managed verification refuses to start below 1 GiB available memory instead of
falling back to the pathological disk lane. Every per-test `tmp_path` tree is
reclaimed in fixture teardown, including failures and interruptions; node
failure evidence remains in the managed event, longrepr, summary, and resource
receipts. The controller removes only the exact basetemp it created. An
explicit `--basetemp` is retained for targeted filesystem diagnosis. The
external supervisor and parent runner independently remove the whole run root
on completion or termination, with startup stale-root cleanup as recovery
after an uncatchable process kill or reboot.

An affected run that selects zero tests is accepted only when no executable,
test, dependency, or harness path changed. A zero selection after such a change
fails loudly with the changed paths instead of granting an empty green check.

The default path does not replay cached verify results. Every invocation runs
the static gates and then invokes pytest-testmon for affected-test selection.
Polylogue does not maintain a parallel changed-file router for helper/config
paths; explicit full collection is limited to `devtools verify --seed-testmon`
for dependency-database refreshes and `devtools verify --all` for diagnostics.

`devtools verify` and `devtools test` treat pytest as a bounded, supervised
child workload, not an unowned shell. Each pytest step gets a run directory
under `.cache/verify/runs/<run-id>/` with stdout/stderr, progress, selection,
summary, merged worker events, raw per-worker event files, resource samples,
and a postmortem diagnosis. The latest run is mirrored to
`.cache/verify/current-run.json`, and the latest pytest step is mirrored to:

- `.cache/verify/current-pytest-progress.json`
- `.cache/verify/current-pytest-selection.json`
- `.cache/verify/current-pytest-summary.json`
- `.cache/verify/current-pytest-events.jsonl`
- `.cache/verify/current-pytest-events/`
- `.cache/verify/current-pytest-resources.jsonl`
- `.cache/verify/current-pytest-postmortem.json`
- `.cache/verify/current-pytest-containment.json`
- `.cache/verify/current-pytest-statistics.json` — derived phase
  distributions, worker count, storage, resource peaks, and cleanup outcome;
  the same file is retained under each run's `steps/*/statistics.json`.
- `.cache/verify/current-pytest-output.log`

The devtools process drains pytest output, prints periodic heartbeat lines, and
samples the pytest process tree and host memory/pressure state. A separate
supervisor owns the pytest controller's process group, watches the devtools
owner process, and enforces `POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S` (default 45
minutes). Termination sends SIGTERM to that exact group, then SIGKILL after
`POLYLOGUE_VERIFY_PYTEST_TERM_GRACE_S` (default 5 seconds). On Sinnix, the
supervisor runs in a unique transient scope under the configured build slice;
`KillMode=control-group` and a slightly later `RuntimeMaxSec` are the final
boundary if ordinary cleanup cannot run. Other Linux hosts retain the external
supervisor and process-group boundary and record that fallback honestly in the
containment receipt. If transient scope creation fails in automatic mode, the
runner records the failure and retries with that process-group boundary. The
managed runner requires Linux process identities so it never substitutes an
unsafe numeric-PGID kill on unsupported hosts. The devtools process
independently enforces the same absolute deadline, including supervisor
startup, and also requests group termination when
pytest produces no output for
`POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S` (default 10 minutes).
`POLYLOGUE_VERIFY_RESOURCE_INTERVAL_S` controls resource sampling cadence
(default 2 seconds). Basetemp size is a recursive filesystem walk, so it is
sampled less frequently; `POLYLOGUE_VERIFY_BASETEMP_SIZE_INTERVAL_S` controls
that cadence (default 15 seconds, `0` disables the size walk). Set timeout
variables to `0` only for an explicit diagnostic run where an unusually long
full-suite pass is expected and supervised.

Selection artifacts preserve exact selected/deselected counts but sample node
IDs by default (`POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT`, default 500) so
broad collection does not retain or write unbounded node-id lists in controller
or worker processes.

The detailed artifacts above are checkout-local and disposable. Each `devtools
verify` or `devtools test` invocation automatically appends its compact run
summary to `$XDG_STATE_HOME/polylogue/devtools/verify-history.jsonl` (or the
corresponding `~/.local/state` path), shared across linked worktrees without a
separate recording command. `devtools verify --history` prints the recent
cross-worktree runs. Setup, call, and teardown timings come only from pytest
reports in the event stream.

`devtools test` uses the same pytest progress plugin and process supervisor for
focused selections. During or after a run, inspect
`.cache/verify/current-pytest-progress.json`,
`.cache/verify/current-pytest-selection.json`,
`.cache/verify/current-pytest-summary.json`,
`.cache/verify/current-pytest-events.jsonl`,
`.cache/verify/current-pytest-containment.json`,
and `.cache/verify/current-pytest-output.log` to see the active/latest test node,
selected/deselected node IDs, collection duration, slowest setup/call/teardown
phases, captured output, and termination reason if a focused run stalls.

Optional lane, mutation-campaign, and benchmark commands remain discoverable
through `devtools --help`; pytest and the concrete commands are the behavioral
authority.

### Known limitation: collection-time-only imports are invisible to testmon

`pytest-testmon` only builds a file-to-test dependency edge while a specific
test is *running* (its `pytest_runtest_protocol` hookwrapper opens the tracing
window). Anything a test module or `conftest.py` executes at **collection
time** — a bare `from polylogue.x import Y` at the top of a test file, before
any test in that file has started — falls outside every test's tracing
window and is never recorded, even though the coverage.py summary for a
normal `--cov` run legitimately counts those lines as executed. The result:
declarative-only modules (`TypedDict`/dataclass/`Protocol`/enum/Pydantic
model definitions, no behavior beyond class/field statements) that are only
ever referenced via a top-level import in test files show **zero** rows in
`.cache/testmon/testmondata`'s `file_fp` table, no matter how much of the
file's statements a full-suite coverage run reports as covered. This is
inherent to how testmon (and coverage-context-based selective testing in
general) works — it is **not** dependency-graph staleness, and running
`devtools verify --seed-testmon` does not fix it.

**Blast radius:** the default `devtools verify` gate (`--testmon
--testmon-forceselect`) is the only local pre-merge signal for a change
scoped to one of these files — `devtools test <file>` forwards a literal
pytest selection and is not testmon-aware, so it does not share this gap
(point it at the file's *owning test module*, not the changed source file).
A change confined to one of these files can select zero tests locally and
still report a clean `devtools verify`. The heavy full-suite `devtools verify
coverage` CI job (`.github/workflows/ci.yml`) does not use testmon selection
and still catches such a regression, but only **post-merge** (it is
intentionally off the per-PR gate) — so the exposure window is "merged before
caught," not "never caught."

**Mitigation:** there is no testmon configuration knob for this — it is
upstream tool behavior. When changing a file that is purely declarative
(only type/model/protocol definitions, no function bodies with real logic),
do not trust "0 tests selected" from the default `devtools verify` gate as
proof of safety; run the file's owning test module directly with `devtools
test <test-file>`, and rely on `mypy --strict` (already in the default gate)
to catch structural regressions in `TypedDict`/protocol shapes.

## Test Suite Layout

```text
tests/
├── conftest.py              # Root fixtures (workspace_env, tmp paths)
├── infra/                   # Shared infrastructure
│   ├── storage_records.py   # SessionBuilder, make_message, db_setup
│   ├── tables.py            # Parametrize tables
│   └── strategies/          # Hypothesis strategies (schema-driven payloads)
├── unit/                    # Fast tests (~95% of suite)
│   ├── core/                # Domain: models, filters, roles, timestamps, schema
│   ├── sources/             # Parser crashlessness, null guards, acquisition
│   ├── storage/             # FTS5, hybrid search, CRUD, scale
│   ├── pipeline/            # Stage independence, resilience
│   ├── cli/                 # Commands, terminal snapshots (syrupy)
│   ├── mcp/                 # MCP tool contracts, edge cases
│   ├── demo/                # Demo archive seed/verify workflows
│   └── security/            # Protected — never delete
├── property/                # Hypothesis property tests
├── integration/             # End-to-end (slow, protected)
├── benchmarks/              # pytest-benchmark suite
└── fuzz/                    # Atheris fuzz targets
```

## Test Patterns

**`workspace_env` fixture** (`conftest.py`): Isolated XDG paths and archive
root in `tmp_path`. Disables schema validation by default. Most tests
that touch storage or pipeline use this.

**`SessionBuilder`** (`infra/storage_records.py`): Fluent builder for
populating a test database. Chain `.title()`, `.provider()`, `.message()`, etc.
and call `.build()` to persist.

**`make_message()` / `make_session()`** (`infra/storage_records.py`):
Quick factories for creating model instances without database setup.

**`corpus_seeded_db`** (`infra/corpus_fixtures.py`): Pre-populated database
fixture using the synthetic corpus generator. For tests needing a realistic archive.

**Hypothesis strategies** (`infra/strategies/`): Schema-driven payload
generators. `schema_conformant_payload(provider)` produces payloads that match
each provider's JSON schema.

### Cardinality guards on loop-only assertions

A test whose only assertions live inside `for item in result: ...` passes
vacuously if `result` is empty — a regression that makes the exercised code
return nothing (the total-failure mode) hides behind the unreached loop body.
Before merging such a test, add an explicit non-empty check ahead of the loop:

```python
result = await SessionFilter(archive_root=repo).exclude_origin("codex-session").list()
assert len(result) >= 1  # or == N when the fixture's count is exact and stable
for session in result:
    assert session.origin != "codex-session"
```

This guard is unnecessary when the loop body delegates to a shared assertion
helper that itself asserts non-emptiness (e.g. `_assert_structured_error`), or
when an earlier line in the same test already establishes a non-empty
invariant (`assert len(result) == count()`, a prior `.first()` call, etc.).
Skip the guard rather than adding a redundant one in those cases.

## Time and Clock

Timestamp-sensitive tests opt into the `frozen_clock` fixture so the test's
"now" and the production code's "now" coincide. Reading the host wall clock
directly creates two failure modes: flakiness at threshold edges (cursor lag
warning/error/critical bands, freshness windows, retry backoff) and snapshot
churn that hides real regressions.

`tests/infra/frozen_clock.py` exports:

- `FrozenClock` — controlled clock with explicit `advance(seconds)` and
  `set_time(epoch)` mutators. Reading the clock does not implicitly advance
  it; a single `now()` read in production code stays stable across the
  whole call.
- `freeze_clock(start=..., patch_datetime_in_modules=[...])` — context
  manager. Patches `time.time` and `time.monotonic` globally for the scope
  and replaces `datetime` in each named production module with a frozen
  subclass whose `.now()` reads the clock.
- `frozen_clock` pytest fixture — yields a `FrozenClock` and honors
  `@pytest.mark.frozen_clock_modules("polylogue.x.y", ...)` to extend the
  `datetime.now` patching list per-test.
- `fixed_now()` — returns a stable `datetime` anchor without patching
  anything (use only when production code does NOT itself read the clock).

Usage:

```python
import pytest
from datetime import timedelta
from tests.infra.frozen_clock import FrozenClock


@pytest.mark.frozen_clock_modules("polylogue.daemon.health")
def test_lag_alert(frozen_clock: FrozenClock, tmp_path):
    now = frozen_clock.now()
    seed_cursor(tmp_path, updated_at=(now - timedelta(seconds=120)).isoformat())
    alerts = _check_cursor_lag_medium()  # production reads frozen now
    assert alerts[0].severity == HealthSeverity.WARNING
```

For a single moment-in-time anchor without patching (e.g. when only
constructing an opaque metadata timestamp), use `fixed_now()` instead of
`datetime.now(UTC)`.

Direct host-clock reads from test code are not merely linted — they are
unreachable. `tests/infra/clock_guard.py` installs an autouse fixture on
every test that patches `time.time` / `time.monotonic` / `time.monotonic_ns`
/ `time.time_ns` (frame-checked, so production code under test still reads
the real clock) and, for a guarded test module's own `datetime` symbol,
swaps it for a subclass whose `.now()` / `.utcnow()` raise. Reaching for the
host clock directly from a test fails immediately with a `RuntimeError`
pointing back at this section — there is no separate lint to remember to run.

Tests that genuinely need the real clock (timing benchmarks, fuzz harnesses,
tests that wait on real OS thread/process state) opt out explicitly, next to
the code that needs it, instead of via an external registry:

```python
pytestmark = pytest.mark.uses_real_clock("timing benchmark measures real latency")
```

`tests/infra/` and `conftest.py` files are exempt (they are the harness).
Tests that request the `frozen_clock` fixture are also exempt from the guard
— `frozen_clock` manages the clock itself.

## Demo and Visual Behavior Checks

The deterministic demo archive is the supported private-data-free way to run
read/search examples and reader smoke checks. The direct seed/verify commands
create a ready-to-query archive without daemon scheduling; the import path uses
the daemon and can wait for the same semantic verifier.

```bash
# Source-only demo archive, no daemon required
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json
polylogue demo script --shell bash

# Daemon-backed demo path, waits for convergence before returning
polylogue import --demo --wait --timeout 30 --with-overlays

# Behavior-backed docs/visual lane
uv run devtools test tests/unit/cli/test_demo_command.py tests/unit/demo/test_demo_seed_verify.py tests/visual
```

Browser or deployment media remains local operator evidence unless the run is
backed by an explicit command artifact. The fast visual lane is browserless and
checks HTTP/DOM/API contracts rather than screenshots.

## Mutation Testing

```bash
devtools bench mutation list
devtools bench mutation run <campaign>
```

Policy:

- keep the committed mutmut configuration broad; narrow work happens through
  focused campaigns
- write per-run JSON artifacts under `.local/mutation-campaigns/`

## Protected Files

Never delete:

- **`tests/unit/sources/test_parsers_props.py`**, **`test_null_guard_properties.py`**:
  Hypothesis property tests ensuring parsers never crash on arbitrary input and
  handle nulls in every field position.

- **`tests/integration/`**: End-to-end pipeline tests against real archive
  shapes.
- **`tests/unit/security/`**: Security boundary tests.
