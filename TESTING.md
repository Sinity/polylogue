# Testing

All commands below assume you are inside the project devshell. See
[CONTRIBUTING.md](CONTRIBUTING.md) for environment setup.

## Running Tests

```bash
# Normal repository verification
devtools verify

# Focused inner-loop runs: prefer `devtools test` over raw pytest. It provides
# the project environment, a single-process default, progress artifacts,
# import-root validation, and a typed outcome receipt. AgentCTL owns execution
# concerns for declared verification jobs.
# Any pytest arguments go after the command name.
devtools test tests/unit/storage/test_hybrid_laws.py
devtools test -k "test_name"
devtools test tests/unit/pipeline -x
POLYLOGUE_PYTEST_WORKERS=8 devtools test tests/unit/storage   # ask for a wider run

# Raw pytest still works for ad-hoc needs the wrapper does not cover:
pytest -x tests/unit/storage/test_hybrid_laws.py

# Complete-corpus baseline (unit/property/fuzz/integration; benchmarks excluded)
devtools verify --all

# Full Nix/CI parity
nix flake check
```

AgentCTL owns scratch placement and cleanup for declared verification jobs.
Foreground commands use pytest's ordinary temporary-directory behavior.

### Managed pytest pools

The operation owns pytest admission. `.agentctl/project.toml` declares
`pytest_focused` in `pytest-quick`, `verify_affected` in `pytest`, and
`verify_all` in `pytest-heavy`; follow those mappings on the current
candidate. A job ID, principal, or operation name is not slot ownership.

- A workstation focused run submits the declared `pytest_focused` operation,
  `agentctl job start <checkout> pytest_focused --workspace <checkout> --
  <launch file>`, waits for the runtime's exit code, and prints the captured
  log path under `.cache/verify/pytest-slot-<pid>.log`. Watch it with
  `agentctl job list --active` or `agentctl job logs <id>`; a killed waiter
  cancels its job with `agentctl job cancel <id>`.
- The `agentctl` client inherits only what the runtime needs (`HOME`, `PATH`,
  the XDG and session variables); the managed pytest environment travels in
  the launch file the job consumes and deletes.
- `agentctl job start polylogue pytest_focused --wait -- <selection>` runs
  `devtools test <selection>` inside the pool directly.
- If the runtime is unreachable the run refuses rather than running unqueued:
  `systemctl --user start pueued`, then rerun.

`POLYLOGUE_PYTEST_SLOT=held` is the explicit assertion that the caller already
holds the slot, for the hermetic test of this mechanism.

Worker width is bounded by the pytest cgroup, never by host cores or by
instantaneous host free memory: this suite is SQLite-IO bound, and admission
under host pressure is already the pool's decision when the job starts. The
default is `worker_memory.CORPUS_MAX_WORKERS`, derived as
`(agentctl-pytest.slice MemoryHigh - controller peak) // worker peak`; the
slice's ceiling is Sinnix's to set (`flake/data/runtime-defaults.nix`) and it
already carries the safety margin, so nothing is discounted a second time
here. A `POLYLOGUE_PYTEST_WORKERS` override above what the slice's *remaining*
budget holds is narrowed when the run takes the slot, and the receipt records
which bound narrowed it.

Managed runs keep their temporary trees under `.cache/verify/tmp-<pid>-*`
inside the checkout and remove them on every exit except a failed run's,
whose leftovers are worth reading.

### Typed WebUI package checks

Run the package-owned generated-contract, lint, type, unit, client-contract,
and build checks through the project-level route:

```bash
devtools gate webui
```

For the managed deployed-reader browser smoke, use the declared AgentCTL
operation:

```bash
agentctl job start polylogue deployment_browser_smoke --workspace <path>
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

Focused `devtools test <selection>` does not load or update testmon. Its
receipt records the checkout graph only as a diagnostic.

Ordinary `devtools verify` runs static gates and then uses a usable
pytest-testmon graph at `.cache/testmon/testmondata` (environment
`polylogue`) to measure and bound an affected selection. It can snapshot a
better primary-checkout graph into a lane through SQLite backup. If no usable
graph is available, or the selection would re-execute the corpus or otherwise
exceed its bound, verification records and refuses the affected plan before
pytest. It never falls back to a full corpus run. `devtools verify --all` is
the explicit complete-corpus command and updates the graph; `--quick` runs
static gates only.

AgentCTL's separately declared `verify_all` operation invokes `devtools verify
--all` and has a 03:00 schedule in `.agentctl/project.toml`. That declaration
does not prove a live timer; inspect AgentCTL runtime state when the schedule
matters.

The corpus runs as ONE collection. testmon drops every recorded test a run did
not collect, so a partitioned run would keep only its last partition's edges.

Every run gets a directory under `.cache/verify/runs/<run-id>/` whose
`run.json` is the receipt: whether the run executed, its exit code, its
diagnosis, and the decoded pytest outcomes per step. Each pytest step keeps
progress, selection, summary, merged worker events, and its statistics under
`steps/<step-id>/`. `devtools test` prints the path of the receipt it just
wrote as its last line, with the verdict; read that path and nothing else. A
receipt that is absent is a run that did not happen.

The latest run in the checkout is mirrored to `.cache/verify/current-run.json`
and its decoded outcomes to `.cache/verify/current-pytest-statistics.json`.
`devtools verify` additionally mirrors its latest pytest step to
`.cache/verify/current-pytest-progress.json`,
`current-pytest-selection.json`, `current-pytest-summary.json`,
`current-pytest-events.jsonl` and `current-pytest-events/`. Those `current-*`
names are last-run-wins across a checkout where lanes and batches run
concurrently, so they answer "what ran here most recently", never "what did
the run I started find".

Focused and verification runs are foreground semantic commands. Devtools records
project selection, gate results, decoded pytest outcomes, and the scope it
actually ran.

Selection artifacts preserve exact selected/deselected counts but sample node
IDs by default (`POLYLOGUE_PYTEST_SELECTION_NODEID_LIMIT`, default 500) so
broad collection does not retain or write unbounded node-id lists in controller
or worker processes.

The detailed artifacts above are checkout-local and disposable. Each `devtools
verify` or `devtools test` invocation appends its compact terminal summary to
`.cache/verify/history.jsonl` before any detail is removed. The append and
retention pass share one authenticated lock. Detail retention keeps the newest
eight successful runs and, for failures, always keeps the newest run plus up to
twelve runs that fit within seven days and 64 MiB. A malformed receipt, unsafe
tree, or active lock retains the affected detail instead of guessing. `devtools
why --history HOURS` reads the compact checkout-local history. A
native verify record carries its selected scope, gate-step results, and decoded
pytest outcomes. Setup, call, and teardown timings come only from pytest
reports in the event stream.

`devtools test` uses the pytest progress plugin for focused selections. During
or after a run, inspect its step directory — `progress.json`, `selection.json`,
`summary.json`, `events.jsonl`, `output.log` — for selected/deselected node
IDs, collection duration, slowest setup/call/teardown phases, and captured
output.

Optional lane and benchmark commands remain discoverable
through `devtools --help`; pytest and the concrete commands are the behavioral
authority.

Declaration-only modules need no cohort or allowlist. Their structural
contracts remain protected by `mypy --strict`, which runs in every verify.

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

**`seeded_archive` / `named_seeded_archive` / `named_seeded_archive_ro`**
(`infra/corpus_fixtures.py`): Pre-populated archive fixtures using the
synthetic corpus generator. `named_seeded_archive_ro` points the archive
root straight at the shared immutable artifact for read-only consumers;
`named_seeded_archive` clones a private writable copy for consumers that
mutate the archive (ingest, insight rebuild, marks, maintenance).

**Shared session archives** (`infra/shared_session_archives.py`): The archives
a whole xdist run shares. A cold build takes a per-key exclusive lock, so the
controller warms all of them before workers start; a build that raises is
reported and left to its own fixture. Every session-scoped archive fixture
resolves its artifact through a builder declared here, so an archive reachable
from a fixture is also warmed.

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

## Golden Bytes for the Root Query Surface

Two harnesses in `scripts/` replay the root `find`/`analyze` surface in-process
against the immutable seeded `cli-mixed` workload with a frozen clock, and
record exit code plus exact stdout/stderr bytes. A refactor that claims to
preserve behaviour runs **both** before and after and diffs them.

```bash
# Every documented `polylogue ...` invocation in docs/cli-reference.md
uv run python scripts/golden_find_bytes.py BEFORE.json

# Hand-picked invocations that produce rows
uv run python scripts/golden_nonempty_bytes.py AFTER.json --compare BEFORE-NONEMPTY.json
```

They are a pair, and the second is not optional. The documented set is the
right acceptance evidence for "the documented surface did not change", but on
the seeded corpus its aggregate and `id:`-scoped examples all match nothing, so
a change can pass every documented invocation without one aggregate row, ranked
hit, page boundary or attached-unit projection ever being rendered. Zeroing
`total` on every list page, for instance, leaves all 37 documented goldens
byte-identical and is caught only by the non-empty companion.

`golden_nonempty_bytes.py --compare` exits nonzero when any case changed, and
normalizes the three authority fields (`run_id`, `generation_id`,
`archive_epoch`) that vary per run on an unmodified checkout. `golden_find_bytes.py`
records only; diff its output yourself.

## Protected Files

Never delete:

- **`tests/unit/sources/test_parsers_props.py`**, **`test_null_guard_properties.py`**:
  Hypothesis property tests ensuring parsers never crash on arbitrary input and
  handle nulls in every field position.

- **`tests/integration/`**: End-to-end pipeline tests against real archive
  shapes.
- **`tests/unit/security/`**: Security boundary tests.
