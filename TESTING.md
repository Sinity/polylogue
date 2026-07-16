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
`.cache/testmon/seed-attempt.json` before work begins. A matching interrupted
attempt resumes only its unseen, failed, or changed tests; `.cache/testmon/seed.json`
is published only after every originally selected node has a failure-free row
in the dependency database.

Plain focused `pytest` runs are single-process by default so small inner-loop
checks do not spawn a worker pool. `devtools verify` keeps pytest-testmon as
the affected-test selector and runs the selected default lane with a bounded
worker pool (`-n 4` by default, override with `POLYLOGUE_PYTEST_WORKERS`) so
a stale or genuinely broad affected set cannot spend the full timeout in one
multi-GiB Python process. Because the default gate also applies marker filters
for scale tiers, it passes `--testmon-forceselect` so pytest-testmon still
selects affected tests instead of letting pytest marker selection expand the
run. Full diagnostic and seed runs use the same worker override and also
default to `-n 4` so database-heavy workers do not multiply memory and I/O
pressure beyond the explicit broad-run lane.

Every collected test has a 120-second `pytest-timeout` budget. A test that
genuinely needs longer must declare the exception at the test site with
`@pytest.mark.timeout(<seconds>)`; a missing marker can never silently turn into
an unbounded wait. The signal method is the repository default so timeout
failures retain the responsible node and Python stacks in ordinary pytest
output.

Pytest temp databases default to `/realm/tmp/polylogue-pytest` so interrupted
full or xdist runs do not leave multi-GiB `/dev/shm` directories resident in
RAM. On btrfs scratch volumes, the harness best-effort marks that root with
`chattr +C` so new per-run SQLite files avoid CoW amplification. Use
`POLYLOGUE_PYTEST_TMPFS=1` only for explicit performance lanes that can afford
tmpfs pressure. `POLYLOGUE_PYTEST_BASETEMP_ROOT=/path` overrides both. Per-run
`pytest-polylogue-*` basetemps are removed at normal pytest shutdown, and
pytest startup sweeps stale per-run dirs from both the configured root and
legacy `/dev/shm`; shared `pytest-polylogue-seeded-*` caches are kept because
they are small and reused.

No verification lane enables tmpfs automatically. Broad xdist runs can retain
many SQLite basetemps concurrently, so an apparently roomy `/dev/shm` can turn
the suite into host-wide memory and swap pressure. Set
`POLYLOGUE_PYTEST_TMPFS=1` explicitly for a measured one-shot performance lane;
normal seed, full, focused, and affected runs stay on the managed scratch root.

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

`devtools workspace tasks recent` shows the run id, diagnosis, and peak pytest
RSS when the current run metadata is available. `devtools workspace tasks stats
--resources` aggregates recorded pytest memory peaks over time.

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

For optional lane, mutation-campaign, and benchmark inventories, see
[docs/test-quality-workflows.md](docs/test-quality-workflows.md). Those registries are
secondary navigation over executable checks; the source of truth for behavior is
pytest plus the concrete `polylogue`/`devtools` commands they invoke.

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

Confirmed reproducible (2026-07-12, polylogue-csg7) with an isolated,
freshly-seeded testmon run scoped to exactly one test file: after
`TESTMON_DATAFILE=<scratch> pytest --testmon --testmon-noselect
tests/unit/devtools/test_verify_manifests.py`, `polylogue/verification/manifests/models.py`
still has 0 `file_fp` rows even though a `--cov` run over the same test file
reports 80% statement coverage on that module — all of it from Pydantic
model/field declarations executed when `devtools.verify_manifests` is
imported at module-collection time; every uncovered line is inside a
`@field_validator`/`@model_validator` method body, which only runs when
`validate_manifest()` is actually called (no test in that file calls it).

Cross-referencing a full-suite `coverage.json` (`--cov=polylogue`) against
`file_fp` filenames finds **95 files** under `polylogue/` with nonzero
covered statements but zero testmon dependency rows — largely `*_models.py`,
`types.py`, `protocols.py`, `enums.py`, and `api/contracts/*.py`, i.e. modules
whose test-suite touch points are import-only. Query:

```python
import json, sqlite3
cov = json.load(open(".cache/coverage/coverage.json"))["files"]
tm = {r[0] for r in sqlite3.connect(".cache/testmon/testmondata")
      .execute("SELECT DISTINCT filename FROM file_fp")}
gaps = [(f, d["summary"]["covered_lines"]) for f, d in cov.items()
        if d["summary"]["covered_lines"] > 0 and f not in tm]
```

**Blast radius:** the default `devtools verify` gate (`--testmon
--testmon-forceselect`) is the only local pre-merge signal for a change
scoped to one of these files — `devtools test <file>` forwards a literal
pytest selection and is not testmon-aware, so it does not share this gap
(point it at the file's *owning test module*, not the changed source file).
A change confined to one of these 95 files can select zero tests locally and
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
to catch structural regressions in `TypedDict`/protocol shapes. See
polylogue-csg7 for the investigation and a follow-up tracking item for making
this gap machine-checkable.

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
    alerts = _check_cursor_lag_medium()   # production reads frozen now
    assert alerts[0].severity == HealthSeverity.WARNING
```

For a single moment-in-time anchor without patching (e.g. when only
constructing an opaque metadata timestamp), use `fixed_now()` instead of
`datetime.now(UTC)`.

The `devtools verify-test-clock-hygiene` lint runs in the default verify
gate. It rejects any new direct call to `datetime.now`, `datetime.utcnow`,
`time.time`, or `time.monotonic` from a test file outside the explicit
allowlist in `docs/plans/test-clock-allowlist.yaml`. Tests that genuinely
need the host clock (timing benchmarks, fuzz harnesses, the
`frozen_clock` self-tests) add their path to the allowlist with a
one-line rationale; everything else migrates to the fixture.

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
devtools bench mutation index
```

Policy:

- keep the committed mutmut configuration broad; narrow work happens through
  focused campaigns
- write local artifacts under `.local/mutation-campaigns/`
- rebuild the mutation index after a campaign run

## Protected Files

Never delete:

- **`tests/unit/sources/test_parsers_props.py`**, **`test_null_guard_properties.py`**:
  Hypothesis property tests ensuring parsers never crash on arbitrary input and
  handle nulls in every field position.

- **`tests/integration/`**: End-to-end pipeline tests against real archive
  shapes.
- **`tests/unit/security/`**: Security boundary tests.
