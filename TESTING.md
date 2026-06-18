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

`devtools verify` uses pytest-testmon for per-test affected selection. The
seed command records `.testmondata` plus `.cache/testmon/seed.json`; those
files are local generated state and are not committed. If the seed is missing,
the default command fails with setup guidance instead of silently running the
whole suite.

Plain focused `pytest` runs are single-process by default so small inner-loop
checks do not spawn a worker pool. `devtools verify` keeps pytest-testmon as
the affected-test selector and runs selected tests single-process (`-n 0` by
default, override with `POLYLOGUE_PYTEST_WORKERS`). Because the default gate
also applies marker filters for scale tiers, it passes `--testmon-forceselect`
so pytest-testmon still selects affected tests instead of letting pytest marker
selection expand the run. Full diagnostic and seed runs use the same worker
override and default to `-n 16`.

Pytest temp databases default to `/realm/tmp/polylogue-pytest` so interrupted
full or xdist runs do not leave multi-GiB `/dev/shm` directories resident in
RAM. Use `POLYLOGUE_PYTEST_TMPFS=1` only for explicit performance lanes that
can afford tmpfs pressure. `POLYLOGUE_PYTEST_BASETEMP_ROOT=/path` overrides
both. Per-run `pytest-polylogue-*` basetemps are removed at normal pytest
shutdown, and pytest startup sweeps stale per-run dirs from both the configured
root and legacy `/dev/shm`; shared `pytest-polylogue-seeded-*` caches are kept
because they are small and reused.

The default path does not replay cached verify results. Every invocation runs
the static gates and then invokes pytest-testmon for affected-test selection.
Polylogue does not maintain a parallel changed-file router for helper/config
paths; explicit full collection is limited to `devtools verify --seed-testmon`
for dependency-database refreshes and `devtools verify --all` for diagnostics.

`devtools verify` treats the pytest subprocess as a bounded child workload, not
an unowned shell. It prints periodic heartbeat lines, drains pytest output
incrementally so real progress resets the stall clock, and terminates the whole
pytest process group if the step exceeds
`POLYLOGUE_VERIFY_PYTEST_TIMEOUT_S` (default 45 minutes) or produces no output
for `POLYLOGUE_VERIFY_PYTEST_STALL_TIMEOUT_S` (default 10 minutes). Set either
variable to `0` only for an explicit diagnostic run where an unusually long
full-suite pass is expected and supervised.

`devtools test` uses the same pytest progress plugin and process supervisor for
focused selections. During or after a run, inspect
`.cache/verify/current-pytest-progress.json`,
`.cache/verify/current-pytest-events.jsonl`, and
`.cache/verify/current-pytest-output.log` to see the active/latest test node,
captured output, and termination reason if a focused run stalls.

For the generated validation-lane, mutation-campaign, and benchmark inventory,
see [docs/test-quality-workflows.md](docs/test-quality-workflows.md).

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
│   ├── showcase/            # QA runner, reports
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

## QA Exercises

```bash
# Seeded (fast, no real data)
POLYLOGUE_FORCE_PLAIN=1 polylogue audit --only exercises --tier 0

# Schema audit (instant)
POLYLOGUE_FORCE_PLAIN=1 polylogue audit --only audit

# Live (against real DB)
POLYLOGUE_FORCE_PLAIN=1 polylogue audit --live --only exercises --tier 0
```

## Mutation Testing

```bash
devtools mutmut-campaign list
devtools mutmut-campaign run <campaign>
devtools mutmut-campaign index
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
