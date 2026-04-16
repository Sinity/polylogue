# Testing

All commands below assume you are inside the project devshell. See
[CONTRIBUTING.md](CONTRIBUTING.md) for environment setup.

## Running Tests

```bash
# Fast unit run (primary workflow)
pytest -q --ignore=tests/integration

# Stop on first failure
pytest -x --ignore=tests/integration

# Specific file or test
pytest tests/unit/storage/test_hybrid_laws.py
pytest -k "test_name"

# Full CI parity
nix flake check
```

For the generated validation-lane, mutation-campaign, and benchmark inventory,
see [docs/test-quality-workflows.md](docs/test-quality-workflows.md).

## Test Suite Layout

```text
tests/
├── conftest.py              # Root fixtures (workspace_env, tmp paths)
├── infra/                   # Shared infrastructure
│   ├── storage_records.py   # ConversationBuilder, make_message, db_setup
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

**`ConversationBuilder`** (`infra/storage_records.py`): Fluent builder for
populating a test database. Chain `.title()`, `.provider()`, `.message()`, etc.
and call `.build()` to persist.

**`make_message()` / `make_conversation()`** (`infra/storage_records.py`):
Quick factories for creating model instances without database setup.

**`corpus_seeded_db`** (`infra/corpus_fixtures.py`): Pre-populated database
fixture using the synthetic corpus generator. For tests needing a realistic archive.

**Hypothesis strategies** (`infra/strategies/`): Schema-driven payload
generators. `schema_conformant_payload(provider)` produces payloads that match
each provider's JSON schema.

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
- **`tests/unit/core/test_properties.py`**: Domain model invariants (role
  normalization, timestamp parsing, hashing determinism).
- **`tests/integration/`**: End-to-end pipeline tests against real archive
  shapes.
- **`tests/unit/security/`**: Security boundary tests.
- **`tests/unit/storage/test_crud.py`**: Core storage contract tests (create,
  read, update, delete round-trips).
