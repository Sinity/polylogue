# Testing

Use the project devshell before running commands:

```bash
cd /realm/project/polylogue
direnv allow   # one-time setup; afterward the devshell loads automatically on cd
```

If you are not using `direnv`, enter the same environment manually:

```bash
nix develop
```

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

## Test Suite Layout

```text
tests/
├── conftest.py              # Root fixtures
├── infra/                   # Shared infrastructure
│   ├── helpers.py           # ConversationBuilder, db_setup
│   ├── tables.py            # Parametrize tables
│   └── strategies/          # Hypothesis strategies
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
│   ├── test_rendering_preservation.py  # Block rendering invariants
│   └── test_schema_roundtrip.py        # Schema extraction roundtrips
├── integration/             # End-to-end (slow, protected)
├── benchmarks/              # pytest-benchmark suite
└── fuzz/                    # Atheris fuzz targets
```

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
mutmut run        # 8 modules: models, filters, roles, timestamps, hashing, json, fts5, hybrid
mutmut results    # View results
```

## Protected Files

- `tests/unit/sources/test_parsers_props.py`, `test_null_guard_properties.py`
- `tests/unit/core/test_properties.py`
- `tests/integration/`
- `tests/unit/security/`
- `tests/unit/storage/test_crud.py`
