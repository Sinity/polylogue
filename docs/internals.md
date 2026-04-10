# Internals Reference

This is the working map of the live codebase: invariants, hot files,
and maintenance landmarks. For the conceptual system shape, see
[architecture.md](architecture.md). For the generated validation and campaign
catalog, see [test-quality-workflows.md](test-quality-workflows.md).

## Fast Commands

```bash
pytest -q --ignore=tests/integration
ruff check polylogue tests devtools
devtools status
devtools render-all --check
polylogue doctor --repair --preview
polylogue audit --only exercises --tier 0
```

## Key Invariants

| Invariant | Notes |
| --- | --- |
| Archive writes are idempotent | unchanged conversations are skipped by content hash |
| Content hash excludes editable metadata | titles, summaries, and tags can change without re-import churn |
| Search and products read from the archive, not from provider-specific files | surface code should not bypass the substrate |
| SQLite schema is fresh-only on mismatch | current storage schema version is `v2` |
| FTS assumptions use `unicode61` | porter stemming is not assumed in this build |
| Repo-maintenance automation goes through `devtools` | avoid new shell-wrapper entrypoints |

## Hot Files

### Root Entry Points

| File | Purpose |
| --- | --- |
| `polylogue/facade.py` | async-first public API |
| `polylogue/config.py` | runtime configuration and XDG resolution |
| `polylogue/cli/click_app.py` | root query-first CLI dispatch |
| `polylogue/cli/command_inventory.py` | command inventory for the public CLI surface |
| `polylogue/operations/archive.py` | archive-facing high-level operations |

### Storage and Search

| File | Purpose |
| --- | --- |
| `polylogue/storage/backends/schema_ddl.py` | schema definition and `SCHEMA_VERSION` |
| `polylogue/storage/backends/schema_upgrade.py` | fresh-init and version guard |
| `polylogue/storage/repository.py` | repository facade over archive reads and writes |
| `polylogue/storage/repository_archive_*.py` | archive read and query helpers |
| `polylogue/storage/repository_product_*.py` | product read helpers |
| `polylogue/storage/search_providers/fts5.py` | lexical search provider |
| `polylogue/storage/search_providers/hybrid.py` | hybrid retrieval routing |
| `polylogue/storage/search_providers/sqlite_vec.py` | vector-search integration |

### Sources and Pipeline

| File | Purpose |
| --- | --- |
| `polylogue/sources/dispatch.py` | provider detection and parser routing |
| `polylogue/sources/parsers/*.py` | provider-specific import logic |
| `polylogue/pipeline/runner.py` | stage orchestration |
| `polylogue/pipeline/services/*.py` | acquire, parse, materialize, render, index services |

### Products and Maintenance

| File | Purpose |
| --- | --- |
| `polylogue/products/*.py` | derived-product models |
| `polylogue/storage/session_product_*.py` | product persistence and rebuild logic |
| `polylogue/schemas/*.py` | schema inference, verification, provider bundles |
| `polylogue/showcase/*.py` | deterministic exercises and acceptance harness |
| `devtools/*.py` | repo maintenance tools |

## Local State

Use the hidden roots instead of creating new top-level scratch directories:

- [`.cache/README.md`](../.cache/README.md) for disposable caches
- [`.local/README.md`](../.local/README.md) for untracked outputs

Important examples:

- `.cache/hypothesis/` for Hypothesis storage
- `.cache/pytest/` and `.cache/pytest-benchmark/`
- `.cache/mypy/` and `.cache/ruff/`
- `.local/mutation-campaigns/`
- `.local/benchmark-campaigns/`
- `.local/showcase/` or other demo or verification outputs

## Debugging Landmarks

When a change lands in one surface, check the adjacent surface that shares the
same archive semantics:

- query behavior: `cli/query*.py`, `lib/filters.py`, `storage/search*.py`
- pipeline behavior: `cli/commands/run.py`, `pipeline/`, `storage/`, `products/`
- maintenance behavior: `cli/commands/check.py`, `storage/repair.py`, `health.py`
- publication behavior: `rendering/`, `site/`, `showcase/`
- schema behavior: `schemas/`, `sources/providers/`, `pipeline/services/validation_*`

The fastest way to spot drift is to compare the archive-facing implementation
and the public docs together:

```bash
devtools render-all --check
devtools verify-showcase
```
