# Test Design and Verification Record

## Test strategy

The verification strategy targets the three failure shapes from the epic and the real production dependencies that previously bypassed the documented resolver:

1. split configuration authority;
2. nested-table replacement;
3. direct environment/cwd/home/path bypasses.

The central regression module is `tests/unit/core/test_config_runtime_closure.py`. Its parameter set comes from `config_inventory()`, so a future scalar `toml_path` entry is automatically exercised rather than relying on a manually maintained allowlist.

## Production dependencies and anti-vacuity mutations

| Test contract | Production dependency exercised | Representative mutation/removal that makes it fail |
|---|---|---|
| Every scalar TOML inventory entry reaches a typed property | `_merge_toml`, inventory metadata, `PolylogueConfig` accessors | Add a `toml_path` entry without merge/property plumbing, or omit an existing key |
| Health tables preserve siblings/families | recursive table merge in `config.py` | Restore full replacement, shallow-spread only the top level, or replace `families` wholesale |
| Subscription plans replace | array-table handling | Apply recursive/list merge to arrays of tables |
| Five-layer precedence and provenance | loader order and per-key layer map | Swap layers, skip TOML, or derive provenance only when values differ |
| Cloud-shell archive env wins | environment layer | Give user TOML priority over `POLYLOGUE_ARCHIVE_ROOT` |
| Snapshot stability | `ResolvedRuntimeConfig`, compatibility/path projections | Reread env, cwd, home, XDG, or `polylogue.paths` after construction |
| TOML Voyage key reaches provider | `execute_embed_stage` and `create_vector_provider` | Restore raw `VOYAGE_API_KEY` lookup or discard `IndexConfig` projection |
| Effective payload | provenance renderer, redaction, runtime tier/identity projection | Render only legacy settings, serialize secret, or omit paths/identity |
| API/CLI/MCP/services agree | `AppEnv`, `RuntimeServices`, `Polylogue`, MCP service scope | Let any object independently resolve archive/db paths |
| Daemon and maintenance use TOML archive | real Click roots | Treat Click defaults as explicit, resolve maintenance paths independently, or keep daemon env-only root |
| AST environment bypass guard | every `polylogue/**/*.py` runtime module | Add `os.environ.get`, `os.getenv`, `os.environ[...]`, or Click `envvar=` for an inventoried variable outside `config.py` |

Existing tests were updated where they deliberately patched the old loader alias or expected live environment rereads. They now install/patch the runtime projection. No tests were deleted.

## Commands and results

### Patch and static checks

Executed successfully:

```bash
git diff --check
ruff check <73 changed Python files>
ruff format --check <73 changed Python files>
python -m py_compile <73 changed Python files>
mypy --follow-imports=skip --show-error-codes \
  polylogue/config.py polylogue/services.py polylogue/paths/_roots.py \
  polylogue/cli/shared/types.py polylogue/api/__init__.py polylogue/mcp/server.py
```

Results:

- Ruff lint: pass
- Ruff formatting: pass, 73 files already formatted
- unified-diff whitespace check: pass
- Python compilation: 73 changed Python files compiled
- focused type check: success, no issues in six authority/composition modules

### All modified test modules

```bash
pytest -q --tb=short --timeout=60 <all 21 changed test modules>
```

Result: **946 passed**, 8 warnings, 17.64 seconds.

The warnings are Python 3.13 `fork()` deprecation warnings from an existing multiprocessing path in `test_execute_ingest_stage_routes_active_archive_to_native_writer`; they are not assertion failures.

### Core resolver/inventory/closure

```bash
pytest -q --tb=short --timeout=60 \
  tests/unit/core/test_config.py \
  tests/unit/core/test_config_inventory.py \
  tests/unit/core/test_config_runtime_closure.py
```

Result in the implementation worktree: **172 passed** in 7.16 seconds.

The same command was run after applying `PATCH.diff` to a fresh detached worktree at the named snapshot. Result: **172 passed** in 5.05 seconds.

### Daemon security

```bash
pytest -q --tb=short --timeout=60 tests/unit/daemon/test_daemon_http_security.py
```

Result: **512 passed** in 3.07 seconds.

### MCP and embedding status

```bash
pytest -q --tb=short --timeout=60 \
  tests/unit/mcp/test_envelope_contracts.py \
  tests/unit/cli/test_embed_status_fast.py
```

Result: **67 passed** in 6.10 seconds.

The targeted MCP runtime module separately passed **3 tests**. The full MCP directory was not certified because unrelated long-running tests exceeded the command window; the one failure initially exposed by that run was repaired and its complete 35-test file now passes.

### Pipeline and source consumers

Executed individually:

- `tests/unit/pipeline/test_archive_ingest_commit_batching.py`: **9 passed**
- `tests/unit/pipeline/test_ingest_batch.py`: **55 passed**
- `tests/unit/pipeline/test_ingest_batch_resource_bounds.py`: **4 passed**
- `tests/unit/pipeline/test_run_stages_runtime.py`: **6 passed**
- `tests/unit/sources/test_drive_auth.py`: **33 passed**
- `tests/unit/daemon/test_http_write_coordination.py`: **10 passed**

`tests/unit/pipeline/test_validation_parallelism_contracts.py` hung before producing a result in this environment and is marked unverified. It is not modified by the patch.

### Unit collection

```bash
pytest --collect-only -q tests/unit
```

Result: **15,456 tests collected** in 8.90 seconds, exit code 0. This proves the patch introduces no unit-suite import or collection failure.

### Baseline-identical failures

The following exact nodes fail identically on the patched worktree and on an untouched detached worktree at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```text
tests/unit/api/test_facade_contracts.py::test_list_methods_return_iterable_on_empty_archive[list_archive_debt_insights]
tests/unit/api/test_facade_contracts.py::test_archive_tiers_api_archive_debt_reads_archive_consistency
```

Both fail with:

```text
sqlite3.OperationalError: database source_debt is locked
```

This node also fails identically on both trees:

```text
tests/unit/cli/test_plain_cli_snapshots.py::test_json_status_snapshot
```

Its stored snapshot expects schema/user version 37 while current source emits 38.

These failures were not waived based on assumption; they were rerun verbatim in the clean baseline worktree.

## Patch application verification

`PATCH.diff` was generated with full Git indexes and binary-safe formatting, then checked and applied to a fresh detached worktree:

```bash
git reset --hard 536a53efac0cbe4a2473ad379e4db49ef3fce74d
git clean -fd
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

All commands passed. The patch contains 73 `diff --git` sections, including the new `tests/unit/core/test_config_runtime_closure.py` file.

## Unverified operational checks

The following were not claimed or simulated:

- a live operator daemon and its existing archive;
- browser extension ingestion over a real network;
- real Voyage or Drive credentials/services;
- systemd/NixOS deployment behavior;
- operator secrets or home-directory configuration;
- a complete execution of all 15,456 unit tests.

The remaining certification value is primarily running the complete suite under the repository's intended CI/Nix environment and investigating the known baseline SQLite/snapshot defects separately.
