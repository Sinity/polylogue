# Test design and execution evidence

## Oracle design

The expected-result authority is `tests/infra/query_manifest_oracle.py`. It is intentionally independent of all production query machinery. It imports only `dataclasses`, stores planted input facts, and computes membership/cardinality with ordinary Python iteration and slicing.

The test fixture converts those facts into existing typed parser models solely to plant them through `ArchiveStore.write_parsed`. The production writer is not consulted for expected output. The archive also contains the existing C-03 provider-native workload, so a dropped predicate can leak realistic irrelevant rows rather than passing against a tiny closed universe.

### Planted session controls

| Canonical session | Purpose |
| --- | --- |
| `codex-session:td01-action` | Repeated action ID, paired results, unmatched result shape, and a non-Bash control. |
| `codex-session:td01-action-decoy` | Same repeated tool ID in another session; detects dropped exact-session selection. |
| `codex-session:td01-delete-a` | Intended destructive target. |
| `codex-session:td01-delete-b` | Second intended target; detects first-row/page-only actions. |
| `claude-code-session:td01-delete-shadow` | Same title marker under another origin; detects dropped origin predicate. |
| `codex-session:td01-shared` | Exact canonical-ID target. |
| `claude-code-session:td01-shared` | Same native suffix under another origin; detects suffix-only identity. |
| `codex-session:td01-keep` | Post-delete retained control. |

### Planted action controls in `td01-action`

| Message | Tool ID | Result | Expected role in law |
| --- | --- | --- | --- |
| `m-action-01` | `td01-repeat` | success, exit 0 | First repeated pair. |
| `m-action-02` | `td01-repeat` | error, exit 7 | Second repeated pair. |
| `m-action-03` | `td01-missing` | none | Left-preserved unmatched use with null result fields. |
| `m-action-04` | `td01-read` | success | Non-Bash exclusion control. |

The expected Bash order is the fact/transcript order: first repeated pair, second repeated pair, unmatched use. The expected `is_error` partition is `0: 1`, `1: 1`, `unknown: 1`.

## Survivor laws

### `test_session_membership_count_and_canonical_identity_agree_across_public_routes`

Production dependencies exercised:

- `parse_expression_ast` and `compile_expression` in the Lark query substrate;
- compiled `SessionQuerySpec` list/count behavior through `Polylogue`;
- canonical identity lookup;
- root CLI bare-list dispatch, `ArchiveStore.list_summaries`, and JSON rendering.

Assertions:

- the explicit Boolean AST retains its predicate after lowering;
- facade list membership is exactly the two Codex cohort IDs;
- facade count equals two;
- exact `codex-session:td01-shared` does not admit the Claude session with the same native ID;
- root CLI items and total agree with the independent oracle;
- the receipt-backed C-03 target remains readable.

Representative mutation/removal: remove `boolean_predicate` from the filter map immediately before the CLI list call. The test fails with a 20-row unfiltered page instead of the two intended IDs. This proves the law depends on the repaired production forwarding edge.

### `test_action_rows_count_partition_and_pages_agree_across_repository_facade_and_cli`

Production dependencies exercised:

- terminal-unit expression parsing/lowering;
- `_action_relation_for_query` and canonical SQLite action relation;
- repository row and aggregate executors;
- `Polylogue.query_units` public envelopes;
- terminal-unit CLI JSON rendering.

Assertions:

- repository rows have exact session/message/tool/output/error/exit-code identities;
- repository count is three;
- two facade pages (`limit=2`) concatenate exactly to the unpaged stable order;
- each page equals the corresponding independent oracle slice;
- count and `group by is_error | count` partition the same three rows;
- CLI rows and total equal repository/facade/oracle results;
- the cross-session same-tool-ID decoy and non-Bash action do not leak.

Representative mutations: move the exact-session selection outside global ranking, drop the session predicate, change sort/page order, or replace rank pairing with equality pairing. Each changes at least one asserted identity, total, partition, or page boundary.

### `test_delete_preview_and_apply_remove_the_same_oracle_membership`

Production dependencies exercised:

- expression parsing/lowering;
- mutation cardinality resolution;
- CLI dry-run payload;
- `ArchiveStore.delete_sessions` apply path;
- post-action facade count and reads.

Assertions:

- preview exposes exactly two unique canonical IDs;
- preview reports zero affected rows and the complete intended target count;
- apply reports two sessions and two affected rows;
- the same expression counts zero afterward;
- every planted non-target remains, including the provider shadow and canonical-ID controls;
- the existing C-03 real-pipeline target remains.

Representative mutations: page-limit the preview/apply selection, drop the origin predicate during either phase, or resolve suffix-only IDs. Preview IDs, apply count, and post-state then disagree.

### `test_rank_pairing_survivor_rejects_plain_equality_join_mutation`

Production dependency exercised: the canonical action relation selected by `polylogue.storage.sqlite.archive_tiers.archive._action_relation_for_query`.

The mutation creates a temporary relation that left-joins tool uses to tool results only on `(session_id, tool_id)`. With two uses and two results sharing `td01-repeat`, that mutation emits a 2×2 product of four rows. The unmatched `td01-missing` use adds one, for a mutant total of five. The independent manifest and production rank-paired relation require exactly three. The test first proves the production baseline, then proves the explicit mutant's wrong cardinality.

## Commands and observed results

All semantic pytest runs below used `-p no:randomly` because the available host plugin failed during fixture setup with an invalid generated seed. The project files themselves were not changed to suppress that external plugin behavior.

### Patch-surface static checks

```bash
python -m ruff format --check \
  polylogue/cli/archive_query.py \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
```

Result: `3 files already formatted`.

```bash
python -m ruff check \
  polylogue/cli/archive_query.py \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
```

Result: `All checks passed!`.

```bash
python -m mypy --strict \
  polylogue/cli/archive_query.py \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
```

Result: `Success: no issues found in 3 source files`.

`python -m compileall` over the same files also completed successfully.

### Managed focused test

```bash
python -m devtools test \
  tests/unit/cli/test_query_composition_laws.py \
  -p no:randomly
```

Result: `4 passed, 8 warnings in 6.22s`; managed command reported `ok (20.0s)`. Warnings were Python 3.13 multiprocessing `fork()` deprecations from the workload artifact builder.

### Fresh detached apply check

`PATCH.diff` was checked and applied to a new detached worktree at `f654480cadb7cc4c194704e24dfd483199547b35`. The three applied files byte-matched the implementation worktree. `git diff --check`, changed-file Ruff format/check, `compileall`, and strict mypy passed there. The managed focused command was then rerun with an explicit disk basetemp because the container exposes only 64 MiB of `/dev/shm`:

```bash
POLYLOGUE_PYTEST_TMPFS=0 \
POLYLOGUE_PYTEST_BASETEMP_ROOT=/mnt/data/work-testdiet-01/pytest-tmp \
python -m devtools test \
  tests/unit/cli/test_query_composition_laws.py \
  -p no:randomly
```

Fresh-apply result: `4 passed, 8 warnings in 6.45s`; managed command reported `ok (19.3s)`.

### New laws plus nearest canaries

A focused selection covering the four new laws and nearest parser/cardinality/C-03 tests produced:

```text
12 passed, 16 warnings in 9.54s
```

### Complete owning files

```bash
python -m pytest -p no:randomly \
  tests/unit/cli/test_query_expression.py \
  tests/unit/cli/test_query_exec_laws.py \
  tests/unit/cli/test_query_composition_laws.py \
  tests/unit/cli/test_verb_cardinality.py \
  tests/unit/storage/test_archive_tiers_archive.py
```

Result: `583 passed, 1 skipped, 17 warnings in 25.01s`.

This run includes the existing C-03 tests:

- `test_exact_session_action_count_bounds_pairing_before_global_ranking`;
- `test_c03_exact_session_actions_uses_real_provider_pipeline_and_planted_facts`.

The first broad attempt exposed a compatibility issue in the draft: always passing `boolean_predicate=None` broke a narrow fake-backed test. The final implementation reuses the conditional typed filter map instead, after which the entire owning-file selection passed. The failed draft is not part of `PATCH.diff`.

### Query security, parity, identity, and diagnostics

```bash
python -m pytest -p no:randomly \
  tests/unit/storage/test_query_security.py \
  tests/unit/storage/test_query_parity.py \
  tests/unit/core/test_query_identity.py \
  tests/unit/archive/test_query_miss_diagnostics.py \
  tests/unit/archive/test_query_runtime_filters.py
```

Result: `62 passed in 13.07s`.

### Reversible production mutation

In a fresh detached worktree with the final patch applied, this line was temporarily inserted before the repaired list call:

```python
filter_kwargs.pop("boolean_predicate", None)
```

Running only the session membership law produced one required failure. The expected IDs were:

```text
codex-session:td01-delete-a
codex-session:td01-delete-b
```

The actual default page contained 20 IDs, including:

```text
claude-code-session:td01-delete-shadow
claude-code-session:td01-shared
codex-session:c03-irrelevant-002
codex-session:td01-action
codex-session:td01-action-decoy
codex-session:td01-keep
codex-session:td01-shared
```

The mutation worktree was isolated from the delivery patch. Removing the mutation returns the law to green.

### Default host-plugin attempt

Running the mutation-witness node without `-p no:randomly` did not reach the test body:

```text
ValueError: Seed must be between 0 and 2**32 - 1
```

This is recorded as an environment limitation, not as a product-test failure.

### Full quick verifier

`devtools verify --quick` did not complete in the available environment. Initial attempts lacked external `ruff`/`mypy` executables. With temporary wrappers, Ruff format/check completed, but the full-repository mypy stage was externally stopped after 240 seconds. There is no full-gate pass claim. The changed-file strict mypy command above is the completed type-check result.

## Remaining verification

The following remain explicitly unverified:

- exact `uv.lock`/Nix toolchain execution;
- full repository test suite and true testmon affected selection;
- daemon/HTTP/MCP/browser behavior and live process boundaries;
- concurrent-write snapshot stability, cursor replay, and cancellation;
- live-scale SQLite plans, VM-step/RSS ceilings beyond the retained C-03 canary;
- deployed archive, operator secrets, and production data.
