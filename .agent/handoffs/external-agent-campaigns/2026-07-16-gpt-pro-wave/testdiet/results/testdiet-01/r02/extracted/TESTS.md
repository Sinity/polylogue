# Test design and execution evidence

## Survivor design

The patch adds two tests in `tests/unit/cli/test_query_composition_laws.py` and one independent fact helper in `tests/infra/query_manifest_oracle.py`.

The helper is intentionally outside production query code. It describes native Codex records, renders JSONL, and derives expected action facts using only the planted calls and results. Expected values are never obtained from a Polylogue parser, compiler, SQL view, repository method, facade, or CLI response.

The micro-corpus is layered onto the existing deterministic C-03 seeded archive. This gives the test a real mixed archive and preserves the current exact-session action work canary while adding a discriminating known-answer relation.

## Planted relation

The public action expression is:

```text
actions where session.origin:codex-session AND command:testdiet-cardinality-law
```

The public session expression used by destructive actions is:

```text
exists action(session.origin:codex-session AND command:testdiet-cardinality-law)
```

The manifest plants seven tool uses overall. Five match the command predicate:

| Stable order | Canonical session | Call ID shape | Command | Result state |
| ---: | --- | --- | --- | --- |
| 1 | `codex-session:testdiet-query-alpha` | duplicate ID, ordinal 1 | `printf testdiet-cardinality-law-alpha-one` | output `alpha-one`, exit 0, `is_error=0` |
| 2 | `codex-session:testdiet-query-alpha` | duplicate ID, ordinal 2 | `printf testdiet-cardinality-law-alpha-two` | output `alpha-two`, exit 2, `is_error=1` |
| 3 | `codex-session:testdiet-query-alpha` | missing result | `printf testdiet-cardinality-law-alpha-missing` | output/error/exit unknown |
| 4 | `codex-session:testdiet-query-beta` | ordinary ID | `printf testdiet-cardinality-law-beta-success` | output `beta-success`, exit 0, `is_error=0` |
| 5 | `codex-session:testdiet-query-beta` | ordinary ID | `printf testdiet-cardinality-law-beta-error` | output `beta-error`, exit 3, `is_error=1` |

Two additional uses are deliberately outside the selected population:

- alpha has an ordinary unselected command and a paired result;
- the decoy session has an unselected command whose result output contains `testdiet-cardinality-law`.

An orphan result containing the token is also planted. It must not manufacture an action row.

Independent expected values:

```text
selected actions: 5
selected sessions: 2
is_error=0: 2
is_error=1: 2
is_error=unknown: 1
page offsets at limit 2: 0, 2, 4
```

## Test 1: real read and action routes

Node:

```text
tests/unit/cli/test_query_composition_laws.py::test_query_algebra_cardinality_survives_real_read_and_action_routes
```

### Production dependencies exercised

- `tests.infra.workload_artifacts.build_seeded_archive`
- `tests.infra.workload_artifacts.clone_seeded_archive`
- `polylogue.config.Source`
- `polylogue.pipeline.services.archive_ingest.parse_sources_archive`
- `polylogue.sources.parsers.codex` through source dispatch
- parsed-session materialization and SQLite indexing
- `polylogue.archive.query.expression.parse_unit_source_expression`
- `polylogue.archive.query.expression.compile_expression`
- `polylogue.storage.sqlite.action_relation.action_relation_select_sql` through `ArchiveStore.query_actions`
- `polylogue.storage.sqlite.archive_tiers.archive.ArchiveStore.query_actions`
- `ArchiveStore.query_unit_counts`
- `polylogue.archive.query.unit_results.query_unit_rows`
- terminal `rows` and `count` executors
- root Click CLI `find`
- public `find ... then delete --dry-run --all`
- public `find ... then delete --yes --all`

### Assertions

1. All three planted Codex source files ingest without parse failures and produce exactly alpha, beta, and decoy session IDs.
2. The action terminal expression parses as unit `action`.
3. The session selector lowers to a boolean predicate.
4. Repository action membership is byte-for-byte equivalent at the chosen public identity grain to the independent oracle.
5. `query_unit_rows` returns the same members and typed `ActionQueryRowPayload` rows.
6. `| count` returns one `all` group with count 5.
7. `group by is_error | count` returns exactly `0 -> 2`, `1 -> 2`, and `unknown -> 1`; the groups sum to 5.
8. Limit-2 pages at offsets 0, 2, and 4 concatenate to the exact stable unpaged order with no duplicate logical member.
9. Root CLI JSON `find` returns the same five identities and total 5.
10. On a private archive clone, delete dry-run previews exactly alpha and beta and reports zero affected rows.
11. Apply reports the same session count and exactly two affected sessions.
12. Alpha and beta are absent after apply; the output-only decoy session remains readable.
13. The original action expression returns the public typed empty result after deletion.

### Representative defects killed

The combined law fails if any of these semantic defects are introduced:

- the command predicate is dropped or broadened to output text;
- ordinal duplicate-ID result pairing is removed;
- an orphan result becomes an action;
- missing results are dropped instead of represented as unknown;
- count/group executes over a different grain than rows;
- limit/offset is applied before the effective selection;
- stable order changes between pages;
- CLI reparsing changes selection;
- delete preview and apply resolve different populations;
- the destructive action truncates to a default display page.

## Test 2: anti-vacuity mutation

Node:

```text
tests/unit/cli/test_query_composition_laws.py::test_survivor_detects_naive_duplicate_id_join_mutation
```

The test clones the prepared archive and replaces that clone's `actions` view with a deliberately wrong implementation. The mutation joins every tool result to every use sharing `(session_id, tool_id)` and omits ordinal ranks.

For alpha's duplicated call ID:

```text
2 uses x 2 results = 4 naive rows
2 ordinal pairs       = 2 correct rows
```

The other three selected uses remain one row each, so the wrong total is 7 rather than 5. The test invokes the same real `ArchiveStore.query_actions` membership helper and requires it to raise. It then verifies the exact failure signal:

```text
expected 5 independently planted action rows, got 7
```

This is a behavioral mutation of the real SQL relation in an isolated archive, not a source-text assertion. Removing the duplicate shape, consulting production output for expected values, or weakening membership to a mere non-empty assertion would make this mutation witness stop proving the intended law.

## Compatibility tests retained and inspected

No existing test or helper was deleted. The affected cluster was inspected broadly:

- `tests/unit/cli/test_query_expression.py`: 6,494 lines, 328 tests. It retains lexer/AST contracts, boolean precedence, quoting and escaped quotes, exact identity/ambiguity, terminal source and pipeline diagnostics, unsupported-shape diagnostics, field registry coverage, FTS/vector/lineage/sequence branches, JSON strictness, and CLI/daemon/MCP compatibility checks.
- `tests/unit/cli/test_query_exec_laws.py`: 3,552 lines, 66 tests. It retains daemon/local routing, structured and lexical paths, count/group forwarding, projection/output contracts, mutations, semantic search, cursor/list behavior, and error handling. Many cases use local fakes; they remain untouched.
- `tests/unit/storage/test_query_security.py`: parameter binding and query-security obligations.
- `tests/unit/cli/test_verb_cardinality.py`: full-set destructive guard, dry-run, and apply identity.
- `tests/unit/storage/test_archive_tiers_archive.py`: current exact-session action bound and C-03 provider-pipeline canary.
- `tests/unit/sources/test_parsers_codex.py`: structured function result exit-code/error mapping.
- `tests/infra/query_cases.py`, `tests/infra/surfaces.py`, and `tests/infra/semantic_facts.py`: existing cross-surface/shadow fact infrastructure retained for later certification.

## Commands and results

### Project-managed survivor run

Command:

```bash
POLYLOGUE_PYTEST_TMPFS=0 .venv/bin/python -m devtools test \
  tests/unit/cli/test_query_composition_laws.py -x
```

Result:

```text
collected 2 items
2 passed in 6.80s
supervisor: ok (12.2s)
```

The explicit environment variable is a supported repository policy choice that disables the managed tmpfs default and uses the configured disk scratch area.

### Focused compatibility run

Command:

```bash
.venv/bin/pytest -q \
  tests/unit/cli/test_query_composition_laws.py \
  tests/unit/storage/test_archive_tiers_archive.py::test_exact_session_action_count_bounds_pairing_before_global_ranking \
  tests/unit/storage/test_archive_tiers_archive.py::test_c03_exact_session_actions_uses_real_provider_pipeline_and_planted_facts \
  tests/unit/sources/test_parsers_codex.py::TestMessageParsing::test_function_call_output_captures_structured_exit_code \
  tests/unit/cli/test_query_expression.py \
  tests/unit/cli/test_query_exec_laws.py \
  tests/unit/storage/test_query_security.py \
  tests/unit/cli/test_verb_cardinality.py::TestDeleteCardinalityLargeNonMocked::test_guard_dry_run_and_deleted_sets_are_identical_and_unlimited
```

Result:

```text
541 passed, 1 skipped in 26.19s
```

The skip was pre-existing in the selected suite; no test in the new module skipped.

### Static checks

Commands:

```bash
.venv/bin/ruff check \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py

.venv/bin/ruff format --check \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py

.venv/bin/mypy \
  tests/infra/query_manifest_oracle.py \
  tests/unit/cli/test_query_composition_laws.py
```

Results:

```text
All checks passed!
2 files already formatted
Success: no issues found in 2 source files
```

### Environmental refusal preserved honestly

The first managed-run attempt used the default adaptive tmpfs policy. It stopped before pytest collection with status 125:

```text
only 64 MiB free in /dev/shm; refusing disk-backed pytest
```

No test failed in that attempt. The managed run passed after setting `POLYLOGUE_PYTEST_TMPFS=0`. During environment preparation, dependency synchronization rewrote registry URLs in `uv.lock`; that unrelated generated delta was reverted and is not present in `PATCH.diff`.

## Verification not performed

- Full repository test suite.
- `devtools verify --quick`.
- Coverage-context or testmon affected-selection certification.
- Daemon HTTP or MCP parity using the new manifest.
- A live operator archive, daemon, browser, deployment, secrets, or NixOS environment.
- New SQLite VM-step or wall-time bounds under irrelevant archive growth; the existing C-03 work-bound tests were rerun instead.

## Proposed dominated tests for later certification

No deletion is included. The following remain candidates only after local dominance evidence, coverage contexts, and semantic mutation receipts exist:

- mock-forwarding permutations in `tests/unit/cli/test_query_exec_laws.py` that are fully dominated by real route laws;
- repetitive one-field lowering examples in `tests/unit/cli/test_query_expression.py`, after a reviewed table retains every unique diagnostic, precedence, exact-ID, quoting, and security branch;
- shadow query fields in `tests/infra/query_cases.py` after all consumers use public expressions;
- the partial query translator in `tests/infra/surfaces.py` after a shared real-route manifest proves each stable surface;
- any synthetic fact duplication in `tests/infra/semantic_facts.py` that becomes redundant with provider-wire known-answer corpora.

Required certification before deletion:

1. Per-test coverage contexts for the candidate and survivor.
2. At least one semantic mutation killed by the survivor and formerly killed by the candidate.
3. A diagnostic/compatibility inventory proving no unique parser, security, serialization, cancellation, vector, lineage, or terminal-unit branch is lost.
4. Focused and full-suite green receipts in the operator environment.
