# Evidence and authority record

## Authority order used

1. The mission prompt for job `testdiet-01`.
2. The supplied project-state archive and its current exported source.
3. Repository instructions and current typed interfaces.
4. Complete relevant Beads records, with later notes preferred over older descriptions.
5. Git history for intent and regression context.
6. The supplied Test Diet dossier as a stale planning aid, never as authority over current source.

No API, helper, query contract, or production seam was invented to satisfy the mission.

## Snapshot identity and integrity

`polylogue-manifest.json` records:

```text
project: polylogue
source: /realm/project/polylogue
generated_at: 2026-07-17T105328Z
branch: master
commit: b9052e09103502017c0f510ecc699aac395de23c
dirty: true
```

`polylogue-branch-delta.md` records `origin/master` as the base, the same commit as merge base, an empty diff stat, and no branch-only commits. The bundled Git commit is:

```text
b9052e09103502017c0f510ecc699aac395de23c
fix(daemon): bound raw maintenance admission (#2975)
2026-07-17T08:28:35+02:00
```

I reconstructed the repository from `polylogue-all-refs.bundle`, checked out that commit detached, overlaid `polylogue-working-tree.tar.gz`, and compared tracked files against the commit. The retained tracked tree was byte-clean. This does not negate the manifest's `dirty=true` flag: the archive does not preserve a recoverable tracked patch or enough metadata to identify omitted ignored/untracked state. The patch is therefore named against the commit and explicitly preserves this uncertainty.

## Repository architecture and instructions

`CLAUDE.md` / `AGENTS.md` establish that the substrate owns semantics and that new adapters or parallel frameworks should not duplicate query logic. `TESTING.md` identifies `devtools test` as the preferred managed entry point. The implementation follows those constraints:

- no new production query abstraction;
- no duplicate SQL evaluator;
- no direct row insertion used as a substitute for ingest;
- no source archive included in the deliverable;
- only test-owned facts and tests are added.

The managed runner's storage policy was inspected in `devtools/verify_runs.py`. It defaults to bounded `/dev/shm` and refuses a disk fallback when adaptive tmpfs capacity falls below 64 MiB. It also explicitly preserves `POLYLOGUE_PYTEST_TMPFS=0`, which was used for the successful managed run in this container.

## Current production route findings

### Provider ingestion

`polylogue/sources/parsers/codex.py` recognizes Codex JSONL `session_meta` and `response_item` records. Function calls become `tool_use` blocks; `function_call_output` records become `tool_result` blocks. Structured output metadata containing `exit_code` drives `tool_result_is_error` and `tool_result_exit_code`.

`polylogue/pipeline/services/archive_ingest.py` supplies `parse_sources_archive`, the real source acquisition/parse/materialize/index service used by the patch. The survivor writes native JSONL and invokes this service rather than calling the parser directly or seeding SQLite tables.

### Public expression parsing and lowering

`polylogue/archive/query/expression.py` provides the current public parser and lowerer:

- `parse_expression_ast`
- `parse_unit_source_expression`
- `compile_expression`
- terminal source and pipeline parsing
- structural `exists` lowering

The current grammar accepts action terminal sources, `where`, `group by`, `count`, `limit`, `offset`, sorting, and session-scoped predicates. It retains explicit errors for unsupported terminal shapes, unknown fields, malformed clauses, invalid numeric values, and ambiguous identities.

### Shared terminal execution

`polylogue/archive/query/unit_results.py` is the shared terminal executor. `query_unit_rows` applies pipeline limit/offset, obtains the query-unit descriptor, and dispatches to one registered terminal executor. `_execute_rows_terminal` calls the descriptor's SQL query method; `_execute_count_terminal` delegates one-field counts to `ArchiveStore.query_unit_counts`. The module states that this executor is shared across CLI find, MCP query units, daemon query units, and Python API.

The new law calls `query_unit_rows` directly for row, count, group, limit, and offset agreement and separately calls the public CLI route to prove reparsing and rendering.

### Canonical action relation

`polylogue/storage/sqlite/action_relation.py` is the source of truth for tool-use/result pairing. `action_relation_select_sql`:

- ranks tool uses by `(session_id, tool_id)` and message/block order;
- ranks tool results by the same key and order;
- pairs on session, tool ID, and `result_rank = use_rank`;
- retains uses with missing results through a left join;
- retains tool uses without an ID through a separate `UNION ALL` branch;
- can place the same exact-session bound on each physical branch.

`bounded_action_relation_cte` applies one selected session set to all three branches.

`polylogue/storage/sqlite/archive_tiers/archive.py` extracts exact owning-session bounds from predicates in `_exact_session_ids_from_predicate` and constructs the bounded relation in `_action_relation_for_query`. `query_actions` uses the canonical relation and stable occurrence/block ordering. `query_unit_counts` compiles the same structural predicate over the SQL-backed unit relation and performs exact grouped counts.

The anti-vacuity mutation removes the rank equality in a private archive view. This targets the concrete relational mechanism rather than a test helper.

### Public read and action routes

The root Click CLI is exported from `polylogue.cli`. Existing routing in `polylogue/cli/archive_query.py` sends terminal unit expressions to the shared query-unit executor and emits typed JSON envelopes. Existing destructive verbs in `polylogue/cli/query_verbs.py` and `polylogue/cli/verb_cardinality.py` resolve the full selected session set, enforce cardinality guards, and support dry-run/all/apply behavior.

The patch uses:

```text
find <action-expression>
find <session-expression> then delete --dry-run --all
find <session-expression> then delete --yes --all
```

against private archive clones. It does not call callbacks or mock the resolver.

## Existing workload canary findings

`tests/infra/workload_artifacts.py` defines deterministic provider-native archive artifacts and immutable cache/clone helpers. The default seeded archive includes the realized C-03 Codex tool-heavy workload: 64 sessions plus an exact target and provider-native call/result relationships.

`tests/unit/storage/test_archive_tiers_archive.py` contains two directly relevant current tests:

- `test_exact_session_action_count_bounds_pairing_before_global_ranking`
- `test_c03_exact_session_actions_uses_real_provider_pipeline_and_planted_facts`

They prove exact-session physical bounding and the generated provider pipeline. They do not independently prove the broad logical agreement of membership, counts, partitions, pages, and destructive preview/apply. The new manifest is layered onto this artifact rather than replacing it.

Both existing nodes passed in the focused run.

## Affected test and helper inspection

### `tests/unit/cli/test_query_expression.py`

At this snapshot: 6,494 lines, 22 classes, 328 test functions. The file covers materially distinct obligations, including:

- lexer and AST shape;
- boolean precedence and negation;
- quoted phrases and escaped quotes;
- exact session identity, native/canonical identity, and ambiguous-prefix errors;
- structural unit `exists` predicates;
- terminal unit parsing and execution;
- group/count/sort/limit/offset ordering;
- explicitly named unsupported pipeline shapes;
- FTS, vector, lineage, sequence, file, action, assertion, run, and context-snapshot paths;
- field registry completeness and examples;
- JSON-spec strictness;
- CLI/root request and cross-surface compatibility;
- malformed count, unknown field, invalid origin, and security-sensitive broadening rejections.

These obligations are not deleted or collapsed by the patch. The entire file passed in the focused run.

### `tests/unit/cli/test_query_exec_laws.py`

At this snapshot: 3,552 lines, 42 classes, 66 test functions. It covers archive-vs-daemon routing, structured and lexical reads, exact/native reference resolution, projection/output, count/group behavior, filters, pagination/cursors, semantic retrieval, mutation actions, delivery, and error paths. The file contains many local `FakeArchiveStore` definitions and patched collaborators, matching the Test Diet dossier's concern about forwarding-heavy tests. They remain present because this package does not include deletion certification.

The entire file passed in the focused run.

### Existing infrastructure

- `tests/infra/query_cases.py`: 54 lines; `ArchiveQueryCase` shadow case representation.
- `tests/infra/surfaces.py`: 616 lines; repository, facade, CLI, MCP, daemon, and SQLite adapters, including partial query translation.
- `tests/infra/semantic_facts.py`: 252 lines; `SessionFacts` and `ArchiveFacts` semantic expectations.

The new helper does not modify or replace these consumers. It is narrower: one independent provider-wire relation specifically for query algebra/cardinality survival.

### Security and destructive cardinality

`tests/unit/storage/test_query_security.py` was included in the compatibility run. `tests/unit/cli/test_verb_cardinality.py::TestDeleteCardinalityLargeNonMocked::test_guard_dry_run_and_deleted_sets_are_identical_and_unlimited` was also rerun. This preserves parameter binding and the historical full-set preview/apply contract that the new survivor exercises through the actual CLI.

## Beads findings

### `polylogue-1xc.14.1` — archive workload profiles

The record requires provider-native bytes, production acquire/parse/materialize/index/query routes, deterministic correlated shapes, and canaries for duplicate/missing/late/error tool-result pairing. Its design explicitly permits hand-authored fixtures for independent known-answer oracles while prohibiting a handwritten replacement for workload profiles. It also names C-03 as the first query canary and requires a global-first mutation to fail.

The patch follows this boundary: it reuses the realized C-03 workload artifact and adds only a fixed independent known-answer micro-corpus.

The issue status remains `in_progress`, and its notes describe an earlier incomplete slice. Current source at `b9052e0` already contains the C-03 artifact and relation changes from `c20286459`; source wins over stale status prose for what exists in this snapshot.

### `polylogue-yeq.3` — query laws and cross-surface parity

This open task defines the exact laws most relevant here:

- page concatenation enumerates each logical member once;
- grouped counts sum to the matching-grain population;
- identity and stable order persist;
- duplicate/missing/late result shapes are represented;
- broken predicate pushdown, continuation state, public type, and reference routes should fail.

The new survivor implements the membership/count/group/page subset and a broken relational-pairing witness. It does not claim completion of the broader cross-surface, cancellation, live-census, or resource-receipt program.

### `polylogue-b054.1.1.4` — real production mutation

This open task rejects source-spelling tests and asks for a semantic mutation that makes a dependent real-route test fail. It specifically concerns the ordinary testmon affected gate, dependency graph, isolated worktrees, and selection boundedness.

The new mutation is semantically real and exercises production SQL/repository behavior, but it does not execute the testmon affected-selection harness. That remaining distinction is stated as a limitation rather than blurred into a completion claim.

## Git history findings

Relevant history inspected:

- `c20286459` (`2026-07-17`) — `feat(schemas): derive archive workload profiles (#2934)`. Added `action_relation.py`, C-03 workload artifacts, exact-session action changes, and canary tests. This is the current structural foundation.
- `fb9073cfc` (`2026-06-27`) — `feat(query): add Terminal pipeline stage; route verbs through one executor (#2006) (#2463)`. Established the shared terminal execution route used by the survivor.
- `0eb2300bb` (`2026-06-14`) — `fix: delete full-set cardinality + Codex P2 read-surface findings (#1873) (#1876)`. Repaired destructive full-set selection and added cardinality guards; this history motivates explicit preview/apply equality.
- `13d19ae36` (`2026-07-13`) — `fix(actions): read legacy Codex commands without rewriting evidence (#2855)`. Reinforces that command extraction belongs in the existing action relation/query route.
- `5d5edaf49` (`2026-07-10`) — `fix: repair 3 flagship CLI query bugs found in prod smoke test (#2626)`. Added query execution regressions around public CLI behavior.
- `fd7b35492` (`2026-07-17`) — archive read admission/cancellation changes touching the action/query route.
- `805d49286` (`2026-07-17`) — raw authority replay changes touching storage route after C-03.

The current snapshot includes all of these ancestors. The patch is based on the exact named snapshot, not a newer bundled remote ref.

## Test Diet packet findings and contradictions

The supplied dossier is explicitly marked `prepared-not-execution-grade` and names stale head `21f78b4db2ba62ff44b5f16dfab96067bc249b4c`. It proposes exactly the two new paths used here:

```text
tests/infra/query_manifest_oracle.py
tests/unit/cli/test_query_composition_laws.py
```

It asks for independent planted facts, exact membership/count/partition/page/preview/apply laws, duplicate/join sensitivity, and retention of diagnostic/security branches. Those recommendations remain applicable.

Material stale/current contradictions:

1. The dossier says no realized sensitivity artifact exists and that C-03/upstream receipts are prerequisites. The current snapshot is later and contains the C-03 implementation from `c20286459`. This patch therefore starts from the realized canary as the mission requires.
2. The dossier names `polylogue/cli/query.py` as an authoritative route. Current source routes root query execution through `polylogue/cli/archive_query.py`, `polylogue/cli/query_verbs.py`, and the exported `polylogue.cli` group. The test uses current source.
3. The area packet suggests eventual repository, facade, CLI, HTTP, and rewritten MCP parity. It also marks MCP/web as rewrite boundaries. The mission asks for stable public read/action routes; this patch uses stable CLI read and delete routes and does not invent cross-surface adapters.
4. The dossier proposes multiple separate survivor nodes and an irrelevant-growth work law. This package chooses one internally coherent real-route behavior plus one mutation witness, preserving C-03's existing bound test. It does not claim the larger `polylogue-yeq.3` program.
5. The dossier identifies deletion candidates but explicitly requires dominance proof first. No deletion is performed.

## Why the selected fact shape is discriminating

- Duplicate call IDs detect many-to-many joins and lost ordinal pairing.
- Two results with different exit codes detect incorrect result association, not just row count.
- Missing result detects accidental inner joins and loss of unknown state.
- Orphan result detects result-first row manufacture.
- An unselected call in a selected session detects session-only selection.
- A decoy whose output contains the token detects broadening from command to output/global text.
- Two selected sessions make destructive preview/apply cardinality non-singleton.
- Five rows with a limit of two require three pages and expose early limit/global-first errors.
- The `0/1/unknown` partition detects null collapse and aggregate grain drift.
- Private archive clones make destructive and mutation tests isolated from the shared C-03 cache.

## Unverified external evidence

No claim is made about:

- the operator's live daemon or archive;
- browser behavior;
- secrets or authentication state;
- a NixOS deployment;
- the current operator worktree beyond the supplied snapshot;
- newer remote refs bundled after the named snapshot;
- full-suite or release-gate status.
