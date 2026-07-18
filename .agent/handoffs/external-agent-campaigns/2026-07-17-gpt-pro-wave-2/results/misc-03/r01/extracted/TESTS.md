# Verification and test design

## Test strategy

The test set is intentionally production-route based. It uses the canonical `messages_fts` table DDL, production block trigger DDL, exact audit/repair functions, daemon startup and convergence entry points, real status/metric emitters, and archive full-replacement/write-effect paths. No parallel fake ledger or alternate scheduler was introduced.

Each new deterministic test identifies the production dependency it exercises and the representative removal or mutation that should make it fail. The central anti-vacuity checks are:

- remove the legacy delete trigger: equal source/docsize counts still produce a ghost term that cites the replacement block;
- remove the production delete trigger: the retained identity row makes rowid reuse fail atomically;
- remove the rowid/block comparison: equal-count corruption is misclassified as ready;
- remove source or recipe comparison: the corresponding exact drift mutation becomes vacuously ready;
- remove a postings or ledger statement from production trigger DDL: insert/update/delete/empty transitions break the three-way invariant;
- update stale counters unconditionally: a corrupt baseline is made to look exact;
- add a `blocks`, docsize, or ledger scan to status/metrics: trace assertions fail;
- remove unique marker checks from the state machine: a rowid-only set proof could overlook stale posting terms.

## Named rowid-reuse regression

`test_count_only_rowid_reuse_can_cite_replacement_block` proves the defect against the count-only design. The sequence is:

1. insert block A at rowid 1 with term `alpha`;
2. drop the delete trigger;
3. delete A;
4. insert block B at reused rowid 1 with term `beta`;
5. verify source count and docsize count are both 1;
6. verify both `alpha` and `beta` join through rowid 1 and cite block B.

`test_identity_ledger_prevents_silent_rowid_rebinding` applies the same missing-delete-arm condition to production DDL. The ledger primary key rejects B’s insert, and SQLite rolls back the source row and all trigger side effects.

`test_exact_audit_detects_equal_count_rowid_reuse_and_repairs` constructs already-existing equal-count corruption, verifies one identity mismatch, repairs one unique affected rowid, removes the retired term, retains the current term, and advances repair generation.

## Hypothesis state machine

`tests/property/test_fts_identity_state_machine.py` uses `RuleBasedStateMachine` with production DDL. Rules cover:

- indexed and empty-text insert;
- text/content-hash update;
- delete;
- full-session replace;
- rowid-reuse-inducing delete plus insert;
- savepoint rollback;
- source identity drift followed by exact repair;
- recipe identity drift followed by exact repair.

After every completed rule, invariants compare canonical indexable rowids, `messages_fts_docsize.id`, and `messages_fts_identity.rowid`; validate block/source/recipe identity; validate current and retired unique marker terms; and require exact-ready durable counters. The configured run uses 30 examples and 45 stateful steps per example with no deadline.

## Static verification

Final changed-file checks in the implementation worktree:

```text
$ git diff --check
PASS

$ /opt/pyvenv/bin/ruff check <29 changed Python files>
All checks passed!

$ /opt/pyvenv/bin/ruff format --check <29 changed Python files>
29 files already formatted

$ /opt/pyvenv/bin/python -m py_compile <29 changed Python files>
PASS

$ PYTHONPATH=. /opt/pyvenv/bin/mypy --strict <16 modified production modules>
Success: no issues found in 16 source files

$ PYTHONPATH=. /opt/pyvenv/bin/python -m devtools lab policy schema-versioning
derived-tier upgrade helpers found: 0
invalid durable migration resources found: 0
undeclared index schema deltas found: 0
Schema evolution policy intact.
```

A prior standalone invocation of `devtools render all --check` completed with synchronized generated surfaces. A later combined static capture reached the broad render command and exceeded a 30-minute command ceiling; it did not modify generated files. No new production module was added.

`devtools verify --quick` completed whole-tree Ruff formatting and lint successfully, then exceeded the 20-minute command ceiling during whole-repository mypy. Its persisted run record shows:

```text
01-ruff-format: success, 0.84 s
02-ruff-check: success, 0.79 s
03-mypy: running when the outer command was terminated
```

The changed-production strict mypy command above completed successfully and is the authoritative type result for this patch.

## Behavior verification

The affected behavior selection was split into bounded groups after one combined run reached 89% with no failures but exceeded the command ceiling. Definitive group results were:

```text
Storage and property selection:             90 passed in 29.66s
Daemon/startup/status/convergence selection: 211 passed in 8.74s
Archive write effects and gateway:           17 passed in 0.29s
Trigger amplification benchmark tests:        2 passed in 464.78s
Total:                                      320 passed
```

The selection covered:

- the three new identity/property/observability files;
- exact and batched FTS repair SQL;
- FTS bloat and full-session replacement invariants;
- dangling derived surfaces;
- revision application exact proof;
- index lifecycle declarations;
- startup readiness, fallback behavior, health, status, metrics, global repair, convergence stages, and daemon CLI orchestration;
- archive write effects and write gateway;
- trigger amplification benchmarks.

## Independent patch-application verification

`PATCH.diff` was generated against the named base and applied to a detached checkout at commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`:

```text
$ git apply --check PATCH.diff
PASS

$ git apply PATCH.diff
PASS

$ git diff --check
PASS

$ compare SHA-256 for every changed file between source worktree and patched checkout
byte-identical changed files: 29
```

The detached checkout then ran this exact critical selection:

```bash
PYTHONPATH=. /opt/pyvenv/bin/pytest -q -p no:randomly \
  tests/unit/storage/test_fts_identity_ledger.py \
  tests/property/test_fts_identity_state_machine.py \
  tests/unit/daemon/test_fts_identity_observability.py \
  tests/unit/storage/test_fts_repair_sql.py \
  tests/unit/storage/test_fts_bloat_invariants.py \
  tests/unit/daemon/test_fts_readiness_fallback.py \
  tests/unit/daemon/test_convergence_stages.py \
  tests/unit/archive/test_write_effects.py \
  tests/unit/archive/test_write_gateway.py
```

Result:

```text
85 passed in 16.15s
```

The patched checkout also passed:

```text
ruff check: All checks passed!
ruff format --check: 29 files already formatted
schema-versioning policy: 0 upgrade helpers, 0 invalid durable migrations, 0 undeclared index deltas
```

## O(1) observability proof

`tests/unit/daemon/test_fts_identity_observability.py` installs a SQLite trace callback around the production metric and status functions. It rejects any statement containing:

- `FROM` or `JOIN blocks`;
- `FROM` or `JOIN messages_fts_docsize`;
- `FROM` or `JOIN messages_fts_identity`.

The tests then validate every drift gauge class and the source/index/identity, exactness, generation, and last-check values. This proves the normal observability seam reads only bounded durable state and schema metadata.

## Repair atomicity and failure behavior

The deterministic suite exercises:

- source/FTS/ledger trigger transitions for text-to-empty, empty-to-text, text-to-text, insert, delete, and rollback;
- exact rowid-set repair for missing, excess, identity, source, and recipe drift;
- global reset/rebuild of postings and ledger under one savepoint;
- per-window savepoints for batched missing/excess repair;
- full-session replacement with exact session IDs, including colon-containing neighboring IDs;
- startup bounded repair and convergence-debt deferral;
- repair-generation increment only after successful exact convergence;
- operational before/after sample recording and per-surface retention;
- missing-trigger restoration without trusting stale durable state.

## Synthetic write-amplification benchmark

Benchmark inputs and method:

- 10,000 indexed blocks per run;
- five repetitions;
- temporary SQLite databases;
- identical FTS5 table and `pl_fold` registration in baseline and ledger variants;
- baseline uses FTS-only block triggers;
- ledger variant uses production postings, identity, and freshness trigger DDL;
- insertion, text/content-hash update, and deletion measured separately;
- page size times page count measured after insertion.

Median results:

```text
Baseline insert:  0.1821874710 s
Ledger insert:    0.2634590040 s   ratio 1.4461
Baseline update:  0.3436288580 s
Ledger update:    0.4460324610 s   ratio 1.2980
Baseline delete:  0.0590124330 s
Ledger delete:    0.1242020550 s   ratio 2.1047
Baseline storage: 2,134,016 bytes
Ledger storage:   3,907,584 bytes  ratio 1.8311
Overhead:         177.3568 bytes per indexed block
```

The production ledger had 10,000 identity rows after insertion. Identity and docsize counts were both zero after deletion. This demonstrates bounded O(1) row work, but it is not a live-corpus performance claim.

## Unverified lanes

The following were not available and are not claimed:

- operator live archive or daemon;
- NixOS and full packaging lane;
- browser acquisition and private provider exports;
- full repository test suite;
- target filesystem/WAL performance on a representative archive;
- integration with code from the parallel `polylogue-wmsc` job, which was absent from the supplied snapshot.
