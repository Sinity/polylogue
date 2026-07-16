---
created: 2026-07-16
purpose: Explain and reproduce the broad implementation-coupling test census
status: active
project: polylogue
---

# Candidate census

The census and dossier commands have different jobs:

- the coupling and infrastructure scripts generate broad review populations;
- `dossier.py` joins exact cluster selectors to current execution evidence and
  reports readiness gaps.

Neither command makes a deletion decision or runs as a CI gate.

Run:

```bash
python .agent/scratch/testsuite_diet/census/test_coupling_census.py
```

Outputs:

- `test-coupling-census.json` — summary, per-test evidence, and SHA-256 of every
  input Python test file;
- `test-coupling-census.tsv` — compact sortable review population.
- `test-coupling-by-file.tsv` and `test-coupling-by-area.tsv` — hierarchical
  rollups for choosing an owning subsystem before inspecting node IDs.

Methodology v1 flags source/text/AST operations, literal membership in those
candidates, mock interaction inspection, collaborator patching within mock
candidates, and tests whose Python assertions are all coarse `hasattr`,
`isinstance`, nullness, or numeric-bound shapes.

Current result:

| Signal | Functions / LOC |
| --- | ---: |
| unique candidates | 1,318 functions |
| source/text/AST | 265 functions |
| source candidate + literal membership | 81 functions |
| mock interaction | 562 functions / 14,043 body-span lines |
| mock interaction + patching | 355 functions |
| coarse assertion shapes only | 503 functions |

Largest area populations are currently:

| Area | Candidates | Source/AST | Mock interactions | Mock body-span LOC |
| --- | ---: | ---: | ---: | ---: |
| `tests/unit/cli` | 253 | 31 | 170 | 4,200 |
| `tests/unit/daemon` | 226 | 20 | 181 | 3,733 |
| `tests/unit/core` | 136 | 5 | 5 | 273 |
| `tests/unit/sources` | 136 | 20 | 44 | 948 |
| `tests/unit/devtools` | 120 | 105 | 6 | 94 |
| `tests/unit/storage` | 108 | 45 | 8 | 408 |
| `tests/unit/mcp` | 98 | 6 | 84 | 2,400 |

These are review populations, not quality rankings. For example, daemon HTTP
mock interactions may represent meaningful network/security boundaries, while
`core` numeric-bound hits are often legitimate property invariants. The rollup
only makes the next area audit deliberate.

The strategy note's earlier 172 / 53 / 479 / 12.3k / 303 / 416 figures came
from an ad-hoc narrower query whose exact predicates were not saved. They are
historical sampling evidence, not a reproducible baseline. V1 is deliberately
broader; do not claim growth between incompatible methods.

False positives are expected. Review by owning subsystem and the criteria in
[`../test-suite-composition-and-scale-2026-07-16.md`](../test-suite-composition-and-scale-2026-07-16.md);
never turn this query into a CI denylist.

## Shared test-infrastructure consumers

Run:

```bash
python .agent/scratch/testsuite_diet/census/test_infra_usage.py
```

This emits `test-infra-consumers.tsv`, joining every `tests/infra` module to
static Python imports and `pytest_plugins` string loads. Current result: 46
modules, 4 with zero detected consumers, and 11 with one detected consumer.
Notable adjudicated leads are the unused 124-line lifecycle harness and the
136-line growth-budget helper consumed only by its own 62-line test.

Zero is not an automatic deletion verdict: indirect re-exports, subprocess
imports, or non-Python consumers still need an exact `rg`/runtime check. The
inventory exists to prevent agents from adding parallel helpers without first
seeing what is already present.

## Cluster dossiers

The selector document is [`clusters.json`](clusters.json). List and render its
five representative clusters with:

```bash
python .agent/scratch/testsuite_diet/census/dossier.py list
python .agent/scratch/testsuite_diet/census/dossier.py render --all
python .agent/scratch/testsuite_diet/census/dossier.py render --cluster exact-query-selection
```

Each render fingerprints exact route/scope files and composes, when available:

- current pytest selection, duration, progress, and summary receipts;
- testmon's per-test duration and production-file dependencies;
- per-test coverage contexts from `.cache/coverage/coverage.json`;
- coupling-census and `tests/infra` consumer rows;
- recent path history and dogfood-keyword commits;
- focused mutation and sensitivity artifacts.
- merged-source coordinator receipts for upstream Bead contracts named by a
  cluster.

Missing inputs remain explicit. In particular, the current coverage JSON is
absent and no first-cluster sensitivity receipts have landed, so the initial
dossiers are expected to say `prepared-not-execution-grade`. Do not edit that
label by hand: generate the missing evidence and rerun the command.

The seeded-artifact selector checks that the assumed realized
`polylogue-1xc.14.1` profile files exist in the merged checkout. Selectors also
name `1xc.14` and `b054.1.1.3`–`.5` contracts where their workload, resource,
testmon-mutation, or repetition evidence is required. `dossier.py` keeps such a
cluster blocked until Sol replaces the Bead-level prerequisite with exact
merged symbols and receipts. The Diet consumes those authorities without
editing or reconstructing them.

Generated JSON and Markdown live under `../dossiers/`. The JSON is suitable for
prompt preparation; the Markdown is the coordinator review view. Both are
disposable evidence and may be regenerated after an upstream merge, coverage
run, mutation witness, or cluster edit.

## Existing economics substrate

[`test-economics-observation-2026-07-16.md`](test-economics-observation-2026-07-16.md)
records a current run of `devtools lab test-economics --json`. The command is
worth reusing, but current package totals are dominated by hub imports and the
coverage JSON is absent. A fresh per-test-context coverage campaign is required
before using it for cluster-level overlap or savings estimates.
