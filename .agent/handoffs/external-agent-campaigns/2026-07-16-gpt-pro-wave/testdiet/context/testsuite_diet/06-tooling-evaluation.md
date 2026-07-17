---
created: 2026-07-16
purpose: Decide which third-party test tools to extend, pilot, or reject
status: design-ready
project: polylogue
---

# Tooling evaluation

## Default position

Polylogue does not need a new test framework. Prefer capabilities already in
the dependency graph and add a tool only when it removes handwritten test LOC
or exposes a missing defect class. A tool earns permanence through a bounded
pilot with actual findings, stable runtime, and a clear owner.

Before adding a new reporting tool, reuse the existing substrates:

- `devtools lab test-economics` already extracts package coverage, conventional
  fix-commit counts, testmon duration exposure, and selection fan-out;
- `devtools workspace failure-context` already joins a failing node to testmon
  dependencies, recent commits, fixtures, and saved witnesses;
- the verify runner already writes selections, durations, progress, peak RSS,
  containment, JSON/JUnit, and output artifacts;
- the isolated mutmut runner already creates machine-readable campaign
  receipts.

Use their raw extraction functions/artifacts for cluster dossiers. Do not build
a parallel dashboard. The current economics report's package-level quadrants
are coarse heuristics and must not authorize deletion; per-test context overlap
and obligation review are still required.

Assume PR #2932 and the completed `polylogue-b054.1.1.3`–`.5` proof work are
the harness baseline: complete seed/resource receipts, a real reversible
production mutation through the ordinary testmon gate, and repeated
isolated/xdist lifecycle receipts already exist. Diet tooling consumes those
artifacts. It does not add another selector, resource sampler, mutation wrapper,
or retry/repetition harness for the same obligations.

| Tool/capability | Decision | Concrete use | Guardrail |
| --- | --- | --- | --- |
| coverage.py + pytest-cov contexts | extend | periodic test→arc overlap/dominance audits | campaign mode, never semantic “coverage” claims |
| Hypothesis stateful/property testing | extend | query algebra, lifecycle model, metamorphic/fault sequences | independent model; retain explicit historical examples |
| mutmut | extend | changed-cluster sensitivity and deletion proof | survivor triage, not global score theater |
| SQLite progress handler | add directly | deterministic VM-step/work-growth laws | assert route-specific envelopes, not generic budgets |
| full pytest on tmpfs | promote after measurement | PR correctness baseline | repeat runs; track RSS, concurrency, and cleanup |
| pytest-xdist work stealing | benchmark | reduce long-tail worker imbalance | adopt only if same selection is faster and stable |
| diff-cover | optional pilot | changed-line coverage report | informational at first; not proof of correctness |
| Schemathesis | bounded pilot | stable OpenAPI daemon protocol + semantic custom check | reject if it only duplicates schema validation |
| Playwright | keep for rewrite | browser-level web-reader rewrite behavior | test user behavior, not JS source spelling |
| Atheris | keep focused | parser/security crash discovery | promote minimized failures to ordinary regressions |
| syrupy snapshots | use sparingly | intentional public render/terminal documents | no snapshots of private structures or incidental ordering |
| pytest-testmon | retain as proved accelerator | local affected edit loop and failure context, with `b054.1.1.4` anti-vacuity | never sole PR correctness signal; reuse its real mutation receipt |
| automatic flaky retries | reject | — | hides nondeterminism and corrupts run economics |
| new declarative coverage catalog | reject | — | declaration and verifier would share the same authority |

## Existing tools in more expressive modes

### Coverage contexts

Use `--cov-context=test` in a new *audit command*, not the default verify path.
Coverage.py can record dynamic execution contexts; pytest-cov labels them with
pytest node IDs and phases. The audit should emit a machine-readable relation:

```text
production file/arc ↔ test node ↔ phase ↔ elapsed time
```

From that relation, generate per-cluster overlap and unique-arc summaries. Do
not commit the full coverage database or a derived allowlist. Official docs:
[coverage contexts](https://coverage.readthedocs.io/en/7.12.0/contexts.html),
[pytest-cov contexts](https://pytest-cov.readthedocs.io/en/latest/contexts.html).

### Hypothesis

Reuse its example database and shrinking for structural generation, but set
profiles by workload purpose rather than using one global low CI example count
for every property. Cheap pure laws can run many examples; SQLite state
machines should use fewer, longer sequences; tail profiles should be explicit
examples. Stateful testing already provides rules, bundles, preconditions, and
invariants: [official guide](https://hypothesis.readthedocs.io/en/latest/stateful.html).

### Mutmut

The current isolated runner is useful. Simplify the campaign authority where
possible: derive targets from the owned production cluster/diff and save raw
receipts; avoid an authored YAML mirror claiming campaign freshness. Enable
covered-line mutation in focused audits to reduce wasted mutants. Mutmut
documents `mutate_only_covered_lines=true`: [official documentation](https://mutmut.readthedocs.io/en/latest/).

### xdist

The suite has highly uneven test durations and expensive session fixtures.
Benchmark the current distribution against `--dist worksteal`. The xdist docs
describe work stealing as useful when tests differ substantially in duration,
while `loadscope`/`loadgroup` can preserve fixture locality. Measure both wall
time and fixture rebuild count: [xdist distribution modes](https://pytest-xdist.readthedocs.io/en/stable/distribution.html).
This scheduling benchmark is distinct from `b054.1.1.5`; do not repeat its
optimize/WAL/embedding hang-witness campaign here.

## Bounded new-tool pilots

### Diff-cover

`diff-cover` consumes a coverage XML report and reports coverage on changed
lines. Use it to focus review and make uncovered changed branches visible. Do
not install a hard threshold until multiline/branch behavior and generated
files are understood, and never use the percentage as a substitute for a
sensitive oracle. [Project documentation](https://pypi.org/project/diff-cover/).

### Schemathesis

One pilot against the existing generated OpenAPI query routes is justified.
The acceptance test is practical: does it find protocol defects or delete a
substantial handwritten permutation cluster while staying cheap? Attach
planted-truth checks for successful searches; otherwise it is only schema and
crash testing. Stateful workflows need real OpenAPI links or explicitly
declared producer/consumer relationships, so do not assume they appear for
free. [Schemathesis](https://schemathesis.readthedocs.io/en/stable/),
[stateful guide](https://schemathesis.readthedocs.io/en/latest/guides/stateful-testing/).

## Tooling not worth inventing

- A Polylogue-specific property language. Python + Hypothesis already express
  the laws and have better shrinking/diagnostics.
- A generic surface parity framework. Keep a small runner over public protocols
  and independent facts; avoid adapter subclasses for every current surface.
- A generic performance-budget DSL. Assert measured work next to the route.
- A second affected-test selector. Make the full suite cheap enough to gate
  merges and leave testmon as an optimization.
- A test “quality score” per file. Coverage, duration, mocks, and mutation
  survivors are incomparable evidence; expose the raw dimensions in a dossier.
- A committed list of implementation spellings forbidden in tests. Remove the
  fossil and prove the behavior.
