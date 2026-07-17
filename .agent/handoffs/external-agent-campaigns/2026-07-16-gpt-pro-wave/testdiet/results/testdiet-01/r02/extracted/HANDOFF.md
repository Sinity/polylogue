# testdiet-01 handoff: query algebra and cardinality survivor

## Mission and delivered result

This package adds one coherent real-route survivor for Polylogue query algebra and relational cardinality. It starts from the realized C-03 seeded archive, layers a small independent provider-wire manifest onto a private clone, ingests those bytes through the production Codex acquisition/parse/materialize/index path, and exercises public expressions through parsing, lowering, SQL relation construction, repository reads, terminal-unit execution, root CLI reads, and the public `find ... then delete` preview/apply route.

The survivor establishes one fixed logical population of five selected actions in two sessions. Membership, ungrouped count, error partition, stable pagination, CLI payloads, destructive preview, destructive apply, and post-apply absence must all agree with that population. The planted shape includes duplicate tool IDs, paired successes and errors, a missing result, an orphan result, an unselected call, and an output-only decoy. A second test installs a deliberately broken same-session/same-tool-ID join and proves that the survivor rejects the resulting row multiplication: seven selected rows instead of five.

No production code was changed. The current snapshot's action relation already implements the required ordinal pairing and selective physical bound. Adding a second seam or query framework would have weakened the substrate-first architecture; the patch therefore consists only of a test-owned provider-wire oracle and real-route survivor tests.

## Snapshot identity

The attached project-state archive is authoritative for this draft.

- Project: `polylogue`
- Snapshot source recorded by manifest: `/realm/project/polylogue`
- Manifest generation time: `2026-07-17T105328Z`
- Branch: `master`
- Commit: `b9052e09103502017c0f510ecc699aac395de23c`
- Commit subject: `fix(daemon): bound raw maintenance admission (#2975)`
- Commit author/commit timestamp: `2026-07-17T08:28:35+02:00`
- Manifest dirty flag: `true`
- Branch delta: empty against `origin/master`; merge base is the same commit

I reconstructed a detached checkout from `polylogue-all-refs.bundle`, checked out the named commit, overlaid the exported working tree, and compared tracked content with the commit. No retained tracked-file delta was present. The snapshot still records `dirty=true`; the supplied artifacts do not preserve enough information to identify whether that flag came from ignored, untracked, or otherwise omitted local state. This package does not claim that the operator's original worktree was clean.

## Inspected evidence

Repository instructions and verification policy:

- `CLAUDE.md` / `AGENTS.md`
- `TESTING.md`
- `pyproject.toml`
- `devtools/verify_runs.py` and the managed pytest scratch policy

Production query and ingest route:

- `polylogue/sources/parsers/codex.py`
- `polylogue/pipeline/services/archive_ingest.py`
- `polylogue/archive/query/expression.py`
- `polylogue/archive/query/unit_results.py`
- `polylogue/storage/sqlite/action_relation.py`
- `polylogue/storage/sqlite/archive_tiers/index.py`
- `polylogue/storage/sqlite/archive_tiers/archive.py`
- `polylogue/cli/archive_query.py`
- `polylogue/cli/query_verbs.py`
- `polylogue/cli/verb_cardinality.py`
- `polylogue/surfaces/payloads.py`

Affected tests and helpers:

- `tests/infra/workload_artifacts.py`
- `tests/infra/query_cases.py`
- `tests/infra/surfaces.py`
- `tests/infra/semantic_facts.py`
- `tests/unit/storage/test_archive_tiers_archive.py`
- `tests/unit/sources/test_parsers_codex.py`
- `tests/unit/cli/test_query_expression.py`
- `tests/unit/cli/test_query_exec_laws.py`
- `tests/unit/cli/test_verb_cardinality.py`
- `tests/unit/storage/test_query_security.py`

Authority and intent records:

- `.beads/issues.jsonl`, especially `polylogue-1xc.14.1`, `polylogue-yeq.3`, and `polylogue-b054.1.1.4`
- Snapshot Beads export and branch-delta artifacts
- Test Diet dossier `dossiers/exact-query-selection.md`
- Test Diet area packet `areas/query-composition.md`
- Relevant Git history, including `c20286459`, `fb9073cfc`, `0eb2300bb`, `13d19ae36`, and `5d5edaf49`

`EVIDENCE.md` records the concrete findings and stale/current contradictions.

## Mechanism

### Independent expected fact set

`tests/infra/query_manifest_oracle.py` owns a fixed Codex JSONL micro-corpus. Its dataclasses are provider facts, not production query models. The helper writes native `session_meta`, `response_item/function_call`, and `response_item/function_call_output` records. It computes expected actions directly from those planted facts by pairing the nth call with the nth result sharing a call ID. It does not import Polylogue parsing, lowering, SQL, repository, or payload code.

The selected public predicate is:

```text
actions where session.origin:codex-session AND command:testdiet-cardinality-law
```

The independently expected relation is:

| Order | Session | Command suffix | Error state | Exit code |
| ---: | --- | --- | --- | ---: |
| 1 | `codex-session:testdiet-query-alpha` | `alpha-one` | success (`0`) | 0 |
| 2 | `codex-session:testdiet-query-alpha` | `alpha-two` | error (`1`) | 2 |
| 3 | `codex-session:testdiet-query-alpha` | `alpha-missing` | unknown | — |
| 4 | `codex-session:testdiet-query-beta` | `beta-success` | success (`0`) | 0 |
| 5 | `codex-session:testdiet-query-beta` | `beta-error` | error (`1`) | 3 |

Expected partition: `0 -> 2`, `1 -> 2`, `unknown -> 1`. Expected selected sessions: alpha and beta. The decoy session has the token only in result output, so command selection must not delete it.

### Production route exercised

1. Build the existing deterministic C-03 seeded archive using `build_seeded_archive`.
2. Clone it with `clone_seeded_archive`; no shared cached artifact is mutated.
3. Render three native Codex JSONL sources from the independent manifest.
4. Ingest them with production `parse_sources_archive` and `Source(name="codex", ...)`.
5. Parse the action terminal expression with `parse_unit_source_expression`.
6. Lower the corresponding session selector with `compile_expression`.
7. Execute action membership through `ArchiveStore.query_actions`, whose source is the canonical action relation SQL.
8. Execute rows, `count`, `group by is_error | count`, limit, and offset through `query_unit_rows` and the shared terminal executor.
9. Reparse and execute the same expression through root CLI `find --format json`.
10. Resolve the session expression through public `find ... then delete --dry-run --all`, then apply it with `--yes --all` against a private archive clone.
11. Verify exact preview/apply cardinality, deletion of the selected sessions, survival of the output-only decoy, and an empty post-delete public read.

### Anti-vacuity mutation

The mutation test replaces only the private test archive's `actions` view with the historical naive relation:

```sql
LEFT JOIN blocks r
  ON r.session_id = u.session_id
 AND r.tool_id = u.tool_id
 AND r.block_type = 'tool_result'
```

It deliberately omits ordinal `ROW_NUMBER()` pairing and `result_rank = use_rank`. In alpha, two tool uses and two results share one ID, so the naive join creates a 2x2 product: four rows where the intended relation has two. The complete selected population becomes seven instead of five. The test requires the real repository-membership survivor to raise and checks the precise `expected 5 ... got 7` diagnostic. This is a semantic behavior mutation; it does not inspect source spelling.

## Decisions

- Reused the realized C-03 artifact rather than inventing a Test Diet corpus or query framework.
- Kept the oracle test-owned and independent from production query semantics.
- Used native Codex wire bytes and production ingestion rather than inserting rows directly.
- Exercised one stable public read surface and one stable destructive action surface: root CLI `find` and `find ... then delete`.
- Kept all existing tests and helpers. No deletion or migration is part of this patch.
- Preserved parser diagnostic, quoting/escaping, security, compatibility, terminal-unit, and cardinality tests; the focused compatibility run is reported below.
- Did not add an HTTP or MCP law. The current Test Diet packet explicitly identifies MCP/web as rewrite boundaries, and the mission can be satisfied through stable CLI read/action routes without broadening the patch.
- Did not add a new work-bound measurement. The test starts from and reruns the existing C-03 exact-session work-bound canary; the new law concentrates on logical membership and cardinality composition.

## Changed files

- `tests/infra/query_manifest_oracle.py` — new independent provider-wire manifest, expected relation, membership, session, and partition oracle.
- `tests/unit/cli/test_query_composition_laws.py` — new real-route survivor and duplicate-ID join mutation witness.

`PATCH.diff` is the complete apply-ready change. `FILES/` is omitted because the unified diff fully and unambiguously represents both new files.

## Acceptance matrix

| Obligation | Evidence in patch | Status |
| --- | --- | --- |
| Independent oracle from planted inputs | Test-owned Codex dataclasses and nth-call/nth-result evaluator | Met |
| Production provider bytes and ingest | JSONL renderer -> `parse_sources_archive` | Met |
| Parse and lower public expressions | `parse_unit_source_expression`; `compile_expression` | Met |
| SQL/repository membership | `ArchiveStore.query_actions` against canonical action relation | Met |
| Count equals membership | `| count` returns exactly 5 | Met |
| Group partitions conserve population | `group by is_error | count`; sum is 5 | Met |
| Stable pagination | pages at offsets 0, 2, 4 concatenate to exact ordered population once | Met |
| Stable public read route | root CLI `find --format json` | Met |
| Preview/apply agreement | dry-run and apply select/delete exactly alpha and beta | Met |
| Decoy exclusion | output-only token session survives | Met |
| Duplicate/missing/orphan result shapes | all planted and asserted | Met |
| Row multiplication sensitivity | naive duplicate-ID join yields 7 and is rejected | Met |
| Existing C-03 selective pairing/work bound | existing two C-03 tests rerun green | Preserved |
| Parser diagnostics/security/compatibility | focused 541-test run includes the affected parser and security suites | Preserved |
| Full repository suite | not run | Unverified |
| `devtools verify --quick` | not run | Unverified |
| Live daemon/browser/NixOS/operator worktree | unavailable by contract | Unverified |

## Apply order

From a clean checkout of `b9052e09103502017c0f510ecc699aac395de23c`:

```bash
git apply --check PATCH.diff
git apply PATCH.diff
POLYLOGUE_PYTEST_TMPFS=0 .venv/bin/python -m devtools test \
  tests/unit/cli/test_query_composition_laws.py -x
```

Then run the broader compatibility command listed in `TESTS.md`, followed by the repository's ordinary full verification in the operator environment.

## Verification performed

- Managed project runner on the new module: **2 passed in 6.80s**; supervisor step **ok in 12.2s**.
- Focused affected compatibility set: **541 passed, 1 skipped in 26.19s**.
- Ruff lint: passed.
- Ruff format check: passed; two files already formatted.
- Mypy on both new files: passed, no issues.
- Patch apply check against a clean named-snapshot worktree: performed during package validation.
- Result ZIP member, placeholder, snapshot-copy, size, hash, and extraction validation: recorded in the final package validation section of the operator response and reproducible from the ZIP itself.

The first managed-run attempt used the default tmpfs policy and was refused before test collection because the container exposed only 64 MiB free in `/dev/shm`. Re-running with the repository-supported explicit `POLYLOGUE_PYTEST_TMPFS=0` storage choice passed. Raw pytest was also used for the 541-test compatibility run. `TESTS.md` preserves both the environmental refusal and successful commands.

## Risks and limitations

The mutation is an executable semantic mutation of the persisted production relation in an isolated archive clone, not a temporary edit of `polylogue/storage/sqlite/action_relation.py` followed by testmon affected-selection. That is sufficient to prove the survivor detects the representative relational defect requested here. A stricter testmon-selection proof belongs to `polylogue-b054.1.1.4` and would be a broader harness task.

The survivor proves stable paging by repeated public terminal execution with limit/offset, not a daemon continuation token. It proves one public read surface and one destructive action surface, not cross-surface CLI/Python/HTTP/MCP parity. It layers facts onto C-03 but does not add a new irrelevant-growth/SQLite-VM-step assertion. Existing parser and security suites were retained and run, but the full repository suite and `devtools verify --quick` remain unverified.

## Value of another iteration

A small repair pass would likely add little value: the patch applies cleanly, both survivor nodes pass, the representative mutation is killed, and the affected compatibility set is green.

A substantial second pass could add meaningful but separate scope: a reversible source mutation wired through the real testmon affected gate, an irrelevant-growth VM-step law, and HTTP/Python cross-surface parity using the same manifest. That would improve sensitivity and surface breadth, but it would no longer be a small repair; it would approach the open `polylogue-b054.1.1.4` and `polylogue-yeq.3` programs.
