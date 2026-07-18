# TESTS — resume-candidate quality repair

## Test strategy

The tests exercise the production ranker and real consumer projections rather than a parallel matcher. The offline evaluator is also wired to the same `_rank_resume_profiles` and `_score_file_overlap` functions used by `find_resume_candidates`; its legacy mode exists only to establish the before baseline.

## Acceptance and anti-vacuity map

| Test | Production dependency exercised | Representative mutation/removal that must fail it |
| --- | --- | --- |
| `test_resume_candidates_recover_100_percent_dead_session_by_directory` | `find_resume_candidates -> _rank_resume_profiles -> _score_file_overlap -> _directory_overlap_pairs` | Return legacy overlap, remove directory matching, or keep every dead path unmatched. |
| `test_resume_candidates_do_not_recover_repo_root_only_directory_prefix` | Directory-prefix guard in `_directory_overlap_pairs` | Permit the repository root as the common prefix; unrelated root/`src` files would spuriously match. |
| `test_resume_candidates_exclude_unmatched_dead_paths_from_jaccard_union` | Candidate partition and corrected union | Reinsert `dead` identities into the union; the candidate carrying an extra ghost path scores lower. |
| `test_resume_candidates_preserve_resolvable_exact_jaccard` | Canonical exact path scoring | Change resolvable exact semantics or denominator; fixed and legacy controls diverge. |
| `test_resume_candidates_credit_file_to_package_correction` | Missing `.py` to package `__init__.py` resolution | Remove package correction; the historical repository path loses exact credit. |
| `test_resume_brief_projects_overlap_basis_for_current_work` | Facade `resume_brief`, logical-family profile load, scorer explanation | Stop forwarding current-work context or omit the brief basis. |
| `test_resume_ranking_eval_catches_legacy_dead_path_anti_selection` | Production legacy/fixed modes and five real ranking pools | Replace fixed mode with legacy mode; four repair scenarios retain ranks 4/5 instead of rank 1. |
| `test_resume_ranking_eval_reproduces_seeded_evidence_recovery` | Production partition and directory recovery over 1,000 paths | Remove existence filtering or directory recovery; 570/254/176 partition and 82.4% usable share fail. |
| `test_resume_ranking_fixture_has_no_manual_relevance_labels` | Lineage-ground-truth fixture contract | Add a hand-authored relevance/target label or omit required lineage. |
| `test_resume_ranking_metrics_assign_zero_reciprocal_rank_to_a_miss` | Standard MRR implementation | Treat a miss as rank `N+1` or any positive reciprocal value. |
| MCP resume brief/candidate contract tests | MCP tool signature, facade forwarding, JSON payload | Remove additive inputs/basis or project only counts. |
| Context preamble contract test | SessionStart production builder and typed surface payload | Omit `_candidate_overlap_basis` or the typed payload field. |
| CLI continuation tests | Live `continue --candidates` command | Remove basis from model dump or terminal projection. |
| Prompt surface test | MCP resume recipe | Stop forwarding the same current-work files to the brief. |

## Final focused execution

A single combined invocation of the following six files exceeded the command execution window after printing 47% progress and no failures. The identical set was rerun by subsystem so every test completed and the slow aggregate process did not obscure results.

### Core scorer and evaluator

```bash
uv run --frozen pytest -q -p no:randomly \
  tests/unit/core/test_resume.py \
  tests/unit/devtools/test_resume_ranking_eval.py
```

Result: `18 passed in 14.50s`.

### MCP contract evidence and SessionStart preamble

```bash
uv run --frozen pytest -q -p no:randomly \
  tests/unit/mcp/test_contract_evidence.py \
  tests/unit/mcp/test_compose_context_preamble.py
```

Result: `48 passed in 9.04s`.

### MCP surface and prompt contracts

```bash
uv run --frozen pytest -q -p no:randomly \
  tests/unit/mcp/test_server_surfaces.py
```

Result: `82 passed in 18.17s`.

### CLI continuation

```bash
uv run --frozen pytest -q -p no:randomly \
  tests/unit/cli/test_continue_absorption.py
```

Result: `5 passed in 1.14s`.

Final changed-surface total: **153 passed**.

### Facade route

```bash
uv run --frozen pytest -q -p no:randomly \
  tests/unit/api/test_facade_contracts.py::test_archive_tiers_api_threads_read_index_tier
```

Result: `1 passed in 1.07s`.

This test invokes both `archive.resume_brief(...)` and `archive.find_resume_candidates(...)` against the archive facade's real index-tier route.

## Ranking evaluator execution

```bash
uv run --frozen python -m devtools.resume_ranking_eval
uv run --frozen python -m devtools.resume_ranking_eval --json
```

Human report result:

```text
Resume ranking evaluation
Fixture v1: Offline resume-ranking fixtures with lineage-derived ground truth and refactor paths sampled from the 2026-07-17 Polylogue snapshot history.

Ranking quality (before -> after)
  overall          n=5  hit@1 20.0% -> 100.0%; hit@3 20.0% -> 100.0%; MRR 0.390 -> 1.000
  snapshot-derived n=2  hit@1 0.0% -> 100.0%; hit@3 0.0% -> 100.0%; MRR 0.250 -> 1.000
  synthetic        n=3  hit@1 33.3% -> 100.0%; hit@3 33.3% -> 100.0%; MRR 0.483 -> 1.000

Scenario ranks (before -> after)
  synthetic-dead-shared-directory                  4 -> 1
  synthetic-dead-union-deflation                   5 -> 1
  synthetic-live-exact-control                     1 -> 1
  snapshot-storage-repository-file-to-package      4 -> 1
  snapshot-pipeline-runner-directory-recovery      4 -> 1

Evidence usability
  seeded-live-mass-equivalent: 57.0% -> 82.4% usable; 59.1% of dead paths directory-recovered (254/430)

Verdict: non-regressing=true, strict-overall-improvement=true, all-fixed-targets-hit@1=true
```

The JSON report parsed successfully with fixture version `1` and all three verdict fields `true`.

## Formatting, lint, typing, and generated contracts

### Ruff

```bash
uv run --frozen ruff format --check <all 15 changed Python files>
uv run --frozen ruff check <all 15 changed Python files>
```

Result: `15 files already formatted`; `All checks passed!`.

### Strict targeted Mypy

```bash
uv run --frozen mypy \
  polylogue/api/archive.py \
  polylogue/cli/query_verbs.py \
  polylogue/context/preamble.py \
  polylogue/insights/resume.py \
  polylogue/mcp/declarations/registry.py \
  polylogue/mcp/server_insight_tools.py \
  polylogue/mcp/server_prompts.py \
  polylogue/surfaces/payloads.py \
  devtools/resume_ranking_eval.py
```

Result: `Success: no issues found in 9 source files`.

### Generated surfaces

```bash
uv run --frozen devtools render all --check
```

Result: all CLI reference, CLI output schema, OpenAPI, devtools reference, demo datasheet, quality reference, product workflows, docs surface, MCP equivalence, MCP tool index, topology status, and site-page checks reported sync/OK.

### Patch hygiene and clean application

```bash
git diff --check
git worktree add --detach <clean-worktree> 536a53efac0cbe4a2473ad379e4db49ef3fce74d
git -C <clean-worktree> apply --check PATCH.diff
git -C <clean-worktree> apply PATCH.diff
git -C <clean-worktree> diff --check
```

Result: all commands succeeded. The clean applied tree contains exactly the 18 intended changed/new files.

## Expanded contract run and known baseline failure

The broader command covered:

```bash
uv run --frozen pytest -q -p no:randomly \
  tests/unit/mcp/test_tool_declarations.py \
  tests/unit/mcp/test_tool_discovery.py \
  tests/unit/mcp/test_server_surfaces.py \
  tests/unit/mcp/test_query_tool_schema_derivation.py \
  tests/unit/cli/test_cli_output_schemas.py
```

Result: **205 passed, 1 failed**.

The failure is:

```text
tests/unit/mcp/test_tool_declarations.py::
  test_target_algebra_preserves_semantic_dimensions_without_self_authorizing
```

The supplied base test asserts `len(TARGET_RESOURCES) == 8`, while the supplied base registry declares nine resources (`session`, `message`, `block`, `action`, `file`, `query`, `result-set`, `recall-pack`, and `capabilities/query`). This patch changes neither that assertion nor the resource set; its only registry edit is the `get_resume_brief` description. It is recorded rather than masked by an unrelated deletion or expectation change.

## Incomplete broad gate

`uv run --frozen devtools verify --quick` was attempted with 120-second and 300-second execution windows. Ruff stages passed; the command remained in full-repository Mypy when each window expired. No orphan process remained. This is an incomplete broad verification, not a passing or failing full-repository Mypy claim. The nine changed production/evaluator files pass strict targeted Mypy, and `render all --check` passes independently.

## Verification not performed

The following were unavailable or outside the supplied authority:

- replay against the operator's live 4,264-session archive;
- live daemon, SessionStart hook process, browser, secrets, or MCP client integration;
- NixOS/deployment verification;
- full repository test matrix;
- performance measurement over the complete live candidate pool.

The offline harness, real-route unit/contract tests, clean patch application, and generated-surface checks are the completed local evidence.
