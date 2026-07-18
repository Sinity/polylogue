# HANDOFF — polylogue-v1vo resume-candidate quality

## Mission completed

This package implements the scoring-time repair requested by bead `polylogue-v1vo`: historical paths that no longer resolve in the current checkout no longer poison exact-path Jaccard, refactor ancestors can recover overlap through directory continuity, and every recovery/exclusion is exposed as an additive `overlap_basis`. It also adds a durable offline ranking evaluator that compares the production legacy scorer with the production refactor-aware scorer over lineage-grounded synthetic and snapshot-history-derived scenarios.

No stored session profile is rewritten. No database or profile schema migration is introduced. The existing file-overlap weight remains `0.25`.

## Snapshot identity and patch base

The supplied Chisel snapshot declares:

- source: `/realm/project/polylogue`
- branch: `master`
- commit: `536a53efac0cbe4a2473ad379e4db49ef3fce74d`
- subject: `fix(repair): harden raw authority convergence (#3046)`
- generated: `2026-07-17T180950Z`
- snapshot flag: `dirty=true`

`polylogue-branch-delta.md` names `origin/master` with merge base `536a53efac0cbe4a2473ad379e4db49ef3fce74d` and contains neither commits nor a diff stat; `polylogue-snapshot-audit.json` reports `branch_delta.patch_bytes = 0`. I reconstructed the repository from `polylogue-all-refs.bundle`, checked out the named commit, and compared the archived tracked worktree. No recoverable tracked branch delta was present, so `PATCH.diff` is deliberately based on the named commit. The dirty flag may cover ignored/runtime/local material that is not an applyable tracked patch.

Input identities:

- `polylogue-all.tar(123).gz`: 128,314,788 bytes; SHA-256 `ec3fa193b2f99a6daee0bc620af4f69b5728834e99f8196792a17c3fbae11155`
- `perf-04-resume-quality(1).md`: 8,057 bytes; SHA-256 `d00852e5ea1e061871f7a25f3dc3446c3c03230e361db68483527eb29cf3f793`

## Fix mechanism

### Query-time path classification

`polylogue/insights/resume.py` now creates one `_PathResolutionContext` for each ranking call. It resolves the caller's repository root once and caches filesystem existence results by canonical local path across every logical candidate in that call.

For each logical candidate family, the scorer preserves the captured profile evidence and adapts only while scoring:

1. Relative candidate paths are rebased under the current repository root.
2. Historical absolute paths are mapped through captured absolute `repo_paths`; older profiles without usable roots can infer a historical checkout root only from a complete, exact repo-relative suffix.
3. Existing paths are classified as resolvable.
4. A missing `foo.py` is also considered resolvable when current `foo/__init__.py` exists, covering the measured file-to-package refactor shape.
5. Paths that cannot map into the current checkout or still do not resolve are classified as dead.

When a valid current repository root is unavailable, the scorer falls back to the previous lexical exact-Jaccard behavior rather than pretending it has filesystem evidence.

### Exact and directory recovery

Exact overlap is computed over canonical current-checkout identities, so relative and historical-absolute representations of the same current file can agree. File-to-package correction participates in this exact bucket while preserving the historical path in the explanation.

Dead paths are then compared with as-yet-unmatched caller files by parent-directory prefix. Matching is path-component-aware, deepest/nearest matches win, and both candidate and caller evidence are consumed at most once. A repository-root-only common prefix is rejected, preventing unrelated top-level areas from matching merely because they share a checkout.

### Corrected Jaccard denominator

The new overlap score is:

`matched exact-or-directory identities / (caller identities union resolvable candidate identities)`

Directory-recovered candidate evidence is represented by the matched caller identity. Still-dead candidate paths do not enter the union. Therefore two otherwise identical candidates receive the same score when only one carries extra unresolvable paths.

### Explainability

The additive model is:

```text
overlap_basis:
  exact: [{candidate_path, recent_file}, ...]
  dir: [{candidate_path, recent_file}, ...]
  dead_excluded: [candidate_path, ...]
```

`ResumeCandidate.file_overlap` remains backward-compatible and contains the matched current file paths. `score_breakdown.file_overlap` remains a float. The new detail is separate and additive.

## Consumer surfaces updated

The production call graph inspected and updated is:

- `polylogue/insights/resume.py`: production ranker, brief composition, typed explanation models.
- `polylogue/api/archive.py`: `Polylogue.resume_brief` accepts optional `repo_path` and `recent_files` and forwards them; `find_resume_candidates` retains its public signature.
- `polylogue/mcp/server_insight_tools.py`: candidate payloads naturally include the additive field; `get_resume_brief` accepts the same current-work context and emits the basis.
- `polylogue/cli/query_verbs.py`: `continue --candidates --format json` includes the complete basis; terminal output reports exact/directory/dead-excluded counts.
- `polylogue/context/preamble.py` and `polylogue/surfaces/payloads.py`: SessionStart/context preambles project the complete typed basis for each related session.
- `polylogue/mcp/server_prompts.py`: the resume recipe passes the same recent-file context to candidate ranking and the selected brief.
- `polylogue/mcp/declarations/registry.py`, `tests/data/witnesses/mcp-tool-schemas.json`, and `docs/generated/mcp-equivalence.json`: discovery/schema/generated witnesses reflect the additive brief inputs and explanation.

`polylogue/cli/shared/resume_rendering.py` was inspected. It currently has no production caller, so this patch does not create a parallel or dead rendering route merely to satisfy a stale shape.

## Ranking evaluation harness

`devtools/resume_ranking_eval.py` loads a strict versioned fixture, creates a temporary current checkout, materializes current files, constructs real `SessionProfileInsight` values, and calls `_rank_resume_profiles` twice:

- `overlap_mode="legacy"` reproduces the previous exact-path Jaccard.
- `overlap_mode="refactor-aware"` runs the repaired production scorer.

Ground truth is derived only from lineage: same logical family, true parent, or true sibling of the current-work fixture. The fixture contains no manual `relevant` or expected-target label.

Harness usage:

```bash
uv run --frozen python -m devtools.resume_ranking_eval
uv run --frozen python -m devtools.resume_ranking_eval --json
uv run --frozen python -m devtools.resume_ranking_eval --fixture path/to/fixture.json
```

Exit status is `0` for a non-regressing result, `1` for a metric regression, and `2` for invalid fixtures or runtime/evidence-partition errors.

### Measured fixture results

| Cohort | Scenarios | hit@1 before | hit@1 after | hit@3 before | hit@3 after | MRR before | MRR after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 5 | 20.0% | 100.0% | 20.0% | 100.0% | 0.390 | 1.000 |
| Synthetic | 3 | 33.3% | 100.0% | 33.3% | 100.0% | 0.483 | 1.000 |
| Snapshot-derived | 2 | 0.0% | 100.0% | 0.0% | 100.0% | 0.250 | 1.000 |

Scenario target ranks move as follows:

- synthetic all-dead shared directory: `4 -> 1`
- synthetic dead-union deflation: `5 -> 1`
- synthetic live exact control: `1 -> 1`
- snapshot-history `storage/repository.py -> storage/repository/__init__.py`: `4 -> 1`
- snapshot-history deleted `pipeline/runner.py` with current pipeline-directory work: `4 -> 1`

The seeded evidence-mass equivalent uses 1,000 paths: 570 resolvable, 254 directory-recoverable dead, and 176 still-dead. It reproduces `57.0% -> 82.4%` usable evidence and `254 / 430 = 59.1%` dead-path recovery.

These are offline fixture measurements. The bead's reported live-corpus counts—4,264 sessions and 27,785 file-touch occurrences—could not be rerun because the supplied snapshot contains source/history but no operator live archive database. They remain explicitly unverified.

## Decisions and invariants

- Stored evidence remains as captured; adaptation occurs at query time only.
- No schema migration or profile materializer rewrite is included.
- The `0.25` file-overlap weight is unchanged. The evaluator shows the repaired signal is not dominated, so there is no basis for retuning it in this patch.
- Directory recovery is one-to-one to keep overlap bounded by caller evidence and preserve a valid Jaccard range.
- Repository-root-only matches are forbidden.
- Filesystem failures or a nonexistent repository root degrade to legacy behavior.
- One resolution cache is shared by the ranking call. The current ranker scores the full filtered logical-session pool before applying `limit`; therefore existence work is bounded by unique candidate paths in that pool, not literally by the output limit. Changing that architecture would be a separate ranking/performance design.
- `RESUME_BRIEF_MATERIALIZER_VERSION` is bumped from 1 to 2 because the composed brief contract gains an optional field.

## Changed files

Production:

- `polylogue/insights/resume.py`
- `polylogue/api/archive.py`
- `polylogue/cli/query_verbs.py`
- `polylogue/context/preamble.py`
- `polylogue/mcp/declarations/registry.py`
- `polylogue/mcp/server_insight_tools.py`
- `polylogue/mcp/server_prompts.py`
- `polylogue/surfaces/payloads.py`

Evaluator and fixtures:

- `devtools/resume_ranking_eval.py`
- `tests/data/resume-ranking-eval-v1.json`

Generated/witness surfaces:

- `docs/generated/mcp-equivalence.json`
- `tests/data/witnesses/mcp-tool-schemas.json`

Tests:

- `tests/unit/core/test_resume.py`
- `tests/unit/devtools/test_resume_ranking_eval.py`
- `tests/unit/mcp/test_contract_evidence.py`
- `tests/unit/mcp/test_compose_context_preamble.py`
- `tests/unit/mcp/test_server_surfaces.py`
- `tests/unit/cli/test_continue_absorption.py`

Complete replacement files are unnecessary; `PATCH.diff` materially and unambiguously contains all changes.

## Acceptance matrix

| Acceptance requirement | Result | Evidence |
| --- | --- | --- |
| 100%-dead candidate with shared directory receives nonzero overlap | Pass | Production-route core test and synthetic evaluator scenario; target rank `4 -> 1`. |
| Extra still-dead paths do not deflate Jaccard | Pass | Two production candidates identical except six dead paths score equally; evaluator target rank `5 -> 1`. |
| Resolvable exact behavior is unchanged | Pass | Legacy/fixed exact control has identical overlap and remains rank 1. |
| File-to-package correction works | Pass | Historical `repository.py` resolves to current `repository/__init__.py`; snapshot-derived target rank `4 -> 1`. |
| Basis is exposed through resume candidates | Pass | MCP candidate JSON, CLI JSON/terminal, and context preamble tests. |
| Basis is exposed through `get_resume_brief` | Pass | Facade composes it from the logical family; MCP schema and payload contract tests pass. |
| Durable hit@1/hit@3/MRR evaluator | Pass | Versioned fixture, lineage-derived truth, human/JSON reports, regression exit code. |
| Usable evidence rises from roughly 57% to 80%+ | Pass on seeded equivalent | `57.0% -> 82.4%`; live corpus unavailable and unverified. |
| Stored profiles and weight remain unchanged | Pass | No storage/schema/materializer write changes; weight remains 0.25. |

## Apply order

1. Start from commit `536a53efac0cbe4a2473ad379e4db49ef3fce74d`.
2. Apply `PATCH.diff`:

   ```bash
   git apply --check PATCH.diff
   git apply PATCH.diff
   ```

3. Run the focused tests and evaluator documented in `TESTS.md`.
4. Run `uv run --frozen devtools render all --check` to verify generated witnesses.

There is no database migration, backfill, or stored-profile rewrite to schedule.

## Verification completed

- Patch applies cleanly to a detached clean worktree at the named commit.
- `git diff --check` passes both in the implementation worktree and after clean application.
- 153 final changed-surface tests pass in split subsystem runs.
- The facade read route test passes.
- Ruff format and lint pass for every changed Python file.
- Strict Mypy passes for all nine changed production/evaluator modules.
- Every generated surface passes `devtools render all --check`.
- The evaluator produces the metrics above and a valid JSON report.
- An expanded MCP/CLI contract run produced 205 passes and one pre-existing snapshot-baseline failure described below.

See `TESTS.md` for exact commands, anti-vacuity mutations, timings, and the full honest result record.

## Risks, limitations, and remaining verification

1. **Live corpus unverified.** No live archive database, daemon, browser, secrets, or deployed NixOS environment was supplied. The reported 4,264-session measurement is authority from the bead, not a rerun claim.
2. **Directory heuristic precision.** Parent-prefix recovery intentionally favors continuity through refactors, but same-directory work can be semantically unrelated. One-to-one matching, deepest-prefix preference, and root-only rejection bound that risk. A live labeled sample is needed to quantify false positives.
3. **Filesystem probe scope.** The resolution cache eliminates duplicate probes, but the existing ranker scores the full filtered pool before slicing to `limit`. A very large pool can therefore perform more probes than the bead's cost note implies. No ranking shortcut was introduced without evidence.
4. **Full repository gate incomplete.** `devtools verify --quick` reached full-repository Mypy but exceeded both 120-second and 300-second execution windows. Targeted strict Mypy and all generated checks pass; the entire repository test/type matrix was not completed.
5. **Pre-existing MCP test mismatch.** `tests/unit/mcp/test_tool_declarations.py::test_target_algebra_preserves_semantic_dimensions_without_self_authorizing` expects eight target resources while the supplied base source declares nine. The patch does not alter that assertion or the resource set. In the expanded contract run, 205 neighboring tests passed and this one failed.
6. **Generated JSON diff attribute.** The repository marks `docs/generated/mcp-equivalence.json` as non-text for ordinary diff display. `PATCH.diff` deliberately carries a textual applyable hunk, and clean-worktree `git apply --check` confirms it applies.

## Value of another iteration

Without a live archive snapshot, another local-only pass would be a **small repair pass**: additional edge-case fixtures, microbenchmarking the existence cache, or polishing explanation rendering. The present implementation already covers the stated production route and acceptance criteria.

With a read-only copy of the measured live corpus, a **substantial second pass** could add high value: rerun the 4,264-session/27,785-touch analysis, sample false-positive directory matches, measure ranking latency and probe volume, compare hit metrics on naturally occurring fork families, and then decide whether any weight retuning or more selective directory rule is justified. That evidence could materially change ranking policy; absent it, retuning would be speculative.
