# EVIDENCE — authority, source, Bead, tests, and history

## Authority inspected

The implementation was derived from the supplied project-state archive, not from an assumed external checkout. The following snapshot artifacts were inspected first:

- `polylogue-overview.md` / `polylogue-overview.json`
- `polylogue-manifest.json`
- `polylogue-snapshot-audit.md` / `polylogue-snapshot-audit.json`
- `polylogue-branch-delta.md`, `.patch`, `.txt`, and log artifacts
- `polylogue-repo-tree.txt`
- `polylogue-all-refs.bundle`
- `polylogue-working-tree.tar.gz`
- `polylogue-beads-export.jsonl` and Beads summaries
- current-working-tree source/test XML slices where useful

Repository operating instructions inspected:

- `CLAUDE.md`
- `CONTRIBUTING.md`
- `TESTING.md`

Snapshot identity is `master` at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`, generated `2026-07-17T180950Z`, with `dirty=true`. The branch-delta artifacts contain no commit or patch delta and report `patch_bytes = 0`; the reconstructed tracked tree matched the named commit. The result therefore targets that commit and does not claim to capture ignored/runtime dirty state.

## Bead record inspected in full

Bead `polylogue-v1vo` is an open P2 bug created and updated on 2026-07-17. Its authoritative findings report, from a live archive analysis:

- 4,264 Polylogue-repository sessions;
- 27,785 file-touch occurrences;
- 43% of historical touches dead after file-to-package credit;
- mean per-session dead share 45%, p50 43%, p90 100%;
- 855 of 2,012 sessions majority-dead and 234 all-dead;
- concentrated examples including `storage/repository.py` (201 sessions), `lib/models.py` (166), and `pipeline/runner.py` (152);
- parent-directory prefix recovery of 59%, with usable evidence estimated to move from roughly 57% to 82%.

Its decided design requires query-time partitioning, file-to-package correction, parent-directory recovery, dead-path exclusion from Jaccard, optional explainability, unchanged weights initially, and no stored-profile rewrite.

The live database behind those numbers is not present in the supplied archive. This package treats the numbers as Bead authority and reproduces their proportions with a seeded equivalent; it does not label them as independently verified.

## Production source inspected

### Ranking and brief composition

`polylogue/insights/resume.py` was followed beyond the obvious score line. Before this patch:

- `_profile_paths` combined `file_paths_touched`, `repo_paths`, and `cwd_paths`;
- `find_resume_candidates` loaded all merged profiles, grouped logical families, optionally narrowed by repository evidence, and scored every remaining family before applying `limit`;
- file overlap was lexical `recent_files & all_paths` divided by `recent_files | all_paths`;
- file overlap contributed `0.25` to the final score;
- `ResumeCandidate.file_overlap` and `score_breakdown.file_overlap` were the only overlap explanation;
- `build_resume_brief` composed profile, events, phases, thread, lineage, and provenance but did not accept current-work context.

The patch preserves `_profile_paths`' current-source behavior rather than inventing a separate file-only profile contract.

### Facade and consumers

The complete relevant call graph was searched with `get_resume_brief`, `resume_brief`, and `find_resume_candidates`:

- `polylogue/api/archive.py`
- `polylogue/mcp/server_insight_tools.py`
- `polylogue/mcp/server_prompts.py`
- `polylogue/mcp/declarations/registry.py`
- `polylogue/cli/query_verbs.py`
- `polylogue/context/preamble.py`
- `polylogue/mcp/server_context_tools.py`
- `polylogue/surfaces/payloads.py`
- generated MCP equivalence/schema witnesses

`polylogue/cli/shared/resume_rendering.py` was also inspected. It defines a human `ResumeBrief` renderer but has no caller in the supplied source. The active CLI resume-ranking surface is `continue --candidates`, which directly consumes `find_resume_candidates`; that is the route patched and tested.

The SessionStart/context route calls the facade ranker through `build_context_preamble_payload`, so explanation must cross a surface-model boundary. The patch adds typed context models and converts through Pydantic serialization rather than importing the insight model into the surface layer.

## Tests and contract evidence inspected

Relevant existing tests and witnesses inspected include:

- `tests/unit/core/test_resume.py`
- `tests/unit/api/test_facade_contracts.py`
- `tests/unit/mcp/test_contract_evidence.py`
- `tests/unit/mcp/test_compose_context_preamble.py`
- `tests/unit/mcp/test_server_surfaces.py`
- `tests/unit/mcp/test_tool_declarations.py`
- `tests/unit/mcp/test_tool_discovery.py`
- `tests/unit/mcp/test_query_tool_schema_derivation.py`
- `tests/unit/cli/test_continue_absorption.py`
- `tests/unit/cli/test_cli_output_schemas.py`
- `tests/data/witnesses/mcp-tool-schemas.json`
- `docs/generated/mcp-equivalence.json`

The new tests use real facade/tool/CLI/context routes. They do not replace production dependencies with a test-only overlap function.

## Git history inspected

The all-refs bundle supplied the relevant implementation and refactor history:

- `e9a247ea981d5cf576bebe1219d97cb4c31c3222` — 2026-05-25 — `feat(insights): rank resume candidates (#1541)`; introduced the ranking behavior being repaired.
- `f4f7ef03a126bae3249b751faea0de216619f6a4` — 2026-05-17 — `feat(insights): resume brief insight with CLI/MCP/facade (#1129) (#1156)`; introduced the brief surface lineage.
- `fa56862b59442aa50c6d9044bb412ad595bb382c` — 2026-04-27 — `refactor(storage): group storage by ownership domain (#425) (#538)`; removes/moves `polylogue/storage/repository.py` into the package shape represented by current `polylogue/storage/repository/__init__.py`.
- `be1fc0054d7e4416dc88b9f926416b16fa35d1a6` — 2026-05-06 — `feat(daemon): replace run ingest with observable convergence (#847)`; removes `polylogue/pipeline/runner.py` while current pipeline work remains under the directory.

Those two verified history shapes are used in the `snapshot-derived` evaluator cohort. The fixture does not claim to contain sampled live session rows; it samples refactor paths from the supplied Git authority and supplies synthetic lineage/profile evidence around them.

## Implementation evidence

The production repair adds:

- a per-ranking-call path-resolution/existence cache;
- current-root validation and safe repo-local rebasing;
- captured-root and exact-suffix historical checkout mapping;
- file-to-package correction;
- resolvable/dead partition diagnostics;
- canonical exact matching;
- one-to-one deepest parent-directory prefix recovery;
- root-only-prefix rejection;
- dead exclusion from the Jaccard union;
- typed `ResumePathOverlap` and `ResumeOverlapBasis` models;
- optional current-work context on the brief facade/MCP route;
- typed context-preamble projection;
- complete JSON and concise terminal CLI projection;
- a versioned offline evaluator and fixture.

The repaired score retains all five existing ranking components and weights:

- recency `0.35`
- file overlap `0.25`
- cwd match `0.15`
- terminal state `0.15`
- workflow shape `0.10`

No storage writer, profile record, SQL schema, migration, or materialization pipeline is changed.

## Contradictions and source-wins decisions

### Dirty snapshot versus empty branch delta

The overview says `dirty=true`, but every provided branch-delta artifact is empty and the tracked worktree reconstruction matches the commit. The applyable authority is therefore the named commit. Ignored/runtime state is not copied into this result.

### Bead cost note versus current ranking architecture

The Bead says an existence check per candidate path should be bounded by candidate limit. Current source scores the entire filtered logical-session pool and only then slices to `limit`. The patch shares a cache and avoids duplicate canonical probes, but does not silently introduce a pre-ranking shortcut that could change ordering. The actual bound is unique paths in the scored pool, plus a package-correction check for missing Python files.

### Bead wording around `_profile_paths`

Current source includes repository and cwd evidence in `_profile_paths`, not only `file_paths_touched`. The patch preserves that behavior. Non-file/unmappable values become dead and are excluded rather than rewriting the profile or creating a new stored projection.

### Stale/unused CLI brief renderer

The source contains `polylogue/cli/shared/resume_rendering.py`, but repository search finds no caller. The live `continue --candidates` path consumes ranking candidates directly. The patch updates and tests the live path and records the unused renderer rather than wiring a duplicate framework.

### Existing MCP target-resource assertion

The supplied test expects eight resources while the supplied registry has nine. This inconsistency predates and is outside the patch. It is preserved and disclosed.

## Evaluation evidence and interpretation

The five ranking scenarios produce:

- overall hit@1: `0.20 -> 1.00`
- overall hit@3: `0.20 -> 1.00`
- overall MRR: `0.390 -> 1.000`
- synthetic hit@1: `0.333333 -> 1.00`
- snapshot-derived hit@1: `0.00 -> 1.00`

The all-live exact control remains first before and after. This is why no weight change is proposed.

The 1,000-path evidence sample is explicitly named `seeded-live-mass-equivalent` and validates the production partition itself: 570 resolve, 430 are dead, 254 dead paths recover by directory, and 176 remain excluded. It yields 57.0% before, 82.4% after, and 59.1% dead recovery.

These fixture metrics establish mechanism and regression sensitivity. They do not establish live precision, latency, or operator outcome quality; those remain for a corpus-backed second pass.
