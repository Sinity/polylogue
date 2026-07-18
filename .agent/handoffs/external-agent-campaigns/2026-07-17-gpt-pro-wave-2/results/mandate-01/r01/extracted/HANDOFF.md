# HANDOFF — Claude Workflow artifact admission through OriginSpec (polylogue-2qx.2)

## Mission and outcome

This package implements the mandate-critical `polylogue-2qx.2` slice against the supplied Polylogue authority snapshot. It extends the existing OriginSpec, source-tier retention, Claude parser, and generic work-evidence graph. It does not add a Workflow-only archive, a second registry, or a private inventory.

The production result admits all six required Claude Code artifact families through one contract:

1. coordinator session streams;
2. Workflow run-state snapshots;
3. mutable Workflow journals;
4. attempt transcripts;
5. paired attempt metadata sidecars; and
6. adopt manifests.

Configured acquisition and live watching retain the current artifact plus all raw revisions in `source.db`. Coordinator streams and attempt transcripts continue through the normal Claude session parser. Provider fact artifacts remain raw source authorities and are parsed during a rebuildable materialization pass. The materializer combines retained raw artifacts with already-indexed coordinator events, sessions, and prompt authorship, then atomically replaces only `claude-workflow:*` graphs in the existing generic work-evidence tables.

The exact synthetic `wf_54d4fb2e-841` corpus passes through the real configured-source route and proves the required counts, exclusions, provenance, missing-member behavior, and reparse implications.

## Snapshot identity and isolation

- Authority branch/ref: `master` and `origin/master`.
- Authority commit: `bf8191b3f56aa40da8f271df7f3385c712825497`.
- Parent commit: `4b574ce66533ac114961a9533ba2f2c4a0c45b83`.
- Snapshot bundle source: the supplied project-state archive, reconstructed from `polylogue-all-refs.bundle`.
- Snapshot generation metadata identified `/realm/project/polylogue` and generation time `2026-07-18T013442Z`.
- Supplied worktree dirt was limited to:
  - `polylogue/archive/query/unit_results.py`
  - `polylogue/daemon/http.py`
  - `polylogue/hooks/__init__.py`
- SHA-256 of that unrelated supplied dirty diff: `f30102453e3576c9049eb65dcea7265fa99ac694c45531b06b0a34dd85279e9f`.

Those three unrelated files are absent from `PATCH.diff`. The packaged patch was generated from a clean detached worktree at the authority commit and applied successfully to a second fresh detached worktree at the same commit.

## Evidence inspected before implementation

The following authority was read before changing code:

- Full Bead records for `polylogue-2qx.1.2` and `polylogue-2qx.2`, including later notes. The latest `2qx.2` note states that a prior claim was orphaned and resets the Bead to open because no corresponding master commit existed.
- Repository operating instructions in `CLAUDE.md`, `CONTRIBUTING.md`, and `TESTING.md`.
- Origin and admission implementation in `polylogue/sources/origin_specs.py`, artifact taxonomy, source dispatch, configured acquisition, direct archive ingestion, live watcher/batch processing, and source-tier raw storage.
- Claude Code parsing, history assembly, event extraction, prompt material-origin classification, and existing orchestration fact parsing.
- Generic work-evidence models, SQLite persistence/traversal, and convergence-stage architecture.
- Existing source-revision and Claude parser/history tests.
- Relevant history, especially `bce7336d3bb2e493080b37fd0bc76b429b0c1cbd` (`feat: harden archive continuity closure (#3051)`), which introduced the OriginSpec/work-evidence substrate that this patch extends.

The current source superseded stale planning assumptions: a prototype Claude Workflow projector existed, but it had no complete production materialization route, did not retain raw artifact ObjectRefs through generic work-evidence persistence, and was not sufficient to satisfy the reopened Bead.

## Production mechanism

### 1. OriginSpec is the artifact contract

`OriginArtifactRule` now declares path suffixes. The Claude Code OriginSpec declares all six artifact families and their parse policy. `artifact_suffixes_for_provider()` projects the live watcher suffix set from that same declaration, yielding `.json`, `.jsonl`, and `.ndjson` for Claude Code. Coordinator streams receive their own artifact kind rather than being conflated with attempt transcripts.

### 2. All intake paths retain authority

Configured direct archive ingestion explicitly walks Claude sources for OriginSpec-declared non-session facts and writes them to the existing raw source tier before normal session parsing. The configured asynchronous ingestion service runs the same Workflow materializer after acquisition/parsing and publishes diagnostics.

The live watcher derives accepted suffixes from OriginSpec. The live batch route recognizes non-session Workflow facts before session qualification, writes their bytes to the existing blob/source tier, treats a zero-session fact artifact as successful admission, and leaves graph projection to convergence. A growing journal is retained as a complete new raw revision rather than as a synthetic append-session fragment.

### 3. Provider-native fields retain source evidence

The Claude orchestration parser retains run/task identity, resume references, workflow/script identity, phases, labels, attempt and content keys, agent/session identifiers, status, phase/progress, model, timing, token/tool data, transcript and metadata paths, structured/final results, adopt fields, and errors. It retains the source path and journal line for each fact and exposes parse errors without fabricating facts.

Coordinator Workflow tool-use events continue through the regular Claude session parser. Direct coordinator prompts retain `human_authored`; generated worker prompts retain `generated_context_pack`.

### 4. Generic work-evidence accepts raw artifact evidence

`WorkEvidenceSourceRef` is now `EvidenceRef | ObjectRef`. Validation permits raw evidence only as `ObjectRef(kind="artifact", object_id="raw:<raw_id>")`; session/message evidence remains `EvidenceRef`. SQLite encoding and decoding round-trip both forms. This is a substrate extension used by the existing generic graph, not a Workflow-specific store.

### 5. Membership is evidence-linked, never topology-inferred

The Workflow projector has no parent/child topology input. Attempts are associated with sidecars and transcripts only through journal fields, sidecar fields, normalized transcript/meta paths, attempt IDs, and agent IDs. Missing or ambiguous counterparts produce unresolved or ambiguous claims backed by the evidence that demonstrates the gap. The 38 unrelated coordinator child sessions are indexed normally but never enter the Workflow graph.

### 6. Derived graphs are replaceable

`claude_workflow_materializer.py` reads current raw artifact pointers from `source.db`, repairs those pointers for direct/live raw writers when needed, reads retained blobs, reuses indexed session/event/authorship rows from `index.db`, computes a semantic corpus snapshot, and executes a single `BEGIN IMMEDIATE` replacement of the `claude-workflow:*` graph family. Raw revisions are never deleted by graph replacement.

The materializer is called by configured ingestion and is registered as a daemon convergence stage between embeddings and general insights. Freshness compares current semantic snapshot refs to stored graph refs.

## Key design decisions

- No new database tier or Workflow registry was introduced.
- No schema version bump was required because the existing generic work-evidence tables already store JSON evidence-ref arrays; only their typed interpretation was broadened.
- Fact artifacts remain raw authorities and are not forced into fake sessions.
- Coordinator and attempt streams continue using the established Claude session parser.
- Graph membership cannot use child-count or parent-session heuristics.
- Every graph node and edge requires at least one source ref at construction time.
- A final run-snapshot result is represented separately from journal result records.
- Rebuilds replace the derived Workflow graph family while retaining all source revisions and updating only current artifact pointers.
- Materialization failures are reported in configured-ingest diagnostics and daemon logs rather than corrupting the source-tier commit.

## Changed files

### Generated topology surfaces

- `docs/plans/topology-target.yaml`
- `docs/topology-status.md`

These were regenerated because the repository requires new production modules to be represented in its topology projection.

### Production source

- `polylogue/archive/artifact_taxonomy/models.py` — adds coordinator stream artifact kind.
- `polylogue/sources/origin_specs.py` — declares suffixes and all six Claude artifact rules.
- `polylogue/sources/live/watcher.py` — derives Claude suffixes from OriginSpec.
- `polylogue/sources/live/batch.py` — admits live fact artifacts and preserves full journal revisions.
- `polylogue/pipeline/services/archive_ingest.py` — retains configured non-session artifacts and materializes after direct ingestion.
- `polylogue/pipeline/services/parsing_workflow.py` — materializes after configured asynchronous ingestion and exposes diagnostics.
- `polylogue/sources/parsers/claude/orchestration.py` — expands provider fact parsing and coverage records.
- `polylogue/pipeline/payload_types.py` — permits Workflow diagnostics in typed observations.
- `polylogue/insights/work_evidence.py` — introduces typed raw-artifact evidence refs.
- `polylogue/storage/sqlite/queries/work_evidence.py` — decodes persisted EvidenceRef/ObjectRef evidence.
- `polylogue/insights/claude_workflow_evidence.py` — evidence-backed, topology-blind graph projection with explicit gaps.
- `polylogue/insights/claude_workflow_materializer.py` — source-to-derived production rebuild, freshness check, and quantified reparse plan.
- `polylogue/daemon/convergence_stages.py` — registers the Workflow materialization stage.

### Tests

- `tests/integration/test_claude_workflow_admission.py`
- `tests/unit/daemon/test_convergence_stages.py`
- `tests/unit/insights/test_claude_workflow_evidence.py`
- `tests/unit/insights/test_work_evidence.py`
- `tests/unit/sources/test_artifact_taxonomy.py`
- `tests/unit/sources/test_live_watcher.py`
- `tests/unit/sources/test_origin_specs.py`

Patch size: 22 files, 2,771 textual insertions, 243 textual deletions, plus two generated binary deltas. `PATCH.diff` SHA-256: `330fe4ee3d81a3779d3349e0cf28c7e61bf0839778f35b2062970e67652ac2ef`.

## Acceptance matrix

| `polylogue-2qx.2` acceptance criterion | Production route | Executable proof |
|---|---|---|
| 1. Acquire, inventory, revision, parse or policy-ignore all six families; gaps actionable | OriginSpec rules; configured raw admission; live watcher/batch; source inventory repair; orchestration parser | OriginSpec/taxonomy unit tests, live fact and live revision tests, full configured-source fixture |
| 2. Exact `wf_54d4fb2e-841` reconstruction | Configured source -> `source.db`/blob -> Claude parser/index -> generic graph materializer | 224 current artifacts; 1 run; 4 invocations; 50 calls; 91 attempts/transcripts/metas; 65 journal results; 49 completed keys; 1 unresolved key; 1 final result |
| 3. Provider fields and links carry raw provenance | Expanded fact parser; raw artifact ObjectRefs; message/session EvidenceRefs; claim nodes and evidence-backed edges | Integration asserts zero evidence-free nodes/edges and at least one persisted `artifact:raw:*` ref; generic SQLite round-trip test covers ObjectRef evidence |
| 4. Exclude 38 unrelated children | Projector has no topology input; association requires provider references | Fixture indexes 130 Claude sessions but links 91; SQL asserts no unrelated node/label; summary reports 38 exclusions |
| 5. Generated prompts are not human; direct prompt remains positive | Existing Claude material-origin classifier reused by materializer/projector | Fixture reports 91 generated prompts and 1 human prompt |
| 6. Missing members become degraded/unresolved | Missing snapshot/journal/meta/transcript/index bindings create evidence-backed gap claims; ambiguous bindings remain ambiguous | Unit gap tests plus integration mutation deleting one sidecar: sidecars 91 -> 90, linked sessions 91 -> 90, explicit missing-sidecar gap |
| 7. Quantified semantic reparse plan and focused tests | `claude_workflow_reparse_plan()`; semantic versioned snapshot; atomic graph-family replacement | Fixture proves 94 projector-only fact reads, 130 parser reads only for session/authorship/event semantic changes, 129 indexed bindings reused, 224 revisions preserved, 1 graph replaced, stale=false after rebuild |

The generic `polylogue-1vpm.6` adapter contract is exercised by the existing work-evidence model/SQLite traversal tests extended with raw artifact refs.

## Exact fixture result

Initial configured admission produces:

- 224 current artifacts and 224 retained raw revisions;
- artifact census: 1 run snapshot, 1 journal, 1 coordinator stream, 1 adopt manifest, 91 sidecars, and 129 attempt-stream artifacts (91 linked plus 38 unrelated);
- 130 indexed Claude sessions;
- one `claude-workflow:wf_54d4fb2e-841` graph;
- 1 run node, 4 invocation nodes, 50 call nodes, 91 attempt nodes, 91 linked session-segment nodes, 94 raw artifact nodes, and 66 structured-result nodes (65 journal records plus one final result);
- no evidence-free node or edge;
- no graph node containing an unrelated session identifier.

A second run-snapshot observation keeps 224 current artifacts, increases retained revisions to 225, advances the current raw pointer, changes the semantic corpus snapshot, atomically leaves exactly one current graph, and leaves no node with the old snapshot ref.

Deleting one retained sidecar then produces 90 sidecars, 90 linked sessions, and an explicit unresolved `missing paired agent metadata sidecar` claim. It does not infer the transcript through filename or child topology.

## Apply order

From a clean checkout of the named authority commit:

```bash
git checkout --detach bf8191b3f56aa40da8f271df7f3385c712825497
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
python -m devtools render topology-projection --check
python -m devtools render topology-status --check
```

Then run the focused tests listed in `TESTS.md`, followed by the repository-managed gate in a complete devshell:

```bash
devtools verify
```

## Verification completed

Verification was repeated from the freshly applied package patch, not only from the implementation worktree:

- `git apply --check PATCH.diff`: passed.
- `git apply PATCH.diff`: passed.
- `git diff --check`: passed.
- Python compilation of all 20 changed Python files: passed.
- Topology projection check: passed; generated projection has 1,035 rows and 9 pre-existing `TBD` ownership rows.
- Topology status check: passed.
- Exact configured-source mandate integration: 1 passed.
- Focused OriginSpec/live/materializer/generic-evidence/convergence tests: 15 passed.
- Existing Claude parser/history compatibility tests: 46 passed.
- Existing source-tier raw-write tests: 6 passed.

Detailed commands, dependencies, anti-vacuity mutations, and execution caveats are in `TESTS.md`.

## Limitations, risks, and remaining verification

The complete native repository gate remains unverified in this snapshot environment. The environment lacked the locked dev dependencies needed by normal Polylogue verification, including `aiosqlite`, `sqlite-vec`, `ijson`, `dateparser`, `tenacity`, `hypothesis`, pytest-timeout, pytest-xdist, Ruff, and mypy. Package-index DNS was unavailable, so `uv sync --extra dev --frozen` could not restore them. Focused tests used temporary import compatibility shims outside the repository and an empty current-version embeddings tier for paths that do not exercise embeddings. No shim is present in `PATCH.diff` or the ZIP.

A prior `devtools render all --check` attempt also encountered a pre-existing out-of-sync `docs/cli-reference.md` and later could not build the demo corpus without `sqlite-vec`; the specific topology surfaces affected by this patch pass their dedicated checks.

No operator live daemon, browser, private Claude archive, secrets, NixOS deployment, or current worktree was accessed. Validation uses only the supplied snapshot and synthetic data.

The exact one-run mandate is covered. The main design risk for a larger archive is scope: several summary counters are computed across the prepared Claude artifact set while graphs are emitted per run. A multi-run corpus should add per-run counter isolation and cross-run collision tests before those counters are treated as run-local operational metrics. The current rebuild also replaces the full `claude-workflow:*` graph family for any relevant semantic change. That is correct and bounded for the mandate corpus, but incremental multi-run performance has not been measured.

A small repair iteration would address ordinary native Ruff/mypy findings or a generated-surface discrepancy discovered by the complete devshell gate; expected value is limited because compile, apply, topology, focused behavior, parser compatibility, and raw-tier compatibility already pass. A substantial second pass would add multi-run fixtures, per-run coverage accounting, incremental invalidation/performance work, and operator-controlled validation against a non-private representative archive. That pass could materially improve scale confidence but is not required to prove the exact `2qx.2` fixture.
