# TESTS — Claude Workflow OriginSpec admission

## Test strategy

The tests are designed around production dependencies rather than a parallel fixture-only framework. The highest-value test creates a complete synthetic Claude directory, invokes the real configured-source archive route, inspects `source.db` and `index.db`, triggers semantic rebuilds, and mutates retained authority to prove failure behavior.

The focused suite covers five boundaries:

1. OriginSpec classification and watcher suffix projection;
2. configured and live raw admission, including mutable journal revisions;
3. Claude orchestration parsing and material-origin preservation;
4. generic work-evidence persistence of raw artifact refs; and
5. atomic Workflow graph materialization and convergence registration.

## Execution environment

Verification ran under Python 3.13.5 from a fresh worktree after applying the packaged `PATCH.diff` to commit `bf8191b3f56aa40da8f271df7f3385c712825497`.

The supplied environment did not contain the full locked development dependency set and could not reach the package index. Focused runs therefore used temporary import shims located outside the repository for unavailable modules and a temporary pytest plugin that pre-created an empty, current-version embeddings database for tests that do not exercise vector search. Those support files are not part of the patch or ZIP. Warnings about missing pytest-timeout configuration and custom marks reflect the incomplete environment, not test failures.

## Commands and results

### Patch and syntax

```bash
git apply --check PATCH.diff
git apply PATCH.diff
git diff --check
```

Result: all passed.

All changed Python files were then compiled with `python -m py_compile`.

Result: 20 files compiled successfully.

### Generated topology surfaces

```bash
python -m devtools render topology-projection --check
python -m devtools render topology-status --check
```

Result: both passed from the freshly applied patch. The projection reports 1,035 rows and 9 existing `TBD` ownership rows.

### Exact configured-source mandate fixture

```bash
PYTHONPATH=/mnt/data:/mnt/data/mandate01-stubs:$PYTHONPATH pytest -q -o addopts='' --confcutdir=tests/integration   -p mandate01_pytest_plugin   tests/integration/test_claude_workflow_admission.py
```

Result: `1 passed, 2 warnings in 21.10s`.

Production dependencies exercised:

- configured `Source(name="claude-code", path=...)` walking;
- `parse_sources_archive()`;
- blob and `source.db` raw writes;
- artifact inventory/current pointers;
- regular Claude transcript parsing and session/event indexing;
- material-origin persistence;
- orchestration fact parsing;
- generic work-evidence SQLite writes;
- materializer freshness and atomic replacement.

Assertions include:

- zero parse failures;
- 224 current artifacts and 224 initial raw revisions;
- artifact census of 1 snapshot, 1 journal, 1 coordinator stream, 1 adopt manifest, 91 sidecars, and 129 transcript streams;
- 130 indexed Claude sessions;
- 1 run, 4 invocations, 50 calls, 91 attempts, 91 linked sessions, 91 sidecars, 65 journal results, 1 final result, 1 unresolved call, and 38 excluded sessions;
- 91 generated prompts and 1 positively human-authored direct prompt;
- zero evidence-free nodes and edges;
- persisted raw evidence beginning with `artifact:raw:`;
- zero unrelated Workflow nodes;
- quantified reparse plan: 94 projector-only raw reads, 130 session-parser reads only for changed session/authorship/event semantics, 129 indexed bindings reused, 224 revisions preserved, one graph replaced, and no staleness after rebuild;
- a second snapshot observation increases retained revisions to 225 while keeping 224 current artifacts and exactly one current graph;
- deleting one sidecar lowers sidecar and linked-session counts to 90 and creates an explicit missing-sidecar gap.

### Focused production/unit seam tests

```bash
PYTHONPATH=/mnt/data:/mnt/data/mandate01-stubs:$PYTHONPATH pytest -q -o addopts='' --confcutdir=tests/unit   -p mandate01_pytest_plugin -p mandate01_no_vec_plugin   tests/unit/insights/test_claude_workflow_evidence.py   tests/unit/insights/test_work_evidence.py   tests/unit/sources/test_origin_specs.py   tests/unit/sources/test_artifact_taxonomy.py   tests/unit/sources/test_live_watcher.py::test_claude_default_source_projects_originspec_suffixes   tests/unit/sources/test_live_watcher.py::test_live_full_ingest_admits_claude_originspec_fact_artifact   tests/unit/sources/test_live_watcher.py::test_live_full_ingest_preserves_complete_workflow_journal_revisions   tests/unit/daemon/test_convergence_stages.py::test_default_convergence_stages_always_register_embed_stage   tests/unit/daemon/test_convergence_stages.py::test_embed_stage_is_noop_when_disabled
```

Result: `15 passed, 4 warnings in 8.21s`.

Production dependencies exercised:

- exact Claude OriginSpec artifact rules and suffix projection;
- artifact taxonomy classification;
- live raw fact admission with zero parsed sessions;
- complete journal revision retention and current-pointer advancement;
- projector evidence requirements and unresolved/ambiguous states;
- generic model and SQLite traversal round-trip for raw artifact ObjectRefs;
- default convergence-stage ordering.

### Existing Claude parser/history compatibility

```bash
PYTHONPATH=/mnt/data:/mnt/data/mandate01-stubs:$PYTHONPATH pytest -q -o addopts='' --confcutdir=tests/unit   -p mandate01_pytest_plugin -p mandate01_no_vec_plugin   tests/unit/sources/test_parsers_claude_code_artifacts.py   tests/unit/sources/test_assembly_claude_code_history.py   tests/unit/sources/test_parsers_claude_history.py
```

Result: `46 passed, 2 warnings in 1.97s`.

This checks that coordinator-event and authorship changes do not regress ordinary Claude Code parsing or history assembly.

### Existing source-tier revision compatibility

```bash
PYTHONPATH=/mnt/data:/mnt/data/mandate01-stubs:$PYTHONPATH pytest -q -o addopts='' --confcutdir=tests/unit   -p mandate01_pytest_plugin -p mandate01_no_vec_plugin   tests/unit/storage/test_archive_tiers_source_write.py
```

Result: `6 passed, 2 warnings in 0.89s`.

This checks the existing source write/revision substrate used by the new Workflow inventory.

## Anti-vacuity mutation matrix

| Representative implementation mutation/removal | Test that must fail | Why the failure is meaningful |
|---|---|---|
| Remove a Claude artifact rule or its suffix | OriginSpec/taxonomy tests and 224-artifact integration census | The production source no longer discovers/adopts the full provider family |
| Restore JSONL-only watcher suffixes | `test_claude_default_source_projects_originspec_suffixes` and live snapshot admission | Run snapshots, sidecars, and adopts disappear from live admission |
| Remove the live non-session fact branch | `test_live_full_ingest_admits_claude_originspec_fact_artifact` | A valid raw fact is falsely marked failed because it yields no session |
| Treat a growing journal as a session append fragment | `test_live_full_ingest_preserves_complete_workflow_journal_revisions` | The second full raw revision or its current pointer disappears |
| Remove current-artifact pointer repair/ranking | Integration revision mutation | `raw_artifacts.raw_id` no longer advances to the newest 225th revision |
| Infer Workflow membership from parent/child topology | Integration unrelated-session assertion | The 38 intentionally unrelated children enter the graph |
| Infer transcript pairing without a sidecar | Integration sidecar-deletion mutation | Linked sessions remain 91 instead of degrading to 90 with a gap |
| Drop raw ObjectRef support in generic work-evidence | Work-evidence round-trip and integration evidence assertions | Raw provider facts lose durable provenance or fail persistence |
| Allow empty evidence refs | Projector unit tests and SQL checks | Nodes/edges can be created without source support |
| Collapse `generated_context_pack` into human authorship | Integration authorship counts | 91 generated prompts become falsely human-authored |
| Remove coordinator event extraction | Integration invocation count | Four provider-reported invocations disappear |
| Stop atomically replacing the graph family | Integration second-observation mutation | Stale snapshot rows or duplicate Workflow graphs remain |
| Fabricate a completed result for unresolved key `call-49` | Integration unresolved count | The required explicit unresolved call disappears |

## Failure mutations already executed

The integration test executes two source mutations rather than merely describing them:

1. It writes a second raw revision of the authoritative run snapshot. The expected outcome is 225 retained revisions, 224 current artifacts, one new semantic snapshot, one current graph, and no stale node snapshot refs.
2. It deletes one metadata sidecar from retained source authority. The expected outcome is 90 sidecars, 90 linked sessions, and an explicit unresolved missing-sidecar claim.

These mutations would fail if the implementation used topology, filename wishful pairing, destructive source replacement, or append-only derived graphs.

## Unverified native gates

The following remain unverified because the snapshot environment could not install or import the complete locked toolchain:

```bash
uv sync --extra dev --frozen
devtools verify
devtools verify --quick
devtools verify --all
nix flake check
```

Consequently native Ruff formatting/lint, strict mypy, pytest-testmon affected selection, full non-integration pytest, and full Nix/CI parity are not claimed. A `devtools render all --check` attempt was also blocked by a pre-existing stale `docs/cli-reference.md` and later by unavailable `sqlite-vec`; the two topology checks changed by this patch pass independently.
