# TESTS — polylogue-4ts.3

## Test design and anti-vacuity

| Test or lane | Production dependency exercised | Representative mutation/removal that must fail it |
|---|---|---|
| `TestClaudeCodeAcompactClassification::test_acompact_classified_as_continuation_not_subagent` | Existing true-main `agent-acompact-*` parser behavior | Classify every acompact as sidechain or remove parent identity |
| `TestClaudeCodeAcompactClassification::test_fresh_task_prompt_head_classifies_acompact_as_sidechain` | `_is_fresh_task_prompt_head` and parser topology | Restore unconditional continuation classification |
| `test_acompact_resume_replayed_prefix_is_stored_once_and_composed[True/False]` | Parent-known and delayed-parent prefix extraction, edge persistence, composed read | Remove continuation extraction, late resolution, or parent composition |
| `test_subagent_self_compaction_stays_whole_and_never_composes_main_prefix[True/False]` | Parser structural signal, writer sidechain inheritance, composed reader | Compose every resolved edge, force continuation, or delete a child prefix |
| `test_ambiguous_acompact_is_reclassified_by_parent_content_membership[True/False]` | Writer's authoritative below-90% membership gate | Implement only filename/fresh-head logic; remove either parent-known or delayed comparison |
| `test_parent_membership_overrides_conservative_fresh_head_hint[True/False]` | Parent content authority over a false-positive parser hint | Never correct `sidechain` to `continuation` once the parser emits it |
| `test_claude_acompact_classifier_matches_eager_and_memory_bounded_routes` | Shared parser plus first-group fallback identity parity | Replace eager first fallback with `sessionId` or duplicate/fork classifier logic |
| Existing `test_subagent_child_arriving_before_parent_resolves_as_spawned_fresh` | General delayed spawned-fresh behavior outside acompact | Infer prefix-sharing solely from a resolved parent link |
| Delegation fact regression | Reclassified continuation child remains excluded from independent delegation work | Count compaction artifacts as delegated subagent work |

The Task fixture is privacy-safe synthetic JSONL. The exact composed text assertions ensure a test cannot pass merely because an edge label changed: the main transcript must be absent, child rows must remain physical, and role/material-origin semantics must remain unchanged.

## Commands and observed results

All commands ran from `/mnt/data/polylogue_repo` unless stated otherwise. The repository's existing `.venv` was used after one `uv run` attempt spent its execution window resolving registry/lock metadata before pytest started. That invocation produced no source-test failure; `uv.lock` was restored and is absent from the patch. A later verification script initially referenced a stale, nonexistent delegation-test path and stopped with pytest collection status 4; the production test was located in `tests/unit/pipeline/test_delegation_provider_fixtures.py`, rerun by its exact node ID below, and passed.

### Focused source, parser, dispatch, and real-route normalization

```bash
.venv/bin/pytest -q \
  tests/unit/sources/test_compaction.py \
  tests/unit/sources/test_claude_code_normalization_laws.py \
  tests/unit/sources/test_dispatch_payloads.py
```

Observed:

```text
56 passed in 2.84s
```

### Writer and lineage normalization

```bash
.venv/bin/pytest -q \
  tests/unit/storage/test_archive_tiers_write.py \
  tests/unit/storage/test_lineage_normalization.py
```

Observed:

```text
91 passed in 15.11s
```

### Additional Claude acquisition/history/dispatch routes

```bash
.venv/bin/pytest -q \
  tests/unit/sources/test_assembly_claude_code_history.py \
  tests/unit/sources/test_dispatch_ordering.py \
  tests/unit/sources/test_parsers_claude_code_artifacts.py
```

Observed:

```text
42 passed in 1.02s
```

### Delegation regression

```bash
.venv/bin/pytest -q \
  tests/unit/pipeline/test_delegation_provider_fixtures.py::TestDelegationIngestShapedFixtures::test_claude_code_auto_compaction_child_excluded_from_delegations
```

Observed:

```text
1 passed in 1.95s
```

The four groups are non-overlapping: 190 passing tests total.

### Ruff lint

```bash
.venv/bin/ruff check \
  polylogue/sources/dispatch.py \
  polylogue/sources/parsers/claude/code_parser.py \
  polylogue/storage/sqlite/archive_tiers/write.py \
  tests/unit/sources/test_compaction.py \
  tests/unit/sources/test_claude_code_normalization_laws.py \
  tests/unit/sources/test_dispatch_payloads.py
```

Observed:

```text
All checks passed!
```

### Ruff formatting

```bash
.venv/bin/ruff format --check \
  polylogue/sources/dispatch.py \
  polylogue/sources/parsers/claude/code_parser.py \
  polylogue/storage/sqlite/archive_tiers/write.py \
  tests/unit/sources/test_compaction.py \
  tests/unit/sources/test_claude_code_normalization_laws.py \
  tests/unit/sources/test_dispatch_payloads.py
```

Observed:

```text
6 files already formatted
```

### Mypy

```bash
.venv/bin/mypy \
  polylogue/sources/dispatch.py \
  polylogue/sources/parsers/claude/code_parser.py \
  polylogue/storage/sqlite/archive_tiers/write.py
```

Observed:

```text
Success: no issues found in 3 source files
```

### Patch whitespace/applicability

```bash
git diff --check 536a53efac0cbe4a2473ad379e4db49ef3fce74d
```

Observed: exit status 0, no output.

A detached pristine worktree was created at `536a53efac0cbe4a2473ad379e4db49ef3fce74d`. From that worktree:

```bash
git apply --check /mnt/data/lin-03-subagent-compaction-r01-work/PATCH.diff
git apply /mnt/data/lin-03-subagent-compaction-r01-work/PATCH.diff
git diff --check
```

Observed: all exit status 0, no output.

Representative tests against the applied worktree, using the original environment only for installed dependencies:

```bash
cd /mnt/data/polylogue_applycheck
PYTHONPATH=/mnt/data/polylogue_applycheck \
  /mnt/data/polylogue_repo/.venv/bin/pytest -q \
  tests/unit/sources/test_claude_code_normalization_laws.py \
  -k 'acompact_resume or parent_membership or self_compaction or ambiguous_acompact'
```

Observed:

```text
8 passed, 4 deselected in 2.46s
```

Python module inspection confirmed the imported `polylogue` package came from `/mnt/data/polylogue_applycheck`, not the modified source worktree.

### Final ZIP extraction and fresh-checkout application

A delivery archive candidate was reopened, its four members were CRC-tested and extracted, and the extracted `PATCH.diff` was applied to a second detached worktree at the named base commit:

```bash
git worktree add --detach /mnt/data/polylogue_zipcheck \
  536a53efac0cbe4a2473ad379e4db49ef3fce74d
cd /mnt/data/polylogue_zipcheck
git apply --check /mnt/data/lin-03-subagent-compaction-r01-validate/PATCH.diff
git apply /mnt/data/lin-03-subagent-compaction-r01-validate/PATCH.diff
git diff --check
PYTHONPATH=/mnt/data/polylogue_zipcheck \
  /mnt/data/polylogue_repo/.venv/bin/pytest -q \
  tests/unit/sources/test_compaction.py \
  tests/unit/sources/test_dispatch_payloads.py \
  tests/unit/sources/test_claude_code_normalization_laws.py \
  -k 'acompact or fresh_task_prompt_head'
```

Observed:

```text
8 passed, 48 deselected in 1.74s
```

Module inspection reported `/mnt/data/polylogue_zipcheck/polylogue/__init__.py`, confirming the tests loaded the freshly patched checkout.

## Verification not performed

- Full repository test suite: not run.
- Live source-to-index rebuild: not run; no operator archive was available.
- Before/after query against the measured ~187-file corpus: not run.
- Daemon/deployment smoke test: not run.
- Performance benchmark on multi-GiB JSONL: not run. The parser adds O(1) state; membership work occurs in the existing archive signature/extraction route and does not retain raw JSONL records.
