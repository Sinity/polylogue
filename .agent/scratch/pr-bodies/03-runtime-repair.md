## Summary

PR 3/9. Stacked on `feature/fix/stack-02-cli-contracts`.

Runtime semantic repair wave triggered by real-archive rebuilds against the default XDG root. Schema baseline reset, session-product freshness, repo attribution tightening, and the start of the memory/perf wave.

## Problem

Live rebuilds surfaced a tangle of product-correctness defects that had accumulated on the real archive:

- The archive schema had drifted to a version that failed to bootstrap cleanly; live rebuilds couldn't proceed against a default XDG root without a manual reset.
- Session-profile rows carried polluted payloads (inferred repo names like `projects`, `root`, `blob-repository`) sourced from transcript-store paths, temp task outputs, snapshot paths, `.config/claude/projects`, `.codex/`, and Nix store paths.
- Repo attribution was willing to synthesize repo and file attribution from arbitrary dialogue text paths, contaminating rollups on `polylogue stats --by repo`.
- `session_product_status` knew rows were incomplete, yet reads served them anyway; repair was silent while running, making the command look hung.
- The repo-identity helper probed `(candidate / ".git").exists()` on arbitrary archived absolute paths and raised `PermissionError` on unreadable admin paths like `/boot/.git`.
- `polylogue doctor --json` took 40+ seconds on large archives because exact `messages_fts_docsize` counting and exact orphan counts ran on every default call.
- `doctor --runtime --json` still executed the full archive-health path and timed out.
- `polylogue --plain --format json project --limit 3` returned `no_results` silently on archives where `messages_fts` existed but was not populated — the search backend swallowed exceptions and degraded without surfacing the incomplete index.
- `update_index_for_conversations()` filtered changed conversation ids through `action_event_repair_candidates_sync()`, so content-block mutations with unchanged materializer version were skipped and stale `action_events_fts` rows persisted.
- Archive stats attachment counts confused `attachment_refs` with distinct attachments in the structured payload, leaked unrelated archive totals under an `embeddings` key.
- Early perf regressions: latest-query memory on large archives, sqlite read-path memory pressure, archive stats and retrieval-band status running together.

## Solution

- Reset the archive schema baseline to `v1` with legacy inline-raw detection; harden live archive reads and overwrites; align live provider ingestion with archive payloads.
- Invalidate stale session-product rows on materializer version bump; expose repair progress; reject incomplete session-product surfaces before repair with an actionable error envelope (`session_profiles` / enrichments / work events / phases / threads / tag rollups / day / week summaries).
- Normalize repo attribution: promote `provider_meta["working_directories"]` to first-class evidence; reject dialogue-derived repo/file guessing; filter transcript-store paths, `.config/claude/projects`, `.claude/`, `.codex/`, snapshot/Nix-store paths, and temp task output paths.
- Harden `normalize_repo_path()` against unreadable git admin paths.
- Switch `doctor` to probe-mode by default: no exact `messages_fts_docsize` counting, no exact orphan counting, presence-only derived `messages_fts` status. Keep exact counts for `--deep`/`--repair`.
- Make `doctor --runtime --json` runtime-only when no archive/debt/schema/proof/maintenance work was requested.
- Surface incomplete search indexes explicitly: `fts_lifecycle.py` and `search_runtime.py` now check that FTS is populated, not just that the table exists; retrieval candidates stop swallowing backend search exceptions.
- Rebuild action-event rows for every conversation explicitly passed to `update_index_for_conversations()`; keep candidate filtering only for full rebuilds.
- Return explicit `attachment_refs` and `distinct_attachments` from archive stats; restrict the nested `embeddings` object to embedding-state fields.
- Early perf cuts: archive stats on single read snapshot; stats off retrieval-band status; cut latest-query memory on large archives; sqlite read-path memory pressure reduction.

## Verification

- `pytest -q --ignore=tests/integration`
- `pytest -q tests/unit/storage/test_action_event_artifacts.py tests/unit/storage/test_derived_status.py tests/unit/storage/test_repair.py tests/unit/storage/test_fts5.py tests/unit/cli/test_check.py tests/unit/cli/test_products.py tests/unit/core/test_query_retrieval_candidates.py tests/unit/core/test_repo_identity.py tests/unit/core/test_semantic_facts.py tests/unit/storage/test_store_ops.py tests/unit/storage/test_session_product_profiles.py`
- `pytest -q tests/integration/test_health.py tests/integration/test_workflows.py`
- `ruff check polylogue tests devtools`
- Live manual probes on default archive root.

Commits on this branch: 35 (delta against `feature/fix/stack-02-cli-contracts`).

## Stack

Base: `feature/fix/stack-02-cli-contracts`. Next: `feature/perf/stack-04-parse-hardening`.
