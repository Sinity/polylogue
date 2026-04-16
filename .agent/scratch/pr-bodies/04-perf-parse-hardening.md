## Summary

PR 4/9. Stacked on `feature/fix/stack-03-runtime-repair`.

Memory/perf hardening and parse/ingest robustness after the live rebuild exposed the heap shape. Also the late doctor-slimming and parser robustness work.

## Problem

The first full live rebuild captured the real cost profile and exposed concrete bottlenecks:

- Full rebuild wall `56m40s`, peak RSS `~6.9 GiB` — memory was dominated by materialize, not render.
- Materialize was using bulk refresh even on empty archives, bypassing the bounded rebuild path (`rebuild_session_products_async` with `_SESSION_PRODUCT_REBUILD_PAGE_SIZE = 1`).
- Materialize progress reporting advanced by the absolute rebuilt profile count instead of per-chunk delta, so the plain observer accumulated to bogus totals like `171` while the description read `18/10343`.
- Single-record ingest batches (giant raws: `claude-code` `1535 MiB`, `codex` `515 MiB`, and several `277-385 MiB` records) still went through `ProcessPoolExecutor(max_workers=1)`, paying full subprocess setup + pickle + result-transfer cost for zero parallelism.
- Ingest batch memory telemetry reported `peak_rss_self_mb` from `VmHWM`/`ru_maxrss`, so once one high-memory batch ran, every later batch inherited the same peak value (`4886.8` even when the process had settled back to ~537 MiB).
- Session refresh chunked by conversation count (`10`), not by message count, so chunks containing several enormous conversations dominated the materialize path.
- Default `doctor` still paid for exact `count_orphaned_messages_sync` (6.10s) and exact `count_empty_conversations_sync` on every call when shallow checks would suffice.
- Indexing deleted and rebuilt the action-event read model from `messages` + `content_blocks` unconditionally, duplicating work that ingest had already done.
- Strict JSONL validation rejected real Claude Code session lines because `orjson` rejects escaped lone-surrogate strings like `\\udce2` that stdlib `json` accepts. The stream parser and the sampling path used different decoders.
- `polylogue run --reparse --preview` called `reset_parse_status()` before building the preview plan, making `--preview` destructive. Source-scoped `--reparse` also reset parse tracking for the entire archive.
- `run --reparse parse` was filtering parse backlog selection by prior validation status, silently excluding previously validation-failed raws even under forced reparse.
- Site build silently showed `Building site...: 0` for 7+ minutes with no subphase visibility. Acquisition showed multi-minute silent windows while one slow file was being streamed.
- Ingest completion logs dumped the full `batch_observations.batches` payload; materialize stage logs dumped the full `update_chunks` array. Maintenance summary line said `Would change N issue(s)` even though the repair engine reports rows/items, not diagnosed issues.
- Oversize HTML message renders blew past reasonable time/memory budgets.

## Solution

- Dispatch materialize to the bounded rebuild path (`rebuild_session_products_async`, page size 1) when `profile_row_count == 0` and all conversations are processed. Mark the observation `mode = "rebuild-from-empty"`. Fix progress reporting to advance observers by the processed chunk delta while keeping absolute counts in `desc`.
- Run single-worker ingest batches inline (`_iter_ingest_results_sync` when `worker_count <= 1`); keep multi-worker batches on the process pool. Throttle worker selection from per-batch blob sizes. Throttle render workers when the process enters render already above the high-RSS threshold.
- Split ingest batch telemetry into `max_current_rss_mb`, `process_peak_rss_self_mb`, and `peak_rss_growth_mb`; summarize `max_peak_rss_growth_mb` in parsing workflow.
- Chunk session refresh by both conversation count (`10`) and estimated message budget (`5000`), driven by `conversation_stats.message_count`.
- Default `doctor` now uses probe semantics unless `--deep`/`--repair`/`--cleanup`: shallow index checks don't count indexed FTS rows, shallow orphan checks don't count all orphans, shallow derived `messages_fts` reports presence rather than fake `0/0`.
- Indexing service queries action-event repair candidates first, skips rebuild when rows are current, repairs only the missing/stale subset, and reports progress from actual work units.
- Add a stdlib `json.loads()` fallback in `sample_jsonl_payload()` only for line-level JSONL cases `orjson` rejects; keep `orjson` as the fast path. Report first-bad-line detail in malformed JSONL errors for validation/quarantine surfaces.
- Add `force_reparse` to preview planning; simulate reset semantics inside backlog collection and planning instead of mutating persisted state. Scope `reset_parse_status()` to selected sources; stop filtering the parse backlog by prior validation status when force-reparsing.
- Thread blob-store streaming heartbeats through `source_acquisition.py` and forward them as `progress_callback(0, desc=...)` updates. Thread site-build progress callbacks through the actual site builder so scan/manifold subphase activity shows up.
- Compact stage-complete logs by dropping list-valued diagnostic payloads; rename maintenance summary to `Would apply N change(s)` / `Applied N change(s)`.
- Fast-path oversize HTML message renders.
- Reset the default doctor cache surface after dropping dead code paths; repair telemetry stays correct.

## Verification

- `pytest -q --ignore=tests/integration`
- `pytest -q tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_indexing.py tests/unit/pipeline/test_run_sources.py tests/unit/pipeline/test_parsing_service.py tests/unit/pipeline/test_render_service.py tests/unit/pipeline/test_acquisition_streams.py tests/unit/site/test_builder.py tests/unit/storage/test_session_product_refresh.py tests/unit/storage/test_blob_store.py tests/unit/sources/test_source_laws.py tests/unit/cli/test_run.py tests/unit/cli/test_check.py tests/unit/core/test_raw_payload_decode.py tests/unit/core/test_health_core.py tests/unit/storage/test_derived_status.py tests/unit/devtools/test_pipeline_probe.py`
- Live measurement (default archive root): full `run all` wall `56m40s`, peak RSS `6.78 GiB`; parse rerun wall `20m54s`, peak RSS `5.35 GiB`. Further memory reduction remains a tracked follow-up.
- `ruff check polylogue tests devtools`

Commits on this branch: 29 (delta against `feature/fix/stack-03-runtime-repair`).

## Stack

Base: `feature/fix/stack-03-runtime-repair`. Next: `feature/refactor/stack-05-artifact-graph`.
