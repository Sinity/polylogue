---
created: "2026-04-11T03:40:00+02:00"
purpose: "Thorough user-style vetting campaign for Polylogue surfaces, schemas, pipeline, and UX"
status: "archived"
project: "polylogue"
---

# Thorough Vetting Campaign

Superseded by `active/009-operator-brief.md` plus `active/005-thorough-vetting-log.md`.

## Scope

- exercise Polylogue as a user would, through the public CLI and packaged wrappers
- verify that schemas, pipeline stages, derived products, and command surfaces behave coherently
- find concrete UX, semantics, and reliability defects rather than only running nominal smoke checks
- fix defects in atomic slices with verification

## Working Rules

- prefer public commands over interpreter/module entrypoints
- log every meaningful probe and outcome in the append log
- treat warnings, stale generated surfaces, drift, and confusing output as candidate defects
- keep this note focused on campaign structure and current findings; keep the append log chronological

## Campaign Areas

1. Command-surface vetting
2. Operator workflow vetting (`doctor`, `audit`, `run`, `products`)
3. Query/search/retrieval vetting
4. Pipeline and schema vetting
5. Archive freshness and live-data workflow vetting
6. Derived-product and reporting vetting

## Active Hypotheses

- user-facing commands may still expose implementation-oriented wording or surprising defaults
- schema and pipeline flows may have stale/generated drift or hidden assumptions about local state
- some commands likely work only in the happy path and degrade poorly when the archive is absent, stale, or partially initialized

## Findings

### Open

- The previous live archive root was still carrying a legacy inline-raw SQLite layout:
  - `raw_conversations` still had `raw_content BLOB NOT NULL`
  - current acquisition persists raw blobs out-of-line and writes only blob metadata
  - before detection was added, `doctor --json` incorrectly reported `DB reachable` and `run all` failed later with misleading storage errors
- Fresh full-run measurement is in progress against a clean live archive again:
  - legacy DB files were moved to `/home/sinity/.local/share/polylogue/legacy-inline-raw-backup-20260412T104354+0200`
  - active runner log: `.local/logs/polylogue-run-all-20260412T104514+0200.log`
  - active sampler metrics: `.local/logs/polylogue-run-all-20260412T104514+0200.metrics`
  - this run is the one that should answer current ingest/render/site memory use, not the stale legacy archive
- The current fresh `v1` archive is only partially materialized:
  - `conversations = 10342`
  - `session_profiles = 4400`
  - `work_threads = 7962`
  - `day_session_summaries = 0`
  - `session_tag_rollups = 0`
- `polylogue doctor --json` is currently hanging on that live snapshot instead of returning promptly, even though the process RSS stays modest while stalled.
- Session-product rebuild still has an unproven high-memory path:
  - a recent live `materialize` refresh climbed to about `5.6 GiB` HWM before it was terminated
  - an uncommitted optimization now narrows conversation loads for session-product rebuilds, streams conversation ids/root ids, and reduces full-rebuild chunking to `10`
  - this needs a fresh end-to-end timed run, not only unit proof
- Session-profile path attribution still needs live re-materialization proof:
  - unit and semantic regressions now cover both dialogue-derived and tool-derived path noise
  - the stored live profiles still show junk paths because they were materialized before the attribution cleanup landed
  - a timed `polylogue run materialize` refresh is now running to validate the fix against the rebuilt archive
- Memory use still has at least two real operator-facing spikes worth fixing:
  - `polylogue --provider codex --latest open --print-path` hit about `2.0 GiB` RSS for a simple path-resolution command
  - the timed `polylogue run materialize` refresh has already climbed past `3.8 GiB` RSS while rebuilding session products
  - both need code-path inspection, not just more measurement
- The current fresh timed rebuild is still in progress:
  - acquire completed in `111.2s` at `112 raw/s`
  - ingest completed in `833.2s` at `15.0 raw/s`
  - render is currently progressing around `15-17 conv/s`
  - the current timing note is still incomplete until render, site, and index finish
- The active timed rebuild hit one real defect during materialization:
  - session-product refresh touched archived host path `/boot/.git` and raised `PermissionError`
  - fixed locally in the code, but the already-running rebuild still uses the old process image
  - rerun or post-run `materialize` is needed to prove the live path end to end
- Render/site timing remains the main unanswered performance question:
  - the previous full run spent disproportionate time in render and never completed site generation before it was terminated
  - the current rebuild must be allowed to reach render on the fixed code so that the oversized-message render fix is measured, not only unit-tested
- Read-side latency under active ingest is still high even after lock avoidance:
  - `list` and `products profiles` dropped from roughly `23–25s` to roughly `4–5s`
  - `stats --format json` dropped from `25.5s` to `16.1s`
  - `stats` now looks like a real aggregation/perf target rather than a lock-path bug
- The current active rebuild is still running on a stale pre-fix process image:
  - it was launched before the latest doctor/read-path fixes landed
  - current live-command probes therefore validate the code against that partial archive state, not the runner implementation itself
  - if we need fresh end-to-end timing for the new code, we will need one more isolated rerun after this pass
- User-facing read latency during active render is now characterized:
  - `list -n 1` is about `4.15s`
  - `products profiles --limit 1` is about `2.39s`
  - `stats --format json` is still bad at about `29.9s`
  - `doctor --json` is still bad at about `40.8s`
- The expensive part of `doctor` is archive-health work, not runtime checks:
  - `run_archive_health(..., deep=False)` costs about `22.0s`
  - `run_runtime_health(...)` costs about `0.1s`
  - archive-health time is dominated by derived-status collection and archive-debt counting
- `stats` likely pays for more than a cheap archive summary:
  - direct timings on the live snapshot show `get_archive_stats()` at about `2.7s`
  - `aggregate_message_stats()` adds about `6.3s`
  - the remaining CLI cost under live render still needs one more pass to separate command/process overhead from concurrent-write contention
  - a targeted optimization landed: archive stats no longer compute retrieval-band readiness
  - live `stats --format json` improved from about `29.9s` to about `12.3s` during the same rebuild
- The current timed run still shows worker-side debug parser lines because it started before the worker-logging fix landed:
  - confirm on the next rerun that normal `run all` output no longer leaks `[debug]` worker lines

### Fixed

- Legacy inline-raw archives now fail fast and truthfully:
  - schema open paths detect `raw_conversations.raw_content` as an unsupported legacy layout
  - `doctor --json` now reports the real incompatibility instead of claiming the DB is healthy
  - compatible `v1` databases without `blob_size` are still extended in place
- Repo control-surface drift:
  - PR template now points at `devtools render-all`
  - CI now uses `uv run devtools ...`
  - README now keeps a shorter generated docs section and points at `docs/README.md` for the full map
  - local-state wording now says plainly which roots stay at the repo top and why
  - interactive direnv loads now show the compact MOTD reliably again
- Stale archive schema errors on plain root-query commands now fail as user-facing CLI errors instead of raw tracebacks.
- The public query help and showcase baselines now use the public `polylogue` entrypoint consistently, and the root help example now uses a valid option order.
- The root `stats` verb no longer drops grouped mode or structured output:
  - `polylogue stats --by provider` now stays on grouped stats instead of falling back to archive-wide SQL stats
  - `polylogue stats --format json` now emits structured stats output on the SQL-backed archive path
- Seeded demo environments are now isolated from the operator's real home and source tree:
  - `polylogue audit generate --seed --env-only` exports `HOME`, `XDG_CONFIG_HOME`, and `XDG_CACHE_HOME` in addition to the archive/data roots
  - generated fixtures are mirrored into the demo inbox so later `polylogue run ...` calls only see seeded data
  - configured source discovery no longer exposes the Gemini drive source unless there is local cache or Drive auth material
- Showcase JSON contract exercises are now curated to runnable commands instead of being inferred from every raw `--json` flag.
- `polylogue audit --only audit` and `polylogue audit --only exercises` now skip artifact proof, matching the user-visible stage selection instead of failing overall on an unrelated hidden proof stage.
- Archive-wide stats now report real role counts instead of hardcoded zeroes, and the filtered stats query no longer drops its temp id table before computing word totals.
- Shell and local-state hygiene:
  - interactive direnv loads now render the MOTD exactly once
  - the shell exports `PYTHONDONTWRITEBYTECODE=1`
  - shell startup removes stale top-level `result` symlinks and root `__pycache__/`
  - `hatch_build.py` now disables bytecode writes during build-hook execution
- Synthetic ChatGPT fixtures now generate clean top-level conversation ids instead of leaking arbitrary schema text into user-facing archive ids.
- Query-mode machine errors now honor `--format json` and extract the real command path instead of echoing option values inside the JSON error envelope.
- Query-similarity guidance now points at the canonical embedding workflow: `polylogue run embed`.
- Source-checkout version resolution now reads commit identity from `.git` metadata directly, so isolated subprocess workflows still work when `git` is not on `PATH`.
- The `frontier-local` validation lane is green again after removing a stale deleted test reference and fixing interruption-test regressions exposed by that lane.
- Fresh-machine JSON query surfaces now return structured `no_results` envelopes instead of plain-text human messages on empty archives.
- `polylogue open` now supports `--print-path`, so users and scripts can resolve the latest rendered conversation target without launching a browser.
- Empty-archive JSON mode for `polylogue --latest open --print-path` now returns the standard `no_results` machine envelope instead of a plain-text fallback.
- Embedding-status reporting now keeps transcript pending counts consistent across the top-level payload and the retrieval-band detail when no transcript embeddings exist yet.
- Multi-provider `schema audit --json` now scopes each emitted check by provider instead of repeating anonymous check blocks that only differed by order.
- `audit generate --seed --env-only` now teaches the correct shell usage in public help and generated docs: `eval "$(…)"`, not a broken `| eval` pipeline.
- `schema explain --json` now deduplicates repeated semantic-role assignments emitted from schema unions instead of surfacing the same `(path, role)` multiple times.
- The root query shape now explains misplaced root-only flags directly instead of degrading into generic parser failures.
- Live health inspection no longer blocks behind an active `run all` writer:
  - `doctor --json` now opens the archive through a read-only probe connection
  - archive index checks reuse that same probe connection instead of silently reopening the database through the write path
- Active-ingest read surfaces no longer depend on writer-style SQLite setup:
  - async repository query reads now use read-oriented connections when the DB already exists
  - sync search/index/embed-status reads do the same
  - during a live `run all`, `list`, `stats`, and `products profiles` now stay responsive instead of stalling on the writer path
- The current live-vetting run starts from a fresh `v1` archive again:
  - the code already expected `v1`, so no further schema edit was needed
  - the previous live archive state was moved aside as `/home/sinity/.local/share/polylogue.backup-20260412T032235+0200`
  - the replacement rebuild has a managed runner, sampler, log, and metrics file under `.local/logs/`
- Claude Code repeated-record drift no longer corrupts ingest:
  - exact duplicate JSONL records sharing the same `uuid` are now deduplicated in the parser
  - the live failing subagent payload no longer produces duplicated message ids or duplicated action-event ids
- Conversation overwrites now replace runtime rows instead of accumulating stale state:
  - content-changed writes now clear the prior conversation-scoped message, content-block, attachment-ref, conversation-stats, and embedding-status rows before re-materialization
  - removed attachment refs are now pruned by ref-count recalculation rather than lingering behind newer conversation content
  - both async repository writes and sync ingest-batch writes now enforce the same replacement contract
- Archive-scope stats now keep a consistent read snapshot during active ingest:
  - `stats --format json` no longer mixes top-level counts from one moment with embedding/product counts from a later moment
  - `summary.conversations`, `summary.providers`, `summary.messages_total`, and `summary.embeddings.*` now agree on the same snapshot while the writer is still running
- Archive-scope stats now name attachment counts honestly:
  - `summary.attachment_refs` counts message-linked attachment references
  - `summary.distinct_attachments` counts unique attachment rows
  - the nested `embeddings` payload now contains only embedding-state fields instead of leaking unrelated archive totals
- Process-pool workers now initialize normal info-level logging:
  - ordinary `run all` executions will no longer leak parser `debug` lines from child workers
  - fixed in commit `5d7ea210`
- Session-product materialization no longer treats unreadable git admin paths as fatal repo candidates:
  - repo-root normalization now ignores unreadable `.git` admin paths such as `/boot/.git`
  - fixed with direct repo-identity regression coverage
- SQLite connection sizing now matches a local CLI/archive utility better:
  - read and write connections no longer reserve a 512 MiB page cache each
  - read pools now open read-only connections instead of writer-style ones
  - real command RSS dropped materially (`stats`: ~482 MiB -> ~306 MiB, `doctor`: ~1.05 GiB -> ~647 MiB)
- Interrupted render runs no longer explode during pooled-read teardown:
  - aborting `polylogue run render` used to trip async-generator cleanup inside `iter_conversation_ids`
  - pooled connections now stop cleanly when the pool owner tears down first
- A post-run `polylogue run materialize` pass is queued behind the current rebuild so the fixed materialize code is exercised on the new archive without discarding the current full-run timing sample.
- Attribution now filters noisy absolute pseudo-paths before they pollute repo/file profile fields:
  - weird absolute parent-traversal repo paths are canonicalized before git-root detection
  - both dialogue-derived and tool-derived absolute paths now pass through the same filter
  - junk like `/#`, `/12`, `/AFK/browser`, `/Codex`, and `/DAG` is now rejected in the runtime profile path
  - direct regressions lock the behavior in `test_repo_identity.py` and `test_semantic_facts.py`
- Summary/list/open query reads no longer drag giant Codex raw payloads through memory:
  - root cause 1: `list_summaries_by_query()` still selected full `conversations.*`, including Codex `provider_meta`
  - measured live Codex payload sizes:
    - average `provider_meta`: about `2.1 MiB`
    - max `provider_meta`: `476,066,408` bytes
    - total Codex `provider_meta`: about `3.89 GiB`
  - fixed by adding a narrow summary projection query that sets `provider_meta = NULL` for summary reads
  - `open` also now resolves the render target by canonical provider/id path instead of scanning the entire render tree
- Read-path SQLite tuning was still massively overprovisioned for a local CLI utility:
  - the async/sync read paths were using `READ_CACHE_SIZE_KIB = 131072` and `mmap_size = 1 GiB`
  - isolated read-profile probes showed:
    - baseline-ish read profile: about `761 MiB` RSS
    - reduced profile (`32 MiB` cache, `128 MiB` mmap): about `65 MiB` RSS
    - minimal profile (`8 MiB` cache, `0` mmap): about `39 MiB` RSS
  - adopted the reduced read profile in the product code (`32 MiB` cache, `128 MiB` mmap)
- Live command measurements after both fixes:
  - `polylogue --provider codex --latest list --format json`
    - before: `14.22s`, `2012944 KB`
    - after summary projection only: `3.27s`, `813396 KB`
    - after read tuning: `3.33s`, `117712 KB`
  - `polylogue --provider codex --latest open --print-path`
    - before: `14.15s`, `2014496 KB`
    - after summary projection only: `3.21s`, `813100 KB`
    - after read tuning: `3.20s`, `117528 KB`

## 2026-04-11 artifact-proof fix
- Removed four stale `doctor semantic-proof` showcase exercises from `polylogue/showcase/exercise_catalog.json` because the surface no longer exists.
- Found remaining seeded full-audit failure in durable artifact proof: large `.json` Claude/Gemini exports were inspected from a 64 KB prefix, then mis-decoded as JSONL and counted as `metadata_document + decode_failed`.
- Implemented bounded full-read fallback for non-stream JSON blobs <= 8 MiB in `polylogue/storage/artifact_inspection.py` and added a regression test for large Claude-style documents.

## 2026-04-12 live-ingest defect wave

Resolved during the fresh archive rebuild:

- Codex schema bundle rejected live `session_meta.payload.source.subagent = "review"` records from review-mode sessions. Fixed in commit `f516a73f` by extending the bundled Codex schema and locking it with a live-shape regression test.
- Raw acquisition archived zero-byte source files and zip entries, which later surfaced as decode failures during ingest. Fixed in commit `4bc1deb7` by rejecting empty artifacts during acquisition and recording them as source failures.
- Active acquisition no longer floods operator logs with one warning per empty artifact. Empty-file and empty-entry details now stay at debug level, while normal runs emit one summary warning per source.
- Grouped stats now honor the canonical no-results contract in JSON mode, and `stats --by repo` routes through the live `repo` dimension again instead of a stale `project` branch check.
- `doctor` default health probes are materially cheaper on large live archives:
  - default `doctor --json` no longer pays for the expensive orphaned-content-block scan
  - derived-status collection no longer recomputes retrieval-band readiness inside archive summary status
  - action-event status now derives total source conversations from the already-computed valid/orphan buckets instead of re-running a full count query
  - a partial `content_blocks(type='tool_use')` conversation index is now ensured for existing archives
- Async first-read schema bootstrap is now race-safe again:
  - a fresh async DB can no longer expose an empty file to concurrent readers before the schema exists
  - already-initialized archives still avoid redundant read-path schema work
- Hybrid search restored a stable read-connection boundary:
  - tests and callers can again inject either DB paths or in-memory sqlite handles without falling through the wrong helper
- Session-product rebuild once again preserves repo names derived from provider git metadata:
  - provider git remotes like `git@github.com:Sinity/sinex.git` survive attribution normalization into `repo_names`
  - dialogue-only language hints no longer infer `r` from arbitrary words like `branch`
- The stale internal `stats_by=project` expectation was removed from the query tests; the public grouped dimension remains `repo`

Follow-up still in scope:

- keep probing live and post-run command latency/memory for `doctor`, `list`, `stats`, and `products`
- keep sampling the in-flight fresh rebuild after restarting on the current code; the previous isolated run captured stale pre-fix ingest/materialize telemetry
- clean the user-facing path attribution noise visible in `products profiles`

## 2026-04-12 memory-status update

- The earlier `~2.2 GiB` live sample undercounted the real hot spots. Stage telemetry from the same run later showed:
  - ingest peak self RSS about `4.9 GiB`
  - materialize peak self RSS about `9.0 GiB`
- Two concrete fixes landed from that investigation:
  - Claude Code duplicate suppression no longer retains full raw record payloads in memory
  - bulk session-product refresh now uses the same small default chunk size as the full rebuild path (`10`, not `100`)
- Direct post-fix probes now show:
  - giant `1.61 GiB` Claude Code raw ingest completes at about `791 MiB` RSS
  - isolated hydration + profile build for the largest `50k`-message conversation peaks at about `330 MiB` RSS
- Practical implication:
  - the stale in-flight run should be restarted on current code before we treat its end-to-end RSS numbers as representative
