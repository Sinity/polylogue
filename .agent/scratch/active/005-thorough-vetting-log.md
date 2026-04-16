---
created: "2026-04-11T03:40:00+02:00"
purpose: "Append-only execution log for the Polylogue thorough vetting campaign"
status: "active"
project: "polylogue"
---

# Thorough Vetting Log

## 2026-04-11 03:40 CEST

- Initialized the thorough vetting campaign.
- Created:
  - `.claude/scratch/archive/005-thorough-vetting-campaign.md`
  - `.claude/scratch/active/005-thorough-vetting-log.md`
- Next:
  - inspect active quality-plan context in `.claude/scratch/concepts/002-scenario-driven-quality-frontier.md`
  - survey the live command surface and docs before driving user-style workflows

## 2026-04-11 03:44 CEST

- Reviewed the active quality-frontier note in `.claude/scratch/concepts/002-scenario-driven-quality-frontier.md`.
- Confirmed it is still forward-looking, but already-landed items are now historical rather than pending implementation.
- Inspected the live command catalog with `devtools --list-commands --json`.
- Confirmed the public root CLI identity with:
  - `polylogue --help`
  - `polylogue --version`

## 2026-04-11 03:48 CEST

- Began user-style probing of public commands.
- `polylogue --plain stats --by provider --format json` failed with a raw traceback on a stale archive schema instead of a structured user-facing failure.
- `polylogue stats --by provider --format json --limit 20` showed that the root query shape is still easy to misuse when flags are placed after the verb.
- `polylogue --plain doctor --json --preview` correctly returned a structured invalid-arguments envelope.
- `polylogue --plain products status --json` correctly returned a structured unknown-command envelope for a bad invocation.

## 2026-04-11 03:56 CEST

- Switched to the requested repo-pristine pass before further operator-style vetting.
- Surveyed the public docs and control surfaces for remaining drift and needless clutter.
- Found concrete cleanup targets:
  - PR template still exposed `python -m devtools render-all`
  - CI still used `python -m devtools ...` internally
  - README still duplicated too much of the full docs index from `docs/README.md`
  - local-state docs and MOTD JSON still used vague wording around roots kept at the repo top
  - `.envrc` only showed the MOTD once per shell session, which explained interactive reloads sometimes showing nothing

## 2026-04-11 04:20 CEST

- Completed the repo-pristine cleanup pass in two commits:
  - `8d6bdddd` `docs: tighten repo guidance and generated surfaces`
  - `342e7c72` `chore: make devtools consistent across shell and ci`
- Cleanup outcomes:
  - shortened the generated README docs section and left `docs/README.md` as the full docs map
  - tightened release/version wording in `README.md` and `CONTRIBUTING.md`
  - removed the last public `python -m devtools` guidance from the PR template
  - made the package expose `devtools` so CI can use `uv run devtools ...`
  - made `.envrc` show the compact MOTD again on interactive direnv loads
  - clarified local-state wording in `docs/devtools.md` and `.cache/README.md`
- Verification:
  - `devtools render-all --check`
  - `uv run devtools render-all --check`
  - `pytest -q tests/unit/devtools/test_render_docs_surface.py tests/unit/devtools/test_project_motd.py tests/unit/devtools/test_render_devtools_reference.py tests/unit/core/test_version.py`
  - `nix build .#polylogue`

## 2026-04-11 04:31 CEST

- Resumed operator-style public CLI vetting against the live archive state.
- Confirmed the stale-schema failure was still raw on normal query verbs:
  - `polylogue --plain stats --by provider --format json`
  - `polylogue --plain --provider claude-code --since 2026-01-01 stats --by repo --format json`
  - `polylogue --plain list --format json --limit 1`
- All three paths still raised a full traceback ending in `polylogue.errors.DatabaseError` for the stale archive schema.
- Confirmed `polylogue --plain doctor --json` handled the same stale schema state correctly with a structured JSON report and actionable remediation text.
- Confirmed the public help was internally inconsistent:
  - `polylogue --plain query --help` advertised `polylogue stats --by repo --provider claude-code --since 2026-01-01 --format json`
  - the real parser rejected that example with `Error: No such option: --provider`

## 2026-04-11 04:43 CEST

- Fixed the root CLI error boundary so plain-command `PolylogueError` failures now render as Click-style user errors instead of raw tracebacks.
- Corrected the root help example so provider/date filters appear before the query verb where the parser actually accepts them.
- Switched showcase CLI verification from `python -m polylogue` to the public `polylogue` command and refreshed the full help baseline corpus to match the public entrypoint.
- Added focused regression coverage for:
  - plain and JSON machine-main error handling
  - the corrected root help example
  - showcase CLI boundary use of the public `polylogue` executable
- Verification:
  - `pytest -q tests/unit/cli/test_machine_main.py tests/unit/cli/test_click_app.py tests/unit/showcase/test_cli_boundary.py tests/unit/showcase/test_verification_lane.py`
  - `devtools render-all --check`
  - `devtools verify-showcase --update`
  - `devtools verify-showcase`

## 2026-04-11 04:55 CEST

- Continued user-style probing against the public query surface and found a grouped-stats regression.
- `polylogue stats --by provider` was silently dropping grouped mode because the positional `stats` verb still forced `stats_only=True`.
- `polylogue stats --format json` on the SQL-backed archive stats path also ignored structured output and always rendered plain text.
- Fixed both defects:
  - `polylogue/cli/query_verbs.py` now keeps `stats_only` false when `--by` is present
  - `polylogue/cli/query.py` now passes the requested output format into SQL-backed stats rendering
  - `polylogue/cli/query_stats.py` now emits structured JSON/YAML/CSV for archive-wide stats, with a summary payload instead of text-only output
- Added regression coverage:
  - `tests/unit/cli/test_click_app.py`
  - `tests/unit/cli/test_query_exec_laws.py`
- Verification:
  - `pytest -q tests/unit/cli/test_click_app.py tests/unit/cli/test_query_exec_laws.py -k 'stats_by_subcommand_preserves_grouped_stats_mode or output_stats_sql'`
  - `polylogue stats --by provider`
  - `polylogue stats --by provider --format json`

## 2026-04-11 05:00 CEST

- Probed the seeded demo environment generated by `polylogue audit generate --seed --env-only`.
- Confirmed the export stream is shell-safe: stdout contains only exports and stderr contains logs.
- Confirmed a deeper defect remained after applying the exported env:
  - `polylogue run --preview all` still scanned live configured sources instead of an isolated demo universe
  - the preview attempted real `claude-code`, `codex`, and Gemini work, including a Gemini auth failure
- Ran tier-0 QA exercises against the seeded environment:
  - `POLYLOGUE_FORCE_PLAIN=1 polylogue audit --only exercises --tier 0`
  - result: `50/56 passed`, `6 failed`
- The failing exercises were all generated JSON contract exercises:
  - `json-audit` timed out
  - `json-run-embed` exited 1
  - `json-schema-compare`, `json-schema-explain`, `json-schema-generate`, `json-schema-promote` exited 2
- Root cause appeared to be over-broad JSON exercise generation from raw `--json` flag discovery rather than a curated runnable contract set.

## 2026-04-11 06:15 CEST

- Fixed the seeded demo environment so later operator commands stay inside the demo universe:
  - `polylogue audit generate --seed --env-only` now exports `HOME`, `XDG_CONFIG_HOME`, and `XDG_CACHE_HOME`
  - the generator creates a fake home and mirrors generated provider fixtures into the demo inbox
  - configured source discovery now omits the Gemini drive source unless there is local cache or Drive auth material
- Verified the operator path:
  - `eval "$(polylogue audit generate --seed --env-only -o "$tmpdir" -n 1 -p chatgpt)"`
  - `polylogue run --preview all`
  - result now scanned only `inbox`, with no live `claude-code`, `codex`, or Gemini auth leakage
- Replaced showcase JSON contract generation based on raw `--json`-flag discovery with a curated runnable command set.
- Found and fixed a second QA semantics bug while verifying the showcase path:
  - `polylogue audit --only audit` still ran artifact proof and could fail overall even though the requested audit stage passed
  - `polylogue audit --only exercises` did the same
  - proof now follows the selected QA stages instead of acting as an unrequested hidden failure source
- Verification:
  - `pytest -q tests/unit/cli/test_click_app.py tests/unit/core/test_config.py tests/unit/showcase/test_exercise_catalog.py tests/unit/showcase/test_qa_runner.py`
  - `polylogue run --preview all` in a seeded env now previews only demo inbox work
  - `POLYLOGUE_FORCE_PLAIN=1 polylogue audit --only exercises --tier 0` now returns `Overall: PASS`
  - `POLYLOGUE_FORCE_PLAIN=1 polylogue audit --only audit --json` now returns `overall_status: ok` with `proof.status: skip`

## 2026-04-11 06:35 CEST

- Ran the broad non-integration suite and found four stale PTY snapshot failures.
- These were not new code bugs:
  - three snapshots still expected the old invalid root-help example ordering
  - one snapshot still expected the pre-fix traceback on stale-schema query failure instead of the newer user-facing CLI error
- Regenerated the PTY snapshot baseline with:
  - `pytest -q tests/unit/cli/test_terminal_snapshots.py --snapshot-update`
- Continued the seeded operator battery and found a real stats defect:
  - `polylogue stats --format json` reported `messages_total=137` while `messages_user=0` and `messages_assistant=0`
  - root cause was in `aggregate_message_stats()`: role counts were hardcoded to zero, and the filtered branch dropped its temp id table before the word-count query
- Fixed the storage-layer stats aggregation and added a direct storage-level regression test.
- Verification:
  - `pytest -q tests/unit/storage/test_store_ops.py -k aggregate_message_stats_reports_role_counts_and_words tests/unit/cli/test_query_exec_laws.py -k 'output_stats_sql_archive_scope_includes_embedding_state or output_stats_sql_json_contract'`
  - seeded `polylogue stats --format json` now reports nonzero `messages_user` and `messages_assistant`
  - `pytest -q --ignore=tests/integration` passed cleanly

## 2026-04-11 07:00 CEST

- Tightened shell and local-state hygiene after another manual operator pass:
  - removed the duplicate `.envrc` MOTD emission and left the shell-hook MOTD as the single source
  - changed the compact MOTD to always show generated-surface status again
  - made the devshell export `PYTHONDONTWRITEBYTECODE=1`
  - made shell startup remove stale top-level `result` symlinks and root `__pycache__/`
  - made `hatch_build.py` disable bytecode writes when Hatch imports the build hook
- Verification:
  - `devtools status`
  - `direnv reload`
  - `pytest -q tests/unit/devtools/test_project_motd.py tests/unit/devtools/test_build_package.py tests/unit/core/test_version.py`

## 2026-04-11 07:08 CEST

- Continued seeded user-style CLI probing and found a synthetic-fixture defect visible at the public surface:
  - `polylogue --format json list -n 10` showed malformed ChatGPT conversation ids like `chatgpt:synthetic-55392 synthetic-55392 syn`
  - the defect came from synthetic tree payload generation leaving the top-level ChatGPT `id` field as arbitrary schema-generated text
- Fixed synthetic ChatGPT conversation envelopes to assign a clean UUID conversation id and stable top-level timestamps.
- Added direct regression coverage in `tests/unit/core/test_synthetic_semantics.py`.
- Re-verified the operator path:
  - seeded `polylogue --format json list -n 10 | jq -r '.[].id'` now yields clean ids for all providers
  - `pytest -q tests/unit/core/test_synthetic_semantics.py tests/unit/devtools/test_project_motd.py tests/unit/devtools/test_build_package.py tests/unit/core/test_version.py`

## 2026-04-11 07:12 CEST

- Built the packaged artifact and checked the packaged wrappers:
  - `devtools build-package`
  - `.local/result/bin/polylogue --version`
  - `.local/result/bin/polylogue --plain --help`
  - `polylogue-python` import path
  - `mcp-polylogue --help`
- No defect surfaced there; the packaged wrappers matched the live repo surface.

## 2026-04-11 07:28 CEST

- Continued real-user probing from the seeded archive and found another command-surface defect:
  - `polylogue --plain --similar "debug parser" list`
  - the failure message still told users to run `polylogue embed`
  - that is not the public workflow anymore; the canonical command is `polylogue run embed`
- Fixed both similarity-error branches in `polylogue/cli/query.py` so the guidance now points at `polylogue run embed`.
- Verification:
  - `pytest -q tests/unit/cli/test_query_exec.py -k similar`
  - `rg -n "polylogue embed" polylogue tests docs README.md CONTRIBUTING.md TESTING.md`

## 2026-04-11 07:36 CEST

- Probed the root query JSON contract against a stale archive and found that machine-mode detection still missed query-mode `--format json`.
- `polylogue --format json list -n 1` now emits a structured runtime envelope instead of falling back to plain early-failure behavior.
- Found and fixed a second issue at the same boundary:
  - the pre-scan command extractor was treating option payloads like `json` and `1` as command words
  - the emitted `command` path is now the real query verb, e.g. `["list"]`
- Verification:
  - `pytest -q tests/unit/cli/test_machine_contract.py tests/unit/cli/test_machine_main.py tests/unit/cli/test_query_exec.py -k 'json or similar or extract_command'`
  - live probe: `polylogue --format json list -n 1`

## 2026-04-13 05:09 CEST

- Implemented the first real unifying vertical slice from `.claude/scratch/completed/015-unifying-vertical-slice-plan.md`.
- Chose the narrow but high-leverage path:
  - `tool_use source -> action_event_rows -> action_event_fts -> doctor/debt/repair`
- Added a shared semantic model in:
  - `polylogue/storage/action_event_artifacts.py`
- Refactored the path projections to reuse that model:
  - `polylogue/storage/action_event_status.py`
  - `polylogue/storage/derived_status_products.py`
  - `polylogue/storage/derived_status.py`
  - `polylogue/storage/embedding_stats_support.py`
  - `polylogue/storage/repair.py`
- The new shared state now makes the following concepts explicit and reused:
  - missing conversations
  - stale action-event rows
  - pending action-event FTS rows
  - stale extra action-event FTS rows
  - canonical repair count/detail
  - canonical row and FTS readiness
- This fixes the specific drift where extra stale action-event FTS rows could make readiness false while remaining undercounted or invisible in debt and retrieval-evidence projections.
- Added focused regression coverage:
  - `tests/unit/storage/test_action_event_artifacts.py`
  - `tests/unit/storage/test_derived_status.py`
  - `tests/unit/storage/test_repair.py`
- Verification:
  - `ruff check polylogue/storage/action_event_artifacts.py polylogue/storage/action_event_status.py polylogue/storage/derived_status.py polylogue/storage/derived_status_products.py polylogue/storage/embedding_stats_support.py polylogue/storage/repair.py tests/unit/storage/test_action_event_artifacts.py tests/unit/storage/test_derived_status.py tests/unit/storage/test_repair.py`
  - `pytest -q tests/unit/storage/test_action_event_artifacts.py tests/unit/storage/test_derived_status.py tests/unit/storage/test_repair.py tests/unit/storage/test_fts5.py -k 'action_event or retrieval or derived_status'`
  - `pytest -q tests/integration/test_health.py -k 'action_event_read_model or uses_preview_counts'`
- Result:
  - the artifact-graph idea is now proven on one real production seam without introducing a broad framework yet
  - next intended falsifier path remains `raw_payload -> validation_state -> parse_backlog -> parsed/quarantine`

## 2026-04-13 05:39 CEST

- Implemented the second falsifier slice for the raw-state planning path.
- Added a shared persisted raw-state semantic model in:
  - `polylogue/storage/raw_ingest_artifacts.py`
- The model now owns:
  - ordinary validation backlog eligibility
  - ordinary parse backlog eligibility
  - force-reparse validation backlog eligibility
  - force-reparse parse backlog eligibility
  - quarantine classification for validation-failed unparsed raws
  - shared SQL query specs for backlog selection
- Refactored both backlog collection and scan-time planning to consume the same semantics:
  - `polylogue/pipeline/services/planning_backlog.py`
  - `polylogue/pipeline/services/planning_runtime.py`
- This removes the previous drift risk where:
  - SQL backlog predicates
  - scan-time in-memory raw-state decisions
  - force-reparse behavior
  could diverge quietly.
- Added focused regression coverage:
  - `tests/unit/storage/test_raw_ingest_artifacts.py`
  - `tests/unit/pipeline/test_parsing_service.py`
- Verification:
  - `ruff check polylogue/storage/raw_ingest_artifacts.py polylogue/pipeline/services/planning_backlog.py polylogue/pipeline/services/planning_runtime.py tests/unit/storage/test_raw_ingest_artifacts.py tests/unit/pipeline/test_parsing_service.py`
  - `pytest -q tests/unit/storage/test_raw_ingest_artifacts.py tests/unit/pipeline/test_parsing_service.py -k 'raw_ingest_artifact or parseable_backlog_statuses or force_reparse'`
  - `pytest -q tests/unit/cli/test_run.py -k reparse`
  - `pytest -q tests/unit/pipeline/test_run_sources.py -k 'reuses_persisted_validation_status or force_reparse'`
- Result:
  - the unifying approach now survives a second path with very different semantics from action-event status/repair
  - the next architectural step can reasonably move upward into explicit artifact/dependency maps and compiled control-plane projections

## 2026-04-13 06:02 CEST

- Introduced the first explicit runtime artifact/dependency map for the two proven paths.
- Added the core model in:
  - `polylogue/artifact_graph.py`
- Added the first projection in:
  - `devtools/artifact_graph.py`
- Current graph coverage is intentionally narrow and concrete:
  - `raw_validation_state -> validation_backlog -> parse_backlog -> parse_quarantine`
  - `tool_use_source_blocks -> action_event_rows -> action_event_fts -> action_event_health`
- This is not yet a generic scenario compiler. It is the first explicit architectural map that names:
  - artifact layers
  - dependency edges
  - repair targets
  - health surfaces
  - curated high-value paths
- Verification:
  - `ruff check polylogue/artifact_graph.py devtools/artifact_graph.py tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py`
  - `pytest -q tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py`
  - `python -m devtools.artifact_graph --json`
- Result:
  - the unifying work now has a shared map, not just two isolated semantic helper modules
  - the next natural step is to compile one real scenario/test/benchmark projection from this graph instead of hand-authoring parallel control-plane metadata

## 2026-04-12 15:42 CEST

- Started a fresh isolated end-to-end vetting run on current code:
  - `POLYLOGUE_MOTD_RENDERED=1 /run/current-system/sw/bin/time -v polylogue --plain run all`
  - isolated root: `~/.local/share/polylogue-vetting-20260412T154222+0200/xdg-data/polylogue`
  - log: `.local/logs/polylogue-run-all-20260412T154222+0200.log`
- The run is still active as of this entry.
- Early observations:
  - source scanning RSS stayed low, around `202 MiB`
  - Gemini/Drive acquisition was much slower than local sources and dominated the first several minutes
  - by `21m38s` elapsed the run had entered deep ingest with process RSS around `1.0 GiB`
  - adaptive ingest worker selection is visibly active in logs:
    - large batches (`~100-126 MiB`) are running with `workers=2`
    - smaller batches (`~12-22 MiB`) still use `workers=8`

## 2026-04-12 15:50 CEST

- Re-probed the live default archive with `polylogue --plain doctor --json`.
- The archive is structurally healthy but still has significant operator debt:
  - `390` empty conversations

## 2026-04-12 16:55 CEST

- Finished the fresh-archive FTS readiness investigation against the isolated
  run output.
- Root cause:
  - `messages_fts` intentionally indexed only `messages.text IS NOT NULL`
  - readiness checks, derived-status summaries, and dangling-FTS repair were
    still comparing against all `messages`
  - trigger DDL also tried to insert null-text rows, so incremental maintenance
    disagreed with full rebuild semantics
- Fixes staged in the repo:
  - shared `FTS_INDEXABLE_MESSAGE_COUNT_SQL`
  - readiness and derived-status now compare against indexable message rows
  - repair only inserts null-safe rows
  - archive trigger DDL now skips null-text inserts and updates
- Verification:
  - focused tests for `doctor` and dangling-FTS repair passed
  - the fresh isolated archive now reports:
    - `index`: `ok`

## 2026-04-12 18:36 CEST

- Backed up the live default archive before destructive local reset:
  - `~/.local/share/polylogue/reset-backup-20260412T183637+0200`
- Started a fresh live rebuild on the default archive with timing capture:
  - `/run/current-system/sw/bin/time -f '\n[time] wall=%E rss_kb=%M cpu=%P' polylogue --plain run all`
  - log: `.local/logs/run-all-default-20260412T183655+0200.log`
- Full live rebuild completed successfully.
- Measured full-run cost on the real archive:
  - wall: `56m40.43s`
  - max RSS: `7109260 kB` (`~6.78 GiB`)
  - CPU: `51%`
- Stage observations from the live rebuild:
  - acquire stayed low, around `196.6 MiB`
  - ingest peaked around `5623.8 MiB`
  - materialize peaked around `6942.6 MiB`
  - site generation completed cleanly once progress reporting existed
- The largest observed raw artifact in the live archive was about `1.5 GiB`.

## 2026-04-12 19:10 CEST

- Diagnosed the malformed JSONL failures more precisely.
- The first remaining failing Claude Code raw was:
  - `/home/sinity/.claude/projects/-realm-project-sinex/da69faf4-16cb-46e7-94ff-ba40cfa9a346.jsonl`
  - first bad line: `819`
- The failure was not invalid UTF-8. The file is valid UTF-8 text containing
  escaped lone-surrogate JSON string content that stdlib `json` accepts and
  `orjson` rejects.
- Fixed the decoder mismatch by keeping `orjson` as the fast path and falling
  back to stdlib `json.loads()` only for JSONL lines rejected on that strictness
  boundary.

## 2026-04-12 19:34 CEST

- Found and fixed a second real operator bug:
  - `polylogue run --reparse --preview ...` mutated parse state during preview
  - source-scoped `--reparse` was resetting parse state too broadly
- Landed repo fixes so preview is side-effect free and reset scope respects the
  selected sources.

## 2026-04-12 19:52 CEST

- A `run --reparse parse` repair launched during active code edits failed
  mid-run for the wrong reason:
  - process-pool workers imported old code while the repo was changing
  - resulting error: `cannot import name '_sample_jsonl_payload_with_detail'`
- This was not a stable product bug; it was a running process surviving across a
  code edit.
- Archive state after that broken run:
  - `9538` raws total
  - `3392` with `parsed_at IS NULL`
  - `3389` with `parse_error IS NOT NULL`

## 2026-04-12 20:00 CEST

- Started a clean rerun from committed code only:
  - `/run/current-system/sw/bin/time -f '\n[time] wall=%E rss_kb=%M cpu=%P' polylogue --plain run --reparse parse`
  - log: `.local/logs/run-parse-rerun-20260412T200000+0200.log`
- The rerun stayed healthy throughout and exposed the heavy-tail corpus shape
  clearly:
  - a few `100–150 MiB` batches forced worker counts down to `1` or `2`
  - the heaviest singleton observed was about `150.4 MiB`
  - many expensive batches were dominated by Claude Code sessions with very high
    message counts
  - repeated `slow_write` events for the same conversation ids showed that raw
    records and final conversation ids are not one-to-one in the live corpus
- Clean rerun result:
  - wall: `20m54.02s`
  - max RSS: `5613500 kB` (`~5.35 GiB`)
  - CPU: `56%`
  - parse failures reported by the rerun itself: `0`
  - summary counts: `1938` changed conversations, `849991` messages, `717`
    attachments

## 2026-04-12 20:24 CEST

- Queried the live DB directly after the clean rerun:
  - total raws: `9538`
  - `parsed_at IS NULL`: `3`
  - `parse_error IS NOT NULL`: `0`
- The `9535` reset count from `run --reparse parse` was therefore explained by
  three rows that never had parse state to clear, not by source-scope drift.
- Those three rows were still unparsed because they were already
  `validation_status='failed'` with malformed JSONL errors:
  - two variants of the same `b8c8d990-...` Claude Code session
  - one `da69faf4-...` Claude Code session
- This exposed a real operator-facing bug:
  - `run --reparse parse` looked clean
  - but the parse backlog silently excluded previously validation-failed raws
  - so those raws never reached the inline revalidation/parse worker path
- Fixed the planning backlog so `force_reparse=True` stops filtering parse
  backlog selection by prior validation status.
- Added regression coverage in `tests/unit/pipeline/test_parsing_service.py`.
- Verification:
  - `pytest -q tests/unit/pipeline/test_parsing_service.py -k 'force_reparse or parseable_backlog_statuses'`
  - `ruff check polylogue/pipeline/services/planning_backlog.py tests/unit/pipeline/test_parsing_service.py`

## 2026-04-12 19:36 CEST

- Backed up the live archive before the schema-reset/rebuild work:
  - `~/.local/share/polylogue/reset-backup-20260412T183637+0200`
- Ran a full live rebuild from a fresh archive:
  - command: `/run/current-system/sw/bin/time -f '\n[time] wall=%E rss_kb=%M cpu=%P' polylogue --plain run all`
  - log: `.local/logs/run-all-default-20260412T183655+0200.log`
- Outcome:
  - completed successfully
  - wall: `56m40.43s`
  - max RSS: `7109260 kB`
  - CPU: `51%`
  - final summary:
    - `9091` conversations
    - `2452078` messages
    - `2839` attachments
    - `7392` materialized
    - `7392` rendered
    - `3` parse failures
- Stage observations from that run:
  - ingest peak RSS was about `5.6 GiB`
  - materialize peak RSS was about `6.9 GiB`
  - site generation completed cleanly once progress reporting was added

## 2026-04-12 19:43 CEST

- Identified the true remaining malformed-input split:
  - two failures were stale fallout from the old lone-surrogate strictness mismatch
  - one failure was a genuinely corrupted Claude Code JSONL raw:
    - `/home/sinity/.claude/projects/-realm-project-sinex/da69faf4-16cb-46e7-94ff-ba40cfa9a346.jsonl`
    - line `819`
- Corruption shape:
  - one tool-result string is truncated mid-text
  - a second JSON object begins on the same logical line
- This is real source corruption, not just `orjson` Unicode strictness.

## 2026-04-12 19:46 CEST

- Fixed a destructive CLI defect:
  - `polylogue run --reparse --preview ...` used to clear parse state before building the preview plan
  - preview is now side-effect free
- Fixed a second scope bug:
  - `polylogue run --source X --reparse ...` used to clear parse tracking for the entire archive
  - reset is now scoped to the selected sources

## 2026-04-12 19:57 CEST

- While adding malformed-JSONL diagnostics, I edited code underneath an already-running `run parse` repair.
- That produced a false mid-run regression:
  - worker processes started before the helper existed
  - later batches failed with `cannot import name '_sample_jsonl_payload_with_detail'`
- This was not a stable-code failure. It was caused by changing the worker import surface during a live process-pool run.
- The interrupted repair run ended with mixed state:
  - `9538` raw total
  - `3392` with `parsed_at IS NULL`
  - `6146` with `parsed_at IS NOT NULL`
  - `3389` with `parse_error IS NOT NULL`

## 2026-04-12 19:59 CEST

- Finished and committed the malformed-JSONL diagnostics improvement:
  - commit: `a6dacd99`
  - subject: `fix: report first malformed jsonl line`
- Behavior change:
  - strict malformed JSONL failures now carry first-bad-line detail
  - verification/quarantine surfaces carry the same detail
  - public `sample_jsonl_payload()` stayed stable; the extra detail path is internal
- Verification:
  - `ruff check polylogue/lib/raw_payload_decode.py polylogue/pipeline/services/validation_runtime.py polylogue/pipeline/services/ingest_worker.py polylogue/schemas/verification_corpus.py tests/unit/core/test_raw_payload_decode.py`
  - `pytest -q tests/unit/core/test_raw_payload_decode.py tests/unit/core/test_schema_validation.py -k 'malformed_jsonl or verify_raw_corpus_quarantines_malformed_jsonl_lines'`
  - `pytest -q tests/unit/pipeline/test_ingestion_chaos.py -k 'malformed_jsonl_line_in_codex_raw or truncated_jsonl_line_in_codex_raw'`

## 2026-04-12 20:00 CEST

- Restarted live parse repair from committed code:
  - command: `/run/current-system/sw/bin/time -f '\n[time] wall=%E rss_kb=%M cpu=%P' polylogue --plain run --reparse parse`
  - log: `.local/logs/run-parse-rerun-20260412T200000+0200.log`
- Early rerun baseline:
  - parse status reset for `9535` raw records
  - first giant singleton batches are again processing cleanly with `workers=1`
  - the rerun no longer shows the mid-run import failure
    - `fts_sync`: `Messages FTS ready (1,747,955/1,747,955 rows)`
    - `dangling_fts`: `0` issues
  - `1,855,291` message FTS rows pending
  - `323,048` action-event FTS rows pending
  - transcript embeddings entirely pending
- The preview maintenance path works, but surfaced two UX issues:
  - `index` was reported as `OK messages indexed: 0` even while message FTS was entirely pending
  - preview summary ended with `Would change 2178339 issue(s)`, which is not operator-grade wording
- Kernel logs showed real NVIDIA driver OOM events around `14:27`, but not Linux OOM-killer events for the Polylogue process itself.

## 2026-04-12 16:00 CEST

- Fixed a real user-facing search lie:
  - on an archive with an existing but incomplete `messages_fts`, text queries were falling through to `no_results`
  - root cause was two-part:
    - search only checked that the FTS table existed, not that it was populated
    - query retrieval swallowed backend search exceptions and silently degraded into a non-search fallback
- Landed fixes in:
  - `polylogue/storage/fts_lifecycle.py`
  - `polylogue/storage/search_runtime.py`
  - `polylogue/storage/backends/queries/conversations_search.py`
  - `polylogue/lib/query_retrieval_candidates.py`
  - `polylogue/health.py`
- Added regression coverage in:
  - `tests/unit/storage/test_fts5.py`
  - `tests/unit/cli/test_check.py`
  - `tests/unit/core/test_query_retrieval_candidates.py`
- Live verification now behaves correctly:
  - `polylogue --plain --format json project --limit 3`
  - now returns `runtime_error` with message `Search index is incomplete. Run \`polylogue doctor --repair\` or \`polylogue run all\`.`

## 2026-04-12 16:04 CEST

- Ran a broader targeted suite after the search-readiness fix.
- That broader run surfaced a separate pre-existing bug unrelated to the FTS readiness slice:
  - `tests/unit/storage/test_fts5.py::test_update_index_refreshes_action_entries_for_updated_tool_blocks`
  - `update_index_for_conversations()` is not clearing stale `action_events_fts` rows when tool blocks change
- This is now tracked as an open bug for the next atomic slice rather than being mixed into the search-readiness commit.

## 2026-04-12 03:44 CEST

- The fresh timed rebuild reached materialization and surfaced a host-path defect:
  - `Session product refresh failed (non-fatal): [Errno 13] Permission denied: '/boot/.git'`
- Traced the failure to `polylogue/lib/repo_identity.py`:
  - repo-root normalization was probing `(candidate / ".git").exists()` for arbitrary archived absolute paths
  - unreadable admin paths like `/boot/.git` could therefore raise `PermissionError` during session-product rebuild
- Fixed the helper, not the caller:
  - added guarded path-existence checks that treat unreadable git admin paths as non-repos
  - taught repo-root normalization to ignore direct `.git` admin paths cleanly
- Verification:
  - `ruff check polylogue/lib/repo_identity.py tests/unit/core/test_repo_identity.py`
  - `pytest -q tests/unit/core/test_repo_identity.py`
  - manual probe: `normalize_repo_path('/boot/.git') -> None`
- Note:
  - the already-running timed rebuild still carried the old process image, so this fix needed the next materialize/rebuild pass to prove the live path end-to-end

## 2026-04-12 03:52 CEST

- Timed user-facing probes against the active render stage:
  - `polylogue --format json list -n 1` -> `4.15s`, exit 0
  - `polylogue products profiles --format json --limit 1` -> `2.39s`, exit 0
  - `polylogue stats --format json` -> `29.91s`, exit 0
  - `polylogue doctor --json` -> `40.80s`, exit 0
- Surface interpretation:
  - `list` and `products profiles` were responsive enough during active render
  - `products profiles` reported `count=0` because the running process had already skipped materialization before the `/boot/.git` fix landed
  - `stats` and `doctor` were the remaining obviously slow user-facing reads during a live rebuild
- Decomposed `doctor` cost:
  - `run_archive_health(cfg, deep=False)` -> `22.022s`
  - `run_runtime_health(cfg)` -> `0.135s`
  - inside archive health:
    - `collect_derived_model_statuses_sync(conn)` -> `4.085s`
    - `collect_archive_debt_statuses_sync(conn, ...)` -> `12.271s`
- Timed the `stats` internals directly on the live archive snapshot:
  - `repo.get_archive_stats(conn=conn)` -> `2.687s`
  - `aggregate_message_stats(conn, None)` -> `6.255s`
- Queued a post-run proof step on the fixed code:
  - waiting background session: `50713`
  - script: `.local/logs/polylogue-postrun-materialize-20260412T035200+0200.sh`

## 2026-04-12 14:33 CEST

- Confirmed the archive schema baseline is back to `v1` on the current branch.
- Located the reset in commit:
  - `2d462c15` `fix: reset archive schema baseline to v1`
- This means the earlier `expected version 2` failures belonged to an older branch state, not the current code.
- Verified the current storage constant directly in:
  - `polylogue/storage/backends/schema_ddl.py`

## 2026-04-12 14:40 CEST

- Continued live-archive probing against the normal archive root:
  - `POLYLOGUE_MOTD_RENDERED=1 time -v polylogue --plain doctor --json`
- Result:
  - exit `0`
  - wall `28.57s`
  - max RSS `168528 kB` (~165 MiB)
- Important interpretation:
  - the current live archive is readable again under the `v1` baseline
  - the JSON payload is machine-wrapped as `{"status":"ok","result":...}`
  - `result.checks` and `result.archive_debt` are arrays keyed by `name`, not maps

## 2026-04-12 14:43 CEST

- Verified the action-event maintenance-debt semantics fix on the live archive after committing:
  - `c886b0d1` `fix: clarify action-event maintenance debt accounting`
- Current `doctor --json` surface now reports:
  - `action_event_read_model`: rows ready
  - `action_event_fts`: warning with pending FTS rows
  - `archive_debt.action_event_read_model`: `Action-event read model pending (323,048 pending action-event FTS rows)`
- This removes the previous misleading wording that conflated stale rows with pure FTS backlog.

## 2026-04-12 14:46 CEST

- A new live-surface oddity surfaced during the same `doctor` probe:
  - `index` check reports `messages indexed: 0`
  - while the same archive reports substantial action-event materialization (`4,229` conversations, `323,048` pending action-event FTS rows)
- This is not yet classified as a bug.
- Next probe:
  - determine whether `index` refers only to lexical/vector search readiness, or whether it is incorrectly reporting an empty message index on a populated archive.

## 2026-04-12 14:48 CEST

- The fresh isolated timed rebuild is still running in the background on current code.
- Current visible progress:
  - acquisition reached `10,235` artifacts and was still advancing through Gemini after `8m07s`
- This remains consistent with the earlier large-payload finding:
  - a small number of extremely large raw blobs dominates ingest time
  - the single-record process-pool overhead fix was necessary, but it does not remove the underlying payload-size cost

## 2026-04-12 15:02 CEST

- Probed the live query and products surface on the current archive root:
  - `polylogue --plain list -n 3 --format json`
  - `polylogue --plain stats --format json`
  - `polylogue --plain products profiles --format json --limit 1`
- Findings:
  - `list` JSON shape is coherent and uses `provider`, not `provider_name`
  - archive stats look sane again under the restored `v1` baseline (`10,343` conversations, `1,855,291` messages)
  - session profiles still exposed noisy inferred repo names on live data:
    - `projects`
    - `blob-repository`
- Those came from real git repos on this machine:
  - `~/.config/claude/projects/.git`
  - `/var/lib/sinex/.local/state/sinex/blob-repository/.git`
- Conclusion:
  - plain "walk upward until you see `.git`" is too permissive for attribution
  - transcript-store and state repos should not count as worked-on code repos

## 2026-04-12 15:10 CEST

- Fixed the repo-attribution noise at the normalization layer:
  - transcript-store repos like `~/.claude/projects`
  - local state repos like `.local/state/.../blob-repository`
  - dialogue no longer infers the language `r` from single-letter text noise
- Regression coverage:
  - `tests/unit/core/test_repo_identity.py`
- Verification:
  - `ruff check polylogue/lib/repo_identity.py polylogue/lib/attribution.py tests/unit/core/test_repo_identity.py`
  - `pytest -q tests/unit/core/test_repo_identity.py` -> `9 passed`
- Follow-up proof step:
  - a live `polylogue --plain run materialize` rebuild is running so the existing session-product rows can be regenerated with the new attribution rules

## 2026-04-12 15:13 CEST

- Inspected the isolated full rebuild process table while the fresh run stayed active.
- Confirmed one stale older `run all` launcher had been left behind, with only `tee` still attached.
- Killed that stale launcher and its `tee` child to reduce stray process churn during the campaign.
- The current active isolated run continues under:
  - one live `polylogue --plain run all`
  - one live `polylogue --plain run materialize`

## 2026-04-12 14:20 CEST

- Identified the next major integrated-path defect while vetting fresh isolated rebuilds.
- The previous timed `run all` showed:
  - render/site completed, then action-event indexing crawled at roughly `1 conversation/s`
  - process RSS stayed around `3.8 GB` during that stage
  - earlier stage logs showed render peaking near `6.5 GB RSS`
- Root cause for the indexing portion was structural, not just slow implementation:
  - ingest already writes durable `action_events` rows during conversation save
  - the indexing stage then deleted and rebuilt the same read model from `messages` + `content_blocks` before rebuilding FTS
  - this duplicate work lived in `polylogue/pipeline/services/indexing.py` calling `rebuild_action_event_read_model_async(...)` unconditionally
- Fixed the indexing service so it now:
  - queries action-event repair candidates first
  - skips the action-event rebuild phase entirely when rows are already current
  - only repairs the subset of conversations whose action-event rows are missing or stale
  - reports progress totals from actual work units instead of assuming a full two-phase rebuild every time
- Added regression coverage in `tests/unit/pipeline/test_indexing.py` for:
  - no-action-repair path
  - partial-action-repair path
- Verification:
  - `ruff check polylogue/pipeline/services/indexing.py tests/unit/pipeline/test_indexing.py`
  - `pytest -q tests/unit/pipeline/test_indexing.py`
- Follow-up proof in progress:
  - terminated the stale background rebuild
  - launched a fresh timed isolated `polylogue --plain run all`
  - new launcher session: `18999`
  - new archive root: `/home/sinity/.local/share/polylogue-vetting-20260412T140846+0200/xdg-data/polylogue`
  - new log: `.local/logs/polylogue-run-all-20260412T140846+0200.log`

## 2026-04-12 12:05 CEST

- The stale pre-fix full-run was still alive and holding roughly `2.8 GiB` RSS in the old process tree. Dropped it before collecting any new memory conclusions.
- The earlier detached relaunch attempts that wrote only `.meta` / `.pid` stubs were unreliable and were abandoned.
- Switched the fresh rebuild to a persistent tool session instead:
  - root shell pid `913393`
  - `time` wrapper pid `913397`
  - Python pid `913399`
  - log: `.local/logs/polylogue-run-all-20260412T120242+0200.log`
  - archive root: `/home/sinity/.local/share/polylogue-vetting-20260412T120242+0200/xdg-data/polylogue`
- Early acquisition telemetry from direct sampler probes on the live session:
  - acquisition throughput stayed around `120-150` artifacts/s for `claude-code`
  - RSS stayed around `112-116 MiB` during the sampled early-acquisition window
  - blob-store growth tracked the incoming raw corpus as expected
- Confirmed the codebase already has `SCHEMA_VERSION = 1`; there was no further schema-counter rollback to do in repo code.

## 2026-04-12 12:12 CEST

- Found and fixed two more user-facing CLI defects against the live archive:
  - root `--format json` was ignored by `products`, so `polylogue --format json products profiles --limit 1` rendered plain text
  - `polylogue doctor --runtime --json` still executed the full archive-health path and timed out instead of acting like a narrow runtime probe
- Fixes:
  - `polylogue/cli/commands/products.py` now inherits the root `output_format` when the subcommand does not override it
  - `polylogue/cli/check_workflow.py` now treats `doctor --runtime` as runtime-only when no archive/debt/schema/proof/maintenance work was requested
- Regression coverage added:
  - `tests/unit/cli/test_products.py::test_products_profiles_inherit_root_format_json`
  - `tests/unit/cli/test_check.py::test_check_runtime_only_skips_archive_health`
- Verification:
  - `pytest -q tests/unit/cli/test_products.py -k 'profiles_format_json_alias or inherit_root_format_json'`
  - `pytest -q tests/unit/cli/test_check.py -k 'runtime_only_skips_archive_health'`
  - `ruff check polylogue/cli/commands/products.py polylogue/cli/check_workflow.py tests/unit/cli/test_products.py tests/unit/cli/test_check.py`
  - live probe: `polylogue --format json products profiles --limit 1`
  - live probe: `polylogue doctor --runtime --json`

## 2026-04-12 12:45 CEST

- Verified the real default archive path and schema state:
  - live DB is `~/.local/share/polylogue/polylogue.db`
  - `PRAGMA user_version` is `1`
  - the stale sibling files `archive.db` and `archive.sqlite` also exist under `~/.local/share/polylogue/`, but they are not the canonical runtime DB
- The current isolated fresh rebuild is still progressing normally:
  - at roughly `21m45s` total elapsed it had reached `6,615 / 9,507` raw inputs
  - sampler peak during ingest has been about `1.38 GiB` tree RSS so far, far below the earlier stale multi-gig run
  - the heaviest sampled ingest windows so far have been large Claude Code batches and occasional Codex singletons
- Found a real user-facing repo-attribution defect in the live archive:
  - `polylogue stats --by repo --format json` on the live DB showed bogus repo names like `projects` and `blob-repository`
  - `projects` came from `/home/sinity/.config/claude/projects`, which is itself a git repo on this machine
  - `blob-repository` came from `/var/lib/sinex/.local/state/sinex/blob-repository`
- Root cause:
  - repo attribution was still willing to synthesize repo/file attribution from arbitrary dialogue text paths
  - this let persisted-output and transcript-store paths contaminate repo rollups
  - at the same time, Claude Code parser metadata already provided legitimate `working_directories`, but attribution was ignoring them
- Fix applied:
  - `polylogue/lib/attribution.py` now accepts `provider_meta["working_directories"]` as first-class repo evidence
  - dialogue text no longer manufactures repo or file attribution; it now contributes only low-risk language hints
  - this keeps repo grouping tied to actual working directories, tool `cwd`, tool path inputs, and explicit git metadata
- Coverage added/updated:
  - `tests/unit/core/test_repo_identity.py`
  - `tests/unit/core/test_semantic_facts.py`

## 2026-04-12 13:55 CEST

- Continued the broad repo/user vetting pass from the live partial archive and the non-integration suite.
- Fixed the default `doctor` hot path on large archives:
  - default archive debt checks now skip the orphaned content-block scan unless `--deep` is requested
  - derived-status collection now avoids recomputing retrieval-band readiness
  - action-event status now derives the total source count from the valid/orphan buckets instead of paying for a second full-source count
  - existing archives now gain a partial `content_blocks(type='tool_use')` conversation index
- Verified the live user-facing effect against the active isolated archive:
  - `doctor --json` improved from timing out at 60s to about `23.9s`
  - `list -n 1 --format json` improved from about `14.7s` to about `2.8s`
- Found and fixed two regressions introduced while tightening the read paths:
  - async readers could race a fresh schema bootstrap and hit `no such table: conversations`
  - hybrid search had lost its module-level `open_connection(...)` boundary, which broke both test patching and in-memory sqlite handles
- Found and fixed an attribution regression during session-product rebuild:
  - provider git remotes were normalized once, then normalized again and silently discarded
  - repo names from provider metadata such as `git@github.com:Sinity/sinex.git` now survive rebuild and profile materialization again
  - dialogue fallback language hints no longer infer `r` from arbitrary text like `branch`
- Removed a stale internal test expectation that still referred to grouped stats `project`; the public grouped dimension remains `repo`.
- Full serial verification on the current patch set:
  - `pytest -q -n 0 --ignore=tests/integration` -> `4657 passed`
  - `pytest -q -n 0 tests/integration` -> `170 passed`
  - `direnv exec . devtools render-all --check`
  - `ruff check polylogue tests devtools hatch_build.py`
- Verification:
  - `ruff check polylogue/lib/attribution.py tests/unit/core/test_repo_identity.py tests/unit/core/test_semantic_facts.py`
  - `pytest -q tests/unit/core/test_repo_identity.py tests/unit/core/test_semantic_facts.py -k 'repo or dialogue or action_path_noise or user_path'`
  - `pytest -q tests/unit/storage/test_session_product_profiles.py tests/unit/core/test_repo_identity.py tests/unit/core/test_semantic_facts.py`
  - `pytest -q tests/unit/cli/test_products.py -k 'repo or profiles'`

## 2026-04-12 13:05 CEST

- The isolated fresh rebuild exposed a telemetry defect in ingest-batch observations:
  - per-batch rows were emitting `peak_rss_self_mb`
  - that field came from process-lifetime `VmHWM` / `ru_maxrss`
  - after one high-memory batch, every later batch inherited the same giant peak value
- This made batch-level memory evidence misleading:
  - the live sampler showed the process settling back near `~537 MiB`
  - later batch rows still reported `peak_rss_self_mb=4886.8`
- Fix applied:
  - batch rows no longer overload the lifetime process peak under a batch-local name
  - `polylogue/pipeline/services/ingest_batch.py` now emits:
    - `max_current_rss_mb` for the honest sampled per-batch envelope
    - `process_peak_rss_self_mb` for the lifetime process high-water mark
    - `peak_rss_growth_mb` for how much the batch raised the lifetime high-water mark
  - `polylogue/pipeline/services/parsing_workflow.py` now summarizes `max_peak_rss_growth_mb`
- Coverage added/updated:
  - `tests/unit/pipeline/test_ingest_batch.py`
  - `tests/unit/pipeline/test_parsing_service.py`
- Verification:
  - `pytest -q tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_parsing_service.py`
  - `ruff check polylogue/pipeline/services/ingest_batch.py polylogue/pipeline/services/parsing_workflow.py tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_parsing_service.py`

## 2026-04-12 13:18 CEST

- Followed the fresh-run evidence into the materialize stage.
- The rebuild path was still using fixed-size refresh chunks of `10` conversations in
  `polylogue/storage/session_product_refresh.py`.
- That was too blunt for real archives:
  - one chunk could contain several extremely large conversations simply because the conversation count stayed under ten
  - the fresh run showed slow `load/hydrate/build` batches and materialize-stage RSS climbing well past the lighter ingest steady state
- Fix applied:
  - refresh chunking now consults `conversation_stats.message_count`
  - batches are capped by both:
    - conversation count (`10`)
    - estimated message budget (`5_000`)
  - chunk observations now record:
    - `estimated_message_count`
    - `max_estimated_conversation_messages`
- This keeps the refresh path zero-config while making the materialize stage respond to archive shape instead of just item count.
- Coverage added/updated:
  - `tests/unit/storage/test_session_product_refresh.py`
  - `tests/unit/pipeline/test_ingest_batch.py`
  - `tests/unit/pipeline/test_parsing_service.py`
- Verification:
  - `pytest -q tests/unit/storage/test_session_product_refresh.py tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_parsing_service.py`
  - `ruff check polylogue/storage/session_product_refresh.py polylogue/pipeline/services/ingest_batch.py polylogue/pipeline/services/parsing_workflow.py tests/unit/storage/test_session_product_refresh.py tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_parsing_service.py`

## 2026-04-12 14:20 CEST

- Continued user-style vetting on the archive-wide `stats --format json` surface.
- Confirmed a semantics defect in the structured payload:
  - top-level `summary.attachments` counted `attachment_refs`
  - nested `summary.embeddings.total_attachments` came from archive inventory rows
  - the nested `embeddings` object was also leaking unrelated archive totals under an embeddings-only key
- Fix applied:
  - `polylogue/storage/backends/queries/stats.py` now returns explicit `attachment_refs` and `distinct_attachments`
  - `polylogue/cli/query_stats.py` now exposes those names directly in structured output
  - the nested `embeddings` object now contains only embedding-state fields
  - plain-text stats now print:
    - `Attachment refs: ...`
    - `Unique attachments: ...`
- Coverage added/updated:
  - `tests/unit/storage/test_store_ops.py`
  - `tests/unit/cli/test_query_exec_laws.py`
- Verification:
  - `ruff check polylogue/cli/query_stats.py polylogue/storage/backends/queries/stats.py tests/unit/cli/test_query_exec_laws.py tests/unit/storage/test_store_ops.py`
  - `pytest -q tests/unit/storage/test_store_ops.py -k 'aggregate_message_stats_reports_role_counts_and_words or stats'`
  - `pytest -q tests/unit/cli/test_query_exec_laws.py -k 'output_stats_sql'`
- A live `polylogue stats --format json` probe against the isolated rebuild is still running under concurrent render load to confirm the repaired shape in practice.

## 2026-04-12 14:55 CEST

- Continued the fresh isolated full-run audit after the action-event indexing fix landed.
- The next real bottleneck is now ingest of very large raw blobs, not indexing:
  - current isolated archive has `9512` raw rows
  - `55` rows are at least `128 MiB`
  - `82` rows are at least `64 MiB`
  - worst raw blobs currently observed:
    - `claude-code`: `1535.0 MiB`
    - `codex`: `515.6 MiB`
    - multiple additional `codex` rows in the `190–385 MiB` range
- The live run shows why these records hurt:
  - batches are shaped by `128 MiB` blob budget, so those rows become one-record batches
  - one-record batches were still going through `ProcessPoolExecutor(max_workers=1)`
  - that paid full subprocess setup + pickle + result-transfer cost for zero parallelism
- Fix applied:
  - `polylogue/pipeline/services/ingest_batch.py`
  - `_iter_ingest_results_sync(...)` now runs inline when `worker_count <= 1`
  - multi-worker batches still use the process pool unchanged
- Why this matters:
  - giant one-record batches are the exact place where memory duplication is most wasteful
  - the previous path created a second Python process and pickled large normalized results back to the parent
  - the new path keeps those batches inside the already-offloaded sync ingest thread
- Coverage added:
  - `tests/unit/pipeline/test_ingest_batch.py`
    - verifies single-worker batches do not touch the process-pool path
- Verification:
  - `ruff check polylogue/pipeline/services/ingest_batch.py tests/unit/pipeline/test_ingest_batch.py`
  - `pytest -q tests/unit/pipeline/test_ingest_batch.py`
  - `pytest -q tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_indexing.py tests/integration/test_workflows.py::test_full_workflow_per_provider`
- Live-run status at the time of this note:
  - no kernel OOM events in `journalctl -k`
  - the still-running old-code isolated run reached about `2.0 GiB RSS` in-process during giant-record ingest
  - that run remains useful as evidence, but it is now stale relative to the single-worker ingest fix

## 2026-04-12 15:18 CEST

- Found and fixed a real materialize-stage progress bug while probing the live archive.
- The full rebuild path in `polylogue/storage/session_product_rebuild.py` had gained
  progress reporting, but it was advancing observers by the absolute rebuilt
  profile count instead of the per-chunk delta.
- Symptom in live plain output:
  - `Materializing: 18/10343: 171`
  - the description carried the absolute `18/10343`, but the plain observer had
    already accumulated to `171` because it kept adding absolute totals
- Fix applied:
  - materialize rebuilds now keep absolute counts in `desc`
  - observers advance by the actual processed chunk count
  - full rebuild page size was also reduced to `1` conversation to bound RSS on
    pathological histories
- Coverage:
  - `tests/unit/pipeline/test_run_sources.py`
  - targeted checks:
    - `pytest -q tests/unit/pipeline/test_run_sources.py tests/unit/cli/test_products.py -k 'materialize_stage_rebuilds_all_when_unscoped or materialize_stage_rebuild_progress_advances_by_processed_chunk or session_product_rebuild_pages_full_rebuild or session_product_rebuild_preserves_profile_semantics_without_loading_full_provider_meta'`
    - `ruff check polylogue/pipeline/run_stages.py polylogue/storage/session_product_rebuild.py tests/unit/pipeline/test_run_sources.py tests/unit/cli/test_products.py`
- Commit:
  - `24d99412` `perf: bound materialize rebuild memory and progress`

## 2026-04-12 15:27 CEST

- Found a deeper derived-product correctness issue while probing `products profiles`.
- Earlier repo-attribution cleanup changed session-product inference semantics, but
  `SESSION_PRODUCT_MATERIALIZER_VERSION` had not been bumped, so old rows still
  looked current.
- Evidence:
  - `polylogue --plain products profiles --format json --limit 1` still emitted
    stale inferred repo names like `projects` and `root`
  - direct status probe after bumping the constant reported:
    - `stale_profile_row_count=2734`
    - `stale_work_event_inference_count=3985`
    - `stale_phase_inference_count=2730`
    - `stale_thread_count=8045`
  - `polylogue --plain doctor --json` then correctly surfaced:
    - `session_products` debt: `25,758 pending/stale/orphaned session-product rows`
- Fix applied:
  - bumped `SESSION_PRODUCT_MATERIALIZER_VERSION` from `3` to `4`
  - aligned session-product DDL defaults to `4`
  - added a regression test asserting older materializer versions count as stale
- Coverage:
  - `pytest -q tests/unit/cli/test_products.py -k 'session_product_rebuild_pages_full_rebuild or session_product_rebuild_preserves_profile_semantics_without_loading_full_provider_meta or session_product_status_marks_older_materializer_versions_stale or targeted_session_product_rebuild_does_not_duplicate_profile_fts'`
  - `ruff check polylogue/storage/store_constants.py polylogue/storage/backends/schema_ddl_product_profiles.py polylogue/storage/backends/schema_ddl_product_timelines.py polylogue/storage/backends/schema_ddl_product_aggregates.py tests/unit/cli/test_products.py`
- Commit:
  - `eada263a` `fix: invalidate stale session product rows`

## 2026-04-12 15:40 CEST

- Live `polylogue --plain run materialize` completed on the active archive.
- Measured outcome:
  - wall time: `9m31.34s`
  - max RSS: `3,886,204 kB`
  - structured end RSS: `954.1 MB`
  - structured peak self RSS: `3795.1 MB`
- This is much better than the earlier pre-fix blowup, but still too high for a
  local session-product rebuild and remains active perf debt.
- Post-rebuild product probe still exposed attribution drift:
  - `repo_names` included generic entries like `projects` and `root`
  - `repo_paths` included `/home/sinity/.config/claude/projects`, which is a
    transcript-store repo and should not count as a worked-on repository
- Fix applied in the working tree:
  - ignore config-backed Claude/Codex transcript-store repos during repo-root normalization
  - keep normalized repo names normalized instead of reintroducing raw values
  - add regression coverage for `.config/claude/projects` and normalized bare repo labels

## 2026-04-12 16:10 CEST

- The isolated fresh `polylogue --plain run all` baseline remained useful while
  the repo moved forward: by the render stage it was sitting around `3.2 GiB RSS`
  with the default concurrency policy still active.
- Two concrete runtime problems were confirmed:
  - ingest batches still defaulted to `8` workers even when a batch already
    carried around `100+ MB` of raw payload
  - render stages also defaulted to broad concurrency even when the process had
    already accumulated a high RSS from earlier stages
- Fix applied in the working tree:
  - ingest worker selection now throttles automatically from the batch's blob
    sizes instead of always using the raw default cap
  - render worker selection now backs off automatically when the process enters
    the render stage already above the high-RSS threshold
  - render observations now record worker count plus start/end/max RSS so
    future vetting does not need ad-hoc `ps` sampling
- Verification:
  - `ruff check polylogue/pipeline/services/ingest_batch.py polylogue/pipeline/services/rendering.py polylogue/pipeline/run_stages.py polylogue/pipeline/run_execution.py tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_render_service.py`
  - `pytest -q tests/unit/pipeline/test_ingest_batch.py tests/unit/pipeline/test_render_service.py`
  - `pytest -q tests/unit/pipeline`

## 2026-04-12T16:11:12+02:00
- Search-readiness fix committed as `cefbf796`.
- Fresh isolated `polylogue run all` still running on XDG archive root `~/.local/share/polylogue-vetting-20260412T154222+0200/xdg-data/polylogue`.
- Latest observed progress: ingest `9363/9523` raw at about `28m22s total`.
- Current live shape: large single-record batches around `298-385 MiB` take about `4-5s`; medium `40-100 MiB` batches often take `5-12s` depending on worker count.
- Observed slow writes include very large conversations (for example `msgs=10967`).
- Still seeing malformed JSONL line warnings on some raw records; keep for follow-up after run completes.
- Next concrete bug: `test_update_index_refreshes_action_entries_for_updated_tool_blocks` leaves stale `action_events_fts` rows after tool-block mutation.

## 2026-04-12T16:13:43+02:00
- Reproduced stale action-event reindex bug with `tests/unit/storage/test_fts5.py::test_update_index_refreshes_action_entries_for_updated_tool_blocks`.
- Root cause: `update_index_for_conversations()` filtered changed conversation ids through `action_event_repair_candidates_sync()`, so content-block mutations with unchanged materializer version were skipped.
- Fix: rebuild action-event rows for every conversation explicitly passed to `update_index_for_conversations()`; keep candidate filtering only for full rebuilds.
- Added regression coverage for both tool-block replacement and complete tool-block removal.
- Verification: `108 passed` across `tests/unit/storage/test_fts5.py`, `tests/unit/cli/test_check.py`, and `tests/unit/core/test_query_retrieval_candidates.py`.

## 2026-04-12T16:17:31+02:00
- Live `run all` surfaced a UX flaw: the normal `Ingest complete` log line dumped the full `batch_observations.batches` payload.
- Kept rich batch telemetry for probe/devtools summaries, but trimmed the stage-complete logger payload to aggregate batch stats only.
- Verification: `77 passed` across `tests/unit/pipeline/test_run_sources.py`, `tests/unit/pipeline/test_parsing_service.py`, and `tests/unit/devtools/test_pipeline_probe.py`.

## 2026-04-12T16:19:32+02:00
- Live run surfaced the same telemetry spill on materialize: `Materialize stage complete` logged the full `update_chunks` array.
- Generalized stage-log compaction to drop list-valued diagnostic payloads from normal stage-complete logs while keeping aggregate counters.
- Verified probe/devtools surfaces still retain rich batch telemetry through their persisted summaries.
- Verification: `78 passed` across `tests/unit/pipeline/test_run_sources.py`, `tests/unit/pipeline/test_parsing_service.py`, and `tests/unit/devtools/test_pipeline_probe.py`.

## 2026-04-12T16:29:00+02:00
- `polylogue --plain doctor --repair --preview` on the default live archive still ended with `Would change 2178339 issue(s)`.
- Root cause: the plain maintenance renderer summarized repaired row counts as `issue(s)` even though the underlying repair engine is reporting rows/items, not diagnosed issues.
- Fix applied in the working tree:
  - change the maintenance summary line to `Would apply N change(s)` / `Applied N change(s)`
  - keep the more detailed per-repair `Would:` lines unchanged for now
- Concurrent isolated `run all` remains healthy in render stage at roughly `11/s`; latest observed progress during this slice was `4726/7391` rendered at `40m06s total`.
- Verification:
  - `pytest -q tests/unit/cli/test_check.py -k 'plain_preview_summarizes_changes_not_issues or records_scoped_maintenance_preview or records_scoped_maintenance_apply'`
  - result: `3 passed`
- Committed as `904ca600` with subject `fix: stop calling maintenance changes issues`.
- Fresh isolated `run all` reached render completion during this slice:
  - render elapsed: `10m57s`
  - render throughput: `11.2/s`
  - render started at roughly `2350 MB RSS` and finished around `2348 MB RSS`
  - process peak RSS still reflected earlier materialize pressure: `6941.0 MB`
- The run then moved into site generation.

## 2026-04-12T16:55:00+02:00
- Fresh-archive `doctor --json` and `products profiles` checks exposed two separate durable-product failures after the session-profile materializer bump to `5`:
  - the live builder for `claude-code:6688691a-6417-416b-8554-932e003faab8` was fixed, but stored `session_profiles` rows still surfaced the old polluted payload because list reads never checked product freshness
  - `products profiles --format json --limit 1` returned stale rows with `provenance.materializer_version = 4` instead of forcing a rebuild or surfacing actionable drift
- Root causes confirmed:
  - branch extraction treated `git checkout dots/transmission/settings.json` as a branch change
  - `ToolCall.affected_paths` trusted noisy `metadata.files` tokens from shell/git tool calls and admitted commit-message fragments as paths
  - repo-name attribution still accepted arbitrary path fragments as repo hints, letting tokens like `.snapshots/root` collapse to `root`
- Fixes applied in the working tree:
  - `metadata.files` now pass through a stricter path-shaped filter before becoming affected paths
  - checkout-derived branch names skip pathspec-looking arguments
  - repo hints now come only from explicit repo signals; path-derived repo names must come from actual repo-root normalization
  - session-product materializer version bumped to `5` so stale stored rows are detectable
- Verification:
  - `pytest -q tests/unit/core/test_models.py tests/unit/core/test_semantic_facts.py tests/unit/core/test_repo_identity.py tests/unit/cli/test_products.py -k 'affected_paths or checkout_pathspec or normalization_filters_noise or provider_git_remote or ignores_configured_claude_transcript_repo or session_product_status_marks_older_materializer_versions_stale'`
  - result: `21 passed`

## 2026-04-12T17:02:00+02:00
- Product-surface vetting found a second bug: durable product reads were willing to return stale stored rows even though `session_product_status` already knew those rows were incomplete.
- Added explicit availability guards for all session-product-backed surfaces:
  - profiles / enrichments
  - work events / phases
  - work threads
  - tag rollups
  - day / week summaries
- The guard now fails with an actionable message:
  - `Session-profile rows are incomplete. Run \`polylogue doctor --repair --target session_products\` or \`polylogue run all\`.`
  - queried profile tiers also refuse incomplete FTS state
- User-path verification against the fresh archive:
  - `polylogue --plain products profiles --format json --limit 1`
  - now returns a structured error envelope instead of silently serving stale rows
- Regression coverage added for stale durable products plus incomplete merged profile FTS.
- Targeted verification:
  - `pytest -q tests/unit/cli/test_products.py -k 'stale_session_product_surfaces or incomplete_profile_search_index or session_product_status_marks_older_materializer_versions_stale'`
  - result: `9 passed`
- Fresh-archive follow-up:
  - `polylogue --plain doctor --repair --target session_products` is still under observation as a live UX/perf path; the command remains too silent while running, which is itself notable vetting feedback.

## 2026-04-12T17:32:00+02:00
- The live `doctor --repair --target session_products` follow-up completed successfully against the fresh vetting archive after the sync repair path was wired to emit progress updates.
- The command no longer looks hung:
  - stderr now streams `Repairing session products: Materializing: N ...` progress continuously during the rebuild
  - the observed throughput ramped from effectively `0/s` at startup to roughly `20/s` once the rebuild settled
- Post-repair product read now succeeds:
  - `polylogue --plain products profiles --format json --limit 1`
  - envelope shape is `{status, result}`, with `result.session_profiles`
  - the previously polluted conversation `claude-code:6688691a-6417-416b-8554-932e003faab8` now reads back with `repo_names = null`, `branch_names = null`, and `file_paths_touched = []` instead of the earlier garbage payload
- This closes the immediate stale-session-product UX loop:
  - stale rows are rejected before repair
  - repair advertises visible progress while rebuilding
  - repaired rows read back cleanly afterward

## 2026-04-12T17:40:00+02:00
- A broader user-level smoke pass against the repaired archive surfaced three more issues:
  - `stats --by provider` only works when root flags are moved before the verb, while other surfaces like `list --format json --limit 3` tolerate post-verb output flags; the command family is still inconsistent from a user point of view
  - `products profiles --format json --limit 1` now succeeds, but the repaired `session_profiles` payload is still noisy in `evidence.file_paths_touched`
  - `open --print-path <conversation-id>` fails with `Got unexpected extra argument`, which is not a natural user experience for opening a known conversation directly
- Concrete payload evidence from the repaired archive:
  - `claude-code:6688691a-6417-416b-8554-932e003faab8` still carries many low-value path entries in `evidence.file_paths_touched`, including local transcript-store files under `/home/sinity/.config/claude/...`, temp task output files under `/tmp/claude-1000/...`, and snapshot/cache paths like `.snapshots/root`
  - the earlier cleanup fixed `branch_names` and repo-hint pollution, but not the wider file-path evidence pollution
- Measured smoke timings from the repaired archive:
  - `polylogue --plain --format json --limit 10 stats --by provider` → `37.51s`, `114936 KB`
  - `polylogue --plain products profiles --format json --limit 1` → `9.55s`, `121440 KB`
  - `polylogue --plain products work-events --format json --limit 1` → `4.28s`, `122148 KB`
  - `polylogue --plain products day-summaries --format json --limit 1` → `3.32s`, `122976 KB`
- Next vetting focus:
  - trace how `evidence.file_paths_touched` is still admitting transcript-store, temp, and snapshot paths
  - then revisit the command-surface inconsistencies (`stats` flag ordering, `open` positional UX)

## 2026-04-12T18:03:11+02:00
- Session-profile evidence attribution was tightened further and revalidated live against the fresh vetting archive.
- First pass:
  - filtered transcript-store paths under `~/.config/claude/projects/...`
  - filtered temp task-output paths under `/tmp/claude-*` and `/tmp/codex-*`
  - filtered snapshot/cache paths like `.snapshot`, `.snapshots`, `.btrfs/snapshot`
  - filtered Nix-store-resolved paths
  - bumped `SESSION_PRODUCT_MATERIALIZER_VERSION` to `7`
- Live validation after the version-`7` rebuild showed one remaining leak class:
  - repo-local agent config paths such as `/realm/project/sinnix/.claude/settings.json` were still preserved because repo-root normalization returned early
- Final fix:
  - repo-relative `.claude/` and `.codex/` paths are now filtered before repo-root paths are accepted
  - bumped `SESSION_PRODUCT_MATERIALIZER_VERSION` again to `8`
- Final live result after rebuilding session products under version `8`:
  - `polylogue --plain products profiles --format json --limit 1` now reports `count = 56`
  - targeted leak checks are all false:
    - `has_claude_projects = false`
    - `has_tmp_claude = false`
    - `has_snapshot = false`
    - `has_nix_store = false`
    - `has_claude_home = false`
  - remaining sample paths are meaningful repo/system references like `/etc/systemd/system/sinex-gateway.service`, `/realm/project/sinex/...`, `/realm/project/sinnix/dots/claude`
- Measured rebuild cost on the fresh archive after the final bump:
  - log: `.local/logs/session-products-repair-v8-20260412T175516+0200.log`
  - wall: `6:53.00`
  - max RSS: `4,011,736 kB`
  - CPU: `81%`
- Conclusion:
  - the attribution cleanup is correct for the intended noise classes
  - session-product rebuild memory is still heavier than it should be for a read-model repair and should stay on the broader vetting/perf list

## 2026-04-12T18:07:12+02:00
- The `open --print-path <conversation-id>` UX defect is fixed.
- Root cause:
  - `open` accepted trailing tokens after the verb, but treated them only as free-text query terms
  - direct conversation IDs like `claude-code:6688691a-6417-416b-8554-932e003faab8` therefore became a search string instead of an exact conversation-id selection
- Fix:
  - `open` now accepts trailing target terms after the verb
  - when there is exactly one trailing `provider:id` token and no pre-verb query terms, it is routed into the exact `conv_id` filter instead of full-text query text
  - ordinary free-text query behavior remains unchanged when using pre-verb terms or non-ID trailing text
- Verification:
  - `pytest -q tests/integration/test_cli_query_mode.py -k 'open_print_path'` → `3 passed`
  - live archive repro now succeeds:
    - `polylogue --plain open --print-path claude-code:6688691a-6417-416b-8554-932e003faab8`
    - returned the concrete render path instead of `Got unexpected extra argument`

## 2026-04-12T18:20:00+02:00
- The `stats` verb now accepts `--limit` after the verb, matching the rest of the query-first CLI more naturally.
- Root cause:
  - `stats` exposed `--format` locally but still relied on the root command for `--limit`
  - users could write `list --limit 3` naturally, but `stats --by provider --limit 10 --format json` failed unless `--limit` was moved before `stats`
- Fix:
  - `stats` now defines a local `--limit / -n` option and forwards it into the shared query params before execution
  - root-level limit behavior remains intact
- Verification:
  - `pytest -q tests/integration/test_cli_query_mode.py -k 'open_print_path or stats_by_provider_accepts_limit_after_verb'` → `4 passed`
  - `ruff check polylogue/cli/query_verbs.py tests/integration/test_cli_query_mode.py`
  - `git diff --check`
- Live archive repro now succeeds:
  - `polylogue --plain stats --by provider --limit 10 --format json`
  - wall: `8.38s`
  - max RSS: `115,816 kB`
  - returned the grouped JSON payload directly instead of forcing root-option reordering

## 2026-04-12T18:34:00+02:00
- The zsh completion surface is now archive-backed instead of being only static command/option scaffolding.
- Added live completions for:
  - `--id` recent conversation IDs, with provider and title descriptions
  - `open <target>` direct conversation-ID targets
  - `--tag` / `--exclude-tag` from archive tag counts
  - `--tool` / `--exclude-tool` from archive action-tool counts
  - comma-aware provider completion with descriptions so `--provider claude-ai,c<TAB>` expands sensibly
- Important UX detail:
  - conversation IDs complete on the bare suffix as well as the stored `provider:id` form, so typing `conv-...` is enough even though the archive stores `unknown:conv-...`
- Verification:
  - `pytest -q tests/integration/test_cli_query_mode.py -k 'completion or open_print_path or stats_by_provider_accepts_limit_after_verb'` → `8 passed`
  - `ruff check polylogue/cli/shell_completion_values.py polylogue/cli/click_option_groups.py polylogue/cli/query_verbs.py tests/integration/test_cli_query_mode.py`
- Public discoverability:
  - README now includes a concise zsh installation snippet for `polylogue completions --shell zsh`

## 2026-04-12T18:52:00+02:00
- A live command-surface inconsistency surfaced during the fresh-archive smoke pass:
  - `polylogue --plain tags --format json`
  - previously failed with the machine envelope:
    - `status = error`
    - `code = invalid_arguments`
    - `message = No such option: --format`
- Fix:
  - `tags` now accepts `--format json` in addition to the older `--json` flag
  - generated CLI reference and the committed `help-tags` showcase baseline were updated with the new option
- Verification:
  - `pytest -q tests/integration/test_cli_tags_surface.py` → `1 passed`
  - `ruff check polylogue/cli/commands/tags.py tests/integration/test_cli_tags_surface.py`
  - live fresh-archive repro now succeeds:
    - `polylogue --plain tags --format json`
    - returned `{status: ok, result: {tags: {}}}` on the fresh archive instead of a usage error

## 2026-04-12T18:36:37+02:00
- Reset the default live archive path in place under `~/.local/share/polylogue/`.
- The root directory itself could not be renamed because it was reported busy, so the active contents were bucketed inside the same root instead.
- New backup bucket:
  - `~/.local/share/polylogue/reset-backup-20260412T183637+0200`
- Reason:
  - the default archive root had accumulated multiple overlapping backup files, rerun artifacts, and a stale `v1` database
  - starting fresh on the canonical default path is clearer than continuing to vet against another isolated XDG root
- Relevant inspection before reset:
  - `polylogue.db` was present at `~9.5G`
  - `PRAGMA user_version` returned `1`
  - the root contained prior ad-hoc backup and rerun artifacts dating from earlier same-day resets

## 2026-04-12T18:36:55+02:00
- Started a full rebuild on the canonical default path with explicit timing and a persistent log:
  - command:
    - `/run/current-system/sw/bin/time -f '\n[time] wall=%E rss_kb=%M cpu=%P' polylogue --plain run all |& tee .local/logs/run-all-default-20260412T183655+0200.log`
  - log:
    - `.local/logs/run-all-default-20260412T183655+0200.log`
- Early observations:
  - acquisition on `claude-code` started around `340-380/s`
  - acquisition on `codex` slowed into roughly `125-220/s`
  - acquisition on `gemini` slowed much further into roughly `18-42/s`
  - empty-artifact skips surfaced but did not fail the run:
    - `59` for `claude-code`
    - `3` for `codex`

## 2026-04-12T18:46:00+02:00
- The rebuild exposed a concrete CLI UX defect even before completion:
  - long acquisition gaps looked indistinguishable from a hung process
  - there was a multi-minute silent window between the end of visible `codex` scanning and the next `gemini` progress line
- The process itself was still healthy during the silent period:
  - observed running `polylogue --plain run all` process at roughly `24-30%` CPU
  - observed RSS stayed around `135 MB`, so the run was not obviously wedged or memory-explosive during this phase
- Root cause candidate from code inspection:
  - acquisition progress is emitted only after a raw artifact is yielded back to the async side
  - one slow file can therefore monopolize the stage with no heartbeat visible to the user
- Likely fix direction:
  - add an acquisition heartbeat that surfaces liveness while a single slow artifact is still being read or decoded, without faking item counts

## 2026-04-12T18:52:00+02:00
- Implemented the acquisition heartbeat fix.
- Change summary:
  - blob-store streaming writes now accept an optional heartbeat callback
  - source acquisition now emits throttled status heartbeats while a slow file or ZIP entry is still being streamed
  - the async acquisition stream forwards those status heartbeats back onto the event-loop thread as ordinary `progress_callback(0, desc=...)` updates
- Intended user-visible effect:
  - `polylogue --plain run all` should no longer look frozen for minutes while one slow artifact is still being read
  - the progress line should retain the real acquired-item count while swapping in a more specific `Scanning [...] reading ...` description during slow-file windows
- Verification:
  - `ruff check polylogue/storage/blob_store.py polylogue/sources/source_acquisition.py polylogue/pipeline/services/acquisition.py polylogue/pipeline/services/acquisition_streams.py tests/unit/storage/test_blob_store.py tests/unit/pipeline/test_acquisition_streams.py tests/unit/sources/test_source_laws.py`
  - `pytest -q tests/unit/storage/test_blob_store.py tests/unit/pipeline/test_acquisition_streams.py tests/unit/sources/test_source_laws.py -k 'iter_raw_record_stream_forwards_source_status_progress or write_from_fileobj_invokes_heartbeat or iter_source_raw_data_skips_zero_byte_plain_files or iter_source_raw_data_summarizes_zero_byte_plain_files or iter_source_raw_data_tracks_read_failures_without_stopping or iter_source_raw_data_summarizes_zero_byte_zip_entries'`
  - result: `6 passed`

## 2026-04-12T18:53:00+02:00
- The default-path rebuild kept running while the heartbeat fix was being validated.
- Additional observations from the live run:
  - after acquisition crossed into parsing, the process RSS had climbed to about `828,560 kB`
  - parse throughput was much lower than the raw scan rates suggested; at one snapshot it was only `716/9,538 raw` after roughly `13m53s`
- This keeps two follow-up questions live:
  - whether parse/materialize throughput is lower than it should be for the current corpus
  - whether the run needs clearer per-stage timing and stronger explanations of where time is going

## 2026-04-12T19:07:00+02:00
- Fixed a real JSONL robustness bug in the validation/sampling path.
- Root cause:
  - `sample_jsonl_payload()` used `orjson.loads()` only
  - the streaming JSONL decoder used stdlib `json.loads()`
  - some real Claude Code lines contain escaped lone-surrogate sequences like `\\udce2`
  - stdlib accepts them, while `orjson` rejects them as `invalid low surrogate in string`
  - strict validation therefore marked otherwise parseable session files as malformed
- Fix:
  - keep `orjson` as the fast path
  - fall back to stdlib `json.loads()` only for individual JSONL lines that `orjson` rejects
- Verification:
  - `pytest -q tests/unit/core/test_raw_payload_decode.py` → `1 passed`
  - `ruff check polylogue/lib/raw_payload_decode.py tests/unit/core/test_raw_payload_decode.py`
  - real-file repro:
    - `sample_jsonl_payload('/home/sinity/.claude/projects/-realm-project-sinex.pre-enrich/b8c8d990-f5c4-4d01-881a-f4af42ceb7f2.jsonl', max_samples=64, jsonl_dict_only=True)`
    - now returns `malformed = 0`
- Important boundary:
  - this does **not** replace `orjson` globally
  - it only widens tolerance on the exact line-level JSONL cases where `orjson` is stricter than the stream parser

## 2026-04-12T19:30:00+02:00
- The live default-path rebuild was still running and was **not** OOM-killed.
- Verification:
  - kernel log check for the previous two hours showed no OOM events
  - the main `polylogue --plain run all` process stayed alive throughout inspection
- Key process observations:
  - late-run steady RSS: about `2.4 GiB`
  - process `VmHWM`: about `7.1 GiB`
- Stage summaries pinned the real peak to materialization, not rendering:
  - acquire:
    - `rss_start_mb=77.9`
    - `rss_end_mb=192.6`
    - `peak_rss_self_mb=196.6`
  - ingest:
    - `rss_start_mb=192.7`
    - `rss_end_mb=1022.0`
    - `peak_rss_self_mb=5623.8`
    - `details.batch_observations.max_current_rss_mb=3830.3`
  - materialize:
    - `rss_start_mb=1022.0`
    - `rss_end_mb=2329.6`
    - `peak_rss_self_mb=6942.6`
    - `rebuilt=False`
  - render:
    - `rss_start_mb=2330.9`
    - `rss_end_mb=2339.7`
    - `max_current_rss_mb=2379.4`
    - `peak_rss_self_mb` remained `6942.6`
- Conclusion:
  - render is not the heap explosion
  - the main memory pathology is the materialize path chosen by `run all`
  - render mostly inherits the already-inflated heap and adds little more

## 2026-04-12T19:34:00+02:00
- Found a dispatch-level cause for the materialize peak:
  - `execute_materialize_stage(stage in {"all", "reprocess"})` always used `refresh_session_products_bulk(...)`
  - this bypassed the already-existing bounded rebuild path, even when the archive had zero existing session-product rows
- This was inconsistent with the current architecture:
  - `rebuild_session_products_async(...)` already had `_SESSION_PRODUCT_REBUILD_PAGE_SIZE = 1`
  - that path was explicitly documented as the safe path for pathological historical payloads
- Fix:
  - when `profile_row_count == 0` and `total_conversations == len(processed_ids)`, the dispatcher now uses the bounded rebuild path
  - the returned observation is marked with `mode = "rebuild-from-empty"`
- Verification:
  - `ruff check polylogue/pipeline/run_stages.py tests/unit/pipeline/test_run_sources.py`
  - `pytest -q tests/unit/pipeline/test_run_sources.py -k 'materialize_stage_rebuild_progress_advances_by_processed_chunk or all_stage_uses_bounded_rebuild_when_products_are_empty'`
  - result: `2 passed`

## 2026-04-12T19:38:00+02:00
- Another user-facing flaw surfaced during the live run:
  - site build exposes only `Building site...: 0`
  - no incremental site progress appears afterward, even while CPU remains busy
- This is a real UX defect independent of memory:
  - the process looks stalled
  - there is no total or subphase visibility

## 2026-04-12T19:56:00+02:00
- The site stage was not hung; it was just silent.
- Later log evidence showed:
  - `Site stage complete`
  - `elapsed_ms=367822.6`
  - `conversations=7392`
  - `rendered_pages=552`
  - `index_pages=5`
  - `rss_start_mb=2339.8`
  - `rss_end_mb=2653.3`
  - `rss_delta_mb=313.5`
- After site completion, the live run moved into FTS indexing and resumed visible progress:
  - `Indexing: full-text search 0/7,392`
  - then 500-row progress increments
- Conclusion:
  - the silent period was a pure progress-plumbing defect
  - site work was real and bounded, but invisible for about 7.5 minutes

## 2026-04-12T20:02:00+02:00
- Landed commit `bdb6b171` (`fix: bound fresh materialization and report site progress`).
- Scope:
  - use bounded session-product rebuild during `run all` / `reprocess` when the archive has no existing session-product rows
  - thread site-build progress callbacks through the actual site builder, so plain runs emit scan/manifold subphase activity instead of a frozen `Building site...: 0`
- Verification:
  - `ruff check polylogue/pipeline/run_stages.py polylogue/site/builder.py polylogue/site/scan.py polylogue/site/site_builder_archive.py tests/unit/pipeline/test_run_sources.py tests/unit/site/test_builder.py`
  - `pytest -q tests/unit/pipeline/test_run_sources.py tests/unit/site/test_builder.py`
  - result: `49 passed`

## 2026-04-12T20:17:00+02:00
- Fixed the `run --reparse --preview ...` mutation bug.
- Root cause:
  - CLI preview planning called `reset_parse_status()` before building the preview plan
  - that made `--preview` destructive
  - the planner also had no way to simulate reparse semantics without mutating the DB first
- Fix:
  - add `force_reparse` to preview planning
  - simulate reset semantics inside backlog collection and planning instead of touching persisted state
  - defer the real `reset_parse_status()` call until execution actually begins
- Added tests for:
  - preview mode not calling the reset path
  - parse-stage planning under simulated reparse
  - `run all` preview correctly counting already-validated scanned raws under simulated reparse
- Verification:
  - `ruff check polylogue/cli/commands/run.py polylogue/pipeline/run_planning.py polylogue/pipeline/services/planning.py polylogue/pipeline/services/planning_backlog.py polylogue/pipeline/services/planning_runtime.py tests/unit/cli/test_run.py tests/unit/pipeline/test_parsing_service.py`
  - `pytest -q tests/unit/cli/test_run.py tests/unit/pipeline/test_parsing_service.py`
  - result: `55 passed`

## 2026-04-12T20:20:00+02:00
- Live probe after the fix:
  - command: `polylogue --plain run --source claude-code --reparse --preview parse`
  - result:
    - `Reparse requested.`
    - `Planning preview...`
    - `Sources: claude-code`
    - `Work: 7429 parse`
    - `State: 7429 parse backlog`
    - wall time about `2.31s`
    - RSS about `86 MiB`
- Verified that the preview no longer mutates the DB during the probe.
- Important context:
  - the live archive's `parsed_at` state was already globally cleared by the earlier buggy preview behavior before this fix landed
  - so the current archive state now needs an explicit repair/reparse pass; the new probe confirmed only that the fix prevents further damage

## 2026-04-12T20:28:00+02:00
- Tightened actual `--reparse` execution scope.
- Before this change:
  - `polylogue run --source claude-code --reparse ...` still cleared parse tracking for the entire archive
  - only the subsequent run stages were source-scoped
- Fix:
  - raw-state reset helpers now accept `source_names`
  - CLI passes the selected sources into `reset_parse_status(...)`
  - validation-state reset helpers were kept symmetric and source-scoped too
- Verification:
  - `ruff check polylogue/storage/backends/queries/raw_state.py polylogue/storage/repository_raw.py polylogue/storage/backends/async_sqlite_raw.py polylogue/cli/commands/run.py tests/unit/storage/test_parse_tracking.py tests/unit/cli/test_run.py`
  - `pytest -q tests/unit/storage/test_parse_tracking.py tests/unit/cli/test_run.py`
  - result: `59 passed`

## 2026-04-12T20:33:00+02:00
- Started explicit parse-state repair run to undo the historical global reset caused by the old preview bug:
  - command: `polylogue --plain run parse`
  - log: `.local/logs/run-parse-repair-20260412T194542+0200.log`
- Early repair-run observations:
  - current live RSS around `3.0 GiB` after about 4 minutes
  - no sign of the earlier `run all` materialization blow-up
  - throughput is heavily shaped by a small set of giant singleton batches
- Largest raw artifacts in the archive right now:
  - Claude Code: `1535.0 MiB` (`/home/sinity/.claude/projects/-realm-project-sinnix/700cfc8b-6125-44f8-a83e-ac593895ea4b.jsonl`)
  - Codex: `515.6 MiB` (`2025/12/24/...019b50a3...jsonl`)
  - several more Codex sessions between about `277 MiB` and `385 MiB`
- Interpretation:
  - parse runtime is now dominated by a small set of pathological giant sessions
  - the current batching logic is correctly collapsing them to single-record batches, but that still leaves very long per-record wall time

## 2026-04-12T20:35:00+02:00
- Quick completions probe:
  - `polylogue completions --shell zsh` currently emits the standard Click zsh wrapper
  - the wrapper already delegates dynamic value completion back into `polylogue`
  - this means the remaining completion work is about richer values/descriptions/UX, not the absence of shell integration

## 2026-04-12T21:12:00+02:00
- Investigated why `polylogue --plain doctor --json` was still much slower than the warm in-process health timings.
- Important correction:
  - the earlier `run_archive_health(..., deep=False) ~4.9s` number was a warm in-process measurement after earlier probes
  - a fresh-process measurement was still much slower, so the remaining latency was real
- Fresh-process timings before the latest probe-mode cut:
  - `polylogue --plain doctor --json`
    - wall `16.56s`
    - RSS `122148 kB`
  - `run_archive_health(get_config(), deep=False)`
    - wall `18.84s`
    - RSS `95504 kB`
- Conclusion:
  - the remaining default-latency problem was still inside archive health itself, not in JSON rendering or Click glue

## 2026-04-12T21:18:00+02:00
- Broke the cold health cost down into fresh-process components.
- Fresh-process timings:
  - shallow message FTS readiness: `6.27s`
  - shallow session-product status: `3.53s`
  - shallow derived-status collection: `2.59s`
  - shallow archive-debt collection: `7.78s`
- Follow-up breakdown inside archive debt:
  - `count_orphaned_messages_sync`: `6.10s`
  - `count_empty_conversations_sync`: `0.25s`
  - `count_orphaned_attachments_sync`: `0.01s`
- Interpretation:
  - default `doctor` was still paying for exact counts in two places that do not need exact counts for the everyday UX path:
    - exact `messages_fts_docsize` counting
    - exact orphaned-message counting

## 2026-04-12T21:26:00+02:00
- Implemented probe-mode health behavior for the normal `doctor` path.
- Scope:
  - ordinary `doctor` now uses probe semantics unless running `--deep`, `--repair`, or `--cleanup`
  - deep and maintenance paths keep exact counts
  - shallow index checks no longer count all indexed FTS rows
  - shallow orphaned-message checks no longer count all orphans
  - shallow derived `messages_fts` detail now reports presence rather than fake `0/0` counts
- Verification:
  - `ruff check polylogue/storage/fts_lifecycle.py polylogue/storage/repair.py polylogue/storage/derived_status.py polylogue/storage/derived_status_products.py polylogue/health.py polylogue/cli/check_workflow.py tests/unit/cli/test_check.py tests/unit/core/test_health_core.py tests/unit/storage/test_derived_status.py`
  - `pytest -q tests/unit/cli/test_check.py tests/unit/core/test_health_core.py tests/unit/storage/test_derived_status.py`
  - result: `62 passed`
- Fresh live timing after the cut:
  - `polylogue --plain doctor --json`
    - wall `12.71s`
    - RSS `123092 kB`
- Net effect:
  - default doctor latency improved by about `3.85s`
  - the command is still slower than it should be, but the most gratuitous exact-count work is now gone from the normal path
## 2026-04-13 06:34 CEST — sync/async schema bootstrap unification

- inspected duplicated bootstrap logic in:
  - `polylogue/storage/backends/schema_upgrade.py`
  - `polylogue/storage/backends/async_sqlite_schema.py`
- observed real semantic drift:
  - async path had weaker current-version extension coverage
  - sync path owned the richer `session_profiles` backfill/update sequence
  - mismatch messaging also differed
- introduced shared bootstrap planning in:
  - `polylogue/storage/backends/schema_bootstrap.py`
- moved both backends onto the same model:
  - shared snapshot capture
  - shared current-version extension plan
  - shared schema mismatch message
  - shared sync/async plan application helpers
- preserved backend-specific execution differences only where necessary:
  - sync vs async SQL execution
  - vec0 helper remains dual entrypoints
- added async regression coverage proving the previously weaker path now applies current-version profile extensions and raw index repair semantics

Verification:

```bash
ruff check polylogue/storage/backends/schema_bootstrap.py polylogue/storage/backends/schema_upgrade.py polylogue/storage/backends/async_sqlite_schema.py tests/unit/storage/test_backend.py tests/unit/storage/test_parse_tracking.py
pytest -q tests/unit/storage/test_backend.py tests/unit/storage/test_parse_tracking.py -k 'ensure_schema or legacy_inline_raw_layout or legacy_v1 or source_mtime_index or async_backend_applies_current_session_profile_extensions'
pytest -q tests/unit/storage/test_backend.py
```

Results:

- focused schema tests: `9 passed in 11.30s`
- full backend unit file: `24 passed in 11.19s`

## 2026-04-13 07:05 CEST — generated exercises compiled from scenarios

- moved generated showcase authoring up one level:
  - introduced `polylogue/showcase/scenario_models.py`
  - `ExerciseScenario` is now the authored root for generated CLI-backed showcase items
- refactored generated help and generated JSON-contract exercises to be authored as scenarios first, then compiled into `Exercise`
- preserved outward showcase/runtime behavior while reducing one semantic root:
  - `Exercise` remains the execution artifact
  - generated exercises are no longer authored directly as `Exercise`

Verification:

```bash
ruff check polylogue/showcase/scenario_models.py polylogue/showcase/generators.py tests/unit/showcase/test_scenario_models.py tests/unit/showcase/test_exercise_catalog.py
pytest -q tests/unit/showcase/test_scenario_models.py tests/unit/showcase/test_exercise_catalog.py
```

Results:

- showcase scenario tests: `24 passed in 11.23s`

## 2026-04-13 07:18 CEST — benchmark campaigns compiled from scenarios

- introduced `BenchmarkScenario` as the authored root for durable benchmark campaigns in `devtools/benchmark_campaign.py`
- replaced the hand-built campaign registry with scenario compilation:
  - `BENCHMARK_SCENARIOS` -> `compile_benchmark_scenarios(...)` -> `CAMPAIGNS`
- kept public campaign names/descriptions/tests stable while moving authorship upward

Verification:

```bash
ruff check devtools/benchmark_campaign.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
pytest -q tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
```

Results:

- benchmark/devtools tests: `9 passed in 8.39s`

## 2026-04-13 07:29 CEST — scenario metadata preserved into showcase and benchmark artifacts

- preserved scenario provenance/targets/tags instead of flattening them away during compilation/serialization
- showcase side:
  - `Exercise` and `ExerciseScenario` now carry scenario metadata through compiled exercises and JSON session/report payloads
  - report serialization now tolerates legacy/mock exercise objects by normalizing metadata through a shared boundary
- benchmark side:
  - benchmark campaigns and durable result artifacts now preserve origin/artifact targets/operation targets/tags

Verification:

```bash
ruff check polylogue/showcase/exercise_models.py polylogue/showcase/scenario_models.py polylogue/showcase/catalog_loader.py polylogue/showcase/showcase_report_payloads.py tests/unit/showcase/test_scenario_models.py tests/unit/cli/test_qa.py tests/unit/showcase/test_report.py
pytest -q tests/unit/showcase/test_scenario_models.py tests/unit/cli/test_qa.py tests/unit/showcase/test_report.py
ruff check devtools/benchmark_campaign.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
pytest -q tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
```

Results:

- showcase/report tests: `21 passed in 11.74s`
- benchmark/devtools tests: `9 passed in 8.51s`

## 2026-04-13 07:42 CEST — shared scenario metadata root

- introduced shared scenario metadata model:
  - `polylogue/scenarios/metadata.py`
- moved duplicated metadata semantics behind one boundary:
  - `ScenarioMetadata.from_payload(...)`
  - `ScenarioMetadata.from_object(...)`
  - `ScenarioMetadata.to_payload(...)`
- rewired both projections to use it:
  - showcase exercise/scenario/catalog/report
  - benchmark campaign definitions/results
- this removed the repeated local coercion/serialization logic and gave both projections one source of truth for:
  - `origin`
  - `artifact_targets`
  - `operation_targets`
  - `tags`
- added direct unit coverage for metadata coercion and mock-safe fallback behavior

Verification:

```bash
ruff check polylogue/scenarios polylogue/showcase/exercise_models.py polylogue/showcase/scenario_models.py polylogue/showcase/catalog_loader.py polylogue/showcase/showcase_report_payloads.py devtools/benchmark_campaign.py tests/unit/scenarios/test_metadata.py tests/unit/showcase/test_scenario_models.py tests/unit/showcase/test_exercise_catalog.py tests/unit/showcase/test_report.py tests/unit/cli/test_qa.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
pytest -q tests/unit/scenarios/test_metadata.py tests/unit/showcase/test_scenario_models.py tests/unit/showcase/test_exercise_catalog.py tests/unit/showcase/test_report.py tests/unit/cli/test_qa.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
```

Results:

- unified scenario-metadata suite: `55 passed in 12.22s`

## 2026-04-13 07:49 CEST — quality registry preserves benchmark metadata

- the shared quality registry had still been flattening benchmark scenario semantics away
- preserved benchmark metadata through `devtools/quality_registry.py` by making `BenchmarkCampaignEntry` carry shared scenario metadata as well
- this keeps the control-plane registry usable for future coverage-map and provenance work without forcing a docs/regeneration slice yet

Verification:

```bash
ruff check devtools/quality_registry.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
pytest -q tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py tests/unit/devtools/test_benchmark_campaign.py
```

Results:

- registry/render tests: `9 passed in 8.41s`

## 2026-04-13 08:03 CEST — artifact graph grows explicit operations

- the artifact graph already modeled the two proven vertical paths, but it still lacked explicit operations
- added `ArtifactOperation` to `polylogue/artifact_graph.py`
- modeled operations for both paths:
  - `plan-validation-backlog`
  - `plan-parse-backlog`
  - `materialize-action-events`
  - `index-action-events`
  - `project-action-event-health`
- extended `devtools/artifact_graph.py` so the control-plane view now renders both:
  - artifact paths
  - artifact operations
- this is the first concrete step toward the earlier `OperationSpec` direction without inventing a parallel framework yet
- importantly, it exercised the raw-validation path and the action-event path under the same graph model

Verification:

```bash
ruff check polylogue/artifact_graph.py devtools/artifact_graph.py tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py tests/unit/storage/test_raw_ingest_artifacts.py
pytest -q tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py tests/unit/storage/test_raw_ingest_artifacts.py
```

Results:

- artifact graph + raw-ingest law tests: `8 passed in 9.44s`

## 2026-04-13 08:17 CEST — first real OperationSpec substrate

- extracted authored runtime operations out of `polylogue/artifact_graph.py`
- introduced a real substrate in:
  - `polylogue/operations/specs.py`
- added:
  - `OperationKind`
  - `OperationSpec`
  - `RUNTIME_OPERATION_SPECS`
  - `build_runtime_operation_specs()`
- moved the five currently-proven operations there:
  - raw validation backlog planning
  - raw parse backlog planning
  - action-event materialization
  - action-event indexing
  - action-event health projection
- rewired `polylogue/artifact_graph.py` so the graph now consumes authored runtime operation specs instead of defining operations itself
- extended the control-plane rendering to expose operation kind
- added direct unit coverage for the operation substrate and graph/spec agreement

Verification:

```bash
ruff check polylogue/operations/specs.py polylogue/operations/__init__.py polylogue/artifact_graph.py devtools/artifact_graph.py tests/unit/operations/test_specs.py tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py tests/unit/storage/test_raw_ingest_artifacts.py
pytest -q tests/unit/operations/test_specs.py tests/unit/core/test_artifact_graph.py tests/unit/devtools/test_artifact_graph.py tests/unit/storage/test_raw_ingest_artifacts.py
```

Results:

- operation-spec + graph suite: `12 passed in 9.39s`

## 2026-04-13 08:31 CEST — synthetic benchmark campaigns become authored scenarios

- the repo still had a second benchmark root in `devtools/benchmark_campaigns.py`:
  - free `CAMPAIGN_REGISTRY` dict
  - match-based dispatcher
  - no shared scenario metadata
- replaced that with authored synthetic benchmark scenarios:
  - `SyntheticBenchmarkScenario`
  - `SYNTHETIC_BENCHMARK_SCENARIOS`
  - `SYNTHETIC_BENCHMARK_REGISTRY`
  - `run_synthetic_benchmark_campaign(...)`
- `run_full_campaign(...)` now iterates over authored scenarios instead of hard-coded repeated blocks
- `devtools/run_campaign.py` now dispatches through the shared synthetic campaign runner instead of a duplicated match statement
- synthetic benchmark result artifacts now preserve:
  - `origin`
  - `artifact_targets`
  - `operation_targets`
  - `tags`

Verification:

```bash
ruff check devtools/benchmark_campaigns.py devtools/run_campaign.py tests/unit/devtools/test_benchmark_campaigns.py tests/unit/devtools/test_campaign_report.py
pytest -q tests/unit/devtools/test_benchmark_campaigns.py tests/unit/devtools/test_campaign_report.py
```

Results:

- synthetic benchmark scenario tests: `5 passed in 8.43s`

## 2026-04-13 08:44 CEST — quality registry now includes synthetic benchmark scenarios

- the shared quality registry had still only represented one benchmark family
- added `synthetic_benchmark_campaigns` as a distinct registry category
- populated it from authored synthetic benchmark scenarios instead of another free-form list
- extended `render_quality_reference` to expose:
  - updated registry snapshot counts
  - synthetic benchmark commands
  - a dedicated synthetic benchmark catalog section
- refreshed the generated reference so the repo surface matches the live registry

Verification:

```bash
ruff check devtools/quality_registry.py devtools/render_quality_reference.py tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py
pytest -q tests/unit/devtools/test_quality_registry.py tests/unit/devtools/test_render_quality_reference.py tests/unit/devtools/test_benchmark_campaign.py tests/unit/devtools/test_benchmark_campaigns.py
devtools render-quality-reference
devtools render-quality-reference --check
```

Results:

- devtools quality tests: `11 passed in 9.04s`
- generated quality reference: `sync OK`
