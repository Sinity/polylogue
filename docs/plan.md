# Polylogue Improvement Plan

This file consolidates the high-impact UX and refactoring issues observed across the CLI, renderers, and supporting workflows. Items are grouped by area with the expected remediation direction to make planning and scoping easier.

## Correctness & Core UX
- Unify version reporting: `build_parser` currently hard-codes `0.1.0` (polylogue/cli/app.py:1159). Derive the version from `pyproject.toml`/package metadata so releases never drift.
- Deduplicate parser construction: `build_parser()` is defined twice in `polylogue/cli/app.py`, inviting divergence in flags/descriptions. Keep a single source of truth and fan out to sub-builders.
- HTML default logic: `default_html_mode` sets `html_mode` to "on"/"off", so the documented default "auto" path never executes. Restore true auto detection.
- Collapse handling: `resolve_collapse_value` rejects zero, so users cannot disable folding; also there is only one collapse threshold for all content types. Allow 0 for "no collapse" and support per-type thresholds (e.g., tool/output blocks vs. messages).
- Branch explorer robustness: `build_branch_html` assumes `conversation.nodes` is non-empty and calls `next(iter(...))`, causing StopIteration on empty graphs. Gracefully show "no branch data" instead.
- Search open target: `search --open` ignores hit offsets/IDs and opens the file top. Add anchors in Markdown/HTML and jump directly to the match.
- Watch help vs. capability: Drive inherits `--watch` flags from the shared parser but immediately errors in `get_local_provider("drive")`. Hide the flag for Drive or fail early with a friendly message.
- Status CLI divergence: `polylogue.py status` demands `--providers` and errors immediately, while `browse status` works without it. Align UX (default providers) or deprecate the redundant command.
- Stats correctness: `browse stats` double-counts canonical `conversation.md` and legacy `*.md` per root with no ignore switch; add an option to exclude legacy to avoid inflated totals.
- Stats parsing: when front matter cannot be imported, the manual YAML parser in `cli/status.py` mis-parses nested/colon values silently; replace with a robust parser and emit warnings.
- Missing dir handling: `browse stats` exits 0 and prints "No Markdown files found" when `--dir` is missing/absent; return non-zero so CI/scripts can detect the failure.
- Status filtering: `run_status_cli` applies `runs_limit` before provider filtering, causing empty/partial results with `--providers`. Filter first, then limit, and warn when truncating.
- Duplicate examples: `COMMAND_EXAMPLES` exists but is not surfaced in `--help` and can drift from the parser. Wire examples into help output and add a `--examples` flag.

## Configuration, Auth, and Schema
- Config visibility: `config show --json` omits credential/token paths and env overrides (`POLYLOGUE_CREDENTIAL_PATH`, `POLYLOGUE_TOKEN_PATH`). Surface resolved auth paths and the source (env vs. config).
- Output root consistency: `config set --output-root` only updates the render root, leaving per-provider paths unchanged. Apply root updates consistently or warn about mixed roots.
- Init wizard drift: `polylogue/cli/init.py` hardcodes `~/polylogue-data` and skips inbox/index/Qdrant prompts, diverging from `docs/polylogue.config.sample.jsonc`. Expand prompts to match documented settings.
- Env preflight: add `polylogue env check` (or doctor) to validate `POLYLOGUE_*` env vars, index backend settings, and path overrides with actionable fixes before commands run.
- Schema consistency: JSON outputs mix casing (`credentials_present` vs. `conversationId` vs. `attachmentBytes`). Centralize field naming and enforce via a shared schema layer.
- Schema/version stamp: embed Polylogue version and schema version in front matter; warn on opening older outputs and offer `polylogue migrate` for metadata updates. Also add a self-update reminder when the CLI lags the flake version.

## Non-Interactive Safety & Dependency Preflight
- Non-TTY prompts: `UI.confirm/choose/input` and `cli_common.choose_single_entry` call `input()` when stdin is not interactive, hanging CI/cron. In plain mode/non-TTY, fail fast or require explicit flags.
- Plain auto-select: plain mode currently auto-selects the first option silently, triggering unintended actions. Require explicit `--all`/IDs or abort with a clear message.
- Dependency coverage: interactive flows only preflight `gum`/`skim`, but previews use `bat`/`glow`/`delta`. Add upfront checks for all required tools and fail early with guidance.
- Drive plain-mode auth: `DriveClient.ensure_credentials` exits with a terse message in plain mode; include hints for `POLYLOGUE_CREDENTIAL_PATH`/`POLYLOGUE_TOKEN_PATH` and non-interactive setup.
  **Status:** non-TTY guards added to UI prompts and skim pickers; dependency preflight expanded; Drive plain-mode hint added; plain auto-select still needs opt-in flags.

## Drive/Import/Filtering
- Drive chat filtering: `filter_chats` drops items without `modifiedTime` silently. Emit warnings and counts for discarded entries so users understand why chats were skipped.
- HTML auto default: default sync/import namespaces force html_mode on/off, bypassing the documented auto behavior. Honor "auto" to match docs and user expectations.
- Filter order: `run_status_cli` provider filtering happens after limiting (see above) and can hide older matching runs; fix ordering and add a warning when results are incomplete.

## Status/Stats/Reporting UX
- Help/examples discoverability: examples are present but hidden; expose in `--help` and via a dedicated flag. Keep a single example table to avoid drift. **(examples flag added)**
- Browse status output: current stats are totals only; add sorting/top lists (largest attachments/tokens/recent) and CSV/JSONL export for ranked data to make outputs actionable.
- Failure signaling: when stats/status inputs are missing or empty (missing dir/provider), exit non-zero and print a clear hint. **(exit codes added for missing dirs)**
- Quiet JSON mode: when `--json` is set, suppress non-JSON logs by default and add `--json-verbose` to interleave structured events without noisy mixed output.
- Structured failure export: on any error, write a standard failure record (run id, provider, file, phase, exception, hint) to a known JSONL path for CI triage.
- Shell-friendly exits: expose `POLYLOGUE_EXIT_REASON` (auth|io|schema|partial) and map common cases to stable exit codes for wrappers.

## HTML/Rendering & Navigation
- Modernize HTML: current template is single-column with minimal metadata. Add anchors/TOC, sticky header, attachment index, and updated layout for large transcripts.
- Permalinks and anchors: add stable heading IDs and a "copy link" control per message; ensure `search --open` jumps to anchors in Markdown/HTML.
- Client-side navigation: add lightweight in-page search/filter (text/role) with keyboard shortcuts (/, j/k) for large HTML exports.
- Media presentation: add attachment gallery/thumbnails for common images and link to originals; optionally sanitize HTML output (`--sanitize-html`) for safer sharing.
- Conversation slicing: support `--last N`/`--since <time>` and reflect slices in headers so partial renders are clearly marked.
- Export options: add `--pdf` alongside HTML using wkhtmltopdf/weasyprint; keep anchors and metadata.
- Branch explorer: include inline diffs/divergence snippets, expand/collapse controls, and a graceful empty-state message.
- Branch context in transcripts: embed a small branch map (links to branch files) near the top of `conversation.md` so branch context is visible without opening branch files.

## Watch/Sync Workflow
- Base dir safety: `_run_watch_sessions` unconditionally `mkdir`s `base_dir`, so typos create junk paths (e.g., `~/.codex/sessions/~/.claude/projects`). Validate existence and warn instead of creating.
- Debounce visibility: watch drops changes within the debounce window silently. Log suppressed paths and emit a summary of dropped events.
- Live feedback: add an optional live tail pane for `--watch` showing recent messages/errors and detect stalls (no progress for X seconds) with suggested next steps.
- Watch wizard: add an interactive helper to assemble watch commands (provider/base-dir/out/html/diff/prune/debounce) and print the exact invocation.

## Guardrails & Reliability
- Dependency preflight (expanded): fail early when required tools are missing (gum/skim/bat/glow/delta).
- Config drift check: detect stale configs (older than code version or containing deprecated keys) and prompt to run config lint/auto-fix before commands proceed.
- Disk/quota preflight: estimate disk needed (attachments + HTML/diffs) and Drive API calls before starting; warn or require `--force` when exceeding thresholds.
- Retry/backoff controls: add per-provider retry/backoff tuning flags (`--retry-limit`, `--retry-base`) and surface retry counts in summaries.
- Redaction/safety: optional scrubber to mask keys/emails/tokens in rendered Markdown/attachments; surface "redacted" in summaries. Add `--sanitize-html` for safe sharing.
- Safe rollback: before destructive actions (`--prune`, overwrites), create a lightweight snapshot/zip and provide `polylogue restore <snapshot>`.
- Offline mode: `--offline` to skip network-dependent steps while processing local data; mark outputs as offline/incomplete.
- Deterministic ordering: standardize ordering (provider, slug, timestamp) in summaries/JSON/HTML to keep diffs stable.
- Quiet long runs: detect stalled runs and surface spinner warnings plus guidance (`--trace`, `--timeout`, retry suggestions).

## Attachments & Inbox Handling
- Attachment lifecycle: add `attachments stats --provider --since/--until` and `--clean-orphans` to summarize and clean orphaned blobs; report deduped bytes when hashing duplicates across runs.
- Attachment routing: allow rules to route certain MIME types to external viewers or skip rendering; report routed/skipped counts.
- Attachment batch extract: `polylogue attachments extract --type pdf --out <dir>` with collision-safe naming and progress.
- Attachment-aware search: index attachment text (with optional OCR) and add `search --in-attachments/--attachment-name`.
- Inbox coverage: add `polylogue browse inbox` to list unprocessed exports/JSONLs per provider with size/mtime to show pending work.
- Inbox quarantine: validate incoming exports/JSONL before processing; move malformed/unknown provider files to quarantine with a report command.

## Search & Filtering Enhancements
- CSV/NDJSON export: `search --csv/--json-lines --fields provider,slug,branch,timestamp,snippet` for pipelines.
- Interactive filter chains: a `polylogue filter` subcommand to trim transcripts by role/attachments/models/time window, producing a filtered copy.
- Piping support: allow `search --from-stdin` for queries and `--output -` for Markdown/HTML to integrate with shells.
- Provider compare: `polylogue compare --provider-a --provider-b --query '...'` to show side-by-side snippets and differences in coverage.
- Structured run history export: `runs export --json-lines` with field selectors for dashboards.
- Path reveal & open helpers: `--print-paths` to list written files and `open last` to jump to the most recent output (with provider filters).

## Performance & Parallelism
- Parallel imports/sync: expose concurrency knobs for Drive/local imports (chat + attachment downloads) with sensible limits; summarize parallelism, retries, and bottlenecks.
- Predictive ETA: show per-provider ETA and retry budget during runs based on past throughput and remaining items.
- Profile toggles: add `--profile-io/--profile-sql` to surface top slow steps (download, parse, render, index) with timings.
- Hash-based freshness: replace mtime-only `_is_up_to_date` with content hashes to avoid missed updates when mtimes are unchanged.
- Pre-run space check: estimate required disk before sync/import; warn or require `--force` when projected usage exceeds available space.

## Metadata, Provenance, and Integrity
- Per-run metadata injection: support `--meta key=value` on render/import/sync to append to front matter and record in the runs DB.
- Session provenance: stamp rendered Markdown/HTML headers with source path, import time, CLI version, and hash so outputs are traceable.
- Integrity verifier: `polylogue verify --slug/--provider` to confirm front matter matches DB state, attachments exist, and branch files are present; emit a single-pass report.
- Front-matter canonicalizer: normalize field order/types, enforce known keys, and flag unknowns to keep diffs clean.
- Partial run recovery: `--resume-from <run-id>` to retry only failed chats/attachments and report what was retried.

## User Preferences & Multi-Root Support
- Multi-root configs: allow multiple inbox/output roots with labels selectable via `--root <name>` for separate work/personal archives.
- Persistent per-command defaults: store user preferences (e.g., `search --limit 50 --no-picker`, `sync drive --links-only`) and provide `polylogue prefs` to list/reset.
- Interactive config editor: a TUI (`polylogue config edit`) that validates fields live, lists defaults, and shows resolved paths/index settings.

## Analytics, Reporting, and Visualization
- Timeline views: add `browse timeline --provider --since/--until` to list conversations chronologically with quick links.
- Conversation/branch maps: generate HTML maps grouped by month/size/attachments and embed a branch map in transcripts for orientation.
- Role/model analytics: `browse roles --provider --since/--until` to show distribution of roles/models/tool calls (table/heatmap).
- Per-branch metrics: extend status to include branch counts/divergence stats per provider and flag branching hotspots.
- Metrics export: `metrics serve` (or `--metrics-file`) to emit provider/run stats for Prometheus scraping.
- Predictive retries and drift alarms: allow guardrails (`--max-failures`, `--max-retries`, `--max-skips`) that flip exit codes and print an alarm section when thresholds are crossed.
