# Implementation Plan (Imperative Checklist)

This document tracks remaining improvements and how to implement them. Treat each bullet as an actionable step.

## Help/Docs & Discoverability
- Update README/help text to include `--ignore-legacy`, `--examples`, `--print-paths`, `--quiet`, and anchor-enabled `search --open` behaviour.
- Wire `COMMAND_EXAMPLES` into per-command help output automatically (not only `help --examples`), ensuring a single source of truth.

## Non-TTY/Prompt Safety
- Audit all remaining prompt paths (skim previews, any raw `input()` or pickers) and add non-TTY guards; ensure plain mode never auto-selects without explicit flags.
- Remove any remaining silent auto-select defaults in plain mode; require explicit `--all`/IDs or abort with a warning.

## Status/Stats Enhancements
- Add sortable/top lists and CSV/JSONL export for stats (largest attachments/tokens/recent). Consider `--sort`, `--limit`, `--csv/--json-lines` flags.
- Implement quiet JSON mode toggle (`--json-verbose` to include tables/logs) in status and stats; default JSON to be noise-free.
- Emit structured failure exports on errors (run id, provider, file, phase, exception class, hint) to a standard JSONL path; add shell-friendly exit reasons (`POLYLOGUE_EXIT_REASON`).
- When falling back to minimal frontmatter parsing, emit a warning in both plain and rich modes.

## Configuration/Auth
- Ensure `config show` (plain and JSON) always surfaces credential/token paths and env overrides consistently.
- Make `config set --output-root` update per-provider output paths (or warn about mixed roots) to avoid split archives.
- Expand `config init` wizard to prompt for inbox/output roots and index/Qdrant settings to match the sample config.
- Add schema/version stamps to front matter and JSON outputs; provide migrate/self-update hints when versions diverge.
- Standardize JSON field casing via a shared schema layer to eliminate mixed `camelCase`/`snake_case` outputs.

## Search & Navigation
- Add Markdown anchors and ensure `search --open` jumps correctly for both Markdown and HTML outputs.
- Add `search --csv/--json-lines --fields ...` and piping support (`--from-stdin`, `--output -`).
- Implement provider compare/filter subcommand (`compare` or `filter`) for side-by-side results and post-processing.

## HTML & Branch Explorer
- Modernize HTML template: add TOC/anchors, sticky header, attachment index/gallery, client-side search/filter, and updated layout.
- Enhance branch explorer: inline diffs/snippets, modernized HTML, and embed a branch map link set into transcripts.
- Add branch map snippet to `conversation.md` headers for quick context.

## Watch/Sync Reliability
- Add live tail/stall detection for `--watch`, with a debounce summary option.
- Expose retry/backoff tuning flags per provider; add offline mode (`--offline`) and drift alarms/guardrails that can flip exit codes.
- Implement safe rollback snapshots before destructive actions and disk/quota preflight estimation.

## Attachments & Inbox
- Implement attachment stats/extract/dedupe/routing commands; add attachment-aware search with optional OCR.
- Add inbox coverage/quarantine commands and support `.polylogueignore` skip rules with a “skipped-by-rule” summary.

## Performance & Freshness
- Add concurrency knobs with visible progress/retry surfacing and predictive ETA/profile flags (`--profile-io/--profile-sql`).
- Replace mtime-only freshness checks with content hash-based checks; add pre-run disk/quota estimation and warnings.
- Enforce deterministic ordering across HTML/JSON/summaries (providers, slugs, timestamps) where not already sorted.

## Preferences & Multi-Root
- Implement per-command defaults/prefs storage and a `polylogue prefs` command to list/reset.
- Add multi-root support with labels (`--root <name>`) for separate archives.
- Add path helpers: `--print-paths` is partially done; consider `open last` or `--print-paths` for browse outputs.

## Analytics & Visualization
- Add timeline views, role/model heatmaps, per-branch metrics, and metrics export/serve endpoint (Prometheus-friendly).
- Add HTML client-side search/filter and conversation map/timeline views for large archives.
