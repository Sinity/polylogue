# Remaining Work

Consolidated outstanding items from impl_plan.md, IMPROVEMENTS.md, STATUS.md, report.md, and live_capture.md.

## P1 – CLI, Config, and Navigation

- Migrate argparse → Click to reduce duplicated parser wiring; keep a single source of truth for examples/help. **(in progress: click_app added; parity validation, examples/help/prefs/completions still needed)**
- Expand `config init` prompts (inbox/output roots, index/Qdrant); make `config set --output-root` update per-provider paths or warn about mixed roots.
- Anchors/open flow: ensure Markdown/HTML renders expose anchors and `search --open` jumps correctly; keep open/print-path helpers consistent across commands.
- Multi-root/prefs polish: labeled roots, persistent per-command defaults, and an interactive config editor (TUI) that shows resolved paths/index settings.
- Schema/version stamps: embed CLI + schema version in front matter/JSON, normalize casing via a shared schema layer, and surface migrate/self-update hints.
- Redaction/sanitize: optional HTML/Markdown scrubbing (`--sanitize-html`, mask keys/emails/tokens) and mark redacted runs in summaries.
- UI flow streamlining: unify DID A LITTLE. DO IT ALL. FUCK.

progress/spinner panels across sync/render/import/search, reduce duplicate summary blocks, and ensure consistent wording/colour tokens between rich and plain modes.

## P2 – Rendering, Attachments, and Inbox

- HTML refresh: TOC/anchors, sticky header, attachment index/gallery, client-side search/filter; modernize layout and branch explorer (inline diffs/snippets, branch map snippet in `conversation.md`).
- Attachments lifecycle: `attachments stats --provider --since/--until` with deduped bytes, `--clean-orphans`, routing rules with skipped/routed counts; collision-safe extraction; attachment-aware search with OCR when enabled.
- Inbox hygiene: coverage/quarantine reports for malformed exports/JSONL with `.polylogueignore`-aware “skipped-by-rule” summaries and a `--quarantine` path.

## P2 – Performance and Freshness

- Content-hash freshness instead of mtime-only checks.
- Concurrency/ETA: clearer knobs with visible progress/retry surfacing and predictive ETA/profiling beyond existing `--profile-io/--profile-sql`.
- Deterministic ordering across HTML/JSON/summaries; emit pre-run disk/quota estimates.

## P3 – Analytics and Reporting

- Timeline views and conversation/branch maps (HTML-side conversation map/timeline).
- Role/model/branch analytics: tables or heatmaps; per-branch metrics and branching hotspots in status.
- Metrics export/serve endpoint (Prometheus-friendly) and structured run history export.

## P3 – Testing and Quality

- Golden master tests for parsers/rendering (Markdown + HTML snapshots per provider/model/tool-call cases).
- CLI integration matrix: cover plain vs interactive (PTY-backed) runs, non-TTY guardrails, and picker flows; add fixtures for `search --open`, `--json/--json-lines/--csv`, `--print-paths`, and watch/dump flows.
- Watch-mode regression: simulated file change streams with debounce windows to catch stalls, suppressed events, and base-dir validation.
- UI surface tests: stub gum/skim/bat/glow/delta (or the pure-Python facade when available) to assert prompts, summaries, anchors, progress output, and warnings render correctly in rich and plain modes.
- Interactive harness: add PTY-driven tests (pexpect or similar) to exercise prompts/confirmations/menus end-to-end, including cancellation paths and default selections.
- Replace custom retry logic with tenacity-based wrappers (with retry-count assertions in tests).

## P3 – Metadata, Provenance, and Integrity

- Per-run metadata injection (`--meta key=value`); stamp renders with source path, import time, CLI version, and content hash.
- Integrity verifier (`polylogue verify`) to check front matter vs DB state, attachments, and branch files; front-matter canonicalizer to normalize keys/order and flag unknowns.
- Partial run recovery (`--resume-from <run-id>`) to retry failed chats/attachments with a clear report.

## Deferred/Optional

- Not currently planned: switching Drive sync to rclone, removing HTML generation entirely, or removing watch mode.
