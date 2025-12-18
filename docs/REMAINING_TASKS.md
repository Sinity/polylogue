# Remaining Work

Consolidated outstanding items from impl_plan.md, IMPROVEMENTS.md, STATUS.md, report.md, and live_capture.md.

## Current State (handoff)

If you’re resuming work later, this is the “what’s already implemented” snapshot:

- Click CLI is the primary entrypoint (`polylogue/cli/click_app.py`); argparse is gone.
- Run logs: `browse runs --json-lines` emits newline-delimited JSON for `load_runs()` records.
- Metrics: `browse metrics` emits Prometheus text (`--json` available) and can `--serve` at `/metrics`.
- Metadata injection: `sync|import|render` accept repeatable `--meta key=value`.
  - Stored in markdown front matter as `polylogue.cliMeta` (to avoid collisions with provider-native `polylogue.meta`).
  - Also recorded in run history as `run.meta` in `runs.metadata_json`.
- Integrity: `polylogue verify` exists (checks front matter vs DB/state, attachments, and branch docs).
- Golden tests: portable Markdown + basic HTML snapshot suites exist.
  - Fixtures live in `tests/fixtures/golden/`, expected outputs in `tests/golden/` and `tests/golden_html/`.
  - Regenerator: `nix develop -c python3 scripts/update_goldens.py` (updates both Markdown + HTML goldens).
  - Snapshots compare after canonicalization/scrubbing via `polylogue/frontmatter_canonical.py`.
  - Determinism hooks: `POLYLOGUE_FIXED_NOW` (fixed timestamps) + tests force a stub `tiktoken` tokenizer for stable counts.

## P1 – CLI, Config, and Navigation

- Migrate argparse → Click to reduce duplicated parser wiring; keep a single source of truth for examples/help. **(done: click_app now covers all commands and is the primary entrypoint)**
- Expand `config init` prompts (inbox/output roots, index/Qdrant); make `config set --output-root` update per-provider paths or warn about mixed roots. **(done: init/set flows aligned, mixed-root guardrails added)**
- Anchors/open flow: ensure Markdown/HTML renders expose anchors and `search --open` jumps correctly; keep open/print-path helpers consistent across commands. **(done: stable msg anchors + open helpers fixed)**
- Multi-root/prefs polish: labeled roots, persistent per-command defaults, and an interactive config editor (TUI) that shows resolved paths/index settings. **(done: labeled roots included in scans, `config edit` TUI added, prefs expanded for root/sanitize/diff)**
- Schema/version stamps: embed CLI + schema version in front matter/JSON, normalize casing via a shared schema layer, and surface migrate/self-update hints. **(done: stamp_payload + front matter versions everywhere)**
- Redaction/sanitize: optional HTML/Markdown scrubbing (`--sanitize-html`, mask keys/emails/tokens) and mark redacted runs in summaries. **(done: `--sanitize-html` end-to-end + redacted run metadata)**
- UI flow streamlining: unify progress/spinner panels across sync/render/import/search, reduce duplicate summary blocks, and ensure consistent wording/colour tokens between rich and plain modes. **(done: shared render summary helper + consistent flags/prefs wiring)**

## P2 – Rendering, Attachments, and Inbox

- HTML refresh: TOC/anchors, sticky header, attachment index/gallery, client-side search/filter; modernize layout and branch explorer (inline diffs/snippets, branch map snippet in `conversation.md`). **(done: conversation HTML includes TOC/sticky header/search, attachment gallery w/ filter + lightbox, and stable anchors; branch map snippet injected into `conversation.md`; branches HTML explorer includes inline diffs)**
- Attachments lifecycle: `attachments stats --provider --since/--until` (index-backed) + `--clean-orphans` + collision-safe extraction. **(done: filters/cleanup + collision-safe `attachments extract`)**
- Attachments lifecycle: routing rules with skipped/routed counts; attachment-aware search with OCR when enabled. **(done: attachment routing counts are recorded in `attachmentPolicy.routing`; attachment search includes OCR text when enabled)**
- Inbox hygiene: `.polylogueignore`-aware “skipped-by-rule” summaries and `--quarantine` path. **(done: ignored counts + quarantine reporting)**
- Inbox hygiene: coverage reports for malformed exports/JSONL beyond current detection. **(done: inbox JSON reports malformed exports/JSONL + reasons; quarantine treats them as candidates)**

## P2 – Performance and Freshness

- Content-hash freshness instead of mtime-only checks. **(done for local session + export sync; prune runs still force full scan)** 
- Concurrency/ETA: clearer knobs with visible progress/retry surfacing and predictive ETA/profiling beyond existing `--profile-io/--profile-sql`. **(done: `sync --jobs N` for local session providers + plain-mode progress/ETA logs for Drive + export bundle syncs)**
- Deterministic ordering across HTML/JSON/summaries; emit pre-run disk/quota estimates. **(done: JSON outputs consistently use `sort_keys=True`; Drive chat lists are sorted; HTML attachment sidebar is sorted deterministically; disk estimates print via `--disk-estimate` and limits enforce via `--max-disk`)**

## P3 – Analytics and Reporting

- Timeline views and conversation/branch maps (HTML-side conversation map/timeline). **(done: `browse timeline` writes a global HTML timeline + `--json`; branch map snippet is injected into each `conversation.md`)**
- Role/model/branch analytics: tables or heatmaps; per-branch metrics and branching hotspots in status. **(done: `browse analytics` writes HTML + `--json` with roles/models and branch hotspots)**
- Metrics export/serve endpoint (Prometheus-friendly). Structured run history export. **(done: `browse metrics` emits Prometheus text + `--serve`; `browse runs --json-lines` emits JSONL)**

## P3 – Testing and Quality

- Golden master tests for parsers/rendering (Markdown + HTML snapshots per provider/model/tool-call cases). **(done: portable Markdown + HTML golden snapshots per provider/tool-call case + generator script)**
- CLI integration matrix: cover plain vs interactive (PTY-backed) runs, non-TTY guardrails, and picker flows; add fixtures for `search --open`, `--json/--json-lines/--csv`, `--print-paths`, and watch/dump flows. **(done: Click wrapper tests + PTY coverage for key interactive flows; `search --open` + `--csv` and `--print-paths` now have unit coverage)**
- Watch-mode regression: simulated file change streams with debounce windows to catch stalls, suppressed events, and base-dir validation. **(done: debounce/stall/base-dir validation are covered with deterministic tests)**
- UI surface tests: stub gum/skim/bat/glow/delta (or the pure-Python facade when available) to assert prompts, summaries, anchors, progress output, and warnings render correctly in rich and plain modes. **(done: prompt helpers + picker + diff output + anchor/open helpers have unit coverage; PTY tests exercise end-to-end prompt flows)**
- Interactive harness: add PTY-driven tests (pexpect or similar) to exercise prompts/confirmations/menus end-to-end, including cancellation paths and default selections. **(done: PTY tests cover `config init` + `sync codex` with cancellation + selection paths)**
- Replace custom retry logic with tenacity-based wrappers (with retry-count assertions in tests). **(done: Drive retry wrappers now use tenacity; existing drive metrics tests cover retry counts)**

## P3 – Metadata, Provenance, and Integrity

- Per-run metadata injection (`--meta key=value`); stamp renders with source path, import time, CLI version, and content hash. **(done: `--meta` stored as `polylogue.cliMeta` + included in run logs)**
- Integrity verifier (`polylogue verify`) to check front matter vs DB state, attachments, and branch files; front-matter canonicalizer to normalize keys/order and flag unknowns. **(done: verify supports `--fix`, `--unknown`, `--strict`, and uses canonicalizer)**
- Partial run recovery (`--resume-from <run-id>`) to retry failed chats/attachments with a clear report. **(done: `--resume-from` uses recorded failures with stage summaries; Drive attachment failures are recorded per chat+attachment (including error), don’t abort the whole run, and can be retried via `sync drive --resume-from <id> --attachments-only`)**
