# Changelog

All notable user-visible changes to Polylogue are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

User-visible: anything an operator running `polylogue` would notice — new
flags, removed or renamed commands, output changes, breaking schema
migrations, security fixes. Internal refactors, test additions, and
documentation polish do not require an entry.

## [Unreleased]

### Security

- Webhook delivery now connects to the IP validated against the SSRF
  denylist rather than re-resolving the hostname, closing a DNS-rebinding
  TOCTOU window. Hostname is preserved for SNI/cert verification.
- Path sanitization (`safe_path_component`) NFC-normalizes input before
  applying the ASCII allowlist; visually-identical NFC and NFD forms now
  hash to the same path.
- `pip-audit` runs in CI on every PR that touches `pyproject.toml` or
  `uv.lock`, on master pushes, and weekly. Bumped four CVE-affected deps
  (`pygments`, `pytest`, `cryptography`, `python-multipart`) and pinned
  the latter two as direct constraints.

### Added

- `polylogue select` as a query-backed selector that prints one matched
  conversation field for shell pipelines, with interactive `fzf`/prompt
  selection when attached to a terminal.
- `polylogued` as the daemon/service executable; live source watching now runs
  as `polylogued watch`.
- `polylogued run` to run live watching and the browser-capture receiver
  together as daemon-owned components.
- `dependabot.yml` for weekly Python and GitHub Actions updates with
  patch grouping.
- `actionlint` workflow validating workflow YAML on PRs that touch
  `.github/`.
- MCP serving now accepts `--role read|write|admin`; mutation tools require
  `write` and maintenance tools require `admin`.
- `devtools verify-schema-roundtrip`, diff-shaped `devtools proof-pack`,
  Markdown `devtools obligation-diff`, and a PR proof-comment workflow expose
  proof coverage, known gaps, and schema package round-trips as reusable gates.
- Codex ingestion now materializes `turn_context.cwd`, token/function events,
  and tool call/result messages so cwd filters and diagnostics work beyond
  Claude Code.
- Expression index `idx_raw_conv_effective_provider` so raw-conversation
  provider-filter queries no longer scan the full table.
- Partial indexes scoped to the gating `WHERE` of each `session_profiles`
  search-text backfill, so repeated bootstraps drain an empty index
  instead of scanning the table.

### Changed

- `--resource-mode` and Polylogue CLI self-demotion were removed from
  foreground maintenance commands; workstation resource policy belongs in the
  host environment and daemon supervision, not in product-level CLI flags.
- Query/file-reference filters now use `referenced_path` / `--referenced-path`
  consistently, and MCP conversation reads return headers while message bodies
  live behind paginated `get_messages` / `messages` reads.
- Browser capture now calls its local artifact directory a `spool`; the stale
  inbox helper/export was removed from public paths.
- `schema list`, `schema explain`, and `schema compare` use canonical
  `--format json` output while retaining `--json` as a strict alias.
- `polylogue audit` was removed from the product CLI; verification-lab audit
  workflows live under `devtools`.
- Standalone live ingestion is no longer a root `polylogue watch` command; use
  `polylogued watch` for the long-running source watcher. `polylogue run
  --watch` remains the continuous pipeline-run mode.
- Browser-capture receiver serving/status moved from root `polylogue
  browser-capture` to `polylogued browser-capture`.
- `polylogued status` now reports configured daemon components, including live
  watch roots and the browser-capture receiver target.
- `polylogue doctor --daemon` now includes the same daemon component status in
  the interactive health surface.
- `Config` rejects relative `archive_root`, `render_root`, or `db_path`
  with `ConfigError` at construction.
- `_privacy_level_value` raises `ValueError` on unknown level strings
  instead of silently returning `"standard"`.
- `get_stats_by` raises `ValueError` on unknown `group_by` instead of
  silently falling back to provider grouping.
- Top-level boundary-catchable errors: `ArchiveOperations` config/repository
  init, parsing-service backend uninitialized, and parser state-machine
  phase violations now raise from the `PolylogueError` hierarchy
  (previously `RuntimeError`).
- Search-path degradation visibility: `search_action_results`,
  `search_hybrid_results`, and `helper_summary` log at `WARNING` (not
  `DEBUG`) when falling back from a failed primary path.

### Fixed

- Schema v2 archives now upgrade in place to the additive schema v3
  `messages.message_type` column/index, preserving existing archive rows
  instead of failing the next run after an older binary rewrites
  `PRAGMA user_version`.
- `sanitize_path` symlink probe narrowed to `OSError` and treats
  uncertainty as suspicious (previously a `PermissionError` on an
  unreadable directory could mask a traversal attempt).
- `_clamp_limit` already enforced an upper bound (1000); confirmed
  resolved.
- `resolve_id` already supports `strict=True` and every MCP destructive
  call site already uses it; confirmed resolved.

## [0.1.0] — Unreleased

Initial development snapshot. Versioned releases begin once the install
path lands (#416).
