# Changelog

All notable user-visible changes to Polylogue are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

User-visible: anything an operator running `polylogue` would notice — new
flags, removed or renamed commands, output changes, breaking schema
migrations, security fixes. Internal refactors, test additions, and
documentation polish do not require an entry.

## [Unreleased]

### Added

- Maintenance replay failures now surface in `polylogue status` and the
  daemon raw-failure health check (#1198). Per-record failures from
  `repair_session_insights`, `repair_action_event_read_model`, and
  `source_replay` are routed through
  `polylogue.maintenance.failure_routing.route_failure_sample` into an
  append-only JSONL file at
  `<archive_root>/.maintenance-state/failures.jsonl`, then merged into
  the existing `_raw_failure_info()` payload with `source="maintenance"`
  and the originating `operation_id`. The `raw_failures` health alert
  escalates through the existing WARNING / ERROR / CRITICAL ladder and
  cites a representative `op=<short_id>` so operators can pull the
  failing replay's resume state file directly. Absolute paths in
  routed messages and locators are redacted at write time.

- Non-log notification backends for the daemon health loop (#1233).
  `notification_backend` now accepts `"webhook"`, `"journald"`,
  `"email"`, `"apprise"`, and lists/comma-separated fan-out specs
  (e.g. `notification_backend = ["log", "apprise"]`). The webhook
  backend signs the JSON envelope with HMAC-SHA256 in
  `X-Polylogue-Signature` when `notification_webhook_secret` is set,
  and uses exponential backoff across up to three attempts. The email
  backend (`[notifications.email]` TOML table or
  `POLYLOGUE_NOTIFICATION_EMAIL_*` env vars) speaks SMTP with
  STARTTLS / implicit-TLS on port 465 and rate-limits to N
  messages/hour. The Apprise backend reads
  `notification_apprise_urls` and dispatches to 100+ services
  (Pushover, Discord, Slack, ntfy, Matrix, Telegram, …) through one
  adapter. Per-backend failures in fan-out mode are isolated so one
  broken destination does not silence the rest.

- Cycle outlook reaches the CLI and MCP surfaces (#1138). New
  `polylogue cost outlook --plan <name>` subcommand renders the typed
  `CycleOutlook` payload from #1137 — cycle window, burn rate,
  projected total, quota pressure, overage rows, coverage, and
  confidence — in JSON or plain mode. Plain mode labels USD totals as
  API-equivalent, quota figures by basis, and surfaces "quota not
  configured" explicitly so subscription-equivalent and
  API-equivalent numbers never collapse into a single unlabelled
  `cost`. The legacy flat surface is preserved as
  `polylogue cost rollup`. New MCP tool `cost_outlook(plan, method)`
  returns the same typed payload; the matching async facade method
  `Polylogue.cost_outlook(plan_name, now=None, method=...)` resolves
  plans against `[[cost.subscription.plans]]` user overrides merged
  with the curated seed.
- Learning feedback loop — corrections as deterministic rebuild signal
  (#1131). New `polylogue feedback` command group (`record`, `list`,
  `clear`) and matching MCP tools (`record_correction`,
  `list_corrections`, `clear_corrections`) let users override the
  heuristic classifier, accept/reject auto-tags, and replace generated
  summaries. Corrections live in the new `user_corrections` table (schema
  v15) outside the content-hash boundary: applying or removing a
  correction never alters `conversations.content_hash`. The
  `classify_session` insight path now consults corrections after the
  heuristic so rebuilds always produce the same merged verdict across
  runs.

### Schema

- Bump `SCHEMA_VERSION` from 14 to 15 to introduce the `user_corrections`
  table. Existing databases are rejected with the usual fresh-first
  message; an explicit upgrade script is required.

- Tool usage analytics with explicit per-provider coverage (#1133). New
  insight type `tool_usage` rolls up per-(provider, tool, action_kind)
  call counts, conversation counts, distinct tool ids, and affected-path
  / output-text density over canonical `action_events`. The same
  envelope carries a per-provider coverage map distinguishing
  "data unavailable" (provider exposes no tool events) from "zero
  observed" so coverage gaps are never collapsed into silent zeros.
  Available as `polylogue insights tool-usage` (`--tool`,
  `--mcp-server`, `--action-kind` filters), MCP tool `tool_usage`, and
  facade `list_tool_usage_insights(query)`. MCP tool names of the form
  `mcp__<server>__<tool>` have their server segment extracted as a
  first-class `mcp_server` field on each entry.

- Resume brief is now a first-class durable insight (#1129). `ResumeBrief`
  carries a typed `provenance` payload citing the session, message,
  work-event, phase, and work-thread IDs it composed from, with a
  `materializer_version` consumers can use to invalidate cached renderings.
  New MCP tool `get_resume_brief(conversation_id, related_limit)` returns the
  typed brief on the shared single-object envelope; the existing
  `polylogue resume <session-id>` CLI continues to render the brief and now
  surfaces the provenance fields via `--format json`.

- Typed maintenance planner contract (#1144): `BackfillOperation`
  envelopes now carry a typed `scope`, `reason`
  (`InvalidationReason`), `resume_cursor`, bounded `failure_samples`
  with a `truncated` flag, and `metrics` alongside the existing
  fields. `polylogue maintenance run --target` exposes a new
  `message_embeddings` target (no-op until the dormant embedding
  pipeline is wired in, #828) so the planner's invalidation-key
  vocabulary covers messages FTS, action-event read model, session
  insights, and embeddings/vector index.
- Realtime update channel for the daemon web reader (#957): new
  `GET /api/events` Server-Sent Events endpoint streams daemon-event
  notifications (`ingestion_batch`, `ingest`, `reset`, `operation`) so the
  reader updates without a manual refresh. `?poll=1` returns the same
  payload as a JSON snapshot for `EventSource`-less fallbacks. `GET
  /api/status` now advertises a monotonic `last_event_id` field and a
  weak ETag, and returns `304 Not Modified` when the client's
  `If-None-Match` matches.

### Security

- FTS5 query escaping now treats `.`, `/`, and `?` as special characters so
  search inputs containing path-like or single-character-wildcard
  punctuation are quoted rather than surfaced to SQLite as syntax errors.
  Previously, inputs such as `../etc/passwd`, `foo.bar`, or `test?` from
  CLI/MCP search raised `sqlite3.OperationalError` and leaked the underlying
  FTS5 grammar to callers.
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

- Broad-distribution packaging readiness (#953): the
  `release verify-distribution` gate now runs as a `distribution` job in
  `ci.yml` on every push, a new `release.yml` workflow publishes wheel
  and sdist to PyPI via OIDC Trusted Publishing on `vX.Y.Z` tag push,
  and a multi-stage `Containerfile` produces an OCI image that
  `release.yml` builds and pushes to `ghcr.io/sinity/polylogue` with
  semver tags. See [`docs/release.md`](docs/release.md) for the cut-time
  checklist.
- Local source discovery and parsing for Gemini CLI `~/.gemini/tmp` sessions
  and Hermes `~/.hermes/sessions` session documents, with distinct
  `gemini-cli` / `hermes` source identities.
- Antigravity source ingestion through its local language-server Markdown
  export surface, with parseable brain artifacts retained as auxiliary
  documents and raw protobuf state classified as non-directly-parseable
  sidecar storage.
- `polylogue insights work-events` now accepts `--session-date-since` and
  `--session-date-until`, exposing canonical-date bounded work-event reads
  through the public insight facade.
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
- Schema v5: tags M2M tables (`tags`, `conversation_tags`), blob GC lease
  tables (`pending_blob_refs`, `gc_generations`), and `insights.db` extraction
  with its own schema lifecycle. Wipe-and-rebuild required.
- `ArchiveWriteGateway` as the single canonical write path (daemon RPC stub for
  slice G). Post-ingest side effects (FTS repair, cache invalidation) now route
  through `polylogue.archive.write_effects`.
- `run_blob_gc()` for safe garbage collection of unreferenced blobs with lease,
  generation, and MIN_AGE guards.
- `devtools run-benchmark-campaigns` now includes a
  `daemon-live-convergence` synthetic campaign that reports live-ingest file
  counts, read/write byte shape, append-tail byte shape, stage timings, and
  archive row counts, plus process and cgroup memory peaks.
- Durable reader workspaces are now persisted across the archive API, daemon
  `/api/user/workspaces`, `polylogue user-state workspaces`, and MCP mutation
  tools, with resolved/degraded target evidence for tabs, stack, compare, and
  timeline modes.
- The local reader daemon now serves `/w/stack` and `/w/compare` shell routes
  plus `/api/stack` and `/api/compare` envelopes for multi-conversation reader
  workflows with explicit missing-target evidence.

### Changed

- Live daemon convergence now drains watcher batches single-flight, records
  stale cursor-write counters, uses bounded tail hashes for same-size rewrite
  detection, and exposes recent source-path churn in the daemon workload probe.
- Session work-event and phase evidence payloads now expose timing/date
  provenance, and work-event insight reads order by event time instead of
  conversation recency.
- The root query `--tail` mode and tail-overlay JSON provenance were removed;
  daemon-owned live ingestion is now the supported path for fresh session state.
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
- Daemon status now caps live cursor file samples while preserving exact
  counts, and full-ingest attempts heartbeat during long storage-write phases.
- Schema v9 adds indexes for message foreign keys on `provider_events` and
  `attachment_refs`, avoiding full child-table scans during conversation
  replacement.
- `polylogued status` now reports recent live-ingest attempts with durable
  phase, file-count, byte-read, timing, RSS, cgroup memory, and stale-heartbeat
  snapshots so interrupted convergence work leaves diagnosable state in the
  archive DB.
- Codex JSONL ingestion now parses hot streams directly from raw records,
  skips validation-off pre-sampling for known stream providers, and reuses
  message hash payloads during materialization, reducing daemon live-ingest
  parse overhead for large Codex sessions.
- Daemon live convergence refreshes affected insight rows in batches and avoids
  process-pool startup for tiny ingest batches, reducing convergence and parse
  overhead for live JSONL workloads.
- Daemon live catch-up no longer pre-hashes uncursored files, reads cursor
  state in bulk, and reports the max ingest worker count in live benchmark
  metrics so many-file convergence throughput is observable.
- Daemon live ingest bounds small-file convergence groups, offloads sync
  parse/write work from the event loop, and drains process-pool results without
  retaining the whole parsed batch in memory.
- Daemon live convergence avoids duplicate message-FTS repair, chunks sync
  insight rebuilds by message budget, samples each JSONL once before raw
  storage, and streams raw blobs in 1 MiB chunks for large catch-up workloads.
- Daemon convergence now scopes embedding work to changed conversations and
  avoids starting an async embedding runner from the synchronous convergence
  stage.
- Daemon live ingest now excludes relationship-index JSONL sidecars before raw
  storage instead of treating scalar `conversation`/`parent`/`child` records as
  provider conversation streams.
- Live watching is no longer exposed through root `polylogue watch` or
  `polylogue run --watch`; use `polylogued watch` for the long-running source
  watcher.
- Root `polylogue run` and its stage subcommands were removed; ingestion is
  daemon-owned through `polylogued run` and explicit `polylogue ingest PATH`
  requests.
- Legacy batch-run state, JSON run artifacts, run observers, and the `runs`
  schema table were removed; schema v8 archives are fresh-only.
- Browser-capture receiver serving/status moved from root `polylogue
  browser-capture` to `polylogued browser-capture`.
- `polylogued status` now reports configured daemon components, including live
  watch roots and the browser-capture receiver target.
- `polylogue doctor --daemon` now includes the same daemon component status in
  the interactive health surface.
- Live watcher cursor and failure state now live in the archive database and
  failed ingests remain retryable after backoff instead of being recorded as
  successful cursor progress.
- `polylogued status` and daemon ingestion events now expose live cursor
  backlog, retry state, batch counters, byte deltas, and convergence timings.
- Live daemon ingestion now uses cursor offsets for append-only JSONL growth so
  completed tails can be read and merged without re-reading unchanged source
  prefixes.
- Daemon convergence now uses the live watcher's batched ingest path as the
  only source-ingest path; post-ingest convergence stages only repair FTS,
  embeddings, and insights.
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
- Schema v2 archives that already have `messages.message_type` skip the
  message-type backfill scan and only repair the missing version/index state.
- Schema v3 archives now upgrade to v4 by rebuilding action-event FTS rows
  with base-table rowids, enabling targeted incremental FTS repairs instead
  of archive-wide FTS scans.
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
