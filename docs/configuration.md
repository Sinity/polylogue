[← Back to README](../README.md)

# Configuration

## File Layout

Polylogue follows XDG Base Directory specification:

```
~/.local/share/polylogue/           # XDG_DATA_HOME/polylogue
├── source.db                       # raw acquisition log
├── index.db                        # Parsed/searchable/read-model index
├── embeddings.db                   # Vector index and embedding status
├── user.db                         # User marks, corrections, and annotations
├── ops.db                          # Daemon/cursor/telemetry state
├── blob/                           # Stored attachment/blob payloads
└── browser-capture/                # Browser-capture artifact spool

~/.claude/projects/                  # Auto-discovered: Claude Code sessions
~/.codex/sessions/                   # Auto-discovered: Codex sessions

~/.config/polylogue/                # XDG_CONFIG_HOME/polylogue
└── polylogue-credentials.json      # Google OAuth credentials (if using Drive)

~/.local/state/polylogue/           # XDG_STATE_HOME/polylogue
└── token.json                      # OAuth token cache
```

## Canonical Paths

Polylogue resolves all filesystem locations through `polylogue.paths` using
XDG environment variables. The defaults (when no override is set) are:

| Location | Default path | Override |
|---|---|---|
| Archive root | `~/.local/share/polylogue/` | `POLYLOGUE_ARCHIVE_ROOT` |
| Source DB | `~/.local/share/polylogue/source.db` | (follows archive root) |
| Index DB | `~/.local/share/polylogue/index.db` | (follows archive root) |
| Embeddings DB | `~/.local/share/polylogue/embeddings.db` | (follows archive root) |
| User DB | `~/.local/share/polylogue/user.db` | (follows archive root) |
| Ops DB | `~/.local/share/polylogue/ops.db` | (follows archive root) |
| Blob store | `~/.local/share/polylogue/blob/` | (follows data home) |
| Config file | `~/.config/polylogue/polylogue.toml` | `POLYLOGUE_CONFIG` |
| Config dir | `~/.config/polylogue/` | `XDG_CONFIG_HOME` |

Use `polylogue config paths` to print the resolved paths for the current
environment, or `polylogue config paths --format json` for machine-readable
output. The command also reports any detected bind mounts.

### Bind Mounts and `/realm/data/captures/`

Some deployments (such as the [sinnix](https://github.com/Sinity/sinnix)
NixOS configuration) expose the archive at two path strings through a
btrfs subvolume bind mount:

```
~/.local/share/polylogue/       → same files as below
/realm/data/captures/polylogue/ → same files as above
```

Both paths refer to the **same physical directory** — they share device
and inode numbers. The bind mount is transparent to Polylogue; the
daemon, CLI, and MCP server all operate on whichever path they resolve.

**Which path to use in `polylogue.toml`:**

- Use `~/.local/share/polylogue` (the XDG default) as the canonical
  path in configuration and MCP server settings. It is portable across
  hosts and does not depend on the sinnix data-lake layout.
- `/realm/data/captures/polylogue` is the same directory via bind mount
  and is equally correct at runtime. It only exists on hosts that run
  the sinnix configuration; referencing it from config files makes them
  host-specific.

If both path strings appear in logs or tool output, they do **not**
indicate two separate archives or a misconfiguration — they are the
same directory tree.

## Input Conventions

- `polylogued run` watches configured source roots and owns ingestion.
- Use `polylogue import PATH` to ask the running daemon to import an explicit
  file or directory.
- Directory names are for organization only; providers are detected from content.
- Supported source formats include `.json`, `.jsonl`, and `.zip`.

## Configuration Model

Polylogue has a small layered configuration surface and a larger archive state
surface. Keep those apart: startup config tells processes how to bind, where to
read/write, which provider work is allowed, and how to present output. User
annotations, saved views, daemon cursors, telemetry, and one-shot smoke results
live in SQLite/log state instead.

Layer precedence is highest first:

1. CLI flag overrides for that invocation.
2. `POLYLOGUE_*`, provider-specific, and presentation environment variables.
3. User `polylogue.toml`, resolved from `$POLYLOGUE_CONFIG`, then
   `$XDG_CONFIG_HOME/polylogue/polylogue.toml`, then `./polylogue.toml`.
4. Site `polylogue.toml`, resolved from `$POLYLOGUE_SITE_CONFIG` or
   `/etc/polylogue/polylogue.toml`.
5. Built-in defaults: local loopback binds, embeddings off, browser capture
   restricted to extension origins, and no remote auth-free write surface.

Use `polylogue config` to print the effective configuration as redacted TOML.
Use `polylogue config --show-layers` for a human provenance view. Use
`polylogue config --format json` for the machine contract; it returns the
redacted effective value, source layer, owner class, reload behavior, TOML path,
environment variable, CLI override, default, and inventory row for each public
key.

The JSON shape is intentionally stable for automation:

```json
{
  "layers": {
    "default": "built-in defaults",
    "site": {"path": "/etc/polylogue/polylogue.toml", "exists": false},
    "user": {"path": "/home/user/.config/polylogue/polylogue.toml", "exists": true},
    "env": "POLYLOGUE_*, provider credential, and presentation environment variables",
    "cli": "CLI overrides (per-invocation)"
  },
  "values": {
    "api_auth_token": {
      "value": "<set>",
      "source_layer": "env",
      "secret": true,
      "secret_present": true,
      "toml_path": "daemon.api.auth_token",
      "env_var": "POLYLOGUE_API_AUTH_TOKEN",
      "owner_class": "network-security",
      "reload_behavior": "startup-bound"
    }
  },
  "inventory": [
    {
      "key": "api_auth_token",
      "toml_path": "daemon.api.auth_token",
      "env_var": "POLYLOGUE_API_AUTH_TOKEN",
      "redaction": "presence"
    }
  ]
}
```

Secrets are never printed in cleartext. Set secrets show as `<set>` and unset or
empty secrets show as `<unset>`, while `secret_present` preserves the useful bit
for diagnostics.

The JSON form also includes a `diagnostics` array. Each item has stable
`code`, `severity`, `key`, `toml_path`, `env_var`, `message`, and
`next_action` fields so deployment smoke, agents, and shell scripts can tell
configuration debt from daemon/runtime failures. Diagnostics also include the
redacted effective `value`, `source_layer`, `secret`, and `secret_present` when
the source can be determined, plus `related_keys` for multi-key contradictions.
Current diagnostics cover operator-supplied path-layout problems, unsafe
network/auth combinations, API/browser-capture port conflicts, web-origin
browser capture without bearer auth, and `embedding.enabled = true` without a
configured Voyage key.

### State classes

| Class | Where it lives | Examples | Reload behavior |
| --- | --- | --- | --- |
| Static startup config | TOML/env/CLI | archive root, API host/port/token, browser-capture host/port/spool/origins, source roots | Restart `polylogued` after changing. |
| Deployment policy | TOML/env/Nix/HM/systemd | remote-bind opt-in, auth requirements, systemd memory/IO limits, schema validation mode | Restart the managed service; policy is outside archive content hashes. |
| Runtime mutable user state | `user.db` | tags, marks, saved views, workspaces, assertions, authored overlays | Mutated through CLI/API; not TOML and not source content. |
| Provider/cost controls | TOML/env | `embedding.enabled`, `embedding.max_cost_usd`, `VOYAGE_API_KEY` | Embedding loops read the gate/cost controls; no provider call happens unless explicitly enabled and credentials are present. |
| Presentation preferences | TOML/env/CLI | `logging.force_plain`, `no_color`, `ui.theme`, `NO_COLOR`, slow-query notices | Per CLI process or web render; does not change archive meaning. |
| Disposable ops state | `ops.db`, logs, smoke JSON | health cursors, convergence debt, deployment-smoke evidence, cgroup signals | Rebuildable/diagnostic; do not place in source archives. |

### TOML schema

```toml
[archive]
root = "/home/user/.local/share/polylogue"

[daemon]
host = "127.0.0.1" # legacy alias used by Nix/HM when api/browser host is omitted
port = 8766         # legacy API port alias

[daemon.api]
host = "127.0.0.1"
port = 8766
# auth_token = "..."   # required for non-loopback API binding

[daemon.browser_capture]
host = "127.0.0.1"
port = 8765
allowed_origins = "chrome-extension://*"
allow_remote = false
# auth_token = "..."   # required for remote binding or web origins
# spool_path = "/home/user/.local/share/polylogue/browser-capture"

[daemon.watch]
debounce_s = 2.0

[sources]
roots = ["/home/user/.claude/projects", "/home/user/.codex/sessions"]

[embedding]
enabled = false
model = "voyage-4"
dimension = 1024
max_cost_usd = 5.0   # soft monthly cap; 0 = unlimited
# voyage_api_key = "..." # prefer VOYAGE_API_KEY from a secret manager

[observability]
enabled = false
otlp_max_body_bytes = 8388608

[logging]
level = "INFO"
force_plain = false

[ui]
theme = "auto" # auto, dark, or light
# slow_query_notice_seconds = 2.5

[schema]
validation = "advisory" # off, advisory, or strict

[notifications]
backend = "log"
# webhook_url = "https://..."
# webhook_secret = "..."

[notifications.email]
port = 587
use_tls = true
use_starttls = true
max_per_hour = 12

[health]
check_interval_s = 300
check_tiers = "fast"
blob_integrity_sample_size = 100

# Convergence-debt alert thresholds. The daemon raises a typed HealthAlert
# when the per-family count of live_convergence_debt rows crosses these levels.
[health.convergence_debt]
default_warning = 1
default_error = 10
dedup_window_s = 3600

[health.convergence_debt.families.claude-code-session]
warning = 1
error = 5

[health.cursor_lag]
default_warning_s = 300
default_error_s = 900

[[cost.subscription.plans]]
name = "team-monthly"
period = "monthly"
included_usd = 0
```

The TOML key map is executable: `polylogue.config` owns the inventory consumed
by the loader, redacted TOML renderer, and JSON effective-state payload. When a
new public key is added, inventory coverage tests require a TOML/env/default
classification instead of letting a second ledger drift away from the code.

### Safety gates

Remote API binding fails closed. `polylogued run --api-host 0.0.0.0` requires
both `--insecure-allow-remote` and an API auth token, or the equivalent TOML/env
settings:

```toml
[daemon.api]
host = "0.0.0.0"
auth_token = "..."

[daemon.browser_capture]
allow_remote = true
auth_token = "..."
```

Browser-capture remote opt-in and web origins fail closed without a
browser-capture auth token. Extension origins such as `chrome-extension://*` are
allowed by default for the local receiver; ordinary web origins such as
`https://workbench.example` require `daemon.browser_capture.auth_token` or
`POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN`. The config diagnostics payload reports
these failures before daemon startup, including whether the missing token would
come from the default, TOML, env, or CLI layer.

Independent of the above: the receiver requires a bearer token by default on
every request (GET and POST alike), even on plain loopback with only the
default extension origin allowed. If `daemon.browser_capture.auth_token` /
`POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN` is unset, one is auto-minted/loaded from
a 0600 file (`polylogued browser-capture token show` prints it for pairing with
the extension popup). `daemon.browser_capture.allow_no_auth` /
`POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH` is the explicit, logged opt-out.

Embeddings are provider-cost work. `VOYAGE_API_KEY` alone supplies a credential
but does not enable daemon embedding convergence. Set `embedding.enabled = true`
or `POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS=true` to opt in; without a provider key,
readiness reports disabled/error state instead of attempting provider calls.

### Presentation and theme policy

`logging.force_plain` and `POLYLOGUE_FORCE_PLAIN` force plain CLI output and
avoid Rich layout primitives. `NO_COLOR` appears in the config inventory as the
env-only `no_color` presentation key and is honored by CLI formatting as a
no-color/plain-output signal for terminal compatibility. `ui.theme` and
`POLYLOGUE_THEME` select semantic token mode: `auto`, `dark`, or `light`. The
semantic token names are stable; individual palettes are UI implementation, not
a deployment policy knob.

Presentation preferences never enter the source/archive content hash and should
not be stored as user assertions. They are per process, per terminal, or per web
render.

### Nix and Home Manager semantics

The bundled NixOS module (`services.polylogue`) and Home Manager module
(`programs.polylogued`) render the same `polylogue.toml` schema that non-Nix
users write by hand. `polylogued run` reads that TOML/env effective state, so
module settings do not need to be duplicated as `ExecStart` flags; secret-bearing
values should come from environment/secret-manager wiring whenever possible
instead of being rendered into the Nix store.

Nix/HM service settings such as `service.memoryHigh`, `service.memoryMax`,
`service.nice`, and `service.ioWeight` are deployment policy. They affect the
systemd unit and show up in deployment-smoke resource evidence when the platform
exposes cgroup files, but they are not Polylogue archive config keys.

Filesystem layout is owned by `polylogue.paths`, which reads directory
environment variables lazily when its path functions are called. Path-safety
helpers are separate from both surfaces. Code that sanitizes provider or session
names imports from `polylogue.paths.sanitize`.

### Additional keys

A few keys not shown in the full example above, with their TOML path:

| Key | TOML path | Meaning |
| --- | --- | --- |
| `daemon_client_mode` | `daemon.client_mode` | How the CLI/MCP client reaches the daemon: `auto` (default), or an explicit forced mode. |
| `no_daemon` | `client.no_daemon` | Force direct in-process archive access, bypassing the daemon client even when one is reachable. |
| `debug_timing` | `ui.debug_timing` | Emit per-stage timing diagnostics in CLI output. |
| `hermes_root` | `sources.hermes.root` | Runtime root watched for Hermes state, snapshots, NeMo Relay ATIF/ATOF artifacts, and verification evidence. Defaults to `~/.hermes`. |
| `hook_sidecar_dir` | `sources.hook_sidecar_dir` | Directory for hook-event sidecar files consumed by the Claude Code/Codex hook harness. |
| `backup_verify_tmpdir` | `maintenance.backup_verify_tmpdir` | Scratch directory for backup-restore verification; defaults to the system temp dir when unset. |
| `antigravity_language_server` | `sources.antigravity_language_server` | Path to an Antigravity language-server binary, when parsing Antigravity sessions needs it. |
| `ingest_commit_batch_messages` | `sources.ingest_commit_batch_messages` | Messages per commit batch during ingest (default 8000). |
| `ingest_parse_workers` | `sources.ingest_parse_workers` | Parallel parse workers during ingest (default 1). |
| `live_full_ingest_workers` | `sources.live_full_ingest_workers` | Parallel workers for a live full-reingest pass (default 1). |

## Environment Policy

Environment variable precedence is:

1. XDG roots define the base config, data, cache, and state directories.
2. `POLYLOGUE_ARCHIVE_ROOT` overrides the archive root and the archive
   databases under it (`source.db`, `index.db`, `embeddings.db`, `user.db`,
   and `ops.db`).
3. `POLYLOGUE_CONFIG` and `POLYLOGUE_SITE_CONFIG` select config files.
4. `POLYLOGUE_*` runtime variables override matching TOML values.
5. Provider credentials such as `VOYAGE_API_KEY` supply secrets; they do not
   enable provider spend by themselves.

Common runtime overrides:

| Variable | Config key | Description |
|----------|------------|-------------|
| `XDG_CONFIG_HOME` | path base | Base directory for `polylogue.toml` and Drive credentials. |
| `XDG_DATA_HOME` | path base | Base directory for the archive, blob store, Drive cache, and browser-capture spool. |
| `XDG_CACHE_HOME` | path base | Base directory for cache/index output. |
| `XDG_STATE_HOME` | path base | Base directory for OAuth token and runtime state. |
| `POLYLOGUE_CONFIG` | config layer | Explicit user config path. |
| `POLYLOGUE_SITE_CONFIG` | config layer | Explicit site config path; empty disables site config. |
| `POLYLOGUE_ARCHIVE_ROOT` | `archive_root` | Override the archive root. |
| `POLYLOGUE_HERMES_ROOT` | `hermes_root` | Override the Hermes runtime root watched by the daemon. |
| `POLYLOGUE_DAEMON_URL` | `daemon_url` | CLI/MCP client daemon base URL. |
| `POLYLOGUE_API_HOST` / `POLYLOGUE_API_PORT` | `api_host` / `api_port` | Daemon HTTP API bind. |
| `POLYLOGUE_API_AUTH_TOKEN` | `api_auth_token` | API bearer token; redacted in config output. |
| `POLYLOGUE_BROWSER_CAPTURE_HOST` / `POLYLOGUE_BROWSER_CAPTURE_PORT` | browser-capture bind | Receiver bind host/port. |
| `POLYLOGUE_BROWSER_CAPTURE_ALLOWED_ORIGINS` | `browser_capture_allowed_origins` | Comma-separated receiver CORS origins. |
| `POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE` | `browser_capture_allow_remote` | Explicit remote-bind opt-in. |
| `POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN` | `browser_capture_auth_token` | Receiver bearer token; redacted. Auto-minted/loaded if unset. |
| `POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH` | `browser_capture_allow_no_auth` | Explicit opt-out of the auto-minted receiver token. |
| `POLYLOGUE_BROWSER_CAPTURE_SPOOL_PATH` | `browser_capture_spool_path` | Receiver spool override. |
| `POLYLOGUE_FORCE_PLAIN` | `force_plain` | Force plain output. |
| `POLYLOGUE_THEME` | `theme` | `auto`, `dark`, or `light`. |
| `NO_COLOR` | `no_color` | Standard no-color request; any non-empty value makes CLI output ANSI-free/plain. |
| `VOYAGE_API_KEY` | `voyage_api_key` | Voyage credential; redacted and spend-gated. |
| `POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS` | `embedding_enabled` | Enable daemon embedding convergence. |
| `POLYLOGUE_OBSERVABILITY_ENABLED` | `observability_enabled` | Enable OTLP/observability HTTP ingestion. |
| `POLYLOGUE_CREDENTIAL_PATH` | Drive auth | OAuth client JSON path. |
| `POLYLOGUE_TOKEN_PATH` | Drive auth | OAuth token path. |

### Theme and no-color policy

Human-facing terminal output becomes plain when `--plain`,
`POLYLOGUE_FORCE_PLAIN`, `NO_COLOR`, or a non-interactive terminal path requests
it. Plain mode removes Rich layout and ANSI escapes before any color theme is
applied. When rich rendering is allowed, `POLYLOGUE_THEME` or `[ui] theme`
selects semantic tokens for terminal panels, code/diff highlighting, and HTML
exports. Future palette sources such as pywal should feed those semantic tokens
rather than adding one-off colors at call sites.

## Backup and Export

The archive uses SQLite files with different durability classes:

- `source.db`, `user.db`, `embeddings.db`, and `blob/` are the expensive or
  irreplaceable state to prioritize in backups.
- `index.db` is rebuildable from `source.db`, but a backup avoids a full
  reindex after restore.
- `ops.db` is disposable daemon state; back it up only when preserving
  operational history matters.

For an offline file-level backup, stop the daemon and copy the archive root or
the specific tier files you need. Include matching `-wal` and `-shm` companions
when copying a live WAL-mode SQLite database.

To export all sessions as JSON:

```bash
polylogue find 'since:1970-01-01' then read --all --format json > sessions.json

# Or with filters
polylogue --origin claude-ai-export find 'origin:claude-ai-export' then read --all --format json > claude-sessions.json
```

Polylogue does not copy one-shot input exports into a managed import directory.
Keep the original provider export files if you need a reproducible rebuild from
the exact downloaded payloads.

## Google Drive Integration

For Gemini sessions via Google Drive:

1. Create OAuth credentials at [Google Cloud Console](https://console.cloud.google.com/)
2. Download to `~/.config/polylogue/polylogue-credentials.json`
3. Run `polylogue ops auth` to complete OAuth flow

Polylogue syncs the fixed `Google AI Studio` folder name used by Gemini exports.
The `drive_credentials_path` and `drive_token_path` config keys override the
default OAuth credential/token file locations under `$XDG_CONFIG_HOME` /
`$XDG_STATE_HOME`.

## Observability

Polylogue exposes daemon health through `polylogued status` and
`polylogue ops status`. The archive database stores ingestion state, live cursors,
and derived read-model freshness so automation does not need to scrape terminal
output.

### Health Checks

- `polylogue ops doctor` validates config, archive root, DB reachability, index status, and Drive credential/token presence.
- `polylogue ops doctor --repair` runs safe derived-data maintenance.
- `polylogue ops doctor --cleanup` runs destructive archive cleanup; preview it first.
- `polylogue ops doctor --repair --vacuum` compacts the database after maintenance.
- Workstation-specific policy such as cgroup slice placement and hard caps belongs in the host environment, not in the product CLI.

---

**See also:** [CLI Reference](cli-reference.md) · [MCP Integration](mcp-integration.md) · [Data Model](data-model.md)
