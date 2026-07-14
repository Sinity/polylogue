[← Back to Docs](README.md)

# Onboarding

A source checkout to a usable archive in three steps.

## 1. Enter the dev shell

```bash
git clone https://github.com/Sinity/polylogue.git
cd polylogue
nix develop
```

The dev shell exposes three executables:

| Executable | Purpose |
|------------|---------|
| `polylogue` | Query-first CLI — `find`/`read`/`analyze`/`mark`/`select`/`delete`/`continue` |
| `polylogued` | Ingest and convergence daemon |
| `polylogue-mcp` | MCP server bridge for AI agents |

## 2. First-run setup: `polylogue init`

```bash
polylogue init
```

`init` scans your home directory for AI chat session roots and writes a
starter `polylogue.toml` to `$XDG_CONFIG_HOME/polylogue/polylogue.toml`
(typically `~/.config/polylogue/polylogue.toml`). Both the daemon and
the CLI read the same file.

The canonical roots it looks for:

| Family | Default path | Notes |
|--------|--------------|-------|
| `claude-code` | `~/.claude/projects/` | Claude Code session JSONL |
| `codex` | `~/.codex/sessions/` | Codex session JSONL |
| `gemini-cli` | `~/.gemini/tmp/` | Gemini CLI workspace exports |
| `hermes` | `~/.hermes/` | Hermes `state.db` plus fallback session exports |
| `antigravity` | `~/.gemini/antigravity/` | Antigravity brain artifacts |
| `hooks` | `$XDG_DATA_HOME/polylogue/hooks/` | Agent hook sidecar spool |

Absent roots are written as commented hints so you can uncomment them
later without re-reading docs.

Useful flags:

```bash
polylogue init --dry-run            # preview without writing
polylogue init --force              # overwrite an existing config
polylogue init --format json        # machine-readable detection report
```

If you run a bare `polylogue` on a fresh install, the status hint will
point you at `polylogue init` until the starter file exists.

## 3. Start the daemon

```bash
polylogued run
```

The daemon watches the configured roots, ingests new sessions,
and keeps insights and FTS indexes converged. The first run does
a catch-up pass over everything it finds.

Check status at any time:

```bash
polylogue ops status
polylogued status
```

## 4. First search

```bash
polylogue "css refactor"
polylogue --since yesterday read --all
polylogue --latest read
```

See [Getting Started](getting-started.md) for the full quickstart and
[Search reference](search.md) for the query grammar.

## Recovery

If something looks wrong (missing FTS coverage, stale daemon, malformed
records), run the health-check command:

```bash
polylogue ops doctor
polylogue ops doctor --repair
polylogue ops doctor --runtime
```

`ops doctor` reports schema mismatches, FTS coverage gaps, blob-store
consistency, daemon liveness, and validation failures. `--repair`
recreates dropped FTS triggers and rebuilds the index where it can do
so safely; deeper recovery still requires the documented in-place
upgrade scripts described in [internals.md](internals.md#schema-versioning-model).

For backup boundaries (SQLite WAL, blob store, service state), see
[daemon.md](daemon.md).
