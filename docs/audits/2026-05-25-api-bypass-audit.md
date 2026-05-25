# API Bypass Audit: CLI, MCP, and Daemon Paths (#1584)

**Date**: 2026-05-25  
**Issue**: [#1584](https://github.com/Sinity/polylogue/issues/1584)  
**Source**: Governance audit [#1310](https://github.com/Sinity/polylogue/issues/1310) finding on [#723](https://github.com/Sinity/polylogue/issues/723)

## Background

Issue #723 (formalize Python library API) was closed with "core
implementation landed. Remaining edges are incremental."  Per the
governance audit, no successor issue tracked which edges remained.

This audit surveys the current state (post-#1022 column consolidation)
and classifies every surviving direct-access path.

## API Surface

The canonical Python API is `Polylogue` (`polylogue/api/__init__.py`),
a mixin-composed async class wrapping `ConversationRepository` +
`SQLiteBackend`.  CLI/MCP callers access it through:

- **CLI**: `env.polylogue` (an `AppEnv` attr, sync-bridged via
  `polylogue.api.sync.bridge.run_coroutine_sync`)
- **MCP**: `_tool_manager` → `ctx.polylogue` (a `Polylogue` instance)
- **Daemon**: Internal services (`SQLiteBackend` directly, since the
  daemon IS the API's runtime host)

## Findings

### CLI Commands

All 28 CLI commands were surveyed.  The vast majority route through
`env.polylogue` for archive operations (query, search, tags, insights,
cost, facets, neighbors, resume, export).  A small number of
maintenance-oriented commands use direct connections legitimately:

| Command | Access pattern | Why direct |
|---------|---------------|------------|
| `check` | `open_connection` for repair operations | Repair is a maintenance operation, not a read API call |
| `reset` | Direct filesystem operations | Schema bootstrap happens before the API exists |
| `doctor` | `open_connection` for deep repair | Same rationale as `check` |
| `embed` | `open_connection` for embedding backfill | Embedding catch-up is a daemon-side batch operation |
| `backup` | Direct filesystem copy | Not an archive operation |
| `config` | TOML read/write | Not an archive operation |

**Verdict**: All direct-access CLI paths are maintenance operations
that legitimately bypass the read API.  No query or search command
constructs queries independently of `env.polylogue`.

### MCP Tools

All MCP tools route through `ctx.polylogue` (a `Polylogue` instance).
The tool manager (`server_tools.py`) resolves tools by name and calls
API methods.  No MCP tool constructs queries independently.

**Verdict**: MCP is fully API-conformant.  No bypass paths found.

### Daemon HTTP Handlers

The daemon IS the API runtime — it owns the `SQLiteBackend` and
`ConversationRepository`.  Daemon HTTP handlers use internal services
(`self.queries`, connection pools) rather than the public `Polylogue`
class.  This is by design: the API is a facade over the daemon's
internals, not the other way around.

**Verdict**: Daemon handlers are intentionally direct.  This is not a
bypass — the daemon is the implementation substrate.

## Summary

| Surface | API-conformant | Intentionally direct | Bypass (needs fix) |
|---------|---------------|---------------------|-------------------|
| CLI | 22 commands | 6 maintenance commands | 0 |
| MCP | All tools | 0 | 0 |
| Daemon | N/A (is the substrate) | All handlers | 0 |

**Overall**: No CLI or MCP path constructs archive queries
independently of the Python API.  The original #723 AC ("every
operation that touches the archive should have an API method" and
"CLI and MCP tools call the same API methods") is satisfied in
current practice.  The six maintenance CLI commands that use direct
connections are legitimate exceptions.

## Recommendation

Close #1584.  The "remaining edges" from #723 have been absorbed
through normal architectural evolution.  If future maintenance
commands proliferate, revisit the API boundary for maintenance
operations specifically.
