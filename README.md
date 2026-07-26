# Polylogue

<p align="center">
  <a href="https://pypi.org/project/polylogue/"><img src="https://img.shields.io/pypi/v/polylogue?label=PyPI" alt="PyPI release"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-4584b6?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://github.com/Sinity/homebrew-polylogue"><img src="https://img.shields.io/badge/Homebrew-tap-fbb040?logo=homebrew&logoColor=111827" alt="Homebrew tap"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/ci.yml?branch=master&label=ci" alt="CI status"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/codeql.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/codeql.yml?branch=master&label=CodeQL" alt="CodeQL analysis"></a>
  <a href="https://sinity.github.io/polylogue/"><img src="https://img.shields.io/badge/docs-live-2563eb" alt="Live documentation"></a>
</p>

<!-- public-claim:category.local-evidence-system -->
Polylogue keeps AI conversations and coding-agent runs from several tools in one
local archive. It ingests supported histories from ChatGPT, Claude and Claude
Code, Codex, Gemini, Hermes, and other sources, then normalizes them into one
model of sessions, messages, content blocks, tool calls and results, branches,
subagents, usage, and costs.

The archive stays on your machine. You can search it from the CLI, read it over
a local HTTP interface, query it from Python, or expose it to agents through
MCP.

The author's live archive currently contains more than **18,000 sessions and
4.7 million messages**, covering several providers and work dating back to
2022. The session count is a verified July 2026 figure after a repair stopped
counting hook events as standalone sessions. The hook records and their payloads
were retained and linked to their parent sessions.

[Getting started](docs/getting-started.md) | [Live documentation](https://sinity.github.io/polylogue/) | [Architecture](docs/architecture.md) | [CLI reference](docs/cli-reference.md)

## What Polylogue is for

AI work is usually split across vendor exports, JSONL logs, browser captures,
and tool-specific directories. Even when the data is available, each source has
a different model of messages, tool use, branches, and usage.

Polylogue provides one place to:

- search conversations and coding sessions across providers;
- inspect tool calls and their recorded outcomes;
- follow forks, resumes, subagents, and compaction lineage;
- distinguish human-authored material from injected runtime context;
- compile context from earlier work for a new agent session;
- calculate usage and cost without collapsing incompatible token lanes;
- attach notes, tags, corrections, and judgments without modifying source data.

## Quick start

Install one of the packaged releases:

```bash
pipx install polylogue
# or
uv tool install polylogue
# or
brew tap sinity/polylogue && brew install polylogue
# or
nix run github:Sinity/polylogue -- --help
```

Detect local sources and start the daemon:

```bash
polylogue init
polylogued run
```

Import a one-off export:

```bash
polylogue import ~/Downloads/chatgpt-export.zip
polylogue import some-file.json --explain
```

Run a query:

```bash
polylogue find 'repo:polylogue since:7d' then analyze --facets
polylogue --origin claude-code-session find "migration" then read --view messages
polylogue find 'actions where tool:shell AND command:pytest' then read
```

`polylogue init` writes a starter `polylogue.toml` from the sources found on the
machine. `polylogued run` imports those sources and continues watching the live
ones.

## Supported sources

| Source | Origin | Typical input |
|---|---|---|
| Claude Code | `claude-code-session` | watched JSONL under `~/.claude/projects` |
| Codex CLI | `codex-session` | watched sessions under `~/.codex/sessions` |
| ChatGPT | `chatgpt-export` | export archive or opt-in browser capture |
| Claude web | `claude-ai-export` | export archive or opt-in browser capture |
| Gemini and AI Studio | `aistudio-drive` | Drive or AI Studio export |
| Gemini CLI | `gemini-cli-session` | watched local session data |
| Hermes | `hermes-session` | runtime-root ATIF or ATOF artifacts |
| Antigravity | `antigravity-session` | exported session data |

Each parser preserves the detail present in its source, including roles, prose,
thinking blocks, tool calls and results, attachments, and session metadata.
Provider-specific caveats are documented in
[docs/provider-origin-identity.md](docs/provider-origin-identity.md).

## A real archive query

Tool execution is stored as structured data. A failed tool result comes from the
provider's `exit_code` or `is_error` field when one exists. Polylogue does not
guess failure by searching assistant prose for words such as "error".

```console
$ polylogue "actions where session.repo:polylogue AND is_error:true | group by tool | count"
tool=Bash count=5663
tool=Read count=1399
tool=Edit count=1167
tool=shell count=533
tool=exec_command count=149
...
```

The same data can separate successful, failed, and unreported `pytest` results:

```console
$ polylogue "actions where tool:Bash AND command:pytest | group by is_error | count"
is_error=0 count=12861
is_error=1 count=1039
is_error=unknown count=115
```

These examples use the author's live archive. Output is shortened where shown.

## Storage model

Polylogue uses five SQLite databases and a SHA-256 content-addressed blob store
under one local archive root.

> Source evidence and user-authored judgments are durable. Search indexes,
> embeddings, analytics, and other derived data can be rebuilt.

| File | Contents | Durability |
|---|---|---|
| `source.db` | acquired source artifacts and hook records | durable |
| `index.db` | normalized sessions, messages, blocks, actions, lineage, FTS, analytics | rebuildable |
| `embeddings.db` | optional semantic-search vectors | rebuildable |
| `user.db` | notes, tags, corrections, candidates, and judgments | durable |
| `ops.db` | daemon cursors, convergence state, and telemetry | disposable |

Several modeling choices are important for trustworthy queries:

- **Tool outcomes remain structural.** Unknown status stays unknown instead of
  being inferred from text.
- **Message role and authorship are separate.** Injected context may arrive as a
  `user` message even though a human did not write it.
- **Copied prefixes are counted once.** Forks, resumes, subagents, and compaction
  may replay parent history in raw logs. Polylogue stores lineage and the
  divergent tail rather than treating every copy as new work.
- **Usage lanes remain separate.** Provider-reported input, cache reads,
  reasoning tokens, catalog prices, and subscription-credit views are not added
  together unless the calculation is valid.

See [docs/data-model.md](docs/data-model.md) and
[docs/architecture.md](docs/architecture.md) for the full model.

## Interfaces

### CLI

The CLI is query-first and supports field filters, booleans, date ranges, action
queries, and pipelines:

```bash
polylogue find 'repo:polylogue since:7d' then analyze --facets
polylogue find "urgent" then mark --tag-add review
polylogue find 'actions where tool:shell AND command:pytest' then read
```

### MCP

`polylogue-mcp` exposes the archive over stdio. It is read-only by default.
Write access requires an explicitly configured role and remains subject to
authorization and confirmation rules.

```json
{
  "mcpServers": {
    "polylogue": { "command": "polylogue-mcp", "args": ["--role", "read"] }
  }
}
```

See [docs/mcp-integration.md](docs/mcp-integration.md).

### HTTP and Python

`polylogued run` serves a local HTTP reader and metrics endpoint. Python callers
can use the asynchronous API over the same archive.

Semantic search is optional. It requires an embedding provider, and it is the
only normal path that sends archive text outside the machine. Run
`polylogue ops embed preflight` to inspect the work and estimated cost before
sending anything.

## Demo data

You can exercise the real ingestion and query paths without importing personal
data:

```bash
polylogue demo seed
polylogue demo verify
```

The demo writes a small synthetic archive to `POLYLOGUE_ARCHIVE_ROOT`.

<!-- BEGIN GENERATED: docs-surface -->
## Documentation

Live site: <https://sinity.github.io/polylogue/>.

Start with the task-oriented guides below; [docs/README.md](docs/README.md) separates guides, reference, internals, operations, evidence, design, and historical records. Current sequencing and active workstreams live in the Beads backlog (`bd ready`, `bd list --status open`).

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | First archive, first query, and the next documentation steps. |
| [Installation](docs/installation.md) | Source checkout, Nix flake, and managed NixOS/Home Manager install paths. |
| [Demos and Proofs](docs/demos.md) | Reproducible proofs, construct-valid demo doctrine, and flagship demonstrations. |
| [Proof Artifacts](docs/proof-artifacts.md) | Claim-to-proof map for public-facing demo and evidence claims. |
| [Architecture](docs/architecture.md) | System rings, ownership boundaries, and data flow. |
| [Search & Query](docs/search.md) | Query grammar, retrieval lanes, ranking policy, and the typed SearchEnvelope contract. |
| [CLI Reference](docs/cli-reference.md) | Generated command reference from live help output. |
| [MCP Integration](docs/mcp-integration.md) | Model Context Protocol server setup and usage. |
| [Configuration](docs/configuration.md) | XDG paths, environment variables, and runtime configuration. |
| [Security](docs/security.md) | Security boundaries for local archives and readers. |
| [Developer Tools](docs/devtools.md) | Generated surfaces, validation, and repo hygiene. |
| [Providers](docs/providers/README.md) | Provider-specific parsing and export-format notes. |

<!-- END GENERATED: docs-surface -->

## Status

Polylogue is pre-1.0 and used daily against the author's multi-year archive.
The deterministic demo, normalized model, CLI, MCP, HTTP, and Python interfaces
are implemented and tested. Interfaces may still change between releases.

The roadmap lives in the committed
[Beads](https://github.com/steveyegge/beads) graph. Browse the
[web board](https://sinity.github.io/polylogue/main/beads/) or run `bd ready`
locally.

## Development

```bash
devtools status
devtools verify --quick
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [TESTING.md](TESTING.md), and
[docs/devtools.md](docs/devtools.md).

## Security

Polylogue assumes a trusted single-user machine. The daemon binds to loopback,
protected routes use bearer tokens, and browser capture is opt-in with its own
token. The archive may contain source code, credentials, and personal
conversations. Use disk encryption and read [docs/security.md](docs/security.md)
and [docs/daemon-threat-model.md](docs/daemon-threat-model.md) before exposing
anything beyond localhost.

## License

MIT
