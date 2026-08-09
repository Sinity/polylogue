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
Polylogue archives AI conversations and coding-agent runs from multiple tools in
one searchable local archive. It imports supported histories from ChatGPT,
Claude and Claude Code, Codex, Gemini, Hermes, and other sources, then exposes
sessions, messages, tool calls and results, branches, subagents, usage, and costs
through a CLI, Python API, local HTTP reader, and MCP server.

By default, the archive stays on your machine. The author's archive contains
more than **18,000 sessions and 4.7 million messages**.

[Getting started](docs/getting-started.md) | [Live documentation](https://sinity.github.io/polylogue/) | [Demo](docs/demos.md) | [Architecture](docs/architecture.md) | [CLI reference](docs/cli-reference.md)

## Try it without importing personal data

Run the tour in a throwaway archive:

```bash
nix run github:Sinity/polylogue -- demo tour
```

The tour imports sample files for the supported providers, queries structured
tool results, and reconstructs a parent session with its child branch while
preserving source references. It does not require a provider account or access
to the author's archive.

From a source checkout:

```bash
git clone https://github.com/Sinity/polylogue.git
cd polylogue
nix develop -c polylogue demo tour
```

## What you can do

- Search conversations and coding sessions across supported sources.
- Inspect tool calls and provider-reported outcomes without inferring success or
  failure from prose.
- Follow branches, continuations, subagent sessions, and compacted histories.
- Distinguish human-authored text from injected runtime context.
- Query usage and cost without combining provider fields that mean different
  things.
- Add notes, tags, corrections, and judgments without changing imported source
  data.
- Let agents read the archive through MCP.

## Install

```bash
# Python package: CLI, daemon, and MCP server
pipx install polylogue
# or
uv tool install polylogue

# Homebrew CLI
brew tap sinity/polylogue
brew install polylogue

# Nix, without installing
nix run github:Sinity/polylogue -- --help
```

Verify the installation:

```bash
polylogue --version
polylogue --help
```

## Build a local archive

Detect supported sources already present on the machine:

```bash
polylogue init
```

Start continuous ingestion of configured live sources:

```bash
polylogued run
```

Import a one-off export or file:

```bash
polylogue import ~/Downloads/chatgpt-export.zip
polylogue import some-file.json --explain
```

Run queries:

```bash
polylogue "css refactor"
polylogue --origin claude-code-session find "migration" then read --view messages
polylogue "actions where tool:shell AND command:pytest | group by is_error | count"
```

`polylogue init` writes a starter `polylogue.toml` from detected sources.
`polylogued run` imports those sources and continues watching the live ones.

## Example: query tool outcomes

Tool execution is stored as structured data when the source provides it. Missing
outcome fields remain unknown rather than being guessed from assistant text.

```console
$ polylogue "actions where tool:Bash AND command:pytest | group by is_error | count"
is_error=0 count=12861
is_error=1 count=1039
is_error=unknown count=115
```

Example output from July 2026; abbreviated.

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

Parsers preserve the structure available in each source, including roles, prose,
thinking blocks, tool calls and results, attachments, and session metadata.
Provider-specific limits are documented in
[docs/provider-origin-identity.md](docs/provider-origin-identity.md).

## Storage

Polylogue keeps imported source artifacts and user-authored changes separate
from data that can be rebuilt.

| File | Contents | Durability |
|---|---|---|
| `source.db` | acquired source artifacts and runtime hook events | durable |
| `index.db` | normalized sessions, messages, blocks, actions, lineage, FTS, analytics | rebuildable |
| `embeddings.db` | optional semantic-search vectors | rebuildable |
| `user.db` | notes, tags, corrections, candidates, and judgments | durable |
| `ops.db` | daemon cursors, convergence state, and telemetry | disposable |

Large payloads are stored in a SHA-256 content-addressed blob store under the
same archive root.

The data model keeps several distinctions explicit:

- Unknown tool outcomes remain unknown.
- A message's provider role is stored separately from who or what supplied its
  content.
- Copied parent history is represented through lineage rather than counted as
  new work in every child session.
- Token counts are not combined when providers define them differently.

See [docs/data-model.md](docs/data-model.md) and
[docs/architecture.md](docs/architecture.md).

## Interfaces

### CLI

The query-first CLI supports full-text search, field filters, booleans, date
ranges, action queries, pipelines, and JSON output.

```bash
polylogue find 'repo:polylogue since:7d' then analyze --facets
polylogue find "urgent" then mark --tag-add review
polylogue find 'actions where tool:shell AND command:pytest' then read
```

### MCP

`polylogue-mcp` exposes the archive over stdio. It is read-only by default and
takes no CLI flags. Write, judge, and maintenance capability are each an
independent `polylogue.toml` `[mcp]` opt-in (or `POLYLOGUE_MCP_*_ENABLED` env
var) resolved at server startup, and remain subject to authorization and
confirmation rules.

```json
{
  "mcpServers": {
    "polylogue": { "command": "polylogue-mcp" }
  }
}
```

See [docs/mcp-integration.md](docs/mcp-integration.md).

### HTTP and Python

`polylogued run` can serve a local HTTP reader and metrics endpoint. Python
callers can use the asynchronous API over the same archive.

Semantic search is optional. It requires an embedding provider and is the only
normal path that sends archive text outside the machine. Run
`polylogue ops embed preflight` to inspect the work and estimated cost first.

<!-- BEGIN GENERATED: docs-surface -->
## Documentation

Live site: <https://sinity.github.io/polylogue/>.

Start with the task-oriented guides below. The complete documentation map is in [docs/README.md](docs/README.md).

| Document | Description |
|----------|-------------|
| [Getting Started](docs/getting-started.md) | Install Polylogue, create an archive, and run a first query. |
| [Installation](docs/installation.md) | Package, source-checkout, Nix, and managed deployment options. |
| [Demos and Proofs](docs/demos.md) | Run the private-data-free tour and see what each demo establishes. |
| [Proof Artifacts](docs/proof-artifacts.md) | Links between public claims and reproducible checks. |
| [Architecture](docs/architecture.md) | Storage, data flow, and component responsibilities. |
| [Code Navigation](docs/code-navigation.md) | Find the owning package, runtime path, and verification for a code change. |
| [Search & Query](docs/search.md) | Search syntax, filters, action queries, ranking, and output formats. |
| [CLI Reference](docs/cli-reference.md) | Commands and options generated from the current CLI. |
| [MCP Integration](docs/mcp-integration.md) | Configure an MCP client to read or write the archive. |
| [Configuration](docs/configuration.md) | Paths, environment variables, and runtime settings. |
| [Security](docs/security.md) | Local trust boundaries, authentication, and privacy controls. |
| [Developer Tools](docs/devtools.md) | Repository commands and validation checks. |
| [Providers](docs/providers/README.md) | Supported inputs and provider-specific parsing limits. |

<!-- END GENERATED: docs-surface -->

## Status

Polylogue is pre-1.0. Import, continuous ingestion, local query, the demo, and
the CLI, MCP, HTTP, and Python interfaces are implemented. Provider formats and
public interfaces may still change between releases.

## Security

Polylogue assumes a trusted single-user machine. The daemon binds to loopback,
protected routes use bearer tokens, and browser capture is opt-in with its own
token. The archive may contain source code, credentials, and personal
conversations. Use disk encryption and read [docs/security.md](docs/security.md)
and [docs/daemon-threat-model.md](docs/daemon-threat-model.md) before exposing
anything beyond localhost.

## Development

```bash
devtools status
devtools verify --quick
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [TESTING.md](TESTING.md), and
[docs/devtools.md](docs/devtools.md).

## License

MIT
