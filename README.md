# Polylogue

<p align="center">
  <a href="https://pypi.org/project/polylogue/"><img src="https://img.shields.io/pypi/v/polylogue?label=PyPI" alt="PyPI release"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-4584b6?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="https://github.com/Sinity/homebrew-polylogue"><img src="https://img.shields.io/badge/Homebrew-tap-fbb040?logo=homebrew&logoColor=111827" alt="Homebrew tap"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/ci.yml?branch=master&label=ci" alt="CI status"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/codeql.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/codeql.yml?branch=master&label=CodeQL" alt="CodeQL analysis"></a>
  <a href="https://sinity.github.io/polylogue/"><img src="https://img.shields.io/badge/docs-live-2563eb" alt="Live documentation"></a>
</p>

**Polylogue archives your AI conversations — all of them, in one place, on your machine.**

Every AI tool keeps its own history in its own format in its own corner:
Claude Code writes JSONL under `~/.claude`, Codex under `~/.codex`, ChatGPT
and Claude web only give you export zips, and none of them can search across
the others. Polylogue ingests all of it into a set of SQLite files you own,
normalized into one model — sessions, messages, content blocks, tool calls
and their results — and serves it back through a query-first CLI, an MCP
server for agents, a local HTTP reader, and a Python API.

## See it work

No accounts, no API keys, no personal data — `demo seed` builds a synthetic
archive through the real ingestion path:

```console
$ polylogue demo seed
Demo archive ready
  Sessions:     19
  Messages:     71
```

Search it. One query hits Codex, Claude Code, and ChatGPT sessions at once:

```console
$ polylogue find "clock"
1. chatgpt-export       Flaky clock test fix summary   The fixture used a shared [clock] instance; switching to an isolated clock fixed it.
2. claude-code-session  Fix the flaky clock test ...   F tests/test_[clock].py::test_uses_monotonic_clock 1 failed in 0.21s
5. codex-session        Fix the clock-sensitive test   exec_command pytest tests/test_[clock].py -q
8. codex-session        Fix the clock-sensitive test   {"metadata": {"exit_code": 1}, "output": "F tests/test_[clock].py..."}
...
```

Read one back as a transcript:

```console
$ polylogue find 'id:codex-session:demo-receipts' then read --view transcript
# Fix the clock-sensitive test and prove the suite passes.

## user
Fix the clock-sensitive test and prove the suite passes.

## tool
{"metadata": {"exit_code": 1}, "output": "F  tests/test_clock.py::test_uses_monotonic_clock\n1 failed in 0.18s"}

## assistant
All tests pass. The clock fix is complete.

## user
The receipt disagrees. Correct the fixture and verify again.
...
```

Query tool activity as data, not text. Tool calls are paired with their
results, and failure comes from the provider's `exit_code`/`is_error`
structure — not from grepping prose for the word "error":

```console
$ polylogue 'actions where is_error:true | group by tool | count'
tool=Bash count=4
tool=exec_command count=2
tool=Edit count=1
```

That is the product: everything your AI tools ever produced, in one queryable
place, with tool activity modeled as work rather than chat.

## Use it on your own history

```bash
polylogue init      # detects Claude Code, Codex, Gemini CLI, Hermes, Antigravity under $HOME; writes polylogue.toml
polylogued run      # the daemon: ingests what init found, then keeps watching as you work
```

`polylogue init` records the sources present on your machine;
`polylogued run` ingests them and stays running — new sessions appear in the
archive as your tools write them. One-off files (a ChatGPT export zip, a
downloaded conversation) go through the same pipeline:

```bash
polylogue import ~/Downloads/chatgpt-export.zip
polylogue import some-file.json --explain   # show detection/parse decisions without importing
```

| You use | Origin | How it arrives |
|---|---|---|
| Claude Code | `claude-code-session` | daemon watches `~/.claude/projects` |
| Codex CLI | `codex-session` | daemon watches `~/.codex/sessions` |
| ChatGPT (web) | `chatgpt-export` | export zip, or opt-in browser capture |
| Claude (web) | `claude-ai-export` | export zip, or opt-in browser capture |
| Gemini / AI Studio | `aistudio-drive` | Drive / AI Studio exports |
| Gemini CLI | `gemini-cli-session` | daemon watch |
| Hermes (Nous) | `hermes-session` | runtime-root import (ATIF/ATOF artifacts) |
| Antigravity | `antigravity-session` | export import |

Each origin is captured at full fidelity — roles, prose, thinking, tool calls
and results, attachments, session metadata — as far as the source provides
them. Per-origin detail: [docs/provider-origin-identity.md](docs/provider-origin-identity.md).

## What the archive is

Five SQLite files plus a SHA-256 content-addressed blob store, under one
local directory. Ingestion flows raw bytes → parsed model → derived indexes;
one rule decides what must survive:

> **Source evidence and your own judgments are durable. Everything derived —
> search indexes, analytics, embeddings — is rebuildable from source.**

| File | Holds | Durability |
|---|---|---|
| `source.db` | raw acquired artifacts, hook events | durable |
| `index.db` | sessions, messages, blocks, actions, lineage, FTS, analytics | rebuildable |
| `embeddings.db` | optional semantic-search vectors | rebuildable |
| `user.db` | your notes, tags, corrections, judgments | durable |
| `ops.db` | daemon cursors and telemetry | disposable |

Modelling decisions that make queries trustworthy:

- **Tool outcomes are structural.** `tool_result_is_error` and
  `tool_result_exit_code` come from provider structure; unknown stays `NULL`
  instead of being guessed from prose.
- **Role ≠ authorship.** Providers encode injected runtime context as
  `role=user` messages; a separate `material_origin` column keeps
  human-authored text distinct from protocol noise, so "what did I actually
  write" and cost accounting stay honest.
- **Forks don't double-count.** Forks, resumes, subagents, and compaction
  physically replay a parent's prefix in the raw logs. The archive stores
  only the divergent tail plus a branch point and recomposes on read — so
  token totals and transcripts count copied history once.
- **Cost is modeled per lane.** Provider-reported usage, cached-token lanes,
  reasoning tokens, catalog prices, and subscription-credit views stay
  separate (Codex "input" is ~mostly cache reads; adding lanes together
  silently inflates spend).

## Interfaces

The CLI is query-first — `find QUERY then ACTION`, with a real query grammar
(fielded predicates, booleans, date ranges, pipeline stages):

```bash
polylogue find 'repo:polylogue since:7d' then analyze --facets
polylogue --origin claude-code-session find "migration" then read --view messages
polylogue find 'actions where tool:shell AND command:pytest' then read
polylogue find "urgent" then mark --tag-add review
```

**Agents** get the same archive over MCP — `polylogue-mcp` is a standalone
stdio server, read-only by default (write access is a config opt-in):

```json
{
  "mcpServers": {
    "polylogue": { "command": "polylogue-mcp", "args": ["--role", "read"] }
  }
}
```

An agent with this block can search its own past sessions, resume prior
work with compiled context, and audit what previous runs actually did. Setup:
[docs/mcp-integration.md](docs/mcp-integration.md).

**The daemon** (`polylogued run`) also serves a local HTTP reader and
metrics; **Python** callers get an async API over the same storage
([docs/data-model.md](docs/data-model.md)). Semantic search is an explicit
opt-in (`--semantic`) that requires configuring an embedding provider —
that is the only path where any text leaves your machine, and
`polylogue ops embed preflight` shows the cost before anything is sent.

## Install

```bash
pipx install polylogue          # or: uv tool install polylogue
brew tap sinity/polylogue && brew install polylogue
nix run github:Sinity/polylogue -- --help
```

All three routes ship the same three commands: `polylogue`, `polylogued`,
`polylogue-mcp`. Source checkout, NixOS/Home Manager modules, and
verification: [docs/installation.md](docs/installation.md).

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

Pre-1.0, under heavy daily dogfooding against the author's own multi-year,
multi-tool archive. The deterministic demo world, the normalized model, and
the CLI/MCP/HTTP/Python surfaces are real and tested; interfaces may still
change between releases. Roadmap lives in the committed
[Beads](https://github.com/steveyegge/beads) graph
([web board](https://sinity.github.io/polylogue/main/beads/), or `bd ready`
locally) — not in GitHub Issues.

## Development

```bash
devtools status          # repo state and next steps
devtools verify --quick  # format + lint + types + generated-surface check
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [TESTING.md](TESTING.md), and
[docs/devtools.md](docs/devtools.md).

## Security

Polylogue assumes a trusted single-user machine. The daemon binds to
loopback; protected routes use bearer tokens; browser capture is opt-in with
its own token. The archive contains whatever your sessions contain — source
code, secrets, personal conversations — so treat it like the private data it
is: use disk encryption, and read [docs/security.md](docs/security.md) and
[docs/daemon-threat-model.md](docs/daemon-threat-model.md) before exposing
anything beyond localhost.
