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

Every block below is a real command against the author's own archive —
Polylogue's own development history, ingested from Codex CLI and Claude
Code sessions that worked on this repository. Output is trimmed with `...`
where noted; nothing is invented.

Search it. One query hits Codex and Claude Code sessions at once — here, a
recurring clock-hygiene lint check landing in both:

```console
$ polylogue --limit 4 find 'repo:polylogue "with host-clock calls"'
1. claude-code-session  sure, I'm going AFK. Do not halt. Remember the word indefinitely. Do start by ex...  test files scanned: 618 files [with host-clock calls]: 23 allowlisted: 23 violations: 0
2. claude-code-session  3b0038ec-234c-44b8-bb88-fe222b44fe0f:agent-a97318edcffaac69d  test files scanned: 738 files [with host-clock calls]: 22 allowlisted: 26 violations: 0
3. claude-code-session  8dab1c19-ef6f-4f49-b4c1-cd9084c1b582  test files scanned: 894 files [with host-clock calls]: 27 allowlisted: 31 violations: 0 "total_duration_s": 28.98, "exit_code": 0 }
4. codex-session  019e1a5c-0abe-71e1-8adf-8ae8d4cc71a6  Chunk ID: 7b81ce Wall time: 0.0000 seconds Process exited with code 0 Original token count: 22 Output: test files scanned: 598 files [with host-clock calls]: 23 allowlisted: 23 violations: 0
```

Read one back as a transcript — real coding-agent sessions carry a lot of
injected system/skill context before the actual exchange, so this excerpt
trims that noise down to the assistant's opening move on a real task:

```console
$ polylogue find 'id:codex-session:019f4b85-1e4a-7060-a8c3-36e4b4175ff2' then read --view transcript
# 019f4b85-1e4a-7060-a8c3-36e4b4175ff2
...
## assistant
I'll independently map the bead's acceptance criteria to the branch diff, then
trace each persistence and reparse path into its production implementation and
focused tests. I'll treat passing tests as evidence only after checking that
they exercise the claimed failure modes.
...
```

Query tool activity as data, not text. Tool calls are paired with their
results, and failure comes from the provider's `exit_code`/`is_error`
structure — not from grepping prose for the word "error" — so this is every
tool that ever failed across this repo's own coding sessions, ranked:

```console
$ polylogue "actions where session.repo:polylogue AND is_error:true | group by tool | count"
tool=Bash count=5663
tool=Read count=1399
tool=Edit count=1167
tool=shell count=533
tool=exec_command count=149
...
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

No accounts, no personal data, no API keys — `demo seed` builds a small
synthetic archive through the real ingestion path, so the commands above
work against something before you point Polylogue at your own history:

```bash
polylogue demo seed      # writes a synthetic archive to POLYLOGUE_ARCHIVE_ROOT
polylogue demo verify    # checks it round-tripped correctly
```

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
