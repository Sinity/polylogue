# Polylogue

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-4584b6?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="MIT License"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/ci.yml?branch=master&label=ci" alt="CI status"></a>
</p>

**Polylogue is a local archive and cockpit for your AI sessions and agent
work.**

> **Skim this in 7 minutes.** Every AI tool you use scatters its transcripts —
> one JSON dump per vendor, one log stream per coding agent, one half-broken
> export per browser plugin. Polylogue pulls all of them into a single
> searchable archive on your own machine, then lets you ask what an agent did
> yesterday: what it cost, what it touched, what it tested, and what to resume
> next.
>
> - **30 seconds** — the bold line above: local, source-agnostic, idempotent.
> - **3 minutes** — this section, the [one-command demo](#try-it-in-one-command),
>   and the [architecture diagram](docs/architecture.md).
> - **30 minutes** — [architecture.md](docs/architecture.md),
>   [internals.md](docs/internals.md), and [search.md](docs/search.md). New
>   vocabulary is translated in the [glossary](docs/glossary.md).

It ingests the session files that ChatGPT, Claude (web and Claude Code),
Codex, Gemini (Gemini CLI and Google Drive exports), Antigravity, and Hermes
agents already leave on disk,
normalizes them into a content-addressed SQLite archive, and gives you
full-text and semantic search, materialized session insights, cost rollups,
git-correlation, a daemon HTTP reader, and an MCP bridge — all running on your
machine, against your data.

- **Local-first.** The archive lives under your XDG data directory. No cloud,
  no upload, and no API key unless you opt into semantic embeddings.
- **Source-agnostic.** Providers are detected by file shape, not filename or
  vendor SDK. The same archive holds web exports, IDE agents, and CLI agents
  side by side.
- **Idempotent.** Re-ingesting the same data is a no-op; content hashes
  prevent duplication, and editable metadata (tags, summaries) does not
  trigger re-import.

## Try it in one command

No real data and no API key required. This seeds a throwaway, deterministic
demo archive (three sessions — one each from ChatGPT, Claude Code, and Codex)
and shows them coexisting in a single queryable substrate:

```bash
export POLYLOGUE_ARCHIVE_ROOT="$(mktemp -d)/archive"
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays
polylogue analyze --facets
```

```text
Facets (global) — matched result set:
  sessions: 3  messages: 19
  Provider origins:
    chatgpt-export: 1
    claude-code-session: 1
    codex-session: 1
  User tags:
    pytest-triage: 1
```

From here, `polylogue find "pytest" then read --view messages` renders the
matched transcript, and `polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT"`
checks the same semantic facts CI does. The screencast media for these flows is
regenerable from committed tape specs with a single command —
`devtools render visual-tapes --capture` (writes `.tape` files, and `.gif`s when
the `vhs` binary is present).

## Why this exists

AI sessions are scattered across one JSON dump per vendor, one JSONL stream
per agent, and one half-broken export per browser plugin. The transcripts are
valuable — running notes, debugging history, design rationale — but they only
become a corpus when something pulls them into a single, queryable substrate.

Polylogue is that substrate, and a cockpit over it:

- **Search across origins.** One query covers every captured session, with
  composable filters for origin, repo, date, tag, tool use, thinking blocks,
  paste detection, and semantic actions. FTS5 lexical search is always on;
  Voyage embeddings and hybrid (RRF) retrieval are opt-in.
- **Read agent work, not just chats.** `read` routes a matched session to a
  view: `summary`, `transcript`, `messages`, `raw`, `context`, `neighbors`
  (sessions around it in time), or `correlation` (the git commits a session
  produced).
- **Insights computed once, read many.** Session profiles, work events,
  phases, threads, tag rollups, and cost rollups are materialized into the
  archive so the CLI, the HTTP reader, the MCP bridge, and the Python API all
  see the same numbers.
- **Lineage & topology.** Continuations, forks, sidechains, and subagent runs
  are linked by typed topology edges and rolled up to one *logical session*, so
  you can ask which branch burned the time and which session is the real root
  worth turning into a skill. Exposed through `get_session_topology`,
  `get_logical_session`, and `get_session_tree`.
- **Resume context.** Polylogue assembles evidence-backed context for successor
  agents — what an interrupted session was doing, what it touched, and what to
  pick up next — and surfaces abandoned or resumable sessions through
  `find_resume_candidates`, `find_abandoned_sessions`, and `get_resume_brief`.
- **Built to be inspected.** Schema is fresh-first (no in-place upgrade chain
  to guess at), blob storage is content-addressed, and the daemon exposes
  health checks and Prometheus metrics.

## Archive model

Polylogue is organized around one rule: source evidence and user-authored
state are durable; search indexes, insight read models, embeddings, and daemon
telemetry are rebuildable. The archive uses a split-tier SQLite file set so
each class can be backed up, rebuilt, and inspected independently:

- `source.db` — raw acquisition evidence and source artifacts;
- `index.db` — parsed sessions, messages, FTS/search indexes, and derived
  insight read models;
- `embeddings.db` — opt-in vector rows and embedding catch-up state;
- `user.db` — irreplaceable user assertions and authored overlays;
- `ops.db` — disposable daemon telemetry, convergence debt, and local
  operational state.

Large binary content lives in a content-addressed blob store keyed by SHA-256.
The CLI, MCP server, daemon reader, and Python API all read through the same
archive/query substrate rather than maintaining separate stores.

See [docs/architecture.md](docs/architecture.md) for the system rings and data
flow, and [docs/internals.md](docs/internals.md) for hot files, invariants, and
extension points.

## Privacy and local-first

Everything Polylogue stores stays under your XDG data directory. Nothing is
uploaded, and no provider API key is needed unless you opt into semantic
embeddings (Voyage AI), which is off by default. The only network calls the
core makes are the ones you explicitly enable: Google Drive OAuth for Gemini
export ingestion, the optional embedding backfill, and the optional GitHub
cross-reference in `read --view correlation`.

If you run Polylogue inside a managed cloud-agent sandbox (Claude Code Web,
Codex Cloud), the data-handling tier follows the account you run under — the
repo cannot enforce it. The operator checklist, including which commands are
safe in a sandbox and which must never touch your real corpus, is in
[docs/cloud-agents.md](docs/cloud-agents.md).

## CLI shape

The root command is **query-first**. Use `find QUERY` for explicit searches,
then route the matched set into `read`, `select`, `analyze`, `mark`, `delete`,
or `continue`:

```bash
polylogue find "error handling"                  # search across all origins
polylogue --origin claude-code-session --since "last week" find "pytest" then read --all
polylogue --has-tool-use --typed-only read --all       # precomputed analytics
polylogue --action file_edit --tool bash read --all    # semantic action filters
polylogue --semantic find "sqlite locking" then read --all --limit 5
polylogue --latest read                          # render the most recent session
polylogue analyze --by origin                    # aggregates
```

You can also bind a query to an explicit verb with `find QUERY then VERB`. The
verbs that act on a matched set are `read`, `select`, `analyze`, `mark`,
`delete`, and `continue`:

```bash
polylogue find id:abc then read --view messages
polylogue find 'repo:polylogue since:7d' then analyze --facets
polylogue find 'repo:polylogue' then read --view correlation --since-hours 4
polylogue find 'repo:polylogue since:7d' then delete --dry-run
```

`analyze` folds the statistics and facet surfaces over the matched result set
(`--by <dimension>`, `--facets`). `read --view correlation` cross-references a
session against the git commits it produced; `read --view neighbors` shows the
sessions around it in time.

Pipeline and maintenance verbs are explicit:

```bash
polylogue init                       # first-run setup (writes polylogue.toml)
polylogue config completions --shell zsh
polylogue import ~/.claude/projects  # ingest sessions from a source path
polylogue ops doctor                 # FTS coverage, blob store, daemon liveness
polylogue ops status                 # daemon + archive status
polylogue config paths               # canonical archive paths
polylogue config --format json       # redacted effective config + source layers
polylogued run                       # daemon: convergence + insights + HTTP reader
polylogued watch                     # watch source dirs and ingest live
polylogue-mcp --role read            # MCP stdio bridge for AI assistants
```

See [docs/cli-reference.md](docs/cli-reference.md) for the full generated
command reference and [docs/search.md](docs/search.md) for the query grammar,
retrieval lanes, and ranking policy.

## Surfaces

### CLI

`polylogue` (aliases `plg`, `plog`) is the primary surface — query-first
search plus the verbs and maintenance commands above.

### Daemon HTTP reader

`polylogued run` starts the ingest daemon and an HTTP reader on `127.0.0.1`
that serves live archive search, session rendering, and insight views. It
uses the same query layer as the CLI, plus health checks
(`polylogued health`) and a Prometheus `/metrics` endpoint. See
[docs/daemon.md](docs/daemon.md). Agent runs project into
OpenTelemetry-shaped traces too: terminal query-unit rows map to spans, log
records, and Polylogue refs via the outbound OTel projection, so an external
observability tool can read session/cost/tool structure without copying message
text or local paths.

### MCP bridge

`polylogue-mcp --role read` exposes the archive as a Model Context Protocol
server so AI assistants can search, list, and retrieve sessions, insights, and
context images from their own sessions. See
[docs/mcp-integration.md](docs/mcp-integration.md). It also composes a context
preamble from the archive — recent lineage, project state, and resume guidance
for a seed session — so a coding agent can inject prior memory at SessionStart
via `compose_context_preamble` and `compile_context` (the same payload backs
`read --view context`).

### Browser capture

For ChatGPT and Claude.ai web sessions that have no on-disk export,
`polylogued browser-capture serve` runs a local receiver (`browser-capture
status` reports its state). The unpacked extension
lives in [`browser-extension/`](browser-extension/) and POSTs captured
sessions to the receiver as you browse. See
[docs/browser-capture.md](docs/browser-capture.md).

### Python API

Polylogue is library-first; the CLI wraps the Python API. The `Polylogue`
context manager owns the archive connection and exposes its repository's
async query methods:

```python
from polylogue import Polylogue

async with Polylogue.open() as archive:
    summaries = await archive.repository.list_summaries(
        title_contains="error handling",
        has_tool_use=True,
        limit=10,
    )
    for session in summaries:
        print(session.id, session.display_title)
```

See [docs/library-api.md](docs/library-api.md) for the full query surface.

## Installing from source

For development or to track `master`:

```bash
git clone https://github.com/sinity/polylogue
cd polylogue
direnv allow   # or: nix develop
```

The devshell installs git hooks, regenerates `AGENTS.md` from `CLAUDE.md`,
and provides the full toolchain (Python 3.13, mypy, ruff, pytest, mmdc).

To try the CLI without committing to an install:

```bash
nix run github:Sinity/polylogue -- --help
```

### Synthetic demo data

To explore features without importing real exports, use an isolated archive
root and schedule the approved deterministic demo fixture world:

```bash
export POLYLOGUE_DEMO_HOME="$(mktemp -d)"
export POLYLOGUE_ARCHIVE_ROOT="$POLYLOGUE_DEMO_HOME/archive"
export XDG_CONFIG_HOME="$POLYLOGUE_DEMO_HOME/config"

polylogue init
polylogued run
```

In a second terminal with the same environment:

```bash
polylogue import --demo --wait --timeout 30 --with-overlays
polylogue ops status
polylogue analyze
polylogue find "pytest" then read --all --limit 5
polylogue find "pytest" then read --view messages
polylogue find "pytest" then analyze --facets
```

`polylogue import --demo` writes only approved synthetic source files and asks
the running daemon to ingest them. Add `--wait` to block until the daemon-built
archive passes the same semantic demo checks used by `polylogue demo verify`;
`--with-overlays` then attaches the deterministic user-tier overlays for the
`pytest-triage` tag, mark, note, saved query, and typed assertions.

For source-only or CI/cloud verification without a daemon, use the direct demo
fixture commands:

```bash
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json
polylogue demo script --shell bash
```

## Developer tools

Repository maintenance, generated-surface rendering, validation lanes, and
verification all live behind `devtools`:

```bash
devtools --help
devtools status                  # repo health, generated-surface drift
devtools render all              # regenerate all generated docs and surfaces
devtools verify                  # local baseline before pushing a PR
```

See [docs/devtools.md](docs/devtools.md) for the full command catalog and
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## Verification lab

Operators choose focused checks via the verification baseline:

```bash
devtools verify --quick
devtools lab smoke run archive-smoke --tier 0
devtools lab smoke run reader-visual-smoke
```

## Mining the archive

`scripts/agent_forensics.py` turns an archive into a longitudinal findings
report — token economy, stored/provider-priced cost, catalog API-equivalent
estimates, subscription-credit views, cache amplification, model evolution, and
the adoption curve — with SVG charts. It is read-only, uses Polylogue's shared
pricing substrate, and is reproducible against any archive:

```bash
python scripts/agent_forensics.py --archive ~/.local/share/polylogue --out ./forensics
# or against the synthetic demo archive (no private data):
polylogue demo seed --root /tmp/demo --force --with-overlays --format json
python scripts/agent_forensics.py --archive /tmp/demo --out ./forensics-demo
```

See [docs/agent-forensics.md](docs/agent-forensics.md) for what it reports and
the token-accounting traps it handles (per-event deltas vs cumulative totals,
stored-vs-catalog cost provenance, subscription cache-read economics).

`scripts/cost_accounting_demo.py` proves the cross-provider cost accounting end
to end with no mocks: it ingests a crafted Codex session through Polylogue's
real writer, reads the materialized rollup back, and shows the corrected
disjoint billing lanes next to what the pre-fix code charged. Codex reports
input *inclusive* of cached and output *inclusive* of reasoning; billing those
naively double-counts, and because an agent re-sends its whole context each turn
cached is ~96% of input — a 7.69x cost inflation on the real corpus.

```bash
uv run python scripts/cost_accounting_demo.py
# operator cross-verify against Codex's authoritative token store (private):
uv run python scripts/cost_accounting_demo.py \
  --archive ~/.local/share/polylogue --codex-state ~/.codex/state_5.sqlite
```

See [docs/cost-model.md § Codex disjoint billing lanes](docs/cost-model.md#codex-disjoint-billing-lanes)
for the token semantics and cross-verification result.

<!-- BEGIN GENERATED: docs-surface -->
## Documentation

Live site: <https://sinity.github.io/polylogue/> (auto-published on every merge to `master`).

Start with the generated command and architecture references; use [docs/README.md](docs/README.md) for the complete map.

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System rings, ownership boundaries, and data flow. |
| [Installation](docs/installation.md) | Source checkout, Nix flake, and managed NixOS/Home Manager install paths. |
| [Architecture Spine](docs/architecture-spine.md) | Target shape, guardrails, and major decisions with rejected alternatives. |
| [Execution Plan](docs/execution-plan.md) | Current issue-driven sequencing plan for the remaining backlog. |
| [Design Direction](docs/design/README.md) | Historical design inputs and current guidance for using them without treating them as parallel roadmaps. |
| [Query-Action Workflows](docs/product/workflows.md) | Executable `find QUERY then ACTION` product contract for workflows, affordances, completions, and golden paths. |
| [CLI Reference](docs/cli-reference.md) | Generated command reference from live help output. |
| [Search & Query](docs/search.md) | Query grammar, retrieval lanes, ranking policy, and the typed SearchEnvelope contract. |
| [Browser Capture](docs/browser-capture.md) | Local browser extension capture for ChatGPT and Claude.ai sessions. |
| [Library API](docs/library-api.md) | Async archive API, filters, and query patterns. |
| [MCP Integration](docs/mcp-integration.md) | Model Context Protocol server setup and usage. |
| [Configuration](docs/configuration.md) | XDG paths, environment variables, and runtime configuration. |
| [Provider, Origin, and Source Identity](docs/provider-origin-identity.md) | Vocabulary map for provider-wire family, public origin, material source, capture mode, parser binding, and refs. |
| [Provider Package Completeness](docs/provider-completeness.md) | Readiness report for provider/importer package modes by origin and capture mode. |
| [Archive Backup](docs/archive-backup.md) | Archive-tier backup profiles, restore boundaries, and blob-GC safety rules. |
| [Developer Tools](docs/devtools.md) | `devtools` guide for generated surfaces, validation, and repo hygiene. |
| [Branch-Local Development Loop](docs/dev-loop.md) | Branch-local daemon, web-shell, browser-capture, and extension debugging workflow. |
| [Visual Evidence](docs/visual-evidence.md) | Synthetic reader DOM/media evidence lanes and local screenshot boundaries. |
| [Providers](docs/providers/README.md) | Provider-specific parsing and export-format notes. |

<!-- END GENERATED: docs-surface -->
