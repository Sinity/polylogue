# Polylogue

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-4584b6?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="MIT License"></a>
  <a href="https://github.com/sinity/polylogue/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/sinity/polylogue/ci.yml?branch=master&label=ci" alt="CI status"></a>
</p>

**Polylogue is the system of record for AI work.** It keeps ChatGPT, Claude,
Codex, Gemini, Antigravity, Hermes, and coding-agent sessions in one local
archive, then lets you search, analyze, audit, and remember what actually
happened.

> **Skim this in 7 minutes.** Polylogue is not a chat export viewer. It is
> closer to git for the work around your code: prompts, tool calls, test runs,
> costs, dead ends, and resume points that otherwise disappear into vendor
> silos. It reads the files your tools already write, adds a browser-capture
> path for web chats, and keeps raw evidence next to every derived claim.
>
> - **30 seconds** — read the headline and the five questions below.
> - **3 minutes** — run the [demo](#try-it-without-private-data) and read
>   [Why you can trust it](#why-you-can-trust-it).
> - **30 minutes** — use [proof artifacts](docs/proof-artifacts.md),
>   [search](docs/search.md), [architecture](docs/architecture.md), and
>   [internals](docs/internals.md) to inspect the claims.

Polylogue answers questions ordinary transcript folders cannot:

- **What did the agent do?** Search sessions, messages, tool calls, files,
  observed events, costs, and outcomes with one query surface.
- **What failed?** Audit claims against structured evidence such as exit codes
  and tool-result metadata, not assistant prose saying "done".
- **What did it cost?** Keep provider-reported usage, catalog estimates, cache
  lanes, subscription-credit views, and coverage caveats separate.
- **What should resume?** Find interrupted work and assemble evidence-backed
  context bundles for the next agent.
- **Where is the raw evidence?** Keep parsed sessions, generated analytics,
  user notes, embeddings, daemon telemetry, and source bytes in inspectable
  local evidence stores.

The archive is local-first. It lives under your XDG data directory. There is no
upload and no API key unless you opt into semantic embeddings. Lexical search
and the core archive stay fully local; semantic search currently uses Voyage AI
when enabled. Re-ingesting the same source is idempotent by content hash, so
repeated imports coalesce instead of duplicating your history.

## Try it without private data

No real data and no API key required. This seeds a throwaway, deterministic
demo archive and verifies the same semantic facts used by CI:

```bash
export POLYLOGUE_ARCHIVE_ROOT="$(mktemp -d)/archive"
polylogue demo seed --root "$POLYLOGUE_ARCHIVE_ROOT" --force --with-overlays --format json
polylogue demo verify --root "$POLYLOGUE_ARCHIVE_ROOT" --require-overlays --format json
polylogue analyze --facets
```

The current demo corpus has 11 sessions, 43 indexed messages, five origins,
attachments with acquired bytes, browser-capture coalescing, lineage links,
subagent runs, terminal-state examples, synthetic embeddings, and user overlays.
The generated construct datasheet is tracked at
[docs/plans/demo-corpus-construct-audit.md](docs/plans/demo-corpus-construct-audit.md).

From the same throwaway archive:

```bash
polylogue find "pytest" then read --view messages
polylogue find "pytest" then analyze --facets
```

Screencast media for public flows is regenerable from committed tape specs with
`devtools render visual-tapes --capture` (writes `.tape` files, and `.gif`s
when `vhs` is present).

## Why this exists

AI work is valuable but poorly instrumented. The transcript is the running
notebook, the debug log, the design record, and often the only proof of what an
agent actually did before it claimed success. Vendor memory is per-vendor,
opaque, and non-portable by design. Polylogue makes that work cross-vendor,
local, queryable, and auditable.

The product is organized around four verbs:

- **Search** every captured source with lexical FTS, optional semantic search,
  and a query language that understands sessions, messages, tool calls,
  actions, files, observed events, context snapshots, and reviewed notes.
- **Analyze** costs, source coverage, model usage, work phases, tool families,
  claim-vs-evidence gaps, and temporal patterns across the archive.
- **Audit** every derived number back to source rows and blob-addressed raw
  bytes where possible; unavailable evidence is reported as unavailable rather
  than guessed.
- **Remember** by turning reviewed findings and notes into context bundles that
  future agents can read. Memory-benefit claims stay capability-phrased until
  measured uplift experiments prove the effect.

## Why you can trust it

Polylogue is built for evidence over plausible prose:

- Provider files are detected by shape, not by filenames or SDK wrappers.
- FTS is an invariant, not a best-effort cache; stale search readiness blocks
  user search until the index is demonstrably current.
- Tool outcomes are read from provider structure, such as `is_error` and exit
  codes, instead of regexing assistant text.
- Source evidence and user-authored notes are durable; indexes, analytics,
  embeddings, and daemon telemetry are rebuildable.
- The project deletes features that guess too much. When an inference cannot be
  supported by structure or review, the right output is a caveat, candidate, or
  unavailable field.

See [proof artifacts](docs/proof-artifacts.md) for the current claim-to-proof
map. The short version: the demo corpus proves private-data-free coverage, the
cost example proves disjoint token accounting, and the claim-vs-evidence packet
shows structured failure follow-up without exposing private transcripts.

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

Polylogue exposes usage forensics through the normal archive analysis surfaces:
coverage, provider/model cost rollups, monthly usage timelines, and structured
claim-vs-evidence packets. These commands read the archive without writing to
it and keep stored/provider-priced cost, catalog API-equivalent estimates,
subscription-credit views, cache amplification, model evolution, and adoption
curves as separate evidence streams:

```bash
polylogue analyze insights coverage --group-by month --format json
polylogue analyze insights cost-rollups --format json
polylogue analyze usage --format json --limit 0
polylogue analyze insights usage-timeline --group-by month-origin-model --format json

devtools workspace claim-vs-evidence --limit 5000 \
  --out-dir .agent/demos/claim-vs-evidence --json
```

See [docs/agent-forensics.md](docs/agent-forensics.md) for the supported
queries and the token-accounting traps they handle: per-event deltas vs
cumulative totals, physical-vs-logical token grain, stored-vs-catalog cost
provenance, and subscription cache-read economics.

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
| [Proof Artifacts](docs/proof-artifacts.md) | Claim-to-proof map for public-facing demo, cost, failure-follow-up, and affordance-analysis claims. |
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
