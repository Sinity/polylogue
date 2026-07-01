# Glossary

Polylogue's docs use a precise internal vocabulary. This page translates it into
plain language for a first-time reader. For the full system shape see
[architecture.md](architecture.md); for the provider/origin/source split see
[provider-origin-identity.md](provider-origin-identity.md).

## Entry layers

Read only as deep as you need:

- **30 seconds** — Polylogue ingests the AI-session files ChatGPT, Claude,
  Codex, Gemini, and coding agents leave on disk into one local, searchable
  SQLite archive, and gives you search, insights, and an MCP/HTTP/CLI cockpit
  over them.
- **3 minutes** — read the [README](../README.md) top section and the
  [architecture diagram](architecture.md).
- **30 minutes** — read [architecture.md](architecture.md),
  [internals.md](internals.md), and the [search grammar](search.md).

## Terms

| Term | Plain-language meaning |
|------|------------------------|
| **Archive** | The on-disk SQLite file set under your XDG data directory that holds every ingested session. Not a single file — see *archive tier*. |
| **Archive tier** | One of the five SQLite databases the archive is split into by durability class: `source.db` (raw evidence), `index.db` (parsed sessions + search + insights), `embeddings.db` (vectors), `user.db` (your tags/notes/assertions), `ops.db` (disposable daemon telemetry). Tiers are backed up and rebuilt independently. |
| **Source / origin / provider** | Three scopes of "where a session came from". *Source* = the material runtime root and lab attribution; *origin* = the public token on query surfaces (`claude-code-session`, `chatgpt-export`); *provider* = the low-level parser/schema identity. Public filters use **origin**. Full table in [provider-origin-identity.md](provider-origin-identity.md). |
| **Ingest / acquisition** | Reading source files, detecting their provider by shape (not filename), parsing them, and writing normalized rows into the archive. |
| **Content hash** | A SHA-256 over the NFC-normalized session payload (title, timestamps, messages, attachments). It is the archive's idempotency key: re-ingesting identical content is a no-op. Editable metadata (tags, summaries) is excluded, so editing those never triggers re-import. |
| **Blob store** | Content-addressed storage for large binary content, keyed by SHA-256 and sharded into 256 subdirectories. Identical content deduplicates automatically. |
| **Insight (read model)** | A value computed once over the archive and stored for fast reads — session profiles, work events, phases, threads, tag and cost rollups. The CLI, daemon, MCP, and Python API all read the same materialized numbers. |
| **Facet** | A grouped count over a result set (e.g. sessions per origin, per repo, per tool). What `analyze --facets` returns. |
| **Projection** | A derived view of stored evidence — e.g. a Run, ContextSnapshot, or ObservedEvent assembled from underlying rows. Projections are views, not new sources of truth. |
| **Assertion** | An authored or transform-produced claim about a target (a session, message, or ref), carrying author, value, evidence, status, and visibility. The unifying substrate for user/agent overlays — tags, marks, corrections, notes, saved views. |
| **Evidence ref / ObjectRef** | A typed pointer to a thing in the archive (a session, message, block, run, …) that a claim or report can cite, so any digest is traceable back to its backing rows. |
| **Work packet / session digest** | A compact, shareable bundle summarizing what an agent session did — cost, repos touched, tools used, what to resume next — assembled from evidence, not free-typed. |
| **Topology edge** | A typed cross-session parent link (continuation, fork, sidechain, subagent). Stored durably even when the parent is ingested out of order. |
| **Logical session** | The resolved root of a session's parent chain. Continuations, forks, and subagent runs all roll up to one logical work session. |
| **Context preamble** | A typed bundle composed from the archive — seed session, recent lineage, project state, and resume guidance — that a coding agent can inject at SessionStart as prior memory. Backs `read --view context` and the MCP `compose_context_preamble` / `compile_context` tools. |
| **Daemon (`polylogued`)** | The background process that watches source directories, ingests live, rebuilds insights, and serves a local HTTP reader plus health checks and Prometheus `/metrics`. |
| **OTel projection** | An outbound, OpenTelemetry-shaped view of evidence rows: terminal query-unit rows mapped to spans, log records, and Polylogue refs for external observability tools. An export format, not a source of truth — message text and local paths are never copied in. |
| **Convergence** | The daemon's process of bringing the archive up to date with its sources: acquire raw rows, parse, materialize index rows, refresh insights, keep FTS in sync. |
| **Readiness** | Whether a tier is trustworthy *right now* — e.g. all raw rows materialized, FTS index current. Stale or unverified state makes a surface report not-ready rather than silently wrong. |
| **Archive debt** | Acquired-but-not-materialized state: a raw row with no matching parsed session, or a referenced blob with no file. Surfaced by `ops diagnostics workload`, not hidden. |
| **Demo archive** | A deterministic, private-data-free fixture archive (`polylogue demo seed/verify`) used for examples, screenshots, and CI without touching your real corpus. |
| **Retrieval lane** | Which search engine answers a query: `dialogue` (FTS5 lexical), vector (semantic), or `hybrid` (both fused via Reciprocal Rank Fusion). `--lexical` and `--semantic` force a lane. |
| **MCP bridge** | The Model Context Protocol server (`polylogue-mcp`) that lets an AI assistant search and read your archive from inside its own session. |
| **Verb** | A query-first action applied to a matched set: `read`, `select`, `analyze`, `mark`, `delete`, `continue`. Spelled `find QUERY then VERB`. |
| **Fresh-first schema** | There is no in-place schema upgrade chain. A version mismatch is rejected and the affected tier is rebuilt from source, rather than migrated step-by-step. |
