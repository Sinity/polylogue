# Glossary

This page defines the few project terms that earn a name and translates the
coined words older code still uses. For the full system shape see
[architecture.md](architecture.md); for the provider/origin/source split see
[provider-origin-identity.md](provider-origin-identity.md).

## Entry layers

Read only as deep as you need:

- **One sentence** — Polylogue ingests the AI-session files ChatGPT, Claude,
  Codex, Gemini, and coding agents leave on disk into one local, searchable
  SQLite archive, and gives you search, insights, and an MCP/HTTP/CLI cockpit
  over them.
- **Overview** — the [README](../README.md) top section and the
  [architecture diagram](architecture.md).
- **Deep** — [architecture.md](architecture.md),
  [internals.md](internals.md), and the [search grammar](search.md).

## Terms

| Term | Plain-language meaning |
|------|------------------------|
| **Archive** | The on-disk SQLite file set under your XDG data directory that holds every ingested session. Not a single file — see *archive tier*. |
| **Archive tier** | One of the six SQLite databases the archive is split into by durability class — see the six-tier table in [schema.md](schema.md#the-six-tiers) (source, index, embeddings, user, audit, ops). Tiers are backed up and rebuilt independently. |
| **Source / origin / provider** | Three scopes of "where a session came from". *Source* = the material runtime root and lab attribution; *origin* = the public token on query surfaces (`claude-code-session`, `chatgpt-export`); *provider* = the low-level parser/schema identity. Public filters use **origin**. Full table in [provider-origin-identity.md](provider-origin-identity.md). |
| **Ingest / acquisition** | Reading source files, detecting their provider by shape (not filename), parsing them, and writing normalized rows into the archive. |
| **Content hash** | A SHA-256 over the canonical session payload (title, timestamps, messages, attachments), with only prose fields NFC-folded. It is the archive's idempotency key: re-ingesting identical content is a no-op. Editable metadata (tags, summaries) is excluded, so editing those never triggers re-import. |
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
| **Demo archive** | A deterministic, private-data-free fixture archive (`polylogue demo tour`, with lower-level `seed`/`verify` checks) used for examples, screenshots, and CI without touching your real corpus. |
| **Retrieval lane** | Which search engine answers a query: `dialogue` (FTS5 lexical), vector (semantic), or `hybrid` (both fused via Reciprocal Rank Fusion). `--lexical` and `--semantic` force a lane. |
| **MCP bridge** | The Model Context Protocol server (`polylogue-mcp`) that lets an AI assistant search and read your archive from inside its own session. |
| **Verb** | A query-first action applied to a matched set: `read`, `select`, `analyze`, `mark`, `delete`, `continue`. Spelled `find QUERY then VERB`. |
| **Fresh start** | The current archive format starts every tier at `PRAGMA user_version=1`; state written by earlier Polylogue versions is set aside, never migrated. Later changes to the durable tiers (`source`, `user`, `audit`) are additive numbered migrations; the derived tiers (`index`, `ops`) are rebuilt instead. |
| **Materialize** | Compute derived rows from stored data and write them to `index.db`. |
| **Lowering** | Converting parser output into table rows. |
| **Hydrate** | Load a full domain object (a session with its messages) from rows. |
| **Derivation** | One derived product's inspect, compute, and publish cycle in the daemon's derivation kernel. |
| **Index generation** | A complete `index.db` built beside the active one and swapped in when finished. Use "generation" for nothing else. |
| **Membership** | Which sessions one stored raw file contains. |
| **Recipe** | For embeddings, every input that changes a vector (model, dimensions, text preparation). |

## Coined words in current code

Older code and docs use invented or overloaded words. New text uses the plain
term on the right; read existing code with this table.

| Word in code | Usually means |
|------|------|
| retained, raw | source bytes stored in `source.db` |
| receipt | result record of a finished step, or just a row |
| lease | a lock (`ActiveWriterLease`), a thread's write permission (`WriteLease`), or a claim on a queue item with expiry |
| custody | the archive write lock (`.archive-write-custody.lock`), an object that must be closed, or (`custody_digest`) a digest of inputs |
| settle, settlement | finish and record the outcome; also close handles, wait for completion, or commit |
| admission, admit | accept or reject an input; schedule work on the compute queue; allow a write |
| seal | a closed scratch file's identity, a snapshot digest, or the check that `user.db`/`audit.db` references survive an index write (`reference_seal.py`) |
| generation (outside the index sense) | an input snapshot (`source_generations`), or a change counter |
| frontier | byte offset accepted so far in an append-only file; "raw frontier" = stored raws not yet indexed |
| census | full inventory, e.g. which sessions each raw contains |
| carrier | the append-only JSONL spool file a hook writes |
| witness | an evidence item or example |
| envelope | a response wrapper |
| disposition | outcome or status |
| demand | a pending request to recompute |
| obligation | a pending outbox row |
| carry | parse state reused across phases |
| durable change train, rider, slot | migration plan, migration step, schema version |
| qualify | measure or test under stated conditions |
| conservation | a check that nothing was lost |
| continuity | the audit hash chain |
| convergence debt | the retry list for daemon stages, in `ops.db` |
| authority, owner | the component responsible for something, or its source of truth |
