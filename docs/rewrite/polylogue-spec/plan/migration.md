# Moving from the Python version

The Python Polylogue is stopped (the daemon is parked; its state was moved
aside in the September reset). The Rust version starts with an empty archive
and builds it from the provider files, as the reset intended. Nothing is
migrated from the old databases except purchased embedding vectors; the
operator confirmed there are no old notes, settings or excisions worth
keeping. But every raw input the old archive kept is fed in as an ordinary
input, because some exist nowhere else (`evidence/old-archive-only.md`):
providers delete and rewrite their files, and keeping every input is what
the blob store is for.

These inventories describe the earlier inspection, not verified current state. Refresh source counts,
provider presence, parked/running services and conservation evidence at execution. Initial bootstrap
runs use isolated state; spool restoration, private-corpus imports and host cutover are separate scoped
operations. Never infer reset authority from the fresh-workspace decision.

## What carries over

| Thing | How |
|---|---|
| Provider files on disk | Read directly by the new collector; nothing to do. |
| Account exports (`/realm/account/chatgpt`, `/realm/account/claude`, Grok) | `polylogue import` each file once. |
| Hook spool and browser captures (live) | The new collector reads the same spool directories. The hook spool layout stays as the Python producer writes it; the Python-era capture envelope is read as an older version of Polylogue's own format (`spec/sources/README.md`, browser captures). |
| Captures and hook spools set aside by the reset | Collect staged copies through the ordinary spool reader, following “Restoring spools” below. Isolated staging and live restoration are separate operations; neither is an archive reset. |
| AI Studio | Not an import: still on Google Drive. 228 of the 239 old chats match current Drive chats; the Drive source collects them once it runs (slice 5). The 11 not on Drive are in the old local Drive cache and are copied into the canonical cache before then. |
| Codex rollouts that Codex rewrote | Codex rewrote its rollout files in place (only 6 of 400 sampled pairs were appends; records gained `ordinal`, older event shapes were converted, 23 files lost records, 39 had timestamps collapsed). The old bytes preserve an earlier representation; record-count changes alone do not prove missing turns. They are added once as earlier revisions of the same inputs (`spec/storage/blobs.md`, Inputs, revisions and extents), from the 2026-08-16 snapshot and the aside store, through a small one-off script in `plan/scripts/`, written in slice 2 with the Codex provider. |
| Other old raw inputs | Inventory every retained revision and referenced blob. A surviving provider path does not establish equal bytes or coverage. Import archived revisions not already retained by digest, including attachment and hook payload references; report missing, malformed and unsupported inputs explicitly. |
| Deleting the old copies | Only after every durable reference resolves to digest-verified retained bytes, revision coverage and parser differences are accounted for, and an independent backup restore reproduces those bytes and rebuilds the supported views. Counts and source-path existence are insufficient. Incomplete stores must be joined to the stores supplying their missing blobs before retirement. Keep unresolved populations visible; delete only the exact approved stores with the operator's go-ahead. |
| Purchased embedding vectors (`embeddings.db` from the September backup) | One-time import. The old embedded unit is the message: its text blocks in order, joined with `\n\n` ("one-vector-per-message-v1"). The importer computes that joined string from the old index and hashes it with BLAKE3. For a message with one text block this equals the block's text hash, and the vector is reused under it; nothing is embedded again. For a multi-block message (measure the share before slice 4) the vector is kept under the joined string's hash only if the new embedding unit for such a message is the same joined string; otherwise those vectors are reported as unmatched, with their count and purchase cost, and embedded again when embeddings are enabled; the operator values the old vectors at a few dollars, so re-embedding is acceptable. Decided by the spec authors (2026-10-11); the operator may overrule. Nothing is dropped silently. This is an operation run once, not code kept in the product. |
| Price table | Re-entered as data (`prices.toml`), checked against current published prices. |
| Python test fixtures (125 files under `tests/fixtures` and `tests/data`) | Synthetic or redacted (the Hermes ATIF/ATOF files are marked `real_redacted`), so after a privacy read they seed `examples/sources/<format>/` as example inputs, each with expected output recorded by the new parser and reviewed against `plan/testing/real-data-cases.md` (`evidence/deep-tests.md`). Code and test structure do not carry over. |

Sources: audit P30.

## Restoring spools

Refresh the historical inventory in `evidence/old-archive-detail.md` against the
actual source and destination. For an isolated real-data build, stage retained
browser captures, `hooks/carriers/` and the older `hooks/pending/` queue under the
candidate's configured sources. Live restoration is a separately scoped operation
using the resolved current spool paths; do not assume the historical default path
is still authoritative.

Record each source's digest, destination and disposition. For colliding names,
reuse identical bytes; retain differing bytes under distinct, format-valid names
with their original identity and capture time. A no-clobber skip is not proof of
restoration. Verify every staged destination against its source before collection,
and account for unavailable or unsupported inputs explicitly. Slice 1 stores the
older pending layout as `no_parser_yet`; slice 2 parses it
(`spec/sources/README.md`). Convergence collects both layouts as ordinary spools.
This procedure preserves the originals and does not authorize their deletion.

## Comparing parsers before switching

Before the Python version is retired for good, `xtask compare-old` runs both
parsers on the same stored files and compares results (the exact procedure,
mapping and comparison rules are in `plan/compare-old.md`):

1. A small Python script (outside the product, kept in the new repository
   at `xtask/compare-old/dump_old.py`; `plan/compare-old.md`) runs the old parsers over the operator's stored inputs and writes
   their normalized output per input to a private directory. Run once.
2. The new parsers produce the same normalized form (the export format in
   `spec/export-format.md`).
3. The comparison reports, per provider: sessions and messages found by each,
   messages only one side found, differing tool outcomes, differing token
   counts. Expected differences (new IDs for messages without native IDs,
   `written_by` replacing the old `material_origin`, fields the old parser
   dropped) are listed once and filtered.

Every remaining difference is either a bug in the new parser or an
improvement, and is decided case by case. The recorded output turns the old
system's parsing knowledge into data the new one is checked against.

## Consumers

- **Lynchpin**: reads Polylogue through a small Python client that runs
  `polylogue … --json` (or speaks the socket protocol directly) and opens no
  SQLite file; like every client it needs the running daemon
  (`decisions/process.md`). The ~10 calls
  it uses are reimplemented in the client; its direct SQLite reads of the old
  schema are replaced by client calls.
- **Agents**: the MCP server and skill are installed by the new binary
  (`polylogue install-agent claude-code`, `polylogue install-agent codex`); the old Python MCP entry is removed from agent
  configs.
- **Sinnix**: the NixOS and Home Manager modules switch the package and
  service to the new binary; hook commands point at `polylogue hook`.

## Switching over

1. Slices 1–4 complete; the comparison shows no unexplained differences.
2. Build the full archive from all sources (`xtask full-run` timings recorded).
3. Point Sinnix at the new package; remove the Python package, its service and
   its MCP entries.
4. Keep the old state directories as they are (they are already moved aside)
   until the operator decides to delete them; deleting them asks the operator
   at that moment, after the complete durable-reference, revision, parser and
   independent-restore criteria in “What carries over” pass. Checking only restored
   spool counts or re-collected Drive chats is insufficient. Unresolved populations
   remain preserved; source replacement does not authorize state retirement.

## The old repository

The source tree is replaced in the same repository through a bootstrap PR (`decisions/repository.md`).
Preserved refs/bundles retain the Python implementation; installed packages remain pinned to it until
separate cutover. Source replacement neither moves archive state nor authorizes its deletion. The Python code's knowledge
survives through the spec, the example files and the recorded parser
output.

- The Python implementation of material protocol v1 and its outbox are retired. The protocol's structure (bytes first, then events pointing into them, then a count-and-digest check) is the starting point for the Rust transport, redesigned from the export format (`decisions/sinex.md`, Transport).

The import reads only raw inputs, capture and spool files and attachments; old user and audit records are not touched. `polylogue-2yke6` stays the place for any later salvage of old notes and references.
