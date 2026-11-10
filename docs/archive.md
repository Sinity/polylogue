# Archive

What Polylogue stores and how it reads it: stored inputs and their chains, why nothing is deleted, versioned rows, how stored formats evolve, parsers and their reading versions, independent goldens, field accounting, and format drift. Where inputs come from is in [capture.md](capture.md); the parsed model the parsers produce is in [sessions.md](sessions.md). The three stores and the durability order are in [architecture.md](architecture.md#stores); the segment container that holds the bytes is the common [hashed-blobs](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/hashed-blobs/README.md) crate.

## Stored inputs

- **`input → chain → extent → blob`.** An *input* is one source object: a file, a capture log, a container member, or a row-dump stream. Each read of it that adds bytes extends its *chain* (a revision that appends to the previous one continues the chain; a rewritten file starts a new one). *Extents* name byte ranges in *blobs*.
- **Blobs** are content-addressed and live in the shared segment container: one sync per write group, extents committed in the same transaction as the rows that cite them, a torn tail truncated on open. The blob store is the durable record of every acquired input, not a cache of bytes that still exist at the source, because tools prune and rewrite their own files. (ruling 2026-10-11, cc-queued:56c7f026-fd50-4480-b484-72f1299a534c@2026-10-11T12:58:04.056Z).
- **An input is named as its origin declares.** A file a tool keeps at one place is a *location*: the first 32 hex digits of BLAKE3 of the machine name, a zero byte and the real path's bytes (its directory's symlinks resolved, so a file reached through a linked directory is one input), so a move or another machine makes another input. A file the tool moves or compresses itself is named by the *provider ID* written in it: a Codex rollout is `codex:rollout:<session id>` wherever it lives and however it is compressed, and is not stored until its first record names the session. An imported file is named by its *content*, the full BLAKE3 hex of its bytes, which is the same on every machine.
- **Observations** record each time an input was seen, with the machine and path, as one of `first`, `appended`, `rewritten`, `deleted`, `back` (seen again unchanged) and `imported`. **A file that disappears is an observation, never a deletion.** A file that comes back, or is replaced at its path by another inode, continues its chain while it still begins with every stored byte: the same bytes are observed `back`, more bytes are appended, and only changed stored bytes start a new chain. A device number alone does not change a file's identity, because mounting assigns it (btrfs subvolumes, network and FUSE mounts).
- **Currency is local receipt order** (`commit_seq`), never wall-clock time.
- **Export containers are kept whole.** A ZIP or tar import stores the container exactly as given as one input and each member as its own input, addressed `container!/member`; identical member bytes are stored once with every observation kept. An exact retry at the same receiving location, with the same machine, author and member bytes, keeps its import namespace and adds no observation; a change to any of those establishes a new namespace.
- **Tool databases** are inputs of row dumps ([capture.md](capture.md#tool-databases)).
- **The archive is complete.** It holds every original input the user has, including the predecessor's hook spools, browser captures and their attachments, imported as declared older formats ([capture.md](capture.md#import)). (ruling 2026-07-22, cc-queued:0398d7a3-0d3d-42e4-acd7-bca417e0c481@2026-07-22T12:20:09.836Z).

## No deletion

Polylogue keeps every input and every reading of it; the deletion stance is stated in [architecture.md](architecture.md#deletion).
- **Re-parse is not deletion.** It replaces a chain's parsed rows with a new reading of the same stored bytes: `retract(chain)`, then apply, in one transaction.
- **Stored text is kept as written.** Polylogue stores and serves text like a text editor does, with no secret detection or masking. If a secret in history matters, the user (or an agent asked to) edits it out of the source file; both revisions of that file stay in the archive.
- A conversation that disappears upstream (absent from a complete newer export, deleted in Drive) is marked `source: gone` and keeps its rows. A complete export is one that lists every conversation of its account (ChatGPT, claude.ai and Grok exports): one input, or every input of its import. It is newer than a conversation when it holds a message later than the conversation's last activity, so the mark depends on the exports' contents, never on the order they were imported.

## Versioned rows

- **Each chain's reading of a row is stored**: one version per input chain, under the current reading version. A re-parse replaces that chain's version; readings under older reading versions are not kept, because the bytes reproduce them. The shown versions form a total order, and a growing file read again adds versions in receipt order.
- **Reading the same bytes again changes nothing**: a row version's hash is a function of the bytes and the reading version. So a chain holding exactly the bytes of a newer chain of the same input (the same length and extents) has that chain's reading: a rebuild records the older chain's rows from the newer chain's reading instead of parsing its bytes again, and every chain a row appeared in stays recorded. Where an input's reading writes rows its chains share (a repository lookup, tool-database rows), or the newer chain stopped, the older chain is parsed. An older chain whose bytes are only a prefix of a newer chain's is parsed too: a later record can change an earlier row, so its reading is not the newer chain's.
- **A version leaves with its last reference.** Version bodies, event and link values and texts are shared between chains and rows; the transaction that removes the last source row, block or kept parser context referring to one deletes it, with its search entries, before it commits. Nothing is left for a later cleanup, and no search index needs a merge pass after a re-parse.
- **Every identity hash reads one JSON encoding**: the shared canonical JSON (a domain tag where the identity has one, object keys sorted, numbers as spelled), so no hash depends on map order or on the features serde_json was built with.
- **Times keep their source precision.** A parsed time keeps the precision, quality and bounds the source gave it ([capture.md](capture.md#source-clocks)); millisecond columns used by queries are an index over it and never replace it.
- **Message IDs are stable across re-parse and rewrite.** A message takes the source's own ID where the source gives one. A record without one (an AI Studio chunk, a Codex item, a Gemini page turn or activity entry, a Claude Code record without `uuid`, every Grok CLI and Aider message) is named by its content: its session, a hash of its role and content, and how many earlier messages of the session had the same role and content (`<session>~<role>-<hash>-<n>`). A position is never part of an ID, because a rewrite moves positions; an edited message gets a new ID. An unchanged message keeps its ID unless an earlier message of its session with the same role and content was removed or inserted: equal messages are numbered in order, so removing an earlier copy gives each later copy the number one lower (`A X A` rewritten as `X A` gives the remaining `A` the first `A`'s ID). Equal messages say the same thing, so a note on the removed copy then names the copy that remains, and the former ID of the last copy reads as missing; nothing rewrites the note. The kind of ID is recorded with each source row (`id_kind`).
- **A rewrite shows only what it still holds, for content-named messages.** Once an input's latest chain is read to its end, a message named by its content that the chain no longer holds stops being shown from that input (it was deleted, or edited into a message with another ID). A natively named message the rewrite dropped stays shown from the older chain that held it.
- **A stored row never reads as a default.** A stored name the model's enum does not have, an ID that does not parse, unparseable attributes or a negative count is a read error naming the row or column; it is never read as a user message, empty attributes or a skipped row.
- Which route's reading of a conversation is shown, field by field, is in [sessions.md](sessions.md#one-conversation-through-several-routes).

## Evolution

- **One baseline.** The segment format and the index schema reset once to version 1, with the earlier migrations collapsed; the live archive's inputs, authored state and vectors are kept across that reset. An archive written before the baseline is a declared older input format, imported once into a fresh archive with `polylogue import --from-archive PATH` ([capture.md](capture.md#an-archive-from-before-the-baseline)), never migrated in place.
- **One current format.** Every database is created and opened only at version 1. A schema change edits the version-1 schema and regenerates test data; no migration step and no reader for an earlier version is kept. A reading-version change re-parses exactly the chains it affects ([Parsing](#parsing)).
- **No code path deletes the index.** An index whose schema does not match is refused with an error that says how to rebuild it deliberately (moving `index.db` aside). A full rebuild from the archive is a rare, deliberate act of the user ([operations.md](operations.md)).

## Parsing

### Parsers

- **One parser per origin, written from format knowledge.** Each parser carries the full knowledge of its origin's formats in every era (Claude Code todos, history and orchestration; Codex; Hermes; the Antigravity RPC; Gemini CLI sidecars; ChatGPT attachment resolution) and reads records as they are; there are no repair passes over parsed rows.
- **A malformed record is presumed to be Polylogue's bug.** It is counted per input with its reason, never skipped silently. (ruling 2026-01-30, cc-queued:e8b61976-2304-4d1c-8529-aba01037683d@2026-01-30T04:10:14.572Z).
- **Origin, format, route and harness are separate coordinates.** A record's model field is trusted over the harness name: a DeepSeek model running under the Claude Code harness is DeepSeek.
- **Markup** (`::kind(args)` in transcripts) is extracted at parse time into assertions authored by the message's author, with IDs derived from the message and position so a re-parse writes the same assertions ([assertions.md](assertions.md#markup)).

### Reading versions

- **Each parser module declares its reading version.** A chain is re-parsed when the reading version of the module that parsed it changes, and nothing else triggers a re-parse; there is no index-wide version that rebuilds everything.
- A re-parse prepares the new reading aside, then retracts and replaces one chain in one transaction; other chains stay readable throughout.
- An incomplete suffix waits at its recorded byte length. A finished input (an import, an export member) has none: the end of its bytes ends its last record, newline or not.
- A stored import whose recorded path no parser reads (or that recorded no path) is refused with that reason in its parse state, and the pass goes on with the other inputs. A refusal or failure also records the executable digest and any parser settings that affected the attempt. Current native parsers expose no configurable parser settings, so their attempts have no settings digest. It stays stopped across restarts while that identity, its input chain, stored length and reading version match. Changed bytes, a new reading version or changed parsing dependencies make it eligible again. A newly eligible stopped attempt replaces its reading from the retained bytes, including when its failed cursor has no parser continuation. An older failure without an attempt identity is unrecorded: its next real attempt records the identity. A first-byte source acquisition failure without a stored chain remains retryable.
- Derivations carry their own versions ([measures.md](measures.md#rules)).

### Independent goldens

- **Expected outputs are written from format knowledge, independently of the parser, and never re-blessed.** No environment variable or command rewrites expected files from parser output; a golden changes only by an edit a reviewer reads.
- Each format era of each origin has goldens, as cases of input and expected outcome checked against the format knowledge.

### Field accounting and outcomes

- **Every leaf is accounted for, at any depth.** Each origin declares the paths it reads and the paths it keeps verbatim. Every leaf of a record, including leaves below a declared path, is one of: read into a column, kept verbatim in the row's attributes, or counted as `unused` with its full path. Declaring a parent path never hides its children.
- Fields that are read, not merely kept, include: Claude `thinking_tokens`, `server_tool_use`, `iterations`, every `tool_result` content part including `tool_reference`, `toolUseResult` for multi-result records, subagent messages present only inside `agent_progress`, signature-only thinking, exit codes, `structuredPatch` edits; Codex `base_instructions.text`, `git.commit_hash`, `originator`, `model_provider`, workspace roots, `context_window`, `agent_role` and `agent_path`, `phase`, `encrypted_content` as bytes, every `turn_context`; every column of Hermes `sessions`; ChatGPT citations, `content_references`, `search_result_groups`, canvas, search queries, selected sources, image results, async tasks, `weight`, inline cite marks; claude.ai `token_budget`, non-text tool results, per-block citations and timestamps, thinking summaries.
- **Record outcomes per input:** `used`, `skipped` (with a declared reason) and `unreadable` (with the reason). They sum to the records read.
- **Parse state per input chain:** parsed, as `parsing` (stored bytes remain), `waiting` (with the stored length it waits at, while a record is still being written) or `done`; `partial`; `refused` (with the reason); `no_parser_yet`; `failed` (with the error). A failure is a parse-state fact of its input; there is no separate failure table. A refused or failed chain records the stored length and stopped attempt it was read at; the dependency rule above governs its next attempt. Successful progress clears the stopped attempt. The executable digest is computed once at the process boundary when a command starts an Engine; offline completion and read-only clients do not hash the executable.

Canonical query records retain their source insertion position for incremental
readers. Older records whose position was never recorded keep an unknown
position; startup snapshots include them. This metadata does not change the
query's sealed bytes, semantic hash or frozen dependencies.

### Keyed continuation

Claude response keys and record UUID aliases remain addressable for the whole input chain. Small in-memory windows cache them; Store retains compact mappings and reuses the existing message versions. The mappings and exact parser checkpoint commit in the same transaction. A replacement keeps one private context across its staged chunks, emitting only each chunk's changed mappings. Its admitted lookup cutoff stays fixed while its private cursor advances. A rejected preparation reports its original error in the pass summary and leaves the admitted checkpoint, context and rows unchanged. If its failed cursor, chain, reading version and stored length exactly match an already admitted stop, that stop keeps its original error and binds the retry's executable identity through the same progress path. Other rejected replacements remain eligible for a later attempt.

A new reading version reconstructs missing historical context from retained bytes through ordinary re-parse. Once recorded, an ordinary restart resumes directly at that checkpoint. Lost derived context can be rebuilt from the admitted prefix. An absent context record means unrecorded context, not a complete empty map.

## Format drift

Providers change their formats without notice. Drift is found by field accounting during ordinary parsing, not by inferring and committing schemas.

- **The walk.** The one full-depth walk per record that parsing already does accounts for every leaf. Origin declarations carry the expected type on each path a parser reads, explicit dynamic-key maps, and the known values of `type` fields.
- **The wire-shape table.** A derived table `wire_shape` records, per format, record kind, path and type, when it was first and last seen and how often. Each reading's rows cite the source receipts (where and when a shape was first and last seen) by number, and `wire_receipt` holds each receipt once per reading.
- **The `drift` query unit** shows new paths, type changes and fields that stopped appearing, per origin and reading version, newest first; status shows the same counts ([capture.md](capture.md#capture-status)). A database table no parser knows appears there too.
- **One shipped notice**, `format-drift`, is passive: it reaches the session-start brief of agents working in Polylogue's own repository ([operations.md](operations.md#notices)).
- **Acting on a drift notice:** compare the example input with an older one, write a golden for the new format, declare the field, and raise the reading version.

### Schema tools

The schema tools describe the real shape of an origin's stored inputs without ingesting anything, for looking at a format by hand. They run in the daemon at idle priority, read only stored bytes (never provider files), and write nothing into the archive or the repository. In primary mode they read the materials through Sinex.

| Command | Does |
|---|---|
| `schema infer [ORIGIN \| --all] [--input ID…] [--since TIME] [--revision N]` | the structure of the selected stored inputs (all of the origin's by default, every origin with `--all`): field paths, types, presence and counts, streamed, with coverage (inputs and bytes read, what was unreadable) |
| `schema diff BEFORE AFTER` | added and removed paths, changed type sets, changed requiredness |
| `schema unread [ORIGIN \| --all]` | paths stored inputs carry that the parser neither reads nor declares kept, at any depth |

- **No values.** Values never appear as examples, lists or enums. Object member names do, except identifier-shaped keys (UUIDs, long hexadecimal hashes, long numerals, dates and timestamps, paths), which are grouped under `*`, as are members of maps with many distinct sibling names. `--field NAME` keeps a name ungrouped.
- **Paths** start at the root and use `/` between components; `[]` is an array element and `*` an unnamed member; literal `~`, `/`, `*` and `[]` in names are escaped as `~0`, `~1`, `~2`, `~3`.
- **Counts per path:** `types` (values of each JSON type), `occurrences` (all values), `present` (parents with at least one value there), `parents` (eligible parents), `missing` (`parents - present`) and `required` (every eligible parent had it). These are observations of the sample, not a guarantee about future inputs.
- **Selection.** Inference reads each input's latest revision, or the exact `--revision N`, reporting an absent revision. `--input` makes the report a sample. `--since` selects by the time an input was received; inputs whose time is unknown are left out and counted. The answer groups schemas by origin and reports the inputs selected, the bytes stored and the bytes read.
- **Background work.** One inference runs at a time and gives way to ordinary daemon requests. There is no per-input size cutoff and no default deadline; cancellation and an explicit `--timeout` apply.
- **Output.** CLI, MCP and HTTP send the same daemon requests. `--output FILE` writes in the CLI process, and `schema diff` reads those files locally.
- **A diff** reports each kind of change separately, ordered by path, and labels populations as incomparable when their origins, revisions or reads differ, while still showing what differs.
- Observation streams records and discards scalar bodies. Memory follows schema shape and nesting depth rather than record count or scalar length; deep JSON is read without a fixed depth cutoff. Malformed input, including an unfinished array or trailing content, is refused.
- Tool databases contribute their row dumps' column shapes and table structure, reported per table: the declared column types, nullability, primary key, hidden or generated columns and `WITHOUT ROWID` ([capture.md](capture.md#tool-databases)) separately from the JSON types observed in rows, with empty tables and never-filled columns listed. SQL text and default expressions stay out. Browser captures are Polylogue's own format and have no inferred schema.
