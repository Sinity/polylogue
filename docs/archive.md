# Archive

What Polylogue stores and how it reads it: stored inputs and their chains, why nothing is deleted, versioned rows, how stored formats evolve, parsers and their reading versions, independent goldens, field accounting, and format drift. Where inputs come from is in [capture.md](capture.md); the parsed model the parsers produce is in [sessions.md](sessions.md). The three stores and the durability order are in [architecture.md](architecture.md#stores); the segment container that holds the bytes is the common [hashed-blobs](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/hashed-blobs/README.md) crate.

## Stored inputs

- **`input → chain → extent → blob`.** An *input* is one source object: a file, a capture log, a container member, or a row-dump stream. Each read of it that adds bytes extends its *chain* (a revision that appends to the previous one continues the chain; a rewritten file starts a new one). *Extents* name byte ranges in *blobs*.
- **Blobs** are content-addressed and live in the shared segment container: one sync per write group, extents committed in the same transaction as the rows that cite them, a torn tail truncated on open. The blob store is the durable record of every acquired input, not a cache of bytes that still exist at the source, because tools prune and rewrite their own files. Ruling, 2026-10-11: "blob store is supposed to hold session logs … this is THE REASAON TO HAVE BLOBSTOER" (cc-queued:56c7f026-fd50-4480-b484-72f1299a534c@2026-10-11T12:58:04.056Z).
- **Observations** record each time an input was seen, with the machine and path: `seen`, `changed`, `renamed`, `gone`, `back`. **A file that disappears is an observation, never a deletion.**
- **Currency is local receipt order** (`commit_seq`), never wall-clock time.
- **Export containers are kept whole.** A ZIP or tar import stores the container exactly as given as one input and each member as its own input, addressed `container!/member`; identical member bytes are stored once with every observation kept.
- **Tool databases** are inputs of row dumps ([capture.md](capture.md#tool-databases)).
- **The archive is complete.** It holds every original input the user has, including the predecessor's hook spools, browser captures and their attachments, imported as declared older formats ([capture.md](capture.md#import)). Ruling, 2026-07-22: "WE ARE SUPPOSED TO KEEP ALL DATA." (cc-queued:0398d7a3-0d3d-42e4-acd7-bca417e0c481@2026-07-22T12:20:09.836Z).

## No deletion

Polylogue keeps every input and every reading of it; the deletion stance is stated in [architecture.md](architecture.md#deletion).
- **Re-parse is not deletion.** It replaces a chain's parsed rows with a new reading of the same stored bytes: `retract(chain)`, then apply, in one transaction.
- **Stored text is kept as written.** Polylogue stores and serves text like a text editor does, with no secret detection or masking. If a secret in history matters, the user (or an agent asked to) edits it out of the source file; both revisions of that file stay in the archive.
- A conversation that disappears upstream (absent from a complete newer export, deleted in Drive) is marked `source: gone` and keeps its rows.

## Versioned rows

- **Every version of a parsed row is stored**, and the shown versions form a total order. A growing file read again adds versions in receipt order.
- **Reading the same bytes again changes nothing**: a row version's hash is a function of the bytes and the reading version.
- A sparse message reading keeps the complete source version's hash, its original block count and the original positions of the blocks it carries. Missing blocks are unknown. Compatible observations of one version union their known blocks; a partial observation cannot replace known complete content, and conflicting facts under one source hash are refused.
- **Message IDs are stable across re-parse.** A source without native message IDs gets a fallback ID from the input and record position (`<stem>@<offset>`), stable for the same bytes.
- Which route's reading of a conversation is shown, field by field, is in [sessions.md](sessions.md#one-conversation-through-several-routes).

## Evolution

- **The segment format starts at version 1.** The initial format baseline collapses the earlier archive migrations. It applies to newly created stores; an older store is preserved and refused until the operator deliberately prepares its transition. Inputs, authored rows and purchased vectors are never reset by opening it. Subsequent changes use numbered migrations.
- **Stored formats evolve in place.** Schema changes are numbered in-place migrations, each with a test that an archive at the previous version keeps every row; a reading-version change re-parses exactly the chains it affects ([Parsing](#parsing)).
- **No code path deletes the index.** An index whose schema does not match is migrated, or refused with an error naming the deliberate move-aside rebuild. A full rebuild from the archive is a rare, deliberate act of the user ([operations.md](operations.md)).

## Parsing

### Parsers

- **One parser per origin, written from format knowledge.** Each parser carries the full knowledge of its origin's formats in every era (Claude Code todos, history and orchestration; Codex; Hermes; the Antigravity RPC; Gemini CLI sidecars; ChatGPT attachment resolution) and reads records as they are; there are no repair passes over parsed rows.
- **A malformed record is presumed to be Polylogue's bug.** It is counted per input with its reason, never skipped silently. Ruling, 2026-01-30: skipping malformed lines "presumes they actualy are malformed rather than, say, we handle the data wrongly" (cc-queued:e8b61976-2304-4d1c-8529-aba01037683d@2026-01-30T04:10:14.572Z).
- **Origin, format, route and harness are separate coordinates.** A record's model field is trusted over the harness name: a DeepSeek model running under the Claude Code harness is DeepSeek.
- **Markup** (`::kind(args)` in transcripts) is extracted at parse time into assertions authored by the message's author, with IDs derived from the message and position so a re-parse writes the same assertions ([assertions.md](assertions.md#markup)).

### Reading versions

- **Each parser module declares its reading version.** A chain is re-parsed when the reading version of the module that parsed it changes, and nothing else triggers a re-parse; there is no index-wide version that rebuilds everything.
- Each origin declares the ordered tuple of parser modules and versions it uses. The tuple commits with each chain's parsed rows and progress. Workers prepare replacement readings in private scratch, then the writer retracts and replaces one chain in one transaction. Other chains stay readable throughout.
- Source selection and chain retraction for messages, documents, events and links share the `VersionKind` declarations in the store.
- Derivations carry their own versions ([measures.md](measures.md#rules)).

### Independent goldens

- **Expected outputs are written from format knowledge, independently of the parser, and never re-blessed.** No environment variable or command rewrites expected files from parser output; a golden changes only by an edit a reviewer reads.
- Each format era of each origin has goldens, as cases of input and expected outcome checked against the format knowledge.

### Field accounting and outcomes

- **Every leaf is accounted for, at any depth.** Each origin declares the paths it reads and the paths it keeps verbatim. Every leaf of a record, including leaves below a declared path, is one of: read into a column, kept verbatim in the row's attributes, or counted as `unused` with its full path. Declaring a parent path never hides its children.
- Fields that are read, not merely kept, include: Claude `thinking_tokens`, `server_tool_use`, `iterations`, every `tool_result` content part including `tool_reference`, `toolUseResult` for multi-result records, subagent messages present only inside `agent_progress`, signature-only thinking, exit codes, `structuredPatch` edits; Codex `base_instructions.text`, `git.commit_hash`, `originator`, `model_provider`, workspace roots, `context_window`, `agent_role` and `agent_path`, `phase`, `encrypted_content` as bytes, every `turn_context`; every column of Hermes `sessions`; ChatGPT citations, `content_references`, `search_result_groups`, canvas, search queries, selected sources, image results, async tasks, `weight`, inline cite marks; claude.ai `token_budget`, non-text tool results, per-block citations and timestamps, thinking summaries.
- **Record outcomes per input:** `used`, `skipped` (with a declared reason) and `unreadable` (with the reason). They sum to the records read.
- **Parse state per input:** `parsed`, `partial`, `refused` (with the reason), `no_parser_yet`, `failed` (with the error). A failure is a parse-state fact of its input; there is no separate failure table.

## Format drift

Providers change their formats without notice. Drift is found by field accounting during ordinary parsing, not by inferring and committing schemas.

- **The walk.** The one full-depth walk per record that parsing already does accounts for every leaf. Origin declarations carry the expected type on each path a parser reads, explicit dynamic-key maps, and the known values of `type` fields.
- **The wire-shape table.** A derived table `wire_shape` records, per format, record kind, path and type, when it was first and last seen and how often.
- **The `drift` query unit** shows new paths, type changes and fields that stopped appearing, per origin and reading version, newest first; status shows the same counts ([capture.md](capture.md#capture-status)). A database table no parser knows appears there too.
- **One shipped notice**, `format-drift`, is passive: it reaches the session-start brief of agents working in Polylogue's own repository ([operations.md](operations.md#notices)).
- **Acting on a drift notice:** compare the example input with an older one, write a golden for the new format, declare the field, and raise the reading version.

### Schema tools

The schema tools describe the real shape of an origin's stored inputs without ingesting anything, for looking at a format by hand. They run in the daemon at idle priority, read only stored bytes (never provider files), and write nothing into the archive or the repository. In primary mode they read the materials through Sinex.

| Command | Does |
|---|---|
| `schema infer ORIGIN [--input ID…] [--since TIME]` | the structure of the selected stored inputs (all of the origin's by default): field paths, types, presence and counts, streamed, with coverage (inputs and bytes read, what was unreadable) |
| `schema diff BEFORE AFTER` | added and removed paths, changed type sets, changed requiredness |
| `schema unread ORIGIN` | paths stored inputs carry that the parser neither reads nor declares kept, at any depth |

- **No values.** Values never appear as examples, lists or enums. Object member names do, except identifier-shaped keys (UUIDs, long hexadecimal hashes, long numerals, dates and timestamps, paths), which are grouped under `*`, as are members of maps with many distinct sibling names. `--field NAME` keeps a name ungrouped.
- **Paths** start at the root and use `/` between components; `[]` is an array element and `*` an unnamed member; literal `~`, `/`, `*` and `[]` in names are escaped as `~0`, `~1`, `~2`, `~3`.
- **Counts per path:** `types` (values of each JSON type), `occurrences` (all values), `present` (parents with at least one value there), `parents` (eligible parents), `missing` (`parents - present`) and `required` (every eligible parent had it). These are observations of the sample, not a guarantee about future inputs.
- **A diff** reports each kind of change separately, ordered by path, and labels populations as incomparable when their origins, revisions or reads differ, while still showing what differs.
- Observation streams records and discards scalar bodies. Memory follows schema shape and nesting depth rather than record count or scalar length; deep JSON is read without a fixed depth cutoff. Malformed input, including an unfinished array or trailing content, is refused.
- Tool databases contribute their row dumps' column shapes and table structure; SQL text stays out. Browser captures are Polylogue's own format and have no inferred schema.

## Laws

- **disappeared-file-is-observation**: a disappeared file yields a `gone` observation and no row is lost; its return yields `back`.
- **crash-leaves-normal-state**: a crash at every point between append, sync and commit leaves a state that the ordinary open path accepts, with no repair step.
- **reparse-idempotent**: parsing the same bytes twice under one reading version yields identical rows.
- **reading-version-scope**: raising one module's reading version re-parses exactly the chains that module parsed.
- **index-never-deleted**: opening an index with a mismatched schema migrates it or refuses with an error naming the deliberate move-aside rebuild; it never deletes the index.
- **inputs-never-rewritten**: no code path deletes or rewrites stored inputs or blobs.
- **migration-keeps-rows**: an archive at the previous version keeps every row through each migration.
- **goldens-hand-written**: there is a hand-written golden for each format era of each origin, and the test harness has no mode that writes expected files.
- **unused-leaf-counted**: an unknown leaf below a declared path is counted as `unused` with its full path.
- **outcomes-sum**: malformed lines are counted `unreadable` with a reason, and record outcomes sum to the records read.
- **drift-reported**: a fixture with a new path, a changed type and a vanished field shows each in the `drift` unit for its origin and reading version.
- **schema-tools-no-values**: `schema infer` output over a fixture with distinctive values contains none of them.
