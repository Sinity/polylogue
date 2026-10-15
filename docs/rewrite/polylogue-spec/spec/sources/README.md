# Sources: general rules, merging and spools

Where Polylogue's inputs come from, what they look like, and exactly how each
provider's records become sessions, messages, blocks, links and events
(`model.md`). This is reference material for whoever writes a parser. The
real-data measurements behind it are in `evidence/data-shapes.md`; scrubbed
raw excerpts are in `examples/raw/<provider>/`.

Provider files: [Claude Code](claude-code.md), [Codex](codex.md), [ChatGPT, claude.ai and Claude Design](chat-exports.md), [AI Studio and Gemini CLI](google.md), [Hermes, Antigravity, Grok and planned sources](others.md).

## How sources work

**Fixed locations, no configuration.** Each provider writes its files to a
standard place (`~/.claude/projects/`, `~/.codex/sessions/`, …). Polylogue
reads those places and nothing else. There is no setting to point it
elsewhere; someone who keeps files somewhere unusual symlinks them into the
standard place. Polylogue does not search other folders (worktrees, backups,
Downloads) for things that look like sessions.

**Exports are imported by command.** Account exports (ChatGPT, claude.ai,
Grok) live wherever the user downloaded them. `polylogue import PATH` hands
one to the daemon, which stores it like any other input. Importing the same
export again, or an overlapping newer export, adds only what is new.

**Strict recognition.** A location tells Polylogue what to expect: a `.jsonl`
under `~/.claude/projects/` is a Claude Code transcript, a file under
`~/.codex/sessions/` is a Codex rollout. If the bytes are not that format, the
input is refused with a reason (`parse_state.status = refused`) and shown by
`polylogue status`. Polylogue never re-detects a refused file as some other
provider. Files the provider keeps next to its sessions (indexes, metadata,
attached files) are recognized by name and read as what they are, never parsed as
sessions. A file under a provider location that matches no known name is
counted as unrecognized, so new attached files are noticed.

**Importing a directory.** `polylogue import DIR` imports every file under
it. A file whose path inside `DIR` starts with an origin name followed by
that provider's location layout (`claude-code/projects/-home-op-morrow/….jsonl`,
`codex/sessions/2026/…/rollout-….jsonl`, `hooks/carriers/…`) is recognized
exactly as if it had been found at that provider location, by the same
strict rules. Old copies of a file that is still at its location (the old
archive's pre-rewrite Codex rollouts, `plan/migration.md`) join that
location's input as earlier revisions. Any other
file is recognized by its structure, as below. `--format NAME` names the
format when structure alone cannot tell (`surfaces/cli.md`).

**Machine.** Every stored location records the machine it was seen on
(`observation.machine`), with the machine's time zone. A machine is
identified by the content of `/etc/machine-id`; the host name is kept as a
label, so renaming a host splits nothing and two hosts with a default name
never collide. Moving an archive to a new machine that keeps the same
provider paths is done with `polylogue setting machine <old id>`, which
makes the new machine continue the old observations. The archive directory
is on a local filesystem and is never synced live between machines;
machines combine archives through Sinex, or export and import. How a
session's `machine` follows is one rule in `model.md` (Session, `machine`).

For `polylogue import`, where there is no location to go by, the format is
recognized from the file's structure (a ChatGPT export ZIP contains
`conversations*.json` with `mapping` trees; a claude.ai export contains
`conversations.json` with `chat_messages`; a lone Claude Code transcript is
a JSONL file whose first decodable record has `sessionId`, `uuid` and a
`type` from Claude Code's record types, `sources/claude-code.md`; a Codex
rollout one whose first record is `session_meta`), and the result is
reported back: "recognized as ChatGPT export (2026 layout), 2,836
conversations". A lone transcript imported from outside the provider
locations is stored as its own input with an `imported` observation; its
messages merge with the live file's by ID (rule 2), so importing a copy of
a live session adds nothing but that observation (law 1, `duplicate`). A
file that matches no signature is refused with the formats tried, and
`--format NAME` overrides recognition.
Recognition reads only a ZIP's member list and the first record of the file
that decides the format, never the whole file, so a 16 GB export is
recognized as fast as a small one and size never changes how a supported
shape is recognized (the Python detector held about twice the file in
memory and refused inputs above 64–256 MB, `polylogue-1o8tb`,
`polylogue-9upnr`).

**Polylogue's own formats.** The hook spool, the browser capture envelope,
the export format and the repo-lookup record are Polylogue's. Polylogue
writes only their current version, and reads every version it ever wrote:
stored inputs in older versions are kept for good, and some exist nowhere
else (the Python-era browser captures, `plan/migration.md`). Reading an old
version is ordinary parsing of a stored format, like reading an old provider
format, not a compatibility path. When a layout changes, tools that read the
live spool (the `enrichment-pass` skill) are updated in the same change;
there is no promise to freeze a layout.

**Unknown content is counted, never dropped.** Every record type and field a
parser does not handle is counted per format (`unrecognized` table, see
`storage/index-db.md`), with first/last seen and an example input. Providers change
their formats every few weeks; this is how the change is noticed.

**Parsers are pure.** A parser receives bytes and returns sessions,
messages, blocks, links, events and unrecognized counts. It does not read
the network, the database, or any file it was not given.

**Companion files are given to the parser with their main file.** A
provider declares which files belong to a main file (Claude Code: the
`tool-results/*.txt` files and the `agent-<id>.meta.json` next to a session
or subagent file). The engine calls one entry per input,
`parse(main, companions, carried)`, which the shared framework implements
by framing the main input's records and calling the provider's per-record
interface (below, the framework owns iteration), with the companions
available to it; it returns one result; when a companion changes, its main file is parsed again
(`convergence.md`, "Sessions built from several inputs"). Everything else
that joins inputs (a subagent's parent in another file, an export and a
capture of the same conversation, hook events, a Codex title in
`state_5.sqlite`) is a merge by ID done by the writer, not by the parser.

**Archives of provider files are inputs too.** A tar or ZIP that holds
provider files (for example `~/.codex/sessions/2026/m8.tar.gz`, 1,817 Codex
rollouts saved before Codex rewrote them in place, and the only copy of what
the rewrite removed):

- *is stored whole*, imported by command as an input named by its hash, and
  read member by member;
- *each member stands for a file*: the archive's folder joined with the
  member's path (`m8.tar.gz`'s `08/02/rollout-….jsonl` stands for
  `~/.codex/sessions/2026/08/02/rollout-….jsonl`), parsed by the provider
  whose location that path is. It becomes an **older chain of the input at
  that location**, observed at `zip:<archive>/<member>` with that location
  recorded as what it stands for (`archive-db.md`, observation), its
  uncompressed bytes stored once as a blob of their own, since a chain is
  made of blob ranges (`storage/blobs.md`), so it merges exactly as a
  rewrite of that file would (rule 5, the older
  revision's times included). When no file has been seen at that location,
  the member's chain starts the input and the file found there later joins
  it as its current chain; either arrival order gives the same input,
  chains and index. A member whose bytes some chain already holds adds
  nothing, so importing an archive again stores only its observation.
  (Checked on 151 real `m8.tar.gz` members: the rewrite route, and the
  archive imported after today's files and before them, give one index;
  `evidence/codex-rewrite-deltas.md`.)
- *is never opened by collection on its own* (strict locations): `polylogue
  status` lists archives found at a provider location as unrecognized
  files, so the operator can import them.

**One bad record never loses a file.** Records are parsed one at a time. A
line that is not valid JSON, or a record whose shape does not match what
its type should have, is skipped and counted (`unrecognized`, or a
`parse_state` detail), and parsing continues with the next record. A lone
UTF-16 surrogate in a JSON `\u` escape (4 lines in 19.9 GB of Claude Code
data) does not make a record bad: the stored bytes stay exactly as they are,
the parser puts U+FFFD in its place in parsed text and counts it, and the
record is kept. Bytes that are not valid UTF-8 are treated the same way: each invalid
sequence becomes U+FFFD in parsed text, counted per input in `unrecognized`
(kind `invalid_utf8`, with the byte offset of the first occurrence), and
the stored bytes are unchanged. A format's own encoding is decoded before
the UTF-8 check, never after: JSON `\u` escapes, protobuf strings, and
zsh's metafied history bytes (0x83 followed by the byte XOR 0x20; Sinex found
45 such commands in the operator's history, every one of which a plain
UTF-8 decode rejected). Only what is still invalid after the format's
decoding is counted as `invalid_utf8`.

**A time that does not parse is unknown, and counted.** A record whose
time field holds text no rule of its format can read keeps the record and
its raw text (in `attrs`), gets an unknown time, and is counted per input as
`unrecognized` kind `time_unparsed` with the first offending form, so a new
provider time format shows in `status` instead of silently emptying times.
An ID never depends on a parsed time (`model.md`, identity), so a new
format learned later fills the time without moving the ID; a Lynchpin
reader that returned nothing for an unknown time format and built IDs from
it collapsed thousands of rows into a few.

**Order comes from the data's own structure, not from timestamps.** Message
order comes from the provider's tree (parent links) and from position in
the file. Provider clocks go backwards often enough (thousands of Claude
Code files) that sorting by time would reorder conversations. A session's
first and last times are the minimum and maximum over its own messages
and events (`model.md`, Session), not the first and last record's.

**Parents skip records that are not messages.** When a provider's tree
links records that become events (Claude Code `attachment`, `progress`,
`system`; ChatGPT message-less nodes), a message's `parent` is the nearest
ancestor record that becomes a message. Branches are found on the record
tree before a provider's split records are folded into one message: two
message records are siblings only when they share that nearest message
ancestor record, and event records are never branch points. A folded
message is on the active path when any of its records is, and its `parent`
is the message of its first record's parent. Results of the calls of one
response all have that response as their parent and are ordered by file
position; they are never branches of each other.

**JSONL lines end at `\n` only.** A trailing `\r` is stripped. Unicode
line and paragraph separators (U+2028, U+2029) inside JSON strings never
split a record; only a reader that splits on bytes keeps this true. For a
JSONL format that permits blank separator lines, a retained append made only
of those separators may frame no records and emit no normalized sessions.
Prove that classification over the entire acquired window, with its known
contiguous predecessor; a prefix sample or zero emitted rows is not that
proof. Preserve its bytes and coordinates, and commit verified progress
without inventing a session (`convergence.md`, Parse). A nonblank unfinished
record is not a separator-only append and retains the partial-tail outcome.

**A read that fails is retried, never taken as absence.** A file or folder
that cannot be read (permission denied, out of file descriptors, a locked
database) is tried again in the next round; a read failure is never recorded
as permanent. An unreadable location shows in `status` as unavailable, with
the error, and is never treated as missing: its sessions do not become
`source:gone`.

**Recognition does not depend on size.** Recognizing a file reads its
location and a bounded prefix (the first record); a file of any size is
recognized the same way and parsed with bounded memory.

**Appended files are parsed incrementally.** Claude Code and Codex append to
their JSONL files while a session runs. The daemon keeps a cursor (bytes
read, last revision) together with a small parse state for the file; when
the file grows, only the new complete lines are parsed. A final line without
a newline is not parsed until it is complete. The parse state holds
identifiers and positions, plus, for a format whose later records change
messages already written, those unsettled messages in full (Codex; never
more than the messages a later record can still change): what each provider
keeps is listed in its section ("What an incremental parse carries
forward" in `claude-code.md` and `codex.md`). An append and a full re-parse of the whole file give the same
result; a law in `plan/testing/laws.md` checks this.

**Every provider timestamp is kept as given**, converted to Unix
milliseconds. Records without a time get none; no time is invented from the
file's modification time. Each timestamp field's unit and form (RFC 3339
text, epoch seconds, epoch milliseconds) is declared by its type in the
provider's wire struct (`polylogue-sources` wire time types `IsoTime`,
`EpochSecs`, `EpochMillis`, each converting to `Timestamp`), never guessed from the value.

**The framework owns iteration; a provider accounts for every record.** A
parser does not loop over a file. The shared framework frames the input
(lines, JSON documents, ZIP members, SQLite rows), decodes the format's
own encoding, records each record's byte range (for a SQLite row, its table
and rowid; rows are framed table by table in rowid order from the stored
snapshot), and calls the provider once per record:

```rust
fn record(&mut self, rec: &Record) -> RecordOutcome<Self::Ignored>;   // Record: bytes, byte range, table
fn finish(self) -> ParserOutput;                  // what was parsed, and the carry
pub enum RecordOutcome<I> { Used, Ignored(I), Unrecognized(&'static str) }
pub struct ParserOutput { pub parsed: Parsed, pub carry: Carry }
```

(The signatures are the skeleton's, `skeleton/crates/sources`, `framework`.)

`IgnoredKind` is a closed list per provider, each entry with a one-line
reason (Claude Code: the counted-only `progress` kinds, `bash_progress`,
`waiting_for_task`, `query_update`, `search_results_received`,
`mcp_progress`, `sources/claude-code.md`; a kind the provider turns into an
event or a message is `Used`, never ignored); a record of a kind the provider has not listed can only be
`Unrecognized`, which the framework counts per input with the record's
offset. A record that fails to decode never reaches the provider; the
framework counts it. So no parser can drop a record silently, attach the
wrong byte range to a message, or forget to count: those are the
framework's, once. The provider's state between records is its carried
state (`parse_state.carry`, `convergence.md`), which is how an append
resumes. Merging records into one message (one model response from several
records) and building trees happen in that state, across calls. Per input,
`Used` + `Ignored` + `Unrecognized` + the framework's undecodable records
equals the records framed, stored per revision chain in `parse_state`
(`framed`, `used`, `ignored`, `unrecognized`, `undecodable`; an append adds
to them, a whole parse replaces them), so `status` can show all four
(law 120). Within a used record, the
framework records which fields the provider read; the fields it did not
read are kept on the message's `attrs` as given, unless the provider's
declared ignore list names them, so a field is never lost by being
overlooked, only by being named (rule R11). `status` lists the field
names that land in `attrs` on more than 1% of a format's messages, as
candidates for a field of their own (the same rule as Sinex's `extra`).

**Numbers are read losslessly.** Integers are `i64`; any other number keeps
its JSON text when it is not exactly representable, and a non-finite value
is kept as text and counted, never crashes a record or becomes 0.

**Model names are normalized per provider.** Each provider maps its raw model
string to one canonical model name in a small table beside its parser
(`models/gemini-2.5-pro` → `gemini-2.5-pro`, dated aliases folded), which is
what prices and `model:` match; the raw string is kept on the message. A model with no price is listed in
`status`, and its cost is unknown, never zero.

**A session is written only on positive evidence of a conversation:** at
least one record of a message kind the format defines. A file that parses to
records but no messages (a relationship index, a metadata sidecar, a
workflow artifact) is counted as unrecognized or attached to the session it
describes; it never becomes an empty session. Conversely, a file with non-empty
message records never parses to zero messages: that is a parser failure,
recorded as one.

**An empty provider response is recorded as empty, not as "no data".** A
capture or Drive listing that comes back empty is stored as what it is and
shown; it never removes or hides what earlier routes delivered.

**A tool result pairs with at most one call:** the same session, the same
call ID, and never a call that comes after it. An empty or missing call ID
is no ID: such a call and such a result never pair with each other by that
empty value (they stay unpaired; the unpaired call remains unknown, while a
result retains any stated status) unless the format's own rule establishes
the correlation (`polylogue-5gj19`, 42.F090). A call re-emitted under a new
message with the same call ID is one call, paired once. A structurally
recorded empty answer is a result, not a missing output. Its outcome follows
the provider's status/exit code or documented status default, never emptiness
alone; absent that evidence it is `unknown` / `result_without_status`
(`model.md`). A provider's error text is kept beside the output.

**Classification comes from structure only.** A bounded prefix of a file may
choose its parser, but never drops records, excludes a file or decides that
a file holds no session; `written_by`
and tool outcomes come only from structural fields and from fixed-format
lines the harness itself writes (each listed in its provider's section:
Codex `Exit code: N` and `Process exited with code N`, Claude Code
`exit code: N`), never from matching prose inside the content; each ZIP
member is classified on its own.

**Every wire field is accounted for.** Each provider's wire struct field is
mapped (in the tables below), kept in `attrs`, or listed in the format's
ignore table with a reason; a test compares the wire structs with these
tables (`plan/testing.md`). Each spelling or era of a field (camelCase and
snake_case, renamed keys) is a serde alias with an example input of its
own. When a record holds a truncated and a full form of the same value, the
full one is stored and a flag records that a truncated form existed.

**`written_by` rules apply to all providers:** `human` only for text a person
typed (a prompt, an edit, a slash command they typed); `model` for anything a
model generated, including a message another agent sent to this one; `harness`
for everything the agent program put into the conversation itself (tool
output wrapped as a user turn, hook-injected context, the expansion of a
slash command or skill, instruction text, "request interrupted" notices,
error notices written under the assistant role, system prompts, the opening
prompt of a non-interactive run whose author is not recorded). A compaction
summary is an event, not a message (`model.md`, Event). A record that
matches none of a provider's listed forms is `written_by: unknown`, never
defaulted to `human`.

**Generated transcript extracts are not sessions**. A file
that republishes turns of other sessions under an identity the provider
never assigned (a tool's extract of transcripts into a corpus file) is not a
session; under a provider location it is counted as unrecognized (a decoy
example for law 73).

## The same conversation through several routes

A ChatGPT conversation can arrive through three exports taken months apart,
through live browser captures, and (rarely) a page-DOM fallback. A Claude Code
session is a live file and may also appear in hook events. All routes for
the same `<origin>:<native id>` are one session.

**Rule:**
1. **Union by message identity.** The session's messages are the union of all
   routes' messages, by message id. A message seen through any route exists.
   A partial capture never deletes messages it did not see. A message
   with a provisional ID (a fallback ID, or a capture's synthetic turn ID for
   an ID-less page turn; `model.md`, provider and provisional IDs) stays in
   the union, marked provisional, unless it is matched to a provider-ID
   message of the same session: the same role, the same text after
   whitespace normalization, and the same position in the conversation (the
   same parent, or the same preceding matched message). Not `version_hash`:
   records from different routes (a page's DOM against API JSON, an export's
   `$date` wrappers against a capture) differ in shape though they hold the
   same turn. A matched
   provisional message leaves the union and its input becomes one of the
   provider-ID message's sources. The mere existence of a provider-ID route
   never removes unmatched provisional messages: a partial capture with
   provider IDs covers only what it captured.
2. **Same message, different content:** a version the provider marks as
   still being produced (ChatGPT `status: in_progress`, a claude.ai block
   without `stop_timestamp`) loses to any finished version, whatever its
   route; a Claude Code response marked `isAbortedMidStream` is finished.
   Among finished versions, take the version from the most faithful route, in this order: native API capture or live provider file,
   then account export, then page DOM. Within the same route kind, the more
   complete version wins, compared in this order: more blocks, not counting
   empty text blocks (the June 2026 claude.ai export drops 536 empty text
   blocks the April export has, with every message's joined text unchanged;
   counting them would let the older copy win and lose the fields June
   added); then more blocks with non-empty text; then more fields present (so an April ChatGPT
   export's message beats the reduced July export's copy of it). If they are
   still tied: between revisions of one input, rule 5 decides; across
   inputs, the version with the higher BLAKE3 of its record as canonical
   JSON, leaving out the fields its format declares volatile (the same
   fields the fallback ID leaves out, `model.md`, identity), wins (comparing
   the 32-byte hashes), so the choice depends only on content and never on
   input IDs, which for growing files follow arrival. The writer merges one
   input at a time against stored rows, so the preferred version's hash is
   stored (`message.version_hash`) and the comparison is the same in any
   arrival order (law 4). Equal hashes are the
   same version, and no difference is recorded: a provider that re-exports a
   record with only a new stamp (an export time, a sync ID) produces the same
   version, not a revision (the trap Sinex found in a phone app that
   re-stamped every record on each sync). The difference is recorded as a `message_disagreement` row
   (`storage/index-db.md`) so it can be inspected and `status` counts it. This section is the one statement of the merge order;
   `convergence.md` and `decisions/merging.md` point here.
   Comparing two versions ignores what is not content: acquisition state
   (whether attachment bytes were fetched yet), volatile provider metadata
   (generation durations, request IDs), and position in a particular export.
3. **Fields missing on one side are filled from the other** (an export's
   attachment bytes, a capture's reasoning time). This is not "latest wins":
   no route overwrites a value another route has unless rule 2 decides
   between them.
   **Tree position merges as a whole.** A message's `parent` belongs to the
   version rule 2 chooses and is never filled in from another route. A route
   that omits nodes (the July 2026 ChatGPT export) re-parents their
   children; its `parent` loses to any route that holds the omitted node.
4. **Copies lose to originals.** A session file copied somewhere and imported
   (an analysis folder, Drive) is the same session id; its messages merge by
   the same rule, and a copy older than the live file adds nothing.
5. **Revisions of one rewritten file.** When a provider rewrites a file in
   place, every revision is kept (`storage/blobs.md`, inputs, revisions and extents), and
   each format declares what its rewrites mean. **Lossy rewrites** (Codex
   rollouts, which drop records; any Claude Code rewrite besides its first
   record) merge every revision by rules 1–3, and the
   format's pairing rules apply across revisions as within one. **Edits**
   (AI Studio prompt files, Gemini CLI JSON documents, Antigravity
   conversations, Hermes `session_*.json`, a Grok export conversation)
   give the session the messages of the newest revision only; messages only
   an older revision holds are not messages of the session, and stay
   readable with `evidence_until:`. Between revisions of one input, the later one
   wins a tie, by the order of chains in `storage/blobs.md`: within a chain
   the longer revision is later, and the current chain (the one holding the
   latest observation at the live location) is later than any older chain
   (copies, archive members and migrated bytes, observed at `copy:` or
   `zip:` paths). Between two older chains, rule 2's content tie-break
   applies. Modification times and Polylogue's arrival order never decide.
   For a field the format lists as damaged by rewrites, the earlier revision
   wins: a Codex message any older chain holds takes the earliest time an
   older chain gives it (on 151 real pairs, 2,315 of the 17,854 messages both
   versions hold had different times).
6. **Session values that legitimately change** (title, the marked leaf of
   the active path, `collection`, archived or starred state): the value
   stated with the latest provider time wins (the conversation's
   `update_time`, the marking record's `timestamp`); a route that states no
   such time never outranks one that does; on equal times, the value from
   the input with the later provider-stated export time wins (an export's
   own creation stamp, such as the time in a claude.ai export's file name:
   claude.ai regenerates `summary` without changing `updated_at`, 17 times
   across the operator's four exports), and then rule 2's order applies. Placeholders (`New chat`, empty) never win a title. A value that
   stands in for a missing one (capture time used as update time) is marked
   as such and never outranks a real value. If the winning leaf is a node
   another route dropped, the leaf comes from the route that has it.

Why this order: `decisions/merging.md`. The rules are implemented once, as
`polylogue_model::merge`; in backed mode Sinex's fusion reconciler calls that
function for AI sessions rather than restating the rules (joint #20).

## Hooks (live capture from agent harnesses)

This section is the one definition of the hook spool (location, file layout,
line format). Other files point here.

**Writer:** `polylogue hook`, a subcommand that the harness runs on each hook
event (installed into Claude Code and Codex hook configuration by `polylogue
install-agent`). It reads the event JSON from stdin and appends one line to
its own file:

```
$XDG_DATA_HOME/polylogue/hooks/carriers/<provider>/<YYYY-MM-DD, UTC>/<pid>-<start>.ndjson
```

`start` is the hook process's own start time (field 22 of
`/proc/self/stat`, read within the budget). The pair (PID, start) is never
reused within a boot, and the file is opened `O_CREAT|O_EXCL` before the
first write, so each file has exactly one writer: a partial write can only
be finished by the process that started it, no two processes interleave
lines, and a reused PID never extends another process's file. The writer does not need the daemon,
does not open any database, and must finish in a few milliseconds. Event
bodies carry IDs, timings, outcomes and the prompt text; they do not repeat
transcript content. The line format is stable:

```json
{"event_id": "<128-bit hex>", "event_type": "PostToolUse", "observed_at_ms": 1791588187676,
 "timestamp": "2026-10-11T13:20:14Z", "provider": "claude-code", "session_id": "…",
 "proc": {"harness_pid": 41237, "harness_start": 98213377, "boot_id": "7b1e…", "btime": 1791500000, "clk_tck": 100,
          "harness_start_ms": 1791500982133, "cwd": "/realm/project/polylogue/repo"},
 "job":  {"agentctl_job_id": "…", "agentctl_attempt": 2, "beads_actor": "…"},
 "payload": { …event as received… }}
```

- **`proc`, the process identity.** `harness_pid` is the nearest ancestor
  of the hook process whose executable is the harness's (found by walking
  `PPid` in `/proc`; a shell between them is skipped). `harness_start` is
  field 22 of `/proc/<pid>/stat` (start time in clock ticks since boot) and
  `boot_id` is `/proc/sys/kernel/random/boot_id`. The triple (boot ID, PID,
  start) identifies the process: a reused PID or a reboot can never make a
  new process inherit an old session. The writer also records `btime`
  (from `/proc/stat`) and `clk_tck`, and computes `harness_start_ms =
  btime·1000 + start·1000/clk_tck`; wall-clock process times are read from
  these stored fields only, so a rebuild on another boot gives the same
  times. It costs a few small `/proc` reads, within the writer's budget.
- **Event identity.** One hook invocation is one event. Its identity is the
  writing hook process (boot ID, its PID, its start) plus the line's
  position in that process's file, all in the stored bytes, so storing a
  spool file again, or seeing it through two revisions, never makes a second
  event. `event_id` is kept as an attribute. A delivery line's identity is
  its delivery `id` (below).
- **Linking and order.** A pre-tool or post-tool line names the provider's
  call ID (Claude Code `tool_use_id`, Codex `call_id`); it is linked to the
  `tool_call` row with that ID in the same session when the transcript has
  it, and the link is made when the transcript arrives if the hook line
  came first (this is what sets `live_open_tool_pk`). Hook events take their
  place among a session's messages by their time (`observed_at_ms`), after
  messages of the same millisecond; the order is for display only and never
  changes a message's position.
- **`job`**, present only when the environment carries any of these
  variables, which are the only ones read: `AGENTCTL_JOB_ID`,
  `AGENTCTL_CORRELATION_ID`, `AGENTCTL_CHECKOUT_ID`, `AGENTCTL_CHECKOUT_HEAD`,
  `AGENTCTL_RUN_ID`, `AGENTCTL_WORKER_ID`, `AGENTCTL_ATTEMPT`, `BEADS_ACTOR`,
  and `POLYLOGUE_JOB_ID` / `POLYLOGUE_JOB_TRY` for other job runners (keys
  are the lower-cased names). AgentCTL sets the `AGENTCTL_*` names
  (`/realm/project/sinnix/repo/docs/agentctl.md`); nothing on this machine
  sets `POLYLOGUE_JOB_ID`.

On `SessionStart`, the writer also adds a `git` object to the line:
`{"remote": …, "branch": …, "commit": …}`, read directly from the session's
working directory (`.git/config`, `HEAD` and the ref file it names; no `git`
subprocess, so it stays within the time budget). The writer walks up from
the working directory to the nearest `.git` and records that directory as
`root` beside the rest (the worktree root: a directory, unlike the session's
`repo_root`, which is the repository's root commit, `model.md`), so a session in a plain folder inside another
repository is visibly that repository's, never silently. It is absent when
no directory above the working directory is a git repository. Commits are
linked to a session only when the session names them (git lookups,
`convergence.md`), never by reading a repository's history as a folder's own
(the Python adapter ran `git -C` on a plain folder and reported the
enclosing repository's commits as that folder's, `polylogue-b5xfk`). This is the main source of `repo` for
Claude Code sessions, which otherwise rarely record one.

**Repository identity**. A repository
is identified by its root commit (the SHA of the first commit reachable from
`HEAD` with no parent), which stays the same across remote renames, forks and
URL spellings; the remote URL and name are kept as labels. The checkout (the
worktree path) is a separate value. The root commit is read at collect time
by the collector (with `gix`, once per repository directory, since walking
history does not fit the hook's few milliseconds), never by the parser, and
stored in the repo-lookup record, so parsing stays pure; when it cannot be
read (no git, a shallow clone), only the remote is recorded.

**The repo-lookup record**, one per repository directory (slice 1), its own
kind of input, not a companion. Its directory is the nearest one with a
`.git` walking up from a session's working directory, the same walk the
hook writer makes. It is an input of format `polylogue-repo-lookup` at the
location `repo-lookup:<that directory's path>`, holding one JSON object:
`path`, `remote`, `root_commit`, `read_at`. The branch is not in it: a
session's branch is its own record's (`gitBranch`), so switching branches
writes no new lookup revision. The collector reads the working directories
the index holds and, for each that exists on this host, stores the lookup
of its repository (with `gix`, no subprocess), again as a new revision only
when the remote or root commit differ. A working directory that does not
exist on this host (a session copied from another machine, the Morrow
example's `/home/op/morrow`) has no lookup, and its repository is unknown;
that is not "unavailable". The derive step joins sessions to lookups by
path (a working directory at or under the lookup's directory, by whole path
components) into the session's derived facts (`session_facts.repo_root`, and
`repo` where no better evidence names a remote), so a lookup never re-parses
a session. When the writer stores a lookup's new revision it bumps `changed`
on every session whose working directory lies under that directory, so their
facts are recomputed, and `repo_command_last` rows for those sessions are
recomputed with them (`storage/index-db.md`; law 141).
`status` lists lookups as inputs with parse status `done`. A working
directory under no git repository gets no lookup. A session with
no git evidence belongs to its directory and is shown as such. `model.md`
owns the field; `query.md`'s `repo:` matches name, URL or root commit.

**Delivery lines.** Whatever hands Polylogue's context to a session (the
hook command for briefings; for an MCP `brief` or a ping, the daemon, which
records the outcome the proxy reports back in its next request)
appends one more line to its spool file at the moment it hands the text over
or fails to:

```json
{"event_id": "<128-bit hex>", "event_type": "polylogue_delivery", "observed_at_ms": …,
 "provider": "claude-code", "session_id": "<recipient session, or null for the operator>",
 "delivery": {"id": "<UUIDv7>", "sender": "hook" | "mcp" | "ping" | "signal",
   "recipient": "session" | "operator",
   "event": "session_start" | "resume" | "compaction" | "prompt_reference" | "mcp_brief" | "ping" | "advisory" | "signal",
   "content_blake3": "…", "bytes": …, "refs": [ … ], "left_out": [ … ],
   "machine_rules": [ … ], "signal_rules": [ … ],
   "rule": "<signal note id>", "rows": [ … ], "author": "<the rule's author>",
   "outcome": "delivered" | "failed" | "unknown" | "not_sent", "reason": …}}
```

A signal's delivery (`user-data.md`, signals) has sender and event
`signal`, recipient `operator` (a notice to status, the status line or the
notifier) or `session` (lines in a briefing's watching section, recorded on
that briefing's own delivery through `signal_rules`), and carries `rule`
(the signal note), `rows` (the references of the rows that fired it) and
`author` (the rule's author); briefing deliveries leave those three empty.

`signal_rules` names Polylogue's own signal rules whose matches filled the
watching section (`user-data.md`, lines agents choose). `machine_rules`
names the Sinex signal rules whose matches filled the
machine section (empty without Sinex), as `brief.machine` returns them, so
agents can judge over time which rules' lines helped and disable the rest
(operator raw log, 2026-06-28: what is injected at session start should be
tunable by the agents themselves).

The same `id` appears in the reference line injected with the text, which
ties the row to the recipient's transcript. These lines become `delivery`
rows (`storage/index-db.md`); a line written twice has the same `id` and is one row,
and of two lines with one `id`, the one with the later `observed_at_ms`
gives the outcome (equal times: the higher BLAKE3 of the line's bytes). The daemon writes such lines to its own spool,
`$XDG_DATA_HOME/polylogue/hooks/carriers/polylogue/<YYYY-MM-DD, UTC>/daemon-<boot id>-<start>.ndjson`,
opening a new file when the UTC date changes.

**Run evidence** (`model.md`, Run; slice 6). Tasks and
attempts come only from what a harness or job runner records: Codex jobs in
its state databases; Codex goals (`thread_goals` in `goals_1.sqlite`; the
`thread_goal_updated` record stays a `goal` event in the session as well);
and a job ID that a job runner passes into the session's environment
(`AGENTCTL_JOB_ID` with `AGENTCTL_ATTEMPT`, or `POLYLOGUE_JOB_ID` with
`POLYLOGUE_JOB_TRY`), which the hook writer copies into the line's `job`
field. For an AgentCTL job, the collector stores AgentCTL's job record
(`agentctl job get <id> --json`) as an input of its own kind (format
`agentctl-job`, location `agentctl-job:<job id>`), joined to sessions by the
derive step through the job ID, as repo lookups are, so a job record never
re-parses a session; it gives operation, project, attempt and state. Hermes
runs are added when its state database is mapped (slice 5). Each `run` row
names the input field that established it. **Retry evidence**: a subagent
call is a retry of an earlier one only when the harness says so (a Codex
spawn edge naming the earlier thread, a job runner's next attempt of the same job, `AGENTCTL_ATTEMPT` or `POLYLOGUE_JOB_TRY`);
the link then carries `retry_of`. Similar instructions
never make a retry.

Three older layouts are read in place by the same parser:

- one file per session, `hooks/<provider>-<session id>.jsonl`, lines with
  `event_type`, `session_id`, `timestamp`, `provider`, `payload`;
- the pending queue, `hooks/pending/` (about 732,000 events from the Python
  version, the same record shape as the current files), which the reset set
  aside and which goes back in place before the first build
  (`plan/migration.md`);
- the Python version's carrier files, `hooks/carriers/<provider>/…`, the
  current line shape without `proc` or `job`.

In all three, a line's event ID comes from its bytes, never from an
arrival-time input ID: its `event_id` when it has one, otherwise
`~h<hash>-<n>`, the BLAKE3 of the line as canonical JSON and its occurrence
among identical lines of that file (the fallback rule of `model.md`,
Identity), so the same line read through two layouts or two machines is one
event.

**Deleting spool files.** Each spool file is an input like any other. Only
the daemon deletes a spool file, and only when all of these hold: its date
directory is at least two UTC days old (today's and yesterday's files are
never deleted); the file has been quiet for the quiet time (`glossary.md`); it ends
with a newline; its full hash equals the stored revision's; and its writer
process (the PID and start in its name) is gone. Thousands of small files a
day cost nothing worth designing around.

Excerpts: `examples/raw/hooks/claude-code-spool-head.jsonl` (previous
layout) and `carrier-aggregate-head.ndjson` (the Python version's carrier layout, which has no `proc` or `job`).

**Events and what they add.** Every event is stored whole in the spool.
A spool line becomes an `event` row in the index only when it adds a fact the
transcript lacks (the volume rule: otherwise `event` would be the largest table). Lines that repeat
transcript content are counted per input and stay in the stored input,
readable with `read --view raw`. Every event carries `session_id`,
`transcript_path` (which names the transcript file directly) and, in Claude
Code, `prompt_id` (all events of one turn) or, in Codex, `turn_id`.

| Event (Claude Code; Codex has those marked *) | Becomes | Why it matters |
|---|---|---|
| `UserPromptSubmit`* | event `prompt_submitted` with the full prompt including paste contents (`model_saw=1`) | transcripts keep only paste markers |
| `PreToolUse`* / `PostToolUse`* | events `tool_pre` / `tool_post` joined to the tool call by its ID | exact timing per call. `PostToolUse.tool_response` is the tool's structured output (`{filePath, type}` for Write, exit code, `duration_ms`), not the text the model saw: it fills structured fields of the result and never replaces the result text |
| `PostToolUseFailure` (`error`, `is_interrupt`, `duration_ms`) | tool outcome `error` (reason `interrupted` when `is_interrupt`) | failures the transcript may not mark |
| `PermissionRequest`*, `PermissionDenied` (`reason`) | event `permission` (requested, denied, with the reason) | not in transcripts |
| `SessionStart`* (`source`: `startup`, `resume`, `clear`, `compact`, `fork`, `new`; any other value kept as given) | event `session_start`; `fork` is evidence for a `continuation` link, `how: fork` | resume, clear and fork boundaries |
| `SessionEnd`*, `Stop`* | events `session_end`, `stop` | |
| `StopFailure` (`error`: `rate_limit`, `overloaded`, `billing_error`, …) | event `rate_limit` or `api_error` | rate-limit stops for the subscription windows |
| `SubagentStart`* / `SubagentStop`* (`agent_id`, `agent_type`; on stop `agent_transcript_path`, `last_assistant_message`) | events `subagent_start` / `subagent_stop`; the transcript path is evidence for the subagent's parent ("Lineage evidence") | a second source for the subagent link, which matters because `fork-context-ref` is in only 3% of subagent files |
| `InstructionsLoaded` (`file_path`, `load_reason`) | event `instructions_loaded`, with the file's BLAKE3 when readable | which instruction files were in force (agent configuration, slice 6) |
| `CwdChanged` (`cwd`, `previous_cwd`) | event `cwd_change` | |
| `PreCompact`* / `PostCompact`* | events `compaction_requested` / `compaction` | when the `PreCompact` payload carries the assembled context, the hook writer stores it as a companion input of the session: a snapshot of what the model had before compacting |
| `PreModelSwitch` / `PostModelSwitch` | event `model_switch` | |
| `TaskCreated` / `TaskCompleted` | event `task` | todo state, which briefings show |
| `Notification` | event `notification` | |
| `Interrupt`* (Codex) | event `interrupt` | |
| any other event (`Setup`, `UserPromptExpansion`, `PostToolBatch`, `TeammateIdle`, `FileChanged`, `DirectoryAdded`, `ConfigChange`, `WorktreeCreate`, `WorktreeRemove`, `Elicitation`, `ElicitationResult`, `MessageDisplay`, …) | stored in the spool; an event of the same name only when it carries a fact the transcript lacks | |

A hook event whose session has no transcript yet is kept; it attaches when
the transcript arrives.

**Every event is recorded, including unknown ones**. The
recording command records every event it receives: an event type it does not
know is recorded and counted by the parser as unrecognized, and an event that
carries transcript text is recorded, its repeated text counted rather than
indexed.

**Durability against the time budget.** The writer appends each line with a single `write` on a
file opened `O_APPEND`, so a crash can lose a line but never leave half of
one, and then calls `fdatasync`. Measured on the operator's spool filesystem
(btrfs on a Crucial MX500 SSD, 2026-10-11, 1,000 appends of 400-byte lines):
`fdatasync` takes 1.8 ms median, 3.0 ms p95, 5.3 ms p99, 6.9 ms at most,
against 0.002 ms for the write alone. The p99 is under the 10 ms threshold
this rule set, so every recorded event line is synced (the briefing hook's
delivery line is the one exception, below), and a hook event that returned is on
disk. Wear is negligible: even 20,000 events a day at about 16 KB written
each (the block plus btrfs metadata) is about 0.1 TB a year against the
drive's 360 TB rating. If a future filesystem measures over the threshold,
the sync is dropped and the loss window becomes the filesystem's commit
interval (btrfs: 30 s) on power loss or kernel crash only; a crashed process
loses nothing either way, because written lines are in the page cache.
The spool must live on a disk without long sync stalls: on this machine
`$XDG_DATA_HOME` on the root SSD, never `/realm`, whose QLC drive stalled a
10-event commit for 17.1 s while another program wrote heavily (Sinex's
import runs). The synchronous briefing hook never waits for a sync at all:
it writes its delivery line unsynced and the daemon syncs the spool when it
collects it, so the only loss window is a delivery line from the moments
before a power loss, which `status` would show as a briefing with no
delivery record.

**Process spool** (slice 2). The daemon keeps a set of live roots: the
harness process identities named by hook lines of the current boot that
have no `root_gone` line yet; there is no time window. A root counts as
alive only while it is in this set. At daemon start and on every sample, a
root whose PID is missing or whose start time differs gets a `root_gone`
line, so a process that died while the daemon was down, or after a long
silence, is ended at the next look; its descendants with no recorded end
get `ended` with `exit: "gone_while_down"`. Lines written at start carry
`observed_at_ms` = the daemon's start and `ended_after_ms` = the last
observation of that root, so the end is shown as a range. The previous sample after a start
is the set the index shows alive. Every 10 seconds while any root is live (none: no
sampling), it reads every process's parent from one scan of `/proc/*/stat`
and builds each root's descendants from that map, remembering known
descendants between samples so a process reparented to PID 1 or a
subreaper stays tracked. (Walking `/proc/<pid>/task/*/children` costs one
read per thread: 982 threads for 74 processes took longer than the full
scan's 4.4 ms; `prototypes/proc-sampler/`.) It compares the result with the
previous sample. Only differences are written, to its own spool, one file
per daemon run and UTC date (the file name carries the daemon's boot ID,
PID and start time, which a line's ID uses, `model.md`, Identity; `exit` is
`exited`, seen gone between two samples, or `gone_while_down`):

```
$XDG_DATA_HOME/polylogue/procs/<YYYY-MM-DD, UTC>/daemon-<boot id>-<pid>-<start>.ndjson
```

```json
{"observed_at_ms": 1791588190000, "boot_id": "7b1e…", "root": {"pid": 41237, "start": 98213377},
 "started": [{"pid": 41390, "ppid": 41237, "start": 98215002, "comm": "cargo", "argv": ["cargo", "nextest", "run"], "cwd": "/realm/worktree/x"}],
 "ended":   [{"pid": 41388, "start": 98214990, "exit": "exited", "cpu_ms": 184320, "rss_peak_kb": 2411520}],
 "root_gone": false}
```

Every line carries `btime` and `clk_tck`, and each started process its
`start_ms`, computed by the daemon as for hook lines. When the daemon starts
with a new boot ID it writes `root_gone` for every root of the earlier boot
without one, with `ended_after_ms` = the last `observed_at_ms` seen from that
boot: the end time is unknown, bounded below by that time. Processes that
start and end between two samples are not recorded, so process counts are
lower bounds, and `status` says so.

**Resource use.** Each sample already reads every descendant's
`/proc/<pid>/stat`; from the same read the daemon keeps, in memory, the
process's CPU time (fields 14 and 15, user plus system, converted with
`clk_tck`) and the largest resident size seen (field 24 times the page
size). They are written only on the process's `ended` entry (and on
`root_gone` for the harness itself), as `cpu_ms` and `rss_peak_kb`, so
sampling still writes nothing while nothing starts or ends. Both are lower
bounds: CPU used in the last interval before the end and memory peaks
between samples are missed, and a daemon restart loses the peak seen so
far, recorded as `"partial": true`. For processes still running, the live
view reads the same fields from `/proc` at read time; that reading is not
stored. This answers which commands agents launched held the machine and
for how long (`analysis.md`, time; `evidence/intent-journal.md`); the whole
machine's load is Sinex's to record. Process spool files are inputs like
hook spool files, deleted by the same rule, so a rebuild reproduces every
process tree. Only descendants of
harness processes are read; `argv` and `cwd` are stored, the environment
never (`behavior.md`, privacy).

### Which session is asking

Some features need the ID of the session that is running Polylogue (the
`self` reference, the author of an agent's note, leaving the asking session
out of its own briefing). The harnesses expose it to the processes they
start: Claude Code sets `CLAUDE_CODE_SESSION_ID`; Codex sets
`CODEX_THREAD_ID` (the name is present in the Codex 0.162.1 binary; confirm
its value in slice 2). A hook gets the ID in the event's `session_id`. When
none is set, the author is `agent` and nothing is excluded.

### Provider links that resolve to sessions

A provider URL pasted as a reference resolves to the session with that
native ID: `https://chatgpt.com/c/<id>` (and `chat.openai.com/c/<id>`) →
`chatgpt:<id>`; `https://claude.ai/chat/<uuid>` → `claude-ai:<uuid>`;
`https://aistudio.google.com/prompts/<drive file id>` → `aistudio:<id>`;
`https://gemini.google.com/app/<id>` → `gemini:<id>` (the Gemini app
conversation, `sources/google.md`). Any other URL is not a reference.

## Browser captures

This section is the one definition of the capture spool. Other files point
here.

**Writer:** the Polylogue browser extension (ChatGPT, claude.ai, Gemini web,
Grok web, Claude Design). It reads the provider web app's own API responses
(never the page's rendered HTML when an API response exists), sends them
through the extension's native-messaging host (`polylogue browser-host`),
and the host writes one file per capture:

```
$XDG_DATA_HOME/polylogue/browser-capture/<provider>/<conversation id>-<captured_at_ms>.json
$XDG_DATA_HOME/polylogue/browser-capture/<provider>/attachments/<blake3>
```

The host writes to a temporary name in the same folder and renames, so the
daemon never sees a half-written file. File names are built only from the
provider name (from a fixed list), the conversation ID (restricted to
letters, digits, `-` and `_`; anything else is refused) and the capture
time; attachment files are named by their content hash. The daemon watches
the folder like any provider location. The daemon may delete a capture file
once its full hash equals the stored input's. The extension is a separate JavaScript
codebase; this spec covers only the files it produces.

Excerpt: `examples/raw/browser-capture/claude-ai-capture-envelope.json`.

**Envelope:**

```json
{"capture_version": 1, "provider": "chatgpt", "captured_at_ms": …,
 "page_url": "https://chatgpt.com/c/…", "capture_kind": "native_api" | "page_dom",
 "browser_instance": "<random ID per browser profile>",
 "payload": { …the provider's own API response… },
 "attachments": [{"name": …, "media_type": …, "blake3": …}]}
```

Attachment bytes are written next to the envelope and stored as their own
inputs before their provider links expire (the operator lost files to
expired links). The payload is parsed by the same code as that provider's
export for the same structure (ChatGPT's conversation API response is the
same tree as an export conversation; claude.ai's is the same as its export
conversation). The envelope is Polylogue's own format: the extension writes
only its current version, and older versions are read as described in "How
sources work" (Polylogue's own formats). The one older version in existence
is the Python version's envelope (`schema_version: 1`; fields `capture_id`,
`polylogue_capture_kind`, `provenance`, `provider_meta`,
`raw_provider_payload`, `session`, `source`). Its `raw_provider_payload` is
the provider's API response and is parsed exactly like a current capture's
payload (1,212 of the 1,242 old captures have one). The other 30 hold only
the Python version's normalized `session`; it is read as the least faithful
route (the same rank as `page_dom`), so any other route to the same
conversation wins.

`capture_kind: page_dom` is a fallback used only when the extension could not
get the API response. It yields text-only messages and loses to any other
route in merging (below).

**Large attachments.** An attachment larger than one native message is sent
in chunks, each carrying the attachment's total size and BLAKE3; the host
appends them to a temporary file, checks the hash, and renames it into
`attachments/<blake3>`. There is no size cap: the host refuses only when
free disk space runs out, and the extension shows the reason.

**Downloads that fail.** An attachment or Drive download that fails is
tried again in later rounds, oldest attempt first, so one failing file never
holds up the others; each attempt and its error show in `status`.
An attempt is an `observation` of the attachment input with class
`download_attempt`, its time in `seen_at`, and URL, outcome, error and
next retry time in `details_json` (`storage/archive-db.md`). A pending
input may have no revision yet; only a completed download produces bytes
and a revision. The latest attempt determines retry eligibility; the
oldest eligible latest attempt goes first. Two
attachments with the same name and size but different bytes are different
blobs. A response shorter than its declared length, or cut off by a
dropped connection, is a failed attempt, never stored as the attachment's
bytes; a server that declares no length is stored, marked `length
unverified`, and fetched again in a later round to compare (`polylogue-1tyq5.26`:
truncated HTTP bodies recorded as complete).

**Linked pages.** Every URL a message links to is kept as a fact of that
message (an inventory, always). Fetching the linked page is opt-in
(`polylogue setting fetch-links on`): each fetch is stored as an input of its
own, linked to the citing message, deduplicated by URL and bytes, retried
like a failed download, and never made for an excised session. A reference
whose bytes were never fetched is shown as "not fetched", distinct from a
message with no attachment (`polylogue-83u`, from `polylogue-ale`).

**What a capture proves.** The extension reads provider responses inside the
page, so a script running on an allowed provider page (a cross-site
scripting bug, a compromised third-party script) could forge one. Polylogue
accepts this: such a script already holds the user's session there. Each
capture records how it was taken (`capture_kind`, extension version), and
merging follows the route order below. The extension
itself never takes conversation data or commands from page scripts
(`surfaces/extension-and-clients.md`).

## Links Polylogue itself causes

A briefing or ping that Polylogue injects into a session starts with one
reference line (for example `[polylogue brief <id>: refs <session> …]`, format
in `user-data.md`), and each hand-over writes a delivery line to the spool
(above). `reference` links come from those delivery rows and from full IDs or
URLs written in messages, by the one rule in `model.md` (Link). Short IDs, titles and
similar wording never create one. claude.ai, Antigravity and Grok record no
links between conversations; ChatGPT's only cross-conversation evidence is
a shared node ID (a branch into a new chat, `chat-exports.md`), and Gemini
CLI's is a shared `sessionId` (`google.md`). Edits and regenerations are the
message tree inside one session, and nothing is invented for them.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.
