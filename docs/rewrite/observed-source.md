# Configured Claude Code sources

The daemon can follow one configured Claude Code main JSONL file or one Claude
projects directory. Choose exactly one of `claude_main`, `claude_projects` and
`codex_home` in the configuration file; no home source is selected by default.

## One main file

Create a configuration outside the repository:

```json
{"claude_main":{"path":"/absolute/path/SESSION.jsonl","machine":"collector-id","host":"collector-label"}}
```

Run `polylogue --archive DIR daemon --config FILE --until-idle --json`. The
daemon reads the file and its required tool-result bodies once at startup,
waits for that parser work and prints the route, input, chain, revision, stored
length and parser outcome. A waiting partial last line or unavailable parse
returns degraded exit 5 with the reason. Interrupting the wait returns exit 3;
a deadline returns `Timeout` (exit 7). The idle check covers this one input,
not the whole archive.

Without `--until-idle` the daemon serves clients and checks the file once per
second. `--json` then prints a `scope: retention_only` report before waiting.
Plain `daemon` with no configuration recovers already stored observed inputs
without opening their provider paths. Directory discovery is not part of this
mode.

### Appends

Each read compares the whole stored prefix before accepting new bytes. Changed
metadata, a short read, a changed prefix, a shrink or a replacement refuses the
attempt and keeps the bytes already acknowledged. Verification and provider IO
run outside the writer; the owner installs the checked result. The read is
bounded by the size seen at open and is not an atomic filesystem snapshot.

The file keeps one input ID and chain across appends, and the first stored path
supplies the Claude context. Every piece keeps its byte position, including
blank separators and partial records. A separator-only append advances byte
progress without creating a Session or record. A partial nonblank record waits;
restart rereads the stored tail and a later append can finish the record.
Complete Messages before the tail can publish while the tail waits, but the
source as a whole stays pending.

Exact Message and Block reads need the selected body and current source evidence
for that prefix in the same pinned view, and they keep pending Session context.
A whole-Session transcript still refuses while work is pending. A changed
acquisition invalidates the old row receipt. Observed receipts use
`scope: observed_source_rows`, name the admitted revision and length, and omit
`original_hash`; their binding is not a full-content hash.

### Reading stored bytes

`polylogue --archive DIR input cat ID` reads stored pieces without touching the
provider file and verifies piece hashes before output. Output goes through the
daemon's credential masker; hashes, citations and byte offsets still refer to
the original bytes. Invalid UTF-8, binary or control bytes and JSON escape
spellings are refused before output. There is no `--reveal`. Errors: unknown
input `NotFound`, unsupported presentation `NotYetAvailable`, excised content is
refused, expiry `Timeout`. Disconnecting a reader does not cancel source work.

### Required tool-result bodies

The daemon also reads tool-result files that complete stored records name
explicitly, at `<main directory>/<main stem>/tool-results/*.txt`. The source
adapter recognizes `persistedOutputPath` and the saved-output text forms. The
directory is never listed. Each attempt scans the whole main in 64 KiB windows,
even when the main is unchanged. Limits: 32 referenced paths, a 1 MiB reference
record and 1 MiB per body. The main's parent directory resolves normally; the
session directory, `tool-results` and the body file are opened `NOFOLLOW`
through held directory handles, and symlinks or non-regular files are refused.
Normalized text must be UTF-8. Full structured output keeps the reader's
65536-byte JSON limit.

Each body is an immutable `claude-code-tool-result` input whose context holds
the owning main input ID, the path and `required` or `inline-duplicate`. The last
local observation of a path selects its snapshot. A checked copy of complete
inline text is redundant: it is not part of the Message version, and losing it
does not affect the inline result.

The sorted body manifest is part of the main parser's input, so a changed
required body reparses from stored bytes. The result Message version includes
its raw record and the body's digest and length; the assistant call's version
is independent. A missing body leaves the Message pending. Denied reads and
corrupt stored bytes are errors, never empty results. Removing a provider copy
keeps stored snapshots, and restart reads only stored bytes.

Session JSON and CLI/MCP rows show the Message's `version_hash`. Full result
text has a `claude-code-tool-result-body` field origin naming the body input,
chain, revision, path, byte range and blob hash; the verifier rereads that
snapshot rather than trusting the inline preview. Reading one Message or block
checks only the bodies its own field origins use, so another Message's lost
body does not block it (the Session context is then degraded). A full Session
read and the owning tool-result read still refuse a missing body.

### Acquisition turns

`claude_main.acquisition_quantum_bytes` sets the work allowed per turn, from
65536 to 8388608 bytes (default 8388608), including verification. It limits a
turn, not the file. Claude and physical Codex history share one parked
acquisition slot with ordinary imports, and readers keep using acknowledged
state while a large capture waits. Cancellation discards uninstalled work; a
later process verifies the stored prefix again.

Status reports `claude_continuation`: active acquisition, attempts,
acknowledged length, unchanged passes and diagnostics. Missing and denied main
paths are reported separately from an empty file. Unchanged polls add no
revisions or body observations and do not restart parser work. A changed
required-body manifest reparses even when the main is unchanged.

## A projects directory

```json
{"claude_projects":{"root":"/absolute/synthetic/projects","machine":"collector-id","host":"collector-label","acquisition_quantum_bytes":65536}}
```

This mode runs continuously; `--until-idle` is refused before the archive is
created. Recognized paths:

- `-project/SESSION.jsonl` or `.ndjson` (main sessions),
- `-project/OWNER/subagents/agent-AGENT.jsonl`,
- `-project/OWNER/subagents/workflows/RUN/agent-AGENT.jsonl`.

Main stems give main context. Subagent records give their own membership and
the owner directory gives lineage. This declaration is stored and survives
provider removal and reparsing; a main declaration cannot replace an
established agent declaration. A subagent file's missing `sessionId` stays
unresolved: the folder never substitutes for it, a later owner does not claim
earlier records, and a conflicting later owner does not move an established
Session.

Traversal reads at most 128 entries per turn with a fixed-depth directory
stack, and continues without waiting for filesystem events; a finished cycle
pauses before rescanning. Each transcript uses the same acquisition slot,
per-turn byte grants and body resolver as the single-file mode, and every
capture verifies the full prefix. The current input is parsed to completion
before another is chosen. Recent changes and background catalog inputs
alternate so ongoing writes cannot starve the scan; a durable catalog drives
restart recovery.

Traversal and capture use held directory descriptors and descriptor-relative,
no-symlink opens beneath the root. The archive cannot be inside the configured
root. Replacing the root starts a new walk; a missing or denied root is
unavailable, distinct from an empty one. Neither an interrupted scan nor a
partial tail means complete coverage.

Status reports scan progress, deferred and excluded entries, attempted
acquisitions, a sample of stored input IDs (never used to drive acquisition)
and the path currently being acquired. Byte totals sum acknowledged lengths.
Memory documents, session indexes, workflow journals, agent metadata, hook
files and unreferenced bodies are not read.

A Session's membership is checked on its own: an unrelated file's partial tail
leaves global relationship coverage pending without making the requested
Session's Messages pending. `settled_through.inputs` is a display sample plus the
exact sources needed; `inputs_total`, `inputs_enumerated` and `inputs_digest`
say whether every input was enumerated and commit to all of them. The digest is
BLAKE3 over `polylogue.session.input-bases.v1\0` followed, in catalog order, by
each input's JSON length as big-endian u64 and its JSON bytes.

Body lookup uses a partial expression index on the main context value, so
unrelated inputs are not scanned. Archive formats 1 through 3 upgrade in place;
unpublished format-4 archives from commit `14cc1c702` predate this index and may
be refused by schema validation.

## What the Claude parser reads

Parser versions are part of the parse input: after a version change, stored
contributions are reparsed before they are current. Selection codec 1 and
adapter recipe 2 keep their meanings.

- `toolUseResult.isImage` and `toolUseResult.noOutputExpected` are kept as named
  execution metadata when supplied. Absence adds no default, and neither infers
  image content, outcome or missing output.
- `message.stop_details` is kept as opaque response metadata `stop_details`.
  Supplied null is a value; absence adds none. The last record that supplied it
  wins, and an overlay that omits it keeps the earlier origin. It never sets
  finish state or stop reason.
- Immediate unknown block members are kept as opaque block extras with their raw
  paths. A member that collides with a structured ToolResult channel is refused.
  JSON and export put extras in `data` on ToolCall and ToolResult and in
  `attrs` on other blocks; a Thinking block's ciphertext `data` stays separate.
- `inference_geo`, `iterations`, `server_tool_use` and `speed` usage members are
  kept in `attrs.provider_usage` as the raw `message.usage` object, separate from
  the normalized counters. A later usage record without them removes the bag;
  an overlay that omits usage keeps it. Unknown usage members, unknown nested
  counter details and invalid counters still block certification.
- Each immediate record extra is selected once per grouped response; the last
  record that supplied it wins, including a correction to an earlier
  contribution. Version evidence appends differing corrections ordered by key,
  tagged `response/record_extra/` plus the lowercase hex of the key's UTF-8
  bytes, after the fixed metadata units and before required bodies.
- These `toolUseResult` members go to `tool_result.data` unchanged:
  `canReadOutputFile`, `content`, `description`, `file`, `filePath`,
  `gitOperation`, `matches`, `message`, `newString`, `oldString`, `outputFile`,
  `pin`, `prompt`, `query`, `replaceAll`, `resolvedModel`, `resumedAgentId`,
  `staleReadFileStateHint`, `staleRecovered`, `success`, `total_deferred_tools`,
  `type`, `userModified`. A string `toolUseResult` becomes
  `data.toolUseResult` only when exactly one result in the record can own it.
- A result content array of `tool_reference` objects (`type`, nonempty
  `tool_name`) and text objects (`type`, string `text`) is kept whole as
  `data.content_items`. It supplies no text output.
- A Message's parent follows recognized non-Message records to the nearest
  Message: attachment, system, event and progress records are skipped, and a
  compaction boundary uses its `logicalParentUuid`. The first contribution owns
  a folded Message's parent. Missing records, invalid relations, conflicting
  UUIDs, cross-Session edges, self-references and cycles leave the parent
  unavailable. Direct parents keep `ParentReferenceV1`; parents found through
  skipped records use `ParentViaRecordsV1`, which cites every traversed record.
  The lookup stops at the nearest Message and does not walk its ancestry.

## Long string fields in documents

Whole Session JSON and directory export can stream a string-valued
`ToolCall.input`, `Thinking.data` or `Other.data` larger than the 64 KiB leaf
limit. Malformed JSON-in-string tool arguments stay the supplied string. The
renderer reads the field in bounded ranges, checks it with the credential
masker, rereads it to confirm the result is unchanged and then writes it, so it
keeps small buffers but makes three passes. Range reads keep the visibility,
deadline and cancellation checks. Oversized objects and arrays keep their
limits. A failed write may leave partial output but is never reported as a
complete document or export.
