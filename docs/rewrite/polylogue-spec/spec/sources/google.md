# Google: AI Studio and Gemini CLI

General rules (identity, companions, unrecognized content, merging, spools) are in [README.md](README.md).

## Google AI Studio (Drive)

**Where:** AI Studio saves prompts into the user's Google Drive folder
"Google AI Studio". Polylogue lists that folder through the Drive API (OAuth
credentials stored as state with `polylogue auth drive`) every few minutes,
by modification time. Only files with the AI Studio prompt MIME type are
sessions; other files in the folder (uploaded documents, images, transcripts)
are attachments. `applet_access_history.json` in the same folder (`{"applets": [...]}`,
which applets the account opened) is stored as an input of its own and not
parsed: it carries no conversation (the Python version kept it the same way).

Volume: 239 sessions pre-reset, about 12k messages, files up to 213 MB
(inline images).

**Format:** JSON (files usually have no extension):
`{runSettings, systemInstruction, chunkedPrompt: {chunks: [...]}}`.

| Field | Becomes | Rule |
|---|---|---|
| Drive file id | `native_id` | |
| Drive file name | `title` | |
| `chunks[].createTime` | message `at` | present on every chunk in current files (e.g. a 2026-07 prompt: four chunks, each with `createTime`); the pre-reset archive dropped it, which was a parser bug |
| Drive `createdTime` / `modifiedTime`; for a Takeout copy, the ZIP entry's modification time | `started_at` / `last_activity_at` bounds, and the fallback `at` for older files whose chunks lack `createTime` | a fallback time is `time_quality: file_mtime` with `precision: bounded` (created to modified; `model.md`, Time), never presented as the message's own time A ZIP entry's time is zoneless local time (DOS format): it is read in the zone recorded with the Takeout archive's observation (the collecting machine's zone, `model.md`, Time), as `time_quality: file_mtime` with `precision: second` (`model.md`, Time), and never enters an ID. |
| `runSettings` (model, temperature, …) | session event `run_settings`; `model` on model messages | |
| `systemInstruction` | message `role=system`, `written_by=harness` (it is what the user configured, but it is not a turn) | |
| `chunks[].role` = `user` / `model` | `role=user, written_by=human` / `role=assistant, written_by=model` | consecutive chunks with the same role form one message: thought chunks become its thinking blocks, attachment chunks its file blocks, and its usage is the sum of the chunks' `tokenCount` |
| `executableCode` / `codeExecutionResult` | tool_call / tool_result pair | |
| `branchParent.promptId` | `continuation` link, `how: fork`, to `aistudio:<promptId>`; chunks equal to the parent's are marked `inherited_from` | `branchChildren` names other sessions, not parents inside this file |
| `chunks[].text` or `parts[].text` | text block | a chunk with both does not repeat the text; a chunk without a role is skipped and counted; an empty `chunkedPrompt`, or `chunks` of the wrong type, is not a session |
| `chunks[].isThought: true` | thinking block | `thoughtSignature(s)` kept |
| `driveDocument`, `driveImage` (Drive file ids) | file/image block | the referenced Drive file is fetched and stored as its own input |
| `inlineImage` (base64) | image block | decoded bytes stored in the blob store |
| `tokenCount` | a model message's usage: its chunks' `tokenCount` as output, and the preceding user chunks' `tokenCount` as that model message's input | marked as the provider's count |

**Takeout route.** Google Takeout exports the same prompt files under
`Takeout/Drive/Google AI Studio/` inside its ZIP (e.g.
`/realm/account/google/raw/takeout/*.zip`); `polylogue import` reads them
with this parser (Takeout has no Drive file ID). A Takeout prompt is read
in place inside the stored Takeout archive. Its session ID comes from its
own bytes alone: `aistudio:~takeout-<first 16 hex of BLAKE3 of the first
chunk's canonical JSON>`. When its bytes equal any stored Drive revision,
not only the newest, it is an observation of that revision, never a new
input, so a Drive edit that dropped messages cannot bring them back through
an old Takeout copy; the derive step then records the Takeout ID as an alias
of the Drive session, which the ID resolves to. Otherwise it is a session
of its own. The alias is recomputed from stored bytes whenever either
arrives, so any arrival order converges to the same state, and no ID
depends on what else is stored.

**Message identity:** a chunk `id`, when present, is the native ID of the
message it starts; otherwise the fallback rule (`model.md`, Identity).
Chunk grouping, chunk IDs, branches and code execution rest on the Python
parser, not on a scan of real files: check each against real files in the
slice-5 field scan.

**Input identity.** A Drive file's location is its Drive file ID
(`drive:<file id>`), never its name: AI Studio allows several prompts with
the same name, and same-name files must never merge into one input or one
session. A catch-up listing reports how many Drive files exist versus how
many are stored, so a partial catch-up is visible.

**Quirks:** a prompt file is rewritten in place as the conversation grows (not
appended): each change is a new revision of the input, and the session is
re-parsed whole. It is an edit format (`README.md`, rule 5): the newest
revision is the conversation, and earlier revisions are history, readable
with the `evidence_until:` query field.

**A round that saw the folder change**. After downloading,
the collector lists the folder again; if any file's ID, media type, revision
or size changed while the round ran, it is marked incomplete and the
next round starts at once. Renames alone do not count. **A download counts
only when complete**: a file is stored as a revision only
when its size matches the listing and it parses to its end; a short download
is retried, never stored as a revision.

## Gemini app (Takeout My Activity)

The Gemini app's conversations (gemini.google.com and the phone app) are
AI chats, so Polylogue reads them (origin `gemini`); in backed mode Sinex
stores the Takeout archive bytes as it stores any Takeout product, and
Polylogue reads the members below as inputs (agreed with the Sinex
session, 2026-10-12).

- **Where.** `Takeout/My Activity/Gemini Apps/MyActivity.html` inside a
  Takeout archive imported with `polylogue import` (the operator's archives
  are under `/realm/account/google/raw/takeout/`), plus the attachments
  beside it (generated images, `combined-sources-*`, `combined-docs-*`,
  `download-*.zip`), which become attachments of the entry that names them.
  `Takeout/Gemini/` (gems, scheduled actions) is stored and counted, not
  parsed (`no_parser_yet`, none planned).
- **What an entry is.** One activity entry holds the operator's prompt, its
  time with zone, and the model's response; it is one human and one model
  message. Measured on the 2026-08-26 export: 103 entries, 41 distinct
  conversation links `gemini.google.com/app/<id>` in them.
- **Sessions.** Entries naming the same conversation link form one session
  `gemini:<id>`, in entry time order (the format records no other order);
  an entry with no link is a session of its own, its native ID a hash of
  its prompt text and its time as a UTC instant (the zone Google prints is
  volatile: exports relabel times into other zones, `model.md`, Time).
  Message IDs: each entry has an entry hash (the same prompt text and UTC
  instant), and its two messages are `<session id>:<entry hash>:h` (the
  prompt) and `<session id>:<entry hash>:m` (the response); the HTML has no
  JSON records, so the fallback rule's canonical JSON does not apply.
- **Coverage.** My Activity is complete as of its export date for the
  activity Google kept (activity settings can delete or not record it), so
  analyses over it carry that coverage bound; overlapping exports merge by
  entry identity like any route.
- **Slice 5**, with the other Google providers; the HTML structure is
  inferred from real exports (`schema infer`) before the parser is written.

## Gemini CLI

**Where:** `~/.gemini/tmp/<project name>/chats/session-<timestamp>-<id>.json`
(a named directory such as `sinex` or `polylogue`, holding a `.project_root`
file; `projectHash` appears only inside the documents; 31 chat files on this
machine, `evidence/gemini-hermes-real-survey.md`),
one whole-session JSON document, rewritten as the session grows (an edit
format, `README.md` rule 5); newer
versions write `chats/session-*.jsonl`, one turn per line, appended. Both are recognized by exact path
pattern, anchored at `tmp/<project>/chats/` (one project on this machine is
named `chats`, so its `logs.json` sits at `tmp/chats/logs.json` and a match on
`chats/` alone would take the prompt log for a chat), and first record; the JSONL form is append-only and parsed
incrementally, the JSON form is a whole-file format. A Gemini JSONL file must
never be taken for a Claude Code file just because a record passes the
Claude Code shape (decoy inputs).

Excerpt: `examples/raw/gemini-cli/session.json`.

| Field | Becomes |
|---|---|
| `sessionId`, `kind`, `startTime` | `native_id` = `<sessionId>/<kind>/<startTime as Unix ms>` (kind one of `chat`, `main`, `subagent`, else empty; no `:`, which session IDs reserve because message IDs append `:<id>`, and the ISO start time would bring colons): one CLI process writes several chats and subagents under one `sessionId`, and `startTime` is fixed when a chat opens while `lastUpdated` changes. A `subagent` chat gets a `subagent` link from the `main` chat with the same `sessionId` (evidence `sessionId+kind`), with no spawning block. A chat at `chats/<main sessionId>/<id>.json` is a `subagent` of the main chat whose `sessionId` is the directory name (evidence `directory`), even with its own `sessionId` (1 of 3 subagents on this machine) |
| `startTime`, `lastUpdated` | times |
| `summary` | the provider `title` (13 of 31 documents) |
| `directories`, `memoryScratchpad`, `userMessageCount`, `hasUserOrAssistantMessage` | kept in `attrs` |
| `projectHash` | session event `project`; the cwd may come from the project directory's `.project_root` file (check its contents in the slice-5 field scan) |
| `messages[].id` | message id |
| `messages[].type` = `user` / `gemini` | `role=user, written_by=human` / `role=assistant, written_by=model`; except a `user` record whose parts are all `functionResponse` (55 in the JSONL files here): `role=tool, written_by=harness`, its parts tool results paired to calls by ID, outcome `error` when `response.error` is present and otherwise `unknown` with reason `result_without_status` |
| `messages[].type` = `info` / `error` | message `role=system, written_by=harness` (error also as event) |
| `messages[].content` | text block; when it is a list of `{text}` parts (213 of 1,078 messages), their joined text |
| `messages[].displayContent` | the typed prompt when `content` is empty: a user turn with empty `content` and non-empty `displayContent` keeps the typed text as its text block |
| `messages[].thoughts` | thinking blocks whose text is the thought's `description`; its `subject` and `timestamp` are kept in the block's `data_json` as `{"subject", "at"}` (no other provider names its thoughts, so they get no field of their own; `model.md`, blocks) |
| `messages[].toolCalls[]` | tool_call and tool_result blocks (name, args, result, status → outcome: `success` ok, `error` error, `cancelled` error with reason `interrupted`); the call's `timestamp` is its time; `displayName`, `description` and `renderOutputAsMarkdown` are kept in `attrs` |
| JSONL `content` parts `{functionCall, thoughtSignature}` (118 parts; `toolCalls` is present on only 82 of 275 model records) | tool_call blocks, deduplicated by call ID against `toolCalls`; `thoughtSignature` becomes the thinking signature |
| `toolCalls[].resultDisplay` | kept in the result block's `data_json`: what the user was shown |
| `toolCalls[].result[].functionResponse` | the result text; when a record holds both a truncated and a full form, the full one is stored and a flag records that a truncated form existed |
| `messages[].tokens` (`input`, `output`, `cached`, `thoughts`, `tool`, `total`) | usage (`cached` as cache read, `thoughts` as reasoning); Gemini's `input` includes cached tokens, so uncached input = `input − cached`, not below 0; `tool` is kept in `attrs`; `total` is a provider total, used as a check against the parts and never as an input (`analysis.md`, token rule 7: a provider total is kept beside, never merged) |
| `messages[].model` | `model` |

**The JSONL form**. A `.jsonl` file starts
with a stub header carrying `projectHash` and `kind`, followed by turn
records and `$set` patch records; `$set.messages` replaces the message list
whole. The parser folds the log into the whole-session shape in file order;
`error` and `info` records are transcript. The stub header is what tells a
just-opened Gemini file from a Claude Code file.

**Tool output files and masked output**. Large tool
output is masked in the transcript with `<tool_output_masked>` ("Output too
large. Showing first N and last M characters": the first 8,000 and last
32,000 characters), so the inline text is not a prefix of the output. The
full output is in `~/.gemini/tmp/<project name>/tool-outputs/session-<id>/<tool>_<epoch ms>_<n>`,
a companion file of the session, matched to its call by, in order: the call
ID in the file name; a path to the file named inside the masked text; the
call ID between `_` delimiters in the file name, longest ID first (about 27
of 218 files have no ID in the name). The file's text replaces the masked
text; a file that cannot be read is counted for that file, not as a failure
of the session.

**Prompt log and checkpoints.** Each project directory has a `logs.json`, a
list of `{message, messageId, sessionId, timestamp, type}` entries: a prompt
log, stored as an input and used like Claude Code's `history.jsonl`
(prompts by session ID). `~/.gemini/history/<name>/` holds Gemini's
checkpoint git repositories (copies of working trees), not transcripts: not
collected. Other files the model wrote into a project directory (scripts,
binaries, option dumps beside `chats/`) are not collected either. A
`*.partial` file in `chats/` (4 here) is an in-progress write: it is stored
as an input but not parsed until it is renamed to its final name, and
`status` counts such files.

**Counts are after folding.** A session's messages are the folded list (the
JSONL log replayed into the whole-session shape), so counts compare after
folding: 1,052 messages on this machine against 1,078 raw message records
(`evidence/gemini-real-accounting.md`).

**No parent links from position**. Gemini CLI
messages carry no parent IDs; their order is file order, and no parent links
are made from position.

Provenance (Bead and audit IDs by section) and the Python-version lessons behind some rules: `evidence/spec-provenance-model-sources.md`.
