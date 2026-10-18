# Claude and Codex features ported from the Python version

The three synthetic fixtures under `crates/sources/tests/fixtures/python/` are
exact copies of files from the Python Polylogue at commit
`b4b61c7eab8daf740bfdccb146e530906839726d`. `manifest.json` records their Git
blobs and SHA-256 digests, together with the Python tests that hold the
expected facts. The Rust tests in `crates/sources/tests/python_fixtures.rs`
restate those facts as literals; they do not run the old parser.

## Claude Code

- Records without a usable UUID become provisional Messages, identified by
  their full content plus an occurrence counter. No provider UUID is invented.
- Provider cost and duration are kept exactly.
- A background-task notification that names the acknowledged task and
  invocation completes the original ToolResult, with its own version evidence
  and field origins. Submission alone leaves the outcome unknown. Output-file
  paths in the notification are data and are never opened.
- Parser 17 keeps the latest `last-prompt` or `summary` leaf marker, even when
  a later marker is invalid. Default membership follows that record's path
  through UUIDs and non-Message relations before folding response fragments. A
  response is active if any of its contributions is. A pure tool-result Message
  stays a peer of an active response when its earlier call is known:
  `sourceToolAssistantUUID` names the caller, otherwise the nearest assistant
  parent does. Identical re-emitted calls with the same ID count as one call,
  and two result Messages cannot answer the same call. Caller and tree parent
  must agree. Conflicting payloads, missing or cross-Session parents, later
  calls, cycles and conflicting paths do not make a Message a member. Position
  at the end of the file never does.
- An off-path tool result is left out of the active set only when its caller and
  call are both known to be inactive. Missing, null or cyclic caller ancestry
  leaves membership unresolved.

## Codex

- The legacy header timestamp sets the first Message's time, with the original
  field as its origin. Later legacy Message times stay unknown.
- Legacy bare response records and current function, custom and code output
  headers are supported. A complete provider header supplies outcome, exit code
  and elapsed time; prose in the output and a multi-line command preamble do
  not. Running stdin stays unknown.
- `last_token_usage` counts keep their original numeric selectors. Uncached
  input is `input_tokens - cached_input_tokens`, saturating at zero
  (`UncachedInputV1`). Without a valid cached count the input count keeps its
  direct selector; absent, null or invalid counts never become zero.
- A flat rollout with no Event, Link, conflicting item or failed reach-back
  defines its complete active order. Flat rollouts may also contain
  instruction, world-state, turn-context, turn start/end and rate-limit events;
  each is decoded and checked for shape and turn and Session scope. Unknown or
  malformed events, compaction, repeated Session metadata, links, conflicts,
  reach-back and pending input keep the order unknown, and readers must ask for
  `branches: true`.
- History SQLite and prompt history are still needed for complete Codex
  conversations; a rollout alone is not enough for a mixed configured home.

## Differences from the Python version

| Source record | Affected field | Rust behavior |
| --- | --- | --- |
| Claude changed contribution with uuid `main-u1` | Message content and origins | The latest contribution, `Conflicting duplicate that must not survive.`, replaces the earlier content. |
| Claude `other-u1` and `other-a1` with a foreign `sessionId` in a main file | Session membership | Kept as copied history of the file's Session, with the original Session claims preserved. |
| Final Claude `isSidechain: true` record | Session membership and parent | A separate subagent Session; the main-chain `parentUuid` stays auxiliary evidence rather than becoming a cross-Session parent. |
| Claude records without a usable identity | Message identity | Provisional identity from full content plus an occurrence counter. |
| Claude `user` record containing tool results | Role and author | Role `user`, `written_by: harness`, typed ToolResult blocks; Python used role `tool`. |
| Code output status item `Script completed` or `Script failed` | Codex wrapper outcome | `ok` or `error`; malformed status items stay unknown. Says nothing about child commands. |
| Legacy Codex reasoning followed by calls | Message grouping | Thinking joins the model request with its calls. |
| Legacy shell output stored as a JSON-encoded string with output, exit code and duration | ToolResult text | The whole recorded string stays the text; decoding it supplies status and duration. Python kept only the inner output. |
| Legacy state record | Record accounting | Counted as ignored with no content. |

## Reparsing and reads

- A new parser version can replace an unfinished Message preparation. The
  owner first checks the current source token and stored revision, then retires
  the old preparation in ordinary bounded turns before publishing again.
  Identical preparations resume. Stored inputs, ready bodies and published
  identities are kept, and old source descriptors are not rewritten.
- One source turn reads a 64 KiB window and admits up to 1,024 records from it,
  so tiny ignored records do not each cost a worker and an install. The 1 MiB
  record limit and the state and install budgets still apply.
- An append can make an in-flight final source check obsolete. That attempt is
  discarded only when the writer finds that the stored source or parser state
  changed; then current work is selected again. Unchanged-state failures and
  corruption are recorded as failures.
- A model-read continuation is checked against the current selected rows
  before any active-path refusal. If competing membership invalidated an
  earlier selection, the reply is `stale_read`; a fresh request with the same
  unknown active path is refused rather than returning an empty page.
