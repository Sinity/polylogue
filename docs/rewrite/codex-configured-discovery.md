# Following a configured Codex home

`daemon --config FILE [--until-idle]` takes exactly one source. A Codex source
names its home:

```json
{"codex_home":{"home":"/absolute/disposable-codex-home","machine":"synthetic-machine","host":"synthetic-host"}}
```

Optional `rollouts`, `catalog` and `history` default to `sessions`,
`state_5.sqlite` and `history.jsonl`. Relative paths, including
`threads.rollout_path` values, resolve against the home; `..`, paths outside the
home and symlinks are refused. No default home is opened, and the provider files
are never modified.

At startup the daemon captures what it finds; then it keeps polling in bounded
turns with a one-second pause, using the store's discovery and the ordinary
capture and parse queues. `--json` prints a `retention_only` startup report.
`--until-idle` disables polling, waits for the startup rollouts to be parsed,
leaves prompt references pending and returns degraded exit 5.

## What is collected

- Rollouts: the store lists `rollout-*.jsonl` files whether or not a thread
  names them. Each readable file becomes an immutable snapshot parsed by the
  Codex parser; changed bytes become another snapshot (no revision lineage).
- Catalog: the SQLite catalog is opened read-only in one read transaction and
  its `threads(id, rollout_path)` rows are admitted. A missing or malformed
  catalog does not stop rollout collection.
- Prompt history: `codex-prompt-history` stores `history.jsonl` bytes. It has
  no conversation parser. The sources crate reads complete records with a
  nonempty `session_id` and string `text`; their times and text do not become
  Messages. Status and queries show them under `pending_references` with byte
  ranges and BLAKE3 hashes. Reference counts count prompt records, not
  conversations. A read of a prompt-only reference returns an unavailable
  conversation with its pending evidence, never an empty conversation. When
  conversation evidence with the same Session ID arrives, the reference reports
  `conversation_evidence`. Malformed records and partial tails are counted.

Sessions are found through the ordinary readers (`origin:codex`, `tool_calls
where session:ID`, returned operation references). Removing a provider file
keeps stored snapshots and citations, and a failed capture does not erase
earlier ones.

## Startup status

`status.acquisition` describes startup only. `accessible_empty` means a
completed traversal or an empty file; `unavailable` carries a reason and
`entries: null`. A failed traversal may still report a captured prefix.
`retained_count` and `refused_count` count capture attempts for rollouts,
prompt history and configured physical history. `retained` samples up to 32
input IDs and `refused` up to 4 failures, with omitted counts. Paths longer than
1024 bytes are replaced by a BLAKE3 `path_key`. Catalog rows, the directory
listing and snapshots are separate observations, not one atomic snapshot.

Startup limits: 512 directory entries, depth 8, 512 catalog rows, 512 rollout
paths. Paths beyond that are reported as `deferred_rollout_paths` (known but not
attempted, not a total). Files and the catalog are at most 16 MiB each, 32 MiB
per startup. Prompt snapshots admit 1 MiB and 4096 references; the public view
examines at most 512 snapshots, 4 MiB of prompt bytes and five seconds of work,
with 8 KiB reference and 4 KiB snapshot samples.

## Continuous polling

`status.continuation` reports the last finished turn: catalog and rollout
availability, admitted path prefix, enumeration cycle and whether it completed,
examined, unchanged, acquired and failed files, charged bytes, deferred
candidates and history availability. `pass_active` marks a running turn.
`partial` means the total is unknown and the cursor is kept; `accessible_empty`
means a completed empty cycle.

A store-owned cursor keeps at most nine directory iterators and reads at most
512 entries per turn. It resumes where it stopped, starts a new cycle after the
end and resets when the root changes. One batch holds up to 512 directory paths,
512 catalog paths and the prompt file (at most 1025 candidates) and drains over
as many turns as needed. Each turn evaluates at most 512 candidates and 32 MiB
of declared source bytes. `budget_exhausted` and `deferred_candidates` describe
unfinished work.

Fingerprinting and capture run outside the writer. Before reading content, the
owner checks its acquisition policy. Content identical to a stored input on the
same machine, host and path is recognized without a new request, observation or
input. Changed files go through the ordinary import slot with their expected
length and hash checked. The single polling worker shares daemon cancellation.
The catalog stays a read-only snapshot of at most 512 rows, reported partial
when larger, and is never repaired. A file that changes while the daemon is
stopped is picked up at the next startup.

## Physical history database

`codex_home.paginated_history` selects a thread-history database (same path
rules, no default). `paginated_interval_seconds` is 1 to 86400, default 300.
Startup reports `paginated_history` as pending, or unavailable if the path check
fails. Status shows `paginated_pending`, `paginated_state` and the latest
`paginated_input` and length. A valid but unknown schema is stored and reported
as `no_parser_yet`, never as an empty conversation. Parsing admits 16 MiB and
4096 rows.

Capture runs in a parked worker that receives reusable work grants, alternating
with regular captures, so other work and reads continue. One read-only
transaction reads the page count and every page through `sqlite_dbpage`,
including committed WAL content and page one; uncommitted writes are excluded,
and no temporary copy is written. A long read transaction can delay WAL
checkpoints, and in rollback-journal mode it blocks the provider's writer; only
WAL mode has been tested with a concurrent writer. A turn admits 8 MiB of work
charges or 512 checkpoints, and a 300-second guard refuses incomplete work.
These are cooperative checkpoints, not hard latency, memory or IO limits.

The first capture stores one complete object. Later captures compare against
that base through a 4 MiB window and store all differing pages as one patch, so
extents reference only the base and the current patch. Over 8192 extents
triggers a full new base from the same read view. Everything is written before
one selection transaction; an interrupted capture can leave unselected objects
(`blobs/tmp/*.part`) but never a selected partial source, and it restarts from a
fresh view rather than resuming.

Metadata of the main and WAL files only schedules work. Restart forces
verification; unchanged metadata skips work until an hourly full check, and a
change waits for the quiet interval or that check. `--until-idle` waits for one
physical attempt (120-second timeout) and then parses that input.
