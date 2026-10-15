# Behavior: speed, resources, failures

## Workload (measured 2026-10-11)

| | Count | Bytes |
|---|---|---|
| Claude Code session files (main + subagent) | ~16,000 | 19.7 GB |
| Codex rollouts | ~9,700 | 52.7 GB |
| ChatGPT exports | 4 (2,836 conversations in the latest) | 18 GB (mostly attachments) |
| Other providers | ~1,500 | ~3 GB |
| Hook spool files | 85,000 now; up to ~950,000 in a backlog | 2.6 GB now |
| **Total** | ~1,000,000 inputs at peak (with the hook backlog) | ~96 GB |

Largest single files: a 1.6 GB Claude Code session (111 messages; the bulk is
huge non-message records) and a 599 MB Codex rollout. Growth: about 3,000 new
Codex files a month during heavy agent use. Machine: 16-core desktop, NVMe,
31 GB RAM, shared with agent workloads.

## Speed targets

All targets are measured end to end through the socket, as a client sees them.
Set from what the hardware can do, not from what the old code achieved: a
plain parse-hash-write pipeline in Rust measured 75–260 MB/s per writer on
these files; writing the full `index.db` schema with search indexes measured
41 MB/s for Claude Code and 56 MB/s for Codex (`evidence/ingest-benchmark.md`). This file owns the hook budgets.

| Operation | Target |
|---|---|
| First build of the whole archive from all sources | under 30 minutes. Inputs on the operator's machine (2026-10-12): Codex 52.8 GB, Claude Code 20.3 GB, Codex's thread-history database 10.5 GB, Hermes and Gemini CLI about 2 GB, plus chat exports. Measured for the index alone: Claude Code 8.3 minutes, Codex 15.8 minutes (56 MB/s), so about 24 minutes for the two main sources and little margin once blob storage and the smaller sources are added (`evidence/ingest-benchmark.md`; deferring FTS merges was measured and made it slower; the writer waits on disk). Unmet today: the prototype's whole loop, blob storage included, measured about 21 MB/s, which over these inputs is more than an hour (next row) |
| First build of all Claude Code data (20.3 GB) | under 10 minutes (measured: 8.3 minutes at 41 MB/s for parsing and writing index.db alone, without blob storage, `evidence/ingest-benchmark.md`; the prototype's whole loop, blob storage included, ran about 21 MB/s on a real 2.4 GB subset, about 16 minutes extrapolated, so it misses this target today. One real run fell to 2.7 MB/s because every new file synced separately; with first observations batched, a 250 MB build took 17.3 s, about 15 MB/s, on a quieter disk, so part of that gain is the disk's; `evidence/real-converge.md`) |
| A new record appended to a live session appears in search | under 2 seconds, whatever the session's length |
| `polylogue` (bare status) | under 100 ms |
| A search over the whole archive | under 200 ms for the first page, warm: met for every word by recency and for words in fewer than about 200,000 texts by relevance (measured on 3 million messages; above that, relevance costs about 0.9 µs per matching text and the result says how many matched; `storage/index-db.md`, measured costs). A first search after a daemon start pays cold reads until the index is warmed (Warm state, below) |
| Opening a 10,000-message session at any position | under 300 ms |
| `polylogue hook` recording an event (async; runs twice per tool call) | under 5 ms typical (measured with a prototype of the recording path, including process start and the `fdatasync`: 4.2–4.6 ms median, 7.5–9 ms p95, 13 ms at most; without the sync 1.1 ms median; `prototypes/hook-writer/`). The hook is asynchronous, so the agent never waits on it, and its tail comes from the sync |
| `polylogue hook` on an event that may inject a briefing (synchronous) | under 20 ms when there is nothing to inject; under 150 ms when it builds a briefing |
| One process sample by the daemon | under 10 ms on this machine's load (measured 2026-10-12 with 6 harness roots, 74 processes and 982 threads: the one `/proc/*/stat` scan the sampler uses took 4.4 ms; walking each thread's `children` file instead took 5.5–6 ms), so under 0.1% of a core at one sample per 10 seconds; nothing at all with no live harness process (`prototypes/proc-sampler/`) |
| CLI startup to first output (a thin client: no database, no engine) | under 50 ms (prototype: 0.7 ms; `evidence/daemon-prototype.md`) |
| A request through the socket, warm (status, a cached plan, a page of results) | under 20 ms in the daemon plus the client's startup (prototype: 73–124 µs warm, 1.2–1.4 ms computed, on Morrow) |
| While a new `index.db` builds beside the old one | reads answer from the old file, marked not up to date; new records appear at the swap (`convergence.md`, rebuilds) |

A first build or rebuild writes its new `index.db` without syncing until the
swap (`storage/index-db.md`). A missed target is a bug to investigate, not a
fact to work around. Each target has an end-to-end benchmark
(`plan/testing.md`).

## Memory

Reasonable for a desktop, not minimal: the daemon stays under about 1 GB
while idle and caught up, warm state other than the vector set included,
and under 2 GB peak resident memory during a full
build, including while parsing a multi-gigabyte file (law 58). Files up to
64 MiB are read whole (most are); larger ones are streamed: record by record
for JSONL, through a streaming JSON reader for whole-JSON documents. Parse
workers together hold at most 512 MiB of file bytes in memory, counted from
actual sizes, not estimated (rule R49): a whole-read file counts its size; a
streamed file counts its live buffers, including a record assembled across
buffer boundaries. A worker waits for room before retaining additional bytes
that would pass the cap. Shared storage is charged once while any borrower
still holds it; distinct copies and overlapping buffers remain charged for
their actual lifetimes. Moving bytes into parser carry or a queued batch does
not release their charge: the receiving owner takes it before the sending
owner releases it, and final release follows physical disposal. Parsed-object
and queue limits account for their own retained storage. These byte counts
are admission bounds, not a measurement of resident memory; law 58 separately
measures the daemon's peak resident memory, including parser objects, carry,
queued batches and SQLite. So a 2 GB file is admitted streamed
while small files keep parsing beside it, and many large files arriving at
once cannot multiply the peak. No input is refused or
truncated for size; a single record larger than available memory is the only
case that fails, with a clear error.

Temporary storage for large reads is on disk, not in memory: connections use
`temp_store = FILE` with SQLite's temporary directory inside the archive
directory, so a large `GROUP BY`, `DISTINCT` or sort spills there instead of
growing the daemon (the Python daemon grew about 450 MiB of in-memory
temporary tables for one count during a build; `polylogue-pn62j`). The
writer is never held for one session's whole work: a huge session is written
in bounded steps (`convergence.md`, parse), including merges, ordering,
obsolete-row deletion and finalization, so other writes wait at most one
step (the same Bead observed one 400 MB rollout holding the writer 150 s).
The approximately 64 MB input step is a parsing target, not a bound on SQL
work: a step that affects many existing rows yields between bounded writer
transactions too. Intermediate commits remain readable and incomplete until
all required work commits (`query.md`, results); law 59 exercises this during
rebuild as well as first ingestion.


### Status totals

`polylogue` (bare) and `status` must answer in under 100 ms even right after
a daemon start, when nothing is cached. Counting rows does not meet that:
on the 3-million-message benchmark, counting blocks took 2 ms warm and 22 s
cold (`evidence/ingest-benchmark.md`). So the totals `status` shows
(sessions, top-level sessions, sessions per origin, messages, tool calls,
failed tool calls, unrecognized records) are counters in `index.db`'s
`meta` (keys `count.sessions`, `count.sessions_top_level`,
`count.sessions.<origin>`, `count.messages`, `count.tool_calls`,
`count.tool_calls_failed`, `count.unrecognized`), changed by the writer in
the same transaction as the rows they count, and rebuilt with the file. The
overview's recent block (the last 14 logical days) is the one query besides
them: it reads only sessions active in that window, through the index on
`last_activity_at` (`session_activity`), so its cost follows recent activity, not the archive's size. Inputs waiting are not a
counter: they change when collection writes `archive.db`, which the index
writer cannot update in the same transaction, so `status` counts them at
read from the `parse_state_status` partial index (rows not `done`), which
holds only the waiting few.
A law checks that after convergence each counter equals the count it
stands for (law 122).

### Warm state

**Warming at start.** After a start, the daemon reads the full-text
indexes' data (`text_fts_data`, `output_fts_data`) once, sequentially, as
background work at idle IO priority, so the first searches do not pay for
cold random reads (a first search took 0.9–4.6 s cold and 5–430 ms after
this read on the benchmark archive; the read itself took about 10 s for
0.9 GB on a non-copy-on-write file). Searches answer during warming, only
slower, and `status` says the daemon is warming.


The daemon keeps, in memory: the embedding vectors of the configured model,
compiled query plans, recent query results and paging anchors, and nothing
else; live state is computed per read from the `session` live columns and the
clock (`model.md`, Live state). Restarting the daemon loses nothing but speed.

- **Vectors** are held as int8 copies (one byte per dimension: about 1 GB per
  million vectors at 1,024 dimensions), loaded when semantic search is
  enabled. `near:` scans all of them and re-scores its top 200 from the f32
  rows in `embeddings.db`. The vector set is never evicted; its size is shown
  in `status` and added to the memory targets above, which apply to
  everything else.
- **Cursors** hold a page's anchor (the sort key of its last row), not a
  snapshot (`query.md`, continuation); no read transaction is held between
  pages. A result that must stay fixed is kept with `keep`.
- **Cache keys.** A result is keyed by the full read view captured once when
  the read's snapshot opens (`storage/README.md`, one view per read: index
  generation and counter, `archive.db` and embeddings counters) and carried
  unchanged through lookup, execution and insertion, so a tag, a hide, a
  setting or a new vector moves the key. A query that reads the clock
  (relative times, `today`, `HH:MM`, `running`, `waiting`, `peers_live`) is
  compiled with the clock resolved to an absolute value (the result's
  `resolved`), and cached only with that value in its key, truncated to the
  coarsest unit the query uses; live state is never cached. Plans are keyed
  by the query text plus everything that resolves it: saved-query revisions,
  the calling session (`self`, the default `cwd`), settings and the
  embeddings model.
- **Excision.** The commit of an `excision` row empties every cache and drops
  the excised texts' vectors from memory (`storage/archive-db.md`).
- A cache that would exceed its share of the memory target evicts; an answer
  too large to cache is computed and not kept. No cache holds a lock that
  writers wait for.

## Concurrency

- Heavy work (parsing, rebuilding, embedding, database snapshots) runs only
  inside the daemon, at idle CPU and IO priority, never in an interactive
  command, so it cannot freeze the desktop.
- One daemon per archive. Only the daemon opens the database files and is
  their only writer (`decisions/process.md`). A second `polylogue daemon` on
  the same archive exits naming the running one.
- Each query sees one
  consistent snapshot. Interactive requests and background work are
  scheduled separately, so a request never waits behind a build: reads use
  a small pool of read connections, writes queue for the one writer. One
  read connection is reserved for requests with a latency budget (hook
  briefings, which give the daemon [300 ms] including connect and inject
  nothing on timeout; `status`; completion; `self` lookups), and other reads
  never use it. Every read has a time limit (`storage/blobs.md`, WAL).
- When the daemon is not running, commands say so in one line and how to
  start it, and exit with code 6 (`decisions/process.md`); the hook command
  exits 0 without a briefing.
  There is no fallback read path.
- Parsing uses the cores the process may actually use (the cgroup CPU quota
  and affinity, not the host's core count); writing is one thread.
- **Every call across a boundary has a deadline** (rule R50): subprocesses
  (`git`), HTTP (Drive, the embedding provider, downloads, Sinex), the SQLite
  busy wait, the native host. A call that times out is cancelled, not left
  running while its caller reports failure. Connections the daemon accepts
  have deadlines too: a request must arrive within 10 s, a response the
  client stops reading is abandoned after 30 s, and live-tail streams send a
  keep-alive and close when a write cannot complete within 30 s, so a stalled
  client never holds a worker others need (starting values, measured in the
  first slice that serves HTTP).
- **A write is never left half done by a deadline**: a timeout before
  `COMMIT` rolls it back and is reported as failed with nothing written; if
  the commit may already have happened, the result is "indeterminate", never
  "failed". Every write request carries a client-chosen ID, so the client
  resends the same request and the daemon writes nothing twice
  (`decisions/process.md`). A queued write whose client deadline passed
  before it started is dropped, not run.
- **Completed means finished in time.** The deadline is checked when the
  work ends as well as when it starts: a read whose deadline passes while it
  is still reading, or a request whose deadline passes while it waits for
  the writer, ends as timed out (a write: the outcomes above), never as
  completed with what it had so far, and owed work stays owed. The timed-out
  error says how long the request waited for the writer and how long it
  ran, kept apart, because a slow answer from waiting behind another write
  and a slow answer from work need different fixes (`polylogue-amrni`, a
  deadline crossed during a read reported completed; `polylogue-l00qp`).
- **Elapsed times inside the daemon** (stall and stuck thresholds, round
  spacing, snapshot spacing, deadlines, sampling) use the monotonic clock,
  which does not advance during suspend; stored times use the wall clock
  (`model.md`, time).
- **The archive location** is the default (`$XDG_DATA_HOME/polylogue/`) or an
  explicit `--archive DIR`; never an environment variable or a pointer file,
  so no command can silently act on another archive. Where the archive is
  elsewhere (the Sinnix module puts it under `/realm/state/polylogue/`), the
  NixOS module installs `polylogue` wrapped with that `--archive` added, so
  every invocation, by the operator, an agent, a hook or an MCP server,
  names the configured archive and reaches its daemon's socket (whose path
  is derived from the archive, `decisions/process.md`); a command given its
  own `--archive` uses that one instead. Every `status` output
  and error names the archive directory it used. `polylogue demo` always uses
  its own directory, marks it as a demo archive when creating it, refuses to
  build into an existing directory it did not create, and never writes into
  the real archive.

### Pacing bulk work on a shared disk

Polylogue's archive shares `/realm` with Sinex, and that drive stalls under
sustained writes. Bulk work (a first build, a rebuild, re-collection after
a parser change, the derive step over many sessions) paces itself by two
signals of the drive's state, with no coordination with any other program
(joint #25): host IO pressure (`/proc/pressure/io`, "some" avg10 above 20%)
and its own recent sync and commit latency (the rate halves while recent
syncs take over 1 s). The long stalls come from the drive's write cache, not
from copy-on-write: a No_COW Postgres cluster still stalled 16 s under mixed
load, so No_COW (`storage/README.md`) speeds reads but pacing is what keeps
writes from piling into a stall. Whatever causes the load, every writer
backs off from it, so a Polylogue build and a Sinex import slow each other without knowing
of each other. `status` says "paced for IO" with the measured pressure and
latency, and names no cause, since Polylogue cannot know it. Live appends,
hook-time writes and interactive requests are never paced.

## Failures

| Situation | Behavior |
|---|---|
| Daemon killed or machine crashes mid-write | On restart the loop recomputes what is out of date and continues. No step needs a special recovery path. |
| A file is captured mid-write (truncated last line) | The complete records are parsed; the partial line waits until the file grows. Not an error. |
| A file is not the format its location promises | Stored as an input like any other; parsing refuses it with a reason (`parse_state` `refused`), shown in `status`. Never re-detected as another provider. |
| A record type the parser does not know | Counted under unrecognized; the rest of the file is parsed. |
| SQLite reports `BUSY`, `LOCKED` or `PROTOCOL` | Transient: retried with backoff, never recorded as a failed input. |
| A parser fails on an input | `parse_state` status `failed` with the error; retried when the parser version or the input changes. Other inputs are unaffected. |
| A provider location disappears | Nothing is deleted from the archive; `status` notes the location is missing, and the time it was first found missing is recorded. |
| A file or folder cannot be read (permission denied, out of file descriptors, a locked database) | Retried in the next round; never recorded as permanent; shown in `status` as unavailable with the error, never treated as missing. Errors saying the process is short of something (too many open files, out of memory) stop the collect pass and retry the same input next pass, recorded for the pass, not against each remaining input; errors about one file (vanished after listing, permission denied, an I/O error on it) are shown for that input. |
| A download (attachment, Drive file) fails | Retried in later rounds, oldest attempt first, without holding up other downloads; each attempt shows in `status`. |
| Sinex configured but unreachable, rejecting or behind | Local work continues; `status` shows the mode, how much is pending, since when, and the last error. Never reported as success. |
| A fresh archive with no inputs | `ok`, with zero counts. |
| A daemon task fails (the file watcher, the embedder, a parse worker) | Only that task stops; `status` names it with its error, and the rest keep working. `status` never reports the archive ready because the daemon process is running: readiness is each task's own state (old tracker `polylogue-kchb`). |
| Drive credentials expired | `status` says so and how to re-authorize; everything else keeps working. |
| The index is deleted | Rebuilt by the normal loop from the blob store. |
| A blob fails its BLAKE3 check, or an `archive.db` or `embeddings.db` page fails its checksum | An error naming the blob or page; reads that need it fail loudly, `status` counts it, and nothing repairs it in place. A blob whose location still holds the bytes is stored again; otherwise the fix is a restore from backup (`storage/README.md`, backup). |
| `archive.db` is lost | A loss, like `blobs/` (`storage/README.md`): it holds the inputs, their revisions and where they were found, the notes and the excisions, and the blobs mean nothing without it. Restored from backup, or, in backed mode, read back from Sinex (`decisions/sinex.md`). A daemon finding `blobs/` but no `archive.db` refuses to start a fresh archive over them and says why; starting fresh re-collects only what provider locations still hold, so every earlier revision and every note is gone. |
| Disk full | Writes stop with a clear error; the loop resumes when space returns. |

Errors are loud and specific. Nothing is mapped to a default, hidden, or
reported as success. A known gap (an unreadable input, an unknown cost) is
shown as such, never as zero or empty.

## Privacy

- Everything stays on the machine except traffic to these destinations,
  every one of them:
  - the providers' own hosts, for attachment files a session references
    (fetched from the service that already holds them; `sources/README.md`,
    downloads that fail; law 68);
  - Google Drive, when the operator authorized it;
  - an embedding provider, only when embeddings are enabled, after a cost
    estimate;
  - a Sinex that `$XDG_CONFIG_HOME/polylogue/sinex.toml` names, which
    receives archive text in backed mode (`decisions/sinex.md`);
  - the sites messages link to, only with `polylogue setting fetch-links on`
    (`sources/README.md`, linked pages).
  All but the first are off until configured, and `status` lists every
  destination in use. Models that interpret sessions are Polylogue's
  clients reading its output, not calls Polylogue makes (`analysis.md`). A
  privacy or installation document names every destination, never a subset
  (`polylogue-dktc4`: a document called embeddings the only egress while
  Sinex publication existed).
- The repository contains no private data: tests use synthetic examples;
  inferred schemas committed to the repository pass a privacy filter that
  removes values that could identify people or content.
- Excision removes content everywhere Polylogue keeps it: stored bytes, the
  index, vectors, caches, kept results and notes about it. It keeps the
  removed locations and hashes (never content) so it is never stored again,
  and reads exclude it from the moment it is recorded
  (`storage/archive-db.md`, how excision works).
- The archive directory is created 0700 and every file in it 0600, whatever
  the umask; `status` warns if either is wider.
- **Credential masking.** A derived flag marks text that looks like a
  credential (fixed, high-precision patterns: `sk-ant-…`, `AKIA…`, `ghp_…`,
  PEM private key headers and similar). It is a query field (`has:secret`)
  and a count in `status`. There is no scanner subsystem. (Decided by the
  spec authors from the rule that data leaves the machine only to
  configured services, joint decision 31; the operator may overrule.)
  - *Where it is masked.* Everywhere text can reach a model, shown as
    `[credential: AKIA…, masked]` (the pattern's kind and first four
    characters): text Polylogue assembles for a model (embeddings,
    briefings, takeovers, MCP and socket reads by an agent, anything an
    agent session is handed for a pass or a distillate), because archive
    text injected into a session goes to that session's model provider; CLI
    output and `export`, because agents run the CLI in their shells; and the
    web reader, because agents read pages in the operator's shared Chrome
    profile. `export` prints how many flagged spans its output holds. An export
    is not a backup (backup is the filesystem's job, `storage/README.md`), so
    a span masked in an export stays masked when that export is imported.
  - *Where masking is applied.* Once, when the daemon encodes a response
    (socket and HTTP alike), so no client masks again or can forget to; the
    CLI, MCP proxy and web pages receive text already masked. `--reveal` is a
    request field the daemon honours only for a request the CLI sent after
    its terminal check, and the reader's per-span reveal is its own request.
  - *Revealing.* `--reveal` unmasks one CLI command and is refused when
    stdout is not a terminal or a harness's environment is present (the job
    variables `sources/README.md` lists, such as `CLAUDECODE` or a Codex
    session variable), so masking never depends on the harness behaving.
    The terminal test carries most of the weight, since an agent's shell
    captures stdout; the variables are a second guard. The web reader
    reveals on click, fetching the span from the daemon only then, never
    placing it in the page's text before. The shared cases run in
    `polylogue-sinex-joint/prototype/masking/`.
  - *Nothing else is redacted*: no personal-data masking (email, paths,
    addresses), which would only hide the operator's own recall. The
    operator decides whether to excise.
  - *How common it is.* On the pre-reset archive (a 5% sample, counts only)
    about 0.08% of tool results matched (39 of 50,941 Codex, 36 of 38,980
    Claude Code; mostly URLs with a password, then GitHub tokens, AWS keys
    and private-key headers), and prose almost never did, so agents
    printing configuration are the source.
- Process evidence is limited: the hook writer reads only its parent's PID,
  start time and working directory, the boot ID and the job variables listed
  in `sources/README.md`; the daemon reads only descendants of harness
  processes and stores their command name, arguments and working directory,
  never their environment. Arguments go through the same `has:secret` flag as
  tool-call commands and can be excised like them.
- Text from the archive is data, never instructions: briefings and MCP
  results label every excerpt with its source and author, and inline markers
  are read only from text a human or model wrote, never from tool output.

## Sources that delete data

Claude Code deletes transcripts older than `cleanupPeriodDays` (default 30)
each time it starts. The blob store keeps what was collected before that, so
the daemon must collect promptly, and `status` warns when the effective
setting is short. A daemon stopped for longer than the retention period loses
whatever the provider deleted meanwhile.

## Untrusted input

Robustness is scoped to one operator on one machine. Three threats are in
scope because ordinary use exposes them: hostile content inside inputs
(provider files, exports and captures carry text from the web and from
tools), web pages in the operator's browser reaching the local web server
(DNS rebinding, cross-site requests; `surfaces/web.md` has the Host check and
token), and other browser extensions talking to the native host
(`allowed_origins`). Attacks that need write access to the operator's files
are out of scope.

- Regular files are collected; special files (devices, sockets, pipes) are
  skipped and counted in `status`. A symlinked directory under a provider
  location is followed (`sources/README.md`), with cycles detected by device and
  inode; a symlinked file is followed only when it points at a regular file.
  Paths enter IDs lexically, never resolved (`model.md`, Identity).
- A path built from an input's content (a ZIP member name, a capture's
  attachment name) must stay inside its root after normalization; one that
  escapes is refused with a reason (rule R53).
- ZIP members are read as streams and never extracted to disk; member names
  are addresses (`zip:<hash>/<member>`), never paths, kept byte for byte, so a
  whitespace-only or very long member name keeps its own identity and is
  never trimmed into another member's (`polylogue-5gj19`, 42.F064, F065). Reading stops only at a
  physical limit (free disk space), with a stated reason.
- JSON trees (ChatGPT `mapping`, Claude Code parent chains) are walked
  iteratively with cycle detection; real data contains cycles.
- Nesting depth never refuses a record. `serde_json`'s default recursion
  limit (128) would reject valid deeply nested JSON, so decoding runs with the
  limit disabled on a growing stack (`serde_stacker`), and Polylogue's own
  walks over decoded values are iterative; only memory can stop a record,
  with a reason (`polylogue-wdubf`).
- JSONL is parsed one line at a time with no fixed line-length cap (real
  lines exceed 50 MB); a line larger than available memory is refused with a
  reason, never a crash.
- IDs from inputs are data only: never file names, always SQL parameters.
- Terminal output escapes control characters and bidirectional overrides
  found in archive text, so a transcript cannot rewrite the terminal.
- The browser extension takes conversation data only from provider responses
  it intercepted itself, and commands only from its own popup, badge and
  background worker, never from page scripts. A script on an allowed provider
  page could still forge what the extension reads; that is accepted, since
  such a script already holds the user's session there (`sources/README.md`, browser
  captures).
- A safety setting with a missing or blank value is refused, never read as
  permissive.
- Downloads (attachments, Drive files, linked files) go only to the
  provider's own hosts, never to private, loopback or link-local addresses,
  and never follow a redirect off those hosts; a URL inside an export is data,
  not permission to fetch.
- Errors returned over MCP and HTTP name the error kind and the reference,
  never local paths or stack traces.
- Sets of IDs reach SQL as one parameter (`storage/README.md`, conventions).

## Observability

- **`polylogue status`** (and the bare `polylogue`) answers "is it working
  and up to date?" in one screen, computed when asked from the archive, the
  index (including the writer's `count.*` counters in `meta`, kept so totals
  need no full scan) and the loop's pure function; it keeps no history of its
  own and computes no scores. It shows: daemon running, last activity,
  per-provider counts and freshness, inputs waiting to be parsed, parse
  failures, new unrecognized record types, disk use, models seen with no
  price, texts the embedding provider rejected, and the running binary's
  version with the index schema version and generation, so a stale daemon is
  visible after an upgrade.
- **No ledger of operations.** What Polylogue stores about its own work is
  exactly this, each for a stated reason: the `count.*` counters and
  `last_stop` in `index.db`'s `meta` (totals without full scans; how the last
  run ended); one `client_request` row per write request in `archive.db`, so
  a resent write writes nothing twice (`storage/archive-db.md`); and
  delivery lines, which are inputs like any spool line (context for a
  session or a notice to the operator); `embed_failure` rows for retrying
  rejected texts (`storage/embeddings.md`); watch baselines as kept results, whose `ran_at` also preserves
  their cooldown across restarts (`user-data.md`, watched queries); and download-attempt observations on an attachment
  input (`storage/archive-db.md`, observation), so pending retries can be
  ordered and shown. These record facts needed for delivery or retry,
  never an operations ledger or a history of `status`. (Decided by the spec authors from the operator's rejection of stored
  process records; development-time benchmark history is not product data;
  the operator may overrule.)
- **No notifications.** Polylogue has no notification channel of its own and
  sends no health alerts; other tools (Sinnix) read the change feed and
  `status` and notify. A signal marked `interrupt` hands its notice to the
  notifier command `signals.toml` names, which is Sinnix's (`user-data.md`,
  signals). (Decided by the
  spec authors from the vision review's "stay out" list; the operator may
  overrule.)
- **One read path.** Search and every other read go through the archive;
  there is no second search over raw provider files, so a file not yet parsed
  is found once the loop parses it. (Decided by the spec authors from one
  mechanism per job; the operator may overrule.)
- **Stalls.** `convergence.md` (parse, stalls are visible) owns the rule: no
  committed work for 10 minutes while work is waiting shows in `status` with
  the step and input, and the systemd watchdog is fed only on progress.
- Every figure in `status` is one of three things: **measured** (a value),
  **measured absent** (looked, found none: `0`), or **unmeasured** (could not
  look, or not yet: `—` with the reason). Unmeasured is never shown as
  healthy, zero or 100%, and a ratio whose denominator is zero is unmeasured.
  An accepted write and an observed completion are separate facts. If
  completion cannot be confirmed because a reply or observation is unavailable,
  preserve any committed result and report completion as unmeasured with the
  reason, never success or a parse failure. Retry follows the committed target
  state (`convergence.md`),
  without duplicating admitted inputs or adding a durable attempt ledger.
  A fresh archive is `ok` for what was measured empty.
- **Stuck versus idle.** `status` tells an idle input (caught up) from a
  stuck one: unprocessed bytes or a pending failure, and no advance for 10
  minutes. Stuck inputs are listed by name.
- **Each figure has its own deadline.** A figure that misses its deadline
  shows as unmeasured ("timed out") while the rest prints within the 100 ms
  target. A figure may be reused only while its inputs' key (file times,
  counters) is unchanged, and is then shown with its age. A figure whose key
  cannot be read is unmeasured, and any earlier value is dropped. Time alone
  never makes an earlier value reusable; status over HTTP follows the same
  rule, with no time-based cache.
- **Why the daemon stopped.** The daemon records how each run ended: a signal
  (its name, written at stop) or a clean stop (`index.db`
  `meta.last_stop`). An end with no record is
  `unknown` and is never upgraded to a cause (an OOM kill, a restart) by
  closeness in time; only evidence naming this process upgrades it.
- **What logs may contain.** Log lines carry only declared fields: IDs,
  formats, sizes, counts, timings, error kinds. Message text, prompts, tool
  output, file contents and credentials are never logged; an error's
  free-text detail is cut to 300 characters; a field outside the list is
  dropped and the drop counted.
- The daemon logs with `tracing`: one line per input parsed (format, size,
  time, sessions), per error, and a summary per minute while busy, so a slow
  input can be found without re-running anything.

Evidence: the rewrite study and `evidence/` benchmarks (pipeline speed);
`polylogue-umd8q` (a manual repair froze Chrome through shared IO);
`polylogue-4az8g` (cgroup cores); `polylogue-og4bn`, `polylogue-wcw9f` (the
largest historical bug class: calls without deadlines); audit `P10-r3:K4`
(client deadlines); audit `P11-A6`, `P11-A7` (half-done writes);
`polylogue-l1qg`, `polylogue-373yt` (archive location, demo);
`polylogue-y2xby`; `polylogue-pt6mo`, `polylogue-tphz4`, `polylogue-zs70u`;
audit `P17` (resource errors); `polylogue-k4thy` (downloads);
`polylogue-mqups` (fresh archive); `polylogue-6j9c` (unpriced models);
`evidence/beads/closed-5.md`, class G, and `polylogue-oitx` (unmeasured
figures); audit `P09-A4` (stuck inputs); audit `P16-K3`, `P16-r1:K4`,
`P10-r1:A1` (figure caching); audit `P10-A2` (stop causes); audit `P01-A8`
(log fields); `polylogue-qut1`, `polylogue-9is5g`, `polylogue-l6v61`
(extension trust); `polylogue-8k10u` (blank settings); `polylogue-56jc6`,
`polylogue-h5l6i` (download hosts); `polylogue-khkup` (error text);
`evidence/questions-resolution.md` #21, #29, #30, #43, #44.
