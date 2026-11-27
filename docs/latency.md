# Latency and warm state

Every operation in the [operation table](interfaces.md#operation-table) declares a latency class. This page owns the classes and their budgets, deadlines, the daemon's states as the command line sees them, and the warm state that keeps the budgets.

## Latency classes

Each table row declares a latency class, and the class applies on every surface (CLI, MCP, HTTP, web). Budgets are end to end (process start to the last byte of the first screen, through the real binary and socket), with a warm daemon and a real-shaped archive under typical load, at p95. The deadline each request carries comes from its class, in the `command-surface` request envelope. What each class reads is computed before the keystroke ([Warm state](#warm-state)).

| Class | Members | Budget (p95) | Deadline sent |
|---|---|---|---|
| **local** | `help`, `--help`, `--version`, grammar refusals, `--explain` of a line's grammar, `install SHELL --dry-run` | ≤ 15 ms | none (no socket) |
| **completion** | Tab in any shell, MCP `completion/complete`, the web query box | ≤ 50 ms per keystroke, process start and one socket round trip included | the CLI waits 250 ms, then offers the grammar alone; 30 ms for the count preview, omitted when late |
| **picker** | first candidates, a keystroke, the preview | 50 ms, 16 ms, 100 ms | the read class for candidates; 100 ms for the preview |
| **status** | bare `polylogue`, `status`, `status --line` | ≤ 100 ms; `status --line` ≤ 30 ms, because it runs in status bars | 80 ms |
| **read** | the first screen of a one-row read, the first page of an indexed listing (`origin:`, `repo:`, `since:`, `tag:`, a shipped saved query), a page turn, `resume` | ≤ 100 ms | 250 ms |
| **write** | one annotation acknowledged as committed | ≤ 100 ms | none ([below](#writes-and-deadlines)) |
| **search** | the first page of a word search, `similar:` with a stored vector | ≤ 250 ms | 2,000 ms |
| **brief** | every form of `brief` | ≤ 300 ms warm, ≤ 1 s cold | 2,000 ms; the session-start hook narrows it to one shared 1.5 s ([agent-context.md](agent-context.md)) |
| **aggregate** | `group by`, `\| measure`, `\| agg`, facet counts, the count `--dry-run` reports | ≤ 400 ms | 400 ms |

A missed budget is a defect in the code, not a reason to relax the number. Budgets are measured as paired benchmark runs ([testing.md](testing.md#benchmarks)); no test asserts elapsed time.

An MCP `_meta.deadline_ms` absolute bound travels separately from the unchanged shared request as the transport's own `parent_deadline_ms`; the daemon intersects it with the operation lifetime. An HTTP `x-polylogue-deadline-ms` header bounds the whole request; a stream ends at the earlier of that bound and its operation's deadline, and without the header a `null` deadline has no limit. A request already past its bound starts no new write; a write the writer has accepted keeps its effects and acknowledgment.

The read-class `internal advisory` operation has a stricter preparation cap of 20 ms, intersected with its caller's bound on every surface. Hooks send a 20 ms request and emit nothing on failure; `--deadline none` does not lift this optional-advisory cap.

## Writes and deadlines

Pure writes, including imports, await their committed result without a read deadline. Mixed operations keep deadlines for their read preparation and end that phase when they hand the mutation to the writer: a preparation that has run out refuses the handoff, and a handed-over write replies with its committed result however long the commit takes, never a timeout. Classification follows the accepted arguments, so showing a setting stays bounded while writing it keeps its result. In a line each step has its own class and shares the original read deadline. After a write step, a reader waits for the index to take in that write before opening its snapshot. If the wait runs out, the line reports the read's timeout alongside the completed write.

**A listing's count** runs after its page, on the request's own reader, until the deadline less the time the reply needs (a fifth of what is left, at most a second); a count that runs out answers the page with `count deferred` and the command that counts without a deadline.

**Deadlines are interactive defaults.** A read whose row declares no latency class, called with the default deadline, stops after 30 seconds; `--deadline none` lifts it. The CLI's socket timeout is the deadline plus 50 ms, so a hung daemon never hangs a shell. Any caller lifts a deadline (`--deadline MS`, `--deadline none`; `deadline: null` over MCP and HTTP); cancellation is in [query.md](query.md#deadlines).

A stopped read names the stage that ran out. A page or similarity stage suggests a shorter query only when the query has no top-level time bound on its unit's main time field. Other stages, and queries already bounded in time, say "rerun with `--deadline none`, or narrow to one session". Recall and briefs give that same hint without a query-shaped suggestion, including when an inner page or similarity stage stopped.

## The daemon's states

| Daemon | Command line |
|---|---|
| busy with a first build or re-parse | interactive replies keep their budgets on the reserved read connection; replies over data not yet indexed are `degraded` (exit 5) and name what is missing in their notes on standard error, each with its count and origin. Inputs that can be read no further (an origin with no parser, a file of no bytes, a failed or refused parse) do not make a reply degraded ([interfaces.md](interfaces.md#exit-codes)) |
| starting (lock held, socket not bound) | waits up to the class deadline, printing "polylogue: waiting for the daemon to finish opening the archive" once on standard error, then exit 7 if the deadline passes |
| down | the local class works, and completion offers the grammar alone; everything else exits 6 with one line naming the archive and `polylogue daemon`. Nothing starts a daemon implicitly |
| hung | the socket timeout ends the command, exit 7 |

Because one daemon assembly answers every transport, the same line gives identical rows, totals and refusals through CLI `--json`, the socket, HTTP and MCP.

## Warm state

The daemon keeps derived state warm so the classes hold. Each item is keyed by what it depends on, so no reply from before a write is served after it; none is a second archive, and each is rebuilt from the archive after a restart: until the status snapshot is read, `status` answers that its figures are not read yet, and `warm.scope_index_loaded` says whether the scope index is built ([operations.md#status](operations.md#status)). An LRU may evict items; it never limits replies.

| Warm item | Serves | Keyed by | Refreshed |
|---|---|---|---|
| sorted session index with titles, last activity, origin, repository, first prompt, shortest unique prefixes | short IDs, prefixes, `latest`, the picker | write position | batch delta |
| status snapshot | bare `polylogue`, `status`, `status --line` | write position and live-state change | after each commit (each batch and each hook event); a reply reads the latest snapshot and never waits for the next |
| compiled plans, the shipped saved queries' precompiled | every query | normal form, definitions and options | on definition change |
| first pages, next pages and counts | repeated listings, page turns, footers, the count preview, `--dry-run` counts | question hash and read position | evicted on write; a live page cell's result stays until the change feed touches one of its sources ([pages.md](pages.md#queries-on-pages)) |
| saved-query names and kept-result shapes | compiling | write position | on user write |
| loaded vectors (int8 in memory, f32 on disk) | `similar:` | model and snapshot revision | when a request needs that revision |
| query vectors | `similar:TEXT` | model and NFC text | on first use |
| search vocabulary | fuzzy matching in the brief of a search | write position | batch delta |
| scope index and session summaries | `brief` | write position and live-state change | batch delta and each hook event |
| rendered overviews of recent sessions | the picker preview | session, read position, width | evicted on write to that session |

A request needing vectors extends the newest held set to its snapshot revision itself, reading appended rows or loading the whole set after a replacement or metadata change. Loading and quantization do not hold the cache mutex. Requests for the same model and revision share that load and its result or failure; each waiting request observes its own deadline and cancellation, including daemon stop. The cache holds the two newest sets; older snapshots load their own without replacing them. Clearing warm state disconnects loads already in progress from the cache, while their requesting callers still receive their results. Query vectors keep the 256 most recently used texts.
