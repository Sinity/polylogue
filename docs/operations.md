# Operations

Running Polylogue: the daemon's lifetime, its idle and steady-state work, the halt state, orientation and `status`, the operation commands, notices, verification on demand, several archives and machines, configuration, deployment, installing into harnesses, upgrades, and backup and restore.

Neighbouring facts live elsewhere: the durability order and pipeline are [architecture.md#durability](architecture.md#durability) and [architecture.md#pipeline](architecture.md#pipeline); the CLI's grammar, latency classes, completion and exit codes are [interfaces.md](interfaces.md); the schema tools are [archive.md#format-drift](archive.md#format-drift); primary mode and the cutover are [primary-mode.md](primary-mode.md); the signal rule schema and its evaluator are common's [signals](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/signals/README.md); the web views of status, notices and settings are [web.md](web.md); setting keys will be generated into `reference/settings.md`.

## The daemon

Polylogue runs as one resident daemon per archive. The daemon is the only process that opens the archive; every client (CLI, MCP, hooks, the web UI, the browser host) is a thin client of its socket, for reads as well as writes. It exists so that Polylogue answers at interactive speed and so that what should be happening can be seen happening. (ruling 2026-09-14, cc:652ae14f-a036-496b-abfa-42e7574c9ec9:b8a68d64-772f-4d8c-8e95-435ce601d245).

**Start.** `polylogue daemon` runs in the foreground under a service manager (a systemd user unit on Linux). In order, it:
1. raises the open-file limit;
2. takes the archive lock, an exclusive `flock` on `<archive>/lock` opened through the resolved directory path, and writes its pid into it; a second daemon on the same archive, by any path, exits 1 naming the holder's pid;
3. applies the numbered migrations of its stores in place ([archive.md#evolution](archive.md#evolution)), or refuses with exit 78 when the archive is newer than the binary or has no migration from its version;
4. opens the segment log at its last committed extent and the index at its saved positions;
5. binds the socket, serves requests, then starts its work.

No step deletes the index or any store.

**Readiness.** With `Type=notify` the daemon signals readiness once the socket answers a `ping`, before catching up. The connection's thread answers `ping` without a reader or any cache, so a vector load or a full reader pool never delays readiness or the watchdog. Catching up is shown as work in `status`, never as unavailability.

**Stop.** On `SIGTERM` or `SIGINT` the daemon stops accepting connections and answers queued requests with `shutting_down`; lets the commit in progress finish or rolls it back; stops its pipeline at a writer-group boundary; checkpoints its SQLite write-ahead logs; records how the run ended; and releases the lock. A second signal ends the process at once, which is safe because the next open resumes every state a crash can leave. Target: a stop within 5 s, and nothing done at the next start that a clean stop would not also need.

**How each run ended:** `clean` (with the signal), `refused` (exit 78, with the reason), `failed` (with the error kind), or `unknown` (no record: killed or crashed). `status` shows it. An `unknown` end is never given a cause by closeness in time to another event.

**Liveness.** With `WatchdogSec`, the daemon feeds the watchdog after each `ping` its own socket answers, so a process that no longer answers its own socket is restarted by the service manager. Slow or stuck work is not a liveness failure: it is reported as stalled, never restarted away, because a restart would hide the defect that caused it.

**Restart** resumes from committed positions; it does not rescan the archive, replay tool-database captures or re-read history. Target: socket ready within 2 s of start.

**Batch mode.** `polylogue daemon --until-idle [--json]` runs the same pipeline and exits when no unit has work, printing a summary and exiting 0, or 1 when a pass or import met an error: the passes and imports it ran and their work added together (inputs seen and changed, bytes read and stored, messages written, sessions derived, unreadable roots and errors). "No work" means a pass ended with no capture continuing, no request queued and no change waiting from the watcher. Tests, benchmarks and the restore drill use it; it is not a second engine.

**Without sources.** `polylogue daemon --no-sources` reads none of its sources: no root is scanned or watched, no spool or tool database is read, no collector runs and no repository lookup is written, so no input, observation or coverage is recorded from the machine. It parses and derives what the archive already stores (an index behind its archive catches up, a reading-version change re-parses) and serves every request, `import` included. A file that has gone from its location is not observed as gone, because nothing looked. With `--until-idle` it processes what is stored and exits. An archive that receives another archive's contents (`import`) is started this way, so that none of its locations is watched before the import, and the restore drill uses it to check a restored copy without touching the machine's live files.

**Exits:** 0 clean stop; 1 failure or lock held; 78 refused archive. The service sets `RestartPreventExitStatus=78`, and the service manager's failure notification reports it, since no daemon is running to send a notice.

Requests that need the archive, schema inference, verification and import included, go to the daemon at idle priority.

## Idle and steady-state work

An idle daemon reads and writes about as much as actually changes:
- Discovery is event-scoped. A full scan of the declared locations runs only at start and after a watcher error or overflow, never on a schedule.
- Busy tool databases are captured on the tool's own change signal (its write-ahead log, its modification time) or a minimum interval, never in a zero-delay loop.
- Idle, the daemon truncates its SQLite write-ahead logs; `journal_size_limit` is set.
- The accept loop blocks; nothing polls.

**Targets:** with nothing changing, under 1 MB/h read, no writes and under one wakeup per second; with live agents writing, read bytes at most about twice the bytes appended to watched files; memory idle and caught up under 1 GB excluding vectors (a target, not a cap). `status` shows the daemon's own I/O over the last minute and hour and the time and reason of the last full scan.

**Interactive work never waits behind bulk work.** Every request's work runs in a bounded executor: one place per configured worker for ordinary requests, one place reserved for the interactive answers (`status`, completion and `brief`) and one background place for idle-priority work (schema inference, verification, imports). Queued work starts in the order of its deadline; work without one ranks as if it had the default read deadline, so it still starts in its turn. Work that waits for something other than its own computation (the next commit, for `follow`) gives its place to queued work for the wait. The read pool has one connection per ordinary and background place and one more that only the interactive answers may take. An authored write waits at most one writer group. A read whose client has gone stops at its next step: one watcher thread notices every client's hang-up and cancels that client's reads, without cancelling work another reader needs, and reads of the same session share one load of its index instead of each loading it.

**Sending never holds a reader.** A finite answer is computed, released from its reader and then sent. A stream's producer writes into a buffer that keeps up to 1 MiB in memory and the rest in a temporary file in the staging directory, so its read transaction lasts as long as producing takes, never as long as a slow client takes to read; a client that stops reading holds only its connection and the buffered bytes, and the write-ahead logs still checkpoint. Every send follows one rule: it waits for the client until the request's deadline (a send past it takes only what the socket accepts at once) or until the client hangs up or the daemon stops, and the access scope is checked before each piece sent. Each connection has one thread, which reads its requests and sends their answers.

**The halt state.** Every declared unit of work (a source location, a parser module, a derivation, the embedder, the sampler, the signal evaluator, the primary-mode sender) has exactly one halt state, read from one place:
- **unschedulable:** a halted unit is left out when work is selected, so nothing downstream of it takes the writer, forms a group or spends a budget;
- **reported:** the halt is durable, with its typed reason and the input or step that produced it; `status` names every halted unit, and a daemon with one is `degraded`.

A halt survives a restart and clears only when what it depends on changes: a new revision of the input, a new reading version of the parser, a configuration change, or a new binary. No command clears a halt.

**Stalled** is distinct from halted and from busy: work pending and no committed step for 10 minutes. `status` keeps idle, busy, stalled and halted apart.

## Status

**Bare `polylogue` is orientation.** In the `status` latency class it prints:
1. a verdict line (`healthy`, `degraded` or `unknown`), the first reason when not healthy, and notices since the operator last looked;
2. **waiting on you:** sessions in live state `waiting`, oldest first ([agents.md#live-sessions](agents.md#live-sessions));
3. **running now:** the count of live sessions and the first rows of `query:running`, grouped by repository;
4. **the quota window:** the current subscription window's use against its limit, per plan ([measures.md#cost-and-usage](measures.md#cost-and-usage)), or `not_measured` with the reason;
5. **recent here:** the latest logical sessions in the current repository, a default page of five;
6. **next commands** for what is shown.

**`status` is the full health view.** Orientation and `status` share one snapshot. `status` shows, in order:
1. **Daemon:** pid, uptime, binary version (crate version and commit), reading versions in force, the previous run's end, warming, I/O rates, and pacing under I/O pressure with the pressure measured.
2. **Up to date:** `up to date`, or what is behind: inputs waiting, chains behind their reading version, derived products pending, positions behind; what holds the writer and its last commit's age; idle, busy, stalled or halted per unit. A rate and an estimate are shown only while the rate is steady (within about 20% over the last few minutes), otherwise "rate varying".
3. **Archive:** the path; bytes of the segment log, each database and the embeddings store, logical and on disk; the mode (`standalone` or `primary`, with the primary section of [primary-mode.md](primary-mode.md)).
4. **Sources:** one line per origin and location with input and session counts, the last new input, and the location's state: present, missing (since when), unreadable (the error), files no declared pattern matched (counted by extension), and for Claude Code a warning when its transcript cleanup period is short.
5. **Problems**, each by name: chains that failed to parse, with path and reason; halted and stalled units; record types and field paths new since the previous reading version; models with no price; texts the embedder rejected; stored bytes whose hash did not match; archive permissions wider than 0700/0600.
6. **Notices:** the count in the last day and the latest three.
7. **Rules:** every enabled signal rule with its author; rules written by anyone but the operator since he last looked are marked.
8. **Settings changed by anyone but the operator** since he last looked.
9. **Harnesses:** for each installed harness, whether its hooks and MCP entry are installed, and the last hook line and MCP call seen.

**Figures.** `status --json` returns the same content as typed figures `{name, value, state, reason, age}` in the shared figure-and-verdict format of common's [command-surface](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/command-surface/README.md). Polylogue's figure states are `measured`, `none_found`, `not_measured` (with the reason) and `unavailable`.
- Every figure has its own deadline; one that misses it is `not_measured: timed out`, and the rest still answers within the class budget.
- A figure is reused only while its key (file times, counters, positions) is unchanged, and is then shown with its age; time alone never makes an old value reusable.
- Totals come from counters the writer keeps in the transaction of the rows they count, never from a recount.
- Whether the index has caught up comes the same way: the writer keeps each stored chain's parse state (no checkpoint, current, no parser, partial, behind) with the change that moves it, counted by origin and reading version, and a read applies the reading contracts to those counts. No read or commit walks every chain; a writer that stopped without closing recounts once at its next start.
- An unmeasured figure is never shown as zero or healthy; a ratio with a zero denominator is `not_measured`.

**Exit code.** `status`, with or without `--line` or `--json`, exits with its answer's code: 0 for a current answer, the degraded code when the answer is degraded (an index behind or a figure not measured), as every other command does ([interfaces.md](interfaces.md#exit-codes)).

**Edge cases.** A fresh archive is `healthy`, with zero counts and the locations watched. An archive whose files cannot be opened is `unknown`. With the daemon down, the CLI prints one line naming the service to start and exits 6. A starting daemon makes the client wait within its deadline and say so. Status over HTTP and MCP carries no local paths in errors.

## Operation commands

The commands this area owns, each with the reason it is a command rather than a query plus a verb:

| Command | Why it is a command |
|---|---|
| bare `polylogue` | orientation |
| `status [--json]` | a snapshot of the daemon, not of archive rows |
| `daemon [--until-idle] [--no-sources] [--json] [--archive DIR]` | runs the service |
| `import PATH…` | brings files on the daemon's host into the archive ([capture.md#import](capture.md#import)); a relative path resolves against the caller's working directory |
| `verify [--manifest FILE]` | an on-demand check of stored bytes |
| `setting [NAME [VALUE]]` | writes an authored `setting` record |
| `schema infer\|diff\|unread` | describes stored inputs' shapes ([archive.md#format-drift](archive.md#format-drift)) |
| `sinex cutover [--dry-run]` | the one-way move to primary mode ([primary-mode.md#cutover](primary-mode.md#cutover)) |

Notices, the running list and the live list are shipped saved queries (`query:notices`, `query:running`), not commands. No command checks for a terminal or harness variables to decide whether it may run, and every write records its computed author ([assertions.md#authorship](assertions.md#authorship)). Every command shows stored text as it is: Polylogue censors no output.

There is no repair, reconcile, reindex or reset command and no scheduled check: the stores are kept consistent at every commit, and a crash leaves only states the ordinary open path resumes ([architecture.md#durability](architecture.md#durability)).

## Notices

Polylogue's signals use the common [signals](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/signals/README.md) rule schema, evaluator, interrupt budget and desktop notifications. (ruling 2026-10-28, cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:3ae3257e-e70a-433a-be97-d1222670c284).

What is Polylogue's:
- **Storage.** A rule is a `signal` assertion ([assertions.md#kinds](assertions.md#kinds)); each firing is a delivery naming the rule and revision, the triggering occurrences with their observation, the time, the channel, and the route decision with its holds and the attention it was decided under ([agents.md#deliveries](agents.md#deliveries)). Deliveries survive an index rebuild and reach Sinex in primary mode.
- **Rules.** `query` is a `query:<hash>` or saved name over Polylogue's catalog; `target` and `command` name entries of the deployment's signal configuration ([Configuration](#configuration)), so a rule can run only what the deployment configured; `scope` routes notices to the session-start brief of a repository or session. Every author may write, revise, enable and withdraw rules; the author is computed.
- **A watch is a rule** with `fires: on_change` and `deliver: notice` ([query.md#watches](query.md#watches)); there is no second watch mechanism.
- **Occurrences** follow lineage: for rows about sessions, the occurrence is the logical session ([sessions.md#logical-sessions](sessions.md#logical-sessions)), so a waiting episode carried across a fork, resume or continuation fires once. Repeats of the same note (same kind, target and value) are one occurrence.
- **Injected text** (`injected_context`) and deliveries are excluded from every rule's input.
- **Imports only:** a rule whose query reads only imported exports can fire no earlier than the next import, and writing it says so.
- **Incomplete:** a rule is `incomplete` while `status` does not say "up to date" for the data it reads.
- **Outlets:** `status` (count and latest), the next session-start brief whose scope the notice concerns, a desktop notification for interrupting notices, the change feed, and `query:notices`. Notices addressed to sessions reach them at their next context-accepting hook event ([agents.md#notices-to-sessions](agents.md#notices-to-sessions)).
- **One deliverer per host.** Standalone, Polylogue's daemon shows its own desktop notifications. In primary mode its evaluator sends its reports (firings, cleared conditions, rule revisions and progress) to Sinex, whose deliverer routes them under the one interrupt budget and quiet hours; Polylogue notifies directly only for `Sinex blocked`.
- **Value and action deliveries** run only targets and commands the deployment configured. No rule starts or resumes an agent session ([agents.md](agents.md#pings)).

**What interrupts.** A notice interrupts only when the operator would want to stop what he is doing for it; everything else is passive (status, briefs, the feed). "Agent waiting on you" is passive unless the operator makes it interrupting.

**Shipped rules.** Shipped rules are data in the code, listed and read like authored rules, with no author record; they count as the operator's for the interrupt budget. An authored rule with the same name replaces the shipped one, withdrawing it brings the shipped one back, and disabling is an authored revision with `enabled: false`.

| Rule | Condition | Fires | Default channel |
|---|---|---|---|
| daemon stalled | a unit has work pending and no committed step | `on_new`, `for: 10m` | interrupting |
| source unreadable | a declared location cannot be read | `on_new`, `for: 1h` | interrupting |
| capture stalled | a watched input has appended bytes not committed | `on_new`, `for: 10m` | interrupting |
| Sinex blocked | primary mode: Sinex unreachable or refusing this producer | `on_new`, `for: 10m` | interrupting |
| agent waiting on you | a session in live state `waiting` | `on_new`, `for: 5m` | passive |
| unit halted | a unit entered its halt state | `on_new` | passive |
| stored bytes mismatch | a read or `verify` found a hash mismatch | `on_new` | passive |
| new parse failures | chains failing that did not fail at the previous reading version | `on_change` (`membership`) | passive |
| Claude Code cleanup short | Claude Code's transcript cleanup period would delete transcripts before collection | `on_new` | passive |
| brief lost | a prepared brief ended `not_sent` | `on_new` | passive, and to the session at its next hook |

Sinex's own notices are Sinex's and reach the operator through Sinex's deliverer; Polylogue does not repeat them. A refused archive (exit 78) and a dead process are reported by the service manager's failure notification.

## Verification

- **At read time,** every read of stored bytes checks the frame's hash; a mismatch fails that read with a named error (the segment, the extent and the citing input) and records a problem, which fires `stored bytes mismatch`.
- **On demand,** `polylogue verify` asks the daemon for a full check at idle priority, with progress in `status`: every segment frame against its hash (the one scan in common's [hashed-blobs](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/hashed-blobs/README.md)), every blob's catalog sizes against its stored descriptor, every extent cited by a row present in its blob, every chain's extents contiguous from its start and covering its longest revision, the kept counters (`input`, `revision`, stored bytes) equal to the rows they count, and page checksums and `quick_check` of each SQLite database. Its answer counts inputs, revisions, extents, observations and blobs, inputs, revisions and bytes per input format, observations per class and revisions per way they were produced, and names every problem; with problems the answer is `degraded` (exit 5). It stores only its result (when it ran, what it covered, the named problems) and changes no row.
- **Manifest.** `verify --manifest FILE` also reads every revision's bytes through its chain (each chain once) and writes one line per revision, `input<TAB>number<TAB>length<TAB>blake3`, ordered by input and then number, to a new file on the daemon's host; a relative path is the caller's. The file appears complete or not at all: when a revision cannot be read the check names it and writes no manifest. An existing file is never replaced. A manifest computed independently from another copy of the same inputs (an older archive, a backup) compares byte for byte.
- **Recovery** from a mismatch is a restore from backup ([Backup and restore](#backup-and-restore)).
- Verification has no schedule; a drill or periodic check is something someone invokes.

## Several archives and machines

**Several archives on one machine.** An archive is a directory. `--archive DIR` selects it on every command, so a client and the daemon it reaches always agree. Each archive has its own daemon, lock, socket (under `$XDG_RUNTIME_DIR/polylogue/<hash of the resolved path>/`) and unit instance. A command never acts on an archive it was not pointed at, and every `status` and error names the archive path. `demo tour DIR` builds its own archive and refuses an existing directory it did not create.

**Several machines.** Every observation records the machine ID and host name, so paths from two machines never collide, and sessions show their host where an archive holds more than one. Archives on different machines combine either through primary mode, each host sending to one Sinex as its own producer ([primary-mode.md](primary-mode.md)), or standalone, by writing a bundle with `polylogue QUERY read --to DIR` on one machine ([reading.md#writing-out](reading.md#writing-out)) and reading it with `polylogue import DIR` on the other. An archive directory is never synced live between machines.

## Configuration

Configuration exists only where the operator's choice is genuinely open. (ruling 2026-07-29, cc-queued:d6f4b0c3-840b-47b6-bd57-5f1bd6c9c7d4@2026-07-29T09:22:39).

Deployment configuration is loaded once by `main`, from the config file, then flags, then environment where meaningful, into one typed value ([architecture.md#configuration-and-clocks](architecture.md#configuration-and-clocks)). It is set through the Nix module options, the files under `$XDG_CONFIG_HOME/polylogue/`, or environment variables. (ruling 2026-10-28, cc:ee52794c-8a9e-43b6-9fbc-9749c7e78e8f:3ae3257e-e70a-433a-be97-d1222670c284).

| What | Where | Why it is a choice |
|---|---|---|
| mode (`standalone` or `primary`), the Sinex endpoint, producer identity and credential file | deployment: `sinex.toml` | whether Polylogue runs on Sinex |
| archive location | `$XDG_DATA_HOME/polylogue` by default, or `--archive DIR` | where the data is kept |
| listen address and authentication of the web UI and HTTP API | deployment: `polylogue.toml`, `POLYLOGUE_LISTEN`, `POLYLOGUE_AUTH_TOKEN_FILE`; default loopback on a fixed default port with a peer-uid check; a non-loopback address needs a token file and refuses to start without one | where the web UI is reached from |
| the logical day (zone and start) | standalone: deployment; in primary mode, read from Sinex, with no value of Polylogue's own | the operator's day; its meaning is common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md) `DaySetting` |
| the away command, value targets and action commands | deployment: `signals.toml` | which programs may be run or written to |
| enabled variants | authored `setting` | which translations or summaries readers show ([assertions.md#variants](assertions.md#variants)) |
| quiet hours and the interrupt budget `signal.interrupts_per_hour` | standalone: authored `setting`; in primary mode Sinex's, applied by Sinex's deliverer | the operator's attention |
| subscription plans (`plan.<provider>`) | authored `setting` | facts only the operator knows |

`polylogue.toml` holds `archive`, `[http]` (`listen`, `auth_token_file`) and, standalone, `[day]` (`zone`, `start`). `--config` chooses the file, and its directory also holds `sinex.toml` and `signals.toml`; without the flag all three live under `$XDG_CONFIG_HOME/polylogue/`. `--archive`, `--listen` and `--auth-token-file` override the file, and `POLYLOGUE_LISTEN` and `POLYLOGUE_AUTH_TOKEN_FILE` override the flags. Relative paths in a file resolve against that file; relative flags against the working directory. Standalone, `[day]` may name either value or neither: the zone defaults to the host's named zone (`TZ`, then `/etc/localtime`) and the start to `04:00:00`, so Polylogue and Sinex on one host share one logical day. An invalid zone, or a host zone without an IANA name, refuses start-up. In primary mode a local `[day]` is refused, because the day is Sinex's.

`signals.toml` holds `[notifications]` (`deadline_ms`, and an optional `away_command = { program = "...", arguments = ["..."] }` that receives each notice the route function sends away, as JSON on standard input), `[value_targets.NAME]` (`path`) and `[action_commands.NAME]` (`program`, `arguments`). Programs and arguments are separate values, never a shell string, and relative paths resolve against the file.

**Authored settings** are insert-only assertions with author, `via` and time, from a declared catalog with a type, default and scope per key; unknown keys and blank values are refused. `polylogue setting` lists every setting with its value, kind, scope, default, who set it and when; `setting KEY` reads one and `setting KEY VALUE` writes one. A host-scoped key applies to the machine that wrote it unless `--machine NAME` names another host; an archive-scoped key refuses `--machine`. In primary mode the keys Sinex owns (quiet hours, the interrupt budget) show Sinex as their owner and have no local value. Any author may change a setting; `status` lists changes by anyone but the operator until he next looks ([assertions.md](assertions.md#authorship)).

**Not configurable,** because a dominant option exists: provider locations (fixed per origin, honouring a tool's own documented home variable such as `CLAUDE_CONFIG_DIR` or `CODEX_HOME`; [capture.md#locations](capture.md#locations)); whether the session-start brief is injected (removing the hooks is the off switch); brief sizes and budgets; worker counts and memory targets (derived from the cgroup CPU quota and affinity); pacing; deadlines (defaults any caller lifts).

**Environment.** No environment variable selects the archive, other than the XDG base directories, `HOME` and the tools' own home variables. The binary reads only those, `PAGER`, `NO_COLOR`, `TERM`, `COMPLETE`, the harness variables read for caller identity, `POLYLOGUE_LISTEN` and `POLYLOGUE_AUTH_TOKEN_FILE`, `DBUS_SESSION_BUS_ADDRESS` for desktop notifications, and `VOYAGE_API_KEY`, the optional credential for paid embeddings. All are read once at start-up.

Terminal styling follows whether stdout and stderr are terminals, each on its own. Pipes are plain, and `NO_COLOR` or `TERM=dumb` turns styling off; JSON and protocol output are always plain. Archived control characters are escaped before any styling is added.

**Broken configuration never falls back.** A `sinex.toml` that does not parse or names an unknown mode does not switch to standalone: `status` shows the error with the file's path, the verdict is `degraded`, and writes wait.

## Deployment

- **The flake** exports `packages.polylogue`, `nixosModules.polylogue` and `homeModules.polylogue`. The modules' options are typed from the same declaration as the configuration above, through one shared settings library; a deployment only sets options. Any deployment works the same way; none is a dependency of Polylogue's behaviour.
- **The package** can be built with the Sinex adapter, a Cargo feature outside the default build, so the standalone build has no NATS dependency. The binary reports its version and commit (`--version`, `status`).
- **The Home Manager module installs:** one wrapper used as the CLI, the daemon command and the browser host, passing `--archive` explicitly and setting the XDG directories; shell completion for every installed name; `polylogue`, `plg` and `plog` as shell names of the same wrapper; the browser extension and its native-messaging manifest; hooks and MCP registration ([Installing into harnesses](#installing-into-harnesses)); `sinex.toml` and `signals.toml`; and one user service per archive with `Type=notify`, `WatchdogSec`, `Restart=on-failure`, `RestartPreventExitStatus=78`, a failure notification, and a resource class that yields I/O and CPU to interactive work with no memory cap.
- **The daemon restarts only when its binary changes;** activating unrelated changes leaves it running.
- **Placement of stores on the host** (separate volumes, snapshot schedules, copy-on-write settings) is deployment configuration. Whatever the placement, the archive stays covered by a backup.

## Installing into harnesses

Polylogue owns one **hook declaration**: per harness, each event with its mode (recording or context), timeout, whether its output is injected, and the harness's measured inline cap. Three consumers read it and none restates it:

- `polylogue hooks print --harness H` emits the harness's settings fragment (JSON; YAML for Hermes), so a host configuration can generate its managed settings from it.
- `polylogue install-agent HARNESS [--remove] [--instructions]` installs, for one harness, the hook entries, the MCP server entry ([agents.md#mcp-server](agents.md#mcp-server)), the generated skill ([agents.md#agent-manual](agents.md#agent-manual)) and, with `--instructions`, one marked line in the harness's global instructions file. It prints every file and entry it will change, writes a backup beside each file (`settings.json.polylogue-backup`), marks its entries as Polylogue's, keeps foreign entries byte for byte, and refuses symlinked (managed) files with a pointer to `hooks print`. `--remove` deletes exactly Polylogue's entries, with any event or `hooks` object only they filled; when the file is still exactly what the install wrote, it gets the backup's bytes back and the backup is removed. Removing the hooks is how a person turns briefs off.
- `polylogue hooks status` reads the spool and the archive: per harness and event, declared, seen in the last 7 days, last seen, and the share of sessions with a `SessionStart` line. An event counts as missing only when an opportunity for it existed.

**Modes.** Recording entries run asynchronously where the harness supports it; context entries (session start, prompt submission, advisories, permission requests) are synchronous because the harness injects or acts on their output. What a recording entry stores is [capture.md#hooks](capture.md#hooks).

`install-agent` installs agent integration only; the browser extension's native-messaging host is installed by `polylogue browser-host install` ([extension.md](extension.md)).

## Upgrades

An upgrade is a version change in `status` and, when a parser changed, a while in which some chains are re-read.

| Change in the release | What happens | Reads meanwhile |
|---|---|---|
| no storage change | restart, resume from committed positions | down for the restart |
| a parser module's reading version | chains read with the old version are re-parsed one at a time, live inputs first, then newest first | every read answers; results touching a chain not yet re-parsed carry `up_to_date: false`; `status` shows progress |
| a schema change of any store | a numbered in-place migration at open | down for the migration |
| an archive newer than the binary, or with no migration from its version | refused, exit 78, not restarted | down |

**Spools across a rollback.** Hooks and the browser host keep writing their spools while the daemon is replaced, and the spools are never deleted, so the build a deployment rolls back to reads what the newer build wrote. A spool line's shape is therefore part of the deployed format: the hook's line shapes are pinned by a fixture that the previous build's reader is checked against, and a change to them is checked against the deployed reader before it ships.

No code path deletes the index. A full rebuild is a rare deliberate act: someone moves `index.db` aside and starts the daemon, which rebuilds from stored bytes and answers reads with `index building` and progress until it can answer. Moving a whole archive aside and starting fresh is likewise a deliberate act, never something code does to get past a change.

## Backup and restore

- **Backup is the deployment's filesystem backup.** Polylogue has no backup command and keeps its stores consistent at every commit. A correct backup takes the segment log, the databases and the embeddings store together; `index.db` may be left out, since it is rebuilt from stored bytes.
  - An atomic snapshot of the archive directory is consistent.
  - A file copy needs either a stopped daemon, or each SQLite database copied through SQLite's backup API followed by the segment log: committed extents never move and no row cites an extent before it commits, so databases first and segments after is consistent.
- **Polylogue provides** the paths and sizes in `status`, and `verify` to check a restored copy.
- **Restore** is putting the files back and starting the daemon. Migrations apply as on any open; an uncommitted segment tail is truncated; an index behind its positions catches up; a daemon that finds a segment log but no `archive.db` refuses to start a fresh archive over it and says why. There is no restore command.
- **A restore drill** restores the latest backup into a scratch directory, runs `polylogue --archive <dir> daemon --no-sources --until-idle`, then starts `polylogue --archive <dir> daemon --no-sources` and runs `polylogue --archive <dir> verify` against it. Someone runs it when they want the evidence, for example after changing the backup route.
- **Primary mode:** Sinex is the store of record, and its backup covers Polylogue's materials and authored records. Losing the local directory means a rebuild from the confirmed stream ([primary-mode.md](primary-mode.md)). The embeddings store stays local and is backed up, because recomputing it costs money.
