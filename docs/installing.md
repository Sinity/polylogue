# Installing and configuring

Setting Polylogue up and keeping it: installing into harnesses, Chrome, shells and Google Drive, configuration and `set`, embedding with `embed`, the `demo` archive, several archives and machines, deployment, upgrades, and backup and restore. Running the daemon, `status`, notices and the change feed are [operations.md](operations.md); the command forms are [interfaces.md](interfaces.md#commands) and exit codes [interfaces.md#exit-codes](interfaces.md#exit-codes).

## Installing

`polylogue install TARGET [--remove] [--dry-run] [--instructions] [--extension-id ID] [--program NAME]` wires Polylogue into something once:

| Target | Writes |
|---|---|
| `claude-code`, `codex`, `gemini-cli`, `hermes` | the harness's hook entries, its MCP server entry ([agents.md#mcp-server](agents.md#mcp-server)) and the generated skill, which is the text of `help --all` ([terminal.md#help](terminal.md#help)) |
| `chrome` | the browser extension's native-messaging host manifest for the packaged extension, with `--extension-id ID` available for another build ([extension.md](extension.md)) |
| `zsh`, `bash`, `fish` | the shell's completion script ([completion.md](completion.md)), and in zsh and fish the Alt-p key, which opens the picker in any shell line and inserts the chosen references at the cursor ([terminal.md#the-picker](terminal.md#the-picker)) |
| `drive` | the Google Drive sign-in for AI Studio capture ([capture.md#ai-studio-through-drive](capture.md#ai-studio-through-drive)) |

Every target follows the same rules:

- **It says what it changes.** `install` prints every file and entry it will change before writing, writes a backup beside each file it edits (`settings.json.polylogue-backup`), marks its entries as Polylogue's, and keeps other tools' entries byte for byte.
- **`--dry-run` prints what it would write** and writes nothing: a harness's settings fragment (JSON; YAML for Hermes) for a managed configuration to include, the Chrome host manifest, the completion script itself, or the Drive consent address with the scopes it asks for. Packagers and deployment modules generate their files this way (`polylogue install zsh --dry-run` is the script the package installs), so a managed configuration holds exactly what `install` would have written.
- **Managed files are left to their manager.** A target file that is a symlink (a file a deployment manages) is refused with the `--dry-run` line that prints the fragment to put in that deployment instead.
- **`--remove` takes back exactly Polylogue's entries**, with any event or `hooks` object only they filled; when the file is still exactly what the install wrote, it gets the backup's bytes back and the backup is removed. Removing a harness's hooks is how a person turns its briefs off.
- **Running it again** changes nothing when the entries are current, and replaces Polylogue's own entries when they are not.

**Chrome.** `install chrome --dry-run` prints the same manifest that an installation writes. The host program is the first executable `polylogue` on `PATH`. Use `--program` to select another command or an executable path. The manifest keeps the wrapper path so Chrome opens the archive chosen by the wrapper. Installation keeps the original manifest bytes and permissions in its adjacent backup. Removal restores that backup when the current manifest still matches an installation. A manifest edited afterward is left in place with its backup, and the command reports the changed file.

**Instructions.** `polylogue install claude-code --instructions` also appends the same short guidance the MCP server supplies, enclosed by `<!-- polylogue -->` and `<!-- /polylogue -->`, to the harness's global instructions. Claude Code uses `$CLAUDE_CONFIG_DIR/CLAUDE.md` (default `~/.claude/CLAUDE.md`), Codex uses `$CODEX_HOME/AGENTS.md` (default `~/.codex/AGENTS.md`), and Gemini CLI uses `~/.gemini/GEMINI.md`. A `--settings` override changes only the hook settings path. Hermes refuses this flag with its `config.yaml` hint.

The block is printed before writing. Installation preserves the original bytes and permissions in the same adjacent backup used by the other installers, updates only the marked block, and leaves text outside it byte for byte. Repeating installation makes no further edit. `--instructions --dry-run` prints the block and changes neither instructions nor hook settings, including when the files are managed. `--instructions --remove` removes both the block and Polylogue's hooks, restoring original bytes when the rest is unchanged; `--remove --dry-run` reports the files without changing them. Incomplete or repeated block markers are refused before hooks change. Without `--instructions` no instructions file is touched.

**Hooks.** Polylogue owns one hook declaration: per harness, each event with its mode (recording or context), timeout, whether its output is injected, and the harness's measured inline cap. `install HARNESS` writes it, `install HARNESS --dry-run` prints it, and the measure `hooks` reports it against what arrived (`polylogue 'sessions | measure hooks'`): per harness and event, declared, seen in the last 7 days, last seen, and the share of sessions with a `SessionStart` line ([measures.md](measures.md#catalog)). An event counts as missing only when an opportunity for it existed. Recording entries run asynchronously where the harness supports it (Claude Code's `async` field; the Codex and Gemini CLI entries are written without one); context entries (session start, prompt submission, advisories, permission requests) are synchronous because the harness injects or acts on their output. What a recording entry stores is [capture.md#hooks](capture.md#hooks).

**Shells.** `install zsh`, `install bash` and `install fish` write the completion script into the user's completion directory: `$XDG_DATA_HOME/zsh/site-functions/_NAME`, `$XDG_DATA_HOME/bash-completion/completions/NAME` or `$XDG_CONFIG_HOME/fish/completions/NAME.fish`. The XDG defaults apply. `--program NAME` installs for a wrapper such as `plg` or one bound to another archive. The generated script calls that same program.

Zsh installation adds a marked source block to `$ZDOTDIR/.zshrc`, using the home directory when `ZDOTDIR` is unset. Fish installation adds a source block in `$XDG_CONFIG_HOME/fish/conf.d/NAME-polylogue.fish`. These ordinary startup files activate both completion and Alt-p when the shell starts. Existing text and later edits outside the marked block are preserved; removal restores the original bytes when the rest of the file is unchanged. A managed startup file is left to its manager: include the dry-run script in its declared source.

Alt-p runs `NAME recent pick` and inserts each chosen reference as one quoted shell word at the cursor. The picker draws on the terminal while the binding captures only its reference output. Cancelling inserts nothing.

**Drive.** `install drive` runs Google's one-time consent for read access to the operator's AI Studio prompts, through a loopback redirect, and stores the refresh token in a file readable only by the user at `drive-token.json` in the archive directory. The loopback flow uses PKCE and requests offline access. It uses the OAuth client ID the operator registered with Google, read from the setting `drive.client_id`; until that is set it refuses with the line that sets it (`polylogue set drive.client_id ID`). The Drive collector starts once the token exists; `install drive --remove` revokes the token and deletes the file. Failed authorization leaves no token; failed revocation keeps the existing file. Repeating installation keeps the authorized token unchanged. A legacy token outside the archive is left in place and does not authorize the current collector.

## Configuration

Configuration exists only where the operator's choice is genuinely open. Deployment configuration is loaded once by `main`, from the config file, then flags, then environment where meaningful, into one typed value ([architecture.md#configuration-and-clocks](architecture.md#configuration-and-clocks)). It is set through the Nix module options, the files under `$XDG_CONFIG_HOME/polylogue/`, or environment variables. Settings any author may change at run time are annotations written with `set`.

| What | Where | Why it is a choice |
|---|---|---|
| mode (`standalone` or `primary`) | deployment: `sinex.toml` | whether Polylogue runs on Sinex; primary mode is not built in this version, and a build with the `sinex` feature refuses to start with it |
| archive location | `$XDG_DATA_HOME/polylogue` by default, or `--archive DIR` | where the data is kept |
| listen address and authentication of the web UI and HTTP API | deployment: `polylogue.toml`, `POLYLOGUE_LISTEN`, `POLYLOGUE_AUTH_TOKEN_FILE`; default loopback on a fixed default port with a peer-uid check; a non-loopback address needs a token file and refuses to start without one | where the web UI is reached from |
| the logical day (zone and start) | standalone: deployment; in primary mode, read from Sinex, with no value of Polylogue's own | the operator's day; its meaning is common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md) `DaySetting` |
| the away command, value targets and action commands | deployment: `signals.toml` | which programs may be run or written to |
| enabled alternates (`reading.alternates`) | `set` | which translations or summaries readers show ([alternates.md](alternates.md)) |
| quiet hours and the interrupt budget `signal.interrupts_per_hour` | standalone: `set`; in primary mode Sinex's, applied by Sinex's deliverer | the operator's attention |
| subscription plans (`plan.<provider>`) | `set` | facts only the operator knows |
| embedding (`embed.calls`, `embed.provider`, `embed.model`, the monthly budget) | `set` | spending money ([search.md#model-calls](search.md#model-calls)) |
| the Drive client ID (`drive.client_id`) | `set` | a credential only the operator can register |

`polylogue.toml` holds `archive`, `[http]` (`listen`, `auth_token_file`) and, standalone, `[day]` (`zone`, `start`). It and `signals.toml` default to `$XDG_CONFIG_HOME/polylogue/`; `sinex.toml` belongs to the selected archive directory. `--config FILE` selects an explicit `polylogue.toml`, and the `sinex.toml` and `signals.toml` beside it are read with it. Deployment modules generate one such configuration per declared archive. Their path options (`path`, `auth_token_file`, and a command's `program` or a value target's `path`) take absolute paths only: the generated files live in the Nix store, where a relative path would resolve to nothing the deployment owns. `--archive`, `--listen` and `--auth-token-file` override the file, and `POLYLOGUE_LISTEN` and `POLYLOGUE_AUTH_TOKEN_FILE` override the flags. These are options of the calling process, accepted before the first word on every line, as `--archive DIR` or `--archive=DIR`. A client reads only `polylogue.toml`, for the archive and the logical day's zone it shows times in; the listen address, the token file, the machine identity, `sinex.toml` and `signals.toml` are read by the daemon and by the commands that run its engine in the calling process (`demo`, `install`, the hook and the browser host). Relative paths in a file resolve against that file; relative flags against the working directory. Standalone, `[day]` may name either value or neither: the zone defaults to the host's named zone (`TZ`, then `/etc/localtime`) and the start to `04:00:00`, so Polylogue and Sinex on one host share one logical day. An invalid zone, or a host zone without an IANA name, refuses start-up. In primary mode a local `[day]` is refused, because the day is Sinex's.

`signals.toml` holds `[notifications]` (`deadline_ms`, and an optional `away_command = { program = "...", arguments = ["..."] }` that receives each notice the route function sends away, as JSON on standard input), `[value_targets.NAME]` (`path`) and `[action_commands.NAME]` (`program`, `arguments`). Programs and arguments are separate values, never a shell string, and relative paths resolve against the file.

### Settings

`polylogue set [KEY [VALUE]]` lists, shows and changes the run-time settings:

- `polylogue set` lists every setting with its value, scope, default, who set it and when.
- `polylogue set reading.alternates` shows one setting: its value, its type and allowed values, its scope, its default, and its history (every setting revision with its exact reference, value, author, channel, recording host, known time and current/withdrawn state). History lists the newest known authored time first, with descending revision ID breaking ties; undated revisions stay undated and appear last. This display order never chooses the effective value, which follows the revision links. Host history includes only the selected target host, including changes written from another machine.
- `polylogue set reading.alternates '["translation:pl"]'` writes it and prints the undo line, as every write does ([annotations.md#the-write-path](annotations.md#the-write-path)).
- `polylogue set signal.interrupts_per_hour 2 --machine laptop` writes a host-scoped setting for the named machine rather than the one running the line.

A setting is an annotation of kind `setting` ([annotations.md#kinds](annotations.md#kinds)) with author, `via` and time, from a declared catalog with a type, default and scope per key; unknown keys and blank values are refused, with the keys or values that exist. A host-scoped key applies to the machine that wrote it, or to the machine `--machine NAME` names (the setting's stored `machine`, [annotations.md](annotations.md#kinds)); an archive-scoped key applies everywhere and refuses `--machine`. In primary mode the keys Sinex holds (quiet hours, the interrupt budget) show Sinex as where they are set and have no local value or local default; retained local setting history remains readable without becoming their effective value. Any author may change a setting; `status` lists those changed by anyone but the operator that he has not judged ([operations.md#status](operations.md#status)). Withdrawing a setting's annotation returns the key to its previous value or default. Settings are listed by `set`, not by a unit of their own.

**Not configurable,** because a dominant option exists: provider locations (fixed per origin, honouring a tool's own documented home variable such as `CLAUDE_CONFIG_DIR` or `CODEX_HOME`; [capture.md#locations](capture.md#locations)); whether the session-start brief is injected (removing the hooks is the off switch); brief sizes and budgets; worker counts and memory targets (derived from the physical cores the process may run on, its cgroup CPU quota, and the memory the system has available at most the cgroup's limit); pacing; deadlines (defaults any caller lifts).

**Environment.** No environment variable selects the archive, other than the XDG base directories, `HOME` and the tools' own home variables. The binary reads only those, `PAGER`, `NO_COLOR`, `TERM`, `COMPLETE`, the harness variables read for caller identity, `POLYLOGUE_LISTEN`, `POLYLOGUE_AUTH_TOKEN_FILE` and `POLYLOGUE_MCP_SOCKET`, `DBUS_SESSION_BUS_ADDRESS` for desktop notifications, `NOTIFY_SOCKET` and `WATCHDOG_USEC`/`WATCHDOG_PID` for the systemd notify and watchdog protocol, and `VOYAGE_API_KEY`, the optional credential for paid embeddings. All are read once at start-up. Terminal styling is [terminal.md#output](terminal.md#output)'s.

**Broken configuration never falls back.** A `sinex.toml` that does not parse or names an unknown mode leaves the archive in its recorded mode: `status` shows the error with the file's path, the verdict is `degraded`, writes are refused with that error and path, and capture pauses. An archive without a recorded mode cannot be created until its configuration is fixed.

## Embedding

`polylogue embed [--query 'QUERY'] [--yes]` buys vectors for semantic ranking ([search.md#semantic-ranking](search.md#semantic-ranking)). It is a command because spending money is an explicit act.

Use a bound in the query to select a finite set, for example `polylogue embed --query 'sessions where physical:true | sort by id | limit 5'`. These bounds apply to session rows, rather than a separate number of texts or chunks. The command has no separate text-count limit.

- **Without `--yes`** it spends nothing: it reports what would be embedded (texts and chunks without a vector for the configured model, over the whole archive or the rows `--query` selects), the tokens, the price and the cost against this month's budget.
- **With `--yes`** it embeds them through the configured provider, reporting progress and what was bought; a retry requests only what is still missing. With model calls off it is refused with the setting that turns them on (`polylogue set embed.calls on`).
- Once model calls are on, the daemon embeds new text in the background at idle priority within the monthly budget; `embed --yes` is for what came before.
- With the provider off, semantic-only queries refuse and name `polylogue set embed.provider local`. A configured model without vectors names `polylogue embed`; preview it before authorizing a build with `--yes`.
- Stored vectors, the provider and the month's spend are in `status` ([operations.md#status](operations.md#status)).

## Demo

`polylogue demo DIR` builds the Morrow sample archive ([testing.md#morrow](testing.md#morrow)) in a new directory `DIR` and walks through it. It creates `DIR`, refusing a directory that exists, lays out Morrow's tool files under `DIR/home`, reads them (only `DIR/home`: the tools' root variables in the caller's environment are not consulted) and imports Morrow's exports into the archive `DIR/archive` with the real engine, runs a daemon on that archive for the walk, and stops it. The walk is ten steps (a search, reading a session, tool failures, parent and child sessions, unknown costs, a usage measure, a note, reading the note and a takeover brief), each shown with its command line and output. The transcript is printed and also written to `DIR/transcript.md`; with `--json` the steps and their timings are printed instead. The demo archive is an ordinary archive; deleting `DIR` removes it.

## Several archives and machines

**Several archives on one machine.** An archive lives in one directory. `--archive DIR` selects it on every line, so a client and the daemon it reaches always agree. Each archive has its own daemon, lock, socket (`daemon.sock` under the archive directory) and unit instance. A command never acts on an archive it was not pointed at, and every `status` and error names the archive path.

**Managed MCP during archive cutovers.** The installed daemon and MCP launcher can share `POLYLOGUE_MCP_SOCKET`, an absolute path in the user's private runtime directory. When the daemon binds its archive socket, it publishes that path as a link to the listening socket. The MCP process connects through the link for each request, so the same stdio session follows the next managed daemon after an archive cutover. During the stop it reports `not_running`. An explicit `--archive` or `--config` uses that deployment's own archive socket. Existing MCP processes launched before this setting require a reconnect through the updated launcher.

**Several machines.** Every observation records the machine ID and host name, so paths from two machines never collide, and sessions show their host where an archive holds more than one. Archives on different machines combine either through primary mode, each host sending to one Sinex as its own producer ([primary-mode.md](primary-mode.md)), or standalone, by writing a bundle with `polylogue 'repo:sinex' read --to DIR --format json` on one machine ([reading.md#formats-and-destinations](reading.md#formats-and-destinations)) and reading it with `polylogue import DIR` on the other. An archive directory is never synced live between machines.

Runtime files use `$XDG_RUNTIME_DIR/polylogue/<archive key>/`. When `XDG_RUNTIME_DIR` is unset, they use `$XDG_STATE_HOME/runtime/polylogue/<archive key>/`; `XDG_STATE_HOME` defaults to `$HOME/.local/state`. The daemon and every client use the same resolved paths and archive key. Reader tokens and ports, mailbox flags, hosted-session records, terminal row files and this boot's storage-failure marker share that directory.

## Deployment

- **The flake** exports `packages.polylogue`, `nixosModules.polylogue` and `homeModules.polylogue`. The modules' options are typed from the same declaration as the configuration above, through one shared settings library; a deployment only sets options. Any deployment works the same way; none is a dependency of Polylogue's behaviour.
- **The package** is built with the `sinex` Cargo feature, which a plain Cargo build leaves out: it holds Polylogue's Sinex event kinds. Primary mode is not built in this version; with the feature, a `sinex.toml` asking for it, or an archive whose `meta.json` records it, is refused at start with a `not_built` failure that names the mode, and the archive is left untouched. The binary reports its version and commit (`--version`, `status`).
- **The Home Manager module installs:** one wrapper used as the CLI, the daemon command and the browser host, passing `--archive` explicitly and setting the XDG directories; `polylogue`, `plg` and `plog` as shell names of the same wrapper, with completion and the Alt-p key for each; the harness hooks and MCP registration; `sinex.toml` and `signals.toml`; and one user service per archive with `Type=notify`, `WatchdogSec`, `Restart=always` with a growing delay, `RestartPreventExitStatus=74 78`, a notification after every end it did not ask for, and a resource class that yields I/O and CPU to interactive work with no memory cap. The completion scripts and harness entries use the output of `install TARGET --dry-run` ([Installing](#installing)).
- Set `services.polylogue.browser.enable = true` to install the browser host for the selected archive and link the unpacked extension at `$XDG_DATA_HOME/polylogue/extension`. `services.polylogue.browser.userDataDirectories` lists the Chrome user-data directories that receive the manifest and defaults to `$XDG_CONFIG_HOME/google-chrome`. Load the linked directory once using **Load unpacked** in `chrome://extensions`.
- **The daemon restarts only when its binary changes;** activating unrelated changes leaves it running.
- **Placement of stores on the host** (separate volumes, snapshot schedules, copy-on-write settings) is deployment configuration. Whatever the placement, the archive stays covered by a backup.

## Upgrades

An upgrade is a version change in `status` and, when a parser changed, a while in which some chains are re-read.

| Change in the release | What happens | Reads meanwhile |
|---|---|---|
| no storage change | restart, resume from committed positions | down for the restart |
| a parser module's parse version | chains read with the old version are re-parsed one at a time, live inputs first, then newest first | every read replies; results touching a chain not yet re-parsed carry `up_to_date: false`; `status` shows progress |
| a change to the index schema | a new index is built from stored bytes in a private file and installed when its build ends ([below](#replacing-an-index-from-another-schema)) | reads reply `index_building` with the build's progress until the new index is installed |
| a table added to `embeddings.db` | added in place at start; stored vectors are untouched | down for the restart |
| `archive.db` or `embeddings.db` in another format than the binary's | refused, exit 78, not restarted | down |
| the disk failed a sync of `archive.db` (any time, not an upgrade) | exit 74, not restarted, refuses until the machine reboots ([Storage failure](operations.md#storage-failure)) | down |

**Spools across a rollback.** Hooks and the browser host keep writing their spools while the daemon is replaced, and the spools are never deleted, so the build a deployment rolls back to reads what the newer build wrote. A spool line's shape is therefore part of the deployed format: the hook's line shapes are pinned by a fixture that the previous build's reader is checked against, and a change to them is checked against the deployed reader before it ships.

### Replacing an index from another schema

The schema identity of `index.db` is a hash of the version, format
identity, tables, views and triggers. Read indexes are completed at open
and a private bulk rebuild can defer their creation. An index that belongs
to another archive, uses an incompatible schema, or is ahead of its
archive cannot serve reads.

When there is no usable index, the daemon creates the current empty
schema in a private file and installs it atomically. The ordinary writer
fills it from live captures and the archive's stored inputs. Sessions and
search results become readable once their own rows and search entries
commit. Queries report incomplete coverage during catch-up, and status
names pending work in capture, parsing, session figures and search.
Stopping during catch-up keeps every committed reading. The next start
continues work that is still due. Moving `index.db` out of the way while
the daemon is stopped has the same effect.

When a usable index is privately rebuilt, readers keep using it until the
replacement is complete. The rebuild's reason and current stage are
reported in status. An interrupted private rebuild can be recomputed from
the record. If the finished replacement survives a stop, the open checks
its archive identity and position before installing it.

Moving a whole archive aside is a deliberate operation. Original input
bytes, authored records and paid vectors stay in the archive across an
index replacement. An incompatible archive or embeddings format is
refused with exit status 78. Compatible embeddings schemas gain newly
added tables ([archive evolution](archive.md#evolution)).

### Continuing an indexed cut of the same archive

For a one-time standalone cutover using the current archive and index formats, first make a coherent copy of the source. Build and verify its index with the destination reading contracts. Keep the copy sealed after verification. Choose a recent cut so the final catch-up has little to read. Build both examples with the destination revision before the write pause.

1. Run `cargo run -p polylogue-store --example reuse_archive_index -- continue <checked-cut> <source> <new-candidate>`. The candidate must be a new path outside both archive directories, on storage that can reflink the source's databases and WALs. The command checks reflinking and compares the sealed copy's records, stored bytes and paid vectors against the current native source, then prepares the index and the checked segment prefix while the source writer may still run.
2. After preparation reports both positions, stop the source writer and enter `finish` in the same process. The command takes the source's exclusive writer lock, refuses a replaced source or a position that went backwards, appends the remaining committed segment bytes, and reflinks the final SQLite databases and WALs. It fails if a required reflink fails. The embeddings database follows its own commits even when the archive position has not moved. Rehearse on the intended storage to time the final capture and catch-up.
3. Run the built `catch_up_archive_suffix <candidate> <index-position> <archive-position>` example from `polylogue-engine` with the two final positions printed by the capture. Check that the installed index serves the final archive position and can read a changed input before routing traffic to it. Do the full stored-byte verification on the checked copy before the write pause. Keep the original archive for recovery.

Both cuts must belong to the same native archive. Between preparation and capture, only its native writer may change it. A restored, rewritten or independently converted source needs a new proof. This procedure depends on the native writer keeping committed segment ranges and captured record rows intact. Independent sealed copies use `reuse_archive_index <earlier> <later>` to check their complete common prefix before indexing the later one.

A read can finish in a later commit while keeping the same revision number. The prefix check accepts that growth when a later committed extent accounts for the new length. It checks the earlier extents and their bytes and rejects changes to the revision's identity or original commit. Suffix planning uses extent commits alongside revisions and observations, and a bounded read uses the extents present at its requested cut.

## Backup and restore

- **Backup is the deployment's filesystem backup.** Polylogue has no backup command and keeps its stores consistent at every commit. A correct backup takes the segment log, the databases and the embeddings store together; `index.db`, its full-text files (`search-<id>/`) and the `index-rebuild/` directory may be left out, since the index is rebuilt from stored bytes.
  - An atomic snapshot of the archive directory is consistent.
  - A file copy needs either a stopped daemon with each database's `-wal` file included when present, or each SQLite database copied through SQLite's backup API followed by the segment log: committed extents never move and no row cites an extent before it commits, so databases first and segments after is consistent.
- **Polylogue provides** the paths and sizes in `status`, and `status --verify` to check a restored copy ([operations.md#checking-every-stored-byte](operations.md#checking-every-stored-byte)).
- **Restore** is putting the files back and starting the daemon. Each store opens as at any start; an uncommitted segment tail is truncated; an index behind its positions catches up; a daemon that finds a segment log but no `archive.db` refuses to start a fresh archive over it and says why. There is no restore command.
- **A restore drill** restores the latest backup into a scratch directory and, with `--archive` naming it on each line, runs `polylogue daemon --no-sources --until-idle`, then starts `polylogue daemon --no-sources` and runs `polylogue status --verify` against it. Someone runs it when they want the check, for example after changing the backup route.
- **Primary mode:** Back up the local record, held bytes and paid embeddings together as described in [primary mode](primary-mode.md#local-state). Sinex backs up what it has received. Its confirmed stream can restore shipped records, while unshipped writes and paid embeddings need the local backup.
