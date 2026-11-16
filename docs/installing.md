# Installing and configuring

Setting Polylogue up and keeping it: installing into harnesses, Chrome, shells and Google Drive, configuration and `set`, embedding with `embed`, the `demo` archive, several archives and machines, deployment, upgrades, and backup and restore. Running the daemon, `status`, notices and the change feed are [operations.md](operations.md); the command forms and exit codes are [interfaces.md](interfaces.md#commands).

## Installing

`polylogue install TARGET [--remove] [--dry-run] [--extension-id ID] [--program NAME]` wires Polylogue into something once:

| Target | Writes |
|---|---|
| `claude-code`, `codex`, `gemini-cli`, `hermes` | the harness's hook entries, its MCP server entry ([agents.md#mcp-server](agents.md#mcp-server)) and the generated skill, which is the text of `help --all` ([terminal.md#help](terminal.md#help)) |
| `chrome` | the browser extension's native-messaging host manifest, for the extension named by `--extension-id ID` ([extension.md](extension.md)) |
| `zsh`, `bash`, `fish` | the shell's completion script ([completion.md](completion.md)), and in zsh and fish the Alt-p key, which opens the picker in any shell line and inserts the chosen references at the cursor ([terminal.md#the-picker](terminal.md#the-picker)) |
| `drive` | the Google Drive sign-in for AI Studio capture ([capture.md#ai-studio-through-drive](capture.md#ai-studio-through-drive)) |

Every target follows the same rules:

- **It says what it changes.** `install` prints every file and entry it will change before writing, writes a backup beside each file it edits (`settings.json.polylogue-backup`), marks its entries as Polylogue's, and keeps other tools' entries byte for byte.
- **`--dry-run` prints what it would write** and writes nothing: a harness's settings fragment (JSON; YAML for Hermes) for a managed configuration to include, the Chrome host manifest, the completion script itself, or the Drive consent address with the scopes it asks for. Packagers and deployment modules generate their files this way (`polylogue install zsh --dry-run` is the script the package installs), so a managed configuration holds exactly what `install` would have written.
- **Managed files are left to their manager.** A target file that is a symlink (a file a deployment manages) is refused with the `--dry-run` line that prints the fragment to put in that deployment instead.
- **`--remove` takes back exactly Polylogue's entries**, with any event or `hooks` object only they filled; when the file is still exactly what the install wrote, it gets the backup's bytes back and the backup is removed. Removing a harness's hooks is how a person turns its briefs off.
- **Running it again** changes nothing when the entries are current, and replaces Polylogue's own entries when they are not.

**Hooks.** Polylogue owns one hook declaration: per harness, each event with its mode (recording or context), timeout, whether its output is injected, and the harness's measured inline cap. `install HARNESS` writes it, `install HARNESS --dry-run` prints it, and the measure `hooks` reports it against what arrived (`polylogue 'sessions | measure hooks'`): per harness and event, declared, seen in the last 7 days, last seen, and the share of sessions with a `SessionStart` line ([measures.md](measures.md#catalog)). An event counts as missing only when an opportunity for it existed. Recording entries run asynchronously where the harness supports it; context entries (session start, prompt submission, advisories, permission requests) are synchronous because the harness injects or acts on their output. What a recording entry stores is [capture.md#hooks](capture.md#hooks).

**Shells.** `install zsh`, `install bash` and `install fish` write the completion script into the shell's own completion directory for the user (for zsh, a directory on `fpath`; when none is writable, the reply names the line to add to `.zshrc`). The script calls the program by the name it was invoked as, so `--program NAME` installs it for a wrapper such as `plg` or one bound to another archive. The Alt-p key is a zsh widget or fish binding that runs `polylogue recent pick` (`Ctrl-e` in the picker switches to a query) and inserts the references it prints at the cursor, in any command line.

**Drive.** `install drive` runs Google's one-time consent for read access to the operator's AI Studio prompts, through a loopback redirect, and stores the refresh token in a file readable only by the user in the archive directory. It uses the OAuth client ID the operator registered with Google, read from the setting `drive.client_id`; until that is set it refuses with the line that sets it (`polylogue set drive.client_id ID`). The Drive collector starts once the token exists; `install drive --remove` revokes the token and deletes the file.

## Configuration

Configuration exists only where the operator's choice is genuinely open. Deployment configuration is loaded once by `main`, from the config file, then flags, then environment where meaningful, into one typed value ([architecture.md#configuration-and-clocks](architecture.md#configuration-and-clocks)). It is set through the Nix module options, the files under `$XDG_CONFIG_HOME/polylogue/`, or environment variables. Settings any author may change at run time are annotations written with `set`.

| What | Where | Why it is a choice |
|---|---|---|
| mode (`standalone` or `primary`), the Sinex endpoint, producer identity and credential file | deployment: `sinex.toml` | whether Polylogue runs on Sinex |
| archive location | `$XDG_DATA_HOME/polylogue` by default, or `--archive DIR` | where the data is kept |
| listen address and authentication of the web UI and HTTP API | deployment: `polylogue.toml`, `POLYLOGUE_LISTEN`, `POLYLOGUE_AUTH_TOKEN_FILE`; default loopback on a fixed default port with a peer-uid check; a non-loopback address needs a token file and refuses to start without one | where the web UI is reached from |
| the logical day (zone and start) | standalone: deployment; in primary mode, read from Sinex, with no value of Polylogue's own | the operator's day; its meaning is common's [query-language](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/query-language/README.md) `DaySetting` |
| the away command, value targets and action commands | deployment: `signals.toml` | which programs may be run or written to |
| enabled alternates (`reading.alternates`) | `set` | which translations or summaries readers show ([alternates.md](alternates.md)) |
| quiet hours and the interrupt budget `signal.interrupts_per_hour` | standalone: `set`; in primary mode Sinex's, applied by Sinex's deliverer | the operator's attention |
| subscription plans (`plan.<provider>`) | `set` | facts only the operator knows |
| embedding (`embed.calls`, `embed.provider`, `embed.model`, the monthly budget) | `set` | spending money ([search.md#model-calls](search.md#model-calls)) |
| the Drive client ID (`drive.client_id`) | `set` | a credential only the operator can register |

`polylogue.toml` holds `archive`, `[http]` (`listen`, `auth_token_file`) and, standalone, `[day]` (`zone`, `start`). It and `signals.toml` default to `$XDG_CONFIG_HOME/polylogue/`; `sinex.toml` belongs to the selected archive directory. `--config DIR` selects an explicit per-archive deployment directory whose `polylogue.toml`, `sinex.toml` and `signals.toml` are read together. Deployment modules generate one such directory per declared archive. `--archive`, `--listen` and `--auth-token-file` override the file, and `POLYLOGUE_LISTEN` and `POLYLOGUE_AUTH_TOKEN_FILE` override the flags. These are options of the calling process, accepted before the first word on every line. Relative paths in a file resolve against that file; relative flags against the working directory. Standalone, `[day]` may name either value or neither: the zone defaults to the host's named zone (`TZ`, then `/etc/localtime`) and the start to `04:00:00`, so Polylogue and Sinex on one host share one logical day. An invalid zone, or a host zone without an IANA name, refuses start-up. In primary mode a local `[day]` is refused, because the day is Sinex's.

`signals.toml` holds `[notifications]` (`deadline_ms`, and an optional `away_command = { program = "...", arguments = ["..."] }` that receives each notice the route function sends away, as JSON on standard input), `[value_targets.NAME]` (`path`) and `[action_commands.NAME]` (`program`, `arguments`). Programs and arguments are separate values, never a shell string, and relative paths resolve against the file.

### Settings

`polylogue set [KEY [VALUE]]` lists, shows and changes the run-time settings:

- `polylogue set` lists every setting with its value, scope, default, who set it and when.
- `polylogue set reading.alternates` shows one setting: its value, its type and allowed values, its scope, its default, and its history (every revision with author and time).
- `polylogue set reading.alternates translation:pl` writes it and prints the undo line, as every write does ([annotations.md#the-write-path](annotations.md#the-write-path)).

A setting is an annotation of kind `setting` ([annotations.md#kinds](annotations.md#kinds)) with author, `via` and time, from a declared catalog with a type, default and scope per key; unknown keys and blank values are refused, with the keys or values that exist. A host-scoped key applies to the machine that wrote it; an archive-scoped key applies everywhere. In primary mode the keys Sinex holds (quiet hours, the interrupt budget) show Sinex as where they are set and have no local value. Any author may change a setting; `status` lists those changed by anyone but the operator that he has not judged ([operations.md#status](operations.md#status)). Withdrawing a setting's annotation returns the key to its previous value or default. Settings are listed by `set`, not by a unit of their own.

**Not configurable,** because a dominant option exists: provider locations (fixed per origin, honouring a tool's own documented home variable such as `CLAUDE_CONFIG_DIR` or `CODEX_HOME`; [capture.md#locations](capture.md#locations)); whether the session-start brief is injected (removing the hooks is the off switch); brief sizes and budgets; worker counts and memory targets (derived from the cgroup CPU quota and affinity); pacing; deadlines (defaults any caller lifts).

**Environment.** No environment variable selects the archive, other than the XDG base directories, `HOME` and the tools' own home variables. The binary reads only those, `PAGER`, `NO_COLOR`, `TERM`, `COMPLETE`, the harness variables read for caller identity, `POLYLOGUE_LISTEN` and `POLYLOGUE_AUTH_TOKEN_FILE`, `DBUS_SESSION_BUS_ADDRESS` for desktop notifications, and `VOYAGE_API_KEY`, the optional credential for paid embeddings. All are read once at start-up. Terminal styling is [terminal.md#output](terminal.md#output)'s.

**Broken configuration never falls back.** A `sinex.toml` that does not parse or names an unknown mode leaves the archive in its recorded mode: `status` shows the error with the file's path, the verdict is `degraded`, writes are refused with that error and path, and capture pauses. An archive without a recorded mode cannot be created until its configuration is fixed.

## Embedding

`polylogue embed [--query 'QUERY'] [--yes]` buys vectors for semantic ranking ([search.md#semantic-ranking](search.md#semantic-ranking)). It is a command because spending money is an explicit act.

- **Without `--yes`** it spends nothing: it reports what would be embedded (texts and chunks without a vector for the configured model, over the whole archive or the rows `--query` selects), the tokens, the price and the cost against this month's budget.
- **With `--yes`** it embeds them through the configured provider, reporting progress and what was bought; a retry requests only what is still missing. With model calls off it is refused with the setting that turns them on (`polylogue set embed.calls on`).
- Once model calls are on, the daemon embeds new text in the background at idle priority within the monthly budget; `embed --yes` is for what came before.
- Stored vectors, the provider and the month's spend are in `status` ([operations.md#status](operations.md#status)).

## Demo

`polylogue demo DIR` builds the Morrow sample archive ([testing.md#morrow](testing.md#morrow)) in a new directory `DIR` and prints the first lines to try on it. It creates `DIR`, refusing a directory that exists, lays out Morrow's tool files under `DIR/home`, reads them and imports Morrow's exports into the archive `DIR/archive` with the real engine, and exits. It then prints the line that starts a daemon on that archive with sources off, and about six lines to try, each with `--archive DIR/archive`: a word search, a listing of one repository's week, a read of one row, a note with its undo line, a measure, and `help`. Each printed line is checked on Morrow by a test. The demo archive is an ordinary archive; deleting `DIR` removes it.

## Several archives and machines

**Several archives on one machine.** An archive is a directory. `--archive DIR` selects it on every line, so a client and the daemon it reaches always agree. Each archive has its own daemon, lock, socket (`daemon.sock` under the archive directory) and unit instance. A command never acts on an archive it was not pointed at, and every `status` and error names the archive path.

**Several machines.** Every observation records the machine ID and host name, so paths from two machines never collide, and sessions show their host where an archive holds more than one. Archives on different machines combine either through primary mode, each host sending to one Sinex as its own producer ([primary-mode.md](primary-mode.md)), or standalone, by writing a bundle with `polylogue 'repo:sinex' read --to DIR` on one machine ([reading.md#formats-and-destinations](reading.md#formats-and-destinations)) and reading it with `polylogue import DIR` on the other. An archive directory is never synced live between machines.

Runtime files use `$XDG_RUNTIME_DIR/polylogue/<archive key>/`. When `XDG_RUNTIME_DIR` is unset, they use `$XDG_STATE_HOME/runtime/polylogue/<archive key>/`; `XDG_STATE_HOME` defaults to `$HOME/.local/state`. The daemon and every client use the same resolved paths and archive key. Reader tokens and ports, mailbox flags, hosted-session records, terminal row files and this boot's storage-failure marker share that directory.

## Deployment

- **The flake** exports `packages.polylogue`, `nixosModules.polylogue` and `homeModules.polylogue`. The modules' options are typed from the same declaration as the configuration above, through one shared settings library; a deployment only sets options. Any deployment works the same way; none is a dependency of Polylogue's behaviour.
- **The package** can be built with the Sinex adapter, a Cargo feature outside the default build, so the standalone build has no NATS dependency. The binary reports its version and commit (`--version`, `status`).
- **The Home Manager module installs:** one wrapper used as the CLI, the daemon command and the browser host, passing `--archive` explicitly and setting the XDG directories; `polylogue`, `plg` and `plog` as shell names of the same wrapper, with completion and the Alt-p key for each; the browser extension and its native-messaging manifest; the harness hooks and MCP registration; `sinex.toml` and `signals.toml`; and one user service per archive with `Type=notify`, `WatchdogSec`, `Restart=on-failure`, `RestartPreventExitStatus=74 78`, a failure notification, and a resource class that yields I/O and CPU to interactive work with no memory cap. The completion scripts, host manifest and harness entries are the output of `install TARGET --dry-run` ([Installing](#installing)), so the module and `install` never differ.
- **The daemon restarts only when its binary changes;** activating unrelated changes leaves it running.
- **Placement of stores on the host** (separate volumes, snapshot schedules, copy-on-write settings) is deployment configuration. Whatever the placement, the archive stays covered by a backup.

## Upgrades

An upgrade is a version change in `status` and, when a parser changed, a while in which some chains are re-read.

| Change in the release | What happens | Reads meanwhile |
|---|---|---|
| no storage change | restart, resume from committed positions | down for the restart |
| a parser module's reading version | chains read with the old version are re-parsed one at a time, live inputs first, then newest first | every read replies; results touching a chain not yet re-parsed carry `up_to_date: false`; `status` shows progress |
| a change to the index schema | a new index is built from stored bytes in a private file and installed when its build ends ([below](#replacing-an-index-from-another-schema)) | reads reply `index_building` with the build's progress until the new index is installed |
| a table added to `embeddings.db` | added in place at start; stored vectors are untouched | down for the restart |
| `archive.db` or `embeddings.db` in another format than the binary's | refused, exit 78, not restarted | down |
| the disk failed a sync of `archive.db` (any time, not an upgrade) | exit 74, not restarted, refuses until the machine reboots ([Storage failure](operations.md#storage-failure)) | down |

**Spools across a rollback.** Hooks and the browser host keep writing their spools while the daemon is replaced, and the spools are never deleted, so the build a deployment rolls back to reads what the newer build wrote. A spool line's shape is therefore part of the deployed format: the hook's line shapes are pinned by a fixture that the previous build's reader is checked against, and a change to them is checked against the deployed reader before it ships.

### Replacing an index from another schema

`index.db` holds only what is parsed and derived from the archive's stored bytes, and it has one current format, so an index this build cannot open is replaced rather than converted or refused. At start, an index whose tables, indexes or triggers differ from the build's, that is at another index format version, or that carries no Polylogue format identity, is closed and moved with any write-ahead log it left into the build directory as `index-build/previous.db`, and a new index is built beside it in `index-build/index.db`. The schema identity is a short hash of the file's version, format identity and its tables, views and triggers; indexes are left out of it, since read indexes are added at open and a build defers some.

The build writes its private file with no journal on disk and no syncs, and only the build pipeline opens it; how a stop, a crash and the end of the build leave it is in [architecture.md#durability](architecture.md#durability). Until the new index is installed, reads are served from the previous index when its schema identity is this build's (an index that differs only in its indexes), so the daemon keeps serving what it had; otherwise, and on a first build with no index at all, every read replies `index_building` with the build's progress (input chains read of those stored), and `status` shows that progress beside the daemon's own state. When the build ends, the new index is renamed over `index.db`, reads move to it, and the previous index is removed.

The new index records why it replaced the old one, and `status` reports it under `store.index_rebuild`: the reason (`schema_differs`, `format_version` with the version found and wanted, or `older_format`), the previous and current schema identities, and whether the build is still running. Progress is the rest of the same reply: `index_build` while the build runs, and afterwards chains by parse state against the stored inputs and the read completeness. The record stays with the index it describes. In primary mode the cache from another schema is set aside under its schema identity as `index-set-aside-<schema identity>.db` and rebuilt from the confirmed stream, and the set-aside file is removed once the rebuild reaches the stream's latest position ([primary-mode.md](primary-mode.md)).

The archive is never handled this way: `archive.db`, the segment log and the embeddings store in another format are refused (exit 78) and left as they were, because they hold what cannot be recomputed. The embeddings store from an earlier build only gains the tables added since ([archive.md#evolution](archive.md#evolution)).

The first build reads every input's newest revision chain first and replies to reads with `index_building` and progress until it can serve them. Once those are read (or refused) the first build ends: its read indexes and search are built, the index is installed and the sessions derived, and the older chains, which hold earlier readings of the same inputs, are read afterwards as regular work, with reads reporting parsing as incomplete until they are. Moving `index.db` aside by hand and starting the daemon builds a new index the same way. Moving a whole archive aside and starting fresh is a deliberate act, never something code does to get past a change. The planning query selects newest or older chains before joining parser progress, so each discovery phase reads only its chain set's parser states. Newest means the chain containing the greatest revision number; its stored length remains that chain's maximum declared length, including stored histories that resume an earlier chain with a shorter revision. Input enumeration and historical source metadata still require their existing reads.

## Backup and restore

- **Backup is the deployment's filesystem backup.** Polylogue has no backup command and keeps its stores consistent at every commit. A correct backup takes the segment log, the databases and the embeddings store together; `index.db` and the `index-build/` directory may be left out, since the index is rebuilt from stored bytes.
  - An atomic snapshot of the archive directory is consistent.
  - A file copy needs either a stopped daemon with each database's `-wal` file included when present, or each SQLite database copied through SQLite's backup API followed by the segment log: committed extents never move and no row cites an extent before it commits, so databases first and segments after is consistent.
- **Polylogue provides** the paths and sizes in `status`, and `status --verify` to check a restored copy ([operations.md#checking-every-stored-byte](operations.md#checking-every-stored-byte)).
- **Restore** is putting the files back and starting the daemon. Each store opens as at any start; an uncommitted segment tail is truncated; an index behind its positions catches up; a daemon that finds a segment log but no `archive.db` refuses to start a fresh archive over it and says why. There is no restore command.
- **A restore drill** restores the latest backup into a scratch directory and, with `--archive` naming it on each line, runs `polylogue daemon --no-sources --until-idle`, then starts `polylogue daemon --no-sources` and runs `polylogue status --verify` against it. Someone runs it when they want the check, for example after changing the backup route.
- **Primary mode:** Sinex is the store of record, and its backup covers Polylogue's materials and authored records. Losing the local directory means a rebuild from the confirmed stream ([primary-mode.md](primary-mode.md)). The embeddings store stays local and is backed up, because recomputing it costs money.
