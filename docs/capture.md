# Capture

Where Polylogue's inputs come from: the location model and every live origin, hooks, tool databases, exports imported by hand, AI Studio through Drive, and capture status. What happens to the bytes once collected is in [archive.md](archive.md); the parsed model is in [sessions.md](sessions.md). The browser extension's capture and provider import are in [extension.md](extension.md). Capture is the same code in both modes; where the collected bytes are kept in primary mode is in [primary-mode.md](primary-mode.md#bytes).

## Principles

- **Every relevant origin, at full fidelity.** Every AI harness whose sessions the user has is captured, with every side file that carries conversation content, context or harness-side state, even before a parser for it exists, because harnesses prune their own files and a pruned file is lost. A degraded form (a flattened transcript, a page scrape where the provider's data is available) is never the stored record.
- **Nothing disappears silently.** Every file under a declared root is collected, ignored with a recorded reason, or reported unrecognized.
- **Fixed locations, strict recognition, fail closed.** Polylogue reads data where each harness writes it; there are no path settings and no custom roots. Input that does not match its declared shape is refused, never reinterpreted.
- **A leading UTF-8 byte order mark is ignored.** A text input saved with one (common from Windows tools) is recognized and parsed as without it. The stored bytes and every recorded byte offset still count the mark.
- **Live is the normal case.** A growing file is tailed; a live origin needs no import step.
- **Exports arrive by hand.** Provider exports are brought in with `polylogue import PATH`; nothing is picked up from a download or inbox directory.
- **Databases are captured as rows, not images** ([Tool databases](#tool-databases)).
- **No size limit on valid input.** Payloads, attachments and export members of any size stream through.

## Locations

### Roots, locations and the ignored list

- Each origin declares its **root**, a fixed set of **location** patterns, and an **ignored list** with one reason per entry (secret or account state, program files, cache, derived pile, the tool's lock files).
- The root honours the harness's own home variable where the harness defines one (`CLAUDE_CONFIG_DIR`, `CODEX_HOME`, `HERMES_HOME`); otherwise it is the harness's fixed path under `$HOME`. Polylogue has no setting of its own for any path.
- Ignoring a harness's credential files is not filtering content: they are account state, not AI work.

### Every file in one class

| Class | Meaning | What happens |
|---|---|---|
| `collected` | matched by a location | stored as bytes (or row dumps), and parsed when the origin has a parser |
| `ignored` | matched by an ignored entry | not stored; counted with its reason |
| `unrecognized` | matched by neither | not stored, not parsed; counted per root, directory pattern and extension, and shown in status |

- **Collecting does not wait for a parser.** An origin has two separate switches: *collect*, on for every declared location, and *parse*. A declared origin without a parser stores its bytes from the first day and is shown as `no_parser` ([archive.md](archive.md#parsing)).
- **Fail closed.** A collected file that fails its origin's shape check is `refused`, with the reason, as a parse-state fact; it is never read as another origin's file. A Claude Code transcript found in Drive is not ingested as a Claude Code session, because it would duplicate the real one.
- **A bad record is not a bad file.** A malformed line in a JSONL file is counted unreadable and the complete records after it are still read; a parser error stops that parser's rows at its last good record while the bytes keep being collected.
- Hidden directories are entered only when a location names them. Symlinked roots resolve once. An unreadable directory is a reported state, never an empty result. A `.git` directory inside a root is ignored with its reason.
- **Discovery follows source changes.** Full walks run at startup and when the watcher reports overflow or an error. Other events schedule affected locations; idle time does not trigger a whole-tree walk. A pass that fails with the same errors as the one before it is not retried at once: the loop waits for a change (a file event, a request, a stop), and a run until idle ends with the errors reported. Collector schedules and live-file checks continue independently.
- **One file never fails a pass.** A file that changes after its pass planned it (a session being written) is planned again from what it holds then. One that only grows while it is read is read up to the length planned, since bytes appended past it leave that range as it was; the next pass reads the rest. One replaced or changed in place before it is read is planned again by the next pass, which starts at once, and is no error of the pass. A file that cannot be read is that file's error. The pass stores and reads every other file; only a failure of the archive or its store ends a pass.
- **The coverage table** `(root, class, pattern or reason, files, bytes, newest mtime)` is computed from the discovery walk the daemon already performs (the start-up walk plus file notifications). It is an index fact, rebuilt with the index; there is no separate scan. An entry is its class with its pattern or reason: the built-in ignore rules that match any path (symbolic links, special files, user projects under a tool's home) share the pattern `**` and are separate entries by reason. A refused coverage report is that pass's error; the files the walk found are still stored and read.

### Live origins

Paths are relative to each root. "Parse" says what becomes sessions, documents or events; everything collected is stored as bytes whether or not it is parsed.

**Claude Code** (root `$CLAUDE_CONFIG_DIR`, default `~/.claude`)

| Location | Parse |
|---|---|
| `projects/*/*.jsonl`; `projects/*/*/subagents/**/agent-*.jsonl`; `subagents/workflows/*/journal.jsonl` | transcripts, subagents, workflow journals |
| `*.meta.json`, `tool-results/*`, `memory/*`, `todos/`, `tasks/`, `plans/`, `file-history/`, `jobs/**` | side files of their session ([sessions.md](sessions.md#sidecars-and-linked-materials)); todo and plan items in full |
| `projects/*/*/workflows/*.json`, `workflows/scripts/*.js` | workflow run snapshots as facts; scripts as documents |
| `projects/*/sessions-index.json` | competing title snapshots |
| `projects/*/bridge-pointer.json`, `projects/*/*/remote-agents/*.meta.json` | remote-run events and spawn links |
| `history.jsonl` | prompt events and paste records |
| `paste-cache/*.txt` | paste bodies |
| `teams/*.json` | team membership and messages as events |
| `usage-data/*`, `history-summaries/*.md` | documents attached to sessions |
| `shell-snapshots/*.sh` | session context documents |
| `sessions/*.json` | live-state records (process to session) |
| `debug/*`, `mcp-logs/**` | bytes only |

Ignored: `.credentials.json`, `.claude.json`, `mcp-needs-auth-cache.json` (secrets or account state); `plugins/`, `skills/`, `agents/`, `hooks/`, `statsig/`, `telemetry/`, `cache/`, `ccusage/`, `stats-cache.json`, `backups/`, `chrome/`, `ide/` (configuration, program files, caches); the user's own projects that happen to sit under the root. A user project is a file no location names that lies inside a git working tree below the root (a `.git` in the root itself does not count). A file a location names is collected whatever git metadata sits above it, so a projects folder kept as a git repository is still collected; only its `.git` is ignored.

**Codex** (root `$CODEX_HOME`, default `~/.codex`)

| Location | Parse |
|---|---|
| `sessions/**/rollout-*.jsonl` and `rollout-*.jsonl.zst`; `archived_sessions/**` | sessions. A compressed rollout is the same input as its plain form: identity is the native session ID and the machine, never the path or the content ([archive.md](archive.md#stored-inputs)). When Codex replaces a plain rollout with its `.zst`, the decoded bytes continue the stored input and any lines written since are appended; the compressed file's own bytes are kept too. Every zstd frame in the file is decoded. A truncated or undecodable file is reported and stores nothing until a complete file is there |
| `history.jsonl`, `session_index.jsonl` | prompt events, titles |
| rows of the `state_*`, `goals_*`, `memories_*` and `queue_*` databases | goals and memories as events; queue rows stay in the stored export |
| `memories/**/*.md` (not `memories/.git`) | memory documents |
| `attachments/**`, `shell_snapshots/**`, `ambient-suggestions/**` | attachments, context documents |
| files named by `local_images` in a rollout line | attachment bytes, read when the line is observed; the opened descriptor must be a regular file, and a nonregular source is reported unreadable without waiting for a pipe writer ([sessions.md](sessions.md#attachments)) |

A rollout whose first complete record does not state a session ID is kept under its location and reported as refused. Its bytes and the refusal remain available after the source disappears. An incomplete first record waits for another read when the file grows.

Ignored: `logs_2.sqlite` (diagnostic log); `thread_history_*.sqlite` (a derived copy of the rollouts); `sqlite/codex-dev.db` (developer database); `worktrees/`, `packages/`, `plugins/`, `vendor_imports/`, `skills/`, `cache/`, `tmp/`, lock directories; `auth.json`; configuration files; backup tarballs left in the tree (importable by hand).

**Gemini CLI** (root `~/.gemini`): `tmp/*/chats/session-*.json` and `.jsonl`; subagent `chats/<id>/*.json`; `tool-outputs/session-*/*` (overflow sidecars, whose full text replaces the truncated inline text); `logs.json` (prompt log); `history/*` checkpoints kept as git refs and objects, so a checkpoint ID in a transcript resolves, with checkpoint commits as events. Ignored: credential files, `skills/`, settings, `*.partial`. The `antigravity*` subtrees belong to Antigravity.

**Antigravity**: everything under `~/.gemini/antigravity/` (`conversations/*.pb`, `brain/**`, `annotations/*.pbtxt`, `implicit/*`, `code_tracker/**`, `knowledge/**`, `user_settings.pb`); `~/.gemini/antigravity-cli/` (`conversations/*.db` and `conversation_summaries.db` as rows, `history.jsonl`, `brain/**` including `.system_generated/logs/*.jsonl`, `annotations/`, `implicit/`, `knowledge/`, `jetski_state.pbtxt`); and `~/.config/Antigravity/User/{globalStorage,workspaceStorage/*}/state.vscdb` as rows. Ignored: the OAuth token, `bin/`, `builtin/`, `cache/`, `updater/`, lock files, `crashes/`, MCP schema caches, `log/`. Its readers, from the plainest format to the hardest:
1. CLI transcripts `brain/<id>/.system_generated/logs/transcript_full.jsonl`: native step numbers identify messages within their conversation (`<id>:step-<idx>`), so different conversations cannot collide. User prompts, model prose/thinking and calls/results retain their line citations; system steps remain events. Results pair in source order with the latest planner's calls, and incomplete or orphan results remain explicit;
2. CLI trajectory stores, decoded with descriptors from the installed language server;
3. summaries and prompts (`conversation_summaries.db`, `history.jsonl`): summary rows offer their titles, workspace directories and explicit parent links; the complete row remains a source-timed event. Summary previews are not messages, and last modification is not session start. Parent links remain independent of the session-kind label: under the shared merge rules, a timed transcript's `main` label can coexist with the summary's explicit `spawn` link;
4. brain artifacts, as versioned documents of their conversation;
5. IDE `.pb` conversations, through the language server's own RPC on a Unix socket or with a per-run token, never an unauthenticated loopback port; without the server, the session exists with its metadata and an event saying the transcript could not be read;
6. `state.vscdb` rows.

**Hermes** (root `$HERMES_HOME`, default `~/.hermes`, and each `profiles/<name>/` as a full home): `state.db` and `verification_evidence.db` as rows, every column; `sessions/session_*.json`, `sessions/saved/*.json`, `sessions/request_dump_*.json`; `observability/nemo-relay/atif/*.json`, `atof/*.jsonl`, `usage.json`; `pastes/*`, `.hermes_history`, `memories/*`, `checkpoints/*`, `cron/*`; `SOUL.md` and `config.yaml` as context documents. Ignored: `.env`, `auth*`, `hermes-agent/` (the program's checkout), `bin/`, `skills/`, caches, `logs/`, `sandboxes/`, `processes.json`. JSON snapshots and `state.db` sessions merge by native ID: the snapshot is the record and live events enrich it. A session is `<native>@profile-<name>`, where the name is the profile's (`default` for the home itself), so moving the home or renaming the machine keeps every session ID.

**Grok CLI** (root `~/.grok`): every file of each session directory `sessions/<cwd>/<id>/` (`chat_history.jsonl`, `updates.jsonl`, `events.jsonl`, `rewind_points.jsonl`, `usage.json`, `summary.json`, `signals.json`, `tool_definitions.json`, `prompt_context.json`, `resources_state.json`, `plan_mode.json`, `subagents/*/*.json`, `recap_requests/*.json`, `goal/*.md`, `terminal/*.log`); `sessions/*/prompt_history.jsonl`; `memory-v2/**` (Markdown, and SQLite as rows); `memtrace/**`; `worktrees.db` as rows. Rewind points are branch events, subagents are linked sessions, terminal logs are tool-output sidecars. Ignored: auth files, program and plugin directories, caches, docs, lock files.

**`llm` CLI** (`~/.config/io.datasette.llm/logs.db`, as rows): a session per `conversation_id`, with prompt and response messages, model, tokens and attachments.

**`pi`** (`~/.pi/agent/sessions/**/*.jsonl`): collected, and shown as `no_parser` until a parser exists.

**Other producers.** Claude Code web runs are their own origin, `claude-code-web`, once a real transcript file exists. Codex Cloud tasks are origin `codex-cloud` with their own ID namespace (`task_e_…`), from `codex.json` in ChatGPT exports and from the extension.

## Hooks

Hooks are a capture channel: every event each harness emits is recorded whole. Hook-only facts (permission answers, pre-compaction context, paste bodies as typed, process identity) exist nowhere else. Which events are declared and how they are installed (`polylogue install HARNESS`) is in [installing.md](installing.md#installing); what context hooks bring to a session is in [agent-context.md#brief](agent-context.md#brief).

- **Every event is installed and spooled.** Claude Code's full set (including `Notification`, `PermissionRequest`, `PermissionDenied`, `PostToolUseFailure`, `SubagentStart`, `SubagentStop`, `PreCompact`, `PostCompact`, `Elicitation`, `ElicitationResult`, `CwdChanged`, `FileChanged`, `WorktreeCreate`, `Setup`, `SessionEnd`), Codex's full set including `PermissionRequest`, Gemini CLI's (`SessionStart`, `BeforeAgent`, `BeforeTool`, `AfterTool`, `AfterAgent`, `Notification`, `PreCompress`, `SessionEnd`), Hermes's lifecycle events, and Grok CLI's if it offers any. The set is taken from each installed harness version by probe.
- **Recording.** `polylogue internal hook EVENT --harness HARNESS`, the entry `install` writes into each harness's settings, appends the payload, whole and with no size cap, as one line of the per-process spool file under `$XDG_STATE_HOME/polylogue/hooks/`, and syncs it (`fdatasync`) before returning. Recording opens no database and contacts nothing. A harness treats a non-zero hook exit as a refusal (Claude Code blocks the tool call on exit 2), so the hook always exits 0: an unwritable spool, an unreadable configuration file, a deleted working directory or an unknown harness name writes nothing to stdout and one line to stderr. A failed spool write sends one native journal entry naming the event and session without waiting for an answer. If the initial event cannot be recorded, the hook returns before requesting context. Session-start context outcomes use this same spool, with the cause when no brief was given ([agent context](agent-context.md#the-brief-for-a-directory)). A recording entry takes under 5 ms typically and 15 ms at p95 when the state directory is on a disk that is not saturated by bulk writes; a sync behind heavy writeback can take hundreds of milliseconds, so deployments keep the state directory off bulk storage.
- **The spool line** is the payload plus:

  ```json
  {"harness": "claude-code", "event": "PreToolUse", "at_ms": 1791588190000, "session_id": "…",
   "proc": {"harness_pid": 41237, "harness_start": 98213377, "boot_id": "7b1e…", "cwd": "/home/op/morrow"},
   "inherited": {"claude_code_session_id": "…"}}
  ```

  - `proc` is the **process identity**: `harness_pid` is the hook's parent (the harness runs the hook directly), `harness_start` the process start time, `boot_id` the kernel's boot ID. PID reuse and reboots never let a new process inherit an old session.
  - `inherited` records another harness's session variables present in the environment (a Codex run launched from a Claude Code shell). It records how a delegation was launched, never the child's identity. No other environment variable is read or stored.
  - At session start the line carries `git_identity` (`common_dir`, `root_commits`, `head`, `branch`) and `repository`, read by common's `git-identity` within the hook's one 1.5 s context deadline: the roots are every root reachable from `HEAD`, sorted, shallow-clone boundaries left out (the set Polylogue's repository lookup and Sinex's shell-session capture also record), and `repository` is the `origin` URL with any user and password removed. Roots are cached per working tree under the cache directory. An unchanged HEAD and shallow state reuse them; a fast-forward adds only new roots, and unrelated history or changed shallow boundaries require a full walk. Branch and remote metadata are read afresh. The cache is disposable and a failed write never discards the observation. A shallow clone whose history reaches no root, a missed deadline, or no `git` on the hook's PATH records the reason in `unstated` instead.
  - Hermes's `conversation_history` is left out of the line, because its transcript is read from `state.db`.
- **A hook event belongs to its session and never creates one.** The daemon reads spool files like any growing file; a hook line keeps its identity across re-storage.
- **Hooks complete transcripts** where the transcript is incomplete: `PreCompact` gives the pre-compaction context and `UserPromptSubmit` gives paste bodies as typed. `PostToolUse` records the reported tool event.

## Tool databases

- **A tool's SQLite database is captured as committed rows, never as a database image.** Each capture keeps rows the tool later deleted (a row seen once stays in the archive) and writes no export side file. Uncommitted write-ahead-log rows are not read.
- Captured: Codex `state`, `goals`, `memories`, `queue`; Hermes `state.db`, `verification_evidence.db` and each profile's `state.db`; Antigravity `conversations/*.db`, `conversation_summaries.db` and each `state.vscdb`; Grok `memory-v2/**/*.sqlite` and `worktrees.db`; `llm` `logs.db`.
- Each row-dump header keeps the table's SQL and, from the same read transaction, its column names and order, declared types, nullability, primary-key order, hidden or generated columns and `WITHOUT ROWID`. The schema tools report this separately from the JSON types seen in rows, and leave SQL and default expressions out ([archive.md](archive.md#schema-tools)).
- A table whose schema no parser knows keeps its raw rows readable and is reported as [format drift](archive.md#format-drift).
- **A database read again only when it changed.** A read compares every row with the rows already captured and stores only those that changed. Each completed read, including one that found no changed row, records an observation with the database file's facts and its write-ahead log's (or that it had none); while the files keep those facts, the database is not read again, after a restart too. Another machine, another path or changed facts mean a full comparison.

## Browser host

The extension sends logical messages to `polylogue internal browser-host` over Chrome native messaging. Every frame carries `version: 2`. A message up to 64 KiB is one frame; larger messages travel as ordered `part` frames with `sequence`, base64 `data` and `last`, in either direction, and are reassembled before being read. Parts stay below Chrome's 1 MB limit per frame. A different version receives a typed `native_version_mismatch` refusal asking the caller to reload the extension.

The host accepts three messages:

- `call {operation, arguments, request_id}` forwards the shared request to the daemon and returns its whole answer, including any typed failure and its reason. The operation table owns these calls ([interfaces.md](interfaces.md#operation-table)).
- `capture {capture}` carries the exact capture JSON as a string. The host appends those bytes to the conversation's capture log, syncs the file, its provider directory and the spool directory, and then acknowledges it, even with the daemon down. The capture log's format and writer lock are the same as any other extension capture.
- `spooled {provider, ids}` returns which conversations have a capture log in the spool, in the order asked. The extension combines this with `internal conversations` to distinguish a capture saved locally from one the daemon has read.

Each reply is an `answer` message carrying the shared answer. `internal capture-files` uses the capture parser to list a conversation's attachment names; the extension combines that reply with `internal stored-files` to find the bytes it still needs to download. If the daemon does not answer, the conversation still goes to the spool and file discovery waits for a later capture. A setting read supplies the extension's daemon and chat-action state. `internal reader-url` supplies the running reader's authenticated address for Open in Polylogue. When no reader is served, it refuses with `reader_unavailable` and the typed reason `reader_not_running`; the extension shows the refusal's sentence.

## Import

`polylogue import PATH… [--origin NAME]` takes files, directories, ZIPs and tar, tgz and tar.zst archives, including nested containers and multi-part Takeouts (each part is its own container; members join by path). The CLI, MCP (`import` with `paths`, in `polylogue_write`) and HTTP all take paths on the daemon's host. Every member streams; there is no size limit. A provider name instead of a path (`polylogue import chatgpt`) has the browser extension fetch that web chat's history ([extension.md#importing-a-provider](extension.md#importing-a-provider)); a file literally named like a provider is written `./chatgpt`, and `--wait D` returns when that job finishes or the time passes (`polylogue import chatgpt --wait 10m`; [interfaces.md](interfaces.md#steps)). Files of annotation lines are imported the same way ([annotations.md#annotation-lines](annotations.md#annotation-lines)). A page's working copy (`flaky-ci.md`, written by `read --to`) is checked in with `polylogue import flaky-ci.md`, with `--withdraw-missing` to withdraw the slots whose markers were deleted from it ([page-working-copy.md](page-working-copy.md#check-in)).

- **Recognition by declared export structure, never by sniffing.** A package is recognized by its structure and its members are parsed by that package's origin; nothing is guessed across origins. Each origin declares one recognition rule, used by import and by the live path alike; an import gives it the file's whole top-level structure, so field order and a long first value do not change the answer. A path that matches no declared structure is refused, with the structures tried, and is not stored: the command replies at once, so refusing loses nothing, while a stored unknown file would have no origin and no reader. Members of a recognized package that no parser reads (`chat.html`, `.dat` assets) are stored as package members. A file whose layout or `--origin` names a declared origin that has no parser yet is stored as that origin's input, as the live path stores it, and reported as `no parser yet`; it is read when the origin gains a parser.
- **Original containers are kept whole**, and their members are also stored decoded, addressed as `container!/member`. A provider export member named by its provider file ID is kept as an attachment, including when the attachment itself is an archive. Members deduplicate across overlapping exports; containers do not.
- **Importing the same bytes again changes nothing.** A newer export merges by field ([sessions.md](sessions.md#one-conversation-through-several-routes)), so a field only the older export had survives.
- **An import has no parse path of its own.** It commits its inputs to the record, and they are read into the index like any other stored input: by upkeep, or by a build when that is less work ([architecture.md](architecture.md#one-path)). History brought in as hundreds of thousands of small files, or as one very large export, is ordinary input.
- An encrypted or corrupt member refuses its whole container before anything is stored. A file opened as a ZIP package by name must start with ZIP magic. Import also refuses a member whose decoded size is more than its container declared, a container that holds its own bytes (directly or through containers inside it), and a container whose declared decoded size does not fit in the free staging space. A container repeated at another path in the same import is kept as an original and expanded once.
- In an export, one element the parser cannot read (nested deeper than the 128 levels the JSON reader builds into a value, for example) is counted as unreadable on its own; the other elements still parse, and the export's bytes stay stored. A key an element repeats is counted as an unread field (`path (repeated key; earlier value)`): readers see only its last value.
- The report lists `imported`, `already_stored` and `refused` (with the structures tried), and per package the counts of sessions, documents and assets once its inputs are read into the index; its inputs are parsed and derived before it answers, so they are readable at once and `readable` says `now`; where that waits for a build, `readable` is `after_build` with the build's progress and the reply is degraded until the build installs. Bytes the request stored are `imported` even when the request is read again because a build installed under it. With `--dry-run` the daemon recognizes and classifies every member and stores nothing; `--explain` says how the line is read and which structures it would try, reading no file ([terminal.md#explain](terminal.md#explain)).

When importing files into the archive, the reply and command exit status follow what was stored.

| Result | Outcome | Exit status |
|---|---|---|
| All named files stored or already present, with new files stored | `ok` | 0 |
| Everything already stored | `ok` with a note | 0 |
| New files stored and some refused or failed | `degraded` with a reason for each refused or failed path | 5 |
| No new files stored and at least one path refused or failed | `error` with the paths and reasons | 2 for an input the caller must change, 4 for `not_found` or `io` |
| Stored, not yet readable (`readable.when` is `after_build`) | `degraded` with index rebuild progress | 5 |

A mistyped `--origin` gives one `bad_value` error listing the importable origins. Failed and partial file imports appear in the daemon's recent log. The text output names the stored paths and gives the reasons for refusals. With `--json`, these facts are in the reply. When an import has a next call, the text output gives a command that runs it.

| Structure | Recognized by | Origin |
|---|---|---|
| ChatGPT export, every vintage | `conversations.json` or `conversations-NNN.json` shards, with `chat.html`, `user.json`, asset members, `conversation_asset_file_names.json`, `library_files.json`, `codex.json` | `chatgpt`, `codex-cloud` |
| ChatGPT single conversation | the conversation API shape (`mapping`, `current_node`, `conversation_id`) | `chatgpt` |
| claude.ai export | `conversations.json`, `users.json`, `projects/`, `memories.json`, `design_chats/` | `claude-ai`, `claude-design` |
| claude.ai single conversation | the `chat_messages` and `uuid` shape | `claude-ai` |
| Grok account export | `prod-grok-backend.json`, shape-checked while streaming | `grok` |
| Google Takeout | `Takeout/My Activity/Gemini Apps/`; `Drive/Google AI Studio/*`; `Gemini/` (Gems); `NotebookLM/` | `gemini-app`, a session named by the conversation ID in its entry's caption link (an entry with no link is a session of its own, named by a hash of the entry); `ai-studio`, each prompt named by its content (its first chunk), never by its file name, which is its title; Gems as documents; NotebookLM stored without a parser. Other Takeout products are refused members, counted and not stored |
| AI Studio archive | prompt JSON members, extensionless ones included | `ai-studio` |
| OpenRouter chat-room export | its JSON shape | `openrouter` |
| OpenTelemetry GenAI | OTLP-JSON `resourceSpans` | `otel-genai` |
| OpenAI chat array | only with `--origin openai-chat`, since the shape has no reliable structure | `openai-chat` |
| aider | `.aider.llm.history` beside `.aider.chat.history.md` | `aider`: the LLM history is the session, the Markdown a companion document |
| Polylogue bundle | the bundle manifest ([reading.md](reading.md#writing-out)) | the bundle's own records |
| Hook spool lines in the older line shapes | only with `--origin hook-spool`: per-session `.jsonl`, `carriers/<harness>/<date>/<pid>.ndjson` | the payload's origin |
| Browser captures in the older envelope | only with `--origin browser-capture`: `browser_llm_session` v1 envelopes | the payload's origin |
| Python Polylogue's vector stores | only with `--origin python-vectors`: a sqlite-vec `message_embeddings` store, read through its backing tables | vectors ([search.md](search.md#what-is-embedded)) |
| A live-origin file outside its root | only with `--origin NAME`, or when the path keeps the harness's own layout; an origin with no parser yet is stored and waits for one | that origin |

### Earlier stores

There is no importer for a store written by an earlier build of Polylogue or by Python Polylogue ([archive.md](archive.md#evolution)), except for Python Polylogue's paid vectors (`--origin python-vectors`, above). What only such a store still holds is written out as ordinary files, in each tool's own folder layout, and brought in with `polylogue import FOLDER`: a tool file outside its root is recognized by its layout and stored as an imported copy, one more route of its session ([sessions.md](sessions.md#one-conversation-through-several-routes)). A file that only grew needs only its newest bytes; each differing version of a rewritten file is its own file and its own input. Authored records move as a Polylogue bundle ([primary-mode.md](primary-mode.md#moving-user-state)).

Aider's LLM log holds snapshots of the context sent and the response returned, with no native session or message IDs: the first recorded request fixes a content-derived session ID, messages are named by their content, and a prefix sent again is not read again. Its times carry no zone, so they stay unknown with the original strings kept. The paired Markdown is a companion document.

Hand-made Markdown chat dumps are not a structure: a flattened transcript is a degraded form of a conversation that has a real source.

## AI Studio through Drive

AI Studio stores its prompts in Google Drive, which is a known location like a harness's home directory.

- The collector follows Drive's changes feed from a start page token, not folder relisting, and writes every Drive revision of each prompt, oldest first, into the spool. Its first sync, and the first after the folder itself is removed, takes a start page token and then lists the folder once; every later sync reads the changes since the saved token and applies them to its listing (`files.json`, each prompt's Drive metadata by file ID). A prompt added to or changed in the folder takes its current metadata; one deleted, trashed or moved out keeps its entry, marked `removed`. The token is saved only once the sync has applied its changes and fetched the revisions they call for, so a sync that fails reads the same changes again.
- The revision list is read through every page and ordered by each revision's modification time read as an instant (Drive writes fractions of varying width). A revision list reply without its `revisions`, or with a time that does not read, is an unreadable reply and the sync fails; it is never an empty list. The cursor holds the accepted revision and its Drive modification time; when Drive has pruned that revision, the walk continues at the first revision newer than it.
- A fetched revision's bytes are kept until the record holds the complete body at that prompt's stable input path; the revision cursor advances only after that. A retry or a restart stores the same bytes again and changes nothing.
- `driveDocument` and `driveImage` attachments are fetched by Drive ID as attachment bytes. Revision lineage and `branchParent` edges are recorded; a Drive mode that cannot give lineage fails closed.
- **Sign-in.** `polylogue install drive` runs the one-time Google OAuth consent. It needs an OAuth client registered under the operator's Google account, read from the setting `drive.client_id`; until that is set it refuses with the instruction and the line that sets it, `polylogue set drive.client_id CLIENT_ID`.
- The collector reads only the archive’s `drive-token.json`, written by `install drive`. The collector is on whenever the user has authorized that token; the one-time consent is the only gate, because a known location is never dark behind a setting. Without a token, collection stays idle and status shows "not authorized" with the line to run, `polylogue install drive`; that absence is not an ingestion fault. An unreadable or invalid token remains a collector error.
- Historical `drive.ai-studio` settings remain readable and replayable as authored history. They no longer control collection; new writes direct the user to `install drive` or `install drive --remove`.
- A prompt deleted in Drive keeps its archived revisions and is marked gone ([archive.md](archive.md#no-deletion)): the newest listing that names it marks it `removed`, and its session reads `source: gone`. A prompt restored to the folder is present again.
- The collector's prompts are named by Drive file ID, which the listing and `branchParent` also use. A Takeout or downloaded copy carries no file ID and is named by its content (its first chunk). Each Drive revision records, as an alias of its session, the content name a copy of that revision would carry (`session_alias`), so a copy taken at any revision the collector saw joins its Drive session.
- **A copy is one session with its Drive prompt.** A copy's reading states its own content session and messages, and stays as parsed. The Drive prompt records the copy's content name as an alias, so the copy's session, its messages (each the one with the same content and place in the Drive prompt), its tool calls and the events it names by their content are the Drive session's: the shown session is `ai-studio:<file id>` with the copy's readings composed by route precedence ([sessions.md](sessions.md#sessions)). This is decided from the alias rows present, so the Drive prompt arriving before or after its copies gives the same rows, and nothing is read again or moved. A content name that two Drive prompts record (a copied prompt) names neither, and its copies are a session of their own, whichever prompt arrived first.

## Source clocks

Parsers keep each source time with its quality, its precision and any interval end the source declares. Whole epoch seconds and offset timestamps without fractions have second precision; fractional timestamps, epoch milliseconds and nanoseconds keep their subsecond instants. A missing or unreadable time has no instant and unknown quality. Capture observations and file times keep their own quality and never fill in a missing message time.

Where a format records events rather than a session creation time, the earliest record or activity time is the session's start, and says so. An OpenTelemetry span event keeps its start and end as an interval; its output message uses the call's start and records that choice with both original fields in `otel_span_clock`. Input history without message times stays unknown.

Gemini activity times keep their original text and zone. Explicit offsets are resolved without the host's time zone; ambiguous zone abbreviations stay unknown. AI Studio chunk times belong to the chunks; the Drive listing's file time is file metadata, separate from message times.

## Capture status

`polylogue status` (CLI, MCP, HTTP and web) says whether everything is being captured, in one place. Its capture section holds:
- **per origin:** state (`live`, `collector off`, `not authorized`, `no parser yet`), newest input time, input and session counts, refusals, and format drift;
- **per root:** the coverage classes with their reasons and counts;
- **hooks:** per harness and event, whether it is wired and whether events arrived recently;
- **extension:** the native-messaging host manifest present and pointing at this binary; per provider, the last capture and its state; running provider imports;
- **Drive:** token state, last change token, pending revisions;
- **imports:** packages imported and paths refused by the last import;
- **field accounting:** `unused` paths per origin and parse version ([archive.md](archive.md#parsing)).

Its figures and verdicts use the shared status format of [command-surface](https://github.com/Sinity/polylogue-sinex-common/blob/master/crates/command-surface/README.md); the rest of status is in [operations.md](operations.md#status). The health report raises an item for every non-empty `unrecognized` class, every refusal, every unwired hook event, and every origin silent for longer than its own usual gap. Status is computed on request from stored facts; nothing is checked on a schedule.

### Prepared Python Polylogue vector sources

A prepared addressed SQLite vector store uses `application_id = 0x504c5650`
and `user_version = 1`. Its existing `message_embeddings_meta` address, model,
dimension and purchase-time columns keep their meaning. Every row additionally
has `request_text TEXT`, the NFC UTF-8 preimage of its historical request.
The reader validates the domain-separated SHA-256 request address against the
model and preimage before deriving its NFC text lookup identity. Null, non-NFC,
mismatched or unsupported prepared vectors are refused. Unmarked original
Python Polylogue stores remain supported without inferring a text hash from an address.
Unknown purchase times remain unmatched under either source layout.
