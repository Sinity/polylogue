# Completion

Tab completes every position of a line, with descriptions and real values from the archive, so the grammar of [interfaces.md](interfaces.md) is learned by pressing Tab rather than from a manual. This page owns what completion offers at each position, the catalog file it reads, and how a shell is registered. The picker and its Alt-p key are [terminal.md](terminal.md#the-picker).

## How completion works

- **It runs in the CLI process,** against the catalog file the daemon publishes, so it never waits on the daemon: at most 50 ms per keystroke with the daemon up or down ([latency.md](latency.md#latency-classes)). The `internal complete` operation serves MCP `completion/complete`, HTTP and the web query box from the same catalog. MCP supplies the whole query in `argument.value`, with an optional UTF-8 byte offset in `_meta["polylogue/position"]` (the end by default); each completion value is the whole query with the replacement applied, and `_meta["polylogue/completions"]` carries the typed replacement range, kind and help.
- **The line positions** (the first word, verbs, a verb's words, options) come from the line parser's tables ([interfaces.md](interfaces.md#verb-words)). **Positions inside a query** come from the shared `query-language` completer: it takes the whole query text and a cursor offset and returns candidates with their replace range; Polylogue supplies the catalog.
- **Never** does completion run a search, open the archive or write.
- **Descriptions** (zsh, fish): `repo:sinex  4,927 sessions`; `7f3a2c10  Back off on 429 · 2h · sinex`; `group by  one row per value of a field`. Bash shows values only.
- **Order:** headings in the order below, grammar candidates in declaration order, values by count descending, sessions by last activity. There are no fixed caps: the catalog holds every session and value, and the shell shows what fits.

## Positions

| Typing after `polylogue ` | Candidates, with descriptions |
|---|---|
| the first word | under headings: **commands**, each with its one line; **verbs on their usual rows** (`read  pick from recent sessions`, `brief  for this directory`, `resume  pick a session that ended open`); **saved queries** (`waiting  sessions waiting on you · 2`, yours marked `yours`); **units**; **recent sessions** (`7f3a2c10  Back off on 429 · 2h · sinex`) |
| `'` | units (`tool_calls  every tool call, with outcome`), fields for sessions, references (`here`, `latest`, `self`, `query:`, `result:`, `page:`) |
| `'repo:` | values with counts (`sinex  4,927 sessions`): origins that have sessions, repository, cwd by path component, model, harness, tool, tag and `tag:key:`, branch; closed sets; after `query:`, saved names with parameters and their one line; after `result:`, kept result IDs with names |
| `'repo:sinex ` | fields valid for the unit, `session.` fields, `exists`, `count(`, `seq(`, `query:`, `result:`, `similar:`, `NOT`, `\|`, and the count preview `→ 212 sessions` |
| after a comparison field | operators; time words (`today`, `7d`, `this week`, month names) |
| `'repo:sinex \| ` | transforms (`group by  one row per value of a field`); after `group by` and `sort by`, the unit's fields and aggregate columns |
| inside `with` | child units and their fields |
| `'repo:sinex since:7d' ` | the verbs valid for the selection's unit, grouped **look** (`read`, `pick`, `brief`, with the brief's form as its description: `brief  take this session over` or `brief  recall what is known`), **annotate** (`note`, `tag`, `save`; on annotations, pages among them, also `edit  open in your editor, or give the new text`, `withdraw`, `judge`), **act** (`send`, `answer`, `interrupt`, `resume`); then the listing options |
| `… note 'x' --as ` | kinds with what each is for (`decision  a choice made, with the alternatives`) |
| `… tag ` | tags in use with counts (`rate-limits  14`) |
| `… judge ` | `accept`, `reject`, `defer` on annotations; on several rows, their row numbers or the labels `A`, `B`, and `tie`, `incomparable`, `abstain`, `insufficient-evidence` |
| `… save ` | your saved names, to save again |
| `… save page:NAME --as ` | the forms valid for the selection's rows, one line each (`card  title, origin, when, cost, outcome`; `excerpt  the exact text with its source line`; `value  one number with its denominator`) ([pages.md](pages.md#forms)) |
| `… --brief ` | `off`, `contents`, `full`, one line each (`full  the page's text goes into briefs for its scope`) |
| `… answer ` | `allow`, `deny`, and for a waiting question its text as the description |
| `… --section ` | the headings of the page named in the line, in page order |
| `… --template ` | templates with their sections (`investigation  Goal, Assumptions, …`) |
| `… resume --` | the resume options, `--fork  use the harness's own fork` and `--anyway  reopen it in a second client` among them |
| `… resume --in ` | harnesses installed (`codex  continue this work in a new Codex session`) |
| `… read --` | the read options, `--follow  keep printing what happens, like tail -f` among them |
| `… read --view ` | views valid for the unit, one line each: `contents` on a page, `trace` on a message, span, block, tool call or file line |
| `… --to ` | `clipboard`, `browser`, `pager`, then file paths |
| any position, `--` | the line options, `--explain  how this line is read; runs nothing` among them |
| `@` | rows of the last list on this terminal, with their titles; beside them, member aliases with their sessions and roles (`reviewer  codex:019d0f2a · reviewer`) |
| `page:` | current page names with their titles and open-item counts (a renamed page's earlier names still resolve but are not offered); after `page:template/`, templates |
| `page:NAME#` | the page's heading slugs in page order, then its slots by short ID with their kind and first words (`7q2m  finding · weekday_offset() compared…`) |
| a reference position | sessions by short ID with title; typed prefixes |
| `set ` | keys with current values, then the key's allowed values |
| `start 'x' --in ` | harnesses and web chats with what each does |
| `start 'x' --` | the start options, `--submit  send the new web chat rather than draft it` among them |
| `import ` | providers first, then file paths |
| `install ` | targets: harnesses, `chrome`, shells, `drive` |
| `help ` | verbs, commands and topics |

- **Which rule a first word meets.** When the typed first word matches no command, verb, saved query, unit or reference shape, zsh and fish show the message line `→ searches the word "chro"`; when it equals a name, the message says which rule took it (`→ saved query waiting (shipped): sessions waiting on you`). The rules are [interfaces.md](interfaces.md#the-first-word).
- **Count preview:** when the text before the cursor is a complete query, zsh and fish show one line that is not inserted, `→ 1,234 sessions`. The count is asked with a 30 ms deadline and left out when not available in time, never estimated.
- **Title matching** at a reference position (zsh, fish): a word that is not a prefix of any ID is matched against session titles with the fuzzy matcher, and those sessions' IDs are offered with the title as description.
- **Quoting:** completing the first word of a query inserts the opening quote, so zsh never sees `(`, `|` or `*` unquoted; candidates inside quotes are inserted without closing the quote; the completer rebuilds the query text and cursor offset from the shell's words and the open quote.
- **Daemon down:** candidates come from the last published catalog, marked with the time it was written in zsh and fish (`values as of 2026-01-05T16:09:00Z`); bash has no message channel and shows the values unmarked. Without a catalog file, only grammar candidates.

## The catalog file

- **Location:** `$XDG_CACHE_HOME/polylogue/<archive key>/catalog`, so it survives a reboot before the daemon starts.
- **Contents:** a header (format version, archive key, write position, time written); values with counts per completable field; the verbs valid per unit; tag names with counts; saved query names with parameters and their one line, and the shipped saved queries ([saved-queries.md](saved-queries.md#shipped-saved-queries)); page names with titles, their headings with slugs, their slots' short IDs, templates and member aliases ([pages.md](pages.md)); kept results (ID, name, unit, row count); settings keys and values; sessions (full ID, shortest unique prefix, title, origin, repository, last activity, first prompt), sorted by ID with a second order by last activity. The picker's candidates and `help`'s filled lines read the same file.
- **Format:** a private, versioned SQLite catalog with indexed values and typed metadata, opened read-only with memory mapping. The publisher owns its WAL and shared-memory files and preserves them for offline readers; a daemon-down read changes none of the cache files. Concurrent readers hold a committed SQL snapshot while publication continues. SQLite may coordinate shared-memory read marks inside the live publisher process. The catalog path selects a uniquely named database file through an atomically published pointer. Rebuilding the same source state uses fresh database and journal names so pinned old snapshots cannot mix with the rebuild. Activation retires the cache namespace of the previous pointer. Missing, wrong-archive, obsolete or corrupt catalogs leave grammar-only completion.
- **Freshness:** the writer records what each session contributes to the catalog in `index.db`, in the same transaction as the rows and from that transaction's change summary: how many of its messages name each model and how many of its calls use each tool, and its first typed prompt. Each changed message takes away what it contributed before and adds what it contributes now, and each changed count records the write counter of its transaction (a value the session no longer has stays at 0, so its removal is seen). The publisher subscribes before opening its catalog. On restart it resumes the published position, session contributions and saved-query dependency state from the same catalog transaction, then copies only committed changes after that position. It reuses a cache only for the same archive, index generation, publishing machine and language catalog, with no position ahead of the source. A missing, corrupt or incompatible cache gets a fresh startup snapshot. Ordinary standalone opening retains that pointer while installing generation invalidation before index setup; an actual generation change or a failed open revokes it synchronously. Primary and otherwise unknown scopes revoke their previous pointer immediately during startup. A concurrent offline reader can temporarily hold a read-only WAL mapping in the publisher process; startup retries the writable reopen after it releases that mapping, retaining the same catalog. Corrupt caches, missing journal mappings or database files without write permission get a fresh replacement. On each coalesced post-commit cue it copies what changed since its position: session rows whose own fields changed, the counts the writer changed, removed sessions, and the annotation families (tags and saved queries among them), settings, kept results and query-body dependencies their committed change source names. A refresh runs statements in proportion to the rows that changed, never to the length of a session it touches, and recomputes nothing from the archive. A Sinex lifecycle rewrite replaces its affected vocabulary category; a change of scope or of index build rebuilds the snapshot. Each catalog transaction publishes its data and position together. A commit that changes nothing the catalog holds writes nothing, so the file is rewritten only when its content changes and its "values as of" time is that of the last change. Activation uses the Sinex read-scope switch, which revokes the old pointer synchronously. Completion never shows staged or producer-acknowledged data as confirmed vocabulary.
- **Budgets:** at most 50 ms per keystroke with the daemon up or down, and at most 2 s from a committed batch to visible vocabulary while a writer commits once per second. Measured latency samples and workload records are reported apart from functional test results; a passing test alone is not a timing measurement.
- It is derived and harmless to delete.

For a bounded synthetic measurement, record successful output from the functional matrix, real daemon up/down shell callbacks, and the committed-write visibility fixture:

```sh
cargo xtask test -p polylogue -E 'test(completion_matrix_thirteen_positions_three_shells_empty_and_seeded) | test(completion_latency_reports_real_shell_callbacks_with_daemon_up_and_down) | test(acknowledged_commits_refresh_catalog_values_while_keystrokes_continue)' --success-output immediate
```

The output records the position matrix distribution, ten real CLI callbacks per shell and daemon state, and five setting batches scheduled one second apart. Record the acknowledged commit offsets with the workload. Submit-to-visible samples bound commit-to-visible latency from above; the acknowledgment-to-visible samples are reported separately. Compare the recorded values with the budgets without adding elapsed-time assertions to tests.

## Registration

`polylogue install zsh`, `polylogue install bash` and `polylogue install fish` write Polylogue's completion script into the shell's completion directory; in zsh and fish they also bind Alt-p, which opens the picker in any shell line and inserts the chosen references at the cursor ([terminal.md](terminal.md#the-picker)). Where the script goes and the other install targets are [installing.md](installing.md#installing).

- The script calls the program by the name it was invoked as, so `plg`, `plog` and wrappers bound to another archive complete against their own archive; it registers every installed name, and passes words, cursor index and open-quote state. zsh groups candidates under headings and uses messages for the count preview, the first-word rule and the catalog's age.
- `--program NAME` writes the script for a wrapper on `PATH` that runs Polylogue under another name: `polylogue install zsh --program plg`.
- `--dry-run` prints the script instead of writing it, which is what packagers use: the output of `polylogue install zsh --dry-run` is the zsh script the package installs.
- `--remove` restores the bytes the install replaced.
