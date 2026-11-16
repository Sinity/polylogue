# Completion

Completion uses the [operation table and grammar](interfaces.md) and a
catalog published by the daemon.

## Completion

Completion is advanced and works with the daemon down.

- **It runs in the CLI process,** against a catalog file the daemon publishes, so it never waits on the daemon. The `complete` operation serves MCP `completion/complete`, HTTP and the web query box from the same catalog. MCP supplies the whole query in `argument.value`, with an optional UTF-8 byte offset in `_meta["polylogue/position"]` (the end by default); each completion value is the whole query with the replacement applied, and `_meta["polylogue/completions"]` keeps the typed replacement range, kind and help.
- **The grammar-position completer** is the shared `query-language` function: it takes the whole query text and a cursor offset and returns candidates with their replace range. Polylogue supplies the catalog.
- **Never** does completion run a search, open the archive or write.

| Position | Candidates |
|---|---|
| first word | command names with help, unit names, references (`last`, `latest`, `self`, `@last`, `@N` rows of the last list with their titles), shipped and saved query names, recent sessions by short ID with title |
| condition start | fields valid for the current unit, `session.` fields, `exists`, `count(`, `seq(`, `query:`, `result:`, `similar:`, `NOT`, `-` |
| after `field:` | closed sets; values present with counts (origins that have sessions, repository, project, cwd by path component, model, harness, tool, MCP server, tag and `tag:key:`, branch); `query:` names with parameters; `result:` IDs with names |
| after a comparison field | operators; time words (`today`, `7d`, `this week`, month names) |
| after `|` | stage names; after `group by` and `sort by`, the unit's fields and aggregate columns |
| inside `with` | child units and their fields |
| after a complete query, outside quotes | verbs valid for the query's unit, with help |
| a verb's options | its flags and closed values; `--view` lists views valid for the unit; `--to` targets; file paths for `--to FILE` and `import` |
| a reference argument | sessions by short ID with title, then rows of the last list; typed prefixes |
| `setting KEY VALUE` | keys with current values; closed values |

- **Order:** grammar candidates in declaration order, values by count descending, sessions by last activity. There are no fixed caps: the catalog holds every session and value, and the shell shows what fits.
- **Descriptions** (zsh, fish): `origin:codex  4,927 sessions`; `7f3a2c10  next friday returns the same day · 2h ago`; `group by  one row per value of a field`. Bash shows values only.
- **Title matching** at a reference position (zsh, fish): a word that is not a prefix of any ID is matched against titles with the fuzzy matcher, and those sessions' IDs are offered with the title as description.
- **Quoting:** the completer rebuilds query text and cursor offset from the shell's words and the open quote, and inserts candidates without closing the quote.
- **Count preview:** when the text before the cursor is a complete query, zsh and fish show one line that is not inserted, `→ 1,234 sessions`. The count is asked with a 30 ms deadline and left out when not available in time, never estimated.
- **Daemon down:** candidates come from the last published catalog, marked with its age in zsh and fish (`values as of 16:09 (daemon not running)`); bash has no message channel and shows the values unmarked. Without a catalog file, only grammar candidates.

### The catalog file

- **Location:** `$XDG_CACHE_HOME/polylogue/<archive key>/catalog`, so it survives a reboot before the daemon starts.
- **Contents:** a header (format version, archive key, write position, time written); values with counts per completable field; saved query names with parameters and help; kept results (ID, name, unit, row count); settings keys and values; sessions (full ID, shortest unique prefix, title, origin, repository, last activity, first prompt), sorted by ID with a second order by last activity.
- **Format:** a private, versioned SQLite derived catalog with indexed values and typed metadata, opened read-only with memory mapping. The publisher owns its WAL and shared-memory files and preserves them for offline readers; a daemon-down read changes none of the cache files. Concurrent readers keep a committed SQL snapshot while publication continues. SQLite may coordinate shared-memory read marks inside the live publisher process. The catalog path selects a uniquely named database file through an atomically published pointer. Rebuilding the same source generation uses fresh database and journal names so pinned old snapshots cannot mix with the rebuild. Activation retires the cache namespace of the previous pointer. Missing, wrong-archive, obsolete or corrupt catalogs leave grammar-only completion.
- **Freshness:** the writer keeps what each session contributes to the catalog in `index.db`, in the same transaction as the rows and from that transaction's change summary: how many of its messages name each model and how many of its calls use each tool, and its first typed prompt. Each changed message takes away what it contributed before and adds what it contributes now, and each changed count records the write counter of its transaction (a value the session no longer has stays at 0, so its removal is seen). The publisher subscribes before its startup snapshot, then on each coalesced post-commit cue copies what changed since its position: session rows whose own fields changed, the counts the writer changed, removed sessions, and the assertion families, settings, kept results and query-body dependencies their committed change source names. A refresh runs statements in proportion to the rows that changed, never to the length of a session it touches, and recomputes nothing from the archive. A Sinex lifecycle rewrite replaces its affected vocabulary category; a scope or index generation change rebuilds the snapshot. Each catalog transaction publishes its data and position together. A commit that changes nothing the catalog holds writes nothing, so the file is rewritten only when its content changes and its "values as of" time is that of the last change. Activation uses the Sinex read-scope fence, which revokes the old pointer synchronously. Completion never shows staged or producer-acknowledged data as confirmed vocabulary.
- **Budgets:** the targets are at most 50 ms per keystroke with the daemon up or down, and at most 2 s from a committed batch to visible vocabulary while a writer commits once per second. Keep measured latency samples and workload records separately from functional test results; a passing test alone is not a timing measurement.
- It is derived and harmless to delete.

For a bounded synthetic measurement, keep successful output from the existing
functional matrix, real daemon up/down shell callbacks, and committed-write
visibility fixture:

```sh
cargo xtask test -p polylogue -E 'test(completion_matrix_thirteen_positions_three_shells_empty_and_seeded) | test(completion_latency_reports_real_shell_callbacks_with_daemon_up_and_down) | test(acknowledged_commits_refresh_catalog_values_while_keystrokes_continue)' --success-output immediate
```

The output records the 78-position matrix distribution, ten real CLI callbacks
per shell and daemon state, and five setting batches scheduled one second
apart. Record the acknowledged commit offsets with the workload.
Submit-to-visible samples bound commit-to-visible latency from above; the
acknowledgment-to-visible samples are reported separately. Compare the recorded
values with the budgets without adding elapsed-time assertions to tests.

### Registration

`polylogue completions SHELL` prints Polylogue's own script for bash, zsh and fish. The script calls the program by the name it was invoked as, so `plg`, `plog` and wrappers bound to another archive complete against their own archive; it registers every installed name; it passes words, cursor index and open-quote state. zsh groups candidates under headings and uses messages for the preview and the catalog's age. The package installs the scripts into the shells' completion directories.
