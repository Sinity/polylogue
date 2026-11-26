# Completion

Tab completes every position of a line, with descriptions and real values from the archive, so the grammar of [interfaces.md](interfaces.md) is learned by pressing Tab rather than from a manual. This page owns what completion offers at each position, where its values come from, and how a shell is registered. The picker and its Alt-p key are [terminal.md](terminal.md#the-picker).

## How completion works

- **The daemon answers values; the CLI knows the grammar.** Each Tab sends one `internal complete` request over the socket with the whole query and the cursor, and waits for it at most 250 ms; the budget for the whole keystroke is 50 ms ([latency.md](latency.md#latency-classes)). The same operation serves MCP `completion/complete`, HTTP and the web query box. MCP supplies the whole query in `argument.value`, with an optional UTF-8 byte offset in `_meta["polylogue/position"]` (the end by default); each completion value is the whole query with the replacement applied, and `_meta["polylogue/completions"]` carries the typed replacement range, kind and help.
- **The line positions** (the first word, verbs, a verb's words, options) come from the line parser's tables ([interfaces.md](interfaces.md#verb-words)). **Positions inside a query** come from the shared `query-language` completer: it takes the whole query text and a cursor offset and returns candidates with their replace range; Polylogue supplies the archive's values.
- **Never** does completion run a search or write, and the CLI never opens the archive.
- **Descriptions** (zsh, fish): `repo:sinex  4,927 sessions`; `7f3a2c10  Back off on 429 · 2h · sinex`; `group by  one row per value of a field`. Bash shows values only.
- **Order:** headings in the order below, grammar candidates in declaration order, values by count descending, sessions by last activity. There are no fixed caps: every matching session and value is offered, and the shell shows what fits.

## Positions

| Typing after `polylogue ` | Candidates, with descriptions |
|---|---|
| the first word | under headings: **commands**, each with its one line; **verbs on their usual rows** (`read  pick from recent sessions`, `brief  for this directory`, `resume  pick a session that ended open`); **saved queries** (`waiting  sessions waiting on you · 2`, yours marked `yours`); **units**; **recent sessions** (`7f3a2c10  Back off on 429 · 2h · sinex`) |
| `'` | units (`tool_calls  every tool call, with outcome`), fields for sessions, references (`here`, `latest`, `self`, `query:`, `result:`, `page:`) |
| `'repo:` | values with counts (`sinex  4,927 sessions`): origins that have sessions, repository, project, cwd by path component, model, harness, tool, MCP server, tag and `tag:key:`, branch; closed sets; after `query:`, saved names with parameters and their one line; after `result:`, kept result IDs with names |
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
- **Daemon down or late:** when no daemon answers within that wait, Tab offers the grammar alone from the CLI's own tables (commands, verbs, options, units, fields, operators, transforms, closed values, reference prefixes, setting keys and their declared choices), with nothing on the terminal about the daemon. No value from the archive is offered then, and none is remembered from an earlier answer.

## Where values come from

The daemon answers each request from its own readers over `index.db`, in the interactive lane, and keeps nothing for completion between requests:

- **Field values with counts:** origin, repository, working directory (each directory also counts for its parents), branch and harness count sessions; model and tool count the messages that name the model and the calls that use the tool, from the per-session counts the writer keeps with the rows; tags count current tag annotations, as `name` and `name:value`.
- **Sessions:** by full ID or native ID prefix through the ID index, each with its shortest unique prefix, title, origin, repository and last activity; a prefix that starts no ID matches titles, its characters in order.
- **Saved queries:** the current authored names, then the shipped ones no authored name replaces, each with its text, its parameters and the fields a query over its rows can name; two current definitions of one name are marked ambiguous.
- **Kept results:** by name or ID, with unit and row count, and their row and match fields.
- **Settings:** the declared keys and the keys set in the archive, with each key's current value, then the key's allowed values.
- **First words:** a single plain word shaped like a saved name is a saved query when completing `query:WORD` offers it; the CLI asks once before running the line ([interfaces.md](interfaces.md#the-first-word)).

The budget is the 50 ms per keystroke of the completion class, process start and the socket round trip included. To record callback latencies on synthetic data, with the daemon up and down:

```sh
cargo xtask test -p polylogue -E 'test(completion_latency_reports_real_shell_callbacks_with_daemon_up_and_down)' --success-output immediate
```

## Registration

`polylogue install zsh`, `polylogue install bash` and `polylogue install fish` write Polylogue's completion script into the shell's completion directory; in zsh and fish they also bind Alt-p, which opens the picker in any shell line and inserts the chosen references at the cursor ([terminal.md](terminal.md#the-picker)). Where the script goes and the other install targets are [installing.md](installing.md#installing).

- The script calls the program by the name it was invoked as, so `plg`, `plog` and wrappers bound to another archive complete against their own archive; it registers every installed name, and passes words, cursor index and open-quote state. zsh groups candidates under headings and uses messages for the count preview and the first-word rule.
- `--program NAME` writes the script for a wrapper on `PATH` that runs Polylogue under another name: `polylogue install zsh --program plg`.
- `--dry-run` prints the script instead of writing it, which is what packagers use: the output of `polylogue install zsh --dry-run` is the zsh script the package installs.
- `--remove` restores the bytes the install replaced.
