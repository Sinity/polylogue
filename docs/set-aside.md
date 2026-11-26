# Set aside

Two things are recorded here: capabilities the operator ruled out of
Polylogue, with where their text survives, and words whose job another line
already does, each with that line. Everything else that was ever designed is
part of the product and described on its own page. `polylogue help WORD`
prints the line for each word below, as it does for the renamed words in the
vocabulary's [removed words](vocabulary.md#removed-words).

## Words another line does

| Word or form | The line that does it |
|---|---|
| `next` (the following page of the last list) | the continuation line each page ends with (`polylogue 'repo:morrow since:7d' --cursor C`); on a terminal the pager carries the whole list |
| `pack [NAME]`, named packs, `--purpose` | a page from the `bundle` template: `polylogue page:release-context brief --for codex:019d0f2a`; the page's title holds the purpose ([pages.md](pages.md#pages-elsewhere)) |
| `find QUERY` | the first-word rules: a single plain word searches, and inner quotes search a command's name (`polylogue '"status"'`) |
| `@last` | `@all`, every row of the last list as shown: `polylogue @all save` keeps exactly what was seen |
| `@*` | `@all` (`@*` is a shell glob) |
| `--all-matches` | `--all` on writes and acts; `read` reads every row it is given |
| `--first` | `'QUERY \| limit 1'` as the selection, or `@1` |
| `ping --live` | select the running sessions: `polylogue 'sessions where running:true repo:here' send 'TEXT' --all` |
| `compare DIMENSION` | `judge ROW --on DIMENSION` |
| `variant` (as a verb) | `note TEXT --as translation:LANG [--parts FILE]` |
| `explain syntax`, `explain capabilities`, `--detail full` | `help query`, `help capabilities`, `--explain --json` |
| `analyze --by --cost --tools --time --windows --compare` (reports), `prices` | queries with `group by`, `agg` and `\| measure`; the unit `prices` |
| the MCP `detail` argument | `show: ["tools:full"]` |
| recall's `--keep`, `--fuzzy` and scope narrowing; the MCP brief's `task` | `polylogue 'WORDS' brief save` keeps the search's rows; `word~` is fuzzy; conditions in the selection narrow; `--for SESSION` says who the brief is for |
| `--latest`, `--at` on the context view, `--with-blobs` | `--last N`; selecting the message; bundles always carry present attachment bytes |
| `schema --all`, `--timeout`, `--output` | `dev schema` with no origin means every origin; `--deadline`; `--to` |
| the clipboard copy of a resume brief; the web's "copy as prompt" | `polylogue codex:019d0f2a brief --to clipboard` |
| the web's share files | `read --to FILE`, offered as a download |

## Ruled out

### OpenTelemetry export

Writing sessions as OpenTelemetry GenAI traces for tools that read them, with
trace and span IDs derived from session and message references by BLAKE3,
paths redacted unless asked, tool output reduced to its presence and length,
and message text only on request. Reading OpenTelemetry GenAI span files as
an input is a capture origin and stays ([capture.md](capture.md)). The text is
in `docs/rewrite/polylogue-spec/spec/export-format.md` at revision
`37d38a6c4a^`.

### Tool results restored from `PostToolUse`

The rule that a `PostToolUse` hook event restores a tool result the
transcript truncated, in the order inline, then sidecar, then hook. Hooks
record the reported tool event and do not replace transcript content
([capture.md](capture.md#hooks)). The text is in `docs/capture.md` at
revision `eac396c371^`.

### Writers

The earlier Python Polylogue had these writers. They are outside the current
reading formats; their format knowledge remains useful if a writer returns.
The implementations are in `polylogue/rendering/formatting.py` at commit
`17c20e5c6dac`, with row-list output in `polylogue/cli/query_output_contracts.py`.

#### CSV

Session CSV wrote one row per message with text: session ID, message ID, role,
timestamp and text. Empty-text messages were skipped. Row-list CSV wrote its
declared headers and values. Both used the standard CSV writer for commas,
quotes and multiline cells, and removed the final record terminator.

The session writer prefixed role, timestamp and text values starting with
`=`, `+`, `-` or `@` with an apostrophe to prevent spreadsheet formulas.
It did not apply that rule to IDs, and row-list CSV had no such transformation.
Keep formula-safe cells and CSV quoting if a spreadsheet writer returns;
apply the formula rule consistently to every untrusted string cell.

#### YAML

Session YAML wrote metadata and full messages unless field selection excluded
them. Row-list YAML wrote an `items`, `total`, `limit`, `offset` document.
The serializer used block style, preserved insertion order and allowed Unicode.
Keep serializer-managed quoting and Unicode rather than assembling YAML text.

#### Obsidian

Obsidian output put YAML front matter before the Markdown transcript: ID,
origin, date (or `unknown`) and a flow-sequence tag list.

Its helper double-quoted strings containing YAML punctuation, either quote,
newline or tab. Inside those quotes it escaped backslashes, double quotes,
newlines and tabs. It did not handle every YAML implicit-type spelling.
Keep the front-matter structure and Unicode support; use a YAML serializer's
string quoting if this writer returns.

#### Org

Org output began with `#+TITLE`, a local `#+DATE` (or `unknown`) and an
`origin` property. Each message with text followed an uppercase role heading,
with `UNKNOWN` for a missing role. Bodies were copied with blank lines between
messages.

There was no Org escaping for bodies or directive values. Keep Unicode, and
quote text that could become a heading, directive or block delimiter so
archived content remains data.
