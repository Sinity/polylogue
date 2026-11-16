# Set aside

Capabilities that were designed or built and are not part of Polylogue now.
Each entry says what it was, why it is out, and where its full text survives,
so a capability can return from its own design without being reinvented.
Revision `058a177d0e` of this repository holds every page named below as it
was described.

## Command words

Each word below was a command, verb, option or reference form. What says the
same thing now is in the vocabulary's [removed words](vocabulary.md#removed-words).

- **`trace`**: a verb printing the source chain of a reference back to its
  stored input, its bytes and its reading version (`interfaces.md`, Verbs;
  `reading.md`). The stored records behind a row are `read --view raw`; the
  chain itself has no verb.
- **`next`**: a command printing the following page of the last list on this
  terminal (`interfaces.md`, Commands; `terminal.md`, Addressing). The pager
  and the continuation line every page prints cover paging.
- **`pack [NAME]`**: a verb building a named pack under a budget for a
  recipient and purpose (`agent-context.md`, Packs; `interfaces.md`, Verbs).
  `brief` covers every form that was used.
- **`find QUERY`**: a form reading the first word as a query whatever it was,
  with unsignalled first words refused (`interfaces.md`, Grammar). A single
  plain word now searches.
- **`compare DIMENSION`** and **`variant`** as verbs (`interfaces.md`, Verbs;
  `assertions.md`, Judgment and Variants). Comparisons are `judge ROW --on
  DIMENSION`; alternates are written with `note --as translation:LANG`.
- **`applies ENV`**: evaluating whether a method's conditions hold in an
  environment, giving `applicable`, `not_applicable` or `unassessed`
  (`assertions.md`, Memory; `interfaces.md`, Verbs). Methods are not in use.
- **`@N.M`, `@last`, `@*`**: hit M of row N, the last result printed here, and
  every row as printed (`terminal.md`, Addressing). Row numbers are `@N`,
  `@A-B`, `@A,B` and `@all`, on a terminal only.
- **`--all-matches`** and **`--first`** on one-row verbs (`terminal.md`,
  Picker). Writes over several query rows take `--all`; one-row verbs never
  guess.
- **`--wait`**: waiting inside a command for a later event, on `import` and
  on `chat send` (`capture.md`, Import; `extension.md`, Chat actions). It
  outlives tool timeouts; replies name the line that checks again.
- **`ping TEXT --live`**: a ping refusing rather than resuming a session that
  was not running (`session-control.md`, Live-session control and Pings).
  `send` reaches running sessions and resumes only one ended session named
  alone.
- **`chat send --submit`** for a new conversation (`extension.md`, Chat
  actions). New web conversations are drafted by `start --in chatgpt` and
  submitted by the operator.
- **Markup sends**: `@[address] text` inside a message as a ping, and
  `@[QUERY]` broadcasting to every session a query selects
  (`session-control.md`, Pings; `pages.md`, Ping records; `assertions.md`,
  Markup). Markup writes annotations only; sending is the `send` verb, and a
  send to several sessions is a query with `--all`.
- **`resume --fork`**, **`resume --anyway`** and the clipboard copy of a resume
  brief (`session-control.md`, Resume). A running session is not reopened; a
  harness's own fork is not used; `brief --to clipboard` copies a brief.

## Pages

Pages were places where work is organised: a page held references, live
queries and headings as slots, rendered as linear, board, wiki or table
layouts (`pages.md`; `web-views.md`, Pages and Workspaces; `assertions.md`,
Kinds and Links).

- **Page verbs**: `add REF…`, `move POS`, `toc` (a page's generated table of
  contents, an agent's starting point), `checkout DIR` and `checkin DIR`
  (editing a page as Markdown files, refused on a stale base), page `keep` (a
  checkpoint fixing every slot at its current revision), `analyze` (per-arm
  intervals of an experiment) and `pin` (placing rows on a page, by default
  `pinned`).
- **Page reading**: `page:X read --at T`, `page:home`, templates
  (`investigation`, `experiment`, `project-understanding`, `plan`),
  investigations, and experiments with a fixed assignment written before any
  exposure.
- **Members as aliases**: `member` records naming a session under an alias
  (`@reviewer`), resolved when a message is sent.

Pages wait for their own design. The `page`, `slot` and `member` kinds stay as
import-only annotation kinds ([annotations.md](annotations.md#kinds)); roles
are [session-control.md](session-control.md#roles); freezing rows is `save`
with no name and starring is `tag star`.

## Judging over many comparisons

- **Rankings** computed from comparisons at read (win rate and a Bradley-Terry
  fit, each rank with its number of comparisons, uncertain pairs listed for
  judging), and comparison verdicts `ranking:[…]`, `tie`, `incomparable`,
  `abstain` and `insufficient_evidence` (`assertions.md`, Judgment).
- **Agreement** of a pass with a declared reference, per pass with its
  denominator (`assertions.md`, Judgment).
- **Sampled judging**: `judge --blind --sample N --seed S QUERY` drawing a
  seeded sample (`assertions.md`, Judgment).
- **The one-key judging loop**: `judge` walking open items in the picker with
  one key per verdict (`assertions.md`, Judgment; `terminal.md`, Picker).
  `polylogue review pick judge accept` covers it.

Single comparisons, blind ones included, are built ([judging.md](judging.md)).

## Options and smaller forms

- **MCP resources and prompts** (`polylogue://agent/manual`, session,
  assertion, page table-of-contents and result resources) and the MCP
  `detail` argument (`agents.md`, MCP server; `reading.md`, Rendering for
  models). The three tools carry every read; `show: ["tools:full"]` says what
  `detail` said.
- **Pack and recall options**: `--purpose`, `--for operator`, named packs,
  recall's `--keep`, `--fuzzy` and query scope narrowing, MCP `brief` with
  `task` (`agent-context.md`, Packs, Brief and Recall; `search.md`, Recall).
  The forms of `brief` and `word~` cover them.
- **Reading options**: `--output-lines`, `--latest`, `--at` on the context
  view and `--with-blobs` (`reading.md`). Ranges, selecting the message, and
  bundles that always carry present attachment bytes cover them.
- **Reports**: the `analyze` verb with `--by`, `--cost`, `--tools`,
  `--time`, `--windows` and `--compare`, and the `prices` command
  (`measures.md`, Reports and The price catalog). Reports are queries with
  `group by`, `agg` and `| measure`; prices are the unit `prices`.
- **`explain syntax`**, **`explain capabilities`** and **`--detail full`**
  (`query.md`, Explain). `help query`, `help --all --json` and
  `--explain --json` cover them.
- **`install-agent --instructions`** and **`setting --machine NAME`**
  (`operations.md`, Installing into harnesses and Configuration).
- **`schema infer|unread --all`**, `--timeout` and `--output`
  (`archive.md`, Schema tools). Naming no origin means every origin;
  `--deadline` and `--to` are the line and output options.
- **Renaming a web conversation in the provider** (`extension.md`, Chat
  actions); no verb does it.

## Web views

- **Pages, Workspaces and Project story** views with the composer, decision
  box, layouts, templates and saved workspaces (`web-views.md`; `web.md`,
  Frame, Keys and Routes `/pages`, `/p/PAGE`, `/w/MODE`).
- **Judging passes on `/judge`**: sampling, agreement, calibration and rubric
  runs again (`web-views.md`, Notes, claims, judging, triage).
- **Compare alignment** by lineage, time or shared file, and its page actions
  (`web-views.md`, Compare).
- **Context bundles** with their manifest and composer, branching through the
  harness's own fork, pin and bundle actions, copy as prompt, and share files
  written by the daemon (`web-reader.md`). `brief`, `tag star`, `resume --in`
  and the share view's download cover what remains.

## Writers

The earlier Python Polylogue had these writers. They are outside the current
reading formats; their format knowledge remains useful if a writer returns.
The implementations are in `polylogue/rendering/formatting.py` at commit
`17c20e5c6dac`, with row-list output in `polylogue/cli/query_output_contracts.py`.

### CSV

Session CSV wrote one row per message with text: session ID, message ID, role,
timestamp and text. Empty-text messages were skipped. Row-list CSV wrote its
declared headers and values. Both used the standard CSV writer for commas,
quotes and multiline cells, and removed the final record terminator.

The session writer prefixed role, timestamp and text values starting with
`=`, `+`, `-` or `@` with an apostrophe to prevent spreadsheet formulas.
It did not apply that rule to IDs, and row-list CSV had no such transformation.
Keep formula-safe cells and CSV quoting if a spreadsheet writer returns;
apply the formula rule consistently to every untrusted string cell.

### YAML

Session YAML wrote metadata and full messages unless field selection excluded
them. Row-list YAML wrote an `items`, `total`, `limit`, `offset` document.
The serializer used block style, preserved insertion order and allowed Unicode.
Keep serializer-managed quoting and Unicode rather than assembling YAML text.

### Obsidian

Obsidian output put YAML front matter before the Markdown transcript: ID,
origin, date (or `unknown`) and a flow-sequence tag list.

Its helper double-quoted strings containing YAML punctuation, either quote,
newline or tab. Inside those quotes it escaped backslashes, double quotes,
newlines and tabs. It did not handle every YAML implicit-type spelling.
Keep the front-matter structure and Unicode support; use a YAML serializer's
string quoting if this writer returns.

### Org

Org output began with `#+TITLE`, a local `#+DATE` (or `unknown`) and an
`origin` property. Each message with text followed an uppercase role heading,
with `UNKNOWN` for a missing role. Bodies were copied with blank lines between
messages.

There was no Org escaping for bodies or directive values. Keep Unicode, and
quote text that could become a heading, directive or block delimiter so
archived content remains data.
