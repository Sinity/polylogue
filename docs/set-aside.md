# Set aside

The earlier Python Polylogue had these writers. They are outside the current
reading formats; their format knowledge remains useful if a writer returns.
The implementations are in `polylogue/rendering/formatting.py` at commit
`17c20e5c6dac`, with row-list output in `polylogue/cli/query_output_contracts.py`.

## CSV

Session CSV wrote one row per message with text: session ID, message ID, role,
timestamp and text. Empty-text messages were skipped. Row-list CSV wrote its
declared headers and values. Both used the standard CSV writer for commas,
quotes and multiline cells, and removed the final record terminator.

Session CSV prefixed an apostrophe to role, timestamp and text cells beginning
with `=`, `+`, `-` or `@` to prevent spreadsheet formula execution. IDs and
row-list cells did not get that protection. Keep formula-safe text cells
throughout any future CSV writer, alongside standard CSV quoting and Unicode.

The session writer prefixed role, timestamp and text values starting with
`=`, `+`, `-` or `@` with an apostrophe to prevent spreadsheet formulas.
It did not apply that rule to IDs, and row-list CSV had no such transformation.
Keep formula-safe cells and CSV quoting if a spreadsheet writer returns;
apply the formula rule consistently to every untrusted string cell.

## YAML

Session YAML wrote metadata and full messages unless field selection excluded
them. Row-list YAML wrote an `items`, `total`, `limit`, `offset` document.
The serializer used block style, preserved insertion order and allowed Unicode.
Keep serializer-managed quoting and Unicode rather than assembling YAML text.

## Obsidian

Obsidian output put YAML front matter before the Markdown transcript: ID,
origin, date (or `unknown`) and a flow-sequence tag list.

Its helper double-quoted strings containing YAML punctuation, either quote,
newline or tab. Inside those quotes it escaped backslashes, double quotes,
newlines and tabs. It did not handle every YAML implicit-type spelling.
Keep the front-matter structure and Unicode support; use a YAML serializer's
string quoting if this writer returns.

## Org

Org output began with `#+TITLE`, a local `#+DATE` (or `unknown`) and an
`origin` property. Each message with text followed an uppercase role heading,
with `UNKNOWN` for a missing role. Bodies were copied with blank lines between
messages.

There was no Org escaping for bodies or directive values. Keep Unicode, and
quote text that could become a heading, directive or block delimiter so
archived content remains data.
