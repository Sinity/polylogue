# References

A reference names one thing in the archive, or one thing outside it, in a form every surface accepts: the CLI, MCP, HTTP, hooks, markup in transcripts, the web reader and the import of user state. One grammar and one parser (in the `model` crate) serve all of them. How references appear inside query text is [query.md](query.md); the CLI's own shorthands (`@N`, `@last`) are [interfaces.md](interfaces.md#addressing).

## Forms

| Form | Names |
|---|---|
| `session:<origin>:<native id>` | a physical session |
| `message:<session>:<message id>` | a message |
| `message:<id>..<id>` | a range of messages in one session, both ends included, in reading order |
| `block:<message>#<n>` | the n-th stored block of a message |
| `tool:<session>#<call id>` | a tool call, by the harness's call ID |
| `span:<container>#<unit>=<a>-<b>` | a span of text. The container is a message, a document, a block, `assertion:<root>@<rev>` or `variant:<root>@<rev>#<n>`; the unit is `b` (UTF-8 bytes), `c` (code points) or `u16` (UTF-16 code units, as browser selections report them) |
| `document:<id>` | an imported document, such as a captured harness memory file |
| `event:<id>` | a Polylogue event |
| `run:<id>` | one run of a query ([query.md](query.md#runs)), by the lowercase UUID the daemon gave it |
| `attachment:<id>` | an attachment |
| `assertion:<root>` | the current revision of an assertion |
| `assertion:<root>@<rev>` | one exact revision of an assertion |
| `variant:<root>[@<rev>][#<n>]` | a variant, one revision of it, or one of its parts |
| `page:<name>`, `page:<root>[@<rev>]` | a page by name (resolved to its root), or by root, optionally at an exact revision |
| `role:<name>`, `role:<root>[@<rev>]` | a role by name (resolved to its root), or by root, optionally at an exact revision ([pages.md](pages.md#members-and-roles)) |
| `query:<hash>`, `query:<name>[(p=v,…)]` | a query object, or a saved query by name with parameters ([query.md](query.md#query-objects)) |
| `result:<id>`, `result:<id>#<row>` | a kept result, or one of its rows |
| `delivery:<id>` | one delivery to an agent ([agents.md](agents.md#deliveries)) |
| `file:<path>@<commit>` | a file at a commit, read through git |
| `repo:<name>`, `cwd:<path>` | a repository, a working directory; used as scopes |
| `url:<absolute URL>` | an outside page, stored as text; reading the reference never fetches it |
| `sx:<sinex reference>` | a Sinex object. Sinex owns the grammar of the string ([Sinex query](https://github.com/Sinity/sinex/blob/master/docs/query.md)); Polylogue resolves it through Sinex's API when Sinex is reachable and shows it `unavailable` otherwise |

Surfaces also accept shorter forms that resolve to one of the above before anything is stored or sent: `self` (the calling session), `last` and `latest`, a bare native ID or a unique prefix of at least eight characters, and a pasted provider URL.

## Text coordinates

Span offsets count in the original stored text, before view filtering or variants. A message's or document's text is its text bodies (recorded text, thinking and textual tool results) in stored order, joined by two line feeds; a block names only its own body, and an exact assertion revision or variant part names its own text (an assertion's string value or `text` member; `instructions` for rubrics and roles; `reason` for judgments). Bounds are half-open, `[a,b)`. A byte or UTF-16 bound that splits a character is refused, as is an out-of-range bound. Empty text is text; a missing or unavailable source is not, and never hashes as the empty string. Text hashes are BLAKE3 of the UTF-8 text, separate from a message's source hash.

## Rules

- A native Claude record UUID from a split response resolves through its recorded source association to the canonical message. Stored assertion targets keep the native reference as written across re-parse.
- **Natural keys only.** No reference contains a row number, so a reference means the same thing in every archive that holds the same data and survives re-parse and rebuild.
- **One parser resolves pasted provider URLs** through each origin's declaration: `chatgpt.com/c/<id>` becomes `session:chatgpt:<id>`.
- **Ambiguity is an error.** A name that two current roots carry (`page:<name>`, `query:<name>`) is refused as `ambiguous`, listing both roots. Time never picks one, because a guess would silently change what an older reference meant.
- **Stored references name an exact revision where meaning requires it.** Links to assertions in the roles `revises`, `answers`, `challenges`, `supports`, `qualifies` and `corrects`, and every judgment, store `assertion:<root>@<rev>`. A `derived_from` link names its exact source: an assertion revision, a query or a kept result. A `withdraws` link names a revision, or a root to withdraw the whole family ([assertions.md](assertions.md#currency)). Otherwise a revision-less `assertion:<root>` is stored only as what a page slot set to `follow` shows ([pages.md](pages.md)). A judgment of one revision therefore never moves to the next.
- **A reference may name something not stored yet.** It is kept as written and attaches, with no rewrite, when the thing arrives; until then it is shown as waiting ([assertions.md](assertions.md#currency)).
- **A reference names a physical thing exactly**, even where listings group physical sessions into logical ones ([sessions.md](sessions.md#logical-sessions)).
