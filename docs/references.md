# References

A reference names one thing in the archive, or one thing outside it, in a form every surface accepts: the CLI, MCP, HTTP, hooks, markup in transcripts, the web reader and the import of user state. One grammar and one parser serve all of them. A reference is one form of a line's selection ([query.md](query.md#selections)); how references appear inside query text is [query.md](query.md#references-inside-queries); row numbers (`@3`, `@3.2`, `@2-5`, `@1,4`, `@all`) are the command line's own and are [terminal.md](terminal.md#row-numbers). The command line resolves `@N.M` to the `span:` of hit M of row N as printed, so a search hit can be placed or cited exactly.

## Forms

| Form | Names |
|---|---|
| `session:<origin>:<native id>` | a physical session |
| `<origin>:<native id prefix>` | the short form of a session reference: `codex:019d0f2a` |
| `message:<session>:<message id>` | a message |
| `message:<id>..<id>` | a range of messages in one session, both ends included, in reading order |
| `block:<message>#<n>` | the n-th stored block of a message |
| `tool:<session>#<call id>` | a tool call, by the harness's call ID |
| `span:<container>#<unit>=<a>-<b>` | a span of text. The container is a message, a document, a block, a tool call (`tool:<session>#<call id>`, a span of its output), `annotation:<root>@<rev>` or `alternate:<root>@<rev>#<n>`; the unit is `b` (UTF-8 bytes), `c` (code points) or `u16` (UTF-16 code units, as browser selections report them) |
| `document:<id>` | an imported document, such as a captured harness memory file |
| `event:<id>` | a Polylogue event |
| `attachment:<message>#<n>`, `attachment:<message>#<n>.<k>` | the attachment of a message's n-th stored block, or the k-th of a tool result's attachments (the first is `#<n>`); `polylogue attachment:<message>#<n> read --to FILE` writes its bytes |
| `annotation:<root>` | the current revision of an annotation ([annotations.md](annotations.md)) |
| `annotation:<root>@<rev>` | one exact revision of an annotation |
| `alternate:<root>[@<rev>][#<n>]` | an alternate, one revision of it, or one of its parts ([alternates.md](alternates.md)) |
| `role:<name>`, `role:<root>[@<rev>]` | a role by name (resolved to its root), or by root, optionally at an exact revision ([session-control.md](session-control.md#roles)) |
| `query:<hash>`, `query:<name>[(p=v,…)]`, `query:<name>@<n>` | a question, a saved query by name with parameters, or one revision of a saved query ([saved-queries.md](saved-queries.md#query-objects)) |
| `result:<id>`, `result:<id>#<row>`, `result:<id>.matches` | a kept result, one of its rows, or its match rows |
| `result:<name>` | the newest kept result saved under that name ([saved-queries.md](saved-queries.md#save)) |
| `run:<id>` | one run of a query, by the lowercase UUID the daemon gave it ([saved-queries.md](saved-queries.md#runs)) |
| `page:<name>`, `page:<root>[@<rev>]`, `page:<name>@builtin` | a page by name (resolved to its root), by root, one revision of its page record, or the shipped page of that name ([pages.md](pages.md#the-model)); `page:template/<name>` is a template. A name the page carried before a rename still resolves to it ([pages.md](pages.md#names-and-renames)) |
| `page:<name>#<anchor>` | a section or slot of a page. The anchor is a heading's slug (`page:flaky-ci#open-questions`) or a slot's short ID (`page:flaky-ci#7q2m`). It is stored as `page:<root>#<slot root>` when written and printed with the heading's current slug, so renaming the page or the heading breaks no reference |
| `@<alias>` | the session a member alias resolves to when the line runs ([pages.md](pages.md#members)); `@` and a name starting with a letter, `all` excepted |
| `delivery:<id>` | one delivery of a brief, message or notice to a session ([agent-context.md](agent-context.md#deliveries)) |
| `file:<path>@<commit>` | a file at a commit, read through git |
| `file:<path>@<commit>#L<a>-L<b>`, `file:<path>@<commit>#L<a>` | lines a to b, or line a, of a file at a commit, both ends included (`file:src/parse.rs@4e1d2c9#L40-L58`) |
| `file:<path>#L<n>` | line n of a file as the archive's captured edits left it; a relative path is under `here`. `read --view trace` names the tool call that last wrote it ([reading.md](reading.md#views)) |
| `repo:<name>`, `cwd:<path>` | a repository, a working directory; used as scopes |
| `url:<absolute URL>` | an outside page, stored as text; reading the reference never fetches it |
| `sinex:<kind>/<id>` | a Sinex object in the shared global form, for example `sinex:ev/<event id>`. Sinex owns its kinds and identities ([Sinex query](https://github.com/Sinity/sinex/blob/master/docs/query.md)); Polylogue resolves it through Sinex's API when Sinex is reachable and shows it `unavailable` otherwise |

Sinex local forms (`ev:`, `record:`, `key:`, `at:`, `window:`) are refused in Polylogue with the corresponding global form as the correction. A Sinex reference here names a whole object; Sinex resolves revisions and places inside it.

Surfaces also accept shorter forms that resolve to one of the above before anything is stored or sent: the reserved words below, a bare native ID or a unique prefix of at least eight hexadecimal characters and dashes, and a pasted provider URL. Replies print the shortest unique prefix on a terminal and over MCP, and full references in `--json` and bundles; writes always store full references.

Where text cites, a source may be given as a container and a quotation instead of a span (`[^1]: "returns the same day" <codex:01a20000-0005>`); the write finds the quotation and stores the exact span ([page-text.md](page-text.md#cite-by-quote)).

The shared language's `answer:<id>` form is refused with `try: polylogue run:<id> read`; typed `annotation:<id>` and `alternate:<id>` spellings are refused with the corresponding `annotation:<id>` and `alternate:<id>` lines. The correction keeps an exact revision, part and span when present. Row numbers that reach the daemon are refused with "row numbers are resolved by the command line".

Query and read preparation match provider conversation URLs through the shared `references::Catalog`, using the origin declarations' hosts, conversation paths and account/project prefixes. Supported `www` aliases and an optional trailing slash name the same session. Native ID segments retain their literal spelling, including percent escapes; neither a query string nor a browser fragment changes the selected physical session. Extra or embedded path segments and lookalike hosts do not name a declared conversation page.

### Reserved words

| Word | Names |
|---|---|
| `latest` | the most recently active main session that meets the line's other session conditions, never the caller |
| `self` | the calling session |
| `here` | the caller's repository: its checkout, the worktrees sharing its Git directory, and the repository names recorded there; also as a value, `repo:here` and `cwd:here` |

`latest` and `self` resolve when the line runs, so a saved query using them follows new sessions. A caller with no session of its own (a person's shell) has no `self`, and a line using it is refused `not_found` with that reason.

## Text coordinates

Span offsets count in the original stored text, before view filtering or alternates. A message's or document's text is its text bodies (recorded text, thinking and textual tool results) in stored order, joined by two line feeds; a block names only its own body, a tool call names its output text, and an exact annotation revision or alternate part names its own text (an annotation's string value or `text` member; `instructions` for rubrics and roles; `reason` for judgments). Bounds are half-open, `[a,b)`. A byte or UTF-16 bound that splits a character is refused, as is an out-of-range bound. Empty text is text; a missing or unavailable source is not, and never hashes as the empty string. Text hashes are BLAKE3 of the UTF-8 text, separate from a message's source hash. For a tool output span, the mark and trace follow the message that holds the recorded result. Portable fragments and replay use that result's original block reference and message version, so reconnecting a call to a different result cannot change the source of a retained span.

## Rules

- A Claude record UUID from a split response resolves through its recorded source association to the resolved message. Stored annotation targets hold the reference as written across re-parse.
- **Natural keys only.** No reference contains a row number, so a reference means the same thing in every archive that holds the same data and survives re-parse and rebuild.
- **One parser resolves pasted provider URLs** through each origin's declaration: `chatgpt.com/c/<id>` becomes `session:chatgpt:<id>`.
- **Ambiguity is an error.** A name that two current roots carry (`role:<name>`, `query:<name>`, `page:<name>`, an alias other than a member alias, where a global member wins over a page member, [pages.md](pages.md#members)) or a prefix that two IDs share is refused as `ambiguous`, listing the candidates. Time never picks one, because a guess would silently change what an older reference meant.
- **Stored references name an exact revision where meaning requires it.** Links to annotations in the roles `revises`, `replies_to`, `challenges`, `supports`, `qualifies` and `corrects`, and every judgment, store the exact canonical `annotation:<root>@<rev>` address, displayed as `annotation:<root>@<rev>`. A `derived_from` link names its exact source: an annotation revision, a question or a kept result. A `withdraws` link names a revision, or a root to withdraw the whole family ([annotations.md](annotations.md#current-revisions)). A judgment of one revision therefore never moves to the next.
- **A reference may name something not stored yet.** It is held as written and attaches, with no rewrite, when the thing arrives; until then it is shown as waiting ([annotations.md](annotations.md#reading)).
- **A reference names a physical thing exactly**, even where listings group physical sessions into logical ones ([sessions.md](sessions.md#logical-sessions)).
