# References

A reference names one thing in the archive, or one thing outside it, in a form every surface accepts: the CLI, MCP, HTTP, hooks, markup in transcripts, the web reader and the import of user state. One grammar and one parser (in the `model` crate) serve all of them. How references appear inside query text is [query.md](query.md); the CLI's own shorthands (`@N`, `@last`) are [interfaces.md](interfaces.md#addressing).

## Forms

| Form | Names |
|---|---|
| `request:<request id>` | one exact retained accepted request, including its original outcome, frozen effects and provenance |
| `session:<origin>:<native id>` | a physical session |
| `message:<session>:<message id>` | a message |
| `message:<id>..<id>` | a range of messages in one session, both ends included, in reading order |
| `block:<message>#<n>` | the n-th stored block of a message |
| `tool:<session>#<call id>` | a tool call, by the harness's call ID |
| `span:<container>#<unit>=<a>-<b>` | a span of text. The container is a message, a document, a block, `assertion:<root>@<rev>` or `variant:<root>@<rev>#<n>`; the unit is `b` (UTF-8 bytes), `c` (code points) or `u16` (UTF-16 code units, as browser selections report them) |
| `document:<id>` | an imported document, such as a captured harness memory file |
| `event:<id>` | a Polylogue event |
| `run:<id>` | one query invocation, by its daemon-minted canonical lowercase UUID |
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

A run ID has the canonical 36-character UUID spelling (`8-4-4-4-12`
lowercase hexadecimal digits). It is minted at the accepting daemon boundary,
independently of a caller's request or retry key. `run:` always names this
namespace; malformed run IDs are errors, never session-origin shorthand. The
canonical run-begin record establishes the run's identity; an outcome points
back to the same ID. Reading a run uses that record and its outcome projection.

Surfaces also accept shorter forms that resolve to one of the above before anything is stored or sent: `self` (the calling session), `last` and `latest`, a bare native ID or a unique prefix of at least eight characters, and a pasted provider URL.

## Text coordinates

Span offsets refer to original stored text, before view filtering or renditions. Message and document text uses the shared text-body recipe with two line feeds between bodies; a block names only its existing text body, and an exact assertion or variant part names its immutable body. UTF-8 bounds must not split a character; UTF-16 bounds must not split a surrogate pair. An accessible empty span differs from an absent or unavailable source.

Portable text fragments retain the original requested spans and their declared normalized UTF-8 bounds. A fragment can preserve selected text when its complete container is unavailable; it does not verify the complete source version from those bytes. Different source versions are not interchangeable.

## Rules

- **Natural keys only.** No reference contains a row number, so a reference means the same thing in every archive that holds the same data and survives re-parse and rebuild.
- **One parser resolves pasted provider URLs** through each origin's declaration: `chatgpt.com/c/<id>` becomes `session:chatgpt:<id>`.
- **Ambiguity is an error.** A name that two current roots carry (`page:<name>`, `query:<name>`) is refused as `ambiguous`, listing both roots. Time never picks one, because a guess would silently change what an older reference meant.
- **Stored references name an exact revision where meaning requires it.** Links to assertions in the roles `revises`, `answers`, `challenges`, `supports`, `qualifies` and `corrects`, and every judgment, store `assertion:<root>@<rev>`. A `derived_from` link names its exact source: an assertion revision, an identified query or result, or another source reference. A `withdraws` link may name a revision or a whole root. A page slot set to `follow` stores a revision-less assertion root ([pages.md](pages.md)). A judgment of one revision therefore never moves to the next.
- **A reference may name something not stored yet.** It is kept as written and attaches, with no rewrite, when the thing arrives; until then it is shown as waiting ([assertions.md](assertions.md#currency)).
- **A reference names a physical thing exactly**, even where listings group physical sessions into logical ones ([sessions.md](sessions.md#logical-sessions)).

## Laws

- **round-trip**: every form parses and prints back to the same string, and a generated reference survives parse, store and read unchanged (property test over generated references).
- **one-parser**: the CLI, MCP, HTTP, hooks, markup and the web reader resolve the same input to the same reference (one test per surface over the same cases).
- **provider-url**: a pasted provider URL for each origin resolves to the same reference as its full `session:` form.
- **shorthand-resolves**: `self`, `last`, `latest`, a unique prefix and a bare native ID each resolve to the same row as the full reference, and a prefix shared by two rows is refused as `ambiguous`.
- **name-ambiguity-refused**: a page name carried by two current roots is refused as `ambiguous`, listing both roots.
- **exact-revision-stored**: a judgment or `revises` link written against the current revision stores `@<rev>`, and still names that revision after a newer one is written.
- **no-row-numbers**: no stored or printed reference contains a database row number (checked over every stored reference in the Morrow archive).

## Recorded text and spans

The store's reference-text stream reads original source bodies before view
filtering, masking or variant substitution. A physical session uses the same
original message and block order as its source reader; a block address uses
its original stored ordinal. Recorded text, thinking text and textual tool
results are joined by two LF characters between bodies. Empty recorded text
stays present. Tool arguments and other structured metadata are separate
from this plain-text stream. Missing targets, unavailable text and accessible
empty text stay distinct. A sparse reading can supply an exact known block,
but a message, range or session with missing original blocks has unavailable
source text; the available subset is never hashed as a complete source.

An exact assertion's plain-text body is its recorded string value or `text`
member. Rubrics and roles use `instructions`; judgments use their recorded
`reason`. Other structured values have no plain-text body. Text and span access
to those values is unavailable, while whole-object reads retain the typed value.
A recorded empty string remains accessible empty text. Absence of a text body
never supplies the hash of an empty string.

Text hashes are BLAKE3 of those UTF-8 chunks. They are separate from a
message's canonical JSON source-version hash. Spans use half-open bounds
`[a,b)` within this text, including separators. Byte and UTF-16 bounds must
fall on a complete Unicode character boundary; code-point bounds count
Unicode scalar values. Out-of-range or split-character bounds are refused
before any span text is emitted.

Accepted request IDs use their existing opaque, nonempty string identity. A
`request:` reference consumes the complete suffix; colons, `@`, `#` and `..`
inside that ID do not introduce revisions, parts or ranges. Exact request
reads expose retained current or historical accepted evidence, including
requests with no effects. Missing content remains unavailable. They do not
reaccept an action or certify native eligibility. Query operations requiring
a row domain refuse request objects until an actual row producer exists.
