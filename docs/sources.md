# Source formats

The [origin declarations](../crates/sources/src/origin.rs) own supported
origins, file locations, recognition, parser choice, resume commands and
browser capabilities. The origin table is also used for recognition and collection.
A declared location alone does not mean a parser exists: `Parsed` origins
produce rows, `Kept` files supply attachment bytes, and `Declared` origins
have no collector or parser.

An input arrives from a live tool location, an explicit import, or a spool
written by Polylogue. [Architecture](architecture.md#storing-bytes) describes
storing and merging them; [attachments](attachments.md) describes export
members and sidecars. These rules apply in both storage modes.

## Agent transcripts

| Origin | Stored form and reading |
|---|---|
| [Claude Code](../crates/sources/src/claude_code.rs) | JSON lines under `CLAUDE_CONFIG_DIR` (default `~/.claude`). Assistant records sharing `message.id` contribute blocks to one message. Copied history keeps its source identities; folder, metadata and copied-record evidence supply parent links. Main transcripts, subagents and Workflow journals are recognized separately. |
| [Codex](../crates/sources/src/codex.rs) | Rollouts, prompt history and the session-name index under `CODEX_HOME` (default `~/.codex`), plus [database row dumps](../crates/sources/src/codex/database.rs). Each response item is an addressable message. Usage and quota reports are separate observations; they do not establish complete response membership or total session cost. |
| [Gemini CLI](../crates/sources/src/gemini_cli.rs) | Rewritten session JSON under `~/.gemini/tmp/`. Tool calls and results can share a model turn; CLI notices become events. |
| [Hermes](../crates/sources/src/hermes.rs) | Stored row dumps from its state and verification databases. Bodies are decoded one record at a time. |
| [Grok CLI](../crates/sources/src/grok_cli.rs) | Files at its declared live locations, read by its own parser. |

A tool database is read as committed rows, not copied SQLite pages. The
[database storage code](../crates/store/src/tool_databases.rs) owns the row
dump format and [offline schema tools](schema-tools.md) inspect that stored
form. Missing provider times stay unknown; message order comes from source
positions rather than a fabricated clock.

## Web conversations and documents

| Origin | Stored form and reading |
|---|---|
| [ChatGPT](../crates/sources/src/chatgpt.rs) | Export or captured conversation mappings. Every branch is stored; active-path selection controls ordinary reads and counts. Attachments keep the direction and identifiers the conversation actually records. |
| [claude.ai and Claude Design](../crates/sources/src/claude_ai.rs) | Export arrays or captured conversation JSON, with branches linked by `parent_message_uuid`. Tool results may occur inside the assistant message. Design chats have their own origin. |
| [AI Studio](../crates/sources/src/ai_studio.rs) | Drive prompt JSON (`runSettings`, `systemInstruction`, `chunkedPrompt`), imported or collected into the Drive spool. A chunk without a native ID is identified by position, so editing it changes that message's shown version. |
| [Grok](../crates/sources/src/grok.rs) | Account-export conversations, decoded individually and kept in supplied response order. |

Instruction, context, memory and transcript documents are separate from
conversation messages. Their [model](../crates/model/src/document.rs) keeps
purpose, source time, authorship, blocks and metadata. Purpose alone does
not establish who wrote a document. Empty later exports remove nothing.

## Polylogue spools

The [hook spool](../crates/sources/src/hook_spool.rs) stores harness events
with the session they name, their time and payload. A hook line without a
session is skipped with a reason. A hook event is evidence of what the
harness saw; it does not manufacture a transcript message.

The [browser capture log](../crates/sources/src/browser_capture.rs) stores
conversation bytes in length-and-digest frames. Its parser hands the body
to the provider parser. The [web reader](web-reader.md) and
[architecture](architecture.md#sinex-primary-mode) describe the client and
storage boundaries.

Synthetic parser cases live in [examples/sources](../examples/sources),
with format tests in [Sources](../crates/sources/tests/formats/main.rs).
[Morrow](../examples/morrow/README.md) exercises several origins together
through collection, import, parsing and queries.
