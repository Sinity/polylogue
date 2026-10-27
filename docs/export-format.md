# Structured export

`polylogue export` writes selected sessions, documents or input events as one
`polylogue-export` JSON document. `export user-state` selects the archive's
client requests and excisions instead. Import reads both through the normal
writer. This is an exchange format, not a backup of the archive's source bytes.

The [model declaration](../crates/model/src/export.rs) owns the JSON fields
and version checks. A document contains `format`, `version` and `sessions`;
nonempty `documents`, `events` and `user_state` arrays are added as needed.
Each session entry contains its `session`, ordered `messages`, and any
session `events` and `links`. IDs keep their full strings. Unknown values
remain unknown; optional fields follow their model's serialization rules.

The writer chooses the least version that represents the content:

| Version | Content requiring it |
|---|---|
| 1 | Sessions, messages, session events and links |
| 2 | Documents |
| 3 | Input events, transcript documents or attachments of unknown direction |
| 4 | User state |

Import refuses another format or a version above the reader's supported
version (currently 4). It also checks that a document does not use content
its declared version cannot represent, and that a session event names its
containing session. An input event preserves the sender's ID and source
citation as claims; its new local source citation names the imported record.
User-state entries replay in commit order through the writer, which checks
each request or excision. See the [import tests](../crates/engine/src/import/tests)
for round trips and refusal cases.

The application's [export operation](../crates/polylogue/src/daemon/service.rs)
selects content through Store readers. The [renderer](../crates/render/src/lib.rs)
can also write terminal text, Markdown and HTML. Only JSON carries user state.
Query JSON is a table answer, and a budgeted read is a page; neither is this
exchange document. [Attachment reads](attachments.md) own access to binary
content. [Output masking](architecture.md#output-and-daemon-lifetime) applies
to exported content too.
