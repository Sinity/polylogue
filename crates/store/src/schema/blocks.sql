CREATE TABLE block (
  pk                INTEGER PRIMARY KEY,
  message_pk        INTEGER NOT NULL,
  session_pk        INTEGER NOT NULL,       -- the message's session, enforced by the composite key below
  idx               INTEGER,                -- compact position among the stored blocks, used for ordering
  kind              TEXT NOT NULL,          -- text | thinking | tool_call | tool_result | image | file | other
  text_id           INTEGER REFERENCES text(id),
  call_id           TEXT,                   -- tool_call and tool_result: the provider's call id (unique only in a session)
  blob_hash         BLOB,                   -- image/file, large tool output: bytes in the blob store; export `blob`
  direction         TEXT,                   -- image/file: input | output | unknown
  media_type        TEXT,
  file_name         TEXT,                   -- export `name`
  size              INTEGER,                -- image/file
  url               TEXT,                   -- image/file downloaded from a link
  encrypted         INTEGER NOT NULL DEFAULT 0,  -- thinking kept encrypted
  other_type        TEXT,                   -- other: the provider's content type name; export `type`
  ordinal           TEXT,                   -- original u64 source position; NULL means native dense idx
  value_pk          INTEGER REFERENCES block_value(pk) DEFERRABLE INITIALLY DEFERRED NOT NULL,
  FOREIGN KEY (message_pk, session_pk) REFERENCES message(pk, session_pk),
  UNIQUE (pk, session_pk)
) STRICT;
CREATE UNIQUE INDEX block_message ON block(message_pk, idx);
CREATE INDEX block_text   ON block(text_id) WHERE text_id IS NOT NULL;   -- search hit -> blocks

CREATE INDEX block_value_ref ON block(value_pk);
