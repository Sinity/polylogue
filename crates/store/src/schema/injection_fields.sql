-- Hook fields a reading's message states, as stated: they stay declarations
-- until the receiver is authenticated. Owned by the reading, like its message.
CREATE TABLE message_injection_field (
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL CHECK(chain>0),
  message_id TEXT NOT NULL,
  source_block_position TEXT NOT NULL CHECK(length(source_block_position)>0),
  field TEXT NOT NULL CHECK(json_valid(field)),
  PRIMARY KEY(input_pk,chain,message_id,source_block_position,field)
) STRICT, WITHOUT ROWID;
