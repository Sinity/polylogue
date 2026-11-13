-- Exact fields remain claims until authenticated receiver evidence exists.
CREATE TABLE message_injection_field (
  message_pk INTEGER NOT NULL REFERENCES message(pk) ON DELETE CASCADE,
  version_pk INTEGER NOT NULL REFERENCES message_projection(version_pk) ON DELETE CASCADE,
  block_ordinal TEXT NOT NULL CHECK(length(block_ordinal)>0),
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL CHECK(chain>0),
  field TEXT NOT NULL CHECK(json_valid(field)),
  PRIMARY KEY(message_pk,version_pk,block_ordinal,input_pk,chain,field)
) STRICT, WITHOUT ROWID;
CREATE INDEX message_injection_source ON message_injection_field(input_pk,chain);
