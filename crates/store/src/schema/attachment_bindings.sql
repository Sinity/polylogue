-- A reproducible projection of canonical attachment claims and frozen ingress
-- witnesses. Original claims survive a reading reset; they grant no availability.
CREATE TABLE attachment_binding (
  message TEXT NOT NULL,
  source_version BLOB NOT NULL CHECK(length(source_version)=32),
  input TEXT NOT NULL,
  chain INTEGER NOT NULL CHECK(chain>0),
  import_id TEXT NOT NULL CHECK(import_id<>''),
  block TEXT NOT NULL,       -- original u64 ordinal, never a compact block index
  attachment TEXT NOT NULL, -- original u64 attachment index
  body BLOB NOT NULL,
  PRIMARY KEY(message,source_version,input,chain,import_id,block,attachment)
) STRICT, WITHOUT ROWID;
CREATE INDEX attachment_binding_source ON attachment_binding(input,chain,import_id);
