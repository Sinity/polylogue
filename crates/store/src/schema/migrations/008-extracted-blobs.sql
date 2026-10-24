-- Decoded bytes belong to their actual retained source envelope, separately
-- from raw byte extents. Late extraction owns an independent publication unit.
CREATE TABLE extracted_blob (
  input TEXT NOT NULL REFERENCES input(id),
  chain INTEGER NOT NULL,
  source_start INTEGER NOT NULL CHECK(source_start>=0),
  source_end INTEGER NOT NULL CHECK(source_end>source_start),
  source_hash BLOB NOT NULL CHECK(length(source_hash)=32),
  parser_version INTEGER NOT NULL CHECK(parser_version>=0),
  blob BLOB NOT NULL REFERENCES blob(hash),
  state TEXT NOT NULL CHECK(state IN ('writing','present')),
  commit_seq INTEGER NOT NULL CHECK(commit_seq>0),
  PRIMARY KEY(input,chain,source_start,source_end,source_hash,parser_version,blob)
) STRICT, WITHOUT ROWID;
CREATE INDEX extracted_blob_hash ON extracted_blob(blob);
CREATE INDEX extracted_blob_sequence ON extracted_blob(commit_seq);
