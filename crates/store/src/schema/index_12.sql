-- Foreign text observations do not require a canonical container row.
CREATE TABLE text_fragment (
  hash BLOB PRIMARY KEY CHECK(length(hash)=32),
  container TEXT NOT NULL,
  version TEXT NOT NULL CHECK(json_valid(version)),
  start BLOB NOT NULL CHECK(length(start)=8),
  end BLOB NOT NULL CHECK(length(end)=8 AND end>=start),
  total BLOB CHECK(total IS NULL OR (length(total)=8 AND total>=end)),
  record BLOB NOT NULL
) STRICT, WITHOUT ROWID;
CREATE INDEX text_fragment_interval ON text_fragment(container,version,start,end,hash);
CREATE TABLE text_fragment_source (
  hash BLOB NOT NULL REFERENCES text_fragment(hash),
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL CHECK(chain>0),
  start INTEGER NOT NULL CHECK(start>=0),
  end INTEGER NOT NULL CHECK(end>=start),
  admitted INTEGER CHECK(admitted>0),
  PRIMARY KEY(hash,input_pk,chain,start,end)
) STRICT, WITHOUT ROWID;
CREATE INDEX text_fragment_reading ON text_fragment_source(input_pk,chain,hash);
CREATE TRIGGER text_fragment_source_removed AFTER DELETE ON text_fragment_source BEGIN
  DELETE FROM text_fragment WHERE hash=OLD.hash
    AND NOT EXISTS(SELECT 1 FROM text_fragment_source WHERE hash=OLD.hash);
END;
