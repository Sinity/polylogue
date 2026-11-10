-- Derived coverage and shape facts, committed with the parse rows they describe.
CREATE TABLE wire_reading (
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  reading_version TEXT NOT NULL,
  format TEXT NOT NULL,
  active INTEGER NOT NULL CHECK (active IN (0,1)),
  state TEXT NOT NULL CHECK (json_valid(state)),
  PRIMARY KEY (input_pk, chain, reading_version)
) STRICT, WITHOUT ROWID;
CREATE UNIQUE INDEX wire_reading_current ON wire_reading(input_pk,chain) WHERE active=1;
-- The receipts a reading's shapes cite, each stored once and cited by number.
CREATE TABLE wire_receipt (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  reading_version TEXT NOT NULL,
  number INTEGER NOT NULL CHECK (number > 0),
  receipt TEXT NOT NULL CHECK (json_valid(receipt)),
  PRIMARY KEY (input_pk,chain,reading_version,number),
  FOREIGN KEY (input_pk,chain,reading_version)
    REFERENCES wire_reading(input_pk,chain,reading_version)
) STRICT, WITHOUT ROWID;
CREATE TABLE wire_shape (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  reading_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL CHECK (json_valid(table_key)),
  shape TEXT NOT NULL CHECK (json_valid(shape)),
  PRIMARY KEY (input_pk,chain,reading_version,record_kind,path,table_key),
  FOREIGN KEY (input_pk,chain,reading_version)
    REFERENCES wire_reading(input_pk,chain,reading_version)
) STRICT, WITHOUT ROWID;
