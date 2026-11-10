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
-- What a reading version declares about a path: once per format.
CREATE TABLE wire_declaration (
  format TEXT NOT NULL,
  reading_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL CHECK (json_valid(table_key)),
  expected TEXT NOT NULL CHECK (json_valid(expected)),
  dynamic_map INTEGER NOT NULL CHECK (dynamic_map IN (0,1)),
  known_kinds TEXT NOT NULL CHECK (json_valid(known_kinds)),
  PRIMARY KEY (format,reading_version,record_kind,path,table_key)
) STRICT, WITHOUT ROWID;
-- One path of a reading: its counts and the receipts (numbers in
-- wire_receipt) where it was first and last seen.
CREATE TABLE wire_shape (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  reading_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL CHECK (json_valid(table_key)),
  unknown_kinds INTEGER NOT NULL CHECK (unknown_kinds >= 0),
  unknown_tables INTEGER NOT NULL CHECK (unknown_tables >= 0),
  unused INTEGER NOT NULL CHECK (unused >= 0),
  first INTEGER NOT NULL,
  last INTEGER NOT NULL,
  PRIMARY KEY (input_pk,chain,reading_version,record_kind,path,table_key),
  FOREIGN KEY (input_pk,chain,reading_version,first)
    REFERENCES wire_receipt(input_pk,chain,reading_version,number),
  FOREIGN KEY (input_pk,chain,reading_version,last)
    REFERENCES wire_receipt(input_pk,chain,reading_version,number)
) STRICT, WITHOUT ROWID;
-- How often a path held each JSON type in a reading.
CREATE TABLE wire_type (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  reading_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL,
  json_type TEXT NOT NULL,
  count INTEGER NOT NULL CHECK (count >= 0),
  first INTEGER NOT NULL,
  last INTEGER NOT NULL,
  PRIMARY KEY (input_pk,chain,reading_version,record_kind,path,table_key,json_type),
  FOREIGN KEY (input_pk,chain,reading_version,record_kind,path,table_key)
    REFERENCES wire_shape(input_pk,chain,reading_version,record_kind,path,table_key),
  FOREIGN KEY (input_pk,chain,reading_version,first)
    REFERENCES wire_receipt(input_pk,chain,reading_version,number),
  FOREIGN KEY (input_pk,chain,reading_version,last)
    REFERENCES wire_receipt(input_pk,chain,reading_version,number)
) STRICT, WITHOUT ROWID;
