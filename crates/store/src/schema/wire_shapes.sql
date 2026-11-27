-- Derived coverage and shape facts, committed with the parse rows they describe.
CREATE TABLE wire_parse (
  input_pk INTEGER NOT NULL REFERENCES input_ref(pk),
  chain INTEGER NOT NULL,
  parser_version TEXT NOT NULL,
  origin TEXT NOT NULL,
  active INTEGER NOT NULL CHECK (active IN (0,1)),
  state TEXT NOT NULL CHECK (json_valid(state)),
  PRIMARY KEY (input_pk, chain, parser_version)
) STRICT, WITHOUT ROWID;
CREATE UNIQUE INDEX wire_parse_current ON wire_parse(input_pk,chain) WHERE active=1;
-- The occurrences a reading's shapes cite, each stored once and cited by position.
CREATE TABLE wire_occurrence (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  parser_version TEXT NOT NULL,
  position INTEGER NOT NULL CHECK (position > 0),
  occurrence TEXT NOT NULL CHECK (json_valid(occurrence)),
  PRIMARY KEY (input_pk,chain,parser_version,position),
  FOREIGN KEY (input_pk,chain,parser_version)
    REFERENCES wire_parse(input_pk,chain,parser_version)
) STRICT, WITHOUT ROWID;
-- What a parse version declares about a path: once per origin.
CREATE TABLE wire_declaration (
  origin TEXT NOT NULL,
  parser_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL CHECK (json_valid(table_key)),
  expected TEXT NOT NULL CHECK (json_valid(expected)),
  dynamic_map INTEGER NOT NULL CHECK (dynamic_map IN (0,1)),
  known_kinds TEXT NOT NULL CHECK (json_valid(known_kinds)),
  PRIMARY KEY (origin,parser_version,record_kind,path,table_key)
) STRICT, WITHOUT ROWID;
-- One path of a reading: its counts and the occurrences (positions in
-- wire_occurrence) where it was first and last seen.
CREATE TABLE wire_shape (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  parser_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL CHECK (json_valid(table_key)),
  unknown_kinds INTEGER NOT NULL CHECK (unknown_kinds >= 0),
  unknown_tables INTEGER NOT NULL CHECK (unknown_tables >= 0),
  unused INTEGER NOT NULL CHECK (unused >= 0),
  first_position INTEGER NOT NULL,
  last_position INTEGER NOT NULL,
  PRIMARY KEY (input_pk,chain,parser_version,record_kind,path,table_key),
  FOREIGN KEY (input_pk,chain,parser_version,first_position)
    REFERENCES wire_occurrence(input_pk,chain,parser_version,position),
  FOREIGN KEY (input_pk,chain,parser_version,last_position)
    REFERENCES wire_occurrence(input_pk,chain,parser_version,position)
) STRICT, WITHOUT ROWID;
-- How often a path held each JSON type in a reading.
CREATE TABLE wire_type (
  input_pk INTEGER NOT NULL,
  chain INTEGER NOT NULL,
  parser_version TEXT NOT NULL,
  record_kind TEXT NOT NULL,
  path TEXT NOT NULL,
  table_key TEXT NOT NULL,
  json_type TEXT NOT NULL,
  count INTEGER NOT NULL CHECK (count >= 0),
  first_position INTEGER NOT NULL,
  last_position INTEGER NOT NULL,
  PRIMARY KEY (input_pk,chain,parser_version,record_kind,path,table_key,json_type),
  FOREIGN KEY (input_pk,chain,parser_version,record_kind,path,table_key)
    REFERENCES wire_shape(input_pk,chain,parser_version,record_kind,path,table_key),
  FOREIGN KEY (input_pk,chain,parser_version,first_position)
    REFERENCES wire_occurrence(input_pk,chain,parser_version,position),
  FOREIGN KEY (input_pk,chain,parser_version,last_position)
    REFERENCES wire_occurrence(input_pk,chain,parser_version,position)
) STRICT, WITHOUT ROWID;
