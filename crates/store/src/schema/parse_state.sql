-- A waiting chain keeps the stored length it waits at in its own integer
-- column, a refused or failed chain its reason or error in its own text
-- column, and the status is one of the parse standings.
CREATE TABLE parse_state_typed (
  input_pk          INTEGER NOT NULL REFERENCES input_ref(pk),
  chain             INTEGER NOT NULL,
  reading_version   TEXT NOT NULL DEFAULT '',
  status            TEXT NOT NULL CHECK(status IN ('parsing','waiting','done','no_parser_yet','partial','refused','failed')),
  parsed_upto       INTEGER NOT NULL,
  at_length        INTEGER CHECK(at_length >= 0),
  records           INTEGER NOT NULL DEFAULT 0,
  used              INTEGER NOT NULL DEFAULT 0,
  skipped           INTEGER NOT NULL DEFAULT 0,
  unreadable        INTEGER NOT NULL DEFAULT 0,
  commit_seq INTEGER CHECK(commit_seq > 0),
  checkpoint BLOB,
  reason            TEXT,
  PRIMARY KEY (input_pk, chain),
  CHECK((status IN ('waiting','refused','failed')) = (at_length IS NOT NULL)),
  CHECK((status IN ('refused','failed')) = (reason IS NOT NULL))
) STRICT, WITHOUT ROWID;
INSERT INTO parse_state_typed(input_pk,chain,reading_version,status,parsed_upto,at_length,records,used,skipped,unreadable,commit_seq,checkpoint,reason)
  SELECT input_pk,chain,reading_version,status,parsed_upto,
    CASE WHEN status='waiting' THEN CAST(detail AS INTEGER) END,
    records,used,skipped,unreadable,commit_seq,checkpoint,NULL
  FROM parse_state;
DROP TABLE parse_state;
ALTER TABLE parse_state_typed RENAME TO parse_state;
CREATE INDEX parse_state_status ON parse_state(status) WHERE status <> 'done';

-- A wire reading's progress is one value instead of two flags.
UPDATE wire_reading SET state=json_set(state,'$.progress',json_object('no_parser_yet',
    json_object('at_end',json(CASE WHEN json_extract(state,'$.at_end') THEN 'true' ELSE 'false' END))))
  WHERE json_extract(state,'$.no_parser_yet');
UPDATE wire_reading SET state=json_set(state,'$.progress','at_end')
  WHERE NOT json_extract(state,'$.no_parser_yet') AND json_extract(state,'$.at_end');
UPDATE wire_reading SET state=json_set(state,'$.progress','partway')
  WHERE NOT json_extract(state,'$.no_parser_yet') AND NOT json_extract(state,'$.at_end');
UPDATE wire_reading SET state=json_remove(state,'$.at_end','$.no_parser_yet');
