-- One counter per import that moves whenever its members, asset metadata or
-- sidecars change; a read binds it instead of re-reading the members.
CREATE TABLE import_change (
  import_id TEXT PRIMARY KEY,
  changed INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
