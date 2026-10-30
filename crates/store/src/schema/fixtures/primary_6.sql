
CREATE TABLE primary_fact (
 event TEXT PRIMARY KEY, admitted INTEGER NOT NULL, changed INTEGER NOT NULL, source_order INTEGER NOT NULL,
 state TEXT NOT NULL CHECK(state IN ('live','archived','tombstoned')),
 product BLOB, input TEXT, chain INTEGER, reference TEXT, kind TEXT NOT NULL, step_start INTEGER, step_end INTEGER, reading_version TEXT, session TEXT
) STRICT;
CREATE INDEX primary_fact_chain ON primary_fact(input,chain,admitted,event);
CREATE INDEX primary_fact_session ON primary_fact(session,kind,state,admitted);
CREATE TABLE primary_fact_blob (
 event TEXT NOT NULL REFERENCES primary_fact(event), hash BLOB NOT NULL,
 PRIMARY KEY(event,hash)
) STRICT, WITHOUT ROWID;
CREATE INDEX primary_fact_blob_hash ON primary_fact_blob(hash);
CREATE TABLE primary_pending (
 n INTEGER PRIMARY KEY, position INTEGER NOT NULL, event TEXT NOT NULL,
 change TEXT NOT NULL, product BLOB, source_order INTEGER NOT NULL,
 UNIQUE(position,event,change)
) STRICT;
CREATE INDEX primary_pending_position ON primary_pending(position,n);
CREATE TABLE primary_complete(position INTEGER PRIMARY KEY) STRICT;
CREATE TABLE source_material (
 id TEXT NOT NULL, event TEXT NOT NULL, input TEXT, chain INTEGER, offset INTEGER,
 length INTEGER NOT NULL, hash BLOB NOT NULL, source_length INTEGER, blob_offset INTEGER NOT NULL,
 PRIMARY KEY(event,id)
) STRICT;
CREATE INDEX source_material_identity ON source_material(id);
CREATE INDEX source_material_place ON source_material(input,chain,offset);
