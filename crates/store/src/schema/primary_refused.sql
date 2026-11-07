CREATE TABLE primary_refused (
 event TEXT PRIMARY KEY, position INTEGER NOT NULL, kind TEXT NOT NULL, reason TEXT NOT NULL,
 unreadable_input INTEGER NOT NULL CHECK(unreadable_input IN (0,1))
) STRICT;
