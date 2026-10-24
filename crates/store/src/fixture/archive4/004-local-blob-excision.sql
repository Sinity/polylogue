-- Only exact deletion intents of committed local excisions are replayed.
CREATE TABLE pending_blob_excision (
    hash BLOB PRIMARY KEY CHECK(length(hash)=32),
    excision_id TEXT NOT NULL
) STRICT, WITHOUT ROWID;
