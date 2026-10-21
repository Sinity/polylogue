-- Exact local cleanup intent survives separate archive/index/vector commits.
CREATE TABLE pending_vector_excision (
    text_hash BLOB PRIMARY KEY CHECK(length(text_hash)=32),
    excision_id TEXT NOT NULL
) STRICT, WITHOUT ROWID;
