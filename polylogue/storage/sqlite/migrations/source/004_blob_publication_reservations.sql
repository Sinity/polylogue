-- Protect the filesystem-publication -> durable-reference window.
--
-- Each publication attempt has independent identity. Two publishers of the
-- same bytes therefore cannot consume one another's protection.
CREATE TABLE blob_publication_reservations (
    publication_id   TEXT PRIMARY KEY,
    blob_hash        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    size_bytes       INTEGER NOT NULL CHECK(size_bytes >= 0),
    publisher_id     TEXT NOT NULL,
    reserved_at_ms   INTEGER NOT NULL
) STRICT;

CREATE INDEX idx_blob_publication_reservations_hash
    ON blob_publication_reservations(blob_hash);
