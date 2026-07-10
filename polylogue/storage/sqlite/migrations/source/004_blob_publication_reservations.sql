-- Protect the filesystem-publication -> durable-reference window.
--
-- A reservation is committed before a content-addressed blob becomes visible
-- at its final path. The raw/blob reference transaction consumes it by hash.
-- Reservations are never expired by age: unresolved rows are recoverable
-- acquisition debt, not evidence that a publisher is dead.
CREATE TABLE blob_publication_reservations (
    blob_hash        BLOB PRIMARY KEY CHECK(length(blob_hash) = 32),
    publisher_id     TEXT NOT NULL,
    reserved_at_ms   INTEGER NOT NULL,
    refreshed_at_ms  INTEGER NOT NULL
) STRICT;
