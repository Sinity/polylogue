-- Durable member identity is authoritative; source_index remains a hint.
-- This migration is intentionally backup-protected because it adds columns to
-- irreplaceable raw evidence and backfills them from the coordinate sidecar.
ALTER TABLE raw_sessions ADD COLUMN addressing_mode TEXT;
ALTER TABLE raw_sessions ADD COLUMN content_identity TEXT
    CHECK(content_identity IS NULL OR length(content_identity) = 64);

UPDATE raw_sessions
   SET addressing_mode = (
           SELECT c.addressing_mode
             FROM raw_container_coordinates AS c
            WHERE c.raw_id = raw_sessions.raw_id
       ),
       content_identity = (
           SELECT c.content_identity
             FROM raw_container_coordinates AS c
            WHERE c.raw_id = raw_sessions.raw_id
       )
 WHERE EXISTS (
           SELECT 1
             FROM raw_container_coordinates AS c
            WHERE c.raw_id = raw_sessions.raw_id
       );
