-- migration-safety: additive-backup-required
-- Persist the structural value identity that makes container replay safe.
ALTER TABLE raw_container_coordinates ADD COLUMN content_identity TEXT
    CHECK(content_identity IS NULL OR length(content_identity) = 64);
