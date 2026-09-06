-- migration-safety: additive-no-backup
ALTER TABLE raw_container_coordinates ADD COLUMN addressing_mode TEXT;
