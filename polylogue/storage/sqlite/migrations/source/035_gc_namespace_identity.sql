-- migration-safety: additive-no-backup
-- A pending physical deletion may resume only in the namespace it observed.
ALTER TABLE gc_generations ADD COLUMN blob_namespace_device INTEGER;
ALTER TABLE gc_generations ADD COLUMN blob_namespace_inode INTEGER;
