-- migration-safety: additive-backup-required
-- polylogue-f1s9a, executing the 2026-09-15 ruling recorded on polylogue-6kur:
-- the raw-authority census ledger is per-pass repair bookkeeping and does not
-- survive the fresh start.  The artifact-census slice introduced at v31
-- (031_raw_authority_artifact_census_receipts.sql) was already omitted from
-- fresh source generations -- it is listed in RETIRED_SOURCE_SCHEMA_OBJECTS --
-- but the objects persisted in every migrated historical database and have no
-- remaining reader or writer anywhere in product code.  Drop them for real.
--
-- Nothing copies forward: these tables are checkpoint/receipt bookkeeping for a
-- census that is itself retired, and the raws they page over remain untouched
-- in raw_sessions.  Children first so the declared foreign keys verify the
-- order under PRAGMA foreign_keys = ON.
DROP TRIGGER IF EXISTS invalidate_pending_raw_authority_artifact_census_checkpoint_on_raw_delete;
DROP INDEX IF EXISTS idx_raw_authority_artifact_census_checkpoint_members_page;
DROP INDEX IF EXISTS idx_raw_authority_artifact_census_receipts_applied_at;
DROP INDEX IF EXISTS idx_raw_sessions_raw_authority_census_candidates;
DROP TABLE IF EXISTS raw_authority_artifact_census_checkpoint_members;
DROP TABLE IF EXISTS raw_authority_artifact_census_checkpoints;
DROP TABLE IF EXISTS raw_authority_artifact_census_receipts;
