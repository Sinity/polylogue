-- Superseded journal roots are replaced atomically; retained journal rows stay intact.
DROP TRIGGER IF EXISTS raw_existence_delete;
DROP TRIGGER IF EXISTS raw_existence_key_change;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_sessions_insert AFTER INSERT ON raw_sessions
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_sessions_update AFTER UPDATE ON raw_sessions
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_sessions_update_old_key AFTER UPDATE OF raw_id ON raw_sessions
WHEN NEW.raw_id IS NOT OLD.raw_id
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_sessions_delete AFTER DELETE ON raw_sessions
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_artifacts_insert AFTER INSERT ON raw_artifacts
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_artifacts_update AFTER UPDATE ON raw_artifacts
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_artifacts_update_old_key AFTER UPDATE OF raw_id ON raw_artifacts
WHEN NEW.raw_id IS NOT OLD.raw_id
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_artifacts_delete AFTER DELETE ON raw_artifacts
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_session_memberships_insert AFTER INSERT ON raw_session_memberships
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_session_memberships_update AFTER UPDATE ON raw_session_memberships
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_session_memberships_update_old_key AFTER UPDATE OF raw_id ON raw_session_memberships
WHEN NEW.raw_id IS NOT OLD.raw_id
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_session_memberships_delete AFTER DELETE ON raw_session_memberships
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_membership_census_insert AFTER INSERT ON raw_membership_census
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_membership_census_update AFTER UPDATE ON raw_membership_census
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_membership_census_update_old_key AFTER UPDATE OF raw_id ON raw_membership_census
WHEN NEW.raw_id IS NOT OLD.raw_id
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_membership_census_delete AFTER DELETE ON raw_membership_census
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_authority_parser_census_insert AFTER INSERT ON raw_authority_parser_census
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_authority_parser_census_update AFTER UPDATE ON raw_authority_parser_census
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_authority_parser_census_update_old_key AFTER UPDATE OF raw_id ON raw_authority_parser_census
WHEN NEW.raw_id IS NOT OLD.raw_id
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_raw_authority_parser_census_delete AFTER DELETE ON raw_authority_parser_census
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.raw_id); END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_verified_blob_receipts_insert AFTER INSERT ON verified_blob_receipts
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=NEW.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_verified_blob_receipts_update AFTER UPDATE ON verified_blob_receipts
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=NEW.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_verified_blob_receipts_delete AFTER DELETE ON verified_blob_receipts
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=OLD.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_gc_generation_members_insert AFTER INSERT ON gc_generation_members
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=NEW.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_gc_generation_members_update AFTER UPDATE ON gc_generation_members
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=NEW.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_gc_generation_members_delete AFTER DELETE ON gc_generation_members
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=OLD.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_verified_blob_receipts_update_old_key AFTER UPDATE OF blob_hash ON verified_blob_receipts
WHEN NEW.blob_hash IS NOT OLD.blob_hash
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=OLD.blob_hash; END;
CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_gc_generation_members_update_old_key AFTER UPDATE OF blob_hash ON gc_generation_members
WHEN NEW.blob_hash IS NOT OLD.blob_hash
BEGIN INSERT INTO raw_existence_changes(raw_id) SELECT raw_id FROM raw_sessions WHERE blob_hash=OLD.blob_hash; END;

CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_blob_refs_insert AFTER INSERT ON blob_refs
WHEN NEW.ref_type='raw_payload'
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.ref_id); END;

CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_blob_refs_update AFTER UPDATE ON blob_refs
WHEN NEW.ref_type='raw_payload'
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (NEW.ref_id); END;

CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_blob_refs_delete AFTER DELETE ON blob_refs
WHEN OLD.ref_type='raw_payload'
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.ref_id); END;

CREATE TRIGGER IF NOT EXISTS raw_existence_frontier_blob_refs_update_old_key AFTER UPDATE ON blob_refs
WHEN OLD.ref_type='raw_payload' AND (NEW.ref_type IS NOT OLD.ref_type OR NEW.ref_id IS NOT OLD.ref_id OR NEW.source_path IS NOT OLD.source_path)
BEGIN INSERT INTO raw_existence_changes(raw_id) VALUES (OLD.ref_id); END;

CREATE INDEX IF NOT EXISTS idx_raw_authority_blockers_frontier_key
ON raw_authority_blockers(json_extract(expected_json,'$.logical_keys[0]'),blocker_id)
WHERE resolved_at_ms IS NULL
AND json_extract(expected_json,'$.authority_witness.schema')='polylogue.raw-authority-frontier-plan.v1';
