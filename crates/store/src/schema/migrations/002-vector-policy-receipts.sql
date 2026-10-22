CREATE TABLE vector_policy_receipt (
 excision_id TEXT PRIMARY KEY,
 archive_id TEXT NOT NULL,
 policy_hash BLOB NOT NULL CHECK(length(policy_hash)=32)
) STRICT, WITHOUT ROWID;
CREATE TABLE vector_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
CREATE TRIGGER vector_receipt_insert AFTER INSERT ON vector_policy_receipt BEGIN
 DELETE FROM vector_meta WHERE key='policy_revision';
END;
CREATE TRIGGER vector_receipt_update AFTER UPDATE ON vector_policy_receipt BEGIN
 DELETE FROM vector_meta WHERE key='policy_revision';
END;
CREATE TRIGGER vector_receipt_delete AFTER DELETE ON vector_policy_receipt BEGIN
 DELETE FROM vector_meta WHERE key='policy_revision';
END;
