CREATE TABLE vector_deletion_policy (
 excision_id TEXT PRIMARY KEY REFERENCES excision(id),
 version INTEGER NOT NULL,
 hashes_json TEXT NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO meta(key,value) VALUES ('vector_policy_revision','');
CREATE TRIGGER vector_policy_insert AFTER INSERT ON vector_deletion_policy BEGIN
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='vector_policy_revision';
 UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
CREATE TRIGGER vector_policy_update AFTER UPDATE ON vector_deletion_policy BEGIN
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='vector_policy_revision';
 UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
CREATE TRIGGER vector_policy_delete AFTER DELETE ON vector_deletion_policy BEGIN
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='vector_policy_revision';
 UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
