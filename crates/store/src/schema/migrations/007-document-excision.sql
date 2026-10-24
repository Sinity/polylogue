-- Standing document selections are canonical rows.
-- A new archive version prevents older writers from discarding this policy;
-- publication includes these rows in the existing excision unit.
CREATE TABLE excision_document (
  excision_id TEXT NOT NULL REFERENCES excision(id) ON DELETE CASCADE,
  document_id TEXT NOT NULL,
  PRIMARY KEY(excision_id,document_id)
) STRICT, WITHOUT ROWID;
CREATE TRIGGER document_policy_insert AFTER INSERT ON excision_document BEGIN
 UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
CREATE TRIGGER document_policy_update AFTER UPDATE ON excision_document BEGIN
 UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
CREATE TRIGGER document_policy_delete AFTER DELETE ON excision_document BEGIN
 UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
 UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
