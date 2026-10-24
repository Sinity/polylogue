-- Old writers must refuse this version: index validity now includes excisions.
INSERT INTO meta(key,value) VALUES ('excision_revision',
    CASE WHEN EXISTS(SELECT 1 FROM excision) THEN lower(hex(randomblob(16))) ELSE '' END);
INSERT INTO meta(key,value) VALUES ('excision_version',CASE WHEN EXISTS(SELECT 1 FROM excision) THEN '1' ELSE '0' END);
CREATE TRIGGER excision_revision_insert AFTER INSERT ON excision BEGIN
    UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
    UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
CREATE TRIGGER excision_revision_update AFTER UPDATE ON excision BEGIN
    UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
    UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
CREATE TRIGGER excision_revision_delete AFTER DELETE ON excision BEGIN
    UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='excision_version';
    UPDATE meta SET value=lower(hex(randomblob(16))) WHERE key='excision_revision';
END;
