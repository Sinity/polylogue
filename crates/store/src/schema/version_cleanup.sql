-- The outer UPSERT can override a trigger conflict policy. Select only absent keys.
CREATE TRIGGER message_source_removed AFTER DELETE ON message_source BEGIN
  INSERT INTO changed_version SELECT 'message',OLD.version_hash
    WHERE NOT EXISTS(SELECT 1 FROM changed_version WHERE kind='message' AND version_hash=OLD.version_hash);
END;
CREATE TRIGGER message_source_changed AFTER UPDATE OF version_hash ON message_source
WHEN OLD.version_hash <> NEW.version_hash BEGIN
  INSERT INTO changed_version SELECT 'message',OLD.version_hash
    WHERE NOT EXISTS(SELECT 1 FROM changed_version WHERE kind='message' AND version_hash=OLD.version_hash);
END;
CREATE TRIGGER document_source_removed AFTER DELETE ON document_source BEGIN
  INSERT INTO changed_version SELECT 'document',OLD.version_hash
    WHERE NOT EXISTS(SELECT 1 FROM changed_version WHERE kind='document' AND version_hash=OLD.version_hash);
END;
CREATE TRIGGER document_source_changed AFTER UPDATE OF version_hash ON document_source
WHEN OLD.version_hash <> NEW.version_hash BEGIN
  INSERT INTO changed_version SELECT 'document',OLD.version_hash
    WHERE NOT EXISTS(SELECT 1 FROM changed_version WHERE kind='document' AND version_hash=OLD.version_hash);
END;
CREATE TRIGGER event_source_removed AFTER DELETE ON event_source BEGIN
  INSERT INTO changed_value SELECT 'event',OLD.value_pk
    WHERE NOT EXISTS(SELECT 1 FROM changed_value WHERE kind='event' AND value_pk=OLD.value_pk);
END;
CREATE TRIGGER event_source_changed AFTER UPDATE OF value_pk ON event_source
WHEN OLD.value_pk <> NEW.value_pk BEGIN
  INSERT INTO changed_value SELECT 'event',OLD.value_pk
    WHERE NOT EXISTS(SELECT 1 FROM changed_value WHERE kind='event' AND value_pk=OLD.value_pk);
END;
CREATE TRIGGER link_source_removed AFTER DELETE ON link_source BEGIN
  INSERT INTO changed_value SELECT 'link',OLD.value_pk
    WHERE NOT EXISTS(SELECT 1 FROM changed_value WHERE kind='link' AND value_pk=OLD.value_pk);
END;
CREATE TRIGGER link_source_changed AFTER UPDATE OF value_pk ON link_source
WHEN OLD.value_pk <> NEW.value_pk BEGIN
  INSERT INTO changed_value SELECT 'link',OLD.value_pk
    WHERE NOT EXISTS(SELECT 1 FROM changed_value WHERE kind='link' AND value_pk=OLD.value_pk);
END;
CREATE TRIGGER block_removed AFTER DELETE ON block
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER block_changed AFTER UPDATE OF text_id ON block
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER document_block_removed AFTER DELETE ON document_block
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER document_block_changed AFTER UPDATE OF text_id ON document_block
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER message_version_text_removed AFTER DELETE ON message_version_text
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER message_version_text_changed AFTER UPDATE OF text_id ON message_version_text
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER document_version_text_removed AFTER DELETE ON document_version_text
WHEN OLD.text_id IS NOT NULL BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
CREATE TRIGGER document_version_text_changed AFTER UPDATE OF text_id ON document_version_text
WHEN OLD.text_id IS NOT NULL AND OLD.text_id IS NOT NEW.text_id BEGIN
  INSERT INTO changed_text SELECT OLD.text_id
    WHERE NOT EXISTS(SELECT 1 FROM changed_text WHERE text_id=OLD.text_id);
END;
