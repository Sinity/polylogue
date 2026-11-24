-- Search membership and the trigram index follow the shown rows. A text is a
-- member of the prose index (text.indexed bit 1) while a prose block or a
-- document block holds it, and of the tool-output index (bit 2) while a tool
-- call or tool result block holds it. A text that joins an index is queued
-- for the search thread; one whose last holder of that kind leaves loses its
-- entry and any queued one. A build creates these after its composition,
-- which sets every text's membership with one statement over the blocks.

CREATE TRIGGER block_prose_text_added AFTER INSERT ON block
WHEN NEW.text_id IS NOT NULL AND NEW.kind NOT IN ('tool_result','tool_call') BEGIN
  INSERT OR IGNORE INTO search_pending(text_id, fts_table)
    SELECT id, 1 FROM text WHERE id = NEW.text_id AND indexed & 1 = 0;
  UPDATE text SET indexed = indexed | 1 WHERE id = NEW.text_id AND indexed & 1 = 0;
END;

CREATE TRIGGER block_output_text_added AFTER INSERT ON block
WHEN NEW.text_id IS NOT NULL AND NEW.kind IN ('tool_result','tool_call') BEGIN
  INSERT OR IGNORE INTO search_pending(text_id, fts_table)
    SELECT id, 2 FROM text WHERE id = NEW.text_id AND indexed & 2 = 0;
  UPDATE text SET indexed = indexed | 2 WHERE id = NEW.text_id AND indexed & 2 = 0;
END;

CREATE TRIGGER document_block_text_added AFTER INSERT ON document_block
WHEN NEW.text_id IS NOT NULL BEGIN
  INSERT OR IGNORE INTO search_pending(text_id, fts_table)
    SELECT id, 1 FROM text WHERE id = NEW.text_id AND indexed & 1 = 0;
  UPDATE text SET indexed = indexed | 1 WHERE id = NEW.text_id AND indexed & 1 = 0;
END;

CREATE TRIGGER block_prose_text_removed AFTER DELETE ON block
WHEN OLD.text_id IS NOT NULL AND OLD.kind NOT IN ('tool_result','tool_call')
  AND NOT EXISTS(SELECT 1 FROM block WHERE text_id = OLD.text_id
                 AND kind NOT IN ('tool_result','tool_call'))
  AND NOT EXISTS(SELECT 1 FROM document_block WHERE text_id = OLD.text_id) BEGIN
  DELETE FROM text_fts WHERE rowid = OLD.text_id;
  DELETE FROM search_pending WHERE text_id = OLD.text_id AND fts_table = 1;
  UPDATE text SET indexed = indexed & ~1 WHERE id = OLD.text_id AND indexed & 1 <> 0;
END;

CREATE TRIGGER block_output_text_removed AFTER DELETE ON block
WHEN OLD.text_id IS NOT NULL AND OLD.kind IN ('tool_result','tool_call')
  AND NOT EXISTS(SELECT 1 FROM block WHERE text_id = OLD.text_id
                 AND kind IN ('tool_result','tool_call')) BEGIN
  DELETE FROM output_fts WHERE rowid = OLD.text_id;
  DELETE FROM search_pending WHERE text_id = OLD.text_id AND fts_table = 2;
  UPDATE text SET indexed = indexed & ~2 WHERE id = OLD.text_id AND indexed & 2 <> 0;
END;

CREATE TRIGGER document_block_text_removed AFTER DELETE ON document_block
WHEN OLD.text_id IS NOT NULL
  AND NOT EXISTS(SELECT 1 FROM block WHERE text_id = OLD.text_id
                 AND kind NOT IN ('tool_result','tool_call'))
  AND NOT EXISTS(SELECT 1 FROM document_block WHERE text_id = OLD.text_id) BEGIN
  DELETE FROM text_fts WHERE rowid = OLD.text_id;
  DELETE FROM search_pending WHERE text_id = OLD.text_id AND fts_table = 1;
  UPDATE text SET indexed = indexed & ~1 WHERE id = OLD.text_id AND indexed & 1 <> 0;
END;

-- tool_call_trigram reads its text from tool_call (external content), so an
-- entry is removed with the values it was entered with.
CREATE TRIGGER tool_call_trigram_added AFTER INSERT ON tool_call BEGIN
  INSERT INTO tool_call_trigram(rowid, command, path) VALUES (NEW.pk, NEW.command, NEW.path);
END;

CREATE TRIGGER tool_call_trigram_removed AFTER DELETE ON tool_call BEGIN
  INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
    VALUES ('delete', OLD.pk, OLD.command, OLD.path);
END;

CREATE TRIGGER tool_call_trigram_changed AFTER UPDATE OF command, path ON tool_call
WHEN OLD.command IS NOT NEW.command OR OLD.path IS NOT NEW.path BEGIN
  INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
    VALUES ('delete', OLD.pk, OLD.command, OLD.path);
  INSERT INTO tool_call_trigram(rowid, command, path) VALUES (NEW.pk, NEW.command, NEW.path);
END;
