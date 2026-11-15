-- A session's counts (crates/store/src/derive/counts.rs) are a fold over
-- its own messages (no duplicate, not off the active path), their thinking
-- blocks and counted tool calls, and its compaction events, kept by these
-- triggers in the transaction of every change to those rows, including the
-- derive step's changes to duplicates and the active path. The first
-- recorded command a script ran stops the session's code calls counting
-- (session_sql::counts_as_action), so its tool counts are counted again.
CREATE TRIGGER message_counts_added AFTER INSERT ON message BEGIN
  UPDATE session SET
    message_count = COALESCE(message_count, 0) + 1,
    human_prompt_count = COALESCE(human_prompt_count, 0) + (NEW.written_by = 'human' AND NEW.prompt_kind IS 'prose'),
    human_command_count = COALESCE(human_command_count, 0) + (NEW.written_by = 'human' AND NEW.prompt_kind IS 'command'),
    word_count = COALESCE(word_count, 0) + (CASE WHEN NEW.written_by IN ('human', 'model') THEN NEW.words ELSE 0 END),
    human_word_count = COALESCE(human_word_count, 0) + (CASE WHEN NEW.written_by = 'human' THEN NEW.words ELSE 0 END),
    model_word_count = COALESCE(model_word_count, 0) + (CASE WHEN NEW.written_by = 'model' THEN NEW.words ELSE 0 END),
    thinking_count = COALESCE(thinking_count, 0) + EXISTS (SELECT 1 FROM block b WHERE b.message_pk = NEW.pk AND b.kind = 'thinking'),
    tool_call_count = COALESCE(tool_call_count, 0) + (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = NEW.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) + (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = NEW.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = NEW.session_pk AND (NEW.duplicate_of IS NULL AND COALESCE(NEW.on_active_path, 1) = 1);
END;
CREATE TRIGGER message_counts_removed AFTER DELETE ON message BEGIN
  UPDATE session SET
    message_count = COALESCE(message_count, 0) - 1,
    human_prompt_count = COALESCE(human_prompt_count, 0) - (OLD.written_by = 'human' AND OLD.prompt_kind IS 'prose'),
    human_command_count = COALESCE(human_command_count, 0) - (OLD.written_by = 'human' AND OLD.prompt_kind IS 'command'),
    word_count = COALESCE(word_count, 0) - (CASE WHEN OLD.written_by IN ('human', 'model') THEN OLD.words ELSE 0 END),
    human_word_count = COALESCE(human_word_count, 0) - (CASE WHEN OLD.written_by = 'human' THEN OLD.words ELSE 0 END),
    model_word_count = COALESCE(model_word_count, 0) - (CASE WHEN OLD.written_by = 'model' THEN OLD.words ELSE 0 END),
    thinking_count = COALESCE(thinking_count, 0) - EXISTS (SELECT 1 FROM block b WHERE b.message_pk = OLD.pk AND b.kind = 'thinking'),
    tool_call_count = COALESCE(tool_call_count, 0) - (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = OLD.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) - (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = OLD.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = OLD.session_pk AND (OLD.duplicate_of IS NULL AND COALESCE(OLD.on_active_path, 1) = 1);
END;
CREATE TRIGGER message_counts_changed AFTER UPDATE OF session_pk, duplicate_of, on_active_path, written_by, prompt_kind, words ON message
WHEN OLD.session_pk IS NOT NEW.session_pk OR OLD.duplicate_of IS NOT NEW.duplicate_of OR OLD.on_active_path IS NOT NEW.on_active_path
  OR OLD.written_by IS NOT NEW.written_by OR OLD.prompt_kind IS NOT NEW.prompt_kind OR OLD.words IS NOT NEW.words
BEGIN
  UPDATE session SET
    message_count = COALESCE(message_count, 0) - 1,
    human_prompt_count = COALESCE(human_prompt_count, 0) - (OLD.written_by = 'human' AND OLD.prompt_kind IS 'prose'),
    human_command_count = COALESCE(human_command_count, 0) - (OLD.written_by = 'human' AND OLD.prompt_kind IS 'command'),
    word_count = COALESCE(word_count, 0) - (CASE WHEN OLD.written_by IN ('human', 'model') THEN OLD.words ELSE 0 END),
    human_word_count = COALESCE(human_word_count, 0) - (CASE WHEN OLD.written_by = 'human' THEN OLD.words ELSE 0 END),
    model_word_count = COALESCE(model_word_count, 0) - (CASE WHEN OLD.written_by = 'model' THEN OLD.words ELSE 0 END),
    thinking_count = COALESCE(thinking_count, 0) - EXISTS (SELECT 1 FROM block b WHERE b.message_pk = OLD.pk AND b.kind = 'thinking'),
    tool_call_count = COALESCE(tool_call_count, 0) - (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = OLD.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) - (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = OLD.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = OLD.session_pk AND (OLD.duplicate_of IS NULL AND COALESCE(OLD.on_active_path, 1) = 1);
  UPDATE session SET
    message_count = COALESCE(message_count, 0) + 1,
    human_prompt_count = COALESCE(human_prompt_count, 0) + (NEW.written_by = 'human' AND NEW.prompt_kind IS 'prose'),
    human_command_count = COALESCE(human_command_count, 0) + (NEW.written_by = 'human' AND NEW.prompt_kind IS 'command'),
    word_count = COALESCE(word_count, 0) + (CASE WHEN NEW.written_by IN ('human', 'model') THEN NEW.words ELSE 0 END),
    human_word_count = COALESCE(human_word_count, 0) + (CASE WHEN NEW.written_by = 'human' THEN NEW.words ELSE 0 END),
    model_word_count = COALESCE(model_word_count, 0) + (CASE WHEN NEW.written_by = 'model' THEN NEW.words ELSE 0 END),
    thinking_count = COALESCE(thinking_count, 0) + EXISTS (SELECT 1 FROM block b WHERE b.message_pk = NEW.pk AND b.kind = 'thinking'),
    tool_call_count = COALESCE(tool_call_count, 0) + (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = NEW.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) + (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk WHERE b.message_pk = NEW.pk AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = NEW.session_pk AND (NEW.duplicate_of IS NULL AND COALESCE(NEW.on_active_path, 1) = 1);
END;
CREATE TRIGGER block_counts_added AFTER INSERT ON block WHEN NEW.kind = 'thinking' AND NOT EXISTS (SELECT 1 FROM block o WHERE o.message_pk = NEW.message_pk AND o.kind = 'thinking' AND o.pk <> NEW.pk) BEGIN
  UPDATE session SET thinking_count = COALESCE(thinking_count, 0) + 1
  WHERE pk = (SELECT m.session_pk FROM message m WHERE m.pk = NEW.message_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1));
END;
CREATE TRIGGER block_counts_removed AFTER DELETE ON block WHEN OLD.kind = 'thinking' AND NOT EXISTS (SELECT 1 FROM block o WHERE o.message_pk = OLD.message_pk AND o.kind = 'thinking' AND o.pk <> OLD.pk) BEGIN
  UPDATE session SET thinking_count = COALESCE(thinking_count, 0) - 1
  WHERE pk = (SELECT m.session_pk FROM message m WHERE m.pk = OLD.message_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1));
END;
CREATE TRIGGER tool_call_counts_added AFTER INSERT ON tool_call WHEN NOT (NEW.run_by_script = 1 AND NOT EXISTS (SELECT 1 FROM tool_call o WHERE o.session_pk = NEW.session_pk AND o.run_by_script = 1 AND o.pk <> NEW.pk)) BEGIN
  UPDATE session SET
    tool_call_count = COALESCE(tool_call_count, 0) + (NOT (COALESCE(NEW.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = NEW.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) + (NOT (COALESCE(NEW.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = NEW.session_pk AND cw.run_by_script = 1)) AND NEW.outcome = 'error')
  WHERE pk = (SELECT m.session_pk FROM block b JOIN message m ON m.pk = b.message_pk
              WHERE b.pk = NEW.call_block_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1));
END;
CREATE TRIGGER tool_call_counts_script_added AFTER INSERT ON tool_call WHEN NEW.run_by_script = 1 AND NOT EXISTS (SELECT 1 FROM tool_call o WHERE o.session_pk = NEW.session_pk AND o.run_by_script = 1 AND o.pk <> NEW.pk) BEGIN
  UPDATE session SET
    tool_call_count = (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = NEW.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = NEW.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = NEW.session_pk;
END;
CREATE TRIGGER tool_call_counts_removed AFTER DELETE ON tool_call WHEN NOT (OLD.run_by_script = 1 AND NOT EXISTS (SELECT 1 FROM tool_call o WHERE o.session_pk = OLD.session_pk AND o.run_by_script = 1 AND o.pk <> OLD.pk)) BEGIN
  UPDATE session SET
    tool_call_count = COALESCE(tool_call_count, 0) - (NOT (COALESCE(OLD.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = OLD.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) - (NOT (COALESCE(OLD.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = OLD.session_pk AND cw.run_by_script = 1)) AND OLD.outcome = 'error')
  WHERE pk = (SELECT m.session_pk FROM block b JOIN message m ON m.pk = b.message_pk
              WHERE b.pk = OLD.call_block_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1));
END;
CREATE TRIGGER tool_call_counts_script_removed AFTER DELETE ON tool_call WHEN OLD.run_by_script = 1 AND NOT EXISTS (SELECT 1 FROM tool_call o WHERE o.session_pk = OLD.session_pk AND o.run_by_script = 1 AND o.pk <> OLD.pk) BEGIN
  UPDATE session SET
    tool_call_count = (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = OLD.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = OLD.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = OLD.session_pk;
END;
CREATE TRIGGER tool_call_counts_changed AFTER UPDATE OF action, outcome, call_block_pk ON tool_call
WHEN OLD.action IS NOT NEW.action OR OLD.outcome IS NOT NEW.outcome OR OLD.call_block_pk IS NOT NEW.call_block_pk
BEGIN
  UPDATE session SET
    tool_call_count = COALESCE(tool_call_count, 0) - (NOT (COALESCE(OLD.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = OLD.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) - (NOT (COALESCE(OLD.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = OLD.session_pk AND cw.run_by_script = 1)) AND OLD.outcome = 'error')
  WHERE pk = (SELECT m.session_pk FROM block b JOIN message m ON m.pk = b.message_pk
              WHERE b.pk = OLD.call_block_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1));
  UPDATE session SET
    tool_call_count = COALESCE(tool_call_count, 0) + (NOT (COALESCE(NEW.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = NEW.session_pk AND cw.run_by_script = 1))),
    error_count = COALESCE(error_count, 0) + (NOT (COALESCE(NEW.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = NEW.session_pk AND cw.run_by_script = 1)) AND NEW.outcome = 'error')
  WHERE pk = (SELECT m.session_pk FROM block b JOIN message m ON m.pk = b.message_pk
              WHERE b.pk = NEW.call_block_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1));
END;
CREATE TRIGGER tool_call_counts_script_changed AFTER UPDATE OF run_by_script, session_pk ON tool_call
WHEN OLD.run_by_script IS NOT NEW.run_by_script OR OLD.session_pk IS NOT NEW.session_pk
BEGIN
  UPDATE session SET
    tool_call_count = (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = OLD.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = OLD.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = OLD.session_pk;
  UPDATE session SET
    tool_call_count = (SELECT count(*) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = NEW.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1))),
    error_count = (SELECT COALESCE(sum(t.outcome = 'error'), 0) FROM tool_call t JOIN block b ON b.pk = t.call_block_pk JOIN message m ON m.pk = b.message_pk WHERE m.session_pk = NEW.session_pk AND (m.duplicate_of IS NULL AND COALESCE(m.on_active_path, 1) = 1) AND NOT (COALESCE(t.action, '') = 'code' AND EXISTS (SELECT 1 FROM tool_call cw WHERE cw.session_pk = t.session_pk AND cw.run_by_script = 1)))
  WHERE pk = NEW.session_pk;
END;
CREATE TRIGGER event_counts_added AFTER INSERT ON event WHEN NEW.kind = 'compaction' BEGIN
  UPDATE session SET compaction_count = COALESCE(compaction_count, 0) + (NEW.kind = 'compaction') WHERE pk = NEW.session_pk;
END;
CREATE TRIGGER event_counts_removed AFTER DELETE ON event WHEN OLD.kind = 'compaction' BEGIN
  UPDATE session SET compaction_count = COALESCE(compaction_count, 0) - (OLD.kind = 'compaction') WHERE pk = OLD.session_pk;
END;
CREATE TRIGGER event_counts_changed AFTER UPDATE OF kind, session_pk ON event
WHEN (OLD.kind = 'compaction' OR NEW.kind = 'compaction') AND (OLD.kind IS NOT NEW.kind OR OLD.session_pk IS NOT NEW.session_pk)
BEGIN
  UPDATE session SET compaction_count = COALESCE(compaction_count, 0) - (OLD.kind = 'compaction') WHERE pk = OLD.session_pk;
  UPDATE session SET compaction_count = COALESCE(compaction_count, 0) + (NEW.kind = 'compaction') WHERE pk = NEW.session_pk;
END;
