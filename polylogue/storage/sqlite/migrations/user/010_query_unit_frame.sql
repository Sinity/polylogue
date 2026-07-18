-- migration-safety: additive-no-backup
-- Tier-local user frame for query-unit continuation correctness. Existing
-- continuations are explicitly decoded as legacy q1 tokens; q2 tokens always
-- carry a frame captured from the reader snapshot.
CREATE TABLE IF NOT EXISTS query_unit_frame_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    epoch INTEGER NOT NULL DEFAULT 0 CHECK (epoch >= 0)
) STRICT;

INSERT OR IGNORE INTO query_unit_frame_state(singleton, epoch) VALUES (1, 0);

CREATE TRIGGER IF NOT EXISTS query_unit_frame_assertions_insert
AFTER INSERT ON assertions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_assertions_update
AFTER UPDATE ON assertions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
CREATE TRIGGER IF NOT EXISTS query_unit_frame_assertions_delete
AFTER DELETE ON assertions BEGIN
    UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE singleton = 1;
END;
