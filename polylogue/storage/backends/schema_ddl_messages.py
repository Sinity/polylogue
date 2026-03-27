"""Message-search schema DDL fragments."""

from __future__ import annotations

MESSAGE_FTS_DDL = """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            text,
            tokenize='unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS messages_fts_ai
        AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_ad
        AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
        END;

        CREATE TRIGGER IF NOT EXISTS messages_fts_au
        AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.rowid;
            INSERT INTO messages_fts(rowid, message_id, conversation_id, text)
            VALUES (new.rowid, new.message_id, new.conversation_id, new.text);
        END;
"""


__all__ = ["MESSAGE_FTS_DDL"]
