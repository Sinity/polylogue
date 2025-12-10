"""Initial schema v5 with database-first architecture.

Revision ID: 001
Revises:
Create Date: 2025-12-10 14:30:00

This is the initial migration containing Schema v5:
- Base tables: conversations, branches, messages, runs, state_meta, raw_imports
- Attachments table with FTS5 search
- Assets table with SHA-256 keys
- Message_assets junction table
- FTS5 triggers for automatic search sync
- View for canonical transcript traversal
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial schema v5."""
    # Enable foreign keys
    op.execute("PRAGMA foreign_keys = ON")

    # Base tables
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            slug TEXT NOT NULL,
            title TEXT,
            current_branch TEXT,
            root_message_id TEXT,
            last_updated TEXT,
            content_hash TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS branches (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            parent_branch_id TEXT,
            label TEXT,
            depth INTEGER DEFAULT 0,
            is_current INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id),
            FOREIGN KEY (provider, conversation_id)
                REFERENCES conversations(provider, conversation_id) ON DELETE CASCADE
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            parent_id TEXT,
            position INTEGER NOT NULL,
            timestamp TEXT,
            role TEXT,
            model TEXT,
            content_hash TEXT,
            content_text TEXT,
            content_json TEXT,
            rendered_text TEXT,
            raw_json TEXT,
            token_count INTEGER DEFAULT 0,
            word_count INTEGER DEFAULT 0,
            attachment_count INTEGER DEFAULT 0,
            attachment_names TEXT,
            is_leaf INTEGER DEFAULT 0,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id, position),
            FOREIGN KEY (provider, conversation_id, branch_id)
                REFERENCES branches(provider, conversation_id, branch_id) ON DELETE CASCADE
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            cmd TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            attachments INTEGER DEFAULT 0,
            attachment_bytes INTEGER DEFAULT 0,
            tokens INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0,
            pruned INTEGER DEFAULT 0,
            diffs INTEGER DEFAULT 0,
            duration REAL,
            out TEXT,
            provider TEXT,
            branch_id TEXT,
            metadata_json TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS state_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS raw_imports (
            hash TEXT PRIMARY KEY,
            imported_at INTEGER DEFAULT (unixepoch()),
            provider TEXT NOT NULL,
            source_path TEXT,
            blob BLOB,
            parser_version TEXT,
            parse_status TEXT DEFAULT 'pending',
            error_message TEXT,
            metadata_json TEXT
        )
        """
    )

    # Indexes for base tables
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_raw_imports_status
            ON raw_imports(parse_status)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_raw_imports_provider
            ON raw_imports(provider, imported_at DESC)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_branch_order
            ON messages(provider, conversation_id, branch_id, position)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_message
            ON messages(provider, conversation_id, message_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_branches_conversation
            ON branches(provider, conversation_id)
        """
    )

    # FTS5 table for messages
    op.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            provider,
            conversation_id,
            branch_id,
            message_id,
            content,
            tokenize='unicode61'
        )
        """
    )

    # Attachments table and FTS
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT,
            message_id TEXT,
            attachment_name TEXT NOT NULL,
            attachment_path TEXT,
            size_bytes INTEGER,
            content_hash TEXT,
            mime_type TEXT,
            text_content TEXT,
            text_bytes INTEGER,
            truncated INTEGER DEFAULT 0,
            ocr_used INTEGER DEFAULT 0,
            PRIMARY KEY (provider, conversation_id, branch_id, message_id, attachment_name),
            FOREIGN KEY (provider, conversation_id)
                REFERENCES conversations(provider, conversation_id) ON DELETE CASCADE,
            FOREIGN KEY (provider, conversation_id, branch_id)
                REFERENCES branches(provider, conversation_id, branch_id) ON DELETE CASCADE
        )
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_attachments_conversation
            ON attachments(provider, conversation_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_attachments_name
            ON attachments(attachment_name)
        """
    )

    op.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS attachments_fts USING fts5(
            provider,
            conversation_id,
            branch_id,
            message_id,
            attachment_name,
            content,
            tokenize='unicode61'
        )
        """
    )

    # Schema v5: Assets table with SHA-256 keys
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS assets (
            id TEXT PRIMARY KEY,
            mime_type TEXT,
            size_bytes INTEGER,
            data BLOB,
            local_path TEXT,
            created_at INTEGER DEFAULT (unixepoch())
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS message_assets (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            filename TEXT,
            PRIMARY KEY (provider, conversation_id, branch_id, message_id, asset_id),
            FOREIGN KEY (provider, conversation_id, branch_id, message_id)
                REFERENCES messages(provider, conversation_id, branch_id, message_id)
                ON DELETE CASCADE,
            FOREIGN KEY (asset_id) REFERENCES assets(id) ON DELETE CASCADE
        )
        """
    )

    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_assets_size ON assets(size_bytes)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_message_assets_message
            ON message_assets(provider, conversation_id, branch_id, message_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_message_assets_asset ON message_assets(asset_id)
        """
    )

    # Schema v5: FTS5 triggers for automatic sync
    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(provider, conversation_id, branch_id, message_id, content)
            VALUES (new.provider, new.conversation_id, new.branch_id, new.message_id,
                    COALESCE(new.content_text, '') || ' ' || COALESCE(new.rendered_text, ''));
        END
        """
    )

    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts
            WHERE provider = old.provider
              AND conversation_id = old.conversation_id
              AND branch_id = old.branch_id
              AND message_id = old.message_id;
            INSERT INTO messages_fts(provider, conversation_id, branch_id, message_id, content)
            VALUES (new.provider, new.conversation_id, new.branch_id, new.message_id,
                    COALESCE(new.content_text, '') || ' ' || COALESCE(new.rendered_text, ''));
        END
        """
    )

    op.execute(
        """
        CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts
            WHERE provider = old.provider
              AND conversation_id = old.conversation_id
              AND branch_id = old.branch_id
              AND message_id = old.message_id;
        END
        """
    )

    # Schema v5: View for canonical transcript traversal
    op.execute(
        """
        CREATE VIEW IF NOT EXISTS view_canonical_transcript AS
        WITH RECURSIVE chat_tree(
            provider, conversation_id, branch_id, message_id, parent_id,
            position, content_text, role, timestamp, depth
        ) AS (
            -- Start from root (position 0)
            SELECT
                provider, conversation_id, branch_id, message_id, parent_id,
                position, content_text, role, timestamp, 0 as depth
            FROM messages
            WHERE position = 0
            UNION ALL
            -- Traverse down the tree
            SELECT
                m.provider, m.conversation_id, m.branch_id, m.message_id, m.parent_id,
                m.position, m.content_text, m.role, m.timestamp, ct.depth + 1
            FROM messages m
            JOIN chat_tree ct ON m.parent_id = ct.message_id
                AND m.provider = ct.provider
                AND m.conversation_id = ct.conversation_id
        )
        SELECT * FROM chat_tree ORDER BY provider, conversation_id, position
        """
    )

    # Set schema version in user_version pragma
    op.execute("PRAGMA user_version = 5")


def downgrade() -> None:
    """Drop all tables and views."""
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS messages_fts_delete")
    op.execute("DROP TRIGGER IF EXISTS messages_fts_update")
    op.execute("DROP TRIGGER IF EXISTS messages_fts_insert")

    # Drop view
    op.execute("DROP VIEW IF EXISTS view_canonical_transcript")

    # Drop FTS tables
    op.execute("DROP TABLE IF EXISTS attachments_fts")
    op.execute("DROP TABLE IF EXISTS messages_fts")

    # Drop junction and assets tables
    op.execute("DROP TABLE IF EXISTS message_assets")
    op.execute("DROP TABLE IF EXISTS assets")

    # Drop attachments
    op.execute("DROP TABLE IF EXISTS attachments")

    # Drop main tables (in reverse dependency order)
    op.execute("DROP TABLE IF EXISTS raw_imports")
    op.execute("DROP TABLE IF EXISTS state_meta")
    op.execute("DROP TABLE IF EXISTS runs")
    op.execute("DROP TABLE IF EXISTS messages")
    op.execute("DROP TABLE IF EXISTS branches")
    op.execute("DROP TABLE IF EXISTS conversations")

    # Reset schema version
    op.execute("PRAGMA user_version = 0")
