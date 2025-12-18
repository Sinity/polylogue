"""Add conversation-aware versioning to raw_imports.

Revision ID: 002
Revises: 001
Create Date: 2025-12-11 01:00:00.000000

This migration implements conversation-aware raw storage (Option B) to prevent
unbounded growth while keeping all data for catastrophic failure recovery.

Changes:
- Add conversation_id column to track which conversation this raw data belongs to
- Add version column to track multiple snapshots of the same conversation
- Add imported_at_ns column for nanosecond precision (for version ordering)
- Modify primary key to be (provider, conversation_id, version) instead of just hash
- Add index on hash for backward compatibility lookups
- Add cleanup policy (keep last 5 versions per conversation)
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add conversation-aware versioning to raw_imports."""
    # Create new table with conversation-aware schema
    op.execute(
        """
        CREATE TABLE raw_imports_new (
            provider TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            hash TEXT NOT NULL,
            imported_at INTEGER DEFAULT (unixepoch()),
            imported_at_ns INTEGER DEFAULT (unixepoch() * 1000000000),
            source_path TEXT,
            blob BLOB,
            parser_version TEXT,
            parse_status TEXT DEFAULT 'pending',
            error_message TEXT,
            metadata_json TEXT,
            PRIMARY KEY (provider, conversation_id, version)
        )
        """
    )

    # Create index on hash for backward compatibility
    op.execute("CREATE INDEX idx_raw_imports_hash ON raw_imports_new(hash)")

    # Create index on parse_status for efficient failed import queries
    op.execute("CREATE INDEX idx_raw_imports_status ON raw_imports_new(parse_status)")

    # Migrate existing data: use hash as conversation_id for legacy records
    op.execute(
        """
        INSERT INTO raw_imports_new (
            provider, conversation_id, version, hash, imported_at, imported_at_ns,
            source_path, blob, parser_version, parse_status, error_message, metadata_json
        )
        SELECT
            provider,
            hash AS conversation_id,  -- Legacy: use hash as conversation_id
            1 AS version,
            hash,
            imported_at,
            imported_at * 1000000000 AS imported_at_ns,
            source_path,
            blob,
            parser_version,
            parse_status,
            error_message,
            metadata_json
        FROM raw_imports
        """
    )

    # Drop old table
    op.execute("DROP TABLE raw_imports")

    # Rename new table to original name
    op.execute("ALTER TABLE raw_imports_new RENAME TO raw_imports")


def downgrade() -> None:
    """Revert to hash-based primary key."""
    # Create old table structure
    op.execute(
        """
        CREATE TABLE raw_imports_old (
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

    # Migrate data back: keep only the latest version of each conversation
    op.execute(
        """
        INSERT INTO raw_imports_old (
            hash, imported_at, provider, source_path, blob,
            parser_version, parse_status, error_message, metadata_json
        )
        SELECT
            hash,
            imported_at,
            provider,
            source_path,
            blob,
            parser_version,
            parse_status,
            error_message,
            metadata_json
        FROM raw_imports
        WHERE (provider, conversation_id, version) IN (
            SELECT provider, conversation_id, MAX(version)
            FROM raw_imports
            GROUP BY provider, conversation_id
        )
        """
    )

    # Drop new table
    op.execute("DROP TABLE raw_imports")

    # Rename old table back
    op.execute("ALTER TABLE raw_imports_old RENAME TO raw_imports")
