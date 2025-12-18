# Polylogue Database Migrations

This directory contains Alembic migrations for the polylogue SQLite database schema.

## Overview

Polylogue uses [Alembic](https://alembic.sqlalchemy.org/) for database schema migrations. Migrations are automatically applied when opening a database connection via `db.open_connection()`.

## Migration Files

Migrations are in `versions/` directory:

- `2025_12_10_1430_001_initial_schema.py` - Initial Schema v5 (December 2025)
  - Base tables: conversations, branches, messages, runs, state_meta, raw_imports
  - Attachments table with FTS5 search
  - Assets table with SHA-256 keys
  - Message_assets junction table
  - FTS5 triggers for automatic search sync
  - View for canonical transcript traversal

## Running Migrations

### Automatic (Normal Usage)

Migrations run automatically when you use polylogue:

```python
from polylogue import db

# Migrations run automatically on first connection
with db.open_connection() as conn:
    # Database is now at latest schema version
    ...
```

### Manual Migration Commands

You can also run migrations manually from the project root:

```bash
# Apply all pending migrations
alembic upgrade head

# Show current migration version
alembic current

# Show migration history
alembic history --verbose

# Downgrade one version
alembic downgrade -1

# Downgrade to specific version
alembic downgrade 001
```

## Creating New Migrations

When you need to change the database schema:

### 1. Create Migration File

```bash
# Auto-generate migration from schema changes
alembic revision --autogenerate -m "add new column to messages"

# Or create empty migration template
alembic revision -m "add new index"
```

This creates a new file in `versions/` with the current timestamp.

### 2. Edit Migration

Edit the generated file in `versions/`:

```python
def upgrade() -> None:
    """Add new column."""
    op.execute("ALTER TABLE messages ADD COLUMN new_field TEXT")

def downgrade() -> None:
    """Remove new column."""
    op.execute("ALTER TABLE messages DROP COLUMN new_field")
```

### 3. Test Migration

```bash
# Apply migration
alembic upgrade head

# Verify schema
sqlite3 ~/.config/polylogue/polylogue.db ".schema messages"

# Test rollback
alembic downgrade -1
alembic upgrade head
```

### 4. Commit Migration

```bash
git add polylogue/alembic/versions/2025_12_10_*.py
git commit -m "feat: add new_field column to messages table"
```

## Migration Guidelines

### DO:
- ✅ **Test migrations** - Apply, rollback, re-apply before committing
- ✅ **Write reversible migrations** - Implement both upgrade() and downgrade()
- ✅ **Use descriptive names** - "add_model_column" not "update_schema"
- ✅ **Add comments** - Explain WHY the migration is needed
- ✅ **Preserve data** - Use UPDATE statements when changing column types
- ✅ **Version bump** - Update SCHEMA_VERSION in db.py

### DON'T:
- ❌ **Edit old migrations** - Create new migration to fix issues
- ❌ **Delete migrations** - History must be complete
- ❌ **Skip versions** - Migrations must be sequential
- ❌ **Assume SQLAlchemy ORM** - Use raw SQL (op.execute)
- ❌ **Forget indexes** - Add indexes for foreign keys and queries

## Example: Adding a New Column

```python
"""Add tags column to conversations.

Revision ID: 002
Revises: 001
Create Date: 2025-12-11 10:00:00
"""

def upgrade() -> None:
    """Add tags column for conversation categorization."""
    op.execute("ALTER TABLE conversations ADD COLUMN tags TEXT")

    # Create index for tag searches
    op.execute("CREATE INDEX idx_conversations_tags ON conversations(tags)")

def downgrade() -> None:
    """Remove tags column."""
    op.execute("DROP INDEX idx_conversations_tags")
    # Note: SQLite doesn't support DROP COLUMN easily,
    # would need table recreation
    op.execute(
        """
        CREATE TABLE conversations_new AS
        SELECT provider, conversation_id, slug, title,
               current_branch, root_message_id, last_updated,
               content_hash, metadata_json
        FROM conversations
        """
    )
    op.execute("DROP TABLE conversations")
    op.execute("ALTER TABLE conversations_new RENAME TO conversations")
```

## Schema Version

The schema version is tracked in two places:

1. **Alembic version table** - `alembic_version` table tracks migration ID
2. **SQLite user_version pragma** - `PRAGMA user_version` stores numeric version (5)

Both should be kept in sync. The migration sets `PRAGMA user_version` explicitly.

## Troubleshooting

### "No tables exist" error

Database needs initialization:

```bash
alembic upgrade head
```

### "Alembic is not installed" error

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

### Migration conflicts

Multiple developers creating migrations simultaneously:

```bash
# Merge migrations (resolve in code)
alembic merge heads -m "merge migration branches"
```

### Reset database

**WARNING: This deletes all data!**

```bash
rm ~/.config/polylogue/polylogue.db
alembic upgrade head
```

## SQLite-Specific Notes

SQLite has limited ALTER TABLE support:

- ✅ Can add columns with `ADD COLUMN`
- ✅ Can rename tables with `RENAME TO`
- ❌ Cannot drop columns (need table recreation)
- ❌ Cannot modify column types (need table recreation)
- ❌ Cannot add/drop constraints (need table recreation)

For complex schema changes, use table recreation pattern:

```python
def upgrade():
    # Create new table with desired schema
    op.execute("CREATE TABLE messages_new (...)")

    # Copy data
    op.execute("INSERT INTO messages_new SELECT ... FROM messages")

    # Swap tables
    op.execute("DROP TABLE messages")
    op.execute("ALTER TABLE messages_new RENAME TO messages")

    # Recreate indexes/triggers
    op.execute("CREATE INDEX ...")
```

## Configuration

Alembic configuration is in `alembic.ini` at project root:

- Migration file naming: `%Y_%m_%d_%H%M_%(rev)s_%(slug)s`
- Auto-formatting: Runs `ruff check --fix` on new migrations
- Database URL: Dynamically set in `env.py` from polylogue.paths.STATE_HOME

## Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLite ALTER TABLE](https://www.sqlite.org/lang_altertable.html)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
