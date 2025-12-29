"""Alembic environment configuration for polylogue database migrations."""

from __future__ import annotations

import sqlite3
from logging.config import fileConfig
from pathlib import Path

from alembic import context

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)


def get_url() -> str:
    """Get database URL from polylogue configuration."""
    from polylogue.paths import STATE_HOME

    db_path = STATE_HOME / "polylogue.db"
    return f"sqlite:///{db_path}"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=None,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    # For SQLite, we use a simple connection rather than SQLAlchemy Engine
    url = get_url()
    db_path = url.replace("sqlite:///", "")

    # Ensure parent directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create connection
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    try:
        with context.begin_transaction():
            context.run_migrations()
    finally:
        conn.close()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
