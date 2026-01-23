"""Tests for StorageBackend protocol conformance."""

from __future__ import annotations

from pathlib import Path

from polylogue.protocols import StorageBackend
from polylogue.storage.backends import SQLiteBackend


def test_sqlite_backend_implements_protocol(tmp_path: Path) -> None:
    """Verify SQLiteBackend implements StorageBackend protocol."""
    backend = SQLiteBackend(db_path=tmp_path / "test.db")
    assert isinstance(backend, StorageBackend)
    backend.close()


def test_backend_has_all_required_methods(tmp_path: Path) -> None:
    """Verify backend has all protocol methods."""
    backend = SQLiteBackend(db_path=tmp_path / "test.db")

    # Check all required methods exist
    assert hasattr(backend, "get_conversation")
    assert hasattr(backend, "list_conversations")
    assert hasattr(backend, "save_conversation")
    assert hasattr(backend, "get_messages")
    assert hasattr(backend, "save_messages")
    assert hasattr(backend, "get_attachments")
    assert hasattr(backend, "save_attachments")
    assert hasattr(backend, "begin")
    assert hasattr(backend, "commit")
    assert hasattr(backend, "rollback")

    # Check they are callable
    assert callable(backend.get_conversation)
    assert callable(backend.list_conversations)
    assert callable(backend.save_conversation)
    assert callable(backend.get_messages)
    assert callable(backend.save_messages)
    assert callable(backend.get_attachments)
    assert callable(backend.save_attachments)
    assert callable(backend.begin)
    assert callable(backend.commit)
    assert callable(backend.rollback)

    backend.close()
