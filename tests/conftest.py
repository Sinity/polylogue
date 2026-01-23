from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_polylogue_env(monkeypatch):
    # Reset the container singleton to ensure clean state between tests
    from polylogue.cli.container import reset_container
    reset_container()

    for key in (
        "POLYLOGUE_CONFIG",
        "POLYLOGUE_ARCHIVE_ROOT",
        "POLYLOGUE_RENDER_ROOT",
        "POLYLOGUE_TEMPLATE_PATH",
        "POLYLOGUE_DECLARATIVE",
        # Prevent tests from hitting external Qdrant
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_dir / "config.json"))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    return {
        "config_path": config_dir / "config.json",
        "archive_root": archive_root,
        "state_root": state_dir,
    }


@pytest.fixture
def db_without_fts(tmp_path):
    """Database with schema but WITHOUT the FTS table (simulates fresh install)."""
    from polylogue.storage.db import open_connection

    db_path = tmp_path / "no_fts.db"
    with open_connection(db_path) as conn:
        # Drop the FTS table to simulate fresh install state
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.commit()
    return db_path


@pytest.fixture
def uppercase_json_inbox(tmp_path):
    """Inbox directory with uppercase extension files."""
    import json

    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    # Create files with various case combinations
    payload = {
        "id": "upper-conv",
        "messages": [{"id": "m1", "role": "user", "text": "uppercase test"}],
    }

    (inbox / "CHATGPT.JSON").write_text(json.dumps(payload), encoding="utf-8")
    (inbox / "Export.JSONL").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (inbox / "data.jsonl.txt").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    return inbox


@pytest.fixture
def storage_repository():
    """Storage repository with its own write lock.

    Use this fixture in tests that need thread-safe storage operations.
    The repository encapsulates the write lock and provides methods for
    saving conversations and recording runs.
    """
    from polylogue.storage.repository import StorageRepository

    return StorageRepository()
