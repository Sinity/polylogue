"""Fixtures for CLI tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from polylogue.storage.index import rebuild_index
from tests.infra.helpers import DbFactory


@pytest.fixture
def search_workspace(cli_workspace, monkeypatch):
    """CLI workspace with searchable conversations."""
    # Set up environment
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cli_workspace["config_path"]))
    monkeypatch.setenv("XDG_STATE_HOME", str(cli_workspace["state_dir"]))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(cli_workspace["archive_root"]))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Create sample conversations with searchable content
    db_path = cli_workspace["db_path"]
    factory = DbFactory(db_path)

    # Conversation 1: Python content, recent
    factory.create_conversation(
        id="conv1",
        provider="chatgpt",
        title="Python Error Handling",
        messages=[
            {"id": "m1", "role": "user", "text": "How to handle exceptions in Python?"},
            {"id": "m2", "role": "assistant", "text": "Use try-except blocks for Python exception handling."},
        ],
        created_at=datetime.now() - timedelta(days=1),
        updated_at=datetime.now() - timedelta(days=1),
    )

    # Conversation 2: JavaScript content, older
    factory.create_conversation(
        id="conv2",
        provider="claude",
        title="JavaScript Async Patterns",
        messages=[
            {"id": "m3", "role": "user", "text": "Explain async/await in JavaScript"},
            {"id": "m4", "role": "assistant", "text": "Async/await is JavaScript syntax for promises."},
        ],
        created_at=datetime.now() - timedelta(days=10),
        updated_at=datetime.now() - timedelta(days=10),
    )

    # Conversation 3: Rust content
    factory.create_conversation(
        id="conv3",
        provider="claude-code",
        title="Rust Ownership",
        messages=[
            {"id": "m5", "role": "user", "text": "What is ownership in Rust?"},
            {
                "id": "m6",
                "role": "assistant",
                "text": "Rust ownership ensures memory safety without garbage collection.",
            },
        ],
        created_at=datetime.now() - timedelta(hours=6),
        updated_at=datetime.now() - timedelta(hours=6),
    )

    # Build FTS index using rebuild_index

    rebuild_index()

    return cli_workspace
