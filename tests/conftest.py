from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

import pytest

from polylogue.lib.messages import MessageCollection


@pytest.fixture(autouse=True)
def _clear_polylogue_env(monkeypatch):
    # Reset the container singleton to ensure clean state between tests
    from polylogue.services import reset

    reset()

    # Clear thread-local database state to prevent connection leaks between tests
    # Import db module fresh each time to handle module reloads from other tests
    import sys

    db_module = sys.modules.get("polylogue.storage.backends.sqlite")
    if db_module is not None and hasattr(db_module, "_LOCAL"):
        state = getattr(db_module._LOCAL, "state", None)
        if state is not None:
            conn = state.get("conn")
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
            state["conn"] = None
            state["path"] = None
            state["depth"] = 0

    # Clear search cache (LRU cache) to prevent cross-test pollution
    search_module = sys.modules.get("polylogue.storage.search")
    if search_module is not None and hasattr(search_module, "_search_messages_cached"):
        search_module._search_messages_cached.cache_clear()

    for key in (
        "POLYLOGUE_CONFIG",
        "POLYLOGUE_ARCHIVE_ROOT",
        "POLYLOGUE_RENDER_ROOT",
        "POLYLOGUE_TEMPLATE_PATH",
        "POLYLOGUE_DECLARATIVE",
        # Prevent tests from hitting external Voyage API
        "VOYAGE_API_KEY",
        "POLYLOGUE_VOYAGE_API_KEY",
        # Clear Drive credentials to ensure test isolation
        "POLYLOGUE_CREDENTIAL_PATH",
        "POLYLOGUE_TOKEN_PATH",
        # Don't inherit real user's XDG directories
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def workspace_env(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_dir / "config.json"))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(archive_root / "render"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    # Reload paths module and dependent modules to pick up new XDG_DATA_HOME
    # Order matters: paths first, then modules that import from paths
    import importlib

    import polylogue.config
    import polylogue.paths
    import polylogue.services
    import polylogue.storage.backends.sqlite
    import polylogue.storage.search

    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)  # Depends on paths
    importlib.reload(polylogue.services)  # Depends on config
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.search)  # Picks up new DatabaseError class

    # Reset services singleton to use fresh config
    from polylogue.services import reset

    reset()

    return {
        "config_path": config_dir / "config.json",
        "archive_root": archive_root,
        "data_root": data_dir,
        "state_root": state_dir,  # Kept for backward compatibility
    }


@pytest.fixture
def db_without_fts(tmp_path):
    """Database with schema but WITHOUT the FTS table (simulates fresh install)."""
    from polylogue.storage.backends.sqlite import open_connection

    db_path = tmp_path / "no_fts.db"
    with open_connection(db_path) as conn:
        # Drop the FTS table and all triggers to simulate fresh install state
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_insert")
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_update")
        conn.execute("DROP TRIGGER IF EXISTS messages_fts_delete")
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
def storage_repository(workspace_env):
    """Storage repository with its own write lock.

    Use this fixture in tests that need thread-safe storage operations.
    The repository encapsulates the write lock and provides methods for
    saving conversations and recording runs.

    Depends on workspace_env to ensure XDG_DATA_HOME is set before
    creating the default backend.
    """
    from polylogue.storage.backends.sqlite import create_default_backend
    from polylogue.storage.repository import ConversationRepository

    backend = create_default_backend()
    return ConversationRepository(backend=backend)


@pytest.fixture
def cli_workspace(tmp_path, monkeypatch):
    """
    Isolated CLI workspace with config, archive, and database.

    Creates a complete test environment for CLI command testing:
    - Config directory with config.json
    - Archive root directory
    - Data directory for database (XDG_DATA_HOME)
    - Inbox directory for test data
    - Pre-configured environment variables

    Returns:
        dict with paths: config_path, archive_root, data_root, inbox_dir, db_path
    """
    import json

    # Create directory structure
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    archive_root = tmp_path / "archive"
    inbox_dir = tmp_path / "inbox"
    render_root = archive_root / "render"

    for path in [config_dir, data_dir, state_dir, archive_root, inbox_dir, render_root]:
        path.mkdir(parents=True, exist_ok=True)

    # Create minimal config.json
    config_path = config_dir / "config.json"
    config = {
        "version": 2,
        "archive_root": str(archive_root),
        "sources": [{"name": "test-inbox", "path": str(inbox_dir)}],
    }
    config_path.write_text(json.dumps(config, indent=2))

    # Set environment variables
    # default_db_path() returns: DATA_HOME / "polylogue" / "polylogue.db"
    db_path = data_dir / "polylogue" / "polylogue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_RENDER_ROOT", str(render_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")  # Plain output for tests
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    # Reload paths module and dependent modules to pick up new XDG_DATA_HOME
    # Order matters: paths first, then modules that import from paths
    import importlib

    import polylogue.config
    import polylogue.paths
    import polylogue.services
    import polylogue.storage.backends.sqlite
    import polylogue.storage.search

    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)  # Depends on paths
    importlib.reload(polylogue.services)  # Depends on config
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.search)  # Picks up new DatabaseError class

    # Ensure schema exists before tests run
    with polylogue.storage.backends.sqlite.connection_context(db_path) as conn:
        polylogue.storage.backends.sqlite._ensure_schema(conn)

    from polylogue.services import reset

    reset()

    return {
        "config_path": config_path,
        "archive_root": archive_root,
        "data_root": data_dir,
        "state_root": state_dir,  # Kept for backward compatibility
        "inbox_dir": inbox_dir,
        "render_root": render_root,
        "db_path": db_path,
    }


@pytest.fixture
def mock_drive_credentials(tmp_path, monkeypatch):
    """
    Mock Google Drive OAuth credentials for testing.

    Creates mock credentials.json and token.json files and sets up
    environment variables to point to them.

    Returns:
        dict with paths: credentials_path, token_path, and MockCredentials instance
    """
    from tests.mocks.drive_mocks import MockCredentials

    creds_dir = tmp_path / "drive_creds"
    creds_dir.mkdir(parents=True, exist_ok=True)

    # Create mock credentials.json (OAuth client config)
    creds_path = creds_dir / "credentials.json"
    creds_path.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "mock_client_id.apps.googleusercontent.com",
                    "client_secret": "mock_client_secret",
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            }
        )
    )

    # Create mock token.json (OAuth access/refresh tokens)
    token_path = creds_dir / "token.json"
    mock_creds = MockCredentials()
    token_path.write_text(mock_creds.to_json())

    # Set environment variables
    monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))
    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

    return {
        "credentials_path": creds_path,
        "token_path": token_path,
        "mock_credentials": mock_creds,
    }


@pytest.fixture
def mock_drive_service(monkeypatch):
    """
    Mock Google Drive service for testing.

    Patches google.auth and googleapiclient.discovery to return mock objects.

    Returns:
        dict with: service (MockDriveService), files (dict), file_content (dict)
    """
    from tests.mocks.drive_mocks import MockCredentials, MockDriveService, mock_drive_file

    # Sample file data
    files_data = {
        "folder1": mock_drive_file(
            file_id="folder1",
            name="Google AI Studio",
            mime_type="application/vnd.google-apps.folder",
        ),
        "prompt1": mock_drive_file(
            file_id="prompt1",
            name="Test Prompt",
            mime_type="application/vnd.google-makersuite.prompt",
            parents=["folder1"],
        ),
    }

    file_content = {
        "prompt1": b'{"title": "Test Prompt", "content": "Test content"}',
    }

    mock_service = MockDriveService(files_data=files_data, file_content=file_content)
    mock_creds = MockCredentials()

    # Patch google.auth.default
    def mock_google_auth_default(*args, **kwargs):
        return mock_creds, None

    # Patch googleapiclient.discovery.build
    def mock_discovery_build(service_name, version, credentials=None, *args, **kwargs):
        return mock_service

    # Patch google.auth.transport.requests.Request
    class MockRequest:
        pass

    monkeypatch.setattr("google.auth.default", mock_google_auth_default, raising=False)
    monkeypatch.setattr("googleapiclient.discovery.build", mock_discovery_build, raising=False)
    monkeypatch.setattr("google.auth.transport.requests.Request", MockRequest, raising=False)

    return {
        "service": mock_service,
        "files": files_data,
        "file_content": file_content,
        "credentials": mock_creds,
    }


@pytest.fixture
def mock_media_downloader(monkeypatch):
    """
    Patch GoogleAPI's MediaIoBaseDownload with our mock.

    This fixture patches the lazy import mechanism in drive_client.py
    to return our MockMediaIoBaseDownload when MediaIoBaseDownload is requested.
    """
    from tests.mocks.drive_mocks import MockMediaIoBaseDownload

    # Patch the _import_module function to return mock for MediaIoBaseDownload

    def mock_import_module(name: str):
        if name == "googleapiclient.http":
            import types

            mock_http = types.ModuleType("googleapiclient.http")
            mock_http.MediaIoBaseDownload = MockMediaIoBaseDownload
            return mock_http
        # Fall through to original for other modules
        import importlib

        return importlib.import_module(name)

    import polylogue.sources.drive_client as drive_client


    monkeypatch.setattr(drive_client, "_import_module", mock_import_module)

    return {"MockMediaIoBaseDownload": MockMediaIoBaseDownload}


# =============================================================================
# NEW FIXTURES FOR TEST CONSOLIDATION (added during aggressive parametrization)
# =============================================================================


@pytest.fixture
def db_path(workspace_env):
    """Shortcut fixture for database path setup.

    Usage in tests:
        def test_something(db_path):
            builder = ConversationBuilder(db_path, "test-conv")
    """
    from tests.helpers import db_setup

    return db_setup(workspace_env)


@pytest.fixture
def conversation_builder(db_path):
    """Fixture that provides ConversationBuilder factory.

    Usage in tests:
        def test_something(conversation_builder):
            conv = (conversation_builder("test-conv")
                   .add_message("m1", text="Hello")
                   .save())
    """
    from tests.helpers import ConversationBuilder

    def _builder(conversation_id: str = "test-conv"):
        return ConversationBuilder(db_path, conversation_id)

    return _builder


# =============================================================================
# CONSOLIDATED FIXTURES (moved from individual test files)
# =============================================================================


@pytest.fixture
def test_db(tmp_path):
    """Create an isolated test database with schema initialized.

    Replaces duplicate fixtures in: test_store.py, test_pipeline.py, test_lib.py,
    test_repository_render.py, test_search_index.py
    """
    from polylogue.storage.backends.sqlite import open_connection

    db_path = tmp_path / "test.db"
    with open_connection(db_path):
        pass  # Schema auto-initializes
    return db_path


@pytest.fixture
def test_conn(test_db):
    """Provide a connection to the test database.

    Replaces duplicate fixtures in: test_store.py, test_pipeline.py, test_search_index.py
    """
    from polylogue.storage.backends.sqlite import open_connection

    with open_connection(test_db) as conn:
        yield conn


@pytest.fixture
def sqlite_backend(tmp_path):
    """Create a SQLite backend for testing.

    Replaces duplicate fixtures in: test_backend_sqlite.py, test_repository_backend.py
    """

    from polylogue.storage.backends import SQLiteBackend

    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(db_path=db_path)
    yield backend
    backend.close()


@pytest.fixture
def sample_conversation():
    """Create a diverse conversation for filter/projection testing.

    Includes:
    - User messages (m1, m5)
    - Assistant messages (m2, m6 short, m7 substantive)
    - System message (m3)
    - Tool message (m4)
    - Mix of short/noise and substantive messages

    Replaces duplicate fixtures in: test_projections.py
    """
    from datetime import datetime

    from polylogue.lib.models import Conversation, Message

    messages = [
        Message(id="m1", role="user", text="User question", timestamp=datetime(2024, 1, 1, 10, 0)),
        Message(id="m2", role="assistant", text="Assistant response", timestamp=datetime(2024, 1, 1, 10, 1)),
        Message(id="m3", role="system", text="System prompt", timestamp=datetime(2024, 1, 1, 10, 2)),
        Message(id="m4", role="tool", text="Tool output", timestamp=datetime(2024, 1, 1, 10, 3)),
        Message(
            id="m5", role="user", text="Another question with searchterm here", timestamp=datetime(2024, 1, 1, 10, 4)
        ),
        Message(id="m6", role="assistant", text="ok", timestamp=datetime(2024, 1, 1, 10, 5)),  # Short (noise)
        Message(
            id="m7", role="assistant", text="Substantial answer with details", timestamp=datetime(2024, 1, 1, 10, 6)
        ),
    ]
    return Conversation(id="conv1", provider="test", messages=MessageCollection(messages=messages))


@pytest.fixture
def cli_runner():
    """Create a CliRunner for testing CLI commands.

    Replaces duplicate definitions across CLI test files.
    """
    from click.testing import CliRunner

    return CliRunner()


# =============================================================================
# SEEDED DATABASE FIXTURE (for tests that need real provider data)
# =============================================================================


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory):
    """Create a database seeded with real fixture data from all providers.

    This fixture ingests actual provider data (ChatGPT, Claude Code, Codex, Gemini)
    from tests/fixtures/real/ through the full pipeline. Use this instead of
    hardcoding production database paths.

    Scope is 'session' for efficiency - the seeded DB is created once and reused.

    Returns:
        Path to the seeded database file
    """
    from pathlib import Path

    from polylogue.config import Source
    from polylogue.pipeline.ingest import prepare_ingest
    from polylogue.sources import iter_source_conversations
    from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
    from polylogue.storage.repository import ConversationRepository

    # Create session-scoped temp directory
    tmp_dir = tmp_path_factory.mktemp("seeded_db")
    db_path = tmp_dir / "polylogue.db"

    # Initialize schema by opening connection
    with open_connection(db_path):
        pass

    # Create a repository that uses our temp database
    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)

    # Find fixtures directory
    fixtures_dir = Path(__file__).parent / "fixtures" / "real"

    # Provider -> fixture paths mapping
    fixture_files = {
        "chatgpt": list(fixtures_dir.glob("chatgpt/*.json")),
        "claude-code": list(fixtures_dir.glob("claude-code/*.jsonl")),
        "codex": list(fixtures_dir.glob("codex/*.jsonl")),
        "gemini": list(fixtures_dir.glob("gemini/*.jsonl")),
    }

    # Ingest each fixture through the full pipeline
    with open_connection(db_path) as conn:
        for provider, files in fixture_files.items():
            for fixture_path in files:
                if not fixture_path.exists():
                    continue

                try:
                    source = Source(name=provider, path=fixture_path)
                    for convo in iter_source_conversations(source):
                        archive_root = tmp_dir / "archive"
                        archive_root.mkdir(exist_ok=True)
                        prepare_ingest(
                            convo,
                            source_name=provider,
                            archive_root=archive_root,
                            conn=conn,
                            repository=repository,  # Pass our repository!
                        )
                except Exception as e:
                    # Log but don't fail - some fixtures may have format issues
                    import warnings

                    warnings.warn(f"Failed to ingest {fixture_path}: {e}", stacklevel=2)

    return db_path


@pytest.fixture
def seeded_repository(seeded_db):
    """Repository backed by the seeded database.

    Use this when tests need a ConversationRepository with real provider data.
    """
    from polylogue.storage.backends.sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

    backend = SQLiteBackend(db_path=seeded_db)
    return ConversationRepository(backend=backend)


# =============================================================================
# RAW CONVERSATION FIXTURES (for database-driven testing)
# =============================================================================


# Pre-compute the REAL database path at module load time, before any test
# fixtures can modify environment variables. This ensures database-driven
# tests always use the user's actual database, not a temp test database.
_REAL_DB_PATH = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "polylogue" / "polylogue.db"


@pytest.fixture(scope="session")
def raw_db_samples():
    """Load samples from user's actual raw_conversations table.

    This enables honest, database-driven testing by using REAL data
    that was acquired via `polylogue run --stage acquire`.

    Control sample count via POLYLOGUE_TEST_SAMPLES environment variable:
    - POLYLOGUE_TEST_SAMPLES=100 (default) - Fast CI, limited samples
    - POLYLOGUE_TEST_SAMPLES=0 - Exhaustive mode, ALL raw conversations

    Returns:
        List of RawConversationRecord objects from the user's database
    """
    import os

    from polylogue.storage.backends.sqlite import SQLiteBackend

    # Get sample limit from environment (use original env, not test-modified)
    limit_str = os.environ.get("POLYLOGUE_TEST_SAMPLES", "100")
    limit = int(limit_str) if limit_str != "0" else None

    # Use pre-computed path that bypasses test env modifications
    if not _REAL_DB_PATH.exists():
        return []  # No database = no samples (skip in tests)

    backend = SQLiteBackend(db_path=_REAL_DB_PATH)

    # Load samples from raw_conversations table
    samples = list(backend.iter_raw_conversations(limit=limit))

    return samples


@pytest.fixture
def raw_backend(tmp_path):
    """SQLite backend for raw conversation testing.

    Use this for tests that need to save/read raw conversation records.
    """
    from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection

    db_path = tmp_path / "raw.db"
    with open_connection(db_path):
        pass

    return SQLiteBackend(db_path=db_path)
