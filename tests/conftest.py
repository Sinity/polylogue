from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _clear_polylogue_env(monkeypatch):
    # Reset the container singleton to ensure clean state between tests
    from polylogue.cli.container import reset_container
    reset_container()

    # Clear thread-local database state to prevent connection leaks between tests
    # Import db module fresh each time to handle module reloads from other tests
    import sys
    db_module = sys.modules.get("polylogue.storage.db")
    if db_module is not None and hasattr(db_module, "_LOCAL"):
        state = getattr(db_module._LOCAL, "state", None)
        if state is not None:
            conn = state.get("conn")
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
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
        # Prevent tests from hitting external Qdrant
        "QDRANT_URL",
        "QDRANT_API_KEY",
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

    import polylogue.paths
    import polylogue.config
    import polylogue.container
    import polylogue.cli.container
    import polylogue.storage.backends.sqlite
    import polylogue.storage.db
    import polylogue.storage.search
    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)  # Depends on paths
    importlib.reload(polylogue.container)  # Depends on config
    importlib.reload(polylogue.cli.container)  # Depends on container
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.db)
    importlib.reload(polylogue.storage.search)  # Picks up new DatabaseError class

    # Reset container singleton to use fresh config
    from polylogue.cli.container import reset_container
    reset_container()

    return {
        "config_path": config_dir / "config.json",
        "archive_root": archive_root,
        "data_root": data_dir,
        "state_root": state_dir,  # Kept for backward compatibility
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
def storage_repository(workspace_env):
    """Storage repository with its own write lock.

    Use this fixture in tests that need thread-safe storage operations.
    The repository encapsulates the write lock and provides methods for
    saving conversations and recording runs.

    Depends on workspace_env to ensure XDG_DATA_HOME is set before
    creating the default backend.
    """
    from polylogue.storage.backends.sqlite import create_default_backend
    from polylogue.storage.repository import StorageRepository

    backend = create_default_backend()
    return StorageRepository(backend=backend)


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

    import polylogue.paths
    import polylogue.config
    import polylogue.container
    import polylogue.cli.container
    import polylogue.storage.backends.sqlite
    import polylogue.storage.db
    import polylogue.storage.search
    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)  # Depends on paths
    importlib.reload(polylogue.container)  # Depends on config
    importlib.reload(polylogue.cli.container)  # Depends on container
    importlib.reload(polylogue.storage.backends.sqlite)
    importlib.reload(polylogue.storage.db)
    importlib.reload(polylogue.storage.search)  # Picks up new DatabaseError class

    # Reset container singleton to use fresh config
    from polylogue.cli.container import reset_container
    reset_container()

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
        json.dumps({
            "installed": {
                "client_id": "mock_client_id.apps.googleusercontent.com",
                "client_secret": "mock_client_secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        })
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
    original_import_module = None

    def mock_import_module(name: str):
        if name == "googleapiclient.http":
            import types
            mock_http = types.ModuleType("googleapiclient.http")
            mock_http.MediaIoBaseDownload = MockMediaIoBaseDownload
            return mock_http
        # Fall through to original for other modules
        import importlib
        return importlib.import_module(name)

    import polylogue.ingestion.drive_client as drive_client
    original_import_module = drive_client._import_module

    monkeypatch.setattr(drive_client, "_import_module", mock_import_module)

    return {"MockMediaIoBaseDownload": MockMediaIoBaseDownload}
