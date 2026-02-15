from __future__ import annotations

import json
import os

import pytest
from hypothesis import HealthCheck, settings

from polylogue.lib.messages import MessageCollection

# ---------------------------------------------------------------------------
# Hypothesis profiles: `--hypothesis-profile ci` uses fewer examples for speed
# ---------------------------------------------------------------------------
settings.register_profile("ci", max_examples=30, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("default", max_examples=100)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def _clear_polylogue_env(monkeypatch):
    # Reset the container singleton to ensure clean state between tests
    from polylogue import services

    services.reset()

    # Also clear async singletons (without awaiting close — test isolation
    # is more important than graceful connection shutdown here)
    services._async_backend = None
    services._async_repository = None

    # Close any cached SQLite connections to prevent WAL sidecar corruption
    # when tests create/move/delete temp database files.
    from polylogue.storage.backends.connection import _clear_connection_cache

    _clear_connection_cache()

    # Clear search cache (LRU cache) to prevent cross-test pollution
    import sys

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

    # No importlib.reload() needed — paths.py uses lazy evaluation (functions, not constants).
    # Just reset the service singletons so they pick up fresh env vars.
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

    # No importlib.reload() needed — paths.py uses lazy evaluation (functions, not constants).
    # Just reset the service singletons so they pick up fresh env vars.
    from polylogue.storage.backends.sqlite import _ensure_schema, connection_context

    with connection_context(db_path) as conn:
        _ensure_schema(conn)

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
    from tests.infra.drive_mocks import MockCredentials

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
    from tests.infra.drive_mocks import MockCredentials, MockDriveService, mock_drive_file

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
    from tests.infra.drive_mocks import MockMediaIoBaseDownload

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
    from tests.infra.helpers import db_setup

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
    from tests.infra.helpers import ConversationBuilder

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
    """Create a database seeded with synthetic data from all providers.

    Uses SyntheticCorpus to generate schema-driven test data for each provider,
    then ingests through the full pipeline. No fixture files required.

    Scope is 'session' for efficiency - the seeded DB is created once and reused.

    Returns:
        Path to the seeded database file
    """
    import hashlib
    from datetime import datetime, timezone

    from polylogue.paths import Source
    from polylogue.pipeline.prepare import prepare_records
    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.sources import iter_source_conversations
    from polylogue.storage.backends.sqlite import SQLiteBackend, open_connection
    from polylogue.storage.repository import ConversationRepository
    from polylogue.storage.store import RawConversationRecord

    # Create session-scoped temp directory
    tmp_dir = tmp_path_factory.mktemp("seeded_db")
    db_path = tmp_dir / "polylogue.db"

    # Initialize schema
    with open_connection(db_path):
        pass

    backend = SQLiteBackend(db_path=db_path)
    repository = ConversationRepository(backend=backend)

    # File extension per provider encoding
    ext_map = {"chatgpt": ".json", "claude-ai": ".json", "gemini": ".json",
               "claude-code": ".jsonl", "codex": ".jsonl"}

    # Generate synthetic data and write to temp files
    corpus_dir = tmp_dir / "corpus"
    corpus_dir.mkdir()

    for provider in SyntheticCorpus.available_providers():
        corpus = SyntheticCorpus.for_provider(provider)
        raw_items = corpus.generate(count=3, messages_per_conversation=range(4, 12), seed=42)
        provider_dir = corpus_dir / provider
        provider_dir.mkdir()

        for idx, raw_bytes in enumerate(raw_items):
            ext = ext_map.get(provider, ".json")
            file_path = provider_dir / f"synthetic-{idx:02d}{ext}"
            file_path.write_bytes(raw_bytes)

            # Step 1: Store as raw conversation (acquire stage)
            try:
                raw_id = hashlib.sha256(raw_bytes).hexdigest()
                record = RawConversationRecord(
                    raw_id=raw_id,
                    provider_name=provider,
                    source_name=provider,
                    source_path=str(file_path),
                    raw_content=raw_bytes,
                    acquired_at=datetime.now(timezone.utc).isoformat(),
                )
                backend.save_raw_conversation(record)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to store raw {provider}/{idx}: {e}", stacklevel=2)

    backend._get_connection().commit()

    # Step 2: Parse and ingest (parse stage)
    archive_root = tmp_dir / "archive"
    archive_root.mkdir()

    with open_connection(db_path) as conn:
        for provider_dir in sorted(corpus_dir.iterdir()):
            provider = provider_dir.name
            for file_path in sorted(provider_dir.iterdir()):
                try:
                    source = Source(name=provider, path=file_path)
                    raw_id = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    for convo in iter_source_conversations(source):
                        prepare_records(
                            convo,
                            source_name=provider,
                            archive_root=archive_root,
                            conn=conn,
                            repository=repository,
                            raw_id=raw_id,
                        )
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to ingest {file_path.name}: {e}", stacklevel=2)

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
# SYNTHETIC RAW SAMPLES (replaces the old raw_db_samples fixture)
# =============================================================================


@pytest.fixture(scope="session")
def raw_synthetic_samples():
    """Generate synthetic raw conversation records for all providers.

    Uses SyntheticCorpus to generate wire-format bytes, wrapped in
    RawConversationRecord objects. This replaces the old raw_db_samples
    fixture that required a real database with imported data.

    Returns:
        List of RawConversationRecord objects (synthetic data, always available)
    """
    import hashlib
    from datetime import datetime, timezone

    from polylogue.schemas.synthetic import SyntheticCorpus
    from polylogue.storage.store import RawConversationRecord

    samples: list[RawConversationRecord] = []
    for provider in SyntheticCorpus.available_providers():
        corpus = SyntheticCorpus.for_provider(provider)
        for idx, raw_bytes in enumerate(corpus.generate(count=5, seed=42)):
            raw_id = hashlib.sha256(raw_bytes).hexdigest()
            samples.append(RawConversationRecord(
                raw_id=raw_id,
                provider_name=provider,
                source_name=provider,
                source_path=f"<synthetic:{provider}:{idx}>",
                raw_content=raw_bytes,
                acquired_at=datetime.now(timezone.utc).isoformat(),
            ))
    return samples


# =============================================================================
# SYNTHETIC SOURCE FIXTURE (replaces FIXTURES_DIR-based sources)
# =============================================================================


@pytest.fixture
def synthetic_source(tmp_path):
    """Factory fixture that generates synthetic Source objects for any provider.

    Writes SyntheticCorpus output to temp files, returning Source objects that
    can be fed through the full pipeline — just like the old fixture-based
    sources, but always available and schema-driven.

    Usage::

        def test_something(synthetic_source):
            source = synthetic_source("chatgpt")
            for convo in iter_source_conversations(source):
                ...

            # Multiple files:
            source = synthetic_source("claude-code", count=3)
    """
    from polylogue.paths import Source
    from polylogue.schemas.synthetic import SyntheticCorpus

    def _factory(
        provider: str,
        count: int = 1,
        messages_per_conversation: range = range(4, 12),
        seed: int = 42,
    ) -> Source:
        corpus = SyntheticCorpus.for_provider(provider)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        raw_items = corpus.generate(
            count=count,
            messages_per_conversation=messages_per_conversation,
            seed=seed,
        )

        provider_dir = tmp_path / "synthetic" / provider
        provider_dir.mkdir(parents=True, exist_ok=True)

        for idx, raw_bytes in enumerate(raw_items):
            (provider_dir / f"synth-{idx:02d}{ext}").write_bytes(raw_bytes)

        if count == 1:
            return Source(name=f"{provider}-test", path=provider_dir / f"synth-00{ext}")
        # For multiple files, return Source pointing to first file
        # (pipeline processes individual files, not directories)
        return Source(name=f"{provider}-test", path=provider_dir / f"synth-00{ext}")

    return _factory


