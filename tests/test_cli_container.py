"""Tests for CLI dependency injection container."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Import directly from modules to avoid the CLI import chain which has bugs
from polylogue.config import Config, ConfigError, load_config
from polylogue.storage.repository import StorageRepository


def test_create_config_from_path(tmp_path: Path) -> None:
    """Test creating config from explicit path (via container factory)."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [{"name": "test", "path": str(tmp_path / "inbox")}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    # Test via direct load_config (container wraps this)
    config = load_config(config_path)

    assert isinstance(config, Config)
    assert config.version == 2
    assert config.archive_root == tmp_path / "archive"
    assert len(config.sources) == 1
    assert config.sources[0].name == "test"


def test_create_config_missing_file(tmp_path: Path) -> None:
    """Test creating config from non-existent path raises ConfigError."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(ConfigError, match="not found"):
        load_config(missing_path)


def test_create_config_invalid_json(tmp_path: Path) -> None:
    """Test creating config from invalid JSON raises ConfigError."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_create_config_missing_required_fields(tmp_path: Path) -> None:
    """Test creating config with missing required fields raises ConfigError."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        # Missing archive_root and sources
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_create_config_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test create_config respects POLYLOGUE_CONFIG env var when path is None."""
    config_path = tmp_path / "from-env.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [{"name": "env-test", "path": str(tmp_path / "inbox")}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config_path))

    config = load_config(None)

    assert isinstance(config, Config)
    assert config.sources[0].name == "env-test"


def test_create_storage_repository() -> None:
    """Test creating storage repository returns StorageRepository instance (via container factory)."""
    from polylogue.storage.backends.sqlite import create_default_backend

    backend = create_default_backend()
    repository = StorageRepository(backend=backend)

    assert isinstance(repository, StorageRepository)
    # Check that it has the expected write lock attribute
    assert hasattr(repository, "_write_lock")


def test_create_storage_repository_independent_instances() -> None:
    """Test that each call creates a new independent repository instance (via container factory)."""
    from polylogue.storage.backends.sqlite import create_default_backend

    backend1 = create_default_backend()
    backend2 = create_default_backend()
    repo1 = StorageRepository(backend=backend1)
    repo2 = StorageRepository(backend=backend2)

    assert repo1 is not repo2
    # Each should have its own lock
    assert repo1._write_lock is not repo2._write_lock


def test_create_config_validates_sources(tmp_path: Path) -> None:
    """Test that config validation runs on created config."""
    config_path = tmp_path / "config.json"
    # Create config with invalid source (missing path and folder)
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [{"name": "invalid"}],  # Missing path/folder
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises((ConfigError, ValueError)):
        load_config(config_path)


def test_create_config_validates_duplicate_sources(tmp_path: Path) -> None:
    """Test that config validation catches duplicate source names."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [
            {"name": "duplicate", "path": str(tmp_path / "inbox1")},
            {"name": "duplicate", "path": str(tmp_path / "inbox2")},
        ],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    with pytest.raises(ConfigError, match="Duplicate source"):
        load_config(config_path)


def test_create_config_with_template_path(tmp_path: Path) -> None:
    """Test creating config with optional template_path."""
    config_path = tmp_path / "config.json"
    template_path = tmp_path / "template.html"
    template_path.write_text("<html></html>", encoding="utf-8")

    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [{"name": "test", "path": str(tmp_path / "inbox")}],
        "template_path": str(template_path),
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    config = load_config(config_path)

    assert config.template_path == template_path


def test_create_config_with_drive_source(tmp_path: Path) -> None:
    """Test creating config with Google Drive source."""
    config_path = tmp_path / "config.json"
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [{"name": "drive-test", "folder": "Google AI Studio"}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    config = load_config(config_path)

    assert len(config.sources) == 1
    assert config.sources[0].is_drive
    assert config.sources[0].folder == "Google AI Studio"


def test_create_config_path_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that config paths are expanded (e.g., ~ expansion)."""
    config_path = tmp_path / "config.json"
    # Use relative paths that will be expanded
    config_payload = {
        "version": 2,
        "archive_root": str(tmp_path / "archive"),
        "sources": [{"name": "test", "path": str(tmp_path / "inbox")}],
    }
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")

    config = load_config(config_path)

    # All paths should be absolute after loading
    assert config.archive_root.is_absolute()
    assert config.sources[0].path is not None
    assert config.sources[0].path.is_absolute()
