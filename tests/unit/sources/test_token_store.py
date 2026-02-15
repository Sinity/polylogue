"""Tests for OAuth token storage backends."""

from __future__ import annotations

from pathlib import Path

from polylogue.sources.token_store import (
    FileTokenStore,
    KeyringTokenStore,
    create_token_store,
)


class TestFileTokenStore:
    """Tests for file-based token storage."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Save and load a token."""
        store = FileTokenStore(tmp_path)
        store.save("test_key", '{"token": "value"}')

        loaded = store.load("test_key")
        assert loaded == '{"token": "value"}'

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Loading a non-existent token returns None."""
        store = FileTokenStore(tmp_path)
        assert store.load("nonexistent") is None

    def test_delete(self, tmp_path: Path) -> None:
        """Delete a token."""
        store = FileTokenStore(tmp_path)
        store.save("test_key", '{"token": "value"}')
        store.delete("test_key")

        assert store.load("test_key") is None

    def test_delete_nonexistent_is_safe(self, tmp_path: Path) -> None:
        """Deleting a non-existent token is safe."""
        store = FileTokenStore(tmp_path)
        store.delete("nonexistent")  # Should not raise

    def test_sanitizes_key(self, tmp_path: Path) -> None:
        """Keys are sanitized to prevent directory traversal."""
        store = FileTokenStore(tmp_path)
        store.save("../../evil", "data")

        # Should be stored safely in tmp_path
        assert not (tmp_path.parent.parent / "evil.json").exists()
        assert store.load("../../evil") == "data"

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        """Directory is created if it doesn't exist."""
        subdir = tmp_path / "nested" / "dir"
        store = FileTokenStore(subdir)
        store.save("key", "value")

        assert (subdir / "key.json").exists()

    def test_file_permissions(self, tmp_path: Path) -> None:
        """Token files have 0o600 permissions."""
        store = FileTokenStore(tmp_path)
        store.save("secure", '{"secret": "data"}')

        token_file = tmp_path / "secure.json"
        assert token_file.exists()
        # 0o600 = owner read/write only
        assert (token_file.stat().st_mode & 0o777) == 0o600


class TestKeyringTokenStore:
    """Tests for keyring-based token storage with file fallback."""

    def test_fallback_to_file_when_keyring_unavailable(self, tmp_path: Path) -> None:
        """Falls back to file storage when keyring is unavailable."""
        file_store = FileTokenStore(tmp_path)
        keyring_store = KeyringTokenStore(fallback=file_store)

        # If keyring is unavailable (which is likely in tests), should use file fallback
        keyring_store.save("test", "data")
        assert keyring_store.load("test") == "data"

        # Verify it was actually stored in file
        assert file_store.load("test") == "data"

    def test_delete_with_fallback(self, tmp_path: Path) -> None:
        """Delete works with file fallback."""
        file_store = FileTokenStore(tmp_path)
        keyring_store = KeyringTokenStore(fallback=file_store)

        keyring_store.save("key", "value")
        keyring_store.delete("key")

        assert keyring_store.load("key") is None


class TestCreateTokenStore:
    """Tests for token store factory function."""

    def test_returns_token_store(self, tmp_path: Path) -> None:
        """create_token_store returns a TokenStore."""
        store = create_token_store(tmp_path)
        assert store is not None

        # Should have the protocol methods
        assert hasattr(store, "load")
        assert hasattr(store, "save")
        assert hasattr(store, "delete")

    def test_store_is_functional(self, tmp_path: Path) -> None:
        """Created store is functional."""
        store = create_token_store(tmp_path)
        store.save("test", "data")
        assert store.load("test") == "data"
