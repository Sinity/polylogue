"""Token storage backends for OAuth credentials.

Provides a protocol for token persistence with two implementations:
- FileTokenStore: Plaintext JSON with file permissions (fallback)
- KeyringTokenStore: System keyring integration via the `keyring` library
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Protocol

from polylogue.lib.log import get_logger

LOGGER = get_logger(__name__)


class TokenStore(Protocol):
    """Protocol for OAuth token persistence."""

    def load(self, key: str) -> str | None:
        """Load token data by key. Returns None if not found."""
        ...

    def save(self, key: str, data: str) -> None:
        """Save token data by key."""
        ...

    def delete(self, key: str) -> None:
        """Delete token data by key."""
        ...


class FileTokenStore:
    """File-based token storage with 0o600 permissions.

    This is the fallback when keyring is not available.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def _path_for_key(self, key: str) -> Path:
        # Sanitize key to prevent directory traversal
        safe_key = key.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._directory / f"{safe_key}.json"

    def load(self, key: str) -> str | None:
        path = self._path_for_key(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def save(self, key: str, data: str) -> None:
        path = self._path_for_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        path.chmod(0o600)

    def delete(self, key: str) -> None:
        path = self._path_for_key(key)
        if path.exists():
            path.unlink()


class KeyringTokenStore:
    """Keyring-based token storage using the system credential store.

    Falls back to FileTokenStore if keyring is not available or fails.
    """

    _SERVICE_NAME = "polylogue-oauth"

    def __init__(self, fallback: FileTokenStore) -> None:
        self._fallback = fallback
        self._keyring = self._try_import_keyring()

    @staticmethod
    def _try_import_keyring() -> object | None:
        try:
            import keyring  # type: ignore[import]
            # Test that keyring backend is functional
            keyring.get_keyring()
            return keyring  # type: ignore[no-any-return]
        except Exception:
            return None

    def load(self, key: str) -> str | None:
        if self._keyring is not None:
            try:
                data: str | None = self._keyring.get_password(self._SERVICE_NAME, key)  # type: ignore[attr-defined]
                if data is not None:
                    return data
            except Exception as exc:
                LOGGER.debug("Keyring load failed for %s, falling back to file: %s", key, exc)
        return self._fallback.load(key)

    def save(self, key: str, data: str) -> None:
        if self._keyring is not None:
            try:
                self._keyring.set_password(self._SERVICE_NAME, key, data)  # type: ignore[attr-defined]
                return
            except Exception as exc:
                LOGGER.debug("Keyring save failed for %s, falling back to file: %s", key, exc)
        self._fallback.save(key, data)

    def delete(self, key: str) -> None:
        if self._keyring is not None:
            with suppress(Exception):
                self._keyring.delete_password(self._SERVICE_NAME, key)  # type: ignore[attr-defined]
        self._fallback.delete(key)


def create_token_store(directory: Path) -> TokenStore:
    """Create the best available token store.

    Tries keyring first, falls back to file-based storage.
    """
    file_store = FileTokenStore(directory)
    keyring_store = KeyringTokenStore(fallback=file_store)
    if keyring_store._keyring is not None:
        LOGGER.debug("Using keyring token storage")
        return keyring_store
    LOGGER.debug("Using file-based token storage (keyring not available)")
    return file_store
