from __future__ import annotations

from types import SimpleNamespace

from polylogue import _sqlite_compat


def test_sqlite_compat_requires_extension_capability_even_with_new_sqlite() -> None:
    module = SimpleNamespace(
        sqlite_version_info=(3, 50, 0),
        Connection=type("ConnectionWithoutExtensionLoading", (), {}),
    )

    assert _sqlite_compat._needs_sqlite_compat(module)


def test_sqlite_compat_accepts_modern_extension_capable_sqlite() -> None:
    module = SimpleNamespace(
        sqlite_version_info=(3, 50, 0),
        Connection=type("Connection", (), {"enable_load_extension": object()}),
    )

    assert not _sqlite_compat._needs_sqlite_compat(module)
