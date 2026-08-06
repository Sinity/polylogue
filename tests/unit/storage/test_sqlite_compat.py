from __future__ import annotations

import builtins
import importlib
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

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


def test_sqlite_compat_boundary_is_inclusive() -> None:
    capable = type("Connection", (), {"enable_load_extension": object()})

    assert _sqlite_compat._needs_sqlite_compat(SimpleNamespace(sqlite_version_info=(3, 42, 9), Connection=capable))
    assert not _sqlite_compat._needs_sqlite_compat(SimpleNamespace(sqlite_version_info=(3, 43, 0), Connection=capable))


def test_sqlite_compat_does_not_select_incompatible_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    original_driver: Any = cast(Any, _sqlite_compat.sqlite3)  # type: ignore[attr-defined]
    incompatible_driver = SimpleNamespace(sqlite_version_info=(3, 51, 0), Connection=object)
    fallback_driver = SimpleNamespace(
        sqlite_version_info=(3, 51, 0),
        Connection=type("Connection", (), {"enable_load_extension": object()}),
    )

    monkeypatch.setattr(_sqlite_compat, "sqlite3", incompatible_driver)
    monkeypatch.setitem(sys.modules, "sqlite3", original_driver)
    monkeypatch.setitem(sys.modules, "pysqlite3", fallback_driver)
    _sqlite_compat._ensure_modern_sqlite()

    assert cast(Any, sys.modules["sqlite3"]) is fallback_driver
    monkeypatch.setattr(_sqlite_compat, "sqlite3", original_driver)


def test_sqlite_compat_keeps_driver_when_fallback_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    original_driver: Any = cast(Any, _sqlite_compat.sqlite3)  # type: ignore[attr-defined]
    incompatible_driver = SimpleNamespace(sqlite_version_info=(3, 42, 0), Connection=object)
    real_import = builtins.__import__

    def import_without_pysqlite3(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pysqlite3":
            raise ImportError("test fallback unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_sqlite_compat, "sqlite3", incompatible_driver)
    monkeypatch.setattr(builtins, "__import__", import_without_pysqlite3)
    _sqlite_compat._ensure_modern_sqlite()

    assert cast(Any, _sqlite_compat.sqlite3) is incompatible_driver  # type: ignore[attr-defined]
    assert sys.modules["sqlite3"] is not incompatible_driver
    monkeypatch.setattr(_sqlite_compat, "sqlite3", original_driver)


def test_sqlite_vec_direct_submodule_import_uses_package_compatibility() -> None:
    sqlite_vec_extension = importlib.import_module("polylogue.storage.sqlite.sqlite_vec_extension")

    assert sqlite_vec_extension.sqlite3 is sys.modules["sqlite3"]
