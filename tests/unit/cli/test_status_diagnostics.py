"""Unit tests for first-run status diagnostics (#1263)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from polylogue.cli.commands.status_diagnostics import (
    StatusDiagnostic,
    diagnose_first_run,
    diagnostic_payload,
)


def _set_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(data_home, config_home)`` derived from the autouse fixture.

    The autouse ``_clear_polylogue_env`` already sets ``XDG_DATA_HOME`` to
    ``tmp_path / "xdg-data"`` and ``XDG_CONFIG_HOME`` to
    ``tmp_path / "xdg-config"`` and strips ``POLYLOGUE_ARCHIVE_ROOT`` so
    archive root defaults to ``data_home()``. We mirror those values here
    and override the daemon URL so the tutorial probe never blocks.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    data_home = tmp_path / "xdg-data" / "polylogue"
    config_home = tmp_path / "xdg-config" / "polylogue"
    return data_home, config_home


class TestDiagnoseNoArchive:
    def test_no_archive_no_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _set_xdg(monkeypatch, tmp_path)
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "no_archive"
        assert "polylogue init" in diag.next_action

    def test_no_archive_with_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _, config_home = _set_xdg(monkeypatch, tmp_path)
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text('[sources]\nroots = ["/x"]\n')
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "no_archive"
        assert diag.next_action == "polylogued run"


class TestDiagnoseSchemaMismatch:
    def test_schema_mismatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        db = data_home / "polylogue.db"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "schema_mismatch"
        assert "99" in diag.headline
        assert "polylogue reset" in diag.next_action

    def test_schema_match_returns_none_for_this_probe(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from polylogue.storage.sqlite.schema_ddl import SCHEMA_VERSION

        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        db = data_home / "polylogue.db"
        conn = sqlite3.connect(db)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
        conn.close()
        diag = diagnose_first_run(daemon_alive=False)
        # Schema matches; whatever else is returned, it must not be a schema mismatch.
        assert diag.kind != "schema_mismatch"


class TestDiagnoseLockedDb:
    def test_locked_db(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        db = data_home / "polylogue.db"

        def _raise_locked(*args: object, **kwargs: object) -> sqlite3.Connection:
            raise sqlite3.OperationalError("database is locked")

        db.touch()
        with patch("sqlite3.connect", side_effect=_raise_locked):
            diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "locked_db"
        assert "polylogue check" in diag.next_action


class TestDiagnoseStalePidfile:
    def test_stale_pidfile_dead_pid(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        db = data_home / "polylogue.db"
        # Healthy db so we reach the pidfile probe.
        sqlite3.connect(db).close()
        archive = data_home  # archive_root() defaults to data_home() with no env override
        archive.mkdir(parents=True, exist_ok=True)
        # Use a PID we can be confident is dead.
        (archive / "daemon.pid").write_text("99999999\n")
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "stale_pidfile"
        assert "polylogued" in diag.next_action

    def test_stale_pidfile_unreadable(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        archive = data_home  # archive_root() defaults to data_home() with no env override
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "daemon.pid").write_text("not-a-number")
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "stale_pidfile"

    def test_pidfile_skipped_when_daemon_alive(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        archive = data_home  # archive_root() defaults to data_home() with no env override
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "daemon.pid").write_text("99999999\n")
        diag = diagnose_first_run(daemon_alive=True)
        assert diag.kind != "stale_pidfile"


class TestDiagnoseNoSources:
    def test_no_sources_when_config_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, config_home = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text("[sources]\nroots = []\n")
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "no_sources"
        assert "polylogue init" in diag.next_action

    def test_no_sources_when_config_missing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, _ = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "no_sources"


class TestDiagnoseNoDaemon:
    def test_no_daemon_when_db_healthy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, config_home = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text('[sources]\nroots = ["/some/path"]\n')
        diag = diagnose_first_run(daemon_alive=False)
        assert diag.kind == "no_daemon"
        assert "polylogued run" in diag.next_action


class TestDiagnoseMissingOptionalDep:
    def test_missing_sqlite_vec(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, config_home = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text('[sources]\nroots = ["/x"]\n')
        # Force find_spec("sqlite_vec") → None.
        import importlib.util

        original = importlib.util.find_spec

        def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sqlite_vec":
                return None
            return original(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        diag = diagnose_first_run(daemon_alive=True)
        assert diag.kind == "missing_optional_dep"
        assert "sqlite-vec" in diag.next_action


class TestHealthy:
    def test_healthy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        data_home, config_home = _set_xdg(monkeypatch, tmp_path)
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "polylogue.db").close()
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text('[sources]\nroots = ["/x"]\n')
        # Pretend sqlite_vec is installed.
        import importlib.util

        # If the env has it, great; if not, fake it.
        if importlib.util.find_spec("sqlite_vec") is None:
            original = importlib.util.find_spec

            def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "sqlite_vec":
                    return object()  # truthy non-None
                return original(name, *args, **kwargs)

            monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        diag = diagnose_first_run(daemon_alive=True)
        assert diag.kind == "healthy"


class TestDiagnosticPayload:
    def test_payload_keys(self) -> None:
        diag = StatusDiagnostic(
            kind="no_archive",
            headline="X",
            detail="Y",
            next_action="Z",
        )
        payload = diagnostic_payload(diag)
        assert payload == {
            "kind": "no_archive",
            "headline": "X",
            "detail": "Y",
            "next_action": "Z",
        }
