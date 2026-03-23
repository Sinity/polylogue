"""Compact unit contracts for health caching and reporting."""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("payload", "write_bytes", "expected"),
    [
        (None, False, None),
        ("{ this is not valid json }", False, None),
        ('["not", "a", "dict"]', False, None),
        (b"\xff\xfe invalid utf8", True, None),
        (json.dumps({"status": "ok", "checks": []}), False, {"status": "ok", "checks": []}),
    ],
)
def test_load_cached_contract(tmp_path: Path, payload: str | bytes | None, write_bytes: bool, expected) -> None:
    from polylogue.health_cache import load_cached

    if payload is not None:
        cache_file = tmp_path / "health.json"
        if write_bytes:
            cache_file.write_bytes(payload)
        else:
            cache_file.write_text(payload, encoding="utf-8")

    assert load_cached(tmp_path) == expected


def test_load_cached_permission_error_contract(tmp_path: Path) -> None:
    from polylogue.health_cache import load_cached

    cache_file = tmp_path / "health.json"
    cache_file.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    try:
        os.chmod(cache_file, 0)
        assert load_cached(tmp_path) is None
    finally:
        os.chmod(cache_file, stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.parametrize("payload", ["{ invalid json", b"\xff\xfe invalid utf-8"])
def test_load_cached_logs_failures(tmp_path: Path, payload: str | bytes, capsys: pytest.CaptureFixture[str]) -> None:
    from polylogue.health_cache import load_cached

    cache_file = tmp_path / "health.json"
    if isinstance(payload, bytes):
        cache_file.write_bytes(payload)
    else:
        cache_file.write_text(payload, encoding="utf-8")

    assert load_cached(tmp_path) is None
    captured = capsys.readouterr()
    assert "cache" in (captured.out + captured.err).lower()


@pytest.mark.parametrize("nested", [False, True])
def test_write_cache_contract(tmp_path: Path, nested: bool) -> None:
    from polylogue.health_cache import write_cache

    archive_root = tmp_path / "nested" / "deep" if nested else tmp_path
    payload = {"status": "ok", "checks": []}
    write_cache(archive_root, payload)
    assert json.loads((archive_root / "health.json").read_text(encoding="utf-8")) == payload


def test_write_cache_logs_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.health_cache import write_cache

    original = Path.write_text

    def failing_write(self: Path, data: str, encoding: str | None = None) -> int:
        if self.name == "health.json":
            raise OSError("boom")
        return original(self, data, encoding=encoding)

    monkeypatch.setattr(Path, "write_text", failing_write)
    write_cache(tmp_path, {"status": "ok"})
    captured = capsys.readouterr()
    assert "cache" in (captured.out + captured.err).lower()


@pytest.mark.parametrize(
    ("name", "status_name", "detail", "expected"),
    [
        ("test", "OK", "All good", {"name": "test", "status": "ok", "count": 0, "detail": "All good", "breakdown": {}}),
        ("database", "ERROR", "Connection failed", {"name": "database", "status": "error", "count": 0, "detail": "Connection failed", "breakdown": {}}),
        ("archive", "WARNING", "Directory missing", {"name": "archive", "status": "warning", "count": 0, "detail": "Directory missing", "breakdown": {}}),
    ],
)
def test_health_check_dataclass_contract(name: str, status_name: str, detail: str, expected: dict[str, object]) -> None:
    from polylogue.health_models import HealthCheck, VerifyStatus

    check = HealthCheck(name=name, status=getattr(VerifyStatus, status_name), summary=detail)
    assert {
        "name": check.name,
        "status": check.status.value,
        "count": check.count,
        "detail": check.summary,
        "breakdown": check.breakdown,
    } == expected


@pytest.mark.parametrize(
    ("deep", "expected_checks"),
    [
        (False, {"config", "archive_root", "render_root", "database", "index", "fts_sync", "schemas_coverage", "schemas_freshness"}),
        (True, {"config", "archive_root", "render_root", "database", "sqlite_integrity", "index", "fts_sync", "schemas_coverage", "schemas_freshness"}),
    ],
)
def test_run_health_core_contract(cli_workspace, deep: bool, expected_checks: set[str]) -> None:
    from polylogue.config import get_config
    from polylogue.health_archive import run_archive_health

    report = run_archive_health(get_config(), deep=deep)
    names = {check.name for check in report.checks}
    assert report.timestamp > 0
    assert expected_checks.issubset(names)
    assert sum(report.summary.values()) == len(report.checks)


@pytest.mark.parametrize(
    ("path_name", "missing"),
    [("archive_root", True), ("render_root", True), ("archive_root", False), ("render_root", False)],
)
def test_run_health_path_contracts(tmp_path: Path, path_name: str, missing: bool) -> None:
    from polylogue.config import Config, Source
    from polylogue.health_archive import run_archive_health
    from polylogue.health_models import VerifyStatus

    archive_root = tmp_path / "archive"
    render_root = tmp_path / "render"
    if path_name != "archive_root" or not missing:
        archive_root.mkdir(parents=True, exist_ok=True)
    if path_name != "render_root" or not missing:
        render_root.mkdir(parents=True, exist_ok=True)

    report = run_archive_health(
        Config(archive_root=archive_root, render_root=render_root, sources=[Source(name="test", path=tmp_path)])
    )
    check = next(c for c in report.checks if c.name == path_name)
    assert check.status == (VerifyStatus.WARNING if missing else VerifyStatus.OK)


def test_run_health_includes_source_checks(cli_workspace) -> None:
    from polylogue.config import get_config
    from polylogue.health_archive import run_archive_health

    config = get_config()
    report = run_archive_health(config)
    source_checks = [check for check in report.checks if check.name.startswith("source:")]
    assert len(source_checks) >= len(config.sources)


def test_run_health_writes_cache(cli_workspace) -> None:
    from polylogue.config import get_config
    from polylogue.health_archive import run_archive_health
    from polylogue.health_cache import cache_path

    config = get_config()
    report = run_archive_health(config)
    cached = json.loads(cache_path(config.archive_root).read_text(encoding="utf-8"))
    assert cached["timestamp"] == report.timestamp
    assert len(cached["checks"]) == len(report.checks)


@pytest.mark.parametrize("deep", [False, True])
def test_get_health_contract(cli_workspace, deep: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from polylogue.config import get_config
    from polylogue.health_archive import get_health
    from polylogue.health_cache import cache_path, load_cached
    from polylogue.health_models import HEALTH_TTL_SECONDS

    config = get_config()
    first = get_health(config, deep=deep)
    assert first.timestamp > 0
    if deep:
        assert first.cached is False
        return

    second = get_health(config)
    assert second.cached is True
    cached = load_cached(config.archive_root)
    assert cached is not None
    cached["timestamp"] = int(time.time()) - HEALTH_TTL_SECONDS - 100
    cache_path(config.archive_root).write_text(json.dumps(cached), encoding="utf-8")
    refreshed = get_health(config)
    assert refreshed.cached is False
    assert refreshed.age_seconds == 0


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, "not run"),
        ("{invalid json", "not run"),
        (json.dumps({"checks": []}), "cached"),
        (json.dumps({"timestamp": "bad", "checks": []}), "unknown"),
        (
            json.dumps(
                {
                    "timestamp": 1700000000,
                    "summary": {"ok": 2, "warning": 1, "error": 1},
                    "checks": [
                        {"name": "check1", "status": "ok", "detail": "ok"},
                        {"name": "check2", "status": "ok", "detail": "ok"},
                        {"name": "check3", "status": "warning", "detail": "warning"},
                        {"name": "check4", "status": "error", "detail": "error"},
                        "not-a-dict",
                    ],
                }
            ),
            "ok=2",
        ),
    ],
)
def test_cached_health_summary_contract(tmp_path: Path, payload: str | None, expected: str) -> None:
    from polylogue.health_cache import cached_health_summary

    if payload is not None:
        (tmp_path / "health.json").write_text(payload, encoding="utf-8")
    result = cached_health_summary(tmp_path)
    assert expected in result if expected.startswith(("cached", "ok=")) else result == expected


def test_verify_status_and_cache_path_contract(tmp_path: Path) -> None:
    from polylogue.health_cache import cache_path
    from polylogue.health_models import HEALTH_TTL_SECONDS, VerifyStatus

    assert str(VerifyStatus.OK) == "ok"
    assert str(VerifyStatus.WARNING) == "warning"
    assert str(VerifyStatus.ERROR) == "error"
    assert 60 <= HEALTH_TTL_SECONDS <= 3600
    assert cache_path(tmp_path) == tmp_path / "health.json"
