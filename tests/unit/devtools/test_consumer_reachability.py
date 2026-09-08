"""Regression coverage for polylogue-gb4e consumer-reachability V3/R2."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools.consumer_reachability import (
    ConsumerReachabilityError,
    _authority,
    _cached_reachability,
    _store_reachability,
    _waivers,
    check,
)


def test_polylogue_gb4e_v3_r2_production_route_has_current_authority() -> None:
    """The owning command executes against the current production checkout."""
    root = Path(__file__).parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report = check(root, base=head, head=head)
    assert report.ok
    assert report.base == report.head


@pytest.mark.parametrize("value", ["HEAD", "not-a-sha", "0" * 39])
def test_polylogue_gb4e_rejects_malformed_authority(value: str) -> None:
    with pytest.raises(ConsumerReachabilityError, match="malformed authority"):
        _authority(Path(__file__).parents[3], value, value)


def test_polylogue_gb4e_rejects_stale_head() -> None:
    root = Path(__file__).parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=root, text=True).strip()
    with pytest.raises(ConsumerReachabilityError, match="stale authority head"):
        _authority(root, head, parent)


def test_polylogue_gb4e_rejects_duplicate_waiver(tmp_path: Path) -> None:
    body = tmp_path / "body.txt"
    body.write_text(
        "consumer-reachability-waiver: polylogue/x.py first reason is sufficiently explicit\n"
        "consumer-reachability-waiver: polylogue/x.py second reason is sufficiently explicit\n",
        encoding="utf-8",
    )
    with pytest.raises(ConsumerReachabilityError, match="duplicate waiver"):
        _waivers(body)


def test_reachability_cache_round_trips_per_worktree(tmp_path: Path) -> None:
    entrypoints = ("polylogue.api", "polylogue.cli")
    reachable = {"polylogue.api.main"}
    reachable_modules = frozenset({"polylogue.api", "polylogue.core"})
    (tmp_path / "polylogue").mkdir()
    (tmp_path / "devtools").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project.scripts]\n", encoding="utf-8")

    assert _cached_reachability(tmp_path, entrypoints) is None
    _store_reachability(tmp_path, entrypoints, reachable, reachable_modules)

    assert _cached_reachability(tmp_path, entrypoints) == (reachable, reachable_modules)


@pytest.mark.parametrize("with_reader", [False, True])
def test_table_reader_can_share_its_creation_module(tmp_path: Path, with_reader: bool) -> None:
    """Removing the SELECT must flag the table even when its module is reachable."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True, stderr=subprocess.PIPE).strip()

    git("init")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.test")
    (tmp_path / "polylogue").mkdir()
    (tmp_path / "pyproject.toml").write_text("[project.scripts]\n", encoding="utf-8")
    module = tmp_path / "polylogue" / "api.py"
    module.write_text('"""Synthetic entrypoint."""\n', encoding="utf-8")
    git("add", "pyproject.toml", "polylogue/api.py")
    git("commit", "-m", "Initialize fixture")
    base = git("rev-parse", "HEAD")
    module.write_text(
        'DDL = "CREATE TABLE source_evidence (value TEXT)"\n'
        + ('QUERY = "SELECT value FROM source_evidence"\n' if with_reader else ""),
        encoding="utf-8",
    )
    git("add", "polylogue/api.py")
    git("commit", "-m", "Add evidence table")

    report = check(tmp_path, base=base, head=git("rev-parse", "HEAD"))

    assert report.ok is with_reader
    assert [(finding.kind, finding.target) for finding in report.findings] == (
        [] if with_reader else [("table", "source_evidence")]
    )
