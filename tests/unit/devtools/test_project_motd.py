from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import project_motd
from devtools.generated_surfaces import GENERATED_SURFACES, GeneratedSurface


class _UnprintableCode:
    def __str__(self) -> str:
        raise RuntimeError("str failed")

    def __repr__(self) -> str:
        raise RuntimeError("repr failed")


def test_read_version_extracts_project_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")

    assert project_motd.read_version(pyproject) == "1.2.3"


@pytest.mark.parametrize(
    "code, expected",
    [
        (None, "stale"),
        (False, "stale"),
        (True, "stale"),
        (0, "ok"),
        (-7, "stale"),
        (7, "stale"),
        ("", "stale"),
        ("0", "stale"),
        (0.0, "stale"),
        ("boom", "stale"),
        (object(), "stale"),
        (_UnprintableCode(), "stale"),
    ],
)
def test_run_check_fails_closed_for_non_integer_system_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    code: object,
    expected: str,
) -> None:
    """Anti-vacuity: local truthiness or ``isinstance`` checks would accept false-green cases."""

    def fake_main(_argv: list[str] | None) -> int:
        raise SystemExit(code)

    surface = GeneratedSurface("fake", "Fake", "test surface", (), fake_main)

    assert project_motd.run_check(tmp_path, surface) == expected
    assert capsys.readouterr().err == ""


def test_status_verify_generated_reports_system_exit_surface_as_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    def fake_main(_argv: list[str] | None) -> int:
        raise SystemExit("boom")

    monkeypatch.setattr(project_motd, "GENERATED_SURFACES", (GeneratedSurface("fake", "Fake", "test", (), fake_main),))
    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/test", 0, 0, 0))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "test")

    assert project_motd.main(["--cwd", str(tmp_path), "--json", "--verify-generated"]) == 0
    payload = capsys.readouterr()
    assert payload.err == ""
    status = json.loads(payload.out)
    assert status["generated_surfaces"] == {"Fake": "stale"}
    assert status["stale_surfaces"] == ["Fake"]


def test_render_motd_contains_expected_sections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/docs/test", 1, 2, 3))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "docs: tighten repo guides")
    monkeypatch.setattr(project_motd, "run_check", lambda cwd, check: "ok")
    monkeypatch.setattr(project_motd, "use_color", lambda stream=None: False)

    rendered = project_motd.render_motd(tmp_path)
    surface_count = len(GENERATED_SURFACES)

    assert "Polylogue  feature/docs/test  v0.1.0+deadbeef-dirty" in rendered
    assert "worktree   dirty · 1 staged · 2 modified · 3 untracked" in rendered
    assert f"generated  {surface_count}/{surface_count} generated unchecked" in rendered
    assert "head       docs: tighten repo guides" in rendered
    assert (
        "ready      devtools render all --check · devtools verify --quick · devtools release build-package" in rendered
    )
    assert "test       devtools verify" in rendered
    assert "roots      keep .venv/ .direnv/ · cache .cache/ · outputs .local/ · build .local/result" in rendered
    assert "dirty · 1 staged · 2 modified · 3 untracked" in rendered


def test_status_snapshot_includes_machine_readable_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/docs/test", 1, 2, 3))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "docs: tighten repo guides")
    monkeypatch.setattr(project_motd, "run_check", lambda cwd, check: "ok")

    snapshot = project_motd.status_snapshot(tmp_path)

    assert snapshot["project"] == "polylogue"
    assert snapshot["revision"] == "deadbeef"
    assert snapshot["commands"]["discover"] == "devtools --list-commands --json"
    assert snapshot["commands"]["status"] == "devtools status --json"
    assert snapshot["commands"]["verify_quick"] == "devtools verify --quick"
    assert snapshot["commands"]["build_package"] == "devtools release build-package"
    assert snapshot["generated_surfaces"]
    assert snapshot["generated_checked"] is False
    assert set(snapshot["generated_surfaces"].values()) == {"unchecked"}
    assert snapshot["stale_surfaces"] == []
    assert set(snapshot["unchecked_surfaces"]) == set(snapshot["generated_surfaces"].keys())
    assert snapshot["local_state"]["root_residents"] == [".venv/", ".direnv/"]
    assert snapshot["local_state"]["preferred_build_out_link"] == ".local/result"


def test_render_motd_can_verify_generated_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/docs/test", 0, 0, 0))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "docs: tighten repo guides")
    monkeypatch.setattr(project_motd, "run_check", lambda cwd, check: "ok")
    monkeypatch.setattr(project_motd, "use_color", lambda stream=None: False)

    rendered = project_motd.render_motd(tmp_path, verify_generated=True)
    surface_count = len(GENERATED_SURFACES)

    assert f"{surface_count}/{surface_count} generated clean" in rendered


def test_main_can_write_motd_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/docs/test", 0, 0, 0))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "docs: tighten repo guides")
    monkeypatch.setattr(project_motd, "use_color", lambda stream=None: False)

    assert project_motd.main(["--cwd", str(tmp_path), "--stderr"]) == 0
    captured = capsys.readouterr()
    assert "Polylogue  feature/docs/test  v0.1.0+deadbeef" in captured.err
    assert captured.out == ""
