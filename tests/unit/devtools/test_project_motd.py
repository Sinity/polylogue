from __future__ import annotations

from pathlib import Path

from devtools import project_motd


def test_read_version_extracts_project_version(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "1.2.3"\n', encoding="utf-8")

    assert project_motd.read_version(pyproject) == "1.2.3"


def test_render_motd_contains_expected_sections(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/docs/test", 1, 2, 3))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "docs: tighten repo guides")
    monkeypatch.setattr(project_motd, "run_check", lambda cwd, check: "ok")
    monkeypatch.setattr(project_motd, "use_color", lambda: False)

    rendered = project_motd.render_motd(tmp_path)

    assert "Polylogue  feature/docs/test  v0.1.0+deadbeef-dirty" in rendered
    assert "worktree   dirty · 1 staged · 2 modified · 3 untracked" in rendered
    assert "generated  5 generated surfaces" in rendered
    assert "head       docs: tighten repo guides" in rendered
    assert "run        devtools render-all --check" in rendered
    assert "test       pytest -q --ignore=tests/integration" in rendered
    assert "dirty · 1 staged · 2 modified · 3 untracked" in rendered


def test_status_snapshot_includes_machine_readable_commands(monkeypatch, tmp_path: Path) -> None:
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
    assert snapshot["generated_surfaces"]
    assert snapshot["generated_checked"] is False
    assert set(snapshot["generated_surfaces"].values()) == {"unchecked"}
    assert snapshot["stale_surfaces"] == []
    assert set(snapshot["unchecked_surfaces"]) == set(snapshot["generated_surfaces"].keys())
    assert snapshot["local_state"]["root_residents"] == [".venv/", ".direnv/"]
    assert snapshot["local_state"]["preferred_build_out_link"] == ".local/result"


def test_render_motd_can_verify_generated_surfaces(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")

    monkeypatch.setattr(project_motd, "git_status_summary", lambda cwd: ("feature/docs/test", 0, 0, 0))
    monkeypatch.setattr(project_motd, "git_short_revision", lambda cwd: "deadbeef")
    monkeypatch.setattr(project_motd, "last_commit_subject", lambda cwd: "docs: tighten repo guides")
    monkeypatch.setattr(project_motd, "run_check", lambda cwd, check: "ok")
    monkeypatch.setattr(project_motd, "use_color", lambda: False)

    rendered = project_motd.render_motd(tmp_path, verify_generated=True)

    assert "5/5 generated clean" in rendered
