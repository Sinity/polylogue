"""polylogue-l1qg: ops/maintenance commands must surface archive-root provenance.

Reproduces the bead's live scenario: ``POLYLOGUE_ARCHIVE_ROOT`` inherited from
a shell environment silently outranks ``polylogue.toml``'s ``[archive] root``
with no indication to the operator that the resolved root came from an env
override rather than the config file they believe is authoritative. These
tests exercise the real CLI entry point (``polylogue ops maintenance ...``
via :func:`polylogue.cli.click_app.cli`), not a bare internal helper, so a
regression that stops the banner from actually reaching command output would
fail here even if the underlying provenance-lookup function still worked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli


def test_maintenance_status_prints_env_archive_root_provenance(
    cli_workspace: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ops maintenance status` names POLYLOGUE_ARCHIVE_ROOT as the source.

    ``cli_workspace`` points ``POLYLOGUE_ARCHIVE_ROOT`` at a real, valid
    archive. A user ``polylogue.toml`` is layered in pointing at a
    *different* (bogus) archive root -- env must still win per the
    documented 5-layer precedence, and the printed banner must say so
    rather than silently naming the config file or nothing at all.
    """
    env_archive_root = cli_workspace["archive_root"]
    config_only_root = tmp_path / "config-file-archive-should-not-be-used"
    user_toml = tmp_path / "user.toml"
    user_toml.write_text(f'[archive]\nroot = "{config_only_root.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(user_toml))

    runner = CliRunner()
    result = runner.invoke(cli, ["ops", "maintenance", "status"])

    assert result.exit_code == 0, result.output
    assert f"Archive root: {env_archive_root}" in result.output
    assert "POLYLOGUE_ARCHIVE_ROOT environment variable" in result.output
    # The config-file-only root must never be presented as the resolved one.
    assert str(config_only_root) not in result.output


def test_maintenance_status_prints_user_config_archive_root_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    empty_archive_template: Path,
) -> None:
    """Without an env override, the banner names the user config file, not "env"."""
    monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "unused-data-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.setenv("POLYLOGUE_SCHEMA_VALIDATION", "off")

    archive_root = tmp_path / "toml-configured-archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    import subprocess

    subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{empty_archive_template}/.", str(archive_root)],
        check=True,
    )
    user_toml = tmp_path / "user.toml"
    user_toml.write_text(f'[archive]\nroot = "{archive_root.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(user_toml))

    runner = CliRunner()
    result = runner.invoke(cli, ["ops", "maintenance", "status"])

    assert result.exit_code == 0, result.output
    assert f"Archive root: {archive_root}" in result.output
    assert f"user config file ({user_toml})" in result.output
    assert "POLYLOGUE_ARCHIVE_ROOT environment variable" not in result.output
