"""TTY-only bare invocation triage contracts."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from polylogue.cli.click_app import _show_bare_tty_triage, cli
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config


def test_bare_tty_triage_falls_back_to_click_help_without_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No archive is not presented as an empty, resumable archive."""

    config = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
    monkeypatch.setattr("polylogue.cli.shared.helpers.load_effective_config", lambda _env: config)
    context = click.Context(cli, info_name="polylogue")
    assert _show_bare_tty_triage(context, AppEnv(plain=True))

    output = capsys.readouterr().out
    assert "Usage: polylogue" in output
    assert "Recent sessions:" not in output
