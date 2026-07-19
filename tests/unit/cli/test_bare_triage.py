"""TTY-only bare invocation triage contracts (polylogue-jnj.8)."""

from __future__ import annotations

from pathlib import Path

import click
import pytest

from polylogue.cli.click_app import _show_bare_tty_triage, cli
from polylogue.cli.onboarding import GUIDED_PATH_STEPS
from polylogue.cli.shared.types import AppEnv
from polylogue.config import Config


def test_bare_tty_triage_shows_guided_path_without_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No archive is not presented as an empty, resumable archive.

    It also isn't a generic Click help dump: a cold reader gets the one
    numbered guided path shared with `polylogue tutorial`, not a raw
    subcommand list they have no way to sequence yet.
    """

    config = Config(archive_root=tmp_path / "archive", render_root=tmp_path / "render", sources=[])
    monkeypatch.setattr("polylogue.cli.shared.helpers.load_effective_config", lambda _env: config)
    context = click.Context(cli, info_name="polylogue")
    assert _show_bare_tty_triage(context, AppEnv(plain=True))

    output = capsys.readouterr().out
    assert "Usage: polylogue" not in output
    assert "Recent sessions:" not in output
    assert "Guided path" in output
    for step in GUIDED_PATH_STEPS:
        assert step.title in output
        assert step.command_text in output
    assert "polylogue manual" in output
    assert "polylogue tutorial" in output
