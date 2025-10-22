from __future__ import annotations

from io import StringIO

import pytest

from polylogue import ui as ui_module
from polylogue.ui import UI


@pytest.mark.skipif(ui_module.Console is None, reason="Rich Console unavailable")
def test_ui_summary_renders_bracket_text_without_markup_error(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("gum not present during test")

    monkeypatch.setattr(ui_module.subprocess, "run", fake_run)

    capture = StringIO()
    console = ui_module.Console(file=capture, force_terminal=False, legacy_windows=False)
    view = UI(plain=False)
    view.console = console

    lines = ["Copy config to [/tmp/example.json, /tmp/fallback.json]"]
    view.summary("Doctor", lines)

    output = capture.getvalue()
    assert "Copy config" in output
