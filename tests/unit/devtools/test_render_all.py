from __future__ import annotations

import pytest

from devtools import render_all


def test_render_all_runs_selected_surfaces(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    class FakeSurface:
        def __init__(self, name: str):
            self.name = name

        def main(self, argv: list[str] | None) -> int:
            calls.append((self.name, tuple(argv or ())))
            return 0

    monkeypatch.setattr(
        render_all,
        "GENERATED_SURFACES",
        (FakeSurface("agents"), FakeSurface("docs-surface")),
    )

    assert render_all.main(["--skip", "agents"]) == 0
    assert calls == [("docs-surface", ())]


def test_render_all_check_passes_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    class FakeSurface:
        name = "agents"

        @staticmethod
        def main(argv: list[str] | None) -> int:
            calls.append(tuple(argv or ()))
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))

    assert render_all.main(["--check"]) == 0
    assert calls == [("--check",)]


def test_render_all_reports_surface_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeSurface:
        name = "agents"

        @staticmethod
        def main(argv: list[str] | None) -> int:
            del argv
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))

    assert render_all.main([]) == 0
    captured = capsys.readouterr()
    assert "render-all: render agents" in captured.err
