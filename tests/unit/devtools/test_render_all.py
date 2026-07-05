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
        (FakeSurface("cli-reference"), FakeSurface("docs-surface")),
    )

    assert render_all.main(["--skip", "cli-reference"]) == 0
    assert calls == [("docs-surface", ())]


def test_render_all_check_passes_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    class FakeSurface:
        name = "cli-reference"

        @staticmethod
        def main(argv: list[str] | None) -> int:
            calls.append(tuple(argv or ()))
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))

    assert render_all.main(["--check"]) == 0
    assert calls == [("--check",)]


def test_render_all_check_runs_surfaces_in_registry_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeSurface:
        def __init__(self, name: str):
            self.name = name

        def main(self, argv: list[str] | None) -> int:
            assert argv == ["--check"]
            calls.append(self.name)
            return 0

    monkeypatch.setattr(
        render_all,
        "GENERATED_SURFACES",
        (FakeSurface("cli-reference"), FakeSurface("devtools-reference"), FakeSurface("quality-reference")),
    )

    assert render_all.main(["--check"]) == 0
    assert calls == ["cli-reference", "devtools-reference", "quality-reference"]


def test_render_all_reports_surface_progress(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeSurface:
        name = "cli-reference"

        @staticmethod
        def main(argv: list[str] | None) -> int:
            del argv
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))

    assert render_all.main([]) == 0
    captured = capsys.readouterr()
    assert "render all: render cli-reference" in captured.err
