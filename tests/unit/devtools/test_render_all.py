from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from devtools import render_all


def test_render_runs_only_the_named_surface(monkeypatch: pytest.MonkeyPatch) -> None:
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

    assert render_all.main(["docs-surface"]) == 0
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
        (FakeSurface("cli-reference"), FakeSurface("devtools-reference")),
    )

    assert render_all.main(["--check"]) == 0
    assert calls == ["cli-reference", "devtools-reference"]


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


def test_render_all_returns_surface_failure_and_does_not_stamp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(argv: list[str] | None) -> int:
            calls.append(tuple(argv or ()))
            return 7

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")

    assert render_all.main([]) == 7
    assert calls == [()]
    assert not (tmp_path / ".cache" / ".render-cli-reference-stamp").exists()


def test_render_all_surface_exception_is_nonzero_and_typed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            raise OSError("renderer launch failed")

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")

    assert render_all.main([]) == 1
    captured = capsys.readouterr()
    assert "diagnosis: render_surface_exception" in captured.err
    assert "renderer launch failed" in captured.err


@pytest.mark.parametrize("check", [False, True])
def test_render_all_unreadable_input_fails_before_renderer_and_is_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    check: bool,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(argv: list[str] | None) -> int:
            calls.append(tuple(argv or ()))
            return 0

    original_read_bytes = Path.read_bytes

    def unreadable(path: Path) -> bytes:
        if path == source:
            raise PermissionError("synthetic unreadable input")
        return original_read_bytes(path)

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")
    monkeypatch.setattr(Path, "read_bytes", unreadable)

    assert render_all.main(["--check"] if check else []) == 1
    assert calls == []
    captured = capsys.readouterr()
    assert "diagnosis: render_input_unreadable" in captured.err
    assert "unreadable=1" in captured.err


@pytest.mark.parametrize("check", [False, True])
def test_render_all_vanished_input_fails_instead_of_matching_old_stamp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], check: bool
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    calls = 0

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            nonlocal calls
            calls += 1
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")
    args = ["--check"] if check else []
    assert render_all.main(args) == 0
    source.unlink()

    assert render_all.main(args) == 1
    assert calls == 1
    assert "diagnosis: render_input_missing" in capsys.readouterr().err


def test_render_all_invalid_declared_pattern_is_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid_path = tmp_path / "input.sock"
    os.mkfifo(invalid_path)

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(invalid_path),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            raise AssertionError("invalid declarations must fail before rendering")

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")

    assert render_all.main([]) == 1
    captured = capsys.readouterr()
    assert "diagnosis: render_input_invalid" in captured.err
    assert "invalid=1" in captured.err


def test_render_all_stamp_binds_matched_path_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    first = input_dir / "first.py"
    first.write_text("same\n", encoding="utf-8")
    calls = 0

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(input_dir),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            nonlocal calls
            calls += 1
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")
    assert render_all.main([]) == 0
    stamp = json.loads((tmp_path / ".cache" / ".render-cli-reference-stamp").read_text())
    assert stamp["paths"] == [str(first)]

    second = input_dir / "second.py"
    second.write_text("same\n", encoding="utf-8")
    assert render_all.main([]) == 0
    assert calls == 2, "a new path with identical bytes must invalidate freshness"


def test_render_all_successfully_skips_only_after_complete_inventory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    calls = 0

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            nonlocal calls
            calls += 1
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")
    assert render_all.main([]) == 0
    assert render_all.main([]) == 0
    assert calls == 1
    assert "inputs unchanged" in capsys.readouterr().err


def test_render_all_stale_stamp_renders_and_recovery_publishes_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    output = tmp_path / "generated.md"
    failures_remaining = 1
    calls = 0

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            nonlocal calls, failures_remaining
            calls += 1
            if failures_remaining:
                failures_remaining -= 1
                return 9
            output.write_text("complete\n", encoding="utf-8")
            return 0

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")
    stamp = tmp_path / ".cache" / ".render-cli-reference-stamp"
    stamp.parent.mkdir()
    stamp.write_text("stale\n", encoding="utf-8")

    assert render_all.main([]) == 9
    assert not stamp.exists()
    assert render_all.main([]) == 0
    assert output.read_text(encoding="utf-8") == "complete\n"
    assert stamp.exists()
    assert render_all.main([]) == 0
    assert calls == 2


@pytest.mark.parametrize(
    ("exit_code", "expected_code"),
    [(0, 1), (None, 1), (7, 7)],
)
def test_render_all_surface_system_exit_fails_closed_without_stamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    exit_code: int | None,
    expected_code: int,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    cache_dir = tmp_path / ".cache"
    stamp = cache_dir / ".render-cli-reference-stamp"
    stamp.parent.mkdir()
    stamp.write_text("old\n", encoding="utf-8")

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(_argv: list[str] | None) -> int:
            raise SystemExit(exit_code)

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", cache_dir)

    assert render_all.main([]) == expected_code
    assert not stamp.exists()
    captured = capsys.readouterr()
    assert "diagnosis: render_surface_system_exit" in captured.err
    if exit_code is None:
        assert "None" in captured.err


@pytest.mark.parametrize("check", [False, True])
@pytest.mark.parametrize("result", [None, "0"])
def test_render_all_non_integer_surface_result_is_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    check: bool,
    result: object,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")

    class FakeSurface:
        name = "cli-reference"
        inputs = (str(source),)

        @staticmethod
        def main(_argv: list[str] | None) -> object:
            return result

    monkeypatch.setattr(render_all, "GENERATED_SURFACES", (FakeSurface(),))
    monkeypatch.setattr(render_all, "CACHE_DIR", tmp_path / ".cache")

    assert render_all.main(["--check"] if check else []) == 1
    captured = capsys.readouterr()
    assert "diagnosis: render_surface_invalid_result" in captured.err
