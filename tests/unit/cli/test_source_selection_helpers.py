"""Focused tests for CLI source-selection and path helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from polylogue.cli import helpers
from polylogue.config import Config, Source


@pytest.fixture
def helpers_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    config_dir = tmp_path / "config"
    for directory in (data_dir, state_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    return {
        "data_dir": data_dir,
        "state_dir": state_dir,
        "config_dir": config_dir,
    }


@pytest.mark.parametrize(
    ("command", "message", "expected"),
    [("test_cmd", "something broke", "test_cmd: something broke"), ("test_cmd", "", "test_cmd:")],
)
def test_fail_raises_system_exit(command: str, message: str, expected: str) -> None:
    with pytest.raises(SystemExit, match=expected):
        helpers.fail(command, message)


def test_source_state_path_defaults_without_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    result = helpers.source_state_path()
    assert result == tmp_path / ".local" / "state" / "polylogue" / "last-source.json"


def test_source_state_path_uses_xdg_state_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", "/custom/state")
    result = helpers.source_state_path()
    assert result == Path("/custom/state/polylogue/last-source.json")


def test_load_last_source_returns_none_for_missing(helpers_workspace: dict[str, Path]) -> None:
    assert helpers.load_last_source() is None


def test_load_last_source_returns_none_for_invalid_json(helpers_workspace: dict[str, Path]) -> None:
    path = helpers.source_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    assert helpers.load_last_source() is None


def test_save_and_load_last_source_roundtrip(helpers_workspace: dict[str, Path]) -> None:
    helpers.save_last_source("chatgpt")
    path = helpers.source_state_path()
    assert json.loads(path.read_text(encoding="utf-8")) == {"source": "chatgpt"}
    assert helpers.load_last_source() == "chatgpt"


@pytest.mark.parametrize(
    ("sources", "expected"),
    [
        (("chatgpt",), ["chatgpt"]),
        (("chatgpt", "claude"), {"chatgpt", "claude"}),
        (("chatgpt", "chatgpt"), ["chatgpt"]),
        ((), None),
    ],
)
def test_resolve_sources_valid_cases(sources: tuple[str, ...], expected: list[str] | set[str] | None) -> None:
    config = MagicMock()
    config.sources = [
        Source(name="chatgpt", path=Path("/data")),
        Source(name="claude", path=Path("/data2")),
    ]
    result = helpers.resolve_sources(config, sources, "test_cmd")
    if isinstance(expected, set):
        assert set(result or []) == expected
    else:
        assert result == expected


@pytest.mark.parametrize("sources", [("unknown",), ("chatgpt", "unknown")])
def test_resolve_sources_rejects_unknown_sources(sources: tuple[str, ...]) -> None:
    config = MagicMock()
    config.sources = [Source(name="chatgpt", path=Path("/data"))]
    with pytest.raises(SystemExit):
        helpers.resolve_sources(config, sources, "test_cmd")


def test_resolve_sources_expands_last(helpers_workspace: dict[str, Path]) -> None:
    helpers.save_last_source("chatgpt")
    config = MagicMock()
    config.sources = [Source(name="chatgpt", path=Path("/data"))]
    assert helpers.resolve_sources(config, ("last",), "test_cmd") == ["chatgpt"]


def test_resolve_sources_rejects_last_without_saved_source(helpers_workspace: dict[str, Path]) -> None:
    config = MagicMock()
    config.sources = []
    with pytest.raises(SystemExit):
        helpers.resolve_sources(config, ("last",), "test_cmd")


def test_resolve_sources_rejects_last_combined_with_other_sources(helpers_workspace: dict[str, Path]) -> None:
    config = MagicMock()
    config.sources = [Source(name="chatgpt", path=Path("/data"))]
    with pytest.raises(SystemExit):
        helpers.resolve_sources(config, ("last", "chatgpt"), "test_cmd")


def _config_with_sources(tmp_path: Path, names: list[str]) -> Config:
    return Config(
        archive_root=tmp_path / "archive",
        render_root=tmp_path / "render",
        sources=[Source(name=name, path=tmp_path / name) for name in names],
    )


def test_maybe_prompt_sources_returns_selected_sources_if_provided(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = True
    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), ["one"], "sync")
    assert result == ["one"]


def test_maybe_prompt_sources_returns_none_in_plain_mode(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = True
    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")
    assert result is None


def test_maybe_prompt_sources_skips_prompt_for_single_source(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose = MagicMock()
    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one"]), None, "sync")
    assert result is None
    env.ui.choose.assert_not_called()


def test_maybe_prompt_sources_prompts_and_persists_selection(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose.return_value = "two"

    result = helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")

    assert result == ["two"]
    assert helpers.load_last_source() == "two"
    prompt, options = env.ui.choose.call_args.args
    assert prompt == "Select source for sync"
    assert options == ["all", "one", "two"]


def test_maybe_prompt_sources_prioritizes_saved_selection(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    helpers.save_last_source("two")
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose.return_value = "two"

    helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")

    _, options = env.ui.choose.call_args.args
    assert options[0] == "two"
    assert options[1:] == ["all", "one"]


def test_maybe_prompt_sources_rejects_empty_choice(
    helpers_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    env = MagicMock()
    env.ui.plain = False
    env.ui.choose.return_value = None
    with pytest.raises(SystemExit):
        helpers.maybe_prompt_sources(env, _config_with_sources(tmp_path, ["one", "two"]), None, "sync")


@pytest.mark.parametrize(
    "case_id",
    [
        "nonexistent-root",
        "empty-root",
        "single-markdown",
        "single-html",
        "latest-mtime-wins",
        "missing-candidate-is-skipped",
    ],
)
def test_latest_render_path_contract(tmp_path: Path, case_id: str) -> None:
    render_root = tmp_path / "render"
    expected: Path | None = None

    if case_id == "nonexistent-root":
        render_root = tmp_path / "missing"
    elif case_id == "empty-root":
        render_root.mkdir()
    elif case_id == "single-markdown":
        conv_dir = render_root / "conv1"
        conv_dir.mkdir(parents=True)
        expected = conv_dir / "conversation.md"
        expected.write_text("# Test", encoding="utf-8")
    elif case_id == "single-html":
        conv_dir = render_root / "conv1"
        conv_dir.mkdir(parents=True)
        expected = conv_dir / "conversation.html"
        expected.write_text("<html>test</html>", encoding="utf-8")
    elif case_id == "latest-mtime-wins":
        import os

        conv1 = render_root / "conv1"
        conv2 = render_root / "conv2"
        conv1.mkdir(parents=True)
        conv2.mkdir(parents=True)
        older = conv1 / "conversation.md"
        expected = conv2 / "conversation.html"
        older.write_text("old", encoding="utf-8")
        expected.write_text("new", encoding="utf-8")
        os.utime(older, (100, 100))
        os.utime(expected, (200, 200))
    elif case_id == "missing-candidate-is-skipped":
        conv_dir = render_root / "conv1"
        conv_dir.mkdir(parents=True)
        expected = conv_dir / "conversation.md"
        expected.write_text("# Existing", encoding="utf-8")
        original_rglob = Path.rglob

        def fake_rglob(self: Path, pattern: str):
            if self == render_root and pattern in {"conversation.md", "conversation.html"}:
                missing = render_root / "deleted" / pattern
                return list(original_rglob(self, pattern)) + [missing]
            return original_rglob(self, pattern)

        with patch.object(Path, "rglob", fake_rglob):
            assert helpers.latest_render_path(render_root) == expected
        return

    assert helpers.latest_render_path(render_root) == expected
