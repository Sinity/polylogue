from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.cli.search_cli import _open_single_search_hit, find_anchor_line, run_search_cli
from polylogue.commands import CommandEnv
from polylogue.options import SearchHit, SearchResult


def test_find_anchor_line(tmp_path: Path) -> None:
    target = tmp_path / "conversation.md"
    target.write_text(
        """
line-one
<a id=\"msg-2\"></a>
body
""".strip()
    )

    assert find_anchor_line(target, "msg-2") == 2
    assert find_anchor_line(target, "#msg-2") == 2
    assert find_anchor_line(target, "msg-99") is None


class _RecordingConsole:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def print(self, *args: object, **_kwargs: object) -> None:
        self.lines.append(" ".join(str(a) for a in args))


class _RecordingUI:
    plain = True

    def __init__(self) -> None:
        self.console = _RecordingConsole()


def _hit_for_path(path: Path, *, kind: str = "message", position: int = 2) -> SearchHit:
    return SearchHit(
        provider="codex",
        conversation_id="c1",
        slug="slug",
        title="Title",
        branch_id="main",
        message_id="m1",
        position=position,
        timestamp="2024-01-01T00:00:00Z",
        attachment_count=0,
        score=0.5,
        snippet="snippet",
        body="body",
        conversation_path=path,
        branch_path=None,
        model="model",
        kind=kind,
    )


def test_open_single_search_hit_opens_html_in_browser(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "conversation.html"
    target.write_text("<!doctype html><html></html>", encoding="utf-8")
    ui = _RecordingUI()
    hit = _hit_for_path(target)

    opened: list[str] = []

    monkeypatch.setattr("polylogue.cli.search_cli.webbrowser.open", lambda url: opened.append(url) or True)
    monkeypatch.setattr("polylogue.cli.search_cli.open_in_editor", lambda *_a, **_k: pytest.fail("open_in_editor should not be called"))

    _open_single_search_hit(ui, hit)

    assert opened and opened[0].endswith("#msg-2")


def test_open_single_search_hit_opens_markdown_in_editor(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "conversation.md"
    target.write_text('line-one\n<a id="msg-2"></a>\nbody\n', encoding="utf-8")
    ui = _RecordingUI()
    hit = _hit_for_path(target)

    called: dict[str, object] = {}

    def fake_open_in_editor(path: Path, *, line=None):  # noqa: ANN001
        called["path"] = path
        called["line"] = line
        return True

    monkeypatch.setattr("polylogue.cli.search_cli.open_in_editor", fake_open_in_editor)
    monkeypatch.setattr("polylogue.cli.search_cli.webbrowser.open", lambda *_a, **_k: pytest.fail("webbrowser.open should not be called"))

    _open_single_search_hit(ui, hit)

    assert called["path"] == target
    assert called["line"] == 2


def test_search_csv_stdout(monkeypatch, capsys) -> None:
    hit = _hit_for_path(Path("/tmp/conversation.md"))
    args = SimpleNamespace(
        query="hello",
        from_stdin=False,
        limit=20,
        provider=None,
        slug=None,
        conversation_id=None,
        branch=None,
        model=None,
        since=None,
        until=None,
        with_attachments=False,
        without_attachments=False,
        in_attachments=False,
        attachment_name=None,
        fields="",
        csv="-",
        json=False,
        json_lines=False,
        no_picker=True,
        open=False,
    )

    monkeypatch.setattr("polylogue.cli.search_cli.search_command", lambda *_a, **_k: SearchResult(hits=[hit]))

    run_search_cli(args, CommandEnv(ui=_RecordingUI()))

    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("provider,conversationId")
