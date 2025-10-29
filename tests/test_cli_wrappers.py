from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from polylogue.commands import CommandEnv
from polylogue.cli.imports import run_import_chatgpt
from polylogue.cli.render import run_render_cli
from polylogue.cli.watch import run_watch_cli
from polylogue.importers import ImportResult
from polylogue.local_sync import LocalSyncResult


class RecordingConsole:
    def __init__(self) -> None:
        self.messages: List[str] = []

    def print(self, *objects: object, **_: object) -> None:
        text = " ".join(str(obj) for obj in objects)
        self.messages.append(text)


@dataclass
class RecordingUI:
    plain: bool = True
    console: RecordingConsole = field(default_factory=RecordingConsole)
    summaries: List[tuple[str, List[str]]] = field(default_factory=list)
    banners: List[tuple[str, Optional[str]]] = field(default_factory=list)

    def summary(self, title: str, lines: List[str]) -> None:
        self.summaries.append((title, list(lines)))

    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        self.banners.append((title, subtitle))

    def confirm(self, prompt: str, *, default: bool = True) -> bool:  # pragma: no cover - defensive
        self.console.print(prompt)
        return default

    def choose(self, *args, **kwargs):  # pragma: no cover - defensive
        return None

    def input(self, *args, **kwargs):  # pragma: no cover - defensive
        return None


def _build_render_args(input_path: Path, out_dir: Path, *, json_mode: bool = False) -> Namespace:
    return Namespace(
        input=input_path,
        out=str(out_dir),
        collapse_threshold=None,
        links_only=True,
        dry_run=True,
        force=False,
        html_mode="off",
        diff=False,
        to_clipboard=False,
        json=json_mode,
    )


def _write_render_input(path: Path) -> None:
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hello"},
                {"role": "model", "text": "Hi there!"},
            ]
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_run_render_cli_json_output(tmp_path, capsys):
    src = tmp_path / "sample.json"
    _write_render_input(src)
    args = _build_render_args(src, tmp_path / "out", json_mode=True)
    ui = RecordingUI()

    run_render_cli(args, CommandEnv(ui=ui), json_output=True)

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["cmd"] == "render"
    assert payload["count"] == 1
    assert payload["files"][0]["slug"] == "sample"


def test_run_render_cli_plain_summary(tmp_path):
    src = tmp_path / "sample.json"
    _write_render_input(src)
    args = _build_render_args(src, tmp_path / "out")
    ui = RecordingUI()

    run_render_cli(args, CommandEnv(ui=ui), json_output=False)

    assert ui.summaries, "expected render summary to be emitted"
    title, lines = ui.summaries[0]
    assert title == "Render"
    assert any("Rendered" in line for line in lines)


def test_run_import_chatgpt_invokes_summary(monkeypatch, tmp_path):
    markdown_path = tmp_path / "chat" / "conversation.md"
    result = ImportResult(
        markdown_path=markdown_path,
        html_path=None,
        attachments_dir=None,
        document=None,
        slug="chat-slug",
    )

    def fake_import_chatgpt_export(*args, **kwargs):
        return [result]

    monkeypatch.setattr("polylogue.cli.imports.import_chatgpt_export", fake_import_chatgpt_export)
    ui = RecordingUI()
    export_path = tmp_path / "export.zip"
    export_path.write_text("stub", encoding="utf-8")
    args = Namespace(
        export_path=export_path,
        out=str(tmp_path / "out"),
        collapse_threshold=None,
        html_mode="off",
        force=False,
        conversation_ids=[],
        all=True,
        json=False,
        to_clipboard=False,
    )

    run_import_chatgpt(args, CommandEnv(ui=ui))

    assert ui.summaries, "expected import summary to be emitted"
    title, _ = ui.summaries[0]
    assert title == "ChatGPT Import"


def test_run_watch_cli_reports_missing_dependency(monkeypatch):
    monkeypatch.setattr("polylogue.cli.watch.watch_directory", None)
    ui = RecordingUI()
    args = Namespace(provider="codex")

    run_watch_cli(args, CommandEnv(ui=ui))

    assert any("watchfiles package is not available" in message for message in ui.console.messages)


def test_run_watch_codex_once_triggers_sync(monkeypatch, tmp_path):
    from polylogue.cli import watch as watch_module

    calls = []

    def fake_sync_codex_sessions(
        *,
        base_dir,
        output_dir,
        collapse_threshold,
        html,
        html_theme,
        force,
        prune,
        diff,
        sessions=None,
    ):
        calls.append(
            {
                "base_dir": Path(base_dir),
                "output_dir": Path(output_dir),
                "collapse": collapse_threshold,
                "html": html,
                "html_theme": html_theme,
                "force": force,
                "prune": prune,
                "diff": diff,
                "sessions": sessions,
            }
        )
        return LocalSyncResult(written=[], skipped=0, pruned=0, output_dir=Path(output_dir))

    monkeypatch.setattr(watch_module, "watch_directory", lambda *args, **kwargs: iter(()))
    monkeypatch.setattr(watch_module, "sync_codex_sessions", fake_sync_codex_sessions)

    records: List[dict] = []
    monkeypatch.setattr("polylogue.cli.sync.add_run", lambda data: records.append(data))

    ui = RecordingUI()
    args = Namespace(
        provider="codex",
        base_dir=str(tmp_path / "sessions"),
        out=str(tmp_path / "out"),
        collapse_threshold=None,
        html_mode="off",
        debounce=0.1,
        once=True,
        diff=False,
        force=False,
        prune=False,
    )

    run_watch_cli(args, CommandEnv(ui=ui))

    assert calls, "expected sync_codex_sessions to be invoked"
    assert ui.banners and ui.banners[0][0] == "Watching Codex sessions"
    assert records and records[0]["cmd"] == "codex-watch"
