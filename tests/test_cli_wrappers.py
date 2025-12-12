from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

from polylogue.commands import CommandEnv
from polylogue.cli.imports import run_import_chatgpt
from polylogue.cli.render import run_render_cli
from polylogue.cli.watch import run_watch_cli
from polylogue.cli.search_cli import run_search_cli
from polylogue.importers import ImportResult
from polylogue.local_sync import LocalSyncResult, LocalSyncProvider


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


def _build_render_args(input_path: Path, out_dir: Path, *, json_mode: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
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
    args = SimpleNamespace(
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


def test_run_search_cli_displays_summary(monkeypatch, tmp_path):
    from polylogue.options import SearchHit, SearchResult

    hit = SearchHit(
        provider="chatgpt",
        conversation_id="conv-1",
        slug="conv-1",
        title="Sample Conversation",
        branch_id="branch-000",
        message_id="msg-1",
        position=1,
        timestamp="2024-01-01T00:00:01Z",
        attachment_count=0,
        score=0.95,
        snippet="example snippet",
        body="example body",
        conversation_path=tmp_path / "conv-1" / "conversation.md",
        branch_path=None,
        model="gpt-test",
    )

    def fake_search_command(options, env=None):  # pylint: disable=unused-argument
        return SearchResult(hits=[hit])

    monkeypatch.setattr("polylogue.cli.search_cli.search_command", fake_search_command)

    args = SimpleNamespace(
        query="example",
        limit=10,
        provider=None,
        slug=None,
        conversation_id=None,
        branch=None,
        model=None,
        since=None,
        until=None,
        with_attachments=False,
        without_attachments=False,
        no_picker=True,
        json=False,
    )

    ui = RecordingUI()
    env = CommandEnv(ui=ui)
    run_search_cli(args, env)

    assert ui.summaries, "expected inspect search to emit summary"
    title, lines = ui.summaries[0]
    assert title == "Search"
    assert any("Hits: 1" in line for line in lines)


def test_run_watch_cli_dispatches_based_on_provider(monkeypatch):
    markers: List[str] = []

    def recorder(_args, _env, provider):
        markers.append(provider.name)

    monkeypatch.setattr("polylogue.cli.watch._run_watch_sessions", recorder)

    run_watch_cli(SimpleNamespace(provider="codex"), CommandEnv(ui=RecordingUI()))
    run_watch_cli(SimpleNamespace(provider="claude-code"), CommandEnv(ui=RecordingUI()))

    assert markers == ["codex", "claude-code"], "watch CLI should delegate to provider-specific runners"


def test_watch_debounce_skips_fast_repeats(monkeypatch, tmp_path):
    from polylogue.cli import watch as watch_module

    base_dir = tmp_path / "sessions"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "first.jsonl").write_text("{}", encoding="utf-8")
    (base_dir / "second.jsonl").write_text("{}", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    events = [
        {("modified", str(base_dir / "first.jsonl"))},
        {("modified", str(base_dir / "second.jsonl"))},
    ]

    def fake_watch(*_args, **_kwargs):
        for change in events:
            yield change

    times = iter([0.0, 0.6, 0.8])
    monkeypatch.setattr(watch_module, "_watch_directory", fake_watch)
    monkeypatch.setattr(watch_module.time, "monotonic", lambda: next(times))

    calls: List[int] = []

    def fake_sync_fn(**_kwargs):
        calls.append(1)
        return LocalSyncResult(written=[], skipped=0, pruned=0, output_dir=out_dir)

    provider = LocalSyncProvider(
        name="codex",
        title="Codex",
        sync_fn=fake_sync_fn,
        default_base=base_dir,
        default_output=out_dir,
        list_sessions=lambda base: [base / "first.jsonl", base / "second.jsonl"],
        watch_banner="Watching Codex sessions",
        watch_log_title="Codex Watch",
        watch_suffixes=(".jsonl",),
    )

    env = CommandEnv(ui=RecordingUI())
    args = SimpleNamespace(
        provider="codex",
        base_dir=str(base_dir),
        out=str(out_dir),
        collapse_threshold=None,
        html_mode="off",
        debounce=0.5,
        once=False,
        diff=False,
        force=False,
        prune=False,
    )

    watch_module._run_watch_sessions(args, env, provider)

    assert len(calls) == 2  # initial sync + first update; second update debounced


def test_run_watch_sessions_once_triggers_sync(monkeypatch, tmp_path):
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
        registrar=None,
        **_kwargs,
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

    monkeypatch.setattr(watch_module, "_watch_directory", lambda *args, **kwargs: iter(()))

    records: List[dict] = []
    monkeypatch.setattr("polylogue.cli.sync.add_run", lambda data: records.append(data))

    ui = RecordingUI()
    args = SimpleNamespace(
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

    provider = LocalSyncProvider(
        name="codex",
        title="Codex",
        sync_fn=fake_sync_codex_sessions,
        default_base=Path(args.base_dir),
        default_output=Path(args.out),
        list_sessions=lambda base: [Path(args.base_dir) / "session.jsonl"],
        watch_banner="Watching Codex sessions",
        watch_log_title="Codex Watch",
        watch_suffixes=(".jsonl",),
    )

    watch_module._run_watch_sessions(args, CommandEnv(ui=ui), provider)

    assert calls, "expected sync_codex_sessions to be invoked"
    assert ui.banners and ui.banners[0][0] == "Watching Codex sessions"
    assert records and records[0]["cmd"] == "codex-watch"
