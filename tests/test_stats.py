import json
from argparse import Namespace
from pathlib import Path

import pytest

from polylogue.commands import CommandEnv
from polylogue.ui import UI
from polylogue.cli import run_stats_cli


def test_run_stats_json(tmp_path, capsys):
    md_dir = tmp_path / "out"
    md_dir.mkdir()
    sample_dir = md_dir / "example"
    sample_dir.mkdir()
    sample = sample_dir / "conversation.md"
    sample.write_text(
        "---\n"
        "title: Example\n"
        "attachmentCount: 2\n"
        "attachmentBytes: 2048\n"
        "totalTokensApprox: 128\n"
        "totalWordsApprox: 96\n"
        "sourcePlatform: chatgpt\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    args = type("Args", (), {"dir": md_dir, "json": True, "since": None, "until": None})()

    env = CommandEnv(ui=UI(plain=True))
    run_stats_cli(args, env)
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["totals"]["files"] == 1
    assert payload["totals"]["attachments"] == 2
    assert payload["totals"]["attachmentBytes"] == 2048
    assert payload["totals"]["tokens"] == 128
    assert payload["totals"].get("words") == 96
    assert payload["providers"]["chatgpt"]["files"] == 1
    assert payload["providers"]["chatgpt"].get("words") == 96

    old = md_dir / "old.md"
    old.write_text(
        "---\n"
        "title: Old\n"
        "attachmentCount: 5\n"
        "attachmentBytes: 1024\n"
        "totalTokensApprox: 64\n"
        "totalWordsApprox: 48\n"
        "sourcePlatform: chatgpt\n"
        "sourceModifiedTime: 2020-01-01T00:00:00Z\n"
        "---\n",
        encoding="utf-8",
    )

    filtered_args = type(
        "Args",
        (),
        {"dir": md_dir, "json": True, "since": "2024-01-01", "until": None},
    )()
    run_stats_cli(filtered_args, env)
    filtered_payload = json.loads(capsys.readouterr().out)
    assert filtered_payload["totals"]["files"] == 1
    assert filtered_payload["totals"].get("words") == 96
    assert filtered_payload["filteredOut"] == 1


def test_run_stats_nested_provider_dirs(tmp_path, capsys):
    archive_dir = tmp_path / "archive"
    conversation_dir = archive_dir / "codex" / "example"
    conversation_dir.mkdir(parents=True)
    sample = conversation_dir / "conversation.md"
    sample.write_text(
        "---\n"
        "title: Nested Example\n"
        "attachmentCount: 1\n"
        "attachmentBytes: 512\n"
        "totalTokensApprox: 32\n"
        "totalWordsApprox: 24\n"
        "sourcePlatform: codex\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    args = Namespace(dir=archive_dir, json=True, since=None, until=None)
    env = CommandEnv(ui=UI(plain=True))
    run_stats_cli(args, env)

    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["files"] == 1
    assert payload["providers"]["codex"]["files"] == 1


def test_run_stats_defaults_to_all_roots(monkeypatch, tmp_path, capsys):
    render_dir = tmp_path / "render" / "chat1"
    render_dir.mkdir(parents=True)
    (render_dir / "conversation.md").write_text(
        "---\n"
        "title: Rendered\n"
        "attachmentCount: 1\n"
        "attachmentBytes: 1024\n"
        "totalTokensApprox: 10\n"
        "totalWordsApprox: 8\n"
        "sourcePlatform: chatgpt\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )
    codex_dir = tmp_path / "codex" / "session"
    codex_dir.mkdir(parents=True)
    (codex_dir / "conversation.md").write_text(
        "---\n"
        "title: Codex\n"
        "attachmentCount: 0\n"
        "totalTokensApprox: 5\n"
        "totalWordsApprox: 4\n"
        "sourcePlatform: codex\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("polylogue.cli.status.DEFAULT_OUTPUT_ROOTS", [render_dir.parent, codex_dir.parent])
    args = Namespace(dir=None, json=True, since=None, until=None)
    env = CommandEnv(ui=UI(plain=True))
    run_stats_cli(args, env)
    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["files"] == 2
    assert set(payload["providers"].keys()) == {"chatgpt", "codex"}
    assert set(payload["directories"]) == {str(render_dir.parent), str(codex_dir.parent)}


def test_run_stats_ignore_legacy(monkeypatch, tmp_path, capsys):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    convo_dir = archive_dir / "provider" / "slug"
    convo_dir.mkdir(parents=True)
    (convo_dir / "conversation.md").write_text("---\nsourcePlatform: provider\n---\n", encoding="utf-8")
    (archive_dir / "legacy.md").write_text("legacy", encoding="utf-8")

    args = Namespace(dir=archive_dir, json=True, since=None, until=None, ignore_legacy=True)
    env = CommandEnv(ui=UI(plain=True))

    run_stats_cli(args, env)

    payload = json.loads(capsys.readouterr().out)
    assert payload["totals"]["files"] == 1
    assert payload["providers"]["provider"]["files"] == 1


def test_run_stats_missing_dir_exits_nonzero_json(tmp_path, capsys):
    missing_dir = tmp_path / "missing"
    args = Namespace(dir=missing_dir, json=True, since=None, until=None)
    env = CommandEnv(ui=UI(plain=True))

    with pytest.raises(SystemExit) as excinfo:
        run_stats_cli(args, env)

    assert excinfo.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["directory"] == str(missing_dir)
    assert payload["totals"]["files"] == 0


def test_run_stats_missing_roots_exits_nonzero_plain(monkeypatch):
    monkeypatch.setattr("polylogue.cli.status.DEFAULT_OUTPUT_ROOTS", [])
    args = Namespace(dir=None, json=False, since=None, until=None)
    env = CommandEnv(ui=UI(plain=True))

    with pytest.raises(SystemExit) as excinfo:
        run_stats_cli(args, env)

    assert excinfo.value.code == 1
