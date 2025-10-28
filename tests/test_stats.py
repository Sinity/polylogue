from pathlib import Path
import json

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
