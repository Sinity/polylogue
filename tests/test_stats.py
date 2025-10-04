from pathlib import Path
import json

from chatmd.commands import CommandEnv
from chatmd.ui import UI
from gmd import run_stats_cli


def test_run_stats_json(tmp_path, capsys):
    md_dir = tmp_path / "out"
    md_dir.mkdir()
    sample = md_dir / "example.md"
    sample.write_text(
        "---\n"
        "title: Example\n"
        "attachmentCount: 2\n"
        "attachmentBytes: 2048\n"
        "totalTokensApprox: 128\n"
        "sourcePlatform: chatgpt\n"
        "---\n"
        "body\n",
        encoding="utf-8",
    )

    args = type("Args", (), {"dir": md_dir, "json": True})()

    env = CommandEnv(ui=UI(plain=True))
    run_stats_cli(args, env)
    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload["totals"]["files"] == 1
    assert payload["totals"]["attachments"] == 2
    assert payload["totals"]["attachmentBytes"] == 2048
    assert payload["totals"]["tokens"] == 128
    assert payload["providers"]["chatgpt"]["files"] == 1
