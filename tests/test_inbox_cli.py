import json
from pathlib import Path
from types import SimpleNamespace

from polylogue.cli.inbox import run_inbox_cli
from polylogue.commands import CommandEnv
from polylogue.ui import create_ui


def test_inbox_cli_lists_and_quarantines(tmp_path: Path, capsys) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()

    good = inbox / "export1"
    good.mkdir()
    (good / "conversations.json").write_text(json.dumps({"conversations": [{"mapping": {}}]}), encoding="utf-8")

    bad = inbox / "bad.zip"
    bad.write_bytes(b"junk")

    ignored = inbox / "ignored.zip"
    ignored.write_bytes(b"junk")
    (inbox / ".polylogueignore").write_text("ignored.zip\n", encoding="utf-8")

    args = SimpleNamespace(
        providers="chatgpt,claude",
        dir=inbox,
        quarantine=True,
        quarantine_dir=None,
        json=True,
    )
    env = CommandEnv(ui=create_ui(plain=True))

    run_inbox_cli(args, env)

    output = json.loads(capsys.readouterr().out)
    providers = {entry["provider"] for entry in output.get("entries", [])}
    assert "chatgpt" in providers
    assert output.get("quarantined"), "expected bad.zip to be quarantined"
    assert output.get("ignoredByRule", 0) == 1
    assert output.get("malformed", 0) == 1
    assert output.get("malformedByReason", {}).get("not-a-zip") == 1
    assert output.get("totals", {}).get("pending", 0) >= 1
    assert (inbox / "quarantine" / "bad.zip").exists()
