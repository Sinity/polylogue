from pathlib import Path

from polylogue.cli.maintain import run_maintain_cli
from polylogue.commands import CommandEnv
from polylogue.ui import create_ui


class Args:
    maintain_cmd = "restore"
    src: Path
    dest: Path
    force: bool = False
    json: bool = False
    max_disk = None


def test_maintain_restore(tmp_path: Path):
    src = tmp_path / "snapshot"
    dest = tmp_path / "restore"
    src.mkdir()
    (src / "file.txt").write_text("hello", encoding="utf-8")

    args = Args()
    args.src = src
    args.dest = dest

    env = CommandEnv(ui=create_ui(plain=True))
    run_maintain_cli(args, env)

    restored = dest / "file.txt"
    assert restored.exists()
    assert restored.read_text(encoding="utf-8") == "hello"
