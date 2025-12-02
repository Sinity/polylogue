import os
from pathlib import Path

from polylogue.local_sync import compute_prune_paths


def test_prune_moves_to_trash_then_deletes(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    keep = output_dir / "keep"
    keep.mkdir()
    remove_me = output_dir / "old"
    remove_me.mkdir()
    (remove_me / "file.txt").write_text("x", encoding="utf-8")

    wanted = {"keep"}
    paths = list(compute_prune_paths(output_dir, wanted))
    assert remove_me in paths

    # simulate prune loop from local_sync using trash
    trash_dir = output_dir / ".trash"
    trash_dir.mkdir()
    for path in paths:
        target_name = f"{path.name}.123"
        trash_path = trash_dir / target_name
        path.rename(trash_path)
        import shutil

        shutil.rmtree(trash_path)

    assert not remove_me.exists()
    assert keep.exists()
