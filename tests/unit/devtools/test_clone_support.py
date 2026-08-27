from __future__ import annotations

from pathlib import Path

import pytest

from devtools.clone_support import CloneSupportError, reflink_clone


def test_reflink_clone_rejects_symlink_sources_and_sqlite_sidecars(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    source.write_bytes(b"snapshot")
    source.with_name("source.db-wal").write_bytes(b"uncheckpointed")

    with pytest.raises(CloneSupportError, match="sidecars"):
        reflink_clone(source, tmp_path / "clone.db")

    source.with_name("source.db-wal").unlink()
    link = tmp_path / "link.db"
    link.symlink_to(source)
    with pytest.raises(CloneSupportError, match="regular"):
        reflink_clone(link, tmp_path / "link-clone.db")


def test_reflink_clone_rejects_existing_destination_and_authenticates_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    source.write_bytes(b"snapshot")
    destination = tmp_path / "clone.db"

    reflink_clone(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert not source.samefile(destination)

    with pytest.raises(CloneSupportError, match="already exists"):
        reflink_clone(source, destination)
