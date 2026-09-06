"""Acquisition refuses a foreign archive's own directories as a source root.

238 raw artifacts in the operator's archive carry source paths under a
development archive's drive cache
(``/realm/tmp/polylogue-dev/xdg/polylogue/drive-cache/gemini/``) -- one
archive's material re-admitted into another as if newly captured.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.sources.source_acquisition import iter_source_raw_data
from polylogue.sources.source_root_admission import SourceRootRefusedError
from polylogue.storage.blob_store import BlobStore

_SESSION_LINE = {
    "sessionId": "capture-1",
    "parentUuid": None,
    "type": "user",
    "message": {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    "uuid": "capture-1-m0",
    "timestamp": "2026-07-20T00:00:00Z",
}


def _write_session(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session.jsonl"
    path.write_bytes(json.dumps(_SESSION_LINE).encode() + b"\n")
    return path


def _archive(root: Path) -> Path:
    """A directory carrying the markers that identify an archive root."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "source.db").write_bytes(b"")
    return root


def _acquire(source_root: Path, destination: Path) -> list[object]:
    store = BlobStore(destination / "blob")
    return list(iter_source_raw_data(Source(name="claude-code", path=source_root), blob_store=store))


def test_acquisition_refuses_a_foreign_archive_drive_cache(tmp_path: Path) -> None:
    """The observed contamination shape: a development archive's drive cache
    read as a source into the operator's archive.

    Mutation: drop the ``refuse_non_capture_source_root`` call from
    ``iter_source_raw_data``. The development archive's cache is admitted and
    its bytes reach the destination blob store."""
    dev_archive = _archive(tmp_path / "polylogue-dev" / "xdg" / "polylogue")
    cache = dev_archive / "drive-cache" / "gemini"
    _write_session(cache)
    destination = _archive(tmp_path / "live")

    with pytest.raises(SourceRootRefusedError) as refusal:
        _acquire(cache, destination)

    assert str(dev_archive) in str(refusal.value)
    assert not (destination / "blob").exists() or not any((destination / "blob").rglob("*"))


def test_acquisition_refuses_a_foreign_archive_even_with_nothing_to_walk(tmp_path: Path) -> None:
    """The refusal precedes the walk. Mutation: move the guard back below
    ``_setup_source_walk``'s ``if walk is None: return``. An empty foreign
    archive root then returns silently instead of refusing."""
    dev_archive = _archive(tmp_path / "polylogue-dev" / "xdg" / "polylogue")
    empty = dev_archive / "drive-cache" / "gemini"
    empty.mkdir(parents=True)

    with pytest.raises(SourceRootRefusedError):
        _acquire(empty, _archive(tmp_path / "live"))


def test_acquisition_refuses_a_foreign_archive_root_itself(tmp_path: Path) -> None:
    """A source root that IS another archive is refused, not only one below it."""
    lane_archive = _archive(tmp_path / "lane" / "xdg-data" / "polylogue")
    _write_session(lane_archive / "inbox")

    with pytest.raises(SourceRootRefusedError):
        _acquire(lane_archive, _archive(tmp_path / "live"))


def test_acquisition_admits_the_destination_archives_own_spool(tmp_path: Path) -> None:
    """The guard is not blanket: an archive's own capture spool is the live
    ingest route, so it stays admissible."""
    destination = _archive(tmp_path / "live")
    spool = destination / "browser-capture" / "claude-code"
    _write_session(spool)

    records = _acquire(spool, destination)

    assert len(records) == 1


def test_acquisition_admits_a_capture_directory_outside_any_archive(tmp_path: Path) -> None:
    """A provider's own capture directory is unaffected by the refusal."""
    capture = tmp_path / "home" / ".claude" / "projects" / "some-project"
    _write_session(capture)

    records = _acquire(capture, _archive(tmp_path / "live"))

    assert len(records) == 1


def test_acquisition_admits_a_symlinked_alias_of_the_destination_archive(tmp_path: Path) -> None:
    """One archive is reachable under several roots -- the operator's XDG data
    home symlinks its tiers at the archive on the NVMe. Mutation: compare only
    directory paths in ``_same_archive``. The alias reads as a foreign archive
    and its drive cache stops being acquirable."""
    destination = _archive(tmp_path / "live")
    alias = tmp_path / "xdg-data" / "polylogue"
    alias.mkdir(parents=True)
    (alias / "source.db").symlink_to(destination / "source.db")
    cache = alias / "drive-cache" / "gemini"
    _write_session(cache)

    records = _acquire(cache, destination)

    assert len(records) == 1
