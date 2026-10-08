"""The per-check active Index path agrees with full archive location resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.storage.archive_identity import (
    ACTIVE_POINTER_FILENAME,
    ArchiveLocationError,
    active_index_configured_path,
    resolve_active_index_path,
)

_DURABLE = ("source.db", "user.db", "audit.db")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _plain_root(tmp_path: Path) -> Path:
    root = tmp_path / "archive"
    for name in (*_DURABLE, "index.db"):
        _touch(root / name)
    return root


def _farm_root(tmp_path: Path) -> tuple[Path, Path]:
    real = tmp_path / "real"
    root = tmp_path / "farm"
    root.mkdir()
    for name in _DURABLE:
        (root / name).symlink_to(_touch(real / name))
    generation = _touch(real / ".index-generations" / "gen-a" / "index.db")
    (root / "index.db").symlink_to(generation)
    return root, generation


def _scenario(tmp_path: Path, name: str) -> Path:
    if name == "no_pointer":
        return _plain_root(tmp_path)
    if name == "file_pointer_inside":
        root = _plain_root(tmp_path)
        generation = _touch(root / ".index-generations" / "gen-a" / "index.db")
        (root / ACTIVE_POINTER_FILENAME).write_text(f"{generation}\n", encoding="utf-8")
        return root
    if name == "link_pointer_inside":
        root = _plain_root(tmp_path)
        generation = _touch(root / ".index-generations" / "gen-a" / "index.db")
        (root / ACTIVE_POINTER_FILENAME).symlink_to(generation)
        return root
    if name == "symlink_farm":
        root, generation = _farm_root(tmp_path)
        (root / ACTIVE_POINTER_FILENAME).write_text(str(generation), encoding="utf-8")
        return root
    if name == "copied_archive":
        # A copy keeps real durable tiers while its index still points away.
        root = _plain_root(tmp_path)
        generation = _touch(tmp_path / "elsewhere" / "index.db")
        (root / "index.db").unlink()
        (root / "index.db").symlink_to(generation)
        (root / ACTIVE_POINTER_FILENAME).write_text(str(generation), encoding="utf-8")
        return root
    if name == "relative_pointer":
        root = _plain_root(tmp_path)
        (root / ACTIVE_POINTER_FILENAME).write_text("gen-a/index.db", encoding="utf-8")
        return root
    raise AssertionError(name)


@pytest.mark.parametrize(
    "scenario",
    ["no_pointer", "file_pointer_inside", "link_pointer_inside", "symlink_farm", "copied_archive", "relative_pointer"],
)
def test_light_active_index_path_matches_full_resolution(tmp_path: Path, scenario: str) -> None:
    root = _scenario(tmp_path, scenario)
    try:
        expected: Path | type[ArchiveLocationError] = resolve_active_index_path(root)
    except ArchiveLocationError:
        expected = ArchiveLocationError
    if expected is ArchiveLocationError:
        with pytest.raises(ArchiveLocationError):
            active_index_configured_path(root)
        assert scenario in {"copied_archive", "relative_pointer"}
    else:
        assert active_index_configured_path(root) == expected
        assert scenario not in {"copied_archive", "relative_pointer"}
