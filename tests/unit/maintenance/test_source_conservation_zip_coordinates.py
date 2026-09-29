"""Source conservation resolves both recorded ZIP member coordinate forms."""

from __future__ import annotations

import zipfile
from pathlib import Path

from polylogue.maintenance.source_conservation import _source_exists


def _zip(path: Path, member: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "{}")
    return path


def test_colon_member_coordinate_of_a_present_zip_is_conserved(tmp_path: Path) -> None:
    """Fails if only the ``!`` form is parsed: the ZIP readers record ``zip:member``."""
    bundle = _zip(tmp_path / "export.zip", "conversations/a.json")
    assert _source_exists(tmp_path, f"{bundle}:conversations/a.json")


def test_colon_coordinate_naming_a_missing_member_is_not_conserved(tmp_path: Path) -> None:
    bundle = _zip(tmp_path / "export.zip", "conversations/a.json")
    assert not _source_exists(tmp_path, f"{bundle}:conversations/gone.json")


def test_colon_coordinate_of_a_deleted_zip_is_not_conserved(tmp_path: Path) -> None:
    assert not _source_exists(tmp_path, f"{tmp_path / 'deleted.zip'}:conversations/a.json")


def test_loose_file_with_a_colon_in_its_name_is_resolved_literally(tmp_path: Path) -> None:
    loose = tmp_path / "notes:draft.json"
    loose.write_text("{}")
    assert _source_exists(tmp_path, str(loose))
    assert not _source_exists(tmp_path, str(tmp_path / "other:draft.json"))


def test_bang_member_coordinate_still_resolves(tmp_path: Path) -> None:
    bundle = _zip(tmp_path / "export.zip", "conversations/a.json")
    assert _source_exists(tmp_path, f"{bundle}!conversations/a.json")
    assert not _source_exists(tmp_path, f"{bundle}!conversations/gone.json")
