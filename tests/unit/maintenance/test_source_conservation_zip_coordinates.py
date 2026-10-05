"""Source conservation resolves ZIP members only through captured coordinates.

A ZIP member's presence is proved by its captured container/member receipt.
Suffix spellings (``zip:member``, ``zip!member``) are literal paths: no reader
infers a member namespace from a source path.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from polylogue.core.raw_coordinates import (
    CapturedZipMemberCoordinate,
    MemberAddressingMode,
    captured_zip_coordinate_receipt,
)
from polylogue.maintenance.source_conservation import _source_presence

_HASH = "0" * 64


def _zip(path: Path, member: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "{}")
    return path


def _receipt(container: Path, member: str) -> str:
    return captured_zip_coordinate_receipt(
        CapturedZipMemberCoordinate(
            canonical_container=str(container),
            declared_container=str(container),
            member_name=member,
            entry_ordinal=0,
            split_index=0,
            addressing_mode=MemberAddressingMode.WHOLE_MEMBER,
            container_blob_hash=_HASH,
            decoder_fingerprint=_HASH,
        )
    )


def test_captured_member_of_a_present_zip_is_conserved(tmp_path: Path) -> None:
    bundle = _zip(tmp_path / "export.zip", "conversations/a.json")
    assert _source_presence(
        tmp_path, f"{bundle}:conversations/a.json", {}, captured_coordinate=_receipt(bundle, "conversations/a.json")
    )


def test_captured_coordinate_naming_a_missing_member_is_not_conserved(tmp_path: Path) -> None:
    bundle = _zip(tmp_path / "export.zip", "conversations/a.json")
    assert not _source_presence(
        tmp_path,
        f"{bundle}:conversations/gone.json",
        {},
        captured_coordinate=_receipt(bundle, "conversations/gone.json"),
    )


def test_captured_coordinate_of_a_deleted_zip_is_not_conserved(tmp_path: Path) -> None:
    deleted = tmp_path / "deleted.zip"
    assert not _source_presence(
        tmp_path, f"{deleted}:conversations/a.json", {}, captured_coordinate=_receipt(deleted, "conversations/a.json")
    )


def test_loose_file_with_a_colon_in_its_name_is_resolved_literally(tmp_path: Path) -> None:
    loose = tmp_path / "notes:draft.json"
    loose.write_text("{}")
    assert _source_presence(tmp_path, str(loose), {})
    assert not _source_presence(tmp_path, str(tmp_path / "other:draft.json"), {})


def test_suffix_spellings_of_a_present_member_are_literal_paths(tmp_path: Path) -> None:
    """Fails if a reader infers a ZIP member from a ``:`` or ``!`` path suffix."""
    bundle = _zip(tmp_path / "export.zip", "conversations/a.json")
    for separator in (":", "!"):
        assert _source_presence(tmp_path, f"{bundle}{separator}conversations/a.json", {}) is False


def test_coordinate_delimiters_inside_container_and_member_names_are_conserved(tmp_path: Path) -> None:
    """The captured receipt carries container and member exactly; delimiters in names never split them."""
    member = "nested/item!part:revision.json"
    bundle = _zip(tmp_path / "export!copy:revision.zip", member)
    assert _source_presence(tmp_path, f"{bundle}:{member}", {}, captured_coordinate=_receipt(bundle, member)) is True
    assert (
        _source_presence(
            tmp_path, f"{bundle}:nested/gone.json", {}, captured_coordinate=_receipt(bundle, "nested/gone.json")
        )
        is False
    )
