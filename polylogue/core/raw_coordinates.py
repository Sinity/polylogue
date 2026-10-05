"""Stable coordinates for raw payloads acquired from container members."""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isqrt
from pathlib import Path

from polylogue.core.enums import PolylogueStrEnum

_ZIP_MEMBER_RAW_ID_DOMAIN = b"polylogue:zip-member-raw:v2\0"


class MemberAddressingMode(PolylogueStrEnum):
    """How one acquired raw payload is addressed inside its container member.

    An element is one session-bearing value inside a member that holds several;
    a whole member is the member document itself. The two are different
    addresses for different content, and a member that holds one document has
    no element 0 -- reading it as one is how a positional consumer returns a
    valid but unrelated conversation.
    """

    ELEMENT_OF_CONTAINER = "element_of_container"
    WHOLE_MEMBER = "whole_member"


@dataclass(frozen=True, slots=True)
class CapturedZipMemberCoordinate:
    """Member evidence carried from the opened container to raw admission."""

    canonical_container: str
    declared_container: str
    member_name: str
    entry_ordinal: int
    split_index: int
    addressing_mode: MemberAddressingMode
    container_blob_hash: str
    decoder_fingerprint: str
    profile_namespace: str | None = None

    def __post_init__(self) -> None:
        for value in (self.container_blob_hash, self.decoder_fingerprint):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("captured ZIP input and decoder require SHA-256 identities")
        if not Path(self.canonical_container).is_absolute() or not Path(self.declared_container).is_absolute():
            raise ValueError("captured ZIP containers require absolute physical and declared coordinates")
        if not self.member_name:
            raise ValueError("captured ZIP member requires its exact name")
        zip_member_source_index(entry_ordinal=self.entry_ordinal, split_index=self.split_index)
        if self.addressing_mode is MemberAddressingMode.WHOLE_MEMBER and self.split_index:
            raise ValueError("a preserved ZIP member has no element index")
        if self.profile_namespace is not None and not Path(self.profile_namespace).is_absolute():
            raise ValueError("captured ZIP profile namespace must be absolute")

    @property
    def source_index(self) -> int:
        return zip_member_source_index(entry_ordinal=self.entry_ordinal, split_index=self.split_index)

    @property
    def declared_member(self) -> str:
        return f"{self.declared_container}:{self.member_name}"

    @property
    def canonical_member(self) -> str:
        return f"{self.canonical_container}:{self.member_name}"


def captured_zip_coordinate_receipt(coordinate: CapturedZipMemberCoordinate) -> str:
    """Serialize the exact acquisition-bound coordinate without filesystem reads."""
    return json.dumps(asdict(coordinate), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def read_captured_zip_coordinate_receipt(receipt: str) -> CapturedZipMemberCoordinate:
    """Read a proved coordinate; malformed evidence never falls back to paths."""
    value = json.loads(receipt)
    fields = {
        "canonical_container",
        "declared_container",
        "member_name",
        "entry_ordinal",
        "split_index",
        "addressing_mode",
        "container_blob_hash",
        "decoder_fingerprint",
        "profile_namespace",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid captured ZIP coordinate receipt")
    if not all(
        isinstance(value[field], str)
        for field in (
            "canonical_container",
            "declared_container",
            "member_name",
            "addressing_mode",
            "container_blob_hash",
            "decoder_fingerprint",
        )
    ) or not all(type(value[field]) is int for field in ("entry_ordinal", "split_index")):
        raise ValueError("invalid captured ZIP coordinate receipt fields")
    if value["profile_namespace"] is not None and not isinstance(value["profile_namespace"], str):
        raise ValueError("invalid captured ZIP profile namespace")
    return CapturedZipMemberCoordinate(
        value["canonical_container"],
        value["declared_container"],
        value["member_name"],
        value["entry_ordinal"],
        value["split_index"],
        MemberAddressingMode(value["addressing_mode"]),
        value["container_blob_hash"],
        value["decoder_fingerprint"],
        value["profile_namespace"],
    )


def captured_zip_member_raw_id(coordinate: CapturedZipMemberCoordinate, blob_hash: str) -> str:
    """Identify new ZIP intake from captured physical and semantic evidence."""
    digest = sha256()
    digest.update(b"polylogue:zip-member-raw:v4\0")
    for value in (
        coordinate.container_blob_hash,
        coordinate.decoder_fingerprint,
        coordinate.canonical_container,
        coordinate.declared_container,
        coordinate.member_name,
        coordinate.addressing_mode.value,
        str(coordinate.entry_ordinal),
        str(coordinate.split_index),
        coordinate.profile_namespace or "",
    ):
        digest.update(value.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    digest.update(bytes.fromhex(blob_hash))
    return digest.hexdigest()


def zip_member_record_coordinate(
    *,
    entry_ordinal: int,
    split_index: int,
    addressing_mode: MemberAddressingMode,
) -> str:
    """Serialize the existing exact central-entry and split membership address."""
    zip_member_source_index(entry_ordinal=entry_ordinal, split_index=split_index)
    if addressing_mode is MemberAddressingMode.WHOLE_MEMBER and split_index:
        raise ValueError("a preserved ZIP member has no element index")
    return json.dumps(["zip-v2", entry_ordinal, split_index, addressing_mode.value], separators=(",", ":"))


def zip_member_source_index(*, entry_ordinal: int, split_index: int) -> int:
    """Encode a ZIP entry ordinal and within-entry split index losslessly."""
    if entry_ordinal < 0 or split_index < 0:
        raise ValueError("ZIP entry ordinal and split index must be non-negative")
    diagonal = entry_ordinal + split_index
    return diagonal * (diagonal + 1) // 2 + split_index


def zip_member_source_coordinate(source_index: int) -> tuple[int, int]:
    """Recover the independent entry ordinal and split index from storage."""
    if source_index < 0:
        raise ValueError("ZIP member source index must be non-negative")
    diagonal = (isqrt(8 * source_index + 1) - 1) // 2
    diagonal_start = diagonal * (diagonal + 1) // 2
    split_index = source_index - diagonal_start
    entry_ordinal = diagonal - split_index
    return entry_ordinal, split_index


def zip_member_raw_id(
    *,
    source_path: str,
    entry_ordinal: int,
    split_index: int,
    blob_hash: str,
) -> str:
    """Identify one ZIP coordinate without giving up blob-level deduplication."""
    digest = sha256()
    digest.update(_ZIP_MEMBER_RAW_ID_DOMAIN)
    digest.update(source_path.encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    digest.update(str(entry_ordinal).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(split_index).encode("utf-8"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(blob_hash))
    return digest.hexdigest()


def zip_member_coordinate_candidates(source_path: str, *, separator: str = ":") -> Iterator[tuple[Path, str]]:
    """Yield lexical coordinate boundaries without claiming a container is readable.

    Shortest container first preserves acquisition's real-ZIP selection law.
    The observer must distinguish unreadable candidates from proven non-ZIPs;
    lexical spelling alone cannot select an arbitrary removed container.
    """
    if separator not in {":", "!"}:
        raise ValueError("ZIP member separator must be colon or exclamation mark")
    start = 0
    while (separator_at := source_path.find(separator, start)) != -1:
        start = separator_at + 1
        if separator_at == 0 or separator_at == len(source_path) - 1:
            continue
        yield Path(source_path[:separator_at]), source_path[start:]


def zip_member_coordinate(source_path: str) -> tuple[Path, str] | None:
    """Split a recorded ``<container>:<member>`` coordinate at its real ZIP.

    A loose file may legally contain a colon, so the literal path wins when it
    exists, and a missing path is read as a member coordinate only when a
    prefix is a real ZIP file. An existing prefix directory or plain file
    proves nothing. The container path itself may hold colons (a Windows
    drive, a legal POSIX filename), so every colon is tried as the separator,
    shortest container first.
    """
    if Path(source_path).exists():
        return None
    for container_path, member in zip_member_coordinate_candidates(source_path):
        if container_path.is_file() and zipfile.is_zipfile(container_path):
            return container_path, member
    return None


_ZIP_MEMBER_SEPARATOR = re.compile(r"\.zip:", re.IGNORECASE)


def split_zip_member_text(source_path: str) -> tuple[str, str] | None:
    """Split ``<container>:<member>`` where the container may not exist here.

    A container present on disk is located by :func:`zip_member_coordinate`.
    A relocated or removed one is split lexically after its ``.zip`` suffix,
    so a colon earlier in the container path (a Windows drive, a legal POSIX
    filename) is not taken as the separator. A loose file that exists at the
    literal path is never a member coordinate, whatever its name holds.
    """
    located = zip_member_coordinate(source_path)
    if located is not None:
        container, member = located
        return source_path[: len(source_path) - len(member) - 1], member
    if Path(source_path).exists():
        return None
    match = _ZIP_MEMBER_SEPARATOR.search(source_path)
    if match is None or match.end() == len(source_path):
        return None
    return source_path[: match.end() - 1], source_path[match.end() :]


def zip_member_container(source_path: str) -> Path | None:
    """The ZIP container a recorded ``<container>:<member>`` coordinate names."""
    coordinate = zip_member_coordinate(source_path)
    return coordinate[0] if coordinate is not None else None


__all__ = [
    "CapturedZipMemberCoordinate",
    "captured_zip_coordinate_receipt",
    "read_captured_zip_coordinate_receipt",
    "captured_zip_member_raw_id",
    "zip_member_container",
    "zip_member_coordinate",
    "zip_member_coordinate_candidates",
    "split_zip_member_text",
    "MemberAddressingMode",
    "zip_member_raw_id",
    "zip_member_source_coordinate",
    "zip_member_source_index",
]
