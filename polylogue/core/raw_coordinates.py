"""Stable coordinates for raw payloads acquired from container members."""

from __future__ import annotations

from hashlib import sha256
from math import isqrt

_ZIP_MEMBER_RAW_ID_DOMAIN = b"polylogue:zip-member-raw:v2\0"


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


def zip_member_identity_coordinate(
    *,
    raw_id: str,
    source_path: str,
    source_index: int,
    blob_hash: str,
) -> tuple[int, int] | None:
    """Decode a v2 raw identity, rejecting legacy or unrelated coordinates."""
    entry_ordinal, split_index = zip_member_source_coordinate(source_index)
    expected_raw_id = zip_member_raw_id(
        source_path=source_path,
        entry_ordinal=entry_ordinal,
        split_index=split_index,
        blob_hash=blob_hash,
    )
    return (entry_ordinal, split_index) if raw_id == expected_raw_id else None


__all__ = [
    "zip_member_identity_coordinate",
    "zip_member_raw_id",
    "zip_member_source_coordinate",
    "zip_member_source_index",
]
