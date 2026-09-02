"""Read-only replay of source ZIP acquisition units for evidence checks."""

from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path

from polylogue.config import Source
from polylogue.core.enums import Origin, Provider
from polylogue.core.sources import provider_from_origin
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    replay_zip_entry_acquisition_payloads,
)
from polylogue.storage.blob_store import BlobStore


def zip_reacquisition_payload(
    row: Mapping[str, object],
    *,
    source_path: str,
    zip_payload_cache: dict[str, dict[int, bytes]],
) -> tuple[bytes | None, str | None]:
    """Replay one ZIP member through the production acquisition splitter."""
    coordinate = _zip_coordinate(row)
    split_index = coordinate[1] if coordinate is not None else _legacy_split_index(row)
    zip_path_text, _separator, member = source_path.partition(":")
    if not zip_path_text or not member:
        return None, "container_coordinate_missing"
    zip_path = Path(zip_path_text)
    if not zip_path.exists():
        return None, "source_missing"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            if coordinate is None:
                central_directory = archive.infolist()
                matching = [
                    (ordinal, entry) for ordinal, entry in enumerate(central_directory) if entry.filename == member
                ]
                if len(matching) != 1:
                    return None, "ambiguous_container_member"
                entry_ordinal, entry = matching[0]
            else:
                entry_ordinal = coordinate[0]
                central_directory = archive.infolist()
                if entry_ordinal >= len(central_directory):
                    return None, "container_coordinate_mismatch"
                entry = central_directory[entry_ordinal]
                if entry.filename != member:
                    return None, "container_coordinate_mismatch"
            cache_key = f"{source_path}\0{entry_ordinal}"
            payloads = zip_payload_cache.get(cache_key)
            if payloads is None:
                provider = Provider.from_string(str(row.get("capture_mode") or ""))
                if provider is Provider.UNKNOWN:
                    provider = provider_from_origin(Origin.from_string(str(row.get("origin") or "")))
                context = ZipEntryReadContext(
                    source=Source(name=provider.value, path=zip_path.parent),
                    zip_path=zip_path,
                    entry=entry,
                    file_mtime=None,
                    provider_hint=provider,
                    blob_store=BlobStore(zip_path.parent / "blob"),
                )
                payloads = {}
                for acquired in replay_zip_entry_acquisition_payloads(archive, context):
                    acquired_index = acquired.source_index if acquired.source_index is not None else 0
                    payloads[acquired_index] = acquired.payload_bytes
                zip_payload_cache[cache_key] = payloads
    except Exception as exc:
        # Source replay is evidence, not a prerequisite for constructing
        # the backup. Any unreadable or unparseable container therefore
        # leaves this reference unproven and lets verification fail closed.
        return None, f"error:{exc}"
    expected_hash = str(row.get("blob_hash") or "")
    if expected_hash:
        # Prefer the recorded bytes and compare values under the provider's
        # structural contract. This admits harmless JSON numeric changes such
        # as 1 versus 1.0 while retaining type distinctions elsewhere.
        try:
            expected_payload = BlobStore(zip_path.parent / "blob").read_all(expected_hash)
        except (OSError, ValueError):
            expected_payload = None
        if expected_payload is not None:
            structural = tuple(
                payload for payload in payloads.values() if _structurally_equal(expected_payload, payload)
            )
            if len(structural) == 1:
                return structural[0], None
            if len(structural) > 1:
                return None, "content_identity:ambiguous"
        # Canonical replay bytes are still an exact identity fallback.
        matches = tuple(
            payload for payload in payloads.values() if hashlib.sha256(payload).hexdigest() == expected_hash
        )
        unique = tuple(dict.fromkeys(matches))
        if len(unique) == 1:
            return unique[0], None
        return None, "content_identity:ambiguous" if len(unique) > 1 else "content_identity:unmatched"
    if split_index is None:
        return (next(iter(payloads.values())), None) if len(payloads) == 1 else (None, "content_identity:missing")
    return None, "content_identity:missing"


def _structurally_equal(left: bytes, right: bytes) -> bool:
    try:
        left_value, right_value = json.loads(left), json.loads(right)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return left == right
    return _json_value_equal(left_value, right_value)


def _json_value_equal(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_value_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_value_equal(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


def _zip_coordinate(row: Mapping[str, object]) -> tuple[int, int] | None:
    if row.get("coordinate_format") == "zip-v2":
        entry_ordinal = row.get("entry_ordinal")
        split_index = row.get("split_index")
        if isinstance(entry_ordinal, (int, str)) and isinstance(split_index, (int, str)):
            return int(entry_ordinal), int(split_index)
    source_path = row.get("source_path")
    source_index = row.get("source_index")
    raw_id = str(row.get("raw_id") or row.get("ref_id") or "")
    blob_hash = str(row.get("blob_hash") or "")
    if (
        not isinstance(source_path, str)
        or not source_path
        or ":" not in source_path
        or not isinstance(source_index, (int, str))
    ):
        return None
    from polylogue.core.raw_coordinates import zip_member_identity_coordinate

    return zip_member_identity_coordinate(
        raw_id=raw_id,
        source_path=source_path,
        source_index=int(source_index),
        blob_hash=blob_hash,
    )


def _legacy_split_index(row: Mapping[str, object]) -> int | None:
    source_index = row.get("source_index")
    if not isinstance(source_index, (int, str)):
        return None
    try:
        return int(source_index)
    except ValueError:
        return None


__all__ = ["zip_reacquisition_payload"]
