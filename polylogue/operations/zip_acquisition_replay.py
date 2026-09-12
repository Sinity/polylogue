"""Content-authoritative replay of source ZIP acquisition units.

A recorded ``source_index`` is a hint, not an address. Members get rewritten,
reordered, and re-exported between acquisitions, so the element that sat at a
recorded position may now be a different, equally valid conversation. Every
value this module returns is the value whose content identity matches what was
recorded; the hint only decides which candidate is tried first.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from polylogue.config import Source
from polylogue.core.content_identity import structural_content_identity
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDecodeError
from polylogue.core.json import loads as json_loads
from polylogue.core.raw_coordinates import MemberAddressingMode
from polylogue.core.sources import provider_from_origin
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    replay_zip_entry_acquisition_payloads,
)
from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class MemberCandidate:
    """One acquisition unit a container member can yield."""

    addressing_mode: MemberAddressingMode
    element_index: int | None
    payload_bytes: bytes

    @property
    def coordinate(self) -> tuple[str, int | None]:
        return (self.addressing_mode.value, self.element_index)

    @property
    def content_identity(self) -> str:
        """Identify this unit by decoded content, not by serialization.

        A unit that does not decode as JSON has no structure to compare, so
        its bytes are its identity.
        """
        try:
            return structural_content_identity(json_loads(self.payload_bytes))
        except (JSONDecodeError, ValueError, TypeError):
            return hashlib.sha256(self.payload_bytes).hexdigest()

    @property
    def byte_identity(self) -> str:
        """Legacy raw-byte identity retained for pre-migration rows."""
        return hashlib.sha256(self.payload_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class MemberResolution:
    """The outcome of resolving one recorded reference against a member."""

    payload_bytes: bytes | None
    outcome: str
    error: str | None


MemberCandidateCache = dict[str, tuple[MemberCandidate, ...]]


def resolve_member_candidate(
    candidates: Sequence[MemberCandidate],
    *,
    expected_digest: str | None,
    hint_mode: MemberAddressingMode | None,
    hint_index: int | None,
) -> MemberResolution:
    """Resolve one recorded reference by content identity, hint first.

    The hint selects the candidate that is *checked* first and never the
    candidate that is returned: a hinted value is accepted only once its
    content proves it is the recorded one.
    """
    if not candidates:
        return MemberResolution(None, "unmatched", "member_yields_no_payload")
    if expected_digest is None:
        # Without a recorded digest a hint cannot be verified. Several
        # candidates are still one logical item when they are the same
        # content; only genuinely different content is ambiguous.
        if len(candidates) == 1:
            return MemberResolution(candidates[0].payload_bytes, "sole_candidate", None)
        if len({candidate.content_identity for candidate in candidates}) == 1:
            return MemberResolution(candidates[0].payload_bytes, "duplicate_observations", None)
        return MemberResolution(None, "ambiguous", "content_identity:unavailable")

    hinted = _hinted_candidate(candidates, hint_mode=hint_mode, hint_index=hint_index)
    if hinted is not None and _matches_expected(hinted, expected_digest):
        return MemberResolution(hinted.payload_bytes, "hint_verified", None)

    matching = [candidate for candidate in candidates if _matches_expected(candidate, expected_digest)]
    if not matching:
        return MemberResolution(None, "unmatched", "content_identity:unmatched")
    # Every match carries identical bytes, so the recovered value is the same
    # whichever occurrence is returned. Several occurrences are duplicate
    # observations of one logical item, not a choice between conversations.
    outcome = "resolved_by_content" if len(matching) == 1 else "duplicate_observations"
    return MemberResolution(matching[0].payload_bytes, outcome, None)


def _digest(candidate: MemberCandidate) -> str:
    return candidate.content_identity


def _matches_expected(candidate: MemberCandidate, expected_digest: str) -> bool:
    return candidate.content_identity == expected_digest or candidate.byte_identity == expected_digest


def _hinted_candidate(
    candidates: Sequence[MemberCandidate],
    *,
    hint_mode: MemberAddressingMode | None,
    hint_index: int | None,
) -> MemberCandidate | None:
    if hint_mode is MemberAddressingMode.WHOLE_MEMBER:
        return next(
            (item for item in candidates if item.addressing_mode is MemberAddressingMode.WHOLE_MEMBER),
            None,
        )
    if hint_index is None:
        return None
    element = next(
        (
            item
            for item in candidates
            if item.addressing_mode is MemberAddressingMode.ELEMENT_OF_CONTAINER and item.element_index == hint_index
        ),
        None,
    )
    if element is not None or hint_mode is MemberAddressingMode.ELEMENT_OF_CONTAINER:
        return element
    # A row written before the addressing mode was recorded stamped a whole
    # member as element 0; try that reading too rather than reporting the
    # member unrecoverable.
    if hint_index != 0:
        return None
    return next(
        (item for item in candidates if item.addressing_mode is MemberAddressingMode.WHOLE_MEMBER),
        None,
    )


def zip_reacquisition_payload(
    row: Mapping[str, object],
    *,
    source_path: str,
    zip_payload_cache: MemberCandidateCache,
) -> tuple[bytes | None, str | None]:
    """Replay one ZIP member and return the recorded value it still holds."""
    coordinate = _zip_coordinate(row)
    hint_index = coordinate[1] if coordinate is not None else _legacy_split_index(row)
    hint_mode = _recorded_addressing_mode(row)
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
            candidates = zip_payload_cache.get(cache_key)
            if candidates is None:
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
                candidates = tuple(
                    MemberCandidate(
                        addressing_mode=acquired.addressing_mode,
                        element_index=acquired.source_index,
                        payload_bytes=acquired.payload_bytes,
                    )
                    for acquired in replay_zip_entry_acquisition_payloads(archive, context)
                )
                zip_payload_cache[cache_key] = candidates
    except Exception as exc:
        # Source replay is evidence, not a prerequisite for constructing
        # the backup. Any unreadable or unparseable container therefore
        # leaves this reference unproven and lets verification fail closed.
        return None, f"error:{exc}"
    resolution = resolve_member_candidate(
        candidates,
        expected_digest=_expected_digest(row),
        hint_mode=hint_mode,
        hint_index=hint_index,
    )
    return resolution.payload_bytes, resolution.error


def _expected_digest(row: Mapping[str, object]) -> str | None:
    for key in ("content_identity", "content_digest"):
        value = row.get(key)
        if isinstance(value, str) and len(value) == 64:
            try:
                bytes.fromhex(value)
            except ValueError:
                continue
            return value.lower()
    blob_hash = row.get("blob_hash")
    if isinstance(blob_hash, (bytes, bytearray)) and len(blob_hash) == 32:
        return bytes(blob_hash).hex()
    if not isinstance(blob_hash, str) or len(blob_hash) != 64:
        return None
    try:
        bytes.fromhex(blob_hash)
    except ValueError:
        return None
    # Legacy rows predate the durable structural digest.  Their raw byte hash
    # remains a safe compatibility identity (and is never used for migrated
    # rows, which carry ``content_identity``).
    return blob_hash.lower()


def _recorded_addressing_mode(row: Mapping[str, object]) -> MemberAddressingMode | None:
    value = row.get("addressing_mode")
    if not isinstance(value, str) or not value:
        return None
    try:
        return MemberAddressingMode(value)
    except ValueError:
        return None


def _zip_coordinate(row: Mapping[str, object]) -> tuple[int, int] | None:
    if row.get("coordinate_format") == "zip-v2":
        entry_ordinal = row.get("entry_ordinal")
        split_index = row.get("split_index")
        if isinstance(entry_ordinal, (int, str)) and isinstance(split_index, (int, str)):
            return int(entry_ordinal), int(split_index)
    source_path = row.get("source_path")
    source_index = row.get("source_index")
    raw_id = str(row.get("raw_id") or row.get("ref_id") or "")
    blob_hash = row.get("blob_hash")
    blob_hash_hex = bytes(blob_hash).hex() if isinstance(blob_hash, (bytes, bytearray)) else str(blob_hash or "")
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
        blob_hash=blob_hash_hex,
    )


def _legacy_split_index(row: Mapping[str, object]) -> int | None:
    source_index = row.get("source_index")
    if not isinstance(source_index, (int, str)):
        return None
    try:
        return int(source_index)
    except ValueError:
        return None


__all__ = [
    "MemberCandidate",
    "MemberCandidateCache",
    "MemberResolution",
    "resolve_member_candidate",
    "zip_reacquisition_payload",
]
