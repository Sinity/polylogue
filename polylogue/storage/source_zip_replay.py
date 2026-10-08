"""Content-authoritative replay of source ZIP acquisition units.

A recorded ``source_index`` is a hint, not an address. Members get rewritten,
reordered, and re-exported between acquisitions, so the element that sat at a
recorded position may now be a different, equally valid conversation. Every
value this module returns is the value whose content identity matches what was
recorded; the hint only decides which candidate is tried first.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from polylogue.config import Source
from polylogue.core.content_identity import ContentIdentityRefusal
from polylogue.core.enums import Origin, Provider
from polylogue.core.raw_coordinates import (
    MemberAddressingMode,
    read_captured_zip_coordinate_receipt,
)
from polylogue.core.sources import origin_provider_fiber
from polylogue.sources.decoder_zip import ZipEntryValidator
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    open_replayed_zip_unit,
    replay_zip_entry_acquisition_revisions,
)
from polylogue.storage.blob_store import BlobStore


@dataclass(frozen=True, slots=True)
class MemberCandidate:
    """One acquisition unit a container member can yield, held by identity.

    A reference is proven by digest, so a candidate keeps the unit's digests
    and never its bytes: a preserved member can be gigabytes, and one
    verification pass caches the candidates of every member it opens.
    """

    addressing_mode: MemberAddressingMode
    element_index: int | None
    #: The value identity acquisition records for this unit.
    content_identity: str
    #: SHA-256 of the unit's bytes, for rows recorded without a value identity.
    byte_identity: str
    size_bytes: int
    #: Reopens this unit's bytes as a stream, for a caller that must carry
    #: them (backup recovery); never part of the candidate's identity.
    open_payload: Callable[[], AbstractContextManager[IO[bytes]]] | None = field(
        default=None, compare=False, repr=False
    )

    @property
    def coordinate(self) -> tuple[str, int | None]:
        return (self.addressing_mode.value, self.element_index)


@dataclass(frozen=True, slots=True)
class MemberResolution:
    """The outcome of resolving one recorded reference against a member."""

    candidate: MemberCandidate | None
    outcome: str
    error: str | None


MemberCandidateCache = dict[str, tuple[MemberCandidate, ...]]


def resolve_member_candidate(
    candidates: Sequence[MemberCandidate],
    *,
    expected_digest: str | None,
    hint_mode: MemberAddressingMode,
    hint_index: int | None,
    expected_is_structural: bool = False,
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
            return MemberResolution(candidates[0], "sole_candidate", None)
        if len({candidate.content_identity for candidate in candidates}) == 1:
            return MemberResolution(candidates[0], "duplicate_observations", None)
        return MemberResolution(None, "ambiguous", "content_identity:unavailable")

    hinted = _hinted_candidate(candidates, hint_mode=hint_mode, hint_index=hint_index)
    if hinted is not None and _matches_expected(hinted, expected_digest, structural_only=expected_is_structural):
        return MemberResolution(hinted, "hint_verified", None)

    matching = [
        candidate
        for candidate in candidates
        if _matches_expected(candidate, expected_digest, structural_only=expected_is_structural)
    ]
    if not matching:
        return MemberResolution(None, "unmatched", "content_identity:unmatched")
    # Every match carries the recorded identity, so the proven value is the
    # same whichever occurrence is returned. Several occurrences are duplicate
    # observations of one logical item, not a choice between conversations.
    outcome = "resolved_by_content" if len(matching) == 1 else "duplicate_observations"
    return MemberResolution(matching[0], outcome, None)


def _matches_expected(candidate: MemberCandidate, expected_digest: str, *, structural_only: bool = False) -> bool:
    """Match the declared identity without weakening a structural claim.

    ``blob_hash`` is the only identity of a row whose coordinate was recorded
    without a content identity.  Once a row carries ``content_identity``,
    however, accepting a byte hash as an alternative would let a deliberately
    colliding/mutated row pass replay verification, so the caller marks
    structural identities explicitly.
    """
    if structural_only:
        return candidate.content_identity == expected_digest
    return candidate.content_identity == expected_digest or candidate.byte_identity == expected_digest


def _hinted_candidate(
    candidates: Sequence[MemberCandidate],
    *,
    hint_mode: MemberAddressingMode,
    hint_index: int | None,
) -> MemberCandidate | None:
    if hint_mode is MemberAddressingMode.WHOLE_MEMBER:
        return next(
            (item for item in candidates if item.addressing_mode is MemberAddressingMode.WHOLE_MEMBER),
            None,
        )
    if hint_index is None:
        return None
    return next(
        (
            item
            for item in candidates
            if item.addressing_mode is MemberAddressingMode.ELEMENT_OF_CONTAINER and item.element_index == hint_index
        ),
        None,
    )


def _unit_opener(
    zip_path: Path, context: ZipEntryReadContext, source_index: int | None
) -> Callable[[], AbstractContextManager[IO[bytes]]]:
    """Reopen one replayed unit from its container, holding nothing until asked."""

    @contextmanager
    def open_unit() -> Iterator[IO[bytes]]:
        with (
            zipfile.ZipFile(zip_path) as archive,
            open_replayed_zip_unit(archive, context, source_index=source_index) as stream,
        ):
            yield stream

    return open_unit


def zip_reacquired_unit(
    row: Mapping[str, object],
    *,
    source_path: str,
    zip_payload_cache: MemberCandidateCache,
    on_failure: Callable[[Exception], None] | None = None,
) -> tuple[MemberCandidate | None, str | None]:
    """Replay one ZIP member and return the recorded unit it still holds.

    The member streams through acquisition's own unit decisions, so proving
    a reference holds a few digests per unit whatever the member's size.
    """
    coordinate = _zip_coordinate(row)
    hint_index = coordinate[1] if coordinate is not None else None
    hint_mode = _recorded_addressing_mode(row)
    receipt = row.get("captured_coordinate")
    if receipt is not None:
        if not isinstance(receipt, str):
            raise ValueError("captured ZIP coordinate receipt must be text")
        captured = read_captured_zip_coordinate_receipt(receipt)
        if hint_mode is not None and hint_mode is not captured.addressing_mode:
            return None, "container_coordinate_mismatch"
        hint_mode = captured.addressing_mode
        if coordinate != (captured.entry_ordinal, captured.split_index):
            return None, "container_coordinate_mismatch"
        # Restoration relocates only the physical container and preserves this
        # exact recorded suffix. Colons in either component are not separators
        # to rediscover from the current filesystem.
        suffix = ":" + captured.member_name
        if not source_path.endswith(suffix):
            return None, "container_coordinate_mismatch"
        zip_path, member = Path(source_path[: -len(suffix)]), captured.member_name
    else:
        return None, "container_coordinate_missing"
    try:
        with zipfile.ZipFile(zip_path) as archive:
            central_directory = archive.infolist()
            assert coordinate is not None
            entry_ordinal = coordinate[0]
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
                    # Recover the provider from the origin only when that
                    # origin names exactly one provider; a non-injective
                    # origin (AI Studio/Drive) must not be guessed.
                    fiber = origin_provider_fiber(Origin.from_string(str(row.get("origin") or "")))
                    if len(fiber) != 1:
                        return None, "replay_provider_unrecorded"
                    provider = fiber[0]
                # Nothing is decompressed until acquisition's own admission
                # admits this exact entry; a cached candidate was admitted.
                if not _acquisition_admits(zip_path, central_directory, entry, provider):
                    return None, "container_member_rejected"
                context = ZipEntryReadContext(
                    source=Source(name=provider.value, path=zip_path.parent),
                    zip_path=zip_path,
                    entry=entry,
                    file_mtime=None,
                    provider_hint=provider,
                    blob_store=BlobStore(zip_path.parent / "blob"),
                )
                replayed: list[MemberCandidate] = []
                try:
                    for acquired in replay_zip_entry_acquisition_revisions(archive, context):
                        replayed.append(
                            MemberCandidate(
                                addressing_mode=acquired.addressing_mode,
                                element_index=acquired.source_index,
                                content_identity=acquired.content_identity,
                                byte_identity=acquired.revision,
                                size_bytes=acquired.size_bytes,
                                open_payload=_unit_opener(zip_path, context, acquired.source_index),
                            )
                        )
                except ContentIdentityRefusal:
                    # A refused element is the member's recorded gap; every
                    # element acquired beside it stays a replay candidate.
                    if not replayed:
                        raise
                candidates = tuple(replayed)
                zip_payload_cache[cache_key] = candidates
    except Exception as exc:
        if on_failure is not None:
            on_failure(exc)
        # Source replay is evidence, not a prerequisite for constructing
        # the backup. Any unreadable or unparseable container therefore
        # leaves this reference unproven and lets verification fail closed.
        return None, f"error:{exc}"
    structural_identity = _has_valid_digest(row.get("content_identity"))
    resolution = resolve_member_candidate(
        candidates,
        expected_digest=_expected_digest(row),
        hint_mode=hint_mode,
        hint_index=hint_index,
        expected_is_structural=structural_identity,
    )
    return resolution.candidate, resolution.error


def _acquisition_admits(
    zip_path: Path,
    central_directory: list[zipfile.ZipInfo],
    entry: zipfile.ZipInfo,
    provider: Provider,
) -> bool:
    """Replay acquisition's ZIP admission for one member of the whole archive.

    Acquisition runs ``ZipEntryValidator`` over the complete central
    directory, and admission is cumulative: the aggregate-size budget counts
    every admitted entry before this one. The member is admitted only if that
    same pass, in directory order, yields this exact ``ZipInfo``.
    """
    validator = ZipEntryValidator(provider, cursor_state=None, zip_path=zip_path)
    return any(admitted is entry for admitted in validator.filter_entries(central_directory))


def _expected_digest(row: Mapping[str, object]) -> str | None:
    value = row.get("content_identity")
    if isinstance(value, str) and _has_valid_digest(value):
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
    # A coordinate may be recorded without a reading (``content_identity``
    # NULL); the retained byte hash is then the only recorded identity.
    return blob_hash.lower()


def _has_valid_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


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
    return None


__all__ = [
    "MemberCandidate",
    "MemberCandidateCache",
    "MemberResolution",
    "resolve_member_candidate",
    "zip_reacquired_unit",
]
