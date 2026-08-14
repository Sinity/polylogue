"""Ordering authority for durable raw parse and validation transitions."""

from __future__ import annotations

from typing import Literal, TypeAlias

RawStateAuthority: TypeAlias = Literal["parse", "validation", "ambiguous"]


def raw_state_authority(
    parsed_at_ms: int | None,
    validated_at_ms: int | None,
) -> RawStateAuthority:
    """Return the proven terminal transition, never assigning equal times.

    New writes make opposing transitions strictly monotonic. Existing rows can
    predate that invariant, so an equal non-null pair remains explicitly
    indeterminate rather than being silently attributed to either stage.

    A pair with both values ``None`` reports ``"validation"``. These legacy
    rows carry no ordering evidence, so callers retain the stored validation
    verdict instead of inventing parse authority.
    """
    if parsed_at_ms is None:
        return "validation"
    if validated_at_ms is None:
        return "parse"
    if parsed_at_ms > validated_at_ms:
        return "parse"
    if validated_at_ms > parsed_at_ms:
        return "validation"
    return "ambiguous"
