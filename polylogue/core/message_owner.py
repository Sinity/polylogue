"""Private parser-to-writer coordinates for message ownership."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MessageOwnerCoordinate:
    """Private linkage between a parsed attachment and its message.

    ``stable_key`` carries reorder-stable provider evidence when the parser
    has it. ``position`` and ``variant_index`` are the complete transport
    coordinate used as the fail-closed fallback. The coordinate is excluded
    from public parser serialization and is never a provider message id.
    """

    stable_key: str | None = None
    position: int | None = None
    variant_index: int = 0

    def __post_init__(self) -> None:
        if self.position is not None and self.position < 0:
            raise ValueError("message owner position cannot be negative")
        if self.variant_index < 0:
            raise ValueError("message owner variant_index cannot be negative")
        if self.stable_key == "":
            raise ValueError("message owner stable_key cannot be empty")

    @property
    def physical_key(self) -> tuple[int, int] | None:
        """Return the full position/variant coordinate when it is present."""
        if self.position is None:
            return None
        return self.position, self.variant_index


class MessageOwnerAmbiguityError(ValueError):
    """Raised when an attachment owner cannot be resolved without guessing."""


__all__ = ["MessageOwnerAmbiguityError", "MessageOwnerCoordinate"]
