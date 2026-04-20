"""Typed raw-conversation processing state and update payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, field_validator

from polylogue.types import Provider, ValidationMode, ValidationStatus


class RawConversationState(BaseModel):
    raw_id: str
    source_name: str | None = None
    source_path: str | None = None
    parsed_at: str | None = None
    parse_error: str | None = None
    payload_provider: Provider | None = None
    validation_status: ValidationStatus | None = None
    validation_provider: Provider | None = None

    @field_validator("payload_provider", "validation_provider", mode="before")
    @classmethod
    def coerce_optional_provider(cls, v: object) -> Provider | None:
        if v is None:
            return None
        if isinstance(v, Provider):
            return v
        return Provider.from_string(str(v))

    @field_validator("validation_status", mode="before")
    @classmethod
    def coerce_state_validation_status(cls, v: object) -> ValidationStatus | None:
        if v is None:
            return None
        return ValidationStatus.from_string(str(v))


class _RawStateUnset:
    """Sentinel for update fields that should remain unchanged."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _RawStateUnset()


def _coerce_update_provider(value: Provider | str | None | _RawStateUnset) -> Provider | None | _RawStateUnset:
    if value is UNSET:
        return UNSET
    if value is None:
        return None
    if isinstance(value, Provider):
        return value
    assert isinstance(value, str)
    return Provider.from_string(value)


def _coerce_update_status(
    value: ValidationStatus | str | None | _RawStateUnset,
) -> ValidationStatus | None | _RawStateUnset:
    if value is UNSET:
        return UNSET
    if value is None:
        return None
    if isinstance(value, ValidationStatus):
        return value
    assert isinstance(value, str)
    return ValidationStatus.from_string(value)


def _coerce_update_mode(value: ValidationMode | str | None | _RawStateUnset) -> ValidationMode | None | _RawStateUnset:
    if value is UNSET:
        return UNSET
    if value is None:
        return None
    if isinstance(value, ValidationMode):
        return value
    assert isinstance(value, str)
    return ValidationMode.from_string(value)


@dataclass(frozen=True)
class RawConversationStateUpdate:
    """Typed raw-state mutation payload used by update_raw_state calls."""

    parsed_at: str | None | _RawStateUnset = UNSET
    parse_error: str | None | _RawStateUnset = UNSET
    payload_provider: Provider | str | None | _RawStateUnset = UNSET
    validation_status: ValidationStatus | str | None | _RawStateUnset = UNSET
    validation_error: str | None | _RawStateUnset = UNSET
    validation_drift_count: int | None | _RawStateUnset = UNSET
    validation_provider: Provider | str | None | _RawStateUnset = UNSET
    validation_mode: ValidationMode | str | None | _RawStateUnset = UNSET

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_provider", _coerce_update_provider(self.payload_provider))
        object.__setattr__(self, "validation_status", _coerce_update_status(self.validation_status))
        object.__setattr__(self, "validation_provider", _coerce_update_provider(self.validation_provider))
        object.__setattr__(self, "validation_mode", _coerce_update_mode(self.validation_mode))
        drift_count = self.validation_drift_count
        if isinstance(drift_count, int):
            object.__setattr__(self, "validation_drift_count", max(0, drift_count))

    @property
    def has_values(self) -> bool:
        """Return whether any field is explicitly provided."""
        return any(
            value is not UNSET
            for value in (
                self.parsed_at,
                self.parse_error,
                self.payload_provider,
                self.validation_status,
                self.validation_error,
                self.validation_drift_count,
                self.validation_provider,
                self.validation_mode,
            )
        )


__all__ = [
    "RawConversationState",
    "RawConversationStateUpdate",
    "UNSET",
    "_RawStateUnset",
]
