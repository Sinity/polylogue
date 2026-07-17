"""Independent instant oracle and Hypothesis cases for temporal laws.

The oracle deliberately uses only the Python standard library.  It does not
call Polylogue timestamp parsers, canonicalizers, storage converters, or
surface builders, so a production normalization defect cannot teach the test
its expected answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from typing import TypeAlias

from hypothesis import strategies as st

TimestampWire: TypeAlias = str | int | float | datetime | None

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_SECOND = 1_000_000
_NUMERIC_WIRE = re.compile(r"[+-]?\d+(?:\.\d+)?\Z")
_OFFSET_MINUTES = (-720, -480, -300, -240, -60, 0, 60, 120, 330, 345, 480, 570, 720, 840)
_OFFSET_PAIRS = tuple((left, right) for left in _OFFSET_MINUTES for right in _OFFSET_MINUTES if left != right)


@dataclass(frozen=True, slots=True)
class EquivalentInstantCase:
    """Two distinct ISO wire values naming one standard instant."""

    first_wire: str
    second_wire: str
    epoch_microseconds: int


def standard_epoch_microseconds(value: TimestampWire) -> int | None:
    """Normalize a supported wire value to integer microseconds since Unix epoch.

    Naive ISO values follow Polylogue's declared provider-input rule and are
    interpreted as UTC.  Numeric values use ``Decimal`` and explicit half-even
    rounding, avoiding binary-float timestamp arithmetic in the oracle.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return _datetime_epoch_microseconds(value)
    if isinstance(value, int | float):
        return _decimal_epoch_microseconds(Decimal(str(value)))
    if _NUMERIC_WIRE.fullmatch(value):
        try:
            return _decimal_epoch_microseconds(Decimal(value))
        except InvalidOperation:
            return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _datetime_epoch_microseconds(parsed)


def standard_datetime(value: TimestampWire) -> datetime | None:
    """Return the oracle's canonical UTC datetime for ``value``."""

    epoch_microseconds = standard_epoch_microseconds(value)
    if epoch_microseconds is None:
        return None
    return _EPOCH + timedelta(microseconds=epoch_microseconds)


def wire_datetime(value: str) -> datetime:
    """Parse one ISO wire while preserving its declared offset representation."""

    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _datetime_epoch_microseconds(value: datetime) -> int:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    normalized = aware.astimezone(UTC)
    delta = normalized - _EPOCH
    return ((delta.days * 86_400 + delta.seconds) * _MICROSECONDS_PER_SECOND) + delta.microseconds


def _decimal_epoch_microseconds(value: Decimal) -> int:
    return int((value * _MICROSECONDS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_EVEN))


def _render_at_offset(value: datetime, offset_minutes: int) -> str:
    rendered = value.astimezone(timezone(timedelta(minutes=offset_minutes))).isoformat(timespec="microseconds")
    return rendered.replace("+00:00", "Z") if offset_minutes == 0 else rendered


@st.composite
def equivalent_instant_case_strategy(draw: st.DrawFn) -> EquivalentInstantCase:
    """Generate distinct offset renderings of one microsecond-precise instant."""

    naive = draw(
        st.datetimes(
            min_value=datetime(2001, 1, 1),
            max_value=datetime(2034, 12, 31, 23, 59, 59, 999999),
        )
    )
    value = naive.replace(tzinfo=UTC)
    first_offset, second_offset = draw(st.sampled_from(_OFFSET_PAIRS))
    return EquivalentInstantCase(
        first_wire=_render_at_offset(value, first_offset),
        second_wire=_render_at_offset(value, second_offset),
        epoch_microseconds=_datetime_epoch_microseconds(value),
    )


__all__ = [
    "EquivalentInstantCase",
    "TimestampWire",
    "equivalent_instant_case_strategy",
    "standard_datetime",
    "standard_epoch_microseconds",
    "wire_datetime",
]
