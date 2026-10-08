"""Canonical SQLite compiler for typed raw-session state mutations."""

from __future__ import annotations

import json
from collections.abc import Callable

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.timestamps import to_epoch_ms
from polylogue.storage.raw.models import UNSET, RawSessionStateUpdate, _RawStateUnset
from polylogue.storage.sqlite.archive_tiers.common import require_vocabulary


def compile_raw_state_update(
    state: RawSessionStateUpdate,
    *,
    now_ms: int,
    literal: Callable[[object], tuple[str, tuple[object, ...]]],
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Compile one typed mutation for either SQLite connection adapter."""
    set_clauses: list[str] = []
    params: list[object] = []

    def operand(value: object) -> str:
        expression, values = literal(value)
        params.extend(values)
        return expression

    parsed_at_ms = to_epoch_ms(state.parsed_at, numeric_unit="seconds") if isinstance(state.parsed_at, str) else None
    validation_transition = state.validation_status is not UNSET or state.validation_error is not UNSET
    if state.parsed_at is not UNSET:
        if parsed_at_ms is None:
            if isinstance(state.parsed_at, str):
                raise ValueError(f"parsed_at must be a valid timestamp, got {state.parsed_at!r}")
            set_clauses.append(f"parsed_at_ms = {operand(None)}")
        elif validation_transition:
            # SQLite evaluates every SET expression from the old row.  A
            # combined update records validation first and parse second, so
            # advance parse by two from either old transition (and one from
            # this validation clock) to preserve that authority ordering even
            # when wall time is equal or moves backward.
            set_clauses.append(
                f"parsed_at_ms = MAX({operand(parsed_at_ms)}, {operand(now_ms)} + 1, "
                f"COALESCE(parsed_at_ms + 2, {operand(parsed_at_ms)}), "
                f"COALESCE(validated_at_ms + 2, {operand(parsed_at_ms)}))"
            )
        else:
            set_clauses.append(
                f"parsed_at_ms = MAX({operand(parsed_at_ms)}, COALESCE(parsed_at_ms + 1, {operand(parsed_at_ms)}), "
                f"COALESCE(validated_at_ms + 1, {operand(parsed_at_ms)}))"
            )
    if state.parse_error is not UNSET:
        set_clauses.append(f"parse_error = {operand(state.parse_error)}")
    if state.validation_status is not UNSET:
        status = state.validation_status
        status_value = (
            require_vocabulary(status, ValidationStatus, field="validation_status") if status is not None else None
        )
        set_clauses.append(f"validation_status = {operand(status_value)}")
    if state.validation_error is not UNSET:
        set_clauses.append(f"validation_error = {operand(state.validation_error)}")
    if state.validation_drift_count is not UNSET:
        drift_count = state.validation_drift_count
        drift_value = max(0, int(drift_count or 0)) if not isinstance(drift_count, _RawStateUnset) else 0
        set_clauses.append(f"validation_drift_count = {operand(drift_value)}")
    if state.validation_mode is not UNSET:
        mode = state.validation_mode
        mode_value = require_vocabulary(mode, ValidationMode, field="validation_mode") if mode is not None else None
        set_clauses.append(f"validation_mode = {operand(mode_value)}")
    if state.payload_provider is not UNSET or state.validation_provider is not UNSET:
        payload_provider = (
            require_vocabulary(state.payload_provider, Provider, field="payload_provider")
            if state.payload_provider is not UNSET and state.payload_provider is not None
            else None
        )
        validation_provider = (
            require_vocabulary(state.validation_provider, Provider, field="validation_provider")
            if state.validation_provider is not UNSET and state.validation_provider is not None
            else None
        )
        set_clauses.append(
            f"detected_provider = COALESCE({operand(payload_provider or validation_provider)}, detected_provider)"
        )
    if state.detection_warnings is not UNSET:
        warnings = state.detection_warnings
        warnings_json = json.dumps([warnings]) if isinstance(warnings, str) and warnings else "[]"
        set_clauses.append(f"detection_warnings_json = {operand(warnings_json)}")
    if validation_transition:
        set_clauses.append(
            f"validated_at_ms = MAX({operand(now_ms)}, COALESCE(validated_at_ms + 1, {operand(now_ms)}), "
            f"COALESCE(parsed_at_ms + 1, {operand(now_ms)}))"
        )
    return tuple(set_clauses), tuple(params)


def raw_state_parameter(value: object) -> tuple[str, tuple[object, ...]]:
    """Bind an ordinary typed mutation without changing its declared value."""
    return "?", (value,)


__all__ = ["compile_raw_state_update", "raw_state_parameter"]
