"""Canonical SQLite compiler for typed raw-session state mutations."""

from __future__ import annotations

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.sources import origin_from_provider
from polylogue.storage.raw.models import UNSET, RawSessionStateUpdate, _RawStateUnset
from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms


def compile_raw_state_update(
    state: RawSessionStateUpdate,
    *,
    now_ms: int,
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Compile one typed mutation for either SQLite connection adapter."""
    set_clauses: list[str] = []
    params: list[object] = []
    if state.parsed_at is not UNSET:
        set_clauses.append("parsed_at_ms = ?")
        params.append(_timestamp_ms(state.parsed_at) if isinstance(state.parsed_at, str) else None)
    if state.parse_error is not UNSET:
        set_clauses.append("parse_error = ?")
        params.append(state.parse_error[:2000] if isinstance(state.parse_error, str) else state.parse_error)
    if state.validation_status is not UNSET:
        status = state.validation_status
        set_clauses.append("validation_status = ?")
        params.append(status.value if isinstance(status, ValidationStatus) else None)
    if state.validation_error is not UNSET:
        set_clauses.append("validation_error = ?")
        params.append(
            state.validation_error[:2000] if isinstance(state.validation_error, str) else state.validation_error
        )
    if state.validation_drift_count is not UNSET:
        drift_count = state.validation_drift_count
        set_clauses.append("validation_drift_count = ?")
        params.append(max(0, int(drift_count or 0)) if not isinstance(drift_count, _RawStateUnset) else 0)
    if state.validation_mode is not UNSET:
        mode = state.validation_mode
        set_clauses.append("validation_mode = ?")
        params.append(mode.value if isinstance(mode, ValidationMode) else None)
    if state.payload_provider is not UNSET or state.validation_provider is not UNSET:
        provider: Provider | None = None
        if isinstance(state.payload_provider, Provider):
            provider = state.payload_provider
        elif isinstance(state.validation_provider, Provider):
            provider = state.validation_provider
        set_clauses.append("origin = COALESCE(?, origin)")
        params.append(origin_from_provider(provider).value if provider is not None else None)
    if state.detection_warnings is not UNSET:
        warnings = state.detection_warnings
        set_clauses.append("detection_warnings_json = ?")
        params.append((warnings[:2000] if isinstance(warnings, str) else warnings) or "[]")
    if state.validation_status is not UNSET or state.validation_error is not UNSET:
        set_clauses.append("validated_at_ms = ?")
        params.append(now_ms)
    return tuple(set_clauses), tuple(params)


__all__ = ["compile_raw_state_update"]
