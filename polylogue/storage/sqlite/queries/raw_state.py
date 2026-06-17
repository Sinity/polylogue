"""Raw session state mutation helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite

from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.core.sources import origin_from_provider
from polylogue.storage.raw.models import UNSET, RawSessionStateUpdate, _RawStateUnset
from polylogue.storage.sqlite.archive_tiers.write import _timestamp_ms
from polylogue.storage.sqlite.connection import _build_source_scope_filter

# raw_sessions carries a single ``origin`` column (#1743). Provider-token
# filters translate the token to its canonical origin value before matching.
RAW_ORIGIN_FILTER_SQL = "origin"


def origin_filter_value(token: str) -> str:
    """Translate a provider-wire token into its canonical ``origin`` value."""
    return origin_from_provider(Provider.from_string(token)).value


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def coerce_provider(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, Provider):
        return value.value
    return Provider.from_string(str(value)).value


def coerce_status(value: object) -> ValidationStatus:
    if isinstance(value, ValidationStatus):
        return value
    return ValidationStatus.from_string(str(value))


def coerce_mode(value: object) -> ValidationMode:
    if isinstance(value, ValidationMode):
        return value
    return ValidationMode.from_string(str(value))


async def apply_raw_state_update(
    conn: aiosqlite.Connection,
    raw_id: str,
    *,
    state: RawSessionStateUpdate,
    transaction_depth: int,
) -> None:
    if not state.has_values:
        if transaction_depth == 0:
            await conn.commit()
        return

    set_clauses: list[str] = []
    params: list[object] = []

    if state.parsed_at is not UNSET:
        set_clauses.append("parsed_at_ms = ?")
        params.append(_timestamp_ms(state.parsed_at) if isinstance(state.parsed_at, str) else None)
    if state.parse_error is not UNSET:
        set_clauses.append("parse_error = ?")
        params.append(state.parse_error[:2000] if isinstance(state.parse_error, str) else state.parse_error)
    if state.validation_status is not UNSET:
        raw_status = state.validation_status
        set_clauses.append("validation_status = ?")
        params.append(coerce_status(raw_status).value if raw_status is not None else None)
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
        raw_mode = state.validation_mode
        set_clauses.append("validation_mode = ?")
        params.append(coerce_mode(raw_mode).value if raw_mode is not None else None)
    # raw_sessions collapses payload/validation provider identity onto a single
    # ``origin`` column (#1743). Either update folds into origin; payload wins.
    if state.payload_provider is not UNSET or state.validation_provider is not UNSET:
        origin_value: str | None = None
        if state.payload_provider is not UNSET:
            payload_token = coerce_provider(state.payload_provider)
            if payload_token:
                origin_value = origin_filter_value(payload_token)
        if origin_value is None and state.validation_provider is not UNSET:
            validation_token = coerce_provider(state.validation_provider)
            if validation_token:
                origin_value = origin_filter_value(validation_token)
        set_clauses.append("origin = COALESCE(?, origin)")
        params.append(origin_value)
    if state.detection_warnings is not UNSET:
        set_clauses.append("detection_warnings_json = ?")
        warnings = state.detection_warnings
        params.append((warnings[:2000] if isinstance(warnings, str) else warnings) or "[]")
    if state.validation_status is not UNSET or state.validation_error is not UNSET:
        set_clauses.append("validated_at_ms = ?")
        params.append(_now_ms())
    if not set_clauses:
        if transaction_depth == 0:
            await conn.commit()
        return

    params.append(raw_id)
    await conn.execute(
        f"UPDATE raw_sessions SET {', '.join(set_clauses)} WHERE raw_id = ?",
        tuple(params),
    )
    if transaction_depth == 0:
        await conn.commit()


async def mark_raw_parsed(
    conn: aiosqlite.Connection,
    raw_id: str,
    *,
    error: str | None = None,
    payload_provider: str | Provider | None = None,
    transaction_depth: int,
) -> None:
    provider_token = coerce_provider(payload_provider)
    if error is None:
        state = RawSessionStateUpdate(
            parsed_at=datetime.now(timezone.utc).isoformat(),
            parse_error=None,
            payload_provider=provider_token,
        )
    else:
        state = RawSessionStateUpdate(
            parse_error=error[:2000],
            payload_provider=provider_token,
        )
    await apply_raw_state_update(
        conn,
        raw_id,
        state=state,
        transaction_depth=transaction_depth,
    )


async def mark_raw_validated(
    conn: aiosqlite.Connection,
    raw_id: str,
    *,
    status: ValidationStatus | str,
    error: str | None = None,
    drift_count: int = 0,
    provider: Provider | str | None = None,
    mode: ValidationMode | str | None = None,
    payload_provider: Provider | str | None = None,
    transaction_depth: int,
) -> None:
    try:
        validation_status = coerce_status(status)
    except ValueError as exc:
        raise ValueError(f"Invalid validation status: {status}") from exc

    validation_mode: ValidationMode | None
    if mode is not None:
        try:
            validation_mode = coerce_mode(mode)
        except ValueError as exc:
            raise ValueError(f"Invalid validation mode: {mode}") from exc
    else:
        validation_mode = None

    state = RawSessionStateUpdate(
        validation_status=validation_status,
        validation_error=(error[:2000] if error else None),
        validation_drift_count=drift_count,
        validation_provider=coerce_provider(provider),
        validation_mode=validation_mode,
        payload_provider=coerce_provider(payload_provider),
    )
    await apply_raw_state_update(
        conn,
        raw_id,
        state=state,
        transaction_depth=transaction_depth,
    )


async def reset_parse_status(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    source_names: list[str] | None = None,
    transaction_depth: int,
) -> int:
    where_clauses = ["(parsed_at_ms IS NOT NULL OR parse_error IS NOT NULL)"]
    params: list[str] = []
    if provider is not None:
        where_clauses.append(f"{RAW_ORIGIN_FILTER_SQL} = ?")
        params.append(origin_filter_value(provider))
    predicate, scope_params = _build_source_scope_filter(
        source_names,
        source_column="origin",
    )
    if predicate:
        where_clauses.append(predicate)
        params.extend(scope_params)
    cursor = await conn.execute(
        f"UPDATE raw_sessions SET parsed_at_ms = NULL, parse_error = NULL WHERE {' AND '.join(where_clauses)}",
        tuple(params),
    )
    if transaction_depth == 0:
        await conn.commit()
    return cursor.rowcount


async def reset_validation_status(
    conn: aiosqlite.Connection,
    *,
    provider: str | None = None,
    source_names: list[str] | None = None,
    transaction_depth: int,
) -> int:
    where_clauses = ["(validated_at_ms IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL)"]
    params: list[str] = []
    if provider is not None:
        where_clauses.append(f"{RAW_ORIGIN_FILTER_SQL} = ?")
        params.append(origin_filter_value(provider))
    predicate, scope_params = _build_source_scope_filter(
        source_names,
        source_column="origin",
    )
    if predicate:
        where_clauses.append(predicate)
        params.extend(scope_params)
    cursor = await conn.execute(
        "UPDATE raw_sessions "
        "SET validated_at_ms = NULL, validation_status = NULL, validation_error = NULL, "
        "validation_drift_count = 0, validation_mode = NULL "
        f"WHERE {' AND '.join(where_clauses)}",
        tuple(params),
    )
    if transaction_depth == 0:
        await conn.commit()
    return cursor.rowcount


__all__ = [
    "RAW_ORIGIN_FILTER_SQL",
    "apply_raw_state_update",
    "coerce_mode",
    "coerce_provider",
    "coerce_status",
    "origin_filter_value",
    "mark_raw_parsed",
    "mark_raw_validated",
    "reset_parse_status",
    "reset_validation_status",
]
