"""Raw conversation state mutation helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import aiosqlite

from polylogue.storage.backends.connection import _build_source_scope_filter
from polylogue.storage.raw_state_models import UNSET, RawConversationStateUpdate, _RawStateUnset
from polylogue.types import Provider, ValidationMode, ValidationStatus

EFFECTIVE_RAW_PROVIDER_SQL = "COALESCE(payload_provider, provider_name)"


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
    state: RawConversationStateUpdate,
    transaction_depth: int,
) -> None:
    if not state.has_values:
        if transaction_depth == 0:
            await conn.commit()
        return

    set_clauses: list[str] = []
    params: list[Any] = []

    if state.parsed_at is not UNSET:
        set_clauses.append("parsed_at = ?")
        params.append(state.parsed_at)
    if state.parse_error is not UNSET:
        set_clauses.append("parse_error = ?")
        params.append(state.parse_error[:2000] if isinstance(state.parse_error, str) else state.parse_error)
    if state.validation_status is not UNSET:
        set_clauses.append("validation_status = ?")
        params.append(coerce_status(state.validation_status))
    if state.validation_error is not UNSET:
        set_clauses.append("validation_error = ?")
        params.append(
            state.validation_error[:2000] if isinstance(state.validation_error, str) else state.validation_error
        )
    if state.validation_drift_count is not UNSET:
        drift_count = state.validation_drift_count
        set_clauses.append("validation_drift_count = ?")
        params.append(max(0, int(drift_count or 0)) if not isinstance(drift_count, _RawStateUnset) else 0)
    if state.validation_provider is not UNSET:
        set_clauses.append("validation_provider = ?")
        params.append(coerce_provider(state.validation_provider))
    if state.validation_mode is not UNSET:
        set_clauses.append("validation_mode = ?")
        params.append(coerce_mode(state.validation_mode))
    if state.payload_provider is not UNSET:
        set_clauses.append("payload_provider = COALESCE(?, payload_provider)")
        params.append(coerce_provider(state.payload_provider))
    if state.validation_status is not UNSET or state.validation_error is not UNSET:
        set_clauses.append("validated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
    if not set_clauses:
        if transaction_depth == 0:
            await conn.commit()
        return

    params.append(raw_id)
    await conn.execute(
        f"UPDATE raw_conversations SET {', '.join(set_clauses)} WHERE raw_id = ?",
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
        state = RawConversationStateUpdate(
            parsed_at=datetime.now(timezone.utc).isoformat(),
            parse_error=None,
            payload_provider=provider_token,
        )
    else:
        state = RawConversationStateUpdate(
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

    state = RawConversationStateUpdate(
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
    where_clauses = ["(parsed_at IS NOT NULL OR parse_error IS NOT NULL)"]
    params: list[str] = []
    if provider is not None:
        where_clauses.append(f"{EFFECTIVE_RAW_PROVIDER_SQL} = ?")
        params.append(provider)
    predicate, scope_params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    if predicate:
        where_clauses.append(predicate)
        params.extend(scope_params)
    cursor = await conn.execute(
        f"UPDATE raw_conversations SET parsed_at = NULL, parse_error = NULL WHERE {' AND '.join(where_clauses)}",
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
    where_clauses = ["(validated_at IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL)"]
    params: list[str] = []
    if provider is not None:
        where_clauses.append(f"{EFFECTIVE_RAW_PROVIDER_SQL} = ?")
        params.append(provider)
    predicate, scope_params = _build_source_scope_filter(
        source_names,
        source_column="source_name",
    )
    if predicate:
        where_clauses.append(predicate)
        params.extend(scope_params)
    cursor = await conn.execute(
        "UPDATE raw_conversations "
        "SET validated_at = NULL, validation_status = NULL, validation_error = NULL, "
        "validation_drift_count = NULL, validation_provider = NULL, validation_mode = NULL "
        f"WHERE {' AND '.join(where_clauses)}",
        tuple(params),
    )
    if transaction_depth == 0:
        await conn.commit()
    return cursor.rowcount


__all__ = [
    "EFFECTIVE_RAW_PROVIDER_SQL",
    "apply_raw_state_update",
    "coerce_mode",
    "coerce_provider",
    "coerce_status",
    "mark_raw_parsed",
    "mark_raw_validated",
    "reset_parse_status",
    "reset_validation_status",
]
