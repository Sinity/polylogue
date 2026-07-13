"""Raw session state mutation helpers."""

from __future__ import annotations

from datetime import datetime, timezone

import aiosqlite

from polylogue.core.enums import Origin, Provider, ValidationMode, ValidationStatus
from polylogue.storage.raw.models import RawSessionStateUpdate
from polylogue.storage.sqlite.connection import _build_source_scope_filter
from polylogue.storage.sqlite.raw_state_update import compile_raw_state_update

# raw_sessions carries a single ``origin`` column (#1743). Provider-token
# filters translate the token to its canonical origin value before matching.
RAW_ORIGIN_FILTER_SQL = "origin"


def origin_filter_value(token: str) -> str:
    """Validate and return a canonical ``origin`` value."""
    return Origin(token).value


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

    set_clauses, compiled_params = compile_raw_state_update(state, now_ms=_now_ms())
    if not set_clauses:
        if transaction_depth == 0:
            await conn.commit()
        return

    params = (*compiled_params, raw_id)
    await conn.execute(
        f"UPDATE raw_sessions SET {', '.join(set_clauses)} WHERE raw_id = ?",
        params,
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
    origin: str | None = None,
    source_names: list[str] | None = None,
    transaction_depth: int,
) -> int:
    where_clauses = ["(parsed_at_ms IS NOT NULL OR parse_error IS NOT NULL)"]
    params: list[str] = []
    if origin is not None:
        where_clauses.append(f"{RAW_ORIGIN_FILTER_SQL} = ?")
        params.append(origin_filter_value(origin))
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
    origin: str | None = None,
    source_names: list[str] | None = None,
    transaction_depth: int,
) -> int:
    where_clauses = ["(validated_at_ms IS NOT NULL OR validation_status IS NOT NULL OR validation_error IS NOT NULL)"]
    params: list[str] = []
    if origin is not None:
        where_clauses.append(f"{RAW_ORIGIN_FILTER_SQL} = ?")
        params.append(origin_filter_value(origin))
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
