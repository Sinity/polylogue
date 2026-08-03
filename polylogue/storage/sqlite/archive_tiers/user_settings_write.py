"""Durable read/write helpers for the ``user_settings`` table.

``user_settings`` (migration ``004_user_settings.sql``) predates any runtime
caller (polylogue-at44): DDL and migration existed, nothing read or wrote it.
This module is the liveness slice only -- a typed key registry with get/set/
list helpers, deliberately narrow so the table cannot silently become a
free-form global key-value junk drawer (guardrail recorded on polylogue-at44).
The full scope x actor x override resolver design belongs to the w8db epic;
this module owns exactly one settings surface: a closed registry of setting
keys, each with its own validator, one row per key.

Writer module: user.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from polylogue.core.json import JSONValue

#: Closed registry of allowed setting keys. Adding a new setting means adding
#: a validator here, not accepting an arbitrary caller-supplied key --
#: keeping ``user_settings`` a typed registry rather than a junk drawer.
SETTING_KEY_SUBSCRIPTION_TIER = "subscription_tier"


def _validate_subscription_tier(value: JSONValue) -> None:
    from polylogue.archive.semantic.subscription_pricing import SUBSCRIPTION_TIERS

    if not isinstance(value, str) or value not in SUBSCRIPTION_TIERS:
        choices = ", ".join(sorted(SUBSCRIPTION_TIERS))
        raise ValueError(f"subscription_tier must be one of: {choices}")


_SETTING_VALIDATORS: dict[str, Callable[[JSONValue], None]] = {
    SETTING_KEY_SUBSCRIPTION_TIER: _validate_subscription_tier,
}


def known_setting_keys() -> frozenset[str]:
    """Return the closed set of setting keys this registry accepts."""

    return frozenset(_SETTING_VALIDATORS)


def _validate_setting(setting_key: str, value: JSONValue) -> None:
    validator = _SETTING_VALIDATORS.get(setting_key)
    if validator is None:
        choices = ", ".join(sorted(_SETTING_VALIDATORS))
        raise ValueError(f"unknown setting key {setting_key!r}; known keys: {choices}")
    validator(value)


@dataclass(frozen=True, slots=True)
class ArchiveUserSettingEnvelope:
    setting_key: str
    value: JSONValue
    updated_at_ms: int
    author_ref: str


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def set_user_setting(
    conn: sqlite3.Connection,
    setting_key: str,
    value: JSONValue,
    *,
    author_ref: str = "user:local",
    now_ms: int | None = None,
) -> ArchiveUserSettingEnvelope:
    """Insert-or-update one typed setting row.

    Rejects unknown keys and validator-rejected values -- there is no
    untyped write path into ``user_settings``.
    """

    _validate_setting(setting_key, value)
    timestamp = now_ms if now_ms is not None else _now_ms()
    value_json = json.dumps(value, sort_keys=True)
    conn.execute(
        """
        INSERT INTO user_settings (setting_key, value_json, updated_at_ms, author_ref)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(setting_key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at_ms = excluded.updated_at_ms,
            author_ref = excluded.author_ref
        """,
        (setting_key, value_json, timestamp, author_ref),
    )
    envelope = get_user_setting(conn, setting_key)
    if envelope is None:
        raise RuntimeError("user_settings insert did not round-trip")
    return envelope


def get_user_setting(conn: sqlite3.Connection, setting_key: str) -> ArchiveUserSettingEnvelope | None:
    """Read one setting row, or ``None`` when unset."""

    row = conn.execute(
        "SELECT setting_key, value_json, updated_at_ms, author_ref FROM user_settings WHERE setting_key = ?",
        (setting_key,),
    ).fetchone()
    if row is None:
        return None
    return ArchiveUserSettingEnvelope(
        setting_key=str(row[0]),
        value=json.loads(str(row[1])),
        updated_at_ms=int(row[2]),
        author_ref=str(row[3]),
    )


def get_user_setting_value(conn: sqlite3.Connection, setting_key: str, *, default: JSONValue) -> JSONValue:
    """Read one setting's value, falling back to ``default`` when unset."""

    envelope = get_user_setting(conn, setting_key)
    return default if envelope is None else envelope.value


def list_user_settings(conn: sqlite3.Connection) -> list[ArchiveUserSettingEnvelope]:
    """List every stored setting row, ordered by key."""

    rows = conn.execute(
        "SELECT setting_key, value_json, updated_at_ms, author_ref FROM user_settings ORDER BY setting_key"
    ).fetchall()
    return [
        ArchiveUserSettingEnvelope(
            setting_key=str(row[0]),
            value=json.loads(str(row[1])),
            updated_at_ms=int(row[2]),
            author_ref=str(row[3]),
        )
        for row in rows
    ]


__all__ = [
    "SETTING_KEY_SUBSCRIPTION_TIER",
    "ArchiveUserSettingEnvelope",
    "get_user_setting",
    "get_user_setting_value",
    "known_setting_keys",
    "list_user_settings",
    "set_user_setting",
]
