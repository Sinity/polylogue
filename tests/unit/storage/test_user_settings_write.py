"""Liveness tests for ``user_settings`` read/write helpers (polylogue-at44).

``user_settings`` had DDL + migration but zero runtime callers before this
module. These tests exercise the typed registry directly against the real
``USER_DDL``, proving get/set/list round-trip and that unknown keys /
invalid values are rejected rather than silently accepted into a junk-drawer
table.
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.storage.sqlite.archive_tiers.user import USER_DDL, USER_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.user_settings_write import (
    SETTING_KEY_SUBSCRIPTION_TIER,
    get_user_setting,
    get_user_setting_value,
    known_setting_keys,
    list_user_settings,
    set_user_setting,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(USER_DDL)
    conn.execute(f"PRAGMA user_version = {USER_SCHEMA_VERSION}")
    return conn


def test_get_user_setting_returns_none_when_unset() -> None:
    conn = _conn()
    assert get_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER) is None
    assert get_user_setting_value(conn, SETTING_KEY_SUBSCRIPTION_TIER, default="pro") == "pro"
    assert list_user_settings(conn) == []


def test_set_user_setting_round_trips_and_upserts() -> None:
    conn = _conn()
    written = set_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER, "max_5x", now_ms=100)
    assert written.setting_key == SETTING_KEY_SUBSCRIPTION_TIER
    assert written.value == "max_5x"
    assert written.updated_at_ms == 100
    assert written.author_ref == "user:local"

    read = get_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER)
    assert read == written
    assert get_user_setting_value(conn, SETTING_KEY_SUBSCRIPTION_TIER, default="pro") == "max_5x"

    updated = set_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER, "max_20x", now_ms=200, author_ref="agent:codex")
    assert updated.value == "max_20x"
    assert updated.updated_at_ms == 200
    assert updated.author_ref == "agent:codex"
    # Upsert, not a second row.
    assert [row.setting_key for row in list_user_settings(conn)] == [SETTING_KEY_SUBSCRIPTION_TIER]


def test_set_user_setting_rejects_unknown_key() -> None:
    conn = _conn()
    with pytest.raises(ValueError, match="unknown setting key"):
        set_user_setting(conn, "not_a_real_setting", "anything")


def test_set_user_setting_rejects_invalid_subscription_tier_value() -> None:
    conn = _conn()
    with pytest.raises(ValueError, match="subscription_tier must be one of"):
        set_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER, "not-a-real-tier")
    with pytest.raises(ValueError, match="subscription_tier must be one of"):
        set_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER, 42)


def test_known_setting_keys_is_the_closed_registry() -> None:
    assert known_setting_keys() == frozenset({SETTING_KEY_SUBSCRIPTION_TIER})


def test_list_user_settings_orders_by_key() -> None:
    conn = _conn()
    set_user_setting(conn, SETTING_KEY_SUBSCRIPTION_TIER, "pro", now_ms=1)
    rows = list_user_settings(conn)
    assert [row.setting_key for row in rows] == sorted(row.setting_key for row in rows)
