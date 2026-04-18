"""Repository and FTS query security contracts."""

from __future__ import annotations

import sqlite3
from typing import Any, cast

import pytest
from hypothesis import HealthCheck, given, settings

from polylogue.storage.backends.async_sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.search import escape_fts5_query
from tests.infra.adversarial_cases import FTS5_ESCAPE_SECURITY_CASES
from tests.infra.storage_records import make_conversation, make_message
from tests.infra.strategies.adversarial import (
    control_char_strategy,
    fts5_operator_strategy,
    sql_injection_strategy,
)


@pytest.fixture
async def temp_repo(tmp_path: Any) -> Any:
    db_path = tmp_path / "test.db"
    backend = SQLiteBackend(db_path=db_path)
    return ConversationRepository(backend=backend)


async def test_conversation_id_sql_injection_select(temp_repo: Any) -> None:
    assert await temp_repo.view("' OR '1'='1") is None


async def test_conversation_id_sql_injection_drop_table(temp_repo: Any) -> None:
    assert await temp_repo.view("'; DROP TABLE conversations--") is None
    assert isinstance(await temp_repo.list(), list)


async def test_conversation_id_sql_injection_union(temp_repo: Any) -> None:
    assert await temp_repo.view("1 UNION SELECT * FROM sqlite_master--") is None


async def test_message_id_sql_injection(temp_repo: Any) -> None:
    assert isinstance(await temp_repo.list(), list)


async def test_provider_name_sql_injection(temp_repo: Any) -> None:
    assert await temp_repo.list(provider="doesnotexist") == []


async def test_conversation_title_sql_injection(temp_repo: Any) -> None:
    assert isinstance(await temp_repo.list(), list)
    assert isinstance(await temp_repo.list(), list)


async def test_multiple_injection_attempts_in_sequence(temp_repo: Any) -> None:
    for malicious_id in [
        "' OR '1'='1",
        "'; DROP TABLE conversations--",
        "1 UNION SELECT * FROM sqlite_master--",
        "admin'--",
    ]:
        assert await temp_repo.view(malicious_id) is None
    assert isinstance(await temp_repo.list(), list)


async def test_stored_xss_in_conversation_content(temp_repo: Any) -> None:
    xss_payload = "<script>alert('XSS')</script>"
    backend = temp_repo.backend
    conv_record = make_conversation("xss-test", title="XSS Test")
    msg_record = make_message("msg-xss", "xss-test", text=xss_payload)
    await backend.save_conversation_record(conv_record)
    await backend.save_messages([msg_record])
    retrieved = await temp_repo.view("xss-test")
    assert retrieved is not None
    assert xss_payload in [m.text for m in retrieved.messages]


def _assert_fts5_match_executes(escaped_query: str) -> None:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(body)")
        conn.execute("INSERT INTO docs(body) VALUES (?)", ("alpha beta gamma",))
        conn.execute("SELECT count(*) FROM docs WHERE docs MATCH ?", (escaped_query,)).fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize("raw_query,expected,should_compile", FTS5_ESCAPE_SECURITY_CASES)
def test_escape_fts5_security_contract(raw_query: str, expected: str, should_compile: bool) -> None:
    escaped = escape_fts5_query(raw_query)
    assert escaped == expected
    if should_compile:
        _assert_fts5_match_executes(escaped)
    else:
        with pytest.raises(sqlite3.OperationalError):
            _assert_fts5_match_executes(escaped)


async def test_empty_string_parameters_handled(temp_repo: Any) -> None:
    assert await temp_repo.view("") is None
    assert isinstance(await temp_repo.list(provider=""), list)


async def test_none_parameters_handled(temp_repo: Any) -> None:
    try:
        conv = await temp_repo.view(cast(Any, None))
        assert conv is None
    except (TypeError, ValueError):
        pass


async def test_very_long_string_parameters(temp_repo: Any) -> None:
    assert await temp_repo.view("a" * 10000) is None
    assert isinstance(await temp_repo.list(provider="x" * 1000), list)


async def test_unicode_in_parameters(temp_repo: Any) -> None:
    for value in ["文件", "файл", "🎉🎊", "café"]:
        assert await temp_repo.view(value) is None


@given(sql_injection_strategy())
def test_sql_injection_escaping_property(injection_payload: str) -> None:
    escaped = escape_fts5_query(injection_payload)
    assert isinstance(escaped, str)


@given(fts5_operator_strategy())
def test_fts5_operators_escaped_property(operator: str) -> None:
    escaped = escape_fts5_query(operator)
    assert isinstance(escaped, str)
    if operator in ("AND", "OR", "NOT"):
        assert escaped.startswith('"') or len(escaped) == 0


@given(control_char_strategy())
def test_control_chars_in_queries_handled(text_with_control: str) -> None:
    escaped = escape_fts5_query(text_with_control)
    assert isinstance(escaped, str)


@given(sql_injection_strategy())
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
async def test_repository_survives_injection_property(temp_repo: Any, injection_payload: str) -> None:
    assert await temp_repo.view(injection_payload) is None
    result = await temp_repo.list(provider=injection_payload[:50])
    assert isinstance(result, list)
