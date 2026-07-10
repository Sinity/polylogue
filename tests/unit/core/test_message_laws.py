"""Semantic invariant laws for Message and Session models.

These laws encode structural guarantees that hold for any Message or
Session. Example-heavy semantic projections live in
``test_session_semantics.py``.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings

from polylogue.archive.message.roles import Role
from polylogue.archive.models import Message, Session
from tests.infra.strategies.messages import message_model_strategy, session_model_strategy


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_message_role_is_typed_enum(msg: Message) -> None:
    assert isinstance(msg.role, Role)


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_word_count_matches_text_split(msg: Message) -> None:
    assert msg.word_count == len((msg.text or "").split())


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_noise_and_substantive_mutually_exclusive(msg: Message) -> None:
    assert not (msg.is_noise and msg.is_substantive)


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_thinking_and_substantive_mutually_exclusive(msg: Message) -> None:
    assert not (msg.is_thinking and msg.is_substantive)


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_tool_use_implies_noise(msg: Message) -> None:
    if msg.is_tool_use:
        assert msg.is_noise


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_context_dump_implies_noise(msg: Message) -> None:
    if msg.is_context_dump:
        assert msg.is_noise


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_substantive_implies_dialogue(msg: Message) -> None:
    if msg.is_substantive:
        assert msg.is_dialogue


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_dialogue_implies_user_or_assistant(msg: Message) -> None:
    if msg.is_dialogue:
        assert msg.is_user or msg.is_assistant


@given(session_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_without_noise_removes_all_noise(conv: Session) -> None:
    clean = conv.without_noise()
    assert all(not msg.is_noise for msg in clean.messages)


@given(session_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_substantive_only_all_substantive(conv: Session) -> None:
    filtered = conv.substantive_only()
    assert all(msg.is_substantive for msg in filtered.messages)


@given(session_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_without_noise_preserves_non_noise_count(conv: Session) -> None:
    expected_ids = [msg.id for msg in conv.messages if not msg.is_noise]
    clean = conv.without_noise()
    assert [msg.id for msg in clean.messages] == expected_ids


@given(session_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_substantive_only_preserves_count(conv: Session) -> None:
    expected_ids = [msg.id for msg in conv.messages if msg.is_substantive]
    filtered = conv.substantive_only()
    assert [msg.id for msg in filtered.messages] == expected_ids


@given(session_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_with_roles_preserves_selected_role_count(conv: Session) -> None:
    expected_ids = [msg.id for msg in conv.messages if msg.is_user]
    filtered = conv.with_roles((Role.USER,))
    assert [msg.id for msg in filtered.messages] == expected_ids


@given(session_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_without_noise_idempotent(conv: Session) -> None:
    once = conv.without_noise()
    twice = once.without_noise()
    assert [msg.id for msg in once.messages] == [msg.id for msg in twice.messages]


@given(message_model_strategy())
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_extract_thinking_never_empty_string(msg: Message) -> None:
    result = msg.extract_thinking()
    assert result is None or len(result.strip()) > 0


def test_extract_thinking_empty_wrapper_returns_none() -> None:
    msg = Message(id="empty-thinking", role=Role.ASSISTANT, text="<thinking></thinking>")

    assert msg.extract_thinking() is None
