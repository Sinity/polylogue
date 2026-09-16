from __future__ import annotations

from typing import Any

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Origin, Provider, TitleSource
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.ids import session_id, session_revision_projection
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers import grok


def _nested_conversation() -> dict[str, Any]:
    """Real production wire shape (grok-export-viewer's ``core.py``):
    responses nested one level under a ``"response"`` key, MongoDB extended
    JSON timestamps."""
    return {
        "conversation": {
            "title": "Debugging a React hook",
            "create_time": {"$date": {"$numberLong": "1712000000000"}},
        },
        "responses": [
            {
                "response": {
                    "sender": "human",
                    "message": "Why is my useEffect running twice?",
                    "create_time": {"$date": {"$numberLong": "1712000001000"}},
                }
            },
            {
                "response": {
                    "sender": "assistant",
                    "message": "This is expected behaviour in React 18 Strict Mode.",
                    "create_time": {"$date": {"$numberLong": "1712000010000"}},
                }
            },
        ],
    }


def _flat_conversation() -> dict[str, Any]:
    """Secondary-source flat variant: sender/message directly on the response
    entry (no nested ``response`` key), plain ISO timestamps."""
    return {
        "conversation": {"title": "Flat shape chat", "create_time": "2026-01-01T00:00:00Z"},
        "responses": [
            {"sender": "user", "message": "hello", "create_time": "2026-01-01T00:00:01Z"},
            {"sender": "grok", "message": "hi there", "create_time": "2026-01-01T00:00:02Z"},
        ],
    }


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_looks_like_conversation_true_for_nested_and_flat_shapes() -> None:
    assert grok.looks_like_conversation(_nested_conversation())
    assert grok.looks_like_conversation(_flat_conversation())


def test_looks_like_conversation_false_for_non_grok_shapes() -> None:
    assert not grok.looks_like_conversation({"chat_messages": []})
    assert not grok.looks_like_conversation({"conversation": "not-a-mapping", "responses": []})
    assert not grok.looks_like_conversation({"conversation": {}, "responses": "not-a-list"})
    assert not grok.looks_like_conversation("not-a-mapping")


def test_looks_like_export_true_for_wrapper_document() -> None:
    assert grok.looks_like_export({"conversations": [_nested_conversation()]})
    assert grok.looks_like_export({"conversations": []})


def test_looks_like_export_false_for_non_export_shapes() -> None:
    assert not grok.looks_like_export({"mapping": {}})
    assert not grok.looks_like_export({"conversations": "not-a-list"})
    assert not grok.looks_like_export({"conversations": [{"not": "a-conversation"}]})


# ---------------------------------------------------------------------------
# parse_conversation
# ---------------------------------------------------------------------------


def test_parse_conversation_nested_response_shape() -> None:
    session = grok.parse_conversation(_nested_conversation(), "grok-1")

    assert session.source_name is Provider.GROK
    assert session.provider_session_id.startswith("conversation-")
    assert session.title == "Debugging a React hook"
    assert session.title_source is TitleSource.ORIGIN
    assert session.created_at == "2024-04-01T19:33:20+00:00"
    assert len(session.messages) == 2
    assert [m.role for m in session.messages] == [Role.USER, Role.ASSISTANT]
    assert session.messages[0].text == "Why is my useEffect running twice?"
    assert session.messages[0].timestamp == "2024-04-01T19:33:21+00:00"
    assert session.messages[0].blocks[0].type is BlockType.TEXT
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[1].provider_message_id.startswith("synthetic-")
    assert session.messages[0].provider_message_id != session.messages[1].provider_message_id
    assert [m.position for m in session.messages] == [0, 1]
    assert [m.is_active_leaf for m in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == session.messages[1].provider_message_id
    assert session.updated_at == session.messages[-1].timestamp


def test_parse_conversation_flat_response_shape() -> None:
    session = grok.parse_conversation(_flat_conversation(), "grok-2")

    assert session.title == "Flat shape chat"
    assert [m.role for m in session.messages] == [Role.USER, Role.ASSISTANT]
    assert session.messages[0].text == "hello"
    assert session.messages[1].text == "hi there"
    assert session.messages[0].timestamp == "2026-01-01T00:00:01+00:00"


def test_parse_conversation_missing_title_falls_back_to_id() -> None:
    payload: dict[str, Any] = {"conversation": {}, "responses": []}
    session = grok.parse_conversation(payload, "fallback-id")
    assert session.title == "fallback-id"
    assert session.title_source is None


def test_parse_conversation_empty_responses_yields_no_messages() -> None:
    payload = {"conversation": {"title": "Empty"}, "responses": []}
    session = grok.parse_conversation(payload, "fallback-id")
    assert session.messages == []
    assert session.active_leaf_message_provider_id is None


def test_parse_conversation_skips_blank_message_text() -> None:
    payload = {
        "conversation": {"title": "T"},
        "responses": [
            {"sender": "human", "message": ""},
            {"sender": "human", "message": None},
            {"sender": "human", "message": "real text"},
        ],
    }
    session = grok.parse_conversation(payload, "fallback-id")
    assert len(session.messages) == 1
    assert session.messages[0].text == "real text"


def test_parse_conversation_unknown_sender_normalizes_via_role_fallback() -> None:
    payload = {
        "conversation": {"title": "T"},
        "responses": [{"sender": "xyz-bot", "message": "hi"}],
    }
    session = grok.parse_conversation(payload, "fallback-id")
    assert session.messages[0].role is Role.UNKNOWN


def test_parse_conversation_missing_sender_yields_unknown_role() -> None:
    payload = {"conversation": {"title": "T"}, "responses": [{"message": "hi"}]}
    session = grok.parse_conversation(payload, "fallback-id")
    assert session.messages[0].role is Role.UNKNOWN


def test_parse_conversation_malformed_mongo_date_does_not_crash() -> None:
    payload = {
        "conversation": {"title": "T", "create_time": {"$date": {}}},
        "responses": [{"sender": "human", "message": "hi", "create_time": {"$date": "not-a-mapping-or-number"}}],
    }
    session = grok.parse_conversation(payload, "fallback-id")
    assert session.created_at is None
    assert session.messages[0].timestamp is None


def test_parse_conversation_epoch_numeric_create_time() -> None:
    payload = {
        "conversation": {"title": "T", "create_time": 1712000000},
        "responses": [{"sender": "human", "message": "hi", "create_time": 1712000001}],
    }
    session = grok.parse_conversation(payload, "fallback-id")
    assert session.created_at == "2024-04-01T19:33:20+00:00"
    assert session.messages[0].timestamp == "2024-04-01T19:33:21+00:00"


def test_parse_conversation_id_stable_across_reparse() -> None:
    payload = _nested_conversation()
    first = grok.parse_conversation(payload, "grok-1")
    second = grok.parse_conversation(payload, "grok-1")
    assert first.provider_session_id == second.provider_session_id
    assert first.provider_session_id.startswith("conversation-")


def test_session_id_distinct_for_same_stem_in_different_directories() -> None:
    """Two different conversations exported as ``~/a/grok.json`` and
    ``~/b/grok.json`` both arrive with fallback id ``grok``; identity must come
    from the conversation, not the acquisition coordinate.

    Anti-vacuity: restoring the stem/index identity (``provider_session_id=
    fallback_id``) makes both ids ``grok`` and turns this red -- which is the
    bug, since the second ingest then full-replaces the first session's
    messages under ``grok:grok`` (polylogue-31zag)."""
    from_dir_a = grok.parse_conversation(_nested_conversation(), "grok")
    from_dir_b = grok.parse_conversation(_flat_conversation(), "grok")

    assert from_dir_a.provider_session_id != from_dir_b.provider_session_id
    assert session_id(Provider.GROK, from_dir_a.provider_session_id) != session_id(
        Provider.GROK, from_dir_b.provider_session_id
    )


def test_session_id_survives_reordered_re_export_under_a_different_filename() -> None:
    """The same conversation re-exported later -- its responses reordered
    within the file, the file itself named differently, and its position in
    the export array changed -- is the same session.

    Anti-vacuity: seeding the identity from the array index or the filename
    stem makes the ids differ; hashing the array-order first message rather
    than the timestamp-ordered opening turn makes the reordering alone make it
    red."""
    payload = _nested_conversation()
    reordered = {**payload, "responses": list(reversed(payload["responses"]))}

    first_export = parse_payload(Provider.GROK, {"conversations": [payload]}, "grok")
    second_export = parse_payload(
        Provider.GROK,
        {"conversations": [_flat_conversation(), reordered]},
        "prod-grok-backend",
    )

    assert len(first_export) == 1
    assert len(second_export) == 2
    assert first_export[0].provider_session_id in {s.provider_session_id for s in second_export}


def test_contentless_conversation_falls_back_to_the_acquisition_id() -> None:
    """The one narrow exception: no admitted response and no create_time means
    there is literally nothing intrinsic to hash.

    Anti-vacuity: widening the fallback to any conversation lacking a
    create_time (or lacking responses) makes the second assertion red."""
    empty = grok.parse_conversation({"conversation": {"title": "T"}, "responses": []}, "stem")
    assert empty.provider_session_id == "stem"

    timestamped = grok.parse_conversation(
        {"conversation": {"title": "T", "create_time": 1712000000}, "responses": []}, "stem"
    )
    assert timestamped.provider_session_id.startswith("conversation-")


def test_parse_conversation_reordered_idless_responses_keep_revision_identity() -> None:
    payload = _flat_conversation()
    reversed_payload = {**payload, "responses": list(reversed(payload["responses"]))}

    forward = grok.parse_conversation(payload, "grok-order")
    reordered = grok.parse_conversation(reversed_payload, "grok-order")

    forward_projection = session_revision_projection(forward)
    reordered_projection = session_revision_projection(reordered)
    assert forward_projection.message_contents == reordered_projection.message_contents
    assert all(
        message.parent_message_provider_id is None and message.parent_message_position is None
        for session in (forward, reordered)
        for message in session.messages
    )


def test_duplicate_grok_response_ids_keep_one_active_leaf() -> None:
    payload = _flat_conversation()
    payload["responses"].append(dict(payload["responses"][-1]))

    session = grok.parse_conversation(payload, "grok-duplicate")

    assert session.messages[-2].provider_message_id == session.messages[-1].provider_message_id
    assert sum(message.is_active_leaf is True for message in session.messages) == 1
    assert session.messages[-1].is_active_leaf is True


# ---------------------------------------------------------------------------
# Dispatch integration: detect_provider / parse_payload through the real path
# ---------------------------------------------------------------------------


def test_detect_provider_recognizes_grok_export_document() -> None:
    document = {"conversations": [_nested_conversation(), _flat_conversation()]}
    assert detect_provider(document) is Provider.GROK


def test_detect_provider_recognizes_single_element_wrapped_grok_document() -> None:
    """Full-ingest can hand a lone JSON document as a one-element list."""
    document = {"conversations": [_nested_conversation()]}
    assert detect_provider([document]) is Provider.GROK


def test_parse_payload_splits_multi_conversation_export_into_n_sessions() -> None:
    document = {"conversations": [_nested_conversation(), _flat_conversation()]}
    sessions = parse_payload(Provider.GROK, document, "grok-export")

    assert len(sessions) == 2
    assert len({s.provider_session_id for s in sessions}) == 2
    assert all(s.provider_session_id.startswith("conversation-") for s in sessions)
    assert all(s.source_name is Provider.GROK for s in sessions)
    assert {s.title for s in sessions} == {"Debugging a React hook", "Flat shape chat"}
    assert origin_from_provider(Provider.GROK) is Origin.GROK_EXPORT


def test_parse_payload_single_conversation_export_matches_direct_parse_identity() -> None:
    document = {"conversations": [_nested_conversation()]}
    sessions = parse_payload(Provider.GROK, document, "grok-export")

    assert len(sessions) == 1
    assert (
        sessions[0].provider_session_id
        == grok.parse_conversation(_nested_conversation(), "grok-export").provider_session_id
    )


def test_parse_payload_unwraps_single_element_wrapped_grok_document() -> None:
    document = {"conversations": [_nested_conversation()]}
    from_dict = parse_payload(Provider.GROK, document, "fallback")
    from_list = parse_payload(Provider.GROK, [document], "fallback")

    assert len(from_dict) == 1
    assert len(from_list) == 1
    assert from_list[0].provider_session_id == from_dict[0].provider_session_id


def test_parse_payload_empty_conversations_list_yields_no_sessions() -> None:
    sessions = parse_payload(Provider.GROK, {"conversations": []}, "grok-export")
    assert sessions == []


def test_parse_payload_skips_malformed_entries_in_mixed_validity_export() -> None:
    """A malformed conversation entry must not become a phantom zero-message
    session -- only entries matching the conversation shape are admitted
    (dispatch.py:_lower_grok_export_payload applies looks_like_conversation
    per entry, mirroring how _bundle_record_specs silently drops non-record
    ChatGPT/Claude AI bundle entries rather than emitting empty sessions)."""
    document = {
        "conversations": [
            _nested_conversation(),
            {"not": "a-conversation-shape"},
            {"conversation": {"title": "missing responses key"}},
            {"conversation": "not-a-mapping", "responses": []},
            _flat_conversation(),
        ]
    }
    sessions = parse_payload(Provider.GROK, document, "grok-export")

    assert len(sessions) == 2
    assert all(s.messages for s in sessions)
    assert {s.title for s in sessions} == {"Debugging a React hook", "Flat shape chat"}
    # Identity is content-derived, so a skipped malformed neighbour cannot
    # shift a surviving conversation's id the way an array-position suffix did.
    assert {s.provider_session_id for s in sessions} == {
        grok.parse_conversation(_nested_conversation(), "x").provider_session_id,
        grok.parse_conversation(_flat_conversation(), "y").provider_session_id,
    }
