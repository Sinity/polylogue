from __future__ import annotations

from typing import cast

import pytest

from polylogue.lib.json import JSONDocument
from polylogue.schemas.pinning import PinDecision, PinSet, apply_pins_to_schema, resolve_pinned_paths


def test_pin_decision_rejects_unknown_actions_and_roles() -> None:
    with pytest.raises(ValueError, match="Pin action"):
        PinDecision(path=".role", role="message_role", action="maybe")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not pinnable"):
        PinDecision.from_dict({"path": ".role", "role": "provider", "action": "confirm"})

    with pytest.raises(ValueError, match="must be a string"):
        PinDecision.from_dict({"path": 123, "role": "message_role", "action": "confirm"})


def test_pin_set_round_trips_confirmed_and_rejected_decisions() -> None:
    pin_set = PinSet(
        provider="chatgpt",
        pins=[
            PinDecision(path=".role", role="message_role", action="confirm", reason="reviewed"),
            PinDecision(path=".body", role="message_body", action="reject", reason="wrong field"),
        ],
    )

    restored = PinSet.from_dict(pin_set.to_dict())

    assert restored.provider == "chatgpt"
    assert restored.confirmed_path("message_role") == ".role"
    assert restored.is_rejected(".body", "message_body") is True
    assert restored.is_rejected(".role", "message_body") is False


def test_resolve_pinned_paths_exposes_confirmed_roles_only() -> None:
    pin_set = PinSet(
        provider="claude-ai",
        pins=[PinDecision(path=".chat_messages[].text", role="message_body", action="confirm")],
    )

    resolved = resolve_pinned_paths({}, pin_set)

    assert resolved["message_body"] == ".chat_messages[].text"
    assert resolved["message_role"] is None


def test_apply_pins_marks_confirmed_nodes_and_suppresses_rejected_annotations() -> None:
    schema = cast(
        JSONDocument,
        {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "x-polylogue-semantic-role": "message_role",
                },
                "body": {
                    "type": "string",
                    "x-polylogue-semantic-role": "message_body",
                    "x-polylogue-evidence": ["candidate"],
                },
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "x-polylogue-semantic-role": "message_container",
                    },
                },
            },
            "anyOf": [
                {
                    "x-polylogue-semantic-role": "conversation_title",
                }
            ],
        },
    )
    pins = PinSet(
        provider="chatgpt",
        pins=[
            PinDecision(path=".role", role="message_role", action="confirm"),
            PinDecision(path=".body", role="message_body", action="reject"),
            PinDecision(path=".messages[]", role="message_container", action="confirm"),
            PinDecision(path=".anyOf[0]", role="conversation_title", action="confirm"),
        ],
    )

    result = apply_pins_to_schema(schema, pins)
    properties = cast(dict[str, JSONDocument], result["properties"])
    role_node = properties["role"]
    body_node = properties["body"]
    messages_node = properties["messages"]
    messages_items = cast(JSONDocument, messages_node["items"])
    any_of = cast(list[JSONDocument], result["anyOf"])

    assert role_node["x-polylogue-pinned"] is True
    assert "x-polylogue-semantic-role" not in body_node
    assert "x-polylogue-evidence" not in body_node
    assert body_node["x-polylogue-rejected"] is True
    assert messages_items["x-polylogue-pinned"] is True
    assert any_of[0]["x-polylogue-pinned"] is True
