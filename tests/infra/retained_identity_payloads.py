"""Neutral retained exports for preimage preservation and changed-ID refusal."""

import json

from polylogue.core.json import JSONDocument, JSONValue

IDENTITY_REFERENCE_CASES: tuple[tuple[str, JSONDocument, JSONDocument], ...] = (
    (
        "ordinary-null-empty",
        {"path": "neutral", "optional": None, "empty": ""},
        {"path": "neutral", "optional": None, "empty": ""},
    ),
    ("ordinary-marker-key", {"__POLYLOGUE_NULL__": "literal key"}, {"__POLYLOGUE_NULL__": "literal key"}),
    ("null-literal", {"nested": [None]}, {"nested": ["__POLYLOGUE_NULL__"]}),
    ("empty-literal", {"nested": [""]}, {"nested": ["__POLYLOGUE_EMPTY__"]}),
    ("nested-marker-list", {"nested": [[None, ""]]}, {"nested": [["__POLYLOGUE_NULL__", "__POLYLOGUE_EMPTY__"]]}),
    ("operational-nfd", {"path": "caf\u00e9"}, {"path": "cafe\u0301"}),
    ("mapping-key-nfd", {"caf\u00e9": "target"}, {"cafe\u0301": "target"}),
    ("colliding-mapping-keys", {"caf\u00e9": "second"}, {"cafe\u0301": "first", "caf\u00e9": "second"}),
)


def identity_export(arguments: JSONDocument) -> JSONDocument:
    return {
        "uuid": "retained-identity",
        "name": "Retained identity",
        "chat_messages": [
            {
                "sender": "assistant",
                "text": "Read the declared path",
                "created_at": "2026-01-01T00:00:00Z",
                "content": [{"type": "tool_use", "name": "read_file", "id": "call-one", "input": arguments}],
            }
        ],
    }


def identity_export_bytes(arguments: JSONDocument) -> bytes:
    payload: JSONValue = identity_export(arguments)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
