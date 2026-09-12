"""ChatGPT parser tests — format detection, message extraction, parent/branch, metadata, parsing, real exports."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, TypeAlias
from unittest.mock import patch

import pytest

from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, MaterialOrigin
from polylogue.pipeline.ids import session_revision_projection
from polylogue.scenarios import CorpusSpec
from polylogue.sources.parsers import chatgpt as chatgpt_parser
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.sources.parsers.chatgpt import (
    SHARED_CONVERSATION_INDEX_INGEST_FLAG,
    _coerce_float,
    extract_messages_from_mapping,
)
from polylogue.sources.parsers.chatgpt import looks_like as chatgpt_looks_like
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.claude import looks_like_ai, looks_like_code
from tests.infra.live_ingest import write_session_sync
from tests.infra.source_builders import make_chatgpt_node

ProviderCheck: TypeAlias = Callable[[object], bool]
ProviderDetectionCase: TypeAlias = tuple[object, bool, ProviderCheck, str]
CoerceFloatCase: TypeAlias = tuple[object, float | None, str]
ChatGPTMapping: TypeAlias = dict[str, object]
ExtractMessagesCase: TypeAlias = tuple[ChatGPTMapping, int, str]
ParentBranchCase: TypeAlias = tuple[ChatGPTMapping, list[str | None], list[int], str]
MetadataCase: TypeAlias = tuple[object, str | None, str]
ParseFn: TypeAlias = Callable[[Mapping[str, object], str], ParsedSession]
ParseSessionCase: TypeAlias = tuple[ParseFn, ChatGPTMapping, str, str]


def _looks_like_code_payload(data: object) -> bool:
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes, bytearray)):
        return False
    return looks_like_code(data)


# =============================================================================
# CHATGPT PARSER TESTS
# =============================================================================


# MERGED FORMAT + COERCE DETECTION
_CHATGPT_MINIMAL_VALID: ChatGPTMapping = {
    "id": "conv-1",
    "conversation_id": "conv-1",
    "create_time": 1_700_000_000.0,
    "current_node": "node1",
    "mapping": {"node1": {"id": "node1", "parent": None, "children": []}},
}
PROVIDER_FORMAT_DETECTION_CASES: list[ProviderDetectionCase] = [
    # ChatGPT: a bare "mapping" dict-key is no longer sufficient (polylogue-t0ta)
    # -- detection also requires the export's stable identity fields
    # (conversation_id/id, create_time, current_node) and every mapping node
    # must validate against the typed ChatGPTNode shape.
    ({**_CHATGPT_MINIMAL_VALID, "mapping": {}}, False, chatgpt_looks_like, "ChatGPT: empty mapping is rejected"),
    (_CHATGPT_MINIMAL_VALID, True, chatgpt_looks_like, "ChatGPT: valid with nodes"),
    ({"mapping": {}}, False, chatgpt_looks_like, "ChatGPT: bare mapping key alone is not enough"),
    ({"mapping": {"node1": {}}}, False, chatgpt_looks_like, "ChatGPT: mapping node missing required id rejected"),
    (
        {**_CHATGPT_MINIMAL_VALID, "mapping": {"node1": {"not_a_node": True}}},
        False,
        chatgpt_looks_like,
        "ChatGPT: malformed/format-drifted node shape rejected",
    ),
    (
        {**_CHATGPT_MINIMAL_VALID, "current_node": 123},
        False,
        chatgpt_looks_like,
        "ChatGPT: non-string current_node rejected",
    ),
    (
        {k: v for k, v in _CHATGPT_MINIMAL_VALID.items() if k != "create_time"},
        False,
        chatgpt_looks_like,
        "ChatGPT: missing create_time rejected",
    ),
    ({}, False, chatgpt_looks_like, "ChatGPT: missing mapping"),
    (None, False, chatgpt_looks_like, "ChatGPT: None input"),
    # Claude AI
    (
        {"chat_messages": []},
        False,
        looks_like_ai,
        "Claude AI: empty chat_messages refused (no positive turn evidence, tightened sibling of #3428)",
    ),
    (
        {"chat_messages": [{"foo": "bar"}]},
        False,
        looks_like_ai,
        "Claude AI: chat_messages entries without role+content shape refused",
    ),
    (
        {"chat_messages": [{"sender": "human", "text": "hi"}]},
        True,
        looks_like_ai,
        "Claude AI: chat_messages entry with role+content shape accepted",
    ),
    ({}, False, looks_like_ai, "Claude AI: missing chat_messages"),
    (None, False, looks_like_ai, "Claude AI: None"),
    # Claude Code
    ([{"parentUuid": "123"}], True, _looks_like_code_payload, "Claude Code: parentUuid"),
    ([], False, _looks_like_code_payload, "Claude Code: empty list"),
    (None, False, _looks_like_code_payload, "Claude Code: None"),
]


@pytest.mark.parametrize("data,expected,check_fn,desc", PROVIDER_FORMAT_DETECTION_CASES)
def test_provider_format_detection(data: object, expected: bool, check_fn: ProviderCheck, desc: str) -> None:
    """Unified format detection across all providers."""
    result = check_fn(data)
    assert result == expected, f"Failed {desc}"


def test_chatgpt_rich_web_metadata_is_promoted_to_typed_constructs() -> None:
    messages, _ = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant", "name": "research_kickoff_tool.start_research_task"},
                    "recipient": "canmore.update_textdoc",
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Research answer"]},
                    "metadata": {
                        "canvas": {"textdoc_id": "canvas-1"},
                        "content_references": [{"url": "https://example.test/ref"}],
                        "search_result_groups": [{"query": "polylogue"}],
                        "search_queries": ["polylogue browser capture"],
                        "selected_sources": [{"title": "Source"}],
                        "async_task_type": "deep_research",
                        "async_task_id": "task-1",
                        "citations": [{"start_ix": 0, "end_ix": 8}],
                    },
                },
            }
        }
    )

    constructs = messages[0].blocks[0].web_constructs
    assert [construct.construct_type.value for construct in constructs] == [
        "canvas",
        "content_reference",
        "content_reference",
        "search_query",
        "selected_source",
        "async_task",
    ]
    assert constructs[0].source_id == "canvas-1"
    assert constructs[1].url == "https://example.test/ref"
    assert constructs[2].start_index == 0
    assert constructs[3].query == "polylogue browser capture"
    assert constructs[4].title == "Source"
    assert constructs[5].task_type == "deep_research"
    assert constructs[5].task_id == "task-1"
    assert messages[0].blocks[0].metadata is None


def test_chatgpt_keeps_image_asset_only_nodes() -> None:
    messages, _attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "image-msg",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [
                            {
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "file-service://image-asset-1",
                            }
                        ],
                    },
                    "metadata": {"model_slug": "gpt-image"},
                },
            }
        }
    )

    assert len(messages) == 1
    assert messages[0].provider_message_id == "image-msg"
    assert messages[0].blocks[0].type is BlockType.IMAGE
    assert messages[0].blocks[0].metadata == {"asset_pointer": "file-service://image-asset-1"}


def test_image_asset_pointer_part_becomes_an_attachment() -> None:
    """bd polylogue-91kys: an asset pointer needs an attachment row to bind to.

    An `image_asset_pointer` part names bytes the export ships. The IMAGE
    block carries the pointer as metadata, but block metadata is not an
    acquisition identity — `assembly_chatgpt.py` joins acquired asset members
    onto attachments by the bare file id, so a pointer with no attachment row
    leaves its acquired bytes unbindable. Red if the branch goes back to
    emitting the IMAGE block alone.
    """
    messages, attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "image-msg",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [
                            {
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "sediment://file_00000000005061f6be1311c8d48a7716",
                                "size_bytes": 27100,
                                "width": 1024,
                                "height": 768,
                            }
                        ],
                    },
                },
            }
        }
    )

    assert messages[0].blocks[0].type is BlockType.IMAGE
    assert messages[0].blocks[0].metadata == {
        "asset_pointer": "sediment://file_00000000005061f6be1311c8d48a7716",
        "width": "1024",
        "height": "768",
        "size_bytes": "27100",
    }

    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment.provider_attachment_id == "sediment://file_00000000005061f6be1311c8d48a7716"
    assert attachment.message_provider_id == "image-msg"
    # The bare id is the join key the export's asset members are named by.
    assert attachment.provider_file_id == "file_00000000005061f6be1311c8d48a7716"
    assert attachment.size_bytes == 27100
    assert attachment.attachment_kind == "image_asset"
    assert attachment.direction == "model_output"
    assert attachment.producer_ref == "message:image-msg"


def test_image_asset_pointer_does_not_duplicate_its_metadata_attachment() -> None:
    """One upload named twice is one attachment.

    A user upload appears both as a message `metadata.attachments` row (bare
    `file-<id>`) and as an `image_asset_pointer` part (`file-service://file-<id>`).
    Both normalize to the same file id, so the pointer must not mint a second
    row for the same bytes. Red if the dedupe by bare id is dropped.
    """
    _messages, attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "upload-msg",
                    "author": {"role": "user"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [
                            {
                                "content_type": "image_asset_pointer",
                                "asset_pointer": "file-service://file-ABC123",
                            },
                            "look at this",
                        ],
                    },
                    "metadata": {
                        "attachments": [
                            {"id": "file-ABC123", "name": "photo.png", "mime_type": "image/png", "size": 4096}
                        ]
                    },
                },
            }
        }
    )

    assert len(attachments) == 1
    assert attachments[0].provider_attachment_id == "file-ABC123"
    assert attachments[0].name == "photo.png"


def test_image_asset_pointer_without_a_pointer_invents_no_attachment() -> None:
    """No pointer, no asset row -- an attachment is never invented."""
    _messages, attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "image-msg",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [{"content_type": "image_asset_pointer"}],
                    },
                },
            }
        }
    )

    assert attachments == []


def test_audio_asset_pointer_part_becomes_a_typed_attachment() -> None:
    """Audio pointers retain the acquisition identity beside the construct.

    The document/web-construct projection alone is insufficient: ChatGPT
    export members join acquired bytes to the session through the attachment's
    normalized ``provider_file_id``. This witness goes red if the parser keeps
    only the transcription-shaped document block.
    """
    messages, attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "audio-msg",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [
                            {
                                "content_type": "audio_asset_pointer",
                                "asset_pointer": "file-service://file-AUDIO7",
                                "mime_type": "audio/wav",
                                "size_bytes": 4096,
                            }
                        ],
                    },
                },
            }
        }
    )

    construct = messages[0].blocks[0].web_constructs[0]
    assert construct.asset_pointer == "file-service://file-AUDIO7"
    assert construct.mime_type == "audio/wav"
    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment.provider_attachment_id == "file-service://file-AUDIO7"
    assert attachment.provider_file_id == "file-AUDIO7"
    assert attachment.mime_type == "audio/wav"
    assert attachment.size_bytes == 4096
    assert attachment.attachment_kind == "audio_asset"
    assert attachment.direction == "model_output"
    assert attachment.producer_ref == "message:audio-msg"


def test_realtime_audio_video_pointer_shapes_retain_each_asset_reference() -> None:
    """Realtime A/V uses type-specific and nested pointer fields."""
    messages, attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "realtime-msg",
                    "author": {"role": "user"},
                    "create_time": 1,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [
                            {
                                "content_type": "real_time_user_audio_video_asset_pointer",
                                "audio_asset_pointer": "file-service://file-REALTIME-AUDIO",
                                "video_container_asset_pointer": {
                                    "asset_pointer": "file-service://file-REALTIME-VIDEO",
                                    "mime_type": "video/mp4",
                                },
                                "frames_asset_pointers": [
                                    "file-service://file-REALTIME-FRAME-1",
                                    {"asset_pointer": "file-service://file-REALTIME-FRAME-2"},
                                ],
                            }
                        ],
                    },
                },
            }
        }
    )

    construct_pointers = {construct.asset_pointer for construct in messages[0].blocks[0].web_constructs}
    assert construct_pointers == {
        "file-service://file-REALTIME-AUDIO",
        "file-service://file-REALTIME-VIDEO",
        "file-service://file-REALTIME-FRAME-1",
        "file-service://file-REALTIME-FRAME-2",
    }
    by_pointer = {attachment.provider_attachment_id: attachment for attachment in attachments}
    assert set(by_pointer) == construct_pointers
    assert by_pointer["file-service://file-REALTIME-AUDIO"].attachment_kind == "audio_asset"
    assert by_pointer["file-service://file-REALTIME-VIDEO"].attachment_kind == "video_asset"
    assert by_pointer["file-service://file-REALTIME-FRAME-1"].attachment_kind == "video_frame_asset"
    assert all(attachment.direction == "user_input" for attachment in attachments)
    assert all(attachment.producer_ref is None for attachment in attachments)


def test_chatgpt_shared_conversation_index_shell_is_tagged() -> None:
    session = chatgpt_parse(
        {
            "conversation_id": "shared-conv",
            "id": "share-row",
            "is_anonymous": True,
            "title": "Shared title only",
        },
        "fallback",
    )

    assert session.messages == []
    assert SHARED_CONVERSATION_INDEX_INGEST_FLAG in session.ingest_flags


def _shared_decode_payload() -> dict[str, object]:
    """A minimal shared-page (chatgpt.com/share/<id>) stream-decode document.

    Mirrors the real recovery-packet decode shape (polylogue-4zqh3): a
    top-level ``messages`` list of flat ``{node_id, parent, children, role,
    text, ...}`` records, no ``mapping`` key anywhere, and no payload-
    asserted ``id`` (only ``conversation_id``/``shared_conversation_id``).
    """
    return {
        "source_url": "https://chatgpt.com/share/shared-conv-id",
        "shared_conversation_id": "shared-conv-id",
        "conversation_id": "shared-conv-id",
        "title": "Shared Decode Title",
        "default_model_slug": "gpt-5-5-pro",
        "create_time": 1700000000.0,
        "update_time": 1700000100.0,
        "mapping_node_count": 3,
        "message_count": 2,
        "messages": [
            {
                "node_id": "root-node",
                "message_id": "root-node",
                "parent": "client-created-root",
                "children": ["user-node"],
                "role": "system",
                "create_time": None,
                "update_time": None,
                "status": "finished_successfully",
                "metadata": {"is_visually_hidden_from_conversation": True},
                "text": "",
            },
            {
                "node_id": "user-node",
                "message_id": "user-node",
                "parent": "root-node",
                "children": ["assistant-node"],
                "role": "user",
                "create_time": 1700000000.0,
                "update_time": None,
                "status": "finished_successfully",
                "metadata": {},
                "text": "Hello from the shared page decode",
            },
            {
                "node_id": "assistant-node",
                "message_id": "assistant-node",
                "parent": "user-node",
                "children": [],
                "role": "assistant",
                "create_time": 1700000050.0,
                "update_time": None,
                "status": "finished_successfully",
                "metadata": {},
                "text": "Reply from the shared page decode",
            },
        ],
    }


def test_chatgpt_looks_like_shared_decode_detects_flat_messages_shape() -> None:
    from polylogue.sources.parsers.chatgpt import looks_like_fragment, looks_like_shared_decode

    payload = _shared_decode_payload()
    assert looks_like_shared_decode(payload)
    # The neighboring mapping-tree detectors must NOT also claim this shape --
    # it genuinely has no "mapping" key, so both would silently reject it
    # anyway, but this pins the non-overlap explicitly.
    assert not chatgpt_looks_like(payload)
    assert not looks_like_fragment(payload)


def test_chatgpt_looks_like_shared_decode_rejects_native_mapping_export() -> None:
    from polylogue.sources.parsers.chatgpt import looks_like_shared_decode

    assert not looks_like_shared_decode({"mapping": {}, "messages": [{"node_id": "x", "role": "user"}]})
    assert not looks_like_shared_decode({"shared_conversation_id": "x", "messages": []})
    assert not looks_like_shared_decode({"shared_conversation_id": "x"})
    assert not looks_like_shared_decode("not-a-dict")


def test_chatgpt_parse_shared_decode_produces_real_messages() -> None:
    session = chatgpt_parse(_shared_decode_payload(), "fallback")

    assert session.provider_session_id == "shared-conv-id"
    assert session.title == "Shared Decode Title"
    assert SHARED_CONVERSATION_INDEX_INGEST_FLAG not in session.ingest_flags
    # Shared-page decodes preserve every mapping node, including the hidden
    # empty system root, so parent/child relationships remain lossless.
    assert len(session.messages) == 3
    by_id = {message.provider_message_id: message for message in session.messages}
    assert by_id["root-node"].role.value == "system"
    assert by_id["root-node"].blocks == []
    assert by_id["user-node"].parent_message_provider_id == "root-node"
    roles_and_text = [(m.role, m.blocks[0].text if m.blocks else "") for m in session.messages]
    assert ("user", "Hello from the shared page decode") in roles_and_text
    assert ("assistant", "Reply from the shared page decode") in roles_and_text
    # The leaf node (no children) is derived as the active path tail.
    assert session.active_leaf_message_provider_id == "assistant-node"


def test_chatgpt_shared_decode_lowering_preserves_conservation_mapping() -> None:
    from polylogue.sources.dispatch import lower_chatgpt_documents

    [document] = lower_chatgpt_documents(_shared_decode_payload(), "fallback")

    assert document.document_id == "shared-conv-id"
    assert document.artifact_class == "shared_page_decode"
    assert set(document.mapping) == {"root-node", "user-node", "assistant-node"}
    user_node = document.mapping["user-node"]
    assistant_node = document.mapping["assistant-node"]
    assert isinstance(user_node, dict)
    assert isinstance(assistant_node, dict)
    assert user_node["parent"] == "root-node"
    assert assistant_node["children"] == []


def test_chatgpt_shared_decode_sequence_lowering_preserves_conservation_mapping() -> None:
    """A sequence wrapper must not make the supported shared document vanish."""
    from polylogue.sources.dispatch import lower_chatgpt_documents

    [document] = lower_chatgpt_documents([_shared_decode_payload()], "fallback")

    assert document.document_id == "shared-conv-id"
    assert document.artifact_class == "shared_page_decode"
    assert set(document.mapping) == {"root-node", "user-node", "assistant-node"}


def test_chatgpt_shared_decode_dispatch_accepts_source_scan_sequence() -> None:
    from polylogue.sources.dispatch import parse_payload

    sessions = parse_payload("chatgpt", [_shared_decode_payload()], "fallback")

    assert len(sessions) == 1
    assert len(sessions[0].messages) == 3


def test_chatgpt_shared_decode_lowering_uses_parser_conversation_id_precedence() -> None:
    from polylogue.sources.dispatch import lower_chatgpt_documents

    payload = {**_shared_decode_payload(), "conversation_id": "native-conversation", "id": "native-id"}
    [document] = lower_chatgpt_documents(payload, "fallback")

    assert document.document_id == "native-id"


def test_chatgpt_temporary_payload_sets_session_kind() -> None:
    session = chatgpt_parse(
        {
            "id": "temporary-native",
            "title": "Temporary",
            "is_temporary": True,
            "mapping": {},
        },
        "fallback",
    )

    assert session.session_kind == "temporary"
    assert "capture:temporary-chat" in session.ingest_flags


# COERCE FLOAT - MERGED WITH FORMAT DETECTION ABOVE

COERCE_FLOAT_CASES: list[CoerceFloatCase] = [
    (42, 42.0, "int"),
    (3.14, 3.14, "float"),
    ("2.5", 2.5, "string number"),
    ("2024-01-15T10:30:00Z", 1705314600.0, "ISO datetime string"),
    ("invalid", None, "invalid string"),
    (None, None, "None"),
]


@pytest.mark.parametrize("input_val,expected,desc", COERCE_FLOAT_CASES)
def test_coerce_float(input_val: object, expected: float | None, desc: str) -> None:
    """Test _coerce_float conversion."""
    result = _coerce_float(input_val)
    assert result == expected, f"Failed {desc}"


def test_chatgpt_message_extraction_sorts_iso_timestamps() -> None:
    mapping = {
        "late": {
            "id": "late",
            "message": {
                "id": "late",
                "author": {"role": "assistant"},
                "content": {"parts": ["later"]},
                "create_time": "2024-01-15T10:31:00Z",
            },
        },
        "early": {
            "id": "early",
            "message": {
                "id": "early",
                "author": {"role": "user"},
                "content": {"parts": ["earlier"]},
                "create_time": "2024-01-15T10:30:00Z",
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert [message.provider_message_id for message in messages] == ["early", "late"]


def test_chatgpt_idless_message_does_not_get_a_positional_provider_id() -> None:
    mapping = {
        "node": {
            "message": {
                "author": {"role": "user"},
                "content": {"parts": ["hello"]},
                "create_time": "2024-01-15T10:30:00Z",
            }
        }
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert [message.provider_message_id for message in messages] == [""]


def test_chatgpt_idless_message_reordering_keeps_revision_identity_and_native_ids() -> None:
    idless: ChatGPTMapping = {
        "message": {
            "author": {"role": "user"},
            "content": {"parts": ["Question"]},
            "create_time": 1_700_000_000.0,
        },
        "parent": None,
        "children": [],
    }
    native: ChatGPTMapping = {
        "id": "native-node",
        "message": {
            "id": "native-message",
            "author": {"role": "assistant"},
            "content": {"parts": ["Answer"]},
            "create_time": 1_700_000_001.0,
        },
        "parent": None,
        "children": [],
    }
    forward = chatgpt_parse(
        {"id": "chatgpt-order", "mapping": {"idless": idless, "native": native}},
        "fallback",
    )
    reordered = chatgpt_parse(
        {"id": "chatgpt-order", "mapping": {"native": native, "idless": idless}},
        "fallback",
    )

    assert [message.provider_message_id for message in forward.messages] == ["", "native-message"]
    assert (
        session_revision_projection(forward).message_contents == session_revision_projection(reordered).message_contents
    )


# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 17)


CHATGPT_EXTRACT_MESSAGES_CASES: list[ExtractMessagesCase] = [
    # Basic extraction
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, 1, "basic message"),
    # Timestamp handling
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=1704067200)}, 1, "with timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=None)}, 1, "null timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=0)}, 1, "zero timestamp"),
    # Mixed timestamps (should sort)
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["First"], timestamp=1000),
            "node2": make_chatgpt_node("msg2", "assistant", ["Second"], timestamp=2000),
            "node3": make_chatgpt_node("msg3", "user", ["Third"], timestamp=500),
        },
        3,
        "mixed timestamps sorted",
    ),
    # Content variants
    ({"node1": make_chatgpt_node("msg1", "user", ["Part1", "Part2"])}, 1, "multiple parts"),
    (
        {
            "node1": {
                "message": {
                    "id": "msg1",
                    "author": {"role": "user"},
                    "content": {"parts": [None, "Valid"]},
                }
            }
        },
        1,
        "parts with None",
    ),
    ({"node1": {"message": {"id": "1", "author": {"role": "user"}, "content": {"parts": []}}}}, 0, "empty parts"),
    # Role normalization
    ({"node1": make_chatgpt_node("msg1", "human", ["Hi"])}, 1, "human role alias"),
    ({"node1": make_chatgpt_node("msg1", "model", ["Response"])}, 1, "model role alias"),
    # Missing fields
    ({"node1": {"id": "1", "message": None}}, 0, "missing message"),
    ({"node1": {"id": "1", "message": {"id": "1"}}}, 0, "missing author"),
    ({"node1": {"id": "1", "message": {"id": "1", "author": {}}}}, 0, "missing role"),
    ({"node1": {"id": "1", "message": {"id": "1", "author": {"role": "user"}}}}, 0, "missing content"),
    # Non-dict nodes
    ({"node1": "not a dict"}, 0, "non-dict node"),
    ({"node1": None}, 0, "None node"),
    # Empty mapping
    ({}, 0, "empty mapping"),
]


@pytest.mark.parametrize("mapping,expected_count,desc", CHATGPT_EXTRACT_MESSAGES_CASES)
def test_chatgpt_extract_messages_comprehensive(mapping: ChatGPTMapping, expected_count: int, desc: str) -> None:
    """Comprehensive message extraction test.

    Replaces 17 individual extraction tests.
    """
    messages, attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == expected_count, f"Failed {desc}: expected {expected_count} messages, got {len(messages)}"

    # Verify all messages have required fields
    for msg in messages:
        assert msg.text is not None
        assert msg.role in ["user", "assistant", "system", "tool"]


# -----------------------------------------------------------------------------
# PARENT & BRANCH INDEX EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_PARENT_BRANCH_CASES: list[ParentBranchCase] = [
    # No parent (root message)
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, [None], [0], "root message no parent"),
    # Simple linear chain
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Hello"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1"),
        },
        [None, "msg1"],
        [0, 0],
        "linear chain parent references",
    ),
    # Branching: one parent with multiple children
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Question"], children=["msg2", "msg3"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Answer 1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["Answer 2"], parent="node1"),
        },
        [None, "msg1", "msg1"],
        [0, 0, 1],
        "branching with branch indexes",
    ),
    # Three-way branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Q"], children=["msg2", "msg3", "msg4"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["A1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["A2"], parent="node1"),
            "node4": make_chatgpt_node("msg4", "assistant", ["A3"], parent="node1"),
        },
        [None, "msg1", "msg1", "msg1"],
        [0, 0, 1, 2],
        "three-way branch indexes",
    ),
    # No parent field in node
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, [None], [0], "missing parent field defaults to None"),
    # Parent node missing from the emitted message set: keep the message, drop
    # the dangling parent edge so storage does not reject the session.
    (
        {"node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1")},
        [None],
        [0],
        "orphaned node drops missing parent",
    ),
    # Mixed chain and branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Start"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Response"], children=["msg3", "msg4"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "user", ["Follow 1"], parent="node2"),
            "node4": make_chatgpt_node("msg4", "user", ["Follow 2"], parent="node2"),
        },
        [None, "msg1", "msg2", "msg2"],
        [0, 0, 0, 1],
        "mixed chain and branch structure",
    ),
]


@pytest.mark.parametrize("mapping,expected_parents,expected_indexes,desc", CHATGPT_PARENT_BRANCH_CASES)
def test_chatgpt_extract_parent_and_branch_index(
    mapping: ChatGPTMapping,
    expected_parents: list[str | None],
    expected_indexes: list[int],
    desc: str,
) -> None:
    """Test extraction of parent_message_provider_id and branch_index.

    Validates parent message references and branch position calculation.
    """
    messages, _ = extract_messages_from_mapping(mapping)

    assert len(messages) == len(expected_parents), (
        f"Failed {desc}: expected {len(expected_parents)} messages, got {len(messages)}"
    )

    for msg, expected_parent, expected_index in zip(messages, expected_parents, expected_indexes, strict=False):
        assert msg.parent_message_provider_id == expected_parent, (
            f"Failed {desc}: message {msg.provider_message_id} expected parent {expected_parent}, "
            f"got {msg.parent_message_provider_id}"
        )
        assert msg.branch_index == expected_index, (
            f"Failed {desc}: message {msg.provider_message_id} expected branch_index {expected_index}, "
            f"got {msg.branch_index}"
        )


def test_chatgpt_drops_parent_links_to_filtered_messages() -> None:
    messages, _ = extract_messages_from_mapping(
        {
            "empty-parent": make_chatgpt_node("parent-msg", "user", [], children=["child"]),
            "child": make_chatgpt_node("child-msg", "assistant", ["survives"], parent="parent-msg"),
            "root": make_chatgpt_node("root-msg", "user", ["root"], children=["valid-child-msg"]),
            "valid-child": make_chatgpt_node("valid-child-msg", "assistant", ["valid"], parent="root-msg"),
        }
    )

    parents = {message.provider_message_id: message.parent_message_provider_id for message in messages}

    assert parents["child-msg"] is None
    assert parents["valid-child-msg"] == "root-msg"


# -----------------------------------------------------------------------------
# METADATA EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_METADATA_CASES: list[MetadataCase] = [
    # Attachments
    ({"attachments": [{"id": "att1", "name": "file.pdf"}]}, "attachments", "attachments field"),
    ({"image_asset_pointer": "asset_123"}, None, "image asset pointer metadata ignored"),
    # Cost/duration
    ({"costUSD": 0.005}, "cost", "cost metadata"),
    ({"durationMs": 2500}, "duration", "duration metadata"),
    # Thinking markers
    ({"content_type": "thoughts"}, "thinking", "thoughts content type"),
    ({"content_type": "reasoning_recap"}, "thinking", "reasoning recap"),
    # Empty
    ({}, None, "no metadata"),
    (None, None, "None metadata"),
]


def test_chatgpt_block_content_type_routes_to_session_events() -> None:
    """``ParsedContentBlock.metadata={"content_type": ...}`` must reach ``session_events``.

    The ``blocks`` table has no metadata column and the write path only
    reads a ``language`` key back out of it (bd polylogue-9x22), so the
    "reasoning_recap" vs "thoughts" distinction on a THINKING block --
    otherwise indistinguishable once both collapse to the same BlockType --
    would be silently dropped at write time. Deleting the
    ``session_events.extend(_block_metadata_evidence_events(messages))``
    call in ``parse()`` makes this fail.
    """
    mapping: dict[str, object] = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"content_type": "reasoning_recap", "parts": [], "text": "Thought about it"},
            },
        }
    }

    session = chatgpt_parse({"mapping": mapping}, "fallback")

    events = [event for event in session.session_events if event.event_type == "chatgpt_block_metadata"]
    assert len(events) == 1
    event = events[0]
    assert event.source_message_provider_id == "msg1"
    assert event.payload == {"block_index": 0, "content_type": "reasoning_recap"}


def test_chatgpt_attachment_payload_preserves_known_bytes_and_unknown_reference() -> None:
    mapping = {
        "node1": make_chatgpt_node("msg1", "assistant", ["answer"]),
    }
    node = mapping["node1"]
    assert isinstance(node, dict)
    message = node["message"]
    assert isinstance(message, dict)
    message["metadata"] = {
        "attachments": [
            {
                "id": "known-attachment",
                "name": "report.txt",
                "mime_type": "text/plain",
                "extracted_content": "known bytes",
            },
            {"name": "unknown-report.txt", "mime_type": "text/plain"},
        ]
    }

    session = chatgpt_parse({"id": "attachment-fixture", "mapping": mapping}, "fallback")

    assert len(session.attachments) == 2
    known, unknown = session.attachments
    assert known.provider_attachment_id == "known-attachment"
    assert known.inline_bytes == b"known bytes"
    assert unknown.provider_attachment_id.startswith("att-")
    assert unknown.inline_bytes is None


@pytest.mark.parametrize("metadata,expected_type,desc", CHATGPT_METADATA_CASES)
def test_chatgpt_metadata_extraction(metadata: object, expected_type: str | None, desc: str) -> None:
    """Test metadata extraction from message metadata field.

    Explicit tests for attachment/cost/thinking metadata.
    """
    mapping: dict[str, object] = {
        "node1": {
            "message": {
                "id": "msg1",
                "author": {"role": "user"},
                "content": {"parts": ["Test"]},
                "metadata": metadata,
            }
        }
    }

    messages, attachments = extract_messages_from_mapping(mapping)

    if expected_type == "attachments":
        # Should have attachment records
        assert len(attachments) > 0
    elif expected_type == "cost":
        # Cost metadata is not retained per-message (cost lives in
        # session_model_usage / reported_cost_usd); the parser must still
        # extract the message rather than drop or crash on the metadata.
        assert len(messages) == 1
    elif expected_type == "thinking":
        # Should mark as thinking
        # (depends on content_blocks implementation)
        pass
    elif expected_type is None:
        pass  # No special metadata expected


def test_chatgpt_thoughts_node_produces_nonempty_thinking_text_end_to_end(test_db: Path) -> None:
    """A ``thoughts`` node's THINKING block must carry real text, and be searchable.

    ``_extract_content_text`` used to only read ``parts``/``text``/``result``;
    a ``thoughts`` content node (the shape reasoning nodes actually use --
    an array of ``{summary, content, ...}`` steps, no top-level text/parts)
    produced an empty string, so ``extract_messages_from_mapping`` built a
    THINKING block with ``text=""`` -- the reasoning content was present in
    the raw payload but invisible everywhere text is read: the block's own
    ``text`` field, and (since ``blocks.search_text`` is generated FROM that
    field) full-text search. Preserving the raw bytes across the
    browser-capture bridge (page_transport.js) is necessary but not
    sufficient if this parser still can't read them -- this test proves the
    text reaches both the parsed block AND a live FTS index, not just that
    JSON round-trips.
    """
    thought_text = "Weighing the tradeoffs between approach A and approach B before answering."
    mapping: dict[str, object] = {
        "node1": {
            "id": "node1",
            "parent": None,
            "message": {
                "id": "reasoning-msg-1",
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "thoughts",
                    "thoughts": [
                        {"summary": "Weighing options", "content": thought_text},
                    ],
                },
                "create_time": 1700000000.0,
            },
        }
    }

    session = chatgpt_parse({"id": "reasoning-only", "mapping": mapping}, "fallback-id")

    assert len(session.messages) == 1
    message = session.messages[0]
    assert message.text == thought_text
    thinking_blocks = [block for block in message.blocks if block.type == BlockType.THINKING]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].text == thought_text

    session_id = write_session_sync(test_db, session)
    conn = sqlite3.connect(str(test_db))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        stored_text = conn.execute(
            "SELECT search_text FROM blocks WHERE session_id = ? AND block_type = 'thinking'",
            (session_id,),
        ).fetchone()
        assert stored_text is not None
        assert thought_text in stored_text[0]
        # messages_fts is a CONTENTLESS FTS5 table (content=''): its own
        # UNINDEXED columns (session_id included) are write-only and never
        # retrievable by SELECT (see FTS_MESSAGES_IDENTITY_TABLE_SQL's
        # docstring in polylogue/storage/fts/sql.py) -- the rowid-to-block_id
        # identity ledger is the real way to resolve a MATCH hit's identity.
        fts_hit = conn.execute(
            """
            SELECT b.session_id
            FROM messages_fts
            JOIN messages_fts_identity ON messages_fts_identity.rowid = messages_fts.rowid
            JOIN blocks AS b ON b.block_id = messages_fts_identity.block_id
            WHERE messages_fts MATCH ?
            """,
            ("tradeoffs",),
        ).fetchone()
        assert fts_hit is not None, "reasoning text must be findable via full-text search, not just stored"
        assert fts_hit[0] == session_id
    finally:
        conn.close()


def test_chatgpt_reasoning_recap_text_reaches_the_thinking_block(test_db: Path) -> None:
    """A ``reasoning_recap`` node carries its recap line under ``content``.

    The record shape is exactly ``{content_type, content}`` -- no ``parts``,
    no top-level ``text`` -- so an extractor that does not try ``content``
    builds the THINKING block with ``text=""`` and the recap is invisible in
    the block, in ``blocks.search_text``, and in search. Dropping ``content``
    from the candidate keys in ``_extract_content_text`` turns this red.
    """
    recap_text = "Thought for 7s about the tradeoffs"
    mapping: dict[str, object] = {
        "node1": {
            "id": "node1",
            "parent": None,
            "message": {
                "id": "recap-msg-1",
                "author": {"role": "assistant"},
                "content": {"content_type": "reasoning_recap", "content": recap_text},
                "create_time": 1700000000.0,
            },
        }
    }

    session = chatgpt_parse({"id": "recap-only", "mapping": mapping}, "fallback-id")

    assert len(session.messages) == 1
    thinking_blocks = [block for block in session.messages[0].blocks if block.type == BlockType.THINKING]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].text == recap_text

    session_id = write_session_sync(test_db, session)
    conn = sqlite3.connect(str(test_db))
    try:
        stored = conn.execute(
            "SELECT search_text FROM blocks WHERE session_id = ? AND block_type = 'thinking'",
            (session_id,),
        ).fetchone()
        assert stored is not None
        assert recap_text in stored[0]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "content_type,content_extra",
    [
        (
            "tether_quote",
            {
                "url": "file-1E2Hm9MJaBwH4JAyEs8EBk",
                "domain": "ledger-2025-06.xml",
                "title": "Quarterly ledger export",
                "tether_id": None,
            },
        ),
        (
            "sonic_webpage",
            {
                "url": "https://example.test/articles/pk-study",
                "domain": "example.test",
                "title": "Safety and Pharmacokinetics of the Candidate",
                "snippet": "a short summary",
                "ref_id": "turn5search0",
            },
        ),
    ],
)
def test_retrieved_source_constructs_conserve_url_and_own_title(
    content_type: str, content_extra: dict[str, object]
) -> None:
    """A retrieved source keeps its address and its own name.

    Both shapes carry ``url`` and ``title`` on every record. Removing ``url=``
    or restoring ``title=domain`` in the retrieval branch turns this red. The
    construct is located by its own type, not by its carrier block: a
    ``role: tool`` node's carrier is re-typed to TOOL_RESULT.
    The construct is what this asserts, not the block type that carries it:
    on a ``role: tool`` node ``_tool_role_result_blocks`` re-types the
    retrieval block to TOOL_RESULT and keeps the construct.
    """
    content: dict[str, object] = {"content_type": content_type, "text": "retrieved excerpt body", **content_extra}
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "tool", "name": "browser"},
                "content": content,
                "create_time": None,
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)
    construct = next(
        construct
        for block in messages[0].blocks
        for construct in block.web_constructs
        if construct.provider_key == content_type
    )
    assert construct.url == content_extra["url"]
    assert construct.title == content_extra["title"]


@pytest.mark.parametrize(
    "code_text",
    [
        pytest.param("print(2 + 3)", id="code-branch"),
        # A recipient-addressed call whose code parses as JSON is claimed by
        # the tool-call branch instead -- the route most `code` contents in
        # the export actually take.
        pytest.param('{"title": "a note", "prompt": "write it"}', id="tool-call-branch"),
    ],
)
def test_chatgpt_code_block_carries_language_to_the_blocks_row(test_db: Path, code_text: str) -> None:
    """A ``code`` node's language must reach ``blocks.language``.

    ``metadata["language"]`` is the sole input the write path reads for that
    column (``archive_tiers/write.py:_block_language``); both TOOL_USE branches
    passed metadata without it, so every chatgpt-export code call wrote NULL.
    Dropping ``language`` from ``tool_use_metadata`` turns this red.
    """
    mapping: dict[str, object] = {
        "node1": {
            "id": "node1",
            "parent": None,
            "message": {
                "id": "code-msg-1",
                "author": {"role": "assistant"},
                "recipient": "python",
                "content": {
                    "content_type": "code",
                    "language": "python",
                    "response_format_name": None,
                    "text": code_text,
                },
                "create_time": 1700000000.0,
            },
        }
    }

    session = chatgpt_parse({"id": "code-only", "mapping": mapping}, "fallback-id")

    tool_use_blocks = [block for block in session.messages[0].blocks if block.type == BlockType.TOOL_USE]
    assert len(tool_use_blocks) == 1
    assert (tool_use_blocks[0].metadata or {}).get("language") == "python"

    session_id = write_session_sync(test_db, session)
    conn = sqlite3.connect(str(test_db))
    try:
        rows = conn.execute(
            "SELECT language FROM blocks WHERE session_id = ? AND block_type = 'tool_use'",
            (session_id,),
        ).fetchall()
        assert rows == [("python",)]
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# FULL PARSE - PARAMETRIZED (1 test replacing 12)
# -----------------------------------------------------------------------------


PARSE_SESSION_CASES: list[ParseSessionCase] = [
    # ChatGPT title extraction
    (chatgpt_parse, {"title": "My Conv", "mapping": {}}, "title", "ChatGPT: title field"),
    (chatgpt_parse, {"name": "Conv Name", "mapping": {}}, "name", "ChatGPT: name field"),
    (chatgpt_parse, {"id": "conv-123", "mapping": {}}, "id", "ChatGPT: id field"),
    (chatgpt_parse, {"mapping": {}}, "fallback", "ChatGPT: uses fallback-id"),
]


@pytest.mark.parametrize("parse_fn,conv_data,check_type,desc", PARSE_SESSION_CASES)
def test_parse_session(parse_fn: ParseFn, conv_data: ChatGPTMapping, check_type: str, desc: str) -> None:
    """Unified session parsing across providers."""
    result = parse_fn(conv_data, "fallback-id")

    if check_type == "title":
        assert result.title in conv_data.values(), f"Failed {desc}"
    elif check_type == "id":
        assert result.provider_session_id == conv_data["id"], f"Failed {desc}"
    elif check_type == "fallback":
        assert result.provider_session_id == "fallback-id", f"Failed {desc}"
    elif check_type == "provider":
        assert result.source_name in ["claude-ai", "claude-code"], f"Failed {desc}"


# -----------------------------------------------------------------------------
# SYNTHETIC DATA INTEGRATION
# -----------------------------------------------------------------------------


def test_chatgpt_parse_synthetic_simple() -> None:
    """Parse synthetic ChatGPT export."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            messages_min=3,
            messages_max=5,
            seed=42,
            origin="generated.test-chatgpt-parser",
            tags=("synthetic", "test", "chatgpt-parser"),
        )
    )[0]
    data = json.loads(raw)

    result = chatgpt_parse(data, "simple-test")

    assert result.source_name == "chatgpt"
    assert len(result.messages) > 0
    assert all(m.text is not None for m in result.messages)


def test_chatgpt_parse_synthetic_branching() -> None:
    """Parse synthetic ChatGPT session with many messages (branching structure)."""
    from polylogue.schemas.synthetic import SyntheticCorpus

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "chatgpt",
            count=1,
            messages_min=12,
            messages_max=19,
            seed=99,
            origin="generated.test-chatgpt-parser",
            tags=("synthetic", "test", "chatgpt-parser"),
        )
    )[0]
    data = json.loads(raw)

    result = chatgpt_parse(data, "branching-test")

    assert result.source_name == "chatgpt"
    assert len(result.messages) > 10  # Multiple messages like branching sessions


# -----------------------------------------------------------------------------
# METADATA ROUNDTRIP: parser → materialization → hydration
# -----------------------------------------------------------------------------


def test_chatgpt_metadata_extracted_into_content_blocks() -> None:
    """ChatGPT message metadata is extracted into typed parser fields."""
    # Build a ChatGPT mapping payload with rich message-level metadata.
    payload: dict[str, object] = {
        "title": "Metadata Roundtrip Test",
        "id": "conv-roundtrip-001",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant", "name": "dalle"},
                    "content": {"parts": ["Generated an image of a cat"], "content_type": "text"},
                    "create_time": 1717430400.0,
                    "recipient": "dalle.text2im",
                    "status": "finished_successfully",
                    "end_turn": True,
                    "metadata": {
                        "model_slug": "gpt-4",
                        "citations": [{"title": "Example", "url": "https://example.com"}],
                        "aggregate_result": {"exit_code": 0, "output": "hello world"},
                        "user_context_message_data": {"about_user_message": "I like cats"},
                    },
                },
            },
            "node2": {
                "id": "node2",
                "message": {
                    "id": "msg-2",
                    "author": {"role": "user"},
                    "content": {"parts": ["Show me a cat"], "content_type": "text"},
                    "create_time": 1717430300.0,
                    "metadata": {
                        "model_slug": "gpt-4",
                    },
                },
                "parent": "node1",
            },
        },
    }

    # --- Stage 1: Parse ---
    parsed = chatgpt_parse(payload, "roundtrip-test")
    assert parsed.source_name == "chatgpt"

    assistant_msg = next(m for m in parsed.messages if m.role == "assistant")
    assert len(assistant_msg.blocks) >= 1
    assert assistant_msg.model_name == "gpt-4"
    assert assistant_msg.sender_name == "dalle"
    assert assistant_msg.recipient == "dalle.text2im"
    assert assistant_msg.delivery_status == "finished_successfully"
    assert assistant_msg.end_turn is True
    assert assistant_msg.user_context_text == "I like cats"
    constructs = assistant_msg.blocks[0].web_constructs
    assert any(construct.construct_type.value == "content_reference" for construct in constructs)
    assert any(construct.construct_type.value == "async_task" for construct in constructs)
    assert assistant_msg.blocks[0].metadata is None

    user_msg = next(m for m in parsed.messages if m.role == "user")
    assert len(user_msg.blocks) >= 1
    assert user_msg.model_name == "gpt-4"
    assert user_msg.sender_name is None
    assert user_msg.recipient is None
    assert user_msg.blocks[0].metadata is None


def test_chatgpt_recipient_addressed_json_payload_parses_as_tool_use() -> None:
    """Regression for polylogue-e2yk.

    A recipient-addressed message (e.g. the web-search tool) whose text is a
    JSON-encoded payload must parse as a BlockType.TOOL_USE block, not raw
    BlockType.TEXT -- the reader already folds tool_use blocks by default, so
    this alone fixes the raw-JSON-dumped-in-transcript symptom.
    """
    payload: dict[str, object] = {
        "title": "Web Search Tool Call",
        "id": "conv-tool-call-001",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "content": {
                        "parts": ['{"search_query":[{"q":"Hetzner Cloud prices"}],"response_length":"medium"}'],
                        "content_type": "text",
                    },
                    "create_time": 1717430400.0,
                    "recipient": "web",
                },
            },
        },
    }

    messages, _ = extract_messages_from_mapping(payload["mapping"])  # type: ignore[arg-type]
    assistant_msg = next(m for m in messages if m.role == "assistant")

    assert assistant_msg.recipient == "web"
    assert len(assistant_msg.blocks) == 1
    block = assistant_msg.blocks[0]
    assert block.type == BlockType.TOOL_USE
    assert block.tool_name == "web"
    assert block.tool_input == {
        "search_query": [{"q": "Hetzner Cloud prices"}],
        "response_length": "medium",
    }


def test_chatgpt_recipient_addressed_non_json_text_stays_text() -> None:
    """A recipient-addressed message whose text is NOT JSON stays BlockType.TEXT.

    Only a JSON-parseable payload is reinterpreted as a tool call -- plain
    prose directed at a recipient (e.g. dalle image-gen captions) must not
    be misclassified.
    """
    payload: dict[str, object] = {
        "title": "Non-JSON Recipient Text",
        "id": "conv-non-json-001",
        "mapping": {
            "node1": {
                "id": "node1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "content": {"parts": ["a plain caption, not JSON"], "content_type": "text"},
                    "create_time": 1717430400.0,
                    "recipient": "dalle.text2im",
                },
            },
        },
    }

    messages, _ = extract_messages_from_mapping(payload["mapping"])  # type: ignore[arg-type]
    assistant_msg = next(m for m in messages if m.role == "assistant")

    assert assistant_msg.recipient == "dalle.text2im"
    assert len(assistant_msg.blocks) == 1
    assert assistant_msg.blocks[0].type == BlockType.TEXT


# =============================================================================
# CATALOG-DRIVEN METADATA ROUNDTRIP PERMUTATIONS
# =============================================================================

# Each entry tests a distinct metadata field or combination surviving the
# parser → materialization → hydration pipeline. The catalog provides the
# message-level metadata dict and the expected assertions keyed by field name.

_METADATA_PERMUTATION_CASES: list[tuple[dict[str, object], str, dict[str, object]]] = [
    # --- Single-field permutations ---
    (
        {"model_slug": "gpt-4"},
        "single: model_slug",
        {"chatgpt_model": "gpt-4"},
    ),
    (
        {"model_slug": "gpt-4o"},
        "single: model_slug variant gpt-4o",
        {"chatgpt_model": "gpt-4o"},
    ),
    (
        {"model_slug": "o1"},
        "single: model_slug variant o1",
        {"chatgpt_model": "o1"},
    ),
    # --- author metadata (tool use messages) ---
    (
        {"model_slug": "gpt-4", "is_tool_message": True},
        "combined: model + tool author metadata",
        {"chatgpt_model": "gpt-4", "chatgpt_author_name": "dalle", "chatgpt_recipient": "dalle.text2im"},
    ),
    # --- status ---
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "message_status": "finished_successfully"},
        "combined: model + author + status finished",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_status": "finished_successfully",
        },
    ),
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "message_status": "failed"},
        "combined: model + author + status failed",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_status": "failed",
        },
    ),
    # --- end_turn ---
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "end_turn": True},
        "combined: model + author + end_turn True",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_end_turn": True,
        },
    ),
    (
        {"model_slug": "gpt-4", "is_tool_message": True, "end_turn": False},
        "combined: model + author + end_turn False",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_end_turn": False,
        },
    ),
    # --- citations ---
    (
        {"model_slug": "gpt-4", "citations": [{"title": "Ref", "url": "https://example.com"}]},
        "single: citations list",
        {"chatgpt_model": "gpt-4", "chatgpt_citations": [{"title": "Ref", "url": "https://example.com"}]},
    ),
    # --- code execution ---
    (
        {
            "model_slug": "gpt-4",
            "aggregate_result": {"status": "success", "run_id": "run-9", "code": "print('ok')"},
        },
        "single: code_execution aggregate_result",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_code_execution": {"status": "success", "run_id": "run-9", "code": "print('ok')"},
        },
    ),
    # --- user context ---
    (
        {"model_slug": "gpt-4", "user_context_message_data": {"about_user_message": "likes cats"}},
        "single: user_context_message_data",
        {"chatgpt_model": "gpt-4", "chatgpt_user_context": {"about_user_message": "likes cats"}},
    ),
    # --- full combination ---
    (
        {
            "model_slug": "gpt-4",
            "is_tool_message": True,
            "message_status": "finished_successfully",
            "end_turn": True,
            "citations": [{"title": "A", "url": "https://a.com"}],
            "aggregate_result": {"status": "success", "run_id": "run-11", "code": "print('done')"},
            "user_context_message_data": {"about_user_message": "needs help"},
        },
        "full: all metadata fields combined",
        {
            "chatgpt_model": "gpt-4",
            "chatgpt_author_name": "dalle",
            "chatgpt_recipient": "dalle.text2im",
            "chatgpt_status": "finished_successfully",
            "chatgpt_end_turn": True,
            "chatgpt_citations": [{"title": "A", "url": "https://a.com"}],
            "chatgpt_code_execution": {"status": "success", "run_id": "run-11", "code": "print('done')"},
            "chatgpt_user_context": {"about_user_message": "needs help"},
        },
    ),
]


def _build_chatgpt_message_metadata_payload(
    meta_spec: dict[str, object],
) -> dict[str, object]:
    """Build a ChatGPT mapping payload with the given metadata spec."""
    author: dict[str, object] = {"role": "assistant"}
    if meta_spec.get("is_tool_message"):
        author["name"] = "dalle"

    metadata: dict[str, object] = {}
    if "model_slug" in meta_spec:
        metadata["model_slug"] = meta_spec["model_slug"]
    if "citations" in meta_spec:
        metadata["citations"] = meta_spec["citations"]
    if "aggregate_result" in meta_spec:
        metadata["aggregate_result"] = meta_spec["aggregate_result"]
    if "user_context_message_data" in meta_spec:
        metadata["user_context_message_data"] = meta_spec["user_context_message_data"]

    message: dict[str, object] = {
        "id": "msg-1",
        "author": author,
        "content": {"parts": ["Test message"]},
        "create_time": 1717430400.0,
        "metadata": metadata,
    }

    if "is_tool_message" in meta_spec and meta_spec["is_tool_message"]:
        message["recipient"] = "dalle.text2im"
    if "message_status" in meta_spec:
        message["status"] = meta_spec["message_status"]
    if "end_turn" in meta_spec:
        message["end_turn"] = meta_spec["end_turn"]

    return {
        "title": "Metadata Permutation Test",
        "id": "conv-perm-001",
        "mapping": {"node1": {"id": "node1", "message": message}},
    }


@pytest.mark.parametrize("meta_spec,desc,expected_fields", _METADATA_PERMUTATION_CASES)
def test_chatgpt_metadata_permutation_extracted_by_parser(
    meta_spec: dict[str, object],
    desc: str,
    expected_fields: dict[str, object],
) -> None:
    """Catalog-driven: each metadata field lands in a typed parser field."""
    _ = expected_fields
    payload = _build_chatgpt_message_metadata_payload(meta_spec)

    parsed = chatgpt_parse(payload, "permutation-test")
    assert parsed.source_name == "chatgpt"
    assert len(parsed.messages) >= 1

    blocks = parsed.messages[0].blocks
    assert len(blocks) >= 1
    message = parsed.messages[0]
    constructs = blocks[0].web_constructs
    assert blocks[0].metadata is None

    if "model_slug" in meta_spec:
        assert message.model_name == meta_spec["model_slug"], desc
    if meta_spec.get("is_tool_message"):
        assert message.sender_name == "dalle", desc
        assert message.recipient == "dalle.text2im", desc
    else:
        assert message.sender_name is None, desc
        assert message.recipient is None, desc
    if "message_status" in meta_spec:
        assert message.delivery_status == meta_spec["message_status"], desc
    if "end_turn" in meta_spec:
        assert message.end_turn is meta_spec["end_turn"], desc
    if "citations" in meta_spec:
        reference = next(construct for construct in constructs if construct.construct_type.value == "content_reference")
        assert reference.title == "Ref" or reference.title == "A", desc
    if "aggregate_result" in meta_spec:
        task = next(construct for construct in constructs if construct.construct_type.value == "async_task")
        # The construct carries the executed program, its run id and the run's
        # own status. `output`/`exit_code` appear on no measured record.
        assert task.text in {"print('ok')", "print('done')"}, desc
        assert task.task_id in {"run-9", "run-11"}, desc
        assert task.status == "success", desc
    if "user_context_message_data" in meta_spec:
        assert message.user_context_text in {"likes cats", "needs help"}, desc


# ---------------------------------------------------------------------------
# #1743 — branch graph preservation with active-path metadata
# ---------------------------------------------------------------------------


def _branch_node(
    msg_id: str,
    role: str,
    text: str,
    *,
    parent: str | None = None,
    children: list[str] | None = None,
    content_type: str = "text",
) -> dict[str, Any]:
    """Mapping node helper that does not force a create_time (graph order is truth)."""
    return {
        "id": msg_id,
        "message": {
            "id": msg_id,
            "author": {"role": role},
            "content": {"content_type": content_type, "parts": [text]},
            "create_time": None,
        },
        "parent": parent,
        "children": children or [],
    }


def test_regeneration_preserves_all_branches_and_marks_active_leaf() -> None:
    """A regenerated assistant turn keeps every branch and marks the active leaf."""
    nodes = [
        _branch_node("root", "system", "", parent=None, children=["u1"]),
        _branch_node("u1", "user", "question", parent="root", children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "OLD wrong answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "NEW correct answer", parent="u1", children=[]),
    ]
    payload = {
        "title": "Regenerated",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a_new",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert texts == ["question", "OLD wrong answer", "NEW correct answer"]
    by_id = {m.provider_message_id: m for m in conv.messages}
    assert by_id["u1"].is_active_path is True
    assert by_id["a_old"].is_active_path is False
    assert by_id["a_new"].is_active_path is True
    assert by_id["a_old"].is_active_leaf is False
    assert by_id["a_new"].is_active_leaf is True
    assert conv.active_leaf_message_provider_id == "a_new"


def test_idless_active_path_marks_only_the_current_node_as_leaf() -> None:
    payload = {
        "title": "Id-less active path",
        "mapping": {
            "first": {
                "message": {"author": {"role": "user"}, "content": {"parts": ["First"]}},
                "parent": None,
                "children": ["last"],
            },
            "last": {
                "message": {"author": {"role": "assistant"}, "content": {"parts": ["Last"]}},
                "parent": "first",
                "children": [],
            },
        },
        "current_node": "last",
    }

    conv = chatgpt_parse(payload, "fallback-id")

    assert [message.provider_message_id for message in conv.messages] == ["", ""]
    assert sum(message.is_active_leaf is True for message in conv.messages) == 1
    assert conv.messages[-1].is_active_leaf is True


def test_chatgpt_position_stays_mapping_order_when_active_path_timestamps_are_scrambled() -> None:
    """Archive row positions remain unique while active-path membership stays explicit."""
    root = _branch_node("root", "system", "", parent=None, children=["u1"])
    user = _branch_node("u1", "user", "first active", parent="root", children=["a1"])
    assistant = _branch_node("a1", "assistant", "second active", parent="u1", children=["u2"])
    followup = _branch_node("u2", "user", "third active", parent="a1", children=[])
    user["message"]["create_time"] = 300.0
    assistant["message"]["create_time"] = 100.0
    followup["message"]["create_time"] = 200.0
    payload = {
        "title": "Scrambled active path",
        "mapping": {n["id"]: n for n in [assistant, followup, root, user]},
        "current_node": "u2",
    }

    conv = chatgpt_parse(payload, "fallback-id")
    by_id = {m.provider_message_id: m for m in conv.messages}

    assert by_id["a1"].position == 0
    assert by_id["u2"].position == 1
    assert by_id["u1"].position == 3
    assert by_id["a1"].is_active_path is True
    assert by_id["u2"].is_active_path is True
    assert by_id["u1"].is_active_path is True
    assert [(m.provider_message_id, m.timestamp) for m in conv.messages] == [
        ("a1", "100.0"),
        ("u2", "200.0"),
        ("u1", "300.0"),
    ]


def test_chatgpt_transport_rows_are_classified_as_protocol_material() -> None:
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["q1"]),
        _branch_node("q1", "assistant", '{"queries":["find context"]}', parent="u1", children=["cmd"]),
        _branch_node("cmd", "assistant", "bash -lc ls -lah", parent="q1", children=["a1"]),
        _branch_node("a1", "assistant", "human-readable answer", parent="cmd", children=[]),
    ]
    payload = {
        "title": "Transport rows",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a1",
        "create_time": 1700000000.0,
    }

    conv = chatgpt_parse(payload, "fallback-id")
    by_id = {m.provider_message_id: m for m in conv.messages}

    assert by_id["q1"].message_type is MessageType.PROTOCOL
    assert by_id["q1"].material_origin is MaterialOrigin.RUNTIME_PROTOCOL
    assert by_id["cmd"].message_type is MessageType.PROTOCOL
    assert by_id["cmd"].material_origin is MaterialOrigin.OPERATOR_COMMAND
    assert by_id["a1"].message_type is MessageType.MESSAGE
    assert by_id["a1"].material_origin is MaterialOrigin.ASSISTANT_AUTHORED


def test_no_current_node_preserves_all_nodes_losslessly() -> None:
    """Without current_node, every node is preserved (lossless fallback) (#1744).

    Real ChatGPT exports always carry current_node; synthetic/edge inputs may
    not. With no active-leaf pointer there is no way to know which branch was
    active, so the fallback keeps all nodes rather than silently dropping one.
    """
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "OLD answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "NEW answer", parent="u1", children=[]),
    ]
    payload = {
        "title": "Regenerated no current_node",
        "mapping": {n["id"]: n for n in nodes},
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert texts == ["question", "OLD answer", "NEW answer"]
    assert [m.is_active_path for m in conv.messages] == [None, None, None]
    assert [m.is_active_leaf for m in conv.messages] == [None, None, None]
    assert conv.active_leaf_message_provider_id is None


def test_current_node_pointing_at_old_branch_marks_that_leaf() -> None:
    """current_node is authoritative active-path metadata, not an emission filter."""
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "kept answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "discarded answer", parent="u1", children=[]),
    ]
    payload = {
        "title": "Old branch active",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a_old",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert texts == ["question", "kept answer", "discarded answer"]
    by_id = {m.provider_message_id: m for m in conv.messages}
    assert by_id["a_old"].is_active_path is True
    assert by_id["a_old"].is_active_leaf is True
    assert by_id["a_new"].is_active_path is False
    assert by_id["a_new"].is_active_leaf is False
    assert conv.active_leaf_message_provider_id == "a_old"


def test_chatgpt_archive_contract_fields() -> None:
    nodes = [
        _branch_node("u1", "user", "question", parent=None, children=["a_old", "a_new"]),
        _branch_node("a_old", "assistant", "OLD answer", parent="u1", children=[]),
        _branch_node("a_new", "assistant", "NEW answer", parent="u1", children=[]),
    ]
    nodes[2]["message"]["metadata"] = {"model_slug": "gpt-4o", "durationMs": 2500}
    payload = {
        "title": "Regenerated archive contract",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "a_new",
        "create_time": 1700000000.0,
    }

    conv = chatgpt_parse(payload, "fallback-id")

    assert [m.position for m in conv.messages] == [0, 1, 2]
    assert [m.variant_index for m in conv.messages] == [0, 0, 1]
    active = next(m for m in conv.messages if m.provider_message_id == "a_new")
    assert active.model_name == "gpt-4o"
    assert active.duration_ms == 2500


def test_chatgpt_generation_timing_uses_one_authoritative_reasoning_owner() -> None:
    """Repeated run-wide metadata must not multiply one Pro generation."""
    nodes = [
        _branch_node("u1", "user", "Do the work", parent=None, children=["thought"]),
        _branch_node("thought", "assistant", "working", parent="u1", children=["recap"]),
        _branch_node("recap", "assistant", "reasoning summary", parent="thought", children=["answer"]),
        _branch_node("answer", "assistant", "substantive answer", parent="recap", children=[]),
    ]
    nodes[1]["message"]["content"]["content_type"] = "thoughts"
    nodes[1]["message"]["metadata"] = {"reasoning_start_time": 1784164544.946}
    nodes[2]["message"]["content"]["content_type"] = "reasoning_recap"
    nodes[2]["message"]["metadata"] = {
        "reasoning_start_time": 1784164541.690012,
        "reasoning_end_time": 1784169732.588194,
        "finished_duration_sec": 5190,
    }
    nodes[3]["message"]["metadata"] = {
        "reasoning_start_time": 1784164544.946,
        "finished_duration_sec": 5190,
    }
    payload = {
        "id": "pro-generation",
        "title": "Long Pro generation",
        "mapping": {node["id"]: node for node in nodes},
        "current_node": "answer",
    }

    session = chatgpt_parse(payload, "fallback-id")
    by_id = {message.provider_message_id: message for message in session.messages}
    lifecycle = [event for event in session.session_events if event.event_type == "generation_lifecycle"]

    assert by_id["recap"].duration_ms == 5_190_000
    assert by_id["thought"].duration_ms is None
    assert by_id["answer"].duration_ms is None
    assert session.reported_duration_ms == 5_190_000
    assert len(lifecycle) == 1
    assert lifecycle[0].source_message_provider_id == "recap"
    assert lifecycle[0].payload == {
        "state": "completed",
        "evidence_source": "provider_native",
        "fidelity": "exact",
        "duration_semantics": "provider_reported_elapsed",
        "elapsed_duration_ms": 5_190_000,
        "started_at_ms": 1_784_164_541_690,
        "ended_at_ms": 1_784_169_732_588,
    }


def test_chatgpt_generation_timing_is_invariant_to_equivalent_provider_timestamp_wires() -> None:
    """Offset-equivalent provider times keep exact units and provenance.

    This kills timezone-strip/local-wall calculations and the representative
    duration-unit mutation that stores provider seconds as milliseconds
    without multiplying by 1000.  It also prevents provider-reported elapsed
    time from being relabeled as observed wall, inferred gap, or model compute.
    """

    def parse_timing(start: str, end: str) -> ParsedSession:
        node = _branch_node("recap", "assistant", "summary", parent=None, children=[])
        node["message"]["content"]["content_type"] = "reasoning_recap"
        node["message"]["metadata"] = {
            "reasoning_start_time": start,
            "reasoning_end_time": end,
            "finished_duration_sec": 5190,
        }
        return chatgpt_parse(
            {"id": "equivalent-timing", "mapping": {"recap": node}, "current_node": "recap"},
            "fallback-id",
        )

    utc_session = parse_timing(
        "2026-07-15T20:30:00.123456Z",
        "2026-07-15T21:56:30.123456Z",
    )
    offset_session = parse_timing(
        "2026-07-15T22:30:00.123456+02:00",
        "2026-07-16T06:56:30.123456+09:00",
    )

    assert utc_session.reported_duration_ms == offset_session.reported_duration_ms == 5_190_000
    assert utc_session.messages[0].duration_ms == offset_session.messages[0].duration_ms == 5_190_000
    assert utc_session.session_events == offset_session.session_events
    lifecycle_events = [event for event in utc_session.session_events if event.event_type == "generation_lifecycle"]
    assert len(lifecycle_events) == 1
    payload = lifecycle_events[0].payload
    assert payload == {
        "state": "completed",
        "evidence_source": "provider_native",
        "fidelity": "exact",
        "duration_semantics": "provider_reported_elapsed",
        "elapsed_duration_ms": 5_190_000,
        "started_at_ms": 1_784_147_400_123,
        "ended_at_ms": 1_784_152_590_123,
    }
    assert not ({"wall_elapsed_ms", "displayed_elapsed_ms", "inferred_gap_ms", "model_compute_ms"} & payload.keys())


@pytest.mark.parametrize(
    ("metadata", "expected_duration_ms"),
    [
        ({"reasoning_start_time": 10.0, "reasoning_end_time": 12.25}, 2250),
        ({"reasoning_start_time": 12.0, "reasoning_end_time": 10.0}, None),
        ({"finished_duration_sec": -3}, None),
        ({"finished_duration_sec": "pending"}, None),
    ],
)
def test_chatgpt_generation_timing_fallback_rejects_malformed_values(
    metadata: dict[str, object], expected_duration_ms: int | None
) -> None:
    node = _branch_node("recap", "assistant", "summary", parent=None, children=[])
    node["message"]["content"]["content_type"] = "reasoning_recap"
    node["message"]["metadata"] = metadata

    session = chatgpt_parse(
        {"id": "timing-edge", "mapping": {"recap": node}, "current_node": "recap"},
        "fallback-id",
    )

    assert session.messages[0].duration_ms == expected_duration_ms
    assert session.reported_duration_ms == expected_duration_ms
    lifecycle_events = [event for event in session.session_events if event.event_type == "generation_lifecycle"]
    assert len(lifecycle_events) == (1 if expected_duration_ms is not None else 0)


def test_chatgpt_generation_timing_anchor_is_stable_across_mapping_order() -> None:
    """generation_lifecycle anchors to the same message id regardless of raw
    mapping key order (bd polylogue-uqwd).

    Two candidate nodes on the same generation branch that fully tie on every
    real scoring signal (source rank, reasoning-recap flag, start/end
    presence, end_turn) previously fell through to ``position`` -- the raw
    ``mapping`` dict's iteration order -- as the deciding tiebreak. ChatGPT
    re-exports of the SAME conversation are not guaranteed to serialize
    ``mapping`` keys in the same order every time, so which of the two tied
    nodes won (and therefore which message id the lifecycle event anchored
    to) silently flipped between export vintages even though every message's
    content, including both candidates' own metadata, was byte-identical.
    That anchor drift is exactly what makes ``event_base_identity_hash``
    (``pipeline/ids.py``) treat the SAME generation event as two disjoint
    identities across revisions, producing a spurious revision conflict.

    Both mapping orders below carry the identical node set; only the dict
    insertion order differs, standing in for two export requests of the same
    conversation. The winning anchor must be identical in both cases.
    """
    user = _branch_node("u1", "user", "do the work", parent=None, children=["node_a"])
    node_a = _branch_node("node_a", "assistant", "first draft", parent="u1", children=["node_b"])
    node_b = _branch_node("node_b", "assistant", "final draft", parent="node_a", children=[])
    for node in (node_a, node_b):
        node["message"]["metadata"] = {"finished_duration_sec": 5}

    def _anchor(order: list[dict[str, Any]]) -> str | None:
        payload = {
            "id": "tie-break-order",
            "mapping": {n["id"]: n for n in order},
            "current_node": "node_b",
        }
        session = chatgpt_parse(payload, "fallback-id")
        lifecycle = [event for event in session.session_events if event.event_type == "generation_lifecycle"]
        assert len(lifecycle) == 1
        return lifecycle[0].source_message_provider_id

    anchor_forward = _anchor([user, node_a, node_b])
    anchor_reversed = _anchor([user, node_b, node_a])
    assert anchor_forward == anchor_reversed


def test_chatgpt_mapping_order_does_not_create_revision_conflict() -> None:
    """The parser's stable tie-break reaches the membership classifier.

    The historical implementation used mapping insertion position as the final
    timing-candidate tiebreak. The two otherwise identical export orders then
    anchored their lifecycle event to different messages, which made the
    production revision classifier quarantine both raws as a conflict.
    """
    from polylogue.archive.session_revision_membership import (
        MembershipRevision,
        _relation,
        classify_membership_revisions,
    )

    user = _branch_node("u1", "user", "do the work", parent=None, children=["node_a"])
    node_a = _branch_node("node_a", "assistant", "first draft", parent="u1", children=["node_b"])
    node_b = _branch_node("node_b", "assistant", "final draft", parent="node_a", children=[])
    for node in (node_a, node_b):
        node["message"]["metadata"] = {"finished_duration_sec": 5}

    def parsed(order: list[dict[str, Any]]) -> ParsedSession:
        return chatgpt_parse(
            {"id": "tie-break-order", "mapping": {node["id"]: node for node in order}, "current_node": "node_b"},
            "fallback-id",
        )

    left, right = parsed([user, node_a, node_b]), parsed([user, node_b, node_a])
    revisions = [
        MembershipRevision(raw_id, session_revision_projection(session))
        for raw_id, session in (("raw-left", left), ("raw-right", right))
    ]

    assert left.session_events[0].source_message_provider_id == right.session_events[0].source_message_provider_id
    assert revisions[0].projection.event_contents == revisions[1].projection.event_contents
    assert _relation(revisions[0].projection, revisions[1].projection) == "equal"
    result = classify_membership_revisions(revisions, existing_accepted_raw_id="raw-left")
    assert result.accepted_raw_ids == ("raw-left",)
    assert result.equivalent_raw_ids == ("raw-right",)
    assert result.ambiguous_raw_ids == ()

    original_extract = chatgpt_parser._extract_generation_timings

    def historical_extract(mapping: Mapping[str, object]) -> list[Any]:
        timings = original_extract(mapping)
        timed_message_ids: list[str] = []
        for node_id, raw_node in mapping.items():
            if not isinstance(raw_node, Mapping):
                continue
            raw_message = raw_node.get("message")
            if not isinstance(raw_message, Mapping):
                continue
            raw_author = raw_message.get("author")
            if not isinstance(raw_author, Mapping) or raw_author.get("role") not in {"assistant", "tool"}:
                continue
            metadata = raw_message.get("metadata")
            if not isinstance(metadata, Mapping) or not any(
                field in metadata for field in ("reasoning_start_time", "reasoning_end_time", "finished_duration_sec")
            ):
                continue
            timed_message_ids.append(str(raw_message.get("id") or raw_node.get("id") or node_id))
        assert timed_message_ids
        return [replace(timing, message_provider_id=timed_message_ids[0]) for timing in timings]

    def historically_parsed(order: list[dict[str, Any]]) -> ParsedSession:
        payload = {"id": "tie-break-order", "mapping": {node["id"]: node for node in order}, "current_node": "node_b"}
        with patch.object(chatgpt_parser, "_extract_generation_timings", historical_extract):
            return chatgpt_parse(payload, "fallback-id")

    historical_left = historically_parsed([user, node_a, node_b])
    historical_right = historically_parsed([user, node_b, node_a])
    historical_revisions = [
        MembershipRevision(raw_id, session_revision_projection(session))
        for raw_id, session in (("raw-left", historical_left), ("raw-right", historical_right))
    ]
    assert historical_left.session_events[0].source_message_provider_id == "node_a"
    assert historical_right.session_events[0].source_message_provider_id == "node_b"
    # The historical parser anchored the same provider-remeasured
    # generation_lifecycle event to whichever timed message the mapping order
    # surfaced first, so these two parses genuinely disagree on the anchor.
    # Since polylogue-uqwd the comparison layer pairs an anchor-moved
    # remeasured event with itself, so an unchanged conversation no longer
    # classifies as a conflict on this axis alone.
    assert _relation(historical_revisions[0].projection, historical_revisions[1].projection) == "equal"
    historical_result = classify_membership_revisions(historical_revisions, existing_accepted_raw_id="raw-left")
    assert historical_result.accepted_raw_ids == ("raw-left",)
    assert historical_result.equivalent_raw_ids == ("raw-right",)
    assert historical_result.ambiguous_raw_ids == ()


# ---------------------------------------------------------------------------
# #1744 — non-`parts` content is preserved (code interpreter, execution output)
# ---------------------------------------------------------------------------


def test_code_interpreter_content_is_preserved() -> None:
    """A code node carries top-level content.text (no parts) — must not drop (#1744)."""
    nodes = [
        _branch_node("u1", "user", "run this", parent=None, children=["tool"]),
        {
            "id": "tool",
            "message": {
                "id": "tool",
                "author": {"role": "assistant", "name": "python"},
                "content": {"content_type": "code", "text": "print(1)"},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool"},
                "content": {"content_type": "execution_output", "text": "1\n"},
                "create_time": None,
            },
            "parent": "tool",
            "children": [],
        },
    ]
    payload = {
        "title": "Code interpreter",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    texts = [m.text for m in conv.messages]
    assert "print(1)" in texts
    assert "1\n" in texts
    # Content-block types reflect the code-interpreter semantics.
    from polylogue.core.enums import BlockType

    # bd polylogue-4fm3: code-interpreter calls are TOOL_USE (not CODE), so
    # they join their execution_output TOOL_RESULT via action_pairs/actions
    # (which only joins block_type='tool_use' rows).
    code_msg = next(m for m in conv.messages if m.text == "print(1)")
    assert any(b.type == BlockType.TOOL_USE for b in code_msg.blocks)
    out_msg = next(m for m in conv.messages if m.text == "1\n")
    assert any(b.type == BlockType.TOOL_RESULT for b in out_msg.blocks)


# ---------------------------------------------------------------------------
# polylogue-grub -- execution_output status -> tool_result_is_error
# ---------------------------------------------------------------------------


def _code_and_output_nodes(status: str | None, *, recipient: str = "python") -> list[ChatGPTMapping]:
    return [
        _branch_node("u1", "user", "run this", parent=None, children=["tool"]),
        {
            "id": "tool",
            "message": {
                "id": "tool",
                "author": {"role": "assistant", "name": recipient},
                "recipient": recipient,
                "content": {"content_type": "code", "text": "print(1)"},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool"},
                "content": {"content_type": "execution_output", "text": "1\n"},
                "status": status,
                "create_time": None,
            },
            "parent": "tool",
            "children": [],
        },
    ]


def _parse_execution_output(status: str | None) -> ParsedContentBlock:
    from polylogue.core.enums import BlockType

    nodes = _code_and_output_nodes(status)
    payload = {
        "title": "Code interpreter status",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    out_msg = next(m for m in conv.messages if m.provider_message_id == "out")
    return next(b for b in out_msg.blocks if b.type == BlockType.TOOL_RESULT)


def test_execution_output_finished_successfully_is_not_error() -> None:
    """The provider's own terminal status resolves the tool_result outcome.

    Before this fix every ChatGPT ``tool_result`` block was
    ``tool_result_is_error IS NULL`` (100% unknown, live-measured) even though
    ``finished_successfully``/``finished_partial_completion`` are exactly the
    export's terminal success/failure states. Deleting the status->is_error
    mapping in the ``execution_output`` branch of
    ``polylogue/sources/parsers/chatgpt.py`` makes this fail.
    """
    block = _parse_execution_output("finished_successfully")
    assert block.is_error is False


def test_execution_output_finished_partial_completion_is_error() -> None:
    block = _parse_execution_output("finished_partial_completion")
    assert block.is_error is True


def test_execution_output_in_progress_stays_unknown() -> None:
    """A still-running tool call has no concluded outcome -- must stay NULL,
    never guessed as success or failure."""
    block = _parse_execution_output("in_progress")
    assert block.is_error is None


def test_execution_output_missing_status_stays_unknown() -> None:
    block = _parse_execution_output(None)
    assert block.is_error is None


def test_code_block_carries_recipient_as_tool_name() -> None:
    """A recipient-addressed code-interpreter call's tool identity (e.g.
    "python", "container.exec") is the provider's own ``recipient`` field --
    not something to infer from prose."""
    from polylogue.core.enums import BlockType

    nodes = _code_and_output_nodes("finished_successfully", recipient="container.exec")
    payload = {
        "title": "Code interpreter recipient",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    code_msg = next(m for m in conv.messages if m.provider_message_id == "tool")
    code_block = next(b for b in code_msg.blocks if b.type == BlockType.TOOL_USE)
    assert code_block.tool_name == "container.exec"


# ---------------------------------------------------------------------------
# polylogue-ah21 -- tool_use/tool_result tool_id pairing
# ---------------------------------------------------------------------------


def test_recipient_tool_use_and_result_share_a_tool_id() -> None:
    """A recipient-addressed tool call and its result node must be joinable.

    Exercises ``polylogue.sources.parsers.chatgpt.extract_messages_from_mapping``
    (via ``parse``): the TOOL_USE block emitted for a recipient-addressed
    assistant node and the TOOL_RESULT block emitted for its
    ``execution_output`` child must carry a matching, non-null ``tool_id`` --
    before this fix both were unconditionally ``None`` (verified live: 100% of
    ChatGPT capture tool_use/tool_result blocks had ``tool_id IS NULL``,
    dropping every pair from the ``action_pairs``/``actions`` join). Deleting
    either ``tool_id=str(msg_id)`` on the TOOL_USE branch or
    ``tool_id=parent_message_provider_id`` on the ``execution_output`` branch
    in ``polylogue/sources/parsers/chatgpt.py`` makes this fail.
    """
    from polylogue.core.enums import BlockType

    nodes = [
        _branch_node("u1", "user", "search for x", parent=None, children=["call"]),
        {
            "id": "call",
            "message": {
                "id": "call",
                "author": {"role": "assistant"},
                "recipient": "web",
                "content": {"content_type": "text", "parts": ['{"search_query": ["x"]}']},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["result"],
        },
        {
            "id": "result",
            "message": {
                "id": "result",
                "author": {"role": "tool"},
                "content": {"content_type": "execution_output", "text": "found x"},
                "create_time": None,
            },
            "parent": "call",
            "children": [],
        },
    ]
    payload = {
        "title": "Tool call pairing",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "result",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    by_id = {m.provider_message_id: m for m in conv.messages}

    tool_use_blocks = [b for b in by_id["call"].blocks if b.type == BlockType.TOOL_USE]
    tool_result_blocks = [b for b in by_id["result"].blocks if b.type == BlockType.TOOL_RESULT]
    assert len(tool_use_blocks) == 1
    assert len(tool_result_blocks) == 1
    assert tool_use_blocks[0].tool_id is not None
    assert tool_use_blocks[0].tool_id == tool_result_blocks[0].tool_id == "call"


def test_code_interpreter_call_and_result_share_a_tool_id() -> None:
    """A code-interpreter call and its execution_output must be joinable too.

    bd polylogue-4fm3: live-measured ~4.6:1 tool_result:tool_use skew on
    browser-captured chatgpt sessions traced to this exact gap -- the
    content_type=="code" branch used to emit BlockType.CODE with no
    tool_id, so `action_pairs` (which only joins block_type='tool_use'
    rows) never saw the call, leaving every execution_output permanently
    unpaired. Deleting either `tool_id=str(msg_id)` on the "code" branch or
    the existing `tool_id=parent_message_provider_id` on the
    "execution_output" branch makes this fail.
    """
    from polylogue.core.enums import BlockType

    nodes = _code_and_output_nodes("finished_successfully")
    payload = {
        "title": "Code interpreter pairing",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    by_id = {m.provider_message_id: m for m in conv.messages}

    tool_use_blocks = [b for b in by_id["tool"].blocks if b.type == BlockType.TOOL_USE]
    tool_result_blocks = [b for b in by_id["out"].blocks if b.type == BlockType.TOOL_RESULT]
    assert len(tool_use_blocks) == 1
    assert len(tool_result_blocks) == 1
    assert tool_use_blocks[0].tool_id is not None
    assert tool_use_blocks[0].tool_id == tool_result_blocks[0].tool_id == "tool"
    assert tool_use_blocks[0].tool_input == {"code": "print(1)"}


# SANDBOX FILE LINKS (assistant-generated downloadable deliverables)


def test_sandbox_links_become_unfetchable_attachments() -> None:
    text = (
        "Kit delivered.\n\n"
        "**[Download the ZIP](sandbox:/mnt/data/compiler-kit.zip)**\n"
        "[Checksum](sandbox:/mnt/data/compiler-kit.zip.sha256): `abc`\n"
        "Also see [the prompts dir](sandbox:/mnt/data/compiler-kit/prompts/) "
        "and again [the ZIP](sandbox:/mnt/data/compiler-kit.zip)."
    )
    mapping = {
        "node1": make_chatgpt_node("msg1", "assistant", [text]),
    }

    _messages, attachments = extract_messages_from_mapping(mapping)

    sandbox = [a for a in attachments if a.attachment_kind == "sandbox_file"]
    assert [a.name for a in sandbox] == [
        "compiler-kit.zip",
        "compiler-kit.zip.sha256",
        None,  # directory link keeps trailing slash; no file name
    ]
    assert [a.source_url for a in sandbox] == [
        "sandbox:/mnt/data/compiler-kit.zip",
        "sandbox:/mnt/data/compiler-kit.zip.sha256",
        "sandbox:/mnt/data/compiler-kit/prompts/",
    ]
    assert all(a.message_provider_id == "msg1" for a in sandbox)
    # Duplicate link in the same message is recorded once.
    assert len(sandbox) == 3


def test_sandbox_links_in_user_messages_are_not_attachments() -> None:
    mapping = {
        "node1": make_chatgpt_node("msg1", "user", ["please regenerate sandbox:/mnt/data/old.zip"]),
    }

    _messages, attachments = extract_messages_from_mapping(mapping)

    assert not [a for a in attachments if a.attachment_kind == "sandbox_file"]


def test_sandbox_link_trailing_punctuation_is_stripped() -> None:
    mapping = {
        "node1": make_chatgpt_node(
            "msg1",
            "assistant",
            ["Saved to sandbox:/mnt/data/report.md. Enjoy, or see sandbox:/mnt/data/data.csv,"],
        ),
    }

    _messages, attachments = extract_messages_from_mapping(mapping)

    sandbox = [a for a in attachments if a.attachment_kind == "sandbox_file"]
    assert [a.source_url for a in sandbox] == [
        "sandbox:/mnt/data/report.md",
        "sandbox:/mnt/data/data.csv",
    ]


# CITATION MARKER HYGIENE + SYSTEM-INJECTED CONTEXT


def test_citation_markers_are_stripped_but_citations_survive() -> None:
    marked = "Per the brief. \ue200filecite\ue202turn0file0\ue201 More text \ue200cite\ue202turn1search2\ue202L10-L20\ue201 end."
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"parts": [marked]},
                "metadata": {
                    "citations": [
                        {
                            "citation_format_type": "berry_file_search",
                            "start_ix": 15,
                            "end_ix": 43,
                            "metadata": {"title": "brief.md", "url": "https://example.test/brief"},
                        }
                    ]
                },
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == 1
    message = messages[0]
    text = message.text
    assert text is not None
    assert "\ue200" not in text
    assert "\ue201" not in text
    assert "\ue202" not in text
    assert "filecite" not in text
    assert text.startswith("Per the brief.")
    assert text.endswith("end.")
    assert all("\ue200" not in (block.text or "") for block in message.blocks)
    constructs = [c for block in message.blocks for c in block.web_constructs]
    assert any(c.provider_key == "citations" for c in constructs)


def test_user_editable_context_becomes_runtime_context_message() -> None:
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "user"},
                "content": {
                    "content_type": "user_editable_context",
                    "user_profile": "Profile: local-first archivist.",
                    "user_instructions": "Always answer with evidence refs.",
                },
                "metadata": {"is_visually_hidden_from_conversation": True},
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == 1
    message = messages[0]
    assert message.message_type is MessageType.CONTEXT
    assert message.material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert message.text is not None
    assert "Profile: local-first archivist." in message.text
    assert "Always answer with evidence refs." in message.text
    assert message.blocks[0].metadata == {"content_type": "user_editable_context"}


def test_nonempty_system_node_becomes_runtime_context_message() -> None:
    mapping = {
        "node1": make_chatgpt_node("msg1", "system", ["Follow the runtime policy."]),
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == 1
    assert messages[0].message_type is MessageType.CONTEXT
    assert messages[0].material_origin is MaterialOrigin.RUNTIME_CONTEXT


def test_empty_system_node_remains_omitted() -> None:
    mapping = {
        "node1": make_chatgpt_node("msg1", "system", []),
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert messages == []


def test_model_editable_context_memory_payload_is_kept_and_empty_is_dropped() -> None:
    def node(msg_id: str, model_set_context: str) -> dict[str, object]:
        return {
            "id": msg_id,
            "message": {
                "id": msg_id,
                "author": {"role": "assistant"},
                "content": {
                    "content_type": "model_editable_context",
                    "model_set_context": model_set_context,
                },
            },
        }

    mapping = {
        "node1": node("msg1", "1. Prefers rigorous verification.\n2. Runs Polylogue."),
        "node2": node("msg2", ""),
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    assert [m.provider_message_id for m in messages] == ["msg1"]
    assert messages[0].message_type is MessageType.CONTEXT
    assert messages[0].text is not None
    assert "Prefers rigorous verification" in messages[0].text


def test_file_citation_nested_metadata_is_surfaced() -> None:
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"parts": ["Cited claim."]},
                "metadata": {
                    "citations": [
                        {
                            "citation_format_type": "berry_file_search",
                            "start_ix": 0,
                            "end_ix": 12,
                            "metadata": {
                                "id": "file_00000000b074",
                                "name": "06-strategy-falsification.md",
                                "source": "my_files",
                                "type": "file",
                                "extra": {
                                    "cited_message_id": "bd889688",
                                    "library_file_id": "libfile_9eba",
                                    "source_url": None,
                                },
                            },
                        }
                    ]
                },
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    constructs = [c for b in messages[0].blocks for c in b.web_constructs if c.provider_key == "citations"]
    assert len(constructs) == 1
    citation = constructs[0]
    # Source identity lives one level down (metadata) and two levels down
    # (metadata.extra); losing it reduces a file citation to a bare span.
    assert citation.title == "06-strategy-falsification.md"
    assert citation.source_id == "file_00000000b074"
    assert citation.start_index == 0
    assert citation.end_index == 12


def test_inline_citation_marker_tokens_become_anchored_constructs() -> None:
    marked = "Claim text. \ue200filecite\ue202turn3file14\ue202L180-L293\ue201 More."
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"parts": [marked]},
                "metadata": {},
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    message = messages[0]
    assert message.text is not None
    assert "\ue200" not in message.text
    markers = [c for b in message.blocks for c in b.web_constructs if c.provider_key == "inline_citation_marker"]
    assert len(markers) == 1
    marker = markers[0]
    # Line ranges often exist ONLY in the marker tokens (metadata line_range
    # is frequently null) — the construct must retain them.
    assert marker.text == "filecite turn3file14 L180-L293"
    # Anchored in ORIGINAL-text coordinates, matching citation start_ix/end_ix.
    assert marker.start_index == marked.index("\ue200")
    assert marker.end_index == marked.index("\ue201") + 1


# ---------------------------------------------------------------------------
# polylogue-xofj \u2014 six April-era content types, anonymized real-node shapes
# ---------------------------------------------------------------------------


def test_computer_output_becomes_paired_tool_result() -> None:
    """computer_output (8,192 measured) is a computer-use tool result.

    Anonymized from a real node: ``computer.do`` tool author, a browser
    state snapshot, and the same ``status``-derived terminal outcome
    ``execution_output`` already uses.
    """
    nodes = [
        _branch_node("u1", "user", "check the site", parent=None, children=["call"]),
        {
            "id": "call",
            "message": {
                "id": "call",
                "author": {"role": "assistant", "name": "computer"},
                "content": {"content_type": "text", "parts": ["let me check"]},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool", "name": "computer.do"},
                "content": {
                    "content_type": "computer_output",
                    "computer_id": "71",
                    "screenshot": {
                        "asset_pointer": "sediment://file_anon",
                        "content_type": "image_asset_pointer",
                        "width": 1024,
                        "height": 768,
                    },
                    "state": {
                        "type": "browser_state",
                        "url": "https://example.test/page",
                        "title": "Example Page",
                    },
                    "tether_id": 12345,
                },
                "status": "finished_successfully",
                "create_time": None,
            },
            "parent": "call",
            "children": [],
        },
    ]
    payload = {
        "title": "Computer use",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    out_msg = next(m for m in conv.messages if m.provider_message_id == "out")
    tool_result = next(b for b in out_msg.blocks if b.type == BlockType.TOOL_RESULT)
    assert tool_result.tool_id == "call"
    assert tool_result.is_error is False
    assert tool_result.text is not None
    assert "Example Page" in tool_result.text
    assert "https://example.test/page" in tool_result.text


def test_computer_output_screenshot_becomes_image_block_and_attachment() -> None:
    """The screenshot asset pointer is a real asset reference, not decoration.

    ``content.screenshot`` is an ``image_asset_pointer`` on every measured
    computer_output node. Dropping it leaves the acquired asset bytes with no
    attachment to bind to. Red if the pointer is read for the summary only:
    no IMAGE block, no attachment row, no ``asset_pointer`` evidence event.
    """
    nodes = [
        _branch_node("u1", "user", "check the site", parent=None, children=["call"]),
        {
            "id": "call",
            "message": {
                "id": "call",
                "author": {"role": "assistant", "name": "computer"},
                "content": {"content_type": "text", "parts": ["let me check"]},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool", "name": "computer.do"},
                "content": {
                    "content_type": "computer_output",
                    "computer_id": "71",
                    "screenshot": {
                        "asset_pointer": "sediment://file_00000000e2f06243a164751a50439fb7",
                        "content_type": "image_asset_pointer",
                        "size_bytes": 27100,
                        "width": 1024,
                        "height": 768,
                    },
                    "state": {"type": "browser_state", "url": "https://example.test/page"},
                    "tether_id": 12345,
                },
                "status": "finished_successfully",
                "create_time": None,
            },
            "parent": "call",
            "children": [],
        },
    ]
    payload = {
        "title": "Computer use",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    out_msg = next(m for m in conv.messages if m.provider_message_id == "out")

    image = next(b for b in out_msg.blocks if b.type == BlockType.IMAGE)
    # `blocks` has no dimension columns; the block metadata is projected into a
    # `chatgpt_block_metadata` event, which is where width/height survive.
    assert image.metadata == {
        "asset_pointer": "sediment://file_00000000e2f06243a164751a50439fb7",
        "width": "1024",
        "height": "768",
        "size_bytes": "27100",
    }
    # The structural tool result still leads the message.
    assert out_msg.blocks[0].type == BlockType.TOOL_RESULT

    attachment = next(
        a for a in conv.attachments if a.provider_attachment_id == "sediment://file_00000000e2f06243a164751a50439fb7"
    )
    assert attachment.message_provider_id == "out"
    # The bare id is the join key the export's asset blobs are named by.
    assert attachment.provider_file_id == "file_00000000e2f06243a164751a50439fb7"
    assert attachment.size_bytes == 27100
    assert attachment.attachment_kind == "computer_screenshot"
    assert attachment.direction == "model_output"
    assert attachment.producer_ref == "message:out"

    events = [
        e
        for e in conv.session_events
        if e.event_type == "chatgpt_block_metadata"
        and e.payload.get("asset_pointer") == "sediment://file_00000000e2f06243a164751a50439fb7"
    ]
    assert len(events) == 1


def test_computer_output_without_screenshot_emits_no_attachment() -> None:
    """No pointer, no asset row -- an attachment is never invented."""
    nodes = [
        _branch_node("u1", "user", "check the site", parent=None, children=["out"]),
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool", "name": "computer.initialize"},
                "content": {
                    "content_type": "computer_output",
                    "computer_id": "23",
                    "screenshot": None,
                    "state": {"type": "computer_initialize_state", "os_name": "Chromium"},
                    "tether_id": 223035702712800,
                },
                "status": "finished_successfully",
                "create_time": None,
            },
            "parent": "u1",
            "children": [],
        },
    ]
    payload = {
        "title": "Computer use",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    out_msg = next(m for m in conv.messages if m.provider_message_id == "out")
    assert [b.type for b in out_msg.blocks] == [BlockType.TOOL_RESULT]
    assert conv.attachments == []


@pytest.mark.parametrize(
    "content_type,content_extra,expected_title",
    [
        (
            "tether_quote",
            {"domain": "notes.txt", "tether_id": None},
            "notes.txt",
        ),
        (
            "tether_browsing_display",
            {"domain": None},
            None,
        ),
        (
            "sonic_webpage",
            {"domain": "example.test", "ref_id": "turn0search1", "snippet": "a short summary"},
            "example.test",
        ),
    ],
)
def test_retrieved_source_content_types_become_search_result_constructs(
    content_type: str, content_extra: dict[str, object], expected_title: str | None
) -> None:
    """tether_quote/tether_browsing_display/sonic_webpage are retrieved-source
    evidence, not free text -- they carry a SEARCH_RESULT web construct
    (polylogue-xofj/polylogue-zocm), not bare text. On a ``role: tool`` node
    the block is the browsing tool's answer, so it is a TOOL_RESULT owned by
    the call and its construct survives that typing.
    """
    content: dict[str, object] = {"content_type": content_type, **content_extra}
    if content_type == "tether_browsing_display":
        content["result"] = "# [0\u2020Example\u2020example.test\u3011retrieved page body"
    else:
        content["text"] = "retrieved excerpt body"

    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "tool", "name": "browser"},
                "content": content,
                "create_time": None,
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)
    message = messages[0]
    result_blocks = [b for b in message.blocks if b.type == BlockType.TOOL_RESULT]
    assert result_blocks, f"expected a TOOL_RESULT block for {content_type}"
    constructs = result_blocks[0].web_constructs
    assert len(constructs) == 1
    construct = constructs[0]
    assert construct.construct_type.value == "search_result"
    assert construct.provider_key == content_type
    assert construct.title == expected_title
    assert construct.text is not None


def test_system_error_becomes_paired_error_tool_result() -> None:
    """system_error (177 measured) is a structural failure marker -- the
    content_type itself is the error signal, never guessed from prose, and
    always ``is_error=True`` regardless of the enclosing message's own
    ``status`` (polylogue-xofj).
    """
    nodes = [
        _branch_node("u1", "user", "click the button", parent=None, children=["call"]),
        {
            "id": "call",
            "message": {
                "id": "call",
                "author": {"role": "assistant", "name": "browser"},
                "content": {"content_type": "text", "parts": ["clicking"]},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool", "name": "browser"},
                "content": {
                    "content_type": "system_error",
                    "name": "tool_error",
                    "text": "Error when executing command `mclick(['1'])`",
                },
                # The error-*report* message itself was delivered fine --
                # `status` here must NOT flip is_error to False.
                "status": "finished_successfully",
                "create_time": None,
            },
            "parent": "call",
            "children": [],
        },
    ]
    payload = {
        "title": "Browsing error",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    out_msg = next(m for m in conv.messages if m.provider_message_id == "out")
    tool_result = next(b for b in out_msg.blocks if b.type == BlockType.TOOL_RESULT)
    assert tool_result.tool_id == "call"
    assert tool_result.is_error is True
    assert tool_result.text == "Error when executing command `mclick(['1'])`"
    assert tool_result.metadata is not None
    assert tool_result.metadata["error_name"] == "tool_error"


def test_citable_code_output_becomes_code_result_with_citation_anchor() -> None:
    """citable_code_output (8 measured) is a connector-sourced retrieval
    result -- a code/tool result whose text lives in ``output_str`` (not
    ``text``/``result``) plus a citation anchor identifying the connector
    document it was read from (polylogue-xofj).
    """
    nodes = [
        _branch_node("u1", "user", "check my email", parent=None, children=["call"]),
        {
            "id": "call",
            "message": {
                "id": "call",
                "author": {"role": "assistant", "name": "api_tool"},
                "content": {"content_type": "text", "parts": ["checking"]},
                "create_time": None,
            },
            "parent": "u1",
            "children": ["out"],
        },
        {
            "id": "out",
            "message": {
                "id": "out",
                "author": {"role": "tool", "name": "api_tool.call_tool"},
                "content": {
                    "content_type": "citable_code_output",
                    "output_str": "L0: Subject: Example receipt",
                    "metadata": {
                        "connector_id": "connector_anon",
                        "connector_source": "Gmail",
                        "display_title": "Example receipt #1",
                        "display_url": "https://mail.example.test/all/anon",
                    },
                    "tether_id": 665493930970187,
                },
                "status": "finished_successfully",
                "create_time": None,
            },
            "parent": "call",
            "children": [],
        },
    ]
    payload = {
        "title": "Connector read",
        "mapping": {n["id"]: n for n in nodes},
        "current_node": "out",
        "create_time": 1700000000.0,
    }
    conv = chatgpt_parse(payload, "fallback-id")
    out_msg = next(m for m in conv.messages if m.provider_message_id == "out")
    tool_result = next(b for b in out_msg.blocks if b.type == BlockType.TOOL_RESULT)
    assert tool_result.tool_id == "call"
    assert tool_result.is_error is False
    assert tool_result.text == "L0: Subject: Example receipt"
    assert len(tool_result.web_constructs) == 1
    citation = tool_result.web_constructs[0]
    assert citation.construct_type.value == "content_reference"
    assert citation.title == "Example receipt #1"
    assert citation.url == "https://mail.example.test/all/anon"
    assert citation.source_id == "connector_anon"


def test_content_reference_grouped_webpages_items_become_constructs() -> None:
    """polylogue-zocm GAP 1: a ``grouped_webpages`` content_reference carries
    no url of its own -- every URL lives in ``items[]``/``fallback_items[]``.
    Anonymized from a real July-export node.
    """
    mapping = {
        "node1": {
            "id": "node1",
            "message": {
                "id": "msg1",
                "author": {"role": "assistant"},
                "content": {"content_type": "text", "parts": ["See sources."]},
                "metadata": {
                    "content_references": [
                        {
                            "type": "grouped_webpages",
                            "alt": "([Example](https://example.test/a))",
                            "items": [
                                {
                                    "title": "Example A",
                                    "url": "https://example.test/a",
                                    "attribution": "example.test",
                                    "snippet": "primary source",
                                }
                            ],
                            "fallback_items": [
                                {
                                    "title": "Example B",
                                    "url": "https://example.test/b",
                                    "snippet": "backup source",
                                }
                            ],
                        }
                    ]
                },
            },
        },
    }

    messages, _attachments = extract_messages_from_mapping(mapping)

    constructs = [
        c
        for b in messages[0].blocks
        for c in b.web_constructs
        if (c.provider_key or "").startswith("content_references")
    ]
    # The bare group envelope carries no url/title/text of its own -- only
    # the two nested entries should surface as constructs.
    assert len(constructs) == 2
    by_url = {c.url: c for c in constructs}
    assert by_url["https://example.test/a"].title == "Example A"
    assert by_url["https://example.test/a"].provider_key == "content_references.items"
    assert by_url["https://example.test/a"].group_title == "([Example](https://example.test/a))"
    assert by_url["https://example.test/b"].title == "Example B"
    assert by_url["https://example.test/b"].provider_key == "content_references.fallback_items"
    # Both nested entries share the same group_id -- they came from the same
    # content_reference envelope.
    assert by_url["https://example.test/a"].group_id == by_url["https://example.test/b"].group_id
    for construct in constructs:
        assert construct.construct_type.value == "content_reference"


# =============================================================================
# CODE-INTERPRETER RUN RECORD, SEARCH RESULTS, AUTHORSHIP AND TERMINAL STATE
# =============================================================================


def _tool_node(
    *,
    content: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
    status: str = "finished_successfully",
) -> ChatGPTMapping:
    return {
        "call": {
            "id": "call",
            "parent": None,
            "children": ["result"],
            "message": {
                "id": "call",
                "author": {"role": "assistant"},
                "recipient": "python",
                "create_time": 1,
                "content": {"content_type": "code", "text": ""},
            },
        },
        "result": {
            "id": "result",
            "parent": "call",
            "children": [],
            "message": {
                "id": "result",
                "author": {"role": "tool"},
                "create_time": 2,
                "status": status,
                "content": dict(content),
                "metadata": dict(metadata or {}),
            },
        },
    }


def _tool_result_block(messages: Sequence[ParsedMessage]) -> ParsedContentBlock:
    return next(block for message in messages for block in message.blocks if block.type is BlockType.TOOL_RESULT)


# A ChatGPT node reports message delivery in `status` and the run's own
# verdict in `metadata.aggregate_result`. Reading only `status` -- what the
# parser used to do -- makes every one of these cases `ok`.
_AGGREGATE_RESULT_OUTCOME_CASES: list[tuple[dict[str, object], bool | None, str | None, str]] = [
    ({"status": "success"}, False, None, "a completed run is a success"),
    (
        {"status": "failed_with_in_kernel_exception", "in_kernel_exception": {"name": "PermissionError"}},
        True,
        None,
        "an in-kernel exception is an error",
    ),
    ({"status": "cancelled"}, None, "unsupported_construct", "a cancelled run states no verdict"),
    ({"status": "running"}, None, "unsupported_construct", "an unfinished run states no verdict"),
    (
        {"status": "success", "timeout_triggered": 60},
        False,
        None,
        "timeout_triggered is an integer the wire sets on successful runs too",
    ),
]


@pytest.mark.parametrize(
    ("aggregate_result", "expected_is_error", "expected_unknown_reason", "desc"),
    _AGGREGATE_RESULT_OUTCOME_CASES,
    ids=[case[3] for case in _AGGREGATE_RESULT_OUTCOME_CASES],
)
def test_chatgpt_execution_output_outcome_reads_the_run_record(
    aggregate_result: dict[str, object],
    expected_is_error: bool | None,
    expected_unknown_reason: str | None,
    desc: str,
) -> None:
    messages, _attachments = extract_messages_from_mapping(
        _tool_node(
            content={"content_type": "execution_output", "text": "out"},
            metadata={"aggregate_result": aggregate_result},
        )
    )

    block = _tool_result_block(messages)
    assert block.is_error is expected_is_error, desc
    assert block.outcome_unknown_reason == expected_unknown_reason, desc


def test_chatgpt_tool_role_text_result_reads_the_run_record() -> None:
    """The role-dispatched carrier path reaches the same verdict as execution_output."""
    messages, _attachments = extract_messages_from_mapping(
        _tool_node(
            content={"content_type": "text", "parts": ["out"]},
            metadata={
                "aggregate_result": {
                    "status": "failed_with_in_kernel_exception",
                    "in_kernel_exception": {"name": "ValueError"},
                }
            },
        )
    )

    assert _tool_result_block(messages).is_error is True


def test_chatgpt_failed_run_materializes_as_an_error_outcome() -> None:
    """End to end: the derived tool_outcome, not just the parser's is_error."""
    from polylogue.core.enums import Origin, ToolOutcome
    from polylogue.sources.tool_outcomes import derive_tool_outcomes

    messages, _attachments = extract_messages_from_mapping(
        _tool_node(
            content={"content_type": "execution_output", "text": "Traceback"},
            metadata={
                "aggregate_result": {
                    "status": "failed_with_in_kernel_exception",
                    "in_kernel_exception": {"name": "PermissionError"},
                }
            },
        )
    )

    resolved = derive_tool_outcomes(list(messages), [], origin=Origin.CHATGPT_EXPORT)
    outcomes = [
        block.tool_outcome for message in resolved for block in message.blocks if block.type is BlockType.TOOL_RESULT
    ]
    assert outcomes == [ToolOutcome.ERROR]


def test_chatgpt_aggregate_result_conserves_the_executed_program() -> None:
    """The run's code exists only here: the calling node's own text is empty."""
    payload: ChatGPTMapping = {
        "id": "conv-run",
        "conversation_id": "conv-run",
        "create_time": 1.0,
        "mapping": _tool_node(
            content={"content_type": "execution_output", "text": "out\n"},
            metadata={
                "aggregate_result": {
                    "status": "success",
                    "code": "print('hello')",
                    "run_id": "run-1",
                    "start_time": 1.0,
                    "end_time": 2.0,
                    "timeout_triggered": 60,
                    "messages": [{"message_type": "stream", "stream_name": "stdout", "text": "out\n"}],
                }
            },
        ),
    }

    session = chatgpt_parse(payload, "conv-run")

    construct = next(
        construct
        for message in session.messages
        for block in message.blocks
        for construct in block.web_constructs
        if construct.provider_key == "aggregate_result"
    )
    assert construct.text == "print('hello')"
    assert construct.task_id == "run-1"
    assert construct.status == "success"

    event = next(event for event in session.session_events if event.event_type == "chatgpt_code_interpreter_run")
    assert event.source_message_provider_id == "result"
    assert event.payload["run_id"] == "run-1"
    assert event.payload["timeout_triggered"] == 60
    # The stream repeats the result node's own text, so only its size is kept.
    assert event.payload["stream_chars"] == len("out\n")
    assert event.payload["stream_retained_as_message_text"] is True
    assert "stream_text" not in event.payload


def test_chatgpt_run_stream_text_is_kept_when_it_is_not_the_message_text() -> None:
    payload: ChatGPTMapping = {
        "id": "conv-run",
        "conversation_id": "conv-run",
        "create_time": 1.0,
        "mapping": _tool_node(
            content={"content_type": "execution_output", "text": "rendered"},
            metadata={
                "aggregate_result": {
                    "status": "failed_with_in_kernel_exception",
                    "in_kernel_exception": {"name": "ValueError", "args": ["bad"]},
                    "jupyter_messages": [{"msg_type": "stream", "content": {"name": "stdout", "text": "raw"}}],
                }
            },
        ),
    }

    event = next(
        event
        for event in chatgpt_parse(payload, "conv-run").session_events
        if event.event_type == "chatgpt_code_interpreter_run"
    )
    assert event.payload["stream_text"] == "raw"
    assert event.payload["stream_retained_as_message_text"] is False
    assert event.payload["in_kernel_exception_name"] == "ValueError"
    assert event.payload["in_kernel_exception_args"] == ["bad"]


_SEARCH_RESULT_GROUP = {
    "type": "search_result_group",
    "domain": "example.test",
    "entries": [
        {
            "type": "search_result",
            "url": "https://example.test/page",
            "title": "Page",
            "snippet": "A snippet.",
            "ref_id": {"turn_index": 0, "ref_type": "search", "ref_index": 4},
            "attribution": "example.test",
        }
    ],
}


@pytest.mark.parametrize(
    ("metadata", "expected_provider_key"),
    [
        ({"search_result_groups": [_SEARCH_RESULT_GROUP]}, "search_result_groups"),
        (
            {"inline_cot_expandable_content": {"search_result_groups": [_SEARCH_RESULT_GROUP]}},
            "inline_cot_expandable_content.search_result_groups",
        ),
    ],
    ids=["answer metadata", "reasoning-trace metadata"],
)
def test_chatgpt_search_result_group_entries_become_search_results(
    metadata: dict[str, object], expected_provider_key: str
) -> None:
    """`entries` is the wire's key; reading only results/items/sources drops every result."""
    messages, _attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Answer"]},
                    "metadata": metadata,
                },
            }
        }
    )

    results = [
        construct
        for block in messages[0].blocks
        for construct in block.web_constructs
        if construct.construct_type.value == "search_result"
    ]
    assert len(results) == 1
    assert results[0].provider_key == expected_provider_key
    assert results[0].url == "https://example.test/page"
    assert results[0].title == "Page"
    assert results[0].text == "A snippet."
    assert results[0].group_title == "example.test"
    # The ref_id triple spells the token the answer's inline citation markers
    # carry, which is what joins a citation anchor back to its result.
    assert results[0].source_id == "turn0search4"


_FINISH_DETAILS_CASES: list[tuple[object, str | None, str]] = [
    ({"type": "stop", "stop_tokens": [200002]}, "end_turn", "a natural stop is end_turn"),
    ({"type": "max_tokens"}, "max_tokens", "a length cut is max_tokens"),
    ({"type": "interrupted"}, None, "interruption names no StopReason member"),
    ({"type": "skipped"}, None, "a skipped turn names no StopReason member"),
    (None, None, "a node with no finish details reports nothing"),
]


@pytest.mark.parametrize(
    ("finish_details", "expected", "desc"),
    _FINISH_DETAILS_CASES,
    ids=[case[2] for case in _FINISH_DETAILS_CASES],
)
def test_chatgpt_finish_details_map_onto_stop_reason(finish_details: object, expected: str | None, desc: str) -> None:
    metadata: dict[str, object] = {} if finish_details is None else {"finish_details": finish_details}
    messages, _attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Answer"]},
                    "metadata": metadata,
                },
            }
        }
    )

    assert messages[0].stop_reason == expected, desc


def test_chatgpt_channel_and_real_author_are_conserved() -> None:
    """A tool-authored message in an assistant envelope is tool material, not model output."""
    payload: ChatGPTMapping = {
        "id": "conv-auth",
        "conversation_id": "conv-auth",
        "create_time": 1.0,
        "mapping": {
            "node-1": {
                "id": "node-1",
                "parent": None,
                "children": [],
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant", "metadata": {"real_author": "tool:web.run"}},
                    "channel": "commentary",
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Retrieved page"]},
                },
            }
        },
    }

    session = chatgpt_parse(payload, "conv-auth")

    assert session.messages[0].material_origin is MaterialOrigin.TOOL_RESULT
    event = next(event for event in session.session_events if event.event_type == "chatgpt_message_authorship")
    assert event.source_message_provider_id == "msg-1"
    assert event.payload == {"channel": "commentary", "real_author": "tool:web.run"}


def test_chatgpt_ordinary_assistant_message_stays_model_output() -> None:
    """Anti-vacuity for the override above: no real_author, no reclassification."""
    messages, _attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "channel": "final",
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Answer"]},
                },
            }
        }
    )

    assert messages[0].material_origin is MaterialOrigin.ASSISTANT_AUTHORED


def test_chatgpt_memory_citation_conserves_the_cited_conversation() -> None:
    messages, _attachments = extract_messages_from_mapping(
        {
            "node-1": {
                "id": "node-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "assistant"},
                    "create_time": 1,
                    "content": {"content_type": "text", "parts": ["Answer"]},
                    "metadata": {
                        "conversation_context_citation_metadata": [
                            {
                                "citation_uuid": "cite-1",
                                "retrieval_origin": "pca",
                                "citation": {
                                    "url": "https://chatgpt.com/c/11111111-2222-3333-4444-555555555555",
                                    "conversation_title": "Earlier chat",
                                    "snippet": "what we decided",
                                    "category": "memory",
                                    "start_idx": 3,
                                    "end_idx": 9,
                                },
                            }
                        ]
                    },
                },
            }
        }
    )

    citation = next(
        construct
        for block in messages[0].blocks
        for construct in block.web_constructs
        if construct.provider_key == "conversation_context_citation_metadata"
    )
    assert citation.url == "https://chatgpt.com/c/11111111-2222-3333-4444-555555555555"
    assert citation.title == "Earlier chat"
    assert citation.text == "what we decided"
    assert citation.group_title == "memory"
    assert citation.start_index == 3
    assert citation.end_index == 9
    # The cited conversation's own native id: the join key an archive-internal
    # edge needs, kept resolvable without reparsing the source.
    assert citation.source_id == "11111111-2222-3333-4444-555555555555"


# -----------------------------------------------------------------------------
# REDUCED EXPORT SHAPE AND METADATA DISPOSITIONS (bd polylogue-rp91m/-vnucj)
# -----------------------------------------------------------------------------


def _reduced_node(
    node_id: str,
    role: str,
    text: str,
    parent: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A node in the reduced export shape: no ``children``, no ``status``.

    The 2026-07 export ships ``{id, message, parent}`` nodes whose messages
    carry only ``{author, content, create_time, id, metadata}`` — measured
    over all 29 shards (75,453 nodes, 0 with ``children``).
    """
    message: dict[str, Any] = {
        "id": node_id,
        "author": {"role": role},
        "create_time": 1_700_000_000.0,
        "content": {"content_type": "text", "parts": [text]},
    }
    if metadata is not None:
        message["metadata"] = metadata
    return {"id": node_id, "parent": parent, "message": message}


def test_chatgpt_branch_index_survives_missing_children_array() -> None:
    """Siblings keep distinct branch indexes when the export omits ``children``.

    Anti-vacuity: delete the ``_sibling_ordinals`` fallback and every
    regenerated alternative collapses to ``branch_index == 0``, which is what
    the reduced export shape produced before it existed.
    """
    messages, _ = extract_messages_from_mapping(
        {
            "root": _reduced_node("root", "user", "Question"),
            "first": _reduced_node("first", "assistant", "Answer 1", parent="root"),
            "second": _reduced_node("second", "assistant", "Answer 2", parent="root"),
            "third": _reduced_node("third", "assistant", "Answer 3", parent="root"),
        }
    )

    by_id = {message.provider_message_id: message for message in messages}
    assert by_id["root"].branch_index == 0
    assert [by_id[node].branch_index for node in ("first", "second", "third")] == [0, 1, 2]


def test_chatgpt_children_array_outranks_arrival_order() -> None:
    """``children`` states sibling order; the parent-edge fallback yields to it."""
    messages, _ = extract_messages_from_mapping(
        {
            "root": make_chatgpt_node("root", "user", ["Q"], children=["second", "first"]),
            "first": make_chatgpt_node("first", "assistant", ["A1"], parent="root"),
            "second": make_chatgpt_node("second", "assistant", ["A2"], parent="root"),
        }
    )

    by_id = {message.provider_message_id: message for message in messages}
    assert by_id["first"].branch_index == 1
    assert by_id["second"].branch_index == 0


def test_chatgpt_reduced_shape_tool_outcome_stays_not_reported() -> None:
    """No ``status`` field means no outcome verdict, never a guessed one."""
    mapping = {
        "call": _reduced_node("call", "assistant", "print(1)"),
        "out": {
            "id": "out",
            "parent": "call",
            "message": {
                "id": "out",
                "author": {"role": "tool"},
                "create_time": 1_700_000_000.0,
                "content": {"content_type": "execution_output", "text": "1"},
            },
        },
    }
    messages, _ = extract_messages_from_mapping(mapping)

    result = next(block for message in messages for block in message.blocks if block.type is BlockType.TOOL_RESULT)
    assert result.is_error is None
    assert result.outcome_unknown_reason == "not_reported"


def test_chatgpt_command_and_args_carry_a_tool_call_without_recipient() -> None:
    """``metadata.command``/``args`` name the tool when ``recipient`` is absent.

    Anti-vacuity: drop the metadata fallback and this message lowers to a
    plain TEXT block — the shape 4,303 of 109,657 messages in the 2026-04
    export take, since they state ``command`` and no ``recipient``.
    """
    messages, _ = extract_messages_from_mapping(
        {
            "root": _reduced_node("root", "user", "find it"),
            "call": _reduced_node(
                "call",
                "assistant",
                "not json",
                parent="root",
                metadata={"command": "search", "args": ["interception tools"]},
            ),
        }
    )

    call = next(message for message in messages if message.provider_message_id == "call")
    block = call.blocks[0]
    assert block.type is BlockType.TOOL_USE
    assert block.tool_name == "search"
    assert block.tool_input == {"args": ["interception tools"]}


def test_chatgpt_finish_details_maps_only_exact_stop_reasons() -> None:
    """``interrupted``/``content_filter`` have no StopReason equivalent."""
    observed = {}
    for finish_type in ("stop", "max_tokens", "interrupted", "content_filter", "unknown"):
        messages, _ = extract_messages_from_mapping(
            {
                "turn": _reduced_node(
                    "turn",
                    "assistant",
                    "answer",
                    metadata={"finish_details": {"type": finish_type}},
                )
            }
        )
        observed[finish_type] = messages[0].stop_reason

    assert observed == {
        "stop": "end_turn",
        "max_tokens": "max_tokens",
        "interrupted": None,
        "content_filter": None,
        "unknown": None,
    }


def test_chatgpt_model_name_falls_back_through_default_slugs() -> None:
    """``model_slug`` wins, then message ``default_model_slug``, then the conversation's."""
    payload = {
        "id": "conv-models",
        "conversation_id": "conv-models",
        "create_time": 1_700_000_000.0,
        "current_node": "bare",
        "title": "Models",
        "default_model_slug": "gpt-5-pro",
        "mapping": {
            "asked": _reduced_node("asked", "user", "Q"),
            "exact": _reduced_node("exact", "assistant", "A", parent="asked", metadata={"model_slug": "o3"}),
            "defaulted": _reduced_node(
                "defaulted", "assistant", "A", parent="asked", metadata={"default_model_slug": "gpt-4o"}
            ),
            "bare": _reduced_node("bare", "assistant", "A", parent="asked"),
        },
    }

    session = chatgpt_parse(payload, "fallback")

    by_id = {message.provider_message_id: message for message in session.messages}
    assert by_id["exact"].model_name == "o3"
    assert by_id["defaulted"].model_name == "gpt-4o"
    assert by_id["bare"].model_name == "gpt-5-pro"
    assert by_id["asked"].model_name is None
    assert session.models_used == ["gpt-4o", "gpt-5-pro", "o3"]


def test_chatgpt_message_metadata_reaches_named_events() -> None:
    """targeted_reply, delivery state and plugin payloads each get their own event."""
    payload = {
        "id": "conv-evidence",
        "conversation_id": "conv-evidence",
        "create_time": 1_700_000_000.0,
        "current_node": "reply",
        "title": "Evidence",
        "mapping": {
            "reply": {
                "id": "reply",
                "parent": None,
                "children": [],
                "message": {
                    "id": "reply",
                    "author": {"role": "user", "metadata": {"real_author": "tool:web.run"}},
                    "create_time": 1_700_000_000.0,
                    "weight": 0.0,
                    "channel": "commentary",
                    "content": {"content_type": "text", "parts": ["answering"]},
                    "metadata": {
                        "targeted_reply": "the words I replied to",
                        "is_visually_hidden_from_conversation": True,
                        "jit_plugin_data": {"from_server": {"type": "preview"}},
                    },
                },
            }
        },
    }

    session = chatgpt_parse(payload, "fallback")

    events = {event.event_type: event for event in session.session_events}
    assert events["chatgpt_targeted_reply"].payload == {"targeted_reply": "the words I replied to"}
    assert events["chatgpt_message_delivery"].payload == {
        "weight": 0.0,
        "is_visually_hidden_from_conversation": True,
        "channel": "commentary",
    }
    assert events["chatgpt_jit_plugin_data"].payload == {"from_server": {"type": "preview"}}
    assert all(event.source_message_provider_id == "reply" for event in events.values())
    assert session.messages[0].sender_name == "tool:web.run"


def test_chatgpt_ada_visualizations_become_attachments() -> None:
    """Code-interpreter charts and tables are real files with their own ids."""
    _messages, attachments = extract_messages_from_mapping(
        {
            "analysis": _reduced_node(
                "analysis",
                "assistant",
                "here is the table",
                metadata={
                    "ada_visualizations": [
                        {"type": "table", "file_id": "file-XRfd", "title": "Invoices — preview"},
                        {"type": "chart", "title": "no file id, no attachment"},
                    ]
                },
            )
        }
    )

    assert [(a.provider_file_id, a.name, a.attachment_kind) for a in attachments] == [
        ("file-XRfd", "Invoices — preview", "ada_visualization")
    ]
    assert attachments[0].message_provider_id == "analysis"


def test_chatgpt_dalle_provenance_rides_the_image_block() -> None:
    """The edge from a derived image back to its original exists nowhere else."""
    provenance = {"from_client": {"operation": {"type": "transformation", "original_gen_id": "gen-1"}}}
    messages, _ = extract_messages_from_mapping(
        {
            "image": {
                "id": "image",
                "parent": None,
                "message": {
                    "id": "image",
                    "author": {"role": "assistant"},
                    "create_time": 1_700_000_000.0,
                    "content": {
                        "content_type": "multimodal_text",
                        "parts": [{"content_type": "image_asset_pointer", "asset_pointer": "file-service://abc"}],
                    },
                    "metadata": {"dalle": provenance},
                },
            }
        }
    )

    block = next(block for block in messages[0].blocks if block.type is BlockType.IMAGE)
    assert block.metadata == {"asset_pointer": "file-service://abc", "dalle": provenance}


def test_chatgpt_conversation_settings_skip_defaults_and_name_the_custom_gpt() -> None:
    """A bare ``g-<id>`` is a custom GPT, not a project, and reaches its own event."""
    payload = {
        "id": "conv-gizmo",
        "conversation_id": "conv-gizmo",
        "create_time": 1_700_000_000.0,
        "current_node": "root",
        "title": "Gizmo",
        "conversation_template_id": "g-bo0FiWLY7",
        "gizmo_type": "gpt",
        "is_archived": True,
        "is_starred": False,
        "moderation_results": [],
        "memory_scope": "global_enabled",
        "mapping": {"root": _reduced_node("root", "user", "hi")},
    }

    session = chatgpt_parse(payload, "fallback")

    events = {event.event_type: event.payload for event in session.session_events}
    assert session.provider_project_ref is None
    assert events["chatgpt_custom_gpt"] == {"gizmo_id": "g-bo0FiWLY7", "gizmo_type": "gpt"}
    # False, empty and account-wide-default settings carry no signal.
    assert events["chatgpt_conversation_settings"] == {"is_archived": True, "gizmo_type": "gpt"}


def test_chatgpt_project_conversation_emits_no_custom_gpt_event() -> None:
    """``g-p-`` is already ``provider_project_ref``; it is not a custom GPT."""
    payload = {
        "id": "conv-project",
        "conversation_id": "conv-project",
        "create_time": 1_700_000_000.0,
        "current_node": "root",
        "title": "Project",
        "conversation_template_id": "g-p-6801608e3ebc819184f4e318bf49f5ff",
        "mapping": {"root": _reduced_node("root", "user", "hi")},
    }

    session = chatgpt_parse(payload, "fallback")

    assert session.provider_project_ref == "g-p-6801608e3ebc819184f4e318bf49f5ff"
    assert not [event for event in session.session_events if event.event_type == "chatgpt_custom_gpt"]
