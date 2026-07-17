"""Composed normalization laws for Gemini CLI and AI Studio/Drive wires.

These tests enter through ``detect_provider`` and ``parse_payload`` rather than
calling parser helpers directly. Fixtures are synthetic, privacy-safe records
shaped from the checked-in generated provider schemas. Expected facts are
literal assertions, not values projected back out of the input fixture.
"""

from __future__ import annotations

from copy import deepcopy

from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, BranchType, MaterialOrigin, Origin, Provider
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.core.sources import origin_from_provider, provider_to_source
from polylogue.sources.dispatch import detect_provider, parse_payload
from polylogue.sources.parsers.base import ParsedMessage


def _ai_studio_payload() -> JSONDocument:
    return {
        "id": "conversation-safe-alpha",
        "title": "Normalization proof",
        "createTime": "2026-06-01T10:00:00Z",
        "updateTime": "2026-06-01T10:00:05Z",
        "runSettings": {
            "model": "models/gemini-safe-test",
            "temperature": 0.25,
            "thinkingBudget": 64,
            "enableCodeExecution": True,
        },
        "systemInstruction": {"text": "Use only fixture-safe data."},
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "turn-user-native",
                    "role": "user",
                    "createTime": "2026-06-01T10:00:01Z",
                    "text": "Inspect the structured fixture.",
                    "tokenCount": 7,
                    "branchChildren": [{"promptId": "turn-thought-native"}],
                    "driveDocument": {
                        "id": "drive-doc-native",
                        "name": "notes.txt",
                        "mimeType": "text/plain",
                        "sizeBytes": "12",
                    },
                    "driveImage": {"id": "drive-image-native", "mimeType": "image/png"},
                    "inlineImage": {"mimeType": "image/png", "data": "aW1hZ2U="},
                },
                {
                    "id": "turn-thought-native",
                    "role": "model",
                    "createTime": "2026-06-01T10:00:02Z",
                    "isThought": True,
                    "text": "Check identity before content.",
                    "thinkingBudget": 64,
                    "thoughtSignatures": ["signature-safe-top"],
                    "tokenCount": 3,
                },
                {
                    "id": "turn-code-native",
                    "role": "model",
                    "createTime": "2026-06-01T10:00:03Z",
                    "parts": [
                        {"text": "Run a bounded calculation."},
                        {
                            "text": "One calculation is sufficient.",
                            "thought": True,
                            "thoughtSignature": "signature-safe-part",
                        },
                        {"executableCode": {"language": "PYTHON", "code": "print(2 + 3)"}},
                        {"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "5\n"}},
                        {"inlineData": {"mimeType": "image/png", "data": "cGFydA=="}},
                        {
                            "fileData": {
                                "mimeType": "text/plain",
                                "fileUri": "drive://fixture-safe-part",
                                "displayName": "part-note.txt",
                            }
                        },
                    ],
                    "tokenCount": 11,
                    "finishReason": "STOP",
                    "branchParent": {"promptId": "turn-thought-native"},
                },
                {
                    "id": "turn-failed-result-native",
                    "role": "model",
                    "createTime": "2026-06-01T10:00:04Z",
                    "codeExecutionResult": {"outcome": "OUTCOME_FAILED", "output": "bounded failure"},
                    "tokenCount": 2,
                    "finishReason": "OTHER",
                    "branchParent": {"promptId": "turn-code-native"},
                },
                {
                    "id": "turn-error-native",
                    "role": "model",
                    "createTime": "2026-06-01T10:00:05Z",
                    "errorMessage": "execution unavailable",
                    "tokenCount": 1,
                    "driveAudio": {"id": "drive-audio-native", "mimeType": "audio/wav"},
                    "driveVideo": {"id": "drive-video-native", "mimeType": "video/mp4"},
                    "inlineFile": {"mimeType": "application/pdf", "data": "cGRm"},
                },
            ]
        },
    }


def _gemini_cli_payload() -> JSONDocument:
    return {
        "sessionId": "cli-session-safe",
        "projectHash": "project-safe",
        "startTime": "2026-06-02T09:00:00Z",
        "lastUpdated": "2026-06-02T09:00:03Z",
        "kind": "main",
        "summary": "CLI normalization proof",
        "directories": ["/workspace/safe-alpha", "/workspace/safe-shared"],
        "messages": [
            {
                "id": "cli-user-native",
                "timestamp": "2026-06-02T09:00:01Z",
                "type": "user",
                "content": [{"text": "Read the fixture."}],
                "tokens": {"input": 12, "output": 0, "cached": 4, "thoughts": 0, "tool": 0, "total": 12},
            },
            {
                "id": "cli-assistant-native",
                "timestamp": "2026-06-02T09:00:02Z",
                "type": "gemini",
                "content": "Checking the fixture.",
                "model": "gemini-safe-cli",
                "thoughts": [
                    {
                        "subject": "Plan",
                        "description": "Inspect the typed fields.",
                        "timestamp": "2026-06-02T09:00:01.500Z",
                    }
                ],
                "tokens": {"input": 20, "output": 5, "cached": 8, "thoughts": 3, "tool": 2, "total": 30},
                "toolCalls": [
                    {
                        "id": "tool-success-native",
                        "name": "read_file",
                        "args": {"file_path": "fixture.txt"},
                        "status": "success",
                        "timestamp": "2026-06-02T09:00:02.100Z",
                        "displayName": "ReadFile",
                        "result": [
                            {
                                "functionResponse": {
                                    "id": "tool-success-native",
                                    "name": "read_file",
                                    "response": {"output": "safe contents"},
                                }
                            }
                        ],
                    },
                    {
                        "id": "tool-failure-native",
                        "name": "shell",
                        "args": {"command": "exit 1"},
                        "status": "error",
                        "timestamp": "2026-06-02T09:00:02.200Z",
                        "displayName": "Shell",
                        "result": [
                            {
                                "functionResponse": {
                                    "id": "tool-failure-native",
                                    "name": "shell",
                                    "response": {"error": "command failed"},
                                }
                            }
                        ],
                    },
                    {
                        "id": "tool-status-only-native",
                        "name": "checkpoint",
                        "args": {},
                        "status": "success",
                        "timestamp": "2026-06-02T09:00:02.300Z",
                    },
                ],
            },
        ],
    }


def _small_ai_studio_session(session_id: str, text: str, *, top_level_chunks: bool = False) -> JSONDocument:
    chunks: list[JSONValue] = [{"role": "user", "text": text, "tokenCount": 1}]
    payload: JSONDocument = {"id": session_id, "runSettings": {"model": "models/gemini-nested-safe"}}
    if top_level_chunks:
        payload["chunks"] = chunks
    else:
        payload["chunkedPrompt"] = {"chunks": chunks}
    return payload


def test_detector_order_is_tight_and_one_document_streams_keep_specificity() -> None:
    """Detector-order mutation: moving the weak Gemini check upward must fail."""
    valid_chunk = {"role": "user", "text": "safe"}
    chatgpt_ambiguous = {"mapping": {}, "chunks": [valid_chunk]}
    cli_ambiguous = {**_gemini_cli_payload(), "chunks": [valid_chunk]}

    assert detect_provider({"chunks": []}) is None
    assert detect_provider({"chunks": [{"role": "user", "timestamp": "2026-06-01T00:00:00Z"}]}) is None
    assert detect_provider({"chunks": [valid_chunk]}) is Provider.GEMINI
    assert detect_provider({"chunkedPrompt": True, "chunks": [valid_chunk]}) is Provider.GEMINI
    assert detect_provider(chatgpt_ambiguous) is Provider.CHATGPT
    assert detect_provider(cli_ambiguous) is Provider.GEMINI_CLI
    assert detect_provider([cli_ambiguous]) is Provider.GEMINI_CLI


def test_provider_tokens_survive_the_non_injective_origin_collapse() -> None:
    """Identity-collapse mutation: canonicalizing source_name from Origin must fail."""
    payload = _ai_studio_payload()
    [gemini_session] = parse_payload(Provider.GEMINI, payload, "gemini-fallback")
    [drive_session] = parse_payload(Provider.DRIVE, payload, "drive-fallback")

    assert gemini_session.source_name is Provider.GEMINI
    assert drive_session.source_name is Provider.DRIVE
    assert provider_to_source(gemini_session.source_name).family == "gemini-export"
    assert provider_to_source(drive_session.source_name).family == "drive-takeout"
    assert origin_from_provider(gemini_session.source_name) is Origin.AISTUDIO_DRIVE
    assert origin_from_provider(drive_session.source_name) is Origin.AISTUDIO_DRIVE
    assert gemini_session.provider_session_id == drive_session.provider_session_id == "conversation-safe-alpha"


def test_ai_studio_normalizes_identity_authorship_config_blocks_artifacts_usage_and_status() -> None:
    """Authoredness, artifact-loss, and thought/tool-outcome mutations must fail."""
    [session] = parse_payload(Provider.GEMINI, _ai_studio_payload(), "unused-fallback")
    by_id = {message.provider_message_id: message for message in session.messages}

    assert session.title == "Normalization proof"
    assert session.created_at == "2026-06-01T10:00:00Z"
    assert session.updated_at == "2026-06-01T10:00:05Z"
    assert session.instructions_text == "Use only fixture-safe data."
    assert session.models_used == ["models/gemini-safe-test"]
    assert list(by_id) == [
        "turn-user-native",
        "turn-thought-native",
        "turn-code-native",
        "turn-failed-result-native",
        "turn-error-native",
    ]

    user = by_id["turn-user-native"]
    assert user.material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert user.input_tokens == 7
    assert user.output_tokens == 0
    assert user.model_name == "models/gemini-safe-test"

    thought = by_id["turn-thought-native"]
    assert thought.parent_message_provider_id == "turn-user-native"
    assert thought.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert thought.output_tokens == 3
    thought_block = next(block for block in thought.blocks if block.type is BlockType.THINKING)
    assert thought_block.metadata == {
        "isThought": True,
        "thinkingBudget": 64,
        "thoughtSignatures": ["signature-safe-top"],
    }

    code = by_id["turn-code-native"]
    assert code.parent_message_provider_id == "turn-thought-native"
    assert code.delivery_status == "STOP"
    assert code.end_turn is True
    assert code.output_tokens == 11
    assert {block.type for block in code.blocks} >= {
        BlockType.TEXT,
        BlockType.THINKING,
        BlockType.CODE,
        BlockType.TOOL_RESULT,
        BlockType.DOCUMENT,
    }
    assert sum(block.type is BlockType.DOCUMENT for block in code.blocks) == 2
    assert next(block for block in code.blocks if block.type is BlockType.CODE).text == "print(2 + 3)"
    successful_result = next(block for block in code.blocks if block.type is BlockType.TOOL_RESULT)
    assert successful_result.text == "5\n"
    assert successful_result.is_error is False
    part_thought = next(block for block in code.blocks if block.type is BlockType.THINKING)
    assert part_thought.metadata and part_thought.metadata["thoughtSignature"] == "signature-safe-part"
    inline_part = next(
        block
        for block in code.blocks
        if block.type is BlockType.DOCUMENT and block.metadata and "inlineData" in block.metadata
    )
    assert inline_part.metadata == {"inlineData": {"mimeType": "image/png"}}
    linked_part = next(
        block
        for block in code.blocks
        if block.type is BlockType.DOCUMENT and block.metadata and "fileData" in block.metadata
    )
    assert linked_part.metadata == {
        "fileData": {
            "mimeType": "text/plain",
            "fileUri": "drive://fixture-safe-part",
            "displayName": "part-note.txt",
        },
        "url": "drive://fixture-safe-part",
    }

    failed_result_message = by_id["turn-failed-result-native"]
    assert failed_result_message.message_type is MessageType.TOOL_RESULT
    assert failed_result_message.material_origin is MaterialOrigin.TOOL_RESULT
    failed_result = failed_result_message.blocks[0]
    assert failed_result.type is BlockType.TOOL_RESULT
    assert failed_result.is_error is True

    error_message = by_id["turn-error-native"]
    assert error_message.delivery_status == "error"
    assert error_message.end_turn is True
    assert error_message.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert any(block.text == "execution unavailable" for block in error_message.blocks)

    model_event = next(event for event in session.session_events if event.event_type == "model_config")
    assert model_event.payload == {
        "model": "models/gemini-safe-test",
        "runSettings": {
            "model": "models/gemini-safe-test",
            "temperature": 0.25,
            "thinkingBudget": 64,
            "enableCodeExecution": True,
        },
    }
    token_events = [event for event in session.session_events if event.event_type == "token_count"]
    assert len(token_events) == 5
    assert token_events[0].payload["last_token_usage"] == {"input_tokens": 7}
    assert token_events[2].payload["last_token_usage"] == {"output_tokens": 11}

    attachment_by_id = {attachment.provider_attachment_id: attachment for attachment in session.attachments}
    assert {"drive-doc-native", "drive-image-native", "drive-audio-native", "drive-video-native"} <= set(
        attachment_by_id
    )
    assert attachment_by_id["drive-doc-native"].provider_file_id == "drive-doc-native"
    assert attachment_by_id["drive-image-native"].attachment_kind == "drive_image"
    assert attachment_by_id["drive-audio-native"].attachment_kind == "drive_audio"
    assert attachment_by_id["drive-video-native"].attachment_kind == "drive_video"
    assert {attachment.attachment_kind for attachment in session.attachments} >= {
        "inline_image",
        "inline_data",
        "inline_file",
        "file_data",
    }
    file_data_attachment = next(
        attachment for attachment in session.attachments if attachment.attachment_kind == "file_data"
    )
    assert file_data_attachment.source_url == "drive://fixture-safe-part"
    assert file_data_attachment.name == "part-note.txt"
    assert len(session.attachments) == 8


def test_native_turn_facts_survive_reordering_while_missing_ids_use_source_position() -> None:
    payload = _ai_studio_payload()
    [ordered] = parse_payload(Provider.GEMINI, payload, "ordered")
    reordered_payload = deepcopy(payload)
    prompt = reordered_payload["chunkedPrompt"]
    assert isinstance(prompt, dict)
    chunks = prompt["chunks"]
    assert isinstance(chunks, list)
    chunks.reverse()
    [reordered] = parse_payload(Provider.GEMINI, reordered_payload, "reordered")

    expected_facts: dict[str, tuple[object, ...]] = {
        "turn-user-native": (
            Role.USER,
            "2026-06-01T10:00:01Z",
            None,
            7,
            0,
            (BlockType.TEXT, BlockType.DOCUMENT, BlockType.IMAGE, BlockType.IMAGE),
        ),
        "turn-thought-native": (
            Role.ASSISTANT,
            "2026-06-01T10:00:02Z",
            "turn-user-native",
            0,
            3,
            (BlockType.THINKING,),
        ),
        "turn-code-native": (
            Role.ASSISTANT,
            "2026-06-01T10:00:03Z",
            "turn-thought-native",
            0,
            11,
            (
                BlockType.TEXT,
                BlockType.THINKING,
                BlockType.CODE,
                BlockType.TOOL_RESULT,
                BlockType.DOCUMENT,
                BlockType.DOCUMENT,
            ),
        ),
        "turn-failed-result-native": (
            Role.ASSISTANT,
            "2026-06-01T10:00:04Z",
            "turn-code-native",
            0,
            2,
            (BlockType.TOOL_RESULT,),
        ),
        "turn-error-native": (
            Role.ASSISTANT,
            "2026-06-01T10:00:05Z",
            None,
            0,
            1,
            (BlockType.DOCUMENT, BlockType.DOCUMENT, BlockType.DOCUMENT, BlockType.TEXT),
        ),
    }

    def native_facts(session_messages: list[ParsedMessage]) -> dict[str, tuple[object, ...]]:
        return {
            message.provider_message_id: (
                message.role,
                message.timestamp,
                message.parent_message_provider_id,
                message.input_tokens,
                message.output_tokens,
                tuple(block.type for block in message.blocks),
            )
            for message in session_messages
        }

    assert native_facts(ordered.messages) == expected_facts
    assert native_facts(reordered.messages) == expected_facts

    missing_id_payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "first"},
                {"role": "model", "text": "second"},
            ]
        }
    }
    [missing_id_session] = parse_payload(Provider.GEMINI, missing_id_payload, "missing-id-session")
    assert missing_id_session.provider_session_id == "missing-id-session"
    assert [message.provider_message_id for message in missing_id_session.messages] == ["chunk-1", "chunk-2"]


def test_ambiguous_branch_child_declarations_do_not_invent_a_parent() -> None:
    """Identity mutation: selecting the first declared parent must fail."""
    payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "parent-safe-a",
                    "role": "user",
                    "text": "first possible parent",
                    "branchChildren": ["child-safe"],
                },
                {
                    "id": "parent-safe-b",
                    "role": "user",
                    "text": "second possible parent",
                    "branchChildren": [{"promptId": "child-safe"}],
                },
                {"id": "child-safe", "role": "model", "text": "ambiguous child"},
            ]
        }
    }

    [ordered] = parse_payload(Provider.GEMINI, payload, "ambiguous-ordered")
    reversed_payload = deepcopy(payload)
    prompt = reversed_payload["chunkedPrompt"]
    assert isinstance(prompt, dict)
    chunks = prompt["chunks"]
    assert isinstance(chunks, list)
    chunks.reverse()
    [reversed_session] = parse_payload(Provider.GEMINI, reversed_payload, "ambiguous-reversed")

    for session in (ordered, reversed_session):
        child = next(message for message in session.messages if message.provider_message_id == "child-safe")
        assert child.parent_message_provider_id is None


def test_nested_drive_like_wrappers_lower_through_the_production_dispatch_route() -> None:
    """Artifact-loss mutation: skipping list values under sessions must fail."""
    first = _small_ai_studio_session("nested-session-one", "first nested turn")
    second = _small_ai_studio_session("nested-session-two", "second nested turn", top_level_chunks=True)
    nested_payload: list[object] = [{"sessions": [[first], {"sessions": [[second]]}]}]

    sessions = parse_payload(Provider.DRIVE, nested_payload, "nested-root")

    assert [session.provider_session_id for session in sessions] == ["nested-session-one", "nested-session-two"]
    assert [session.source_name for session in sessions] == [Provider.DRIVE, Provider.DRIVE]
    assert [session.messages[0].text for session in sessions] == ["first nested turn", "second nested turn"]
    assert [session.messages[0].provider_message_id for session in sessions] == ["chunk-1", "chunk-1"]
    assert [session.models_used for session in sessions] == [
        ["models/gemini-nested-safe"],
        ["models/gemini-nested-safe"],
    ]


def test_gemini_cli_schema_fields_survive_dispatch_without_export_authorship_upgrade() -> None:
    """Authoredness and thought/tool outcome mutations must fail on runtime data."""
    payload = _gemini_cli_payload()
    assert detect_provider(payload) is Provider.GEMINI_CLI
    assert detect_provider([payload]) is Provider.GEMINI_CLI

    [session] = parse_payload(Provider.GEMINI_CLI, [payload], "unused-fallback")
    assert session.provider_session_id == "cli-session-safe"
    assert session.title == "CLI normalization proof"
    assert session.created_at == "2026-06-02T09:00:00Z"
    assert session.updated_at == "2026-06-02T09:00:03Z"
    assert session.models_used == ["gemini-safe-cli"]
    assert session.provider_project_ref == "project-safe"
    assert session.branch_type is None
    assert session.working_directories == ["/workspace/safe-alpha", "/workspace/safe-shared"]

    user, assistant = session.messages
    assert user.provider_message_id == "cli-user-native"
    assert user.material_origin is MaterialOrigin.UNKNOWN
    assert user.input_tokens == 8
    assert user.cache_read_tokens == 4
    assert user.output_tokens == 0

    assert assistant.provider_message_id == "cli-assistant-native"
    assert assistant.material_origin is MaterialOrigin.ASSISTANT_AUTHORED
    assert assistant.input_tokens == 12
    assert assistant.cache_read_tokens == 8
    assert assistant.output_tokens == 5
    thinking = next(block for block in assistant.blocks if block.type is BlockType.THINKING)
    assert thinking.text == "Inspect the typed fields."
    assert thinking.metadata == {
        "index": 1,
        "subject": "Plan",
        "timestamp": "2026-06-02T09:00:01.500Z",
    }

    tool_uses = [block for block in assistant.blocks if block.type is BlockType.TOOL_USE]
    tool_results = [block for block in assistant.blocks if block.type is BlockType.TOOL_RESULT]
    assert [(block.tool_id, block.tool_input) for block in tool_uses] == [
        ("tool-success-native", {"file_path": "fixture.txt"}),
        ("tool-failure-native", {"command": "exit 1"}),
        ("tool-status-only-native", {}),
    ]
    assert [(block.tool_id, block.text, block.is_error) for block in tool_results] == [
        ("tool-success-native", "safe contents", False),
        ("tool-failure-native", "command failed", True),
        ("tool-status-only-native", "[success]", False),
    ]

    usage_by_message = {
        event.source_message_provider_id: event.payload
        for event in session.session_events
        if event.event_type == "message_usage"
    }
    assert usage_by_message["cli-assistant-native"]["last_token_usage"] == {
        "input_tokens": 12,
        "output_tokens": 5,
        "cached_input_tokens": 8,
        "cache_write_tokens": 0,
        "reasoning_output_tokens": 3,
        "total_tokens": 30,
    }
    assert usage_by_message["cli-assistant-native"]["tool_output_tokens"] == 2


def test_gemini_cli_subagent_kind_survives_as_typed_branch_semantics() -> None:
    """Identity mutation: dropping the subagent session kind must fail."""
    payload = _gemini_cli_payload()
    payload["sessionId"] = "cli-subagent-safe"
    payload["kind"] = "subagent"

    [session] = parse_payload(Provider.GEMINI_CLI, payload, "unused-fallback")

    assert session.provider_session_id == "cli-subagent-safe"
    assert session.provider_project_ref == "project-safe"
    assert session.branch_type is BranchType.SUBAGENT
