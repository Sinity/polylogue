"""The human_authored-implies-user-role law at the parser boundary (polylogue-n6gnd).

All three session-count paths in the index writer count
``material_origin='human_authored'`` messages into ``authored_user_*`` columns
with no role filter, while the ``user_*`` columns filter ``role='user'``. Those
columns only mean "authored USER words" if no parser can emit
``human_authored`` on a non-user role. No CHECK constraint expresses that
cross-column implication, so it is pinned here over the real fixture corpus.
"""

from __future__ import annotations

import pytest

from polylogue.archive.message.artifacts import classify_material_origin
from polylogue.core.enums import BlockType, MaterialOrigin, MessageType, Provider, Role
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication, parse_payload
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.sources.parsers.base_support import human_authored_override
from tests.infra.origin_capability_matrix import CapabilityWitness, load_manifest, load_witness_fixture


def _supported_witnesses() -> list[tuple[str, CapabilityWitness]]:
    manifest = load_manifest()
    witnesses: list[tuple[str, CapabilityWitness]] = []
    for entry in manifest.entries:
        if entry.unsupported is not None:
            continue
        for witness in entry.witnesses:
            if witness.route == "vendor":
                # Vendor-client routes need a language-server shim; they are
                # covered by tests/unit/sources/test_origin_capability_matrix.py.
                continue
            witnesses.append((str(entry.origin), witness))
    return witnesses


_WITNESSES = _supported_witnesses()


@pytest.mark.parametrize(("origin", "witness"), _WITNESSES, ids=[name for name, _ in _WITNESSES])
def test_human_authored_messages_always_carry_the_user_role(origin: str, witness: CapabilityWitness) -> None:
    """Anti-vacuity: a parser emitting HUMAN_AUTHORED on a non-user role turns this red.

    The denominator assertion below also fails if the witness stops producing
    messages at all, so a parser that silently drops its corpus cannot pass
    this law vacuously.
    """
    claim = witness.parser_claims[0]
    payload = load_witness_fixture(witness)
    sessions = parse_payload(
        claim.provider,
        payload,
        witness.fallback_id,
        source_path=witness.fixture_path,
    )
    accepted = admit_parsed_sessions_for_publication(
        sessions,
        provider=claim.provider,
        source_path=witness.fixture_path,
    )

    messages = [message for session in accepted for message in session.messages]
    assert messages, f"{origin}: witness produced no messages"

    offenders = [
        (message.provider_message_id, str(message.role))
        for message in messages
        if message.material_origin is MaterialOrigin.HUMAN_AUTHORED and message.role is not Role.USER
    ]
    assert offenders == [], f"{origin}: human_authored on non-user roles: {offenders}"


def test_the_corpus_actually_contains_human_authored_messages() -> None:
    """Denominator guard: the law above is only meaningful over a non-empty population."""
    human_authored = 0
    for _origin, witness in _WITNESSES:
        claim = witness.parser_claims[0]
        payload = load_witness_fixture(witness)
        sessions = parse_payload(
            claim.provider,
            payload,
            witness.fallback_id,
            source_path=witness.fixture_path,
        )
        for session in sessions:
            for message in session.messages:
                if message.material_origin is MaterialOrigin.HUMAN_AUTHORED:
                    human_authored += 1
    assert human_authored > 0


_HUMAN_MARKERS = (
    "# AGENTS.md instructions for my project",
    "<environment_context>pasted by the human</environment_context>",
    "<task-notification>pasted notification</task-notification>",
    "bash -lc git status",
    "# Commit neutral generated context",
    "Generate all artifacts for this neutral example",
    "Generate a retrospective for: neutral example",
    '{"queries":[{"q":"neutral"}]}',
)


@pytest.mark.parametrize("text", _HUMAN_MARKERS)
@pytest.mark.parametrize("provider", [Provider.GROK, Provider.CHATGPT])
def test_positive_chat_human_input_survives_text_origin_markers(text: str, provider: Provider) -> None:
    payload: dict[str, object]
    if provider is Provider.GROK:
        payload = {
            "conversations": [
                {
                    "conversation": {"id": "human-marker-conversation", "title": "Neutral"},
                    "responses": [{"id": "human-marker", "sender": "human", "message": text}],
                }
            ]
        }
    else:
        payload = {
            "id": "human-marker-conversation",
            "mapping": {
                "human-marker": {
                    "id": "human-marker",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "human-marker",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": [text]},
                    },
                }
            },
        }
    [session] = parse_payload(provider, payload, "fallback")
    [message] = list(session.messages)
    assert message.role is Role.USER
    assert message.text == text
    assert message.material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert (
        classify_material_origin(role=Role.USER, message_type=message.message_type, text=text)
        is not MaterialOrigin.HUMAN_AUTHORED
    )


@pytest.mark.parametrize(
    ("role", "message_type", "origin"),
    [
        (Role.USER, MessageType.TOOL_RESULT, MaterialOrigin.TOOL_RESULT),
        (Role.USER, MessageType.TOOL_USE, MaterialOrigin.UNKNOWN),
        (Role.USER, MessageType.SUMMARY, MaterialOrigin.GENERATED_CONTEXT_PACK),
        (Role.ASSISTANT, MessageType.MESSAGE, MaterialOrigin.ASSISTANT_AUTHORED),
        (Role.SYSTEM, MessageType.CONTEXT, MaterialOrigin.RUNTIME_CONTEXT),
        (Role.USER, MessageType.MESSAGE, MaterialOrigin.TOOL_RESULT),
        (Role.USER, MessageType.MESSAGE, MaterialOrigin.ASSISTANT_AUTHORED),
    ],
)
def test_positive_human_upgrade_preserves_independent_structural_provenance(
    role: Role, message_type: MessageType, origin: MaterialOrigin
) -> None:
    assert human_authored_override(role, message_type, origin) is origin
    session = ParsedSession(
        source_name=Provider.GROK,
        provider_session_id="structural-evidence",
        messages=[
            ParsedMessage(
                provider_message_id="structured", role=role, message_type=message_type, material_origin=origin
            )
        ],
    )
    assert session.messages[0].material_origin is origin


def test_chatgpt_user_shaped_tool_use_keeps_structural_provenance() -> None:
    payload = {
        "id": "structural-tool-use",
        "mapping": {
            "tool": {
                "id": "tool",
                "parent": None,
                "children": [],
                "message": {
                    "id": "tool",
                    "author": {"role": "user"},
                    "recipient": "Read",
                    "content": {"content_type": "text", "parts": ['{"path":"neutral.txt"}']},
                },
            }
        },
    }
    [session] = parse_payload(Provider.CHATGPT, payload, "fallback")
    [message] = list(session.messages)
    assert message.message_type is MessageType.TOOL_USE
    assert message.material_origin is not MaterialOrigin.HUMAN_AUTHORED


@pytest.mark.parametrize("subagent", [False, True])
def test_gemini_positive_human_marker_upgrade_requires_the_ordinary_channel(subagent: bool) -> None:
    text = "# AGENTS.md instructions for my project"
    payload = {
        "sessionId": "qualified-channel",
        "startTime": "2026-04-08T20:45:00Z",
        "lastUpdated": "2026-04-08T20:47:00Z",
        "kind": "subagent" if subagent else "chat",
        "messages": [{"id": "human", "timestamp": "2026-04-08T20:45:01Z", "type": "user", "content": [text]}],
    }
    [session] = parse_payload(Provider.GEMINI_CLI, payload, "fallback")
    [message] = list(session.messages)
    assert message.role is Role.USER
    assert message.text == text
    assert message.message_type is MessageType.CONTEXT
    assert message.material_origin is (MaterialOrigin.RUNTIME_CONTEXT if subagent else MaterialOrigin.HUMAN_AUTHORED)


def test_claude_project_supplied_document_does_not_claim_human_authorship() -> None:
    payload = {
        "uuid": "neutral-project",
        "prompt_template": "Use evidence.",
        "docs": [{"uuid": "vendor-doc", "filename": "manual.txt", "content": "Neutral vendor manual."}],
    }
    [session] = parse_payload(Provider.CLAUDE_AI, payload, "fallback")
    prompt, document = list(session.messages)
    assert prompt.material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert document.role is Role.USER
    assert document.message_type is MessageType.CONTEXT
    assert document.material_origin is MaterialOrigin.RUNTIME_CONTEXT
    assert document.text == "Neutral vendor manual."
    assert document.blocks[0].type is BlockType.DOCUMENT
    assert document.blocks[0].metadata == {"doc_uuid": "vendor-doc", "name": "manual.txt"}


def test_claude_human_chat_with_document_keeps_positive_authorship() -> None:
    payload = {
        "uuid": "neutral-chat",
        "chat_messages": [
            {
                "uuid": "human",
                "sender": "human",
                "text": "# AGENTS.md instructions for my project",
                "content": [
                    {"type": "text", "text": "# AGENTS.md instructions for my project"},
                    {"type": "document", "media_type": "text/plain"},
                ],
                "attachments": [
                    {
                        "file_name": "manual.txt",
                        "file_type": "text/plain",
                        "extracted_content": "Neutral vendor manual.",
                    }
                ],
            }
        ],
    }
    [session] = parse_payload(Provider.CLAUDE_AI, payload, "fallback")
    [message] = list(session.messages)
    assert message.role is Role.USER
    assert message.material_origin is MaterialOrigin.HUMAN_AUTHORED
    assert any(block.type is BlockType.DOCUMENT for block in message.blocks)
    assert next(block for block in message.blocks if block.type is BlockType.DOCUMENT).media_type == "text/plain"
    assert session.attachments[0].name == "manual.txt"
    assert session.attachments[0].inline_bytes == b"Neutral vendor manual."
