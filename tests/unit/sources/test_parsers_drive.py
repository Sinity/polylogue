"""Focused contracts for the Drive/Gemini chunked-prompt parser.

Broader generated-provider coverage already lives in:
- `test_parsers_props.py`
- `test_source_laws.py`
- `test_unified_semantic_laws.py`

This file keeps only the concrete parser behaviors that are still clearest as
direct contracts.
"""

from __future__ import annotations

import json

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.scenarios import CorpusSpec
from polylogue.sources.parsers.drive import (
    _attachment_from_doc,
    _collect_drive_docs,
    parse_chunked_prompt,
)
from polylogue.sources.parsers.drive_support import extract_text_from_chunk
from polylogue.sources.parsers.drive_support_attachments import DRIVE_LIVE_FETCH_DATA_KEY


@pytest.fixture
def synthetic_gemini_payload() -> JSONDocument:
    from polylogue.schemas.synthetic import SyntheticCorpus

    raw = SyntheticCorpus.generate_for_spec(
        CorpusSpec.for_provider(
            "gemini",
            count=1,
            messages_min=4,
            messages_max=7,
            seed=42,
            origin="generated.test-drive-parser",
            tags=("synthetic", "test", "drive-parser"),
        )
    )[0]
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        ({"text": "hello"}, "hello"),
        ({"content": "alt"}, "alt"),
        ({"message": "msg"}, "msg"),
        ({"parts": [{"text": "alpha"}, "beta", {"text": "gamma"}]}, "alpha\nbeta\ngamma"),
        ({"text": None, "parts": [{"text": "fallback"}]}, "fallback"),
        ({"data": {"text": "nested"}}, None),
        ("not a dict", None),
        (None, None),
    ],
    ids=[
        "text",
        "content",
        "message",
        "parts",
        "parts-fallback",
        "nested-dict-not-recursed",
        "string-input",
        "none-input",
    ],
)
def test_extract_text_from_chunk_contract(chunk: object, expected: str | None) -> None:
    assert extract_text_from_chunk(chunk) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("not a dict", []),
        ({"driveDocument": "doc-1"}, ["doc-1"]),
        ({"driveDocuments": [{"id": "doc-2"}, "doc-3"]}, [{"id": "doc-2"}, "doc-3"]),
        (
            {"metadata": {"driveDocument": "nested-doc"}},
            ["nested-doc"],
        ),
    ],
    ids=["non-dict", "single-doc", "list-docs", "nested-doc"],
)
def test_collect_drive_docs_contract(payload: object, expected: list[object]) -> None:
    assert _collect_drive_docs(payload) == expected


@pytest.mark.parametrize(
    ("doc", "expected_id", "expected_size"),
    [
        ("doc-string-id", "doc-string-id", None),
        ({"id": "doc-1", "sizeBytes": "5000"}, "doc-1", 5000),
        ({"fileId": "doc-2", "size": 12}, "doc-2", 12),
        (123, None, None),
        ({"name": "missing-id"}, None, None),
    ],
    ids=["string-doc", "size-bytes-string", "file-id-int-size", "invalid-type", "missing-id"],
)
def test_attachment_from_doc_contract(doc: object, expected_id: str | None, expected_size: int | None) -> None:
    attachment = _attachment_from_doc(doc if isinstance(doc, dict | str) else {}, "msg-1")
    if expected_id is None:
        assert attachment is None
    else:
        assert attachment is not None
        assert attachment.provider_attachment_id == expected_id
        assert attachment.message_provider_id == "msg-1"
        assert attachment.size_bytes == expected_size


def test_parse_chunked_prompt_preserves_core_session_metadata() -> None:
    payload: JSONDocument = {
        "id": "gemini-conv",
        "displayName": "Gemini Session",
        "createTime": "2024-01-15T10:30:00Z",
        "updateTime": "2024-01-15T11:45:00Z",
        "chunkedPrompt": {
            "chunks": [
                {"id": "msg-user", "role": "user", "text": "Question"},
                {
                    "id": "msg-model",
                    "role": "model",
                    "text": "Answer",
                    "model": "gemini-2.5-pro",
                    "durationMs": 1500,
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.source_name == "gemini"
    assert result.provider_session_id == "gemini-conv"
    assert result.title == "Gemini Session"
    assert result.title_source == "origin"
    assert result.created_at == "2024-01-15T10:30:00Z"
    assert result.updated_at == "2024-01-15T11:45:00Z"
    assert [message.provider_message_id for message in result.messages] == ["msg-user", "msg-model"]
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert [message.text for message in result.messages] == ["Question", "Answer"]
    assert [message.position for message in result.messages] == [0, 1]
    assert [message.variant_index for message in result.messages] == [0, 0]
    assert [message.is_active_path for message in result.messages] == [True, True]
    assert [message.is_active_leaf for message in result.messages] == [False, True]
    assert result.active_leaf_message_provider_id == "msg-model"
    assert result.messages[1].model_name == "gemini-2.5-pro"
    assert result.messages[1].duration_ms == 1500


def test_parse_chunked_prompt_persists_run_settings_verbatim() -> None:
    """runSettings (polylogue-2qx.4 / polylogue-cgfy) must reach ``ParsedSession.run_settings``.

    Deleting the ``run_settings=dict(run_settings) if run_settings else None``
    kwarg on the ``ParsedSession`` return in ``parse_chunked_prompt`` makes
    this assert None even though the value already feeds the ``model_config``
    session_event.
    """
    payload: JSONDocument = {
        "id": "gemini-run-settings",
        "runSettings": {
            "temperature": 0.7,
            "topP": 0.9,
            "topK": 40,
            "maxOutputTokens": 8192,
            "thinkingLevel": "high",
        },
        "chunkedPrompt": {
            "chunks": [{"id": "msg-user", "role": "user", "text": "hi"}],
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.run_settings == {
        "temperature": 0.7,
        "topP": 0.9,
        "topK": 40,
        "maxOutputTokens": 8192,
        "thinkingLevel": "high",
    }
    model_config_events = [e for e in result.session_events if e.event_type == "model_config"]
    assert len(model_config_events) == 1


def test_parse_chunked_prompt_without_run_settings_leaves_it_none() -> None:
    payload: JSONDocument = {
        "id": "gemini-no-run-settings",
        "chunkedPrompt": {"chunks": [{"id": "msg-user", "role": "user", "text": "hi"}]},
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.run_settings is None


def test_parse_chunked_prompt_records_nonempty_pending_input_as_draft() -> None:
    """``chunkedPrompt.pendingInputs`` (polylogue-o4j2) is the operator's
    not-yet-submitted textbox content -- unrecoverable once overwritten if
    dropped at parse. A non-blank entry must survive on
    ``ParsedSession.pending_drafts``.

    Deliberately NOT a session_event: a draft is mutable current state, and
    session_events feed session_revision_projection's append-only
    comparison axes (polylogue-aggz Invariant 1) -- see
    ``test_pending_draft_mutation_does_not_break_revision_containment``
    below for the failure this would otherwise reproduce.
    """
    payload: JSONDocument = {
        "id": "gemini-pending-draft",
        "updateTime": "2024-01-15T11:45:00Z",
        "chunkedPrompt": {
            "chunks": [{"id": "msg-user", "role": "user", "text": "hi"}],
            "pendingInputs": [{"text": "unsent follow-up question", "role": "user", "tokenCount": 4}],
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.pending_drafts == [{"text": "unsent follow-up question", "role": "user", "token_count": 4}]
    assert not [e for e in result.session_events if e.event_type == "draft_input"]


def test_parse_chunked_prompt_skips_blank_pending_input() -> None:
    """The wire-common case -- the textbox was empty when Drive synced -- carries
    no evidence and must not be recorded as a draft.
    """
    payload: JSONDocument = {
        "id": "gemini-pending-blank",
        "chunkedPrompt": {
            "chunks": [{"id": "msg-user", "role": "user", "text": "hi"}],
            "pendingInputs": [{"text": "", "role": "user"}, {"text": "   ", "role": "user"}],
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.pending_drafts == []


def test_parse_chunked_prompt_without_pending_inputs_has_no_drafts() -> None:
    payload: JSONDocument = {
        "id": "gemini-no-pending",
        "chunkedPrompt": {"chunks": [{"id": "msg-user", "role": "user", "text": "hi"}]},
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.pending_drafts == []


def test_pending_draft_mutation_does_not_break_revision_containment() -> None:
    """Regression for the P1 a reviewer traced on drive.py's original
    draft-as-session_event design (polylogue-o4j2 fix-up).

    A draft is mutable: the operator edits the textbox, then eventually
    submits it (at which point the pendingInputs entry disappears and a real
    message appears instead). If the draft were folded into
    session_revision_projection's event axis, editing it would create
    disjoint event identities (comparing as a conflict/fork) and submitting
    it would shrink the event axis while the message axis grows --
    ``_relation`` requires every non-equal axis to agree on direction, so
    both cases would misclassify revision membership
    (classify_membership_revisions). This walks retain -> edit draft ->
    retain -> submit and asserts containment holds at every step now that
    drafts live outside every comparison axis.
    """
    from polylogue.archive.session_revision_membership import (
        MembershipRevision,
        _relation,
        classify_membership_revisions,
    )
    from polylogue.pipeline.ids import session_revision_projection

    chunks_before_submit: list[JSONValue] = [{"id": "msg-1", "role": "user", "text": "hi"}]

    # Revision 1: retain with an initial draft in the textbox.
    payload_1: JSONDocument = {
        "id": "gemini-draft-lifecycle",
        "chunkedPrompt": {
            "chunks": chunks_before_submit,
            "pendingInputs": [{"text": "draft v1", "role": "user"}],
        },
    }
    revision_1 = parse_chunked_prompt("gemini", payload_1, "fallback-id")
    # Revision 2: the SAME conversation retained again after the operator
    # edited the draft text (no new message yet).
    payload_2: JSONDocument = {
        "id": "gemini-draft-lifecycle",
        "chunkedPrompt": {
            "chunks": chunks_before_submit,
            "pendingInputs": [{"text": "draft v2, much longer now", "role": "user"}],
        },
    }
    revision_2 = parse_chunked_prompt("gemini", payload_2, "fallback-id")
    # Revision 3: the draft was submitted -- it becomes a real message and
    # pendingInputs is empty again.
    payload_3: JSONDocument = {
        "id": "gemini-draft-lifecycle",
        "chunkedPrompt": {
            "chunks": [
                *chunks_before_submit,
                {"id": "msg-2", "role": "user", "text": "draft v2, much longer now"},
            ],
            "pendingInputs": [{"text": "", "role": "user"}],
        },
    }
    revision_3 = parse_chunked_prompt("gemini", payload_3, "fallback-id")

    projection_1 = session_revision_projection(revision_1)
    projection_2 = session_revision_projection(revision_2)
    projection_3 = session_revision_projection(revision_3)

    # Editing the draft alone (same messages, different draft text) must not
    # look like a fork -- both revisions carry the exact same content-bearing
    # evidence once drafts are excluded from comparison identity.
    assert _relation(projection_1, projection_2) == "equal"
    # Submitting must read as ordinary append-only growth (revision 3
    # contains revision 2), not a conflict from the event axis shrinking.
    assert _relation(projection_3, projection_2) == "a_contains_b"

    classification = classify_membership_revisions(
        [
            MembershipRevision(raw_id="r1", projection=projection_1),
            MembershipRevision(raw_id="r2", projection=projection_2),
            MembershipRevision(raw_id="r3", projection=projection_3),
        ]
    )
    # accepted_raw_ids is the whole append-only growth chain, oldest to
    # newest (r1/r2 collapse to one "equal" representative -- edit-only
    # revisions -- which then chains into r3's growth); the key assertion is
    # what is ABSENT: no conflict, so nothing lands in ambiguous_raw_ids.
    assert classification.accepted_raw_ids == ("r1", "r3")
    assert classification.equivalent_raw_ids == ("r2",)
    assert not classification.ambiguous_raw_ids


def test_parse_chunked_prompt_records_fallback_title_source() -> None:
    payload: JSONDocument = {
        "id": "gemini-fallback-title",
        "chunkedPrompt": {"chunks": [{"id": "msg-user", "role": "user", "text": "Question"}]},
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert result.title == "fallback-id"
    assert result.title_source == "unknown"


def test_parse_chunked_prompt_preserves_reasoning_code_tool_results_and_attachments() -> None:
    payload: JSONDocument = {
        "id": "gemini-rich",
        "displayName": "Gemini Rich",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-user",
                    "role": "user",
                    "text": "question",
                    "driveDocument": {
                        "id": "doc-1",
                        "name": "spec.pdf",
                        "mimeType": "application/pdf",
                        "sizeBytes": "12",
                    },
                },
                {
                    "id": "msg-thought",
                    "role": "model",
                    "text": "reasoning",
                    "isThought": True,
                    "thinkingBudget": 32,
                },
                {
                    "id": "msg-code",
                    "role": "model",
                    "parts": [{"text": "inline"}],
                    "executableCode": {"language": "python", "code": "print('ok')"},
                    "codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "ok"},
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [message.provider_message_id for message in result.messages] == [
        "msg-user",
        "msg-thought",
        "msg-code",
    ]
    assert [block.type for block in result.messages[0].blocks] == ["text", "document"]
    assert [block.type for block in result.messages[1].blocks] == ["thinking"]
    assert [block.type for block in result.messages[2].blocks] == ["text", "code", "tool_result"]

    user_msg = result.messages[0]
    assert user_msg.blocks[0].text == "question"
    assert user_msg.blocks[1].metadata is not None
    drive_document = user_msg.blocks[1].metadata.get("driveDocument")
    assert isinstance(drive_document, dict)
    assert drive_document["id"] == "doc-1"

    thought_msg = result.messages[1]
    assert thought_msg.blocks[0].text == "reasoning"

    code_msg = result.messages[2]
    assert code_msg.blocks[0].text == "inline"
    assert code_msg.blocks[1].text == "print('ok')"
    assert code_msg.blocks[2].text == "ok"
    assert len(result.attachments) == 1
    assert result.attachments[0].provider_attachment_id == "doc-1"
    assert result.attachments[0].mime_type == "application/pdf"
    assert result.attachments[0].size_bytes == 12


def test_thinking_block_reasoning_continuity_evidence_routes_to_session_events() -> None:
    """Gemini's thoughtSignatures/thinkingBudget are dropped by ``blocks.metadata``
    (bd polylogue-9x22: the ``blocks`` table has no metadata column, and the
    write path only reads a ``language`` key back out of it), so
    ``session_events_from_meta_blocks`` must carry them through
    ``session_events`` instead. Deleting that wiring in
    ``parse_chunked_prompt`` makes this fail.
    """
    payload: JSONDocument = {
        "id": "gemini-thinking",
        "displayName": "Gemini Thinking",
        "chunkedPrompt": {
            "chunks": [
                {"id": "msg-user", "role": "user", "text": "question"},
                {
                    "id": "msg-thought",
                    "role": "model",
                    "text": "reasoning",
                    "isThought": True,
                    "thinkingBudget": 32,
                    "thoughtSignatures": ["sig-1"],
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    thinking_events = [event for event in result.session_events if event.event_type == "gemini_thinking_evidence"]
    assert len(thinking_events) == 1
    event = thinking_events[0]
    assert event.source_message_provider_id == "msg-thought"
    assert event.payload == {
        "block_index": 0,
        "thinkingBudget": 32,
        "thoughtSignatures": ["sig-1"],
    }


def test_live_fetched_drive_attachment_bytes_reach_inline_bytes_not_block_metadata() -> None:
    """polylogue-83u.2 CodeRabbit/Codex P1: the live-fetch sidecar
    (`polylogue.sources.drive.attachment_fetch`) injects fetched bytes as
    base64 directly into the same driveDocument dict this parser reads.
    `attachment_from_doc` must decode it into `ParsedAttachment.inline_bytes`
    -- but the sidecar must NEVER also reach `ContentBlock.raw`/block
    metadata, which (unlike attachments) is not content-addressed and would
    duplicate the full attachment bytes into the index for every acquired
    Drive attachment.
    """
    import base64

    fetched_bytes = b"live-fetched drive attachment bytes"
    payload: JSONDocument = {
        "id": "gemini-live-fetch",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-user",
                    "role": "user",
                    "text": "question",
                    "driveDocument": {
                        "id": "doc-1",
                        "name": "spec.pdf",
                        "mimeType": "application/pdf",
                        DRIVE_LIVE_FETCH_DATA_KEY: base64.b64encode(fetched_bytes).decode("ascii"),
                    },
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    user_msg = result.messages[0]
    document_block = user_msg.blocks[1]
    assert document_block.metadata is not None
    assert DRIVE_LIVE_FETCH_DATA_KEY not in document_block.metadata
    drive_document_metadata = document_block.metadata.get("driveDocument")
    assert isinstance(drive_document_metadata, dict)
    assert DRIVE_LIVE_FETCH_DATA_KEY not in drive_document_metadata
    assert drive_document_metadata["id"] == "doc-1"

    assert len(result.attachments) == 1
    assert result.attachments[0].inline_bytes == fetched_bytes


def test_parse_chunked_prompt_preserves_token_count_usage_events() -> None:
    payload: JSONDocument = {
        "id": "gemini-usage",
        "displayName": "Gemini Usage",
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-user",
                    "role": "user",
                    "text": "question",
                    "tokenCount": 7,
                    "createTime": "2026-01-01T00:00:01Z",
                },
                {
                    "id": "msg-model",
                    "role": "model",
                    "text": "answer",
                    "tokenCount": 11,
                    "finishReason": "STOP",
                    "model": "gemini-2.5-pro",
                    "createTime": "2026-01-01T00:00:02Z",
                },
            ]
        },
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [event.event_type for event in result.session_events] == ["token_count", "token_count"]
    assert result.session_events[0].source_message_provider_id == "msg-user"
    assert result.session_events[0].timestamp == "2026-01-01T00:00:01Z"
    assert result.session_events[0].payload == {
        "type": "token_count",
        "last_token_usage": {"input_tokens": 7},
    }
    assert result.session_events[1].source_message_provider_id == "msg-model"
    assert result.session_events[1].timestamp == "2026-01-01T00:00:02Z"
    assert result.session_events[1].payload == {
        "type": "token_count",
        "last_token_usage": {"output_tokens": 11},
        "finish_reason": "STOP",
        "model": "gemini-2.5-pro",
    }


def test_parse_chunked_prompt_preserves_attachment_only_chunks_and_chunk_timestamps() -> None:
    payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "id": "msg-doc",
                    "role": "user",
                    "createTime": "2024-01-15T10:30:00Z",
                    "driveDocument": {"id": "doc-1", "name": "notes.txt"},
                },
                {
                    "id": "msg-inline",
                    "role": "user",
                    "createTime": "2024-01-15T10:31:00Z",
                    "inlineFile": {"mimeType": "text/plain", "data": "aGVsbG8="},
                },
                {
                    "id": "msg-video",
                    "role": "model",
                    "createTime": "2024-01-15T10:32:00Z",
                    "youtubeVideo": {"id": "vid-1"},
                },
            ]
        }
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [message.provider_message_id for message in result.messages] == ["msg-doc", "msg-inline", "msg-video"]
    assert [message.timestamp for message in result.messages] == [
        "2024-01-15T10:30:00Z",
        "2024-01-15T10:31:00Z",
        "2024-01-15T10:32:00Z",
    ]
    assert [message.text for message in result.messages] == [None, None, None]
    assert [message.position for message in result.messages] == [0, 1, 2]
    assert result.active_leaf_message_provider_id == "msg-video"
    assert result.created_at == "2024-01-15T10:30:00Z"
    assert result.updated_at == "2024-01-15T10:32:00Z"
    assert [block.type for block in result.messages[0].blocks] == ["document"]
    assert len(result.attachments) == 3
    assert result.attachments[0].provider_attachment_id == "doc-1"
    assert result.attachments[1].provider_attachment_id.startswith("inline-file-")
    assert result.attachments[1].mime_type == "text/plain"
    assert result.attachments[1].size_bytes == 5
    assert result.attachments[2].provider_attachment_id == "youtube-video-vid-1"
    assert result.attachments[2].mime_type == "video/youtube"


def test_parse_chunked_prompt_skips_chunks_without_text_or_role() -> None:
    payload: JSONDocument = {
        "chunkedPrompt": {
            "chunks": [
                "string chunk without role",
                {"text": "missing role"},
                {"role": "user"},
                {"role": "user", "text": "kept"},
                {"role": "model", "parts": [{"text": "also kept"}]},
            ]
        }
    }

    result = parse_chunked_prompt("gemini", payload, "fallback-id")

    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert [message.text for message in result.messages] == ["kept", "also kept"]


def test_parse_chunked_prompt_accepts_synthetic_exports(synthetic_gemini_payload: JSONDocument) -> None:
    result = parse_chunked_prompt("gemini", synthetic_gemini_payload, "synthetic-fallback")

    assert result.source_name == "gemini"
    assert result.messages
    assert all(message.text for message in result.messages)
    assert all(len(message.blocks) > 0 for message in result.messages)
