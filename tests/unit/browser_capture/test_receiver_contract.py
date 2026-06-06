"""Contract tests for the browser-capture receiver.

Verifies the receiver accepts valid payloads and rejects invalid ones
at the model and write layers.  These are contract-level assertions,
not integration tests of the HTTP server (which live in test_receiver.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from polylogue.browser_capture.models import (
    BrowserCaptureEnvelope,
    looks_like_browser_capture,
)
from polylogue.browser_capture.receiver import (
    write_capture_envelope,
)
from polylogue.sources.parsers.browser_capture import parse as parse_browser_capture

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_payload(provider: str = "chatgpt", session_id: str = "conv-123") -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-123",
            "page_title": "ChatGPT - Work plan",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "chatgpt-dom-v1",
        },
        "session": {
            "provider": provider,
            "provider_session_id": session_id,
            "title": "Work plan",
            "turns": [
                {"provider_turn_id": "u1", "role": "user", "text": "Draft"},
                {"provider_turn_id": "a1", "role": "assistant", "text": "Here"},
            ],
        },
    }


def _invalid_payloads() -> list[tuple[str, dict[str, object]]]:
    """Return (label, payload) pairs that must be rejected."""
    return [
        (
            "missing provenance",
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "conv-123",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "x"}],
                },
            },
        ),
        (
            "missing session",
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "provenance": {
                    "source_url": "https://chatgpt.com/c/conv-123",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
            },
        ),
        (
            "empty turns",
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "provenance": {
                    "source_url": "https://chatgpt.com/c/conv-123",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "conv-123",
                    "turns": [],
                },
            },
        ),
        (
            "turn with no text and no attachments",
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 1,
                "provenance": {
                    "source_url": "https://chatgpt.com/c/conv-123",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "conv-123",
                    "turns": [{"provider_turn_id": "e1", "role": "user", "text": ""}],
                },
            },
        ),
        (
            "wrong polylogue_capture_kind",
            {
                "polylogue_capture_kind": "something_else",
                "schema_version": 1,
                "provenance": {
                    "source_url": "https://chatgpt.com/c/conv-123",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "conv-123",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "x"}],
                },
            },
        ),
        (
            "wrong schema_version",
            {
                "polylogue_capture_kind": "browser_llm_session",
                "schema_version": 99,
                "provenance": {
                    "source_url": "https://chatgpt.com/c/conv-123",
                    "captured_at": "2026-04-24T00:00:00+00:00",
                    "adapter_name": "chatgpt-dom-v1",
                },
                "session": {
                    "provider": "chatgpt",
                    "provider_session_id": "conv-123",
                    "turns": [{"provider_turn_id": "u1", "role": "user", "text": "x"}],
                },
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Acceptance: valid payloads
# ---------------------------------------------------------------------------


class TestReceiverAcceptsValidPayload:
    def test_chatgpt_envelope_validates(self) -> None:
        envelope = BrowserCaptureEnvelope.model_validate(_valid_payload("chatgpt"))
        assert envelope.provider.value == "chatgpt"
        assert envelope.provider_session_id == "conv-123"
        assert envelope.session.turns[0].role.value == "user"

    def test_claude_ai_envelope_validates(self) -> None:
        payload = _valid_payload("claude-ai", session_id="claude-session")
        envelope = BrowserCaptureEnvelope.model_validate(payload)
        assert envelope.provider.value == "claude-ai"
        assert envelope.provider_session_id == "claude-session"

    def test_write_capture_envelope_writes_and_returns_path(
        self,
        tmp_path: Path,
    ) -> None:
        envelope = BrowserCaptureEnvelope.model_validate(_valid_payload())
        result = write_capture_envelope(envelope, spool_path=tmp_path)
        assert result.provider == "chatgpt"
        assert result.provider_session_id == "conv-123"
        assert result.bytes_written > 0
        assert result.path.exists()
        assert result.replaced is False

    def test_write_capture_envelope_is_idempotent(self, tmp_path: Path) -> None:
        envelope = BrowserCaptureEnvelope.model_validate(_valid_payload())
        first = write_capture_envelope(envelope, spool_path=tmp_path)
        second = write_capture_envelope(envelope, spool_path=tmp_path)
        assert first.path == second.path
        assert first.replaced is False
        assert second.replaced is True

    def test_provenance_sets_capture_id_when_none(self) -> None:
        payload = _valid_payload()
        envelope = BrowserCaptureEnvelope.model_validate(payload)
        assert envelope.capture_id == "chatgpt:conv-123"

    def test_capture_id_preserved_when_provided(self) -> None:
        payload = _valid_payload()
        payload["capture_id"] = "custom-capture-id"
        envelope = BrowserCaptureEnvelope.model_validate(payload)
        assert envelope.capture_id == "custom-capture-id"

    def test_unknown_provider_maps_to_unknown_enum(self) -> None:
        """Unknown provider strings are accepted and mapped to Provider.UNKNOWN
        without raising ValidationError."""
        payload = _valid_payload("not-a-real-provider", "bad-session")
        envelope = BrowserCaptureEnvelope.model_validate(payload)
        assert envelope.provider.value == "unknown"

    def test_parser_populates_archive_message_contract(self) -> None:
        payload = _valid_payload()
        assert isinstance(payload["session"], dict)
        payload["session"]["model"] = "gpt-4o"
        parsed = parse_browser_capture(payload, "fallback-id")

        assert [message.position for message in parsed.messages] == [0, 1]
        assert [message.variant_index for message in parsed.messages] == [0, 0]
        assert [message.is_active_path for message in parsed.messages] == [True, True]
        assert [message.is_active_leaf for message in parsed.messages] == [False, True]
        assert [message.model_name for message in parsed.messages] == ["gpt-4o", "gpt-4o"]
        assert parsed.active_leaf_message_provider_id == "a1"


# ---------------------------------------------------------------------------
# Rejection: invalid payloads
# ---------------------------------------------------------------------------


class TestReceiverRejectsInvalidPayload:
    @pytest.mark.parametrize(
        "label,payload",
        _invalid_payloads(),
        ids=[label for label, _ in _invalid_payloads()],
    )
    def test_invalid_payload_raises_validation_error(
        self,
        label: str,
        payload: dict[str, object],
    ) -> None:
        with pytest.raises(ValidationError):
            BrowserCaptureEnvelope.model_validate(payload)

    def test_empty_role_on_turn_is_rejected(self) -> None:
        """Empty role raises ValidationError — Role.normalize rejects empty
        string with 'Role cannot be empty'."""
        payload = _valid_payload()
        assert isinstance(payload["session"], dict)
        payload["session"]["turns"] = [
            {"provider_turn_id": "u1", "role": "", "text": "Draft"},
        ]
        with pytest.raises(ValidationError, match="cannot be empty"):
            BrowserCaptureEnvelope.model_validate(payload)

    def test_null_text_turn_with_attachment_passes(self) -> None:
        """A turn with no text but with an attachment is valid."""
        payload = _valid_payload()
        assert isinstance(payload["session"], dict)
        payload["session"]["turns"] = [
            {
                "provider_turn_id": "a1",
                "role": "assistant",
                "text": None,
                "attachments": [
                    {
                        "provider_attachment_id": "att-1",
                        "name": "file.txt",
                        "mime_type": "text/plain",
                    }
                ],
            },
        ]
        envelope = BrowserCaptureEnvelope.model_validate(payload)
        assert envelope.session.turns[0].attachments[0].name == "file.txt"


# ---------------------------------------------------------------------------
# looks_like_browser_capture
# ---------------------------------------------------------------------------


class TestLooksLikeBrowserCapture:
    def test_recognizes_valid_envelope(self) -> None:
        assert looks_like_browser_capture(_valid_payload()) is True

    def test_rejects_non_dict(self) -> None:
        assert looks_like_browser_capture(None) is False
        assert looks_like_browser_capture("not-a-dict") is False
        assert looks_like_browser_capture([]) is False

    def test_rejects_unknown_kind(self) -> None:
        assert looks_like_browser_capture({"polylogue_capture_kind": "other"}) is False

    def test_rejects_wrong_schema_version(self) -> None:
        assert (
            looks_like_browser_capture(
                {
                    "polylogue_capture_kind": "browser_llm_session",
                    "schema_version": 0,
                }
            )
            is False
        )

    def test_rejects_missing_session(self) -> None:
        assert (
            looks_like_browser_capture(
                {
                    "polylogue_capture_kind": "browser_llm_session",
                    "schema_version": 1,
                    "provenance": {},
                }
            )
            is False
        )

    def test_rejects_non_dict_session(self) -> None:
        assert (
            looks_like_browser_capture(
                {
                    "polylogue_capture_kind": "browser_llm_session",
                    "schema_version": 1,
                    "provenance": {},
                    "session": "not-a-dict",
                }
            )
            is False
        )
