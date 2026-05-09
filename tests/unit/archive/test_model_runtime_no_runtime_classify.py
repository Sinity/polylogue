"""Tests pinning that ``Message.is_context_dump`` and
``Message.is_protocol_artifact`` use stored ``message_type`` only.

Issue #839 AC #3: stop using runtime classification. Pre-#839 archive rows
without persisted ``message_type`` are not recognized as CONTEXT/PROTOCOL
through text inspection at read time. Backfill (AC #2) is tracked separately.
"""

from __future__ import annotations

import pytest

import polylogue.archive.message.artifacts as artifacts_module
import polylogue.archive.message.model_runtime as model_runtime
from polylogue.archive.message.types import MessageType
from tests.infra.builders import make_msg


def test_is_context_dump_uses_stored_message_type() -> None:
    msg = make_msg(
        id="m-context",
        role="user",
        text="Just plain dialogue text.",
        message_type=MessageType.CONTEXT,
    )
    assert msg.is_context_dump is True


def test_is_protocol_artifact_uses_stored_message_type() -> None:
    msg = make_msg(
        id="m-protocol",
        role="user",
        text="Just plain dialogue text.",
        message_type=MessageType.PROTOCOL,
    )
    assert msg.is_protocol_artifact is True


def test_model_runtime_does_not_import_classifier() -> None:
    """``model_runtime`` must not reference the runtime text classifier."""
    assert not hasattr(model_runtime, "classify_text_message_type")


def test_is_context_dump_does_not_call_runtime_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stored message_type is the only source of truth — no runtime classify."""
    calls: list[str | None] = []

    real_classify = artifacts_module.classify_text_message_type

    def spy(text: str | None) -> MessageType | None:
        calls.append(text)
        return real_classify(text)

    monkeypatch.setattr(artifacts_module, "classify_text_message_type", spy)

    # Text that the heuristic classifier WOULD label CONTEXT.
    msg = make_msg(
        id="m1",
        role="user",
        text="<environment_context>\n<cwd>/x</cwd>\n</environment_context>",
    )
    assert msg.is_context_dump is False
    assert msg.is_protocol_artifact is False
    assert calls == [], "model_runtime must not invoke classify_text_message_type"


def test_is_protocol_artifact_does_not_call_runtime_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str | None] = []

    real_classify = artifacts_module.classify_text_message_type

    def spy(text: str | None) -> MessageType | None:
        calls.append(text)
        return real_classify(text)

    monkeypatch.setattr(artifacts_module, "classify_text_message_type", spy)

    msg = make_msg(id="m1", role="user", text="<bash-input>ls</bash-input>")
    assert msg.is_protocol_artifact is False
    assert msg.is_context_dump is False
    assert calls == []


def test_pre_839_row_without_persisted_type_returns_false() -> None:
    """A row with default MessageType.MESSAGE and a text-marker is NOT context.

    This is the AC #3 contract: backfill (AC #2) is what re-classifies
    pre-#839 rows; runtime no longer infers from text.
    """
    msg = make_msg(
        id="m-pre-839",
        role="user",
        text="<system-reminder>old marker</system-reminder>",
    )
    assert msg.message_type == MessageType.MESSAGE
    assert msg.is_context_dump is False
    assert msg.is_protocol_artifact is False
