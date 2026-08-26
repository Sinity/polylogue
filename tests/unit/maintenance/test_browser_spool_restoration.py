"""Production browser-capture admission laws used by source restoration.

These tests intentionally exercise the ordinary receiver writer.  The
one-time restoration manifest and its retained bytes are not test fixtures or
repository authority; this file only protects the reusable admission
boundary that consumes them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    BrowserCaptureSpoolConflictError,
    capture_artifact_path,
    write_capture_envelope_bytes,
)


def _payload(*, session_id: str = "conversation-1", fidelity: str = "native") -> dict[str, object]:
    return {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conversation-1",
            "captured_at": "2026-08-26T12:00:00+00:00",
            "adapter_name": "chatgpt-test-v1",
            "extension_instance_id": "restoration-test",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": session_id,
            "turns": [
                {
                    "provider_turn_id": "turn-1",
                    "role": "assistant",
                    "text": "captured text",
                    "identity_observation": {
                        "origin": "chatgpt",
                        "provider_conversation_id": session_id,
                        "provider_message_id": "turn-1",
                        "adapter_name": "chatgpt-test-v1",
                        "fidelity": fidelity,
                    },
                }
            ],
        },
    }


def _raw(payload: dict[str, object], *, indent: int | None = None) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=indent).encode("utf-8")


def test_admission_preserves_exact_native_and_dom_degraded_envelopes(tmp_path: Path) -> None:
    """Anti-vacuity: dropping raw publication would lose the fidelity marker."""
    for fidelity in ("native", "dom_degraded"):
        raw = _raw(_payload(session_id=f"{fidelity}-capture", fidelity=fidelity), indent=2)

        result = write_capture_envelope_bytes(raw, spool_path=tmp_path)

        assert result.path.read_bytes() == raw
        admitted = BrowserCaptureEnvelope.model_validate(json.loads(result.path.read_bytes()))
        assert admitted.session.turns[0].identity_observation is not None
        assert admitted.session.turns[0].identity_observation.fidelity == fidelity


def test_duplicate_delivery_is_exact_content_idempotent_and_keeps_first_bytes(tmp_path: Path) -> None:
    """Anti-vacuity: changing only JSON formatting must not publish twice."""
    payload = _payload()
    first_raw = _raw(payload, indent=2)
    second_raw = _raw(payload)

    first = write_capture_envelope_bytes(first_raw, spool_path=tmp_path)
    second = write_capture_envelope_bytes(second_raw, spool_path=tmp_path)

    assert second.deduplicated is True
    assert second.path == first.path
    assert second.path.read_bytes() == first_raw
    assert len(tuple(tmp_path.rglob("*.json"))) == 1


def test_name_collision_with_different_identity_refuses_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: treating a name collision as a duplicate drops one capture."""
    import polylogue.browser_capture.receiver as receiver

    collision_path = tmp_path / "chatgpt" / "same-name.json"
    monkeypatch.setattr(receiver, "capture_artifact_path", lambda envelope, spool_path=None: collision_path)

    first_raw = _raw(_payload(session_id="first-session"))
    second_raw = _raw(_payload(session_id="second-session"))
    write_capture_envelope_bytes(first_raw, spool_path=tmp_path)

    with pytest.raises(BrowserCaptureSpoolConflictError, match="collision"):
        write_capture_envelope_bytes(second_raw, spool_path=tmp_path)

    assert collision_path.read_bytes() == first_raw


def test_malformed_existing_artifact_refuses_without_overwrite(tmp_path: Path) -> None:
    """Anti-vacuity: corruption is not an invitation to replace evidence."""
    envelope = BrowserCaptureEnvelope.model_validate(_payload())
    target = capture_artifact_path(envelope, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"truncated envelope")

    with pytest.raises(BrowserCaptureSpoolConflictError, match="malformed"):
        write_capture_envelope_bytes(_raw(_payload()), spool_path=tmp_path)

    assert target.read_bytes() == b"truncated envelope"


def test_interrupted_atomic_publication_leaves_no_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: bypassing the temp-file rename leaves partial JSON visible."""
    import polylogue.browser_capture.receiver as receiver

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("injected publication interruption")

    monkeypatch.setattr(receiver.os, "replace", fail_replace)

    with pytest.raises(OSError, match="interruption"):
        write_capture_envelope_bytes(_raw(_payload()), spool_path=tmp_path)

    assert not tuple(tmp_path.rglob("*.json"))
    assert not tuple(tmp_path.rglob(".*.tmp"))
