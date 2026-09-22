"""Spool admission must agree with the archive's own capture precedence.

Three rules the receiver applied without them:

- A provider-native capture arriving over a DOM fallback is a fidelity
  improvement, so a lower turn count must not discard it. The archive boundary
  (`browser_capture_precedence`) already admits exactly that case, and the
  receiver refusing it first retained the lower-fidelity spool item forever.
- `provenance.captured_at` is one extension instance's observation clock, not a
  session revision, so it must not veto a strictly newer provider revision.
- A superseded delivery acknowledges the artifact that stays. Echoing the
  rejected incoming envelope's identities told the extension a branch was
  captured whose messages were never written.

Anti-vacuity is in each test's docstring; `test_stale_smaller_snapshot_is_refused`
pins the opposite direction so an admission rule that simply always publishes
cannot pass.
"""

from __future__ import annotations

import copy
from pathlib import Path

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import CaptureConvergence, write_capture_envelope

_ADAPTER = "chatgpt-dom-v1"


def _turn(turn_id: str, *, ordinal: int) -> dict[str, object]:
    return {
        "provider_turn_id": turn_id,
        "role": "user" if ordinal % 2 == 0 else "assistant",
        "text": f"turn {turn_id}",
        "ordinal": ordinal,
        "identity_observation": {
            "origin": "provider",
            "provider_message_id": turn_id,
            "adapter_name": _ADAPTER,
            "fidelity": "native",
        },
    }


def _payload(
    *,
    turn_ids: list[str],
    captured_at: str,
    updated_at: str,
    native: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "provenance": {
            "source_url": "https://chatgpt.com/c/conv-1",
            "page_title": "ChatGPT",
            "captured_at": captured_at,
            "adapter_name": _ADAPTER,
            "extension_instance_id": "instance-a",
        },
        "session": {
            "provider": "chatgpt",
            "provider_session_id": "conv-1",
            "title": "Spool admission",
            "updated_at": updated_at,
            "turns": [_turn(turn_id, ordinal=index) for index, turn_id in enumerate(turn_ids)],
        },
    }
    if native:
        payload["raw_provider_payload"] = {
            "id": "conv-1",
            "mapping": {turn_id: {"id": turn_id} for turn_id in turn_ids},
        }
    return payload


def _write(payload: dict[str, object], root: Path) -> object:
    return write_capture_envelope(BrowserCaptureEnvelope.model_validate(copy.deepcopy(payload)), spool_path=root)


def test_native_replaces_fallback_with_fewer_turns(tmp_path: Path) -> None:
    """Anti-vacuity: drop `native_over_fallback` and this is `SUPERSEDED`."""
    fallback = _payload(
        turn_ids=["t1", "t2", "t3"],
        captured_at="2026-04-24T00:00:00+00:00",
        updated_at="2026-04-24T00:00:00+00:00",
    )
    native = _payload(
        turn_ids=["t1", "t2"],
        captured_at="2026-04-24T00:01:00+00:00",
        updated_at="2026-04-24T00:00:00+00:00",
        native=True,
    )

    _write(fallback, tmp_path)
    result = _write(native, tmp_path)

    assert result.convergence is CaptureConvergence.PUBLISH
    assert result.deduplicated is False


def test_skewed_capture_clock_does_not_veto_revision(tmp_path: Path) -> None:
    """Anti-vacuity: restore the leading `captured_at` veto and this is `SUPERSEDED`."""
    observed_later = _payload(
        turn_ids=["t1"],
        captured_at="2026-04-24T00:05:00+00:00",
        updated_at="2026-04-24T00:00:00+00:00",
    )
    newer_revision = _payload(
        turn_ids=["t1", "t2"],
        captured_at="2026-04-24T00:00:30+00:00",
        updated_at="2026-04-24T00:04:00+00:00",
    )

    _write(observed_later, tmp_path)
    result = _write(newer_revision, tmp_path)

    assert result.convergence is CaptureConvergence.PUBLISH


def test_superseded_ack_names_the_retained_turns(tmp_path: Path) -> None:
    """Anti-vacuity: echo the incoming envelope and the refs become `b1`."""
    retained = _payload(
        turn_ids=["t1", "t2"],
        captured_at="2026-04-24T00:05:00+00:00",
        updated_at="2026-04-24T00:05:00+00:00",
    )
    rejected = _payload(
        turn_ids=["b1"],
        captured_at="2026-04-24T00:01:00+00:00",
        updated_at="2026-04-24T00:01:00+00:00",
    )

    _write(retained, tmp_path)
    result = _write(rejected, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED
    assert [identity.message_ref.rsplit(":", 1)[-1] for identity in result.accepted_identities] == ["t1", "t2"]


def test_stale_smaller_snapshot_is_refused(tmp_path: Path) -> None:
    """Two DOM fallbacks: the older, smaller one must still lose."""
    richer = _payload(
        turn_ids=["t1", "t2"],
        captured_at="2026-04-24T00:05:00+00:00",
        updated_at="2026-04-24T00:05:00+00:00",
    )
    stale = _payload(
        turn_ids=["t1"],
        captured_at="2026-04-24T00:01:00+00:00",
        updated_at="2026-04-24T00:01:00+00:00",
    )

    _write(richer, tmp_path)
    result = _write(stale, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED
    assert result.deduplicated is True
