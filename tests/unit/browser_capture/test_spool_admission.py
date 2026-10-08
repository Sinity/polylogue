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
from typing import Any

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    BrowserCaptureWriteResult,
    CaptureConvergence,
    write_capture_envelope,
)

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


def _write(payload: dict[str, object], root: Path) -> BrowserCaptureWriteResult:
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
    assert [str(identity.message_ref).rsplit(":", 1)[-1] for identity in result.accepted_identities] == ["t1", "t2"]


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


def test_session_attachment_does_not_shift_turn_attachment_identity(tmp_path: Path) -> None:
    """Anti-vacuity: flattened positional comparison rejects [session, turn] vs [turn]."""
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", "inline_base64": "YQ=="}]
    incoming: dict[str, Any] = _payload(
        turn_ids=["t1", "t2"], captured_at="2026-04-24T00:01:00Z", updated_at="2026-04-24T00:01:00Z"
    )
    incoming["session"]["attachments"] = [{"provider_attachment_id": "B", "content_base64": "Yg=="}]
    incoming["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", "inline_base64": "YQ=="}]

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.PUBLISH


def test_content_carrier_cannot_disagree_with_existing_inline_bytes(tmp_path: Path) -> None:
    """Anti-vacuity: accepting content_base64 without comparing inline bytes changes archive data."""
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", "inline_base64": "YQ=="}]
    incoming: dict[str, Any] = _payload(
        turn_ids=["t1", "t2"], captured_at="2026-04-24T00:01:00Z", updated_at="2026-04-24T00:01:00Z"
    )
    incoming["session"]["turns"][0]["attachments"] = [
        {"provider_attachment_id": "A", "inline_base64": "YQ==", "content_base64": "Yg=="}
    ]

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED


def test_inline_carrier_cannot_be_replaced_by_newer_turn_snapshot(tmp_path: Path) -> None:
    """Inline/data bytes are carrier evidence too, not replaceable descriptors."""
    for field_name in ("inline_base64", "data"):
        root = tmp_path / field_name
        resident: dict[str, Any] = _payload(
            turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
        )
        resident["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", field_name: "YQ=="}]
        incoming = copy.deepcopy(resident)
        incoming["provenance"]["captured_at"] = "2026-04-24T00:01:00Z"
        incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
        incoming["session"]["turns"].append(_turn("t2", ordinal=1))
        incoming["session"]["turns"][0]["attachments"][0][field_name] = "YWI="

        _write(resident, root)
        result = _write(incoming, root)

        assert result.convergence is CaptureConvergence.SUPERSEDED


def test_malformed_inline_carrier_cannot_replace_a_retained_snapshot(tmp_path: Path) -> None:
    """Malformed inline/data bytes cannot evade the invalid-carrier refusal."""
    for field_name in ("inline_base64", "data"):
        root = tmp_path / field_name
        resident: dict[str, Any] = _payload(
            turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
        )
        resident["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", field_name: "YQ=="}]
        incoming = copy.deepcopy(resident)
        incoming["provenance"]["captured_at"] = "2026-04-24T00:01:00Z"
        incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
        incoming["session"]["turns"].append(_turn("t2", ordinal=1))
        incoming["session"]["turns"][0]["attachments"][0][field_name] = "not-base64!"

        _write(resident, root)
        result = _write(incoming, root)

        assert result.convergence is CaptureConvergence.SUPERSEDED


def test_inline_carrier_enrichment_is_accepted_at_same_observation(tmp_path: Path) -> None:
    """An inline/data body can enrich an unchanged attachment without newer timestamps."""
    for field_name in ("inline_base64", "data"):
        root = tmp_path / field_name
        observed: dict[str, Any] = _payload(
            turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
        )
        observed["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A"}]
        acquired = copy.deepcopy(observed)
        acquired["session"]["turns"][0]["attachments"][0][field_name] = "YQ=="

        _write(observed, root)
        result = _write(acquired, root)

        assert result.convergence is CaptureConvergence.PUBLISH


def test_self_declared_wrong_identity_is_not_acknowledged_as_native(tmp_path: Path) -> None:
    """Anti-vacuity: copying observation fidelity alone incorrectly returns native."""
    payload: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    payload["session"]["turns"][0]["identity_observation"] = {
        "origin": "claude-ai-export",
        "provider_conversation_id": "other",
        "provider_message_id": "other-message",
        "adapter_name": "chatgpt",
        "fidelity": "native",
    }

    result = _write(payload, tmp_path)

    assert result.accepted_identities[0].fidelity == "unknown"


def test_empty_carrier_is_present_evidence_not_absence(tmp_path: Path) -> None:
    """Anti-vacuity: treating ``content_base64=""`` as absent lets different bytes replace a zero-byte carrier."""
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", "content_base64": ""}]
    incoming: dict[str, Any] = _payload(
        turn_ids=["t1", "t2"], captured_at="2026-04-24T00:01:00Z", updated_at="2026-04-24T00:01:00Z"
    )
    incoming["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", "content_base64": "YQ=="}]

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED


def test_duplicate_attachment_ids_in_one_scope_are_all_retained(tmp_path: Path) -> None:
    """Anti-vacuity: a dict keyed by scoped id keeps only the last duplicate, so dropping one passes."""
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [
        {"provider_attachment_id": "A", "inline_base64": "Yg=="},
        {"provider_attachment_id": "A", "inline_base64": "YQ=="},
    ]
    incoming: dict[str, Any] = _payload(
        turn_ids=["t1", "t2"], captured_at="2026-04-24T00:01:00Z", updated_at="2026-04-24T00:01:00Z"
    )
    incoming["session"]["turns"][0]["attachments"] = [
        {"provider_attachment_id": "A", "inline_base64": "YQ==", "content_base64": "YQ=="}
    ]

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED


def test_native_snapshot_replay_ignores_raw_attachment_coordinates_but_keeps_occurrences(tmp_path: Path) -> None:
    """Changed plan coordinates and acquired coverage do not contradict stable owners."""
    older: dict[str, Any] = _payload(
        turn_ids=["t1", "t2"],
        captured_at="2026-04-24T00:00:00Z",
        updated_at="2026-04-24T00:00:00Z",
        native=True,
    )
    older["session"]["turns"][0]["attachments"] = [
        {
            "provider_attachment_id": "shared-file",
            "message_provider_id": "m1",
            "attachment_kind": "sandbox_file",
            "name": "result.bin",
            "mime_type": "application/octet-stream",
            "provider_meta": {
                "provider_file_id": "file-1",
                "native_attachment_ordinal": 9,
                "native_turn_ordinal": 0,
                "native_raw_position": 40,
                "asset_acquisition": {"status": "recovered_bytes_unavailable"},
            },
        }
    ]
    older["session"]["turns"][1]["attachments"] = [
        {
            **older["session"]["turns"][0]["attachments"][0],
            "message_provider_id": "m2",
            "provider_meta": {
                "provider_file_id": "file-1",
                "native_attachment_ordinal": 10,
                "native_turn_ordinal": 1,
                "native_raw_position": 41,
                "asset_acquisition": {"status": "recovered_bytes_unavailable"},
            },
        }
    ]
    _write(older, tmp_path)

    newer = copy.deepcopy(older)
    newer["provenance"]["captured_at"] = "2026-04-24T00:01:00Z"
    newer["session"]["updated_at"] = "2026-04-24T00:01:00Z"
    newer["session"]["turns"].append(_turn("t3", ordinal=2))
    for owner, attachment_ordinal, raw_position in (("m1", 35, 33), ("m2", 36, 34)):
        attachment = next(
            attachment
            for turn in newer["session"]["turns"]
            for attachment in turn.get("attachments", [])
            if attachment.get("message_provider_id") == owner
        )
        attachment["size_bytes"] = 1
        attachment["content_base64"] = "YQ=="
        attachment["provider_meta"].update(
            {
                "native_attachment_ordinal": attachment_ordinal,
                "native_turn_ordinal": 0 if owner == "m1" else 1,
                "native_raw_position": raw_position,
                "asset_acquisition": {"status": "acquired"},
                "content_sha256": "a" * 64,
            }
        )

    accepted = _write(newer, tmp_path)

    assert accepted.convergence is CaptureConvergence.PUBLISH
    assert accepted.accepted_identities
    assert [str(identity.message_ref).rsplit(":", 1)[-1] for identity in accepted.accepted_identities] == [
        "t1",
        "t2",
        "t3",
    ]
    stored = BrowserCaptureEnvelope.model_validate_json(accepted.path.read_bytes())
    assert stored.session.turns[0].attachments[0].provider_meta["asset_acquisition"] == {"status": "acquired"}
    assert stored.session.turns[1].attachments[0].provider_meta["native_raw_position"] == 34

    stale = _write(older, tmp_path)

    assert stale.convergence is CaptureConvergence.SUPERSEDED
    assert [str(identity.message_ref).rsplit(":", 1)[-1] for identity in stale.accepted_identities] == [
        "t1",
        "t2",
        "t3",
    ]


def test_same_revision_acquisition_enrichment_needs_no_newer_observation(tmp_path: Path) -> None:
    """Carrier receipt metadata and an absent size can be enriched at the same observation."""
    unavailable: dict[str, Any] = _payload(
        turn_ids=["t1"],
        captured_at="2026-04-24T00:00:00Z",
        updated_at="2026-04-24T00:00:00Z",
        native=True,
    )
    unavailable["session"]["turns"][0]["attachments"] = [
        {
            "provider_attachment_id": "file-1",
            "message_provider_id": "m1",
            "attachment_kind": "sandbox_file",
            "provider_meta": {
                "provider_file_id": "file-1",
                "native_attachment_ordinal": 7,
                "native_turn_ordinal": 0,
                "native_raw_position": 21,
                "asset_acquisition": {"status": "recovered_bytes_unavailable"},
            },
        }
    ]
    _write(unavailable, tmp_path)
    acquired = copy.deepcopy(unavailable)
    acquired_attachment = acquired["session"]["turns"][0]["attachments"][0]
    acquired_attachment["size_bytes"] = 1
    acquired_attachment["content_base64"] = "YQ=="
    acquired_attachment["provider_meta"].update(
        {
            "native_attachment_ordinal": 3,
            "native_raw_position": 15,
            "asset_acquisition": {"status": "acquired"},
            "content_sha256": "a" * 64,
        }
    )

    result = _write(acquired, tmp_path)

    assert result.convergence is CaptureConvergence.PUBLISH
    assert result.deduplicated is False


def test_known_attachment_size_change_remains_a_conflict(tmp_path: Path) -> None:
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [
        {
            "provider_attachment_id": "A",
            "message_provider_id": "m1",
            "size_bytes": 1,
            "content_base64": "YQ==",
        }
    ]
    incoming = copy.deepcopy(resident)
    incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["turns"].append(_turn("t2", ordinal=1))
    incoming["session"]["turns"][0]["attachments"][0]["size_bytes"] = 2
    incoming["session"]["turns"][0]["attachments"][0]["content_base64"] = "YWI="

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED


def test_stable_provider_attachment_metadata_remains_a_conflict(tmp_path: Path) -> None:
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [
        {
            "provider_attachment_id": "A",
            "message_provider_id": "m1",
            "content_base64": "YQ==",
            "provider_meta": {"provider_file_id": "provider-file-1"},
        }
    ]
    incoming = copy.deepcopy(resident)
    incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["turns"].append(_turn("t2", ordinal=1))
    incoming["session"]["turns"][0]["attachments"][0]["provider_meta"]["provider_file_id"] = "provider-file-2"
    incoming["session"]["turns"][0]["attachments"][0]["content_base64"] = "YWI="

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED


def test_idless_attachment_keeps_declared_raw_ordinal_identity(tmp_path: Path) -> None:
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [
        {
            "provider_attachment_id": "A",
            "provider_meta": {"native_turn_ordinal": 0, "native_attachment_ordinal": 0},
        }
    ]
    incoming = copy.deepcopy(resident)
    incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["turns"].append(_turn("t2", ordinal=1))
    incoming_attachment = incoming["session"]["turns"][0]["attachments"][0]
    incoming_attachment["content_base64"] = "YQ=="
    incoming_attachment["provider_meta"]["native_turn_ordinal"] = 1

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED


def test_newer_metadata_only_attachment_revision_keeps_freshness_admission(tmp_path: Path) -> None:
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["turns"][0]["attachments"] = [{"provider_attachment_id": "A", "name": "temporary.bin"}]
    incoming = copy.deepcopy(resident)
    incoming["provenance"]["captured_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["turns"][0]["attachments"][0]["name"] = "final.bin"
    incoming["session"]["turns"].append(_turn("t2", ordinal=1))

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.PUBLISH


def test_session_attachment_owner_insertion_does_not_shift_existing_carriers(tmp_path: Path) -> None:
    """A new owner's repeated ID does not re-pair other session-level occurrences."""
    resident: dict[str, Any] = _payload(
        turn_ids=["t1", "t2"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["attachments"] = [
        {"provider_attachment_id": "shared", "message_provider_id": owner, "content_base64": "YQ=="}
        for owner in ("m1", "m2")
    ]
    incoming = copy.deepcopy(resident)
    incoming["provenance"]["captured_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["updated_at"] = "2026-04-24T00:01:00Z"
    incoming["session"]["attachments"].insert(
        0,
        {"provider_attachment_id": "shared", "message_provider_id": "m0", "content_base64": "Yg=="},
    )
    incoming["session"]["turns"].append(_turn("t3", ordinal=2))

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.PUBLISH


def test_empty_message_owner_keeps_declared_ordinal_identity(tmp_path: Path) -> None:
    """An empty owner ID is id-less; changing its ordinal cannot enrich bytes."""
    resident: dict[str, Any] = _payload(
        turn_ids=["t1"], captured_at="2026-04-24T00:00:00Z", updated_at="2026-04-24T00:00:00Z"
    )
    resident["session"]["attachments"] = [
        {
            "provider_attachment_id": "shared",
            "message_provider_id": "",
            "provider_meta": {"native_attachment_ordinal": 0, "native_turn_ordinal": 0},
        }
    ]
    incoming = copy.deepcopy(resident)
    incoming["session"]["attachments"][0]["provider_meta"]["native_attachment_ordinal"] = 1
    incoming["session"]["attachments"][0]["content_base64"] = "YQ=="

    _write(resident, tmp_path)
    result = _write(incoming, tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED
