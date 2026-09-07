"""Production browser-capture admission laws used by source restoration.

These tests intentionally exercise the ordinary receiver writer.  The
one-time restoration manifest and its retained bytes are not test fixtures or
repository authority; this file only protects the reusable admission
boundary that consumes them.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import resource
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from polylogue.browser_capture.models import BrowserCaptureEnvelope
from polylogue.browser_capture.receiver import (
    BrowserCaptureSpoolConflictError,
    CaptureConvergence,
    capture_artifact_path,
    capture_convergence,
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


def _rss_write_worker(raw: bytes, spool_path: str, result_queue: object) -> None:
    """Measure one production admission in a fresh child process."""
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_capture_envelope_bytes(raw, spool_path=Path(spool_path))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_queue.put((baseline, peak))  # type: ignore[attr-defined]


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


def test_attachment_content_enrichment_publishes_and_preserves_exact_bytes(tmp_path: Path) -> None:
    """Only an added valid carrier may enrich an otherwise identical resident."""
    resident = _payload()
    resident["session"]["turns"][0]["attachments"] = [  # type: ignore[index]
        {"provider_attachment_id": "upload-1", "name": "payload.bin"}
    ]
    enriched = json.loads(json.dumps(resident))
    enriched["session"]["turns"][0]["attachments"][0]["content_base64"] = "aGVsbG8="

    resident_raw = _raw(resident)
    enriched_raw = _raw(enriched)
    first = write_capture_envelope_bytes(resident_raw, spool_path=tmp_path)
    second = write_capture_envelope_bytes(enriched_raw, spool_path=tmp_path)

    assert first.convergence is CaptureConvergence.PUBLISH
    assert second.convergence is CaptureConvergence.PUBLISH
    assert second.path.read_bytes() == enriched_raw
    assert (
        capture_convergence(
            BrowserCaptureEnvelope.model_validate(enriched),
            BrowserCaptureEnvelope.model_validate(resident),
        )
        is CaptureConvergence.PUBLISH
    )

    reverse_root = tmp_path / "reverse"
    reverse_root.mkdir()
    reverse_first = write_capture_envelope_bytes(enriched_raw, spool_path=reverse_root)
    reverse_second = write_capture_envelope_bytes(resident_raw, spool_path=reverse_root)
    assert reverse_first.path.read_bytes() == enriched_raw
    assert reverse_second.convergence is CaptureConvergence.SUPERSEDED
    assert reverse_second.path.read_bytes() == enriched_raw


def test_attachment_enrichment_is_rejected_for_stale_changed_or_invalid_input() -> None:
    resident = _payload()
    resident["session"]["turns"][0]["attachments"] = [  # type: ignore[index]
        {"provider_attachment_id": "upload-1", "name": "payload.bin"}
    ]
    enriched = json.loads(json.dumps(resident))
    enriched["session"]["turns"][0]["attachments"][0]["content_base64"] = "aGVsbG8="

    changed = json.loads(json.dumps(enriched))
    changed["session"]["turns"][0]["text"] = "different"
    invalid = json.loads(json.dumps(enriched))
    invalid["session"]["turns"][0]["attachments"][0]["content_base64"] = "not base64"
    conflicting = json.loads(json.dumps(enriched))
    conflicting["session"]["turns"][0]["attachments"][0]["content_base64"] = "d29ybGQ="
    changed_observation = json.loads(json.dumps(enriched))
    changed_observation["provenance"]["adapter_version"] = "different"

    resident_model = BrowserCaptureEnvelope.model_validate(resident)
    assert (
        capture_convergence(BrowserCaptureEnvelope.model_validate(changed), resident_model)
        is CaptureConvergence.SUPERSEDED
    )
    assert (
        capture_convergence(BrowserCaptureEnvelope.model_validate(invalid), resident_model)
        is CaptureConvergence.SUPERSEDED
    )
    assert (
        capture_convergence(
            BrowserCaptureEnvelope.model_validate(conflicting), BrowserCaptureEnvelope.model_validate(enriched)
        )
        is CaptureConvergence.SUPERSEDED
    )
    assert (
        capture_convergence(BrowserCaptureEnvelope.model_validate(changed_observation), resident_model)
        is CaptureConvergence.SUPERSEDED
    )


def test_attachment_enrichment_rereads_resident_after_lock_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resident changed before the locked decision wins over stale enrichment."""
    import polylogue.browser_capture.receiver as receiver

    resident = _payload()
    resident["session"]["turns"][0]["attachments"] = [  # type: ignore[index]
        {"provider_attachment_id": "upload-1", "name": "payload.bin"}
    ]
    enriched = json.loads(json.dumps(resident))
    enriched["session"]["turns"][0]["attachments"][0]["content_base64"] = "aGVsbG8="
    drifted = json.loads(json.dumps(resident))
    drifted["session"]["turns"][0]["text"] = "resident changed while waiting for lock"

    first = write_capture_envelope_bytes(_raw(resident), spool_path=tmp_path)
    drifted_raw = _raw(drifted)
    original_lock = receiver._spool_file_lock

    @contextmanager
    def lock_with_drift(root: Path):
        with original_lock(root):
            first.path.write_bytes(drifted_raw)
            yield

    monkeypatch.setattr(receiver, "_spool_file_lock", lock_with_drift)
    result = write_capture_envelope_bytes(_raw(enriched), spool_path=tmp_path)

    assert result.convergence is CaptureConvergence.SUPERSEDED
    assert result.path.read_bytes() == drifted_raw


@pytest.mark.parametrize("body_bytes", (4 * 1024 * 1024, 8 * 1024 * 1024, 16 * 1024 * 1024))
def test_large_capture_admission_stays_within_memory_bound(tmp_path: Path, body_bytes: int) -> None:
    """Anti-vacuity: large carriers cannot trigger whole-envelope amplification."""
    payload = _payload(session_id=f"rss-{body_bytes}")
    session = cast(dict[str, object], payload["session"])
    turns = cast(list[dict[str, object]], session["turns"])
    turn = turns[0]
    turn["text"] = "x" * (body_bytes - 1_000)
    turn["attachments"] = [
        {
            "provider_attachment_id": "rss-attachment",
            "name": "payload.bin",
            "content_base64": "AAECAwQF",
        }
    ]
    raw = _raw(payload)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_rss_write_worker, args=(raw, str(tmp_path), result_queue))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    baseline, peak = result_queue.get(timeout=5)
    delta_bytes = (peak - baseline) * 1024
    bound = max(256 * 1024 * 1024, 12 * len(raw))
    print(f"capture_rss body={body_bytes} input={len(raw)} delta={delta_bytes} bound={bound}")
    assert delta_bytes <= bound


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

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("injected publication interruption")

    monkeypatch.setattr("polylogue.browser_capture.receiver.os.replace", fail_replace)

    with pytest.raises(OSError, match="interruption"):
        write_capture_envelope_bytes(_raw(_payload()), spool_path=tmp_path)

    assert not tuple(tmp_path.rglob("*.json"))
    assert not tuple(tmp_path.rglob(".*.tmp"))


def _replay(entries: list[tuple[str, bytes]], spool: Path, ledger: Path) -> dict[str, str]:
    """Admit recorded payloads through the production route, once each.

    Mirrors how a controlled restoration consumes retained bytes: verify the
    recorded digest, admit, then record the receipt. The ledger is the resume
    point, so an interruption before the receipt is written replays that item.
    """
    done = {line.split(" ", 1)[0] for line in ledger.read_text().splitlines() if line} if ledger.exists() else set()
    outcomes: dict[str, str] = {}
    with ledger.open("a", encoding="utf-8") as handle:
        for identity, raw in entries:
            if identity in done:
                outcomes[identity] = "already-received"
                continue
            if hashlib.sha256(raw).hexdigest() != identity:
                outcomes[identity] = "refused-digest"
                continue
            result = write_capture_envelope_bytes(raw, spool_path=spool)
            outcomes[identity] = "deduplicated" if result.deduplicated else "admitted"
            handle.write(f"{identity} {outcomes[identity]}\n")
            handle.flush()
    return outcomes


def _entry(session_id: str, *, fidelity: str = "native") -> tuple[str, bytes]:
    raw = _raw(_payload(session_id=session_id, fidelity=fidelity), indent=2)
    return hashlib.sha256(raw).hexdigest(), raw


def test_replay_admits_every_recorded_payload_and_a_dropped_one_stays_missing(tmp_path: Path) -> None:
    """Anti-vacuity: silently skipping a payload leaves its artifact absent."""
    entries = [_entry(f"session-{index}") for index in range(4)]

    _replay(entries[:-1], tmp_path, tmp_path / "ledger")

    published = {path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert published == {raw for _, raw in entries[:-1]}
    assert entries[-1][1] not in published


def test_altered_payload_byte_is_refused_before_admission(tmp_path: Path) -> None:
    """Anti-vacuity: admitting unverified bytes would publish corrupted content."""
    identity, raw = _entry("session-altered")
    tampered = raw.replace(b"captured text", b"captured texs")
    assert len(tampered) == len(raw)

    outcomes = _replay([(identity, tampered)], tmp_path, tmp_path / "ledger")

    assert outcomes[identity] == "refused-digest"
    assert not tuple(tmp_path.rglob("*.json"))


def test_duplicate_logical_identity_converges_on_one_artifact(tmp_path: Path) -> None:
    """Anti-vacuity: one session captured twice must not publish two artifacts."""
    first = _entry("session-duplicated")
    second_raw = _raw(_payload(session_id="session-duplicated"))
    second = (hashlib.sha256(second_raw).hexdigest(), second_raw)
    assert first[0] != second[0]

    outcomes = _replay([first, second], tmp_path, tmp_path / "ledger")

    assert outcomes[first[0]] == "admitted"
    assert outcomes[second[0]] == "deduplicated"
    assert len(tuple(tmp_path.rglob("*.json"))) == 1


def test_replay_keeps_later_richer_capture_when_earlier_update_was_capture_fallback(tmp_path: Path) -> None:
    fallback_payload = _payload()
    fallback_payload["provenance"]["captured_at"] = "2026-07-10T18:06:24.501Z"  # type: ignore[index]
    fallback_payload["session"]["updated_at"] = "2026-07-10T18:06:24.501Z"  # type: ignore[index]
    fallback_payload["session"]["turns"] = [  # type: ignore[index]
        {"provider_turn_id": f"fallback-{index}", "role": "assistant", "text": "old"} for index in range(14)
    ]

    richer_payload = json.loads(json.dumps(fallback_payload))
    richer_payload["provenance"]["captured_at"] = "2026-07-12T20:59:18.942Z"
    richer_payload["session"]["updated_at"] = "2026-07-10T18:06:11.776Z"
    richer_payload["session"]["turns"] = [
        {"provider_turn_id": f"richer-{index}", "role": "assistant", "text": "new"} for index in range(120)
    ]

    entries = [(hashlib.sha256(raw).hexdigest(), raw) for raw in (_raw(fallback_payload), _raw(richer_payload))]
    outcomes = _replay(entries, tmp_path, tmp_path / "ledger")

    assert list(outcomes.values()) == ["admitted", "admitted"]
    stored = json.loads(next(tmp_path.rglob("*.json")).read_bytes())
    stored_session = stored.get("session")
    assert isinstance(stored_session, dict)
    stored_turns = stored_session.get("turns")
    assert isinstance(stored_turns, list)
    assert len(stored_turns) == 120


def test_interruption_between_publication_and_receipt_resumes_without_double_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anti-vacuity: resuming from an unwritten receipt must not admit twice."""
    ledger = tmp_path / "ledger"
    entries = [_entry("session-interrupted")]

    def refuse_second_publication(source: Path, target: Path) -> None:
        raise AssertionError("resume republished an already-admitted capture")

    _replay(entries, tmp_path, ledger)
    published = tmp_path / "chatgpt"
    before = {path.name: path.read_bytes() for path in published.glob("*.json")}
    ledger.write_text("")  # the receipt never reached the ledger
    monkeypatch.setattr("polylogue.browser_capture.receiver.os.replace", refuse_second_publication)

    outcomes = _replay(entries, tmp_path, ledger)

    assert outcomes[entries[0][0]] == "deduplicated"
    assert {path.name: path.read_bytes() for path in published.glob("*.json")} == before
    assert ledger.read_text().split()[1] == "deduplicated"
