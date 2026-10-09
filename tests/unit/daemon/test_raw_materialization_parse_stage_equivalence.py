"""Raw materialization hands its current output to canonical session derivation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.config import Config
from polylogue.core.compute import BoundedComputeAdapter
from polylogue.core.enums import Provider, ValidationMode, ValidationStatus
from polylogue.daemon import cli as daemon_cli
from polylogue.daemon.session_profile_composition import compose_session_profile_callback
from polylogue.daemon.write_coordinator import DaemonWriteCoordinator, DaemonWriteThreadBridge
from polylogue.schemas import RetainedValidationVerdict
from polylogue.schemas.drift_sentinel import DriftSignature, SchemaDriftObservation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from tests.infra.raw_owner_routes import converge_pending_raws_with_owner, replay_retained_raws


def _codex_session(native_id: str, messages: tuple[tuple[str, str], ...]) -> bytes:
    rows: list[dict[str, object]] = [
        {"type": "session_meta", "payload": {"id": native_id, "timestamp": "2026-07-19T00:00:00Z"}}
    ]
    for position, (role, text) in enumerate(messages):
        rows.append(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-m{position}",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text" if role == "user" else "output_text",
                            "text": text,
                        }
                    ],
                },
            }
        )
    return b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[])


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.mark.asyncio
async def test_raw_materialization_hands_current_output_to_the_canonical_session_derivation(tmp_path: Path) -> None:
    """Raw admission targets its actual output after releasing the writer lease.

    Anti-vacuity: removing the raw-to-session query or calling the profile
    owner before raw publication leaves the materialized session without its
    canonical profile partition. Repeating the handoff proves that inspection
    rather than the intake item is the source of idempotence.
    """
    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)

    def acquire() -> str:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return archive.write_raw_payload(
                provider=Provider.CODEX,
                payload=_codex_session("raw-profile-handoff", (("user", "question"), ("assistant", "answer"))),
                source_path="raw-profile-handoff.jsonl",
                canonical_source_path="raw-profile-handoff.jsonl",
                acquired_at_ms=1,
            )

    raw_id = await asyncio.to_thread(acquire)
    result = await asyncio.to_thread(
        converge_pending_raws_with_owner,
        archive_root,
        limit=1,
    )
    assert result.done == 1 and result.failed == 0
    session_ids = daemon_cli._raw_materialized_session_ids(archive_root, raw_id)
    assert session_ids == ("codex-session:raw-profile-handoff",)

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, composed.callback)
        with _connect(archive_root / "index.db") as conn:
            first = tuple(
                conn.execute(
                    "SELECT session_id, input_content_hash FROM session_profiles WHERE session_id = ?",
                    session_ids,
                ).fetchall()
            )
        assert len(first) == 1

        await daemon_cli._converge_raw_materialized_session_profiles(archive_root, raw_id, composed.callback)
        with _connect(archive_root / "index.db") as conn:
            second = tuple(
                conn.execute(
                    "SELECT session_id, input_content_hash FROM session_profiles WHERE session_id = ?",
                    session_ids,
                ).fetchall()
            )
        assert second == first
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


@pytest.mark.asyncio
async def test_strict_retained_validation_refusal_does_not_publish_marker_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)
    from tests.infra.retained_jsonl import acquire_full_revision

    source_path = tmp_path / "strict-marker.jsonl"
    payload = _codex_session("strict-marker", (("user", "question"), ("assistant", "::note: retained note")))

    def acquire() -> str:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return acquire_full_revision(
                archive,
                provider=Provider.CODEX,
                source_path=source_path,
                payload=payload,
                native_id="strict-marker",
                generation=0,
                acquired_at_ms=1,
            )

    raw_id = await asyncio.to_thread(acquire)

    validations: list[tuple[str, str, ValidationMode]] = []

    def refuse_retained_document(
        _provider: Provider,
        _path: Path,
        *,
        mode: ValidationMode,
        raw_id: str,
        revision_sha256: str,
        evidence_id: str,
        **_kwargs: object,
    ) -> RetainedValidationVerdict:
        validations.append((raw_id, revision_sha256, mode))
        return RetainedValidationVerdict(
            raw_id=raw_id,
            revision_sha256=revision_sha256,
            evidence_id=evidence_id,
            mode=mode,
            status=ValidationStatus.FAILED,
            sample_count=1,
            invalid_count=1,
            error_count=0,
            drift_count=0,
            first_diagnostic="synthetic strict schema refusal",
            schema_resolution=None,
            drift_observation=SchemaDriftObservation(
                origin="codex-session",
                element_kind="session_record",
                classification="new_field",
                unseen_key_signature=DriftSignature.from_text("payload.synthetic", directory=tmp_path),
                native_id_example="strict-marker",
                raw_id=raw_id,
            ),
            strict_refusal=True,
        )

    monkeypatch.setattr("polylogue.schemas.validate_retained_document", refuse_retained_document)
    from tests.infra.live_ingest import prepared_live_convergence_owner

    async with prepared_live_convergence_owner(archive_root, validation_mode=ValidationMode.STRICT) as owner:
        first = await owner.converge_raw_id(raw_id)
        second = await owner.converge_raw_id(raw_id)
    assert first.failed == second.failed == 0, (first.outcomes, second.outcomes, validations)
    assert second.pending == 0, (first.outcomes, second.outcomes, validations)

    assert len(validations) == 1
    assert validations[0][0] == raw_id
    assert validations[0][2] is ValidationMode.STRICT
    with sqlite3.connect(archive_root / "source.db") as source:
        assert source.execute(
            "SELECT validation_status,validation_mode,parse_error FROM raw_sessions WHERE raw_id=?", (raw_id,)
        ).fetchone() == ("failed", "strict", None)
        assert source.execute("SELECT COUNT(*) FROM accepted_marker_inputs WHERE raw_id=?", (raw_id,)).fetchone() == (
            0,
        )
        assert source.execute(
            "SELECT COUNT(*) FROM raw_artifacts WHERE raw_id=? AND parse_as_session=1 AND schema_eligible=1",
            (raw_id,),
        ).fetchone() == (1,)
    with sqlite3.connect(archive_root / "ops.db") as ops:
        assert ops.execute("SELECT COUNT(*) FROM schema_drift_samples WHERE raw_id=?", (raw_id,)).fetchone() == (1,)
    assert len(validations) == 1


@pytest.mark.asyncio
async def test_two_accepted_revisions_survive_one_periodic_profile_pass(tmp_path: Path) -> None:
    """Accepted R1/R2 marker inputs survive a coalesced profile publication.

    Two retained revisions of one Codex session each carry a note marker. They
    publish through the canonical raw-observation route in arrival order.
    """
    archive_root = tmp_path / "archive"
    await asyncio.to_thread(initialize_active_archive_root, archive_root)
    from tests.infra.retained_jsonl import retained_append_fixture

    source_path = tmp_path / "coalesced-profile.jsonl"
    native_id = "coalesced-profile"
    baseline = _codex_session(native_id, (("user", "question"), ("assistant", "::note: first retained note")))
    delta = (
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "id": f"{native_id}-append-m0",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "::note: second retained note"}],
                },
            },
            sort_keys=True,
        ).encode()
        + b"\n"
    )

    def acquire() -> tuple[str, str]:
        with retained_append_fixture(
            root=archive_root,
            provider=Provider.CODEX,
            source_path=source_path,
            native_id=native_id,
            logical_source_key=f"codex-session:{native_id}",
            baseline=baseline,
            delta=delta,
        ) as (_reader, baseline_raw_id, append_raw_id, *_evidence):
            return baseline_raw_id, append_raw_id

    baseline_raw_id, append_raw_id = await asyncio.to_thread(acquire)
    assert baseline_raw_id != append_raw_id

    # Prove these are both accepted members of one chain before exercising one
    # retained preparation. This cannot pass by replaying R1 and R2 separately.
    with ArchiveStore.open_existing(archive_root, read_only=True) as archive:
        plan = archive.raw_revision_replay_plan(f"codex-session:{native_id}")
        assert plan.accepted_raw_ids == (baseline_raw_id, append_raw_id)

    await asyncio.to_thread(replay_retained_raws, archive_root, (baseline_raw_id, append_raw_id))

    with sqlite3.connect(archive_root / "source.db") as source:
        retained = source.execute("SELECT sequence, raw_id FROM accepted_marker_inputs ORDER BY sequence").fetchall()
        assert len(retained) == 2
        assert retained[0][0] < retained[1][0]
        assert source.execute(
            "SELECT DISTINCT validation_mode FROM raw_sessions WHERE source_path=?", (str(source_path),)
        ).fetchall() == [(ValidationMode.ADVISORY.value,)]

    compute = BoundedComputeAdapter(max_workers=1, queue_units=1)
    coordinator = DaemonWriteCoordinator(archive_root=archive_root)
    try:
        composed = compose_session_profile_callback(
            archive_root,
            compute_adapter=compute,
            write_bridge=DaemonWriteThreadBridge(coordinator, asyncio.get_running_loop()),
            now=lambda: 0.0,
        )
        periodic = await composed.callback(None)
        assert periodic.outcomes
        # Marker delivery intentionally advances one accepted source batch per
        # transaction. The scheduler may drain both transactions in this tick
        # or leave R2 for the next one, without republishing the already-current
        # profile of the coalesced index revision.
        second = await composed.callback(None)
        assert all(item.key.domain != "session_profile" for item in second.outcomes)
        with sqlite3.connect(archive_root / "user.db") as user:
            assert user.execute("SELECT applied_sequence FROM accepted_marker_delivery_cursor").fetchone() == (2,)
        with sqlite3.connect(archive_root / "index.db") as index:
            assert index.execute(
                "SELECT COUNT(*) FROM session_profiles WHERE session_id = ?", ("codex-session:coalesced-profile",)
            ).fetchone() == (1,)
            assert (
                index.execute(
                    "SELECT 1 FROM session_profile_demand WHERE session_id = ?", ("codex-session:coalesced-profile",)
                ).fetchone()
                is None
            )
        with sqlite3.connect(archive_root / "user.db") as user:
            bodies = [str(row[0]) for row in user.execute("SELECT body_text FROM assertions ORDER BY body_text")]
            assert any("first retained note" in body for body in bodies)
            assert any("second retained note" in body for body in bodies)
        unchanged = await composed.callback(None)
        assert unchanged.made_no_publication_attempts
        assert unchanged.work.inspected == 0
    finally:
        compute.shutdown(wait=True)
        await coordinator.shutdown(timeout=1.0)


def test_raw_materialized_session_ids_exclude_stale_component_sessions_without_current_heads(tmp_path: Path) -> None:
    """Raw-to-profile handoff follows authoritative heads, not residual session rows.

    Anti-vacuity: querying ``sessions`` by raw component alone includes the
    deliberately orphaned split member below and schedules a non-current
    session partition.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload: list[dict[str, object]] = [
        {
            "id": native_id,
            "title": native_id,
            "create_time": 1,
            "current_node": "m",
            "mapping": {
                "m": {
                    "id": "m",
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "create_time": 1,
                        "content": {"content_type": "text", "parts": [native_id]},
                    },
                }
            },
        }
        for native_id in ("active", "stale")
    ]
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CHATGPT,
            payload=json.dumps(payload).encode(),
            source_path="split.json",
            canonical_source_path="split.json",
            acquired_at_ms=1,
        )
    result = converge_pending_raws_with_owner(archive_root, limit=1)
    assert result.done == 1 and result.failed == 0
    with sqlite3.connect(archive_root / "index.db") as index:
        index.execute("DELETE FROM raw_revision_heads WHERE session_id = ?", ("chatgpt-export:stale",))
        index.commit()

    assert daemon_cli._raw_materialized_session_ids(archive_root, raw_id) == ("chatgpt-export:active",)
