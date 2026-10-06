"""Retained replay revalidates its accepted head before writer mutation.

The canonical route prepares a raw's replay off the writer against the
accepted head it read. If a concurrent publication moves that head before the
prepared replacement is published, the stale replacement must not write: the
head row is an observed input of the original reference seal.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.stage_admission import admit_stage_write
from polylogue.daemon.derivation import Budget, DerivationRegistry, converge
from polylogue.operations.raw_observation_derivation import raw_observation_frame
from polylogue.storage.derived.raw import RawObservationDerivation
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.reference_seal import ReferenceSealError
from tests.infra.archive_templates import bootstrap_archive_root
from tests.infra.prepared_replay import run_on_convergence_owner


def _record(index: int) -> bytes:
    return (
        b'{"type":"response_item","payload":{"type":"message","id":"m-'
        + str(index).encode()
        + b'","role":"user","content":[{"type":"input_text","text":"turn '
        + str(index).encode()
        + b'"}]}}\n'
    )


def _revision(turns: int) -> bytes:
    return b'{"type":"session_meta","payload":{"id":"head-race"}}\n' + b"".join(_record(i) for i in range(turns))


def _admit(root: Path, payload: bytes, acquired_at_ms: int) -> str:
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        return archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=payload,
            source_path=str(root / "sessions" / "head-race.jsonl"),
            canonical_source_path=str(root / "sessions" / "head-race.jsonl"),
            acquired_at_ms=acquired_at_ms,
        )


def _converge(root: Path, compute: object, raw_id: str) -> int:
    report = converge(
        DerivationRegistry((RawObservationDerivation(root, compute_adapter=compute),)),  # type: ignore[arg-type]
        raw_observation_frame(root, raw_ids=(raw_id,)),
        budget=Budget(page=4, discovery=4, inspection=8, compute=4, publication=4),
        publisher=admit_stage_write,
    )
    assert report.failed == 0, report.outcomes
    return report.done


def _head(root: Path) -> tuple[object, ...] | None:
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        row = index.execute(
            "SELECT accepted_raw_id, accepted_content_hash FROM raw_revision_heads "
            "WHERE logical_source_key = 'codex-session:head-race'"
        ).fetchone()
        return None if row is None else tuple(row)


def _turns(root: Path) -> int:
    with ArchiveStore.open_existing(root, read_only=True) as archive:
        index = archive.index_connection
        assert index is not None
        return int(
            index.execute("SELECT COUNT(*) FROM messages WHERE session_id = 'codex-session:head-race'").fetchone()[0]
        )


def test_prepared_replay_refuses_after_a_concurrent_head_move(tmp_path: Path) -> None:
    """Anti-vacuity: drop the accepted-head observer registration in
    ``prepare_membership_head_plan`` (or skip ``validate_observers_current`` in
    publication) and the stale two-turn replacement overwrites the three-turn head.
    """
    bootstrap_archive_root(tmp_path)
    (tmp_path / "sessions").mkdir()
    first = _admit(tmp_path, _revision(1), 1)
    assert run_on_convergence_owner(tmp_path, "test.head.first", lambda compute: _converge(tmp_path, compute, first))
    second = _admit(tmp_path, _revision(2), 2)

    def scenario(compute: object) -> tuple[bool | None, str | None]:
        adapter = RawObservationDerivation(tmp_path, compute_adapter=compute)  # type: ignore[arg-type]
        frame = raw_observation_frame(tmp_path, raw_ids=(second,))
        stale = adapter.compute(frame, second)
        # A concurrent ingest accepts a newer revision of the same session.
        third = _admit(tmp_path, _revision(3), 3)
        assert _converge(tmp_path, compute, third) >= 1
        moved = _head(tmp_path)
        assert moved is not None and moved[0] == third
        try:
            published = admit_stage_write("test.head.stale", lambda: adapter.publish(frame, stale))
        except ReferenceSealError as refusal:
            return None, type(refusal).__name__
        return published, None

    published, refusal = run_on_convergence_owner(tmp_path, "test.head.race", scenario)
    assert published is False or refusal is not None
    head = _head(tmp_path)
    assert head is not None and head[0] != second
    assert _turns(tmp_path) == 3
