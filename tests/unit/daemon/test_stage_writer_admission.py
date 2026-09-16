"""The convergence stage engine must not hold the writer across stage compute.

``converge_batch`` used to run entirely inside ``DaemonWriteCoordinator.run_sync``,
so a Drive download, an archive-wide graph materialization and a Sinex transport
drain all executed with the sole archive writer held and every other writer
queued behind them (polylogue-ssplv). The engine now runs off the lease and each
stage brackets only its own short publication with ``admit_stage_write``.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

from polylogue.core.stage_admission import stage_write_admission
from polylogue.daemon.convergence import DaemonConverger
from polylogue.operations.attachment_convergence import make_attachment_convergence_stage
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from tests.unit.daemon.test_attachment_convergence import _open_index, _session


class _SerializingWriter:
    """Stands in for the daemon's sole writer: one holder, and observable."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.admitted_actors: list[str] = []

    def admission(self, actor: str, work: object) -> object:
        assert callable(work)
        with self._lock:
            self.admitted_actors.append(actor)
            return work()

    def free_right_now(self) -> bool:
        """Whether an unrelated publication could be admitted at this instant."""
        if not self._lock.acquire(blocking=False):
            return False
        self._lock.release()
        return True


def _seed_drive_attachments(tmp_path: Path) -> tuple[sqlite3.Connection, sqlite3.Connection, dict[str, bytes]]:
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("slow-one", file_id="drive-slow"), raw_id="slow-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    return index, source, {"drive-slow": b"attachment bytes behind a slow provider"}


def test_drive_download_runs_with_the_writer_free(tmp_path: Path) -> None:
    """A slow attachment download leaves the daemon writer available.

    Anti-vacuity: give the attachment stage back ``writer_admission="whole_execute"``
    (so the engine brackets the whole ``execute``), or restore the caller that
    wrapped ``converge_batch`` in ``run_sync``, and ``writer_free_during_download``
    becomes ``[False]`` -- the download would again be inside the writer hold.
    The test is not vacuous on the stage simply not running either: the pass is
    asserted to have downloaded exactly once and to have admitted its
    publication.
    """
    index, source, payloads = _seed_drive_attachments(tmp_path)
    writer = _SerializingWriter()
    writer_free_during_download: list[bool] = []

    class _SlowDriveClient:
        def download_bytes(self, file_id: str) -> bytes:
            # "Slow" is expressed as an observation, not a sleep: the question
            # is whether an independent publication could be admitted while
            # this request is in flight, and that is decidable right here.
            writer_free_during_download.append(writer.free_right_now())
            return payloads[file_id]

    stage = make_attachment_convergence_stage(
        tmp_path / "index.db",
        archive_root=tmp_path,
        client_factory=_SlowDriveClient,
        limit=10,
    )
    assert stage.writer_admission == "bridged"

    converger = DaemonConverger(stages=(stage,))
    with stage_write_admission(writer.admission):
        states, _timings = converger.converge_batch((tmp_path / "source-batch.jsonl",))

    assert states, "the stage pass produced no subject state"
    assert writer_free_during_download == [True], (
        "the Drive download ran while the daemon writer was held; stage compute must happen outside writer admission"
    )
    assert writer.admitted_actors == ["convergence.stage.attachment_bytes.publish"]

    index.close()
    source.close()
    verify = sqlite3.connect(tmp_path / "index.db")
    try:
        statuses = [str(row[0]) for row in verify.execute("SELECT acquisition_status FROM attachments")]
    finally:
        verify.close()
    assert statuses == ["acquired"], "the bridged publication must still be durable"
    blob_hash = hashlib.sha256(payloads["drive-slow"]).digest()
    check_source = sqlite3.connect(tmp_path / "source.db")
    try:
        rows = check_source.execute("SELECT 1 FROM blob_refs WHERE blob_hash = ?", (blob_hash,)).fetchall()
    finally:
        check_source.close()
    assert rows, "the admitted publication must have written the durable source blob ref"


def test_unsplit_stage_is_bracketed_by_the_engine(tmp_path: Path) -> None:
    """A ``whole_execute`` stage still gets the writer -- that is the named residual.

    Anti-vacuity: drop the ``whole_execute`` branch from ``_run_stage_execute``
    and this stage's write section would run with no admission at all, leaving
    ``admitted_actors`` empty.
    """
    from polylogue.daemon.convergence import ConvergenceStage

    writer = _SerializingWriter()
    held_during_execute: list[bool] = []

    def execute(_path: Path) -> bool:
        held_during_execute.append(not writer.free_right_now())
        return True

    stage = ConvergenceStage(
        name="unsplit",
        description="a stage that has not split compute from publication",
        check=lambda _path: True,
        execute=execute,
    )
    assert stage.writer_admission == "whole_execute"

    converger = DaemonConverger(stages=(stage,))
    with stage_write_admission(writer.admission):
        converger.converge_batch((tmp_path / "subject.jsonl",))

    assert held_during_execute == [True]
    assert writer.admitted_actors == ["convergence.stage.unsplit"]
