"""Capture declared mutable SQLite inputs before the writer is admitted."""

from __future__ import annotations

import os
import threading
import time
from builtins import BaseExceptionGroup
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.compute import BoundedComputeAdapter, DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled, compute_cancel
from polylogue.core.enums import Provider
from polylogue.core.prepared_file import VerificationCancelledError
from polylogue.core.sql_settlement import retain_native_sql_lifetimes
from polylogue.sources.live.batch_support import PreAcquisitionDecision
from polylogue.sources.sqlite_snapshot import SQLiteBlobSnapshot
from polylogue.storage.blob_publication import ArchiveBlobPublisher


@dataclass(frozen=True, slots=True)
class PreparedLiveSQLiteCapture:
    snapshot: SQLiteBlobSnapshot | None
    publisher: ArchiveBlobPublisher
    admission: PreAcquisitionDecision
    source_stat: os.stat_result
    observed_at_ns: int

    def discard(self) -> None:
        self.publisher.discard_pending()


class LiveSQLiteCaptureStage:
    """Borrow one supplied compute owner and settle every admitted capture."""

    def __init__(self, *, compute_adapter: BoundedComputeAdapter) -> None:
        self._executor = compute_adapter
        self._stage_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None
        self._closing = False

    def prepare_sqlite_paths(
        self,
        paths: Sequence[Path],
        *,
        archive_root: Path,
        cancelled: threading.Event,
        fallback_provider: Provider,
    ) -> dict[Path, PreparedLiveSQLiteCapture | Exception]:
        """Acquire and seal declared Codex state before writer admission."""
        from polylogue.sources.live.batch_support import classify_pre_acquisition
        from polylogue.sources.source_staging import bind_source_input
        from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
        from polylogue.storage.blob_publication import ArchiveBlobPublisher

        captures: dict[Path, PreparedLiveSQLiteCapture | Exception] = {}
        with self._stage_lock:
            with self._publish_lock:
                if self._closing:
                    raise DaemonOperationCancelled("state preparation is closing")
                self._active_cancel = cancelled

            def prepare(path: Path) -> tuple[Path, PreparedLiveSQLiteCapture | Exception]:
                with retain_native_sql_lifetimes():
                    token = compute_cancel.set(cancelled)
                    publisher = ArchiveBlobPublisher(archive_root / "source.db", archive_root / "blob")
                    try:
                        check_compute_cancelled()
                        with bind_source_input(path) as binding:
                            observed_at_ns = time.time_ns()
                            source_stat = os.stat(
                                binding.physical_path.name, dir_fd=binding.parent_anchor, follow_symlinks=False
                            )
                            admission = classify_pre_acquisition(
                                binding.source_path,
                                fallback_provider=fallback_provider,
                                source_only=True,
                                size_bytes=source_stat.st_size,
                                source_binding=binding,
                            )
                            if admission.excluded_reason is not None:
                                return path, PreparedLiveSQLiteCapture(
                                    None, publisher, admission, source_stat, observed_at_ns
                                )
                            snapshot = snapshot_sqlite_to_blob(path, publisher, source_binding=binding)
                        check_compute_cancelled()
                        return path, PreparedLiveSQLiteCapture(
                            snapshot, publisher, admission, source_stat, observed_at_ns
                        )
                    except (DaemonOperationCancelled, VerificationCancelledError) as cancelled_failure:
                        try:
                            publisher.discard_pending()
                        except BaseException as cleanup:
                            raise BaseExceptionGroup(
                                "state cancellation and capture cleanup failed", [cancelled_failure, cleanup]
                            ) from cancelled_failure
                        if isinstance(cancelled_failure, DaemonOperationCancelled):
                            raise
                        raise DaemonOperationCancelled("state preparation cancelled") from cancelled_failure
                    except Exception as failure:
                        try:
                            publisher.discard_pending()
                        except BaseException as cleanup:
                            raise BaseExceptionGroup(
                                "state preparation and capture cleanup failed", [failure, cleanup]
                            ) from failure
                        return path, failure
                    finally:
                        compute_cancel.reset(token)

            try:
                with closing(
                    self._executor.map(
                        prepare,
                        paths,
                        # Logical export may hydrate one SQLite value; its working set is unknown.
                        estimated_bytes=lambda path: path.stat().st_size,
                        exclusive_bytes=True,
                        discard_unconsumed=lambda item: (
                            item[1].discard() if isinstance(item[1], PreparedLiveSQLiteCapture) else None
                        ),
                    )
                ) as prepared:
                    for path, capture in prepared:
                        captures[path] = capture
                return captures
            except BaseException as primary:
                failures: list[BaseException] = [primary]
                for capture in captures.values():
                    if isinstance(capture, PreparedLiveSQLiteCapture):
                        try:
                            capture.discard()
                        except BaseException as cleanup:
                            failures.append(cleanup)
                if len(failures) > 1:
                    raise BaseExceptionGroup("state capture and cleanup failed", failures) from primary
                raise
            finally:
                with self._publish_lock:
                    self._active_cancel = None

    def shutdown(self) -> None:
        with self._publish_lock:
            self._closing = True
            active = self._active_cancel
        if active is not None:
            active.set()
        # map drains each original physical Future before releasing this lock.
        # This stage borrows the kernel; it never shuts down a shared owner.
        with self._stage_lock:
            pass
