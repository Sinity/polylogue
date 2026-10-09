"""Ordinary Source preparation is pure, ordered, and physically settled."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, NoReturn, TypedDict, TypeVar, Unpack

import pytest

from polylogue.core.compute import AdmissionClass, BoundedComputeAdapter, CancellationHandle, SubmittedOperation
from polylogue.core.enums import Provider
from polylogue.core.sql_settlement import (
    NativeSQLSettlementEvidence,
    SQLCustodyOwner,
    SQLSettlementRetry,
    settle_native_sql,
)
from polylogue.sources.acquisition_boundary import bound_source_observation, open_bound_path
from polylogue.sources.live import production_baseline as module
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.source_layout import export_drop_layout
from polylogue.sources.source_staging import SourceInputBinding
from polylogue.storage.blob_store import BlobStore

T = TypeVar("T")


class _PreparationOptions(TypedDict, total=False):
    cancelled: Callable[[], bool] | None
    source_binding: SourceInputBinding | None
    expected_observation: tuple[int, int, int, int, int] | None


class _SubmissionOptions(TypedDict, total=False):
    admission_class: AdmissionClass
    units: int
    estimated_bytes: int
    exclusive_bytes: bool
    cancellation: CancellationHandle | None


pytestmark = pytest.mark.uses_real_clock("compute preparation uses physical worker synchronization")


def test_preparation_workers_preserve_seal_when_completion_reorders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    payload = (
        b'{"type":"session_meta","payload":{"id":"neutral"}}\n'
        b'{"type":"response_item","payload":{"type":"message","role":"user",'
        b'"content":[{"type":"input_text","text":"neutral"}]}}\n'
    )
    for name in ("a.jsonl", "b.jsonl"):
        (root / name).write_bytes(payload)
    sources = (WatchSource("codex", root, layout=export_drop_layout((".jsonl",))),)
    creator = threading.get_ident()
    adapters = [BoundedComputeAdapter(max_workers=count) for count in (1, 2)]
    real_prepare = module._prepare_file_decision
    real_seal = module._seal
    completed: list[str] = []
    second_finished = threading.Event()

    def seal(
        operation_id: str, source_signature: str, decisions: tuple[module.SourceDecision, ...]
    ) -> module.ProductionSourceBaseline:
        assert threading.get_ident() == creator
        return real_seal(operation_id, source_signature, decisions)

    def progress(_phase: str, **_counts: int) -> None:
        assert threading.get_ident() == creator

    def forbid_publication(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("ordinary preparation cannot publish blobs")

    monkeypatch.setattr(module, "_seal", seal)
    monkeypatch.setattr(BlobStore, "publish_prepared", forbid_publication)
    try:
        monkeypatch.setattr(module, "compute_adapter", lambda: adapters[0])
        single = module.capture_production_source_baseline(sources, operation_id="neutral", progress=progress)

        def prepared(
            observation: tuple[str, Path, str, str], **kwargs: Unpack[_PreparationOptions]
        ) -> module.SourceDecision:
            assert threading.get_ident() != creator
            name = observation[1].name
            if name == "a.jsonl":
                assert second_finished.wait(5)
            result = real_prepare(observation, **kwargs)
            completed.append(name)
            if name == "b.jsonl":
                second_finished.set()
            return result

        monkeypatch.setattr(module, "compute_adapter", lambda: adapters[1])
        monkeypatch.setattr(module, "_prepare_file_decision", prepared)
        parallel = module.capture_production_source_baseline(sources, operation_id="neutral", progress=progress)
        assert completed == ["b.jsonl", "a.jsonl"]
        assert parallel == single
        assert len(parallel.accepted) == 2
        assert adapters[1].snapshot().used_units == 0
    finally:
        second_finished.set()
        for adapter in adapters:
            adapter.shutdown(wait=True)


@pytest.mark.parametrize("replace_inode", [False, True])
def test_parallel_revision_refuses_mutation_at_equal_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_inode: bool
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    path = root / "one.json"
    path.write_bytes(b"{}")
    before = path.stat()
    real_open = open_bound_path
    adapter = BoundedComputeAdapter(max_workers=2)

    @contextmanager
    def opened(current: Path | str, location: Provider | str | None) -> Iterator[BinaryIO]:
        with real_open(current, location) as stream:
            if replace_inode:
                replacement = root / "replacement"
                replacement.write_bytes(b"{}")
                os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
                replacement.replace(path)
            else:
                path.write_bytes(b"[]")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            yield stream

    monkeypatch.setattr(module, "open_bound_path", opened)
    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)
    try:
        result = module.capture_production_source_baseline(
            (WatchSource("account", root, layout=export_drop_layout((".json",))),), operation_id="neutral"
        )
        assert not result.accepted
        faults = [row for row in result.decisions if row.disposition == "fault"]
        assert len(faults) == 1
        assert faults[0].reason.startswith("revision_io_unavailable:")
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)


def test_source_cancellation_drains_reading_worker_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    (root / "one.json").write_bytes(b"{}")
    adapter = BoundedComputeAdapter(max_workers=2)
    entered = threading.Event()
    stopped = threading.Event()
    cancelled = threading.Event()
    outcome: Future[object] = Future()

    def prepared(
        _observation: tuple[str, Path, str, str],
        *,
        cancelled: Callable[[], bool] | None,
        source_binding: SourceInputBinding | None = None,
        expected_observation: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        entered.set()
        try:
            assert stopped.wait(5)
            module._check_observation_cancelled(cancelled)
        finally:
            stopped.set()

    def checkpoint() -> bool:
        # The creator wait must notice cancellation before the worker is free.
        if cancelled.is_set():
            stopped.set()
            return True
        return False

    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(module, "_prepare_file_decision", prepared)

    def observe() -> None:
        try:
            outcome.set_result(
                module.capture_production_source_baseline(
                    (WatchSource("account", root, layout=export_drop_layout((".json",))),),
                    operation_id="neutral",
                    cancelled=checkpoint,
                )
            )
        except BaseException as exc:
            outcome.set_exception(exc)

    creator = threading.Thread(target=observe)
    creator.start()
    try:
        assert entered.wait(5)
        cancelled.set()
        with pytest.raises(module.ProductionBaselineObservationCancelledError):
            outcome.result(5)
        assert stopped.is_set()
        assert adapter.snapshot().used_units == 0
    finally:
        stopped.set()
        creator.join(5)
        adapter.shutdown(wait=True)


def test_source_cancellation_removes_queued_preparation_without_touching_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    (root / "one.json").write_bytes(b"{}")
    adapter = BoundedComputeAdapter(max_workers=1, queue_units=4)
    busy = threading.Event()
    release = threading.Event()
    queued = threading.Event()
    cancelled = threading.Event()
    worker_invoked = threading.Event()
    outcome: Future[object] = Future()
    real_submit = adapter.submit

    def unrelated() -> str:
        busy.set()
        assert release.wait(5)
        return "unrelated"

    def submit(function: Callable[[], T], **kwargs: Unpack[_SubmissionOptions]) -> SubmittedOperation[T]:
        result = real_submit(function, **kwargs)
        if kwargs.get("admission_class") == "bulk-candidate":
            queued.set()
        return result

    blocker = real_submit(unrelated, admission_class="bulk-candidate")
    assert busy.wait(5)
    monkeypatch.setattr(adapter, "submit", submit)
    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(module, "_prepare_file_decision", lambda *_args, **_kwargs: worker_invoked.set())

    def observe() -> None:
        try:
            outcome.set_result(
                module.capture_production_source_baseline(
                    (WatchSource("account", root, layout=export_drop_layout((".json",))),),
                    operation_id="neutral",
                    cancelled=cancelled.is_set,
                )
            )
        except BaseException as exc:
            outcome.set_exception(exc)

    creator = threading.Thread(target=observe)
    creator.start()
    try:
        assert queued.wait(5)
        cancelled.set()
        with pytest.raises(module.ProductionBaselineObservationCancelledError):
            outcome.result(5)
        assert not worker_invoked.is_set()
        assert not blocker.future.done()
        assert adapter.snapshot().used_units == 1
        release.set()
        assert blocker.future.result(5) == "unrelated"
    finally:
        release.set()
        creator.join(5)
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("cancelled_during_cleanup", [False, True])
def test_source_preparation_retains_native_cleanup_and_scratch_until_creator_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled_during_cleanup: bool
) -> None:
    from tempfile import TemporaryDirectory

    from polylogue.core import compute as compute_module
    from polylogue.core.sql_settlement import retain_native_sql_lifetimes
    from polylogue.storage.io_phase_metrics import connect_measured
    from polylogue.storage.sqlite.connection_profile import NativeConnectionSettlementError, NativeSQLCustodyOwner

    root = tmp_path / "neutral"
    root.mkdir()
    (root / "one.json").write_bytes(b"{}")
    adapter = BoundedComputeAdapter(max_workers=2)
    retained = threading.Event()
    allow_close = threading.Event()
    cancelled = threading.Event()
    cancellation_seen = threading.Event()
    outcome: Future[object] = Future()
    scratch_paths: list[Path] = []
    close_threads: list[int] = []
    worker_threads: list[int] = []
    real_prepare = module._prepare_file_decision
    real_settle = settle_native_sql

    def settling(
        *,
        retry: SQLSettlementRetry,
        on_pending: Callable[[NativeSQLSettlementEvidence], None],
        on_settled: Callable[[], None],
        preserved_native_owners: tuple[SQLCustodyOwner, ...] = (),
        initial_observed_generation: int | None = None,
    ) -> BaseException | None:
        def pending(evidence: NativeSQLSettlementEvidence) -> None:
            on_pending(evidence)
            retained.set()

        return real_settle(
            retry=retry,
            on_pending=pending,
            on_settled=on_settled,
            preserved_native_owners=preserved_native_owners,
            initial_observed_generation=initial_observed_generation,
        )

    def prepared(
        observation: tuple[str, Path, str, str], **kwargs: Unpack[_PreparationOptions]
    ) -> module.SourceDecision:
        result = real_prepare(observation, **kwargs)
        scratch = TemporaryDirectory(dir=tmp_path, prefix="owned-source-")
        scratch_paths.append(Path(scratch.name))
        worker_threads.append(threading.get_ident())
        with retain_native_sql_lifetimes(scratch):
            connection = connect_measured(":memory:")
            actual_close = type(connection).close

            def close(current: sqlite3.Connection) -> None:
                if current is connection:
                    close_threads.append(threading.get_ident())
                    if not allow_close.is_set():
                        raise OSError("synthetic native close refusal")
                actual_close(current)

            monkeypatch.setattr(type(connection), "close", close)
            NativeSQLCustodyOwner(connection, scratch_directory=scratch)
        return result

    monkeypatch.setattr(compute_module, "settle_native_sql", settling)
    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(module, "_prepare_file_decision", prepared)

    def checkpoint() -> bool:
        if cancelled.is_set():
            cancellation_seen.set()
            return True
        return False

    def observe() -> None:
        try:
            outcome.set_result(
                module.capture_production_source_baseline(
                    (WatchSource("account", root, layout=export_drop_layout((".json",))),),
                    operation_id="neutral",
                    cancelled=checkpoint,
                )
            )
        except BaseException as exc:
            outcome.set_exception(exc)

    creator = threading.Thread(target=observe)
    creator.start()
    try:
        assert retained.wait(5)
        assert not outcome.done()
        assert scratch_paths[0].is_dir()
        assert adapter.snapshot().used_units == 1
        assert adapter.retained_sql_settlements()[0].owner_count == 1
        if cancelled_during_cleanup:
            cancelled.set()
            assert cancellation_seen.wait(5)
            adapter.shutdown(wait=False)
            # Cancellation cannot release the physical owner or its scratch.
            assert not outcome.done()
            assert scratch_paths[0].is_dir()
        allow_close.set()
        adapter.retry_sql_settlement()
        with pytest.raises(
            module.ProductionBaselineObservationCancelledError
            if cancelled_during_cleanup
            else NativeConnectionSettlementError
        ):
            outcome.result(5)
        assert len(close_threads) >= 2
        assert close_threads == worker_threads * len(close_threads)
        assert not scratch_paths[0].exists()
        assert adapter.snapshot().used_units == 0
        assert adapter.retained_sql_settlements() == ()
    finally:
        allow_close.set()
        adapter.retry_sql_settlement()
        creator.join(5)
        adapter.shutdown(wait=True)


@pytest.mark.parametrize("replace_inode", [False, True])
def test_preparation_charge_refuses_growth_before_worker_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_inode: bool
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    path = root / "one.json"
    path.write_bytes(b"{}")
    before = path.stat()
    adapter = BoundedComputeAdapter(max_workers=2, queue_bytes=100)
    real_submit = adapter.submit
    declared_bytes: list[int] = []

    def submit(function: Callable[[], T], **kwargs: Unpack[_SubmissionOptions]) -> SubmittedOperation[T]:
        declared_bytes.append(kwargs["estimated_bytes"])
        if replace_inode:
            replacement = root / "replacement"
            replacement.write_bytes(b"{}" + b" " * 200)
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            replacement.replace(path)
        else:
            path.write_bytes(b"{}" + b" " * 200)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return real_submit(function, **kwargs)

    def refuse_read(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("queued currency mismatch must be refused before payload inspection")

    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(adapter, "submit", submit)
    monkeypatch.setattr(module, "classify_pre_acquisition", refuse_read)
    try:
        result = module.capture_production_source_baseline(
            (WatchSource("account", root, layout=export_drop_layout((".json",))),), operation_id="neutral"
        )
        assert declared_bytes == [2]
        assert not result.accepted
        [fault] = [row for row in result.decisions if row.disposition == "fault"]
        assert fault.reason.startswith("revision_io_unavailable:")
        assert adapter.snapshot().used_units == adapter.snapshot().used_bytes == 0
    finally:
        adapter.shutdown(wait=True)


def test_failed_first_input_cancels_another_workers_hash_before_drain_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from polylogue.core.compute import current_cancellation

    root = tmp_path / "neutral"
    root.mkdir()
    for name in ("a.json", "b.json"):
        (root / name).write_bytes(b"{}" + b" " * (2 * 1024 * 1024))
    adapter = BoundedComputeAdapter(max_workers=2)
    reading = threading.Event()
    resume = threading.Event()
    closed = threading.Event()
    real_prepare = module._prepare_file_decision
    real_open = open_bound_path
    real_observation = bound_source_observation
    read_counts: list[int] = []

    class PausedRead:
        def __init__(self, stream: BinaryIO) -> None:
            self.stream = stream

        def read(self, size: int) -> bytes:
            data = self.stream.read(size)
            read_counts.append(len(data))
            handle = current_cancellation()
            assert handle is not None
            handle.add_listener(resume.set)
            reading.set()
            assert resume.wait(5)
            return data

    @contextmanager
    def opened(path: Path | str, location: Provider | str | None) -> Iterator[PausedRead]:
        with real_open(path, location) as stream:
            try:
                yield PausedRead(stream)
            finally:
                closed.set()

    def prepared(
        observation: tuple[str, Path, str, str], **kwargs: Unpack[_PreparationOptions]
    ) -> module.SourceDecision:
        if observation[1].name == "a.json":
            assert reading.wait(5)
            raise RuntimeError("synthetic first-input failure")
        return real_prepare(observation, **kwargs)

    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)
    monkeypatch.setattr(module, "_prepare_file_decision", prepared)
    monkeypatch.setattr(module, "open_bound_path", opened)
    monkeypatch.setattr(module, "bound_source_observation", lambda stream: real_observation(stream.stream))
    try:
        with pytest.raises(RuntimeError, match="synthetic first-input failure"):
            module.capture_production_source_baseline(
                (WatchSource("account", root, layout=export_drop_layout((".json",))),), operation_id="neutral"
            )
        assert read_counts == [1024 * 1024]
        assert closed.is_set()
        assert adapter.snapshot().used_units == adapter.snapshot().used_bytes == 0
    finally:
        resume.set()
        adapter.shutdown(wait=True)


def test_cancellation_after_final_progress_refuses_source_sealing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    (root / "one.json").write_bytes(b"{}")
    adapter = BoundedComputeAdapter(max_workers=2)
    cancelled = threading.Event()
    monkeypatch.setattr(module, "compute_adapter", lambda: adapter)

    def progress(_phase: str, **counts: int) -> None:
        if counts.get("revisions"):
            cancelled.set()

    try:
        with pytest.raises(module.ProductionBaselineObservationCancelledError):
            module.capture_production_source_baseline(
                (WatchSource("account", root, layout=export_drop_layout((".json",))),),
                operation_id="neutral",
                cancelled=cancelled.is_set,
                progress=progress,
            )
        assert adapter.snapshot().used_units == 0
    finally:
        adapter.shutdown(wait=True)


def test_preparation_refuses_a_new_fifo_before_payload_inspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "neutral"
    root.mkdir()
    path = root / "one.json"
    path.write_bytes(b"{}")
    adapter = BoundedComputeAdapter(max_workers=2)

    def owner() -> BoundedComputeAdapter:
        path.unlink()
        os.mkfifo(path)
        return adapter

    def refuse_read(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("changed nonregular input must not enter a blocking parser open")

    monkeypatch.setattr(module, "compute_adapter", owner)
    monkeypatch.setattr(module, "classify_pre_acquisition", refuse_read)
    try:
        result = module.capture_production_source_baseline(
            (WatchSource("account", root, layout=export_drop_layout((".json",))),), operation_id="neutral"
        )
        assert not result.accepted
        [fault] = [row for row in result.decisions if row.disposition == "fault"]
        assert fault.reason.startswith("revision_io_unavailable:")
    finally:
        adapter.shutdown(wait=True)
