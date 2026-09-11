"""Daemon state convergence — check and converge to desired archive state.

The daemon owns all writes. For each source file, the desired state is:
  1. Raw blob stored (content-addressed)
  2. Parsed into records (provider detection + record extraction)
  3. Messages materialized (normalized into messages table)
  4. FTS indexed (searchable)
  5. Derived tables refreshed (session profiles, work events, etc.)

Convergence means checking the current state for each file and doing
only the missing work. Cursor records track the last-known file state
so we skip unchanged files entirely.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

from polylogue.daemon.derivation import (
    Budget,
    DerivationAdapter,
    DerivationFrame,
    DerivationRegistry,
    DerivationReport,
    PassCursor,
    converge,
)
from polylogue.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from polylogue.daemon.execution import BoundedComputeAdapter
    from polylogue.daemon.write_coordinator import DaemonWriteThreadBridge


class _DerivationAdmission:
    """Bridge one short publish from a compute worker to the daemon writer."""

    def __init__(self, bridge: DaemonWriteThreadBridge, *, loop_thread_id: int) -> None:
        self._bridge = bridge
        self._loop_thread_id = loop_thread_id

    def __call__(self, domain: str, publish: Callable[[], bool]) -> bool:
        if threading.get_ident() == self._loop_thread_id:
            raise RuntimeError("session derivation publish was invoked on the daemon event loop thread")
        # The bridge owns the coordinator until the transaction really returns;
        # a caller-side timeout must not admit a second archive writer.
        return cast("bool", self._bridge.run_sync_with_timeout(f"derivation.{domain}", None, publish))


class SessionProfileConvergenceOwner:
    """Run the registered session domain on daemon-shared lease-free compute.

    Composition supplies the already-constructed converger and its one
    session-profile adapter.  This owner deliberately does not construct a
    pool or a writer: it borrows the process adapter and bridges only each
    short publication back to the daemon's coordinator.
    """

    def __init__(
        self,
        converger: DaemonConverger,
        *,
        compute_adapter: BoundedComputeAdapter,
        write_bridge: DaemonWriteThreadBridge,
    ) -> None:
        self._converger = converger
        self._compute_adapter = compute_adapter
        self._write_bridge = write_bridge
        # The kernel retains a process-local pagination cursor.  One owner may
        # receive a periodic no-hint sweep and a watcher-targeted pass at once;
        # serialize their scheduling without extending any writer hold.
        self._converge_lock = asyncio.Lock()

    async def converge(
        self,
        frame: DerivationFrame,
        *,
        budget: Budget | int | None = None,
        deadline_s: float | None = None,
        resume: bool = True,
    ) -> DerivationReport:
        async with self._converge_lock:
            return await self._converge_serialized(
                frame,
                budget=budget,
                deadline_s=deadline_s,
                resume=resume,
            )

    async def _converge_serialized(
        self,
        frame: DerivationFrame,
        *,
        budget: Budget | int | None = None,
        deadline_s: float | None = None,
        resume: bool = True,
    ) -> DerivationReport:
        from polylogue.daemon.write_coordinator import daemon_write_lease_active

        if daemon_write_lease_active():
            raise RuntimeError("session profile convergence must start after the daemon writer lease is released")
        loop = asyncio.get_running_loop()
        admission = _DerivationAdmission(self._write_bridge, loop_thread_id=threading.get_ident())
        # A targeted ingest scope is not a continuation of archive keyset
        # paging: reusing the archive cursor could skip an earlier changed id.
        # Only no-hint archive sweeps retain their own cursor across passes.
        pass_resume = resume if frame.scope is None else False
        submitted = self._compute_adapter.submit(
            partial(
                self._converger.converge_derivations,
                frame,
                budget=budget,
                deadline_s=deadline_s,
                domains=("session_profile",),
                resume=pass_resume,
                publisher=admission,
            ),
            admission_class="incremental-background",
        )
        operation = asyncio.wrap_future(submitted.future, loop=loop)  # type: ignore[arg-type]
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError:
            # A caller may stop awaiting this sweep, but cannot let a compute
            # worker that already owns a bridged publication outlive owner
            # shutdown.  Settle it before propagating cancellation so the
            # composition layer can drain the coordinator safely.
            with contextlib.suppress(BaseException):
                await asyncio.shield(operation)
            raise


def make_session_profile_derivation(
    index_db_path: Path,
    *,
    archive_root: Path,
    materializer_version: int,
    now: Callable[[], float],
) -> DerivationAdapter:
    """Build the one daemon-owned adapter for one active index generation.

    The protocol composition layer calls this once for its active generation
    and registers the returned adapter exactly once with ``DaemonConverger``.
    The frame's scope is either the bounded durable changes from live ingest or
    ``None`` for an archive-wide no-hint sweep.
    """
    from polylogue.storage.archive_identity import resolve_active_index_path
    from polylogue.storage.derived.session.derivation import SessionProfileDerivation
    from polylogue.storage.sqlite.connection_profile import open_daemon_connection, open_readonly_connection

    def active_index_path() -> Path:
        # ``root/index.db`` can be a pointer stub.  Consult the configured
        # root's canonical active-index authority for every compute/write
        # boundary so a promoted generation is never treated as the old path.
        return resolve_active_index_path(archive_root)

    def read_connection() -> sqlite3.Connection:
        return open_readonly_connection(active_index_path(), timeout_class="background-read")

    def write_connection() -> sqlite3.Connection:
        return open_daemon_connection(active_index_path(), archive_root=archive_root)

    def generation_binding() -> str:
        return str(active_index_path())

    def quiet_key(frame: object, session_id: str) -> bool:
        # Check one key at a time inside the compute pass.  This preserves hot
        # source deferral without materializing an archive-wide quiet set for a
        # no-hint sweep, and keeps clock authority injected by composition.
        from polylogue.daemon.convergence_stages import _archive_hot_insight_session_ids

        conn = read_connection()
        try:
            return session_id in _archive_hot_insight_session_ids(
                conn,
                (session_id,),
                now=now(),
                archive_root=archive_root,
            )
        finally:
            conn.close()

    user_db = archive_root / "user.db"

    def marker_read_connection() -> sqlite3.Connection:
        return open_readonly_connection(user_db, timeout_class="background-read")

    def marker_write_connection() -> sqlite3.Connection:
        return open_daemon_connection(user_db, archive_root=archive_root)

    def scope(frame: object) -> Sequence[str] | None:
        value = getattr(frame, "scope", None)
        if value is None:
            return None
        if not isinstance(value, tuple):
            raise TypeError("session profile frame scope must be a tuple of session ids or None")
        return tuple(str(item) for item in value)

    return cast(
        "DerivationAdapter",
        SessionProfileDerivation(
            read_connection,
            write_connection,
            materializer_version=materializer_version,
            session_scope=scope,
            quiet_key=quiet_key,
            marker_read_connection=marker_read_connection if user_db.exists() else None,
            marker_write_connection=marker_write_connection if user_db.exists() else None,
            generation_binding=generation_binding,
        ),
    )


def make_session_profile_frame(
    index_db_path: Path,
    *,
    archive_root: Path,
    scope: Sequence[str] | None,
) -> DerivationFrame:
    """Describe one bounded session pass at the current index generation.

    The physical generation is observed from the active anchor, not invented
    from raw-ingest hints.  The adapter opens its own one-connection read
    transaction against that generation and binds each computed key to the
    exact values it consumed; publication rejects if the anchor promotes in
    between.  ``scope=None`` is the restart-safe archive sweep.
    """
    from polylogue.storage.archive_identity import resolve_active_index_path

    del index_db_path
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=f"index-generation:{resolve_active_index_path(archive_root)}",
        recipe_versions={"session_profile": "session-profile"},
        scope=None if scope is None else tuple(dict.fromkeys(str(session_id) for session_id in scope)),
    )


def _stage_false_error(stage_name: str, *, scope: str) -> str:
    if scope == "stage":
        return f"stage {stage_name} returned False"
    return f"{scope} stage {stage_name} returned False"


class StageState(Enum):
    PENDING = "pending"  # work needed
    IN_PROGRESS = "in_progress"  # work running
    DONE = "done"  # converged
    SKIPPED = "skipped"  # not applicable
    FAILED = "failed"  # error, will retry


@dataclass(frozen=True, slots=True)
class StageExecutionResult:
    """Optional rich result for convergence stages that report sub-timings."""

    success: bool
    stage_timings_s: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


StageExecuteReturn = bool | StageExecutionResult


@dataclass(frozen=True, slots=True)
class ConvergenceStage:
    """A named pipeline stage with a check function and an execute function."""

    name: str
    description: str
    # Check returns True if this stage needs work for the given file.
    check: Callable[[Path], bool]
    # Execute performs the work. Returns True on success.
    execute: Callable[[Path], StageExecuteReturn]
    # Optional batch check/execute pair for stages that can collapse many
    # changed source paths into one repair transaction.
    check_many: Callable[[Sequence[Path]], set[Path]] | None = None
    execute_many: Callable[[Sequence[Path]], StageExecuteReturn] | None = None
    # Optional session-scoped pair for durable convergence debt retries.
    # These avoid resolving a failed derived subject back to source files.
    check_sessions: Callable[[Sequence[str]], set[str]] | None = None
    execute_sessions: Callable[[Sequence[str]], StageExecuteReturn] | None = None
    # Some stages intentionally return False after doing bounded successful
    # work so the remaining backlog is retried as convergence debt.
    false_means_pending: bool = False
    # A primary-authority stage may block every later projection stage for the
    # affected subjects until its durable receipt predicate is satisfied.
    blocks_following_stages: bool = False
    barrier_check: Callable[[Path], bool] | None = None
    barrier_check_many: Callable[[Sequence[Path]], set[Path]] | None = None
    barrier_check_sessions: Callable[[Sequence[str]], set[str]] | None = None
    # Optional bounded, secret-safe operator status payload.
    status: Callable[[], Mapping[str, object]] | None = None
    # The stage's work is a function of the whole archive, not of the batch's
    # subjects (a graph rebuilt from every raw artifact, an exact archive-wide
    # readiness audit). ``converge_batch(whole_archive=False)`` skips such
    # stages so a catch-up chunk's cost stays bounded by its own input; the
    # catch-up's final chunk runs them once for the whole backlog.
    whole_archive: bool = False


@dataclass(slots=True)
class FileState:
    """Tracked convergence state for a single source file."""

    path: Path
    stages: dict[str, StageState] = field(default_factory=dict)
    stage_times: dict[str, float] = field(default_factory=dict)
    last_stage_times: dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    last_error: str | None = None

    @property
    def converged(self) -> bool:
        return all(s in (StageState.DONE, StageState.SKIPPED) for s in self.stages.values())

    @property
    def pending_stages(self) -> list[str]:
        return [name for name, s in self.stages.items() if s == StageState.PENDING]


@dataclass(slots=True)
class SessionState:
    """Tracked convergence state for one session subject."""

    session_id: str
    stages: dict[str, StageState] = field(default_factory=dict)
    stage_times: dict[str, float] = field(default_factory=dict)
    last_stage_times: dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    last_error: str | None = None

    @property
    def converged(self) -> bool:
        return all(s in (StageState.DONE, StageState.SKIPPED) for s in self.stages.values())


def _record_execute_result(
    state: FileState | SessionState,
    *,
    stage_name: str,
    stage: ConvergenceStage,
    success: bool,
    scope: str,
) -> None:
    if success:
        state.stages[stage_name] = StageState.DONE
        return
    state.last_error = _stage_false_error(stage_name, scope=scope)
    if stage.false_means_pending:
        state.stages[stage_name] = StageState.PENDING
        return
    state.stages[stage_name] = StageState.FAILED
    state.error_count += 1


def _coerce_execute_result(result: StageExecuteReturn) -> tuple[bool, dict[str, float]]:
    if isinstance(result, StageExecutionResult):
        return result.success, dict(result.stage_timings_s)
    return bool(result), {}


def _record_stage_times(
    batch_stage_times: dict[str, float],
    stage_name: str,
    elapsed: float,
    extra_stage_timings_s: dict[str, float],
) -> None:
    batch_stage_times[stage_name] = batch_stage_times.get(stage_name, 0.0) + elapsed
    for name, value in extra_stage_timings_s.items():
        batch_stage_times[name] = batch_stage_times.get(name, 0.0) + float(value)


class DaemonConverger:
    """Drives archive state toward desired state for all source files.

    Runs a set of :class:`ConvergenceStage` checks against each file.
    The main process is the only SQLite writer.
    """

    def __init__(
        self,
        stages: Iterable[ConvergenceStage],
        *,
        derivations: Iterable[object] = (),
    ) -> None:
        self._stages: dict[str, ConvergenceStage] = {s.name: s for s in stages}
        self._file_states: dict[Path, FileState] = {}
        self._session_states: dict[str, SessionState] = {}
        self._derivations = DerivationRegistry(cast("Iterable[DerivationAdapter]", derivations))
        self._derivation_cursor = PassCursor()

    @property
    def derivation_domains(self) -> tuple[str, ...]:
        return tuple(adapter.domain for adapter in self._derivations.ordered())

    def converge_derivations(
        self,
        frame: DerivationFrame,
        *,
        budget: Budget | int | None = None,
        deadline_s: float | None = None,
        domains: Sequence[str] | None = None,
        resume: bool = True,
        publisher: Callable[[str, Callable[[], bool]], bool] | None = None,
    ) -> DerivationReport:
        """Converge the migrated domains from their own output relations.

        The pending set is ``required`` minus ``valid``, recomputed here rather
        than read from stage state, so this pass is identical after a restart
        that lost every scheduling hint. Adapters own their write acquisition:
        this facade must not wrap compute or publication in an outer lease, or
        one long batch would hold the writer across every domain's computation.

        The one thing carried between calls is a process-local
        :class:`PassCursor`: where the last bounded pass stopped looking. It is
        deliberately not durable and cannot certify a key -- a restart drops it
        and converges the same set, one sweep later. Without it a bounded pass
        would re-examine the same prefix forever, so a permanently quiet head
        would starve the tail; that is scheduling fairness, not authority.
        """
        report = converge(
            self._derivations,
            frame,
            budget=budget,
            deadline_s=deadline_s,
            domains=domains,
            cursor=self._derivation_cursor if resume else None,
            publisher=publisher,
        )
        self._derivation_cursor = report.cursor
        return report

    @property
    def stage_names(self) -> list[str]:
        return list(self._stages)

    def stage_status(self) -> dict[str, dict[str, object]]:
        """Return bounded stage-owned status without propagating secret detail."""
        result: dict[str, dict[str, object]] = {}
        for stage_name, stage in self._stages.items():
            if stage.status is None:
                continue
            try:
                result[stage_name] = dict(stage.status())
            except Exception:
                logger.warning("converger: status probe failed stage=%s", stage_name, exc_info=True)
                result[stage_name] = {"state": "unavailable"}
        return result

    @staticmethod
    def _mark_barrier_failure(
        state: FileState | SessionState,
        *,
        stage_name: str,
    ) -> None:
        state.stages[stage_name] = StageState.FAILED
        state.error_count += 1
        state.last_error = f"stage {stage_name} barrier check failed"

    def _path_barrier_blocked(
        self,
        stage_name: str,
        stage: ConvergenceStage,
        path: Path,
    ) -> bool:
        if not stage.blocks_following_stages:
            return False
        state = self._file_states[path]
        if stage.barrier_check is None:
            return state.stages.get(stage_name) is not StageState.DONE
        try:
            return bool(stage.barrier_check(path))
        except Exception:
            logger.warning(
                "converger: barrier check failed for %s stage=%s",
                path,
                stage_name,
                exc_info=True,
            )
            self._mark_barrier_failure(state, stage_name=stage_name)
            return True

    def _path_barriers_blocked(
        self,
        stage_name: str,
        stage: ConvergenceStage,
        paths: Sequence[Path],
    ) -> set[Path]:
        if not stage.blocks_following_stages or not paths:
            return set()
        if stage.barrier_check_many is None:
            return {path for path in paths if self._path_barrier_blocked(stage_name, stage, path)}
        try:
            blocked = set(stage.barrier_check_many(paths))
        except Exception:
            logger.warning("converger: batch barrier check failed stage=%s", stage_name, exc_info=True)
            for path in paths:
                self._mark_barrier_failure(self._file_states[path], stage_name=stage_name)
            return set(paths)
        return blocked.intersection(paths)

    def _session_barriers_blocked(
        self,
        stage_name: str,
        stage: ConvergenceStage,
        session_ids: Sequence[str],
    ) -> set[str]:
        if not stage.blocks_following_stages or not session_ids:
            return set()
        if stage.barrier_check_sessions is None:
            return {
                session_id
                for session_id in session_ids
                if self._session_states[session_id].stages.get(stage_name) is not StageState.DONE
            }
        try:
            blocked = set(stage.barrier_check_sessions(session_ids))
        except Exception:
            logger.warning("converger: session barrier check failed stage=%s", stage_name, exc_info=True)
            for session_id in session_ids:
                self._mark_barrier_failure(self._session_states[session_id], stage_name=stage_name)
            return set(session_ids)
        return blocked.intersection(session_ids)

    def converge_file(self, path: Path) -> FileState:
        """Converge one file while honoring durable stage barriers."""
        if path not in self._file_states:
            self._file_states[path] = FileState(path=path)
        state = self._file_states[path]
        state.last_stage_times.clear()
        downstream_blocked = False

        for stage_name, stage in self._stages.items():
            if downstream_blocked:
                state.stages[stage_name] = StageState.PENDING
                continue

            current = state.stages.get(stage_name)
            if current is not StageState.DONE:
                try:
                    needs_work = stage.check(path)
                except Exception:
                    logger.warning(
                        "converger: check failed for %s stage=%s",
                        path,
                        stage_name,
                        exc_info=True,
                    )
                    state.stages[stage_name] = StageState.FAILED
                    state.error_count += 1
                else:
                    if not needs_work:
                        state.stages[stage_name] = StageState.DONE
                    else:
                        state.stages[stage_name] = StageState.IN_PROGRESS
                        t_stage = time.perf_counter()
                        try:
                            execute_result = stage.execute(path)
                        except Exception as exc:
                            logger.warning(
                                "converger: execute failed for %s stage=%s: %s",
                                path,
                                stage_name,
                                exc,
                            )
                            state.stages[stage_name] = StageState.FAILED
                            state.error_count += 1
                            state.last_error = str(exc)
                        else:
                            elapsed = time.perf_counter() - t_stage
                            success, extra_stage_timings_s = _coerce_execute_result(execute_result)
                            state.stage_times[stage_name] = elapsed
                            state.last_stage_times[stage_name] = elapsed
                            for name, value in extra_stage_timings_s.items():
                                state.stage_times[name] = value
                                state.last_stage_times[name] = value
                            _record_execute_result(
                                state,
                                stage_name=stage_name,
                                stage=stage,
                                success=success,
                                scope="stage",
                            )

            if self._path_barrier_blocked(stage_name, stage, path):
                downstream_blocked = True

        return state

    def invalidate_file(self, path: Path) -> None:
        """Mark a changed file as needing stage checks again."""
        state = self._file_states.get(path)
        if state is None:
            return
        state.stages.clear()
        state.last_stage_times.clear()

    def _evict_converged_files(self, paths: Iterable[Path]) -> None:
        for path in paths:
            state = self._file_states.get(path)
            if state is not None and state.converged:
                del self._file_states[path]

    def _evict_converged_sessions(self, session_ids: Iterable[str]) -> None:
        for session_id in session_ids:
            state = self._session_states.get(session_id)
            if state is not None and state.converged:
                del self._session_states[session_id]

    def converge_batch(
        self, files: Iterable[Path], *, whole_archive: bool = True
    ) -> tuple[dict[Path, FileState], dict[str, float]]:
        """Converge a changed source batch with per-subject stage barriers.

        ``whole_archive=False`` bounds the pass to the batch's own subjects:
        stages declared ``whole_archive`` are recorded ``SKIPPED`` (converged,
        no debt) because their staleness is re-derived from archive content by
        the next whole-archive pass, never from this batch's outcome.

        Stage ``check``/``check_many`` time is charged to ``<stage>.check`` in
        the returned ledger so the batch's convergence time is fully attributed.
        """
        paths = tuple(dict.fromkeys(files))
        if not paths:
            return {}, {}

        for path in paths:
            if path not in self._file_states:
                self._file_states[path] = FileState(path=path)
            state = self._file_states[path]
            state.stages.clear()
            state.last_stage_times.clear()

        batch_stage_times: dict[str, float] = {}
        blocked_paths: set[Path] = set()
        for stage_name, stage in self._stages.items():
            for path in blocked_paths:
                self._file_states[path].stages[stage_name] = StageState.PENDING
            active_paths = tuple(path for path in paths if path not in blocked_paths)
            if not active_paths:
                continue
            if stage.whole_archive and not whole_archive:
                for path in active_paths:
                    self._file_states[path].stages[stage_name] = StageState.SKIPPED
                continue

            if stage.check_many is None or stage.execute_many is None:
                for path in active_paths:
                    state = self._file_states[path]
                    t_check = time.perf_counter()
                    try:
                        needs_work = stage.check(path)
                    except Exception:
                        logger.warning(
                            "converger: check failed for %s stage=%s",
                            path,
                            stage_name,
                            exc_info=True,
                        )
                        state.stages[stage_name] = StageState.FAILED
                        state.error_count += 1
                        continue
                    finally:
                        _record_stage_times(batch_stage_times, f"{stage_name}.check", time.perf_counter() - t_check, {})

                    if not needs_work:
                        state.stages[stage_name] = StageState.DONE
                        continue

                    state.stages[stage_name] = StageState.IN_PROGRESS
                    t_stage = time.perf_counter()
                    try:
                        execute_result = stage.execute(path)
                    except Exception as exc:
                        logger.warning(
                            "converger: execute failed for %s stage=%s: %s",
                            path,
                            stage_name,
                            exc,
                        )
                        state.stages[stage_name] = StageState.FAILED
                        state.error_count += 1
                        state.last_error = str(exc)
                        continue

                    elapsed = time.perf_counter() - t_stage
                    success, extra_stage_timings_s = _coerce_execute_result(execute_result)
                    _record_stage_times(batch_stage_times, stage_name, elapsed, extra_stage_timings_s)
                    state.stage_times[stage_name] = elapsed
                    state.last_stage_times[stage_name] = elapsed
                    for name, value in extra_stage_timings_s.items():
                        state.stage_times[name] = value
                        state.last_stage_times[name] = value
                    _record_execute_result(
                        state,
                        stage_name=stage_name,
                        stage=stage,
                        success=success,
                        scope="stage",
                    )
            else:
                t_check = time.perf_counter()
                try:
                    batch_needs_work = set(stage.check_many(active_paths)).intersection(active_paths)
                except Exception:
                    logger.warning("converger: batch check failed stage=%s", stage_name, exc_info=True)
                    for path in active_paths:
                        state = self._file_states[path]
                        state.stages[stage_name] = StageState.FAILED
                        state.error_count += 1
                else:
                    _record_stage_times(batch_stage_times, f"{stage_name}.check", time.perf_counter() - t_check, {})
                    for path in active_paths:
                        if path not in batch_needs_work:
                            self._file_states[path].stages[stage_name] = StageState.DONE

                    if batch_needs_work:
                        for path in batch_needs_work:
                            self._file_states[path].stages[stage_name] = StageState.IN_PROGRESS

                        # The batch's own order, not set-iteration order: a set
                        # of paths iterates by string hash, which is randomized
                        # per process, so a stage would see its subjects in a
                        # different order on every run.
                        ordered_needs_work = tuple(path for path in active_paths if path in batch_needs_work)
                        t_stage = time.perf_counter()
                        try:
                            execute_result = stage.execute_many(ordered_needs_work)
                        except Exception as exc:
                            logger.warning("converger: batch execute failed stage=%s: %s", stage_name, exc)
                            for path in batch_needs_work:
                                state = self._file_states[path]
                                state.stages[stage_name] = StageState.FAILED
                                state.error_count += 1
                                state.last_error = str(exc)
                        else:
                            elapsed = time.perf_counter() - t_stage
                            success, extra_stage_timings_s = _coerce_execute_result(execute_result)
                            remaining_needs_work: set[Path] | None = None
                            if not success and stage.false_means_pending:
                                try:
                                    remaining_needs_work = set(stage.check_many(ordered_needs_work)).intersection(
                                        batch_needs_work
                                    )
                                except Exception:
                                    logger.warning(
                                        "converger: batch recheck failed stage=%s",
                                        stage_name,
                                        exc_info=True,
                                    )
                            _record_stage_times(
                                batch_stage_times,
                                stage_name,
                                elapsed,
                                extra_stage_timings_s,
                            )
                            for path in ordered_needs_work:
                                state = self._file_states[path]
                                state.stage_times[stage_name] = elapsed
                                state.last_stage_times[stage_name] = elapsed
                                for name, value in extra_stage_timings_s.items():
                                    state.stage_times[name] = value
                                    state.last_stage_times[name] = value
                                path_success = success
                                if remaining_needs_work is not None:
                                    path_success = path not in remaining_needs_work
                                _record_execute_result(
                                    state,
                                    stage_name=stage_name,
                                    stage=stage,
                                    success=path_success,
                                    scope="batch",
                                )

            blocked_paths.update(self._path_barriers_blocked(stage_name, stage, active_paths))

        results = {path: self._file_states[path] for path in paths}
        self._evict_converged_files(paths)
        return results, batch_stage_times

    def converge_sessions(
        self,
        session_ids: Iterable[str],
    ) -> tuple[dict[str, SessionState], dict[str, float]]:
        """Converge known session subjects while honoring primary barriers."""
        ids = tuple(dict.fromkeys(str(session_id) for session_id in session_ids if session_id))
        if not ids:
            return {}, {}

        for session_id in ids:
            if session_id not in self._session_states:
                self._session_states[session_id] = SessionState(session_id=session_id)
            state = self._session_states[session_id]
            state.stages.clear()
            state.last_stage_times.clear()

        batch_stage_times: dict[str, float] = {}
        blocked_ids: set[str] = set()
        for stage_name, stage in self._stages.items():
            for session_id in blocked_ids:
                self._session_states[session_id].stages[stage_name] = StageState.PENDING
            active_ids = tuple(session_id for session_id in ids if session_id not in blocked_ids)
            if not active_ids:
                continue

            if stage.check_sessions is None or stage.execute_sessions is None:
                for session_id in active_ids:
                    self._session_states[session_id].stages[stage_name] = StageState.SKIPPED
            else:
                try:
                    batch_needs_work = set(stage.check_sessions(active_ids)).intersection(active_ids)
                except Exception:
                    logger.warning("converger: session batch check failed stage=%s", stage_name, exc_info=True)
                    for session_id in active_ids:
                        state = self._session_states[session_id]
                        state.stages[stage_name] = StageState.FAILED
                        state.error_count += 1
                else:
                    for session_id in active_ids:
                        if session_id not in batch_needs_work:
                            self._session_states[session_id].stages[stage_name] = StageState.DONE

                    if batch_needs_work:
                        for session_id in batch_needs_work:
                            self._session_states[session_id].stages[stage_name] = StageState.IN_PROGRESS

                        t_stage = time.perf_counter()
                        try:
                            execute_result = stage.execute_sessions(tuple(batch_needs_work))
                        except Exception as exc:
                            logger.warning(
                                "converger: session batch execute failed stage=%s: %s",
                                stage_name,
                                exc,
                            )
                            for session_id in batch_needs_work:
                                state = self._session_states[session_id]
                                state.stages[stage_name] = StageState.FAILED
                                state.error_count += 1
                                state.last_error = str(exc)
                        else:
                            elapsed = time.perf_counter() - t_stage
                            success, extra_stage_timings_s = _coerce_execute_result(execute_result)
                            remaining_needs_work: set[str] | None = None
                            if not success and stage.false_means_pending:
                                try:
                                    remaining_needs_work = set(
                                        stage.check_sessions(tuple(batch_needs_work))
                                    ).intersection(batch_needs_work)
                                except Exception:
                                    logger.warning(
                                        "converger: session batch recheck failed stage=%s",
                                        stage_name,
                                        exc_info=True,
                                    )
                            _record_stage_times(
                                batch_stage_times,
                                stage_name,
                                elapsed,
                                extra_stage_timings_s,
                            )
                            for session_id in batch_needs_work:
                                state = self._session_states[session_id]
                                state.stage_times[stage_name] = elapsed
                                state.last_stage_times[stage_name] = elapsed
                                for name, value in extra_stage_timings_s.items():
                                    state.stage_times[name] = value
                                    state.last_stage_times[name] = value
                                session_success = success
                                if remaining_needs_work is not None:
                                    session_success = session_id not in remaining_needs_work
                                _record_execute_result(
                                    state,
                                    stage_name=stage_name,
                                    stage=stage,
                                    success=session_success,
                                    scope="session",
                                )

            blocked_ids.update(self._session_barriers_blocked(stage_name, stage, active_ids))

        results = {session_id: self._session_states[session_id] for session_id in ids}
        self._evict_converged_sessions(ids)
        return results, batch_stage_times

    def converge_all(
        self,
        files: Iterable[Path],
    ) -> dict[Path, FileState]:
        """Converge all files. Returns state map."""
        results: dict[Path, FileState] = {}
        for path in files:
            results[path] = self.converge_file(path)
        return results

    def pending_files(self) -> Iterator[Path]:
        """Yield files that haven't fully converged."""
        for path, state in self._file_states.items():
            if not state.converged:
                yield path

    def summary(self) -> dict[str, int]:
        """Return counts of files by convergence state."""
        total = len(self._file_states)
        converged = sum(1 for s in self._file_states.values() if s.converged)
        failed = sum(1 for s in self._file_states.values() if s.error_count > 0)
        return {
            "total": total,
            "converged": converged,
            "in_progress": total - converged - failed,
            "failed": failed,
        }


__all__ = [
    "SessionState",
    "ConvergenceStage",
    "DaemonConverger",
    "FileState",
    "StageState",
]
