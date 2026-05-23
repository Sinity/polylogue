"""Daemon state convergence — check and converge to desired archive state.

The daemon owns all writes. For each source file, the desired state is:
  1. Raw blob stored (content-addressed)
  2. Parsed into records (provider detection + record extraction)
  3. Messages materialized (normalized into messages table)
  4. FTS indexed (searchable)
  5. Insights refreshed (session profiles, work events, etc.)

Convergence means checking the current state for each file and doing
only the missing work. Cursor records track the last-known file state
so we skip unchanged files entirely.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.pipeline.services.process_pool import process_pool_executor

logger = get_logger(__name__)


class StageState(Enum):
    PENDING = "pending"  # work needed
    IN_PROGRESS = "in_progress"  # work running
    DONE = "done"  # converged
    SKIPPED = "skipped"  # not applicable
    FAILED = "failed"  # error, will retry


@dataclass(frozen=True, slots=True)
class ConvergenceStage:
    """A named pipeline stage with a check function and an execute function."""

    name: str
    description: str
    # Check returns True if this stage needs work for the given file.
    check: Callable[[Path], bool]
    # Execute performs the work. Returns True on success.
    execute: Callable[[Path], bool]
    # Optional batch check/execute pair for stages that can collapse many
    # changed source paths into one repair transaction.
    check_many: Callable[[Sequence[Path]], set[Path]] | None = None
    execute_many: Callable[[Sequence[Path]], bool] | None = None
    # Optional conversation-scoped pair for durable convergence debt retries.
    # These avoid resolving a failed derived subject back to source files.
    check_conversations: Callable[[Sequence[str]], set[str]] | None = None
    execute_conversations: Callable[[Sequence[str]], bool] | None = None
    # Can run in a worker process (CPU-bound, no SQLite write).
    cpu_bound: bool = False


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
class ConversationState:
    """Tracked convergence state for one conversation subject."""

    conversation_id: str
    stages: dict[str, StageState] = field(default_factory=dict)
    stage_times: dict[str, float] = field(default_factory=dict)
    last_stage_times: dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    last_error: str | None = None

    @property
    def converged(self) -> bool:
        return all(s in (StageState.DONE, StageState.SKIPPED) for s in self.stages.values())


class DaemonConverger:
    """Drives archive state toward desired state for all source files.

    Runs a set of :class:`ConvergenceStage` checks against each file.
    CPU-bound stages are dispatched to a :class:`~concurrent.futures.ProcessPoolExecutor`.
    The main process is the only SQLite writer.
    """

    def __init__(
        self,
        stages: Iterable[ConvergenceStage],
        *,
        max_workers: int | None = None,
    ) -> None:
        self._stages: dict[str, ConvergenceStage] = {s.name: s for s in stages}
        self._max_workers = max_workers or 2
        self._file_states: dict[Path, FileState] = {}
        self._conversation_states: dict[str, ConversationState] = {}
        self._executor: ProcessPoolExecutor | None = None

    @property
    def stage_names(self) -> list[str]:
        return list(self._stages)

    def _has_cpu_bound_stage(self) -> bool:
        return any(stage.cpu_bound for stage in self._stages.values())

    async def start(self) -> None:
        if self._executor is not None:
            return
        if not self._has_cpu_bound_stage():
            logger.info(
                "converger: started without worker pool, stages=%s",
                list(self._stages),
            )
            return
        self._executor = process_pool_executor(max_workers=self._max_workers)
        logger.info(
            "converger: started with %d worker(s), stages=%s",
            self._max_workers,
            list(self._stages),
        )

    async def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("converger: stopped")

    def converge_file(self, path: Path) -> FileState:
        """Converge a single file through all stages.

        Returns the final :class:`FileState`.
        """
        if path not in self._file_states:
            self._file_states[path] = FileState(path=path)
        state = self._file_states[path]
        state.last_stage_times.clear()

        for stage_name, stage in self._stages.items():
            current = state.stages.get(stage_name)
            if current == StageState.DONE:
                continue

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

            if not needs_work:
                state.stages[stage_name] = StageState.DONE
                continue

            state.stages[stage_name] = StageState.IN_PROGRESS

            t_stage = time.perf_counter()
            try:
                if stage.cpu_bound and self._executor is not None:
                    future = self._executor.submit(stage.execute, path)
                    success = future.result()
                else:
                    success = stage.execute(path)
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
            state.stage_times[stage_name] = elapsed
            state.last_stage_times[stage_name] = elapsed
            state.stages[stage_name] = StageState.DONE if success else StageState.FAILED
            if not success:
                state.error_count += 1
                state.last_error = f"stage {stage_name} returned False"

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

    def _evict_converged_conversations(self, conversation_ids: Iterable[str]) -> None:
        for conversation_id in conversation_ids:
            state = self._conversation_states.get(conversation_id)
            if state is not None and state.converged:
                del self._conversation_states[conversation_id]

    def converge_batch(self, files: Iterable[Path]) -> tuple[dict[Path, FileState], dict[str, float]]:
        """Converge a changed source batch and return per-stage batch timings."""
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
        for stage_name, stage in self._stages.items():
            if stage.check_many is None or stage.execute_many is None or stage.cpu_bound:
                for path in paths:
                    state = self._file_states[path]
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

                    if not needs_work:
                        state.stages[stage_name] = StageState.DONE
                        continue

                    state.stages[stage_name] = StageState.IN_PROGRESS
                    t_stage = time.perf_counter()
                    try:
                        success = stage.execute(path)
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
                    batch_stage_times[stage_name] = batch_stage_times.get(stage_name, 0.0) + elapsed
                    state.stage_times[stage_name] = elapsed
                    state.last_stage_times[stage_name] = elapsed
                    state.stages[stage_name] = StageState.DONE if success else StageState.FAILED
                    if not success:
                        state.error_count += 1
                        state.last_error = f"stage {stage_name} returned False"
                continue

            try:
                batch_needs_work = stage.check_many(paths)
            except Exception:
                logger.warning("converger: batch check failed stage=%s", stage_name, exc_info=True)
                for path in paths:
                    state = self._file_states[path]
                    state.stages[stage_name] = StageState.FAILED
                    state.error_count += 1
                continue

            for path in paths:
                if path not in batch_needs_work:
                    self._file_states[path].stages[stage_name] = StageState.DONE

            if not batch_needs_work:
                continue

            for path in batch_needs_work:
                self._file_states[path].stages[stage_name] = StageState.IN_PROGRESS

            t_stage = time.perf_counter()
            try:
                success = stage.execute_many(tuple(batch_needs_work))
            except Exception as exc:
                logger.warning("converger: batch execute failed stage=%s: %s", stage_name, exc)
                success = False

            elapsed = time.perf_counter() - t_stage
            batch_stage_times[stage_name] = elapsed
            for path in batch_needs_work:
                state = self._file_states[path]
                state.stage_times[stage_name] = elapsed
                state.last_stage_times[stage_name] = elapsed
                state.stages[stage_name] = StageState.DONE if success else StageState.FAILED
                if not success:
                    state.error_count += 1
                    state.last_error = f"batch stage {stage_name} returned False"

        results = {path: self._file_states[path] for path in paths}
        self._evict_converged_files(paths)
        return results, batch_stage_times

    def converge_conversations(
        self,
        conversation_ids: Iterable[str],
    ) -> tuple[dict[str, ConversationState], dict[str, float]]:
        """Converge derived state for known conversation IDs without source-path lookup."""
        ids = tuple(dict.fromkeys(str(conversation_id) for conversation_id in conversation_ids if conversation_id))
        if not ids:
            return {}, {}

        for conversation_id in ids:
            if conversation_id not in self._conversation_states:
                self._conversation_states[conversation_id] = ConversationState(conversation_id=conversation_id)
            state = self._conversation_states[conversation_id]
            state.stages.clear()
            state.last_stage_times.clear()

        batch_stage_times: dict[str, float] = {}
        for stage_name, stage in self._stages.items():
            if stage.check_conversations is None or stage.execute_conversations is None or stage.cpu_bound:
                for conversation_id in ids:
                    state = self._conversation_states[conversation_id]
                    state.stages[stage_name] = StageState.SKIPPED
                continue

            try:
                batch_needs_work = stage.check_conversations(ids)
            except Exception:
                logger.warning("converger: conversation batch check failed stage=%s", stage_name, exc_info=True)
                for conversation_id in ids:
                    state = self._conversation_states[conversation_id]
                    state.stages[stage_name] = StageState.FAILED
                    state.error_count += 1
                continue

            for conversation_id in ids:
                if conversation_id not in batch_needs_work:
                    self._conversation_states[conversation_id].stages[stage_name] = StageState.DONE

            if not batch_needs_work:
                continue

            for conversation_id in batch_needs_work:
                self._conversation_states[conversation_id].stages[stage_name] = StageState.IN_PROGRESS

            t_stage = time.perf_counter()
            try:
                success = stage.execute_conversations(tuple(batch_needs_work))
            except Exception as exc:
                logger.warning("converger: conversation batch execute failed stage=%s: %s", stage_name, exc)
                success = False

            elapsed = time.perf_counter() - t_stage
            batch_stage_times[stage_name] = elapsed
            for conversation_id in batch_needs_work:
                state = self._conversation_states[conversation_id]
                state.stage_times[stage_name] = elapsed
                state.last_stage_times[stage_name] = elapsed
                state.stages[stage_name] = StageState.DONE if success else StageState.FAILED
                if not success:
                    state.error_count += 1
                    state.last_error = f"conversation stage {stage_name} returned False"

        results = {conversation_id: self._conversation_states[conversation_id] for conversation_id in ids}
        self._evict_converged_conversations(ids)
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
    "ConversationState",
    "ConvergenceStage",
    "DaemonConverger",
    "FileState",
    "StageState",
]
