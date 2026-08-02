"""Structured per-file acquisition decision logging + unclaimed-file sweep.

Before this module, the acquisition path had no per-file trail: nothing
recorded what a watched file was classified as, what evidence decided that
classification, or how long detection took -- and a file under a watched root
that no detector claimed left no record it was ever seen. This module gives
both halves a place to live, using the repo's existing structured-logging
convention (``polylogue.logging.get_logger`` / structlog keyword events), not
a bespoke framework or a new persisted table.

Two log events:

- ``file_acquisition_decision`` -- one per file the acquisition path (daemon
  watcher / ``ingest_batch`` pipeline) reads: path, size, mtime, the detected
  origin (or the literal string ``"UNRECOGNIZED"``), the specific detector
  evidence that decided it, and per-stage elapsed time.
- ``file_acquisition_unclaimed`` -- one per file a watch-root sweep finds that
  no detector/acquisition claimed, with the specific reason.

Both are wired into real production call sites (see
``polylogue/sources/source_acquisition_components.py:read_plain_source_file``
and ``polylogue/sources/live/watcher.py:LiveWatcher._scan_catch_up_candidates``),
not only exercised from tests.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.logging import get_logger

logger = get_logger(__name__)

#: The operator deliberately keeps a ``.git`` directory under some watched
#: roots (e.g. ``~/.claude/projects``) for their own versioning safety. It
#: must never be walked into, stat'd, read, or reported as an unclaimed file
#: -- exclusion happens at the directory-walk level below, not as a post-hoc
#: filter over an already-collected file list.
GIT_DIR_NAME = ".git"

UNRECOGNIZED_ORIGIN = "UNRECOGNIZED"


class AcquisitionStageTimings:
    """Accumulates named elapsed-time-in-milliseconds spans for one file."""

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._stages[name] = self._stages.get(name, 0.0) + elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        self._stages[name] = self._stages.get(name, 0.0) + elapsed_ms

    def as_dict(self) -> dict[str, float]:
        return {name: round(elapsed_ms, 3) for name, elapsed_ms in self._stages.items()}


def log_file_acquisition_decision(
    *,
    path: Path | str,
    size: int | None,
    mtime: float | None,
    origin: str | None,
    evidence: str,
    stage_timings: AcquisitionStageTimings | dict[str, float] | None = None,
    source_name: str | None = None,
) -> None:
    """Emit one structured ``file_acquisition_decision`` record.

    ``origin`` should be the resolved ``Origin``/``Provider`` token, or
    ``None``/``"UNKNOWN"`` when nothing claimed the file -- normalized here to
    the literal string :data:`UNRECOGNIZED_ORIGIN` so a log consumer can grep
    for refusals without special-casing every falsy spelling a caller might
    pass. ``evidence`` names the specific rule that decided the outcome (e.g.
    ``"claude.looks_like_code (envelope marker)"`` or
    ``"no detector matched (record)"``), never just the bare outcome.
    """
    normalized_origin = origin if origin and origin.upper() != "UNKNOWN" else UNRECOGNIZED_ORIGIN
    timings = (
        stage_timings.as_dict() if isinstance(stage_timings, AcquisitionStageTimings) else dict(stage_timings or {})
    )
    logger.info(
        "file_acquisition_decision",
        path=str(path),
        size=size,
        mtime=mtime,
        source_name=source_name,
        origin=normalized_origin,
        evidence=evidence,
        stage_timings_ms=timings,
    )


def log_unclaimed_file(
    *,
    path: Path | str,
    size: int | None,
    mtime: float | None,
    reason: str,
    source_name: str | None = None,
) -> None:
    """Emit one structured ``file_acquisition_unclaimed`` record.

    Used by :func:`sweep_unclaimed_files` and by the watcher's catch-up scan
    for a file under a watched root that no detector/acquisition claimed.
    ``reason`` should be specific (``"no detector matched"``,
    ``"matched an origin but failed validation"``, ``"suffix not in watched
    set for source <name>"``), not just "unclaimed".
    """
    logger.warning(
        "file_acquisition_unclaimed",
        path=str(path),
        size=size,
        mtime=mtime,
        source_name=source_name,
        reason=reason,
    )


def iter_files_excluding_git(
    root: Path,
    *,
    ignored_dir_names: frozenset[str] = frozenset({GIT_DIR_NAME}),
) -> Iterator[Path]:
    """Walk ``root`` yielding every regular file, never descending into ``.git``.

    Exclusion happens via ``os.walk``'s ``dirnames`` pruning (mutating the
    list in place before the walk descends), not as a filter applied to an
    already-collected listing -- a ``.git`` directory anywhere under ``root``
    (not only at its top level) is never opened, stat'd, or read.
    """
    if not root.exists():
        return
    if root.is_file():
        yield root
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in ignored_dir_names]
        for filename in filenames:
            yield Path(dirpath) / filename


@dataclass(frozen=True, slots=True)
class UnclaimedSweepResult:
    """Outcome of one :func:`sweep_unclaimed_files` call."""

    scanned: int
    unclaimed: tuple[Path, ...]


ClaimCheck = Callable[[Path], tuple[bool, str]]


def sweep_unclaimed_files(
    root: Path,
    *,
    source_name: str,
    is_claimed: ClaimCheck,
    ignored_dir_names: frozenset[str] = frozenset({GIT_DIR_NAME}),
) -> UnclaimedSweepResult:
    """Enumerate every file under ``root`` (excluding ``.git``), logging unclaimed ones.

    ``is_claimed(path) -> (claimed, reason)`` lets a caller supply whatever
    acquisition-claim evidence it has -- a real shape-detection check
    (:func:`default_file_claim_check`), a set of paths already ingested this
    run, or a source's suffix filter -- without this sweep needing to know
    about the ingest pipeline's internals. ``reason`` is logged verbatim for
    every unclaimed file via :func:`log_unclaimed_file`.
    """
    scanned = 0
    unclaimed: list[Path] = []
    for path in iter_files_excluding_git(root, ignored_dir_names=ignored_dir_names):
        scanned += 1
        claimed, reason = is_claimed(path)
        if claimed:
            continue
        unclaimed.append(path)
        try:
            stat_result = path.stat()
            size: int | None = stat_result.st_size
            mtime: float | None = stat_result.st_mtime
        except OSError:
            size, mtime = None, None
        log_unclaimed_file(path=path, size=size, mtime=mtime, reason=reason, source_name=source_name)
    return UnclaimedSweepResult(scanned=scanned, unclaimed=tuple(unclaimed))


#: Bounded prefix read for the sweep's own best-effort shape check -- mirrors
#: ``source_acquisition_components._DETECTION_PREFIX_SIZE``'s order of
#: magnitude without importing it (this module must stay import-light: it is
#: reused from both the daemon watcher and any diagnostic CLI/test caller).
_SWEEP_DETECTION_PREFIX_SIZE = 65536


def default_file_claim_check(path: Path) -> tuple[bool, str]:
    """Real shape-detection claim check for :func:`sweep_unclaimed_files`.

    Reads a bounded prefix and runs it through the same detector stack
    production acquisition uses (``sources.dispatch``), so "unclaimed" here
    means the same thing it means during real ingest: no detector recognized
    the file's shape, not merely "not on some caller-tracked allowlist".
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.dispatch import detect_provider_from_raw_bytes_evidence

    try:
        with path.open("rb") as handle:
            prefix = handle.read(_SWEEP_DETECTION_PREFIX_SIZE)
    except OSError as exc:
        return False, f"unreadable: {type(exc).__name__}: {exc}"
    if not prefix.strip():
        return False, "empty file"
    detected, evidence = detect_provider_from_raw_bytes_evidence(
        prefix,
        path.name,
        Provider.UNKNOWN,
        truncated_tail_ok=True,
    )
    if detected is Provider.UNKNOWN:
        return False, f"no detector matched ({evidence})"
    return True, evidence


__all__ = [
    "GIT_DIR_NAME",
    "UNRECOGNIZED_ORIGIN",
    "AcquisitionStageTimings",
    "ClaimCheck",
    "UnclaimedSweepResult",
    "default_file_claim_check",
    "iter_files_excluding_git",
    "log_file_acquisition_decision",
    "log_unclaimed_file",
    "sweep_unclaimed_files",
]
