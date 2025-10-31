from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .cli_common import compute_prune_paths
from .importers import import_claude_code_session, import_codex_session
from .importers.base import ImportResult
from .importers.claude_code import DEFAULT_PROJECT_ROOT as CLAUDE_CODE_DEFAULT
from .importers.codex import _DEFAULT_BASE as CODEX_DEFAULT
from .util import sanitize_filename, snapshot_for_diff, write_delta_diff


@dataclass
class LocalSyncResult:
    written: List[ImportResult]
    skipped: int
    pruned: int
    output_dir: Path
    attachments: int = 0
    attachment_bytes: int = 0
    tokens: int = 0
    words: int = 0
    diffs: int = 0
    duration: float = 0.0


def _mtime_ns(path: Path) -> int:
    stat_result = path.stat()
    return getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))


def _is_up_to_date(source: Path, target: Path) -> bool:
    if not target.exists():
        return False
    try:
        source_ns = _mtime_ns(source)
        target_ns = _mtime_ns(target)
    except OSError:
        return False
    if target_ns < source_ns:
        return False
    delta = target_ns - source_ns
    if delta <= 1_000_000:  # ≤1ms difference, verify contents match
        try:
            if target.read_bytes() != source.read_bytes():
                return False
        except OSError:
            return False
    return True


def _session_order_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, str(path))
    except OSError:
        return (0.0, str(path))


def _sync_sessions(
    sessions: Iterable[Path],
    *,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    import_fn,
    importer_kwargs: Optional[dict] = None,
) -> LocalSyncResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    written: List[ImportResult] = []
    skipped = 0
    wanted: set[str] = set()
    importer_kwargs = importer_kwargs or {}
    attachments_total = 0
    attachment_bytes_total = 0
    tokens_total = 0
    words_total = 0

    diff_total = 0
    if isinstance(sessions, (list, tuple)):
        iterable: Iterable[Path] = sorted(sessions, key=_session_order_key)
    else:
        iterable = sessions

    for session_path in iterable:
        if not session_path.is_file():
            continue
        safe_name = sanitize_filename(session_path.stem)
        conversation_dir = output_dir / safe_name
        md_path = conversation_dir / "conversation.md"
        wanted.add(safe_name)
        if not force and _is_up_to_date(session_path, md_path):
            skipped += 1
            continue
        snapshot = None
        if diff:
            snapshot = snapshot_for_diff(md_path)
        result = import_fn(
            str(session_path),
            output_dir=output_dir,
            collapse_threshold=collapse_threshold,
            html=html,
            html_theme=html_theme,
            force=force,
            **importer_kwargs,
        )
        if result.skipped:
            if snapshot is not None:
                try:
                    snapshot.unlink()
                except Exception:
                    pass
            skipped += 1
            continue
        if diff and snapshot is not None:
            result.diff_path = write_delta_diff(snapshot, result.markdown_path)
            try:
                snapshot.unlink()
            except Exception:
                pass
        elif snapshot is not None:
            try:
                snapshot.unlink()
            except Exception:
                pass
        if result.diff_path:
            diff_total += 1
        session_mtime = int(session_path.stat().st_mtime)
        try:
            result.markdown_path.parent.mkdir(parents=True, exist_ok=True)
            os.utime(result.markdown_path, (session_mtime, session_mtime))
        except OSError:
            pass
        if result.html_path:
            try:
                os.utime(result.html_path, (session_mtime, session_mtime))
            except OSError:
                pass
        if not result.skipped:
            wanted.add(result.slug)
        if result.document:
            attachments_total += len(result.document.attachments)
            attachment_bytes_total += result.document.metadata.get("attachmentBytes", 0) or 0
            tokens_total += result.document.stats.get("totalTokensApprox", 0) or 0
            words_total += result.document.stats.get("totalWordsApprox", 0) or 0
        written.append(result)

    pruned = 0
    if prune:
        for path in compute_prune_paths(output_dir, wanted):
            try:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path)
                else:
                    path.unlink()
                pruned += 1
            except OSError:
                continue

    duration = time.perf_counter() - start_time

    return LocalSyncResult(
        written=written,
        skipped=skipped,
        pruned=pruned,
        output_dir=output_dir,
        attachments=attachments_total,
        attachment_bytes=attachment_bytes_total,
        tokens=tokens_total,
        words=words_total,
        diffs=diff_total,
        duration=duration,
    )


def sync_codex_sessions(
    *,
    base_dir: Path = CODEX_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    if sessions is None:
        sessions = base_dir.rglob("*.jsonl")
    return _sync_sessions(
        sessions,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        html=html,
        html_theme=html_theme,
        force=force,
        prune=prune,
        diff=diff,
        import_fn=lambda session_id, **kwargs: import_codex_session(
            session_id,
            base_dir=base_dir,
            **kwargs,
        ),
    )


def sync_claude_code_sessions(
    *,
    base_dir: Path = CLAUDE_CODE_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    if sessions is None:
        sessions = base_dir.rglob("*.jsonl")
    return _sync_sessions(
        sessions,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        html=html,
        html_theme=html_theme,
        force=force,
        prune=prune,
        diff=diff,
        import_fn=lambda session_id, **kwargs: import_claude_code_session(
            session_id,
            base_dir=base_dir,
            **kwargs,
        ),
    )
