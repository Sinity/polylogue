from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .cli_common import compute_prune_paths
from .importers import import_claude_code_session, import_codex_session
from .importers.base import ImportResult
from .importers.claude_code import DEFAULT_PROJECT_ROOT as CLAUDE_CODE_DEFAULT
from .importers.codex import _DEFAULT_BASE as CODEX_DEFAULT
from .render import MarkdownDocument
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
    diffs: int = 0


def _is_up_to_date(source: Path, target: Path) -> bool:
    if not target.exists():
        return False
    try:
        return int(target.stat().st_mtime) >= int(source.stat().st_mtime)
    except OSError:
        return False


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
    written: List[ImportResult] = []
    skipped = 0
    wanted: List[str] = []
    importer_kwargs = importer_kwargs or {}
    attachments_total = 0
    attachment_bytes_total = 0
    tokens_total = 0

    diff_total = 0
    for session_path in sorted(sessions):
        if not session_path.is_file():
            continue
        safe_name = sanitize_filename(session_path.stem)
        md_path = output_dir / f"{safe_name}.md"
        wanted.append(safe_name)
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
            **importer_kwargs,
        )
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
            os.utime(result.markdown_path, (session_mtime, session_mtime))
        except OSError:
            pass
        if result.html_path:
            try:
                os.utime(result.html_path, (session_mtime, session_mtime))
            except OSError:
                pass
        attachments_total += len(result.document.attachments)
        attachment_bytes_total += result.document.metadata.get("attachmentBytes", 0) or 0
        tokens_total += result.document.stats.get("totalTokensApprox", 0) or 0
        written.append(result)

    pruned = 0
    if prune:
        wanted_set = set(wanted)
        for path in compute_prune_paths(output_dir, wanted_set):
            try:
                if path.is_dir():
                    import shutil

                    shutil.rmtree(path)
                else:
                    path.unlink()
                pruned += 1
            except OSError:
                continue

    return LocalSyncResult(
        written=written,
        skipped=skipped,
        pruned=pruned,
        output_dir=output_dir,
        attachments=attachments_total,
        attachment_bytes=attachment_bytes_total,
        tokens=tokens_total,
        diffs=diff_total,
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
