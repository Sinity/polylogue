from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .cli_common import compute_prune_paths
from .importers import import_claude_code_session, import_codex_session
from .importers.base import ImportResult
from .importers.claude_code import (
    DEFAULT_PROJECT_ROOT as CLAUDE_CODE_DEFAULT,
    list_claude_code_sessions,
)
from .importers.codex import _DEFAULT_BASE as CODEX_DEFAULT
from .util import DiffTracker, path_order_key, sanitize_filename, slugify_title
from .services.conversation_registrar import ConversationRegistrar, create_default_registrar
from .pipeline_runner import Pipeline, PipelineContext
from .config import CONFIG


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


LocalSyncFn = Callable[..., LocalSyncResult]


@dataclass(frozen=True)
class LocalSyncProvider:
    name: str
    title: str
    sync_fn: LocalSyncFn
    default_base: Path
    default_output: Path
    list_sessions: Callable[[Path], List[Path]]
    watch_banner: str
    watch_log_title: str
    watch_suffixes: Tuple[str, ...] = (".jsonl",)


class _LocalSessionStage:
    def __init__(self, processor: Callable[[Path], Dict[str, object]]):
        self._processor = processor
        self.name = "LocalSessionProcess"

    def run(self, context: PipelineContext) -> None:
        session_path: Path = context.require("session_path")
        context.set("session_result", self._processor(session_path))


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
    # Allow for minor clock skew (≤5ms) to avoid unnecessary re-imports on equal times.
    tolerance = 5_000_000
    if target_ns + tolerance < source_ns:
        return False
    return True


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
    provider: str,
    import_fn,
    importer_kwargs: Optional[dict] = None,
    registrar: Optional[ConversationRegistrar] = None,
    branch_mode: str = "full",
) -> LocalSyncResult:
    registrar = registrar or create_default_registrar()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    written: List[ImportResult] = []
    skipped = 0
    wanted: set[str] = set()
    importer_kwargs = importer_kwargs or {}
    importer_kwargs = dict(importer_kwargs)
    importer_kwargs.setdefault("registrar", registrar)
    attachments_total = 0
    attachment_bytes_total = 0
    tokens_total = 0
    words_total = 0

    diff_total = 0
    if isinstance(sessions, (list, tuple)):
        iterable: Iterable[Path] = sorted(sessions, key=path_order_key)
    else:
        iterable = sessions

    output_root = output_dir.resolve()

    def _process_single(session_path: Path) -> Dict[str, object]:
        entry: Dict[str, object] = {
            "session_path": session_path,
            "skipped": False,
            "result": None,
            "prune_slug": None,
            "diff": False,
        }
        if not session_path.is_file():
            entry["skipped"] = True
            return entry

        conversation_id = str(session_path)
        state_entry = registrar.get_state(provider, conversation_id)
        state_slug = None
        if isinstance(state_entry, dict):
            raw_slug = state_entry.get("slug")
            if isinstance(raw_slug, str) and raw_slug.strip():
                state_slug = raw_slug.strip()

        fallback_title = session_path.stem
        fallback_slug = slugify_title(fallback_title)
        if not fallback_slug:
            fallback_slug = sanitize_filename(fallback_title).replace(" ", "-") or fallback_title

        slug_hint = state_slug or fallback_slug
        md_path = output_dir / slug_hint / "conversation.md"
        if isinstance(state_entry, dict):
            raw_output = state_entry.get("outputPath")
            if isinstance(raw_output, str) and raw_output.strip():
                candidate_path = Path(raw_output)
                if candidate_path.exists():
                    md_path = candidate_path

        slug_for_prune = slug_hint
        try:
            rel = md_path.parent.resolve().relative_to(output_root)
            if rel.parts:
                slug_for_prune = rel.parts[0]
        except ValueError:
            pass

        entry["prune_slug"] = slug_for_prune

        if not force and _is_up_to_date(session_path, md_path):
            entry["skipped"] = True
            return entry

        diff_tracker = DiffTracker(md_path, diff)
        result = import_fn(
            str(session_path),
            output_dir=output_dir,
            collapse_threshold=collapse_threshold,
            html=html,
            html_theme=html_theme,
            force=force,
            branch_mode=branch_mode,
            **importer_kwargs,
        )
        if result.skipped:
            diff_tracker.cleanup()
            entry["skipped"] = True
            return entry

        result.diff_path = diff_tracker.finalize(result.markdown_path)
        entry["diff"] = bool(result.diff_path)

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

        entry["result"] = result
        return entry

    pipeline = Pipeline([_LocalSessionStage(_process_single)])

    for session_path in iterable:
        ctx = PipelineContext(env=None, options=None, data={"session_path": session_path})
        pipeline.run(ctx)
        session_output = ctx.get("session_result", {})
        prune_slug = session_output.get("prune_slug")
        if isinstance(prune_slug, str):
            wanted.add(prune_slug)

        if session_output.get("skipped"):
            skipped += 1
            continue

        result = session_output.get("result")
        if not isinstance(result, ImportResult):
            continue

        if session_output.get("diff"):
            diff_total += 1

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
    registrar: Optional[ConversationRegistrar] = None,
    branch_mode: str = "full",
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
        provider="codex",
        import_fn=lambda session_id, **kwargs: import_codex_session(
            session_id,
            base_dir=base_dir,
            **kwargs,
        ),
        registrar=registrar,
        branch_mode=branch_mode,
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
    registrar: Optional[ConversationRegistrar] = None,
    branch_mode: str = "full",
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
        provider="claude-code",
        import_fn=lambda session_id, **kwargs: import_claude_code_session(
            session_id,
            base_dir=base_dir,
            **kwargs,
        ),
        registrar=registrar,
        branch_mode=branch_mode,
    )


def _list_codex_paths(base_dir: Path) -> List[Path]:
    return sorted(base_dir.expanduser().rglob("*.jsonl"), key=path_order_key, reverse=True)


def _list_claude_code_paths(base_dir: Path) -> List[Path]:
    entries = list_claude_code_sessions(base_dir)
    paths: List[Path] = []
    for entry in entries:
        raw = entry.get("path")
        if isinstance(raw, str):
            paths.append(Path(raw))
    return paths


LOCAL_SYNC_PROVIDERS: Dict[str, LocalSyncProvider] = {
    "codex": LocalSyncProvider(
        name="codex",
        title="Codex",
        sync_fn=sync_codex_sessions,
        default_base=CODEX_DEFAULT,
        default_output=CONFIG.defaults.output_dirs.sync_codex,
        list_sessions=_list_codex_paths,
        watch_banner="Watching Codex sessions",
        watch_log_title="Codex Watch",
        watch_suffixes=(".jsonl",),
    ),
    "claude-code": LocalSyncProvider(
        name="claude-code",
        title="Claude Code",
        sync_fn=sync_claude_code_sessions,
        default_base=CLAUDE_CODE_DEFAULT,
        default_output=CONFIG.defaults.output_dirs.sync_claude_code,
        list_sessions=_list_claude_code_paths,
        watch_banner="Watching Claude Code sessions",
        watch_log_title="Claude Code Watch",
        watch_suffixes=(".jsonl",),
    ),
}

LOCAL_SYNC_PROVIDER_NAMES: Tuple[str, ...] = tuple(LOCAL_SYNC_PROVIDERS.keys())


def get_local_provider(name: str) -> LocalSyncProvider:
    try:
        return LOCAL_SYNC_PROVIDERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown local provider: {name}") from exc
