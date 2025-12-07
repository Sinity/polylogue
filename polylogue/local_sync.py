from __future__ import annotations

import os
import time
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .cli_common import compute_prune_paths
from .importers import (
    import_chatgpt_export,
    import_claude_code_session,
    import_claude_export,
    import_codex_session,
)
from .importers.base import ImportResult
from .importers.claude_code import (
    DEFAULT_PROJECT_ROOT as CLAUDE_CODE_DEFAULT,
    list_claude_code_sessions,
)
from .importers.codex import _DEFAULT_BASE as CODEX_DEFAULT
from .paths import DATA_HOME
from .util import DiffTracker, path_order_key, sanitize_filename, slugify_title
from .services.conversation_registrar import ConversationRegistrar, create_default_registrar
from .pipeline_runner import Pipeline, PipelineContext
from .config import CONFIG
from .ui import UI


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
    supports_watch: bool = True
    supports_diff: bool = True
    create_base_dir: bool = False
    watch_attachments: Tuple[str, ...] = ()


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


INBOX_ROOT = DATA_HOME / "inbox"
CHATGPT_EXPORTS_DEFAULT = CONFIG.exports.chatgpt
CLAUDE_EXPORTS_DEFAULT = CONFIG.exports.claude


class _NoopProgress:
    def __enter__(self):
        return self

    def advance(self, *_args, **_kwargs) -> None:
        return None

    def __exit__(self, *_exc) -> None:
        return None


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
    ui: Optional[UI] = None,
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
        iterable: Iterable[Path] = sorted((Path(p) for p in sessions), key=path_order_key)
    else:
        iterable = sorted(list(sessions), key=path_order_key)

    session_list = list(iterable)
    progress_ctx = ui.progress(f"Syncing {provider} sessions", total=len(session_list)) if ui else _NoopProgress()

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

    with progress_ctx as tracker:
        for session_path in session_list:
            ctx = PipelineContext(env=None, options=None, data={"session_path": session_path})
            pipeline.run(ctx)
            session_output = ctx.get("session_result", {})
            prune_slug = session_output.get("prune_slug")
            if isinstance(prune_slug, str):
                wanted.add(prune_slug)

            if session_output.get("skipped"):
                skipped += 1
                tracker.advance()
                continue

            result = session_output.get("result")
            if not isinstance(result, ImportResult):
                tracker.advance()
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
            tracker.advance()

    pruned = 0
    if prune:
        trash_dir = output_dir / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        for path in compute_prune_paths(output_dir, wanted):
            try:
                target_name = f"{path.name}.{int(time.time())}"
                trash_path = trash_dir / target_name
                path.rename(trash_path)
                import shutil

                shutil.rmtree(trash_path)
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


def _detect_export_provider(path: Path) -> Optional[str]:
    conv_path: Optional[Path] = None
    suffix = path.suffix.lower()
    if path.is_dir():
        candidate = path / "conversations.json"
        if candidate.exists():
            conv_path = candidate
    elif path.is_file() and path.name.lower() == "conversations.json":
        conv_path = path
    elif path.is_file() and suffix == ".zip" and zipfile.is_zipfile(path):
        try:
            with zipfile.ZipFile(path) as zf:
                if "conversations.json" in zf.namelist():
                    with zf.open("conversations.json") as fh:
                        data = json.loads(fh.read().decode("utf-8"))
                        return _classify_conversations_json(data)
        except Exception:
            return None

    if conv_path and conv_path.exists():
        try:
            data = json.loads(conv_path.read_text(encoding="utf-8"))
            return _classify_conversations_json(data)
        except Exception:
            return None

    # Fallback: hint from filename
    name = path.name.lower()
    if "chatgpt" in name:
        return "chatgpt"
    if "claude" in name:
        return "claude"
    return None


def _classify_conversations_json(data: object) -> Optional[str]:
    if isinstance(data, dict):
        conversations = data.get("conversations") if isinstance(data.get("conversations"), list) else None
        if conversations:
            data = conversations
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            if "mapping" in first:
                return "chatgpt"
            # Claude exports typically lack mapping and include simple message lists
            return "claude"
    return None


def _discover_export_targets(base_dir: Path, *, provider: Optional[str] = None) -> List[Path]:
    base = base_dir.expanduser()
    if not base.exists():
        return []
    provider_hint = base.name.lower() if provider else None
    candidates: set[Path] = set()
    try:
        for zip_path in base.rglob("*.zip"):
            candidates.add(zip_path)
    except OSError:
        pass
    try:
        for conv_file in base.rglob("conversations.json"):
            candidates.add(conv_file.parent)
    except OSError:
        pass
    results: List[Path] = []
    for cand in sorted(candidates, key=path_order_key, reverse=True):
        detected = _detect_export_provider(cand)
        if provider:
            if detected and detected != provider:
                continue
            if detected is None:
                hint = provider_hint or ""
                path_str = str(cand).lower()
                if provider not in hint and provider not in path_str:
                    continue
        results.append(cand)
    return results


def _normalize_export_target(path: Path) -> Optional[Path]:
    candidate = Path(path).expanduser()
    suffix = candidate.suffix.lower()
    if suffix == ".zip":
        return candidate
    if candidate.name.lower() == "conversations.json":
        return candidate.parent
    if candidate.is_dir() and (candidate / "conversations.json").exists():
        return candidate
    return None


def _sync_export_bundles(
    bundles: Iterable[Path],
    *,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    provider: str,
    import_fn: Callable[..., Sequence[ImportResult] | ImportResult | None],
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
) -> LocalSyncResult:
    registrar = registrar or create_default_registrar()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    written: List[ImportResult] = []
    skipped_exports = 0
    wanted: set[str] = set()
    attachments_total = 0
    attachment_bytes_total = 0
    tokens_total = 0
    words_total = 0

    normalized = [_normalize_export_target(Path(p)) for p in bundles]
    filtered = [p for p in normalized if p is not None]
    if filtered:
        iterable: Iterable[Path] = sorted(filtered, key=path_order_key)
    else:
        iterable = sorted([Path(p) for p in bundles], key=path_order_key)

    exports = list(iterable)
    progress_ctx = ui.progress(f"Processing {provider} exports", total=len(exports)) if ui else _NoopProgress()

    with progress_ctx as tracker:
        for export_path in exports:
            if not export_path.exists():
                tracker.advance()
                continue
            results_raw = import_fn(
                export_path=export_path,
                output_dir=output_dir,
                collapse_threshold=collapse_threshold,
                html=html,
                html_theme=html_theme,
                selected_ids=None,
                force=force,
                registrar=registrar,
            )
            if results_raw is None:
                result_list: List[ImportResult] = []
            elif isinstance(results_raw, ImportResult):
                result_list = [results_raw]
            else:
                result_list = list(results_raw)

            export_wrote = False
            for result in result_list:
                if not isinstance(result, ImportResult):
                    continue
                wanted.add(result.slug)
                if result.skipped:
                    continue
                export_wrote = True
                if result.document:
                    attachments_total += len(result.document.attachments)
                    attachment_bytes_total += result.document.metadata.get("attachmentBytes", 0) or 0
                    tokens_total += result.document.stats.get("totalTokensApprox", 0) or 0
                    words_total += result.document.stats.get("totalWordsApprox", 0) or 0
                written.append(result)
            if not export_wrote:
                skipped_exports += 1
            tracker.advance()

    pruned = 0
    if prune:
        trash_dir = output_dir / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        for path in compute_prune_paths(output_dir, wanted):
            try:
                target_name = f"{path.name}.{int(time.time())}"
                trash_path = trash_dir / target_name
                path.rename(trash_path)
                import shutil

                shutil.rmtree(trash_path)
                pruned += 1
            except OSError:
                continue

    duration = time.perf_counter() - start_time

    return LocalSyncResult(
        written=written,
        skipped=skipped_exports,
        pruned=pruned,
        output_dir=output_dir,
        attachments=attachments_total,
        attachment_bytes=attachment_bytes_total,
        tokens=tokens_total,
        words=words_total,
        diffs=0,
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
    ui: Optional[UI] = None,
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
        ui=ui,
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
    ui: Optional[UI] = None,
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
        ui=ui,
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


def _list_chatgpt_exports(base_dir: Path) -> List[Path]:
    return _discover_export_targets(base_dir)


def _list_claude_exports(base_dir: Path) -> List[Path]:
    return _discover_export_targets(base_dir)


def sync_chatgpt_exports(
    *,
    base_dir: Path = CHATGPT_EXPORTS_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    targets = sessions or _discover_export_targets(base_dir, provider="chatgpt")
    if sessions:
        targets = [_normalize_export_target(Path(p)) for p in sessions if p]
    targets = [t for t in targets if t]
    if not targets:
        targets = _discover_export_targets(base_dir, provider="chatgpt")
    return _sync_export_bundles(
        targets,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        html=html,
        html_theme=html_theme,
        force=force,
        prune=prune,
        provider="chatgpt",
        import_fn=import_chatgpt_export,
        registrar=registrar,
        ui=ui,
    )


def sync_claude_exports(
    *,
    base_dir: Path = CLAUDE_EXPORTS_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    targets = sessions or _discover_export_targets(base_dir, provider="claude")
    if sessions:
        targets = [_normalize_export_target(Path(p)) for p in sessions if p]
    targets = [t for t in targets if t]
    if not targets:
        targets = _discover_export_targets(base_dir, provider="claude")
    return _sync_export_bundles(
        targets,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        html=html,
        html_theme=html_theme,
        force=force,
        prune=prune,
        provider="claude",
        import_fn=import_claude_export,
        registrar=registrar,
        ui=ui,
    )


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
    "chatgpt": LocalSyncProvider(
        name="chatgpt",
        title="ChatGPT Exports",
        sync_fn=sync_chatgpt_exports,
        default_base=CHATGPT_EXPORTS_DEFAULT,
        default_output=CONFIG.defaults.output_dirs.import_chatgpt,
        list_sessions=_list_chatgpt_exports,
        watch_banner="Watching ChatGPT exports",
        watch_log_title="ChatGPT Export Watch",
        watch_suffixes=(".zip", ".json"),
        supports_watch=True,
        supports_diff=False,
        create_base_dir=True,
    ),
    "claude": LocalSyncProvider(
        name="claude",
        title="Claude.ai Exports",
        sync_fn=sync_claude_exports,
        default_base=CLAUDE_EXPORTS_DEFAULT,
        default_output=CONFIG.defaults.output_dirs.import_claude,
        list_sessions=_list_claude_exports,
        watch_banner="Watching Claude exports",
        watch_log_title="Claude Export Watch",
        watch_suffixes=(".zip", ".json"),
        supports_watch=True,
        supports_diff=False,
        create_base_dir=True,
    ),
}

LOCAL_SYNC_PROVIDER_NAMES: Tuple[str, ...] = tuple(LOCAL_SYNC_PROVIDERS.keys())
WATCHABLE_LOCAL_PROVIDER_NAMES: Tuple[str, ...] = tuple(
    name for name, provider in LOCAL_SYNC_PROVIDERS.items() if provider.supports_watch
)


def get_local_provider(name: str) -> LocalSyncProvider:
    try:
        return LOCAL_SYNC_PROVIDERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown local provider: {name}") from exc
