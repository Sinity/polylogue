from __future__ import annotations

import os
import time
import json
import zipfile
import fnmatch
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
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
    extract_claude_code_session_id,
    list_claude_code_sessions,
)
from .importers.codex import _DEFAULT_BASE as CODEX_DEFAULT
from .paths import DATA_HOME
from .util import DiffTracker, format_duration, path_order_key, sanitize_filename, slugify_title
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
    ignored: int = 0
    attachments: int = 0
    attachment_bytes: int = 0
    tokens: int = 0
    words: int = 0
    diffs: int = 0
    duration: float = 0.0
    failures: int = 0
    failed: List[Dict[str, str]] = field(default_factory=list)
    skip_reasons: Dict[str, int] = field(default_factory=dict)


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
    supports_jobs: bool = False
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


def _latest_mtime_ns(paths: Iterable[Path]) -> Optional[int]:
    latest: Optional[int] = None
    for path in paths:
        try:
            value = _mtime_ns(path)
        except OSError:
            continue
        if latest is None or value > latest:
            latest = value
    return latest


def _resolve_source_paths(
    provider: str,
    session_path: Path,
    state_entry: Optional[Dict[str, object]],
    session_id: Optional[str],
) -> List[Path]:
    if provider != "claude-code":
        return [session_path]
    sources: List[Path] = []
    if isinstance(state_entry, dict):
        raw_files = state_entry.get("sessionFiles") or state_entry.get("sessionFile")
        if isinstance(raw_files, list):
            for entry in raw_files:
                if isinstance(entry, str) and entry.strip():
                    sources.append(Path(entry.strip()))
        elif isinstance(raw_files, str) and raw_files.strip():
            sources.append(Path(raw_files.strip()))
    if session_id:
        for candidate in session_path.parent.glob("*.jsonl"):
            if candidate in sources:
                continue
            if extract_claude_code_session_id(candidate) == session_id:
                sources.append(candidate)
    return sources or [session_path]


def _is_up_to_date_multi(sources: Iterable[Path], target: Path) -> bool:
    if not target.exists():
        return False
    try:
        target_ns = _mtime_ns(target)
    except OSError:
        return False
    source_ns = _latest_mtime_ns(sources)
    if source_ns is None:
        return False
    # Allow for minor clock skew (≤5ms) to avoid unnecessary re-imports on equal times.
    tolerance = 5_000_000
    if target_ns + tolerance < source_ns:
        return False
    return True


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
    collapse_thresholds: Optional[Dict[str, int]] = None,
    base_dir: Optional[Path] = None,
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
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
    jobs: int = 1,
) -> LocalSyncResult:
    if registrar is None:
        state_dir = output_dir / ".polylogue-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        registrar = create_default_registrar(database_path=state_dir / "polylogue.db")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    written: List[ImportResult] = []
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    wanted: set[str] = set()
    importer_kwargs = importer_kwargs or {}
    importer_kwargs = dict(importer_kwargs)
    importer_kwargs.setdefault("registrar", registrar)
    importer_kwargs.setdefault("attachment_ocr", attachment_ocr)
    importer_kwargs.setdefault("sanitize_html", sanitize_html)
    if meta:
        importer_kwargs.setdefault("meta", dict(meta))
    attachments_total = 0
    attachment_bytes_total = 0
    tokens_total = 0
    words_total = 0

    diff_total = 0
    failures: List[Dict[str, str]] = []
    session_list: List[Path]
    if isinstance(sessions, (list, tuple)):
        session_list = [Path(p) for p in sessions]
    else:
        session_list = [Path(p) for p in sessions]
    ignore_patterns: List[str] = []
    if base_dir:
        ignore_patterns = _load_ignore_patterns(base_dir.expanduser())
    filtered_sessions: List[Path] = []
    ignored = 0
    for path in sorted(session_list, key=path_order_key):
        if base_dir and _is_ignored(path, base_dir.expanduser(), ignore_patterns):
            ignored += 1
            continue
        filtered_sessions.append(path)
    if ignored and ui:
        ui.console.print(f"[yellow]Skipped {ignored} session(s) via .polylogueignore")
    iterable: Iterable[Path] = filtered_sessions

    session_list = list(iterable)
    progress_ctx = ui.progress(f"Syncing {provider} sessions", total=len(session_list)) if ui else _NoopProgress()

    output_root = output_dir.resolve()
    jobs = max(1, int(jobs or 1))

    last_progress_time = 0.0

    def _maybe_plain_progress(done: int, total: int) -> None:
        nonlocal last_progress_time
        if not ui or not ui.plain or total <= 0:
            return
        now = time.perf_counter()
        if done != total and done != 1 and (done % 25) != 0 and (now - last_progress_time) < 10.0:
            return
        last_progress_time = now
        elapsed = now - start_time
        rate = (done / elapsed) if elapsed > 0 else 0.0
        eta = ((total - done) / rate) if rate > 0 else None
        pct = (done / total) * 100.0
        ui.console.print(
            f"[dim]{provider} progress: {done}/{total} ({pct:.1f}%) elapsed={format_duration(elapsed)} eta={format_duration(eta)}[/dim]"
        )

    def _process_single(session_path: Path) -> Dict[str, object]:
        entry: Dict[str, object] = {
            "session_path": session_path,
            "skipped": False,
            "skip_reason": None,
            "result": None,
            "prune_slug": None,
            "diff": False,
            "error": None,
        }
        if not session_path.is_file():
            entry["skipped"] = True
            entry["skip_reason"] = "missing"
            return entry

        conversation_id = str(session_path)
        session_id: Optional[str] = None
        if provider == "claude-code":
            session_id = extract_claude_code_session_id(session_path)
            if session_id:
                conversation_id = session_id
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

        source_paths = _resolve_source_paths(provider, session_path, state_entry, session_id)

        slug_for_prune = slug_hint
        try:
            rel = md_path.parent.resolve().relative_to(output_root)
            if rel.parts:
                slug_for_prune = rel.parts[0]
        except ValueError:
            pass

        entry["prune_slug"] = slug_for_prune

        stored_hash = None
        if isinstance(state_entry, dict):
            stored_hash = state_entry.get("contentHash")
        existing_dirty = False
        if stored_hash and md_path.exists():
            try:
                from .document_store import read_existing_document

                existing_doc = read_existing_document(md_path)
                if existing_doc and existing_doc.content_hash != stored_hash:
                    existing_dirty = True
            except Exception:
                existing_dirty = False

        if not force and not existing_dirty and provider != "claude-code":
            try:
                from .importers.raw_storage import compute_hash
                from .db import get_raw_import_by_conversation, open_connection

                current_hash = compute_hash(session_path.read_bytes())
                with open_connection(registrar.database.resolve_path()) as conn:
                    raw_row = get_raw_import_by_conversation(conn, provider, conversation_id)
                if raw_row and raw_row["hash"] == current_hash:
                    entry["skipped"] = True
                    entry["skip_reason"] = "up-to-date"
                    return entry
            except Exception:
                pass

        if not force and _is_up_to_date_multi(source_paths, md_path):
            entry["skipped"] = True
            entry["skip_reason"] = "up-to-date"
            return entry

        diff_tracker = DiffTracker(md_path, diff)
        try:
            result = import_fn(
                str(session_path),
                output_dir=output_dir,
                collapse_threshold=collapse_threshold,
                collapse_thresholds=collapse_thresholds,
                html=html,
                html_theme=html_theme,
                force=force,
                **importer_kwargs,
            )
        except Exception as exc:
            diff_tracker.cleanup()
            entry["error"] = str(exc)
            return entry
        if result.skipped:
            diff_tracker.cleanup()
            entry["skipped"] = True
            entry["skip_reason"] = result.skip_reason or "skipped"
            return entry

        result.diff_path = diff_tracker.finalize(result.markdown_path)
        entry["diff"] = bool(result.diff_path)

        try:
            if provider == "claude-code":
                state_after = registrar.get_state(provider, conversation_id)
                source_paths = _resolve_source_paths(provider, session_path, state_after, session_id)
            session_ns = _latest_mtime_ns(source_paths) or _mtime_ns(session_path)
            result.markdown_path.parent.mkdir(parents=True, exist_ok=True)
            os.utime(result.markdown_path, ns=(session_ns, session_ns))
            if result.html_path:
                os.utime(result.html_path, ns=(session_ns, session_ns))
        except OSError:
            pass

        entry["result"] = result
        return entry

    pipeline = Pipeline([_LocalSessionStage(_process_single)])

    with progress_ctx as tracker:
        if jobs <= 1:
            done = 0
            for session_path in session_list:
                ctx = PipelineContext(env=None, options=None, data={"session_path": session_path})
                try:
                    pipeline.run(ctx)
                except Exception as exc:
                    failures.append({"path": str(session_path), "error": str(exc)})
                    tracker.advance()
                    done += 1
                    _maybe_plain_progress(done, len(session_list))
                    continue
                session_output = ctx.get("session_result", {})
                prune_slug = session_output.get("prune_slug")
                if isinstance(prune_slug, str):
                    wanted.add(prune_slug)

                if session_output.get("skipped"):
                    reason = session_output.get("skip_reason") or "skipped"
                    skip_reasons[reason] += 1
                    skipped += 1
                    tracker.advance()
                    done += 1
                    _maybe_plain_progress(done, len(session_list))
                    continue

                result = session_output.get("result")
                if not isinstance(result, ImportResult):
                    error = session_output.get("error")
                    if isinstance(error, str) and error:
                        failures.append({"path": str(session_path), "error": error})
                    tracker.advance()
                    done += 1
                    _maybe_plain_progress(done, len(session_list))
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
                done += 1
                _maybe_plain_progress(done, len(session_list))
        else:
            results_by_index: List[Dict[str, object]] = [{} for _ in session_list]
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                future_map = {executor.submit(_process_single, session_path): idx for idx, session_path in enumerate(session_list)}
                done = 0
                for future in as_completed(future_map):
                    idx = future_map[future]
                    session_path = session_list[idx]
                    try:
                        results_by_index[idx] = future.result()
                    except Exception as exc:
                        failures.append({"path": str(session_path), "error": str(exc)})
                        results_by_index[idx] = {"session_path": session_path, "error": str(exc)}
                    tracker.advance()
                    done += 1
                    _maybe_plain_progress(done, len(session_list))

            for idx, session_output in enumerate(results_by_index):
                prune_slug = session_output.get("prune_slug")
                if isinstance(prune_slug, str):
                    wanted.add(prune_slug)

                if session_output.get("skipped"):
                    reason = session_output.get("skip_reason") or "skipped"
                    skip_reasons[reason] += 1
                    skipped += 1
                    continue

                result = session_output.get("result")
                if not isinstance(result, ImportResult):
                    error = session_output.get("error")
                    if isinstance(error, str) and error:
                        failures.append({"path": str(session_list[idx]), "error": error})
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
        ignored=ignored,
        attachments=attachments_total,
        attachment_bytes=attachment_bytes_total,
        tokens=tokens_total,
        words=words_total,
        diffs=diff_total,
        duration=duration,
        failures=len(failures),
        failed=failures,
        skip_reasons=dict(skip_reasons),
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
    ignore_patterns = _load_ignore_patterns(base)
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
        if _is_ignored(cand, base, ignore_patterns):
            continue
        results.append(cand)
    return results


def _load_ignore_patterns(base_dir: Path) -> List[str]:
    path = base_dir / ".polylogueignore"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    patterns: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns


def _is_ignored(path: Path, base_dir: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    try:
        rel = path.relative_to(base_dir)
        candidate = rel.as_posix()
    except ValueError:
        candidate = path.name
    return any(fnmatch.fnmatch(candidate, pat) for pat in patterns)


def _normalize_export_target(path: Path) -> Optional[Path]:
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return None
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
    collapse_thresholds: Optional[Dict[str, int]] = None,
    base_dir: Optional[Path] = None,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    provider: str,
    import_fn: Callable[..., Sequence[ImportResult] | ImportResult | None],
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
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
    failures: List[Dict[str, str]] = []

    normalized = [_normalize_export_target(Path(p)) for p in bundles]
    filtered = [p for p in normalized if p is not None]
    if filtered:
        iterable: Iterable[Path] = sorted(filtered, key=path_order_key)
    else:
        iterable = sorted([Path(p) for p in bundles], key=path_order_key)

    ignore_patterns: List[str] = []
    if base_dir:
        ignore_patterns = _load_ignore_patterns(base_dir.expanduser())

    exports = [
        path for path in iterable if not (base_dir and _is_ignored(path, base_dir.expanduser(), ignore_patterns))
    ]
    progress_ctx = ui.progress(f"Processing {provider} exports", total=len(exports)) if ui else _NoopProgress()
    last_progress_time = 0.0

    def _maybe_plain_progress(done: int, total: int) -> None:
        nonlocal last_progress_time
        if not ui or not ui.plain or total <= 0:
            return
        now = time.perf_counter()
        if done != total and done != 1 and (done % 5) != 0 and (now - last_progress_time) < 10.0:
            return
        last_progress_time = now
        elapsed = now - start_time
        rate = (done / elapsed) if elapsed > 0 else 0.0
        eta = ((total - done) / rate) if rate > 0 else None
        pct = (done / total) * 100.0
        ui.console.print(
            f"[dim]{provider} progress: {done}/{total} ({pct:.1f}%) elapsed={format_duration(elapsed)} eta={format_duration(eta)}[/dim]"
        )

    with progress_ctx as tracker:
        for idx, export_path in enumerate(exports, start=1):
            if not export_path.exists():
                tracker.advance()
                _maybe_plain_progress(idx, len(exports))
                continue
            bundle_hash = None
            bundle_provider = f"{provider}-export"
            export_id = str(export_path)
            if not force and not prune:
                try:
                    from .importers.raw_storage import compute_hash

                    target = export_path
                    if export_path.is_dir():
                        conv_file = export_path / "conversations.json"
                        if conv_file.exists() and conv_file.is_file():
                            target = conv_file
                    bundle_hash = compute_hash(target.read_bytes())
                    state = registrar.get_state(bundle_provider, export_id)
                    if isinstance(state, dict) and state.get("bundleHash") == bundle_hash:
                        skipped_exports += 1
                        tracker.advance()
                        _maybe_plain_progress(idx, len(exports))
                        continue
                except Exception:
                    bundle_hash = None

            try:
                results_raw = import_fn(
                    export_path=export_path,
                    output_dir=output_dir,
                    collapse_threshold=collapse_threshold,
                    collapse_thresholds=collapse_thresholds,
                    html=html,
                    html_theme=html_theme,
                    selected_ids=None,
                    force=force,
                    registrar=registrar,
                    attachment_ocr=attachment_ocr,
                    sanitize_html=sanitize_html,
                    meta=dict(meta) if meta else None,
                )
            except Exception as exc:
                failures.append({"path": str(export_path), "error": str(exc)})
                tracker.advance()
                _maybe_plain_progress(idx, len(exports))
                continue
            if bundle_hash:
                from .util import current_utc_timestamp

                registrar.state_repo.upsert(
                    bundle_provider,
                    export_id,
                    {
                        "bundleHash": bundle_hash,
                        "bundlePath": export_id,
                        "lastImported": current_utc_timestamp(),
                    },
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
            _maybe_plain_progress(idx, len(exports))

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
        failures=len(failures),
        failed=failures,
    )


def sync_codex_sessions(
    *,
    base_dir: Path = CODEX_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
    jobs: int = 1,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    if sessions is None:
        sessions = _list_codex_paths(base_dir)
    return _sync_sessions(
        sessions,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        collapse_thresholds=collapse_thresholds,
        base_dir=base_dir,
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
        attachment_ocr=attachment_ocr,
        sanitize_html=sanitize_html,
        meta=meta,
        jobs=jobs,
    )


def sync_claude_code_sessions(
    *,
    base_dir: Path = CLAUDE_CODE_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
    jobs: int = 1,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    if sessions is None:
        sessions = base_dir.rglob("*.jsonl")
    return _sync_sessions(
        sessions,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        collapse_thresholds=collapse_thresholds,
        base_dir=base_dir,
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
        attachment_ocr=attachment_ocr,
        sanitize_html=sanitize_html,
        meta=meta,
        jobs=jobs,
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
    return _discover_export_targets(base_dir, provider="chatgpt")


def _list_claude_exports(base_dir: Path) -> List[Path]:
    return _discover_export_targets(base_dir, provider="claude")


def sync_chatgpt_exports(
    *,
    base_dir: Path = CHATGPT_EXPORTS_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    invalid_inputs: List[Path] = []
    targets: List[Path] = []
    if sessions is not None:
        for raw in sessions:
            normalized = _normalize_export_target(Path(raw))
            if normalized is None:
                invalid_inputs.append(Path(raw))
            else:
                targets.append(normalized)
        if invalid_inputs:
            bad_list = ", ".join(str(p) for p in invalid_inputs)
            raise ValueError(f"Invalid ChatGPT export path(s): {bad_list}")
        if not targets:
            raise ValueError("No valid ChatGPT exports found for the provided --session paths")
    else:
        targets = _discover_export_targets(base_dir, provider="chatgpt")
    return _sync_export_bundles(
        targets,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        collapse_thresholds=collapse_thresholds,
        base_dir=base_dir,
        html=html,
        html_theme=html_theme,
        force=force,
        prune=prune,
        provider="chatgpt",
        import_fn=import_chatgpt_export,
        registrar=registrar,
        ui=ui,
        attachment_ocr=attachment_ocr,
        sanitize_html=sanitize_html,
        meta=meta,
    )


def sync_claude_exports(
    *,
    base_dir: Path = CLAUDE_EXPORTS_DEFAULT,
    output_dir: Path,
    collapse_threshold: int,
    collapse_thresholds: Optional[Dict[str, int]] = None,
    html: bool,
    html_theme: str,
    force: bool,
    prune: bool,
    diff: bool = False,
    sessions: Optional[Iterable[Path]] = None,
    registrar: Optional[ConversationRegistrar] = None,
    ui: Optional[UI] = None,
    attachment_ocr: bool = True,
    sanitize_html: bool = False,
    meta: Optional[Dict[str, str]] = None,
) -> LocalSyncResult:
    base_dir = base_dir.expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    invalid_inputs: List[Path] = []
    targets: List[Path] = []
    if sessions is not None:
        for raw in sessions:
            normalized = _normalize_export_target(Path(raw))
            if normalized is None:
                invalid_inputs.append(Path(raw))
            else:
                targets.append(normalized)
        if invalid_inputs:
            bad_list = ", ".join(str(p) for p in invalid_inputs)
            raise ValueError(f"Invalid Claude export path(s): {bad_list}")
        if not targets:
            raise ValueError("No valid Claude exports found for the provided --session paths")
    else:
        targets = _discover_export_targets(base_dir, provider="claude")
    return _sync_export_bundles(
        targets,
        output_dir=output_dir,
        collapse_threshold=collapse_threshold,
        collapse_thresholds=collapse_thresholds,
        base_dir=base_dir,
        html=html,
        html_theme=html_theme,
        force=force,
        prune=prune,
        provider="claude",
        import_fn=import_claude_export,
        registrar=registrar,
        ui=ui,
        attachment_ocr=attachment_ocr,
        sanitize_html=sanitize_html,
        meta=meta,
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
        supports_jobs=True,
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
        supports_jobs=True,
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
