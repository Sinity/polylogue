from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from .archive import Archive
from .branch_explorer import list_branch_conversations
from .cli_common import compute_prune_paths, filter_chats
from .drive import snapshot_drive_metrics
from .drive_client import (
    DEFAULT_CREDENTIALS,
    DEFAULT_FOLDER_NAME,
    DEFAULT_TOKEN,
    DriveClient,
)
from .models import validate_chunks
from .options import (
    ListOptions,
    ListResult,
    RenderFile,
    RenderOptions,
    RenderResult,
    StatusResult,
    SyncItem,
    SyncOptions,
    SyncResult,
    BranchExploreOptions,
    BranchExploreResult,
    SearchOptions,
    SearchResult,
)
from .config import CONFIG, Config
from .render import MarkdownDocument, build_markdown_from_chunks
from .conversation import process_conversation
from .validation import SchemaError, ensure_gemini_payload
from .settings import Settings, ensure_settings_defaults
from .services.conversation_registrar import ConversationRegistrar
from .services.conversation_service import ConversationService, create_conversation_service
from .ui import UI
from .util import (
    DiffTracker,
    RunAccumulator,
    add_run,
    format_duration,
    load_runs,
    parse_rfc3339_to_epoch,
    sanitize_filename,
)
from .search import execute_search
from .pipeline import (
    AttachmentDownloadError,
    ChatContext,
    build_message_records_from_chunks,
    prepare_render_assets,
)
from .importers.base import ImportResult
from .pipeline_runner import Pipeline, PipelineContext
from .persistence.database import ConversationDatabase
from .persistence.state import ConversationStateRepository
from .providers import DriveProviderSession, ProviderRegistry
from .repository import ConversationRepository


PROVIDER_ALIASES = {
    "render": "render",
    "sync drive": "drive",
    "sync codex": "codex",
    "sync claude-code": "claude-code",
    "polylogue-sync-drive": "drive",
    "polylogue-sync-codex": "codex",
    "polylogue-sync-claude-code": "claude-code",
}


def _provider_from_cmd(cmd: str) -> str:
    if not cmd:
        return "unknown"
    key = cmd.strip().lower().replace("_", "-")
    if key in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[key]
    if key.startswith("sync "):
        return key.split(" ", 1)[1]
    if key.startswith("polylogue-sync-"):
        return key[len("polylogue-sync-") :]
    if key.startswith("codex"):
        return "codex"
    if key.startswith("claude-code"):
        return "claude-code"
    if key.startswith("drive"):
        return "drive"
    return key


@dataclass
class CommandEnv:
    ui: UI
    drive: Optional[DriveClient] = None
    config: Config = field(default_factory=lambda: CONFIG)
    repository: ConversationRepository = field(default_factory=ConversationRepository)
    settings: Settings = field(default_factory=Settings)
    prefs: Dict[str, Dict[str, str]] = field(default_factory=dict)
    state_repo: ConversationStateRepository = field(default_factory=ConversationStateRepository)
    database: ConversationDatabase = field(default_factory=ConversationDatabase)
    archive: Archive = field(default_factory=lambda: Archive(CONFIG))
    providers: ProviderRegistry = field(default_factory=ProviderRegistry)
    drive_constructor: Callable[..., DriveClient] = field(init=False)
    registrar: ConversationRegistrar = field(init=False)
    conversations: ConversationService = field(init=False)

    def __post_init__(self) -> None:
        _ensure_ui_contract(self.ui)
        ensure_settings_defaults(self.settings)
        self.registrar = ConversationRegistrar(
            state_repo=self.state_repo,
            database=self.database,
            archive=self.archive,
        )
        self.conversations = create_conversation_service(self.registrar)
        self.drive_constructor = DriveClient


class RenderReadStage:
    def run(self, context: PipelineContext) -> None:
        source: Path = context.get("source_path")
        env: CommandEnv = context.env
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            env.ui.console.print(f"[yellow]Skipping {source.name}: {exc}")
            context.abort()
            return
        context.set("raw_json", payload)


class RenderNormalizeStage:
    def run(self, context: PipelineContext) -> None:
        raw = context.get("raw_json")
        env: CommandEnv = context.env
        source: Path = context.get("source_path")
        try:
            chunks_raw = ensure_gemini_payload(raw, source=source.name)
        except SchemaError as exc:
            env.ui.console.print(f"[red]{exc}")
            context.abort()
            return
        chunks = validate_chunks(chunks_raw)
        slug = sanitize_filename(source.stem)
        chat_context = _context_from_local(raw, source.stem)
        conversation_id = chat_context.chat_id or slug
        context.set("chunks", chunks)
        context.set("slug", slug)
        context.set("chat_context", chat_context)
        context.set("conversation_id", conversation_id)
        context.set("provider", "render")


class RenderDocumentStage:
    def run(self, context: PipelineContext) -> None:
        env: CommandEnv = context.env
        options: RenderOptions = context.options
        chunks = context.get("chunks")
        chat_context: ChatContext = context.get("chat_context")
        slug: str = context.get("slug")
        md_path = (options.output_dir / slug) / "conversation.md"
        download_att = bool(getattr(options, "download_attachments", False))
        attachment_failures: Optional[List[Dict[str, str]]] = None
        try:
            per_chunk_links, attachments = prepare_render_assets(
                chunks,
                md_path=md_path,
                download_attachments=download_att,
                drive=env.drive if download_att else None,
                force=getattr(options, "force", False),
                dry_run=getattr(options, "dry_run", False),
            )
        except AttachmentDownloadError as exc:
            per_chunk_links = exc.per_index_links
            attachments = exc.attachments
            attachment_failures = exc.failed_items
            context.set("attachment_failures", list(attachment_failures))
            env.ui.console.print(
                f"[yellow]Warning: failed to download {len(exc.failed_ids)} attachment(s) "
                f"for {chat_context.title!r}: {', '.join(exc.failed_ids)}[/yellow]"
            )
        context.set("per_chunk_links", per_chunk_links)
        context.set("attachments", attachments)
        context.set("markdown_path", md_path)

        if options.dry_run:
            doc = build_markdown_from_chunks(
                chunks,
                per_chunk_links,
                chat_context.title,
                chat_context.chat_id,
                chat_context.modified_time,
                chat_context.created_time,
                run_settings=chat_context.run_settings,
                citations=chat_context.citations,
                source_mime=chat_context.source_mime,
                collapse_threshold=options.collapse_threshold,
                collapse_thresholds=getattr(options, "collapse_thresholds", None),
                attachments=attachments,
            )
            context.set("document", doc)


class RenderPersistStage:
    def run(self, context: PipelineContext) -> None:
        env: CommandEnv = context.env
        options: RenderOptions = context.options
        chat_context: ChatContext = context.get("chat_context")
        slug: str = context.get("slug")
        md_path: Path = context.get("markdown_path")
        conversation_id: str = context.get("conversation_id")
        provider = context.get("provider", "render")
        extra_state = context.get("extra_state", {}) or {}
        extra_yaml = context.get("extra_yaml", {}) or {}
        per_chunk_links = context.get("per_chunk_links") or {}
        attachments = context.get("attachments") or []
        chunks = context.get("chunks") or []
        source_path = context.get("source_path")

        message_records = build_message_records_from_chunks(
            chunks,
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            per_chunk_links=per_chunk_links,
        )
        if not message_records:
            context.set("import_result", None)
            return

        if options.dry_run:
            doc: Optional[MarkdownDocument] = context.get("document")
            if doc is None:
                doc = build_markdown_from_chunks(
                    chunks,
                    per_chunk_links,
                    chat_context.title,
                    chat_context.chat_id,
                    chat_context.modified_time,
                    chat_context.created_time,
                    run_settings=chat_context.run_settings,
                    citations=chat_context.citations,
                    source_mime=chat_context.source_mime,
                    collapse_threshold=options.collapse_threshold,
                    collapse_thresholds=getattr(options, "collapse_thresholds", None),
                    attachments=attachments,
                )
            import_result = ImportResult(
                markdown_path=md_path,
                html_path=None,
                attachments_dir=md_path.parent / "attachments",
                document=doc,
                slug=slug,
            )
            context.set("import_result", import_result)
            return

        diff_tracker = DiffTracker(md_path, bool(getattr(options, "diff", False)))
        extra_yaml_payload = dict(extra_yaml)
        if source_path:
            extra_yaml_payload.setdefault("sourceFile", str(source_path))
        extra_yaml_payload.setdefault("sourcePlatform", provider)

        extra_state_payload: Dict[str, Any] = dict(extra_state)
        if source_path:
            extra_state_payload.setdefault("sourceFile", str(source_path))
        if chat_context.source_mime:
            extra_state_payload.setdefault("sourceMimeType", chat_context.source_mime)

        import_result = process_conversation(
            provider=provider,
            conversation_id=conversation_id,
            slug=slug,
            title=chat_context.title,
            message_records=message_records,
            attachments=attachments,
            canonical_leaf_id=message_records[-1].message_id if message_records else None,
            collapse_threshold=options.collapse_threshold,
            collapse_thresholds=getattr(options, "collapse_thresholds", None),
            html=getattr(options, "html", False),
            html_theme=getattr(options, "html_theme", "light"),
            output_dir=options.output_dir,
            extra_yaml=extra_yaml_payload,
            extra_state=extra_state_payload,
            source_file_id=conversation_id,
            modified_time=chat_context.modified_time,
            created_time=chat_context.created_time,
            run_settings=chat_context.run_settings,
            source_mime=chat_context.source_mime,
            source_size=None,
            attachment_policy=None,
            force=getattr(options, "force", False),
            attachment_ocr=getattr(options, "attachment_ocr", True),
            sanitize_html=getattr(options, "sanitize_html", False),
            registrar=env.registrar,
            citations=chat_context.citations,
        )
        import_result.diff_path = diff_tracker.finalize(import_result.markdown_path)
        context.set("import_result", import_result)
        context.set("document", import_result.document)


class DriveDownloadStage:
    def run(self, context: PipelineContext) -> None:
        env: CommandEnv = context.env
        options: SyncOptions = context.options  # type: ignore[assignment]
        meta = context.get("metadata")
        file_id = meta.get("id") if isinstance(meta, dict) else None
        data_bytes = env.drive.download_chat_bytes(file_id) if file_id else None
        if data_bytes is None:
            env.ui.console.print(f"[red]Failed to download {meta.get('name') if isinstance(meta, dict) else file_id}")
            context.abort()
            return
        try:
            payload = json.loads(data_bytes.decode("utf-8", errors="replace"))
        except Exception:
            env.ui.console.print(f"[yellow]Invalid JSON: {meta.get('name') if isinstance(meta, dict) else file_id}")
            context.abort()
            return
        context.set("raw_json", payload)
        context.set("file_id", file_id)
        context.set("chat_name", meta.get("name") if isinstance(meta, dict) else None)


class DriveNormalizeStage:
    def run(self, context: PipelineContext) -> None:
        env: CommandEnv = context.env
        options: SyncOptions = context.options  # type: ignore[assignment]
        meta = context.get("metadata")
        raw = context.get("raw_json")
        if not isinstance(raw, dict) or not isinstance(meta, dict):
            context.abort()
            return
        file_id = context.get("file_id")
        name_safe = sanitize_filename(meta.get("name") or file_id or "chat")
        try:
            chunks_raw = ensure_gemini_payload(raw, source=meta.get("name") or file_id or name_safe)
        except SchemaError as exc:
            env.ui.console.print(f"[red]{exc}")
            context.abort()
            return
        chunks = validate_chunks(chunks_raw)
        chat_context = _context_from_drive(meta, raw, name_safe)
        conversation_id = file_id or name_safe
        context.set("chunks", chunks)
        context.set("slug", name_safe)
        context.set("chat_context", chat_context)
        context.set("conversation_id", conversation_id)
        context.set("provider", "drive-sync")
        extra_state = dict(context.get("extra_state") or {})
        if getattr(options, "meta", None):
            extra_state.setdefault("cliMeta", dict(getattr(options, "meta") or {}))
        extra_state.update(
            {
                "driveFileId": file_id,
                "driveFolder": options.folder_name or DEFAULT_FOLDER_NAME,
            }
        )
        context.set("extra_state", extra_state)
        context.set("source_path", options.output_dir / name_safe / "conversation.json")


@dataclass
class RunSummaryEntry:
    command: str
    provider: str
    count: int = 0
    attachments: int = 0
    attachment_bytes: int = 0
    tokens: int = 0
    words: int = 0
    skipped: int = 0
    pruned: int = 0
    diffs: int = 0
    duration: float = 0.0
    last: Optional[str] = None
    last_out: Optional[str] = None
    last_count: Optional[int] = None
    retries: int = 0
    failures: int = 0
    last_error: Optional[str] = None

    def update_from_run(self, record: Dict[str, Any]) -> None:
        self.count += int(record.get("count", 0) or 0)
        self.attachments += int(record.get("attachments", 0) or 0)
        self.attachment_bytes += int(record.get("attachmentBytes", 0) or 0)
        self.tokens += int(record.get("tokens", 0) or 0)
        self.words += int(record.get("words", 0) or 0)
        self.skipped += int(record.get("skipped", 0) or 0)
        self.pruned += int(record.get("pruned", 0) or 0)
        self.diffs += int(record.get("diffs", 0) or 0)
        self.duration += float(record.get("duration", 0.0) or 0.0)
        self.retries += int(record.get("driveRetries", record.get("retries", 0)) or 0)
        self.failures += int(record.get("driveFailures", record.get("failures", 0)) or 0)
        err = record.get("driveLastError") or record.get("lastError")
        if isinstance(err, str) and err.strip():
            self.last_error = err.strip()
        ts = record.get("timestamp")
        if isinstance(ts, str) and (self.last is None or ts > self.last):
            self.last = ts
            self.last_out = record.get("out")
            self.last_count = record.get("count")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "attachments": self.attachments,
            "attachmentBytes": self.attachment_bytes,
            "tokens": self.tokens,
            "words": self.words,
            "skipped": self.skipped,
            "pruned": self.pruned,
            "diffs": self.diffs,
            "duration": self.duration,
            "last": self.last,
            "last_out": self.last_out,
            "last_count": self.last_count,
            "provider": self.provider,
            "retries": self.retries,
            "failures": self.failures,
            "last_error": self.last_error,
        }


@dataclass
class ProviderSummaryEntry:
    provider: str
    commands: set[str] = field(default_factory=set)
    count: int = 0
    attachments: int = 0
    attachment_bytes: int = 0
    tokens: int = 0
    words: int = 0
    skipped: int = 0
    pruned: int = 0
    diffs: int = 0
    duration: float = 0.0
    last: Optional[str] = None
    last_out: Optional[str] = None
    last_count: Optional[int] = None
    retries: int = 0
    failures: int = 0
    last_error: Optional[str] = None

    def merge(self, run: RunSummaryEntry) -> None:
        self.commands.add(run.command)
        self.count += run.count
        self.attachments += run.attachments
        self.attachment_bytes += run.attachment_bytes
        self.tokens += run.tokens
        self.words += run.words
        self.skipped += run.skipped
        self.pruned += run.pruned
        self.diffs += run.diffs
        self.duration += run.duration
        self.retries += run.retries
        self.failures += run.failures
        if run.last and (self.last is None or run.last > self.last):
            self.last = run.last
            self.last_out = run.last_out
            self.last_count = run.last_count
        if run.last_error:
            self.last_error = run.last_error

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "commands": sorted(self.commands),
            "count": self.count,
            "attachments": self.attachments,
            "attachmentBytes": self.attachment_bytes,
            "tokens": self.tokens,
            "words": self.words,
            "skipped": self.skipped,
            "pruned": self.pruned,
            "diffs": self.diffs,
            "duration": self.duration,
            "last": self.last,
            "last_out": self.last_out,
            "last_count": self.last_count,
            "retries": self.retries,
            "failures": self.failures,
            "last_error": self.last_error,
        }
        return data


def _ensure_drive(env: CommandEnv) -> DriveClient:
    if env.drive is None:
        session = DriveProviderSession(env.ui, client_factory=env.drive_constructor, **_drive_client_kwargs(env))
        env.providers.register(session)
        env.drive = session
    return env.drive


def _drive_client_kwargs(
    env: CommandEnv,
    *,
    retries: Optional[int] = None,
    retry_base: Optional[float] = None,
) -> Dict[str, object]:
    drive_cfg = getattr(env, "config", None).drive if hasattr(env, "config") else None
    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    kwargs: Dict[str, object] = {}
    if credential_env:
        kwargs["credentials_path"] = Path(credential_env).expanduser()
    elif drive_cfg and getattr(drive_cfg, "credentials_path", None):
        kwargs["credentials_path"] = drive_cfg.credentials_path
    if token_env:
        kwargs["token_path"] = Path(token_env).expanduser()
    elif drive_cfg and getattr(drive_cfg, "token_path", None):
        kwargs["token_path"] = drive_cfg.token_path
    if drive_cfg:
        if getattr(drive_cfg, "retries", None) is not None:
            kwargs["retries"] = drive_cfg.retries
        if getattr(drive_cfg, "retry_base", None) is not None:
            kwargs["retry_base"] = drive_cfg.retry_base
    if retries is not None:
        kwargs["retries"] = retries
    if retry_base is not None:
        kwargs["retry_base"] = retry_base
    return kwargs


def build_drive_client(
    env: CommandEnv,
    *,
    retries: Optional[int] = None,
    retry_base: Optional[float] = None,
) -> DriveClient:
    kwargs = _drive_client_kwargs(env, retries=retries, retry_base=retry_base)
    return env.drive_constructor(env.ui, **kwargs)




def branches_command(options: BranchExploreOptions) -> BranchExploreResult:
    conversations = list_branch_conversations(
        provider=options.provider,
        slug=options.slug,
        conversation_id=options.conversation_id,
        min_branches=options.min_branches,
    )
    return BranchExploreResult(conversations=conversations)


def search_command(options: SearchOptions, env: Optional[CommandEnv] = None) -> SearchResult:
    if env is not None:
        return execute_search(options, service=env.conversations)
    return execute_search(options, service=create_conversation_service())


def render_command(options: RenderOptions, env: CommandEnv) -> RenderResult:
    from .cli.path_policy import PathPolicy, resolve_path

    ui = env.ui
    output_dir = options.output_dir
    if not options.dry_run:
        resolved = resolve_path(output_dir, PathPolicy.create_ok(), ui)
        if not resolved:
            raise SystemExit(1)
        output_dir = resolved
    start_time = time.perf_counter()
    if options.download_attachments:
        _ensure_drive(env)

    pipeline = Pipeline(
        [
            RenderReadStage(),
            RenderNormalizeStage(),
            RenderDocumentStage(),
            RenderPersistStage(),
        ]
    )

    render_files: List[RenderFile] = []
    failures: List[Dict[str, Any]] = []
    totals_acc = RunAccumulator()

    with ui.progress("Rendering files", total=len(options.inputs)) as tracker:
        for src in options.inputs:
            extra_state: Dict[str, object] = {}
            if getattr(options, "meta", None):
                extra_state["cliMeta"] = dict(getattr(options, "meta") or {})
            data: Dict[str, object] = {"source_path": src}
            if extra_state:
                data["extra_state"] = extra_state
            ctx = PipelineContext(env=env, options=options, data=data)
            try:
                pipeline.run(ctx)
            except Exception as exc:
                failures.append({"source": str(src), "error": str(exc)})
                tracker.advance()
                continue
            tracker.advance()

            if ctx.aborted:
                continue

            result: Optional[ImportResult] = ctx.get("import_result")
            if result is None:
                continue

            if result.skipped:
                totals_acc.increment("skipped")
                continue

            doc: Optional[MarkdownDocument] = result.document
            if doc is not None:
                totals_acc.add_stats(len(doc.attachments), doc.stats)
            if result.diff_path:
                totals_acc.increment("diffs")

            render_files.append(
                RenderFile(
                    output=result.markdown_path,
                    slug=result.slug,
                    attachments=len(doc.attachments) if doc else 0,
                    stats=doc.stats if doc else {},
                    html=result.html_path,
                    diff=result.diff_path,
                )
            )

    duration = time.perf_counter() - start_time
    totals = totals_acc.totals()
    run_payload = {
        "cmd": "render",
        "provider": "render",
        "count": len(render_files),
        "out": str(output_dir),
        "attachments": totals.get("attachments", 0),
        "attachmentBytes": totals.get("attachmentBytes", 0),
        "tokens": totals.get("totalTokensApprox", 0),
        "words": totals.get("totalWordsApprox", 0),
        "diffs": totals.get("diffs", 0),
        "duration": duration,
    }
    if failures:
        run_payload["failures"] = len(failures)
        run_payload["failedFiles"] = failures
    if getattr(options, "meta", None):
        run_payload["meta"] = dict(getattr(options, "meta") or {})
    if getattr(options, "sanitize_html", False):
        run_payload["redacted"] = True
    add_run(run_payload)
    result = RenderResult(
        count=len(render_files),
        output_dir=output_dir,
        files=render_files,
        total_stats=totals,
    )
    setattr(result, "failed_files", failures)
    return result


def _context_from_local(obj: Dict, fallback: str) -> ChatContext:
    return ChatContext(
        title=obj.get("title") or fallback,
        chat_id=obj.get("id"),
        modified_time=obj.get("modifiedTime"),
        created_time=obj.get("createTime") or obj.get("createdTime"),
        run_settings=obj.get("runSettings"),
        citations=obj.get("citations"),
        source_mime=obj.get("mimeType"),
    )


def _context_from_drive(meta: Dict, obj: Dict, fallback: str) -> ChatContext:
    return ChatContext(
        title=meta.get("name") or obj.get("title") or fallback,
        chat_id=meta.get("id") or obj.get("id"),
        modified_time=meta.get("modifiedTime") or obj.get("modifiedTime"),
        created_time=meta.get("createdTime")
        or obj.get("createTime")
        or obj.get("createdTime"),
        run_settings=obj.get("runSettings"),
        citations=obj.get("citations"),
        source_mime=meta.get("mimeType") or obj.get("mimeType"),
    )


def list_command(options: ListOptions, env: CommandEnv) -> ListResult:
    drive = _ensure_drive(env)
    chats = drive.list_chats(options.folder_name, options.folder_id)
    chats = filter_chats(chats, options.name_filter, options.since, options.until)
    folder_id = drive.resolve_folder_id(options.folder_name, options.folder_id)
    return ListResult(folder_name=options.folder_name or DEFAULT_FOLDER_NAME, folder_id=folder_id, files=chats)


def sync_command(options: SyncOptions, env: CommandEnv) -> SyncResult:
    from .cli.path_policy import PathPolicy, resolve_path

    drive = _ensure_drive(env)
    folder_id = options.folder_id or drive.resolve_folder_id(options.folder_name, options.folder_id)
    chats = options.prefetched_chats if options.prefetched_chats is not None else drive.list_chats(options.folder_name, folder_id)
    if options.prefetched_chats is None:
        chats = filter_chats(chats, options.name_filter, options.since, options.until)

    start_time = time.perf_counter()

    if options.selected_ids:
        ids = set(options.selected_ids)
        chats = [c for c in chats if c.get("id") in ids]

    if not chats:
        snapshot_drive_metrics(reset=True)
        return SyncResult(
            count=0,
            output_dir=options.output_dir,
            folder_name=options.folder_name,
            folder_id=folder_id,
            items=[],
            total_stats={
                "attachments": 0,
                "attachmentBytes": 0,
                "totalTokensApprox": 0,
                "totalWordsApprox": 0,
                "skipped": 0,
            },
        )

    if not options.dry_run:
        ui = env.ui
        resolved = resolve_path(options.output_dir, PathPolicy.create_ok(), ui)
        if not resolved:
            raise SystemExit(1)
        options.output_dir = resolved

    pipeline = Pipeline(
        [
            DriveDownloadStage(),
            DriveNormalizeStage(),
            RenderDocumentStage(),
            RenderPersistStage(),
        ]
    )

    items: List[SyncItem] = []
    wanted_slugs: set[str] = set()
    totals_acc = RunAccumulator()
    failures: List[Dict[str, Any]] = []
    attachment_failures: List[Dict[str, Any]] = []
    attachment_failures_total = 0
    description = f"Syncing {options.folder_name or 'Drive'} chats"
    last_progress_time = 0.0

    def _maybe_plain_progress(done: int, total: int) -> None:
        nonlocal last_progress_time
        if not env.ui.plain or total <= 0:
            return
        now = time.perf_counter()
        if done != total and done != 1 and (done % 25) != 0 and (now - last_progress_time) < 10.0:
            return
        last_progress_time = now
        elapsed = now - start_time
        rate = (done / elapsed) if elapsed > 0 else 0.0
        eta = ((total - done) / rate) if rate > 0 else None
        pct = (done / total) * 100.0
        env.ui.console.print(
            f"[dim]drive progress: {done}/{total} ({pct:.1f}%) elapsed={format_duration(elapsed)} eta={format_duration(eta)}[/dim]"
        )
    with env.ui.progress(description, total=len(chats)) as tracker:
        for idx, meta in enumerate(chats, start=1):
            extra_state: Dict[str, object] = {}
            if getattr(options, "meta", None):
                extra_state["cliMeta"] = dict(getattr(options, "meta") or {})
            data: Dict[str, object] = {"metadata": meta}
            if extra_state:
                data["extra_state"] = extra_state
            ctx = PipelineContext(env=env, options=options, data=data)
            try:
                pipeline.run(ctx)
            except Exception as exc:
                stage = None
                if ctx.history:
                    stage = ctx.history[-1].get("name")
                failures.append(
                    {
                        "id": meta.get("id"),
                        "name": meta.get("name"),
                        "error": str(exc),
                        "stage": stage,
                    }
                )
                tracker.advance()
                _maybe_plain_progress(idx, len(chats))
                continue
            if ctx.aborted:
                stage = None
                if ctx.history:
                    stage = ctx.history[-1].get("name")
                failures.append(
                    {
                        "id": meta.get("id"),
                        "name": meta.get("name"),
                        "error": "aborted",
                        "stage": stage,
                    }
                )
                tracker.advance()
                _maybe_plain_progress(idx, len(chats))
                continue

            result: Optional[ImportResult] = ctx.get("import_result")
            chat_context: ChatContext = ctx.get("chat_context")
            file_id = ctx.get("file_id")
            slug_value = ctx.get("slug")
            if isinstance(slug_value, str) and slug_value:
                wanted_slugs.add(slug_value)

            if result is None:
                tracker.advance()
                _maybe_plain_progress(idx, len(chats))
                continue

            if result.skipped:
                totals_acc.increment("skipped")
                tracker.advance()
                _maybe_plain_progress(idx, len(chats))
                continue

            if not options.dry_run and chat_context.modified_time:
                mtime = parse_rfc3339_to_epoch(chat_context.modified_time)
                if mtime is not None:
                    try:
                        os.utime(result.markdown_path, (mtime, mtime))
                    except Exception:
                        pass
                    if result.html_path:
                        try:
                            os.utime(result.html_path, (mtime, mtime))
                        except Exception:
                            pass

            doc: Optional[MarkdownDocument] = result.document
            att_failures = ctx.get("attachment_failures")
            if isinstance(att_failures, list) and att_failures:
                cleaned: List[Dict[str, str]] = []
                for entry in att_failures:
                    if isinstance(entry, dict):
                        att_id = str(entry.get("id") or "").strip()
                        if not att_id:
                            continue
                        item: Dict[str, str] = {"id": att_id}
                        filename = entry.get("filename")
                        if isinstance(filename, str) and filename.strip():
                            item["filename"] = filename.strip()
                        path_value = entry.get("path")
                        if isinstance(path_value, str) and path_value.strip():
                            item["path"] = path_value.strip()
                        cleaned.append(item)
                    elif isinstance(entry, str) and entry.strip():
                        cleaned.append({"id": entry.strip()})
                if cleaned:
                    attachment_failures.append(
                        {
                            "id": file_id or meta.get("id"),
                            "name": meta.get("name"),
                            "slug": result.slug,
                            "attachments": cleaned,
                        }
                    )
                    attachment_failures_total += len(cleaned)
            items.append(
                SyncItem(
                    id=file_id,
                    name=meta.get("name"),
                    output=result.markdown_path,
                    slug=result.slug,
                    attachments=len(doc.attachments) if doc else 0,
                    stats=doc.stats if doc else {},
                    html=result.html_path,
                    diff=result.diff_path,
                )
            )
            if result.diff_path:
                totals_acc.increment("diffs")
            if doc is not None:
                totals_acc.add_stats(len(doc.attachments), doc.stats)
            tracker.advance()
            _maybe_plain_progress(idx, len(chats))

    pruned_count = 0
    if options.prune:
        wanted = wanted_slugs if wanted_slugs else {item.slug for item in items}
        to_delete = compute_prune_paths(options.output_dir, wanted)
        if options.dry_run:
            env.ui.console.print(f"[yellow][dry-run] Would prune {len(to_delete)} path(s)")
            for path in to_delete:
                env.ui.console.print(f"  rm {'-r ' if path.is_dir() else ''}{path}")
        else:
            removed = 0
            for path in to_delete:
                try:
                    if path.is_dir():
                        import shutil

                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    removed += 1
                except Exception:
                    env.ui.console.print(f"[red]Failed to remove {path}")
            if removed:
                env.ui.console.print(f"Removed {removed} stale path(s)")
            pruned_count = removed

    duration = time.perf_counter() - start_time
    totals = totals_acc.totals()
    drive_stats = snapshot_drive_metrics(reset=True)
    run_payload = {
        "cmd": "sync drive",
        "provider": "drive",
        "count": len(items),
        "out": str(options.output_dir),
        "folder_name": options.folder_name,
        "folder_id": folder_id,
        "attachments": totals.get("attachments", 0),
        "attachmentBytes": totals.get("attachmentBytes", 0),
        "tokens": totals.get("totalTokensApprox", 0),
        "words": totals.get("totalWordsApprox", 0),
        "skipped": totals.get("skipped", 0),
        "pruned": pruned_count,
        "diffs": totals.get("diffs", 0),
        "duration": duration,
        "driveRequests": drive_stats.get("requests", 0),
        "driveRetries": drive_stats.get("retries", 0),
        "driveFailures": drive_stats.get("failures", 0),
        "driveLastError": drive_stats.get("lastError"),
    }
    if failures:
        run_payload["failures"] = len(failures)
        run_payload["failedChats"] = failures
    if attachment_failures:
        run_payload["attachmentFailures"] = attachment_failures_total
        run_payload["failedAttachments"] = attachment_failures
    if getattr(options, "meta", None):
        run_payload["meta"] = dict(getattr(options, "meta") or {})
    if getattr(options, "sanitize_html", False):
        run_payload["redacted"] = True
    add_run(run_payload)
    totals.setdefault("attachments", 0)
    totals.setdefault("skipped", 0)
    totals.setdefault("diffs", totals.get("diffs", 0))
    if attachment_failures_total:
        totals["attachmentFailures"] = attachment_failures_total
    totals["pruned"] = pruned_count
    result = SyncResult(
        count=len(items),
        output_dir=options.output_dir,
        folder_name=options.folder_name,
        folder_id=folder_id,
        items=items,
        total_stats=totals,
    )
    setattr(result, "failed_chats", failures)
    setattr(result, "failed_attachments", attachment_failures)
    return result


def status_command(env: CommandEnv, runs_limit: Optional[int] = 200, provider_filter: Optional[Set[str]] = None) -> StatusResult:
    drive_cfg = getattr(env, "config", None).drive if hasattr(env, "config") else None
    credential_env = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    token_env = os.environ.get("POLYLOGUE_TOKEN_PATH")
    credentials_path = (
        Path(credential_env).expanduser()
        if credential_env
        else (drive_cfg.credentials_path if drive_cfg else DEFAULT_CREDENTIALS)
    )
    token_path = (
        Path(token_env).expanduser()
        if token_env
        else (drive_cfg.token_path if drive_cfg else DEFAULT_TOKEN)
    )
    credentials_present = credentials_path.exists()
    token_present = token_path.exists()
    limit = runs_limit if runs_limit and runs_limit > 0 else None

    # When filtering by provider, load all runs so provider-specific history is not
    # accidentally truncated before filtering.
    if provider_filter:
        limit = None

    run_data = load_runs(limit=limit)

    def _matches_provider(entry: dict) -> bool:
        if not provider_filter:
            return True
        provider_value = (entry.get("provider") or _provider_from_cmd(entry.get("cmd") or "") or "").lower()
        return provider_value in provider_filter

    if provider_filter:
        run_data = [entry for entry in run_data if _matches_provider(entry)]

    if runs_limit and runs_limit > 0:
        run_data = run_data[-runs_limit:]
    recent_runs: List[dict] = run_data[-10:]
    run_summary_entries: Dict[str, RunSummaryEntry] = {}
    for entry in run_data:
        if not isinstance(entry, dict):
            continue
        cmd = entry.get("cmd") or "unknown"
        provider_hint = entry.get("provider") or _provider_from_cmd(cmd)
        summary = run_summary_entries.setdefault(
            cmd,
            RunSummaryEntry(command=cmd, provider=provider_hint),
        )
        if provider_hint and summary.provider != provider_hint:
            summary.provider = provider_hint
        summary.update_from_run(entry)
    provider_summary_entries: Dict[str, ProviderSummaryEntry] = {}
    for summary in run_summary_entries.values():
        provider = summary.provider or _provider_from_cmd(summary.command)
        entry = provider_summary_entries.setdefault(provider, ProviderSummaryEntry(provider=provider))
        entry.merge(summary)

    run_summary = {cmd: run_summary_entries[cmd].as_dict() for cmd in sorted(run_summary_entries)}
    provider_summary = {provider: provider_summary_entries[provider].as_dict() for provider in sorted(provider_summary_entries)}
    return StatusResult(
        credentials_present=credentials_present,
        token_present=token_present,
        credential_path=credentials_path,
        token_path=token_path,
        credential_env=credential_env,
        token_env=token_env,
        state_path=env.conversations.state_path,
        runs_path=env.database.resolve_path(),
        recent_runs=recent_runs,
        run_summary=run_summary,
        provider_summary=provider_summary,
        runs=run_data,
    )


def _ensure_ui_contract(ui: Any) -> None:
    console = getattr(ui, "console", None)

    if not hasattr(ui, "summary") or not callable(getattr(ui, "summary")):
        def _fallback_summary(title: str, lines: Iterable[str]) -> None:
            items = list(lines)
            header = title or "Summary"
            if console is not None and hasattr(console, "print"):
                console.print(header)
                for line in items:
                    console.print(f"  {line}")
            else:
                print(header)
                for line in items:
                    print(f"  {line}")

        setattr(ui, "summary", _fallback_summary)

    if not hasattr(ui, "progress") or not callable(getattr(ui, "progress")):
        setattr(ui, "progress", lambda *_args, **_kwargs: _FallbackProgress())


class _FallbackProgress:
    def __enter__(self):
        return self

    def advance(self, *_args, **_kwargs) -> None:
        return None

    def __exit__(self, *_exc) -> None:
        return None
