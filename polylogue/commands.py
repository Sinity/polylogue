from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .branch_explorer import list_branch_conversations
from .cli_common import compute_prune_paths, filter_chats
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
from .document_store import DocumentPersistenceResult
from .render import MarkdownDocument
from .settings import Settings
from .ui import UI
from .util import (
    RUNS_PATH,
    STATE_PATH,
    DiffTracker,
    RunAccumulator,
    add_run,
    parse_rfc3339_to_epoch,
    sanitize_filename,
)
from .search import execute_search
from .pipeline import ChatContext, build_document_from_chunks
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
    repository: ConversationRepository = field(default_factory=ConversationRepository)
    settings: Settings = field(default_factory=Settings)


@dataclass
class PersistOutcome:
    slug: str
    output_path: Path
    html_path: Optional[Path]
    diff_path: Optional[Path]
    skipped: bool
    persist_result: Optional[DocumentPersistenceResult]


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
        if run.last and (self.last is None or run.last > self.last):
            self.last = run.last
            self.last_out = run.last_out
            self.last_count = run.last_count

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
        }
        return data


def _ensure_drive(env: CommandEnv) -> DriveClient:
    if env.drive is None:
        env.drive = DriveClient(env.ui)
    return env.drive


def _persist_conversation(
    *,
    env: CommandEnv,
    options,
    doc: MarkdownDocument,
    context: ChatContext,
    output_dir: Path,
    slug_hint: str,
    conversation_id: str,
    provider: str,
    md_path: Path,
    extra_state: Optional[Dict[str, Any]] = None,
) -> PersistOutcome:
    if not options.dry_run:
        md_path.parent.mkdir(parents=True, exist_ok=True)
    diff_tracker = DiffTracker(md_path, bool(getattr(options, "diff", False) and not options.dry_run))
    if options.dry_run:
        diff_tracker.cleanup()
        return PersistOutcome(
            slug=slug_hint,
            output_path=md_path,
            html_path=None,
            diff_path=None,
            skipped=False,
            persist_result=None,
        )

    persist_result = env.repository.persist(
        provider=provider,
        conversation_id=conversation_id,
        title=context.title,
        document=doc,
        output_dir=output_dir,
        collapse_threshold=options.collapse_threshold,
        attachments=doc.attachments,
        updated_at=context.modified_time,
        created_at=context.created_time,
        html=getattr(options, "html", False),
        html_theme=getattr(options, "html_theme", "light"),
        attachment_policy=None,
        extra_state=extra_state,
        slug_hint=slug_hint,
        id_hint=slug_hint[:8] if slug_hint else None,
        force=getattr(options, "force", False),
    )

    if persist_result.skipped:
        diff_tracker.cleanup()
        return PersistOutcome(
            slug=persist_result.slug,
            output_path=persist_result.markdown_path,
            html_path=persist_result.html_path,
            diff_path=None,
            skipped=True,
            persist_result=persist_result,
        )

    diff_path = diff_tracker.finalize(persist_result.markdown_path)
    return PersistOutcome(
        slug=persist_result.slug,
        output_path=persist_result.markdown_path,
        html_path=persist_result.html_path,
        diff_path=diff_path,
        skipped=False,
        persist_result=persist_result,
    )


def branches_command(options: BranchExploreOptions) -> BranchExploreResult:
    conversations = list_branch_conversations(
        provider=options.provider,
        slug=options.slug,
        conversation_id=options.conversation_id,
        min_branches=options.min_branches,
    )
    return BranchExploreResult(conversations=conversations)


def search_command(options: SearchOptions) -> SearchResult:
    return execute_search(options)


def render_command(options: RenderOptions, env: CommandEnv) -> RenderResult:
    ui = env.ui
    output_dir = options.output_dir
    if not options.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    drive = _ensure_drive(env) if options.download_attachments else None

    render_files: List[RenderFile] = []
    totals_acc = RunAccumulator()
    for src in options.inputs:
        try:
            obj = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            ui.console.print(f"[yellow]Skipping {src.name}: {exc}")
            continue
        chunks_raw = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
        if not isinstance(chunks_raw, list):
            ui.console.print(f"[yellow]No chunks in {src.name}")
            continue
        chunks = validate_chunks(chunks_raw)
        slug = sanitize_filename(src.stem)
        md_path = (output_dir / slug) / "conversation.md"
        context = _context_from_local(obj, src.stem)
        conversation_id = context.chat_id or slug
        doc = build_document_from_chunks(
            chunks,
            context,
            md_path,
            collapse_threshold=options.collapse_threshold,
            download_attachments=options.download_attachments,
            drive=drive,
            force=options.force,
            dry_run=options.dry_run,
        )
        outcome = _persist_conversation(
            env=env,
            options=options,
            doc=doc,
            context=context,
            output_dir=output_dir,
            slug_hint=slug,
            conversation_id=conversation_id,
            provider="render",
            md_path=md_path,
            extra_state={
                "sourceFile": str(src),
                "sourceMimeType": context.source_mime,
            },
        )

        if outcome.skipped:
            totals_acc.increment("skipped")
            continue

        totals_acc.add_stats(len(doc.attachments), doc.stats)
        if outcome.diff_path:
            totals_acc.increment("diffs")

        render_files.append(
            RenderFile(
                output=outcome.output_path,
                slug=outcome.slug,
                attachments=len(doc.attachments),
                stats=doc.stats,
                html=outcome.html_path,
                diff=outcome.diff_path,
            )
        )

    duration = time.perf_counter() - start_time
    totals = totals_acc.totals()
    add_run(
        {
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
    )
    return RenderResult(
        count=len(render_files),
        output_dir=output_dir,
        files=render_files,
        total_stats=totals,
    )


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
    drive = env.drive or DriveClient(env.ui)
    env.drive = drive
    chats = drive.list_chats(options.folder_name, options.folder_id)
    chats = filter_chats(chats, options.name_filter, options.since, options.until)
    folder_id = drive.resolve_folder_id(options.folder_name, options.folder_id)
    return ListResult(folder_name=options.folder_name or DEFAULT_FOLDER_NAME, folder_id=folder_id, files=chats)


def sync_command(options: SyncOptions, env: CommandEnv) -> SyncResult:
    drive = _ensure_drive(env)
    folder_id = drive.resolve_folder_id(options.folder_name, options.folder_id)
    chats = drive.list_chats(options.folder_name, folder_id)
    chats = filter_chats(chats, options.name_filter, options.since, options.until)

    start_time = time.perf_counter()

    if options.selected_ids:
        ids = set(options.selected_ids)
        chats = [c for c in chats if c.get("id") in ids]

    if not chats:
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
        options.output_dir.mkdir(parents=True, exist_ok=True)

    items: List[SyncItem] = []
    totals_acc = RunAccumulator()
    for meta in chats:
        file_id = meta.get("id")
        name_safe = sanitize_filename(meta.get("name") or file_id or "chat")
        md_path = (options.output_dir / name_safe) / "conversation.md"
        data_bytes = drive.download_chat_bytes(file_id) if file_id else None
        if data_bytes is None:
            env.ui.console.print(f"[red]Failed to download {meta.get('name')}")
            continue
        try:
            obj = json.loads(data_bytes.decode("utf-8", errors="replace"))
        except Exception:
            env.ui.console.print(f"[yellow]Invalid JSON: {meta.get('name')}")
            continue
        chunks_raw = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
        if not isinstance(chunks_raw, list):
            env.ui.console.print(f"[yellow]No chunks: {meta.get('name')}")
            continue
        chunks = validate_chunks(chunks_raw)
        context = _context_from_drive(meta, obj, name_safe)
        conversation_id = file_id or name_safe
        doc = build_document_from_chunks(
            chunks,
            context,
            md_path,
            collapse_threshold=options.collapse_threshold,
            download_attachments=options.download_attachments,
            drive=drive,
            force=options.force,
            dry_run=options.dry_run,
        )
        outcome = _persist_conversation(
            env=env,
            options=options,
            doc=doc,
            context=context,
            output_dir=options.output_dir,
            slug_hint=name_safe,
            conversation_id=conversation_id,
            provider="drive-sync",
            md_path=md_path,
            extra_state={
                "driveFileId": file_id,
                "driveFolder": options.folder_name or DEFAULT_FOLDER_NAME,
            },
        )

        if outcome.skipped:
            totals_acc.increment("skipped")
            continue

        if not options.dry_run and context.modified_time and outcome.persist_result:
            mtime = parse_rfc3339_to_epoch(context.modified_time)
            if mtime is not None:
                try:
                    os.utime(outcome.persist_result.markdown_path, (mtime, mtime))
                except Exception:
                    pass
                if outcome.persist_result.html_path:
                    try:
                        os.utime(outcome.persist_result.html_path, (mtime, mtime))
                    except Exception:
                        pass

        item_slug = outcome.slug
        items.append(
            SyncItem(
                id=file_id,
                name=meta.get("name"),
                output=outcome.output_path,
                slug=item_slug,
                attachments=len(doc.attachments),
                stats=doc.stats,
                html=outcome.html_path,
                diff=outcome.diff_path,
            )
        )
        if outcome.diff_path:
            totals_acc.increment("diffs")
        totals_acc.add_stats(len(doc.attachments), doc.stats)

    pruned_count = 0
    if options.prune:
        wanted = {item.slug for item in items}
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
    add_run(
        {
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
        }
    )
    totals.setdefault("attachments", 0)
    totals.setdefault("skipped", 0)
    totals.setdefault("diffs", totals.get("diffs", 0))
    totals["pruned"] = pruned_count
    return SyncResult(
        count=len(items),
        output_dir=options.output_dir,
        folder_name=options.folder_name,
        folder_id=folder_id,
        items=items,
        total_stats=totals,
    )


def status_command(env: CommandEnv) -> StatusResult:
    credentials_present = DEFAULT_CREDENTIALS.exists()
    token_present = DEFAULT_TOKEN.exists()
    recent_runs: List[dict] = []
    run_summary_entries: Dict[str, RunSummaryEntry] = {}
    if RUNS_PATH.exists():
        try:
            data = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                recent_runs = data[-10:]
                for entry in data:
                    cmd = entry.get("cmd") or "unknown"
                    provider_hint = entry.get("provider") or _provider_from_cmd(cmd)
                    summary = run_summary_entries.setdefault(
                        cmd,
                        RunSummaryEntry(command=cmd, provider=provider_hint),
                    )
                    if provider_hint and summary.provider != provider_hint:
                        summary.provider = provider_hint
                    summary.update_from_run(entry)
        except Exception:
            pass
    provider_summary_entries: Dict[str, ProviderSummaryEntry] = {}
    for summary in run_summary_entries.values():
        provider = summary.provider or _provider_from_cmd(summary.command)
        entry = provider_summary_entries.setdefault(provider, ProviderSummaryEntry(provider=provider))
        entry.merge(summary)

    run_summary = {cmd: summary.as_dict() for cmd, summary in run_summary_entries.items()}
    provider_summary = {provider: entry.as_dict() for provider, entry in provider_summary_entries.items()}
    return StatusResult(
        credentials_present=credentials_present,
        token_present=token_present,
        state_path=STATE_PATH,
        runs_path=RUNS_PATH,
        recent_runs=recent_runs,
        run_summary=run_summary,
        provider_summary=provider_summary,
    )
