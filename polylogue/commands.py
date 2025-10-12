from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
)
from .render import (
    AttachmentInfo,
    MarkdownDocument,
    build_markdown_from_chunks,
    extract_drive_ids,
    per_chunk_remote_links,
    remote_attachment_info,
)
from .document_store import persist_document
from .ui import UI
from .util import (
    RUNS_PATH,
    STATE_PATH,
    add_run,
    parse_rfc3339_to_epoch,
    sanitize_filename,
    snapshot_for_diff,
    write_delta_diff,
)


PROVIDER_ALIASES = {
    "sync": "drive-sync",
    "polylogue-sync-drive": "drive-sync",
    "render": "render",
    "polylogue-sync-codex": "codex",
    "polylogue-sync-claude-code": "claude-code",
}


def _provider_from_cmd(cmd: str) -> str:
    if cmd in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[cmd]
    if cmd.startswith("codex"):
        return "codex"
    if cmd.startswith("claude-code"):
        return "claude-code"
    if cmd.startswith("drive"):
        return "drive-sync"
    return cmd


@dataclass
class CommandEnv:
    ui: UI
    drive: Optional[DriveClient] = None


@dataclass
class ChatContext:
    title: str
    chat_id: Optional[str]
    modified_time: Optional[str]
    created_time: Optional[str]
    run_settings: Optional[Any]
    citations: Optional[Any]
    source_mime: Optional[str]


def render_command(options: RenderOptions, env: CommandEnv) -> RenderResult:
    ui = env.ui
    output_dir = options.output_dir
    if not options.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    attachment_needed = options.download_attachments
    drive = env.drive if attachment_needed else None
    if attachment_needed and drive is None:
        drive = DriveClient(ui)
        env.drive = drive

    from collections import defaultdict

    render_files: List[RenderFile] = []
    totals = defaultdict(int)
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
        md_path = output_dir / f"{sanitize_filename(src.stem)}.md"
        context = _context_from_local(obj, src.stem)
        conversation_id = context.chat_id or md_path.stem
        doc = _build_markdown_output(
            chunks,
            context,
            md_path,
            output_dir,
            collapse_threshold=options.collapse_threshold,
            download_attachments=attachment_needed,
            drive=drive,
            force=options.force,
            dry_run=options.dry_run,
        )
        snapshot = None
        persist_result = None
        html_path: Optional[Path] = None
        diff_path: Optional[Path] = None
        if not options.dry_run:
            if options.diff:
                snapshot = snapshot_for_diff(md_path)
            persist_result = persist_document(
                provider="render",
                conversation_id=conversation_id,
                title=context.title,
                document=doc,
                output_dir=output_dir,
                collapse_threshold=options.collapse_threshold,
                attachments=doc.attachments,
                updated_at=context.modified_time,
                created_at=context.created_time,
                html=options.html,
                html_theme=options.html_theme,
                attachment_policy=None,
                extra_state={
                    "sourceFile": str(src),
                    "sourceMimeType": context.source_mime,
                },
                slug_hint=md_path.stem,
                id_hint=md_path.stem[:8],
                force=options.force,
            )
            if options.diff and snapshot is not None and persist_result and not persist_result.skipped:
                diff_path = write_delta_diff(snapshot, persist_result.markdown_path) or None
            if snapshot is not None:
                try:
                    snapshot.unlink()
                except Exception:
                    pass
                snapshot = None
            html_path = persist_result.html_path if persist_result else None
            output_path = persist_result.markdown_path if persist_result else md_path
            attachments_dir = persist_result.attachments_dir if persist_result else None
        else:
            if snapshot is not None:
                try:
                    snapshot.unlink()
                except Exception:
                    pass
            output_path = md_path
            attachments_dir = None

        if persist_result and persist_result.skipped:
            totals.setdefault("skipped", 0)
            totals["skipped"] += 1
            if options.diff and snapshot is not None:
                try:
                    snapshot.unlink()
                except Exception:
                    pass
            continue

        render_files.append(
            RenderFile(
                output=output_path,
                attachments=len(doc.attachments),
                stats=doc.stats,
                html=html_path,
                diff=diff_path,
            )
        )
        totals["attachments"] += len(doc.attachments)
        for key, value in doc.stats.items():
            if isinstance(value, (int, float)):
                totals[key] += value

    diff_count = sum(1 for item in render_files if item.diff)
    duration = time.perf_counter() - start_time
    add_run(
        {
            "cmd": "render",
            "count": len(render_files),
            "out": str(output_dir),
            "attachments": totals.get("attachments", 0),
            "attachmentBytes": totals.get("attachmentBytes", 0),
            "tokens": totals.get("totalTokensApprox", 0),
            "diffs": diff_count,
            "duration": duration,
        }
    )
    totals.setdefault("attachments", 0)
    totals.setdefault("attachments", 0)
    totals.setdefault("skipped", 0)
    return RenderResult(
        count=len(render_files),
        output_dir=output_dir,
        files=render_files,
        total_stats=dict(totals),
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


def _build_markdown_output(
    chunks: List[Dict],
    context: ChatContext,
    md_path: Path,
    out_dir: Path,
    *,
    collapse_threshold: int,
    download_attachments: bool,
    drive: Optional[DriveClient],
    force: bool,
    dry_run: bool,
) -> MarkdownDocument:
    if download_attachments and drive is not None:
        per_index_links, attachments = _collect_attachments(
            drive,
            chunks,
            md_path,
            out_dir,
            force=force,
            dry_run=dry_run,
        )
    else:
        per_index_links = per_chunk_remote_links(chunks)
        attachments = remote_attachment_info(per_index_links)

    doc = build_markdown_from_chunks(
        chunks,
        per_index_links,
        context.title,
        context.chat_id,
        context.modified_time,
        context.created_time,
        run_settings=context.run_settings,
        citations=context.citations,
        source_mime=context.source_mime,
        collapse_threshold=collapse_threshold,
        attachments=attachments,
    )
    return doc


def _collect_attachments(
    drive: DriveClient,
    chunks: List[Dict],
    md_path: Path,
    out_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> Tuple[Dict[int, List], List[AttachmentInfo]]:
    per_index_links: Dict[int, List] = {}
    attachments: List[AttachmentInfo] = []
    seen: set[Tuple[str, str]] = set()
    attachments_dir = out_dir / f"{md_path.stem}_attachments"
    if not dry_run:
        attachments_dir.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        ids = extract_drive_ids(chunk)
        if not ids:
            continue
        per_index_links[idx] = []
        for att_id in ids:
            meta_att = drive.attachment_meta(att_id)
            if not meta_att:
                continue
            fname = sanitize_filename(meta_att.get("name", att_id))
            local_path = attachments_dir / fname
            if not dry_run:
                if force or not local_path.exists():
                    ok = drive.download_attachment(att_id, local_path)
                    if not ok:
                        continue
                drive.touch_mtime(local_path, meta_att.get("modifiedTime"))
            try:
                rel = local_path.relative_to(out_dir)
            except Exception:
                rel = local_path
            per_index_links[idx].append((fname, rel))
            size = None
            if not dry_run and local_path.exists():
                try:
                    size = local_path.stat().st_size
                except OSError:
                    size = None
            key = (fname, att_id)
            if key not in seen:
                seen.add(key)
                attachments.append(
                    AttachmentInfo(
                        name=fname,
                        link=f"attachment://{att_id}",
                        local_path=None if dry_run else rel,
                        size_bytes=size,
                        remote=False,
                    )
                )
    return per_index_links, attachments


def list_command(options: ListOptions, env: CommandEnv) -> ListResult:
    drive = env.drive or DriveClient(env.ui)
    env.drive = drive
    chats = drive.list_chats(options.folder_name, options.folder_id)
    chats = filter_chats(chats, options.name_filter, options.since, options.until)
    folder_id = drive.resolve_folder_id(options.folder_name, options.folder_id)
    return ListResult(folder_name=options.folder_name or DEFAULT_FOLDER_NAME, folder_id=folder_id, files=chats)


def sync_command(options: SyncOptions, env: CommandEnv) -> SyncResult:
    drive = env.drive or DriveClient(env.ui)
    env.drive = drive
    folder_id = drive.resolve_folder_id(options.folder_name, options.folder_id)
    chats = drive.list_chats(options.folder_name, folder_id)
    chats = filter_chats(chats, options.name_filter, options.since, options.until)

    start_time = time.perf_counter()

    if options.selected_ids:
        ids = set(options.selected_ids)
        chats = [c for c in chats if c.get("id") in ids]

    if not chats:
        return SyncResult(0, options.output_dir, options.folder_name, folder_id, [])

    if not options.dry_run:
        options.output_dir.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict

    items: List[SyncItem] = []
    totals = defaultdict(int)
    diff_count = 0
    for meta in chats:
        file_id = meta.get("id")
        name_safe = sanitize_filename(meta.get("name") or file_id or "chat")
        md_path = options.output_dir / f"{name_safe}.md"
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
        doc = _build_markdown_output(
            chunks,
            context,
            md_path,
            options.output_dir,
            collapse_threshold=options.collapse_threshold,
            download_attachments=options.download_attachments,
            drive=drive,
            force=options.force,
            dry_run=options.dry_run,
        )
        snapshot = None
        persist_result = None
        html_path: Optional[Path] = None
        diff_path: Optional[Path] = None
        output_path = md_path
        if not options.dry_run:
            if options.diff:
                snapshot = snapshot_for_diff(md_path)
            persist_result = persist_document(
                provider="drive-sync",
                conversation_id=conversation_id,
                title=context.title,
                document=doc,
                output_dir=options.output_dir,
                collapse_threshold=options.collapse_threshold,
                attachments=doc.attachments,
                updated_at=context.modified_time,
                created_at=context.created_time,
                html=options.html,
                html_theme=options.html_theme,
                attachment_policy=None,
                extra_state={
                    "driveFileId": file_id,
                    "driveFolder": options.folder_name or DEFAULT_FOLDER_NAME,
                },
                slug_hint=name_safe,
                id_hint=name_safe[:8],
                force=options.force,
            )
            if options.diff and snapshot is not None and persist_result and not persist_result.skipped:
                diff_path = write_delta_diff(snapshot, persist_result.markdown_path) or None
            if snapshot is not None:
                try:
                    snapshot.unlink()
                except Exception:
                    pass
                snapshot = None
            if persist_result:
                output_path = persist_result.markdown_path
            html_path = persist_result.html_path if persist_result else None

        if persist_result and persist_result.skipped:
            totals.setdefault("skipped", 0)
            totals["skipped"] += 1
            continue

        if not options.dry_run and context.modified_time and persist_result:
            mtime = parse_rfc3339_to_epoch(context.modified_time)
            if mtime:
                try:
                    os.utime(persist_result.markdown_path, (mtime, mtime))
                except Exception:
                    pass
                if persist_result.html_path:
                    try:
                        os.utime(persist_result.html_path, (mtime, mtime))
                    except Exception:
                        pass

        items.append(
            SyncItem(
                id=file_id,
                name=meta.get("name"),
                output=output_path,
                attachments=len(doc.attachments),
                stats=doc.stats,
                html=html_path,
                diff=diff_path,
            )
        )
        if diff_path:
            diff_count += 1
        totals["attachments"] += len(doc.attachments)
        for key, value in doc.stats.items():
            if isinstance(value, (int, float)):
                totals[key] += value

    pruned_count = 0
    if options.prune:
        wanted = {sanitize_filename(item.name or item.id or "") for item in items}
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
    add_run(
        {
            "cmd": "sync",
            "count": len(items),
            "out": str(options.output_dir),
            "folder_name": options.folder_name,
            "folder_id": folder_id,
            "attachments": totals.get("attachments", 0),
            "attachmentBytes": totals.get("attachmentBytes", 0),
            "tokens": totals.get("totalTokensApprox", 0),
            "skipped": totals.get("skipped", 0),
            "pruned": pruned_count,
            "diffs": diff_count,
            "duration": duration,
        }
    )

    totals.setdefault("attachments", 0)
    totals.setdefault("attachments", 0)
    totals.setdefault("skipped", 0)
    return SyncResult(
        count=len(items),
        output_dir=options.output_dir,
        folder_name=options.folder_name,
        folder_id=folder_id,
        items=items,
        total_stats=dict(totals),
    )


def status_command(env: CommandEnv) -> StatusResult:
    credentials_present = DEFAULT_CREDENTIALS.exists()
    token_present = DEFAULT_TOKEN.exists()
    recent_runs: List[dict] = []
    run_summary: Dict[str, Any] = {}
    if RUNS_PATH.exists():
        try:
            data = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                recent_runs = data[-10:]
                for entry in data:
                    cmd = entry.get("cmd") or "unknown"
                    stats = run_summary.setdefault(
                        cmd,
                        {
                            "count": 0,
                            "attachments": 0,
                            "attachmentBytes": 0,
                            "tokens": 0,
                            "skipped": 0,
                            "pruned": 0,
                            "diffs": 0,
                            "duration": 0.0,
                            "last": None,
                            "last_out": None,
                            "last_count": None,
                        },
                    )
                    stats["count"] += entry.get("count", 0) or 0
                    stats["attachments"] += entry.get("attachments", 0) or 0
                    stats["attachmentBytes"] += entry.get("attachmentBytes", 0) or 0
                    stats["tokens"] += entry.get("tokens", 0) or 0
                    stats["skipped"] += entry.get("skipped", 0) or 0
                    stats["pruned"] += entry.get("pruned", 0) or 0
                    stats["diffs"] += entry.get("diffs", 0) or 0
                    stats["duration"] += entry.get("duration", 0.0) or 0.0
                    stats["last"] = entry.get("timestamp")
                    stats["last_out"] = entry.get("out")
                    stats["last_count"] = entry.get("count")
        except Exception:
            pass
    provider_summary: Dict[str, Any] = {}
    for cmd, stats in run_summary.items():
        provider = _provider_from_cmd(cmd)
        entry = provider_summary.setdefault(
            provider,
            {
                "commands": set(),
                "count": 0,
                "attachments": 0,
                "attachmentBytes": 0,
                "tokens": 0,
                "skipped": 0,
                "pruned": 0,
                "diffs": 0,
                "duration": 0.0,
                "last": None,
                "last_out": None,
                "last_count": None,
            },
        )
        entry["commands"].add(cmd)
        entry["count"] += stats.get("count", 0) or 0
        entry["attachments"] += stats.get("attachments", 0) or 0
        entry["attachmentBytes"] += stats.get("attachmentBytes", 0) or 0
        entry["tokens"] += stats.get("tokens", 0) or 0
        entry["skipped"] += stats.get("skipped", 0) or 0
        entry["pruned"] += stats.get("pruned", 0) or 0
        entry["diffs"] += stats.get("diffs", 0) or 0
        entry["duration"] += stats.get("duration", 0.0) or 0.0
        last_ts = stats.get("last")
        if last_ts and (entry["last"] is None or last_ts > entry["last"]):
            entry["last"] = last_ts
            entry["last_out"] = stats.get("last_out")
            entry["last_count"] = stats.get("last_count")
    for value in provider_summary.values():
        value["commands"] = sorted(value["commands"])
    return StatusResult(
        credentials_present=credentials_present,
        token_present=token_present,
        state_path=STATE_PATH,
        runs_path=RUNS_PATH,
        recent_runs=recent_runs,
        run_summary=run_summary,
        provider_summary=provider_summary,
    )
