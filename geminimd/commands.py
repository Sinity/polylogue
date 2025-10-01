from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .cli_common import compute_prune_paths, filter_chats
from .drive_client import (
    DEFAULT_CREDENTIALS,
    DEFAULT_FOLDER_NAME,
    DEFAULT_TOKEN,
    DriveClient,
)
from .options import (
    ListOptions,
    ListResult,
    RenderOptions,
    RenderResult,
    StatusResult,
    SyncItem,
    SyncOptions,
    SyncResult,
)
from .render import build_markdown_from_chunks, extract_drive_ids, per_chunk_remote_links
from .ui import UI
from .util import RUNS_PATH, STATE_PATH, add_run, parse_rfc3339_to_epoch, sanitize_filename


@dataclass
class CommandEnv:
    ui: UI
    drive: Optional[DriveClient] = None


def render_command(options: RenderOptions, env: CommandEnv) -> RenderResult:
    ui = env.ui
    output_dir = options.output_dir
    if not options.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    files_written: List[str] = []
    attachment_needed = options.download_attachments
    drive = env.drive if attachment_needed else None
    if attachment_needed and drive is None:
        drive = DriveClient(ui)
        env.drive = drive

    for src in options.inputs:
        try:
            obj = json.loads(src.read_text(encoding="utf-8"))
        except Exception as exc:
            ui.console.print(f"[yellow]Skipping {src.name}: {exc}")
            continue
        chunks = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
        if not isinstance(chunks, list):
            ui.console.print(f"[yellow]No chunks in {src.name}")
            continue
        md_path = output_dir / f"{sanitize_filename(src.stem)}.md"
        if attachment_needed and drive is not None:
            per_index_links = _collect_attachments_for_render(
                drive,
                chunks,
                md_path,
                output_dir,
                force=options.force,
                dry_run=options.dry_run,
            )
        else:
            per_index_links = per_chunk_remote_links(chunks)
        md = build_markdown_from_chunks(
            chunks,
            per_index_links,
            obj.get("title") or src.stem,
            obj.get("id"),
            obj.get("modifiedTime"),
            obj.get("createTime") or obj.get("createdTime"),
            run_settings=obj.get("runSettings"),
            citations=obj.get("citations"),
            source_mime=obj.get("mimeType"),
            collapse_threshold=options.collapse_threshold,
        )
        if not options.dry_run:
            md_path.write_text(md, encoding="utf-8")
        files_written.append(md_path.name)

    add_run({"cmd": "render", "count": len(files_written), "out": str(output_dir)})
    return RenderResult(count=len(files_written), output_dir=output_dir, files=files_written)


def _collect_attachments_for_render(
    drive: DriveClient,
    chunks: List[Dict],
    md_path: Path,
    out_dir: Path,
    *,
    force: bool,
    dry_run: bool,
) -> Dict[int, List]:
    per_index_links: Dict[int, List] = {}
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
    return per_index_links


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

    if options.selected_ids:
        ids = set(options.selected_ids)
        chats = [c for c in chats if c.get("id") in ids]

    if not chats:
        return SyncResult(0, options.output_dir, options.folder_name, folder_id, [])

    if not options.dry_run:
        options.output_dir.mkdir(parents=True, exist_ok=True)

    items: List[SyncItem] = []
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
        chunks = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
        if not isinstance(chunks, list):
            env.ui.console.print(f"[yellow]No chunks: {meta.get('name')}")
            continue
        if options.download_attachments:
            per_index_links = _collect_attachments_for_render(
                drive,
                chunks,
                md_path,
                options.output_dir,
                force=options.force,
                dry_run=options.dry_run,
            )
        else:
            per_index_links = per_chunk_remote_links(chunks)
        md = build_markdown_from_chunks(
            chunks,
            per_index_links,
            meta.get("name", name_safe),
            meta.get("id"),
            meta.get("modifiedTime"),
            meta.get("createdTime"),
            run_settings=obj.get("runSettings"),
            citations=obj.get("citations"),
            source_mime=meta.get("mimeType"),
            collapse_threshold=options.collapse_threshold,
        )
        if not options.dry_run:
            md_path.write_text(md, encoding="utf-8")
            mtime = parse_rfc3339_to_epoch(meta.get("modifiedTime"))
            if mtime:
                try:
                    os.utime(md_path, (mtime, mtime))
                except Exception:
                    pass
        attachments_count = 0
        if isinstance(per_index_links, dict):
            attachments_count = sum(len(v) for v in per_index_links.values())
        items.append(SyncItem(id=file_id, name=meta.get("name"), output=md_path, attachments=attachments_count))

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

    add_run(
        {
            "cmd": "sync",
            "count": len(items),
            "out": str(options.output_dir),
            "folder_name": options.folder_name,
            "folder_id": folder_id,
        }
    )

    return SyncResult(
        count=len(items),
        output_dir=options.output_dir,
        folder_name=options.folder_name,
        folder_id=folder_id,
        items=items,
    )


def status_command(env: CommandEnv) -> StatusResult:
    credentials_present = DEFAULT_CREDENTIALS.exists()
    token_present = DEFAULT_TOKEN.exists()
    recent_runs: List[dict] = []
    if RUNS_PATH.exists():
        try:
            data = json.loads(RUNS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                recent_runs = data[-10:]
        except Exception:
            pass
    return StatusResult(
        credentials_present=credentials_present,
        token_present=token_present,
        state_path=STATE_PATH,
        runs_path=RUNS_PATH,
        recent_runs=recent_runs,
    )
