#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import json
import os

from geminimd.util import colorize, parse_input_time_to_epoch, parse_rfc3339_to_epoch, sanitize_filename
from geminimd.drive import get_drive_service, find_folder_id, list_children, get_file_meta, download_file, download_to_path
from geminimd.render import build_markdown_from_chunks, per_chunk_remote_links, extract_drive_ids


def main():
    parser = argparse.ArgumentParser(description="Gemini Markdown (gmd): opinionated JSON→Markdown and Drive sync")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # render local
    p_render = sub.add_parser("render", help="Render local Gemini JSONs to Markdown")
    p_render.add_argument("input_dir", type=Path, help="Directory with Gemini JSON chats (no extensions)")
    p_render.add_argument("--out-dir", type=Path, default=Path("gmd_out"))
    p_render.add_argument("-v", "--verbose", action="store_true")

    # sync drive
    p_sync = sub.add_parser("sync", help="Sync Drive folder to local Markdown + attachments")
    p_sync.add_argument("--folder-id", type=str, default=None)
    p_sync.add_argument("--folder-name", type=str, default="AI Studio")
    p_sync.add_argument("--out-dir", type=Path, default=Path("gemini_synced"))
    p_sync.add_argument("--credentials", type=Path, default=Path("credentials.json"))
    p_sync.add_argument("--remote-links", action="store_true")
    p_sync.add_argument("--since", type=str, default=None)
    p_sync.add_argument("--until", type=str, default=None)
    p_sync.add_argument("--name-filter", type=str, default=None)
    p_sync.add_argument("--collapse-threshold", type=int, default=10)
    p_sync.add_argument("--force", action="store_true")
    p_sync.add_argument("--prune", action="store_true")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.cmd == "render":
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(args.input_dir.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            if "." in name and p.suffix.lower() != ".json":
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            chunks = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
            if not isinstance(chunks, list):
                continue
            links = per_chunk_remote_links(chunks)
            md = build_markdown_from_chunks(
                chunks,
                links,
                title=p.stem,
                source_file_id=None,
                modified_time=None,
                created_time=None,
                run_settings=obj.get("runSettings") if isinstance(obj, dict) else None,
                citations=obj.get("citations") if isinstance(obj, dict) else None,
            )
            md_path = out_dir / f"{sanitize_filename(p.stem)}.md"
            md_path.write_text(md, encoding="utf-8")
        return

    if args.cmd == "sync":
        svc = get_drive_service(args.credentials, verbose=args.verbose)
        if not svc:
            print(colorize("Drive auth failed", "red"), file=sys.stderr)
            sys.exit(2)
        fid = args.folder_id or find_folder_id(svc, args.folder_name)
        if not fid:
            print(colorize(f"Folder not found: {args.folder_name}", "red"), file=sys.stderr)
            sys.exit(1)
        children = list_children(svc, fid)
        # Chats = files without extension and not Google Docs types
        chats = [c for c in children if ("." not in (c.get("name") or "")) and not (c.get("mimeType", "").startswith("application/vnd.google-apps."))]
        # Filters
        import re as _re
        if args.name_filter:
            try:
                rx = _re.compile(args.name_filter)
                chats = [c for c in chats if rx.search(c.get("name", "") or "")]
            except _re.error:
                print(colorize("Invalid --name-filter regex; ignoring.", "yellow"), file=sys.stderr)
        s_epoch = parse_input_time_to_epoch(args.since)
        u_epoch = parse_input_time_to_epoch(args.until)
        if s_epoch or u_epoch:
            _tmp = []
            for c in chats:
                mt = parse_rfc3339_to_epoch(c.get("modifiedTime"))
                if mt is None:
                    continue
                if s_epoch and mt < s_epoch:
                    continue
                if u_epoch and mt > u_epoch:
                    continue
                _tmp.append(c)
            chats = _tmp
        if args.dry_run:
            print(f"Found {len(chats)} chat candidate(s). Output dir would be: {args.out_dir}")
            for c in chats[:20]:
                print(f"- {c['name']} ({c['id']})")
            return
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for meta in chats:
            file_id = meta["id"]
            name_safe = sanitize_filename(meta.get("name") or file_id[:8])
            md_path = out_dir / f"{name_safe}.md"
            attachments_dir = out_dir / f"{name_safe}_attachments"
            # Download chat JSON
            data_bytes = download_file(svc, file_id)
            if data_bytes is None:
                print(colorize(f"Failed to download chat: {meta.get('name')}", "red"), file=sys.stderr)
                continue
            try:
                obj = json.loads(data_bytes.decode("utf-8", errors="replace"))
            except Exception:
                print(colorize(f"Invalid JSON: {meta.get('name')}", "red"), file=sys.stderr)
                continue
            chunks = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
            if not isinstance(chunks, list):
                print(colorize(f"No chunks: {meta.get('name')}", "yellow"), file=sys.stderr)
                continue
            # Attachments
            per_index_links = {}
            if args.remote_links:
                per_index_links = per_chunk_remote_links(chunks)
            else:
                per_index_links = {}
                for idx, ch in enumerate(chunks):
                    ids = extract_drive_ids(ch)
                    if not ids:
                        continue
                    per_index_links[idx] = []
                    for att_id in ids:
                        att_meta = get_file_meta(svc, att_id)
                        if not att_meta:
                            continue
                        fname = sanitize_filename(att_meta.get("name", att_id))
                        local_path = attachments_dir / fname
                        ok = True
                        if not local_path.exists() or args.force:
                            ok = download_to_path(svc, att_id, local_path)
                        if ok:
                            try:
                                mtime = parse_rfc3339_to_epoch(att_meta.get("modifiedTime"))
                                if mtime:
                                    os.utime(local_path, (mtime, mtime))
                            except Exception:
                                pass
                            try:
                                rel = local_path.relative_to(out_dir)
                            except Exception:
                                rel = local_path
                            per_index_links[idx].append((fname, rel))
            # Render
            md = build_markdown_from_chunks(
                chunks,
                per_index_links,
                meta.get("name", name_safe),
                meta.get("id"),
                meta.get("modifiedTime"),
                meta.get("createdTime"),
                run_settings=obj.get("runSettings") if isinstance(obj, dict) else None,
                citations=obj.get("citations") if isinstance(obj, dict) else None,
                source_mime=meta.get("mimeType"),
                source_size=int(meta.get("size", 0)) if meta.get("size") is not None else None,
                collapse_threshold=args.collapse_threshold,
            )
            md_path.write_text(md, encoding="utf-8")
            # Set MD mtime to Drive modifiedTime
            try:
                mtime = parse_rfc3339_to_epoch(meta.get("modifiedTime"))
                if mtime:
                    os.utime(md_path, (mtime, mtime))
            except Exception:
                pass
        return


if __name__ == "__main__":
    main()
