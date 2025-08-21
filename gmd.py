#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import sync_ai_studio as syncmod


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
        syncmod.process_local_directory(args.input_dir, args.out_dir, verbose=args.verbose)
        return

    if args.cmd == "sync":
        if args.dry_run:
            # Call sync_ai_studio main dry-run discovery path
            # Minimal duplication: reuse module utilities
            svc = syncmod.get_drive_service(args.credentials, verbose=args.verbose)
            if not svc:
                print("Drive auth failed", file=sys.stderr)
                sys.exit(2)
            folder_id = args.folder_id or syncmod.find_folder_id(svc, args.folder_name)
            children = syncmod.list_children(svc, folder_id)
            chats = [c for c in children if syncmod.is_chat_candidate(c)]
            print(f"Found {len(chats)} chat candidate(s). Output dir would be: {args.out_dir}")
            for c in chats[:20]:
                print(f"- {c['name']} ({c['id']})")
            return
        syncmod.sync_folder(
            service=syncmod.get_drive_service(args.credentials, verbose=args.verbose),
            folder_id=args.folder_id or syncmod.find_folder_id(syncmod.get_drive_service(args.credentials), args.folder_name),
            out_dir=args.out_dir,
            credentials_path=args.credentials,
            force=args.force,
            prune=args.prune,
            verbose=args.verbose,
            remote_links=args.remote_links,
            since=args.since,
            until=args.until,
            name_filter=args.name_filter,
            collapse_threshold=args.collapse_threshold,
        )


if __name__ == "__main__":
    main()

