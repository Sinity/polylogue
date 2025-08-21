#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import json
import os
import shutil
import subprocess

from geminimd.util import colorize, parse_input_time_to_epoch, parse_rfc3339_to_epoch, sanitize_filename, add_run, STATE_PATH, RUNS_PATH
from geminimd.drive import get_drive_service, find_folder_id, list_children, get_file_meta, download_file, download_to_path
from geminimd.render import build_markdown_from_chunks, per_chunk_remote_links, extract_drive_ids
from geminimd.config import load_config, find_conf_path, default_conf_text


def main():
    parser = argparse.ArgumentParser(description="Gemini Markdown (gmd): opinionated JSON→Markdown and Drive sync")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # render local (files or directory)
    p_render = sub.add_parser("render", help="Render local Gemini JSON(s) to Markdown")
    p_render.add_argument("input", type=Path, help="Input file or directory containing Gemini JSON chats")
    p_render.add_argument("--out-dir", type=Path, default=None, help="Output directory for Markdown (default from config)")
    p_render.add_argument("--credentials", type=Path, default=None, help="Path to Google OAuth credentials.json (default from config)")
    p_render.add_argument("--remote-links", action="store_true", help="Do not download attachments; link Drive URLs instead")
    p_render.add_argument("--download-dir", type=Path, default=None, help="Base folder for downloaded attachments (default: per-file _attachments next to MD)")
    p_render.add_argument("--force", action="store_true", help="Force re-download attachments if present")
    p_render.add_argument("--collapse-threshold", type=int, default=None, help="Lines after which model responses fold (0 disables; default from config)")
    p_render.add_argument("--interactive", action="store_true", help="Use fzf to select files from directory input")
    p_render.add_argument("-v", "--verbose", action="store_true")

    # sync drive
    p_sync = sub.add_parser("sync", help="Sync Drive folder to local Markdown + attachments")
    p_sync.add_argument("--folder-id", type=str, default=None)
    p_sync.add_argument("--folder-name", type=str, default=None, help="Default from config")
    p_sync.add_argument("--out-dir", type=Path, default=None, help="Default from config")
    p_sync.add_argument("--credentials", type=Path, default=None, help="Default from config")
    p_sync.add_argument("--remote-links", action="store_true")
    p_sync.add_argument("--since", type=str, default=None)
    p_sync.add_argument("--until", type=str, default=None)
    p_sync.add_argument("--name-filter", type=str, default=None)
    p_sync.add_argument("--collapse-threshold", type=int, default=None, help="Default from config")
    p_sync.add_argument("--interactive", action="store_true", help="Use fzf to pick chats interactively")
    p_sync.add_argument("--force", action="store_true")
    p_sync.add_argument("--prune", action="store_true")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("-v", "--verbose", action="store_true")

    # status
    p_status = sub.add_parser("status", help="Show cached folder IDs and recent runs")

    # init config
    p_init = sub.add_parser("init", help="Create a project-local .gmdrc with documented defaults")

    # auth check
    p_auth = sub.add_parser("auth", help="Authenticate with Google Drive and cache token")
    p_auth.add_argument("--credentials", type=Path, default=None, help="Path to Google OAuth credentials.json (default from config)")

    args = parser.parse_args()
    cfg = load_config()

    if args.cmd == "render":
        out_dir = args.out_dir or Path(cfg["out_dir_render"]) 
        out_dir.mkdir(parents=True, exist_ok=True)
        targets = []
        if args.input.is_dir():
            cand = [p for p in sorted(args.input.iterdir()) if p.is_file() and (p.suffix.lower() == ".json" or "." not in p.name)]
            if args.interactive and shutil.which("fzf"):
                try:
                    proc = subprocess.run(["fzf", "--multi"], input="\n".join(str(p) for p in cand).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    sel = [Path(line) for line in proc.stdout.decode("utf-8").splitlines() if line.strip()]
                    targets = sel or cand
                except subprocess.CalledProcessError:
                    targets = cand
            else:
                targets = cand
        else:
            targets = [args.input]

        svc = None
        credentials = args.credentials or Path(cfg["credentials"]) 
        collapse_threshold = args.collapse_threshold if args.collapse_threshold is not None else int(cfg["collapse_threshold"]) 
        remote_links = args.remote_links or bool(cfg.get("remote_links"))
        if not remote_links:
            svc = get_drive_service(credentials, verbose=args.verbose)
            if not svc:
                print(colorize("Drive auth failed; use --remote-links to skip downloads.", "red"), file=sys.stderr)
                return
        # Process
        rendered = 0
        for p in tqdm(targets, total=len(targets), desc=colorize("Rendering", "cyan"), unit="file"):
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                print(colorize(f"Skip invalid JSON: {p.name} ({e})", "yellow"), file=sys.stderr)
                continue
            chunks = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
            if not isinstance(chunks, list):
                print(colorize(f"No chunks in {p.name}", "yellow"), file=sys.stderr)
                continue
            # Build links
            per_index_links = {}
            if remote_links:
                per_index_links = per_chunk_remote_links(chunks)
            else:
                per_index_links = {}
                # download attachments
                for idx, ch in enumerate(chunks):
                    ids = extract_drive_ids(ch)
                    if not ids:
                        continue
                    per_index_links[idx] = []
                    for att_id in ids:
                        meta_att = get_file_meta(svc, att_id)
                        if not meta_att:
                            continue
                        fname = sanitize_filename(meta_att.get("name", att_id))
                        attach_dir = args.download_dir or (out_dir / f"{sanitize_filename(p.stem)}_attachments")
                        attach_dir = Path(attach_dir)
                        attach_dir.mkdir(parents=True, exist_ok=True)
                        local_path = attach_dir / fname
                        need = not local_path.exists() or args.force
                        if need:
                            ok = download_to_path(svc, att_id, local_path)
                            if not ok:
                                continue
                        try:
                            mtime = parse_rfc3339_to_epoch(meta_att.get("modifiedTime"))
                            if mtime:
                                os.utime(local_path, (mtime, mtime))
                        except Exception:
                            pass
                        try:
                            rel = local_path.relative_to(out_dir)
                        except Exception:
                            rel = local_path
                        per_index_links[idx].append((fname, rel))

            md = build_markdown_from_chunks(
                chunks,
                per_index_links,
                title=p.stem,
                source_file_id=None,
                modified_time=None,
                created_time=None,
                run_settings=obj.get("runSettings") if isinstance(obj, dict) else None,
                citations=obj.get("citations") if isinstance(obj, dict) else None,
                collapse_threshold=collapse_threshold,
            )
            md_path = out_dir / f"{sanitize_filename(p.stem)}.md"
            md_path.write_text(md, encoding="utf-8")
            print(colorize(f"Rendered {p.name} → {md_path}", "green"))
            rendered += 1
        add_run({"cmd": "render", "count": rendered, "out": str(out_dir)})
        return

    if args.cmd == "sync":
        folder_name = args.folder_name or cfg["folder_name"]
        out_dir = args.out_dir or Path(cfg["out_dir_sync"]) 
        out_dir = Path(out_dir)
        credentials = args.credentials or Path(cfg["credentials"]) 
        collapse_threshold = args.collapse_threshold if args.collapse_threshold is not None else int(cfg["collapse_threshold"]) 
        remote_links = args.remote_links or bool(cfg.get("remote_links"))

        svc = get_drive_service(credentials, verbose=args.verbose)
        if not svc:
            print(colorize("Drive auth failed", "red"), file=sys.stderr)
            sys.exit(2)
        fid = args.folder_id or find_folder_id(svc, folder_name)
        if not fid:
            print(colorize(f"Folder not found: {folder_name}", "red"), file=sys.stderr)
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
        if args.interactive and shutil.which("fzf"):
            # Present: name \t modifiedTime \t id
            lines = [f"{c.get('name')}\t{c.get('modifiedTime')}\t{c.get('id')}" for c in chats]
            try:
                proc = subprocess.run(["fzf", "--multi"], input="\n".join(lines).encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                selected_ids = set([ln.split("\t")[-1] for ln in proc.stdout.decode("utf-8").splitlines() if ln.strip()])
                chats = [c for c in chats if c.get("id") in selected_ids] or chats
            except subprocess.CalledProcessError:
                pass
        if args.dry_run:
            print(f"Found {len(chats)} chat candidate(s). Output dir would be: {args.out_dir}")
            for c in chats[:20]:
                print(f"- {c['name']} ({c['id']})")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        synced = 0
        for meta in tqdm(chats, total=len(chats), desc=colorize("Syncing", "cyan"), unit="chat"):
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
            if remote_links:
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
                collapse_threshold=collapse_threshold,
            )
            md_path.write_text(md, encoding="utf-8")
            # Set MD mtime to Drive modifiedTime
            try:
                mtime = parse_rfc3339_to_epoch(meta.get("modifiedTime"))
                if mtime:
                    os.utime(md_path, (mtime, mtime))
            except Exception:
                pass
            synced += 1
        add_run({"cmd": "sync", "count": synced, "out": str(out_dir), "folder_id": fid, "folder_name": folder_name, "remote_links": remote_links})
        return

    if args.cmd == "status":
        print("Config:")
        from geminimd.config import CONF_PATH
        print(f"  file: {CONF_PATH}")
        print(f"  folder_name: {cfg['folder_name']}")
        print(f"  credentials: {cfg['credentials']}")
        print(f"  collapse_threshold: {cfg['collapse_threshold']}")
        print(f"  out_dir_render: {cfg['out_dir_render']}")
        print(f"  out_dir_sync: {cfg['out_dir_sync']}")
        print(f"  remote_links: {cfg['remote_links']}")
        print("")
        print(f"Folder cache: {STATE_PATH}")
        if STATE_PATH.exists():
            try:
                st = json.loads(STATE_PATH.read_text(encoding='utf-8'))
                print(json.dumps(st, indent=2))
            except Exception:
                print("  (unreadable)")
        else:
            print("  (none)")
        print("")
        print(f"Recent runs: {RUNS_PATH}")
        if RUNS_PATH.exists():
            try:
                runs = json.loads(RUNS_PATH.read_text(encoding='utf-8'))
                print("  cmd   count   details")
                for r in runs[-10:]:
                    if r.get('cmd') == 'sync':
                        print(f"  sync  {r.get('count'):>5}   {r.get('folder_name')} → {r.get('out')}")
                    else:
                        print(f"  rend  {r.get('count'):>5}   {r.get('out')}")
            except Exception:
                print("  (unreadable)")
        else:
            print("  (none)")
        return


if __name__ == "__main__":
    main()
