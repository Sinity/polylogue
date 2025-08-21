#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

from geminimd.util import colorize, sanitize_filename, parse_rfc3339_to_epoch
from geminimd.drive import get_drive_service, get_file_meta, download_to_path
from geminimd.render import build_markdown_from_chunks, per_chunk_remote_links, extract_drive_ids


def process_file(
    input_file: Path,
    output_file: Path,
    attachments_dir: Path,
    credentials: Path,
    remote_links: bool,
    force_download: bool,
    collapse_threshold: int,
    verbose: bool,
) -> None:
    try:
        obj = json.loads(input_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(colorize(f"Error reading {input_file}: {e}", "red"), file=sys.stderr)
        return
    chunks = obj.get("chunkedPrompt", {}).get("chunks") if isinstance(obj, dict) else None
    if not isinstance(chunks, list):
        print(colorize(f"No chunks in {input_file.name}", "yellow"), file=sys.stderr)
        return

    per_index_links: Dict[int, List[Tuple[str, Union[Path, str]]]] = {}
    if remote_links:
        per_index_links = per_chunk_remote_links(chunks)
    else:
        svc = get_drive_service(credentials, verbose=verbose)
        if not svc:
            print(colorize("Drive auth failed; use --remote-links to skip downloads.", "red"), file=sys.stderr)
            return
        for idx, ch in enumerate(chunks):
            ids = extract_drive_ids(ch)
            if not ids:
                continue
            per_index_links[idx] = []
            for fid in ids:
                meta = get_file_meta(svc, fid)
                if not meta:
                    continue
                fname = sanitize_filename(meta.get("name", fid))
                target = attachments_dir / fname
                ok = True
                if not target.exists() or force_download:
                    ok = download_to_path(svc, fid, target)
                if ok:
                    try:
                        mtime = parse_rfc3339_to_epoch(meta.get("modifiedTime"))
                        if mtime:
                            os.utime(target, (mtime, mtime))
                    except Exception:
                        pass
                    try:
                        rel = target.relative_to(output_file.parent)
                    except Exception:
                        rel = target
                    per_index_links[idx].append((fname, rel))

    md = build_markdown_from_chunks(
        chunks,
        per_index_links,
        title=input_file.stem,
        source_file_id=None,
        modified_time=None,
        created_time=None,
        run_settings=obj.get("runSettings") if isinstance(obj, dict) else None,
        citations=obj.get("citations") if isinstance(obj, dict) else None,
        collapse_threshold=collapse_threshold,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(md, encoding="utf-8")
    print(colorize(f"Wrote {output_file}", "green"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemini chat JSON to Markdown (opinionated callout style)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_files", nargs="+", type=Path, help="One or more input JSON files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults alongside input)")
    parser.add_argument("--credentials", type=Path, default=Path("credentials.json"))
    parser.add_argument("--remote-links", action="store_true", help="Do not download attachments; link to Drive URLs")
    parser.add_argument("--download-dir", type=Path, default=None, help="Base dir for attachments (default: per-file _attachments next to MD)")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--collapse-threshold", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    for inp in args.input_files:
        inp = inp.resolve()
        out_dir = args.output_dir.resolve() if args.output_dir else inp.parent
        out_path = out_dir / f"{inp.stem}.md"
        attach_dir = args.download_dir.resolve() if args.download_dir else (out_dir / f"{out_path.stem}_attachments")
        process_file(
            input_file=inp,
            output_file=out_path,
            attachments_dir=attach_dir,
            credentials=args.credentials.resolve(),
            remote_links=args.remote_links,
            force_download=args.force_download,
            collapse_threshold=args.collapse_threshold,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()

