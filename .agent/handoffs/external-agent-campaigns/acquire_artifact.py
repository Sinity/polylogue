#!/usr/bin/env python3
"""Acquire one externally linked artifact into immutable campaign raw custody."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


def safe_filename(value: str) -> str:
    name = Path(unquote(value)).name
    if not name or name in {".", ".."} or name != value and "/" in value:
        raise ValueError("filename must be one basename")
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wave_root", type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--filename")
    parser.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    if not re.fullmatch(r"a[0-9]{2}", args.attempt):
        raise SystemExit("error: --attempt must be aNN")
    if args.max_bytes <= 0:
        raise SystemExit("error: --max-bytes must be positive")
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise SystemExit("error: only http, https, and file URLs are supported")
    filename = safe_filename(args.filename or parsed.path.rsplit("/", 1)[-1])
    raw_dir = args.wave_root / args.workload / "results" / args.job / args.attempt / "raw"
    receipt = raw_dir.parent / "acquisition.json"
    target = raw_dir / filename
    if receipt.exists() or target.exists():
        raise SystemExit("error: immutable acquisition target already exists")
    raw_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(
            Request(args.url, headers={"User-Agent": "polylogue-campaign-acquirer/1"}), timeout=30
        ) as response:
            with tempfile.NamedTemporaryFile("wb", dir=raw_dir, delete=False) as temporary:
                temp_path = Path(temporary.name)
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > args.max_bytes:
                        raise ValueError(f"artifact exceeds --max-bytes ({args.max_bytes})")
                    digest.update(chunk)
                    temporary.write(chunk)
            content_type = response.headers.get("Content-Type")
    except Exception as exc:
        if "temp_path" in locals():
            temp_path.unlink(missing_ok=True)
        raise SystemExit(f"error: acquisition failed: {exc}") from exc
    os.replace(temp_path, target)
    payload = {
        "schema_version": 1,
        "workload_id": args.workload,
        "job_id": args.job,
        "attempt_id": args.attempt,
        "source_url": args.url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "artifact": {
            "filename": filename,
            "sha256": digest.hexdigest(),
            "size_bytes": size,
            "custody_path": str(target.relative_to(args.wave_root)),
        },
        "content_type": content_type,
    }
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
