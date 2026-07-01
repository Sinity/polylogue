#!/usr/bin/env python3
"""Keep the active conductor packet small without deleting local history."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

HEADING_RE = re.compile(r"^## \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [A-Z]+ [—-] ", re.M)


def split_operating_log(text: str) -> tuple[str, list[str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return text, []
    preamble = text[: matches[0].start()].rstrip() + "\n\n"
    entries: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entries.append(text[match.start() : end].strip() + "\n")
    return preamble, entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/realm/project/polylogue/.agent/conductor-devloop",
        help="conductor-devloop directory",
    )
    parser.add_argument(
        "--keep-entries",
        type=int,
        default=160,
        help="number of recent operating-log entries to keep active",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=262_144,
        help="active OPERATING-LOG.md size threshold before compaction",
    )
    parser.add_argument("--check", action="store_true", help="fail if compaction would be needed")
    args = parser.parse_args()

    root = Path(args.root)
    log_path = root / "OPERATING-LOG.md"
    if not log_path.exists():
        print(f"missing operating log: {log_path}")
        return 1

    text = log_path.read_text(encoding="utf-8")
    size = len(text.encode("utf-8"))
    preamble, entries = split_operating_log(text)
    needs_compaction = size > args.max_bytes and len(entries) > args.keep_entries

    if not needs_compaction:
        print(f"conductor log compact enough: bytes={size} entries={len(entries)}")
        return 0

    if args.check:
        print(f"conductor log needs compaction: bytes={size} entries={len(entries)} keep_entries={args.keep_entries}")
        return 1

    archive_dir = root.parent / "archive" / "conductor-history"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_entries = entries[: -args.keep_entries]
    kept_entries = entries[-args.keep_entries :]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    archive_path = archive_dir / "OPERATING-LOG.archive.md"

    with archive_path.open("a", encoding="utf-8") as handle:
        if archive_path.stat().st_size == 0:
            handle.write("# Archived Polylogue Operating Log Entries\n\n")
        handle.write(f"\n<!-- compacted {timestamp}; moved {len(archived_entries)} entries from {log_path} -->\n\n")
        handle.write("\n".join(entry.strip() for entry in archived_entries).strip())
        handle.write("\n")

    active_note = (
        "This is the rolling active window. Older local entries are preserved in "
        "`.agent/archive/conductor-history/OPERATING-LOG.archive.md`.\n\n"
    )
    if active_note not in preamble:
        preamble = preamble.rstrip() + "\n\n" + active_note
    log_path.write_text(preamble + "\n".join(entry.strip() for entry in kept_entries).strip() + "\n", encoding="utf-8")

    new_size = log_path.stat().st_size
    print(
        "compacted conductor log: "
        f"archived={len(archived_entries)} kept={len(kept_entries)} "
        f"bytes_before={size} bytes_after={new_size} archive={archive_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
