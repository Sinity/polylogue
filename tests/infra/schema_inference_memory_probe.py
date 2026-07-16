"""Subprocess target for production schema-inference memory scaling tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _payload(index: int) -> bytes:
    document = {
        "title": f"synthetic conversation {index}",
        "conversation_id": f"conversation-{index}",
        "mapping": {
            f"node-{index}": {
                "id": f"node-{index}",
                "message": {
                    "author": {"role": "user" if index % 2 == 0 else "assistant"},
                    "content": {"content_type": "text", "parts": [f"message {index}"]},
                },
            }
        },
    }
    return json.dumps(document, sort_keys=True).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args(argv)

    archive_root = args.archive_root.resolve()
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["XDG_CACHE_HOME"] = str(archive_root.parent / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(archive_root.parent / "config")
    os.environ["XDG_DATA_HOME"] = str(archive_root.parent / "data")
    os.environ["XDG_STATE_HOME"] = str(archive_root.parent / "state")

    from polylogue.core.enums import Provider
    from polylogue.schemas.generation.workflow import generate_provider_schema
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        for index in range(args.count):
            archive.write_raw_payload(
                provider=Provider.CHATGPT,
                payload=_payload(index),
                source_path=f"/synthetic/chatgpt/session-{index}.json",
                acquired_at_ms=1_700_000_000_000 + index,
                raw_id=f"synthetic-raw-{index}",
            )

    print("READY", flush=True)
    if not sys.stdin.readline():
        return 2

    result = generate_provider_schema(
        "chatgpt",
        db_path=archive_root / "index.db",
        full_corpus=True,
    )
    journal_root = Path(os.environ["XDG_CACHE_HOME"]) / "polylogue" / "schema-observation-journals"
    print(
        json.dumps(
            {
                "success": result.success,
                "error": result.error,
                "sample_count": result.sample_count,
                "cluster_count": result.cluster_count,
                "journal_remaining": sorted(path.name for path in journal_root.glob("run-*.sqlite3*")),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
