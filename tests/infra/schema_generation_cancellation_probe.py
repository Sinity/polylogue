"""Subprocess target for the schema-generation cancellation-equivalence proof.

``devtools/schema_generate.py`` reports its own resume semantics as
``restart_from_acquisition``: a killed run's private ``ObservationJournal`` is
always removed, so there is no partial-resume state to diverge from a fresh
run. This script lets a test prove that claim empirically against the real
``generate_provider_schema`` entrypoint (not just the raw journal cleanup
contract already covered by ``test_journal_cleanup_runs_for_real_terminated_process``):

1. Write a small deterministic synthetic ChatGPT archive once (idempotent
   across repeated invocations against the same ``--archive-root``).
2. Optionally pause right after the ``observe_and_cluster`` phase starts
   (``--pause-after-observe``) so the parent can send SIGTERM mid-run and
   confirm the journal directory is empty afterward.
3. Run generation to completion and print the resulting schema (minus the
   wall-clock-dependent ``x-polylogue-generated-at`` field) as JSON, so the
   parent can diff an uninterrupted reference run against an
   interrupted-then-restarted run on the same archive.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--pause-after-observe", action="store_true")
    args = parser.parse_args(argv)

    archive_root = args.archive_root.resolve()
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["XDG_CACHE_HOME"] = str(archive_root.parent / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(archive_root.parent / "config")
    os.environ["XDG_DATA_HOME"] = str(archive_root.parent / "data")
    os.environ["XDG_STATE_HOME"] = str(archive_root.parent / "state")

    from polylogue.core.enums import Provider
    from polylogue.core.json import JSONDocument
    from polylogue.schemas.generation.workflow import generate_provider_schema
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    if not (archive_root / "index.db").exists():
        with ArchiveStore(archive_root) as archive:
            for index in range(args.count):
                archive.write_raw_payload(
                    provider=Provider.CHATGPT,
                    payload=_payload(index),
                    source_path=f"/synthetic/chatgpt/session-{index}.json",
                    acquired_at_ms=1_700_000_000_000 + index,
                    raw_id=f"synthetic-raw-{index}",
                )

    def on_progress(phase: str, payload: JSONDocument) -> None:
        if args.pause_after_observe and phase == "observe_and_cluster" and payload.get("state") == "started":
            print("PAUSED", flush=True)
            time.sleep(10)

    result = generate_provider_schema(
        "chatgpt",
        db_path=archive_root / "index.db",
        progress_callback=on_progress if args.pause_after_observe else None,
    )
    schema = dict(result.schema or {})
    schema.pop("x-polylogue-generated-at", None)
    print(
        json.dumps(
            {
                "success": result.success,
                "error": result.error,
                "sample_count": result.sample_count,
                "cluster_count": result.cluster_count,
                "package_count": result.package_count,
                "default_version": result.default_version,
                "schema": schema,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
