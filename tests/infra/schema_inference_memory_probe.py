"""Subprocess target for production schema-inference memory scaling tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tracemalloc
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


def _codex_payload(record_count: int) -> bytes:
    return "\n".join(
        json.dumps(
            {"type": "session_meta", "payload": {"id": "synthetic-session"}}
            if index == 0
            else {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user" if index % 2 else "assistant",
                    "content": [{"type": "input_text", "text": f"message-{index}"}],
                },
            },
            sort_keys=True,
        )
        for index in range(record_count)
    ).encode("utf-8")


def _resource_snapshot(*, phase: str, journal_root: Path) -> dict[str, int | str | None]:
    smaps: dict[str, int] = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            key, separator, raw = line.partition(":")
            if separator and key in {"Pss", "Pss_Anon", "Pss_File", "SwapPss"}:
                smaps[key] = int(raw.split()[0])
    except (OSError, ValueError, IndexError):
        pass
    current, peak = tracemalloc.get_traced_memory()
    journal_bytes = sum(path.stat().st_size for path in journal_root.glob("run-*.sqlite3*") if path.is_file())
    return {
        "phase": phase,
        "pss_kb": smaps.get("Pss"),
        "anon_pss_kb": smaps.get("Pss_Anon"),
        "file_pss_kb": smaps.get("Pss_File"),
        "swap_pss_kb": smaps.get("SwapPss"),
        "tracemalloc_current_bytes": current,
        "tracemalloc_peak_bytes": peak,
        "journal_bytes": journal_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--provider", choices=("chatgpt", "codex"), default="chatgpt")
    parser.add_argument("--record-count", type=int, default=None)
    args = parser.parse_args(argv)

    archive_root = args.archive_root.resolve()
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    os.environ["XDG_CACHE_HOME"] = str(archive_root.parent / "cache")
    os.environ["XDG_CONFIG_HOME"] = str(archive_root.parent / "config")
    os.environ["XDG_DATA_HOME"] = str(archive_root.parent / "data")
    os.environ["XDG_STATE_HOME"] = str(archive_root.parent / "state")

    from polylogue.core.enums import Provider
    from polylogue.schemas.generation import provider_bundle
    from polylogue.schemas.generation.workflow import generate_provider_schema
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore(archive_root) as archive:
        provider = Provider.CODEX if args.provider == "codex" else Provider.CHATGPT
        for index in range(args.count):
            archive.write_raw_payload(
                provider=provider,
                payload=_codex_payload(args.record_count) if args.record_count is not None else _payload(index),
                source_path=f"/synthetic/{args.provider}/session-{index}.{'jsonl' if provider is Provider.CODEX else 'json'}",
                acquired_at_ms=1_700_000_000_000 + index,
                raw_id=f"synthetic-raw-{index}",
            )

    print("READY", flush=True)
    if not sys.stdin.readline():
        return 2

    journal_root = Path(os.environ["XDG_CACHE_HOME"]) / "polylogue" / "schema-observation-journals"
    snapshots: list[dict[str, int | str | None]] = []
    tracemalloc.start()
    snapshots.append(_resource_snapshot(phase="before-provider-bundle", journal_root=journal_root))
    for name in ("_collect_cluster_accumulators", "_build_package_candidates", "build_provider_catalog_artifacts"):
        original = getattr(provider_bundle, name)

        def wrapped(
            *call_args: object, __name: str = name, __original: object = original, **call_kwargs: object
        ) -> object:
            snapshots.append(_resource_snapshot(phase=f"before-{__name}", journal_root=journal_root))
            result = __original(*call_args, **call_kwargs)  # type: ignore[operator]
            snapshots.append(_resource_snapshot(phase=f"after-{__name}", journal_root=journal_root))
            return result

        setattr(provider_bundle, name, wrapped)
    try:
        result = generate_provider_schema(
            args.provider,
            db_path=archive_root / "index.db",
            full_corpus=True,
        )
        snapshots.append(_resource_snapshot(phase="after-provider-bundle", journal_root=journal_root))
    finally:
        tracemalloc.stop()
    print(
        json.dumps(
            {
                "success": result.success,
                "error": result.error,
                "sample_count": result.sample_count,
                "cluster_count": result.cluster_count,
                "journal_remaining": sorted(path.name for path in journal_root.glob("run-*.sqlite3*")),
                "phases": snapshots,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
