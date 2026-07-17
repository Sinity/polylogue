"""Subprocess target for production schema-inference memory scaling tests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast


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


def _process_io_counters() -> dict[str, int | None]:
    """Return Linux process I/O counters without guessing from wall time."""
    counters: dict[str, int | None] = {
        "rchar": None,
        "wchar": None,
        "read_bytes": None,
        "write_bytes": None,
    }
    try:
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            key, separator, raw_value = line.partition(":")
            if separator and key in counters:
                counters[key] = int(raw_value.strip())
    except (OSError, ValueError):
        pass
    return counters


def _journal_storage_bytes(journal_root: Path) -> dict[str, int]:
    """Keep journal database and WAL growth separately observable."""
    totals = {"journal_db_bytes": 0, "journal_wal_bytes": 0, "journal_shm_bytes": 0}
    for path in journal_root.glob("run-*.sqlite3*"):
        if not path.is_file():
            continue
        if path.name.endswith("-wal"):
            totals["journal_wal_bytes"] += path.stat().st_size
        elif path.name.endswith("-shm"):
            totals["journal_shm_bytes"] += path.stat().st_size
        else:
            totals["journal_db_bytes"] += path.stat().st_size
    return totals


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
    journal_storage = _journal_storage_bytes(journal_root)
    return {
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
        "pss_kb": smaps.get("Pss"),
        "anon_pss_kb": smaps.get("Pss_Anon"),
        "file_pss_kb": smaps.get("Pss_File"),
        "swap_pss_kb": smaps.get("SwapPss"),
        "tracemalloc_current_bytes": current,
        "tracemalloc_peak_bytes": peak,
        "journal_bytes": sum(journal_storage.values()),
        **journal_storage,
        **_process_io_counters(),
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
    from polylogue.schemas.generation import observation_journal, provider_bundle
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
    method_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "wall_ns": 0, "yielded": 0})
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

    def wrap_journal_method(name: str, *, iterable: bool = False) -> None:
        original = getattr(observation_journal.ObservationJournal, name)

        def wrapped(self: object, *call_args: object, **call_kwargs: object) -> object:
            started_ns = time.monotonic_ns()
            result = original(self, *call_args, **call_kwargs)
            if not iterable:
                metric = method_metrics[name]
                metric["calls"] += 1
                metric["wall_ns"] += time.monotonic_ns() - started_ns
                return result

            def measured() -> Iterator[object]:
                yielded = 0
                try:
                    for item in cast(Iterable[object], result):
                        yielded += 1
                        yield item
                finally:
                    metric = method_metrics[name]
                    metric["calls"] += 1
                    metric["yielded"] += yielded
                    metric["wall_ns"] += time.monotonic_ns() - started_ns

            return measured()

        setattr(observation_journal.ObservationJournal, name, wrapped)

    for method_name in ("append_unit", "flush", "assign_canonical_package_families"):
        wrap_journal_method(method_name)
    for method_name in ("_iter_joined_memberships", "iter_distinct_membership_values"):
        wrap_journal_method(method_name, iterable=True)
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
                "phase_receipt": result.phase_receipt,
                "journal_remaining": sorted(path.name for path in journal_root.glob("run-*.sqlite3*")),
                "phases": snapshots,
                "journal_method_metrics": dict(method_metrics),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
