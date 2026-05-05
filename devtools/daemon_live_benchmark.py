"""Generated live-daemon convergence workloads for benchmark campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DaemonLiveScaleSpec:
    files: int
    messages_per_file: int


DAEMON_LIVE_SCALE_SPECS: dict[str, DaemonLiveScaleSpec] = {
    "small": DaemonLiveScaleSpec(files=5, messages_per_file=10),
    "medium": DaemonLiveScaleSpec(files=20, messages_per_file=25),
    "large": DaemonLiveScaleSpec(files=50, messages_per_file=50),
    "stretch": DaemonLiveScaleSpec(files=100, messages_per_file=100),
}


@dataclass(frozen=True, slots=True)
class DaemonLiveGeneratedWorkload:
    root: Path
    files: list[Path]
    source_bytes: int
    message_count: int
    append_delta_bytes: int = 0


def scale_from_db_path(db_path: Path) -> str:
    """Infer the generated benchmark scale from the archive directory name."""
    name = db_path.parent.name
    for scale in DAEMON_LIVE_SCALE_SPECS:
        if name.endswith(f"-{scale}") or name == scale:
            return scale
    return "small"


def generate_daemon_live_workload(root: Path, *, scale: str) -> DaemonLiveGeneratedWorkload:
    spec = DAEMON_LIVE_SCALE_SPECS.get(scale, DAEMON_LIVE_SCALE_SPECS["small"])
    source_root = root / "live-sources" / "claude-code" / "project"
    source_root.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    source_bytes = 0
    message_count = 0
    for file_index in range(spec.files):
        session_id = f"daemon-live-{scale}-{file_index:04d}"
        path = source_root / f"{session_id}.jsonl"
        records = _make_claude_code_session(session_id, spec.messages_per_file)
        _write_jsonl(path, records)
        files.append(path)
        source_bytes += path.stat().st_size
        message_count += len(records)
    return DaemonLiveGeneratedWorkload(
        root=source_root.parent,
        files=files,
        source_bytes=source_bytes,
        message_count=message_count,
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _make_claude_code_session(session_id: str, message_count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(message_count):
        role = "user" if index % 2 == 0 else "assistant"
        records.append(
            {
                "parentUuid": None if index == 0 else f"{session_id}-msg-{index - 1:04d}",
                "sessionId": session_id,
                "type": role,
                "message": {
                    "role": role,
                    "content": f"Daemon live benchmark message {index} for {session_id}. "
                    f"This generated payload keeps parser work deterministic.",
                },
                "uuid": f"{session_id}-msg-{index:04d}",
                "timestamp": f"2026-05-05T00:{index // 60:02d}:{index % 60:02d}.000Z",
                "cwd": "/realm/project/polylogue",
                "version": "1.0.0",
                "isSidechain": False,
                "userType": "external",
            }
        )
    return records


async def run_daemon_live_convergence_workload(db_path: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Run live batch ingestion in-process and return normalized metrics/stats."""
    from polylogue.api import Polylogue
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_embed_stage, make_fts_stage, make_insights_stage
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import WatchSource

    scale = scale_from_db_path(db_path)
    workload = generate_daemon_live_workload(db_path.parent, scale=scale)
    converger = DaemonConverger(
        stages=(
            make_fts_stage(db_path),
            make_embed_stage(db_path),
            make_insights_stage(db_path),
        ),
        max_workers=2,
    )
    await converger.start()
    try:
        async with Polylogue(archive_root=db_path.parent, db_path=db_path) as polylogue:
            processor = LiveBatchProcessor(
                polylogue,
                (WatchSource(name="claude-code", root=workload.root),),
                cursor=CursorStore(db_path),
                parser_fingerprint="daemon-live-benchmark-v1",
                converger=converger,
            )
            metrics = await processor.ingest_files(workload.files, emit_event=False)
    finally:
        await converger.stop()

    total_wall_s = metrics.total_time_s
    files_per_s = (metrics.succeeded_file_count / total_wall_s) if total_wall_s > 0 else 0.0
    messages_per_s = (workload.message_count / total_wall_s) if total_wall_s > 0 else 0.0
    normalized_metrics = {
        "total_wall_s": total_wall_s,
        "parse_wall_s": metrics.parse_time_s,
        "convergence_wall_s": metrics.convergence_time_s,
        "files_per_s": round(files_per_s, 3),
        "messages_per_s": round(messages_per_s, 3),
        "archive_write_bytes_delta": float(metrics.archive_write_bytes_delta),
        "payload_read_bytes": float(metrics.input_bytes),
        "cursor_fingerprint_read_bytes": float(metrics.cursor_fingerprint_read_bytes),
        "succeeded_files": float(metrics.succeeded_file_count),
        "failed_files": float(metrics.failed_file_count),
    }
    normalized_metrics.update({f"stage_{name}_wall_s": elapsed for name, elapsed in metrics.stage_timings_s.items()})
    db_stats = {
        "files_count": len(workload.files),
        "source_bytes": workload.source_bytes,
        "append_delta_bytes": workload.append_delta_bytes,
        "messages_count": workload.message_count,
    }
    return normalized_metrics, db_stats


__all__ = [
    "DAEMON_LIVE_SCALE_SPECS",
    "DaemonLiveGeneratedWorkload",
    "DaemonLiveScaleSpec",
    "generate_daemon_live_workload",
    "run_daemon_live_convergence_workload",
    "scale_from_db_path",
]
