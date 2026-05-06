"""Generated live-daemon convergence workloads for benchmark campaigns."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from polylogue.scenarios import CorpusSpec
from polylogue.schemas.synthetic import SyntheticCorpus


@dataclass(frozen=True, slots=True)
class DaemonLiveScaleSpec:
    files_per_provider: int
    messages_per_file: int


DAEMON_LIVE_SCALE_SPECS: dict[str, DaemonLiveScaleSpec] = {
    "small": DaemonLiveScaleSpec(files_per_provider=5, messages_per_file=10),
    "medium": DaemonLiveScaleSpec(files_per_provider=20, messages_per_file=25),
    "large": DaemonLiveScaleSpec(files_per_provider=50, messages_per_file=50),
    "stretch": DaemonLiveScaleSpec(files_per_provider=100, messages_per_file=100),
}


@dataclass(frozen=True, slots=True)
class DaemonLiveSourceRoot:
    name: str
    root: Path


@dataclass(frozen=True, slots=True)
class DaemonLiveGeneratedWorkload:
    sources: tuple[DaemonLiveSourceRoot, ...]
    files: list[Path]
    files_by_provider: dict[str, list[Path]]
    source_bytes: int
    message_count: int
    append_delta_bytes: int = 0


DAEMON_LIVE_PROVIDERS: tuple[str, ...] = ("claude-code", "codex")


def scale_from_db_path(db_path: Path) -> str:
    """Infer the generated benchmark scale from the archive directory name."""
    name = db_path.parent.name
    for scale in DAEMON_LIVE_SCALE_SPECS:
        if name.endswith(f"-{scale}") or name == scale:
            return scale
    return "small"


def generate_daemon_live_workload(root: Path, *, scale: str) -> DaemonLiveGeneratedWorkload:
    spec = DAEMON_LIVE_SCALE_SPECS.get(scale, DAEMON_LIVE_SCALE_SPECS["small"])
    sources: list[DaemonLiveSourceRoot] = []
    files: list[Path] = []
    files_by_provider: dict[str, list[Path]] = {}
    source_bytes = 0
    message_count = 0
    for provider_index, provider in enumerate(DAEMON_LIVE_PROVIDERS):
        source_root = root / "live-sources" / provider / "project"
        source_root.mkdir(parents=True, exist_ok=True)
        for stale_path in source_root.glob("*.jsonl"):
            stale_path.unlink()
        corpus_spec = CorpusSpec.for_provider(
            provider,
            count=spec.files_per_provider,
            messages_min=spec.messages_per_file,
            messages_max=spec.messages_per_file,
            seed=_seed_for_scale(scale) + provider_index * 10_000,
            style="tool-heavy",
        )
        batch = SyntheticCorpus.generate_batch_for_spec(corpus_spec)
        provider_files: list[Path] = []
        for file_index, artifact in enumerate(batch.artifacts):
            session_id = f"daemon-live-{provider}-{scale}-{file_index:04d}"
            path = source_root / f"{session_id}.jsonl"
            path.write_bytes(_jsonl_bytes_with_terminal_newline(artifact.raw_bytes))
            files.append(path)
            provider_files.append(path)
            source_bytes += path.stat().st_size
            message_count += artifact.message_count
        files_by_provider[provider] = provider_files
        sources.append(DaemonLiveSourceRoot(name=provider, root=source_root.parent))
    return DaemonLiveGeneratedWorkload(
        sources=tuple(sources),
        files=files,
        files_by_provider=files_by_provider,
        source_bytes=source_bytes,
        message_count=message_count,
    )


def append_daemon_live_workload(
    workload: DaemonLiveGeneratedWorkload, *, message_index: int
) -> DaemonLiveGeneratedWorkload:
    """Append one generated JSONL record per file and return updated workload stats."""
    append_delta_bytes = 0
    for provider, paths in workload.files_by_provider.items():
        for file_index, path in enumerate(paths):
            record = _make_generated_append_record(
                provider,
                path,
                message_index=message_index,
                seed=file_index,
            )
            before = path.stat().st_size
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True))
                handle.write("\n")
            append_delta_bytes += path.stat().st_size - before
    return DaemonLiveGeneratedWorkload(
        sources=workload.sources,
        files=workload.files,
        files_by_provider=workload.files_by_provider,
        source_bytes=sum(path.stat().st_size for path in workload.files),
        message_count=workload.message_count + len(workload.files),
        append_delta_bytes=append_delta_bytes,
    )


def _seed_for_scale(scale: str) -> int:
    return 845_000 + sum(ord(char) for char in scale)


def _jsonl_bytes_with_terminal_newline(raw_bytes: bytes) -> bytes:
    return raw_bytes if raw_bytes.endswith(b"\n") else raw_bytes + b"\n"


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"daemon-live source contains non-object record: {path}")
        records.append(value)
    if not records:
        raise ValueError(f"daemon-live source has no JSONL records: {path}")
    return records


def _make_generated_append_record(
    provider: str,
    path: Path,
    *,
    message_index: int,
    seed: int,
) -> dict[str, object]:
    existing_records = _read_jsonl_records(path)
    corpus_spec = CorpusSpec.for_provider(
        provider,
        count=1,
        messages_min=1,
        messages_max=1,
        seed=945_000 + message_index * 10_000 + seed,
        style="tool-heavy",
    )
    artifact = SyntheticCorpus.generate_batch_for_spec(corpus_spec).artifacts[0]
    record = _read_generated_record(artifact.raw_bytes, path)
    role = "user" if message_index % 2 == 0 else "assistant"
    record["timestamp"] = f"2026-05-05T01:{message_index // 60:02d}:{message_index % 60:02d}.000Z"
    if provider == "codex":
        record["role"] = role
        record["id"] = f"{path.stem}-append-{message_index:04d}"
        return record
    last_record = existing_records[-1]
    session_id = last_record.get("sessionId")
    parent_uuid = last_record.get("uuid")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError(f"daemon-live source lacks a sessionId: {path}")
    if not isinstance(parent_uuid, str) or not parent_uuid:
        raise ValueError(f"daemon-live source lacks a parent uuid: {path}")
    record["sessionId"] = session_id
    record["parentUuid"] = parent_uuid
    record["uuid"] = f"{session_id}-append-{message_index:04d}"
    record["type"] = role
    message = record.get("message")
    if not isinstance(message, dict):
        message = {}
        record["message"] = message
    message["role"] = role
    return record


def _read_generated_record(raw_bytes: bytes, path: Path) -> dict[str, object]:
    lines = [line for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one generated append record for {path}, got {len(lines)}")
    value = json.loads(lines[0])
    if not isinstance(value, dict):
        raise ValueError(f"generated append record is not a JSON object for {path}")
    return value


async def run_daemon_live_convergence_workload(db_path: Path) -> tuple[dict[str, float], dict[str, int]]:
    """Run live batch ingestion in-process and return normalized metrics/stats."""
    from polylogue.api import Polylogue
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.daemon.convergence_stages import make_default_convergence_stages
    from polylogue.sources.live.batch import LiveBatchProcessor
    from polylogue.sources.live.cursor import CursorStore
    from polylogue.sources.live.watcher import WatchSource

    scale = scale_from_db_path(db_path)
    workload = generate_daemon_live_workload(db_path.parent, scale=scale)
    converger = DaemonConverger(stages=make_default_convergence_stages(db_path), max_workers=2)
    await converger.start()
    try:
        async with Polylogue(archive_root=db_path.parent, db_path=db_path) as polylogue:
            processor = LiveBatchProcessor(
                polylogue,
                tuple(WatchSource(name=source.name, root=source.root) for source in workload.sources),
                cursor=CursorStore(db_path),
                parser_fingerprint="daemon-live-benchmark-v1",
                converger=converger,
            )
            initial_metrics = await processor.ingest_files(workload.files, emit_event=False)
            workload = append_daemon_live_workload(
                workload,
                message_index=DAEMON_LIVE_SCALE_SPECS.get(scale, DAEMON_LIVE_SCALE_SPECS["small"]).messages_per_file,
            )
            append_metrics = await processor.ingest_files(workload.files, emit_event=False)
    finally:
        await converger.stop()

    total_wall_s = initial_metrics.total_time_s + append_metrics.total_time_s
    files_per_s = (
        (initial_metrics.succeeded_file_count + append_metrics.succeeded_file_count) / total_wall_s
        if total_wall_s > 0
        else 0.0
    )
    messages_per_s = (workload.message_count / total_wall_s) if total_wall_s > 0 else 0.0
    normalized_metrics = {
        "total_wall_s": total_wall_s,
        "parse_wall_s": initial_metrics.parse_time_s + append_metrics.parse_time_s,
        "convergence_wall_s": initial_metrics.convergence_time_s + append_metrics.convergence_time_s,
        "files_per_s": round(files_per_s, 3),
        "messages_per_s": round(messages_per_s, 3),
        "archive_write_bytes_delta": float(
            initial_metrics.archive_write_bytes_delta + append_metrics.archive_write_bytes_delta
        ),
        "payload_read_bytes": float(
            initial_metrics.source_payload_read_bytes + append_metrics.source_payload_read_bytes
        ),
        "initial_payload_read_bytes": float(initial_metrics.source_payload_read_bytes),
        "append_payload_read_bytes": float(append_metrics.source_payload_read_bytes),
        "append_input_bytes": float(append_metrics.input_bytes),
        "cursor_fingerprint_read_bytes": float(
            initial_metrics.cursor_fingerprint_read_bytes + append_metrics.cursor_fingerprint_read_bytes
        ),
        "ingest_worker_count_max": float(
            max(initial_metrics.ingest_worker_count_max, append_metrics.ingest_worker_count_max)
        ),
        "succeeded_files": float(initial_metrics.succeeded_file_count + append_metrics.succeeded_file_count),
        "failed_files": float(initial_metrics.failed_file_count + append_metrics.failed_file_count),
        "append_files": float(append_metrics.append_file_count),
        "full_files": float(initial_metrics.full_file_count + append_metrics.full_file_count),
    }
    normalized_metrics.update(
        {f"stage_initial_{name}_wall_s": elapsed for name, elapsed in initial_metrics.stage_timings_s.items()}
    )
    normalized_metrics.update(
        {f"stage_append_{name}_wall_s": elapsed for name, elapsed in append_metrics.stage_timings_s.items()}
    )
    db_stats = {
        "files_count": len(workload.files),
        "claude_code_files": len(workload.files_by_provider.get("claude-code", ())),
        "codex_files": len(workload.files_by_provider.get("codex", ())),
        "source_bytes": workload.source_bytes,
        "append_delta_bytes": workload.append_delta_bytes,
        "messages_count": workload.message_count,
    }
    return normalized_metrics, db_stats


__all__ = [
    "DAEMON_LIVE_SCALE_SPECS",
    "DAEMON_LIVE_PROVIDERS",
    "DaemonLiveGeneratedWorkload",
    "DaemonLiveScaleSpec",
    "DaemonLiveSourceRoot",
    "append_daemon_live_workload",
    "generate_daemon_live_workload",
    "run_daemon_live_convergence_workload",
    "scale_from_db_path",
]
