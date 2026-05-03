"""Fixture generation, staging, and fingerprinting for pipeline probes."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from devtools.pipeline_probe.request import PathFingerprint, SourceInputsSummary, StagedSourceEntry
from polylogue.scenarios import CorpusRequest
from polylogue.schemas.synthetic import SyntheticCorpus


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_path(path: Path) -> PathFingerprint:
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file",
            "sha256": _sha256_file(path),
            "file_count": 1,
            "total_bytes": path.stat().st_size,
        }

    if path.is_dir():
        digest = hashlib.sha256()
        file_count = 0
        total_bytes = 0
        for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = file_path.relative_to(path).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            with file_path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total_bytes += len(chunk)
            digest.update(b"\0")
            file_count += 1
        return {
            "path": str(path),
            "kind": "dir",
            "sha256": digest.hexdigest(),
            "file_count": file_count,
            "total_bytes": total_bytes,
        }

    raise FileNotFoundError(f"Cannot fingerprint missing path: {path}")


def _write_probe_sources(
    *,
    request: CorpusRequest,
    source_root: Path,
) -> tuple[list[Path], int]:
    scenarios = request.resolve_scenarios(
        origin="compiled.pipeline-probe",
        tags=("synthetic", "probe", "scenario"),
    )
    corpus_specs = tuple(spec for scenario in scenarios for spec in scenario.corpus_specs)
    source_root.mkdir(parents=True, exist_ok=True)
    provider = request.providers[0] if request.providers else "probe"
    written_batches = SyntheticCorpus.write_specs_artifacts(corpus_specs, source_root, prefix=provider, index_width=3)
    files = [path for written in written_batches for path in written.files]
    total_bytes = sum(len(artifact.raw_bytes) for written in written_batches for artifact in written.batch.artifacts)
    return files, total_bytes


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return 0


def _copy_source_subset_entry(*, source_path: Path, destination_path: Path) -> tuple[str, int, int]:
    if source_path.is_file():
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return "file", 1, destination_path.stat().st_size
    if source_path.is_dir():
        shutil.copytree(source_path, destination_path)
        staged_file_count = sum(1 for child in destination_path.rglob("*") if child.is_file())
        return "directory", staged_file_count, _path_size_bytes(destination_path)
    raise FileNotFoundError(f"source-subset probe input does not exist: {source_path}")


def _stage_source_subset(
    *,
    source_paths: list[Path],
    source_root: Path,
) -> SourceInputsSummary:
    source_root.mkdir(parents=True, exist_ok=True)
    entries: list[StagedSourceEntry] = []
    staged_file_count = 0
    total_bytes = 0

    for index, raw_source_path in enumerate(source_paths):
        source_path = raw_source_path.expanduser().resolve()
        destination_name = f"{index:03d}-{source_path.name or 'source'}"
        destination_path = source_root / destination_name
        entry_kind, entry_file_count, entry_bytes = _copy_source_subset_entry(
            source_path=source_path,
            destination_path=destination_path,
        )
        staged_file_count += entry_file_count
        total_bytes += entry_bytes
        entries.append(
            {
                "input_path": str(source_path),
                "staged_path": str(destination_path),
                "kind": entry_kind,
                "file_count": entry_file_count,
                "bytes": entry_bytes,
            }
        )

    return {
        "input_count": len(source_paths),
        "staged_entry_count": len(entries),
        "staged_file_count": staged_file_count,
        "total_bytes": total_bytes,
        "entries": entries,
    }


__all__ = [
    "_copy_source_subset_entry",
    "_fingerprint_path",
    "_path_size_bytes",
    "_sha256_file",
    "_stage_source_subset",
    "_write_probe_sources",
]
