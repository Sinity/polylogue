"""Receipt-backed, real-pipeline seeded archive artifacts for composition tests.

This is deliberately an adapter over production corpus generation, ingestion,
archive tiers, and workload receipts.  It owns no alternate generator, query
language, or semantic profile.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from polylogue.config import Config, Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusSpec
from polylogue.scenarios.workload import (
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SyntheticArtifactFacts
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

_ARTIFACT_PROTOCOL_VERSION = 1
_CACHE_ROOT = Path("/realm/tmp/polylogue-seeded-artifacts")
_RECIPE_PATHS = (
    Path("polylogue/schemas/synthetic/build_batch.py"),
    Path("polylogue/schemas/synthetic/core.py"),
    Path("polylogue/pipeline/services/archive_ingest.py"),
    Path("polylogue/storage/sqlite/archive_tiers/bootstrap.py"),
    Path("polylogue/storage/raw_reconciler.py"),
    Path("polylogue/storage/archive_readiness.py"),
)
_ARCHIVE_DB_NAMES = ("source.db", "index.db", "user.db", "ops.db", "embeddings.db")


@dataclass(frozen=True)
class SeededArchiveKey:
    spec_payload: dict[str, object]
    build_id: str
    recipe_id: str

    @property
    def value(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return f"seeded-archive:sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class SeededArchiveManifest:
    protocol_version: int
    key: str
    archive_id: str
    profile_id: str
    build_id: str
    recipe_id: str
    facts: tuple[SyntheticArtifactFacts, ...]
    files: tuple[dict[str, object], ...]
    receipt: dict[str, object]

    @property
    def manifest_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return f"seeded-manifest:sha256:{hashlib.sha256(payload).hexdigest()}"

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["manifest_id"] = self.manifest_id
        return payload


@dataclass(frozen=True)
class SeededArchiveArtifact:
    root: Path
    manifest: SeededArchiveManifest

    @property
    def facts(self) -> tuple[SyntheticArtifactFacts, ...]:
        return self.manifest.facts


@dataclass(frozen=True)
class SeededArchiveClone:
    root: Path
    source_manifest_id: str
    clone_method: str


def c03_semantic_corpus_spec() -> CorpusSpec:
    """Smallest named semantic canary with a pinned selective Codex session."""
    count = 64
    native_ids = ("c03-target", *(f"c03-irrelevant-{index:03d}" for index in range(count - 1)))
    return CorpusSpec.for_provider(
        "codex",
        count=count,
        messages_min=4,
        messages_max=4,
        seed=71,
        style="tool-heavy",
        session_native_ids=native_ids,
        origin="generated.test-workload-c03",
        tags=("synthetic", "test", "workload-c03"),
    )


def schema_coverage_corpus_specs() -> tuple[CorpusSpec, ...]:
    """Named all-provider schema workload; no caller chooses ad-hoc shape."""
    return tuple(
        CorpusSpec.for_provider(
            provider,
            count=2,
            messages_min=4,
            messages_max=4,
            seed=42,
            origin="generated.test-schema-coverage",
            tags=("synthetic", "test", "schema-coverage"),
        )
        for provider in SyntheticCorpus.available_providers()
    )


def named_corpus_specs(name: str) -> tuple[CorpusSpec, ...]:
    """Resolve the finite shared workload catalog used by test consumers."""
    profiles: dict[str, tuple[tuple[str, int], ...]] = {
        "schema-small": (("chatgpt", 10),),
        "schema-medium": (("chatgpt", 50),),
        "cli-chatgpt": (("chatgpt", 2),),
        "cli-mixed": (("chatgpt", 2), ("claude-code", 2)),
        "completion": (("chatgpt", 3), ("claude-ai", 3)),
    }
    selected = profiles.get(name)
    if selected is None:
        raise ValueError(f"unknown named seeded archive workload {name!r}")
    return tuple(
        CorpusSpec.for_provider(
            provider,
            count=count,
            messages_min=4,
            messages_max=11,
            seed=1271 if name == "completion" else 42,
            origin=f"generated.test-workload-{name}",
            tags=("synthetic", "test", name),
        )
        for provider, count in selected
    )


def _recipe_id() -> str:
    digest = hashlib.sha256()
    for path in _RECIPE_PATHS:
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"recipe:sha256:{digest.hexdigest()}"


def _build_id() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "git:unavailable"
    return f"git:{result.stdout.strip()}"


def _archive_build_spec(
    *,
    key: SeededArchiveKey,
    archive_id: str,
    profile_id: str,
) -> WorkloadEnvelopeSpec:
    """Declare the artifact's production generation route, separately from C-03."""
    return WorkloadEnvelopeSpec(
        workload_id="seeded-archive:production-build",
        family_id="schema-workload-artifact",
        version=_ARTIFACT_PROTOCOL_VERSION,
        inputs=(
            WorkloadInputRef(
                input_id=key.value,
                corpus_id=archive_id,
                profile_id=profile_id,
            ),
        ),
        phases=(
            "generate",
            "acquire",
            "parse",
            "materialize",
            "index",
            "raw_authority_frontier",
            "validate",
            "publish",
        ),
    )


def _profile_id(key: SeededArchiveKey) -> str:
    payload = json.dumps(key.spec_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return f"workload-profile:sha256:{hashlib.sha256(payload).hexdigest()}"


def seeded_archive_key(specs: Iterable[CorpusSpec]) -> SeededArchiveKey:
    return SeededArchiveKey(
        spec_payload={"corpus_specs": [spec.to_payload() for spec in specs]},
        build_id=_build_id(),
        recipe_id=_recipe_id(),
    )


@contextlib.contextmanager
def _configured_archive_root(root: Path) -> Iterator[None]:
    previous = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = previous


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_files(root: Path) -> tuple[dict[str, object], ...]:
    entries = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name in {"manifest.json", ".build.lock"}:
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return tuple(entries)


def _sqlite_integrity(root: Path) -> None:
    checked: list[str] = []
    for name in _ARCHIVE_DB_NAMES:
        path = root / name
        if not path.exists():
            continue
        with sqlite3.connect(path) as conn:
            quick = conn.execute("PRAGMA quick_check").fetchone()
            foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
            if quick != ("ok",) or foreign:
                raise RuntimeError(f"invalid seeded archive tier {name}")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
            if mode != ("delete",):
                raise RuntimeError(f"could not close seeded archive tier {name} into a snapshot")
        checked.append(name)
    for name in checked:
        for suffix in ("-wal", "-shm"):
            sidecar = root / f"{name}{suffix}"
            if sidecar.exists():
                sidecar.unlink()


def _remove_tree(path: Path) -> None:
    """Remove a locally-owned stale artifact even after immutable publication."""
    if not path.exists():
        return
    for candidate in sorted(path.rglob("*"), reverse=True):
        candidate.chmod(candidate.stat().st_mode | stat.S_IWUSR)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def _validate_facts(root: Path, facts: tuple[SyntheticArtifactFacts, ...]) -> None:
    with sqlite3.connect(root / "index.db") as conn:
        session_ids = {str(row[0]) for row in conn.execute("SELECT session_id FROM sessions")}
        tool_ids = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT tool_id FROM blocks WHERE tool_id IS NOT NULL")
        }
    for fact in facts:
        if fact.expected_session_id is not None and fact.expected_session_id not in session_ids:
            raise RuntimeError(f"missing planted session {fact.expected_session_id}")
        if not set(fact.tool_use_ids) <= tool_ids:
            raise RuntimeError(f"missing planted tool action for {fact.expected_session_id}")


def _validate_frontier_convergence(root: Path) -> None:
    """Require a published artifact to be query-ready, not merely ingested."""
    readiness = raw_materialization_readiness_snapshot(root)
    if not raw_materialization_ready(readiness):
        raise RuntimeError("seeded archive is missing completed raw-authority frontier convergence")


def _read_manifest(path: Path) -> SeededArchiveManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    facts = tuple(SyntheticArtifactFacts(**item) for item in payload.pop("facts"))
    payload.pop("manifest_id", None)
    return SeededArchiveManifest(facts=facts, **payload)


def _validate_artifact(root: Path, key: SeededArchiveKey) -> SeededArchiveArtifact | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _read_manifest(manifest_path)
        if manifest.protocol_version != _ARTIFACT_PROTOCOL_VERSION or manifest.key != key.value:
            return None
        for item in manifest.files:
            path = root / str(item["path"])
            if not path.is_file() or path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
                return None
        _sqlite_integrity(root)
        _validate_facts(root, manifest.facts)
        _validate_frontier_convergence(root)
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, sqlite3.Error):
        return None
    return SeededArchiveArtifact(root=root, manifest=manifest)


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        else:
            path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def build_seeded_archive(
    specs: Iterable[CorpusSpec] | None = None,
    *,
    cache_root: Path | None = None,
) -> SeededArchiveArtifact:
    """Build-or-reuse one atomic immutable real-pipeline archive artifact."""
    selected_specs = tuple(specs) if specs is not None else (c03_semantic_corpus_spec(),)
    if not selected_specs:
        raise ValueError("seeded archive requires at least one named corpus specification")
    key = seeded_archive_key(selected_specs)
    cache_root = (cache_root or _CACHE_ROOT).expanduser()
    artifacts = cache_root / "artifacts"
    locks = cache_root / ".locks"
    staging_root = cache_root / ".staging"
    artifacts.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    final_root = artifacts / key.value.rsplit(":", 1)[-1]
    lock_path = locks / f"{final_root.name}.lock"

    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        cached = _validate_artifact(final_root, key)
        if cached is not None:
            return cached
        if final_root.exists():
            _remove_tree(final_root)
        staging = staging_root / f"{final_root.name}.{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            corpus_root = staging / "wire"
            written_batches = tuple(
                SyntheticCorpus.write_spec_artifacts(spec, corpus_root / spec.provider, prefix=f"seed-{index:02d}")
                for index, spec in enumerate(selected_specs)
            )
            sources = [
                Source(name=spec.provider, path=path)
                for spec, written in zip(selected_specs, written_batches, strict=True)
                for path in written.files
            ]
            with _configured_archive_root(staging):
                asyncio.run(parse_sources_archive(staging, sources))
            inspect_raw_authority_frontier(
                Config(
                    archive_root=staging,
                    render_root=staging / "render",
                    sources=[],
                    db_path=staging / "index.db",
                )
            )
            facts = tuple(item.facts for written in written_batches for item in written.batch.artifacts)
            _sqlite_integrity(staging)
            _validate_facts(staging, facts)
            _validate_frontier_convergence(staging)
            archive_id = f"archive:seeded:{final_root.name}"
            profile_id = _profile_id(key)
            receipt = WorkloadReceipt.from_observations(
                spec=_archive_build_spec(key=key, archive_id=archive_id, profile_id=profile_id),
                status=WorkloadRunStatus.SUCCEEDED,
                build_id=key.build_id,
                runtime_id="synthetic-real-pipeline",
                archive_id=archive_id,
                generation_id=key.value,
                frame_id=None,
                phases=(
                    WorkloadPhaseObservation(name="generate"),
                    WorkloadPhaseObservation(name="acquire"),
                    WorkloadPhaseObservation(name="parse"),
                    WorkloadPhaseObservation(name="materialize"),
                    WorkloadPhaseObservation(name="index"),
                    WorkloadPhaseObservation(name="raw_authority_frontier"),
                    WorkloadPhaseObservation(name="validate"),
                    WorkloadPhaseObservation(name="publish", cleanup_complete=True, quiescent=True),
                ),
                cleanup_complete=True,
            )
            manifest = SeededArchiveManifest(
                protocol_version=_ARTIFACT_PROTOCOL_VERSION,
                key=key.value,
                archive_id=archive_id,
                profile_id=profile_id,
                build_id=key.build_id,
                recipe_id=key.recipe_id,
                facts=facts,
                files=_archive_files(staging),
                receipt=dict(receipt.to_payload()),
            )
            (staging / "manifest.json").write_text(
                json.dumps(manifest.to_payload(), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _make_read_only(staging)
            os.replace(staging, final_root)
        except Exception:
            _remove_tree(staging)
            raise
        artifact = _validate_artifact(final_root, key)
        if artifact is None:
            raise RuntimeError("published seeded archive failed its own validation")
        return artifact


def clone_seeded_archive(artifact: SeededArchiveArtifact, destination: Path) -> SeededArchiveClone:
    """Create a complete private writable archive clone, recording its method."""
    if destination.exists():
        _remove_tree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["cp", "-a", "--reflink=always", str(artifact.root), str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        method = "reflink"
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        _remove_tree(destination)
        shutil.copytree(artifact.root, destination)
        method = "copy"
    for path in destination.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    return SeededArchiveClone(
        root=destination,
        source_manifest_id=artifact.manifest.manifest_id,
        clone_method=method,
    )


__all__ = [
    "SeededArchiveArtifact",
    "SeededArchiveClone",
    "SeededArchiveKey",
    "SeededArchiveManifest",
    "build_seeded_archive",
    "c03_semantic_corpus_spec",
    "clone_seeded_archive",
    "named_corpus_specs",
    "schema_coverage_corpus_specs",
    "seeded_archive_key",
]
