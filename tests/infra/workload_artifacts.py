"""Receipt-backed, real-pipeline seeded archive artifacts for composition tests.

This is deliberately an adapter over production corpus generation, ingestion,
archive tiers, and workload receipts.  It owns no alternate generator, query
language, or semantic profile.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import subprocess
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from unittest.mock import patch

from polylogue.config import Config, Source
from polylogue.core.enums import Provider
from polylogue.core.sqlite_locking import is_transient_sqlite_lock
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.scenarios import CorpusProfile, CorpusSpec
from polylogue.scenarios.workload import (
    WorkloadEnvelopeSpec,
    WorkloadInputRef,
    WorkloadPhaseObservation,
    WorkloadReceipt,
    WorkloadRunStatus,
)
from polylogue.schemas.synthetic import SyntheticCorpus
from polylogue.schemas.synthetic.models import SyntheticArtifactFacts
from polylogue.sources.origin_specs import (
    lowering_fingerprint,
    materializer_fingerprint,
    origin_specs,
    replay_routing_fingerprint,
)
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.blob_gc import unlink_unreferenced_blob_hashes_under_exclusion
from polylogue.storage.blob_integrity import scan_blob_integrity
from polylogue.storage.blob_publication import abandon_blob_publication_receipts, inspect_blob_publication_receipts
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER, schema_identity
from tests.infra.source_builders import SyntheticAntigravityLanguageServerClient, provider_source_package

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

# v3 includes the explicit disposition of unreferenced publication bytes
# before a seeded archive is sealed.
_ARTIFACT_PROTOCOL_VERSION = 3
_SEEDED_KEY = re.compile(r"seeded-archive:sha256:([0-9a-f]{64})\Z")
#: Bounded rebuild attempts when a same-process SQLite lock (SQLITE_LOCKED,
#: not SQLITE_BUSY) aborts an artifact build. See the retry site below.
_BUILD_LOCK_ATTEMPTS = 3
_SCRATCH_CACHE_ROOT = Path("/realm/tmp/polylogue-seeded-artifacts")
_CLOUD_CACHE_ROOT = Path("/tmp/polylogue-seeded-artifacts")


def default_cache_root() -> Path:
    """Where published artifacts live when a caller names no cache root.

    Uses NVMe storage when ``/realm/tmp`` is mounted, and the ``/tmp`` fallback
    only when ``/realm`` is absent entirely (a genuine cloud sandbox). The
    previous hard-coded ``/realm/tmp`` path made every consumer
    of this module raise ``OSError`` on a host without ``/realm``, since the
    ``mkdir(parents=True)`` below cannot create a directory under a
    nonexistent mount point.
    """
    if _SCRATCH_CACHE_ROOT.parent.is_dir():
        return _SCRATCH_CACHE_ROOT
    return _CLOUD_CACHE_ROOT


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# These are the actual route roots, rather than the whole repository.  The
# synthetic generator and archive materializer are intentionally explicit so
# an unrelated UI/API edit does not evict every seeded artifact.  Parser
# fingerprints below cover provider parser modules; schema packages are added
# separately because their gzipped catalogs are runtime inputs, not Python.
_SOURCE_DEPENDENCY_ROOTS = (
    _REPOSITORY_ROOT / "polylogue" / "schemas" / "synthetic",
    _REPOSITORY_ROOT / "polylogue" / "pipeline" / "services" / "archive_ingest.py",
    _REPOSITORY_ROOT / "polylogue" / "pipeline" / "services" / "ingest_worker.py",
    _REPOSITORY_ROOT / "polylogue" / "sources" / "source_parsing.py",
    _REPOSITORY_ROOT / "polylogue" / "schemas" / "runtime_registry.py",
    _REPOSITORY_ROOT / "polylogue" / "schemas" / "operator",
    _REPOSITORY_ROOT / "polylogue" / "storage" / "sqlite" / "archive_tiers",
    _REPOSITORY_ROOT / "polylogue" / "storage" / "sqlite" / "archive_tiers" / "user_annotations.py",
    _REPOSITORY_ROOT / "polylogue" / "storage" / "archive_readiness.py",
    _REPOSITORY_ROOT / "polylogue" / "storage" / "raw_reconciler.py",
    _REPOSITORY_ROOT / "polylogue" / "scenarios" / "corpus.py",
    _REPOSITORY_ROOT / "tests" / "infra" / "source_builders.py",
)
_RECIPE_INPUT_ROOTS = (
    _REPOSITORY_ROOT / "pyproject.toml",
    _REPOSITORY_ROOT / "polylogue" / "__init__.py",
    _REPOSITORY_ROOT / "polylogue" / "schemas" / "runtime_registry.py",
    _REPOSITORY_ROOT / "polylogue" / "schemas" / "registry.py",
    _REPOSITORY_ROOT / "polylogue" / "insights" / "claude_workflow_materializer.py",
)
_RECIPE_PROVIDER_ROOT = _REPOSITORY_ROOT / "polylogue" / "schemas" / "providers"
_ARCHIVE_DB_NAMES = ("source.db", "index.db", "embeddings.db", "user.db", "audit.db", "ops.db")
_OBSOLETE_STAGING_SCAN_BUDGET = 32
_KNOWN_PROVIDERS = frozenset(SyntheticCorpus.available_providers())
_PROVIDER_COMPONENT = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_SEMANTIC_METADATA_PREFIXES = ("expected_", "oracle_", "pathology_", "case_")


def _reject_semantic_metadata(value: object, *, location: str) -> None:
    """Keep workload identity and publication records free of semantic oracles."""
    if isinstance(value, str) and value.startswith(_SEMANTIC_METADATA_PREFIXES):
        raise ValueError(f"{location} cannot carry semantic metadata: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and key.startswith(_SEMANTIC_METADATA_PREFIXES):
                raise ValueError(f"{location} cannot carry semantic metadata: {key}")
            _reject_semantic_metadata(child, location=location)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_semantic_metadata(child, location=location)


def _validate_provider(provider: object) -> str:
    if not isinstance(provider, str) or not _PROVIDER_COMPONENT.fullmatch(provider):
        raise ValueError("corpus provider must be one safe path component")
    if provider not in _KNOWN_PROVIDERS:
        raise ValueError(f"unknown corpus provider: {provider!r}")
    return provider


@dataclass(frozen=True)
class SeededArchiveKey:
    """Content identity of a published artifact: what would change its bytes.

    Deliberately does NOT carry the git commit. An artifact is a function of
    the corpus specification, the generation/ingest recipe, the parser
    semantics, and the archive DDL -- not of every unrelated commit that
    happens to be checked out. Keying on ``git rev-parse HEAD`` (as this did
    until polylogue-1xc.14.1) made the key change on every commit, so the
    cache degenerated into a per-commit rebuild that still paid the full
    publish cost while accumulating one immutable multi-MB artifact per
    commit per workload. The commit is kept on the manifest as provenance,
    where it answers "which checkout built this" without gating reuse.
    """

    spec_payload: dict[str, object]
    artifact_protocol_version: int
    recipe_id: str
    source_semantics_id: str
    archive_schema_id: str

    @property
    def value(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        return f"seeded-archive:sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class CorpusArtifactManifest:
    """Authenticated publication record for a deterministic corpus artifact.

    The manifest describes construction and storage identity only.  It does
    not carry an expected semantic result, incident classification, or case
    admission state.
    """

    protocol_version: int
    key: str
    archive_id: str
    profile_id: str
    build_id: str
    recipe_id: str
    source_semantics_id: str
    archive_schema_id: str
    facts: tuple[SyntheticArtifactFacts, ...]
    files: tuple[dict[str, object], ...]
    receipt: dict[str, object]

    def __post_init__(self) -> None:
        _reject_semantic_metadata(self.receipt, location="corpus artifact manifest receipt")
        _reject_semantic_metadata(self.files, location="corpus artifact manifest files")

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
    manifest: CorpusArtifactManifest

    @property
    def facts(self) -> tuple[SyntheticArtifactFacts, ...]:
        return self.manifest.facts


class ArtifactGcDisposition(str, Enum):
    """The fail-closed result for one published seeded-artifact directory."""

    REACHABLE = "reachable"
    GRACE = "grace"
    STALE = "stale"
    ACTIVE_LOCK = "active-lock"
    ACTIVE_LEASE = "active-lease"
    ACTIVE_WORKTREE = "active-worktree"
    CORRUPT = "corrupt"
    DELETED = "deleted"
    DELETION_FAILED = "deletion-failed"


@dataclass(frozen=True)
class ArtifactGcEntry:
    """Receipt row for a final artifact considered by cache GC."""

    name: str
    path: str
    key: str | None
    manifest_id: str | None
    size_bytes: int
    age_seconds: float | None
    disposition: ArtifactGcDisposition
    detail: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "key": self.key,
            "manifest_id": self.manifest_id,
            "size_bytes": self.size_bytes,
            "age_seconds": self.age_seconds,
            "disposition": self.disposition.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ArtifactGcReport:
    """Bounded, serializable preview/apply receipt for final-artifact GC."""

    cache_root: Path
    dry_run: bool
    grace_period_s: float
    reachable_keys: tuple[str, ...]
    entries: tuple[ArtifactGcEntry, ...]
    delete_corrupt: bool = False

    @property
    def deleted_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries if entry.disposition is ArtifactGcDisposition.DELETED)

    @property
    def reclaimable_bytes(self) -> int:
        return sum(
            entry.size_bytes
            for entry in self.entries
            if entry.disposition in {ArtifactGcDisposition.STALE, ArtifactGcDisposition.DELETED}
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "cache_root": str(self.cache_root),
            "dry_run": self.dry_run,
            "grace_period_s": self.grace_period_s,
            "reachable_keys": list(self.reachable_keys),
            "reclaimable_bytes": self.reclaimable_bytes,
            "deleted_bytes": self.deleted_bytes,
            "delete_corrupt": self.delete_corrupt,
            "entries": [entry.to_payload() for entry in self.entries],
        }


@dataclass
class SeededArchiveClone:
    root: Path
    source_manifest_id: str
    clone_method: str
    _integrity_fd: int = field(default=-1, repr=False, compare=False)

    def close(self) -> None:
        """Release this clone root's pinned integrity capability."""
        if self._integrity_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._integrity_fd)
            self._integrity_fd = -1

    def __del__(self) -> None:
        with contextlib.suppress(BaseException):
            self.close()

    def __enter__(self) -> SeededArchiveClone:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class SeededArchiveQueryLease:
    """Pinned capability that authenticates a shared artifact at consumer open."""

    artifact: SeededArchiveArtifact
    key: SeededArchiveKey
    _root_fd: int = field(default=-1, repr=False, compare=False)

    @property
    def root(self) -> Path:
        return self.artifact.root

    @property
    def path(self) -> Path:
        return self.root / "index.db"

    def _assert_current(self) -> None:
        if self._root_fd < 0:
            raise RuntimeError("query-only capability is finalized")
        if self.root.parent.name != "artifacts":
            raise RuntimeError("query-only capability source placement changed")
        try:
            placed = os.lstat(self.root)
            pinned = os.fstat(self._root_fd)
        except OSError as exc:
            raise RuntimeError("query-only capability source placement changed") from exc
        if (
            not stat.S_ISDIR(placed.st_mode)
            or stat.S_ISLNK(placed.st_mode)
            or (placed.st_dev, placed.st_ino) != (pinned.st_dev, pinned.st_ino)
        ):
            raise RuntimeError("query-only capability source placement changed")
        validated = _validate_artifact_with_retry(self.root, self.key)
        if validated is None or validated.manifest.manifest_id != self.artifact.manifest.manifest_id:
            raise RuntimeError("query-only capability source content changed")

    def open(self, *, read_only: bool = True) -> ArchiveStore:
        """Open only a revalidated, descriptor-bound read connection."""
        if not read_only:
            raise RuntimeError("query-only capability refuses write-capable connections")
        self._assert_current()
        index_fd = -1
        try:
            index_fd = _open_file_fd(self.path)
            self._assert_current()
            from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

            return ArchiveStore.open_existing(self.root, read_only=True, opened_main_fd=index_fd)
        finally:
            if index_fd >= 0:
                os.close(index_fd)

    def close(self) -> None:
        """Finalize the capability and refuse later opens."""
        if self._root_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._root_fd)
            self._root_fd = -1

    def __enter__(self) -> SeededArchiveQueryLease:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class ImmutableTreeArtifact:
    """An atomically published, read-only fixture tree.

    The caller supplies the stable key and a builder that writes one complete
    tree. This deliberately shares the cache/clone discipline with seeded
    archives without pretending every fixture is a ``CorpusSpec``.
    """

    root: Path
    key: str
    files: tuple[dict[str, object], ...]

    @property
    def manifest_id(self) -> str:
        payload = json.dumps({"key": self.key, "files": self.files}, sort_keys=True).encode()
        return f"immutable-tree:sha256:{hashlib.sha256(payload).hexdigest()}"


def build_immutable_tree(
    *,
    cache_root: Path | None,
    key: str,
    builder: Callable[[Path], object],
) -> ImmutableTreeArtifact:
    """Build or reuse one atomically published immutable fixture tree."""
    cache_root = (cache_root or default_cache_root()).expanduser()
    artifacts = cache_root / "artifacts"
    locks = cache_root / ".locks"
    staging_root = cache_root / ".staging"
    _mkdir_pinned(cache_root)
    _mkdir_pinned(artifacts)
    _mkdir_pinned(locks)
    _mkdir_pinned(staging_root)
    name = hashlib.sha256(key.encode()).hexdigest()
    final_root = artifacts / name
    lock_path = locks / f"{name}.lock"

    def load() -> ImmutableTreeArtifact | None:
        manifest_path = final_root / "manifest.json"
        if _is_symlink_node(final_root) or _is_symlink_node(manifest_path) or not _is_regular(manifest_path):
            return None
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("protocol_version") != _ARTIFACT_PROTOCOL_VERSION or payload.get("key") != key:
                return None
            files = _manifest_file_entries(tuple(payload["files"]))
            expected_paths = {path for path, _, _ in files}
            actual_paths = {
                str(path.relative_to(final_root))
                for path in _pinned_paths(final_root)
                if _is_regular(path) and not _is_reserved_root_file(path, final_root)
            }
            if actual_paths != expected_paths:
                return None
            write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
            for path in (final_root, *_pinned_paths(final_root)):
                if _is_symlink_node(path) or _safe_stat(path).st_mode & write_bits:
                    return None
            for relative, size, digest in files:
                path = final_root / relative
                if not _is_regular(path) or _safe_stat(path).st_size != size or _sha256(path) != digest:
                    return None
            return ImmutableTreeArtifact(
                root=final_root,
                key=key,
                files=tuple({"path": path, "size": size, "sha256": digest} for path, size, digest in files),
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    # Cache GC takes the same cache-root capability before it inspects or
    # removes a final artifact. Holding it across validation, construction,
    # and publication closes the gap between the per-key lock and the final
    # tree lock, including for generic fixture trees.
    domain = _open_lock_domain(cache_root)
    try:
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            cached = load()
            if cached is not None:
                return cached
            if final_root.exists():
                _remove_tree(final_root)
            staging = staging_root / f"{name}.{uuid.uuid4().hex}"
            staging.mkdir()
            try:
                builder(staging)
                files = _archive_files(staging)
                manifest = {
                    "protocol_version": _ARTIFACT_PROTOCOL_VERSION,
                    "key": key,
                    "files": files,
                }
                (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
                _publish_sealed_staging(staging, final_root)
            except Exception:
                _remove_tree(staging)
                raise
            artifact = load()
            if artifact is None:
                raise RuntimeError("published immutable tree failed validation")
            return artifact
    finally:
        _release_lock_domain(domain)


def _describe_file_set_mismatch(
    expected: dict[str, tuple[int, str]],
    actual: dict[str, tuple[int, str]],
) -> str:
    """Name the diverging paths so a clone failure is diagnosable from its message."""

    def summarize(label: str, paths: list[str]) -> str:
        head = paths[:8]
        suffix = f" +{len(paths) - len(head)} more" if len(paths) > len(head) else ""
        return f"{label}={head}{suffix}"

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])
    return " ".join((summarize("missing", missing), summarize("extra", extra), summarize("changed", changed)))


def clone_immutable_tree(artifact: ImmutableTreeArtifact, destination: Path) -> SeededArchiveClone:
    """Clone an immutable tree while pinning its publication capability."""
    with _shared_artifact_read_locks(artifact.root):
        return _clone_immutable_tree_unlocked(artifact, destination)


def _clone_immutable_tree_unlocked(artifact: ImmutableTreeArtifact, destination: Path) -> SeededArchiveClone:
    """Clone an immutable tree into a private writable root."""
    if _is_symlink_node(destination):
        raise ValueError(f"clone destination is a symlink: {destination}")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"clone destination is not a directory: {destination}")
        _remove_tree(destination)
    _assert_no_symlink_ancestors(destination.parent)
    _mkdir_pinned(destination.parent)
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
        _copy_tree(artifact.root, destination)
        method = "copy"
    _assert_no_symlinks(destination)
    source_files = _manifest_file_entries(artifact.files or _archive_files(artifact.root))
    expected = {path: (size, digest) for path, size, digest in source_files}
    actual = {
        str(path.relative_to(destination)): (_safe_stat(path).st_size, _sha256(path))
        for path in _pinned_paths(destination)
        if _is_regular(path) and not _is_reserved_root_file(path, destination)
    }
    if actual != expected:
        _remove_tree(destination)
        raise ValueError(
            "immutable fixture clone failed authenticated file-set validation "
            f"({_describe_file_set_mismatch(expected, actual)})"
        )
    for path in destination.rglob("*"):
        if not path.is_symlink():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    if _safe_exists(destination / "manifest.json"):
        _safe_unlink(destination / "manifest.json")
    return SeededArchiveClone(destination, artifact.manifest_id, method)


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


@dataclass(frozen=True)
class WorkloadSessionShape:
    """One provider-native population within a semantic workload."""

    provider: str
    count: int
    messages_min: int
    messages_max: int
    seed_offset: int = 0
    style: str = "tool-heavy"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("workload session shape requires a positive session count")
        if self.messages_min < 1 or self.messages_max < self.messages_min:
            raise ValueError("workload session shape has invalid message bounds")
        if self.seed_offset < 0:
            raise ValueError("workload session shape seed offset must be non-negative")


@dataclass(frozen=True)
class WorkloadProfile:
    """Shared semantic identity and provider-native spec constructor for workloads."""

    name: str
    purpose: str
    seed: int
    family_ids: tuple[str, ...]
    profile_tokens: tuple[str, ...]
    origin: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.purpose:
            raise ValueError("workload profile requires a name and purpose")
        if not self.family_ids or not self.profile_tokens:
            raise ValueError("workload profile requires semantic corpus identity")
        _reject_semantic_metadata(self.family_ids, location="workload profile")
        _reject_semantic_metadata(self.profile_tokens, location="workload profile")

    def corpus_specs(self, shapes: tuple[WorkloadSessionShape, ...]) -> tuple[CorpusSpec, ...]:
        if not shapes:
            raise ValueError("workload profile requires provider-native session shapes")
        corpus_profile = CorpusProfile(
            family_ids=self.family_ids,
            profile_tokens=self.profile_tokens,
            artifact_kind="archive",
        )
        return tuple(
            CorpusSpec.for_provider(
                shape.provider,
                count=shape.count,
                messages_min=shape.messages_min,
                messages_max=shape.messages_max,
                seed=self.seed + shape.seed_offset,
                style=shape.style,
                profile=corpus_profile,
                origin=self.origin,
                tags=self.tags,
            )
            for shape in shapes
        )


@dataclass(frozen=True)
class NamedWorkloadProfile:
    """A semantic, deterministic workload used by shared test fixtures."""

    workload: WorkloadProfile
    provider_session_counts: tuple[tuple[str, int], ...]
    messages_min: int = 4
    messages_max: int = 11

    @property
    def name(self) -> str:
        return self.workload.name

    @property
    def purpose(self) -> str:
        return self.workload.purpose

    @property
    def seed(self) -> int:
        return self.workload.seed

    def __post_init__(self) -> None:
        if not self.provider_session_counts or any(count < 1 for _provider, count in self.provider_session_counts):
            raise ValueError("named workload profile requires positive provider session counts")
        if self.messages_min < 1 or self.messages_max < self.messages_min:
            raise ValueError("named workload profile has invalid message bounds")

    def corpus_specs(self) -> tuple[CorpusSpec, ...]:
        return self.workload.corpus_specs(
            tuple(
                WorkloadSessionShape(provider, count, self.messages_min, self.messages_max)
                for provider, count in self.provider_session_counts
            )
        )


def _named_workload(name: str, purpose: str, *, seed: int = 42) -> WorkloadProfile:
    return WorkloadProfile(
        name=name,
        purpose=purpose,
        seed=seed,
        family_ids=("test-workload",),
        profile_tokens=(name, purpose, "provider-native"),
        origin=f"generated.test-workload-{name}",
        tags=("synthetic", "test", name, purpose),
    )


NAMED_WORKLOAD_PROFILES = (
    NamedWorkloadProfile(_named_workload("schema-small", "schema-scaling"), (("chatgpt", 10),)),
    NamedWorkloadProfile(_named_workload("schema-medium", "schema-scaling"), (("chatgpt", 50),)),
    NamedWorkloadProfile(_named_workload("cli-chatgpt", "cli-read"), (("chatgpt", 2),)),
    NamedWorkloadProfile(_named_workload("cli-mixed", "cli-read"), (("chatgpt", 2), ("claude-code", 2))),
    NamedWorkloadProfile(_named_workload("completion", "completion", seed=1271), (("chatgpt", 3), ("claude-ai", 3))),
)


def named_workload_profile(name: str) -> NamedWorkloadProfile:
    """Resolve one finite, semantically named test workload."""
    try:
        return next(profile for profile in NAMED_WORKLOAD_PROFILES if profile.name == name)
    except StopIteration as exc:
        raise ValueError(f"unknown named seeded archive workload {name!r}") from exc


def named_corpus_specs(name: str) -> tuple[CorpusSpec, ...]:
    """Resolve the finite shared workload catalog used by test consumers."""
    return named_workload_profile(name).corpus_specs()


class BenchmarkWorkloadTier(str, Enum):
    """Semantic benchmark projections backed by the shared archive artifact."""

    SMOKE = "smoke"
    REPRESENTATIVE = "representative"
    ARCHIVE_SCALE = "archive-scale"
    STRESS = "stress"


@dataclass(frozen=True)
class BenchmarkWorkloadProfile:
    """A deterministic mixed-origin benchmark projection.

    The target is expressed as messages because benchmark operations scale with
    indexed message and block populations. The tier name records why the
    projection exists, rather than treating an arbitrary row count as its
    identity.
    """

    tier: BenchmarkWorkloadTier
    workload: WorkloadProfile
    target_messages: int
    provider_session_counts: tuple[tuple[str, int], ...]
    messages_per_session: int = 10

    def __post_init__(self) -> None:
        if self.target_messages < 1 or self.messages_per_session < 1:
            raise ValueError("benchmark workload dimensions must be positive")
        if not self.provider_session_counts or any(count < 1 for _provider, count in self.provider_session_counts):
            raise ValueError("benchmark workload requires every configured provider to have sessions")
        if (
            sum(count for _provider, count in self.provider_session_counts) * self.messages_per_session
            != self.target_messages
        ):
            raise ValueError("benchmark workload session composition must exactly produce target_messages")

    @property
    def purpose(self) -> str:
        return self.workload.purpose


@dataclass(frozen=True)
class SeededArchiveReachabilityEntry:
    """One intentionally reusable seeded-archive recipe and its current key."""

    kind: str
    name: str
    key: SeededArchiveKey

    def to_payload(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name, "key": self.key.value}


@dataclass(frozen=True)
class SeededArchiveReachabilityInventory:
    """Generated reachability authority for the persistent seeded-artifact cache."""

    entries: tuple[SeededArchiveReachabilityEntry, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(entry.key.value for entry in self.entries)

    def to_payload(self) -> dict[str, object]:
        return {
            "entry_count": len(self.entries),
            "kinds": {
                kind: sum(entry.kind == kind for entry in self.entries) for kind in ("default", "named", "benchmark")
            },
            "entries": [entry.to_payload() for entry in self.entries],
        }


_BENCHMARK_PROVIDER_MIX = (
    ("claude-code", 80),
    ("codex", 15),
    ("chatgpt", 2),
    ("claude-ai", 1),
    ("gemini", 2),
)


def _benchmark_workload(tier: BenchmarkWorkloadTier, purpose: str) -> WorkloadProfile:
    return WorkloadProfile(
        name=tier.value,
        purpose=purpose,
        seed=42,
        family_ids=("benchmark-archive",),
        profile_tokens=(tier.value, "mixed-origin", "provider-native"),
        origin=f"generated.benchmark-{tier.value}",
        tags=("synthetic", "benchmark", tier.value),
    )


BENCHMARK_WORKLOAD_PROFILES = (
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.SMOKE,
        _benchmark_workload(BenchmarkWorkloadTier.SMOKE, "fast-benchmark"),
        1_000,
        _BENCHMARK_PROVIDER_MIX,
    ),
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.REPRESENTATIVE,
        _benchmark_workload(BenchmarkWorkloadTier.REPRESENTATIVE, "broad-benchmark"),
        5_000,
        tuple((provider, count * 5) for provider, count in _BENCHMARK_PROVIDER_MIX),
    ),
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.ARCHIVE_SCALE,
        _benchmark_workload(BenchmarkWorkloadTier.ARCHIVE_SCALE, "archive-scale-benchmark"),
        10_000,
        tuple((provider, count * 10) for provider, count in _BENCHMARK_PROVIDER_MIX),
    ),
    BenchmarkWorkloadProfile(
        BenchmarkWorkloadTier.STRESS,
        _benchmark_workload(BenchmarkWorkloadTier.STRESS, "stress-benchmark"),
        50_000,
        tuple((provider, count * 50) for provider, count in _BENCHMARK_PROVIDER_MIX),
    ),
)


def benchmark_workload_profile(tier: BenchmarkWorkloadTier | str) -> BenchmarkWorkloadProfile:
    """Resolve one named benchmark workload without exposing round-count labels."""
    resolved = BenchmarkWorkloadTier(tier)
    return next(profile for profile in BENCHMARK_WORKLOAD_PROFILES if profile.tier is resolved)


def benchmark_workload_tier(target_messages: int) -> BenchmarkWorkloadTier:
    """Map the former direct-seeder message targets to their semantic tiers."""
    for profile in BENCHMARK_WORKLOAD_PROFILES:
        if profile.target_messages == target_messages:
            return profile.tier
    supported = ", ".join(str(profile.target_messages) for profile in BENCHMARK_WORKLOAD_PROFILES)
    raise ValueError(f"no named benchmark workload for {target_messages} messages; supported targets: {supported}")


def benchmark_corpus_specs(
    tier: BenchmarkWorkloadTier | str,
    *,
    seed: int = 42,
) -> tuple[CorpusSpec, ...]:
    """Build provider-native corpus specs for a semantic benchmark tier."""
    profile = benchmark_workload_profile(tier)
    session_shapes: list[WorkloadSessionShape] = []
    for provider, count in profile.provider_session_counts:
        # Keep a bounded tail in every tier. The former direct generator sampled
        # a six-bucket session-depth distribution; these three deterministic
        # depths preserve the short, ordinary, and tail activation conditions
        # while retaining an exact message target for reproducible benchmarks.
        provider_shapes: tuple[tuple[int, int], ...]
        if provider == "claude-code":
            multiplier, remainder = divmod(count, 80)
            if remainder:
                raise ValueError("benchmark Claude Code composition must retain the 80-session provider mix")
            provider_shapes = ((50 * multiplier, 2), (25 * multiplier, 8), (5 * multiplier, 100))
        else:
            provider_shapes = ((count, profile.messages_per_session),)
        for shape_count, messages_per_session in provider_shapes:
            session_shapes.append(
                WorkloadSessionShape(
                    provider,
                    shape_count,
                    messages_per_session,
                    messages_per_session,
                    len(session_shapes),
                )
            )
    return replace(profile.workload, seed=seed).corpus_specs(tuple(session_shapes))


def current_seeded_archive_reachability() -> SeededArchiveReachabilityInventory:
    """Generate all current reusable seeded-artifact keys from fixture registries.

    The default and named fixture registries are the only production cache
    authority. Benchmark tiers are included at their named default seed. Test-
    local, property-generated, and caller-supplied ``CorpusSpec`` values are
    deliberately not reachable through this inventory and therefore remain
    eligible for the cache's age-gated GC once no capability protects them.
    """
    entries: list[SeededArchiveReachabilityEntry] = []

    def add(kind: str, name: str, specs: Iterable[CorpusSpec]) -> None:
        entries.append(SeededArchiveReachabilityEntry(kind, name, seeded_archive_key(tuple(specs))))

    add("default", "c03", (c03_semantic_corpus_spec(),))
    add("default", "schema-coverage", schema_coverage_corpus_specs())
    from tests.infra.integration_profile import default_integration_selection

    add("default", "integration", default_integration_selection().corpus_specs())
    for named_profile in NAMED_WORKLOAD_PROFILES:
        add("named", named_profile.name, named_profile.corpus_specs())
    for benchmark_profile in BENCHMARK_WORKLOAD_PROFILES:
        add("benchmark", benchmark_profile.tier.value, benchmark_corpus_specs(benchmark_profile.tier))

    inventory = SeededArchiveReachabilityInventory(tuple(entries))
    validate_seeded_archive_reachability(inventory)
    return inventory


def validate_seeded_archive_reachability(inventory: SeededArchiveReachabilityInventory) -> None:
    """Reject empty, duplicate, or partial inventories before cache GC runs."""
    if not inventory.entries:
        raise ValueError("seeded archive reachability inventory is empty")
    expected = {
        ("default", "c03"),
        ("default", "schema-coverage"),
        ("default", "integration"),
        *(("named", profile.name) for profile in NAMED_WORKLOAD_PROFILES),
        *(("benchmark", profile.tier.value) for profile in BENCHMARK_WORKLOAD_PROFILES),
    }
    actual = {(entry.kind, entry.name) for entry in inventory.entries}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"seeded archive reachability inventory is incomplete (missing={missing}, extra={extra})")
    if len(actual) != len(inventory.entries):
        raise ValueError("seeded archive reachability inventory has duplicate recipes")
    if len(set(inventory.keys)) != len(inventory.keys):
        raise ValueError("seeded archive reachability inventory has duplicate keys")
    if any(not _SEEDED_KEY.fullmatch(key) for key in inventory.keys):
        raise ValueError("seeded archive reachability inventory has malformed keys")


def build_benchmark_archive(
    tier: BenchmarkWorkloadTier | str,
    *,
    seed: int = 42,
    cache_root: Path | None = None,
) -> SeededArchiveArtifact:
    """Build or reuse a benchmark archive through the shared production route."""
    return build_seeded_archive(benchmark_corpus_specs(tier, seed=seed), cache_root=cache_root)


def _recipe_id(providers: Iterable[str] = ()) -> str:
    """Fingerprint the generation/materialization dependency and input closure."""
    digest = hashlib.sha256()
    files: set[Path] = set()
    safe_providers = tuple(_validate_provider(provider) for provider in providers)
    provider_roots = tuple(_RECIPE_PROVIDER_ROOT / provider for provider in sorted(set(safe_providers)))
    for root in (*_SOURCE_DEPENDENCY_ROOTS, *_RECIPE_INPUT_ROOTS, *provider_roots):
        if root.is_file():
            files.add(root)
            continue
        if not stat.S_ISDIR(_safe_stat(root).st_mode):
            continue
        for path in _pinned_paths(root):
            if _is_regular(path) and path.suffix.lower() in {
                ".py",
                ".json",
                ".gz",
                ".sql",
                ".toml",
                ".yaml",
                ".yml",
            }:
                files.add(path)
    for path in sorted(files):
        try:
            label = path.relative_to(_REPOSITORY_ROOT)
        except ValueError:
            label = path
        digest.update(str(label).encode())
        digest.update(b"\0")
        digest.update(_read_private_bytes(path))
        digest.update(b"\0")
    return f"recipe:sha256:{digest.hexdigest()}"


def _archive_schema_id() -> str:
    """Bind cached archives to the archive DDL that shaped them.

    The recipe fingerprint covers source semantics, while this component
    covers the rendered DDL that the tiers are actually created from. A schema
    change arriving through ``index.py``/``source.py``/``user.py`` (the normal
    route) therefore changes the key instead of leaving a stale-schema artifact
    reusable. Hash the rendered DDL
    the tiers are actually created from, plus each tier's declared version,
    rather than the Python module text: the DDL registry changes exactly when
    the created schema changes, and is immune to comment/docstring edits in
    the modules that build it.
    """
    digest = hashlib.sha256()
    for tier in sorted(ARCHIVE_DDL_BY_TIER, key=lambda item: item.value):
        digest.update(tier.value.encode())
        digest.update(b"\0")
        digest.update(str(ARCHIVE_VERSION_BY_TIER[tier]).encode())
        digest.update(b"\0")
        digest.update(ARCHIVE_DDL_BY_TIER[tier].encode())
        digest.update(b"\0")
    digest.update(schema_identity.DERIVED_SCHEMA_META_DDL.encode())
    digest.update(b"\0")
    for derived_tier in (schema_identity.DerivedTier.INDEX, schema_identity.DerivedTier.OPS):
        digest.update(derived_tier.value.encode())
        digest.update(b"\0")
        digest.update(schema_identity.derived_schema_identity(derived_tier).encode())
        digest.update(b"\0")
    return f"archive-schema:sha256:{digest.hexdigest()}"


def _build_id() -> str:
    """Provenance only: which checkout published an artifact, never part of the key."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "git:unavailable"
    commit = result.stdout.strip()
    return f"git:{commit}" if re.fullmatch(r"[0-9a-f]{40}", commit) else "git:unavailable"


def _valid_build_id(value: object) -> bool:
    return value == "git:unavailable" or (
        isinstance(value, str) and re.fullmatch(r"git:[0-9a-f]{40}", value) is not None
    )


_PROVIDER_ORIGIN_WIRES = {
    "chatgpt": Provider.CHATGPT,
    "claude-ai": Provider.CLAUDE_AI,
    "claude-code": Provider.CLAUDE_CODE,
    "codex": Provider.CODEX,
    "gemini": Provider.GEMINI,
}


def _source_semantics_id(providers: Iterable[str] = ()) -> str:
    """Bind identity to selected provider parser/materializer semantics only."""

    selected_wires = {_PROVIDER_ORIGIN_WIRES[name] for name in providers if name in _PROVIDER_ORIGIN_WIRES}
    selected_specs = tuple(
        spec for spec in origin_specs() if not selected_wires or selected_wires.intersection(spec.provider_wires)
    )
    payload = {
        "lowering": lowering_fingerprint(),
        "replay_routing": replay_routing_fingerprint(),
        "materializer": materializer_fingerprint(),
        "parsers": {
            spec.origin.value: spec.parser_fingerprint()
            for spec in selected_specs
            if spec.parser_paths or spec.stream_parser_path or spec.assembly_paths or spec.assembly_spec_path
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"source-semantics:sha256:{hashlib.sha256(encoded).hexdigest()}"


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


def _canonical_receipt(
    *,
    key: SeededArchiveKey,
    archive_id: str,
    profile_id: str,
    build_id: str,
) -> WorkloadReceipt:
    """Construct the sole accepted receipt shape from the cache recipe."""
    return WorkloadReceipt.from_observations(
        spec=_archive_build_spec(key=key, archive_id=archive_id, profile_id=profile_id),
        status=WorkloadRunStatus.SUCCEEDED,
        build_id=build_id,
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


def _profile_id(key: SeededArchiveKey) -> str:
    payload = json.dumps(key.spec_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return f"workload-profile:sha256:{hashlib.sha256(payload).hexdigest()}"


def seeded_archive_key(specs: Iterable[CorpusSpec]) -> SeededArchiveKey:
    selected_specs = tuple(specs)
    providers = tuple(_validate_provider(spec.provider) for spec in selected_specs)
    return SeededArchiveKey(
        spec_payload={"corpus_specs": [spec.to_payload() for spec in selected_specs]},
        artifact_protocol_version=_ARTIFACT_PROTOCOL_VERSION,
        recipe_id=_recipe_id(providers),
        source_semantics_id=_source_semantics_id(providers),
        archive_schema_id=_archive_schema_id(),
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


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _open_file_fd(path: Path) -> int:
    return _open_no_follow(path, os.O_RDONLY | os.O_NONBLOCK)


def _sha256(path: Path) -> str:
    fd = _open_file_fd(path)
    try:
        return _sha256_fd(fd)
    finally:
        os.close(fd)


def _is_reserved_root_file(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return len(relative.parts) == 1 and relative.name in {"manifest.json", ".build.lock"}


def _is_symlink_node(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _components(path: Path) -> tuple[bool, tuple[str, ...]]:
    raw = os.fspath(path)
    absolute = os.path.isabs(raw)
    parts = tuple(part for part in raw.split(os.sep) if part not in ("", "."))
    if any(part == ".." for part in parts):
        raise ValueError(f"parent traversal is not allowed: {path}")
    return absolute, parts


def _open_pinned_dir(path: Path, *, allow_missing: bool = False) -> int:
    """Open a directory and every ancestor with openat/O_NOFOLLOW.

    Once returned, callers use this descriptor as their capability.  No later
    pathname walk can be redirected by replacing an ancestor or inserting a
    symlink; each component is opened relative to the already pinned parent.
    """
    absolute, parts = _components(path)
    fd = os.open(os.sep if absolute else ".", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    try:
        for part in parts:
            try:
                next_fd = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if allow_missing:
                    return fd
                raise
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _assert_no_symlink_ancestors(path: Path, *, allow_missing_leaf: bool = True) -> None:
    fd = _open_pinned_dir(path, allow_missing=allow_missing_leaf)
    os.close(fd)


def _open_pinned_parent(path: Path, *, create: bool = False) -> tuple[int, str]:
    absolute, parts = _components(path)
    if not parts:
        raise ValueError(f"path has no leaf: {path}")
    parent_parts, leaf = parts[:-1], parts[-1]
    parent = Path(os.sep if absolute else ".", *parent_parts)
    if create:
        _mkdir_pinned(parent)
    return _open_pinned_dir(parent), leaf


def _mkdir_pinned(path: Path, mode: int = 0o700) -> None:
    absolute, parts = _components(path)
    fd = os.open(os.sep if absolute else ".", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    try:
        for part in parts:
            try:
                os.mkdir(part, mode, dir_fd=fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _open_no_follow(path: Path, flags: int, mode: int = 0o600, *, create_parent: bool = False) -> int:
    parent, leaf = _open_pinned_parent(path, create=create_parent)
    try:
        return os.open(leaf, flags | _O_NOFOLLOW, mode, dir_fd=parent)
    finally:
        os.close(parent)


def _safe_stat(path: Path) -> os.stat_result:
    parent, leaf = _open_pinned_parent(path)
    try:
        return os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        raise
    finally:
        os.close(parent)


def _chmod_at(directory_fd: int, leaf: str, mode: int) -> None:
    """Change mode through an O_NOFOLLOW descriptor, never a symlink target."""
    fd = os.open(leaf, os.O_RDONLY | os.O_NONBLOCK | _O_NOFOLLOW, dir_fd=directory_fd)
    try:
        os.fchmod(fd, mode)
    finally:
        os.close(fd)


def _safe_chmod(path: Path, mode: int) -> None:
    parent, leaf = _open_pinned_parent(path)
    try:
        _chmod_at(parent, leaf, mode)
    finally:
        os.close(parent)


def _is_regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(_safe_stat(path).st_mode)
    except FileNotFoundError:
        return False


def _safe_exists(path: Path) -> bool:
    try:
        _safe_stat(path)
    except FileNotFoundError:
        return False
    return True


def _safe_unlink(path: Path) -> None:
    parent, leaf = _open_pinned_parent(path)
    try:
        os.unlink(leaf, dir_fd=parent)
    finally:
        os.close(parent)


def _safe_replace(source: Path, destination: Path) -> None:
    src_parent, src_leaf = _open_pinned_parent(source)
    dst_parent, dst_leaf = _open_pinned_parent(destination)
    try:
        os.replace(src_leaf, dst_leaf, src_dir_fd=src_parent, dst_dir_fd=dst_parent)
    finally:
        os.close(src_parent)
        os.close(dst_parent)


def _assert_lock_identity(fd: int, path: Path) -> None:
    parent, leaf = _open_pinned_parent(path)
    try:
        named = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
    finally:
        os.close(parent)
    opened = os.fstat(fd)
    if (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise OSError("lock pathname was replaced while lock was held")


def _open_authenticated_lock(path: Path, *, nonblocking: bool = False, shared: bool = False) -> int:
    """Open and flock a lock inode, rejecting pathname replacement.

    A flock authenticates an inode, not a pathname.  Comparing the opened
    descriptor's identity with the directory entry after flock prevents a
    regular-file replacement from turning the caller into the owner of a new,
    unrelated lock inode.
    """
    fd = _open_no_follow(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        operation = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(fd, operation)
        try:
            _assert_lock_identity(fd, path)
        except OSError:
            fcntl.flock(fd, fcntl.LOCK_UN)
            raise
        return fd
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


class _DirectoryEntryIterator(Protocol):
    def __iter__(self) -> Iterator[os.DirEntry[str]]: ...

    def __next__(self) -> os.DirEntry[str]: ...

    def close(self) -> None: ...


def _pinned_paths(root: Path, *, budget: int = 100_000) -> Iterator[Path]:
    """Stream a tree from pinned descriptors with depth and node bounds."""
    if budget <= 0:
        raise ValueError("cache enumeration budget must be positive")
    root_fd = _open_pinned_dir(root)
    owned_fds: set[int] = {root_fd}
    stack: list[tuple[int, Path, int, _DirectoryEntryIterator]] = []
    try:
        stack.append((root_fd, root, 0, os.scandir(root_fd)))
        seen = 0
        while stack:
            fd, prefix, depth, entries = stack[-1]
            try:
                entry = next(entries)
            except StopIteration:
                entries.close()
                stack.pop()
                if fd != root_fd:
                    os.close(fd)
                    owned_fds.discard(fd)
                continue
            seen += 1
            if seen > budget:
                raise RuntimeError("cache enumeration exceeded bounded node budget")
            path = prefix / entry.name
            info = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"symlink node is not allowed: {path}")
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError(f"unsupported cache node is not allowed: {path}")
            yield path
            if stat.S_ISDIR(info.st_mode):
                if depth >= 256:
                    raise RuntimeError("cache enumeration exceeded maximum tree depth")
                child = os.open(entry.name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=fd)
                owned_fds.add(child)
                try:
                    stack.append((child, path, depth + 1, os.scandir(child)))
                except BaseException:
                    os.close(child)
                    owned_fds.discard(child)
                    raise
    finally:
        for _, _, _, entries in stack:
            with contextlib.suppress(OSError):
                entries.close()
        for fd in owned_fds:
            with contextlib.suppress(OSError):
                os.close(fd)


def _archive_files(root: Path) -> tuple[dict[str, object], ...]:
    if _is_symlink_node(root):
        raise ValueError(f"cannot archive symlink root: {root}")
    entries = []
    for path in sorted(_pinned_paths(root)):
        if _is_symlink_node(path):
            raise ValueError(f"cannot archive symlink node: {path}")
        if not _is_regular(path):
            continue
        if _is_reserved_root_file(path, root):
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size": _safe_stat(path).st_size,
                "sha256": _sha256(path),
            }
        )
    return tuple(entries)


def _manifest_file_entries(files: tuple[dict[str, object], ...]) -> tuple[tuple[str, int, str], ...]:
    """Validate manifest file records before any keyed access or filesystem use."""
    entries: list[tuple[str, int, str]] = []
    for item in files:
        _reject_semantic_metadata(item, location="seeded archive manifest file")
        path_value = item.get("path")
        size_value = item.get("size")
        hash_value = item.get("sha256")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("seeded archive manifest file path is malformed")
        relative = Path(path_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("seeded archive manifest file path escapes root")
        if not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
            raise ValueError("seeded archive manifest file size is malformed")
        if not isinstance(hash_value, str) or len(hash_value) != 64:
            raise ValueError("seeded archive manifest file hash is malformed")
        try:
            int(hash_value, 16)
        except ValueError as exc:
            raise ValueError("seeded archive manifest file hash is malformed") from exc
        entries.append((path_value, size_value, hash_value))
    if len({path for path, _, _ in entries}) != len(entries):
        raise ValueError("seeded archive manifest contains duplicate files")
    return tuple(entries)


def _journal_mode_delete_with_retry(conn: sqlite3.Connection, *, name: str) -> None:
    """Flip WAL to a DELETE-journal snapshot, tolerant of a same-process zombie closer.

    CPython's ``sqlite3`` module closes connections via ``sqlite3_close_v2``
    (not the older non-``_v2`` API): a connection whose last statement/cursor
    hasn't been finalized yet becomes a "zombie" that keeps SQLite's
    per-process shared pager-cache entry for that file alive until the
    lingering ``Cursor``/``Connection`` object is actually garbage-collected.
    While a zombie is pending, ``PRAGMA journal_mode=DELETE`` on a *different*
    (fully legitimate, single) connection to the same file raises
    ``sqlite3.OperationalError: database is locked`` -- SQLite reports this
    specific condition as ``SQLITE_LOCKED`` (a same-process/shared-cache
    conflict), which, unlike plain ``SQLITE_BUSY``, is **not** retried by
    ``sqlite3``'s own busy-timeout/busy-handler mechanism, so a plain
    ``timeout=`` connect argument cannot absorb it (polylogue-lbgc).

    Confirmed empirically: two genuinely separate OS processes building the
    same corpus key concurrently (an isolated ``cache_root``, no pytest)
    serialize cleanly through the ``build_seeded_archive`` file lock with no
    error. The failure reproduces only under real system load (pytest-xdist
    workers plus a busy machine), which is exactly when CPython's cyclic GC
    is more likely to have deferred collecting a zombie connection/cursor
    from earlier in this same worker process's own write pipeline. Forcing a
    ``gc.collect()`` finalizes any such zombie so its shared-cache slot is
    actually released, then a short bounded retry absorbs the remaining
    scheduling jitter.
    """
    deadline = time.monotonic() + 5.0
    attempt = 0
    while True:
        attempt += 1
        try:
            mode = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        except sqlite3.OperationalError as exc:
            if not is_transient_sqlite_lock(exc) or time.monotonic() >= deadline:
                raise
            gc.collect()
            time.sleep(min(0.05 * attempt, 0.5))
            continue
        if mode != ("delete",):
            raise RuntimeError(f"could not close seeded archive tier {name} into a snapshot")
        return


def _sqlite_integrity(root: Path) -> None:
    """Validate + collapse each tier's journal to a snapshot, one connection at a time.

    Uses ``contextlib.closing`` (the idiom already used for this same
    connection-lifecycle footgun in ``polylogue/storage/raw_reconciler.py``)
    so each connection is actually closed, not merely committed/rolled back
    the way a bare ``with sqlite3.connect(...) as conn:`` would leave it.
    """
    checked: list[str] = []
    read_only = not (_safe_stat(root).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    for name in _ARCHIVE_DB_NAMES:
        path = root / name
        try:
            db_fd = _open_no_follow(path, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            connection = sqlite3.connect(f"file:/proc/self/fd/{db_fd}?mode={'ro' if read_only else 'rw'}", uri=True)
            with contextlib.closing(connection) as conn, conn:
                quick = conn.execute("PRAGMA quick_check").fetchone()
                foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
                if quick != ("ok",) or foreign:
                    raise RuntimeError(f"invalid seeded archive tier {name}")
                if not read_only:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    _journal_mode_delete_with_retry(conn, name=name)
        finally:
            os.close(db_fd)
        checked.append(name)
    for name in checked:
        for suffix in ("-wal", "-shm"):
            sidecar = root / f"{name}{suffix}"
            try:
                _safe_unlink(sidecar)
            except FileNotFoundError:
                pass


_MAX_DELETE_NODES = 10_000


def _remove_tree(path: Path, *, budget: int = _MAX_DELETE_NODES) -> None:
    """Delete a locally-owned tree with iterative, bounded descriptor walks."""
    if budget <= 0:
        raise ValueError("cache deletion budget must be positive")
    parent, leaf = _open_pinned_parent(path)
    original_parent_mode: int | None = None
    owned_fds: set[int] = set()
    iterators: set[_DirectoryEntryIterator] = set()
    try:
        original_parent_mode = os.fstat(parent).st_mode
        os.fchmod(parent, original_parent_mode | stat.S_IWUSR)
        stack: list[tuple[str, int, str, int, int | None, _DirectoryEntryIterator | None]] = [
            ("entry", parent, leaf, 0, None, None)
        ]
        inspected = 0
        while stack:
            action, directory_fd, name, depth, child_fd, entries = stack.pop()
            if action == "scan":
                assert child_fd is not None and entries is not None
                try:
                    entry = next(entries)
                except StopIteration:
                    entries.close()
                    iterators.discard(entries)
                    continue
                stack.append(("scan", directory_fd, name, depth, child_fd, entries))
                stack.append(("entry", child_fd, entry.name, depth + 1, None, None))
                continue
            if action == "rmdir":
                assert child_fd is not None
                os.close(child_fd)
                owned_fds.discard(child_fd)
                _chmod_at(directory_fd, name, os.lstat(name, dir_fd=directory_fd).st_mode | stat.S_IWUSR)
                os.rmdir(name, dir_fd=directory_fd)
                continue
            inspected += 1
            if inspected > budget:
                raise RuntimeError("cache deletion exceeded bounded node budget")
            try:
                info = os.lstat(name, dir_fd=directory_fd)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                if depth >= 256:
                    raise RuntimeError("cache deletion exceeded maximum tree depth")
                child = os.open(name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=directory_fd)
                owned_fds.add(child)
                try:
                    os.fchmod(child, os.fstat(child).st_mode | stat.S_IWUSR)
                    child_entries = os.scandir(child)
                    iterators.add(child_entries)
                except BaseException:
                    os.close(child)
                    owned_fds.discard(child)
                    raise
                stack.append(("rmdir", directory_fd, name, depth, child, None))
                stack.append(("scan", child, name, depth, child, child_entries))
            elif not stat.S_ISREG(info.st_mode):
                os.unlink(name, dir_fd=directory_fd)
            else:
                # Unlinking needs a writable parent, not a writable file.
                # Do not chmod regular files here: a rejected reflink/copy
                # may contain hardlinks to the immutable source tree, and a
                # chmod would mutate the source inode before removing the
                # destination name.
                os.unlink(name, dir_fd=directory_fd)
    finally:
        for entries in iterators:
            with contextlib.suppress(OSError):
                entries.close()
        for owned_fd in owned_fds:
            with contextlib.suppress(OSError):
                os.close(owned_fd)
        if original_parent_mode is not None:
            with contextlib.suppress(OSError):
                os.fchmod(parent, original_parent_mode)
        os.close(parent)


def _recover_stale_staging(*, staging_root: Path, artifact_name: str) -> tuple[str, ...]:
    """Remove crash-left staging trees while never racing an active builder."""
    removed: list[str] = []
    for candidate in _bounded_cache_candidates(
        staging_root, cursor="", budget=_MAX_DELETE_NODES, prefix=f"{artifact_name}."
    ):
        if _is_symlink_node(candidate):
            _remove_tree(candidate)
            removed.append(candidate.name)
            continue
        try:
            if not stat.S_ISDIR(_safe_stat(candidate).st_mode):
                continue
        except (FileNotFoundError, NotADirectoryError, OSError, ValueError):
            continue
        try:
            _remove_tree(candidate)
        except (FileNotFoundError, NotADirectoryError, OSError, ValueError):
            continue
        removed.append(candidate.name)
    return tuple(removed)


def _read_private_bytes(path: Path) -> bytes:
    fd = _open_no_follow(path, os.O_RDONLY)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            return handle.read()
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _read_private_text(path: Path) -> str:
    fd = _open_no_follow(path, os.O_RDONLY)
    try:
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as handle:
            return handle.read()
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _write_private_text(path: Path, text: str) -> None:
    fd = _open_no_follow(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _bounded_cache_candidates(
    directory: Path,
    *,
    cursor: str,
    budget: int,
    suffix: str | None = None,
    prefix: str | None = None,
) -> list[Path]:
    """Enumerate at most ``budget`` candidates from one pinned directory."""
    if budget <= 0:
        return []
    directory_fd = _open_pinned_dir(directory)
    try:

        def scan(*, after_cursor: bool) -> list[Path]:
            selected: list[Path] = []
            inspected = 0
            with os.scandir(directory_fd) as entries:
                preview: list[os.DirEntry[str]] = []
                for _ in range(min(2, budget)):
                    try:
                        preview.append(next(entries))
                    except StopIteration:
                        break
                descending = (len(preview) == 2 and preview[1].name < preview[0].name) or (
                    len(preview) == 1 and bool(cursor) and preview[0].name < cursor
                )
                ordered_entries = chain(preview, entries)
                for candidate in ordered_entries:
                    if (
                        after_cursor
                        and cursor
                        and ((not descending and candidate.name <= cursor) or (descending and candidate.name >= cursor))
                    ):
                        # The persisted cursor has already accounted for this
                        # prefix. It must not consume the new scan budget.
                        continue
                    inspected += 1
                    if inspected > budget:
                        break
                    if len(selected) >= budget:
                        break
                    if prefix is not None and not candidate.name.startswith(prefix):
                        continue
                    if suffix is not None and not candidate.name.endswith(suffix):
                        continue
                    info = candidate.stat(follow_symlinks=False)
                    if suffix is None and "." not in candidate.name:
                        continue
                    if suffix is None and not (stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                        continue
                    selected.append(directory / candidate.name)
            return selected

        selected = scan(after_cursor=True)
        return selected
    finally:
        os.close(directory_fd)


def _bounded_scan_last_name(directory: Path, *, cursor: str, budget: int) -> str:
    """Return the last inspected entry even when no candidate matched."""
    if budget <= 0:
        return cursor
    directory_fd = _open_pinned_dir(directory)
    last = cursor
    try:
        inspected = 0
        with os.scandir(directory_fd) as entries:
            preview: list[os.DirEntry[str]] = []
            for _ in range(min(2, budget)):
                try:
                    preview.append(next(entries))
                except StopIteration:
                    break
            descending = (len(preview) == 2 and preview[1].name < preview[0].name) or (
                len(preview) == 1 and bool(cursor) and preview[0].name < cursor
            )
            for entry in chain(preview, entries):
                if cursor and ((not descending and entry.name <= cursor) or (descending and entry.name >= cursor)):
                    continue
                inspected += 1
                if inspected > budget:
                    break
                last = entry.name
        return last
    finally:
        os.close(directory_fd)


_MAX_CLEANUP_SEEN_BYTES = 1 << 20


def _cleanup_identity(name: str, info: os.stat_result) -> str:
    return f"{name}\0{info.st_dev}:{info.st_ino}:{info.st_ctime_ns}:{info.st_size}"


def _seen_cleanup_identity(path: Path, identity: str, *, budget: int) -> tuple[bool, int]:
    if budget <= 0:
        return False, 0
    try:
        fd = _open_no_follow(path, os.O_RDONLY)
    except FileNotFoundError:
        return False, 0
    scanned = 0
    try:
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as handle:
            for line in handle:
                scanned += 1
                if line.rstrip("\n") == identity:
                    return True, scanned
                if scanned >= budget:
                    break
        return False, scanned
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _mark_cleanup_identity(path: Path, identity: str) -> None:
    fd = _open_no_follow(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.fstat(fd).st_size >= _MAX_CLEANUP_SEEN_BYTES:
            os.ftruncate(fd, 0)
        with os.fdopen(fd, "a", encoding="utf-8", closefd=True) as handle:
            handle.write(identity + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise


def _recover_obsolete_staging(
    *,
    cache_root: Path,
    staging_root: Path,
    budget: int = _OBSOLETE_STAGING_SCAN_BUDGET,
) -> tuple[str, ...]:
    """Boundedly sweep abandoned keys with a persistent continuation cursor.

    The cache-wide cleanup lock serializes cursor updates.  Each candidate is
    still authenticated by a non-blocking per-key flock before removal, so an
    active builder is never touched.  A bounded batch plus cursor prevents a
    large obsolete cache from turning one build into an unbounded cleanup pass;
    subsequent builders continue from the last inspected name. Lock files are
    retained because unlinking one can create two independent locks.
    """
    if budget <= 0:
        return ()
    locks_root = cache_root / ".locks"
    cleanup_lock = cache_root / ".cleanup.lock"
    cursor_path = cache_root / ".cleanup.cursor"
    # A replaced lock is ambiguous: unlinking it can strand the owner holding
    # the old inode and let a second cleaner enter.  Refuse the cleanup pass;
    # never repair suspicious lock/cursor paths by deletion.
    if _is_symlink_node(cleanup_lock) or _is_symlink_node(cursor_path):
        return ()
    removed: list[str] = []
    try:
        lock_fd = _open_authenticated_lock(cleanup_lock)
    except OSError:
        return ()
    with os.fdopen(lock_fd, "a+", encoding="utf-8") as cleanup_handle:
        fcntl.flock(cleanup_handle.fileno(), fcntl.LOCK_EX)
        _assert_lock_identity(cleanup_handle.fileno(), cleanup_lock)
        cursor = _read_private_text(cursor_path).strip() if _safe_exists(cursor_path) else ""
        seen_path = cache_root / ".cleanup.seen"
        if _is_symlink_node(seen_path):
            return ()
        staging_fd = _open_pinned_dir(staging_root)
        inspected = 0
        last_seen = ""
        try:
            with os.scandir(staging_fd) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    identity = _cleanup_identity(entry.name, info)
                    # Journal lookup is independently bounded; node
                    # selection retains its own bounded batch.
                    seen, _journal_work = _seen_cleanup_identity(seen_path, identity, budget=max(1, budget - inspected))
                    if seen:
                        continue
                    inspected += 1
                    last_seen = entry.name
                    candidate = staging_root / entry.name
                    if "." not in entry.name or not (stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
                        _mark_cleanup_identity(seen_path, identity)
                        continue
                    artifact_name = entry.name.split(".", 1)[0]
                    lock_path = locks_root / f"{artifact_name}.lock"
                    if _is_symlink_node(lock_path):
                        break
                    try:
                        lock_fd = _open_authenticated_lock(lock_path, nonblocking=True)
                        with os.fdopen(lock_fd, "a+") as handle:
                            try:
                                _assert_lock_identity(handle.fileno(), lock_path)
                                _remove_tree(candidate)
                                removed.append(entry.name)
                                _mark_cleanup_identity(seen_path, identity)
                            finally:
                                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        break
        finally:
            os.close(staging_fd)
        if last_seen and last_seen != cursor:
            _write_private_text(cursor_path, last_seen + "\n")
        fcntl.flock(cleanup_handle.fileno(), fcntl.LOCK_UN)

    return tuple(removed)


def _recover_stale_handoffs(
    *,
    cache_root: Path,
    artifacts_root: Path,
    budget: int = _OBSOLETE_STAGING_SCAN_BUDGET,
) -> tuple[str, ...]:
    """Boundedly reclaim sealed handoffs left by a killed publication."""
    if budget <= 0:
        return ()
    locks_root = cache_root / ".locks"
    cleanup_lock = cache_root / ".cleanup.lock"
    cursor_path = cache_root / ".handoff.cursor"
    if _is_symlink_node(cleanup_lock) or _is_symlink_node(cursor_path):
        return ()
    removed: list[str] = []
    try:
        lock_fd = _open_authenticated_lock(cleanup_lock)
    except OSError:
        return ()
    with os.fdopen(lock_fd, "a+", encoding="utf-8") as cleanup_handle:
        fcntl.flock(cleanup_handle.fileno(), fcntl.LOCK_EX)
        _assert_lock_identity(cleanup_handle.fileno(), cleanup_lock)
        cursor = _read_private_text(cursor_path).strip() if _safe_exists(cursor_path) else ""
        seen_path = cache_root / ".handoff.seen"
        if _is_symlink_node(seen_path):
            return ()
        artifacts_fd = _open_pinned_dir(artifacts_root)
        inspected = 0
        last_seen = ""
        try:
            with os.scandir(artifacts_fd) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    identity = _cleanup_identity(entry.name, info)
                    # Journal lookup is independently bounded; node
                    # selection retains its own bounded batch.
                    seen, _journal_work = _seen_cleanup_identity(seen_path, identity, budget=max(1, budget - inspected))
                    if seen:
                        continue
                    inspected += 1
                    last_seen = entry.name
                    candidate = artifacts_root / entry.name
                    if stat.S_ISLNK(info.st_mode):
                        _remove_tree(candidate)
                        removed.append(entry.name)
                        _mark_cleanup_identity(seen_path, identity)
                        continue
                    if not stat.S_ISDIR(info.st_mode):
                        _mark_cleanup_identity(seen_path, identity)
                        continue
                    parts = entry.name.removeprefix(".").split(".", 1)
                    if len(parts) != 2:
                        _mark_cleanup_identity(seen_path, identity)
                        continue
                    lock_path = locks_root / f"{parts[0]}.lock"
                    if _is_symlink_node(lock_path):
                        break
                    try:
                        lock_fd = _open_authenticated_lock(lock_path, nonblocking=True)
                        with os.fdopen(lock_fd, "a+") as handle:
                            try:
                                _assert_lock_identity(handle.fileno(), lock_path)
                                _remove_tree(candidate)
                                removed.append(entry.name)
                                _mark_cleanup_identity(seen_path, identity)
                            finally:
                                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        break
        finally:
            os.close(artifacts_fd)
        if last_seen and last_seen != cursor:
            _write_private_text(cursor_path, last_seen + "\n")
        fcntl.flock(cleanup_handle.fileno(), fcntl.LOCK_UN)

    return tuple(removed)


def _validate_facts(root: Path, facts: tuple[SyntheticArtifactFacts, ...]) -> None:
    """Authenticate the receipt's planted facts against the materialized index."""
    if not facts:
        raise RuntimeError("seeded archive manifest has no planted facts")
    with contextlib.closing(sqlite3.connect(root / "index.db")) as conn:
        sessions = {
            str(row[0]): int(row[1])
            for row in conn.execute("SELECT session_id, COUNT(*) FROM messages GROUP BY session_id")
        }
        tool_ids = {
            str(row[0]) for row in conn.execute("SELECT DISTINCT tool_id FROM blocks WHERE tool_id IS NOT NULL")
        }
    for fact in facts:
        if fact.expected_session_id is not None:
            actual_messages = sessions.get(fact.expected_session_id)
            if actual_messages is None:
                raise RuntimeError(f"missing planted session {fact.expected_session_id}")
            if actual_messages != fact.message_count:
                raise RuntimeError(f"planted message count mismatch for {fact.expected_session_id}")
        if not set(fact.tool_use_ids) <= tool_ids:
            raise RuntimeError(f"missing planted tool action for {fact.expected_session_id}")
        if not set(fact.tool_result_ids) <= tool_ids:
            raise RuntimeError(f"missing planted tool result for {fact.expected_session_id}")


def _canonical_facts(key: SeededArchiveKey) -> tuple[SyntheticArtifactFacts, ...]:
    """Recompute generator facts independently of the disk manifest."""
    raw_specs = key.spec_payload.get("corpus_specs")
    if not isinstance(raw_specs, list) or not all(isinstance(item, dict) for item in raw_specs):
        raise ValueError("seeded archive key has malformed corpus specifications")
    specs = tuple(CorpusSpec.from_payload(item) for item in raw_specs)
    return tuple(
        artifact.facts for spec in specs for artifact in SyntheticCorpus.generate_batch_for_spec(spec).artifacts
    )


def _validate_frontier_convergence(root: Path) -> None:
    """Require a published artifact to be query-ready, not merely ingested."""
    readiness = raw_materialization_readiness_snapshot(root)
    if not raw_materialization_ready(readiness):
        raise RuntimeError("seeded archive is missing completed raw-authority frontier convergence")


def _manifest_from_payload(payload: object) -> CorpusArtifactManifest:
    if not isinstance(payload, dict):
        raise ValueError("seeded archive manifest must be an object")
    _reject_semantic_metadata(
        {key: value for key, value in payload.items() if key not in {"facts", "files"}},
        location="seeded archive manifest",
    )
    stored_manifest_id = payload.pop("manifest_id", None)
    if not isinstance(stored_manifest_id, str):
        raise ValueError("seeded archive manifest is missing manifest_id")
    raw_facts = payload.pop("facts", None)
    if not isinstance(raw_facts, list) or not all(isinstance(item, dict) for item in raw_facts):
        raise ValueError("seeded archive manifest has malformed facts")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not all(isinstance(item, dict) for item in raw_files):
        raise ValueError("seeded archive manifest has malformed files")
    if not isinstance(payload.get("receipt"), dict):
        raise ValueError("seeded archive manifest has malformed receipt")
    try:
        facts = tuple(SyntheticArtifactFacts(**item) for item in raw_facts)
        manifest = CorpusArtifactManifest(facts=facts, **payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("seeded archive manifest has malformed metadata") from exc
    if stored_manifest_id != manifest.manifest_id:
        raise ValueError("seeded archive manifest identity mismatch")
    return manifest


def _read_manifest_fd(fd: int) -> CorpusArtifactManifest:
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), "r", encoding="utf-8") as handle:
        return _manifest_from_payload(json.load(handle))


def _read_manifest(path: Path) -> CorpusArtifactManifest:
    fd = _open_no_follow(path, os.O_RDONLY)
    try:
        return _read_manifest_fd(fd)
    finally:
        os.close(fd)


def _manifest_binds_to_key(manifest: CorpusArtifactManifest, root: Path, key: SeededArchiveKey) -> bool:
    """Require manifest metadata to agree with the cache key and publication path."""
    if not isinstance(manifest.receipt, dict):
        return False
    if manifest.protocol_version != _ARTIFACT_PROTOCOL_VERSION:
        return False
    if manifest.key != key.value:
        return False
    if manifest.archive_id != f"archive:seeded:{root.name}":
        return False
    if manifest.profile_id != _profile_id(key):
        return False
    if manifest.recipe_id != key.recipe_id:
        return False
    if manifest.source_semantics_id != key.source_semantics_id:
        return False
    if manifest.archive_schema_id != key.archive_schema_id:
        return False
    if not _valid_build_id(manifest.build_id):
        return False
    if manifest.build_id != manifest.receipt.get("build_id"):
        return False
    if not _valid_build_id(manifest.receipt.get("build_id")):
        return False
    expected_receipt = _canonical_receipt(
        key=key,
        archive_id=manifest.archive_id,
        profile_id=manifest.profile_id,
        build_id=manifest.build_id,
    ).to_payload()
    return manifest.receipt == dict(expected_receipt)


_GC_WORKTREE_MARKERS = (".worktree", ".worktree.lock", ".artifact-worktree.lock")
_GC_LEASE_MARKERS = (".lease", ".artifact.lease", ".query.lease")
_GC_CONTROL_MARKERS = frozenset((*_GC_WORKTREE_MARKERS, *_GC_LEASE_MARKERS))
# The cache, per-key, and final-root flocks hold across every inspect/delete
# interval, so an in-flight build, lease, or clone is already excluded without
# reference to age. Leases carry no time bound of their own; this covers only
# the unlocked instant between publication and the first lease of a new tree.
SEEDED_ARTIFACT_GC_GRACE_PERIOD_S = 10 * 60


def _gc_tree_size(root: Path) -> int:
    """Measure regular files without following a corrupt link node."""
    total = 0
    try:
        for path in _pinned_paths(root):
            if _is_regular(path):
                total += _safe_stat(path).st_size
    except (OSError, RuntimeError, ValueError):
        return total
    return total


def _gc_manifest_integrity(root: Path) -> tuple[CorpusArtifactManifest | None, int, str | None]:
    """Check only the authenticated final-tree shape needed before GC.

    Full semantic validation remains the build/query authority. GC does not
    rebuild or reinterpret an artifact; it merely refuses to delete anything
    whose self-authenticated manifest or content-addressed file set is not
    intact.
    """
    size = _gc_tree_size(root)
    try:
        manifest = _read_manifest(root / "manifest.json")
        match = _SEEDED_KEY.fullmatch(manifest.key)
        if match is None or root.name != match.group(1):
            return manifest, size, "manifest key does not match final path"
        expected = _manifest_file_entries(manifest.files)
        expected_paths = {relative for relative, _, _ in expected}
        actual_paths: set[str] = set()
        for path in _pinned_paths(root):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0] in _GC_CONTROL_MARKERS:
                continue
            if _is_symlink_node(path):
                return manifest, size, f"symlink node: {relative}"
            if _is_regular(path) and not _is_reserved_root_file(path, root):
                actual_paths.add(str(relative))
        if actual_paths != expected_paths:
            return manifest, size, "manifest file set does not match final tree"
        for expected_relative, expected_size, expected_digest in expected:
            path = root / expected_relative
            if not _is_regular(path) or _safe_stat(path).st_size != expected_size or _sha256(path) != expected_digest:
                return manifest, size, f"file digest mismatch: {expected_relative}"
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, size, f"unreadable or malformed artifact: {type(exc).__name__}"
    return manifest, size, None


def _try_exclusive_path_lock(path: Path) -> tuple[int | None, bool]:
    """Return ``(fd, True)`` for an acquired lock, ``(None, False)`` if busy."""
    try:
        fd = _open_authenticated_lock(path, nonblocking=True)
    except BlockingIOError:
        return None, False
    except OSError as exc:
        if getattr(exc, "errno", None) in {11, 13}:
            return None, False
        raise
    return fd, True


def _gc_active_marker(root: Path, names: tuple[str, ...]) -> bool:
    for name in names:
        marker = root / name
        if not _safe_exists(marker):
            continue
        if name == ".worktree":
            return True
        if _is_symlink_node(marker):
            return True
        try:
            fd = _open_no_follow(marker, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return True
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    return False


def _write_gc_receipt(path: Path, report: ArtifactGcReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_private_text(path, json.dumps(report.to_payload(), sort_keys=True, indent=2) + "\n")


def gc_seeded_archive_artifacts(
    *,
    cache_root: Path,
    reachable_keys: Iterable[SeededArchiveKey | str],
    grace_period_s: float = SEEDED_ARTIFACT_GC_GRACE_PERIOD_S,
    now: float | None = None,
    dry_run: bool = True,
    delete_corrupt: bool = False,
    protected_worktrees: Iterable[Path] = (),
    receipt_path: Path | None = None,
) -> ArtifactGcReport:
    """Preview or delete unreachable, aged final seeded-artifact trees.

    ``reachable_keys`` is mandatory by design. It is the current workload
    authority, not a build-id heuristic: a different checkout can publish the
    same bytes and a current checkout can legitimately reuse an older build.
    Missing, malformed, or corrupt artifacts are retained and reported unless
    ``delete_corrupt`` explicitly authorizes age-gated deletion of a directory
    whose path identity is not reachable. Reachable corrupt artifacts are
    always retained so their recipe can rebuild them. Every
    destructive decision holds the cache, per-key, and final-root locks for
    the complete inspect/delete interval, so active builders, query leases,
    clones, and explicitly protected worktrees remain untouched.
    """
    if not math.isfinite(grace_period_s) or grace_period_s < 0:
        raise ValueError("artifact GC grace period must be finite and non-negative")
    reachable_values = {item.value if isinstance(item, SeededArchiveKey) else str(item) for item in reachable_keys}
    if any(_SEEDED_KEY.fullmatch(value) is None for value in reachable_values):
        raise ValueError("artifact GC reachability must use complete seeded archive keys")
    reachable = tuple(sorted(reachable_values))
    if not reachable:
        raise ValueError("artifact GC requires explicit current reachable keys")
    cache_root = cache_root.expanduser()
    artifacts_root = cache_root / "artifacts"
    if not artifacts_root.is_dir():
        report = ArtifactGcReport(cache_root, dry_run, grace_period_s, reachable, (), delete_corrupt)
        if receipt_path is not None:
            _write_gc_receipt(receipt_path, report)
        return report

    protected = tuple(path.expanduser().resolve(strict=False) for path in protected_worktrees)
    current_time = time.time() if now is None else now
    entries: list[ArtifactGcEntry] = []
    cache_fd = -1
    try:
        cache_fd = _open_pinned_dir(cache_root)
        try:
            fcntl.flock(cache_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report = ArtifactGcReport(
                cache_root,
                dry_run,
                grace_period_s,
                reachable,
                (
                    ArtifactGcEntry(
                        name="<cache>",
                        path=str(cache_root),
                        key=None,
                        manifest_id=None,
                        size_bytes=0,
                        age_seconds=None,
                        disposition=ArtifactGcDisposition.ACTIVE_LOCK,
                        detail="cache root is owned by an active builder",
                    ),
                ),
                delete_corrupt,
            )
            if receipt_path is not None:
                _write_gc_receipt(receipt_path, report)
            return report
        candidates = sorted(artifacts_root.iterdir(), key=lambda item: item.name)
        for root in candidates:
            if root.name.startswith("."):
                continue
            if _is_symlink_node(root) or not root.is_dir():
                entries.append(
                    ArtifactGcEntry(
                        root.name,
                        str(root),
                        None,
                        None,
                        _gc_tree_size(root),
                        None,
                        ArtifactGcDisposition.CORRUPT,
                        "final entry is not a directory",
                    )
                )
                continue
            manifest, size, corruption = _gc_manifest_integrity(root)
            key = manifest.key if manifest is not None else None
            manifest_id = manifest.manifest_id if manifest is not None else None
            path_key = f"seeded-archive:sha256:{root.name}" if re.fullmatch(r"[0-9a-f]{64}", root.name) else None
            try:
                newest_mtime = root.stat().st_mtime
                with contextlib.suppress(OSError):
                    newest_mtime = max(newest_mtime, (root / "manifest.json").stat().st_mtime)
                age = max(0.0, current_time - newest_mtime)
            except OSError:
                age = None
            if corruption is not None:
                if path_key in reachable:
                    entries.append(
                        ArtifactGcEntry(
                            root.name,
                            str(root),
                            path_key,
                            manifest_id,
                            size,
                            age,
                            ArtifactGcDisposition.CORRUPT,
                            f"reachable artifact is corrupt: {corruption}",
                        )
                    )
                    continue
                if delete_corrupt and path_key is not None:
                    key = path_key
                    deletion_detail = f"unreachable corrupt artifact: {corruption}"
                else:
                    entries.append(
                        ArtifactGcEntry(
                            root.name, str(root), key, manifest_id, size, age, ArtifactGcDisposition.CORRUPT, corruption
                        )
                    )
                    continue
            else:
                deletion_detail = None
            assert key is not None
            if key in reachable:
                entries.append(
                    ArtifactGcEntry(root.name, str(root), key, manifest_id, size, age, ArtifactGcDisposition.REACHABLE)
                )
                continue
            try:
                resolved_root = root.resolve(strict=True)
            except OSError:
                resolved_root = root
            if any(
                resolved_root == protected_root or resolved_root in protected_root.parents
                for protected_root in protected
            ):
                entries.append(
                    ArtifactGcEntry(
                        root.name,
                        str(root),
                        key,
                        manifest_id,
                        size,
                        age,
                        ArtifactGcDisposition.ACTIVE_WORKTREE,
                        "explicitly protected worktree",
                    )
                )
                continue
            if _gc_active_marker(root, _GC_LEASE_MARKERS):
                entries.append(
                    ArtifactGcEntry(
                        root.name,
                        str(root),
                        key,
                        manifest_id,
                        size,
                        age,
                        ArtifactGcDisposition.ACTIVE_LEASE,
                        "active lease marker",
                    )
                )
                continue
            if _gc_active_marker(root, _GC_WORKTREE_MARKERS):
                entries.append(
                    ArtifactGcEntry(
                        root.name,
                        str(root),
                        key,
                        manifest_id,
                        size,
                        age,
                        ArtifactGcDisposition.ACTIVE_WORKTREE,
                        "active worktree marker",
                    )
                )
                continue
            lock_path = cache_root / ".locks" / f"{root.name}.lock"
            if _is_symlink_node(lock_path):
                entries.append(
                    ArtifactGcEntry(
                        root.name,
                        str(root),
                        key,
                        manifest_id,
                        size,
                        age,
                        ArtifactGcDisposition.CORRUPT,
                        "per-key lock path is a symlink",
                    )
                )
                continue
            lock_fd, acquired = _try_exclusive_path_lock(lock_path)
            if not acquired:
                entries.append(
                    ArtifactGcEntry(
                        root.name,
                        str(root),
                        key,
                        manifest_id,
                        size,
                        age,
                        ArtifactGcDisposition.ACTIVE_LOCK,
                        "per-key build lock is held",
                    )
                )
                continue
            root_fd = -1
            try:
                root_fd = _open_pinned_dir(root)
                try:
                    fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    entries.append(
                        ArtifactGcEntry(
                            root.name,
                            str(root),
                            key,
                            manifest_id,
                            size,
                            age,
                            ArtifactGcDisposition.ACTIVE_LEASE,
                            "artifact root is leased",
                        )
                    )
                    continue
                if age is None or age < grace_period_s:
                    entries.append(
                        ArtifactGcEntry(
                            root.name,
                            str(root),
                            key,
                            manifest_id,
                            size,
                            age,
                            ArtifactGcDisposition.GRACE,
                            deletion_detail,
                        )
                    )
                elif dry_run:
                    entries.append(
                        ArtifactGcEntry(
                            root.name,
                            str(root),
                            key,
                            manifest_id,
                            size,
                            age,
                            ArtifactGcDisposition.STALE,
                            deletion_detail,
                        )
                    )
                else:
                    try:
                        _remove_tree(root)
                    except (OSError, RuntimeError, ValueError) as exc:
                        entries.append(
                            ArtifactGcEntry(
                                root.name,
                                str(root),
                                key,
                                manifest_id,
                                size,
                                age,
                                ArtifactGcDisposition.DELETION_FAILED,
                                str(exc),
                            )
                        )
                    else:
                        for memo_key, artifact in tuple(_VALIDATED_ARTIFACTS.items()):
                            if artifact.root == root:
                                _VALIDATED_ARTIFACTS.pop(memo_key, None)
                        entries.append(
                            ArtifactGcEntry(
                                root.name,
                                str(root),
                                key,
                                manifest_id,
                                size,
                                age,
                                ArtifactGcDisposition.DELETED,
                                deletion_detail,
                            )
                        )
            finally:
                if root_fd >= 0:
                    with contextlib.suppress(OSError):
                        fcntl.flock(root_fd, fcntl.LOCK_UN)
                    os.close(root_fd)
                if lock_fd is not None:
                    with contextlib.suppress(OSError):
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)
    finally:
        if cache_fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(cache_fd, fcntl.LOCK_UN)
            os.close(cache_fd)
    report = ArtifactGcReport(cache_root, dry_run, grace_period_s, reachable, tuple(entries), delete_corrupt)
    if receipt_path is not None:
        _write_gc_receipt(receipt_path, report)
    return report


_VALIDATED_ARTIFACTS: dict[tuple[str, str], SeededArchiveArtifact] = {}


class _ArtifactValidationContentionError(RuntimeError):
    """A good published artifact could not be inspected yet, not rejected."""


def _artifact_still_placed(artifact: SeededArchiveArtifact) -> bool:
    """Revalidate every published byte and permission before memoized reuse.

    Memoization skips SQLite semantic checks, but it must not skip the cache
    contract. This pass rejects same-size corruption, manifest replacement,
    unexpected files, symlinks, and any write bit on the root or descendants.
    """
    write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    try:
        if _is_symlink_node(artifact.root) or not stat.S_ISDIR(_safe_stat(artifact.root).st_mode):
            return False
        paths = (artifact.root, *_pinned_paths(artifact.root))
        for path in paths:
            if _is_symlink_node(path) or _safe_stat(path).st_mode & write_bits:
                return False
        manifest_path = artifact.root / "manifest.json"
        if _is_symlink_node(manifest_path) or not _is_regular(manifest_path):
            return False
        disk_manifest = _read_manifest(manifest_path)
        if disk_manifest != artifact.manifest:
            return False
        file_entries = _manifest_file_entries(artifact.manifest.files)
        expected_paths = {path for path, _, _ in file_entries}
        actual_paths: set[str] = set()
        for path in _pinned_paths(artifact.root):
            if _is_symlink_node(path):
                return False
            if _is_regular(path) and not _is_reserved_root_file(path, artifact.root):
                actual_paths.add(str(path.relative_to(artifact.root)))
        if actual_paths != expected_paths:
            return False
        for relative, size, digest in file_entries:
            path = artifact.root / relative
            if _is_symlink_node(path) or not _is_regular(path):
                return False
            if _safe_stat(path).st_size != size or _sha256(path) != digest:
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _memoized_artifact(memo_key: tuple[str, str]) -> SeededArchiveArtifact | None:
    """Return this process's already-validated artifact, if it is still placed."""
    artifact = _VALIDATED_ARTIFACTS.get(memo_key)
    if artifact is None:
        return None
    if not _artifact_still_placed(artifact):
        _VALIDATED_ARTIFACTS.pop(memo_key, None)
        return None
    return artifact


def _validate_artifact(root: Path, key: SeededArchiveKey) -> SeededArchiveArtifact | None:
    try:
        if _is_symlink_node(root) or not stat.S_ISDIR(_safe_stat(root).st_mode):
            return None
        manifest_path = root / "manifest.json"
        if _is_symlink_node(manifest_path) or not _is_regular(manifest_path):
            return None
    except (OSError, ValueError):
        return None
    try:
        manifest = _read_manifest(manifest_path)
        if manifest.protocol_version != _ARTIFACT_PROTOCOL_VERSION or manifest.key != key.value:
            return None
        file_entries = _manifest_file_entries(manifest.files)
        expected_paths = {path for path, _, _ in file_entries}
        actual_paths: set[str] = set()
        for path in _pinned_paths(root):
            if _is_symlink_node(path):
                return None
            if _is_regular(path) and not _is_reserved_root_file(path, root):
                actual_paths.add(str(path.relative_to(root)))
        if actual_paths != expected_paths:
            return None
        for relative, size, digest in file_entries:
            path = root / relative
            if (
                _is_symlink_node(path)
                or not _is_regular(path)
                or _safe_stat(path).st_size != size
                or _sha256(path) != digest
            ):
                return None
        _sqlite_integrity(root)
        expected_facts = _canonical_facts(key)
        if json.dumps([asdict(fact) for fact in manifest.facts], sort_keys=True) != json.dumps(
            [asdict(fact) for fact in expected_facts], sort_keys=True
        ):
            return None
        _validate_facts(root, manifest.facts)
        _validate_frontier_convergence(root)
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        for path in (root, *_pinned_paths(root)):
            if _is_symlink_node(path) or _safe_stat(path).st_mode & write_bits:
                return None
        if not _manifest_binds_to_key(manifest, root, key):
            return None
    except sqlite3.Error as exc:
        if is_transient_sqlite_lock(exc):
            raise _ArtifactValidationContentionError(f"published artifact validation is contended: {root}") from exc
        return None
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return SeededArchiveArtifact(root=root, manifest=manifest)


def _validate_artifact_with_retry(root: Path, key: SeededArchiveKey) -> SeededArchiveArtifact | None:
    """Wait briefly for a concurrent reader before treating an artifact as stale."""
    deadline = time.monotonic() + 5.0
    attempt = 0
    while True:
        try:
            return _validate_artifact(root, key)
        except _ArtifactValidationContentionError:
            if time.monotonic() >= deadline:
                raise
            attempt += 1
            gc.collect()
            time.sleep(min(0.05 * attempt, 0.5))


def _assert_no_symlinks(root: Path) -> None:
    """Reject symlink nodes before any traversal operation can follow them."""
    _assert_no_symlink_ancestors(root.parent, allow_missing_leaf=False)
    if _is_symlink_node(root):
        raise ValueError(f"symlink root is not allowed: {root}")
    for path in _pinned_paths(root):
        if _is_symlink_node(path):
            raise ValueError(f"symlink node is not allowed: {path}")


def _make_read_only(root: Path) -> None:
    """Remove write permission without ever following a symlink node."""
    if _is_symlink_node(root):
        raise ValueError(f"cannot seal symlink root: {root}")
    for path in sorted(_pinned_paths(root), key=lambda item: len(item.parts), reverse=True):
        if _is_symlink_node(path):
            raise ValueError(f"cannot seal symlink node: {path}")
        mode = _safe_stat(path).st_mode
        _safe_chmod(path, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    mode = _safe_stat(root).st_mode
    _safe_chmod(root, mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def _rename_sealed(source: Path, destination: Path) -> None:
    """Rename a sealed tree after opening only its parent directories."""
    parents = {source.parent, destination.parent}
    original_modes: dict[Path, int] = {}
    try:
        for parent in parents:
            if _is_symlink_node(parent):
                raise ValueError(f"rename parent is a symlink: {parent}")
            original_modes[parent] = _safe_stat(parent).st_mode
            _safe_chmod(parent, original_modes[parent] | stat.S_IWUSR)
        _safe_replace(source, destination)
    finally:
        for parent, mode in original_modes.items():
            try:
                _safe_chmod(parent, mode)
            except FileNotFoundError:
                pass


def _publish_sealed_staging(staging: Path, final_root: Path) -> None:
    """Seal, atomically hand off, then leave no writable final on failure.

    The source tree is sealed before the first rename attempt. Some filesystems
    reject renaming a read-only directory even though POSIX permits it. In
    that case a fresh writable handoff copy is made from the sealed source;
    the source itself is never reopened, and the handoff is still atomically
    renamed before the final tree is sealed.
    """
    _assert_no_symlinks(staging)
    _make_read_only(staging)
    handoff: Path | None = None
    try:
        try:
            _rename_sealed(staging, final_root)
        except PermissionError:
            # Cross-parent moves of sealed directories are rejected by this
            # host's VFS.  Copy into a sealed sibling of the final root, then
            # rename within the final directory's parent (the same-parent
            # operation remains atomic without reopening either tree).
            handoff = final_root.parent / f".{final_root.name}.{uuid.uuid4().hex}.handoff"
            _copy_tree(staging, handoff)
            # Seal the handoff before it can be renamed.  A killed copy can
            # leave only a private handoff; final visibility is one rename of
            # an already sealed tree, never a writable directory.
            _assert_no_symlinks(handoff)
            _make_read_only(handoff)
            # Remove the original staging tree before visibility. Any cleanup
            # failure therefore leaves no published final root behind.
            _remove_tree(staging)
            _rename_sealed(handoff, final_root)
    except Exception:
        if _safe_exists(final_root):
            _remove_tree(final_root)
        if handoff is not None and _safe_exists(handoff):
            _remove_tree(handoff)
        if _safe_exists(staging):
            _remove_tree(staging)
        raise


@dataclass(frozen=True)
class _LockDomain:
    ancestor_fd: int
    ancestor_mode: int
    root_fd: int
    root_mode: int
    locks_fd: int


def _assert_named_directory(fd: int, name: str, opened_fd: int) -> None:
    named = os.stat(name, dir_fd=fd, follow_symlinks=False)
    opened = os.fstat(opened_fd)
    if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino):
        raise OSError("lock domain pathname was replaced while locked")


def _open_lock_domain(cache_root: Path) -> _LockDomain:
    """Pin and protect a stable ancestor, cache root, and ``.locks``."""
    locks = cache_root / ".locks"
    _mkdir_pinned(cache_root / "artifacts")
    _mkdir_pinned(locks)
    _mkdir_pinned(cache_root / ".staging")
    for control_file in (
        cache_root / ".cleanup.lock",
        cache_root / ".cleanup.cursor",
        cache_root / ".handoff.cursor",
        cache_root / ".cleanup.seen",
        cache_root / ".handoff.seen",
    ):
        fd = _open_no_follow(control_file, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(fd)

    ancestor_fd, root_name = _open_pinned_parent(cache_root)
    root_fd = -1
    locks_fd = -1
    ancestor_mode = 0
    root_mode = 0
    try:
        ancestor_mode = os.fstat(ancestor_fd).st_mode
        fcntl.flock(ancestor_fd, fcntl.LOCK_EX)
        root_fd = os.open(root_name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=ancestor_fd)
        _assert_named_directory(ancestor_fd, root_name, root_fd)
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        _assert_named_directory(ancestor_fd, root_name, root_fd)
        locks_fd = os.open(".locks", os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=root_fd)
        _assert_named_directory(root_fd, ".locks", locks_fd)
        fcntl.flock(locks_fd, fcntl.LOCK_EX)
        root_mode = os.fstat(root_fd).st_mode
        # Prevent replacement of cache_root or cache_root/.locks while the
        # descriptor capabilities are live. Child directories remain writable.
        os.fchmod(ancestor_fd, ancestor_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        os.fchmod(root_fd, root_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        _assert_named_directory(ancestor_fd, root_name, root_fd)
        _assert_named_directory(root_fd, ".locks", locks_fd)
        return _LockDomain(
            ancestor_fd=ancestor_fd,
            ancestor_mode=ancestor_mode,
            root_fd=root_fd,
            root_mode=root_mode,
            locks_fd=locks_fd,
        )
    except BaseException:
        if locks_fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(locks_fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(locks_fd)
        if root_fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(root_fd)
        with contextlib.suppress(OSError):
            fcntl.flock(ancestor_fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(ancestor_fd)
        raise


def _release_lock_domain(domain: _LockDomain) -> None:
    """Restore modes while capabilities remain locked, then close independently."""
    try:
        with contextlib.suppress(OSError):
            os.fchmod(domain.root_fd, domain.root_mode)
        with contextlib.suppress(OSError):
            os.fchmod(domain.ancestor_fd, domain.ancestor_mode)
    finally:
        try:
            with contextlib.suppress(OSError):
                fcntl.flock(domain.locks_fd, fcntl.LOCK_UN)
        finally:
            try:
                with contextlib.suppress(OSError):
                    fcntl.flock(domain.root_fd, fcntl.LOCK_UN)
            finally:
                try:
                    with contextlib.suppress(OSError):
                        fcntl.flock(domain.ancestor_fd, fcntl.LOCK_UN)
                finally:
                    try:
                        os.close(domain.locks_fd)
                    finally:
                        try:
                            os.close(domain.root_fd)
                        finally:
                            os.close(domain.ancestor_fd)


def build_seeded_archive(
    specs: Iterable[CorpusSpec] | None = None,
    *,
    cache_root: Path | None = None,
) -> SeededArchiveArtifact:
    selected_root = (cache_root or default_cache_root()).expanduser()
    domain = _open_lock_domain(selected_root)
    try:
        return _build_seeded_archive_inner(specs, cache_root=selected_root)
    finally:
        _release_lock_domain(domain)


def _build_seeded_archive_inner(
    specs: Iterable[CorpusSpec] | None = None,
    *,
    cache_root: Path | None = None,
) -> SeededArchiveArtifact:
    """Build-or-reuse one atomic immutable real-pipeline archive artifact."""
    selected_specs = tuple(specs) if specs is not None else (c03_semantic_corpus_spec(),)
    if not selected_specs:
        raise ValueError("seeded archive requires at least one named corpus specification")
    key = seeded_archive_key(selected_specs)
    cache_root = (cache_root or default_cache_root()).expanduser()
    memo_key = (str(cache_root), key.value)
    # Validate-once-per-process: after this process has fully validated an
    # artifact, later hits skip BOTH the per-key flock and the full
    # revalidation (re-SHA256 of every file, five ``PRAGMA quick_check``
    # runs, the planted-facts query, and the frontier-convergence read).
    # That work is per-CACHE-HIT today, so a module whose every test seeds
    # the same named workload pays it once per test. Memoizing here and not
    # inside ``_validate_artifact`` is deliberate: the lock acquisition is
    # itself contended under xdist, and a memo behind the lock would still
    # serialize every worker's every test on it.
    memoized = _memoized_artifact(memo_key)
    if memoized is not None:
        return memoized
    _assert_no_symlink_ancestors(cache_root)
    artifacts = cache_root / "artifacts"
    locks = cache_root / ".locks"
    staging_root = cache_root / ".staging"
    _mkdir_pinned(artifacts)
    _mkdir_pinned(locks)
    _mkdir_pinned(staging_root)
    _assert_no_symlink_ancestors(artifacts, allow_missing_leaf=False)
    _assert_no_symlink_ancestors(locks, allow_missing_leaf=False)
    _assert_no_symlink_ancestors(staging_root, allow_missing_leaf=False)
    final_root = artifacts / key.value.rsplit(":", 1)[-1]
    lock_path = locks / f"{final_root.name}.lock"

    lock_fd = _open_authenticated_lock(lock_path)
    with os.fdopen(lock_fd, "a+", encoding="utf-8") as lock_handle:
        _assert_lock_identity(lock_handle.fileno(), lock_path)
        _recover_stale_staging(staging_root=staging_root, artifact_name=final_root.name)
        _assert_lock_identity(lock_handle.fileno(), lock_path)
        _recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root)
        _assert_lock_identity(lock_handle.fileno(), lock_path)
        _recover_stale_handoffs(cache_root=cache_root, artifacts_root=artifacts)
        _assert_lock_identity(lock_handle.fileno(), lock_path)
        cached = _validate_artifact_with_retry(final_root, key)
        if cached is not None:
            _VALIDATED_ARTIFACTS[memo_key] = cached
            return cached
        if _safe_exists(final_root):
            _assert_lock_identity(lock_handle.fileno(), lock_path)
            _remove_tree(final_root)
        for attempt in range(1, _BUILD_LOCK_ATTEMPTS + 1):
            staging = staging_root / f"{final_root.name}.{uuid.uuid4().hex}"
            _mkdir_pinned(staging)
            try:
                corpus_root = staging / "wire"
                written_batches = tuple(
                    SyntheticCorpus.write_spec_artifacts(spec, corpus_root / spec.provider, prefix=f"seed-{index:02d}")
                    for index, spec in enumerate(selected_specs)
                )
                sources: list[Source] = []
                for spec, written in zip(selected_specs, written_batches, strict=True):
                    source_paths = (
                        (written.files[0].parent.parent,) if spec.provider == "antigravity" else written.files
                    )
                    package = provider_source_package(spec.provider, written.files, source_paths=source_paths)
                    sources.extend(package.admitted_sources())
                with _configured_archive_root(staging):
                    with patch(
                        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
                        SyntheticAntigravityLanguageServerClient,
                    ):
                        # This fixture builds many tiny, isolated archives under
                        # pytest-xdist.  Letting each outer test worker inherit
                        # the production cpu-count default creates a nested
                        # process fan-out (xdist workers × parse workers) whose
                        # spawn/import memory dwarfs the corpus itself.  Pool
                        # semantics have dedicated importer tests; this helper's
                        # contract is the real admission/write route, which the
                        # exact sequential escape hatch preserves.
                        asyncio.run(parse_sources_archive(staging, sources, parse_workers=1))
                blob_report = scan_blob_integrity(
                    staging / "source.db",
                    store=BlobStore(staging / "blob"),
                    full=True,
                    configured_root=staging,
                )
                orphan_finding = next(
                    (finding for finding in blob_report.findings if finding.kind == "orphan_blobs"), None
                )
                if orphan_finding is not None:
                    orphan_hashes = set(orphan_finding.sample)
                    receipts = inspect_blob_publication_receipts(
                        staging / "source.db", staging / "blob", index_db_path=staging / "index.db"
                    )
                    abandoned = abandon_blob_publication_receipts(
                        staging / "source.db",
                        staging / "blob",
                        [receipt.publication_id for receipt in receipts if receipt.blob_hash in orphan_hashes],
                        confirmed=True,
                        index_db_path=staging / "index.db",
                    )
                    deleted, _deleted_bytes, blockers = unlink_unreferenced_blob_hashes_under_exclusion(
                        staging / "source.db", staging / "index.db", staging / "blob", orphan_hashes
                    )
                    if blockers or abandoned.abandoned != orphan_finding.count or deleted != orphan_finding.count:
                        raise AssertionError(
                            "seeded archive orphan disposition failed: "
                            f"found={orphan_finding.count} abandoned={abandoned.abandoned} deleted={deleted} "
                            f"blockers={blockers}"
                        )
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
                build_id = _build_id()
                receipt = _canonical_receipt(
                    key=key,
                    archive_id=archive_id,
                    profile_id=profile_id,
                    build_id=build_id,
                )
                manifest = CorpusArtifactManifest(
                    protocol_version=_ARTIFACT_PROTOCOL_VERSION,
                    key=key.value,
                    archive_id=archive_id,
                    profile_id=profile_id,
                    build_id=build_id,
                    recipe_id=key.recipe_id,
                    source_semantics_id=key.source_semantics_id,
                    archive_schema_id=key.archive_schema_id,
                    facts=facts,
                    files=_archive_files(staging),
                    receipt=dict(receipt.to_payload()),
                )
                _write_private_text(
                    staging / "manifest.json",
                    json.dumps(manifest.to_payload(), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                )
                _publish_sealed_staging(staging, final_root)
                break
            except sqlite3.OperationalError as exc:
                # Same-process zombie-connection lock (polylogue-lbgc): a
                # not-yet-finalized cursor from earlier in this worker keeps
                # SQLite's shared pager-cache entry alive, and SQLITE_LOCKED is
                # NOT absorbed by the busy-timeout. The module already applies
                # this exact remedy to the DELETE-journal pragma; the real-ingest
                # build needs it too, because a cache-invalidating key change
                # makes many workers rebuild artifacts at once and a setup ERROR
                # here fails the whole consuming test.
                _remove_tree(staging)
                if not is_transient_sqlite_lock(exc) or attempt == _BUILD_LOCK_ATTEMPTS:
                    raise
                gc.collect()
                time.sleep(min(0.25 * attempt, 2.0))
            except Exception:
                _remove_tree(staging)
                raise
        artifact = _validate_artifact_with_retry(final_root, key)
        if artifact is None:
            raise RuntimeError("published seeded archive failed its own validation")
        _VALIDATED_ARTIFACTS[memo_key] = artifact
        return artifact


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short cache write")
        view = view[written:]


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy a tree through pinned directory descriptors, never shutil/pathname walks."""
    src_fd = _open_pinned_dir(source)
    try:
        dst_parent, dst_leaf = _open_pinned_parent(destination, create=True)
    except BaseException:
        os.close(src_fd)
        raise
    try:
        original_parent_mode = os.fstat(dst_parent).st_mode
    except BaseException:
        os.close(dst_parent)
        os.close(src_fd)
        raise
    dst_fd = -1
    try:
        os.fchmod(dst_parent, original_parent_mode | stat.S_IWUSR)
        os.mkdir(dst_leaf, 0o700, dir_fd=dst_parent)
        dst_fd = os.open(dst_leaf, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=dst_parent)
    except BaseException:
        with contextlib.suppress(OSError):
            os.fchmod(dst_parent, original_parent_mode)
        os.close(dst_parent)
        os.close(src_fd)
        raise
    try:

        def copy_dir(src: int, dst: int) -> None:
            with os.scandir(src) as entries:
                for entry in entries:
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(info.st_mode):
                        raise ValueError(f"cannot copy symlink node: {entry.name}")
                    if stat.S_ISDIR(info.st_mode):
                        os.mkdir(entry.name, (info.st_mode & 0o777) | stat.S_IWUSR, dir_fd=dst)
                        child_src = os.open(entry.name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=src)
                        child_dst = -1
                        try:
                            child_dst = os.open(entry.name, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW, dir_fd=dst)
                        except BaseException:
                            os.close(child_src)
                            raise
                        try:
                            copy_dir(child_src, child_dst)
                            os.fchmod(child_dst, info.st_mode & 0o777)
                        finally:
                            os.close(child_src)
                            os.close(child_dst)
                    elif stat.S_ISREG(info.st_mode):
                        in_fd = os.open(entry.name, os.O_RDONLY | _O_NOFOLLOW, dir_fd=src)
                        out_fd = -1
                        try:
                            out_fd = os.open(
                                entry.name,
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
                                info.st_mode & 0o777,
                                dir_fd=dst,
                            )
                        except BaseException:
                            os.close(in_fd)
                            raise
                        try:
                            while True:
                                chunk = os.read(in_fd, 1024 * 1024)
                                if not chunk:
                                    break
                                _write_all(out_fd, chunk)
                            os.fchmod(out_fd, info.st_mode & 0o777)
                        finally:
                            os.close(in_fd)
                            os.close(out_fd)
                    else:
                        raise ValueError(f"unsupported cache node: {entry.name}")

        copy_dir(src_fd, dst_fd)
    finally:
        os.close(src_fd)
        os.close(dst_fd)
        with contextlib.suppress(OSError):
            os.fchmod(dst_parent, original_parent_mode)
        os.close(dst_parent)


def _authenticate_clone_copy(
    source: SeededArchiveArtifact,
    destination: Path,
    manifest: CorpusArtifactManifest,
    *,
    ignored_relatives: frozenset[str] = frozenset(),
) -> None:
    expected = tuple(item for item in _manifest_file_entries(manifest.files) if item[0] not in ignored_relatives)
    expected_paths = {relative for relative, _, _ in expected}
    actual_paths: set[str] = set()
    for path in _pinned_paths(destination):
        relative = str(path.relative_to(destination))
        if _is_regular(path) and not _is_reserved_root_file(path, destination) and relative not in ignored_relatives:
            actual_paths.add(relative)
        elif relative in ignored_relatives:
            continue
        elif not _is_regular(path) and not stat.S_ISDIR(_safe_stat(path).st_mode):
            raise ValueError(f"clone contains unsupported node: {relative}")

    if actual_paths != expected_paths:
        raise ValueError("clone contains unexpected or missing files")

    def authenticate_pair(relative: str, size: int, digest: str) -> None:
        source_fd = -1
        clone_fd = -1
        try:
            source_fd = _open_file_fd(source.root / relative)
            clone_fd = _open_file_fd(destination / relative)
            source_stat = os.fstat(source_fd)
            clone_stat = os.fstat(clone_fd)
            if not stat.S_ISREG(clone_stat.st_mode) or clone_stat.st_size != size:
                raise ValueError(f"clone file metadata mismatch: {relative}")
            if (source_stat.st_dev, source_stat.st_ino) == (clone_stat.st_dev, clone_stat.st_ino):
                raise ValueError(f"clone file inode was not detached: {relative}")
            if _sha256_fd(clone_fd) != digest:
                raise ValueError(f"clone file content mismatch: {relative}")
        finally:
            if source_fd >= 0:
                os.close(source_fd)
            if clone_fd >= 0:
                os.close(clone_fd)

    for relative, size, digest in expected:
        authenticate_pair(relative, size, digest)

    source_manifest_fd = -1
    clone_manifest_fd = -1
    try:
        source_manifest_fd = _open_file_fd(source.root / "manifest.json")
        clone_manifest_fd = _open_file_fd(destination / "manifest.json")
        source_manifest_stat = os.fstat(source_manifest_fd)
        clone_manifest_stat = os.fstat(clone_manifest_fd)
        if (source_manifest_stat.st_dev, source_manifest_stat.st_ino) == (
            clone_manifest_stat.st_dev,
            clone_manifest_stat.st_ino,
        ):
            raise ValueError("clone manifest inode was not detached")
        if _sha256_fd(source_manifest_fd) != _sha256_fd(clone_manifest_fd):
            raise ValueError("clone manifest bytes mismatch")
        if _read_manifest_fd(clone_manifest_fd) != manifest:
            raise ValueError("clone manifest mismatch")
    finally:
        if source_manifest_fd >= 0:
            os.close(source_manifest_fd)
        if clone_manifest_fd >= 0:
            os.close(clone_manifest_fd)


@contextlib.contextmanager
def _shared_artifact_read_locks(root: Path) -> Iterator[None]:
    """Hold cache and publication-root leases while reading an artifact."""
    cache_root = root.parent.parent
    cache_fd = _open_pinned_dir(cache_root)
    key_fd = -1
    root_fd = -1
    try:
        fcntl.flock(cache_fd, fcntl.LOCK_SH)
        try:
            key_fd = _open_authenticated_lock(cache_root / ".locks" / f"{root.name}.lock", shared=True)
        except FileNotFoundError:
            pass
        root_fd = _open_pinned_dir(root)
        fcntl.flock(root_fd, fcntl.LOCK_SH)
        yield
    finally:
        if root_fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)
        if key_fd >= 0:
            with contextlib.suppress(OSError):
                fcntl.flock(key_fd, fcntl.LOCK_UN)
            os.close(key_fd)
        with contextlib.suppress(OSError):
            fcntl.flock(cache_fd, fcntl.LOCK_UN)
        os.close(cache_fd)


@contextlib.contextmanager
def _shared_seeded_artifact_read_locks(artifact: SeededArchiveArtifact) -> Iterator[None]:
    """Hold the cache, per-key, and source-root leases during a clone.

    Lock order is cache root, per-key lock, then final artifact root.  GC
    acquires the same sequence exclusively, so it cannot unlink the source
    after validation and before the clone has finished reading it.
    """
    with _shared_artifact_read_locks(artifact.root):
        yield


def acquire_query_only_seeded_archive(
    artifact: SeededArchiveArtifact,
    key: SeededArchiveKey,
) -> SeededArchiveQueryLease:
    """Pin a shared source and validate it before every lease-mediated use."""
    root_fd = -1
    try:
        root_fd = _open_pinned_dir(artifact.root)
        fcntl.flock(root_fd, fcntl.LOCK_SH)
        lease = SeededArchiveQueryLease(artifact=artifact, key=key, _root_fd=root_fd)
        lease._assert_current()
        return lease
    except BaseException:
        if root_fd >= 0:
            with contextlib.suppress(OSError):
                os.close(root_fd)
        raise


def clone_seeded_archive(artifact: SeededArchiveArtifact, destination: Path) -> SeededArchiveClone:
    """Create an authenticated private clone with a clone-scoped capability.

    Only the returned clone root is pinned and shared-locked.  Ancestor
    directories remain entirely caller-owned, so closing one clone cannot
    restore modes or release locks belonging to a sibling.
    """
    with _shared_seeded_artifact_read_locks(artifact):
        _assert_no_symlinks(artifact.root)
        disk_manifest = _read_manifest(artifact.root / "manifest.json")
        if disk_manifest != artifact.manifest:
            raise ValueError("published artifact manifest changed before clone")
        if _is_symlink_node(destination):
            _safe_unlink(destination)
        _remove_tree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        method = "reflink"
        try:
            subprocess.run(
                ["cp", "-a", "--reflink=always", str(artifact.root), str(destination)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            _remove_tree(destination)
            try:
                _copy_tree(artifact.root, destination)
            except BaseException:
                with contextlib.suppress(BaseException):
                    _remove_tree(destination)
                raise
            method = "copy"
        integrity_fd = -1
        try:
            _assert_no_symlinks(destination)
            # Authenticate inode detachment before chmodding the writable
            # clone. A hostile hardlink would otherwise receive the write bit
            # through the destination path and mutate the source inode.
            _authenticate_clone_copy(
                artifact,
                destination,
                disk_manifest,
                ignored_relatives=frozenset({".maintenance-state/durable-change-trains/.bootstrap"}),
            )
            for path in _pinned_paths(destination):
                _safe_chmod(path, _safe_stat(path).st_mode | stat.S_IWUSR)
            _safe_chmod(destination, _safe_stat(destination).st_mode | stat.S_IWUSR)
            bootstrap_marker = destination / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
            if stat.S_ISREG(_safe_stat(bootstrap_marker).st_mode):
                from polylogue.storage.sqlite.durable_change_train import _record_fresh_durable_bootstrap

                _safe_unlink(bootstrap_marker)
                _record_fresh_durable_bootstrap(destination)
            integrity_fd = _open_pinned_dir(destination)
            fcntl.flock(integrity_fd, fcntl.LOCK_SH)
            _authenticate_clone_copy(
                artifact,
                destination,
                disk_manifest,
                ignored_relatives=frozenset({".maintenance-state/durable-change-trains/.bootstrap"}),
            )
        except BaseException:
            if integrity_fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(integrity_fd)
            # Cleanup is deliberately best-effort, but never touches the
            # source tree.  _remove_tree unlinks residue without chmodding
            # regular files, preserving source modes for hardlink failures.
            with contextlib.suppress(BaseException):
                _remove_tree(destination)
            raise
        return SeededArchiveClone(
            root=destination,
            source_manifest_id=disk_manifest.manifest_id,
            clone_method=method,
            _integrity_fd=integrity_fd,
        )


__all__ = [
    "ArtifactGcDisposition",
    "ArtifactGcEntry",
    "ArtifactGcReport",
    "ImmutableTreeArtifact",
    "CorpusArtifactManifest",
    "BENCHMARK_WORKLOAD_PROFILES",
    "BenchmarkWorkloadProfile",
    "BenchmarkWorkloadTier",
    "NAMED_WORKLOAD_PROFILES",
    "NamedWorkloadProfile",
    "SeededArchiveArtifact",
    "SeededArchiveQueryLease",
    "WorkloadProfile",
    "WorkloadSessionShape",
    "acquire_query_only_seeded_archive",
    "build_immutable_tree",
    "clone_immutable_tree",
    "SeededArchiveClone",
    "SeededArchiveKey",
    "SeededArchiveReachabilityEntry",
    "SeededArchiveReachabilityInventory",
    "benchmark_corpus_specs",
    "benchmark_workload_profile",
    "benchmark_workload_tier",
    "build_benchmark_archive",
    "build_seeded_archive",
    "c03_semantic_corpus_spec",
    "clone_seeded_archive",
    "default_cache_root",
    "SEEDED_ARTIFACT_GC_GRACE_PERIOD_S",
    "gc_seeded_archive_artifacts",
    "named_corpus_specs",
    "named_workload_profile",
    "current_seeded_archive_reachability",
    "schema_coverage_corpus_specs",
    "seeded_archive_key",
    "validate_seeded_archive_reachability",
]
