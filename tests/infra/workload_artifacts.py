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
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

from polylogue.config import Config, Source
from polylogue.core.enums import Provider
from polylogue.core.sqlite_locking import is_transient_sqlite_lock
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
from polylogue.sources.origin_specs import (
    lowering_fingerprint,
    materializer_fingerprint,
    origin_specs,
    replay_routing_fingerprint,
)
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from tests.infra.source_builders import SyntheticAntigravityLanguageServerClient

_ARTIFACT_PROTOCOL_VERSION = 2
#: Bounded rebuild attempts when a same-process SQLite lock (SQLITE_LOCKED,
#: not SQLITE_BUSY) aborts an artifact build. See the retry site below.
_BUILD_LOCK_ATTEMPTS = 3
_SCRATCH_CACHE_ROOT = Path("/realm/tmp/polylogue-seeded-artifacts")
_CLOUD_CACHE_ROOT = Path("/tmp/polylogue-seeded-artifacts")


def default_cache_root() -> Path:
    """Where published artifacts live when a caller names no cache root.

    Mirrors :func:`devtools.verify_runs.resolve_pytest_basetemp_root`'s
    placement family: NVMe scratch when ``/realm/tmp`` is mounted, and the
    ``/tmp`` fallback only when ``/realm`` is absent entirely (a genuine cloud
    sandbox). The previous hard-coded ``/realm/tmp`` path made every consumer
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
_ARCHIVE_DB_NAMES = ("source.db", "index.db", "user.db", "ops.db", "embeddings.db")
_OBSOLETE_STAGING_SCAN_BUDGET = 32


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
class SeededArchiveManifest:
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


def _recipe_id(providers: Iterable[str] = ()) -> str:
    """Fingerprint the generation/materialization dependency and input closure.

    Only route-owned Python modules and runtime schema/config inputs participate.
    In particular, this must not become a repository-wide ``*.py`` fingerprint:
    an unrelated surface edit is not a semantic change to a seeded archive.
    """
    digest = hashlib.sha256()
    files: set[Path] = set()
    provider_roots = tuple(_RECIPE_PROVIDER_ROOT / provider for provider in sorted(set(providers)))
    for root in (*_SOURCE_DEPENDENCY_ROOTS, *_RECIPE_INPUT_ROOTS, *provider_roots):
        if root.is_file():
            files.add(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {
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
    providers = tuple(spec.provider for spec in selected_specs)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    fd = _open_no_follow(path, os.O_RDONLY)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    return digest.hexdigest()


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


def _assert_no_symlink_ancestors(path: Path, *, allow_missing_leaf: bool = True) -> None:
    """Reject symlinks in every existing ancestor before a path operation.

    ``Path.is_file``/``open`` perform a second pathname walk after a check;
    this guard is intentionally paired with descriptor-based opens for reads
    and lock files.  It still closes the common parent-substitution hole for
    operations that must use a pathname (rename and mkdir), and callers fail
    closed when an ancestor disappears during the operation.
    """
    absolute = path.absolute()
    current = absolute if not (allow_missing_leaf and not absolute.exists()) else absolute.parent
    while True:
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if current == absolute.parent or allow_missing_leaf:
                current = current.parent
                if current == current.parent:
                    break
                continue
            raise
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink ancestor is not allowed: {current}")
        if current.parent == current:
            break
        current = current.parent


def _open_no_follow(path: Path, flags: int, mode: int = 0o600) -> int:
    """Open one filesystem node without following a symlink at any level."""
    _assert_no_symlink_ancestors(path)
    return os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode)


def _safe_stat(path: Path) -> os.stat_result:
    _assert_no_symlink_ancestors(path)
    return os.stat(path, follow_symlinks=False)


def _safe_chmod(path: Path, mode: int) -> None:
    _assert_no_symlink_ancestors(path)
    os.chmod(path, mode, follow_symlinks=False)


def _archive_files(root: Path) -> tuple[dict[str, object], ...]:
    if _is_symlink_node(root):
        raise ValueError(f"cannot archive symlink root: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if _is_symlink_node(path):
            raise ValueError(f"cannot archive symlink node: {path}")
        if not path.is_file():
            continue
        if _is_reserved_root_file(path, root):
            continue
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return tuple(entries)


def _manifest_file_entries(files: tuple[dict[str, object], ...]) -> tuple[tuple[str, int, str], ...]:
    """Validate manifest file records before any keyed access or filesystem use."""
    entries: list[tuple[str, int, str]] = []
    for item in files:
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
    for name in _ARCHIVE_DB_NAMES:
        path = root / name
        if _is_symlink_node(path):
            raise RuntimeError(f"seeded archive tier is a symlink: {name}")
        if not path.exists():
            continue
        with contextlib.closing(sqlite3.connect(path)) as conn, conn:
            quick = conn.execute("PRAGMA quick_check").fetchone()
            foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
            if quick != ("ok",) or foreign:
                raise RuntimeError(f"invalid seeded archive tier {name}")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            _journal_mode_delete_with_retry(conn, name=name)
        checked.append(name)
    for name in checked:
        for suffix in ("-wal", "-shm"):
            sidecar = root / f"{name}{suffix}"
            if sidecar.exists():
                sidecar.unlink()


def _remove_tree(path: Path) -> None:
    """Remove a locally-owned stale artifact without following symlinks."""
    _assert_no_symlink_ancestors(path.parent, allow_missing_leaf=False)
    if _is_symlink_node(path) or path.is_file():
        path.unlink(missing_ok=True)
        return
    if not path.exists():
        return
    for candidate in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if _is_symlink_node(candidate):
            candidate.unlink(missing_ok=True)
            continue
        try:
            _safe_chmod(candidate, _safe_stat(candidate).st_mode | stat.S_IWUSR)
        except FileNotFoundError:
            continue
    _safe_chmod(path, _safe_stat(path).st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def _recover_stale_staging(*, staging_root: Path, artifact_name: str) -> tuple[str, ...]:
    """Remove crash-left staging trees while never racing an active builder."""
    removed: list[str] = []
    for candidate in sorted(staging_root.glob(f"{artifact_name}.*")):
        if _is_symlink_node(candidate):
            _remove_tree(candidate)
            removed.append(candidate.name)
            continue
        if not candidate.is_dir():
            continue
        _remove_tree(candidate)
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


def _bounded_cache_candidates(directory: Path, *, cursor: str, budget: int, suffix: str | None = None) -> list[Path]:
    """Enumerate at most ``budget`` candidates, without sorting the directory."""
    if budget <= 0:
        return []
    _assert_no_symlink_ancestors(directory, allow_missing_leaf=False)
    selected: list[Path] = []
    for candidate in os.scandir(directory):
        if len(selected) >= budget:
            break
        if cursor and candidate.name <= cursor:
            continue
        if suffix is not None and not candidate.name.endswith(suffix):
            continue
        if "." not in candidate.name and suffix is None:
            continue
        if suffix is None and not (
            candidate.is_dir(follow_symlinks=False) or stat.S_ISLNK(candidate.stat(follow_symlinks=False).st_mode)
        ):
            continue
        selected.append(Path(candidate.path))
    if selected or not cursor:
        return selected
    # A cursor can point past all current entries.  Wrap once, still bounded.
    for candidate in os.scandir(directory):
        if len(selected) >= budget:
            break
        if suffix is not None and not candidate.name.endswith(suffix):
            continue
        if "." not in candidate.name and suffix is None:
            continue
        if suffix is None and not (
            candidate.is_dir(follow_symlinks=False) or stat.S_ISLNK(candidate.stat(follow_symlinks=False).st_mode)
        ):
            continue
        selected.append(Path(candidate.path))
    return selected


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
        lock_fd = _open_no_follow(cleanup_lock, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return ()
    with os.fdopen(lock_fd, "a+", encoding="utf-8") as cleanup_handle:
        fcntl.flock(cleanup_handle.fileno(), fcntl.LOCK_EX)
        cursor = _read_private_text(cursor_path).strip() if cursor_path.exists() else ""
        candidates = _bounded_cache_candidates(staging_root, cursor=cursor, budget=budget)
        inspected = 0
        last_seen = ""
        for candidate in candidates:
            if inspected >= budget:
                break
            inspected += 1
            last_seen = candidate.name
            artifact_name = candidate.name.split(".", 1)[0]
            lock_path = locks_root / f"{artifact_name}.lock"
            if _is_symlink_node(lock_path):
                # Never unlink/recreate a suspicious lock pathname.  The
                # existing owner may hold the replaced inode.
                continue
            try:
                lock_fd = _open_no_follow(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                with os.fdopen(lock_fd, "a+") as handle:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    try:
                        _remove_tree(candidate)
                        removed.append(candidate.name)
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                continue
        if candidates and last_seen:
            _write_private_text(cursor_path, last_seen + "\\n")
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
        lock_fd = _open_no_follow(cleanup_lock, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError:
        return ()
    with os.fdopen(lock_fd, "a+", encoding="utf-8") as cleanup_handle:
        fcntl.flock(cleanup_handle.fileno(), fcntl.LOCK_EX)
        cursor = _read_private_text(cursor_path).strip() if cursor_path.exists() else ""
        candidates = _bounded_cache_candidates(artifacts_root, cursor=cursor, budget=budget, suffix=".handoff")
        inspected = 0
        last_seen = ""
        for candidate in candidates:
            if inspected >= budget:
                break
            inspected += 1
            last_seen = candidate.name
            if _is_symlink_node(candidate):
                _remove_tree(candidate)
                removed.append(candidate.name)
                continue
            parts = candidate.name.removeprefix(".").split(".", 1)
            if len(parts) != 2:
                continue
            lock_path = locks_root / f"{parts[0]}.lock"
            if _is_symlink_node(lock_path):
                # Never unlink/recreate a suspicious lock pathname.  The
                # existing owner may hold the replaced inode.
                continue
            try:
                lock_fd = _open_no_follow(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                with os.fdopen(lock_fd, "a+") as handle:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    try:
                        _remove_tree(candidate)
                        removed.append(candidate.name)
                    finally:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                continue
        if candidates and last_seen:
            _write_private_text(cursor_path, last_seen + "\\n")
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


def _read_manifest(path: Path) -> SeededArchiveManifest:
    fd = _open_no_follow(path, os.O_RDONLY)
    try:
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as handle:
            payload = json.load(handle)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    if not isinstance(payload, dict):
        raise ValueError("seeded archive manifest must be an object")
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
        manifest = SeededArchiveManifest(facts=facts, **payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("seeded archive manifest has malformed metadata") from exc
    if stored_manifest_id != manifest.manifest_id:
        raise ValueError("seeded archive manifest identity mismatch")
    return manifest


def _manifest_binds_to_key(manifest: SeededArchiveManifest, root: Path, key: SeededArchiveKey) -> bool:
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
        if _is_symlink_node(artifact.root) or not artifact.root.is_dir():
            return False
        paths = (artifact.root, *artifact.root.rglob("*"))
        for path in paths:
            if _is_symlink_node(path) or path.stat().st_mode & write_bits:
                return False
        manifest_path = artifact.root / "manifest.json"
        if _is_symlink_node(manifest_path) or not manifest_path.is_file():
            return False
        disk_manifest = _read_manifest(manifest_path)
        if disk_manifest != artifact.manifest:
            return False
        file_entries = _manifest_file_entries(artifact.manifest.files)
        expected_paths = {path for path, _, _ in file_entries}
        actual_paths: set[str] = set()
        for path in artifact.root.rglob("*"):
            if _is_symlink_node(path):
                return False
            if path.is_file() and not _is_reserved_root_file(path, artifact.root):
                actual_paths.add(str(path.relative_to(artifact.root)))
        if actual_paths != expected_paths:
            return False
        for relative, size, digest in file_entries:
            path = artifact.root / relative
            if _is_symlink_node(path) or not path.is_file():
                return False
            if path.stat().st_size != size or _sha256(path) != digest:
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
    if _is_symlink_node(root) or not root.is_dir():
        return None
    manifest_path = root / "manifest.json"
    if _is_symlink_node(manifest_path) or not manifest_path.is_file():
        return None
    try:
        manifest = _read_manifest(manifest_path)
        if manifest.protocol_version != _ARTIFACT_PROTOCOL_VERSION or manifest.key != key.value:
            return None
        file_entries = _manifest_file_entries(manifest.files)
        expected_paths = {path for path, _, _ in file_entries}
        actual_paths: set[str] = set()
        for path in root.rglob("*"):
            if _is_symlink_node(path):
                return None
            if path.is_file() and not _is_reserved_root_file(path, root):
                actual_paths.add(str(path.relative_to(root)))
        if actual_paths != expected_paths:
            return None
        for relative, size, digest in file_entries:
            path = root / relative
            if _is_symlink_node(path) or not path.is_file() or path.stat().st_size != size or _sha256(path) != digest:
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
        for path in (root, *root.rglob("*")):
            if _is_symlink_node(path) or path.stat().st_mode & write_bits:
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
    for path in root.rglob("*"):
        if _is_symlink_node(path):
            raise ValueError(f"symlink node is not allowed: {path}")


def _make_read_only(root: Path) -> None:
    """Remove write permission without ever following a symlink node."""
    if _is_symlink_node(root):
        raise ValueError(f"cannot seal symlink root: {root}")
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
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
        os.replace(source, destination)
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
            shutil.copytree(staging, handoff, copy_function=shutil.copy2)
            # Seal the handoff before it can be renamed.  A killed copy can
            # leave only a private handoff; final visibility is one rename of
            # an already sealed tree, never a writable directory.
            _assert_no_symlinks(handoff)
            _make_read_only(handoff)
            _rename_sealed(handoff, final_root)
            _remove_tree(staging)
        _make_read_only(final_root)
    except Exception:
        if final_root.exists():
            _remove_tree(final_root)
        if handoff is not None and handoff.exists():
            _remove_tree(handoff)
        if staging.exists():
            _remove_tree(staging)
        raise


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
    artifacts.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_ancestors(artifacts, allow_missing_leaf=False)
    _assert_no_symlink_ancestors(locks, allow_missing_leaf=False)
    _assert_no_symlink_ancestors(staging_root, allow_missing_leaf=False)
    final_root = artifacts / key.value.rsplit(":", 1)[-1]
    lock_path = locks / f"{final_root.name}.lock"

    lock_fd = _open_no_follow(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(lock_fd, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _recover_stale_staging(staging_root=staging_root, artifact_name=final_root.name)
        _recover_obsolete_staging(cache_root=cache_root, staging_root=staging_root)
        _recover_stale_handoffs(cache_root=cache_root, artifacts_root=artifacts)
        cached = _validate_artifact_with_retry(final_root, key)
        if cached is not None:
            _VALIDATED_ARTIFACTS[memo_key] = cached
            return cached
        if final_root.exists():
            _remove_tree(final_root)
        for attempt in range(1, _BUILD_LOCK_ATTEMPTS + 1):
            staging = staging_root / f"{final_root.name}.{uuid.uuid4().hex}"
            staging.mkdir()
            try:
                corpus_root = staging / "wire"
                written_batches = tuple(
                    SyntheticCorpus.write_spec_artifacts(spec, corpus_root / spec.provider, prefix=f"seed-{index:02d}")
                    for index, spec in enumerate(selected_specs)
                )
                sources = []
                for spec, written in zip(selected_specs, written_batches, strict=True):
                    if spec.provider == "antigravity":
                        sources.append(Source(name=spec.provider, path=written.files[0].parent.parent))
                    else:
                        sources.extend(Source(name=spec.provider, path=path) for path in written.files)
                with _configured_archive_root(staging):
                    with patch(
                        "polylogue.sources.parsers.antigravity.AntigravityLanguageServerClient",
                        SyntheticAntigravityLanguageServerClient,
                    ):
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
                build_id = _build_id()
                receipt = _canonical_receipt(
                    key=key,
                    archive_id=archive_id,
                    profile_id=profile_id,
                    build_id=build_id,
                )
                manifest = SeededArchiveManifest(
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
                (staging / "manifest.json").write_text(
                    json.dumps(manifest.to_payload(), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
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


def clone_seeded_archive(artifact: SeededArchiveArtifact, destination: Path) -> SeededArchiveClone:
    """Create a complete private writable archive clone, recording its method."""
    _assert_no_symlinks(artifact.root)
    # The in-memory dataclass is frozen, but its nested dictionaries are not.
    # Authenticate the disk carrier immediately before copying so a caller
    # cannot mutate ``artifact.manifest`` and obtain a false provenance link.
    disk_manifest = _read_manifest(artifact.root / "manifest.json")
    if disk_manifest != artifact.manifest:
        raise ValueError("published artifact manifest changed before clone")
    source_manifest_id = disk_manifest.manifest_id
    if _is_symlink_node(destination):
        destination.unlink()
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
    _assert_no_symlinks(destination)
    for path in destination.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
    bootstrap_marker = destination / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    if bootstrap_marker.is_file():
        from polylogue.storage.sqlite.durable_change_train import _record_fresh_durable_bootstrap

        bootstrap_marker.unlink()
        _record_fresh_durable_bootstrap(destination)
    return SeededArchiveClone(
        root=destination,
        source_manifest_id=source_manifest_id,
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
    "default_cache_root",
    "named_corpus_specs",
    "schema_coverage_corpus_specs",
    "seeded_archive_key",
]
