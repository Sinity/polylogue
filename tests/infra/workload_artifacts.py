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
from polylogue.sources.origin_specs import lowering_fingerprint, origin_specs
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from tests.infra.source_builders import SyntheticAntigravityLanguageServerClient

_ARTIFACT_PROTOCOL_VERSION = 2
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


def _recipe_id() -> str:
    digest = hashlib.sha256()
    for path in _RECIPE_PATHS:
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"recipe:sha256:{digest.hexdigest()}"


def _archive_schema_id() -> str:
    """Bind cached archives to the archive DDL that shaped them.

    ``_recipe_id`` hashes a fixed six-file list that names ``bootstrap.py``
    but none of the ``archive_tiers`` DDL modules, so a schema change arriving
    through ``index.py``/``source.py``/``user.py`` (the normal route) left the
    key untouched and a stale-schema artifact reusable. Hash the rendered DDL
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
    return f"git:{result.stdout.strip()}"


def _source_semantics_id() -> str:
    """Bind cached archives to the parser semantics that produced them."""

    payload = {
        "lowering": lowering_fingerprint(),
        "parsers": {
            spec.origin.value: spec.parser_fingerprint()
            for spec in origin_specs()
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


def _profile_id(key: SeededArchiveKey) -> str:
    payload = json.dumps(key.spec_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return f"workload-profile:sha256:{hashlib.sha256(payload).hexdigest()}"


def seeded_archive_key(specs: Iterable[CorpusSpec]) -> SeededArchiveKey:
    return SeededArchiveKey(
        spec_payload={"corpus_specs": [spec.to_payload() for spec in specs]},
        recipe_id=_recipe_id(),
        source_semantics_id=_source_semantics_id(),
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
            if "locked" not in str(exc).lower() or time.monotonic() >= deadline:
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
    """Remove a locally-owned stale artifact even after immutable publication."""
    if not path.exists():
        return
    for candidate in sorted(path.rglob("*"), reverse=True):
        candidate.chmod(candidate.stat().st_mode | stat.S_IWUSR)
    path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(path)


def _recover_stale_staging(*, staging_root: Path, artifact_name: str) -> tuple[str, ...]:
    """Remove only crash-left staging trees for the currently owned build.

    The per-key flock is held by the caller, so no live builder for this
    artifact can be using these paths while this sweep runs.  A completed
    artifact is published by ``os.replace``; anything left under the matching
    staging prefix is therefore an incomplete build from a process that died
    before publication.  Keeping those trees made a SIGKILL leak large SQLite
    databases indefinitely and allowed a later cache inspection to mistake a
    partial build for reusable state.
    """
    removed: list[str] = []
    for candidate in sorted(staging_root.glob(f"{artifact_name}.*")):
        if not candidate.is_dir():
            continue
        _remove_tree(candidate)
        removed.append(candidate.name)
    return tuple(removed)


def _validate_facts(root: Path, facts: tuple[SyntheticArtifactFacts, ...]) -> None:
    with contextlib.closing(sqlite3.connect(root / "index.db")) as conn:
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


_VALIDATED_ARTIFACTS: dict[tuple[str, str], SeededArchiveArtifact] = {}


def _artifact_still_placed(artifact: SeededArchiveArtifact) -> bool:
    """Cheap presence/size check standing in for a full revalidation.

    ``os.stat`` per manifest entry, versus re-SHA256-ing multi-MB databases,
    running ``PRAGMA quick_check`` on five tiers, and re-querying the planted
    facts and the raw-authority frontier. It catches the failure a live
    process can actually cause -- an artifact deleted or truncated out from
    under it -- while content corruption of an unchanged-size file is left to
    the full validation every NEW process performs on its first hit.
    """
    manifest_path = artifact.root / "manifest.json"
    if not manifest_path.is_file():
        return False
    for item in artifact.manifest.files:
        path = artifact.root / str(item["path"])
        try:
            if path.stat().st_size != item["size"]:
                return False
        except OSError:
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
        _recover_stale_staging(staging_root=staging_root, artifact_name=final_root.name)
        cached = _validate_artifact(final_root, key)
        if cached is not None:
            _VALIDATED_ARTIFACTS[memo_key] = cached
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
            receipt = WorkloadReceipt.from_observations(
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
            _make_read_only(staging)
            os.replace(staging, final_root)
        except Exception:
            _remove_tree(staging)
            raise
        artifact = _validate_artifact(final_root, key)
        if artifact is None:
            raise RuntimeError("published seeded archive failed its own validation")
        _VALIDATED_ARTIFACTS[memo_key] = artifact
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
    bootstrap_marker = destination / ".maintenance-state" / "durable-change-trains" / ".bootstrap"
    if bootstrap_marker.is_file():
        from polylogue.storage.sqlite.durable_change_train import _record_fresh_durable_bootstrap

        bootstrap_marker.unlink()
        _record_fresh_durable_bootstrap(destination)
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
    "default_cache_root",
    "named_corpus_specs",
    "schema_coverage_corpus_specs",
    "seeded_archive_key",
]
