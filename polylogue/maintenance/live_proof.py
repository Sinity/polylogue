"""Static, read-only live-proof receipt protocol for the reindex campaign.

This module collects immutable evidence only. Its fixed registry has no
runtime-registration seam, and the CLI adapter accepts neither commands nor
callables. A consumer can therefore validate a receipt without turning this
proof protocol into mutation authority.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, cast

from polylogue.core.hashing import hash_file, hash_payload, hash_text
from polylogue.core.json import JSONDocument, JSONValue, is_json_document, require_json_document
from polylogue.version import VERSION_INFO

LIVE_PROOF_RECEIPT_SCHEMA: Final = "polylogue.live-proof-receipt.v1"
EXISTING_APPLY_RECEIPT_SCHEMA: Final = "polylogue.apply-receipt.v1"
LIVE_PROOF_REGISTRY_VERSION: Final = 1

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_CODE_SHA_RE: Final = re.compile(r"[0-9a-f]{40,64}")
_GENERATION_ID_RE: Final = re.compile(r"gen-[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RESIDUE_CODE_RE: Final = re.compile(r"[a-z][a-z0-9]*(?:[-_.:][a-z0-9]+)*")
_ABSOLUTE_PATH_RE: Final = re.compile(r"(?<![:\w/])/")
_GIT_TIMEOUT_SECONDS: Final = 2
_SQLITE_SIDECARS: Final = ("-wal", "-journal")
_APPLY_RESULT_STATUSES: Final = frozenset({"applied", "already_satisfied", "not_applicable", "failed", "blocked"})


class LiveProofError(ValueError):
    """A live-proof request, receipt, or current binding is invalid."""


class LiveProofMode(StrEnum):
    READ_ONLY = "read_only"
    CANDIDATE = "candidate"
    EXISTING_APPLY_RECEIPT = "existing_apply_receipt"


class LiveProofResidueKind(StrEnum):
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"
    CHECK_FAILED = "check_failed"
    UNVERIFIED = "unverified"


class LiveProofStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class LiveProofId(StrEnum):
    ARCHIVE_VERIFICATION = "archive-verification"
    CANDIDATE_ARCHIVE_VERIFICATION = "candidate-archive-verification"
    EXISTING_APPLY_RECEIPT = "existing-apply-receipt"


@dataclass(frozen=True, slots=True)
class LiveProofResidue:
    """One closed-vocabulary residual recorded instead of silently omitting it."""

    kind: LiveProofResidueKind
    code: str

    def to_document(self) -> JSONDocument:
        return {"kind": self.kind.value, "code": self.code}


@dataclass(frozen=True, slots=True)
class PrivatePathReference:
    """A private local path represented only by its basename and opaque digest."""

    basename: str
    sha256: str

    @classmethod
    def capture(cls, path: Path) -> PrivatePathReference:
        resolved = path.expanduser().resolve()
        return cls(basename=resolved.name, sha256=hash_text(str(resolved)))

    def to_document(self) -> JSONDocument:
        return {"basename": self.basename, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class LiveProofArchiveProfile:
    """One fixed archive-verification profile for a proof route."""

    name: str
    checks: tuple[str, ...]
    target: Literal["active", "candidate_index", "candidate_cross_tier"]


@dataclass(frozen=True, slots=True)
class LiveProofSpec:
    """One compile-time proof route with no executable registration seam."""

    proof_id: LiveProofId
    bead_id: str
    mode: LiveProofMode
    producer: Literal["archive_verification", "existing_apply_receipt"]
    archive_profiles: tuple[LiveProofArchiveProfile, ...] = ()


@dataclass(frozen=True, slots=True)
class LiveProofBindings:
    code_sha: str
    archive_identity_digest: str
    source_snapshot: str
    schema_versions: tuple[tuple[str, int], ...]
    candidate_index_schema_version: int | None
    parser_fingerprints: tuple[tuple[str, str], ...]
    lowering_fingerprint: str
    active_index_sha256: str
    candidate_generation_id: str | None
    candidate_index_sha256: str | None
    private_paths: tuple[tuple[str, PrivatePathReference], ...]

    def to_document(self) -> JSONDocument:
        return {
            "code_sha": self.code_sha,
            "archive_identity_digest": self.archive_identity_digest,
            "source_snapshot": self.source_snapshot,
            "schema_versions": dict(self.schema_versions),
            "candidate_index_schema_version": self.candidate_index_schema_version,
            "parser_fingerprints": dict(self.parser_fingerprints),
            "lowering_fingerprint": self.lowering_fingerprint,
            "active_index_sha256": self.active_index_sha256,
            "candidate_generation_id": self.candidate_generation_id,
            "candidate_index_sha256": self.candidate_index_sha256,
            "private_paths": {name: ref.to_document() for name, ref in self.private_paths},
        }


@dataclass(frozen=True, slots=True)
class LiveProofReceipt:
    proof_id: LiveProofId
    bead_id: str
    mode: LiveProofMode
    registry_version: int
    bindings: LiveProofBindings
    result: JSONDocument
    residues: tuple[LiveProofResidue, ...]
    input_receipt_digests: tuple[str, ...]

    def payload(self) -> JSONDocument:
        return {
            "receipt_schema": LIVE_PROOF_RECEIPT_SCHEMA,
            "proof_id": self.proof_id.value,
            "bead_id": self.bead_id,
            "mode": self.mode.value,
            "registry_version": self.registry_version,
            "bindings": self.bindings.to_document(),
            "result": self.result,
            "residues": [residue.to_document() for residue in self.residues],
            "input_receipt_digests": list(self.input_receipt_digests),
        }

    @property
    def receipt_sha256(self) -> str:
        return hash_payload(self.payload())

    def to_document(self) -> JSONDocument:
        return {**self.payload(), "receipt_sha256": self.receipt_sha256}


def _archive_profiles() -> tuple[LiveProofArchiveProfile, ...]:
    from polylogue.maintenance.archive_verification import (
        ARCHIVE_VERIFICATION_CHECK_NAMES,
        REINDEX_ACCEPTANCE_CHECKS,
        REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS,
    )

    return (
        LiveProofArchiveProfile("active-archive", ARCHIVE_VERIFICATION_CHECK_NAMES, "active"),
        LiveProofArchiveProfile("candidate-index", REINDEX_ACCEPTANCE_CHECKS, "candidate_index"),
        LiveProofArchiveProfile("candidate-cross-tier", REINDEX_CROSS_TIER_ACCEPTANCE_CHECKS, "candidate_cross_tier"),
    )


_ACTIVE_ARCHIVE_PROFILE, _CANDIDATE_INDEX_PROFILE, _CANDIDATE_CROSS_TIER_PROFILE = _archive_profiles()

LIVE_PROOF_SPECS: Final[tuple[LiveProofSpec, ...]] = (
    LiveProofSpec(
        proof_id=LiveProofId.ARCHIVE_VERIFICATION,
        bead_id="polylogue-x97cf",
        mode=LiveProofMode.READ_ONLY,
        producer="archive_verification",
        archive_profiles=(_ACTIVE_ARCHIVE_PROFILE,),
    ),
    LiveProofSpec(
        proof_id=LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION,
        bead_id="polylogue-x97cf",
        mode=LiveProofMode.CANDIDATE,
        producer="archive_verification",
        archive_profiles=(_CANDIDATE_INDEX_PROFILE, _CANDIDATE_CROSS_TIER_PROFILE),
    ),
    LiveProofSpec(
        proof_id=LiveProofId.EXISTING_APPLY_RECEIPT,
        bead_id="polylogue-x97cf",
        mode=LiveProofMode.EXISTING_APPLY_RECEIPT,
        producer="existing_apply_receipt",
    ),
)


def validate_live_proof_registry(specs: Sequence[LiveProofSpec] = LIVE_PROOF_SPECS) -> None:
    """Require the complete fixed protocol registry and canonical profiles."""

    from polylogue.maintenance.archive_verification import ARCHIVE_VERIFICATION_CHECKS

    expected = set(LiveProofId)
    actual = {spec.proof_id for spec in specs}
    if actual != expected or len(specs) != len(expected):
        raise LiveProofError("live-proof registry must contain every fixed proof id exactly once")
    for spec in specs:
        if not spec.bead_id.startswith("polylogue-"):
            raise LiveProofError("live-proof spec has an invalid bead id")
        if spec.mode is LiveProofMode.EXISTING_APPLY_RECEIPT:
            if spec.producer != "existing_apply_receipt" or spec.archive_profiles:
                raise LiveProofError("existing-apply proof spec may only validate an input receipt")
            continue
        if spec.producer != "archive_verification" or not spec.archive_profiles:
            raise LiveProofError("archive proof specs require registered archive checks")
        for profile in spec.archive_profiles:
            if not profile.checks or len(set(profile.checks)) != len(profile.checks):
                raise LiveProofError("live-proof archive profile is incomplete")
            for check_name in profile.checks:
                if next((check for check in ARCHIVE_VERIFICATION_CHECKS if check.name == check_name), None) is None:
                    raise LiveProofError("live-proof spec references an unknown archive verification check")
    by_id = {spec.proof_id: spec for spec in specs}
    if by_id[LiveProofId.ARCHIVE_VERIFICATION].archive_profiles != (_ACTIVE_ARCHIVE_PROFILE,):
        raise LiveProofError("read-only proof must use the complete active archive profile")
    if by_id[LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION].archive_profiles != (
        _CANDIDATE_INDEX_PROFILE,
        _CANDIDATE_CROSS_TIER_PROFILE,
    ):
        raise LiveProofError("candidate proof must use both canonical candidate acceptance profiles")


def live_proof_spec(proof_id: str) -> LiveProofSpec:
    try:
        parsed = LiveProofId(proof_id)
    except ValueError as exc:
        raise LiveProofError("unknown live-proof id") from exc
    for spec in LIVE_PROOF_SPECS:
        if spec.proof_id is parsed:
            return spec
    raise LiveProofError("live-proof registry is incomplete")


def _run_git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise LiveProofError("exact code SHA is unavailable") from exc


def _installed_code_sha() -> str:
    commit = (VERSION_INFO.commit or "").lower()
    if not _CODE_SHA_RE.fullmatch(commit):
        raise LiveProofError("installed package has no exact build commit")
    return commit


def _code_sha() -> str:
    configured = os.environ.get("POLYLOGUE_CODE_SHA", "").strip().lower()
    if configured:
        if not _CODE_SHA_RE.fullmatch(configured):
            raise LiveProofError("POLYLOGUE_CODE_SHA must be an exact git commit SHA")
        return configured
    repository = Path(__file__).resolve().parents[2]
    if not (repository / ".git").exists():
        return _installed_code_sha()
    dirty = _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise LiveProofError("live proofs require a clean git worktree")
    completed = _run_git(repository, "rev-parse", "--verify", "HEAD")
    sha = completed.stdout.strip().lower()
    if completed.returncode != 0 or not _CODE_SHA_RE.fullmatch(sha):
        raise LiveProofError("exact code SHA is unavailable")
    return sha


def _readonly_uri(path: Path) -> str:
    return f"{path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"


def _open_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(_readonly_uri(path), uri=True, timeout=2)


def _require_quiescent_sqlite(paths: Sequence[Path]) -> None:
    if any(
        path.with_name(path.name + "-wal").exists() or path.with_name(path.name + "-wal").is_symlink() for path in paths
    ):
        raise LiveProofError("live-proof requires a quiescent archive without SQLite WAL files")


def _require_uri_safe_location(location: object) -> None:
    """Fail closed before dependencies that still interpolate SQLite URIs."""

    from polylogue.storage.archive_identity import ArchiveLocation

    assert isinstance(location, ArchiveLocation)
    paths = (location.configured_root, location.active_index_path) + tuple(
        location.configured_tier(name).configured_path for name in ("audit", "source", "embeddings", "ops", "user")
    )
    if any(any(character in str(path) for character in ("%", "?", "#")) for path in paths):
        raise LiveProofError("live-proof archive paths cannot contain SQLite URI query characters")
    _require_quiescent_sqlite(paths[1:])


def _sqlite_file_state(path: Path, *, allow_symlink: bool) -> JSONDocument:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LiveProofError("live-proof SQLite binding is unavailable") from exc
    if path.is_symlink():
        if not allow_symlink:
            raise LiveProofError("live-proof SQLite binding is not a regular file")
        path = path.resolve(strict=True)
        metadata = path.stat()
    if not path.is_file():
        raise LiveProofError("live-proof SQLite binding is not a regular file")
    files: dict[str, JSONValue] = {
        "database": {
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "sha256": hash_file(path),
        }
    }
    for suffix in _SQLITE_SIDECARS:
        sidecar = path.with_name(path.name + suffix)
        if not sidecar.exists() and not sidecar.is_symlink():
            files[suffix] = {"exists": False}
            continue
        try:
            sidecar_metadata = sidecar.lstat()
        except OSError as exc:
            raise LiveProofError("live-proof SQLite sidecar is unavailable") from exc
        if sidecar.is_symlink() or not sidecar.is_file():
            raise LiveProofError("live-proof SQLite sidecar is not a regular file")
        # SQLite may create an empty WAL for a read-only connection. It holds
        # no logical pages and is equivalent to an absent WAL, so normalize it
        # before comparing the pre/post quiescence snapshots.
        if sidecar_metadata.st_size == 0:
            files[suffix] = {"exists": False}
            continue
        files[suffix] = {
            "exists": True,
            "size": sidecar_metadata.st_size,
            "mtime_ns": sidecar_metadata.st_mtime_ns,
            "sha256": hash_file(sidecar),
        }
    return files


def _sqlite_file_set_digest(path: Path, *, allow_symlink: bool = False) -> str:
    """Bind a stable SQLite database plus WAL/journal sidecars as one file set."""

    before = _sqlite_file_state(path, allow_symlink=allow_symlink)
    try:
        with _open_readonly(path) as connection:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("BEGIN")
            connection.execute("PRAGMA schema_version").fetchone()
    except (OSError, sqlite3.Error) as exc:
        raise LiveProofError("live-proof SQLite binding is unavailable") from exc
    after = _sqlite_file_state(path, allow_symlink=allow_symlink)
    if after != before:
        raise LiveProofError("live-proof SQLite file set changed while capturing binding")
    return hash_payload(before)


def _schema_version(path: Path) -> int:
    try:
        with _open_readonly(path) as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
    except (OSError, sqlite3.Error) as exc:
        raise LiveProofError("live-proof schema binding is unavailable") from exc
    return int(row[0]) if row is not None else 0


def _schema_versions(location: object) -> tuple[tuple[str, int], ...]:
    from polylogue.storage.archive_identity import ArchiveLocation

    assert isinstance(location, ArchiveLocation)
    return tuple(
        (name, _schema_version(location.active_tier(name).configured_path))
        for name in ("audit", "source", "index", "embeddings", "ops", "user")
    )


def _candidate_index(location: object, generation_id: str, *, source_snapshot: str) -> Path:
    """Resolve one inactive generation through the lifecycle store's canonical root."""

    from polylogue.storage.archive_identity import ArchiveLocation, TierFileIdentity
    from polylogue.storage.index_generation import IndexGenerationStore

    assert isinstance(location, ArchiveLocation)
    if not _GENERATION_ID_RE.fullmatch(generation_id):
        raise LiveProofError("candidate generation id is invalid")
    # The store bootstraps an absent pointer, which would violate this module's
    # read-only contract. Real generations can only exist after that lifecycle
    # anchor exists, so fail closed instead of asking the store to create it.
    if location.active_pointer is None:
        raise LiveProofError("candidate generation requires an archive lifecycle pointer")
    if ".index-generations" in location.active_pointer.parts:
        raise LiveProofError("candidate generation requires a canonical active-index pointer")
    store = IndexGenerationStore(location)
    try:
        generation = store.load(generation_id)
        index_path = Path(generation.index_path)
        expected_root = store.generations_root / generation_id
        index_resolved = index_path.resolve(strict=True)
        expected_resolved = expected_root.resolve(strict=True) / "index.db"
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LiveProofError("candidate generation metadata is unavailable") from exc
    if (
        generation.generation_id != generation_id
        or generation.state != "inactive"
        or Path(generation.archive_root).resolve() != location.configured_root.resolve()
        or generation.source_snapshot != source_snapshot
        or index_path != expected_root / "index.db"
        or index_path.is_symlink()
        or not index_path.is_file()
        or index_resolved != expected_resolved
        or location.active_index.same_file(TierFileIdentity.resolve("index", index_path))
    ):
        raise LiveProofError("candidate generation binding is stale or invalid")
    _require_quiescent_sqlite((index_path,))
    return index_path


def capture_live_proof_bindings(archive_root: Path, *, candidate_generation_id: str | None = None) -> LiveProofBindings:
    """Capture one coherent, read-only binding snapshot for a proof receipt."""

    from polylogue.maintenance.schema_inference_gate import rebuild_source_revision_snapshot
    from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

    root = Path(archive_root).expanduser().resolve()
    try:
        location = ArchiveLocation.resolve(root)
        _require_uri_safe_location(location)
        source_snapshot = rebuild_source_revision_snapshot(root)
        with _open_readonly(location.configured_tier("source").configured_path) as source:
            origins = sorted(str(row[0]) for row in source.execute("SELECT DISTINCT origin FROM raw_sessions"))
        candidate_index = (
            _candidate_index(location, candidate_generation_id, source_snapshot=source_snapshot)
            if candidate_generation_id is not None
            else None
        )
        identity = ArchiveIdentity.resolve_location(location)
        active_index_sha256 = _sqlite_file_set_digest(location.active_index_path, allow_symlink=True)
        candidate_index_sha256 = _sqlite_file_set_digest(candidate_index) if candidate_index is not None else None
        schema_versions = _schema_versions(location)
        candidate_index_schema_version = _schema_version(candidate_index) if candidate_index is not None else None
        parser_fingerprints = tuple((origin, parser_fingerprint_for_origin(origin)) for origin in origins)
        lowering = lowering_fingerprint()
    except LiveProofError:
        raise
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        raise LiveProofError("live-proof archive binding is unavailable") from exc
    private_paths: list[tuple[str, PrivatePathReference]] = [
        ("archive_root", PrivatePathReference.capture(root)),
        ("active_index", PrivatePathReference.capture(location.active_index_path)),
    ]
    if candidate_index is not None:
        private_paths.append(("candidate_index", PrivatePathReference.capture(candidate_index)))
    return LiveProofBindings(
        code_sha=_code_sha(),
        archive_identity_digest=identity.authority_identity_digest,
        source_snapshot=source_snapshot,
        schema_versions=schema_versions,
        candidate_index_schema_version=candidate_index_schema_version,
        parser_fingerprints=parser_fingerprints,
        lowering_fingerprint=lowering,
        active_index_sha256=active_index_sha256,
        candidate_generation_id=candidate_generation_id,
        candidate_index_sha256=candidate_index_sha256,
        private_paths=tuple(private_paths),
    )


def _redacted_report(report: object, *, roots: Sequence[Path]) -> JSONDocument:
    if not hasattr(report, "to_json"):
        raise LiveProofError("archive verification report is unavailable")
    document = cast(JSONDocument, report.to_json())
    if not isinstance(document, Mapping):
        raise LiveProofError("archive verification report is malformed")
    replacements = {str(path.resolve()): f"[private-path:{hash_text(str(path.resolve()))}]" for path in roots}

    def redact(value: object) -> JSONValue:
        if isinstance(value, Mapping):
            return {str(key): redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [redact(item) for item in value]
        if isinstance(value, str):
            for raw, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
                value = value.replace(raw, replacement)
            if _ABSOLUTE_PATH_RE.search(value):
                return f"[private-text:{hash_text(value)}]"
            return value
        if value is None or isinstance(value, bool | int | float):
            return value
        raise LiveProofError("archive verification emitted non-JSON evidence")

    evidence = {key: value for key, value in document.items() if key not in {"archive_root", "generated_at"}}
    redacted = redact(evidence)
    if not isinstance(redacted, dict):
        raise LiveProofError("archive verification emitted malformed evidence")
    return redacted


def _archive_verification_result(
    spec: LiveProofSpec,
    archive_root: Path,
    *,
    candidate_generation_id: str | None,
    bindings: LiveProofBindings,
) -> tuple[JSONDocument, tuple[LiveProofResidue, ...]]:
    from polylogue.maintenance.archive_verification import verify_archive
    from polylogue.storage.archive_identity import ArchiveLocation

    location = ArchiveLocation.resolve(archive_root)
    archive_paths = (archive_root, location.active_index_path)
    candidate_index: Path | None = None
    candidate_root: Path | None = None
    if candidate_generation_id is not None:
        candidate_index = _candidate_index(location, candidate_generation_id, source_snapshot=bindings.source_snapshot)
        candidate_root = candidate_index.parent
    profiles: dict[str, JSONValue] = {}
    residues: list[LiveProofResidue] = []
    for profile in spec.archive_profiles:
        if profile.target == "active":
            report = verify_archive(archive_root, checks=profile.checks)
            roots: tuple[Path, ...] = archive_paths
        elif profile.target == "candidate_index":
            assert candidate_root is not None
            report = verify_archive(candidate_root, checks=profile.checks)
            roots = (*archive_paths, candidate_root)
        else:
            assert candidate_index is not None
            report = verify_archive(archive_root, checks=profile.checks, index_path_override=candidate_index)
            roots = (*archive_paths, candidate_index.parent)
        profiles[profile.name] = _redacted_report(report, roots=roots)
        for check in report.checks:
            if check.status.value != "ok":
                residues.append(LiveProofResidue(LiveProofResidueKind.CHECK_FAILED, f"{profile.name}:{check.name}"))
    status = LiveProofStatus.PASSED if not residues else LiveProofStatus.FAILED
    return cast(JSONDocument, {"status": status.value, "archive_verification": {"profiles": profiles}}), tuple(residues)


def _validate_private_path_references(value: object) -> None:
    if not isinstance(value, Mapping):
        raise LiveProofError("input receipt private paths are malformed")
    for reference in value.values():
        if not isinstance(reference, Mapping):
            raise LiveProofError("input receipt private paths are malformed")
        basename = reference.get("basename")
        digest = reference.get("sha256")
        if (
            not isinstance(basename, str)
            or Path(basename).name != basename
            or not isinstance(digest, str)
            or not _SHA256_RE.fullmatch(digest)
        ):
            raise LiveProofError("input receipt private paths are malformed")


def _apply_proof_status(status: str) -> LiveProofStatus:
    if status in {"applied", "already_satisfied"}:
        return LiveProofStatus.PASSED
    if status == "not_applicable":
        return LiveProofStatus.NOT_APPLICABLE
    if status == "blocked":
        return LiveProofStatus.BLOCKED
    if status == "failed":
        return LiveProofStatus.FAILED
    raise LiveProofError("existing apply receipt result status is invalid")


def _validate_existing_apply_document(document: object, bindings: LiveProofBindings) -> tuple[JSONDocument, str]:
    if not is_json_document(document):
        raise LiveProofError("existing apply receipt is malformed")
    payload = dict(document)
    digest = payload.pop("receipt_sha256", None)
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest) or hash_payload(payload) != digest:
        raise LiveProofError("existing apply receipt self-hash is invalid")
    if payload.get("receipt_schema") != EXISTING_APPLY_RECEIPT_SCHEMA:
        raise LiveProofError("existing apply receipt schema is not accepted")
    if not isinstance(payload.get("operation_id"), str) or not payload["operation_id"]:
        raise LiveProofError("existing apply receipt operation binding is invalid")
    receipt_bindings = payload.get("bindings")
    if not isinstance(receipt_bindings, Mapping) or receipt_bindings != bindings.to_document():
        raise LiveProofError("existing apply receipt bindings are stale or mismatched")
    result = payload.get("result")
    if not isinstance(result, Mapping) or not is_json_document(result):
        raise LiveProofError("existing apply receipt result is malformed")
    status = result.get("status")
    if not isinstance(status, str) or status not in _APPLY_RESULT_STATUSES:
        raise LiveProofError("existing apply receipt result status is not successful")
    _validate_private_path_references(receipt_bindings.get("private_paths"))
    return require_json_document(document, context="existing apply receipt"), digest


def _validated_existing_apply_receipt(path: Path, bindings: LiveProofBindings) -> tuple[JSONDocument, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveProofError("existing apply receipt is unavailable") from exc
    return _validate_existing_apply_document(document, bindings)


def _apply_residues(status: LiveProofStatus) -> tuple[LiveProofResidue, ...]:
    if status is LiveProofStatus.NOT_APPLICABLE:
        return (LiveProofResidue(LiveProofResidueKind.NOT_APPLICABLE, "apply-result-not-applicable"),)
    if status is LiveProofStatus.BLOCKED:
        return (LiveProofResidue(LiveProofResidueKind.BLOCKED, "apply-result-blocked"),)
    if status is LiveProofStatus.FAILED:
        return (LiveProofResidue(LiveProofResidueKind.CHECK_FAILED, "apply-result-failed"),)
    return ()


def collect_live_proof(
    proof_id: str,
    archive_root: Path,
    *,
    candidate_generation_id: str | None = None,
    apply_receipt_path: Path | None = None,
) -> LiveProofReceipt:
    """Collect exactly one fixed proof route without operating on the archive."""

    validate_live_proof_registry()
    spec = live_proof_spec(proof_id)
    if spec.mode is LiveProofMode.CANDIDATE:
        if candidate_generation_id is None or apply_receipt_path is not None:
            raise LiveProofError("candidate proof requires only an inactive candidate generation id")
    elif spec.mode is LiveProofMode.EXISTING_APPLY_RECEIPT:
        if apply_receipt_path is None or candidate_generation_id is not None:
            raise LiveProofError("existing-apply proof requires only an existing apply receipt")
    elif candidate_generation_id is not None or apply_receipt_path is not None:
        raise LiveProofError("read-only proof accepts no candidate or apply receipt input")

    bindings = capture_live_proof_bindings(archive_root, candidate_generation_id=candidate_generation_id)
    if spec.producer == "archive_verification":
        result, residues = _archive_verification_result(
            spec,
            archive_root,
            candidate_generation_id=candidate_generation_id,
            bindings=bindings,
        )
        input_digests: tuple[str, ...] = ()
    else:
        assert apply_receipt_path is not None
        apply_receipt, digest = _validated_existing_apply_receipt(apply_receipt_path, bindings)
        apply_result = require_json_document(apply_receipt["result"], context="existing apply receipt result")
        proof_status = _apply_proof_status(cast(str, apply_result["status"]))
        result = {"status": proof_status.value, "apply_receipt": apply_receipt}
        input_digests = (digest,)
        residues = _apply_residues(proof_status)
    return LiveProofReceipt(
        proof_id=spec.proof_id,
        bead_id=spec.bead_id,
        mode=spec.mode,
        registry_version=LIVE_PROOF_REGISTRY_VERSION,
        bindings=bindings,
        result=result,
        residues=residues,
        input_receipt_digests=input_digests,
    )


def _parse_residues(value: object) -> tuple[LiveProofResidue, ...]:
    if not isinstance(value, list):
        raise LiveProofError("live-proof receipt residues are malformed")
    parsed: list[LiveProofResidue] = []
    seen: set[tuple[LiveProofResidueKind, str]] = set()
    for residue in value:
        if not isinstance(residue, Mapping):
            raise LiveProofError("live-proof receipt residues are malformed")
        try:
            parsed_residue = LiveProofResidue(
                LiveProofResidueKind(cast(str, residue["kind"])), cast(str, residue["code"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LiveProofError("live-proof receipt residues are malformed") from exc
        residue_key = (parsed_residue.kind, parsed_residue.code)
        if not _RESIDUE_CODE_RE.fullmatch(parsed_residue.code) or residue_key in seen:
            raise LiveProofError("live-proof receipt residues are malformed")
        seen.add(residue_key)
        parsed.append(parsed_residue)
    return tuple(parsed)


def _validate_status_residues(status: LiveProofStatus, residues: tuple[LiveProofResidue, ...]) -> None:
    expected = {
        LiveProofStatus.PASSED: frozenset(),
        LiveProofStatus.FAILED: frozenset({LiveProofResidueKind.CHECK_FAILED, LiveProofResidueKind.UNVERIFIED}),
        LiveProofStatus.BLOCKED: frozenset({LiveProofResidueKind.BLOCKED}),
        LiveProofStatus.NOT_APPLICABLE: frozenset({LiveProofResidueKind.NOT_APPLICABLE}),
    }[status]
    if (not residues and status is not LiveProofStatus.PASSED) or any(
        residue.kind not in expected for residue in residues
    ):
        raise LiveProofError("live-proof receipt status and residues are inconsistent")
    if status is LiveProofStatus.PASSED and residues:
        raise LiveProofError("live-proof receipt status and residues are inconsistent")


def _validate_archive_result(
    spec: LiveProofSpec, result: JSONDocument
) -> tuple[LiveProofStatus, tuple[LiveProofResidue, ...]]:
    archive_verification = result.get("archive_verification")
    if not isinstance(archive_verification, Mapping):
        raise LiveProofError("live-proof receipt archive verification evidence is malformed")
    profiles = archive_verification.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != {profile.name for profile in spec.archive_profiles}:
        raise LiveProofError("live-proof receipt archive verification evidence is incomplete")
    residues: list[LiveProofResidue] = []
    for profile in spec.archive_profiles:
        evidence = profiles.get(profile.name)
        if not isinstance(evidence, Mapping):
            raise LiveProofError("live-proof receipt archive verification evidence is malformed")
        checks = evidence.get("checks")
        if not isinstance(checks, list):
            raise LiveProofError("live-proof receipt archive verification evidence is malformed")
        names: list[str] = []
        statuses: list[str] = []
        for check in checks:
            if not isinstance(check, Mapping):
                raise LiveProofError("live-proof receipt archive verification evidence is malformed")
            name = check.get("name")
            status = check.get("status")
            if not isinstance(name, str) or not isinstance(status, str):
                raise LiveProofError("live-proof receipt archive verification evidence is malformed")
            names.append(name)
            statuses.append(status)
        if tuple(names) != profile.checks:
            raise LiveProofError("live-proof receipt archive verification profile is not canonical")
        residues.extend(
            LiveProofResidue(LiveProofResidueKind.CHECK_FAILED, f"{profile.name}:{name}")
            for name, status in zip(names, statuses, strict=True)
            if status != "ok"
        )
    return (LiveProofStatus.FAILED, tuple(residues)) if residues else (LiveProofStatus.PASSED, ())


def _validate_route_result(
    spec: LiveProofSpec, result: JSONDocument
) -> tuple[LiveProofStatus, tuple[LiveProofResidue, ...]]:
    status = result.get("status")
    try:
        parsed_status = LiveProofStatus(cast(str, status))
    except (TypeError, ValueError) as exc:
        raise LiveProofError("live-proof receipt result status is malformed") from exc
    if spec.producer == "archive_verification":
        expected, residues = _validate_archive_result(spec, result)
        if parsed_status is not expected:
            raise LiveProofError("live-proof receipt archive verification status is inconsistent")
    else:
        apply_receipt = result.get("apply_receipt")
        if not isinstance(apply_receipt, Mapping) or not is_json_document(apply_receipt):
            raise LiveProofError("live-proof receipt apply evidence is malformed")
        apply_result = apply_receipt.get("result")
        if not isinstance(apply_result, Mapping) or not is_json_document(apply_result):
            raise LiveProofError("live-proof receipt apply evidence is malformed")
        raw_status = apply_result.get("status")
        if not isinstance(raw_status, str) or raw_status not in _APPLY_RESULT_STATUSES:
            raise LiveProofError("live-proof receipt apply evidence is malformed")
        expected = _apply_proof_status(raw_status)
        if parsed_status is not expected:
            raise LiveProofError("live-proof receipt apply status is inconsistent")
        residues = _apply_residues(expected)
    return parsed_status, residues


def _capture_expected_bindings(archive_root: Path, candidate_generation_id: str | None) -> LiveProofBindings:
    try:
        return capture_live_proof_bindings(archive_root, candidate_generation_id=candidate_generation_id)
    except LiveProofError as exc:
        raise LiveProofError("live-proof receipt bindings are stale or mismatched") from exc


def _validate_live_proof_receipt(
    document: object,
    archive_root: Path,
    *,
    candidate_generation_id: str | None = None,
    expected_bindings: LiveProofBindings | None = None,
) -> LiveProofReceipt:
    if not is_json_document(document):
        raise LiveProofError("live-proof receipt is malformed")
    payload = dict(document)
    digest = payload.pop("receipt_sha256", None)
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest) or hash_payload(payload) != digest:
        raise LiveProofError("live-proof receipt self-hash is invalid")
    try:
        proof_id = LiveProofId(cast(str, payload["proof_id"]))
        mode = LiveProofMode(cast(str, payload["mode"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveProofError("live-proof receipt route is invalid") from exc
    spec = live_proof_spec(proof_id.value)
    if payload.get("receipt_schema") != LIVE_PROOF_RECEIPT_SCHEMA or payload.get("bead_id") != spec.bead_id:
        raise LiveProofError("live-proof receipt protocol identity is invalid")
    if payload.get("registry_version") != LIVE_PROOF_REGISTRY_VERSION or mode is not spec.mode:
        raise LiveProofError("live-proof receipt registry binding is stale")
    receipt_bindings = payload.get("bindings")
    if not isinstance(receipt_bindings, Mapping):
        raise LiveProofError("live-proof receipt bindings are malformed")
    if mode is LiveProofMode.CANDIDATE:
        recorded_candidate = receipt_bindings.get("candidate_generation_id")
        if not isinstance(recorded_candidate, str):
            raise LiveProofError("candidate proof receipt has no candidate binding")
        if candidate_generation_id is None:
            candidate_generation_id = recorded_candidate
        elif candidate_generation_id != recorded_candidate:
            raise LiveProofError("candidate proof receipt targets a different generation")
    elif candidate_generation_id is not None:
        raise LiveProofError("candidate validation requires a candidate proof receipt")
    expected = expected_bindings or _capture_expected_bindings(archive_root, candidate_generation_id)
    if receipt_bindings != expected.to_document():
        raise LiveProofError("live-proof receipt bindings are stale or mismatched")
    result = payload.get("result")
    input_digests = payload.get("input_receipt_digests")
    if not is_json_document(result) or not isinstance(input_digests, list):
        raise LiveProofError("live-proof receipt evidence is malformed")
    residues = _parse_residues(payload.get("residues"))
    status, expected_residues = _validate_route_result(spec, result)
    if residues != expected_residues:
        raise LiveProofError("live-proof receipt status and residues are inconsistent")
    _validate_status_residues(status, residues)
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in input_digests):
        raise LiveProofError("live-proof receipt input digests are malformed")
    if spec.mode is LiveProofMode.EXISTING_APPLY_RECEIPT:
        apply_receipt = result.get("apply_receipt")
        _apply_result, embedded_digest = _validate_existing_apply_document(apply_receipt, expected)
        if input_digests != [embedded_digest]:
            raise LiveProofError("live-proof receipt apply evidence is not bound to its input digest")
    if spec.mode is not LiveProofMode.EXISTING_APPLY_RECEIPT and input_digests:
        raise LiveProofError("live-proof receipt input digests are malformed")
    return LiveProofReceipt(
        proof_id=proof_id,
        bead_id=spec.bead_id,
        mode=mode,
        registry_version=LIVE_PROOF_REGISTRY_VERSION,
        bindings=expected,
        result=result,
        residues=residues,
        input_receipt_digests=tuple(cast(list[str], input_digests)),
    )


def validate_live_proof_receipt(
    document: object,
    archive_root: Path,
    *,
    candidate_generation_id: str | None = None,
) -> LiveProofReceipt:
    """Validate a self-hashed proof receipt against current archive bindings."""

    return _validate_live_proof_receipt(document, archive_root, candidate_generation_id=candidate_generation_id)


def _require_acceptable_result(receipt: LiveProofReceipt) -> None:
    status, expected_residues = _validate_route_result(live_proof_spec(receipt.proof_id.value), receipt.result)
    if receipt.residues != expected_residues:
        raise LiveProofError("live-proof receipt status and residues are inconsistent")
    _validate_status_residues(status, receipt.residues)
    if status is LiveProofStatus.PASSED:
        return
    if status is LiveProofStatus.NOT_APPLICABLE:
        return
    raise LiveProofError("live-proof receipt result is not acceptable to an aggregate")


def _candidate_id_from_documents(receipts: Sequence[object]) -> str | None:
    candidates: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or receipt.get("mode") != LiveProofMode.CANDIDATE.value:
            continue
        bindings = receipt.get("bindings")
        if not isinstance(bindings, Mapping) or not isinstance(bindings.get("candidate_generation_id"), str):
            raise LiveProofError("candidate proof receipt has no candidate binding")
        candidates.add(cast(str, bindings["candidate_generation_id"]))
    if len(candidates) > 1:
        raise LiveProofError("proof aggregate targets multiple candidate generations")
    return next(iter(candidates), None)


def _active_bindings(bindings: LiveProofBindings) -> LiveProofBindings:
    return replace(
        bindings,
        candidate_generation_id=None,
        candidate_index_sha256=None,
        candidate_index_schema_version=None,
        private_paths=tuple((name, path) for name, path in bindings.private_paths if name != "candidate_index"),
    )


def _validate_aggregate(receipts: Sequence[object], archive_root: Path) -> tuple[LiveProofReceipt, ...]:
    if not receipts:
        raise LiveProofError("live-operation aggregate requires at least one proof receipt")
    candidate_id = _candidate_id_from_documents(receipts)
    candidate_bindings = _capture_expected_bindings(archive_root, candidate_id) if candidate_id is not None else None
    active_bindings = (
        _active_bindings(candidate_bindings)
        if candidate_bindings is not None
        else _capture_expected_bindings(archive_root, None)
    )
    validated: list[LiveProofReceipt] = []
    for receipt in receipts:
        mode = receipt.get("mode") if isinstance(receipt, Mapping) else None
        expected = candidate_bindings if mode == LiveProofMode.CANDIDATE.value else active_bindings
        validated.append(_validate_live_proof_receipt(receipt, archive_root, expected_bindings=expected))
    return tuple(validated)


def validate_live_operation_aggregate(receipts: Sequence[object], archive_root: Path) -> tuple[LiveProofReceipt, ...]:
    """Consumer seam for a validated aggregate, without scheduling work."""

    validated = _validate_aggregate(receipts, archive_root)
    for receipt in validated:
        _require_acceptable_result(receipt)
    return validated


def validate_candidate_proof_receipts(
    receipts: Sequence[object], archive_root: Path, *, candidate_generation_id: str
) -> tuple[LiveProofReceipt, ...]:
    """Accept only the canonical candidate route for one inactive generation."""

    candidate_bindings = _capture_expected_bindings(archive_root, candidate_generation_id)
    validated = tuple(
        _validate_live_proof_receipt(
            receipt,
            archive_root,
            candidate_generation_id=candidate_generation_id,
            expected_bindings=candidate_bindings,
        )
        for receipt in receipts
    )
    if not validated:
        raise LiveProofError("candidate proof consumer requires at least one proof receipt")
    if any(receipt.proof_id is not LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION for receipt in validated):
        raise LiveProofError("candidate proof consumer requires candidate route evidence")
    for receipt in validated:
        _require_acceptable_result(receipt)
    return validated


def validate_final_proof_receipts(receipts: Sequence[object], archive_root: Path) -> tuple[LiveProofReceipt, ...]:
    """Require every fixed proof route before treating a campaign proof as final."""

    validated = validate_live_operation_aggregate(receipts, archive_root)
    proof_ids = [receipt.proof_id for receipt in validated]
    if len(proof_ids) != len(LiveProofId) or set(proof_ids) != set(LiveProofId):
        raise LiveProofError("final proof requires complete route coverage exactly once")
    return validated


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_live_proof_receipt(path: Path, receipt: LiveProofReceipt) -> None:
    """Atomically publish a fully durable receipt without replacing an existing one."""

    target = Path(path).expanduser().resolve()
    encoded = (
        json.dumps(receipt.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, target)
        _fsync_directory(target.parent)
        temporary.unlink()
        temporary = None
        _fsync_directory(target.parent)
    except FileExistsError as exc:
        raise LiveProofError("live-proof receipt output already exists") from exc
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
                _fsync_directory(target.parent)
            except OSError:
                pass
        raise LiveProofError("live-proof receipt output could not be written") from exc


validate_live_proof_registry()

__all__ = [
    "EXISTING_APPLY_RECEIPT_SCHEMA",
    "LIVE_PROOF_RECEIPT_SCHEMA",
    "LIVE_PROOF_REGISTRY_VERSION",
    "LIVE_PROOF_SPECS",
    "LiveProofArchiveProfile",
    "LiveProofBindings",
    "LiveProofError",
    "LiveProofId",
    "LiveProofMode",
    "LiveProofReceipt",
    "LiveProofResidue",
    "LiveProofResidueKind",
    "LiveProofSpec",
    "capture_live_proof_bindings",
    "collect_live_proof",
    "live_proof_spec",
    "validate_candidate_proof_receipts",
    "validate_final_proof_receipts",
    "validate_live_operation_aggregate",
    "validate_live_proof_receipt",
    "validate_live_proof_registry",
    "write_live_proof_receipt",
]
