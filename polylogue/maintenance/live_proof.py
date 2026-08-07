"""Static, read-only live-proof receipt protocol for the reindex campaign.

This module deliberately collects evidence only.  Its registry has no plugin
or runtime-registration seam, and its CLI adapter accepts neither commands nor
callables.  A future campaign consumer can therefore validate a receipt without
turning the proof protocol into an executor, scheduler, or mutation authority.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, cast

from polylogue.core.hashing import hash_file, hash_payload, hash_text
from polylogue.core.json import JSONDocument, is_json_document, require_json_document

LIVE_PROOF_RECEIPT_SCHEMA: Final = "polylogue.live-proof-receipt.v1"
EXISTING_APPLY_RECEIPT_SCHEMA: Final = "polylogue.apply-receipt.v1"
LIVE_PROOF_REGISTRY_VERSION: Final = 1

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_CODE_SHA_RE: Final = re.compile(r"[0-9a-f]{40,64}")
_GENERATION_ID_RE: Final = re.compile(r"gen-[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


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
class LiveProofSpec:
    """One compile-time proof route.

    ``producer`` is intentionally a symbolic key rather than a callable.  The
    collector recognizes only the fixed literal producer keys below, so command
    input can never supply executable behavior.
    """

    proof_id: LiveProofId
    bead_id: str
    mode: LiveProofMode
    producer: Literal["archive_verification", "existing_apply_receipt"]
    archive_checks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LiveProofBindings:
    code_sha: str
    archive_identity_digest: str
    source_snapshot: str
    schema_versions: tuple[tuple[str, int], ...]
    parser_fingerprints: tuple[tuple[str, str], ...]
    lowering_fingerprint: str
    candidate_generation_id: str | None
    candidate_index_sha256: str | None
    private_paths: tuple[tuple[str, PrivatePathReference], ...]

    def to_document(self) -> JSONDocument:
        return {
            "code_sha": self.code_sha,
            "archive_identity_digest": self.archive_identity_digest,
            "source_snapshot": self.source_snapshot,
            "schema_versions": dict(self.schema_versions),
            "parser_fingerprints": dict(self.parser_fingerprints),
            "lowering_fingerprint": self.lowering_fingerprint,
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


LIVE_PROOF_SPECS: Final[tuple[LiveProofSpec, ...]] = (
    LiveProofSpec(
        proof_id=LiveProofId.ARCHIVE_VERIFICATION,
        bead_id="polylogue-x97cf",
        mode=LiveProofMode.READ_ONLY,
        producer="archive_verification",
        archive_checks=("tier-schema", "counts-summary"),
    ),
    LiveProofSpec(
        proof_id=LiveProofId.CANDIDATE_ARCHIVE_VERIFICATION,
        bead_id="polylogue-x97cf",
        mode=LiveProofMode.CANDIDATE,
        producer="archive_verification",
        archive_checks=("corpus-absences",),
    ),
    LiveProofSpec(
        proof_id=LiveProofId.EXISTING_APPLY_RECEIPT,
        bead_id="polylogue-x97cf",
        mode=LiveProofMode.EXISTING_APPLY_RECEIPT,
        producer="existing_apply_receipt",
    ),
)


def validate_live_proof_registry(specs: Sequence[LiveProofSpec] = LIVE_PROOF_SPECS) -> None:
    """Require the complete fixed protocol registry and no executable seam."""

    from polylogue.maintenance.archive_verification import ARCHIVE_VERIFICATION_CHECKS

    expected = set(LiveProofId)
    actual = {spec.proof_id for spec in specs}
    if actual != expected or len(specs) != len(expected):
        raise LiveProofError("live-proof registry must contain every fixed proof id exactly once")
    for spec in specs:
        if not spec.bead_id.startswith("polylogue-"):
            raise LiveProofError("live-proof spec has an invalid bead id")
        if spec.mode is LiveProofMode.EXISTING_APPLY_RECEIPT:
            if spec.producer != "existing_apply_receipt" or spec.archive_checks:
                raise LiveProofError("existing-apply proof spec may only validate an input receipt")
        elif spec.producer != "archive_verification" or not spec.archive_checks:
            raise LiveProofError("read-only and candidate proof specs require registered archive checks")
        for check_name in spec.archive_checks:
            check = next((candidate for candidate in ARCHIVE_VERIFICATION_CHECKS if candidate.name == check_name), None)
            if check is None:
                raise LiveProofError("live-proof spec references an unknown archive verification check")
            if spec.mode is LiveProofMode.CANDIDATE and check.candidate_run is None:
                raise LiveProofError("candidate live-proof spec requires candidate-capable archive checks")


def live_proof_spec(proof_id: str) -> LiveProofSpec:
    try:
        parsed = LiveProofId(proof_id)
    except ValueError as exc:
        raise LiveProofError("unknown live-proof id") from exc
    for spec in LIVE_PROOF_SPECS:
        if spec.proof_id is parsed:
            return spec
    raise LiveProofError("live-proof registry is incomplete")


def _code_sha() -> str:
    configured = os.environ.get("POLYLOGUE_CODE_SHA", "").strip().lower()
    if configured:
        if not _CODE_SHA_RE.fullmatch(configured):
            raise LiveProofError("POLYLOGUE_CODE_SHA must be an exact git commit SHA")
        return configured
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "--verify", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    sha = completed.stdout.strip().lower()
    if completed.returncode != 0 or not _CODE_SHA_RE.fullmatch(sha):
        raise LiveProofError("exact code SHA is unavailable")
    return sha


def _schema_versions(root: Path, *, candidate_index: Path | None) -> tuple[tuple[str, int], ...]:
    paths = {
        "audit": root / "audit.db",
        "source": root / "source.db",
        "index": candidate_index or root / "index.db",
        "embeddings": root / "embeddings.db",
        "ops": root / "ops.db",
        "user": root / "user.db",
    }
    versions: list[tuple[str, int]] = []
    for name, path in sorted(paths.items()):
        try:
            connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            finally:
                connection.close()
        except sqlite3.Error as exc:
            raise LiveProofError("live-proof schema binding is unavailable") from exc
        versions.append((name, version))
    return tuple(versions)


def _candidate_index(root: Path, generation_id: str) -> Path:
    if not _GENERATION_ID_RE.fullmatch(generation_id):
        raise LiveProofError("candidate generation id is invalid")
    generation_root = root / ".index-generations" / generation_id
    metadata_path = generation_root / "generation.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveProofError("candidate generation metadata is unavailable") from exc
    if not isinstance(metadata, dict):
        raise LiveProofError("candidate generation metadata is invalid")
    index_path = generation_root / "index.db"
    if (
        metadata.get("generation_id") != generation_id
        or metadata.get("state") != "inactive"
        or metadata.get("archive_root") != str(root)
        or metadata.get("index_path") != str(index_path)
        or not index_path.is_file()
    ):
        raise LiveProofError("candidate generation binding is stale or invalid")
    return index_path


def capture_live_proof_bindings(archive_root: Path, *, candidate_generation_id: str | None = None) -> LiveProofBindings:
    """Capture every current, read-only binding required by a receipt."""

    from polylogue.maintenance.schema_inference_gate import rebuild_source_revision_snapshot
    from polylogue.sources.origin_specs import lowering_fingerprint, parser_fingerprint_for_origin
    from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation

    root = Path(archive_root).expanduser().resolve()
    candidate_index = _candidate_index(root, candidate_generation_id) if candidate_generation_id is not None else None
    try:
        location = ArchiveLocation.resolve(root)
        identity = ArchiveIdentity.resolve_location(
            location,
            generation_owner=None,
            generation_state="inactive" if candidate_index is not None else "active",
        )
        with sqlite3.connect(f"file:{root / 'source.db'}?mode=ro", uri=True) as source:
            origins = sorted(str(row[0]) for row in source.execute("SELECT DISTINCT origin FROM raw_sessions"))
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        raise LiveProofError("live-proof archive binding is unavailable") from exc
    try:
        parser_fingerprints = tuple((origin, parser_fingerprint_for_origin(origin)) for origin in origins)
        source_snapshot = rebuild_source_revision_snapshot(root)
        lowering = lowering_fingerprint()
    except (OSError, RuntimeError, ValueError) as exc:
        raise LiveProofError("live-proof semantic binding is unavailable") from exc
    private_paths: list[tuple[str, PrivatePathReference]] = [("archive_root", PrivatePathReference.capture(root))]
    if candidate_index is not None:
        private_paths.append(("candidate_index", PrivatePathReference.capture(candidate_index)))
    return LiveProofBindings(
        code_sha=_code_sha(),
        archive_identity_digest=identity.authority_identity_digest,
        source_snapshot=source_snapshot,
        schema_versions=_schema_versions(root, candidate_index=candidate_index),
        parser_fingerprints=parser_fingerprints,
        lowering_fingerprint=lowering,
        candidate_generation_id=candidate_generation_id,
        candidate_index_sha256=hash_file(candidate_index) if candidate_index is not None else None,
        private_paths=tuple(private_paths),
    )


def _archive_verification_result(
    spec: LiveProofSpec, archive_root: Path, *, candidate_generation_id: str | None
) -> tuple[JSONDocument, tuple[LiveProofResidue, ...]]:
    from polylogue.maintenance.archive_verification import verify_archive

    candidate_index = (
        _candidate_index(archive_root, candidate_generation_id) if candidate_generation_id is not None else None
    )
    report = verify_archive(archive_root, checks=spec.archive_checks, index_path_override=candidate_index)
    statuses: JSONDocument = {check.name: check.status.value for check in report.checks}
    residues = tuple(
        LiveProofResidue(LiveProofResidueKind.CHECK_FAILED, check.name)
        for check in report.checks
        if check.status.value in {"error", "warning"}
    )
    archive_verification: JSONDocument = {"checks": statuses, "blocking": report.blocking}
    result: JSONDocument = {
        "status": LiveProofStatus.FAILED.value if report.blocking else LiveProofStatus.PASSED.value,
        "archive_verification": archive_verification,
    }
    return result, residues


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


def _validated_existing_apply_receipt(path: Path, bindings: LiveProofBindings) -> tuple[JSONDocument, str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveProofError("existing apply receipt is unavailable") from exc
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
    if not is_json_document(payload.get("result")):
        raise LiveProofError("existing apply receipt result is malformed")
    _validate_private_path_references(receipt_bindings.get("private_paths"))
    return require_json_document(payload["result"], context="existing apply receipt result"), digest


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
            spec, archive_root, candidate_generation_id=candidate_generation_id
        )
        input_digests: tuple[str, ...] = ()
    else:
        assert apply_receipt_path is not None
        apply_result, digest = _validated_existing_apply_receipt(apply_receipt_path, bindings)
        result = {"status": LiveProofStatus.PASSED.value, "apply_receipt": apply_result}
        input_digests = (digest,)
        residues = ()
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


def validate_live_proof_receipt(
    document: object,
    archive_root: Path,
    *,
    candidate_generation_id: str | None = None,
) -> LiveProofReceipt:
    """Validate a self-hashed proof receipt against present archive bindings."""

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
    if candidate_generation_id is not None and mode is not LiveProofMode.CANDIDATE:
        raise LiveProofError("candidate validation requires a candidate proof receipt")
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
    expected = capture_live_proof_bindings(archive_root, candidate_generation_id=candidate_generation_id)
    if payload.get("bindings") != expected.to_document():
        raise LiveProofError("live-proof receipt bindings are stale or mismatched")
    result = payload.get("result")
    residues = payload.get("residues")
    input_digests = payload.get("input_receipt_digests")
    if not is_json_document(result) or not isinstance(residues, list) or not isinstance(input_digests, list):
        raise LiveProofError("live-proof receipt evidence is malformed")
    try:
        LiveProofStatus(cast(str, result["status"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveProofError("live-proof receipt result status is malformed") from exc
    parsed_residues: list[LiveProofResidue] = []
    for residue in residues:
        if not isinstance(residue, Mapping):
            raise LiveProofError("live-proof receipt residues are malformed")
        try:
            parsed_residues.append(
                LiveProofResidue(LiveProofResidueKind(cast(str, residue["kind"])), cast(str, residue["code"]))
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LiveProofError("live-proof receipt residues are malformed") from exc
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in input_digests):
        raise LiveProofError("live-proof receipt input digests are malformed")
    return LiveProofReceipt(
        proof_id=proof_id,
        bead_id=spec.bead_id,
        mode=mode,
        registry_version=LIVE_PROOF_REGISTRY_VERSION,
        bindings=expected,
        result=result,
        residues=tuple(parsed_residues),
        input_receipt_digests=tuple(cast(list[str], input_digests)),
    )


def _require_acceptable_result(receipt: LiveProofReceipt) -> None:
    """Reject failed proof evidence at every aggregate boundary."""

    try:
        status = LiveProofStatus(cast(str, receipt.result["status"]))
    except (KeyError, TypeError, ValueError) as exc:  # validated above; keeps this boundary total.
        raise LiveProofError("live-proof receipt result status is malformed") from exc
    if status is LiveProofStatus.PASSED:
        return
    if status is LiveProofStatus.NOT_APPLICABLE and any(
        residue.kind is LiveProofResidueKind.NOT_APPLICABLE for residue in receipt.residues
    ):
        return
    raise LiveProofError("live-proof receipt result is not acceptable to an aggregate")


def validate_live_operation_aggregate(receipts: Sequence[object], archive_root: Path) -> tuple[LiveProofReceipt, ...]:
    """Consumer seam for the live-operation aggregate, without scheduling work."""

    validated = tuple(validate_live_proof_receipt(receipt, archive_root) for receipt in receipts)
    if not validated:
        raise LiveProofError("live-operation aggregate requires at least one proof receipt")
    for receipt in validated:
        _require_acceptable_result(receipt)
    return validated


def validate_candidate_proof_receipts(
    receipts: Sequence[object], archive_root: Path, *, candidate_generation_id: str
) -> tuple[LiveProofReceipt, ...]:
    """Consumer seam for candidate acceptance, restricted to one inactive generation."""

    validated = tuple(
        validate_live_proof_receipt(receipt, archive_root, candidate_generation_id=candidate_generation_id)
        for receipt in receipts
    )
    if not validated:
        raise LiveProofError("candidate proof consumer requires at least one proof receipt")
    for receipt in validated:
        _require_acceptable_result(receipt)
    return validated


def validate_final_proof_receipts(receipts: Sequence[object], archive_root: Path) -> tuple[LiveProofReceipt, ...]:
    """Consumer seam for final-proof aggregation, without emitting a terminal proof."""

    return validate_live_operation_aggregate(receipts, archive_root)


def write_live_proof_receipt(path: Path, receipt: LiveProofReceipt) -> None:
    """Write a receipt once, outside the archive, with exclusive creation."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(receipt.to_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise LiveProofError("live-proof receipt output already exists") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


validate_live_proof_registry()

__all__ = [
    "EXISTING_APPLY_RECEIPT_SCHEMA",
    "LIVE_PROOF_RECEIPT_SCHEMA",
    "LIVE_PROOF_REGISTRY_VERSION",
    "LIVE_PROOF_SPECS",
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
