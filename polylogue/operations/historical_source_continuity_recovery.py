"""One-purpose recovery of a pre-#3868 blob-liveness transition.

This bridge exists because an historical receipt predates the normal receipt
postcondition.  It does not relax that normal validator: it reconstructs the
missing authority from the attested pre/post backups and a fresh read-only
liveness census, then records a normal retained source-continuity receipt.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from polylogue.config import Config
from polylogue.maintenance.blob_ref_liveness_reconciliation import census_blob_ref_liveness
from polylogue.maintenance.offline_guard import offline_writer_block_reason
from polylogue.operations._maintenance_receipt_fs import (
    MaintenanceReceiptPathError,
    atomic_replace_receipt,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation, OwnedArchiveLocation
from polylogue.storage.backup_attestation import BackupAttestationError, verify_verification_receipt
from polylogue.storage.blob_ref_liveness import (
    BlobRefLivenessCandidate,
    BlobRefLivenessCandidateDigest,
    classify_blob_ref_liveness,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DURABLE_MIGRATION_ADOPTION_FLOORS,
    DurableChangeTrain,
    DurableChangeTrainError,
    DurableChangeTrainState,
    _released_train_manifests_by_target,
    _require_released_train_chain,
    _validate_source_continuity_refresh_receipt,
    load_durable_change_train_manifest,
    recover_released_source_train_continuity,
    write_durable_change_train_manifest,
)
from polylogue.storage.sqlite.migration_runner import (
    DurableDatabaseEvidence,
    capture_durable_database_evidence,
    capture_durable_schema_inventory,
)

PLAN_FORMAT: Literal["polylogue.historical-source-continuity-recovery-plan.v1"] = (
    "polylogue.historical-source-continuity-recovery-plan.v1"
)
RECEIPT_FORMAT: Literal["polylogue.historical-source-continuity-recovery-receipt.v1"] = (
    "polylogue.historical-source-continuity-recovery-receipt.v1"
)
_HISTORICAL_OPERATION_EVIDENCE = Path(__file__).with_name("historical-source-continuity-operation-20260807.json")


class HistoricalSourceContinuityRecoveryError(RuntimeError):
    """Historical evidence cannot prove this one recovery transition."""


class HistoricalSourceContinuityRecoveryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.historical-source-continuity-recovery-plan.v1"] = PLAN_FORMAT
    old_configured_root: str
    old_resolved_root: str
    new_configured_root: str
    new_resolved_root: str
    mutation_receipt_path: str
    mutation_receipt_sha256: str
    historical_evidence_sha256: str
    legacy_candidate_count: int
    legacy_candidate_digest: str
    pre_backup_manifest_path: str
    pre_backup_manifest_sha256: str
    pre_backup_receipt_path: str
    pre_backup_receipt_sha256: str
    post_backup_manifest_path: str
    post_backup_manifest_sha256: str
    post_backup_receipt_path: str
    post_backup_receipt_sha256: str
    source_train_path: str
    source_train_revision: int
    source_train_sha256: str
    source_before: dict[str, object]
    source_after: dict[str, object]
    census: dict[str, object]
    stopped_daemon_evidence_ref: str
    single_writer_evidence_ref: str
    bound_confirmation: Literal["historical-source-continuity-recovery"]
    plan_sha256: str


class HistoricalSourceContinuityRecoveryReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.historical-source-continuity-recovery-receipt.v1"] = RECEIPT_FORMAT
    state: Literal["prepared", "committed"]
    revision: int
    plan_sha256: str
    authorization: str
    train_before_sha256: str
    train_after_sha256: str | None
    refresh_receipt_sha256: str
    resume_command: str
    receipt_sha256: str


class HistoricalSourceContinuityRecoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ok: Literal[True] = True
    state: Literal["prepared", "committed"]
    plan_sha256: str
    receipt_path: str
    refresh_receipt_path: str


class HistoricalOperationEvidence(BaseModel):
    """Digest-only authority for the one historical liveness operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["polylogue.historical-source-continuity-operation-evidence.v1"]
    operation: Literal["blob-ref-liveness-reconciliation-20260807"]
    mutation_receipt_sha256: str
    candidate_count: int
    candidate_digest: str
    pre_backup_manifest_sha256: str
    pre_backup_receipt_sha256: str
    pre_source_sha256: str
    post_backup_manifest_sha256: str
    post_backup_receipt_sha256: str
    post_source_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _real_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HistoricalSourceContinuityRecoveryError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise HistoricalSourceContinuityRecoveryError(f"{label} is not a real single-linked file: {path}")


def _real_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise HistoricalSourceContinuityRecoveryError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HistoricalSourceContinuityRecoveryError(f"{label} is not a real directory: {path}")
    absolute = Path(os.path.abspath(path))
    resolved = path.resolve(strict=True)
    if absolute != resolved:
        raise HistoricalSourceContinuityRecoveryError(f"{label} traverses a symbolic link: {path}")
    return resolved


def _historical_operation_evidence() -> HistoricalOperationEvidence:
    _real_file(_HISTORICAL_OPERATION_EVIDENCE, label="immutable historical operation evidence")
    try:
        return HistoricalOperationEvidence.model_validate_json(
            _HISTORICAL_OPERATION_EVIDENCE.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise HistoricalSourceContinuityRecoveryError("immutable historical operation evidence is unreadable") from exc


def _verify_historical_operation_evidence(
    *,
    mutation_receipt: Path,
    candidates: int,
    candidate_digest: str,
    pre_manifest: Path,
    pre_receipt: Path,
    pre_source: Path,
    post_manifest: Path,
    post_receipt: Path,
    post_source: Path,
) -> str:
    """Bind recovery to the real 69,340-row operation without retaining private bytes or paths."""
    evidence = _historical_operation_evidence()
    actual = {
        "mutation_receipt_sha256": _sha256(mutation_receipt),
        "candidate_count": candidates,
        "candidate_digest": candidate_digest,
        "pre_backup_manifest_sha256": _sha256(pre_manifest),
        "pre_backup_receipt_sha256": _sha256(pre_receipt),
        "pre_source_sha256": _sha256(pre_source),
        "post_backup_manifest_sha256": _sha256(post_manifest),
        "post_backup_receipt_sha256": _sha256(post_receipt),
        "post_source_sha256": _sha256(post_source),
    }
    if any(getattr(evidence, key) != value for key, value in actual.items()):
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery inputs do not match immutable offline evidence"
        )
    return _sha256(_HISTORICAL_OPERATION_EVIDENCE)


def _sealed_plan(**values: object) -> HistoricalSourceContinuityRecoveryPlan:
    plan = HistoricalSourceContinuityRecoveryPlan.model_validate({**values, "plan_sha256": ""})
    return plan.model_copy(
        update={"plan_sha256": _canonical_json_sha256(plan.model_dump(mode="json", exclude={"plan_sha256"}))}
    )


def _sealed_receipt(**values: object) -> HistoricalSourceContinuityRecoveryReceipt:
    receipt = HistoricalSourceContinuityRecoveryReceipt.model_validate(
        {"format": RECEIPT_FORMAT, **values, "receipt_sha256": ""}
    )
    return receipt.model_copy(
        update={"receipt_sha256": _canonical_json_sha256(receipt.model_dump(mode="json", exclude={"receipt_sha256"}))}
    )


def _verify_plan(plan: HistoricalSourceContinuityRecoveryPlan) -> None:
    if plan.plan_sha256 != _canonical_json_sha256(plan.model_dump(mode="json", exclude={"plan_sha256"})):
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery plan checksum mismatch")


def _verify_receipt(receipt: HistoricalSourceContinuityRecoveryReceipt) -> None:
    if receipt.receipt_sha256 != _canonical_json_sha256(receipt.model_dump(mode="json", exclude={"receipt_sha256"})):
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery receipt checksum mismatch")


def _backup_source_evidence(
    manifest_path: Path, *, old_source_path: Path
) -> tuple[Path, dict[str, object], DurableDatabaseEvidence]:
    """Authenticate one old-path source backup without assuming it is full-evidence."""
    _real_file(manifest_path, label="historical backup manifest")
    backup_root = _real_directory(manifest_path.parent, label="historical backup directory")
    receipt_path = backup_root / "verification-receipt.json"
    _real_file(receipt_path, label="historical backup verification receipt")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalSourceContinuityRecoveryError("historical backup authority is unreadable") from exc
    if not isinstance(manifest, dict) or not isinstance(receipt, dict):
        raise HistoricalSourceContinuityRecoveryError("historical backup authority is not an object")
    if manifest.get("format") != "polylogue-backup-v1" or receipt.get("verdict") != "success":
        raise HistoricalSourceContinuityRecoveryError("historical backup is not a successful polylogue backup")
    if receipt.get("manifest_sha256") != _sha256(manifest_path):
        raise HistoricalSourceContinuityRecoveryError("historical backup receipt does not bind manifest bytes")
    try:
        verify_verification_receipt(receipt, tier="source", live_tier_path=old_source_path)
    except BackupAttestationError as exc:
        raise HistoricalSourceContinuityRecoveryError(
            "historical backup does not authenticate the old source path"
        ) from exc
    fingerprints = manifest.get("tier_source_fingerprints")
    artifacts = receipt.get("tier_artifacts")
    if not isinstance(fingerprints, dict) or not isinstance(artifacts, list):
        raise HistoricalSourceContinuityRecoveryError("historical backup lacks source fingerprint authority")
    fingerprint = fingerprints.get("source.db")
    artifact = next((item for item in artifacts if isinstance(item, dict) and item.get("tier") == "source"), None)
    if not isinstance(fingerprint, dict) or not isinstance(artifact, dict):
        raise HistoricalSourceContinuityRecoveryError("historical backup lacks source artifact authority")
    if fingerprint.get("path") != str(old_source_path) or artifact.get("source_fingerprint") != fingerprint:
        raise HistoricalSourceContinuityRecoveryError("historical backup source path authority changed")
    backup_source = backup_root / "source.db"
    _real_file(backup_source, label="historical backup source.db")
    actual = {"sha256": _sha256(backup_source), "size_bytes": backup_source.stat().st_size}
    if any(fingerprint.get(key) != value or artifact.get(key) != value for key, value in actual.items()):
        raise HistoricalSourceContinuityRecoveryError("historical backup source bytes differ from its receipt")
    try:
        with sqlite3.connect(f"file:{backup_source}?mode=ro&immutable=1", uri=True) as connection:
            evidence = capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    except sqlite3.Error as exc:
        raise HistoricalSourceContinuityRecoveryError("historical backup source is unreadable") from exc
    if (
        fingerprint.get("user_version") != evidence.user_version
        or artifact.get("user_version") != evidence.user_version
    ):
        raise HistoricalSourceContinuityRecoveryError("historical backup source version differs from its receipt")
    return receipt_path, manifest, evidence


def _legacy_liveness_receipt(receipt_path: Path, *, old_source_path: Path, pre_manifest: Path) -> tuple[int, str]:
    """Validate exactly the historical receipt shape, including every candidate row."""
    _real_file(receipt_path, label="historical liveness receipt")
    header: dict[str, object] | None = None
    footer: dict[str, object] | None = None
    digest = BlobRefLivenessCandidateDigest()
    count = 0
    try:
        for line in io.BytesIO(receipt_path.read_bytes()):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise HistoricalSourceContinuityRecoveryError("historical liveness receipt contains a non-object")
            if header is None:
                header = cast(dict[str, object], record)
                continue
            if footer is not None:
                raise HistoricalSourceContinuityRecoveryError("historical liveness receipt has data after its footer")
            if record.get("kind") == "candidate":
                try:
                    size = record["size_bytes"]
                    acquired = record["acquired_at_ms"]
                    if (
                        not isinstance(size, int)
                        or isinstance(size, bool)
                        or not isinstance(acquired, int)
                        or isinstance(acquired, bool)
                    ):
                        raise TypeError
                    digest.update(
                        BlobRefLivenessCandidate(
                            blob_hash=str(record["blob_hash"]),
                            ref_type=str(record["ref_type"]),
                            ref_id=str(record["ref_id"]),
                            source_path=str(record["source_path"]) if record.get("source_path") is not None else None,
                            size_bytes=size,
                            acquired_at_ms=acquired,
                            referent_table=str(record["referent_table"]),
                            referent_column=str(record["referent_column"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise HistoricalSourceContinuityRecoveryError(
                        "historical liveness receipt has an invalid candidate"
                    ) from exc
                count += 1
            else:
                footer = cast(dict[str, object], record)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalSourceContinuityRecoveryError("historical liveness receipt is not valid JSONL") from exc
    if header is None or footer is None:
        raise HistoricalSourceContinuityRecoveryError("historical liveness receipt is incomplete")
    if (
        header.get("kind") != "blob_ref_liveness_reconciliation"
        or header.get("phase") != "prepared"
        or header.get("source_db") != str(old_source_path)
        or header.get("backup_manifest") != str(pre_manifest)
        or "backup_manifest_sha256" in header
        or footer.get("kind") != "blob_ref_liveness_reconciliation"
        or footer.get("phase") != "committed"
        or "post_orphaned_count" in footer
        or header.get("candidate_count") != count
        or header.get("candidate_digest") != digest.hexdigest()
        or footer.get("deleted_count") != count
    ):
        raise HistoricalSourceContinuityRecoveryError("historical liveness receipt does not bind the legacy operation")
    return count, digest.hexdigest()


def _evidence_payload(evidence: DurableDatabaseEvidence) -> dict[str, object]:
    return {
        "tier": evidence.tier.value,
        "user_version": evidence.user_version,
        "quick_check": list(evidence.quick_check),
        "schema_inventory_sha256": evidence.schema_inventory_sha256,
        "row_counts": [[table, count] for table, count in evidence.row_counts],
        "archive_identity_digest": evidence.archive_identity_digest,
        "content_sha256": evidence.content_sha256,
        "observed_at_ms": evidence.observed_at_ms,
    }


def _assert_pre_train_authority(
    train_path: Path, pre: DurableDatabaseEvidence
) -> tuple[DurableChangeTrain, dict[str, object]]:
    train = load_durable_change_train_manifest(train_path)
    if (
        train.state is not DurableChangeTrainState.RELEASED
        or train.tier is not ArchiveTier.SOURCE
        or train.apply_evidence is None
    ):
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery requires a released source train")
    expected = train.apply_evidence.post
    fields = ("user_version", "schema_inventory_sha256", "content_sha256", "quick_check")
    if any(getattr(expected, field) != getattr(pre, field) for field in fields):
        raise HistoricalSourceContinuityRecoveryError("pre-mutation backup does not match the released source train")
    return train, _evidence_payload(expected)


def _current_evidence(root: Path) -> DurableDatabaseEvidence:
    _real_file(root / "source.db", label="live source.db")
    for suffix in ("-wal", "-shm", "-journal"):
        if (root / f"source.db{suffix}").exists() or (root / f"source.db{suffix}").is_symlink():
            raise HistoricalSourceContinuityRecoveryError(
                "historical continuity recovery refuses source SQLite sidecars"
            )
    try:
        with sqlite3.connect(f"file:{root / 'source.db'}?mode=ro&immutable=1", uri=True) as connection:
            return capture_durable_database_evidence(connection, ArchiveTier.SOURCE)
    except sqlite3.Error as exc:
        raise HistoricalSourceContinuityRecoveryError("cannot read current source evidence") from exc


def _blob_ref_rows_digest(connection: sqlite3.Connection, *, excluded: set[tuple[str, str, str]]) -> tuple[int, str]:
    """Digest the exact non-candidate blob-ref relation without writing SQLite."""
    digest = hashlib.sha256()
    count = 0
    try:
        rows = connection.execute(
            "SELECT hex(blob_hash), ref_type, ref_id, source_path, size_bytes, acquired_at_ms "
            "FROM blob_refs ORDER BY ref_type, ref_id, blob_hash"
        )
        for blob_hash, ref_type, ref_id, source_path, size_bytes, acquired_at_ms in rows:
            key = (str(blob_hash).lower(), str(ref_type), str(ref_id))
            if key in excluded:
                continue
            digest.update(
                json.dumps(
                    [
                        str(blob_hash).lower(),
                        str(ref_type),
                        str(ref_id),
                        source_path,
                        int(size_bytes),
                        int(acquired_at_ms),
                    ],
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode()
                + b"\n"
            )
            count += 1
    except sqlite3.Error as exc:
        raise HistoricalSourceContinuityRecoveryError("historical backup cannot read blob-ref relation") from exc
    return count, digest.hexdigest()


def _table_content_digest(connection: sqlite3.Connection, table: str) -> tuple[int, str]:
    """Stream a deterministic typed row digest without materializing a table."""
    quoted = '"' + table.replace('"', '""') + '"'
    columns = [str(row[1]) for row in connection.execute(f"PRAGMA table_xinfo({quoted})") if int(row[6]) == 0]
    if not columns:
        raise HistoricalSourceContinuityRecoveryError(f"source table has no readable columns: {table}")
    quoted_columns = ['"' + column.replace('"', '""') + '"' for column in columns]
    digest = hashlib.sha256()
    count = 0
    try:
        for row in connection.execute(
            f"SELECT {', '.join(quoted_columns)} FROM {quoted} ORDER BY {', '.join(quoted_columns)}"
        ):
            encoded = [_typed_sqlite_value(value) for value in row]
            digest.update(json.dumps(encoded, separators=(",", ":"), ensure_ascii=True).encode() + b"\n")
            count += 1
    except (sqlite3.Error, TypeError) as exc:
        raise HistoricalSourceContinuityRecoveryError(f"cannot deterministically read source table: {table}") from exc
    return count, digest.hexdigest()


def _typed_sqlite_value(value: object) -> list[str]:
    """Encode SQLite's storage class as well as its visible value."""
    if value is None:
        return ["null", ""]
    if isinstance(value, bool):
        return ["integer", "1" if value else "0"]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, float):
        return ["real", value.hex()]
    if isinstance(value, str):
        return ["text", value]
    if isinstance(value, bytes):
        return ["blob", value.hex()]
    raise HistoricalSourceContinuityRecoveryError(
        f"source table returned unsupported SQLite value type: {type(value)!r}"
    )


def _assert_complete_source_semantic_delta(pre_source: Path, post_source: Path) -> None:
    """Require every non-blob-ref schema object and relation to be identical."""
    try:
        with (
            sqlite3.connect(f"file:{pre_source}?mode=ro&immutable=1", uri=True) as pre,
            sqlite3.connect(f"file:{post_source}?mode=ro&immutable=1", uri=True) as post,
        ):
            if capture_durable_schema_inventory(pre) != capture_durable_schema_inventory(post):
                raise HistoricalSourceContinuityRecoveryError("pre/post backups have source schema or object drift")
            tables = [
                str(row[0])
                for row in pre.execute(
                    "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
                if str(row[0]) != "blob_refs"
            ]
            for table in tables:
                if _table_content_digest(pre, table) != _table_content_digest(post, table):
                    raise HistoricalSourceContinuityRecoveryError(
                        f"post-mutation backup changed non-blob-ref source table: {table}"
                    )
    except sqlite3.Error as exc:
        raise HistoricalSourceContinuityRecoveryError("cannot compare complete pre/post source authority") from exc


def _assert_exact_liveness_delta(
    pre_source: Path, post_source: Path, candidates: tuple[BlobRefLivenessCandidate, ...]
) -> None:
    """Prove the post backup differs only by deleting the historical candidates."""
    candidate_keys = {(candidate.blob_hash.lower(), candidate.ref_type, candidate.ref_id) for candidate in candidates}
    try:
        with (
            sqlite3.connect(f"file:{pre_source}?mode=ro&immutable=1", uri=True) as pre,
            sqlite3.connect(f"file:{post_source}?mode=ro&immutable=1", uri=True) as post,
        ):
            pre_has_blob_refs = (
                pre.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'blob_refs'").fetchone()
                is not None
            )
            post_has_blob_refs = (
                post.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'blob_refs'").fetchone()
                is not None
            )
            if not pre_has_blob_refs and not post_has_blob_refs:
                if candidate_keys:
                    raise HistoricalSourceContinuityRecoveryError(
                        "historical receipt names candidates but its backup has no blob-ref relation"
                    )
                return
            if not pre_has_blob_refs or not post_has_blob_refs:
                raise HistoricalSourceContinuityRecoveryError("pre/post backups disagree on the blob-ref relation")
            pre_count, pre_digest = _blob_ref_rows_digest(pre, excluded=candidate_keys)
            post_count, post_digest = _blob_ref_rows_digest(post, excluded=set())
    except sqlite3.Error as exc:
        raise HistoricalSourceContinuityRecoveryError("cannot compare pre/post blob-ref authority") from exc
    if (pre_count, pre_digest) != (post_count, post_digest):
        raise HistoricalSourceContinuityRecoveryError(
            "post-mutation backup changed blob refs beyond the historical candidates"
        )


def _census(root: Path) -> dict[str, object]:
    census = census_blob_ref_liveness(root)
    payload = census.to_privacy_safe_dict()
    if (
        census.total
        or census.schema_unavailable_count
        or payload.get("unknown_ref_type_count")
        or census.deferred_by_ref_type
    ):
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery requires a zero-orphan complete liveness census"
        )
    return payload


def _evidence_matches_plan(actual: DurableDatabaseEvidence, planned: dict[str, object]) -> bool:
    """Evidence timestamps are observations, not a mutable archive identity fact."""
    actual_payload = _evidence_payload(actual)
    return all(actual_payload.get(key) == value for key, value in planned.items() if key != "observed_at_ms")


def _evidence_from_plan(payload: dict[str, object]) -> DurableDatabaseEvidence:
    """Decode the sealed evidence payload without accepting an untyped shape."""
    try:
        row_counts = payload["row_counts"]
        if not isinstance(row_counts, list):
            raise TypeError
        normalized_counts = tuple(
            (str(item[0]), int(item[1])) for item in row_counts if isinstance(item, list) and len(item) == 2
        )
        if len(normalized_counts) != len(row_counts):
            raise TypeError
        user_version = payload["user_version"]
        observed_at_ms = payload["observed_at_ms"]
        if not isinstance(user_version, int) or isinstance(user_version, bool):
            raise TypeError
        if not isinstance(observed_at_ms, int) or isinstance(observed_at_ms, bool):
            raise TypeError
        return DurableDatabaseEvidence(
            tier=ArchiveTier(str(payload["tier"])),
            user_version=user_version,
            quick_check=tuple(str(item) for item in cast(list[object], payload["quick_check"])),
            schema_inventory_sha256=str(payload["schema_inventory_sha256"]),
            row_counts=normalized_counts,
            archive_identity_digest=str(payload["archive_identity_digest"]),
            content_sha256=str(payload["content_sha256"]),
            observed_at_ms=observed_at_ms,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery plan has invalid source evidence"
        ) from exc


def _receipt_path(root: Path, plan: HistoricalSourceContinuityRecoveryPlan) -> Path:
    return root / ".maintenance-state" / "historical-source-continuity-recoveries" / f"{plan.plan_sha256}.json"


def _refresh_path(root: Path, digest: str) -> Path:
    return root / ".maintenance-state" / "source-continuity-refreshes" / f"{digest}.json"


def _require_offline_ownership_boundary(root: Path) -> None:
    """Make offline authority real for callers that bypass the Click adapter."""
    reason = offline_writer_block_reason(Config(archive_root=root, render_root=render_root(), sources=[]))
    if reason is not None:
        raise HistoricalSourceContinuityRecoveryError(
            f"historical continuity recovery requires the daemon to be stopped; {reason}"
        )


def _write_refresh_receipt(path: Path, payload: dict[str, object]) -> None:
    """Publish a retained receipt beneath a pinned, non-symlink directory."""
    state_root = path.parent.parent
    if state_root.name != ".maintenance-state" or path.suffix != ".json":
        raise HistoricalSourceContinuityRecoveryError("invalid source continuity refresh receipt path")
    try:
        with maintenance_receipt_directory(state_root.parent, path.parent.name) as directory_fd:
            current = read_optional_receipt(directory_fd, path.name)
            if current is not None:
                try:
                    current_payload = json.loads(current)
                except json.JSONDecodeError as exc:
                    raise HistoricalSourceContinuityRecoveryError(
                        "historical continuity recovery refresh receipt is unreadable"
                    ) from exc
                if current_payload != payload:
                    raise HistoricalSourceContinuityRecoveryError(
                        "historical continuity recovery refresh receipt collision"
                    )
                return
            encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            atomic_replace_receipt(directory_fd, path.name, encoded)
    except MaintenanceReceiptPathError as exc:
        raise HistoricalSourceContinuityRecoveryError(
            f"unsafe historical continuity refresh receipt path: {path}"
        ) from exc


def prepare_historical_source_continuity_recovery(
    *,
    old_root: Path,
    new_root: Path,
    mutation_receipt: Path,
    pre_backup_manifest: Path,
    post_backup_manifest: Path,
    stopped_daemon_evidence_ref: str,
    single_writer_evidence_ref: str,
) -> HistoricalSourceContinuityRecoveryPlan:
    """Seal a read-only recovery plan for the one historical liveness receipt."""
    old_configured = old_root.absolute()
    old_resolved = old_root.resolve(strict=False)
    root = _real_directory(new_root, label="configured archive root")
    if old_resolved == root:
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery requires distinct old and new roots"
        )
    old_source = old_resolved / "source.db"
    pre_receipt, _pre_manifest, pre = _backup_source_evidence(pre_backup_manifest, old_source_path=old_source)
    post_receipt, _post_manifest, post = _backup_source_evidence(post_backup_manifest, old_source_path=old_source)
    candidates, candidate_digest = _legacy_liveness_receipt(
        mutation_receipt, old_source_path=old_source, pre_manifest=pre_backup_manifest.absolute()
    )
    historical_evidence_sha256 = _verify_historical_operation_evidence(
        mutation_receipt=mutation_receipt,
        candidates=candidates,
        candidate_digest=candidate_digest,
        pre_manifest=pre_backup_manifest,
        pre_receipt=pre_receipt,
        pre_source=pre_backup_manifest.parent / "source.db",
        post_manifest=post_backup_manifest,
        post_receipt=post_receipt,
        post_source=post_backup_manifest.parent / "source.db",
    )
    try:
        with sqlite3.connect(
            f"file:{pre_backup_manifest.parent / 'source.db'}?mode=ro&immutable=1", uri=True
        ) as connection:
            prior = classify_blob_ref_liveness(connection)
    except sqlite3.Error as exc:
        raise HistoricalSourceContinuityRecoveryError("cannot recompute historical liveness candidates") from exc
    prior_digest = BlobRefLivenessCandidateDigest()
    for candidate in prior.candidates:
        prior_digest.update(candidate)
    if prior.orphaned_count != candidates or prior_digest.hexdigest() != candidate_digest:
        raise HistoricalSourceContinuityRecoveryError("historical backup liveness candidates differ from the receipt")
    _assert_exact_liveness_delta(
        pre_backup_manifest.parent / "source.db",
        post_backup_manifest.parent / "source.db",
        prior.candidates,
    )
    _assert_complete_source_semantic_delta(
        pre_backup_manifest.parent / "source.db", post_backup_manifest.parent / "source.db"
    )
    current = _current_evidence(root)
    current_path = root / "source.db"
    if (
        current.content_sha256 != post.content_sha256
        or current.user_version != post.user_version
        or _sha256(current_path) != _sha256(post_backup_manifest.parent / "source.db")
    ):
        raise HistoricalSourceContinuityRecoveryError(
            "current source bytes do not match the authenticated post-mutation backup"
        )
    manifest_root = root / ".maintenance-state" / "durable-change-trains"
    manifests = _released_train_manifests_by_target(manifest_root, ArchiveTier.SOURCE)
    try:
        _require_released_train_chain(ArchiveTier.SOURCE, manifests, current_version=current.user_version)
    except DurableChangeTrainError as exc:
        raise HistoricalSourceContinuityRecoveryError("released source train chain is not authoritative") from exc
    expected_targets = set(range(DURABLE_MIGRATION_ADOPTION_FLOORS[ArchiveTier.SOURCE] + 1, current.user_version + 1))
    if set(manifests) != expected_targets:
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery found an unexpected source train set"
        )
    train = manifests.get(current.user_version)
    if train is None:
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery lacks the current released source train"
        )
    train_path = manifest_root / f"source-{train.slot:03d}.json"
    _real_file(train_path, label="current released source train")
    _, source_before = _assert_pre_train_authority(train_path, pre)
    if train.source_continuity_evidence is not None:
        raise HistoricalSourceContinuityRecoveryError("current released source train already has continuity authority")
    census = _census(root)
    return _sealed_plan(
        old_configured_root=str(old_configured),
        old_resolved_root=str(old_resolved),
        new_configured_root=str(new_root.absolute()),
        new_resolved_root=str(root),
        mutation_receipt_path=str(mutation_receipt.absolute()),
        mutation_receipt_sha256=_sha256(mutation_receipt),
        historical_evidence_sha256=historical_evidence_sha256,
        legacy_candidate_count=candidates,
        legacy_candidate_digest=candidate_digest,
        pre_backup_manifest_path=str(pre_backup_manifest.absolute()),
        pre_backup_manifest_sha256=_sha256(pre_backup_manifest),
        pre_backup_receipt_path=str(pre_receipt),
        pre_backup_receipt_sha256=_sha256(pre_receipt),
        post_backup_manifest_path=str(post_backup_manifest.absolute()),
        post_backup_manifest_sha256=_sha256(post_backup_manifest),
        post_backup_receipt_path=str(post_receipt),
        post_backup_receipt_sha256=_sha256(post_receipt),
        source_train_path=str(train_path),
        source_train_revision=train.revision,
        source_train_sha256=_sha256(train_path),
        source_before=source_before,
        source_after=_evidence_payload(current),
        census=census,
        stopped_daemon_evidence_ref=stopped_daemon_evidence_ref,
        single_writer_evidence_ref=single_writer_evidence_ref,
        bound_confirmation="historical-source-continuity-recovery",
    )


def write_historical_source_continuity_recovery_plan(
    plan: HistoricalSourceContinuityRecoveryPlan, output: Path
) -> None:
    _verify_plan(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_historical_source_continuity_recovery_plan(path: Path) -> HistoricalSourceContinuityRecoveryPlan:
    try:
        plan = HistoricalSourceContinuityRecoveryPlan.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HistoricalSourceContinuityRecoveryError("invalid historical continuity recovery plan") from exc
    _verify_plan(plan)
    return plan


def _recovery_receipt_directory_binding(path: Path) -> tuple[Path, str]:
    state_root = path.parent.parent
    if state_root.name != ".maintenance-state" or path.suffix != ".json":
        raise HistoricalSourceContinuityRecoveryError("invalid historical continuity recovery receipt path")
    return state_root.parent, path.parent.name


def _decode_recovery_receipt(encoded: bytes) -> HistoricalSourceContinuityRecoveryReceipt:
    try:
        receipt = HistoricalSourceContinuityRecoveryReceipt.model_validate_json(encoded)
    except ValueError as exc:
        raise HistoricalSourceContinuityRecoveryError("invalid historical continuity recovery receipt") from exc
    _verify_receipt(receipt)
    return receipt


def _load_recovery_receipt_for_update(path: Path) -> HistoricalSourceContinuityRecoveryReceipt | None:
    root, directory_name = _recovery_receipt_directory_binding(path)
    try:
        with maintenance_receipt_directory(root, directory_name) as directory_fd:
            encoded = read_optional_receipt(directory_fd, path.name)
    except MaintenanceReceiptPathError as exc:
        raise HistoricalSourceContinuityRecoveryError(
            f"unsafe historical continuity recovery receipt path: {path}"
        ) from exc
    return None if encoded is None else _decode_recovery_receipt(encoded)


def _write_receipt(path: Path, receipt: HistoricalSourceContinuityRecoveryReceipt, *, expected: str | None) -> None:
    _verify_receipt(receipt)
    root, directory_name = _recovery_receipt_directory_binding(path)
    try:
        with maintenance_receipt_directory(root, directory_name) as directory_fd:
            current_bytes = read_optional_receipt(directory_fd, path.name)
            if current_bytes is not None:
                if _decode_recovery_receipt(current_bytes).receipt_sha256 != expected:
                    raise HistoricalSourceContinuityRecoveryError(
                        "historical continuity recovery receipt CAS state changed"
                    )
            elif expected is not None:
                raise HistoricalSourceContinuityRecoveryError("historical continuity recovery receipt disappeared")
            encoded = (json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode()
            atomic_replace_receipt(directory_fd, path.name, encoded)
    except MaintenanceReceiptPathError as exc:
        raise HistoricalSourceContinuityRecoveryError(
            f"unsafe historical continuity recovery receipt path: {path}"
        ) from exc


def load_historical_source_continuity_recovery_receipt(path: Path) -> HistoricalSourceContinuityRecoveryReceipt:
    _real_file(path, label="historical continuity recovery receipt")
    try:
        encoded = path.read_bytes()
    except (OSError, ValueError) as exc:
        raise HistoricalSourceContinuityRecoveryError("invalid historical continuity recovery receipt") from exc
    return _decode_recovery_receipt(encoded)


def assert_no_prepared_historical_source_continuity_recovery(root: Path) -> None:
    receipt_root = root / ".maintenance-state" / "historical-source-continuity-recoveries"
    if not receipt_root.exists():
        return
    _real_directory(receipt_root, label="historical continuity recovery receipt directory")
    for path in sorted(receipt_root.glob("*.json")):
        receipt = load_historical_source_continuity_recovery_receipt(path)
        if receipt.state == "prepared":
            raise HistoricalSourceContinuityRecoveryError(
                "historical source continuity recovery is prepared but incomplete; rerun " + receipt.resume_command
            )


def _revalidate(
    root: Path, plan: HistoricalSourceContinuityRecoveryPlan, *, stopped: str, writer: str
) -> DurableDatabaseEvidence:
    if stopped != plan.stopped_daemon_evidence_ref or writer != plan.single_writer_evidence_ref:
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery writer evidence changed")
    if str(root) != plan.new_resolved_root:
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery configured root changed")
    old_source = Path(plan.old_resolved_root) / "source.db"
    pre_receipt, _m, pre = _backup_source_evidence(Path(plan.pre_backup_manifest_path), old_source_path=old_source)
    post_receipt, _m2, post = _backup_source_evidence(Path(plan.post_backup_manifest_path), old_source_path=old_source)
    bindings = (
        (Path(plan.mutation_receipt_path), plan.mutation_receipt_sha256),
        (Path(plan.pre_backup_manifest_path), plan.pre_backup_manifest_sha256),
        (pre_receipt, plan.pre_backup_receipt_sha256),
        (Path(plan.post_backup_manifest_path), plan.post_backup_manifest_sha256),
        (post_receipt, plan.post_backup_receipt_sha256),
    )
    if any(_sha256(path) != digest for path, digest in bindings):
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery authority bytes changed")
    count, digest = _legacy_liveness_receipt(
        Path(plan.mutation_receipt_path), old_source_path=old_source, pre_manifest=Path(plan.pre_backup_manifest_path)
    )
    if count != plan.legacy_candidate_count or digest != plan.legacy_candidate_digest:
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery legacy receipt changed")
    current = _current_evidence(root)
    if not _evidence_matches_plan(current, plan.source_after) or current.content_sha256 != post.content_sha256:
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery current source changed")
    train = load_durable_change_train_manifest(Path(plan.source_train_path))
    if _sha256(Path(plan.source_train_path)) != plan.source_train_sha256 and train.source_continuity_evidence is None:
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery source train changed")
    if train.source_continuity_evidence is None:
        _assert_pre_train_authority(Path(plan.source_train_path), pre)
    if _census(root) != plan.census:
        raise HistoricalSourceContinuityRecoveryError("historical continuity recovery liveness census changed")
    return current


def apply_historical_source_continuity_recovery(
    *,
    root: Path,
    plan: HistoricalSourceContinuityRecoveryPlan,
    authorization: str,
    stopped_daemon_evidence_ref: str,
    single_writer_evidence_ref: str,
) -> HistoricalSourceContinuityRecoveryResult:
    """Acquire archive ownership before the API can publish receipts or a CAS revision."""
    resolved = _real_directory(root, label="configured archive root")
    _require_offline_ownership_boundary(resolved)
    with OwnedArchiveLocation.acquire(
        ArchiveLocation.resolve(resolved),
        owner_id=f"historical-source-continuity-recovery:{os.getpid()}",
        allow_reentrant=True,
    ):
        return _apply_historical_source_continuity_recovery_locked(
            root=resolved,
            plan=plan,
            authorization=authorization,
            stopped_daemon_evidence_ref=stopped_daemon_evidence_ref,
            single_writer_evidence_ref=single_writer_evidence_ref,
        )


def _apply_historical_source_continuity_recovery_locked(
    *,
    root: Path,
    plan: HistoricalSourceContinuityRecoveryPlan,
    authorization: str,
    stopped_daemon_evidence_ref: str,
    single_writer_evidence_ref: str,
) -> HistoricalSourceContinuityRecoveryResult:
    _verify_plan(plan)
    if authorization != plan.plan_sha256 or plan.bound_confirmation != "historical-source-continuity-recovery":
        raise HistoricalSourceContinuityRecoveryError(
            "historical continuity recovery authorization does not bind this plan"
        )
    resolved = _real_directory(root, label="configured archive root")
    _revalidate(resolved, plan, stopped=stopped_daemon_evidence_ref, writer=single_writer_evidence_ref)
    planned_current = _evidence_from_plan(plan.source_after)
    refresh_payload = {
        "format": "polylogue.source-continuity-refresh.v1",
        "operation_id": plan.legacy_candidate_digest,
        "evidence_ref": "proof:historical-source-continuity-recovery:" + plan.plan_sha256,
        "backup_manifest": plan.pre_backup_manifest_path,
        "backup_manifest_sha256": plan.pre_backup_manifest_sha256,
        "mutation_receipt": plan.mutation_receipt_path,
        "mutation_receipt_sha256": plan.mutation_receipt_sha256,
        "train_id": load_durable_change_train_manifest(Path(plan.source_train_path)).train_id,
        "source_before": plan.source_before,
        "source_after": plan.source_after,
        "refreshed_at_ms": planned_current.observed_at_ms,
        "historical_bridge": {
            "pre_backup": plan.pre_backup_manifest_sha256,
            "post_backup": plan.post_backup_manifest_sha256,
            "legacy_candidate_count": plan.legacy_candidate_count,
            "legacy_candidate_digest": plan.legacy_candidate_digest,
            "census": plan.census,
        },
    }
    refresh_digest = _canonical_json_sha256(refresh_payload)
    refresh_path = _refresh_path(resolved, refresh_digest)
    command = f"POLYLOGUE_ARCHIVE_ROOT={plan.new_configured_root} polylogue ops maintenance source-continuity-recovery apply --plan <plan.json> --authorize {plan.plan_sha256} --output-format json"
    receipt_path = _receipt_path(resolved, plan)
    prepared = _sealed_receipt(
        state="prepared",
        revision=0,
        plan_sha256=plan.plan_sha256,
        authorization=authorization,
        train_before_sha256=plan.source_train_sha256,
        train_after_sha256=None,
        refresh_receipt_sha256=refresh_digest,
        resume_command=command,
    )
    existing_receipt = _load_recovery_receipt_for_update(receipt_path)
    if existing_receipt is not None:
        receipt = existing_receipt
        if receipt.plan_sha256 != plan.plan_sha256 or receipt.authorization != authorization:
            raise HistoricalSourceContinuityRecoveryError(
                "historical continuity recovery receipt belongs to another plan"
            )
        if receipt.state == "committed":
            train = load_durable_change_train_manifest(Path(plan.source_train_path))
            _validate_source_continuity_refresh_receipt(resolved, train)
            return HistoricalSourceContinuityRecoveryResult(
                state="committed",
                plan_sha256=plan.plan_sha256,
                receipt_path=str(receipt_path),
                refresh_receipt_path=str(refresh_path),
            )
    else:
        _write_receipt(receipt_path, prepared, expected=None)
        receipt = prepared
    encoded = {**refresh_payload, "refresh_sha256": refresh_digest}
    _write_refresh_receipt(refresh_path, encoded)
    path = Path(plan.source_train_path)
    train = load_durable_change_train_manifest(path)
    if _sha256(path) == plan.source_train_sha256:
        updated = recover_released_source_train_continuity(
            train, current_evidence=planned_current, proof_ref="proof:source-continuity-refresh:" + refresh_digest
        )
        write_durable_change_train_manifest(path, updated, expected_revision=plan.source_train_revision)
    else:
        _validate_source_continuity_refresh_receipt(resolved, train)
        if train.source_continuity_evidence is None or not _evidence_matches_plan(
            train.source_continuity_evidence, plan.source_after
        ):
            raise HistoricalSourceContinuityRecoveryError(
                "historical continuity recovery source train is neither exact before nor after"
            )
    committed = _sealed_receipt(
        state="committed",
        revision=1,
        plan_sha256=plan.plan_sha256,
        authorization=authorization,
        train_before_sha256=plan.source_train_sha256,
        train_after_sha256=_sha256(path),
        refresh_receipt_sha256=refresh_digest,
        resume_command=command,
    )
    _write_receipt(receipt_path, committed, expected=receipt.receipt_sha256)
    return HistoricalSourceContinuityRecoveryResult(
        state="committed",
        plan_sha256=plan.plan_sha256,
        receipt_path=str(receipt_path),
        refresh_receipt_path=str(refresh_path),
    )
