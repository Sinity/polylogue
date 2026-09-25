"""Build-specific source denominator from the daemon's production discovery route."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from polylogue.archive.zip_admission import ZIP_JSON_SUFFIXES, ZipAdmission, ZipBombError
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.provider_identity import canonical_acquisition_provider
from polylogue.core.raw_coordinates import zip_member_source_index
from polylogue.maintenance.receipt_fs import (
    atomic_replace_receipt,
    existing_maintenance_receipt_directory,
    maintenance_receipt_directory,
    read_optional_receipt,
)
from polylogue.sources.decoder_zip import (
    ZipEntryValidator,
    declared_artifact_provider,
    is_declared_artifact_path,
    provider_detection_path,
)
from polylogue.sources.live.discovery import _bounded_source_paths
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.source_acquisition_components import (
    ZipEntryReadContext,
    replay_zip_entry_acquisition_payloads,
    sniff_zip_provider,
)
from polylogue.sources.sqlite_snapshot import is_sqlite_path, sqlite_member_revision
from polylogue.sources.walk_faults import WalkRefusedError
from polylogue.storage.archive_identity import MAINTENANCE_STATE_DIRNAME

_PENDING_DIR = "production-source-baseline"
_PENDING_FILE = "pending.json"


class ProductionBaselineError(RuntimeError):
    """The build cannot prove its discovered source revisions were retained."""


@dataclass(frozen=True, slots=True)
class SourceDecision:
    source: str
    path: str
    disposition: str
    reason: str
    revision: str | None = None
    source_index: int | None = None


@dataclass(frozen=True, slots=True)
class ProductionSourceBaseline:
    operation_id: str
    source_signature: str
    decisions: tuple[SourceDecision, ...]
    digest: str

    @property
    def accepted(self) -> tuple[SourceDecision, ...]:
        return tuple(row for row in self.decisions if row.disposition == "accepted")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "polylogue.production-source-baseline.v1",
            "operation_id": self.operation_id,
            "source_signature": self.source_signature,
            "decisions": [
                {
                    "source": row.source,
                    "path": row.path,
                    "disposition": row.disposition,
                    "reason": row.reason,
                    "revision": row.revision,
                    "source_index": row.source_index,
                }
                for row in self.decisions
            ],
            "digest": self.digest,
        }

    def verify(self, source_db: Path) -> None:
        self.verify_integrity()
        faults = [row for row in self.decisions if row.disposition == "fault"]
        if faults:
            raise ProductionBaselineError(f"production source baseline has {len(faults)} unresolved discovery fault(s)")
        conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
        try:
            retained = {
                (
                    str(path),
                    int(source_index),
                    bytes(blob_hash).hex() if isinstance(blob_hash, (bytes, memoryview)) else str(blob_hash),
                )
                for path, source_index, blob_hash in conn.execute(
                    "SELECT source_path, source_index, blob_hash FROM raw_sessions"
                )
            }
        finally:
            conn.close()
        missing: list[str] = []
        for row in self.accepted:
            if (row.path, row.source_index or 0, row.revision) not in retained:
                missing.append(row.path)
        if missing:
            raise ProductionBaselineError(
                f"production source baseline has {len(missing)} unretained revision(s): {missing[:3]}"
            )

    def verify_integrity(self) -> None:
        payload = self.as_dict()
        digest = payload.pop("digest")
        if hashlib.sha256(_json(payload)).hexdigest() != digest:
            raise ProductionBaselineError("production source baseline integrity failed")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProductionSourceBaseline:
        try:
            if payload["schema"] != "polylogue.production-source-baseline.v1":
                raise ValueError("schema")
            raw_decisions = payload["decisions"]
            if not isinstance(raw_decisions, list):
                raise ValueError("decisions")
            decisions = tuple(
                SourceDecision(
                    source=str(row["source"]),
                    path=str(row["path"]),
                    disposition=str(row["disposition"]),
                    reason=str(row["reason"]),
                    revision=None if row["revision"] is None else str(row["revision"]),
                    source_index=None if row.get("source_index") is None else int(row["source_index"]),
                )
                for row in raw_decisions
                if isinstance(row, dict)
            )
            if len(decisions) != len(raw_decisions):
                raise ValueError("decisions")
            result = cls(
                operation_id=str(payload["operation_id"]),
                source_signature=str(payload["source_signature"]),
                decisions=decisions,
                digest=str(payload["digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionBaselineError("invalid pending production source baseline") from exc
        result.verify_integrity()
        return result


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _seal(operation_id: str, source_signature: str, decisions: tuple[SourceDecision, ...]) -> ProductionSourceBaseline:
    rows = tuple(
        sorted(
            decisions,
            key=lambda row: (row.source, row.path, row.disposition, row.source_index or 0, row.revision or ""),
        )
    )
    provisional = ProductionSourceBaseline(operation_id, source_signature, rows, "")
    payload = provisional.as_dict()
    payload.pop("digest")
    return ProductionSourceBaseline(operation_id, source_signature, rows, hashlib.sha256(_json(payload)).hexdigest())


def load_pending_production_baseline(archive_root: Path) -> ProductionSourceBaseline | None:
    with existing_maintenance_receipt_directory(archive_root, _PENDING_DIR) as directory_fd:
        if directory_fd is None:
            return None
        raw = read_optional_receipt(directory_fd, _PENDING_FILE)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionBaselineError("pending production source baseline is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ProductionBaselineError("pending production source baseline is not a JSON object")
    return ProductionSourceBaseline.from_dict(payload)


def publish_pending_production_baseline(archive_root: Path, baseline: ProductionSourceBaseline) -> None:
    baseline.verify_integrity()
    (archive_root / MAINTENANCE_STATE_DIRNAME).mkdir(mode=0o700, exist_ok=True)
    with maintenance_receipt_directory(archive_root, _PENDING_DIR) as directory_fd:
        atomic_replace_receipt(directory_fd, _PENDING_FILE, _json(baseline.as_dict()))


def clear_pending_production_baseline(archive_root: Path, baseline: ProductionSourceBaseline) -> None:
    with existing_maintenance_receipt_directory(archive_root, _PENDING_DIR) as directory_fd:
        if directory_fd is None:
            raise ProductionBaselineError("pending production source baseline disappeared before promotion")
        raw = read_optional_receipt(directory_fd, _PENDING_FILE)
        if raw is None or ProductionSourceBaseline.from_dict(json.loads(raw)).digest != baseline.digest:
            raise ProductionBaselineError("pending production source baseline changed before promotion")
        os.unlink(_PENDING_FILE, dir_fd=directory_fd)
        os.fsync(directory_fd)


def merge_pending_production_baseline(
    current: ProductionSourceBaseline, previous: ProductionSourceBaseline | None
) -> ProductionSourceBaseline:
    if previous is None:
        return current
    previous.verify_integrity()
    current_by_coordinate = {(row.source, row.path): row for row in current.decisions}
    rows = list(current.decisions)
    keys = {(row.source, row.path, row.disposition, row.source_index, row.revision) for row in rows}
    for row in previous.decisions:
        if row.disposition == "accepted":
            key = (row.source, row.path, row.disposition, row.source_index, row.revision)
            if key not in keys:
                rows.append(row)
                keys.add(key)
        elif row.disposition == "fault":
            replacement = current_by_coordinate.get((row.source, row.path))
            resolved = replacement is not None and replacement.disposition in {"accepted", "alias"}
            if row.reason == "absent_root" and replacement is not None and replacement.reason == "available_root":
                resolved = True
            if not resolved:
                key = (row.source, row.path, row.disposition, row.source_index, row.revision)
                if key not in keys:
                    rows.append(row)
                    keys.add(key)
    return _seal(current.operation_id, current.source_signature, tuple(rows))


def _revision(path: Path) -> str:
    if is_sqlite_path(path):
        return sqlite_member_revision(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_members(path: Path, source_name: str) -> tuple[SourceDecision, ...]:
    members: list[SourceDecision] = []
    with zipfile.ZipFile(path) as archive:
        central_directory = archive.infolist()
        ordinals = {id(info): ordinal for ordinal, info in enumerate(central_directory)}
        detection_entries = [
            info
            for info in ZipAdmission(zip_path=path).filter_entries(
                central_directory, allowed_suffixes=ZIP_JSON_SUFFIXES
            )
            if provider_detection_path(info.filename)
        ]
        provider = Provider.from_string(canonical_acquisition_provider(source_name, source_name=source_name))
        if provider is Provider.UNKNOWN:
            provider = sniff_zip_provider(archive, detection_entries) or provider
        allowed_path = is_declared_artifact_path if provider is Provider.UNKNOWN else None

        def excluded(info: zipfile.ZipInfo, reason: str) -> None:
            members.append(SourceDecision(source_name, f"{path}:{info.filename}", "excluded", reason))

        def fault(info: zipfile.ZipInfo, reason: str) -> None:
            members.append(SourceDecision(source_name, f"{path}:{info.filename}", "fault", reason))

        entries = ZipEntryValidator(provider, cursor_state=None, zip_path=path).filter_entries(
            central_directory,
            allowed_path=allowed_path,
            on_rejected=fault,
            on_unselected=excluded,
        )
        for info in entries:
            if info.file_size == 0:
                excluded(info, "empty_member")
                continue
            try:
                entry_provider = provider
                if provider is Provider.UNKNOWN:
                    entry_provider = declared_artifact_provider(info.filename) or provider
                context = ZipEntryReadContext(
                    Source(name=source_name, path=path.parent),
                    path,
                    info,
                    None,
                    entry_provider,
                    None,  # type: ignore[arg-type]
                )
                for payload in replay_zip_entry_acquisition_payloads(archive, context):
                    split = payload.source_index or 0
                    members.append(
                        SourceDecision(
                            source_name,
                            f"{path}:{info.filename}",
                            "accepted",
                            "archive_member",
                            hashlib.sha256(payload.payload_bytes).hexdigest(),
                            zip_member_source_index(entry_ordinal=ordinals[id(info)], split_index=split),
                        )
                    )
            except (OSError, UnicodeError, ValueError, ZipBombError, zipfile.BadZipFile) as exc:
                fault(info, f"archive_member_unreadable:{exc}")
                continue
    return tuple(members)


def capture_production_source_baseline(
    sources: tuple[WatchSource, ...], *, operation_id: str
) -> ProductionSourceBaseline:
    """Observe the exact typed sources through the same walker as file intake.

    This call runs before any intake cursor filtering. The baseline is immutable;
    files that arrive after this observation are outside this build's denominator.
    """
    if not sources:
        raise ProductionBaselineError("cold build has no effective watch sources")
    signature = hashlib.sha256(
        _json(
            [
                (
                    source.name,
                    str(source.root),
                    source.suffixes,
                    sorted(source.ignored_dir_names),
                    source.source_id,
                    source.role,
                    source.allow_path_scoped_artifacts,
                    source.required,
                )
                for source in sources
            ]
        )
    ).hexdigest()
    decisions: list[SourceDecision] = []
    observed: list[tuple[str, Path, str, str]] = []
    for source in sources:
        if not source.root.is_dir():
            decisions.append(
                SourceDecision(
                    source.name,
                    str(source.root),
                    "fault" if source.required else "excluded",
                    "absent_root",
                )
            )
            continue
        decisions.append(SourceDecision(source.name, str(source.root), "excluded", "available_root"))

        def record(path: Path, disposition: str, reason: str, source_name: str = source.name) -> None:
            observed.append((source_name, path, disposition, reason))

        try:
            _bounded_source_paths(source, sources, limit=sys.maxsize, after=None, on_disposition=record, collect=False)
        except WalkRefusedError as exc:
            decisions.append(SourceDecision(source.name, str(source.root), "fault", str(exc)))
            continue
    accepted_real = {
        str(path.resolve())
        for _, path, disposition, _ in observed
        if disposition == "accepted" and not path.is_symlink()
    }
    independent_roots = {
        (source.name, source.root.resolve())
        for source in sources
        if source.root.is_dir() and not source.root.is_symlink()
    }
    for source_name, path, disposition, reason in observed:
        if path.is_symlink():
            target = str(path.resolve())
            independently_accepted = (
                target in accepted_real
                or any(candidate.startswith(target + os.sep) for candidate in accepted_real)
                or (
                    path.is_dir()
                    and any(
                        other_name != source_name and Path(target).is_relative_to(root)
                        for other_name, root in independent_roots
                    )
                )
            )
            if disposition in {"accepted", "alias"} or reason in {
                "escaping_symlink",
                "broken_symlink",
                "symlink_cycle",
            }:
                if independently_accepted:
                    disposition, reason = "alias", "independently_accepted_target"
                else:
                    disposition, reason = "fault", "alias_target_not_independently_accepted"
        if disposition == "accepted":
            try:
                if path.suffix.lower() == ".zip":
                    decisions.append(SourceDecision(source_name, str(path), "excluded", "expanded_to_members"))
                    decisions.extend(_archive_members(path, source_name))
                    continue
                revision = _revision(path)
            except (OSError, sqlite3.Error, ValueError, zipfile.BadZipFile) as exc:
                decisions.append(SourceDecision(source_name, str(path), "fault", f"revision_unreadable:{exc}"))
                continue
        else:
            revision = None
        decisions.append(SourceDecision(source_name, str(path), disposition, reason, revision))
    return _seal(operation_id, signature, tuple(decisions))
