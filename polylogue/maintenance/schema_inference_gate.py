"""Read-only go/no-go receipt for the schema-inference prerequisite.

This gate composes the existing corpus-fidelity registry with the four
source-tier hard gates named by polylogue-r9xsj. It never runs a repair or
cleanup actuator. SQLite inputs are opened through the repository's
read-only connection helper; the only write is the caller-selected JSON
receipt path.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NotRequired, TypeAlias, TypedDict, cast

from polylogue.maintenance.archive_verification import CORPUS_FIDELITY_CHECKS, verify_archive
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.introspection import table_exists
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.version import POLYLOGUE_VERSION

RECEIPT_SCHEMA = "polylogue.schema-inference-gate.v2"
GATE_VERSION = "3"
DEFAULT_SAMPLE_LIMIT = 10
RECEIPT_FILENAME = "schema-inference-gate-receipt.json"
SCHEMA_INFERENCE_RECEIPT_ENV = "POLYLOGUE_SCHEMA_INFERENCE_RECEIPT"
RECEIPT_TTL = timedelta(hours=24)
RECEIPT_CLOCK_SKEW = timedelta(minutes=5)

_ALLOWED_RESIDUAL_EXPLANATIONS = frozenset(
    {"materialized", "superseded-duplicate", "legitimately-excluded-non-conversation"}
)
_CAUSE_EXPLANATIONS = {"byte-revision-governed": "superseded-duplicate"}
GroundTruthKey: TypeAlias = tuple[str, int]
GroundTruthDisposition: TypeAlias = Literal[
    "materialized",
    "superseded-duplicate",
    "legitimately-excluded-non-conversation",
    "unreconciled-source-raw",
]
ExternalDisposition: TypeAlias = Literal["unmatched-external-file", "cross-origin-source"]


@dataclass(frozen=True, slots=True)
class _ExternalGroundTruthFile:
    root_index: int
    relative_path: str
    content_hash: str
    size: int
    path: Path

    @property
    def key(self) -> GroundTruthKey:
        return (self.content_hash, self.size)


@dataclass(frozen=True, slots=True)
class _SourceRawGroundTruth:
    raw_id: str
    origin: str
    native_id: str | None
    logical_source_key: str | None
    source_path: str
    blob_hash: str
    blob_size: int
    disposition: GroundTruthDisposition

    @property
    def key(self) -> GroundTruthKey:
        return (self.blob_hash, self.blob_size)


class _RawGroundTruthProvenance(TypedDict):
    raw_id: str
    origin: str
    native_id: str | None
    logical_source_key: str | None
    source_path: str
    blob_hash: str
    blob_size: int
    matched_external_relative_path: str | None
    matched_external_hash: str | None
    matched_external_size: int | None
    disposition: GroundTruthDisposition


class _ExternalGroundTruthReceipt(TypedDict):
    root_index: int
    relative_path: str
    hash: str
    size: int
    disposition: ExternalDisposition
    other_origins: NotRequired[list[str]]


class _UnmatchedSourceRawReceipt(TypedDict):
    raw_id: str
    blob_hash: str
    blob_size: int
    disposition: Literal["unreconciled-source-raw"]


# Hooks and browser capture are the explicit no-external-ground-truth
# exceptions from r9xsj. Every other origin present in source.db must receive
# a caller-declared, readable external root and is reconciled by content hash.
GROUND_TRUTH_INPUTS: dict[str, dict[str, object]] = {
    "codex-session": {
        "exempt": False,
    },
    "claude-code-session": {
        "exempt": False,
    },
    "chatgpt-export": {"exempt": False},
    "claude-ai-export": {"exempt": False},
    "aistudio-drive": {"exempt": False},
    "grok-export": {"exempt": False},
    "hermes-session": {"exempt": False},
    "antigravity-session": {"exempt": False},
    "gemini-cli-session": {"exempt": False},
    "hooks": {
        "paths": ["source.db:raw_hook_events"],
        "exempt": True,
        "reason": "hooks are the live origin and have no external filesystem ground truth",
    },
    "browser-capture": {
        "paths": ["source.db:raw_sessions"],
        "exempt": True,
        "reason": "browser capture is the origin and has no re-scannable external export",
    },
}

_HARD_GATE_SQL: dict[str, str] = {
    "zero-surviving-quarantine": "SELECT COUNT(*) FROM raw_sessions WHERE revision_authority = 'quarantined'",
    "zero-quarantine-without-logical-source-key": (
        "SELECT COUNT(*) FROM raw_sessions WHERE revision_authority = 'quarantined' AND logical_source_key IS NULL"
    ),
    "zero-unresolved-raw-authority-blockers": (
        "SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL"
    ),
}


class SchemaInferenceGateError(ValueError):
    """The gate input or receipt shape is invalid."""


@dataclass(frozen=True, slots=True)
class SchemaInferenceGateResult:
    """The durable receipt payload and its process exit verdict."""

    payload: dict[str, object]

    @property
    def passed(self) -> bool:
        return self.payload.get("verdict") == "PASS"


def _sample(values: list[str], limit: int) -> list[str]:
    return values[:limit]


def _query_count(
    conn: sqlite3.Connection,
    *,
    gate_id: str,
    sql: str,
    sample_sql: str,
    sample_limit: int,
) -> dict[str, object]:
    count = int(conn.execute(sql).fetchone()[0])
    samples = [str(row[0]) for row in conn.execute(sample_sql, (sample_limit,))]
    return {
        "gate": gate_id,
        "sql": sql,
        "count": count,
        "samples": samples,
        "passed": count == 0,
        "reason": None if count == 0 else f"{count} row(s) matched the required zero-result query",
    }


def _source_counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    rows = conn.execute(
        """
        SELECT origin, COUNT(*), COUNT(DISTINCT blob_hash), COALESCE(SUM(blob_size), 0)
        FROM raw_sessions
        GROUP BY origin
        ORDER BY origin
        """
    ).fetchall()
    return {
        str(origin): {
            "raw_rows": int(raw_rows),
            "distinct_blobs": int(distinct_blobs),
            "bytes": int(bytes_total),
        }
        for origin, raw_rows, distinct_blobs, bytes_total in rows
    }


def _referenced_blob_hashes(conn: sqlite3.Connection) -> set[str]:
    """Return the durable source-tier blob universe as canonical hashes."""

    sql = "SELECT blob_hash FROM raw_sessions"
    if table_exists(conn, "blob_refs"):
        sql += " UNION SELECT blob_hash FROM blob_refs"
    return {bytes(row[0]).hex() for row in conn.execute(sql) if row[0] is not None}


def _source_blob_denominators(conn: sqlite3.Connection) -> dict[str, int]:
    """Return the source-tier hash universe an independent blob scan must cover."""

    return {"distinct_referenced_blob_hashes": len(_referenced_blob_hashes(conn))}


def _duplicate_gate(
    source: sqlite3.Connection,
    *,
    index_path: Path,
    sample_limit: int,
) -> dict[str, object]:
    """Classify every duplicate of an indexed twin under explicit rules."""

    sql = (
        "raw_sessions rows sharing blob_hash with an indexed twin, excluding "
        "materialized rows, valid supersession receipts, and typed "
        "non-session census rows"
    )
    source.row_factory = sqlite3.Row
    try:
        source.execute("ATTACH DATABASE ? AS idx_tier", (f"file:{index_path}?mode=ro",))
        has_receipts = table_exists(source, "raw_byte_duplicate_supersession_receipts")
        has_census = table_exists(source, "raw_membership_census")
        census_status = (
            "(SELECT c.status FROM raw_membership_census c WHERE c.raw_id = r.raw_id)" if has_census else "NULL"
        )
        rows = source.execute(
            f"""
            SELECT r.raw_id, hex(r.blob_hash) AS blob_hash, r.blob_size,
                   {census_status} AS census_status,
                   EXISTS(SELECT 1 FROM idx_tier.sessions own WHERE own.raw_id = r.raw_id) AS own_indexed
            FROM raw_sessions r
            WHERE EXISTS (
                SELECT 1
                FROM raw_sessions twin
                JOIN idx_tier.sessions twin_session ON twin_session.raw_id = twin.raw_id
                WHERE twin.blob_hash = r.blob_hash AND twin.raw_id != r.raw_id
            )
            ORDER BY r.raw_id
            """
        ).fetchall()

        unresolved: list[dict[str, object]] = []
        invalid_receipts: list[dict[str, object]] = []
        resolved_by_rule: Counter[str] = Counter()
        for row in rows:
            raw_id = str(row["raw_id"])
            if bool(row["own_indexed"]):
                resolved_by_rule["materialized"] += 1
                continue

            receipt = None
            if has_receipts:
                receipt = source.execute(
                    "SELECT blob_hash, duplicate_of_raw_id, duplicate_of_session_id, blob_size "
                    "FROM raw_byte_duplicate_supersession_receipts WHERE raw_id = ?",
                    (raw_id,),
                ).fetchone()
            if receipt is not None:
                receipt_hash = bytes(receipt[0]).hex()
                twin_ok = source.execute(
                    """
                    SELECT 1
                    FROM raw_sessions twin
                    JOIN idx_tier.sessions twin_session ON twin_session.raw_id = twin.raw_id
                    WHERE twin.raw_id = ? AND twin_session.session_id = ?
                      AND twin.blob_hash = ? AND twin.blob_size = ?
                    """,
                    (
                        str(receipt[1]),
                        str(receipt[2]),
                        bytes.fromhex(str(row["blob_hash"])),
                        int(row["blob_size"]),
                    ),
                ).fetchone()
                if (
                    receipt_hash == str(row["blob_hash"]).lower()
                    and int(receipt[3]) == int(row["blob_size"])
                    and twin_ok
                ):
                    resolved_by_rule["superseded-duplicate"] += 1
                    continue
                invalid_receipts.append(
                    {
                        "raw_id": raw_id,
                        "reason": "supersession receipt does not match raw hash/size and indexed twin",
                    }
                )
                continue

            if str(row["census_status"] or "") == "non_session":
                exclusion = source.execute(
                    """
                    SELECT blob_hash, blob_size, indexed_twin_raw_id,
                           indexed_twin_session_id, parser_fingerprint
                    FROM raw_non_session_duplicate_exclusion_receipts
                    WHERE raw_id = ?
                    """,
                    (raw_id,),
                ).fetchone()
                if exclusion is not None:
                    exclusion_hash = bytes(exclusion[0]).hex()
                    exclusion_twin_ok = source.execute(
                        """
                        SELECT 1
                        FROM raw_sessions twin
                        JOIN idx_tier.sessions twin_session ON twin_session.raw_id = twin.raw_id
                        WHERE twin.raw_id = ? AND twin_session.session_id = ?
                          AND twin.blob_hash = ? AND twin.blob_size = ?
                        """,
                        (
                            str(exclusion[2]),
                            str(exclusion[3]),
                            bytes.fromhex(str(row["blob_hash"])),
                            int(row["blob_size"]),
                        ),
                    ).fetchone()
                    if (
                        exclusion_hash == str(row["blob_hash"]).lower()
                        and int(exclusion[1]) == int(row["blob_size"])
                        and bool(str(exclusion[4]).strip())
                        and exclusion_twin_ok
                    ):
                        resolved_by_rule["legitimately-excluded-non-conversation"] += 1
                        continue
                invalid_receipts.append(
                    {
                        "raw_id": raw_id,
                        "reason": "non-session exclusion lacks a content-bound receipt to an indexed twin",
                    }
                )
                continue
            unresolved.append(
                {
                    "raw_id": raw_id,
                    "blob_hash": str(row["blob_hash"]).lower(),
                    "blob_size": int(row["blob_size"]),
                    "reason": "no materialization, valid supersession receipt, or typed exclusion",
                }
            )
    finally:
        source.row_factory = None

    failure_count = len(unresolved) + len(invalid_receipts)
    return {
        "gate": "zero-unexplained-byte-duplicates",
        "rule": sql,
        "receipt_table_present": has_receipts,
        "candidate_count": len(rows),
        "resolved_by_rule": dict(sorted(resolved_by_rule.items())),
        "unresolved_count": len(unresolved),
        "unresolved": unresolved[:sample_limit],
        "invalid_receipt_count": len(invalid_receipts),
        "invalid_receipts": invalid_receipts[:sample_limit],
        "count": failure_count,
        "passed": failure_count == 0,
        "reason": None
        if failure_count == 0
        else f"{failure_count} duplicate disposition(s) are unexplained or invalid",
    }


def _run_source_gates(archive_root: Path, *, index_path: Path, sample_limit: int) -> dict[str, object]:
    source_path = archive_root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
    if not source_path.exists() or not index_path.exists():
        return {
            "gates": {
                gate: {
                    "gate": gate,
                    "sql": sql,
                    "count": 1,
                    "samples": [],
                    "passed": False,
                    "reason": "required source.db or active index.db is missing",
                }
                for gate, sql in _HARD_GATE_SQL.items()
            },
            "duplicate_gate": {
                "gate": "zero-unexplained-byte-duplicates",
                "count": 1,
                "passed": False,
                "reason": "required source.db or active index.db is missing",
            },
            "source_counts": {},
            "blob_denominators": {"distinct_referenced_blob_hashes": 0},
        }

    with open_readonly_connection(source_path) as source:
        source.row_factory = sqlite3.Row
        gates = {
            "zero-surviving-quarantine": _query_count(
                source,
                gate_id="zero-surviving-quarantine",
                sql=_HARD_GATE_SQL["zero-surviving-quarantine"],
                sample_sql="SELECT raw_id FROM raw_sessions WHERE revision_authority = 'quarantined' ORDER BY raw_id LIMIT ?",
                sample_limit=sample_limit,
            ),
            "zero-quarantine-without-logical-source-key": _query_count(
                source,
                gate_id="zero-quarantine-without-logical-source-key",
                sql=_HARD_GATE_SQL["zero-quarantine-without-logical-source-key"],
                sample_sql="SELECT raw_id FROM raw_sessions WHERE revision_authority = 'quarantined' AND logical_source_key IS NULL ORDER BY raw_id LIMIT ?",
                sample_limit=sample_limit,
            ),
        }
        if table_exists(source, "raw_authority_blockers"):
            gates["zero-unresolved-raw-authority-blockers"] = _query_count(
                source,
                gate_id="zero-unresolved-raw-authority-blockers",
                sql=_HARD_GATE_SQL["zero-unresolved-raw-authority-blockers"],
                sample_sql="SELECT blocker_id FROM raw_authority_blockers WHERE resolved_at_ms IS NULL ORDER BY blocker_id LIMIT ?",
                sample_limit=sample_limit,
            )
        else:
            gates["zero-unresolved-raw-authority-blockers"] = {
                "gate": "zero-unresolved-raw-authority-blockers",
                "sql": _HARD_GATE_SQL["zero-unresolved-raw-authority-blockers"],
                "count": 1,
                "samples": [],
                "passed": False,
                "reason": "required raw_authority_blockers table is missing",
            }
        source_counts = _source_counts(source)
        blob_denominators = _source_blob_denominators(source)
        duplicate_gate = _duplicate_gate(source, index_path=index_path, sample_limit=sample_limit)
    return {
        "gates": gates,
        "duplicate_gate": duplicate_gate,
        "source_counts": source_counts,
        "blob_denominators": blob_denominators,
    }


def _failed_source_gates(reason: str) -> dict[str, object]:
    gates = {
        gate: {
            "gate": gate,
            "sql": sql,
            "count": 1,
            "samples": [],
            "passed": False,
            "reason": reason,
        }
        for gate, sql in _HARD_GATE_SQL.items()
    }
    return {
        "gates": gates,
        "duplicate_gate": {
            "gate": "zero-unexplained-byte-duplicates",
            "count": 1,
            "passed": False,
            "reason": reason,
        },
        "source_counts": {},
        "blob_denominators": {"distinct_referenced_blob_hashes": 0},
    }


def _tier_schema_identity(archive_root: Path, location: ArchiveLocation) -> dict[str, object]:
    tiers: dict[str, object] = {}
    for tier, spec in ARCHIVE_TIER_SPECS.items():
        path = location.active_index_path if tier is ArchiveTier.INDEX else archive_root / spec.filename
        entry: dict[str, object] = {
            "path": str(path),
            "expected_user_version": spec.version,
            "durability": spec.durability,
            "exists": path.exists(),
            "actual_user_version": None,
        }
        if path.exists():
            try:
                with open_readonly_connection(path) as conn:
                    entry["actual_user_version"] = int(conn.execute("PRAGMA user_version").fetchone()[0])
            except sqlite3.Error as exc:
                entry["error"] = str(exc)
        entry["matches_expected"] = entry["actual_user_version"] == spec.version
        tiers[tier.value] = entry
    return {"archive": ArchiveIdentity.resolve_location(location).as_dict(), "tiers": tiers}


def _resolve_receipt_path(receipt_path: Path, *, archive_root: Path) -> Path:
    """Accept only the dedicated receipt filename outside the archive file set."""

    target = receipt_path.expanduser().resolve()
    root = archive_root.resolve()
    if target.name != RECEIPT_FILENAME:
        raise SchemaInferenceGateError(f"receipt filename must be {RECEIPT_FILENAME!r}")
    try:
        target.relative_to(root)
    except ValueError:
        return target
    raise SchemaInferenceGateError("receipt path must be outside the archive root")


def resolve_schema_inference_receipt_reference(archive_root: Path, receipt_path: Path | None = None) -> Path:
    """Resolve the policy-controlled receipt reference used by rebuild callers."""

    if receipt_path is not None:
        return receipt_path.expanduser().resolve()
    configured = os.environ.get(SCHEMA_INFERENCE_RECEIPT_ENV, "").strip()
    if not configured:
        raise SchemaInferenceGateError(
            f"a fresh schema-inference receipt is required; pass a receipt path or set {SCHEMA_INFERENCE_RECEIPT_ENV}"
        )
    return Path(configured).expanduser().resolve()


def _archive_receipt_identity(location: ArchiveLocation) -> dict[str, object]:
    identity = ArchiveIdentity.resolve_location(location)
    return {
        "configured_root": str(location.configured_root),
        "durable_id": identity.durable_id,
        "source_tier": identity.tier("source").as_dict(),
        "user_tier": identity.tier("user").as_dict(),
    }


def _parse_receipt_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SchemaInferenceGateError(f"receipt field {field!r} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SchemaInferenceGateError(f"receipt field {field!r} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise SchemaInferenceGateError(f"receipt field {field!r} must include a timezone")
    return parsed.astimezone(UTC)


def _canonical_external_ground_truth_digest(origins: Mapping[str, object]) -> str:
    """Digest the complete external corpus inventory recorded in a receipt."""

    canonical: list[dict[str, object]] = []
    for origin in sorted(origins):
        evidence = _as_dict(origins[origin])
        mapping = evidence.get("raw_external_mapping")
        if not isinstance(mapping, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("raw_id"), str)
            and isinstance(item.get("source_path"), str)
            and isinstance(item.get("disposition"), str)
            for item in mapping
        ):
            raise SchemaInferenceGateError(f"raw external mapping for {origin} is missing or malformed")
        if bool(evidence.get("exempt")):
            canonical.append(
                {
                    "origin": origin,
                    "exempt": True,
                    "reason": evidence.get("reason"),
                    "mapping": sorted(mapping, key=lambda item: str(item["raw_id"])),
                }
            )
            continue
        roots = evidence.get("declared_roots")
        inventory = evidence.get("external_inventory")
        if not isinstance(roots, list) or not all(isinstance(root, str) for root in roots):
            raise SchemaInferenceGateError(f"ground truth roots for {origin} are missing or malformed")
        if not isinstance(inventory, list):
            raise SchemaInferenceGateError(f"ground truth inventory for {origin} is missing or malformed")
        files: list[dict[str, object]] = []
        for item in inventory:
            record = _as_dict(item)
            root_index = record.get("root_index")
            relative_path = record.get("relative_path")
            content_hash = record.get("hash")
            size = record.get("size")
            if (
                not isinstance(root_index, int)
                or isinstance(root_index, bool)
                or not isinstance(relative_path, str)
                or not isinstance(content_hash, str)
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
            ):
                raise SchemaInferenceGateError(f"ground truth inventory for {origin} contains a malformed file")
            files.append(
                {
                    "root_index": root_index,
                    "relative_path": relative_path,
                    "hash": content_hash.lower(),
                    "size": size,
                }
            )
        mappings: list[dict[str, object]] = []
        for item in mapping:
            record = _as_dict(item)
            raw_id = record.get("raw_id")
            source_path = record.get("source_path")
            disposition = record.get("disposition")
            blob_hash = record.get("blob_hash")
            blob_size = record.get("blob_size")
            external_relative_path = record.get("external_relative_path")
            external_hash = record.get("external_hash")
            external_size = record.get("external_size")
            if (
                not isinstance(raw_id, str)
                or not isinstance(source_path, str)
                or not isinstance(disposition, str)
                or not isinstance(blob_hash, str)
                or not isinstance(blob_size, int)
                or isinstance(blob_size, bool)
                or blob_size < 0
                or (external_relative_path is not None and not isinstance(external_relative_path, str))
                or (external_hash is not None and not isinstance(external_hash, str))
                or (
                    external_size is not None
                    and (not isinstance(external_size, int) or isinstance(external_size, bool))
                )
            ):
                raise SchemaInferenceGateError(f"raw external mapping for {origin} contains a malformed row")
            mappings.append(
                {
                    "raw_id": raw_id,
                    "source_path": source_path,
                    "blob_hash": blob_hash.lower(),
                    "blob_size": blob_size,
                    "external_relative_path": external_relative_path,
                    "external_hash": external_hash.lower() if isinstance(external_hash, str) else None,
                    "external_size": external_size,
                    "disposition": disposition,
                }
            )
        canonical.append(
            {
                "origin": origin,
                "roots": sorted(str(Path(root).expanduser().resolve()) for root in roots),
                "files": sorted(
                    files,
                    key=lambda item: (int(cast(int, item["root_index"])), str(item["relative_path"])),
                ),
                "mapping": sorted(mappings, key=lambda item: str(item["raw_id"])),
            }
        )
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_blob_root(blob_root: Path) -> dict[str, object]:
    """Fingerprint the canonical namespace before and after full verification."""

    entries: list[dict[str, object]] = []
    for entry in BlobStore(blob_root).iter_namespace():
        item: dict[str, object] = {
            "path": entry.relative_path,
            "kind": entry.kind.value,
            "hash": entry.hash_hex,
            "issue": entry.issue.value if entry.issue is not None else None,
        }
        try:
            stat = entry.path.stat()
            item["size"] = stat.st_size
            item["inode"] = stat.st_ino
            item["mtime_ns"] = stat.st_mtime_ns
        except OSError as exc:
            item["stat_error"] = str(exc)
        entries.append(item)
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(blob_root),
        "entry_count": len(entries),
        "canonical_blob_count": sum(1 for entry in entries if entry["kind"] == "blob"),
        "digest": hashlib.sha256(encoded).hexdigest(),
    }


def _full_blob_hash_evidence(archive_root: Path, *, referenced_hashes: set[str]) -> dict[str, object]:
    """Run the production BlobStore full verification route and bind its snapshot."""

    blob_root = archive_root / "blob"
    before = _snapshot_blob_root(blob_root)
    verification = BlobStore(blob_root).verify_all(max_failures=100)
    after = _snapshot_blob_root(blob_root)
    canonical_hashes = set(BlobStore(blob_root).iter_all())
    missing_references = sorted(referenced_hashes - canonical_hashes)
    errors: list[str] = []
    if before["digest"] != after["digest"]:
        errors.append("blob-root snapshot changed during full verification")
    if verification.failures:
        errors.append("BlobStore.verify_all reported integrity failures")
    if verification.truncated:
        errors.append("BlobStore.verify_all truncated before a complete result")
    if missing_references:
        errors.append("referenced source blobs are absent from the verified blob root")
    return {
        "passed": not errors,
        "verifier": {
            "identity": "polylogue.storage.blob_store.BlobStore.verify_all",
            "polylogue_version": POLYLOGUE_VERSION,
        },
        "before_snapshot": before,
        "after_snapshot": after,
        "counts": {
            "scanned_blobs": verification.checked,
            "scanned_bytes": verification.checked_bytes,
            "referenced_hashes": len(referenced_hashes),
            "missing_referenced_hashes": len(missing_references),
        },
        "failures": [
            {"hash": failure.hash, "reason": failure.reason, "detail": failure.detail, "path": failure.path}
            for failure in verification.failures
        ],
        "missing_references": _sample(missing_references, DEFAULT_SAMPLE_LIMIT),
        "errors": errors,
        "reason": "; ".join(errors) if errors else None,
    }


def _iter_ground_truth_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _file_hash(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


def _external_inventory(roots: Sequence[Path]) -> list[_ExternalGroundTruthFile]:
    """Return a stable, relative-path inventory for declared external roots."""

    inventory: list[_ExternalGroundTruthFile] = []
    for root_index, root in enumerate(sorted({path.resolve() for path in roots}, key=str)):
        for path in _iter_ground_truth_files(root):
            resolved_path = path.resolve()
            digest, size = _file_hash(resolved_path)
            relative_path = resolved_path.name if root.is_file() else resolved_path.relative_to(root).as_posix()
            inventory.append(
                _ExternalGroundTruthFile(
                    root_index=root_index,
                    relative_path=relative_path,
                    content_hash=digest,
                    size=size,
                    path=resolved_path,
                )
            )
    inventory.sort(key=lambda item: (item.relative_path, item.root_index))
    return inventory


def _external_receipt(
    item: _ExternalGroundTruthFile,
    disposition: ExternalDisposition,
    *,
    other_origins: list[str] | None = None,
) -> _ExternalGroundTruthReceipt:
    receipt: _ExternalGroundTruthReceipt = {
        "root_index": item.root_index,
        "relative_path": item.relative_path,
        "hash": item.content_hash,
        "size": item.size,
        "disposition": disposition,
    }
    if other_origins:
        receipt["other_origins"] = other_origins
    return receipt


def _raw_provenance(
    raw: _SourceRawGroundTruth,
    matched_external: _ExternalGroundTruthFile | None,
) -> _RawGroundTruthProvenance:
    return {
        "raw_id": raw.raw_id,
        "origin": raw.origin,
        "native_id": raw.native_id,
        "logical_source_key": raw.logical_source_key,
        "source_path": raw.source_path,
        "blob_hash": raw.blob_hash,
        "blob_size": raw.blob_size,
        "matched_external_relative_path": (matched_external.relative_path if matched_external is not None else None),
        "matched_external_hash": matched_external.content_hash if matched_external is not None else None,
        "matched_external_size": matched_external.size if matched_external is not None else None,
        "disposition": raw.disposition,
    }


def _raw_external_mapping(
    source: sqlite3.Connection,
    *,
    origin: str,
    inventory: Sequence[_ExternalGroundTruthFile],
    exempt: bool,
) -> list[dict[str, object]]:
    """Persist one deterministic external-file choice for every source raw."""

    rows = source.execute(
        """
        SELECT raw_id, source_path, hex(blob_hash), blob_size
        FROM raw_sessions
        WHERE origin = ?
        ORDER BY raw_id
        """,
        (origin,),
    ).fetchall()
    by_key: dict[GroundTruthKey, list[_ExternalGroundTruthFile]] = {}
    for item in inventory:
        by_key.setdefault(item.key, []).append(item)
    mapping: list[dict[str, object]] = []
    for raw_id, source_path, blob_hash, blob_size in rows:
        canonical_hash = str(blob_hash).lower()
        canonical_size = int(blob_size)
        match = by_key.get((canonical_hash, canonical_size), [None])[0]
        mapping.append(
            {
                "raw_id": str(raw_id),
                "source_path": str(source_path),
                "blob_hash": canonical_hash,
                "blob_size": canonical_size,
                "external_relative_path": match.relative_path if match is not None else None,
                "external_hash": match.content_hash if match is not None else None,
                "external_size": match.size if match is not None else None,
                "disposition": "origin-exempt"
                if exempt
                else "matched-external"
                if match is not None
                else "unmatched-source-raw",
            }
        )
    return mapping


def _raw_dispositions(
    source: sqlite3.Connection,
    *,
    index_path: Path,
) -> dict[str, list[_SourceRawGroundTruth]]:
    """Classify raw rows using durable receipts and the indexed read model."""

    source_rows = source.execute(
        """
        SELECT raw_id, origin, native_id, logical_source_key, source_path, hex(blob_hash), blob_size
        FROM raw_sessions
        ORDER BY origin, raw_id
        """
    ).fetchall()
    indexed_raw_ids: set[str] = set()
    if index_path.exists():
        with open_readonly_connection(index_path) as index:
            if table_exists(index, "sessions"):
                indexed_raw_ids = {
                    str(row[0]) for row in index.execute("SELECT raw_id FROM sessions WHERE raw_id IS NOT NULL")
                }

    def receipt_ids(table: str) -> set[str]:
        return (
            {str(row[0]) for row in source.execute(f"SELECT raw_id FROM {table}")}
            if table_exists(source, table)
            else set()
        )

    superseded_ids = receipt_ids("raw_byte_duplicate_supersession_receipts")
    superseded_ids.update(receipt_ids("raw_quarantine_group_dedup_receipts"))
    excluded_ids = receipt_ids("raw_non_session_duplicate_exclusion_receipts")

    rows_by_origin: dict[str, list[_SourceRawGroundTruth]] = {}
    for row in source_rows:
        raw_id = str(row[0])
        disposition: GroundTruthDisposition
        if raw_id in indexed_raw_ids:
            disposition = "materialized"
        elif raw_id in superseded_ids:
            disposition = "superseded-duplicate"
        elif raw_id in excluded_ids:
            disposition = "legitimately-excluded-non-conversation"
        else:
            disposition = "unreconciled-source-raw"
        rows_by_origin.setdefault(str(row[1]), []).append(
            _SourceRawGroundTruth(
                raw_id=raw_id,
                origin=str(row[1]),
                native_id=None if row[2] is None else str(row[2]),
                logical_source_key=None if row[3] is None else str(row[3]),
                source_path=str(row[4]),
                blob_hash=str(row[5]).lower(),
                blob_size=int(row[6]),
                disposition=disposition,
            )
        )
    return rows_by_origin


def _ground_truth_evidence(
    archive_root: Path,
    *,
    index_path: Path,
    source_counts: Mapping[str, Mapping[str, int]],
    roots: Mapping[str, Sequence[Path]] | None,
) -> dict[str, object]:
    """Reconcile source raws and external files in both directions by origin."""

    root_map = roots or {}
    source_path = archive_root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
    evidence: dict[str, object] = {}
    errors: list[str] = []
    with open_readonly_connection(source_path) as source:
        rows_by_origin = _raw_dispositions(source, index_path=index_path)
        origins = sorted(set(source_counts) | set(root_map))
        external_claims: dict[Path, set[str]] = {}
        inventories: dict[str, list[_ExternalGroundTruthFile]] = {}
        declared_roots_by_origin: dict[str, tuple[Path, ...]] = {}
        for origin in origins:
            declared = GROUND_TRUTH_INPUTS.get(origin, {"exempt": False})
            if bool(declared.get("exempt")):
                evidence[origin] = {
                    "exempt": True,
                    "reason": declared.get("reason"),
                    "raw_external_mapping": _raw_external_mapping(source, origin=origin, inventory=(), exempt=True),
                }
                continue
            declared_roots = tuple(Path(path).expanduser().resolve() for path in root_map.get(origin, ()))
            unavailable = [str(path) for path in declared_roots if not path.exists()]
            if not declared_roots or unavailable:
                errors.append(f"ground truth for {origin} is unavailable or undeclared")
                evidence[origin] = {
                    "exempt": False,
                    "declared_roots": [str(path) for path in declared_roots],
                    "unavailable_roots": unavailable,
                    "external_inventory": [],
                    "raw_external_mapping": _raw_external_mapping(source, origin=origin, inventory=(), exempt=False),
                    "passed": False,
                }
                continue
            try:
                inventory = _external_inventory(declared_roots)
            except OSError as exc:
                errors.append(f"ground truth for {origin} could not be fully scanned: {exc}")
                evidence[origin] = {
                    "exempt": False,
                    "declared_roots": [str(path) for path in declared_roots],
                    "external_inventory": [],
                    "raw_external_mapping": _raw_external_mapping(source, origin=origin, inventory=(), exempt=False),
                    "passed": False,
                }
                continue
            inventories[origin] = inventory
            declared_roots_by_origin[origin] = declared_roots
            for item in inventory:
                external_claims.setdefault(item.path, set()).add(origin)

        for path, claimed_origins in sorted(external_claims.items(), key=lambda item: str(item[0])):
            if len(claimed_origins) > 1:
                errors.append(
                    "external ground-truth file is claimed by multiple origins: "
                    f"{path.name} ({', '.join(sorted(claimed_origins))})"
                )

        source_key_origins: dict[GroundTruthKey, set[str]] = {}
        for origin, rows in rows_by_origin.items():
            for row in rows:
                source_key_origins.setdefault(row.key, set()).add(origin)

        for origin in origins:
            declared = GROUND_TRUTH_INPUTS.get(origin, {"exempt": False})
            if bool(declared.get("exempt")) or origin not in inventories:
                continue
            rows = rows_by_origin.get(origin, [])
            inventory = inventories[origin]
            declared_roots = declared_roots_by_origin[origin]
            by_key: dict[GroundTruthKey, list[_ExternalGroundTruthFile]] = {}
            for item in inventory:
                by_key.setdefault(item.key, []).append(item)
            provenance: list[_RawGroundTruthProvenance] = []
            matched_external_paths: set[tuple[int, str]] = set()
            unmatched_raws: list[_UnmatchedSourceRawReceipt] = []
            for row in rows:
                matches = by_key.get(row.key, [])
                match = matches[0] if matches else None
                if match is None:
                    unmatched_raws.append(
                        {
                            "raw_id": row.raw_id,
                            "blob_hash": row.blob_hash,
                            "blob_size": row.blob_size,
                            "disposition": "unreconciled-source-raw",
                        }
                    )
                else:
                    matched_external_paths.add((match.root_index, match.relative_path))
                provenance.append(_raw_provenance(row, match))

            unmatched_external = [
                item for item in inventory if (item.root_index, item.relative_path) not in matched_external_paths
            ]
            unmatched_external_files: list[_ExternalGroundTruthReceipt] = []
            cross_origin_mismatches: list[_ExternalGroundTruthReceipt] = []
            for item in unmatched_external:
                other_origins = sorted(source_key_origins.get(item.key, set()) - {origin})
                if other_origins:
                    receipt = _external_receipt(item, "cross-origin-source", other_origins=other_origins)
                    unmatched_external_files.append(receipt)
                    cross_origin_mismatches.append(receipt)
                else:
                    unmatched_external_files.append(_external_receipt(item, "unmatched-external-file"))
            for item in inventory:
                claimed_origins = external_claims.get(item.path, set()) - {origin}
                if claimed_origins:
                    cross_origin_mismatches.append(
                        _external_receipt(item, "cross-origin-source", other_origins=sorted(claimed_origins))
                    )

            source_keys: set[GroundTruthKey] = {row.key for row in rows}
            source_distinct_bytes = sum(size for _hash, size in source_keys)
            external_bytes = sum(item.size for item in inventory)
            count_discrepancy = len(source_keys) != len(inventory)
            byte_discrepancy = source_distinct_bytes != external_bytes
            origin_errors: list[str] = []
            if unmatched_raws:
                origin_errors.append(f"{len(unmatched_raws)} source raw(s) have no external file match")
            if unmatched_external_files:
                origin_errors.append(f"{len(unmatched_external_files)} external file(s) have no source raw match")
            if count_discrepancy:
                origin_errors.append(
                    f"source distinct blob count {len(source_keys)} disagrees with external file count {len(inventory)}"
                )
            if byte_discrepancy:
                origin_errors.append(
                    f"source distinct blob bytes {source_distinct_bytes} disagrees with external bytes {external_bytes}"
                )
            if cross_origin_mismatches:
                origin_errors.append(f"{len(cross_origin_mismatches)} external file(s) match another origin")
            if origin_errors:
                errors.extend(f"ground truth for {origin}: {reason}" for reason in origin_errors)
            hashes = {item.content_hash for item in inventory}
            missing = sorted(
                source_keys_hash for source_keys_hash, _size in source_keys if source_keys_hash not in hashes
            )
            mapping = _raw_external_mapping(source, origin=origin, inventory=inventory, exempt=False)
            unmatched_mapping = [item for item in mapping if item.get("disposition") == "unmatched-source-raw"]
            if missing:
                errors.append(f"ground truth for {origin} does not verify every source raw blob")
            if unmatched_mapping:
                errors.append(f"ground truth for {origin} has {len(unmatched_mapping)} unmapped source raw(s)")
            evidence[origin] = {
                "exempt": False,
                "declared_roots": [str(path) for path in declared_roots],
                "external_files": len(inventory),
                "external_bytes": external_bytes,
                "external_hashes": len(hashes),
                "source_raw_count": len(rows),
                "source_raw_bytes": sum(row.blob_size for row in rows),
                "source_blob_hashes": len(source_keys),
                "source_blob_bytes": source_distinct_bytes,
                "count_discrepancy": count_discrepancy,
                "byte_discrepancy": byte_discrepancy,
                "unverified_source_blob_hashes": len(missing),
                "unverified_samples": _sample(missing, DEFAULT_SAMPLE_LIMIT),
                "unmatched_source_raws": unmatched_raws[:DEFAULT_SAMPLE_LIMIT],
                "unmatched_external_files": unmatched_external_files[:DEFAULT_SAMPLE_LIMIT],
                "cross_origin_mismatches": cross_origin_mismatches[:DEFAULT_SAMPLE_LIMIT],
                "provenance": provenance,
                "external_inventory": [
                    {
                        "root_index": item.root_index,
                        "relative_path": item.relative_path,
                        "hash": item.content_hash,
                        "size": item.size,
                    }
                    for item in inventory
                ],
                "raw_external_mapping": mapping,
                "passed": not missing and not unmatched_mapping,
            }
    return {
        "passed": not errors,
        "origins": evidence,
        "reasons": errors,
        "external_ground_truth_digest": _canonical_external_ground_truth_digest(evidence),
    }


def _fidelity_evidence(report: object) -> dict[str, object]:
    report_to_json = cast(Any, report).to_json
    payload = dict(report_to_json())
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return {"report": payload, "passed": False, "reasons": ["corpus report has no checks list"]}
    by_name: dict[str, dict[str, object]] = {
        str(check.get("name")): check for check in checks if isinstance(check, dict)
    }
    reasons: list[str] = []
    typed_residuals: list[dict[str, object]] = []

    required_checks = set(CORPUS_FIDELITY_CHECKS)
    missing_checks = sorted(required_checks - by_name.keys())
    if missing_checks:
        reasons.append(f"corpus fidelity checks missing from report: {', '.join(missing_checks)}")
    for name in CORPUS_FIDELITY_CHECKS:
        check = by_name.get(name)
        if check is None:
            continue
        status = check.get("status")
        if status not in {"ok", "error"}:
            reasons.append(f"corpus fidelity check {name} did not run successfully: status={status!r}")

    def required_count(evidence: dict[str, object], key: str, label: str) -> int:
        value = evidence.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            reasons.append(f"corpus fidelity evidence {label}.{key} is missing or invalid")
            return 0
        return value

    def nonnegative_mapping(value: object, label: str) -> dict[str, int]:
        if not isinstance(value, dict):
            reasons.append(f"corpus fidelity evidence {label} is missing or invalid")
            return {}
        normalized: dict[str, int] = {}
        for key, raw_value in value.items():
            if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value < 0:
                reasons.append(f"corpus fidelity evidence {label}.{key} is not a non-negative integer")
            else:
                normalized[str(key)] = raw_value
        return normalized

    absence = by_name.get("corpus-absences", {})
    absence_evidence = _as_dict(absence.get("evidence"))
    absent_total = required_count(absence_evidence, "absent_total", "corpus-absences")
    absent_by_origin = absence_evidence.get("absent_by_origin_cause")
    if not isinstance(absent_by_origin, dict):
        reasons.append("corpus fidelity evidence corpus-absences.absent_by_origin_cause is missing or invalid")
        absent_by_origin = {}
    per_origin_total = 0
    for bucket, raw_count in absent_by_origin.items():
        if not isinstance(raw_count, int) or isinstance(raw_count, bool) or raw_count < 0:
            reasons.append(f"corpus fidelity evidence corpus-absences.{bucket} is not a non-negative integer")
            continue
        count = raw_count
        per_origin_total += count
        if count == 0:
            continue
        origin, _, cause = str(bucket).partition("/")
        explanation = _CAUSE_EXPLANATIONS.get(cause)
        if explanation not in _ALLOWED_RESIDUAL_EXPLANATIONS:
            reasons.append(f"untyped corpus residual {bucket}={count}; bare quarantine is never accepted")
        else:
            typed_residuals.append({"origin": origin, "cause": cause, "count": count, "explanation": explanation})
    if per_origin_total != absent_total:
        reasons.append(f"absence aggregate {absent_total} disagrees with per-origin residual total {per_origin_total}")
    unattributable = required_count(absence_evidence, "raws_without_attributable_identity", "corpus-absences")
    if unattributable:
        reasons.append("corpus residual contains raw rows without attributable identity")
    known_by_origin = nonnegative_mapping(
        absence_evidence.get("documents_known_by_origin"), "corpus-absences.documents_known_by_origin"
    )
    present_by_origin = nonnegative_mapping(
        absence_evidence.get("documents_present_by_origin"), "corpus-absences.documents_present_by_origin"
    )
    documents_known = required_count(absence_evidence, "documents_known", "corpus-absences")
    documents_present = required_count(absence_evidence, "documents_present", "corpus-absences")
    if sum(known_by_origin.values()) != documents_known:
        reasons.append("corpus known-document aggregate disagrees with per-origin denominators")
    if sum(present_by_origin.values()) != documents_present:
        reasons.append("corpus present-document aggregate disagrees with per-origin denominators")
    if documents_present > documents_known or documents_present + absent_total != documents_known:
        reasons.append("corpus known/present/absent document denominators are inconsistent")
    for origin in sorted(set(known_by_origin) | set(present_by_origin)):
        if present_by_origin.get(origin, 0) > known_by_origin.get(origin, 0):
            reasons.append(f"corpus present denominator exceeds known denominator for {origin}")
    if absence.get("status") == "error" and not absent_total and not unattributable:
        reasons.append("corpus-absences reported error without a residual explanation")

    revision = by_name.get("corpus-revision-fidelity", {})
    revision_evidence = _as_dict(revision.get("evidence"))
    unexplained_shortfall = required_count(revision_evidence, "unexplained_shortfall", "corpus-revision-fidelity")
    unexplained_by_origin = revision_evidence.get("unexplained_by_origin")
    worst = revision_evidence.get("worst")
    unexplained_by_origin = nonnegative_mapping(unexplained_by_origin, "corpus-revision-fidelity.unexplained_by_origin")
    if not isinstance(worst, list):
        reasons.append("corpus fidelity evidence corpus-revision-fidelity.worst is missing or invalid")
        worst = []
    if unexplained_shortfall != sum(_int_or_zero(value) for value in unexplained_by_origin.values()):
        reasons.append("revision aggregate disagrees with per-origin residual total")
    if unexplained_shortfall or unexplained_by_origin or worst:
        reasons.append("corpus revision fidelity has an untyped residual")
    if revision.get("status") == "error" and not unexplained_shortfall and not unexplained_by_origin and not worst:
        reasons.append("corpus-revision-fidelity reported error without a residual explanation")

    attachment = by_name.get("corpus-attachment-fidelity", {})
    attachment_evidence = _as_dict(attachment.get("evidence"))
    refs_unfetched = required_count(attachment_evidence, "refs_unfetched", "corpus-attachment-fidelity")
    if refs_unfetched:
        reasons.append("corpus attachment fidelity has unfetched references")
    if attachment.get("status") == "error" and not refs_unfetched:
        reasons.append("corpus-attachment-fidelity reported error without a residual explanation")

    skipped = [name for name, check in by_name.items() if check.get("status") == "skip"]
    if skipped:
        reasons.append(f"corpus fidelity checks skipped: {', '.join(sorted(skipped))}")
    return {
        "report": payload,
        "typed_residuals": typed_residuals,
        "passed": not reasons,
        "reasons": reasons,
        "denominators": {
            "documents_known": documents_known,
            "documents_present": documents_present,
            "documents_known_by_origin": known_by_origin,
            "documents_present_by_origin": present_by_origin,
            "revision_origins": sorted(str(key) for key in _as_dict(revision_evidence.get("explained_by_origin"))),
        },
    }


def _as_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value) if isinstance(value, dict) else {}


def _int_or_zero(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _current_external_ground_truth_digest(archive_root: Path, ground_truth: Mapping[str, object]) -> str:
    origins = _as_dict(ground_truth.get("origins"))
    current: dict[str, object] = {}
    with open_readonly_connection(archive_root / "source.db") as source:
        for origin, raw_evidence in origins.items():
            evidence = _as_dict(raw_evidence)
            if bool(evidence.get("exempt")):
                current[origin] = {
                    "exempt": True,
                    "reason": evidence.get("reason"),
                    "raw_external_mapping": _raw_external_mapping(source, origin=origin, inventory=(), exempt=True),
                }
                continue
            roots = evidence.get("declared_roots")
            if not isinstance(roots, list) or not all(isinstance(root, str) and root for root in roots):
                raise SchemaInferenceGateError(f"ground truth roots for {origin} are missing or malformed")
            resolved_roots = tuple(Path(root).expanduser().resolve() for root in roots)
            unavailable = [str(root) for root in resolved_roots if not root.exists()]
            if unavailable:
                raise SchemaInferenceGateError(
                    f"ground truth roots for {origin} are unavailable: {', '.join(unavailable)}"
                )
            inventory = _external_inventory(resolved_roots)
            current[origin] = {
                "declared_roots": [str(root) for root in resolved_roots],
                "external_inventory": inventory,
                "raw_external_mapping": _raw_external_mapping(source, origin=origin, inventory=inventory, exempt=False),
            }
    return _canonical_external_ground_truth_digest(current)


def validate_schema_inference_receipt(
    archive_root: Path,
    receipt_path: Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate the evidence a rebuild is authorized to consume.

    This validator deliberately checks the embedded subgates instead of
    trusting the top-level verdict. It also recomputes archive/source identity,
    the source snapshot, and the external corpus digest against the live
    read-only inputs.
    """

    root = Path(archive_root).absolute()
    path = resolve_schema_inference_receipt_reference(root, receipt_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaInferenceGateError(f"could not read schema-inference receipt: {exc}") from exc
    if not isinstance(document, dict):
        raise SchemaInferenceGateError("schema-inference receipt root must be an object")

    errors: list[str] = []
    if document.get("schema") != RECEIPT_SCHEMA:
        errors.append(f"schema must be {RECEIPT_SCHEMA!r}")
    if document.get("verdict") != "PASS":
        errors.append("receipt verdict must be PASS")
    if document.get("archive_root") != str(root):
        errors.append("receipt archive_root does not match the requested archive")

    generated_at: datetime | None = None
    try:
        generated_at = _parse_receipt_datetime(document.get("generated_at"), field="generated_at")
    except SchemaInferenceGateError as exc:
        errors.append(str(exc))
    effective_now = (now or datetime.now(UTC)).astimezone(UTC)
    if generated_at is not None:
        if generated_at > effective_now + RECEIPT_CLOCK_SKEW:
            errors.append("receipt generated_at is in the future")
        if effective_now - generated_at > RECEIPT_TTL + RECEIPT_CLOCK_SKEW:
            errors.append("schema-inference receipt is stale")

    try:
        location = ArchiveLocation.resolve(root)
        current_identity = _archive_receipt_identity(location)
        receipt_identity = _as_dict(document.get("archive_identity"))
        for field in ("configured_root", "durable_id", "source_tier", "user_tier"):
            if receipt_identity.get(field) != current_identity.get(field):
                errors.append(f"receipt archive identity field {field!r} does not match the requested archive")
        source_path = root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
        if not source_path.exists():
            errors.append("source.db is missing")
        else:
            from polylogue.storage.index_generation import source_revision_snapshot

            expected_snapshot = document.get("source_snapshot")
            actual_snapshot = source_revision_snapshot(root)
            if not isinstance(expected_snapshot, str) or expected_snapshot != actual_snapshot:
                errors.append("receipt source snapshot does not match source.db")
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        errors.append(f"archive/source identity validation failed: {exc}")

    source_identity = _as_dict(document.get("source_identity"))
    if source_identity.get("durable_id") != _as_dict(document.get("archive_identity")).get("durable_id"):
        errors.append("receipt source identity is missing or does not match durable identity")
    if source_identity.get("source_tier") != _as_dict(document.get("archive_identity")).get("source_tier"):
        errors.append("receipt source identity is missing or does not match source tier identity")
    if _as_dict(document.get("source_schema_identity")).get("matches_expected") is not True:
        errors.append("receipt source schema identity is not PASS")

    query_results = document.get("query_results")
    required_gates = (*_HARD_GATE_SQL, "zero-unexplained-byte-duplicates")
    if not isinstance(query_results, dict):
        errors.append("receipt query_results are missing or malformed")
    else:
        for gate_id in required_gates:
            result = query_results.get(gate_id)
            if not isinstance(result, dict) or result.get("passed") is not True:
                errors.append(f"receipt subgate {gate_id} is not PASS")

    for field in ("corpus_fidelity", "full_blob_hash_verification", "ground_truth_inputs"):
        if _as_dict(document.get(field)).get("passed") is not True:
            errors.append(f"receipt subgate {field} is not PASS")

    ground_truth = document.get("ground_truth_inputs")
    if not isinstance(ground_truth, dict):
        errors.append("receipt ground_truth_inputs are missing or malformed")
    else:
        recorded_digest = document.get("external_ground_truth_digest")
        nested_digest = ground_truth.get("external_ground_truth_digest")
        if not isinstance(recorded_digest, str) or recorded_digest != nested_digest:
            errors.append("receipt external ground-truth digest is missing or inconsistent")
        try:
            recorded_structure_digest = _canonical_external_ground_truth_digest(
                cast(Mapping[str, object], _as_dict(ground_truth.get("origins")))
            )
            if recorded_digest != recorded_structure_digest:
                errors.append("receipt raw external mapping or inventory does not match its digest")
            current_digest = _current_external_ground_truth_digest(root, ground_truth)
            if recorded_digest != current_digest:
                errors.append("external ground-truth corpus changed since the receipt was produced")
        except (OSError, SchemaInferenceGateError, sqlite3.Error) as exc:
            errors.append(str(exc))
        try:
            with open_readonly_connection(root / "source.db") as source:
                source_origins = {str(row[0]) for row in source.execute("SELECT DISTINCT origin FROM raw_sessions")}
            receipt_origins = set(_as_dict(ground_truth.get("origins")))
            if source_origins != receipt_origins:
                errors.append("receipt ground-truth origins do not match source.db")
            for origin, raw_evidence in _as_dict(ground_truth.get("origins")).items():
                evidence = _as_dict(raw_evidence)
                if not bool(evidence.get("exempt")) and evidence.get("passed") is not True:
                    errors.append(f"receipt ground-truth subgate for {origin} is not PASS")
        except (OSError, sqlite3.Error) as exc:
            errors.append(f"could not compare receipt ground-truth origins: {exc}")

    if errors:
        raise SchemaInferenceGateError("schema-inference receipt rejected: " + "; ".join(errors))

    identity = _as_dict(document.get("archive_identity"))
    return {
        "receipt_path": str(path),
        "schema": document.get("schema"),
        "generated_at": document.get("generated_at"),
        "validated_at": effective_now.isoformat(),
        "archive_root": str(root),
        "durable_id": identity.get("durable_id"),
        "source_identity": document.get("source_identity"),
        "source_snapshot": document.get("source_snapshot"),
        "external_ground_truth_digest": document.get("external_ground_truth_digest"),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_schema_inference_gate(
    archive_root: Path,
    *,
    receipt_path: Path,
    ground_truth_roots: Mapping[str, Sequence[Path]] | None = None,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> SchemaInferenceGateResult:
    """Run and persist the schema-inference prerequisite receipt."""

    if sample_limit <= 0:
        raise SchemaInferenceGateError("sample_limit must be positive")
    root = Path(archive_root).absolute()
    safe_receipt_path = _resolve_receipt_path(Path(receipt_path), archive_root=root)
    try:
        location = ArchiveLocation.resolve(root)
        index_path = location.active_index_path
        schema_identity = _tier_schema_identity(root, location)
        archive_receipt_identity = _archive_receipt_identity(location)
    except (OSError, ValueError, RuntimeError) as exc:
        schema_identity = {"error": str(exc), "tiers": {}}
        index_path = root / "index.db"
        archive_receipt_identity = {"error": str(exc)}

    source_entry = _as_dict(_as_dict(schema_identity.get("tiers")).get("source"))
    source_schema_ok = bool(source_entry.get("matches_expected"))

    try:
        source_gates = _run_source_gates(root, index_path=index_path, sample_limit=sample_limit)
    except (OSError, sqlite3.Error) as exc:
        source_gates = _failed_source_gates(f"source-tier read failed: {exc}")
    blob_denominators = _as_dict(source_gates.get("blob_denominators"))
    try:
        corpus_report = verify_archive(root, checks=CORPUS_FIDELITY_CHECKS, sample_limit=sample_limit)
        fidelity = _fidelity_evidence(corpus_report)
    except Exception as exc:
        fidelity = {"report": {}, "typed_residuals": [], "passed": False, "reasons": [f"corpus audit raised: {exc}"]}
    try:
        with open_readonly_connection(root / "source.db") as source:
            referenced_hashes = _referenced_blob_hashes(source)
        full_blob_hash_verification = _full_blob_hash_evidence(root, referenced_hashes=referenced_hashes)
    except (OSError, sqlite3.Error) as exc:
        full_blob_hash_verification = {
            "passed": False,
            "errors": [f"full BlobStore verification could not read source evidence: {exc}"],
            "reason": f"full BlobStore verification could not read source evidence: {exc}",
        }
    try:
        ground_truth = _ground_truth_evidence(
            root,
            index_path=index_path,
            source_counts=cast(Mapping[str, Mapping[str, int]], source_gates.get("source_counts", {})),
            roots=ground_truth_roots,
        )
    except (OSError, sqlite3.Error) as exc:
        ground_truth = {
            "passed": False,
            "origins": {},
            "reasons": [f"ground truth reconciliation could not read source evidence: {exc}"],
            "external_ground_truth_digest": None,
        }

    source_snapshot: str | None = None
    reasons_for_snapshot: str | None = None
    try:
        from polylogue.storage.index_generation import source_revision_snapshot

        source_snapshot = source_revision_snapshot(root)
    except (OSError, sqlite3.Error) as exc:
        reasons_for_snapshot = f"source revision snapshot could not be read: {exc}"
    gate_results = _as_dict(source_gates.get("gates"))
    duplicate_gate = source_gates.get("duplicate_gate", {})
    if isinstance(duplicate_gate, dict):
        gate_results["zero-unexplained-byte-duplicates"] = duplicate_gate
    passed_hard_gates = all(bool(_as_dict(result).get("passed")) for result in gate_results.values())
    reasons = [
        str(_as_dict(result).get("reason")) for result in gate_results.values() if _as_dict(result).get("reason")
    ]
    if not source_schema_ok:
        reasons.append("source.db schema identity is missing or does not match the packaged schema")
    if not bool(fidelity.get("passed")):
        fidelity_reasons = fidelity.get("reasons", [])
        if isinstance(fidelity_reasons, list):
            reasons.extend(str(reason) for reason in fidelity_reasons)
    if not bool(full_blob_hash_verification.get("passed")):
        reasons.append(str(full_blob_hash_verification.get("reason") or "full BlobStore verification is not a PASS"))
    if not bool(ground_truth.get("passed")):
        ground_truth_reasons = ground_truth.get("reasons", [])
        if isinstance(ground_truth_reasons, list):
            reasons.extend(str(reason) for reason in ground_truth_reasons)
    if source_snapshot is None:
        reasons.append(reasons_for_snapshot or "source revision snapshot could not be computed")

    payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "gate_version": GATE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if not reasons and passed_hard_gates else "FAIL",
        "archive_root": str(root),
        "archive_identity": archive_receipt_identity,
        "source_identity": {
            "durable_id": archive_receipt_identity.get("durable_id"),
            "source_tier": archive_receipt_identity.get("source_tier"),
        },
        "source_snapshot": source_snapshot,
        "external_ground_truth_digest": ground_truth.get("external_ground_truth_digest"),
        "schema_identity": schema_identity,
        "source_schema_identity": source_entry,
        "query_results": gate_results,
        "source_denominators": source_gates.get("source_counts", {}),
        "blob_denominators": blob_denominators,
        "ground_truth_denominators": fidelity.get("denominators", {}),
        "ground_truth_inputs": ground_truth,
        "exemptions": {name: value for name, value in GROUND_TRUTH_INPUTS.items() if bool(value.get("exempt"))},
        "corpus_fidelity": fidelity,
        "full_blob_hash_verification": full_blob_hash_verification,
        "input_paths": {
            "archive_root": str(root),
            "source_db": str(root / "source.db"),
            "active_index_db": str(index_path),
            "receipt": str(safe_receipt_path),
        },
        "tool_versions": {
            "polylogue": POLYLOGUE_VERSION,
            "gate": GATE_VERSION,
            "python": platform.python_version(),
            "executable": sys.executable,
            "corpus_audit": "polylogue.maintenance.corpus_fidelity via archive verification registry",
        },
        "pass_fail_reasons": reasons,
    }
    _write_json(safe_receipt_path, payload)
    return SchemaInferenceGateResult(payload)


__all__ = [
    "DEFAULT_SAMPLE_LIMIT",
    "GROUND_TRUTH_INPUTS",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "RECEIPT_CLOCK_SKEW",
    "RECEIPT_TTL",
    "SCHEMA_INFERENCE_RECEIPT_ENV",
    "SchemaInferenceGateError",
    "SchemaInferenceGateResult",
    "resolve_schema_inference_receipt_reference",
    "run_schema_inference_gate",
    "validate_schema_inference_receipt",
]
