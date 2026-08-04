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
import platform
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from polylogue.maintenance.archive_verification import CORPUS_FIDELITY_CHECKS, verify_archive
from polylogue.storage.archive_identity import ArchiveIdentity, ArchiveLocation
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.introspection import table_exists
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_readonly_connection
from polylogue.version import POLYLOGUE_VERSION

RECEIPT_SCHEMA = "polylogue.schema-inference-gate.v1"
GATE_VERSION = "2"
DEFAULT_SAMPLE_LIMIT = 10
RECEIPT_FILENAME = "schema-inference-gate-receipt.json"

_ALLOWED_RESIDUAL_EXPLANATIONS = frozenset(
    {"materialized", "superseded-duplicate", "legitimately-excluded-non-conversation"}
)
_CAUSE_EXPLANATIONS = {"byte-revision-governed": "superseded-duplicate"}

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


def _ground_truth_evidence(
    archive_root: Path,
    *,
    source_counts: Mapping[str, Mapping[str, int]],
    roots: Mapping[str, Sequence[Path]] | None,
) -> dict[str, object]:
    """Compare source raw blobs with the files actually present under declared roots."""

    root_map = roots or {}
    source_path = archive_root / ARCHIVE_TIER_SPECS[ArchiveTier.SOURCE].filename
    evidence: dict[str, object] = {}
    errors: list[str] = []
    with open_readonly_connection(source_path) as source:
        for origin in sorted(source_counts):
            declared = GROUND_TRUTH_INPUTS.get(origin, {"exempt": False})
            if bool(declared.get("exempt")):
                evidence[origin] = {"exempt": True, "reason": declared.get("reason")}
                continue
            declared_roots = tuple(Path(path).expanduser().resolve() for path in root_map.get(origin, ()))
            unavailable = [str(path) for path in declared_roots if not path.exists()]
            if not declared_roots or unavailable:
                errors.append(f"ground truth for {origin} is unavailable or undeclared")
                evidence[origin] = {
                    "exempt": False,
                    "declared_roots": [str(path) for path in declared_roots],
                    "unavailable_roots": unavailable,
                    "passed": False,
                }
                continue
            files = [path for root in declared_roots for path in _iter_ground_truth_files(root)]
            hashes: set[str] = set()
            bytes_total = 0
            try:
                for path in files:
                    digest, size = _file_hash(path)
                    hashes.add(digest)
                    bytes_total += size
            except OSError as exc:
                errors.append(f"ground truth for {origin} could not be fully scanned: {exc}")
                evidence[origin] = {
                    "exempt": False,
                    "declared_roots": [str(path) for path in declared_roots],
                    "passed": False,
                }
                continue
            source_hashes = {
                bytes(row[0]).hex()
                for row in source.execute("SELECT DISTINCT blob_hash FROM raw_sessions WHERE origin = ?", (origin,))
            }
            missing = sorted(source_hashes - hashes)
            if missing:
                errors.append(f"ground truth for {origin} does not verify every source raw blob")
            evidence[origin] = {
                "exempt": False,
                "declared_roots": [str(path) for path in declared_roots],
                "external_files": len(files),
                "external_bytes": bytes_total,
                "external_hashes": len(hashes),
                "source_blob_hashes": len(source_hashes),
                "unverified_source_blob_hashes": len(missing),
                "unverified_samples": _sample(missing, DEFAULT_SAMPLE_LIMIT),
                "passed": not missing,
            }
    return {"passed": not errors, "origins": evidence, "reasons": errors}


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
    except (OSError, ValueError, RuntimeError) as exc:
        schema_identity = {"error": str(exc), "tiers": {}}
        index_path = root / "index.db"

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
            source_counts=cast(Mapping[str, Mapping[str, int]], source_gates.get("source_counts", {})),
            roots=ground_truth_roots,
        )
    except (OSError, sqlite3.Error) as exc:
        ground_truth = {
            "passed": False,
            "origins": {},
            "reasons": [f"ground truth reconciliation could not read source evidence: {exc}"],
        }

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

    payload: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "gate_version": GATE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "verdict": "PASS" if not reasons and passed_hard_gates else "FAIL",
        "archive_root": str(root),
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
    "SchemaInferenceGateError",
    "SchemaInferenceGateResult",
    "run_schema_inference_gate",
]
