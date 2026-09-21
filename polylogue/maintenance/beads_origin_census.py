"""Read-only census and exact removal plan for the retired Beads origin."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.config import ResolvedRuntimeConfig
from polylogue.storage.sqlite.managed_connection import sqlite_connection

FORMAT = "polylogue.beads-origin-census.v1"
PLAN_FORMAT = "polylogue.beads-origin-removal-plan.v1"
ORIGIN = "beads-issue"


class BeadsOriginCensusError(RuntimeError):
    """The census could not produce a trustworthy result."""


def _json_value(value: object) -> object:
    return value.hex() if isinstance(value, (bytes, bytearray, memoryview)) else value


@dataclass(frozen=True, slots=True)
class SurfaceResult:
    name: str
    path: str
    state: str
    files: tuple[dict[str, Any], ...] = ()
    rows: dict[str, int] | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "state": self.state,
            "files": list(self.files),
            "rows": self.rows,
            "evidence": list(self.evidence),
            "error": self.error,
        }


def _file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve(strict=False)),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _artifact_files(root: Path) -> tuple[dict[str, Any], ...]:
    candidates: tuple[Path, ...]
    if root.is_file():
        candidates = (root,) if root.name in {"interactions.jsonl", "issues.jsonl"} else ()
    elif root.is_dir():
        candidates = tuple(
            sorted(
                (p for p in root.rglob("*") if p.is_file() and p.name in {"interactions.jsonl", "issues.jsonl"}),
                key=str,
            )
        )
    else:
        return ()
    return tuple(_file_record(path) for path in candidates)


def _configured_roots(runtime: ResolvedRuntimeConfig) -> tuple[tuple[str, Path], ...]:
    roots: list[tuple[str, Path]] = [("archive", runtime.paths.archive_root)]
    roots.extend((f"configured:{source.name}", source.path) for source in runtime.sources if source.path is not None)
    roots.extend((f"source:{path}", path) for path in runtime.source_paths.explicit)
    seen: set[Path] = set()
    result: list[tuple[str, Path]] = []
    for name, path in roots:
        resolved = path.resolve(strict=False)
        if resolved not in seen:
            seen.add(resolved)
            result.append((name, resolved))
    return tuple(result)


def _db_rows(
    path: Path, queries: dict[str, str], evidence_query: str
) -> tuple[str, dict[str, int] | None, tuple[dict[str, Any], ...], str | None]:
    if not path.is_file():
        return "unavailable", None, (), f"missing database: {path}"
    try:
        with sqlite_connection(f"file:{path}?mode=ro", uri=True, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            rows = {name: int(conn.execute(sql).fetchone()[0]) for name, sql in queries.items()}
            evidence = tuple(
                {str(key): _json_value(value) for key, value in zip(row.keys(), row, strict=True)}
                for row in conn.execute(evidence_query).fetchall()
            )
    except (OSError, sqlite3.Error) as exc:
        return "failed", None, (), f"{type(exc).__name__}: {exc}"
    return ("populated" if any(rows.values()) else "zero"), rows, evidence, None


def census_beads_origin(runtime: ResolvedRuntimeConfig) -> dict[str, Any]:
    """Inspect all configured archive and raw roots without opening a writer."""
    surfaces: list[dict[str, Any]] = []
    for name, root in _configured_roots(runtime):
        if not root.exists():
            surfaces.append(SurfaceResult(name, str(root), "unavailable", error="missing configured root").to_dict())
            continue
        try:
            # The archive root is a database/blob file set, not a raw artifact
            # tree. Avoid walking its potentially large blob store.
            files = () if name == "archive" else _artifact_files(root)
        except OSError as exc:
            surfaces.append(SurfaceResult(name, str(root), "failed", error=f"{type(exc).__name__}: {exc}").to_dict())
            continue
        surfaces.append(SurfaceResult(name, str(root), "populated" if files else "zero", files=files).to_dict())

    archive = runtime.paths.archive_root
    db_surfaces = (
        (
            "source.db",
            archive / "source.db",
            {
                "raw_sessions": "SELECT COUNT(*) FROM raw_sessions WHERE origin = 'beads-issue'",
                "raw_artifacts": "SELECT COUNT(*) FROM raw_artifacts WHERE origin = 'beads-issue'",
            },
            "SELECT 'raw_session' AS evidence_kind, raw_id, origin, source_path, blob_hash, blob_size "
            "FROM raw_sessions WHERE origin = 'beads-issue' "
            "UNION ALL SELECT 'raw_artifact', raw_id, origin, source_path, NULL, NULL "
            "FROM raw_artifacts WHERE origin = 'beads-issue' ORDER BY raw_id",
        ),
        (
            "index.db",
            runtime.paths.index_db,
            {
                "sessions": "SELECT COUNT(*) FROM sessions WHERE origin = 'beads-issue'",
                "session_links": "SELECT COUNT(*) FROM session_links WHERE dst_origin = 'beads-issue'",
            },
            "SELECT 'session' AS evidence_kind, session_id, origin, native_id, raw_id "
            "FROM sessions WHERE origin = 'beads-issue' ORDER BY session_id",
        ),
    )
    for name, path, queries, evidence_query in db_surfaces:
        state, rows, evidence, error = _db_rows(path, queries, evidence_query)
        surfaces.append(SurfaceResult(name, str(path), state, rows=rows, evidence=evidence, error=error).to_dict())

    affected_tables = {
        "source.db": ["raw_sessions.origin", "raw_artifacts.origin"],
        "index.db": ["sessions.origin", "session_links.dst_origin", "derived FTS/insight projections"],
    }
    # Where the token's admission actually lives, and the asymmetry that
    # decides the removal route. The DURABLE source tier carries no
    # enum-generated ``CHECK(origin IN (...))`` -- only ``length(trim(origin))
    # > 0`` -- because vocabulary membership there is validated at the write
    # boundary by ``require_vocabulary``. The DERIVED index tier does generate
    # its CHECK from the Origin enum, so the token disappears from that DDL
    # when the tier is re-created, never by a migration against a live file.
    constraints = {
        "source.raw_sessions.origin": (
            "durable: no enum CHECK, only a non-empty check; admission is the Origin "
            "vocabulary at the write boundary (require_vocabulary)"
        ),
        "source.raw_artifacts.origin": (
            "durable: no enum CHECK, only a non-empty check; admission is the Origin "
            "vocabulary at the write boundary (require_vocabulary)"
        ),
        "index.sessions.origin": (
            "derived: enum-generated CHECK(origin IN (...)) currently admits the token; "
            "it narrows when the tier is re-created from the narrowed enum"
        ),
        "index.session_links.dst_origin": (
            "derived: enum-generated CHECK(dst_origin IN (...)) currently admits the token; "
            "it narrows when the tier is re-created from the narrowed enum"
        ),
    }
    # A different vocabulary that happens to spell the same string. These are
    # work-effect object kinds on a live route (a ``.beads/interactions.jsonl``
    # ledger becomes work-evidence nodes), not session origins. Removing the
    # retired Origin token must not touch any of them.
    non_targets = {
        "polylogue/core/refs.py": "ObjectRef kind 'beads-issue' in the work-evidence ref vocabulary",
        "polylogue/analysis/work_evidence.py": "'beads-issue' in the effect-node kind vocabulary",
        "polylogue/analysis/work_effects.py": "live ObjectRef(kind='beads-issue') minted from a Beads ledger",
    }
    plan = {
        "format": PLAN_FORMAT,
        "action": "remove-retired-origin-vocabulary",
        "preconditions": [
            "fresh read-only census receipt is populated or explicitly reviewed for every surface",
            "verified backup manifest covers source.db and required durable blobs",
            "daemon is stopped and archive ownership is acquired",
            "operator authorization is bound to this plan digest",
        ],
        "steps": [
            "confirm this census reports zero rows on every database surface, with no surface left unavailable",
            "delete the Origin/Provider token and its source and origin-spec mappings; leave the non_targets alone",
            "leave the durable source tier alone: its origin columns carry no enum CHECK to narrow",
            "accept that the derived schema identity moves, because the Origin enum is inside its closure",
            "reconverge index.db through the production daemon, which re-creates the narrowed derived CHECK",
            "recheck affected counts and fresh DDL, then emit an immutable apply receipt",
        ],
        "idempotence": "exact-plan-bound and resumable by generation; never rediscover targets",
        "no_apply_in_this_operation": True,
        "no_durable_migration_required": True,
        "affected_tables": affected_tables,
        "constraints": constraints,
        "non_targets": non_targets,
        "derived_rebuild": "index.db FTS, lineage links, and materialized insights reconverge from durable evidence",
    }
    payload: dict[str, Any] = {
        "format": FORMAT,
        "created_at_ms": int(time.time() * 1000),
        "archive_root": str(archive.resolve(strict=False)),
        "origin": ORIGIN,
        "surfaces": surfaces,
        "affected_tables": affected_tables,
        "constraints": constraints,
        "plan": plan,
        "production_mutation_performed": False,
    }
    plan["census_digest"] = _digest({"surfaces": surfaces, "affected_tables": affected_tables})
    payload["plan_digest"] = _digest(plan)
    return payload


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_census_receipt(runtime: ResolvedRuntimeConfig, receipt_path: Path) -> dict[str, Any]:
    """Write one immutable census receipt; refuse to overwrite an existing one."""
    if receipt_path.exists():
        raise BeadsOriginCensusError(f"receipt already exists and is immutable: {receipt_path}")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = census_beads_origin(runtime)
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    receipt_path.chmod(0o444)
    return payload
