"""Durable, provider-neutral admission of linked materials.

Materials are observations, not sessions.  The record is committed even when
acquisition fails, while successfully obtained bytes are published to the
existing content-addressed store before the observation is committed.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import sqlite3
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from polylogue.storage.blob_store import BlobStore, get_blob_store

MaterialState = Literal[
    "claimed",
    "acquired",
    "duplicate",
    "partial",
    "unavailable",
    "expired",
    "access_denied",
    "malformed",
    "superseded",
]
MaterialPrivacy = Literal["private", "restricted", "public", "synthetic"]
MaterialRelation = Literal["refers_to", "acquired_from", "supports", "affected"]


@dataclass(frozen=True, slots=True)
class MaterialObservation:
    material_id: str
    referrer_ref: str
    source_uri: str
    acquisition_state: MaterialState
    diagnostic: str
    retryable: bool
    blob_hash: str | None
    byte_size: int | None
    media_type: str | None
    media_charset: str | None
    filename: str | None
    extraction_manifest: dict[str, object]
    custody: Literal["claimed", "retained", "verified", "released"]
    privacy_classification: MaterialPrivacy
    acquired_at_ms: int
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class MaterialEvidenceLink:
    evidence_ref: str
    relation: MaterialRelation
    authority: Literal["provider", "operator", "repository", "inferred", "unknown"]
    confidence: float
    observed_at_ms: int
    source_diagnostic: str


def _material_id(source_uri: str, referrer_ref: str, payload: bytes | None) -> str:
    digest = hashlib.sha256()
    digest.update(source_uri.encode("utf-8"))
    digest.update(b"\0")
    digest.update(referrer_ref.encode("utf-8"))
    digest.update(b"\0")
    if payload is not None:
        digest.update(payload)
    return "material:" + digest.hexdigest()


def extraction_manifest(payload: bytes, media_type: str | None) -> dict[str, object]:
    """Return bounded, type-aware metadata without treating parsing as required."""
    manifest: dict[str, object] = {"bytes": len(payload), "extractor": "materials-v1", "entries": []}
    kind = (media_type or mimetypes.guess_type("material")[0] or "").lower()
    if kind in {"application/json", "application/ndjson", "text/json"}:
        try:
            value = json.loads(payload[:2_000_000].decode("utf-8"))
            manifest["json_type"] = type(value).__name__
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            manifest["diagnostic"] = f"json extraction failed: {type(exc).__name__}: {exc}"
    elif kind in {"application/zip", "application/x-zip-compressed"}:
        try:
            with zipfile.ZipFile(BytesIO(payload)) as archive:
                manifest["entries"] = [info.filename for info in archive.infolist()[:1000]]
                manifest["entry_count"] = len(archive.infolist())
        except (OSError, zipfile.BadZipFile) as exc:
            manifest["diagnostic"] = f"zip extraction failed: {type(exc).__name__}: {exc}"
    elif kind.startswith("text/") or not kind:
        # Keep the manifest safe to expose through query surfaces. Raw text is
        # available from the CAS blob; it must not be copied into indexable
        # metadata or synthetic/public fixtures by default.
        manifest["text"] = {"available": True, "encoding": media_type or "unknown"}
    return manifest


def admit_material(
    conn: sqlite3.Connection,
    *,
    blob_store: BlobStore | None,
    source_uri: str,
    referrer_ref: str,
    observed_at_ms: int,
    payload: bytes | None = None,
    media_type: str | None = None,
    media_charset: str | None = None,
    filename: str | None = None,
    state: MaterialState | None = None,
    diagnostic: str = "",
    retryable: bool = False,
    privacy_classification: MaterialPrivacy = "private",
    supersedes_material_id: str | None = None,
) -> MaterialObservation:
    """Record one claim/acquisition, publishing bytes before durable linkage."""
    if not source_uri.strip() or not referrer_ref.strip():
        raise ValueError("source_uri and referrer_ref are required")
    if privacy_classification == "synthetic" and payload is not None:
        raise ValueError("synthetic materials cannot carry arbitrary raw bytes")
    material_state: MaterialState = state or ("acquired" if payload is not None else "claimed")
    if payload is not None:
        resolved_blob_store = blob_store or get_blob_store()
        blob_hash, byte_size = resolved_blob_store.write_from_bytes(payload)
        custody: Literal["claimed", "retained", "verified", "released"] = "retained"
        media_type = media_type or mimetypes.guess_type(filename or "")[0]
        manifest = extraction_manifest(payload, media_type)
        if manifest.get("diagnostic") and material_state == "acquired":
            material_state = "malformed"
    else:
        blob_hash, byte_size, manifest, custody = (
            None,
            None,
            {"bytes": None, "extractor": "materials-v1"},
            "claimed",
        )
    material_id = _material_id(source_uri, referrer_ref, payload)
    if blob_hash is not None:
        duplicate = (
            conn.execute(
                "SELECT 1 FROM material_observations WHERE blob_hash = ? LIMIT 1",
                (bytes.fromhex(blob_hash),),
            ).fetchone()
            is not None
        )
        if duplicate and state is None:
            material_state = "duplicate"
    now = observed_at_ms
    conn.execute(
        """INSERT INTO material_observations
        (material_id, referrer_ref, source_uri, acquisition_state, diagnostic,
         retryable, supersedes_material_id, blob_hash, byte_size, media_type,
         media_charset, filename, extraction_manifest_json, custody,
         privacy_classification, acquired_at_ms, created_at_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(material_id) DO UPDATE SET
          acquisition_state=excluded.acquisition_state, diagnostic=excluded.diagnostic,
          retryable=excluded.retryable, blob_hash=excluded.blob_hash,
          byte_size=excluded.byte_size, extraction_manifest_json=excluded.extraction_manifest_json,
          custody=excluded.custody, acquired_at_ms=excluded.acquired_at_ms""",
        (
            material_id,
            referrer_ref,
            source_uri,
            material_state,
            diagnostic[:4096],
            int(retryable),
            supersedes_material_id,
            bytes.fromhex(blob_hash) if blob_hash else None,
            byte_size,
            media_type,
            media_charset,
            filename,
            json.dumps(manifest, sort_keys=True),
            custody,
            privacy_classification,
            observed_at_ms,
            now,
        ),
    )
    conn.commit()
    return MaterialObservation(
        material_id,
        referrer_ref,
        source_uri,
        material_state,
        diagnostic[:4096],
        retryable,
        blob_hash,
        byte_size,
        media_type,
        media_charset,
        filename,
        manifest,
        custody,
        privacy_classification,
        observed_at_ms,
        now,
    )


def link_material(
    conn: sqlite3.Connection,
    material_id: str,
    evidence_ref: str,
    *,
    relation: MaterialRelation,
    authority: Literal["provider", "operator", "repository", "inferred", "unknown"] = "unknown",
    confidence: float = 1.0,
    observed_at_ms: int,
    source_diagnostic: str = "",
) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    conn.execute(
        """INSERT INTO material_evidence_links
      (material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic)
      VALUES (?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(material_id, evidence_ref, relation) DO UPDATE SET
        authority=excluded.authority, confidence=excluded.confidence,
        observed_at_ms=excluded.observed_at_ms, source_diagnostic=excluded.source_diagnostic""",
        (material_id, evidence_ref, relation, authority, confidence, observed_at_ms, source_diagnostic[:4096]),
    )
    conn.commit()


def get_material(conn: sqlite3.Connection, material_id: str) -> MaterialObservation | None:
    cursor = conn.execute("SELECT * FROM material_observations WHERE material_id = ?", (material_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [column[0] for column in cursor.description or ()]
    row = dict(zip(columns, row, strict=True))
    return MaterialObservation(
        material_id=row["material_id"],
        referrer_ref=row["referrer_ref"],
        source_uri=row["source_uri"],
        acquisition_state=row["acquisition_state"],
        diagnostic=row["diagnostic"],
        retryable=bool(row["retryable"]),
        blob_hash=bytes(row["blob_hash"]).hex() if row["blob_hash"] is not None else None,
        byte_size=row["byte_size"],
        media_type=row["media_type"],
        media_charset=row["media_charset"],
        filename=row["filename"],
        extraction_manifest=json.loads(row["extraction_manifest_json"]),
        custody=row["custody"],
        privacy_classification=row["privacy_classification"],
        acquired_at_ms=row["acquired_at_ms"],
        created_at_ms=row["created_at_ms"],
    )


def read_material(conn: sqlite3.Connection, material_id: str, *, blob_store: BlobStore | None = None) -> bytes:
    """Read retained bytes for one material, preserving claim-only absence."""
    observation = get_material(conn, material_id)
    if observation is None:
        raise KeyError(material_id)
    if observation.blob_hash is None:
        raise FileNotFoundError(f"material {material_id!r} has no retained bytes")
    return (blob_store or get_blob_store()).read_all(observation.blob_hash)


def list_materials(conn: sqlite3.Connection, *, evidence_ref: str | None = None) -> list[MaterialObservation]:
    """List truthful observations, optionally through a direct evidence link."""
    if evidence_ref is None:
        rows = conn.execute("SELECT * FROM material_observations ORDER BY created_at_ms, material_id").fetchall()
    else:
        rows = conn.execute(
            "SELECT m.* FROM material_observations m JOIN material_evidence_links l USING(material_id) "
            "WHERE l.evidence_ref = ? ORDER BY m.created_at_ms, m.material_id",
            (evidence_ref,),
        ).fetchall()
    observations: list[MaterialObservation] = []
    for row in rows:
        observation = get_material(conn, row["material_id"])
        if observation is not None:
            observations.append(observation)
    return observations


__all__ = [
    "MaterialObservation",
    "admit_material",
    "extraction_manifest",
    "get_material",
    "link_material",
    "list_materials",
    "read_material",
    "MaterialEvidenceLink",
]
