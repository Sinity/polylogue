"""Utilities for storing and retrieving raw imports in the database."""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

from ..db import get_raw_import, list_failed_imports, open_connection, upsert_raw_import


# Current parser version - increment when making breaking changes
PARSER_VERSION = "0.2.0"


def compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def store_raw_import(
    *,
    data: bytes,
    provider: str,
    source_path: Optional[Path] = None,
    db_path: Optional[Path] = None,
    compress: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Store raw import data in the database before parsing.

    Args:
        data: Raw bytes from the import file
        provider: Provider name (chatgpt, claude, etc.)
        source_path: Original file path
        db_path: Database path (uses default if None)
        compress: Whether to compress the blob (recommended for large files)
        metadata: Optional metadata to store

    Returns:
        Hash of the stored data
    """
    data_hash = compute_hash(data)

    # Check if already stored
    with open_connection(db_path) as conn:
        existing = get_raw_import(conn, data_hash)
        if existing:
            # Already stored, skip
            return data_hash

        # Compress if requested
        blob = zlib.compress(data, level=6) if compress else data

        # Store with metadata about compression
        store_metadata = metadata or {}
        store_metadata["compressed"] = compress
        store_metadata["original_size"] = len(data)
        store_metadata["stored_size"] = len(blob)

        upsert_raw_import(
            conn,
            hash=data_hash,
            provider=provider,
            source_path=str(source_path) if source_path else None,
            blob=blob,
            parser_version=PARSER_VERSION,
            parse_status="pending",
            metadata=store_metadata,
        )
        conn.commit()

    return data_hash


def retrieve_raw_import(data_hash: str, db_path: Optional[Path] = None) -> Optional[bytes]:
    """Retrieve raw import data from the database.

    Args:
        data_hash: Hash of the data to retrieve
        db_path: Database path (uses default if None)

    Returns:
        Decompressed raw bytes, or None if not found
    """
    with open_connection(db_path) as conn:
        row = get_raw_import(conn, data_hash)
        if not row:
            return None

        blob = row["blob"]
        metadata_json = row["metadata_json"]

        # Check if compressed
        metadata = json.loads(metadata_json) if metadata_json else {}
        if metadata.get("compressed"):
            blob = zlib.decompress(blob)

        return blob


def mark_parse_success(data_hash: str, db_path: Optional[Path] = None) -> None:
    """Mark a raw import as successfully parsed.

    Args:
        data_hash: Hash of the data
        db_path: Database path (uses default if None)
    """
    with open_connection(db_path) as conn:
        conn.execute(
            "UPDATE raw_imports SET parse_status = 'success' WHERE hash = ?",
            (data_hash,),
        )
        conn.commit()


def mark_parse_failed(
    data_hash: str,
    error_message: str,
    db_path: Optional[Path] = None,
) -> None:
    """Mark a raw import as failed to parse.

    Args:
        data_hash: Hash of the data
        error_message: Error message or stack trace
        db_path: Database path (uses default if None)
    """
    with open_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE raw_imports
               SET parse_status = 'failed',
                   error_message = ?
             WHERE hash = ?
            """,
            (error_message, data_hash),
        )
        conn.commit()


def get_failed_imports(provider: Optional[str] = None, db_path: Optional[Path] = None) -> list:
    """Get all failed imports for reprocessing.

    Args:
        provider: Optional provider filter
        db_path: Database path (uses default if None)

    Returns:
        List of failed import records
    """
    with open_connection(db_path) as conn:
        return list_failed_imports(conn, provider)


def get_import_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Get statistics about raw imports.

    Args:
        db_path: Database path (uses default if None)

    Returns:
        Dictionary with stats
    """
    with open_connection(db_path) as conn:
        # Total imports
        total = conn.execute("SELECT COUNT(*) FROM raw_imports").fetchone()[0]

        # By status
        by_status = {}
        for row in conn.execute("SELECT parse_status, COUNT(*) as count FROM raw_imports GROUP BY parse_status"):
            by_status[row["parse_status"]] = row["count"]

        # By provider
        by_provider = {}
        for row in conn.execute("SELECT provider, COUNT(*) as count FROM raw_imports GROUP BY provider"):
            by_provider[row["provider"]] = row["count"]

        # Storage size
        total_size = conn.execute("SELECT SUM(LENGTH(blob)) FROM raw_imports").fetchone()[0] or 0

        return {
            "total": total,
            "by_status": by_status,
            "by_provider": by_provider,
            "storage_size_mb": total_size / (1024 * 1024),
        }
