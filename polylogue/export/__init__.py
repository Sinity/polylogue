"""Fail-closed sanitized export of shareable archive datasets (#2381).

This package turns a matched session scope into a publishable bundle —
``dataset.jsonl`` + ``redaction-manifest.json`` + ``README.md`` — with private
absolute paths and known secrets scrubbed. The leak-check gate
(:func:`verify_sanitized_export`) is authoritative: the writer refuses to
publish if the written files still contain a private absolute path, a
``$HOME``-relative path, or a known-secret pattern.
"""

from __future__ import annotations

from polylogue.export.sanitize import (
    REDACTED_PATH_PREFIX,
    REDACTED_SECRET,
    SANITIZED_EXPORT_BUNDLE_VERSION,
    SanitizedExportError,
    SanitizedExportRequest,
    SanitizedExportResult,
    SanitizedExportVerifyResult,
    produce_sanitized_export,
    sanitize_rows,
    verify_sanitized_export,
    write_sanitized_bundle,
)

__all__ = [
    "REDACTED_PATH_PREFIX",
    "REDACTED_SECRET",
    "SANITIZED_EXPORT_BUNDLE_VERSION",
    "SanitizedExportError",
    "SanitizedExportRequest",
    "SanitizedExportResult",
    "SanitizedExportVerifyResult",
    "produce_sanitized_export",
    "sanitize_rows",
    "verify_sanitized_export",
    "write_sanitized_bundle",
]
