"""Shared magic-byte detection for non-conversational binary payloads.

polylogue-hbtj2: a "detection/parse strictness" bug let binary files
(concretely, SQLite databases such as Hermes ``state.db``/
``verification_evidence.db`` and Codex ``state_5.sqlite``) reach session
governance opportunistically -- via a file-extension match or a bare
provider hint, never a verified content check. The audit that found this
(``/realm/data/derived/reports/polylogue-authority-dataflow-2026-08-02.html``,
"the residue, dissected") measured ~450 MB of such miscaptures sitting in
``raw_sessions`` marked "genuinely diverged" by a byte-diff classifier that
was never designed to reason about opaque binary snapshots.

This module is the single, shared, magic-byte-based detector: identity is
decided from the leading bytes of the payload, never a file extension and
never a provider hint alone (an extension can be renamed; a hint only says
*whose* directory a file came from, not what its bytes actually are). It
is intentionally provider-agnostic and format-agnostic -- new binary
formats are added to :data:`BINARY_SIGNATURES` once, and every call site
that consults :func:`detect_binary_signature` gets the new refusal for
free, rather than each acquisition/detection route hand-rolling its own
check (the broader "propose a general strategy, not only for SQLite"
direction from the same audit).

ZIP is deliberately excluded from :data:`BINARY_SIGNATURES`: ChatGPT and
Claude-ai GDPR export bundles are legitimately zip-shaped session sources,
handled end-to-end by the ``decoder_zip`` pipeline. ``looks_like_zip_bytes``
still exists for read-only reporting (the broader binary sweep wants to
know a raw row is zip-shaped without asserting it is a miscapture), but no
call site here treats a plain ZIP magic match as "refuse this artifact".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BinarySignature:
    """One recognized non-conversational binary format, by magic bytes."""

    name: str
    magic: bytes


SQLITE_MAGIC_HEADER = b"SQLite format 3\x00"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_GZIP_MAGIC = b"\x1f\x8b"
_JPEG_MAGIC = b"\xff\xd8\xff"
_PDF_MAGIC = b"%PDF-"
ZIP_MAGIC_VARIANTS: tuple[bytes, ...] = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")

# Ordered so the most specific / most-frequently-seen-as-a-miscapture
# signature (SQLite -- the concrete finding this bead fixes) is checked
# first. ZIP is intentionally not a member: see module docstring.
BINARY_SIGNATURES: tuple[BinarySignature, ...] = (
    BinarySignature("sqlite", SQLITE_MAGIC_HEADER),
    BinarySignature("png", _PNG_MAGIC),
    BinarySignature("gzip", _GZIP_MAGIC),
    BinarySignature("jpeg", _JPEG_MAGIC),
    BinarySignature("pdf", _PDF_MAGIC),
)


def looks_like_sqlite_bytes(payload: bytes) -> bool:
    """Return whether *payload* starts with the SQLite file-format magic header."""
    return payload.startswith(SQLITE_MAGIC_HEADER)


def looks_like_zip_bytes(payload: bytes) -> bool:
    """Return whether *payload* starts with a ZIP local/empty/spanned-archive signature.

    Reporting-only (see module docstring) -- legitimate export bundles are
    zip-shaped, so this is never used to auto-refuse an artifact.
    """
    return payload.startswith(ZIP_MAGIC_VARIANTS)


def detect_binary_signature(payload: bytes) -> BinarySignature | None:
    """Return the first recognized non-session binary signature *payload* matches, if any."""
    for signature in BINARY_SIGNATURES:
        if payload.startswith(signature.magic):
            return signature
    return None


__all__ = [
    "BINARY_SIGNATURES",
    "SQLITE_MAGIC_HEADER",
    "ZIP_MAGIC_VARIANTS",
    "BinarySignature",
    "detect_binary_signature",
    "looks_like_sqlite_bytes",
    "looks_like_zip_bytes",
]
