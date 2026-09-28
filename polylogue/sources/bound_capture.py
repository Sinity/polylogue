"""Capture a source file's bytes and validate what was captured.

Validating a source *path* and then copying it leaves a window in which the
file can change; the bytes retained are then not the bytes checked. Every
acquisition route that retains a bound source file therefore captures first
and validates the captured blob, once, before any raw row is recorded. A
refusal drops the capture's queued publication, so refused bytes never gain
a reservation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.storage.blob_publication import discard_pending_blob
from polylogue.storage.blob_store import BlobStore

from .dispatch import (
    LOCATION_VALIDATION_PREFIX_BYTES,
    ForeignOriginContentError,
    bound_location_provider,
    refuse_foreign_material,
)


def validate_captured_blob(
    blob_store: BlobStore,
    blob_hash: str,
    path: Path | str,
    location: Provider | str | None,
) -> None:
    """Refuse a captured blob whose bytes carry another origin's shape.

    Raises :class:`ForeignOriginContentError` after dropping the blob's
    queued publication.
    """
    if bound_location_provider(location) is None:
        return
    prefix = blob_store.read_prefix(blob_hash, LOCATION_VALIDATION_PREFIX_BYTES)
    try:
        refuse_foreign_material(path, location, prefix=prefix)
    except ForeignOriginContentError:
        discard_pending_blob(blob_store, blob_hash)
        raise


def capture_bound_source(
    blob_store: BlobStore,
    path: Path,
    location: Provider | str | None,
    capture: Callable[[], tuple[str, int]],
) -> tuple[str, int]:
    """Run ``capture`` (which writes the blob) and validate the captured bytes."""
    blob_hash, blob_size = capture()
    validate_captured_blob(blob_store, blob_hash, path, location)
    return blob_hash, blob_size


__all__ = ["capture_bound_source", "validate_captured_blob"]
