"""Shared tail-overlay metadata for ahead-of-archive query results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TAIL_OVERLAY_PROVIDER_META_KEY = "tail_overlay"
TailArchiveState = Literal["ahead_of_archive", "unseen"]


@dataclass(frozen=True, slots=True)
class TailOverlayInfo:
    """Machine-readable freshness/provenance for tailed query results."""

    source_name: str
    source_path: str
    archive_state: TailArchiveState
    file_mtime: str | None = None

    def to_provider_meta_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "source_name": self.source_name,
            "source_path": self.source_path,
            "archive_state": self.archive_state,
        }
        if self.file_mtime is not None:
            payload["file_mtime"] = self.file_mtime
        return payload


def tail_overlay_from_provider_meta(provider_meta: dict[str, object] | None) -> TailOverlayInfo | None:
    """Parse tail-overlay provenance from provider metadata."""
    if provider_meta is None:
        return None

    raw = provider_meta.get(TAIL_OVERLAY_PROVIDER_META_KEY)
    if not isinstance(raw, dict):
        return None

    source_name = raw.get("source_name")
    source_path = raw.get("source_path")
    archive_state = raw.get("archive_state")
    if not isinstance(source_name, str) or not source_name.strip():
        return None
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    if archive_state not in {"ahead_of_archive", "unseen"}:
        return None

    file_mtime = raw.get("file_mtime")
    if file_mtime is not None and not isinstance(file_mtime, str):
        file_mtime = str(file_mtime)

    return TailOverlayInfo(
        source_name=source_name,
        source_path=source_path,
        archive_state=archive_state,
        file_mtime=file_mtime,
    )


def with_tail_overlay_provider_meta(
    provider_meta: dict[str, object] | None,
    info: TailOverlayInfo,
) -> dict[str, object]:
    """Return provider metadata augmented with tail-overlay provenance."""
    payload = dict(provider_meta or {})
    payload[TAIL_OVERLAY_PROVIDER_META_KEY] = info.to_provider_meta_payload()
    return payload


__all__ = [
    "TAIL_OVERLAY_PROVIDER_META_KEY",
    "TailArchiveState",
    "TailOverlayInfo",
    "tail_overlay_from_provider_meta",
    "with_tail_overlay_provider_meta",
]
