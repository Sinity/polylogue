"""Typed payloads for the deterministic Polylogue demo archive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DemoSeedResult:
    """Result of materializing and ingesting the deterministic demo archive."""

    archive_root: Path
    source_root: Path
    session_count: int
    message_count: int
    session_ids: tuple[str, ...]
    overlays_seeded: bool
    assertion_count: int

    def to_payload(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "source_root": str(self.source_root),
            "session_count": self.session_count,
            "message_count": self.message_count,
            "session_ids": list(self.session_ids),
            "overlays_seeded": self.overlays_seeded,
            "assertion_count": self.assertion_count,
        }


@dataclass(frozen=True, slots=True)
class DemoVerifyResult:
    """Semantic verification result for the deterministic demo archive."""

    archive_root: Path
    ok: bool
    session_count: int
    message_count: int
    query_hits: tuple[str, ...]
    overlays_present: bool
    absolute_path_leaks: tuple[str, ...]
    problems: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        return {
            "archive_root": str(self.archive_root),
            "ok": self.ok,
            "session_count": self.session_count,
            "message_count": self.message_count,
            "query_hits": list(self.query_hits),
            "overlays_present": self.overlays_present,
            "absolute_path_leaks": list(self.absolute_path_leaks),
            "problems": list(self.problems),
        }


__all__ = ["DemoSeedResult", "DemoVerifyResult"]
