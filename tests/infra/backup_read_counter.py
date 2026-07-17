"""Measurement harness for archive-backup verification I/O.

The production verifier intentionally has two distinct roots while it works:
the immutable backup itself and a temporary scratch restore.  Keeping those
reads separate lets regression scenarios distinguish the restore proof from
post-proof receipt construction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch


@dataclass
class BackupVerificationReadCounter:
    """Count blob bytes read by verifier operation and physical root."""

    bytes_by_site: Counter[str] = field(default_factory=Counter)
    calls_by_site: Counter[str] = field(default_factory=Counter)

    def record(self, site: str, path: Path) -> None:
        if "blob" not in path.parts:
            return
        size = path.stat().st_size
        self.calls_by_site[site] += 1
        self.bytes_by_site[site] += size


def _root_name(path: Path) -> str:
    return "scratch" if any(part.startswith("polylogue-backup-verify-") for part in path.parts) else "backup"


@contextmanager
def backup_verification_read_counter() -> Iterator[BackupVerificationReadCounter]:
    """Measure blob reads performed by one real backup verification pass."""
    from polylogue.daemon import backup as backup_mod

    counter = BackupVerificationReadCounter()
    real_sha256_file = backup_mod._sha256_file
    real_read_bytes = Path.read_bytes

    def counted_sha256_file(path: Path) -> str:
        counter.record(f"sha256_file:{_root_name(path)}", path)
        return real_sha256_file(path)

    def counted_read_bytes(path: Path) -> bytes:
        counter.record(f"read_bytes:{_root_name(path)}", path)
        return real_read_bytes(path)

    with (
        patch.object(backup_mod, "_sha256_file", counted_sha256_file),
        patch.object(Path, "read_bytes", counted_read_bytes),
    ):
        yield counter


__all__ = ["BackupVerificationReadCounter", "backup_verification_read_counter"]
