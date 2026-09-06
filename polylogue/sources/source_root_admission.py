"""Admission refusal for source roots inside a foreign Polylogue archive.

An archive's own directories -- capture spool, inbox, drive cache -- hold
material it already owns. Reading them as a *source* re-admits one archive's
contents into another as if newly captured.

The destination archive is exempt: its own spool and inbox are the live
capture route. So is any directory that aliases it, since one archive is
routinely reachable under several roots through symlinked tiers.
"""

from __future__ import annotations

from pathlib import Path

#: The durable tier whose presence makes a directory an archive root, and
#: whose real path is that archive's identity.
_ARCHIVE_MARKER = "source.db"


class SourceRootRefusedError(Exception):
    """Acquisition refused a source root that is not a capture location."""


def _real(path: Path) -> Path | None:
    try:
        return path.expanduser().resolve()
    except OSError:
        return None


def containing_archive_root(path: Path) -> Path | None:
    """The Polylogue archive whose directory tree contains *path*, if any."""
    for candidate in (path, *path.parents):
        if (candidate / _ARCHIVE_MARKER).exists():
            return candidate
    return None


def _same_archive(one: Path, other: Path) -> bool:
    if _real(one) == _real(other):
        return True
    marker = _real(one / _ARCHIVE_MARKER)
    return marker is not None and marker == _real(other / _ARCHIVE_MARKER)


def refuse_non_capture_source_root(path: Path, *, destination: Path | None = None) -> None:
    """Raise :class:`SourceRootRefusedError` when *path* belongs to another archive.

    Fail closed on ownership, open on resolution: an unresolvable source root
    is left to the walk, which reports its own read failure.
    """
    resolved = _real(path)
    if resolved is None:
        return
    owner = containing_archive_root(resolved)
    if owner is None:
        return
    if destination is not None and _same_archive(owner, destination):
        return
    raise SourceRootRefusedError(
        f"Source root {resolved} lies inside the Polylogue archive {owner}, "
        "which is not the archive being written. An archive's own material is "
        "not a capture location; point the source at the provider's capture "
        "directory."
    )


__all__ = [
    "SourceRootRefusedError",
    "containing_archive_root",
    "refuse_non_capture_source_root",
]
