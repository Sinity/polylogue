"""Shared profile-qualified session-identity join keys for the Hermes bridge.

Hermes evidence arrives from several independently-acquired artifact
families for the same logical install: the durable ``state.db`` snapshot
(``hermes_state.py``), NeMo Relay ATIF trajectory documents and raw ATOF
event streams (``hermes_spans.py``), and the lifecycle/verification spool
parsers. Each artifact only ever carries the *raw* Hermes session id (the
value Hermes itself assigns) -- there is no cross-artifact install
identifier in the wire formats. Two separate Hermes installs (profiles) can
legitimately reuse the same raw session id, so a join key built from the raw
id alone silently collapses evidence from different installs onto one
archive session identity.

This module is the single source of truth for turning "the directory a raw
Hermes artifact file lives under" into a stable, hashed profile qualifier,
and for building/parsing the qualified session id
(``<raw_session_id>@profile-<profile_key>``) every Hermes parser uses. It
existed previously as private helpers duplicated inside ``hermes_state.py``;
centralizing it here is what lets ``hermes_spans.py`` (fs1.14) qualify ATIF/
ATOF observer-evidence session identity with the *same* key the state.db
parser computes for the conversational session it correlates with, instead
of inventing a second, incompatible scheme.
"""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.provider_identity import captured_hermes_profile_key, profile_root_for_artifact

__all__ = [
    "declares_profile_identity",
    "profile_key",
    "qualified_session_id",
    "split_qualified_session_id",
]


def declares_profile_identity(provider: Provider | None) -> bool:
    """Whether acquisition records a profile namespace for this provider's inputs.

    Only Hermes inputs, and inputs whose provider is not yet detected, carry
    one. Raws, cursors and cursor reconciliation all read through this rule.
    """
    return provider is None or provider in {Provider.HERMES, Provider.UNKNOWN}


def profile_key(profile_root: Path) -> str:
    """Return a stable, hashed qualifier for a Hermes install root directory.

    Raw profile paths are never exposed in archive identity -- only this
    truncated SHA-256 digest of the normalized (expanded, resolved) path.
    """
    return captured_hermes_profile_key(profile_root.expanduser().resolve(strict=False))


@dataclass(frozen=True, slots=True)
class CapturedHermesProfile:
    """Accepted profile namespace; physical file aliases do not change it."""

    root: Path
    key: str
    source_path: Path


@contextmanager
def capture_profile_namespace(artifact_path: Path, declared_parent: int) -> Iterator[CapturedHermesProfile]:
    """Bind the existing declared-profile convention to its directory input.

    Bind each declared directory alias once and keep its actual directory FD.
    The resulting parent must be the same directory already accepted by the
    acquisition owner; a nested alias does not change the declared profile.
    """
    declared_root = profile_root_for_artifact(artifact_path)
    root = declared_root.expanduser().resolve(strict=True)
    flags = getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW
    anchor = os.open(root, flags)
    walked = anchor
    components: list[tuple[int, str, int]] = []
    try:
        for component in artifact_path.parent.relative_to(declared_root).parts:
            next_descriptor = os.open(component, flags & ~os.O_NOFOLLOW, dir_fd=walked)
            components.append((walked, component, next_descriptor))
            current = os.stat(component, dir_fd=walked)
            opened = os.fstat(next_descriptor)
            if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                raise OSError(errno.ESTALE, "Hermes profile subtree changed", str(artifact_path))
            walked = next_descriptor
        expected = os.fstat(declared_parent)
        actual = os.fstat(walked)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise OSError(errno.ESTALE, "Hermes declared profile parent changed", str(artifact_path))
        accepted = os.fstat(anchor)
        named = declared_root.stat()
        if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (accepted.st_dev, accepted.st_ino):
            raise OSError(errno.ESTALE, "Hermes declared profile namespace changed", str(artifact_path))
        yield CapturedHermesProfile(
            root, captured_hermes_profile_key(root), root / artifact_path.relative_to(declared_root)
        )
    finally:
        for _, _, descriptor in reversed(components):
            os.close(descriptor)
        os.close(anchor)


def observe_profile_namespace(artifact_path: Path, expected: os.stat_result) -> CapturedHermesProfile:
    """Measure a namespace for a cursor comparison, without opening database bytes.

    This is observation evidence only. Acquisition captures its own receipt
    from its actual accepted input and never stamps this observation instead.
    """
    source = artifact_path.absolute()
    flags = getattr(os, "O_PATH", getattr(os, "O_SEARCH", os.O_RDONLY)) | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open(source.parent.resolve(strict=True), flags)
    try:
        with capture_profile_namespace(source, parent) as profile:
            current = os.stat(source.name, dir_fd=parent)
            named_parent = source.parent.stat()
            opened_parent = os.fstat(parent)
            if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                expected.st_dev,
                expected.st_ino,
                expected.st_size,
                expected.st_mtime_ns,
                expected.st_ctime_ns,
            ) or (named_parent.st_dev, named_parent.st_ino) != (opened_parent.st_dev, opened_parent.st_ino):
                raise OSError(errno.ESTALE, "Hermes cursor namespace observation changed", str(source))
            return profile
    finally:
        os.close(parent)


def qualified_session_id(raw_session_id: str, key: str) -> str:
    """Return the profile-qualified session id for a raw Hermes session id."""
    return f"{raw_session_id}@profile-{key}"


def split_qualified_session_id(qualified_id: str) -> tuple[str, str | None]:
    """Split a possibly-qualified session id into ``(raw_id, profile_key)``.

    Returns ``(qualified_id, None)`` unchanged when no ``@profile-`` marker is
    present (legacy/unqualified identity) -- callers must not silently invent
    a profile key that was never asserted by a producer.
    """
    raw_id, marker, key = qualified_id.partition("@profile-")
    return (raw_id, key) if marker else (qualified_id, None)
