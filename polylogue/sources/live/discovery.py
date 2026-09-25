"""Canonical ordered discovery walk shared by daemon intake and source baselines."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from polylogue.logging import WARNING, emit
from polylogue.sources.live.acquisition_log import log_unclaimed_file
from polylogue.sources.live.source_selection import deepest_source_for_path
from polylogue.sources.live.watcher import WatchSource
from polylogue.sources.walk_faults import WalkFault, WalkRefusedError


def _walk_entry_key(path: Path, *, is_dir: bool) -> str:
    """The order key a walk entry occupies among its siblings.

    A file's key is its path string -- exactly the key the resume cursor
    compares against. A directory's key is its path string plus the path
    separator, which is what makes descending at the keyed position exact:
    ``root/a.json`` sorts below ``root/a`` + separator because ``.`` (46)
    sorts below ``/`` (47), so ``root/a.json`` is emitted before every file
    under ``root/a/``. Both comparisons are plain code-point ordering, so
    unicode and mixed-case names order identically here and at the cursor.
    """

    text = str(path)
    return text + os.sep if is_dir else text


def _real_path(path: Path) -> str:
    """The resolved real path used as this walk's cycle-detection identity."""

    return os.path.realpath(path)


def _emit_discovery_fault(
    source: WatchSource,
    fault: WalkFault,
    *,
    reason: str,
    on_disposition: Callable[[Path, str, str], None] | None = None,
) -> None:
    """Report one non-fatal discovery fault without losing the rest of the page.

    A symlink cycle and a dangling symlink are both permanent properties of
    the tree: refusing the whole class for them would strand every real file
    beside them forever. They are still evidence -- the same
    ``daemon.intake.discovery_failed`` shape the dispatcher emits -- so the
    entry is counted per path instead of silently vanishing.
    """

    if on_disposition is not None:
        on_disposition(fault.path, "fault", reason)
    emit(
        "daemon.intake.discovery_failed",
        level=WARNING,
        outcome="degraded",
        reason=reason,
        component=source.name,
        path=str(fault.path),
        error_detail=fault.detail,
    )


def _admit_linked_directory(
    source: WatchSource,
    path: Path,
    *,
    visited_real_paths: set[str] | None,
    on_disposition: Callable[[Path, str, str], None] | None = None,
) -> bool:
    """Whether a directory symlink may be descended into on this walk.

    Containment is the standing rule the live watcher already enforces
    through ``deepest_source_for_path``: a link whose resolved target lies
    outside the source root is rejected, never admitted. Only a link that
    stays inside the root is followed, which is what makes a
    ``current -> 2026-09`` style export pointer discoverable without letting
    a link hand intake material the source was never configured to read.
    """

    try:
        real = _real_path(path)
        root_real = _real_path(source.root)
    except OSError as exc:
        _emit_discovery_fault(
            source,
            WalkFault(path, f"symlink target could not be resolved: {exc}"),
            reason="unresolvable_symlink",
            on_disposition=on_disposition,
        )
        return False
    if real != root_real and not real.startswith(root_real + os.sep):
        _emit_discovery_fault(
            source,
            WalkFault(path, f"symlink target {real} escapes the source root"),
            reason="escaping_symlink",
            on_disposition=on_disposition,
        )
        return False
    if visited_real_paths is not None:
        if real in visited_real_paths:
            _emit_discovery_fault(
                source,
                WalkFault(path, f"symlink cycle: {real} was already visited"),
                reason="symlink_cycle",
                on_disposition=on_disposition,
            )
            return False
        visited_real_paths.add(real)
    return True


def _ordered_children(
    source: WatchSource,
    directory: Path,
    after: str | None,
    scandir: Callable[[Path], Any] = os.scandir,
    *,
    visited_real_paths: set[str] | None = None,
    on_disposition: Callable[[Path, str, str], None] | None = None,
) -> list[tuple[str, Path, bool]]:
    """Siblings of ``directory``, reverse-sorted so a stack pops them in order.

    Directory symlinks are followed: an operator who mounts an export tree
    through a symlink configured a real source root, and refusing to enter it
    made the source silently unacquired. Following links needs the two guards
    below -- ``visited_real_paths`` (resolved real paths already entered on
    this walk) terminates cycles, and a resolved target outside the source
    root is rejected rather than followed.
    """

    children: list[tuple[str, Path, bool]] = []
    try:
        entries = scandir(directory)
    except OSError as exc:
        # Refuse rather than return an empty sibling list. Returning ``[]``
        # here removed every file beneath ``directory`` from the walk while
        # the caller saw a result indistinguishable from "that subtree is
        # empty"; the daemon's dispatcher already turns a raising ``discover``
        # into a counted ``daemon.intake.discovery_failed`` event plus a class
        # report reason, and retries the class on the next pass.
        raise WalkRefusedError(
            "intake discovery could not read a source directory",
            [WalkFault(directory, f"scandir failed: {exc}")],
        ) from exc
    with entries:
        for entry in entries:
            path = Path(entry.path)
            try:
                is_link = entry.is_symlink()
                if entry.is_dir(follow_symlinks=False) or (is_link and entry.is_dir()):
                    if source.ignores_directory(path):
                        if on_disposition is not None:
                            on_disposition(path, "excluded", "ignored_directory")
                        continue
                    if is_link and not _admit_linked_directory(
                        source,
                        path,
                        visited_real_paths=visited_real_paths,
                        on_disposition=on_disposition,
                    ):
                        continue
                    key = _walk_entry_key(path, is_dir=True)
                    # Every descendant path begins with ``key``. When the
                    # cursor sorts above ``key`` without having it as a
                    # prefix, it sorts above every such descendant too, so
                    # the whole subtree is already behind the cursor.
                    if after is not None and after > key and not after.startswith(key):
                        continue
                    children.append((key, path, True))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    if is_link and entry.is_file():
                        target = path.resolve()
                        if not target.is_relative_to(source.root.resolve()):
                            _emit_discovery_fault(
                                source,
                                WalkFault(path, f"symlink target {target} escapes the source root"),
                                reason="escaping_symlink",
                                on_disposition=on_disposition if source.accepts(path) else None,
                            )
                            continue
                        if on_disposition is not None and source.accepts(path):
                            on_disposition(path, "alias", "candidate_link")
                        continue
                    elif is_link:
                        # A dangling link is a fault, not an absence: the
                        # export it named is missing. Counting it keeps the
                        # walk alive over the rest of the directory.
                        _emit_discovery_fault(
                            source,
                            WalkFault(path, "symlink target does not exist"),
                            reason="broken_symlink",
                            on_disposition=on_disposition if source.accepts(path) else None,
                        )
                        continue
                    else:
                        continue
            except FileNotFoundError:
                # Ordinary producer churn: the entry vanished between the
                # listing and the type probe. Nothing was hidden.
                continue
            except OSError as exc:
                raise WalkRefusedError(
                    "intake discovery could not inspect a source entry",
                    [WalkFault(path, f"stat failed: {exc}")],
                ) from exc
            children.append((_walk_entry_key(path, is_dir=False), path, False))
    children.sort(key=lambda child: child[0], reverse=True)
    return children


def _log_unclaimed_intake_candidate(path: Path, *, source_name: str, suffixes: tuple[str, ...]) -> None:
    """Log one discovered file no configured suffix accepts.

    Best-effort ``stat``: a file that vanished between the listing and this
    call was still seen and unclaimed, just without size/mtime detail.
    """
    try:
        stat_result = path.stat()
        size: int | None = stat_result.st_size
        mtime: float | None = stat_result.st_mtime
    except OSError:
        size, mtime = None, None
    log_unclaimed_file(
        path=path,
        size=size,
        mtime=mtime,
        reason=f"suffix not in watched set {suffixes} for source {source_name!r}",
        source_name=source_name,
    )


def _bounded_source_paths(
    source: WatchSource,
    all_sources: tuple[WatchSource, ...],
    *,
    limit: int,
    after: str | None,
    scandir: Callable[[Path], Any] = os.scandir,
    on_disposition: Callable[[Path, str, str], None] | None = None,
    collect: bool = True,
) -> list[Path]:
    """Collect at most ``limit`` files, stopping as soon as it is full.

    Files are emitted in exact lexicographic order of their path strings.
    That order is authoritative because it is the order the resume cursor
    compares in (``after``, advanced in ``acknowledge`` over consumed
    items): a producer that emitted ``os.scandir`` order instead let a
    high-water mark skip files it had never emitted, which was permanent
    loss rather than delay. Emitting in cursor order also makes the
    ``limit`` early exit safe -- the next pass resumes at exactly the key
    the previous one stopped on, mid-directory or not.

    Directory symlinks whose target stays inside the source root are
    followed, so an export tree mounted behind a link is discovered. Every
    directory entered on this walk records its resolved real path, which is
    what terminates a cycle.
    """

    if limit <= 0:
        return []
    if not source.root.is_dir():
        # A missing or unmounted root is a refusal, not an empty backlog.
        # ``operations/raw_sessions/sessions.py`` already raises for exactly
        # this condition; returning ``[]`` reported a fully ingested source
        # when the export drive was simply not mounted.
        raise WalkRefusedError(
            "intake discovery could not read a source root",
            [WalkFault(source.root, "source root is unavailable")],
        )
    found: list[Path] = []
    accepted_count = 0
    visited_real_paths: set[str] = {_real_path(source.root)}
    stack: list[list[tuple[str, Path, bool]]] = [
        _ordered_children(
            source,
            source.root,
            after,
            scandir,
            visited_real_paths=visited_real_paths,
            on_disposition=on_disposition,
        )
    ]
    while stack and accepted_count < limit:
        level = stack[-1]
        if not level:
            stack.pop()
            continue
        key, path, is_dir = level.pop()
        if is_dir:
            visited_real_paths.add(_real_path(path))
            stack.append(
                _ordered_children(
                    source,
                    path,
                    after,
                    scandir,
                    visited_real_paths=visited_real_paths,
                    on_disposition=on_disposition,
                )
            )
            continue
        if after is not None and key <= after:
            continue
        try:
            if deepest_source_for_path(path, all_sources) is not source:
                if on_disposition is not None:
                    on_disposition(path, "excluded", "owned_by_other_source")
                continue
            if not source.accepts(path):
                if on_disposition is not None:
                    on_disposition(path, "excluded", "artifact_rule")
                # A file this source's own walk reached but whose suffix no
                # detector is configured to accept. The record exists whether
                # or not an operator runs the standalone sweep, and discovery
                # is the only production walk left that reaches it.
                _log_unclaimed_intake_candidate(path, source_name=source.name, suffixes=source.suffixes)
                continue
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise WalkRefusedError(
                "intake discovery could not resolve a source file's owner",
                [WalkFault(path, f"ownership resolution failed: {exc}")],
            ) from exc
        accepted_count += 1
        if collect:
            found.append(path)
        if on_disposition is not None:
            on_disposition(path, "accepted", "source_artifact")
    return found
