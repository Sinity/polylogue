"""Archive templates a law builds in place, sealed and cloned by the shared route.

A template is an :class:`~tests.infra.workload_artifacts.ImmutableTreeArtifact`
whose builder ran outside the artifact cache: the consuming law owns the tree's
location and lifetime, and everything after construction -- tier validation,
sealing, authenticated detached cloning, durable-identity rebinding -- is the
one publication route in :mod:`tests.infra.workload_artifacts`.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import threading
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from tests.infra.workload_artifacts import (
    ImmutableTreeArtifact,
    clone_immutable_tree,
    rebind_durable_identity,
    seal_fixture_tree,
)


def finalize_archive_template(root: Path) -> None:
    """Publish a reusable archive template only after a verified SQLite snapshot."""
    seal_fixture_tree(root)


def _template_key(template: Path) -> str:
    """Bind a clone to the exact tree it came from.

    A law-built template has no content-addressed cache identity; its location
    is what distinguishes it, and the resulting clone's ``source_manifest_id``
    must not read as a claim about a cached artifact.
    """
    return "archive-template:" + sha256(str(template.resolve()).encode()).hexdigest()


def clone_archive_template(template: Path, destination: Path) -> str:
    """Clone a sealed template into a private writable archive; report the method.

    Workspace fixtures create sibling directories (a render root, an inbox)
    under the archive root before seeding it, so the clone lands beside the
    destination and its entries are moved in: names the template supplies are
    replaced, names it does not are left alone. The durable identity names the
    tree's own path, so it is rebound once the tree reaches that path.
    """
    artifact = ImmutableTreeArtifact.adopt(template, key=_template_key(template))
    destination.mkdir(parents=True, exist_ok=True)
    staged = destination.parent / f".{destination.name}.clone.{uuid4().hex}"
    try:
        method = clone_immutable_tree(artifact, staged).clone_method
        for entry in sorted(staged.iterdir()):
            target = destination / entry.name
            if target.is_symlink() or target.is_file():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
            entry.replace(target)
        rebind_durable_identity(destination)
    finally:
        shutil.rmtree(staged, ignore_errors=True)
    return method


#: Name of the sealed template under a run root. Kept beside
#: ``.empty-archive-template`` and deliberately distinct from it: that one
#: carries a completed raw-authority census, which a plain bootstrap does not,
#: so the two are not interchangeable.
_BOOTSTRAP_TEMPLATE_NAME = ".bootstrap-archive-template"

_BOOTSTRAP_TEMPLATE_ROOT: Path | None = None
_BOOTSTRAP_TEMPLATE_LOCK = threading.Lock()
_BOOTSTRAP_TEMPLATE: Path | None = None


def register_bootstrap_template_root(run_root: Path | None) -> Path | None:
    """Name the run root whose sealed bootstrap template clones are taken from.

    Returns the root it replaced, so a caller that redirects the process for
    one test restores the session's own root instead of leaving every later
    archive on the production route. Without a registered root
    :func:`bootstrap_archive_root` has nowhere to cache a template and every
    caller takes the production bootstrap, which is correct but pays full DDL
    write cost per archive.
    """
    global _BOOTSTRAP_TEMPLATE_ROOT, _BOOTSTRAP_TEMPLATE
    previous = _BOOTSTRAP_TEMPLATE_ROOT
    _BOOTSTRAP_TEMPLATE_ROOT = None if run_root is None else Path(run_root)
    _BOOTSTRAP_TEMPLATE = None
    return previous


def build_bootstrap_archive_template(run_root: Path) -> Path:
    """Seal one plain empty archive per run root, built by the production route.

    The template is what ``initialize_active_archive_root`` produces on an
    empty directory and nothing more, so a clone of it is that function's
    result with its path-bound durable identity rebound.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    template = run_root / _BOOTSTRAP_TEMPLATE_NAME
    ready = run_root / f"{_BOOTSTRAP_TEMPLATE_NAME}.ready"
    lock_path = run_root / f"{_BOOTSTRAP_TEMPLATE_NAME}.lock"

    run_root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        if ready.exists() and template.is_dir():
            return template
        building = run_root / f"{_BOOTSTRAP_TEMPLATE_NAME}.building-{os.getpid()}"
        shutil.rmtree(building, ignore_errors=True)
        try:
            initialize_active_archive_root(building)
            finalize_archive_template(building)
            building.replace(template)
            ready.touch()
        finally:
            shutil.rmtree(building, ignore_errors=True)
    return template


def _bootstrap_template() -> Path | None:
    global _BOOTSTRAP_TEMPLATE
    if _BOOTSTRAP_TEMPLATE is not None:
        return _BOOTSTRAP_TEMPLATE
    run_root = _BOOTSTRAP_TEMPLATE_ROOT
    if run_root is None:
        return None
    with _BOOTSTRAP_TEMPLATE_LOCK:
        if _BOOTSTRAP_TEMPLATE is None:
            _BOOTSTRAP_TEMPLATE = build_bootstrap_archive_template(run_root)
    return _BOOTSTRAP_TEMPLATE


def _is_pristine_destination(root: Path) -> bool:
    """True when nothing under ``root`` could be lost or reinterpreted by a clone."""
    if root.is_symlink():
        return False
    if not root.exists():
        return True
    return root.is_dir() and next(os.scandir(root), None) is None


def bootstrap_archive_root(root: Path) -> Path:
    """Materialise an empty archive root, by reflink clone where that is faithful.

    A page-copy prototype still writes every tier's pages: measured on this
    checkout, ``initialize_active_archive_root`` costs 4.33 MB of write bytes
    per root, while cloning the sealed template costs 0.15 MB. Cloning is only
    equivalent for a destination with nothing in it -- an archive the caller
    has already seeded, or one carrying durable-train state, must take the
    production route so that route's own reconciliation decides its outcome.
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    template = _bootstrap_template() if _is_pristine_destination(root) else None
    if template is None:
        initialize_active_archive_root(root)
        return root
    clone_archive_template(template, root)
    return root


__all__ = [
    "bootstrap_archive_root",
    "build_bootstrap_archive_template",
    "clone_archive_template",
    "finalize_archive_template",
    "register_bootstrap_template_root",
]
