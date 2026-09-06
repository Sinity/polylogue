"""Archive templates a law builds in place, sealed and cloned by the shared route.

A template is an :class:`~tests.infra.workload_artifacts.ImmutableTreeArtifact`
whose builder ran outside the artifact cache: the consuming law owns the tree's
location and lifetime, and everything after construction -- tier validation,
sealing, authenticated detached cloning, durable-identity rebinding -- is the
one publication route in :mod:`tests.infra.workload_artifacts`.
"""

from __future__ import annotations

import shutil
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


__all__ = ["clone_archive_template", "finalize_archive_template"]
