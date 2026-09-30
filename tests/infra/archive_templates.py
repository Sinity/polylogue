"""Archive templates a law builds in place, sealed and cloned by the shared route.

A template is an :class:`~tests.infra.workload_artifacts.ImmutableTreeArtifact`
whose builder ran outside the artifact cache: the consuming law owns the tree's
location and lifetime, and everything after construction -- tier validation,
sealing, authenticated detached cloning, destination-owned train admission -- is the
one publication route in :mod:`tests.infra.workload_artifacts`.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from tests.infra.workload_artifacts import (
    ImmutableTreeArtifact,
    clone_immutable_tree,
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
    """Clone into the actual reserved destination through the shared owner."""
    artifact = ImmutableTreeArtifact.adopt(template, key=_template_key(template))
    return clone_immutable_tree(artifact, destination).clone_method


def bootstrap_archive_root(root: Path) -> Path:
    """Construct each empty fixture through the canonical baseline and train owner."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(root)
    return root


def bootstrap_ready_archive_root(root: Path) -> Path:
    """Construct an empty fixture and complete its real raw-authority census."""
    from polylogue.config import Config
    from polylogue.storage.raw_reconciler import inspect_raw_authority_frontier

    bootstrap_archive_root(root)
    inspect_raw_authority_frontier(
        Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")
    )
    return root


__all__ = [
    "bootstrap_archive_root",
    "bootstrap_ready_archive_root",
    "clone_archive_template",
    "finalize_archive_template",
]
