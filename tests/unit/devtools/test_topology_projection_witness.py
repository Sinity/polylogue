"""Witness regression: the topology projection has the documented invariants.

Promotes the structural-cohesion projection from a generated artifact to a
committed witness (Phase 2 of `#432 <https://github.com/Sinity/polylogue/issues/432>`_).
The witness metadata records the invariants; this test verifies the live
projection still satisfies them.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import yaml

from polylogue.proof.witnesses import WITNESS_SCHEMA_VERSION, WitnessMetadata

REPO_ROOT = Path(__file__).resolve().parents[3]
WITNESS_PATH = REPO_ROOT / "tests" / "witnesses" / "topology-projection.witness.json"
PROJECTION_PATH = REPO_ROOT / "docs" / "plans" / "topology-target.yaml"


def _load_projection() -> dict[str, object]:
    return cast(dict[str, object], yaml.safe_load(PROJECTION_PATH.read_text(encoding="utf-8")))


def _files(projection: dict[str, object]) -> list[dict[str, object]]:
    files = projection.get("files", [])
    assert isinstance(files, list)
    return cast(list[dict[str, object]], files)


def test_committed_witness_metadata_validates() -> None:
    metadata = WitnessMetadata.read(WITNESS_PATH)
    assert metadata.validation_errors() == ()
    assert metadata.schema_version == WITNESS_SCHEMA_VERSION
    assert metadata.committed is True


def test_every_cell_has_target_path() -> None:
    projection = _load_projection()
    files = _files(projection)
    assert files, "topology projection has no files"
    missing = [cell.get("path") for cell in files if not cell.get("target")]
    assert not missing, f"files without target: {missing!r}"


def test_projection_files_are_well_formed() -> None:
    """Each file entry has a ``path`` and ``loc``; a target may be ``TBD`` during migration."""
    projection = _load_projection()
    files = _files(projection)
    bad = [item for item in files if not item.get("path") or not isinstance(item.get("loc"), int)]
    assert not bad, f"malformed file entries (missing path or loc): {bad[:3]!r}"


def test_every_owner_appears_in_files() -> None:
    """Each owner declared in the projection's owner-counts has at least one file."""
    projection = _load_projection()
    files = _files(projection)
    file_owners = {str(item.get("owner") or "stable") for item in files}
    counts = projection.get("owners", {})
    declared_owners = set(counts.keys()) if isinstance(counts, dict) else set()
    missing = declared_owners - file_owners
    assert not missing, f"owners declared with no member files: {missing!r}"
