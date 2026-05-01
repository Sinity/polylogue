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
    projection = yaml.safe_load(PROJECTION_PATH.read_text(encoding="utf-8"))
    assert isinstance(projection, dict)
    return cast(dict[str, object], projection)


def _files(projection: dict[str, object]) -> list[dict[str, object]]:
    files = projection.get("files", [])
    assert isinstance(files, list)
    assert all(isinstance(item, dict) for item in files)
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
    missing = [cell.get("path") for cell in files if not cell.get("target") or cell.get("target") == "TBD"]
    assert not missing, f"files without target: {missing!r}"


def test_projection_covers_live_polylogue_python_files() -> None:
    projection = _load_projection()
    projected_paths = {str(cell.get("path")) for cell in _files(projection)}
    live_paths = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "polylogue").glob("**/*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(live_paths - projected_paths)
    extra = sorted(projected_paths - live_paths)
    assert not missing, f"live Python files missing from topology projection: {missing[:10]!r}"
    assert not extra, f"topology projection references absent Python files: {extra[:10]!r}"


def test_projection_files_are_well_formed() -> None:
    """Each file entry has a ``path`` and ``loc``."""
    projection = _load_projection()
    files = _files(projection)
    bad = [item for item in files if not item.get("path") or not isinstance(item.get("loc"), int)]
    assert not bad, f"malformed file entries (missing path or loc): {bad[:3]!r}"


def test_every_owner_appears_in_files() -> None:
    """Each file row declares a concrete owner from the projection vocabulary."""
    projection = _load_projection()
    files = _files(projection)
    file_owners = {str(item.get("owner") or "stable") for item in files}
    assert file_owners
    malformed = sorted(
        owner
        for owner in file_owners
        if owner not in {"stable", "kernel", "lib-root", "storage-root"} and not owner.startswith("#")
    )
    assert not malformed, f"malformed topology owners: {malformed!r}"
