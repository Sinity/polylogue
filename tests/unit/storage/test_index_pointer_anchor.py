"""The canonical index pointer must survive blue-green promotion.

`promote()` replaces the archive's own `index.db` with a symlink to the newly
built generation. The canonical pointer is therefore that `index.db` path --
never the generation the symlink currently targets.

`IndexGenerationStore.__init__` used to follow the symlink and write the
resolved `.index-generations/gen-*/index.db` path into `.index-active-pointer`,
then reject exactly that value on the next construction with "invalid canonical
index pointer anchor". The store poisoned its own anchor on first use and
refused every run afterwards, which blocked an offline rebuild of a real
archive that had been promoted at least once. It also made `generations_root`
nest as `.index-generations/gen-*/.index-generations`.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.storage.index_generation import IndexGenerationStore


def _archive_with_promoted_generation(root: Path) -> Path:
    """An archive shaped exactly like one that has been promoted once."""
    root.mkdir(parents=True, exist_ok=True)
    generation = root / ".index-generations" / "gen-1784807190100-34534407"
    generation.mkdir(parents=True, exist_ok=True)
    (generation / "index.db").write_bytes(b"")
    (root / "index.db").symlink_to(generation / "index.db")
    return generation


def test_promoted_archive_anchors_on_index_db_not_the_generation(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    generation = _archive_with_promoted_generation(root)

    store = IndexGenerationStore.for_archive_root(root)

    assert store.active_pointer == root / "index.db"
    assert ".index-generations" not in store.active_pointer.parts
    assert store.generations_root == root / ".index-generations"
    assert store.generations_root != generation / ".index-generations"


def test_construction_is_repeatable_on_a_promoted_archive(tmp_path: Path) -> None:
    """The regression: the second construction used to raise on the first's own anchor."""
    root = tmp_path / "archive"
    _archive_with_promoted_generation(root)

    first = IndexGenerationStore.for_archive_root(root)
    second = IndexGenerationStore.for_archive_root(root)

    assert first.active_pointer == second.active_pointer == root / "index.db"


def test_a_poisoned_anchor_heals_instead_of_failing(tmp_path: Path) -> None:
    """An archive already carrying the bad value must recover on open, not need hand repair."""
    root = tmp_path / "archive"
    generation = _archive_with_promoted_generation(root)
    (root / ".index-active-pointer").write_text(str(generation / "index.db"), encoding="utf-8")

    store = IndexGenerationStore.for_archive_root(root)

    assert store.active_pointer == root / "index.db"
    assert (root / ".index-active-pointer").read_text(encoding="utf-8").strip() == str(root / "index.db")


def test_symlink_leaving_the_archive_is_still_followed(tmp_path: Path) -> None:
    """An archive root that is a symlink farm keeps its canonical pointer elsewhere."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "index.db").write_bytes(b"")
    root = tmp_path / "farm"
    root.mkdir()
    (root / "index.db").symlink_to(real / "index.db")

    store = IndexGenerationStore.for_archive_root(root)

    assert store.active_pointer == real / "index.db"


def test_farm_target_directly_inside_a_similarly_named_directory_is_followed(tmp_path: Path) -> None:
    """A name match alone must not condemn a valid pointer.

    Rejecting any path that merely mentions `.index-generations` treats a
    legitimate symlink-farm target that happens to sit in a directory of that
    name as poisoned, and silently rewrites the pointer back to a stale local
    `index.db`. What actually breaks the invariant is being *inside a
    generation*, one level deeper -- because `generations_root` derives from the
    pointer's parent. A file sitting directly in such a directory derives
    self-consistent roots and is fine.
    """
    real = tmp_path / "outside" / ".index-generations"
    real.mkdir(parents=True)
    (real / "index.db").write_bytes(b"")
    root = tmp_path / "farm"
    root.mkdir()
    (root / "index.db").symlink_to(real / "index.db")

    store = IndexGenerationStore.for_archive_root(root)

    assert store.active_pointer == real / "index.db"
    assert store.generations_root == real / ".index-generations"


def test_farm_target_inside_another_archives_generation_is_refused(tmp_path: Path) -> None:
    """Depth, not the directory name, is the invariant -- and it holds off-archive too.

    A farm whose `index.db` points into some *other* archive's generation would
    derive `generations_root` as `<gen>/.index-generations`, the same nested
    nonsense the local case produced. Narrowing the check to depth must not
    narrow it to locality.
    """
    foreign_generation = tmp_path / "other" / ".index-generations" / "gen-1784807190100-34534407"
    foreign_generation.mkdir(parents=True)
    (foreign_generation / "index.db").write_bytes(b"")
    root = tmp_path / "farm"
    root.mkdir()
    (root / "index.db").symlink_to(foreign_generation / "index.db")

    store = IndexGenerationStore.for_archive_root(root)

    assert store.active_pointer == root / "index.db"
    assert store.generations_root == root / ".index-generations"
