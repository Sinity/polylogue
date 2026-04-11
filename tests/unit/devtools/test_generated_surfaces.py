from __future__ import annotations

from devtools.generated_surfaces import GENERATED_SURFACES


def test_generated_surfaces_use_public_devtools_commands() -> None:
    assert GENERATED_SURFACES
    for surface in GENERATED_SURFACES:
        assert surface.command[0] == "devtools"
        assert len(surface.command) == 2
        assert callable(surface.main)


def test_generated_surface_names_and_labels_are_unique() -> None:
    assert len({surface.name for surface in GENERATED_SURFACES}) == len(GENERATED_SURFACES)
    assert len({surface.label for surface in GENERATED_SURFACES}) == len(GENERATED_SURFACES)
