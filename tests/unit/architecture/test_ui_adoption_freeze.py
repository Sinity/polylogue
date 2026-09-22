"""``polylogue.ui`` may lose adopters, never gain them.

polylogue-4wqi2 recorded an adoption freeze over ``polylogue/ui/``: the
canonical-versus-migrate ownership call belongs to polylogue-4p1, and until
that call lands the importer set is a ratchet baseline that may shrink and
must not grow. Nothing enforced the freeze, so this module is the ratchet.

The freeze distinguishes two kinds of adopter, because they fail differently:

* **Importers** -- modules whose Python import graph reaches
  ``polylogue.ui``. A new one makes the ownership decision more expensive
  by adding another consumer to migrate.
* **Theme-file readers** -- tooling that loads ``polylogue/ui/theme.py`` by
  path instead of importing it (``devtools/render_webui_design_system.py``
  runs it with :mod:`runpy` precisely to avoid executing the package, and
  ``devtools/generated_surfaces.py`` names it as a generated-surface
  source). An import census cannot see these, so they are frozen by path.

A prose cross-reference is deliberately NOT an adopter. ``polylogue/config.py``
mentions :func:`polylogue.ui.theme.resolve_theme_mode` in a property docstring
and imports nothing from ``polylogue.ui`` -- the dependency runs the other way
(``polylogue/ui/theme.py`` imports ``polylogue.config``). The bead's recorded
nine-importer baseline counted that docstring, both path readers, and
``polylogue/sources/parsers/chatgpt.py`` -- whose only match is the *bead id*
``polylogue-ui3q4`` in a comment. The measured import set is five.

Anti-vacuity: add any module importing ``polylogue.ui`` and
``test_ui_importer_set_never_grows`` goes red naming it; name
``polylogue/ui/theme.py`` in a new tooling file and
``test_ui_theme_path_reader_set_never_grows`` goes red. Deleting an adopter
is allowed by construction and
``test_the_freeze_allows_an_adopter_to_be_removed`` pins that direction, so
the ratchet cannot be satisfied by forbidding all change. Emptying either
baseline turns ``test_the_freeze_still_has_something_to_protect`` red.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Package under freeze.
FROZEN_PACKAGE = "polylogue.ui"

#: Trees searched for adopters. ``tests/`` is excluded on purpose: the freeze
#: is about production/tooling adoption, and a test importing the facade it
#: covers is not a migration cost.
SEARCHED_TREES = ("polylogue", "devtools")

#: Modules whose import graph reaches ``polylogue.ui`` at the freeze baseline,
#: measured by :func:`ui_importers` at 6490c82f3. May shrink, never grow.
FROZEN_IMPORTERS = frozenset(
    {
        "polylogue/cli/click_app.py",
        "polylogue/cli/query_output.py",
        "polylogue/cli/shared/helper_summary.py",
        "polylogue/cli/shared/types.py",
        "polylogue/rendering/renderers/html.py",
    }
)

#: Tooling that reads ``polylogue/ui/theme.py`` as a file rather than
#: importing it. Invisible to an import census, so frozen separately.
FROZEN_THEME_PATH_READERS = frozenset(
    {
        "devtools/generated_surfaces.py",
        "devtools/render_webui_design_system.py",
    }
)

_THEME_PATH_TOKEN = "polylogue/ui/theme.py"


def _searched_python_files() -> list[Path]:
    """Every ``.py`` under the searched trees, excluding ``ui/`` itself.

    Deliberately a filesystem walk and not ``git ls-files``: a census
    restricted to tracked files cannot see the adopter a worker just wrote,
    so the freeze would only bite after the file was staged. That failure was
    measured -- the first version of this ratchet used ``git ls-files`` and
    stayed green with a live sixth importer sitting on disk.
    """
    found: list[Path] = []
    for tree in SEARCHED_TREES:
        for path in sorted((REPO_ROOT / tree).rglob("*.py")):
            relative = path.relative_to(REPO_ROOT)
            if relative.as_posix().startswith("polylogue/ui/"):
                continue
            found.append(relative)
    return found


def _imported_modules(tree: ast.AST, *, module_parts: tuple[str, ...]) -> set[str]:
    """Absolute dotted module targets of every import statement in ``tree``.

    ``module_parts`` is the importing module's own dotted path, used to
    resolve relative imports the way Python does, so ``from . import ui``
    inside ``polylogue/`` is counted exactly like ``from polylogue import ui``.
    """
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = module_parts[: len(module_parts) - node.level]
            prefix = ".".join((*base, node.module)) if node.module else ".".join(base)
        else:
            prefix = node.module or ""
        if not prefix:
            continue
        targets.add(prefix)
        targets.update(f"{prefix}.{alias.name}" for alias in node.names)
    return targets


def ui_importers() -> set[str]:
    """Repo-relative paths of modules importing :data:`FROZEN_PACKAGE`."""
    found: set[str] = set()
    for relative in _searched_python_files():
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        if FROZEN_PACKAGE.split(".")[0] not in source:
            continue
        module_parts = (*relative.parts[:-1], relative.stem)
        for target in _imported_modules(ast.parse(source), module_parts=module_parts):
            if target == FROZEN_PACKAGE or target.startswith(f"{FROZEN_PACKAGE}."):
                found.add(relative.as_posix())
                break
    return found


def unfrozen_adopters(measured: set[str], baseline: frozenset[str]) -> set[str]:
    """Adopters present in ``measured`` that the freeze baseline does not carry."""
    return measured - baseline


def theme_path_readers() -> set[str]:
    """Tooling naming ``polylogue/ui/theme.py`` as a filesystem path.

    Same filesystem walk as :func:`ui_importers`, for the same reason: a
    tracked-only census does not see a new reader until it is staged.
    """
    return {
        relative.as_posix()
        for relative in _searched_python_files()
        if _THEME_PATH_TOKEN in (REPO_ROOT / relative).read_text(encoding="utf-8")
    }


def test_ui_importer_set_never_grows() -> None:
    """No module may start importing ``polylogue.ui`` while the freeze holds."""
    new = unfrozen_adopters(ui_importers(), FROZEN_IMPORTERS)
    assert not new, (
        "these modules import polylogue.ui but are not in the polylogue-4wqi2 freeze "
        "baseline:\n  " + "\n  ".join(sorted(new)) + "\n"
        "polylogue/ui/ is under an adoption freeze until polylogue-4p1 rules on "
        "presentation ownership: the importer set may shrink, never grow. Consume "
        "polylogue.rendering or the surface contracts instead, or land the 4p1 ruling "
        "and retire this ratchet with it."
    )


def test_ui_theme_path_reader_set_never_grows() -> None:
    """No new tooling may load ``polylogue/ui/theme.py`` by path."""
    new = unfrozen_adopters(theme_path_readers(), FROZEN_THEME_PATH_READERS)
    assert not new, (
        f"these files name {_THEME_PATH_TOKEN} but are not in the freeze baseline:\n  "
        + "\n  ".join(sorted(new))
        + "\nLoading the theme module by path is adoption an import census cannot see, "
        "and it is frozen with the importers."
    )


def test_the_freeze_allows_an_adopter_to_be_removed() -> None:
    """Migrating an adopter away must not fail the ratchet.

    A ratchet that rejected every difference would also reject the migration
    it exists to encourage, and would be indistinguishable from a blanket
    refusal to change the tree.
    """
    for dropped in sorted(FROZEN_IMPORTERS):
        shrunk = set(FROZEN_IMPORTERS) - {dropped}
        assert unfrozen_adopters(shrunk, FROZEN_IMPORTERS) == set()
    assert unfrozen_adopters(set(), FROZEN_IMPORTERS) == set()


def test_the_freeze_still_has_something_to_protect() -> None:
    """The census must really see the adopters, not silently find nothing."""
    measured = ui_importers()
    assert measured == set(FROZEN_IMPORTERS), (
        "the measured polylogue.ui importer set drifted from the recorded baseline; "
        f"measured={sorted(measured)} baseline={sorted(FROZEN_IMPORTERS)}. A shrink is "
        "expected and welcome -- update FROZEN_IMPORTERS in the same change that "
        "migrates the adopter."
    )
    assert theme_path_readers() == set(FROZEN_THEME_PATH_READERS)
    # A module reached only through a lazy function-body import must still be
    # counted, or the census would under-report the real migration cost.
    assert "polylogue/cli/click_app.py" in measured


def test_prose_cross_reference_is_not_an_adopter() -> None:
    """``polylogue/config.py`` names the theme resolver in prose and imports nothing.

    The recorded nine-importer baseline counted this docstring as adoption.
    It is not: ``polylogue/ui/theme.py`` imports ``polylogue.config``, so
    treating the mention as an edge would invert the real dependency.
    """
    config = (REPO_ROOT / "polylogue/config.py").read_text(encoding="utf-8")
    assert "polylogue.ui.theme.resolve_theme_mode" in config
    assert "polylogue/config.py" not in ui_importers()
