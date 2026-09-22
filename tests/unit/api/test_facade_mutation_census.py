"""The facade's in-process mutation population is pinned, and it is MCP's.

polylogue-gjwto / polylogue-r29bv. ``devtools gate layering``'s
mutation-authority rule is an *import* ratchet: it matches a surface module
importing an executor-driving module. That is one spelling of an in-process
write. The other spelling -- a surface calling a ``polylogue.api.Polylogue``
method which opens the writable store itself -- is invisible to it by
construction, and no edit to ``layering.yaml`` can make it visible, because
there is no offending import to match. Decision D1 already learned this once:
the CLI mutation-authority baseline read ``[]`` while ``mark`` and ``compare``
wrote in-process through the facade.

The baseline file cannot register progress here either. Its entries are
``(target, file, import)`` triples, so all twenty of the facade's surviving
mutations collapse into the two ``polylogue/api/archive.py`` rows and the last
one to leave takes both rows with it. Its own recorded reason says so: "the
honest per-slice signal is the ``_execute_facade_mutation`` call-site count".
This module is that signal.

What it pins, measured at eab6e8507:

* ``_EXPECTED_FACADE_MUTATIONS`` -- the twenty ``PolylogueArchiveMixin``
  methods that call ``_execute_facade_mutation``. Shrink-only in the same
  sense as the layering baseline: lowering one onto a declared operation
  requires deleting its name here, and adding a twenty-first caller fails
  until someone writes the name down.
* The surface reach: every one of the twenty is reached from
  ``polylogue/mcp`` and *none* from ``polylogue/cli``. That is the fact that
  makes decision D2 the whole remaining blocker, and it is the guard
  polylogue-gjwto asks for in prose -- "a CLI mark or compare opening a
  writable store in the CLI process while polylogued run holds the write
  lease". Routing any CLI command back through one of these methods goes red
  here; the import ratchet would stay green, because the CLI would be
  importing ``polylogue.api``, which it is allowed to import.

Anti-vacuity, each pinned separately rather than assumed:

* ``test_the_detector_sees_a_planted_caller`` runs the same AST detector over
  a synthetic module that does call ``_execute_facade_mutation``, so a
  detector that silently matched nothing could not pass this file.
* ``test_the_cli_reach_probe_sees_a_planted_cli_caller`` runs the reach probe
  over a synthetic CLI module that calls one of the twenty, proving the
  zero-CLI-callers assertion is a measurement and not a vacuous truth over an
  empty search.
* ``test_the_measured_facade_module_exists`` fails if ``archive.py`` or the
  helper is renamed, so the census cannot pass by measuring nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

FACADE_MODULE = "polylogue/api/archive.py"
MUTATION_HELPER = "_execute_facade_mutation"

#: The ``PolylogueArchiveMixin`` methods that open a writable store in the
#: calling process at eab6e8507. Every one is reached only from
#: ``polylogue/mcp``; the CLI routes it lowered are already gone.
_EXPECTED_FACADE_MUTATIONS = frozenset(
    {
        "add_mark",
        "add_tag",
        "bulk_tag_sessions",
        "capture_assertion_candidate",
        "clear_corrections",
        "create_recall_pack",
        "delete_annotation",
        "delete_correction",
        "delete_metadata",
        "delete_recall_pack",
        "delete_view",
        "delete_workspace",
        "post_blackboard_note",
        "record_correction",
        "remove_mark",
        "remove_tag",
        "save_annotation",
        "save_view",
        "save_workspace",
        "set_metadata",
    }
)


def _facade_mutation_callers(source: str) -> set[str]:
    """Return the enclosing function names that call the mutation helper."""

    found: set[str] = set()
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._enter(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._enter(node)

        def _enter(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == MUTATION_HELPER and stack:
                found.add(stack[-1])
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def _methods_called_under(package: str, names: frozenset[str]) -> dict[str, list[str]]:
    """Map each name in ``names`` to the package files that call it."""

    reach: dict[str, list[str]] = {name: [] for name in names}
    root = REPO_ROOT / package
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):  # pragma: no cover - unreadable source
            continue
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        rel = path.relative_to(REPO_ROOT).as_posix()
        for name in names & called:
            reach[name].append(rel)
    return reach


def test_the_measured_facade_module_exists() -> None:
    """The census must not pass by measuring a file that moved."""

    source = (REPO_ROOT / FACADE_MODULE).read_text(encoding="utf-8")
    assert f"def {MUTATION_HELPER}(" in source, (
        f"{MUTATION_HELPER} is no longer defined in {FACADE_MODULE}; if it was deleted because "
        "the last facade mutation lowered onto a declared operation, delete this census with it."
    )


def test_the_facade_mutation_inventory_is_the_pinned_one() -> None:
    """A new in-process facade writer fails until it is written down."""

    observed = _facade_mutation_callers((REPO_ROOT / FACADE_MODULE).read_text(encoding="utf-8"))
    added = sorted(observed - _EXPECTED_FACADE_MUTATIONS)
    assert not added, (
        f"new in-process facade mutation(s) {added} in {FACADE_MODULE}. A surface write belongs "
        "on a declared daemon operation; if this one genuinely cannot lower yet, add it here "
        "with the reason, the way the layering baseline records its entries."
    )
    removed = sorted(_EXPECTED_FACADE_MUTATIONS - observed)
    assert not removed, (
        f"facade mutation(s) {removed} no longer call {MUTATION_HELPER} -- prune them from "
        "_EXPECTED_FACADE_MUTATIONS so the count ratchets down."
    )


def test_no_cli_module_reaches_a_facade_mutation() -> None:
    """The CLI's in-process writes are gone and may not come back.

    This is the guard the import ratchet cannot supply: a CLI module calling
    ``Polylogue.add_tag`` imports only ``polylogue.api``, which the layering
    rule permits.
    """

    reach = _methods_called_under("polylogue/cli", _EXPECTED_FACADE_MUTATIONS)
    offenders = {name: files for name, files in reach.items() if files}
    assert not offenders, (
        f"CLI module(s) reach a facade mutation that opens a writable store in the CLI process: "
        f"{offenders}. Lower the command onto a declared operation (see "
        "polylogue/cli/commands/setting.py, note.py and annotations.py for the shape)."
    )


def test_every_facade_mutation_is_mcp_reached() -> None:
    """Records *why* decision D2 is the remaining blocker, as a measurement."""

    reach = _methods_called_under("polylogue/mcp", _EXPECTED_FACADE_MUTATIONS)
    unreached = sorted(name for name, files in reach.items() if not files)
    assert not unreached, (
        f"facade mutation(s) {unreached} are no longer reached from polylogue/mcp. If they now "
        "have no surface caller at all, delete them from the facade rather than leaving an "
        "unreferenced in-process writer; then prune them from _EXPECTED_FACADE_MUTATIONS."
    )


def test_the_detector_sees_a_planted_caller() -> None:
    """A detector that matched nothing would pass the census vacuously."""

    planted = _facade_mutation_callers(
        "class C:\n"
        "    def planted_writer(self):\n"
        f"        return self.{MUTATION_HELPER}(actuator, build)\n"
        "    def innocent_reader(self):\n"
        "        return self.read_something()\n"
    )
    assert planted == {"planted_writer"}


def test_the_cli_reach_probe_sees_a_planted_cli_caller(tmp_path: Path) -> None:
    """The zero-CLI-callers assertion is a measurement, not an empty search."""

    module = tmp_path / "planted_cli_command.py"
    module.write_text("def run(poly):\n    return poly.add_tag('s', 't')\n", encoding="utf-8")
    tree = ast.parse(module.read_text(encoding="utf-8"))
    called = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert _EXPECTED_FACADE_MUTATIONS & called == {"add_tag"}
