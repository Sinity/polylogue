"""Every registered gate's declared argv satisfies its module's required arguments.

A gate whose registration omits an argument its module marks ``required=True``
exits 2 on argument parsing on every invocation, and the registry still
advertises it through ``gate --list``.  ``schema-inference-gate`` shipped that
way: registered with no argv against a parser requiring ``--archive-root`` and
``--receipt``.

Anti-vacuity: re-registering any gate without an option its module requires --
or adding a ``required=True`` argument to a gated module without extending the
registration -- makes ``test_every_gate_argv_supplies_its_modules_required_arguments``
red.  The assertion is not vacuous while
``test_at_least_one_gate_module_declares_a_required_argument`` holds, which
proves the AST scan actually finds required arguments in this tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from devtools.gate import GATES, Gate

ROOT = Path(__file__).resolve().parents[3]


def _module_source(gate: Gate) -> Path | None:
    """The source file the gate executes, when it is an in-repo module."""
    if gate.kind == "module":
        dotted = gate.args[0]
    elif gate.kind == "devtools":
        return ROOT / "devtools" / "__main__.py"
    else:
        return None
    path = ROOT / Path(*dotted.split("."))
    for candidate in (path.with_suffix(".py"), path / "__main__.py"):
        if candidate.is_file():
            return candidate
    return None


def _required_option_strings(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    required: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        is_required = any(
            keyword.arg == "required" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            for keyword in node.keywords
        )
        if not is_required:
            continue
        options = {
            argument.value
            for argument in node.args
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value.startswith("-")
        }
        if options:
            required.add(min(options, key=len) if len(options) == 1 else max(options, key=len))
    return required


_MODULE_GATES = [gate for gate in GATES if _module_source(gate) is not None]


def test_at_least_one_gate_module_declares_a_required_argument() -> None:
    """The AST scan really finds ``required=True`` arguments.

    A scan that silently found nothing would make the parametrized assertion
    below pass for every possible registration, including the broken one.
    """
    assert _MODULE_GATES
    assert _required_option_strings(ROOT / "tests/unit/devtools/_required_argument_probe.py") == {"--needed"}


@pytest.mark.parametrize("gate", _MODULE_GATES, ids=lambda gate: gate.name)
def test_every_gate_argv_supplies_its_modules_required_arguments(gate: Gate) -> None:
    source = _module_source(gate)
    assert source is not None
    declared = set(gate.args[1:] if gate.kind == "module" else gate.args)
    missing = sorted(option for option in _required_option_strings(source) if option not in declared)
    assert not missing, (
        f"gate {gate.name!r} runs {source} with argv {list(gate.args)}, "
        f"but that module requires {missing}; it would exit 2 on argument parsing"
    )
