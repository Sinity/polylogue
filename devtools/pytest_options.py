"""Which arguments after a pytest option are its value, as pytest decides it.

``devtools test`` reads the caller's pytest arguments twice: to count the
modules a selection names (sizing it for xdist) and to carry execution options
into a failure rerun. Both need to know whether the argument after an option
is that option's value or a path. A hand-kept table of value-taking options
misses options (``--show-capture no``, ``--assert plain``), and a missed option
turns its value into a path operand. The table here is read from pytest's own
argument parser for this checkout, with its installed plugins, the devtools
plugins, and the suite's conftest options registered.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from devtools.pytest_invocation import DEVTOOLS_PLUGIN_NAMES, SUITE_COST_PLUGIN_NAME
from devtools.toolchain import venv_python

#: The checkout whose interpreter and suite define the table: this one.
_CHECKOUT = Path(__file__).resolve().parents[1]

#: Runs in the checkout's interpreter; prints ``{option: nargs}`` as JSON.
#: ``tests/benchmarks`` is named so its conftest registers its options too.
_PROBE = """
import json, sys
from _pytest.config import _prepareconfig
config = _prepareconfig(sys.argv[1:])
arity = {}
for action in config._parser.optparser._actions:
    for option in action.option_strings:
        arity[option] = action.nargs
print(json.dumps(arity))
"""


class PytestOptionTableError(RuntimeError):
    """Pytest's option table could not be read, so arguments cannot be classified."""


@functools.cache
def pytest_option_nargs() -> Mapping[str, object]:
    """``{option: argparse nargs}`` for every option pytest accepts in this checkout.

    Fails closed: a caller that cannot tell a value from a path must not guess.
    """
    plugins = [argument for name in (*DEVTOOLS_PLUGIN_NAMES, SUITE_COST_PLUGIN_NAME) for argument in ("-p", name)]
    # Autoloaded plugins are read too: a managed run loads its plugins by name,
    # and the superset only makes more options known.
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTEST_ADDOPTS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    }
    result = subprocess.run(
        [venv_python(root=_CHECKOUT), "-c", _PROBE, *plugins, "tests", "tests/benchmarks"],
        cwd=_CHECKOUT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PytestOptionTableError(
            f"reading pytest's option table failed (exit {result.returncode}): {result.stderr.strip()[-2000:]}"
        )
    try:
        table = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise PytestOptionTableError(f"pytest's option table was unreadable: {result.stdout[-2000:]!r}") from exc
    if not isinstance(table, dict):
        raise PytestOptionTableError("pytest's option table was not a mapping")
    return table


def takes_value(option: str) -> bool:
    """Whether ``option`` (without an attached value) accepts a value."""
    return pytest_option_nargs().get(option, 0) != 0


def operand_count(arguments: Sequence[str], index: int) -> int:
    """How many arguments after ``arguments[index]`` pytest reads as its value.

    Follows argparse: ``nargs=None`` takes the next argument, ``?`` takes it
    unless it is an option, ``+``/``*`` take every following non-option. An
    attached value (``--tb=short``, ``-kexpr``) leaves the next argument alone,
    and an option pytest does not know takes none (pytest rejects it anyway).
    """
    argument = arguments[index]
    if not argument.startswith("-") or argument == "--":
        return 0
    if argument.startswith("--") and "=" in argument:
        return 0
    if not argument.startswith("--") and len(argument) > 2:
        return 0
    nargs = pytest_option_nargs().get(argument, 0)
    following = list(arguments[index + 1 :])
    if nargs is None:
        return min(1, len(following))
    if nargs == "?":
        return 1 if following and not following[0].startswith("-") else 0
    if isinstance(nargs, int):
        return min(nargs, len(following))
    count = 0
    for candidate in following:
        if candidate.startswith("-"):
            break
        count += 1
    return count


def short_options_with_value() -> frozenset[str]:
    """Single-letter options that take a value, which may be attached (``-n8``)."""
    return frozenset(
        option
        for option, nargs in pytest_option_nargs().items()
        if len(option) == 2 and option.startswith("-") and nargs != 0
    )


__all__ = [
    "PytestOptionTableError",
    "operand_count",
    "pytest_option_nargs",
    "short_options_with_value",
    "takes_value",
]
