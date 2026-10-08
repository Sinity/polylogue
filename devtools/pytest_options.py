"""Which arguments after a pytest option are its value, as pytest decides it.

``devtools test`` reads the caller's pytest arguments to count the modules a
selection names (sizing it for xdist) and to expand short-option clusters.
Both need to know whether the argument after an option is that option's value
or a path. A hand-kept table of value-taking options
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
#: A ``-p`` plugin that cannot be imported by module name is left out: pytest
#: refuses such a run itself, and an entry-point name is autoloaded anyway.
_PROBE = """
import importlib.util, json, sys
from _pytest.config import _prepareconfig
def importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
arguments, index = [], 0
raw = sys.argv[1:]
while index < len(raw):
    if raw[index] == "-p" and index + 1 < len(raw):
        if importable(raw[index + 1]):
            arguments += raw[index : index + 2]
        index += 2
        continue
    arguments.append(raw[index])
    index += 1
config = _prepareconfig(arguments)
arity = {}
for action in config._parser.optparser._actions:
    for option in action.option_strings:
        arity[option] = action.nargs
print(json.dumps(arity))
"""


class PytestOptionTableError(RuntimeError):
    """Pytest's option table could not be read, so arguments cannot be classified."""


def caller_plugins(arguments: Sequence[str]) -> tuple[str, ...]:
    """The plugins a caller's ``-p name`` (``-pname``, ``-p=name``) loads early.

    ``-p no:name`` blocks a plugin rather than loading one, so it registers
    no options and is left out.
    """
    plugins: list[str] = []
    for index, argument in enumerate(arguments):
        if argument == "-p":
            name = arguments[index + 1] if index + 1 < len(arguments) else ""
        elif argument.startswith("-p") and not argument.startswith("--"):
            name = argument[2:].removeprefix("=")
        else:
            continue
        if name and not name.startswith("no:") and name not in plugins:
            plugins.append(name)
    return tuple(plugins)


@functools.cache
def pytest_option_nargs(plugins: tuple[str, ...] = ()) -> Mapping[str, object]:
    """``{option: argparse nargs}`` for every option pytest accepts in this checkout.

    ``plugins`` are the caller's early-loaded ``-p`` plugins
    (:func:`caller_plugins`): an option such a plugin registers is as real
    to pytest as a built-in one, so its arity must be read too.

    Fails closed: a caller that cannot tell a value from a path must not guess.
    """
    names = dict.fromkeys((*DEVTOOLS_PLUGIN_NAMES, SUITE_COST_PLUGIN_NAME, *plugins))
    plugins_args = [argument for name in names for argument in ("-p", name)]
    # Autoloaded plugins are read too: a managed run loads its plugins by name,
    # and the superset only makes more options known.
    env = {
        key: value
        for key, value in os.environ.items()
        # The same ambient pytest state the managed run removes
        # (``_normalize_managed_pytest_environment``): a probe that loads a
        # caller's ``PYTEST_PLUGINS`` fails where the admitted run would not.
        if key
        not in {
            "PYTEST_ADDOPTS",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
            "PYTEST_PLUGINS",
            "PYTEST_XDIST_WORKER",
            "PYTEST_CURRENT_TEST",
        }
    }
    result = subprocess.run(
        [venv_python(root=_CHECKOUT), "-c", _PROBE, *plugins_args, "tests", "tests/benchmarks"],
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


def takes_value(option: str, plugins: tuple[str, ...] = ()) -> bool:
    """Whether ``option`` (without an attached value) accepts a value."""
    return pytest_option_nargs(plugins).get(option, 0) != 0


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
    table = pytest_option_nargs(caller_plugins(arguments))
    if not argument.startswith("--") and len(argument) > 2 and argument not in table:
        # A clustered short-option run (``-qW``): walk it character by
        # character, as argparse does. A no-value option in the cluster is
        # consumed and skipped; the first value-taking option ends the
        # cluster, taking any remaining characters as its attached value, or
        # the next argument if none remain.
        for offset, letter in enumerate(argument[1:], start=1):
            option = f"-{letter}"
            option_nargs = table.get(option, 0)
            if option_nargs == 0:
                continue
            if offset < len(argument) - 1:
                # Characters remain after this option: they are its attached
                # value (``-Werror``), so nothing further is consumed.
                return 0
            argument = option
            break
        else:
            return 0
    nargs = table.get(argument, 0)
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


def split_short_cluster(argument: str, value_short: frozenset[str]) -> tuple[list[str], str | None, str | None]:
    """Decompose a short-option cluster as argparse reads it.

    Returns ``(flags, value_option, attached_value)``: the no-value options
    in order, then the first value-taking option (if any) with the rest of
    the cluster as its attached value -- ``""`` when the value is the next
    argument. ``-vn8`` is ``(["-v"], "-n", "8")``; ``-sv`` is
    ``(["-s", "-v"], None, None)``.
    """
    flags: list[str] = []
    for offset, letter in enumerate(argument[1:], start=1):
        option = f"-{letter}"
        if option in value_short:
            return flags, option, argument[offset + 1 :]
        flags.append(option)
    return flags, None, None


def expand_short_clusters(arguments: Sequence[str]) -> list[str]:
    """``arguments`` with each short-option cluster written as separate options.

    ``-vn8`` becomes ``-v -n 8`` and ``-vpno:xdist`` becomes ``-v -p no:xdist``,
    as argparse reads them. Option values, long options and everything after
    ``--`` are left as given. Pytest's option table is read only when a
    cluster is present.
    """
    expanded: list[str] = []
    value_short: frozenset[str] | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            expanded.extend(arguments[index:])
            break
        if argument.startswith("-") and not argument.startswith("--") and len(argument) > 2:
            if value_short is None:
                value_short = short_options_with_value(caller_plugins(arguments))
            flags, option, attached = split_short_cluster(argument, value_short)
            expanded.extend(flags)
            if option is not None:
                expanded.append(option)
                if attached:
                    expanded.append(attached.removeprefix("="))
                else:
                    # The value is the next argument, taken as is.
                    expanded.extend(arguments[index + 1 : index + 2])
                    index += 1
            index += 1
            continue
        span = 1 + operand_count(arguments, index) if argument.startswith("-") else 1
        expanded.extend(arguments[index : index + span])
        index += span
    return expanded


def short_options_with_value(plugins: tuple[str, ...] = ()) -> frozenset[str]:
    """Single-letter options that take a value, which may be attached (``-n8``)."""
    return frozenset(
        option
        for option, nargs in pytest_option_nargs(plugins).items()
        if len(option) == 2 and option.startswith("-") and nargs != 0
    )


def declared_testmon_environment(command: Sequence[str]) -> str | None:
    """The testmon environment a pytest ``command`` traces into, or ``None``.

    ``None`` means the command runs without testmon and writes no
    fingerprints. The last ``--testmon``/``--no-testmon`` (or ``-p no:testmon``)
    wins, and ``--testmon-env`` names the environment (``default`` otherwise).
    """
    arguments = list(command)
    arguments = arguments[arguments.index("pytest") + 1 :] if "pytest" in arguments else arguments
    enabled = False
    environment = "default"
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            break
        if argument == "--testmon":
            enabled = True
        elif argument in {"--no-testmon", "-pno:testmon", "-p=no:testmon"} or (
            argument == "-p" and index + 1 < len(arguments) and arguments[index + 1] == "no:testmon"
        ):
            enabled = False
        elif argument.startswith("--testmon-env="):
            environment = argument.split("=", 1)[1]
        elif argument == "--testmon-env" and index + 1 < len(arguments):
            environment = arguments[index + 1]
            index += 1
        index += 1
    return environment if enabled else None


__all__ = [
    "PytestOptionTableError",
    "caller_plugins",
    "declared_testmon_environment",
    "expand_short_clusters",
    "split_short_cluster",
    "operand_count",
    "pytest_option_nargs",
    "short_options_with_value",
    "takes_value",
]
