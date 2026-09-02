"""The parts of the managed pytest invocation that decide WHAT gets collected.

Every collection-affecting value of the managed pytest command -- markers,
plugins, ini overrides, collection roots, ignored paths -- is declared here so
the lanes cannot drift apart from what the corpus is defined to be. Change it
here, not in the callers that assemble the command.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CLEAR_CONFIGURED_ADDOPTS",
    "CLOSED_WORLD_COLLECTION_ARGS",
    "IGNORED_COLLECTION_ARGS",
    "MANAGED_PLUGIN_ARGS",
    "MANAGED_PLUGIN_NAMES",
    "PROGRESS_PLUGIN_NAME",
]

#: Neutralize any addopts configured in pyproject so the invocation is closed.
CLEAR_CONFIGURED_ADDOPTS: Final = "--override-ini=addopts="

#: The progress plugin is loaded by module path rather than entry-point name.
PROGRESS_PLUGIN_NAME: Final = "devtools.pytest_progress_plugin"

#: Plugins loaded explicitly, because autoload is disabled for reproducibility.
#: Adding or removing one changes which hooks run during collection.
MANAGED_PLUGIN_NAMES: Final[tuple[str, ...]] = (
    "anyio",
    "asyncio",
    "hypothesispytest",
    "benchmark",
    "pytest_cov",
    "pytest_jsonreport",
    "randomly",
    "syrupy",
    "timeout",
    "xdist",
    "pytest-testmon",
)

MANAGED_PLUGIN_ARGS: Final[tuple[str, ...]] = tuple(
    argument for name in MANAGED_PLUGIN_NAMES for argument in ("-p", name)
)

#: Ini overrides plus the collection root. These define the corpus exactly.
CLOSED_WORLD_COLLECTION_ARGS: Final[tuple[str, ...]] = (
    CLEAR_CONFIGURED_ADDOPTS,
    "--override-ini=python_files=test_*.py *_test.py fuzz_*.py",
    "--override-ini=python_classes=Test",
    "--override-ini=python_functions=test",
    "--override-ini=norecursedirs=",
    "tests",
)

#: Benchmarks are excluded from correctness runs.
IGNORED_COLLECTION_ARGS: Final[tuple[str, ...]] = ("--ignore=tests/benchmarks",)
