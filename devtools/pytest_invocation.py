"""The parts of the managed pytest invocation that decide WHAT gets collected.

Every collection-affecting value of the managed pytest command -- markers,
plugins, ini overrides, collection roots, ignored paths -- is declared here so
the lanes cannot drift apart from what the corpus is defined to be. Change it
here, not in the callers that assemble the command.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

__all__ = [
    "CLEAR_CONFIGURED_ADDOPTS",
    "CLOSED_WORLD_COLLECTION_ARGS",
    "DEVTOOLS_PLUGIN_ARGS",
    "DEVTOOLS_PLUGIN_NAMES",
    "devtools_plugin_args",
    "effective_hypothesis_profile",
    "IGNORED_COLLECTION_ARGS",
    "MANAGED_PLUGIN_ARGS",
    "managed_plugin_args",
    "MANAGED_PLUGIN_NAMES",
    "PROGRESS_PLUGIN_NAME",
    "REPORT_PLUGIN_ARGS",
    "STREAM_REPORT_PLUGIN_NAME",
    "TESTMON_RETENTION_PLUGIN_NAME",
    "SUITE_COST_PLUGIN_NAME",
]

#: Neutralize any addopts configured in pyproject so the invocation is closed.
CLEAR_CONFIGURED_ADDOPTS: Final = "--override-ini=addopts="

#: The repository's own plugins, loaded by module path rather than entry-point
#: name. The progress plugin writes the incremental ledgers, the report plugin
#: writes the JSON report a step is judged from, and the retention plugin frees
#: what testmon has already consumed on the controller.
PROGRESS_PLUGIN_NAME: Final = "devtools.pytest_progress_plugin"
STREAM_REPORT_PLUGIN_NAME: Final = "devtools.pytest_stream_report"
TESTMON_RETENTION_PLUGIN_NAME: Final = "devtools.pytest_testmon_retention"

DEVTOOLS_PLUGIN_NAMES: Final[tuple[str, ...]] = (
    PROGRESS_PLUGIN_NAME,
    STREAM_REPORT_PLUGIN_NAME,
    TESTMON_RETENTION_PLUGIN_NAME,
)

DEVTOOLS_PLUGIN_ARGS: Final[tuple[str, ...]] = tuple(
    argument for name in DEVTOOLS_PLUGIN_NAMES for argument in ("-p", name)
)


def devtools_plugin_args(*, testmon: bool) -> tuple[str, ...]:
    """Return repository plugins appropriate for this collection mode.

    The retention hook is coupled to testmon's controller-side batches. A
    focused selection does not trace or select through testmon, so loading the
    hook there would create a second testmon surface.
    """
    names = (
        DEVTOOLS_PLUGIN_NAMES
        if testmon
        else tuple(name for name in DEVTOOLS_PLUGIN_NAMES if name != TESTMON_RETENTION_PLUGIN_NAME)
    )
    return tuple(argument for name in names for argument in ("-p", name))


def effective_hypothesis_profile(argv: Sequence[str], env: Mapping[str, str], *, default: str) -> tuple[str, str]:
    """Return the effective profile and its source: CLI, environment, default.

    Hypothesis itself gives its command-line option precedence. Recording the
    same resolution in the managed receipt keeps that fact available after the
    process exits.
    """
    cli_profile: str | None = None
    for index, argument in enumerate(argv):
        if argument == "--hypothesis-profile" and index + 1 < len(argv):
            cli_profile = argv[index + 1]
        if argument.startswith("--hypothesis-profile="):
            cli_profile = argument.split("=", 1)[1]
    if cli_profile is not None:
        return cli_profile, "cli"
    configured = env.get("HYPOTHESIS_PROFILE", "").strip()
    return (configured, "environment") if configured else (default, "default")


#: A run that needs the report but none of the session-wide ledgers: the rerun
#: of failed tests writes its own report beside the step it adjudicates.
REPORT_PLUGIN_ARGS: Final[tuple[str, ...]] = ("-p", STREAM_REPORT_PLUGIN_NAME)

#: Archive-construction and write-cost receipt. Collects nothing and hooks no
#: collection stage; it is inert unless ``POLYLOGUE_SUITE_COST_DIR`` is set.
SUITE_COST_PLUGIN_NAME: Final = "devtools.pytest_suite_cost_plugin"

#: Plugins loaded explicitly, because autoload is disabled for reproducibility.
#: Adding or removing one changes which hooks run during collection.
MANAGED_PLUGIN_NAMES: Final[tuple[str, ...]] = (
    "anyio",
    "asyncio",
    "hypothesispytest",
    "benchmark",
    "pytest_cov",
    "randomly",
    "syrupy",
    "timeout",
    "xdist",
    "pytest-testmon",
)

MANAGED_PLUGIN_ARGS: Final[tuple[str, ...]] = tuple(
    argument for name in MANAGED_PLUGIN_NAMES for argument in ("-p", name)
)


def managed_plugin_args(*, testmon: bool, xdist: bool = True) -> tuple[str, ...]:
    """Return the explicit plugin profile for a managed pytest mode.

    A run that omits testmon writes no fingerprints, so it leaves the datafile
    exactly as it found it. Only a mode whose collection is not a corpus opts
    out of tracing.
    """
    names = MANAGED_PLUGIN_NAMES
    if not testmon:
        names = tuple(name for name in names if name != "pytest-testmon")
    if not xdist:
        names = tuple(name for name in names if name != "xdist")
    return tuple(argument for name in names for argument in ("-p", name))


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
