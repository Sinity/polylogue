"""The parts of the managed pytest invocation that decide WHAT gets collected.

This module exists to separate two things that `devtools/verify.py` had fused,
at a cost measured in hours.

pytest-testmon identifies a dependency graph by an *environment digest*. The
digest must cover anything that changes collection semantics but that testmon
cannot observe by tracing files -- the installed distribution set, pytest
configuration, the plugin list, the marker expressions, the collection roots.
When one of those changes, every recorded fingerprint is suspect and the graph
must be rebuilt.

`devtools/verify.py` was itself a digest input, because it constructs that
invocation. But it is also a ~3,500-line orchestrator that formats output,
records receipts, prunes old runs and parses flags -- none of which touches
collection. The consequence was that editing a COMMENT in verify.py discarded
every testmon graph and forced a full-corpus bootstrap, roughly 9.5x a warm run
(2701s vs 285s, measured over 1233 recorded runs).

That was not theoretical. On the branch this module was written for, 16 of 132
commits touched a digest input, and most of them were fixes to the verification
harness itself -- including "stop tracker writes from destroying the testmon
graph" and "keep an interrupted bootstrap's graph instead of discarding it".
Every attempt to fix graph destruction destroyed the graph.

So the collection-affecting values live here, alone, and THIS module is the
digest input. Editing verify.py's logic no longer invalidates anything; changing
what pytest collects still does, because it must be changed here.

**If you change how the managed pytest command selects or collects tests --
markers, plugins, ini overrides, collection roots, ignored paths -- change it
HERE.** Adding such a value to verify.py directly would escape the digest and
make affected selection unsound, which is the one failure mode worse than a
slow bootstrap.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "CLEAR_CONFIGURED_ADDOPTS",
    "CLOSED_WORLD_COLLECTION_ARGS",
    "IGNORED_COLLECTION_ARGS",
    "MANAGED_PLUGIN_ARGS",
    "MANAGED_PLUGIN_NAMES",
    "PARALLEL_MARKER_EXPRESSION",
    "PROGRESS_PLUGIN_NAME",
    "SERIAL_MARKER_EXPRESSION",
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

#: The two lanes partition the corpus; together they must cover it.
PARALLEL_MARKER_EXPRESSION: Final = "not load_sensitive"
SERIAL_MARKER_EXPRESSION: Final = "load_sensitive"
