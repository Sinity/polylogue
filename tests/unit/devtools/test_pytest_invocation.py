"""The managed pytest command must collect exactly what the contract declares.

Anti-vacuity: a plugin or ini override written into the command builder
instead of `pytest_invocation` makes a run collect something the declared
corpus does not describe, and these assertions go red.
"""

from __future__ import annotations

import pytest

from devtools.pytest_invocation import (
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    MANAGED_PLUGIN_ARGS,
    PROGRESS_PLUGIN_NAME,
)
from devtools.verify import _pytest_steps, build_verify_steps
from devtools.worker_memory import (
    CONTROLLER_PEAK_MIB,
    CORPUS_MAX_WORKERS,
    MEMORY_HEADROOM_FRACTION,
    PYTEST_SLICE_MEMORY_HIGH_MIB,
    WORKER_PEAK_MIB,
)


def _command(selection: str) -> list[str]:
    steps = _pytest_steps(selection=selection, worker_args=[])
    assert len(steps) == 1, "the corpus must run as one collection"
    return steps[0][1]


def test_every_declared_plugin_reaches_the_built_command() -> None:
    command = _command("affected")
    for argument in MANAGED_PLUGIN_ARGS:
        assert argument in command, f"missing declared plugin arg {argument!r}"
    assert PROGRESS_PLUGIN_NAME in " ".join(command)


def test_closed_world_collection_args_reach_the_built_command() -> None:
    command = _command("affected")
    for argument in (*CLOSED_WORLD_COLLECTION_ARGS, *IGNORED_COLLECTION_ARGS):
        assert argument in command, f"missing collection arg {argument!r}"


def test_the_default_tier_selects_and_the_complete_tier_only_traces() -> None:
    """Affected verification selects from testmon; `--all` executes the whole
    collection and records what it traced, which is what the next affected run
    selects against.

    Anti-vacuity: dropping testmon from the complete route leaves the datafile
    describing whatever the last affected run happened to collect, and giving
    the complete route ``--testmon-forceselect`` makes it select instead of
    running the corpus.
    """
    affected = _command("affected")
    complete = _command("all")

    assert "--testmon" in affected and "--testmon-forceselect" in affected
    assert "--testmon-noselect" not in affected
    assert "--testmon" in complete and "--testmon-noselect" in complete
    assert "--testmon-forceselect" not in complete


def test_the_corpus_runs_as_one_unpartitioned_collection() -> None:
    """testmon drops every recorded test a run did not collect, so a sharded
    complete run would keep only its last shard's edges."""
    command = _command("all")

    # The first `-m` is `python -m pytest`; a second one would be a marker
    # expression, which partitions the collection.
    assert command[3:].count("-m") == 0


def test_the_managed_width_fits_the_pytest_pool_by_construction() -> None:
    """The corpus command cannot ask for more memory than its slice allows.

    Anti-vacuity: declare ``CORPUS_MAX_WORKERS`` as a literal wider than
    ``width_within(PYTEST_SLICE_MEMORY_HIGH_MIB)`` and this goes red -- a run
    that wide parks above ``memory.high``, crawls under allocation throttling,
    and holds the host's one pytest slot until systemd-oomd kills it.
    """
    command = build_verify_steps(quick=False, selection="all")[-1][1]
    workers = int(command[command.index("-n") + 1])

    assert workers >= 1
    assert workers * WORKER_PEAK_MIB + CONTROLLER_PEAK_MIB <= PYTEST_SLICE_MEMORY_HIGH_MIB * (
        1.0 - MEMORY_HEADROOM_FRACTION
    )


def test_a_wider_configured_width_is_reduced_to_what_the_pool_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anti-vacuity: honour the override unbounded and a run configured wider
    than its slice starts anyway and throttles instead of reporting.
    """
    monkeypatch.setenv("POLYLOGUE_PYTEST_WORKERS", "32")

    command = build_verify_steps(quick=False, selection="all")[-1][1]

    assert command[command.index("-n") + 1] == str(CORPUS_MAX_WORKERS)
