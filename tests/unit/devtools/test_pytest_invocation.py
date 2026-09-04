"""The managed pytest command must collect exactly what the contract declares.

Anti-vacuity: a plugin or ini override written into the command builder
instead of `pytest_invocation` makes a run collect something the declared
corpus does not describe, and these assertions go red.
"""

from __future__ import annotations

from devtools.pytest_invocation import (
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    MANAGED_PLUGIN_ARGS,
    PROGRESS_PLUGIN_NAME,
)
from devtools.verify import _pytest_steps


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


def test_the_default_tier_selects_and_the_all_tier_drops_testmon() -> None:
    """Affected verification selects from testmon; `--all` runs every test.

    Anti-vacuity: adding testmon to the complete route recreates dependency
    state in every worker without changing the selected corpus.
    """
    affected = _command("affected")
    complete = _command("all")

    assert "--testmon" in affected and "--testmon-forceselect" in affected
    assert "--testmon-noselect" not in affected
    assert "--testmon" not in complete
    assert "pytest-testmon" not in complete


def test_the_corpus_runs_as_one_unpartitioned_collection() -> None:
    """testmon drops every recorded test a run did not collect, so a sharded
    complete run would keep only its last shard's edges."""
    command = _command("all")

    # The first `-m` is `python -m pytest`; a second one would be a marker
    # expression, which partitions the collection.
    assert command[3:].count("-m") == 0
