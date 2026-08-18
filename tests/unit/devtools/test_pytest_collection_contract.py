"""Soundness contract for the testmon environment digest.

`devtools/verify.py` is deliberately NOT a digest input, because hashing a
3,500-line orchestrator meant a comment there discarded every dependency graph.
That split is only safe while every collection-affecting value in the managed
pytest command actually comes from `pytest_collection_contract`, which IS
hashed. A marker, plugin or ini override written directly into verify.py would
escape the digest, and affected selection would then silently skip tests -- the
one failure mode worse than a slow bootstrap.

These are behavioural assertions on the command that gets built, not a text
scan of the source: the invariant is "the command's collection surface equals
the declared contract", which is what actually has to hold.
"""

from __future__ import annotations

from devtools.pytest_collection_contract import (
    CLOSED_WORLD_COLLECTION_ARGS,
    IGNORED_COLLECTION_ARGS,
    MANAGED_PLUGIN_ARGS,
    PARALLEL_MARKER_EXPRESSION,
    PROGRESS_PLUGIN_NAME,
    SERIAL_MARKER_EXPRESSION,
)
from devtools.verify import _native_pytest_steps


def _commands() -> dict[str, list[str]]:
    steps = _native_pytest_steps(
        testmon_mode="affected",
        testmon_environment="env-under-test",
        parallel_worker_args=["-n", "4"],
    )
    return dict(steps)


def test_every_declared_plugin_reaches_the_built_command() -> None:
    for label, cmd in _commands().items():
        joined = " ".join(cmd)
        for argument in MANAGED_PLUGIN_ARGS:
            assert argument in cmd, f"{label} is missing declared plugin arg {argument!r}"
        assert PROGRESS_PLUGIN_NAME in joined, f"{label} does not load the progress plugin"


def test_closed_world_collection_args_reach_the_built_command() -> None:
    for label, cmd in _commands().items():
        for argument in (*CLOSED_WORLD_COLLECTION_ARGS, *IGNORED_COLLECTION_ARGS):
            assert argument in cmd, f"{label} is missing collection arg {argument!r}"


def test_the_two_lanes_use_the_declared_marker_expressions() -> None:
    """The lanes partition the corpus; the expressions that split it are digest
    inputs precisely because changing them changes what each lane collects."""
    commands = _commands()
    # The FIRST "-m" is `python -m pytest`; the marker flag is the last one.
    markers = {label: cmd[len(cmd) - cmd[::-1].index("-m")] for label, cmd in commands.items() if "-m" in cmd}

    assert PARALLEL_MARKER_EXPRESSION in markers.values()
    assert SERIAL_MARKER_EXPRESSION in markers.values()
    assert set(markers.values()) == {PARALLEL_MARKER_EXPRESSION, SERIAL_MARKER_EXPRESSION}, (
        "a lane is collecting under an expression the digest does not cover"
    )


def test_marker_expressions_are_complementary() -> None:
    """Together the lanes must cover the corpus, or the union silently omits tests."""
    assert SERIAL_MARKER_EXPRESSION in PARALLEL_MARKER_EXPRESSION
    assert f"not {SERIAL_MARKER_EXPRESSION}" == PARALLEL_MARKER_EXPRESSION
