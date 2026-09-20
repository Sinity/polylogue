from __future__ import annotations

import runpy
from typing import Any
from unittest.mock import patch

import pytest


def test_module_entrypoints_delegate_to_click_main() -> None:
    with patch("polylogue.cli.click_app.main") as click_main:
        runpy.run_module("polylogue.__main__", run_name="__main__")
        runpy.run_module("polylogue.cli.__main__", run_name="__main__")

    assert click_main.call_count == 2


#: Console scripts that deliberately do NOT establish the runtime contract,
#: with the reason. An entry here is a decision, not an oversight: adding one
#: is the deliberate act this test exists to force.
RUNTIME_CONTRACT_EXEMPT: dict[str, str] = {
    "polylogue-hook": (
        "records one harness hook event without loading the archive runtime; "
        "the 183ms native-extension probe would be charged to every hook fire, "
        "and the daemon ingests the sidecar spool under the contracted interpreter"
    ),
}


def _console_scripts() -> dict[str, str]:
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parents[2]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return {name: str(target) for name, target in payload["project"]["scripts"].items()}


def _entrypoint_callable(target: str) -> Any:
    import importlib

    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


@pytest.mark.parametrize("script", sorted(set(_console_scripts()) - set(RUNTIME_CONTRACT_EXEMPT)))
def test_every_console_script_refuses_an_unsupported_interpreter(script: str) -> None:
    """Each packaged entrypoint establishes the one supported interpreter.

    ``flake.nix`` selects ``python314FreeThreading`` and `requires-python` is
    `>=3.14`, but neither makes a *running* entrypoint check what it is
    running on. `require_free_threaded_runtime` is that check, and an
    entrypoint that omits it starts under a stock interpreter and reaches
    archive or network work anyway -- the silent fallback polylogue-xikl AC2
    forbids. This drives each script's real callable with the contract made
    unsatisfiable and requires the refusal to propagate before any work.

    Anti-vacuity: the enumeration comes from `[project.scripts]`, so adding a
    console script without the guard makes a NEW parametrization go red
    rather than passing unnoticed; emptying `RUNTIME_CONTRACT_EXEMPT` makes
    `polylogue-hook` red, so the exemption list cannot be silently widened to
    make the check vacuous either. Deleting the guard from any one entrypoint
    reddens exactly that script's case.
    """
    import click

    from polylogue.runtime import RuntimeContractError

    target = _console_scripts()[script]
    entrypoint = _entrypoint_callable(target)

    def refuse(*, consumer: str) -> None:
        raise RuntimeContractError(f"{consumer}: unsupported interpreter (test)")

    args: list[str] = []
    if isinstance(entrypoint, click.Group) and entrypoint.commands:
        # A group with ``no_args_is_help`` short-circuits on an empty argv
        # before its callback -- and the callback is where the contract is
        # established. Name a real subcommand so the group callback runs.
        args = [sorted(entrypoint.commands)[0], "--help"]

    with patch("polylogue.runtime.require_free_threaded_runtime", refuse):
        with pytest.raises(RuntimeContractError):
            if isinstance(entrypoint, click.Command):
                entrypoint.main(args=args, standalone_mode=False)
            else:
                entrypoint()


def test_the_exempt_entrypoint_is_still_a_declared_console_script() -> None:
    """A stale exemption must not quietly excuse a script that was renamed."""
    assert set(RUNTIME_CONTRACT_EXEMPT) <= set(_console_scripts())
    assert all(reason.strip() for reason in RUNTIME_CONTRACT_EXEMPT.values())
