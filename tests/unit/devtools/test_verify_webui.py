"""Regression coverage for the ``devtools gate webui`` route (polylogue-hohj7)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from devtools import repo_root
from devtools.gate import GATES_BY_NAME
from devtools.verify_webui import CHECK_SCRIPT_CLAIMS, check_script_steps, main


def test_missing_node_runtime_is_a_blocked_environment_not_a_green_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No Node runtime must exit non-zero with a blocked-env receipt.

    CI carries no Node runtime, so this is the branch it actually takes.
    Anti-vacuity: mutating the ``return 2`` in ``verify_webui.main`` to ``0``
    launders a missing runtime into a green receipt and turns this red.
    """

    def _no_node(*args: object, **kwargs: object) -> object:
        raise OSError(2, "No such file or directory: 'npm'")

    monkeypatch.setattr(subprocess, "run", _no_node)

    exit_code = main(["--json"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "blocked-env"
    assert payload["returncode"] is None


def test_gate_claim_is_bound_to_the_package_check_script() -> None:
    """The gate's claimed check categories cover exactly the real check steps.

    Anti-vacuity: removing ``&& npm run test`` (or any other step) from
    ``webui/package.json``'s ``check`` script, or adding a step no claimed
    category owns, turns this red instead of leaving the gate description
    asserting coverage the script no longer composes.
    """
    claimed = {step for steps in CHECK_SCRIPT_CLAIMS.values() for step in steps}
    assert claimed == set(check_script_steps(repo_root()))

    description = GATES_BY_NAME["webui"].description.lower()
    for category in CHECK_SCRIPT_CLAIMS:
        assert category in description, f"gate description no longer claims {category}"


def test_check_script_steps_reads_the_declared_script(tmp_path: Path) -> None:
    """The step reader parses the ``&&`` composition, not a hard-coded list."""
    (tmp_path / "webui").mkdir()
    (tmp_path / "webui" / "package.json").write_text(
        json.dumps({"scripts": {"check": "npm run lint && npm run test && echo done"}}),
        encoding="utf-8",
    )
    assert check_script_steps(tmp_path) == ("lint", "test")
