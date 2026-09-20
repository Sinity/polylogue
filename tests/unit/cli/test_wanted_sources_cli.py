"""The operator route to the frozen rebuild denominator (polylogue-co2iz).

Anti-vacuity for every test here: each one runs the production Click tree
(`ops maintenance wanted-sources`) against a real archive root and a real
configured source root.  Dropping the preflight call -- or letting the command
re-enumerate the roots instead of loading the receipt -- turns the tamper and
missing-receipt cases into exit 0, and the substituted-policy case into a
silent acceptance, which these assertions fail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli


@pytest.fixture
def configured_source_root(
    cli_workspace: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    """Declare one explicit, neutral local source root for the runtime."""
    root = tmp_path / "declared-source"
    root.mkdir()
    (root / "one.jsonl").write_text('{"kind": "synthetic"}\n', encoding="utf-8")
    (root / "two.jsonl").write_text('{"kind": "synthetic-two"}\n', encoding="utf-8")
    config = tmp_path / "polylogue.toml"
    config.write_text(f'[sources]\nroots = ["{root}"]\n', encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(config))
    return root


def _run(runner: CliRunner, *args: str) -> tuple[int, dict[str, Any]]:
    result = runner.invoke(cli, ["ops", "maintenance", "wanted-sources", *args, "--output-format", "json"])
    start = result.output.index("{")
    return result.exit_code, json.loads(result.output[start:])


def test_polylogue_co2iz_preflight_refuses_until_a_receipt_is_frozen(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    configured_source_root: Path,
) -> None:
    """No receipt is a refusal, and freezing one authorizes the same digests."""
    exit_code, refusal = _run(cli_runner)
    assert exit_code == 1
    assert refusal["outcome"] == "error"
    assert refusal["action"] == "preflight"
    assert "missing" in str(refusal["reason"])

    frozen_exit, frozen = _run(cli_runner, "--freeze")
    assert frozen_exit == 0
    assert frozen["complete"] is True
    assert frozen["item_count"] == 2
    assert frozen["blocker_count"] == 0
    assert frozen["byte_count"] > 0

    ok_exit, authorized = _run(cli_runner)
    assert ok_exit == 0
    assert authorized["outcome"] == "ok"
    assert authorized["receipt_sha256"] == frozen["receipt_sha256"]
    assert authorized["policy_identity"] == frozen["policy_identity"]
    assert authorized["declaration_sha256"] == frozen["declaration_sha256"]
    assert authorized["frontier_sha256"] == frozen["frontier_sha256"]
    assert authorized["item_count"] == 2


def test_polylogue_co2iz_preflight_refuses_a_tampered_receipt(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    configured_source_root: Path,
) -> None:
    """An edited denominator fails integrity instead of authorizing a rebuild."""
    assert _run(cli_runner, "--freeze")[0] == 0
    receipt_path = cli_workspace["archive_root"] / ".maintenance-state" / "wanted-sources" / "selected.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["item_count"] = 1
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code, refusal = _run(cli_runner)
    assert exit_code == 1
    assert refusal["outcome"] == "error"
    assert "integrity" in str(refusal["reason"]) or "denominator" in str(refusal["reason"])


def test_polylogue_co2iz_preflight_refuses_when_a_declared_root_disappears(
    cli_workspace: dict[str, Path],
    cli_runner: CliRunner,
    configured_source_root: Path,
) -> None:
    """A vanished declared root blocks the rebuild rather than shrinking it.

    This is the case a fresh source walk would silently absorb: the walk would
    simply enumerate nothing and report a smaller, self-consistent
    denominator.
    """
    assert _run(cli_runner, "--freeze")[0] == 0
    for child in configured_source_root.iterdir():
        child.unlink()
    configured_source_root.rmdir()

    exit_code, refusal = _run(cli_runner)
    assert exit_code == 1
    assert refusal["outcome"] == "error"
    assert "root is missing or unavailable" in str(refusal["reason"])
