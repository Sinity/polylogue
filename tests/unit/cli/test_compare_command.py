"""``polylogue compare`` real production route (rxdo.9.6/.9.7/.9.11/.9.12).

Before this, ``blind_items``/``BlindingReceipt``, ``ClaimWithControls``,
``compute_calibration``, and the storage chokepoint
``upsert_comparative_judgment_assertion`` had zero production callers -- only
their own unit tests and, for ``blind_items``, an internal caller
(``ElicitationSession``) that itself had no production caller either. This
test exercises the real CLI command against a real archive: no mocked
facade, no test double for the storage layer.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from polylogue.cli.click_app import cli


def test_compare_without_verdict_prints_blinded_pair_and_records_nothing(
    cli_workspace: dict[str, Path],
) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "compare",
            "--left",
            "session:codex:a",
            "--left-field",
            "model=gpt-5",
            "--right",
            "session:codex:b",
            "--right-field",
            "model=claude-sonnet-5",
            "--dimension",
            "quality",
            "--rubric",
            "quality-v1",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert len(body["items"]) == 2
    # model is a masked provenance field -- must not leak into the blinded view.
    for item in body["items"]:
        assert "model" not in item["visible_fields"]
        assert "ref" in item["visible_fields"]
    assert body["receipt"]["revealed_at_ms"] is None


def test_compare_with_verdict_records_and_is_readable_via_calibration(
    cli_workspace: dict[str, Path],
) -> None:
    runner = CliRunner()
    record = runner.invoke(
        cli,
        [
            "compare",
            "--left",
            "session:codex:a",
            "--right",
            "session:codex:b",
            "--dimension",
            "quality",
            "--rubric",
            "quality-v1",
            "--verdict",
            "prefer_left",
            "--actor-ref",
            "agent:worker",
            "--exec-context-id",
            "ctx:1",
            "--json",
        ],
    )
    assert record.exit_code == 0, record.output
    recorded = json.loads(record.output)
    assert recorded["verdict"] == "prefer_left"
    assert recorded["revealed"]["left"]["ref"] == "session:codex:a"

    # A second recording of the same operator (gold) judgment on the same
    # comparison lets --calibration compute a real agreement rate against it.
    gold = runner.invoke(
        cli,
        [
            "compare",
            "--left",
            "session:codex:a",
            "--right",
            "session:codex:b",
            "--dimension",
            "quality",
            "--rubric",
            "quality-v1",
            "--verdict",
            "prefer_left",
            "--actor-ref",
            "user:local",
            "--exec-context-id",
            "ctx:gold",
            "--json",
        ],
    )
    assert gold.exit_code == 0, gold.output

    calibration = runner.invoke(
        cli,
        ["compare", "--calibration", "--gold-actor", "user:local", "--json"],
    )
    assert calibration.exit_code == 0, calibration.output
    reports = json.loads(calibration.output)
    worker_reports = [r for r in reports if r["actor_ref"] == "agent:worker"]
    assert len(worker_reports) == 1
    assert worker_reports[0]["agreement_rate"] == 1.0
    assert worker_reports[0]["n_gold_overlap"] == 1
