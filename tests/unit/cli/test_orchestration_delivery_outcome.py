"""Delivered orchestration rows retain their supplied terminal gaps."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import click
import pytest
from click.testing import CliRunner

from polylogue.cli.read_views.base import ReadViewInvocation
from polylogue.cli.read_views.orchestration import run_read_orchestration
from polylogue.cli.root_request import RootModeRequest
from polylogue.cli.shared.types import AppEnv


@pytest.mark.parametrize(("row_gap", "raw_gap"), [(True, False), (False, True), (True, True), (False, False)])
def test_orchestration_delivers_document_before_finishing_supplied_gaps(
    row_gap: bool, raw_gap: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload: dict[str, object] = {
        "version": 1,
        "outcome": "degraded" if row_gap else "ok",
        "session_id": "codex-session:example",
        "children": [],
        "topology": None,
        "launches": [],
        "model_segments": [],
        "bead_mentions": [],
        "rate_limits": [],
        "usage": {},
        "coverage": {},
        "gaps": ["usage_unmeasured"] if row_gap else [],
    }
    result: dict[str, object] = {"view": "orchestration", "payload": payload}
    if raw_gap:
        result["materialization"] = {
            "state": "populated_unconverged",
            "reason": "uncensused_raw",
            "raw_artifact_count": 2,
            "materialized_raw_artifact_count": 1,
            "raw_state_revision": "source-frame-7",
            "blocking_counts": {"uncensused_raw": 1},
        }
        result["outcome"] = {"state": "degraded", "reason": "archive_not_converged", "detail": {}}

    def dispatch(*args: object, **kwargs: object) -> tuple[dict[str, object], None]:
        return result, None

    monkeypatch.setattr("polylogue.cli.read_views.orchestration.dispatch_read", dispatch)

    @click.command()
    def command() -> None:
        run_read_orchestration(
            cast(AppEnv, SimpleNamespace(config=None)),
            RootModeRequest(params={}, query_terms=()),
            ReadViewInvocation(
                view="orchestration",
                session_id="codex-session:example",
                output_format="json",
                destination="stdout",
                out_path=None,
            ),
        )

    delivered = CliRunner().invoke(command)
    assert json.loads(delivered.stdout) == payload
    assert delivered.exit_code == (1 if row_gap or raw_gap else 0), delivered.exception
    if row_gap or raw_gap:
        outcome = json.loads(delivered.stderr)["outcome"]
        assert outcome["state"] == "degraded"
        if row_gap:
            assert (
                outcome["reason"] == "orchestration_evidence_gaps"
                or "orchestration_evidence_gaps" in outcome["detail"]["gaps"]
            )
            assert "usage_unmeasured" in outcome["detail"]["gaps"]
        if raw_gap:
            assert "archive_not_converged" in outcome["detail"]["gaps"]
            assert outcome["detail"]["archive_materialization"]["raw_state_revision"] == "source-frame-7"
    else:
        assert delivered.stderr == ""
