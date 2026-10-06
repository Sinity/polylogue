"""Named analyze projections render the format inherited from the root query."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from polylogue import Polylogue
from polylogue.cli.click_app import cli
from polylogue.surfaces.outcome import decide_outcome

# The CLI decides every named projection's terminal outcome from the bundle
# (``postmortem_outcome``/``portfolio_outcome``, #5855); the stand-in bundle
# carries no coverage, so the outcome owner is replaced with a completed one.
_COMPLETED = decide_outcome(matched=1)


@pytest.mark.parametrize("projection", ["postmortem", "portfolio"])
def test_named_analysis_renders_inherited_json(projection: str) -> None:
    """Fails if the named projection renders from its local ``None`` format instead of the root JSON."""
    with (
        patch.object(Polylogue, f"{projection}_bundle", new_callable=AsyncMock, return_value=object()),
        patch(f"polylogue.analysis.{projection}.{projection}_outcome", return_value=_COMPLETED),
        patch("polylogue.surfaces.payloads.model_json_document", return_value={"fixture": "w12"}),
    ):
        result = CliRunner().invoke(cli, ["--format", "json", "find", "repo:w12", "then", "analyze", projection])
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["status"] == "ok"
    assert "w12" in result.output


def test_named_cost_outlook_renders_inherited_json() -> None:
    """Fails if a no-window outlook prints plain text despite the inherited root JSON format."""
    with patch.object(Polylogue, "cost_outlook", new_callable=AsyncMock, return_value=None):
        result = CliRunner().invoke(
            cli,
            ["--format", "json", "find", "repo:w12", "then", "analyze", "cost-outlook", "--plan", "claude-pro"],
        )
    assert result.exit_code == 0, result.output
    document = json.loads(result.output)
    assert document["outlook"] is None
    assert "availability" in document


def test_named_analysis_local_format_overrides_inherited_json() -> None:
    """Fails if the inherited root format overrides an explicit local ``--format``."""
    with (
        patch.object(Polylogue, "postmortem_bundle", new_callable=AsyncMock, return_value=object()),
        patch("polylogue.analysis.postmortem.postmortem_outcome", return_value=_COMPLETED),
        patch("polylogue.analysis.postmortem.render_postmortem_markdown", return_value="# W12 explicit markdown"),
    ):
        result = CliRunner().invoke(
            cli,
            ["--format", "json", "find", "repo:w12", "then", "analyze", "postmortem", "--format", "markdown"],
        )
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "# W12 explicit markdown"
