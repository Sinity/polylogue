from __future__ import annotations

import json

import pytest

from devtools.semantic_axis_evidence import (
    _reproducer_command,
    build_scale_observation,
    build_semantic_axis_evidence,
)
from polylogue.core.json import JSONDocument
from polylogue.core.outcomes import OutcomeStatus
from polylogue.scenarios import CorpusSourceKind


def test_semantic_axis_evidence_builds_growth_shape_and_trust() -> None:
    observations = [
        build_scale_observation(
            scale="small",
            metric="rebuild_wall_s",
            axis_stat_key="messages_count",
            metrics={"rebuild_wall_s": 1.0},
            db_stats={"messages_count": 100},
        ),
        build_scale_observation(
            scale="medium",
            metric="rebuild_wall_s",
            axis_stat_key="messages_count",
            metrics={"rebuild_wall_s": 2.1},
            db_stats={"messages_count": 200},
        ),
    ]

    envelope = build_semantic_axis_evidence(
        campaign="fts-rebuild",
        semantic_axis="messages",
        axis_stat_key="messages_count",
        metric="rebuild_wall_s",
        observations=observations,
        reproducer=("devtools semantic-axis-evidence --campaign fts-rebuild --axis messages",),
        environment={"python": "test", "platform": "test"},
        reviewed_at="2026-04-22T00:00:00+00:00",
    )

    assert envelope.status is OutcomeStatus.OK
    assert envelope.evidence["growth_shape"] == "approximately-linear"
    assert envelope.evidence["semantic_axis"] == "messages"
    assert envelope.evidence["axis_db_stat"] == "messages_count"
    assert envelope.evidence["metric"] == "rebuild_wall_s"
    scale_tiers = envelope.evidence["scale_tiers"]
    assert isinstance(scale_tiers, list)
    assert len(scale_tiers) == 2
    assert len(envelope.trust.input_fingerprint or "") == 64
    assert len(envelope.trust.environment_fingerprint or "") == 64
    assert envelope.trust.runner_version == "semantic-axis-evidence.v1"
    assert envelope.trust.origin == "semantic-axis-evidence"


def test_semantic_axis_evidence_reports_changed_growth_behavior_against_baseline() -> None:
    observations = [
        build_scale_observation(
            scale="small",
            metric="rebuild_wall_s",
            axis_stat_key="messages_count",
            metrics={"rebuild_wall_s": 1.0},
            db_stats={"messages_count": 100},
        ),
        build_scale_observation(
            scale="medium",
            metric="rebuild_wall_s",
            axis_stat_key="messages_count",
            metrics={"rebuild_wall_s": 5.0},
            db_stats={"messages_count": 200},
        ),
    ]
    baseline_payload: JSONDocument = {"evidence": {"growth_shape": "approximately-linear"}}

    envelope = build_semantic_axis_evidence(
        campaign="fts-rebuild",
        semantic_axis="messages",
        axis_stat_key="messages_count",
        metric="rebuild_wall_s",
        observations=observations,
        reproducer=("devtools semantic-axis-evidence --campaign fts-rebuild --axis messages",),
        environment={"python": "test", "platform": "test"},
        baseline_payload=baseline_payload,
        reviewed_at="2026-04-22T00:00:00+00:00",
    )

    assert envelope.status is OutcomeStatus.ERROR
    assert envelope.evidence["growth_shape"] == "superlinear"
    assert envelope.counterexample is not None
    assert envelope.evidence["baseline_comparison"] == {
        "compared": True,
        "baseline_growth_shape": "approximately-linear",
        "candidate_growth_shape": "superlinear",
        "changed_growth_behavior": True,
    }


def test_scale_observation_rejects_missing_axis_or_metric() -> None:
    with pytest.raises(ValueError, match="metric"):
        build_scale_observation(
            scale="small",
            metric="rebuild_wall_s",
            axis_stat_key="messages_count",
            metrics={},
            db_stats={"messages_count": 100},
        )

    with pytest.raises(ValueError, match="axis stat"):
        build_scale_observation(
            scale="small",
            metric="rebuild_wall_s",
            axis_stat_key="messages_count",
            metrics={"rebuild_wall_s": 1.0},
            db_stats={},
        )


def test_semantic_axis_evidence_payload_is_json_serializable() -> None:
    observation = build_scale_observation(
        scale="small",
        metric="rebuild_wall_s",
        axis_stat_key="messages_count",
        metrics={"rebuild_wall_s": 1.0},
        db_stats={"messages_count": 100},
    )
    envelope = build_semantic_axis_evidence(
        campaign="fts-rebuild",
        semantic_axis="messages",
        axis_stat_key="messages_count",
        metric="rebuild_wall_s",
        observations=[observation, observation],
        reproducer=("devtools semantic-axis-evidence --campaign fts-rebuild --axis messages",),
        environment={"python": "test", "platform": "test"},
        reviewed_at="2026-04-22T00:00:00+00:00",
    )

    json.dumps(envelope.to_payload(), sort_keys=True)


def test_reproducer_command_is_single_shell_command() -> None:
    reproducer = _reproducer_command(
        campaign="fts-rebuild",
        semantic_axis="messages",
        metric="rebuild_wall_s",
        scales=("small", "medium"),
        corpus_source=CorpusSourceKind.DEFAULT,
        seed=42,
    )

    assert reproducer == (
        "devtools semantic-axis-evidence --campaign fts-rebuild --axis messages --metric rebuild_wall_s "
        "--corpus-source default --seed 42 --scales small medium",
    )
