from __future__ import annotations

import pytest

from polylogue.showcase.qa_runner_request import QAStage, build_qa_session_request


def test_build_qa_session_request_defaults_to_fresh_synthetic_full_run() -> None:
    request = build_qa_session_request(
        synthetic=True,
        source_names=None,
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage=None,
        skip_stages=(),
        workspace=None,
        report_dir=None,
        verbose=False,
        fail_fast=False,
        tier_filter=None,
    )

    assert request.live is False
    assert request.fresh is True
    assert request.ingest is True
    assert request.skip_audit is False
    assert request.skip_proof is False
    assert request.skip_exercises is False
    assert request.skip_invariants is False


def test_build_qa_session_request_respects_live_source_fresh_workspace() -> None:
    request = build_qa_session_request(
        synthetic=False,
        source_names=("inbox",),
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage=None,
        skip_stages=(),
        workspace=None,
        report_dir=None,
        verbose=False,
        fail_fast=False,
        tier_filter=None,
    )

    assert request.live is True
    assert request.fresh is True
    assert request.ingest is True
    assert request.source_names == ("inbox",)


def test_build_qa_session_request_audit_only_skips_follow_on_work() -> None:
    request = build_qa_session_request(
        synthetic=True,
        source_names=None,
        fresh=None,
        ingest=None,
        regenerate_schemas=False,
        only_stage=QAStage.AUDIT,
        skip_stages=(),
        workspace=None,
        report_dir=None,
        verbose=False,
        fail_fast=False,
        tier_filter=None,
    )

    assert request.skip_audit is False
    assert request.skip_proof is True
    assert request.skip_exercises is True
    assert request.skip_invariants is True


def test_build_qa_session_request_rejects_conflicting_stage_options() -> None:
    with pytest.raises(ValueError, match="--only and --skip are mutually exclusive"):
        build_qa_session_request(
            synthetic=True,
            source_names=None,
            fresh=None,
            ingest=None,
            regenerate_schemas=False,
            only_stage=QAStage.AUDIT,
            skip_stages=(QAStage.EXERCISES,),
            workspace=None,
            report_dir=None,
            verbose=False,
            fail_fast=False,
            tier_filter=None,
        )
