"""Typed request construction for QA command surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QASessionRequest:
    """Canonical request for one QA workflow run."""

    live: bool = False
    fresh: bool = True
    ingest: bool = False
    source_names: tuple[str, ...] | None = None
    regenerate_schemas: bool = False
    skip_audit: bool = False
    skip_proof: bool = False
    skip_exercises: bool = False
    skip_invariants: bool = False
    workspace_dir: Path | None = None
    workspace_env: dict[str, str] | None = None
    report_dir: Path | None = None
    provider: str | None = None
    verbose: bool = False
    fail_fast: bool = False
    tier_filter: int | None = None
    synthetic_count: int = 3

    @property
    def needs_workspace(self) -> bool:
        return not self.skip_exercises


def build_qa_session_request(
    *,
    synthetic: bool,
    source_names: tuple[str, ...] | None,
    fresh: bool | None,
    ingest: bool | None,
    regenerate_schemas: bool,
    only_stage: str | None,
    skip_stages: tuple[str, ...],
    workspace: Path | None,
    report_dir: Path | None,
    verbose: bool,
    fail_fast: bool,
    tier_filter: int | None,
) -> QASessionRequest:
    """Resolve CLI-facing QA options into the canonical request model."""
    if only_stage and skip_stages:
        raise ValueError("--only and --skip are mutually exclusive")

    live = not synthetic
    if source_names:
        live = True
        if fresh is None:
            fresh = True

    if fresh is None:
        fresh = not live

    if ingest is None:
        ingest = fresh

    run_audit = True
    run_exercises = True
    run_invariants = True
    if only_stage:
        run_audit = only_stage == "audit"
        run_exercises = only_stage == "exercises"
        run_invariants = only_stage == "invariants"
    else:
        if "audit" in skip_stages:
            run_audit = False
        if "exercises" in skip_stages:
            run_exercises = False
        if "invariants" in skip_stages:
            run_invariants = False
    run_proof = only_stage is None and run_audit

    return QASessionRequest(
        live=live,
        fresh=fresh,
        ingest=ingest,
        source_names=source_names,
        regenerate_schemas=regenerate_schemas,
        skip_audit=not run_audit,
        skip_proof=not run_proof,
        skip_exercises=not run_exercises,
        skip_invariants=not run_invariants,
        workspace_dir=workspace,
        report_dir=report_dir,
        verbose=verbose,
        fail_fast=fail_fast,
        tier_filter=tier_filter,
    )


__all__ = ["QASessionRequest", "build_qa_session_request"]
