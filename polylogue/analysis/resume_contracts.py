"""Closed continuation candidate contracts shared by readers and surfaces."""

from __future__ import annotations

from pydantic import Field, field_validator

from polylogue.analysis.archive_models import ArchiveInsightModel


class ResumePathOverlap(ArchiveInsightModel):
    candidate_path: str
    recent_file: str


class ResumeOverlapBasis(ArchiveInsightModel):
    exact: tuple[ResumePathOverlap, ...] = ()
    dir: tuple[ResumePathOverlap, ...] = ()
    dead_excluded: tuple[str, ...] = ()


class ResumeCandidate(ArchiveInsightModel):
    logical_session_id: str
    canonical_session_date: str | None = None
    last_message_at: str | None = None
    title: str
    terminal_state: str = "unknown"
    # polylogue-37t.23: structural_inference-tier objective posture (the
    # strongest posture across the logical session's member profiles). Not
    # overlaid with the assertion tier here -- that would require a live
    # user.db query per candidate on every ranking call. Callers that need
    # the full authority-blended posture for one candidate should follow up
    # with `resume_brief`, whose `inferences.objective_posture` overlays it.
    objective_posture: str = "unknown"
    workflow_shape: str = "unknown"
    file_overlap: tuple[str, ...] = ()
    overlap_basis: ResumeOverlapBasis = Field(default_factory=ResumeOverlapBasis)
    score: float
    score_breakdown: dict[str, float]
    brief_url: str

    @field_validator("logical_session_id", "title", "brief_url")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field cannot be empty")
        return value
