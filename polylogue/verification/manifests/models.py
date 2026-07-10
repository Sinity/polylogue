"""Pydantic v2 models for all 16 YAML manifests under docs/plans/.

Every manifest type from the topology projection to the oracle-quality taxonomy
gets its own validated model.  Models use ``extra='forbid'`` so unknown fields
are caught during validation rather than silently ignored.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ──────────────────────────────────────────────────────────────────────
# Topology manifest        (topology-target.yaml)
# ──────────────────────────────────────────────────────────────────────


class TopologyEntry(BaseModel):
    """A single file entry in the topology projection."""

    model_config = ConfigDict(extra="forbid")
    path: str
    loc: int | None = None
    target: str | None = None
    owner: str
    reason: str | None = None
    cross_cut: dict[str, str] = Field(default_factory=dict)


class TopologyManifest(BaseModel):
    """Root of topology-target.yaml."""

    model_config = ConfigDict(extra="forbid")
    files: list[TopologyEntry]


# ──────────────────────────────────────────────────────────────────────
# Lint-escalation manifest  (lint-escalation.yaml)
# ──────────────────────────────────────────────────────────────────────


class LintRule(BaseModel):
    """A single lint-escalation rule."""

    model_config = ConfigDict(extra="forbid")
    id: str
    description: str
    severity: str
    sunset: str  # ISO-8601 date string

    VALID_SEVERITIES: ClassVar[frozenset[str]] = frozenset({"soft", "hard"})

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, v: str) -> str:
        if v not in cls.VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(cls.VALID_SEVERITIES)}, got {v!r}")
        return v

    @field_validator("sunset")
    @classmethod
    def _check_sunset(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except (ValueError, TypeError) as err:
            raise ValueError(f"sunset is not a valid ISO date: {v!r}") from err
        return v


class LintEscalationManifest(BaseModel):
    """Root of lint-escalation.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    rules: list[LintRule]


# ──────────────────────────────────────────────────────────────────────
# Coverage Gap  (shared fragment in many *coverage*.yaml manifests)
# ──────────────────────────────────────────────────────────────────────


class CoverageGap(BaseModel):
    """A known coverage gap record."""

    model_config = ConfigDict(extra="forbid")
    id: str
    gap: str
    owner: str
    severity: str
    declared_at: str  # ISO-8601 date
    review_after: str  # ISO-8601 date
    issue: int | str | None = None
    suppression: str | None = None
    bead: str | None = None
    next_evidence: str | None = None
    subject: str | None = None
    dimension: str | None = None
    axis: str | None = None
    area: str | None = None
    artifact: str | None = None
    platform: str | None = None
    concern: str | None = None
    domain: str | None = None

    VALID_SEVERITIES: ClassVar[frozenset[str]] = frozenset({"info", "minor", "major", "serious"})

    @field_validator("severity")
    @classmethod
    def _check_severity(cls, v: str) -> str:
        if v not in cls.VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(cls.VALID_SEVERITIES)}, got {v!r}")
        return v

    @field_validator("declared_at", "review_after")
    @classmethod
    def _check_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except (ValueError, TypeError) as err:
            raise ValueError(f"not a valid ISO date: {v!r}") from err
        return v


# ──────────────────────────────────────────────────────────────────────
# Generic coverage manifest  (*coverage*.yaml)
# ──────────────────────────────────────────────────────────────────────


class CoverageManifest(BaseModel):
    """Generic root for *coverage*.yaml files that carry a coverage_gaps list."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Scenario-coverage manifest  (scenario-coverage.yaml)
# ──────────────────────────────────────────────────────────────────────


class ScenarioFamily(BaseModel):
    """A single scenario family."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    subject: str
    scenario_count: int | str  # int literal or "dynamic"
    location: str
    bead: str | None = None
    notes: str | None = None


class ScenarioCoverageManifest(BaseModel):
    """Root of scenario-coverage.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    families: list[ScenarioFamily]
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Campaign-coverage manifest  (campaign-coverage.yaml)
# ──────────────────────────────────────────────────────────────────────


class MutationCampaignEntry(BaseModel):
    """A single mutation-campaign record."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    paths_to_mutate: list[str]
    tests: list[str]
    status: str = "active"
    freshness_days: int | None = None
    artifact_glob: str | None = None
    min_kill_rate: float | None = None

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset({"active", "inactive", "draft", "archived"})

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        if v not in cls.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(cls.VALID_STATUSES)}, got {v!r}")
        return v

    @field_validator("freshness_days")
    @classmethod
    def _check_freshness(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"freshness_days must be positive, got {v!r}")
        return v

    @field_validator("min_kill_rate")
    @classmethod
    def _check_min_kill_rate(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"min_kill_rate must be within [0, 1], got {v!r}")
        return v


class BenchmarkCampaignEntry(BaseModel):
    """A single benchmark-campaign record."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    tests: list[str]
    status: str = "active"
    freshness_days: int | None = None
    artifact_glob: str | None = None

    VALID_STATUSES: ClassVar[frozenset[str]] = frozenset({"active", "inactive", "draft", "archived"})

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: str) -> str:
        if v not in cls.VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(cls.VALID_STATUSES)}, got {v!r}")
        return v

    @field_validator("freshness_days")
    @classmethod
    def _check_freshness(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError(f"freshness_days must be positive, got {v!r}")
        return v


class CampaignCoverageManifest(BaseModel):
    """Root of campaign-coverage.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    default_min_kill_rate: float | None = None
    mutation_campaigns: list[MutationCampaignEntry] = Field(default_factory=list)
    benchmark_campaigns: list[BenchmarkCampaignEntry] = Field(default_factory=list)

    @field_validator("default_min_kill_rate")
    @classmethod
    def _check_default_min_kill_rate(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"default_min_kill_rate must be within [0, 1], got {v!r}")
        return v


# ──────────────────────────────────────────────────────────────────────
# Layering manifest  (layering.yaml)
# ──────────────────────────────────────────────────────────────────────


class LayeringConstraint(BaseModel):
    """Allow or disallow constraint for a layering rule."""

    model_config = ConfigDict(extra="forbid")
    from_targets: list[str] | None = Field(default=None, validation_alias="from")  # YAML uses "from" keyword


class LayeringRule(BaseModel):
    """A single layering rule."""

    model_config = ConfigDict(extra="forbid")
    target: str
    description: str
    disallow: LayeringConstraint | None = None
    allow: LayeringConstraint | None = None


class LayeringManifest(BaseModel):
    """Root of layering.yaml."""

    model_config = ConfigDict(extra="forbid")
    rules: list[LayeringRule]


# ──────────────────────────────────────────────────────────────────────
# Oracle-quality manifest  (oracle-quality.yaml)
# ──────────────────────────────────────────────────────────────────────


class OracleEntry(BaseModel):
    """A single oracle-quality taxonomy entry."""

    model_config = ConfigDict(extra="forbid")
    counts_as: str
    independence: str
    example: str
    typical_use: str


class OracleQualityManifest(BaseModel):
    """Root of oracle-quality.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    oracles: dict[str, OracleEntry]


# ──────────────────────────────────────────────────────────────────────
# Evidence-freshness manifest  (evidence-freshness.yaml)
# ──────────────────────────────────────────────────────────────────────


class FreshnessPolicy(BaseModel):
    """A single evidence-freshness policy."""

    model_config = ConfigDict(extra="forbid")
    description: str
    max_age_days: int
    applies_to: list[str] = Field(default_factory=list)


class StaleThreshold(BaseModel):
    """Staleness thresholds per oracle type."""

    model_config = ConfigDict(extra="forbid")
    behavior_evidence_days: int = 14
    smoke_evidence_days: int = 30
    manual_review_evidence_days: int = 90


class EvidenceFreshnessManifest(BaseModel):
    """Root of evidence-freshness.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    freshness_policies: dict[str, FreshnessPolicy] = Field(default_factory=dict)
    stale_threshold: StaleThreshold = Field(default_factory=StaleThreshold)


# ──────────────────────────────────────────────────────────────────────
# API-parity manifest  (api-parity.yaml)
# ──────────────────────────────────────────────────────────────────────


class SurfaceEntry(BaseModel):
    """A single surface entry in the API-parity manifest."""

    model_config = ConfigDict(extra="forbid")
    description: str
    coverage: str
    path: str
    gaps: list[str] = Field(default_factory=list)


class OperationEntry(BaseModel):
    """A single operation entry tracking surface support."""

    model_config = ConfigDict(extra="forbid")
    name: str
    cli: bool = False
    mcp: bool = False
    api: bool = False
    sync: bool = False
    notes: str | None = None


class ParityCheckEntry(BaseModel):
    """A single parity check definition."""

    model_config = ConfigDict(extra="forbid")
    description: str
    status: str
    path: str | None = None
    owner: str | None = None


class ApiParityManifest(BaseModel):
    """Root of api-parity.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    surfaces: dict[str, SurfaceEntry] = Field(default_factory=dict)
    operations: list[OperationEntry] = Field(default_factory=list)
    parity_checks: dict[str, ParityCheckEntry] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Security-privacy-coverage manifest  (security-privacy-coverage.yaml)
# ──────────────────────────────────────────────────────────────────────


class TestCoverage(BaseModel):
    """Test-coverage metadata for a security area."""

    model_config = ConfigDict(extra="forbid")
    location: str | None = None
    hypothesis: bool = False


class SecurityControl(BaseModel):
    """A single security control entry."""

    model_config = ConfigDict(extra="forbid")
    description: str
    implemented: bool = False
    controls: list[dict[str, str]] | dict[str, str] = Field(default_factory=dict)
    test_coverage: TestCoverage = Field(default_factory=TestCoverage)
    notes: str | None = None


class SecurityPrivacyManifest(BaseModel):
    """Root of security-privacy-coverage.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    areas: dict[str, SecurityControl] = Field(default_factory=dict)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Distribution-coverage manifest  (distribution-coverage.yaml)
# ──────────────────────────────────────────────────────────────────────


class DistributionArtifact(BaseModel):
    """A single distribution artifact entry.

    Only fields consumed by an executable check are retained
    (#1064 Pack C). ``ci_build`` / ``ci_test`` / ``ci_present`` drive
    ``verify_manifests.check_distribution_ci_claims`` against committed
    workflow YAML; ``build_command`` / ``verification_command`` /
    ``config_location`` resolve through ``check_coverage_references``.
    The previous ``freshness_days`` field was removed because no check
    consumed it.
    """

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    build_system: str | None = None
    config_location: str | None = None
    build_command: str | None = None
    install_command: str | None = None
    verification_command: str | None = None
    ci_build: bool = False
    ci_test: bool = False
    notes: str | None = None
    ci_present: bool = False


class PlatformCoverage(BaseModel):
    """Platform coverage entry."""

    model_config = ConfigDict(extra="forbid")
    linux: str | bool = False
    macos: bool = False
    windows: bool = False
    notes: str | None = None


class PipDependencies(BaseModel):
    """Pip dependencies metadata stored inside the artifacts dict."""

    model_config = ConfigDict(extra="forbid")
    count: int | None = None
    resolved_by: str | None = None


class DistributionCoverageManifest(BaseModel):
    """Root of distribution-coverage.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    artifacts: dict[str, DistributionArtifact | PipDependencies | PlatformCoverage] = Field(default_factory=dict)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Docs-media-coverage manifest  (docs-media-coverage.yaml)
# ──────────────────────────────────────────────────────────────────────


class DocMediaSurface(BaseModel):
    """A single documentation surface entry.

    Only fields consumed by an executable check are retained
    (#1064 Pack C). ``path`` is verified against the filesystem and
    ``generated_by`` / ``verified_by`` resolve against the devtools
    command catalog via ``check_coverage_references``. The previous
    ``related_paths``, ``sections``, ``freshness_days``, ``count``,
    and ``providers`` fields were removed because no check consumed
    them.
    """

    model_config = ConfigDict(extra="forbid")
    path: str | None = None
    generated_by: str | None = None
    verified_by: str | None = None
    notes: str | None = None


class DocsMediaCoverageManifest(BaseModel):
    """Root of docs-media-coverage.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    surfaces: dict[str, DocMediaSurface] = Field(default_factory=dict)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Test-quality-coverage manifest  (test-quality-coverage.yaml)
# ──────────────────────────────────────────────────────────────────────


class FuzzTool(BaseModel):
    """Fuzz-tool entry."""

    model_config = ConfigDict(extra="forbid")
    name: str
    locations: list[str] = Field(default_factory=list)
    strategies_location: str | None = None
    schema_driven_strategies: bool = False
    notes: str | None = None


class FlakyTest(BaseModel):
    """Known flaky test entry."""

    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    location: str | None = None
    intermittent_on: str | None = None
    behavior: str | None = None
    workaround: str | None = None


class TestQualityDimension(BaseModel):
    """A single test-quality dimension."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    measured: bool = False
    value_percent: int | None = None
    fail_under_percent: int | None = None
    tool: str | None = None
    config_location: str | None = None
    ci_gate: bool = False
    last_verified: str | None = None
    notes: str | None = None
    known_flaky: list[FlakyTest] = Field(default_factory=list)
    ci_retry: bool = False
    flakiness_dashboard: bool = False
    tools: list[FuzzTool] = Field(default_factory=list)
    policy: str | None = None


class TestLocations(BaseModel):
    """Test-location groups."""

    model_config = ConfigDict(extra="forbid")
    unit_core: list[str] = Field(default_factory=list)
    unit_sources: list[str] = Field(default_factory=list)
    unit_storage: list[str] = Field(default_factory=list)
    unit_pipeline: list[str] = Field(default_factory=list)
    unit_cli: list[str] = Field(default_factory=list)
    unit_mcp: list[str] = Field(default_factory=list)
    unit_security: list[str] = Field(default_factory=list)
    unit_rendering: list[str] = Field(default_factory=list)
    integration: list[str] = Field(default_factory=list)
    fuzz: list[str] = Field(default_factory=list)


class TestCount(BaseModel):
    """Test-count record."""

    model_config = ConfigDict(extra="forbid")
    total: int | None = None
    unit: str | None = None
    property: str | None = None
    integration: str | None = None
    snapshot: str | int | None = None
    last_measured: str | None = None


class TestQualityCoverageManifest(BaseModel):
    """Root of test-quality-coverage.yaml."""

    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    dimensions: dict[str, TestQualityDimension | TestCount | TestLocations] = Field(default_factory=dict)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Manifest-type dispatch table
# ──────────────────────────────────────────────────────────────────────

# Maps YAML filename → Pydantic model class for structural validation.
MANIFEST_MODELS: dict[str, type[BaseModel]] = {
    "topology-target.yaml": TopologyManifest,
    "lint-escalation.yaml": LintEscalationManifest,
    "scenario-coverage.yaml": ScenarioCoverageManifest,
    "campaign-coverage.yaml": CampaignCoverageManifest,
    "layering.yaml": LayeringManifest,
    "oracle-quality.yaml": OracleQualityManifest,
    "evidence-freshness.yaml": EvidenceFreshnessManifest,
    "api-parity.yaml": ApiParityManifest,
    "security-privacy-coverage.yaml": SecurityPrivacyManifest,
    "distribution-coverage.yaml": DistributionCoverageManifest,
    "docs-media-coverage.yaml": DocsMediaCoverageManifest,
    "test-quality-coverage.yaml": TestQualityCoverageManifest,
}


def validate_manifest(manifest_path: str, data: dict[str, object]) -> list[str]:
    """Validate a single parsed YAML manifest against its Pydantic model.

    Returns a list of human-readable error strings (empty == valid).
    Each error includes the manifest file name and the field path so
    that operators can locate the problem without opening the file
    in an editor.
    """
    import os

    filename = os.path.basename(manifest_path)
    model_cls = MANIFEST_MODELS.get(filename)
    if model_cls is None:
        return []  # unknown manifest skipped (not an error)

    try:
        model_cls.model_validate(data)
    except Exception as exc:
        errors = _format_pydantic_errors(manifest_path, exc)
        return errors

    return []


def _format_pydantic_errors(path: str, exc: Exception) -> list[str]:
    """Format Pydantic validation errors into operator-actionable lines.

    Pydantic v2 raises ``ValidationError`` with a ``.errors()`` list
    that contains ``loc`` (field path as tuple), ``msg``, and ``type``.
    """
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        # Non-validation exception (e.g. TypeError during model construction).
        return [f"{path}: Pydantic validation failed: {exc}"]

    lines: list[str] = []
    for err in exc.errors():
        loc = " → ".join(str(part) for part in err.get("loc", ()))
        msg = err.get("msg", "unknown error")
        lines.append(f"{path}: {loc}: {msg}")
    return lines


__all__ = [
    "ApiParityManifest",
    "BenchmarkCampaignEntry",
    "CampaignCoverageManifest",
    "CoverageGap",
    "CoverageManifest",
    "DistributionArtifact",
    "DistributionCoverageManifest",
    "DocMediaSurface",
    "DocsMediaCoverageManifest",
    "EvidenceFreshnessManifest",
    "FlakyTest",
    "FreshnessPolicy",
    "FuzzTool",
    "LayeringManifest",
    "LayeringRule",
    "LintEscalationManifest",
    "LintRule",
    "MANIFEST_MODELS",
    "MutationCampaignEntry",
    "OperationEntry",
    "OracleEntry",
    "OracleQualityManifest",
    "ParityCheckEntry",
    "PlatformCoverage",
    "ScenarioCoverageManifest",
    "ScenarioFamily",
    "SecurityControl",
    "SecurityPrivacyManifest",
    "StaleThreshold",
    "SurfaceEntry",
    "TestCount",
    "TestCoverage",
    "TestLocations",
    "TestQualityCoverageManifest",
    "TestQualityDimension",
    "TopologyEntry",
    "TopologyManifest",
    "validate_manifest",
]
