"""Pydantic v2 models for the schema-validated YAML manifests under docs/plans/.

Only manifests with real downstream consumers (a check that reads their
content to catch actual drift, not just self-referential shape validation)
get a model here.  Models use ``extra='forbid'`` so unknown fields are caught
during validation rather than silently ignored.
"""

from __future__ import annotations

from datetime import date
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    # polylogue-2ciy: a checked-in known-violations ratchet baseline (see
    # devtools/verify_layering.py:_load_baseline) -- repo-relative path to a
    # JSON file of pre-existing (target, file, import) triples exempted from
    # this constraint. Only meaningful on `disallow`.
    baseline: str | None = None


class LayeringRule(BaseModel):
    """A single layering rule."""

    model_config = ConfigDict(extra="forbid")
    target: str
    description: str
    disallow: LayeringConstraint | None = None
    allow: LayeringConstraint | None = None


class WriterModuleSurface(BaseModel):
    """Durability and interruption contract for one writer-owned tier."""

    model_config = ConfigDict(extra="forbid")
    tier: Literal["source", "index", "embeddings", "user", "ops"]
    durability: Literal["durable", "rebuildable", "disposable"]
    interruption: Literal["atomic", "replayable", "restartable"]

    _TIER_CONTRACTS: ClassVar[dict[str, tuple[str, str]]] = {
        "source": ("durable", "atomic"),
        "index": ("rebuildable", "replayable"),
        "embeddings": ("rebuildable", "replayable"),
        "user": ("durable", "atomic"),
        "ops": ("disposable", "restartable"),
    }

    @model_validator(mode="after")
    def _check_tier_contract(self) -> WriterModuleSurface:
        expected = self._TIER_CONTRACTS[self.tier]
        if (self.durability, self.interruption) != expected:
            raise ValueError(f"{self.tier} requires durability/interruption {expected}")
        return self


class WriterModuleEntry(BaseModel):
    """One production module that owns one or more SQLite write surfaces."""

    model_config = ConfigDict(extra="forbid")
    path: str
    surfaces: list[WriterModuleSurface] = Field(min_length=1)
    entrypoints: list[str] = Field(min_length=1)
    twin_write_contract: str | None = None


class TwinWriteContract(BaseModel):
    """Explicit exception for one module that atomically spans two tiers."""

    model_config = ConfigDict(extra="forbid")
    name: str
    module: str
    surfaces: list[Literal["source", "index", "embeddings", "user", "ops"]] = Field(min_length=2)
    entrypoints: list[str] = Field(min_length=1)
    reason: str


class WriterModulePolicy(BaseModel):
    """Production writer-module inventory and its interruption doctrine."""

    model_config = ConfigDict(extra="forbid")
    marker: str = "Writer module:"
    mutation_roots: list[str] = Field(min_length=1)
    modules: list[WriterModuleEntry] = Field(min_length=1)
    twin_write_contracts: list[TwinWriteContract] = Field(default_factory=list)


class LayeringManifest(BaseModel):
    """Root of layering.yaml."""

    model_config = ConfigDict(extra="forbid")
    writer_modules: WriterModulePolicy | None = None
    rules: list[LayeringRule]


# ──────────────────────────────────────────────────────────────────────
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
    "campaign-coverage.yaml": CampaignCoverageManifest,
    "layering.yaml": LayeringManifest,
    "distribution-coverage.yaml": DistributionCoverageManifest,
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
    "BenchmarkCampaignEntry",
    "CampaignCoverageManifest",
    "CoverageGap",
    "CoverageManifest",
    "DistributionArtifact",
    "DistributionCoverageManifest",
    "FlakyTest",
    "FuzzTool",
    "LayeringManifest",
    "LayeringRule",
    "TwinWriteContract",
    "WriterModuleEntry",
    "WriterModulePolicy",
    "WriterModuleSurface",
    "MANIFEST_MODELS",
    "MutationCampaignEntry",
    "PlatformCoverage",
    "TestCount",
    "TestLocations",
    "TestQualityCoverageManifest",
    "TestQualityDimension",
    "validate_manifest",
]
