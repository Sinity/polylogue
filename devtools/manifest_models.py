"""Pydantic v2 models for the schema-validated YAML manifests under docs/plans/.

Only manifests with real downstream consumers (a check that reads their
content to catch actual drift, not just self-referential shape validation)
get a model here.  Models use ``extra='forbid'`` so unknown fields are caught
during validation rather than silently ignored.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


def validate_layering_manifest(data: dict[str, object], *, path: str) -> list[str]:
    """Validate the layering policy before its owning gate consumes it.

    Returns a list of human-readable error strings (empty == valid).
    Each error includes the manifest file name and the field path so
    that operators can locate the problem without opening the file
    in an editor.
    """
    try:
        LayeringManifest.model_validate(data)
    except Exception as exc:
        errors = _format_pydantic_errors(path, exc)
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
    "LayeringManifest",
    "LayeringRule",
    "TwinWriteContract",
    "WriterModuleEntry",
    "WriterModulePolicy",
    "WriterModuleSurface",
    "validate_layering_manifest",
]
