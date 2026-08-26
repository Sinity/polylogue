"""The capability-negative candidate generation operation.

Candidate construction is deliberately a small operation contract.  The
request is the authenticated *what* of a build; the daemon supplies all
physical and execution facts when it plans the request.  Keeping those two
sets of facts separate is important: a client may ask for an inactive
candidate, but it must not be able to choose a pathname, generation id,
budget, acceptance profile, or lifecycle transition.

This module is also the wire boundary for the operation.  Its Pydantic
models reject unknown fields and its identity digest is computed from a
canonical representation, so CLI, daemon, HTTP, and MCP adapters can all
lower to exactly the same request.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, StrictInt, field_validator, model_validator

from polylogue.core.enums import OperationStatus
from polylogue.core.json import JSONDocument, json_document
from polylogue.operations.specs import OperationKind
from polylogue.surfaces.payloads import SurfacePayloadModel

if TYPE_CHECKING:
    from polylogue.storage.index_generation import IndexGeneration

CANDIDATE_BUILD_OPERATION = "candidate-build"
CANDIDATE_BUILD_PROTOCOL = "polylogue.candidate-build/v1"
CANDIDATE_BUILD_CLASS = "inactive-candidate"
CANDIDATE_BUILD_POLICY = "inactive-candidate-v1"

_IDENTIFIER_RE = re.compile(r"^[^/\\\x00]+$")
_DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PLAN_STATUSES = frozenset({OperationStatus.ACCEPTED, OperationStatus.PENDING, OperationStatus.RUNNING})


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    value = value.strip()
    if not _IDENTIFIER_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{label} must be an opaque identifier, not a path")
    return value


def _ordered_unique(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, label=label) for value in values)
    if not normalized:
        raise ValueError(f"{label} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} must contain unique values")
    return tuple(sorted(normalized))


class SourceSeal(SurfacePayloadModel):
    """The source authority consumed by a candidate build.

    These are identities, not locators.  In particular, no source pathname
    is accepted.  The daemon constructs this value from its authenticated
    source descriptor and compares it again at execution boundaries.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    archive_identity: str
    source_identity: str
    source_snapshot: str
    source_schema_version: StrictInt = Field(gt=0)

    @field_validator("archive_identity", "source_identity", "source_snapshot")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _identifier(value, label=info.field_name)

    @property
    def digest(self) -> str:
        return _sha256(self.model_dump(mode="json"))

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class CandidateBuildRequest(SurfacePayloadModel):
    """Authenticated semantic identity for one inactive candidate build.

    The deliberately closed field set is the capability boundary.  Physical
    paths, generation IDs, selection/member IDs, resource knobs, callbacks,
    arbitrary checks, and lifecycle verbs are not request fields and are
    rejected by the shared ``extra='forbid'`` wire model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_kind: ClassVar[OperationKind] = OperationKind.MATERIALIZATION
    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    source_seal: SourceSeal
    package: str
    code: str
    schemas: tuple[str, ...]
    parser_declarations: tuple[str, ...]
    lowering_declarations: tuple[str, ...]
    origin_declarations: tuple[str, ...]
    recipe_version: str
    semantic_version: str
    generation_policy: Literal["inactive-candidate-v1"] = CANDIDATE_BUILD_POLICY
    build_class: Literal["inactive-candidate"] = CANDIDATE_BUILD_CLASS

    @field_validator("package", "code", "recipe_version", "semantic_version")
    @classmethod
    def validate_scalar_identity(cls, value: str, info: Any) -> str:
        return _identifier(value, label=info.field_name)

    @field_validator("schemas", "parser_declarations", "lowering_declarations", "origin_declarations")
    @classmethod
    def validate_declarations(cls, value: tuple[str, ...], info: Any) -> tuple[str, ...]:
        return _ordered_unique(value, label=info.field_name)

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CandidateBuildRequest:
        """Validate one untrusted request object at a surface boundary."""

        return cls.model_validate(raw)

    def identity_document(self) -> JSONDocument:
        """Return only semantic authority, with unordered declarations sorted."""

        return json_document(
            {
                "operation": self.operation,
                "source_seal": self.source_seal.to_dict(),
                "package": self.package,
                "code": self.code,
                "schemas": sorted(self.schemas),
                "parser_declarations": sorted(self.parser_declarations),
                "lowering_declarations": sorted(self.lowering_declarations),
                "origin_declarations": sorted(self.origin_declarations),
                "recipe_version": self.recipe_version,
                "semantic_version": self.semantic_version,
                "generation_policy": self.generation_policy,
                "build_class": self.build_class,
            }
        )

    @property
    def identity_digest(self) -> str:
        return _sha256(self.identity_document())

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.identity_document(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class CandidateBuildBudget(SurfacePayloadModel):
    """Server-selected planning budgets; never part of request identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_bytes: StrictInt = Field(ge=0)
    work_units: StrictInt = Field(ge=0)
    memory_bytes: StrictInt = Field(ge=0)


class CandidateBuildObligation(SurfacePayloadModel):
    """One server-owned obligation in the current candidate plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    required: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, label="obligation name")


class CandidateBuildGeneration(SurfacePayloadModel):
    """The narrow generation identity exposed after server-side resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generation_id: str
    owner_id: str
    archive_root: str
    index_path: str
    state: Literal["inactive"] = "inactive"

    @field_validator("generation_id", "owner_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _identifier(value, label=info.field_name)

    @field_validator("archive_root", "index_path")
    @classmethod
    def validate_paths(cls, value: str, info: Any) -> str:
        if not value or not Path(value).is_absolute():
            raise ValueError(f"{info.field_name} must be an absolute server-resolved path")
        return value


class CandidateBuildPlan(SurfacePayloadModel):
    """Server-resolved plan for the validated request.

    The physical roots and target generation are intentionally introduced
    here, after request validation.  ``preview`` changes no authority: it is
    only a projection of this same plan shape.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["polylogue.candidate-build/v1"] = CANDIDATE_BUILD_PROTOCOL
    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    request_digest: str
    archive_root: str
    target_generation: CandidateBuildGeneration
    budget: CandidateBuildBudget
    obligations: tuple[CandidateBuildObligation, ...]
    preview: bool = False
    authority: Literal["inactive-only"] = "inactive-only"

    @field_validator("request_digest")
    @classmethod
    def validate_request_digest(cls, value: str) -> str:
        if _DIGEST_RE.fullmatch(value) is None:
            raise ValueError("request_digest must be a SHA-256 hex digest")
        return value.lower()

    @field_validator("archive_root")
    @classmethod
    def validate_archive_root(cls, value: str) -> str:
        if not value or not Path(value).is_absolute():
            raise ValueError("archive_root is a server-resolved absolute path")
        return value

    @model_validator(mode="after")
    def validate_target(self) -> CandidateBuildPlan:
        if self.target_generation.state != "inactive":
            raise ValueError("candidate target generation must be inactive")
        return self

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))

    @property
    def plan_digest(self) -> str:
        return _sha256(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CandidateBuildPlan:
        return cls.model_validate(raw)


class CandidateBuildProgress(SurfacePayloadModel):
    """Bounded progress for one operation, not a scheduling command."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    operation_id: str
    request_digest: str
    generation_id: str
    status: OperationStatus
    processed_units: StrictInt = Field(ge=0)
    total_units: StrictInt = Field(ge=0)
    processed_bytes: StrictInt = Field(ge=0)
    total_bytes: StrictInt = Field(ge=0)

    @field_validator("operation_id", "generation_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _identifier(value, label=info.field_name)

    @field_validator("request_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST_RE.fullmatch(value) is None:
            raise ValueError("request_digest must be a SHA-256 hex digest")
        return value.lower()

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class CandidateBuildResult(SurfacePayloadModel):
    """Successful candidate-build result with an explicit inactive boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    status: Literal["completed"] = "completed"
    request_digest: str
    plan_digest: str
    generation: CandidateBuildGeneration
    source_seal_digest: str
    processed_units: StrictInt = Field(ge=0)
    processed_bytes: StrictInt = Field(ge=0)
    lifecycle: Literal["inactive"] = "inactive"

    @field_validator("request_digest", "plan_digest", "source_seal_digest")
    @classmethod
    def validate_digests(cls, value: str) -> str:
        if _DIGEST_RE.fullmatch(value) is None:
            raise ValueError("candidate result identities must be SHA-256 hex digests")
        return value.lower()

    @model_validator(mode="after")
    def validate_inactive_generation(self) -> CandidateBuildResult:
        if self.generation.state != "inactive":
            raise ValueError("candidate result cannot contain an active generation")
        return self

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class CandidateBuildError(SurfacePayloadModel):
    """Typed failure that cannot grant a lifecycle capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    status: Literal["failed", "interrupted"]
    code: Literal[
        "invalid_request",
        "source_changed",
        "generation_conflict",
        "capacity_blocked",
        "lineage_invalid",
        "execution_failed",
    ]
    message: str
    request_digest: str
    generation_id: str | None = None
    retryable: bool = False

    @field_validator("request_digest")
    @classmethod
    def validate_request_digest(cls, value: str) -> str:
        if _DIGEST_RE.fullmatch(value) is None:
            raise ValueError("request_digest must be a SHA-256 hex digest")
        return value.lower()

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


class CandidateBuildReceipt(SurfacePayloadModel):
    """Terminal, replayable evidence for candidate construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["polylogue.candidate-build/v1"] = CANDIDATE_BUILD_PROTOCOL
    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    operation_id: str
    request_digest: str
    plan_digest: str
    status: Literal["completed", "failed", "interrupted"]
    result: CandidateBuildResult | None = None
    error: CandidateBuildError | None = None
    source_seal_digest: str
    generation_id: str

    @field_validator("operation_id", "generation_id")
    @classmethod
    def validate_identifiers(cls, value: str, info: Any) -> str:
        return _identifier(value, label=info.field_name)

    @field_validator("request_digest", "plan_digest", "source_seal_digest")
    @classmethod
    def validate_digests(cls, value: str) -> str:
        if _DIGEST_RE.fullmatch(value) is None:
            raise ValueError("candidate receipt identities must be SHA-256 hex digests")
        return value.lower()

    @model_validator(mode="after")
    def validate_terminal_shape(self) -> CandidateBuildReceipt:
        if (self.status == "completed") != (self.result is not None and self.error is None):
            raise ValueError("completed receipts require exactly one successful result")
        if self.status != "completed" and (self.error is None or self.result is not None):
            raise ValueError("failed receipts require exactly one typed error")
        if self.result is not None:
            if self.result.generation.state != "inactive":
                raise ValueError("candidate receipt cannot attest activation")
            if self.result.request_digest != self.request_digest:
                raise ValueError("candidate receipt/result request identity mismatch")
            if self.result.plan_digest != self.plan_digest:
                raise ValueError("candidate receipt/result plan identity mismatch")
            if self.result.source_seal_digest != self.source_seal_digest:
                raise ValueError("candidate receipt/result source seal mismatch")
            if self.result.generation.generation_id != self.generation_id:
                raise ValueError("candidate receipt/result generation identity mismatch")
        if self.error is not None and self.error.request_digest != self.request_digest:
            raise ValueError("candidate receipt/error request identity mismatch")
        return self

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CandidateBuildReceipt:
        return cls.model_validate(raw)


class CandidateBuildWireRequest(SurfacePayloadModel):
    """Strict protocol envelope shared by every adapter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["polylogue.candidate-build/v1"] = CANDIDATE_BUILD_PROTOCOL
    operation: Literal["candidate-build"] = CANDIDATE_BUILD_OPERATION
    request: CandidateBuildRequest

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> CandidateBuildWireRequest:
        return cls.model_validate(raw)

    def to_dict(self) -> JSONDocument:
        return json_document(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class CandidateBuildPlanningContext:
    """Daemon-owned facts used to resolve a request into a plan."""

    archive_root: Path
    source_seal: SourceSeal
    generation: IndexGeneration
    budget: CandidateBuildBudget
    obligations: tuple[CandidateBuildObligation, ...] = ()

    def plan(self, request: CandidateBuildRequest, *, preview: bool = False) -> CandidateBuildPlan:
        if request.source_seal != self.source_seal:
            raise ValueError("candidate request source seal is stale")
        if self.generation.state != "inactive":
            raise ValueError("candidate target generation is not inactive")
        if Path(self.generation.archive_root).absolute() != self.archive_root.absolute():
            raise ValueError("candidate generation belongs to a different archive root")
        return CandidateBuildPlan(
            request_digest=request.identity_digest,
            archive_root=str(self.archive_root.absolute()),
            target_generation=CandidateBuildGeneration(
                generation_id=self.generation.generation_id,
                owner_id=self.generation.owner_id,
                archive_root=self.generation.archive_root,
                index_path=self.generation.index_path,
            ),
            budget=self.budget,
            obligations=tuple(sorted(self.obligations, key=lambda item: item.name)),
            preview=preview,
        )


def plan_candidate_build(
    request: CandidateBuildRequest,
    context: CandidateBuildPlanningContext,
    *,
    preview: bool = False,
) -> CandidateBuildPlan:
    """Resolve one plan; preview and execution use this identical path."""

    return context.plan(request, preview=preview)


def lower_candidate_build_wire(
    raw: dict[str, object],
    *,
    surface: Literal["cli", "daemon", "mcp", "api"],
) -> CandidateBuildRequest:
    """Lower any public adapter to the canonical typed request.

    ``surface`` is an authorization label only; it cannot add fields or
    capabilities.  The operation declaration in ``specs.py`` remains the
    source of the public surface list.
    """

    if surface not in {"cli", "daemon", "mcp", "api"}:
        raise ValueError(f"candidate build is not available on surface {surface!r}")
    return CandidateBuildWireRequest.from_dict(raw).request


def _sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CANDIDATE_BUILD_CLASS",
    "CANDIDATE_BUILD_OPERATION",
    "CANDIDATE_BUILD_POLICY",
    "CANDIDATE_BUILD_PROTOCOL",
    "CandidateBuildBudget",
    "CandidateBuildError",
    "CandidateBuildGeneration",
    "CandidateBuildObligation",
    "CandidateBuildPlan",
    "CandidateBuildPlanningContext",
    "CandidateBuildProgress",
    "CandidateBuildReceipt",
    "CandidateBuildRequest",
    "CandidateBuildResult",
    "CandidateBuildWireRequest",
    "SourceSeal",
    "lower_candidate_build_wire",
    "plan_candidate_build",
]
