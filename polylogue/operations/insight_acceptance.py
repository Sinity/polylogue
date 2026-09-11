"""Immutable, page-bounded authority contracts for daemon insight maintenance.

This module deliberately has no daemon imports.  The daemon owner computes and
publishes one accepted part at a time, while the audit/executor layer owns the
authority and finalization boundaries represented here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, TypeAlias

from polylogue.operations.mutation_transaction import MutationPlan, MutationPreview

if TYPE_CHECKING:
    from polylogue.operations.audit import AuditRepository, MachineRequestBinding
    from polylogue.operations.bindings import OperationBinding
    from polylogue.operations.mutation_transaction import MutationAuthorization, MutationPrincipal, OperationExecutor

InsightScopeKind: TypeAlias = Literal["explicit", "full"]
InsightTargetDisposition: TypeAlias = Literal["required", "excess"]
InsightPartDisposition: TypeAlias = Literal["already_satisfied", "published", "pending", "stale", "failed", "unknown"]

MAX_INSIGHT_PART_TARGETS = 256
MAX_INSIGHT_ACCEPTED_PARTS = 4096


@dataclass(frozen=True, slots=True)
class AcceptedInsightTarget:
    """One exact frozen target and why it belongs to a maintenance sweep."""

    target_ref: str
    disposition: InsightTargetDisposition

    def __post_init__(self) -> None:
        if not self.target_ref.startswith("session:"):
            raise ValueError("accepted insight target must be a session reference")


@dataclass(frozen=True, slots=True)
class AcceptedInsightPart:
    """One sealed, independently startable page of an insight rebuild."""

    preview_ref: str
    authorization_ref: str
    plan_hash: str
    ordinal: int
    page_count: int
    manifest_digest: str
    previous_preview_ref: str | None
    scope_kind: InsightScopeKind
    index_generation: str
    recipe_version: str
    targets: tuple[AcceptedInsightTarget, ...]

    def __post_init__(self) -> None:
        if not self.preview_ref or not self.authorization_ref or len(self.plan_hash) != 64:
            raise ValueError("accepted insight part lacks an exact authority reference")
        if not 0 <= self.ordinal < self.page_count <= MAX_INSIGHT_ACCEPTED_PARTS:
            raise ValueError("accepted insight page ordinal is outside the bounded manifest")
        if len(self.manifest_digest) != 64 or any(char not in "0123456789abcdef" for char in self.manifest_digest):
            raise ValueError("accepted insight manifest digest must be SHA-256 hex")
        if not self.index_generation.startswith("index-generation:") or not self.recipe_version:
            raise ValueError("accepted insight part lacks its generation or recipe binding")
        if len(self.targets) > MAX_INSIGHT_PART_TARGETS:
            raise ValueError("accepted insight part exceeds the per-page target budget")
        if self.ordinal == 0 and self.previous_preview_ref is not None:
            raise ValueError("first accepted insight page cannot name a predecessor")
        if self.ordinal > 0 and not self.previous_preview_ref:
            raise ValueError("non-first accepted insight page requires its predecessor reference")


@dataclass(frozen=True, slots=True)
class InsightCertifiedCounts:
    """Certified derived outputs, deliberately distinct from physical writes."""

    profiles: int
    work_events: int
    phases: int

    def __post_init__(self) -> None:
        if min(self.profiles, self.work_events, self.phases) < 0:
            raise ValueError("certified insight counts cannot be negative")


@dataclass(frozen=True, slots=True)
class SessionInsightTargetReceipt:
    """Complete publication observation for a single accepted target."""

    target_ref: str
    disposition: InsightPartDisposition
    input_binding: str | None
    output_binding: str | None
    certified_counts: InsightCertifiedCounts
    publication_known_committed: bool

    def __post_init__(self) -> None:
        if not self.target_ref.startswith("session:"):
            raise ValueError("insight receipt target must be a session reference")
        if self.disposition == "published" and not self.publication_known_committed:
            raise ValueError("a published insight result must establish its committed publication")


@dataclass(frozen=True, slots=True)
class SessionInsightPartReceipt:
    """Owner result for exactly one accepted page and its unattempted suffix."""

    targets: tuple[SessionInsightTargetReceipt, ...]
    remaining_unattempted_target_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen = [item.target_ref for item in self.targets]
        if len(seen) != len(set(seen)):
            raise ValueError("insight part receipt repeats a target")
        if len(self.remaining_unattempted_target_refs) != len(set(self.remaining_unattempted_target_refs)):
            raise ValueError("insight part receipt repeats an unattempted target")
        if set(seen) & set(self.remaining_unattempted_target_refs):
            raise ValueError("insight part receipt overlaps attempted and unattempted targets")


def insight_page_context(part: AcceptedInsightPart) -> dict[str, object]:
    """Return the closed plan context whose hash covers one immutable page."""

    return {
        "scope_kind": part.scope_kind,
        "index_generation": part.index_generation,
        "recipe_version": part.recipe_version,
        "page_ordinal": part.ordinal,
        "page_count": part.page_count,
        "manifest_digest": part.manifest_digest,
        "previous_preview_ref": part.previous_preview_ref,
        "targets": [{"target_ref": target.target_ref, "disposition": target.disposition} for target in part.targets],
    }


def build_insight_page_context(
    *,
    scope_kind: InsightScopeKind,
    index_generation: str,
    recipe_version: str,
    ordinal: int,
    page_count: int,
    manifest_digest: str,
    previous_preview_ref: str | None,
    targets: tuple[AcceptedInsightTarget, ...],
) -> dict[str, object]:
    """Build closed context before the audit allocates preview/auth references."""

    checked = AcceptedInsightPart(
        preview_ref="pending-preview",
        authorization_ref="pending-authorization",
        plan_hash="0" * 64,
        ordinal=ordinal,
        page_count=page_count,
        manifest_digest=manifest_digest,
        previous_preview_ref=previous_preview_ref,
        scope_kind=scope_kind,
        index_generation=index_generation,
        recipe_version=recipe_version,
        targets=targets,
    )
    return insight_page_context(checked)


def accepted_part_from_plan(
    plan: MutationPlan,
    *,
    preview_ref: str,
    authorization_ref: str,
) -> AcceptedInsightPart:
    """Decode a closed insights page and prove it matches its typed plan targets."""

    if plan.operation != "mutate-rebuild-insights":
        raise ValueError("insight acceptance requires the rebuild-insights operation plan")
    context = plan.context
    expected = {
        "scope_kind",
        "index_generation",
        "recipe_version",
        "page_ordinal",
        "page_count",
        "manifest_digest",
        "previous_preview_ref",
        "targets",
    }
    if set(context) != expected:
        raise ValueError("insight acceptance context has unknown or missing fields")
    scope_kind = context["scope_kind"]
    index_generation = context["index_generation"]
    recipe_version = context["recipe_version"]
    ordinal = context["page_ordinal"]
    page_count = context["page_count"]
    digest = context["manifest_digest"]
    predecessor = context["previous_preview_ref"]
    raw_targets = context["targets"]
    if (
        scope_kind not in {"explicit", "full"}
        or not isinstance(index_generation, str)
        or not isinstance(recipe_version, str)
        or type(ordinal) is not int
        or type(page_count) is not int
        or not isinstance(digest, str)
        or (predecessor is not None and not isinstance(predecessor, str))
        or not isinstance(raw_targets, list)
    ):
        raise ValueError("insight acceptance context is not closed typed data")
    targets: list[AcceptedInsightTarget] = []
    for item in raw_targets:
        if not isinstance(item, dict) or set(item) != {"target_ref", "disposition"}:
            raise ValueError("insight acceptance target is malformed")
        target_ref, disposition = item["target_ref"], item["disposition"]
        if not isinstance(target_ref, str) or disposition not in {"required", "excess"}:
            raise ValueError("insight acceptance target is not typed")
        targets.append(AcceptedInsightTarget(target_ref, disposition))
    part = AcceptedInsightPart(
        preview_ref=preview_ref,
        authorization_ref=authorization_ref,
        plan_hash=plan.plan_hash,
        ordinal=ordinal,
        page_count=page_count,
        manifest_digest=digest,
        previous_preview_ref=predecessor,
        scope_kind=scope_kind,
        index_generation=index_generation,
        recipe_version=recipe_version,
        targets=tuple(targets),
    )
    if tuple(target.target_ref for target in part.targets) != plan.target_refs:
        raise ValueError("insight acceptance context does not match the plan target sequence")
    return part


def insight_manifest_digest(parts: tuple[AcceptedInsightPart, ...]) -> str:
    """Hash exact ordered page facts, never an ambient cursor or rediscovery."""

    payload = [
        {
            "ordinal": part.ordinal,
            "scope_kind": part.scope_kind,
            "index_generation": part.index_generation,
            "recipe_version": part.recipe_version,
            "targets": [
                {"target_ref": target.target_ref, "disposition": target.disposition} for target in part.targets
            ],
        }
        for part in parts
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class InsightAcceptance:
    """Seal/reload only the existing bounded audit authority for one request."""

    audit: AuditRepository
    binding: MachineRequestBinding
    principal: MutationPrincipal

    def stage_preview(self, plan: MutationPlan) -> MutationPreview:
        """Persist/reload one page before acceptance without treating it as work.

        Pages are appended in ordinal order so each later context can name the
        real durable predecessor reference.  A retry reloads that exact page;
        it never allocates a substitute preview.
        """

        if plan.archive_identity_digest != self.binding.archive_identity:
            raise ValueError("insight preview archive differs from its machine request")
        expected = accepted_part_from_plan(
            plan,
            preview_ref="pending-preview",
            authorization_ref="pending-authorization",
        )
        parts = self.audit.machine_parts(self.binding)
        if expected.ordinal < len(parts):
            raw = parts[expected.ordinal]
            preview = self.audit.preview_for_principal(str(raw["preview_ref"]), self.principal)
            durable = accepted_part_from_plan(
                preview.plan,
                preview_ref=preview.preview_ref,
                authorization_ref="pending-authorization",
            )
            if durable != replace(expected, preview_ref=preview.preview_ref):
                raise ValueError("staged insight page differs from the supplied immutable plan")
            return preview
        if expected.ordinal != len(parts):
            raise ValueError("insight pages must stage in contiguous ordinal order")
        with self.audit.bind_machine_request(self.binding, transition="append_insight_preview"):
            preview_ref = self.audit.append_insight_preview(plan, self.principal)
        return MutationPreview(preview_ref, plan)

    def staged_authorization(self, preview: MutationPreview):
        """Reload an active staged authorization; callers must refuse if absent/expired."""

        return self.audit.active_authorization_for_preview(preview.preview_ref, self.principal)

    def ensure_staged_authorization(
        self,
        executor: OperationExecutor,
        binding: OperationBinding[object, object],
        preview: MutationPreview,
    ) -> MutationAuthorization:
        """Issue once for an unissued page, never replace expired/consumed authority."""

        existing = self.staged_authorization(preview)
        if existing is not None:
            return existing
        if self.audit.has_authorization_for_preview(preview.preview_ref):
            raise ValueError("staged insight authority is no longer active; start a new request")
        return executor.authorize_bound(binding, preview, self.principal)

    def seal(
        self,
        *,
        head_preview_ref: str,
        page_count: int,
        manifest_digest: str,
        deadline_unix_ms: int,
    ) -> tuple[AcceptedInsightPart, ...]:
        """Atomically reserve a complete immutable page chain, or reload it.

        The caller may prepare pages and authorizations before this transition,
        but no owner receives a part until this method has persisted all exact
        references. Repeating the same machine binding returns that persisted
        manifest rather than allocating a replacement authorization.
        """

        existing = self.audit.machine_request(self.binding)
        if existing is None or existing.get("artifact_kind") == "insight-preview-pages":
            with self.audit.bind_machine_request(
                self.binding,
                transition="seal_insight_execution",
                deadline_unix_ms=deadline_unix_ms,
            ):
                self.audit.seal_insight_execution(
                    head_preview_ref,
                    page_count,
                    manifest_digest,
                    self.principal,
                )
        elif existing.get("artifact_kind") != "execution-batch":
            raise ValueError("machine request is already bound to another staged artifact")
        parts = self.audit.sealed_insight_parts(self.binding, self.principal)
        typed = tuple(part for part in parts if isinstance(part, AcceptedInsightPart))
        if len(typed) != len(parts):
            raise RuntimeError("sealed insight authority did not reconstruct typed parts")
        if (
            len(typed) != page_count
            or typed[-1].preview_ref != head_preview_ref
            or typed[0].manifest_digest != manifest_digest
        ):
            raise ValueError("existing insight request differs from the supplied sealed manifest")
        return typed


__all__ = [
    "AcceptedInsightPart",
    "AcceptedInsightTarget",
    "InsightCertifiedCounts",
    "InsightAcceptance",
    "InsightPartDisposition",
    "InsightScopeKind",
    "InsightTargetDisposition",
    "MAX_INSIGHT_ACCEPTED_PARTS",
    "MAX_INSIGHT_PART_TARGETS",
    "SessionInsightPartReceipt",
    "SessionInsightTargetReceipt",
    "accepted_part_from_plan",
    "build_insight_page_context",
    "insight_manifest_digest",
    "insight_page_context",
]
