"""Public-claim projection over reviewed findings and evidence-integrity verdicts.

This module deliberately does not walk evidence ancestry.  Bead
``polylogue-37t.14`` owns support, drift, cycle, frame, and privacy semantics;
this consumer accepts its bounded verdict through :class:`EvidenceIntegrityProvider`
and applies only publication policy over existing FINDING assertion lifecycle
facts.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal, Protocol

from polylogue.core.enums import AssertionStatus, AssertionVisibility, PolylogueStrEnum
from polylogue.core.json import JSONValue


class EvidenceIntegrityStatus(PolylogueStrEnum):
    """Authoritative 37t.14 evidence-integrity vocabulary."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    STALE = "stale"
    CLOSED_LOOP = "closed_loop"
    CYCLE = "cycle"
    UNRESOLVED = "unresolved"
    FRAME_INCOMPLETE = "frame_incomplete"
    HELD_PRIVATE = "held_private"


class PublicClaimStatus(PolylogueStrEnum):
    """Stable public status vocabulary required by ``polylogue-3tl.16``."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"
    STALE_NEEDS_RERUN = "stale/needs-rerun"
    HELD_PRIVATE = "held_private"
    UNKNOWN = "unknown"
    CAPABILITY_ONLY = "capability-only"


class PublicClaimPresetName(PolylogueStrEnum):
    """Named public surfaces that parameterize one projection."""

    README = "readme"
    LAUNCH = "launch"
    FINDINGS_PAGE = "findings-page"
    VERIFIED_EXPORT = "verified-export"


PublicClaimSourceKind = Literal["finding", "capability"]
PublicClaimDisclosure = Literal["public", "held_private"]
PublicClaimReviewStatus = Literal["approved", "pending", "rejected", "deferred", "superseded", "unknown"]
PublicClaimPrivacyStatus = Literal["approved", "held_private", "pending", "unknown"]

_CLAIM_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_REF_KIND_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_PRIVATE_PATH_RE = re.compile(r"(?:/home/|/Users/|/realm/|/mnt/data/|[A-Za-z]:[\\/])")


def _public_text(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"public claim {field} must be non-empty")
    if "\x00" in normalized or _PRIVATE_PATH_RE.search(normalized):
        raise ValueError(f"public claim {field} contains a private or absolute path")
    return normalized


def _claim_key(value: str) -> str:
    normalized = value.strip()
    if not _CLAIM_KEY_RE.fullmatch(normalized):
        raise ValueError(f"public claim key is invalid: {value!r}")
    return normalized


def _public_ref(value: str, *, field: str) -> str:
    normalized = value.strip()
    kind, separator, object_id = normalized.partition(":")
    if (
        not separator
        or not _REF_KIND_RE.fullmatch(kind)
        or not object_id
        or any(char.isspace() for char in normalized)
        or "\x00" in normalized
    ):
        raise ValueError(f"public claim {field} must be a non-empty typed ref: {value!r}")

    path_text = object_id.split("#", maxsplit=1)[0]
    if _PRIVATE_PATH_RE.search(normalized):
        raise ValueError(f"public claim {field} contains a private or absolute path: {value!r}")
    if kind == "file" and (
        path_text.startswith("~")
        or PurePosixPath(path_text).is_absolute()
        or PureWindowsPath(path_text).is_absolute()
        or ".." in PurePosixPath(path_text).parts
        or ".." in PureWindowsPath(path_text).parts
    ):
        raise ValueError(f"public claim file ref must be repository-relative: {value!r}")
    return normalized


def _public_refs(values: Sequence[str], *, field: str, required: bool = False) -> tuple[str, ...]:
    refs = tuple(dict.fromkeys(_public_ref(value, field=field) for value in values))
    if required and not refs:
        raise ValueError(f"public claim {field} must contain at least one ref")
    return refs


def _public_codes(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    codes = tuple(dict.fromkeys(value.strip() for value in values))
    if any(not _CODE_RE.fullmatch(value) for value in codes):
        raise ValueError(f"public claim {field} must contain stable code tokens")
    return codes


def _optional_public_ref(value: str | None, *, field: str) -> str | None:
    return None if value is None else _public_ref(value, field=field)


def _optional_public_text(value: str | None, *, field: str) -> str | None:
    return None if value is None else _public_text(value, field=field)


def _public_json_value(value: JSONValue, *, field: str) -> JSONValue:
    """Copy one JSON value while rejecting private paths at every string leaf."""

    if isinstance(value, str):
        return _public_text(value, field=field)
    if isinstance(value, list):
        return [_public_json_value(item, field=f"{field}[]") for item in value]
    if isinstance(value, dict):
        sanitized: dict[str, JSONValue] = {}
        for key, item in value.items():
            public_key = _public_text(key, field=f"{field} key")
            if public_key in sanitized:
                raise ValueError(f"public claim {field} contains duplicate keys after normalization")
            sanitized[public_key] = _public_json_value(item, field=f"{field}.{public_key}")
        return sanitized
    return value


def _public_statistic(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    sanitized: dict[str, JSONValue] = {}
    for key, item in value.items():
        public_key = _public_text(key, field="statistic key")
        if public_key in sanitized:
            raise ValueError("public claim statistic contains duplicate keys after normalization")
        sanitized[public_key] = _public_json_value(item, field=f"statistic.{public_key}")
    return sanitized


@dataclass(frozen=True, slots=True)
class EvidenceIntegrityVerdict:
    """Narrow verdict interface consumed from the future 37t.14 evaluator.

    Free-form ancestry details are intentionally absent.  Public consumers get
    closed status/reason vocabularies and already-sanitized refs; decisive
    private paths remain inside the evaluator/storage boundary.
    """

    finding_ref: str
    status: EvidenceIntegrityStatus
    public_evidence_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    blind_spot_codes: tuple[str, ...] = ()
    as_of_epoch: str | None = None
    frame_ref: str | None = None
    definition_ref: str | None = None
    public_remediation_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_ref", _public_ref(self.finding_ref, field="finding_ref"))
        object.__setattr__(
            self,
            "public_evidence_refs",
            _public_refs(self.public_evidence_refs, field="public_evidence_refs"),
        )
        object.__setattr__(
            self,
            "public_remediation_refs",
            _public_refs(self.public_remediation_refs, field="public_remediation_refs"),
        )
        object.__setattr__(self, "reason_codes", _public_codes(self.reason_codes, field="reason_codes"))
        object.__setattr__(
            self,
            "blind_spot_codes",
            _public_codes(self.blind_spot_codes, field="blind_spot_codes"),
        )
        object.__setattr__(
            self,
            "as_of_epoch",
            _optional_public_text(self.as_of_epoch, field="as_of_epoch"),
        )
        object.__setattr__(
            self,
            "frame_ref",
            _optional_public_ref(self.frame_ref, field="frame_ref"),
        )
        object.__setattr__(
            self,
            "definition_ref",
            _optional_public_ref(self.definition_ref, field="definition_ref"),
        )
        if self.status in {
            EvidenceIntegrityStatus.SUPPORTED,
            EvidenceIntegrityStatus.PARTIALLY_SUPPORTED,
        } and (not self.as_of_epoch or not self.frame_ref or not self.definition_ref):
            raise ValueError("supported evidence-integrity verdicts require as_of_epoch, frame_ref, and definition_ref")


class EvidenceIntegrityProvider(Protocol):
    """37t.14 dependency seam; implementations return one current verdict."""

    def verdict_for(self, finding_ref: str) -> EvidenceIntegrityVerdict | None:
        """Return the authoritative verdict for ``finding_ref`` when computed."""


@dataclass(frozen=True, slots=True)
class MappingEvidenceIntegrityProvider:
    """Small adapter for persisted/exported verdict receipts and tests."""

    verdicts: Mapping[str, EvidenceIntegrityVerdict]

    def verdict_for(self, finding_ref: str) -> EvidenceIntegrityVerdict | None:
        return self.verdicts.get(finding_ref)


@dataclass(frozen=True, slots=True)
class PublicFindingInput:
    """Storage-neutral projection input for one FINDING assertion row."""

    assertion_ref: str
    claim_key: str
    publication: str
    scope: str
    caveat: str
    public_evidence_refs: tuple[str, ...]
    presets: tuple[PublicClaimPresetName, ...]
    disclosure: PublicClaimDisclosure
    statistic: Mapping[str, JSONValue]
    finding_epoch: str | None
    evaluation_ref: str | None
    frame_ref: str | None
    assertion_status: AssertionStatus
    assertion_visibility: AssertionVisibility
    author_kind: str
    judgment_ref: str | None
    judgment_decision: str | None
    supersedes: tuple[str, ...]
    updated_at_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "assertion_ref", _public_ref(self.assertion_ref, field="assertion_ref"))
        object.__setattr__(self, "claim_key", _claim_key(self.claim_key))
        for field_name in ("publication", "scope", "caveat"):
            object.__setattr__(self, field_name, _public_text(getattr(self, field_name), field=field_name))
        object.__setattr__(
            self,
            "public_evidence_refs",
            _public_refs(self.public_evidence_refs, field="public_evidence_refs", required=True),
        )
        object.__setattr__(
            self,
            "finding_epoch",
            _optional_public_text(self.finding_epoch, field="finding_epoch"),
        )
        object.__setattr__(
            self,
            "evaluation_ref",
            _optional_public_ref(self.evaluation_ref, field="evaluation_ref"),
        )
        object.__setattr__(
            self,
            "frame_ref",
            _optional_public_ref(self.frame_ref, field="frame_ref"),
        )
        object.__setattr__(
            self,
            "judgment_ref",
            _optional_public_ref(self.judgment_ref, field="judgment_ref"),
        )
        object.__setattr__(
            self,
            "supersedes",
            _public_refs(self.supersedes, field="supersedes"),
        )
        object.__setattr__(self, "statistic", _public_statistic(self.statistic))
        if not self.presets:
            raise ValueError("public finding must name at least one preset")
        if not self.author_kind.strip():
            raise ValueError("public finding author_kind must be non-empty")


@dataclass(frozen=True, slots=True)
class CapabilityClaimInput:
    """Explicit non-measurement claim; no FINDING or support verdict is implied."""

    claim_key: str
    publication: str
    scope: str
    caveat: str
    public_evidence_refs: tuple[str, ...]
    presets: tuple[PublicClaimPresetName, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_key", _claim_key(self.claim_key))
        for field_name in ("publication", "scope", "caveat"):
            object.__setattr__(self, field_name, _public_text(getattr(self, field_name), field=field_name))
        object.__setattr__(
            self,
            "public_evidence_refs",
            _public_refs(self.public_evidence_refs, field="public_evidence_refs", required=True),
        )
        if not self.presets:
            raise ValueError("capability claim must name at least one preset")


@dataclass(frozen=True, slots=True)
class PublicClaimProjection:
    """One sanitized, rendered claim row shared by every public preset."""

    claim_key: str
    source_kind: PublicClaimSourceKind
    source_ref: str | None
    status: PublicClaimStatus
    integrity_status: EvidenceIntegrityStatus | None
    integrity_verdict_present: bool
    publication: str
    scope: str
    caveat: str
    public_evidence_refs: tuple[str, ...]
    public_remediation_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]
    blind_spot_codes: tuple[str, ...]
    presets: tuple[PublicClaimPresetName, ...]
    statistic: Mapping[str, JSONValue] | None
    finding_epoch: str | None
    verdict_as_of_epoch: str | None
    finding_frame_ref: str | None
    verdict_frame_ref: str | None
    evaluation_ref: str | None
    definition_ref: str | None
    assertion_status: AssertionStatus | None
    publication_review: PublicClaimReviewStatus
    privacy_review: PublicClaimPrivacyStatus
    judgment_ref: str | None
    blocker_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_key", _claim_key(self.claim_key))
        for field_name in ("publication", "scope", "caveat"):
            object.__setattr__(self, field_name, _public_text(getattr(self, field_name), field=field_name))
        object.__setattr__(
            self,
            "public_evidence_refs",
            _public_refs(
                self.public_evidence_refs,
                field="public_evidence_refs",
                required=self.privacy_review != "held_private",
            ),
        )
        object.__setattr__(
            self,
            "public_remediation_refs",
            _public_refs(self.public_remediation_refs, field="public_remediation_refs"),
        )
        object.__setattr__(self, "reason_codes", _public_codes(self.reason_codes, field="reason_codes"))
        object.__setattr__(
            self,
            "blind_spot_codes",
            _public_codes(self.blind_spot_codes, field="blind_spot_codes"),
        )
        object.__setattr__(
            self,
            "blocker_codes",
            _public_codes(self.blocker_codes, field="blocker_codes"),
        )
        for field_name in (
            "source_ref",
            "finding_frame_ref",
            "verdict_frame_ref",
            "evaluation_ref",
            "definition_ref",
            "judgment_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_public_ref(getattr(self, field_name), field=field_name),
            )
        for field_name in ("finding_epoch", "verdict_as_of_epoch"):
            object.__setattr__(
                self,
                field_name,
                _optional_public_text(getattr(self, field_name), field=field_name),
            )
        if self.statistic is not None:
            object.__setattr__(self, "statistic", _public_statistic(self.statistic))
        if not self.presets:
            raise ValueError("public claim projection must name at least one preset")
        if self.source_kind == "capability":
            if (
                self.source_ref is not None
                or self.statistic is not None
                or self.integrity_status is not None
                or self.status is not PublicClaimStatus.CAPABILITY_ONLY
            ):
                raise ValueError("capability-only projections cannot carry finding or integrity state")
        elif (
            self.source_ref is None
            or (self.statistic is None and self.privacy_review != "held_private")
            or self.status is PublicClaimStatus.CAPABILITY_ONLY
        ):
            raise ValueError("finding projections require a source ref/statistic and cannot be capability-only")
        if self.status in {PublicClaimStatus.SUPPORTED, PublicClaimStatus.PARTIALLY_SUPPORTED}:
            expected_integrity = (
                EvidenceIntegrityStatus.SUPPORTED
                if self.status is PublicClaimStatus.SUPPORTED
                else EvidenceIntegrityStatus.PARTIALLY_SUPPORTED
            )
            if (
                self.integrity_status is not expected_integrity
                or not self.integrity_verdict_present
                or self.publication_review != "approved"
                or self.privacy_review != "approved"
                or not self.verdict_as_of_epoch
                or not self.verdict_frame_ref
                or not self.definition_ref
            ):
                raise ValueError("supported public claims require an approved qualified integrity verdict")

    @property
    def badge(self) -> str:
        """Return the compact badge used by every Markdown rendering."""

        if self.status is PublicClaimStatus.CAPABILITY_ONLY:
            return "[CAPABILITY ONLY]"
        if self.status is PublicClaimStatus.STALE_NEEDS_RERUN:
            return "[STALE / NEEDS RERUN]"
        if self.status is PublicClaimStatus.PARTIALLY_SUPPORTED:
            return "[PARTIALLY SUPPORTED]"
        if self.status is PublicClaimStatus.NOT_SUPPORTED:
            return "[NOT SUPPORTED]"
        if self.status is PublicClaimStatus.HELD_PRIVATE:
            return "[HELD PRIVATE]"
        if self.status is PublicClaimStatus.SUPPORTED:
            return "[SUPPORTED]"
        if self.integrity_status is not None:
            detail = self.integrity_status.value.replace("_", " ").upper()
            return f"[UNKNOWN · {detail}]"
        return "[UNKNOWN]"

    def to_payload(
        self,
        *,
        include_review: bool = True,
        include_reason_codes: bool = True,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "claim_key": self.claim_key,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "status": self.status.value,
            "integrity_status": None if self.integrity_status is None else self.integrity_status.value,
            "integrity_verdict_present": self.integrity_verdict_present,
            "badge": self.badge,
            "publication": self.publication,
            "scope": self.scope,
            "caveat": self.caveat,
            "public_evidence_refs": list(self.public_evidence_refs),
            "public_remediation_refs": list(self.public_remediation_refs),
            "presets": [preset.value for preset in self.presets],
            "statistic": None if self.statistic is None else dict(self.statistic),
            "qualifiers": {
                "finding_epoch": self.finding_epoch,
                "verdict_as_of_epoch": self.verdict_as_of_epoch,
                "finding_frame_ref": self.finding_frame_ref,
                "verdict_frame_ref": self.verdict_frame_ref,
                "evaluation_ref": self.evaluation_ref,
                "definition_ref": self.definition_ref,
            },
        }
        if include_review:
            payload["review"] = {
                "assertion_status": None if self.assertion_status is None else self.assertion_status.value,
                "publication": self.publication_review,
                "privacy": self.privacy_review,
                "judgment_ref": self.judgment_ref,
            }
        if include_reason_codes:
            payload.update(
                {
                    "reason_codes": list(self.reason_codes),
                    "blind_spot_codes": list(self.blind_spot_codes),
                    "blocker_codes": list(self.blocker_codes),
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class PublicClaimPreset:
    name: PublicClaimPresetName
    title: str
    include_review: bool
    include_reason_codes: bool


PUBLIC_CLAIM_PRESETS: Mapping[PublicClaimPresetName, PublicClaimPreset] = {
    PublicClaimPresetName.README: PublicClaimPreset(
        PublicClaimPresetName.README,
        "README Public Claims",
        include_review=False,
        include_reason_codes=False,
    ),
    PublicClaimPresetName.LAUNCH: PublicClaimPreset(
        PublicClaimPresetName.LAUNCH,
        "Launch Public Claims",
        include_review=False,
        include_reason_codes=True,
    ),
    PublicClaimPresetName.FINDINGS_PAGE: PublicClaimPreset(
        PublicClaimPresetName.FINDINGS_PAGE,
        "Findings-Page Public Claims",
        include_review=True,
        include_reason_codes=True,
    ),
    PublicClaimPresetName.VERIFIED_EXPORT: PublicClaimPreset(
        PublicClaimPresetName.VERIFIED_EXPORT,
        "Verified Public-Claims Export",
        include_review=True,
        include_reason_codes=True,
    ),
}


def project_public_claims(
    findings: Sequence[PublicFindingInput],
    capabilities: Sequence[CapabilityClaimInput],
    *,
    integrity: EvidenceIntegrityProvider,
) -> tuple[PublicClaimProjection, ...]:
    """Project one stable row per claim key without evaluating ancestry locally."""

    grouped: dict[str, list[PublicFindingInput]] = defaultdict(list)
    for finding in findings:
        grouped[finding.claim_key].append(finding)

    capability_by_key: dict[str, CapabilityClaimInput] = {}
    for capability in capabilities:
        if capability.claim_key in capability_by_key or capability.claim_key in grouped:
            raise ValueError(f"duplicate public claim key: {capability.claim_key}")
        capability_by_key[capability.claim_key] = capability

    projected: list[PublicClaimProjection] = []
    for claim_key in sorted(grouped):
        selected, selection_conflict = _select_current_finding(grouped[claim_key])
        projected.append(_project_finding(selected, integrity=integrity, selection_conflict=selection_conflict))

    for claim_key in sorted(capability_by_key):
        capability = capability_by_key[claim_key]
        projected.append(
            PublicClaimProjection(
                claim_key=capability.claim_key,
                source_kind="capability",
                source_ref=None,
                status=PublicClaimStatus.CAPABILITY_ONLY,
                integrity_status=None,
                integrity_verdict_present=False,
                publication=capability.publication,
                scope=capability.scope,
                caveat=capability.caveat,
                public_evidence_refs=tuple(dict.fromkeys(capability.public_evidence_refs)),
                public_remediation_refs=(),
                reason_codes=(),
                blind_spot_codes=(),
                presets=capability.presets,
                statistic=None,
                finding_epoch=None,
                verdict_as_of_epoch=None,
                finding_frame_ref=None,
                verdict_frame_ref=None,
                evaluation_ref=None,
                definition_ref=None,
                assertion_status=None,
                publication_review="approved",
                privacy_review="approved",
                judgment_ref=None,
                blocker_codes=(),
            )
        )

    return tuple(sorted(projected, key=lambda claim: claim.claim_key))


def _select_current_finding(findings: Sequence[PublicFindingInput]) -> tuple[PublicFindingInput, bool]:
    superseded_refs = {ref for finding in findings for ref in finding.supersedes}
    live = [
        finding
        for finding in findings
        if finding.assertion_ref not in superseded_refs
        and finding.assertion_status not in {AssertionStatus.SUPERSEDED, AssertionStatus.DELETED}
    ]
    if not live:
        live = list(findings)

    active = [finding for finding in live if finding.assertion_status is AssertionStatus.ACTIVE]
    if active:
        selected = max(active, key=lambda item: (item.updated_at_ms, item.assertion_ref))
        return selected, len(active) > 1

    selected = max(live, key=lambda item: (item.updated_at_ms, item.assertion_ref))
    return selected, False


def _publication_review(finding: PublicFindingInput) -> PublicClaimReviewStatus:
    status = finding.assertion_status
    if status is AssertionStatus.REJECTED:
        return "rejected"
    if status is AssertionStatus.DEFERRED:
        return "deferred"
    if status is AssertionStatus.SUPERSEDED:
        return "superseded"
    if status is AssertionStatus.CANDIDATE:
        return "pending"
    if status is AssertionStatus.ACCEPTED:
        return "approved"
    if status is not AssertionStatus.ACTIVE:
        return "unknown"

    promoted_review = finding.judgment_ref is not None and finding.judgment_decision in {"accept", "supersede"}
    direct_user_review = finding.author_kind == "user" and not finding.supersedes
    return "approved" if promoted_review or direct_user_review else "pending"


def _privacy_review(
    finding: PublicFindingInput,
    publication_review: PublicClaimReviewStatus,
) -> PublicClaimPrivacyStatus:
    if finding.disclosure == "held_private":
        return "held_private"
    if publication_review != "approved":
        return "pending"
    if finding.assertion_visibility is AssertionVisibility.PUBLIC or finding.disclosure == "public":
        return "approved"
    return "unknown"


def _project_finding(
    finding: PublicFindingInput,
    *,
    integrity: EvidenceIntegrityProvider,
    selection_conflict: bool,
) -> PublicClaimProjection:
    verdict = integrity.verdict_for(finding.assertion_ref)
    verdict_present = verdict is not None
    if verdict is None:
        verdict = EvidenceIntegrityVerdict(
            finding_ref=finding.assertion_ref,
            status=EvidenceIntegrityStatus.UNRESOLVED,
            reason_codes=("integrity-verdict-not-computed",),
            public_remediation_refs=("bead:polylogue-37t.14",),
        )
    elif verdict.finding_ref != finding.assertion_ref:
        raise ValueError(
            f"evidence-integrity verdict ref mismatch: {verdict.finding_ref!r} != {finding.assertion_ref!r}"
        )

    publication_review = _publication_review(finding)
    privacy_review = _privacy_review(finding, publication_review)
    if verdict.status is EvidenceIntegrityStatus.HELD_PRIVATE:
        privacy_review = "held_private"
    status, blocker_codes = _public_status(
        finding,
        verdict,
        selection_conflict=selection_conflict,
        publication_review=publication_review,
        privacy_review=privacy_review,
    )
    evidence_refs = tuple(dict.fromkeys((*finding.public_evidence_refs, *verdict.public_evidence_refs)))
    publication = finding.publication
    scope = finding.scope
    caveat = finding.caveat
    statistic: Mapping[str, JSONValue] | None = dict(finding.statistic)
    finding_epoch = finding.finding_epoch
    finding_frame_ref = finding.frame_ref
    evaluation_ref = finding.evaluation_ref
    verdict_as_of_epoch = verdict.as_of_epoch
    verdict_frame_ref = verdict.frame_ref
    definition_ref = verdict.definition_ref
    if privacy_review == "held_private":
        publication = "Claim text withheld pending public privacy review."
        scope = "The stable claim key remains visible; claim content and evidence are withheld."
        caveat = "No inference should be drawn from a privacy hold."
        evidence_refs = ()
        statistic = None
        finding_epoch = None
        finding_frame_ref = None
        evaluation_ref = None
        verdict_as_of_epoch = None
        verdict_frame_ref = None
        definition_ref = None

    return PublicClaimProjection(
        claim_key=finding.claim_key,
        source_kind="finding",
        source_ref=finding.assertion_ref,
        status=status,
        integrity_status=verdict.status,
        integrity_verdict_present=verdict_present,
        publication=publication,
        scope=scope,
        caveat=caveat,
        public_evidence_refs=evidence_refs,
        public_remediation_refs=verdict.public_remediation_refs,
        reason_codes=verdict.reason_codes,
        blind_spot_codes=verdict.blind_spot_codes,
        presets=finding.presets,
        statistic=statistic,
        finding_epoch=finding_epoch,
        verdict_as_of_epoch=verdict_as_of_epoch,
        finding_frame_ref=finding_frame_ref,
        verdict_frame_ref=verdict_frame_ref,
        evaluation_ref=evaluation_ref,
        definition_ref=definition_ref,
        assertion_status=finding.assertion_status,
        publication_review=publication_review,
        privacy_review=privacy_review,
        judgment_ref=finding.judgment_ref,
        blocker_codes=blocker_codes,
    )


def _public_status(
    finding: PublicFindingInput,
    verdict: EvidenceIntegrityVerdict,
    *,
    selection_conflict: bool,
    publication_review: PublicClaimReviewStatus,
    privacy_review: PublicClaimPrivacyStatus,
) -> tuple[PublicClaimStatus, tuple[str, ...]]:
    blockers: list[str] = []
    if privacy_review == "held_private":
        if finding.disclosure == "held_private":
            blockers.append("publication-held-private")
        if verdict.status is EvidenceIntegrityStatus.HELD_PRIVATE:
            blockers.append("integrity-held_private")
        return PublicClaimStatus.HELD_PRIVATE, tuple(blockers)

    if selection_conflict:
        blockers.append("multiple-live-findings")
        return PublicClaimStatus.UNKNOWN, tuple(blockers)

    if finding.assertion_status is AssertionStatus.REJECTED:
        blockers.append("finding-rejected")
        return PublicClaimStatus.NOT_SUPPORTED, tuple(blockers)
    if finding.assertion_status in {
        AssertionStatus.CANDIDATE,
        AssertionStatus.ACCEPTED,
        AssertionStatus.DEFERRED,
        AssertionStatus.SUPERSEDED,
        AssertionStatus.DELETED,
    }:
        blockers.append(f"finding-{finding.assertion_status.value}")
        return PublicClaimStatus.UNKNOWN, tuple(blockers)

    if publication_review != "approved":
        blockers.append("publication-review-pending")
        return PublicClaimStatus.UNKNOWN, tuple(blockers)
    if privacy_review != "approved":
        blockers.append("privacy-review-pending")
        return PublicClaimStatus.UNKNOWN, tuple(blockers)

    status_map: Mapping[EvidenceIntegrityStatus, PublicClaimStatus] = {
        EvidenceIntegrityStatus.SUPPORTED: PublicClaimStatus.SUPPORTED,
        EvidenceIntegrityStatus.PARTIALLY_SUPPORTED: PublicClaimStatus.PARTIALLY_SUPPORTED,
        EvidenceIntegrityStatus.NOT_SUPPORTED: PublicClaimStatus.NOT_SUPPORTED,
        EvidenceIntegrityStatus.STALE: PublicClaimStatus.STALE_NEEDS_RERUN,
        EvidenceIntegrityStatus.HELD_PRIVATE: PublicClaimStatus.HELD_PRIVATE,
        EvidenceIntegrityStatus.CLOSED_LOOP: PublicClaimStatus.UNKNOWN,
        EvidenceIntegrityStatus.CYCLE: PublicClaimStatus.UNKNOWN,
        EvidenceIntegrityStatus.UNRESOLVED: PublicClaimStatus.UNKNOWN,
        EvidenceIntegrityStatus.FRAME_INCOMPLETE: PublicClaimStatus.UNKNOWN,
    }
    projected = status_map[verdict.status]
    if projected is not PublicClaimStatus.SUPPORTED:
        blockers.append(f"integrity-{verdict.status.value}")
    return projected, tuple(blockers)


def claims_for_preset(
    claims: Sequence[PublicClaimProjection],
    preset_name: PublicClaimPresetName,
) -> tuple[PublicClaimProjection, ...]:
    """Filter one already-computed projection; status is never recomputed."""

    return tuple(claim for claim in claims if preset_name in claim.presets)


def build_public_claims_payload(
    claims: Sequence[PublicClaimProjection],
    preset_name: PublicClaimPresetName,
) -> dict[str, object]:
    """Return the shared machine rendering for one preset."""

    preset = PUBLIC_CLAIM_PRESETS[preset_name]
    selected = claims_for_preset(claims, preset_name)
    return {
        "schema": "polylogue.public-claims-view.v1",
        "preset": preset_name.value,
        "authority": {
            "claims": "AssertionKind.FINDING + explicit capability declarations",
            "integrity": "polylogue-37t.14 EvidenceIntegrityVerdict",
            "projection": "polylogue/insights/measurement/public_claims.py",
        },
        "claim_count": len(selected),
        "publishable_claim_keys": [
            claim.claim_key
            for claim in selected
            if claim.status
            in {
                PublicClaimStatus.SUPPORTED,
                PublicClaimStatus.PARTIALLY_SUPPORTED,
                PublicClaimStatus.CAPABILITY_ONLY,
            }
        ],
        "claims": [
            claim.to_payload(
                include_review=preset.include_review,
                include_reason_codes=preset.include_reason_codes,
            )
            for claim in selected
        ],
    }


def render_public_claims_json(
    claims: Sequence[PublicClaimProjection],
    preset_name: PublicClaimPresetName,
) -> str:
    return json.dumps(build_public_claims_payload(claims, preset_name), indent=2, sort_keys=True) + "\n"


def render_public_claims_markdown(
    claims: Sequence[PublicClaimProjection],
    preset_name: PublicClaimPresetName,
) -> str:
    """Render one preset through the same generic Markdown implementation."""

    preset = PUBLIC_CLAIM_PRESETS[preset_name]
    selected = claims_for_preset(claims, preset_name)
    lines = [
        f"# {preset.title}",
        "",
        (
            "This file is generated from FINDING assertions, canonical judgment state, and the shared "
            "evidence-integrity verdict. A badge is a current publication status, not a substitute for the cited evidence."
        ),
        "",
    ]
    for claim in selected:
        lines.extend(
            [
                f"## `{claim.claim_key}` {claim.badge}",
                "",
                claim.publication,
                "",
                f"- Scope: {claim.scope}",
                f"- Caveat: {claim.caveat}",
                "- Evidence: " + _markdown_refs(claim.public_evidence_refs),
                "- Epoch/frame: " + _markdown_qualifiers(claim),
            ]
        )
        if preset.include_review:
            lines.append(
                "- Review: "
                f"assertion={_optional_enum(claim.assertion_status)}, "
                f"publication={claim.publication_review}, privacy={claim.privacy_review}, "
                f"judgment={claim.judgment_ref or 'none'}"
            )
        if preset.include_reason_codes and (claim.reason_codes or claim.blocker_codes):
            codes = tuple(dict.fromkeys((*claim.blocker_codes, *claim.reason_codes)))
            lines.append("- Status reasons: " + ", ".join(f"`{code}`" for code in codes))
        if claim.public_remediation_refs:
            lines.append("- Remediation: " + _markdown_refs(claim.public_remediation_refs))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _optional_enum(value: AssertionStatus | None) -> str:
    return "n/a" if value is None else value.value


def _markdown_refs(refs: Sequence[str]) -> str:
    if not refs:
        return "none published"
    return ", ".join(f"`{ref}`" for ref in refs)


def _markdown_qualifiers(claim: PublicClaimProjection) -> str:
    if claim.source_kind == "capability":
        return "not applicable (capability statement; no measured result)"
    values = (
        f"finding epoch={claim.finding_epoch or 'unknown'}",
        f"verdict as-of={claim.verdict_as_of_epoch or 'unknown'}",
        f"finding frame={claim.finding_frame_ref or 'unknown'}",
        f"verdict frame={claim.verdict_frame_ref or 'unknown'}",
        f"evaluation={claim.evaluation_ref or 'unknown'}",
        f"definition={claim.definition_ref or 'unknown'}",
    )
    return "; ".join(values)


__all__ = [
    "CapabilityClaimInput",
    "EvidenceIntegrityProvider",
    "EvidenceIntegrityStatus",
    "EvidenceIntegrityVerdict",
    "MappingEvidenceIntegrityProvider",
    "PUBLIC_CLAIM_PRESETS",
    "PublicClaimDisclosure",
    "PublicClaimPreset",
    "PublicClaimPresetName",
    "PublicClaimProjection",
    "PublicClaimStatus",
    "PublicFindingInput",
    "build_public_claims_payload",
    "claims_for_preset",
    "project_public_claims",
    "render_public_claims_json",
    "render_public_claims_markdown",
]
