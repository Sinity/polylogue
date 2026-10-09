"""Closed, bounded terminal receipts retained only by audit operation events.

These models are separate from the paged source-generation evidence. That
evidence is a current projection over live source and index
tiers; these values are the immutable fact checkpointed when a machine
operation terminalizes.  They are the only domain payload eligible for the
audit continuity command and final event.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

MAX_MACHINE_RECEIPT_PAGES = 40
MAX_PAGE_ITEMS = 256
MAX_RAW_IDS_PER_INPUT = 10_000
MAX_INLINE_RAW_IDS_PER_INPUT = MAX_RAW_IDS_PER_INPUT
MAX_INLINE_INGEST_SESSION_IDS = 10_000


def ingest_session_ids_digest(session_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(session_ids, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class InsightCertifiedCountsHistorical(_Receipt):
    profiles: int = Field(ge=0)


class InsightTargetHistoricalReceipt(_Receipt):
    target_ref: str = Field(min_length=1)
    disposition: Literal["already_satisfied", "published", "pending", "stale", "failed", "unknown"]
    input_binding: str | None = None
    output_binding: str | None = None
    certified_counts: InsightCertifiedCountsHistorical
    publication_known_committed: bool

    @model_validator(mode="after")
    def valid_publication(self) -> InsightTargetHistoricalReceipt:
        if not self.target_ref.startswith("session:"):
            raise ValueError("historical insight target must be a session reference")
        if self.disposition == "published" and not self.publication_known_committed:
            raise ValueError("published historical insight target lacks committed publication")
        return self


class InsightTerminalSummaryHistorical(_Receipt):
    profiles: int = Field(ge=0)
    threads: int = Field(ge=0)
    tag_rollups: int = Field(ge=0)


class InsightPartHistoricalReceipt(_Receipt):
    kind: Literal["insight-part/v1"] = "insight-part/v1"
    ordinal: int = Field(ge=0)
    page_count: int = Field(ge=1)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    index_generation: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    targets: list[InsightTargetHistoricalReceipt] = Field(default_factory=list, max_length=MAX_PAGE_ITEMS)
    unattempted_target_refs: list[str] = Field(default_factory=list, max_length=MAX_PAGE_ITEMS)
    terminal_summary: InsightTerminalSummaryHistorical | None = None

    @model_validator(mode="after")
    def valid_page(self) -> InsightPartHistoricalReceipt:
        if self.ordinal >= self.page_count:
            raise ValueError("historical insight ordinal is outside its page count")
        refs = [target.target_ref for target in self.targets]
        if len(refs) != len(set(refs)) or len(self.unattempted_target_refs) != len(set(self.unattempted_target_refs)):
            raise ValueError("historical insight page repeats a target")
        if set(refs) & set(self.unattempted_target_refs):
            raise ValueError("historical insight page overlaps attempted and unattempted targets")
        if len(refs) + len(self.unattempted_target_refs) > MAX_PAGE_ITEMS:
            raise ValueError("historical insight page exceeds its target budget")
        return self


class IngestInputHistoricalReceipt(_Receipt):
    """One retained input's exact terminal attribution, or an explicit gap."""

    source_item_id: str = Field(min_length=1)
    logical_coordinate: str = Field(min_length=1)
    denominator: int = Field(ge=0)
    raw_ids: list[str] | None = Field(default=None, max_length=MAX_RAW_IDS_PER_INPUT)
    unresolved_raw_ids: list[str] = Field(default_factory=list, max_length=MAX_RAW_IDS_PER_INPUT)
    raw_id_pages_ref: str | None = Field(default=None, min_length=1)
    raw_id_page_count: int = Field(default=0, ge=0)
    raw_id_count: int = Field(default=0, ge=0)
    unresolved_raw_count: int = Field(default=0, ge=0)
    raw_ids_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    unknown_attribution: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def valid_attribution(self) -> IngestInputHistoricalReceipt:
        paged = self.raw_id_pages_ref is not None
        if paged != bool(self.raw_id_page_count) or paged != bool(self.raw_ids_digest):
            raise ValueError("historical raw attribution page reference is incomplete")
        if paged and (
            self.raw_ids is not None
            or self.unresolved_raw_ids
            or self.raw_id_count <= MAX_INLINE_RAW_IDS_PER_INPUT
            or self.raw_id_page_count != (self.raw_id_count + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS
            or self.unresolved_raw_count > self.raw_id_count
            or self.denominator < self.raw_id_count
        ):
            raise ValueError("historical raw attribution pages disagree with their count")
        if self.raw_ids is None and self.unknown_attribution is None and not paged:
            raise ValueError("missing raw attribution must name an explicit unknown reason")
        if self.raw_ids is not None:
            if len(self.raw_ids) != len(set(self.raw_ids)):
                raise ValueError("historical ingest input repeats a raw id")
            if not set(self.unresolved_raw_ids).issubset(self.raw_ids):
                raise ValueError("historical unresolved raw id is not attributed to its input")
            if self.denominator < len(self.raw_ids):
                raise ValueError("historical ingest denominator is smaller than known membership")
        elif self.unresolved_raw_ids:
            raise ValueError("unknown attribution cannot invent unresolved raw ids")
        if not paged and any((self.raw_id_count, self.unresolved_raw_count)):
            raise ValueError("inline raw attribution cannot claim referenced counts")
        return self


class IngestInputRawMemberHistorical(_Receipt):
    raw_id: str = Field(min_length=1)
    unresolved: bool


class IngestInputRawPageHistoricalReceipt(_Receipt):
    source_item_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    raws: list[IngestInputRawMemberHistorical] = Field(min_length=1, max_length=MAX_PAGE_ITEMS)

    @model_validator(mode="after")
    def valid_page(self) -> IngestInputRawPageHistoricalReceipt:
        ids = [raw.raw_id for raw in self.raws]
        if ids != sorted(set(ids)):
            raise ValueError("historical input raw page repeats or reorders a raw")
        return self


def ingest_input_raw_pages_digest(pages: Iterable[IngestInputRawPageHistoricalReceipt]) -> str:
    digest = IngestInputRawPagesDigest()
    for page in pages:
        digest.update(page)
    return digest.hexdigest()


def _page_digest(items: list[IngestInputHistoricalReceipt]) -> str:
    payload = [item.model_dump(mode="json") for item in items]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class IngestInputPageHistoricalReceipt(_Receipt):
    ordinal: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[IngestInputHistoricalReceipt] = Field(min_length=1, max_length=MAX_PAGE_ITEMS)

    @model_validator(mode="after")
    def valid_digest(self) -> IngestInputPageHistoricalReceipt:
        if self.digest != _page_digest(self.items):
            raise ValueError("historical ingest page digest does not bind its items")
        return self

    @classmethod
    def from_items(cls, ordinal: int, items: list[IngestInputHistoricalReceipt]) -> IngestInputPageHistoricalReceipt:
        return cls(ordinal=ordinal, digest=_page_digest(items), items=items)


class IngestInsightPageHistoricalReceipt(_Receipt):
    ordinal: int = Field(ge=0)
    targets: list[InsightTargetHistoricalReceipt] = Field(default_factory=list, max_length=MAX_PAGE_ITEMS)
    unattempted_target_refs: list[str] = Field(default_factory=list, max_length=MAX_PAGE_ITEMS)

    @model_validator(mode="after")
    def valid_page(self) -> IngestInsightPageHistoricalReceipt:
        refs = [target.target_ref for target in self.targets]
        if len(refs) != len(set(refs)) or len(self.unattempted_target_refs) != len(set(self.unattempted_target_refs)):
            raise ValueError("historical ingest insight page repeats a target")
        if set(refs) & set(self.unattempted_target_refs):
            raise ValueError("historical ingest insight page overlaps attempted and unattempted targets")
        if len(refs) + len(self.unattempted_target_refs) > MAX_PAGE_ITEMS:
            raise ValueError("historical ingest insight page exceeds its target budget")
        return self


def ingest_insight_pages_digest(pages: list[IngestInsightPageHistoricalReceipt]) -> str:
    payload = [page.model_dump(mode="json") for page in pages]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class IngestRefusedMembershipHistorical(_Receipt):
    """One logical source key whose cohort could not be prepared."""

    logical_source_key: str = Field(min_length=1)
    raw_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=512)


class IngestRefusalPageHistoricalReceipt(_Receipt):
    """One page of refused memberships, retained in audit before the root cites it."""

    ordinal: int = Field(ge=0)
    refusals: list[IngestRefusedMembershipHistorical] = Field(min_length=1, max_length=MAX_PAGE_ITEMS)


class _CanonicalPagesDigest:
    """Canonical JSON array digest of receipt pages, one page at a time.

    Equal to the SHA-256 of the canonical JSON array of the pages, so a
    builder can persist each page and forget it.
    """

    def __init__(self) -> None:
        self._hasher = hashlib.sha256(b"[")
        self._pages = 0

    def update(self, page: _Receipt) -> None:
        if self._pages:
            self._hasher.update(b",")
        self._hasher.update(json.dumps(page.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode())
        self._pages += 1

    def hexdigest(self) -> str:
        final = self._hasher.copy()
        final.update(b"]")
        return final.hexdigest()


class IngestRefusalPagesDigest(_CanonicalPagesDigest):
    """Incremental canonical JSON array digest for refusal pages."""


class IngestInputRawPagesDigest(_CanonicalPagesDigest):
    """Incremental canonical JSON array digest for input raw pages."""


def ingest_refusal_pages_digest(pages: Iterable[IngestRefusalPageHistoricalReceipt]) -> str:
    digest = IngestRefusalPagesDigest()
    for page in pages:
        digest.update(page)
    return digest.hexdigest()


class IngestTerminalSummaryHistorical(_Receipt):
    enumeration_complete: bool
    source_complete: bool
    confirmed_raw_count: int = Field(ge=0)
    unresolved_raw_count: int = Field(ge=0)
    profile_targets_observed: int = Field(ge=0)
    # polylogue-163ku: a per-key cohort refusal no longer fences the whole
    # source generation, so the count of refusals it *did* cost must be
    # reported here -- a skipped key that nothing counts would be a silent
    # degradation traded for the loud one.
    refused_membership_count: int = Field(ge=0, default=0)
    # Every refusal is named: inline up to one page, otherwise in referenced
    # audit pages that together hold exactly ``refused_membership_count``.
    refused_memberships: list[IngestRefusedMembershipHistorical] = Field(
        default_factory=list, max_length=MAX_PAGE_ITEMS
    )
    refused_membership_pages_ref: str | None = Field(default=None, min_length=1)
    refused_membership_page_count: int = Field(default=0, ge=0)
    refused_memberships_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # Captured by the admitted writer during cohort publication. Public API
    # clients project ParseResult from this terminal fact, never a later index
    # read that may already describe another generation or replacement.
    parse_projection_known: bool = False
    processed_session_ids: list[str] = Field(default_factory=list, max_length=MAX_INLINE_INGEST_SESSION_IDS)
    processed_session_id_pages_ref: str | None = Field(default=None, min_length=1)
    processed_session_id_page_count: int = Field(default=0, ge=0)
    processed_session_ids_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    processed_message_count: int = Field(default=0, ge=0)
    changed_session_count: int = Field(default=0, ge=0)
    changed_message_count: int = Field(default=0, ge=0)
    # Whether every session this generation proved was carried through profile
    # and insight convergence. ``None`` only on receipts written before the
    # fact was recorded. A committed ingest whose convergence stopped on a
    # retryable target is ``degraded``, never ``completed`` (polylogue-5639
    # review): its rows are durable, its derived readiness is not.
    profile_convergence_complete: bool | None = None

    @property
    def converged(self) -> bool | None:
        """Whether the ingest's source and derived convergence both finished."""
        if not self.source_complete:
            # Known on every receipt, including ones written before profile
            # convergence was recorded.
            return False
        return self.profile_convergence_complete

    @model_validator(mode="after")
    def valid_refusals(self) -> IngestTerminalSummaryHistorical:
        paged = self.refused_membership_pages_ref is not None
        if paged != bool(self.refused_membership_page_count) or paged != bool(self.refused_memberships_digest):
            raise ValueError("ingest refusal page reference is incomplete")
        if paged:
            if self.refused_memberships or self.refused_membership_count <= MAX_PAGE_ITEMS:
                raise ValueError("ingest refusal pages are only for more refusals than one page")
            if (
                self.refused_membership_page_count
                != (self.refused_membership_count + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS
            ):
                raise ValueError("ingest refusal page count does not match its refusals")
        elif self.refused_membership_count != len(self.refused_memberships):
            raise ValueError("ingest refusal count does not match its named refusals")
        return self

    @model_validator(mode="after")
    def valid_parse_projection(self) -> IngestTerminalSummaryHistorical:
        if len(self.processed_session_ids) != len(set(self.processed_session_ids)):
            raise ValueError("ingest parse projection repeats a session")
        paged = self.processed_session_id_pages_ref is not None
        if paged != bool(self.processed_session_id_page_count) or paged != bool(self.processed_session_ids_digest):
            raise ValueError("ingest parse projection page reference is incomplete")
        if paged and self.processed_session_ids:
            raise ValueError("paged ingest parse projection cannot also inline session ids")
        if self.parse_projection_known:
            if not paged and self.changed_session_count != len(self.processed_session_ids):
                raise ValueError("ingest parse projection session count does not match its ids")
            if paged and self.changed_session_count <= MAX_INLINE_INGEST_SESSION_IDS:
                raise ValueError("ingest parse projection pages require more than 10000 sessions")
            if (
                paged
                and self.processed_session_id_page_count
                != (self.changed_session_count + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS
            ):
                raise ValueError("ingest parse projection page count does not match its sessions")
            if self.changed_message_count != self.processed_message_count:
                raise ValueError("ingest parse projection message counts disagree")
        elif (
            self.processed_session_ids
            or paged
            or any((self.processed_message_count, self.changed_session_count, self.changed_message_count))
        ):
            raise ValueError("unknown ingest parse projection cannot invent changed rows")
        return self


def ingest_input_pages_digest(pages: list[IngestInputPageHistoricalReceipt]) -> str:
    """Bind the exact ordered page sequence without copying its item payloads."""
    digest = hashlib.sha256()
    for page in pages:
        digest.update(f"{page.ordinal}:{page.digest}\n".encode("ascii"))
    return digest.hexdigest()


class IngestHistoricalReceiptV2(_Receipt):
    """Terminal root whose input denominator lives in immutable audit pages."""

    kind: Literal["ingest/v2"] = "ingest/v2"
    source_generation_id: str = Field(min_length=1)
    final_sequence: int = Field(ge=1)
    input_count: int = Field(ge=1)
    input_pages_ref: str = Field(min_length=1)
    input_page_count: int = Field(ge=1)
    input_pages_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    insight_pages: list[IngestInsightPageHistoricalReceipt] = Field(
        default_factory=list, max_length=MAX_MACHINE_RECEIPT_PAGES
    )
    insight_pages_ref: str | None = Field(default=None, min_length=1)
    insight_page_count: int = Field(default=0, ge=0)
    insight_pages_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    summary: IngestTerminalSummaryHistorical

    @model_validator(mode="after")
    def valid_root(self) -> IngestHistoricalReceiptV2:
        if self.input_page_count != (self.input_count + MAX_PAGE_ITEMS - 1) // MAX_PAGE_ITEMS:
            raise ValueError("historical ingest input page count differs from denominator")
        paged_insights = self.insight_pages_ref is not None
        if paged_insights != bool(self.insight_page_count) or paged_insights != bool(self.insight_pages_digest):
            raise ValueError("historical ingest insight page reference is incomplete")
        if paged_insights and self.insight_pages:
            raise ValueError("historical ingest cannot mix inline and referenced insight pages")
        if [page.ordinal for page in self.insight_pages] != list(range(len(self.insight_pages))):
            raise ValueError("historical ingest insight pages are not contiguous")
        if not paged_insights and self.summary.profile_targets_observed != sum(
            len(page.targets) for page in self.insight_pages
        ):
            raise ValueError("historical ingest profile total does not match its pages")
        if paged_insights and self.summary.profile_targets_observed > self.insight_page_count * MAX_PAGE_ITEMS:
            raise ValueError("historical ingest profile total exceeds its referenced pages")
        if self.summary.refused_membership_count and self.summary.source_complete:
            raise ValueError("historical ingest cannot be source-complete while memberships were refused")
        return self


class IdentityResetHistoricalReceipt(_Receipt):
    kind: Literal["identity-reset/v1"] = "identity-reset/v1"
    count_scope: Literal["completing-apply"] = "completing-apply"
    suppressed_count: int = Field(ge=0)
    deleted_archive_rows: int = Field(ge=0)
    tombstoned_without_index_row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def consistent_counts(self) -> IdentityResetHistoricalReceipt:
        if self.deleted_archive_rows > self.suppressed_count:
            raise ValueError("deleted rows exceed suppression targets")
        if self.tombstoned_without_index_row_count > self.suppressed_count:
            raise ValueError("absent rows exceed suppression targets")
        return self

    def result_counts(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"kind"})


IngestTerminalReceipt: TypeAlias = IngestHistoricalReceiptV2
MachineHistoricalReceipt: TypeAlias = (
    InsightPartHistoricalReceipt | IngestTerminalReceipt | IdentityResetHistoricalReceipt
)


def ingest_terminal_outcome(history: IngestHistoricalReceiptV2) -> str:
    """``completed`` only when the committed ingest also finished converging.

    The ingest's rows are durable either way. When source enumeration refused
    members or profile convergence stopped on a retryable target, the daemon's
    ordinary convergence owns the rest, and the terminal outcome says so
    instead of certifying readiness it did not reach.
    """
    summary = getattr(history, "summary", None)
    converged = getattr(summary, "converged", None)
    return "degraded" if converged is False else "completed"


def ingest_unconverged_error(history: IngestHistoricalReceiptV2) -> dict[str, object]:
    summary = getattr(history, "summary", None)
    source_complete = bool(getattr(summary, "source_complete", False))
    if not source_complete:
        # Recorded membership refusals and unresolved raws are this source
        # generation's terminal facts; nothing retries them at this head.
        return {
            "code": "ingest_source_incomplete",
            "detail": (
                "ingest committed its rows, but source admission refused or could not resolve some members; "
                "the receipt names them"
            ),
            "retryable": False,
            "data": {
                "source_complete": False,
                "refused_membership_count": getattr(summary, "refused_membership_count", 0),
                "unresolved_raw_count": getattr(summary, "unresolved_raw_count", 0),
                "profile_convergence_complete": getattr(summary, "profile_convergence_complete", None),
            },
        }
    return {
        "code": "ingest_convergence_pending",
        "detail": (
            "ingest committed its rows, but profile and insight convergence did not finish; "
            "the daemon's convergence continues it"
        ),
        "retryable": True,
        "data": {
            "source_complete": source_complete,
            "profile_convergence_complete": getattr(summary, "profile_convergence_complete", None),
        },
    }


def encode_machine_receipt(receipt: MachineHistoricalReceipt) -> dict[str, object]:
    """Return the sole JSON representation accepted by audit persistence."""

    return receipt.model_dump(mode="json")


def decode_machine_receipt(raw: object) -> MachineHistoricalReceipt:
    """Reject arbitrary JSON rather than treating audit detail as an open payload."""

    if not isinstance(raw, dict):
        raise ValueError("historical machine receipt is not an object")
    try:
        kind = raw.get("kind")
        if kind == "identity-reset/v1":
            return IdentityResetHistoricalReceipt.model_validate(raw)
        if kind == "insight-part/v1":
            return InsightPartHistoricalReceipt.model_validate(raw)
        if kind == "ingest/v2":
            return IngestHistoricalReceiptV2.model_validate(raw)
    except ValidationError as exc:
        raise ValueError("historical machine receipt is malformed") from exc
    raise ValueError("historical machine receipt kind is not recognized")


__all__ = [
    "ingest_terminal_outcome",
    "ingest_unconverged_error",
    "IngestRefusalPageHistoricalReceipt",
    "IngestHistoricalReceiptV2",
    "IngestTerminalReceipt",
    "IngestInputHistoricalReceipt",
    "IngestInputPageHistoricalReceipt",
    "IngestInsightPageHistoricalReceipt",
    "IngestTerminalSummaryHistorical",
    "InsightCertifiedCountsHistorical",
    "InsightPartHistoricalReceipt",
    "InsightTargetHistoricalReceipt",
    "MachineHistoricalReceipt",
    "IdentityResetHistoricalReceipt",
    "decode_machine_receipt",
    "encode_machine_receipt",
    "ingest_input_pages_digest",
    "IngestRefusalPagesDigest",
    "ingest_refusal_pages_digest",
]
