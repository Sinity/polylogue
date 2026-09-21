"""Bounded Hermes-to-Polylogue integration health rollup (polylogue-fs1.15).

fs1.15 asked for one composed, read-only view of Hermes integration
liveness: enabled/disabled, per-source-class freshness/cursor position,
parser/materializer failures, unpaired/debt counts, latest imported
evidence refs, and context-delivery correlation state — with explicit
degraded states rather than a crash or a silent zero.

Per the bead's own design note ("compose EXISTING daemon health, source
cursors, OriginSpec fidelity, hook liveness, and context-delivery records
... no new monitoring database"), and per the 2026-07-18 investigation note
recorded on the bead, the per-source freshness primitive
(``project_named_source_freshness``, polylogue-1xc.13) already proves
Hermes coverage. This module adds only the rollup on top, composing:

- :func:`polylogue.sources.import_explain.explain_import_path` — a bounded,
  non-mutating dry-run parse of every file under the Hermes runtime root,
  already returning per-file fidelity declarations and parser-failure
  reasons (no new parser logic).
- :func:`polylogue.archive.query.source_freshness.project_named_source_freshness`
  — per-file freshness/cursor/parse/index/FTS evidence.
- :func:`polylogue.daemon.convergence_debt_status.convergence_debt_summary_info`
  — durable post-ingest convergence debt, bucketed to the Hermes source
  family. This module does not import ``polylogue.daemon`` directly
  (``docs/plans/layering.yaml`` forbids insights -> daemon): the caller
  (a surface adapter, e.g. ``polylogue.api.archive``) computes the
  Hermes-scoped debt counts and passes them in.
- :func:`polylogue.context.hermes_lifecycle_reconciliation.reconcile_hermes_session_lifecycle`
  — per-session lifecycle-event pairing debt (fs1.7).
- :func:`polylogue.context.hermes_delivery_correlation.correlate_hermes_context_deliveries`
  — per-session context-delivery correlation state (fs1.11).

No new schema, table, or persistent write path is introduced. The response
never carries raw transcript text, credentials, or absolute filesystem
paths: source references are filenames only (``source_ref``), and evidence
refs are ids/hashes, never rendered bytes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from polylogue.archive.query.source_freshness import project_named_source_freshness
from polylogue.context.hermes_delivery_correlation import correlate_hermes_context_deliveries
from polylogue.context.hermes_lifecycle_reconciliation import reconcile_hermes_session_lifecycle
from polylogue.core.enums import Origin
from polylogue.logging import get_logger
from polylogue.sources.import_explain import explain_import_path
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

logger = get_logger(__name__)

HermesHealthVerdict = Literal["disabled", "healthy", "degraded", "unavailable"]

_DEFAULT_SOURCE_LIMIT = 25
_DEFAULT_SESSION_LIMIT = 10


@dataclass(frozen=True, slots=True)
class HermesSourceStatus:
    """Freshness/cursor evidence for one discovered Hermes source file."""

    source_ref: str
    source_class: Literal["state_db", "verification_evidence_db", "atof_stream", "atif_document", "other"]
    stage: str
    operational_state: str
    operational_reason: str
    parse_state: str
    byte_lag_bytes: int | None
    fts_converged: bool
    insights_converged: bool
    projection_error_count: int
    session_ref: str | None


@dataclass(frozen=True, slots=True)
class HermesParserFailure:
    """One file the dry-run explain pass could not parse or read."""

    source_ref: str
    reason: str


@dataclass(frozen=True, slots=True)
class HermesFidelityCapabilityStatus:
    """Aggregated fidelity-capability status across discovered Hermes sources."""

    capability: str
    status: str
    observed: int
    expected: int
    detail: str
    source_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesLifecycleDebtSummary:
    """Lifecycle-event pairing debt (fs1.7) across sampled Hermes sessions."""

    sessions_checked: int
    total_events: int
    unpaired_event_count: int
    unknown_message_reference_count: int
    caveats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesDeliveryCorrelationSummary:
    """Context-delivery correlation state (fs1.11) across sampled Hermes sessions."""

    sessions_checked: int
    events_checked: int
    available_count: int
    unavailable_count: int
    caveats: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HermesMeasurementCoverage:
    """What this rollup actually measured, and what it tried to and could not.

    The verdict is only as good as its inputs, so the inputs are reported
    beside it. Each probe contributes two distinct counters -- what it
    measured and what it could not -- rather than one number that folds an
    unreadable tier into the same zero as a genuinely empty one. This is the
    same discipline
    :class:`polylogue.analysis.measurement.outcome_coverage.ToolOutcomeAggregate`
    applies to tool outcomes, where ``unknown_n`` and ``no_result_n`` stay
    separate uncounted buckets and are never folded into the numerator.

    ``unmeasured_reasons`` names every probe that was attempted and did not
    produce a measurement. "Nothing in scope" is not an entry here: a Hermes
    root with no files and an archive with no Hermes hook events were both
    fully measured and found empty. Only a failure to look is recorded.
    """

    sources_projected: int = 0
    sources_unprojected: int = 0
    lifecycle_sessions_sampled: int = 0
    lifecycle_sessions_unsampled: int = 0
    delivery_sessions_sampled: int = 0
    delivery_sessions_unsampled: int = 0
    unmeasured_reasons: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every probe the rollup attempted returned a measurement."""

        return not self.unmeasured_reasons

    def merge(self, other: HermesMeasurementCoverage) -> HermesMeasurementCoverage:
        """Combine two probes' coverage, keeping distinct reasons in order."""

        return HermesMeasurementCoverage(
            sources_projected=self.sources_projected + other.sources_projected,
            sources_unprojected=self.sources_unprojected + other.sources_unprojected,
            lifecycle_sessions_sampled=self.lifecycle_sessions_sampled + other.lifecycle_sessions_sampled,
            lifecycle_sessions_unsampled=self.lifecycle_sessions_unsampled + other.lifecycle_sessions_unsampled,
            delivery_sessions_sampled=self.delivery_sessions_sampled + other.delivery_sessions_sampled,
            delivery_sessions_unsampled=self.delivery_sessions_unsampled + other.delivery_sessions_unsampled,
            unmeasured_reasons=tuple(dict.fromkeys((*self.unmeasured_reasons, *other.unmeasured_reasons))),
        )


@dataclass(frozen=True, slots=True)
class HermesIntegrationHealth:
    """One bounded, composed Hermes integration health rollup."""

    checked_at: str
    enabled: bool
    enabled_reason: str
    verdict: HermesHealthVerdict
    sources: tuple[HermesSourceStatus, ...] = ()
    parser_failures: tuple[HermesParserFailure, ...] = ()
    fidelity_capabilities: tuple[HermesFidelityCapabilityStatus, ...] = ()
    convergence_debt_failed_count: int = 0
    convergence_debt_deferred_count: int = 0
    convergence_debt_retry_due_count: int = 0
    lifecycle_debt: HermesLifecycleDebtSummary = field(
        default_factory=lambda: HermesLifecycleDebtSummary(0, 0, 0, 0, ())
    )
    delivery_correlation: HermesDeliveryCorrelationSummary = field(
        default_factory=lambda: HermesDeliveryCorrelationSummary(0, 0, 0, 0, ())
    )
    measurement_coverage: HermesMeasurementCoverage = field(default_factory=lambda: HermesMeasurementCoverage())
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", _jsonable(self))


def _jsonable(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {f: _jsonable(getattr(value, f)) for f in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _classify_source_ref(
    source_ref: str,
) -> Literal["state_db", "verification_evidence_db", "atof_stream", "atif_document", "other"]:
    lowered = source_ref.lower()
    if lowered == "state.db" or lowered.endswith("_state.db"):
        return "state_db"
    if "verification_evidence" in lowered:
        return "verification_evidence_db"
    if lowered.endswith((".jsonl", ".ndjson")):
        return "atof_stream"
    if lowered.endswith(".json"):
        return "atif_document"
    return "other"


def _recent_hermes_session_native_ids(source_db: Path, *, limit: int) -> tuple[tuple[str, ...], str | None]:
    """Return up to ``limit`` recently-observed Hermes session ids from the durable spool.

    Never raises. The second element is ``None`` when the spool was read and
    ``str`` naming why it was not: an unreadable durable tier and a spool that
    genuinely holds no Hermes events both yield no ids, and collapsing them
    into one bare empty tuple is what let an unreadable ``source.db`` reach a
    "healthy" verdict. The caller keeps them apart.
    """
    if not source_db.exists():
        return (), "source tier unavailable; recent Hermes session sample not drawn"
    try:
        conn = open_readonly_connection(source_db)
        try:
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'raw_hook_events'"
            ).fetchone()
            if has_table is None:
                return (), "source tier carries no raw_hook_events spool; recent Hermes session sample not drawn"
            rows = conn.execute(
                """
                SELECT DISTINCT session_native_id
                FROM raw_hook_events
                WHERE origin = ? AND session_native_id IS NOT NULL
                ORDER BY observed_at_ms DESC
                LIMIT ?
                """,
                (Origin.HERMES_SESSION.value, limit),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("hermes session-id sample query failed for %s: %s", source_db, exc, exc_info=True)
        return (), f"source tier read failed: {type(exc).__name__}; recent Hermes session sample not drawn"
    return tuple(str(row[0]) for row in rows), None


def build_hermes_integration_health(
    archive_root: Path,
    *,
    hermes_root: Path,
    now: datetime | None = None,
    source_limit: int = _DEFAULT_SOURCE_LIMIT,
    session_limit: int = _DEFAULT_SESSION_LIMIT,
    convergence_debt_failed_count: int = 0,
    convergence_debt_deferred_count: int = 0,
    convergence_debt_retry_due_count: int = 0,
) -> HermesIntegrationHealth:
    """Project one bounded Hermes integration health rollup.

    ``hermes_root`` is the configured/discovered Hermes runtime root
    (typically ``~/.hermes`` or a configured override) — the caller resolves
    it, this function only reads it. ``archive_root`` is read-only. No
    directory content is retained past this call; the response carries
    filenames, not absolute paths, and evidence refs/ids, never rendered
    bytes or transcript text.

    ``convergence_debt_failed_count``/``convergence_debt_deferred_count``/
    ``convergence_debt_retry_due_count``
    are pre-computed by the caller (bucketed to the Hermes source family via
    :func:`polylogue.daemon.convergence_debt_status.convergence_debt_summary_info`)
    because this module may not import ``polylogue.daemon`` directly.
    """
    observed = now or datetime.now(UTC)
    checked_at = observed.isoformat()

    if not hermes_root.exists():
        return HermesIntegrationHealth(
            checked_at=checked_at,
            enabled=False,
            enabled_reason="hermes runtime root is not present on this host",
            verdict="disabled",
        )

    if not archive_root.exists():
        return HermesIntegrationHealth(
            checked_at=checked_at,
            enabled=True,
            enabled_reason="hermes runtime root is present",
            verdict="unavailable",
            measurement_coverage=HermesMeasurementCoverage(
                unmeasured_reasons=(
                    "archive root does not exist; freshness, debt, and delivery evidence are unavailable.",
                )
            ),
            caveats=("archive root does not exist; freshness, debt, and delivery evidence are unavailable.",),
        )

    caveats: list[str] = []
    unmeasured_reasons: list[str] = []
    sources_projected = 0
    sources_unprojected = 0

    explain = explain_import_path(hermes_root, source_name="hermes", limit=source_limit)
    sources: list[HermesSourceStatus] = []
    capability_totals: dict[str, HermesFidelityCapabilityStatus] = {}
    for entry in explain.entries:
        source_ref = Path(entry.source_path).name if entry.source_path else "unknown"
        source_class = _classify_source_ref(source_ref)
        session_ref = entry.produced.session_refs[0] if entry.produced.session_refs else None
        stage = "unknown"
        operational_state = "unknown"
        operational_reason = "unknown"
        parse_state = "unseen" if not entry.produced.sessions else "parsed"
        byte_lag_bytes: int | None = None
        fts_converged = False
        insights_converged = False
        projection_error_count = 0
        if entry.source_path:
            try:
                freshness = project_named_source_freshness(archive_root, Path(entry.source_path))
                stage = freshness.stage.value
                operational_state = freshness.operational_state.value
                operational_reason = freshness.operational_reason.value
                parse_state = freshness.parse.state
                byte_lag_bytes = freshness.byte_lag.value
                fts_converged = freshness.fts.converged
                insights_converged = freshness.insights.converged
                projection_error_count = len(freshness.errors)
                sources_projected += 1
            except Exception as exc:  # defensive: freshness projection must never crash the rollup
                logger.warning(
                    "hermes health: named-source freshness projection failed for %s: %s",
                    source_ref,
                    exc,
                    exc_info=True,
                )
                # This source's stage/operational_state stay at their
                # pre-probe "unknown" placeholders. "unknown" is not
                # "degraded", so the verdict's equality tests below cannot
                # see this failure; the coverage record is how it is seen.
                sources_unprojected += 1
                reason = f"freshness projection failed for {source_ref}: {type(exc).__name__}"
                caveats.append(reason)
                unmeasured_reasons.append(reason)
        sources.append(
            HermesSourceStatus(
                source_ref=source_ref,
                source_class=source_class,
                stage=stage,
                operational_state=operational_state,
                operational_reason=operational_reason,
                parse_state=parse_state,
                byte_lag_bytes=byte_lag_bytes,
                fts_converged=fts_converged,
                insights_converged=insights_converged,
                projection_error_count=projection_error_count,
                session_ref=session_ref,
            )
        )
        if entry.fidelity is not None:
            for name, capability in entry.fidelity.capabilities.items():
                existing = capability_totals.get(name)
                refs = (source_ref,) if existing is None else (*existing.source_refs, source_ref)
                capability_totals[name] = HermesFidelityCapabilityStatus(
                    capability=name,
                    status=capability.status,
                    observed=capability.observed + (existing.observed if existing else 0),
                    expected=capability.expected + (existing.expected if existing else 0),
                    detail=capability.detail,
                    source_refs=refs,
                )

    # ``explain_import_path`` appends one root-level skip row (whose
    # ``source_path`` is the scanned root itself, not a discovered file) when
    # the directory contains no candidate files at all -- that is "nothing to
    # report yet", not a per-file parser failure, and must not be counted as
    # integration debt.
    resolved_hermes_root = str(hermes_root.expanduser().resolve())
    parser_failures = tuple(
        HermesParserFailure(
            source_ref=Path(skip.source_path).name if skip.source_path else "unknown",
            reason=skip.reason,
        )
        for skip in explain.skipped
        if skip.source_path != resolved_hermes_root
    )
    for skip in explain.skipped:
        if skip.source_path == resolved_hermes_root:
            caveats.append(skip.reason)
    caveats.extend(explain.caveats)

    session_ids, sample_unmeasured = _recent_hermes_session_native_ids(archive_root / "source.db", limit=session_limit)
    lifecycle_debt = HermesLifecycleDebtSummary(0, 0, 0, 0, ())
    delivery_correlation = HermesDeliveryCorrelationSummary(0, 0, 0, 0, ())
    sample_coverage = HermesMeasurementCoverage()
    if sample_unmeasured is not None:
        caveats.append(sample_unmeasured)
        unmeasured_reasons.append(sample_unmeasured)
    elif session_ids:
        lifecycle_debt, delivery_correlation, sample_coverage = _sample_session_debt(archive_root, session_ids)

    coverage = HermesMeasurementCoverage(
        sources_projected=sources_projected,
        sources_unprojected=sources_unprojected,
        unmeasured_reasons=tuple(unmeasured_reasons),
    ).merge(sample_coverage)

    # Every unmeasured reason is also an operator-visible caveat, so the
    # plaintext surface names the gap without needing its own vocabulary.
    all_caveats = tuple(dict.fromkeys((*caveats, *coverage.unmeasured_reasons)))
    verdict = _aggregate_verdict(
        coverage=coverage,
        sources=sources,
        parser_failures=parser_failures,
        convergence_debt_failed_count=convergence_debt_failed_count,
        lifecycle_debt=lifecycle_debt,
        delivery_correlation=delivery_correlation,
    )

    return HermesIntegrationHealth(
        checked_at=checked_at,
        enabled=True,
        enabled_reason="hermes runtime root is present",
        verdict=verdict,
        sources=tuple(sources),
        parser_failures=parser_failures,
        fidelity_capabilities=tuple(sorted(capability_totals.values(), key=lambda item: item.capability)),
        convergence_debt_failed_count=convergence_debt_failed_count,
        convergence_debt_deferred_count=convergence_debt_deferred_count,
        convergence_debt_retry_due_count=convergence_debt_retry_due_count,
        lifecycle_debt=lifecycle_debt,
        delivery_correlation=delivery_correlation,
        measurement_coverage=coverage,
        caveats=all_caveats,
    )


def _sample_session_debt(
    archive_root: Path,
    session_ids: tuple[str, ...],
) -> tuple[HermesLifecycleDebtSummary, HermesDeliveryCorrelationSummary, HermesMeasurementCoverage]:
    """Reconcile lifecycle debt and delivery correlation for a bounded session sample.

    Returns explicit zero-with-caveat summaries, never raises, when
    ``index.db``/``user.db`` are unavailable — "sampled, zero events" and
    "could not sample" are kept distinguishable via the caveat text and,
    structurally, via the returned :class:`HermesMeasurementCoverage`: a
    caveat string informs a reader, the coverage record is what the verdict
    can act on.
    """
    source_db = archive_root / "source.db"
    index_db = archive_root / "index.db"
    user_db = archive_root / "user.db"

    lifecycle_caveats: list[str] = []
    delivery_caveats: list[str] = []
    total_events = 0
    unpaired = 0
    unknown_ref = 0
    lifecycle_checked = 0
    events_checked = 0
    available = 0
    unavailable = 0
    delivery_checked = 0

    unsampled = len(session_ids)

    if not source_db.exists():
        reason = "source tier unavailable; lifecycle debt and delivery correlation not sampled."
        return (
            HermesLifecycleDebtSummary(0, 0, 0, 0, (reason,)),
            HermesDeliveryCorrelationSummary(0, 0, 0, 0, (reason,)),
            HermesMeasurementCoverage(
                lifecycle_sessions_unsampled=unsampled,
                delivery_sessions_unsampled=unsampled,
                unmeasured_reasons=(reason,),
            ),
        )

    try:
        source_conn = open_readonly_connection(source_db)
    except sqlite3.Error as exc:
        logger.warning("hermes health: source.db open failed: %s", exc, exc_info=True)
        reason = f"source tier read failed: {type(exc).__name__}"
        return (
            HermesLifecycleDebtSummary(0, 0, 0, 0, (reason,)),
            HermesDeliveryCorrelationSummary(0, 0, 0, 0, (reason,)),
            HermesMeasurementCoverage(
                lifecycle_sessions_unsampled=unsampled,
                delivery_sessions_unsampled=unsampled,
                unmeasured_reasons=(reason,),
            ),
        )

    lifecycle_unmeasured: list[str] = []
    delivery_unmeasured: list[str] = []
    lifecycle_unsampled = 0
    delivery_unsampled = 0

    try:
        index_conn: sqlite3.Connection | None = None
        if index_db.exists():
            try:
                index_conn = open_readonly_connection(index_db)
            except sqlite3.Error as exc:
                logger.warning("hermes health: index.db open failed: %s", exc, exc_info=True)
                lifecycle_caveats.append(f"index tier read failed: {type(exc).__name__}")
                lifecycle_unmeasured.append(f"index tier read failed: {type(exc).__name__}; lifecycle debt not sampled")
                lifecycle_unsampled += unsampled
        else:
            # Reconciling against an empty snapshot is not a measurement of
            # zero debt: every event would look unknown, so no session is
            # reconciled at all and the loop below is skipped entirely.
            lifecycle_caveats.append("index tier unavailable; lifecycle debt reconciled against an empty snapshot.")
            lifecycle_unmeasured.append("index tier unavailable; lifecycle debt not sampled")
            lifecycle_unsampled += unsampled

        try:
            if index_conn is not None:
                for session_id in session_ids:
                    try:
                        reconciliation = reconcile_hermes_session_lifecycle(
                            source_conn, index_conn, hermes_session_native_id=session_id
                        )
                    except sqlite3.Error as exc:
                        logger.warning(
                            "hermes health: lifecycle reconciliation failed for %s: %s", session_id, exc, exc_info=True
                        )
                        lifecycle_caveats.append(
                            f"lifecycle reconciliation failed for one sampled session: {type(exc).__name__}"
                        )
                        lifecycle_unmeasured.append(
                            f"lifecycle reconciliation failed for at least one sampled session: {type(exc).__name__}"
                        )
                        lifecycle_unsampled += 1
                        continue
                    lifecycle_checked += 1
                    total_events += reconciliation.total_events
                    unpaired += len(reconciliation.unpaired_event_ids)
                    unknown_ref += len(reconciliation.events_referencing_unknown_messages)
        finally:
            if index_conn is not None:
                index_conn.close()

        user_conn: sqlite3.Connection | None = None
        if user_db.exists():
            try:
                user_conn = open_readonly_connection(user_db)
            except sqlite3.Error as exc:
                logger.warning("hermes health: user.db open failed: %s", exc, exc_info=True)
                delivery_caveats.append(f"user tier read failed: {type(exc).__name__}")
                delivery_unmeasured.append(
                    f"user tier read failed: {type(exc).__name__}; delivery correlation not sampled"
                )
                delivery_unsampled += unsampled
        else:
            delivery_caveats.append("user tier unavailable; delivery correlation not sampled.")
            delivery_unmeasured.append("user tier unavailable; delivery correlation not sampled")
            delivery_unsampled += unsampled

        try:
            if user_conn is not None:
                for session_id in session_ids:
                    try:
                        correlations = correlate_hermes_context_deliveries(
                            source_conn, user_conn, hermes_session_native_id=session_id
                        )
                    except (sqlite3.Error, ValueError) as exc:
                        logger.warning(
                            "hermes health: delivery correlation failed for %s: %s", session_id, exc, exc_info=True
                        )
                        delivery_caveats.append(
                            f"delivery correlation failed for one sampled session: {type(exc).__name__}"
                        )
                        delivery_unmeasured.append(
                            f"delivery correlation failed for at least one sampled session: {type(exc).__name__}"
                        )
                        delivery_unsampled += 1
                        continue
                    delivery_checked += 1
                    for correlation in correlations:
                        events_checked += 1
                        if correlation.available:
                            available += 1
                        else:
                            unavailable += 1
                        delivery_caveats.extend(correlation.caveats)
        finally:
            if user_conn is not None:
                user_conn.close()
    finally:
        source_conn.close()

    # Bound the caveat list: distinct reasons only, ordered stable.
    lifecycle_caveats_dedup = tuple(dict.fromkeys(lifecycle_caveats))
    delivery_caveats_dedup = tuple(dict.fromkeys(delivery_caveats))

    lifecycle_summary = HermesLifecycleDebtSummary(
        sessions_checked=lifecycle_checked,
        total_events=total_events,
        unpaired_event_count=unpaired,
        unknown_message_reference_count=unknown_ref,
        caveats=lifecycle_caveats_dedup,
    )
    delivery_summary = HermesDeliveryCorrelationSummary(
        sessions_checked=delivery_checked,
        events_checked=events_checked,
        available_count=available,
        unavailable_count=unavailable,
        caveats=delivery_caveats_dedup,
    )
    coverage = HermesMeasurementCoverage(
        lifecycle_sessions_sampled=lifecycle_checked,
        lifecycle_sessions_unsampled=lifecycle_unsampled,
        delivery_sessions_sampled=delivery_checked,
        delivery_sessions_unsampled=delivery_unsampled,
        unmeasured_reasons=tuple(dict.fromkeys((*lifecycle_unmeasured, *delivery_unmeasured))),
    )
    return lifecycle_summary, delivery_summary, coverage


def _aggregate_verdict(
    *,
    coverage: HermesMeasurementCoverage,
    sources: list[HermesSourceStatus],
    parser_failures: tuple[HermesParserFailure, ...],
    convergence_debt_failed_count: int,
    lifecycle_debt: HermesLifecycleDebtSummary,
    delivery_correlation: HermesDeliveryCorrelationSummary,
) -> HermesHealthVerdict:
    """Decide one verdict that is total over its own input.

    Every branch below tests a *positive* signal, and a positive signal can
    only come from a probe that ran. ``healthy`` is therefore only reachable
    once ``coverage`` says every attempted probe produced a measurement:
    without that guard an unreadable ``source.db``, an absent ``index.db``
    and a freshness projection that raised all fell through to ``healthy``,
    and a false healthy stops inquiry.

    ``degraded`` is checked before ``unavailable`` on purpose. An observed
    failure is a stronger and more actionable statement than "the probe did
    not run", and the unmeasured portion is not lost either way: it is named
    in ``measurement_coverage.unmeasured_reasons`` and echoed into
    ``caveats``. ``unavailable`` is already in
    :data:`HermesHealthVerdict`; nothing new is added to the vocabulary.
    """

    if parser_failures or convergence_debt_failed_count or lifecycle_debt.unpaired_event_count:
        return "degraded"
    if any(source.operational_state == "degraded" or source.projection_error_count for source in sources):
        return "degraded"
    if delivery_correlation.unavailable_count and not delivery_correlation.available_count:
        return "degraded"
    if not coverage.complete:
        return "unavailable"
    return "healthy"


__all__ = [
    "HermesDeliveryCorrelationSummary",
    "HermesFidelityCapabilityStatus",
    "HermesHealthVerdict",
    "HermesIntegrationHealth",
    "HermesLifecycleDebtSummary",
    "HermesMeasurementCoverage",
    "HermesParserFailure",
    "HermesSourceStatus",
    "build_hermes_integration_health",
]
