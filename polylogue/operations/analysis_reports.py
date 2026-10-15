"""Analysis reports over the one archive reader that selected their scope."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from polylogue.analysis.cost_enrichment import enrich_session_cost_insight
from polylogue.analysis.pathology import compile_pathology_report
from polylogue.analysis.portfolio import compile_portfolio_bundle, portfolio_outcome
from polylogue.analysis.postmortem import PostmortemScope, compile_postmortem_bundle, postmortem_outcome
from polylogue.analysis.transforms import compile_session_digest
from polylogue.archive.hydration import archive_envelope_to_session
from polylogue.archive.query.spec import parse_query_date
from polylogue.cost.aggregation import session_costs_to_daily_usd
from polylogue.cost.outlook import ProjectionMethod, build_cycle_outlook
from polylogue.cost.plans import cycle_for, resolve_plan
from polylogue.operations.operation_context import open_operation_read
from polylogue.storage.derived.session.profiles import hydrate_session_profile
from polylogue.storage.hydrators import session_event_from_record
from polylogue.storage.sqlite.queries.session_events import read_session_events
from polylogue.storage.sqlite.queries.session_links import read_session_digest_links

if TYPE_CHECKING:
    from polylogue.analysis.archive import SessionCostInsight
    from polylogue.analysis.pathology import PathologyReport
    from polylogue.analysis.portfolio import PortfolioBundle
    from polylogue.analysis.postmortem import PostmortemBundle
    from polylogue.analysis.transforms import SessionDigest
    from polylogue.archive.query.execution_control import QueryExecutionContext
    from polylogue.archive.query.spec import SessionQuerySpec
    from polylogue.cost.outlook import CycleOutlook
    from polylogue.cost.plans import SubscriptionPlan
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


@contextmanager
def open_analysis_read(root: Path, context: QueryExecutionContext) -> Iterator[ArchiveStore]:
    with open_operation_read(root, execution_context=context) as pinned:
        yield pinned.archive


def read_session_digest(archive: ArchiveStore, session_id: str) -> SessionDigest | None:
    """Compile a session, its events and child links from the same pinned frame."""
    try:
        session_id = archive.resolve_session_id(session_id)
    except KeyError:
        return None
    session = archive_envelope_to_session(archive.read_session(session_id))
    connection = archive.index_connection
    assert connection is not None
    events = tuple(session_event_from_record(row) for row in read_session_events(connection, session_id))
    session = session.model_copy(update={"session_events": events})
    return compile_session_digest(session, session_links=read_session_digest_links(connection, session_id))


def compile_analysis_report(
    archive: ArchiveStore,
    spec: SessionQuerySpec | None,
    *,
    kind: Literal["postmortem", "portfolio", "pathology"],
    limit: int | None = None,
    top_n: int = 10,
    checkpoint: Callable[[], None],
) -> PostmortemBundle | PortfolioBundle | PathologyReport:
    """Preserve the declared analysis window and count the whole matched relation."""
    from polylogue.api.archive import _archive_text_query, _iter_archive_session_identity_candidates

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    cap = 200 if limit is None else limit
    scope_spec = replace(spec, limit=None, offset=0) if spec is not None else None
    selected_ids: list[str] = []
    matched = 0
    checkpoint()
    with closing(_iter_archive_session_identity_candidates(archive, scope_spec)) as selected:
        for row in selected:
            checkpoint()
            matched += 1
            if len(selected_ids) < cap:
                selected_ids.append(row.session_id)
    profiles = []
    digests: dict[str, SessionDigest] = {}
    for session_id in selected_ids:
        checkpoint()
        record = archive.get_session_profile_record(session_id)
        if record is not None:
            profiles.append(hydrate_session_profile(record))
        digest = read_session_digest(archive, session_id)
        if digest is not None:
            digests[session_id] = digest
        checkpoint()
    scope = PostmortemScope(
        since=spec.since if spec is not None else None,
        until=spec.until if spec is not None else None,
        query=_archive_text_query(spec) if spec is not None else None,
        matched_session_count=matched,
        analyzed_session_count=len(profiles),
        truncated=matched > cap,
        dropped_session_count=max(0, matched - cap),
    )
    result: PostmortemBundle | PortfolioBundle | PathologyReport
    if kind == "postmortem":
        result = compile_postmortem_bundle(profiles, digests, scope=scope)
    elif kind == "portfolio":
        result = compile_portfolio_bundle(profiles, digests, scope=scope, top_n=top_n)
    else:
        result = compile_pathology_report([digest.run_projection for digest in digests.values()]).model_copy(
            update={
                "matched_session_count": matched,
                "analyzed_session_count": len(digests),
                "truncated": matched > cap,
                "dropped_session_count": max(0, matched - cap),
                "failed_session_count": len(selected_ids) - len(digests),
            }
        )
    checkpoint()
    return result


def cost_outlook_inputs(plan_name: str, now: datetime | None = None) -> tuple[SubscriptionPlan, datetime] | None:
    """Resolve the plan and reference time before a caller admits archive work."""
    from polylogue.config import load_polylogue_config

    plan = resolve_plan(plan_name, user_rows=load_polylogue_config().subscription_plans)
    if plan is None:
        return None
    when = (now or datetime.now(UTC)).astimezone(UTC)
    return None if cycle_for(plan, when) is None else (plan, when)


def compile_cost_outlook(
    archive: ArchiveStore,
    plan: SubscriptionPlan,
    *,
    now: datetime,
    method: ProjectionMethod = ProjectionMethod.linear,
    checkpoint: Callable[[], None],
) -> CycleOutlook | None:
    checkpoint()
    when = now.astimezone(UTC)
    cycle = cycle_for(plan, when)
    if cycle is None:
        return None
    cycle_start = parse_query_date("since", cycle[0])
    assert cycle_start is not None

    def costs() -> Iterator[SessionCostInsight]:
        with closing(
            archive.iter_session_cost_insights(
                since_ms=int(cycle_start.timestamp() * 1000),
                until_ms=int(when.timestamp() * 1000),
                time_basis="created",
                limit=None,
            )
        ) as rows:
            for row in rows:
                checkpoint()
                yield enrich_session_cost_insight(archive, row)

    daily = session_costs_to_daily_usd(costs(), as_of=when)
    result = build_cycle_outlook(plan, daily, now=when, method=method)
    checkpoint()
    return result


def execute_analysis_read(
    name: str, payload: dict[str, object], *, archive: ArchiveStore, checkpoint: Callable[[], None]
) -> dict[str, object]:
    from polylogue.operations.daemon_protocol import AnalysisReportRequest, CostOutlookRequest
    from polylogue.operations.query_lowering import cli_query_spec

    if name == "analysis.cost-outlook":
        cost_request = CostOutlookRequest.model_validate(payload)
        if cli_query_spec(cost_request.params).has_filters():
            raise ValueError("cost outlook is global to its billing plan and cannot accept session filters")
        inputs = cost_outlook_inputs(cost_request.plan_name)
        if inputs is None:
            return {"outlook": None}
        plan, when = inputs
        outlook = compile_cost_outlook(
            archive, plan, now=when, method=ProjectionMethod(cost_request.method), checkpoint=checkpoint
        )
        return {"outlook": None if outlook is None else outlook.model_dump(mode="json")}
    request = AnalysisReportRequest.model_validate(payload)
    kind: Literal["postmortem", "portfolio"] = "postmortem" if name == "analysis.postmortem" else "portfolio"
    report = compile_analysis_report(
        archive,
        cli_query_spec(request.params),
        kind=kind,
        limit=request.limit,
        top_n=request.top_n,
        checkpoint=checkpoint,
    )
    outcome = (
        postmortem_outcome(cast("PostmortemBundle", report))
        if kind == "postmortem"
        else portfolio_outcome(cast("PortfolioBundle", report))
    )
    return {"view": kind, "payload": report.model_dump(mode="json"), "outcome": outcome.to_dict()}
