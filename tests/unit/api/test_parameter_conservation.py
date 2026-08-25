"""Generative parameter-conservation checks for the Python API facades.

The request models are the population under test.  Each parametrized case
builds one request with every model field populated, then asks whether the
field's distinguishable value reaches the lower reader (or the transaction's
pagination boundary).  A facade may account for an intentionally local
effect, but it must say so in the typed registry below; silence is not a
disposition.

This is deliberately a route test rather than a database test.  The injected
reader is the effect boundary, so a real archive is unnecessary and a missing
forward cannot be masked by an empty result set.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from polylogue import Polylogue
from polylogue.api import archive as archive_api
from polylogue.api import insights as insights_api
from polylogue.core.enums import Origin
from polylogue.insights.archive import (
    ArchiveCoverageInsightQuery,
    ArchiveDebtInsightQuery,
    CostRollupInsightQuery,
    SessionCostInsightQuery,
    SessionLatencyProfileInsightQuery,
    SessionPhaseInsightQuery,
    SessionProfileInsightQuery,
    SessionTagRollupQuery,
    SessionWorkEventInsightQuery,
    ThreadInsightQuery,
    UsageTimelineInsightQuery,
)
from polylogue.insights.audit import InsightRigorAuditQuery
from polylogue.insights.readiness import InsightReadinessQuery
from polylogue.insights.tool_usage import ToolUsageInsightQuery


@dataclass(frozen=True)
class FacadeConservationSpec:
    """One public facade/request-model pair in the conservation population."""

    name: str
    method_name: str
    request_model: type[BaseModel]
    module: object
    # These are effects owned by the facade operation rather than by the
    # lower reader: cost model/status are applied after enrichment, and the
    # stuck route selects a reader whose operation itself means only_stuck.
    facade_effect_fields: frozenset[str] = field(default_factory=frozenset)
    # A future facade may explicitly account for a field it intentionally
    # drops.  Keep this typed and validated by the test; no current V1 field
    # is silently dropped or declared dropped.
    dropped_fields: frozenset[str] = field(default_factory=frozenset)


def _insight_spec(
    method_name: str,
    request_model: type[BaseModel],
    *,
    facade_effect_fields: frozenset[str] = frozenset(),
) -> FacadeConservationSpec:
    return FacadeConservationSpec(
        name=f"insights.{method_name}",
        method_name=method_name,
        request_model=request_model,
        module=insights_api,
        facade_effect_fields=facade_effect_fields,
    )


def _archive_spec(method_name: str, request_model: type[BaseModel]) -> FacadeConservationSpec:
    return FacadeConservationSpec(
        name=f"archive.{method_name}",
        method_name=method_name,
        request_model=request_model,
        module=archive_api,
    )


# The registry names routes, not fields.  Fields are enumerated mechanically
# from each Pydantic model below, so adding a request field creates a test
# case without a second hand-maintained list.
FACADE_CONSERVATION_REGISTRY: tuple[FacadeConservationSpec, ...] = (
    _insight_spec("list_session_tag_rollup_insights", SessionTagRollupQuery),
    _insight_spec(
        "list_session_work_event_insights",
        SessionWorkEventInsightQuery,
        facade_effect_fields=frozenset({"session_date_since", "session_date_until"}),
    ),
    _insight_spec(
        "list_session_phase_insights",
        SessionPhaseInsightQuery,
        facade_effect_fields=frozenset({"session_date_since", "session_date_until"}),
    ),
    _insight_spec("list_thread_insights", ThreadInsightQuery),
    _insight_spec("list_archive_coverage_insights", ArchiveCoverageInsightQuery),
    _insight_spec("list_tool_usage_insights", ToolUsageInsightQuery),
    _insight_spec(
        "list_session_cost_insights",
        SessionCostInsightQuery,
        facade_effect_fields=frozenset({"model", "status"}),
    ),
    _insight_spec("list_session_latency_profile_insights", SessionLatencyProfileInsightQuery),
    _insight_spec(
        "find_stuck_session_latency_profile_insights",
        SessionLatencyProfileInsightQuery,
        facade_effect_fields=frozenset({"only_stuck"}),
    ),
    _insight_spec("list_cost_rollup_insights", CostRollupInsightQuery),
    _insight_spec("list_usage_timeline_insights", UsageTimelineInsightQuery),
    _insight_spec("list_archive_debt_insights", ArchiveDebtInsightQuery),
    # Archive-side routes prove the same harness works across the composed
    # facade's other mixin, including two request models outside insights.py.
    _archive_spec("list_session_profile_insights", SessionProfileInsightQuery),
    _archive_spec("insight_readiness_report", InsightReadinessQuery),
    _archive_spec("insight_rigor_audit", InsightRigorAuditQuery),
)


_KNOWN_XFAILS: dict[tuple[str, str], str] = {
    (
        "insights.list_session_work_event_insights",
        "query",
    ): "polylogue-3nah4: work-event query is accepted but not forwarded",
    (
        "archive.list_session_profile_insights",
        "query",
    ): "polylogue-3nah4: session-profile query is accepted but not forwarded",
    (
        "insights.find_stuck_session_latency_profile_insights",
        "session_id",
    ): "polylogue-o90gu: stuck-latency session_id is accepted but not forwarded",
    (
        "insights.find_stuck_session_latency_profile_insights",
        "offset",
    ): "polylogue-o90gu: stuck-latency offset is accepted but not forwarded",
}


def _sentinel_for(field_name: str, annotation: object) -> object:
    """Return a valid, distinguishable value for one request-model field."""

    if field_name == "origin":
        return Origin.CODEX_SESSION.value
    if field_name in {"since", "first_message_since", "session_date_since"}:
        return "2026-08-10"
    if field_name in {"until", "first_message_until", "session_date_until"}:
        return "2026-08-11"
    if field_name == "group_by":
        return "month-origin-model"
    if field_name == "tier":
        return "merged"
    if field_name == "sort":
        return "source"
    if field_name == "insights":
        return ("sentinel-insight",)
    if field_name == "offset":
        return 17
    if field_name == "limit":
        return 7
    if field_name in {"only_stuck", "only_actionable", "reverse", "latest", "typed_only"}:
        return True
    integer_sentinels = {
        "min_wallclock_seconds": 1901,
        "max_wallclock_seconds": 1902,
        "sample_limit": 1903,
        "since_ms": 1904,
    }
    if field_name in integer_sentinels:
        return integer_sentinels[field_name]
    if annotation is bool:
        return True
    if annotation is int:
        return 19
    return f"sentinel-{field_name}"


def _request_for(spec: FacadeConservationSpec) -> BaseModel:
    values = {
        field_name: _sentinel_for(field_name, model_field.annotation)
        for field_name, model_field in spec.request_model.model_fields.items()
    }
    request = spec.request_model.model_validate(values)
    assert set(request.model_fields_set) == set(spec.request_model.model_fields)
    return request


def _date_sentinel_ms(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError:
        return None
    return int(parsed.timestamp() * 1000)


def _matches(value: object, expected: object) -> bool:
    if value == expected:
        return True
    if isinstance(value, BaseModel):
        return any(_matches(field_value, expected) for field_value in value.__dict__.values())
    if isinstance(value, Mapping):
        return any(_matches(item, expected) for item in value.values())
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_matches(item, expected) for item in value)
    return False


def _field_aliases(field_name: str) -> frozenset[str]:
    aliases = {field_name}
    alias = {
        "since": "since_ms",
        "until": "until_ms",
        "limit": "page_size",
    }.get(field_name)
    if alias is not None:
        aliases.add(alias)
    return frozenset(aliases)


def _field_reaches(value: object, field_name: str, expected: tuple[object, ...]) -> bool:
    """Find one field by name and then compare its distinguishable value."""

    aliases = _field_aliases(field_name)
    if isinstance(value, BaseModel):
        if field_name in type(value).model_fields and any(
            _matches(getattr(value, field_name), candidate) for candidate in expected
        ):
            return True
        return any(_field_reaches(field_value, field_name, expected) for field_value in value.__dict__.values())
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in aliases and any(_matches(item, candidate) for candidate in expected):
                return True
            if _field_reaches(item, field_name, expected):
                return True
        return False
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_field_reaches(item, field_name, expected) for item in value)
    return False


def _effect_values(request: BaseModel, field_name: str) -> tuple[object, ...]:
    value = getattr(request, field_name)
    values = [value]
    date_ms = _date_sentinel_ms(value)
    if date_ms is not None:
        values.append(date_ms)
    return tuple(values)


@dataclass
class _ReaderSpy:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = field(default_factory=list)

    def __getattr__(self, method_name: str) -> Callable[..., list[object]]:
        def read(*args: object, **kwargs: object) -> list[object]:
            self.calls.append((method_name, args, dict(kwargs)))
            return []

        return read


@dataclass
class _RunSpy:
    reader: _ReaderSpy
    calls: list[dict[str, object]] = field(default_factory=list)

    async def run(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        work = cast(Callable[[object], object], kwargs["work"])
        result = work(self.reader)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result


async def _run_archive_read_spy(
    archive_root: Path,
    *,
    operation: str,
    arguments: Mapping[str, object],
    work: Callable[[object], object],
    **kwargs: object,
) -> object:
    del archive_root
    # The active spy is installed by the test through this module-level holder
    # because both facade mixins import the same function by name.
    assert _ACTIVE_RUNNER is not None
    spy = _ACTIVE_RUNNER
    return await spy.run(operation=operation, arguments=dict(arguments), work=work, **kwargs)


_ACTIVE_RUNNER: _RunSpy | None = None


@pytest.fixture
async def facade(tmp_path: Path) -> AsyncIterator[Polylogue]:
    instance = Polylogue(archive_root=tmp_path, db_path=tmp_path / "index.db")
    try:
        yield instance
    finally:
        await instance.close()


def _cases() -> tuple[object, ...]:
    cases: list[object] = []
    for spec in FACADE_CONSERVATION_REGISTRY:
        for field_name in spec.request_model.model_fields:
            marker = _KNOWN_XFAILS.get((spec.name, field_name))
            parameter = pytest.param(
                spec,
                field_name,
                id=f"{spec.name}:{field_name}",
                marks=pytest.mark.xfail(strict=True, reason=marker) if marker else (),
            )
            cases.append(parameter)
    return tuple(cases)


@pytest.mark.parametrize("spec,field_name", _cases())
async def test_facade_conserves_every_request_field(
    spec: FacadeConservationSpec,
    field_name: str,
    facade: Polylogue,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every generated field has a typed effect disposition."""

    model_fields = set(spec.request_model.model_fields)
    assert spec.facade_effect_fields <= model_fields
    assert spec.dropped_fields <= model_fields
    assert not spec.facade_effect_fields & spec.dropped_fields

    request = _request_for(spec)
    reader = _ReaderSpy()
    runner = _RunSpy(reader)
    global _ACTIVE_RUNNER
    _ACTIVE_RUNNER = runner
    monkeypatch.setattr(spec.module, "run_archive_read", _run_archive_read_spy)
    monkeypatch.setattr(insights_api, "synthesize_origin_tag_rollups", lambda *args, **kwargs: [])
    monkeypatch.setattr(insights_api, "enrich_session_cost_insights", lambda archive, rows: rows)

    await cast(Callable[[BaseModel], Awaitable[object]], getattr(facade, spec.method_name))(request)

    if field_name in spec.dropped_fields or field_name in spec.facade_effect_fields:
        return

    expected_values = _effect_values(request, field_name)
    observed = any(_field_reaches(call, field_name, expected_values) for call in runner.calls)
    observed = observed or any(
        _field_reaches(value, field_name, expected_values)
        for _method_name, args, kwargs in reader.calls
        for value in (*args, kwargs)
    )
    assert observed, (
        f"{spec.name} accepted {field_name}={getattr(request, field_name)!r} but the value "
        "did not reach the transaction or lower-reader effect boundary; add the forward "
        "or an explicit typed disposition."
    )


def test_registry_enumerates_pydantic_population_without_manual_fields() -> None:
    """The registry is extensible by facade/model; fields come from Pydantic."""

    assert FACADE_CONSERVATION_REGISTRY
    assert all(spec.request_model.model_fields for spec in FACADE_CONSERVATION_REGISTRY)
    assert len({spec.name for spec in FACADE_CONSERVATION_REGISTRY}) == len(FACADE_CONSERVATION_REGISTRY)
