"""Executes the declared query laws across CLI, Python, HTTP and MCP.

One request is normalized into a :class:`NormalizedPage` per surface so laws
compare *semantics* -- selected identities, order, paging facts, field
vocabulary, refusal class -- and never a serialization format. The four
surfaces reach the archive through their own production adapters
(:meth:`polylogue.api.Polylogue.query_units`, the root ``find`` CLI verb,
``GET /api/query-units``, and the MCP ``query`` tool), so a surface that
grows a private filter, sort or continuation is what makes a law red.

The laws themselves are declared in :mod:`tests.infra.query_contract`; the
mutants at the bottom of this module are keyed on the ``anti_vacuity``
sentence each law states, so a law whose stated mutation does not actually
turn it red is visible rather than assumed.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from polylogue.api import Polylogue
from polylogue.archive.query.metadata import QueryUnitName
from tests.infra.query_contract import (
    REF_FAMILIES_BY_UNIT,
    SURFACE_NAMES,
    UNCOMPILABLE_EXPRESSION,
    UNIT_IDENTITY_BY_UNIT,
    UNIT_PROBE_BY_UNIT,
    UNIT_PROBES,
    RefFamily,
    SurfaceName,
    UnitProbe,
    exemption_reason,
    law_applies,
)

RowMapping = Mapping[str, Any]

#: Every surface refuses an uncompilable expression with this class. The laws
#: compare the class, never the human-readable detail, which is free to differ.
REFUSAL_INVALID_QUERY = "invalid_query"

_HTTP_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class NormalizedPage:
    """One surface's answer to one request, in surface-independent terms."""

    surface: SurfaceName
    unit: QueryUnitName
    mode: str
    rows: tuple[RowMapping, ...]
    total: int
    limit: int
    offset: int
    next_offset: int | None
    continuation: str | None
    query_ref: str | None
    result_ref: str | None
    response_bytes: int

    @property
    def identities(self) -> tuple[tuple[str, ...], ...]:
        """The declared stable identity of each row, in emitted order."""

        fields = UNIT_IDENTITY_BY_UNIT[self.unit].fields
        return tuple(tuple(str(row.get(name)) for name in fields) for row in self.rows)

    @property
    def field_vocabulary(self) -> frozenset[str]:
        """The union of field names the surface emitted for this page."""

        return frozenset(name for row in self.rows for name in row)


@dataclass(frozen=True, slots=True)
class Refusal:
    """A surface's refusal of an unusable request, reduced to its class."""

    surface: SurfaceName
    refusal_class: str
    detail: str


class SurfaceUnavailableError(RuntimeError):
    """Raised when a surface cannot express the requested shape at all."""


class QuerySurface:
    """One production read surface, driven through its own adapter."""

    name: SurfaceName

    async def page(
        self,
        expression: str,
        *,
        unit: QueryUnitName,
        limit: int | None = None,
        offset: int | None = None,
        continuation: str | None = None,
    ) -> NormalizedPage:
        raise NotImplementedError

    async def refusal(self, expression: str) -> Refusal:
        raise NotImplementedError


def _normalize(
    surface: SurfaceName,
    unit: QueryUnitName,
    payload: RowMapping,
    *,
    response_bytes: int,
) -> NormalizedPage:
    rows = tuple(cast(RowMapping, row) for row in cast(Sequence[Any], payload.get("items", ())))
    next_offset = payload.get("next_offset")
    return NormalizedPage(
        surface=surface,
        unit=unit,
        mode=str(payload.get("mode", "query-unit")),
        rows=rows,
        total=int(cast(int, payload.get("total", len(rows)))),
        limit=int(cast(int, payload.get("limit", len(rows)))),
        offset=int(cast(int, payload.get("offset", 0))),
        next_offset=None if next_offset is None else int(cast(int, next_offset)),
        continuation=cast(str | None, payload.get("continuation")),
        query_ref=cast(str | None, payload.get("query_ref")),
        result_ref=cast(str | None, payload.get("result_ref")),
        response_bytes=response_bytes,
    )


class PythonSurface(QuerySurface):
    """``Polylogue.query_units`` -- the facade every other surface delegates to."""

    name: SurfaceName = "python"

    def __init__(self, archive: Polylogue) -> None:
        self._archive = archive

    async def page(
        self,
        expression: str,
        *,
        unit: QueryUnitName,
        limit: int | None = None,
        offset: int | None = None,
        continuation: str | None = None,
    ) -> NormalizedPage:
        if continuation is not None:
            envelope = await self._archive.query_units(continuation=continuation)
        else:
            envelope = await self._archive.query_units(expression, limit=limit, offset=offset)
        payload = envelope.model_dump(mode="json")
        return _normalize(self.name, unit, payload, response_bytes=len(json.dumps(payload).encode("utf-8")))

    async def refusal(self, expression: str) -> Refusal:
        from polylogue.archive.query.expression import ExpressionCompileError

        try:
            await self._archive.query_units(expression, limit=1)
        except ExpressionCompileError as exc:
            return Refusal(self.name, REFUSAL_INVALID_QUERY, str(exc))
        raise AssertionError(f"python surface accepted an unusable expression: {expression!r}")


class HttpSurface(QuerySurface):
    """The daemon's ``GET /api/query-units`` reader route."""

    name: SurfaceName = "http"

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def _fetch(self, query: str) -> tuple[RowMapping, int]:
        request = Request(f"{self._base_url}/api/query-units?{query}")
        with urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            body = cast(bytes, response.read())
        return cast(RowMapping, json.loads(body)), len(body)

    async def page(
        self,
        expression: str,
        *,
        unit: QueryUnitName,
        limit: int | None = None,
        offset: int | None = None,
        continuation: str | None = None,
    ) -> NormalizedPage:
        if continuation is not None:
            query = urlencode({"continuation": continuation})
        else:
            params: dict[str, str] = {"expression": expression}
            if limit is not None:
                params["limit"] = str(limit)
            if offset is not None:
                params["offset"] = str(offset)
            query = urlencode(params)
        payload, size = await asyncio.to_thread(self._fetch, query)
        return _normalize(self.name, unit, payload, response_bytes=size)

    async def refusal(self, expression: str) -> Refusal:
        def _call() -> Refusal:
            try:
                request = Request(f"{self._base_url}/api/query-units?expression={quote(expression)}&limit=1")
                with urlopen(request, timeout=_HTTP_TIMEOUT_S):
                    raise AssertionError(f"http surface accepted an unusable expression: {expression!r}")
            except HTTPError as exc:
                body = cast(RowMapping, json.loads(exc.read()))
                return Refusal(self.name, str(body.get("error", "")), str(body.get("detail", "")))

        return await asyncio.to_thread(_call)


def _unwrap_budget_envelope(payload: RowMapping) -> RowMapping:
    """Reduce an MCP byte-budget envelope to the page it actually delivered.

    The byte budget is a transport bound, not a semantic one: the envelope
    carries the retained prefix under ``page`` and the only advancing cursor
    under ``continuation.arguments.continuation``. Reading the envelope's
    outer shape as the page would report a truthy continuation with zero rows
    and hand a dict back to the server as a cursor.
    """

    if payload.get("status") != "response_budget_exceeded":
        return payload
    page = payload.get("page")
    if not isinstance(page, dict):
        raise SurfaceUnavailableError(
            "the mcp response budget refused the whole page: no row fits the transport budget"
        )
    continuation = payload.get("continuation")
    token: str | None = None
    if isinstance(continuation, dict):
        arguments = continuation.get("arguments")
        if isinstance(arguments, dict):
            token = cast(str | None, arguments.get("continuation"))
    return {**cast(RowMapping, page), "continuation": token}


class McpSurface(QuerySurface):
    """The MCP ``query`` dispatcher tool."""

    name: SurfaceName = "mcp"

    def __init__(self, handler: Callable[..., Any], archive_root: Path) -> None:
        self._handler = handler
        self._archive_root = archive_root

    async def _invoke(self, **kwargs: object) -> tuple[RowMapping, int]:
        from tests.infra.mcp import invoke_surface_async

        with (
            patch(
                "polylogue.mcp.server._get_config",
                return_value=SimpleNamespace(archive_root=self._archive_root),
            ),
            patch(
                "polylogue.mcp.server._get_polylogue",
                return_value=Polylogue(archive_root=self._archive_root),
            ),
        ):
            body = await invoke_surface_async(self._handler, **kwargs)
        return cast(RowMapping, json.loads(body)), len(body.encode("utf-8"))

    async def page(
        self,
        expression: str,
        *,
        unit: QueryUnitName,
        limit: int | None = None,
        offset: int | None = None,
        continuation: str | None = None,
    ) -> NormalizedPage:
        if continuation is not None:
            payload, size = await self._invoke(continuation=continuation)
        else:
            payload, size = await self._invoke(expression=expression, limit=limit, offset=offset)
        if payload.get("ok") is False:
            raise AssertionError(f"mcp surface refused a usable request: {payload}")
        return _normalize(self.name, unit, _unwrap_budget_envelope(payload), response_bytes=size)

    async def refusal(self, expression: str) -> Refusal:
        payload, _ = await self._invoke(expression=expression, limit=1)
        if payload.get("ok") is not False:
            raise AssertionError(f"mcp surface accepted an unusable expression: {expression!r}")
        return Refusal(self.name, str(payload.get("error", "")), str(payload.get("message", "")))


class CliSurface(QuerySurface):
    """The root ``polylogue find`` query verb, rendered as JSON."""

    name: SurfaceName = "cli"

    def __init__(self, archive_root: Path) -> None:
        self._archive_root = archive_root

    def _invoke(self, args: list[str]) -> tuple[int, str, BaseException | None]:
        from click.testing import CliRunner

        from polylogue.cli.click_app import cli as click_cli

        env = {
            "POLYLOGUE_ARCHIVE_ROOT": str(self._archive_root),
            "POLYLOGUE_FORCE_PLAIN": "1",
            "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:1",
        }
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            result = CliRunner().invoke(click_cli, args, catch_exceptions=True)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        return result.exit_code, result.output, result.exception

    async def page(
        self,
        expression: str,
        *,
        unit: QueryUnitName,
        limit: int | None = None,
        offset: int | None = None,
        continuation: str | None = None,
    ) -> NormalizedPage:
        if continuation is not None:
            raise SurfaceUnavailableError("the CLI query route emits no continuation token")
        args: list[str] = ["--format", "json"]
        if limit is not None:
            args += ["--limit", str(limit)]
        if offset is not None:
            args += ["--offset", str(offset)]
        args += ["find", expression]
        exit_code, output, exception = await asyncio.to_thread(self._invoke, args)
        # An empty terminal page is reported through exit status 2 with an
        # empty-items envelope; that is still a well-formed answer.
        if exit_code not in (0, 2):
            raise AssertionError(f"cli surface failed ({exit_code}) for {expression!r}: {exception or output}")
        payload = cast(RowMapping, json.loads(output))
        return _normalize(self.name, unit, payload, response_bytes=len(output.encode("utf-8")))

    async def refusal(self, expression: str) -> Refusal:
        from polylogue.archive.query.expression import ExpressionCompileError

        exit_code, output, exception = await asyncio.to_thread(
            self._invoke, ["--format", "json", "--limit", "1", "find", expression]
        )
        if exit_code == 0:
            raise AssertionError(f"cli surface accepted an unusable expression: {expression!r}")
        if isinstance(exception, ExpressionCompileError):
            return Refusal(self.name, REFUSAL_INVALID_QUERY, str(exception))
        return Refusal(self.name, "unclassified", output)


@dataclass(frozen=True, slots=True)
class SurfaceBench:
    """The four live surfaces plus the archive facade the laws read through."""

    archive_root: Path
    archive: Polylogue
    surfaces: Mapping[SurfaceName, QuerySurface]

    def surface(self, name: SurfaceName) -> QuerySurface:
        return self.surfaces[name]


@contextmanager
def _running_daemon_http() -> Iterator[str]:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

    server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
    server.auth_token = ""
    server.api_host = "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, name="query-differential-http", daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


@asynccontextmanager
async def surface_bench(archive_root: Path) -> AsyncIterator[SurfaceBench]:
    """Start every read surface once against ``archive_root``."""

    from polylogue.mcp.server import build_server

    previous_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    os.environ["POLYLOGUE_ARCHIVE_ROOT"] = str(archive_root)
    archive = Polylogue(archive_root=archive_root)
    mcp_server = build_server()
    handler = mcp_server._tool_manager._tools["query"].fn
    try:
        with _running_daemon_http() as base_url:
            yield SurfaceBench(
                archive_root=archive_root,
                archive=archive,
                surfaces={
                    "python": PythonSurface(archive),
                    "cli": CliSurface(archive_root),
                    "http": HttpSurface(base_url),
                    "mcp": McpSurface(handler, archive_root),
                },
            )
    finally:
        await archive.close()
        if previous_root is None:
            os.environ.pop("POLYLOGUE_ARCHIVE_ROOT", None)
        else:
            os.environ["POLYLOGUE_ARCHIVE_ROOT"] = previous_root


LawStatus = Literal["held", "violated", "exempt"]


@dataclass(frozen=True, slots=True)
class LawOutcome:
    """The result of evaluating one law at one unit/surface coordinate."""

    law_id: str
    unit: QueryUnitName | None
    surface: SurfaceName | None
    status: LawStatus
    detail: str = ""
    #: Populated for ``exempt``: the declared reason the law does not apply.
    exemption: str = ""

    @property
    def coordinate(self) -> str:
        return f"{self.law_id}[unit={self.unit or '*'},surface={self.surface or '*'}]"


@dataclass
class LawRun:
    """Every outcome produced by one pass of the law suite."""

    outcomes: list[LawOutcome] = field(default_factory=list)

    def record(
        self,
        law_id: str,
        *,
        unit: QueryUnitName | None = None,
        surface: SurfaceName | None = None,
        held: bool,
        detail: str = "",
    ) -> None:
        self.outcomes.append(
            LawOutcome(
                law_id=law_id,
                unit=unit,
                surface=surface,
                status="held" if held else "violated",
                detail=detail,
            )
        )

    def exempt(
        self,
        law_id: str,
        *,
        unit: QueryUnitName | None = None,
        surface: SurfaceName | None = None,
        reason: str,
    ) -> None:
        self.outcomes.append(LawOutcome(law_id=law_id, unit=unit, surface=surface, status="exempt", exemption=reason))

    @property
    def violations(self) -> tuple[LawOutcome, ...]:
        return tuple(outcome for outcome in self.outcomes if outcome.status == "violated")

    @property
    def covered_law_ids(self) -> frozenset[str]:
        return frozenset(outcome.law_id for outcome in self.outcomes)

    def covered_units(self, law_id: str) -> frozenset[QueryUnitName]:
        return frozenset(
            outcome.unit for outcome in self.outcomes if outcome.law_id == law_id and outcome.unit is not None
        )


# ---------------------------------------------------------------------------
# Law execution
# ---------------------------------------------------------------------------

#: Page size the paging laws concatenate with, and the whole-population limit.
LAW_PAGE_SIZE = 3
LAW_UNPAGED_LIMIT = 10_000


def _render_predicate(node: RowMapping) -> str:
    """Render one typed predicate AST node back into DSL source text.

    The rendered text is deliberately not the original source: the law it
    serves fails if execution reads the raw expression rather than the
    compiled predicate.
    """

    kind = node.get("kind")
    if kind in {"and", "or"}:
        children = cast(Sequence[RowMapping], node.get("children", ()))
        joiner = " AND " if kind == "and" else " OR "
        return "(" + joiner.join(_render_predicate(child) for child in children) + ")"
    if kind == "not":
        child = cast(RowMapping, node["child"])
        return f"NOT {_render_predicate(child)}"
    if kind != "field":
        raise SurfaceUnavailableError(f"the law harness cannot render predicate node {kind!r}")
    field = str(node["field"])
    op = str(node.get("op", "="))
    values = [str(value) for value in cast(Sequence[object], node.get("values", ()))]
    if not values:
        raise SurfaceUnavailableError(f"predicate field {field!r} carries no value to render")
    prefix = "" if op == "=" else op
    rendered = [f"{field}:{prefix}{value}" for value in values]
    if len(rendered) == 1:
        return rendered[0]
    return "(" + " OR ".join(rendered) + ")"


async def _expression_from_ast(archive: Polylogue, probe: UnitProbe, expression: str) -> str:
    """Recompile ``expression`` from its published typed AST."""

    explained = cast(RowMapping, await archive.explain_query_expression(expression))
    predicate = explained.get("predicate")
    if not isinstance(predicate, dict):
        raise SurfaceUnavailableError(f"{expression!r} published no typed predicate to re-render")
    return f"{probe.source} where {_render_predicate(cast(RowMapping, predicate))}"


async def _identities(
    surface: QuerySurface,
    expression: str,
    *,
    unit: QueryUnitName,
    limit: int = LAW_UNPAGED_LIMIT,
    offset: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    page = await surface.page(expression, unit=unit, limit=limit, offset=offset)
    return page.identities


@dataclass(frozen=True, slots=True)
class ContinuationWalk:
    """What following one surface's continuations to exhaustion produced.

    ``transport_bounded`` marks a walk a surface ended by refusing a row its
    transport cannot carry. That is a stated terminal refusal, not a
    non-progressing continuation: the members collected before it must still
    be an exact duplicate-free prefix of the population, and the surface must
    not have handed back a cursor it cannot honour.
    """

    members: tuple[tuple[str, ...], ...]
    pages: int
    detail: str
    transport_bounded: bool = False


async def _walk_continuation(
    surface: QuerySurface,
    expression: str,
    *,
    unit: QueryUnitName,
    page_size: int,
    max_pages: int = 200,
) -> ContinuationWalk:
    """Follow continuations to exhaustion."""

    try:
        page = await surface.page(expression, unit=unit, limit=page_size)
    except SurfaceUnavailableError as exc:
        return ContinuationWalk((), 0, str(exc), transport_bounded=True)
    members = list(page.identities)
    seen_tokens: set[str] = set()
    pages = 1
    token = page.continuation
    while token is not None:
        if token in seen_tokens:
            return ContinuationWalk(tuple(members), pages, f"continuation token replayed after {pages} pages")
        seen_tokens.add(token)
        if pages >= max_pages:
            return ContinuationWalk(tuple(members), pages, f"continuation did not terminate within {max_pages} pages")
        try:
            page = await surface.page(expression, unit=unit, continuation=token)
        except SurfaceUnavailableError:
            return ContinuationWalk(tuple(members), pages, "", transport_bounded=True)
        pages += 1
        if not page.rows and page.continuation is not None:
            return ContinuationWalk(tuple(members), pages, "continuation advertised more work but emitted no rows")
        members.extend(page.identities)
        token = page.continuation
    return ContinuationWalk(tuple(members), pages, "")


def _duplicate_detail(members: Sequence[tuple[str, ...]]) -> str:
    seen: set[tuple[str, ...]] = set()
    duplicates: list[tuple[str, ...]] = []
    for member in members:
        if member in seen:
            duplicates.append(member)
        seen.add(member)
    return f"duplicated members: {duplicates[:5]}" if duplicates else ""


def _ref_values(row: RowMapping, family: RefFamily) -> tuple[str, ...]:
    raw = row.get(family.field)
    if raw is None:
        return ()
    if family.many:
        return tuple(str(value) for value in cast(Sequence[object], raw))
    return (str(raw),)


def _canonical_ref_text(family: RefFamily, value: str) -> str:
    from polylogue.core.refs import normalize_object_ref_text, normalize_public_ref_text

    if family.shape == "archive-id" and family.target != "none":
        return normalize_object_ref_text(f"{family.target}:{value}")
    return normalize_public_ref_text(value)


def _expected_ref_text(family: RefFamily, value: str) -> str:
    if family.shape == "archive-id" and family.target != "none":
        return f"{family.target}:{value}"
    return value


async def _check_ref_canonicalization(run: LawRun, probe: UnitProbe, page: NormalizedPage) -> None:
    failures: list[str] = []
    for family in REF_FAMILIES_BY_UNIT[probe.unit]:
        for row in page.rows:
            if family.field not in row:
                failures.append(f"{family.unit}.{family.field}: declared ref family absent from the row payload")
                break
            for value in _ref_values(row, family):
                try:
                    canonical = _canonical_ref_text(family, value)
                except ValueError as exc:
                    failures.append(f"{family.unit}.{family.field}={value!r}: {exc}")
                    continue
                expected = _expected_ref_text(family, value)
                if canonical != expected:
                    failures.append(f"{family.unit}.{family.field}={value!r} re-formats as {canonical!r}")
    run.record(
        "ref-canonicalization",
        unit=probe.unit,
        held=not failures,
        detail="; ".join(failures[:5]),
    )


async def _check_ref_detail_closure(run: LawRun, archive: Polylogue, probe: UnitProbe, page: NormalizedPage) -> None:
    closures = tuple(family for family in REF_FAMILIES_BY_UNIT[probe.unit] if family.detail_closure)
    if not closures:
        run.exempt(
            "ref-detail-closure",
            unit=probe.unit,
            reason="no ref family on this unit names an archive record with a detail route",
        )
        return
    failures: list[str] = []
    resolved = 0
    for family in closures:
        for row in page.rows:
            for value in _ref_values(row, family):
                ref = f"{family.target}:{value}"
                resolution = await archive.resolve_ref(ref)
                if not resolution.resolved:
                    failures.append(f"{family.unit}.{family.field}: {ref!r} did not resolve")
                elif resolution.kind != family.target:
                    failures.append(f"{ref!r} resolved as {resolution.kind!r}, not {family.target!r}")
                else:
                    resolved += 1
    if not failures and resolved == 0:
        failures.append("no declared archive ref was emitted, so closure was never exercised")
    run.record("ref-detail-closure", unit=probe.unit, held=not failures, detail="; ".join(failures[:5]))


async def _check_unit_metamorphic_laws(run: LawRun, bench: SurfaceBench, probe: UnitProbe) -> None:
    surface = bench.surface("python")
    unit = probe.unit
    scoped = probe.scoped_expression

    forward = await _identities(surface, probe.conjunction(probe.scope, probe.narrowing), unit=unit)
    reverse = await _identities(surface, probe.conjunction(probe.narrowing, probe.scope), unit=unit)
    whole = await _identities(surface, scoped, unit=unit)
    run.record(
        "predicate-commutativity",
        unit=unit,
        held=forward == reverse and 0 < len(forward) < len(whole),
        detail=(
            f"forward={len(forward)} reverse={len(reverse)} whole={len(whole)}"
            if forward != reverse or not 0 < len(forward) < len(whole)
            else ""
        ),
    )

    doubled = await _identities(surface, probe.conjunction(probe.narrowing, probe.narrowing), unit=unit)
    single = await _identities(surface, f"{probe.source} where {probe.narrowing}", unit=unit)
    run.record(
        "predicate-idempotence",
        unit=unit,
        held=doubled == single and bool(single),
        detail=f"doubled={len(doubled)} single={len(single)}" if doubled != single or not single else "",
    )

    concatenated: list[tuple[str, ...]] = []
    offset = 0
    while offset < len(whole) + LAW_PAGE_SIZE:
        page = await _identities(surface, scoped, unit=unit, limit=LAW_PAGE_SIZE, offset=offset)
        if not page:
            break
        concatenated.extend(page)
        offset += LAW_PAGE_SIZE
    duplicate = _duplicate_detail(concatenated)
    run.record(
        "page-concatenation",
        unit=unit,
        held=tuple(concatenated) == whole and not duplicate,
        detail=duplicate or (f"paged={len(concatenated)} unpaged={len(whole)}" if tuple(concatenated) != whole else ""),
    )

    small = await _identities(surface, scoped, unit=unit, limit=LAW_PAGE_SIZE)
    larger = await _identities(surface, scoped, unit=unit, limit=LAW_PAGE_SIZE * 2)
    run.record(
        "limit-monotonicity",
        unit=unit,
        held=larger[: len(small)] == small and bool(small),
        detail=f"small={len(small)} larger={len(larger)}" if larger[: len(small)] != small or not small else "",
    )

    if law_applies("group-count-population", unit=unit) and probe.group_field is not None:
        grouped = await surface.page(
            f"{scoped} | group by {probe.group_field} | count", unit=unit, limit=LAW_UNPAGED_LIMIT
        )
        counted = await surface.page(f"{scoped} | count", unit=unit, limit=LAW_UNPAGED_LIMIT)
        group_total = sum(int(cast(int, row.get("count", 0))) for row in grouped.rows)
        flat_total = sum(int(cast(int, row.get("count", 0))) for row in counted.rows)
        run.record(
            "group-count-population",
            unit=unit,
            held=group_total == flat_total == len(whole) and bool(whole),
            detail=f"grouped={group_total} counted={flat_total} population={len(whole)}",
        )
    else:
        run.exempt(
            "group-count-population",
            unit=unit,
            reason=exemption_reason("group-count-population", unit=unit)
            or "the unit declares no aggregate group fields",
        )

    recompiled = await _expression_from_ast(bench.archive, probe, probe.conjunction(probe.scope, probe.narrowing))
    from_ast = await _identities(surface, recompiled, unit=unit)
    run.record(
        "structured-plan-equivalence",
        unit=unit,
        held=from_ast == forward and bool(forward),
        detail=f"recompiled={recompiled!r} rows={len(from_ast)} vs {len(forward)}" if from_ast != forward else "",
    )

    whole_page = await surface.page(scoped, unit=unit, limit=LAW_UNPAGED_LIMIT)
    await _check_ref_canonicalization(run, probe, whole_page)
    await _check_ref_detail_closure(run, bench.archive, probe, whole_page)


async def _check_unit_cross_surface_laws(run: LawRun, bench: SurfaceBench, probe: UnitProbe) -> None:
    unit = probe.unit
    scoped = probe.scoped_expression
    reference = await bench.surface("python").page(scoped, unit=unit, limit=LAW_PAGE_SIZE)

    offset_reference = await bench.surface("python").page(scoped, unit=unit, limit=LAW_PAGE_SIZE, offset=LAW_PAGE_SIZE)
    for name in SURFACE_NAMES:
        if name == "python":
            continue
        mismatches: list[str] = []
        first_page = await bench.surface(name).page(scoped, unit=unit, limit=LAW_PAGE_SIZE)
        for label, expected, observed in (
            ("first page", reference, first_page),
            (
                "offset page",
                offset_reference,
                await bench.surface(name).page(scoped, unit=unit, limit=LAW_PAGE_SIZE, offset=LAW_PAGE_SIZE),
            ),
        ):
            if observed.identities != expected.identities:
                mismatches.append(f"{label} selection/order differs: {observed.identities} vs {expected.identities}")
            for attribute in ("total", "limit", "offset", "next_offset", "mode"):
                if getattr(observed, attribute) != getattr(expected, attribute):
                    mismatches.append(
                        f"{label} {attribute}={getattr(observed, attribute)!r} vs {getattr(expected, attribute)!r}"
                    )
        run.record(
            "cross-surface-selection", unit=unit, surface=name, held=not mismatches, detail="; ".join(mismatches)
        )

        vocabulary_diff = sorted(first_page.field_vocabulary ^ reference.field_vocabulary)
        run.record(
            "cross-surface-typing",
            unit=unit,
            surface=name,
            held=not vocabulary_diff,
            detail=f"field vocabulary differs: {vocabulary_diff}",
        )

    population = await _identities(bench.surface("python"), scoped, unit=unit)
    for name in SURFACE_NAMES:
        if not law_applies("continuation-progress", surface=name):
            run.exempt(
                "continuation-progress",
                unit=unit,
                surface=name,
                reason=exemption_reason("continuation-progress", surface=name),
            )
            continue
        walk = await _walk_continuation(bench.surface(name), scoped, unit=unit, page_size=LAW_PAGE_SIZE)
        duplicate = _duplicate_detail(walk.members)
        if walk.transport_bounded:
            complete = walk.members == population[: len(walk.members)]
            note = f"transport-bounded after {len(walk.members)} of {len(population)} members: {walk.detail}"
        else:
            complete = walk.members == population
            note = f"walked={len(walk.members)} in {walk.pages} pages, population={len(population)}"
        held = (not walk.detail or walk.transport_bounded) and not duplicate and complete
        run.record(
            "continuation-progress",
            unit=unit,
            surface=name,
            held=held,
            detail=(walk.detail if not walk.transport_bounded else "") or duplicate or ("" if held else note),
        )


async def _collect_rows(
    surface: QuerySurface,
    expression: str,
    *,
    unit: QueryUnitName,
    page_size: int = LAW_PAGE_SIZE,
    max_pages: int = 200,
) -> tuple[RowMapping, ...]:
    """Page a surface to exhaustion by explicit offset.

    Advancing by the rows actually emitted (not by the requested size) keeps
    a transport that delivers a shorter page than it was asked for -- the MCP
    byte budget -- enumerating rather than skipping.
    """

    rows: list[RowMapping] = []
    offset = 0
    for _ in range(max_pages):
        try:
            page = await surface.page(expression, unit=unit, limit=page_size, offset=offset)
        except SurfaceUnavailableError:
            return tuple(rows)
        if not page.rows:
            return tuple(rows)
        rows.extend(page.rows)
        offset += len(page.rows)
    raise SurfaceUnavailableError(f"{surface.name} did not exhaust {expression!r} within {max_pages} pages")


async def _check_cross_surface_error(run: LawRun, bench: SurfaceBench) -> None:
    classes: dict[SurfaceName, str] = {}
    for name in SURFACE_NAMES:
        refusal = await bench.surface(name).refusal(UNCOMPILABLE_EXPRESSION)
        classes[name] = refusal.refusal_class
    distinct = set(classes.values())
    run.record(
        "cross-surface-error",
        held=len(distinct) == 1 and distinct == {REFUSAL_INVALID_QUERY},
        detail=f"refusal classes: {classes}",
    )


#: The structural outcome vocabulary the action unit publishes. ``unknown`` is
#: a retained deliberate unknown, never a success.
UNKNOWN_OUTCOME_STATES = frozenset({"outcome_unknown", "no_result"})


async def _check_unknown_outcome_agreement(run: LawRun, bench: SurfaceBench) -> None:
    probe = UNIT_PROBE_BY_UNIT["action"]
    facts: dict[SurfaceName, tuple[tuple[str, str, str, str], ...]] = {}
    for name in SURFACE_NAMES:
        rows = await _collect_rows(bench.surface(name), probe.scoped_expression, unit="action")
        facts[name] = tuple(
            (
                str(row.get("tool_use_block_id")),
                str(row.get("result_state")),
                str(row.get("is_error")),
                str(row.get("exit_code")),
            )
            for row in rows
        )
    reference = facts["python"]
    # A transport that stops early has still not disagreed: compare the prefix
    # it delivered, and let the continuation law own completeness.
    mismatched = sorted(
        name for name, observed in facts.items() if observed != reference[: len(observed)] or not observed
    )
    unknown_rows = tuple(row for row in reference if row[1] in UNKNOWN_OUTCOME_STATES)
    laundered = tuple(row for row in unknown_rows if row[2] == "False")
    detail = ""
    if mismatched:
        detail = f"surfaces disagree: {mismatched}"
    elif not unknown_rows:
        detail = "the corpus emitted no unknown/missing tool outcome, so the law was never exercised"
    elif laundered:
        detail = f"unknown outcome reported as a successful result: {laundered[:3]}"
    run.record("unknown-outcome-agreement", held=not detail, detail=detail)


async def _check_cancellation(run: LawRun, bench: SurfaceBench) -> None:
    """Cancel a real archive read before it runs and require a clean abort."""

    from polylogue.archive.query.execution_control import (
        QueryCancelledError,
        QueryExecutionContext,
        execute_archive_read,
    )
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    ctx = QueryExecutionContext.create(query_text="cancellation-law", timeout_s=30.0)
    ctx.cancel()

    def _work(archive: ArchiveStore) -> int:
        return int(archive._conn.execute("SELECT count(*) FROM messages").fetchone()[0])

    detail = ""
    try:
        await execute_archive_read(bench.archive_root, _work, ctx=ctx)
    except QueryCancelledError:
        pass
    except Exception as exc:
        detail = f"cancelled read raised {type(exc).__name__} instead of QueryCancelledError"
    else:
        detail = "a cancelled read completed and returned a page"
    if not detail:
        if ctx.receipt.state != "cancelled":
            detail = f"receipt state is {ctx.receipt.state!r}, not 'cancelled'"
        elif not ctx.receipt.cleanup_complete:
            detail = "the cancelled read did not complete its cleanup"
        elif ctx.receipt.rows_emitted:
            detail = f"the cancelled read emitted {ctx.receipt.rows_emitted} rows"
    run.record("cancellation-halts-work", held=not detail, detail=detail)


async def evaluate_query_laws(bench: SurfaceBench, *, probes: Sequence[UnitProbe] = UNIT_PROBES) -> LawRun:
    """Execute every declared law across every declared unit and surface."""

    run = LawRun()
    for probe in probes:
        await _check_unit_metamorphic_laws(run, bench, probe)
        await _check_unit_cross_surface_laws(run, bench, probe)
    await _check_cross_surface_error(run, bench)
    await _check_unknown_outcome_agreement(run, bench)
    await _check_cancellation(run, bench)
    return run


# ---------------------------------------------------------------------------
# Mutants
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LawMutant:
    """One production mutation and the law coordinates it must turn red.

    ``mutation`` names the defect in the same words the law's
    ``anti_vacuity`` sentence uses, so a law whose stated mutation is not
    actually wired here is visible rather than assumed.
    """

    mutant_id: str
    mutation: str
    expected_violations: tuple[str, ...]
    apply: Callable[[], Any]


@contextmanager
def _mutate_predicate_pushdown() -> Iterator[None]:
    """Lower every structural predicate as an always-true clause.

    The restriction stops reaching SQL, so a narrowing conjunct selects the
    whole relation and the census can no longer find its pushed-down marker.
    """

    from polylogue.storage.sqlite.archive_tiers import archive_query_reads

    def _always_true(
        unit: str, row_alias: str, predicate: object, *, session_alias: str | None = None
    ) -> tuple[str, list[object]]:
        return "1 = 1", []

    with patch.object(archive_query_reads, "_structural_predicate_clause", _always_true):
        yield


@contextmanager
def _mutate_continuation_state() -> Iterator[None]:
    """Hand back a continuation that replays its own offset."""

    from polylogue.archive.query.transaction import QueryTransactionRequest

    def _no_advance(self: QueryTransactionRequest, *, offset: int) -> QueryTransactionRequest:
        return self

    with patch.object(QueryTransactionRequest, "next", _no_advance):
        yield


@contextmanager
def _mutate_public_type() -> Iterator[None]:
    """Drop a declared public field from one surface's serialization."""

    from polylogue.mcp import server_support

    real = server_support._serialize_payload

    def _drop_field(payload: Any, *, exclude_none: bool = False) -> str:
        body = json.loads(real(payload, exclude_none=exclude_none))
        items = body.get("items") if isinstance(body, dict) else None
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    item.pop("title", None)
        return json.dumps(body)

    with patch.object(server_support, "_serialize_payload", _drop_field):
        yield


@contextmanager
def _mutate_ref_route() -> Iterator[None]:
    """Emit a non-canonical session ref the detail route cannot resolve."""

    from polylogue.archive.query import unit_results

    real = unit_results._row_payload_model

    def _mutated(descriptor: Any) -> Any:
        resolved = real(descriptor)
        if resolved is None:
            return None
        model = resolved

        class _Proxy:
            @classmethod
            def from_row(cls, row: Any) -> Any:
                payload = model.from_row(row)
                session_id = getattr(payload, "session_id", None)
                if isinstance(session_id, str):
                    return payload.model_copy(update={"session_id": f"{session_id}:"})
                return payload

        return _Proxy

    with patch.object(unit_results, "_row_payload_model", _mutated):
        yield


QUERY_LAW_MUTANTS: tuple[LawMutant, ...] = (
    LawMutant(
        "broken-predicate-pushdown",
        "lowering that folds one conjunct into a text filter applied after ordering",
        ("predicate-commutativity",),
        _mutate_predicate_pushdown,
    ),
    LawMutant(
        "broken-continuation-state",
        "a continuation that replays its own offset, or never clears has_next",
        ("continuation-progress",),
        _mutate_continuation_state,
    ),
    LawMutant(
        "broken-public-type",
        "a surface projecting a private column or dropping a declared public field",
        ("cross-surface-typing",),
        _mutate_public_type,
    ),
    LawMutant(
        "broken-ref-route",
        "a list route emitting an id shape the detail route cannot resolve",
        ("ref-canonicalization", "ref-detail-closure"),
        _mutate_ref_route,
    ),
)

QUERY_LAW_MUTANTS_BY_ID: Mapping[str, LawMutant] = {mutant.mutant_id: mutant for mutant in QUERY_LAW_MUTANTS}
