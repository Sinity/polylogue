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
    UNIT_IDENTITY_BY_UNIT,
    SurfaceName,
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
        return _normalize(self.name, unit, payload, response_bytes=size)

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
