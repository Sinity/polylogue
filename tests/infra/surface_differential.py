"""Cross-surface differential: one request, four public routes, one oracle.

Every adapter here carries one request to a public read surface and projects
the answer down to :class:`SurfaceQueryFacts` — session ids and the reported
total.  Comparing those facts against the reference model turns cross-surface
parity from "these two hand-written expectations agree" into "every surface
computes what the declared corpus says".  Three of the four carry the query
DSL itself; the fourth carries the same filter through the only parameters it
has, and says so.

The surfaces are the ones an operator or client actually reaches:

* ``api`` — ``compile_expression`` plus the ``Polylogue`` facade, the shared
  lowering every other surface funnels through.
* ``cli`` — ``polylogue find <expr> --format json``.
* ``mcp`` — ``query(projection="sessions", ...)``, whose session projection
  carries named filters rather than the DSL, so the adapter translates the
  requests those filters can express and declines the rest.
* ``daemon`` — ``GET /api/sessions?query=<expr>``.

``tests/infra/surfaces.py`` holds the older scenario parity harness, which
compares surfaces to each other over a hand-written ``ArchiveQueryCase``.
This module is the oracle-backed route: the expectation is computed, and a
false-empty on one surface cannot be absorbed by the other three agreeing.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from polylogue.archive.query.expression import _FieldToken, parse_expression_ast
from polylogue.archive.query.predicate import QueryBoolPredicate, QueryFieldPredicate, QueryPredicate
from tests.infra.reference_model import ModelRequest, ReferenceArchive

#: Surface names, in the order a report lists them.
SURFACE_NAMES: tuple[str, ...] = ("api", "cli", "mcp", "daemon")


@dataclass(frozen=True, slots=True)
class SurfaceQueryFacts:
    """What one surface answered, reduced to comparable meaning."""

    surface: str
    #: The request this surface was actually asked.  A surface whose transport
    #: cannot express part of a request answers a weaker one and says so here,
    #: so the comparison is against the model's answer to the question that was
    #: really put — never against a stronger one the surface never saw.
    request: ModelRequest
    #: Session ids in envelope order, duplicates preserved.  A session listed
    #: twice by a session-grain envelope has a join or pagination defect, and
    #: collapsing the rows here would hide exactly that.
    session_ids: tuple[str, ...]
    #: The surface's own reported total, or ``None`` when its envelope does
    #: not carry one.  ``None`` is not zero and is never compared as one.
    total: int | None
    #: Whether one row means one session.  A ranked envelope's rows are
    #: matches, so the same session appearing twice there is two hits and not
    #: a defect; a session list repeating a row is.
    rows_are_sessions: bool = True
    #: ``False`` when this request has FTS/vector rank order. The reference
    #: model predicts membership and count for those requests, but does not
    #: duplicate production's ranking policy. Structured session pages have a
    #: declared date/id order and remain exact tuple comparisons.
    order_is_declared: bool = True

    @property
    def id_set(self) -> frozenset[str]:
        return frozenset(self.session_ids)

    @property
    def duplicate_ids(self) -> tuple[str, ...]:
        if not self.rows_are_sessions:
            return ()
        counts = Counter(self.session_ids)
        return tuple(sorted(session_id for session_id, seen in counts.items() if seen > 1))


class ExpressionSurface(Protocol):
    """A public read route that can carry a query-DSL expression."""

    @property
    def name(self) -> str: ...

    async def execute(self, request: ModelRequest) -> SurfaceQueryFacts | None:
        """Answer ``request``, or ``None`` when this surface cannot carry it."""
        ...

    async def close(self) -> None: ...


def _ids_from_rows(rows: object) -> tuple[tuple[str, ...], bool]:
    """Project a row list into session ids, and say what one row means.

    Two row shapes reach here.  A session row carries ``id`` directly and one
    row is one session, so a repeat is a defect.  A ranked row nests the
    session under ``session`` beside its ``match``, and one row is one match,
    so the same session may legitimately appear several times.  Envelope order
    is preserved either way, so a diagnostic reads the way the surface
    paginated.
    """
    if not isinstance(rows, list):
        return (), True
    ids: list[str] = []
    rows_are_sessions = True
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "id" in row:
            ids.append(str(row["id"]))
            continue
        session = row.get("session")
        if isinstance(session, dict) and "id" in session:
            rows_are_sessions = False
            ids.append(str(session["id"]))
    return tuple(ids), rows_are_sessions


def _rows_and_total(payload: object) -> tuple[object, int | None]:
    """Split a read envelope into its rows and its reported total.

    Four envelopes reach here: a bare array, the ``items`` list (whose rows are
    sessions on a listing route and matches on the ranked one), the ``hits``
    list, and the single-session document a surface may answer an id filter
    with.  What one row *means* is read off the row shape, not the key.
    """
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        return [], None
    total = payload.get("total")
    reported = total if isinstance(total, int) and not isinstance(total, bool) else None
    rows = payload.get("items")
    if not isinstance(rows, list):
        rows = payload.get("hits")
    if isinstance(rows, list):
        return rows, reported
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        # A single-session document: the surface collapsed a filter selecting
        # one session into a read of it.  That is one session matched.
        return [{"id": session_id}], reported
    return [], reported


def _request_has_ranked_order(request: ModelRequest) -> bool:
    """Whether production gives this request retrieval rather than list order.

    ``SessionQueryPlan`` preserves the retrieval order whenever its FTS or
    vector lane is active. This asks production's compiler, instead of
    creating a second classification of expression spellings in the test
    harness.
    """
    from polylogue.archive.query.expression import compile_expression

    spec = compile_expression(request.expression)
    return bool(
        spec.query_terms
        or spec.contains_terms
        or spec.similar_text
        or spec.similar_session_id
        or spec.retrieval_lane in {"semantic", "hybrid"}
    )


class ApiExpressionSurface:
    """The Python facade over the shared expression lowering."""

    name = "api"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        from polylogue.api import Polylogue

        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def execute(self, request: ModelRequest) -> SurfaceQueryFacts:
        from polylogue.archive.query.expression import compile_expression

        # The adapter supplies the requested window to the product route. It
        # may observe the result, but must not implement pagination by
        # fetching an unbounded set and slicing ids itself.
        spec = replace(compile_expression(request.expression), limit=request.limit, offset=request.offset)
        sessions = await self._archive.list_sessions_for_spec(spec)
        ids = tuple(str(session.id) for session in sessions)
        # Count through the same product plan without its display window.
        # This preserves the envelope contract that ``total`` names matches
        # before paging, while the page itself remains entirely product-owned.
        total = await replace(spec, limit=None, offset=0).count(self._archive.config)
        return SurfaceQueryFacts(
            surface=self.name,
            request=request,
            session_ids=ids,
            total=total,
            order_is_declared=not bool(
                spec.query_terms
                or spec.contains_terms
                or spec.similar_text
                or spec.similar_session_id
                or spec.retrieval_lane in {"semantic", "hybrid"}
            ),
        )

    async def close(self) -> None:
        await self._archive.close()


class CliExpressionSurface:
    """``polylogue find <expression> --format json``."""

    name = "cli"

    def __init__(self, *, db_path: Path) -> None:
        from click.testing import CliRunner

        self._db_path = db_path
        self._runner = CliRunner()

    def _invoke(self, args: list[str]) -> tuple[int, str]:
        from polylogue.cli.click_app import cli

        result = self._runner.invoke(cli, args, catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        return result.exit_code, result.output

    async def execute(self, request: ModelRequest) -> SurfaceQueryFacts:
        args = ["find", request.expression, "--format", "json"]
        if request.limit is not None:
            args.extend(["--limit", str(request.limit)])
        if request.offset:
            args.extend(["--offset", str(request.offset)])
        exit_code, output = self._invoke(args)
        # An empty match set exits 2 while still emitting the ordinary list
        # envelope; that is a shell status, not a failed read, and the
        # envelope's own ``total`` is the fact to compare.
        if exit_code not in {0, 2}:
            raise AssertionError(f"cli find {request.expression!r} exited {exit_code}: {output!r}")
        payload = json.loads(output)
        if exit_code == 2 and not (isinstance(payload, dict) and isinstance(payload.get("items"), list)):
            raise AssertionError(f"cli find {request.expression!r} exited 2 without a result envelope: {output!r}")
        # ``find id:<session>`` is answered with the session document rather
        # than a one-row list; ``_rows_and_total`` reads it as the one session
        # it names.
        rows, total = _rows_and_total(payload)
        session_ids, rows_are_sessions = _ids_from_rows(rows)
        return SurfaceQueryFacts(
            surface=self.name,
            request=request,
            session_ids=session_ids,
            total=total,
            rows_are_sessions=rows_are_sessions,
            order_is_declared=not _request_has_ranked_order(request),
        )

    async def close(self) -> None:
        return None


#: The largest page the MCP session projection is asked for.  Its ``limit``
#: is clamped, and omitting it falls back to a small default page, so an
#: exhaustive listing has to name a ceiling above the corpus size.
MCP_EXHAUSTIVE_LIMIT: int = 1000


def mcp_session_filters(expression: str) -> dict[str, object] | None:
    """Translate ``expression`` into MCP session-filter parameters.

    ``query(projection="sessions")`` reads its ``expression`` as free text for
    a ranked search: the session projection carries no DSL, only the named
    filters ``origin``/``tag``/``repo``/``since``/``until``/``min_messages``/
    ``max_messages``/``min_words``.  This translates the conjunctions those
    parameters can express and returns ``None`` for everything else, so the
    MCP leg compares the filters MCP really has rather than sending it a
    expression it would silently reduce to a word search.
    """
    from tests.infra.reference_model import predicate_from_ast

    predicate = predicate_from_ast(parse_expression_ast(expression))
    if predicate is None:
        return {}
    conjuncts: list[QueryPredicate] = []
    pending = [predicate]
    while pending:
        current = pending.pop()
        if isinstance(current, QueryBoolPredicate) and current.op == "and":
            pending.extend(current.children)
            continue
        conjuncts.append(current)

    filters: dict[str, object] = {}

    def claim(name: str, value: object) -> bool:
        # A second value for one parameter is a filter MCP cannot express:
        # ``origin:a AND origin:b`` is empty, ``tag:x AND tag:y`` needs both,
        # and one slot can only carry one of them.
        if name in filters:
            return False
        filters[name] = value
        return True

    for conjunct in conjuncts:
        if not isinstance(conjunct, QueryFieldPredicate) or len(conjunct.values) != 1:
            return None
        field = conjunct.field.removeprefix("session.")
        value = conjunct.values[0]
        if field in {"origin", "tag", "repo"} and conjunct.op == "=":
            if not claim(field, value):
                return None
        elif field == "since" or (field == "date" and conjunct.op == ">="):
            if not claim("since", value):
                return None
        elif field == "until" or (field == "date" and conjunct.op == "<="):
            if not claim("until", value):
                return None
        elif field == "messages" and conjunct.op == ">=":
            if not claim("min_messages", int(value)):
                return None
        elif field == "messages" and conjunct.op == "<=":
            if not claim("max_messages", int(value)):
                return None
        elif field == "words" and conjunct.op == ">=":
            if not claim("min_words", int(value)):
                return None
        else:
            return None
    return filters


class McpExpressionSurface:
    """``query(projection="sessions", ...)`` on the registered tool.

    A request whose filter MCP cannot express is not answered at all.
    """

    name = "mcp"

    def __init__(self, *, db_path: Path) -> None:
        from polylogue.mcp.server import build_server
        from polylogue.mcp.server_support import _set_runtime_services
        from polylogue.services import build_runtime_services
        from tests.infra.mcp import ALL_CAPABILITIES

        self._services = build_runtime_services(db_path=db_path)
        _set_runtime_services(self._services)
        self._server = build_server(capabilities=ALL_CAPABILITIES)

    def _tool(self, name: str) -> Any:
        return self._server._tool_manager._tools[name].fn

    async def execute(self, request: ModelRequest) -> SurfaceQueryFacts | None:
        filters = mcp_session_filters(request.expression)
        if filters is None:
            return None
        limit = request.limit if request.limit is not None else MCP_EXHAUSTIVE_LIMIT
        payload = await self._tool("query")(projection="sessions", limit=limit, offset=request.offset, **filters)
        rows, total = _rows_and_total(json.loads(payload))
        session_ids, rows_are_sessions = _ids_from_rows(rows)
        return SurfaceQueryFacts(
            surface=self.name,
            request=request,
            session_ids=session_ids,
            total=total,
            rows_are_sessions=rows_are_sessions,
            order_is_declared=not _request_has_ranked_order(request),
        )

    async def close(self) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        await self._services.close()
        _set_runtime_services(None)


class DaemonExpressionSurface:
    """``GET /api/sessions?query=<expression>`` on the production HTTP server."""

    name = "daemon"

    def __init__(self, *, db_path: Path) -> None:
        from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

        self._db_path = db_path
        self._server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
        self._server.auth_token = ""
        self._server.api_host = "127.0.0.1"
        self._base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="daemon-expression-surface",
            daemon=True,
        )
        self._thread.start()

    async def execute(self, request: ModelRequest) -> SurfaceQueryFacts:
        params: dict[str, object] = {"query": request.expression, "limit": request.limit or 1000}
        if request.offset:
            params["offset"] = request.offset
        url = f"{self._base_url}/api/sessions?{urlencode(params)}"
        http_request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(http_request, timeout=10.0) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise AssertionError(f"daemon /api/sessions {request.expression!r} returned {exc.code}") from exc
        rows, total = _rows_and_total(json.loads(body))
        session_ids, rows_are_sessions = _ids_from_rows(rows)
        return SurfaceQueryFacts(
            surface=self.name,
            request=request,
            session_ids=session_ids,
            total=total,
            rows_are_sessions=rows_are_sessions,
            order_is_declared=not _request_has_ranked_order(request),
        )

    async def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


@dataclass(slots=True)
class ExpressionSurfaceSet:
    """Closable set of expression-carrying surfaces."""

    surfaces: tuple[ExpressionSurface, ...]

    async def execute(self, request: ModelRequest) -> tuple[SurfaceQueryFacts, ...]:
        """Answer ``request`` on every surface that can carry it.

        A surface that cannot express the request is absent from the result
        rather than answering a different question.  Callers assert their own
        coverage floor: silently answering nowhere is how a differential goes
        vacuous.
        """
        answered: list[SurfaceQueryFacts] = []
        for surface in self.surfaces:
            facts = await surface.execute(request)
            if facts is not None:
                answered.append(facts)
        return tuple(answered)

    async def close(self) -> None:
        for surface in self.surfaces:
            await surface.close()


def build_expression_surface_set(
    *,
    archive_root: Path,
    db_path: Path,
    names: Sequence[str] = SURFACE_NAMES,
) -> ExpressionSurfaceSet:
    """Build the named expression surfaces over one archive."""
    builders: dict[str, Callable[[], ExpressionSurface]] = {
        "api": lambda: ApiExpressionSurface(archive_root=archive_root, db_path=db_path),
        "cli": lambda: CliExpressionSurface(db_path=db_path),
        "mcp": lambda: McpExpressionSurface(db_path=db_path),
        "daemon": lambda: DaemonExpressionSurface(db_path=db_path),
    }
    unknown = sorted(set(names) - set(builders))
    if unknown:
        raise ValueError(f"unknown expression surfaces: {unknown}")
    return ExpressionSurfaceSet(surfaces=tuple(builders[name]() for name in names))


@dataclass(frozen=True, slots=True)
class Divergence:
    """One surface disagreeing with the model about one request."""

    request: ModelRequest
    surface: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.surface}] {self.request.name} {self.request.expression!r}: {self.detail}"


def compare_to_model(
    model: ReferenceArchive,
    observed: Sequence[SurfaceQueryFacts],
    *,
    compare_totals: bool = True,
) -> tuple[Divergence, ...]:
    """Return every way ``observed`` disagrees with the model.

    Each surface is asked for the model's answer to the request *it* was
    given, which is how a surface that cannot express an offset is still
    compared honestly.  Ranked-search rows are compared as sets because the
    model deliberately does not predict relevance order.  A session listing,
    however, has a declared default order and a page boundary: compare that
    tuple exactly.  The separate repeat check makes a duplicated one-row page
    visible too.  Totals are compared only where a surface reports one, and
    always at the pre-window grain — a surface that reports its page size as
    the total is a count-grain error, which is exactly what this comparison
    exists to name.
    """
    divergences: list[Divergence] = []
    for facts in observed:
        expected = model.query(facts.request)
        if facts.id_set != expected.id_set:
            missing = sorted(expected.id_set - facts.id_set)
            extra = sorted(facts.id_set - expected.id_set)
            divergences.append(
                Divergence(
                    request=facts.request,
                    surface=facts.surface,
                    detail=f"ids differ (missing={missing}, unexpected={extra})",
                )
            )
        if facts.rows_are_sessions and facts.order_is_declared and facts.session_ids != expected.session_ids:
            divergences.append(
                Divergence(
                    request=facts.request,
                    surface=facts.surface,
                    detail=(
                        f"session page order differs "
                        f"(expected={list(expected.session_ids)}, observed={list(facts.session_ids)})"
                    ),
                )
            )
        if facts.duplicate_ids:
            divergences.append(
                Divergence(
                    request=facts.request,
                    surface=facts.surface,
                    detail=f"sessions returned more than once: {list(facts.duplicate_ids)}",
                )
            )
        if compare_totals and facts.total is not None and facts.total != expected.total:
            divergences.append(
                Divergence(
                    request=facts.request,
                    surface=facts.surface,
                    detail=f"total {facts.total} != model total {expected.total}",
                )
            )
    return tuple(divergences)


def format_divergences(divergences: Sequence[Divergence]) -> str:
    return "\n".join(f"  {item}" for item in divergences)


# ---------------------------------------------------------------------------
# Known divergences
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KnownDivergence:
    """A defect the surfaces are already known to have.

    Generated requests matching one are held out of the differential, so a
    seed that happens to draw the shape does not re-report the same known
    defect as a new one.  Each entry is pinned by a test that reproduces it
    directly: when the defect is fixed that test goes red, and the entry has
    to be deleted before the suite is green again.  That is the only thing
    keeping this list from becoming a place to hide failures.
    """

    name: str
    statement: str
    surfaces: tuple[str, ...]
    matches: Callable[[str], bool]


def _carries_compact_tag_clause(expression: str) -> bool:
    ast = parse_expression_ast(expression)
    return ast.boolean_predicate is None and any(
        isinstance(token, _FieldToken) and token.field == "tag" for token in ast.clauses
    )


KNOWN_DIVERGENCES: tuple[KnownDivergence, ...] = (
    KnownDivergence(
        name="spec-tags-postfilter-drops-user-tags",
        statement=(
            "A compact `tag:<name>` clause compiles to `SessionQuerySpec.tags`, which "
            "`Polylogue.list_sessions_for_spec` answers with an in-memory postfilter over each "
            "hydrated `Session.tags`. Hydration does not carry user tag assertions, so a session "
            "tagged through `polylogue mark --tag` is filtered out and the route returns nothing. "
            "The Boolean spelling of the same clause queries the union of auto tags and user "
            "assertions in SQL and is correct, and the CLI and daemon reach the compact clause "
            "through a different reader and are correct too."
        ),
        surfaces=("api",),
        matches=_carries_compact_tag_clause,
    ),
)


def known_divergence_for(request: ModelRequest) -> KnownDivergence | None:
    """The ledger entry covering ``request``, if any."""
    for entry in KNOWN_DIVERGENCES:
        if entry.matches(request.expression):
            return entry
    return None


__all__ = [
    "ApiExpressionSurface",
    "CliExpressionSurface",
    "DaemonExpressionSurface",
    "Divergence",
    "ExpressionSurface",
    "ExpressionSurfaceSet",
    "KNOWN_DIVERGENCES",
    "KnownDivergence",
    "MCP_EXHAUSTIVE_LIMIT",
    "McpExpressionSurface",
    "SURFACE_NAMES",
    "SurfaceQueryFacts",
    "build_expression_surface_set",
    "compare_to_model",
    "format_divergences",
    "known_divergence_for",
    "mcp_session_filters",
]
