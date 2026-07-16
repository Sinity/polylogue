"""Archive surface adapters for scenario-driven verification tests.

This module hosts the cross-surface parity harness.  The substrate surfaces
(``SQLiteRecordSurface``, ``SQLiteHydratedSurface``, ``RepositorySurface``,
``FacadeSurface``) project the same archive through progressively higher
abstractions.  The adapter surfaces (``CLISurface``, ``MCPSurface``) close
the loop on the public surfaces — the Click query CLI and the MCP server
tools — so any drift between substrate and adapter is caught by the same
parity assertions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from click.testing import CliRunner

from polylogue.api import Polylogue
from tests.infra.archive_scenarios import ArchiveScenario, archive_for_scenario_db, open_index_db
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.semantic_facts import ArchiveFacts, SessionFacts


def _sorted_unique(values: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _origin_for_provider_token(provider: str) -> str:
    from polylogue.core.enums import Provider
    from polylogue.core.sources import origin_from_provider

    return origin_from_provider(Provider.from_string(provider)).value


def _provider_ids_from_connection(conn: sqlite3.Connection, provider: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT session_id FROM sessions WHERE origin = ? ORDER BY session_id",
        (_origin_for_provider_token(provider),),
    ).fetchall()
    return tuple(str(row["session_id"]) for row in rows)


def _search_ids_from_connection(conn: sqlite3.Connection, search_text: str) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT DISTINCT session_id
        FROM messages_fts
        WHERE messages_fts MATCH ?
        ORDER BY session_id
        """,
        (search_text,),
    ).fetchall()
    return tuple(str(row["session_id"]) for row in rows)


class ArchiveSurfaceAdapter(Protocol):
    """Common semantic projection surface over a seeded archive."""

    @property
    def name(self) -> str: ...

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts: ...

    async def archive_facts(self) -> ArchiveFacts: ...

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]: ...

    async def query_count(self, query_case: ArchiveQueryCase) -> int: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SQLiteRecordSurface:
    """Direct SQL/storage-record projection."""

    db_path: Path
    name: str = "sqlite-records"

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        with open_index_db(self.db_path) as conn:
            return scenario.facts_from_connection(conn)

    async def archive_facts(self) -> ArchiveFacts:
        with open_index_db(self.db_path) as conn:
            return ArchiveFacts.from_db_connection(conn)

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        with open_index_db(self.db_path) as conn:
            if query_case.provider is not None:
                return _provider_ids_from_connection(conn, query_case.provider)
            if query_case.search_text is not None:
                return _search_ids_from_connection(conn, query_case.search_text)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class SQLiteHydratedSurface:
    """Direct SQL projection after storage-record hydration."""

    db_path: Path
    scenarios: tuple[ArchiveScenario, ...]
    name: str = "sqlite-hydrated"

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        with open_index_db(self.db_path) as conn:
            return scenario.hydrated_facts_from_connection(conn)

    async def archive_facts(self) -> ArchiveFacts:
        with open_index_db(self.db_path) as conn:
            sessions = [scenario.hydrated_facts_from_connection(conn) for scenario in _current_archive_scenarios(conn)]
        return ArchiveFacts(
            total_sessions=len(sessions),
            provider_counts=_provider_counts(sessions),
            total_messages=sum(facts.message_count for facts in sessions),
            session_ids=tuple(facts.session_id for facts in sessions),
        )

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        with open_index_db(self.db_path) as conn:
            if query_case.provider is not None:
                return _provider_ids_from_connection(conn, query_case.provider)
            if query_case.search_text is not None:
                return _search_ids_from_connection(conn, query_case.search_text)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        return None


def _query_case_to_spec(query_case: ArchiveQueryCase) -> Any:
    """Build a SessionQuerySpec from a query case.

    Imported lazily so test modules that never construct spec-routed
    surfaces avoid the import-time cost of the archive query stack.
    """
    from polylogue.archive.query.spec import SessionQuerySpec

    origins: tuple[str, ...] = ()
    if query_case.provider is not None:
        origins = (_origin_for_provider_token(query_case.provider),)
    # Route the search text through ``contains_terms`` so it follows the
    # FTS-AND semantics the CLI exposes via --contains; the MCP
    # list_sessions ``contains`` arg lands on the same spec field
    # downstream.
    contains_terms = (query_case.search_text,) if query_case.search_text else ()
    return SessionQuerySpec(
        contains_terms=contains_terms,
        origins=origins,
        since=query_case.since,
        until=query_case.until,
        limit=query_case.limit,
        offset=query_case.offset,
        filter_has_tool_use=query_case.has_tool_use,
        filter_has_thinking=query_case.has_thinking,
        min_messages=query_case.min_messages,
        max_messages=query_case.max_messages,
        min_words=query_case.min_words,
    )


def _ids_from_sessions(sessions: list[Any]) -> tuple[str, ...]:
    return _sorted_unique([str(conv.id) for conv in sessions])


class RepositorySurface:
    """Facade projection with a long-lived mutating handle.

    Distinct from ``FacadeSurface`` only in lifecycle: the repository
    lifecycle harness holds this handle across tag/metadata mutations and
    delete transitions. Both project through the same ``Polylogue`` API over
    ``index.db``.
    """

    name = "repository"

    def __init__(self, db_path: Path) -> None:
        self._archive = archive_for_scenario_db(db_path)

    @property
    def archive(self) -> Polylogue:
        return self._archive

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        return await scenario.facts_from_archive(self._archive)

    async def archive_facts(self) -> ArchiveFacts:
        return ArchiveFacts.from_sessions(await self._archive.list_sessions(limit=None))

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        if query_case.has_extended_filters or query_case.search_text is not None:
            spec = _query_case_to_spec(query_case)
            sessions = await self._archive.list_sessions_for_spec(spec)
            return _ids_from_sessions(sessions)
        if query_case.provider is not None:
            sessions = await self._archive.list_sessions(
                origin=_origin_for_provider_token(query_case.provider), limit=None
            )
            return _ids_from_sessions(sessions)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        await self._archive.close()


class FacadeSurface:
    """Public async facade projection."""

    name = "facade"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        session = await self._archive.get_session(scenario.native_session_id)
        if session is None:
            raise AssertionError(f"Scenario {scenario.resolved_session_id!r} missing through facade")
        return SessionFacts.from_domain_session(session)

    async def archive_facts(self) -> ArchiveFacts:
        return ArchiveFacts.from_sessions(await self._archive.list_sessions(limit=None))

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        if query_case.has_extended_filters or query_case.search_text is not None:
            spec = _query_case_to_spec(query_case)
            # query_sessions runs the same filter chain that MCP/CLI use.
            sessions = await self._archive.list_sessions_for_spec(spec)
            return _ids_from_sessions(sessions)
        if query_case.provider is not None:
            sessions = await self._archive.list_sessions(
                origin=_origin_for_provider_token(query_case.provider), limit=None
            )
            return _ids_from_sessions(sessions)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        await self._archive.close()


class CLISurface:
    """Click-CLI adapter projection.

    Invokes the same ``polylogue [filters] read --all --format json`` path that
    operators run, then projects the parsed envelope into the canonical
    id-tuple shape used by the parity harness.

    Requires ``workspace_env`` (or equivalent ``POLYLOGUE_ARCHIVE_ROOT`` /
    ``XDG_DATA_HOME`` env vars) so the CLI process resolves the same
    backing database as the substrate surfaces.
    """

    name = "cli"

    def __init__(self, *, db_path: Path) -> None:
        # db_path is held for diagnostics; the CLI resolves it via env vars.
        self._db_path = db_path
        self._runner = CliRunner()

    def _invoke(self, query_case: ArchiveQueryCase) -> tuple[int, str]:
        from polylogue.cli.click_app import cli

        args: list[str] = []
        if query_case.provider is not None:
            args.extend(["--origin", _origin_for_provider_token(query_case.provider)])
        if query_case.since is not None:
            args.extend(["--since", query_case.since])
        if query_case.until is not None:
            args.extend(["--until", query_case.until])
        if query_case.has_tool_use:
            args.append("--has-tool-use")
        if query_case.has_thinking:
            args.append("--has-thinking")
        if query_case.min_messages is not None:
            args.extend(["--min-messages", str(query_case.min_messages)])
        if query_case.max_messages is not None:
            args.extend(["--max-messages", str(query_case.max_messages)])
        if query_case.min_words is not None:
            args.extend(["--min-words", str(query_case.min_words)])
        if query_case.limit is not None:
            args.extend(["--limit", str(query_case.limit)])
        if query_case.offset:
            args.extend(["--offset", str(query_case.offset)])
        # Use --contains for FTS so the option parser routes the term
        # deterministically as a query token rather than relying on the
        # positional bare-token interception.
        if query_case.search_text is not None:
            args.extend(["--contains", query_case.search_text])
        args.extend(["read", "--all", "--format", "json"])
        result = self._runner.invoke(cli, args, catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        return result.exit_code, result.output

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        raise NotImplementedError("CLISurface only supports query-level projections")

    async def archive_facts(self) -> ArchiveFacts:
        raise NotImplementedError("CLISurface only supports query-level projections")

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        exit_code, output = self._invoke(query_case)
        # Empty result: exit 2 with the no_results error envelope (JSON object).
        if exit_code == 2:
            try:
                parsed = json.loads(output)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"CLI no-results output is not JSON: {output!r}") from exc
            if isinstance(parsed, dict) and parsed.get("status") == "error" and parsed.get("code") == "no_results":
                return ()
            if isinstance(parsed, dict) and parsed.get("items") == [] and parsed.get("total") == 0:
                return ()
            raise AssertionError(f"CLI exited 2 but envelope is not no_results: {parsed!r}")
        if exit_code != 0:
            raise AssertionError(
                f"polylogue read --all --format json exited {exit_code} for {query_case.name!r}: {output!r}"
            )
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"CLI list output is not JSON ({query_case.name!r}): {output!r}") from exc
        # Three shapes can land here:
        #   * list mode envelope (#1618): ``{"items": [...], "total": ..., ...}``
        #   * search-hit mode (historical bare array): ``[{"session": {"id": ...}, ...}, ...]``
        #   * typed ranked-result envelope (PR #1370):
        #     ``{"hits": [{"session": {"id": ...}, "match": {...}}, ...], "limit": ..., ...}``
        # All three project to the same id tuple for parity comparison.
        if isinstance(parsed, dict) and isinstance(parsed.get("hits"), list):
            parsed = parsed["hits"]
        elif isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
            parsed = parsed["items"]
        if not isinstance(parsed, list):
            raise AssertionError(f"CLI list output is not a JSON array ({query_case.name!r}): {parsed!r}")
        ids: list[str] = []
        for row in parsed:
            if not isinstance(row, dict):
                continue
            if "id" in row:
                ids.append(str(row["id"]))
            elif "session" in row and isinstance(row["session"], dict):
                conv = row["session"]
                if "id" in conv:
                    ids.append(str(conv["id"]))
        return _sorted_unique(ids)

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        return None


class MCPSurface:
    """MCP server-tool adapter projection.

    Invokes ``list_sessions`` directly on the registered FastMCP tool
    function, bound to a ``RuntimeServices`` instance pointed at the parity
    database.  This is the same code path MCP clients hit over the JSON-RPC
    transport.
    """

    name = "mcp"

    def __init__(self, *, db_path: Path) -> None:
        from polylogue.mcp.server import build_server
        from polylogue.mcp.server_support import _set_runtime_services
        from polylogue.services import build_runtime_services

        self._services = build_runtime_services(db_path=db_path)
        _set_runtime_services(self._services)
        self._server = build_server(role="admin")

    def _tool(self, name: str) -> Any:
        return self._server._tool_manager._tools[name].fn

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        raise NotImplementedError("MCPSurface only supports query-level projections")

    async def archive_facts(self) -> ArchiveFacts:
        raise NotImplementedError("MCPSurface only supports query-level projections")

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        # MCP enforces clamped limits (min 1); use a generous ceiling so the
        # parity matrix is dominated by the case-level expected count rather
        # than transport-level bounds.
        request_limit = query_case.limit if query_case.limit is not None else 1000
        payload_json = await self._tool("list_sessions")(
            limit=request_limit,
            origin=_origin_for_provider_token(query_case.provider) if query_case.provider is not None else None,
            contains=query_case.search_text,
            since=query_case.since,
            until=query_case.until,
            has_tool_use=query_case.has_tool_use,
            has_thinking=query_case.has_thinking,
            min_messages=query_case.min_messages,
            max_messages=query_case.max_messages,
            min_words=query_case.min_words,
            offset=query_case.offset,
        )
        parsed = json.loads(payload_json)
        items = parsed.get("items", [])
        ids: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "id" in item:
                ids.append(str(item["id"]))
            elif isinstance(item.get("session"), dict) and "id" in item["session"]:
                ids.append(str(item["session"]["id"]))
        return _sorted_unique(ids)

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        await self._services.close()
        _set_runtime_services(None)


class DaemonHTTPSurface:
    """Daemon HTTP adapter projection.

    Starts the production ``DaemonAPIHTTPServer`` against the archive resolved
    by ``workspace_env`` and projects ``GET /api/sessions`` rows into the
    canonical id tuple used by the parity harness.
    """

    name = "daemon-http"

    def __init__(self, *, db_path: Path) -> None:
        from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer

        self._db_path = db_path
        self._server = DaemonAPIHTTPServer(("127.0.0.1", 0), DaemonAPIHandler)
        self._server.auth_token = ""
        self._server.api_host = "127.0.0.1"
        port = self._server.server_address[1]
        self._base_url = f"http://127.0.0.1:{port}"
        self._thread = threading.Thread(target=self._server.serve_forever, name="daemon-http-surface", daemon=True)
        self._thread.start()

    async def session_facts(self, scenario: ArchiveScenario) -> SessionFacts:
        raise NotImplementedError("DaemonHTTPSurface only supports query-level projections")

    async def archive_facts(self) -> ArchiveFacts:
        raise NotImplementedError("DaemonHTTPSurface only supports query-level projections")

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        params: dict[str, object] = {}
        if query_case.provider is not None:
            params["origin"] = _origin_for_provider_token(query_case.provider)
        if query_case.search_text is not None:
            params["contains"] = query_case.search_text
        if query_case.since is not None:
            params["since"] = query_case.since
        if query_case.until is not None:
            params["until"] = query_case.until
        if query_case.has_tool_use:
            params["has_tool_use"] = "true"
        if query_case.has_thinking:
            params["has_thinking"] = "true"
        if query_case.min_messages is not None:
            params["min_messages"] = query_case.min_messages
        if query_case.max_messages is not None:
            params["max_messages"] = query_case.max_messages
        if query_case.min_words is not None:
            params["min_words"] = query_case.min_words
        if query_case.limit is not None:
            params["limit"] = query_case.limit
        if query_case.offset:
            params["offset"] = query_case.offset

        query = urlencode(params)
        url = f"{self._base_url}/api/sessions"
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items = []
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                items = payload["items"]
            elif isinstance(payload.get("hits"), list):
                items = payload["hits"]
        ids: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if "id" in item:
                ids.append(str(item["id"]))
            elif isinstance(item.get("session"), dict) and "id" in item["session"]:
                ids.append(str(item["session"]["id"]))
        return _sorted_unique(ids)

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


@dataclass(slots=True)
class ArchiveSurfaceSet:
    """Closable group of archive surface adapters."""

    surfaces: tuple[ArchiveSurfaceAdapter, ...]

    async def close(self) -> None:
        for surface in self.surfaces:
            await surface.close()


def build_archive_surface_set(
    *,
    db_path: Path,
    archive_root: Path,
    scenarios: tuple[ArchiveScenario, ...],
) -> ArchiveSurfaceSet:
    """Build the standard cross-surface set for a scenario archive."""
    surfaces: tuple[ArchiveSurfaceAdapter, ...] = (
        SQLiteRecordSurface(db_path),
        SQLiteHydratedSurface(db_path, scenarios),
        RepositorySurface(db_path),
        FacadeSurface(archive_root=archive_root, db_path=db_path),
    )
    return ArchiveSurfaceSet(
        surfaces=surfaces,
    )


def build_adapter_surface_set(
    *,
    db_path: Path,
    archive_root: Path,
) -> ArchiveSurfaceSet:
    """Build the repo/facade/cli/mcp/daemon adapter parity set.

    Excludes the raw-SQL projections (``SQLiteRecordSurface``,
    ``SQLiteHydratedSurface``) because they only express provider and
    text-search projections; the adapter parity matrix needs to exercise
    the full filter chain (date range, limit/offset, stats-join
    pushdowns) which the substrate SQL views cannot project without
    re-implementing the spec.
    """
    surfaces: tuple[ArchiveSurfaceAdapter, ...] = (
        RepositorySurface(db_path),
        FacadeSurface(archive_root=archive_root, db_path=db_path),
        CLISurface(db_path=db_path),
        MCPSurface(db_path=db_path),
        DaemonHTTPSurface(db_path=db_path),
    )
    return ArchiveSurfaceSet(surfaces=surfaces)


def _provider_counts(sessions: list[SessionFacts]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for session in sessions:
        counts[session.provider] = counts.get(session.provider, 0) + 1
    return counts


def _current_archive_scenarios(conn: sqlite3.Connection) -> tuple[ArchiveScenario, ...]:
    rows = conn.execute("SELECT origin, native_id FROM sessions ORDER BY session_id").fetchall()
    from polylogue.core.enums import Origin
    from polylogue.core.sources import provider_from_origin

    scenarios: list[ArchiveScenario] = []
    for row in rows:
        # native_id is ``ext-<resolved_session_id>``; recover the raw id
        # so the rebuilt scenario re-derives the same archive session id.
        native_id = str(row["native_id"])
        raw_id = native_id[len("ext-") :] if native_id.startswith("ext-") else native_id
        provider = str(provider_from_origin(Origin.from_string(str(row["origin"]))))
        scenarios.append(ArchiveScenario(name=raw_id, provider=provider))
    return tuple(scenarios)


__all__ = [
    "ArchiveSurfaceAdapter",
    "ArchiveSurfaceSet",
    "CLISurface",
    "DaemonHTTPSurface",
    "FacadeSurface",
    "MCPSurface",
    "RepositorySurface",
    "SQLiteHydratedSurface",
    "SQLiteRecordSurface",
    "build_adapter_surface_set",
    "build_archive_surface_set",
]
