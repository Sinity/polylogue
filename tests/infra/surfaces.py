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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from click.testing import CliRunner

from polylogue.api import Polylogue
from polylogue.storage.sqlite.connection import open_connection
from tests.infra.archive_scenarios import ArchiveScenario, repository_for_scenario_db
from tests.infra.query_cases import ArchiveQueryCase
from tests.infra.semantic_facts import ArchiveFacts, ConversationFacts


def _sorted_unique(values: list[str] | tuple[str, ...] | set[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _provider_ids_from_connection(conn: sqlite3.Connection, provider: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT conversation_id FROM conversations WHERE provider_name = ? ORDER BY conversation_id",
        (provider,),
    ).fetchall()
    return tuple(str(row["conversation_id"]) for row in rows)


def _search_ids_from_connection(conn: sqlite3.Connection, search_text: str) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT DISTINCT conversation_id
        FROM messages_fts
        WHERE messages_fts MATCH ?
        ORDER BY conversation_id
        """,
        (search_text,),
    ).fetchall()
    return tuple(str(row["conversation_id"]) for row in rows)


class ArchiveSurfaceAdapter(Protocol):
    """Common semantic projection surface over a seeded archive."""

    @property
    def name(self) -> str: ...

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts: ...

    async def archive_facts(self) -> ArchiveFacts: ...

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]: ...

    async def query_count(self, query_case: ArchiveQueryCase) -> int: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SQLiteRecordSurface:
    """Direct SQL/storage-record projection."""

    db_path: Path
    name: str = "sqlite-records"

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        with open_connection(self.db_path) as conn:
            return scenario.facts_from_connection(conn)

    async def archive_facts(self) -> ArchiveFacts:
        with open_connection(self.db_path) as conn:
            return ArchiveFacts.from_db_connection(conn)

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        with open_connection(self.db_path) as conn:
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

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        with open_connection(self.db_path) as conn:
            return scenario.hydrated_facts_from_connection(conn)

    async def archive_facts(self) -> ArchiveFacts:
        with open_connection(self.db_path) as conn:
            conversations = [
                scenario.hydrated_facts_from_connection(conn) for scenario in _current_archive_scenarios(conn)
            ]
        return ArchiveFacts(
            total_conversations=len(conversations),
            provider_counts=_provider_counts(conversations),
            total_messages=sum(facts.message_count for facts in conversations),
            conversation_ids=tuple(facts.conversation_id for facts in conversations),
        )

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        with open_connection(self.db_path) as conn:
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
    """Build a ConversationQuerySpec from a query case.

    Imported lazily so test modules that never construct spec-routed
    surfaces avoid the import-time cost of the archive query stack.
    """
    from polylogue.archive.query.spec import ConversationQuerySpec
    from polylogue.types import Provider

    providers: tuple[Any, ...] = ()
    if query_case.provider is not None:
        providers = (Provider.from_string(query_case.provider),)
    # Route the search text through ``contains_terms`` so it follows the
    # FTS-AND semantics the CLI exposes via --contains; the MCP
    # list_conversations ``contains`` arg lands on the same spec field
    # downstream.
    contains_terms = (query_case.search_text,) if query_case.search_text else ()
    return ConversationQuerySpec(
        contains_terms=contains_terms,
        providers=providers,
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


def _ids_from_conversations(conversations: list[Any]) -> tuple[str, ...]:
    return _sorted_unique([str(conv.id) for conv in conversations])


class RepositorySurface:
    """Async repository projection."""

    name = "repository"

    def __init__(self, db_path: Path) -> None:
        self._repository = repository_for_scenario_db(db_path)

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        return await scenario.facts_from_repository(self._repository)

    async def archive_facts(self) -> ArchiveFacts:
        return ArchiveFacts.from_conversations(await self._repository.list(limit=None))

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        if query_case.has_extended_filters or query_case.search_text is not None:
            spec = _query_case_to_spec(query_case)
            conversations = await spec.list(self._repository)
            return _ids_from_conversations(conversations)
        if query_case.provider is not None:
            conversations = await self._repository.list(provider=query_case.provider, limit=None)
            return _ids_from_conversations(conversations)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        if query_case.has_extended_filters or query_case.search_text is not None:
            spec = _query_case_to_spec(query_case)
            return int(await spec.count(self._repository))
        if query_case.provider is not None:
            return await self._repository.count(provider=query_case.provider)
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        await self._repository.close()


class FacadeSurface:
    """Public async facade projection."""

    name = "facade"

    def __init__(self, *, archive_root: Path, db_path: Path) -> None:
        self._archive = Polylogue(archive_root=archive_root, db_path=db_path)

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        conversation = await self._archive.get_conversation(scenario.resolved_conversation_id)
        if conversation is None:
            raise AssertionError(f"Scenario {scenario.resolved_conversation_id!r} missing through facade")
        return ConversationFacts.from_domain_conversation(conversation)

    async def archive_facts(self) -> ArchiveFacts:
        return ArchiveFacts.from_conversations(await self._archive.list_conversations(limit=None))

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        if query_case.has_extended_filters or query_case.search_text is not None:
            spec = _query_case_to_spec(query_case)
            # query_conversations runs the same filter chain that MCP/CLI use.
            conversations = await self._archive.operations.query_conversations(spec)
            return _ids_from_conversations(conversations)
        if query_case.provider is not None:
            conversations = await self._archive.list_conversations(provider=query_case.provider, limit=None)
            return _ids_from_conversations(conversations)
        raise ValueError(f"Query case {query_case.name!r} does not define a supported projection")

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        await self._archive.close()


class CLISurface:
    """Click-CLI adapter projection.

    Invokes the same ``polylogue [filters] list --format json`` path that
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
            args.extend(["--provider", query_case.provider])
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
        args.extend(["list", "--format", "json"])
        result = self._runner.invoke(cli, args, catch_exceptions=True)
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise result.exception
        return result.exit_code, result.output

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
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
            raise AssertionError(f"CLI exited 2 but envelope is not no_results: {parsed!r}")
        if exit_code != 0:
            raise AssertionError(f"polylogue list --format json exited {exit_code} for {query_case.name!r}: {output!r}")
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"CLI list output is not JSON ({query_case.name!r}): {output!r}") from exc
        # Three shapes can land here:
        #   * list mode: ``[{"id": ..., ...}, ...]``
        #   * search-hit mode: ``[{"conversation": {"id": ...}, ...}, ...]``
        #   * typed ranked-result envelope (PR #1370):
        #     ``{"hits": [{"conversation": {"id": ...}, "match": {...}}, ...], "limit": ..., ...}``
        # All three project to the same id tuple for parity comparison.
        if isinstance(parsed, dict) and isinstance(parsed.get("hits"), list):
            parsed = parsed["hits"]
        if not isinstance(parsed, list):
            raise AssertionError(f"CLI list output is not a JSON array ({query_case.name!r}): {parsed!r}")
        ids: list[str] = []
        for row in parsed:
            if not isinstance(row, dict):
                continue
            if "id" in row:
                ids.append(str(row["id"]))
            elif "conversation" in row and isinstance(row["conversation"], dict):
                conv = row["conversation"]
                if "id" in conv:
                    ids.append(str(conv["id"]))
        return _sorted_unique(ids)

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        return None


class MCPSurface:
    """MCP server-tool adapter projection.

    Invokes ``list_conversations`` directly on the registered FastMCP tool
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

    async def conversation_facts(self, scenario: ArchiveScenario) -> ConversationFacts:
        raise NotImplementedError("MCPSurface only supports query-level projections")

    async def archive_facts(self) -> ArchiveFacts:
        raise NotImplementedError("MCPSurface only supports query-level projections")

    async def query_ids(self, query_case: ArchiveQueryCase) -> tuple[str, ...]:
        # MCP enforces clamped limits (min 1); use a generous ceiling so the
        # parity matrix is dominated by the case-level expected count rather
        # than transport-level bounds.
        request_limit = query_case.limit if query_case.limit is not None else 1000
        payload_json = await self._tool("list_conversations")(
            limit=request_limit,
            provider=query_case.provider,
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
        return _sorted_unique([str(item["id"]) for item in items])

    async def query_count(self, query_case: ArchiveQueryCase) -> int:
        return len(await self.query_ids(query_case))

    async def close(self) -> None:
        from polylogue.mcp.server_support import _set_runtime_services

        await self._services.close()
        _set_runtime_services(None)


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
    """Build the repo/facade/cli/mcp adapter parity set.

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
    )
    return ArchiveSurfaceSet(surfaces=surfaces)


def _provider_counts(conversations: list[ConversationFacts]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for conversation in conversations:
        counts[conversation.provider] = counts.get(conversation.provider, 0) + 1
    return counts


def _current_archive_scenarios(conn: sqlite3.Connection) -> tuple[ArchiveScenario, ...]:
    rows = conn.execute("SELECT conversation_id FROM conversations ORDER BY conversation_id").fetchall()
    return tuple(ArchiveScenario(name=str(row["conversation_id"])) for row in rows)


__all__ = [
    "ArchiveSurfaceAdapter",
    "ArchiveSurfaceSet",
    "CLISurface",
    "FacadeSurface",
    "MCPSurface",
    "RepositorySurface",
    "SQLiteHydratedSurface",
    "SQLiteRecordSurface",
    "build_adapter_surface_set",
    "build_archive_surface_set",
]
