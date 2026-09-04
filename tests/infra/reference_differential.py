"""Generated cross-surface differentials for the query reference model.

The corpus program is executed through the production acquisition and parsing
routes. The reference model receives the program's planted semantic facts, then
evaluates the same parsed AST without calling the query planner or SQL layer.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from polylogue.api import Polylogue
from polylogue.archive.models import Session
from polylogue.archive.query.expression import parse_expression_ast
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.cli.click_app import cli
from polylogue.mcp.server import build_server
from polylogue.mcp.server_support import _set_runtime_services
from polylogue.services import build_runtime_services
from tests.infra.builders import make_conv, make_msg
from tests.infra.corpus_program import Acquire, Converge, CorpusProgram, RawArtifact
from tests.infra.query_manifest_oracle import PlantedCodexCall, PlantedCodexSession
from tests.infra.reference_model import ReferenceArchive, ReferenceResult


@dataclass(frozen=True, slots=True)
class QuerySemanticFacts:
    """Meaningful query output shared by the model and public projections."""

    session_ids: tuple[str, ...]
    count: int

    @classmethod
    def from_reference(cls, result: ReferenceResult) -> QuerySemanticFacts:
        return cls(session_ids=result.session_ids, count=result.count)

    @classmethod
    def from_ids(cls, session_ids: Iterable[str]) -> QuerySemanticFacts:
        ids = tuple(sorted({str(value) for value in session_ids}))
        return cls(session_ids=ids, count=len(ids))


def assert_query_facts_equal(expected: QuerySemanticFacts, actual: QuerySemanticFacts, *, surface: str) -> None:
    """Reject a surface that changes membership or count at session grain."""
    assert actual == expected, f"{surface} query facts disagree: expected={expected!r}, actual={actual!r}"


def codex_query_program_strategy(*, max_sessions: int = 3) -> Any:
    """Generate small, valid Codex programs that exercise production ingest."""
    from hypothesis import strategies as st

    @st.composite
    def _program(draw: Any) -> CorpusProgram:
        session_count = draw(st.integers(min_value=1, max_value=max_sessions))
        operations: list[Acquire | Converge] = []
        for index in range(session_count):
            native_id = f"generated-query-{index}"
            calls = (
                PlantedCodexCall(
                    call_id=f"call-{index}",
                    command=f"printf generated-query-marker-{index}",
                    timestamp=f"2026-01-01T00:00:{index + 1:02d}Z",
                ),
            )
            session = PlantedCodexSession(
                native_session_id=native_id,
                canonical_session_id=f"codex-session:{native_id}",
                timestamp=f"2026-01-01T00:{index:02d}:00Z",
                calls=calls,
                results=(),
            )
            operations.append(
                Acquire(
                    operation_id=f"acquire-{index}",
                    artifact=RawArtifact(
                        artifact_id=f"artifact-{index}",
                        payload=(
                            "".join(
                                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                                for record in session.wire_records()
                            )
                        ).encode("utf-8"),
                        source_name="codex",
                        source_path=f"sources/generated-query-{index}.jsonl",
                        metadata={"session_id": native_id},
                    ),
                )
            )
        operations.append(Converge(operation_id="converge"))
        return CorpusProgram(operations=tuple(operations))

    return _program()


def reference_archive_for_program(program: CorpusProgram) -> ReferenceArchive:
    """Build the model from planted program facts, independently of ingestion."""
    sessions: list[Session] = []
    for operation in program.ordered_operations():
        if not isinstance(operation, Acquire):
            continue
        native_id = operation.artifact.metadata.get("session_id")
        if not isinstance(native_id, str):
            raise AssertionError("generated query artifact has no session_id metadata")
        session_id = f"codex-session:{native_id}"
        sessions.append(
            make_conv(
                id=session_id,
                provider="codex",
                messages=(
                    make_msg(id=f"{session_id}:user", role="user", text=f"planted request for {native_id}"),
                    make_msg(id=f"{session_id}:assistant", role="assistant", text=f"planted reply for {native_id}"),
                ),
            )
        )
    return ReferenceArchive.from_sessions(sessions)


def query_api(archive_root: Path, expression: str) -> QuerySemanticFacts:
    """Run a parsed expression through the public Python API."""
    spec = SessionQuerySpec.from_expression(expression)

    async def read() -> list[Session]:
        async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as archive:
            return await archive.list_sessions_for_spec(spec)

    sessions = asyncio.run(read())
    return QuerySemanticFacts.from_ids(str(session.id) for session in sessions)


def _ids_from_payload(payload: object) -> tuple[str, ...]:
    if isinstance(payload, Mapping):
        items = payload.get("items")
        if not isinstance(items, list):
            items = payload.get("hits")
    else:
        items = payload
    if not isinstance(items, list):
        raise AssertionError(f"query projection has no item list: {payload!r}")
    ids: list[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("id"), str):
            ids.append(item["id"])
        session = item.get("session")
        if isinstance(session, Mapping) and isinstance(session.get("id"), str):
            ids.append(session["id"])
    return tuple(sorted(set(ids)))


def query_cli(archive_root: Path, expression: str) -> QuerySemanticFacts:
    """Run the operator CLI query-result read route and normalize its output."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--plain",
            "--no-daemon",
            "find",
            expression,
            "then",
            "read",
            "--all",
            "--format",
            "json",
        ],
        env={"POLYLOGUE_ARCHIVE_ROOT": str(archive_root)},
    )
    payload = json.loads(result.output)
    facts = QuerySemanticFacts.from_ids(_ids_from_payload(payload))
    valid_no_match = (
        result.exit_code == 2
        and facts.count == 0
        and isinstance(payload, Mapping)
        and payload.get("mode") == "search"
        and payload.get("total") == 0
    )
    if result.exit_code != 0 and not valid_no_match:
        raise AssertionError(f"CLI query failed with {result.exit_code}: {result.output}")
    return facts


def query_mcp(archive_root: Path, expression: str) -> QuerySemanticFacts:
    """Run the registered MCP query tool and normalize its session projection."""

    async def read() -> QuerySemanticFacts:
        from polylogue.config import Config

        services = build_runtime_services(
            config=Config(
                archive_root=archive_root,
                render_root=archive_root / "render",
                sources=[],
                db_path=archive_root / "index.db",
            )
        )
        _set_runtime_services(services)
        try:
            server = build_server(services=services)
            payload = await server._tool_manager._tools["query"].fn(
                expression=expression,
                projection="sessions",
                limit=10_000,
            )
            return QuerySemanticFacts.from_ids(_ids_from_payload(json.loads(payload)))
        finally:
            await services.close()
            _set_runtime_services(None)

    return asyncio.run(read())


def differential_for_program(
    archive_root: Path, program: CorpusProgram, expression: str
) -> tuple[QuerySemanticFacts, ...]:
    """Return model, API, CLI, and MCP facts for one expression."""
    model = QuerySemanticFacts.from_reference(
        reference_archive_for_program(program).query(parse_expression_ast(expression))
    )
    return (
        model,
        query_api(archive_root, expression),
        query_cli(archive_root, expression),
        query_mcp(archive_root, expression),
    )


__all__ = [
    "QuerySemanticFacts",
    "assert_query_facts_equal",
    "codex_query_program_strategy",
    "differential_for_program",
    "reference_archive_for_program",
    "query_api",
    "query_cli",
    "query_mcp",
]
