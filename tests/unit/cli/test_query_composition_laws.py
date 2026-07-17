"""Real-route query algebra and relational-cardinality survivor laws."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from polylogue.archive.query.expression import compile_expression, parse_unit_source_expression
from polylogue.archive.query.unit_results import query_unit_rows
from polylogue.cli import cli
from polylogue.config import Source
from polylogue.pipeline.services.archive_ingest import parse_sources_archive
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import (
    ActionQueryRowPayload,
    QueryUnitAggregateEnvelope,
    QueryUnitEnvelope,
)
from tests.infra.query_manifest_oracle import (
    QUERY_CARDINALITY_TOKEN,
    ActionIdentity,
    QueryCardinalityManifest,
    query_cardinality_manifest,
)
from tests.infra.workload_artifacts import build_seeded_archive, clone_seeded_archive

_ACTION_EXPRESSION = f"actions where session.origin:codex-session AND command:{QUERY_CARDINALITY_TOKEN}"
_SESSION_EXPRESSION = f"exists action(session.origin:codex-session AND command:{QUERY_CARDINALITY_TOKEN})"


@dataclass(frozen=True)
class _PreparedArchive:
    root: Path
    manifest: QueryCardinalityManifest


@contextmanager
def _archive_environment(root: Path) -> Iterator[None]:
    updates = {
        "POLYLOGUE_ARCHIVE_ROOT": str(root),
        "POLYLOGUE_INGEST_PARSE_WORKERS": "1",
        "POLYLOGUE_FORCE_PLAIN": "1",
    }
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def query_cardinality_archive(tmp_path_factory: pytest.TempPathFactory) -> _PreparedArchive:
    """Layer independent Codex facts onto the immutable realized C-03 canary."""
    work = tmp_path_factory.mktemp("query-cardinality-survivor")
    artifact = build_seeded_archive(cache_root=work / "seeded-cache")
    clone = clone_seeded_archive(artifact, work / "augmented-archive")
    manifest = query_cardinality_manifest()
    sources = [Source(name="codex", path=path) for path in manifest.write_sources(work / "wire")]
    with _archive_environment(clone.root):
        result = asyncio.run(parse_sources_archive(clone.root, sources))
    assert result.parse_failures == 0
    assert result.processed_ids == set(manifest.matching_session_ids()) | {manifest.decoy_session_id}
    return _PreparedArchive(root=clone.root, manifest=manifest)


def _action_identity(row: Any) -> ActionIdentity:
    return (
        str(row.session_id),
        str(row.tool_command),
        cast(str | None, row.output_text),
        cast(int | None, row.is_error),
        cast(int | None, row.exit_code),
    )


def _payload_action_identity(row: dict[str, object]) -> ActionIdentity:
    return (
        str(row["session_id"]),
        str(row["tool_command"]),
        cast(str | None, row.get("output_text")),
        cast(int | None, row.get("is_error")),
        cast(int | None, row.get("exit_code")),
    )


def _assert_repository_membership(root: Path, manifest: QueryCardinalityManifest) -> None:
    """Assert the action_relation_select_sql -> query_actions production path."""
    source = parse_unit_source_expression(_ACTION_EXPRESSION)
    assert source is not None
    with ArchiveStore.open_existing(root) as archive:
        rows = archive.query_actions(source.predicate, limit=100)
    actual = tuple(_action_identity(row) for row in rows)
    expected = manifest.matching_action_identities()
    assert actual == expected, (
        f"expected {len(expected)} independently planted action rows, got {len(actual)}; "
        f"expected={expected!r}, actual={actual!r}"
    )


def _copy_archive(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _cli_env(root: Path) -> dict[str, str]:
    return {
        "POLYLOGUE_ARCHIVE_ROOT": str(root),
        "POLYLOGUE_FORCE_PLAIN": "1",
    }


def test_query_algebra_cardinality_survives_real_read_and_action_routes(
    query_cardinality_archive: _PreparedArchive,
    tmp_path: Path,
) -> None:
    """Survive parser/lowerer, action relation, repository, CLI, and delete.

    Production dependencies exercised: Codex parser/materializer,
    ``parse_unit_source_expression``/``compile_expression``,
    ``action_relation_select_sql``, ``ArchiveStore.query_actions`` and
    ``query_unit_counts``, ``query_unit_rows``, root CLI ``find``, and the
    public ``find ... then delete`` preview/apply route.  Removing the command
    predicate, losing ordinal result pairing, applying LIMIT before selection,
    or letting preview/apply resolve different sets makes this test fail.
    """
    root = query_cardinality_archive.root
    manifest = query_cardinality_archive.manifest
    expected = manifest.matching_action_identities()

    # Parse the public terminal expression and lower its action-existence form.
    source = parse_unit_source_expression(_ACTION_EXPRESSION)
    assert source is not None
    assert source.unit == "action"
    compiled_selector = compile_expression(_SESSION_EXPRESSION)
    assert compiled_selector.boolean_predicate is not None

    # SQL repository membership and the shared terminal-row envelope agree with
    # the independent provider-wire oracle, including duplicate IDs, a missing
    # result, and an orphan result that must not create an action.
    _assert_repository_membership(root, manifest)
    with ArchiveStore.open_existing(root) as archive:
        envelope = query_unit_rows(archive, source, query=_ACTION_EXPRESSION, limit=100)
    assert isinstance(envelope, QueryUnitEnvelope)
    assert tuple(_action_identity(row) for row in envelope.items) == expected
    assert all(isinstance(row, ActionQueryRowPayload) for row in envelope.items)

    # Exact count and a complete partition must conserve the same row grain.
    count_expression = f"{_ACTION_EXPRESSION} | count"
    count_source = parse_unit_source_expression(count_expression)
    assert count_source is not None
    partition_expression = f"{_ACTION_EXPRESSION} | group by is_error | count"
    partition_source = parse_unit_source_expression(partition_expression)
    assert partition_source is not None
    with ArchiveStore.open_existing(root) as archive:
        count_envelope = query_unit_rows(archive, count_source, query=count_expression, limit=20)
        partition_envelope = query_unit_rows(
            archive,
            partition_source,
            query=partition_expression,
            limit=20,
        )
    assert isinstance(count_envelope, QueryUnitAggregateEnvelope)
    assert [(item.group_key, item.count) for item in count_envelope.items] == [("all", len(expected))]
    assert isinstance(partition_envelope, QueryUnitAggregateEnvelope)
    partition = {str(item.group_key): item.count for item in partition_envelope.items}
    assert partition == manifest.is_error_partition()
    assert sum(partition.values()) == len(expected)

    # Concatenating bounded public pages returns every logical member once and
    # in the same stable order as the independent timestamp-ordered fact set.
    paged: list[ActionIdentity] = []
    offsets: list[int] = []
    offset = 0
    with ArchiveStore.open_existing(root) as archive:
        while True:
            page = query_unit_rows(
                archive,
                source,
                query=_ACTION_EXPRESSION,
                limit=2,
                offset=offset,
            )
            assert isinstance(page, QueryUnitEnvelope)
            offsets.append(offset)
            paged.extend(_action_identity(row) for row in page.items)
            if page.next_offset is None:
                break
            assert page.next_offset == offset + len(page.items)
            offset = page.next_offset
    assert offsets == [0, 2, 4]
    assert tuple(paged) == expected
    assert len(set(paged)) == len(expected)

    # The stable public CLI read route reparses and reexecutes the expression.
    read_result = CliRunner().invoke(
        cli,
        ["--plain", "--format", "json", "find", _ACTION_EXPRESSION],
        env=_cli_env(root),
    )
    assert read_result.exit_code == 0, read_result.output
    read_payload = cast(dict[str, object], json.loads(read_result.output))
    read_items = cast(list[dict[str, object]], read_payload["items"])
    assert tuple(_payload_action_identity(item) for item in read_items) == expected
    assert read_payload["total"] == len(expected)

    # Preview and apply run against a private clone.  The previewed identities
    # equal the planted session set, apply reports the same cardinality, the
    # selected sessions disappear, and the output-only decoy remains.
    mutation_root = _copy_archive(root, tmp_path / "mutation-archive")
    runner = CliRunner()
    preview_result = runner.invoke(
        cli,
        ["--plain", "find", _SESSION_EXPRESSION, "then", "delete", "--dry-run", "--all"],
        env=_cli_env(mutation_root),
    )
    assert preview_result.exit_code == 0, preview_result.output
    preview = cast(dict[str, object], json.loads(preview_result.output))
    preview_ids = tuple(cast(list[str], preview["session_ids"]))
    assert preview["status"] == "preview"
    assert preview["affected_count"] == 0
    assert set(preview_ids) == set(manifest.matching_session_ids())
    assert preview["session_count"] == len(manifest.matching_session_ids())

    apply_result = runner.invoke(
        cli,
        ["--plain", "find", _SESSION_EXPRESSION, "then", "delete", "--yes", "--all"],
        env=_cli_env(mutation_root),
    )
    assert apply_result.exit_code == 0, apply_result.output
    applied = cast(dict[str, object], json.loads(apply_result.output))
    assert applied["session_count"] == preview["session_count"]
    assert applied["affected_count"] == preview["session_count"]

    with ArchiveStore.open_existing(mutation_root) as archive:
        for session_id in preview_ids:
            with pytest.raises(KeyError):
                archive.read_summary(session_id)
        decoy = archive.read_summary(manifest.decoy_session_id)
    assert decoy.session_id == manifest.decoy_session_id

    post_result = runner.invoke(
        cli,
        ["--plain", "--format", "json", "find", _ACTION_EXPRESSION],
        env=_cli_env(mutation_root),
    )
    assert post_result.exit_code == 2, post_result.output
    post_payload = cast(dict[str, object], json.loads(post_result.output))
    assert post_payload["items"] == []
    assert post_payload["total"] == 0


def test_survivor_detects_naive_duplicate_id_join_mutation(
    query_cardinality_archive: _PreparedArchive,
    tmp_path: Path,
) -> None:
    """Dropping ordinal pairing creates seven rows where the oracle requires five.

    The mutation replaces the production ``actions`` view with the historical
    same-session/same-tool-id join, equivalent to removing
    ``result_rank = use_rank`` from ``action_relation_select_sql``.  The two
    duplicate uses and two duplicate results then form a 2x2 product (four
    rows instead of two), and the real-route membership survivor must reject it.
    """
    root = _copy_archive(query_cardinality_archive.root, tmp_path / "naive-join-archive")
    with sqlite3.connect(root / "index.db") as conn:
        conn.executescript(
            """
            DROP VIEW actions;
            CREATE VIEW actions AS
            SELECT
                u.session_id,
                u.message_id,
                u.block_id AS tool_use_block_id,
                u.tool_name,
                u.semantic_type,
                u.tool_command,
                u.tool_path,
                u.tool_input,
                r.text AS output_text,
                r.tool_result_is_error AS is_error,
                r.tool_result_exit_code AS exit_code,
                r.block_id AS tool_result_block_id
            FROM blocks u
            LEFT JOIN blocks r
              ON r.session_id = u.session_id
             AND r.tool_id = u.tool_id
             AND r.block_type = 'tool_result'
            WHERE u.block_type = 'tool_use';
            """
        )

    with pytest.raises(AssertionError) as caught:
        _assert_repository_membership(root, query_cardinality_archive.manifest)
    message = str(caught.value)
    assert "expected 5 independently planted action rows, got 7" in message
    assert QUERY_CARDINALITY_TOKEN in message
