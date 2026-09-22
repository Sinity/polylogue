"""Relation-scoped continuation frames: invalidate on dependency, not on noise.

The defect these tests pin: one archive-wide epoch, bumped by every trigger
on every tracked relation, meant the daemon's own session-profile sweep
turned the next page of an unrelated ``/api/query-units`` read into a
``query_continuation_stale`` 409.

Anti-vacuity runs in both directions, because an epoch scheme that never
invalidates is worse than one that over-invalidates:

* ``test_unrelated_write_keeps_continuation_valid`` goes red if the frame is
  restored to a single archive-wide counter (or if any trigger is left
  bumping a shared row).
* ``test_read_relation_write_invalidates`` and
  ``test_tag_write_invalidates_tag_scoped_read`` go red if scoping degenerates
  into a frame that drops the components a page actually depends on.
* ``test_declared_read_set_covers_traced_sql`` goes red if a lowering gains a
  join onto a tracked relation that ``frame_scope`` has not declared --
  the under-declaration that would let a resume skip or duplicate rows.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from polylogue.archive.query.expression import parse_unit_source_expression
from polylogue.archive.query.frame_scope import (
    _SESSION_FILTER_RELATIONS,
    _UNIT_RELATIONS,
    query_unit_frame_relations,
)
from polylogue.archive.query.transaction import (
    QueryContinuationStaleError,
    decode_query_units_continuation,
)
from polylogue.archive.query.unit_results import QueryUnitRequest, query_unit_envelope
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL
from polylogue.storage.sqlite.archive_tiers.query_unit_frame import (
    ALL_FRAME_RELATIONS,
    INDEX_FRAME_RELATIONS,
)

_UNSCOPED_EXPRESSION = "messages where role:user"
_TAG_SCOPED_EXPRESSION = "messages where role:user and session.tag:pinned"


def _seed(root: Path) -> None:
    """Three sessions with one message and one block each, plus satellites."""
    with ArchiveStore(root) as archive:
        archive.close()
    with sqlite3.connect(root / "index.db") as conn:
        for index in range(3):
            session_id = f"codex-session:s{index}"
            conn.execute(
                "INSERT INTO sessions(native_id, origin, title, content_hash) "
                "VALUES (?, 'codex-session', ?, zeroblob(32))",
                (f"s{index}", f"title {index}"),
            )
            conn.execute(
                "INSERT INTO messages(session_id, native_id, position, role, content_hash) "
                "VALUES (?, ?, 0, 'user', zeroblob(32))",
                (session_id, f"m{index}"),
            )
            conn.execute(
                "INSERT INTO blocks(message_id, session_id, position, block_type, text) "
                "VALUES (?, ?, 0, 'text', 'body')",
                (f"{session_id}:n:m{index}", session_id),
            )
            conn.execute(
                "INSERT INTO session_tags(session_id, tag, tag_source) VALUES (?, 'pinned', 'user')",
                (session_id,),
            )
            conn.execute(
                "INSERT INTO session_profiles(session_id, first_message_at) VALUES (?, '1')",
                (session_id,),
            )


def _page(
    root: Path,
    expression: str,
    *,
    offset: int = 0,
    transaction_request: object | None = None,
    session_filters: Mapping[str, object] | None = None,
    trace: list[str] | None = None,
) -> object:
    source = parse_unit_source_expression(expression)
    assert source is not None, expression
    with ArchiveStore.open_existing(root) as archive:
        archive.begin_read_snapshot()
        if trace is not None:
            archive._conn.set_trace_callback(trace.append)
        try:
            return query_unit_envelope(
                archive,
                QueryUnitRequest(
                    expression=expression,
                    source=source,
                    limit=1,
                    offset=offset,
                    session_filters=dict(session_filters) if session_filters else None,
                ),
                transaction_request=transaction_request,  # type: ignore[arg-type]
            )
        finally:
            if trace is not None:
                archive._conn.set_trace_callback(None)
            archive.end_read_snapshot()


def _resume(root: Path, expression: str, token: str) -> object:
    continuation = decode_query_units_continuation(token)
    return _page(
        root,
        expression,
        offset=continuation.request.offset,
        transaction_request=continuation.request,
    )


def _write_session_profile(root: Path, session_id: str) -> None:
    """The session-profile derivation's DELETE+INSERT, verbatim in shape."""
    with sqlite3.connect(root / "index.db") as conn:
        conn.execute("DELETE FROM session_profiles WHERE session_id = ?", (session_id,))
        conn.execute(
            "INSERT INTO session_profiles(session_id, first_message_at) VALUES (?, '2')",
            (session_id,),
        )


def test_unrelated_write_keeps_continuation_valid(tmp_path: Path) -> None:
    """A session-profile sweep must not 409 a page that never read profiles.

    This is the live read-availability defect: under ``polylogued run`` the
    periodic convergence check rewrites every ``session_profiles`` row, and a
    single archive-wide epoch made that invalidate every outstanding
    continuation. Restore the global counter and this test fails with
    ``QueryContinuationStaleError``.
    """
    _seed(tmp_path)
    first = _page(tmp_path, _UNSCOPED_EXPRESSION)
    token = first.continuation  # type: ignore[attr-defined]
    assert token

    _write_session_profile(tmp_path, "codex-session:s2")

    second = _resume(tmp_path, _UNSCOPED_EXPRESSION, token)
    assert len(second.items) == 1  # type: ignore[attr-defined]


def test_read_relation_write_invalidates(tmp_path: Path) -> None:
    """A write to a relation the page *did* read must still invalidate it.

    Without this the scoped frame could degenerate into a no-op that never
    refuses a resume, which is strictly worse than over-invalidating: offset
    paging over a moved relation duplicates or skips rows.
    """
    _seed(tmp_path)
    first = _page(tmp_path, _UNSCOPED_EXPRESSION)
    token = first.continuation  # type: ignore[attr-defined]
    assert token

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            "INSERT INTO messages(session_id, native_id, position, role, content_hash) "
            "VALUES ('codex-session:s0', 'extra', 1, 'user', zeroblob(32))"
        )

    with pytest.raises(QueryContinuationStaleError):
        _resume(tmp_path, _UNSCOPED_EXPRESSION, token)


def test_tag_write_invalidates_tag_scoped_read(tmp_path: Path) -> None:
    """``session_tags`` is scoped, not ignored: a tag page depends on it."""
    _seed(tmp_path)
    first = _page(tmp_path, _TAG_SCOPED_EXPRESSION)
    token = first.continuation  # type: ignore[attr-defined]
    assert token

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM session_tags WHERE session_id = 'codex-session:s2'")

    with pytest.raises(QueryContinuationStaleError):
        _resume(tmp_path, _TAG_SCOPED_EXPRESSION, token)


def test_tag_write_keeps_untagged_read_valid(tmp_path: Path) -> None:
    """The same tag write leaves a page that declared no tag scope alone."""
    _seed(tmp_path)
    first = _page(tmp_path, _UNSCOPED_EXPRESSION)
    token = first.continuation  # type: ignore[attr-defined]
    assert token

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DELETE FROM session_tags WHERE session_id = 'codex-session:s2'")

    second = _resume(tmp_path, _UNSCOPED_EXPRESSION, token)
    assert len(second.items) == 1  # type: ignore[attr-defined]


def _trigger_bodies() -> dict[str, list[str]]:
    """Map each ``query_unit_frame_*`` trigger name to its statement lines."""
    bodies: dict[str, list[str]] = {}
    current: str | None = None
    for line in INDEX_DDL.splitlines():
        stripped = line.strip()
        match = re.match(r"CREATE TRIGGER IF NOT EXISTS (query_unit_frame_\S+)", stripped)
        if match:
            current = match.group(1)
            bodies[current] = []
            continue
        if current is not None:
            if stripped.startswith("END;"):
                current = None
            elif stripped and not stripped.startswith(("AFTER ", "BEFORE ", "FOR EACH ", "WHEN ")):
                bodies[current].append(stripped)
    return bodies


def test_every_frame_trigger_bumps_its_own_relation() -> None:
    """No trigger may be left advancing a shared or foreign counter.

    One trigger still writing ``WHERE singleton = 1`` -- or pointing at
    another relation's row -- silently reintroduces the global epoch for
    every read that depends on it, which no paging test would notice.
    """
    bodies = _trigger_bodies()
    assert bodies, "index DDL declares no query-unit frame triggers"
    for relation in INDEX_FRAME_RELATIONS:
        for event in ("insert", "update", "delete"):
            name = f"query_unit_frame_{relation}_{event}"
            assert name in bodies, f"missing frame trigger {name}"
            assert bodies[name] == [
                f"UPDATE query_unit_frame_state SET epoch = epoch + 1 WHERE relation = '{relation}';"
            ], f"{name} does not advance only its own relation"
    expected = {
        f"query_unit_frame_{relation}_{event}"
        for relation in INDEX_FRAME_RELATIONS
        for event in ("insert", "update", "delete")
    }
    assert set(bodies) == expected


def test_frame_table_seeds_a_row_per_tracked_relation(tmp_path: Path) -> None:
    """Every tracked index relation must have a row to advance."""
    _seed(tmp_path)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        rows = {str(relation) for (relation,) in conn.execute("SELECT relation FROM query_unit_frame_state")}
    assert rows == set(INDEX_FRAME_RELATIONS)


def _view_relations(root: Path) -> dict[str, frozenset[str]]:
    """Tracked relations behind each index-tier view, transitively."""
    with sqlite3.connect(root / "index.db") as conn:
        definitions = {
            str(name): str(sql or "")
            for name, sql in conn.execute("SELECT name, sql FROM sqlite_schema WHERE type = 'view'")
        }
    resolved = {
        name: frozenset(r for r in ALL_FRAME_RELATIONS if re.search(rf"\b{r}\b", sql))
        for name, sql in definitions.items()
    }
    for _ in range(len(definitions)):
        for name, sql in definitions.items():
            for other, relations in list(resolved.items()):
                if other != name and re.search(rf"\b{other}\b", sql):
                    resolved[name] = resolved[name] | relations
    return resolved


def _traced_relations(statements: list[str], views: Mapping[str, frozenset[str]]) -> frozenset[str]:
    sql = "\n".join(statements)
    observed = {relation for relation in ALL_FRAME_RELATIONS if re.search(rf"\b{relation}\b", sql)}
    for view, relations in views.items():
        if re.search(rf"\b{view}\b", sql):
            observed |= relations
    return frozenset(observed)


def _filter_probe_value(keyword: str) -> object:
    if keyword in {"has_tool_use", "has_thinking", "has_paste", "typed_only", "root"}:
        return True
    if keyword in {"min_messages", "min_words", "since_ms"}:
        return 1
    if keyword in {"max_messages", "max_words", "until_ms"}:
        return 10**12
    if keyword in {"origin", "cwd_prefix", "title", "message_type"}:
        return {"origin": "codex-session", "cwd_prefix": "/x", "title": "t", "message_type": "message"}[keyword]
    if keyword == "action_sequence":
        return ["file_edit", "file_edit"]
    return ["pinned"] if keyword in {"tags", "excluded_tags"} else ["probe"]


def _unit_expressions() -> Iterator[tuple[str, str]]:
    """One parseable terminal expression per declared query unit."""
    from polylogue.archive.query.metadata import (
        STRUCTURAL_QUERY_UNIT_REGISTRY,
        query_unit_descriptors,
    )

    for descriptor in query_unit_descriptors():
        info = STRUCTURAL_QUERY_UNIT_REGISTRY.get(descriptor.unit)
        if info is None:
            continue
        for field in info.fields:
            if "." in field.name:
                continue
            expression = f"{descriptor.plural_source} where {field.example}"
            try:
                if parse_unit_source_expression(expression) is None:
                    continue
            except Exception:
                continue
            yield descriptor.unit, expression
            break


def test_declared_read_set_covers_traced_sql(tmp_path: Path) -> None:
    """Every tracked relation a real page touches must be declared.

    Runs one page per declared session-scope keyword and per declared query
    unit under a SQLite trace callback, expands index-tier views down to their
    base relations, and fails if the SQL reached a tracked relation that
    ``query_unit_frame_relations`` left out. Under-declaration is the failure
    that would let a resume page over a relation that moved.
    """
    _seed(tmp_path)
    views = _view_relations(tmp_path)

    cases: list[tuple[str, str, Mapping[str, object]]] = [
        (f"filter:{keyword}", _UNSCOPED_EXPRESSION, {keyword: _filter_probe_value(keyword)})
        for keyword in sorted(_SESSION_FILTER_RELATIONS)
    ]
    covered_units: set[str] = set()
    for unit, expression in _unit_expressions():
        covered_units.add(unit)
        cases.append((f"unit:{unit}", expression, {}))

    assert covered_units == set(_UNIT_RELATIONS), (
        "every declared query unit needs a traced case; "
        f"missing={sorted(set(_UNIT_RELATIONS) - covered_units)} "
        f"undeclared={sorted(covered_units - set(_UNIT_RELATIONS))}"
    )

    failures: list[str] = []
    for label, expression, filters in cases:
        source = parse_unit_source_expression(expression)
        assert source is not None
        declared = query_unit_frame_relations(source, filters)
        statements: list[str] = []
        try:
            _page(tmp_path, expression, session_filters=filters, trace=statements)
        except Exception as exc:
            failures.append(f"{label}: page failed: {type(exc).__name__}: {exc}")
            continue
        observed = _traced_relations(statements, views)
        undeclared = observed - declared
        if undeclared:
            failures.append(f"{label}: SQL read {sorted(undeclared)} outside declared {sorted(declared)}")
    assert not failures, "\n".join(failures)
