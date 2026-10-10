"""Scoped discovery stops only after proving its continuation row."""

import sqlite3
from pathlib import Path

import pytest

from polylogue.daemon.derivation import DerivationFrame
from polylogue.operations.session_profile_convergence import make_session_profile_frame
from polylogue.storage.derived.session.derivation import SessionProfileDerivation
from polylogue.storage.derived.session.summary import SessionSummaryDerivation
from polylogue.storage.derived.session.usage_rollup import SessionUsageRollupDerivation
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION


@pytest.mark.parametrize("route", ["profile", "profile-demand", "summary-demand", "usage-demand"])
def test_dense_scoped_page_queries_only_its_first_chunk(tmp_path: Path, route: str) -> None:
    keys = tuple(f"s{i:04}" for i in range(1001))
    adapter, frame, queries = _adapter(tmp_path, route, list(reversed(keys)) + [keys[0]], keys, keys)
    assert adapter.required_page(frame, cursor=None, limit=2) == (("s0000", "s0001"), "s0001")
    assert len(queries) == 1


@pytest.mark.parametrize("route", ["profile", "profile-demand", "summary-demand", "usage-demand"])
def test_scoped_pages_keep_sparse_cursor_and_mutable_scope_semantics(tmp_path: Path, route: str) -> None:
    scope = [f"s{i:04}" for i in reversed(range(1001))] + ["s0500", "absent"]
    adapter, frame, queries = _adapter(
        tmp_path, route, scope, ("s0500", "s0999", "s1000"), ("s0500", "s0999", "s1000", "orphan")
    )
    assert adapter.required_page(frame, cursor="s0499x", limit=2) == (("s0500", "s0999"), "s0999")
    assert len(queries) == 2
    assert adapter.required_page(frame, cursor="s0999", limit=2) == (("s1000",), None)
    assert adapter.required_page(frame, cursor="z", limit=2) == ((), None)
    scope.clear()
    assert adapter.required_page(frame, cursor=None, limit=2) == ((), None)
    scope.extend(["s1000", "s1000", "missing"])
    assert adapter.required_page(frame, cursor=None, limit=2) == (("s1000",), None)


@pytest.mark.parametrize("route", ["summary", "usage"])
def test_non_demand_scopes_keep_absent_keys_for_retirement(tmp_path: Path, route: str) -> None:
    adapter, frame, _ = _adapter(tmp_path, route, ["present", "absent", "absent"], ("present",), ())
    assert adapter.required_page(frame, cursor=None, limit=1) == (("absent",), "absent")
    assert adapter.required_page(frame, cursor="absent", limit=1) == (("present",), None)


def _adapter(
    tmp_path: Path, route: str, scope: list[str], present: tuple[str, ...], demand: tuple[str, ...]
) -> tuple[
    SessionProfileDerivation | SessionSummaryDerivation | SessionUsageRollupDerivation, DerivationFrame, list[str]
]:
    path = tmp_path / "keys.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE session_profile_demand(session_id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO sessions VALUES (?)", ((key,) for key in present))
        conn.executemany("INSERT INTO session_profile_demand VALUES (?)", ((key,) for key in demand))
    queries: list[str] = []

    def read() -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.set_trace_callback(lambda sql: queries.append(sql) if sql.startswith("SELECT") else None)
        return conn

    def session_scope(frame: object) -> list[str]:
        return scope

    if route.startswith("profile"):
        adapter: SessionProfileDerivation | SessionSummaryDerivation | SessionUsageRollupDerivation = (
            SessionProfileDerivation(
                read,
                read,
                materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
                session_scope=session_scope,
                archive_root=tmp_path,
            )
        )
    elif route.startswith("summary"):
        adapter = SessionSummaryDerivation(read, read, session_scope=session_scope, archive_root=tmp_path)
    else:
        adapter = SessionUsageRollupDerivation(read, read, session_scope=session_scope, archive_root=tmp_path)
    return (
        adapter,
        DerivationFrame(
            archive_root=str(tmp_path), source_revision="neutral", profile_demand_only=route.endswith("-demand")
        ),
        queries,
    )


@pytest.mark.parametrize("route", ["profile", "profile-demand", "summary-demand", "usage-demand"])
def test_scoped_page_can_fill_across_chunk_boundaries(tmp_path: Path, route: str) -> None:
    keys = tuple(f"s{i:04}" for i in range(1001))
    adapter, frame, queries = _adapter(tmp_path, route, list(reversed(keys)), keys, keys)
    assert adapter.required_page(frame, cursor=None, limit=501) == (keys[:501], keys[500])
    assert len(queries) == 2
    assert adapter.required_page(frame, cursor=keys[500], limit=501) == (keys[501:], None)


def test_profile_frame_copies_a_sorted_unique_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.operations.session_profile_convergence.resolve_active_index_path", lambda root: tmp_path / "index.db"
    )
    scope = ["z", "a", "z"]
    frame = make_session_profile_frame(tmp_path / "index.db", archive_root=tmp_path, scope=scope)
    scope.append("changed")
    assert frame.scope == ("a", "z")
