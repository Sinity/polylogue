"""Invalid optional numeric evidence cannot drop authored provider content."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.pipeline.ids import session_content_hash
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.index_writer import write_fixture_index_session


def _persist(root: Path, session: ParsedSession) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    with write_lease("optional numeric parser fixture", archive_root=root):
        initialize_active_archive_root(root)
        with closing(open_connection(root / "index.db", tier=ArchiveTier.INDEX, archive_root=root)) as conn:
            conn.row_factory = sqlite3.Row
            with conn:
                session_id = write_fixture_index_session(conn, session, content_hash=session_content_hash(session))
    return root / "index.db", session_id


@pytest.mark.parametrize("provider", ["gemini-cli", "drive", "chatgpt"])
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (True, None),
        (-1, None),
        (0, 0),
        (7, 7),
        (7.0, 7),
        (0.5, None),
        (float("inf"), None),
        (float("nan"), None),
    ],
)
def test_optional_counts_and_duration_preserve_prose_in_storage(
    tmp_path: Path, provider: str, value: JSONValue, expected: int | None
) -> None:
    if provider == "gemini-cli":
        payload: JSONDocument = {
            "sessionId": "numeric-fixture",
            "projectHash": "neutral-project",
            "kind": "chat",
            "messages": [
                {
                    "id": "answer",
                    "type": "gemini",
                    "content": "Neutral authored answer",
                    "tokens": {"input": value, "output": 3},
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            ],
        }
        [session] = parse_payload("gemini-cli", payload, "fallback")
        assert session.messages[0].input_tokens == expected
        column = "input_tokens"
    elif provider == "drive":
        session = parse_chunked_prompt(
            "gemini",
            {
                "id": "numeric-fixture",
                "chunkedPrompt": {
                    "chunks": [
                        {
                            "id": "answer",
                            "role": "model",
                            "text": "Neutral authored answer",
                            "tokenCount": value,
                            "createTime": "2026-01-01T00:00:00Z",
                        }
                    ]
                },
            },
            "fallback",
        )
        assert session.messages[0].output_tokens == expected
        column = "output_tokens"
    else:
        session = chatgpt_parse(
            {
                "id": "numeric-fixture",
                "mapping": {
                    "answer-node": {
                        "parent": None,
                        "children": [],
                        "message": {
                            "id": "answer",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Neutral authored answer"]},
                            "metadata": {"durationMs": value},
                        },
                    }
                },
                "current_node": "answer-node",
            },
            "fallback",
        )
        assert session.messages[0].duration_ms == expected
        column = "duration_ms"
    path, session_id = _persist(tmp_path / "archive", session)
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(
            f"SELECT b.text,m.{column} FROM messages AS m JOIN blocks AS b ON b.message_id=m.message_id "
            "WHERE m.session_id=?",
            (session_id,),
        ).fetchall()
    assert rows == [("Neutral authored answer", expected)]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("7", 7),
        ("7.0", 7),
        ("7e0", 7),
        ("0.5", None),
        ("1.0000000000000001", None),
        ("1e309", None),
        ("Infinity", None),
        ("NaN", None),
        ("9007199254740993", 9007199254740993),
    ],
)
@pytest.mark.parametrize("provider", ["gemini-cli", "drive"])
def test_numeric_string_counts_keep_exact_integral_value(provider: str, value: str, expected: int | None) -> None:
    if provider == "gemini-cli":
        [session] = parse_payload(
            "gemini-cli",
            {
                "sessionId": "numeric-string",
                "projectHash": "neutral-project",
                "kind": "chat",
                "messages": [
                    {"id": "answer", "type": "gemini", "content": "Neutral answer", "tokens": {"input": value}}
                ],
            },
            "fallback",
        )
        assert session.messages[0].input_tokens == expected
    else:
        session = parse_chunked_prompt(
            "gemini",
            {
                "id": "numeric-string",
                "chunkedPrompt": {
                    "chunks": [{"id": "answer", "role": "model", "text": "Neutral answer", "tokenCount": value}]
                },
            },
            "fallback",
        )
        assert session.messages[0].output_tokens == expected
    assert session.messages[0].text == "Neutral answer"


@pytest.mark.parametrize(
    "value,code,outcome",
    [
        (None, None, "unknown"),
        (True, None, "unknown"),
        (0.5, None, "unknown"),
        (float("inf"), None, "unknown"),
        (float("nan"), None, "unknown"),
        (0, 0, "ok"),
        (0.0, 0, "ok"),
        (-1, -1, "error"),
        (7, 7, "error"),
    ],
)
def test_drive_execution_optional_exit_code_keeps_unknown_and_output(
    tmp_path: Path, value: JSONValue, code: int | None, outcome: str
) -> None:
    session = parse_chunked_prompt(
        "gemini",
        {
            "id": "execution-fixture",
            "chunkedPrompt": {
                "chunks": [
                    {
                        "id": "result",
                        "role": "model",
                        "codeExecutionResult": {"output": "Neutral tool output", "exitCode": value},
                    }
                ]
            },
        },
        "fallback",
    )
    result = session.messages[0].blocks[0]
    assert result.exit_code == code
    if isinstance(value, float) and not math.isfinite(value):
        assert result.metadata is not None and "exitCode" in result.metadata
        assert result.metadata["exitCode"] is None
    assert result.is_error is (None if outcome == "unknown" else outcome == "error")
    path, session_id = _persist(tmp_path / "archive", session)
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            "SELECT text,tool_result_exit_code,tool_outcome FROM blocks WHERE session_id=?", (session_id,)
        ).fetchone()
    assert row == ("Neutral tool output", code, outcome)


@pytest.mark.parametrize(
    "value,code,outcome",
    [
        (0.5, None, "unknown"),
        (float("inf"), None, "unknown"),
        (float("nan"), None, "unknown"),
        (True, None, "unknown"),
        (0, 0, "ok"),
        (7.0, 7, "error"),
        (-1, -1, "error"),
    ],
)
def test_antigravity_exit_code_does_not_truncate_into_success(
    tmp_path: Path, value: JSONValue, code: int | None, outcome: str
) -> None:
    from tests.infra.antigravity_parser import parse_trajectory_db

    source = tmp_path / "trajectory.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.executescript(
            "CREATE TABLE trajectory_meta(trajectory_id TEXT,cascade_id TEXT); "
            "INSERT INTO trajectory_meta VALUES('trajectory','cascade'); "
            "CREATE TABLE steps(idx INTEGER,step_type TEXT,step_format TEXT,step_payload TEXT);"
        )
        conn.execute(
            "INSERT INTO steps VALUES(0,'tool_result','v1',?)",
            (json.dumps({"tool_name": "shell", "output": "Neutral output", "exit_code": value}),),
        )
        conn.commit()
    [session] = list(parse_trajectory_db(source))
    path, session_id = _persist(tmp_path / "archive", session)
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            "SELECT text,tool_result_exit_code,tool_outcome FROM blocks WHERE session_id=?", (session_id,)
        ).fetchone()
    assert row == ("Neutral output", code, outcome)


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), (float("inf"), None), (float("nan"), None), (0.5, None), (-1, None), (0, 0), (7.0, 7)],
)
def test_hermes_sqlite_optional_numeric_fields_preserve_stored_answer(
    tmp_path: Path, value: float | int | None, expected: int | None
) -> None:
    from polylogue.sources.parsers.hermes_state import parse_state_db

    source = tmp_path / "state.db"
    with closing(sqlite3.connect(source)) as conn:
        conn.executescript(
            "CREATE TABLE schema_version(version INTEGER); INSERT INTO schema_version VALUES(16); "
            "CREATE TABLE sessions(id TEXT PRIMARY KEY,source TEXT,model TEXT,started_at REAL,ended_at REAL,"
            "actual_cost_usd REAL,input_tokens INTEGER,title TEXT,model_config TEXT,parent_session_id TEXT); "
            "CREATE TABLE messages(id INTEGER PRIMARY KEY,session_id TEXT,role TEXT,content TEXT,timestamp REAL,token_count INTEGER,tool_calls TEXT,observed INTEGER,active INTEGER,compacted INTEGER);"
        )
        conn.execute(
            "INSERT INTO sessions VALUES('numeric','hermes','neutral-model',?,?,?,?,'Neutral title','{}',NULL)",
            (value, value, value, value),
        )
        conn.execute(
            "INSERT INTO messages VALUES(1,'numeric','assistant','Neutral authored answer',?,?,NULL,0,1,0)",
            (value, value),
        )
        conn.commit()
    [session] = parse_state_db(source)
    assert session.messages[0].output_tokens == expected
    expected_cost = value if value is not None and math.isfinite(value) and value >= 0 else None
    assert session.reported_cost_usd == expected_cost
    [usage] = [event for event in session.session_events if event.event_type == "token_count"]
    assert usage.payload["actual_cost_usd"] == expected_cost
    path, session_id = _persist(tmp_path / "archive", session)
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute(
            "SELECT b.text,m.output_tokens FROM messages AS m JOIN blocks AS b "
            "ON b.message_id=m.message_id WHERE m.session_id=?",
            (session_id,),
        ).fetchone()
        costs = conn.execute(
            "SELECT provider_cost_usd FROM session_model_usage WHERE session_id=? AND model_name='neutral-model'",
            (session_id,),
        ).fetchall()
    assert row == ("Neutral authored answer", expected)
    assert costs == [(expected_cost,)]


@pytest.mark.parametrize(
    "value,expected", [(None, None), (float("inf"), None), (0.5, None), (0, 0), (7.0, 7), (-1, -1)]
)
def test_codex_sqlite_optional_integer_stays_exact_through_logical_export(
    tmp_path: Path, value: float | int | None, expected: int | None
) -> None:
    from polylogue.sources.parsers.codex_state import iter_codex_state_parts
    from polylogue.sources.sqlite_export import write_logical_export

    source = tmp_path / "goals_1.sqlite"
    with closing(sqlite3.connect(source)) as conn:
        conn.executescript(
            "CREATE TABLE thread_goals(thread_id TEXT PRIMARY KEY,goal_id TEXT,objective TEXT,"
            "status TEXT,token_budget INTEGER,tokens_used INTEGER,time_used_seconds INTEGER,created_at_ms INTEGER,updated_at_ms INTEGER);"
        )
        conn.execute(
            "INSERT INTO thread_goals VALUES('thread','goal','Neutral authored objective','active',?,0,0,0,0)", (value,)
        )
        conn.commit()
    export = tmp_path / "goals.export"
    with export.open("wb") as sink:
        write_logical_export(source, sink)
    for path in (source, export):
        [record] = [part for part in iter_codex_state_parts(path, state_kind="goals") if part.part_kind == "record"]
        assert record.payload["token_budget"] == expected
        assert record.payload["objective"] == "Neutral authored objective"
    with closing(sqlite3.connect(source)) as conn:
        assert conn.execute("SELECT objective,token_budget FROM thread_goals").fetchone() == (
            "Neutral authored objective",
            value,
        )


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), (True, None), (0.5, None), (float("inf"), None), (float("nan"), None), (0, 0), (7.0, 7), (-1, -1)],
)
def test_chatgpt_citation_optional_ranges_preserve_stored_evidence(
    tmp_path: Path, value: JSONValue, expected: int | None
) -> None:
    session = chatgpt_parse(
        {
            "id": "citation-numeric-fixture",
            "mapping": {
                "answer-node": {
                    "parent": None,
                    "children": [],
                    "message": {
                        "id": "answer",
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Neutral cited answer"]},
                        "metadata": {
                            "citations": [
                                {
                                    "start_ix": value,
                                    "end_ix": value,
                                    "metadata": {"title": "Neutral brief", "url": "https://example.test/brief"},
                                }
                            ]
                        },
                    },
                }
            },
            "current_node": "answer-node",
        },
        "fallback",
    )
    path, session_id = _persist(tmp_path / "archive", session)
    with closing(sqlite3.connect(path)) as conn:
        [(text, extras)] = conn.execute(
            "SELECT b.text,b.semantic_extra_json FROM blocks AS b JOIN messages AS m "
            "ON m.message_id=b.message_id WHERE m.session_id=?",
            (session_id,),
        ).fetchall()
    assert text == "Neutral cited answer"
    [citation] = json.loads(extras)["web_constructs"]
    assert citation["title"] == "Neutral brief"
    assert citation["url"] == "https://example.test/brief"
    assert citation["start_index"] == expected
    assert citation["end_index"] == expected
