from __future__ import annotations

from polylogue.core.tool_identity import extract_mcp_server, parse_mcp_tool_name


def test_mcp_identity_preserves_server_and_server_local_tool_name() -> None:
    identity = parse_mcp_tool_name("mcp__github__search__issues")

    assert identity is not None
    assert identity.server == "github"
    assert identity.tool == "search__issues"
    assert extract_mcp_server(identity.raw_name) == "github"


def test_mcp_identity_rejects_names_without_both_structural_segments() -> None:
    assert parse_mcp_tool_name("mcp__github") is None
    assert parse_mcp_tool_name("mcp____tool") is None
    assert parse_mcp_tool_name("mcp__github__") is None
    assert parse_mcp_tool_name("search") is None


def test_a_lone_surrogate_projection_stays_readable() -> None:
    """The exact JSON keeps a lone-surrogate escape; its projection stays UTF-8.

    Anti-vacuity: project with a bare ``json_extract`` and reading the
    generated column raises ``Could not decode to UTF-8``.
    """
    import sqlite3

    from polylogue.core.tool_identity import sql_coalesced_json_extract

    projection = sql_coalesced_json_extract("tool_input", ("command", "cmd"))
    connection = sqlite3.connect(":memory:")
    connection.execute(f"CREATE TABLE t (tool_input TEXT, tool_command TEXT GENERATED ALWAYS AS ({projection}))")
    connection.execute("INSERT INTO t (tool_input) VALUES (?)", ('{"cmd":"ls \\ud800 x"}',))
    connection.execute("INSERT INTO t (tool_input) VALUES (?)", ('{"command":"ls é"}',))
    assert [row[0] for row in connection.execute("SELECT tool_command FROM t ORDER BY rowid")] == [
        "ls \\ud800 x",
        "ls é",
    ]


def test_a_lone_surrogate_projection_decodes_ordinary_escapes() -> None:
    """Only the surrogate stays escaped; quotes, newlines and backslashes decode.

    Anti-vacuity: project the raw JSON spelling again and the command keeps
    ``\\"``, ``\\n`` and doubled backslashes.
    """
    import sqlite3

    from polylogue.core.tool_identity import sql_coalesced_json_extract

    projection = sql_coalesced_json_extract("tool_input", ("command",))
    connection = sqlite3.connect(":memory:")
    connection.execute(f"CREATE TABLE t (tool_input TEXT, tool_command TEXT GENERATED ALWAYS AS ({projection}))")
    stored = '{"command":"echo \\"a\\"\\n\\\\path \\\\ud800 \\ud800"}'
    connection.execute("INSERT INTO t (tool_input) VALUES (?)", (stored,))
    (command,) = connection.execute("SELECT tool_command FROM t").fetchone()
    assert command == 'echo "a"\n\\path \\ud800 \\ud800'


def test_a_lone_surrogate_projection_keeps_noncharacters_and_decodes_controls() -> None:
    """A real U+FFFF survives and ``\\u00XX`` control escapes decode.

    Anti-vacuity: restore the U+FFFF backslash placeholder and the provider's
    U+FFFF comes back as a backslash; decode only the short escapes and the
    NUL stays spelled ``\\u0000``.
    """
    import sqlite3

    from polylogue.core.tool_identity import sql_coalesced_json_extract

    projection = sql_coalesced_json_extract("tool_input", ("command",))
    connection = sqlite3.connect(":memory:")
    connection.execute(f"CREATE TABLE t (tool_input TEXT, tool_command TEXT GENERATED ALWAYS AS ({projection}))")
    stored = '{"command":"a￿b \\u0000 \\u001f \\\\ \\uDFFF"}'
    connection.execute("INSERT INTO t (tool_input) VALUES (?)", (stored,))
    (command,) = connection.execute("SELECT tool_command FROM t").fetchone()
    assert command == "a￿b \x00 \x1f \\ \\uDFFF"
