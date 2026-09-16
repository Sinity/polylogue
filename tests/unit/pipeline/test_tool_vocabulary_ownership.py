"""One key vocabulary and one operation taxonomy for tool calls.

polylogue-7k3n0: "which tool_input key holds the command / the path" was
answered five ways, and the narrowest answer was the one the generated
``blocks.tool_path`` column and the FTS ``search_text`` projection used --
so a NotebookEdit's ``notebook_path`` rendered but was unsearchable.

polylogue-exxly: "is this a file / git operation" was decided twice, once
from the origin-neutral ``ToolCategory`` taxonomy and once from a hardcoded
Claude-Code tool-name allowlist.

Anti-vacuity: narrowing ``TOOL_PATH_INPUT_KEYS``, re-spelling the key list in
the DDL, or restoring the tool-name allowlist in ``semantic_capture`` makes
these red.
"""

from __future__ import annotations

from polylogue.archive.viewport.enums import ToolCategory
from polylogue.archive.viewport.tools import classify_tool
from polylogue.core.tool_identity import (
    TOOL_COMMAND_INPUT_KEYS,
    TOOL_PATH_INPUT_KEYS,
    sql_coalesced_json_extract,
    tool_input_command,
    tool_input_path,
)
from polylogue.pipeline.semantic_capture import extract_tool_invocations, parse_git_operation
from polylogue.storage.sqlite.archive_tiers.archive_tiers_specs import BLOCKS_SPEC


def _blocks_column_sql(name: str) -> str:
    for line in BLOCKS_SPEC.ddl_body.splitlines():
        if line.strip().startswith(f"{name} "):
            return line
    raise AssertionError(f"no {name} column on blocks")


def test_generated_columns_carry_the_declared_key_vocabulary() -> None:
    path_ddl = _blocks_column_sql("tool_path")
    search_ddl = _blocks_column_sql("search_text")
    command_ddl = _blocks_column_sql("tool_command")
    for key in TOOL_PATH_INPUT_KEYS:
        assert f"'$.{key}'" in path_ddl, key
        assert f"'$.{key}'" in search_ddl, key
    for key in TOOL_COMMAND_INPUT_KEYS:
        assert f"'$.{key}'" in command_ddl, key
        assert f"'$.{key}'" in search_ddl, key
    assert sql_coalesced_json_extract("tool_input", TOOL_PATH_INPUT_KEYS) in path_ddl


def test_notebook_path_is_a_path_everywhere() -> None:
    payload = {"notebook_path": "/repo/analysis.ipynb"}
    assert tool_input_path(payload) == "/repo/analysis.ipynb"
    assert "'$.notebook_path'" in _blocks_column_sql("tool_path")


def test_command_keys_cover_both_spellings() -> None:
    assert tool_input_command({"command": "git status"}) == "git status"
    assert tool_input_command({"cmd": "git status"}) == "git status"
    assert tool_input_command({}) is None


def test_shell_classification_reads_both_command_keys() -> None:
    assert classify_tool("bash", {"command": "git log"}) is ToolCategory.GIT
    assert classify_tool("bash", {"cmd": "git log"}) is ToolCategory.GIT
    assert classify_tool("bash", {"command": "ls"}) is ToolCategory.SHELL


def test_capture_predicates_follow_the_taxonomy_not_a_tool_name_allowlist() -> None:
    blocks = [
        {"type": "tool_use", "name": "apply_patch", "id": "t1", "input": {"file_path": "/a.py"}},
        {"type": "tool_use", "name": "shell", "id": "t2", "input": {"cmd": "git commit -m x"}},
        {"type": "tool_use", "name": "MultiEdit", "id": "t3", "input": {"path": "/b.py"}},
    ]
    invocations = extract_tool_invocations(blocks)
    # None of these three names appear in the retired allowlist.
    assert invocations[0]["is_file_operation"] is True
    assert invocations[1]["is_git_operation"] is True
    assert invocations[2]["is_file_operation"] is True


def test_git_metadata_is_not_gated_on_the_tool_being_named_bash() -> None:
    assert parse_git_operation({"tool_name": "shell", "input": {"cmd": "git commit -m x"}}) is not None
    assert parse_git_operation({"tool_name": "shell", "input": {"cmd": "ls"}}) is None
