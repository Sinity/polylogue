"""Generated-style coverage for public CLI action contracts (#1816)."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import click
from click.testing import CliRunner
from rich.console import Console

from polylogue.cli.click_app import cli
from polylogue.cli.command_inventory import CommandPath, iter_command_paths
from polylogue.cli.commands.config import config_command
from polylogue.cli.query_group import _split_query_mode_args
from polylogue.operations.action_contracts import (
    ACTION_CONTRACT_BY_PATH,
    ACTION_CONTRACTS,
    PUBLIC_ACTION_FLOOR,
    VIRTUAL_ACTION_PATHS,
    CliActionContract,
    action_affordance_payloads,
    action_completion_contexts,
)
from tests.infra.app_env import make_app_env

SCHEMAS_DIR = Path("docs/schemas/cli-output")

GUARD_BEHAVIOR_COVERAGE: dict[str, str] = {
    "daemon_accepts_schedule": "test_import_contract_guard_requires_daemon_acceptance",
    "dry_run_or_yes_required": "test_delete_contract_guard_refuses_plain_forceless_delete",
    "explicit_query_intent": "test_virtual_find_counterpart_is_query_parser_keyword",
    "file_destination_requires_out": "test_read_contract_guard_requires_out_for_file_destination",
    "path_exists_or_demo": "test_import_contract_guard_rejects_missing_source_path",
    "secret_values_redacted": "test_config_contract_guard_redacts_secret_values",
    "single_match_unless_all": "test_delete_contract_guard_requires_all_for_multi_match",
    "single_match_unless_all_or_first": "test_mark_contract_guard_requires_all_or_first_for_multi_match",
}

COMPLETION_CONTEXT_COVERAGE: dict[str, str] = {
    "config_key": "config command key space is covered by config guard/schema tests",
    "filesystem_path": "import path handling is covered by path guard tests",
    "query_expression": "find/analyze query grammar is covered by query parser tests",
    "session_id": "session-id shell completion is covered by test_completion_matrix.py",
}


def _load_cli_output_schema(name: str) -> dict[str, object]:
    target = SCHEMAS_DIR / f"{name}.schema.json"
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _command_paths() -> dict[tuple[str, ...], click.Command]:
    return {entry.path: entry.command for entry in iter_command_paths(cli)}


def _command_for_contract(entry: CliActionContract, commands: dict[tuple[str, ...], click.Command]) -> click.Command:
    if entry.path == ("find",):
        return cli
    return commands[entry.path]


def _assert_contract_declares_guard(path: tuple[str, ...], guard: str) -> None:
    contract = ACTION_CONTRACT_BY_PATH[path]
    assert guard in contract.guards, f"{path!r} no longer declares guard {guard!r}"


def _choice_values(command: click.Command) -> set[str]:
    values: set[str] = set()
    ctx = click.Context(command)
    for param in command.get_params(ctx):
        if not isinstance(param, click.Option):
            continue
        if not {"--format", "-f", "--output-format", "--json"}.intersection(param.opts):
            continue
        if isinstance(param.type, click.Choice):
            values.update(str(choice) for choice in param.type.choices)
        if "--json" in param.opts:
            values.add("json")
    return values


def test_every_public_floor_action_has_exactly_one_contract() -> None:
    """Every v0 public floor action must declare one executable contract."""
    declared = set(ACTION_CONTRACT_BY_PATH)
    expected = set(PUBLIC_ACTION_FLOOR)
    assert declared == expected, (
        "The #1816 public CLI action floor and ACTION_CONTRACTS drifted.\n"
        f"Missing contracts: {sorted(expected - declared)}\n"
        f"Unexpected contracts: {sorted(declared - expected)}"
    )
    assert len(ACTION_CONTRACTS) == len(declared), "ACTION_CONTRACTS contains duplicate path entries"


def test_contract_paths_resolve_to_click_or_virtual_counterpart() -> None:
    """Every contract path must map to a live Click path or declared virtual grammar action."""
    commands = _command_paths()
    stale = sorted(
        path for path in ACTION_CONTRACT_BY_PATH if path not in commands and path not in VIRTUAL_ACTION_PATHS
    )
    assert not stale, (
        f"These CLI action contracts do not correspond to a live Click command or declared virtual action: {stale}"
    )


def test_virtual_find_counterpart_is_query_parser_keyword() -> None:
    """`find` is public grammar, so prove the parser still recognizes it."""
    click_args, query_terms, has_subcommand, explicit_query = _split_query_mode_args(cli, ["find", "needle"])
    assert click_args == []
    assert query_terms == ("needle",)
    assert not has_subcommand
    assert explicit_query


def test_every_declared_guard_has_behavior_coverage() -> None:
    """Guard metadata must be bound to executable behavior coverage."""
    declared = {guard for contract in ACTION_CONTRACTS for guard in contract.guards}
    covered = set(GUARD_BEHAVIOR_COVERAGE)
    assert declared == covered, (
        "#1816 guards should not be documentation-only metadata.\n"
        f"Missing behavior coverage: {sorted(declared - covered)}\n"
        f"Stale behavior coverage: {sorted(covered - declared)}"
    )


def test_every_completion_context_has_declared_coverage() -> None:
    """Completion-context metadata must stay tied to executable surfaces."""
    declared: set[str] = set(action_completion_contexts())
    covered = set(COMPLETION_CONTEXT_COVERAGE)
    assert declared == covered, (
        "#1816 completion contexts should not be free-form notes.\n"
        f"Missing coverage: {sorted(declared - covered)}\n"
        f"Stale coverage: {sorted(covered - declared)}"
    )


def test_floor_click_paths_without_contract_are_reported() -> None:
    """The non-virtual floor paths currently present in Click must be contracted."""
    commands = _command_paths()
    missing = sorted(path for path in PUBLIC_ACTION_FLOOR if path in commands and path not in ACTION_CONTRACT_BY_PATH)
    assert not missing, f"Public floor Click paths lack CliActionContract entries: {missing}"


def test_declared_machine_formats_are_supported_by_click_options() -> None:
    """Contracts may not claim JSON/NDJSON unless the live command exposes it."""
    commands = _command_paths()
    unsupported: list[str] = []
    for entry in ACTION_CONTRACTS:
        command = _command_for_contract(entry, commands)
        choices = _choice_values(command)
        if "json" in entry.formats and "json" not in choices:
            unsupported.append(f"{CommandPath(entry.path, command).display_name}: json")
        if "ndjson" in entry.formats and "ndjson" not in choices:
            unsupported.append(f"{CommandPath(entry.path, command).display_name}: ndjson")
    assert not unsupported, f"Contracts declare unsupported machine formats: {unsupported}"


def test_default_format_is_declared_for_every_contract() -> None:
    """The default format must be one of the contract's declared formats."""
    invalid = [entry.path for entry in ACTION_CONTRACTS if entry.default_format not in entry.formats]
    assert not invalid, f"Contracts have default_format outside formats: {invalid}"


def test_action_contracts_emit_shared_affordance_payloads() -> None:
    """The public floor exposes the #2305 affordance fields as JSON-native data."""
    payloads = action_affordance_payloads()
    by_id = {str(payload["id"]): payload for payload in payloads}

    assert set(by_id) == {".".join(contract.path) for contract in ACTION_CONTRACTS}

    read: dict[str, Any] = by_id["read"]
    assert read["target"] == "selection"
    assert read["input_unit"] == "query_result_set"
    assert read["cardinality_state"] == "explicit_multi"
    assert read["safety_level"] == "safe"
    assert read["selection_command"] == "polylogue find QUERY then select"
    assert "browser" in read["destination_support"]
    assert read["format_support"] == ["human", "json", "ndjson"]
    assert "continue" in read["next_actions"]

    delete: dict[str, Any] = by_id["delete"]
    assert delete["safety_level"] == "destructive"
    assert delete["confirmation_command"] == "polylogue find QUERY then delete --dry-run"
    assert "dry_run_or_yes_required" in delete["guards"]


def test_query_result_actions_declare_selection_or_confirmation_affordance() -> None:
    """Singleton/multi query-result actions must expose how to disambiguate targets."""
    missing = [
        contract.path
        for contract in ACTION_CONTRACTS
        if contract.input_unit == "query_result_set"
        and contract.cardinality in {"singleton", "explicit_multi", "destructive_multi"}
        and contract.path != ("select",)
        and contract.selection_command is None
    ]
    assert not missing


def test_destructive_actions_declare_confirmation_affordance() -> None:
    """Destructive actions must publish an explicit preview/confirmation command."""
    missing = [
        contract.path
        for contract in ACTION_CONTRACTS
        if contract.safety_level == "destructive" and contract.confirmation_command is None
    ]
    assert not missing


def test_mutation_contracts_have_published_schema() -> None:
    """Mutation action contracts must be backed by a generated JSON Schema."""
    from devtools.render_cli_output_schemas import SCHEMAS
    from polylogue.surfaces.payloads import MutationResultPayload

    mutation_paths = sorted(entry.path for entry in ACTION_CONTRACTS if entry.machine_envelope == "mutation")
    assert mutation_paths == [("delete",), ("import",), ("mark",)]

    schema_by_name = {entry.name: entry for entry in SCHEMAS}
    mutation_schema = schema_by_name.get("mutation-result")
    assert mutation_schema is not None, "mutation-result schema missing from render_cli_output_schemas.SCHEMAS"
    assert mutation_schema.model is MutationResultPayload
    assert (SCHEMAS_DIR / "mutation-result.schema.json").exists()


def test_delete_contract_guard_refuses_plain_forceless_delete(workspace_env: dict[str, Path]) -> None:
    """`dry_run_or_yes_required` is enforced by the public delete CLI."""
    _assert_contract_declares_guard(("delete",), "dry_run_or_yes_required")

    runner = CliRunner()
    with patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=["session-1"]):
        result = runner.invoke(cli, ["--plain", "find", "needle", "then", "delete"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"] == "delete"
    assert payload["status"] == "aborted"
    assert payload["detail"] == "confirmation_required"
    assert payload["affected_count"] == 0


def test_delete_contract_aborted_payload_matches_mutation_schema(workspace_env: dict[str, Path]) -> None:
    """The destructive-action guard emits the declared mutation envelope."""
    import jsonschema

    _assert_contract_declares_guard(("delete",), "dry_run_or_yes_required")
    schema = _load_cli_output_schema("mutation-result")

    runner = CliRunner()
    with patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=["session-1"]):
        result = runner.invoke(cli, ["--plain", "find", "needle", "then", "delete"])

    assert result.exit_code == 0, result.output
    jsonschema.validate(instance=json.loads(result.output), schema=schema)


def test_delete_contract_guard_allows_dry_run_preview(workspace_env: dict[str, Path]) -> None:
    """`dry_run_or_yes_required` permits previewing the full resolved set."""
    _assert_contract_declares_guard(("delete",), "dry_run_or_yes_required")

    session_ids = ["session-1", "session-2"]
    runner = CliRunner()
    with patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=session_ids):
        result = runner.invoke(cli, ["--plain", "find", "needle", "then", "delete", "--dry-run"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["operation"] == "delete"
    assert payload["status"] == "preview"
    assert payload["session_count"] == len(session_ids)
    assert payload["affected_count"] == 0
    assert payload["session_ids"] == session_ids


def test_delete_contract_preview_payload_matches_mutation_schema(workspace_env: dict[str, Path]) -> None:
    """The dry-run destructive action emits a schema-valid mutation preview."""
    import jsonschema

    _assert_contract_declares_guard(("delete",), "dry_run_or_yes_required")
    schema = _load_cli_output_schema("mutation-result")

    session_ids = ["session-1", "session-2"]
    runner = CliRunner()
    with patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=session_ids):
        result = runner.invoke(cli, ["--plain", "find", "needle", "then", "delete", "--dry-run"])

    assert result.exit_code == 0, result.output
    jsonschema.validate(instance=json.loads(result.output), schema=schema)


def test_delete_contract_guard_requires_all_for_multi_match(workspace_env: dict[str, Path]) -> None:
    """`single_match_unless_all` is enforced before destructive deletion."""
    _assert_contract_declares_guard(("delete",), "single_match_unless_all")

    runner = CliRunner()
    with (
        patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=["session-1", "session-2"]),
        patch("polylogue.cli.archive_query.execute_delete_by_session_ids") as execute_delete,
    ):
        result = runner.invoke(cli, ["--plain", "find", "needle", "then", "delete", "--yes"])

    assert result.exit_code != 0
    assert "--all" in result.output
    execute_delete.assert_not_called()


def test_read_contract_guard_requires_out_for_file_destination(workspace_env: dict[str, Path]) -> None:
    """`file_destination_requires_out` is enforced by the public read verb."""
    _assert_contract_declares_guard(("read",), "file_destination_requires_out")

    result = CliRunner().invoke(cli, ["find", "needle", "then", "read", "--to", "file"])

    assert result.exit_code != 0
    assert "--out" in result.output


def test_mark_contract_guard_requires_all_or_first_for_multi_match(workspace_env: dict[str, Path]) -> None:
    """`single_match_unless_all_or_first` is enforced before mark mutation."""
    _assert_contract_declares_guard(("mark",), "single_match_unless_all_or_first")

    runner = CliRunner()
    with (
        patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=["session-1", "session-2"]),
        patch("polylogue.api.sync.bridge.run_coroutine_sync") as run_coroutine_sync,
    ):
        result = runner.invoke(cli, ["find", "needle", "then", "mark", "--tag-add", "reviewed"])

    assert result.exit_code != 0
    assert "--all" in result.output
    assert "--first" in result.output
    run_coroutine_sync.assert_not_called()


def test_mark_contract_guard_allows_first_for_multi_match(workspace_env: dict[str, Path]) -> None:
    """The mark guard permits the explicitly first-only multi-match path."""
    _assert_contract_declares_guard(("mark",), "single_match_unless_all_or_first")

    def _close_coroutine(coro: object) -> None:
        close = getattr(coro, "close", None)
        if callable(close):
            close()

    runner = CliRunner()
    with (
        patch("polylogue.cli.verb_cardinality.resolve_session_ids_for_verb", return_value=["session-1", "session-2"]),
        patch("polylogue.api.sync.bridge.run_coroutine_sync", side_effect=_close_coroutine) as run_coroutine_sync,
    ):
        result = runner.invoke(cli, ["find", "needle", "then", "mark", "--tag-add", "reviewed", "--first"])

    assert result.exit_code == 0, result.output
    assert "Marked 1 session" in result.output
    run_coroutine_sync.assert_called_once()


def test_explicit_mark_candidates_terms_remain_query_text() -> None:
    """`find mark candidates` searches those words instead of dispatching `mark candidates`."""
    click_args, query_terms, has_subcommand, explicit_query = _split_query_mode_args(
        cli,
        ["find", "mark", "candidates"],
    )

    assert click_args == []
    assert query_terms == ("mark", "candidates")
    assert has_subcommand is False
    assert explicit_query is True


def test_import_contract_guard_rejects_missing_source_path(tmp_path: Path) -> None:
    """`path_exists_or_demo` is enforced by the public import CLI."""
    _assert_contract_declares_guard(("import",), "path_exists_or_demo")

    missing = tmp_path / "missing-export.jsonl"
    result = CliRunner().invoke(cli, ["import", str(missing)])

    assert result.exit_code != 0
    assert "does not exist" in result.output.lower() or "no such" in result.output.lower()


def test_import_contract_guard_requires_daemon_acceptance(tmp_path: Path, workspace_env: dict[str, Path]) -> None:
    """`daemon_accepts_schedule` refuses to claim success on unreachable daemon."""
    _assert_contract_declares_guard(("import",), "daemon_accepts_schedule")

    source = tmp_path / "session.json"
    source.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(cli, ["import", str(source), "--daemon-url", "http://127.0.0.1:9"])

    assert result.exit_code != 0
    assert "Could not reach daemon" in result.output
    assert "polylogued run" in result.output


def test_config_contract_guard_redacts_secret_values(tmp_path: Path, monkeypatch: object) -> None:
    """`secret_values_redacted` is enforced by the public config command."""
    _assert_contract_declares_guard(("config",), "secret_values_redacted")

    from pytest import MonkeyPatch

    assert isinstance(monkeypatch, MonkeyPatch)
    secret = "sk-voyage-CONTRACT-LEAKME"
    cfg = tmp_path / "polylogue.toml"
    cfg.write_text("[embedding]\nenabled = true\n", encoding="utf-8")
    monkeypatch.setenv("POLYLOGUE_CONFIG", str(cfg))
    monkeypatch.setenv("POLYLOGUE_SITE_CONFIG", str(tmp_path / "absent-site.toml"))
    monkeypatch.setenv("VOYAGE_API_KEY", secret)

    env = make_app_env()
    result = CliRunner().invoke(config_command, obj=env, args=["-f", "json"])
    output = cast(StringIO, cast(Console, env.ui.console).file).getvalue()

    assert result.exit_code == 0, result.output
    assert secret not in output
    payload = json.loads(output)
    assert payload["voyage_api_key"] == "<set>"
