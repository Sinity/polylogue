"""Verify six-tool agent-manual generation, declarations, parser, delivery, and live cutover state."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import click

from devtools.render_agent_manual import REPO_ROOT, continuation_example_token, expected_outputs
from polylogue.agent_integration.assets import ALL_ASSETS, agent_asset_metadata, read_agent_asset
from polylogue.agent_integration.installer import AgentIntegrationManager, InstallOptions
from polylogue.agent_integration.manifest import (
    build_live_manifest,
    declared_runtime_tool_names,
    target_contract_schemas_are_live_verified,
    target_tool_names,
    target_tool_names_are_registered,
)
from polylogue.agent_integration.spec import (
    ALL_TARGET_TOOLS,
    CLIENTS,
    DEFAULT_READ_TOOLS,
    ORIGIN_MEANINGS,
    QUERY_EXAMPLES,
    RECIPES,
    TOOL_CONTRACT_BY_NAME,
    TOOL_CONTRACTS,
)
from polylogue.core.enums import Origin
from polylogue.mcp.declarations import PRIVILEGED_ALGEBRA, TARGET_DEFAULT_READ_ALGEBRA, TARGET_PROMPTS, TARGET_RESOURCES

LaneStatus = Literal["pass", "fail", "unverified"]


class _ToolSurface(Protocol):
    fn: Callable[..., object]


class _ToolManager(Protocol):
    _tools: dict[str, _ToolSurface]


class _FastMCPServer(Protocol):
    _tool_manager: _ToolManager


@dataclass(frozen=True, slots=True)
class LaneResult:
    """One independently reportable verification lane."""

    name: str
    status: LaneStatus
    detail: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status, **self.detail}


def _pass(name: str, **detail: object) -> LaneResult:
    return LaneResult(name, "pass", detail)


def _fail(name: str, error: BaseException | str, **detail: object) -> LaneResult:
    return LaneResult(name, "fail", {"error": str(error), **detail})


def _unverified(name: str, reason: BaseException | str, **detail: object) -> LaneResult:
    return LaneResult(name, "unverified", {"reason": str(reason), **detail})


def _generated_assets_lane() -> LaneResult:
    name = "generated-assets"
    try:
        drift = [
            str(path.relative_to(REPO_ROOT))
            for path, expected in expected_outputs().items()
            if not path.exists() or path.read_text(encoding="utf-8") != expected
        ]
        if drift:
            return _fail(name, "generated agent surfaces are out of date", drift=drift)
        manual = read_agent_asset("standing-manual.md")
        reference = read_agent_asset("deep-reference.md")
        if manual != (REPO_ROOT / "docs" / "agent-manual.md").read_text(encoding="utf-8"):
            return _fail(name, "standing manual package/doc mirrors differ")
        if reference != (REPO_ROOT / "docs" / "agent-integration-reference.md").read_text(encoding="utf-8"):
            return _fail(name, "deep reference package/doc mirrors differ")
        for asset in ALL_ASSETS:
            if not read_agent_asset(asset):
                return _fail(name, f"packaged asset is empty: {asset}")
        metadata = agent_asset_metadata()
        return _pass(
            name,
            assets=list(ALL_ASSETS),
            asset_digest=metadata["asset_digest"],
            cache_key=metadata["cache_key"],
            standing_manual_bytes=metadata["standing_manual_bytes"],
            deep_reference_bytes=metadata["deep_reference_bytes"],
        )
    except Exception as exc:
        return _fail(name, exc)


def _json_calls(text: str) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    for raw in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL):
        payload = json.loads(raw)
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            continue
        tool = payload["name"]
        arguments = payload["arguments"]
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            raise ValueError("manual MCP call must contain string name and object arguments")
        calls.append((tool, cast(dict[str, object], arguments)))
    return calls


def _argument_matches_kind(value: object, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    raise AssertionError(f"unknown manual argument kind: {kind}")


def _validate_call(tool: str, arguments: Mapping[str, object]) -> str | None:
    contract = TOOL_CONTRACT_BY_NAME.get(tool)
    if contract is None:
        return f"unknown target tool {tool!r}"
    names = set(arguments)
    unknown = names - set(contract.argument_names)
    if unknown:
        return f"{tool}: unknown arguments {sorted(unknown)}"
    if names == {"continuation"}:
        if not contract.supports_continuation:
            return f"{tool}: continuation-only call documented for non-continuable transaction"
        value = arguments.get("continuation")
        if not isinstance(value, str) or not value.startswith("q1."):
            return f"{tool}: continuation example is not an opaque q1 token"
        return None
    missing = set(contract.required_initial_arguments) - names
    if missing:
        return f"{tool}: initial call omits required arguments {sorted(missing)}"
    if "continuation" in names:
        return f"{tool}: continuation must be the only argument"
    argument_index = {argument.name: argument for argument in contract.arguments}
    wrong_types = [
        f"{name} expected {argument_index[name].kind}"
        for name, value in arguments.items()
        if not _argument_matches_kind(value, argument_index[name].kind)
    ]
    if wrong_types:
        return f"{tool}: invalid argument type(s): {', '.join(wrong_types)}"
    return None


def _manual_compilation_lane() -> LaneResult:
    name = "manual-compilation"
    try:
        declaration_index = {
            declaration.name: declaration for declaration in (*TARGET_DEFAULT_READ_ALGEBRA, *PRIVILEGED_ALGEBRA)
        }
        problems: list[str] = []
        for contract in TOOL_CONTRACTS:
            has_continuation_argument = "continuation" in contract.argument_names
            if contract.supports_continuation != has_continuation_argument:
                problems.append(f"{contract.name}: continuation support and declared continuation argument disagree")
            source_rows = []
            for source_name in contract.source_declarations:
                declaration = declaration_index.get(source_name)
                if declaration is None:
                    problems.append(f"{contract.name}: missing source declaration {source_name}")
                    continue
                source_rows.append(declaration)
            source_roles = {row.minimum_role for row in source_rows}
            if source_roles != {contract.minimum_role}:
                problems.append(
                    f"{contract.name}: role is not derived from source declarations "
                    f"({contract.minimum_role!r} versus {sorted(source_roles)!r})"
                )
            source_semantics = tuple(
                dict.fromkeys(semantic for row in source_rows for semantic in row.result_semantics)
            )
            if source_semantics != contract.result_semantics:
                problems.append(f"{contract.name}: result semantics are not the ordered union of source declarations")
            for example in contract.examples:
                problem = _validate_call(contract.name, example.arguments_dict())
                if problem is not None:
                    problems.append(f"typed example {example.id}: {problem}")
        for recipe in RECIPES:
            for step in recipe.steps:
                if step.tool not in DEFAULT_READ_TOOLS:
                    problems.append(f"{recipe.id}: continuity step uses non-six-tool transaction {step.tool}")
                problem = _validate_call(step.tool, step.arguments_dict())
                if problem is not None:
                    problems.append(f"{recipe.id}: {problem}")
        resource_names = {item.uri_template for item in TARGET_RESOURCES}
        prompt_names = {item.name for item in TARGET_PROMPTS}
        for recipe in RECIPES:
            for resource in recipe.resources:
                if resource not in resource_names:
                    problems.append(f"{recipe.id}: unknown target resource {resource}")
            for prompt in recipe.prompts:
                if prompt not in prompt_names:
                    problems.append(f"{recipe.id}: unknown target prompt {prompt}")
        manual = read_agent_asset("standing-manual.md")
        reference = read_agent_asset("deep-reference.md")
        manual_calls = _json_calls(manual)
        reference_calls = _json_calls(reference)
        for surface, calls in (("standing", manual_calls), ("reference", reference_calls)):
            for tool, arguments in calls:
                problem = _validate_call(tool, arguments)
                if problem is not None:
                    problems.append(f"{surface} manual: {problem}")
        manual_tool_names = {tool for tool, _ in manual_calls}
        missing_normal = set(DEFAULT_READ_TOOLS) - manual_tool_names
        if missing_normal:
            problems.append(f"standing manual omits normal invocation(s): {sorted(missing_normal)}")
        reference_tool_names = {tool for tool, _ in reference_calls}
        missing_reference = set(ALL_TARGET_TOOLS) - reference_tool_names
        if missing_reference:
            problems.append(f"deep reference omits target invocation(s): {sorted(missing_reference)}")
        required_phrases = (
            "same tool with **only** the returned opaque token",
            "Never cite a continuation token",
            "preview-bound confirmation",
            "strict command floor",
            "find` keyword",
            "quoted expression",
            "field syntax",
        )
        for phrase in required_phrases:
            if phrase not in manual:
                problems.append(f"standing manual omits required teaching: {phrase}")
        origin_tokens = tuple(item.token for item in ORIGIN_MEANINGS)
        if origin_tokens != tuple(item.value for item in Origin):
            problems.append("source coverage does not match the authoritative Origin enum")
        if problems:
            return _fail(name, "manual contract does not compile", problems=problems)
        return _pass(
            name,
            typed_transactions=len(TOOL_CONTRACTS),
            standing_calls=len(manual_calls),
            reference_calls=len(reference_calls),
            recipes=len(RECIPES),
            origins=len(ORIGIN_MEANINGS),
            source_declarations=sorted(declaration_index),
        )
    except Exception as exc:
        return _fail(name, exc)


def _query_parser_roundtrip_lane() -> LaneResult:
    name = "query-parser-roundtrip"
    try:
        from polylogue.archive.query.expression import (
            compile_expression,
            explain_expression,
            parse_unit_source_expression,
        )
        from polylogue.cli.query_group import _looks_like_query_expression, _split_query_mode_args
    except ModuleNotFoundError as exc:
        return _unverified(name, exc, production_dependency="locked query parser dependencies")
    try:
        session_count = 0
        terminal_count = 0
        for query in QUERY_EXAMPLES:
            first = explain_expression(query.expression).to_payload()
            second = explain_expression(query.expression).to_payload()
            stable_keys = ("source_text", "lowerer", "ast", "selected_units", "execution_legs")
            if any(first[key] != second[key] for key in stable_keys):
                return _fail(name, f"parser structure is not deterministic: {query.expression}")
            if query.surface == "terminal":
                first_source = parse_unit_source_expression(query.expression)
                second_source = parse_unit_source_expression(query.expression)
                if first_source is None or first_source != second_source:
                    return _fail(name, f"terminal expression did not round-trip: {query.expression}")
                terminal_count += 1
            else:
                compile_expression(query.expression)
                session_count += 1
        group = click.Group()
        _, find_terms, _, explicit_find = _split_query_mode_args(group, ["find", "prior", "art"])
        intent_signals = {
            "find_keyword": explicit_find and find_terms == ("prior", "art"),
            "quoted_expression": _looks_like_query_expression(("prior art",)),
            "field_syntax": _looks_like_query_expression(("repo:polylogue",)),
            "bare_word_refused": not _looks_like_query_expression(("prior",)),
        }
        if not all(intent_signals.values()):
            return _fail(name, "strict command-floor intent signals changed", intent_signals=intent_signals)
        return _pass(
            name,
            session_queries=session_count,
            terminal_queries=terminal_count,
            parser_examples=len(QUERY_EXAMPLES),
            strict_command_floor=intent_signals,
        )
    except Exception as exc:
        return _fail(name, exc)


def _continuation_contract_lane() -> LaneResult:
    name = "continuation-contract"
    try:
        from polylogue.archive.query.transaction import QueryContinuation

        token = continuation_example_token()
        decoded = QueryContinuation.decode(token)
        continuation_calls = [
            (tool, arguments)
            for text in (read_agent_asset("standing-manual.md"), read_agent_asset("deep-reference.md"))
            for tool, arguments in _json_calls(text)
            if set(arguments) == {"continuation"}
        ]
        if not continuation_calls:
            return _fail(name, "generated manuals contain no continuation-only request")
        problems: list[str] = []
        for tool, arguments in continuation_calls:
            if tool != "query":
                problems.append(f"continuation example unexpectedly targets {tool}")
            if arguments != {"continuation": token}:
                problems.append("continuation request is not the exact generated token-only shape")
        if decoded.request.offset != 20:
            problems.append(f"decoded offset is {decoded.request.offset}, expected 20")
        if decoded.request.operation != "query":
            problems.append(f"decoded operation is {decoded.request.operation!r}, expected 'query'")
        if decoded.result_ref != "result:0123456789abcdef01234567":
            problems.append(f"decoded result ref changed: {decoded.result_ref}")
        if problems:
            return _fail(name, "continuation contract failed", problems=problems)
        return _pass(
            name,
            token_version=token.split(".", 1)[0],
            requests=len(continuation_calls),
            offset=decoded.request.offset,
            result_ref=decoded.result_ref,
        )
    except Exception as exc:
        return _fail(name, exc)


def _target_declaration_lane() -> LaneResult:
    name = "target-declaration-reconciliation"
    try:
        manifest = build_live_manifest("admin")
        expected_default = ("query", "read", "get", "explain", "context", "status")
        if target_tool_names("read") != expected_default:
            return _fail(name, "default target transaction order changed", target=list(target_tool_names("read")))
        mappings = {contract.name: list(contract.source_declarations) for contract in TOOL_CONTRACTS}
        if mappings["read"] not in (["read"], ["read", "graph"]):
            return _fail(name, "read compatibility mapping is not mechanical", mapping=mappings["read"])
        if mappings["operate"] not in (["operate"], ["maintenance"]):
            return _fail(name, "operate compatibility mapping is not mechanical", mapping=mappings["operate"])
        return _pass(
            name,
            schema_status=manifest["schema_status"],
            cutover_ready=manifest["cutover_ready"],
            tool_names_registered=manifest["tool_names_registered"],
            contract_schemas_verified=manifest["contract_schemas_verified"],
            runtime_tools=len(cast(list[str], manifest["tools"])),
            target_tools=list(target_tool_names("admin")),
            source_mappings=mappings,
        )
    except Exception as exc:
        return _fail(name, exc)


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _native_installer_lane() -> LaneResult:
    name = "native-installer-roundtrip"
    try:
        with tempfile.TemporaryDirectory(prefix="polylogue-agent-verify-") as temporary:
            root = Path(temporary)
            home = root / "home"
            home.mkdir()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            polylogue = bin_dir / "polylogue"
            server = bin_dir / "polylogue-mcp"
            _make_executable(polylogue)
            _make_executable(server)
            manager = AgentIntegrationManager(home=home, environment={"HOME": str(home), "PATH": str(bin_dir)})
            options = InstallOptions(
                clients=CLIENTS,
                role="read",
                archive_root=root / "archive",
                config_path=root / "polylogue.toml",
                server_command=str(server),
                polylogue_command=str(polylogue),
            )
            first = manager.install(options)
            files = [path for path in home.rglob("*") if path.is_file()]
            mtimes = {path: path.stat().st_mtime_ns for path in files}
            time.sleep(0.002)
            second = manager.install(options)
            unchanged = all(path.stat().st_mtime_ns == mtimes[path] for path in files)
            upgraded = manager.install(
                InstallOptions(
                    clients=options.clients,
                    role="review",
                    archive_root=root / "archive-2",
                    config_path=root / "polylogue-2.toml",
                    server_command=str(server),
                    polylogue_command=str(polylogue),
                )
            )
            doctor = manager.doctor()
            uninstall = manager.uninstall()
            clean_home = list(home.iterdir()) == []
            if not all(
                (
                    first["ok"] is True,
                    second["ok"] is True,
                    upgraded["ok"] is True,
                    doctor["ok"] is True,
                    uninstall["ok"] is True,
                    unchanged,
                    clean_home,
                )
            ):
                return _fail(
                    name,
                    "installer round trip failed",
                    first=first,
                    second=second,
                    upgraded=upgraded,
                    doctor=doctor,
                    uninstall=uninstall,
                    no_rewrite=unchanged,
                    clean_home=clean_home,
                )
            return _pass(
                name,
                clients=list(options.clients),
                no_rewrite_idempotence=True,
                role_archive_upgrade=True,
                doctor=True,
                exact_clean_uninstall=True,
            )
    except Exception as exc:
        return _fail(name, exc)


def _packaging_home_manager_lane() -> LaneResult:
    name = "packaging-and-home-manager"
    try:
        module = REPO_ROOT / "nix" / "agent-integration-module.nix"
        module_text = module.read_text(encoding="utf-8")
        flake_text = (REPO_ROOT / "flake.nix").read_text(encoding="utf-8")
        required = (
            "programs.polylogueAgent",
            'types.enum [ "claude-code" "codex" "gemini" "hermes" ]',
            'types.enum [ "read" "write" "review" "admin" ]',
            '"agent"',
            '"install"',
            '"${cfg.package}/bin/polylogue"',
        )
        missing = [token for token in required if token not in module_text]
        if "homeManagerModules.agentIntegration" not in flake_text:
            missing.append("flake export homeManagerModules.agentIntegration")
        if missing:
            return _fail(name, "packaging/Home Manager contract incomplete", missing=missing)
        if "polylogued run" in module_text or "systemd.user.services.polylogued" in module_text:
            return _fail(name, "agent integration module improperly owns daemon lifecycle")
        return _pass(
            name,
            package_assets=len(ALL_ASSETS),
            home_manager_module=str(module.relative_to(REPO_ROOT)),
            daemon_owned=False,
        )
    except Exception as exc:
        return _fail(name, exc)


def _live_fastmcp_signature_lane() -> LaneResult:
    name = "live-fastmcp-signatures"
    try:
        if not target_tool_names_are_registered("admin"):
            return _unverified(
                name,
                "six-tool cutover is not registered in this snapshot",
                runtime_tools=list(declared_runtime_tool_names("admin")),
                target_tools=list(target_tool_names("admin")),
                blocked_checks=(
                    "exact FastMCP argument names and required/default status",
                    "continuation field and initial-vs-resume signature exclusivity",
                    "read-owned graph/topology selector",
                    "operate preview/token/execute field names",
                ),
                repair="rebase after t46.8.2/t46.8.3, render agent-manual, then rerun with --require-live",
            )
        from polylogue.mcp.server import build_server

        server = cast(_FastMCPServer, build_server(role="admin"))
        surfaces = server._tool_manager._tools
        problems: list[str] = []
        for contract in TOOL_CONTRACTS:
            surface = surfaces.get(contract.name)
            if surface is None:
                problems.append(f"missing registered target tool {contract.name}")
                continue
            fn = surface.fn
            signature = inspect.signature(fn)
            actual_names = set(signature.parameters)
            manual_names = set(contract.argument_names)
            missing_live = manual_names - actual_names
            undocumented_live = actual_names - manual_names
            if missing_live:
                problems.append(f"{contract.name}: manual arguments missing from live signature {sorted(missing_live)}")
            if undocumented_live:
                problems.append(
                    f"{contract.name}: live arguments absent from manual contract {sorted(undocumented_live)}"
                )
            live_required = {
                param.name
                for param in signature.parameters.values()
                if param.default is inspect.Parameter.empty
                and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
            manual_required = set(contract.required_initial_arguments)
            if live_required != manual_required:
                problems.append(
                    f"{contract.name}: required-argument mismatch "
                    f"(live={sorted(live_required)}, manual={sorted(manual_required)})"
                )
        if problems:
            return _fail(name, "live target signatures diverge from generated manual", problems=problems)
        if not target_contract_schemas_are_live_verified():
            return _unverified(
                name,
                "live target signatures match, but generated contracts remain staged",
                signature_match=True,
                schema_status=TOOL_CONTRACTS[0].schema_status,
                repair=(
                    "set TARGET_SCHEMA_STATUS to live-verified, regenerate assets, and rerun --require-live "
                    "only after reviewing this exact signature match"
                ),
            )
        return _pass(name, registered_tools=list(target_tool_names("admin")), checked=len(TOOL_CONTRACTS))
    except ModuleNotFoundError as exc:
        return _unverified(name, exc, production_dependency="MCP SDK and locked runtime dependencies")
    except Exception as exc:
        return _fail(name, exc)


LANES: tuple[tuple[str, Callable[[], LaneResult]], ...] = (
    ("generated-assets", _generated_assets_lane),
    ("manual-compilation", _manual_compilation_lane),
    ("query-parser-roundtrip", _query_parser_roundtrip_lane),
    ("continuation-contract", _continuation_contract_lane),
    ("target-declaration-reconciliation", _target_declaration_lane),
    ("native-installer-roundtrip", _native_installer_lane),
    ("packaging-and-home-manager", _packaging_home_manager_lane),
    ("live-fastmcp-signatures", _live_fastmcp_signature_lane),
)


def run_verification(selected: set[str] | None = None) -> dict[str, object]:
    """Run selected verification lanes and return one stable report."""

    results = [lane() for lane_name, lane in LANES if selected is None or lane_name in selected]
    counts = {status: sum(result.status == status for result in results) for status in ("pass", "fail", "unverified")}
    return {
        "ok": counts["fail"] == 0,
        "complete": counts["fail"] == 0 and counts["unverified"] == 0,
        "counts": counts,
        "lanes": [result.to_dict() for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", action="append", choices=[name for name, _ in LANES])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-live", action="store_true", help="Treat the cutover-blocked live lane as failure.")
    args = parser.parse_args(argv)
    report = run_verification(set(args.lane) if args.lane else None)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for lane in cast(list[dict[str, object]], report["lanes"]):
            print(f"{lane['name']:<36} {str(lane['status']).upper()}")
            if lane["status"] != "pass":
                print(f"  {lane.get('error') or lane.get('reason')}")
    if report["ok"] is not True:
        return 1
    if args.require_live and report["complete"] is not True:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
