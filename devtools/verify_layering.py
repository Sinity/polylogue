"""Verify import layering and production SQLite writer ownership.

Gate classification: **blocking architectural boundary check**.

The writer doctrine is intentionally rooted in real mutation surfaces rather
than voluntary class labels.  Each archive-tier module that performs a direct
SQL mutation must be inventoried in ``docs/plans/layering.yaml`` and declare
its owned tier(s) in its module docstring.  A module spanning two tiers is
allowed only when the same manifest names a reviewed twin-write contract.

Usage:
  devtools verify layering
  devtools verify layering --json
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from devtools import repo_root as _get_root
from polylogue.core.json import dumps
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER

_DEFAULT_WRITER_MODULE_MARKER = "Writer module:"
_TWIN_WRITE_CONTRACT_MARKER = "Twin-write contract:"
_WRITER_TIER_RE = re.compile(r"[a-z][a-z0-9_-]*")
_SQL_MUTATION_RE = re.compile(r"(?:^|\n)\s*(?:INSERT|UPDATE|DELETE|REPLACE)\b", re.IGNORECASE)
_SQL_MUTATION_TABLE_RE = re.compile(
    r"^\s*(?:INSERT(?:\s+OR\s+\w+)?\s+INTO|REPLACE\s+INTO|UPDATE|DELETE\s+FROM)\s+[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_SQL_EXECUTION_METHODS = frozenset({"execute", "executemany", "executescript"})
_WRITER_SURFACE_CONTRACTS = {
    "source": ("durable", "atomic"),
    "index": ("rebuildable", "replayable"),
    "embeddings": ("rebuildable", "replayable"),
    "user": ("durable", "atomic"),
    "ops": ("disposable", "restartable"),
}


@dataclass(frozen=True)
class WriterModuleSurface:
    tier: str
    durability: str
    interruption: str


@dataclass(frozen=True)
class WriterModuleSpec:
    path: str
    surfaces: tuple[WriterModuleSurface, ...]
    entrypoints: tuple[str, ...]
    twin_write_contract: str | None


@dataclass(frozen=True)
class TwinWriteContract:
    name: str
    module: str
    surfaces: tuple[str, ...]
    entrypoints: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class WriterModulePolicy:
    marker: str
    mutation_roots: tuple[str, ...]
    modules: tuple[WriterModuleSpec, ...]
    twin_write_contracts: dict[str, TwinWriteContract]


@dataclass(frozen=True)
class WriterModuleDeclaration:
    file: str
    tiers: tuple[str, ...]
    contract: str | None
    line: int
    well_formed: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enforce inter-package layering rules.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def _load_manifest(rules_path: Path) -> dict[str, object]:
    import yaml

    with open(rules_path, encoding="utf-8") as f:
        data: object = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_rules(rules_path: Path) -> list[dict[str, object]]:
    data = _load_manifest(rules_path)
    rules: object = data.get("rules", [])
    if isinstance(rules, list):
        return [rule for rule in rules if isinstance(rule, dict)]
    return []


def _collect_imports(package_dir: Path, *, repo_root: Path) -> dict[str, set[str]]:
    imports: dict[str, set[str]] = {}
    for py_file in package_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = py_file.relative_to(repo_root).as_posix()
        imports.setdefault(rel, set())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[rel].add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports[rel].add(node.module)
    return imports


def _package_name(value: str) -> str:
    return value.strip().replace("/", ".").removesuffix(".__init__").removesuffix(".py")


def _package_matches(target: str, import_module: str) -> bool:
    """Check if import_module falls under the target package."""
    normalized_target = _package_name(target)
    normalized_import = _package_name(import_module)
    return normalized_import == normalized_target or normalized_import.startswith(normalized_target + ".")


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))


def _writer_module_policy(manifest: dict[str, object]) -> WriterModulePolicy | None:
    raw_policy = manifest.get("writer_modules")
    if not isinstance(raw_policy, dict):
        return None

    raw_modules = raw_policy.get("modules")
    modules: list[WriterModuleSpec] = []
    if isinstance(raw_modules, list):
        for raw_module in raw_modules:
            if not isinstance(raw_module, dict):
                continue
            raw_surfaces = raw_module.get("surfaces")
            surfaces: list[WriterModuleSurface] = []
            if isinstance(raw_surfaces, list):
                for raw_surface in raw_surfaces:
                    if not isinstance(raw_surface, dict):
                        continue
                    tier = raw_surface.get("tier")
                    durability = raw_surface.get("durability")
                    interruption = raw_surface.get("interruption")
                    if isinstance(tier, str) and isinstance(durability, str) and isinstance(interruption, str):
                        surfaces.append(
                            WriterModuleSurface(
                                tier=tier,
                                durability=durability,
                                interruption=interruption,
                            )
                        )
            path = raw_module.get("path")
            if isinstance(path, str):
                contract = raw_module.get("twin_write_contract")
                modules.append(
                    WriterModuleSpec(
                        path=path,
                        surfaces=tuple(surfaces),
                        entrypoints=_strings(raw_module.get("entrypoints")),
                        twin_write_contract=contract if isinstance(contract, str) else None,
                    )
                )

    contracts: dict[str, TwinWriteContract] = {}
    raw_contracts = raw_policy.get("twin_write_contracts")
    if isinstance(raw_contracts, list):
        for raw_contract in raw_contracts:
            if not isinstance(raw_contract, dict):
                continue
            name = raw_contract.get("name")
            module = raw_contract.get("module")
            reason = raw_contract.get("reason")
            if isinstance(name, str) and isinstance(module, str) and isinstance(reason, str):
                contracts[name] = TwinWriteContract(
                    name=name,
                    module=module,
                    surfaces=_strings(raw_contract.get("surfaces")),
                    entrypoints=_strings(raw_contract.get("entrypoints")),
                    reason=reason,
                )

    return WriterModulePolicy(
        marker=str(raw_policy.get("marker") or _DEFAULT_WRITER_MODULE_MARKER),
        mutation_roots=_strings(raw_policy.get("mutation_roots")),
        modules=tuple(modules),
        twin_write_contracts=contracts,
    )


def _string_fragments(expression: ast.expr, values: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
        return (expression.value,)
    if isinstance(expression, ast.Name) and values is not None:
        return values.get(expression.id, ())
    if isinstance(expression, ast.JoinedStr):
        return tuple(
            value.value
            for value in expression.values
            if isinstance(value, ast.Constant) and isinstance(value.value, str)
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        return _string_fragments(expression.left, values) + _string_fragments(expression.right, values)
    return ()


def _string_assignments(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for _ in range(3):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            fragments = _string_fragments(node.value, values)
            if fragments:
                values[node.targets[0].id] = fragments
    return values


def _mutation_sql(node: ast.Call, *, values: dict[str, tuple[str, ...]]) -> str | None:
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in _SQL_EXECUTION_METHODS:
        return None
    if not node.args:
        return None
    sql = "".join(_string_fragments(node.args[0], values))
    return sql if sql and _SQL_MUTATION_RE.search(sql) else None


def _mutation_table(sql: str) -> str | None:
    match = _SQL_MUTATION_TABLE_RE.match(sql)
    return match.group(1) if match is not None else None


def _archive_table_tiers() -> dict[str, str]:
    table_tiers: dict[str, str] = {}
    for tier, ddl in ARCHIVE_DDL_BY_TIER.items():
        tier_name = str(getattr(tier, "value", tier))
        for table in _CREATE_TABLE_RE.findall(ddl):
            table_tiers[table] = tier_name
    return table_tiers


def _mutation_calls(tree: ast.AST) -> tuple[ast.Call, ...]:
    values = _string_assignments(tree)
    return tuple(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and _mutation_sql(node, values=values) is not None
    )


def _mutation_tiers(tree: ast.AST) -> frozenset[str]:
    values = _string_assignments(tree)
    table_tiers = _archive_table_tiers()
    tiers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        sql = _mutation_sql(node, values=values)
        if sql is None:
            continue
        table = _mutation_table(sql)
        if table is not None and (tier := table_tiers.get(table)) is not None:
            tiers.add(tier)
    return frozenset(tiers)


def _parse_writer_module_declaration(tree: ast.Module, *, file: str, marker: str) -> WriterModuleDeclaration | None:
    docstring = ast.get_docstring(tree, clean=False)
    if docstring is None or marker not in docstring or not tree.body:
        return None

    marker_lines = [line.strip() for line in docstring.splitlines() if line.strip().startswith(marker)]
    if len(marker_lines) != 1:
        return WriterModuleDeclaration(file=file, tiers=(), contract=None, line=tree.body[0].lineno, well_formed=False)

    raw_tiers = marker_lines[0][len(marker) :].strip().removesuffix(".")
    tiers = tuple(part.strip() for part in raw_tiers.split(",") if part.strip())
    well_formed = bool(tiers) and all(_WRITER_TIER_RE.fullmatch(tier) is not None for tier in tiers)

    contract_lines = [
        line.strip() for line in docstring.splitlines() if line.strip().startswith(_TWIN_WRITE_CONTRACT_MARKER)
    ]
    contract: str | None = None
    if len(contract_lines) == 1:
        candidate = contract_lines[0][len(_TWIN_WRITE_CONTRACT_MARKER) :].strip().removesuffix(".")
        contract = candidate or None
    elif len(contract_lines) > 1:
        well_formed = False

    return WriterModuleDeclaration(
        file=file,
        tiers=tiers,
        contract=contract,
        line=tree.body[0].lineno,
        well_formed=well_formed,
    )


def _writer_module_files(repo_root: Path, policy: WriterModulePolicy) -> dict[str, tuple[Path, ast.Module]]:
    files: dict[str, tuple[Path, ast.Module]] = {}
    for root in policy.mutation_roots:
        root_path = repo_root / root
        if not root_path.exists():
            continue
        for py_file in root_path.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            rel = py_file.relative_to(repo_root).as_posix()
            files[rel] = (py_file, tree)
    return files


def _function_definitions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}


def _imported_writer_modules(tree: ast.Module) -> dict[str, str]:
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("polylogue.storage.sqlite.archive_tiers."):
            continue
        path = node.module.replace(".", "/") + ".py"
        for alias in node.names:
            imports[alias.asname or alias.name] = path
    return imports


def _imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _imported_sql_execution_lines(tree: ast.Module) -> list[int]:
    """Find execute calls whose SQL text is hidden behind an import.

    The layering gate cannot classify an imported constant against this
    module's owned tiers. Keep executable SQL beside its owning writer so a
    change to the statement cannot bypass the writer inventory.
    """
    imported = _imported_names(tree)
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _SQL_EXECUTION_METHODS
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in imported
    )


def _called_function_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _entrypoint_tiers(
    tree: ast.Module,
    entrypoint: str,
    specs: dict[str, WriterModuleSpec],
    *,
    follow_imports: bool = True,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    imported_modules: dict[str, str] | None = None,
    direct_tiers: dict[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    functions = functions or _function_definitions(tree)
    imported_modules = imported_modules or _imported_writer_modules(tree)
    direct_tiers = direct_tiers or {name: _mutation_tiers(function) for name, function in functions.items()}
    pending = [entrypoint]
    visited: set[str] = set()
    tiers: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        function = functions.get(name)
        if function is None:
            spec = specs.get(imported_modules.get(name, "")) if follow_imports else None
            if spec is not None:
                tiers.update(surface.tier for surface in spec.surfaces)
            continue
        tiers.update(direct_tiers[name])
        for called_name in _called_function_names(function):
            if called_name in functions or called_name in imported_modules:
                pending.append(called_name)
    return frozenset(tiers)


def _contract_is_audited(
    declaration: WriterModuleDeclaration,
    spec: WriterModuleSpec,
    policy: WriterModulePolicy,
) -> bool:
    if declaration.contract is None or declaration.contract != spec.twin_write_contract:
        return False
    contract = policy.twin_write_contracts.get(declaration.contract)
    if contract is None or contract.module != spec.path:
        return False
    expected_tiers = tuple(surface.tier for surface in spec.surfaces)
    return (
        tuple(contract.surfaces) == expected_tiers
        and set(contract.entrypoints).issubset(spec.entrypoints)
        and bool(contract.reason)
    )


def _collect_writer_module_violations(repo_root: Path, policy: WriterModulePolicy | None) -> list[dict[str, object]]:
    if policy is None:
        return []

    violations: list[dict[str, object]] = []
    files = _writer_module_files(repo_root, policy)
    specs = {spec.path: spec for spec in policy.modules}
    declarations = {
        rel: _parse_writer_module_declaration(tree, file=rel, marker=policy.marker) for rel, (_, tree) in files.items()
    }
    mutation_files = {rel for rel, (_, tree) in files.items() if _mutation_calls(tree)}

    for rel in sorted(mutation_files):
        spec = specs.get(rel)
        declaration = declarations[rel]
        if spec is None or declaration is None:
            violations.append(
                {
                    "file": rel,
                    "rule": "writer_module_unmarked_mutation",
                    "marker": policy.marker,
                }
            )

    for rel, declaration in declarations.items():
        if declaration is not None and rel not in specs:
            violations.append(
                {
                    "file": rel,
                    "line": declaration.line,
                    "rule": "writer_module_uninventoried_declaration",
                }
            )

    for spec in policy.modules:
        file_entry = files.get(spec.path)
        if file_entry is None:
            violations.append({"file": spec.path, "rule": "writer_module_missing_file"})
            continue
        _, tree = file_entry
        declaration = declarations[spec.path]
        if spec.path not in mutation_files:
            violations.append({"file": spec.path, "rule": "writer_module_inventory_without_mutation"})
            continue
        if declaration is None:
            continue
        for surface in spec.surfaces:
            if _WRITER_SURFACE_CONTRACTS.get(surface.tier) != (surface.durability, surface.interruption):
                violations.append(
                    {
                        "file": spec.path,
                        "rule": "writer_module_invalid_surface",
                        "tier": surface.tier,
                        "durability": surface.durability,
                        "interruption": surface.interruption,
                    }
                )
        if not declaration.well_formed:
            violations.append(
                {
                    "file": spec.path,
                    "line": declaration.line,
                    "rule": "writer_module_invalid_declaration",
                }
            )
            continue

        expected_tiers = tuple(surface.tier for surface in spec.surfaces)
        for line in _imported_sql_execution_lines(tree):
            violations.append(
                {
                    "file": spec.path,
                    "line": line,
                    "rule": "writer_module_imported_sql_opaque",
                }
            )
        observed_tiers = _mutation_tiers(tree)
        unexpected_tiers = sorted(observed_tiers.difference(expected_tiers))
        if unexpected_tiers:
            violations.append(
                {
                    "file": spec.path,
                    "rule": "writer_module_observed_tier_mismatch",
                    "observed": sorted(observed_tiers),
                    "expected": list(expected_tiers),
                }
            )
        if declaration.tiers != expected_tiers:
            violations.append(
                {
                    "file": spec.path,
                    "line": declaration.line,
                    "rule": "writer_module_declaration_mismatch",
                    "declared": list(declaration.tiers),
                    "expected": list(expected_tiers),
                }
            )

        functions = _function_definitions(tree)
        imported_modules = _imported_writer_modules(tree)
        direct_tiers = {name: _mutation_tiers(function) for name, function in functions.items()}

        if len(declaration.tiers) > 1:
            if not _contract_is_audited(declaration, spec, policy):
                violations.append(
                    {
                        "file": spec.path,
                        "line": declaration.line,
                        "rule": "writer_module_mixed_file",
                        "declared": list(declaration.tiers),
                        "contract": declaration.contract,
                    }
                )
            else:
                contract_name = declaration.contract
                assert contract_name is not None
                contract = policy.twin_write_contracts[contract_name]
                for entrypoint in contract.entrypoints:
                    contract_tiers = _entrypoint_tiers(
                        tree,
                        entrypoint,
                        specs,
                        functions=functions,
                        imported_modules=imported_modules,
                        direct_tiers=direct_tiers,
                    )
                    if set(contract.surfaces) != contract_tiers:
                        violations.append(
                            {
                                "file": spec.path,
                                "rule": "writer_module_contract_unproven",
                                "entrypoint": entrypoint,
                                "observed": sorted(contract_tiers),
                                "expected": list(contract.surfaces),
                            }
                        )
        elif len(declaration.tiers) == 1 and declaration.contract is not None:
            violations.append(
                {
                    "file": spec.path,
                    "line": declaration.line,
                    "rule": "writer_module_unneeded_contract",
                    "contract": declaration.contract,
                }
            )

        observed_entrypoints = {
            name
            for name in functions
            if not name.startswith("_")
            and _entrypoint_tiers(
                tree,
                name,
                specs,
                follow_imports=False,
                functions=functions,
                imported_modules=imported_modules,
                direct_tiers=direct_tiers,
            )
        }
        if observed_entrypoints != set(spec.entrypoints):
            violations.append(
                {
                    "file": spec.path,
                    "rule": "writer_module_entrypoint_inventory_mismatch",
                    "observed": sorted(observed_entrypoints),
                    "expected": list(spec.entrypoints),
                }
            )
        for entrypoint in spec.entrypoints:
            function = functions.get(entrypoint)
            if function is None:
                violations.append(
                    {
                        "file": spec.path,
                        "rule": "writer_module_missing_entrypoint",
                        "entrypoint": entrypoint,
                    }
                )
            elif not _entrypoint_tiers(
                tree,
                entrypoint,
                specs,
                follow_imports=False,
                functions=functions,
                imported_modules=imported_modules,
                direct_tiers=direct_tiers,
            ):
                violations.append(
                    {
                        "file": spec.path,
                        "line": function.lineno,
                        "rule": "writer_module_entrypoint_not_mutating",
                        "entrypoint": entrypoint,
                    }
                )
    return violations


def _format_violation(violation: dict[str, object]) -> str:
    rule = str(violation.get("rule"))
    if rule.startswith("writer_module_"):
        detail = ""
        if "entrypoint" in violation:
            detail = f" entrypoint={violation['entrypoint']}"
        elif "declared" in violation:
            detail = f" declared={violation['declared']}"
        elif "contract" in violation:
            detail = f" contract={violation['contract']}"
        line = f":{violation['line']}" if "line" in violation else ""
        return f"  {violation['file']}{line}: {rule}{detail}"
    return f"  {violation['file']}: imports {violation['import']} ({violation['rule']})"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = _get_root()
    rules_path = repo_root / "docs" / "plans" / "layering.yaml"

    if not rules_path.exists():
        print(f"error: {rules_path} not found", file=sys.stderr)
        return 1

    manifest = _load_manifest(rules_path)
    rules = _load_rules(rules_path)
    violations: list[dict[str, object]] = []

    for rule in rules:
        target = str(rule["target"])
        target_dir = repo_root / target
        if not target_dir.exists():
            continue

        allow_block = rule.get("allow") or {}
        disallow_block = rule.get("disallow") or {}
        allow_from: list[str] = []
        disallow_from: list[str] = []
        if isinstance(allow_block, dict):
            af = allow_block.get("from")
            if isinstance(af, list):
                allow_from = [str(x) for x in af]
        if isinstance(disallow_block, dict):
            df = disallow_block.get("from")
            if isinstance(df, list):
                disallow_from = [str(x) for x in df]

        imports = _collect_imports(target_dir, repo_root=repo_root)
        for file_rel, file_imports in imports.items():
            for imp in file_imports:
                if not imp.startswith("polylogue"):
                    continue
                for disallowed in disallow_from:
                    if _package_matches(str(disallowed), imp):
                        violations.append(
                            {
                                "target": target,
                                "file": file_rel,
                                "import": imp,
                                "rule": "disallow",
                                "disallowed_package": disallowed,
                            }
                        )
                        break
                else:
                    if allow_from and not any(_package_matches(str(allowed), imp) for allowed in allow_from):
                        violations.append(
                            {
                                "target": target,
                                "file": file_rel,
                                "import": imp,
                                "rule": "not_allowed",
                                "allowed": allow_from,
                            }
                        )

    violations.extend(_collect_writer_module_violations(repo_root, _writer_module_policy(manifest)))

    if args.json:
        print(dumps({"violations": violations, "count": len(violations)}))
    else:
        for violation in violations:
            print(_format_violation(violation))
        if not violations:
            print("  No layering violations found.")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
