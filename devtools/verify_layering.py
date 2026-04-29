"""Verify inter-package layering rules from docs/plans/layering.yaml.

Usage:
  devtools verify-layering
  devtools verify-layering --json
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from polylogue.lib.json import dumps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enforce inter-package layering rules.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    return parser


def _load_rules(rules_path: Path) -> list[dict[str, object]]:
    import yaml

    with open(rules_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("rules", [])


def _collect_imports(package_dir: Path) -> dict[str, set[str]]:
    imports: dict[str, set[str]] = {}
    for py_file in package_dir.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = str(py_file.relative_to(package_dir.parent))
        imports.setdefault(rel, set())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[rel].add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports[rel].add(node.module)
    return imports


def _package_matches(target: str, import_module: str) -> bool:
    """Check if import_module falls under the target package."""
    return import_module == target or import_module.startswith(target + ".")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    rules_path = repo_root / "docs" / "plans" / "layering.yaml"

    if not rules_path.exists():
        print(f"error: {rules_path} not found", file=sys.stderr)
        return 1

    rules = _load_rules(rules_path)
    violations: list[dict[str, object]] = []

    for rule in rules:
        target = str(rule["target"])
        target_dir = repo_root / target
        if not target_dir.exists():
            continue

        allow_from = rule.get("allow", {}).get("from", []) or []
        disallow_from = rule.get("disallow", {}).get("from", []) or []

        imports = _collect_imports(target_dir)
        for file_rel, file_imports in imports.items():
            for imp in file_imports:
                # Only check polylogue-internal imports
                module = imp
                if not module.startswith("polylogue"):
                    continue

                # Check disallow first
                for disallowed in disallow_from:
                    if _package_matches(str(disallowed), module):
                        violations.append(
                            {
                                "target": target,
                                "file": f"{target}/{file_rel}",
                                "import": module,
                                "rule": "disallow",
                                "disallowed_package": disallowed,
                            }
                        )
                        break
                else:
                    # If allow list is non-empty, check the import is allowed
                    if allow_from and not any(_package_matches(str(allowed), module) for allowed in allow_from):
                        violations.append(
                            {
                                "target": target,
                                "file": f"{target}/{file_rel}",
                                "import": module,
                                "rule": "not_allowed",
                                "allowed": allow_from,
                            }
                        )

    if args.json:
        print(dumps({"violations": violations, "count": len(violations)}))
    else:
        for v in violations:
            print(f"  {v['file']}: imports {v['import']} ({v['rule']})")
        if not violations:
            print("  No layering violations found.")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
