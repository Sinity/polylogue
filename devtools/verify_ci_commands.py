"""Validate catalogued devtools commands in structured CI run fields."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from devtools import repo_root
from devtools.command_catalog import COMMAND_SPECS, command_name_from_tokens


def _run_scripts(value: object) -> Iterator[str]:
    """Yield executable scripts from YAML ``run`` fields, never prose fields."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "run":
                if isinstance(child, str):
                    yield child
                elif isinstance(child, Mapping) and isinstance(child.get("command"), str):
                    yield child["command"]
            yield from _run_scripts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _run_scripts(child)


def _shell_tokens(script: str) -> tuple[str, ...]:
    lexer = shlex.shlex(script, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return tuple(lexer)
    except ValueError:
        return ()


def _invocations(script: str) -> Iterator[tuple[str, ...]]:
    """Yield argv tails following exact ``devtools`` executable tokens."""
    tokens = _shell_tokens(script.replace("\\\n", " "))
    separators = {";", "&", "&&", "|", "||", "(", ")"}
    for index, token in enumerate(tokens):
        if Path(token).name != "devtools":
            continue
        tail: list[str] = []
        for candidate in tokens[index + 1 :]:
            if candidate in separators:
                break
            tail.append(candidate)
        yield tuple(tail)


def _unknown_command(argv: Sequence[str]) -> str | None:
    if not argv or argv[0].startswith("-"):
        return None
    matched = command_name_from_tokens(argv)
    if matched is None:
        return argv[0]
    matched_path = next(spec.command_path for spec in COMMAND_SPECS if spec.name == matched)
    remaining = argv[len(matched_path) :]
    has_subcommands = any(
        len(spec.command_path) > len(matched_path) and spec.command_path[: len(matched_path)] == matched_path
        for spec in COMMAND_SPECS
    )
    if has_subcommands and remaining and not remaining[0].startswith("-"):
        return " ".join((*matched_path, remaining[0]))
    return None


def validate_ci_commands(root: Path) -> tuple[str, ...]:
    """Return parse and command errors from GitHub Actions and CircleCI YAML."""
    paths = sorted((root / ".github" / "workflows").glob("*.yml"))
    circle = root / ".circleci" / "config.yml"
    if circle.exists():
        paths.append(circle)
    errors: list[str] = []
    for path in paths:
        relative = path.relative_to(root)
        try:
            document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            errors.append(f"{relative}: invalid YAML: {exc}")
            continue
        for script in _run_scripts(document):
            for argv in _invocations(script):
                unknown = _unknown_command(argv)
                if unknown is not None:
                    errors.append(f"{relative}: unknown devtools command {unknown!r}")
    return tuple(errors)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = repo_root()
    errors = validate_ci_commands(root)
    if args.json:
        print(json.dumps({"blocking": bool(errors), "errors": list(errors)}, indent=2))
    elif errors:
        for error in errors:
            print(f"[BLOCK] {error}")
    else:
        print("CI devtools commands match the live command catalog")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
