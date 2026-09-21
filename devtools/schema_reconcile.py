"""Reconcile one provider-schema generation pass into an aggregate matrix.

Reads the run receipts a generation pass wrote, derives the provider
denominator independently of those receipts and of the committed packages, and
prints one recorded outcome per declared subject bound to the baseline digest,
the provider declaration digest, the code revision, the generator semantics
revision and the resolved inference configuration.

Exits non-zero when any subject lands on a blocking outcome -- a failure, an
unexplained zero sample count, narrowing, or a subject the pass never reached.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.privacy import PUBLISHABLE_VOCABULARY_ROLES
from polylogue.schemas.privacy_config import PrivacyConfigSection, load_privacy_config
from polylogue.schemas.provider_reconciliation import (
    ProviderMatrix,
    load_receipts,
    reconcile_provider_matrix,
)
from polylogue.schemas.source_frontier import SchemaFrontierError, check_frontier, load_frontier

REPO_ROOT = Path(__file__).resolve().parents[1]


def _code_revision(explicit: str | None) -> str:
    if explicit:
        return explicit
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _inference_configuration(privacy: str | None, privacy_config_path: Path | None) -> JSONDocument:
    """Resolve the privacy declaration the pass ran under, never a summary of it.

    The resolved values are recorded rather than the preset name alone, so a
    later change to a preset cannot silently re-describe what this pass did.
    """

    overrides: PrivacyConfigSection = {"level": privacy} if privacy else {}
    config = load_privacy_config(
        cli_overrides=overrides,
        project_path=privacy_config_path.parent if privacy_config_path else None,
    )
    payload: JSONDocument = {
        "privacy_level": config.level,
        "safe_enum_max_length": config.safe_enum_max_length,
        "high_entropy_min_length": config.high_entropy_min_length,
        "source_selection": "declared_frontier",
    }
    payload["field_overrides"] = _string_map(config.field_overrides)
    payload["allow_value_patterns"] = _json_strings(config.allow_value_patterns)
    payload["deny_value_patterns"] = _json_strings(config.deny_value_patterns)
    payload["publishable_vocabulary_roles"] = _json_strings(PUBLISHABLE_VOCABULARY_ROLES)
    return payload


def _json_strings(values: Iterable[str]) -> list[JSONValue]:
    items: list[JSONValue] = []
    items.extend(sorted(values))
    return items


def _string_map(values: dict[str, str]) -> JSONDocument:
    payload: JSONDocument = {}
    for key, value in sorted(values.items()):
        payload[key] = value
    return payload


def _load_receipt_payloads(directory: Path) -> list[JSONDocument]:
    payloads: list[JSONDocument] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"run receipt {path} is not an object")
        payloads.append(payload)
    return payloads


def _write_matrix(matrix: ProviderMatrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(matrix.to_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconcile a provider schema generation pass.")
    parser.add_argument("--receipts", type=Path, required=True, help="Directory of per-subject run receipts.")
    parser.add_argument("--frontier", type=Path, default=None, help="Frontier document (default: archive state).")
    parser.add_argument("--code-revision", default=None, help="Revision the pass ran at (default: git HEAD).")
    parser.add_argument("--privacy", default=None, help="Privacy preset the pass declared.")
    parser.add_argument("--privacy-config", type=Path, default=None, help="Privacy config overrides the pass used.")
    parser.add_argument("--write", type=Path, default=None, help="Write the matrix document to this path.")
    parser.add_argument("--json", action="store_true", help="Output as JSON.")
    args = parser.parse_args(argv)

    try:
        frontier = load_frontier(args.frontier)
        check = check_frontier(frontier)
    except SchemaFrontierError as exc:
        print(f"schema-reconcile: {exc}", file=sys.stderr)
        return 1

    try:
        receipts = load_receipts(_load_receipt_payloads(args.receipts))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"schema-reconcile: {exc}", file=sys.stderr)
        return 1

    matrix = reconcile_provider_matrix(
        frontier=frontier,
        check=check,
        receipts=receipts,
        code_revision=_code_revision(args.code_revision),
        inference_configuration=_inference_configuration(args.privacy, args.privacy_config),
    )

    if args.write is not None:
        _write_matrix(matrix, args.write)

    if args.json:
        print(json.dumps(matrix.to_payload(), indent=2, sort_keys=True))
    else:
        print(matrix.format_text())

    return 0 if matrix.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
