"""Enforce the provider_meta classification policy declared in
``docs/plans/provider-meta-policy.yaml``.

Production parsers, source-assembly modules, and the session-rebuild insight
loader are scanned for string-literal ``provider_meta`` key writes. Each
observed key must be declared in the policy manifest with a classification.
The lint fails on:

- Undeclared keys (new field that requires a manifest row).
- Stale tombstones (a key marked ``removed`` is still being written).
- Manifest entries pointing at a key not observed anywhere in the scanned
  paths (kept as a soft warning so we notice when a parser stops writing a
  field but the row was not retired to ``removed``).

The lint is intentionally syntactic: it grep-walks production code for
literal subscripts and ``.get(...)`` reads against the dedicated dict
variable names ``provider_meta`` and ``conv_meta`` used by the parsers.
The generic name ``meta`` is intentionally excluded because parsers reuse
it for upstream provider envelopes, not for our ``provider_meta`` write
surface. New fields that bypass these patterns will not be detected, and
that is intended: new ingestion paths still need to land via PR review.

Wired into ``devtools verify --quick``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from devtools import repo_root as _get_root

ROOT = _get_root()
MANIFEST_REL = "docs/plans/provider-meta-policy.yaml"

# Scan paths: production code that writes provider_meta dicts.
SCAN_ROOTS: tuple[str, ...] = (
    "polylogue/sources/parsers",
    "polylogue/sources/assembly_codex.py",
    "polylogue/sources/assembly_gemini.py",
    "polylogue/storage/insights/session/rebuild.py",
)

# Variable names that hold a provider_meta-shaped dict during parser
# construction. Restricted to the canonical authoring surface — the
# generic name ``meta`` is intentionally excluded because parsers also
# use it for upstream provider envelopes (e.g.
# ``attachment_from_meta(meta=...)``) where a ``.get()`` call reads from
# a foreign dict, not the provider_meta we own.
PROVIDER_META_VARS: frozenset[str] = frozenset({"provider_meta", "conv_meta"})

ALLOWED_CLASSIFICATIONS: frozenset[str] = frozenset({"provider-specific-retained", "raw-only", "promoted", "removed"})


@dataclass(frozen=True)
class ManifestField:
    key: str
    classification: str
    scope: str | None
    providers: tuple[str, ...]
    promoted_column: str | None
    note: str | None
    row_index: int  # for stable reporting


@dataclass(frozen=True)
class Observation:
    key: str
    path: str
    line: int


def load_manifest(path: Path) -> list[ManifestField]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: top-level must be a mapping")
    rows = data.get("fields")
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: missing `fields:` list")
    out: list[ManifestField] = []
    seen_keys: set[tuple[str, str | None]] = set()
    for idx, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            raise SystemExit(f"{path}: fields[{idx}] is not a mapping")
        key = raw_row.get("key")
        classification = raw_row.get("classification")
        if not isinstance(key, str) or not key:
            raise SystemExit(f"{path}: fields[{idx}].key must be a non-empty string")
        if classification not in ALLOWED_CLASSIFICATIONS:
            raise SystemExit(
                f"{path}: fields[{idx}].classification={classification!r} not in {sorted(ALLOWED_CLASSIFICATIONS)}"
            )
        scope = raw_row.get("scope")
        if scope is not None and scope not in {"session", "message", "attachment"}:
            raise SystemExit(f"{path}: fields[{idx}].scope={scope!r} must be session|message|attachment")
        dedup_key = (key, scope)
        if dedup_key in seen_keys:
            raise SystemExit(f"{path}: duplicate manifest entry for key={key!r} scope={scope!r}")
        seen_keys.add(dedup_key)
        providers_raw = raw_row.get("providers", [])
        providers: tuple[str, ...]
        if providers_raw is None:
            providers = ()
        elif isinstance(providers_raw, list):
            providers = tuple(str(item) for item in providers_raw)
        else:
            raise SystemExit(f"{path}: fields[{idx}].providers must be a list")
        out.append(
            ManifestField(
                key=key,
                classification=classification,
                scope=scope,
                providers=providers,
                promoted_column=(str(raw_row["promoted_column"]) if "promoted_column" in raw_row else None),
                note=str(raw_row["note"]) if raw_row.get("note") else None,
                row_index=idx,
            )
        )
    return out


def iter_scan_files(root: Path) -> Iterable[Path]:
    for rel in SCAN_ROOTS:
        target = root / rel
        if target.is_file():
            yield target
        elif target.is_dir():
            for path in target.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                yield path


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def collect_observations(path: Path, rel: str) -> list[Observation]:
    """Walk the AST and collect provider_meta key writes/reads.

    Patterns recognised:

    - ``provider_meta["KEY"] = ...``         (Subscript assignment)
    - ``provider_meta.get("KEY")``           (.get call)
    - ``{"KEY": value, ...}`` literal passed as ``provider_meta=`` kwarg
    - ``meta["KEY"] = ...`` / ``conv_meta[...] = ...`` (same patterns)

    Only string literal keys are observed; dynamic keys are deliberately
    skipped (they cannot be classified statically).
    """

    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return []

    observations: list[Observation] = []

    def _record(key: str, line: int) -> None:
        observations.append(Observation(key=key, path=rel, line=line))

    for node in ast.walk(tree):
        # provider_meta["KEY"] = ...   and   provider_meta["KEY"] inside expressions
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in PROVIDER_META_VARS:
            literal = _string_literal(node.slice)
            if literal is not None:
                _record(literal, node.lineno)

        # provider_meta.get("KEY", ...)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in PROVIDER_META_VARS
            and node.func.attr in {"get", "setdefault", "pop"}
            and node.args
        ):
            literal = _string_literal(node.args[0])
            if literal is not None:
                _record(literal, node.lineno)

        # provider_meta={"KEY": ...} as a keyword argument
        if isinstance(node, ast.keyword) and node.arg == "provider_meta":
            value = node.value
            if isinstance(value, ast.Dict):
                for k in value.keys:
                    if k is None:
                        continue
                    literal = _string_literal(k)
                    if literal is not None:
                        _record(literal, k.lineno)

    return observations


def lint(root: Path, manifest_path: Path) -> dict[str, Any]:
    fields = load_manifest(manifest_path)
    declared_by_key: dict[str, list[ManifestField]] = {}
    for field in fields:
        declared_by_key.setdefault(field.key, []).append(field)

    observations: list[Observation] = []
    for path in iter_scan_files(root):
        rel = path.relative_to(root).as_posix()
        observations.extend(collect_observations(path, rel))

    observed_keys: dict[str, list[Observation]] = {}
    for obs in observations:
        observed_keys.setdefault(obs.key, []).append(obs)

    undeclared: list[dict[str, Any]] = []
    revived_tombstones: list[dict[str, Any]] = []
    unobserved_declared: list[str] = []

    for key, obs_list in sorted(observed_keys.items()):
        declared = declared_by_key.get(key)
        if not declared:
            sample = obs_list[0]
            undeclared.append(
                {
                    "key": key,
                    "first_seen": f"{sample.path}:{sample.line}",
                    "write_count": len(obs_list),
                }
            )
            continue
        if all(field.classification == "removed" for field in declared):
            sample = obs_list[0]
            revived_tombstones.append(
                {
                    "key": key,
                    "first_seen": f"{sample.path}:{sample.line}",
                    "write_count": len(obs_list),
                }
            )

    for key, declared in sorted(declared_by_key.items()):
        if key in observed_keys:
            continue
        # If declared as `removed`, absence is expected; otherwise flag soft.
        if all(field.classification == "removed" for field in declared):
            continue
        unobserved_declared.append(key)

    blocking = bool(undeclared or revived_tombstones)

    return {
        "blocking": blocking,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "scanned_paths": list(SCAN_ROOTS),
        "declared_count": len(fields),
        "observed_key_count": len(observed_keys),
        "undeclared": undeclared,
        "revived_tombstones": revived_tombstones,
        "unobserved_declared": unobserved_declared,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST_REL)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = lint(ROOT, args.manifest)

    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        manifest = report["manifest"]
        print(f"provider_meta policy: {manifest}")
        print(f"    declared fields: {report['declared_count']}")
        print(f"    observed keys:   {report['observed_key_count']}")
        if report["undeclared"]:
            print(f"[BLOCK] undeclared provider_meta keys: {len(report['undeclared'])}")
            for row in report["undeclared"]:
                print(f"    {row['key']} (first at {row['first_seen']}, writes={row['write_count']})")
            print(
                "    Add each key to docs/plans/provider-meta-policy.yaml with a classification "
                "(provider-specific-retained | raw-only | promoted | removed)."
            )
        if report["revived_tombstones"]:
            print(
                f"[BLOCK] revived tombstones (manifest says removed but code still writes): "
                f"{len(report['revived_tombstones'])}"
            )
            for row in report["revived_tombstones"]:
                print(f"    {row['key']} (first at {row['first_seen']})")
        if report["unobserved_declared"]:
            print(f"[warn] manifest entries with no observed writes: {len(report['unobserved_declared'])}")
            for key in report["unobserved_declared"]:
                print(f"    {key}")
            print(
                "    Either restore the writer, retire the row with classification=removed, "
                "or accept the soft warning if the key is written behind a dynamic pattern."
            )
        if not report["undeclared"] and not report["revived_tombstones"] and not report["unobserved_declared"]:
            print("provider_meta policy: clean")
        print()
        print(f"blocking={report['blocking']}")

    return 1 if report["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
