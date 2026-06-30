"""Render a declarative package of Polylogue read artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ReadPackageArtifact:
    name: str
    view: str
    output_format: str
    path: str
    spec: bool = False
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ReadPackageSpec:
    version: int
    artifacts: tuple[ReadPackageArtifact, ...]
    prune: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadPackagePlanItem:
    artifact: ReadPackageArtifact
    out_path: Path
    argv: tuple[str, ...]


def _expect_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _expect_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _expect_optional_positive_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_read_package_spec(path: Path) -> ReadPackageSpec:
    """Load a JSON/YAML read-package spec from disk."""

    raw_text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(raw_text) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(raw_text)
    data = _expect_mapping(raw, "read package")
    version = data.get("version")
    if version != 1:
        raise ValueError(f"read package version must be 1, got {version!r}")
    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("read package artifacts must be a non-empty list")
    artifacts = []
    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact_data = _expect_mapping(raw_artifact, f"artifacts[{index}]")
        artifact = ReadPackageArtifact(
            name=_expect_string(artifact_data.get("name"), f"artifacts[{index}].name"),
            view=_expect_string(artifact_data.get("view"), f"artifacts[{index}].view"),
            output_format=_expect_string(artifact_data.get("format"), f"artifacts[{index}].format"),
            path=_expect_string(artifact_data.get("path"), f"artifacts[{index}].path"),
            spec=bool(artifact_data.get("spec", False)),
            max_tokens=_expect_optional_positive_int(artifact_data.get("max_tokens"), f"artifacts[{index}].max_tokens"),
        )
        if artifact.name in seen_names:
            raise ValueError(f"duplicate artifact name: {artifact.name}")
        if artifact.path in seen_paths:
            raise ValueError(f"duplicate artifact path: {artifact.path}")
        if Path(artifact.path).is_absolute() or ".." in Path(artifact.path).parts:
            raise ValueError(f"artifact path must stay inside --out-dir: {artifact.path}")
        seen_names.add(artifact.name)
        seen_paths.add(artifact.path)
        artifacts.append(artifact)

    raw_prune = data.get("prune", [])
    if not isinstance(raw_prune, list):
        raise ValueError("read package prune must be a list")
    prune = []
    for index, raw_path in enumerate(raw_prune):
        rel = _expect_string(raw_path, f"prune[{index}]")
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"prune path must stay inside --out-dir: {rel}")
        prune.append(rel)
    return ReadPackageSpec(version=1, artifacts=tuple(artifacts), prune=tuple(prune))


def build_read_package_plan(
    spec: ReadPackageSpec,
    *,
    session_id: str,
    out_dir: Path,
    polylogue_bin: str = "polylogue",
) -> tuple[ReadPackagePlanItem, ...]:
    """Build concrete read commands for a package spec."""

    out_dir = out_dir.expanduser()
    plan = []
    for artifact in spec.artifacts:
        out_path = out_dir / artifact.path
        argv = (
            polylogue_bin,
            "--id",
            session_id,
            "read",
            "--view",
            artifact.view,
            "--format",
            artifact.output_format,
            *(("--spec",) if artifact.spec else ()),
            *(("--max-tokens", str(artifact.max_tokens)) if artifact.max_tokens is not None else ()),
            "--to",
            "file",
            "--out",
            str(out_path),
        )
        plan.append(ReadPackagePlanItem(artifact=artifact, out_path=out_path, argv=argv))
    return tuple(plan)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devtools workspace read-package",
        description="Render a declarative package of Polylogue read artifacts.",
    )
    parser.add_argument("--spec", type=Path, required=True, help="JSON/YAML read-package spec.")
    parser.add_argument("--session-id", required=True, help="Session id or native id accepted by `polylogue --id`.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated package artifacts.")
    parser.add_argument("--polylogue-bin", default=os.environ.get("POLYLOGUE_BIN", "polylogue"))
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running commands.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    return parser


def _summary(
    *,
    spec_path: Path,
    session_id: str,
    out_dir: Path,
    spec: ReadPackageSpec,
    plan: tuple[ReadPackagePlanItem, ...],
    pruned: list[str],
    dry_run: bool,
) -> dict[str, object]:
    return {
        "spec": str(spec_path),
        "session_id": session_id,
        "out_dir": str(out_dir),
        "version": spec.version,
        "dry_run": dry_run,
        "prune": list(spec.prune),
        "pruned": pruned,
        "artifacts": [
            {
                "name": item.artifact.name,
                "view": item.artifact.view,
                "format": item.artifact.output_format,
                "spec": item.artifact.spec,
                "max_tokens": item.artifact.max_tokens,
                "path": str(item.out_path),
                "argv": list(item.argv),
                "bytes": item.out_path.stat().st_size if item.out_path.exists() else None,
            }
            for item in plan
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        spec = load_read_package_spec(args.spec)
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"read-package: {exc}", file=sys.stderr)
        return 2

    out_dir = args.out_dir.expanduser()
    plan = build_read_package_plan(
        spec,
        session_id=args.session_id,
        out_dir=out_dir,
        polylogue_bin=args.polylogue_bin,
    )
    pruned = []
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel in spec.prune:
            target = out_dir / rel
            if target.exists():
                target.unlink()
                pruned.append(str(target))
        for item in plan:
            item.out_path.parent.mkdir(parents=True, exist_ok=True)
            child_stdout = sys.stderr if args.json else None
            subprocess.run(item.argv, check=True, stdout=child_stdout)

    summary = _summary(
        spec_path=args.spec,
        session_id=args.session_id,
        out_dir=out_dir,
        spec=spec,
        plan=plan,
        pruned=pruned,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        action = "planned" if args.dry_run else "rendered"
        print(f"{action} {len(plan)} read artifact(s) under {out_dir}")
        for item in plan:
            print(f"- {item.artifact.name}: {item.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
