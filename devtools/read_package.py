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
class ReadPackageProjection:
    max_tokens: int | None = None
    body_limit: int | None = None
    body_offset: int | None = None
    edge_limit: int | None = None
    neighbor_limit: int | None = None
    neighbor_window_hours: int | None = None

    def as_payload(self) -> dict[str, int]:
        return {
            key: value
            for key, value in {
                "max_tokens": self.max_tokens,
                "body_limit": self.body_limit,
                "body_offset": self.body_offset,
                "edge_limit": self.edge_limit,
                "neighbor_limit": self.neighbor_limit,
                "neighbor_window_hours": self.neighbor_window_hours,
            }.items()
            if value is not None
        }

    def cli_args(self) -> tuple[str, ...]:
        limit_values = tuple(
            value for value in (self.body_limit, self.edge_limit, self.neighbor_limit) if value is not None
        )
        if len(limit_values) > 1:
            raise ValueError("projection may set only one of body_limit, edge_limit, or neighbor_limit")
        args: list[str] = []
        if self.max_tokens is not None:
            args.extend(("--max-tokens", str(self.max_tokens)))
        if limit_values:
            args.extend(("--limit", str(limit_values[0])))
        if self.body_offset is not None:
            args.extend(("--offset", str(self.body_offset)))
        if self.neighbor_window_hours is not None:
            args.extend(("--window-hours", str(self.neighbor_window_hours)))
        return tuple(args)


@dataclass(frozen=True, slots=True)
class ReadPackageRender:
    fields: str | None = None

    def as_payload(self) -> dict[str, str]:
        return {"fields": self.fields} if self.fields is not None else {}

    def cli_args(self) -> tuple[str, ...]:
        if self.fields is None:
            return ()
        return ("--fields", self.fields)


@dataclass(frozen=True, slots=True)
class ReadPackageArtifact:
    name: str
    view: str
    output_format: str
    path: str
    spec: bool = False
    projection: ReadPackageProjection = ReadPackageProjection()
    render: ReadPackageRender = ReadPackageRender()


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


def _read_projection(data: dict[str, Any], label: str) -> ReadPackageProjection:
    if "max_tokens" in data:
        raise ValueError(f"{label}.max_tokens moved under {label}.projection.max_tokens")
    raw = data.get("projection", {})
    projection = _expect_mapping(raw, f"{label}.projection")
    allowed = {
        "max_tokens",
        "body_limit",
        "body_offset",
        "edge_limit",
        "neighbor_limit",
        "neighbor_window_hours",
    }
    unknown = sorted(set(projection) - allowed)
    if unknown:
        raise ValueError(f"{label}.projection has unsupported key(s): {', '.join(unknown)}")
    result = ReadPackageProjection(
        max_tokens=_expect_optional_positive_int(projection.get("max_tokens"), f"{label}.projection.max_tokens"),
        body_limit=_expect_optional_positive_int(projection.get("body_limit"), f"{label}.projection.body_limit"),
        body_offset=_expect_optional_positive_int(projection.get("body_offset"), f"{label}.projection.body_offset"),
        edge_limit=_expect_optional_positive_int(projection.get("edge_limit"), f"{label}.projection.edge_limit"),
        neighbor_limit=_expect_optional_positive_int(
            projection.get("neighbor_limit"), f"{label}.projection.neighbor_limit"
        ),
        neighbor_window_hours=_expect_optional_positive_int(
            projection.get("neighbor_window_hours"), f"{label}.projection.neighbor_window_hours"
        ),
    )
    result.cli_args()
    return result


def _read_render(data: dict[str, Any], label: str) -> ReadPackageRender:
    if "fields" in data:
        raise ValueError(f"{label}.fields moved under {label}.render.fields")
    raw = data.get("render", {})
    render = _expect_mapping(raw, f"{label}.render")
    allowed = {"fields"}
    unknown = sorted(set(render) - allowed)
    if unknown:
        raise ValueError(f"{label}.render has unsupported key(s): {', '.join(unknown)}")
    fields = render.get("fields")
    if fields is not None:
        fields = _expect_string(fields, f"{label}.render.fields")
    return ReadPackageRender(fields=fields)


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
            projection=_read_projection(artifact_data, f"artifacts[{index}]"),
            render=_read_render(artifact_data, f"artifacts[{index}]"),
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
            *artifact.projection.cli_args(),
            *artifact.render.cli_args(),
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
                "projection": item.artifact.projection.as_payload(),
                "render": item.artifact.render.as_payload(),
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
