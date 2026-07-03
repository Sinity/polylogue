"""Render a declarative package of Polylogue read artifacts."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TextIO

import yaml

from polylogue.paths import archive_root


@dataclass(frozen=True, slots=True)
class ReadPackageProjection:
    max_tokens: int | None = None
    body_full: bool = False
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
                "body_full": self.body_full if self.body_full else None,
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
        if self.body_full and self.body_limit is not None:
            raise ValueError("projection may not set both body_full and body_limit")
        args: list[str] = []
        if self.max_tokens is not None:
            args.extend(("--max-tokens", str(self.max_tokens)))
        if self.body_full:
            args.append("--full")
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


ReadPackageRunner = Literal["in-process", "subprocess"]


@dataclass(frozen=True, slots=True)
class ReadPackageArtifactResult:
    status: str
    duration_ms: float | None
    bytes: int | None

    def as_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "duration_ms": self.duration_ms,
            "bytes": self.bytes,
        }


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_from_ms(value: object) -> str | None:
    if value is None:
        return None
    timestamp_ms = int(value) if isinstance(value, int | float | str | bytes | bytearray) else int(str(value))
    return (
        dt.datetime.fromtimestamp(timestamp_ms / 1000, tz=dt.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _read_package_freshness(session_ref: str, *, generated_at: str) -> dict[str, object]:
    """Return archive freshness metadata for a generated read package."""

    root = archive_root().expanduser()
    index_db = root / "index.db"
    payload: dict[str, object] = {
        "generated_at": generated_at,
        "archive_root": str(root),
        "archive_cursor": {
            "index_db": str(index_db),
            "index_schema_version": None,
        },
        "source_session": None,
        "successor_count": 0,
        "successors": [],
        "state": "archive_unavailable",
        "warnings": [],
    }
    if not index_db.exists():
        payload["warnings"] = ["index_db_missing"]
        return payload

    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            schema_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            payload["archive_cursor"] = {
                "index_db": str(index_db),
                "index_schema_version": schema_version,
            }
            session = conn.execute(
                """
                SELECT session_id, native_id, origin, title, created_at_ms, updated_at_ms,
                       session_kind, parent_session_id, root_session_id
                FROM sessions
                WHERE session_id = ? OR native_id = ?
                ORDER BY CASE WHEN session_id = ? THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (session_ref, session_ref, session_ref),
            ).fetchone()
            if session is None:
                payload["state"] = "session_not_found"
                payload["warnings"] = ["session_not_found"]
                return payload

            session_id = str(session["session_id"])
            source_updated_at = _iso_from_ms(session["updated_at_ms"])
            payload["source_session"] = {
                "session_id": session_id,
                "native_id": str(session["native_id"]),
                "origin": str(session["origin"]),
                "title": str(session["title"]) if session["title"] is not None else None,
                "session_kind": str(session["session_kind"] or "standard"),
                "created_at": _iso_from_ms(session["created_at_ms"]),
                "updated_at": source_updated_at,
                "parent_session_id": (
                    str(session["parent_session_id"]) if session["parent_session_id"] is not None else None
                ),
                "root_session_id": str(session["root_session_id"]) if session["root_session_id"] is not None else None,
            }
            count_row = conn.execute(
                """
                WITH successor_ids(session_id) AS (
                    SELECT child.session_id
                    FROM sessions child
                    WHERE child.parent_session_id = ?
                    UNION
                    SELECT links.src_session_id
                    FROM session_links links
                    WHERE links.resolved_dst_session_id = ?
                )
                SELECT COUNT(*) AS count FROM successor_ids
                """,
                (session_id, session_id),
            ).fetchone()
            successor_count = int(count_row["count"] if count_row is not None else 0)
            payload["successor_count"] = successor_count
            successor_rows = conn.execute(
                """
                WITH successors(session_id, relation, link_type, inheritance, observed_at_ms, resolved_at_ms) AS (
                    SELECT child.session_id, 'parent_session_id', child.branch_type, NULL, NULL, NULL
                    FROM sessions child
                    WHERE child.parent_session_id = ?
                    UNION
                    SELECT links.src_session_id, 'session_links', links.link_type, links.inheritance,
                           links.observed_at_ms, links.resolved_at_ms
                    FROM session_links links
                    WHERE links.resolved_dst_session_id = ?
                )
                SELECT child.session_id, child.native_id, child.origin, child.title,
                       child.created_at_ms, child.updated_at_ms, child.session_kind,
                       json_group_array(DISTINCT successors.relation) AS relations_json,
                       json_group_array(DISTINCT successors.link_type)
                           FILTER (WHERE successors.link_type IS NOT NULL) AS link_types_json,
                       json_group_array(DISTINCT successors.inheritance)
                           FILTER (WHERE successors.inheritance IS NOT NULL) AS inheritance_json,
                       MIN(successors.observed_at_ms) AS first_observed_at_ms,
                       MAX(successors.resolved_at_ms) AS latest_resolved_at_ms
                FROM successors
                JOIN sessions child ON child.session_id = successors.session_id
                GROUP BY child.session_id
                ORDER BY COALESCE(child.updated_at_ms, child.created_at_ms, 0) DESC, child.session_id
                LIMIT 20
                """,
                (session_id, session_id),
            ).fetchall()
            successors: list[dict[str, object]] = []
            for row in successor_rows:
                relations = [str(value) for value in json.loads(str(row["relations_json"] or "[]")) if value]
                link_types = [str(value) for value in json.loads(str(row["link_types_json"] or "[]")) if value]
                inheritance = [str(value) for value in json.loads(str(row["inheritance_json"] or "[]")) if value]
                successors.append(
                    {
                        "session_id": str(row["session_id"]),
                        "native_id": str(row["native_id"]),
                        "origin": str(row["origin"]),
                        "title": str(row["title"]) if row["title"] is not None else None,
                        "session_kind": str(row["session_kind"] or "standard"),
                        "created_at": _iso_from_ms(row["created_at_ms"]),
                        "updated_at": _iso_from_ms(row["updated_at_ms"]),
                        "relations": relations,
                        "link_types": link_types,
                        "inheritance": inheritance,
                        "first_observed_at": _iso_from_ms(row["first_observed_at_ms"]),
                        "latest_resolved_at": _iso_from_ms(row["latest_resolved_at_ms"]),
                    }
                )
            payload["successors"] = successors
            latest_successor_updated_at = next(
                (str(item["updated_at"]) for item in successors if item.get("updated_at") is not None),
                None,
            )
            payload["latest_successor_updated_at"] = latest_successor_updated_at
            if successor_count:
                payload["state"] = "successors_present"
                warnings = ["successors_present"]
                if successor_count > len(successors):
                    warnings.append("successor_list_truncated")
                payload["warnings"] = warnings
            else:
                payload["state"] = "current_at_generation"
                payload["warnings"] = []
            payload["has_later_successors"] = bool(
                latest_successor_updated_at is not None
                and source_updated_at is not None
                and latest_successor_updated_at > source_updated_at
            )
            return payload
    except sqlite3.Error as exc:
        payload["state"] = "archive_read_failed"
        payload["warnings"] = [f"archive_read_failed:{exc.__class__.__name__}"]
        return payload


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


def _expect_optional_bool(value: object, label: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _read_projection(data: dict[str, Any], label: str) -> ReadPackageProjection:
    if "max_tokens" in data:
        raise ValueError(f"{label}.max_tokens moved under {label}.projection.max_tokens")
    raw = data.get("projection", {})
    projection = _expect_mapping(raw, f"{label}.projection")
    allowed = {
        "max_tokens",
        "body_full",
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
        body_full=_expect_optional_bool(projection.get("body_full"), f"{label}.projection.body_full"),
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
    parser.add_argument(
        "--runner",
        choices=("in-process", "subprocess"),
        default="in-process",
        help="Artifact execution strategy. In-process avoids one Python cold start per artifact.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without running commands.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    return parser


def _planned_artifact_results(plan: tuple[ReadPackagePlanItem, ...]) -> dict[str, ReadPackageArtifactResult]:
    return {
        item.artifact.name: ReadPackageArtifactResult(status="planned", duration_ms=None, bytes=None) for item in plan
    }


def _run_artifact_command(
    item: ReadPackagePlanItem,
    *,
    runner: ReadPackageRunner,
    progress_stream: TextIO,
) -> ReadPackageArtifactResult:
    started = time.perf_counter()
    print(f"[read-package] {item.artifact.name}: rendering {item.out_path}", file=progress_stream)
    if runner == "in-process":
        from polylogue.cli.click_app import cli

        with contextlib.redirect_stdout(progress_stream), contextlib.redirect_stderr(progress_stream):
            cli.main(args=list(item.argv[1:]), prog_name=item.argv[0], standalone_mode=False)
    else:
        subprocess.run(item.argv, check=True, stdout=progress_stream, stderr=progress_stream)
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    if not item.out_path.exists():
        raise RuntimeError(f"read-package artifact {item.artifact.name!r} did not write {item.out_path}")
    size = item.out_path.stat().st_size
    print(f"[read-package] {item.artifact.name}: wrote {size} byte(s) in {duration_ms:.1f} ms", file=progress_stream)
    return ReadPackageArtifactResult(status="rendered", duration_ms=duration_ms, bytes=size)


def _summary(
    *,
    spec_path: Path,
    session_id: str,
    out_dir: Path,
    spec: ReadPackageSpec,
    plan: tuple[ReadPackagePlanItem, ...],
    pruned: list[str],
    dry_run: bool,
    generated_at: str,
    artifact_results: dict[str, ReadPackageArtifactResult],
) -> dict[str, object]:
    return {
        "generated_at": generated_at,
        "spec": str(spec_path),
        "session_id": session_id,
        "out_dir": str(out_dir),
        "version": spec.version,
        "dry_run": dry_run,
        "prune": list(spec.prune),
        "pruned": pruned,
        "freshness": _read_package_freshness(session_id, generated_at=generated_at),
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
                **artifact_results[item.artifact.name].as_payload(),
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
    artifact_results = _planned_artifact_results(plan)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for rel in spec.prune:
            target = out_dir / rel
            if target.exists():
                target.unlink()
                pruned.append(str(target))
        progress_stream = sys.stderr if args.json else sys.stdout
        for item in plan:
            item.out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                artifact_results[item.artifact.name] = _run_artifact_command(
                    item,
                    runner=args.runner,
                    progress_stream=progress_stream,
                )
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                print(f"read-package: {exc}", file=sys.stderr)
                return 1

    generated_at = _utc_now_iso()
    summary = _summary(
        spec_path=args.spec,
        session_id=args.session_id,
        out_dir=out_dir,
        spec=spec,
        plan=plan,
        pruned=pruned,
        dry_run=args.dry_run,
        generated_at=generated_at,
        artifact_results=artifact_results,
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        action = "planned" if args.dry_run else "rendered"
        print(f"{action} {len(plan)} read artifact(s) under {out_dir}")
        freshness = summary["freshness"]
        if isinstance(freshness, dict):
            warnings = freshness.get("warnings")
            state = freshness.get("state")
            successor_count = freshness.get("successor_count")
            if warnings:
                print(f"! freshness: {state} ({successor_count} successor(s)); see --json for details")
        for item in plan:
            print(f"- {item.artifact.name}: {item.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
