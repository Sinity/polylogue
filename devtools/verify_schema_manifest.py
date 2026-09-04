"""Verify canonical SQLite schema manifests against archive tier files."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, canonical_schema_manifest, schema_manifest_diff

ROOT = Path(__file__).parents[1]
_DURABLE_TIERS = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
_MIGRATIONS_ROOT = "polylogue/storage/sqlite/migrations"
_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d{3,})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True, slots=True)
class _SchemaState:
    ddl: dict[ArchiveTier, str]
    versions: dict[ArchiveTier, int]


@dataclass(frozen=True, slots=True)
class _MigrationChange:
    status: str
    old_path: str
    new_path: str


def _current_schema_state() -> _SchemaState:
    return _SchemaState(
        ddl=dict(ARCHIVE_DDL_BY_TIER),
        versions={tier: int(version) for tier, version in ARCHIVE_VERSION_BY_TIER.items()},
    )


def _render_schema_state(ref: str | None) -> _SchemaState:
    """Render the effective archive state from a commit or this checkout."""
    if ref is None:
        return _current_schema_state()

    archive = subprocess.run(["git", "archive", ref], check=True, capture_output=True, cwd=ROOT).stdout
    scratch_root = "/realm/tmp/work"
    with tempfile.TemporaryDirectory(
        prefix="polylogue-schema-", dir=scratch_root if os.path.isdir(scratch_root) else None
    ) as checkout:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(checkout, filter="data")
        build_info = Path(checkout) / "polylogue" / "_build_info.py"
        build_info.write_text(f'BUILD_COMMIT = "{ref}"\nBUILD_DIRTY = False\n', encoding="utf-8")
        script = (
            "import json\n"
            "from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER\n"
            "print(json.dumps({'ddl': {tier.value: ddl for tier, ddl in ARCHIVE_DDL_BY_TIER.items()}, "
            "'versions': {tier.value: version for tier, version in ARCHIVE_VERSION_BY_TIER.items()}}))\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = checkout
        for name in ("_PYTHON_SYSCONFIGDATA_NAME", "_PYTHON_HOST_PLATFORM", "PYTHONHOME"):
            environment.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=checkout,
            env=environment,
            text=True,
        )
    payload = json.loads(result.stdout)
    return _SchemaState(
        ddl={ArchiveTier(tier): str(ddl) for tier, ddl in cast(dict[str, object], payload["ddl"]).items()},
        versions={
            ArchiveTier(tier): cast(int, version)
            for tier, version in cast(dict[str, object], payload["versions"]).items()
        },
    )


def _git_text(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=ROOT).stdout


def _merge_base(explicit_base: str | None = None) -> str:
    """Resolve a usable merge base without requiring an ``origin/master`` ref."""
    requested = explicit_base or os.environ.get("POLYLOGUE_SCHEMA_MERGE_BASE")
    if requested:
        try:
            base = _git_text("merge-base", "HEAD", requested).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"cannot determine a merge base from explicit ref {requested!r}: {exc}") from exc
        if base:
            return base
        raise RuntimeError(f"explicit schema comparison ref {requested!r} has no merge base with HEAD")

    candidates: list[str] = []
    github_base = os.environ.get("GITHUB_BASE_REF")
    if github_base:
        candidates.extend((f"origin/{github_base}", github_base))
    candidates.extend(("origin/master", "master", "origin/HEAD", "HEAD^"))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            base = _git_text("merge-base", "HEAD", candidate).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if base:
            return base
    try:
        return _git_text("rev-parse", "--verify", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot determine a schema comparison base: {exc}") from exc


def _migration_changes(base: str, tier: ArchiveTier) -> tuple[_MigrationChange, ...]:
    prefix = f"{_MIGRATIONS_ROOT}/{tier.value}/"
    changes: list[_MigrationChange] = []
    output = _git_text("diff", "--name-status", "--find-renames", base, "--", prefix)
    for line in output.splitlines():
        if not line:
            continue
        fields = line.split("\t")
        status = fields[0]
        if status.startswith(("R", "C")) and len(fields) >= 3:
            changes.append(_MigrationChange(status, fields[1], fields[2]))
        elif len(fields) >= 2:
            changes.append(_MigrationChange(status, fields[1], fields[1]))

    status_output = _git_text("status", "--porcelain=v1", "--untracked-files=all", "--", prefix)
    for line in status_output.splitlines():
        if line.startswith("?? "):
            path = line[3:]
            if path.startswith(prefix):
                changes.append(_MigrationChange("A", path, path))
    return tuple(changes)


def _migration_version(path: str, tier: ArchiveTier) -> int | None:
    prefix = f"{_MIGRATIONS_ROOT}/{tier.value}/"
    if not path.startswith(prefix):
        return None
    match = _MIGRATION_NAME_RE.fullmatch(path[len(prefix) :])
    return int(match.group("version")) if match else None


def _migration_integrity_violations(base: str, tier: ArchiveTier) -> list[str]:
    """Reject deletion or modification of an existing durable SQL migration."""
    violations: list[str] = []
    for change in _migration_changes(base, tier):
        if not (change.old_path.endswith(".sql") or change.new_path.endswith(".sql")):
            continue
        if change.status.startswith("A"):
            if _migration_version(change.new_path, tier) is None:
                violations.append(f"{tier.value}: added migration has an invalid numbered name: {change.new_path}")
        elif change.status.startswith("D"):
            violations.append(f"{tier.value}: required migration was deleted: {change.old_path}")
        elif change.status.startswith(("M", "R", "C")):
            violations.append(f"{tier.value}: required migration was modified: {change.old_path}")
    return violations


def _added_migration_versions(base: str, tier: ArchiveTier) -> tuple[dict[int, tuple[str, ...]], list[str]]:
    versions: dict[int, list[str]] = {}
    invalid: list[str] = []
    for change in _migration_changes(base, tier):
        if not change.status.startswith("A") or not change.new_path.endswith(".sql"):
            continue
        version = _migration_version(change.new_path, tier)
        if version is None:
            invalid.append(change.new_path)
        else:
            versions.setdefault(version, []).append(change.new_path)
    return {version: tuple(paths) for version, paths in versions.items()}, invalid


def _durable_ddl_evolution_violations(explicit_base: str | None = None) -> list[str]:
    """Require effective durable schema changes to use an exact migration chain."""
    base = _merge_base(explicit_base)
    previous = _render_schema_state(base)
    current = _render_schema_state(None)
    violations: list[str] = []

    for tier in _DURABLE_TIERS:
        violations.extend(_migration_integrity_violations(base, tier))
        old_version = previous.versions.get(tier)
        new_version = current.versions.get(tier)
        old_ddl = previous.ddl.get(tier)
        new_ddl = current.ddl.get(tier)
        if old_version is None or new_version is None or old_ddl is None or new_ddl is None:
            violations.append(f"{tier.value}: durable schema state is missing from the comparison")
            continue

        added_by_version, invalid_added = _added_migration_versions(base, tier)
        added_versions = set(added_by_version)
        violations.extend(
            f"{tier.value}: added migration has an invalid numbered name: {path}" for path in invalid_added
        )
        for version, paths in sorted(added_by_version.items()):
            if len(paths) > 1:
                violations.append(f"{tier.value}: multiple added migrations claim v{version}: {', '.join(paths)}")
        if new_version < old_version:
            violations.append(f"{tier.value}: schema version moved backwards from v{old_version} to v{new_version}")
        elif new_version != old_version:
            expected = set(range(old_version + 1, new_version + 1))
            missing = sorted(expected - added_versions)
            unexpected = sorted(added_versions - expected)
            if missing:
                violations.append(
                    f"{tier.value}: schema version bump v{old_version}->v{new_version} is missing added migrations "
                    f"for {', '.join(f'v{version}' for version in missing)}"
                )
            if unexpected:
                violations.append(
                    f"{tier.value}: added migration numbers {unexpected} are not the contiguous "
                    f"v{old_version + 1}->v{new_version} chain"
                )
        elif added_versions:
            violations.append(
                f"{tier.value}: added durable migrations without a schema-version bump: {sorted(added_versions)}"
            )

        if old_ddl != new_ddl and old_version == new_version:
            violations.append(f"{tier.value}: rendered DDL changed without a schema-version bump")
    return violations


def _check_tier(tier: ArchiveTier, path: Path | None) -> dict[str, Any]:
    expected = canonical_schema_manifest(tier)
    result: dict[str, Any] = {"tier": tier.value, "version": expected.version, "ok": True}
    if path is None or not path.exists():
        return result
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as conn:
        actual = SchemaManifest.from_connection(conn, tier)
    diff = schema_manifest_diff(expected, actual)
    if actual.version != expected.version:
        diff["version"] = {"expected": expected.version, "actual": actual.version}
    result["ok"] = not any(diff.values())
    if not result["ok"]:
        result["diff"] = diff
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify canonical archive SQLite schema manifests.")
    parser.add_argument("--archive-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check-evolution", action="store_true")
    parser.add_argument("--base", default=None, help="Explicit ref from which to compute the schema merge base.")
    args = parser.parse_args(argv)
    if args.check_evolution:
        try:
            violations = _durable_ddl_evolution_violations(args.base)
        except (
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            UnicodeError,
            json.JSONDecodeError,
            tarfile.TarError,
        ) as exc:
            violations = [f"cannot compare durable schema evolution: {exc}"]
        payload = {"kind": "polylogue.durable-schema-evolution", "ok": not violations, "violations": violations}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            for violation in violations:
                print(f"FAIL: {violation}")
            print("durable-schema-evolution: PASS" if not violations else "durable-schema-evolution: FAIL")
        return 0 if not violations else 1
    results = []
    for tier in ArchiveTier:
        path = args.archive_root / f"{tier.value}.db" if args.archive_root is not None else None
        results.append(_check_tier(tier, path))
    payload = {"kind": "polylogue.schema-manifest", "ok": all(item["ok"] for item in results), "tiers": results}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for item in results:
            print(f"{item['tier']}: {'PASS' if item['ok'] else 'FAIL'} (v{item['version']})")
        print("schema-manifest: PASS" if payload["ok"] else "schema-manifest: FAIL")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
