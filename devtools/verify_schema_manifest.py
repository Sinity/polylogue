"""Verify canonical SQLite schema manifests against archive tier files."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
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

from polylogue.storage.archive_identity import ArchiveLocation as ArchiveLocation
from polylogue.storage.sqlite.archive_tiers import (
    ARCHIVE_DDL_BY_TIER,
    ARCHIVE_FORMAT_FLOOR_VERSION,
    ARCHIVE_VERSION_BY_TIER,
)
from polylogue.storage.sqlite.archive_tiers.archive_plan import ARCHIVE_FORMAT_LINEAGE
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.migration_runner import MigrationError, durable_migration_claims
from polylogue.storage.sqlite.schema_manifest import SchemaManifest, canonical_schema_manifest, schema_manifest_diff

ROOT = Path(__file__).parents[1]
_DURABLE_TIERS = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
_MIGRATIONS_ROOT = "polylogue/storage/sqlite/migrations"
_MIGRATION_NAME_RE = re.compile(r"^(?P<version>\d{3,})_[a-z0-9_]+\.sql$")


@dataclass(frozen=True, slots=True)
class _SchemaState:
    ddl: dict[ArchiveTier, str]
    versions: dict[ArchiveTier, int]
    lineage: str = ARCHIVE_FORMAT_LINEAGE


@dataclass(frozen=True, slots=True)
class _MigrationChange:
    status: str
    old_path: str
    new_path: str


def _current_schema_state() -> _SchemaState:
    return _SchemaState(
        ddl=dict(ARCHIVE_DDL_BY_TIER),
        versions={tier: int(version) for tier, version in ARCHIVE_VERSION_BY_TIER.items()},
        lineage=ARCHIVE_FORMAT_LINEAGE,
    )


#: Rendered schema states of past commits. A commit's tree cannot change, so a
#: rendering keyed by commit, renderer and interpreter is computed once per
#: checkout instead of on every gate run; rendering a commit extracts and
#: cold-imports its whole package, which is most of this gate's cost.
_SCHEMA_STATE_CACHE = ROOT / ".cache" / "verify" / "schema-state"

_RENDER_SCRIPT = (
    "import json\n"
    "from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER, ARCHIVE_VERSION_BY_TIER\n"
    "from polylogue.storage.sqlite.archive_tiers.archive_plan import ARCHIVE_FORMAT_LINEAGE\n"
    "print(json.dumps({'ddl': {tier.value: ddl for tier, ddl in ARCHIVE_DDL_BY_TIER.items()}, "
    "'versions': {tier.value: version for tier, version in ARCHIVE_VERSION_BY_TIER.items()}, "
    "'lineage': ARCHIVE_FORMAT_LINEAGE}))\n"
)


def _render_schema_state(ref: str | None) -> _SchemaState:
    """Render the effective archive state from a commit or this checkout."""
    if ref is None:
        return _current_schema_state()
    commit = _git_text("rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    key = hashlib.sha256(f"{commit}\0{sys.version}\0{_RENDER_SCRIPT}".encode()).hexdigest()
    cached = _SCHEMA_STATE_CACHE / f"{key}.json"
    try:
        payload = json.loads(cached.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = _render_commit_payload(commit)
        with contextlib.suppress(OSError):
            _SCHEMA_STATE_CACHE.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=_SCHEMA_STATE_CACHE, suffix=".tmp", delete=False
            ) as handle:
                json.dump(payload, handle)
            os.replace(handle.name, cached)
    return _SchemaState(
        ddl={ArchiveTier(tier): str(ddl) for tier, ddl in cast(dict[str, object], payload["ddl"]).items()},
        versions={
            ArchiveTier(tier): cast(int, version)
            for tier, version in cast(dict[str, object], payload["versions"]).items()
        },
        lineage=str(payload["lineage"]),
    )


def _render_commit_payload(ref: str) -> dict[str, Any]:
    """Extract *ref* to scratch and print its effective schema from a fresh interpreter."""
    archive = subprocess.run(["git", "archive", ref], check=True, capture_output=True, cwd=ROOT).stdout
    scratch_root = "/realm/tmp/work"
    with tempfile.TemporaryDirectory(
        prefix="polylogue-schema-", dir=scratch_root if os.path.isdir(scratch_root) else None
    ) as checkout:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(checkout, filter="data")
        build_info = Path(checkout) / "polylogue" / "_build_info.py"
        build_info.write_text(f'BUILD_COMMIT = "{ref}"\nBUILD_DIRTY = False\n', encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = checkout
        for name in ("_PYTHON_SYSCONFIGDATA_NAME", "_PYTHON_HOST_PLATFORM", "PYTHONHOME"):
            environment.pop(name, None)
        result = subprocess.run(
            [sys.executable, "-c", _RENDER_SCRIPT],
            check=True,
            capture_output=True,
            cwd=checkout,
            env=environment,
            text=True,
        )
    return cast(dict[str, Any], json.loads(result.stdout))


def _package_unchanged_since(base: str) -> bool:
    """Whether this checkout's ``polylogue/`` is byte-identical to *base*'s.

    The effective schema is a function of the package source alone, so an
    unchanged package renders the base state exactly and needs no extraction.
    """
    try:
        changed = subprocess.run(
            ["git", "diff", "--quiet", base, "--", "polylogue"], capture_output=True, cwd=ROOT, check=False
        )
        untracked = _git_text("ls-files", "--others", "--exclude-standard", "--", "polylogue")
    except (OSError, subprocess.CalledProcessError):
        return False
    return changed.returncode == 0 and not untracked.strip()


def _git_text(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=ROOT).stdout


def _merge_base(explicit_base: str | None = None) -> str:
    """Resolve the commit the candidate's durable schema is compared against.

    An explicit ref is used exactly. Otherwise the base is the merge base with
    the default branch; when HEAD is itself on that branch (a default-branch
    push, or a checkout with no commits of its own) that merge base is HEAD,
    so the comparison steps to HEAD's first parent. A base that cannot be
    resolved, as in a shallow clone, is a refusal: comparing HEAD with itself
    would pass any schema change.
    """
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
    candidates.extend(("origin/master", "master", "origin/HEAD"))
    base = ""
    for candidate in dict.fromkeys(candidates):
        try:
            base = _git_text("merge-base", "HEAD", candidate).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if base:
            break
    if not base:
        raise RuntimeError(
            f"no schema comparison base: none of {', '.join(dict.fromkeys(candidates))} shares history with HEAD"
            f"{_shallow_hint()}; fetch the default branch or pass --base"
        )
    if base != _git_text("rev-parse", "--verify", "HEAD^{commit}").strip():
        return base
    try:
        return _git_text("rev-parse", "--verify", "HEAD^1^{commit}").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"no schema comparison base: HEAD is on the default branch and its first parent is unavailable"
            f"{_shallow_hint()}; fetch its history or pass --base"
        ) from exc


def _shallow_hint() -> str:
    try:
        shallow = _git_text("rev-parse", "--is-shallow-repository").strip() == "true"
    except (OSError, subprocess.CalledProcessError):
        return ""
    return " (this clone is shallow)" if shallow else ""


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


def _migration_integrity_violations(
    base: str,
    tier: ArchiveTier,
    *,
    allow_predecessor_retirement: bool = False,
) -> list[str]:
    """Reject mutation of durable SQL or its frozen train sidecar, except a complete lineage reset's retirement."""
    violations: list[str] = []
    for change in _migration_changes(base, tier):
        paths = (change.old_path, change.new_path)
        if any(path.endswith(".sql") for path in paths):
            kind = "migration"
        elif any(path.endswith(".train.json") for path in paths):
            kind = "migration train sidecar"
        else:
            continue
        if change.status.startswith("A"):
            if kind == "migration" and _migration_version(change.new_path, tier) is None:
                violations.append(f"{tier.value}: added migration has an invalid numbered name: {change.new_path}")
        elif change.status.startswith("D") and not allow_predecessor_retirement:
            violations.append(f"{tier.value}: required {kind} was deleted: {change.old_path}")
        elif change.status.startswith(("M", "R", "C", "T")):
            violations.append(f"{tier.value}: required {kind} was modified: {change.old_path}")
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


def _semantic_ddl_objects(ddl: str, tier: ArchiveTier) -> dict[tuple[str, str], str] | None:
    """Render the normalized schema manifest projection for arbitrary DDL."""
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(ddl)
        manifest = SchemaManifest.from_connection(connection, tier)
        return {(kind, name): definition for kind, name, definition in manifest.objects}
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def _durable_ddl_evolution_violations(explicit_base: str | None = None) -> list[str]:
    """Require effective durable schema changes to use an exact migration chain."""
    base = _merge_base(explicit_base)
    current = _render_schema_state(None)
    previous = current if _package_unchanged_since(base) else _render_schema_state(base)
    violations: list[str] = []

    # A reset to v1 is valid only as one complete archive-format transition.
    # Treating an individual tier's downgrade as sufficient would turn an
    # accidental edit (or a forgotten migration) into an apparent new floor.
    # The active-root marker provides the runtime admission proof; this gate
    # enforces the corresponding repository-shape proof.
    reset_to_new_floor = all(
        previous.versions.get(tier, 0) > ARCHIVE_FORMAT_FLOOR_VERSION
        and current.versions.get(tier) == ARCHIVE_FORMAT_FLOOR_VERSION
        for tier in _DURABLE_TIERS
    )
    retire_predecessor_chain = all(
        current.versions.get(tier) == ARCHIVE_FORMAT_FLOOR_VERSION for tier in _DURABLE_TIERS
    )
    # A new marker lineage fences every earlier archive before its tiers open.
    # It may therefore revise the fresh floor's DDL, or fold a predecessor's
    # numbered migrations into it, provided every durable user_version counter
    # returns to one and no migration route is added alongside.
    new_fresh_lineage = (
        previous.lineage != current.lineage
        and all(current.versions.get(tier) == ARCHIVE_FORMAT_FLOOR_VERSION for tier in _DURABLE_TIERS)
        and not any(_added_migration_versions(base, tier)[0] for tier in _DURABLE_TIERS)
    )

    for tier in _DURABLE_TIERS:
        # The runtime's own discovery route: it refuses a post-floor slot whose
        # NNN.train.json is missing, malformed, or bound to other SQL.
        try:
            durable_migration_claims(tier)
        except MigrationError as exc:
            violations.append(f"{tier.value}: shipped migrations fail runtime admission: {exc}")
        violations.extend(
            _migration_integrity_violations(base, tier, allow_predecessor_retirement=retire_predecessor_chain)
        )
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
            # The archive floor is a deliberately new format lineage.  Its
            # sidecar marker is checked by bootstrap before opening any tier;
            # treating the reset as an ordinary schema downgrade here would
            # reject the intentional v1 floor while allowing no useful
            # migration path.  Other backwards moves remain prohibited.
            if new_version != ARCHIVE_FORMAT_FLOOR_VERSION or not (reset_to_new_floor or new_fresh_lineage):
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

        old_manifest = _semantic_ddl_objects(old_ddl, tier)
        new_manifest = _semantic_ddl_objects(new_ddl, tier)
        schema_changed = (
            old_manifest != new_manifest
            if old_manifest is not None and new_manifest is not None
            else old_ddl != new_ddl
        )
        if schema_changed and old_version == new_version and not new_fresh_lineage:
            violations.append(f"{tier.value}: rendered DDL changed without a schema-version bump")
    return violations


#: Source-origin tokens that must not name an index-tier relation or column
#: (polylogue-qvxun). Provider-specific evidence belongs in the neutral
#: relations -- the work-evidence graph, session_links, session_events -- with
#: the provider identity carried in the row, never in the schema.
_PROVIDER_TOKENS = (
    "codex",
    "claude",
    "gemini",
    "hermes",
    "chatgpt",
    "antigravity",
    "aistudio",
    "anthropic",
    "openai",
)

_RELATION_RE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?(?:TABLE|VIEW|TRIGGER|(?:UNIQUE\s+)?INDEX)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
    re.I,
)
_COLUMN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(?:TEXT|INTEGER|REAL|BLOB|ANY)\b", re.I | re.M)


def _strip_sql_noise(ddl: str) -> str:
    """Drop comments and string literals so only identifiers remain.

    ``origin IN ('claude-code-session', ...)`` is a vocabulary value, not a
    provider-named schema object; the check must not confuse the two.
    """
    without_comments = re.sub(r"--[^\n]*", "", ddl)
    return re.sub(r"'[^']*'", "''", without_comments)


def _provider_named_index_objects() -> list[str]:
    """Return index-tier relation/column names carrying a provider token."""
    ddl = _strip_sql_noise(ARCHIVE_DDL_BY_TIER[ArchiveTier.INDEX])
    names = {match.group(1) for match in _RELATION_RE.finditer(ddl)}
    names.update(match.group(1) for match in _COLUMN_RE.finditer(ddl))
    return sorted(
        f"{name} (provider token '{token}')" for name in names for token in _PROVIDER_TOKENS if token in name.lower()
    )


def _check_tier(tier: ArchiveTier, path: Path | None) -> dict[str, Any]:
    expected = canonical_schema_manifest(tier)
    result: dict[str, Any] = {"tier": tier.value, "version": expected.version, "ok": True}
    if path is None:
        return result
    if not path.exists():
        result["ok"] = False
        result["diff"] = {"file": {"expected": "present", "actual": "missing"}}
        return result
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
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
    location = ArchiveLocation.resolve(args.archive_root) if args.archive_root is not None else None
    results = []
    for tier in ArchiveTier:
        path = None
        if location is not None:
            path = location.active_index_path if tier is ArchiveTier.INDEX else args.archive_root / f"{tier.value}.db"
        results.append(_check_tier(tier, path))
    provider_named = _provider_named_index_objects()
    payload = {
        "kind": "polylogue.schema-manifest",
        "ok": all(item["ok"] for item in results) and not provider_named,
        "tiers": results,
        "provider_named_index_objects": provider_named,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for item in results:
            print(f"{item['tier']}: {'PASS' if item['ok'] else 'FAIL'} (v{item['version']})")
        for violation in provider_named:
            print(f"FAIL: provider-named index-tier object {violation}")
        print("schema-manifest: PASS" if payload["ok"] else "schema-manifest: FAIL")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
