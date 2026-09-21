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
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from polylogue.storage.sqlite.archive_tiers import (
    ARCHIVE_DDL_BY_TIER,
    ARCHIVE_FORMAT_FLOOR_VERSION,
    ARCHIVE_VERSION_BY_TIER,
)
from polylogue.storage.sqlite.archive_tiers.index_convergence import INDEX_BENIGN_DDL_REGISTRY
from polylogue.storage.sqlite.archive_tiers.ops import OPS_BENIGN_DDL_CONVERGENCE_PLAN
from polylogue.storage.sqlite.archive_tiers.schema_inventory import _objects_from_connection
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


#: The only statement shapes a same-version benign-DDL entry may take. Each is
#: idempotent on re-application and adds or removes an empty-of-consequence
#: schema object.
_ALLOWED_BENIGN_DDL = (
    re.compile(r"^CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+\S", re.I),
    re.compile(r"^CREATE\s+(?:UNIQUE\s+)?INDEX\s+IF\s+NOT\s+EXISTS\s+\S", re.I),
    re.compile(r"^DROP\s+TABLE\s+IF\s+EXISTS\s+\S", re.I),
)
#: Tails and verbs that transform data. ``AS``/``SELECT``/``VALUES`` are the
#: reason a prefix allowlist is not enough on its own: ``CREATE TABLE IF NOT
#: EXISTS x AS SELECT ...`` matches the idempotent prefix, carries no second
#: statement, and still rewrites derived content on every same-version archive
#: open, with no version bump and no reparse (polylogue-d0kj).
_FORBIDDEN_BENIGN_DDL = (
    re.compile(r"\bAS\b", re.I),
    re.compile(r"\bSELECT\b", re.I),
    re.compile(r"\bVALUES\b", re.I),
    re.compile(r"\bWITH\b", re.I),
    re.compile(r"\bALTER\s+TABLE\b", re.I),
    re.compile(r"\bINSERT\s+INTO\b", re.I),
    re.compile(r"\bUPDATE\b", re.I),
    re.compile(r"\bDELETE\s+FROM\b", re.I),
    re.compile(r"\bREPLACE\s+INTO\b", re.I),
)


def invalid_benign_ddl_entries(entries: Iterable[tuple[str, str]]) -> list[str]:
    """Return one violation per ``(name, sql)`` that is not benign same-version DDL.

    ``entries`` are the registered statements applied on every same-version
    open of an already-populated archive, so each must be idempotent and
    data-non-transforming. Validation is shape-based on purpose: the statement
    must match exactly one allowed idempotent form, carry no second statement,
    and contain no data-producing tail.
    """
    violations: list[str] = []
    for name, sql in entries:
        statement = " ".join(sql.split()).strip()
        body = statement[:-1].strip() if statement.endswith(";") else statement
        if ";" in body:
            violations.append(f"{name}: benign DDL carries more than one statement")
            continue
        if not any(pattern.match(body) for pattern in _ALLOWED_BENIGN_DDL):
            violations.append(f"{name}: benign DDL is not an allowed idempotent shape: {body}")
            continue
        for pattern in _FORBIDDEN_BENIGN_DDL:
            match = pattern.search(body)
            if match is not None:
                violations.append(f"{name}: benign DDL carries a data-transforming tail: {match.group(0).upper()}")
                break
    return violations


def _benign_ddl_violations() -> list[str]:
    """Validate every registered same-version benign-DDL statement."""
    entries: list[tuple[str, str]] = [(entry.name, entry.sql) for entry in INDEX_BENIGN_DDL_REGISTRY]
    entries.extend((f"ops[{index}]", sql) for index, sql in enumerate(OPS_BENIGN_DDL_CONVERGENCE_PLAN))
    return invalid_benign_ddl_entries(entries)


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


def _migration_integrity_violations(
    base: str,
    tier: ArchiveTier,
    *,
    allow_predecessor_retirement: bool = False,
) -> list[str]:
    """Reject mutation of durable SQL, except a complete lineage reset's retirement."""
    violations: list[str] = []
    for change in _migration_changes(base, tier):
        if not (change.old_path.endswith(".sql") or change.new_path.endswith(".sql")):
            continue
        if change.status.startswith("A"):
            if _migration_version(change.new_path, tier) is None:
                violations.append(f"{tier.value}: added migration has an invalid numbered name: {change.new_path}")
        elif change.status.startswith("D") and not allow_predecessor_retirement:
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


def _ddl_objects(ddl: str, tier: ArchiveTier) -> dict[str, str] | None:
    """Return object_ref -> definition digest, or None if the DDL does not render."""
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(ddl)
        return {obj.object_ref: obj.definition_sha256 for obj in _objects_from_connection(connection, tier)}
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def _is_retirement_only(old_ddl: str, new_ddl: str, tier: ArchiveTier) -> bool:
    """True when the rendered DDL differs solely by declared retired removals.

    A retired object is omitted from fresh generations while migrated
    historical tiers keep it, so no migration runs and no database changes.
    Anything added or redefined is an ordinary durable evolution.
    """
    from polylogue.storage.sqlite.migration_runner import _retired_schema_objects_for_parity

    retired = _retired_schema_objects_for_parity(tier)
    if not retired:
        return False
    old_objects = _ddl_objects(old_ddl, tier)
    new_objects = _ddl_objects(new_ddl, tier)
    if old_objects is None or new_objects is None:
        return False
    if set(new_objects) - set(old_objects):
        return False
    retired_tables = {ref.split(":", 1)[1] for ref in retired if ref.startswith("table:")}
    removed = set(old_objects) - set(new_objects)
    # Tables that lost a declared-retired column of their own. Their
    # ``CREATE TABLE`` text necessarily differs, so the table object's digest
    # moves even though the table itself was not redefined (polylogue-48bos).
    tables_losing_a_retired_column: set[str] = set()
    for ref in removed:
        kind_and_name = ref.split(":", 1)[1]
        owning_table = kind_and_name.split(":", 1)[1].split(".", 1)[0] if kind_and_name.startswith("column:") else ""
        if kind_and_name in retired:
            if owning_table:
                tables_losing_a_retired_column.add(owning_table)
            continue
        if owning_table and owning_table in retired_tables:
            continue
        return False
    for ref in set(old_objects) & set(new_objects):
        if old_objects[ref] == new_objects[ref]:
            continue
        _tier_token, kind, name = ref.split(":", 2)
        # Only the owning table's own declaration may move, and only when
        # every column it kept is byte-identical -- so the digest change is
        # the retired column's removal and nothing else rode along with it.
        if kind != "table" or name not in tables_losing_a_retired_column:
            return False
        prefix = f"{tier.value}:column:{name}."
        if any(
            old_objects[column_ref] != new_objects[column_ref]
            for column_ref in set(old_objects) & set(new_objects)
            if column_ref.startswith(prefix)
        ):
            return False
    return True


def _durable_ddl_evolution_violations(explicit_base: str | None = None) -> list[str]:
    """Require effective durable schema changes to use an exact migration chain."""
    base = _merge_base(explicit_base)
    previous = _render_schema_state(base)
    current = _render_schema_state(None)
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

    for tier in _DURABLE_TIERS:
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
            if new_version != ARCHIVE_FORMAT_FLOOR_VERSION or not reset_to_new_floor:
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

        if old_ddl != new_ddl and old_version == new_version and not _is_retirement_only(old_ddl, new_ddl, tier):
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
    benign_violations = _benign_ddl_violations()
    provider_named = _provider_named_index_objects()
    payload = {
        "kind": "polylogue.schema-manifest",
        "ok": all(item["ok"] for item in results) and not benign_violations and not provider_named,
        "tiers": results,
        "benign_ddl_violations": benign_violations,
        "provider_named_index_objects": provider_named,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for item in results:
            print(f"{item['tier']}: {'PASS' if item['ok'] else 'FAIL'} (v{item['version']})")
        for violation in benign_violations:
            print(f"FAIL: benign-ddl {violation}")
        for violation in provider_named:
            print(f"FAIL: provider-named index-tier object {violation}")
        print("schema-manifest: PASS" if payload["ok"] else "schema-manifest: FAIL")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
