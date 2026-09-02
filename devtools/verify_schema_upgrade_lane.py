"""Verify schema-evolution policy boundaries.

Background
----------

Polylogue has two schema-evolution regimes:

* Durable tiers (``source.db``, ``user.db``, and ``audit.db``) may use explicit additive SQL
  migrations with a backup gate.
* Derived/rebuildable tiers (``index.db`` and ``embeddings.db``) do not use
  migration chains. They are rebuilt or blue-green replaced from durable source
  evidence, except for explicitly declared, clone-validated SQL plans.

What this lint checks
---------------------

1. Fail when the current index schema version lacks a delta-class declaration.
   Durable-tier migrations
   must live under ``polylogue/storage/sqlite/migrations/{source,user,audit}/`` as
   numbered SQL resources.

2. Validate every entry in the index-tier benign-DDL convergence registry
   (``polylogue.storage.sqlite.archive_tiers.index_convergence.
   INDEX_BENIGN_DDL_REGISTRY``, polylogue-jc1b): each entry's SQL must be
   exactly one of ``CREATE TABLE IF NOT EXISTS``, ``CREATE INDEX IF NOT
   EXISTS``, or ``DROP TABLE IF EXISTS`` -- idempotent and data-non-
   transforming by construction. Anything else (a bare ``CREATE``/``DROP``
   missing its guard clause, ``ALTER TABLE``, or a data-mutating statement)
   is the sanctioned same-version-open path being asked to do something a
   version bump should gate instead, and is rejected here.

The lint validates structured schema carriers and executable SQL shapes. It
does not infer architecture from Python function names.

Wired into the ordinary semantic verification baseline because the policy
boundary is a production correctness concern.

**Out of scope:** this lint is keyed entirely to ``INDEX_SCHEMA_VERSION``.
Parser and lowering drift use the production fingerprints declared by
``polylogue.sources.origin_specs`` instead: archive rows and candidate
operation results carry those fingerprints, and archive verification rejects
stale or mixed values. A green run of *this* lint alone is therefore not
evidence that no reparse is needed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from devtools import repo_root as _get_root
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.index_convergence import (
    INDEX_BENIGN_DDL_REGISTRY,
    BenignDDLEntry,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.durable_change_train import (
    DurableChangeTrainError,
    DurableMigrationClaim,
    durable_change_train_from_payload,
    durable_change_train_policy_report,
    durable_migration_claim_for_sql,
)
from polylogue.storage.sqlite.durable_change_train import (
    durable_migration_collision_report as _durable_migration_collision_report,
)
from polylogue.storage.sqlite.lifecycle import (
    INDEX_DELTA_DECLARATIONS,
    IndexDeltaDeclarationReport,
    index_delta_declaration_report,
)
from polylogue.storage.sqlite.migration_runner import DURABLE_MIGRATION_TIERS

ROOT = _get_root()
MIGRATIONS_DIR = ROOT / "polylogue" / "storage" / "sqlite" / "migrations"
ALLOWED_MIGRATION_TIERS = {"source", "user", "audit"}

_DURABLE_MIGRATION_SQL_RE = re.compile(r"^\d{3,}_[a-z0-9_]+\.sql$")
_DURABLE_MIGRATION_SIDECAR_RE = re.compile(r"^\d{3,}\.train\.json$")


# Index-tier benign-DDL registry entries (polylogue-jc1b) must be exactly one
# of these idempotent shapes -- the whole point is that re-applying an entry
# on every same-version open is always a no-op past the first time.
_ALLOWED_BENIGN_DDL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)^\s*CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s"),
    re.compile(r"(?is)^\s*CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s"),
    re.compile(r"(?is)^\s*DROP\s+TABLE\s+IF\s+EXISTS\s"),
)

# Any of these appearing anywhere in an entry's SQL means it mutates row data
# (not merely schema) or alters an existing object in place -- both outside
# the "idempotent, data-non-transforming" class a same-version-open hook may
# apply without an INDEX_SCHEMA_VERSION bump.
_FORBIDDEN_BENIGN_DDL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?is)\bALTER\s+TABLE\b"),
    re.compile(r"(?is)\bINSERT\s+INTO\b"),
    re.compile(r"(?is)\bUPDATE\s+\S"),
    re.compile(r"(?is)\bDELETE\s+FROM\b"),
)


@dataclass(frozen=True, slots=True)
class BenignDDLViolation:
    entry_name: str
    reason: str


@dataclass(frozen=True, slots=True)
class DDLLifecycleViolation:
    tier: str
    path: str
    reason: str


_DDL_TEXT_RE = re.compile(r"(?i)\b(?:CREATE|ALTER|DROP)\s+(?:TABLE|INDEX|TRIGGER|VIEW)\b")
_VERSION_RE = re.compile(r"(?m)^\s*(?P<name>[A-Z]+_SCHEMA_VERSION)\s*=\s*(?P<value>\d+)\s*$")
_WAIVER_RE = re.compile(r"(?i)ddl-lifecycle-waiver:\s*(?P<option>version|migration|derived|benign)\b(?P<reason>.+)")
_TIER_FILES = {
    "source.py": "source",
    "index.py": "index",
    "embeddings.py": "embeddings",
    "user.py": "user",
    "audit.py": "audit",
    "ops.py": "ops",
}
_DDL_ASSIGNMENT_RE = re.compile(r"(?m)^\s*[A-Z][A-Z0-9_]*DDL\s*=\s*(?:f)?\"\"\"")


def _git_text(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True, cwd=ROOT).stdout


def _schema_version(text: str, tier: str) -> int | None:
    match = _VERSION_RE.search(text)
    if match is None or match.group("name") != f"{tier.upper()}_SCHEMA_VERSION":
        return None
    return int(match.group("value"))


def _ddl_line_numbers(text: str) -> set[int]:
    """Return source lines contained in uppercase ``*_DDL`` triple strings."""
    lines = text.splitlines()
    result: set[int] = set()
    for index, line in enumerate(lines):
        if _DDL_ASSIGNMENT_RE.search(line) is None:
            continue
        for end in range(index, len(lines)):
            result.add(end + 1)
            if end > index and '"""' in lines[end]:
                break
    return result


def _registry_covers_changed_line(line: str) -> bool:
    normalized = " ".join(line.strip().rstrip(";").split()).lower()
    if not normalized:
        return False
    for entry in INDEX_BENIGN_DDL_REGISTRY:
        entry_sql = " ".join(entry.sql.strip().rstrip(";").split()).lower()
        if normalized.startswith(entry_sql) or entry_sql.startswith(normalized):
            return True
    return False


def _render_archive_ddl(ref: str | None) -> dict[ArchiveTier, str]:
    """Render every tier's fresh-init DDL from ``ref`` or this checkout."""
    if ref is None:
        return dict(ARCHIVE_DDL_BY_TIER)

    archive = subprocess.run(
        ["git", "archive", ref],
        check=True,
        capture_output=True,
        cwd=ROOT,
    ).stdout
    scratch = "/realm/tmp/work"
    with tempfile.TemporaryDirectory(
        prefix="polylogue-schema-", dir=scratch if os.path.isdir(scratch) else None
    ) as checkout:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
            tar.extractall(checkout, filter="data")
        (Path(checkout) / "polylogue" / "_build_info.py").write_text(
            f'BUILD_COMMIT = "{ref}"\nBUILD_DIRTY = False\n',
            encoding="utf-8",
        )
        script = (
            "import json\n"
            "from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER\n"
            "print(json.dumps({tier.value: ddl for tier, ddl in ARCHIVE_DDL_BY_TIER.items()}))\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = checkout
        # The rendering interpreter is this checkout's own; a shell that
        # exports another interpreter's sysconfig identity (a free-threaded
        # devshell, a relocated venv) must not steer it.
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
    rendered = json.loads(result.stdout)
    return {ArchiveTier(tier): ddl for tier, ddl in rendered.items()}


def _diff_base() -> str:
    try:
        return _git_text("rev-parse", "--verify", "origin/master").strip()
    except subprocess.CalledProcessError:
        return "HEAD"


def _ddl_lifecycle_report() -> list[DDLLifecycleViolation]:
    """Return changed tier DDL that has no declared evolution disposition.

    The patch is compared with the current branch base and includes staged and
    unstaged worktree changes.  A deliberately deleted ``CREATE TABLE`` line
    therefore remains visible to the anti-vacuity check.
    """
    base = _diff_base()
    patch = _git_text("diff", "--no-ext-diff", "--unified=0", base, "--", "polylogue/storage/sqlite/archive_tiers")
    name_status = _git_text("diff", "--name-status", base, "--", "polylogue/storage/sqlite")
    rendered_ddl_changes: dict[str, str] = {}
    try:
        previous_ddl = _render_archive_ddl(base)
        current_ddl = _render_archive_ddl(None)
        for archive_tier in ArchiveTier:
            if previous_ddl.get(archive_tier) != current_ddl.get(archive_tier):
                rendered_ddl_changes[f"polylogue/storage/sqlite/archive_tiers/{archive_tier.value}.py"] = (
                    f"rendered {archive_tier.value} tier DDL differs from {base}"
                )
    except (OSError, json.JSONDecodeError, KeyError, subprocess.SubprocessError, tarfile.TarError) as exc:
        return [
            DDLLifecycleViolation(
                "unknown",
                "<rendered archive DDL>",
                f"cannot compare rendered DDL against {base}: {exc}",
            )
        ]
    versions: dict[str, tuple[int | None, int | None]] = {}
    ddl_lines: dict[str, tuple[set[int], set[int]]] = {}
    for filename, tier_name in _TIER_FILES.items():
        tier_path = ROOT / "polylogue" / "storage" / "sqlite" / "archive_tiers" / filename
        try:
            current = tier_path.read_text(encoding="utf-8")
            previous = _git_text("show", f"{base}:polylogue/storage/sqlite/archive_tiers/{filename}")
        except (OSError, subprocess.CalledProcessError):
            continue
        versions[tier_name] = (_schema_version(previous, tier_name), _schema_version(current, tier_name))
        ddl_lines[tier_name] = (_ddl_line_numbers(previous), _ddl_line_numbers(current))

    added_migrations = {
        match.group(2)
        for line in name_status.splitlines()
        if (match := re.match(r"^(A|R\d*)\s+(.*)$", line)) and match.group(2).endswith(".sql")
    }
    changed: dict[str, list[str]] = {}
    current_path: str | None = None
    old_line = new_line = 0
    current_tier: str | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current_path = line.split(" b/", 1)[-1]
            current_tier = _TIER_FILES.get(current_path.rsplit("/", 1)[-1])
            continue
        if line.startswith("@@"):
            header = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if header is not None:
                old_line, new_line = int(header.group(1)), int(header.group(2))
            continue
        if current_path is None:
            continue
        if not (line.startswith("+") or line.startswith("-")):
            if line.startswith(" "):
                old_line += 1
                new_line += 1
            continue
        if line[:3] in {"+++", "---"}:
            continue
        line_number = new_line if line.startswith("+") else old_line
        old_ddl_lines, new_ddl_lines = ddl_lines.get(current_tier or "", (set(), set()))
        if _DDL_TEXT_RE.search(line[1:]) or line_number in (new_ddl_lines if line.startswith("+") else old_ddl_lines):
            changed.setdefault(current_path, []).append(line[1:])
        if line.startswith("+"):
            new_line += 1
        else:
            old_line += 1

    violations: list[DDLLifecycleViolation] = []
    for path, lines in changed.items():
        filename = path.rsplit("/", 1)[-1]
        changed_tier = _TIER_FILES.get(filename)
        if changed_tier is None:
            continue
        old_version, new_version = versions.get(changed_tier, (None, None))
        if old_version != new_version and new_version is not None:
            if changed_tier == "index" and not any(
                d.version == new_version and d.classes for d in INDEX_DELTA_DECLARATIONS
            ):
                violations.append(
                    DDLLifecycleViolation(changed_tier, path, f"v{new_version} has no DerivedDeltaClass declaration")
                )
            continue
        if path.startswith(f"polylogue/storage/sqlite/migrations/{changed_tier}/") and path in added_migrations:
            continue
        if changed_tier == "index" and any(_registry_covers_changed_line(line) for line in lines):
            continue
        if any(line.lstrip().startswith("#") and _WAIVER_RE.search(line) for line in lines):
            continue
        violations.append(
            DDLLifecycleViolation(
                changed_tier,
                path,
                "DDL changed without a schema-version bump, added durable migration, "
                "DerivedDeltaClass declaration, benign-DDL registry entry, or "
                "ddl-lifecycle-waiver comment naming the applicable option and reason",
            )
        )
    for path in rendered_ddl_changes:
        if path in changed:
            continue
        changed_tier = path.rsplit("/", 1)[-1][:-3]
        old_version, new_version = versions.get(changed_tier, (None, None))
        if old_version != new_version and new_version is not None:
            continue
        violations.append(
            DDLLifecycleViolation(
                changed_tier,
                path,
                "rendered DDL changed without a schema-version bump, added durable migration, "
                "DerivedDeltaClass declaration, benign-DDL registry entry, or "
                "ddl-lifecycle-waiver comment naming the applicable option and reason",
            )
        )
    return violations


def _invalid_benign_ddl_entries(
    entries: Iterable[BenignDDLEntry] = INDEX_BENIGN_DDL_REGISTRY,
) -> list[BenignDDLViolation]:
    """Return every registry entry whose SQL is not an allowed idempotent shape."""
    violations: list[BenignDDLViolation] = []
    for entry in entries:
        sql = entry.sql.strip()
        # Reject multi-statement smuggling -- a registry entry is exactly one
        # DDL statement (an optional single trailing semicolon is fine).
        body = sql[:-1] if sql.endswith(";") else sql
        if ";" in body:
            violations.append(BenignDDLViolation(entry.name, "registry entry must be exactly one statement"))
            continue
        if not any(pattern.match(sql) for pattern in _ALLOWED_BENIGN_DDL_PATTERNS):
            violations.append(
                BenignDDLViolation(
                    entry.name,
                    "must be CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS / DROP TABLE IF EXISTS",
                )
            )
            continue
        for forbidden in _FORBIDDEN_BENIGN_DDL_PATTERNS:
            if forbidden.search(sql):
                violations.append(
                    BenignDDLViolation(entry.name, f"data-transforming or non-idempotent clause: {forbidden.pattern}")
                )
                break
    return violations


def _invalid_migration_paths() -> list[Path]:
    if not MIGRATIONS_DIR.exists():
        return []
    invalid: list[Path] = []
    for path in sorted(MIGRATIONS_DIR.rglob("*")):
        if path.is_dir() or path.name == "__init__.py":
            continue
        rel = path.relative_to(MIGRATIONS_DIR)
        if (
            len(rel.parts) != 2
            or rel.parts[0] not in ALLOWED_MIGRATION_TIERS
            or not (
                _DURABLE_MIGRATION_SQL_RE.fullmatch(rel.parts[1])
                or _DURABLE_MIGRATION_SIDECAR_RE.fullmatch(rel.parts[1])
            )
        ):
            invalid.append(path)
    return invalid


def _durable_migration_claims_on_disk() -> tuple[DurableMigrationClaim, ...]:
    """Build the shared migration-slot claims from checked-in SQL resources."""
    claims: list[DurableMigrationClaim] = []
    for tier_name in sorted(ALLOWED_MIGRATION_TIERS):
        tier = ArchiveTier(tier_name)
        tier_dir = MIGRATIONS_DIR / tier_name
        if not tier_dir.exists():
            continue
        for path in sorted(tier_dir.glob("*.sql")):
            if re.fullmatch(r"\d{3,}_[a-z0-9_]+\.sql", path.name) is None:
                continue
            claims.append(
                durable_migration_claim_for_sql(
                    tier,
                    path.relative_to(ROOT),
                    path.read_text(encoding="utf-8"),
                    owner_ref=str(path.relative_to(ROOT)),
                )
            )
    return tuple(claims)


def _master_migration_head(tier: ArchiveTier) -> int:
    """Return the numbered migration head recorded by ``origin/master``."""
    prefix = f"polylogue/storage/sqlite/migrations/{tier.value}/"
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "origin/master", "--", prefix],
        check=True,
        capture_output=True,
        text=True,
    )
    versions = [
        int(match.group(1))
        for path in result.stdout.splitlines()
        if (match := re.fullmatch(rf"{re.escape(prefix)}(\d{{3,}})_[a-z0-9_]+\.sql", path))
    ]
    if not versions:
        raise RuntimeError(f"origin/master has no durable migrations for {tier.value}")
    return max(versions)


def _changed_paths_against_master() -> tuple[str, ...]:
    """Return paths changed by this checkout relative to the integration head."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/master", "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    paths = {path for path in result.stdout.splitlines() if path}
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in status.stdout.splitlines():
        if line.startswith("?? "):
            paths.add(line[3:])
    return tuple(sorted(paths))


def _durable_slot_reservation_violations(
    *,
    changed_paths: Iterable[str] | None = None,
    master_heads: dict[ArchiveTier, int] | None = None,
) -> list[str]:
    """Ensure changed trains reserve the next slot after master's head.

    Historical sidecars intentionally retain their predecessor version. Only
    sidecars changed by this checkout are reservations, so old released trains
    do not become false positives when the gate runs on master itself.
    """
    paths = tuple(changed_paths) if changed_paths is not None else _changed_paths_against_master()
    heads = master_heads or {}
    violations: list[str] = []
    for tier in sorted(DURABLE_MIGRATION_TIERS, key=lambda item: item.value):
        prefix = f"polylogue/storage/sqlite/migrations/{tier.value}/"
        for relative in paths:
            match = re.fullmatch(rf"{re.escape(prefix)}(\d{{3,}})\.train\.json", relative)
            if match is None:
                continue
            sidecar_path = ROOT / relative
            try:
                payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                train = durable_change_train_from_payload(payload)
                master_head = heads.get(tier)
                if master_head is None:
                    master_head = _master_migration_head(tier)
            except (OSError, json.JSONDecodeError, ValueError, RuntimeError, DurableChangeTrainError) as exc:
                violations.append(f"{relative}: cannot validate durable slot reservation: {exc}")
                continue
            if train.current_version != master_head:
                violations.append(
                    f"{relative}: durable slot reservation is stale: train current_version "
                    f"v{train.current_version} does not match origin/master {tier.value} head v{master_head}; "
                    "rebase and renumber the train onto the next unowned target slot"
                )
            elif train.target_version != master_head + 1:
                violations.append(
                    f"{relative}: durable slot reservation targets v{train.target_version}, but "
                    f"origin/master {tier.value} head v{master_head} requires v{master_head + 1}; "
                    "rebase and renumber the train onto the next unowned target slot"
                )
    return violations


def durable_migration_collision_report(
    claims: Iterable[DurableMigrationClaim],
) -> dict[str, object]:
    """Expose the shared slot report to schema-policy callers and tests."""
    return _durable_migration_collision_report(claims)


def _format_report(
    *,
    invalid_migrations: list[Path],
    delta_report: IndexDeltaDeclarationReport,
    benign_ddl_violations: list[BenignDDLViolation],
    ddl_lifecycle_violations: list[DDLLifecycleViolation],
    durable_change_train_reports: dict[str, dict[str, object]] | None = None,
    durable_migration_collisions: dict[str, object] | None = None,
    durable_slot_reservation_violations: list[str] | None = None,
) -> str:
    durable_change_train_reports = durable_change_train_reports or {}
    durable_violations = [
        violation
        for report in durable_change_train_reports.values()
        for violation in cast(tuple[object, ...], report.get("violations", ()))
        if isinstance(violation, str)
    ]
    durable_reservations = [
        reservation
        for report in durable_change_train_reports.values()
        for reservation in cast(tuple[object, ...], report.get("reservations", ()))
    ]
    durable_migration_collisions = durable_migration_collisions or {}
    durable_slot_reservation_violations = durable_slot_reservation_violations or []
    collision_entries = cast(tuple[object, ...], durable_migration_collisions.get("collisions", ()))
    lines = [
        f"invalid durable migration resources found: {len(invalid_migrations)}",
        f"durable change-train reservations found: {len(durable_reservations)}",
        f"durable change-train violations found: {len(durable_violations)}",
        f"durable migration slot collisions found: {len(collision_entries)}",
        f"durable migration slot reservation violations found: {len(durable_slot_reservation_violations)}",
        f"undeclared index schema deltas found: {len(delta_report['missing_versions'])}",
        f"invalid index benign-DDL registry entries found: {len(benign_ddl_violations)}",
        f"undeclared tier DDL lifecycle changes found: {len(ddl_lifecycle_violations)}",
    ]
    if invalid_migrations:
        lines.append("")
        lines.append("Invalid migration resources:")
        for path in invalid_migrations:
            lines.append(f"  {path.relative_to(ROOT)}")
    if not bool(delta_report["ok"]):
        lines.append("")
        lines.append("Index fast-forward declaration drift:")
        lines.append(f"  compatibility floor: v{delta_report['compatibility_floor']}")
        lines.append(f"  missing: {list(delta_report['missing_versions'])}")
        lines.append(f"  duplicate: {list(delta_report['duplicate_versions'])}")
        lines.append(f"  invalid: {list(delta_report['invalid_versions'])}")
        lines.append("")
        lines.append("Policy violation: each index schema bump needs a declared delta class.")
    if benign_ddl_violations:
        lines.append("")
        lines.append("Invalid index benign-DDL registry entries:")
        for benign_violation in benign_ddl_violations:
            lines.append(f"  {benign_violation.entry_name}: {benign_violation.reason}")
        lines.append("")
        lines.append(
            "Policy violation: index-tier same-version convergence entries must be idempotent, "
            "data-non-transforming CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS / "
            "DROP TABLE IF EXISTS statements."
        )
    if ddl_lifecycle_violations:
        lines.append("")
        lines.append("Undeclared tier DDL lifecycle changes:")
        for lifecycle_violation in ddl_lifecycle_violations:
            lines.append(f"  {lifecycle_violation.tier} ({lifecycle_violation.path}): {lifecycle_violation.reason}")
    if durable_reservations:
        lines.append("")
        lines.append("Durable change-train reservations:")
        lines.extend(f"  {reservation}" for reservation in durable_reservations)
    if durable_violations:
        lines.append("")
        lines.append("Durable change-train violations:")
        lines.extend(f"  {violation}" for violation in durable_violations)
    if collision_entries:
        lines.append("")
        lines.append("Durable migration slot collisions:")
        lines.extend(f"  {collision}" for collision in collision_entries)
    if durable_slot_reservation_violations:
        lines.append("")
        lines.append("Durable migration slot reservation violations:")
        lines.extend(f"  {violation}" for violation in durable_slot_reservation_violations)
    if (
        not invalid_migrations
        and bool(delta_report["ok"])
        and not benign_ddl_violations
        and not ddl_lifecycle_violations
        and not durable_violations
        and not collision_entries
        and not durable_slot_reservation_violations
    ):
        lines.append("")
        lines.append("Schema evolution policy intact.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    invalid_migrations = _invalid_migration_paths()
    durable_change_train_reports = {
        tier.value: durable_change_train_policy_report(tier)
        for tier in (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)
    }
    durable_migration_collisions = durable_migration_collision_report(_durable_migration_claims_on_disk())
    try:
        durable_slot_reservation_violations = _durable_slot_reservation_violations()
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        durable_slot_reservation_violations = [f"cannot inspect origin/master for durable slot reservations: {exc}"]
    delta_report = index_delta_declaration_report(INDEX_SCHEMA_VERSION)
    benign_ddl_violations = _invalid_benign_ddl_entries()
    ddl_lifecycle_violations = _ddl_lifecycle_report()

    ok = (
        not invalid_migrations
        and bool(delta_report["ok"])
        and not benign_ddl_violations
        and not ddl_lifecycle_violations
        and all(bool(report.get("ok")) for report in durable_change_train_reports.values())
        and bool(durable_migration_collisions["ok"])
        and not durable_slot_reservation_violations
    )

    if args.json:
        payload = {
            "invalid_migration_resources": [str(path.relative_to(ROOT)) for path in invalid_migrations],
            "durable_change_trains": durable_change_train_reports,
            "durable_migration_collisions": durable_migration_collisions,
            "durable_slot_reservation_violations": durable_slot_reservation_violations,
            "index_delta_declarations": delta_report,
            "invalid_benign_ddl_entries": [
                {"name": violation.entry_name, "reason": violation.reason} for violation in benign_ddl_violations
            ],
            "ddl_lifecycle_violations": [
                {"tier": violation.tier, "path": violation.path, "reason": violation.reason}
                for violation in ddl_lifecycle_violations
            ],
            "ok": ok,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(
            _format_report(
                invalid_migrations=invalid_migrations,
                delta_report=delta_report,
                benign_ddl_violations=benign_ddl_violations,
                ddl_lifecycle_violations=ddl_lifecycle_violations,
                durable_change_train_reports=durable_change_train_reports,
                durable_migration_collisions=durable_migration_collisions,
                durable_slot_reservation_violations=durable_slot_reservation_violations,
            )
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
