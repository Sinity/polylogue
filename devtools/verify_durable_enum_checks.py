"""Verify durable-tier DDL carries no enum-derived membership CHECK.

Durable tiers (``source.db``, ``user.db``, ``audit.db``) evolve only by
additive numbered migration behind a verified backup. A membership CHECK
generated from a ``PolylogueStrEnum`` pins that enum's value set into durable
DDL, so every later token added to the enum becomes a durable migration. The
vocabulary is validated at the write boundary instead
(``require_vocabulary`` in ``storage/sqlite/archive_tiers/common.py``), which
costs nothing when a token is added.

Hand-written closed-vocabulary literals stay legal: a structural list an
author typed is a deliberate schema constraint, not a mirror of a Python
type, and does not drag an enum's evolution into the durable regime. Only a
list whose member set *is* some enum's value set is refused, because that is
what ``check()``/``nullable_check()`` render.

Derived and disposable tiers (``index.db``, ``embeddings.db``, ``ops.db``)
are out of scope: they are rebuilt on a version bump, so an enum CHECK there
is free.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass

from polylogue.core.enums import PolylogueStrEnum

# Quoted SQL tokens are indivisible: comment markers and parentheses inside
# them are data, never executable syntax.
_SQL_TOKEN = re.compile(
    r"--[^\n]*|/\*.*?(?:\*/|\Z)|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]"
    r"|[A-Za-z_][A-Za-z0-9_]*|[^\s]",
    re.DOTALL,
)

#: Structural literal lists that predate this gate and whose member set happens
#: to coincide with an enum's value set. Keyed on tier, column, and the exact
#: member set, so any change to one of these vocabularies breaks its waiver and
#: forces the column to be re-examined rather than silently re-widened.
GRANDFATHERED: frozenset[tuple[str, str, frozenset[str]]] = frozenset(
    {
        (
            "source",
            "outcome_code",
            frozenset(
                {
                    "success",
                    "validation_rejected",
                    "unsupported_shape",
                    "corrupt_input",
                    "transient_error",
                    "parser_defect",
                    "downstream_failure",
                    "canceled",
                    "interrupted",
                    "legacy_unknown",
                }
            ),
        ),
        (
            "source",
            "previous_revision_authority",
            frozenset({"asserted", "byte_proven", "quarantined"}),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class EnumCheckViolation:
    tier: str
    column: str
    enum_name: str
    members: tuple[str, ...]


def _reachable_enums() -> dict[frozenset[str], str]:
    """Map each ``PolylogueStrEnum`` value set to the enum that declares it.

    Importing the durable tier modules first ensures every enum those modules
    reference is loaded, so ``__subclasses__`` reaches the full population
    rather than whatever an unrelated import happened to pull in.
    """
    _durable_tier_ddl()

    def walk(cls: type[PolylogueStrEnum]) -> list[type[PolylogueStrEnum]]:
        found: list[type[PolylogueStrEnum]] = []
        for sub in cls.__subclasses__():
            found.append(sub)
            found.extend(walk(sub))
        return found

    by_members: dict[frozenset[str], str] = {}
    for enum in walk(PolylogueStrEnum):
        members = frozenset(member.value for member in enum)
        if not members:
            continue
        by_members.setdefault(members, enum.__name__)
    return by_members


def scan_ddl_for_enum_membership_checks(
    ddl: str,
    *,
    tier: str,
    enums_by_members: dict[frozenset[str], str] | None = None,
) -> list[EnumCheckViolation]:
    """Return literal enum memberships in executable table CHECK constraints."""
    resolved = _reachable_enums() if enums_by_members is None else enums_by_members
    tokens = [token for token in _SQL_TOKEN.findall(ddl) if not token.startswith(("--", "/*"))]
    checks: list[list[str]] = []
    for offset, token in enumerate(tokens[:-1]):
        if token.upper() != "CHECK" or tokens[offset + 1] != "(":
            continue
        depth, end = 1, offset + 2
        while end < len(tokens) and depth:
            if tokens[end] == "(":
                depth += 1
            elif tokens[end] == ")":
                depth -= 1
            end += 1
        if depth == 0:
            checks.append(tokens[offset + 2 : end - 1])
    violations: list[EnumCheckViolation] = []
    for body in checks:
        for offset in range(1, len(body) - 1):
            if body[offset].upper() != "IN" or body[offset + 1] != "(":
                continue
            column_offset = offset - 2 if body[offset - 1].upper() == "NOT" else offset - 1
            if column_offset < 0:
                continue
            column = body[column_offset]
            if column.startswith(('"', "`", "[")):
                quote = column[0]
                column = column[1:-1].replace(quote * 2, quote)
            elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column) is None:
                continue
            literals: list[str] = []
            end = offset + 2
            while end < len(body) and body[end].startswith("'") and body[end].endswith("'"):
                literals.append(body[end][1:-1].replace("''", "'"))
                end += 1
                if end >= len(body) or body[end] != ",":
                    break
                end += 1
            if not literals or end >= len(body) or body[end] != ")":
                continue
            members = frozenset(literals)
            enum_name = resolved.get(members)
            if enum_name is None or (tier, column, members) in GRANDFATHERED:
                continue
            violations.append(
                EnumCheckViolation(tier=tier, column=column, enum_name=enum_name, members=tuple(sorted(members)))
            )
    return violations


def _durable_tier_ddl() -> tuple[tuple[str, str], ...]:
    from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL
    from polylogue.storage.sqlite.archive_tiers.source import SOURCE_DDL
    from polylogue.storage.sqlite.archive_tiers.user import USER_DDL

    return (("source", SOURCE_DDL), ("user", USER_DDL), ("audit", AUDIT_DDL))


def _collect_durable_tier_violations() -> list[EnumCheckViolation]:
    enums_by_members = _reachable_enums()
    violations: list[EnumCheckViolation] = []
    for tier, ddl in _durable_tier_ddl():
        violations.extend(scan_ddl_for_enum_membership_checks(ddl, tier=tier, enums_by_members=enums_by_members))
        # Grandfathering applies only to literal SQL written into DDL. A
        # generated helper call is a policy violation even if it renders the
        # same historical column and member set.
        module = __import__(f"polylogue.storage.sqlite.archive_tiers.{tier}", fromlist=["__file__"])
        module_file = module.__file__
        if module_file is None:
            raise RuntimeError(f"module polylogue.storage.sqlite.archive_tiers.{tier} has no __file__")
        with open(module_file, encoding="utf-8") as source_file:
            source = source_file.read()
        for _waived_tier, column, _members in GRANDFATHERED:
            if _waived_tier == tier and re.search(rf"\b(?:nullable_)?check\(\s*['\"]{re.escape(column)}['\"]", source):
                enum_name = enums_by_members.get(_members, "unknown")
                violations.append(EnumCheckViolation(tier, column, enum_name, tuple(sorted(_members))))
    return violations


def _format_report(violations: list[EnumCheckViolation]) -> str:
    if not violations:
        return "Durable tiers carry no enum-derived membership CHECK."
    lines = [f"Enum-derived membership CHECKs found in durable tiers: {len(violations)}", ""]
    for violation in violations:
        lines.append(f"  {violation.tier}: {violation.column} matches {violation.enum_name}")
    lines.append("")
    lines.append(
        "Policy violation: durable-tier vocabulary membership is validated at the "
        "write boundary with require_vocabulary(), never pinned into DDL by "
        "check()/nullable_check() -- an enum CHECK makes every later token a "
        "durable migration (docs/internals.md)."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    violations = _collect_durable_tier_violations()

    if args.json:
        payload = {
            "violations": [
                {
                    "tier": violation.tier,
                    "column": violation.column,
                    "enum": violation.enum_name,
                    "members": list(violation.members),
                }
                for violation in violations
            ],
            "ok": not violations,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_report(violations))

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
