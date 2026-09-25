"""Inventory string-membership checks in the six canonical archive tiers.

The inventory is deliberately derived from the executable DDL, not from a
second list of columns.  Durable tiers are reported separately because their
membership is validated at the write boundary (``require_vocabulary``) and
must not be generated into durable DDL. Derived/disposable checks without a
discovered owner remain unknown until a declaration supplies a reviewed
storage-local reason.
"""

from __future__ import annotations

import json
import re
import typing
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

from polylogue.core import types as core_types
from polylogue.core.enums import PolylogueStrEnum
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER as ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_MEMBERSHIP = re.compile(r"(?<!NOT )\b([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\(([^()]*)\)", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'((?:[^']|'')*)'")
_TABLE = re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

Disposition = Literal[
    "DURABLE_WRITE_BOUNDARY",
    "GENERATED_FROM_OWNER",
    "OWNER_CANDIDATE",
    "STORAGE_LOCAL",
    "UNKNOWN_OWNERSHIP",
]
Lifecycle = Literal["durable", "derived"]


@dataclass(frozen=True, slots=True)
class VocabularyCheck:
    """One string-membership CHECK from canonical executable DDL."""

    tier: ArchiveTier
    table: str
    column: str
    values: tuple[str, ...]
    owner: str | None
    lifecycle: Lifecycle
    disposition: Disposition
    reason: str

    @property
    def ref(self) -> str:
        return f"{self.tier.value}.{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class VocabularyInventory:
    """Complete six-tier vocabulary evidence."""

    checks: tuple[VocabularyCheck, ...]
    equivalent_owner_groups: tuple[tuple[str, ...], ...] = ()

    @property
    def denominator(self) -> int:
        return len(self.checks)

    @property
    def durable_exclusions(self) -> int:
        return sum(item.disposition == "DURABLE_WRITE_BOUNDARY" for item in self.checks)

    @property
    def unknown_ownership(self) -> int:
        return sum(
            item.lifecycle == "derived" and item.disposition in {"OWNER_CANDIDATE", "UNKNOWN_OWNERSHIP"}
            for item in self.checks
        )

    @property
    def unverified_owner_bindings(self) -> int:
        """Derived rows matched by members but not tied to their DDL declaration."""
        return sum(item.disposition == "OWNER_CANDIDATE" for item in self.checks)

    def to_payload(self) -> dict[str, object]:
        return {
            "format": "polylogue-vocabulary-inventory/v1",
            "denominator": self.denominator,
            "durable_exclusions": self.durable_exclusions,
            "unknown_ownership": self.unknown_ownership,
            "unverified_owner_bindings": self.unverified_owner_bindings,
            "equivalent_owner_groups": [list(group) for group in self.equivalent_owner_groups],
            "checks": [
                {
                    "tier": item.tier.value,
                    "table": item.table,
                    "column": item.column,
                    "values": list(item.values),
                    "owner": item.owner,
                    "lifecycle": item.lifecycle,
                    "disposition": item.disposition,
                    "reason": item.reason,
                }
                for item in self.checks
            ],
        }


def _walk_enums() -> tuple[type[PolylogueStrEnum], ...]:
    found: list[type[PolylogueStrEnum]] = []

    def visit(cls: type[PolylogueStrEnum]) -> None:
        for child in cls.__subclasses__():
            found.append(child)
            visit(child)

    visit(PolylogueStrEnum)
    return tuple(found)


def _owner_sets() -> tuple[dict[frozenset[str], str], tuple[tuple[str, ...], ...]]:
    """Return value-set owners and equivalent aliases.

    This is a candidate-discovery projection. A generated report still needs
    to bind a candidate to the declaration that supplies the CHECK; value
    equality alone is not sufficient evidence of ownership.
    """
    # Importing the canonical DDL loads the archive-owned Literal aliases and
    # all enum classes referenced by generated checks.
    tuple(ARCHIVE_DDL_BY_TIER.values())
    names_by_values: dict[frozenset[str], list[tuple[str, object]]] = {}
    for enum in _walk_enums():
        values = frozenset(member.value for member in enum)
        if values:
            names_by_values.setdefault(values, []).append((enum.__name__, enum))

    modules: tuple[ModuleType, ...]
    from polylogue.storage.sqlite import query_objects
    from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs, ops, query_unit_frame, types

    # Declaration modules precede consumers that re-export their types.  This
    # makes the canonical owner stable while still retaining those re-exports
    # in the alias groups below.
    modules = (core_types, types, archive_tiers_specs, ops, query_objects, query_unit_frame)
    for module in modules:
        for name, value in vars(module).items():
            args = typing.get_args(value)
            if args and all(isinstance(item, str) for item in args):
                names_by_values.setdefault(frozenset(args), []).append((f"{module.__name__}.{name}", value))
    owners = {
        values: next(name for name, value in declarations if value is owner_object)
        for values, declarations in names_by_values.items()
        for owner_object in {id(value): value for _, value in declarations}.values()
        if len({id(value) for _, value in declarations}) == 1
    }
    aliases = tuple(
        tuple(sorted({name for name, _ in declarations}))
        for declarations in sorted(names_by_values.values(), key=lambda group: tuple(sorted(name for name, _ in group)))
        if len({id(value) for _, value in declarations}) == 1 and len({name for name, _ in declarations}) > 1
    )
    return owners, aliases


def _check_expressions(sql: str) -> tuple[tuple[int, str], ...]:
    """Return balanced CHECK bodies and their source offsets.

    Membership-looking predicates in partial-index WHERE clauses are not
    CHECK constraints and must not enter the inventory denominator.
    """
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"\bCHECK\s*\(", sql, re.IGNORECASE):
        open_index = sql.find("(", match.start(), match.end())
        depth = 1
        quote: str | None = None
        index = open_index + 1
        while index < len(sql) and depth:
            char = sql[index]
            if quote is not None:
                if char == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
            elif char in {"'", '"', "`"}:
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        if depth == 0:
            found.append((match.start(), sql[open_index + 1 : index - 1]))
    return tuple(found)


def build_inventory() -> VocabularyInventory:
    """Build the complete inventory from all six canonical tier DDL strings."""
    owners, equivalent_owner_groups = _owner_sets()
    rows: list[VocabularyCheck] = []
    expected_tiers = set(ArchiveTier)
    actual_tiers = set(ARCHIVE_DDL_BY_TIER)
    if actual_tiers != expected_tiers:
        missing = sorted(tier.value for tier in expected_tiers - actual_tiers)
        extra = sorted(tier.value for tier in actual_tiers - expected_tiers)
        raise ValueError(f"canonical tier DDL mismatch: missing={missing}, extra={extra}")
    for tier, ddl in ARCHIVE_DDL_BY_TIER.items():
        tables = list(_TABLE.finditer(ddl))
        for offset, expression in _check_expressions(ddl):
            for match in _MEMBERSHIP.finditer(expression):
                if match.group(1).upper() == "NOT":
                    continue
                values = tuple(value.replace("''", "'") for value in _STRING_LITERAL.findall(match.group(2)))
                if not values:
                    continue
                table_match = next((candidate for candidate in reversed(tables) if candidate.start() < offset), None)
                table = table_match.group(1) if table_match is not None else "<unknown-table>"
                owner = owners.get(frozenset(values))
                if tier in {ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT}:
                    lifecycle: Lifecycle = "durable"
                    disposition: Disposition = "DURABLE_WRITE_BOUNDARY"
                    reason = "durable DDL carries no enum-generated membership CHECK; validate at the write boundary"
                elif owner is not None:
                    lifecycle = "derived"
                    disposition = "OWNER_CANDIDATE"
                    reason = "candidate owner has the same members; direct declaration binding remains to be verified"
                else:
                    lifecycle = "derived"
                    disposition = "UNKNOWN_OWNERSHIP"
                    reason = "no declared domain owner or storage-local reason was found"
                rows.append(
                    VocabularyCheck(
                        tier=tier,
                        table=table,
                        column=match.group(1),
                        values=tuple(sorted(values)),
                        owner=owner,
                        lifecycle=lifecycle,
                        disposition=disposition,
                        reason=reason,
                    )
                )
    return VocabularyInventory(checks=tuple(rows), equivalent_owner_groups=equivalent_owner_groups)


def main(argv: list[str] | None = None) -> int:
    """Print the inventory as JSON for audits and receipts."""
    del argv
    inventory = build_inventory()
    print(json.dumps(inventory.to_payload(), indent=2, sort_keys=True))
    return 0 if inventory.unknown_ownership == 0 else 1


__all__ = ["Lifecycle", "VocabularyCheck", "VocabularyInventory", "build_inventory", "main"]
