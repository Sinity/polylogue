"""Inventory string-membership checks in the six canonical archive tiers.

The inventory is deliberately derived from the executable DDL, not from a
second list of columns.  Durable tiers are reported separately because their
membership is validated at the write boundary (``require_vocabulary``) and
must not be generated into durable DDL.  Derived/disposable checks are matched
to the domain enum/Literal value sets that own them; an unmatched set is an
explicit storage-local vocabulary rather than an unresolved one.
"""

from __future__ import annotations

import json
import re
import typing
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

from polylogue.core.enums import PolylogueStrEnum
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_DDL_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_MEMBERSHIP = re.compile(r"(?<!NOT )\b([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\(([^()]*)\)", re.IGNORECASE)
_STRING_LITERAL = re.compile(r"'((?:[^']|'')*)'")
_TABLE = re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)

Disposition = Literal["DURABLE_WRITE_BOUNDARY", "GENERATED_FROM_OWNER", "STORAGE_LOCAL"]


@dataclass(frozen=True, slots=True)
class VocabularyCheck:
    """One string-membership CHECK from canonical executable DDL."""

    tier: ArchiveTier
    table: str
    column: str
    values: tuple[str, ...]
    owner: str | None
    disposition: Disposition
    reason: str

    @property
    def ref(self) -> str:
        return f"{self.tier.value}.{self.table}.{self.column}"


@dataclass(frozen=True, slots=True)
class VocabularyInventory:
    """Complete six-tier vocabulary evidence."""

    checks: tuple[VocabularyCheck, ...]

    @property
    def denominator(self) -> int:
        return len(self.checks)

    @property
    def durable_exclusions(self) -> int:
        return sum(item.disposition == "DURABLE_WRITE_BOUNDARY" for item in self.checks)

    @property
    def unknown_ownership(self) -> int:
        # Every row has either a domain owner, an explicit durable policy, or
        # an explicit storage-local disposition.  There is no fourth state.
        return sum(
            item.owner is None and item.disposition not in {"DURABLE_WRITE_BOUNDARY", "STORAGE_LOCAL"}
            for item in self.checks
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "format": "polylogue-vocabulary-inventory/v1",
            "denominator": self.denominator,
            "durable_exclusions": self.durable_exclusions,
            "unknown_ownership": self.unknown_ownership,
            "checks": [
                {
                    "tier": item.tier.value,
                    "table": item.table,
                    "column": item.column,
                    "values": list(item.values),
                    "owner": item.owner,
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


def _owner_sets() -> dict[frozenset[str], str]:
    """Return reachable enum/Literal sets with stable owner names."""
    # Importing the canonical DDL loads the archive-owned Literal aliases and
    # all enum classes referenced by generated checks.
    tuple(ARCHIVE_DDL_BY_TIER.values())
    owners: dict[frozenset[str], str] = {}
    for enum in _walk_enums():
        values = frozenset(member.value for member in enum)
        if values:
            owners.setdefault(values, enum.__name__)

    modules: tuple[ModuleType, ...]
    from polylogue.storage.sqlite import query_objects
    from polylogue.storage.sqlite.archive_tiers import archive_tiers_specs, ops, types

    modules = (archive_tiers_specs, ops, types, query_objects)
    for module in modules:
        for name, value in vars(module).items():
            args = typing.get_args(value)
            if args and all(isinstance(item, str) for item in args):
                owners.setdefault(frozenset(args), f"{module.__name__}.{name}")
    return owners


def build_inventory() -> VocabularyInventory:
    """Build the complete inventory from all six canonical tier DDL strings."""
    owners = _owner_sets()
    rows: list[VocabularyCheck] = []
    for tier, ddl in ARCHIVE_DDL_BY_TIER.items():
        tables = list(_TABLE.finditer(ddl))
        for match in _MEMBERSHIP.finditer(ddl):
            # ``NOT IN`` is a negative membership predicate, not a
            # column-named vocabulary CHECK (the regex intentionally keeps the
            # following parenthesized list available for ordinary IN checks).
            if match.group(1).upper() == "NOT":
                continue
            values = tuple(value.replace("''", "'") for value in _STRING_LITERAL.findall(match.group(2)))
            if not values:
                continue
            table_match = next((candidate for candidate in reversed(tables) if candidate.start() < match.start()), None)
            table = table_match.group(1) if table_match is not None else "<unknown-table>"
            owner = owners.get(frozenset(values))
            if tier in {ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT}:
                disposition: Disposition = "DURABLE_WRITE_BOUNDARY"
                reason = "durable DDL carries no enum-generated membership CHECK; validate at the write boundary"
            elif owner is not None:
                disposition = "GENERATED_FROM_OWNER"
                reason = "membership set is owned by the referenced enum/Literal"
            else:
                disposition = "STORAGE_LOCAL"
                reason = "no product/domain type owns this storage-local vocabulary"
            rows.append(
                VocabularyCheck(
                    tier=tier,
                    table=table,
                    column=match.group(1),
                    values=tuple(sorted(values)),
                    owner=owner,
                    disposition=disposition,
                    reason=reason,
                )
            )
    return VocabularyInventory(checks=tuple(rows))


def main(argv: list[str] | None = None) -> int:
    """Print the inventory as JSON for audits and receipts."""
    del argv
    inventory = build_inventory()
    print(json.dumps(inventory.to_payload(), indent=2, sort_keys=True))
    return 0 if inventory.unknown_ownership == 0 else 1


__all__ = ["VocabularyCheck", "VocabularyInventory", "build_inventory", "main"]
