"""Deterministic registry for shared declaration records."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.declarations.models import DeclarationSpec, FamilySpec


@dataclass(frozen=True, slots=True)
class DeclarationConflictError(ValueError):
    """Raised when a duplicate or incompatible declaration is registered."""

    message: str
    declaration_id: str
    owner_path: str
    differences: tuple[tuple[str, str, str], ...] = ()

    def __str__(self) -> str:
        if not self.differences:
            return self.message
        detail = ", ".join(f"{name}: {left!r} != {right!r}" for name, left, right in self.differences)
        return f"{self.message} ({detail})"


class DeclarationRegistry:
    """Register declarations while preserving deterministic projections."""

    def __init__(self) -> None:
        self._by_id: dict[str, DeclarationSpec] = {}
        self._by_public_name: dict[str, DeclarationSpec] = {}
        self._family_compatibility: dict[str, DeclarationSpec] = {}

    def register(self, declaration: DeclarationSpec) -> DeclarationSpec:
        """Register one declaration or raise a source-locatable conflict."""

        existing = self._by_id.get(declaration.declaration_id)
        if existing is not None:
            raise DeclarationConflictError(
                message=f"duplicate declaration id {declaration.declaration_id!r}",
                declaration_id=declaration.declaration_id,
                owner_path=declaration.owner_path,
            )
        existing_name = self._by_public_name.get(declaration.public_name)
        if existing_name is not None:
            raise DeclarationConflictError(
                message=(
                    f"public name {declaration.public_name!r} is already owned by "
                    f"{existing_name.declaration_id!r} at {existing_name.owner_path}"
                ),
                declaration_id=declaration.declaration_id,
                owner_path=declaration.owner_path,
            )
        family_member = self._family_compatibility.get(declaration.family_id)
        if family_member is not None:
            differences = family_member.compatibility.differences(declaration.compatibility)
            if differences:
                raise DeclarationConflictError(
                    message=(
                        f"declaration {declaration.declaration_id!r} cannot join family "
                        f"{declaration.family_id!r}; existing owner is {family_member.owner_path}"
                    ),
                    declaration_id=declaration.declaration_id,
                    owner_path=declaration.owner_path,
                    differences=differences,
                )
        else:
            self._family_compatibility[declaration.family_id] = declaration

        self._by_id[declaration.declaration_id] = declaration
        self._by_public_name[declaration.public_name] = declaration
        return declaration

    def get(self, declaration_id: str) -> DeclarationSpec:
        return self._by_id[declaration_id]

    def by_public_name(self, public_name: str) -> DeclarationSpec:
        return self._by_public_name[public_name]

    def declarations(self) -> tuple[DeclarationSpec, ...]:
        """Return declarations in stable id order, independent of registration order."""

        return tuple(self._by_id[key] for key in sorted(self._by_id))

    def families(self) -> tuple[FamilySpec, ...]:
        """Return deterministic family projections."""

        grouped: dict[str, list[DeclarationSpec]] = {}
        for declaration in self._by_id.values():
            grouped.setdefault(declaration.family_id, []).append(declaration)
        return tuple(
            FamilySpec(
                family_id=family_id,
                compatibility=members[0].compatibility,
                declaration_ids=tuple(sorted(member.declaration_id for member in members)),
                owner_paths=tuple(sorted({member.owner_path for member in members})),
            )
            for family_id, members in sorted(grouped.items())
        )

    def __len__(self) -> int:
        return len(self._by_id)


__all__ = ["DeclarationConflictError", "DeclarationRegistry"]
