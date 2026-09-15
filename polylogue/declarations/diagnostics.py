"""Actionable registration diagnostics for declaration registries.

``validation.py`` answers "is this declaration structurally complete?" against
the record alone. This module answers the question a failing extension author
actually has: *which* declaration, *which* file, and *which* command repairs
the break -- by resolving every declared binding against the live checkout
instead of waiting for an opaque downstream failure.

It deliberately lives outside :mod:`polylogue.declarations.validation` because
that module is inside the derived-schema identity closure; resolution
diagnostics are a developer-experience concern and must not move the archive's
schema identity.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from polylogue.declarations.models import DeclarationSpec
from polylogue.declarations.registry import DeclarationRegistryProtocol
from polylogue.declarations.validation import Diagnostic, validate_declaration


@dataclass(frozen=True, slots=True)
class Resolution:
    """Where a declaration's binding was looked for and what was found."""

    declaration_id: str
    kind: str
    subject: str
    resolved: bool
    detail: str = ""


def _diagnostic(declaration: DeclarationSpec, code: str, message: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        message=f"{declaration.declaration_id}: {message}",
        declaration_id=declaration.declaration_id,
        owner_path=declaration.owner_path,
        repair_command=declaration.repair_command,
    )


def _import_symbol(binding_key: str) -> object:
    """Resolve ``module:symbol`` or a dotted path to a live object."""

    module_name, separator, symbol = binding_key.partition(":")
    if not separator:
        parts = binding_key.split(".")
        for split in range(len(parts) - 1, 0, -1):
            try:
                module = importlib.import_module(".".join(parts[:split]))
            except ImportError:
                continue
            target: object = module
            for part in parts[split:]:
                target = getattr(target, part)
            return target
        raise ImportError(f"no importable module prefix in {binding_key!r}")
    module = importlib.import_module(module_name)
    target = module
    for part in symbol.split("."):
        target = getattr(target, part)
    return target


def _is_importable_module_path(module_path: str) -> bool:
    """Return whether a dotted path could name an importable module."""

    return bool(module_path) and all(part.isidentifier() for part in module_path.split("."))


def _defines_symbol(owner: Path, symbol: str) -> bool:
    """Return whether ``owner`` defines ``symbol`` as a function or assignment."""

    try:
        source = owner.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(
        marker in source for marker in (f"def {symbol}(", f"async def {symbol}(", f"\n{symbol} = ", f"class {symbol}(")
    )


def resolve_declaration(declaration: DeclarationSpec, *, root: Path) -> tuple[Resolution, ...]:
    """Resolve every declared owner path and handler symbol against ``root``."""

    resolutions: list[Resolution] = [
        Resolution(
            declaration_id=declaration.declaration_id,
            kind="owner-path",
            subject=declaration.owner_path,
            resolved=(root / declaration.owner_path).exists(),
        )
    ]
    for handler in declaration.handlers:
        owner = root / handler.owner_path
        resolutions.append(
            Resolution(
                declaration_id=declaration.declaration_id,
                kind="handler-owner-path",
                subject=handler.owner_path,
                resolved=owner.exists(),
            )
        )
        module_path = handler.owner_path.removesuffix(".py").replace("/", ".")
        detail = ""
        resolved = True
        if not _is_importable_module_path(module_path):
            # A declaration may own a file outside the importable package tree
            # (a generated scaffold bundle, for instance). Its symbol is still
            # checkable against the source.
            resolved = _defines_symbol(owner, handler.symbol)
            detail = "" if resolved else f"{handler.symbol!r} is not defined in {handler.owner_path}"
            resolutions.append(
                Resolution(
                    declaration_id=declaration.declaration_id,
                    kind="handler-symbol",
                    subject=f"{handler.owner_path}:{handler.symbol}",
                    resolved=resolved,
                    detail=detail,
                )
            )
            continue
        try:
            _import_symbol(f"{module_path}:{handler.symbol}")
        except AttributeError as exc:
            # A registered handler is often defined inside its registrar
            # function, so it is never a module attribute. Fall back to the
            # owning source: a renamed or deleted handler still fails.
            resolved = _defines_symbol(owner, handler.symbol)
            detail = "" if resolved else str(exc)
        except ImportError as exc:
            resolved = False
            detail = str(exc)
        resolutions.append(
            Resolution(
                declaration_id=declaration.declaration_id,
                kind="handler-symbol",
                subject=f"{module_path}:{handler.symbol}",
                resolved=resolved,
                detail=detail,
            )
        )
    return tuple(resolutions)


def diagnose_registry(
    registry: DeclarationRegistryProtocol, *, root: Path, include_structural: bool = True
) -> tuple[Diagnostic, ...]:
    """Return structural *and* resolution diagnostics, deterministically ordered.

    Every diagnostic names the owning declaration, the owning path, and the
    declaration's own repair command, so one break produces one actionable
    error instead of a cascade of opaque downstream failures.
    """

    diagnostics: list[Diagnostic] = []
    declarations = registry.declarations()
    for declaration in declarations:
        for resolution in resolve_declaration(declaration, root=root):
            if resolution.resolved:
                continue
            if resolution.kind == "owner-path":
                diagnostics.append(
                    _diagnostic(
                        declaration,
                        "unresolved_owner_path",
                        f"declared owner path {resolution.subject!r} does not exist",
                    )
                )
            elif resolution.kind == "handler-owner-path":
                diagnostics.append(
                    _diagnostic(
                        declaration,
                        "unresolved_handler_path",
                        f"declared handler file {resolution.subject!r} does not exist",
                    )
                )
            else:
                diagnostics.append(
                    _diagnostic(
                        declaration,
                        "unresolved_handler_symbol",
                        f"declared handler {resolution.subject!r} does not resolve ({resolution.detail})",
                    )
                )
        for output in declaration.outputs:
            if not output.target_path:
                diagnostics.append(
                    _diagnostic(
                        declaration,
                        "missing_output_target",
                        f"generated output {output.name!r} declares no target path",
                    )
                )
        for example in declaration.examples:
            if not example.summary:
                diagnostics.append(
                    _diagnostic(declaration, "missing_example_summary", f"example {example.name!r} has no summary")
                )
    # Structural completeness too, so one call answers the whole question.
    # A caller may narrow to resolution only for a registry whose domain has
    # not yet declared examples or completeness edges.
    if include_structural:
        for declaration in declarations:
            diagnostics.extend(validate_declaration(declaration))
    return tuple(sorted(diagnostics, key=lambda item: (item.declaration_id, item.code, item.message)))


def format_diagnostic(diagnostic: Diagnostic) -> str:
    """Render one diagnostic as a single actionable line."""

    return f"{diagnostic.owner_path}: {diagnostic.code}: {diagnostic.message}; repair: `{diagnostic.repair_command}`"


__all__ = ["Resolution", "diagnose_registry", "format_diagnostic", "resolve_declaration"]
