"""Prove every committed schema package describes its own subject.

polylogue-n61h5: ``polylogue/schemas/providers/grok`` shipped a package whose
element schemas carried ``$id: polylogue://schemas/claude-ai/v2/session_document``
and Claude.ai's document shape (``chat_messages``, ``uuid``, ``account``,
``project``). That is a source-selection defect -- another subject's export
folded into this subject's package -- and it is worse than a missing package:
a consumer reading the package believes Grok's wire shape is Claude.ai's, so a
Grok document that diverges from Grok's real shape validates against the wrong
contract and produces no drift signal at all.

The check is deliberately structural and total. It reads every committed
package element from disk rather than through the registry's default-version
resolution, which would hide a mislabelled non-default version, and it reads
the gzipped element schemas directly so an element the registry does not
currently route is still audited.

Scope is the element ``$id`` -- the generator's own statement of which subject
the element describes. Observed provider *values* are deliberately not a
violation on their own: ``claude-design`` legitimately observes
``provider: claude-ai`` in its payloads, because Claude Design documents are
produced by Claude.ai. The ``$id`` carries no such ambiguity.
"""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Root the committed packages live under.
SCHEMA_PROVIDERS_ROOT = Path(__file__).resolve().parent / "providers"

#: ``polylogue://schemas/<subject>/<version>/<element>`` -- the ``$id`` shape
#: the generator emits.
_SCHEMA_ID_RE = re.compile(r"^polylogue://schemas/(?P<subject>[^/]+)/")


@dataclass(frozen=True, slots=True)
class ProviderIdentityViolation:
    """One committed element whose ``$id`` names a subject it does not belong to."""

    package_dir: str
    element_path: str
    declared_subject: str

    def format_text(self) -> str:
        return (
            f"{self.element_path}: $id names subject {self.declared_subject!r} inside the "
            f"{self.package_dir!r} package -- another subject's schema is folded into this one, so a "
            f"{self.package_dir} document diverging from {self.package_dir}'s real shape raises no drift"
        )


@dataclass(frozen=True, slots=True)
class ProviderIdentityReport:
    """Result of auditing every committed package element."""

    elements_checked: int
    violations: tuple[ProviderIdentityViolation, ...]

    @property
    def all_passed(self) -> bool:
        return not self.violations

    def format_text(self) -> str:
        if self.all_passed:
            return f"schema provider identity OK ({self.elements_checked} committed element(s) checked)"
        lines = [
            f"schema provider identity FAILED ({len(self.violations)} violation(s) "
            f"across {self.elements_checked} committed element(s))"
        ]
        lines.extend(f"  {violation.format_text()}" for violation in self.violations)
        return "\n".join(lines)


def _iter_element_documents(root: Path) -> Iterator[tuple[str, Path, object]]:
    """Yield ``(package_dir, element_path, document)`` for every committed element."""
    for package_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for element_path in sorted(package_dir.rglob("*.schema.json.gz")):
            with gzip.open(element_path, "rt", encoding="utf-8") as handle:
                yield package_dir.name, element_path, json.load(handle)
        for element_path in sorted(package_dir.rglob("*.schema.json")):
            yield package_dir.name, element_path, json.loads(element_path.read_text(encoding="utf-8"))


def audit_committed_provider_identity(root: Path | None = None) -> ProviderIdentityReport:
    """Refuse any committed package element whose ``$id`` names another subject."""
    providers_root = root if root is not None else SCHEMA_PROVIDERS_ROOT
    if not providers_root.is_dir():
        return ProviderIdentityReport(elements_checked=0, violations=())

    violations: list[ProviderIdentityViolation] = []
    elements = 0
    for package_dir, element_path, document in _iter_element_documents(providers_root):
        elements += 1
        if not isinstance(document, dict):
            continue
        schema_id = document.get("$id")
        if not isinstance(schema_id, str):
            continue
        match = _SCHEMA_ID_RE.match(schema_id)
        if match is None or match.group("subject") == package_dir:
            continue
        violations.append(
            ProviderIdentityViolation(
                package_dir=package_dir,
                element_path=str(element_path.relative_to(providers_root.parent.parent)),
                declared_subject=match.group("subject"),
            )
        )

    return ProviderIdentityReport(elements_checked=elements, violations=tuple(violations))


__all__ = [
    "ProviderIdentityReport",
    "ProviderIdentityViolation",
    "SCHEMA_PROVIDERS_ROOT",
    "audit_committed_provider_identity",
]
