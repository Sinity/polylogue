"""Promotion-time privacy and validity audit for staged schema artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import jsonschema

from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.field_stats.detection import is_dynamic_key

AuditSeverity: TypeAlias = Literal["blocker", "review"]

_REVIEW_FIELDS = frozenset(
    {
        "dominant_keys",
        "privacy_approved_values",
        "profile_tokens",
        "x-polylogue-values",
    }
)
_FORBIDDEN_PROVENANCE_FIELDS = frozenset({"bundle_scopes", "representative_paths"})
_SECRET_PATTERNS = {
    "anthropic_api_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential_url": re.compile(r"(?i)https?://[^\s/:]+:[^\s/@]+@"),
}
_EMAIL = re.compile(r"(?i)^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$")
_ABSOLUTE_PATH = re.compile(r"^(?:/|[A-Za-z]:[\\/])")
_URL = re.compile(r"(?i)^https?://")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)")
_IDENTIFIER = re.compile(r"^(?:[0-9a-f]{16,}|[0-9a-f]{8}-[0-9a-f-]{27,}|(?:rollout|session|agent)-)", re.I)


@dataclass(frozen=True, order=True)
class PromotionAuditFinding:
    """One deterministic blocker or operator-review observation."""

    severity: AuditSeverity
    category: str
    artifact: str
    json_path: str
    value: str


@dataclass(frozen=True)
class PromotionAuditReport:
    """Complete promotion verdict and review inventory."""

    root: str
    artifact_count: int
    findings: tuple[PromotionAuditFinding, ...]

    @property
    def blockers(self) -> tuple[PromotionAuditFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "blocker")

    @property
    def review_items(self) -> tuple[PromotionAuditFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "review")

    def grouped_review_items(self, *, sample_limit: int = 3) -> tuple[JSONDocument, ...]:
        """Return a lossless, operator-sized view over repeated review values.

        The full finding inventory remains in ``to_payload``.  This projection
        merely prevents a repeated structural token from hiding the few values
        an operator must actually inspect.
        """
        grouped: dict[tuple[str, str], list[PromotionAuditFinding]] = {}
        for item in self.review_items:
            grouped.setdefault((item.category, item.value), []).append(item)
        rows: list[JSONDocument] = []
        for (category, value), items in sorted(grouped.items()):
            artifacts = sorted({item.artifact for item in items})
            locations = sorted({f"{item.artifact}:{item.json_path}" for item in items})
            sample_locations: list[JSONValue] = list(locations[:sample_limit])
            rows.append(
                {
                    "category": category,
                    "value": value,
                    "occurrence_count": len(items),
                    "artifact_count": len(artifacts),
                    "sample_locations": sample_locations,
                }
            )
        return tuple(rows)

    def to_payload(self) -> JSONDocument:
        category_counts = Counter(f"{item.severity}:{item.category}" for item in self.findings)
        return json_document(
            {
                "audit_version": 1,
                "root": self.root,
                "artifact_count": self.artifact_count,
                "verdict": "blocked" if self.blockers else "review_required",
                "blocker_count": len(self.blockers),
                "review_count": len(self.review_items),
                "category_counts": dict(sorted(category_counts.items())),
                "review_summary": list(self.grouped_review_items()),
                "findings": [asdict(item) for item in self.findings],
            }
        )


def _load_artifact(path: Path) -> JSONValue:
    opener = gzip.open if path.name.endswith(".json.gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        value: JSONValue = json.load(stream)
    return value


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for child in value for text in _strings(child)]
    if isinstance(value, dict):
        return [text for child in value.values() for text in _strings(child)]
    return []


def _review_category(field: str, value: str) -> str:
    if field == "representative_paths" or _ABSOLUTE_PATH.search(value):
        return "filesystem_path"
    if field == "bundle_scopes" or _IDENTIFIER.search(value):
        return "identifier"
    if _EMAIL.fullmatch(value):
        return "email_or_account"
    if _URL.search(value):
        return "url_or_domain"
    if _DATE.search(value):
        return "date_or_time"
    if field in {"dominant_keys", "profile_tokens"}:
        return "structural_vocabulary"
    return "approved_readable_value"


def _unsafe_profile_token(value: str) -> bool:
    if ":" not in value:
        return False
    token_kind, _, observed_name = value.rpartition(":")
    return token_kind.startswith(("child:", "field:", "item:")) and is_dynamic_key(observed_name)


def _secret_findings(*, artifact: str, json_path: str, value: str) -> list[PromotionAuditFinding]:
    findings = []
    for category, pattern in _SECRET_PATTERNS.items():
        if pattern.search(value):
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category=category,
                    artifact=artifact,
                    json_path=json_path,
                    value=f"sha256:{digest};length={len(value)}",
                )
            )
    return findings


def _walk_artifact(
    value: object,
    *,
    artifact: str,
    json_path: str,
    findings: list[PromotionAuditFinding],
) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            for name in properties:
                property_path = f"{json_path}.properties[{name!r}]"
                secret_findings = _secret_findings(artifact=artifact, json_path=property_path, value=name)
                findings.extend(secret_findings)
                if is_dynamic_key(name):
                    findings.append(
                        PromotionAuditFinding(
                            severity="blocker",
                            category="unsafe_property_name",
                            artifact=artifact,
                            json_path=property_path,
                            value=secret_findings[0].value if secret_findings else name,
                        )
                    )
        for key, child in value.items():
            child_path = f"{json_path}.{key}"
            if key in _FORBIDDEN_PROVENANCE_FIELDS:
                findings.append(
                    PromotionAuditFinding(
                        severity="blocker",
                        category="raw_local_provenance",
                        artifact=artifact,
                        json_path=child_path,
                        value=f"field={key};value_count={len(_strings(child))}",
                    )
                )
            if key in _REVIEW_FIELDS:
                for text in _strings(child):
                    secret_findings = _secret_findings(artifact=artifact, json_path=child_path, value=text)
                    findings.extend(secret_findings)
                    if secret_findings:
                        continue
                    if key == "profile_tokens" and _unsafe_profile_token(text):
                        findings.append(
                            PromotionAuditFinding(
                                severity="blocker",
                                category="unsafe_structural_identifier",
                                artifact=artifact,
                                json_path=child_path,
                                value=text,
                            )
                        )
                        continue
                    findings.append(
                        PromotionAuditFinding(
                            severity="review",
                            category=_review_category(key, text),
                            artifact=artifact,
                            json_path=child_path,
                            value=text,
                        )
                    )
            _walk_artifact(child, artifact=artifact, json_path=child_path, findings=findings)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_artifact(child, artifact=artifact, json_path=f"{json_path}[{index}]", findings=findings)


def _element_kinds(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item.get("element_kind")) for item in value if isinstance(item, dict))


def _privacy_guard_findings(root: Path) -> list[PromotionAuditFinding]:
    """Run the enum-value privacy guard over every committed element schema.

    ``polylogue.schemas.audit.checks.check_privacy_guards`` catches UUIDs,
    hex ids, high-entropy tokens and otherwise-unsafe values recorded in
    ``x-polylogue-values`` annotations.  It is the sharper check for that
    class -- and it ran nowhere.  ``devtools schema-audit`` is the only
    caller, and that command is not part of any ``devtools verify`` gate,
    so nothing enforced it on the path that actually publishes schemas.

    That gap was not theoretical.  ``promote_cluster``'s samples path calls
    ``generate_schema_from_samples()``, which -- unlike the full ``generate``
    pipeline -- has no ``privacy_config`` plumbed through it, so a
    low-cardinality "safe enum" heuristic recorded seven literal
    conversation UUIDs verbatim into claude-ai v2.  The committed
    promotion audit reported zero blockers throughout, because it does not
    duplicate this check.

    Wiring it here means the gate that guards publication runs the guard
    that understands publication, rather than relying on a separate command
    nobody invokes.

    ``check_privacy_guards`` also has a broad catch-all branch (any value
    that fails its readable-enum allowlist, e.g. an email address or a
    private-TLD hostname) that this function deliberately does not surface:
    ``_walk_artifact`` already classifies every ``x-polylogue-values`` entry
    through ``_review_category``/``_secret_findings`` with a finer-grained
    severity split (secrets are specific blocker categories such as
    ``github_token``; emails/URLs/etc. are ``review`` items, not blockers).
    Escalating the catch-all branch here would re-flag those same values
    under a coarser ``unsafe_enum_value`` blocker and collapse that
    distinction. This function only escalates the three violation shapes
    ``_walk_artifact`` has no equivalent for: verbatim UUIDs, hex ids, and
    high-entropy tokens -- and even then skips a value already caught by
    ``_secret_findings``'s specific secret patterns, since that is already a
    (differently named) blocker.
    """
    import ast

    from polylogue.core.outcomes import OutcomeStatus
    from polylogue.schemas.audit.checks import check_privacy_guards

    detail_re = re.compile(r"^(.*?): (UUID leak|hex-id leak|high-entropy token) (.+)$")
    findings: list[PromotionAuditFinding] = []
    for path in sorted(root.rglob("*.schema.json.gz")):
        artifact = str(path.relative_to(root))
        try:
            schema = _load_artifact(path)
        except Exception:
            continue
        document = json_document(schema)
        if not document:
            continue
        result = check_privacy_guards(document)
        if result.status is not OutcomeStatus.ERROR:
            continue
        for detail in result.details:
            match = detail_re.match(detail)
            if match is None:
                # The generic "unsafe value" catch-all; already classified
                # (as a review item, not a blocker) by _walk_artifact.
                continue
            json_path, _kind, rendered_value = match.groups()
            try:
                raw_value = ast.literal_eval(rendered_value)
            except (ValueError, SyntaxError):
                raw_value = rendered_value
            if isinstance(raw_value, str) and any(pattern.search(raw_value) for pattern in _SECRET_PATTERNS.values()):
                # Already reported under its specific secret category.
                continue
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category="unsafe_enum_value",
                    artifact=artifact,
                    json_path=json_path,
                    value=detail,
                )
            )
    return findings


def _catalog_coherence_findings(root: Path) -> list[PromotionAuditFinding]:
    """Require each provider's catalog.json to agree with its packages.

    ``catalog.json`` is the resolution authority: ``runtime_registry`` reads it
    to pick a package and then reads that package's element list.  A promotion
    that rewrites ``versions/<v>/package.json`` without regenerating the
    catalog is therefore *inert* -- the new elements and profile families are
    never resolved, while the tree looks promoted.  That happened once
    already, so it is a blocker rather than a review item.  Writing through
    ``SchemaRegistry.replace_provider_packages`` keeps the two in step.
    """
    findings: list[PromotionAuditFinding] = []
    for catalog_path in sorted(root.rglob("catalog.json")):
        provider_dir = catalog_path.parent
        relative = str(catalog_path.relative_to(root))
        try:
            catalog = _load_artifact(catalog_path)
        except Exception:
            continue  # already reported as malformed_artifact
        if not isinstance(catalog, dict):
            continue
        packages = catalog.get("packages")
        if not isinstance(packages, list):
            continue
        catalogued = {
            str(entry.get("version")): entry for entry in packages if isinstance(entry, dict) and entry.get("version")
        }
        on_disk = {path.parent.name for path in provider_dir.glob("versions/*/package.json") if path.is_file()}
        for missing in sorted(on_disk - set(catalogued)):
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category="catalog_incoherent",
                    artifact=relative,
                    json_path="$.packages",
                    value=f"version={missing};reason=package_on_disk_absent_from_catalog",
                )
            )
        for stale in sorted(set(catalogued) - on_disk):
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category="catalog_incoherent",
                    artifact=relative,
                    json_path="$.packages",
                    value=f"version={stale};reason=catalogued_version_has_no_package",
                )
            )
        for version in sorted(on_disk & set(catalogued)):
            manifest_path = provider_dir / "versions" / version / "package.json"
            try:
                manifest = _load_artifact(manifest_path)
            except Exception:
                continue
            if not isinstance(manifest, dict):
                continue
            entry = catalogued[version]
            catalog_kinds = _element_kinds(entry.get("elements"))
            manifest_kinds = _element_kinds(manifest.get("elements"))
            if catalog_kinds != manifest_kinds:
                findings.append(
                    PromotionAuditFinding(
                        severity="blocker",
                        category="catalog_incoherent",
                        artifact=relative,
                        json_path=f"$.packages[version={version}].elements",
                        value=f"catalog={catalog_kinds};package={manifest_kinds}",
                    )
                )
            for field in ("sample_count", "first_seen", "last_seen"):
                if entry.get(field) != manifest.get(field):
                    findings.append(
                        PromotionAuditFinding(
                            severity="blocker",
                            category="catalog_incoherent",
                            artifact=relative,
                            json_path=f"$.packages[version={version}].{field}",
                            value=f"catalog={entry.get(field)!r};package={manifest.get(field)!r}",
                        )
                    )
    return findings


def audit_schema_artifacts(root: Path) -> PromotionAuditReport:
    """Audit every JSON/gzip-JSON artifact below ``root`` without mutating it."""
    resolved = root.expanduser().resolve()
    artifacts = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz"))
    )
    findings: list[PromotionAuditFinding] = []
    for path in artifacts:
        relative = str(path.relative_to(resolved))
        try:
            payload = _load_artifact(path)
        except Exception as error:
            findings.append(
                PromotionAuditFinding(
                    severity="blocker",
                    category="malformed_artifact",
                    artifact=relative,
                    json_path="$",
                    value=f"{type(error).__name__}: {error}",
                )
            )
            continue
        if path.name.endswith(".schema.json.gz"):
            try:
                jsonschema.Draft202012Validator.check_schema(payload)
            except Exception as error:
                findings.append(
                    PromotionAuditFinding(
                        severity="blocker",
                        category="invalid_json_schema",
                        artifact=relative,
                        json_path="$",
                        value=f"{type(error).__name__}: {error}",
                    )
                )
        _walk_artifact(payload, artifact=relative, json_path="$", findings=findings)
    findings.extend(_catalog_coherence_findings(resolved))
    findings.extend(_privacy_guard_findings(resolved))
    return PromotionAuditReport(
        root=str(resolved),
        artifact_count=len(artifacts),
        findings=tuple(sorted(set(findings))),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit staged provider schemas before promotion")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_schema_artifacts(args.root)
    rendered = json.dumps(report.to_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if report.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PromotionAuditFinding", "PromotionAuditReport", "audit_schema_artifacts", "main"]
